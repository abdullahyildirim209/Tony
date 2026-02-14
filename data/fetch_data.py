"""Data pipeline: download OHLCV, compute indicators, split & save."""

import os
import sys

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing prices, drop leading NaN rows."""
    df = df.ffill()
    df = df.dropna()
    return df


def add_indicators(
    df: pd.DataFrame, sma_short: int, sma_long: int, rsi_period: int
) -> pd.DataFrame:
    """Compute SMA and RSI using pure pandas (Wilder's EWM for RSI)."""
    df = df.copy()
    df["sma_short"] = df["close"].rolling(window=sma_short).mean()
    df["sma_long"] = df["close"].rolling(window=sma_long).mean()

    # RSI via Wilder's smoothing (EWM with alpha=1/period)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    return df


def compute_features(df: pd.DataFrame, window_size: int) -> dict:
    """Produce feature arrays from indicator DataFrame (no clipping).

    Returns dict with keys: close_prices, pct_changes, sma_ratios, rsi_norm, dates.
    Clipping is deferred to after train/val/test split so stats come from train only.
    """
    df = df.copy()

    # Pct change of close (close / prev_close - 1)
    df["pct_change"] = df["close"].pct_change()

    # SMA ratio
    df["sma_ratio"] = df["sma_short"] / df["sma_long"]

    # Normalized RSI
    df["rsi_norm"] = df["rsi"] / 100.0

    # Drop rows with NaN (from indicators + pct_change)
    df = df.dropna()

    # Drop first window_size rows to ensure enough history
    if len(df) > window_size:
        df = df.iloc[window_size:]

    return {
        "close_prices": df["close"].values.astype(np.float32),
        "pct_changes": df["pct_change"].values.astype(np.float32),
        "sma_ratios": df["sma_ratio"].values.astype(np.float32),
        "rsi_norm": df["rsi_norm"].values.astype(np.float32),
        "dates": df.index.strftime("%Y-%m-%d").values,
    }


def compute_clip_stats(split: dict) -> dict:
    """Compute mean/std for clipping from a single split (train).

    Returns dict mapping feature name to (mean, std).
    """
    stats = {}
    for col in ["pct_changes", "sma_ratios", "rsi_norm"]:
        mean = float(np.mean(split[col]))
        std = float(np.std(split[col]))
        stats[col] = (mean, std)
    return stats


def apply_clip(split: dict, clip_stats: dict) -> dict:
    """Clip features to ±5 std devs using pre-computed stats."""
    split = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in split.items()}
    for col in ["pct_changes", "sma_ratios", "rsi_norm"]:
        mean, std = clip_stats[col]
        if std > 0:
            split[col] = np.clip(split[col], mean - 5 * std, mean + 5 * std)
    return split


def split_data(
    features: dict, dates: np.ndarray, train_end: str, val_end: str
) -> tuple[dict, dict, dict]:
    """Chronological split by date boundaries."""
    train_mask = dates < train_end
    val_mask = (dates >= train_end) & (dates < val_end)
    test_mask = dates >= val_end

    def select(mask):
        return {
            "close_prices": features["close_prices"][mask],
            "pct_changes": features["pct_changes"][mask],
            "sma_ratios": features["sma_ratios"][mask],
            "rsi_norm": features["rsi_norm"][mask],
            "dates": features["dates"][mask],
        }

    return select(train_mask), select(val_mask), select(test_mask)


def main(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    data_cfg = config["data"]
    ind_cfg = config["indicators"]
    env_cfg = config["env"]

    print(f"Fetching {data_cfg['asset']} from {data_cfg['start_date']} to {data_cfg['end_date']}...")
    df = fetch_ohlcv(data_cfg["asset"], data_cfg["start_date"], data_cfg["end_date"])
    print(f"  Raw data: {len(df)} rows")

    df = clean_data(df)
    print(f"  After cleaning: {len(df)} rows")

    df = add_indicators(df, ind_cfg["sma_short"], ind_cfg["sma_long"], ind_cfg["rsi_period"])
    print(f"  After indicators: {len(df)} rows")

    features = compute_features(df, env_cfg["window_size"])
    print(f"  After features: {len(features['close_prices'])} rows")

    # Split first, THEN clip using train-only stats (prevents data leakage)
    train, val, test = split_data(
        features, features["dates"], data_cfg["train_end"], data_cfg["val_end"]
    )

    clip_stats = compute_clip_stats(train)
    print(f"  Clip stats (from train): { {k: (f'{m:.6f}', f'{s:.6f}') for k, (m, s) in clip_stats.items()} }")

    train = apply_clip(train, clip_stats)
    val = apply_clip(val, clip_stats)
    test = apply_clip(test, clip_stats)

    save_dir = data_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Save clip stats in train.npz for reproducibility
    train_extra = dict(train)
    for col, (mean, std) in clip_stats.items():
        train_extra[f"clip_{col}_mean"] = np.float32(mean)
        train_extra[f"clip_{col}_std"] = np.float32(std)

    for name, split in [("train", train_extra), ("val", val), ("test", test)]:
        path = os.path.join(save_dir, f"{name}.npz")
        np.savez(path, **split)
        print(f"  Saved {name}: {len(split['close_prices'])} rows -> {path}")

    print("Data pipeline complete.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config_path)
