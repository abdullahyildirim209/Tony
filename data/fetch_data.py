"""Data pipeline: download OHLCV, compute indicators, split & save."""

import os
import sys

import numpy as np
import pandas as pd
import requests
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
    df["taker_buy_volume"] = df["volume"] * 0.5  # neutral fallback (yfinance lacks this)
    df.index.name = "date"
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing prices, drop leading NaN rows."""
    df = df.ffill()
    df = df.dropna()
    return df


def fetch_fng(start_date: str, end_date: str, cache_path: str = "data/fng_cache.csv") -> pd.Series:
    """Fetch Crypto Fear & Greed Index from alternative.me API.

    Returns a pd.Series indexed by date with integer values 0-100.
    Caches to CSV to avoid repeated API calls.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Try loading from cache
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached_series = cached["fng"].squeeze()
        if cached_series.index.min() <= start and cached_series.index.max() >= end:
            return cached_series.loc[start:end]

    # Fetch from API
    print("  Fetching Fear & Greed Index from alternative.me...")
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": "0", "format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]

        dates = []
        values = []
        for entry in data:
            dates.append(pd.Timestamp.fromtimestamp(int(entry["timestamp"])).normalize())
            values.append(int(entry["value"]))

        fng = pd.Series(values, index=dates, name="fng").sort_index()
        fng = fng[~fng.index.duplicated(keep="first")]

        # Cache
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        fng.to_frame().to_csv(cache_path)
        print(f"  Cached FNG data ({len(fng)} days) -> {cache_path}")

        return fng.loc[start:end]

    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"  WARNING: FNG API failed ({e}). Using neutral value (50).")
        idx = pd.bdate_range(start, end)
        return pd.Series(50, index=idx, name="fng")


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


def compute_turbulence(df: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """Compute turbulence as z-score of rolling volatility.

    Uses a 63-day (quarterly) rolling window for volatility estimation,
    then z-scores against the expanding mean/std of that rolling vol.
    """
    daily_returns = df["close"].pct_change()
    rolling_vol = daily_returns.rolling(window=lookback).std()
    expanding_mean = rolling_vol.expanding().mean()
    expanding_std = rolling_vol.expanding().std()
    turbulence = (rolling_vol - expanding_mean) / expanding_std.replace(0, np.nan)
    return turbulence.fillna(0.0)


def compute_features(df: pd.DataFrame, window_size: int) -> dict:
    """Produce feature arrays from indicator DataFrame (no clipping).

    Returns dict with keys: close_prices, pct_changes, sma_ratios, rsi_norm, fng_norm,
    buy_pressure, turbulence, dates.
    Clipping is deferred to after train/val/test split so stats come from train only.
    """
    df = df.copy()

    # Pct change of close (close / prev_close - 1)
    df["pct_change"] = df["close"].pct_change()

    # SMA ratio
    df["sma_ratio"] = df["sma_short"] / df["sma_long"]

    # Normalized RSI
    df["rsi_norm"] = df["rsi"] / 100.0

    # Normalized FNG (0-100 -> 0-1)
    df["fng_norm"] = df["fng"] / 100.0

    # Buy pressure (taker_buy_volume / volume), 0-1, 0.5 = neutral
    if "taker_buy_volume" in df.columns and "volume" in df.columns:
        df["buy_pressure"] = (df["taker_buy_volume"] / df["volume"]).clip(0, 1).fillna(0.5)
    else:
        df["buy_pressure"] = 0.5

    # Turbulence: z-score of rolling volatility
    df["turbulence"] = compute_turbulence(df)

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
        "fng_norm": df["fng_norm"].values.astype(np.float32),
        "buy_pressure": df["buy_pressure"].values.astype(np.float32),
        "turbulence": df["turbulence"].values.astype(np.float32),
        "dates": df.index.strftime("%Y-%m-%d").values,
    }


CLIPPED_FEATURES = ["pct_changes", "sma_ratios", "rsi_norm", "fng_norm", "buy_pressure", "turbulence"]


def compute_clip_stats(split: dict) -> dict:
    """Compute mean/std for clipping from a single split (train).

    Returns dict mapping feature name to (mean, std).
    """
    stats = {}
    for col in CLIPPED_FEATURES:
        if col in split:
            mean = float(np.mean(split[col]))
            std = float(np.std(split[col]))
            stats[col] = (mean, std)
    return stats


def apply_clip(split: dict, clip_stats: dict) -> dict:
    """Clip features to ±5 std devs using pre-computed stats."""
    split = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in split.items()}
    for col in CLIPPED_FEATURES:
        if col in clip_stats and col in split:
            mean, std = clip_stats[col]
            if std > 0:
                split[col] = np.clip(split[col], mean - 5 * std, mean + 5 * std)
    return split


FEATURE_KEYS = ["close_prices", "pct_changes", "sma_ratios", "rsi_norm", "fng_norm", "buy_pressure", "turbulence", "dates"]


def split_data(
    features: dict, dates: np.ndarray, train_end: str, val_end: str
) -> tuple[dict, dict, dict]:
    """Chronological split by date boundaries."""
    train_mask = dates < train_end
    val_mask = (dates >= train_end) & (dates < val_end)
    test_mask = dates >= val_end

    def select(mask):
        return {k: features[k][mask] for k in FEATURE_KEYS if k in features}

    return select(train_mask), select(val_mask), select(test_mask)


def main(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    data_cfg = config["data"]
    ind_cfg = config["indicators"]
    env_cfg = config["env"]

    print(f"Fetching {data_cfg['asset']} from {data_cfg['start_date']} to {data_cfg['end_date']}...")
    if data_cfg.get("source") == "huggingface":
        from data.fetch_hf import fetch_ohlcv_hf
        hf_symbol = data_cfg.get("binance_symbol", "BTCUSDT")
        hf_cache = data_cfg.get("hf_cache_dir", "data/hf_cache")
        df = fetch_ohlcv_hf(hf_symbol, data_cfg["start_date"], data_cfg["end_date"], hf_cache)
    else:
        df = fetch_ohlcv(data_cfg["asset"], data_cfg["start_date"], data_cfg["end_date"])
    print(f"  Raw data: {len(df)} rows")

    df = clean_data(df)
    print(f"  After cleaning: {len(df)} rows")

    # Fetch Fear & Greed Index
    fng_cache = data_cfg.get("fng_cache_path", "data/fng_cache.csv")
    fng_series = fetch_fng(data_cfg["start_date"], data_cfg["end_date"], cache_path=fng_cache)
    df["fng"] = fng_series.reindex(df.index, method="ffill")
    df["fng"] = df["fng"].bfill()  # fill any leading NaNs
    print(f"  FNG data: {fng_series.notna().sum()} days matched")

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

    # Compute turbulence threshold (90th percentile of training data)
    turb_threshold_pct = config.get("env", {}).get("turbulence_threshold_pct", 90)
    turb_threshold = float(np.percentile(train["turbulence"], turb_threshold_pct))
    print(f"  Turbulence threshold ({turb_threshold_pct}th pct of train): {turb_threshold:.4f}")

    # Save clip stats in train.npz for reproducibility
    train_extra = dict(train)
    for col, (mean, std) in clip_stats.items():
        train_extra[f"clip_{col}_mean"] = np.float32(mean)
        train_extra[f"clip_{col}_std"] = np.float32(std)
    train_extra["turbulence_threshold"] = np.float32(turb_threshold)

    for name, split in [("train", train_extra), ("val", val), ("test", test)]:
        path = os.path.join(save_dir, f"{name}.npz")
        np.savez(path, **split)
        print(f"  Saved {name}: {len(split['close_prices'])} rows -> {path}")

    print("Data pipeline complete.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config_path)
