"""HuggingFace data loader: download Binance OHLCV, aggregate to daily bars."""

import os

import numpy as np
import pandas as pd


def load_hf_ohlcv(symbol: str = "BTCUSDT", cache_dir: str = "data/hf_cache") -> pd.DataFrame:
    """Load 1-min candles from HF dataset, filtered by symbol. Caches to parquet."""
    from datasets import load_dataset

    cache_path = os.path.join(cache_dir, f"{symbol}_1min.parquet")
    if os.path.exists(cache_path):
        print(f"  Loading cached 1-min data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"  Streaming HF dataset for {symbol} (this may take a while)...")
    ds = load_dataset(
        "123olp/binance-futures-ohlcv-2018-2026",
        streaming=True,
        split="train",
    )

    rows = []
    count = 0
    for row in ds:
        if row.get("symbol") == symbol:
            rows.append(row)
            count += 1
            if count % 500_000 == 0:
                print(f"    ...loaded {count:,} rows")

    if not rows:
        raise ValueError(f"No data found for symbol '{symbol}' in HF dataset")

    df = pd.DataFrame(rows)
    print(f"  Total rows for {symbol}: {len(df):,}")

    # Standardize column names (dataset uses: open_time, open, high, low, close, volume, ...)
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("open_time", "timestamp", "time", "date", "bucket_ts"):
            col_map[col] = "timestamp"
        elif lower == "open":
            col_map[col] = "open"
        elif lower == "high":
            col_map[col] = "high"
        elif lower == "low":
            col_map[col] = "low"
        elif lower == "close":
            col_map[col] = "close"
        elif lower == "volume":
            col_map[col] = "volume"
        elif lower in ("taker_buy_volume", "taker_buy_base_asset_volume"):
            col_map[col] = "taker_buy_volume"
    df = df.rename(columns=col_map)

    # Parse timestamp
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.notna().any() and ts.median() > 1e12:
            df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
        elif ts.notna().any():
            df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
        else:
            # String timestamps (e.g. '2020-01-01 00:00:00+00')
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep only OHLCV + taker_buy_volume + timestamp
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "taker_buy_volume" in df.columns:
        df["taker_buy_volume"] = pd.to_numeric(df["taker_buy_volume"], errors="coerce")
    else:
        df["taker_buy_volume"] = df["volume"] * 0.5  # neutral fallback
    df = df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_volume"]].dropna()

    # Cache
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  Cached 1-min data -> {cache_path}")

    return df


def aggregate_to_daily(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min candles to daily bars (UTC day boundaries).

    Returns DataFrame with columns [open, high, low, close, volume, taker_buy_volume], DatetimeIndex named 'date'.
    """
    df = df_1min.copy()
    df = df.set_index("timestamp")

    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "taker_buy_volume" in df.columns:
        agg_dict["taker_buy_volume"] = "sum"
    daily = df.resample("1D").agg(agg_dict).dropna()

    daily.index = daily.index.tz_localize(None)
    daily.index.name = "date"

    return daily


def fetch_ohlcv_hf(
    symbol: str = "BTCUSDT",
    start_date: str = "2018-02-01",
    end_date: str = "2026-01-01",
    cache_dir: str = "data/hf_cache",
) -> pd.DataFrame:
    """Top-level function. Returns daily OHLCV DataFrame matching fetch_ohlcv() schema.

    Columns: [open, high, low, close, volume], DatetimeIndex named 'date'.
    """
    # Check for cached daily bars first
    daily_cache = os.path.join(cache_dir, f"{symbol}_daily.parquet")
    if os.path.exists(daily_cache):
        print(f"  Loading cached daily bars from {daily_cache}")
        daily = pd.read_parquet(daily_cache)
        if not isinstance(daily.index, pd.DatetimeIndex):
            if "date" in daily.columns:
                daily = daily.set_index("date")
            daily.index = pd.to_datetime(daily.index)
        daily.index.name = "date"
    else:
        df_1min = load_hf_ohlcv(symbol=symbol, cache_dir=cache_dir)
        daily = aggregate_to_daily(df_1min)

        # Cache daily bars
        os.makedirs(cache_dir, exist_ok=True)
        daily.to_parquet(daily_cache)
        print(f"  Cached daily bars -> {daily_cache}")

    # Filter date range
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    daily = daily.loc[start:end]

    print(f"  HF daily bars: {len(daily)} rows ({daily.index[0].date()} to {daily.index[-1].date()})")
    return daily


if __name__ == "__main__":
    df = fetch_ohlcv_hf()
    print(f"\nShape: {df.shape}")
    print(df.head())
    print(df.tail())
