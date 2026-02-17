"""HuggingFace data loader: query Binance OHLCV via DuckDB, aggregate to daily bars."""

import os

import numpy as np
import pandas as pd


def load_hf_ohlcv(symbol: str = "BTCUSDT", cache_dir: str = "data/hf_cache") -> pd.DataFrame:
    """Load 1-min candles from HF dataset via DuckDB, filtered by symbol. Caches to parquet."""
    import duckdb

    cache_path = os.path.join(cache_dir, f"{symbol}_1min.parquet")
    if os.path.exists(cache_path):
        print(f"  Loading cached 1-min data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"  Querying HF dataset for {symbol} via DuckDB (first run may take a few minutes)...")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    df = con.execute("""
        SELECT
            bucket_ts AS timestamp,
            open,
            high,
            low,
            close,
            volume,
            taker_buy_volume
        FROM read_csv_auto('hf://datasets/123olp/binance-futures-ohlcv-2018-2026/candles_1m.csv.gz')
        WHERE symbol = ?
        ORDER BY bucket_ts
    """, [symbol]).fetchdf()
    con.close()

    if len(df) == 0:
        raise ValueError(f"No data found for symbol '{symbol}' in HF dataset")

    print(f"  Total rows for {symbol}: {len(df):,}")

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["taker_buy_volume"] = pd.to_numeric(df["taker_buy_volume"], errors="coerce")
    # Neutral fallback if taker_buy_volume is all null
    if df["taker_buy_volume"].isna().all():
        df["taker_buy_volume"] = df["volume"] * 0.5

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
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

    # Filter corrupt 1-min rows where taker_buy_volume > volume
    if "taker_buy_volume" in df.columns:
        mask = df["taker_buy_volume"] <= df["volume"]
        n_bad = (~mask).sum()
        if n_bad > 0:
            print(f"  Filtered {n_bad} corrupt rows (taker_buy_volume > volume)")
            df = df[mask]

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
