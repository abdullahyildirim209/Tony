"""Validate HuggingFace data availability for all 15 target assets.

Downloads the full CSV.GZ once via DuckDB, splits into per-symbol parquet caches,
then aggregates to daily bars. Reports row counts for each symbol.

Run from tony/: .venv/bin/python scripts/validate_multi_asset_data.py
"""

import os
import sys

import duckdb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fetch_hf import aggregate_to_daily

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT",
    "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "ATOMUSDT",
]

CACHE_DIR = "data/hf_cache"
MIN_ROWS = 1000


def bulk_download_and_cache():
    """Download full HF dataset once, split into per-symbol 1-min parquet caches."""
    # Check which symbols still need caching
    needed = []
    for sym in SYMBOLS:
        cache_path = os.path.join(CACHE_DIR, f"{sym}_1min.parquet")
        if not os.path.exists(cache_path):
            needed.append(sym)

    if not needed:
        print("All symbols already cached.")
        return

    print(f"Need to download data for {len(needed)} symbols: {needed}")
    print("Downloading full HF dataset via DuckDB (13GB compressed, may take 10-20 min)...")

    os.makedirs(CACHE_DIR, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # Read full dataset once, filter to our 15 symbols, partition into parquet files
    symbols_str = ", ".join(f"'{s}'" for s in needed)
    query = f"""
        SELECT
            symbol,
            bucket_ts AS timestamp,
            open,
            high,
            low,
            close,
            volume,
            taker_buy_volume
        FROM read_csv_auto('hf://datasets/123olp/binance-futures-ohlcv-2018-2026/candles_1m.csv.gz')
        WHERE symbol IN ({symbols_str})
        ORDER BY symbol, bucket_ts
    """

    print("  Running query...")
    df = con.execute(query).fetchdf()
    con.close()

    print(f"  Total rows fetched: {len(df):,}")

    # Split by symbol and cache each
    for sym in needed:
        sym_df = df[df["symbol"] == sym].drop(columns=["symbol"]).reset_index(drop=True)
        if len(sym_df) == 0:
            print(f"  WARNING: No data for {sym}")
            continue

        # Ensure types
        for col in ["open", "high", "low", "close", "volume"]:
            sym_df[col] = pd.to_numeric(sym_df[col], errors="coerce")
        sym_df["taker_buy_volume"] = pd.to_numeric(sym_df["taker_buy_volume"], errors="coerce")
        sym_df["timestamp"] = pd.to_datetime(sym_df["timestamp"], utc=True)
        sym_df = sym_df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_volume"]].dropna()

        cache_path = os.path.join(CACHE_DIR, f"{sym}_1min.parquet")
        sym_df.to_parquet(cache_path, index=False)
        print(f"  Cached {sym}: {len(sym_df):,} 1-min rows -> {cache_path}")


def build_daily_caches():
    """Aggregate 1-min parquets to daily and cache."""
    for sym in SYMBOLS:
        daily_cache = os.path.join(CACHE_DIR, f"{sym}_daily.parquet")
        if os.path.exists(daily_cache):
            continue

        min_cache = os.path.join(CACHE_DIR, f"{sym}_1min.parquet")
        if not os.path.exists(min_cache):
            continue

        df_1min = pd.read_parquet(min_cache)
        daily = aggregate_to_daily(df_1min)

        os.makedirs(CACHE_DIR, exist_ok=True)
        daily.to_parquet(daily_cache)
        print(f"  Cached daily bars for {sym}: {len(daily)} rows -> {daily_cache}")


def report():
    """Report row counts for all symbols."""
    results = []
    for symbol in SYMBOLS:
        daily_cache = os.path.join(CACHE_DIR, f"{symbol}_daily.parquet")
        if os.path.exists(daily_cache):
            daily = pd.read_parquet(daily_cache)
            if not isinstance(daily.index, pd.DatetimeIndex):
                if "date" in daily.columns:
                    daily = daily.set_index("date")
                daily.index = pd.to_datetime(daily.index)
            n_rows = len(daily)
            start = daily.index[0].date()
            end = daily.index[-1].date()
            status = "OK" if n_rows >= MIN_ROWS else "LOW"
            results.append((symbol, n_rows, start, end, status))
        else:
            results.append((symbol, 0, None, None, "MISSING"))

    print(f"\n{'=' * 70}")
    print(f"  {'Symbol':<12} {'Rows':>8} {'Start':>12} {'End':>12} {'Status':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
    for symbol, n_rows, start, end, status in results:
        print(f"  {symbol:<12} {n_rows:>8} {str(start or ''):>12} {str(end or ''):>12} {status:>10}")
    print(f"{'=' * 70}")

    ok = sum(1 for *_, s in results if s == "OK")
    print(f"\n  {ok}/{len(SYMBOLS)} symbols have {MIN_ROWS}+ daily rows.")

    missing = [sym for sym, _, _, _, s in results if s == "MISSING"]
    if missing:
        print(f"\n  Missing: {missing}")


if __name__ == "__main__":
    bulk_download_and_cache()
    build_daily_caches()
    report()
