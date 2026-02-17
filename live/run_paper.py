"""CLI entry point for paper trading.

Usage:
    python live/run_paper.py --mode replay    # test pipeline against historical data
    python live/run_paper.py --mode live      # real-time paper trading
    python live/run_paper.py --mode live --resume live/state/latest.json
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.data_feed import BinanceLiveFeed, HistoricalReplayFeed
from live.paper_trader import PaperTrader


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_replay_feed(config: dict) -> tuple[HistoricalReplayFeed, list[dict]]:
    """Build a historical replay feed from test data + warmup bars from train/val."""
    data_dir = config["data"]["save_dir"]
    live_cfg = config.get("live", {})
    warmup_days = live_cfg.get("warmup_days", 60)

    # Load test data for replay
    test_data = np.load(os.path.join(data_dir, "test.npz"), allow_pickle=True)
    test_dates = test_data["dates"]
    test_closes = test_data["close_prices"]

    # We need OHLCV bars — for replay we reconstruct from close prices
    # (the env only uses close, sma, rsi, fng anyway)
    # For a proper replay we'd want the original daily OHLCV, but close is sufficient
    # since indicators are pre-computed in the npz files.

    # Load train data for warmup bars
    train_data = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=True)

    # Build warmup bars from end of train data
    warmup_bars = []
    train_closes = train_data["close_prices"]
    train_dates = train_data["dates"]
    train_fng = train_data["fng_norm"] * 100  # denormalize

    start = max(0, len(train_closes) - warmup_days)
    for i in range(start, len(train_closes)):
        warmup_bars.append({
            "date": str(train_dates[i]),
            "open": float(train_closes[i]),
            "high": float(train_closes[i]),
            "low": float(train_closes[i]),
            "close": float(train_closes[i]),
            "volume": 0.0,
            "fng": float(train_fng[i]),
        })

    # Build test bars for replay feed
    test_fng = test_data["fng_norm"] * 100
    test_bars_df = pd.DataFrame({
        "open": test_closes,
        "high": test_closes,
        "low": test_closes,
        "close": test_closes,
        "volume": np.zeros(len(test_closes)),
    }, index=pd.to_datetime(test_dates))

    fng_series = pd.Series(test_fng, index=pd.to_datetime(test_dates))

    feed = HistoricalReplayFeed(test_bars_df, fng_series)
    return feed, warmup_bars


def main():
    parser = argparse.ArgumentParser(description="Tony Paper Trader")
    parser.add_argument("--mode", choices=["replay", "live"], default="replay",
                        help="Trading mode: replay (historical) or live (Binance)")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--resume", default=None,
                        help="Path to saved state JSON for crash recovery")
    parser.add_argument("--model", default=None,
                        help="Override model path")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    live_cfg = config.get("live", {})

    model_path = args.model or os.path.join(
        train_cfg["model_save_dir"], train_cfg["best_model_name"]
    )
    train_npz_path = os.path.join(config["data"]["save_dir"], "train.npz")
    log_dir = live_cfg.get("log_dir", "live/logs")

    if args.mode == "replay":
        print("=== Paper Trading: Historical Replay ===")
        feed, warmup_bars = build_replay_feed(config)

        trader = PaperTrader(
            model_path=model_path,
            train_npz_path=train_npz_path,
            config=config,
            feed=feed,
            log_dir=log_dir,
        )

        if args.resume:
            trader.load_state(args.resume)
        else:
            trader.warmup(warmup_bars)

        metrics = trader.run_replay()

    elif args.mode == "live":
        print("=== Paper Trading: Live Mode ===")
        symbol = config["data"].get("binance_symbol", "BTCUSDT")
        feed = BinanceLiveFeed(symbol=symbol)

        trader = PaperTrader(
            model_path=model_path,
            train_npz_path=train_npz_path,
            config=config,
            feed=feed,
            log_dir=log_dir,
        )

        if args.resume:
            trader.load_state(args.resume)
        else:
            # Warmup from Binance live historical bars (real OHLCV, not stale train data)
            warmup_days = live_cfg.get("warmup_days", 60)
            print(f"  Fetching {warmup_days} warmup bars from Binance...")
            warmup_bars = feed.fetch_historical_bars(limit=warmup_days)
            if len(warmup_bars) < warmup_days:
                print(f"  WARNING: Only got {len(warmup_bars)}/{warmup_days} warmup bars from Binance")
            trader.warmup(warmup_bars)

        check_interval = live_cfg.get("check_interval_hours", 1)
        trader.run_live(check_interval_hours=check_interval)


if __name__ == "__main__":
    main()
