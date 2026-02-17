"""CLI entry point for multi-asset paper trading.

Usage:
    python live/run_multi.py --mode replay   # test all assets against historical data
    python live/run_multi.py --mode live     # real-time multi-asset paper trading (loop)
    python live/run_multi.py --mode once     # single daily cycle (for cron/daily.sh)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.multi_asset_trader import MultiAssetTrader


def main():
    parser = argparse.ArgumentParser(description="Tony Multi-Asset Paper Trader")
    parser.add_argument("--mode", choices=["replay", "live", "once"], default="replay",
                        help="Trading mode: replay (historical), live (loop), once (single cycle)")
    parser.add_argument("--config", default="configs/paper_trading.yaml",
                        help="Path to multi-asset paper trading config")
    parser.add_argument("--default-config", default="configs/default.yaml",
                        help="Path to default config (env/agent settings)")
    args = parser.parse_args()

    trader = MultiAssetTrader(
        paper_trading_config_path=args.config,
        default_config_path=args.default_config,
    )

    if args.mode == "replay":
        trader.run_replay()
    elif args.mode == "live":
        trader.initialize_traders(mode="live")
        trader.run_live()
    elif args.mode == "once":
        trader.initialize_traders(mode="live")
        trader.run_once()


if __name__ == "__main__":
    main()
