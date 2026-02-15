"""Daily entry point for Binance testnet paper trading.

Usage:
    # Single cycle (for cron):
    python live/run_daily.py --once

    # Self-scheduling loop (sleeps until 00:05 UTC):
    python live/run_daily.py --loop

    # With testnet execution (set env vars first):
    export BINANCE_TESTNET_API_KEY="..."
    export BINANCE_TESTNET_API_SECRET="..."
    python live/run_daily.py --once

Cron example:
    5 0 * * * cd /path/to/tony && .venv/bin/python live/run_daily.py --once >> live/logs/cron.log 2>&1
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.data_feed import BinanceLiveFeed
from live.paper_trader import PaperTrader


DEFAULT_MODEL = "models/walk_forward/fold_5/best_model"
HEARTBEAT_FILE = "live/state/heartbeat.json"


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def check_heartbeat(state_dir: str) -> None:
    """Warn if last run was more than 36 hours ago."""
    hb_path = os.path.join(state_dir, "heartbeat.json")
    if not os.path.exists(hb_path):
        return
    try:
        with open(hb_path) as f:
            hb = json.load(f)
        last_run = datetime.datetime.fromisoformat(hb["last_run"])
        now = datetime.datetime.now(datetime.timezone.utc)
        gap = now - last_run
        if gap > datetime.timedelta(hours=36):
            print(f"  WARNING: Last run was {gap} ago ({hb['last_run']}). "
                  f"Possible missed days.")
    except (json.JSONDecodeError, KeyError, ValueError):
        pass


def update_heartbeat(state_dir: str) -> None:
    """Update heartbeat timestamp."""
    os.makedirs(state_dir, exist_ok=True)
    hb_path = os.path.join(state_dir, "heartbeat.json")
    with open(hb_path, "w") as f:
        json.dump({
            "last_run": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, f)


def get_last_bar_date(state_path: str) -> str | None:
    """Extract the last processed bar date from saved state."""
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path) as f:
            state = json.load(f)
        bars = state.get("feature_engine", {}).get("bars", [])
        if bars:
            return bars[-1].get("date")
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def run_once(config: dict, model_path: str) -> None:
    """Execute one daily trading cycle."""
    live_cfg = config.get("live", {})
    state_dir = live_cfg.get("state_dir", "live/state")
    log_dir = live_cfg.get("log_dir", "live/logs")
    warmup_days = live_cfg.get("warmup_days", 60)
    state_path = os.path.join(state_dir, "latest.json")
    train_npz_path = os.path.join(config["data"]["save_dir"], "train.npz")
    symbol = config["data"].get("binance_symbol", "BTCUSDT")

    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n=== Daily Trading Cycle: {now_utc} ===")

    # Heartbeat check
    check_heartbeat(state_dir)

    # Create feed
    feed = BinanceLiveFeed(symbol=symbol)

    # Optional testnet executor
    testnet_executor = None
    testnet_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
    testnet_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
    if testnet_key and testnet_secret:
        from live.testnet_executor import TestnetExecutor
        testnet_executor = TestnetExecutor(api_key=testnet_key, api_secret=testnet_secret)
        print(f"  Testnet executor enabled")
        usdt_bal = testnet_executor.get_balance("USDT")
        btc_bal = testnet_executor.get_balance("BTC")
        if usdt_bal is not None:
            print(f"  Testnet balances: {usdt_bal:.2f} USDT, {btc_bal} BTC")
    else:
        print("  Testnet executor disabled (no env vars set)")

    # Create trader
    trader = PaperTrader(
        model_path=model_path,
        train_npz_path=train_npz_path,
        config=config,
        feed=feed,
        log_dir=log_dir,
        testnet_executor=testnet_executor,
    )

    # Resume or warmup
    if os.path.exists(state_path):
        trader.load_state(state_path)
        last_date = get_last_bar_date(state_path)
        print(f"  Resumed from state (last bar: {last_date})")
    else:
        print(f"  No saved state — warming up from {warmup_days} Binance bars...")
        warmup_bars = feed.fetch_historical_bars(limit=warmup_days)
        if not warmup_bars:
            print("  ERROR: Failed to fetch warmup bars from Binance. Aborting.")
            sys.exit(1)
        trader.warmup(warmup_bars)

    # Duplicate date check
    last_processed = get_last_bar_date(state_path) if os.path.exists(state_path) else None

    # Step
    result = trader.step()

    if result is None:
        print("  No new bar available (already processed today or insufficient data)")
    else:
        # Check for duplicate
        if last_processed and result["date"] == last_processed:
            print(f"  WARNING: Duplicate bar date {result['date']}, skipping save")
        else:
            print(f"  [{result['date']}] {result['action_name']:>4} | "
                  f"Price: ${result['price']:,.2f} | "
                  f"Portfolio: ${result['portfolio_value']:,.2f} | "
                  f"Trade: {result['trade_executed']} | "
                  f"Return: {result.get('cumulative_return', 0):.2%}")

            # Save state
            trader.save_state(state_path)

    # Update heartbeat
    update_heartbeat(state_dir)
    print("  Done.\n")


def run_loop(config: dict, model_path: str) -> None:
    """Self-scheduling loop: run at 00:05 UTC daily."""
    print("=== Daily Trading Loop (00:05 UTC) ===")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            run_once(config, model_path)

            # Sleep until next 00:05 UTC
            now = datetime.datetime.now(datetime.timezone.utc)
            target = now.replace(hour=0, minute=5, second=0, microsecond=0)
            if target <= now:
                target += datetime.timedelta(days=1)
            sleep_secs = (target - now).total_seconds()
            print(f"  Sleeping {sleep_secs / 3600:.1f}h until {target.strftime('%Y-%m-%d %H:%M UTC')}")
            time.sleep(sleep_secs)

    except KeyboardInterrupt:
        print("\nLoop stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Tony Daily Paper Trader")
    parser.add_argument("--once", action="store_true",
                        help="Run one trading cycle and exit (for cron)")
    parser.add_argument("--loop", action="store_true",
                        help="Self-scheduling loop, sleeps until 00:05 UTC")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--model", default=None,
                        help="Override model path (default: fold_5 walk-forward)")
    args = parser.parse_args()

    if not args.once and not args.loop:
        parser.error("Specify --once or --loop")

    config = load_config(args.config)
    model_path = args.model or DEFAULT_MODEL

    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    if args.once:
        run_once(config, model_path)
    elif args.loop:
        run_loop(config, model_path)


if __name__ == "__main__":
    main()
