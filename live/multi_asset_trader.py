"""Multi-asset paper trading orchestrator.

Runs independent DQN models for multiple crypto assets in parallel,
with aggregated logging and risk management.

Usage:
    python live/run_multi.py --mode replay
    python live/run_multi.py --mode live
"""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import yaml

from live.data_feed import BinanceLiveFeed, HistoricalReplayFeed
from live.paper_trader import PaperTrader


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class MultiAssetTrader:
    """Manages multiple PaperTrader instances, one per asset."""

    def __init__(self, paper_trading_config_path: str, default_config_path: str = "configs/default.yaml"):
        self.pt_config = load_config(paper_trading_config_path)
        self.default_config = load_config(default_config_path)

        log_cfg = self.pt_config.get("logging", {})
        self.log_dir = log_cfg.get("log_dir", "paper_trading_logs")
        self.state_dir = log_cfg.get("state_dir", "paper_trading_logs/state")

        self.traders: dict[str, PaperTrader] = {}
        self.asset_configs: dict[str, dict] = {}
        self._disabled_assets: set[str] = set()

    def _compute_allocations(self, total_capital: float) -> dict[str, float]:
        """Compute per-asset capital allocation weighted by walk-forward win rate.

        Loads fold_metrics.json for each asset, averages win_rate across folds,
        then allocates proportionally. Assets without metrics get equal share of
        a 10% floor pool; the rest is split by win rate.
        """
        alloc_cfg = self.pt_config.get("allocation", {})
        method = alloc_cfg.get("method", "equal")

        enabled_tickers = [
            a.get("yf_ticker", a["ticker"])
            for a in self.pt_config["assets"]
            if a.get("enabled", True)
        ]

        if method == "equal" or len(enabled_tickers) == 0:
            per = total_capital / max(len(enabled_tickers), 1)
            return {t: per for t in enabled_tickers}

        # Load win rates and Sortino ratios from walk-forward results
        win_rates: dict[str, float] = {}
        avg_sortinos: dict[str, float] = {}
        latest_sortinos: dict[str, float] = {}
        pct_positive_folds: dict[str, float] = {}
        for ticker in enabled_tickers:
            metrics_path = f"results/{ticker}/walk_forward/fold_metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    fold_metrics = json.load(f)
                rates = [m["win_rate"] for m in fold_metrics.values() if "win_rate" in m]
                win_rates[ticker] = np.mean(rates) if rates else 0.0
                sortinos = [m["sortino_ratio"] for m in fold_metrics.values() if "sortino_ratio" in m]
                avg_sortinos[ticker] = np.mean(sortinos) if sortinos else 0.0
                # Latest fold Sortino (deployment fold)
                fold_names = sorted(fold_metrics.keys())
                latest_fold = fold_metrics[fold_names[-1]] if fold_names else {}
                latest_sortinos[ticker] = latest_fold.get("sortino_ratio", 0.0)
                returns = [m["cumulative_return"] for m in fold_metrics.values() if "cumulative_return" in m]
                n_positive = sum(1 for r in returns if r > 0)
                pct_positive_folds[ticker] = n_positive / len(returns) if returns else 0.0
            else:
                win_rates[ticker] = 0.0
                avg_sortinos[ticker] = 0.0
                latest_sortinos[ticker] = 0.0
                pct_positive_folds[ticker] = 0.0

        # Performance filter: flag unhealthy assets
        # Unhealthy if EITHER:
        #   1. avg Sortino < -1.0 AND < 40% positive folds, OR
        #   2. latest (deployment) fold Sortino < -2.0
        unhealthy: set[str] = set()
        for ticker in enabled_tickers:
            reason = ""
            if avg_sortinos[ticker] < -1.0 and pct_positive_folds[ticker] < 0.40:
                reason = f"avg_sortino={avg_sortinos[ticker]:.2f}, positive_folds={pct_positive_folds[ticker]:.0%}"
            elif latest_sortinos[ticker] < -2.0:
                reason = f"latest_fold_sortino={latest_sortinos[ticker]:.2f}"
            if reason:
                unhealthy.add(ticker)
                print(f"  UNHEALTHY: {ticker} ({reason}) → floor allocation only")

        # Floor: every asset gets at least floor_pct of equal share
        floor_pct = alloc_cfg.get("floor_pct", 0.10)
        equal_share = total_capital / len(enabled_tickers)
        floor_amount = equal_share * floor_pct

        # Unhealthy assets get only floor allocation; remaining capital goes to healthy
        unhealthy_total = floor_amount * len(unhealthy)
        healthy_tickers = [t for t in enabled_tickers if t not in unhealthy]
        remaining = total_capital - unhealthy_total - floor_amount * len(healthy_tickers)

        # Weight healthy assets by win rate (clip negatives to 0)
        clipped = {t: max(wr, 0.0) for t, wr in win_rates.items() if t in healthy_tickers}
        total_wr = sum(clipped.values())

        allocations = {}
        # Unhealthy assets get floor only
        for t in unhealthy:
            allocations[t] = floor_amount

        if total_wr > 0:
            for t in healthy_tickers:
                allocations[t] = floor_amount + remaining * (clipped[t] / total_wr)
        else:
            # All zero win rates — fall back to equal among healthy
            healthy_equal = (total_capital - unhealthy_total) / max(len(healthy_tickers), 1)
            for t in healthy_tickers:
                allocations[t] = healthy_equal

        print("  Capital allocation (win-rate weighted + performance filter):")
        for t in enabled_tickers:
            wr_str = f"{win_rates[t]:.1%}" if t in win_rates else "N/A"
            flag = " [UNHEALTHY]" if t in unhealthy else ""
            print(f"    {t:<12} win_rate={wr_str:>6}  sortino={avg_sortinos.get(t, 0):.2f}  "
                  f"latest={latest_sortinos.get(t, 0):.2f}  capital=${allocations[t]:>10,.2f}{flag}")

        return allocations

    def initialize_traders(self, mode: str = "live") -> None:
        """Create PaperTrader instances for each enabled asset.

        Args:
            mode: "live" or "replay". Determines the feed type.
        """
        risk_cfg = self.pt_config.get("risk", {})
        total_capital = risk_cfg.get("total_capital", 10000)
        allocations = self._compute_allocations(total_capital)

        for asset_cfg in self.pt_config["assets"]:
            if not asset_cfg.get("enabled", True):
                continue

            ticker = asset_cfg["ticker"]  # Binance symbol (e.g. BTCUSDT)
            asset_id = asset_cfg.get("yf_ticker", ticker)  # asset identifier for paths/logs
            model_path = asset_cfg["model_path"]
            train_npz_path = asset_cfg["train_npz_path"]

            if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
                # Try without .zip extension
                if not os.path.exists(model_path):
                    print(f"  SKIP {asset_id}: model not found at {model_path}")
                    continue

            if not os.path.exists(train_npz_path):
                print(f"  SKIP {asset_id}: train.npz not found at {train_npz_path}")
                continue

            # Build per-asset config (override capital with allocation)
            asset_capital = allocations.get(asset_id, total_capital / 8)
            asset_config = dict(self.default_config)
            asset_config["env"] = dict(self.default_config["env"])
            asset_config["env"]["initial_cash"] = asset_capital
            asset_config["data"] = dict(self.default_config["data"])
            asset_config["data"]["binance_symbol"] = ticker
            asset_config["data"]["asset"] = asset_id

            asset_log_dir = os.path.join(self.log_dir, asset_id)

            if mode == "live":
                feed = BinanceLiveFeed(symbol=ticker)
            else:
                feed = None  # Set up in run_replay per asset

            if feed is not None:
                trader = PaperTrader(
                    model_path=model_path,
                    train_npz_path=train_npz_path,
                    config=asset_config,
                    feed=feed,
                    log_dir=asset_log_dir,
                )
                self.traders[asset_id] = trader
                self.asset_configs[asset_id] = asset_cfg

            elif mode == "replay":
                # Store config for replay setup later
                self.asset_configs[asset_id] = asset_cfg
                self.asset_configs[asset_id]["_asset_config"] = asset_config
                self.asset_configs[asset_id]["_log_dir"] = asset_log_dir

        print(f"  Initialized {len(self.traders) or len(self.asset_configs)} asset(s)")

    def warmup_live(self) -> None:
        """Fetch and feed warmup bars from Binance for all live traders."""
        warmup_days = self.default_config.get("live", {}).get("warmup_days", 60)

        for asset_id, trader in self.traders.items():
            print(f"  Warming up {asset_id}...")
            warmup_bars = trader.feed.fetch_historical_bars(limit=warmup_days)
            if len(warmup_bars) < warmup_days:
                print(f"    WARNING: Only got {len(warmup_bars)}/{warmup_days} bars")
            trader.warmup(warmup_bars)

    def daily_step(self) -> dict:
        """Execute one daily cycle for all active traders.

        Returns dict of {ticker: step_result_or_None}.
        """
        results = {}
        for asset_id, trader in self.traders.items():
            if asset_id in self._disabled_assets:
                results[asset_id] = None
                continue

            result = trader.step()
            results[asset_id] = result

            if result is not None:
                print(f"  [{asset_id}] {result['date']} | {result['action_name']:>4} | "
                      f"${result['portfolio_value']:,.2f} | Trade: {result['trade_executed']}")

                # Check per-asset max drawdown
                max_dd = self.pt_config.get("risk", {}).get("max_drawdown_per_asset", 0.50)
                metrics = trader.state_manager.get_metrics()
                if metrics["max_drawdown"] >= max_dd:
                    print(f"  WARNING: {asset_id} hit max drawdown ({metrics['max_drawdown']:.1%}), disabling")
                    self._disabled_assets.add(asset_id)

        return results

    def get_aggregate_stats(self) -> dict:
        """Combined portfolio statistics across all assets."""
        total_trades = 0
        total_value = 0.0
        total_initial = 0.0
        per_asset = {}

        for asset_id, trader in self.traders.items():
            metrics = trader.state_manager.get_metrics()
            total_trades += metrics["trade_count"]
            total_value += metrics["final_value"]
            total_initial += trader.state_manager.initial_cash
            per_asset[asset_id] = metrics

        combined_return = (total_value / total_initial - 1.0) if total_initial > 0 else 0.0

        return {
            "total_trades": total_trades,
            "total_value": total_value,
            "combined_return": combined_return,
            "active_assets": len(self.traders) - len(self._disabled_assets),
            "disabled_assets": list(self._disabled_assets),
            "per_asset": per_asset,
        }

    def run_replay(self) -> dict:
        """Run historical replay for all assets using walk-forward fold_5 test data."""
        print("\n=== Multi-Asset Paper Trading: Historical Replay ===")

        risk_cfg = self.pt_config.get("risk", {})
        total_capital = risk_cfg.get("total_capital", 10000)
        allocations = self._compute_allocations(total_capital)
        warmup_days = self.default_config.get("live", {}).get("warmup_days", 60)

        for asset_cfg in self.pt_config["assets"]:
            if not asset_cfg.get("enabled", True):
                continue

            asset_id = asset_cfg.get("yf_ticker", asset_cfg["ticker"])
            model_path = asset_cfg["model_path"]
            train_npz_path = asset_cfg["train_npz_path"]

            # For replay, we need test.npz from the same fold
            test_npz_path = train_npz_path.replace("train.npz", "test.npz")

            if not os.path.exists(train_npz_path):
                print(f"  SKIP {asset_id}: {train_npz_path} not found")
                continue
            if not os.path.exists(test_npz_path):
                print(f"  SKIP {asset_id}: {test_npz_path} not found")
                continue

            model_path_check = model_path if os.path.exists(model_path) else model_path + ".zip"
            if not os.path.exists(model_path_check) and not os.path.exists(model_path):
                print(f"  SKIP {asset_id}: model not found at {model_path}")
                continue

            # Build per-asset config with win-rate-weighted capital
            asset_capital = allocations.get(asset_id, total_capital / 8)
            asset_config = dict(self.default_config)
            asset_config["env"] = dict(self.default_config["env"])
            asset_config["env"]["initial_cash"] = asset_capital
            asset_config["data"] = dict(self.default_config["data"])
            asset_config["data"]["binance_symbol"] = asset_cfg["ticker"]
            asset_config["data"]["asset"] = asset_id

            # Build replay feed from test.npz
            feed, warmup_bars = self._build_replay_feed(
                train_npz_path, test_npz_path, warmup_days
            )

            asset_log_dir = os.path.join(self.log_dir, asset_id)
            trader = PaperTrader(
                model_path=model_path,
                train_npz_path=train_npz_path,
                config=asset_config,
                feed=feed,
                log_dir=asset_log_dir,
            )
            trader.warmup(warmup_bars)
            self.traders[asset_id] = trader

        # Run replay for each asset
        for asset_id, trader in self.traders.items():
            print(f"\n--- {asset_id} Replay ---")
            trader.run_replay()

        # Print combined summary
        stats = self.get_aggregate_stats()
        self._print_summary(stats)
        self._save_summary(stats)

        return stats

    def _load_states(self) -> int:
        """Load saved state for all traders. Returns count of restored traders."""
        restored = 0
        for asset_id, trader in self.traders.items():
            state_path = os.path.join(self.state_dir, f"{asset_id}.json")
            if os.path.exists(state_path):
                trader.load_state(state_path)
                restored += 1
        return restored

    def run_once(self) -> dict:
        """Execute one daily cycle for all assets, then exit.

        Suitable for cron: initializes, resumes state or warms up,
        steps once, saves state, prints summary.
        """
        import datetime

        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n=== Multi-Asset Daily Cycle: {now_utc} ===")

        # Resume from saved state or warmup
        restored = self._load_states()
        if restored > 0:
            print(f"  Resumed {restored}/{len(self.traders)} asset(s) from saved state")
            # Warmup any that weren't restored
            warmup_needed = [a for a in self.traders if not os.path.exists(
                os.path.join(self.state_dir, f"{a}.json"))]
            if warmup_needed:
                print(f"  Warming up {len(warmup_needed)} new asset(s)...")
                for asset_id in warmup_needed:
                    warmup_bars = self.traders[asset_id].feed.fetch_historical_bars(
                        limit=self.default_config.get("live", {}).get("warmup_days", 60))
                    self.traders[asset_id].warmup(warmup_bars)
        else:
            print("  No saved state — warming up all assets from Binance...")
            self.warmup_live()

        # Execute one step
        results = self.daily_step()
        any_new = any(r is not None for r in results.values())

        if any_new:
            stats = self.get_aggregate_stats()
            self._print_summary(stats)
            self._save_summary(stats)
            self._save_states()
            self._write_combined_log(results)

            # Check total drawdown
            max_total_dd = self.pt_config.get("risk", {}).get("max_drawdown_total", 0.30)
            if stats["combined_return"] < -max_total_dd:
                print(f"  EMERGENCY: Total drawdown {stats['combined_return']:.1%} "
                      f"exceeds limit {-max_total_dd:.1%}.")
        else:
            print("  No new bars available (already processed today)")
            stats = self.get_aggregate_stats()

        print("  Done.\n")
        return stats

    def run_live(self) -> None:
        """Live paper trading loop: check for new daily bars periodically."""
        schedule_cfg = self.pt_config.get("schedule", {})
        check_interval = schedule_cfg.get("check_interval_hours", 1.0)
        interval_secs = check_interval * 3600

        print(f"\n=== Multi-Asset Paper Trading: Live Mode ===")
        print(f"  Assets: {list(self.traders.keys())}")
        print(f"  Check interval: {check_interval}h")

        self.warmup_live()

        try:
            while True:
                results = self.daily_step()

                # Check if any new data was processed
                any_new = any(r is not None for r in results.values())
                if any_new:
                    stats = self.get_aggregate_stats()
                    print(f"  [TOTAL] Trades: {stats['total_trades']} | "
                          f"Value: ${stats['total_value']:,.2f} | "
                          f"Return: {stats['combined_return']:.2%}")
                    self._save_summary(stats)
                    self._save_states()

                    # Check total drawdown
                    max_total_dd = self.pt_config.get("risk", {}).get("max_drawdown_total", 0.30)
                    if stats["combined_return"] < -max_total_dd:
                        print(f"  EMERGENCY: Total drawdown {stats['combined_return']:.1%} "
                              f"exceeds limit {-max_total_dd:.1%}. Stopping.")
                        break

                time.sleep(interval_secs)

        except KeyboardInterrupt:
            print("\nLive trading stopped by user.")
            stats = self.get_aggregate_stats()
            self._print_summary(stats)
            self._save_summary(stats)

    def _build_replay_feed(
        self, train_npz_path: str, test_npz_path: str, warmup_days: int
    ) -> tuple:
        """Build replay feed and warmup bars from npz files."""
        import pandas as pd

        train_data = np.load(train_npz_path, allow_pickle=True)
        test_data = np.load(test_npz_path, allow_pickle=True)

        # Warmup bars from end of train data
        warmup_bars = []
        train_closes = train_data["close_prices"]
        train_dates = train_data["dates"]
        train_fng = train_data["fng_norm"] * 100

        start = max(0, len(train_closes) - warmup_days)
        for i in range(start, len(train_closes)):
            warmup_bars.append({
                "date": str(train_dates[i]),
                "open": float(train_closes[i]),
                "high": float(train_closes[i]),
                "low": float(train_closes[i]),
                "close": float(train_closes[i]),
                "volume": 0.0,
                "taker_buy_volume": 0.0,
                "fng": float(train_fng[i]),
            })

        # Test bars for replay
        test_closes = test_data["close_prices"]
        test_dates = test_data["dates"]
        test_fng = test_data["fng_norm"] * 100

        test_df = pd.DataFrame({
            "open": test_closes,
            "high": test_closes,
            "low": test_closes,
            "close": test_closes,
            "volume": np.zeros(len(test_closes)),
        }, index=pd.to_datetime(test_dates))

        fng_series = pd.Series(test_fng, index=pd.to_datetime(test_dates))
        feed = HistoricalReplayFeed(test_df, fng_series)

        return feed, warmup_bars

    def _print_summary(self, stats: dict) -> None:
        """Print multi-asset summary table."""
        print(f"\n{'=' * 70}")
        print("  Multi-Asset Paper Trading Summary")
        print(f"{'=' * 70}")
        print(f"  {'Asset':<12} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinRate':>8}")
        print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for asset_id, metrics in stats["per_asset"].items():
            disabled = " [OFF]" if asset_id in self._disabled_assets else ""
            print(f"  {asset_id + disabled:<12} "
                  f"{metrics['cumulative_return']:>9.2%} "
                  f"{metrics['sharpe_ratio']:>8.3f} "
                  f"{metrics['max_drawdown']:>8.2%} "
                  f"{metrics['trade_count']:>8d} "
                  f"{metrics['win_rate']:>7.1%}")

        print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'TOTAL':<12} "
              f"{stats['combined_return']:>9.2%} "
              f"{'':>8} "
              f"{'':>8} "
              f"{stats['total_trades']:>8d} "
              f"{'':>8}")
        print(f"  Portfolio value: ${stats['total_value']:,.2f}")
        print(f"  Active assets: {stats['active_assets']}")
        if stats["disabled_assets"]:
            print(f"  Disabled (max DD): {stats['disabled_assets']}")
        print(f"{'=' * 70}")

    def _save_summary(self, stats: dict) -> None:
        """Save summary to JSON."""
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

    def _save_states(self) -> None:
        """Save state for all traders for crash recovery."""
        os.makedirs(self.state_dir, exist_ok=True)
        for asset_id, trader in self.traders.items():
            state_path = os.path.join(self.state_dir, f"{asset_id}.json")
            trader.save_state(state_path)

    def _write_combined_log(self, results: dict) -> None:
        """Append daily results to combined CSV log."""
        os.makedirs(self.log_dir, exist_ok=True)
        csv_path = os.path.join(self.log_dir, "combined_log.csv")
        file_exists = os.path.exists(csv_path)

        fieldnames = ["date", "asset", "action", "price", "portfolio_value",
                       "trade_executed", "cumulative_return"]

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for asset_id, result in results.items():
                if result is not None:
                    writer.writerow({
                        "date": result["date"],
                        "asset": asset_id,
                        "action": result["action_name"],
                        "price": f"{result['price']:.2f}",
                        "portfolio_value": f"{result['portfolio_value']:.2f}",
                        "trade_executed": result["trade_executed"],
                        "cumulative_return": f"{result.get('cumulative_return', 0):.4f}",
                    })
