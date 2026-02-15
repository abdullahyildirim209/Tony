"""Paper trading orchestrator: runs PPO model against live or historical data."""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO

from live.data_feed import BinanceLiveFeed, HistoricalReplayFeed
from live.feature_engine import FeatureEngine
from live.state_manager import StateManager


class PaperTrader:
    """Main paper trading loop: feed bars -> build obs -> predict -> execute -> log."""

    def __init__(
        self,
        model_path: str,
        train_npz_path: str,
        config: dict,
        feed,
        log_dir: str = "live/logs",
        testnet_executor=None,
    ):
        self.model = PPO.load(model_path)
        self.config = config
        self.feed = feed
        self.log_dir = log_dir
        self.testnet_executor = testnet_executor

        env_cfg = config["env"]
        ind_cfg = config["indicators"]

        clip_stats = FeatureEngine.load_clip_stats(train_npz_path)
        self.feature_engine = FeatureEngine(
            clip_stats=clip_stats,
            window_size=env_cfg["window_size"],
            sma_short=ind_cfg["sma_short"],
            sma_long=ind_cfg["sma_long"],
            rsi_period=ind_cfg["rsi_period"],
        )
        self.state_manager = StateManager(
            initial_cash=env_cfg["initial_cash"],
            transaction_cost=env_cfg["transaction_cost"],
            max_drawdown_threshold=env_cfg["max_drawdown_threshold"],
        )

        os.makedirs(log_dir, exist_ok=True)
        self._step_log: list[dict] = []
        self._csv_path = os.path.join(log_dir, "paper_trading.csv")

    def warmup(self, historical_bars: list[dict]) -> None:
        """Feed historical bars to build indicator buffers before trading starts."""
        for bar in historical_bars:
            self.feature_engine.add_bar(bar)
        print(f"  Warmup complete: {len(historical_bars)} bars fed, "
              f"buffer size: {len(self.feature_engine.bars)}")

    def step(self) -> dict | None:
        """Execute one daily cycle: get bar -> build obs -> predict -> execute -> log.

        Returns step result dict or None if no bar available or insufficient history.
        """
        if isinstance(self.feed, HistoricalReplayFeed):
            bar = self.feed.next_bar()
        elif isinstance(self.feed, BinanceLiveFeed):
            bar = self.feed.get_latest_daily_bar()
        else:
            bar = self.feed.next_bar()

        if bar is None:
            return None

        price = bar["close"]

        # Add bar to feature engine
        self.feature_engine.add_bar(bar)

        # Build observation
        position_state = self.state_manager.get_position_state(price)
        obs = self.feature_engine.get_obs(position_state)

        if obs is None:
            return None

        # Validate observation — skip day if NaN/Inf
        if not np.all(np.isfinite(obs)):
            bad = np.where(~np.isfinite(obs))[0].tolist()
            print(f"  WARNING: NaN/Inf in obs at indices {bad} on {bar['date']}, skipping")
            return None

        # Model prediction
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        # Capture pre-trade state for testnet order quantities
        shares_before = self.state_manager.shares
        cash_before = self.state_manager.cash

        # Execute
        result = self.state_manager.execute_action(action, price)

        # Mirror trade on testnet if executor provided
        if self.testnet_executor is not None and result["trade_executed"]:
            symbol = self.config["data"].get("binance_symbol", "BTCUSDT")
            if action == 0:  # Buy
                self.testnet_executor.place_market_buy(cash_before, symbol=symbol)
            elif action == 2:  # Sell
                self.testnet_executor.place_market_sell(shares_before, symbol=symbol)

        # Cumulative return
        cum_return = (result["portfolio_value"] / self.state_manager.initial_cash) - 1.0

        # Unrealized PnL
        unrealized_pnl = 0.0
        if self.state_manager.shares > 0 and self.state_manager.entry_price > 0:
            unrealized_pnl = (price / self.state_manager.entry_price) - 1.0

        # Log
        step_info = {
            "date": bar["date"],
            "price": price,
            "action": action,
            "action_name": ["Buy", "Hold", "Sell"][action],
            "trade_executed": result["trade_executed"],
            "portfolio_value": result["portfolio_value"],
            "cash": self.state_manager.cash,
            "shares": self.state_manager.shares,
            "reward": result["reward"],
            "unrealized_pnl": unrealized_pnl,
            "cumulative_return": cum_return,
        }
        self._step_log.append(step_info)

        # CSV logging
        self._append_csv(step_info)

        return step_info

    def _append_csv(self, step_info: dict) -> None:
        """Append one row to the CSV trade log."""
        file_exists = os.path.exists(self._csv_path)
        fieldnames = [
            "date", "close_price", "action", "position", "portfolio_value",
            "unrealized_pnl", "trade_executed", "cumulative_return",
        ]
        row = {
            "date": step_info["date"],
            "close_price": f"{step_info['price']:.2f}",
            "action": step_info["action_name"],
            "position": "Long" if self.state_manager.shares > 0 else "Flat",
            "portfolio_value": f"{step_info['portfolio_value']:.2f}",
            "unrealized_pnl": f"{step_info.get('unrealized_pnl', 0):.4f}",
            "trade_executed": step_info["trade_executed"],
            "cumulative_return": f"{step_info.get('cumulative_return', 0):.4f}",
        }
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def run_replay(self) -> dict:
        """Run full historical replay. Returns metrics dict."""
        print("Running historical replay...")
        step_count = 0
        while True:
            result = self.step()
            if result is None:
                if isinstance(self.feed, HistoricalReplayFeed) and self.feed.remaining() == 0:
                    break
                if result is None:
                    break
            else:
                step_count += 1
                if step_count % 100 == 0:
                    print(f"  Step {step_count}: {result['date']} | "
                          f"${result['portfolio_value']:,.2f} | {result['action_name']}")

            # Check termination
            if self.state_manager.portfolio_values:
                peak = max(self.state_manager.portfolio_values)
                current = self.state_manager.portfolio_values[-1]
                if peak > 0 and (peak - current) / peak >= self.state_manager.max_drawdown_threshold:
                    print(f"  Max drawdown reached at step {step_count}")
                    break

        metrics = self.state_manager.get_metrics()
        self._save_log("replay")

        print(f"\nReplay complete ({step_count} steps)")
        print(f"  Return: {metrics['cumulative_return']:.2%}")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
        print(f"  MaxDD:  {metrics['max_drawdown']:.2%}")
        print(f"  Trades: {metrics['trade_count']}")
        print(f"  WinRate:{metrics['win_rate']:.1%}")
        print(f"  Final:  ${metrics['final_value']:,.2f}")

        return metrics

    def run_live(self, check_interval_hours: float = 1.0) -> None:
        """Live paper trading loop: check for new daily bars periodically."""
        print(f"Starting live paper trading (checking every {check_interval_hours}h)...")
        print(f"  Model: {self.config['training']['best_model_name']}")
        print(f"  Initial cash: ${self.state_manager.initial_cash:,.2f}")

        interval_secs = check_interval_hours * 3600

        try:
            while True:
                result = self.step()
                if result is not None:
                    print(f"  [{result['date']}] {result['action_name']:>4} | "
                          f"Price: ${result['price']:,.2f} | "
                          f"Portfolio: ${result['portfolio_value']:,.2f} | "
                          f"Trade: {result['trade_executed']}")
                    self._save_log("live")
                    self.save_state(os.path.join(
                        self.config.get("live", {}).get("state_dir", "live/state"),
                        "latest.json",
                    ))

                time.sleep(interval_secs)

        except KeyboardInterrupt:
            print("\nLive trading stopped by user.")
            metrics = self.state_manager.get_metrics()
            print(f"  Final portfolio: ${metrics['final_value']:,.2f}")
            print(f"  Return: {metrics['cumulative_return']:.2%}")
            self._save_log("live")

    def _save_log(self, prefix: str) -> None:
        """Save step log to JSON."""
        path = os.path.join(self.log_dir, f"{prefix}_log.json")
        with open(path, "w") as f:
            json.dump(self._step_log, f, indent=2, default=str)

    def save_state(self, path: str) -> None:
        """Persist full state for crash recovery."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "state_manager": self.state_manager.to_dict(),
            "feature_engine": self.feature_engine.get_state(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, path: str) -> None:
        """Restore state from JSON."""
        with open(path, "r") as f:
            state = json.load(f)
        self.state_manager = StateManager.from_dict(state["state_manager"])
        self.feature_engine.load_state(state["feature_engine"])
        print(f"  Restored state from {path} (step {self.state_manager.current_step})")
