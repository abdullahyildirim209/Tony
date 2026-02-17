"""Position and portfolio tracker for paper trading.

Mirrors the position management logic from env/trading_env.py TradingEnv.step()
without requiring a Gymnasium environment.
"""

import numpy as np


class StateManager:
    """Tracks cash, shares, portfolio value, and trade history."""

    def __init__(
        self,
        initial_cash: float = 10000.0,
        transaction_cost: float = 0.001,
        max_drawdown_threshold: float = 0.50,
    ):
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_drawdown_threshold = max_drawdown_threshold
        self.trade_penalty = 0.0005  # default, overridden by PaperTrader from config

        self.cash = initial_cash
        self.shares = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.current_step = 0

        self.portfolio_values: list[float] = [initial_cash]
        self.trade_log: list[dict] = []
        self.actions_taken: list[int] = []
        self.win_count = 0
        self.loss_count = 0
        self.trade_count = 0

    def execute_action(self, action: int, price: float) -> dict:
        """Execute a trading action. Mirrors TradingEnv.step() logic.

        Args:
            action: 0=Buy (all-in), 1=Hold, 2=Sell (all)
            price: Current close price

        Returns:
            Dict with trade_executed, portfolio_value, reward, terminated, truncated.
        """
        prev_portfolio = self.portfolio_values[-1]
        trade_executed = False

        if action == 0:  # Buy
            if self.shares == 0 and self.cash > 0:
                affordable = self.cash / (price * (1.0 + self.transaction_cost))
                self.shares = affordable
                self.cash -= affordable * price * (1.0 + self.transaction_cost)
                self.entry_price = price
                self.entry_step = self.current_step
                self.trade_count += 1
                trade_executed = True
        elif action == 2:  # Sell
            if self.shares > 0:
                proceeds = self.shares * price * (1.0 - self.transaction_cost)
                self.cash += proceeds
                self.trade_log.append({
                    "entry_price": self.entry_price,
                    "exit_price": price,
                    "pnl_pct": (price / self.entry_price) - 1.0,
                    "hold_steps": self.current_step - self.entry_step,
                })
                if price > self.entry_price:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                self.shares = 0.0
                self.entry_price = 0.0
                self.trade_count += 1
                trade_executed = True

        self.current_step += 1
        self.actions_taken.append(action)

        # Portfolio value
        portfolio_value = self.cash + self.shares * price
        self.portfolio_values.append(portfolio_value)

        # Reward: log return - trade penalty
        if prev_portfolio > 0 and portfolio_value > 0:
            reward = float(np.log(portfolio_value / prev_portfolio))
        else:
            reward = 0.0
        reward -= self.trade_penalty * float(trade_executed)

        # Termination: max drawdown
        terminated = False
        peak = max(self.portfolio_values)
        if peak > 0:
            drawdown = (peak - portfolio_value) / peak
            if drawdown >= self.max_drawdown_threshold:
                terminated = True

        return {
            "trade_executed": trade_executed,
            "portfolio_value": portfolio_value,
            "reward": reward,
            "terminated": terminated,
        }

    def get_position_state(self, current_price: float) -> dict:
        """Returns position info for FeatureEngine.get_obs()."""
        return {
            "is_long": self.shares > 0,
            "entry_price": self.entry_price,
            "current_price": current_price,
        }

    def get_metrics(self) -> dict:
        """Compute episode performance metrics. Same as TradingEnv.get_episode_metrics()."""
        values = np.array(self.portfolio_values, dtype=np.float64)

        cumulative_return = (values[-1] / values[0]) - 1.0 if values[0] > 0 else 0.0

        daily_returns = np.diff(values) / values[:-1]
        daily_returns = daily_returns[np.isfinite(daily_returns)]

        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe = 0.0

        running_max = np.maximum.accumulate(values)
        drawdowns = (running_max - values) / running_max
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0.0

        return {
            "cumulative_return": float(cumulative_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "trade_count": self.trade_count,
            "win_rate": float(win_rate),
            "final_value": float(values[-1]),
            "n_steps": len(values) - 1,
        }

    def to_dict(self) -> dict:
        """Serialize full state for crash recovery."""
        return {
            "initial_cash": self.initial_cash,
            "transaction_cost": self.transaction_cost,
            "max_drawdown_threshold": self.max_drawdown_threshold,
            "cash": self.cash,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_step": self.entry_step,
            "current_step": self.current_step,
            "portfolio_values": self.portfolio_values,
            "trade_log": self.trade_log,
            "actions_taken": self.actions_taken,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "trade_count": self.trade_count,
            "trade_penalty": self.trade_penalty,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StateManager":
        """Restore from serialized dict."""
        sm = cls(
            initial_cash=d["initial_cash"],
            transaction_cost=d["transaction_cost"],
            max_drawdown_threshold=d["max_drawdown_threshold"],
        )
        sm.cash = d["cash"]
        sm.shares = d["shares"]
        sm.entry_price = d["entry_price"]
        sm.entry_step = d["entry_step"]
        sm.current_step = d["current_step"]
        sm.portfolio_values = d["portfolio_values"]
        sm.trade_log = d["trade_log"]
        sm.actions_taken = d["actions_taken"]
        sm.win_count = d["win_count"]
        sm.loss_count = d["loss_count"]
        sm.trade_count = d["trade_count"]
        sm.trade_penalty = d.get("trade_penalty", 0.0005)
        return sm
