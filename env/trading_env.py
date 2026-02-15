"""Custom Gymnasium trading environment for single-asset DQN trading."""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """Single-asset trading environment with discrete Buy/Hold/Sell actions.

    Observation space (37,):
        [0:30]  - Last 30 daily pct changes
        [30]    - SMA ratio (sma_short / sma_long)
        [31]    - RSI normalized (0-1)
        [32]    - FNG normalized (0-1)
        [33]    - Buy pressure (taker_buy_volume / volume, 0-1)
        [34]    - Position one-hot: flat
        [35]    - Position one-hot: long
        [36]    - Unrealized PnL % (0 if flat)

    Action space: Discrete(3) — 0=Buy, 1=Hold, 2=Sell
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        close_prices: np.ndarray,
        pct_changes: np.ndarray,
        sma_ratios: np.ndarray,
        rsi_norm: np.ndarray,
        fng_norm: Optional[np.ndarray] = None,
        buy_pressure: Optional[np.ndarray] = None,
        window_size: int = 30,
        episode_length: int = 252,
        initial_cash: float = 10000.0,
        transaction_cost: float = 0.001,
        max_drawdown_threshold: float = 0.50,
        mode: str = "train",
    ):
        super().__init__()

        self.close_prices = close_prices.astype(np.float32)
        self.pct_changes = pct_changes.astype(np.float32)
        self.sma_ratios = sma_ratios.astype(np.float32)
        self.rsi_norm = rsi_norm.astype(np.float32)
        n = len(close_prices)
        self.fng_norm = fng_norm.astype(np.float32) if fng_norm is not None else np.full(n, 0.5, dtype=np.float32)
        self.buy_pressure = buy_pressure.astype(np.float32) if buy_pressure is not None else np.full(n, 0.5, dtype=np.float32)

        self.window_size = window_size
        self.episode_length = episode_length
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_drawdown_threshold = max_drawdown_threshold
        self.mode = mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Will be set in reset()
        self.cash = initial_cash
        self.shares = 0.0  # fractional shares held
        self.entry_price = 0.0
        self.portfolio_values = []
        self.start_idx = window_size
        self.current_step = 0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.actions_taken = []

        self._rng = np.random.default_rng()

    def _get_obs(self) -> np.ndarray:
        idx = self.start_idx + self.current_step

        # Last window_size pct changes
        start = idx - self.window_size + 1
        end = idx + 1
        pct_window = self.pct_changes[start:end]

        # Pad if needed (shouldn't happen if data is prepared correctly)
        if len(pct_window) < self.window_size:
            pct_window = np.pad(
                pct_window, (self.window_size - len(pct_window), 0), constant_values=0.0
            )

        sma_ratio = self.sma_ratios[idx]
        rsi = self.rsi_norm[idx]
        fng = self.fng_norm[idx]
        bp = self.buy_pressure[idx]

        # Position encoding
        is_long = self.shares > 0
        flat = 1.0 if not is_long else 0.0
        long = 1.0 if is_long else 0.0

        # Unrealized PnL
        if is_long and self.entry_price > 0:
            unrealized_pnl = (self.close_prices[idx] / self.entry_price) - 1.0
        else:
            unrealized_pnl = 0.0

        obs = np.concatenate([
            pct_window,
            [sma_ratio, rsi, fng, bp, flat, long, unrealized_pnl],
        ]).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        data_len = len(self.close_prices)

        # Support forced_start_idx via options dict (for multi-episode validation)
        forced = (options or {}).get("forced_start_idx", None)

        if forced is not None:
            self.start_idx = int(forced)
        elif self.mode == "train":
            max_start = data_len - self.episode_length - 1
            if max_start <= self.window_size:
                self.start_idx = self.window_size
            else:
                self.start_idx = self._rng.integers(self.window_size, max_start)
        else:
            self.start_idx = self.window_size

        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.entry_price = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.portfolio_values = [self.initial_cash]
        self.actions_taken = []

        obs = self._get_obs()
        info = {"portfolio_value": self.initial_cash}
        return obs, info

    def step(self, action):
        idx = self.start_idx + self.current_step
        price = self.close_prices[idx]
        prev_portfolio = self.portfolio_values[-1]
        trade_executed = False

        # Execute action (fractional shares: go all-in or sell all)
        if action == 0:  # Buy
            if self.shares == 0 and self.cash > 0:
                # Buy as many fractional shares as cash allows
                affordable = self.cash / (price * (1.0 + self.transaction_cost))
                self.shares = affordable
                self.cash -= affordable * price * (1.0 + self.transaction_cost)
                self.entry_price = price
                self.trade_count += 1
                trade_executed = True
        elif action == 2:  # Sell
            if self.shares > 0:
                proceeds = self.shares * price * (1.0 - self.transaction_cost)
                self.cash += proceeds
                # Track win/loss
                if price > self.entry_price:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                self.shares = 0.0
                self.entry_price = 0.0
                self.trade_count += 1
                trade_executed = True
        # action == 1 (Hold) or invalid action → no-op

        # Advance step
        self.current_step += 1
        new_idx = self.start_idx + self.current_step

        # Compute portfolio value
        if new_idx < len(self.close_prices):
            portfolio_value = self.cash + self.shares * self.close_prices[new_idx]
        else:
            portfolio_value = self.cash + self.shares * price

        self.portfolio_values.append(portfolio_value)
        self.actions_taken.append(action)

        # Reward: log return - trade penalty
        if prev_portfolio > 0 and portfolio_value > 0:
            reward = float(np.log(portfolio_value / prev_portfolio))
        else:
            reward = 0.0
        reward -= 0.001 * float(trade_executed)

        # Termination conditions
        terminated = False
        truncated = False

        # Max drawdown check
        peak = max(self.portfolio_values)
        if peak > 0:
            drawdown = (peak - portfolio_value) / peak
            if drawdown >= self.max_drawdown_threshold:
                terminated = True

        # Episode length (train only) or data exhaustion
        if self.mode == "train" and self.current_step >= self.episode_length:
            truncated = True
        if new_idx >= len(self.close_prices) - 1:
            truncated = True

        info = {
            "portfolio_value": portfolio_value,
            "trade_executed": trade_executed,
            "shares": self.shares,
            "step": self.current_step,
        }

        obs = self._get_obs() if not (terminated or truncated) else self._get_terminal_obs(new_idx)

        return obs, reward, terminated, truncated, info

    def _get_terminal_obs(self, idx):
        """Return observation at terminal state, handling boundary."""
        idx = min(idx, len(self.close_prices) - 1)

        start = idx - self.window_size + 1
        start = max(start, 0)
        end = idx + 1
        pct_window = self.pct_changes[start:end]

        if len(pct_window) < self.window_size:
            pct_window = np.pad(
                pct_window, (self.window_size - len(pct_window), 0), constant_values=0.0
            )

        sma_ratio = self.sma_ratios[idx]
        rsi = self.rsi_norm[idx]
        fng = self.fng_norm[idx]
        bp = self.buy_pressure[idx]
        is_long = self.shares > 0
        flat = 1.0 if not is_long else 0.0
        long = 1.0 if is_long else 0.0

        if is_long and self.entry_price > 0:
            unrealized_pnl = (self.close_prices[idx] / self.entry_price) - 1.0
        else:
            unrealized_pnl = 0.0

        return np.concatenate([
            pct_window,
            [sma_ratio, rsi, fng, bp, flat, long, unrealized_pnl],
        ]).astype(np.float32)

    def get_episode_metrics(self) -> dict:
        """Compute episode performance metrics."""
        values = np.array(self.portfolio_values, dtype=np.float64)

        # Cumulative return
        cumulative_return = (values[-1] / values[0]) - 1.0 if values[0] > 0 else 0.0

        # Daily returns
        daily_returns = np.diff(values) / values[:-1]
        daily_returns = daily_returns[np.isfinite(daily_returns)]

        # Annualized Sharpe ratio
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(values)
        drawdowns = (running_max - values) / running_max
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Win rate
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0.0

        return {
            "cumulative_return": float(cumulative_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "trade_count": self.trade_count,
            "win_rate": float(win_rate),
        }
