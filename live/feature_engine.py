"""Standalone observation builder for paper trading.

Replicates TradingEnv._get_obs() logic using a rolling bar buffer,
without requiring a Gymnasium environment.
"""

from __future__ import annotations

import numpy as np


class FeatureEngine:
    """Builds 37-dim observation vectors from streaming daily bars.

    Mirrors the observation construction in env/trading_env.py:
        [0:30]  - Last 30 daily pct changes
        [30]    - SMA ratio (sma_short / sma_long)
        [31]    - RSI normalized (0-1)
        [32]    - FNG normalized (0-1)
        [33]    - Buy pressure (taker_buy_volume / volume, 0-1)
        [34]    - Position one-hot: flat
        [35]    - Position one-hot: long
        [36]    - Unrealized PnL % (0 if flat)
    """

    def __init__(
        self,
        clip_stats: dict,
        window_size: int = 30,
        sma_short: int = 7,
        sma_long: int = 21,
        rsi_period: int = 14,
    ):
        self.clip_stats = clip_stats
        self.window_size = window_size
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.rsi_period = rsi_period

        # Rolling buffer of daily bars: {date, open, high, low, close, volume, fng}
        self.bars: list[dict] = []
        # Need enough history for sma_long + window_size + some padding
        self.buffer_size = sma_long + window_size + 10

        # RSI state (Wilder's EWM)
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._rsi_count = 0

    def add_bar(self, bar: dict) -> None:
        """Append a daily bar. Expected keys: date, open, high, low, close, volume, fng."""
        self.bars.append(bar)
        if len(self.bars) > self.buffer_size:
            self.bars = self.bars[-self.buffer_size:]

        # Update RSI incrementally
        if len(self.bars) >= 2:
            delta = self.bars[-1]["close"] - self.bars[-2]["close"]
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            self._rsi_count += 1

            alpha = 1.0 / self.rsi_period
            if self._rsi_count <= self.rsi_period:
                # Simple average during warmup
                self._avg_gain = self._avg_gain + (gain - self._avg_gain) / self._rsi_count
                self._avg_loss = self._avg_loss + (loss - self._avg_loss) / self._rsi_count
            else:
                # Wilder's EWM
                self._avg_gain = alpha * gain + (1 - alpha) * self._avg_gain
                self._avg_loss = alpha * loss + (1 - alpha) * self._avg_loss

    def get_obs(self, position_state: dict) -> np.ndarray | None:
        """Build 37-dim observation vector.

        Args:
            position_state: {is_long: bool, entry_price: float, current_price: float}

        Returns:
            np.ndarray of shape (37,) or None if insufficient history.
        """
        min_bars = self.sma_long + self.window_size
        if len(self.bars) < min_bars:
            return None

        closes = [b["close"] for b in self.bars]

        # Last window_size pct changes
        pct_changes = []
        for i in range(len(closes) - self.window_size, len(closes)):
            if closes[i - 1] > 0:
                pct_changes.append(closes[i] / closes[i - 1] - 1.0)
            else:
                pct_changes.append(0.0)
        pct_changes = np.array(pct_changes, dtype=np.float32)

        # SMA ratio
        recent = closes[-self.sma_long:]
        sma_short_val = np.mean(closes[-self.sma_short:])
        sma_long_val = np.mean(recent)
        sma_ratio = sma_short_val / sma_long_val if sma_long_val > 0 else 1.0

        # RSI normalized
        if self._avg_loss > 0:
            rs = self._avg_gain / self._avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi = 100.0 if self._avg_gain > 0 else 50.0
        rsi_norm = rsi / 100.0

        # FNG normalized
        fng = self.bars[-1].get("fng", 50)
        fng_norm = fng / 100.0

        # Buy pressure
        latest_vol = self.bars[-1].get("volume", 1.0)
        latest_tbv = self.bars[-1].get("taker_buy_volume", latest_vol * 0.5)
        bp = (latest_tbv / latest_vol) if latest_vol > 0 else 0.5
        bp = float(np.clip(bp, 0.0, 1.0))

        # Position encoding
        is_long = position_state.get("is_long", False)
        flat = 1.0 if not is_long else 0.0
        long = 1.0 if is_long else 0.0

        # Unrealized PnL
        if is_long and position_state.get("entry_price", 0) > 0:
            unrealized_pnl = (position_state["current_price"] / position_state["entry_price"]) - 1.0
        else:
            unrealized_pnl = 0.0

        # Apply clip stats
        pct_changes = self._clip_feature(pct_changes, "pct_changes")
        sma_ratio = self._clip_scalar(sma_ratio, "sma_ratios")
        rsi_norm = self._clip_scalar(rsi_norm, "rsi_norm")
        fng_norm = self._clip_scalar(fng_norm, "fng_norm")
        bp = self._clip_scalar(bp, "buy_pressure")

        obs = np.concatenate([
            pct_changes,
            [sma_ratio, rsi_norm, fng_norm, bp, flat, long, unrealized_pnl],
        ]).astype(np.float32)

        return obs

    def _clip_feature(self, arr: np.ndarray, name: str) -> np.ndarray:
        if name in self.clip_stats:
            mean, std = self.clip_stats[name]
            if std > 0:
                return np.clip(arr, mean - 5 * std, mean + 5 * std)
        return arr

    def _clip_scalar(self, val: float, name: str) -> float:
        if name in self.clip_stats:
            mean, std = self.clip_stats[name]
            if std > 0:
                return float(np.clip(val, mean - 5 * std, mean + 5 * std))
        return val

    def get_state(self) -> dict:
        """Serialize state for persistence."""
        return {
            "bars": self.bars,
            "avg_gain": self._avg_gain,
            "avg_loss": self._avg_loss,
            "rsi_count": self._rsi_count,
        }

    def load_state(self, state: dict) -> None:
        """Restore state from dict."""
        self.bars = state["bars"]
        self._avg_gain = state["avg_gain"]
        self._avg_loss = state["avg_loss"]
        self._rsi_count = state["rsi_count"]

    @staticmethod
    def load_clip_stats(train_npz_path: str) -> dict:
        """Load clip stats from train.npz (saved by data/fetch_data.py)."""
        data = np.load(train_npz_path, allow_pickle=True)
        stats = {}
        for col in ["pct_changes", "sma_ratios", "rsi_norm", "fng_norm", "buy_pressure"]:
            mean_key = f"clip_{col}_mean"
            std_key = f"clip_{col}_std"
            if mean_key in data and std_key in data:
                stats[col] = (float(data[mean_key]), float(data[std_key]))
        return stats
