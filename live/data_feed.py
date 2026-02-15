"""Data feed providers for paper trading: historical replay and Binance live."""

from __future__ import annotations

import datetime

import pandas as pd
import requests


class HistoricalReplayFeed:
    """Replays daily bars from a pre-loaded DataFrame. For testing the paper trading pipeline."""

    def __init__(self, daily_df: pd.DataFrame, fng_series: pd.Series | None = None):
        """
        Args:
            daily_df: DataFrame with columns [open, high, low, close, volume], DatetimeIndex.
            fng_series: Optional Series of FNG values indexed by date. Defaults to 50 (neutral).
        """
        self.daily_df = daily_df.copy()
        self.fng_series = fng_series
        self._idx = 0

    def reset(self):
        self._idx = 0

    def next_bar(self) -> dict | None:
        """Return next daily bar as dict, or None if exhausted."""
        if self._idx >= len(self.daily_df):
            return None

        row = self.daily_df.iloc[self._idx]
        date = self.daily_df.index[self._idx]
        self._idx += 1

        fng = 50  # default neutral
        if self.fng_series is not None:
            # Find closest FNG value
            if date in self.fng_series.index:
                fng = int(self.fng_series.loc[date])
            else:
                # Forward-fill: find most recent FNG before this date
                mask = self.fng_series.index <= date
                if mask.any():
                    fng = int(self.fng_series.loc[mask].iloc[-1])

        vol = float(row["volume"])
        tbv = float(row.get("taker_buy_volume", vol * 0.5)) if hasattr(row, "get") else vol * 0.5

        return {
            "date": str(date.date()) if hasattr(date, "date") else str(date),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": vol,
            "taker_buy_volume": tbv,
            "fng": fng,
        }

    def remaining(self) -> int:
        return len(self.daily_df) - self._idx


class BinanceLiveFeed:
    """Fetches latest daily candle from Binance public REST API (no auth needed)."""

    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, symbol: str = "BTCUSDT", fng_api: bool = True):
        self.symbol = symbol
        self.fng_api = fng_api
        self._last_bar_date: str | None = None

    def get_latest_daily_bar(self) -> dict | None:
        """Fetch the most recent completed daily candle.

        Returns dict with {date, open, high, low, close, volume, fng} or None on failure.
        """
        try:
            resp = requests.get(
                self.BASE_URL,
                params={"symbol": self.symbol, "interval": "1d", "limit": "2"},
                timeout=15,
            )
            resp.raise_for_status()
            klines = resp.json()

            if len(klines) < 2:
                return None

            # Use the second-to-last candle (most recent completed)
            k = klines[-2]
            bar_date = datetime.datetime.fromtimestamp(
                k[0] / 1000, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d")

            # Skip if we already processed this date
            if bar_date == self._last_bar_date:
                return None
            self._last_bar_date = bar_date

            fng = self._fetch_fng() if self.fng_api else 50

            return {
                "date": bar_date,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "taker_buy_volume": float(k[9]),
                "fng": fng,
            }

        except (requests.RequestException, ValueError, IndexError) as e:
            print(f"  WARNING: Binance API error: {e}")
            return None

    def fetch_historical_bars(self, limit: int = 60) -> list[dict]:
        """Fetch recent completed daily bars from Binance for warmup.

        Args:
            limit: Number of completed daily bars to fetch (max ~1000).

        Returns:
            List of bar dicts sorted oldest-first, or empty list on failure.
        """
        try:
            # Fetch limit+1 to drop the last (incomplete) candle
            resp = requests.get(
                self.BASE_URL,
                params={"symbol": self.symbol, "interval": "1d", "limit": str(limit + 1)},
                timeout=30,
            )
            resp.raise_for_status()
            klines = resp.json()

            if len(klines) < 2:
                return []

            # Drop last candle (still open/incomplete)
            klines = klines[:-1]

            # Fetch FNG history for the same date range
            fng_map = self._fetch_fng_history(len(klines))

            bars = []
            for k in klines:
                bar_date = datetime.datetime.fromtimestamp(
                    k[0] / 1000, tz=datetime.timezone.utc
                ).strftime("%Y-%m-%d")
                bars.append({
                    "date": bar_date,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "taker_buy_volume": float(k[9]),
                    "fng": fng_map.get(bar_date, 50),
                })

            return bars

        except (requests.RequestException, ValueError, IndexError) as e:
            print(f"  WARNING: Failed to fetch historical bars: {e}")
            return []

    def _fetch_fng_history(self, days: int) -> dict[str, int]:
        """Fetch FNG values for the last N days. Returns {date_str: fng_value}."""
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/",
                params={"limit": str(days), "format": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            result = {}
            for entry in resp.json().get("data", []):
                ts = int(entry["timestamp"])
                date_str = datetime.datetime.fromtimestamp(
                    ts, tz=datetime.timezone.utc
                ).strftime("%Y-%m-%d")
                result[date_str] = int(entry["value"])
            return result
        except (requests.RequestException, KeyError, ValueError):
            return {}

    def _fetch_fng(self) -> int:
        """Fetch current Fear & Greed Index."""
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/",
                params={"limit": "1", "format": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            return int(resp.json()["data"][0]["value"])
        except (requests.RequestException, KeyError, ValueError, IndexError):
            return 50  # neutral fallback
