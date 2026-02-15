"""Binance Spot Testnet order execution via raw requests + HMAC-SHA256 signing.

Uses exactly 2 endpoints (order, balance) — no python-binance dependency needed.
Testnet errors never block the model; all methods return None on failure.

Requires env vars: BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_API_SECRET
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from urllib.parse import urlencode

import requests


class TestnetExecutor:
    """Binance Spot Testnet market order client."""

    BASE_URL = "https://testnet.binance.vision"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("BINANCE_TESTNET_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("BINANCE_TESTNET_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Testnet credentials required. Set BINANCE_TESTNET_API_KEY "
                "and BINANCE_TESTNET_API_SECRET environment variables."
            )

    def _sign(self, params: dict) -> dict:
        """Add timestamp and HMAC-SHA256 signature to params."""
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _headers(self) -> dict:
        return {"X-MBX-APIKEY": self.api_key}

    def place_market_buy(self, quote_qty: float, symbol: str = "BTCUSDT") -> dict | None:
        """MARKET BUY spending quote_qty USDT. Returns order response or None."""
        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": f"{quote_qty:.2f}",
        }
        return self._place_order(params)

    def place_market_sell(self, quantity: float, symbol: str = "BTCUSDT") -> dict | None:
        """MARKET SELL quantity BTC. Returns order response or None."""
        params = {
            "symbol": symbol,
            "side": "SELL",
            "type": "MARKET",
            "quantity": f"{quantity:.8f}",
        }
        return self._place_order(params)

    def _place_order(self, params: dict) -> dict | None:
        """Send signed order request."""
        try:
            resp = requests.post(
                f"{self.BASE_URL}/api/v3/order",
                params=self._sign(params),
                headers=self._headers(),
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()
            print(f"  TESTNET ORDER: {result.get('side')} {result.get('symbol')} "
                  f"status={result.get('status')} fills={result.get('fills', [])}")
            return result
        except (requests.RequestException, ValueError) as e:
            print(f"  WARNING: Testnet order failed: {e}")
            return None

    def get_balance(self, asset: str = "USDT") -> float | None:
        """Query testnet balance for an asset. Returns free balance or None."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/api/v3/account",
                params=self._sign({}),
                headers=self._headers(),
                timeout=15,
            )
            resp.raise_for_status()
            for bal in resp.json().get("balances", []):
                if bal["asset"] == asset:
                    return float(bal["free"])
            return 0.0
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"  WARNING: Testnet balance query failed: {e}")
            return None
