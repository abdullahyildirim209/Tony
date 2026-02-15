# Tried Methods

All experiments and methods tried throughout the Tony project, organized chronologically.

---

## 1. Baseline 35-dim DQN (yfinance) — Feb 14

- **Obs space (35,):** `[30 pct_changes, sma_ratio, rsi_norm, flat_flag, long_flag, unrealized_pnl]`
- **Data:** yfinance BTC-USD, 2019–2024
- **Result:** Sharpe 2.169, Return 196.29%, MaxDD 18.68%, 21 trades
- **Key fixes applied:**
  - Fractional shares (BTC too expensive for whole-unit trading with $10k)
  - Data leakage fix (clip stats computed on train only, not full dataset)
  - Multi-episode validation (5 episodes averaged across val data)
  - Uncapped val/test episode length (train still capped at 252 steps)

---

## 2. Walk-Forward Testing (5 market regimes) — Feb 15

- **Method:** Expanding window across 5 regimes: 2020 crash, 2021 bull, 2022 bear, 2023 recovery, 2024 bull
- **Purpose:** Test generalization across different market conditions
- **Results:** `results/walk_forward/`

---

## 3. Multi-Asset Testing (5 assets) — Feb 15

- **Assets:** BTC-USD, ETH-USD, SOL-USD, SPY, QQQ
- **Notes:** Crypto assets get real FNG; stocks get neutral 0.5
- **Results:** `results/multi_asset/`

---

## 4. Transaction Cost Sensitivity (6 levels) — Feb 15

- **TC levels:** 0.0%, 0.05%, 0.1%, 0.2%, 0.5%, 1.0%
- **Phase A:** Evaluate pretrained model at each TC level
- **Phase B:** Retrain from scratch at each TC level
- **Results:** `results/tc_sensitivity/`

---

## 5. Statistical Analysis — Feb 15

- **Methods:** Bootstrap Sharpe confidence intervals, per-trade P&L distribution, t-test, Wilcoxon signed-rank test
- **Results:** `results/statistics/`

---

## 6. 36-dim: +Fear & Greed Index (FNG) — Feb 15

- **Change:** Added `fng_norm` from alternative.me Crypto Fear & Greed API
- **Obs space:** Expanded from (35,) to (36,)
- **Notes:** FNG normalized 0–1, cached to CSV, non-crypto assets use 0.5 (neutral)

---

## 7. 37-dim: +Buy Pressure (yfinance fallback) — Feb 15

- **Change:** Added `buy_pressure = taker_buy_volume / volume`
- **Problem:** yfinance doesn't provide `taker_buy_volume` → hardcoded 0.5 (no signal)
- **Obs space:** Expanded from (36,) to (37,)
- **Result:** Sharpe 1.434 — worse than baseline (pure noise, zero information content)

---

## 8. 37-dim: +Buy Pressure (HuggingFace real data) — Feb 15

- **Change:** Switched to `123olp/binance-futures-ohlcv-2018-2026` HuggingFace dataset for real `taker_buy_volume`
- **Data fixes:** Filtered 12,960 corrupt rows from raw 1-min candles
- **Signal quality:** `buy_pressure` std = 0.0156 (real variance, not constant)
- **Result:** Sharpe 1.564, Return 119.58%, MaxDD 26.26%, 1 trade
- **Notes:**
  - Agent converged to buy-and-hold strategy (only 1 trade)
  - Fewer training rows (1,046 vs ~1,460) because HF data starts Feb 2020 (when FNG begins), not 2019
  - Still below baseline Sharpe of 2.169

---

## 9. Paper Trading System — Feb 15

- **Components:**
  - `FeatureEngine` — Standalone 37-dim obs builder with rolling buffer (mirrors `TradingEnv._get_obs()`)
  - `StateManager` — Position/portfolio tracker (mirrors `TradingEnv.step()`)
  - `HistoricalReplayFeed` — Replays test data for offline validation
  - `BinanceLiveFeed` — Fetches latest daily candle from Binance REST API
- **Modes:** `--mode replay` (historical) and `--mode live` (real-time)
- **Features:** Crash recovery via JSON state persistence (`--resume`)
