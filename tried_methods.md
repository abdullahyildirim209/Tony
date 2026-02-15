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

---

## 10. DQN → PPO Migration — Feb 15

- **Change:** Replaced DQN with PPO (Proximal Policy Optimization)
- **Reason:** PPO's on-policy learning and entropy bonus better suited for exploration
- **Config:** `n_steps=2048, batch_size=64, n_epochs=10, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01`
- **Result:** Walk-forward 5-fold test showed PPO converges to passive strategies

---

## 11. Anti-Passivity Reward & Validation Fixes — Feb 15

- **Problem:** Walk-forward results (500k timesteps, 5 folds) showed PPO converging to passive strategies:
  - fold_1: -50.61% return, 1 trade (bought, never sold through 2022 crash)
  - fold_2: +55.53% return, 23 trades (**good** — active trading, beat B&H)
  - fold_3: +50.58% return, 3 trades (barely trades, matches B&H)
  - fold_4: +80.62% return, 1 trade (mimics B&H exactly)
  - fold_5: 0.00% return, 0 trades (zero trades, missed 60% rally)
- **Root causes:** Holding cash gives reward=0 (risk-free local optimum); validation selects passive models (Sharpe=0 beats negative Sharpe); early stopping too tight; deterministic inference amplifies passivity
- **Changes:**
  1. **Opportunity cost reward** (`env/trading_env.py`): When agent is flat and market rises, penalize with `0.5 * market_return`. Breaks the "do nothing" attractor without forcing buying.
  2. **Minimum trade filter** (`agent/train.py`): Validation episodes with <3 trades get Sharpe=-inf. Prevents passive models from being saved as "best".
  3. **Config tuning** (`configs/default.yaml`): `ent_coef: 0.01→0.02` (more exploration), `early_stopping_patience: 5→10` (more time to escape local minima)
- **Files modified:** `env/trading_env.py`, `agent/train.py`, `configs/default.yaml`
- **Result (v1):** No change — 0.5x opportunity cost too weak, min-trades filter had fallback bug (passive final model saved anyway)
- **Fixes (v2):**
  - Opportunity cost: increased to 1.0x missed upside + constant flat penalty (`-0.0005/step`)
  - Min-trades: switched from per-episode `-inf` to avg trade count < 3 threshold
  - Fallback bug: added `ever_saved` flag, WARNING when no active model found
  - `ent_coef: 0.02→0.05` (5x original)
- **Result (v2):**
  - fold_5: **fixed** — 0 trades → 3 trades, 0% → +60.31% (beats B&H)
  - Mean PPO Sharpe: -0.081 → +0.169 (now matches B&H)
  - fold_2: slight regression (23→15 trades, 55%→48% return, still beats baselines)
  - fold_1/3/4: still converge to buy-and-hold (4/5 folds show WARNING: no active model)
  - Agent rationally prefers B&H in trending markets — may need structural changes (e.g. short selling) to differentiate
