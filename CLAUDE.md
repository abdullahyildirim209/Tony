# Tony - Single-Asset RL Trading Agent

Tony is a reinforcement learning agent that trades a single asset (BTC-USD by default) using Stable Baselines3. It downloads OHLCV data, computes technical indicators, trains RL agents (PPO/A2C/DQN ensemble) in a custom Gymnasium environment, and evaluates performance against baselines (Buy & Hold, Random, SMA Crossover).

## Project Structure

```
tony/
  configs/default.yaml     - All hyperparameters and paths
  data/fetch_data.py       - Download, indicators, feature engineering, train/val/test split
  data/fetch_hf.py         - HuggingFace Binance dataset loader + daily aggregation
  env/trading_env.py       - Gymnasium TradingEnv (discrete Buy/Hold/Sell)
  agent/train.py           - SB3 PPO training with multi-episode validation & early stopping
  agent/ensemble.py        - Multi-algorithm ensemble (PPO, A2C, DQN), best-by-Sortino selection
  evaluation/backtest.py   - Run agent + baselines, compute metrics, generate plots
  experiments/walk_forward.py        - Walk-forward testing across market regimes
  experiments/statistical_analysis.py - Bootstrap CIs, significance tests
  live/feature_engine.py   - Standalone 38-dim obs builder with rolling buffer
  live/state_manager.py    - Position/portfolio tracker (mirrors env logic)
  live/data_feed.py        - Historical replay + Binance live feed providers
  live/paper_trader.py     - Paper trading orchestrator
  live/run_paper.py        - CLI entry point for paper trading
```

## How to Run

All commands run from the `tony/` directory:

```bash
# 1. Fetch data, compute features, split into train/val/test
python data/fetch_data.py

# 2. Train agent (early stops on val Sortino)
python agent/train.py

# 3. Backtest on test set + baselines
python evaluation/backtest.py

# 4. Walk-forward experiment (trains ensemble per fold)
python experiments/walk_forward.py
python experiments/statistical_analysis.py

# 5. Paper trading (after training)
python live/run_paper.py --mode replay    # test against historical data
python live/run_paper.py --mode live      # real-time paper trading via Binance
python live/run_paper.py --mode live --resume live/state/latest.json  # resume after crash
```

Config override: pass a YAML path as the first argument, e.g. `python agent/train.py configs/custom.yaml`.

## Reference Repos ("Bible")

The initial implementation drew patterns from these open-source projects:

| Pattern | Source |
|---------|--------|
| Yahoo Finance download, forward-fill cleaning | `FinRL/finrl/meta/data_processors/processor_yahoofinance.py` |
| `df_to_array()` feature array pattern | `FinRL/finrl/meta/data_processor.py` |
| State construction via `np.hstack()`, random start pattern | `FinRL-Meta/meta/env_stock_trading/env_stock_trading.py` |
| Portfolio tracking (`asset_memory`), transaction cost logic, invalid-action-as-Hold | `FinRL/finrl/meta/env_stock_trading/env_stocktrading.py` |
| Env scaling patterns | `ElegantRL/elegantrl/envs/StockTradingEnv.py` |
| SB3 model construction, TensorBoard callback | `FinRL/finrl/agents/stablebaselines3/models.py` |
| DQN algorithm reference | `ElegantRL/elegantrl/agents/AgentDQN.py` |
| Sharpe/drawdown formulas | `FinRL-Trading/src/backtest/backtest_engine.py` |
| Rolling Sharpe pattern | `FinRL-Trading/src/web/components.py` |
| Terminal reward bonus | `ElegantRL/elegantrl/envs/` |
| Multi-algorithm ensemble selection | `FinRL/finrl/agents/` |
| Turbulence regime indicator | `FinRL/finrl/meta/` |
| Sortino-based validation | `FinRL-Trading/src/backtest/` |

## Changelog (from initial implementation)

1. **Fractional shares fix** - BTC is too expensive for whole-unit trading with $10k initial cash. Changed to fractional share trading (go all-in or sell all) so the agent can actually open positions.

2. **Data leakage fix** (`data/fetch_data.py`) - Feature clipping (+-5 std devs) was computed on the full dataset before splitting, leaking val/test statistics into train features. Now: split first, compute clip stats from train only, apply those constants to val/test. Clip stats are saved in `train.npz` for reproducibility.

3. **Multi-episode validation** (`agent/train.py`) - Validation previously ran a single deterministic episode from a fixed start index, causing model selection to overfit to one market path. Now runs `n_val_episodes` (default 5) episodes with start offsets spread evenly across available val data, averages Sharpe/return/drawdown for model selection and early stopping.

4. **Uncapped episode length for val/test** (`env/trading_env.py`) - Val/test episodes were truncated at `episode_length=252` steps, wasting ~80 days of data per split. Now: the 252-step cap only applies in train mode; val/test episodes run until data exhaustion or max drawdown.

5. **Forced start index support** (`env/trading_env.py`) - Added `forced_start_idx` option via `reset(options={"forced_start_idx": N})` to support multi-episode validation with varied start positions.

6. **Fear & Greed Index feature** (`data/fetch_data.py`, `env/trading_env.py`) - Added Crypto Fear & Greed Index from alternative.me as an orthogonal sentiment signal. FNG is fetched via API, cached to CSV, normalized 0-1 (`fng_norm`), and included in the observation space.

7. **HuggingFace data integration** (`data/fetch_hf.py`, `data/fetch_data.py`) - Added support for `123olp/binance-futures-ohlcv-2018-2026` HuggingFace dataset as an alternative data source.

8. **Paper trading system** (`live/`) - Added a complete paper trading pipeline that reuses the trained model for inference without real money.

9. **Buy pressure feature** (`data/fetch_hf.py`, `data/fetch_data.py`, `env/trading_env.py`, `live/`) - Added `buy_pressure = taker_buy_volume / volume` as a new feature.

10. **Terminal reward bonus** (`env/trading_env.py`) - At episode end, adds `mean(episode_returns) / (1 - gamma)` as a terminal bonus. Provides ~100x stronger gradient signal about overall episode quality, directly attacking the passivity problem where PPO converges to buy-and-hold. Configurable via `env.terminal_reward_bonus`.

11. **Random initialization** (`env/trading_env.py`) - In train mode: randomizes initial cash ±10% and has a 30% chance of starting already long at a random entry price. Forces the agent to learn exit strategies, not just entry. Configurable via `env.random_init` and `env.random_init_long_prob`.

12. **Enhanced metrics** (`env/trading_env.py`, `evaluation/backtest.py`) - Added Sortino ratio (downside risk only), Calmar ratio (annualized return / max drawdown), skewness, and excess kurtosis to `get_episode_metrics()` and `print_metrics_table()`.

13. **Turbulence feature** (`data/fetch_data.py`, `env/trading_env.py`, `live/feature_engine.py`) - Computes turbulence as z-score of rolling volatility (63-day lookback). Added as observation feature (obs dim 37 → 38). Optionally force-sells when turbulence exceeds the 90th percentile of training data. Gives the agent explicit regime information.

14. **Sortino-based model selection** (`agent/train.py`) - Changed `ValidationCallback` selection criterion from Sharpe to Sortino. Sharpe penalizes upside vol equally — passive models score well. Sortino better identifies actively profitable models. Configurable via `validation.selection_metric`.

15. **Multi-algorithm ensemble** (`agent/ensemble.py`, `experiments/walk_forward.py`) - Trains PPO, A2C, and DQN per fold, selects best by validation Sortino. DQN's epsilon-greedy exploration is fundamentally different from PPO's entropy — may escape the passivity trap. Configurable via `ensemble.algorithms`.

16. **Vectorized parallel environments** (`agent/train.py`) - Uses `SubprocVecEnv` with configurable parallel envs (default 4) for 2-4x training speedup. Each env gets a different random start for more diverse experience per batch. Configured via `training.n_envs`.

17. **Trade logging** (`env/trading_env.py`) - Added `trade_log` list to TradingEnv that records entry_price, exit_price, pnl_pct, hold_steps for each completed trade. Used by statistical analysis for per-trade P&L distribution.

## Key Architectural Decisions

- **Observation space (38,):** `[30 pct_changes, sma_ratio, rsi_norm, fng_norm, buy_pressure, turbulence, flat_flag, long_flag, unrealized_pnl]`. Window of 30 daily returns gives the agent recent price context without raw price levels (scale-invariant). Turbulence provides explicit regime awareness.
- **Action space:** Discrete(3) - Buy (all-in), Hold, Sell (all). Invalid actions (e.g., Buy when already long) are treated as Hold.
- **Reward:** `log(portfolio_value_t / portfolio_value_{t-1}) - 0.001 * trade_executed` + terminal bonus of `mean(episode_returns) / (1 - gamma)` at episode end.
- **Termination:** Max drawdown >= 50% (terminated) or data exhaustion / episode length cap in train (truncated).
- **Feature clipping:** +-5 std devs using train-only statistics to prevent leakage.
- **Validation:** 5 episodes averaged, spread across val data, model selected by Sortino ratio.
- **Ensemble:** PPO, A2C, DQN trained per fold; best by validation Sortino deployed for testing.
- **Anti-passivity measures:** Terminal reward bonus, random init (30% chance starting long), Sortino selection, multi-algo ensemble (DQN epsilon-greedy), turbulence force-sell.
