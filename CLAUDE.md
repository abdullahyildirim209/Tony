# Tony - Single-Asset DQN Trading Agent

Tony is a DQN-based reinforcement learning agent that trades a single asset (BTC-USD by default) using Stable Baselines3. It downloads OHLCV data, computes technical indicators, trains a DQN agent in a custom Gymnasium environment, and evaluates performance against baselines (Buy & Hold, Random, SMA Crossover).

## Project Structure

```
tony/
  configs/default.yaml   - All hyperparameters and paths
  data/fetch_data.py     - Download, indicators, feature engineering, train/val/test split
  env/trading_env.py     - Gymnasium TradingEnv (discrete Buy/Hold/Sell)
  agent/train.py         - SB3 DQN training with multi-episode validation & early stopping
  evaluation/backtest.py - Run agent + baselines, compute metrics, generate plots
```

## How to Run

All commands run from the `tony/` directory:

```bash
# 1. Fetch data, compute features, split into train/val/test
python data/fetch_data.py

# 2. Train DQN agent (early stops on val Sharpe)
python agent/train.py

# 3. Backtest on test set + baselines
python evaluation/backtest.py
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

## Changelog (from initial implementation)

1. **Fractional shares fix** - BTC is too expensive for whole-unit trading with $10k initial cash. Changed to fractional share trading (go all-in or sell all) so the agent can actually open positions.

2. **Data leakage fix** (`data/fetch_data.py`) - Feature clipping (+-5 std devs) was computed on the full dataset before splitting, leaking val/test statistics into train features. Now: split first, compute clip stats from train only, apply those constants to val/test. Clip stats are saved in `train.npz` for reproducibility.

3. **Multi-episode validation** (`agent/train.py`) - Validation previously ran a single deterministic episode from a fixed start index, causing model selection to overfit to one market path. Now runs `n_val_episodes` (default 5) episodes with start offsets spread evenly across available val data, averages Sharpe/return/drawdown for model selection and early stopping.

4. **Uncapped episode length for val/test** (`env/trading_env.py`) - Val/test episodes were truncated at `episode_length=252` steps, wasting ~80 days of data per split. Now: the 252-step cap only applies in train mode; val/test episodes run until data exhaustion or max drawdown.

5. **Forced start index support** (`env/trading_env.py`) - Added `forced_start_idx` option via `reset(options={"forced_start_idx": N})` to support multi-episode validation with varied start positions.

## Key Architectural Decisions

- **Observation space (35,):** `[30 pct_changes, sma_ratio, rsi_norm, flat_flag, long_flag, unrealized_pnl]`. Window of 30 daily returns gives the agent recent price context without raw price levels (scale-invariant).
- **Action space:** Discrete(3) - Buy (all-in), Hold, Sell (all). Invalid actions (e.g., Buy when already long) are treated as Hold.
- **Reward:** `log(portfolio_value_t / portfolio_value_{t-1}) - 0.001 * trade_executed`. Log returns align with Sharpe maximization; the trade penalty discourages churning.
- **Termination:** Max drawdown >= 50% (terminated) or data exhaustion / episode length cap in train (truncated).
- **Feature clipping:** +-5 std devs using train-only statistics to prevent leakage.
- **Validation:** 5 episodes averaged, spread across val data, to reduce model selection noise.
