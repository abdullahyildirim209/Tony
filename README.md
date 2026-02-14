# Tony: Single-Asset DQN RL Trading Bot

An end-to-end reinforcement learning trading bot that learns to trade a single asset (BTC-USD) using DQN with discrete Buy/Hold/Sell actions.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Fetch and preprocess data
python data/fetch_data.py

# 2. Train the DQN agent
python agent/train.py

# 3. Backtest against baselines
python evaluation/backtest.py
```

## Project Structure

```
tony/
├── configs/default.yaml       # All hyperparameters
├── data/fetch_data.py         # Data pipeline (download, indicators, split)
├── env/trading_env.py         # Custom Gymnasium trading environment
├── agent/train.py             # SB3 DQN training with validation
├── evaluation/backtest.py     # Baselines, metrics, plots
└── notebooks/analysis.ipynb   # Interactive analysis
```

## Pipeline

1. **Data**: Downloads BTC-USD OHLCV from Yahoo Finance, computes SMA/RSI indicators, splits chronologically (60/20/20)
2. **Environment**: Custom Gymnasium env with 35-dim observation (30 pct changes + SMA ratio + RSI + position + PnL), discrete actions (Buy/Hold/Sell), log-return reward
3. **Training**: SB3 DQN with periodic validation, best-model checkpointing, early stopping
4. **Evaluation**: Compares agent against Buy & Hold, Random, and SMA Crossover baselines

## Monitoring

```bash
tensorboard --logdir runs/
```
