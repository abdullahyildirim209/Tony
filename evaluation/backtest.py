"""Backtesting: run agent & baselines through env, compute metrics, generate plots."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import DQN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.trading_env import TradingEnv


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(data_path: str, config: dict) -> TradingEnv:
    data = np.load(data_path, allow_pickle=True)
    env_cfg = config["env"]
    return TradingEnv(
        close_prices=data["close_prices"],
        pct_changes=data["pct_changes"],
        sma_ratios=data["sma_ratios"],
        rsi_norm=data["rsi_norm"],
        fng_norm=data["fng_norm"] if "fng_norm" in data else None,
        buy_pressure=data["buy_pressure"] if "buy_pressure" in data else None,
        window_size=env_cfg["window_size"],
        episode_length=env_cfg["episode_length"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        max_drawdown_threshold=env_cfg["max_drawdown_threshold"],
        mode="test",
    )


# --- Strategy runners ---


def run_agent(model_path: str, env: TradingEnv) -> dict:
    """Run trained DQN agent deterministically."""
    model = DQN.load(model_path)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    return {
        "name": "DQN Agent",
        "portfolio_values": list(env.portfolio_values),
        "actions": list(env.actions_taken),
        "metrics": env.get_episode_metrics(),
    }


def run_buy_and_hold(env: TradingEnv) -> dict:
    """Buy on step 1, Hold forever."""
    obs, _ = env.reset()
    done = False
    first_step = True
    while not done:
        action = 0 if first_step else 1  # Buy then Hold
        first_step = False
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return {
        "name": "Buy & Hold",
        "portfolio_values": list(env.portfolio_values),
        "actions": list(env.actions_taken),
        "metrics": env.get_episode_metrics(),
    }


def run_random(env: TradingEnv, seed: int = 42) -> dict:
    """Uniform random actions."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    done = False
    while not done:
        action = int(rng.integers(0, 3))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return {
        "name": "Random",
        "portfolio_values": list(env.portfolio_values),
        "actions": list(env.actions_taken),
        "metrics": env.get_episode_metrics(),
    }


def run_sma_crossover(env: TradingEnv) -> dict:
    """Buy when SMA ratio > 1.0, Sell when < 1.0."""
    obs, _ = env.reset()
    done = False
    while not done:
        sma_ratio = obs[30]
        if sma_ratio > 1.0:
            action = 0  # Buy
        elif sma_ratio < 1.0:
            action = 2  # Sell
        else:
            action = 1  # Hold
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return {
        "name": "SMA Crossover",
        "portfolio_values": list(env.portfolio_values),
        "actions": list(env.actions_taken),
        "metrics": env.get_episode_metrics(),
    }


# --- Plotting ---


def plot_portfolio_curves(results: list[dict], save_path: str):
    """Plot portfolio value curves for all strategies."""
    plt.figure(figsize=(14, 6))
    for r in results:
        plt.plot(r["portfolio_values"], label=r["name"], alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Portfolio Value Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved portfolio curves -> {save_path}")


def plot_price_with_trades(
    close_prices: np.ndarray,
    actions: list[int],
    start_idx: int,
    save_path: str,
):
    """Plot price chart with buy/sell markers for DQN agent."""
    plt.figure(figsize=(14, 6))

    n = len(actions)
    prices = close_prices[start_idx : start_idx + n + 1]
    steps = np.arange(len(prices))
    plt.plot(steps, prices, color="gray", alpha=0.6, label="Price")

    buy_steps = [i + 1 for i, a in enumerate(actions) if a == 0]
    sell_steps = [i + 1 for i, a in enumerate(actions) if a == 2]

    if buy_steps:
        plt.scatter(
            buy_steps,
            prices[buy_steps],
            marker="^",
            color="green",
            s=60,
            label="Buy",
            zorder=3,
        )
    if sell_steps:
        plt.scatter(
            sell_steps,
            prices[sell_steps],
            marker="v",
            color="red",
            s=60,
            label="Sell",
            zorder=3,
        )

    plt.xlabel("Step")
    plt.ylabel("Price ($)")
    plt.title("DQN Agent Trades on Price Chart")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved trade chart -> {save_path}")


def plot_drawdown_curves(results: list[dict], save_path: str):
    """Plot drawdown curves for all strategies."""
    plt.figure(figsize=(14, 4))
    for r in results:
        values = np.array(r["portfolio_values"])
        running_max = np.maximum.accumulate(values)
        drawdown = (running_max - values) / running_max
        plt.plot(drawdown, label=r["name"], alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Drawdown")
    plt.title("Drawdown Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved drawdown curves -> {save_path}")


def print_metrics_table(results: list[dict]):
    """Print formatted comparison table."""
    header = f"{'Strategy':<16} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["metrics"]
        print(
            f"{r['name']:<16} "
            f"{m['cumulative_return']:>9.2%} "
            f"{m['sharpe_ratio']:>8.3f} "
            f"{m['max_drawdown']:>7.2%} "
            f"{m['trade_count']:>7d} "
            f"{m['win_rate']:>7.1%}"
        )
    print("=" * len(header) + "\n")


# --- Main ---


def backtest(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    eval_cfg = config["evaluation"]
    train_cfg = config["training"]

    test_data_path = os.path.join(config["data"]["save_dir"], "test.npz")
    model_path = os.path.join(train_cfg["model_save_dir"], train_cfg["best_model_name"])

    os.makedirs(eval_cfg["plots_dir"], exist_ok=True)

    results = []

    # Run DQN agent
    print("Running DQN Agent...")
    env = make_env(test_data_path, config)
    agent_result = run_agent(model_path, env)
    results.append(agent_result)
    agent_start_idx = env.start_idx

    # Run baselines
    for baseline in eval_cfg["baselines"]:
        print(f"Running {baseline}...")
        env = make_env(test_data_path, config)
        if baseline == "buy_and_hold":
            results.append(run_buy_and_hold(env))
        elif baseline == "random":
            results.append(run_random(env, seed=config["seed"]))
        elif baseline == "sma_crossover":
            results.append(run_sma_crossover(env))

    # Metrics table
    print_metrics_table(results)

    # Plots
    print("Generating plots...")
    plot_portfolio_curves(
        results, os.path.join(eval_cfg["plots_dir"], "portfolio_curves.png")
    )

    test_data = np.load(test_data_path)
    plot_price_with_trades(
        test_data["close_prices"],
        agent_result["actions"],
        agent_start_idx,
        os.path.join(eval_cfg["plots_dir"], "agent_trades.png"),
    )

    plot_drawdown_curves(
        results, os.path.join(eval_cfg["plots_dir"], "drawdown_curves.png")
    )

    print("Backtesting complete.")
    return results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    backtest(config_path)
