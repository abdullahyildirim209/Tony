"""Transaction cost sensitivity analysis.

Evaluates how varying transaction costs affect DQN performance vs baselines.
Phase A: evaluate pre-trained model at each TC level.
Phase B: retrain a new model at each TC level, then evaluate.

Run from tony/: python experiments/tc_sensitivity.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.train import set_seeds, ValidationCallback
from env.trading_env import TradingEnv
from evaluation.backtest import run_agent, run_buy_and_hold, run_sma_crossover


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env_from_npz(data_path: str, config: dict, mode: str, tc: float) -> TradingEnv:
    """Create TradingEnv from .npz with a custom transaction cost."""
    data = np.load(data_path, allow_pickle=True)
    env_cfg = config["env"]
    return TradingEnv(
        close_prices=data["close_prices"],
        pct_changes=data["pct_changes"],
        sma_ratios=data["sma_ratios"],
        rsi_norm=data["rsi_norm"],
        fng_norm=data["fng_norm"],
        window_size=env_cfg["window_size"],
        episode_length=env_cfg["episode_length"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=tc,
        max_drawdown_threshold=env_cfg["max_drawdown_threshold"],
        mode=mode,
    )


def run_dqn_eval(model_path: str, env: TradingEnv) -> dict:
    """Run pre-trained DQN deterministically and return metrics."""
    model = DQN.load(model_path)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    return env.get_episode_metrics()


def train_dqn_at_tc(config: dict, tc: float, save_dir: str) -> str:
    """Train a new DQN model with a specific transaction cost. Returns model path."""
    agent_cfg = config["agent"]
    train_cfg = config["training"]
    data_dir = config["data"]["save_dir"]

    train_env = make_env_from_npz(
        os.path.join(data_dir, "train.npz"), config, mode="train", tc=tc
    )
    val_env = make_env_from_npz(
        os.path.join(data_dir, "val.npz"), config, mode="val", tc=tc
    )

    os.makedirs(save_dir, exist_ok=True)

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=agent_cfg["learning_rate"],
        buffer_size=agent_cfg["buffer_size"],
        learning_starts=agent_cfg["learning_starts"],
        batch_size=agent_cfg["batch_size"],
        gamma=agent_cfg["gamma"],
        target_update_interval=agent_cfg["target_update_interval"],
        exploration_initial_eps=agent_cfg["exploration_initial_eps"],
        exploration_final_eps=agent_cfg["exploration_final_eps"],
        exploration_fraction=agent_cfg["exploration_fraction"],
        policy_kwargs={"net_arch": agent_cfg["net_arch"]},
        seed=config["seed"],
        verbose=0,
    )

    val_cfg = config.get("validation", {})
    val_callback = ValidationCallback(
        val_env=val_env,
        validate_every=train_cfg["validate_every"],
        patience=train_cfg["early_stopping_patience"],
        model_save_dir=save_dir,
        best_model_name="best_model",
        n_val_episodes=val_cfg.get("n_val_episodes", 5),
        verbose=0,
    )

    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=val_callback,
    )

    best_path = os.path.join(save_dir, "best_model")
    if not os.path.exists(best_path + ".zip"):
        model.save(best_path)

    return best_path


def print_table(title: str, tc_levels: list, rows: dict):
    """Print a formatted table of metrics across TC levels."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    header = f"{'TC':>8} | {'DQN Sharpe':>11} {'DQN Ret':>9} {'DQN Trades':>11} | {'B&H Sharpe':>11} {'B&H Ret':>9} | {'SMA Sharpe':>11} {'SMA Ret':>9}"
    print(header)
    print("-" * len(header))
    for tc in tc_levels:
        d = rows[tc]["dqn"]
        b = rows[tc]["bh"]
        s = rows[tc]["sma"]
        print(
            f"{tc:>8.4f} | "
            f"{d['sharpe_ratio']:>11.3f} {d['cumulative_return']:>8.2%} {d['trade_count']:>11d} | "
            f"{b['sharpe_ratio']:>11.3f} {b['cumulative_return']:>8.2%} | "
            f"{s['sharpe_ratio']:>11.3f} {s['cumulative_return']:>8.2%}"
        )
    print("=" * len(header))


def generate_plots(
    tc_levels: list,
    eval_rows: dict,
    retrain_rows,
    plots_dir: str,
):
    """Generate TC sensitivity plots."""
    os.makedirs(plots_dir, exist_ok=True)

    tcs = np.array(tc_levels)

    # Extract metrics
    def extract(rows, strategy, metric):
        return np.array([rows[tc][strategy][metric] for tc in tc_levels])

    # Sharpe vs TC
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tcs, extract(eval_rows, "dqn", "sharpe_ratio"), "o-", label="DQN (fixed model)")
    if retrain_rows:
        ax.plot(tcs, extract(retrain_rows, "dqn", "sharpe_ratio"), "s--", label="DQN (retrained)")
    ax.plot(tcs, extract(eval_rows, "bh", "sharpe_ratio"), "^-", label="Buy & Hold")
    ax.plot(tcs, extract(eval_rows, "sma", "sharpe_ratio"), "d-", label="SMA Crossover")
    ax.set_xlabel("Transaction Cost")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio vs Transaction Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "sharpe_vs_tc.png"), dpi=150)
    plt.close(fig)

    # Return vs TC
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tcs, extract(eval_rows, "dqn", "cumulative_return") * 100, "o-", label="DQN (fixed)")
    if retrain_rows:
        ax.plot(tcs, extract(retrain_rows, "dqn", "cumulative_return") * 100, "s--", label="DQN (retrained)")
    ax.plot(tcs, extract(eval_rows, "bh", "cumulative_return") * 100, "^-", label="Buy & Hold")
    ax.plot(tcs, extract(eval_rows, "sma", "cumulative_return") * 100, "d-", label="SMA Crossover")
    ax.set_xlabel("Transaction Cost")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Return vs Transaction Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "return_vs_tc.png"), dpi=150)
    plt.close(fig)

    # Trade count vs TC
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tcs, extract(eval_rows, "dqn", "trade_count"), "o-", label="DQN (fixed)")
    if retrain_rows:
        ax.plot(tcs, extract(retrain_rows, "dqn", "trade_count"), "s--", label="DQN (retrained)")
    ax.plot(tcs, extract(eval_rows, "sma", "trade_count"), "d-", label="SMA Crossover")
    ax.set_xlabel("Transaction Cost")
    ax.set_ylabel("Trade Count")
    ax.set_title("Trade Count vs Transaction Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "trades_vs_tc.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved TC sensitivity plots -> {plots_dir}")


def main(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    set_seeds(config["seed"])

    exp_cfg = config["experiments"]["tc_sensitivity"]
    tc_levels = exp_cfg["tc_levels"]
    do_retrain = exp_cfg.get("retrain", False)

    train_cfg = config["training"]
    data_dir = config["data"]["save_dir"]
    model_path = os.path.join(train_cfg["model_save_dir"], train_cfg["best_model_name"])

    plots_dir = "results/tc_sensitivity"

    # Phase A: Evaluate pre-trained model at each TC
    print("Phase A: Evaluating pre-trained model at varying TC levels...")
    eval_rows = {}
    for tc in tc_levels:
        print(f"  TC = {tc:.4f}")
        test_env = make_env_from_npz(os.path.join(data_dir, "test.npz"), config, "test", tc)
        dqn_metrics = run_dqn_eval(model_path, test_env)

        bh_env = make_env_from_npz(os.path.join(data_dir, "test.npz"), config, "test", tc)
        run_buy_and_hold(bh_env)
        bh_metrics = bh_env.get_episode_metrics()

        sma_env = make_env_from_npz(os.path.join(data_dir, "test.npz"), config, "test", tc)
        run_sma_crossover(sma_env)
        sma_metrics = sma_env.get_episode_metrics()

        eval_rows[tc] = {"dqn": dqn_metrics, "bh": bh_metrics, "sma": sma_metrics}

    print_table("Phase A: Pre-trained Model at Varying TC", tc_levels, eval_rows)

    # Phase B: Retrain at each TC
    retrain_rows = None
    if do_retrain:
        print("\nPhase B: Retraining at each TC level...")
        retrain_rows = {}
        for tc in tc_levels:
            print(f"  TC = {tc:.4f} — training...")
            tc_str = f"{tc:.4f}".replace(".", "_")
            save_dir = f"models/tc_sensitivity/tc_{tc_str}"

            retrained_path = train_dqn_at_tc(config, tc, save_dir)

            test_env = make_env_from_npz(os.path.join(data_dir, "test.npz"), config, "test", tc)
            dqn_metrics = run_dqn_eval(retrained_path, test_env)

            bh_env = make_env_from_npz(os.path.join(data_dir, "test.npz"), config, "test", tc)
            run_buy_and_hold(bh_env)
            bh_metrics = bh_env.get_episode_metrics()

            sma_env = make_env_from_npz(os.path.join(data_dir, "test.npz"), config, "test", tc)
            run_sma_crossover(sma_env)
            sma_metrics = sma_env.get_episode_metrics()

            retrain_rows[tc] = {"dqn": dqn_metrics, "bh": bh_metrics, "sma": sma_metrics}

        print_table("Phase B: Retrained Models at Varying TC", tc_levels, retrain_rows)

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(tc_levels, eval_rows, retrain_rows, plots_dir)

    print("TC sensitivity analysis complete.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config_path)
