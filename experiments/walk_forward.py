"""Walk-forward testing across multiple market regimes.

Trains and evaluates the DQN agent on expanding windows covering different
market conditions (crash, bull, bear, recovery).

Run from tony/: python experiments/walk_forward.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import DQN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.train import set_seeds, ValidationCallback
from data.fetch_data import (
    fetch_ohlcv,
    clean_data,
    fetch_fng,
    add_indicators,
    compute_features,
    compute_clip_stats,
    apply_clip,
)
from env.trading_env import TradingEnv
from evaluation.backtest import (
    run_agent,
    run_buy_and_hold,
    run_random,
    run_sma_crossover,
    print_metrics_table,
)


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def split_by_dates(features: dict, train_start: str, train_end: str, val_end: str, test_end: str):
    """Split features dict by date boundaries, filtering to [train_start, test_end)."""
    dates = features["dates"]
    keys = ["close_prices", "pct_changes", "sma_ratios", "rsi_norm", "fng_norm", "dates"]

    def select(data, mask):
        return {k: data[k][mask] for k in keys}

    # Only use data within the fold's full range
    range_mask = (dates >= train_start) & (dates < test_end)
    ranged = select(features, range_mask)
    d = ranged["dates"]

    train_mask = d < train_end
    val_mask = (d >= train_end) & (d < val_end)
    test_mask = d >= val_end

    return select(ranged, train_mask), select(ranged, val_mask), select(ranged, test_mask)


def make_env_from_dict(data: dict, config: dict, mode: str) -> TradingEnv:
    """Create TradingEnv from a feature dict."""
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
        transaction_cost=env_cfg["transaction_cost"],
        max_drawdown_threshold=env_cfg["max_drawdown_threshold"],
        mode=mode,
    )


def train_fold(config: dict, train_data: dict, val_data: dict, save_dir: str, total_timesteps: int) -> str:
    """Train DQN for one fold. Returns best model path."""
    agent_cfg = config["agent"]
    train_cfg = config["training"]

    train_env = make_env_from_dict(train_data, config, mode="train")
    val_env = make_env_from_dict(val_data, config, mode="val")

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

    model.learn(total_timesteps=total_timesteps, callback=val_callback)

    best_path = os.path.join(save_dir, "best_model")
    if not os.path.exists(best_path + ".zip"):
        model.save(best_path)

    return best_path


def run_backtest_on_fold(config: dict, test_data: dict, model_path: str) -> list[dict]:
    """Run DQN + baselines on a test split. Returns list of result dicts."""
    results = []

    # DQN
    env = make_env_from_dict(test_data, config, mode="test")
    results.append(run_agent(model_path, env))

    # Buy & Hold
    env = make_env_from_dict(test_data, config, mode="test")
    results.append(run_buy_and_hold(env))

    # Random
    env = make_env_from_dict(test_data, config, mode="test")
    results.append(run_random(env, seed=config["seed"]))

    # SMA Crossover
    env = make_env_from_dict(test_data, config, mode="test")
    results.append(run_sma_crossover(env))

    return results


def print_regime_table(fold_results: dict):
    """Print per-regime summary across all strategies."""
    strategies = ["DQN Agent", "Buy & Hold", "Random", "SMA Crossover"]
    print(f"\n{'=' * 100}")
    print("  Walk-Forward Results by Regime")
    print(f"{'=' * 100}")

    header = f"{'Regime':<16}"
    for s in strategies:
        header += f" | {s + ' Sharpe':>14} {s + ' Ret':>10}"
    print(header)
    print("-" * len(header))

    for fold_name, results in fold_results.items():
        row = f"{fold_name:<16}"
        for s in strategies:
            m = next((r["metrics"] for r in results if r["name"] == s), None)
            if m:
                row += f" | {m['sharpe_ratio']:>14.3f} {m['cumulative_return']:>9.2%}"
            else:
                row += f" | {'N/A':>14} {'N/A':>9}"
        print(row)

    print("=" * len(header))

    # Cross-fold aggregates
    print("\n  Cross-Fold Aggregates (mean +/- std):")
    for s in strategies:
        sharpes = []
        returns = []
        for results in fold_results.values():
            m = next((r["metrics"] for r in results if r["name"] == s), None)
            if m:
                sharpes.append(m["sharpe_ratio"])
                returns.append(m["cumulative_return"])
        if sharpes:
            print(
                f"    {s:<16} Sharpe: {np.mean(sharpes):.3f} +/- {np.std(sharpes):.3f}  "
                f"Return: {np.mean(returns):.2%} +/- {np.std(returns):.2%}"
            )


def generate_plots(fold_results: dict, plots_dir: str):
    """Generate walk-forward bar charts."""
    os.makedirs(plots_dir, exist_ok=True)

    strategies = ["DQN Agent", "Buy & Hold", "Random", "SMA Crossover"]
    fold_names = list(fold_results.keys())
    n_folds = len(fold_names)
    n_strats = len(strategies)
    x = np.arange(n_folds)
    width = 0.8 / n_strats

    def get_metric(fold_name, strategy, metric):
        results = fold_results[fold_name]
        m = next((r["metrics"] for r in results if r["name"] == strategy), None)
        return m[metric] if m else 0.0

    # Sharpe by regime
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, s in enumerate(strategies):
        vals = [get_metric(f, s, "sharpe_ratio") for f in fold_names]
        ax.bar(x + i * width, vals, width, label=s)
    ax.set_xticks(x + width * (n_strats - 1) / 2)
    ax.set_xticklabels(fold_names, rotation=30, ha="right")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio by Market Regime")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "sharpe_by_regime.png"), dpi=150)
    plt.close(fig)

    # Return by regime
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, s in enumerate(strategies):
        vals = [get_metric(f, s, "cumulative_return") * 100 for f in fold_names]
        ax.bar(x + i * width, vals, width, label=s)
    ax.set_xticks(x + width * (n_strats - 1) / 2)
    ax.set_xticklabels(fold_names, rotation=30, ha="right")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Return by Market Regime")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "return_by_regime.png"), dpi=150)
    plt.close(fig)

    # Aggregate with error bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, label in [
        (axes[0], "sharpe_ratio", "Sharpe Ratio"),
        (axes[1], "cumulative_return", "Cumulative Return"),
    ]:
        means = []
        stds = []
        for s in strategies:
            vals = [get_metric(f, s, metric) for f in fold_names]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        sx = np.arange(n_strats)
        multiplier = 100 if metric == "cumulative_return" else 1
        ax.bar(sx, [m * multiplier for m in means], yerr=[s * multiplier for s in stds], capsize=5)
        ax.set_xticks(sx)
        ax.set_xticklabels(strategies, rotation=30, ha="right")
        ax.set_ylabel(label + (" (%)" if metric == "cumulative_return" else ""))
        ax.set_title(f"Aggregate {label} (mean +/- std)")
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "aggregate_summary.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved walk-forward plots -> {plots_dir}")


def main(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    set_seeds(config["seed"])

    exp_cfg = config["experiments"]["walk_forward"]
    folds = exp_cfg["folds"]
    total_timesteps = exp_cfg.get("total_timesteps", 300000)
    ind_cfg = config["indicators"]
    env_cfg = config["env"]

    # Download full date range once
    global_start = min(f["train_start"] for f in folds)
    global_end = max(f["test_end"] for f in folds)
    asset = config["data"]["asset"]

    print(f"Downloading {asset} from {global_start} to {global_end}...")
    df = fetch_ohlcv(asset, global_start, global_end)
    df = clean_data(df)
    fng_cache = config["data"].get("fng_cache_path", "data/fng_cache.csv")
    fng_series = fetch_fng(global_start, global_end, cache_path=fng_cache)
    df["fng"] = fng_series.reindex(df.index, method="ffill")
    df["fng"] = df["fng"].bfill()
    df = add_indicators(df, ind_cfg["sma_short"], ind_cfg["sma_long"], ind_cfg["rsi_period"])
    features = compute_features(df, env_cfg["window_size"])
    print(f"  Full dataset: {len(features['close_prices'])} rows")

    fold_results = {}

    for fold in folds:
        name = fold["name"]
        print(f"\n{'=' * 60}")
        print(f"  Fold: {name}")
        print(f"  Train: {fold['train_start']} - {fold['train_end']}")
        print(f"  Val:   {fold['train_end']} - {fold['val_end']}")
        print(f"  Test:  {fold['val_end']} - {fold['test_end']}")
        print(f"{'=' * 60}")

        # Split and clip
        train_split, val_split, test_split = split_by_dates(
            features,
            fold["train_start"],
            fold["train_end"],
            fold["val_end"],
            fold["test_end"],
        )

        print(f"  Train: {len(train_split['close_prices'])} rows, Val: {len(val_split['close_prices'])} rows, Test: {len(test_split['close_prices'])} rows")

        if len(train_split["close_prices"]) < env_cfg["window_size"] + 10:
            print(f"  SKIP: not enough train data for fold {name}")
            continue
        if len(test_split["close_prices"]) < env_cfg["window_size"] + 10:
            print(f"  SKIP: not enough test data for fold {name}")
            continue

        clip_stats = compute_clip_stats(train_split)
        train_split = apply_clip(train_split, clip_stats)
        val_split = apply_clip(val_split, clip_stats)
        test_split = apply_clip(test_split, clip_stats)

        # Save fold data
        data_dir = f"data/walk_forward/{name}"
        os.makedirs(data_dir, exist_ok=True)
        for split_name, split_data in [("train", train_split), ("val", val_split), ("test", test_split)]:
            np.savez(os.path.join(data_dir, f"{split_name}.npz"), **split_data)

        # Train
        model_dir = f"models/walk_forward/{name}"
        print(f"  Training ({total_timesteps} timesteps)...")
        model_path = train_fold(config, train_split, val_split, model_dir, total_timesteps)
        print(f"  Model saved: {model_path}")

        # Backtest
        print("  Running backtest...")
        results = run_backtest_on_fold(config, test_split, model_path)
        fold_results[name] = results

        # Print fold metrics
        print_metrics_table(results)

    # Summary
    if fold_results:
        print_regime_table(fold_results)
        generate_plots(fold_results, "results/walk_forward")

    print("\nWalk-forward testing complete.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config_path)
