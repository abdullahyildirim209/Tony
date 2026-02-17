"""Walk-forward testing across multiple market regimes.

Trains and evaluates the PPO agent on expanding windows covering different
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
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.train import set_seeds, ValidationCallback
from agent.ensemble import train_ensemble
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
    keys = [k for k in features.keys() if isinstance(features[k], np.ndarray)]

    def select(data, mask):
        return {k: data[k][mask] for k in keys if k in data}

    # Only use data within the fold's full range
    range_mask = (dates >= train_start) & (dates < test_end)
    ranged = select(features, range_mask)
    d = ranged["dates"]

    train_mask = d < train_end
    val_mask = (d >= train_end) & (d < val_end)
    test_mask = d >= val_end

    return select(ranged, train_mask), select(ranged, val_mask), select(ranged, test_mask)


def make_env_from_dict(data: dict, config: dict, mode: str, turbulence_threshold: float = None) -> TradingEnv:
    """Create TradingEnv from a feature dict."""
    env_cfg = config["env"]
    agent_cfg = config.get("agent", {})
    is_test = mode in ("val", "test")
    return TradingEnv(
        close_prices=data["close_prices"],
        pct_changes=data["pct_changes"],
        sma_ratios=data["sma_ratios"],
        rsi_norm=data["rsi_norm"],
        fng_norm=data["fng_norm"],
        buy_pressure=data.get("buy_pressure"),
        turbulence=data.get("turbulence"),
        window_size=env_cfg["window_size"],
        episode_length=env_cfg["episode_length"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        max_drawdown_threshold=env_cfg["max_drawdown_threshold"],
        gamma=agent_cfg.get("gamma", 0.99),
        terminal_reward_bonus=not is_test and env_cfg.get("terminal_reward_bonus", True),
        random_init=env_cfg.get("random_init", False),
        random_init_long_prob=env_cfg.get("random_init_long_prob", 0.3),
        turbulence_threshold=turbulence_threshold,
        mode=mode,
    )


def compute_turbulence_threshold(train_data: dict, percentile: float = 90) -> float:
    """Compute turbulence threshold from training data."""
    if "turbulence" in train_data:
        return float(np.percentile(train_data["turbulence"], percentile))
    return None


def train_fold(config: dict, train_data: dict, val_data: dict, save_dir: str, total_timesteps: int,
               turbulence_threshold: float = None) -> str:
    """Train PPO for one fold. Returns best model path."""
    agent_cfg = config["agent"]
    train_cfg = config["training"]

    train_env = make_env_from_dict(train_data, config, mode="train", turbulence_threshold=turbulence_threshold)
    val_env = make_env_from_dict(val_data, config, mode="val", turbulence_threshold=turbulence_threshold)

    os.makedirs(save_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=agent_cfg["learning_rate"],
        n_steps=agent_cfg["n_steps"],
        batch_size=agent_cfg["batch_size"],
        n_epochs=agent_cfg["n_epochs"],
        gamma=agent_cfg["gamma"],
        gae_lambda=agent_cfg["gae_lambda"],
        clip_range=agent_cfg["clip_range"],
        ent_coef=agent_cfg["ent_coef"],
        vf_coef=agent_cfg["vf_coef"],
        max_grad_norm=agent_cfg["max_grad_norm"],
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
        selection_metric=val_cfg.get("selection_metric", "sortino_ratio"),
        verbose=0,
    )

    model.learn(total_timesteps=total_timesteps, callback=val_callback)

    best_path = os.path.join(save_dir, "best_model")
    if not os.path.exists(best_path + ".zip"):
        model.save(best_path)
        print("  WARNING: No best model saved during validation. Saving final model as fallback.")

    return best_path


def run_backtest_on_fold(config: dict, test_data: dict, model_path: str,
                         turbulence_threshold: float = None) -> list[dict]:
    """Run PPO + baselines on a test split. Returns list of result dicts."""
    results = []

    # PPO
    env = make_env_from_dict(test_data, config, mode="test", turbulence_threshold=turbulence_threshold)
    results.append(run_agent(model_path, env))

    # Buy & Hold
    env = make_env_from_dict(test_data, config, mode="test", turbulence_threshold=None)
    results.append(run_buy_and_hold(env))

    # Random
    env = make_env_from_dict(test_data, config, mode="test", turbulence_threshold=None)
    results.append(run_random(env, seed=config["seed"]))

    # SMA Crossover
    env = make_env_from_dict(test_data, config, mode="test", turbulence_threshold=None)
    results.append(run_sma_crossover(env))

    return results


def print_regime_table(fold_results: dict):
    """Print per-regime summary across all strategies."""
    strategies = ["PPO Agent", "Buy & Hold", "Random", "SMA Crossover"]
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

    strategies = ["PPO Agent", "Buy & Hold", "Random", "SMA Crossover"]
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


def main(config_path: str = "configs/default.yaml", symbol: str = None):
    config = load_config(config_path)
    set_seeds(config["seed"])

    # Override asset/binance_symbol if symbol specified
    if symbol is not None:
        config["data"]["binance_symbol"] = symbol
        config["data"]["asset"] = symbol

    asset = config["data"]["asset"]

    exp_cfg = config["experiments"]["walk_forward"]
    folds = exp_cfg["folds"]
    total_timesteps = exp_cfg.get("total_timesteps", 300000)
    ind_cfg = config["indicators"]
    env_cfg = config["env"]

    # Download full date range once
    global_start = min(f["train_start"] for f in folds)
    global_end = max(f["test_end"] for f in folds)

    print(f"Downloading {asset} from {global_start} to {global_end}...")
    if config["data"].get("source") == "huggingface":
        from data.fetch_hf import fetch_ohlcv_hf
        df = fetch_ohlcv_hf(config["data"]["binance_symbol"], global_start, global_end, config["data"]["hf_cache_dir"])
    else:
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

        # Compute turbulence threshold from training data
        turb_threshold = compute_turbulence_threshold(
            train_split,
            config.get("env", {}).get("turbulence_threshold_pct", 90),
        )
        if turb_threshold is not None:
            print(f"  Turbulence threshold: {turb_threshold:.4f}")

        # Save fold data (include turbulence threshold)
        data_dir = f"data/walk_forward/{asset}/{name}"
        os.makedirs(data_dir, exist_ok=True)
        train_extra = dict(train_split)
        if turb_threshold is not None:
            train_extra["turbulence_threshold"] = np.float32(turb_threshold)
        for split_name, split_data_dict in [("train", train_extra), ("val", val_split), ("test", test_split)]:
            np.savez(os.path.join(data_dir, f"{split_name}.npz"), **split_data_dict)

        # Train
        model_dir = f"models/walk_forward/{asset}/{name}"
        ensemble_cfg = config.get("ensemble", {})
        ensemble_algos = ensemble_cfg.get("algorithms", [])

        if len(ensemble_algos) >= 1:
            print(f"  Ensemble training ({total_timesteps} timesteps, algos: {ensemble_algos})...")
            model_path, best_algo = train_ensemble(
                make_train_env=lambda: make_env_from_dict(train_split, config, "train", turb_threshold),
                make_val_env=lambda: make_env_from_dict(val_split, config, "val", turb_threshold),
                config=config,
                save_dir=model_dir,
                total_timesteps=total_timesteps,
            )
            # Copy best ensemble model to standard location for compatibility
            import shutil
            canonical_path = os.path.join(model_dir, "best_model")
            shutil.copy2(model_path + ".zip", canonical_path + ".zip")
            model_path = canonical_path
            print(f"  Best algorithm: {best_algo}, model saved: {model_path}")
        else:
            print(f"  Training PPO ({total_timesteps} timesteps)...")
            model_path = train_fold(config, train_split, val_split, model_dir, total_timesteps,
                                    turbulence_threshold=turb_threshold)
            print(f"  Model saved: {model_path}")

        # Backtest
        print("  Running backtest...")
        results = run_backtest_on_fold(config, test_split, model_path, turbulence_threshold=turb_threshold)
        fold_results[name] = results

        # Print fold metrics
        print_metrics_table(results)

    # Summary
    if fold_results:
        print_regime_table(fold_results)
        generate_plots(fold_results, f"results/{asset}/walk_forward")

    print(f"\nWalk-forward testing complete for {asset}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Walk-forward testing")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--symbol", default=None, help="Binance symbol (e.g. ETHUSDT). Overrides config.")
    args = parser.parse_args()
    main(config_path=args.config, symbol=args.symbol)
