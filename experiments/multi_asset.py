"""Multi-asset testing: train and evaluate independent DQN models per asset.

Run from tony/: python experiments/multi_asset.py
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
    split_data,
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


def train_asset(config: dict, train_data: dict, val_data: dict, save_dir: str) -> str:
    """Train DQN for one asset. Returns best model path."""
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

    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=val_callback,
    )

    best_path = os.path.join(save_dir, "best_model")
    if not os.path.exists(best_path + ".zip"):
        model.save(best_path)

    return best_path


def print_cross_asset_table(asset_results: dict):
    """Print cross-asset comparison table."""
    strategies = ["DQN Agent", "Buy & Hold", "Random", "SMA Crossover"]
    print(f"\n{'=' * 100}")
    print("  Multi-Asset Results")
    print(f"{'=' * 100}")

    header = f"{'Asset':<12}"
    for s in strategies:
        header += f" | {s + ' Sharpe':>14} {s + ' Ret':>10}"
    print(header)
    print("-" * len(header))

    for asset, results in asset_results.items():
        row = f"{asset:<12}"
        for s in strategies:
            m = next((r["metrics"] for r in results if r["name"] == s), None)
            if m:
                row += f" | {m['sharpe_ratio']:>14.3f} {m['cumulative_return']:>9.2%}"
            else:
                row += f" | {'N/A':>14} {'N/A':>9}"
        print(row)

    print("=" * len(header))


def generate_plots(asset_results: dict, plots_dir: str):
    """Generate multi-asset bar charts."""
    os.makedirs(plots_dir, exist_ok=True)

    strategies = ["DQN Agent", "Buy & Hold", "Random", "SMA Crossover"]
    assets = list(asset_results.keys())
    n_assets = len(assets)
    n_strats = len(strategies)
    x = np.arange(n_assets)
    width = 0.8 / n_strats

    def get_metric(asset, strategy, metric):
        results = asset_results[asset]
        m = next((r["metrics"] for r in results if r["name"] == strategy), None)
        return m[metric] if m else 0.0

    # Sharpe by asset
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, s in enumerate(strategies):
        vals = [get_metric(a, s, "sharpe_ratio") for a in assets]
        ax.bar(x + i * width, vals, width, label=s)
    ax.set_xticks(x + width * (n_strats - 1) / 2)
    ax.set_xticklabels(assets)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio by Asset")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "sharpe_by_asset.png"), dpi=150)
    plt.close(fig)

    # Return by asset
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, s in enumerate(strategies):
        vals = [get_metric(a, s, "cumulative_return") * 100 for a in assets]
        ax.bar(x + i * width, vals, width, label=s)
    ax.set_xticks(x + width * (n_strats - 1) / 2)
    ax.set_xticklabels(assets)
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Return by Asset")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "return_by_asset.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved multi-asset plots -> {plots_dir}")


def main(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    set_seeds(config["seed"])

    exp_cfg = config["experiments"]["multi_asset"]
    assets = exp_cfg["assets"]
    data_cfg = config["data"]
    ind_cfg = config["indicators"]
    env_cfg = config["env"]

    # Crypto assets get real FNG; non-crypto get neutral 0.5
    crypto_suffixes = ("-USD", "-USDT", "-BTC", "-ETH")
    fng_cache = data_cfg.get("fng_cache_path", "data/fng_cache.csv")

    asset_results = {}

    for asset in assets:
        print(f"\n{'=' * 60}")
        print(f"  Asset: {asset}")
        print(f"{'=' * 60}")

        # Download and process
        print(f"  Downloading {asset}...")
        try:
            df = fetch_ohlcv(asset, data_cfg["start_date"], data_cfg["end_date"])
        except Exception as e:
            print(f"  SKIP: failed to download {asset}: {e}")
            continue

        df = clean_data(df)
        if len(df) < env_cfg["window_size"] * 3:
            print(f"  SKIP: not enough data for {asset} ({len(df)} rows)")
            continue

        # Add FNG: real data for crypto, neutral 0.5 for non-crypto
        is_crypto = any(asset.endswith(s) for s in crypto_suffixes)
        if is_crypto:
            fng_series = fetch_fng(data_cfg["start_date"], data_cfg["end_date"], cache_path=fng_cache)
            df["fng"] = fng_series.reindex(df.index, method="ffill")
            df["fng"] = df["fng"].bfill()
        else:
            df["fng"] = 50.0  # neutral

        df = add_indicators(df, ind_cfg["sma_short"], ind_cfg["sma_long"], ind_cfg["rsi_period"])
        features = compute_features(df, env_cfg["window_size"])

        # Split
        train_split, val_split, test_split = split_data(
            features, features["dates"], data_cfg["train_end"], data_cfg["val_end"]
        )

        print(f"  Train: {len(train_split['close_prices'])} rows, Val: {len(val_split['close_prices'])} rows, Test: {len(test_split['close_prices'])} rows")

        if len(train_split["close_prices"]) < env_cfg["window_size"] + 10:
            print(f"  SKIP: not enough train data for {asset}")
            continue
        if len(test_split["close_prices"]) < env_cfg["window_size"] + 10:
            print(f"  SKIP: not enough test data for {asset}")
            continue

        # Clip
        clip_stats = compute_clip_stats(train_split)
        train_split = apply_clip(train_split, clip_stats)
        val_split = apply_clip(val_split, clip_stats)
        test_split = apply_clip(test_split, clip_stats)

        # Save
        asset_dir = f"data/multi_asset/{asset}"
        os.makedirs(asset_dir, exist_ok=True)
        for split_name, split_data_dict in [("train", train_split), ("val", val_split), ("test", test_split)]:
            np.savez(os.path.join(asset_dir, f"{split_name}.npz"), **split_data_dict)

        # Train
        model_dir = f"models/multi_asset/{asset}"
        print(f"  Training...")
        model_path = train_asset(config, train_split, val_split, model_dir)
        print(f"  Model saved: {model_path}")

        # Backtest
        print("  Running backtest...")
        results = []

        env = make_env_from_dict(test_split, config, mode="test")
        results.append(run_agent(model_path, env))

        env = make_env_from_dict(test_split, config, mode="test")
        results.append(run_buy_and_hold(env))

        env = make_env_from_dict(test_split, config, mode="test")
        results.append(run_random(env, seed=config["seed"]))

        env = make_env_from_dict(test_split, config, mode="test")
        results.append(run_sma_crossover(env))

        asset_results[asset] = results
        print_metrics_table(results)

    # Summary
    if asset_results:
        print_cross_asset_table(asset_results)
        generate_plots(asset_results, "results/multi_asset")

    print("\nMulti-asset testing complete.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config_path)
