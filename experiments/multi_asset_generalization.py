"""Multi-asset generalization test: evaluate BTC-trained DQN on ETH-USD and SOL-USD.

Tests whether learned patterns from BTC-USD transfer to other crypto assets
without retraining. Uses BTC clip stats and turbulence thresholds for all assets
(model expects BTC-scale inputs).

Run from tony/: python experiments/multi_asset_generalization.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fetch_data import (
    fetch_ohlcv,
    clean_data,
    fetch_fng,
    add_indicators,
    compute_features,
    compute_clip_stats,
    apply_clip,
)
from experiments.walk_forward import (
    split_by_dates,
    make_env_from_dict,
    compute_turbulence_threshold,
    run_backtest_on_fold,
)
from evaluation.backtest import print_metrics_table


ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]
DATE_RANGE = ("2019-01-01", "2025-01-01")
RESULTS_DIR = "results/multi_asset_generalization"


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_all_assets(config: dict) -> dict:
    """Download and compute features for all assets. Returns {asset: features_dict}."""
    ind_cfg = config["indicators"]
    env_cfg = config["env"]
    start, end = DATE_RANGE

    fng_cache = config["data"].get("fng_cache_path", "data/fng_cache.csv")
    fng_series = fetch_fng(start, end, cache_path=fng_cache)

    all_features = {}
    for asset in ASSETS:
        print(f"\nDownloading {asset} ({start} to {end})...")
        df = fetch_ohlcv(asset, start, end)
        df = clean_data(df)
        print(f"  Raw data: {len(df)} rows")

        if len(df) < env_cfg["window_size"] + 50:
            print(f"  SKIP {asset}: insufficient data ({len(df)} rows)")
            continue

        # Same FNG for all crypto assets
        df["fng"] = fng_series.reindex(df.index, method="ffill")
        df["fng"] = df["fng"].bfill()

        df = add_indicators(df, ind_cfg["sma_short"], ind_cfg["sma_long"], ind_cfg["rsi_period"])
        features = compute_features(df, env_cfg["window_size"])
        print(f"  Features: {len(features['close_prices'])} rows, date range: {features['dates'][0]} to {features['dates'][-1]}")
        all_features[asset] = features

    return all_features


def run_generalization_test(config: dict, all_features: dict) -> dict:
    """Run all folds for all assets. Returns nested dict of results."""
    folds = config["experiments"]["walk_forward"]["folds"]
    env_cfg = config["env"]

    all_results = {}  # {fold_name: {asset: [result_dicts]}}

    for fold in folds:
        name = fold["name"]
        print(f"\n{'=' * 70}")
        print(f"  {name}: train {fold['train_start']}-{fold['train_end']}, "
              f"val {fold['train_end']}-{fold['val_end']}, test {fold['val_end']}-{fold['test_end']}")
        print(f"{'=' * 70}")

        # Check model exists
        model_path = f"models/walk_forward/{name}/best_model"
        if not os.path.exists(model_path + ".zip"):
            print(f"  SKIP: no model at {model_path}.zip")
            continue

        # Split BTC to get clip stats and turbulence threshold
        if "BTC-USD" not in all_features:
            print("  SKIP: BTC-USD features not available")
            continue

        btc_train, _, _ = split_by_dates(
            all_features["BTC-USD"],
            fold["train_start"], fold["train_end"], fold["val_end"], fold["test_end"],
        )

        if len(btc_train["close_prices"]) < env_cfg["window_size"] + 10:
            print(f"  SKIP: insufficient BTC train data for {name}")
            continue

        clip_stats = compute_clip_stats(btc_train)
        turb_threshold = compute_turbulence_threshold(
            btc_train,
            config.get("env", {}).get("turbulence_threshold_pct", 90),
        )
        print(f"  BTC clip stats computed from {len(btc_train['close_prices'])} train rows")
        if turb_threshold is not None:
            print(f"  Turbulence threshold: {turb_threshold:.4f}")

        fold_results = {}

        for asset in ASSETS:
            if asset not in all_features:
                print(f"  {asset}: skipped (no data)")
                continue

            # Split by same fold dates
            train, val, test = split_by_dates(
                all_features[asset],
                fold["train_start"], fold["train_end"], fold["val_end"], fold["test_end"],
            )

            if len(test["close_prices"]) < env_cfg["window_size"] + 10:
                print(f"  {asset}: skipped (only {len(test['close_prices'])} test rows, need {env_cfg['window_size'] + 10})")
                continue

            # Apply BTC clip stats to this asset
            test_clipped = apply_clip(test, clip_stats)

            # Run backtest
            print(f"\n  --- {asset} on {name} ({len(test_clipped['close_prices'])} test rows) ---")
            results = run_backtest_on_fold(config, test_clipped, model_path, turbulence_threshold=turb_threshold)
            fold_results[asset] = results
            print_metrics_table(results)

        all_results[name] = fold_results

    return all_results


def print_comparison_table(all_results: dict):
    """Print per-fold and cross-fold comparison table."""
    print(f"\n{'=' * 110}")
    print("  Multi-Asset Generalization Results")
    print(f"{'=' * 110}")

    header = f"{'Asset':<10} {'Fold':<8} {'Return':>10} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8}"
    print(header)
    print("-" * len(header))

    # Collect per-asset stats for cross-fold averages
    asset_sharpes = {a: [] for a in ASSETS}
    asset_returns = {a: [] for a in ASSETS}
    asset_trades = {a: [] for a in ASSETS}

    for fold_name, fold_results in all_results.items():
        for asset in ASSETS:
            if asset not in fold_results:
                continue
            # Get PPO Agent (first result)
            agent_result = next((r for r in fold_results[asset] if r["name"] == "PPO Agent"), None)
            if agent_result is None:
                continue
            m = agent_result["metrics"]
            print(
                f"{asset:<10} {fold_name:<8} "
                f"{m['cumulative_return']:>9.2%} "
                f"{m['sharpe_ratio']:>8.3f} "
                f"{m.get('sortino_ratio', 0):>8.3f} "
                f"{m['max_drawdown']:>7.2%} "
                f"{m['trade_count']:>7d} "
                f"{m['win_rate']:>7.1%}"
            )
            asset_sharpes[asset].append(m["sharpe_ratio"])
            asset_returns[asset].append(m["cumulative_return"])
            asset_trades[asset].append(m["trade_count"])
        # Separator between folds
        print()

    # Cross-fold averages
    print(f"{'=' * 110}")
    print("  Cross-Fold Averages (DQN Agent only)")
    print(f"{'=' * 110}")
    avg_header = f"{'Asset':<10} {'Folds':>6} {'Avg Return':>12} {'Avg Sharpe':>12} {'Avg Trades':>12} {'Std Sharpe':>12}"
    print(avg_header)
    print("-" * len(avg_header))

    for asset in ASSETS:
        if not asset_sharpes[asset]:
            continue
        n = len(asset_sharpes[asset])
        print(
            f"{asset:<10} {n:>6d} "
            f"{np.mean(asset_returns[asset]):>11.2%} "
            f"{np.mean(asset_sharpes[asset]):>12.3f} "
            f"{np.mean(asset_trades[asset]):>12.1f} "
            f"{np.std(asset_sharpes[asset]):>12.3f}"
        )
    print(f"{'=' * 110}")


def save_results(all_results: dict, output_dir: str):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for fold_name, fold_results in all_results.items():
        serializable[fold_name] = {}
        for asset, results in fold_results.items():
            serializable[fold_name][asset] = []
            for r in results:
                serializable[fold_name][asset].append({
                    "name": r["name"],
                    "metrics": r["metrics"],
                })

    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Saved results -> {path}")


def generate_plots(all_results: dict, output_dir: str):
    """Generate comparison bar charts."""
    os.makedirs(output_dir, exist_ok=True)

    fold_names = list(all_results.keys())
    assets_present = [a for a in ASSETS if any(a in all_results[f] for f in fold_names)]

    if not assets_present:
        print("  No results to plot.")
        return

    # Sharpe by fold per asset
    fig, ax = plt.subplots(figsize=(12, 6))
    n_folds = len(fold_names)
    n_assets = len(assets_present)
    x = np.arange(n_folds)
    width = 0.8 / n_assets
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, asset in enumerate(assets_present):
        sharpes = []
        for fold_name in fold_names:
            if asset in all_results[fold_name]:
                agent = next((r for r in all_results[fold_name][asset] if r["name"] == "PPO Agent"), None)
                sharpes.append(agent["metrics"]["sharpe_ratio"] if agent else 0)
            else:
                sharpes.append(0)
        ax.bar(x + i * width, sharpes, width, label=asset, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (n_assets - 1) / 2)
    ax.set_xticklabels(fold_names, rotation=30, ha="right")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("DQN Agent Sharpe by Fold — Multi-Asset Generalization")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sharpe_by_fold.png"), dpi=150)
    plt.close(fig)

    # Returns by fold per asset
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, asset in enumerate(assets_present):
        returns = []
        for fold_name in fold_names:
            if asset in all_results[fold_name]:
                agent = next((r for r in all_results[fold_name][asset] if r["name"] == "PPO Agent"), None)
                returns.append(agent["metrics"]["cumulative_return"] * 100 if agent else 0)
            else:
                returns.append(0)
        ax.bar(x + i * width, returns, width, label=asset, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (n_assets - 1) / 2)
    ax.set_xticklabels(fold_names, rotation=30, ha="right")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("DQN Agent Return by Fold — Multi-Asset Generalization")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "return_by_fold.png"), dpi=150)
    plt.close(fig)

    # Aggregate comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, label in [
        (axes[0], "sharpe_ratio", "Sharpe Ratio"),
        (axes[1], "cumulative_return", "Cumulative Return"),
    ]:
        means = []
        stds = []
        for asset in assets_present:
            vals = []
            for fold_name in fold_names:
                if asset in all_results[fold_name]:
                    agent = next((r for r in all_results[fold_name][asset] if r["name"] == "PPO Agent"), None)
                    if agent:
                        vals.append(agent["metrics"][metric])
            multiplier = 100 if metric == "cumulative_return" else 1
            means.append(np.mean(vals) * multiplier if vals else 0)
            stds.append(np.std(vals) * multiplier if vals else 0)

        sx = np.arange(len(assets_present))
        ax.bar(sx, means, yerr=stds, capsize=5, color=colors[:len(assets_present)])
        ax.set_xticks(sx)
        ax.set_xticklabels(assets_present)
        ax.set_ylabel(label + (" (%)" if metric == "cumulative_return" else ""))
        ax.set_title(f"Aggregate {label} (mean +/- std)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "aggregate_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved plots -> {output_dir}")


def main(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)

    print("=" * 70)
    print("  Multi-Asset Generalization Test")
    print(f"  Assets: {', '.join(ASSETS)}")
    print(f"  Date range: {DATE_RANGE[0]} to {DATE_RANGE[1]}")
    print(f"  Models: BTC-trained DQN from walk-forward folds")
    print("=" * 70)

    # Step 1: Download all assets
    all_features = download_all_assets(config)

    if len(all_features) == 0:
        print("ERROR: No asset data downloaded.")
        return

    # Step 2: Run generalization test
    all_results = run_generalization_test(config, all_features)

    if not all_results:
        print("ERROR: No results produced.")
        return

    # Step 3: Print comparison table
    print_comparison_table(all_results)

    # Step 4: Save results and plots
    save_results(all_results, RESULTS_DIR)
    generate_plots(all_results, RESULTS_DIR)

    print("\nMulti-asset generalization test complete.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config_path)
