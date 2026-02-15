"""Statistical analysis of trading results.

Computes bootstrap confidence intervals on Sharpe and returns,
per-trade P&L distribution, and significance tests (t-test, Wilcoxon).

Can run in two modes:
  - Single run: analyze default backtest (test.npz + pre-trained model)
  - Walk-forward aggregate: pool results from all walk-forward folds

Run from tony/: python experiments/statistical_analysis.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats
from stable_baselines3 import DQN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.train import set_seeds
from env.trading_env import TradingEnv
from evaluation.backtest import run_buy_and_hold


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env_from_npz(data_path: str, config: dict, mode: str) -> TradingEnv:
    """Create TradingEnv from .npz file."""
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
        transaction_cost=env_cfg["transaction_cost"],
        max_drawdown_threshold=env_cfg["max_drawdown_threshold"],
        mode=mode,
    )


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


def run_and_collect(model_path: str, env: TradingEnv) -> tuple[list[dict], np.ndarray]:
    """Run DQN agent and return (trade_log, daily_returns)."""
    model = DQN.load(model_path)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    values = np.array(env.portfolio_values, dtype=np.float64)
    daily_returns = np.diff(values) / values[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    return list(env.trade_log), daily_returns


def run_bh_and_collect(env: TradingEnv) -> np.ndarray:
    """Run Buy & Hold and return daily returns."""
    run_buy_and_hold(env)
    values = np.array(env.portfolio_values, dtype=np.float64)
    daily_returns = np.diff(values) / values[:-1]
    return daily_returns[np.isfinite(daily_returns)]


def compute_sharpe(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2 or np.std(daily_returns) == 0:
        return 0.0
    return float((np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252))


def bootstrap_sharpe(daily_returns: np.ndarray, n_bootstrap: int, confidence: float, rng: np.random.Generator):
    """Bootstrap confidence interval for Sharpe ratio."""
    n = len(daily_returns)
    sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(daily_returns, size=n, replace=True)
        std = np.std(sample)
        if std > 0:
            sharpes[i] = (np.mean(sample) / std) * np.sqrt(252)
        else:
            sharpes[i] = 0.0

    alpha = (1 - confidence) / 2
    lo = float(np.percentile(sharpes, alpha * 100))
    hi = float(np.percentile(sharpes, (1 - alpha) * 100))
    return sharpes, lo, hi


def bootstrap_return(daily_returns: np.ndarray, n_bootstrap: int, confidence: float, rng: np.random.Generator):
    """Bootstrap CI for cumulative return."""
    n = len(daily_returns)
    cum_returns = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(daily_returns, size=n, replace=True)
        cum_returns[i] = float(np.prod(1 + sample) - 1)

    alpha = (1 - confidence) / 2
    lo = float(np.percentile(cum_returns, alpha * 100))
    hi = float(np.percentile(cum_returns, (1 - alpha) * 100))
    return lo, hi


def generate_plots(
    trade_pnls: np.ndarray,
    bootstrap_sharpes: np.ndarray,
    sharpe_ci: tuple[float, float],
    dqn_daily: np.ndarray,
    bh_daily: np.ndarray,
    plots_dir: str,
):
    """Generate statistical analysis plots."""
    os.makedirs(plots_dir, exist_ok=True)

    # Per-trade P&L histogram
    if len(trade_pnls) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(trade_pnls * 100, bins=max(10, len(trade_pnls) // 3), edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(trade_pnls) * 100, color="red", linestyle="--", label=f"Mean: {np.mean(trade_pnls)*100:.2f}%")
        ax.axvline(np.median(trade_pnls) * 100, color="orange", linestyle="--", label=f"Median: {np.median(trade_pnls)*100:.2f}%")
        ax.set_xlabel("P&L (%)")
        ax.set_ylabel("Count")
        ax.set_title("Per-Trade P&L Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "trade_pnl_histogram.png"), dpi=150)
        plt.close(fig)

    # Bootstrap Sharpe distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(bootstrap_sharpes, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(sharpe_ci[0], color="red", linestyle="--", label=f"95% CI lower: {sharpe_ci[0]:.3f}")
    ax.axvline(sharpe_ci[1], color="red", linestyle="--", label=f"95% CI upper: {sharpe_ci[1]:.3f}")
    ax.axvline(np.mean(bootstrap_sharpes), color="orange", linestyle="-", label=f"Mean: {np.mean(bootstrap_sharpes):.3f}")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Bootstrap Sharpe Ratio Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "bootstrap_sharpe.png"), dpi=150)
    plt.close(fig)

    # Daily returns comparison
    min_len = min(len(dqn_daily), len(bh_daily))
    if min_len > 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(np.cumsum(dqn_daily[:min_len]) * 100, label="DQN", alpha=0.8)
        ax.plot(np.cumsum(bh_daily[:min_len]) * 100, label="Buy & Hold", alpha=0.8)
        ax.set_xlabel("Day")
        ax.set_ylabel("Cumulative Daily Return (%)")
        ax.set_title("DQN vs Buy & Hold: Cumulative Daily Returns")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "daily_returns_comparison.png"), dpi=150)
        plt.close(fig)

    print(f"  Saved statistical plots -> {plots_dir}")


def collect_walk_forward_results(config: dict):
    """Load walk-forward fold data and run agent on each to pool trade logs and daily returns."""
    folds_dir = "data/walk_forward"
    models_dir = "models/walk_forward"

    if not os.path.isdir(folds_dir):
        return None, None, None, None

    all_trade_logs = []
    all_dqn_daily = []
    all_bh_daily = []

    fold_names = sorted(os.listdir(folds_dir))
    for fold_name in fold_names:
        test_path = os.path.join(folds_dir, fold_name, "test.npz")
        model_path = os.path.join(models_dir, fold_name, "best_model")
        if not os.path.exists(test_path) or not os.path.exists(model_path + ".zip"):
            continue

        env = make_env_from_npz(test_path, config, mode="test")
        trade_log, dqn_daily = run_and_collect(model_path, env)
        all_trade_logs.extend(trade_log)
        all_dqn_daily.append(dqn_daily)

        bh_env = make_env_from_npz(test_path, config, mode="test")
        bh_daily = run_bh_and_collect(bh_env)
        all_bh_daily.append(bh_daily)

    if not all_dqn_daily:
        return None, None, None, None

    return (
        all_trade_logs,
        np.concatenate(all_dqn_daily),
        np.concatenate(all_bh_daily),
        len(fold_names),
    )


def main(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    set_seeds(config["seed"])
    rng = np.random.default_rng(config["seed"])

    stat_cfg = config["experiments"]["statistical"]
    n_bootstrap = stat_cfg.get("n_bootstrap", 10000)
    confidence = stat_cfg.get("confidence_level", 0.95)

    train_cfg = config["training"]
    data_dir = config["data"]["save_dir"]
    plots_dir = "results/statistics"

    # Try walk-forward aggregate first
    print("Checking for walk-forward results...")
    wf_result = collect_walk_forward_results(config)
    wf_trades, wf_dqn_daily, wf_bh_daily, n_folds = wf_result

    if wf_trades is not None and len(wf_trades) > 0:
        print(f"  Found walk-forward data ({n_folds} folds, {len(wf_trades)} trades)")
        trade_log = wf_trades
        dqn_daily = wf_dqn_daily
        bh_daily = wf_bh_daily
        source = f"walk-forward ({n_folds} folds)"
    else:
        print("  No walk-forward data found. Using default backtest...")
        test_path = os.path.join(data_dir, "test.npz")
        model_path = os.path.join(train_cfg["model_save_dir"], train_cfg["best_model_name"])

        if not os.path.exists(test_path):
            print(f"  ERROR: {test_path} not found. Run data/fetch_data.py first.")
            return
        if not os.path.exists(model_path + ".zip"):
            print(f"  ERROR: {model_path}.zip not found. Run agent/train.py first.")
            return

        env = make_env_from_npz(test_path, config, mode="test")
        trade_log, dqn_daily = run_and_collect(model_path, env)

        bh_env = make_env_from_npz(test_path, config, mode="test")
        bh_daily = run_bh_and_collect(bh_env)
        source = "single test period"

    # Compute statistics
    print(f"\n{'=' * 60}")
    print(f"  Statistical Analysis ({source})")
    print(f"{'=' * 60}")

    # Per-trade stats
    trade_pnls = np.array([t["pnl_pct"] for t in trade_log]) if trade_log else np.array([])
    hold_steps = np.array([t["hold_steps"] for t in trade_log]) if trade_log else np.array([])

    n_trades = len(trade_pnls)
    print(f"\n  Trades: {n_trades}")
    if n_trades > 0:
        print(f"  Mean P&L: {np.mean(trade_pnls)*100:+.2f}%")
        print(f"  Median P&L: {np.median(trade_pnls)*100:+.2f}%")
        print(f"  Std P&L: {np.std(trade_pnls)*100:.2f}%")
        if n_trades > 2:
            print(f"  Skew: {float(stats.skew(trade_pnls)):.3f}")
        print(f"  Mean hold: {np.mean(hold_steps):.1f} steps")

    # Sharpe
    sharpe = compute_sharpe(dqn_daily)
    print(f"\n  Sharpe: {sharpe:.3f}")

    # Bootstrap CIs
    if len(dqn_daily) > 10:
        bootstrap_sharpes, sharpe_lo, sharpe_hi = bootstrap_sharpe(
            dqn_daily, n_bootstrap, confidence, rng
        )
        print(f"  Sharpe [{confidence*100:.0f}% CI]: [{sharpe_lo:.3f}, {sharpe_hi:.3f}]")

        ret_lo, ret_hi = bootstrap_return(dqn_daily, n_bootstrap, confidence, rng)
        cum_return = float(np.prod(1 + dqn_daily) - 1)
        print(f"  Return: {cum_return:.2%} [{confidence*100:.0f}% CI: {ret_lo:.2%}, {ret_hi:.2%}]")
    else:
        bootstrap_sharpes = np.array([sharpe])
        sharpe_lo, sharpe_hi = sharpe, sharpe
        print("  (Not enough data for bootstrap)")

    # t-test on per-trade returns
    if n_trades > 2:
        t_stat, p_value = stats.ttest_1samp(trade_pnls, 0.0)
        sig = "*" if p_value < 0.05 else ""
        print(f"\n  t-test (returns > 0): t={t_stat:.3f}, p={p_value:.4f} {sig}")
    else:
        print("\n  t-test: not enough trades")

    # Wilcoxon signed-rank: DQN vs B&H daily returns
    min_len = min(len(dqn_daily), len(bh_daily))
    if min_len > 10:
        diff = dqn_daily[:min_len] - bh_daily[:min_len]
        # Remove zeros for Wilcoxon
        nonzero = diff[diff != 0]
        if len(nonzero) > 10:
            w_stat, w_pvalue = stats.wilcoxon(nonzero)
            sig = "*" if w_pvalue < 0.05 else ""
            print(f"  DQN vs B&H (Wilcoxon): W={w_stat:.0f}, p={w_pvalue:.4f} {sig}")
        else:
            print("  Wilcoxon: not enough non-zero differences")
    else:
        print("  Wilcoxon: not enough paired data")

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(
        trade_pnls,
        bootstrap_sharpes,
        (sharpe_lo, sharpe_hi),
        dqn_daily,
        bh_daily,
        plots_dir,
    )

    print("\nStatistical analysis complete.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config_path)
