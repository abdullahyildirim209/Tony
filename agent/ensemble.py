"""Multi-algorithm ensemble: train PPO, A2C, DQN per fold, select best by validation metric.

Inspired by FinRL's ensemble strategy where multiple algorithms are trained and
the best performer per validation window is selected for deployment.
"""

import os
import sys

import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.train import ValidationCallback
from env.trading_env import TradingEnv

ALGO_MAP = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
}

# Default hyperparameters per algorithm (can be overridden via config)
DEFAULT_KWARGS = {
    "ppo": {
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    "a2c": {
        "n_steps": 128,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "gae_lambda": 0.95,
    },
    "dqn": {
        "batch_size": 64,
        "learning_starts": 1000,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
    },
}


def train_single_algo(
    algo_name: str,
    train_env: TradingEnv,
    val_env: TradingEnv,
    config: dict,
    save_dir: str,
    total_timesteps: int,
) -> tuple[str, float]:
    """Train a single algorithm and return (model_path, best_validation_metric).

    Returns:
        (model_path, best_metric): Path to saved model and best validation score.
    """
    agent_cfg = config.get("agent", {})
    train_cfg = config.get("training", {})
    val_cfg = config.get("validation", {})
    ensemble_cfg = config.get("ensemble", {})

    algo_class = ALGO_MAP[algo_name]
    algo_kwargs = ensemble_cfg.get("algorithm_kwargs", {}).get(algo_name, {})

    # Merge default kwargs with config overrides
    merged_kwargs = {**DEFAULT_KWARGS.get(algo_name, {}), **algo_kwargs}

    # Common kwargs
    common = {
        "learning_rate": agent_cfg.get("learning_rate", 3e-4),
        "gamma": agent_cfg.get("gamma", 0.99),
        "policy_kwargs": {"net_arch": agent_cfg.get("net_arch", [128, 128])},
        "seed": config.get("seed", 42),
        "verbose": 0,
    }

    # DQN doesn't support some PPO/A2C params
    if algo_name == "dqn":
        # DQN-specific: remove unsupported keys
        for k in ["n_steps", "n_epochs", "gae_lambda", "clip_range", "ent_coef", "vf_coef"]:
            merged_kwargs.pop(k, None)
    elif algo_name == "a2c":
        for k in ["n_epochs", "clip_range", "batch_size"]:
            merged_kwargs.pop(k, None)

    model = algo_class("MlpPolicy", train_env, **common, **merged_kwargs)

    os.makedirs(save_dir, exist_ok=True)
    selection_metric = val_cfg.get("selection_metric", "sortino_ratio")

    val_callback = ValidationCallback(
        val_env=val_env,
        validate_every=train_cfg.get("validate_every", 10000),
        patience=train_cfg.get("early_stopping_patience", 5),
        model_save_dir=save_dir,
        best_model_name="best_model",
        n_val_episodes=val_cfg.get("n_val_episodes", 5),
        selection_metric=selection_metric,
        verbose=0,
    )

    model.learn(total_timesteps=total_timesteps, callback=val_callback)

    best_path = os.path.join(save_dir, "best_model")
    if not os.path.exists(best_path + ".zip"):
        model.save(best_path)

    return best_path, val_callback.best_metric


def train_ensemble(
    make_train_env,
    make_val_env,
    config: dict,
    save_dir: str,
    total_timesteps: int,
) -> tuple[str, str]:
    """Train multiple algorithms and select the best by validation metric.

    Args:
        make_train_env: Callable that returns a fresh TradingEnv (train mode).
        make_val_env: Callable that returns a fresh TradingEnv (val mode).
        config: Full config dict.
        save_dir: Base directory for saving models.
        total_timesteps: Training timesteps per algorithm.

    Returns:
        (best_model_path, best_algo_name): Path and name of the winning algorithm.
    """
    ensemble_cfg = config.get("ensemble", {})
    algorithms = ensemble_cfg.get("algorithms", ["ppo"])

    results = {}
    for algo_name in algorithms:
        algo_name = algo_name.lower()
        if algo_name not in ALGO_MAP:
            print(f"  WARNING: Unknown algorithm '{algo_name}', skipping.")
            continue

        print(f"  Training {algo_name.upper()}...")
        algo_dir = os.path.join(save_dir, algo_name)

        # Each algorithm gets fresh envs
        train_env = make_train_env()
        val_env = make_val_env()

        model_path, best_metric = train_single_algo(
            algo_name, train_env, val_env, config, algo_dir, total_timesteps,
        )

        selection_metric = config.get("validation", {}).get("selection_metric", "sortino_ratio")
        results[algo_name] = {
            "model_path": model_path,
            "best_metric": best_metric,
        }
        print(f"    {algo_name.upper()} best {selection_metric}: {best_metric:.3f}")

    if not results:
        raise ValueError("No algorithms trained successfully.")

    # Select best by validation metric
    best_algo = max(results, key=lambda k: results[k]["best_metric"])
    best_path = results[best_algo]["model_path"]

    print(f"  Ensemble winner: {best_algo.upper()} ({results[best_algo]['best_metric']:.3f})")
    return best_path, best_algo
