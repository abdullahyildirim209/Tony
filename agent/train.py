"""SB3 PPO training with validation callback and early stopping."""

import os
import random
import sys
from functools import partial

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.trading_env import TradingEnv


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(data_path: str, config: dict, mode: str) -> TradingEnv:
    """Load .npz data and create TradingEnv."""
    data = np.load(data_path, allow_pickle=True)
    env_cfg = config["env"]
    agent_cfg = config.get("agent", {})

    # Load turbulence threshold from train.npz if available
    turb_threshold = None
    if "turbulence_threshold" in data:
        turb_threshold = float(data["turbulence_threshold"])

    return TradingEnv(
        close_prices=data["close_prices"],
        pct_changes=data["pct_changes"],
        sma_ratios=data["sma_ratios"],
        rsi_norm=data["rsi_norm"],
        fng_norm=data["fng_norm"] if "fng_norm" in data else None,
        buy_pressure=data["buy_pressure"] if "buy_pressure" in data else None,
        turbulence=data["turbulence"] if "turbulence" in data else None,
        window_size=env_cfg["window_size"],
        episode_length=env_cfg["episode_length"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        max_drawdown_threshold=env_cfg["max_drawdown_threshold"],
        gamma=agent_cfg.get("gamma", 0.99),
        terminal_reward_bonus=env_cfg.get("terminal_reward_bonus", True),
        random_init=env_cfg.get("random_init", False),
        random_init_long_prob=env_cfg.get("random_init_long_prob", 0.3),
        turbulence_threshold=turb_threshold,
        mode=mode,
    )


class ValidationCallback(BaseCallback):
    """Periodically evaluate on validation set, save best model, early stop.

    Runs multiple episodes with varied start offsets and averages metrics
    to reduce noise in model selection.
    """

    def __init__(
        self,
        val_env: TradingEnv,
        validate_every: int,
        patience: int,
        model_save_dir: str,
        best_model_name: str,
        n_val_episodes: int = 5,
        selection_metric: str = "sortino_ratio",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.val_env = val_env
        self.validate_every = validate_every
        self.patience = patience
        self.model_save_dir = model_save_dir
        self.best_model_name = best_model_name
        self.n_val_episodes = n_val_episodes
        self.selection_metric = selection_metric

        self.best_metric = -np.inf
        self.no_improvement_count = 0
        self.last_validate_step = 0

    def _compute_start_offsets(self) -> list[int]:
        """Spread start indices evenly across available val data."""
        data_len = len(self.val_env.close_prices)
        ws = self.val_env.window_size
        # Maximum start index that still leaves room for a meaningful episode
        max_start = data_len - ws  # at least ws steps of episode
        if max_start <= ws or self.n_val_episodes <= 1:
            return [ws] * self.n_val_episodes
        step = (max_start - ws) // (self.n_val_episodes - 1)
        return [ws + i * step for i in range(self.n_val_episodes)]

    def _run_episode(self, start_idx: int) -> dict:
        """Run one deterministic episode from a given start index."""
        obs, _ = self.val_env.reset(options={"forced_start_idx": start_idx})
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.val_env.step(int(action))
            done = terminated or truncated
        return self.val_env.get_episode_metrics()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_validate_step < self.validate_every:
            return True

        self.last_validate_step = self.num_timesteps

        # Run multiple episodes with varied start offsets
        start_offsets = self._compute_start_offsets()
        all_metrics = [self._run_episode(s) for s in start_offsets]

        # Average metrics across episodes
        metric_keys = [
            "sharpe_ratio", "sortino_ratio", "cumulative_return",
            "max_drawdown", "calmar_ratio", "trade_count", "win_rate",
        ]
        avg_metrics = {}
        for key in metric_keys:
            vals = [m[key] for m in all_metrics if key in m]
            avg_metrics[key] = float(np.mean(vals)) if vals else 0.0

        # Log to TensorBoard
        self.logger.record("val/sharpe", avg_metrics["sharpe_ratio"])
        self.logger.record("val/sortino", avg_metrics["sortino_ratio"])
        self.logger.record("val/return", avg_metrics["cumulative_return"])
        self.logger.record("val/max_drawdown", avg_metrics["max_drawdown"])
        self.logger.record("val/calmar", avg_metrics["calmar_ratio"])
        self.logger.record("val/trades", avg_metrics["trade_count"])
        self.logger.record("val/win_rate", avg_metrics["win_rate"])

        current_metric = avg_metrics.get(self.selection_metric, avg_metrics["sortino_ratio"])

        if self.verbose:
            print(
                f"  [Val @ {self.num_timesteps}] "
                f"{self.n_val_episodes} episodes  "
                f"Sharpe={avg_metrics['sharpe_ratio']:.3f}  "
                f"Sortino={avg_metrics['sortino_ratio']:.3f}  "
                f"Return={avg_metrics['cumulative_return']:.3%}  "
                f"DD={avg_metrics['max_drawdown']:.3%}  "
                f"Trades={avg_metrics['trade_count']:.0f}  "
                f"WR={avg_metrics['win_rate']:.1%}"
            )

        # Check improvement using configurable selection metric
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.no_improvement_count = 0
            save_path = os.path.join(self.model_save_dir, self.best_model_name)
            self.model.save(save_path)
            if self.verbose:
                print(f"  -> New best {self.selection_metric}={self.best_metric:.3f}, saved to {save_path}")
        else:
            self.no_improvement_count += 1
            if self.verbose:
                print(
                    f"  -> No improvement ({self.no_improvement_count}/{self.patience})"
                )

        # Early stopping
        if self.no_improvement_count >= self.patience:
            if self.verbose:
                print("  -> Early stopping triggered.")
            return False

        return True


def _make_env_fn(data_path: str, config: dict, mode: str):
    """Return a callable that creates a TradingEnv (for SubprocVecEnv)."""
    def _init():
        return make_env(data_path, config, mode)
    return _init


def train(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    set_seeds(config["seed"])

    agent_cfg = config["agent"]
    train_cfg = config["training"]

    train_data_path = os.path.join(config["data"]["save_dir"], "train.npz")
    n_envs = train_cfg.get("n_envs", 1)

    # Create vectorized or single train env
    if n_envs > 1:
        train_env = SubprocVecEnv(
            [_make_env_fn(train_data_path, config, "train") for _ in range(n_envs)]
        )
        # Adjust n_steps per env to maintain same total rollout size
        n_steps = max(agent_cfg["n_steps"] // n_envs, 64)
    else:
        train_env = make_env(train_data_path, config, mode="train")
        n_steps = agent_cfg["n_steps"]

    val_env = make_env(
        os.path.join(config["data"]["save_dir"], "val.npz"), config, mode="val"
    )

    # Create model save directory
    os.makedirs(train_cfg["model_save_dir"], exist_ok=True)

    # Instantiate PPO
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=agent_cfg["learning_rate"],
        n_steps=n_steps,
        batch_size=agent_cfg["batch_size"],
        n_epochs=agent_cfg["n_epochs"],
        gamma=agent_cfg["gamma"],
        gae_lambda=agent_cfg["gae_lambda"],
        clip_range=agent_cfg["clip_range"],
        ent_coef=agent_cfg["ent_coef"],
        vf_coef=agent_cfg["vf_coef"],
        max_grad_norm=agent_cfg["max_grad_norm"],
        policy_kwargs={"net_arch": agent_cfg["net_arch"]},
        tensorboard_log=train_cfg["tensorboard_log"],
        seed=config["seed"],
        verbose=1,
    )

    # Validation callback
    val_cfg = config.get("validation", {})
    val_callback = ValidationCallback(
        val_env=val_env,
        validate_every=train_cfg["validate_every"],
        patience=train_cfg["early_stopping_patience"],
        model_save_dir=train_cfg["model_save_dir"],
        best_model_name=train_cfg["best_model_name"],
        n_val_episodes=val_cfg.get("n_val_episodes", 5),
        selection_metric=val_cfg.get("selection_metric", "sortino_ratio"),
    )

    data_len = len(np.load(train_data_path)["close_prices"])
    print(f"Starting PPO training for {train_cfg['total_timesteps']} timesteps...")
    print(f"  Train data: {data_len} steps ({n_envs} parallel envs, n_steps={n_steps})")
    print(f"  Val data: {len(val_env.close_prices)} steps")
    print(f"  Selection metric: {val_cfg.get('selection_metric', 'sortino_ratio')}")

    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=val_callback,
        tb_log_name="PPO_tony",
    )

    best_path = os.path.join(train_cfg["model_save_dir"], train_cfg["best_model_name"])
    if not os.path.exists(best_path + ".zip"):
        model.save(best_path)
        print("  WARNING: No best model saved during validation. Saving final model as fallback.")

    print(f"Training complete. Best model: {best_path}.zip")
    return best_path


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    train(config_path)
