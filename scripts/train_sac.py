"""Train residual SAC on top of a frozen base regressor.

Example:
    python -m scripts.train_sac \
        --regressor results/regressor/regressor_<run>/best_model.pth \
        --seed 16

Each run is saved under results/rl/ with configuration, checkpoints, model,
episode monitor data, SB3 training metrics, a summary, and diagnostic plots.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    StopTrainingOnMaxEpisodes,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from src.config import (
    DISPERSION_M,
    DV_MAX_REG,
    DVMAX_SAC,
    MASCON_FRACTIONS,
    MASS_VARIATION,
    N_STEPS,
    RESULTS_DIR_RL,
    SHELL_GAMMA,
    SHELL_MAX,
    SHELL_MIN,
    TF_S,
)
from src.regressor.PolicyNet import PolicyNet
from src.rl.callbacks import RewardPrint, TrainingDataCallback
from src.rl.envs.environment_perturbation import OrbitEnv
from src.rl.rl_plotting import plot_reward_curve, plot_training_losses
from src.rl.scenario import build_scenario
from src.utils.tracking import get_git_commit, make_run_dir
from src.utils.rollout import DV_OFF_THRESHOLD


def parse_args():
    parser = argparse.ArgumentParser(description="Train the residual SAC agent.")
    parser.add_argument(
        "--regressor",
        required=True,
        help="path to a trained base-regressor best_model.pth checkpoint",
    )
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=2500)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_base_policy(checkpoint_path):
    """Rebuild and load the regressor architecture recorded for the checkpoint."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Regressor checkpoint not found: {checkpoint_path}")

    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    config = {}
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)

    model = PolicyNet(
        state_dim=config.get("state_dim", 8),
        hidden_dim=config.get("hidden_dim", 256),
        dv_max=config.get("dv_max", DV_MAX_REG),
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"[regressor] loaded from {checkpoint_path}", flush=True)
    return model


def read_best_threshold(regressor_path):
    """Read the validation-selected classifier threshold, defaulting to 0.5."""
    summary_path = os.path.join(os.path.dirname(regressor_path), "summary.json")
    if not os.path.isfile(summary_path):
        print(f"[warning] no summary.json at {summary_path}; using 0.5", flush=True)
        return 0.5

    with open(summary_path, encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    return summary.get("best_threshold", summary.get("threshold", 0.5))


def build_env(base_policy, seed, log_dir, cls_threshold):
    """Build and validate the monitored single-environment SAC setup."""
    asteroid_dict, optim_dict = build_scenario()
    env = OrbitEnv(
        asteroid_dict,
        optim_dict,
        base_policy=base_policy,
        cls_threshold=cls_threshold,
    )
    env = Monitor(env, filename=os.path.join(log_dir, f"seed{seed}"))
    check_env(env)
    return env


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = make_run_dir(RESULTS_DIR_RL, prefix=f"sac_{args.seed}")
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    regressor_path = os.path.abspath(args.regressor)
    base_policy = load_base_policy(regressor_path)
    cls_threshold = read_best_threshold(regressor_path)
    env = build_env(base_policy, args.seed, log_dir, cls_threshold)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=args.learning_rate,
        buffer_size=1_000_000,
        learning_starts=100,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        seed=args.seed,
    )
    model.set_logger(configure(log_dir, ["csv"]))

    config = {
        "git_commit": get_git_commit(),
        "seed": args.seed,
        "regressor_path": regressor_path,
        "classifier_threshold": cls_threshold,
        "max_episodes": args.max_episodes,
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "tau": 0.005,
        "gamma": 0.99,
        "policy": "MlpPolicy",
        "action_space": "normalized [-1, 1] residual delta-v",
        "episode_duration_s": TF_S,
        "episode_steps": N_STEPS,
        "residual_dv_max_m_s": DVMAX_SAC,
        "thruster_off_threshold_m_s": DV_OFF_THRESHOLD,
        "shell_bounds_m": [SHELL_MIN, SHELL_MAX],
        "shell_reward_scale": SHELL_GAMMA,
        "mass_fraction_1_nominal": float(MASCON_FRACTIONS[0]),
        "mass_fraction_1_range": [
            float(MASCON_FRACTIONS[0] * (1 - MASS_VARIATION)),
            float(MASCON_FRACTIONS[0] * (1 + MASS_VARIATION)),
        ],
        "mass_variation_relative": MASS_VARIATION,
        "initial_position_sampling": "uniform in volume within a sphere",
        "initial_position_dispersion_m": DISPERSION_M,
        "reward_trace_path": os.path.join(log_dir, "training_steps.csv"),
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    callbacks = [
        StopTrainingOnMaxEpisodes(max_episodes=args.max_episodes, verbose=0),
        CheckpointCallback(
            save_freq=optim_dict_steps(env) * 250,
            save_path=checkpoint_dir,
            name_prefix=f"sac_seed{args.seed}",
        ),
        RewardPrint(every_episodes=10),
        TrainingDataCallback(config["reward_trace_path"]),
    ]
    print(
        f"[train] SAC, up to {args.max_episodes} episodes, seed={args.seed}",
        flush=True,
    )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=1,
        )
        final_path = os.path.join(out_dir, f"sac_model_final_seed{args.seed}")
        model.save(final_path)

        monitor_csv = os.path.join(log_dir, f"seed{args.seed}.monitor.csv")
        plot_reward_curve(monitor_csv, out_dir)
        plot_training_losses(os.path.join(log_dir, "progress.csv"), out_dir)
    finally:
        env.close()

    summary = {
        **config,
        "model_path": final_path + ".zip",
        "monitor_path": monitor_csv,
        "training_metrics_path": os.path.join(log_dir, "progress.csv"),
        "reward_trace_path": config["reward_trace_path"],
        "checkpoint_dir": checkpoint_dir,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[saved] {final_path}.zip", flush=True)
    print(f"[done] results in: {out_dir}", flush=True)


def optim_dict_steps(monitored_env):
    """Return the number of control steps in one episode."""
    return monitored_env.get_wrapper_attr("optim_dict")["N"]


if __name__ == "__main__":
    main()
