"""Plots for SAC rewards and optimizer diagnostics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.plotting_utils import PALETTE, save_figure


def plot_reward_curve(monitor_csv, out_dir, smooth_window=50):
    """Save episode reward and rolling range against cumulative environment steps."""
    df = pd.read_csv(monitor_csv, skiprows=1)
    rewards = df["r"].to_numpy()
    lengths = df["l"].to_numpy()
    steps = np.cumsum(lengths) / 1e6
    rolling = pd.Series(rewards).rolling(smooth_window, min_periods=1)

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.fill_between(
        steps,
        rolling.min().to_numpy(),
        rolling.max().to_numpy(),
        alpha=0.3,
        color=PALETTE["blue"],
        label="min-max range",
    )
    ax.plot(steps, rolling.mean().to_numpy(), color=PALETTE["blue"], label="mean")
    ax.axhline(0.0, color=PALETTE["red"], linestyle="--", label="zero reward")
    ax.set_xlabel("million steps")
    ax.set_ylabel("episode total reward")
    ax.legend()
    save_figure(fig, out_dir, "reward_curve.png")


def plot_training_losses(progress_csv, out_dir, smooth_window=20):
    """Save the SAC losses available in Stable-Baselines3's progress CSV."""
    try:
        df = pd.read_csv(progress_csv)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"[warning] no training metrics available at {progress_csv}")
        return
    if "time/total_timesteps" in df:
        steps = df["time/total_timesteps"].to_numpy() / 1e6
        xlabel = "million steps"
    else:
        steps = np.arange(len(df))
        xlabel = "logger updates"

    candidates = (
        ("train/critic_loss", "Critic loss"),
        ("train/actor_loss", "Actor loss"),
        ("train/ent_coef", "Entropy coefficient"),
    )
    present = [(name, title) for name, title in candidates if name in df]
    if not present:
        print("[warning] no known SAC loss columns found in progress.csv")
        return

    fig, axes = plt.subplots(
        1, len(present), figsize=(4.5 * len(present), 4),
        constrained_layout=True, squeeze=False,
    )
    for ax, (name, title) in zip(axes[0], present):
        values = pd.Series(df[name].to_numpy())
        ax.plot(steps, values, color=PALETTE["blue"], alpha=0.25, linewidth=1)
        ax.plot(
            steps,
            values.rolling(smooth_window, min_periods=1).mean(),
            color=PALETTE["blue"],
            linewidth=1.8,
        )
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0][0].set_ylabel("value")
    save_figure(fig, out_dir, "training_losses.png")
