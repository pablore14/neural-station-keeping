"""Compare episode rewards during training for SAC seeds 1 through 10.

Run from the repository root with:

    python -m src.plots.plot_sac_training
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, RESULTS_DIR_RL
from src.utils.plotting_utils import save_figure, set_paper_style


EXPECTED_SEEDS = tuple(range(1, 11))
SMOOTH_WINDOW = 100
OUTPUT_DIR_DEFAULT = os.path.join(
    PROJECT_ROOT, "results", "plots_comparison_agents"
)


def _load_training_rewards(root):
    paths = glob.glob(os.path.join(root, "sac_*", "logs", "*.monitor.csv"))
    rewards_by_seed = {}
    for path in paths:
        run_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        match = re.match(r"sac_(\d+)(?:_|$)", run_name)
        if not match:
            continue
        seed = int(match.group(1))
        if seed not in EXPECTED_SEEDS:
            continue
        if seed in rewards_by_seed:
            raise ValueError("Multiple monitor logs found for seed {}.".format(seed))

        data = pd.read_csv(path, skiprows=1)
        if "r" not in data or data.empty:
            raise ValueError("{} has no episode rewards.".format(path))
        rewards = data["r"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(rewards)):
            raise ValueError("{} contains non-finite episode rewards.".format(path))
        rewards_by_seed[seed] = rewards

    missing = sorted(set(EXPECTED_SEEDS) - set(rewards_by_seed))
    if missing:
        raise FileNotFoundError(
            "Missing monitor logs for SAC seeds: {}.".format(
                ", ".join(map(str, missing))
            )
        )
    return rewards_by_seed


def plot_training_rewards(root=RESULTS_DIR_RL, out_dir=OUTPUT_DIR_DEFAULT):
    """Plot all ten episode-reward histories and report their progress."""
    rewards_by_seed = _load_training_rewards(root)
    set_paper_style()
    colors = plt.get_cmap("tab10").colors
    common_episodes = min(map(len, rewards_by_seed.values()))
    episodes = np.arange(1, common_episodes + 1)
    smoothed_rewards = []
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)

    for seed in EXPECTED_SEEDS:
        rewards = rewards_by_seed[seed][:common_episodes]
        rolling = pd.Series(rewards).rolling(
            SMOOTH_WINDOW, min_periods=1
        ).mean().to_numpy()
        smoothed_rewards.append(rolling)
        start_mean = float(np.mean(rewards[:SMOOTH_WINDOW]))
        end_mean = float(np.mean(rewards[-SMOOTH_WINDOW:]))
        color = colors[(seed - 1) % len(colors)]

        ax.scatter(episodes[::25], rewards[::25], color=color, alpha=0.035, s=4)
        ax.plot(
            episodes, rolling, color=color, alpha=0.48, linewidth=1.1,
            label="Seed {}".format(seed),
        )
        print(
            "[seed {}] {:,} episodes; first {} mean={:.4f}; "
            "last {} mean={:.4f}".format(
                seed, len(rewards), min(SMOOTH_WINDOW, len(rewards)),
                start_mean, min(SMOOTH_WINDOW, len(rewards)), end_mean,
            ),
            flush=True,
        )

    smoothed_rewards = np.asarray(smoothed_rewards)
    mean_reward = smoothed_rewards.mean(axis=0)
    std_reward = smoothed_rewards.std(axis=0)
    ax.fill_between(
        episodes,
        mean_reward - std_reward,
        mean_reward + std_reward,
        color="black",
        alpha=0.08,
        label="Across-seed standard deviation",
    )
    ax.plot(
        episodes,
        mean_reward,
        color="black",
        linewidth=3.0,
        label="Mean of 10 seeds ({}-episode moving mean)".format(SMOOTH_WINDOW),
        zorder=10,
    )
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
    ax.set(
        xlabel="Training episode",
        ylabel="Episode reward",
        title="SAC training reward — ten seeds and their mean",
    )
    ax.grid(True, alpha=0.2)
    ax.ticklabel_format(axis="x", style="plain")
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    os.makedirs(out_dir, exist_ok=True)
    save_figure(fig, out_dir, "training_reward_all_agents.png")
    print(
        "[done] saved PNG/PDF in {}".format(os.path.abspath(out_dir)),
        flush=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot SAC episode-reward histories for seeds 1 through 10."
    )
    parser.add_argument(
        "--root",
        default=RESULTS_DIR_RL,
        help="results/rl directory containing SAC run directories",
    )
    parser.add_argument(
        "--out-dir",
        default=OUTPUT_DIR_DEFAULT,
        help="directory for the comparison figure",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plot_training_rewards(args.root, args.out_dir)


if __name__ == "__main__":
    main()
