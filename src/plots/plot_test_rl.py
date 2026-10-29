"""Generate paper-style plots from SAC test ``run_data.pkl`` files.

Run from the repository root with:

    python -m src.plots.plot_test_rl

The script discovers ``results/rl/sac_*/tests/run_data.pkl`` by default and
saves PNG and PDF figures next to each file. If the test bundle points to an
available MPC reference run, paired SAC-vs-MPC figures are included.
"""

import argparse
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.config import (
    ASTEROID_AXES_M,
    RL_CONTROL_STEP_S,
    RESULTS_DIR_RL,
    SHELL_MAX,
    SHELL_MIN,
    TF_S,
)
from src.utils.plotting_utils import PALETTE, save_figure, set_paper_style


OUTCOMES = ("success", "partial", "collision", "escape")
OUTCOME_COLORS = {
    "success": PALETTE["green"],
    "partial": PALETTE["orange"],
    "collision": PALETTE["red"],
    "escape": PALETTE["blue"],
}
AGENT_COLORS = {"SAC": PALETTE["blue"], "MPC": PALETTE["orange"]}
MAX_TRAJECTORIES = 60


def _load_pickle(path):
    with open(path, "rb") as file:
        value = pickle.load(file)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a dictionary.".format(path))
    return value


def _episodes(run_data, path):
    episodes = run_data.get("episodes")
    if episodes is None:
        episodes = run_data.get("all_results", {}).get("sac")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("{} does not contain any SAC test episodes.".format(path))
    return episodes


def _episode_index(episode, fallback):
    return int(episode.get("ep_idx", fallback))


def _episode_map(episodes):
    indexed = {_episode_index(episode, i): episode
               for i, episode in enumerate(episodes)}
    if len(indexed) != len(episodes):
        raise ValueError("Episode indices are not unique.")
    return indexed


def _load_reference(data, explicit_path):
    config = data.get("config", {})
    path = explicit_path or config.get("mpc_reference_path")
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        print("[skip] MPC reference not found: {}".format(path), file=sys.stderr)
        return None
    reference = _load_pickle(path)
    episodes = reference.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("{} does not contain MPC episodes.".format(path))
    return episodes


def _paired_episodes(episodes, reference):
    sac_by_index = _episode_map(episodes)
    mpc_by_index = _episode_map(reference)
    shared = sorted(set(sac_by_index) & set(mpc_by_index))
    if not shared:
        return [], []

    sac = [sac_by_index[index] for index in shared]
    mpc = [mpc_by_index[index] for index in shared]
    for sac_episode, mpc_episode in zip(sac, mpc):
        sac_scenario = sac_episode.get("scenario", {})
        mpc_scenario = mpc_episode.get("scenario", {})
        sac_x0 = sac_scenario.get("x0")
        mpc_x0 = mpc_scenario.get("x0")
        if sac_x0 is not None and mpc_x0 is not None:
            if not np.allclose(sac_x0, mpc_x0, rtol=0.0, atol=1e-8):
                raise ValueError(
                    "SAC and MPC initial states differ for episode {}.".format(
                        _episode_index(sac_episode, -1)
                    )
                )
    return sac, mpc


def _array(episode, key, dtype=float):
    return np.asarray(episode[key], dtype=dtype)


def _dv(episode, key="dv_total_sum"):
    if key in episode:
        return float(episode[key])
    if key == "dv_total_sum" and "dv_sum" in episode:
        return float(episode["dv_sum"])
    series_key = key.replace("_sum", "_series")
    if series_key in episode:
        return float(np.abs(_array(episode, series_key)).sum())
    return 0.0


def _radius(episode):
    states = _array(episode, "trajectory")
    if states.ndim != 2 or states.shape[1] < 3 or not len(states):
        raise ValueError("Episode trajectory must have shape (n, >=3).")
    return np.linalg.norm(states[:, :3], axis=1)


def _save(fig, out_dir, name):
    save_figure(fig, out_dir, name)
    print("[plot] {}".format(os.path.join(out_dir, name)), flush=True)


def _style_3d(ax):
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-40, 40)


def _plot_asteroid(ax):
    axes = ASTEROID_AXES_M / 1000.0
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 24)
    x = axes[0] * np.outer(np.cos(u), np.sin(v))
    y = axes[1] * np.outer(np.sin(u), np.sin(v))
    z = axes[2] * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#b5651d", alpha=0.35,
                    linewidth=0.15, edgecolor="#4a4a4a")


def _plot_outcomes(episodes, reference, out_dir):
    groups = [("SAC", episodes)]
    if reference:
        groups.append(("MPC", reference))
    counts = [[sum(ep.get("outcome") == outcome for ep in rows)
               for outcome in OUTCOMES] for _, rows in groups]

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    x = np.arange(len(OUTCOMES))
    width = 0.8 / len(groups)
    for i, ((label, rows), values) in enumerate(zip(groups, counts)):
        bars = ax.bar(
            x + (i - (len(groups) - 1) / 2) * width,
            values,
            width,
            label=label,
            color=AGENT_COLORS[label],
        )
        for bar, value in zip(bars, values):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    "{}\n{:.1f}%".format(value, 100 * value / len(rows)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x, OUTCOMES)
    ax.set_ylabel("episodes")
    ax.set_title("Test outcomes")
    ax.legend()
    _save(fig, out_dir, "01_outcomes.png")

    fig, axes = plt.subplots(
        1, len(groups), figsize=(4.8 * len(groups), 4.8), squeeze=False,
        constrained_layout=True,
    )
    for ax, (label, rows) in zip(axes[0], groups):
        values = [sum(ep.get("outcome") == outcome for ep in rows)
                  for outcome in OUTCOMES]
        present = [(outcome, value) for outcome, value in zip(OUTCOMES, values)
                   if value]
        ax.pie(
            [value for _, value in present],
            labels=[outcome for outcome, _ in present],
            colors=[OUTCOME_COLORS[outcome] for outcome, _ in present],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        ax.set_title(label)
    _save(fig, out_dir, "02_outcome_pies.png")


def _plot_delta_v(episodes, reference, paired_sac, paired_mpc, out_dir):
    groups = [("SAC", episodes)]
    if reference:
        groups.append(("MPC", reference))
    series = [[_dv(ep) for ep in rows] for _, rows in groups]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    violin = ax.violinplot(series, showmedians=True, showextrema=False)
    for body, (label, _) in zip(violin["bodies"], groups):
        body.set_facecolor(AGENT_COLORS[label])
        body.set_alpha(0.55)
    ax.set_xticks(np.arange(1, len(groups) + 1), [label for label, _ in groups])
    ax.set_ylabel("commanded total $\\Delta v$ per episode [m/s]")
    ax.set_title("Delta-v distribution")
    _save(fig, out_dir, "03_dv_distribution.png")

    acted = [float(ep.get("steps_acted", 0)) for ep in episodes]
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    ax.hist(acted, bins=min(30, max(1, len(set(acted)))),
            color=AGENT_COLORS["SAC"], edgecolor="white")
    ax.set_xlabel("steps with an applied action")
    ax.set_ylabel("episodes")
    ax.set_title("SAC action activations")
    _save(fig, out_dir, "04_activations.png")

    if paired_sac and paired_mpc:
        sac_dv = np.asarray([_dv(ep) for ep in paired_sac])
        mpc_dv = np.asarray([_dv(ep) for ep in paired_mpc])
        upper = max(float(sac_dv.max()), float(mpc_dv.max()), 1e-9) * 1.05

        fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
        for outcome in OUTCOMES:
            mask = np.asarray([ep.get("outcome") == outcome for ep in paired_sac])
            if mask.any():
                ax.scatter(
                    mpc_dv[mask], sac_dv[mask], s=20, alpha=0.7,
                    color=OUTCOME_COLORS[outcome], label=outcome,
                )
        ax.plot([0, upper], [0, upper], "--", color=PALETTE["grey"])
        ax.set(xlim=(0, upper), ylim=(0, upper),
               xlabel="MPC commanded total $\\Delta v$ [m/s]",
               ylabel="SAC commanded total $\\Delta v$ [m/s]",
               title="Paired delta-v by test scenario")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        _save(fig, out_dir, "11_paired_dv_scatter.png")

        savings = mpc_dv - sac_dv
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        ax.hist(savings, bins=40, color=PALETTE["green"], edgecolor="white")
        ax.axvline(0, color=PALETTE["black"], linewidth=1)
        ax.axvline(
            savings.mean(), color=PALETTE["red"], linestyle="--",
            label="mean = {:+.4f} m/s; SAC lower in {:.1f}%".format(
                savings.mean(), 100 * np.mean(savings > 0)
            ),
        )
        ax.set_xlabel("MPC $\\Delta v$ - SAC $\\Delta v$ [m/s]")
        ax.set_ylabel("paired episodes")
        ax.set_title("Paired delta-v savings")
        ax.legend()
        _save(fig, out_dir, "12_dv_savings.png")

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for label, values in zip((name for name, _ in groups), series):
        sorted_values = np.sort(values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax.step(sorted_values, cdf, where="post", color=AGENT_COLORS[label],
                label=label)
    ax.set(xlabel="commanded total $\\Delta v$ per episode [m/s]",
           ylabel="cumulative fraction", ylim=(0, 1),
           title="Delta-v cumulative distribution")
    ax.legend()
    _save(fig, out_dir, "13_dv_cdf.png")

    present_outcomes = [
        outcome for outcome in OUTCOMES
        if any(ep.get("outcome") == outcome for ep in episodes)
    ]
    dv_by_outcome = [
        [_dv(ep) for ep in episodes if ep.get("outcome") == outcome]
        for outcome in present_outcomes
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    boxes = ax.boxplot(dv_by_outcome, patch_artist=True, showmeans=True)
    for box, outcome in zip(boxes["boxes"], present_outcomes):
        box.set_facecolor(OUTCOME_COLORS[outcome])
        box.set_alpha(0.55)
    ax.set_xticks(np.arange(1, len(present_outcomes) + 1), present_outcomes)
    ax.set_ylabel("commanded total $\\Delta v$ per episode [m/s]")
    ax.set_title("Delta-v by SAC outcome")
    _save(fig, out_dir, "14_dv_vs_outcome.png")


def _plot_trajectories(episodes, reference, paired_sac, paired_mpc, out_dir):
    indices = np.linspace(
        0, len(episodes) - 1, min(MAX_TRAJECTORIES, len(episodes)), dtype=int
    )
    samples = [episodes[index] for index in indices]
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    _plot_asteroid(ax)
    for ep in samples:
        trajectory = _array(ep, "trajectory")[:, :3] / 1000.0
        color = OUTCOME_COLORS.get(ep.get("outcome"), PALETTE["grey"])
        ax.plot(*trajectory.T, color=color, alpha=0.28, linewidth=0.8)
    _style_3d(ax)
    ax.set_title("Representative SAC trajectories")
    _save(fig, out_dir, "05_sac_trajectories_3d.png")

    if paired_sac and paired_mpc:
        fig = plt.figure(figsize=(14, 6))
        for panel, (label, rows) in enumerate(
            (("MPC", paired_mpc), ("SAC", paired_sac)), 1
        ):
            ax = fig.add_subplot(1, 2, panel, projection="3d")
            _plot_asteroid(ax)
            for ep in rows[:MAX_TRAJECTORIES]:
                trajectory = _array(ep, "trajectory")[:, :3] / 1000.0
                color = OUTCOME_COLORS.get(ep.get("outcome"), PALETTE["grey"])
                ax.plot(*trajectory.T, color=color, alpha=0.25, linewidth=0.7)
            _style_3d(ax)
            ax.set_title("{} trajectories".format(label))
        _save(fig, out_dir, "06_paired_trajectories_3d.png")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for ep in episodes:
        times = _array(ep, "times") / 3600.0
        radius = _radius(ep) / 1000.0
        n = min(len(times), len(radius))
        if n:
            ax.plot(times[:n], radius[:n],
                    color=OUTCOME_COLORS.get(ep.get("outcome"), PALETTE["grey"]),
                    alpha=0.12, linewidth=0.65)
    ax.axhline(SHELL_MIN / 1000.0, color=PALETTE["red"], linestyle="--")
    ax.axhline(SHELL_MAX / 1000.0, color=PALETTE["red"], linestyle="--")
    ax.set(xlabel="simulation time [h]", ylabel="orbital radius [km]",
           title="SAC orbital radius histories", xlim=(0, TF_S / 3600.0))
    _save(fig, out_dir, "07_radius_vs_time.png")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    groups = [("SAC", episodes)]
    if reference:
        groups.append(("MPC", reference))
    for label, rows in groups:
        max_len = max(len(_radius(ep)) for ep in rows)
        values = np.full((len(rows), max_len), np.nan)
        times = np.full((len(rows), max_len), np.nan)
        for i, ep in enumerate(rows):
            radius = _radius(ep) / 1000.0
            episode_times = _array(ep, "times")[:len(radius)] / 3600.0
            values[i, :len(radius)] = radius
            times[i, :len(episode_times)] = episode_times
        valid = np.isfinite(values).any(axis=0)
        lower, median, upper = np.nanpercentile(
            values[:, valid], [10, 50, 90], axis=0
        )
        median_time = np.nanmedian(times[:, valid], axis=0)
        ax.fill_between(median_time, lower, upper,
                        color=AGENT_COLORS[label], alpha=0.16)
        ax.plot(median_time, median, color=AGENT_COLORS[label], label=label)
    ax.axhline(SHELL_MIN / 1000.0, color=PALETTE["red"], linestyle="--")
    ax.axhline(SHELL_MAX / 1000.0, color=PALETTE["red"], linestyle="--")
    ax.set(xlabel="simulation time [h]", ylabel="orbital radius [km]",
           title="Radius median and 10th-90th percentile")
    ax.legend()
    _save(fig, out_dir, "08_radius_envelopes.png")

    initial = np.asarray([_array(ep, "trajectory")[0, :3]
                          for ep in episodes]) / 1000.0
    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    _plot_asteroid(ax)
    for outcome in OUTCOMES:
        mask = np.asarray([ep.get("outcome") == outcome for ep in episodes])
        if mask.any():
            ax.scatter(*initial[mask].T, color=OUTCOME_COLORS[outcome],
                       s=14, alpha=0.65, label=outcome)
    _style_3d(ax)
    ax.set_title("SAC initial states by outcome")
    ax.legend()
    _save(fig, out_dir, "09_initial_cloud.png")

    final_radius = np.asarray([_radius(ep)[-1] for ep in episodes]) / 1000.0
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.hist(final_radius, bins=40, color=AGENT_COLORS["SAC"], alpha=0.8,
            edgecolor="white")
    ax.axvspan(SHELL_MIN / 1000.0, SHELL_MAX / 1000.0,
               color=PALETTE["green"], alpha=0.1)
    ax.axvline(SHELL_MIN / 1000.0, color=PALETTE["red"], linestyle="--")
    ax.axvline(SHELL_MAX / 1000.0, color=PALETTE["red"], linestyle="--")
    ax.set(xlabel="final orbital radius [km]", ylabel="episodes",
           title="Final orbital radius")
    _save(fig, out_dir, "10_final_radius.png")

    theta = np.linspace(0, 2 * np.pi, 300)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.add_patch(plt.Circle((0, 0), SHELL_MAX / 1000.0,
                            color="0.85", zorder=0))
    ax.add_patch(plt.Circle((0, 0), SHELL_MIN / 1000.0,
                            color="white", zorder=0))
    ax.plot(
        (SHELL_MIN / 1000.0) * np.cos(theta),
        (SHELL_MIN / 1000.0) * np.sin(theta),
        color=PALETTE["red"], linestyle="--",
    )
    ax.plot(
        (SHELL_MAX / 1000.0) * np.cos(theta),
        (SHELL_MAX / 1000.0) * np.sin(theta),
        color=PALETTE["red"], linestyle="--",
    )
    for ep in samples:
        trajectory = _array(ep, "trajectory")[:, :2] / 1000.0
        color = OUTCOME_COLORS.get(ep.get("outcome"), PALETTE["grey"])
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                color=color, alpha=0.25, linewidth=0.7)
    ax.set(xlim=(-40, 40), ylim=(-40, 40), xlabel="X [km]", ylabel="Y [km]",
           title="SAC top view (XY)")
    ax.set_aspect("equal", adjustable="box")
    _save(fig, out_dir, "18_topview_sac.png")

    if paired_sac and paired_mpc:
        _plot_paired_topview(paired_mpc, paired_sac, out_dir)


def _plot_paired_topview(mpc_episodes, sac_episodes, out_dir):
    """Plot all paired episodes, emphasizing trajectories outside success."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, (label, rows) in zip(
        axes, (("MPC", mpc_episodes), ("SAC", sac_episodes))
    ):
        ax.add_patch(plt.Circle((0, 0), SHELL_MAX / 1000.0,
                                color="0.85", zorder=0))
        ax.add_patch(plt.Circle((0, 0), SHELL_MIN / 1000.0,
                                color="white", zorder=0))
        ax.add_patch(plt.Circle((0, 0), SHELL_MIN / 1000.0,
                                fill=False, edgecolor="0.55",
                                linewidth=0.8, zorder=1))
        ax.add_patch(plt.Circle((0, 0), SHELL_MAX / 1000.0,
                                fill=False, edgecolor="0.55",
                                linewidth=0.8, zorder=1))

        counts = {outcome: 0 for outcome in OUTCOMES}
        outcome_order = {
            "success": 0,
            "partial": 1,
            "escape": 2,
            "collision": 3,
        }
        ordered_rows = sorted(
            rows,
            key=lambda episode: outcome_order.get(
                episode.get("outcome"), len(outcome_order)
            ),
        )
        outcome_alpha = {
            "success": 0.14,
            "partial": 0.45,
            "escape": 0.14,
            "collision": 0.85,
        }
        for episode in ordered_rows:
            outcome = episode.get("outcome")
            counts[outcome] = counts.get(outcome, 0) + 1
            trajectory = _array(episode, "trajectory")[:, :2] / 1000.0
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=OUTCOME_COLORS.get(outcome, PALETTE["grey"]),
                alpha=outcome_alpha.get(outcome, 0.14),
                linewidth=0.5,
                zorder=2,
            )

        handles = [
            Line2D(
                [0], [0],
                color=OUTCOME_COLORS[outcome],
                linewidth=1.5,
                label="{}: {}".format(outcome, counts.get(outcome, 0)),
            )
            for outcome in OUTCOMES
            if counts.get(outcome, 0)
        ]
        ax.set(
            xlim=(-40, 40),
            ylim=(-40, 40),
            xlabel="X [km]",
            ylabel="Y [km]",
            title="{} top view — {} trajectories".format(label, len(rows)),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    _save(fig, out_dir, "19_topview_mpc_vs_sac.png")


def _plot_dv_components(episodes, out_dir):
    component_keys = (
        ("dv_base_sum", "Regressor component", PALETTE["blue"]),
        ("dv_res_sum", "SAC residual", PALETTE["orange"]),
        ("dv_total_sum", "Combined command", PALETTE["green"]),
    )
    available = [
        (key, label, color) for key, label, color in component_keys
        if all(key in ep for ep in episodes)
    ]
    if available:
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        values = [[float(ep[key]) for ep in episodes] for key, _, _ in available]
        violin = ax.violinplot(values, showmedians=True, showextrema=False)
        for body, (_, _, color) in zip(violin["bodies"], available):
            body.set_facecolor(color)
            body.set_alpha(0.55)
        ax.set_xticks(np.arange(1, len(available) + 1),
                      [label for _, label, _ in available])
        ax.set_ylabel("$\\sum |\\Delta v|$ per episode [m/s]")
        ax.set_title("SAC delta-v components")
        _save(fig, out_dir, "15_dv_breakdown.png")

    series_keys = (
        ("dv_base_series", "Regressor component", PALETTE["blue"]),
        ("dv_res_series", "SAC residual", PALETTE["orange"]),
        ("dv_total_series", "Combined command", PALETTE["green"]),
    )
    if not all(all(key in ep for ep in episodes) for key, _, _ in series_keys):
        return
    fig, axes = plt.subplots(
        1, len(series_keys), figsize=(15, 4.5), sharey=True,
        constrained_layout=True,
    )
    for ax, (key, label, color) in zip(axes, series_keys):
        curves = []
        for ep in episodes:
            increments = np.abs(_array(ep, key)).ravel()
            curves.append(np.concatenate(([0.0], np.cumsum(increments))))
        width = max(map(len, curves))
        values = np.full((len(curves), width), np.nan)
        for i, curve in enumerate(curves):
            values[i, :len(curve)] = curve
        step = np.arange(width) * RL_CONTROL_STEP_S / 3600.0
        valid = np.isfinite(values).any(axis=0)
        low, median, high = np.nanpercentile(values[:, valid],
                                              [10, 50, 90], axis=0)
        ax.fill_between(step[valid], low, high, color=color, alpha=0.18)
        ax.plot(step[valid], median, color=color)
        ax.set_title(label)
        ax.set_xlabel("simulation time [h]")
    axes[0].set_ylabel("cumulative $\\Delta v$ [m/s]")
    _save(fig, out_dir, "16_cumulative_dv.png")

    if all("action_residual" in ep and "dv_res_series" in ep for ep in episodes):
        actions = np.concatenate([
            _array(ep, "action_residual").ravel() for ep in episodes
        ])
        residual = np.concatenate([
            _array(ep, "dv_res_series").ravel() for ep in episodes
        ])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                                 constrained_layout=True)
        axes[0].hist(actions, bins=60, color=PALETTE["orange"],
                     edgecolor="white")
        axes[0].axvline(0, color=PALETTE["black"], linewidth=1)
        axes[0].set(xlabel="normalized SAC residual action",
                    ylabel="control steps",
                    title="{:.1f}% with |action| < 0.05".format(
                        100 * np.mean(np.abs(actions) < 0.05)
                    ))
        axes[1].hist(residual, bins=60, color=PALETTE["orange"],
                     edgecolor="white")
        axes[1].axvline(0, color=PALETTE["black"], linewidth=1)
        axes[1].set(xlabel="SAC residual $\\Delta v$ [m/s]",
                    ylabel="control steps",
                    title="Mean absolute residual: {:.5g} m/s".format(
                        np.mean(np.abs(residual))
                    ))
        _save(fig, out_dir, "17_sac_residual_distribution.png")


def _compute_times(episode):
    values = episode.get("compute_times")
    if values is None:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float).ravel()


def _plot_compute_time(episodes, meta, out_dir):
    time_arrays = [_compute_times(ep) for ep in episodes]
    nonempty_times = [values for values in time_arrays if values.size]
    all_times = (
        np.concatenate(nonempty_times) if nonempty_times else np.asarray([])
    )
    if all_times.size:
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        ax.hist(all_times * 1000.0, bins=50, color=PALETTE["blue"],
                edgecolor="white")
        ax.axvline(np.median(all_times) * 1000.0, color=PALETTE["red"],
                   linestyle="--", label="median")
        ax.set(xlabel="controller compute time per decision [ms]",
               ylabel="control steps", title="SAC controller latency")
        ax.legend()
        _save(fig, out_dir, "20_controller_time_per_step.png")

    totals = np.asarray([values.sum() for values in
                         (_compute_times(ep) for ep in episodes)])
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.hist(totals, bins=40, color=PALETTE["green"], edgecolor="white")
    ax.set(xlabel="controller compute time per episode [s]",
           ylabel="episodes", title="Controller compute time per episode")
    _save(fig, out_dir, "21_controller_time_per_episode.png")

    timed = [(ep.get("outcome", "unknown"), float(ep.get(
        "ctrl_time_sum", _compute_times(ep).sum()
    ))) for ep in episodes]
    outcomes = [outcome for outcome in OUTCOMES
                if any(label == outcome for label, _ in timed)]
    values = [[seconds for label, seconds in timed if label == outcome]
              for outcome in outcomes]
    if outcomes:
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        boxes = ax.boxplot(values, patch_artist=True, showmeans=True)
        for box, outcome in zip(boxes["boxes"], outcomes):
            box.set_facecolor(OUTCOME_COLORS[outcome])
            box.set_alpha(0.55)
        ax.set_xticks(np.arange(1, len(outcomes) + 1), outcomes)
        ax.set(ylabel="controller compute time per episode [s]",
               title="Compute time by SAC outcome")
        _save(fig, out_dir, "22_controller_time_by_outcome.png")

    step_counts = [len(_compute_times(ep)) for ep in episodes]
    if max(step_counts, default=0):
        matrix = np.full((len(episodes), max(step_counts)), np.nan)
        for i, ep in enumerate(episodes):
            values = _compute_times(ep)
            matrix[i, :len(values)] = values * 1000.0
        steps = np.arange(matrix.shape[1])
        valid = np.isfinite(matrix).any(axis=0)
        low, median, high = np.nanpercentile(
            matrix[:, valid], [10, 50, 90], axis=0
        )
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        ax.fill_between(steps[valid], low, high, color=PALETTE["blue"],
                        alpha=0.18, label="10th-90th percentile")
        ax.plot(steps[valid], median, color=PALETTE["blue"], label="median")
        ax.set(xlabel="control step", ylabel="controller compute time [ms]",
               title="Controller latency by step")
        ax.legend()
        _save(fig, out_dir, "23_controller_time_by_step.png")

    elapsed = meta.get("elapsed_s")
    if elapsed is not None:
        controller_total = float(totals.sum())
        fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
        ax.bar(
            ["test wall time", "sum of controller calls"],
            [float(elapsed), controller_total],
            color=[PALETTE["grey"], PALETTE["blue"]],
        )
        ax.set(ylabel="seconds", title="Test runtime")
        _save(fig, out_dir, "24_test_runtime.png")


def plot_test_run(run_data_path, reference_path=None):
    run_data_path = os.path.abspath(run_data_path)
    if not os.path.isfile(run_data_path):
        raise FileNotFoundError(run_data_path)
    data = _load_pickle(run_data_path)
    episodes = _episodes(data, run_data_path)
    reference = _load_reference(data, reference_path)
    paired_sac, paired_mpc = ([], [])
    if reference:
        paired_sac, paired_mpc = _paired_episodes(episodes, reference)
        if not paired_sac:
            print("[skip] no episode indices shared with the MPC reference",
                  file=sys.stderr)
        else:
            print("[data] paired SAC/MPC episodes: {}".format(len(paired_sac)))

    out_dir = os.path.dirname(run_data_path)
    set_paper_style()
    _plot_outcomes(episodes, reference, out_dir)
    _plot_delta_v(episodes, reference, paired_sac, paired_mpc, out_dir)
    _plot_trajectories(episodes, reference, paired_sac, paired_mpc, out_dir)
    _plot_dv_components(episodes, out_dir)
    _plot_compute_time(episodes, data.get("meta", {}), out_dir)
    print("[done] {} episodes; figures saved in {}".format(
        len(episodes), out_dir
    ), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot SAC test run_data.pkl files."
    )
    parser.add_argument(
        "--run-data",
        nargs="+",
        help="one or more test run_data.pkl files (default: discover all SAC tests)",
    )
    parser.add_argument(
        "--root",
        default=RESULTS_DIR_RL,
        help="root directory to search when --run-data is omitted",
    )
    parser.add_argument(
        "--mpc-reference",
        help="MPC run_data.pkl to use instead of each test's configured reference",
    )
    parser.add_argument(
        "--only-paired-topview",
        action="store_true",
        help="regenerate only figure 19, the paired MPC/SAC top view",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_data_paths = args.run_data or sorted(
        os.path.join(directory, "run_data.pkl")
        for directory, _, files in os.walk(args.root)
        if os.path.basename(directory) == "tests" and "run_data.pkl" in files
    )
    if not run_data_paths:
        raise FileNotFoundError(
            "No tests/run_data.pkl files found under {}.".format(args.root)
        )
    failures = []
    for path in run_data_paths:
        try:
            print("[data] {}".format(path), flush=True)
            if args.only_paired_topview:
                data = _load_pickle(path)
                episodes = _episodes(data, path)
                reference = _load_reference(data, args.mpc_reference)
                if not reference:
                    raise ValueError(
                        "{} has no available MPC reference.".format(path)
                    )
                paired_sac, paired_mpc = _paired_episodes(episodes, reference)
                if not paired_sac:
                    raise ValueError(
                        "{} has no episodes paired with its MPC reference."
                        .format(path)
                    )
                _plot_paired_topview(
                    paired_mpc, paired_sac, os.path.dirname(os.path.abspath(path))
                )
                print(
                    "[done] figure 19; {} paired episodes saved in {}".format(
                        len(paired_sac), os.path.dirname(os.path.abspath(path))
                    ),
                    flush=True,
                )
            else:
                plot_test_run(path, args.mpc_reference)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append((path, error))
            print("[error] {}: {}: {}".format(
                path, type(error).__name__, error
            ), file=sys.stderr, flush=True)
    if failures:
        raise SystemExit("{} test run(s) could not be plotted.".format(
            len(failures)
        ))


if __name__ == "__main__":
    main()
