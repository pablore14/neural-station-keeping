"""Compare the ten saved SAC test runs and write figures under results/.

Run from the repository root:

    python -m src.plots.plots_SAC_agents

By default, the script requires one completed ``tests/run_data.pkl`` for each
agent seed 1 through 10. All figures are saved as both PNG and PDF.
"""

import argparse
import glob
import os
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from src.config import N_STEPS, PROJECT_ROOT, RESULTS_DIR_RL, SHELL_MAX, SHELL_MIN, TF_S
from src.utils.plotting_utils import PALETTE, save_figure, set_paper_style


EXPECTED_SEEDS = tuple(range(1, 11))
OUTCOMES = ("success", "partial", "collision", "escape")
OUTCOME_COLORS = {
    "success": PALETTE["green"],
    "partial": PALETTE["orange"],
    "collision": PALETTE["red"],
    "escape": PALETTE["blue"],
}
SHELL_COLOR = "#d62728"
OUTPUT_DIR_DEFAULT = os.path.join(
    PROJECT_ROOT, "results", "plots_comparison_agents"
)


def _pastel_palette():
    base = plt.get_cmap("tab10")
    return {
        seed: tuple(
            0.68 * channel + 0.32
            for channel in to_rgb(base((seed - 1) / 10.0))
        )
        for seed in EXPECTED_SEEDS
    }


SEED_COLORS = _pastel_palette()


def _load_pickle(path):
    with open(path, "rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict):
        raise ValueError("{} must contain a dictionary.".format(path))
    return data


def _seed_from_run_data(path, data):
    candidates = (
        os.path.basename(os.path.dirname(os.path.dirname(path))),
        data.get("config", {}).get("agent_run_dir", ""),
    )
    for candidate in candidates:
        match = re.search(r"(?:^|[/\\])sac_(\d+)(?:_|$)", candidate)
        if match:
            return int(match.group(1))
    raise ValueError(
        "Could not determine the SAC seed from the test directory: {}".format(
            path
        )
    )


def _episode_index(episode, position):
    return int(episode.get("ep_idx", position))


def _load_agents(run_data_paths):
    agents = {}
    for path in run_data_paths:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        data = _load_pickle(path)
        episodes = data.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != 500:
            raise ValueError(
                "{} must contain exactly 500 test episodes.".format(path)
            )
        seed = _seed_from_run_data(path, data)
        if seed not in EXPECTED_SEEDS:
            raise ValueError(
                "Expected seeds 1-10, found seed {} in {}.".format(seed, path)
            )
        if seed in agents:
            raise ValueError("Multiple test files found for seed {}.".format(seed))
        ordered = sorted(
            enumerate(episodes),
            key=lambda pair: _episode_index(pair[1], pair[0]),
        )
        indices = [_episode_index(episode, position)
                   for position, episode in ordered]
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate episode indices in {}.".format(path))
        agents[seed] = {
            "seed": seed,
            "path": path,
            "data": data,
            "episodes": [episode for _, episode in ordered],
            "indices": indices,
        }

    missing = sorted(set(EXPECTED_SEEDS) - set(agents))
    if missing:
        raise FileNotFoundError(
            "Missing tests/run_data.pkl for seeds: {}.".format(
                ", ".join(map(str, missing))
            )
        )
    return [agents[seed] for seed in EXPECTED_SEEDS]


def _validate_paired_scenarios(agents):
    reference_indices = agents[0]["indices"]
    reference_episodes = agents[0]["episodes"]
    for agent in agents[1:]:
        if agent["indices"] != reference_indices:
            raise ValueError(
                "Seeds {} and {} have different episode indices.".format(
                    agents[0]["seed"], agent["seed"]
                )
            )
        for ref_episode, episode in zip(reference_episodes, agent["episodes"]):
            ref_x0 = ref_episode.get("scenario", {}).get("x0")
            x0 = episode.get("scenario", {}).get("x0")
            if ref_x0 is not None and x0 is not None:
                if not np.allclose(ref_x0, x0, rtol=0.0, atol=1e-8):
                    raise ValueError(
                        "Initial conditions differ between seeds {} and {} "
                        "for episode {}.".format(
                            agents[0]["seed"],
                            agent["seed"],
                            _episode_index(episode, -1),
                        )
                    )


def _episode_array(episode, key, default=None):
    value = episode.get(key, default)
    if value is None:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float)


def _episode_dv(episode, total_key, series_key):
    if total_key in episode:
        return float(episode[total_key])
    series = _episode_array(episode, series_key)
    return float(np.abs(series).sum())


def _radius_history(episode):
    trajectory = _episode_array(episode, "trajectory")
    if trajectory.ndim != 2 or trajectory.shape[1] < 3 or not len(trajectory):
        raise ValueError("Every test episode must have a valid trajectory.")
    return np.linalg.norm(trajectory[:, :3], axis=1) / 1000.0


def _compute_times(episode):
    return _episode_array(episode, "compute_times").ravel()


def _metric_matrix(agents, metric):
    return np.asarray([
        [metric(episode) for episode in agent["episodes"]]
        for agent in agents
    ], dtype=float)


def _save(fig, out_dir, name):
    save_figure(fig, out_dir, name)
    print("[plot] {}".format(os.path.join(out_dir, name)), flush=True)


def _set_shell_x(ax):
    ax.axvline(SHELL_MIN / 1000.0, color=SHELL_COLOR, linestyle="--",
               linewidth=1.5, label="Shell: 22 km")
    ax.axvline(SHELL_MAX / 1000.0, color=SHELL_COLOR, linestyle="--",
               linewidth=1.5, label="Shell: 30 km")


def _set_shell_y(ax):
    ax.axhline(SHELL_MIN / 1000.0, color=SHELL_COLOR, linestyle="--",
               linewidth=1.5, label="Shell: 22 km")
    ax.axhline(SHELL_MAX / 1000.0, color=SHELL_COLOR, linestyle="--",
               linewidth=1.5, label="Shell: 30 km")


def _plot_final_radius_distribution(agents, out_dir):
    final_radii = _metric_matrix(agents, lambda ep: _radius_history(ep)[-1])
    all_radii = final_radii.ravel()
    bins = np.linspace(float(all_radii.min()), float(all_radii.max()), 45)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for row, agent in zip(final_radii, agents):
        ax.hist(
            row,
            bins=bins,
            color=SEED_COLORS[agent["seed"]],
            alpha=0.34,
            edgecolor=SEED_COLORS[agent["seed"]],
            linewidth=0.6,
            label="Seed {}".format(agent["seed"]),
        )
    _set_shell_x(ax)
    ax.set(
        xlabel="Final orbital radius [km]",
        ylabel="Episodes per bin",
        title="Final-radius distribution by SAC seed",
    )
    ax.legend(ncol=4, fontsize=9)
    _save(fig, out_dir, "01_final_radius_distribution_by_seed.png")

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    box = ax.boxplot(
        [row for row in final_radii],
        patch_artist=True,
        showmeans=True,
        meanline=True,
        tick_labels=["Seed {}".format(agent["seed"]) for agent in agents],
    )
    for patch, agent in zip(box["boxes"], agents):
        patch.set_facecolor(SEED_COLORS[agent["seed"]])
        patch.set_alpha(0.75)
    ax.axhline(SHELL_MIN / 1000.0, color=SHELL_COLOR, linestyle="--",
               linewidth=1.5, label="Shell bounds")
    ax.axhline(SHELL_MAX / 1000.0, color=SHELL_COLOR, linestyle="--",
               linewidth=1.5)
    ax.set(ylabel="Final orbital radius [km]",
           title="Final orbital radius by seed")
    ax.legend()
    _save(fig, out_dir, "02_final_radius_boxplot_by_seed.png")


def _plot_radius_histories(agents, out_dir):
    histories = [
        np.asarray([_radius_history(ep) for ep in agent["episodes"]])
        for agent in agents
    ]
    max_steps = max(history.shape[1] for history in histories)
    time_grid = np.arange(max_steps) * TF_S / N_STEPS / 3600.0

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for history, agent in zip(histories, agents):
        widths = np.full((history.shape[0], max_steps), np.nan)
        widths[:, :history.shape[1]] = history
        valid = np.isfinite(widths).any(axis=0)
        lower, median, upper = np.nanpercentile(
            widths[:, valid], [10, 50, 90], axis=0
        )
        color = SEED_COLORS[agent["seed"]]
        ax.fill_between(time_grid[valid], lower, upper,
                        color=color, alpha=0.10)
        ax.plot(time_grid[valid], median, color=color,
                label="Seed {}".format(agent["seed"]), linewidth=1.5)
    _set_shell_y(ax)
    ax.set(
        xlabel="Simulation time [h]",
        ylabel="Orbital radius [km]",
        title="Radius histories: median and 10th-90th percentile by seed",
        xlim=(0, TF_S / 3600.0),
    )
    ax.legend(ncol=2, fontsize=9)
    _save(fig, out_dir, "03_radius_histories_by_seed.png")


def _plot_delta_v(agents, out_dir):
    commanded = _metric_matrix(
        agents, lambda ep: _episode_dv(ep, "dv_total_sum", "dv_total_series")
    )
    executed = _metric_matrix(
        agents, lambda ep: _episode_dv(ep, "dv_exec_sum", "dv_exec_planet")
    )
    labels = ["Seed {}".format(agent["seed"]) for agent in agents]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    box = ax.boxplot(
        [row for row in commanded],
        patch_artist=True,
        showmeans=True,
        tick_labels=labels,
        showfliers=False,
    )
    for patch, agent in zip(box["boxes"], agents):
        patch.set_facecolor(SEED_COLORS[agent["seed"]])
        patch.set_alpha(0.8)
    ax.set(ylabel="Commanded total delta-v per episode [m/s]",
           title="Commanded delta-v distribution by seed")
    _save(fig, out_dir, "04_commanded_delta_v_by_seed.png")

    positions = np.arange(len(agents))
    width = 0.36
    commanded_mean = commanded.mean(axis=1)
    executed_mean = executed.mean(axis=1)
    commanded_std = commanded.std(axis=1)
    executed_std = executed.std(axis=1)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar(positions - width / 2, commanded_mean, width,
           yerr=commanded_std, capsize=3, color=PALETTE["blue"],
           label="Commanded")
    ax.bar(positions + width / 2, executed_mean, width,
           yerr=executed_std, capsize=3, color=PALETTE["orange"],
           label="Executed")
    ax.set_xticks(positions, labels)
    ax.set(ylabel="Delta-v per episode [m/s]",
           title="Mean commanded and executed delta-v (mean +/- SD)")
    ax.legend()
    _save(fig, out_dir, "05_mean_commanded_vs_executed_delta_v.png")

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for matrix, label, color, marker in (
        (commanded, "Commanded", PALETTE["blue"], "o"),
        (executed, "Executed", PALETTE["orange"], "s"),
    ):
        means = matrix.mean(axis=1)
        stds = matrix.std(axis=1)
        ax.errorbar(positions, means, yerr=stds, marker=marker, capsize=3,
                    color=color, label=label, linewidth=1.2)
    ax.set_xticks(positions, labels)
    ax.set(ylabel="Delta-v per episode [m/s]",
           title="Delta-v mean and variation across episodes")
    ax.legend()
    _save(fig, out_dir, "06_delta_v_mean_and_variability.png")

    components = (
        ("dv_base_sum", "Regressor", PALETTE["blue"]),
        ("dv_res_sum", "SAC residual", PALETTE["orange"]),
        ("dv_total_sum", "Combined command", PALETTE["green"]),
    )
    available = [
        (key, label, color)
        for key, label, color in components
        if all(key in episode for agent in agents for episode in agent["episodes"])
    ]
    if available:
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        width = 0.8 / len(available)
        for component_index, (key, label, color) in enumerate(available):
            means = np.asarray([
                np.mean([float(ep[key]) for ep in agent["episodes"]])
                for agent in agents
            ])
            stds = np.asarray([
                np.std([float(ep[key]) for ep in agent["episodes"]])
                for agent in agents
            ])
            offset = (component_index - (len(available) - 1) / 2) * width
            ax.bar(positions + offset, means, width, yerr=stds, capsize=2,
                   color=color, label=label, alpha=0.85)
        ax.set_xticks(positions, labels)
        ax.set(ylabel="Sum of absolute delta-v per episode [m/s]",
               title="Delta-v component means (error bars: SD)")
        ax.legend()
        _save(fig, out_dir, "07_delta_v_components_by_seed.png")

    if len(commanded) > 1:
        baseline = commanded[0]
        differences = commanded[1:] - baseline[None, :]
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        box = ax.boxplot(
            [row for row in differences],
            patch_artist=True,
            showmeans=True,
            tick_labels=[
                "Seed {}".format(agent["seed"]) for agent in agents[1:]
            ],
            showfliers=False,
        )
        for patch, agent in zip(box["boxes"], agents[1:]):
            patch.set_facecolor(SEED_COLORS[agent["seed"]])
            patch.set_alpha(0.8)
        ax.axhline(0, color=PALETTE["black"], linewidth=1)
        ax.set(
            ylabel="Per-scenario delta-v difference from seed 1 [m/s]",
            title="Paired commanded delta-v differences (same scenarios)",
        )
        _save(fig, out_dir, "08_paired_delta_v_difference_from_seed1.png")


def _plot_outcomes(agents, out_dir):
    counts = np.asarray([
        [sum(ep.get("outcome") == outcome for ep in agent["episodes"])
         for outcome in OUTCOMES]
        for agent in agents
    ], dtype=float)
    percentages = counts / counts.sum(axis=1, keepdims=True) * 100.0
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    bottom = np.zeros(len(agents))
    labels = ["Seed {}".format(agent["seed"]) for agent in agents]
    for index, outcome in enumerate(OUTCOMES):
        ax.bar(np.arange(len(agents)), percentages[:, index], bottom=bottom,
               color=OUTCOME_COLORS[outcome], label=outcome)
        bottom += percentages[:, index]
    ax.set_xticks(np.arange(len(agents)), labels)
    ax.set(ylabel="Episodes [%]", ylim=(0, 100),
           title="Test outcome distribution by seed")
    ax.legend(ncol=len(OUTCOMES))
    _save(fig, out_dir, "09_outcomes_by_seed.png")


def _plot_compute_time(agents, out_dir):
    per_episode_mean = np.asarray([
        [
            float(np.mean(_compute_times(episode)))
            if _compute_times(episode).size
            else np.nan
            for episode in agent["episodes"]
        ]
        for agent in agents
    ]) * 1000.0
    per_episode_total = np.asarray([
        [
            float(episode.get("ctrl_time_sum", _compute_times(episode).sum()))
            for episode in agent["episodes"]
        ]
        for agent in agents
    ])
    test_elapsed = np.asarray([
        float(agent["data"].get("meta", {}).get("elapsed_s", np.nan))
        for agent in agents
    ])
    labels = ["Seed {}".format(agent["seed"]) for agent in agents]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    box = ax.boxplot(
        [row[np.isfinite(row)] for row in per_episode_mean],
        patch_artist=True,
        showmeans=True,
        tick_labels=labels,
        showfliers=False,
    )
    for patch, agent in zip(box["boxes"], agents):
        patch.set_facecolor(SEED_COLORS[agent["seed"]])
        patch.set_alpha(0.8)
    ax.set(ylabel="Mean controller time per decision [ms]",
           title="Controller latency by seed (episode-level mean)")
    _save(fig, out_dir, "10_controller_latency_by_seed.png")

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    box = ax.boxplot(
        [row for row in per_episode_total],
        patch_artist=True,
        showmeans=True,
        tick_labels=labels,
        showfliers=False,
    )
    for patch, agent in zip(box["boxes"], agents):
        patch.set_facecolor(SEED_COLORS[agent["seed"]])
        patch.set_alpha(0.8)
    ax.set(ylabel="Controller computation per episode [s]",
           title="Controller computation time per episode by seed")
    _save(fig, out_dir, "11_controller_time_per_episode_by_seed.png")

    if np.isfinite(test_elapsed).any():
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        colors = [SEED_COLORS[agent["seed"]] for agent in agents]
        ax.bar(np.arange(len(agents)), test_elapsed, color=colors)
        ax.set_xticks(np.arange(len(agents)), labels)
        ax.set(ylabel="Elapsed wall time [s]",
               title="Total evaluation runtime by seed")
        _save(fig, out_dir, "12_test_runtime_by_seed.png")

    p95_latency = np.asarray([
        np.percentile(
            np.concatenate([
                values for values in
                (_compute_times(ep) for ep in agent["episodes"])
                if values.size
            ]),
            95,
        ) * 1000.0
        for agent in agents
    ])
    mean_latency = np.nanmean(per_episode_mean, axis=1)
    mean_dv = np.asarray([
        np.mean([
            _episode_dv(ep, "dv_total_sum", "dv_total_series")
            for ep in agent["episodes"]
        ])
        for agent in agents
    ])
    success_rate = np.asarray([
        np.mean([episode.get("outcome") == "success"
                 for episode in agent["episodes"]]) * 100.0
        for agent in agents
    ])
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    points = ax.scatter(mean_latency, mean_dv, c=success_rate,
                        cmap="viridis", s=80, edgecolor="black", linewidth=0.4)
    for agent, x, y in zip(agents, mean_latency, mean_dv):
        ax.annotate("{}".format(agent["seed"]), (x, y),
                    xytext=(5, 4), textcoords="offset points", fontsize=9)
    fig.colorbar(points, ax=ax, label="Success rate [%]")
    ax.set(xlabel="Mean controller latency [ms]",
           ylabel="Mean commanded delta-v [m/s]",
           title="Delta-v vs computation-time trade-off by seed")
    _save(fig, out_dir, "13_delta_v_vs_latency.png")

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    positions = np.arange(len(agents))
    ax.bar(positions - 0.18, mean_latency, 0.36, color=PALETTE["blue"],
           label="Mean latency [ms]")
    ax.bar(positions + 0.18, p95_latency, 0.36, color=PALETTE["orange"],
           label="95th percentile [ms]")
    ax.set_xticks(positions, labels)
    ax.set(ylabel="Controller latency [ms]",
           title="Mean and 95th-percentile controller latency")
    ax.legend()
    _save(fig, out_dir, "14_controller_latency_percentiles.png")


def _plot_actions(agents, out_dir):
    acted = np.asarray([
        [float(ep.get("steps_acted", 0)) for ep in agent["episodes"]]
        for agent in agents
    ])
    means = acted.mean(axis=1)
    stds = acted.std(axis=1)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    bars = ax.bar(np.arange(len(agents)), means, yerr=stds, capsize=3,
                  color=[SEED_COLORS[agent["seed"]] for agent in agents])
    ax.set_xticks(np.arange(len(agents)),
                  ["Seed {}".format(agent["seed"]) for agent in agents])
    ax.set(ylabel="Active steps per episode",
           title="SAC activations per episode (mean +/- SD)",
           ylim=(0, N_STEPS * 1.12))
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.5,
                "{:.1f}".format(mean), ha="center", va="bottom", fontsize=8)
    _save(fig, out_dir, "15_active_steps_by_seed.png")


def _plot_final_radius_vs_dv(agents, out_dir):
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for agent in agents:
        radius = np.asarray([
            _radius_history(ep)[-1] for ep in agent["episodes"]
        ])
        dv = np.asarray([
            _episode_dv(ep, "dv_total_sum", "dv_total_series")
            for ep in agent["episodes"]
        ])
        ax.scatter(dv, radius, s=10, alpha=0.25,
                   color=SEED_COLORS[agent["seed"]],
                   label="Seed {}".format(agent["seed"]))
    _set_shell_y(ax)
    ax.set(xlabel="Commanded total delta-v [m/s]",
           ylabel="Final orbital radius [km]",
           title="Final radius and commanded delta-v by seed")
    ax.legend(ncol=2, fontsize=9)
    _save(fig, out_dir, "16_final_radius_vs_delta_v.png")


def _plot_cumulative_delta_v(agents, out_dir):
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for agent in agents:
        curves = []
        for episode in agent["episodes"]:
            increments = _episode_array(episode, "dv_total_series")
            if increments.size:
                curves.append(
                    np.concatenate(([0.0], np.cumsum(np.abs(increments))))
                )
        if not curves:
            continue
        width = max(map(len, curves))
        matrix = np.full((len(curves), width), np.nan)
        for index, curve in enumerate(curves):
            matrix[index, :len(curve)] = curve
        valid = np.isfinite(matrix).any(axis=0)
        lower, median, upper = np.nanpercentile(
            matrix[:, valid], [10, 50, 90], axis=0
        )
        time = np.arange(width)[valid] * TF_S / N_STEPS / 3600.0
        color = SEED_COLORS[agent["seed"]]
        ax.fill_between(time, lower, upper, color=color, alpha=0.10)
        ax.plot(time, median, color=color,
                label="Seed {}".format(agent["seed"]), linewidth=1.5)
    ax.set(xlabel="Simulation time [h]",
           ylabel="Cumulative commanded delta-v [m/s]",
           title="Cumulative delta-v: median and 10th-90th percentile")
    ax.legend(ncol=2, fontsize=9)
    _save(fig, out_dir, "17_cumulative_delta_v_by_seed.png")


def _plot_seed_summary(agents, out_dir):
    labels = ["Seed {}".format(agent["seed"]) for agent in agents]
    mean_dv = np.asarray([
        np.mean([
            _episode_dv(ep, "dv_total_sum", "dv_total_series")
            for ep in agent["episodes"]
        ])
        for agent in agents
    ])
    median_radius = _metric_matrix(
        agents, lambda ep: _radius_history(ep)[-1]
    )
    median_radius = np.median(median_radius, axis=1)
    success_rate = np.asarray([
        np.mean([ep.get("outcome") == "success"
                 for ep in agent["episodes"]]) * 100.0
        for agent in agents
    ])
    summary = np.column_stack((mean_dv, median_radius, success_rate))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, values, title, ylabel in zip(
        axes,
        (mean_dv, median_radius, success_rate),
        ("Mean delta-v", "Median final radius", "Success rate"),
        ("[m/s]", "[km]", "[%]"),
    ):
        ax.bar(np.arange(len(agents)), values,
               color=[SEED_COLORS[agent["seed"]] for agent in agents])
        ax.set_xticks(np.arange(len(agents)), labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    _save(fig, out_dir, "18_agent_summary.png")

    csv_path = os.path.join(out_dir, "agent_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write(
            "seed,n_episodes,success_rate_percent,mean_dv_total_m_s,"
            "median_final_radius_km,mean_ctrl_time_ms,p95_ctrl_time_ms,"
            "mean_ctrl_time_per_episode_s,test_elapsed_s\n"
        )
        for agent, row in zip(agents, summary):
            episode_latency_ms = np.asarray([
                np.mean(_compute_times(ep)) * 1000.0
                if _compute_times(ep).size
                else np.nan
                for ep in agent["episodes"]
            ])
            episode_compute_s = np.asarray([
                float(ep.get("ctrl_time_sum", _compute_times(ep).sum()))
                for ep in agent["episodes"]
            ])
            decision_times = np.concatenate([
                values for values in
                (_compute_times(ep) for ep in agent["episodes"])
                if values.size
            ])
            elapsed_s = agent["data"].get("meta", {}).get("elapsed_s", "")
            file.write(
                "{},{},{:.3f},{:.8f},{:.6f},{:.6f},{:.6f},{:.8f},{}\n".format(
                    agent["seed"],
                    len(agent["episodes"]),
                    row[2],
                    row[0],
                    row[1],
                    float(np.nanmean(episode_latency_ms)),
                    float(np.percentile(decision_times, 95) * 1000.0),
                    float(np.mean(episode_compute_s)),
                    elapsed_s,
                )
            )
    print("[summary] {}".format(csv_path), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare the completed SAC agent test runs (seeds 1-10)."
    )
    parser.add_argument(
        "--run-data",
        nargs="+",
        help="test run_data.pkl files; default discovers all ten SAC agents",
    )
    parser.add_argument(
        "--root",
        default=RESULTS_DIR_RL,
        help="results/rl directory to search when --run-data is omitted",
    )
    parser.add_argument(
        "--out-dir",
        default=OUTPUT_DIR_DEFAULT,
        help="directory for the comparison figures and summary CSV",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = args.run_data
    if paths is None:
        paths = sorted(glob.glob(
            os.path.join(args.root, "sac_*", "tests", "run_data.pkl")
        ))
    if not paths:
        raise FileNotFoundError(
            "No tests/run_data.pkl files found under {}.".format(args.root)
        )

    agents = _load_agents(paths)
    _validate_paired_scenarios(agents)
    os.makedirs(args.out_dir, exist_ok=True)
    set_paper_style()
    print(
        "[data] loaded {} agents and {} paired episodes per seed".format(
            len(agents), len(agents[0]["episodes"])
        ),
        flush=True,
    )
    print("[output] {}".format(os.path.abspath(args.out_dir)), flush=True)

    _plot_final_radius_distribution(agents, args.out_dir)
    _plot_radius_histories(agents, args.out_dir)
    _plot_delta_v(agents, args.out_dir)
    _plot_outcomes(agents, args.out_dir)
    _plot_compute_time(agents, args.out_dir)
    _plot_actions(agents, args.out_dir)
    _plot_final_radius_vs_dv(agents, args.out_dir)
    _plot_cumulative_delta_v(agents, args.out_dir)
    _plot_seed_summary(agents, args.out_dir)
    print("[done] Comparison plots generated.", flush=True)


if __name__ == "__main__":
    main()
