"""Evaluate trained residual-SAC agents on the shared 500-scenario MPC batch.

Example:
    python -m scripts.test_sac

The default run evaluates every trained agent under results/rl/ with the same
regressor and actuation-noise seeds as the nominal MPC evaluation. Each agent's
results, summary, and comparison plots are saved under its tests/ directory.
"""

import argparse
import copy
import datetime
import glob
import json
import os
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC

from src.config import N_STEPS, RESULTS_DIR_RL, TF_S
from src.regressor.data_preparation.state_scaling import adimensionalize_state
from src.rl.scenario import build_scenario
from src.utils.plotting_utils import PALETTE, save_figure, set_paper_style
from src.utils.rollout import DV_OFF_THRESHOLD, run_episode
from scripts.train_sac import load_base_policy


DEFAULT_SCENARIOS = os.path.join("data", "initial_conditions_500.pkl")
DEFAULT_MPC_RESULTS = os.path.join(
    "results", "mpc", "mpc_nominal_nearest_20261005_203926", "run_data.pkl"
)
NOISE_SEED_STRIDE = 1_000_000
NOMINAL_MPC_SEED = 123
OUTCOMES = ("success", "partial", "collision", "escape")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate trained SAC agents on the shared MPC test batch."
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="SAC run directories (default: all trained runs under results/rl/).",
    )
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--mpc-results", default=DEFAULT_MPC_RESULTS)
    parser.add_argument("--noise-seed", type=int, default=NOMINAL_MPC_SEED)
    return parser.parse_args(argv)


def _load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def load_test_batch(path):
    """Load and validate the initial-condition batch without importing MPC."""
    data = _load_pickle(path)
    if not isinstance(data, dict) or "batch" not in data:
        raise ValueError("{} is not an initial-condition batch.".format(path))
    batch = data["batch"]
    for key in ("x0", "muM", "xyzM"):
        if key not in batch:
            raise ValueError("{} is missing batch['{}'].".format(path, key))
    n_episodes = len(batch["x0"])
    if np.asarray(batch["x0"]).shape != (n_episodes, 6):
        raise ValueError("{}: batch['x0'] must have shape (n, 6).".format(path))
    if len(batch["muM"]) != n_episodes or len(batch["xyzM"]) != n_episodes:
        raise ValueError("{}: batch scenario arrays have inconsistent lengths."
                         .format(path))
    return batch, data.get("meta", {})


def episode_asteroid(nominal, batch, index):
    """Apply one test episode's perturbed mascon to the nominal asteroid."""
    asteroid = copy.deepcopy(nominal)
    xyz_m = np.asarray(batch["xyzM"][index], dtype=np.float64)
    asteroid["mascon"]["muM"] = np.asarray(batch["muM"][index], dtype=np.float64)
    asteroid["mascon"]["xyzM"] = xyz_m
    asteroid["xyzM"] = xyz_m.copy()
    return asteroid


def _dump_pickle(obj, path):
    temporary_path = path + ".tmp"
    with open(temporary_path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, path)


def _dump_json(obj, path):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, default=_json_default)
    os.replace(temporary_path, path)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def discover_agents(agent_dirs):
    """Load each run's saved config and verify its model and shared regressor."""
    if agent_dirs is None:
        agent_dirs = sorted(
            path
            for path in glob.glob(os.path.join(RESULTS_DIR_RL, "sac_*"))
            if os.path.isdir(path)
            and os.path.isfile(os.path.join(path, "config.json"))
            and os.path.isfile(os.path.join(path, "summary.json"))
        )
    if not agent_dirs:
        raise FileNotFoundError("No trained SAC run directories were found.")

    agents = []
    for agent_dir in agent_dirs:
        agent_dir = os.path.abspath(agent_dir)
        config_path = os.path.join(agent_dir, "config.json")
        summary_path = os.path.join(agent_dir, "summary.json")
        if not os.path.isfile(config_path) or not os.path.isfile(summary_path):
            raise FileNotFoundError(
                f"{agent_dir} must contain config.json and summary.json."
            )
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
        with open(summary_path, encoding="utf-8") as file:
            summary = json.load(file)

        model_path = summary.get("model_path")
        if not model_path:
            model_path = os.path.join(
                agent_dir, "sac_model_final_seed{}.zip".format(config["seed"])
            )
        model_path = os.path.abspath(model_path)
        regressor_path = os.path.abspath(config["regressor_path"])
        if not os.path.isfile(model_path):
            raise FileNotFoundError("SAC model not found: {}".format(model_path))
        if not os.path.isfile(regressor_path):
            raise FileNotFoundError(
                "Regressor checkpoint not found: {}".format(regressor_path)
            )
        agents.append({
            "run_dir": agent_dir,
            "seed": int(config["seed"]),
            "model_path": model_path,
            "regressor_path": regressor_path,
            "classifier_threshold": float(config["classifier_threshold"]),
            "config": config,
        })

    shared_regressors = {agent["regressor_path"] for agent in agents}
    if len(shared_regressors) != 1:
        raise ValueError(
            "All SAC agents must use the same regressor checkpoint; found: "
            + ", ".join(sorted(shared_regressors))
        )
    thresholds = {agent["classifier_threshold"] for agent in agents}
    if len(thresholds) != 1:
        raise ValueError(
            "All SAC agents must use the same classifier threshold; found: "
            + ", ".join(map(str, sorted(thresholds)))
        )
    return agents


def validate_test_batch(batch, mpc_run):
    """Ensure this is the exact 500-scenario batch evaluated by the MPC run."""
    x0_batch = np.asarray(batch["x0"], dtype=np.float64)
    if x0_batch.shape != (500, 6):
        raise ValueError(
            "Expected exactly 500 initial states with shape (500, 6); got "
            + str(x0_batch.shape)
        )

    mpc_episodes = mpc_run.get("episodes")
    if mpc_episodes is None or len(mpc_episodes) != 500:
        raise ValueError("The MPC reference must contain all 500 test episodes.")
    by_index = {int(ep["ep_idx"]): ep for ep in mpc_episodes}
    if set(by_index) != set(range(500)):
        raise ValueError("The MPC reference does not contain episode indices 0-499.")
    for index, x0 in enumerate(x0_batch):
        mpc_x0 = np.asarray(by_index[index]["scenario"]["x0"], dtype=np.float64)
        mpc_scenario = by_index[index]["scenario"]
        if not np.array_equal(mpc_x0, x0):
            raise ValueError(
                "MPC initial condition does not match the batch at episode {}."
                .format(index)
            )
        if not np.array_equal(
            np.asarray(mpc_scenario["muM"]), np.asarray(batch["muM"][index])
        ) or not np.array_equal(
            np.asarray(mpc_scenario["xyzM"]), np.asarray(batch["xyzM"][index])
        ):
            raise ValueError(
                "MPC mascon scenario does not match the batch at episode {}."
                .format(index)
            )
    return by_index


class SACRolloutPolicy:
    """Adapt the saved SB3 policy and its frozen base regressor to run_episode."""

    def __init__(self, model, base_policy, classifier_threshold, optim_dict):
        self.model = model
        self.base_policy = base_policy
        self.classifier_threshold = classifier_threshold
        _, self.r_ad, self.v_ad = optim_dict["c"]
        self.dvmax_residual = float(optim_dict["dvmax_SAC"])
        self.n_steps = int(optim_dict["N"])
        self.cumulative_dv_ad = 0.0
        self.device = next(base_policy.parameters()).device

    def __call__(self, step, state):
        state_ad = np.concatenate(
            (state[:3] / self.r_ad, state[3:6] / self.v_ad)
        )
        features = adimensionalize_state(state[None, :], [step])
        inputs = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, base_dv = self.base_policy(inputs)
            active = torch.sigmoid(logits) > self.classifier_threshold
            base_dv_m_s = float(torch.where(active, base_dv, 0.0).item())

        observation = np.concatenate(
            (
                state_ad,
                [self.cumulative_dv_ad, base_dv_m_s / self.v_ad],
            )
        ).astype(np.float32)
        action, _ = self.model.predict(observation, deterministic=True)
        action = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        residual_dv_m_s = action * self.dvmax_residual
        total_dv_m_s = base_dv_m_s + residual_dv_m_s
        return total_dv_m_s, {
            "action_residual": action,
            "dv_base_m_s": base_dv_m_s,
            "dv_residual_m_s": residual_dv_m_s,
            "dv_total_m_s": total_dv_m_s,
            "cumulative_dv_ad": self.cumulative_dv_ad,
        }

    def record_transition(self, _step, _state, _command, _info, _ctrl_time,
                          execution, _next_state):
        self.cumulative_dv_ad += (
            np.linalg.norm(execution["dv_planet"]) / self.v_ad / self.n_steps
        )


def make_episode_record(roll, ep_index, batch, noise_seed):
    """Convert a shared rollout into the episode schema used by MPC."""
    infos = roll["infos"]
    dv_base = np.asarray([info["dv_base_m_s"] for info in infos], dtype=np.float64)
    dv_res = np.asarray(
        [info["dv_residual_m_s"] for info in infos], dtype=np.float64
    )
    dv_total = np.asarray(
        [info["dv_total_m_s"] for info in infos], dtype=np.float64
    )
    dv_exec = np.linalg.norm(roll["dv_exec_planet"], axis=1)
    trajectory = roll["trajectory"]
    controls_vnb = np.zeros((len(dv_total), 3), dtype=np.float64)
    controls_vnb[:, 0] = dv_total

    return {
        "ep_idx": ep_index,
        "noise_seed": noise_seed,
        "outcome": roll["outcome"],
        "terminated": roll["terminated"],
        "n_steps": roll["n_steps"],
        "times": roll["times"],
        "trajectory": trajectory,
        "dv_vnb": controls_vnb,
        "dv_exec_vnb": roll["dv_exec_vnb"],
        "dv_exec_planet": roll["dv_exec_planet"],
        "fired": roll["fired"],
        "dv_base_series": dv_base,
        "dv_res_series": dv_res,
        "dv_total_series": dv_total,
        "action_res_norm": dv_res.copy(),
        "action_residual": np.asarray(
            [info["action_residual"] for info in infos], dtype=np.float64
        ),
        "compute_times": roll["compute_times"],
        "final_state": trajectory[-1].copy(),
        "final_radius_m": float(np.linalg.norm(trajectory[-1, :3])),
        "dv_base_sum": float(np.abs(dv_base).sum()),
        "dv_res_sum": float(np.abs(dv_res).sum()),
        "dv_total_sum": float(np.abs(dv_total).sum()),
        "dv_exec_sum": float(dv_exec.sum()),
        "dv_vnb_norm_sum": float(np.abs(dv_total).sum()),
        "steps_acted_base": int(np.count_nonzero(np.abs(dv_base) > DV_OFF_THRESHOLD)),
        "steps_acted_res": int(np.count_nonzero(np.abs(dv_res) > DV_OFF_THRESHOLD)),
        "steps_acted": int(np.count_nonzero(roll["fired"])),
        "ctrl_time_sum": float(roll["compute_times"].sum()),
        "ctrl_time_mean": float(roll["compute_times"].mean()),
        "ctrl_time_median": float(np.median(roll["compute_times"])),
        "ctrl_time_max": float(roll["compute_times"].max()),
        "scenario": {
            "x0": np.asarray(batch["x0"][ep_index], dtype=np.float64).copy(),
            "muM": np.asarray(batch["muM"][ep_index], dtype=np.float64).copy(),
            "xyzM": np.asarray(batch["xyzM"][ep_index], dtype=np.float64).copy(),
            "eps": float(batch["eps"][ep_index]) if "eps" in batch else None,
            "rho": float(batch["rho"][ep_index]) if "rho" in batch else None,
        },
    }


def compute_stats(episodes, mpc_episodes):
    """Summarize SAC outcomes and paired results against MPC."""
    counts = {outcome: 0 for outcome in OUTCOMES}
    for episode in episodes:
        counts[episode["outcome"]] += 1
    success = [ep for ep in episodes if ep["outcome"] == "success"]

    def mean(key, rows):
        return float(np.mean([row[key] for row in rows])) if rows else float("nan")

    paired_outcomes = {}
    for episode in episodes:
        reference = mpc_episodes[episode["ep_idx"]]["outcome"]
        key = "{}->{}".format(reference, episode["outcome"])
        paired_outcomes[key] = paired_outcomes.get(key, 0) + 1

    return {
        **counts,
        "n_episodes": len(episodes),
        "success_rate": counts["success"] / max(len(episodes), 1),
        "mean_dv_total": mean("dv_total_sum", episodes),
        "mean_dv_exec": mean("dv_exec_sum", episodes),
        "mean_dv_base": mean("dv_base_sum", episodes),
        "mean_dv_res": mean("dv_res_sum", episodes),
        "mean_dv_success": mean("dv_total_sum", success),
        "mean_acts_total": mean("steps_acted", episodes),
        "mean_acts_res": mean("steps_acted_res", episodes),
        "mean_ctrl_time": mean("ctrl_time_mean", episodes),
        "max_ctrl_time": max(
            (ep["ctrl_time_max"] for ep in episodes), default=0.0
        ),
        "total_steps": sum(ep["n_steps"] for ep in episodes),
        "mpc_outcome_to_sac_outcome": paired_outcomes,
        "mpc_mean_dv_total": mean("dv_total_sum", list(mpc_episodes.values())),
        "mpc_mean_dv_exec": mean("dv_exec_sum", list(mpc_episodes.values())),
    }


def _episode_matrix(episodes, n_steps=N_STEPS):
    values = np.full((len(episodes), n_steps + 1), np.nan, dtype=np.float64)
    for row, episode in enumerate(episodes):
        states = np.asarray(episode["trajectory"], dtype=np.float64)
        n = min(len(states), n_steps + 1)
        values[row, :n] = np.linalg.norm(states[:n, :3], axis=1) / 1e3
    return values


def plot_test_results(episodes, mpc_episodes, out_dir):
    """Save outcome, paired delta-v, and radius-history comparisons."""
    set_paper_style()
    indices = np.arange(len(episodes))

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    x = np.arange(len(OUTCOMES))
    width = 0.36
    sac_counts = [sum(ep["outcome"] == outcome for ep in episodes)
                  for outcome in OUTCOMES]
    mpc_counts = [sum(ep["outcome"] == outcome for ep in mpc_episodes)
                  for outcome in OUTCOMES]
    ax.bar(x - width / 2, sac_counts, width, color=PALETTE["blue"], label="SAC")
    ax.bar(x + width / 2, mpc_counts, width, color=PALETTE["orange"], label="MPC")
    ax.set_xticks(x, OUTCOMES)
    ax.set_ylabel("number of episodes")
    ax.set_title("Outcomes on the shared 500-scenario batch")
    ax.legend()
    save_figure(fig, out_dir, "outcomes.png")

    sac_dv = np.asarray([ep["dv_total_sum"] for ep in episodes])
    mpc_dv = np.asarray([mpc_episodes[i]["dv_total_sum"] for i in indices])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].scatter(mpc_dv, sac_dv, s=16, alpha=0.65, color=PALETTE["blue"])
    upper = max(float(mpc_dv.max()), float(sac_dv.max()), 1e-6)
    axes[0].plot([0, upper], [0, upper], "--", color=PALETTE["grey"])
    axes[0].set_xlabel("MPC commanded delta-v (m/s)")
    axes[0].set_ylabel("SAC commanded delta-v (m/s)")
    axes[0].set_title("Per-scenario total delta-v")
    axes[1].hist(mpc_dv, bins=25, alpha=0.65, color=PALETTE["orange"], label="MPC")
    axes[1].hist(sac_dv, bins=25, alpha=0.65, color=PALETTE["blue"], label="SAC")
    axes[1].set_xlabel("commanded delta-v (m/s)")
    axes[1].set_ylabel("number of episodes")
    axes[1].set_title("Delta-v distribution")
    axes[1].legend()
    save_figure(fig, out_dir, "delta_v_comparison.png")

    sac_exec = np.asarray([ep["dv_exec_sum"] for ep in episodes])
    mpc_exec = np.asarray([mpc_episodes[i]["dv_exec_sum"] for i in indices])
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    ax.scatter(mpc_exec, sac_exec, s=16, alpha=0.65, color=PALETTE["green"])
    upper = max(float(mpc_exec.max()), float(sac_exec.max()), 1e-6)
    ax.plot([0, upper], [0, upper], "--", color=PALETTE["grey"])
    ax.set_xlabel("MPC executed delta-v (m/s)")
    ax.set_ylabel("SAC executed delta-v (m/s)")
    ax.set_title("Per-scenario executed delta-v")
    save_figure(fig, out_dir, "executed_delta_v_comparison.png")

    sac_radius = _episode_matrix(episodes)
    mpc_radius = _episode_matrix(mpc_episodes)
    step_times = np.arange(sac_radius.shape[1]) * TF_S / N_STEPS / 3600.0
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for radius, label, color in (
        (sac_radius, "SAC", PALETTE["blue"]),
        (mpc_radius, "MPC", PALETTE["orange"]),
    ):
        valid_steps = np.isfinite(radius).any(axis=0)
        lower, median, upper = np.nanpercentile(
            radius[:, valid_steps], [10, 50, 90], axis=0
        )
        ax.fill_between(
            step_times[valid_steps], lower, upper, color=color, alpha=0.18
        )
        ax.plot(
            step_times[valid_steps], median,
            color=color, label="{} median".format(label)
        )
    ax.axhline(22.0, color=PALETTE["grey"], linestyle="--", linewidth=1)
    ax.axhline(30.0, color=PALETTE["grey"], linestyle="--", linewidth=1)
    ax.set_xlabel("simulation time (hours)")
    ax.set_ylabel("orbit radius (km)")
    ax.set_title("Radius histories (median and 10th-90th percentile)")
    ax.legend()
    save_figure(fig, out_dir, "radius_envelopes.png")


def evaluate_agent(agent, base_policy, threshold, batch, batch_meta,
                   mpc_by_index, scenarios_path, mpc_path, noise_seed):
    """Run all 500 scenarios for one SAC checkpoint and save its test bundle."""
    out_dir = os.path.join(agent["run_dir"], "tests")
    os.makedirs(out_dir, exist_ok=True)
    run_data_path = os.path.join(out_dir, "run_data.pkl")
    config = {
        "agent_run_dir": agent["run_dir"],
        "agent_model_path": agent["model_path"],
        "regressor_path": agent["regressor_path"],
        "classifier_threshold": threshold,
        "scenarios_path": os.path.abspath(scenarios_path),
        "n_scenarios": 500,
        "mpc_reference_path": os.path.abspath(mpc_path),
        "noise_seed": noise_seed,
        "noise_seed_stride": NOISE_SEED_STRIDE,
        "deterministic_policy": True,
    }
    if os.path.isfile(run_data_path):
        saved_data = _load_pickle(run_data_path)
        saved_config = saved_data.get("config", {})
        if any(saved_config.get(key) != value for key, value in config.items()):
            raise ValueError(
                "{} exists but was generated with different evaluation inputs."
                .format(run_data_path)
            )
        episodes = saved_data.get("episodes", [])
        if len(episodes) != 500:
            raise ValueError(
                "{} does not contain all 500 episodes.".format(run_data_path)
            )
        stats = compute_stats(episodes, mpc_by_index)
        _dump_json(
            {"config": config, "summary": stats},
            os.path.join(out_dir, "summary.json"),
        )
        plot_test_results(
            episodes,
            [mpc_by_index[index] for index in range(500)],
            out_dir,
        )
        print(
            "[cached] rebuilt plots for {} from {}".format(agent["run_dir"], run_data_path),
            flush=True,
        )
        return

    model = SAC.load(agent["model_path"], device="cpu")
    nominal, optim_dict = build_scenario()
    episodes = []
    started = time.perf_counter()
    started_at = datetime.datetime.now().isoformat()

    for index in range(500):
        noise_seed_i = int(noise_seed * NOISE_SEED_STRIDE + index)
        scenario_asteroid = episode_asteroid(nominal, batch, index)
        policy = SACRolloutPolicy(model, base_policy, threshold, optim_dict)
        rng = np.random.default_rng(noise_seed_i)
        rollout = run_episode(
            policy,
            batch["x0"][index],
            scenario_asteroid,
            optim_dict,
            rng,
            on_transition=policy.record_transition,
        )
        episode = make_episode_record(rollout, index, batch, noise_seed_i)
        episodes.append(episode)
        if (index + 1) % 25 == 0 or index == 0:
            print(
                "[{}] {}/500: outcome={}, dv={:.4f} m/s".format(
                    os.path.basename(agent["run_dir"]),
                    index + 1,
                    episode["outcome"],
                    episode["dv_total_sum"],
                ),
                flush=True,
            )

    mpc_episodes = [mpc_by_index[index] for index in range(500)]
    stats = compute_stats(episodes, mpc_by_index)
    data = {
        "schema_version": 1,
        "meta": {
            "script": "test_sac",
            "started": started_at,
            "finished": datetime.datetime.now().isoformat(),
            "elapsed_s": time.perf_counter() - started,
        },
        "config": config,
        "summary": stats,
        "scenarios_meta": batch_meta,
        "scenario_batch": {
            key: np.asarray(value).copy() for key, value in batch.items()
        },
        "episodes": episodes,
        "initial_states": np.asarray([ep["trajectory"][0] for ep in episodes]),
        "all_results": {"sac": episodes},
    }
    _dump_pickle(data, run_data_path)
    _dump_json({"config": config, "summary": stats}, os.path.join(out_dir, "summary.json"))
    plot_test_results(episodes, mpc_episodes, out_dir)
    print(
        "[done] {}: success={}/500, mean delta-v={:.4f} m/s; {}".format(
            agent["run_dir"],
            stats["success"],
            stats["mean_dv_total"],
            run_data_path,
        ),
        flush=True,
    )


def main(argv=None):
    args = parse_args(argv)
    agents = discover_agents(args.agents)
    batch, batch_meta = load_test_batch(args.scenarios)
    mpc_run = _load_pickle(args.mpc_results)
    mpc_by_index = validate_test_batch(batch, mpc_run)

    shared_regressor = agents[0]["regressor_path"]
    base_policy = load_base_policy(shared_regressor)
    threshold = agents[0]["classifier_threshold"]
    for agent in agents:
        print(
            "[agent] {} seed={} | model={} | shared regressor={}".format(
                agent["run_dir"],
                agent["seed"],
                agent["model_path"],
                shared_regressor,
            ),
            flush=True,
        )
        evaluate_agent(
            agent,
            base_policy,
            threshold,
            batch,
            batch_meta,
            mpc_by_index,
            args.scenarios,
            args.mpc_results,
            args.noise_seed,
        )


if __name__ == "__main__":
    main()
