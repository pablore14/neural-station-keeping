"""
Closed-loop MPC evaluation on the shared batch of initial conditions.

This script only RUNS the MPC and SAVES everything. It makes no plots: all
plots are produced afterwards from <run_dir>/run_data.pkl by a separate script.

===========================================================================
SCENARIOS AND SIMULATION
===========================================================================
Episode i is read from an initial-conditions pickle already generated
(--scenarios, default data/initial_conditions_500.pkl):

    initial state   batch["x0"][i]                    (6,)  m, m/s
    mass split      batch["muM"][i], batch["xyzM"][i]

and simulated with src.simulation.rollout, the closed loop every method uses:
impulse at the start of each step with actuation errors (only when the
thruster fires, i.e. |dv| > DV_OFF_THRESHOLD), then propagation for tf/N
seconds with src.propagators.integrate3D_scipy. Collision and escape are
detected continuously with the integrator events; the episode ends there or
at step N.

The actuation-error RNG of episode i is seeded with noise_seed*1e6 + i.

===========================================================================
MODEL SEEN BY THE MPC   (--model)
===========================================================================
  nominal   (default; the right choice for comparing against RL)
            The MPC plans with the NOMINAL mascon while the simulation uses the
            episode's one: realistic model mismatch.

  oracle    The MPC plans with the episode's exact mascon. Optimistic upper
            bound: quantifies the cost of not knowing the mass distribution.

  The MPC always works on its own copy of the asteroid dict. Actuation errors
  are never modelled: closed loop absorbs them (plan, apply the first
  impulse, replan).

===========================================================================
INITIAL GUESS   (--guess)
===========================================================================
    nearest   (default) Cold start from the stored solution whose initial state
                        is closest in non-dimensional units. The default library is
                        results/direct_results_VNBTangential_v2.pkl. Accepted layouts:
                        OCP datasets (open-loop plans), a run_data.pkl of this script, or
                        a legacy all_results.pkl.

    natural   Free dynamics from x0 with dV = 0.

  ref       *** DIAGNOSTIC ONLY, DO NOT REPORT ***
            Uses a reference trajectory (--ref-results) as the guess. It only
            answers "is this OCP feasible at all?".

  Only the first solve of each episode is cold; later solves warm-start from
  the previous plan shifted by one step.

===========================================================================
ELLIPSOID HOMOTOPY
===========================================================================
The guess is scored against the collision ellipsoid,

    q_min = min_k [ (x/a)^2 + (y/b)^2 + (z/c)^2 ].

If q_min < 1 the guess crosses the asteroid; the exclusion constraint is
non-convex with a vanishing gradient at the centre and IPOPT cannot escape.
The ellipsoid is then shrunk by f0 = 0.8*sqrt(q_min) (guess strictly outside),
then by (f0+1)/2, each stage seeding the next, and finally solved at full size.

REQUIREMENT: src/optimizers/optimizersv2.py must honour optim_dict["max_iter"]:
    'ipopt.max_iter': optim_dict.get('max_iter', 250)

===========================================================================
OUTPUT   <run_dir>/
===========================================================================
  run_config.json         arguments, written at start (checked on --resume)
  episodes/ep_00000.pkl   one file per finished episode (crash-safe)
  run_data.pkl            every episode saved in episodes/ (also those of other
                          processes sharing the directory), assembled at the end:
      schema_version, meta, config, summary,
      episodes            list of per-episode dicts
      initial_states      (n, 6)
      all_results         {"mpc": episodes}   legacy alias, same objects
  summary.json            meta + config essentials + summary statistics

Per-episode dict, main keys (n = n_steps; states and times have n+1 rows):
  scenario           x0, muM, xyzM, eps, rho
  trajectory (n+1,6) times (n+1,)
  dv_vnb     (n,3)   impulse commanded by the MPC (VNB)
  dv_exec_vnb (n,3), dv_exec_planet (n,3), fired (n,)   what was executed
  dv_*_series        legacy (base = 0, res = total = commanded tangential dv)
  compute_times, t_final_solve, solver_status, solver_status_raw,
  action_source ("solve" | "reused_plan" | "zero"), guess_source, guess_q,
  guess_q_min, homotopy_stages, solver_extra, library_match,
  plans              per step {"T","X","dV"} of the accepted plan (None when
                     the solve failed; None for all steps with --no-plans)
  plus aggregates: dv sums, act counts, timing, failure counts, outcome.

===========================================================================
USAGE
===========================================================================
    python -u -m scripts.mpc

    # a subset, e.g. to split the batch across machines into the same dir:
    ... --out-dir results/mpc/mpc_run --ep-start 0   --n-episodes 250
    ... --out-dir results/mpc/mpc_run --ep-start 250 --n-episodes 250 --resume

    # continue an interrupted run:
    ... --out-dir results/mpc/mpc_nominal_natural_20260930_180000 --resume

    # upper bound with a perfect model:
    ... --model oracle
    # cold start with natural dynamics instead of the tangential library:
    ... --guess natural
"""

import argparse
import copy
import datetime
import json
import os
import pickle
import platform
import socket
import sys
import time

import numpy as np

from src.config import (DT_STEP, SHELL_MIN, SHELL_MAX, DVMAX_MPC,
                        SIGMA_DV_ABS, SIGMA_THETA)
from src.utils.scenario import build_scenario
from src.utils.tracking import get_git_commit
from src.propagators import integrate3D_dV
from src.optimizers.optimizersv2 import optimize_safeorbit2
from src.optimizers.integrator_casadi import integrator3D_casadi
from src.utils.rollout import run_episode, control_dt, DV_OFF_THRESHOLD


SCHEMA_VERSION = 2

DEFAULT_SCENARIOS = os.path.join("data", "initial_conditions_500.pkl")
DEFAULT_OUT_ROOT = os.path.join("results", "mpc")

RES_ACTION_DEADBAND = 1e-8
MAX_ITER_FINAL = 250          # IPOPT iterations, final solve
MAX_ITER_HOMOT = 80           # IPOPT iterations, intermediate homotopy stages
HOMOTOPY_MARGIN = 0.8         # f0 = margin * sqrt(q_min)
REF_IC_TOL_M = 1.0            # a reference run must share the batch within this
NOISE_SEED_STRIDE = 1_000_000

# Arguments that must not change when resuming into an existing run dir.
RESUME_KEYS = ("scenarios", "model", "guess", "guess_library",
               "library_max_fail", "ref_results", "ref_mode", "noise_seed",
               "no_plans")


# ===========================================================================
# Generic helpers
# ===========================================================================
def _pad_to(arr, n):
    """Truncate to n rows, or repeat the last row up to n. Returns a copy."""
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) >= n:
        return arr[:n].copy()
    extra = np.repeat(arr[-1:], n - len(arr), axis=0)
    return np.vstack([arr, extra])


def is_success(status):
    """'success' and 'success_homotopia' both count as success."""
    return str(status).startswith("success")


def _picklable(obj):
    """Return obj if it pickles; otherwise keep the parts that do, repr the rest."""
    try:
        pickle.dumps(obj)
        return obj
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _picklable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_picklable(v) for v in obj]
    return repr(obj)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return repr(o)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _dump_pickle(obj, path):
    """Write atomically: a crash never leaves a half-written file behind."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _dump_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def guess_quality(X, ellip_axes):
    """Score a guess against the collision ellipsoid (q < 1: node inside)."""
    X = np.asarray(X, dtype=np.float64)
    a, b, c = np.asarray(ellip_axes, dtype=np.float64)
    q = (X[:, 0] / a) ** 2 + (X[:, 1] / b) ** 2 + (X[:, 2] / c) ** 2
    r = np.linalg.norm(X[:, :3], axis=1)
    return {
        "q_min":      float(q.min()),
        "q_argmin":   int(q.argmin()),
        "r_min_km":   float(r.min()) / 1e3,
        "r_max_km":   float(r.max()) / 1e3,
        "viola_elip": bool(q.min() < 1.0),
    }


# ===========================================================================
# Initial conditions: the batch pickle {"batch": {...}, "meta": {...}}
# ===========================================================================
NOMINAL_TOL_REL = 1e-9       # relative tolerance on the total mu
NOMINAL_TOL_M = 1e-6         # tolerance on the mascon positions, m


def load_batch(path):
    """Load and validate the initial-conditions pickle. Returns (batch, meta).

    Needs batch["x0"] (n, 6) in m and m/s, batch["muM"] and batch["xyzM"]
    (per-episode mascon); "eps" and "rho" are stored with each episode if present.
    """
    d = _load_pickle(path)
    if not isinstance(d, dict) or "batch" not in d:
        raise SystemExit(f"{path} has no 'batch' key; is it an initial-conditions file?")
    batch, meta = d["batch"], d.get("meta", {})
    for key in ("x0", "muM", "xyzM"):
        if key not in batch:
            raise SystemExit(f"{path}: batch is missing '{key}'")
    n = len(batch["x0"])
    if np.asarray(batch["x0"]).shape != (n, 6):
        raise SystemExit(f"{path}: batch['x0'] must be (n, 6)")
    for key in ("muM", "xyzM"):
        if len(batch[key]) != n:
            raise SystemExit(f"{path}: batch['{key}'] has {len(batch[key])} rows, x0 has {n}")
    return batch, meta


def episode_asteroid(nominal, batch, i):
    """Deep copy of the nominal asteroid with episode i's mascon.

    Starting from the full nominal dict keeps every other key the propagators
    use (omega, sun_pert, ...).
    """
    d = copy.deepcopy(nominal)
    xyzM = np.array(batch["xyzM"][i], dtype=np.float64)
    d["mascon"]["muM"] = np.array(batch["muM"][i], dtype=np.float64)
    d["mascon"]["xyzM"] = xyzM
    d["xyzM"] = xyzM.copy()
    return d


def check_nominal_mascon(asteroid_dict, meta):
    """Compare the nominal mascon with the one the batch was generated from.

    A mismatch means the MPC nominal model and the batch perturbations are
    centred on different asteroids, which would bias the comparison.
    """
    mu = np.asarray(asteroid_dict["mascon"]["muM"], dtype=np.float64)
    xyz = np.asarray(asteroid_dict["mascon"]["xyzM"], dtype=np.float64)
    check = {}
    if "mascon_mu_total" in meta:
        check["mu_total_rel_diff"] = float(abs(mu.sum() / float(meta["mascon_mu_total"]) - 1.0))
    if "mascon_positions_nominal" in meta:
        ref = np.asarray(meta["mascon_positions_nominal"], dtype=np.float64)
        check["xyz_max_diff_m"] = (float(np.abs(xyz - ref).max())
                                   if ref.shape == xyz.shape else float("inf"))
    check["ok"] = (check.get("mu_total_rel_diff", 0.0) <= NOMINAL_TOL_REL
                   and check.get("xyz_max_diff_m", 0.0) <= NOMINAL_TOL_M)
    return check


# ===========================================================================
# Optional inputs: reference run, solution library
# ===========================================================================
def load_reference(path, mode, x0_batch, indices):
    """Load a reference run (for --guess ref and the outcome cross table).

    Returns (episodes, mode, max initial-state difference against the batch).
    """
    d = _load_pickle(path)
    all_res = {"episodes": d["episodes"]} if "episodes" in d else d["all_results"]
    mode = mode or ("total" if "total" in all_res else next(iter(all_res)))
    ref = all_res[mode]
    if len(ref) <= max(indices):
        raise SystemExit(f"reference {path} [{mode}] has {len(ref)} episodes, "
                         f"episode {max(indices)} requested")
    diff = max(float(np.linalg.norm(np.asarray(ref[i]["trajectory"][0][:3], dtype=np.float64)
                                    - x0_batch[i, :3])) for i in indices)
    return ref, mode, diff


def load_solution_library(path, c, max_fail_frac=0.0):
    """Load previously solved OCPs to use as cold-start guesses.

    Layouts accepted:
      a) OCP dataset {"results_dict": [{"status","T","X","dV"}, ...]}.
         Open-loop plans: the best possible guess.
      b) run_data.pkl of this script ({"episodes": [...]}) or a legacy
         all_results.pkl ({"all_results": {mode: [...]}}). Realised closed-
         loop trajectories: a worse guess, supported as a fallback.

    Only converged / successful entries are kept. Initial states are stored
    non-dimensional so position and velocity weigh comparably in the
    nearest-neighbour distance.

    Returns (X0 (m, 6) non-dimensional, entries[j] = {"X": (n+1,6), "dV": (n,3)}).
    """
    r_ad, v_ad = c[1], c[2]
    d = _load_pickle(path)
    x0_list, entries, n_drop = [], [], 0

    def _add(X, dV):
        x0_list.append(np.concatenate([X[0, :3] / r_ad, X[0, 3:6] / v_ad]))
        entries.append({"X": X, "dV": dV})

    if isinstance(d, dict) and "results_dict" in d:
        source = "OCP dataset (open loop)"
        for entry in d["results_dict"]:
            if entry.get("status", "success") != "success":
                n_drop += 1
                continue
            X = np.atleast_2d(np.asarray(entry["X"], dtype=np.float64))
            dV = np.atleast_2d(np.asarray(entry["dV"], dtype=np.float64))
            if len(X) < 2 or len(dV) == 0:
                n_drop += 1
                continue
            _add(X, dV)
    else:
        source = "closed-loop MPC run"
        if "episodes" in d:
            episodes = d["episodes"]
        else:
            all_res = d.get("all_results", d)
            episodes = [ep for mode in all_res for ep in all_res[mode]]
        for ep in episodes:
            if ep.get("outcome") != "success":
                n_drop += 1
                continue
            status = list(ep.get("solver_status") or [])
            if status:
                fail = sum(1 for st in status if not is_success(st)) / len(status)
                if fail > max_fail_frac:
                    n_drop += 1
                    continue
            X = np.asarray(ep["trajectory"], dtype=np.float64)
            if "dv_vnb" in ep:
                dV = np.asarray(ep["dv_vnb"], dtype=np.float64).reshape(-1, 3)
            else:
                dv = np.asarray(ep.get("dv_total_series", []), dtype=np.float64)
                dV = np.zeros((len(dv), 3))
                dV[:, 0] = dv
            if len(X) < 2 or len(dV) == 0:
                n_drop += 1
                continue
            _add(X, dV)

    if not entries:
        raise SystemExit(f"no usable solutions in {path} (all filtered out)")
    print(f"[library] {len(entries)} solutions from {source}: {path} "
          f"({n_drop} dropped)", flush=True)
    return np.asarray(x0_list), entries


# ===========================================================================
# MPC
# ===========================================================================
class MPCController:
    """Shrinking-horizon MPC with warm start, ellipsoid homotopy and
    per-solve diagnostics (exposed in self.last_info after every solve).

    Self-contained: plans with integrate3D_dV (natural guess) and
    optimize_safeorbit2 over an integrator3D_casadi model of its own copy of
    the asteroid.
    """

    def __init__(self, optim_dict, dt_step, N, nominal_dict,
                 model="nominal", guess_mode="natural",
                 lib_X0=None, lib_entries=None, store_plans=True):
        self.optim_dict = dict(optim_dict)
        self.dt_step = dt_step
        self.N = N                      # current horizon (shrinks every step)
        self.N_total = N                # full episode horizon
        self.asteroid_dict = copy.deepcopy(nominal_dict)

        self.model = model
        self.guess_mode = guess_mode
        self.store_plans = store_plans

        self.lib_X0 = lib_X0
        self.lib_entries = lib_entries
        self.ref_traj = None
        self.ref_dv = None

        self._intg = None
        self._last_sol = None
        self._lib_match = None
        self.last_info = {}

    # ------------------------------------------------------------------
    def set_episode_model(self, episode_dict):
        """Oracle: plan with this episode's asteroid. Nominal: no-op."""
        if self.model != "oracle":
            return
        self.asteroid_dict = copy.deepcopy(episode_dict)
        self._intg = None

    def reset(self, ref_traj=None, ref_dv=None):
        self._last_sol = None
        self._lib_match = None
        self.last_info = {}
        self.N = self.N_total
        self.ref_traj = None if ref_traj is None else np.asarray(ref_traj, dtype=np.float64)
        self.ref_dv = None if ref_dv is None else np.asarray(ref_dv, dtype=np.float64)

    def _integrator(self):
        # Built once per model rather than once per solve: it only depends on
        # the mascon and the scaling, and on board it would be built once too.
        if self._intg is None:
            self._intg = integrator3D_casadi(self.asteroid_dict,
                                             c=self.optim_dict["c"])
        return self._intg

    def _current_step(self):
        return self.N_total - self.N

    # ------------------------------------------------------------------
    def _guess_natural(self, x0):
        """Free dynamics from x0 with no impulses."""
        n = self.N + 1
        dV_dict = {"T": np.linspace(0.0, self.N * self.dt_step, n),
                   "dV": np.zeros((self.N, 3))}
        # integrate3D_dV mutates x0 in place -> pass a copy.
        _, X, _, _, _ = integrate3D_dV(np.array(x0, dtype=np.float64),
                                       self.asteroid_dict, dV_dict)
        return {"T": np.arange(n) * self.dt_step, "X": X,
                "dV": np.zeros((self.N, 3))}

    def _guess_ref(self, x0):
        """Reference trajectory as guess. DIAGNOSTIC ONLY."""
        if self.ref_traj is None or self.ref_dv is None or len(self.ref_dv) == 0:
            return self._guess_natural(x0)
        n, k = self.N + 1, self._current_step()
        X = _pad_to(self.ref_traj[k:k + n], n)
        X[0] = x0
        dv = _pad_to(self.ref_dv[k:k + self.N].reshape(-1, 1), self.N).ravel()
        dv_cap = float(np.max(DVMAX_MPC))
        if dv_cap > 0:
            dv = np.clip(dv, -dv_cap, dv_cap)
        dV = np.zeros((self.N, 3))
        dV[:, 0] = dv
        return {"T": np.arange(n) * self.dt_step, "X": X, "dV": dV}

    def _guess_nearest(self, x0):
        """Closest stored OCP solution, in non-dimensional state space."""
        if self.lib_X0 is None or len(self.lib_X0) == 0:
            return self._guess_natural(x0)
        r_ad, v_ad = self.optim_dict["c"][1], self.optim_dict["c"][2]
        x0_ad = np.concatenate([np.asarray(x0[:3]) / r_ad,
                                np.asarray(x0[3:6]) / v_ad])
        dist = np.linalg.norm(self.lib_X0 - x0_ad, axis=1)
        j = int(np.argmin(dist))
        self._lib_match = {"idx": j, "dist": float(dist[j])}

        n = self.N + 1
        X = _pad_to(self.lib_entries[j]["X"], n)
        X[0] = x0
        dV = _pad_to(self.lib_entries[j]["dV"], self.N)
        cap = np.asarray(DVMAX_MPC, dtype=np.float64)
        if cap.ndim == 0:
            cap = np.full(3, float(cap))
        dV = np.clip(dV, -cap, cap)
        return {"T": np.arange(n) * self.dt_step, "X": X, "dV": dV}

    def _build_initguess(self, x0):
        """Return (guess, source)."""
        n = self.N + 1
        if self._last_sol is not None:
            X = _pad_to(np.vstack([self._last_sol["X"][1:],
                                   self._last_sol["X"][-1:]]), n)
            dV = _pad_to(np.vstack([self._last_sol["dV"][1:],
                                    self._last_sol["dV"][-1:]]), self.N)
            X[0] = x0
            return {"T": np.arange(n) * self.dt_step, "X": X, "dV": dV}, "warm"
        if self.guess_mode == "ref":
            return self._guess_ref(x0), "ref"
        if self.guess_mode == "nearest":
            return self._guess_nearest(x0), "nearest"
        return self._guess_natural(x0), "natural"

    # ------------------------------------------------------------------
    def solve(self, x):
        od = dict(self.optim_dict)
        od["x0"] = np.asarray(x, dtype=np.float64)
        od["dvmax"] = DVMAX_MPC
        od["tf"] = self.N * self.dt_step
        r_ad = od["c"][1]
        od["shell"] = dict(od["shell"])
        od["shell"]["bounds"] = tuple(
            np.asarray(od["shell"]["bounds"], dtype=np.float64) / r_ad)
        intg = self._integrator()

        guess, source = self._build_initguess(od["x0"])
        q = guess_quality(guess["X"], od["ellip_axes"])
        info = {
            "guess_source":      source,
            "guess_q":           q,
            "library_match":     self._lib_match if source == "nearest" else None,
            "homotopy_used":     q["q_min"] < 1.0,
            "homotopy_stages":   [],
            "status":            None,
            "solver_status_raw": None,
            "action_source":     None,
            "t_final_solve":     None,
            "solver_extra":      {},
            "plan":              None,
        }
        self.last_info = info

        g = guess
        if info["homotopy_used"]:
            f0 = HOMOTOPY_MARGIN * np.sqrt(q["q_min"])
            for f in (f0, 0.5 * (f0 + 1.0)):
                if not 0.0 < f < 1.0:
                    continue
                od_h = dict(od)
                od_h["ellip_axes"] = np.asarray(od["ellip_axes"], dtype=np.float64) * f
                od_h["max_iter"] = MAX_ITER_HOMOT
                t0 = time.perf_counter()
                r = optimize_safeorbit2(g, od_h, intg)
                info["homotopy_stages"].append({
                    "factor": float(f), "status": str(r["status"]),
                    "time_s": time.perf_counter() - t0})
                if r["status"] != "success":
                    break                       # never chain a corrupted guess
                g = {"T": r["T"], "X": r["X"], "dV": r["dV"]}

        od["max_iter"] = MAX_ITER_FINAL
        t0 = time.perf_counter()
        result = optimize_safeorbit2(g, od, intg)
        info["t_final_solve"] = time.perf_counter() - t0
        info["solver_status_raw"] = str(result["status"])
        info["solver_extra"] = {k: v for k, v in result.items()
                                if k not in ("T", "X", "dV", "status")
                                and np.isscalar(v)}

        ok = result["status"] == "success"
        if ok:
            info["status"] = "success_homotopia" if info["homotopy_used"] else "success"
            self._last_sol = result
            if self.store_plans:
                info["plan"] = {"T": np.array(result["T"], dtype=np.float64),
                                "X": np.array(result["X"], dtype=np.float64),
                                "dV": np.array(result["dV"], dtype=np.float64)}
            info["action_source"] = "solve"
            return np.array(result["dV"][0], dtype=np.float64)

        info["status"] = "fail"
        # On failure, apply the next impulse of the last valid plan instead of
        # coasting, and shift that plan so the next warm start stays aligned.
        if self._last_sol is not None and len(self._last_sol["dV"]) > 1:
            dv = np.array(self._last_sol["dV"][1], dtype=np.float64)
            self._last_sol = {"T": self._last_sol["T"][1:],
                              "X": self._last_sol["X"][1:],
                              "dV": self._last_sol["dV"][1:],
                              "status": "reused"}
            info["action_source"] = "reused_plan"
            return dv

        self._last_sol = None
        info["action_source"] = "zero"
        return np.zeros(3)


def make_mpc_policy(mpc, N):
    """Adapt MPCController to the rollout interface policy(k, x) -> (dv, info).

    The OCP works in VNB axes with dvmax = [dv, 0, 0]: the first component of
    the planned impulse is the tangential command that gets executed.
    """
    def policy(k, x):
        mpc.N = N - k                               # shrinking horizon
        dV_vnb = np.asarray(mpc.solve(x), dtype=np.float64).reshape(3)
        info = dict(mpc.last_info)
        info["dv_vnb"] = dV_vnb
        return float(dV_vnb[0]), info
    return policy


def make_step_logger(ep_idx, verbose_steps):
    """Print the first steps of each episode and every failed solve."""
    def on_step(k, x, dv_cmd, info, t_ctrl):
        if k >= verbose_steps and is_success(info["status"]):
            return
        q = info["guess_q"]
        flag = "  <<< GUESS INSIDE THE ELLIPSOID" if q["viola_elip"] else ""
        lib = info["library_match"]
        lib_s = f" | lib#{lib['idx']} d={lib['dist']:.4f}" if lib else ""
        print(f"    [ep{ep_idx} s{k:02d}] r={np.linalg.norm(x[:3]) / 1e3:6.3f} km "
              f"dv={dv_cmd:+.4e} t={t_ctrl:6.1f}s "
              f"status={info['status']}({info['action_source']}) | "
              f"guess={info['guess_source']} q_min={q['q_min']:.4f} "
              f"r=[{q['r_min_km']:.2f}, {q['r_max_km']:.2f}] km"
              f"{flag}{lib_s}", flush=True)
    return on_step


# ===========================================================================
# Episode record
# ===========================================================================
def finalize_episode(roll):
    """Merge the rollout and the per-step MPC diagnostics into one dict."""
    infos = roll["infos"]
    dv_vnb = np.asarray([inf["dv_vnb"] for inf in infos],
                        dtype=np.float64).reshape(-1, 3)
    dv_tan = roll["dv_cmd"].copy()
    abs_tan = np.abs(dv_tan)
    ct = roll["compute_times"]
    status = [inf["status"] for inf in infos]
    action_source = [inf["action_source"] for inf in infos]
    q_min = np.array([inf["guess_q"]["q_min"] for inf in infos], dtype=np.float64)
    plans = [inf["plan"] for inf in infos]
    dv_exec = np.linalg.norm(roll["dv_exec_planet"], axis=1)

    return {
        "outcome":           roll["outcome"],
        "terminated":        roll["terminated"],
        "n_steps":           roll["n_steps"],

        # time series
        "times":             roll["times"],
        "trajectory":        roll["trajectory"],
        "dv_vnb":            dv_vnb,
        "dv_exec_vnb":       roll["dv_exec_vnb"],
        "dv_exec_planet":    roll["dv_exec_planet"],
        "fired":             roll["fired"],
        "dv_base_series":    np.zeros_like(dv_tan),   # the MPC has no base part
        "dv_res_series":     dv_tan.copy(),
        "dv_total_series":   dv_tan.copy(),
        "action_res_norm":   dv_tan.copy(),
        "compute_times":     ct,
        "t_final_solve":     np.array([inf["t_final_solve"] for inf in infos],
                                      dtype=np.float64),

        # solver diagnostics, one entry per step
        "solver_status":     status,
        "solver_status_raw": [inf["solver_status_raw"] for inf in infos],
        "action_source":     action_source,
        "guess_source":      [inf["guess_source"] for inf in infos],
        "guess_q":           [inf["guess_q"] for inf in infos],
        "guess_q_min":       q_min,
        "homotopy_stages":   [inf["homotopy_stages"] for inf in infos],
        "solver_extra":      [_picklable(inf["solver_extra"]) for inf in infos],
        "library_match":     infos[0]["library_match"] if infos else None,
        "plans":             plans if any(p is not None for p in plans) else None,

        # aggregates
        "final_state":       roll["trajectory"][-1].copy(),
        "final_radius_m":    float(np.linalg.norm(roll["trajectory"][-1, :3])),
        "dv_base_sum":       0.0,
        "dv_res_sum":        float(abs_tan.sum()),
        "dv_total_sum":      float(abs_tan.sum()),
        "dv_exec_sum":       float(dv_exec.sum()),
        "dv_vnb_norm_sum":   float(np.linalg.norm(dv_vnb, axis=1).sum()),
        "steps_acted_base":  0,
        "steps_acted_res":   int(np.count_nonzero(abs_tan > RES_ACTION_DEADBAND)),
        "steps_acted":       int(np.count_nonzero(roll["fired"])),
        "mean_abs_dv_res_ep":    float(abs_tan.mean()) if abs_tan.size else 0.0,
        "mean_signed_dv_res_ep": float(dv_tan.mean()) if dv_tan.size else 0.0,
        "ctrl_time_sum":     float(ct.sum()) if ct.size else 0.0,
        "ctrl_time_mean":    float(ct.mean()) if ct.size else 0.0,
        "ctrl_time_median":  float(np.median(ct)) if ct.size else 0.0,
        "ctrl_time_max":     float(ct.max()) if ct.size else 0.0,
        "n_solver_fail":     int(sum(1 for s in status if not is_success(s))),
        "n_homotopia_ep":    int(sum(1 for s in status if s == "success_homotopia")),
        "n_guess_infeasible": int(np.count_nonzero(q_min < 1.0)),
        "n_reused_plan":     int(action_source.count("reused_plan")),
        "n_zero_action":     int(action_source.count("zero")),
    }


# ===========================================================================
# Summary
# ===========================================================================
def compute_stats(episodes):
    counts = {"success": 0, "partial": 0, "collision": 0, "escape": 0}
    for ep in episodes:
        counts[ep["outcome"]] = counts.get(ep["outcome"], 0) + 1
    n = len(episodes)

    def _mean(key, eps=episodes):
        return float(np.mean([ep[key] for ep in eps])) if eps else float("nan")

    succ = [ep for ep in episodes if ep["outcome"] == "success"]
    stats = {
        **counts,
        "n_episodes":        n,
        "success_rate":      counts["success"] / max(n, 1),
        "mean_dv_total":     _mean("dv_total_sum"),
        "mean_dv_exec":      _mean("dv_exec_sum"),
        "mean_dv_base":      _mean("dv_base_sum"),
        "mean_dv_res":       _mean("dv_res_sum"),
        "mean_dv_success":   _mean("dv_total_sum", succ),
        "mean_acts_total":   _mean("steps_acted"),
        "mean_acts_res":     _mean("steps_acted_res"),
        "mean_ctrl_time":    _mean("ctrl_time_mean"),
        "max_ctrl_time":     float(max((ep["ctrl_time_max"] for ep in episodes), default=0.0)),
        "solver_fail_steps": sum(ep["n_solver_fail"] for ep in episodes),
        "homotopy_steps":    sum(ep["n_homotopia_ep"] for ep in episodes),
        "guess_infeasible":  sum(ep["n_guess_infeasible"] for ep in episodes),
        "reused_plan_steps": sum(ep["n_reused_plan"] for ep in episodes),
        "zero_action_steps": sum(ep["n_zero_action"] for ep in episodes),
        "total_steps":       sum(ep["n_steps"] for ep in episodes),
    }
    cross = {}
    for ep in episodes:
        if ep.get("ref_outcome") is not None:
            key = f"{ep['ref_outcome']}->{ep['outcome']}"
            cross[key] = cross.get(key, 0) + 1
    stats["ref_to_mpc_outcome"] = cross
    return stats


def print_stats(stats, model, guess):
    den = max(stats["total_steps"], 1)

    def _pct(k):
        return f"{stats[k]}/{stats['total_steps']} ({100.0 * stats[k] / den:.1f}%)"

    print(f"\n  === model={model}  guess={guess}  "
          f"episodes={stats['n_episodes']} ===")
    print(f"  success={stats['success']}  partial={stats['partial']}  "
          f"collision={stats['collision']}  escape={stats['escape']}  "
          f"(success rate {100.0 * stats['success_rate']:.1f}%)")
    print(f"  guess inside ellipsoid    : {_pct('guess_infeasible')}")
    print(f"  solved with homotopy      : {_pct('homotopy_steps')}")
    print(f"  steps with solver failure : {_pct('solver_fail_steps')}")
    print(f"    -> reused previous plan : {_pct('reused_plan_steps')}")
    print(f"    -> zero impulse         : {_pct('zero_action_steps')}")
    print(f"  mean dv per episode       : {stats['mean_dv_total']:.4f} m/s commanded, "
          f"{stats['mean_dv_exec']:.4f} m/s executed "
          f"(successful: {stats['mean_dv_success']:.4f} m/s)")
    print(f"  mean time per solve       : {stats['mean_ctrl_time']:.2f} s "
          f"(max {stats['max_ctrl_time']:.1f} s)")
    if stats["ref_to_mpc_outcome"]:
        print("\n  ref_outcome -> mpc_outcome:")
        for key, n in sorted(stats["ref_to_mpc_outcome"].items()):
            print(f"    {key:<24} {n}")
    sys.stdout.flush()


# ===========================================================================
# CLI and run directory
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Closed-loop MPC over the shared batch of initial conditions. "
                    "Saves everything, plots nothing.")
    p.add_argument("--scenarios", default=DEFAULT_SCENARIOS,
                   help=f"initial-conditions pickle (default: {DEFAULT_SCENARIOS})")
    p.add_argument("--ep-start", type=int, default=0,
                   help="first episode of the batch to run")
    p.add_argument("--n-episodes", type=int, default=None,
                   help="number of episodes (default: all from --ep-start)")
    p.add_argument("--noise-seed", type=int, default=123,
                   help="episode i seeds its actuation errors with noise_seed*1e6 + i")
    p.add_argument("--model", default="nominal", choices=["nominal", "oracle"])
    p.add_argument("--guess", default="nearest",
                   choices=["natural", "nearest", "ref"],
                   help="cold-start guess; 'ref' is DIAGNOSTIC ONLY")
    p.add_argument("--guess-library",
                   default=os.path.join("results", "direct_results_VNBTangential_v2.pkl"),
                   help="solution library for --guess nearest (OCP dataset, "
                        "run_data.pkl or legacy all_results.pkl; defaults to "
                        "the tangential OCP results)")
    p.add_argument("--library-max-fail", type=float, default=0.0,
                   help="max fraction of failed solves for a closed-loop "
                        "episode to enter the library")
    p.add_argument("--ref-results", default=None,
                   help="optional reference run on the same batch (run_data.pkl "
                        "or all_results.pkl): needed for --guess ref, and adds a "
                        "ref->mpc outcome table")
    p.add_argument("--ref-mode", default=None,
                   help="mode inside a legacy all_results.pkl (base_only/total/...)")
    p.add_argument("--out-dir", default=None,
                   help="run directory (default: <out-root>/mpc_<model>_<guess>_<timestamp>)")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--resume", action="store_true",
                   help="continue a run in --out-dir, skipping saved episodes")
    p.add_argument("--no-plans", action="store_true",
                   help="do not store the plan of every solve (largest part of the output)")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--verbose-steps", type=int, default=5,
                   help="steps logged in detail at the start of each episode "
                        "(failed steps are always logged)")
    args = p.parse_args(argv)

    if args.guess == "nearest" and not args.guess_library:
        p.error("--guess nearest requires --guess-library")
    if args.guess == "ref" and not args.ref_results:
        p.error("--guess ref requires --ref-results")
    if args.resume and not args.out_dir:
        p.error("--resume requires --out-dir")
    return args


def prepare_run_dir(args):
    if args.out_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(args.out_root,
                                    f"mpc_{args.model}_{args.guess}_{stamp}")
    ep_dir = os.path.join(args.out_dir, "episodes")
    cfg_path = os.path.join(args.out_dir, "run_config.json")

    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)["args"]
        if not args.resume:
            raise SystemExit(f"{args.out_dir} already holds a run; use --resume "
                             f"to continue it or choose another --out-dir")
        diff = {k: (old.get(k), getattr(args, k)) for k in RESUME_KEYS
                if old.get(k) != getattr(args, k)}
        if diff:
            lines = "\n".join(f"  {k}: saved={a!r} now={b!r}" for k, (a, b) in diff.items())
            raise SystemExit(f"cannot resume, arguments differ from the saved run:\n{lines}")

    os.makedirs(ep_dir, exist_ok=True)
    if not os.path.exists(cfg_path):
        _dump_json({"args": vars(args), "argv": sys.argv,
                    "created": datetime.datetime.now().isoformat()}, cfg_path)
    return args.out_dir, ep_dir


def episode_path(ep_dir, i):
    return os.path.join(ep_dir, f"ep_{i:05d}.pkl")


# ===========================================================================
# Main
# ===========================================================================
def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    started = datetime.datetime.now().isoformat()
    out_dir, ep_dir = prepare_run_dir(args)

    # --- Scenarios ---------------------------------------------------------
    batch, scen_meta = load_batch(args.scenarios)
    n_batch = len(batch["x0"])
    x0_batch = np.asarray(batch["x0"], dtype=np.float64)
    if not 0 <= args.ep_start < n_batch:
        raise SystemExit(f"--ep-start {args.ep_start} outside the batch (0..{n_batch - 1})")
    n_ep = n_batch - args.ep_start if args.n_episodes is None else args.n_episodes
    indices = list(range(args.ep_start, min(args.ep_start + n_ep, n_batch)))
    print(f"[data]  {args.scenarios}: {n_batch} episodes "
          f"(master_seed={scen_meta.get('master_seed')}), running "
          f"{indices[0]}..{indices[-1]} ({len(indices)})", flush=True)

    # --- Nominal scenario --------------------------------------------------
    asteroid_dict, optim_dict = build_scenario()
    nominal = copy.deepcopy(asteroid_dict)
    N = int(optim_dict["N"])
    if "rmax" not in optim_dict:
        raise SystemExit("optim_dict has no 'rmax': needed for the escape check")

    dt_sim = control_dt(optim_dict)
    if not np.isclose(dt_sim, DT_STEP, rtol=1e-9, atol=0.0):
        print(f"[warning] tf/N = {dt_sim} s differs from DT_STEP = {DT_STEP} s: "
              f"the MPC and the simulation use different step lengths", flush=True)

    nominal_check = check_nominal_mascon(nominal, scen_meta)
    if not nominal_check["ok"]:
        print(f"[warning] nominal mascon of build_scenario() differs from the "
              f"one the batch was generated with: {nominal_check}", flush=True)

    # --- Optional inputs ---------------------------------------------------
    lib_X0 = lib_entries = None
    if args.guess == "nearest":
        lib_X0, lib_entries = load_solution_library(
            args.guess_library, optim_dict["c"], args.library_max_fail)

    ref = ref_mode = ref_ic_diff = None
    if args.ref_results:
        ref, ref_mode, ref_ic_diff = load_reference(
            args.ref_results, args.ref_mode, x0_batch, indices)
        print(f"[ref]   {args.ref_results} [{ref_mode}]: max initial-state "
              f"difference vs batch {ref_ic_diff:.3e} m", flush=True)
        if ref_ic_diff > REF_IC_TOL_M:
            if args.guess == "ref":
                raise SystemExit("the reference run was not made on this batch; "
                                 "--guess ref would be meaningless")
            print("[warning] the reference run was not made on this batch; "
                  "ignoring it", flush=True)
            ref = None

    mpc = MPCController(optim_dict, dt_step=DT_STEP, N=N,
                     nominal_dict=nominal, model=args.model,
                     guess_mode=args.guess, lib_X0=lib_X0,
                     lib_entries=lib_entries, store_plans=not args.no_plans)

    print(f"[ocp]   rmin={optim_dict.get('rmin')} rmax={optim_dict.get('rmax')} "
          f"shell={optim_dict['shell']['bounds']}", flush=True)
    print(f"[ocp]   ellip_axes={optim_dict['ellip_axes']}  dvmax={DVMAX_MPC}  "
          f"N={N}  dt={DT_STEP}", flush=True)
    print(f"[sim]   sigma_dv={SIGMA_DV_ABS} m/s  sigma_theta={SIGMA_THETA} rad  "
          f"thruster off below {DV_OFF_THRESHOLD} m/s", flush=True)
    print(f"[model] {args.model}  |  nominal muM = {nominal['mascon']['muM']}", flush=True)
    if args.model == "oracle":
        print("[model] *** ORACLE: the MPC sees each episode's exact mascon. "
              "Optimistic upper bound, NOT a realistic result. ***", flush=True)
    print(f"[guess] {args.guess}", flush=True)
    if args.guess == "ref":
        print("[guess] *** REF: diagnostic only, do not report these numbers. ***",
              flush=True)
    print(f"[out]   {out_dir}", flush=True)

    # --- Solver warmup (CasADi compilation; not measured) -----------------
    if not args.no_warmup:
        print("[warmup] ...", flush=True)
        mpc.set_episode_model(nominal)
        mpc.reset()
        _ = mpc.solve(x0_batch[indices[0]])
        mpc.reset()
        print("[warmup] done", flush=True)

    # --- Main loop ---------------------------------------------------------
    n_skipped = 0
    for count, i in enumerate(indices):
        path = episode_path(ep_dir, i)
        if os.path.exists(path):
            n_skipped += 1
            continue

        x0_i = x0_batch[i]
        sim_dict = episode_asteroid(nominal, batch, i)
        noise_seed_i = int(args.noise_seed * NOISE_SEED_STRIDE + i)
        rng = np.random.default_rng(noise_seed_i)

        mpc.set_episode_model(sim_dict)
        ref_traj = ref_dv = None
        if ref is not None and args.guess == "ref":
            ref_traj = ref[i]["trajectory"]
            ref_dv = ref[i].get("dv_total_series", [])
        mpc.reset(ref_traj=ref_traj, ref_dv=ref_dv)

        ref_outcome = ref[i]["outcome"] if ref is not None else None
        print(f"  ep {i:>4d}  [{count + 1}/{len(indices)}]  "
              f"r0={np.linalg.norm(x0_i[:3]) / 1e3:6.3f} km"
              + (f"  ref_outcome={ref_outcome}" if ref_outcome else ""), flush=True)

        t_ep = time.perf_counter()
        roll = run_episode(make_mpc_policy(mpc, N), x0_i, sim_dict, optim_dict,
                           rng, on_step=make_step_logger(i, args.verbose_steps))
        ep = finalize_episode(roll)

        ep["ep_idx"] = i
        ep["noise_seed"] = noise_seed_i
        ep["ref_outcome"] = ref_outcome
        ep["wall_time_s"] = time.perf_counter() - t_ep
        ep["scenario"] = {
            "x0":   x0_i.copy(),
            "muM":  np.array(batch["muM"][i], dtype=np.float64),
            "xyzM": np.array(batch["xyzM"][i], dtype=np.float64),
            **{k: (float(batch[k][i]) if k in batch else None)
               for k in ("eps", "rho")},
        }
        _dump_pickle(ep, path)

        print(f"  -> outcome={ep['outcome']}  dv={ep['dv_total_sum']:.4f} m/s  "
              f"fails={ep['n_solver_fail']}/{ep['n_steps']}  "
              f"homotopy={ep['n_homotopia_ep']}  "
              f"guess_inside={ep['n_guess_infeasible']}  "
              f"wall={ep['wall_time_s']:.0f}s", flush=True)

    if n_skipped:
        print(f"[resume] {n_skipped} episodes were already saved and skipped", flush=True)

    # --- Assemble ----------------------------------------------------------
    # run_data.pkl gathers EVERY episode saved in the directory, not only this
    # process's range, so a batch split across processes ends up complete in
    # whichever process finishes last.
    missing = [i for i in indices if not os.path.exists(episode_path(ep_dir, i))]
    if missing:
        raise SystemExit(f"{len(missing)} episodes missing in {ep_dir}: {missing[:10]}...")
    saved = sorted(int(f[3:-4]) for f in os.listdir(ep_dir)
                   if f.startswith("ep_") and f.endswith(".pkl"))
    episodes = [_load_pickle(episode_path(ep_dir, i)) for i in saved]
    if len(saved) < n_batch:
        print(f"[assemble] {len(saved)}/{n_batch} batch episodes in {ep_dir}; "
              f"run the remaining ones with --resume to complete the set", flush=True)
    stats = compute_stats(episodes)
    print_stats(stats, args.model, args.guess)

    try:
        optim_saved = copy.deepcopy(optim_dict)
    except Exception:
        optim_saved = optim_dict
    meta = {
        "schema_version": SCHEMA_VERSION,
        "script":         "mpc",
        "git_commit":     get_git_commit(),
        "started":        started,
        "finished":       datetime.datetime.now().isoformat(),
        "elapsed_s":      time.time() - t_start,
        "host":           socket.gethostname(),
        "python":         platform.python_version(),
        "numpy":          np.__version__,
        "argv":           sys.argv,
        "resumed":        bool(n_skipped),
    }
    config = {
        "args":             vars(args),
        "episode_indices":  saved,
        "constants": {
            "DT_STEP": DT_STEP, "SHELL_MIN": SHELL_MIN, "SHELL_MAX": SHELL_MAX,
            "DVMAX_MPC": np.asarray(DVMAX_MPC).tolist(),
            "MAX_ITER_FINAL": MAX_ITER_FINAL, "MAX_ITER_HOMOT": MAX_ITER_HOMOT,
            "HOMOTOPY_MARGIN": HOMOTOPY_MARGIN,
            "RES_ACTION_DEADBAND": RES_ACTION_DEADBAND,
            "NOISE_SEED_STRIDE": NOISE_SEED_STRIDE,
        },
        "simulation": {
            "dt_s": dt_sim,
            "sigma_dv_abs": SIGMA_DV_ABS,
            "sigma_theta": SIGMA_THETA,
            "dv_off_threshold": DV_OFF_THRESHOLD,
            "rmax": optim_dict["rmax"],
            "ellip_axes": np.asarray(optim_dict["ellip_axes"]).tolist(),
        },
        "optim_dict":       _picklable(optim_saved),
        "nominal_mascon":   {"muM": np.array(nominal["mascon"]["muM"], dtype=np.float64),
                             "xyzM": np.array(nominal["mascon"]["xyzM"], dtype=np.float64)},
        "nominal_check":    nominal_check,
        "scenarios_meta":   scen_meta,
        "library":          ({"path": args.guess_library, "n_entries": len(lib_entries)}
                             if lib_entries is not None else None),
        "reference":        ({"path": args.ref_results, "mode": ref_mode,
                              "ic_max_diff_m": ref_ic_diff, "used": ref is not None}
                             if args.ref_results else None),
    }
    run_data = {
        "schema_version": SCHEMA_VERSION,
        "meta":           meta,
        "config":         config,
        "summary":        stats,
        "episodes":       episodes,
        "initial_states": np.array([ep["trajectory"][0] for ep in episodes]),
        "all_results":    {"mpc": episodes},       # legacy alias, no extra size
    }
    _dump_pickle(run_data, os.path.join(out_dir, "run_data.pkl"))
    _dump_json({"meta": meta,
                "config": {k: config[k] for k in ("args", "constants", "simulation",
                                                  "nominal_check", "library", "reference")},
                "scenarios": {"path": args.scenarios,
                              "master_seed": scen_meta.get("master_seed"),
                              "n_saved": len(saved), "n_batch": n_batch},
                "summary": stats},
               os.path.join(out_dir, "summary.json"))

    print(f"\n[done] {os.path.join(out_dir, 'run_data.pkl')}  "
          f"({time.time() - t_start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()