"""
Closed-loop simulation shared by every controller (MPC, regressor, natural
dynamics and, once migrated, the RL environment), so all methods are simulated
with exactly the same physics.

One control step:
  1. Execute the commanded tangential impulse. Actuation errors (magnitude
     N(0, SIGMA_DV_ABS) and pointing N(0, SIGMA_THETA) per VNB axis) are only
     applied when the thruster fires, i.e. |dv| > DV_OFF_THRESHOLD.
  2. Propagate tf/N seconds with src.propagators.integrate3D_scipy
     (gravity + mascon in the rotating frame).
  3. Collision with the ellipsoid and escape beyond rmax are detected
     CONTINUOUSLY with the events of src.propagators.events: the episode stops
     at the crossing instant, not at the end of the step.

Physics taken from OrbitEnv.step() of the previous repository, with two
deliberate changes: no actuation error on a zero command (the old env kicked
the spacecraft every step), and continuous rather than end-of-step
collision/escape detection.

States are DIMENSIONAL here, [x, y, z, vx, vy, vz] in m and m/s, body-fixed
frame. The conversion to the integrator's non-dimensional units happens inside
propagate().

A controller is any callable

    policy(k, x) -> (dv_cmd, info)

with k the 0-based step, x the current state (6,), dv_cmd the commanded
tangential impulse in m/s and info a dict of diagnostics, stored as is.
"""

import time

import numpy as np

from src.config import SIGMA_DV_ABS, SIGMA_THETA, SHELL_MIN, SHELL_MAX
from src.utils.frames import vnb_to_planetary, small_angle_rotation
from src.propagators.integrator_scipy import integrate3D_scipy
from src.propagators.events import make_ellipsoid_event, make_rmax_event


DV_OFF_THRESHOLD = 1e-8                 # m/s; |command| at or below: thruster off

OUTCOMES = ("success", "partial", "collision", "escape")
TERMINATIONS = ("collision", "escape")  # same order as make_events()


# ===========================================================================
# One control step
# ===========================================================================
def control_dt(optim_dict):
    """Duration of one control step, s."""
    return optim_dict["tf"] / optim_dict["N"]


def make_events(optim_dict):
    """Collision and escape events, in the integrator's non-dimensional units."""
    r_ad = optim_dict["c"][1]
    axes_ad = np.asarray(optim_dict["ellip_axes"], dtype=np.float64) / r_ad
    return [make_ellipsoid_event(axes_ad),
            make_rmax_event(optim_dict["rmax"] / r_ad)]


def execute_impulse(dv_cmd, x, omega, rng, dv_off=DV_OFF_THRESHOLD):
    """Turn a commanded tangential impulse into the executed one.

    Returns:
        dv_planet (3,)  executed impulse in the body-fixed frame, m/s
        dv_vnb    (3,)  executed impulse in VNB, m/s
        fired     bool  whether the thruster fired

    Draw order when firing (magnitude, then the three angles) is the one of
    the old env.
    """
    dv_cmd = float(dv_cmd)
    if abs(dv_cmd) <= dv_off:
        return np.zeros(3), np.zeros(3), False

    dv_mag = dv_cmd + rng.normal(0.0, SIGMA_DV_ABS)
    dtheta = rng.normal(0.0, SIGMA_THETA, 3)
    dv_vnb = small_angle_rotation(dtheta) @ np.array([dv_mag, 0.0, 0.0])
    dv_planet = vnb_to_planetary(x[:3], x[3:6], omega) @ dv_vnb
    return (np.asarray(dv_planet, dtype=np.float64),
            np.asarray(dv_vnb, dtype=np.float64), True)


def propagate(x, asteroid_dict, optim_dict, events=None):
    """Free dynamics over one control step, stopping at a terminal event.

    Returns (x_end (6,) m and m/s, t_end s, termination) where termination is
    'collision', 'escape' or None; on an event, x_end and t_end are the state
    and time of the crossing.
    """
    t_ad, r_ad, v_ad = optim_dict["c"]
    dt_ad = control_dt(optim_dict) / t_ad
    x_ad = np.concatenate([np.asarray(x[:3], dtype=np.float64) / r_ad,
                           np.asarray(x[3:6], dtype=np.float64) / v_ad])

    _, X, flags, t_ev, x_ev = integrate3D_scipy(x_ad, dt_ad, asteroid_dict,
                                                c=optim_dict["c"], N=1,
                                                events=events)

    termination = next((name for name, hit in zip(TERMINATIONS, flags) if hit),
                       None)
    if termination is None:
        x_end, t_end = np.asarray(X[-1], dtype=np.float64), dt_ad
    else:
        x_end, t_end = np.asarray(x_ev, dtype=np.float64), float(t_ev)
    return (np.concatenate([x_end[:3] * r_ad, x_end[3:6] * v_ad]),
            t_end * t_ad, termination)


def simulate_step(x, dv_cmd, asteroid_dict, optim_dict, rng, events=None,
                  dv_off=DV_OFF_THRESHOLD):
    """Impulse at the start of the step, then propagation.

    Returns (x_next (6,), record) with record = {"dv_cmd", "dv_vnb",
    "dv_planet", "fired", "t_step", "termination"}.
    """
    x = np.asarray(x, dtype=np.float64)
    dv_planet, dv_vnb, fired = execute_impulse(
        dv_cmd, x, asteroid_dict["omega"], rng, dv_off)
    x_post = x.copy()
    x_post[3:6] += dv_planet
    x_next, t_step, termination = propagate(x_post, asteroid_dict,
                                            optim_dict, events)
    return x_next, {"dv_cmd": float(dv_cmd), "dv_vnb": dv_vnb,
                    "dv_planet": dv_planet, "fired": fired,
                    "t_step": t_step, "termination": termination}


def final_outcome(x):
    """Outcome of an episode that reached the horizon: inside the shell or not."""
    r = float(np.linalg.norm(x[:3]))
    return "success" if SHELL_MIN <= r <= SHELL_MAX else "partial"


# ===========================================================================
# Episode
# ===========================================================================
def run_episode(policy, x0, asteroid_dict, optim_dict, rng,
                on_step=None, dv_off=DV_OFF_THRESHOLD, on_transition=None):
    """Run one episode of at most optim_dict['N'] control steps.

    Args:
        policy:        controller, see the module docstring.
        x0:            initial state (6,), m and m/s.
        asteroid_dict: asteroid simulated in this episode (episode mascon).
        optim_dict:    scenario settings (c, N, tf, ellip_axes, rmax).
        rng:           np.random.Generator for the actuation errors.
        on_step:       optional callback(k, x, dv_cmd, info, t_ctrl).
        on_transition: optional callback(k, x, dv_cmd, info, t_ctrl, execution,
                       x_next), invoked after impulse and propagation.

    Returns a dict with the realised episode. Time series have n_steps rows,
    states and times n_steps + 1; on a collision or escape the last state and
    time are those of the crossing.
    """
    N = int(optim_dict["N"])
    events = make_events(optim_dict)
    x = np.array(x0, dtype=np.float64).reshape(6)

    t = 0.0
    states, times = [x.copy()], [t]
    dv_cmd, dv_vnb, dv_planet, fired, ctrl_times, infos = [], [], [], [], [], []
    termination = None

    for k in range(N):
        t0 = time.perf_counter()
        cmd, info = policy(k, x.copy())
        t_ctrl = time.perf_counter() - t0
        cmd = float(cmd)

        x_next, ex = simulate_step(x, cmd, asteroid_dict, optim_dict, rng,
                                   events, dv_off)
        if on_transition is not None:
            on_transition(k, x.copy(), cmd, info, t_ctrl, ex, x_next.copy())
        dv_cmd.append(cmd)
        dv_vnb.append(ex["dv_vnb"])
        dv_planet.append(ex["dv_planet"])
        fired.append(ex["fired"])
        ctrl_times.append(t_ctrl)
        infos.append(info)

        if on_step is not None:
            on_step(k, x, cmd, info, t_ctrl)

        x = x_next
        t += ex["t_step"]
        states.append(x.copy())
        times.append(t)
        termination = ex["termination"]
        if termination is not None:
            break

    return {
        "outcome":        termination or final_outcome(x),
        "terminated":     termination is not None,
        "n_steps":        len(dv_cmd),
        "times":          np.asarray(times, dtype=np.float64),
        "trajectory":     np.asarray(states, dtype=np.float64),
        "dv_cmd":         np.asarray(dv_cmd, dtype=np.float64),
        "dv_exec_vnb":    np.asarray(dv_vnb, dtype=np.float64).reshape(-1, 3),
        "dv_exec_planet": np.asarray(dv_planet, dtype=np.float64).reshape(-1, 3),
        "fired":          np.asarray(fired, dtype=bool),
        "compute_times":  np.asarray(ctrl_times, dtype=np.float64),
        "infos":          infos,
    }