"""Gymnasium environment for residual-SAC orbit station-keeping."""

import time

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from src.config import (
    COLLISION_REWARD,
    DISPERSION_M,
    MASS_VARIATION,
    OUT_OF_SHELL_REWARD,
    R_REF,
    RMAX_M,
    V_REF,
)
from src.propagators.events import make_ellipsoid_event, make_rmax_event
from src.propagators.integrator_scipy import integrate3D_scipy
from src.regressor.data_preparation.state_scaling import adimensionalize_state
from src.utils.rollout import DV_OFF_THRESHOLD, execute_impulse

SPHERE_RADIUS_M = DISPERSION_M


def _make_inner_shell_event(radius):
    """Stop propagation on the first inward crossing of the shell boundary."""
    def event(_, state):
        return np.linalg.norm(state[:3]) - radius

    event.terminal = True
    event.direction = -1
    return event


def _shell_penalty(radius, inner, outer, slope=1.0, sharpness=5.0):
    """Numerically stable scalar equivalent of the project's shell penalty."""
    x = 2 * (radius - inner) / (outer - inner) - 1
    lower, upper = -1.0, 1.0

    def softplus(value):
        return np.logaddexp(0.0, sharpness * value) / sharpness

    raw = -(
        slope * x
        - slope * softplus(x - lower)
        - slope * softplus(x - upper)
    )
    center = (lower + upper) / 2
    flat = -(
        slope * center
        - slope * softplus(center - lower)
        - slope * softplus(center - upper)
    )
    return raw - flat


class OrbitEnv(gymnasium.Env):
    """Orbit environment whose action is a residual to a frozen regressor."""

    metadata = {"render_modes": []}

    def __init__(self, asteroid_dict, optim_dict, base_policy, cls_threshold=0.5):
        super().__init__()
        self.asteroid_dict = asteroid_dict
        self.optim_dict = optim_dict
        self.base_policy = base_policy
        self.cls_threshold = cls_threshold

        obs_high = np.array([10, 10, 10, 10, 10, 10, 100, 1], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self.mu_total_nom = float(np.sum(asteroid_dict["mascon"]["muM"]))
        self.f1_nom = asteroid_dict["mascon"]["muM"][0] / self.mu_total_nom
        self.mascon_positions_nom = asteroid_dict["mascon"]["xyzM"].copy()
        self.mass_fraction_1 = self.f1_nom

    def _perturb_masses(self):
        """Perturb the two mascon masses without changing total mass or CoM."""
        eps = self.np_random.uniform(-MASS_VARIATION, MASS_VARIATION)
        f1 = self.f1_nom * (1.0 + eps)
        f2 = 1.0 - f1
        mu_masses = np.array([f1, f2]) * self.mu_total_nom

        positions = self.mascon_positions_nom.copy()
        separation = positions[0, 0] - positions[1, 0]
        positions[0, 0] = f2 * separation
        positions[1, 0] = -f1 * separation

        self.asteroid_dict["mascon"]["muM"] = mu_masses
        self.asteroid_dict["mascon"]["xyzM"] = positions
        self.asteroid_dict["xyzM"] = positions
        self.mass_fraction_1 = f1
        self.mass_fraction_2 = f2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._perturb_masses()
        self.k = 0
        self.total_dv_m_s = 0.0

        _, r_ad, v_ad = self.optim_dict["c"]
        direction = self.np_random.normal(size=3)
        direction /= np.linalg.norm(direction)
        rho = SPHERE_RADIUS_M * self.np_random.random() ** (1 / 3)

        position = np.array([0.0, R_REF, 0.0]) + rho * direction
        velocity = np.array([-V_REF, 0.0, 0.0])
        velocity -= np.cross(self.asteroid_dict["omega"], position)
        self.initial_position_m = position.copy()
        self.initial_velocity_m_s = velocity.copy()

        self.x = np.concatenate((position, velocity, np.zeros(2)))
        self.x[:3] /= r_ad
        self.x[3:6] /= v_ad
        started = time.perf_counter()
        self.x[7] = self._base_dv(self.x[:6])

        return self.x.astype(np.float32), {
            "ctrl_dt_base": time.perf_counter() - started,
            "mass_fraction_1": self.mass_fraction_1,
            "mass_fraction_2": self.mass_fraction_2,
        }

    def step(self, action):
        xk = self.x[:6].copy()
        t_ad, r_ad, v_ad = self.optim_dict["c"]
        n_steps = self.optim_dict["N"]
        tf_ad = self.optim_dict["tf"] / t_ad
        dt = tf_ad / n_steps

        residual_ad = (
            np.clip(float(action[0]), -1.0, 1.0)
            * self.optim_dict["dvmax_SAC"]
            / v_ad
        )
        dv_base_m_s = self.x[7] * v_ad
        dv_residual_m_s = residual_ad * v_ad
        dv_total_vnb = dv_base_m_s + dv_residual_m_s
        position = self.x[:3] * r_ad
        velocity = self.x[3:6] * v_ad

        dv_planet, _, thruster_fired = execute_impulse(
            dv_total_vnb,
            np.concatenate((position, velocity)),
            self.asteroid_dict["omega"],
            self.np_random,
        )
        dv_ad = dv_planet / v_ad
        xk[3:6] += dv_ad

        ellipsoid_axes_ad = self.optim_dict["ellip_axes"] / r_ad
        inner_shell_ad = self.optim_dict["shell"]["bounds"][0] / r_ad
        outer_shell_ad = self.optim_dict["shell"]["bounds"][1] / r_ad
        events = [
            make_ellipsoid_event(ellipsoid_axes_ad),
            make_rmax_event(self.optim_dict.get("rmax", RMAX_M) / r_ad),
            _make_inner_shell_event(inner_shell_ad),
            make_rmax_event(outer_shell_ad),
        ]
        _, states, event_flags, _, event_state = integrate3D_scipy(
            xk,
            dt,
            self.asteroid_dict,
            c=self.optim_dict["c"],
            N=1,
            events=events,
        )
        self.k += 1
        if event_state is not None:
            self.x[:6] = event_state
        else:
            self.x[:6] = states[-1]
        self.x[6] += np.linalg.norm(dv_ad) / n_steps
        self.total_dv_m_s += float(np.linalg.norm(dv_ad) * v_ad)

        collision = bool(event_flags[0]) if event_flags else self._check_collision()
        escape = bool(event_flags[1]) if event_flags else self._check_escape()
        out_of_shell = bool(event_flags[2] or event_flags[3]) if event_flags else False
        if event_state is None:
            out_of_shell = out_of_shell or self._is_out_of_shell(self.x[:3], r_ad)

        if out_of_shell or collision or escape:
            self.x[7] = 0.0
            ctrl_dt_base = 0.0
        else:
            started = time.perf_counter()
            self.x[7] = self._base_dv(self.x[:6])
            ctrl_dt_base = time.perf_counter() - started
        info = {"ctrl_dt_base": ctrl_dt_base}

        radius = np.linalg.norm(self.x[:3])
        shell = self.optim_dict["shell"]
        reward_shell = -_shell_penalty(
            radius,
            shell["bounds"][0] / r_ad,
            shell["bounds"][1] / r_ad,
        )
        reward_shell *= shell["gam"] / n_steps
        reward_delta_v = (
            -self.optim_dict["dv_reward_scale"] * np.linalg.norm(dv_ad) / n_steps
        )

        terminated = bool(out_of_shell or collision or escape)
        reward_terminal = 0.0
        termination_reason = None
        if out_of_shell:
            reward_terminal = self.optim_dict.get("G", OUT_OF_SHELL_REWARD)
            termination_reason = "out_of_shell"
        elif collision:
            reward_terminal = self.optim_dict.get("H", COLLISION_REWARD)
            termination_reason = "collision"
        elif escape:
            reward_terminal = self.optim_dict.get("H", COLLISION_REWARD)
            termination_reason = "escape"

        if terminated:
            reward_shell = 0.0
            reward_delta_v = 0.0

        reward = reward_shell + reward_delta_v + reward_terminal

        truncated = self.k >= n_steps and not terminated
        info.update({
            "ctrl_dt_base": info["ctrl_dt_base"],
            "dv_base_m_s": dv_base_m_s,
            "dv_residual_m_s": dv_residual_m_s,
            "dv_command_m_s": dv_total_vnb,
            "thruster_fired": thruster_fired,
            "dv_off_threshold_m_s": DV_OFF_THRESHOLD,
            "dv_executed_m_s": float(np.linalg.norm(dv_ad) * v_ad),
            "radius_m": float(radius * r_ad),
            "cumulative_dv_m_s": self.total_dv_m_s,
            "mass_fraction_1": self.mass_fraction_1,
            "mass_fraction_2": self.mass_fraction_2,
            "initial_position_x_m": float(self.initial_position_m[0]),
            "initial_position_y_m": float(self.initial_position_m[1]),
            "initial_position_z_m": float(self.initial_position_m[2]),
            "initial_velocity_x_m_s": float(self.initial_velocity_m_s[0]),
            "initial_velocity_y_m_s": float(self.initial_velocity_m_s[1]),
            "initial_velocity_z_m_s": float(self.initial_velocity_m_s[2]),
            "position_x_m": float(self.x[0] * r_ad),
            "position_y_m": float(self.x[1] * r_ad),
            "position_z_m": float(self.x[2] * r_ad),
            "velocity_x_m_s": float(self.x[3] * v_ad),
            "velocity_y_m_s": float(self.x[4] * v_ad),
            "velocity_z_m_s": float(self.x[5] * v_ad),
            "reward_shell": float(reward_shell),
            "reward_delta_v": float(reward_delta_v),
            "reward_terminal": float(reward_terminal),
            "out_of_shell": out_of_shell,
            "termination_reason": termination_reason or "",
            "terminated": terminated,
            "truncated": truncated,
        })
        if truncated:
            info["TimeLimit.truncated"] = not terminated

        return self.x.astype(np.float32), reward, terminated, truncated, info

    def _check_collision(self):
        r_ad = self.optim_dict["c"][1]
        axes = self.optim_dict["ellip_axes"] / r_ad
        x, y, z = self.x[:3]
        return (x / axes[0]) ** 2 + (y / axes[1]) ** 2 + (z / axes[2]) ** 2 < 1

    def _check_escape(self):
        r_ad = self.optim_dict["c"][1]
        return np.linalg.norm(self.x[:3]) > self.optim_dict["rmax"] / r_ad

    def _is_out_of_shell(self, state_ad, r_ad):
        radius_m = np.linalg.norm(state_ad) * r_ad
        inner, outer = self.optim_dict["shell"]["bounds"]
        return radius_m < inner or radius_m > outer

    def _base_dv(self, state_ad):
        """Predict and gate the base delta-v; return it in environment units."""
        if self.base_policy is None:
            return 0.0

        _, r_ad, v_ad = self.optim_dict["c"]
        state_dim = np.empty(6)
        state_dim[:3] = state_ad[:3] * r_ad
        state_dim[3:] = state_ad[3:6] * v_ad
        features = adimensionalize_state(state_dim[None, :], [self.k])

        device = next(self.base_policy.parameters()).device
        inputs = torch.as_tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, dv = self.base_policy(inputs)
            active = torch.sigmoid(logits) > self.cls_threshold
        return float(torch.where(active, dv, 0.0).item()) / v_ad
