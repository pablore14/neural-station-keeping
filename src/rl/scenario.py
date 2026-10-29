"""Build the asteroid and episode settings used by the RL environment."""

import numpy as np

from src.config import (
    ASTEROID_AXES_M,
    ASTEROID_ROTATION_PERIOD_S,
    COLLISION_REWARD,
    DVMAX_MPC,
    DVMAX_SAC,
    DV_REWARD_SCALE,
    DV_MAX_REG,
    KM2M,
    MASCON_FRACTIONS,
    MASCON_MU_TOTAL,
    MASCON_POSITIONS_KM,
    RL_N_STEPS,
    OUT_OF_SHELL_REWARD,
    RMAX_M,
    SHELL_GAMMA,
    SHELL_MAX,
    SHELL_MIN,
    TF_S,
)


def build_asteroid_dict():
    """Return the asteroid parameters in the format expected by the dynamics."""
    mu_masses = MASCON_FRACTIONS * MASCON_MU_TOTAL
    positions = MASCON_POSITIONS_KM * KM2M
    omega = np.array([0.0, 0.0, 2 * np.pi / ASTEROID_ROTATION_PERIOD_S])

    return {
        "mascon": {"muM": mu_masses, "xyzM": positions},
        "xyzM": positions,
        "omega": omega,
        "axes": ASTEROID_AXES_M,
    }


def build_optim_dict(asteroid_dict):
    """Return the episode, control, and reward settings for the environment."""
    mu = np.sum(asteroid_dict["mascon"]["muM"])
    r_ad = asteroid_dict["axes"][0]
    v_ad = np.sqrt(mu / r_ad)
    t_ad = r_ad / v_ad

    return {
        "tf": TF_S,
        "N": RL_N_STEPS,
        "dvmax": DV_MAX_REG,
        "dvmax_SAC": DVMAX_SAC,
        "dvmax_MPC": DVMAX_MPC,
        "rmax": RMAX_M,
        "ellip_axes": asteroid_dict["axes"],
        "c": (t_ad, r_ad, v_ad),
        "shell": {"bounds": (SHELL_MIN, SHELL_MAX), "gam": SHELL_GAMMA},
        "dv_reward_scale": DV_REWARD_SCALE,
        "H": COLLISION_REWARD,
        "G": OUT_OF_SHELL_REWARD,
        "omega": asteroid_dict["omega"],
    }


def build_scenario():
    """Return the asteroid and optimization dictionaries together."""
    asteroid_dict = build_asteroid_dict()
    return asteroid_dict, build_optim_dict(asteroid_dict)
