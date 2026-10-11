"""
Scenario builders for the asteroid orbit problem.

Assemble the asteroid_dict and optim_dict from the constants in config, in the
exact format the environment expects. Both SAC training and evaluation call
these builders, so the scenario definition lives in a single place and can
never drift between the two.
"""

import numpy as np

from src.config import (
    KM2M,
    MASCON_FRACTIONS, MASCON_MU_TOTAL, MASCON_POSITIONS_KM,
    ASTEROID_AXES_M, ASTEROID_ROTATION_PERIOD_S,
    SHELL_MIN, SHELL_MAX, SHELL_GAMMA,
    TF_S, N_STEPS, DV_MAX_REG ,DVMAX_SAC, DVMAX_MPC, RMAX_M, COLLISION_REWARD,OUT_OF_SHELL_REWARD
)


def build_asteroid_dict():
    """Assemble the asteroid properties dictionary."""
    mu_masses = MASCON_FRACTIONS * MASCON_MU_TOTAL
    positions = MASCON_POSITIONS_KM * KM2M
    omega = np.array([0.0, 0.0, 2 * np.pi / ASTEROID_ROTATION_PERIOD_S])

    return {
        "mascon": {"muM": mu_masses, "xyzM": positions},
        "xyzM":   positions,
        "omega":  omega,
        "axes":   ASTEROID_AXES_M,
        
    }


def build_optim_dict(asteroid_dict):
    """Assemble the optimization/scenario dictionary, including scaling factors.

    The non-dimensionalization factors c = (t_ad, r_ad, v_ad) are derived from
    the asteroid's total mu and its reference axis, so they always stay
    consistent with the asteroid definition.
    """
    mu = np.sum(asteroid_dict["mascon"]["muM"])
    r_ad = asteroid_dict["axes"][0]
    v_ad = np.sqrt(mu / r_ad)
    t_ad = r_ad / v_ad
    c = (t_ad, r_ad, v_ad)

    shell_dict = {"bounds": (SHELL_MIN,SHELL_MAX), "gam": SHELL_GAMMA}

    return {
        "tf":         TF_S,
        "N":          N_STEPS,
        "dvmax":      DV_MAX_REG,
        'dvmax_SAC':  DVMAX_SAC,
        "dvmax_MPC":  DVMAX_MPC,
        "rmax":       RMAX_M,
        "ellip_axes": asteroid_dict["axes"],
        "c":          c,
        "shell":      shell_dict,
        "H":          COLLISION_REWARD,
        "G":          OUT_OF_SHELL_REWARD,
        "omega":      asteroid_dict['omega']
    }


def build_scenario():
    """Convenience builder returning both dictionaries."""
    asteroid_dict = build_asteroid_dict()
    optim_dict = build_optim_dict(asteroid_dict)
    return asteroid_dict, optim_dict