# -*- coding: utf-8 -*-
"""
State non-dimensionalization for the base regressor.

Single source of truth for turning a physical (dimensional) state into the
8-feature input the regressor expects:

    [r_hat(3), dr, v_adim(3), t_norm]

The function is vectorized: it works on a batch of states (N, 6). To process a
single state, pass it as a batch of one, e.g. state[None, :]. This keeps one
implementation for both the per-step (evaluation / environment) and the
full-dataset (training) use cases, so they can never diverge.
"""

import numpy as np

from src.config import R_REF, V_REF, N_STEPS_NOM


def adimensionalize_state(states, step_indices):
    """Map physical states to the regressor's 8-feature input.

    Args:
        states:       array of shape (N, 6) with columns [r(3), v(3)] in
                      meters and m/s.
        step_indices: array of shape (N,) with the integer time step of each
                      state (used to build the normalized time feature).

    Returns:
        Array of shape (N, 8): [r_hat(3), dr, v_adim(3), t_norm], float32.
    """
    states = np.asarray(states, dtype=np.float64)
    r = states[:, :3]
    v = states[:, 3:6]

    r_norm = np.linalg.norm(r, axis=1, keepdims=True)   # (N, 1)
    r_hat  = r / r_norm                                  # (N, 3)
    dr     = (r_norm - R_REF) / R_REF                    # (N, 1)
    v_adim = v / V_REF                                   # (N, 3)

    t_norm = (np.asarray(step_indices, dtype=np.float64)
              / N_STEPS_NOM).reshape(-1, 1)              # (N, 1)

    features = np.concatenate([r_hat, dr, v_adim, t_norm], axis=1)
    return features.astype(np.float32)