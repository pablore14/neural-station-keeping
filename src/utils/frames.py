
"""
Reference-frame utilities.

Pure geometric helpers for transforming between reference frames. These
functions take DIMENSIONAL quantities (m, m/s) and know nothing about
non-dimensionalization: any scaling must be done by the caller beforehand.
Keeping each function to a single, clearly-typed input avoids silent unit
bugs (passing a scaled state where a physical one is expected).
"""

import numpy as np


def vnb_to_planetary(r, v_planet, omega):
    """Build the rotation matrix from the VNB frame to the planetary frame.

    The VNB frame is defined by the inertial velocity direction (V), the
    orbital angular-momentum direction (N), and their cross product (B).

    Args:
        r:        position vector in the planetary frame, m, shape (3,).
        v_planet: velocity vector in the planetary frame, m/s, shape (3,).
        omega:    body angular velocity vector, rad/s, shape (3,).

    Returns:
        3x3 rotation matrix whose columns are [V_hat, N_hat, B_hat].
        A vector expressed in VNB is mapped to the planetary frame via R @ vec.
    """
    # Inertial velocity: add the rigid-body rotation contribution.
    v_inertial = v_planet + np.cross(omega, r)

    v_hat = v_inertial / np.linalg.norm(v_inertial)

    n_hat = np.cross(r, v_inertial)
    n_hat /= np.linalg.norm(n_hat)

    b_hat = np.cross(v_hat, n_hat)

    return np.column_stack([v_hat, n_hat, b_hat])

def small_angle_rotation(dtheta):
    """R = I + [dtheta]_x  (aproximación de ángulo pequeño)."""
    tx, ty, tz = dtheta
    K = np.array([[0, -tz,  ty],
                  [tz,  0, -tx],
                  [-ty, tx,  0]])
    return np.eye(3) + K