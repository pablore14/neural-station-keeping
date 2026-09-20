import casadi as ca


def xyz2vnb(x, dv_vnb, omega, eps=1e-12):
    """
    Transforms an impulse in local VNB frame
    to asteroid-centred asteroid-fixed cartesian

    Args:
        x (array): initial state (6D)
        dv_vnb (array): final time
        omega: object containing target info
        eps (float): maximum allowed orbital radius (optional)
    """
    # Position and velocity
    pos = x[0:3]
    vel = x[3:6]

    # Inertial velocity
    vel_N = vel + ca.cross(omega, pos)

    # Velocity
    v_hat = vel_N / (ca.norm_2(vel_N) + eps)

    # Normal
    h = ca.cross(pos, vel_N)
    n_hat = h / (ca.norm_2(h) + eps)

    # Binormal
    b_hat = ca.cross(v_hat, n_hat)

    # Projection
    dv_xyz = (dv_vnb[0]*v_hat
              + dv_vnb[1]*n_hat
              + dv_vnb[2]*b_hat)

    return dv_xyz
