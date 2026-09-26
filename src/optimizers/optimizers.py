import casadi as ca
import numpy as np

from src.optimizers import (collision_constraint, escape_constraint,
                            shell_penalty, unpack_wopt)
from src.optimizers.xyz2vnb import xyz2vnb


def optimize_safeorbit(initguess_dict,
                       optim_dict,
                       intg):
    """
    Optimize a trajectory with impulsive dV applied AFTER each integration step,
    and with path constraints:
      - collision avoidance with ellipsoid
      - maximum orbital radius

    Args:
        x0 (array): initial state (6D)
        tf (float): final time
        target_body: object containing target info
        init_guess: dict with 'X_ad', 'dV_ad', 'N'
        adim_factors: scaling factors
        intg: CasADi integrator with params = dt
        ellipsoid_params (tuple): (a,b,c) semi-axes of forbidden ellipsoid (optional)
        r_max (float): maximum allowed orbital radius (optional)
    """
    # Adimensional factors
    t_ad, r_ad, v_ad = optim_dict['c']

    # Retrieve initial guess and adimensionalize
    T0 = initguess_dict['T'] / t_ad
    X0 = np.hstack([initguess_dict['X'][:,0:3]/r_ad,
                    initguess_dict['X'][:,3:6]/v_ad])
    dV0 = initguess_dict['dV'] / v_ad
    N = len(T0)

    # Adimensionalize optimization variables
    tf = optim_dict['tf'] / t_ad
    dvmax = optim_dict['dvmax'] / v_ad
    if 'ellip_axes' in optim_dict:
        ellip_axes = optim_dict['ellip_axes'] / r_ad
    if 'rmax' in optim_dict:
        rmax = optim_dict['rmax'] / r_ad

    # Adimensionalize asteroid sidereal rotation
    omega = ca.DM(optim_dict['omega']) * t_ad

    # Adimensionalize x0
    x0 = np.hstack((optim_dict['x0'][0:3] / r_ad,
                    optim_dict['x0'][3:6] / v_ad))

    # Unfold dict of orbit corridor
    r_in = optim_dict['shell']['bounds'][0]
    r_out = optim_dict['shell']['bounds'][1]
    gam = optim_dict['shell']['gam']

    # State and delta-v variables
    w_x, w0_x, lbw_x, ubw_x = [], [], [], []
    w_dv, w0_dv, lbw_dv, ubw_dv = [], [], [], []

    # Constraints variables
    g_dyn, lbg_dyn, ubg_dyn = [], [], []
    g1, lbg1, ubg1 = [], [], []
    g2, lbg2, ubg2 = [], [], []

    # Shell penalty
    shell_penalties = []

    # Initial state
    xk = ca.DM(x0)

    # Loop through impulses
    for k in range(N-1):
        # Get times
        dt = T0[k+1] - T0[k]

        # 1. Auxiliary variables for L1 norm
        dvk_plus = ca.MX.sym(f'dv_plus_{k}', 3)
        dvk_minus = ca.MX.sym(f'dv_minus_{k}', 3)

        # Delta-v as difference
        dvk = dvk_plus - dvk_minus

        # Store delta-v decision variables
        w_dv += [dvk_plus,
                 dvk_minus]
        w0_dv += (dV0[k, :].tolist() + dV0[k, :].tolist())
        lbw_dv += [0, 0, 0] * 2
        ubw_dv += dvmax.tolist() * 2

        # Apply impulse
        dvk_xyz = xyz2vnb(xk, dvk, omega)
        xk_plus = ca.vertcat(xk[0:3],
                             xk[3:6] + dvk_xyz)

        # 2. Propagate unforced dynamics after delta-v
        Fk = intg(x0=xk_plus, p=dt)
        xk_unforced = Fk['xf']

        # Compute penalty term to orbit bandwidth
        rk = ca.norm_2(xk_unforced[0:3])
        penalty_k = shell_penalty(rk, a=r_in, b=r_out)
        shell_penalties.append(penalty_k)

        # 3. Next state decision variable
        xk_next = ca.MX.sym(f'x_{k+1}', 6)
        w_x  += [xk_next]
        w0_x += X0[k+1,:].tolist()
        lbw_x += [-np.inf]*6
        ubw_x += [ np.inf]*6

        # 4. Enforce dynamics
        g_dyn   += [xk_next - xk_unforced]
        lbg_dyn += [0]*6
        ubg_dyn += [0]*6

        # ---------------- Path constraints ---------------- #
        # Collision avoidance: ellipsoid inequality
        if 'ellip_axes' in optim_dict:
            g1_k, lbg1_k, ubg1_k = collision_constraint(xk_unforced,
                                                        ellip_axes)
            g1 += g1_k
            lbg1 += lbg1_k
            ubg1 += ubg1_k

        # Escape avoidance: maximum radius
        if 'rmax' in optim_dict:
            g2_k, lbg2_k, ubg2_k = escape_constraint(xk_unforced,
                                                     rmax)
            g2 += g2_k
            lbg2 += lbg2_k
            ubg2 += ubg2_k
        # --------------------------------------------------- #

        # Update for next iteration
        xk = xk_next

    # Objective: sum of ∆V magnitudes
    J_dv = sum(ca.sum1(dv) for dv in w_dv) / np.max(dvmax)
    J_shell = gam*sum(shell_penalties)
    J = J_dv/N + J_shell/N

    # Place all decision variables in vectors
    w = w_x + w_dv
    w0 = w0_x + w0_dv
    lbw = lbw_x + lbw_dv
    ubw = ubw_x + ubw_dv

    # Place all constraints in vectors
    g = g_dyn + g1 + g2
    lbg = lbg_dyn + lbg1 + lbg2
    ubg = ubg_dyn + ubg1 + ubg2

    # NLP setup
    prob = {'f': J,
            'x': ca.vertcat(*w),
            'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', prob,
                       {'ipopt.print_level': 5,
                        'print_time': True,
                        'ipopt.max_iter': 250})

    # Solve
    sol = solver(x0=ca.vertcat(*w0),
                 lbx=lbw, ubx=ubw,
                 lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    # Unpack solution and save in dict
    T, X, dV = unpack_wopt(w_opt, T0, x0, c=optim_dict['c'])
    status = 'success'
    if solver.stats()['return_status'] != 'Solve_Succeeded':
        status = 'fail'
        dV *= 0
    results_dict = {'T': T,
                    'X': X,
                    'dV': dV,
                    'status': status}

    return results_dict

def optimize_safeorbit2(initguess_dict,
                       optim_dict,
                       intg):
    """
    Optimize a trajectory with impulsive dV applied AFTER each integration step,
    and with path constraints:
      - collision avoidance with ellipsoid
      - maximum orbital radius

    Args:
        x0 (array): initial state (6D)
        tf (float): final time
        target_body: object containing target info
        init_guess: dict with 'X_ad', 'dV_ad', 'N'
        adim_factors: scaling factors
        intg: CasADi integrator with params = dt
        ellipsoid_params (tuple): (a,b,c) semi-axes of forbidden ellipsoid (optional)
        r_max (float): maximum allowed orbital radius (optional)
    """
    # Adimensional factors
    t_ad, r_ad, v_ad = optim_dict['c']

    # Retrieve initial guess and adimensionalize
    T0 = initguess_dict['T'] / t_ad
    X0 = np.hstack([initguess_dict['X'][:,0:3]/r_ad,
                    initguess_dict['X'][:,3:6]/v_ad])
    dV0 = initguess_dict['dV'] / v_ad
    N = len(T0)

    # Adimensionalize optimization variables
    tf = optim_dict['tf'] / t_ad
    dvmax = optim_dict['dvmax'] / v_ad
    if 'ellip_axes' in optim_dict:
        ellip_axes = optim_dict['ellip_axes'] / r_ad
    if 'rmin' in optim_dict:
            rmin = optim_dict['rmin'] / r_ad
    if 'rmax' in optim_dict:
            rmax = optim_dict['rmax'] / r_ad

    # Adimensionalize asteroid sidereal rotation
    omega = ca.DM(optim_dict['omega']) * t_ad

    # Adimensionalize x0
    x0 = np.hstack((optim_dict['x0'][0:3] / r_ad,
                    optim_dict['x0'][3:6] / v_ad))

    # Unfold dict of orbit corridor
    r_in = optim_dict['shell']['bounds'][0]
    r_out = optim_dict['shell']['bounds'][1]
    gam = optim_dict['shell']['gam']

    # State and delta-v variables
    w_x, w0_x, lbw_x, ubw_x = [], [], [], []
    w_dv, w0_dv, lbw_dv, ubw_dv = [], [], [], []

    # Constraints variables
    g_dyn, lbg_dyn, ubg_dyn = [], [], []
    g1, lbg1, ubg1 = [], [], []
    g2, lbg2, ubg2 = [], [], []

    # Shell penalty
    shell_penalties = []

    # Initial state
    xk = ca.DM(x0)

    # Loop through impulses
    for k in range(N-1):
        # Get times
        dt = T0[k+1] - T0[k]

        # 1. Auxiliary variables for L1 norm
        dvk_plus = ca.MX.sym(f'dv_plus_{k}', 3)
        dvk_minus = ca.MX.sym(f'dv_minus_{k}', 3)

        # Delta-v as difference
        dvk = dvk_plus - dvk_minus

        # Store delta-v decision variables
        w_dv += [dvk_plus,
                 dvk_minus]
        w0_dv += (dV0[k, :].tolist() + dV0[k, :].tolist())
        lbw_dv += [0, 0, 0] * 2
        ubw_dv += dvmax.tolist() * 2

        # Apply impulse
        dvk_xyz = xyz2vnb(xk, dvk, omega)
        xk_plus = ca.vertcat(xk[0:3],
                             xk[3:6] + dvk_xyz)

        # 2. Propagate unforced dynamics after delta-v
        Fk = intg(x0=xk_plus, p=dt)
        xk_unforced = Fk['xf']

        # Compute penalty term to orbit bandwidth
        rk = ca.norm_2(xk_unforced[0:3])
        penalty_k = shell_penalty(rk, a=r_in, b=r_out)
        shell_penalties.append(penalty_k)

        # 3. Next state decision variable
        xk_next = ca.MX.sym(f'x_{k+1}', 6)
        w_x  += [xk_next]
        w0_x += X0[k+1,:].tolist()
        lbw_x += [-np.inf]*6
        ubw_x += [ np.inf]*6

        # 4. Enforce dynamics
        g_dyn   += [xk_next - xk_unforced]
        lbg_dyn += [0]*6
        ubg_dyn += [0]*6

        # ---------------- Path constraints ---------------- #
        # Collision avoidance: ellipsoid inequality
        if 'ellip_axes' in optim_dict:
            g1_k, lbg1_k, ubg1_k = collision_constraint(xk_unforced,
                                                        ellip_axes)
            g1 += g1_k
            lbg1 += lbg1_k
            ubg1 += ubg1_k
        
        


        # Restricción dura: r >= rmin
        if 'rmin' in optim_dict:
            r2 = ca.sumsqr(xk_unforced[0:3])
            g2 += [r2 - rmin**2]
            lbg2 += [0.0]
            ubg2 += [ca.inf]

        # Restricción dura: r <= rmax
        if 'rmax' in optim_dict:
            r2 = ca.sumsqr(xk_unforced[0:3])
            g2 += [rmax**2 - r2]
            lbg2 += [0.0]
            ubg2 += [ca.inf]
        # Escape avoidance: maximum radius
        # if 'rmax' in optim_dict:
        #     g2_k, lbg2_k, ubg2_k = escape_constraint(xk_unforced,
        #                                              rmax)
        #     g2 += g2_k
        #     lbg2 += lbg2_k
        #     ubg2 += ubg2_k
        # --------------------------------------------------- #

        # Update for next iteration
        xk = xk_next

    # Objective: sum of ∆V magnitudes
    J_dv = sum(ca.sum1(dv) for dv in w_dv) / np.max(dvmax)
    J_shell = gam*sum(shell_penalties)
    J = J_dv/N + J_shell/N

    # Place all decision variables in vectors
    w = w_x + w_dv
    w0 = w0_x + w0_dv
    lbw = lbw_x + lbw_dv
    ubw = ubw_x + ubw_dv

    # Place all constraints in vectors
    g = g_dyn + g1 + g2
    lbg = lbg_dyn + lbg1 + lbg2
    ubg = ubg_dyn + ubg1 + ubg2

    # NLP setup
    prob = {'f': J,
            'x': ca.vertcat(*w),
            'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', prob,
                       {'ipopt.print_level': 5,
                        'print_time': True,
                        'ipopt.max_iter': 250})

    # Solve
    sol = solver(x0=ca.vertcat(*w0),
                 lbx=lbw, ubx=ubw,
                 lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    # Unpack solution and save in dict
    T, X, dV = unpack_wopt(w_opt, T0, x0, c=optim_dict['c'])
    status = 'success'
    if solver.stats()['return_status'] != 'Solve_Succeeded':
        status = 'fail'
        dV *= 0
    results_dict = {'T': T,
                    'X': X,
                    'dV': dV,
                    'status': status}

    return results_dict


def optimize_periodic(initguess_dict,
                      optim_dict,
                      intg):
    """
    Optimize a periodic trajectory with impulsive dV applied AFTER each integration step.
    
    The final state is constrained to equal the initial state (xf = x0).
    The total time (tf) is treated as a decision variable to minimize fuel consumption
    while finding the period that allows periodic orbits.
    
    Path constraints:
      - collision avoidance with ellipsoid
      - maximum orbital radius
      - periodicity: x(tf) = x(0)

    Args:
        initguess_dict: dict with 'T', 'X', 'dV' (initial guess)
        optim_dict: dict with 'c' (scaling factors), 'dvmax', 'omega', 'x0',
                   'shell' (orbit bounds), and optional 'ellip_axes', 'rmax'
        intg: CasADi integrator with params = dt
    """
    # Adimensional factors
    t_ad, r_ad, v_ad = optim_dict['c']

    # Retrieve initial guess and adimensionalize
    T0 = initguess_dict['T'] / t_ad
    X0 = np.hstack([initguess_dict['X'][:,0:3]/r_ad,
                    initguess_dict['X'][:,3:6]/v_ad])
    dV0 = initguess_dict['dV'] / v_ad
    N = len(T0)

    # Adimensionalize optimization variables
    dvmax = optim_dict['dvmax'] / v_ad
    if 'ellip_axes' in optim_dict:
        ellip_axes = optim_dict['ellip_axes'] / r_ad
    if 'rmax' in optim_dict:
        rmax = optim_dict['rmax'] / r_ad

    # Adimensionalize asteroid sidereal rotation
    omega = ca.DM(optim_dict['omega']) * t_ad

    # Adimensionalize x0
    x0 = np.hstack((optim_dict['x0'][0:3] / r_ad,
                    optim_dict['x0'][3:6] / v_ad))

    # Unfold dict of orbit corridor
    r_in = optim_dict['shell']['bounds'][0]
    r_out = optim_dict['shell']['bounds'][1]
    gam = optim_dict['shell']['gam']

    # State and delta-v variables
    w_x, w0_x, lbw_x, ubw_x = [], [], [], []
    w_dv, w0_dv, lbw_dv, ubw_dv = [], [], [], []

    # Time as decision variable
    w_tf = []
    w0_tf = []
    lbw_tf = []
    ubw_tf = []

    # Constraints variables
    g_dyn, lbg_dyn, ubg_dyn = [], [], []
    g1, lbg1, ubg1 = [], [], []
    g2, lbg2, ubg2 = [], [], []
    g_periodic, lbg_periodic, ubg_periodic = [], [], []

    # Shell penalty
    shell_penalties = []

    # Initial state (fixed, not a decision variable)
    xk = ca.DM(x0)

    # ========== DECISION VARIABLE: total time tf ========== #
    tf_decision = ca.MX.sym('tf', 1)
    w_tf += [tf_decision]
    w0_tf += [T0[-1]]  # Initial guess: last time from initial guess
    lbw_tf += [0.1]    # Lower bound: minimum positive time
    ubw_tf += [100.0]  # Upper bound: maximum time

    # Scale time intervals uniformly based on the decision variable tf
    # This requires interpolating time steps from the initial guess
    time_fractions = T0 / T0[-1]  # Normalize initial times to [0, 1]

    # Loop through impulses
    for k in range(N-1):
        # Compute dt based on scaled time fractions
        # dt_k = (t_frac[k+1] - t_frac[k]) * tf_decision
        dt_k = (time_fractions[k+1] - time_fractions[k]) * tf_decision[0]

        # 1. Auxiliary variables for L1 norm
        dvk_plus = ca.MX.sym(f'dv_plus_{k}', 3)
        dvk_minus = ca.MX.sym(f'dv_minus_{k}', 3)

        # Delta-v as difference
        dvk = dvk_plus - dvk_minus

        # Store delta-v decision variables
        w_dv += [dvk_plus,
                 dvk_minus]
        w0_dv += (dV0[k, :].tolist() + dV0[k, :].tolist())
        lbw_dv += [0, 0, 0] * 2
        ubw_dv += dvmax.tolist() * 2

        # Apply impulse
        dvk_xyz = xyz2vnb(xk, dvk, omega)
        xk_plus = ca.vertcat(xk[0:3],
                             xk[3:6] + dvk_xyz)

        # 2. Propagate unforced dynamics after delta-v
        Fk = intg(x0=xk_plus, p=dt_k)
        xk_unforced = Fk['xf']

        # Compute penalty term to orbit bandwidth
        rk = ca.norm_2(xk_unforced[0:3])
        penalty_k = shell_penalty(rk, a=r_in, b=r_out)
        shell_penalties.append(penalty_k)

        # 3. Next state decision variable (except for last step)
        if k < N - 2:
            xk_next = ca.MX.sym(f'x_{k+1}', 6)
            w_x  += [xk_next]
            w0_x += X0[k+1,:].tolist()
            lbw_x += [-np.inf]*6
            ubw_x += [ np.inf]*6

            # 4. Enforce dynamics
            g_dyn   += [xk_next - xk_unforced]
            lbg_dyn += [0]*6
            ubg_dyn += [0]*6

            # Update for next iteration
            xk = xk_next
        else:
            # Last step: xk_next should equal x0 (periodicity constraint)
            g_periodic += [xk_unforced - ca.DM(x0)]
            lbg_periodic += [0]*6
            ubg_periodic += [0]*6

        # ---------------- Path constraints ---------------- #
        # Collision avoidance: ellipsoid inequality
        if 'ellip_axes' in optim_dict:
            g1_k, lbg1_k, ubg1_k = collision_constraint(xk_unforced,
                                                        ellip_axes)
            g1 += g1_k
            lbg1 += lbg1_k
            ubg1 += ubg1_k

        # Escape avoidance: maximum radius
        if 'rmax' in optim_dict:
            g2_k, lbg2_k, ubg2_k = escape_constraint(xk_unforced,
                                                     rmax)
            g2 += g2_k
            lbg2 += lbg2_k
            ubg2 += ubg2_k
        # --------------------------------------------------- #

    # Objective: sum of ∆V magnitudes (minimize fuel)
    J_dv = sum(ca.sum1(dv) for dv in w_dv) / np.max(dvmax)
    J_shell = gam*sum(shell_penalties)
    J = J_dv/N + J_shell/N

    # Place all decision variables in vectors
    w = w_x + w_dv + w_tf
    w0 = w0_x + w0_dv + w0_tf
    lbw = lbw_x + lbw_dv + lbw_tf
    ubw = ubw_x + ubw_dv + ubw_tf

    # Place all constraints in vectors
    g = g_dyn + g1 + g2 + g_periodic
    lbg = lbg_dyn + lbg1 + lbg2 + lbg_periodic
    ubg = ubg_dyn + ubg1 + ubg2 + ubg_periodic

    # NLP setup
    prob = {'f': J,
            'x': ca.vertcat(*w),
            'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', prob,
                       {'ipopt.print_level': 5,
                        'print_time': True,
                        'ipopt.max_iter': 500})

    # Solve
    sol = solver(x0=ca.vertcat(*w0),
                 lbx=lbw, ubx=ubw,
                 lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    # Extract optimized tf
    tf_opt = w_opt[-1]  # Last decision variable is tf
    
    # Reconstruct time array with optimized tf
    T_opt = time_fractions * tf_opt * t_ad  # Dimensional time

    # Unpack solution manually (excluding tf and last state)
    # w_opt structure: [x_1, ..., x_{N-2}, dv_plus_0, dv_minus_0, ..., dv_plus_{N-2}, dv_minus_{N-2}, tf]
    N_minus_2_states = 6 * (N - 2)
    x_block_opt = w_opt[:N_minus_2_states]
    dv_block_opt = w_opt[N_minus_2_states:-1]  # Exclude tf
    
    # Rebuild trajectory (N states total)
    X = [x0]
    for k in range(N - 2):
        Xk = x_block_opt[6*k:6*(k+1)]
        X.append(Xk)
    # Last state equals first state (periodic orbit)
    X.append(x0)
    
    # Rebuild impulses (N-1 impulses, then one zero for closure)
    dV_list = []
    for k in range(N - 2):
        dv_plus = dv_block_opt[6*k:6*k+3]
        dv_minus = dv_block_opt[6*k+3:6*k+6]
        dV_list.append(dv_plus - dv_minus)
    
    # Add zeros for last two impulses
    dV_list.append(np.zeros(3))
    dV_list.append(np.zeros(3))
    
    # Make arrays and dimensionalize
    X = np.array(X)
    dV = np.array(dV_list)
    
    X[:,0:3] *= r_ad
    X[:,3:6] *= v_ad
    dV *= v_ad
    T = T_opt
    
    status = 'success'
    if solver.stats()['return_status'] != 'Solve_Succeeded':
        status = 'fail'
        dV *= 0
    
    results_dict = {'T': T,
                    'X': X,
                    'dV': dV,
                    'tf': tf_opt * t_ad,  # Return dimensional time
                    'status': status}

    return results_dict

