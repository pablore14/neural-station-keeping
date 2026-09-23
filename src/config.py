
"""
Central project configuration.

All physical, numerical and reference constants live HERE and only here.
Any script must import them from this file, never redefine them:

    from src.config import MU_ASTEROID_KM3_S2, R_REF, V_REF

Rule of thumb: if a value appears in more than one place, it is a bug waiting
to happen. Change it here once and it updates across the whole project.
"""

import os
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# Built relative to the project root, NEVER absolute like /home/pabredama/...
# This way the project runs on any machine and for anyone who opens it.
# ══════════════════════════════════════════════════════════════════════════
# Project root = folder containing src/ (two levels above this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR_REG = os.path.join(PROJECT_ROOT, "results/REGRESSOR")
RESULTS_DIR_RL = os.path.join(PROJECT_ROOT, "results/RL")


# ══════════════════════════════════════════════════════════════════════════
# CONVERSION FACTORS
# ══════════════════════════════════════════════════════════════════════════
KM2M = 1e3
M2KM = 1e-3
AU2M = 1.495978707e11   # 1 astronomical unit in meters


# ══════════════════════════════════════════════════════════════════════════
# GRAVITATIONAL PARAMETER (MU) — NOTE: these are TWO different things
# ══════════════════════════════════════════════════════════════════════════
# (1) Total physical MU of the asteroid = sum of the mascon point masses.
#     This is what governs the real dynamics of the spacecraft.
MU_ASTEROID_M3_S2 = np.sum(np.array([0.6, 0.4]) * 4.46275472004e5)  # check units

# (2) REFERENCE MU used to non-dimensionalize the state and train the v3 regressor.
#     Used only to normalize the network input state, NOT for the physics.
MU_REF = 4.46275472004e5

# NOTE: numerically (1) and (2) may coincide, but they are distinct concepts.
# Keeping them separate avoids silent bugs if they ever change independently.


# ══════════════════════════════════════════════════════════════════════════
# ORBITAL REFERENCES (used to non-dimensionalize the regressor state)
# ══════════════════════════════════════════════════════════════════════════
R_REF = 26_000.0                    # reference radius, m (nominal orbit)
V_REF = np.sqrt(MU_REF / R_REF)     # reference circular velocity, m/s


# ══════════════════════════════════════════════════════════════════════════
# EPISODE INITIAL CONDITIONS
# r0 = [0, 26 km, 0] with 3 km spherical dispersion, circular velocity
# ══════════════════════════════════════════════════════════════════════════
R0_NOMINAL   = np.array([0.0, 26.0 * KM2M, 0.0])   # nominal position, m
DISPERSION_M = 3.0 * KM2M                            # 1-sigma dispersion, m


# ══════════════════════════════════════════════════════════════════════════
# SIMULATION TIME PARAMETERS
# ══════════════════════════════════════════════════════════════════════════
T_FINAL     = 36_000.0    # episode duration, s
DT_STEP     = 600.0       # control step, s
N_STEPS_NOM = 60          # nominal number of steps per episode


# ══════════════════════════════════════════════════════════════════════════
# CONTROL LIMITS (DELTA-V)
# ══════════════════════════════════════════════════════════════════════════
DV_MAX_REG  = 0.2         # base regressor dv_max, m/s
DV_MAX_SAC  = 0.05        # SAC residual dv_max, m/s
THR_REG_ABS = 5e-3        # threshold below which the regressor dv is set to 0


# ══════════════════════════════════════════════════════════════════════════
# TARGET SHELL (orbital band where we want to keep the spacecraft)
# ══════════════════════════════════════════════════════════════════════════
SHELL_MIN = 22_000.0      # shell inner radius, m
SHELL_MAX = 30_000.0      # shell outer radius, m



# ══════════════════════════════════════════════════════════════════════════
# THRESHOLD REGRESSOR
# ══════════════════════════════════════════════════════════════════════════
DELTA = 1e-8   # threshold below which a delta-v is treated as exactly zero

# --- Mascon model of the asteroid ---
MASCON_FRACTIONS   = np.array([0.6, 0.4])            # mass fractions, dimensionless
MASCON_MU_TOTAL    = 4.46275472004e5                 # total mu, m^3/s^2 (check units)
MASCON_POSITIONS_KM = np.array([[(0.4 / 0.6) * 8, 0, 0],
                                [-8,              0, 0]])   # point-mass positions, km
 
# --- Asteroid geometry and rotation ---
ASTEROID_AXES_M           = np.array([16, 8, 5]) * KM2M   # ellipsoid semi-axes, m
ASTEROID_ROTATION_PERIOD_S = 5.27 * 3600                   # rotation period, s
ASTEROID_OMEGA = np.array([0.0, 0.0, 2 * np.pi / ASTEROID_ROTATION_PERIOD_S])
SHELL_GAMMA    = 0.1        
 
# --- Episode / optimization scenario ---
TF_S            = 10 * 3600     # episode duration, s (10 h)
N_STEPS         = 60            # control steps per episode
DVMAX_SAC       = 0.05          # SAC residual dv_max, m/s
RMAX_M          = 50 * KM2M     # escape radius, m
COLLISION_REWARD = -5.0         # terminal reward on collision/escape
OUT_OF_SHELL_REWARD = -3.0


SIGMA_DV_ABS = 0.005 * (DV_MAX_REG + DV_MAX_SAC)

SIGMA_THETA = np.deg2rad(3)

MASS_VARIATION = 0.25

DVMAX_MPC = 0.25*np.array([1, 0, 0])