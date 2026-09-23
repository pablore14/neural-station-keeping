"""Generate the shared batch of initial conditions for every agent and method.

The whole batch is determined by a single MASTER_SEED. Two independent child
streams are spawned from it, one for the mass distribution and one for the
initial states, so that changing the number of episodes or reordering the code
never reshuffles the other stream.

Each episode stores its own perturbed mascon together with its initial state,
so any run (imitation, SAC, MPC, natural dynamics) can replay exactly the same
scenarios without regenerating anything.

Usage:
    python -m src.data.generate_initial_conditions
"""

from pathlib import Path
import pickle

import numpy as np

# Adjust this import to wherever the constants live in the repository.
from src.config import (
    KM2M,
    R0_NOMINAL,
    DISPERSION_M,
    MASS_VARIATION,
    MASCON_FRACTIONS,
    MASCON_MU_TOTAL,
    MASCON_POSITIONS_KM,
    ASTEROID_AXES_M,
    ASTEROID_OMEGA,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "data" / "initial_conditions_500.pkl"

N_EPISODES = 500
MASTER_SEED = 1


def perturbed_mascon(rng):
    """Return the perturbed mascon (muM, positions) for one episode.

    The mass split between the two point masses varies uniformly by up to
    MASS_VARIATION, while the total mu stays constant. Positions are relocated
    along the x-axis so the center of mass remains exactly at the origin.
    """
    eps = rng.uniform(-MASS_VARIATION, MASS_VARIATION)
    f1 = MASCON_FRACTIONS[0] * (1.0 + eps)
    f2 = 1.0 - f1
    fractions = np.array([f1, f2])
    mu_masses = fractions * MASCON_MU_TOTAL

    positions = MASCON_POSITIONS_KM * KM2M
    separation = positions[0, 0] - positions[1, 0]

    new_positions = positions.copy()
    new_positions[0, 0] = f2 * separation
    new_positions[1, 0] = -f1 * separation

    return mu_masses, new_positions, eps


def sample_initial_state(rng):
    """Draw one state from the spherical cloud around the nominal orbit.

    The reference velocity is the point-mass circular velocity at the nominal
    radius and is deliberately NOT corrected for the actual radius of the
    sampled position, which is what produces the family of slightly perturbed
    orbits. Since the perturbation keeps the total mu constant, the reference
    velocity is identical for every episode.
    """
    r_ref = R0_NOMINAL
    v_ref_mag = np.sqrt(MASCON_MU_TOTAL / np.linalg.norm(r_ref))
    v_ref = np.array([-v_ref_mag, 0.0, 0.0])

    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    rho = DISPERSION_M * rng.random() ** (1.0 / 3.0)

    pos = r_ref + rho * direction
    vel = v_ref - np.cross(ASTEROID_OMEGA, pos)

    return pos, vel, rho


def build_batch(n_episodes=N_EPISODES, master_seed=MASTER_SEED):
    """Build the full batch of episodes as plain arrays."""
    mass_seed, state_seed = np.random.SeedSequence(master_seed).spawn(2)
    rng_mass = np.random.default_rng(mass_seed)
    rng_state = np.random.default_rng(state_seed)

    positions = np.empty((n_episodes, 3))
    velocities = np.empty((n_episodes, 3))
    rho = np.empty(n_episodes)
    mu_masses = np.empty((n_episodes, len(MASCON_FRACTIONS)))
    mascon_positions = np.empty((n_episodes,) + MASCON_POSITIONS_KM.shape)
    eps = np.empty(n_episodes)

    for i in range(n_episodes):
        mu_masses[i], mascon_positions[i], eps[i] = perturbed_mascon(rng_mass)
        positions[i], velocities[i], rho[i] = sample_initial_state(rng_state)

    return {
        "x0": np.hstack([positions, velocities]),
        "pos0": positions,
        "vel0": velocities,
        "rho": rho,
        "muM": mu_masses,
        "xyzM": mascon_positions,
        "eps": eps,
    }


def episode_asteroid_dict(batch, i):
    """Rebuild the asteroid_dict of episode i from the stored batch."""
    return {
        "axes": ASTEROID_AXES,
        "omega": ASTEROID_OMEGA,
        "mascon": {"muM": batch["muM"][i], "xyzM": batch["xyzM"][i]},
        "xyzM": batch["xyzM"][i],
    }


def main():
    batch = build_batch()

    output = {
        "batch": batch,
        "meta": {
            "n_episodes": N_EPISODES,
            "master_seed": MASTER_SEED,
            "r0_nominal": R0_NOMINAL,
            "dispersion_m": DISPERSION_M,
            "mass_variation": MASS_VARIATION,
            "mascon_fractions_nominal": MASCON_FRACTIONS,
            "mascon_mu_total": MASCON_MU_TOTAL,
            "mascon_positions_nominal": MASCON_POSITIONS_KM * KM2M,
            "asteroid_axes": ASTEROID_AXES_M,
            "asteroid_omega": ASTEROID_OMEGA,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(output, f)

    radii = np.linalg.norm(batch["pos0"], axis=1)
    com = np.sum(batch["muM"][:, :, None] * batch["xyzM"], axis=1) \
        / np.sum(batch["muM"], axis=1)[:, None]

    print(f"Saved {N_EPISODES} episodes to {OUTPUT_PATH}")
    print(f"  seed             : {MASTER_SEED}")
    print(f"  |r0| range       : {radii.min() / KM2M:.3f} - "
          f"{radii.max() / KM2M:.3f} km")
    print(f"  mass fraction f1 : {batch['muM'][:, 0].min() / MASCON_MU_TOTAL:.4f} - "
          f"{batch['muM'][:, 0].max() / MASCON_MU_TOTAL:.4f}")
    print(f"  total mu error   : "
          f"{np.abs(batch['muM'].sum(axis=1) / MASCON_MU_TOTAL - 1).max():.2e}")
    print(f"  max |com|        : {np.abs(com).max():.2e} m")


if __name__ == "__main__":
    main()