# Neural Guidance Policies for Autonomous Station-Keeping Around Asteroids

Code for the master's thesis and paper by Pablo Redondo Amaro,
Universidad de Sevilla.

Work in progress.

## Residual SAC training

Train SAC with a previously trained base regressor:

```bash
python -m scripts.train_sac \
  --regressor results/regressor/regressor_20261026_204905/best_model.pth \
  --seed 16
```

Each run is written to a timestamped directory under `results/rl/`. It includes
the run configuration and summary, the final policy and periodic checkpoints,
episode rewards (`logs/seed<seed>.monitor.csv`), Stable-Baselines3 training
metrics (`logs/progress.csv`), a per-timestep trace (`logs/training_steps.csv`)
with reward components, `dvBase`, SAC residual, applied delta-v, orbit radius
and randomized mass fraction, and reward/loss plots.

RL training uses the same mass distribution as the MPC initial-condition
batch: the first mascon is nominally 60% of total mass with a relative
variation of +/-25% (a 45%-75% fraction, uniformly sampled per episode).
The 30-minute SAC experiment uses one control action every 1800 seconds, for
20 actions per 10-hour episode. Its base policy is the 30-minute regressor
trained with seed 3. SAC retains its existing discount factor of `0.99`.
Impulse noise and pointing error are disabled for commands with magnitude at
or below the shared `1e-8 m/s` thruster-off threshold, just as in MPC rollouts.
During SAC training, crossing either shell boundary (22 km or 30 km) terminates
the episode at the crossing and gives the configured `-3.0` terminal reward.
The executed-delta-v reward penalty is scaled by `10.0` to give fuel use more
weight. To store a set of agents together, pass a common output directory and
use a distinct seed for each run; for example, pass `--output-dir` with
`results/rl/morepenaltyaction` to create `agent_<seed>` subdirectories there.

## SAC evaluation on the shared MPC batch

Evaluate the completed 30-minute SAC runs using the seed-3 regressor on the
same 500 initial conditions and mass-perturbed asteroid scenarios as the
30-minute MPC run:

```bash
python -m scripts.test_sac
```

The evaluation discovers only 30-minute SAC runs using the seed-3 regressor,
verifies their control cadence and shared model, and checks the batch against
the 30-minute MPC results. Each agent gets a
`tests/` directory containing `run_data.pkl` (all 500 full episode records and
the input scenario batch), `summary.json`, and PNG/PDF comparisons of outcomes,
delta-v, and orbit-radius histories against MPC. The MPC-compatible per-episode
actuation-noise seeds are reused, and the SAC policy is evaluated
deterministically.

Rebuild the test plots, including controller-computation-time figures, from
the saved test bundles:

```bash
python -m src.plots.plot_test_rl
```

The script discovers every `results/rl/sac_*/tests/run_data.pkl` and saves
PNG/PDF figures in that test's `tests/` directory. It uses the MPC reference
path recorded in each bundle for paired comparisons when that file is
available. To process selected bundles or specify a reference explicitly:

```bash
python -m src.plots.plot_test_rl \
  --run-data results/rl/sac_1_<timestamp>/tests/run_data.pkl \
  --mpc-reference results/mpc/<run>/run_data.pkl
```

Compare the ten tested SAC seeds on shared test scenarios and save the figures
under `results/plots_comparison_agents/`:

```bash
python -m src.plots.plots_SAC_agents
```

This creates PNG/PDF comparisons of final-radius distributions (including the
22 km and 30 km shell limits), radius histories, delta-v, outcomes, controller
latency, evaluation runtime, and action activations, plus an `agent_summary.csv`.

Compare the episode-reward histories from training for SAC seeds 1-10:

```bash
python -m src.plots.plot_sac_training
```

This saves `training_reward_all_agents.png` and `.pdf` in
`results/plots_comparison_agents/`.
