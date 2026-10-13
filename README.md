# Neural Guidance Policies for Autonomous Station-Keeping Around Asteroids

Code for the master's thesis and paper by Pablo Redondo Amaro,
Universidad de Sevilla.

Work in progress.

## Residual SAC training

Train SAC with a previously trained base regressor:

```bash
python -m scripts.train_sac \
  --regressor results/regressor/<run>/best_model.pth \
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
Impulse noise and pointing error are disabled for commands with magnitude at
or below the shared `1e-8 m/s` thruster-off threshold, just as in MPC rollouts.
During SAC training, crossing either shell boundary (22 km or 30 km) terminates
the episode at the crossing and gives the configured `-3.0` terminal reward.
