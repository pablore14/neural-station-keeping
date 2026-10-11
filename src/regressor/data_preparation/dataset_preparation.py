"""
Dataset and data preparation for training the hybrid policy network.

Turns raw trajectory results into training tensors: the non-dimensional state
features, the binary "act / do not act" labels, and the target delta-v.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.config import DELTA


class JointDataset(Dataset):
    """Dataset yielding (state, label, delta_v) triples for the two heads."""

    def __init__(self, states, labels, dv):
        self.states = torch.from_numpy(states).float()
        self.labels = torch.from_numpy(labels).float()
        self.dv     = torch.from_numpy(dv).float()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx], self.dv[idx]


def make_balanced_sampler(dv, w_neg_boost=5.0):
    """Weighted sampler that oversamples negative-delta-v points.

    Active points with negative delta-v are rarer, so they get a higher weight
    to appear more often per batch. Note: use shuffle=False in the DataLoader
    when passing a sampler.
    """
    w = np.ones(len(dv), dtype=np.float32)
    w[dv < -DELTA]        = w_neg_boost
    w[dv > DELTA]         = 1.0
    w[np.abs(dv) <= DELTA] = 0.5
    return WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)


def process_results(results_list):
    """Flatten a list of successful trajectories into training arrays.

    Args:
        results_list: list of trajectory dicts, each with keys
                      "status", "X" (states) and "dV" (delta-v).

    Returns:
        states: array (N, 6), stacked physical states.
        dv:     array (N,),  target delta-v (small values snapped to 0).
        steps:  array (N,),  integer time-step index within each trajectory.
    """
    states_list, dv_list, steps_list = [], [], []
    for r in results_list:
        if r["status"] != "success":
            continue
        n = len(r["dV"])
        states_list.append(r["X"])
        dv_list.append(r["dV"][:, 0])
        steps_list.append(np.arange(n, dtype=np.int64))

    states = np.concatenate(states_list, axis=0)
    dv     = np.concatenate(dv_list,     axis=0)
    steps  = np.concatenate(steps_list,  axis=0)

    dv = np.where(np.abs(dv) < DELTA, 0.0, dv)
    return states, dv, steps