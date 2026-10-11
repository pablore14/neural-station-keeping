import pickle
import random
from src.regressor.data_preparation.dataset_preparation import JointDataset, process_results
from src.regressor.data_preparation.state_scaling import adimensionalize_state
import numpy as np
from src.regressor.plot_train_regressor import infer_traj_ids
from src.config import DELTA
def load_split(data_path, seed):
    """Load the pickle and split trajectories into train / val / test (70/15/15)."""
    with open(data_path, "rb") as f:
        ocp = pickle.load(f)
    results = ocp["results_dict"]

    rng = random.Random(seed)      # local RNG, does not touch the global state
    rng.shuffle(results)

    n = len(results)
    train = results[:int(0.70 * n)]
    val   = results[int(0.70 * n):int(0.85 * n)]
    test  = results[int(0.85 * n):]
    print(f"[data] total={n}  train={len(train)}  val={len(val)}  test={len(test)}")
    return train, val, test


def build_split(results):
    """Turn a list of trajectories into a dict with everything needed downstream."""
    states, dv, steps = process_results(results)
    features = adimensionalize_state(states, steps)
    dv = np.asarray(dv, dtype=np.float32)
    return {
        "x":       np.asarray(features, dtype=np.float32),
        "states":  np.asarray(states),                 # dimensional, original dtype
        "steps":   np.asarray(steps),
        "traj_id": infer_traj_ids(steps),
        "dv_true": dv,
        "labels":  (np.abs(dv) > DELTA).astype(np.float32),
    }