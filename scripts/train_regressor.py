"""
Train the hybrid base regressor (classifier + delta-v regressor).

Usage:
    python -u -m scripts.train_regressor --data path/to/results.pkl --seed 42 [--plot]

Trains the network, evaluates it on train/val/test and dumps everything needed
for post-processing to <run_dir>/run_data.pkl (loss history, ground truth and
raw network outputs per split, threshold sweep and metrics). Figures are made
separately, without retraining:
    python -m scripts.plot_regressor <run_dir>
"""

import os
import json
import pickle
import random
import argparse
import datetime

import numpy as np
import torch

from src.config import RESULTS_DIR_REG, DELTA, DV_MAX_REG
from src.regressor import PolicyNet
from src.regressor.trainer import Trainer
from src.regressor.data_preparation.dataset_preparation import JointDataset, process_results
from src.regressor.data_preparation.state_scaling import adimensionalize_state
from src.utils.tracking import get_git_commit, make_run_dir
from src.regressor.plot_train_regressor import (compute_split_metrics, format_summary,
                                    infer_traj_ids, threshold_sweep, MAIN_MODE)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the hybrid base regressor.")
    parser.add_argument("--data", required=True,
                        help="path to the .pkl with 'results_dict'")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lambda-reg", type=float, default=200.0)
    parser.add_argument("--plot", action="store_true",
                        help="also generate all figures from run_data.pkl at the end")
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


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


@torch.no_grad()
def predict(model, x, batch_size=8192):
    """Raw network outputs: classifier logit/probability and ungated delta-v."""
    model.eval()
    device = next(model.parameters()).device
    logits, probs, dvs = [], [], []
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device=device)
        logit, dv = model(xb)   # NOTE: adapt if PolicyNet.forward returns something else
        logits.append(logit.cpu().numpy())
        probs.append(torch.sigmoid(logit).cpu().numpy())
        dvs.append(dv.cpu().numpy())
    n = len(x)
    return {
        "logit":  np.concatenate(logits).reshape(n, -1),
        "prob":   np.concatenate(probs).reshape(n, -1),
        "dv_raw": np.concatenate(dvs).reshape(n, -1),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- Data ---
    train_results, val_results, test_results = load_split(args.data, args.seed)
    splits = {"train": build_split(train_results),
              "val":   build_split(val_results),
              "test":  build_split(test_results)}

    tr, va = splits["train"], splits["val"]

    # Positive-class weight to counter the active/inactive imbalance.
    n_pos = tr["labels"].sum()
    n_neg = tr["labels"].size - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
    print(f"[classes] pos={int(n_pos)}  neg={int(n_neg)}  "
          f"pos_weight={pos_weight.item():.2f}")

    train_ds = JointDataset(tr["x"], tr["labels"], tr["dv_true"])
    val_ds   = JointDataset(va["x"], va["labels"], va["dv_true"])

    # --- Output directory + config snapshot ---
    out_dir = make_run_dir(RESULTS_DIR_REG, prefix="regressor")
    state_dim = tr["x"].shape[1]

    config = {
        "git_commit":  get_git_commit(),
        "seed":        args.seed,
        "state_dim":   state_dim,
        "hidden_dim":  args.hidden_dim,
        "epochs":      args.epochs,
        "lambda_reg":  args.lambda_reg,
        "dv_max":      DV_MAX_REG,
        "delta":       DELTA,
        "loss":        "BCEWithLogitsLoss + lambda_reg * SmoothL1Loss",
        "data_path":   args.data,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- Model + training ---
    model = PolicyNet(state_dim=state_dim, hidden_dim=args.hidden_dim,
                      dv_max=DV_MAX_REG)
    trainer = Trainer(model, train_ds, val_ds,
                      epochs=args.epochs, lambda_reg=args.lambda_reg,
                      pos_weight=pos_weight)
    history = trainer.train(out_dir)

    # Reload the best checkpoint before evaluating.
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pth"),
                                     map_location=device))

    # --- Raw predictions for every split ---
    for split in splits.values():
        split.update(predict(model, split["x"]))

    # --- Threshold (F1-optimal on VAL) + metrics ---
    sweep = threshold_sweep(va["labels"], va["prob"])
    best_threshold = sweep["best_threshold"]
    metrics = {name: compute_split_metrics(s, best_threshold)
               for name, s in splits.items()}

    # --- Everything needed to re-plot / re-evaluate without retraining ---
    meta = dict(config)
    meta.update({
        "out_dir":        out_dir,
        "created":        datetime.datetime.now().isoformat(timespec="seconds"),
        "epochs_trained": len(history["train_cls"]),
        "best_val":       float(trainer.best_val),
    })
    run_data = {
        "meta":            meta,
        "history":         {k: np.asarray(v) for k, v in history.items()},
        "splits":          splits,
        "threshold_sweep": sweep,
        "best_threshold":  best_threshold,
        "metrics":         metrics,
    }
    pkl_path = os.path.join(out_dir, "run_data.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(run_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Lightweight JSON copy of the scalars (no arrays).
    summary = {"meta": meta, "best_threshold": best_threshold, "metrics": metrics}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(format_summary(run_data, mode=MAIN_MODE))
    print(f"[done] run data: {pkl_path}")

    if args.plot:
        from scripts.plot_regressor import make_all_plots
        make_all_plots(run_data, os.path.join(out_dir, "plots"))


if __name__ == "__main__":
    main()