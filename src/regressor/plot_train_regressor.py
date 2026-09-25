"""
Post-process a regressor run from its run_data.pkl (no retraining needed).

Usage:
    python -m scripts.plot_regressor results/regressor_XXXX
    python -m scripts.plot_regressor results/regressor_XXXX/run_data.pkl --n-traj 6 --fmt pdf
    python -m scripts.plot_regressor <run> --threshold 0.4   # re-evaluate with another threshold

All figures are written to disk (Agg backend); nothing is shown on screen.
"""

import os
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_DATA = "#1f5fa8"      # blue: ground truth / data
C_ACTION = "#7b3fa0"    # purple: predicted actions
C_NEUTRAL = "0.65"

MODES = ("gated_hard", "gated_hard_0.5", "gated_soft", "regressor_only")
MAIN_MODE = "gated_hard"
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def as_2d(a):
    a = np.asarray(a)
    return a.reshape(len(a), -1)


def infer_traj_ids(steps):
    """Trajectory index per sample. Assumes samples are stored trajectory by
    trajectory and the step counter restarts at every new trajectory."""
    s = np.asarray(steps).reshape(len(steps), -1)[:, 0]
    new = np.r_[True, np.diff(s) <= 0]
    return np.cumsum(new) - 1


def component_names(k):
    return ["x", "y", "z"] if k == 3 else [str(i) for i in range(k)]


def cls_targets(labels, prob):
    """Binary targets with the same number of columns as the classifier output."""
    y = as_2d(labels).astype(bool)
    if prob.shape[1] == 1 and y.shape[1] > 1:
        y = y.any(axis=1, keepdims=True)
    return y


def apply_mode(prob, dv_raw, mode, thr):
    if mode == "gated_hard":
        return dv_raw * (prob >= thr)
    if mode == "gated_hard_0.5":
        return dv_raw * (prob >= 0.5)
    if mode == "gated_soft":
        return dv_raw * prob
    if mode == "regressor_only":
        return dv_raw
    raise ValueError(f"unknown mode: {mode}")


def predicted_dv(split, thr, mode=MAIN_MODE):
    return apply_mode(as_2d(split["prob"]), as_2d(split["dv_raw"]), mode, thr)


# ----------------------------------------------------------------------------
# Metrics (also imported by train_regressor.py)
# ----------------------------------------------------------------------------
def regression_metrics(dv_true, dv_pred):
    if len(dv_true) == 0:
        return None
    err = dv_pred - dv_true
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((dv_true - dv_true.mean(axis=0)) ** 2, axis=0)
    err_norm = np.linalg.norm(err, axis=1)
    return {
        "n":            int(len(err)),
        "rmse":         float(np.sqrt(np.mean(err ** 2))),
        "rmse_comp":    np.sqrt(ss_res / len(err)).tolist(),
        "mae":          float(np.mean(np.abs(err))),
        "mae_comp":     np.mean(np.abs(err), axis=0).tolist(),
        "r2":           float(1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-30)),
        "r2_comp":      (1.0 - ss_res / np.maximum(ss_tot, 1e-30)).tolist(),
        "rmse_norm":    float(np.sqrt(np.mean(err_norm ** 2))),
        "max_err_norm": float(err_norm.max()),
    }


def classification_metrics(y, prob, thr):
    y = np.asarray(y, dtype=bool).ravel()
    p = np.asarray(prob).ravel() >= thr
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / max(len(y), 1),
            "precision": precision, "recall": recall, "f1": f1}


def threshold_sweep(labels, prob, thresholds=None):
    prob = as_2d(prob)
    y = cls_targets(labels, prob)
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.98, 49)
    rows = [classification_metrics(y, prob, t) for t in thresholds]
    out = {"thresholds": np.asarray(thresholds)}
    for key in ("precision", "recall", "f1", "accuracy"):
        out[key] = np.array([r[key] for r in rows])
    out["best_threshold"] = float(thresholds[int(np.argmax(out["f1"]))])
    return out


def compute_split_metrics(split, thr):
    dv_true = as_2d(split["dv_true"])
    prob = as_2d(split["prob"])
    dv_raw = as_2d(split["dv_raw"])
    y = cls_targets(split["labels"], prob)
    active = y.any(axis=1)
    budget_true = float(np.linalg.norm(dv_true, axis=1).sum())

    out = {"cls": classification_metrics(y, prob, thr),
           "cls_0.5": classification_metrics(y, prob, 0.5),
           "n_active": int(active.sum()), "n_total": int(len(active)),
           "modes": {}}
    for mode in MODES:
        dv_pred = apply_mode(prob, dv_raw, mode, thr)
        budget_pred = float(np.linalg.norm(dv_pred, axis=1).sum())
        out["modes"][mode] = {
            "all":      regression_metrics(dv_true, dv_pred),
            "active":   regression_metrics(dv_true[active], dv_pred[active]),
            "inactive": regression_metrics(dv_true[~active], dv_pred[~active]),
            "dv_budget_true":    budget_true,
            "dv_budget_pred":    budget_pred,
            "dv_budget_rel_err": (budget_pred - budget_true) / max(budget_true, 1e-30),
        }
    return out


def format_summary(run, mode=MAIN_MODE):
    m, meta = run["metrics"], run["meta"]
    nan = float("nan")

    def g(d, key):
        return d[key] if d else nan

    bar = "=" * 86
    lines = [bar, "REGRESSOR SUMMARY", bar,
             f"run dir        : {meta.get('out_dir')}",
             f"git commit     : {meta.get('git_commit')}   seed: {meta.get('seed')}",
             f"epochs trained : {meta.get('epochs_trained')}   "
             f"best val loss: {meta.get('best_val', nan):.4e}",
             f"threshold      : {run['best_threshold']:.3f} (F1-optimal on VAL)",
             f"main mode      : {mode}", "",
             f"{'split':<7}{'RMSE':>11}{'RMSE_act':>11}{'MAE':>11}{'R2':>8}"
             f"{'F1':>7}{'Prec':>7}{'Rec':>7}{'dV budget':>12}"]
    for s in SPLITS:
        if s not in m:
            continue
        mm, c = m[s]["modes"][mode], m[s]["cls"]
        lines.append(f"{s.upper():<7}{mm['all']['rmse']:>11.3e}{g(mm['active'], 'rmse'):>11.3e}"
                     f"{mm['all']['mae']:>11.3e}{mm['all']['r2']:>8.3f}"
                     f"{c['f1']:>7.3f}{c['precision']:>7.3f}{c['recall']:>7.3f}"
                     f"{100 * mm['dv_budget_rel_err']:>+11.2f}%")
    if "test" in m:
        lines += ["", "Mode comparison (TEST)",
                  f"{'mode':<17}{'RMSE':>11}{'RMSE_act':>11}{'RMSE_inact':>12}{'dV budget':>12}"]
        for md in MODES:
            mm = m["test"]["modes"][md]
            lines.append(f"{md:<17}{mm['all']['rmse']:>11.3e}{g(mm['active'], 'rmse'):>11.3e}"
                         f"{g(mm['inactive'], 'rmse'):>12.3e}"
                         f"{100 * mm['dv_budget_rel_err']:>+11.2f}%")
        c = m["test"]["cls"]
        lines.append(f"\nTEST confusion  TP={c['tp']}  FP={c['fp']}  FN={c['fn']}  TN={c['tn']}")
    lines.append(bar)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def _save(fig, out_dir, name, fmts):
    os.makedirs(out_dir, exist_ok=True)
    for ext in fmts:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_training_curves(history, lambda_reg, out_dir, fmts):
    series = {k: np.asarray(v, dtype=float) for k, v in history.items()
              if np.ndim(v) == 1 and len(v) > 1}
    bases = sorted({k[len("train_"):] for k in series if k.startswith("train_")})
    panels = [(b, series.get("train_" + b), series.get("val_" + b)) for b in bases]
    if {"cls", "reg"} <= set(bases) and "total" not in bases and lambda_reg is not None:
        tr = series["train_cls"] + lambda_reg * series["train_reg"]
        va = (series["val_cls"] + lambda_reg * series["val_reg"]
              if "val_cls" in series and "val_reg" in series else None)
        panels.append(("total", tr, va))
    others = [k for k in series if not k.startswith(("train_", "val_"))]

    best_ep = None
    total = [p for p in panels if p[0] == "total"]
    if total and total[0][2] is not None:
        best_ep = int(np.argmin(total[0][2]))

    n = len(panels) + len(others)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 2.7), squeeze=False)
    for ax, (name, tr, va) in zip(axes[0], panels):
        if tr is not None:
            ax.plot(np.arange(len(tr)), tr, color=C_DATA, lw=1, label="train")
        if va is not None:
            ax.plot(np.arange(len(va)), va, color=C_ACTION, lw=1, label="val")
        if best_ep is not None:
            ax.axvline(best_ep, color=C_NEUTRAL, ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(name)
    for ax, key in zip(axes[0][len(panels):], others):
        ax.plot(series[key], color=C_NEUTRAL, lw=1)
        ax.set_xlabel("epoch")
        ax.set_title(key)
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "training_curves", fmts)


def plot_parity(split, name, thr, out_dir, fmts, mode=MAIN_MODE):
    dv_true = as_2d(split["dv_true"])
    dv_pred = predicted_dv(split, thr, mode)
    active = cls_targets(split["labels"], as_2d(split["prob"])).any(axis=1)
    k = dv_true.shape[1]
    comps = component_names(k)

    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.1), squeeze=False)
    for j, ax in enumerate(axes[0]):
        t, p = dv_true[:, j], dv_pred[:, j]
        ax.scatter(t[~active], p[~active], s=2, color=C_NEUTRAL, alpha=0.4, rasterized=True)
        ax.scatter(t[active], p[active], s=2, color=C_DATA, alpha=0.5, rasterized=True)
        lim = np.max(np.abs(np.r_[t, p])) * 1.05 or 1.0
        ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        rmse = np.sqrt(np.mean((p - t) ** 2))
        ss_tot = max(np.sum((t - t.mean()) ** 2), 1e-30)
        r2 = 1 - np.sum((p - t) ** 2) / ss_tot
        ax.text(0.04, 0.96, f"RMSE = {rmse:.2e}\n$R^2$ = {r2:.3f}",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel(rf"$\Delta v_{{{comps[j]}}}$ true")
        ax.set_ylabel(rf"$\Delta v_{{{comps[j]}}}$ predicted")
    fig.tight_layout()
    _save(fig, out_dir, f"parity_{name}", fmts)


def plot_threshold_sweep(sweep, out_dir, fmts):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    t = sweep["thresholds"]
    ax.plot(t, sweep["precision"], color=C_DATA, lw=1.2, label="precision")
    ax.plot(t, sweep["recall"], color=C_ACTION, lw=1.2, label="recall")
    ax.plot(t, sweep["f1"], color="k", lw=1.4, label="F1")
    ax.axvline(sweep["best_threshold"], color=C_NEUTRAL, ls="--", lw=0.8)
    ax.set_xlabel("classification threshold")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "threshold_sweep_val", fmts)


def plot_probability_distribution(split, name, thr, out_dir, fmts):
    prob = as_2d(split["prob"])
    y = cls_targets(split["labels"], prob)
    bins = np.linspace(0, 1, 51)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(prob[~y], bins=bins, color=C_NEUTRAL, alpha=0.8, label="inactive")
    ax.hist(prob[y], bins=bins, color=C_DATA, alpha=0.8, label="active")
    ax.axvline(thr, color="k", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("predicted firing probability")
    ax.set_ylabel("samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, f"probability_{name}", fmts)


def plot_error_analysis(split, name, thr, out_dir, fmts, mode=MAIN_MODE):
    dv_true = as_2d(split["dv_true"])
    err = np.linalg.norm(predicted_dv(split, thr, mode) - dv_true, axis=1)
    mag = np.linalg.norm(dv_true, axis=1)
    active = cls_targets(split["labels"], as_2d(split["prob"])).any(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    bins = np.logspace(np.log10(max(err[err > 0].min(), 1e-12)) if np.any(err > 0) else -12,
                       np.log10(max(err.max(), 1e-11)), 60)
    axes[0].hist(err[~active], bins=bins, color=C_NEUTRAL, alpha=0.8, label="inactive")
    axes[0].hist(err[active], bins=bins, color=C_DATA, alpha=0.8, label="active")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\|\Delta v_{pred} - \Delta v_{true}\|$")
    axes[0].set_ylabel("samples")
    axes[0].legend(frameon=False)

    axes[1].scatter(mag[active], err[active], s=2, color=C_DATA, alpha=0.5, rasterized=True)
    axes[1].set_xlabel(r"$\|\Delta v_{true}\|$")
    axes[1].set_ylabel(r"$\|\Delta v_{pred} - \Delta v_{true}\|$")
    fig.tight_layout()
    _save(fig, out_dir, f"error_analysis_{name}", fmts)


def plot_mode_comparison(metrics, name, out_dir, fmts):
    mm = metrics["modes"]
    keys = [("RMSE (all)", lambda d: d["all"]["rmse"]),
            ("RMSE (active)", lambda d: d["active"]["rmse"] if d["active"] else np.nan),
            (r"$\Delta v$ budget error [%]", lambda d: 100 * d["dv_budget_rel_err"])]
    x = np.arange(len(MODES))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.0))
    for ax, (title, fn) in zip(axes, keys):
        vals = [fn(mm[md]) for md in MODES]
        ax.bar(x, vals, color=[C_ACTION if md == MAIN_MODE else C_NEUTRAL for md in MODES])
        ax.set_xticks(x)
        ax.set_xticklabels(MODES, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, out_dir, f"mode_comparison_{name}", fmts)


def plot_confusion(metrics, name, out_dir, fmts):
    c = metrics["cls"]
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]}", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "k")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no fire", "fire"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no fire", "fire"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    _save(fig, out_dir, f"confusion_{name}", fmts)


def plot_trajectories(split, name, thr, out_dir, fmts, n_traj=4, seed=0, mode=MAIN_MODE):
    dv_true = as_2d(split["dv_true"])
    dv_pred = predicted_dv(split, thr, mode)
    prob = as_2d(split["prob"])
    tids = np.asarray(split["traj_id"])
    steps = as_2d(split["steps"])[:, 0]
    comps = component_names(dv_true.shape[1])

    rng = np.random.default_rng(seed)
    uniq = np.unique(tids)
    picks = rng.choice(uniq, size=min(n_traj, len(uniq)), replace=False)
    traj_dir = os.path.join(out_dir, f"trajectories_{name}")

    for tid in picks:
        idx = np.flatnonzero(tids == tid)
        t = steps[idx]
        k = dv_true.shape[1]
        fig, axes = plt.subplots(k + 1, 1, figsize=(6.0, 1.5 * (k + 1)), sharex=True)
        for j in range(k):
            ax = axes[j]
            ax.axhline(0, color=C_NEUTRAL, lw=0.5)
            ax.plot(t, dv_true[idx, j], "o", ms=3, color=C_DATA, label="OCP")
            ax.plot(t, dv_pred[idx, j], "x", ms=3, color=C_ACTION, label="network")
            ax.set_ylabel(rf"$\Delta v_{{{comps[j]}}}$")
        axes[0].legend(frameon=False, ncol=2, fontsize=8)
        axes[-1].plot(t, prob[idx].max(axis=1), color=C_ACTION, lw=1)
        axes[-1].axhline(thr, color="k", ls="--", lw=0.7)
        axes[-1].set_ylim(-0.02, 1.02)
        axes[-1].set_ylabel("p(fire)")
        axes[-1].set_xlabel("step")
        fig.tight_layout()
        _save(fig, traj_dir, f"traj_{int(tid):04d}", fmts)


def make_all_plots(run, out_dir, fmts=("png", "pdf"), n_traj=4, threshold=None, mode=MAIN_MODE):
    try:
        from src.utils.plotting_utils import set_paper_style
        set_paper_style()
    except ImportError:
        plt.rcParams.update({"font.family": "serif"})

    thr = run["best_threshold"] if threshold is None else threshold
    metrics = run["metrics"] if threshold is None else {
        s: compute_split_metrics(run["splits"][s], thr) for s in run["splits"]}

    plot_training_curves(run["history"], run["meta"].get("lambda_reg"), out_dir, fmts)
    plot_threshold_sweep(run["threshold_sweep"], out_dir, fmts)
    for s, split in run["splits"].items():
        plot_parity(split, s, thr, out_dir, fmts, mode)
    test = run["splits"]["test"]
    plot_probability_distribution(test, "test", thr, out_dir, fmts)
    plot_error_analysis(test, "test", thr, out_dir, fmts, mode)
    plot_mode_comparison(metrics["test"], "test", out_dir, fmts)
    plot_confusion(metrics["test"], "test", out_dir, fmts)
    plot_trajectories(test, "test", thr, out_dir, fmts, n_traj=n_traj,
                      seed=run["meta"].get("seed", 0), mode=mode)
    print(f"[plots] saved to {out_dir}")
    return metrics


def load_run(path):
    if os.path.isdir(path):
        path = os.path.join(path, "run_data.pkl")
    with open(path, "rb") as f:
        return pickle.load(f), os.path.dirname(os.path.abspath(path))


def main():
    parser = argparse.ArgumentParser(description="Plot a regressor run from run_data.pkl.")
    parser.add_argument("run", help="run directory or path to run_data.pkl")
    parser.add_argument("--out", default=None, help="output dir (default: <run>/plots)")
    parser.add_argument("--fmt", nargs="+", default=["png", "pdf"])
    parser.add_argument("--n-traj", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the VAL F1-optimal threshold")
    parser.add_argument("--mode", default=MAIN_MODE, choices=MODES)
    args = parser.parse_args()

    run, run_dir = load_run(args.run)
    out_dir = args.out or os.path.join(run_dir, "plots")
    metrics = make_all_plots(run, out_dir, args.fmt, args.n_traj, args.threshold, args.mode)
    if args.threshold is not None:
        run = dict(run, metrics=metrics, best_threshold=args.threshold)
    print(format_summary(run, mode=args.mode))


if __name__ == "__main__":
    main()