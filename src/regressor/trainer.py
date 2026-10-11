"""
Trainer for the hybrid policy network.

Trains the two heads jointly: binary cross-entropy (or optional focal loss) on
the classification head, plus a Huber loss on the regression head, combined as

    total = cls_loss + lambda_reg * reg_loss

Handles the optimizer, LR scheduling, gradient clipping, early stopping and
saving the best model by validation loss.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader




class Trainer:
    """Joint classifier + regressor trainer.

    Args:
        model:        the PolicyNet to train.
        train_ds:     training JointDataset.
        val_ds:       validation JointDataset.
        epochs:       maximum number of epochs.
        batch_size:   mini-batch size.
        lr:           initial learning rate.
        weight_decay: AdamW weight decay.
        lambda_reg:   weight of the regression loss relative to the classifier.
        pos_weight:   positive-class weight for the classification loss.
        patience:     early-stopping patience (epochs without improvement).
        huber_beta:   beta of the SmoothL1 (Huber) regression loss.
        grad_clip:    max gradient norm.
        use_focal:    if True, use focal loss instead of BCE for the classifier.
        focal_gamma:  focal-loss focusing parameter.
        focal_alpha:  focal-loss positive-class weighting.
    """

    def __init__(self, model, train_ds, val_ds, *,
                 epochs=5000, batch_size=256, lr=1e-3, weight_decay=1e-4,
                 lambda_reg=300.0, pos_weight=None, patience=1000,
                 huber_beta=0.01, grad_clip=1.0,
                 use_focal=False, focal_gamma=2.0, focal_alpha=None):
        self.model      = model
        self.epochs     = epochs
        self.patience   = patience
        self.lambda_reg = lambda_reg
        self.grad_clip  = grad_clip

        self.opt = optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=350,
            min_lr=1e-6, threshold=1e-4)

        if use_focal:
            print(f"[cls loss] FocalLoss  gamma={focal_gamma}  alpha={focal_alpha}")
            # self.cls_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha,
            #                           pos_weight=pos_weight)
        else:
            print(f"[cls loss] BCEWithLogitsLoss  pos_weight={pos_weight}")
            self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.reg_loss = nn.SmoothL1Loss(beta=huber_beta)

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    def _losses(self, states, labels, dv):
        """Return (total_loss, cls_value, reg_value) for one batch."""
        logit, u_hat = self.model(states)
        loss_cls = self.cls_loss(logit, labels)
        loss_reg = self.reg_loss(u_hat, dv)
        total = loss_cls + self.lambda_reg * loss_reg
        return total, loss_cls.item(), loss_reg.item()

    def _run_epoch(self, loader, train):
        """Run one epoch; return mean (cls, reg) losses over the loader."""
        self.model.train(train)
        sum_cls = sum_reg = 0.0
        for states, labels, dv in loader:
            if train:
                self.opt.zero_grad()
            total, cls_val, reg_val = self._losses(states, labels, dv)
            if train:
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
            sum_cls += cls_val
            sum_reg += reg_val
        n = len(loader)
        return sum_cls / n, sum_reg / n

    def train(self, out_dir):
        history = {"train_cls": [], "val_cls": [], "train_reg": [], "val_reg": []}
        best_val, no_improve = float("inf"), 0

        for epoch in range(self.epochs):
            if epoch % 100 == 0:
                print(f"epoch {epoch}")
            print(f"epoch {epoch}", flush = True)
            train_cls, train_reg = self._run_epoch(self.train_loader, train=True)
            history["train_cls"].append(train_cls)
            history["train_reg"].append(train_reg)

            with torch.no_grad():
                val_cls, val_reg = self._run_epoch(self.val_loader, train=False)
            val_total = val_cls + self.lambda_reg * val_reg
            self.scheduler.step(val_total)
            history["val_cls"].append(val_cls)
            history["val_reg"].append(val_reg)

            if val_total < best_val:
                best_val   = val_total
                no_improve = 0
                torch.save(self.model.state_dict(),
                           os.path.join(out_dir, "best_model.pth"))
            else:
                no_improve += 1

            if epoch % 25 == 0:
                print(f"epoch {epoch:>5d}  val_cls={val_cls:.15e}  "
                      f"val_reg={val_reg:.6f}  best={best_val:.15e}  "
                      f"patience={no_improve}/{self.patience}")

            if no_improve >= self.patience:
                print(f"[early stopping at epoch {epoch}]")
                break

        self.best_val = best_val
        return history