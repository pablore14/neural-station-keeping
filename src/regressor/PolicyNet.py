
"""
Hybrid policy network (base regressor).

Single source of truth for the PolicyNet architecture. Import it wherever the
network is needed (training, evaluation, environment) instead of redefining it:

    from src.models.policy_net import PolicyNet

The network has two heads sharing a common backbone:
  - classification head (head_cls): predicts a logit for "act / do not act"
  - regression head (head_reg):     predicts the tangential delta-v magnitude
"""

import torch
import torch.nn as nn

from src.config import DV_MAX_REG


# Default architecture hyperparameters.
# Kept here (not scattered across scripts) so every consumer builds the exact
# same network. Override via constructor arguments only when experimenting.
STATE_DIM  = 8
HIDDEN_DIM = 256
P_DROP     = 0.15    # dropout in backbone and regression head
P_DROP_CLS = 0.30    # dropout in classification head


class PolicyNet(nn.Module):
    """Hybrid classifier + regressor for the base delta-v policy.

    Args:
        state_dim:  dimension of the input state vector.
        hidden_dim: width of the hidden layers.
        p_drop:     dropout probability in backbone and regression head.
        p_drop_cls: dropout probability in the classification head.
        dv_max:     maximum delta-v magnitude (m/s); scales the tanh output.

    Forward returns:
        logit: raw classification score (apply sigmoid for probability).
        u_hat: predicted delta-v in [-dv_max, dv_max] (m/s).
    """

    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM,
                 p_drop=P_DROP, p_drop_cls=P_DROP_CLS, dv_max=DV_MAX_REG):
        super().__init__()
        self.dv_max = dv_max

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(p_drop),
        )
        self.head_cls = nn.Sequential(
            nn.Dropout(p_drop_cls),
            nn.Linear(hidden_dim, 1),
        )
        self.head_reg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        h     = self.backbone(x)
        logit = self.head_cls(h).squeeze(-1)
        u_hat = torch.tanh(self.head_reg(h)).squeeze(-1) * self.dv_max
        return logit, u_hat