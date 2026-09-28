"""
Propensity model: estimates ê(s) ≈ π_b^marg(1|s) from (s, a) data.

Trained BEFORE representation learning and frozen during SPRL-cell training.
Used to compute the propensity homogeneity penalty L_prop.
"""

import torch
import torch.nn as nn
import numpy as np
from utils.device import resolve_device


class PropensityModel(nn.Module):
    """MLP propensity estimator: P(A=1|S)."""

    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s):
        return torch.sigmoid(self.net(s)).squeeze(-1)


def train_propensity(model, data_s, data_a, epochs=100, batch_size=256,
                     lr=1e-3, device=None, verbose=True):
    """
    Train propensity model via BCE on (s, a) data.

    Returns
    -------
    losses : list of float
    """
    device = resolve_device(device)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    s_t = torch.tensor(data_s, dtype=torch.float32, device=device)
    a_t = torch.tensor(data_a, dtype=torch.float32, device=device)
    n = len(s_t)
    bce = nn.BCELoss()
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            s_b, a_b = s_t[idx], a_t[idx]

            e_pred = model(s_b)
            loss = bce(e_pred, a_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Propensity epoch {epoch+1}/{epochs}: BCE={avg_loss:.4f}",
                  flush=True)

    model.eval()
    return losses


def get_propensities(model, data_s, device=None):
    """Get frozen propensity estimates ê(s) as numpy array."""
    device = resolve_device(device)
    model.eval()
    with torch.no_grad():
        s_t = torch.tensor(data_s, dtype=torch.float32, device=device)
        e_hat = model(s_t).cpu().numpy()
    return e_hat
