"""
Balanced Representation baseline (MMD penalty).

Encoder: same MLP architecture as VAE/SPRL encoder, maps S → Z.
Reward predictor: same as SPRL reward predictor.
MMD penalty: match P(Z|A=0) and P(Z|A=1) using Gaussian kernel MMD.
Loss: L = L_reward + α_MMD · MMD(Z_{A=0}, Z_{A=1})

This baseline (from counterfactual reasoning literature) balances
treatment groups in representation space. Unlike SPRL, it matches
action-conditional distributions rather than adversarially decorrelating.

Cite: Shalit et al. (2017), "Estimating Individual Treatment Effects
with Neural Networks".
"""

import torch
import torch.nn as nn
import numpy as np
from utils.device import resolve_device


class BalancedEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class BalancedRewardPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, a):
        za = torch.cat([z, a.unsqueeze(-1) if a.dim() == 1 else a], dim=-1)
        return self.net(za).squeeze(-1)


class BalancedRep(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.encoder = BalancedEncoder(input_dim, latent_dim, hidden_dim)
        self.reward_predictor = BalancedRewardPredictor(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, s, a):
        z = self.encoder(s)
        r_pred = self.reward_predictor(z, a)
        return z, r_pred

    def encode(self, x):
        return self.encoder(x)


def gaussian_kernel_matrix(x, y, sigma=1.0):
    """Compute Gaussian kernel matrix K(x_i, y_j)."""
    x_sq = torch.sum(x ** 2, dim=1, keepdim=True)
    y_sq = torch.sum(y ** 2, dim=1, keepdim=True)
    dist = x_sq + y_sq.t() - 2 * torch.mm(x, y.t())
    return torch.exp(-dist / (2 * sigma ** 2))


def compute_mmd(z0, z1, sigma=1.0):
    """
    Compute MMD^2 between two sets of samples using Gaussian kernel.

    MMD^2 = E[k(z0,z0')] + E[k(z1,z1')] - 2·E[k(z0,z1)]
    """
    if len(z0) < 2 or len(z1) < 2:
        return torch.tensor(0.0, device=z0.device)

    k00 = gaussian_kernel_matrix(z0, z0, sigma)
    k11 = gaussian_kernel_matrix(z1, z1, sigma)
    k01 = gaussian_kernel_matrix(z0, z1, sigma)

    n0, n1 = len(z0), len(z1)

    # Unbiased estimate: exclude diagonal for same-sample terms
    mmd = (k00.sum() - k00.trace()) / (n0 * (n0 - 1)) \
        + (k11.sum() - k11.trace()) / (n1 * (n1 - 1)) \
        - 2 * k01.mean()

    return mmd


def train_balanced(model, data_s, data_a, data_r, alpha_mmd=1.0,
                   epochs=200, batch_size=256, lr=1e-3, mmd_sigma=1.0,
                   device=None, verbose=True):
    """
    Train balanced representation model.

    Loss = MSE(r_pred, r) + α_MMD · MMD(Z_{A=0}, Z_{A=1})
    """
    device = resolve_device(device)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    s_t = torch.tensor(data_s, dtype=torch.float32, device=device)
    a_t = torch.tensor(data_a, dtype=torch.float32, device=device)
    r_t = torch.tensor(data_r, dtype=torch.float32, device=device)
    n = len(s_t)
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_reward = 0.0
        epoch_mmd = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            s_b, a_b, r_b = s_t[idx], a_t[idx], r_t[idx]

            z, r_pred = model(s_b, a_b)
            loss_reward = nn.functional.mse_loss(r_pred, r_b)

            # MMD between Z|A=0 and Z|A=1
            mask_0 = a_b == 0
            mask_1 = a_b == 1
            if mask_0.sum() >= 2 and mask_1.sum() >= 2:
                loss_mmd = compute_mmd(z[mask_0], z[mask_1], sigma=mmd_sigma)
            else:
                loss_mmd = torch.tensor(0.0, device=device)

            loss = loss_reward + alpha_mmd * loss_mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_reward += loss_reward.item()
            epoch_mmd += loss_mmd.item()
            n_batches += 1

        avg_reward = epoch_reward / n_batches
        avg_mmd = epoch_mmd / n_batches
        losses.append({'reward': avg_reward, 'mmd': avg_mmd})

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Balanced epoch {epoch+1}/{epochs}: "
                  f"reward={avg_reward:.4f} mmd={avg_mmd:.4f}")

    model.eval()
    return losses


def get_balanced_encoder_fn(model, device=None):
    """Return a numpy-in/numpy-out encoder function."""
    device = resolve_device(device)
    model.eval()

    @torch.no_grad()
    def encode(s):
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        z = model.encode(s_t)
        return z.cpu().numpy()

    return encode
