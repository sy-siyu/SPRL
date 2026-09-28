"""
Contrastive representation baseline (SimCLR-style).

Encoder: same MLP architecture as VAE encoder, maps S → Z.
Projection head: MLP(Z → 128 → 64), discarded after training.
Loss: InfoNCE with augmented positives (Gaussian noise perturbation).

Expected to absorb confounding information because contrastive learning
captures all structure in S, including U-correlated features.

Cite: Chen et al. (2020), "A Simple Framework for Contrastive Learning
of Visual Representations".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.device import resolve_device


class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Projection head (discarded after training)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        z = self.backbone(x)
        p = self.projector(z)
        return z, p

    def encode(self, x):
        return self.backbone(x)


class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.encoder = ContrastiveEncoder(input_dim, latent_dim, hidden_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        return self.encoder(x)

    def encode(self, x):
        return self.encoder.encode(x)


def info_nce_loss(p1, p2, temperature=0.5):
    """InfoNCE loss for a batch of positive pairs (p1[i], p2[i])."""
    batch_size = p1.shape[0]
    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=1)

    # Similarity matrix: all pairs
    representations = torch.cat([p1, p2], dim=0)  # (2B, d)
    sim_matrix = torch.mm(representations, representations.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = ~torch.eye(2 * batch_size, device=sim_matrix.device, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(~mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_sim = torch.sum(p1 * p2, dim=1) / temperature  # (B,)

    # For each anchor in first half, positive is in second half
    labels_top = torch.arange(batch_size, 2 * batch_size, device=sim_matrix.device)
    labels_bot = torch.arange(0, batch_size, device=sim_matrix.device)
    labels = torch.cat([labels_top, labels_bot])

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def train_contrastive(model, data_s, epochs=200, batch_size=256, lr=1e-3,
                      noise_std=0.1, temperature=0.5,
                      device=None, verbose=True):
    """
    Train contrastive model with Gaussian noise augmentation.

    Positive pair: (s, s + ε) where ε ~ N(0, noise_std²·I).
    """
    device = resolve_device(device)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    s_tensor = torch.tensor(data_s, dtype=torch.float32, device=device)
    n = len(s_tensor)
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = s_tensor[idx]

            # Two augmented views
            x1 = x + noise_std * torch.randn_like(x)
            x2 = x + noise_std * torch.randn_like(x)

            _, p1 = model(x1)
            _, p2 = model(x2)

            loss = info_nce_loss(p1, p2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Contrastive epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    model.eval()
    return losses


def get_contrastive_encoder_fn(model, device=None):
    """Return a numpy-in/numpy-out encoder function (backbone only, no projector)."""
    device = resolve_device(device)
    model.eval()

    @torch.no_grad()
    def encode(s):
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        z = model.encode(s_t)
        return z.cpu().numpy()

    return encode
