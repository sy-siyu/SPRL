"""
Autoencoder (AE) baseline representation.

Same architecture as VAE but without KL divergence — pure reconstruction loss.
Expected to absorb confounding information similarly to VAE.
"""

import torch
import torch.nn as nn
import numpy as np
from utils.device import resolve_device


class AEEncoder(nn.Module):
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


class AEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class AE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.encoder = AEEncoder(input_dim, latent_dim, hidden_dim)
        self.decoder = AEDecoder(latent_dim, input_dim, hidden_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        return self.encoder(x)


def train_ae(ae, data_s, epochs=200, batch_size=256, lr=1e-3,
             device=None, verbose=True):
    """Train AE on state data with MSE reconstruction loss."""
    device = resolve_device(device)
    ae = ae.to(device)
    ae.train()
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)

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

            x_recon, z = ae(x)
            loss = nn.functional.mse_loss(x_recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  AE epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    ae.eval()
    return losses


def get_ae_encoder_fn(ae, device=None):
    """Return a numpy-in/numpy-out encoder function."""
    device = resolve_device(device)
    ae.eval()

    @torch.no_grad()
    def encode(s):
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        z = ae.encode(s_t)
        return z.cpu().numpy()

    return encode
