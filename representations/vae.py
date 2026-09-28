"""
Variational Autoencoder (VAE) baseline representation.

Encodes S → Z via an MLP encoder, reconstructs via MLP decoder.
Loss: ELBO = reconstruction (MSE) + KL divergence.

This baseline is expected to absorb confounding information because
reconstruction incentivises capturing everything predictive of S,
including U-correlated features.
"""

import torch
import torch.nn as nn
import numpy as np
from utils.device import resolve_device


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
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


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dim)
        self.decoder = VAEDecoder(latent_dim, input_dim, hidden_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def encode(self, x):
        """Deterministic encoding (use mean)."""
        mu, _ = self.encoder(x)
        return mu

    def loss(self, x, x_recon, mu, logvar, beta=1.0):
        """ELBO loss = reconstruction + β · KL."""
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(vae, data_s, epochs=200, batch_size=256, lr=1e-3, beta=1.0,
              device=None, verbose=True):
    """
    Train VAE on state data.

    Parameters
    ----------
    vae : VAE
    data_s : array (n, d)
        Raw states.
    epochs : int
    batch_size : int
    lr : float
    beta : float
        KL weight.
    device : str

    Returns
    -------
    losses : list of (total, recon, kl) per epoch
    """
    device = resolve_device(device)
    vae = vae.to(device)
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    s_tensor = torch.tensor(data_s, dtype=torch.float32, device=device)
    n = len(s_tensor)
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = s_tensor[idx]

            x_recon, mu, logvar = vae(x)
            loss, recon, kl = vae.loss(x, x_recon, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            n_batches += 1

        avg = (total_loss / n_batches, total_recon / n_batches, total_kl / n_batches)
        losses.append(avg)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  VAE epoch {epoch+1}/{epochs}: loss={avg[0]:.4f} "
                  f"recon={avg[1]:.4f} kl={avg[2]:.4f}")

    vae.eval()
    return losses


def get_vae_encoder_fn(vae, device=None):
    """Return a numpy-in/numpy-out encoder function."""
    device = resolve_device(device)
    vae.eval()

    @torch.no_grad()
    def encode(s):
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        z = vae.encode(s_t)
        return z.cpu().numpy()

    return encode
