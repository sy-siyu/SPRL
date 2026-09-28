"""
SPRL: Sensitivity-Preserving Representation Learning with propensity homogeneity.

L_SPRL = L_dyn + λ_prop · L_prop

- L_dyn = L_reward (+ L_transition for multi-step)
- L_prop = kernel-weighted propensity variance within Z-neighborhoods

The penalty encourages nearby codes to have similar frozen fitted propensities.
It is a training objective, not an estimator of the population sensitivity bound.

World Model = same architecture with lambda_prop=0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.device import resolve_device


class SPRLEncoder(nn.Module):
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


class RewardPredictor(nn.Module):
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


class TransitionPredictor(nn.Module):
    """Predicts z_{t+1} from (z_t, a_t). JEPA-style latent transition model."""
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z, a):
        za = torch.cat([z, a.unsqueeze(-1) if a.dim() == 1 else a], dim=-1)
        return self.net(za)


class SPRL(nn.Module):
    """SPRL model with optional transition predictor."""

    def __init__(self, input_dim, latent_dim, hidden_dim=256,
                 use_transitions=False):
        super().__init__()
        self.encoder = SPRLEncoder(input_dim, latent_dim, hidden_dim)
        self.reward_predictor = RewardPredictor(latent_dim)
        self.transition_predictor = (
            TransitionPredictor(latent_dim) if use_transitions else None
        )
        self.latent_dim = latent_dim
        self.use_transitions = use_transitions

    def encode(self, x):
        return self.encoder(x)


def propensity_homogeneity_loss(z, e_hat, sigma=1.0):
    """
    L_prop = Σ_{i≠j} w_ij * (ê(s_i) - ê(s_j))² / Σ_{i≠j} w_ij

    where w_ij = exp(-||z_i - z_j||² / (2σ²))

    Diagonal (i=j) terms are excluded: they contribute zero to the numerator
    (propensity difference with yourself is always 0) but add maximal weight
    (w_ii=1) to the denominator, diluting the penalty signal.
    """
    B = z.shape[0]

    z_diff = z.unsqueeze(0) - z.unsqueeze(1)  # (B, B, d_z)
    z_dist_sq = (z_diff ** 2).sum(-1)           # (B, B)
    w = torch.exp(-z_dist_sq / (2 * sigma ** 2))  # (B, B)

    # Mask out diagonal (self-pairs carry no signal)
    off_diag = ~torch.eye(B, device=z.device, dtype=torch.bool)
    w = w * off_diag

    e_diff_sq = (e_hat.unsqueeze(0) - e_hat.unsqueeze(1)) ** 2  # (B, B)
    loss = (w * e_diff_sq).sum() / (w.sum() + 1e-8)
    return loss


def _median_heuristic(z):
    """Compute median pairwise distance for kernel bandwidth."""
    with torch.no_grad():
        dists = torch.cdist(z, z)
        mask = ~torch.eye(len(z), device=z.device, dtype=torch.bool)
        median_dist = dists[mask].median()
        return max(float(median_dist), 0.1)


def train_sprl(model, data_s, data_a, data_r, e_hat,
                    data_s_next=None,
                    lambda_prop=1.0, sigma=None, use_median_heuristic=True,
                    epochs=200, batch_size=256, lr=1e-3,
                    warmup_epochs=50, lambda_ramp_epochs=50,
                    grad_clip=1.0, device=None, verbose=True,
                    e_noise_sigma=0.0, normalize_penalty_geometry=False):
    """
    Train SPRL model.

    Parameters
    ----------
    model : SPRL
    data_s : array (n, d) — states
    data_a : array (n,) — actions
    data_r : array (n,) — rewards
    e_hat : array (n,) — frozen propensity estimates ê(s)
    data_s_next : array (n, d) or None — next states (for multi-step)
    lambda_prop : float — target weight of propensity homogeneity penalty
    sigma : float or None — kernel bandwidth (None = use median heuristic)
    use_median_heuristic : bool — recompute σ from each minibatch (detached)
    epochs : int
    batch_size : int
    lr : float
    warmup_epochs : int — train dynamics only before turning on L_prop
    lambda_ramp_epochs : int — ramp λ from 0 to target after warmup
    grad_clip : float — max gradient norm (0 to disable)
    device : str
    verbose : bool
    e_noise_sigma : float — if > 0, add fresh Gaussian noise N(0, σ²) to
        ê(s) at each minibatch (Gaussian perturbation ablation).
        Result is clipped to [0.01, 0.99].
        Default 0.0 = no noise (existing behavior unchanged).
    normalize_penalty_geometry : bool — when true, compute the propensity
        kernel on L2-normalized codes while reward and transition heads retain
        the unnormalized code. Defaults to false for the paper's method.

    Returns
    -------
    losses : list of dicts per epoch
    """
    device = resolve_device(device)
    model = model.to(device)

    params = list(model.encoder.parameters()) + list(model.reward_predictor.parameters())
    if model.transition_predictor is not None:
        params += list(model.transition_predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    s_t = torch.tensor(data_s, dtype=torch.float32, device=device)
    a_t = torch.tensor(data_a, dtype=torch.float32, device=device)
    r_t = torch.tensor(data_r, dtype=torch.float32, device=device)
    e_t = torch.tensor(e_hat, dtype=torch.float32, device=device)
    n = len(s_t)

    has_transitions = (data_s_next is not None and model.transition_predictor is not None)
    if has_transitions:
        s_next_t = torch.tensor(data_s_next, dtype=torch.float32, device=device)

    use_prop = (lambda_prop > 0.0)
    losses = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = {'total': 0, 'reward': 0, 'transition': 0, 'prop': 0}
        n_batches = 0

        # Compute effective λ_prop with warmup + ramp
        if not use_prop or epoch < warmup_epochs:
            lambda_eff = 0.0
        elif lambda_ramp_epochs > 0:
            adv_epoch = epoch - warmup_epochs
            ramp = min(1.0, (adv_epoch + 1) / lambda_ramp_epochs)
            lambda_eff = lambda_prop * ramp
        else:
            lambda_eff = lambda_prop

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            s_b, a_b, r_b, e_b = s_t[idx], a_t[idx], r_t[idx], e_t[idx]

            # Per-batch propensity-noise injection (variance test).
            # Fresh draw each minibatch — does NOT mutate e_t.
            if e_noise_sigma > 0.0:
                noise = torch.randn_like(e_b) * e_noise_sigma
                e_b = (e_b + noise).clamp(0.01, 0.99)

            z = model.encoder(s_b)

            # Reward loss
            r_pred = model.reward_predictor(z, a_b)
            loss_reward = nn.functional.mse_loss(r_pred, r_b)

            # Transition loss (JEPA-style)
            if has_transitions:
                s_next_b = s_next_t[idx]
                z_next_target = model.encoder(s_next_b).detach()
                z_next_pred = model.transition_predictor(z, a_b)
                loss_transition = nn.functional.mse_loss(z_next_pred, z_next_target)
            else:
                loss_transition = torch.tensor(0.0, device=device)

            loss_dyn = loss_reward + loss_transition

            # Propensity homogeneity penalty
            if lambda_eff > 0:
                z_for_propensity = (
                    F.normalize(z, p=2, dim=-1, eps=1e-8)
                    if normalize_penalty_geometry else z
                )
                # Compute sigma via median heuristic if needed
                if use_median_heuristic and sigma is None:
                    sig = _median_heuristic(z_for_propensity)
                else:
                    sig = sigma if sigma is not None else 1.0
                loss_prop = propensity_homogeneity_loss(z_for_propensity, e_b, sigma=sig)
                loss_total = loss_dyn + lambda_eff * loss_prop
            else:
                loss_prop = torch.tensor(0.0, device=device)
                loss_total = loss_dyn

            optimizer.zero_grad()
            loss_total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            epoch_loss['total'] += loss_total.item()
            epoch_loss['reward'] += loss_reward.item()
            epoch_loss['transition'] += loss_transition.item()
            epoch_loss['prop'] += loss_prop.item()
            n_batches += 1

        for k in epoch_loss:
            epoch_loss[k] /= max(n_batches, 1)
        losses.append(epoch_loss)

        if verbose and (epoch + 1) % 50 == 0:
            trans_str = f" trans={epoch_loss['transition']:.4f}" if has_transitions else ""
            print(f"  SPRL epoch {epoch+1}/{epochs}: "
                  f"reward={epoch_loss['reward']:.4f}{trans_str} "
                  f"prop={epoch_loss['prop']:.4f} (λ_eff={lambda_eff:.2f})",
                  flush=True)

    model.eval()
    return losses


def get_sprl_encoder_fn(model, device=None):
    """Return a numpy-in/numpy-out encoder function."""
    device = resolve_device(device)
    model.eval()

    @torch.no_grad()
    def encode(s):
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        z = model.encode(s_t)
        return z.cpu().numpy()

    return encode
