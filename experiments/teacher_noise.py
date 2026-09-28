"""Gaussian teacher-perturbation ablation for SPRL on Setting B (w_u=0.5, λ=1).

Two Gaussian-noise protocols:

  per_batch  (paper protocol):
      ê is corrupted with fresh Gaussian noise N(0, σ²) at each minibatch
      during SPRL training. Tests whether the encoder is robust to noisy
      propensity signals on average.

  fixed      (optional fixed random perturbation):
      ê is corrupted ONCE per seed, then SPRL trains against this fixed
      misspecified estimate for all epochs. This is a fixed random corruption, not a general systematic-bias model.

Pipeline per (σ, seed):
  1. Sample DGP and train clean propensity model ê(s) (frozen).
  2. (fixed mode only) Pre-corrupt: ê_fixed = clip(ê + N(0,σ²), 0.01, 0.99).
  3. Train SPRL(λ_prop=1.0).
       - per_batch: pass clean ê + e_noise_sigma=σ
       - fixed:     pass ê_fixed + e_noise_sigma=0
  4. Measure Γ_Z (interaction estimator), U-AUC, transition MSE.

Reports Med, p95, U-AUC, transition MSE — mean ± std over K seeds, per σ.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dgp.unified_cmdp import UnifiedCMDP, flatten_trajectories
from representations.propensity import PropensityModel, train_propensity, get_propensities
from representations.sprl_cell import SPRL, train_sprl, get_sprl_encoder_fn
from evaluation.gamma_z import compute_gamma_z_logistic
from utils.device import resolve_device

P = lambda *a, **kw: print(*a, **kw, flush=True)


def compute_transition_mse(model, s, a, s_next, device):
    """JEPA-style latent-target transition MSE on the full dataset."""
    model.eval()
    with torch.no_grad():
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        a_t = torch.tensor(a, dtype=torch.float32, device=device)
        s_next_t = torch.tensor(s_next, dtype=torch.float32, device=device)
        z = model.encode(s_t)
        z_next_target = model.encode(s_next_t)
        z_next_pred = model.transition_predictor(z, a_t)
        mse = torch.nn.functional.mse_loss(z_next_pred, z_next_target).item()
    return float(mse)


def run_one(dgp, seed, sigma, lambda_prop, n_traj, latent_dim, epochs,
            warmup_epochs, lambda_ramp_epochs, prop_epochs, grad_clip,
            noise_mode, device):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    traj = dgp.sample_trajectories(n_traj, rng=rng)
    flat = flatten_trajectories(traj)
    s, a, r = flat['s'], flat['a'], flat['r']
    s_next, u = flat['s_next'], flat['u']
    d = dgp.state_dim
    gamma_s = dgp.gamma_s

    # 1. Train clean propensity model and freeze
    torch.manual_seed(seed + 9000)
    prop_model = PropensityModel(d).to(device)
    train_propensity(prop_model, s, a, epochs=prop_epochs,
                     device=device, verbose=False)
    e_clean = get_propensities(prop_model, s, device=device)

    # 2. Decide what ê to feed into SPRL training, and whether per-batch
    #    noise is added inside the training loop
    if noise_mode == 'fixed' and sigma > 0:
        noise_rng = np.random.RandomState(seed + 7777)
        eps = noise_rng.normal(0.0, sigma, size=e_clean.shape)
        e_to_use = np.clip(e_clean + eps, 0.01, 0.99)
        train_e_noise_sigma = 0.0
        diff_l1   = float(np.mean(np.abs(e_to_use - e_clean)))
        diff_corr = float(np.corrcoef(e_to_use, e_clean)[0, 1]) if sigma > 0 else 1.0
    elif noise_mode == 'per_batch':
        # Pass clean ê; train_sprl injects fresh noise each minibatch.
        e_to_use = e_clean
        train_e_noise_sigma = float(sigma)
        # Diagnostic: expected one-batch perturbation magnitude
        if sigma > 0:
            diff_l1 = sigma * np.sqrt(2.0 / np.pi)  # E[|N(0,σ²)|]
            diff_corr = float('nan')                 # not meaningful for per-batch
        else:
            diff_l1 = 0.0
            diff_corr = 1.0
    else:  # sigma == 0 in either mode → identical
        e_to_use = e_clean
        train_e_noise_sigma = 0.0
        diff_l1 = 0.0
        diff_corr = 1.0

    # 3. Train SPRL
    torch.manual_seed(seed + 6000 + 1000)  # matches setting_b_unified for λ=1.0
    model = SPRL(d, latent_dim, use_transitions=True).to(device)
    train_sprl(
        model, s, a, r, e_hat=e_to_use, data_s_next=s_next,
        lambda_prop=lambda_prop, sigma=None,
        epochs=epochs,
        warmup_epochs=warmup_epochs, lambda_ramp_epochs=lambda_ramp_epochs,
        grad_clip=grad_clip, device=device, verbose=False,
        e_noise_sigma=train_e_noise_sigma,
    )
    phi = get_sprl_encoder_fn(model, device)
    z = phi(s)

    # 4. Metrics
    gz_p95, gi = compute_gamma_z_logistic(z, a, u, percentile=95,
                                          use_interactions=True)
    gz_med = float(gi['gamma_z_median'])
    amp_p95 = float(gz_p95 / gamma_s)
    amp_med = gz_med / gamma_s

    lr_probe = LogisticRegression(max_iter=1000, C=1.0).fit(z, u)
    u_auc = float(roc_auc_score(u, lr_probe.predict_proba(z)[:, 1]))

    trans_mse = compute_transition_mse(model, s, a, s_next, device)

    return {
        'amp_p95': amp_p95,
        'amp_med': amp_med,
        'gamma_z_p95': float(gz_p95),
        'gamma_z_med': gz_med,
        'u_auc': u_auc,
        'trans_mse': trans_mse,
        'e_hat_diff_l1': diff_l1,
        'e_hat_corr_with_clean': diff_corr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=20)
    ap.add_argument('--w_u', type=float, default=0.5)
    ap.add_argument('--lambda_prop', type=float, default=1.0)
    ap.add_argument('--sigmas', type=float, nargs='+',
                    default=[0.0, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument('--noise_mode', choices=['per_batch', 'fixed'],
                    default='per_batch',
                    help='per_batch (default) = fresh-noise test (fresh '
                         'noise each minibatch). fixed = fixed-noise test '
                         '(one corrupted ê used for the whole training run).')
    ap.add_argument('--n_traj', type=int, default=5000)
    ap.add_argument('--latent_dim', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--warmup_epochs', type=int, default=50)
    ap.add_argument('--lambda_ramp_epochs', type=int, default=50)
    ap.add_argument('--prop_epochs', type=int, default=100)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--save', type=str,
                    default='outputs/teacher_noise/'
                            'settingB_lam1_wu0.5_perbatch.json')
    args = ap.parse_args()

    device = resolve_device(args.device)

    P(f"Propensity-noise ablation on Setting B [{args.noise_mode} mode]")
    P(f"  w_u={args.w_u}, λ={args.lambda_prop}, K={args.K}")
    P(f"  σ levels: {args.sigmas}")
    P(f"  d_Z={args.latent_dim}, epochs={args.epochs}, "
      f"warmup={args.warmup_epochs}, ramp={args.lambda_ramp_epochs}")
    P(f"  device={device}\n")

    dgp = UnifiedCMDP(
        dim_noise=2, dim_instrument=2, dim_confounded=3, dim_clean=5,
        conf_strength=2.0, w_u=args.w_u, horizon=5, seed=0)
    gamma_s = dgp.gamma_s
    P(f"  Γ_S = {gamma_s:.4f}\n")

    all_results = {}

    for sigma in args.sigmas:
        P(f"==== σ = {sigma:.2f} ({args.noise_mode}) ====")
        per_seed = {'amp_p95': [], 'amp_med': [], 'u_auc': [],
                    'gamma_z_p95': [], 'gamma_z_med': [],
                    'trans_mse': [],
                    'e_hat_diff_l1': [], 'e_hat_corr_with_clean': []}

        for k in range(1, args.K + 1):
            t0 = time.time()
            res = run_one(
                dgp, seed=k, sigma=sigma,
                lambda_prop=args.lambda_prop,
                n_traj=args.n_traj, latent_dim=args.latent_dim,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                lambda_ramp_epochs=args.lambda_ramp_epochs,
                prop_epochs=args.prop_epochs,
                grad_clip=args.grad_clip,
                noise_mode=args.noise_mode, device=device,
            )
            for k2 in per_seed:
                per_seed[k2].append(res[k2])
            P(f"  seed {k:>3}/{args.K} ({time.time()-t0:.1f}s): "
              f"med={res['amp_med']:.2f} p95={res['amp_p95']:.2f} "
              f"u_auc={res['u_auc']:.3f} trans={res['trans_mse']:.3f}")

        med_arr = np.array(per_seed['amp_med'])
        p95_arr = np.array(per_seed['amp_p95'])
        ua_arr  = np.array(per_seed['u_auc'])
        tr_arr  = np.array(per_seed['trans_mse'])

        summary = {
            'sigma': sigma,
            'K': args.K,
            'noise_mode': args.noise_mode,
            'amp_med_mean': float(med_arr.mean()),
            'amp_med_std':  float(med_arr.std()),
            'amp_p95_mean': float(p95_arr.mean()),
            'amp_p95_std':  float(p95_arr.std()),
            'u_auc_mean':   float(ua_arr.mean()),
            'u_auc_std':    float(ua_arr.std()),
            'trans_mse_mean': float(tr_arr.mean()),
            'trans_mse_std':  float(tr_arr.std()),
            'per_seed': per_seed,
        }
        all_results[f'sigma_{sigma:g}'] = summary
        P(f"  → med={summary['amp_med_mean']:.3f}±{summary['amp_med_std']:.3f}  "
          f"p95={summary['amp_p95_mean']:.3f}±{summary['amp_p95_std']:.3f}  "
          f"u_auc={summary['u_auc_mean']:.3f}±{summary['u_auc_std']:.3f}  "
          f"trans={summary['trans_mse_mean']:.3f}±{summary['trans_mse_std']:.3f}\n")

    # Final table
    P("=" * 100)
    P(f"PROPENSITY-NOISE ABLATION  ({args.noise_mode}, Setting B, "
      f"w_u={args.w_u}, λ={args.lambda_prop}, K={args.K}, "
      f"Γ_S={gamma_s:.4f})")
    P("=" * 100)
    P(f"{'σ':>6} {'Med Γ_Z/Γ_S':>14} {'p95 Γ_Z/Γ_S':>14} "
      f"{'U-AUC':>14} {'Trans MSE':>14}")
    P("-" * 100)
    for sigma in args.sigmas:
        s = all_results[f'sigma_{sigma:g}']
        P(f"{sigma:>6.2f} "
          f"{s['amp_med_mean']:>7.2f}±{s['amp_med_std']:<5.2f}  "
          f"{s['amp_p95_mean']:>7.2f}±{s['amp_p95_std']:<5.2f}  "
          f"{s['u_auc_mean']:>5.3f}±{s['u_auc_std']:<5.3f}  "
          f"{s['trans_mse_mean']:>6.3f}±{s['trans_mse_std']:<5.3f}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump({
            'config': {
                'K': args.K, 'w_u': args.w_u, 'lambda_prop': args.lambda_prop,
                'sigmas': args.sigmas, 'noise_mode': args.noise_mode,
                'n_traj': args.n_traj,
                'latent_dim': args.latent_dim, 'epochs': args.epochs,
                'warmup_epochs': args.warmup_epochs,
                'lambda_ramp_epochs': args.lambda_ramp_epochs,
                'prop_epochs': args.prop_epochs,
                'grad_clip': args.grad_clip,
                'gamma_s': gamma_s,
            },
            'results': all_results,
        }, f, indent=2)
    P(f"\nSaved: {args.save}")


if __name__ == '__main__':
    main()
