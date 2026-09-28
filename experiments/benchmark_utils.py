"""Shared data injection and diagnostic readouts for the public benchmarks."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler


def make_default_policy_weights() -> np.ndarray:
    w_s = np.zeros(17, dtype=np.float32)
    w_s[16] = 0.3
    w_s[8] = -0.2
    w_s[1] = 0.15
    w_s[10] = 0.1
    return w_s


def inject_confounding_multistep(episodes, conf_dims, conf_strength,
                                w_u, w_s, bias, seed=None):
    """Inject one U per episode; retain physics rewards and transitions."""
    rng = np.random.RandomState(seed)
    all_s, all_a, all_r, all_s_next, all_u, all_pi_marg = [], [], [], [], [], []
    for ep in episodes:
        t = len(ep["r"])
        u_val = rng.binomial(1, 0.5)
        s = ep["s"].copy()
        s_next = ep["s_next"].copy()
        for dim in conf_dims:
            s[:, dim] += u_val * conf_strength
            s_next[:, dim] += u_val * conf_strength
        logits = s @ w_s + w_u * u_val + bias
        pi_b = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        a = rng.binomial(1, pi_b).astype(np.float32)
        logit_u0 = s @ w_s + bias
        logit_u1 = s @ w_s + w_u + bias
        pi_u0 = 1.0 / (1.0 + np.exp(-np.clip(logit_u0, -500, 500)))
        pi_u1 = 1.0 / (1.0 + np.exp(-np.clip(logit_u1, -500, 500)))
        all_s.append(s)
        all_a.append(a)
        all_r.append(ep["r"])
        all_s_next.append(s_next)
        all_u.append(np.full(t, u_val, dtype=np.float32))
        all_pi_marg.append(0.5 * pi_u0 + 0.5 * pi_u1)
    return {"s": np.concatenate(all_s), "a": np.concatenate(all_a),
            "r": np.concatenate(all_r), "s_next": np.concatenate(all_s_next),
            "u": np.concatenate(all_u), "pi_b_marg": np.concatenate(all_pi_marg)}


def split_halfcheetah(episodes: list[dict], seed: int, output: Path,
                     *, smoke: bool = False) -> dict:
    """Original injection order and 50/10/10 episode split; scaler uses train only."""
    n_episodes = len(episodes)
    lengths = {len(ep["r"]) for ep in episodes}
    if not smoke and (n_episodes != 1000 or lengths != {1000}):
        raise ValueError("Paper protocol requires 1000 episodes of 1000 transitions")
    if smoke and (n_episodes < 7 or len(lengths) != 1):
        raise ValueError("Smoke fixture needs at least seven equal-length episodes")
    episode_length = lengths.pop()
    all_data = inject_confounding_multistep(
        episodes, (0, 1, 8, 11, 12), 0.5, 0.5,
        make_default_policy_weights(), 0.0, seed=seed)
    episode_perm = np.random.RandomState(seed).permutation(n_episodes)
    if smoke:
        selected = {"train": episode_perm[:5], "fit": episode_perm[5:6],
                    "fresh": episode_perm[6:7]}
    else:
        selected = {"train": episode_perm[:50], "fit": episode_perm[50:60],
                    "fresh": episode_perm[60:70]}
    np.savez_compressed(output / "episode_split.npz", **selected)
    raw = {}
    for label, ids in selected.items():
        raw[label] = {
            key: values.reshape(n_episodes, episode_length, *values.shape[1:])[ids]
                       .reshape(len(ids) * episode_length, *values.shape[1:])
            for key, values in all_data.items() if key != "pi_b_marg"
        }
        raw[label]["episode_id"] = np.repeat(ids, episode_length)
    scaler = StandardScaler().fit(raw["train"]["s"])
    joblib.dump(scaler, output / "state_scaler.joblib")
    data = {}
    for label, values in raw.items():
        data[label] = {**values, "s": scaler.transform(values["s"]),
                       "s_next": scaler.transform(values["s_next"]),
                       "s_raw": values["s"], "s_next_raw": values["s_next"]}
        np.savez_compressed(output / f"{label}_observations.npz", **data[label])
    return {"train": data["train"], "fit": data["fit"], "fresh": data["fresh"],
            "target": None, "gamma_reference": float(np.exp(0.5)),
            "episode_split": {k: v.tolist() for k, v in selected.items()}}


def warning_records(items) -> list[dict]:
    return [{"category": x.category.__name__, "message": str(x.message)} for x in items]


def call_recorded(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, warning_records(caught)


def ece(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.minimum((p * 10).astype(int), 9)
    return float(sum(np.count_nonzero(bins == b) / len(p)
                     * abs(float(np.mean(y[bins == b])) - float(np.mean(p[bins == b])))
                     for b in range(10) if np.any(bins == b)))


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p_safe = np.clip(p, 1e-12, 1 - 1e-12)
    return {"log_loss": float(log_loss(y, p_safe, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "ece_10_fixed_bins": ece(y, p),
            "n_metric_epsilon_clipped": int(np.count_nonzero(p != p_safe))}


def interaction_design(z_scaled: np.ndarray, u: np.ndarray) -> np.ndarray:
    u_col = u.reshape(-1, 1)
    return np.hstack((z_scaled, u_col, z_scaled * u_col))


def readout_predictions(scaler: StandardScaler, model: LogisticRegression,
                        z: np.ndarray, u: np.ndarray) -> dict:
    z_scaled = scaler.transform(z)
    n = len(z)
    x_actual = interaction_design(z_scaled, u)
    p_actual = model.predict_proba(x_actual)[:, 1]
    zeros, ones = np.zeros((n, 1)), np.ones((n, 1))
    x0 = np.hstack((z_scaled, zeros, np.zeros_like(z_scaled)))
    x1 = np.hstack((z_scaled, ones, z_scaled))
    raw0, raw1 = model.predict_proba(x0)[:, 1], model.predict_proba(x1)[:, 1]
    p0, p1 = np.clip(raw0, 0.005, 0.995), np.clip(raw1, 0.005, 0.995)
    odds0, odds1 = p0 / (1 - p0), p1 / (1 - p1)
    ratios = np.maximum(odds1 / odds0, odds0 / odds1)
    return {"p_actual": p_actual, "p_u0_raw": raw0, "p_u1_raw": raw1,
            "p_u0_clipped": p0, "p_u1_clipped": p1, "odds_ratio": ratios,
            "n_counterfactual_probability_clipped": int(np.count_nonzero(raw0 != p0) + np.count_nonzero(raw1 != p1)),
            "gamma_z_p95": float(np.percentile(ratios, 95)),
            "gamma_z_median": float(np.median(ratios))}


def fit_interaction_readout(z: np.ndarray, a: np.ndarray, u: np.ndarray,
                            output: Path):
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    x = interaction_design(z_scaled, u)
    model = LogisticRegression(max_iter=2000, C=min(1.0, 5.0 / z.shape[1]), solver="lbfgs")
    _, seen = call_recorded(model.fit, x, a)
    joblib.dump({"scaler": scaler, "model": model}, output)
    return scaler, model, seen


def fit_additive_readout(z: np.ndarray, a: np.ndarray, u: np.ndarray,
                         output: Path):
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    x = np.column_stack((z_scaled, u))
    model = LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs")
    _, seen = call_recorded(model.fit, x, a)
    joblib.dump({"scaler": scaler, "model": model}, output)
    return scaler, model, seen


def additive_predictions(scaler, model, z: np.ndarray, u: np.ndarray) -> dict:
    p = model.predict_proba(np.column_stack((scaler.transform(z), u)))[:, 1]
    ratio = float(np.exp(abs(model.coef_[0, -1])))
    return {"p_actual": p, "odds_ratio": np.full(len(z), ratio),
            "gamma_z_p95": ratio, "gamma_z_median": ratio,
            "n_counterfactual_probability_clipped": 0}


def predict_model(model: torch.nn.Module, s: np.ndarray, a: np.ndarray,
                  s_next: np.ndarray | None) -> dict:
    model.eval()
    with torch.no_grad():
        st = torch.tensor(s, dtype=torch.float32)
        at = torch.tensor(a, dtype=torch.float32)
        z = model.encode(st)
        reward_pred = model.reward_predictor(z, at)
        result = {"z": z.numpy(), "reward_prediction": reward_pred.numpy()}
        if s_next is not None and getattr(model, "transition_predictor", None) is not None:
            sn = torch.tensor(s_next, dtype=torch.float32)
            result["latent_next_target"] = model.encode(sn).numpy()
            result["latent_next_prediction"] = model.transition_predictor(z, at).numpy()
    return result
