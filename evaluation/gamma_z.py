"""
Γ_Z computation: measure effective latent-space confounding strength.

Estimators:
1. compute_gamma_z() — k-means binning (legacy, has sparse-bin artifacts)
2. compute_gamma_z_logistic() — logistic regression (preferred, no binning artifacts)
3. compute_gamma_z_logistic_categorical() — logistic regression for
   categorical confounders with two or more levels

Γ_Z / Γ_S > 1 means the representation amplifies confounding.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def compute_gamma_z(z, a, u, n_bins=30, min_count=20):
    """
    Compute Γ_Z = max_{z_bin} odds(π_b(1|z_bin, u=1)) / odds(π_b(1|z_bin, u=0)).

    Parameters
    ----------
    z : array (n, d_z)
    a : array (n,) — actions (0 or 1)
    u : array (n,) — confounder (0 or 1)
    n_bins : int — number of k-means clusters
    min_count : int — minimum samples per (bin, u) group

    Returns
    -------
    gamma_z : float
    """
    z = np.atleast_2d(z)
    n = len(z)
    if n_bins > n // (2 * min_count):
        n_bins = max(3, n // (2 * min_count))

    labels = KMeans(n_clusters=n_bins, random_state=0, n_init=10).fit_predict(z)

    max_gamma = 1.0
    for b in np.unique(labels):
        mask_b = (labels == b)
        a_b = a[mask_b]
        u_b = u[mask_b]

        n_u0 = (u_b == 0).sum()
        n_u1 = (u_b == 1).sum()
        if n_u0 < min_count or n_u1 < min_count:
            continue

        e_u0 = np.clip(a_b[u_b == 0].mean(), 0.02, 0.98)
        e_u1 = np.clip(a_b[u_b == 1].mean(), 0.02, 0.98)

        odds_ratio = (e_u1 / (1 - e_u1)) / (e_u0 / (1 - e_u0))
        max_gamma = max(max_gamma, odds_ratio, 1.0 / odds_ratio)

    return max_gamma


def compute_gamma_z_logistic(z, a, u, percentile=95, use_interactions=False):
    """
    Compute Γ_Z via logistic regression — no binning artifacts.

    Default (additive): fits logit P(A=1|z,u) = β^T z + γ u.
    Γ_Z = exp(|γ|) — constant across samples.
    This is the correct estimator when the DGP has additive U effect.

    Optional (interactions): fits A ~ Z + U + Z*U for non-additive settings.
    Returns per-sample odds ratios.

    Parameters
    ----------
    z : array (n, d_z)
    a : array (n,) — actions (0 or 1)
    u : array (n,) — confounder (0 or 1)
    percentile : float — percentile of per-sample odds ratios (interaction mode)
    use_interactions : bool — include Z*U interaction terms (default False)

    Returns
    -------
    gamma_z : float — Γ_Z estimate
    info : dict — additional diagnostics
    """
    z = np.atleast_2d(z)
    n, d = z.shape
    u = u.ravel()
    a = a.ravel()

    # Standardize Z for numerical stability
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    u_col = u.reshape(-1, 1)

    if use_interactions:
        # Full interaction model: [Z, U, Z*U]
        z_times_u = z_scaled * u_col
        X = np.hstack([z_scaled, u_col, z_times_u])  # (n, 2d+1)
        C = min(1.0, 5.0 / d)
    else:
        # Additive model: [Z, U] — no interactions
        X = np.hstack([z_scaled, u_col])  # (n, d+1)
        C = 10.0  # weak regularization — enough features, no overfitting risk

    lr = LogisticRegression(max_iter=2000, C=C, solver='lbfgs')
    lr.fit(X, a)

    if use_interactions:
        # Per-sample odds ratios from interaction model
        X_u0 = np.hstack([z_scaled, np.zeros((n, 1)), np.zeros_like(z_scaled)])
        X_u1 = np.hstack([z_scaled, np.ones((n, 1)), z_scaled])
        p_u0 = np.clip(lr.predict_proba(X_u0)[:, 1], 0.005, 0.995)
        p_u1 = np.clip(lr.predict_proba(X_u1)[:, 1], 0.005, 0.995)
        odds_u0 = p_u0 / (1 - p_u0)
        odds_u1 = p_u1 / (1 - p_u1)
        odds_ratios = np.maximum(odds_u1 / odds_u0, odds_u0 / odds_u1)
    else:
        # Additive model: constant odds ratio = exp(|β_u|)
        beta_u = lr.coef_[0][-1]  # last coefficient = U's weight
        gamma_z_val = np.exp(abs(beta_u))
        odds_ratios = np.full(n, gamma_z_val)
        p_u0 = None
        p_u1 = None

    gamma_z_pct = float(np.percentile(odds_ratios, percentile))
    gamma_z_max = float(np.max(odds_ratios))
    gamma_z_median = float(np.median(odds_ratios))

    info = {
        'gamma_z_p95': float(np.percentile(odds_ratios, 95)),
        'gamma_z_max': gamma_z_max,
        'gamma_z_median': gamma_z_median,
        'odds_ratios_mean': float(np.mean(odds_ratios)),
        'odds_ratios_std': float(np.std(odds_ratios)),
        'lr_accuracy': float(lr.score(X, a)),
        'beta_u': float(lr.coef_[0][-1]),
    }
    if p_u0 is not None:
        info['p_u0_mean'] = float(p_u0.mean())
        info['p_u1_mean'] = float(p_u1.mean())

    return gamma_z_pct, info


def compute_gamma_z_logistic_categorical(
    z,
    a,
    u,
    percentile=95,
    C=None,
    max_iter=4000,
):
    """
    Estimate Γ_Z for a binary action and a categorical hidden confounder.

    Fits the symmetric, fully interacted logistic model

        logit P(A=1 | Z=z, U=u)
            = sum_k 1{u=k} (α_k + β_kᵀz),

    using one intercept and one Z-slope vector for every observed category.
    The full one-hot parameterization, with no separate global intercept,
    makes L2 regularization symmetric across categories.  Consequently, the
    estimate does not depend on which category happens to appear first.
    At each observed ``z_i``, the estimator predicts the action odds under
    every category and computes

        max_u odds(A=1 | z_i, u) / min_u odds(A=1 | z_i, u).

    The returned estimate is the requested percentile of these per-sample
    pairwise ranges.  In particular, it is not an odds ratio against one
    arbitrarily chosen reference category.

    Categories are listed in order of first appearance for diagnostics, but
    the symmetric fit is invariant to that order and to a bijective
    relabeling.  This function is separate from
    ``compute_gamma_z_logistic`` so the existing binary estimator and its
    reported results are unaffected.

    Parameters
    ----------
    z : array (n, d_z) or (n,)
        Learned representations.
    a : array (n,)
        Binary actions encoded as 0/1.
    u : array (n,)
        Categorical confounder labels. Labels need only be hashable.
    percentile : float
        Percentile of the per-sample maximum pairwise odds range.
    C : float or None
        Inverse L2 regularization strength. If None, uses the same
        dimension-aware default as the existing interaction estimator.
    max_iter : int
        Maximum logistic-regression iterations.

    Returns
    -------
    gamma_z : float
        Requested percentile of the per-sample pairwise odds ranges.
    info : dict
        Diagnostics, including the median, p95, maximum, category ordering,
        fitted probabilities, and convergence information.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.ndim != 2:
        raise ValueError("z must have shape (n, d_z) or (n,)")

    a = np.asarray(a).ravel()
    u = np.asarray(u).ravel()
    n, d = z.shape

    if n == 0:
        raise ValueError("z, a, and u must be non-empty")
    if len(a) != n or len(u) != n:
        raise ValueError("z, a, and u must contain the same number of samples")
    if not np.all(np.isfinite(z)):
        raise ValueError("z must contain only finite values")
    if not np.all(np.isfinite(a.astype(float))):
        raise ValueError("a must contain only finite values")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must lie in [0, 100]")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    action_levels = np.unique(a)
    if not np.all(np.isin(action_levels, [0, 1])):
        raise ValueError("a must be binary and encoded as 0/1")
    if len(action_levels) != 2:
        raise ValueError("both action classes must be present")
    a = a.astype(int)

    # Manual first-seen encoding makes the fitted design invariant to an
    # arbitrary bijective renaming of the category labels.
    category_to_index = {}
    categories = []
    u_index = np.empty(n, dtype=int)
    for i, label in enumerate(u.tolist()):
        is_missing = label is None
        if not is_missing:
            try:
                is_missing = bool(np.isscalar(label) and np.isnan(label))
            except TypeError:
                is_missing = False
        if is_missing:
            raise ValueError("u must not contain missing category labels")
        try:
            index = category_to_index.get(label)
        except TypeError as exc:
            raise ValueError("u category labels must be hashable") from exc
        if index is None:
            index = len(categories)
            category_to_index[label] = index
            categories.append(label)
        u_index[i] = index

    n_categories = len(categories)
    if n_categories < 2:
        raise ValueError("u must contain at least two categories")

    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    dummies = np.eye(n_categories, dtype=float)[u_index]
    interaction_blocks = [
        z_scaled * dummies[:, j : j + 1]
        for j in range(n_categories)
    ]
    X = np.hstack([dummies, *interaction_blocks])

    if C is None:
        C = min(1.0, 5.0 / max(d, 1))
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be a positive finite number")

    lr = LogisticRegression(
        max_iter=max_iter,
        C=float(C),
        solver="lbfgs",
        random_state=0,
        fit_intercept=False,
    )
    lr.fit(X, a)

    # Predict counterfactual action logits for every U category at every
    # observed z. Using log-odds spreads avoids probability clipping and
    # considers all K(K-1)/2 pairs simultaneously.
    logits = np.empty((n, n_categories), dtype=float)
    probability_means = []
    for category_index in range(n_categories):
        category_dummies = np.zeros_like(dummies)
        category_dummies[:, category_index] = 1.0
        category_interactions = [
            z_scaled * category_dummies[:, j : j + 1]
            for j in range(n_categories)
        ]
        X_category = np.hstack([category_dummies, *category_interactions])
        category_logits = lr.decision_function(X_category)
        logits[:, category_index] = category_logits
        category_probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(category_logits, -700.0, 700.0))
        )
        probability_means.append(float(np.mean(category_probabilities)))

    log_odds_ranges = np.max(logits, axis=1) - np.min(logits, axis=1)
    odds_ratios = np.exp(np.clip(log_odds_ranges, 0.0, 700.0))

    gamma_z_pct = float(np.percentile(odds_ratios, percentile))
    info = {
        "gamma_z_percentile": gamma_z_pct,
        "percentile": float(percentile),
        "gamma_z_p95": float(np.percentile(odds_ratios, 95)),
        "gamma_z_max": float(np.max(odds_ratios)),
        "gamma_z_median": float(np.median(odds_ratios)),
        "odds_ratios_mean": float(np.mean(odds_ratios)),
        "odds_ratios_std": float(np.std(odds_ratios)),
        "log_odds_range_max": float(np.max(log_odds_ranges)),
        "lr_accuracy": float(lr.score(X, a)),
        "n_categories": int(n_categories),
        "categories": categories,
        "category_counts": np.bincount(
            u_index, minlength=n_categories
        ).astype(int).tolist(),
        "category_order": categories,
        "category_probability_means": probability_means,
        "category_intercepts": lr.coef_[0][:n_categories].tolist(),
        "feature_dim": int(X.shape[1]),
        "C": float(C),
        "n_iter": int(np.max(lr.n_iter_)),
        "converged": bool(np.max(lr.n_iter_) < max_iter),
    }

    return gamma_z_pct, info
