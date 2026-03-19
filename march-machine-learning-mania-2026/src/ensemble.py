"""
Ensemble: combine model probabilities into a single P(Team A wins).

Role: Provide a small, reusable blending helper for submission inference.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np


def _normalize_weights(weights: Optional[Sequence[float]], n: int) -> np.ndarray:
    """
    Normalize weights to sum to 1.

    Notes
    -----
    - If `weights` is None, returns equal weights.
    - Requires `n > 0` and (after conversion) `sum(weights) > 0`.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    if len(weights) != n:
        raise ValueError(f"Expected {n} weights, got {len(weights)}")
    w = np.asarray(list(weights), dtype=np.float64)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Sum of ensemble weights must be > 0")
    return w / total


def ensemble_predict(
    models_or_preds: Sequence[Union[Any, np.ndarray]],
    X=None,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Blend probabilities from multiple sources.

    Parameters
    ----------
    models_or_preds:
        Either a list of fitted estimators exposing `predict_proba(X)`, or a list
        of probability arrays. If estimators are provided, `X` must be set.
    X:
        Feature matrix passed to each estimator's `predict_proba`.
    weights:
        Optional weights per probability vector. If None, uses equal weights.

    Returns
    -------
    np.ndarray
        Blended P(Team A wins) per row.
    """
    if not models_or_preds:
        raise ValueError("models_or_preds must be non-empty")

    first = models_or_preds[0]
    n_sources = len(models_or_preds)
    w = _normalize_weights(weights, n_sources)

    # Model path: estimators exposing predict_proba.
    if hasattr(first, "predict_proba"):
        if X is None:
            raise ValueError("X must be provided when blending model estimators")
        probs: List[np.ndarray] = []
        for m in models_or_preds:
            if not hasattr(m, "predict_proba"):
                raise TypeError("All elements must expose predict_proba when first does")
            p = m.predict_proba(X)[:, 1].astype(np.float64, copy=False)
            probs.append(p)
        stack = np.vstack(probs)  # (n_models, n_rows)
        return np.tensordot(w, stack, axes=(0, 0))  # (n_rows,)

    # Array path: raw probability vectors.
    probs_arr: List[np.ndarray] = []
    for p in models_or_preds:
        arr = np.asarray(p, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("Probability arrays must be 1D")
        probs_arr.append(arr)

    lengths = {int(a.shape[0]) for a in probs_arr}
    if len(lengths) != 1:
        raise ValueError(f"All probability vectors must have the same length; got {sorted(lengths)}")

    out = np.zeros_like(probs_arr[0], dtype=np.float64)
    for i, arr in enumerate(probs_arr):
        out += w[i] * arr
    return out


def blend_rating_with_models(
    rating_probs: np.ndarray,
    model_probs_list: List[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Blend rating-derived probabilities with ML model probabilities.

    When `weights` is provided, its length must be `1 + len(model_probs_list)`.
    If `weights` is None, uses equal weights across all sources.
    """
    all_probs: List[np.ndarray] = [np.asarray(rating_probs, dtype=np.float64)] + [
        np.asarray(p, dtype=np.float64) for p in model_probs_list
    ]
    return ensemble_predict(all_probs, weights=weights)
