# objective_minimax.py
from __future__ import annotations
import numpy as np

def smooth_min(values: np.ndarray, t: float) -> float:
    """
    smin(x) = -t * log(sum_i exp(-x_i / t))
    Stable implementation via log-sum-exp.
    """
    x = np.asarray(values, dtype=float)
    if t <= 0:
        raise ValueError("t must be > 0")
    y = -x / t
    y_max = np.max(y)                 # for stability
    lse = y_max + np.log(np.sum(np.exp(y - y_max)))
    return float(-t * lse)

def smooth_min_weights(values: np.ndarray, t: float) -> np.ndarray:
    """
    alpha_i = d smin / d x_i = softmax(-x/t)_i
    """
    x = np.asarray(values, dtype=float)
    if t <= 0:
        raise ValueError("t must be > 0")
    y = -x / t
    y_max = np.max(y)
    w = np.exp(y - y_max)
    alpha = w / np.sum(w)
    return alpha

def J_minimax_Cii(C11: float, C22: float, C33: float, t: float):
    vals = np.array([C11, C22, C33], dtype=float)
    smin = smooth_min(vals, t)
    alphas = smooth_min_weights(vals, t)

    # shift so that if all vals equal, smin_avg == that value
    smin_avg = smin + t * np.log(len(vals))

    J = -smin_avg
    return float(J), alphas

