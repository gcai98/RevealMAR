import math
from typing import Optional

import numpy as np


def _as_float_array(values):
    return np.asarray(values, dtype=np.float64)


def _rankdata(values):
    values = _as_float_array(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman_corr(x, y) -> Optional[float]:
    x = _as_float_array(x)
    y = _as_float_array(y)
    if x.size < 2 or y.size < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    if denom <= 0:
        return None
    return float((rx * ry).sum() / denom)


def kendall_tau(x, y) -> Optional[float]:
    x = _as_float_array(x)
    y = _as_float_array(y)
    n = x.size
    if n < 2:
        return None

    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
                continue
            if dy == 0:
                tie_y += 1
                continue
            if np.sign(dx) == np.sign(dy):
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt(float((concordant + discordant + tie_x) * (concordant + discordant + tie_y)))
    if denom <= 0:
        return None
    return float((concordant - discordant) / denom)


def topk_overlap(x, y, k) -> Optional[float]:
    x = _as_float_array(x)
    y = _as_float_array(y)
    k = min(int(k), x.size, y.size)
    if k <= 0:
        return None
    pred = set(np.argsort(-x)[:k].tolist())
    oracle = set(np.argsort(-y)[:k].tolist())
    return float(len(pred & oracle) / float(k))


def ndcg_at_k(scores, gains, k) -> Optional[float]:
    scores = _as_float_array(scores)
    gains = _as_float_array(gains)
    k = min(int(k), scores.size, gains.size)
    if k <= 0:
        return None

    shifted_gains = gains - np.min(gains)
    pred_order = np.argsort(-scores)[:k]
    ideal_order = np.argsort(-shifted_gains)[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    actual = float(np.sum(shifted_gains[pred_order] * discounts))
    ideal = float(np.sum(shifted_gains[ideal_order] * discounts))
    if ideal <= 0:
        return None
    return actual / ideal


def regret(selected_by_signal, oracle_gains, k) -> Optional[float]:
    selected_by_signal = _as_float_array(selected_by_signal)
    oracle_gains = _as_float_array(oracle_gains)
    k = min(int(k), selected_by_signal.size, oracle_gains.size)
    if k <= 0:
        return None
    selected = np.argsort(-selected_by_signal)[:k]
    oracle = np.argsort(-oracle_gains)[:k]
    return float(np.mean(oracle_gains[oracle]) - np.mean(oracle_gains[selected]))


def nanmean_or_none(values) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))
