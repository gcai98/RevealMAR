# util/revealmar_candidates.py
# Copyright (c) 2026
#
# Candidate subset construction for RevealMAR / PlanMAR-S.
#
# This module implements candidate filtering for masked-token reveal planning:
#   - topk:         score-biased candidates
#   - random:       deterministic pseudo-random candidates
#   - uncertainty:  high-score candidates, assuming masked_scores is a proposal score
#   - spatial:      farthest-point spatially diverse candidates
#   - mixed:        random + uncertainty-biased + spatially diverse proposals
#
# Input:
#   masked_scores: [B, M] tensor, where M is the number of currently masked tokens.
#   masked_coords: optional [B, M, 2] tensor containing grid coordinates of masked tokens.
#
# Output:
#   candidate_indices: [B, K] long tensor indexing masked-token positions, not full sequence positions.

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch


_VALID_SELECTION_MODES = {"topk", "random", "uncertainty", "spatial", "mixed"}


def _safe_scores(scores: torch.Tensor) -> torch.Tensor:
    """Convert scores to finite float32 values for stable sorting/proposal construction."""
    return torch.nan_to_num(
        scores.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _deterministic_pseudo_random_order(
    num_items: int,
    device: torch.device,
    seed_scalar: torch.Tensor | float,
) -> torch.Tensor:
    """
    Return a deterministic pseudo-random permutation on the same device.

    This avoids CPU/GPU Generator differences and works under DDP without needing
    per-rank torch.Generator plumbing. The order is deterministic given num_items
    and seed_scalar.
    """
    if num_items <= 0:
        return torch.empty(0, device=device, dtype=torch.long)

    base = torch.arange(num_items, device=device, dtype=torch.float32)

    if not torch.is_tensor(seed_scalar):
        seed_scalar = torch.tensor(float(seed_scalar), device=device, dtype=torch.float32)
    else:
        seed_scalar = seed_scalar.to(device=device, dtype=torch.float32)

    # Hash-like sinusoidal noise. This is not cryptographic randomness; it is only
    # used to get a stable pseudo-random ordering for candidate proposals.
    noise = torch.frac(
        torch.sin((base + 1.0) * 12.9898 + seed_scalar * 78.233) * 43758.5453
    )
    return torch.argsort(noise, descending=True)


def _append_unique(
    chosen: List[int],
    selected_mask: torch.Tensor,
    order: Iterable[int],
    max_count: int,
) -> None:
    """Append unique indices from order until len(chosen) reaches max_count."""
    if max_count <= 0:
        return

    for idx in order:
        if len(chosen) >= max_count:
            break

        idx = int(idx)
        if idx < 0 or idx >= selected_mask.numel():
            continue

        if not bool(selected_mask[idx].item()):
            selected_mask[idx] = True
            chosen.append(idx)


def _counts_from_ratios(
    k: int,
    random_ratio: float,
    uncertainty_ratio: float,
    spatial_ratio: float,
) -> Tuple[int, int, int]:
    """
    Convert proposal ratios to integer counts that sum to k.

    Uses floor allocation plus largest-remainder correction, which is less brittle
    than repeated round() when k is small.
    """
    ratios = torch.tensor(
        [float(random_ratio), float(uncertainty_ratio), float(spatial_ratio)],
        dtype=torch.float64,
    )

    if torch.any(ratios < 0):
        raise ValueError(
            "candidate ratios must be non-negative, got "
            f"random={random_ratio}, uncertainty={uncertainty_ratio}, spatial={spatial_ratio}"
        )

    ratio_sum = float(ratios.sum().item())
    if ratio_sum <= 0.0:
        raise ValueError("candidate ratios must sum to > 0 when selection_mode='mixed'")

    raw = ratios / ratio_sum * int(k)
    counts = torch.floor(raw).to(torch.long)
    remainder = int(k) - int(counts.sum().item())

    if remainder > 0:
        frac = raw - torch.floor(raw)
        order = torch.argsort(frac, descending=True)
        for i in range(remainder):
            counts[order[i % 3]] += 1

    return int(counts[0].item()), int(counts[1].item()), int(counts[2].item())


def _topk_order(scores: torch.Tensor) -> torch.Tensor:
    """Return indices sorted by descending score."""
    return torch.argsort(_safe_scores(scores), descending=True)


def _greedy_spatial_diverse_order(
    coords: torch.Tensor,
    preselected_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Return a farthest-point-style spatially diverse ordering.

    coords:
        [M, 2], usually row/col coordinates on latent grid.

    preselected_mask:
        Optional [M] bool tensor. Already selected tokens are treated as anchors,
        so spatial candidates will be far away from them when possible.
    """
    if coords is None:
        raise ValueError("coords must not be None for spatial candidate construction")

    if coords.dim() != 2 or coords.size(-1) != 2:
        raise ValueError(f"coords must have shape [M, 2], got {tuple(coords.shape)}")

    device = coords.device
    num_items = int(coords.size(0))

    if num_items <= 0:
        return torch.empty(0, device=device, dtype=torch.long)

    coords_f = torch.nan_to_num(coords.float(), nan=0.0, posinf=0.0, neginf=0.0)

    if preselected_mask is None:
        selected = torch.zeros(num_items, device=device, dtype=torch.bool)
    else:
        if preselected_mask.shape != (num_items,):
            raise ValueError(
                "preselected_mask must have shape [M], got "
                f"{tuple(preselected_mask.shape)} for M={num_items}"
            )
        selected = preselected_mask.clone().to(device=device, dtype=torch.bool)

    order: List[int] = []

    # If there are existing anchors, choose points farthest from anchors first.
    # Otherwise start from the point farthest from the masked-token centroid.
    while len(order) < num_items:
        remaining = torch.nonzero(~selected, as_tuple=False).flatten()
        if remaining.numel() == 0:
            break

        if selected.any():
            anchor_coords = coords_f[selected]
            dist = torch.cdist(coords_f[remaining], anchor_coords).min(dim=1).values
        else:
            center = coords_f.mean(dim=0, keepdim=True)
            dist = torch.norm(coords_f[remaining] - center, dim=-1)

        next_idx = int(remaining[torch.argmax(dist)].item())
        selected[next_idx] = True
        order.append(next_idx)

    return torch.tensor(order, device=device, dtype=torch.long)


def _validate_inputs(
    masked_scores: torch.Tensor,
    candidate_pool_size: int,
    masked_coords: Optional[torch.Tensor],
    selection_mode: str,
) -> Tuple[int, int, int]:
    if not torch.is_tensor(masked_scores):
        raise TypeError("masked_scores must be a torch.Tensor")

    if masked_scores.dim() != 2:
        raise ValueError(f"masked_scores must have shape [B, M], got {tuple(masked_scores.shape)}")

    if selection_mode not in _VALID_SELECTION_MODES:
        raise ValueError(
            f"Unsupported selection_mode={selection_mode}. "
            f"Expected one of {sorted(_VALID_SELECTION_MODES)}"
        )

    bsz, masked_token_count = masked_scores.shape
    k = min(max(int(candidate_pool_size), 0), int(masked_token_count))

    if masked_coords is not None:
        if not torch.is_tensor(masked_coords):
            raise TypeError("masked_coords must be None or a torch.Tensor")
        if masked_coords.shape[:2] != masked_scores.shape or masked_coords.size(-1) != 2:
            raise ValueError(
                "masked_coords must have shape [B, M, 2], got "
                f"{tuple(masked_coords.shape)} for masked_scores={tuple(masked_scores.shape)}"
            )

    return int(bsz), int(masked_token_count), int(k)


def _fallback_fill_to_k(
    chosen: List[int],
    selected_mask: torch.Tensor,
    scores: torch.Tensor,
    k: int,
    seed_scalar: torch.Tensor,
) -> None:
    """
    Fill remaining candidate slots. Prefer high-score tokens, then pseudo-random order.
    """
    if len(chosen) >= k:
        return

    _append_unique(chosen, selected_mask, _topk_order(scores).tolist(), k)

    if len(chosen) >= k:
        return

    random_order = _deterministic_pseudo_random_order(
        selected_mask.numel(),
        device=selected_mask.device,
        seed_scalar=seed_scalar + 1.0,
    )
    _append_unique(chosen, selected_mask, random_order.tolist(), k)


def build_candidate_subset(
    masked_scores: torch.Tensor,
    candidate_pool_size: int,
    masked_coords: Optional[torch.Tensor] = None,
    selection_mode: str = "topk",
    random_ratio: float = 0.25,
    uncertainty_ratio: float = 0.50,
    spatial_ratio: float = 0.25,
    subset_seed: int = 123,
) -> torch.Tensor:
    """
    Build a candidate subset for planner target construction or inference pre-filtering.

    Parameters
    ----------
    masked_scores:
        Tensor of shape [B, M]. Higher score means higher proposal priority.
        For planner sampling, this is usually planner score. For entropy/uncertainty
        ablations, this can be an uncertainty-derived score.

    candidate_pool_size:
        Number of candidates K to return per sample. The returned K is clipped to M.

    masked_coords:
        Optional tensor of shape [B, M, 2], containing grid coordinates for masked
        tokens. Required for meaningful spatial or mixed-spatial proposals.

    selection_mode:
        One of:
            - "topk":        top-K by masked_scores
            - "random":      deterministic pseudo-random K
            - "uncertainty": top-K by masked_scores
            - "spatial":     farthest-point spatially diverse K
            - "mixed":       random + uncertainty-biased + spatially diverse

    random_ratio / uncertainty_ratio / spatial_ratio:
        Ratios used only for selection_mode="mixed".

    subset_seed:
        Seed-like scalar for deterministic pseudo-random ordering.

    Returns
    -------
    candidate_indices:
        Long tensor of shape [B, K]. Values are indices in the masked-token list
        dimension, not full sequence indices.
    """
    bsz, masked_token_count, k = _validate_inputs(
        masked_scores=masked_scores,
        candidate_pool_size=candidate_pool_size,
        masked_coords=masked_coords,
        selection_mode=selection_mode,
    )

    device = masked_scores.device

    if k == 0:
        return torch.empty(bsz, 0, device=device, dtype=torch.long)

    if k == masked_token_count and selection_mode in {"topk", "uncertainty"}:
        return torch.argsort(_safe_scores(masked_scores), dim=-1, descending=True)

    candidate_indices = torch.empty(bsz, k, device=device, dtype=torch.long)

    for b in range(bsz):
        scores_b = _safe_scores(masked_scores[b])
        selected_mask = torch.zeros(masked_token_count, device=device, dtype=torch.bool)
        chosen: List[int] = []

        # Make random order depend on both user seed and sample index. Add a small
        # score-dependent scalar so that repeated calls at different states do not
        # get exactly the same random order.
        seed_scalar = scores_b.mean() + float(int(subset_seed) + b * 1009)

        if selection_mode == "topk":
            _append_unique(chosen, selected_mask, _topk_order(scores_b).tolist(), k)

        elif selection_mode == "uncertainty":
            # Here "uncertainty" means: use masked_scores as the uncertainty/proposal
            # score. The caller decides what masked_scores represents.
            _append_unique(chosen, selected_mask, _topk_order(scores_b).tolist(), k)

        elif selection_mode == "random":
            random_order = _deterministic_pseudo_random_order(
                masked_token_count,
                device=device,
                seed_scalar=seed_scalar,
            )
            _append_unique(chosen, selected_mask, random_order.tolist(), k)

        elif selection_mode == "spatial":
            if masked_coords is not None:
                spatial_order = _greedy_spatial_diverse_order(masked_coords[b])
                _append_unique(chosen, selected_mask, spatial_order.tolist(), k)
            else:
                # Safe fallback: without coordinates, spatial diversity is undefined.
                _append_unique(chosen, selected_mask, _topk_order(scores_b).tolist(), k)

        elif selection_mode == "mixed":
            random_count, uncertainty_count, spatial_count = _counts_from_ratios(
                k=k,
                random_ratio=random_ratio,
                uncertainty_ratio=uncertainty_ratio,
                spatial_ratio=spatial_ratio,
            )

            # 1) Uncertainty/score-biased candidates.
            if uncertainty_count > 0:
                uncertainty_order = _topk_order(scores_b)
                _append_unique(
                    chosen,
                    selected_mask,
                    uncertainty_order.tolist(),
                    min(k, len(chosen) + uncertainty_count),
                )

            # 2) Random candidates.
            if random_count > 0:
                random_order = _deterministic_pseudo_random_order(
                    masked_token_count,
                    device=device,
                    seed_scalar=seed_scalar,
                )
                _append_unique(
                    chosen,
                    selected_mask,
                    random_order.tolist(),
                    min(k, len(chosen) + random_count),
                )

            # 3) Spatially diverse candidates, anchored by already selected tokens.
            if spatial_count > 0:
                if masked_coords is not None:
                    spatial_order = _greedy_spatial_diverse_order(
                        masked_coords[b],
                        preselected_mask=selected_mask,
                    )
                    _append_unique(
                        chosen,
                        selected_mask,
                        spatial_order.tolist(),
                        min(k, len(chosen) + spatial_count),
                    )
                else:
                    # Without coords, reallocate spatial slots to score-biased fallback.
                    _append_unique(
                        chosen,
                        selected_mask,
                        _topk_order(scores_b).tolist(),
                        min(k, len(chosen) + spatial_count),
                    )

        else:
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")

        _fallback_fill_to_k(
            chosen=chosen,
            selected_mask=selected_mask,
            scores=scores_b,
            k=k,
            seed_scalar=seed_scalar,
        )

        if len(chosen) != k:
            raise RuntimeError(
                f"Internal error: selected {len(chosen)} candidates, expected {k}"
            )

        candidate_indices[b] = torch.tensor(chosen, device=device, dtype=torch.long)

    return candidate_indices


__all__ = ["build_candidate_subset"]