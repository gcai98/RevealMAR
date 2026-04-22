import torch


def _deterministic_pseudo_random_order(num_items, device, dtype, seed_scalar):
    base = torch.arange(num_items, device=device, dtype=dtype)
    noise = torch.frac(torch.sin((base + 1.0) * 12.9898 + seed_scalar * 78.233) * 43758.5453)
    return torch.argsort(noise, descending=True)


def _greedy_spatial_diverse_indices(coords, candidate_count, already_selected_mask):
    remaining_idx = (~already_selected_mask).nonzero(as_tuple=True)[0]
    if candidate_count <= 0 or remaining_idx.numel() == 0:
        return []

    diverse = []
    if already_selected_mask.any():
        selected_coords = coords[already_selected_mask]
        min_dist = torch.cdist(coords[remaining_idx].float(), selected_coords.float()).min(dim=1).values
        first_idx = remaining_idx[torch.argmax(min_dist)]
    else:
        coords_float = coords.float()
        center = coords_float.mean(dim=0, keepdim=True)
        dist_to_center = torch.norm(coords_float[remaining_idx] - center, dim=-1)
        first_idx = remaining_idx[torch.argmax(dist_to_center)]

    diverse.append(int(first_idx.item()))
    while len(diverse) < candidate_count:
        already_selected_mask[diverse[-1]] = True
        remaining_idx = (~already_selected_mask).nonzero(as_tuple=True)[0]
        if remaining_idx.numel() == 0:
            break
        selected_coords = coords[already_selected_mask]
        min_dist = torch.cdist(coords[remaining_idx].float(), selected_coords.float()).min(dim=1).values
        next_idx = remaining_idx[torch.argmax(min_dist)]
        diverse.append(int(next_idx.item()))
    return diverse


def build_candidate_subset(masked_scores, candidate_pool_size, masked_coords=None, selection_mode='topk'):
    """Select candidate masked tokens per sample by planner score or a mixed proposal."""
    assert masked_scores.dim() == 2, "masked_scores must be [bsz, masked_token_count]"
    bsz, masked_token_count = masked_scores.shape
    if masked_token_count == 0:
        empty = torch.empty(bsz, 0, device=masked_scores.device, dtype=torch.long)
        return empty

    k = min(int(candidate_pool_size), masked_token_count)
    if selection_mode == 'topk' or k <= 1:
        return torch.topk(masked_scores, k=k, dim=-1, largest=True).indices
    if selection_mode != 'mixed':
        raise ValueError("Unsupported selection_mode: {}".format(selection_mode))

    candidate_indices = torch.empty(bsz, k, device=masked_scores.device, dtype=torch.long)
    topk_count = min(k, max(1, k // 2))
    random_count = min(k - topk_count, max(0, k // 4))
    diverse_count = k - topk_count - random_count

    for b in range(bsz):
        scores = masked_scores[b]
        selected_mask = torch.zeros(masked_token_count, device=masked_scores.device, dtype=torch.bool)
        chosen = []

        top_idx = torch.topk(scores, k=topk_count, dim=-1, largest=True).indices.tolist()
        for idx in top_idx:
            if not selected_mask[idx]:
                selected_mask[idx] = True
                chosen.append(idx)

        if random_count > 0:
            seed_scalar = torch.nan_to_num(scores.float().mean(), nan=0.0, posinf=0.0, neginf=0.0)
            random_order = _deterministic_pseudo_random_order(
                masked_token_count, device=masked_scores.device, dtype=scores.dtype, seed_scalar=seed_scalar
            )
            for idx in random_order.tolist():
                if len(chosen) >= topk_count + random_count:
                    break
                if not selected_mask[idx]:
                    selected_mask[idx] = True
                    chosen.append(idx)

        if diverse_count > 0 and masked_coords is not None:
            diverse = _greedy_spatial_diverse_indices(masked_coords[b], diverse_count, selected_mask.clone())
            for idx in diverse:
                if len(chosen) >= k:
                    break
                if not selected_mask[idx]:
                    selected_mask[idx] = True
                    chosen.append(idx)

        if len(chosen) < k:
            fallback = torch.argsort(scores, descending=True)
            for idx in fallback.tolist():
                if len(chosen) >= k:
                    break
                if not selected_mask[idx]:
                    selected_mask[idx] = True
                    chosen.append(idx)

        candidate_indices[b] = torch.tensor(chosen, device=masked_scores.device, dtype=torch.long)

    return candidate_indices


def _scatter_candidate_values(base_tensor, candidate_indices, candidate_values):
    """Scatter candidate-only values back to masked-token layout."""
    candidate_values = candidate_values.to(dtype=base_tensor.dtype, device=base_tensor.device)
    out = torch.zeros_like(base_tensor)
    out.scatter_(dim=1, index=candidate_indices, src=candidate_values)
    return out


def _normalize_per_sample(values, eps=1e-6):
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    v_min = values.min(dim=1, keepdim=True).values
    v_max = values.max(dim=1, keepdim=True).values
    normalized = (values - v_min) / (v_max - v_min + eps)
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def _local_discrepancy_l2(tokens):
    """Average L2-to-local-mean discrepancy in feature space."""
    if tokens.size(0) <= 1:
        return tokens.new_zeros(())
    tokens_fp32 = torch.nan_to_num(tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
    local_mean = tokens_fp32.mean(dim=0, keepdim=True)
    discrepancy = ((tokens_fp32 - local_mean) ** 2).sum(dim=-1).mean()
    discrepancy = torch.nan_to_num(discrepancy, nan=0.0, posinf=0.0, neginf=0.0)
    return discrepancy.to(dtype=tokens.dtype)


def build_gt_reveal_target(masked_scores, candidate_indices, masked_decoder_tokens, masked_gt_decoder_tokens, masked_coords):
    """
    Local counterfactual pseudo-utility v1:
    utility = local_discrepancy_before - local_discrepancy_after_reveal_once.
    """
    if candidate_indices.numel() == 0:
        return torch.zeros_like(masked_scores)

    bsz, masked_token_count, hidden_dim = masked_decoder_tokens.shape
    assert masked_scores.shape == (bsz, masked_token_count), "masked_scores shape mismatch"
    assert masked_gt_decoder_tokens.shape == (bsz, masked_token_count, hidden_dim), "masked_gt_decoder_tokens shape mismatch"
    assert masked_coords.shape == (bsz, masked_token_count, 2), "masked_coords shape mismatch"

    neighborhood_radius = 1.0
    utilities = masked_scores.new_zeros(candidate_indices.shape)
    masked_decoder_tokens_detached = masked_decoder_tokens.detach()
    masked_gt_decoder_tokens_detached = masked_gt_decoder_tokens.detach()

    for b in range(bsz):
        cur_tokens = masked_decoder_tokens_detached[b]
        cur_gt_tokens = masked_gt_decoder_tokens_detached[b]
        cur_coords = masked_coords[b]
        cur_candidate_indices = candidate_indices[b]

        for j in range(cur_candidate_indices.size(0)):
            cand_idx = int(cur_candidate_indices[j].item())
            cand_coord = cur_coords[cand_idx]

            # Local masked neighborhood in latent grid (Chebyshev radius=1, includes candidate).
            dist = torch.max(torch.abs(cur_coords - cand_coord), dim=-1).values
            neighborhood_mask = dist <= neighborhood_radius
            neighborhood_idx = neighborhood_mask.nonzero(as_tuple=True)[0]
            assert neighborhood_idx.numel() > 0, "Candidate neighborhood should not be empty"

            local_before_tokens = cur_tokens[neighborhood_idx]
            discrepancy_before = _local_discrepancy_l2(local_before_tokens)

            # Counterfactual reveal-once: replace candidate with GT feature in same decoder space.
            local_after_tokens = local_before_tokens.clone()
            local_cand_pos = (neighborhood_idx == cand_idx).nonzero(as_tuple=True)[0]
            assert local_cand_pos.numel() == 1, "Candidate index should appear exactly once in neighborhood"
            local_after_tokens[local_cand_pos[0]] = cur_gt_tokens[cand_idx]
            discrepancy_after = _local_discrepancy_l2(local_after_tokens)

            utility = discrepancy_before - discrepancy_after
            utilities[b, j] = torch.nan_to_num(utility, nan=0.0, posinf=0.0, neginf=0.0)

    utilities = _normalize_per_sample(utilities)
    return _scatter_candidate_values(masked_scores, candidate_indices, utilities)


def build_pred_reveal_target(masked_scores, candidate_indices):
    """
    Build reference-continuation pseudo target from planner logits on candidates.
    This is detached and candidate-subset-only to avoid coupling with planner loss.
    """
    if candidate_indices.numel() == 0:
        return torch.zeros_like(masked_scores)

    candidate_scores = torch.gather(masked_scores.detach(), dim=1, index=candidate_indices)
    continuation_probs = torch.softmax(candidate_scores, dim=1)
    continuation_probs = _normalize_per_sample(continuation_probs)
    return _scatter_candidate_values(masked_scores, candidate_indices, continuation_probs)


def build_mixed_reveal_target(gt_target, pred_target):
    """
    Build mixed reveal target as an equal blend.
    Mixed-policy training is intentionally not enabled in this phase.
    """
    return 0.5 * gt_target + 0.5 * pred_target


def build_pseudo_target(
    masked_scores,
    candidate_indices,
    masked_decoder_tokens,
    masked_gt_decoder_tokens,
    masked_coords,
    pseudo_target_type,
):
    """
    Build pseudo-utility targets with candidate-subset-only construction.
    This keeps rollout/counterfactual refinement pluggable for later phases.
    """
    if pseudo_target_type == 'none':
        return None
    if pseudo_target_type == 'gt_reveal':
        return build_gt_reveal_target(
            masked_scores, candidate_indices, masked_decoder_tokens, masked_gt_decoder_tokens, masked_coords
        )
    if pseudo_target_type == 'pred_reveal':
        return build_pred_reveal_target(masked_scores, candidate_indices)
    if pseudo_target_type == 'mixed_reveal':
        gt_target = build_gt_reveal_target(
            masked_scores, candidate_indices, masked_decoder_tokens, masked_gt_decoder_tokens, masked_coords
        )
        pred_target = build_pred_reveal_target(masked_scores, candidate_indices)
        return build_mixed_reveal_target(gt_target, pred_target)
    raise ValueError("Unsupported pseudo_target_type: {}".format(pseudo_target_type))
