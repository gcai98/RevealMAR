import torch


def build_candidate_subset(masked_scores, candidate_pool_size):
    """Select top-k candidate masked tokens per sample by planner score."""
    assert masked_scores.dim() == 2, "masked_scores must be [bsz, masked_token_count]"
    bsz, masked_token_count = masked_scores.shape
    if masked_token_count == 0:
        empty = torch.empty(bsz, 0, device=masked_scores.device, dtype=torch.long)
        return empty
    k = min(int(candidate_pool_size), masked_token_count)
    candidate_indices = torch.topk(masked_scores, k=k, dim=-1, largest=True).indices
    return candidate_indices


def _scatter_candidate_values(base_tensor, candidate_indices, candidate_values):
    """Scatter candidate-only values back to masked-token layout."""
    out = torch.zeros_like(base_tensor)
    out.scatter_(dim=1, index=candidate_indices, src=candidate_values)
    return out


def _normalize_per_sample(values, eps=1e-6):
    v_min = values.min(dim=1, keepdim=True).values
    v_max = values.max(dim=1, keepdim=True).values
    return (values - v_min) / (v_max - v_min + eps)


def build_gt_reveal_target(masked_scores, candidate_indices, masked_decoder_tokens):
    """
    Build a lightweight local future-aware surrogate target on candidate subset.
    This approximates reveal utility using local token novelty + latent energy.
    """
    if candidate_indices.numel() == 0:
        return torch.zeros_like(masked_scores)

    bsz, masked_token_count, hidden_dim = masked_decoder_tokens.shape
    assert masked_scores.shape == (bsz, masked_token_count), "masked_scores shape mismatch"

    gather_index = candidate_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
    candidate_tokens = torch.gather(masked_decoder_tokens.detach(), dim=1, index=gather_index)
    candidate_scores = torch.gather(masked_scores.detach(), dim=1, index=candidate_indices)

    # Local continuation proxy: compare each candidate against the masked-token context centroid.
    context_token = masked_decoder_tokens.detach().mean(dim=1, keepdim=True)
    cosine_sim = torch.nn.functional.cosine_similarity(candidate_tokens, context_token, dim=-1)
    novelty = 1.0 - cosine_sim

    token_energy = candidate_tokens.pow(2).mean(dim=-1).sqrt()
    score_prior = torch.tanh(candidate_scores)
    surrogate_utility = novelty + 0.25 * token_energy + 0.1 * score_prior
    surrogate_utility = _normalize_per_sample(surrogate_utility)
    return _scatter_candidate_values(masked_scores, candidate_indices, surrogate_utility)


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


def build_pseudo_target(masked_scores, candidate_indices, masked_decoder_tokens, pseudo_target_type):
    """
    Build pseudo-utility targets with candidate-subset-only construction.
    This keeps rollout/counterfactual refinement pluggable for later phases.
    """
    if pseudo_target_type == 'none':
        return None
    if pseudo_target_type == 'gt_reveal':
        return build_gt_reveal_target(masked_scores, candidate_indices, masked_decoder_tokens)
    if pseudo_target_type == 'pred_reveal':
        return build_pred_reveal_target(masked_scores, candidate_indices)
    if pseudo_target_type == 'mixed_reveal':
        gt_target = build_gt_reveal_target(masked_scores, candidate_indices, masked_decoder_tokens)
        pred_target = build_pred_reveal_target(masked_scores, candidate_indices)
        return build_mixed_reveal_target(gt_target, pred_target)
    raise ValueError("Unsupported pseudo_target_type: {}".format(pseudo_target_type))
