from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import util.misc as misc
from models.mar import MAR
from util.revealmar_utils import build_candidate_subset, build_pseudo_target


class RevealMAR(MAR):
    """Minimal RevealMAR wrapper around baseline MAR."""

    def __init__(
        self,
        planner_hidden_dim=128,
        planner_loss_weight=1.0,
        candidate_pool_size=8,
        pseudo_target_type='none',
        sampling_policy='baseline',
        uncertainty_mc_samples=2,
        uncertainty_policy_temperature=1.0,
        candidate_selection_mode='topk',
        budget_mode='soft',
        mixed_policy_ratio=0.0,
        **kwargs
    ):
        self.planner_hidden_dim = planner_hidden_dim
        self.planner_loss_weight = planner_loss_weight
        self.candidate_pool_size = candidate_pool_size
        self.pseudo_target_type = pseudo_target_type
        self.sampling_policy = sampling_policy
        self.uncertainty_mc_samples = uncertainty_mc_samples
        self.uncertainty_policy_temperature = uncertainty_policy_temperature
        self.candidate_selection_mode = candidate_selection_mode
        self.budget_mode = budget_mode
        self.mixed_policy_ratio = mixed_policy_ratio

        super().__init__(**kwargs)

        valid_pseudo_target_types = {'none', 'gt_reveal', 'pred_reveal', 'mixed_reveal'}
        valid_sampling_policies = {'baseline', 'planner', 'random', 'confidence', 'entropy'}
        valid_candidate_selection_modes = {'topk', 'mixed'}
        if self.pseudo_target_type not in valid_pseudo_target_types:
            raise ValueError(
                "Unsupported pseudo_target_type {}. Expected one of {}".format(
                    self.pseudo_target_type, sorted(valid_pseudo_target_types)
                )
            )
        if self.sampling_policy not in valid_sampling_policies:
            raise ValueError(
                "Unsupported sampling_policy {}. Expected one of {}".format(
                    self.sampling_policy, sorted(valid_sampling_policies)
                )
            )
        if self.candidate_selection_mode not in valid_candidate_selection_modes:
            raise ValueError(
                "Unsupported candidate_selection_mode {}. Expected one of {}".format(
                    self.candidate_selection_mode, sorted(valid_candidate_selection_modes)
                )
            )
        if not 0.0 <= self.mixed_policy_ratio <= 1.0:
            raise ValueError(
                "Unsupported mixed_policy_ratio {}. Expected value in [0, 1]".format(self.mixed_policy_ratio)
            )
        if int(self.uncertainty_mc_samples) < 1:
            raise ValueError(
                "Unsupported uncertainty_mc_samples {}. Expected value >= 1".format(self.uncertainty_mc_samples)
            )

        # Planner head is intentionally separate from baseline diffusion/value path.
        self.planner_head = nn.Sequential(
            nn.Linear(self.decoder_norm.normalized_shape[0], self.planner_hidden_dim),
            nn.GELU(),
            nn.Linear(self.planner_hidden_dim, 1),
        )

        # Runtime debug cache for inspection; not used by baseline training loop.
        self.latest_planner_scores_masked = None
        self.latest_candidate_indices = None
        self.latest_pseudo_target = None
        self.latest_diffloss = None
        self.latest_planner_aux_loss = None
        self.latest_total_loss = None
        self.latest_mixed_policy_applied = False
        self.latest_mixed_policy_swap_count = 0
        self.latest_mixed_policy_fraction = 0.0
        self.latest_mixed_policy_mask_delta = 0
        self.latest_mixed_policy_ratio_effective = 0.0
        self.latest_sampling_budget_trajectory = None
        self.latest_sampling_selected_trajectory = None
        self.latest_sampling_entropy_trajectory = None
        self.latest_sampling_score_stats_trajectory = None
        self.latest_sampling_uncertainty_stats_trajectory = None

    def extra_repr(self):
        return (
            f"planner_hidden_dim={self.planner_hidden_dim}, "
            f"planner_loss_weight={self.planner_loss_weight}, "
            f"candidate_pool_size={self.candidate_pool_size}, "
            f"pseudo_target_type={self.pseudo_target_type}, "
            f"sampling_policy={self.sampling_policy}, "
            f"uncertainty_mc_samples={self.uncertainty_mc_samples}, "
            f"uncertainty_policy_temperature={self.uncertainty_policy_temperature}, "
            f"candidate_selection_mode={self.candidate_selection_mode}, "
            f"budget_mode={self.budget_mode}, "
            f"mixed_policy_ratio={self.mixed_policy_ratio}"
        )

    def _get_effective_mixed_policy_ratio(self):
        schedule = getattr(self, '_revealmar_mixed_policy_ratio_schedule', 'constant')
        warmup_epochs = int(getattr(self, '_revealmar_mixed_policy_ratio_warmup_epochs', 0) or 0)
        current_epoch = int(getattr(self, '_revealmar_current_epoch', 0) or 0)
        base_ratio = float(self.mixed_policy_ratio)
        if schedule == 'constant' or warmup_epochs <= 0:
            return base_ratio
        if schedule == 'linear_warmup':
            progress = min(max((current_epoch + 1) / float(warmup_epochs), 0.0), 1.0)
            return base_ratio * progress
        raise ValueError("Unsupported mixed-policy ratio schedule: {}".format(schedule))

    def _maybe_log_mixed_policy_debug(self):
        log_freq = int(getattr(self, '_revealmar_mixed_policy_log_freq', 0) or 0)
        if log_freq <= 0:
            return
        self._revealmar_mixed_policy_fwd_count = getattr(self, '_revealmar_mixed_policy_fwd_count', 0) + 1
        if self._revealmar_mixed_policy_fwd_count % log_freq == 0 and misc.is_main_process():
            print(
                '[RevealMAR][mixed-policy] fwd={} applied={} ratio_eff={:.4f} swap_count={} perturbed_fraction={:.4f} mask_delta={}'.format(
                    self._revealmar_mixed_policy_fwd_count,
                    int(bool(self.latest_mixed_policy_applied)),
                    float(self.latest_mixed_policy_ratio_effective),
                    int(self.latest_mixed_policy_swap_count),
                    float(self.latest_mixed_policy_fraction),
                    int(self.latest_mixed_policy_mask_delta),
                )
            )

    def _planner_score_entropy_summary(self, planner_scores_masked):
        if planner_scores_masked.numel() == 0:
            return 0.0
        probs = torch.softmax(planner_scores_masked.float(), dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy_norm = entropy / math.log(float(max(planner_scores_masked.size(1), 2)))
        concentration = (1.0 - entropy_norm).clamp(0.0, 1.0)
        return float(concentration.mean().item())

    def _planner_score_debug_summary(self, planner_scores_masked, topk_count):
        if planner_scores_masked.numel() == 0 or planner_scores_masked.size(1) == 0:
            return {
                'score_mean': 0.0,
                'score_std': 0.0,
                'score_min': 0.0,
                'score_max': 0.0,
                'top1_score_mean': 0.0,
                'topk_score_mean': 0.0,
            }

        scores = torch.nan_to_num(planner_scores_masked.float(), nan=0.0, posinf=0.0, neginf=0.0)
        top1_values = torch.topk(scores, k=1, dim=-1, largest=True).values
        k = min(max(int(topk_count), 0), scores.size(1))
        if k > 0:
            topk_values = torch.topk(scores, k=k, dim=-1, largest=True).values
            topk_score_mean = float(topk_values.mean().item())
        else:
            topk_score_mean = 0.0

        return {
            'score_mean': float(scores.mean().item()),
            'score_std': float(scores.std(unbiased=False).item()),
            'score_min': float(scores.min().item()),
            'score_max': float(scores.max().item()),
            'top1_score_mean': float(top1_values.mean().item()),
            'topk_score_mean': topk_score_mean,
        }

    def _uncertainty_debug_summary(self, uncertainty_scores_masked):
        if uncertainty_scores_masked is None or uncertainty_scores_masked.numel() == 0:
            return None

        uncertainty = torch.nan_to_num(uncertainty_scores_masked.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return {
            'uncertainty_mean': float(uncertainty.mean().item()),
            'uncertainty_std': float(uncertainty.std(unbiased=False).item()),
            'uncertainty_min': float(uncertainty.min().item()),
            'uncertainty_max': float(uncertainty.max().item()),
        }

    def _build_masked_coords(self, mask_bool, dtype, device):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.seq_h, device=device),
            torch.arange(self.seq_w, device=device),
            indexing='ij',
        )
        all_coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1).to(dtype)
        masked_counts = mask_bool.sum(dim=1)
        assert torch.all(masked_counts == masked_counts[0]), "Expected equal masked-token count per sample"
        masked_token_count = int(masked_counts[0].item())
        masked_coords = all_coords.unsqueeze(0).expand(mask_bool.size(0), -1, -1)[mask_bool]
        return masked_coords.view(mask_bool.size(0), masked_token_count, 2)

    def _compute_masked_planner_scores(self, z, mask):
        planner_scores = torch.nan_to_num(self.planner_head(z).squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
        mask_bool = mask.bool()
        assert planner_scores.shape == mask_bool.shape, "Planner score/mask shape mismatch"
        masked_counts = mask_bool.sum(dim=1)
        assert torch.all(masked_counts == masked_counts[0]), "Expected equal masked-token count per sample"
        masked_token_count = int(masked_counts[0].item())
        planner_scores_masked = planner_scores[mask_bool].view(planner_scores.size(0), masked_token_count)
        masked_positions = mask_bool.nonzero(as_tuple=True)[1].view(planner_scores.size(0), masked_token_count)
        masked_coords = self._build_masked_coords(mask_bool, dtype=z.dtype, device=z.device)
        return planner_scores_masked, masked_positions, masked_coords

    def _compute_random_masked_scores(self, mask):
        mask_bool = mask.bool()
        masked_counts = mask_bool.sum(dim=1)
        assert torch.all(masked_counts == masked_counts[0]), "Expected equal masked-token count per sample"
        masked_token_count = int(masked_counts[0].item())
        random_scores_masked = torch.rand(mask.size(0), masked_token_count, device=mask.device, dtype=torch.float32)
        masked_positions = mask_bool.nonzero(as_tuple=True)[1].view(mask.size(0), masked_token_count)
        masked_coords = self._build_masked_coords(mask_bool, dtype=mask.dtype, device=mask.device)
        return random_scores_masked, masked_positions, masked_coords

    def _compute_uncertainty_masked_scores(self, z, mask, mc_samples=None, temperature=None):
        mask_bool = mask.bool()
        masked_counts = mask_bool.sum(dim=1)
        assert torch.all(masked_counts == masked_counts[0]), "Expected equal masked-token count per sample"
        masked_token_count = int(masked_counts[0].item())
        masked_positions = mask_bool.nonzero(as_tuple=True)[1].view(mask.size(0), masked_token_count)
        masked_coords = self._build_masked_coords(mask_bool, dtype=z.dtype, device=z.device)
        if masked_token_count <= 0:
            empty = torch.empty(mask.size(0), 0, device=z.device, dtype=torch.float32)
            return empty, masked_positions, masked_coords

        mc = max(1, int(self.uncertainty_mc_samples if mc_samples is None else mc_samples))
        sample_temperature = float(
            self.uncertainty_policy_temperature if temperature is None else temperature
        )
        z_masked = z[mask_bool].view(mask.size(0), masked_token_count, z.size(-1)).detach()
        z_repeated = z_masked.reshape(-1, z_masked.size(-1)).repeat(mc, 1)

        with torch.no_grad():
            sampled = self.diffloss.sample(z_repeated, sample_temperature, cfg=1.0)
            sampled = sampled.detach().view(mc, mask.size(0), masked_token_count, -1).float()
            uncertainty_scores_masked = sampled.var(dim=0, unbiased=False).mean(dim=-1)
            uncertainty_scores_masked = torch.nan_to_num(
                uncertainty_scores_masked, nan=0.0, posinf=0.0, neginf=0.0
            )

        return uncertainty_scores_masked, masked_positions, masked_coords

    def _compute_scheduled_reveal_count(self, masked_token_count, step, num_iter):
        if masked_token_count <= 0:
            return 0
        if step >= num_iter - 1:
            return masked_token_count

        mask_ratio = math.cos(math.pi / 2.0 * float(step + 1) / float(num_iter))
        next_mask_len = int(math.floor(self.seq_len * mask_ratio))
        next_mask_len = max(1, min(masked_token_count - 1, next_mask_len))
        reveal_count = masked_token_count - next_mask_len
        if masked_token_count > 1:
            reveal_count = max(1, reveal_count)
        return reveal_count

    def _compute_reveal_budget(self, planner_scores_masked, masked_token_count, step, num_iter):
        hard_reveal_count = self._compute_scheduled_reveal_count(masked_token_count, step, num_iter)
        if self.budget_mode == 'hard' or masked_token_count <= 1 or step >= num_iter - 1:
            return hard_reveal_count

        # "soft" budget uses planner-score concentration to scale the fixed schedule budget:
        # lower entropy => sharper planner preference => reveal more this step.
        probs = torch.softmax(planner_scores_masked.float(), dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy_norm = entropy / math.log(float(max(masked_token_count, 2)))
        concentration = (1.0 - entropy_norm).clamp(0.0, 1.0).mean().item()

        scaled_reveal = int(round(hard_reveal_count * (0.5 + 0.5 * concentration)))
        return max(1, min(masked_token_count - 1, scaled_reveal))

    def _select_reveal_mask_from_scores(self, mask, reveal_count, masked_scores, masked_positions, masked_coords):
        bsz = mask.size(0)
        mask_bool = mask.bool()
        if reveal_count <= 0 or not mask_bool.any():
            return torch.zeros_like(mask_bool)

        masked_token_count = masked_scores.size(1)
        reveal_count = min(int(reveal_count), masked_token_count)
        if reveal_count == masked_token_count:
            return mask_bool.clone()

        candidate_pool_size = max(self.candidate_pool_size, reveal_count)
        candidate_indices = build_candidate_subset(
            masked_scores,
            candidate_pool_size,
            masked_coords=masked_coords,
            selection_mode=self.candidate_selection_mode,
        )

        candidate_scores = torch.gather(masked_scores, dim=1, index=candidate_indices)
        selected_within_candidate = torch.topk(candidate_scores, k=reveal_count, dim=-1, largest=True).indices
        selected_masked_indices = torch.gather(candidate_indices, dim=1, index=selected_within_candidate)
        selected_full_positions = torch.gather(masked_positions, dim=1, index=selected_masked_indices)

        mask_to_pred = torch.zeros(bsz, self.seq_len, device=mask.device, dtype=torch.bool)
        mask_to_pred.scatter_(dim=1, index=selected_full_positions, src=torch.ones_like(selected_full_positions, dtype=torch.bool))
        return mask_to_pred

    def _select_planner_reveal_mask(self, z, mask, reveal_count):
        planner_scores_masked, masked_positions, masked_coords = self._compute_masked_planner_scores(z, mask)
        return self._select_reveal_mask_from_scores(
            mask, reveal_count, planner_scores_masked, masked_positions, masked_coords
        )

    def _fallback_reveal_mask(self, mask, reveal_all=False):
        mask_bool = mask.bool()
        if not mask_bool.any():
            return torch.zeros_like(mask_bool)
        if reveal_all:
            return mask_bool.clone()

        masked_counts = mask_bool.sum(dim=1)
        assert torch.all(masked_counts == masked_counts[0]), "Expected equal masked-token count per sample"
        masked_positions = mask_bool.nonzero(as_tuple=True)[1].view(mask.size(0), int(masked_counts[0].item()))
        fallback_positions = masked_positions[:, :1]
        fallback_mask = torch.zeros_like(mask_bool)
        fallback_mask.scatter_(dim=1, index=fallback_positions, src=torch.ones_like(fallback_positions, dtype=torch.bool))
        return fallback_mask

    def _compute_planner_supervision_loss(self, planner_scores_masked, candidate_indices, pseudo_target):
        """
        Compute planner supervision only on candidate subset.
        Pseudo targets are detached to avoid backprop through target construction.
        """
        if pseudo_target is None or candidate_indices.numel() == 0:
            return planner_scores_masked.new_zeros(())

        candidate_scores = torch.gather(planner_scores_masked, dim=1, index=candidate_indices)
        candidate_targets = torch.gather(pseudo_target.detach(), dim=1, index=candidate_indices)

        # Regression term aligns absolute utility scale on candidate subset.
        regression_loss = F.mse_loss(candidate_scores, candidate_targets)

        # Pairwise term aligns candidate ranking induced by pseudo-utility.
        k = candidate_scores.size(1)
        if k < 2:
            return regression_loss

        score_diff = candidate_scores.unsqueeze(2) - candidate_scores.unsqueeze(1)
        target_diff = candidate_targets.unsqueeze(2) - candidate_targets.unsqueeze(1)
        rank_sign = torch.sign(target_diff)

        pair_mask = torch.triu(torch.ones(k, k, device=candidate_scores.device, dtype=torch.bool), diagonal=1)
        valid_pair_mask = pair_mask.unsqueeze(0) & (rank_sign != 0)
        if not valid_pair_mask.any():
            return regression_loss

        pairwise_loss = F.softplus(-(rank_sign * score_diff))[valid_pair_mask].mean()
        return regression_loss + pairwise_loss

    def _apply_mixed_policy_training_mask(self, x, mask, orders, class_embedding):
        """
        Mixed-policy exposure for training only.

        Start from the reference random-mask state, then swap a small fraction of tokens:
        planner-preferred masked tokens become visible, and an equal number of currently
        visible reference-order frontier tokens are masked again. This preserves the total
        masked-token count while shifting the observed partial state toward planner influence.
        """
        effective_ratio = self._get_effective_mixed_policy_ratio()
        self.latest_mixed_policy_applied = False
        self.latest_mixed_policy_swap_count = 0
        self.latest_mixed_policy_fraction = 0.0
        self.latest_mixed_policy_mask_delta = 0
        self.latest_mixed_policy_ratio_effective = effective_ratio
        if not self.training or effective_ratio <= 0.0:
            return mask

        mask_bool = mask.bool()
        masked_counts = mask_bool.sum(dim=1)
        assert torch.all(masked_counts == masked_counts[0]), "Expected equal masked-token count per sample"
        masked_token_count = int(masked_counts[0].item())
        visible_token_count = self.seq_len - masked_token_count
        if masked_token_count <= 0 or visible_token_count <= 0:
            return mask

        swap_count = int(round(masked_token_count * effective_ratio))
        swap_count = min(swap_count, masked_token_count, visible_token_count)
        if swap_count <= 0:
            return mask

        with torch.no_grad():
            x_ref = self.forward_mae_encoder(x, mask, class_embedding)
            z_ref = self.forward_mae_decoder(x_ref, mask)
            planner_scores_masked, masked_positions, _ = self._compute_masked_planner_scores(z_ref, mask)
            reveal_within_masked = torch.topk(
                planner_scores_masked, k=swap_count, dim=-1, largest=True
            ).indices
            reveal_positions = torch.gather(masked_positions, dim=1, index=reveal_within_masked)

        # Re-mask an equal-size slice from the reference visible frontier to keep mask count unchanged.
        remask_positions = orders[:, masked_token_count:masked_token_count + swap_count]
        mixed_mask = mask_bool.clone()
        for b in range(mask.size(0)):
            mixed_mask[b, reveal_positions[b]] = False
            mixed_mask[b, remask_positions[b]] = True

        mixed_counts = mixed_mask.sum(dim=1)
        assert torch.all(mixed_counts == masked_counts), "Mixed-policy mask should preserve masked-token count"
        self.latest_mixed_policy_applied = True
        self.latest_mixed_policy_swap_count = swap_count
        self.latest_mixed_policy_fraction = float(swap_count) / float(max(masked_token_count, 1))
        self.latest_mixed_policy_mask_delta = int(torch.logical_xor(mask_bool, mixed_mask).sum(dim=1)[0].item())
        return mixed_mask.to(dtype=mask.dtype)

    def forward(self, imgs, labels):
        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)
        mask = self._apply_mixed_policy_training_mask(x, mask, orders, class_embedding)

        # mae encoder + decoder
        x = self.forward_mae_encoder(x, mask, class_embedding)
        z = self.forward_mae_decoder(x, mask)

        # diffloss remains the baseline optimization objective.
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)
        self.latest_diffloss = loss.detach()

        mask_bool = mask.bool()
        masked_counts = mask_bool.sum(dim=1)
        masked_token_count = int(masked_counts[0].item())
        planner_scores_masked, _, masked_coords = self._compute_masked_planner_scores(z, mask)
        masked_decoder_tokens = z[mask_bool].view(z.size(0), masked_token_count, z.size(-1))
        # Project GT latent tokens through the same token->encoder->decoder path to match decoder feature space.
        gt_encoder_tokens = self.z_proj_ln(self.z_proj(gt_latents))
        gt_decoder_tokens = self.decoder_embed(gt_encoder_tokens) + self.diffusion_pos_embed_learned
        masked_gt_decoder_tokens = gt_decoder_tokens[mask_bool].view(z.size(0), masked_token_count, z.size(-1))
        self.latest_planner_scores_masked = planner_scores_masked

        candidate_indices = build_candidate_subset(
            planner_scores_masked,
            self.candidate_pool_size,
            masked_coords=masked_coords,
            selection_mode=self.candidate_selection_mode,
        )
        pseudo_target = build_pseudo_target(
            planner_scores_masked,
            candidate_indices,
            masked_decoder_tokens,
            masked_gt_decoder_tokens,
            masked_coords,
            pseudo_target_type=self.pseudo_target_type,
        )
        self.latest_candidate_indices = candidate_indices
        self.latest_pseudo_target = pseudo_target

        planner_aux_loss = self._compute_planner_supervision_loss(
            planner_scores_masked=planner_scores_masked,
            candidate_indices=candidate_indices,
            pseudo_target=pseudo_target,
        )
        self.latest_planner_aux_loss = planner_aux_loss.detach()
        total_loss = loss + self.planner_loss_weight * planner_aux_loss
        self.latest_total_loss = total_loss.detach()

        log_freq = int(getattr(self, '_revealmar_loss_log_freq', 0) or 0)
        if log_freq > 0:
            self._revealmar_fwd_count = getattr(self, '_revealmar_fwd_count', 0) + 1
            if self._revealmar_fwd_count % log_freq == 0 and misc.is_main_process():
                print(
                    '[RevealMAR] fwd={} total_loss={:.6f} diffloss={:.6f} planner_aux_loss={:.6f}'.format(
                        self._revealmar_fwd_count,
                        float(self.latest_total_loss),
                        float(self.latest_diffloss),
                        float(self.latest_planner_aux_loss),
                    )
                )

        self._maybe_log_mixed_policy_debug()
        return total_loss

    @torch.no_grad()
    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        if self.sampling_policy == 'baseline':
            return super().sample_tokens(
                bsz, num_iter=num_iter, cfg=cfg, cfg_schedule=cfg_schedule,
                labels=labels, temperature=temperature, progress=progress
            )
        if self.sampling_policy in ('confidence', 'entropy') and cfg != 1.0:
            raise NotImplementedError(
                "confidence/entropy sampling policies do not support cfg != 1.0 yet; "
                "use --cfg 1.0 or sampling_policy baseline/planner/random"
            )

        mask = torch.ones(bsz, self.seq_len, device=self.mask_token.device)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=self.mask_token.device)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        debug_sampling = bool(getattr(self, '_revealmar_sampling_debug', False))
        debug_sampling_steps = int(getattr(self, '_revealmar_sampling_debug_steps', 8) or 8)
        budget_trajectory = []
        selected_trajectory = []
        entropy_trajectory = []
        score_stats_trajectory = []
        uncertainty_stats_trajectory = []

        for step in indices:
            if not mask.bool().any():
                break

            cur_tokens = tokens.clone()
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)

            tokens_model = tokens
            mask_model = mask
            if cfg != 1.0:
                tokens_model = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask_model = torch.cat([mask, mask], dim=0)

            x = self.forward_mae_encoder(tokens_model, mask_model, class_embedding)
            z = self.forward_mae_decoder(x, mask_model)
            z_cond = z[:bsz]

            masked_counts = mask.sum(dim=1)
            assert torch.all(masked_counts == masked_counts[0]), "Expected equal masked-token count per sample"
            masked_token_count = int(masked_counts[0].item())
            if self.sampling_policy == 'planner':
                sampling_scores_masked, masked_positions, masked_coords = self._compute_masked_planner_scores(z_cond, mask)
                uncertainty_scores_masked = None
            elif self.sampling_policy == 'random':
                sampling_scores_masked, masked_positions, masked_coords = self._compute_random_masked_scores(mask)
                uncertainty_scores_masked = None
            elif self.sampling_policy in ('confidence', 'entropy'):
                uncertainty_scores_masked, masked_positions, masked_coords = self._compute_uncertainty_masked_scores(
                    z_cond,
                    mask,
                    mc_samples=self.uncertainty_mc_samples,
                    temperature=self.uncertainty_policy_temperature,
                )
                if self.sampling_policy == 'entropy':
                    sampling_scores_masked = uncertainty_scores_masked
                else:
                    sampling_scores_masked = -uncertainty_scores_masked
            else:
                raise ValueError("Unsupported sampling_policy during sampling: {}".format(self.sampling_policy))

            score_concentration = self._planner_score_entropy_summary(sampling_scores_masked)
            score_stats = self._planner_score_debug_summary(sampling_scores_masked, self.candidate_pool_size)
            uncertainty_stats = self._uncertainty_debug_summary(uncertainty_scores_masked)
            reveal_count = self._compute_reveal_budget(sampling_scores_masked, masked_token_count, step, num_iter)
            if reveal_count <= 0 and masked_token_count > 0:
                reveal_count = masked_token_count if step >= num_iter - 1 else 1
            if self.sampling_policy == 'planner':
                mask_to_pred = self._select_planner_reveal_mask(z_cond, mask, reveal_count)
            else:
                mask_to_pred = self._select_reveal_mask_from_scores(
                    mask, reveal_count, sampling_scores_masked, masked_positions, masked_coords
                )
            if not mask_to_pred.any() and masked_token_count > 0:
                mask_to_pred = self._fallback_reveal_mask(mask, reveal_all=(step >= num_iter - 1))

            selected_count = int(mask_to_pred.sum(dim=1)[0].item()) if mask_to_pred.any() else 0
            assert torch.all(mask_to_pred.sum(dim=1) == selected_count), "Expected equal selected-token count per sample"
            if selected_count <= 0:
                raise RuntimeError("Planner sampling selected no tokens to reveal")
            budget_trajectory.append(int(reveal_count))
            selected_trajectory.append(int(selected_count))
            entropy_trajectory.append(float(score_concentration))
            score_stats_trajectory.append(score_stats)
            uncertainty_stats_trajectory.append(uncertainty_stats)

            if not cfg == 1.0:
                mask_to_pred_model = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            else:
                mask_to_pred_model = mask_to_pred

            z_target = z[mask_to_pred_model.nonzero(as_tuple=True)]
            if cfg_schedule == "linear":
                remaining_after_step = float(masked_token_count - selected_count)
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - remaining_after_step) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            sampled_token_latent = self.diffloss.sample(z_target, temperature, cfg_iter)
            if cfg != 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
            expected_latent_count = int(mask_to_pred.sum().item())
            assert sampled_token_latent.size(0) == expected_latent_count, "Sampled latent count mismatch"

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens
            mask = mask.masked_fill(mask_to_pred, 0.0)

        self.latest_sampling_budget_trajectory = budget_trajectory
        self.latest_sampling_selected_trajectory = selected_trajectory
        self.latest_sampling_entropy_trajectory = entropy_trajectory
        self.latest_sampling_score_stats_trajectory = score_stats_trajectory
        self.latest_sampling_uncertainty_stats_trajectory = uncertainty_stats_trajectory
        if debug_sampling and misc.is_main_process():
            shown_steps = min(debug_sampling_steps, len(budget_trajectory))
            summary = []
            for i in range(shown_steps):
                stats = score_stats_trajectory[i]
                item = (
                    'step{}:budget={},selected={},conc={:.4f},mean={:.4f},std={:.4f},min={:.4f},max={:.4f},top1={:.4f},topk={:.4f}'.format(
                        i,
                        budget_trajectory[i],
                        selected_trajectory[i],
                        entropy_trajectory[i],
                        stats['score_mean'],
                        stats['score_std'],
                        stats['score_min'],
                        stats['score_max'],
                        stats['top1_score_mean'],
                        stats['topk_score_mean'],
                    )
                )
                uncertainty_stats = uncertainty_stats_trajectory[i]
                if uncertainty_stats is not None:
                    item += ',umean={:.4f},ustd={:.4f},umin={:.4f},umax={:.4f}'.format(
                        uncertainty_stats['uncertainty_mean'],
                        uncertainty_stats['uncertainty_std'],
                        uncertainty_stats['uncertainty_min'],
                        uncertainty_stats['uncertainty_max'],
                    )
                summary.append(item)
            print('[RevealMAR][planner-sampling] ' + ' | '.join(summary))

        if mask.bool().any():
            tokens[mask.bool()] = 0.0
        return self.unpatchify(tokens)


def revealmar_base(**kwargs):
    return RevealMAR(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=768,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )


def revealmar_large(**kwargs):
    return RevealMAR(
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )


def revealmar_huge(**kwargs):
    return RevealMAR(
        encoder_embed_dim=1280,
        encoder_depth=20,
        encoder_num_heads=16,
        decoder_embed_dim=1280,
        decoder_depth=20,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
