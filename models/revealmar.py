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
        self.candidate_selection_mode = candidate_selection_mode
        self.budget_mode = budget_mode
        self.mixed_policy_ratio = mixed_policy_ratio

        super().__init__(**kwargs)

        valid_pseudo_target_types = {'none', 'gt_reveal', 'pred_reveal', 'mixed_reveal'}
        valid_sampling_policies = {'baseline', 'planner'}
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

    def extra_repr(self):
        return (
            f"planner_hidden_dim={self.planner_hidden_dim}, "
            f"planner_loss_weight={self.planner_loss_weight}, "
            f"candidate_pool_size={self.candidate_pool_size}, "
            f"pseudo_target_type={self.pseudo_target_type}, "
            f"sampling_policy={self.sampling_policy}, "
            f"candidate_selection_mode={self.candidate_selection_mode}, "
            f"budget_mode={self.budget_mode}, "
            f"mixed_policy_ratio={self.mixed_policy_ratio}"
        )

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

    def _select_planner_reveal_mask(self, z, mask, reveal_count):
        bsz = mask.size(0)
        mask_bool = mask.bool()
        if reveal_count <= 0 or not mask_bool.any():
            return torch.zeros_like(mask_bool)

        planner_scores_masked, masked_positions, masked_coords = self._compute_masked_planner_scores(z, mask)
        masked_token_count = planner_scores_masked.size(1)
        reveal_count = min(int(reveal_count), masked_token_count)
        if reveal_count == masked_token_count:
            return mask_bool.clone()

        candidate_pool_size = max(self.candidate_pool_size, reveal_count)
        candidate_indices = build_candidate_subset(
            planner_scores_masked,
            candidate_pool_size,
            masked_coords=masked_coords,
            selection_mode=self.candidate_selection_mode,
        )

        candidate_scores = torch.gather(planner_scores_masked, dim=1, index=candidate_indices)
        selected_within_candidate = torch.topk(candidate_scores, k=reveal_count, dim=-1, largest=True).indices
        selected_masked_indices = torch.gather(candidate_indices, dim=1, index=selected_within_candidate)
        selected_full_positions = torch.gather(masked_positions, dim=1, index=selected_masked_indices)

        mask_to_pred = torch.zeros(bsz, self.seq_len, device=mask.device, dtype=torch.bool)
        mask_to_pred.scatter_(dim=1, index=selected_full_positions, src=torch.ones_like(selected_full_positions, dtype=torch.bool))
        return mask_to_pred

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

    def forward(self, imgs, labels):
        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

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

        return total_loss

    @torch.no_grad()
    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        if self.sampling_policy == 'baseline':
            return super().sample_tokens(
                bsz, num_iter=num_iter, cfg=cfg, cfg_schedule=cfg_schedule,
                labels=labels, temperature=temperature, progress=progress
            )

        mask = torch.ones(bsz, self.seq_len, device=self.mask_token.device)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=self.mask_token.device)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

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
            planner_scores_masked, _, _ = self._compute_masked_planner_scores(z_cond, mask)
            reveal_count = self._compute_reveal_budget(planner_scores_masked, masked_token_count, step, num_iter)
            if reveal_count <= 0 and masked_token_count > 0:
                reveal_count = masked_token_count if step >= num_iter - 1 else 1
            mask_to_pred = self._select_planner_reveal_mask(z_cond, mask, reveal_count)
            if not mask_to_pred.any() and masked_token_count > 0:
                mask_to_pred = self._fallback_reveal_mask(mask, reveal_all=(step >= num_iter - 1))

            selected_count = int(mask_to_pred.sum(dim=1)[0].item()) if mask_to_pred.any() else 0
            assert torch.all(mask_to_pred.sum(dim=1) == selected_count), "Expected equal selected-token count per sample"
            if selected_count <= 0:
                raise RuntimeError("Planner sampling selected no tokens to reveal")

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
