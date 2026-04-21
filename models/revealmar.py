from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        budget_mode='soft',
        mixed_policy_ratio=0.0,
        **kwargs
    ):
        self.planner_hidden_dim = planner_hidden_dim
        self.planner_loss_weight = planner_loss_weight
        self.candidate_pool_size = candidate_pool_size
        self.pseudo_target_type = pseudo_target_type
        self.budget_mode = budget_mode
        self.mixed_policy_ratio = mixed_policy_ratio

        super().__init__(**kwargs)

        valid_pseudo_target_types = {'none', 'gt_reveal', 'pred_reveal', 'mixed_reveal'}
        if self.pseudo_target_type not in valid_pseudo_target_types:
            raise ValueError(
                "Unsupported pseudo_target_type {}. Expected one of {}".format(
                    self.pseudo_target_type, sorted(valid_pseudo_target_types)
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
            f"budget_mode={self.budget_mode}, "
            f"mixed_policy_ratio={self.mixed_policy_ratio}"
        )

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

        # Planner scores are produced per token, then sliced to masked tokens.
        planner_scores = self.planner_head(z).squeeze(-1)  # [bsz, seq_len]
        mask_bool = mask.bool()
        assert planner_scores.shape == mask_bool.shape, "Planner score/mask shape mismatch"

        # random_masking currently uses a shared masked-token count across batch.
        masked_counts = mask_bool.sum(dim=1)
        assert torch.all(masked_counts == masked_counts[0]), "Expected equal masked-token count per sample"

        masked_token_count = int(masked_counts[0].item())
        planner_scores_masked = planner_scores[mask_bool].view(planner_scores.size(0), masked_token_count)
        masked_decoder_tokens = z[mask_bool].view(z.size(0), masked_token_count, z.size(-1))
        # Project GT latent tokens through the same token->encoder->decoder path to match decoder feature space.
        gt_encoder_tokens = self.z_proj_ln(self.z_proj(gt_latents))
        gt_decoder_tokens = self.decoder_embed(gt_encoder_tokens) + self.diffusion_pos_embed_learned
        masked_gt_decoder_tokens = gt_decoder_tokens[mask_bool].view(z.size(0), masked_token_count, z.size(-1))

        # Build per-token 2D latent-grid coordinates, then index masked coordinates.
        # seq_h * seq_w == seq_len in MAR; coordinates align with token flatten order.
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.seq_h, device=z.device),
            torch.arange(self.seq_w, device=z.device),
            indexing='ij',
        )
        all_coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1).to(z.dtype)
        masked_coords = all_coords.unsqueeze(0).expand(z.size(0), -1, -1)[mask_bool].view(z.size(0), masked_token_count, 2)
        self.latest_planner_scores_masked = planner_scores_masked

        candidate_indices = build_candidate_subset(planner_scores_masked, self.candidate_pool_size)
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
