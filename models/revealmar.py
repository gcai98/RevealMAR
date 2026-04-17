from functools import partial

import torch.nn as nn

from models.mar import MAR


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

        # Planner head is intentionally separate from baseline diffusion/value path.
        self.planner_head = nn.Sequential(
            nn.Linear(self.decoder_norm.normalized_shape[0], self.planner_hidden_dim),
            nn.GELU(),
            nn.Linear(self.planner_hidden_dim, 1),
        )

        # Runtime debug cache for inspection; not used by baseline training loop.
        self.latest_planner_scores_masked = None

    def extra_repr(self):
        return (
            f"planner_hidden_dim={self.planner_hidden_dim}, "
            f"planner_loss_weight={self.planner_loss_weight}, "
            f"candidate_pool_size={self.candidate_pool_size}, "
            f"pseudo_target_type={self.pseudo_target_type}, "
            f"budget_mode={self.budget_mode}, "
            f"mixed_policy_ratio={self.mixed_policy_ratio}"
        )

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

        # Planner scores are produced per token, then sliced to masked tokens.
        planner_scores = self.planner_head(z).squeeze(-1)  # [bsz, seq_len]
        mask_bool = mask.bool()
        assert planner_scores.shape == mask_bool.shape, "Planner score/mask shape mismatch"

        # random_masking currently uses a shared masked-token count across batch.
        masked_counts = mask_bool.sum(dim=1)
        assert torch.all(masked_counts == masked_counts[0]), "Expected equal masked-token count per sample"

        masked_token_count = int(masked_counts[0].item())
        planner_scores_masked = planner_scores[mask_bool].view(planner_scores.size(0), masked_token_count)
        self.latest_planner_scores_masked = planner_scores_masked

        # Stub for future phases: pseudo-target utility, mixed policy, and score-budget coupling.
        planner_aux_loss = planner_scores_masked.new_zeros(())
        return loss + self.planner_loss_weight * planner_aux_loss


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
