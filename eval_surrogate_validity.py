#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0-3 Surrogate Validity Table for PlanMAR-S / RevealMAR.

This is a standalone evaluation script. It loads a trained RevealMAR checkpoint,
samples masked decoding states from validation images, compares candidate
ranking signals, and writes paper-table-ready CSV/JSON outputs.

Approximation notes:
- No model training is performed.
- The rollout proxy is teacher-forced: "revealing" a token means making its
  ground-truth latent token visible to the MAR encoder, not sampling an image.
- The reference continuation policy pi_ref uses the existing cosine reveal
  schedule with a fixed random order, matching the baseline MAR ordering rule.
- The local future loss is decoder-feature MSE in a small token neighborhood;
  this keeps the proxy affordable for 128/512-image mechanism runs.
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from models import revealmar
from models.vae import AutoencoderKL, DiagonalGaussianDistribution
from util.crop import center_crop_arr
from util.loader import CachedFolder
from util.revealmar_utils import build_candidate_subset
from util.surrogate_eval_utils import (
    kendall_tau,
    nanmean_or_none,
    ndcg_at_k,
    regret,
    spearman_corr,
    topk_overlap,
)


ROW_ORDER = [
    ("Random", "random"),
    ("Confidence", "confidence"),
    ("Entropy", "entropy"),
    ("Current Loss", "current_loss"),
    ("PlanMAR-S Surrogate", "planner"),
    ("Rollout Proxy / Oracle-like", "oracle"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("P0-3 Surrogate Validity Table")

    # Existing RevealMAR-style args.
    parser.add_argument("--data_path", default="./data/imagenet", type=str)
    parser.add_argument("--vae_path", default="pretrained_models/vae/kl16.ckpt", type=str)
    parser.add_argument("--resume", default="", type=str, help="Checkpoint directory or checkpoint file")
    parser.add_argument("--checkpoint", default="", type=str, help="Alias for --resume")
    parser.add_argument("--output_dir", default="./output_dir/surrogate_validity", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--model", default="revealmar_base", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--vae_embed_dim", default=16, type=int)
    parser.add_argument("--vae_stride", default=16, type=int)
    parser.add_argument("--patch_size", default=1, type=int)
    parser.add_argument("--class_num", default=1000, type=int)
    parser.add_argument("--mask_ratio_min", default=0.7, type=float)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--attn_dropout", default=0.1, type=float)
    parser.add_argument("--proj_dropout", default=0.1, type=float)
    parser.add_argument("--buffer_size", default=64, type=int)
    parser.add_argument("--diffloss_d", default=6, type=int)
    parser.add_argument("--diffloss_w", default=1024, type=int)
    parser.add_argument("--num_sampling_steps", default="100", type=str)
    parser.add_argument("--diffusion_batch_mul", default=1, type=int)
    parser.add_argument("--grad_checkpointing", action="store_true")

    parser.add_argument("--planner_hidden_dim", default=128, type=int)
    parser.add_argument("--planner_loss_weight", default=1.0, type=float)
    parser.add_argument("--candidate_pool_size", default=8, type=int)
    parser.add_argument(
        "--pseudo_target_type",
        default="mixed_reveal",
        choices=["none", "gt_reveal", "pred_reveal", "mixed_reveal"],
    )
    parser.add_argument("--sampling_policy", default="planner", choices=["baseline", "planner"])
    parser.add_argument("--candidate_selection_mode", default="mixed", choices=["topk", "mixed"])
    parser.add_argument("--budget_mode", default="soft", choices=["soft", "hard"])
    parser.add_argument("--mixed_policy_ratio", default=0.0, type=float)

    # P0-3 controls.
    parser.add_argument("--num_eval_images", default=128, type=int)
    parser.add_argument("--eval_bsz", default=8, type=int)
    parser.add_argument("--num_states_per_image", default=1, type=int)
    parser.add_argument("--rollout_horizon", default=2, type=int)
    parser.add_argument("--topk_eval", default=3, type=int)

    # Lightweight extra controls for reproducibility and cost.
    parser.add_argument("--num_iter", default=64, type=int, help="Reference cosine schedule length")
    parser.add_argument("--state_step_min", default=0, type=int)
    parser.add_argument("--state_step_max", default=-1, type=int, help="-1 means num_iter - rollout_horizon - 1")
    parser.add_argument("--local_radius", default=1, type=int)
    parser.add_argument("--uncertainty_mc_samples", default=4, type=int)
    parser.add_argument("--uncertainty_temperature", default=1.0, type=float)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--no_use_ema", action="store_false", dest="use_ema")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--latent_mode", default="mode", choices=["mode", "sample"])
    parser.add_argument("--use_cached", action="store_true", dest="use_cached")
    parser.add_argument("--cached_path", default="", type=str)
    parser.add_argument("--save_raw", action="store_true", help="Save surrogate_validity_raw.npz")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    path = args.checkpoint or args.resume
    if not path:
        raise ValueError("Provide --resume or --checkpoint")
    ckpt = Path(path)
    if ckpt.is_dir():
        ckpt = ckpt / "checkpoint-last.pth"
    if not ckpt.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(ckpt))
    return ckpt


def build_dataset(args: argparse.Namespace):
    if args.use_cached:
        if not args.cached_path:
            raise ValueError("--cached_path is required with --use_cached")
        return CachedFolder(args.cached_path)

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=transform) \
        if os.path.isdir(os.path.join(args.data_path, "val")) \
        else datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)


def build_model(args: argparse.Namespace, device: torch.device):
    model = revealmar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
        planner_hidden_dim=args.planner_hidden_dim,
        planner_loss_weight=args.planner_loss_weight,
        candidate_pool_size=args.candidate_pool_size,
        pseudo_target_type=args.pseudo_target_type,
        sampling_policy=args.sampling_policy,
        candidate_selection_mode=args.candidate_selection_mode,
        budget_mode=args.budget_mode,
        mixed_policy_ratio=args.mixed_policy_ratio,
    )

    ckpt_path = resolve_checkpoint(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_key = "model_ema" if args.use_ema and checkpoint.get("model_ema") is not None else "model"
    result = model.load_state_dict(checkpoint[state_key], strict=False)
    if result.missing_keys:
        print("[WARN] Missing model keys: {}".format(result.missing_keys))
    if result.unexpected_keys:
        print("[WARN] Unexpected model keys: {}".format(result.unexpected_keys))
    model.to(device).eval()
    return model, ckpt_path, state_key


def batch_to_latent_tokens(batch, vae, model, args, device) -> Tuple[torch.Tensor, torch.Tensor]:
    samples, labels = batch[:2]
    labels = labels.to(device, non_blocking=True)
    if args.use_cached:
        posterior = DiagonalGaussianDistribution(samples.to(device, non_blocking=True))
    else:
        posterior = vae.encode(samples.to(device, non_blocking=True))
    latent_image = posterior.mode() if args.latent_mode == "mode" else posterior.sample()
    return model.patchify(latent_image.mul(0.2325)), labels


def make_orders(bsz: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.stack([torch.randperm(seq_len, device=device) for _ in range(bsz)], dim=0)


def mask_len_for_step(seq_len: int, step: int, num_iter: int) -> int:
    if step <= 0:
        return seq_len
    ratio = math.cos(math.pi / 2.0 * float(step) / float(num_iter))
    return max(1, min(seq_len - 1, int(math.floor(seq_len * ratio))))


def mask_from_orders(orders: torch.Tensor, mask_len: int) -> torch.Tensor:
    bsz, seq_len = orders.shape
    mask = torch.zeros(bsz, seq_len, device=orders.device, dtype=torch.bool)
    src = torch.ones(bsz, mask_len, device=orders.device, dtype=torch.bool)
    mask.scatter_(1, orders[:, :mask_len], src)
    return mask


def full_token_coords(model, device: torch.device) -> torch.Tensor:
    grid_y, grid_x = torch.meshgrid(
        torch.arange(model.seq_h, device=device),
        torch.arange(model.seq_w, device=device),
        indexing="ij",
    )
    return torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)


@torch.no_grad()
def decode_state(model, tokens: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    class_embedding = model.class_emb(labels)
    x_enc = model.forward_mae_encoder(tokens, mask.to(tokens.dtype), class_embedding)
    z = model.forward_mae_decoder(x_enc, mask.to(tokens.dtype))
    planner_scores_masked, masked_positions, masked_coords = model._compute_masked_planner_scores(
        z, mask.to(tokens.dtype)
    )
    gt_encoder_tokens = model.z_proj_ln(model.z_proj(tokens))
    gt_decoder_tokens = model.decoder_embed(gt_encoder_tokens) + model.diffusion_pos_embed_learned
    per_token_loss = (z.float() - gt_decoder_tokens.float()).pow(2).mean(dim=-1)
    return {
        "z": z,
        "planner_scores_masked": planner_scores_masked,
        "masked_positions": masked_positions,
        "masked_coords": masked_coords,
        "gt_decoder_tokens": gt_decoder_tokens,
        "per_token_loss": per_token_loss,
    }


def local_loss_from_state(
    state: Dict[str, torch.Tensor],
    eval_mask: torch.Tensor,
    candidate_positions: torch.Tensor,
    coords: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    losses = []
    sq = (state["z"].float() - state["gt_decoder_tokens"].float()).pow(2).mean(dim=-1)
    for i in range(state["z"].size(0)):
        pos = int(candidate_positions[i].item())
        dist = torch.max(torch.abs(coords - coords[pos]), dim=-1).values
        local = dist <= radius
        cur_eval = eval_mask[i] & local
        if not cur_eval.any():
            cur_eval = eval_mask[i]
        losses.append(sq[i, cur_eval].mean() if cur_eval.any() else sq.new_zeros(()))
    return torch.stack(losses)


@torch.no_grad()
def rollout_proxy_gains(
    model,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    orders: torch.Tensor,
    state_step: int,
    candidate_positions: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """
    Approximate gain_i = L_default - L_reveal_first_i.

    L_default follows the fixed reference order for H cosine-schedule steps.
    L_reveal_first_i first reveals candidate i, then uses the same reference
    continuation. Both losses are local decoder-feature MSE and are detached.
    """
    bsz, k = candidate_positions.shape
    coords = full_token_coords(model, mask.device)
    final_step = min(args.num_iter - 1, state_step + max(1, args.rollout_horizon))
    final_mask_len = mask_len_for_step(model.seq_len, final_step, args.num_iter)
    default_final_mask = mask_from_orders(orders, final_mask_len)

    default_state = decode_state(model, tokens, labels, default_final_mask)
    gains = torch.empty(bsz, k, device=mask.device, dtype=torch.float32)

    for b in range(bsz):
        tok_rep = tokens[b : b + 1].repeat(k, 1, 1)
        lab_rep = labels[b : b + 1].repeat(k)
        order_rep = orders[b : b + 1].repeat(k, 1)
        cand = candidate_positions[b]

        default_mask_rep = default_final_mask[b : b + 1].repeat(k, 1)
        default_state_rep = {
            "z": default_state["z"][b : b + 1].repeat(k, 1, 1),
            "gt_decoder_tokens": default_state["gt_decoder_tokens"][b : b + 1].repeat(k, 1, 1),
        }
        eval_mask = default_mask_rep.clone()
        l_default = local_loss_from_state(default_state_rep, eval_mask, cand, coords, args.local_radius)

        reveal_mask = mask_from_orders(order_rep, final_mask_len)
        false_src = torch.zeros(k, 1, device=mask.device, dtype=torch.bool)
        reveal_mask.scatter_(1, cand.unsqueeze(1), false_src)
        reveal_state = decode_state(model, tok_rep, lab_rep, reveal_mask)
        l_reveal = local_loss_from_state(reveal_state, eval_mask, cand, coords, args.local_radius)
        gains[b] = l_default - l_reveal
    return gains


@torch.no_grad()
def uncertainty_scores(model, z: torch.Tensor, candidate_positions: torch.Tensor, args: argparse.Namespace):
    candidate_z = torch.gather(z, 1, candidate_positions.unsqueeze(-1).expand(-1, -1, z.size(-1)))
    if args.uncertainty_mc_samples <= 1 or not z.is_cuda:
        # Cheap fallback for CPU/syntax environments: decoder feature energy acts
        # as a deterministic uncertainty proxy. CUDA runs use diffusion dispersion.
        energy = candidate_z.float().pow(2).mean(dim=-1)
        confidence = -energy
        entropy = -torch.log(energy.clamp_min(1e-8))
        return confidence, entropy, "decoder_feature_energy_fallback"

    bsz, k, dim = candidate_z.shape
    flat_z = candidate_z.reshape(bsz * k, dim)
    samples = []
    for _ in range(args.uncertainty_mc_samples):
        samples.append(model.diffloss.sample(flat_z, args.uncertainty_temperature, cfg=1.0).float())
    sample_stack = torch.stack(samples, dim=0)
    var = sample_stack.var(dim=0, unbiased=False).mean(dim=-1).view(bsz, k)
    confidence = -var
    entropy = -torch.log(var.clamp_min(1e-8))
    return confidence, entropy, "diffusion_sample_dispersion"


def metrics_for_state(scores: np.ndarray, gains: np.ndarray, k: int) -> Dict[str, Optional[float]]:
    return {
        "spearman": spearman_corr(scores, gains),
        "kendall_tau": kendall_tau(scores, gains),
        "topk_overlap": topk_overlap(scores, gains, k),
        "ndcg_at_k": ndcg_at_k(scores, gains, k),
        "regret": regret(scores, gains, k),
    }


def aggregate(rows: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    return {
        "spearman": nanmean_or_none(r["spearman"] for r in rows),
        "kendall_tau": nanmean_or_none(r["kendall_tau"] for r in rows),
        "topk_overlap": nanmean_or_none(r["topk_overlap"] for r in rows),
        "ndcg_at_k": nanmean_or_none(r["ndcg_at_k"] for r in rows),
        "regret": nanmean_or_none(r["regret"] for r in rows),
    }


def fmt(x: Optional[float]) -> str:
    return "NA" if x is None else "{:.4f}".format(x)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["row", "spearman", "kendall_tau", "topk_overlap", "ndcg_at_k", "regret"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, ckpt_path, state_key = build_model(args, device)

    vae = None
    if not args.use_cached:
        vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path)
        vae.to(device).eval()
        for p in vae.parameters():
            p.requires_grad = False

    dataset = build_dataset(args)
    indices = list(range(min(args.num_eval_images, len(dataset))))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.eval_bsz,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    metric_rows = {key: [] for _, key in ROW_ORDER}
    raw_scores = {key: [] for _, key in ROW_ORDER}
    raw_gains = []
    uncertainty_source = None
    state_count = 0

    max_step_default = args.num_iter - max(1, args.rollout_horizon) - 1
    if max_step_default < 0:
        raise ValueError("--num_iter must be larger than --rollout_horizon for reference-state evaluation")
    max_step = max_step_default if args.state_step_max < 0 else args.state_step_max
    max_step = max(args.state_step_min, min(max_step, max_step_default))
    if args.state_step_min > max_step:
        raise ValueError("--state_step_min={} is larger than valid max step {}".format(args.state_step_min, max_step))

    with torch.no_grad():
        image_offset = 0
        for batch in loader:
            tokens, labels = batch_to_latent_tokens(batch, vae, model, args, device)
            bsz = tokens.size(0)
            for _ in range(args.num_states_per_image):
                state_step = random.randint(args.state_step_min, max_step)
                orders = make_orders(bsz, model.seq_len, device)
                mask = mask_from_orders(orders, mask_len_for_step(model.seq_len, state_step, args.num_iter))
                state = decode_state(model, tokens, labels, mask)

                candidate_indices = build_candidate_subset(
                    state["planner_scores_masked"],
                    args.candidate_pool_size,
                    masked_coords=state["masked_coords"],
                    selection_mode=args.candidate_selection_mode,
                )
                candidate_positions = torch.gather(state["masked_positions"], 1, candidate_indices)
                gains = rollout_proxy_gains(model, tokens, labels, mask, orders, state_step, candidate_positions, args)

                planner_scores = torch.gather(state["planner_scores_masked"], 1, candidate_indices).float()
                current_loss_scores = torch.gather(state["per_token_loss"], 1, candidate_positions).float()
                confidence_scores, entropy_scores, source = uncertainty_scores(model, state["z"], candidate_positions, args)
                uncertainty_source = source
                random_scores = torch.rand_like(planner_scores)

                score_tensors = {
                    "random": random_scores,
                    "confidence": confidence_scores,
                    "entropy": entropy_scores,
                    "current_loss": current_loss_scores,
                    "planner": planner_scores,
                    "oracle": gains,
                }

                gains_np = gains.detach().cpu().numpy()
                for b in range(bsz):
                    oracle = gains_np[b]
                    raw_gains.append(oracle)
                    for _, key in ROW_ORDER:
                        scores_np = score_tensors[key][b].detach().cpu().numpy()
                        metric_rows[key].append(metrics_for_state(scores_np, oracle, args.topk_eval))
                        raw_scores[key].append(scores_np)
                state_count += bsz

            image_offset += bsz
            print("[SurrogateValidity] processed images={} states={}".format(image_offset, state_count), flush=True)

    summary_rows = []
    for label, key in ROW_ORDER:
        agg = aggregate(metric_rows[key])
        row = {"row": label}
        row.update(agg)
        summary_rows.append(row)

    csv_path = output_dir / "surrogate_validity.csv"
    json_path = output_dir / "surrogate_validity.json"
    raw_path = output_dir / "surrogate_validity_raw.npz"
    write_csv(csv_path, summary_rows)

    payload = {
        "metadata": {
            "checkpoint": str(ckpt_path),
            "checkpoint_state_key": state_key,
            "model": args.model,
            "data_path": args.data_path,
            "num_eval_images": len(indices),
            "num_states": state_count,
            "eval_bsz": args.eval_bsz,
            "candidate_pool_size": args.candidate_pool_size,
            "num_states_per_image": args.num_states_per_image,
            "rollout_horizon": args.rollout_horizon,
            "topk_eval": args.topk_eval,
            "candidate_selection_mode": args.candidate_selection_mode,
            "num_iter": args.num_iter,
            "local_radius": args.local_radius,
            "uncertainty_source": uncertainty_source,
            "ranking_direction": "larger score means higher priority / more useful",
            "approximations": [
                "Rollout proxy is teacher-forced in latent token space.",
                "Reference continuation uses cosine schedule plus fixed random order.",
                "Local future loss is decoder-feature MSE in a small neighborhood.",
                "Confidence and entropy use diffusion-sample dispersion on CUDA; CPU falls back to decoder feature energy.",
                "Current Loss uses positive current decoder-feature MSE, so higher-loss tokens rank earlier.",
            ],
        },
        "summary": summary_rows,
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "raw_npz": str(raw_path) if args.save_raw else None,
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if args.save_raw:
        np.savez_compressed(
            raw_path,
            gains=np.asarray(raw_gains, dtype=np.float32),
            random=np.asarray(raw_scores["random"], dtype=np.float32),
            confidence=np.asarray(raw_scores["confidence"], dtype=np.float32),
            entropy=np.asarray(raw_scores["entropy"], dtype=np.float32),
            current_loss=np.asarray(raw_scores["current_loss"], dtype=np.float32),
            planner=np.asarray(raw_scores["planner"], dtype=np.float32),
            oracle=np.asarray(raw_scores["oracle"], dtype=np.float32),
        )

    print("\nP0-3 Surrogate Validity Table")
    print("{:<28} {:>11} {:>13} {:>14} {:>10} {:>10}".format(
        "Row", "Spearman", "Kendall Tau", "Top-k Overlap", "NDCG@k", "Regret"
    ))
    print("-" * 92)
    for row in summary_rows:
        print("{:<28} {:>11} {:>13} {:>14} {:>10} {:>10}".format(
            row["row"],
            fmt(row["spearman"]),
            fmt(row["kendall_tau"]),
            fmt(row["topk_overlap"]),
            fmt(row["ndcg_at_k"]),
            fmt(row["regret"]),
        ))
    print("\n[DONE] Wrote {}".format(csv_path))
    print("[DONE] Wrote {}".format(json_path))
    if args.save_raw:
        print("[DONE] Wrote {}".format(raw_path))


if __name__ == "__main__":
    main()
