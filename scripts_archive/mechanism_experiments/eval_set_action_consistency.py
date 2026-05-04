#!/usr/bin/env python3
"""
Smoke-scale set-action consistency evaluator for PlanMAR-S.

This supports the approximation claim that independent token utilities followed
by top-k selection are a tractable proxy for set-action selection. The evaluator
compares deployed individual_topk selection against small-pool alternatives.

Important approximation note:
greedy_marginal here is a local proxy baseline, not a true global optimal set
search. It uses per-token rollout-proxy gains plus a spatial redundancy penalty
to approximate diminishing returns on a small candidate pool.
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

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


METHODS = [
    "individual_topk",
    "greedy_marginal",
    "pairwise_deredundancy",
    "random_set",
    "confidence_topk",
    "entropy_topk",
]


def parse_args():
    parser = argparse.ArgumentParser("Set-action consistency evaluator")
    parser.add_argument("--data_path", default="./data/imagenet", type=str)
    parser.add_argument("--vae_path", default="pretrained_models/vae/kl16.ckpt", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--output_dir", default="./output_dir/set_action_consistency", type=str)
    parser.add_argument("--model", default="revealmar_base", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

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

    parser.add_argument("--num_eval_images", default=16, type=int)
    parser.add_argument("--eval_bsz", default=4, type=int)
    parser.add_argument("--num_states_per_image", default=1, type=int)
    parser.add_argument("--num_iter", default=64, type=int)
    parser.add_argument("--state_step_min", default=0, type=int)
    parser.add_argument("--state_step_max", default=16, type=int)
    parser.add_argument("--candidate_pool_size", default=8, type=int)
    parser.add_argument("--set_k", default=4, type=int)
    parser.add_argument("--local_radius", default=1, type=int)
    parser.add_argument("--rollout_horizon", default=1, type=int)
    parser.add_argument("--deredundancy_lambda", default=0.2, type=float)
    parser.add_argument(
        "--pseudo_target_type",
        default="mixed_reveal",
        choices=[
            "none", "gt_reveal", "pred_reveal", "mixed_reveal",
            "ref_gt_reveal", "ref_pred_reveal", "ref_mixed_reveal",
        ],
    )
    parser.add_argument("--ref_target_horizon", default=1, type=int)
    parser.add_argument("--ref_target_local_radius", default=1, type=int)
    parser.add_argument("--ref_target_mix_alpha", default=0.5, type=float)
    parser.add_argument("--cfg", default=1.0, type=float)
    parser.add_argument("--latent_mode", default="mode", choices=["mode", "sample"])
    parser.add_argument("--use_cached", action="store_true", dest="use_cached")
    parser.add_argument("--cached_path", default="", type=str)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(resume):
    if not resume:
        raise ValueError("Provide --resume")
    path = Path(resume)
    if path.is_dir():
        path = path / "checkpoint-last.pth"
    if not path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    return path


def build_dataset(args):
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
    split = "val" if os.path.isdir(os.path.join(args.data_path, "val")) else "train"
    return datasets.ImageFolder(os.path.join(args.data_path, split), transform=transform)


def build_model(args, device):
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
        candidate_pool_size=args.candidate_pool_size,
        pseudo_target_type=args.pseudo_target_type,
        ref_target_horizon=args.ref_target_horizon,
        ref_target_local_radius=args.ref_target_local_radius,
        ref_target_mix_alpha=args.ref_target_mix_alpha,
        sampling_policy="planner",
        budget_mode="soft",
    )
    checkpoint = torch.load(resolve_checkpoint(args.resume), map_location="cpu", weights_only=False)
    state_key = "model_ema" if checkpoint.get("model_ema") is not None else "model"
    result = model.load_state_dict(checkpoint[state_key], strict=False)
    if result.missing_keys:
        print("[WARN] Missing model keys: {}".format(result.missing_keys))
    if result.unexpected_keys:
        print("[WARN] Unexpected model keys: {}".format(result.unexpected_keys))
    model.to(device).eval()
    return model


def batch_to_latent_tokens(batch, vae, model, args, device):
    samples, labels = batch[:2]
    labels = labels.to(device, non_blocking=True)
    if args.use_cached:
        posterior = DiagonalGaussianDistribution(samples.to(device, non_blocking=True))
    else:
        posterior = vae.encode(samples.to(device, non_blocking=True))
    latent_image = posterior.mode() if args.latent_mode == "mode" else posterior.sample()
    return model.patchify(latent_image.mul(0.2325)), labels


def make_orders(bsz, seq_len, device):
    return torch.stack([torch.randperm(seq_len, device=device) for _ in range(bsz)], dim=0)


def mask_len_for_step(seq_len, step, num_iter):
    if step <= 0:
        return seq_len
    ratio = math.cos(math.pi / 2.0 * float(step) / float(num_iter))
    return max(1, min(seq_len - 1, int(math.floor(seq_len * ratio))))


def mask_from_orders(orders, mask_len):
    bsz, seq_len = orders.shape
    mask = torch.zeros(bsz, seq_len, device=orders.device, dtype=torch.bool)
    src = torch.ones(bsz, mask_len, device=orders.device, dtype=torch.bool)
    mask.scatter_(1, orders[:, :mask_len], src)
    return mask


def full_token_coords(model, device):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(model.seq_h, device=device),
        torch.arange(model.seq_w, device=device),
        indexing="ij",
    )
    return torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)


@torch.no_grad()
def decode_state(model, tokens, labels, mask):
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


def local_loss_from_state(state, eval_mask, candidate_positions, coords, radius):
    sq = (state["z"].float() - state["gt_decoder_tokens"].float()).pow(2).mean(dim=-1)
    losses = []
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
def rollout_proxy_gains(model, tokens, labels, mask, orders, state_step, candidate_positions, args):
    """Approximate candidate utility as local loss reduction after reveal-first.

    This mirrors the surrogate-validity proxy: ground-truth reveal-first
    counterfactuals plus fixed cosine reference continuation. It is local and
    teacher-forced, not a full image-generation rollout.
    """
    bsz, cand_count = candidate_positions.shape
    coords = full_token_coords(model, mask.device)
    final_step = min(args.num_iter - 1, state_step + max(1, args.rollout_horizon))
    final_mask_len = mask_len_for_step(model.seq_len, final_step, args.num_iter)
    default_final_mask = mask_from_orders(orders, final_mask_len)
    default_state = decode_state(model, tokens, labels, default_final_mask)
    gains = torch.empty(bsz, cand_count, device=mask.device, dtype=torch.float32)

    for b in range(bsz):
        cand = candidate_positions[b]
        default_mask_rep = default_final_mask[b : b + 1].repeat(cand_count, 1)
        default_state_rep = {
            "z": default_state["z"][b : b + 1].repeat(cand_count, 1, 1),
            "gt_decoder_tokens": default_state["gt_decoder_tokens"][b : b + 1].repeat(cand_count, 1, 1),
        }
        l_default = local_loss_from_state(
            default_state_rep, default_mask_rep, cand, coords, args.local_radius
        )
        tok_one = tokens[b : b + 1]
        lab_one = labels[b : b + 1]
        order_one = orders[b : b + 1]
        false_src = torch.zeros(1, 1, device=mask.device, dtype=torch.bool)
        for j in range(cand_count):
            cand_one = candidate_positions[b, j : j + 1]
            reveal_mask_one = mask_from_orders(order_one, final_mask_len)
            reveal_mask_one.scatter_(1, cand_one.unsqueeze(1), false_src)
            reveal_state = decode_state(model, tok_one, lab_one, reveal_mask_one)
            reveal_loss = local_loss_from_state(
                reveal_state, default_mask_rep[j : j + 1], cand_one, coords, args.local_radius
            )[0]
            gains[b, j] = l_default[j] - reveal_loss
    return torch.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0)


def spatial_similarity(pos_a, pos_b, coords):
    ca = coords[pos_a].float()
    cb = coords[pos_b].float()
    cheb = torch.max(torch.abs(ca - cb)).item()
    return 1.0 / (1.0 + float(cheb))


def set_redundancy(selected, coords):
    selected = list(selected)
    if len(selected) <= 1:
        return 0.0
    sims = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            sims.append(spatial_similarity(selected[i], selected[j], coords))
    return float(sum(sims) / len(sims)) if sims else 0.0


def topk_indices(scores, k):
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    return [int(i) for i in order[:k]]


def greedy_marginal_indices(gains, positions, coords, k, lam):
    selected = []
    remaining = set(range(len(gains)))
    for _ in range(min(k, len(gains))):
        best_idx = None
        best_score = None
        for idx in sorted(remaining):
            redundancy = 0.0
            if selected:
                redundancy = max(spatial_similarity(int(positions[idx]), int(positions[j]), coords) for j in selected)
            marginal = float(gains[idx]) - float(lam) * redundancy
            if best_score is None or marginal > best_score:
                best_score = marginal
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def deredundancy_indices(scores, positions, coords, k, lam):
    selected = []
    remaining = set(range(len(scores)))
    for _ in range(min(k, len(scores))):
        best_idx = None
        best_score = None
        for idx in sorted(remaining):
            redundancy = 0.0
            if selected:
                redundancy = max(spatial_similarity(int(positions[idx]), int(positions[j]), coords) for j in selected)
            adjusted = float(scores[idx]) - float(lam) * redundancy
            if best_score is None or adjusted > best_score:
                best_score = adjusted
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def method_sets(planner_scores, gains, current_loss, positions, coords, k, lam):
    rng_order = list(range(len(gains)))
    random.shuffle(rng_order)
    # Optional continuous-latent uncertainty baselines: current local loss is a
    # cheap deterministic proxy here. Confidence selects low-loss tokens first;
    # entropy selects high-loss tokens first.
    confidence_scores = -np.asarray(current_loss, dtype=np.float64)
    entropy_scores = np.asarray(current_loss, dtype=np.float64)
    return {
        "individual_topk": topk_indices(planner_scores, k),
        "greedy_marginal": greedy_marginal_indices(gains, positions, coords, k, lam),
        "pairwise_deredundancy": deredundancy_indices(planner_scores, positions, coords, k, lam),
        "random_set": rng_order[:k],
        "confidence_topk": topk_indices(confidence_scores, k),
        "entropy_topk": topk_indices(entropy_scores, k),
    }


def update_aggregates(aggregates, details, method, selected, greedy, gains, positions, coords, set_k, pool_size):
    selected = list(selected)
    greedy = list(greedy)
    gain = float(np.sum(np.asarray(gains, dtype=np.float64)[selected])) if selected else 0.0
    greedy_gain = float(np.sum(np.asarray(gains, dtype=np.float64)[greedy])) if greedy else 0.0
    overlap = len(set(selected).intersection(greedy)) / float(max(1, min(set_k, len(gains))))
    redundancy = set_redundancy([int(positions[i]) for i in selected], coords)
    row = {
        "method": method,
        "set_gain": gain,
        "regret_vs_greedy": greedy_gain - gain,
        "overlap_with_greedy": overlap,
        "redundancy": redundancy,
        "set_k": set_k,
        "candidate_pool_size": pool_size,
    }
    aggregates[method].append(row)
    details.append(row)


def mean_or_empty(values):
    return "" if not values else sum(values) / len(values)


def aggregate_rows(aggregates, set_k, candidate_pool_size):
    rows = []
    for method in METHODS:
        vals = aggregates[method]
        rows.append(
            {
                "method": method,
                "mean_set_gain": mean_or_empty([v["set_gain"] for v in vals]),
                "mean_regret_vs_greedy": mean_or_empty([v["regret_vs_greedy"] for v in vals]),
                "mean_overlap_with_greedy": mean_or_empty([v["overlap_with_greedy"] for v in vals]),
                "mean_redundancy": mean_or_empty([v["redundancy"] for v in vals]),
                "num_states": len(vals),
                "set_k": set_k,
                "candidate_pool_size": candidate_pool_size,
            }
        )
    return rows


def write_csv(path, rows):
    fieldnames = [
        "method", "mean_set_gain", "mean_regret_vs_greedy",
        "mean_overlap_with_greedy", "mean_redundancy",
        "num_states", "set_k", "candidate_pool_size",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value == "" or value is None:
        return "NA"
    return "{:.4f}".format(float(value))


def print_table(rows):
    print("| method | set_gain | regret_vs_greedy | overlap_with_greedy | redundancy | num_states |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print("| {} | {} | {} | {} | {} | {} |".format(
            row["method"],
            fmt(row["mean_set_gain"]),
            fmt(row["mean_regret_vs_greedy"]),
            fmt(row["mean_overlap_with_greedy"]),
            fmt(row["mean_redundancy"]),
            row["num_states"],
        ))


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cfg != 1.0:
        print("[WARN] This evaluator uses teacher-forced proxy states; --cfg is recorded but not used.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(args, device)
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

    max_step_default = args.num_iter - max(1, args.rollout_horizon) - 1
    max_step = args.state_step_max if args.state_step_max >= 0 else max_step_default
    max_step = max(args.state_step_min, min(max_step, max_step_default))
    if args.state_step_min > max_step:
        raise ValueError("--state_step_min is larger than the valid state-step max")

    aggregates = {method: [] for method in METHODS}
    details = []
    coords = full_token_coords(model, device)
    processed = 0

    with torch.no_grad():
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
                    selection_mode="topk",
                )
                candidate_positions = torch.gather(state["masked_positions"], 1, candidate_indices)
                gains = rollout_proxy_gains(model, tokens, labels, mask, orders, state_step, candidate_positions, args)
                planner_scores = torch.gather(state["planner_scores_masked"], 1, candidate_indices).float()
                current_loss = torch.gather(state["per_token_loss"], 1, candidate_positions).float()

                for b in range(bsz):
                    gains_np = gains[b].detach().cpu().numpy()
                    scores_np = planner_scores[b].detach().cpu().numpy()
                    current_loss_np = current_loss[b].detach().cpu().numpy()
                    positions_np = candidate_positions[b].detach().cpu().numpy()
                    k = min(args.set_k, len(gains_np))
                    sets = method_sets(
                        scores_np,
                        gains_np,
                        current_loss_np,
                        positions_np,
                        coords,
                        k,
                        args.deredundancy_lambda,
                    )
                    greedy = sets["greedy_marginal"]
                    for method, selected in sets.items():
                        update_aggregates(
                            aggregates,
                            details,
                            method,
                            selected,
                            greedy,
                            gains_np,
                            positions_np,
                            coords,
                            k,
                            args.candidate_pool_size,
                        )
                processed += bsz
            print("[SetActionConsistency] processed_images={} states={}".format(processed, processed * args.num_states_per_image), flush=True)

    rows = aggregate_rows(aggregates, args.set_k, args.candidate_pool_size)
    csv_path = output_dir / "set_action_consistency.csv"
    json_path = output_dir / "set_action_consistency.json"
    details_path = output_dir / "set_action_consistency_details.json"
    write_csv(csv_path, rows)
    payload = {
        "metadata": vars(args),
        "approximation_note": (
            "greedy_marginal is a local proxy baseline using rollout-proxy token gains "
            "and spatial redundancy; it is not an exact global set optimum."
        ),
        "results": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with details_path.open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    print_table(rows)
    print("CSV saved to {}".format(csv_path))
    print("JSON saved to {}".format(json_path))
    print("Details saved to {}".format(details_path))


if __name__ == "__main__":
    main()
