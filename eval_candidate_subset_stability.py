#!/usr/bin/env python3
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
from models.vae import AutoencoderKL
from util.crop import center_crop_arr
from util.revealmar_utils import build_candidate_subset
from util.surrogate_eval_utils import kendall_tau, ndcg_at_k, spearman_corr, topk_overlap


def parse_args():
    parser = argparse.ArgumentParser("Candidate subset stability evaluator")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="revealmar_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_eval_images", default=16, type=int)
    parser.add_argument("--eval_bsz", default=4, type=int)
    parser.add_argument("--num_states_per_image", default=1, type=int)
    parser.add_argument("--num_iter", default=64, type=int)
    parser.add_argument("--state_step_min", default=0, type=int)
    parser.add_argument("--state_step_max", default=16, type=int)
    parser.add_argument("--candidate_pool_size", default=8, type=int)
    parser.add_argument("--candidate_modes", default="topk,random,uncertainty,spatial,mixed")
    parser.add_argument("--candidate_random_ratio", default=0.25, type=float)
    parser.add_argument("--candidate_uncertainty_ratio", default=0.50, type=float)
    parser.add_argument("--candidate_spatial_ratio", default=0.25, type=float)
    parser.add_argument("--candidate_subset_seed", default=123, type=int)
    parser.add_argument("--local_radius", default=1, type=int)
    parser.add_argument("--rollout_horizon", default=1, type=int)
    parser.add_argument("--pseudo_target_type", default="mixed_reveal")
    parser.add_argument("--cfg", default=1.0, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--vae_embed_dim", default=16, type=int)
    parser.add_argument("--vae_stride", default=16, type=int)
    parser.add_argument("--patch_size", default=1, type=int)
    parser.add_argument("--class_num", default=1000, type=int)
    parser.add_argument("--diffloss_d", default=6, type=int)
    parser.add_argument("--diffloss_w", default=1024, type=int)
    parser.add_argument("--num_sampling_steps", default="100")
    parser.add_argument("--diffusion_batch_mul", default=1, type=int)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(resume):
    path = Path(resume)
    if path.is_dir():
        path = path / "checkpoint-last.pth"
    if not path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    return path


def build_model(args, device):
    model = revealmar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        class_num=args.class_num,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        candidate_pool_size=args.candidate_pool_size,
        pseudo_target_type=args.pseudo_target_type,
        sampling_policy="planner",
        candidate_selection_mode="topk",
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


def build_dataset(args):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    split = "val" if os.path.isdir(os.path.join(args.data_path, "val")) else "train"
    return datasets.ImageFolder(os.path.join(args.data_path, split), transform=transform)


def mask_len_for_step(seq_len, step, num_iter):
    if step <= 0:
        return seq_len
    ratio = math.cos(math.pi / 2.0 * float(step) / float(num_iter))
    return max(1, min(seq_len - 1, int(math.floor(seq_len * ratio))))


def make_orders(bsz, seq_len, device):
    return torch.stack([torch.randperm(seq_len, device=device) for _ in range(bsz)], dim=0)


def mask_from_orders(orders, mask_len):
    bsz, seq_len = orders.shape
    mask = torch.zeros(bsz, seq_len, device=orders.device, dtype=torch.bool)
    mask.scatter_(1, orders[:, :mask_len], torch.ones(bsz, mask_len, device=orders.device, dtype=torch.bool))
    return mask


def full_coords(model, device):
    y, x = torch.meshgrid(torch.arange(model.seq_h, device=device), torch.arange(model.seq_w, device=device), indexing="ij")
    return torch.stack([y.reshape(-1), x.reshape(-1)], dim=-1)


@torch.no_grad()
def decode_state(model, tokens, labels, mask):
    emb = model.class_emb(labels)
    x = model.forward_mae_encoder(tokens, mask.to(tokens.dtype), emb)
    z = model.forward_mae_decoder(x, mask.to(tokens.dtype))
    scores, positions, coords = model._compute_masked_planner_scores(z, mask.to(tokens.dtype))
    gt = model.decoder_embed(model.z_proj_ln(model.z_proj(tokens))) + model.diffusion_pos_embed_learned
    return {"z": z, "scores": scores, "positions": positions, "coords": coords, "gt": gt}


def local_loss(state, eval_mask, positions, coords_all, radius):
    sq = (state["z"].float() - state["gt"].float()).pow(2).mean(dim=-1)
    vals = []
    for i in range(state["z"].size(0)):
        pos = int(positions[i].item())
        dist = torch.max(torch.abs(coords_all - coords_all[pos]), dim=-1).values
        local = dist <= radius
        cur = eval_mask[i] & local
        if not cur.any():
            cur = eval_mask[i]
        vals.append(sq[i, cur].mean() if cur.any() else sq.new_zeros(()))
    return torch.stack(vals)


@torch.no_grad()
def proxy_gains(model, tokens, labels, orders, state_step, candidate_positions, args):
    bsz, k = candidate_positions.shape
    coords_all = full_coords(model, tokens.device)
    final_step = min(args.num_iter - 1, state_step + max(1, args.rollout_horizon))
    final_mask = mask_from_orders(orders, mask_len_for_step(model.seq_len, final_step, args.num_iter))
    default_state = decode_state(model, tokens, labels, final_mask)
    gains = torch.empty(bsz, k, device=tokens.device)
    false_src = torch.zeros(1, 1, device=tokens.device, dtype=torch.bool)
    for b in range(bsz):
        cand = candidate_positions[b]
        default_rep = {
            "z": default_state["z"][b:b + 1].repeat(k, 1, 1),
            "gt": default_state["gt"][b:b + 1].repeat(k, 1, 1),
        }
        eval_mask = final_mask[b:b + 1].repeat(k, 1)
        l_default = local_loss(default_rep, eval_mask, cand, coords_all, args.local_radius)
        for j in range(k):
            reveal_mask = final_mask[b:b + 1].clone()
            reveal_mask.scatter_(1, candidate_positions[b, j:j + 1].unsqueeze(1), false_src)
            reveal_state = decode_state(model, tokens[b:b + 1], labels[b:b + 1], reveal_mask)
            reveal_loss = local_loss(reveal_state, eval_mask[j:j + 1], candidate_positions[b, j:j + 1], coords_all, args.local_radius)[0]
            gains[b, j] = l_default[j] - reveal_loss
    return torch.nan_to_num(gains.float(), nan=0.0, posinf=0.0, neginf=0.0)


def pairwise_distance(coords):
    if len(coords) < 2:
        return 0.0
    vals = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            vals.append(math.dist(coords[i], coords[j]))
    return sum(vals) / len(vals) if vals else 0.0


def coverage_stats(indices, coords, masked_count, model):
    vals = indices.tolist()
    unique = len(set(vals))
    coords_np = coords.detach().cpu().numpy().tolist()
    if coords_np:
        rows = [c[0] for c in coords_np]
        cols = [c[1] for c in coords_np]
        spatial_spread = ((max(rows) - min(rows)) + (max(cols) - min(cols))) / 2.0
        center = ((model.seq_h - 1) / 2.0, (model.seq_w - 1) / 2.0)
        max_dist = math.dist((0, 0), center) or 1.0
        norm_dists = [math.dist(c, center) / max_dist for c in coords_np]
        center_bias = 1.0 - sum(norm_dists) / len(norm_dists)
        edge_bias = sum(norm_dists) / len(norm_dists)
    else:
        spatial_spread = center_bias = edge_bias = 0.0
    return {
        "spatial_spread": spatial_spread,
        "pairwise_dist": pairwise_distance(coords_np),
        "center_bias": center_bias,
        "edge_bias": edge_bias,
        "duplicate_rate": 1.0 - unique / float(max(1, len(vals))),
        "masked_coverage_fraction": unique / float(max(1, masked_count)),
    }


def nanmean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else ""


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.candidate_modes.split(",") if m.strip()]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(args, device)
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).to(device).eval()
    dataset = build_dataset(args)
    loader = DataLoader(Subset(dataset, list(range(min(args.num_eval_images, len(dataset))))), batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers)

    buckets = {m: [] for m in modes}
    details = []
    max_step = min(args.state_step_max, args.num_iter - max(1, args.rollout_horizon) - 1)
    max_step = max(args.state_step_min, max_step)
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[:2]
            labels = labels.to(device)
            tokens = model.patchify(vae.encode(images.to(device)).mode().mul(0.2325))
            bsz = tokens.size(0)
            for _ in range(args.num_states_per_image):
                step = random.randint(args.state_step_min, max_step)
                orders = make_orders(bsz, model.seq_len, device)
                mask = mask_from_orders(orders, mask_len_for_step(model.seq_len, step, args.num_iter))
                state = decode_state(model, tokens, labels, mask)
                per_mode_indices = {}
                per_mode_scores = {}
                per_mode_gains = {}
                for mode in modes:
                    candidate_indices = build_candidate_subset(
                        state["scores"], args.candidate_pool_size, masked_coords=state["coords"],
                        selection_mode=mode, random_ratio=args.candidate_random_ratio,
                        uncertainty_ratio=args.candidate_uncertainty_ratio,
                        spatial_ratio=args.candidate_spatial_ratio, subset_seed=args.candidate_subset_seed,
                    )
                    candidate_positions = torch.gather(state["positions"], 1, candidate_indices)
                    gains = proxy_gains(model, tokens, labels, orders, step, candidate_positions, args)
                    scores = torch.gather(state["scores"], 1, candidate_indices)
                    per_mode_indices[mode] = candidate_indices
                    per_mode_scores[mode] = scores
                    per_mode_gains[mode] = gains
                mixed_sets = [set(per_mode_indices["mixed"][b].detach().cpu().tolist()) for b in range(bsz)] if "mixed" in per_mode_indices else None
                topk_sets = [set(per_mode_indices["topk"][b].detach().cpu().tolist()) for b in range(bsz)] if "topk" in per_mode_indices else None
                for mode in modes:
                    for b in range(bsz):
                        scores_np = per_mode_scores[mode][b].detach().cpu().numpy()
                        gains_np = per_mode_gains[mode][b].detach().cpu().numpy()
                        k_eval = min(3, len(scores_np))
                        coords = state["coords"][b, per_mode_indices[mode][b]]
                        stats = coverage_stats(per_mode_indices[mode][b].detach().cpu(), coords, state["scores"].size(1), model)
                        idx_set = set(per_mode_indices[mode][b].detach().cpu().tolist())
                        overlap_mixed = "" if mixed_sets is None else len(idx_set & mixed_sets[b]) / float(max(1, len(idx_set)))
                        overlap_topk = "" if topk_sets is None else len(idx_set & topk_sets[b]) / float(max(1, len(idx_set)))
                        row = {
                            "mode": mode,
                            "spearman": spearman_corr(scores_np, gains_np),
                            "kendall_tau": kendall_tau(scores_np, gains_np),
                            "topk_overlap": topk_overlap(scores_np, gains_np, k_eval),
                            "ndcg_at_k": ndcg_at_k(scores_np, gains_np, k_eval),
                            "overlap_with_mixed": overlap_mixed,
                            "overlap_with_topk": overlap_topk,
                            "status": "ok",
                        }
                        row.update(stats)
                        buckets[mode].append(row)
                        details.append(row)
            print("[CandidateSubset] processed batch", flush=True)

    rows = []
    fields = ["mode", "spearman", "kendall_tau", "topk_overlap", "ndcg_at_k", "spatial_spread", "pairwise_dist", "center_bias", "edge_bias", "duplicate_rate", "masked_coverage_fraction", "overlap_with_mixed", "overlap_with_topk", "status"]
    for mode in modes:
        vals = buckets[mode]
        row = {"mode": mode, "status": "ok" if vals else "no_metrics"}
        for key in fields:
            if key not in ("mode", "status"):
                row[key] = nanmean([v.get(key) for v in vals if v.get(key) != ""])
        rows.append(row)

    with (output_dir / "candidate_subset_stability.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "candidate_subset_stability.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": vars(args), "results": rows}, f, indent=2)
    with (output_dir / "candidate_subset_stability_details.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    print("| mode | spearman | kendall_tau | topk_overlap | ndcg_at_k | spatial_spread | pairwise_dist | duplicate_rate | overlap_with_mixed | status |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print("| {mode} | {spearman} | {kendall_tau} | {topk_overlap} | {ndcg_at_k} | {spatial_spread} | {pairwise_dist} | {duplicate_rate} | {overlap_with_mixed} | {status} |".format(**{k: r.get(k, "") for k in fields}))


if __name__ == "__main__":
    main()
