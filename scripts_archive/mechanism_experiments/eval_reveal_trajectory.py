#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from models import revealmar


def parse_args():
    parser = argparse.ArgumentParser("Reveal trajectory diagnostics")
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--vae_path", default="", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--output_dir", default="./output_dir/reveal_trajectory", type=str)
    parser.add_argument("--model", default="revealmar_base", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num_images", default=16, type=int)
    parser.add_argument("--class_num", default=16, type=int)
    parser.add_argument("--eval_bsz", default=4, type=int)
    parser.add_argument("--num_iter", default=64, type=int)
    parser.add_argument("--num_sampling_steps", default="100", type=str)
    parser.add_argument("--cfg", default=1.0, type=float)
    parser.add_argument("--cfg_schedule", default="linear", type=str)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--sampling_policy", default="planner", choices=["baseline", "planner", "random", "confidence", "entropy"])
    parser.add_argument("--budget_mode", default="hard", choices=["hard", "soft"])
    parser.add_argument("--pseudo_target_type", default="mixed_reveal", type=str)
    parser.add_argument("--candidate_pool_size", default=8, type=int)
    parser.add_argument("--save_heatmaps", action="store_true")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--max_visualizations", default=16, type=int)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--vae_embed_dim", default=16, type=int)
    parser.add_argument("--vae_stride", default=16, type=int)
    parser.add_argument("--patch_size", default=1, type=int)
    parser.add_argument("--class_num_model", default=1000, type=int)
    parser.add_argument("--diffloss_d", default=6, type=int)
    parser.add_argument("--diffloss_w", default=1024, type=int)
    parser.add_argument("--diffusion_batch_mul", default=1, type=int)
    return parser.parse_args()


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
        class_num=args.class_num_model,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        candidate_pool_size=args.candidate_pool_size,
        pseudo_target_type=args.pseudo_target_type,
        sampling_policy=args.sampling_policy,
        budget_mode=args.budget_mode,
    )
    checkpoint = torch.load(resolve_checkpoint(args.resume), map_location="cpu", weights_only=False)
    state_key = "model_ema" if checkpoint.get("model_ema") is not None else "model"
    result = model.load_state_dict(checkpoint[state_key], strict=False)
    if result.missing_keys:
        print("[WARN] Missing model keys: {}".format(result.missing_keys))
    if result.unexpected_keys:
        print("[WARN] Unexpected model keys: {}".format(result.unexpected_keys))
    model.to(device).eval()
    model._revealmar_sampling_debug = False
    return model


def mean(values):
    return sum(values) / len(values) if values else ""


def pairwise_distance(coords):
    if len(coords) < 2:
        return 0.0
    vals = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            vals.append(math.dist(coords[i], coords[j]))
    return mean(vals)


def summarize(details, model, args):
    selected_counts = []
    early_counts = []
    spread_values = []
    pairwise_values = []
    center_bias_values = []
    edge_bias_values = []
    concentrations = []
    score_stds = []
    center = ((model.seq_h - 1) / 2.0, (model.seq_w - 1) / 2.0)
    max_center_dist = math.dist((0, 0), center) or 1.0

    for image in details:
        for step in image["steps"]:
            count = len(step["selected_coords"])
            selected_counts.append(count)
            if step["step"] < 8:
                early_counts.append(count)
            coords = step["selected_coords"]
            if coords:
                dists = [math.dist(coord, center) / max_center_dist for coord in coords]
                center_bias_values.append(1.0 - mean(dists))
                edge_bias_values.append(mean(dists))
                rows = [coord[0] for coord in coords]
                cols = [coord[1] for coord in coords]
                spread_values.append((max(rows) - min(rows) + max(cols) - min(cols)) / 2.0 if len(coords) > 1 else 0.0)
                pairwise_values.append(pairwise_distance(coords))
            if step.get("concentration") != "":
                concentrations.append(step["concentration"])
            if step.get("score_std") != "":
                score_stds.append(step["score_std"])

    return {
        "policy": args.sampling_policy,
        "budget_mode": args.budget_mode,
        "num_images": len(details),
        "num_iter": args.num_iter,
        "mean_selected_per_step": mean(selected_counts),
        "max_selected_per_step": max(selected_counts) if selected_counts else "",
        "early_selected_mean": mean(early_counts),
        "center_bias_score": mean(center_bias_values),
        "edge_bias_score": mean(edge_bias_values),
        "spatial_spread_mean": mean(spread_values),
        "pairwise_distance_mean": mean(pairwise_values),
        "concentration_mean": mean(concentrations),
        "score_std_mean": mean(score_stds),
    }


def save_heatmap(path, heatmap, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.imshow(heatmap, cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_summary_csv(path, row):
    fields = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(args, device)

    details = []
    heat_all = torch.zeros(model.seq_h, model.seq_w)
    heat_early = torch.zeros(model.seq_h, model.seq_w)
    heat_late = torch.zeros(model.seq_h, model.seq_w)
    generated = 0

    with torch.no_grad():
        while generated < args.num_images:
            bsz = min(args.eval_bsz, args.num_images - generated)
            labels = torch.arange(generated, generated + bsz, device=device) % max(1, args.class_num)
            model.sample_tokens(
                bsz=bsz,
                num_iter=args.num_iter,
                cfg=args.cfg,
                cfg_schedule=args.cfg_schedule,
                labels=labels.long(),
                temperature=args.temperature,
            )
            coords_traj = model.latest_sampling_selected_coords_trajectory or []
            idx_traj = model.latest_sampling_selected_indices_trajectory or []
            budget_traj = model.latest_sampling_budget_trajectory or []
            selected_traj = model.latest_sampling_selected_trajectory or []
            conc_traj = model.latest_sampling_entropy_trajectory or []
            score_stats_traj = model.latest_sampling_score_stats_trajectory or []
            for b in range(bsz):
                image_steps = []
                for step, coords_per_batch in enumerate(coords_traj):
                    coords = coords_per_batch[b] if b < len(coords_per_batch) else []
                    indices = idx_traj[step][b] if step < len(idx_traj) and b < len(idx_traj[step]) else []
                    for row, col in coords:
                        heat_all[row, col] += 1
                        if step < 8:
                            heat_early[row, col] += 1
                        else:
                            heat_late[row, col] += 1
                    stats = score_stats_traj[step] if step < len(score_stats_traj) else {}
                    image_steps.append({
                        "step": step,
                        "selected_indices": indices,
                        "selected_coords": coords,
                        "selected_count": selected_traj[step] if step < len(selected_traj) else len(coords),
                        "budget": budget_traj[step] if step < len(budget_traj) else "",
                        "concentration": conc_traj[step] if step < len(conc_traj) else "",
                        "score_std": stats.get("score_std", ""),
                        "policy": args.sampling_policy,
                        "budget_mode": args.budget_mode,
                    })
                details.append({"image_index": generated + b, "steps": image_steps})
            generated += bsz
            print("[RevealTrajectory] generated={}".format(generated), flush=True)

    summary = summarize(details, model, args)
    write_summary_csv(output_dir / "reveal_trajectory_summary.csv", summary)
    with (output_dir / "reveal_trajectory_details.json").open("w", encoding="utf-8") as f:
        json.dump(details if args.save_json else details[:args.max_visualizations], f, indent=2)
    with (output_dir / "reveal_trajectory_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.save_heatmaps:
        save_heatmap(output_dir / "selected_frequency_all_steps.png", heat_all.numpy(), "All steps")
        save_heatmap(output_dir / "selected_frequency_early_steps.png", heat_early.numpy(), "Early steps")
        save_heatmap(output_dir / "selected_frequency_late_steps.png", heat_late.numpy(), "Late steps")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
