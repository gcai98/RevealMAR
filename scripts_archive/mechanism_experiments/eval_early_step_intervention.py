#!/usr/bin/env python3
import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser("Early-step intervention evaluator wrapper")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="revealmar_base")
    parser.add_argument("--num_images", default=1000, type=int)
    parser.add_argument("--class_num", default=1000, type=int)
    parser.add_argument("--eval_bsz", default=32, type=int)
    parser.add_argument("--num_iter", default=64, type=int)
    parser.add_argument("--num_sampling_steps", default="100")
    parser.add_argument("--cfg", default=1.0, type=float)
    parser.add_argument("--cfg_schedule", default="linear")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--base_policy", default="planner")
    parser.add_argument("--intervention_policy", default="random", choices=["none", "random", "confidence", "entropy"])
    parser.add_argument("--intervention_steps", default=8, type=int)
    parser.add_argument("--budget_mode", default="hard")
    parser.add_argument("--pseudo_target_type", default="mixed_reveal")
    parser.add_argument("--candidate_pool_size", default=8, type=int)
    parser.add_argument("--log_planner_sampling_steps", default=8, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    intervention_policy = args.intervention_policy
    intervention_steps = args.intervention_steps if intervention_policy != "none" else 0
    cmd = [
        sys.executable, "main_revealmar.py",
        "--evaluate",
        "--model", args.model,
        "--img_size", "256",
        "--vae_path", args.vae_path,
        "--vae_embed_dim", "16",
        "--vae_stride", "16",
        "--patch_size", "1",
        "--data_path", args.data_path,
        "--resume", args.resume,
        "--output_dir", args.output_dir,
        "--class_num", str(args.class_num),
        "--diffloss_d", "6",
        "--diffloss_w", "1024",
        "--diffusion_batch_mul", "1",
        "--num_images", str(args.num_images),
        "--eval_bsz", str(args.eval_bsz),
        "--num_iter", str(args.num_iter),
        "--num_sampling_steps", str(args.num_sampling_steps),
        "--cfg", str(args.cfg),
        "--cfg_schedule", args.cfg_schedule,
        "--temperature", str(args.temperature),
        "--num_workers", str(args.num_workers),
        "--sampling_policy", args.base_policy,
        "--pseudo_target_type", args.pseudo_target_type,
        "--budget_mode", args.budget_mode,
        "--candidate_pool_size", str(args.candidate_pool_size),
        "--early_intervention_policy", intervention_policy,
        "--early_intervention_steps", str(intervention_steps),
        "--log_planner_sampling_debug",
        "--log_planner_sampling_steps", str(args.log_planner_sampling_steps),
        "--dist_url", "env://",
    ]
    print(" ".join('"{}"'.format(x) if " " in x else x for x in cmd), flush=True)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
