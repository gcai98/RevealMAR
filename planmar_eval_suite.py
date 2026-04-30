#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlanMAR-S print-only evaluation suite.

Purpose:
- One-click run multiple evaluation settings.
- Image generation and metric computation are NOT reimplemented here.
- This script only calls:
      python main_revealmar.py --evaluate ...
  so generation, image saving, torch_fidelity metrics, and [METRICS_JSON]
  are all produced by your existing engine_mar.py logic.
- After each run, this script parses that run's log and prints a compact summary.
- No CSV and no figures are written.

Default runs:
- MAR-Baseline: sampling_policy=baseline, num_iter=64/128/256
- PlanMAR-S:    sampling_policy=planner,  num_iter=64/128/256
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PlanMAR-S print-only evaluation suite")

    parser.add_argument("--code_dir", type=str, default="/root/autodl-tmp/RevealMAR-revealmar-dev")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/imagenet/imagenet1k_imagefolder_full")
    parser.add_argument("--pretrain_root", type=str, default="/root/autodl-tmp/pretrained_models")

    parser.add_argument(
        "--planmar_ckpt",
        type=str,
        default="/root/autodl-tmp/outputs/revealmar_train1_eval1000_planner_bs64",
        help="Checkpoint directory for PlanMAR-S, containing checkpoint-last.pth.",
    )
    parser.add_argument(
        "--baseline_ckpt",
        type=str,
        default="/root/autodl-tmp/outputs/revealmar_train1_eval1000_planner_bs64",
        help=(
            "Checkpoint directory for baseline policy. "
            "Default uses the same trained checkpoint but baseline sampling."
        ),
    )

    parser.add_argument("--eval_root", type=str, default="/root/autodl-tmp/outputs/planmar_paper_eval_print_only")
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--eval_bsz", type=int, default=96)
    parser.add_argument("--num_iters", type=str, default="64,128,256")

    parser.add_argument("--model", type=str, default="revealmar_base")
    parser.add_argument("--diffloss_d", type=int, default=6)
    parser.add_argument("--diffloss_w", type=int, default=1024)
    parser.add_argument("--num_sampling_steps", type=str, default="100")
    parser.add_argument("--cfg", type=float, default=2.9)
    parser.add_argument("--cfg_schedule", type=str, default="linear")
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--pseudo_target_type", type=str, default="mixed_reveal")
    parser.add_argument("--planner_loss_weight", type=float, default=0.3)
    parser.add_argument("--candidate_pool_size", type=int, default=8)
    parser.add_argument("--mixed_policy_ratio", type=float, default=0.0)

    parser.add_argument(
        "--methods",
        type=str,
        default="baseline,planner",
        help="Comma-separated policies to evaluate. Current RevealMAR code supports baseline,planner.",
    )

    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")

    return parser.parse_args()


def ensure_exists(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{desc} does not exist: {path}")


def run_cmd(cmd: List[str], log_path: Path, cwd: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", errors="ignore") as f:
        f.write("[COMMAND] " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.wait()


def parse_float_or_none(x: Optional[object]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def parse_log(log_path: Path) -> Dict[str, Optional[float]]:
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""

    result: Dict[str, Optional[float]] = {
        "fid": None,
        "is": None,
        "kid": None,
        "kid_std": None,
        "precision": None,
        "recall": None,
        "total_gen_seconds_measured": None,
        "sec_per_image": None,
        "generated_images_timed": None,
    }

    # Preferred format from your modified engine_mar.py:
    # [METRICS_JSON] {"fid": ..., "is": ..., "kid": ..., "kid_std": ..., "precision": ..., "recall": ...}
    metric_json_re = re.compile(r"\[METRICS_JSON\]\s*(\{.*?\})")
    matches = metric_json_re.findall(text)
    if matches:
        try:
            m = json.loads(matches[-1])
            for k in ["fid", "is", "kid", "kid_std", "precision", "recall"]:
                result[k] = parse_float_or_none(m.get(k))
        except Exception as e:
            print(f"[WARN] Failed to parse METRICS_JSON from {log_path}: {e}")

    # Fallback: parse human-readable metric line.
    # Example:
    # FID: 1.234000, Inception Score: 10.123000, KID: ..., Precision: ..., Recall: ...
    fid_is_re = re.compile(
        r"FID:\s*([0-9eE\.\+\-]+)\s*,\s*Inception Score:\s*([0-9eE\.\+\-]+)"
    )
    fid_matches = fid_is_re.findall(text)
    if fid_matches and (result["fid"] is None or result["is"] is None):
        result["fid"] = float(fid_matches[-1][0])
        result["is"] = float(fid_matches[-1][1])

    def last_match(regex):
        vals = regex.findall(text)
        if not vals:
            return None
        val = vals[-1]
        return None if val == "NA" else parse_float_or_none(val)

    if result["kid"] is None:
        result["kid"] = last_match(re.compile(r"KID:\s*([0-9eE\.\+\-]+|NA)"))

    if result["kid_std"] is None:
        result["kid_std"] = last_match(
            re.compile(r"KID[_\s-]*std:\s*([0-9eE\.\+\-]+|NA)", re.IGNORECASE)
        )

    if result["precision"] is None:
        result["precision"] = last_match(re.compile(r"Precision:\s*([0-9eE\.\+\-]+|NA)"))

    if result["recall"] is None:
        result["recall"] = last_match(re.compile(r"Recall:\s*([0-9eE\.\+\-]+|NA)"))

    # Parse latency line printed inside engine_mar.evaluate:
    # Generating 1000 images takes xxx seconds, xxx sec per image
    time_re = re.compile(
        r"Generating\s+([0-9]+)\s+images\s+takes\s+([0-9eE\.\+\-]+)\s+seconds,\s+([0-9eE\.\+\-]+)\s+sec per image"
    )
    t_matches = time_re.findall(text)

    if t_matches:
        n, total_sec, sec_img = t_matches[-1]
        result["generated_images_timed"] = float(n)
        result["total_gen_seconds_measured"] = float(total_sec)
        result["sec_per_image"] = float(sec_img)

    return result


def log_has_metrics(log_path: Path) -> bool:
    return log_path.exists() and parse_log(log_path).get("fid") is not None


def format_float(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def print_result_summary(row: Dict[str, object]) -> None:
    print("\n" + "-" * 88)
    print("[RESULT]")
    print(f"method              : {row['method']}")
    print(f"sampling_policy     : {row['sampling_policy']}")
    print(f"checkpoint          : {row['checkpoint']}")
    print(f"num_iter            : {row['num_iter']}")
    print(f"num_images          : {row['num_images']}")
    print(f"eval_bsz            : {row['eval_bsz']}")
    print(f"cfg                 : {row['cfg']}")
    print(f"num_sampling_steps  : {row['num_sampling_steps']}")
    print(f"FID                 : {format_float(row.get('fid'))}")
    print(f"IS                  : {format_float(row.get('is'))}")
    print(f"KID                 : {format_float(row.get('kid'))}")
    print(f"KID_std             : {format_float(row.get('kid_std'))}")
    print(f"Precision           : {format_float(row.get('precision'))}")
    print(f"Recall              : {format_float(row.get('recall'))}")
    print(f"generated_images    : {format_float(row.get('generated_images_timed'), 0)}")
    print(f"total_gen_seconds   : {format_float(row.get('total_gen_seconds_measured'))}")
    print(f"sec_per_image       : {format_float(row.get('sec_per_image'))}")
    print(f"throughput img/sec  : {format_float(row.get('throughput_img_per_sec'))}")
    print(f"status              : {row['status']}")
    print(f"log_path            : {row['log_path']}")
    print(f"output_dir          : {row['output_dir']}")
    print("-" * 88 + "\n")


def print_final_table(rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 132)
    print("[FINAL SUMMARY TABLE]")
    print("=" * 132)

    header = (
        f"{'Method':<16} "
        f"{'Policy':<10} "
        f"{'Iter':>6} "
        f"{'Images':>8} "
        f"{'FID':>10} "
        f"{'IS':>10} "
        f"{'KID':>10} "
        f"{'Prec':>10} "
        f"{'Rec':>10} "
        f"{'sec/img':>10} "
        f"{'Status':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        line = (
            f"{str(r['method']):<16} "
            f"{str(r['sampling_policy']):<10} "
            f"{int(r['num_iter']):>6} "
            f"{int(r['num_images']):>8} "
            f"{format_float(r.get('fid')):>10} "
            f"{format_float(r.get('is')):>10} "
            f"{format_float(r.get('kid')):>10} "
            f"{format_float(r.get('precision')):>10} "
            f"{format_float(r.get('recall')):>10} "
            f"{format_float(r.get('sec_per_image')):>10} "
            f"{str(r['status']):>10}"
        )
        print(line)

    print("=" * 132 + "\n")


def main() -> None:
    args = parse_args()

    code_dir = Path(args.code_dir)
    data_root = Path(args.data_root)
    pretrain_root = Path(args.pretrain_root)
    planmar_ckpt = Path(args.planmar_ckpt)
    baseline_ckpt = Path(args.baseline_ckpt)
    eval_root = Path(args.eval_root)
    log_dir = eval_root / "logs"

    ensure_exists(code_dir / "main_revealmar.py", "main_revealmar.py")
    ensure_exists(code_dir / "engine_mar.py", "engine_mar.py")
    ensure_exists(data_root / "train", "ImageNet train directory")
    ensure_exists(pretrain_root / "vae" / "kl16.ckpt", "VAE checkpoint")
    ensure_exists(planmar_ckpt / "checkpoint-last.pth", "PlanMAR-S checkpoint")
    ensure_exists(baseline_ckpt / "checkpoint-last.pth", "Baseline checkpoint")

    eval_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    num_iters = [int(x.strip()) for x in args.num_iters.split(",") if x.strip()]
    policies = [x.strip() for x in args.methods.split(",") if x.strip()]

    method_items = []
    for policy in policies:
        if policy == "baseline":
            method_items.append(("MAR-Baseline", "baseline", baseline_ckpt))
        elif policy == "planner":
            method_items.append(("PlanMAR-S", "planner", planmar_ckpt))
        else:
            raise ValueError(f"Unsupported policy by current RevealMAR code: {policy}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "8")

    # Syntax checks only. Metrics/generation are still handled by engine_mar.py through main_revealmar.py.
    for f in ["main_revealmar.py", "engine_mar.py"]:
        subprocess.check_call(
            [sys.executable, "-m", "py_compile", f],
            cwd=str(code_dir),
            env=env,
        )

    rows: List[Dict[str, object]] = []

    print("\n" + "=" * 88)
    print("[PlanMAR-S Evaluation Suite: print-only]")
    print("This script does NOT recompute metrics.")
    print("It only calls main_revealmar.py --evaluate, which uses engine_mar.evaluate().")
    print(f"code_dir     : {code_dir}")
    print(f"data_root    : {data_root}")
    print(f"eval_root    : {eval_root}")
    print(f"num_images   : {args.num_images}")
    print(f"eval_bsz     : {args.eval_bsz}")
    print(f"num_iters    : {num_iters}")
    print(f"methods      : {policies}")
    print("=" * 88 + "\n")

    for method_name, policy, ckpt in method_items:
        for num_iter in num_iters:
            run_name = f"{method_name}_policy-{policy}_iter-{num_iter}_n-{args.num_images}"
            run_out = eval_root / run_name
            run_log = log_dir / f"{run_name}.log"
            run_out.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and log_has_metrics(run_log):
                print(f"[SKIP] Existing metrics found: {run_log}")
                retcode = 0
            else:
                cmd = [
                    sys.executable, "main_revealmar.py",
                    "--model", args.model,
                    "--img_size", "256",
                    "--vae_path", str(pretrain_root / "vae" / "kl16.ckpt"),
                    "--vae_embed_dim", "16",
                    "--vae_stride", "16",
                    "--patch_size", "1",
                    "--data_path", str(data_root),
                    "--resume", str(ckpt),
                    "--diffloss_d", str(args.diffloss_d),
                    "--diffloss_w", str(args.diffloss_w),
                    "--diffusion_batch_mul", "1",
                    "--evaluate",
                    "--eval_bsz", str(args.eval_bsz),
                    "--num_images", str(args.num_images),
                    "--num_iter", str(num_iter),
                    "--num_sampling_steps", str(args.num_sampling_steps),
                    "--cfg", str(args.cfg),
                    "--cfg_schedule", args.cfg_schedule,
                    "--temperature", str(args.temperature),
                    "--pseudo_target_type", args.pseudo_target_type,
                    "--planner_loss_weight", str(args.planner_loss_weight),
                    "--candidate_pool_size", str(args.candidate_pool_size),
                    "--sampling_policy", policy,
                    "--mixed_policy_ratio", str(args.mixed_policy_ratio),
                    "--log_planner_sampling_debug",
                    "--log_planner_sampling_steps", "8",
                    "--output_dir", str(run_out),
                    "--dist_url", "env://",
                ]

                print("=" * 88)
                print(f"[RUN] {run_name}")
                print(f"[POLICY] {policy}")
                print(f"[CKPT] {ckpt}")
                print(f"[OUT] {run_out}")
                print(f"[LOG] {run_log}")
                print("[COMMAND] " + " ".join(shlex.quote(x) for x in cmd))
                print("=" * 88)

                retcode = run_cmd(cmd, run_log, cwd=code_dir, env=env)

                if retcode != 0 and not args.continue_on_error:
                    print(f"[ERROR] Run failed: {run_name}. See log: {run_log}")
                    sys.exit(retcode)

            metrics = parse_log(run_log)

            sec_per_image = metrics.get("sec_per_image")
            throughput = 1.0 / sec_per_image if sec_per_image and sec_per_image > 0 else None

            text = run_log.read_text(encoding="utf-8", errors="ignore") if run_log.exists() else ""

            status = "ok"
            if retcode != 0 or "Traceback" in text or "CUDA out of memory" in text:
                status = "failed"
            elif metrics.get("fid") is None:
                status = "no_metrics"

            row = {
                "method": method_name,
                "sampling_policy": policy,
                "checkpoint": str(ckpt),
                "num_iter": num_iter,
                "num_images": args.num_images,
                "eval_bsz": args.eval_bsz,
                "cfg": args.cfg,
                "num_sampling_steps": args.num_sampling_steps,
                "fid": metrics.get("fid"),
                "is": metrics.get("is"),
                "kid": metrics.get("kid"),
                "kid_std": metrics.get("kid_std"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "generated_images_timed": metrics.get("generated_images_timed"),
                "total_gen_seconds_measured": metrics.get("total_gen_seconds_measured"),
                "sec_per_image": sec_per_image,
                "throughput_img_per_sec": throughput,
                "status": status,
                "log_path": str(run_log),
                "output_dir": str(run_out),
            }

            rows.append(row)
            print_result_summary(row)

    print_final_table(rows)

    print("[DONE] PlanMAR-S evaluation suite finished.")
    print(f"Results root: {eval_root}")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
