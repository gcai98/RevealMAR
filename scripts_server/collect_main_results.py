#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_ROOT = "/root/autodl-tmp/outputs/planmar_main"
MODELS = ("base", "large", "huge")
MODEL_SET = set(MODELS)
POLICIES = ("baseline", "confidence", "entropy", "planner")
NUM_ITERS = (64, 128, 256)

FIELDS = [
    "model_name",
    "policy",
    "num_iter",
    "FID",
    "Inception Score",
    "sec_per_image",
    "total_generation_seconds",
    "budget_mean",
    "budget_max",
    "conc_mean",
    "score_std_mean",
    "status",
    "run_dir",
    "log_path",
]


def read_text(path):
    if not path.exists():
        return ""
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "gbk"):
        try:
            return path.read_text(encoding=enc).replace("\x00", "")
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="ignore").replace("\x00", "")


def fmt(value):
    return "NA" if value in ("", None) else str(value)


def parse_models(models_arg):
    if models_arg.strip().lower() == "all":
        return list(MODELS)
    models = [item.strip().lower() for item in models_arg.split(",") if item.strip()]
    if not models:
        raise ValueError("--models must be 'all' or a comma-separated list")
    invalid = [name for name in models if name not in MODEL_SET]
    if invalid:
        raise ValueError("Invalid model name(s): {}. Expected one of: {}".format(
            ", ".join(invalid),
            ", ".join(MODELS),
        ))
    return models


def output_name(prefix, base_name):
    return base_name if not prefix else f"{prefix}_{base_name}"


def parse_float(text):
    try:
        return str(float(text))
    except (TypeError, ValueError):
        return ""


def parse_metric(text, names):
    matches = []
    for name in names:
        pattern = re.compile(rf"{re.escape(name)}\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.I)
        matches.extend(pattern.findall(text))
    return parse_float(matches[-1]) if matches else ""


def parse_generation_timing(text):
    pattern = re.compile(
        r"Generating\s+(\d+)\s+images\s+takes\s+([-+]?\d+(?:\.\d+)?)\s+seconds,\s+([-+]?\d+(?:\.\d+)?)\s+sec\s+per\s+image",
        re.I,
    )
    matches = pattern.findall(text)
    if not matches:
        return "", ""
    _count, total, sec_per = matches[-1]
    return parse_float(total), parse_float(sec_per)


def parse_summary_line(text):
    out = {
        "budget_mean": "",
        "budget_max": "",
        "conc_mean": "",
        "score_std_mean": "",
    }
    lines = [line for line in text.splitlines() if "[RevealMAR][sampling-summary]" in line]
    if not lines:
        return out
    line = lines[-1]
    payload = line.split("]", 2)[-1]
    pairs = dict(re.findall(r"([A-Za-z0-9_]+)=([^,\s]+)", payload))
    mapping = {
        "summary_budget_mean": "budget_mean",
        "budget_mean": "budget_mean",
        "summary_budget_max": "budget_max",
        "budget_max": "budget_max",
        "summary_conc_mean": "conc_mean",
        "conc_mean": "conc_mean",
        "summary_score_std_mean": "score_std_mean",
        "score_std_mean": "score_std_mean",
    }
    for src, dst in mapping.items():
        if src in pairs and not out[dst]:
            out[dst] = parse_float(pairs[src])
    return out


def status_for(text, fid):
    failed = any(marker in text for marker in ("Traceback", "RuntimeError", "CUDA out of memory"))
    planner_missing = "planner_head" in text and (
        "Resume missing model keys" in text or "EMA missing keys" in text
    )
    if failed:
        return "failed"
    if planner_missing:
        return "planner_head_missing"
    if not fid:
        return "no_metrics"
    return "ok"


def find_log(root, model_name, policy, num_iter):
    run_dir = root / model_name / "eval_main" / f"{policy}_iter{num_iter}"
    run_log = run_dir / "eval.log"
    if run_log.exists():
        return run_dir, run_log
    log_path = root / model_name / "logs" / f"eval_{policy}_iter{num_iter}.log"
    return run_dir, log_path


def collect(root, models, skip_missing_models=False):
    rows = []
    for model_name in models:
        model_root = root / model_name
        if skip_missing_models and not model_root.exists():
            print(f"[WARN] Skipping missing model directory: {model_root}")
            continue
        for policy in POLICIES:
            for num_iter in NUM_ITERS:
                run_dir, log_path = find_log(root, model_name, policy, num_iter)
                text = read_text(log_path)
                fid = parse_metric(text, ("FID", "fid"))
                inception = parse_metric(text, ("Inception Score", "IS", "inception_score"))
                total_sec, sec_per = parse_generation_timing(text)
                summary = parse_summary_line(text)
                row = {
                    "model_name": model_name,
                    "policy": policy,
                    "num_iter": str(num_iter),
                    "FID": fid,
                    "Inception Score": inception,
                    "sec_per_image": sec_per,
                    "total_generation_seconds": total_sec,
                    "budget_mean": summary["budget_mean"],
                    "budget_max": summary["budget_max"],
                    "conc_mean": summary["conc_mean"],
                    "score_std_mean": summary["score_std_mean"],
                    "status": status_for(text, fid),
                    "run_dir": str(run_dir),
                    "log_path": str(log_path),
                }
                rows.append(row)
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def print_table(rows):
    print("| model | policy | iter | FID | IS | sec/img | budget_mean | conc_mean | status |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            "| {model_name} | {policy} | {num_iter} | {FID} | {iscore} | {sec} | {budget} | {conc} | {status} |".format(
                model_name=fmt(row["model_name"]),
                policy=fmt(row["policy"]),
                num_iter=fmt(row["num_iter"]),
                FID=fmt(row["FID"]),
                iscore=fmt(row["Inception Score"]),
                sec=fmt(row["sec_per_image"]),
                budget=fmt(row["budget_mean"]),
                conc=fmt(row["conc_mean"]),
                status=fmt(row["status"]),
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--models", default="all")
    parser.add_argument("--skip_missing_models", action="store_true")
    parser.add_argument("--output_prefix", default="")
    args = parser.parse_args()

    root = Path(args.root)
    models = parse_models(args.models)

    print(f"[collect_main_results] root={root}")
    print(f"[collect_main_results] models={','.join(models)}")
    print(f"[collect_main_results] skip_missing_models={args.skip_missing_models}")
    print(f"[collect_main_results] output_prefix={args.output_prefix}")

    rows = collect(root, models, args.skip_missing_models)
    summary_csv = root / output_name(args.output_prefix, "main_results_summary.csv")
    summary_json = root / output_name(args.output_prefix, "main_results_summary.json")
    pareto_csv = root / output_name(args.output_prefix, "pareto_data.csv")

    write_csv(summary_csv, rows, FIELDS)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(
        pareto_csv,
        rows,
        ["model_name", "policy", "num_iter", "FID", "Inception Score", "sec_per_image", "status"],
    )

    print_table(rows)
    print(f"CSV saved to {summary_csv}")
    print(f"JSON saved to {summary_json}")
    print(f"Pareto CSV saved to {pareto_csv}")

    bad = [row for row in rows if row["status"] != "ok"]
    if bad:
        counts = {}
        for row in bad:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        print("[WARN] Non-ok rows:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
