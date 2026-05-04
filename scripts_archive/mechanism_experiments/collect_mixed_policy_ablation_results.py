#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean


DEFAULT_ROOT = r"p0_runs\mixed_policy_ablation"

TRAIN_COLUMNS = [
    "row_type", "ratio_name", "mixed_policy_ratio", "train_status",
    "mixed_debug_lines", "ref_debug_lines", "ref_target_mean",
    "ref_target_std", "ref_target_pos_frac", "loss_mean",
    "diffloss_mean", "planner_aux_loss_mean", "has_nan",
    "has_traceback", "has_runtime_error", "run_dir",
]

EVAL_COLUMNS = [
    "row_type", "ratio_name", "mixed_policy_ratio", "eval_variant",
    "sampling_policy", "budget_mode", "FID", "Inception Score",
    "sec_per_image", "total_generation_seconds", "summary_budget_mean",
    "summary_budget_max", "summary_conc_mean", "summary_score_std_mean",
    "status", "run_dir",
]

COLUMNS = TRAIN_COLUMNS + [c for c in EVAL_COLUMNS if c not in TRAIN_COLUMNS]


def parse_args():
    parser = argparse.ArgumentParser("Collect mixed-policy ablation results")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    return parser.parse_args()


def read_text(path):
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "gbk"):
        try:
            return data.decode(enc).replace("\x00", "")
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="ignore").replace("\x00", "")


def load_json(path):
    text = read_text(path).lstrip("\ufeff").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def clean_float(value):
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def avg(values):
    return mean(values) if values else ""


def failed_status(text):
    lower = text.lower()
    return "traceback" in lower or "runtimeerror" in lower or "cuda out of memory" in lower


def latest_fid_is(text):
    pattern = (
        r"FID:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*"
        r"Inception Score:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    )
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return "", ""
    fid, inception = matches[-1]
    return clean_float(fid), clean_float(inception)


def latest_generation_timing(text):
    pattern = (
        r"Generating\s+\d+\s+images\s+takes\s+"
        r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+seconds,\s+"
        r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+sec per image"
    )
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return "", ""
    total_seconds, sec_per_image = matches[-1]
    return clean_float(total_seconds), clean_float(sec_per_image)


def parse_ref_target_stats(text):
    lines = [line for line in text.splitlines() if "[RevealMAR][ref-target]" in line]
    buckets = {"mean": [], "std": [], "pos_frac": []}
    for line in lines:
        for key, value in re.findall(r"(mean|std|pos_frac)=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line):
            buckets[key].append(float(value))
    return {
        "ref_debug_lines": len(lines),
        "ref_target_mean": avg(buckets["mean"]),
        "ref_target_std": avg(buckets["std"]),
        "ref_target_pos_frac": avg(buckets["pos_frac"]),
    }


def parse_loss_stats(text):
    buckets = {"total_loss": [], "diffloss": [], "planner_aux_loss": []}
    pattern = (
        r"total_loss=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
        r"diffloss=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
        r"planner_aux_loss=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    )
    for total, diffloss, planner in re.findall(pattern, text):
        buckets["total_loss"].append(float(total))
        buckets["diffloss"].append(float(diffloss))
        buckets["planner_aux_loss"].append(float(planner))
    return {
        "loss_mean": avg(buckets["total_loss"]),
        "diffloss_mean": avg(buckets["diffloss"]),
        "planner_aux_loss_mean": avg(buckets["planner_aux_loss"]),
    }


def parse_sampling_summary(text):
    key_map = {
        "budget_mean": "summary_budget_mean",
        "budget_max": "summary_budget_max",
        "conc_mean": "summary_conc_mean",
        "score_std_mean": "summary_score_std_mean",
    }
    buckets = {value: [] for value in key_map.values()}
    for line in text.splitlines():
        if "[RevealMAR][sampling-summary]" not in line:
            continue
        for key, value in re.findall(r"([A-Za-z0-9_]+)=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line):
            out_key = key_map.get(key)
            if out_key:
                buckets[out_key].append(float(value))
    return {key: avg(values) for key, values in buckets.items()}


def ratio_from_dir(name):
    match = re.search(r"ratio\d+p?\d*", name)
    return match.group(0) if match else ""


def ratio_value_from_name(name):
    mapping = {
        "ratio0p0": 0.0,
        "ratio0p25": 0.25,
        "ratio0p5": 0.5,
        "ratio0p75": 0.75,
    }
    return mapping.get(name, "")


def eval_variant_from_dir(name):
    if name.endswith("_baseline"):
        return "baseline"
    if name.endswith("_planner_hard"):
        return "planner_hard"
    if name.endswith("_planner_soft"):
        return "planner_soft"
    return ""


def train_status(text):
    if failed_status(text):
        return "failed"
    if "[RevealMAR][mixed-policy]" not in text and "[RevealMAR][ref-target]" not in text:
        return "no_debug"
    return "ok"


def eval_status(text, fid):
    if failed_status(text):
        return "failed"
    has_missing = ("Resume missing model keys" in text or "EMA missing keys" in text)
    if "planner_head" in text and has_missing:
        return "planner_head_missing"
    if fid == "":
        return "no_metrics"
    return "ok"


def build_train_row(run_dir):
    config = load_json(run_dir / "config.json")
    text = read_text(run_dir / "train.log")
    lower = text.lower()
    ratio_name = config.get("ratio_name") or ratio_from_dir(run_dir.name)
    row = {
        "row_type": "train",
        "ratio_name": ratio_name,
        "mixed_policy_ratio": clean_float(config.get("mixed_policy_ratio", ratio_value_from_name(ratio_name))),
        "train_status": train_status(text),
        "mixed_debug_lines": sum(1 for line in text.splitlines() if "[RevealMAR][mixed-policy]" in line),
        "has_nan": bool(re.search(r"\bnan\b", lower)),
        "has_traceback": "traceback" in lower,
        "has_runtime_error": "runtimeerror" in lower,
        "run_dir": str(run_dir),
    }
    row.update(parse_ref_target_stats(text))
    row.update(parse_loss_stats(text))
    return row


def build_eval_row(run_dir):
    config = load_json(run_dir / "config.json")
    text = read_text(run_dir / "eval.log")
    fid, inception = latest_fid_is(text)
    total_seconds, sec_per_image = latest_generation_timing(text)
    ratio_name = config.get("ratio_name") or ratio_from_dir(run_dir.name)
    eval_variant = config.get("eval_variant") or eval_variant_from_dir(run_dir.name)
    row = {
        "row_type": "eval",
        "ratio_name": ratio_name,
        "mixed_policy_ratio": clean_float(config.get("mixed_policy_ratio", ratio_value_from_name(ratio_name))),
        "eval_variant": eval_variant,
        "sampling_policy": "baseline" if eval_variant == "baseline" else "planner",
        "budget_mode": config.get("budget_mode", ""),
        "FID": fid,
        "Inception Score": inception,
        "sec_per_image": sec_per_image,
        "total_generation_seconds": total_seconds,
        "status": eval_status(text, fid),
        "run_dir": str(run_dir),
    }
    row.update(parse_sampling_summary(text))
    return row


def discover_rows(root):
    train_rows = []
    eval_rows = []
    if not root.exists():
        return train_rows, eval_rows
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("train_"):
            train_rows.append(build_train_row(child))
        elif child.name.startswith("eval_"):
            eval_rows.append(build_eval_row(child))
    return train_rows, eval_rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in COLUMNS})


def write_json(path, root, train_rows, eval_rows):
    payload = {"root": str(root), "train": train_rows, "eval": eval_rows}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def fmt(value):
    if value in ("", None):
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def print_train_table(rows):
    print("\nTraining")
    print("| ratio | train_status | mixed_debug_lines | ref_lines | ref_mean | ref_std | pos_frac |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print("| {} | {} | {} | {} | {} | {} | {} |".format(
            row.get("ratio_name", "NA"),
            row.get("train_status", "NA"),
            row.get("mixed_debug_lines", "NA"),
            row.get("ref_debug_lines", "NA"),
            fmt(row.get("ref_target_mean")),
            fmt(row.get("ref_target_std")),
            fmt(row.get("ref_target_pos_frac")),
        ))


def print_eval_table(rows):
    print("\nEvaluation")
    print("| ratio | eval_variant | FID | IS | sec/img | budget_mean | budget_max | conc_mean | score_std | status |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row.get("ratio_name", "NA"),
            row.get("eval_variant", "NA"),
            fmt(row.get("FID")),
            fmt(row.get("Inception Score")),
            fmt(row.get("sec_per_image")),
            fmt(row.get("summary_budget_mean")),
            fmt(row.get("summary_budget_max")),
            fmt(row.get("summary_conc_mean")),
            fmt(row.get("summary_score_std_mean")),
            row.get("status", "NA"),
        ))


def main():
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows = discover_rows(root)
    all_rows = train_rows + eval_rows

    csv_path = root / "mixed_policy_ablation_summary.csv"
    json_path = root / "mixed_policy_ablation_summary.json"
    write_csv(csv_path, all_rows)
    write_json(json_path, root, train_rows, eval_rows)

    print_train_table(train_rows)
    print_eval_table(eval_rows)
    print("\nCSV saved to {}".format(csv_path))
    print("JSON saved to {}".format(json_path))

    bad_train = [row for row in train_rows if row.get("train_status") != "ok"]
    bad_eval = [row for row in eval_rows if row.get("status") != "ok"]
    if bad_train or bad_eval:
        print("\nWARNING: non-ok mixed-policy ablation rows detected")
        for row in bad_train:
            print("- train {}: {}".format(row.get("ratio_name", "NA"), row.get("train_status", "NA")))
        for row in bad_eval:
            print("- eval {} {}: {}".format(
                row.get("ratio_name", "NA"), row.get("eval_variant", "NA"), row.get("status", "NA")
            ))


if __name__ == "__main__":
    main()
