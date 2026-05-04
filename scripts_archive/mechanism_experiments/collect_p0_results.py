#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean


DEFAULT_ROOT = r"p0_runs\main_results_trained_ckpt_full_policies"

SUMMARY_COLUMNS = [
    "summary_steps",
    "summary_budget_mean",
    "summary_budget_min",
    "summary_budget_max",
    "summary_selected_mean",
    "summary_selected_min",
    "summary_selected_max",
    "summary_conc_mean",
    "summary_conc_min",
    "summary_conc_max",
    "summary_score_mean_mean",
    "summary_score_std_mean",
    "summary_score_std_min",
    "summary_score_std_max",
    "summary_score_min_mean",
    "summary_score_max_mean",
    "summary_top1_mean",
    "summary_topk_mean",
    "summary_uncertainty_mean",
    "summary_uncertainty_std_mean",
    "summary_uncertainty_min_mean",
    "summary_uncertainty_max_mean",
]

COLUMNS = [
    "variant",
    "method",
    "reveal_strategy",
    "budget_rule",
    "sampling_policy",
    "pseudo_target_type",
    "budget_mode",
    "resume",
    "data_path",
    "num_images",
    "class_num",
    "eval_bsz",
    "num_iter",
    "cfg",
    "FID",
    "Inception Score",
    "total_generation_seconds",
    "generated_images_timed",
    "sec_per_image",
    "throughput_img_per_sec",
    "has_resume_missing_model_keys",
    "has_ema_missing_keys",
    "has_planner_head",
    "planner_debug_lines",
    "avg_budget",
    "avg_selected",
    "avg_concentration",
    "avg_score_mean",
    "avg_score_std",
    "avg_score_min",
    "avg_score_max",
    "avg_top1_score_mean",
    "avg_topk_score_mean",
    *SUMMARY_COLUMNS,
    "status",
    "run_dir",
]


def parse_args():
    parser = argparse.ArgumentParser("Collect P0 main-result runs")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Root containing one subfolder per variant")
    return parser.parse_args()


def read_text(path):
    encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "gbk"]
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    for encoding in encodings:
        try:
            return data.decode(encoding).replace("\x00", "")
        except UnicodeDecodeError:
            continue
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


def clean_int(value):
    if value in ("", None):
        return ""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return ""


def fmt(value, digits=4):
    if value == "" or value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value) if str(value) else "NA"


def first_nonempty(*values):
    for value in values:
        if value not in ("", None):
            return value
    return ""


def latest_fid_is(text):
    patterns = [
        r"FID:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*Inception Score:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        r"\bFID\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?).*?\b(?:Inception Score|IS)\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            fid, inception = matches[-1]
            return clean_float(fid), clean_float(inception)
    return "", ""


def latest_generation_timing(text):
    pattern = (
        r"Generating\s+(\d+)\s+images\s+takes\s+"
        r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+seconds,\s+"
        r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+sec per image"
    )
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return "", "", ""
    generated, total_seconds, sec_per_image = matches[-1]
    return clean_int(generated), clean_float(total_seconds), clean_float(sec_per_image)


def parse_debug_values(line):
    out = {}
    aliases = {
        "budget": "budget",
        "selected": "selected",
        "conc": "concentration",
        "concentration": "concentration",
        "mean": "score_mean",
        "score_mean": "score_mean",
        "std": "score_std",
        "score_std": "score_std",
        "min": "score_min",
        "score_min": "score_min",
        "max": "score_max",
        "score_max": "score_max",
        "top1": "top1_score_mean",
        "top1_score_mean": "top1_score_mean",
        "topk": "topk_score_mean",
        "topk_score_mean": "topk_score_mean",
    }
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line):
        canonical = aliases.get(key)
        if canonical:
            out.setdefault(canonical, []).append(float(value))
    return out


def average_or_empty(values):
    return mean(values) if values else ""


def aggregate_summary_values(key, values):
    if not values:
        return ""
    if key.endswith("_min"):
        return min(values)
    if key.endswith("_max"):
        return max(values)
    return mean(values)


def parse_debug_stats(text):
    debug_lines = [line for line in text.splitlines() if "[RevealMAR][planner-sampling]" in line]
    buckets = {
        "budget": [],
        "selected": [],
        "concentration": [],
        "score_mean": [],
        "score_std": [],
        "score_min": [],
        "score_max": [],
        "top1_score_mean": [],
        "topk_score_mean": [],
    }
    for line in debug_lines:
        parsed = parse_debug_values(line)
        for key in buckets:
            buckets[key].extend(parsed.get(key, []))

    return {
        "planner_debug_lines": len(debug_lines),
        "avg_budget": average_or_empty(buckets["budget"]),
        "avg_selected": average_or_empty(buckets["selected"]),
        "avg_concentration": average_or_empty(buckets["concentration"]),
        "avg_score_mean": average_or_empty(buckets["score_mean"]),
        "avg_score_std": average_or_empty(buckets["score_std"]),
        "avg_score_min": average_or_empty(buckets["score_min"]),
        "avg_score_max": average_or_empty(buckets["score_max"]),
        "avg_top1_score_mean": average_or_empty(buckets["top1_score_mean"]),
        "avg_topk_score_mean": average_or_empty(buckets["topk_score_mean"]),
    }


def parse_sampling_summary(text):
    key_map = {
        "steps": "summary_steps",
        "budget_mean": "summary_budget_mean",
        "budget_min": "summary_budget_min",
        "budget_max": "summary_budget_max",
        "selected_mean": "summary_selected_mean",
        "selected_min": "summary_selected_min",
        "selected_max": "summary_selected_max",
        "conc_mean": "summary_conc_mean",
        "conc_min": "summary_conc_min",
        "conc_max": "summary_conc_max",
        "score_mean_mean": "summary_score_mean_mean",
        "score_std_mean": "summary_score_std_mean",
        "score_std_min": "summary_score_std_min",
        "score_std_max": "summary_score_std_max",
        "score_min_mean": "summary_score_min_mean",
        "score_max_mean": "summary_score_max_mean",
        "top1_mean": "summary_top1_mean",
        "topk_mean": "summary_topk_mean",
        "uncertainty_mean": "summary_uncertainty_mean",
        "uncertainty_std_mean": "summary_uncertainty_std_mean",
        "uncertainty_min_mean": "summary_uncertainty_min_mean",
        "uncertainty_max_mean": "summary_uncertainty_max_mean",
    }
    buckets = {column: [] for column in SUMMARY_COLUMNS}
    for line in text.splitlines():
        if "[RevealMAR][sampling-summary]" not in line:
            continue
        for key, value in re.findall(r"([A-Za-z0-9_]+)=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line):
            column = key_map.get(key)
            if column:
                buckets[column].append(float(value))

    return {
        column: aggregate_summary_values(column, values)
        for column, values in buckets.items()
    }


def status_for(text, fid, planner_head_missing):
    lower = text.lower()
    if "traceback" in lower or "cuda out of memory" in lower or "runtimeerror" in lower:
        return "failed"
    if planner_head_missing:
        return "planner_head_missing"
    if fid == "":
        return "no_metrics"
    return "ok"


def discover_run_dirs(root):
    if not root.exists():
        return []
    dirs = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            dirs.append(child)
    return dirs


def build_row(run_dir):
    config = load_json(run_dir / "config.json")
    log_text = read_text(run_dir / "eval.log")

    extra = config.get("extra_params", {})
    if not isinstance(extra, dict):
        extra = {}

    fid, inception = latest_fid_is(log_text)
    generated_images, total_seconds, sec_per_image = latest_generation_timing(log_text)
    throughput = ""
    if sec_per_image not in ("", 0):
        try:
            throughput = 1.0 / float(sec_per_image)
        except (TypeError, ValueError, ZeroDivisionError):
            throughput = ""

    has_resume_missing = "Resume missing model keys" in log_text
    has_ema_missing = "EMA missing keys" in log_text
    has_planner_head = "planner_head" in log_text
    planner_head_missing = has_planner_head and (has_resume_missing or has_ema_missing)
    debug_stats = parse_debug_stats(log_text)
    summary_stats = parse_sampling_summary(log_text)

    row = {
        "variant": first_nonempty(config.get("variant"), run_dir.name),
        "method": config.get("method", ""),
        "reveal_strategy": config.get("reveal_strategy", ""),
        "budget_rule": config.get("budget_rule", ""),
        "sampling_policy": config.get("sampling_policy", ""),
        "pseudo_target_type": config.get("pseudo_target_type", ""),
        "budget_mode": config.get("budget_mode", ""),
        "resume": config.get("resume", ""),
        "data_path": config.get("data_path", ""),
        "num_images": first_nonempty(config.get("num_images"), generated_images),
        "class_num": config.get("class_num", ""),
        "eval_bsz": config.get("eval_bsz", ""),
        "num_iter": config.get("num_iter", ""),
        "cfg": extra.get("cfg", ""),
        "FID": fid,
        "Inception Score": inception,
        "total_generation_seconds": total_seconds,
        "generated_images_timed": generated_images,
        "sec_per_image": sec_per_image,
        "throughput_img_per_sec": throughput,
        "has_resume_missing_model_keys": has_resume_missing,
        "has_ema_missing_keys": has_ema_missing,
        "has_planner_head": has_planner_head,
        "run_dir": str(run_dir),
    }
    row.update(debug_stats)
    row.update(summary_stats)
    row["status"] = status_for(log_text, fid, planner_head_missing)
    return row


def csv_value(value):
    if value is None:
        return ""
    return value


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in COLUMNS})


def json_value(value):
    if value is None:
        return ""
    return value


def write_json(path, rows):
    payload = {
        "root": str(path.parent),
        "num_runs": len(rows),
        "runs": [{key: json_value(row.get(key, "")) for key in COLUMNS} for row in rows],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_markdown_table(rows):
    headers = [
        "variant",
        "policy",
        "budget",
        "FID",
        "IS",
        "sec/img",
        "status",
        "debug_lines",
        "avg_budget",
        "avg_conc",
        "avg_score_std",
        "summary_budget_mean",
        "summary_budget_max",
        "summary_conc_mean",
        "summary_score_std_mean",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            str(row.get("variant", "") or "NA"),
            str(row.get("sampling_policy", "") or "NA"),
            str(row.get("budget_mode", "") or "NA"),
            fmt(row.get("FID")),
            fmt(row.get("Inception Score")),
            fmt(row.get("sec_per_image")),
            str(row.get("status", "") or "NA"),
            str(row.get("planner_debug_lines", "") if row.get("planner_debug_lines", "") != "" else "NA"),
            fmt(row.get("avg_budget")),
            fmt(row.get("avg_concentration")),
            fmt(row.get("avg_score_std")),
            fmt(row.get("summary_budget_mean")),
            fmt(row.get("summary_budget_max")),
            fmt(row.get("summary_conc_mean")),
            fmt(row.get("summary_score_std_mean")),
        ]
        print("| " + " | ".join(values) + " |")


def main():
    args = parse_args()
    root = Path(args.root)
    rows = [build_row(run_dir) for run_dir in discover_run_dirs(root)]

    csv_path = root / "p0_summary.csv"
    json_path = root / "p0_summary.json"
    root.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows)
    write_json(json_path, rows)

    print_markdown_table(rows)
    print(f"\nCSV saved to {csv_path}")
    print(f"JSON saved to {json_path}")

    bad_rows = [row for row in rows if row.get("status") != "ok"]
    if bad_rows:
        print("\nWARNING: non-ok run statuses detected:")
        for row in bad_rows:
            print(f"- {row.get('variant', 'NA')}: {row.get('status', 'NA')}")


if __name__ == "__main__":
    main()
