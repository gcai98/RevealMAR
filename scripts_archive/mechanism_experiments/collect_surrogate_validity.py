#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from pathlib import Path


DEFAULT_ROOT = r"p0_runs\surrogate_validity_trained_ckpt"

METHOD_ALIASES = {
    "random": "random",
    "confidence": "confidence",
    "entropy": "entropy",
    "current loss": "current_loss",
    "current_loss": "current_loss",
    "planmar-s surrogate": "planner",
    "planmar-s": "planner",
    "planner": "planner",
    "oracle": "oracle",
    "rollout proxy / oracle-like": "oracle",
    "rollout_proxy": "oracle",
    "oracle_like": "oracle",
}

METHOD_ORDER = ["random", "confidence", "entropy", "current_loss", "planner", "oracle"]
METRICS = ["spearman", "kendall_tau", "topk_overlap", "ndcg_at_k", "regret"]
CONFIG_FIELDS = [
    "resume",
    "data_path",
    "num_eval_images",
    "eval_bsz",
    "num_states_per_image",
    "rollout_horizon",
    "topk_eval",
    "num_iter",
    "state_step_min",
    "state_step_max",
    "local_radius",
    "candidate_pool_size",
    "candidate_selection_mode",
    "uncertainty_mc_samples",
    "uncertainty_temperature",
]
CSV_COLUMNS = ["method"] + METRICS + ["status"] + CONFIG_FIELDS


def parse_args():
    parser = argparse.ArgumentParser("Collect surrogate-validity P0 results")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Output folder from run_surrogate_validity.bat")
    return parser.parse_args()


def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def load_json(path):
    try:
        return json.loads(read_text(path))
    except json.JSONDecodeError:
        return {}


def normalize_method(name):
    clean = str(name or "").strip().lower()
    clean = clean.replace("-", "-")
    clean = re.sub(r"\s+", " ", clean)
    return METHOD_ALIASES.get(clean, clean.replace(" ", "_"))


def normalize_metric_name(name):
    clean = str(name or "").strip().lower()
    clean = clean.replace("-", "_").replace("@", "_at_")
    clean = re.sub(r"[^a-z0-9_]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    aliases = {
        "kendall": "kendall_tau",
        "kendall_tau": "kendall_tau",
        "top_k_overlap": "topk_overlap",
        "topk_overlap": "topk_overlap",
        "ndcg_k": "ndcg_at_k",
        "ndcg_at_k": "ndcg_at_k",
    }
    return aliases.get(clean, clean)


def clean_value(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.upper() in ("", "NA", "NAN", "NONE", "NULL"):
        return ""
    try:
        return float(text)
    except ValueError:
        return text


def empty_metric_row(method):
    row = {"method": method}
    for metric in METRICS:
        row[metric] = ""
    return row


def merge_metric_row(rows, method, values):
    method = normalize_method(method)
    if method not in rows:
        rows[method] = empty_metric_row(method)
    for key, value in values.items():
        metric = normalize_metric_name(key)
        if metric in METRICS:
            rows[method][metric] = clean_value(value)


def parse_summary_json(path):
    payload = load_json(path)
    rows = {}
    summary = payload.get("summary", [])
    if isinstance(summary, list):
        for item in summary:
            if not isinstance(item, dict):
                continue
            method = item.get("row") or item.get("method") or item.get("name")
            if not method:
                continue
            merge_metric_row(rows, method, item)
    elif isinstance(summary, dict):
        for method, values in summary.items():
            if isinstance(values, dict):
                merge_metric_row(rows, method, values)
    return rows


def parse_summary_csv(path):
    rows = {}
    try:
        with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for item in reader:
                method = item.get("row") or item.get("method") or item.get("name")
                if method:
                    merge_metric_row(rows, method, item)
    except OSError:
        return {}
    return rows


def parse_log_table(text):
    rows = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("-"):
            continue
        match = re.match(
            r"^\s*(Random|Confidence|Entropy|Current Loss|PlanMAR-S Surrogate|Rollout Proxy / Oracle-like)"
            r"\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$",
            line,
        )
        if not match:
            continue
        method, spearman, kendall, overlap, ndcg, regret = match.groups()
        merge_metric_row(
            rows,
            method,
            {
                "spearman": spearman,
                "kendall_tau": kendall,
                "topk_overlap": overlap,
                "ndcg_at_k": ndcg,
                "regret": regret,
            },
        )
    return rows


def find_metric_rows(root, log_text):
    candidates = [
        root / "surrogate_validity.json",
        root / "surrogate_validity.csv",
    ]
    rows = {}
    for path in candidates:
        if not path.exists():
            continue
        parsed = parse_summary_json(path) if path.suffix.lower() == ".json" else parse_summary_csv(path)
        if parsed:
            rows.update(parsed)
            return rows, str(path)

    for path in sorted(root.glob("*.json")):
        if path.name == "config.json":
            continue
        parsed = parse_summary_json(path)
        if parsed:
            rows.update(parsed)
            return rows, str(path)

    for path in sorted(root.glob("*.csv")):
        parsed = parse_summary_csv(path)
        if parsed:
            rows.update(parsed)
            return rows, str(path)

    rows = parse_log_table(log_text)
    return rows, "eval.log" if rows else ""


def load_config(root, structured_source):
    config = load_json(root / "config.json")
    structured = load_json(Path(structured_source)) if structured_source and structured_source.endswith(".json") else {}
    metadata = structured.get("metadata", {}) if isinstance(structured, dict) else {}
    out = {}
    for field in CONFIG_FIELDS:
        out[field] = config.get(field, metadata.get(field, ""))
    if not out.get("resume"):
        out["resume"] = metadata.get("checkpoint", "")
    return out


def status_for(log_text, rows):
    lower = log_text.lower()
    if "traceback" in lower or "runtimeerror" in lower or "cuda out of memory" in lower:
        return "failed"
    has_metric = False
    for row in rows.values():
        for metric in METRICS:
            if row.get(metric, "") != "":
                has_metric = True
                break
        if has_metric:
            break
    return "ok" if has_metric else "no_metrics"


def ordered_rows(rows, status, config):
    ordered = []
    seen = set()
    for method in METHOD_ORDER:
        if method in rows:
            item = dict(rows[method])
            item["status"] = status
            item.update(config)
            ordered.append(item)
            seen.add(method)
    for method in sorted(rows):
        if method in seen:
            continue
        item = dict(rows[method])
        item["status"] = status
        item.update(config)
        ordered.append(item)
    return ordered


def csv_value(value):
    return "" if value is None else value


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in CSV_COLUMNS})


def write_json(path, root, source, rows):
    payload = {
        "root": str(root),
        "source": source,
        "num_rows": len(rows),
        "rows": [{key: csv_value(row.get(key, "")) for key in CSV_COLUMNS} for row in rows],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def fmt(value):
    if value in ("", None):
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def print_table(rows):
    headers = ["method", "spearman", "kendall_tau", "topk_overlap", "ndcg_at_k", "regret", "status"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            str(row.get("method", "") or "NA"),
            fmt(row.get("spearman")),
            fmt(row.get("kendall_tau")),
            fmt(row.get("topk_overlap")),
            fmt(row.get("ndcg_at_k")),
            fmt(row.get("regret")),
            str(row.get("status", "") or "NA"),
        ]
        print("| " + " | ".join(values) + " |")


def main():
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    log_text = read_text(root / "eval.log")
    metric_rows, source = find_metric_rows(root, log_text)
    status = status_for(log_text, metric_rows)
    config = load_config(root, source)
    rows = ordered_rows(metric_rows, status, config)

    csv_path = root / "surrogate_validity_summary.csv"
    json_path = root / "surrogate_validity_summary.json"
    write_csv(csv_path, rows)
    write_json(json_path, root, source, rows)

    print_table(rows)
    print(f"\nCSV saved to {csv_path}")
    print(f"JSON saved to {json_path}")

    if status != "ok":
        print(f"WARNING: surrogate validity status is {status}")
    methods = {row.get("method") for row in rows}
    if "planner" not in methods:
        print("WARNING: planner row missing")
    if "oracle" not in methods:
        print("WARNING: oracle row missing")


if __name__ == "__main__":
    main()
