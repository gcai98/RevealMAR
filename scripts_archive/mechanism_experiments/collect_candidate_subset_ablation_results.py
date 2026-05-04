#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


DEFAULT_ROOT = r"p0_runs\candidate_subset_ablation"
FIELDS = [
    "mode", "spearman", "kendall_tau", "topk_overlap", "ndcg_at_k",
    "spatial_spread", "pairwise_dist", "center_bias", "edge_bias",
    "duplicate_rate", "masked_coverage_fraction", "overlap_with_mixed",
    "overlap_with_topk", "status",
]


def parse_args():
    parser = argparse.ArgumentParser("Collect candidate subset ablation results")
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


def failed(text):
    lower = text.lower()
    return "traceback" in lower or "runtimeerror" in lower or "cuda out of memory" in lower


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def collect(root):
    log = read_text(root / "eval.log")
    rows = read_csv(root / "candidate_subset_stability.csv")
    if not rows:
        data = load_json(root / "candidate_subset_stability.json")
        rows = data.get("results", []) if isinstance(data, dict) else []
    status = "failed" if failed(log) else ("ok" if rows else "no_metrics")
    out = []
    for row in rows:
        item = {field: row.get(field, "") for field in FIELDS}
        item["status"] = status if status != "ok" else row.get("status", "ok")
        out.append(item)
    if not out:
        item = {field: "" for field in FIELDS}
        item["status"] = status
        out.append(item)
    return out


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, rows, config):
    with path.open("w", encoding="utf-8") as f:
        json.dump({"config": config, "results": rows}, f, indent=2)


def fmt(value):
    if value in ("", None):
        return "NA"
    try:
        return "{:.4f}".format(float(value))
    except (TypeError, ValueError):
        return str(value)


def print_table(rows):
    print("| mode | spearman | kendall_tau | topk_overlap | ndcg_at_k | spatial_spread | pairwise_dist | duplicate_rate | overlap_with_mixed | status |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row.get("mode") or "NA",
            fmt(row.get("spearman")),
            fmt(row.get("kendall_tau")),
            fmt(row.get("topk_overlap")),
            fmt(row.get("ndcg_at_k")),
            fmt(row.get("spatial_spread")),
            fmt(row.get("pairwise_dist")),
            fmt(row.get("duplicate_rate")),
            fmt(row.get("overlap_with_mixed")),
            row.get("status", "NA"),
        ))


def main():
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    csv_path = root / "candidate_subset_ablation_summary.csv"
    json_path = root / "candidate_subset_ablation_summary.json"
    write_csv(csv_path, rows)
    write_json(json_path, rows, load_json(root / "config.json"))
    print_table(rows)
    print("\nCSV saved to {}".format(csv_path))
    print("JSON saved to {}".format(json_path))


if __name__ == "__main__":
    main()
