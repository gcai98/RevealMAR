#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_ROOT = r"p0_runs\set_action_consistency"

COLUMNS = [
    "method",
    "mean_set_gain",
    "mean_regret_vs_greedy",
    "mean_overlap_with_greedy",
    "mean_redundancy",
    "num_states",
    "set_k",
    "candidate_pool_size",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser("Collect set-action consistency results")
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


def failed_status(text):
    lower = text.lower()
    return "traceback" in lower or "runtimeerror" in lower or "cuda out of memory" in lower


def read_csv_rows(path):
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def rows_from_json(path):
    data = load_json(path)
    rows = data.get("results") if isinstance(data, dict) else None
    return rows if isinstance(rows, list) else []


def parse_rows(root):
    csv_rows = read_csv_rows(root / "set_action_consistency.csv")
    rows = csv_rows if csv_rows else rows_from_json(root / "set_action_consistency.json")
    log_text = read_text(root / "eval.log")
    if failed_status(log_text):
        status = "failed"
    elif rows:
        status = "ok"
    else:
        status = "no_metrics"
    out = []
    for row in rows:
        item = {key: row.get(key, "") for key in COLUMNS}
        item["status"] = status
        out.append(item)
    if not out and status != "ok":
        out.append({key: "" for key in COLUMNS})
        out[0]["status"] = status
    return out


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in COLUMNS})


def write_json(path, root, rows):
    payload = {
        "root": str(root),
        "config": load_json(root / "config.json"),
        "results": rows,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def fmt(value):
    if value in ("", None):
        return "NA"
    try:
        return "{:.4f}".format(float(value))
    except (TypeError, ValueError):
        return str(value)


def print_table(rows):
    print("| method | set_gain | regret_vs_greedy | overlap_with_greedy | redundancy | status |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print("| {} | {} | {} | {} | {} | {} |".format(
            row.get("method") or "NA",
            fmt(row.get("mean_set_gain")),
            fmt(row.get("mean_regret_vs_greedy")),
            fmt(row.get("mean_overlap_with_greedy")),
            fmt(row.get("mean_redundancy")),
            row.get("status", "NA"),
        ))


def main():
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = parse_rows(root)

    csv_path = root / "set_action_consistency_summary.csv"
    json_path = root / "set_action_consistency_summary.json"
    write_csv(csv_path, rows)
    write_json(json_path, root, rows)

    print_table(rows)
    print("\nCSV saved to {}".format(csv_path))
    print("JSON saved to {}".format(json_path))
    bad = [row for row in rows if row.get("status") != "ok"]
    if bad:
        print("\nWARNING: set-action consistency status is not ok: {}".format(
            ", ".join(sorted(set(row.get("status", "NA") for row in bad)))
        ))


if __name__ == "__main__":
    main()
