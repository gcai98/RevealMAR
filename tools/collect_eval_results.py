#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect P0 evaluation results from existing run directories.

This script parses config.json, run_args.txt, and evaluation logs produced by
the shell runners. It does not generate samples, recompute FID/IS, or import
model code.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


COLUMNS = [
    "table",
    "variant",
    "method",
    "reveal_strategy",
    "budget_rule",
    "resume",
    "output_dir",
    "num_images",
    "eval_bsz",
    "num_iter",
    "fid",
    "is",
    "latency",
    "total_time",
    "extra_params",
    "pseudo_target_type",
    "sampling_policy",
    "budget_mode",
]

TABLE_TO_FILE = {
    "main": "main_results.csv",
    "core": "core_ablation.csv",
    "oracle": "oracle_mismatch.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Collect RevealMAR P0 evaluation results")
    parser.add_argument("--root", required=True, type=str, help="OUTPUT_ROOT containing run directories")
    parser.add_argument("--out", default="", type=str, help="Output directory for CSV tables")
    parser.add_argument("--table", default="all", choices=["all", "main", "core", "oracle"])
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def parse_float(value: object) -> str:
    if value is None:
        return ""
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return ""


def first_float(patterns: Iterable[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return parse_float(match.group(1))
    return ""


def last_float(patterns: Iterable[str], text: str) -> str:
    found = ""
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = next((x for x in value if x not in ("", None)), "")
            found = parse_float(value)
    return found


def parse_metrics_json(text: str) -> Dict[str, str]:
    out = {"fid": "", "is": "", "latency": "", "total_time": "", "num_images": ""}
    matches = re.findall(r"\[METRICS_JSON\]\s*(\{.*?\})", text, flags=re.IGNORECASE | re.DOTALL)
    for raw in matches:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        out["fid"] = parse_float(payload.get("fid") or payload.get("frechet_inception_distance")) or out["fid"]
        out["is"] = parse_float(payload.get("is") or payload.get("inception_score_mean")) or out["is"]
        out["latency"] = parse_float(
            payload.get("latency")
            or payload.get("sec_per_image")
            or payload.get("seconds_per_image")
        ) or out["latency"]
        out["total_time"] = parse_float(
            payload.get("total_time")
            or payload.get("total_gen_seconds_measured")
            or payload.get("generation_time")
        ) or out["total_time"]
        out["num_images"] = parse_float(payload.get("num_images") or payload.get("generated_images_timed")) or out["num_images"]
    return out


def parse_log_metrics(text: str) -> Dict[str, str]:
    metrics = parse_metrics_json(text)

    if not metrics["fid"]:
        metrics["fid"] = last_float(
            [
                r"\bFID\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                r"\bfid['\"]?\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                r"frechet_inception_distance['\"]?\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            ],
            text,
        )

    if not metrics["is"]:
        metrics["is"] = last_float(
            [
                r"Inception Score\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                r"\bIS\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                r"inception_score_mean['\"]?\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            ],
            text,
        )

    gen_line = re.findall(
        r"Generating\s+(\d+)\s+images\s+takes\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+seconds,\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+sec per image",
        text,
        flags=re.IGNORECASE,
    )
    if gen_line:
        n_images, total_time, latency = gen_line[-1]
        metrics["num_images"] = metrics["num_images"] or parse_float(n_images)
        metrics["total_time"] = metrics["total_time"] or parse_float(total_time)
        metrics["latency"] = metrics["latency"] or parse_float(latency)

    if not metrics["total_time"]:
        metrics["total_time"] = last_float(
            [
                r"Total generation time\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                r"generation time\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                r"generated?.*?\btakes\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+seconds",
            ],
            text,
        )

    if not metrics["latency"]:
        metrics["latency"] = last_float(
            [
                r"latency\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                r"sec(?:onds)?\s*/\s*image\s*[:=]?\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+sec per image",
            ],
            text,
        )

    samples_per_sec = first_float(
        [
            r"samples\s*/\s*s\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+samples\s*/\s*s",
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+img\s*/\s*s",
        ],
        text,
    )
    if not metrics["latency"] and samples_per_sec:
        try:
            value = float(samples_per_sec)
            if value > 0:
                metrics["latency"] = str(1.0 / value)
        except ValueError:
            pass

    return metrics


def load_config(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(read_text(path))
    except json.JSONDecodeError:
        return {}


def discover_run_dirs(root: Path) -> List[Path]:
    dirs = set()
    for path in root.rglob("config.json"):
        dirs.add(path.parent)
    for path in root.rglob("run_args.txt"):
        dirs.add(path.parent)
    for path in root.rglob("*.log"):
        dirs.add(path.parent)
    return sorted(dirs)


def infer_table(run_dir: Path, config: Dict[str, object]) -> str:
    table = str(config.get("table", "")).strip().lower()
    if table:
        return table
    lower_path = str(run_dir).lower()
    if "main_results" in lower_path:
        return "main"
    if "core_ablation" in lower_path:
        return "core"
    if "oracle_mismatch" in lower_path:
        return "oracle"
    return ""


def collect_log_text(run_dir: Path) -> str:
    chunks = []
    for path in sorted(run_dir.glob("*.log")):
        chunks.append(read_text(path))
    if not chunks:
        for path in sorted(run_dir.glob("*.txt")):
            if path.name != "run_args.txt":
                chunks.append(read_text(path))
    return "\n".join(chunks)


def build_row(run_dir: Path) -> Dict[str, str]:
    config = load_config(run_dir)
    run_args = read_text(run_dir / "run_args.txt") if (run_dir / "run_args.txt").exists() else ""
    metrics = parse_log_metrics(collect_log_text(run_dir))
    table = infer_table(run_dir, config)

    extra_params = config.get("extra_params", "")
    if isinstance(extra_params, (dict, list)):
        extra_params = json.dumps(extra_params, sort_keys=True)

    row = {
        "table": table,
        "variant": str(config.get("variant", run_dir.name)),
        "method": str(config.get("method", "")),
        "reveal_strategy": str(config.get("reveal_strategy", "")),
        "budget_rule": str(config.get("budget_rule", "")),
        "resume": str(config.get("resume", "")),
        "output_dir": str(run_dir),
        "num_images": str(config.get("num_images", "")) or metrics.get("num_images", ""),
        "eval_bsz": str(config.get("eval_bsz", "")),
        "num_iter": str(config.get("num_iter", "")),
        "fid": metrics.get("fid", ""),
        "is": metrics.get("is", ""),
        "latency": metrics.get("latency", ""),
        "total_time": metrics.get("total_time", ""),
        "extra_params": str(extra_params),
        "pseudo_target_type": str(config.get("pseudo_target_type", "")),
        "sampling_policy": str(config.get("sampling_policy", "")),
        "budget_mode": str(config.get("budget_mode", "")),
    }

    if run_args and not row["extra_params"]:
        row["extra_params"] = run_args.strip()
    return row


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [build_row(run_dir) for run_dir in discover_run_dirs(root)]
    rows = [row for row in rows if row["table"] or row["fid"] or row["is"] or row["latency"] or row["total_time"]]

    selected_rows = rows if args.table == "all" else [row for row in rows if row["table"] == args.table]

    write_csv(out_dir / "all_results.csv", rows)
    for table, filename in TABLE_TO_FILE.items():
        if args.table in ("all", table):
            write_csv(out_dir / filename, [row for row in rows if row["table"] == table])

    print("Collected {} runs from {}".format(len(selected_rows), root))
    print("Wrote {}".format(out_dir / "all_results.csv"))
    for table, filename in TABLE_TO_FILE.items():
        if args.table in ("all", table):
            print("Wrote {}".format(out_dir / filename))


if __name__ == "__main__":
    main()
