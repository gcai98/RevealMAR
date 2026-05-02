#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_TRAJ = r"C:\caogang\RevealMAR\RevealMAR\p0_runs\reveal_trajectory"
DEFAULT_INTR = r"C:\caogang\RevealMAR\RevealMAR\p0_runs\early_step_intervention"


def parse_args():
    parser = argparse.ArgumentParser("Collect reveal trajectory and intervention results")
    parser.add_argument("--trajectory_root", default=DEFAULT_TRAJ)
    parser.add_argument("--intervention_root", default=DEFAULT_INTR)
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


def clean_float(value):
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def latest_fid_is(text):
    pattern = r"FID:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*Inception Score:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return "", ""
    return clean_float(matches[-1][0]), clean_float(matches[-1][1])


def latest_timing(text):
    pattern = r"Generating\s+\d+\s+images\s+takes\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+seconds,\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+sec per image"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return ""
    return clean_float(matches[-1][1])


def collect_trajectory(root):
    rows = []
    root.mkdir(parents=True, exist_ok=True)
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        summary = load_json(child / "reveal_trajectory_summary.json")
        log = read_text(child / "eval.log")
        status = "failed" if failed(log) else ("ok" if summary else "no_metrics")
        cfg = load_json(child / "config.json")
        rows.append({
            "method": cfg.get("run_name") or child.name,
            "selected_mean": summary.get("mean_selected_per_step", ""),
            "early_selected_mean": summary.get("early_selected_mean", ""),
            "spatial_spread": summary.get("spatial_spread_mean", ""),
            "center_bias": summary.get("center_bias_score", ""),
            "edge_bias": summary.get("edge_bias_score", ""),
            "score_std": summary.get("score_std_mean", ""),
            "status": status,
        })
    return rows


def collect_intervention(root):
    rows = []
    root.mkdir(parents=True, exist_ok=True)
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        cfg = load_json(child / "config.json")
        log = read_text(child / "eval.log")
        fid, inception = latest_fid_is(log)
        status = "failed" if failed(log) else ("ok" if fid != "" else "no_metrics")
        rows.append({
            "run_name": cfg.get("run_name") or child.name,
            "intervention_policy": cfg.get("intervention_policy", ""),
            "intervention_steps": cfg.get("intervention_steps", ""),
            "FID": fid,
            "Inception Score": inception,
            "sec_per_image": latest_timing(log),
            "status": status,
        })
    return rows


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in fields})


def write_json(path, rows):
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def fmt(value):
    if value in ("", None):
        return "NA"
    try:
        return "{:.4f}".format(float(value))
    except (TypeError, ValueError):
        return str(value)


def print_tables(traj_rows, intr_rows):
    print("\nReveal trajectory")
    print("| method | selected_mean | early_selected_mean | spatial_spread | center_bias | edge_bias | score_std | status |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in traj_rows:
        print("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["method"], fmt(r["selected_mean"]), fmt(r["early_selected_mean"]),
            fmt(r["spatial_spread"]), fmt(r["center_bias"]), fmt(r["edge_bias"]),
            fmt(r["score_std"]), r["status"],
        ))
    print("\nEarly-step intervention")
    print("| run_name | intervention_policy | intervention_steps | FID | IS | sec/img | status |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in intr_rows:
        print("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["run_name"], r["intervention_policy"], r["intervention_steps"],
            fmt(r["FID"]), fmt(r["Inception Score"]), fmt(r["sec_per_image"]), r["status"],
        ))


def main():
    args = parse_args()
    traj_root = Path(args.trajectory_root)
    intr_root = Path(args.intervention_root)
    traj_rows = collect_trajectory(traj_root)
    intr_rows = collect_intervention(intr_root)
    traj_fields = ["method", "selected_mean", "early_selected_mean", "spatial_spread", "center_bias", "edge_bias", "score_std", "status"]
    intr_fields = ["run_name", "intervention_policy", "intervention_steps", "FID", "Inception Score", "sec_per_image", "status"]
    write_csv(traj_root / "reveal_trajectory_summary_all.csv", traj_rows, traj_fields)
    write_json(traj_root / "reveal_trajectory_summary_all.json", traj_rows)
    write_csv(intr_root / "early_step_intervention_summary.csv", intr_rows, intr_fields)
    write_json(intr_root / "early_step_intervention_summary.json", intr_rows)
    print_tables(traj_rows, intr_rows)
    print("\nTrajectory CSV saved to {}".format(traj_root / "reveal_trajectory_summary_all.csv"))
    print("Intervention CSV saved to {}".format(intr_root / "early_step_intervention_summary.csv"))


if __name__ == "__main__":
    main()
