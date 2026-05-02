#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_ROOT = r"C:\caogang\RevealMAR\RevealMAR\p0_runs\budget_calibration"

COLUMNS = [
    "run_name",
    "budget_temperature",
    "budget_score_scale",
    "budget_ema_beta",
    "budget_min",
    "budget_max",
    "FID",
    "Inception Score",
    "sec_per_image",
    "total_generation_seconds",
    "summary_budget_mean",
    "summary_budget_min",
    "summary_budget_max",
    "summary_conc_mean",
    "summary_conc_min",
    "summary_conc_max",
    "summary_score_std_mean",
    "summary_score_std_max",
    "status",
    "run_dir",
]


def parse_args():
    parser = argparse.ArgumentParser("Collect budget calibration results")
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


def parse_sampling_summary(text):
    key_map = {
        "budget_mean": "summary_budget_mean",
        "budget_min": "summary_budget_min",
        "budget_max": "summary_budget_max",
        "conc_mean": "summary_conc_mean",
        "conc_min": "summary_conc_min",
        "conc_max": "summary_conc_max",
        "score_std_mean": "summary_score_std_mean",
        "score_std_max": "summary_score_std_max",
    }
    result = {value: "" for value in key_map.values()}
    summary_lines = [line for line in text.splitlines() if "[RevealMAR][sampling-summary]" in line]
    if not summary_lines:
        return result
    latest = summary_lines[-1]
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", latest):
        out_key = key_map.get(key)
        if out_key:
            result[out_key] = clean_float(value)
    return result


def failed_status(text):
    lower = text.lower()
    return "traceback" in lower or "cuda out of memory" in lower or "runtimeerror" in lower


def status_for_log(text, fid):
    if failed_status(text):
        return "failed"
    has_missing = ("Resume missing model keys" in text or "EMA missing keys" in text)
    if "planner_head" in text and has_missing:
        return "planner_head_missing"
    if fid == "":
        return "no_metrics"
    return "ok"


def parse_arg_value(run_args, name):
    pattern = r"--{}\s+\"?([^\"\s]+)\"?".format(re.escape(name))
    match = re.search(pattern, run_args)
    return match.group(1) if match else ""


def config_value(config, run_args, key, arg_name=None):
    if key in config:
        return config.get(key)
    return parse_arg_value(run_args, arg_name or key)


def build_row(run_dir):
    config = load_json(run_dir / "config.json")
    log_text = read_text(run_dir / "eval.log")
    run_args = read_text(run_dir / "run_args.txt")
    fid, inception = latest_fid_is(log_text)
    total_seconds, sec_per_image = latest_generation_timing(log_text)
    row = {
        "run_name": config.get("run_name") or run_dir.name,
        "budget_temperature": clean_float(config_value(config, run_args, "budget_temperature")),
        "budget_score_scale": clean_float(config_value(config, run_args, "budget_score_scale")),
        "budget_ema_beta": clean_float(config_value(config, run_args, "budget_ema_beta")),
        "budget_min": config_value(config, run_args, "budget_min"),
        "budget_max": config_value(config, run_args, "budget_max"),
        "FID": fid,
        "Inception Score": inception,
        "sec_per_image": sec_per_image,
        "total_generation_seconds": total_seconds,
        "status": status_for_log(log_text, fid),
        "run_dir": str(run_dir),
    }
    row.update(parse_sampling_summary(log_text))
    return row


def discover_rows(root):
    if not root.exists():
        return []
    rows = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (
            (child / "eval.log").exists()
            or (child / "config.json").exists()
            or (child / "run_args.txt").exists()
        ):
            rows.append(build_row(child))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in COLUMNS})


def write_json(path, root, rows):
    payload = {"root": str(root), "runs": rows}
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
    print("| run_name | tau | scale | beta | FID | IS | sec/img | budget_mean | budget_max | conc_mean | score_std | status |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row.get("run_name", "NA"),
            fmt(row.get("budget_temperature")),
            fmt(row.get("budget_score_scale")),
            fmt(row.get("budget_ema_beta")),
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
    rows = discover_rows(root)

    csv_path = root / "budget_calibration_summary.csv"
    json_path = root / "budget_calibration_summary.json"
    write_csv(csv_path, rows)
    write_json(json_path, root, rows)

    print_table(rows)
    print("\nCSV saved to {}".format(csv_path))
    print("JSON saved to {}".format(json_path))

    bad_rows = [row for row in rows if row.get("status") != "ok"]
    if bad_rows:
        print("\nWARNING: non-ok budget calibration rows detected")
        for row in bad_rows:
            print("- {}: {}".format(row.get("run_name", "NA"), row.get("status", "NA")))


if __name__ == "__main__":
    main()
