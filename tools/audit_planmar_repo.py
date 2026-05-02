#!/usr/bin/env python3
import fnmatch
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORE_FILES = [
    "main_revealmar.py",
    "engine_mar.py",
    "models/revealmar.py",
    "util/revealmar_utils.py",
]

EVALUATORS = [
    "eval_surrogate_validity.py",
    "eval_set_action_consistency.py",
    "eval_candidate_subset_stability.py",
    "eval_reveal_trajectory.py",
    "eval_early_step_intervention.py",
]

COLLECTORS = [
    "collect_p0_results.py",
    "collect_surrogate_validity.py",
    "collect_oracle_mismatch_results.py",
    "collect_budget_calibration_results.py",
    "collect_set_action_consistency_results.py",
    "collect_mixed_policy_ablation_results.py",
    "collect_same_parameter_control_results.py",
    "collect_reveal_trajectory_results.py",
    "collect_candidate_subset_ablation_results.py",
]

WINDOWS_SCRIPTS = [
    "run_smoke_health_check.bat",
    "run_ref_target_smoke_train.bat",
    "run_candidate_subset_ablation.bat",
    "run_set_action_consistency.bat",
    "run_reveal_trajectory.bat",
    "run_budget_calibration.bat",
    "run_oracle_mismatch_ref_targets.bat",
    "run_mixed_policy_ablation.bat",
    "run_same_parameter_control.bat",
    "run_early_step_intervention.bat",
    "run_main_results.bat",
]

LINUX_SCRIPTS = [
    "scripts_linux/common.sh",
    "scripts_linux/run_linux_preflight.sh",
    "scripts_linux/run_smoke_health_check.sh",
    "scripts_linux/run_main_results.sh",
    "scripts_linux/run_ref_target_smoke_train.sh",
    "scripts_linux/run_candidate_subset_ablation.sh",
    "scripts_linux/run_set_action_consistency.sh",
    "scripts_linux/run_reveal_trajectory.sh",
    "scripts_linux/run_budget_calibration.sh",
    "scripts_linux/run_oracle_mismatch_ref_targets.sh",
    "scripts_linux/run_mixed_policy_ablation.sh",
    "scripts_linux/run_same_parameter_control.sh",
    "scripts_linux/run_early_step_intervention.sh",
]

CACHE_DIR_NAMES = {".pyc_tmp", ".pycache_prefix", ".pycache_tmp", "__pycache__", ".pytest_cache"}
ARTIFACT_PATTERNS = [
    "*.pth",
    "*.pt",
    "*.ckpt",
    "*.safetensors",
    "events.out.tfevents.*",
    "*.log",
]


def rel(path):
    return path.as_posix()


def git_ls_files():
    try:
        proc = subprocess.run(
            ["git", "--git-dir=.git", "--work-tree=.", "ls-files"],
            cwd=str(ROOT),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def check_required(label, paths):
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        print("[FAIL] Missing {}:".format(label))
        for path in missing:
            print("  - {}".format(path))
    else:
        print("[OK] {} present ({})".format(label, len(paths)))
    return missing


def tracked_cache_files(tracked):
    out = []
    for item in tracked:
        parts = Path(item).parts
        if any(part in CACHE_DIR_NAMES for part in parts):
            out.append(item)
        elif item.endswith((".pyc", ".pyo", ".pyd")):
            out.append(item)
    return out


def tracked_artifacts(tracked):
    out = []
    for item in tracked:
        name = Path(item).name
        if any(fnmatch.fnmatch(name, pattern) for pattern in ARTIFACT_PATTERNS):
            out.append(item)
        if item.startswith("p0_runs/") and "/" in item[len("p0_runs/"):]:
            if not fnmatch.fnmatch(name, "README_*.txt") and not name.endswith(".txt"):
                out.append(item)
    return sorted(set(out))


def present_cache_dirs():
    found = []
    for dirpath, dirnames, _filenames in os.walk(ROOT):
        keep = []
        for name in dirnames:
            if name in CACHE_DIR_NAMES:
                found.append(rel(Path(dirpath).joinpath(name).relative_to(ROOT)))
            else:
                keep.append(name)
        dirnames[:] = keep
    return found


def root_artifact_warnings():
    found = []
    for path in ROOT.iterdir():
        if path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in ARTIFACT_PATTERNS):
            found.append(path.name)
        if path.is_dir() and path.name in {"generated", "samples", "outputs"}:
            found.append(path.name + "/")
    return found


def main():
    failures = []
    tracked = git_ls_files()

    failures.extend(check_required("core files", CORE_FILES))
    failures.extend(check_required("evaluators", EVALUATORS))
    failures.extend(check_required("collectors", COLLECTORS))
    failures.extend(check_required("Windows scripts", WINDOWS_SCRIPTS))
    failures.extend(check_required("Linux scripts", LINUX_SCRIPTS))

    cache_tracked = tracked_cache_files(tracked)
    if cache_tracked:
        print("[FAIL] Tracked cache files/directories:")
        for item in cache_tracked[:50]:
            print("  - {}".format(item))
        if len(cache_tracked) > 50:
            print("  ... {} more".format(len(cache_tracked) - 50))
        failures.append("tracked cache files")
    else:
        print("[OK] No tracked cache files detected")

    artifact_tracked = tracked_artifacts(tracked)
    if artifact_tracked:
        print("[WARN] Tracked generated artifacts or weights:")
        for item in artifact_tracked[:50]:
            print("  - {}".format(item))
        if len(artifact_tracked) > 50:
            print("  ... {} more".format(len(artifact_tracked) - 50))
    else:
        print("[OK] No tracked generated artifacts detected")

    cache_present = present_cache_dirs()
    if cache_present:
        print("[WARN] Cache directories present locally:")
        for item in cache_present[:50]:
            print("  - {}".format(item))
        if len(cache_present) > 50:
            print("  ... {} more".format(len(cache_present) - 50))

    root_artifacts = root_artifact_warnings()
    if root_artifacts:
        print("[WARN] Root-level generated artifacts present locally:")
        for item in root_artifacts:
            print("  - {}".format(item))

    if failures:
        print("AUDIT FAILED")
        raise SystemExit(1)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
