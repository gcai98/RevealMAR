#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser("Create a model-only checkpoint")
    parser.add_argument("--src", required=True, help="Source checkpoint path")
    parser.add_argument("--dst_dir", required=True, help="Destination directory")
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    dst_dir = Path(args.dst_dir)
    if not src.exists():
        raise FileNotFoundError("Source checkpoint not found: {}".format(src))

    dst_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(str(src), map_location="cpu", weights_only=False)
    out = {key: checkpoint[key] for key in ("model", "model_ema") if key in checkpoint}
    if not out:
        raise KeyError("Checkpoint does not contain 'model' or 'model_ema': {}".format(src))

    dst = dst_dir / "checkpoint-last.pth"
    torch.save(out, str(dst))
    print("saved model-only checkpoint to {}".format(os.fspath(dst)))


if __name__ == "__main__":
    main()
