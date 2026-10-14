"""
Split the preprocessed SHIFT-WING .zarr samples into train/ and val/ subdirectories
(80/20, random, fixed seed — no repeated geometry across samples, so no need to
stratify). Moves (does not copy) each sample's .zarr directory into place, and
writes a manifest recording exactly which sample went where, for reproducibility.

Usage:
    python split_train_val.py <zarr_dir> --val-fraction 0.2 --seed 0
"""

import json
import random
import shutil
from pathlib import Path


def split_samples(zarr_dir: Path, val_fraction: float, seed: int) -> dict:
    samples = sorted(p for p in zarr_dir.iterdir() if p.is_dir() and p.suffix == ".zarr")
    if not samples:
        raise ValueError(f"No .zarr directories found directly in {zarr_dir}")

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    n_val = round(len(shuffled) * val_fraction)
    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]

    return {"train": train_samples, "val": val_samples}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_dir", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Only print the split, don't move files")
    args = parser.parse_args()

    split = split_samples(args.zarr_dir, args.val_fraction, args.seed)
    print(f"Total: {sum(len(v) for v in split.values())} samples")
    print(f"  train: {len(split['train'])}")
    print(f"  val:   {len(split['val'])}")

    manifest = {
        phase: [p.name for p in paths] for phase, paths in split.items()
    }
    manifest["seed"] = args.seed
    manifest["val_fraction"] = args.val_fraction

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
    else:
        for phase, paths in split.items():
            phase_dir = args.zarr_dir / phase
            phase_dir.mkdir(exist_ok=True)
            for p in paths:
                shutil.move(str(p), str(phase_dir / p.name))

        manifest_path = args.zarr_dir / "split_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Moved samples into {args.zarr_dir}/train and {args.zarr_dir}/val")
        print(f"Manifest saved to {manifest_path}")
