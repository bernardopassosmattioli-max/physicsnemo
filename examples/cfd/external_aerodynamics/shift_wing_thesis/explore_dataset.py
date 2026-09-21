"""
Characterize the SHIFT-WING sample dataset: read params.json + forces.json for every
sample and produce summary plots + a CSV table, so we can see what parameter space
(flow conditions, geometry, resulting aero coefficients) we are actually feeding the
GeoTransolver.

This only reads the small JSON files per sample (not the multi-hundred-MB .vtp/.stl),
so it is cheap and safe to run directly on a login node -- no PBS job needed.

Usage:
    python explore_dataset.py <raw_dir> <output_dir>

Where <raw_dir> contains sample_XXXXXX/ subfolders, each with params.json and
forces.json (as downloaded from the SHIFT-WING Hugging Face dataset).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on the HPC login node
import matplotlib.pyplot as plt
import pandas as pd


def load_dataset_table(raw_dir: Path) -> pd.DataFrame:
    rows = []
    sample_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())

    for sample_dir in sample_dirs:
        params_path = sample_dir / "params.json"
        forces_path = sample_dir / "forces.json"
        if not params_path.exists():
            continue

        with open(params_path) as f:
            params = json.load(f)
        row = {"sample": sample_dir.name}
        row.update(params)

        if forces_path.exists():
            with open(forces_path) as f:
                forces = json.load(f)
            row.update(forces)

        rows.append(row)

    df = pd.DataFrame(rows)
    # params.json stores the geometry params as strings; make everything numeric
    # where possible so plotting/statistics work directly.
    for col in df.columns:
        if col == "sample":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Flow-condition coverage: what regimes does the dataset actually span? ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].hist(df["mach"], bins=20, color="steelblue", edgecolor="black")
    axes[0, 0].set_xlabel("Mach number")
    axes[0, 0].set_ylabel("Number of samples")
    axes[0, 0].set_title("Samples per Mach")

    axes[0, 1].hist(df["reynolds_number"], bins=20, color="darkorange", edgecolor="black")
    axes[0, 1].set_xlabel("Reynolds number")
    axes[0, 1].set_title("Samples per Reynolds number")
    axes[0, 1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    axes[0, 2].hist(df["alpha"], bins=20, color="seagreen", edgecolor="black")
    axes[0, 2].set_xlabel("Angle of attack, alpha (deg)")
    axes[0, 2].set_title("Samples per angle of attack")

    axes[1, 0].scatter(df["mach"], df["reynolds_number"], alpha=0.6, color="steelblue")
    axes[1, 0].set_xlabel("Mach number")
    axes[1, 0].set_ylabel("Reynolds number")
    axes[1, 0].set_title("Mach vs Reynolds (flight envelope covered)")

    axes[1, 1].hist(df["air_density"], bins=20, color="slategray", edgecolor="black")
    axes[1, 1].set_xlabel("Air density (kg/m^3)")
    axes[1, 1].set_title("Samples per air density (altitude proxy)")

    axes[1, 2].hist(df["temperature"], bins=20, color="firebrick", edgecolor="black")
    axes[1, 2].set_xlabel("Freestream temperature (K)")
    axes[1, 2].set_title("Samples per temperature")

    fig.suptitle("SHIFT-WING dataset: flow-condition coverage", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "flow_conditions.png", dpi=150)
    plt.close(fig)

    # --- Geometry coverage: how varied are the wing shapes? ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].hist(df["AR"], bins=20, color="mediumpurple", edgecolor="black")
    axes[0, 0].set_xlabel("Aspect ratio (AR)")
    axes[0, 0].set_title("Samples per aspect ratio")

    axes[0, 1].hist(df["S_ref_m2"], bins=20, color="teal", edgecolor="black")
    axes[0, 1].set_xlabel("Reference area, S_ref (m^2)")
    axes[0, 1].set_title("Samples per wing area")

    axes[0, 2].hist(
        df["leading_edge_sweep_deg"], bins=20, color="goldenrod", edgecolor="black"
    )
    axes[0, 2].set_xlabel("Leading-edge sweep (deg)")
    axes[0, 2].set_title("Samples per sweep angle")

    axes[1, 0].hist(df["lambda"], bins=20, color="indianred", edgecolor="black")
    axes[1, 0].set_xlabel("Taper ratio (lambda)")
    axes[1, 0].set_title("Samples per taper ratio")

    axes[1, 1].scatter(df["AR"], df["S_ref_m2"], alpha=0.6, color="mediumpurple")
    axes[1, 1].set_xlabel("Aspect ratio")
    axes[1, 1].set_ylabel("Reference area (m^2)")
    axes[1, 1].set_title("AR vs wing area (planform diversity)")

    axes[1, 2].hist(df["twist_r"], bins=20, color="chocolate", edgecolor="black")
    axes[1, 2].set_xlabel("Root twist (deg)")
    axes[1, 2].set_title("Samples per root twist")

    fig.suptitle("SHIFT-WING dataset: wing geometry coverage", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "geometry_coverage.png", dpi=150)
    plt.close(fig)

    # --- Output/label space: what is the model actually learning to predict? ---
    if "Cl" in df.columns and "Cd" in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        axes[0, 0].hist(df["Cl"], bins=20, color="steelblue", edgecolor="black")
        axes[0, 0].set_xlabel("Lift coefficient (Cl)")
        axes[0, 0].set_title("Samples per Cl")

        axes[0, 1].hist(df["Cd"], bins=20, color="darkorange", edgecolor="black")
        axes[0, 1].set_xlabel("Drag coefficient (Cd)")
        axes[0, 1].set_title("Samples per Cd")

        axes[1, 0].scatter(df["Cd"], df["Cl"], alpha=0.6, color="seagreen")
        axes[1, 0].set_xlabel("Cd")
        axes[1, 0].set_ylabel("Cl")
        axes[1, 0].set_title("Drag polar (Cl vs Cd) across samples")

        axes[1, 1].scatter(df["alpha"], df["Cl"], alpha=0.6, color="firebrick")
        axes[1, 1].set_xlabel("Angle of attack (deg)")
        axes[1, 1].set_ylabel("Cl")
        axes[1, 1].set_title("Lift curve across samples")

        fig.suptitle("SHIFT-WING dataset: aerodynamic output coverage", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "output_coverage.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    df = load_dataset_table(args.raw_dir)
    print(f"Loaded {len(df)} samples")
    print(df.describe().T[["count", "mean", "std", "min", "max"]])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "dataset_summary.csv", index=False)

    make_plots(df, args.output_dir)
    print(f"Wrote plots and dataset_summary.csv to {args.output_dir}")
