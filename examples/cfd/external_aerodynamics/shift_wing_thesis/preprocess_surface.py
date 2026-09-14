"""
Convert a raw SHIFT-WING sample (merged_surfaces.vtp + a decimated STL + params.json)
into the per-sample .zarr layout expected by physicsnemo's TransolverDataPipe /
CAEDataset for surface-mode GeoTransolver training.

Expected raw sample directory (as downloaded from Hugging Face):
    sample_000001/
        merged_surfaces.vtp                # full-res surface mesh + CFD fields
        merged_surfaces_onshape_200k.stl   # decimated geometry-only mesh
        params.json                        # flow conditions (air_density, stream_velocity, ...)

Output:
    <output_dir>/sample_000001.zarr/
        surface_fields          (N_cells, 4)   float32   [pressure, wss_x, wss_y, wss_z]
        surface_mesh_centers    (N_cells, 3)   float32
        surface_normals         (N_cells, 3)   float32
        surface_areas           (N_cells,)     float32
        stl_coordinates         (N_pts, 3)     float32
        stl_faces               (N_faces * 3,) int32
        stl_centers             (N_faces, 3)   float32
        air_density             ()             float32
        stream_velocity         ()             float32
"""

import json
from pathlib import Path

import numpy as np
import pyvista as pv
import zarr


def _find_array(data, substrings: list[str]) -> np.ndarray:
    """Look up a point/cell data array by substring match (avoids unicode-superscript
    issues with names like "Wall Shear Stress (N/m²)")."""
    for name in data.keys():
        if all(s in name for s in substrings):
            return np.asarray(data[name])
    raise KeyError(f"No array matching {substrings} found among {list(data.keys())}")


def build_surface_fields(vtp_path: Path) -> dict[str, np.ndarray]:
    mesh = pv.read(str(vtp_path))

    pressure = _find_array(mesh.cell_data, ["Pressure"]).astype(np.float32)  # (N,)
    wss = _find_array(mesh.cell_data, ["Wall Shear Stress"]).astype(np.float32)  # (N, 3)
    surface_fields = np.concatenate([pressure[:, None], wss], axis=1)  # (N, 4)

    surface_mesh_centers = np.asarray(mesh.cell_centers().points, dtype=np.float32)

    normals = _find_array(mesh.cell_data, ["Normals"]).astype(np.float32)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    cell_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    surface_areas = np.asarray(cell_sizes.cell_data["Area"], dtype=np.float32)

    return {
        "surface_fields": surface_fields,
        "surface_mesh_centers": surface_mesh_centers,
        "surface_normals": normals,
        "surface_areas": surface_areas,
    }


def build_geometry(stl_path: Path) -> dict[str, np.ndarray]:
    mesh = pv.read(str(stl_path))

    stl_coordinates = np.asarray(mesh.points, dtype=np.float32)

    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # drop the leading "3" (verts-per-face) column
    stl_faces = faces.astype(np.int32).flatten()

    stl_centers = np.asarray(mesh.cell_centers().points, dtype=np.float32)

    return {
        "stl_coordinates": stl_coordinates,
        "stl_faces": stl_faces,
        "stl_centers": stl_centers,
    }


def preprocess_sample(sample_dir: Path, output_dir: Path) -> None:
    vtp_path = sample_dir / "merged_surfaces.vtp"
    stl_path = sample_dir / "merged_surfaces_onshape_200k.stl"
    params_path = sample_dir / "params.json"

    with open(params_path) as f:
        params = json.load(f)

    arrays = {}
    arrays.update(build_surface_fields(vtp_path))
    arrays.update(build_geometry(stl_path))
    arrays["air_density"] = np.float32(params["air_density"])
    arrays["stream_velocity"] = np.float32(params["stream_velocity"])

    out_path = output_dir / f"{sample_dir.name}.zarr"
    group = zarr.open_group(str(out_path), mode="w")
    for key, value in arrays.items():
        group[key] = np.asarray(value)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw_dir", type=Path, help="Directory containing sample_XXXXXX/ subfolders"
    )
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p for p in args.raw_dir.iterdir() if p.is_dir())
    for sample_dir in sample_dirs:
        preprocess_sample(sample_dir, args.output_dir)
