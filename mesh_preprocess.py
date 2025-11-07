import os
import sys
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import csv

# ---------- HELPER FUNCTIONS ----------
def load_vertices(path):
    mesh = trimesh.load(path, process=False)
    return np.asarray(mesh.vertices), mesh

def minmax_normalize(vertices):
    v_min, v_max = vertices.min(axis=0), vertices.max(axis=0)
    diff = np.where((v_max - v_min) == 0, 1e-9, v_max - v_min)
    normalized = (vertices - v_min) / diff
    meta = {"v_min": v_min, "v_max": v_max}
    return normalized, meta

def minmax_denormalize(normalized, meta):
    v_min, v_max = meta["v_min"], meta["v_max"]
    diff = np.where((v_max - v_min) == 0, 1e-9, v_max - v_min)
    return normalized * diff + v_min

def unit_sphere_normalize(vertices):
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    radius = np.max(np.linalg.norm(centered, axis=1))
    normalized = centered / radius
    meta = {"centroid": centroid, "radius": radius}
    return normalized, meta

def unit_sphere_denormalize(normalized, meta):
    return normalized * meta["radius"] + meta["centroid"]

def quantize(values, n_bins=1024, input_range=(0, 1)):
    a, b = input_range
    mapped = (values - a) / (b - a)
    mapped = np.clip(mapped, 0, 1)
    q = np.floor(mapped * (n_bins - 1)).astype(np.int32)
    return q

def dequantize(q, n_bins=1024, output_range=(0, 1)):
    a, b = output_range
    mapped = q / (n_bins - 1)
    return mapped * (b - a) + a

def mse(a, b):
    return np.mean((a - b) ** 2)

def mae(a, b):
    return np.mean(np.abs(a - b))

# ---------- PIPELINE ----------
def process_mesh(mesh_path, n_bins=1024):
    vertices, mesh_obj = load_vertices(mesh_path)
    mesh_name = os.path.basename(mesh_path)
    print(f"\nProcessing: {mesh_name} ({len(vertices)} vertices)")

    results = []

    for method in ["minmax", "unit_sphere"]:
        if method == "minmax":
            normalized, meta = minmax_normalize(vertices)
            q = quantize(normalized, n_bins=n_bins, input_range=(0, 1))
            deq = dequantize(q, n_bins=n_bins, output_range=(0, 1))
            reconstructed = minmax_denormalize(deq, meta)
        else:
            normalized, meta = unit_sphere_normalize(vertices)
            q = quantize(normalized, n_bins=n_bins, input_range=(-1, 1))
            deq = dequantize(q, n_bins=n_bins, output_range=(-1, 1))
            reconstructed = unit_sphere_denormalize(deq, meta)

        err_mse, err_mae = mse(vertices, reconstructed), mae(vertices, reconstructed)
        print(f"{method} → MSE={err_mse:.8f}, MAE={err_mae:.8f}")

        # Save reconstructed mesh
        out_name = f"reconstructed_{method}_{os.path.splitext(mesh_name)[0]}.ply"
        mesh_obj.vertices = reconstructed
        mesh_obj.export(out_name)

        results.append({
            "mesh": mesh_name,
            "method": method,
            "mse": err_mse,
            "mae": err_mae
        })

    return results

# ---------- MAIN LOOP ----------
def run_all(mesh_dir="meshes", n_bins=1024):
    all_results = []
    obj_files = [f for f in os.listdir(mesh_dir) if f.endswith(".obj")]

    for obj_file in obj_files:
        mesh_path = os.path.join(mesh_dir, obj_file)
        results = process_mesh(mesh_path, n_bins)
        all_results.extend(results)

    # Write summary CSV
    with open("results_summary.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["mesh", "method", "mse", "mae"])
        writer.writeheader()
        writer.writerows(all_results)

    print("\n✅ All meshes processed! Summary saved to results_summary.csv")
    for res in all_results:
        print(f"{res['mesh']:15} | {res['method']:10} | MSE={res['mse']:.8f} | MAE={res['mae']:.8f}")

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    mesh_dir = "meshes"
    if len(sys.argv) > 1:
        mesh_dir = sys.argv[1]
    run_all(mesh_dir)
