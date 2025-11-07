"""
Adaptive Quantization Experiment
--------------------------------
Implements rotation + translation invariant normalization
and adaptive quantization based on vertex density.
All results and plots are saved in the outputs/ folder.
"""

import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

def load_vertices(path):
    mesh = trimesh.load(path, process=False)
    return np.asarray(mesh.vertices), mesh

def normalize_unit_sphere(vertices):
    """Normalize vertices into unit sphere coordinates."""
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    radius = np.max(np.linalg.norm(centered, axis=1))
    normalized = centered / radius
    return normalized, centroid, radius

def denormalize_unit_sphere(vertices, centroid, radius):
    """Restore vertices back to original space."""
    return vertices * radius + centroid

def adaptive_quantize(vertices, n_bins=1024, k=10):
    """Adaptive quantization based on local vertex density."""
    tree = cKDTree(vertices)
    dists, _ = tree.query(vertices, k=k)
    density = 1.0 / (np.mean(dists, axis=1) + 1e-8)
    density_norm = (density - density.min()) / (density.max() - density.min() + 1e-8)
    adaptive_bins = (n_bins * (0.5 + 0.5 * density_norm)).astype(int)
    adaptive_bins = np.clip(adaptive_bins, 16, n_bins)

    quantized = []
    for i in range(vertices.shape[0]):
        v = np.clip(vertices[i], -1, 1)
        bin_size = 2 / adaptive_bins[i]
        q = np.floor((v + 1) / bin_size) * bin_size - 1
        quantized.append(q)
    return np.array(quantized)

def uniform_quantize(vertices, n_bins=1024):
    """Simple uniform quantization."""
    v = np.clip(vertices, -1, 1)
    bin_size = 2 / n_bins
    q = np.floor((v + 1) / bin_size) * bin_size - 1
    return q

def mse(a, b):
    return np.mean((a - b) ** 2)

def generate_transforms(vertices, num_versions=5):
    """Generate rotated & translated versions of the mesh."""
    versions = []
    for _ in range(num_versions):
        theta = np.random.rand() * 2 * np.pi
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        c, s = np.cos(theta), np.sin(theta)
        x, y, z = axis
        R = np.array([
            [c + x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
            [y*x*(1-c)+z*s, c + y*y*(1-c), y*z*(1-c)-x*s],
            [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c + z*z*(1-c)]
        ])
        t = np.random.uniform(-0.1, 0.1, 3)
        v_new = vertices @ R.T + t
        versions.append(v_new)
    return versions

def main():
    mesh_path = "meshes/branch.obj"
    vertices, mesh = load_vertices(mesh_path)

    print(f"Loaded mesh with {len(vertices)} vertices for adaptive quantization test.")

    versions = generate_transforms(vertices, num_versions=5)
    uniform_mses, adaptive_mses = [], []

    for i, v in enumerate(versions, 1):
        normalized, centroid, radius = normalize_unit_sphere(v)
        uniform_q = uniform_quantize(normalized)
        adaptive_q = adaptive_quantize(normalized)

        recon_uniform = denormalize_unit_sphere(uniform_q, centroid, radius)
        recon_adaptive = denormalize_unit_sphere(adaptive_q, centroid, radius)

        mse_u = mse(v, recon_uniform)
        mse_a = mse(v, recon_adaptive)

        uniform_mses.append(mse_u)
        adaptive_mses.append(mse_a)

        print(f"Version {i}: Uniform MSE={mse_u:.6e}, Adaptive MSE={mse_a:.6e}")

    avg_uniform = np.mean(uniform_mses)
    avg_adaptive = np.mean(adaptive_mses)

    print(f"\nAverage Uniform MSE: {avg_uniform}")
    print(f"Average Adaptive MSE: {avg_adaptive}")

    # Save results
    results_path = os.path.join("outputs", "adaptive_results.txt")
    with open(results_path, "w") as f:
        f.write("Adaptive Quantization Experiment Results\n")
        f.write("----------------------------------------\n")
        for i in range(len(uniform_mses)):
            f.write(f"Version {i+1}: Uniform={uniform_mses[i]:.6e}, Adaptive={adaptive_mses[i]:.6e}\n")
        f.write(f"\nAverage Uniform MSE: {avg_uniform}\n")
        f.write(f"Average Adaptive MSE: {avg_adaptive}\n")

    # Save plot
    plt.figure(figsize=(7, 5))
    plt.plot(uniform_mses, label="Uniform Quantization", marker="o")
    plt.plot(adaptive_mses, label="Adaptive Quantization", marker="s")
    plt.xlabel("Mesh Version")
    plt.ylabel("MSE")
    plt.title("Uniform vs Adaptive Quantization Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "adaptive_vs_uniform_error.png"))
    plt.close()

    print("Adaptive Quantization Experiment completed successfully.")
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
