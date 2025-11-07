"""
Bonus Task: Option 2 - Rotation & Translation Invariance + Adaptive Quantization
"""

import trimesh
import numpy as np
import matplotlib.pyplot as plt

# ---------- Utilities ----------
def unit_sphere_normalize(vertices):
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    radius = np.max(np.linalg.norm(centered, axis=1))
    normalized = centered / radius
    return normalized, {"centroid": centroid, "radius": radius}

def unit_sphere_denormalize(normalized, meta):
    return normalized * meta["radius"] + meta["centroid"]

def quantize_uniform(values, n_bins=1024, input_range=(-1, 1)):
    a, b = input_range
    mapped = (values - a) / (b - a)
    q = np.floor(mapped * (n_bins - 1)).astype(np.int32)
    return q

def dequantize_uniform(q, n_bins=1024, output_range=(-1, 1)):
    a, b = output_range
    mapped = q / (n_bins - 1)
    return mapped * (b - a) + a

# ---------- Adaptive Quantization ----------
def adaptive_quantize(values, densities, base_bins=1024, input_range=(-1, 1)):
    """
    values: normalized coordinates
    densities: local vertex densities (inverse of avg distance)
    """
    a, b = input_range
    mapped = (values - a) / (b - a)
    mapped = np.clip(mapped, 0, 1)
    # normalize densities to [0.5, 1.5] bin scaling
    density_scale = (densities - densities.min()) / (densities.max() - densities.min() + 1e-9)
    bin_scale = 0.5 + density_scale  # 0.5 to 1.5 range
    adaptive_bins = np.clip((base_bins * bin_scale).astype(int), 128, 2048)
    # quantize each vertex adaptively
    q = np.floor(mapped * (adaptive_bins[:, None] - 1)).astype(np.int32)
    return q, adaptive_bins

def adaptive_dequantize(q, adaptive_bins, output_range=(-1, 1)):
    a, b = output_range
    mapped = q / (adaptive_bins[:, None] - 1)
    return mapped * (b - a) + a

# ---------- Local Density ----------
def compute_local_density(vertices, k=10):
    from scipy.spatial import cKDTree
    tree = cKDTree(vertices)
    dists, _ = tree.query(vertices, k=k)
    mean_dists = dists.mean(axis=1)
    density = 1.0 / (mean_dists + 1e-9)
    return density

# ---------- Transformations ----------
def random_transform(vertices):
    # Random rotation matrix
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    rotated = vertices @ R.T
    # Random translation
    t = np.random.uniform(-0.2, 0.2, size=(1, 3))
    translated = rotated + t
    return translated

# ---------- Error ----------
def mse(a, b):
    return np.mean((a - b) ** 2)

# ---------- Main ----------
def main():
    mesh = trimesh.load("meshes/branch.obj", process=False)
    vertices = np.asarray(mesh.vertices)

    uniform_errors = []
    adaptive_errors = []

    for i in range(5):  # generate 5 random transformations
        transformed = random_transform(vertices)
        normalized, meta = unit_sphere_normalize(transformed)

        # Uniform Quantization
        q_uniform = quantize_uniform(normalized)
        deq_uniform = dequantize_uniform(q_uniform)
        recon_uniform = unit_sphere_denormalize(deq_uniform, meta)

        # Adaptive Quantization
        densities = compute_local_density(normalized)
        q_adapt, bins_adapt = adaptive_quantize(normalized, densities)
        deq_adapt = adaptive_dequantize(q_adapt, bins_adapt)
        recon_adapt = unit_sphere_denormalize(deq_adapt, meta)

        # Errors
        err_uniform = mse(transformed, recon_uniform)
        err_adaptive = mse(transformed, recon_adapt)
        uniform_errors.append(err_uniform)
        adaptive_errors.append(err_adaptive)

        print(f"Version {i+1}: Uniform MSE={err_uniform:.6e}, Adaptive MSE={err_adaptive:.6e}")

    # Plot Comparison
    plt.figure(figsize=(6, 4))
    x = np.arange(5)
    plt.bar(x - 0.15, uniform_errors, width=0.3, label="Uniform Quantization")
    plt.bar(x + 0.15, adaptive_errors, width=0.3, label="Adaptive Quantization")
    plt.xlabel("Transformed Mesh Versions")
    plt.ylabel("Reconstruction MSE")
    plt.title("Uniform vs Adaptive Quantization Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("adaptive_vs_uniform_error.png")
    plt.show()

    # Summary
    avg_uniform = np.mean(uniform_errors)
    avg_adaptive = np.mean(adaptive_errors)
    print("\nAverage Uniform MSE:", avg_uniform)
    print("Average Adaptive MSE:", avg_adaptive)

    # Save results
    with open("adaptive_results.txt", "w") as f:
        f.write("Uniform MSEs:\n" + str(uniform_errors) + "\n")
        f.write("Adaptive MSEs:\n" + str(adaptive_errors) + "\n")
        f.write(f"\nAverage Uniform MSE: {avg_uniform}\n")
        f.write(f"Average Adaptive MSE: {avg_adaptive}\n")

    print("✅ Results saved to adaptive_results.txt and plot saved as adaptive_vs_uniform_error.png")

if __name__ == "__main__":
    main()
