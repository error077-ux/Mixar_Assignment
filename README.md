# Mesh Assignment Project

## Overview

This project implements a complete **3D Mesh Preprocessing and Quantization Pipeline**, including normalization, quantization, seam tokenization, and adaptive quantization analysis.
It evaluates mesh compression, reconstruction accuracy, and rotation/translation invariance using Python-based processing.

## Objectives

- Extract vertex coordinates from 3D `.obj` files.
- Normalize mesh data using:
  - **Min–Max Normalization**
  - **Unit Sphere Normalization**
- Quantize and dequantize vertex coordinates for mesh reconstruction.
- Compute reconstruction errors using **MSE** and **MAE** metrics.
- Compare normalization methods to determine which provides the least reconstruction error.
- **Bonus tasks:**
  - **Seam Tokenization Prototype**: Encode mesh seams as discrete tokens.
  - **Adaptive Quantization**: Implement rotation & translation invariant normalization and adaptive quantization based on vertex density.

## Project Structure

```
MeshAssignment/
│
├── meshes/                      # Input mesh (.obj) files
│   ├── branch.obj
│   ├── cylinder.obj
│   ├── fence.obj
│   ├── girl.obj
│   └── ...
│
├── outputs/                     # Generated results per run
│   ├── results_summary.csv
│   ├── reconstructed_*.ply
│   ├── seam_tokens.txt
│   ├── adaptive_results.txt
│   └── adaptive_vs_uniform_error.png
│
├── testing_output/              # Logs and verification reports
│   ├── run_log.txt
│   └── integrity_log.txt
│
├── mesh_preprocess.py           # Core mesh normalization and quantization
├── seam_tokenization.py         # Seam edge extraction and token encoding
├── adaptive_quantization.py     # Rotation & translation invariant adaptive quantization
├── test_project_integrity.py    # Automated validation of output structure and results
└── main.py                      # Master script to execute all components
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/MeshAssignment.git
cd MeshAssignment
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/Mac:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

To execute the full project pipeline:

```bash
python main.py
```

This command:

- Clears old results automatically.
- Runs all tasks in sequence.
- Saves new outputs in `outputs/`.
- Saves logs and integrity verification reports in `testing_output/`.

## Generated Outputs

| File | Description |
|------|-------------|
| `results_summary.csv` | Quantization results for all meshes (MSE & MAE). |
| `reconstructed_*.ply` | Reconstructed mesh files after quantization. |
| `seam_tokens.txt` | Encoded seam edges as tokenized strings. |
| `adaptive_results.txt` | Comparison of adaptive vs uniform quantization errors. |
| `adaptive_vs_uniform_error.png` | Graph comparing quantization accuracy. |
| `run_log.txt` | Execution logs for all stages. |
| `integrity_log.txt` | Results of automated project verification. |

## Key Concepts

### Normalization

- **Min–Max Normalization**: Scales vertex coordinates between 0 and 1.
- **Unit Sphere Normalization**: Normalizes vertices to fit inside a sphere of radius 1.

### Quantization and Dequantization

Continuous vertex coordinates are converted into discrete bins (quantized), stored efficiently, and later reconstructed (dequantized).

### Seam Tokenization

Encodes mesh seams (UV breaks) as sequential tokens for potential use in machine learning-based 3D mesh analysis.

### Adaptive Quantization

Quantization bin sizes vary based on local vertex density, ensuring higher precision in dense areas and reducing information loss.

## Example Output (Console)

```
========== MESH ASSIGNMENT PROJECT EXECUTION ==========

>>> Running: Core Mesh Preprocessing (Normalization + Quantization)
Core Mesh Preprocessing completed successfully.

>>> Running: Seam Tokenization Prototype (Bonus Task 1)
Seam Tokenization Prototype completed successfully.

>>> Running: Adaptive Quantization Experiment (Bonus Task 2)
Adaptive Quantization Experiment completed successfully.

>>> Running: Project Integrity Verification
Project Integrity Verification completed successfully.

=======================================================
All processes completed successfully.
Previous results cleared before execution.
Results saved in the 'outputs/' directory.
Logs saved in the 'testing_output/' directory.
=======================================================
```

## Requirements

- numpy
- scipy
- trimesh
- matplotlib
- psutil
- open3d

## Verification

To run only the project integrity test:

```bash
python test_project_integrity.py
```

This verifies:

- Validity of generated mesh files.
- Presence and correctness of output files.
- Structure and formatting of the results CSV.

## Conclusion

This project successfully demonstrates:

- Mesh normalization and quantization for geometric data compression.
- Seam tokenization for structured mesh representation.
- Adaptive quantization for improved reconstruction fidelity.
- A fully automated and reproducible workflow for mesh preprocessing.
