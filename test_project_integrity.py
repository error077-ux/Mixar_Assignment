"""
Final Project Integrity Checker
--------------------------------
Verifies:
1. Core mesh preprocessing pipeline
2. Seam Tokenization (Bonus Task 1)
3. Adaptive Quantization (Bonus Task 2)
4. Output file presence and integrity
"""

import os
import subprocess
import trimesh
import csv
import sys
from datetime import datetime

def run_script(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            print(f"ERROR in {command}\n{result.stderr}")
            return False
        else:
            print(f"{command} ran successfully.")
            return True
    except Exception as e:
        print(f"Exception in {command}: {e}")
        return False

def check_file_exists(file_path):
    if os.path.exists(file_path):
        print(f"File exists: {file_path}")
        return True
    else:
        print(f"Missing: {file_path}")
        return False

def check_csv_validity(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return False
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            if headers != ["mesh", "method", "mse", "mae"]:
                print(f"Warning: CSV headers incorrect: {headers}")
                return False
        print("results_summary.csv is valid and formatted correctly.")
        return True
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

def check_mesh_load(path):
    try:
        mesh = trimesh.load(path)
        print(f"Mesh loaded successfully: {path} ({len(mesh.vertices)} vertices)")
        return True
    except Exception as e:
        print(f"Failed to load mesh {path}: {e}")
        return False

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("testing_output", exist_ok=True)
    log_file = os.path.join("testing_output", f"integrity_log_{timestamp}.txt")
    sys.stdout = open(log_file, "w", encoding="utf-8")

    print("Starting project integrity verification...\n")

    success = True

    if not run_script("python mesh_preprocess.py"):
        success = False
    else:
        check_file_exists("results_summary.csv")
        check_csv_validity("results_summary.csv")

    recon_files = [f for f in os.listdir(".") if f.startswith("reconstructed_") and f.endswith(".ply")]
    if recon_files:
        print(f"Found {len(recon_files)} reconstructed meshes.")
        check_mesh_load(recon_files[0])
    else:
        print("No reconstructed mesh files found.")
        success = False

    if not run_script("python seam_tokenization.py"):
        success = False
    else:
        check_file_exists("seam_tokens.txt")

    if not run_script("python adaptive_quantization.py"):
        success = False
    else:
        check_file_exists("adaptive_results.txt")
        check_file_exists("adaptive_vs_uniform_error.png")

    print("\nFinal Verification Summary:")
    if success:
        print("ALL CHECKS PASSED! Your project is fully functional and ready for submission.")
    else:
        print("Some checks failed. Review logs above to fix issues before submitting.")

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"Integrity test results saved to {log_file}")

if __name__ == "__main__":
    main()
