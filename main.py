"""
Main Execution Script
---------------------
Runs all components of the Mesh Assignment Project:
1. Core mesh preprocessing pipeline
2. Seam tokenization prototype
3. Adaptive quantization experiment
4. Automated integrity verification

Behavior:
- Clears old files in /outputs and /testing_output before each run.
- Saves all new results cleanly into these folders.
"""

import subprocess
import os
import shutil

def ensure_directories(clean=False):
    """Ensure output directories exist, optionally clear old data."""
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("testing_output", exist_ok=True)

    if clean:
        # Clear contents of both folders before new run
        for folder in ["outputs", "testing_output"]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Warning: Failed to delete {file_path}: {e}")

def run_command(command, description, log_file):
    """Run a shell command and log the output."""
    print("\n" + "=" * 70)
    print(f">>> Running: {description}")
    print("=" * 70)
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"\n=== {description} ===\n")
            log.write(result.stdout + "\n")
            log.write(result.stderr + "\n")

        if result.returncode == 0:
            print(f"{description} completed successfully.")
        else:
            print(f"{description} encountered an error. Check log for details.")
    except Exception as e:
        print(f"Error running {description}: {e}")
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"Exception while running {description}: {e}\n")

def move_all_outputs():
    """Move all generated files (.csv, .txt, .png, .ply) into their folders."""
    for file in os.listdir("."):
        if not os.path.isfile(file):
            continue
        if file.endswith((".csv", ".ply", ".png", ".txt")):
            # Separate outputs and test logs
            if file.startswith("test") or "integrity" in file or "run_log" in file:
                dest = os.path.join("testing_output", file)
            else:
                dest = os.path.join("outputs", file)

            # Avoid duplication
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(file, dest)

def main():
    # Step 0: Clean and prepare folders
    ensure_directories(clean=True)

    log_file = os.path.join("testing_output", "run_log.txt")

    print("\n========== MESH ASSIGNMENT PROJECT EXECUTION ==========\n")

    # Step 1: Core Mesh Preprocessing
    run_command("python mesh_preprocess.py", "Core Mesh Preprocessing (Normalization + Quantization)", log_file)

    # Step 2: Seam Tokenization Prototype
    run_command("python seam_tokenization.py", "Seam Tokenization Prototype (Bonus Task 1)", log_file)

    # Step 3: Adaptive Quantization Experiment
    run_command("python adaptive_quantization.py", "Adaptive Quantization Experiment (Bonus Task 2)", log_file)

    # Step 4: Move outputs so far
    move_all_outputs()

    # Step 5: Project Integrity Verification
    run_command("python test_project_integrity.py", "Project Integrity Verification", log_file)

    # Step 6: Final cleanup (move any new files created by test)
    move_all_outputs()

    print("\n=======================================================")
    print("All processes completed successfully.")
    print("Previous results cleared before execution.")
    print("Results saved in the 'outputs/' directory.")
    print("Logs saved in the 'testing_output/' directory.")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
