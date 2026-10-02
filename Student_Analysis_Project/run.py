"""
Main Application Launcher for Student Analysis Project.
Ensures data and Python models are initialized, then runs the web application server.
"""
import sys
import os
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from src.config import DATASET_FILE, HOST, PORT, DEBUG
from src.dataset_generator import generate_student_dataset
from src.train import train_all_models

def setup_environment():
    """Initializes data directory and trains models if needed."""
    print("=" * 65)
    print(" Initializing Student Performance Analytics & ML Platform")
    print("=" * 65)

    # 1. Dataset check
    if not DATASET_FILE.exists():
        print(f"[*] Dataset not found at {DATASET_FILE}.")
        print("[*] Generating reference dataset with 30,641 records...")
        generate_student_dataset(output_path=DATASET_FILE)
    else:
        print(f"[+] Dataset verified: {DATASET_FILE}")

    # 2. Models check
    print("[+] Using Python Model Modules in models/ folder:")
    print("    - models/linear_regression_model.py")
    print("    - models/random_forest_grade_model.py")
    train_all_models(DATASET_FILE)

def main():
    setup_environment()
    print("\n" + "=" * 65)
    print(f" Starting Web Server on http://{HOST}:{PORT}")
    print(f" Interactive Dashboard:  http://{HOST}:{PORT}")
    print(f" Swagger API Docs:       http://{HOST}:{PORT}/docs")
    print("=" * 65 + "\n")
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=DEBUG)

if __name__ == "__main__":
    main()
