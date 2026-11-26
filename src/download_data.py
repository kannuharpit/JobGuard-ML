"""
download_data.py
----------------
Downloads the "Real or Fake Job Posting Prediction" dataset from Kaggle
using the opendatasets library and places the CSV at data/fake_job_postings.csv.

Usage:
    python src/download_data.py

You will be prompted for your Kaggle username and API key the first time.
Get them from: https://www.kaggle.com/account → "Create New API Token"
"""

import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "fake_job_postings.csv")
KAGGLE_URL = "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction"

os.makedirs(DATA_DIR, exist_ok=True)

if os.path.exists(OUTPUT_CSV):
    print(f"[INFO] Dataset already exists at {OUTPUT_CSV}. Skipping download.")
    sys.exit(0)

try:
    import opendatasets as od
    print("[INFO] Downloading dataset via opendatasets ...")
    od.download(KAGGLE_URL, data_dir=DATA_DIR)
    # opendatasets creates a sub-folder; find the CSV
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".csv"):
                src = os.path.join(root, f)
                if src != OUTPUT_CSV:
                    print(f"[INFO] Moving {src} → {OUTPUT_CSV}")
                    shutil.move(src, OUTPUT_CSV)
    print(f"[OK] Dataset saved to {OUTPUT_CSV}")
except ImportError:
    print("[ERROR] opendatasets not installed. Run: pip install opendatasets")
    sys.exit(1)
