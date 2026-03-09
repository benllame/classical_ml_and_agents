"""Download the Telco Customer Churn dataset from Kaggle.

Downloads the IBM Telco Customer Churn dataset — the standard benchmark
for churn prediction — containing 7,043 customers and 21 features
(demographics, account info, services subscribed).  Originally published
by IBM as a sample dataset for Watson Analytics.
Citation: IBM (2018), via Kaggle user blastchar.

Usage:
    python src/download_data.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

from src.config import RAW_CSV, RAW_DIR


def download_telco_churn() -> Path:
    """Download the Telco Customer Churn dataset via kagglehub.

    Uses ``kagglehub`` for automated, authenticated downloading.  The
    function is idempotent — if the CSV already exists at the expected
    path, the download is skipped entirely.  The ``ImportError`` fallback
    ensures the project works even without ``kagglehub`` installed by
    printing manual download instructions instead of crashing silently.

    Returns
    -------
    Path
        Path to the downloaded CSV file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_CSV.exists():
        logger.info(f"Dataset already exists at {RAW_CSV}")
        return RAW_CSV

    try:
        import kagglehub

        logger.info("Downloading dataset via kagglehub...")
        downloaded_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        downloaded_dir = Path(downloaded_path)

        # kagglehub downloads to a cache dir — find the CSV and copy to data/raw/
        csv_files = list(downloaded_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV found in {downloaded_dir}")

        source = csv_files[0]
        shutil.copy2(source, RAW_CSV)
        logger.success(f"Dataset saved to {RAW_CSV}")

    except ImportError:
        logger.warning(
            "kagglehub not installed. Please download manually:\n"
            "  1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "  2. Download WA_Fn-UseC_-Telco-Customer-Churn.csv\n"
            f"  3. Place it in {RAW_DIR}/"
        )
        raise

    return RAW_CSV


if __name__ == "__main__":
    download_telco_churn()
