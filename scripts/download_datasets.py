"""
scripts/download_datasets.py
Downloads all three required datasets from Kaggle using kagglehub.
Run this ONCE before running embed_and_index.py.

Setup:
1. Create a Kaggle account at https://kaggle.com
2. Go to Settings → API → Create New Token
3. Save the downloaded kaggle.json to:
     Linux/Mac: ~/.kaggle/kaggle.json
     Windows:   C:/Users/<you>/.kaggle/kaggle.json
4. Run: python scripts/download_datasets.py

Datasets downloaded:
  - Amazon Reviews 2023 (subset: Electronics + Books)
  - Yelp Dataset (businesses + reviews)
  - Goodreads Books
"""
import os
import json
import shutil
import logging
from pathlib import Path

import kagglehub

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("kudiwise.downloader")

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─── Dataset definitions ─────────────────────────────────────────────────────
# Each entry: (kaggle_handle, description, output_filename_hint)

DATASETS = [
    {
        "handle": "snap/amazon-fine-food-reviews",
        "description": "Amazon Reviews (food — good proxy for student survival items)",
        "out_name": "amazon",
        "fallback": "mcpenguin/amazon-reviews-2018-converted",
    },
    {
        "handle": "yelp-dataset/yelp-dataset",
        "description": "Yelp Dataset (restaurants + businesses)",
        "out_name": "yelp",
        "fallback": "omkarborikar/yelp-reviews-dataset",
    },
    {
        "handle": "jealousleopard/goodreadsbooks",
        "description": "Goodreads Books",
        "out_name": "goodreads",
        "fallback": "bahramjannesari/goodreads-book-datasets-with-user-rating-10m",
    },
]


def download_dataset(handle: str, out_name: str) -> Path | None:
    """Download a Kaggle dataset and return its local path."""
    try:
        logger.info("Downloading: %s", handle)
        path = kagglehub.dataset_download(handle)
        dest = DATA_DIR / out_name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(path, dest)
        logger.info("✓ Saved to: %s", dest)
        return dest
    except Exception as exc:
        logger.warning("Failed (%s): %s", handle, exc)
        return None


def find_jsonl_or_csv(folder: Path) -> list[Path]:
    """Find all .json, .jsonl, or .csv files in a dataset folder."""
    files = []
    for ext in ["*.jsonl", "*.json", "*.csv"]:
        files += list(folder.rglob(ext))
    return files


def main():
    logger.info("=== KudiWise AI — Dataset Downloader ===")
    logger.info("Output directory: %s", DATA_DIR.resolve())

    # Check Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error(
            "kaggle.json not found at %s\n"
            "Steps:\n"
            "  1. Go to https://kaggle.com/settings\n"
            "  2. Click 'Create New Token'\n"
            "  3. Save kaggle.json to ~/.kaggle/kaggle.json\n"
            "  4. Run: chmod 600 ~/.kaggle/kaggle.json",
            kaggle_json,
        )
        return

    results = {}

    for ds in DATASETS:
        path = download_dataset(ds["handle"], ds["out_name"])

        # Try fallback if primary fails
        if path is None and ds.get("fallback"):
            logger.info("Trying fallback: %s", ds["fallback"])
            path = download_dataset(ds["fallback"], ds["out_name"])

        if path and path.exists():
            files = find_jsonl_or_csv(path)
            results[ds["out_name"]] = {
                "path": str(path),
                "files": [str(f) for f in files],
                "status": "ok",
            }
            logger.info(
                "✓ %s — %d files found: %s",
                ds["out_name"],
                len(files),
                [f.name for f in files[:3]],
            )
        else:
            results[ds["out_name"]] = {"status": "failed"}
            logger.error("✗ %s — download failed. See manual links below.", ds["out_name"])

    # Save manifest so embed_and_index knows where files are
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Manifest saved: %s", manifest_path)

    # Print next step
    logger.info("")
    logger.info("=== NEXT STEP ===")
    logger.info("Run the indexer:")
    logger.info(
        "  python scripts/embed_and_index.py "
        "--amazon data/raw/amazon "
        "--yelp data/raw/yelp "
        "--goodreads data/raw/goodreads "
        "--limit 50000"
    )


if __name__ == "__main__":
    main()
