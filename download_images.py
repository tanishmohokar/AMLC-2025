"""
Image pre-downloader (optional, step 1).

Reads train.csv and test.csv, and downloads every product image in parallel
into IMAGE_CACHE_DIR. The Qwen scripts read from the same cache, so running
this first removes network latency from fine-tuning, embedding extraction
and inference.

Usage: python download_images.py
"""
import os
from io import BytesIO
from multiprocessing import Pool

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from common import IMAGE_CACHE_DIR, TEST_CSV, TRAIN_CSV, image_cache_path


def download_image(args):
    """Downloads and saves a single image, skipping files that already exist."""
    url, sample_id = args
    if pd.isna(url):
        return f"Skipped: {sample_id}"

    cache_path = image_cache_path(sample_id, IMAGE_CACHE_DIR)
    if os.path.exists(cache_path):
        return f"Exists: {sample_id}"

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        Image.open(BytesIO(response.content)).convert("RGB").save(cache_path)
        return f"Downloaded: {sample_id}"
    except Exception as e:
        return f"Failed: {sample_id} | Error: {e}"


if __name__ == "__main__":
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    print(f"Image cache directory: '{IMAGE_CACHE_DIR}'")

    combined_df = pd.concat([
        pd.read_csv(TRAIN_CSV, usecols=["sample_id", "image_link"]),
        pd.read_csv(TEST_CSV, usecols=["sample_id", "image_link"]),
    ])
    tasks = list(zip(combined_df["image_link"], combined_df["sample_id"]))
    print(f"Found {len(tasks)} images to fetch.")

    num_processes = os.cpu_count()
    print(f"Starting download with {num_processes} parallel processes...")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(download_image, tasks), total=len(tasks), desc="Downloading Images"))

    print("\n--- Download Summary ---")
    for status in ("Downloaded", "Exists", "Skipped", "Failed"):
        print(f"{status}: {sum(1 for r in results if r.startswith(status))}")
