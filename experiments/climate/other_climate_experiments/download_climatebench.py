"""Download ClimateBench dataset from Zenodo to local storage.

Usage:
    python download_climatebench.py

Data is saved to /proj/heal_pangu/users/x_tagty/climatebench/
Source: https://zenodo.org/record/7064308
"""

import os
import tarfile
import urllib.request

DEST_DIR = "/proj/heal_pangu/users/x_tagty/climatebench"
ZENODO_BASE = "https://zenodo.org/record/7064308/files"
FILES = ["train_val.tar.gz", "test.tar.gz"]


def _progress(count, block_size, total_size):
    pct = min(count * block_size * 100 / total_size, 100)
    print(f"\r  {pct:.1f}%", end="", flush=True)


def download_and_extract(filename: str, dest_dir: str) -> None:
    url = f"{ZENODO_BASE}/{filename}"
    archive_path = os.path.join(dest_dir, filename)

    print(f"Downloading {filename} ...")
    urllib.request.urlretrieve(url, archive_path, reporthook=_progress)
    print()  # newline after progress

    print(f"Extracting {filename} ...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(path=dest_dir)

    os.remove(archive_path)
    print(f"Done: {filename}\n")


if __name__ == "__main__":
    os.makedirs(DEST_DIR, exist_ok=True)
    for f in FILES:
        download_and_extract(f, DEST_DIR)
    print(f"ClimateBench data ready at: {DEST_DIR}")
