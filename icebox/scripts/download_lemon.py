import os
import requests
from bs4 import BeautifulSoup
import argparse
import urllib.parse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/"
DEST_DIR = "data/lemon"


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        logger.info(f"Skipping {os.path.basename(dest_path)}, already exists.")
        return

    logger.info(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info(f"Successfully downloaded {os.path.basename(dest_path)}.")


def main():
    parser = argparse.ArgumentParser(description="Download LEMON EEG Preprocessed Dataset")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of files to download (each subject has 4 files typically). Set to 0 for "
        "all.",
    )
    args = parser.parse_args()

    os.makedirs(DEST_DIR, exist_ok=True)

    logger.info(f"Fetching index from {BASE_URL}")
    response = requests.get(BASE_URL)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all links that end with .set or .fdt
    files_to_download = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and (href.endswith(".set") or href.endswith(".fdt")):
            files_to_download.append(href)

    if not files_to_download:
        logger.error("No .set or .fdt files found on the page.")
        return

    logger.info(f"Found {len(files_to_download)} files on the server.")

    if args.limit > 0:
        files_to_download = files_to_download[: args.limit]
        logger.info(f"Limiting download to the first {args.limit} files as requested.")

    for filename in files_to_download:
        # Some URLs might have special characters
        clean_filename = urllib.parse.unquote(filename)
        url = urllib.parse.urljoin(BASE_URL, filename)
        dest_path = os.path.join(DEST_DIR, clean_filename)

        try:
            download_file(url, dest_path)
        except Exception as e:
            logger.error(f"Failed to download {clean_filename}: {e}")


if __name__ == "__main__":
    main()
