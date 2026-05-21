from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


EPOCH_GPU_CLUSTERS_ZIP_URL = "https://epoch.ai/data/gpu_clusters.zip"


def load_epoch_gpu_clusters_from_zip(zip_bytes: bytes, extract_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the first CSV found in an Epoch GPU clusters zip archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        csv_files = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_files:
            raise ValueError("No CSV file found in the ZIP archive.")

        if extract_dir is not None:
            archive.extractall(extract_dir)

        with archive.open(csv_files[0]) as csv_file:
            return pd.read_csv(csv_file)


def download_epoch_gpu_clusters(zip_url: str = EPOCH_GPU_CLUSTERS_ZIP_URL, extract_dir: str | Path | None = None) -> pd.DataFrame:
    """Download and load the Epoch GPU clusters dataset."""
    with urllib.request.urlopen(zip_url) as response:
        return load_epoch_gpu_clusters_from_zip(response.read(), extract_dir=extract_dir)


def filter_epoch_gpu_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a chip count and remove US and China clusters."""
    filtered = df[df["Total number of AI chips"].notna()]
    filtered = filtered[filtered["Total number of AI chips"] != ""]

    countries_to_exclude = {"United States of America", "China"}
    filtered = filtered[~filtered["Country"].isin(countries_to_exclude)]
    return filtered.reset_index(drop=True)


def load_and_filter_epoch_gpu_clusters(zip_url: str = EPOCH_GPU_CLUSTERS_ZIP_URL, extract_dir: str | Path | None = None) -> pd.DataFrame:
    """Convenience helper for the notebook workflow."""
    return filter_epoch_gpu_clusters(download_epoch_gpu_clusters(zip_url=zip_url, extract_dir=extract_dir))
