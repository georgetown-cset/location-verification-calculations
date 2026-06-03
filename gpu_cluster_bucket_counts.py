from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


SOURCE_URL = "https://epoch.ai/data/gpu_clusters.csv"
DATA_DIR = Path("data")
SOURCE_CSV_PATH = DATA_DIR / "gpu_clusters.csv"
OUTPUT_DIR = Path("output")
OUTPUT_CSV_PATH = OUTPUT_DIR / "gpu_cluster_bucket_counts.csv"
EXCLUDED_OWNERS = {
    "amazon",
    "google",
    "google deepmind",
    "microsoft",
    "meta ai",
    "oracle",
    "openai",
    "stargate (openai)",
    "xai",
}
INCLUDED_PRIMARY_CHIP_TYPES = (
    "NVIDIA GB200",
    "NVIDIA B200",
    "NVIDIA H100",
    "NVIDIA GH200",
    "NVIDIA A100",
    "AMD MI300X",
    "AMD MI250X"
)


def download_source_csv() -> Path:
    """Download the source CSV once and reuse the cached local copy."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if SOURCE_CSV_PATH.exists():
        return SOURCE_CSV_PATH

    with urlopen(SOURCE_URL) as response, SOURCE_CSV_PATH.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)

    return SOURCE_CSV_PATH


def extract_cluster_quantities(df: pd.DataFrame) -> pd.Series:
    """Return a single chip-quantity series from the dataset."""
    quantity_columns = [
        "Total number of AI chips",
        "Chip quantity (primary)",
        "Chip quantity (secondary)",
    ]

    available_columns = [column for column in quantity_columns if column in df.columns]
    if not available_columns:
        raise ValueError(
            "Could not find any chip quantity columns in the downloaded dataset."
        )

    quantities = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for column in available_columns:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        quantities = quantities.fillna(numeric_values)

    return quantities


def filter_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the requested dataset filters before bucket counting."""
    filtered_df = df.copy()

    country_series = filtered_df["Country"].fillna("")
    owner_series = filtered_df["Owner"].fillna("")
    status_series = filtered_df["Status"].fillna("").astype(str)
    chip_type_series = filtered_df["Chip type (primary)"].fillna("").astype(str)

    country_mask = ~country_series.astype(str).isin(
        {"United States of America", "China"}
    )
    owner_mask = ~owner_series.astype(str).apply(
        lambda owner: any(
            part.strip().lower() in EXCLUDED_OWNERS
            for part in owner.split(",")
        )
    )
    status_mask = status_series.eq("Existing")
    chip_type_mask = chip_type_series.apply(
        lambda chip_type: any(
            chip_type.casefold().find(chip.casefold()) != -1
            for chip in INCLUDED_PRIMARY_CHIP_TYPES
        )
    )

    return filtered_df[country_mask & owner_mask & status_mask & chip_type_mask]


def build_bucket_summary(df: pd.DataFrame, quantities: pd.Series) -> pd.DataFrame:
    """Count unique cluster names in each chip-quantity bucket."""
    buckets = [
        ("10 <= X < 100", 10, 100),
        ("100 <= X < 1,000", 100, 1_000),
        ("1,000 <= X < 10,000", 1_000, 10_000),
        ("10,000 <= X < 100,000", 10_000, 100_000),
        ("X >= 100,000", 100_000, None),
    ]

    rows = []
    valid_quantities = quantities.dropna()
    valid_names = df.loc[valid_quantities.index, "Name"].fillna("").astype(str)

    for bucket_label, lower_bound, upper_bound in buckets:
        if upper_bound is None:
            mask = valid_quantities >= lower_bound
        else:
            mask = (valid_quantities >= lower_bound) & (valid_quantities < upper_bound)

        bucket_names = valid_names[mask]

        rows.append(
            {
                "bucket": bucket_label,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound if upper_bound is not None else "",
                "unique_name_count": int(bucket_names[bucket_names != ""].nunique()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    source_csv_path = download_source_csv()
    df = pd.read_csv(source_csv_path)
    filtered_df = filter_clusters(df)
    quantities = extract_cluster_quantities(filtered_df)
    summary_df = build_bucket_summary(filtered_df, quantities)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Unique Name counts by chip quantity bucket after filtering")
    print(f"Total clusters after filters: {len(filtered_df)}")
    print(summary_df.to_string(index=False))
    print(f"\nWrote bucket summary to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
