from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCE_URL = "https://epoch.ai/data/gpu_clusters.csv"
OUTPUT_DIR = Path("output")
OUTPUT_CSV_PATH = OUTPUT_DIR / "gpu_cluster_bucket_counts.csv"


def load_cluster_quantities() -> pd.Series:
    """Download the Epoch AI cluster dataset and return one chip-quantity series."""
    df = pd.read_csv(SOURCE_URL)

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


def build_bucket_summary(quantities: pd.Series) -> pd.DataFrame:
    """Count clusters in each chip-quantity bucket."""
    buckets = [
        ("100 <= X < 1,000", 100, 1_000),
        ("1,000 <= X < 10,000", 1_000, 10_000),
        ("10,000 <= X < 100,000", 10_000, 100_000),
        ("X >= 100,000", 100_000, None),
    ]

    rows = []
    valid_quantities = quantities.dropna()

    for bucket_label, lower_bound, upper_bound in buckets:
        if upper_bound is None:
            mask = valid_quantities >= lower_bound
        else:
            mask = (valid_quantities >= lower_bound) & (valid_quantities < upper_bound)

        rows.append(
            {
                "bucket": bucket_label,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound if upper_bound is not None else "",
                "cluster_count": int(mask.sum()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    quantities = load_cluster_quantities()
    summary_df = build_bucket_summary(quantities)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Chip quantity bucket counts")
    print(summary_df.to_string(index=False))
    print(f"\nWrote bucket summary to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
