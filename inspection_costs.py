from __future__ import annotations

import time
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Shared constants for all inspection-cost workflow stages live here so the
# stage modules can import them from a single source of truth.
TARGET_CHIPS = 3_000_000  # Total number of chips in the scenarios, used to size mixes.
SHARE_OF_CLUSTERS_WITH_SMUGGLING = 0.25  # Share of clusters in a scenario component assumed to contain smuggling.
NUMBER_OF_PHYSICAL_INSPECTIONS_PER_CLUSTER_PER_YEAR = 2

PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP = (9, 48.8)
PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION = (2_590, 6_050)
PLV_RENTING_COST_PER_TOTAL_CHIP = (2_251_688 / TARGET_CHIPS, 72_427_200 / TARGET_CHIPS)  # 12 - 500 landmark servers
PLV_OWNING_COST_PER_TOTAL_CHIP = (3_028_862 / TARGET_CHIPS, 28_715_814 / TARGET_CHIPS)  # 12 - 500 landmark servers
PLV_DISCOUNT_RATE = 0.5  # Fractional multiplier applied to PLV value.
MIN_DIVERTED_CHIPS = 100  # Minimum allowed diverted-chip count (K) for positive-K components of each scenario.
MIN_SCENARIO_DIVERTED_CHIPS = 114_000  # Minimum number of total diverted chips across a full scenario.

MIX_STEP_SIZE = 7  # Number of equal steps when iterating through mixes with different shares of clusters with smuggling.
CLUSTER_SIZES = [10, 100, 1_000, 10_000, 100_000]  # Sizes of clusters to consider in mixes.
K_VALS = [0, 10, 100, 1_000, 10_000, 100_000]  # Diverted chips in a cluster with smuggling.
CHIPS_INSPECTED_PER_CLUSTER_VALS = [0, 1, 10, 100, 1000]  # Chips inspected per cluster.
PHYSICAL_INSPECTION_M_VALS = [0.05]  # Failure probability for a physical inspection test.
PLV_M_VALS = [0.1]  # Failure probability for a PLV test.
ADDITIONAL_MIN_CLUSTERS_BY_SIZE: dict[int, int] = {
    # One relevant cluster is present in Epoch AI's Frontier Data Centers but missing from Epoch's GPU Clusters dataset.
    # Relevant cluster means a GPU cluster that is:
        # (a) not located in the U.S. or China, 
        # (b) is not owned by a U.S. cloud service provider or AI lab, 
        # (c) is currently operational or expected to be operational by the end of 2026, and
        # (d) deploys (or will deploy) export controlled chips
    # Missing clusters: DayOne Nusajaya
    # Epoch AI estimates that DayOne Nusajaya will have ~179k export controlled chips by Oct 2026.
    100_000: 1,
    # We allow for scenarios with zero size-10 clusters.
    10: 0
}

MIX_STEPS = np.linspace(
    0.0,
    1.0,
    num=int(MIX_STEP_SIZE) + 1,
)
ALL_M_VALS = sorted(set(PHYSICAL_INSPECTION_M_VALS) | set(PLV_M_VALS))
WORKFLOW_CONSTANT_NAMES_THROUGH_ALL_M_VALS = (
    "TARGET_CHIPS",
    "SHARE_OF_CLUSTERS_WITH_SMUGGLING",
    "NUMBER_OF_PHYSICAL_INSPECTIONS_PER_CLUSTER_PER_YEAR",
    "PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP",
    "PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION",
    "PLV_RENTING_COST_PER_TOTAL_CHIP",
    "PLV_OWNING_COST_PER_TOTAL_CHIP",
    "PLV_DISCOUNT_RATE",
    "MIN_DIVERTED_CHIPS",
    "MIN_SCENARIO_DIVERTED_CHIPS",
    "MIX_STEP_SIZE",
    "CLUSTER_SIZES",
    "K_VALS",
    "CHIPS_INSPECTED_PER_CLUSTER_VALS",
    "PHYSICAL_INSPECTION_M_VALS",
    "PLV_M_VALS",
    "ADDITIONAL_MIN_CLUSTERS_BY_SIZE",
    "MIX_STEPS",
    "ALL_M_VALS",
)
OUTPUT_DIR = "output"
ALL_OUTPUT_DIR = f"{OUTPUT_DIR}/all"
PERFECT_INFORMATION_OUTPUT_DIR = f"{OUTPUT_DIR}/perfect_information"
DATA_SAVED_DIR = "data/saved"
PARQUET_COMPRESSION = "zstd"
DETECTION_LOOKUP_TABLE_PARQUET_PATH = f"{DATA_SAVED_DIR}/detection_lookup_table.parquet"
SCENARIOS_PARQUET_PATH = "data/scenario_components"
SCENARIOS_COMBINED_PARQUET_PATH = f"{DATA_SAVED_DIR}/scenarios_combined.parquet"
SCENARIOS_COSTED_PARQUET_PATH = f"{DATA_SAVED_DIR}/scenarios_costed.parquet"
SCENARIOS_COSTED_PERFECT_INFORMATION_PARQUET_PATH = f"{DATA_SAVED_DIR}/scenarios_costed_perfect_information.parquet"
RELATIONSHIP_SUMMARY_CSV_PATH = f"{ALL_OUTPUT_DIR}/relationship_summary.csv"
RELATIONSHIP_BOXPLOT_VALUES_CSV_PATH = f"{ALL_OUTPUT_DIR}/relationship_boxplot_values.csv"
RELATIONSHIP_RULES_CSV_PATH = f"{ALL_OUTPUT_DIR}/relationship_code_rules.csv"
WORKFLOW_CONSTANTS_TEXT_PATH = f"{OUTPUT_DIR}/inspection_cost_assumptions.txt"
MIXES_USED_CSV_PATH = f"{OUTPUT_DIR}/mixes_used.csv"
RELATIONSHIP_BOXPLOT_IMAGES_DIR = f"{ALL_OUTPUT_DIR}/images"
PERFECT_INFORMATION_RELATIONSHIP_SUMMARY_CSV_PATH = f"{PERFECT_INFORMATION_OUTPUT_DIR}/relationship_summary.csv"
PERFECT_INFORMATION_RELATIONSHIP_BOXPLOT_VALUES_CSV_PATH = f"{PERFECT_INFORMATION_OUTPUT_DIR}/relationship_boxplot_values.csv"
PERFECT_INFORMATION_RELATIONSHIP_RULES_CSV_PATH = f"{PERFECT_INFORMATION_OUTPUT_DIR}/relationship_code_rules.csv"
PERFECT_INFORMATION_RELATIONSHIP_BOXPLOT_IMAGES_DIR = f"{PERFECT_INFORMATION_OUTPUT_DIR}/images"
PLV_VARIANT_ORDER = ["PLV Renting", "PLV Owning"]

def _workflow_stage(stage: str) -> None:
    print(f"\n=== {stage} ===")


def _workflow_log(stage: str, message: str, *, kind: str = "INFO") -> None:
    normalized_kind = kind.upper()
    completion_kinds = {"DONE", "END", "COMPLETE", "COMPLETED", "FINISH", "FINISHED"}
    if normalized_kind in completion_kinds and stage != "Workflow":
        return
    print(f"[{normalized_kind:<5} | {stage}] {message}")


def _format_column_preview(columns: Iterable[object], max_columns: int = 8) -> str:
    column_list = [str(column) for column in columns]
    if not column_list:
        return "(none)"
    preview = ", ".join(column_list[:max_columns])
    remaining = len(column_list) - max_columns
    if remaining > 0:
        preview += f", ... (+{remaining} more)"
    return preview


GPU_CLUSTER_BUCKET_COUNTS_CSV_PATH = Path(OUTPUT_DIR) / "gpu_cluster_bucket_counts.csv"
_GPU_CLUSTER_BUCKET_COUNTS_READY = False


# Regenerate the GPU cluster bucket-count CSV and return the path used by later stages.
def ensure_gpu_cluster_bucket_counts() -> Path:
    global _GPU_CLUSTER_BUCKET_COUNTS_READY
    csv_path = GPU_CLUSTER_BUCKET_COUNTS_CSV_PATH
    if _GPU_CLUSTER_BUCKET_COUNTS_READY and csv_path.exists():
        return csv_path
    script_path = Path(__file__).with_name("gpu_cluster_bucket_counts.py")
    subprocess.run([sys.executable, str(script_path)], check=True)
    _GPU_CLUSTER_BUCKET_COUNTS_READY = True
    return csv_path


# Load minimum cluster-count constraints by cluster-size bucket from the generated CSV.
def load_min_clusters_by_size_from_csv(
    csv_path: str | Path = GPU_CLUSTER_BUCKET_COUNTS_CSV_PATH,
) -> dict[int, int]:
    csv_path = Path(csv_path)
    bucket_counts_df = pd.read_csv(csv_path)
    required_columns = {"lower_bound", "unique_name_count"}
    missing_columns = required_columns - set(bucket_counts_df.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing_columns)}"
        )
    
    result: dict[int, int] = {}
    for lower_bound, cluster_count in bucket_counts_df.loc[:, ["lower_bound", "unique_name_count"]].itertuples(index=False, name=None):
        if pd.isna(lower_bound) or pd.isna(cluster_count):
            continue
        result[int(lower_bound)] = int(cluster_count)
    return result


# Build the cluster-size minimums used to keep generated mixes consistent with known GPU clusters.
def get_min_clusters_by_size() -> dict[int, int]:
    return load_min_clusters_by_size_from_csv()

COLUMN_NAMES = {
    "mix_id": "Mix ID",
    "scenario_id": "Scenario ID",
    "mix_description": "Mix Description",
    "total_clusters_in_mix": "Total Clusters in Mix",
    "total_tests": "Total Tests",
    "total_component_chips": "Total Component Chips",
    "scenario_component_count": "Scenario Component Count",
    "k_combo": "K Combo",
    "chips_inspected_per_cluster_combo": "Chips Inspected Per Cluster Combo",
    "cluster_size": "Cluster Size (N)",
    "bad_records": "Bad Records (K)",
    "total_bad_records": "Total Bad Records",
    "number_of_clusters": "Number of Clusters",
    "number_of_clusters_with_smuggling": "Number of Clusters with Smuggling",
    "chips_inspected_per_cluster": "Chips Inspected per Cluster",
    "share_diverted": "Share Diverted",
    "chip_level_miss_prob": "Chip-level Miss Prob (m)",
    "physical_inspection_chip_level_miss_prob": "Physical Inspection - Chip-level Miss Prob (m)",
    "plv_chip_level_miss_prob": "PLV - Chip-level Miss Prob (m)",
    "physical_inspection_p_detect": "Physical Inspection - P(Detect)",
    "physical_inspection_diverted_chips_identified": "Physical Inspection - Diverted Chips Identified",
    "plv_p_detect": "PLV - P(Detect)",
    "plv_diverted_chips_identified": "PLV - Diverted Chips Identified",
    "physical_inspection_total_diverted_chips_identified": "Physical Inspection - Total Diverted Chips Identified",
    "plv_total_diverted_chips_identified": "PLV - Total Diverted Chips Identified",
    "metric_family": "Metric Family",
    "relationship_code": "Relationship Code",
    "relationship_description": "Relationship Description",
    "scenario_group": "Scenario Group",
    "scenario_variant": "Scenario Variant",
    "scenario_type": "Scenario Type",
    "value": "Value",
    "scenario_count": "Scenario Count",
    "plv_type": "PLV Type",
    "value_per_cost_relationship_description": "Value Per Cost Relationship Description",
    "value_per_cost_scenario_count": "Value Per Cost Scenario Count",
}
SCENARIO_COLUMNS = [
    COLUMN_NAMES["mix_id"],
    COLUMN_NAMES["scenario_id"],
    COLUMN_NAMES["mix_description"],
    COLUMN_NAMES["total_clusters_in_mix"],
    COLUMN_NAMES["total_tests"],
    COLUMN_NAMES["total_component_chips"],
    COLUMN_NAMES["scenario_component_count"],
    COLUMN_NAMES["k_combo"],
    COLUMN_NAMES["chips_inspected_per_cluster_combo"],
    COLUMN_NAMES["cluster_size"],
    COLUMN_NAMES["bad_records"],
    COLUMN_NAMES["total_bad_records"],
    COLUMN_NAMES["number_of_clusters"],
    COLUMN_NAMES["number_of_clusters_with_smuggling"],
    COLUMN_NAMES["chips_inspected_per_cluster"],
    COLUMN_NAMES["physical_inspection_chip_level_miss_prob"],
    COLUMN_NAMES["physical_inspection_p_detect"],
    COLUMN_NAMES["physical_inspection_diverted_chips_identified"],
    COLUMN_NAMES["plv_chip_level_miss_prob"],
    COLUMN_NAMES["plv_p_detect"],
    COLUMN_NAMES["plv_diverted_chips_identified"],
    COLUMN_NAMES["physical_inspection_total_diverted_chips_identified"],
    COLUMN_NAMES["plv_total_diverted_chips_identified"],
]
LONG_SCENARIO_VALUE_VARS = (
    "Physical - Min Value Per Cost",
    "Physical - Max Value Per Cost",
    "PLV Renting - Min Value Per Cost",
    "PLV Renting - Max Value Per Cost",
    "PLV Owning - Min Value Per Cost",
    "PLV Owning - Max Value Per Cost",
)
LONG_SCENARIO_VALUE_SCENARIOS = {
    "Physical - Min Value Per Cost": "Physical (Conservative)",
    "Physical - Max Value Per Cost": "Physical (Optimistic)",
}


# Convert an internal scenario metric column name into a label used in long-form outputs.
def _scenario_variant_label(scenario_type: str, plv_type: str | None = None) -> str:
    if scenario_type.startswith("Physical"):
        return LONG_SCENARIO_VALUE_SCENARIOS[scenario_type]
    if scenario_type.startswith("PLV"):
        if plv_type is None:
            plv_type = "PLV"
        suffix = "Conservative" if "Min" in scenario_type else "Optimistic"
        return f"{plv_type} ({suffix})"
    if scenario_type == SMUGGLED_CHIPS_SCENARIO_TYPE:
        return SMUGGLED_CHIPS_SCENARIO_VARIANT
    return scenario_type


SMUGGLED_CHIPS_METRIC_FAMILY = "Smuggled Chips"
SMUGGLED_CHIPS_SCENARIO_GROUP = "Scenario Total"
SMUGGLED_CHIPS_SCENARIO_VARIANT = "Smuggled Chips"
SMUGGLED_CHIPS_SCENARIO_TYPE = "Total Smuggled Chips"
VALUE_PER_COST_RELATIONSHIP_COLUMN = "Physical vs PLV Value Per Cost Relationship"
VALUE_PER_COST_RELATIONSHIP_LABELS = {
    "a": "Physical max value per cost is less than PLV min value per cost",
    "b": "Physical max value per cost is within PLV range and Physical min value per cost is below PLV min value per cost",
    "c": "Physical min and max value per cost are both within PLV range",
    "d": "Physical min value per cost is within PLV range and Physical max value per cost is above PLV max value per cost",
    "e": "Physical min value per cost is greater than PLV max value per cost",
    "f": "Physical value per cost range spans both sides of PLV range",
}
RELATIONSHIP_CODE_ORDER = list(VALUE_PER_COST_RELATIONSHIP_LABELS.keys())


# Return relationship-code descriptions customized for the selected PLV cost variant.
def _plv_relationship_labels(plv_type: str) -> dict[str, str]:
    return {
        "a": f"Physical max value per cost is less than {plv_type} min value per cost",
        "b": f"Physical max value per cost is within {plv_type} range and Physical min value per cost is below {plv_type} min value per cost",
        "c": f"Physical min and max value per cost are both within {plv_type} range",
        "d": f"Physical min value per cost is within {plv_type} range and Physical max value per cost is above {plv_type} max value per cost",
        "e": f"Physical min value per cost is greater than {plv_type} max value per cost",
        "f": f"Physical value per cost range spans both sides of {plv_type} range",
    }


# Define the PLV cost variants that should be costed and summarized throughout the pipeline.
def _plv_variant_specs() -> list[dict[str, object]]:
    return [
        {
            "plv_type": "PLV Renting",
            "cost_per_total_chip": PLV_RENTING_COST_PER_TOTAL_CHIP,
        },
        {
            "plv_type": "PLV Owning",
            "cost_per_total_chip": PLV_OWNING_COST_PER_TOTAL_CHIP,
        },
    ]


# Generate the summary CSV, boxplot-value CSV, and boxplot images for one scenario parquet.
def _write_relationship_outputs(
    *,
    parquet_path: Path,
    summary_path: Path,
    boxplot_values_path: Path,
    boxplot_images_path: Path,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from inspection_costs_stage1 import _overview_csv_file, _overview_dataframe
    from inspection_costs_stage3 import _write_relationship_outputs_from_parquet

    _workflow_log("Stage 3 / Relationships", f"Generating {label} relationship summary at {summary_path}", kind="START")
    _workflow_log("Stage 3 / Boxplots", f"Generating {label} relationship boxplot summary at {boxplot_values_path}", kind="START")
    relationship_summary_df, relationship_boxplot_df = _write_relationship_outputs_from_parquet(
        parquet_path=parquet_path,
        summary_csv_path=summary_path,
        boxplot_csv_path=boxplot_values_path,
        output_images_dir=boxplot_images_path,
    )
    _overview_dataframe(f"{label} relationship summary", relationship_summary_df)
    _overview_csv_file(summary_path, f"{label} relationship summary CSV")

    _overview_dataframe(f"{label} relationship boxplot values", relationship_boxplot_df)
    _overview_csv_file(boxplot_values_path, f"{label} relationship boxplot values CSV")
    return relationship_summary_df, relationship_boxplot_df


# Generate the relationship-code rule CSV from scenario-level and component-level dataframes.
def _write_relationship_code_rules(
    *,
    scenario_df: pd.DataFrame,
    scenario_component_df: pd.DataFrame,
    output_csv_path: Path,
    label: str,
) -> pd.DataFrame:
    from inspection_costs_stage1 import _overview_csv_file, _overview_dataframe
    from inspection_costs_stage3 import _write_relationship_code_rules_from_dataframes

    _workflow_log("Stage 3 / Rules", f"Generating {label} relationship code rules at {output_csv_path}", kind="START")
    relationship_rules_df = _write_relationship_code_rules_from_dataframes(
        scenario_df=scenario_df,
        scenario_component_df=scenario_component_df,
        output_csv_path=output_csv_path,
        label=label,
    )
    _overview_dataframe(f"{label} relationship code rules", relationship_rules_df)
    _overview_csv_file(output_csv_path, f"{label} relationship code rules CSV")
    return relationship_rules_df


# Orchestrate the full inspection-cost workflow from detection lookup through final relationship outputs.
def run_inspection_costs_workflow(
    *,
    cluster_sizes: Iterable[int] = CLUSTER_SIZES,
    k_vals: Iterable[int] = K_VALS,
    min_scenario_diverted_chips: int = MIN_SCENARIO_DIVERTED_CHIPS,
    chips_inspected_per_cluster_vals: Iterable[int] = CHIPS_INSPECTED_PER_CLUSTER_VALS,
    extra_min_clusters_by_size: Optional[dict[int, int]] = ADDITIONAL_MIN_CLUSTERS_BY_SIZE,
    physical_m_vals: Iterable[float] = PHYSICAL_INSPECTION_M_VALS,
    plv_m_vals: Iterable[float] = PLV_M_VALS,
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = MIX_STEPS,
    phys_inspection_salary_cost_per_tested_chip: tuple[float, float] = PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP,
    phys_inspection_fixed_cost_per_inspection: tuple[float, float] = PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION,
    plv_renting_cost_per_total_chip: tuple[float, float] = PLV_RENTING_COST_PER_TOTAL_CHIP,
    plv_owning_cost_per_total_chip: tuple[float, float] = PLV_OWNING_COST_PER_TOTAL_CHIP,
) -> dict[str, object]:
    ensure_gpu_cluster_bucket_counts()
    min_clusters_by_size = get_min_clusters_by_size()
    if extra_min_clusters_by_size is not None:
        # Merge hand-added cluster minimums with the constraints derived from the source GPU dataset.
        for cluster_size, min_count in extra_min_clusters_by_size.items():
            cluster_size = int(cluster_size)
            min_count = int(min_count)
            min_clusters_by_size[cluster_size] = min_clusters_by_size.get(cluster_size, 0) + min_count

    from inspection_costs_stage1 import (
        _overview_dataframe,
        _overview_parquet_dataset,
        _overview_parquet_file,
        _reset_workflow_artifacts,
        build_detection_lookup_table,
        build_mix_data,
        build_scenarios,
        write_mix_data_csv,
    )
    from inspection_costs_stage2 import (
        _derive_scenario_summary_from_components,
        add_cost_value_columns,
        filter_perfect_information_scenarios,
    )
    from inspection_costs_stage3 import write_workflow_constants_report

    cluster_sizes = list(cluster_sizes)
    k_vals = list(k_vals)
    chips_inspected_per_cluster_vals = list(chips_inspected_per_cluster_vals)
    physical_m_vals = list(physical_m_vals)
    plv_m_vals = list(plv_m_vals)
    detection_m_vals = sorted(set(physical_m_vals) | set(plv_m_vals))
    steps = list(steps) if steps is not None else None

    # Resolve all output locations once so the rest of the workflow can pass concrete Path objects.
    scenarios_combined_path = Path(SCENARIOS_COMBINED_PARQUET_PATH)
    scenarios_costed_path = Path(SCENARIOS_COSTED_PARQUET_PATH)
    scenarios_costed_perfect_information_path = Path(SCENARIOS_COSTED_PERFECT_INFORMATION_PARQUET_PATH)
    relationship_summary_path = Path(RELATIONSHIP_SUMMARY_CSV_PATH)
    relationship_boxplot_values_path = Path(RELATIONSHIP_BOXPLOT_VALUES_CSV_PATH)
    relationship_rules_path = Path(RELATIONSHIP_RULES_CSV_PATH)
    relationship_boxplot_images_path = Path(RELATIONSHIP_BOXPLOT_IMAGES_DIR)
    perfect_information_relationship_summary_path = Path(PERFECT_INFORMATION_RELATIONSHIP_SUMMARY_CSV_PATH)
    perfect_information_relationship_boxplot_values_path = Path(PERFECT_INFORMATION_RELATIONSHIP_BOXPLOT_VALUES_CSV_PATH)
    perfect_information_relationship_rules_path = Path(PERFECT_INFORMATION_RELATIONSHIP_RULES_CSV_PATH)
    perfect_information_relationship_boxplot_images_path = Path(PERFECT_INFORMATION_RELATIONSHIP_BOXPLOT_IMAGES_DIR)
    detection_lookup_table = None
    scenario_component_df = None
    final_df = None
    perfect_information_final_df = None
    relationship_summary_df = None
    relationship_boxplot_df = None
    relationship_rules_df = None
    perfect_information_relationship_summary_df = None
    perfect_information_relationship_boxplot_df = None
    perfect_information_relationship_rules_df = None
    start_time = time.perf_counter()

    _workflow_stage("Inspection Costs Workflow")
    _workflow_log("Workflow", "Starting inspection costs workflow", kind="START")
    _workflow_log("Workflow", "Clearing prior workflow artifacts from data and output directories", kind="STEP")
    _reset_workflow_artifacts()

    # Stage 1 starts with a lookup table for detection probability and expected identified diversion.
    _workflow_stage("Stage 1 / Detection Lookup")
    _workflow_log("Stage 1 / Detection Lookup", f"Ensuring detection lookup table exists at {DETECTION_LOOKUP_TABLE_PARQUET_PATH}", kind="START")
    detection_lookup_table = build_detection_lookup_table(
        cluster_sizes=cluster_sizes,
        K_vals=k_vals,
        chips_inspected_per_cluster_vals=chips_inspected_per_cluster_vals,
        m_vals=detection_m_vals,
    )
    _overview_dataframe("Detection lookup table", detection_lookup_table)

    _workflow_stage("Stage 1 / Scenario Components")
    _workflow_log("Stage 1 / Scenario Components", "Building scenario-component rows", kind="START")
    mix_data = build_mix_data(
        cluster_sizes=cluster_sizes,
        target_chips=target_chips,
        steps=steps,
        min_clusters_by_size=min_clusters_by_size,
    )
    write_mix_data_csv(
        cluster_sizes=cluster_sizes,
        target_chips=target_chips,
        steps=steps,
        min_clusters_by_size=min_clusters_by_size,
        mix_data=mix_data,
    )
    scenario_component_df = build_scenarios(
        cluster_sizes=cluster_sizes,
        detection_lookup_table=detection_lookup_table,
        k_vals=k_vals,
        min_scenario_diverted_chips=min_scenario_diverted_chips,
        min_clusters_by_size=min_clusters_by_size,
        physical_m_vals=physical_m_vals,
        plv_m_vals=plv_m_vals,
        target_chips=target_chips,
        steps=steps,
        mix_data=mix_data,
        return_dataframe=True,
    )
    _overview_dataframe("Scenario-component rows", scenario_component_df)
    scenarios_combined_path.parent.mkdir(parents=True, exist_ok=True)
    _workflow_log("Stage 1 / Scenario Components", f"Saving scenario-component rows to {scenarios_combined_path}", kind="STEP")
    scenario_component_df.to_parquet(
        scenarios_combined_path,
        index=False,
        compression=PARQUET_COMPRESSION,
    )

    _overview_parquet_dataset(Path(SCENARIOS_PARQUET_PATH), "Scenario-component parquet dataset")
    _overview_parquet_file(scenarios_combined_path, "Combined scenario-component parquet")

    # Stage 2 collapses component rows to one row per scenario, then adds cost and value metrics.
    _workflow_stage("Stage 2 / Scenario Summary")
    _workflow_log("Stage 2 / Scenario Summary", "Aggregating scenario-component rows to scenario level", kind="START")
    scenario_df = _derive_scenario_summary_from_components(scenario_component_df)
    _overview_dataframe("Scenario-level rows", scenario_df)
    _workflow_log("Stage 2 / Scenario Summary", "Adding cost and value columns", kind="STEP")
    final_df = add_cost_value_columns(
        scenario_df,
        phys_inspection_salary_cost_per_tested_chip=phys_inspection_salary_cost_per_tested_chip,
        phys_inspection_fixed_cost_per_inspection=phys_inspection_fixed_cost_per_inspection,
        plv_renting_cost_per_total_chip=plv_renting_cost_per_total_chip,
        plv_owning_cost_per_total_chip=plv_owning_cost_per_total_chip,
    )
    scenarios_costed_path.parent.mkdir(parents=True, exist_ok=True)
    _workflow_log("Stage 2 / Scenario Summary", f"Saving final scenarios to {scenarios_costed_path}", kind="STEP")
    final_df.to_parquet(scenarios_costed_path, index=False, compression=PARQUET_COMPRESSION)

    _overview_parquet_file(scenarios_costed_path, "Costed scenarios parquet")

    _workflow_log("Stage 2 / Scenario Summary", "Filtering costed scenarios for perfect-information subset", kind="STEP")
    perfect_information_final_df = filter_perfect_information_scenarios(final_df)
    scenarios_costed_perfect_information_path.parent.mkdir(parents=True, exist_ok=True)
    _workflow_log(
        "Stage 2 / Scenario Summary",
        f"Saving perfect-information scenarios to {scenarios_costed_perfect_information_path}",
        kind="STEP",
    )
    perfect_information_final_df.to_parquet(
        scenarios_costed_perfect_information_path,
        index=False,
        compression=PARQUET_COMPRESSION,
    )

    _overview_parquet_file(scenarios_costed_perfect_information_path, "Perfect-information scenarios parquet")

    # Stage 3 writes comparable relationship artifacts for all scenarios and the perfect-information subset.
    relationship_outputs = [
        {
            "label": "all",
            "parquet_path": scenarios_costed_path,
            "summary_path": relationship_summary_path,
            "boxplot_values_path": relationship_boxplot_values_path,
            "boxplot_images_path": relationship_boxplot_images_path,
        },
        {
            "label": "perfect-information",
            "parquet_path": scenarios_costed_perfect_information_path,
            "summary_path": perfect_information_relationship_summary_path,
            "boxplot_values_path": perfect_information_relationship_boxplot_values_path,
            "boxplot_images_path": perfect_information_relationship_boxplot_images_path,
        },
    ]
    relationship_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for output_config in relationship_outputs:
        # Run the same relationship-output bundle for each scenario population.
        relationship_results[output_config["label"]] = _write_relationship_outputs(
            parquet_path=output_config["parquet_path"],
            summary_path=output_config["summary_path"],
            boxplot_values_path=output_config["boxplot_values_path"],
            boxplot_images_path=output_config["boxplot_images_path"],
            label=output_config["label"],
        )

    relationship_summary_df, relationship_boxplot_df = relationship_results["all"]
    perfect_information_relationship_summary_df, perfect_information_relationship_boxplot_df = relationship_results["perfect-information"]

    # Rule summaries need both scenario totals and the component rows that produced those totals.
    relationship_rules_df = _write_relationship_code_rules(
        scenario_df=final_df,
        scenario_component_df=scenario_component_df,
        output_csv_path=relationship_rules_path,
        label="all",
    )
    perfect_information_relationship_rules_df = _write_relationship_code_rules(
        scenario_df=perfect_information_final_df,
        scenario_component_df=scenario_component_df,
        output_csv_path=perfect_information_relationship_rules_path,
        label="perfect-information",
    )
    write_workflow_constants_report(Path(WORKFLOW_CONSTANTS_TEXT_PATH))
    elapsed_seconds = time.perf_counter() - start_time
    _workflow_log("Workflow", f"Inspection costs workflow complete in {elapsed_seconds:.2f} seconds", kind="DONE")
    return {
        "scenario_component_df": scenario_component_df,
        "final_df": final_df,
        "perfect_information_final_df": perfect_information_final_df,
        "relationship_summary_df": relationship_summary_df,
        "relationship_boxplot_df": relationship_boxplot_df,
        "relationship_rules_df": relationship_rules_df,
        "perfect_information_relationship_summary_df": perfect_information_relationship_summary_df,
        "perfect_information_relationship_boxplot_df": perfect_information_relationship_boxplot_df,
        "perfect_information_relationship_rules_df": perfect_information_relationship_rules_df,
    }


if __name__ == "__main__":
    run_inspection_costs_workflow()
