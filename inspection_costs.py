from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Shared constants for all inspection-cost workflow stages live here so the
# stage modules can import them from a single source of truth.
TARGET_CHIPS = 3_000_000  # Total number of chips in the scenarios, used to size mixes.
SHARE_OF_CLUSTERS_WITH_SMUGGLING = 0.25  # Share of clusters in a scenario component assumed to contain smuggling.
NUMBER_OF_PHYSICAL_INSPECTIONS_PER_CLUSTER_PER_YEAR = 2

PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP = (8.3, 48.8)
PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION = (2545, 5900)
PLV_RENTING_COST_PER_TOTAL_CHIP = (2_251_688 / TARGET_CHIPS, 72_427_200 / TARGET_CHIPS)  # 12 - 500 landmark servers
PLV_OWNING_COST_PER_TOTAL_CHIP = (3_028_862 / TARGET_CHIPS, 28_715_814 / TARGET_CHIPS)  # 12 - 500 landmark servers
PLV_DISCOUNT_RATE = 0.5  # Discount rate applied to PLV benefits.
MIN_SHARE_DIVERTED = 0.1  # Minimum allowed share diverted (K / N) for each scenario component in mix scenarios.

MIX_STEP_SIZE = 0.1
CLUSTER_SIZES = [10, 100, 1000, 10000, 100000]  # Sizes of clusters to consider in mixes.
K_VALS = [0, 1, 10, 100, 1000, 10000, 100000]  # Diverted chips in a cluster with smuggling.
N_VALS = [0, 1, 10, 100, 1000]  # Tests conducted on a cluster.
PHYSICAL_INSPECTION_M_VALS = [0.05]  # Miss probability for a physical inspection test.
PLV_M_VALS = [0.1]  # Miss probability for a PLV test.

MIX_STEPS = np.linspace(
    0.0,
    1.0,
    int(round(1.0 / MIX_STEP_SIZE)) + 1,
)
ALL_M_VALS = sorted(set(PHYSICAL_INSPECTION_M_VALS) | set(PLV_M_VALS))
OUTPUT_DIR = "output"
DATA_SAVED_DIR = "data/saved"
PARQUET_COMPRESSION = "zstd"
DETECTION_LOOKUP_TABLE_PARQUET_PATH = f"{DATA_SAVED_DIR}/detection_lookup_table.parquet"
SCENARIOS_PARQUET_PATH = "data/scenario_components"
SCENARIOS_COMBINED_PARQUET_PATH = f"{DATA_SAVED_DIR}/scenarios_combined.parquet"
SCENARIOS_COSTED_PARQUET_PATH = f"{DATA_SAVED_DIR}/scenarios_costed.parquet"
RELATIONSHIP_SUMMARY_CSV_PATH = f"{OUTPUT_DIR}/relationship_summary.csv"
RELATIONSHIP_BOXPLOT_VALUES_CSV_PATH = f"{OUTPUT_DIR}/relationship_boxplot_values.csv"
RELATIONSHIP_BOXPLOT_IMAGES_DIR = f"{OUTPUT_DIR}/images"
RELATIONSHIP_MODEL_TEXT_PATH = f"{OUTPUT_DIR}/relationship_code_model.txt"
PLV_VARIANT_ORDER = ["PLV Renting", "PLV Owning"]

COLUMN_NAMES = {
    "mix_id": "Mix ID",
    "scenario_id": "Scenario ID",
    "mix_description": "Mix Description",
    "total_clusters_in_mix": "Total Clusters in Mix",
    "total_tests": "Total Tests",
    "total_component_chips": "Total Component Chips",
    "scenario_component_count": "Scenario Component Count",
    "k_combo": "K Combo",
    "n_combo": "N Combo",
    "cluster_size": "Cluster Size (N)",
    "bad_records": "Bad Records (K)",
    "total_bad_records": "Total Bad Records",
    "number_of_clusters": "Number of Clusters",
    "number_of_clusters_with_smuggling": "Number of Clusters with Smuggling",
    "tests": "Tests (n)",
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
    "benefit_per_dollar_relationship_description": "Benefit Per Dollar Relationship Description",
    "benefit_per_dollar_scenario_count": "Benefit Per Dollar Scenario Count",
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
    COLUMN_NAMES["n_combo"],
    COLUMN_NAMES["cluster_size"],
    COLUMN_NAMES["bad_records"],
    COLUMN_NAMES["total_bad_records"],
    COLUMN_NAMES["number_of_clusters"],
    COLUMN_NAMES["number_of_clusters_with_smuggling"],
    COLUMN_NAMES["tests"],
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
    "Physical - Min Benefit Per Dollar",
    "Physical - Max Benefit Per Dollar",
    "PLV Renting - Min Benefit Per Dollar",
    "PLV Renting - Max Benefit Per Dollar",
    "PLV Owning - Min Benefit Per Dollar",
    "PLV Owning - Max Benefit Per Dollar",
)
LONG_SCENARIO_BENEFIT_SCENARIOS = {
    "Physical - Min Benefit Per Dollar": "Physical (Conservative)",
    "Physical - Max Benefit Per Dollar": "Physical (Optimistic)",
}


def _scenario_variant_label(scenario_type: str, plv_type: str | None = None) -> str:
    if scenario_type.startswith("Physical"):
        return LONG_SCENARIO_BENEFIT_SCENARIOS[scenario_type]
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
BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN = "Physical vs PLV Benefit Per Dollar Relationship"
BENEFIT_PER_DOLLAR_RELATIONSHIP_LABELS = {
    "a": "Physical max benefit per dollar is less than PLV min benefit per dollar",
    "b": "Physical max benefit per dollar is within PLV range and Physical min benefit per dollar is below PLV min benefit per dollar",
    "c": "Physical min and max benefit per dollar are both within PLV range",
    "d": "Physical min benefit per dollar is within PLV range and Physical max benefit per dollar is above PLV max benefit per dollar",
    "e": "Physical min benefit per dollar is greater than PLV max benefit per dollar",
    "f": "Physical benefit per dollar range spans both sides of PLV range",
}
RELATIONSHIP_CODE_ORDER = list(BENEFIT_PER_DOLLAR_RELATIONSHIP_LABELS.keys())


def _plv_relationship_labels(plv_type: str) -> dict[str, str]:
    return {
        "a": f"Physical max benefit per dollar is less than {plv_type} min benefit per dollar",
        "b": f"Physical max benefit per dollar is within {plv_type} range and Physical min benefit per dollar is below {plv_type} min benefit per dollar",
        "c": f"Physical min and max benefit per dollar are both within {plv_type} range",
        "d": f"Physical min benefit per dollar is within {plv_type} range and Physical max benefit per dollar is above {plv_type} max benefit per dollar",
        "e": f"Physical min benefit per dollar is greater than {plv_type} max benefit per dollar",
        "f": f"Physical benefit per dollar range spans both sides of {plv_type} range",
    }


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


RELATIONSHIP_MODEL_BASE_FEATURE_COLUMNS = [
    COLUMN_NAMES["total_clusters_in_mix"],
    COLUMN_NAMES["scenario_component_count"],
    COLUMN_NAMES["total_tests"],
    COLUMN_NAMES["total_component_chips"],
    COLUMN_NAMES["bad_records"],
    COLUMN_NAMES["share_diverted"],
    COLUMN_NAMES["physical_inspection_total_diverted_chips_identified"],
    "PLV Renting - Total Diverted Chips Identified",
    "PLV Owning - Total Diverted Chips Identified",
    "Physical Inspection - Min Total Cost",
    "Physical Inspection - Max Total Cost",
    "PLV Renting - Min Total Cost",
    "PLV Renting - Max Total Cost",
    "PLV Owning - Min Total Cost",
    "PLV Owning - Max Total Cost",
]

RELATIONSHIP_MODEL_FEATURE_COLUMNS = [
    "log_total_clusters_in_mix",
    "scenario_component_count",
    "log_total_tests",
    "log_bad_records",
    "share_diverted",
    "tests_per_cluster",
    "bad_records_per_test",
    "physical_identified_share",
    "plv_renting_identified_share",
    "plv_owning_identified_share",
    "log_physical_detected",
    "log_plv_renting_detected",
    "log_plv_owning_detected",
    "plv_renting_minus_physical_detected_share",
    "plv_owning_minus_physical_detected_share",
    "physical_to_plv_renting_detected_ratio",
    "physical_to_plv_owning_detected_ratio",
    "physical_min_cost_per_test",
    "physical_cost_range_per_test",
    "plv_renting_cost_per_chip",
    "plv_owning_cost_per_chip",
    "plv_cost_spread_per_chip",
]

from inspection_costs_stage1 import (
    _overview_csv_file,
    _overview_dataframe,
    _overview_parquet_dataset,
    _overview_parquet_file,
    _reset_workflow_artifacts,
    build_detection_lookup_table,
    build_scenarios,
)
from inspection_costs_stage2 import _derive_scenario_summary_from_components, add_cost_benefit_columns
from inspection_costs_stage3 import (
    _summarize_relationships_from_parquet,
    _write_relationship_boxplot_values_from_parquet,
    write_relationship_code_model_report,
)


def run_inspection_costs_workflow(
    *,
    cluster_sizes: Iterable[int] = CLUSTER_SIZES,
    k_vals: Iterable[int] = K_VALS,
    n_vals: Iterable[int] = N_VALS,
    physical_m_vals: Iterable[float] = PHYSICAL_INSPECTION_M_VALS,
    plv_m_vals: Iterable[float] = PLV_M_VALS,
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = MIX_STEPS,
    phys_inspection_salary_cost_per_tested_chip: tuple[float, float] = PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP,
    phys_inspection_fixed_cost_per_inspection: tuple[float, float] = PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION,
    plv_renting_cost_per_total_chip: tuple[float, float] = PLV_RENTING_COST_PER_TOTAL_CHIP,
    plv_owning_cost_per_total_chip: tuple[float, float] = PLV_OWNING_COST_PER_TOTAL_CHIP,
) -> dict[str, object]:
    cluster_sizes = list(cluster_sizes)
    k_vals = list(k_vals)
    n_vals = list(n_vals)
    physical_m_vals = list(physical_m_vals)
    plv_m_vals = list(plv_m_vals)
    steps = list(steps) if steps is not None else None
    scenarios_combined_path = Path(SCENARIOS_COMBINED_PARQUET_PATH)
    scenarios_costed_path = Path(SCENARIOS_COSTED_PARQUET_PATH)
    relationship_summary_path = Path(RELATIONSHIP_SUMMARY_CSV_PATH)
    relationship_boxplot_values_path = Path(RELATIONSHIP_BOXPLOT_VALUES_CSV_PATH)
    relationship_boxplot_images_path = Path(RELATIONSHIP_BOXPLOT_IMAGES_DIR)
    relationship_model_text_path = Path(RELATIONSHIP_MODEL_TEXT_PATH)
    detection_lookup_table = None
    scenario_component_df = None
    final_df = None
    relationship_summary_df = None
    relationship_boxplot_df = None
    relationship_model_text = None
    start_time = time.perf_counter()

    print("Starting inspection costs workflow")
    print("Clearing prior workflow artifacts from data and output directories")
    _reset_workflow_artifacts()
    print(f"Ensuring detection lookup table exists at {DETECTION_LOOKUP_TABLE_PARQUET_PATH}")
    detection_lookup_table = build_detection_lookup_table(
        cluster_sizes=cluster_sizes,
        K_vals=k_vals,
        n_vals=n_vals,
        m_vals=ALL_M_VALS,
    )
    _overview_dataframe("Detection lookup table", detection_lookup_table)

    if not scenarios_combined_path.exists():
        print(f"Scenario-component combined parquet is missing; building {scenarios_combined_path}")
        print("Building scenario-component rows")
        scenario_component_df = build_scenarios(
            cluster_sizes=cluster_sizes,
            detection_lookup_table=detection_lookup_table,
            k_vals=k_vals,
            physical_m_vals=physical_m_vals,
            plv_m_vals=plv_m_vals,
            target_chips=target_chips,
            steps=steps,
            return_dataframe=True,
        )
        _overview_dataframe("Scenario-component rows", scenario_component_df)

        scenarios_combined_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving scenario-component rows to {scenarios_combined_path}")
        scenario_component_df.to_parquet(
            scenarios_combined_path,
            index=False,
            compression=PARQUET_COMPRESSION,
        )

    _overview_parquet_dataset(Path(SCENARIOS_PARQUET_PATH), "Scenario-component parquet dataset")
    _overview_parquet_file(scenarios_combined_path, "Combined scenario-component parquet")

    if not scenarios_costed_path.exists():
        print(f"Costed scenarios parquet is missing; building {scenarios_costed_path}")
        if scenario_component_df is None:
            print(f"Loading scenario-component rows from {scenarios_combined_path}")
            scenario_component_df = pd.read_parquet(scenarios_combined_path)
        if scenario_component_df is None:
            raise ValueError("scenario_component_df is required when costed scenarios are not saved")
        print("Aggregating scenario-component rows to scenario level")
        scenario_df = _derive_scenario_summary_from_components(scenario_component_df)
        _overview_dataframe("Scenario-level rows", scenario_df)
        print("Adding cost and benefit columns")
        final_df = add_cost_benefit_columns(
            scenario_df,
            phys_inspection_salary_cost_per_tested_chip=phys_inspection_salary_cost_per_tested_chip,
            phys_inspection_fixed_cost_per_inspection=phys_inspection_fixed_cost_per_inspection,
            plv_renting_cost_per_total_chip=plv_renting_cost_per_total_chip,
            plv_owning_cost_per_total_chip=plv_owning_cost_per_total_chip,
        )
        scenarios_costed_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving final scenarios to {scenarios_costed_path}")
        final_df.to_parquet(scenarios_costed_path, index=False, compression=PARQUET_COMPRESSION)

    _overview_parquet_file(scenarios_costed_path, "Costed scenarios parquet")

    print(f"Generating relationship summary at {relationship_summary_path}")
    relationship_summary_df = _summarize_relationships_from_parquet(scenarios_costed_path)
    relationship_summary_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing relationship summary CSV to {relationship_summary_path}")
    relationship_summary_df.to_csv(relationship_summary_path, index=False)
    _overview_dataframe("Relationship summary", relationship_summary_df)

    print(f"Generating relationship boxplot summary at {relationship_boxplot_values_path}")
    relationship_boxplot_df = _write_relationship_boxplot_values_from_parquet(
        scenarios_costed_path,
        relationship_boxplot_values_path,
        relationship_boxplot_images_path,
    )
    _overview_dataframe("Relationship boxplot values", relationship_boxplot_df)

    relationship_model_text = write_relationship_code_model_report(
        scenarios_costed_path,
        relationship_model_text_path,
    )
    _overview_csv_file(relationship_summary_path, "Relationship summary CSV")
    _overview_csv_file(relationship_boxplot_values_path, "Relationship boxplot values CSV")
    elapsed_seconds = time.perf_counter() - start_time
    print(f"Inspection costs workflow complete in {elapsed_seconds:.2f} seconds")
    return {
        "scenario_component_df": scenario_component_df,
        "final_df": final_df,
        "relationship_summary_df": relationship_summary_df,
        "relationship_boxplot_df": relationship_boxplot_df,
        "relationship_model_text": relationship_model_text,
    }


if __name__ == "__main__":
    run_inspection_costs_workflow()
