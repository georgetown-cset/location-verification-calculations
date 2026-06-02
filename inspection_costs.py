from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from inspection_costs_stage1 import (
    ALL_M_VALS,
    CLUSTER_SIZES,
    DETECTION_LOOKUP_TABLE_PARQUET_PATH,
    K_VALS,
    MIX_STEPS,
    N_VALS,
    PARQUET_COMPRESSION,
    PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION,
    PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP,
    PHYSICAL_INSPECTION_M_VALS,
    PLV_M_VALS,
    PLV_OWNING_COST_PER_TOTAL_CHIP,
    PLV_RENTING_COST_PER_TOTAL_CHIP,
    SCENARIOS_COMBINED_PARQUET_PATH,
    SCENARIOS_COSTED_PARQUET_PATH,
    SCENARIOS_PARQUET_PATH,
    OUTPUT_DIR,
    RELATIONSHIP_BOXPLOT_IMAGES_DIR,
    RELATIONSHIP_BOXPLOT_VALUES_CSV_PATH,
    RELATIONSHIP_MODEL_TEXT_PATH,
    RELATIONSHIP_SUMMARY_CSV_PATH,
    TARGET_CHIPS,
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
