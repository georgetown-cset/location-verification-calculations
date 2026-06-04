from __future__ import annotations

import numpy as np
import pandas as pd

from inspection_costs import (
    COLUMN_NAMES,
    NUMBER_OF_PHYSICAL_INSPECTIONS_PER_CLUSTER_PER_YEAR,
    PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION,
    PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP,
    PLV_DISCOUNT_RATE,
    PLV_OWNING_COST_PER_TOTAL_CHIP,
    PLV_RENTING_COST_PER_TOTAL_CHIP,
    TARGET_CHIPS,
    _plv_variant_specs,
)


# Aggregate component-level rows into one scenario-level row with totals and identifying combos.
def _derive_scenario_summary_from_components(scenario_component_df: pd.DataFrame) -> pd.DataFrame:
    if scenario_component_df.empty:
        return pd.DataFrame(
            columns=[
                COLUMN_NAMES['mix_id'],
                COLUMN_NAMES['scenario_id'],
                COLUMN_NAMES['mix_description'],
                COLUMN_NAMES['total_clusters_in_mix'],
                COLUMN_NAMES['scenario_component_count'],
                COLUMN_NAMES['k_combo'],
                COLUMN_NAMES['n_combo'],
                COLUMN_NAMES['physical_inspection_chip_level_miss_prob'],
                COLUMN_NAMES['plv_chip_level_miss_prob'],
                COLUMN_NAMES['total_tests'],
                COLUMN_NAMES['total_component_chips'],
                COLUMN_NAMES['bad_records'],
                COLUMN_NAMES['number_of_clusters_with_smuggling'],
                COLUMN_NAMES['physical_inspection_total_diverted_chips_identified'],
                COLUMN_NAMES['plv_total_diverted_chips_identified'],
            ]
        )

    # Keep only the fields needed for scenario-level totals before grouping to reduce memory pressure.
    summary_source = scenario_component_df.loc[
        :,
        [
            COLUMN_NAMES['scenario_id'],
            COLUMN_NAMES['mix_id'],
            COLUMN_NAMES['mix_description'],
            COLUMN_NAMES['k_combo'],
            COLUMN_NAMES['n_combo'],
            COLUMN_NAMES['physical_inspection_chip_level_miss_prob'],
            COLUMN_NAMES['plv_chip_level_miss_prob'],
            COLUMN_NAMES['total_tests'],
            COLUMN_NAMES['total_component_chips'],
            COLUMN_NAMES['total_bad_records'],
            COLUMN_NAMES['number_of_clusters'],
            COLUMN_NAMES['number_of_clusters_with_smuggling'],
            COLUMN_NAMES['physical_inspection_total_diverted_chips_identified'],
            COLUMN_NAMES['plv_total_diverted_chips_identified'],
        ],
    ].copy()

    grouped = summary_source.groupby(COLUMN_NAMES['scenario_id'], sort=False, observed=True)
    total_scenarios = grouped.ngroups
    print(
        f"_derive_scenario_summary_from_components: deriving {total_scenarios} scenario summaries "
        f"from {len(scenario_component_df)} scenario-component rows"
    )

    # Sum additive quantities while carrying through the identifying mix and parameter-combo fields.
    scenario_summary = grouped.agg(
        Mix_ID=(COLUMN_NAMES['mix_id'], "first"),
        Mix_Description=(COLUMN_NAMES['mix_description'], "first"),
        Total_Clusters_in_Mix=(COLUMN_NAMES['number_of_clusters'], "sum"),
        Scenario_Component_Count=(COLUMN_NAMES['number_of_clusters'], "size"),
        K_Combo=(COLUMN_NAMES['k_combo'], "first"),
        N_Combo=(COLUMN_NAMES['n_combo'], "first"),
        Physical_Inspection_Chip_Level_Miss_Prob=(
            COLUMN_NAMES['physical_inspection_chip_level_miss_prob'],
            "first",
        ),
        PLV_Chip_Level_Miss_Prob=(COLUMN_NAMES['plv_chip_level_miss_prob'], "first"),
        Total_Tests=(COLUMN_NAMES['total_tests'], "sum"),
        Total_Component_Chips=(COLUMN_NAMES['total_component_chips'], "sum"),
        Bad_Records=(COLUMN_NAMES['total_bad_records'], "sum"),
        Number_of_Clusters_with_Smuggling=(
            COLUMN_NAMES['number_of_clusters_with_smuggling'],
            "sum",
        ),
        Physical_Inspection_Total_Diverted_Chips_Identified=(
            COLUMN_NAMES['physical_inspection_total_diverted_chips_identified'],
            "sum",
        ),
        PLV_Total_Diverted_Chips_Identified=(COLUMN_NAMES['plv_total_diverted_chips_identified'], "sum"),
    ).reset_index()
    scenario_summary = scenario_summary.rename(
        columns={
            "Mix_ID": COLUMN_NAMES['mix_id'],
            "Mix_Description": COLUMN_NAMES['mix_description'],
            "Total_Clusters_in_Mix": COLUMN_NAMES['total_clusters_in_mix'],
            "Scenario_Component_Count": COLUMN_NAMES['scenario_component_count'],
            "K_Combo": COLUMN_NAMES['k_combo'],
            "N_Combo": COLUMN_NAMES['n_combo'],
            "Physical_Inspection_Chip_Level_Miss_Prob": COLUMN_NAMES['physical_inspection_chip_level_miss_prob'],
            "PLV_Chip_Level_Miss_Prob": COLUMN_NAMES['plv_chip_level_miss_prob'],
            "Total_Tests": COLUMN_NAMES['total_tests'],
            "Total_Component_Chips": COLUMN_NAMES['total_component_chips'],
            "Bad_Records": COLUMN_NAMES['bad_records'],
            "Number_of_Clusters_with_Smuggling": COLUMN_NAMES['number_of_clusters_with_smuggling'],
            "Physical_Inspection_Total_Diverted_Chips_Identified": COLUMN_NAMES['physical_inspection_total_diverted_chips_identified'],
            "PLV_Total_Diverted_Chips_Identified": COLUMN_NAMES['plv_total_diverted_chips_identified'],
        }
    )

    # Reorder columns to the stable scenario-level schema expected by downstream cost calculations.
    scenario_summary = scenario_summary[
        [
            COLUMN_NAMES['mix_id'],
            COLUMN_NAMES['scenario_id'],
            COLUMN_NAMES['mix_description'],
            COLUMN_NAMES['total_clusters_in_mix'],
            COLUMN_NAMES['scenario_component_count'],
            COLUMN_NAMES['k_combo'],
            COLUMN_NAMES['n_combo'],
            COLUMN_NAMES['physical_inspection_chip_level_miss_prob'],
            COLUMN_NAMES['plv_chip_level_miss_prob'],
            COLUMN_NAMES['total_tests'],
            COLUMN_NAMES['total_component_chips'],
            COLUMN_NAMES['bad_records'],
            COLUMN_NAMES['number_of_clusters_with_smuggling'],
            COLUMN_NAMES['physical_inspection_total_diverted_chips_identified'],
            COLUMN_NAMES['plv_total_diverted_chips_identified'],
        ]
    ]

    print(
        f"_derive_scenario_summary_from_components: completed {len(scenario_summary)} scenario summaries"
    )
    return scenario_summary


# Select one best physical-inspection scenario per mix and smuggling configuration.
def filter_perfect_information_scenarios(final_df: pd.DataFrame) -> pd.DataFrame:
    if final_df.empty:
        return final_df.copy()

    group_columns = [
        COLUMN_NAMES['mix_id'],
        COLUMN_NAMES['k_combo'],
        COLUMN_NAMES['physical_inspection_chip_level_miss_prob'],
        COLUMN_NAMES['plv_chip_level_miss_prob'],
    ]
    score_column = "Physical - Max Value Per Cost"

    # Stable sorting makes ties deterministic before choosing the best physical-inspection outcome.
    ordered_df = final_df.sort_values(
        group_columns + [COLUMN_NAMES['scenario_id']],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    )
    selected_indices = ordered_df.groupby(group_columns, sort=False, observed=True)[score_column].idxmax()
    filtered_df = ordered_df.loc[selected_indices].copy()
    filtered_df = filtered_df.sort_values(
        group_columns + [COLUMN_NAMES['scenario_id']],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    print(
        "filter_perfect_information_scenarios: kept "
        f"{len(filtered_df):,} of {len(final_df):,} scenarios across "
        f"{len(selected_indices):,} grouped combinations"
    )
    return filtered_df


# Classify how one numeric interval sits relative to another interval using stable relationship codes.
def _classify_interval_relationship(
    summary_df: pd.DataFrame,
    *,
    physical_min_column: str,
    physical_max_column: str,
    compare_min_column: str,
    compare_max_column: str,
    output_column: str,
    error_message: str,
) -> pd.Series:
    physical_min = summary_df[physical_min_column]
    physical_max = summary_df[physical_max_column]
    compare_min = summary_df[compare_min_column]
    compare_max = summary_df[compare_max_column]

    # Codes describe the full interval relationship between physical and comparison value-per-cost ranges.
    relationship = np.select(
        [
            physical_max < compare_min,
            physical_min > compare_max,
            (physical_min < compare_min) & (physical_max > compare_max),
            (compare_min <= physical_min) & (physical_max <= compare_max),
            (physical_min < compare_min) & (compare_min <= physical_max) & (physical_max <= compare_max),
            (compare_min <= physical_min) & (physical_min <= compare_max) & (compare_max < physical_max),
        ],
        ["a", "e", "f", "c", "b", "d"],
        default="unknown",
    )
    result = pd.Series(relationship, index=summary_df.index, name=output_column)
    if (result == "unknown").any():
        raise ValueError(error_message)
    return result


# Add cost, value-per-cost, and relationship-code columns for a single PLV variant.
def _add_plv_variant_cost_value_columns(
    result: pd.DataFrame,
    *,
    plv_type: str,
    plv_cost_per_total_chip: tuple[float, float],
) -> None:
    min_cost_column = f"{plv_type} - Min Total Cost"
    max_cost_column = f"{plv_type} - Max Total Cost"
    min_value_per_cost_column = f"{plv_type} - Min Value Per Cost"
    max_value_per_cost_column = f"{plv_type} - Max Value Per Cost"
    detected_column = f"{plv_type} - Total Diverted Chips Identified"

    # PLV cost is modeled against the full target chip population, independent of test count.
    result[detected_column] = result[COLUMN_NAMES['plv_total_diverted_chips_identified']]
    result[min_cost_column] = TARGET_CHIPS * plv_cost_per_total_chip[0]
    result[max_cost_column] = TARGET_CHIPS * plv_cost_per_total_chip[1]
    result[min_value_per_cost_column] = (
        result[detected_column] * PLV_DISCOUNT_RATE / result[max_cost_column]
    )
    result[max_value_per_cost_column] = (
        result[detected_column] * PLV_DISCOUNT_RATE / result[min_cost_column]
    )

    vpc_relationship_column = f"Physical vs {plv_type} Value Per Cost Relationship"

    # Compare physical and PLV value ranges after both have min/max value-per-cost columns.
    result[vpc_relationship_column] = _classify_interval_relationship(
        result,
        physical_min_column="Physical - Min Value Per Cost",
        physical_max_column="Physical - Max Value Per Cost",
        compare_min_column=min_value_per_cost_column,
        compare_max_column=max_value_per_cost_column,
        output_column=vpc_relationship_column,
        error_message=(
            f"Encountered an unexpected physical-vs-{plv_type} value-per-cost relationship; "
            "check the interval classification logic."
        ),
    )


# Add physical-inspection and PLV cost-value columns to a scenario summary dataframe.
def add_cost_value_columns(
    summary_df: pd.DataFrame,
    phys_inspection_salary_cost_per_tested_chip: tuple[float, float] = PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP,
    phys_inspection_fixed_cost_per_inspection: tuple[float, float] = PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION,
    plv_renting_cost_per_total_chip: tuple[float, float] = PLV_RENTING_COST_PER_TOTAL_CHIP,
    plv_owning_cost_per_total_chip: tuple[float, float] = PLV_OWNING_COST_PER_TOTAL_CHIP,
) -> pd.DataFrame:
    result = summary_df.copy()

    # Scenario summaries use total component chips; smaller component-only callers can fall back to cluster size.
    if COLUMN_NAMES['total_component_chips'] in result.columns:
        result[COLUMN_NAMES['share_diverted']] = result[COLUMN_NAMES['bad_records']] / result[COLUMN_NAMES['total_component_chips']]
    else:
        result[COLUMN_NAMES['share_diverted']] = result[COLUMN_NAMES['bad_records']] / result[COLUMN_NAMES['cluster_size']]

    # Physical-inspection costs include per-cluster fixed cost and per-tested-chip labor cost, repeated annually.
    result["Physical Inspection - Min Total Cost"] = (
        (result[COLUMN_NAMES['total_clusters_in_mix']] * phys_inspection_fixed_cost_per_inspection[0]
        + result[COLUMN_NAMES['total_tests']] * phys_inspection_salary_cost_per_tested_chip[0])*NUMBER_OF_PHYSICAL_INSPECTIONS_PER_CLUSTER_PER_YEAR
    )
    result["Physical Inspection - Max Total Cost"] = (
        (result[COLUMN_NAMES['total_clusters_in_mix']] * phys_inspection_fixed_cost_per_inspection[1]
        + result[COLUMN_NAMES['total_tests']] * phys_inspection_salary_cost_per_tested_chip[1])*NUMBER_OF_PHYSICAL_INSPECTIONS_PER_CLUSTER_PER_YEAR
    )
    result["Physical - Min Value Per Cost"] = (
        result[COLUMN_NAMES['physical_inspection_total_diverted_chips_identified']] / result["Physical Inspection - Max Total Cost"]
    )
    result["Physical - Max Value Per Cost"] = (
        result[COLUMN_NAMES['physical_inspection_total_diverted_chips_identified']] / result["Physical Inspection - Min Total Cost"]
    )

    # Add the same PLV-derived columns for each configured PLV cost model.
    for variant in _plv_variant_specs():
        plv_type = str(variant["plv_type"])
        plv_cost_per_total_chip = (
            plv_renting_cost_per_total_chip if plv_type == "PLV Renting" else plv_owning_cost_per_total_chip
        )
        _add_plv_variant_cost_value_columns(
            result,
            plv_type=plv_type,
            plv_cost_per_total_chip=plv_cost_per_total_chip,
        )

    return result
