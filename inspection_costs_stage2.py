from __future__ import annotations

from inspection_costs_stage1 import *
from inspection_costs_stage1 import _plv_variant_specs
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
                COLUMN_NAMES['total_tests'],
                COLUMN_NAMES['total_component_chips'],
                COLUMN_NAMES['bad_records'],
                COLUMN_NAMES['number_of_clusters_with_smuggling'],
            ]
        )

    summary_source = scenario_component_df.loc[
        :,
        [
            COLUMN_NAMES['scenario_id'],
            COLUMN_NAMES['mix_id'],
            COLUMN_NAMES['mix_description'],
            COLUMN_NAMES['k_combo'],
            COLUMN_NAMES['n_combo'],
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

    scenario_summary = grouped.agg(
        Mix_ID=(COLUMN_NAMES['mix_id'], "first"),
        Mix_Description=(COLUMN_NAMES['mix_description'], "first"),
        Total_Clusters_in_Mix=(COLUMN_NAMES['number_of_clusters'], "sum"),
        Scenario_Component_Count=(COLUMN_NAMES['number_of_clusters'], "size"),
        K_Combo=(COLUMN_NAMES['k_combo'], "first"),
        N_Combo=(COLUMN_NAMES['n_combo'], "first"),
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
            "Total_Tests": COLUMN_NAMES['total_tests'],
            "Total_Component_Chips": COLUMN_NAMES['total_component_chips'],
            "Bad_Records": COLUMN_NAMES['bad_records'],
            "Number_of_Clusters_with_Smuggling": COLUMN_NAMES['number_of_clusters_with_smuggling'],
            "Physical_Inspection_Total_Diverted_Chips_Identified": COLUMN_NAMES['physical_inspection_total_diverted_chips_identified'],
            "PLV_Total_Diverted_Chips_Identified": COLUMN_NAMES['plv_total_diverted_chips_identified'],
        }
    )
    scenario_summary = scenario_summary[
        [
            COLUMN_NAMES['mix_id'],
            COLUMN_NAMES['scenario_id'],
            COLUMN_NAMES['mix_description'],
            COLUMN_NAMES['total_clusters_in_mix'],
            COLUMN_NAMES['scenario_component_count'],
            COLUMN_NAMES['k_combo'],
            COLUMN_NAMES['n_combo'],
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


def _add_plv_variant_cost_benefit_columns(
    result: pd.DataFrame,
    *,
    plv_type: str,
    plv_cost_per_total_chip: tuple[float, float],
) -> None:
    min_cost_column = f"{plv_type} - Min Total Cost"
    max_cost_column = f"{plv_type} - Max Total Cost"
    min_benefit_per_dollar_column = f"{plv_type} - Min Benefit Per Dollar"
    max_benefit_per_dollar_column = f"{plv_type} - Max Benefit Per Dollar"
    detected_column = f"{plv_type} - Total Diverted Chips Identified"

    result[detected_column] = result[COLUMN_NAMES['plv_total_diverted_chips_identified']]
    result[min_cost_column] = TARGET_CHIPS * plv_cost_per_total_chip[0]
    result[max_cost_column] = TARGET_CHIPS * plv_cost_per_total_chip[1]
    result[min_benefit_per_dollar_column] = (
        result[detected_column] * PLV_DISCOUNT_RATE / result[max_cost_column]
    )
    result[max_benefit_per_dollar_column] = (
        result[detected_column] * PLV_DISCOUNT_RATE / result[min_cost_column]
    )

    bpd_relationship_column = f"Physical vs {plv_type} Benefit Per Dollar Relationship"
    result[bpd_relationship_column] = _classify_interval_relationship(
        result,
        physical_min_column="Physical - Min Benefit Per Dollar",
        physical_max_column="Physical - Max Benefit Per Dollar",
        compare_min_column=min_benefit_per_dollar_column,
        compare_max_column=max_benefit_per_dollar_column,
        output_column=bpd_relationship_column,
        error_message=(
            f"Encountered an unexpected physical-vs-{plv_type} benefit-per-dollar relationship; "
            "check the interval classification logic."
        ),
    )


def add_cost_benefit_columns(
    summary_df: pd.DataFrame,
    phys_inspection_salary_cost_per_tested_chip: tuple[float, float] = PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP,
    phys_inspection_fixed_cost_per_inspection: tuple[float, float] = PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION,
    plv_renting_cost_per_total_chip: tuple[float, float] = PLV_RENTING_COST_PER_TOTAL_CHIP,
    plv_owning_cost_per_total_chip: tuple[float, float] = PLV_OWNING_COST_PER_TOTAL_CHIP,
) -> pd.DataFrame:
    result = summary_df.copy()
    if COLUMN_NAMES['total_component_chips'] in result.columns:
        result[COLUMN_NAMES['share_diverted']] = result[COLUMN_NAMES['bad_records']] / result[COLUMN_NAMES['total_component_chips']]
    else:
        result[COLUMN_NAMES['share_diverted']] = result[COLUMN_NAMES['bad_records']] / result[COLUMN_NAMES['cluster_size']]

    result["Physical Inspection - Min Total Cost"] = (
        (result[COLUMN_NAMES['total_clusters_in_mix']] * phys_inspection_fixed_cost_per_inspection[0]
        + result[COLUMN_NAMES['total_tests']] * phys_inspection_salary_cost_per_tested_chip[0])*NUMBER_OF_PHYSICAL_INSPECTIONS_PER_CLUSTER_PER_YEAR
    )
    result["Physical Inspection - Max Total Cost"] = (
        (result[COLUMN_NAMES['total_clusters_in_mix']] * phys_inspection_fixed_cost_per_inspection[1]
        + result[COLUMN_NAMES['total_tests']] * phys_inspection_salary_cost_per_tested_chip[1])*NUMBER_OF_PHYSICAL_INSPECTIONS_PER_CLUSTER_PER_YEAR
    )
    result["Physical - Min Benefit Per Dollar"] = (
        result[COLUMN_NAMES['physical_inspection_total_diverted_chips_identified']] / result["Physical Inspection - Max Total Cost"]
    )
    result["Physical - Max Benefit Per Dollar"] = (
        result[COLUMN_NAMES['physical_inspection_total_diverted_chips_identified']] / result["Physical Inspection - Min Total Cost"]
    )

    for variant in _plv_variant_specs():
        plv_type = str(variant["plv_type"])
        plv_cost_per_total_chip = (
            plv_renting_cost_per_total_chip if plv_type == "PLV Renting" else plv_owning_cost_per_total_chip
        )
        _add_plv_variant_cost_benefit_columns(
            result,
            plv_type=plv_type,
            plv_cost_per_total_chip=plv_cost_per_total_chip,
        )

    return result
