from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from inspection_costs import (
    COLUMN_NAMES,
    LONG_SCENARIO_VALUE_VARS,
    PLV_VARIANT_ORDER,
    RELATIONSHIP_CODE_ORDER,
    SMUGGLED_CHIPS_METRIC_FAMILY,
    SMUGGLED_CHIPS_SCENARIO_GROUP,
    SMUGGLED_CHIPS_SCENARIO_TYPE,
    SMUGGLED_CHIPS_SCENARIO_VARIANT,
    _plv_relationship_labels,
    _plv_variant_specs,
    _scenario_variant_label,
)
from inspection_costs_stage1 import (
    _iter_parquet_batches,
)

# Format a numeric or missing value for human-readable rule clauses.
def _format_rule_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        if np.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(round(numeric_value)))
        return f"{numeric_value:.6g}"
    return str(value)


# Join scenario-level rows with component-level min/max fields used to describe relationship rules.
def _build_relationship_rule_frame(
    scenario_df: pd.DataFrame,
    scenario_component_df: pd.DataFrame,
) -> pd.DataFrame:
    if scenario_df.empty:
        return scenario_df.copy()

    source_df = scenario_df.copy()
    component_summary = scenario_component_df.loc[
        :,
        [
            COLUMN_NAMES['scenario_id'],
            COLUMN_NAMES['cluster_size'],
            COLUMN_NAMES['number_of_clusters'],
            COLUMN_NAMES['bad_records'],
            COLUMN_NAMES['chips_inspected_per_cluster'],
        ],
    ].groupby(COLUMN_NAMES['scenario_id'], sort=False, observed=True).agg(
        component_cluster_size_min=(COLUMN_NAMES['cluster_size'], "min"),
        component_cluster_size_max=(COLUMN_NAMES['cluster_size'], "max"),
        component_cluster_count_min=(COLUMN_NAMES['number_of_clusters'], "min"),
        component_cluster_count_max=(COLUMN_NAMES['number_of_clusters'], "max"),
        component_k_min=(COLUMN_NAMES['bad_records'], "min"),
        component_k_max=(COLUMN_NAMES['bad_records'], "max"),
        component_chips_inspected_per_cluster_min=(COLUMN_NAMES['chips_inspected_per_cluster'], "min"),
        component_chips_inspected_per_cluster_max=(COLUMN_NAMES['chips_inspected_per_cluster'], "max"),
    ).reset_index()
    source_df = source_df.merge(component_summary, on=COLUMN_NAMES['scenario_id'], how='left')

    source_df["K Combo Min"] = source_df["component_k_min"]
    source_df["K Combo Max"] = source_df["component_k_max"]
    source_df["Chips Inspected Per Cluster Combo Min"] = source_df["component_chips_inspected_per_cluster_min"]
    source_df["Chips Inspected Per Cluster Combo Max"] = source_df["component_chips_inspected_per_cluster_max"]
    source_df = source_df.drop(
        columns=[
            "component_k_min",
            "component_k_max",
            "component_chips_inspected_per_cluster_min",
            "component_chips_inspected_per_cluster_max",
        ]
    )

    return source_df


# Summarize observed feature bounds for each relationship code and PLV variant.
def _summarize_relationship_rule_bounds(
    source_df: pd.DataFrame,
    *,
    relationship_code_column: str,
    plv_type: str,
) -> pd.DataFrame:
    relationship_labels = _plv_relationship_labels(plv_type)

    # Feature specs define which scenario fields are summarized into human-readable rule clauses.
    feature_specs = [
        {
            "feature": COLUMN_NAMES['total_clusters_in_mix'],
            "group": "Scenario Summary",
            "source_columns": [COLUMN_NAMES['total_clusters_in_mix']],
        },
        {
            "feature": COLUMN_NAMES['total_tests'],
            "group": "Scenario Summary",
            "source_columns": [COLUMN_NAMES['total_tests']],
        },
        {
            "feature": COLUMN_NAMES['bad_records'],
            "group": "Scenario Summary",
            "source_columns": [COLUMN_NAMES['bad_records']],
        },
        {
            "feature": COLUMN_NAMES['number_of_clusters_with_smuggling'],
            "group": "Scenario Summary",
            "source_columns": [COLUMN_NAMES['number_of_clusters_with_smuggling']],
        },
        {
            "feature": COLUMN_NAMES['share_diverted'],
            "group": "Scenario Summary",
            "source_columns": [COLUMN_NAMES['share_diverted']],
        },
        {
            "feature": COLUMN_NAMES['physical_inspection_chip_level_miss_prob'],
            "group": "Scenario Summary",
            "source_columns": [COLUMN_NAMES['physical_inspection_chip_level_miss_prob']],
        },
        {
            "feature": COLUMN_NAMES['plv_chip_level_miss_prob'],
            "group": "Scenario Summary",
            "source_columns": [COLUMN_NAMES['plv_chip_level_miss_prob']],
        },
        {
            "feature": "K Combo",
            "group": "Combo Derived",
            "source_columns": ["K Combo Min", "K Combo Max"],
        },
        {
            "feature": "Chips Inspected Per Cluster Combo",
            "group": "Combo Derived",
            "source_columns": ["Chips Inspected Per Cluster Combo Min", "Chips Inspected Per Cluster Combo Max"],
        },
        {
            "feature": "component_cluster_size",
            "group": "Component Summary",
            "source_columns": ["component_cluster_size_min", "component_cluster_size_max"],
        },
        {
            "feature": "component_cluster_count",
            "group": "Component Summary",
            "source_columns": ["component_cluster_count_min", "component_cluster_count_max"],
        },
    ]

    # Drop invariant features so the rules focus on values that actually separate scenarios.
    feature_specs = [
        feature_spec
        for feature_spec in feature_specs
        if pd.concat(
            [pd.to_numeric(source_df[column_name], errors="coerce") for column_name in feature_spec["source_columns"] if column_name in source_df.columns],
            ignore_index=True,
        ).dropna().nunique() > 1
    ]
    rows: list[dict[str, object]] = []

    # Each relationship code gets a list of simple observed-range clauses for the varying features.
    relationship_values = source_df[relationship_code_column].astype(str)
    for relationship_code in RELATIONSHIP_CODE_ORDER:
        code_mask = relationship_values == relationship_code
        code_df = source_df.loc[code_mask]
        if code_df.empty:
            continue

        scenario_count = int(len(code_df))
        feature_clauses: list[str] = []
        for feature_spec in feature_specs:
            feature_name = feature_spec["feature"]
            present_columns = [
                column_name
                for column_name in feature_spec["source_columns"]
                if column_name in code_df.columns
            ]
            if not present_columns:
                continue
            values = pd.concat(
                [pd.to_numeric(code_df[column_name], errors="coerce") for column_name in present_columns],
                ignore_index=True,
            ).dropna()
            if values.empty:
                continue

            observed_min = values.min()
            observed_max = values.max()
            lower_text = _format_rule_value(observed_min)
            upper_text = _format_rule_value(observed_max)
            if observed_min == observed_max:
                clause_text = f"{feature_name} = {lower_text}"
            else:
                clause_text = f"{lower_text} <= {feature_name} <= {upper_text}"
            feature_clauses.append(clause_text)
            rows.append(
                {
                    COLUMN_NAMES['plv_type']: plv_type,
                    COLUMN_NAMES['relationship_code']: relationship_code,
                    COLUMN_NAMES['relationship_description']: relationship_labels[relationship_code],
                    COLUMN_NAMES['scenario_count']: scenario_count,
                    "Rule Component Index": len(feature_clauses),
                    "Feature Group": feature_spec["group"],
                    "Feature": feature_name,
                    "Distinct Values": int(values.nunique(dropna=True)),
                    "Observed Min": float(observed_min),
                    "Observed Max": float(observed_max),
                    "Rule Component": clause_text,
                }
            )

        if not feature_clauses:
            rows.append(
                {
                    COLUMN_NAMES['plv_type']: plv_type,
                    COLUMN_NAMES['relationship_code']: relationship_code,
                    COLUMN_NAMES['relationship_description']: relationship_labels[relationship_code],
                    COLUMN_NAMES['scenario_count']: scenario_count,
                    "Rule Component Index": 1,
                    "Feature Group": "N/A",
                    "Feature": "N/A",
                    "Distinct Values": 0,
                    "Observed Min": np.nan,
                    "Observed Max": np.nan,
                    "Rule Component": "TRUE",
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                COLUMN_NAMES['plv_type'],
                COLUMN_NAMES['relationship_code'],
                COLUMN_NAMES['relationship_description'],
                COLUMN_NAMES['scenario_count'],
                "Rule Component Index",
                "Feature Group",
                "Feature",
                "Distinct Values",
                "Observed Min",
                "Observed Max",
                "Rule Component",
            ]
        )

    result[COLUMN_NAMES['plv_type']] = pd.Categorical(result[COLUMN_NAMES['plv_type']], categories=PLV_VARIANT_ORDER, ordered=True)
    result[COLUMN_NAMES['relationship_code']] = pd.Categorical(
        result[COLUMN_NAMES['relationship_code']],
        categories=RELATIONSHIP_CODE_ORDER,
        ordered=True,
    )
    result = result.sort_values(
        [COLUMN_NAMES['plv_type'], COLUMN_NAMES['relationship_code'], "Rule Component Index"],
        kind="mergesort",
    ).reset_index(drop=True)
    return result


# Write the relationship-code rule summary CSV for all PLV variants present in the scenario data.
def _write_relationship_code_rules_from_dataframes(
    *,
    scenario_df: pd.DataFrame,
    scenario_component_df: pd.DataFrame,
    output_csv_path: Path,
    label: str,
) -> pd.DataFrame:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if scenario_df.empty:
        empty_df = pd.DataFrame(
            columns=[
                COLUMN_NAMES['plv_type'],
                COLUMN_NAMES['relationship_code'],
                COLUMN_NAMES['relationship_description'],
                COLUMN_NAMES['scenario_count'],
                "Rule Component Index",
                "Feature Group",
                "Feature",
                "Distinct Values",
                "Observed Min",
                "Observed Max",
                "Rule Component",
            ]
        )
        empty_df.to_csv(output_csv_path, index=False)
        print(f"Wrote empty relationship rule summary for {label} to {output_csv_path}")
        return empty_df

    rule_source_df = _build_relationship_rule_frame(scenario_df, scenario_component_df)
    summary_frames: list[pd.DataFrame] = []
    for variant in _plv_variant_specs():
        plv_type = str(variant["plv_type"])
        relationship_code_column = f"Physical vs {plv_type} Value Per Cost Relationship"
        if relationship_code_column not in rule_source_df.columns:
            continue
        summary_frames.append(
            _summarize_relationship_rule_bounds(
                rule_source_df,
                relationship_code_column=relationship_code_column,
                plv_type=plv_type,
            )
        )

    result = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    print(f"Writing relationship code rules CSV to {output_csv_path}")
    result.to_csv(output_csv_path, index=False)
    return result


# Count relationship-code frequencies from a cached costed scenario parquet file.
def _summarize_relationships_from_parquet(
    parquet_path: Path,
    batch_size: int = 65_536,
) -> pd.DataFrame:
    print(f"Summarizing relationship counts from cached costed parquet: {parquet_path}")
    summaries = []
    for variant in _plv_variant_specs():
        plv_type = str(variant["plv_type"])
        vpc_relationship_column = f"Physical vs {plv_type} Value Per Cost Relationship"
        vpc_relationship_labels = _plv_relationship_labels(plv_type)
        value_per_cost_counts: Counter[str] = Counter()

        for batch_df in _iter_parquet_batches(
            parquet_path,
            columns=[vpc_relationship_column],
            batch_size=batch_size,
        ):
            value_per_cost_counts.update(batch_df[vpc_relationship_column].astype(str).tolist())

        value_per_cost_summary = pd.DataFrame(
            {
                COLUMN_NAMES['plv_type']: [plv_type] * len(vpc_relationship_labels),
                COLUMN_NAMES['relationship_code']: list(vpc_relationship_labels.keys()),
                COLUMN_NAMES['value_per_cost_relationship_description']: list(vpc_relationship_labels.values()),
                COLUMN_NAMES['value_per_cost_scenario_count']: [
                    int(value_per_cost_counts.get(code, 0)) for code in vpc_relationship_labels.keys()
                ],
            }
        )
        total_row = pd.DataFrame(
            {
                COLUMN_NAMES['plv_type']: [plv_type],
                COLUMN_NAMES['relationship_code']: ["Total"],
                COLUMN_NAMES['value_per_cost_relationship_description']: ["All scenarios"],
                COLUMN_NAMES['value_per_cost_scenario_count']: [int(sum(value_per_cost_counts.values()))],
            }
        )
        summaries.append(pd.concat([value_per_cost_summary, total_row], ignore_index=True))

    result = pd.concat(summaries, ignore_index=True)
    result[COLUMN_NAMES['plv_type']] = pd.Categorical(result[COLUMN_NAMES['plv_type']], categories=PLV_VARIANT_ORDER, ordered=True)
    relationship_sort_order = RELATIONSHIP_CODE_ORDER + ["Total"]
    result[COLUMN_NAMES['relationship_code']] = pd.Categorical(
        result[COLUMN_NAMES['relationship_code']],
        categories=relationship_sort_order,
        ordered=True,
    )
    return result.sort_values([COLUMN_NAMES['plv_type'], COLUMN_NAMES['relationship_code']]).reset_index(drop=True)


# Return tidy raw rows for box plots split by relationship category, PLV variant, and metric family.
def _build_relationship_boxplot_dataframe(final_df: pd.DataFrame) -> pd.DataFrame:
    """Return tidy raw rows for box plots split by relationship category, PLV variant, and metric family."""

    boxplot_frames: list[pd.DataFrame] = []
    for variant in _plv_variant_specs():
        plv_type = str(variant["plv_type"])
        vpc_relationship_column = f"Physical vs {plv_type} Value Per Cost Relationship"
        vpc_relationship_labels = _plv_relationship_labels(plv_type)

        # Melt min/max physical and PLV value-per-cost columns into one plotting value column.
        value_per_cost_source_columns = [
            column
            for column in [
                COLUMN_NAMES['mix_id'],
                COLUMN_NAMES['scenario_id'],
                vpc_relationship_column,
                "Physical - Min Value Per Cost",
                "Physical - Max Value Per Cost",
                f"{plv_type} - Min Value Per Cost",
                f"{plv_type} - Max Value Per Cost",
            ]
            if column in final_df.columns
        ]
        value_per_cost_id_vars = [
            column
            for column in [COLUMN_NAMES['mix_id'], COLUMN_NAMES['scenario_id'], vpc_relationship_column]
            if column in value_per_cost_source_columns
        ]
        value_per_cost_df = final_df[value_per_cost_source_columns].melt(
            id_vars=value_per_cost_id_vars,
            value_vars=[
                "Physical - Min Value Per Cost",
                "Physical - Max Value Per Cost",
                f"{plv_type} - Min Value Per Cost",
                f"{plv_type} - Max Value Per Cost",
            ],
            var_name=COLUMN_NAMES['scenario_type'],
            value_name=COLUMN_NAMES['value'],
        )
        value_per_cost_df[COLUMN_NAMES['metric_family']] = "Value Per Cost"
        value_per_cost_df[COLUMN_NAMES['plv_type']] = plv_type
        value_per_cost_df[COLUMN_NAMES['relationship_code']] = value_per_cost_df[vpc_relationship_column]
        value_per_cost_df[COLUMN_NAMES['relationship_description']] = value_per_cost_df[COLUMN_NAMES['relationship_code']].map(
            vpc_relationship_labels
        )
        value_per_cost_df[COLUMN_NAMES['scenario_group']] = np.where(
            value_per_cost_df[COLUMN_NAMES['scenario_type']].str.startswith("Physical"),
            "Physical Inspection",
            plv_type,
        )
        value_per_cost_df[COLUMN_NAMES['scenario_variant']] = value_per_cost_df[COLUMN_NAMES['scenario_type']].map(
            lambda scenario_type: _scenario_variant_label(scenario_type, plv_type)
        )

        # Add a second metric family that totals smuggled chips by relationship code.
        smuggled_chips_df = (
            final_df.groupby(vpc_relationship_column, sort=False, observed=True)[COLUMN_NAMES['bad_records']]
            .sum()
            .reindex(RELATIONSHIP_CODE_ORDER)
            .dropna()
            .reset_index()
            .rename(
                columns={
                    vpc_relationship_column: COLUMN_NAMES['relationship_code'],
                    COLUMN_NAMES['bad_records']: COLUMN_NAMES['value'],
                }
            )
        )
        smuggled_chips_df[COLUMN_NAMES['plv_type']] = plv_type
        smuggled_chips_df[COLUMN_NAMES['metric_family']] = SMUGGLED_CHIPS_METRIC_FAMILY
        smuggled_chips_df[COLUMN_NAMES['relationship_description']] = smuggled_chips_df[COLUMN_NAMES['relationship_code']].map(
            vpc_relationship_labels
        )
        smuggled_chips_df[COLUMN_NAMES['scenario_group']] = SMUGGLED_CHIPS_SCENARIO_GROUP
        smuggled_chips_df[COLUMN_NAMES['scenario_variant']] = SMUGGLED_CHIPS_SCENARIO_VARIANT
        smuggled_chips_df[COLUMN_NAMES['scenario_type']] = SMUGGLED_CHIPS_SCENARIO_TYPE
        smuggled_chips_df[COLUMN_NAMES['mix_id']] = "All Scenarios"
        smuggled_chips_df[COLUMN_NAMES['scenario_id']] = "All Scenarios"
        smuggled_chips_df[COLUMN_NAMES['value']] = smuggled_chips_df[COLUMN_NAMES['value']].astype("float64")

        boxplot_frames.extend([value_per_cost_df, smuggled_chips_df])

    boxplot_df = pd.concat(boxplot_frames, ignore_index=True)
    output_columns = [
        COLUMN_NAMES['metric_family'],
        COLUMN_NAMES['plv_type'],
        COLUMN_NAMES['relationship_code'],
        COLUMN_NAMES['relationship_description'],
        COLUMN_NAMES['scenario_group'],
        COLUMN_NAMES['scenario_variant'],
    ]
    if COLUMN_NAMES['mix_id'] in boxplot_df.columns:
        output_columns.append(COLUMN_NAMES['mix_id'])
    output_columns.extend([COLUMN_NAMES['scenario_id'], COLUMN_NAMES['scenario_type'], COLUMN_NAMES['value']])
    boxplot_df = boxplot_df[output_columns]
    return boxplot_df


# Return box-plot summary statistics for each metric family and relationship group.
def _summarize_relationship_boxplot_dataframe(boxplot_df: pd.DataFrame) -> pd.DataFrame:
    """Return box-plot summary statistics for each metric family and relationship group."""

    group_columns = [
        COLUMN_NAMES['metric_family'],
        COLUMN_NAMES['plv_type'],
        COLUMN_NAMES['relationship_code'],
        COLUMN_NAMES['relationship_description'],
        COLUMN_NAMES['scenario_group'],
        COLUMN_NAMES['scenario_variant'],
        COLUMN_NAMES['scenario_type'],
    ]
    if boxplot_df.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                COLUMN_NAMES['scenario_count'],
                "Min",
                "Q1",
                "Median",
                "Q3",
                "Max",
                "Mean",
            ]
        )

    grouped = boxplot_df.groupby(group_columns, dropna=False)[COLUMN_NAMES['value']]
    summary = grouped.agg(
        Scenario_Count="count",
        Min="min",
        Median="median",
        Max="max",
        Mean="mean",
    )
    summary["Q1"] = grouped.quantile(0.25)
    summary["Q3"] = grouped.quantile(0.75)
    summary = summary.reset_index()
    summary = summary.rename(columns={"Scenario_Count": COLUMN_NAMES['scenario_count']})
    summary[COLUMN_NAMES['plv_type']] = pd.Categorical(summary[COLUMN_NAMES['plv_type']], categories=PLV_VARIANT_ORDER, ordered=True)
    return summary[
        [
            *group_columns,
            COLUMN_NAMES['scenario_count'],
                "Min",
                "Q1",
            "Median",
            "Q3",
            "Max",
            "Mean",
        ]
    ].sort_values(group_columns).reset_index(drop=True)


# Convert a label into a conservative filename stem for generated image outputs.
def _slugify_filename(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "_" for character in value)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


# Render relationship boxplots from accumulated grouped values and return the image paths.
def _render_relationship_boxplot_images(
    grouped_values: dict[tuple[object, ...], list[float]],
    output_images_dir: Path,
) -> list[Path]:
    output_images_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    if not grouped_values:
        return saved_paths

    # Render one image per metric-family and PLV-type pair so legends and axes stay readable.
    family_names = sorted(
        {(group_key[0], group_key[1]) for group_key in grouped_values},
        key=lambda item: (item[0], PLV_VARIANT_ORDER.index(item[1]) if item[1] in PLV_VARIANT_ORDER else len(PLV_VARIANT_ORDER)),
    )
    for metric_family, plv_type in family_names:
        family_groups = {
            group_key: values
            for group_key, values in grouped_values.items()
            if group_key[0] == metric_family and group_key[1] == plv_type
        }
        if not family_groups:
            continue

        relationship_codes = [code for code in RELATIONSHIP_CODE_ORDER if any(group_key[2] == code for group_key in family_groups)]
        scenario_types = list(dict.fromkeys(group_key[6] for group_key in family_groups))
        if not relationship_codes or not scenario_types:
            continue

        family_label = f"{metric_family} - {plv_type}"

        # Width and offsets scale with the number of relationship codes and scenario series plotted together.
        fig_width = max(10.0, len(relationship_codes) * max(1.6, 0.55 * len(scenario_types)))
        fig, ax = plt.subplots(figsize=(fig_width, 6.5))
        color_map = plt.get_cmap("tab10")
        colors = [color_map(index % 10) for index in range(len(scenario_types))]
        group_width = 0.78
        box_width = min(0.22, group_width / max(1, len(scenario_types) + 1))
        offsets = np.linspace(
            -group_width / 2 + box_width,
            group_width / 2 - box_width,
            num=len(scenario_types),
        )

        legend_handles = []
        for type_index, scenario_type in enumerate(scenario_types):
            type_values = []
            type_positions = []
            for relationship_index, relationship_code in enumerate(relationship_codes, start=1):
                group_key = next(
                    (
                        key
                        for key in family_groups
                        if key[2] == relationship_code and key[6] == scenario_type
                    ),
                    None,
                )
                if group_key is None:
                    continue
                values = family_groups[group_key]
                if not values:
                    continue
                type_values.append(values)
                type_positions.append(relationship_index + offsets[type_index])

            if not type_values:
                continue

            ax.boxplot(
                type_values,
                positions=type_positions,
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                boxprops={"facecolor": colors[type_index], "edgecolor": "black"},
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"color": "black", "linewidth": 1.0},
                capprops={"color": "black", "linewidth": 1.0},
                flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": colors[type_index], "markeredgecolor": colors[type_index], "alpha": 0.35},
            )
            legend_handles.append(Patch(facecolor=colors[type_index], edgecolor="black", label=scenario_type))

        ax.set_title(f"{family_label} by Relationship Code")
        ax.set_xlabel("Relationship Code")
        ax.set_ylabel(family_label)
        ax.set_xticks(range(1, len(relationship_codes) + 1))
        ax.set_xticklabels(relationship_codes)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if legend_handles:
            ax.legend(handles=legend_handles, title="Scenario Type", loc="best")
        fig.tight_layout()

        output_path = output_images_dir / f"relationship_boxplot_{_slugify_filename(family_label)}.jpeg"
        fig.savefig(output_path, format="jpeg", dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)
        print(f"Wrote relationship boxplot image to {output_path}")

    return saved_paths


# Stream costed scenarios into boxplot summaries and image files without loading all rows at once.
def _write_relationship_boxplot_values_from_parquet(
    parquet_path: Path,
    output_csv_path: Path,
    output_images_dir: Path,
    batch_size: int = 65_536,
) -> pd.DataFrame:
    print(f"Building relationship boxplot summary from cached costed parquet: {parquet_path}")
    columns = [
        COLUMN_NAMES['mix_id'],
        COLUMN_NAMES['scenario_id'],
        "Physical vs PLV Renting Value Per Cost Relationship",
        "Physical vs PLV Owning Value Per Cost Relationship",
        COLUMN_NAMES['bad_records'],
        *LONG_SCENARIO_VALUE_VARS,
    ]
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_values: dict[tuple[object, ...], list[float]] = {}
    smuggled_chip_totals: dict[tuple[object, ...], float] = {}

    # The group key matches the eventual summary CSV dimensions and plot series.
    group_columns = [
        COLUMN_NAMES['metric_family'],
        COLUMN_NAMES['plv_type'],
        COLUMN_NAMES['relationship_code'],
        COLUMN_NAMES['relationship_description'],
        COLUMN_NAMES['scenario_group'],
        COLUMN_NAMES['scenario_variant'],
        COLUMN_NAMES['scenario_type'],
    ]
    for batch_df in _iter_parquet_batches(parquet_path, columns=columns, batch_size=batch_size):
        boxplot_df = _build_relationship_boxplot_dataframe(batch_df)
        if boxplot_df.empty:
            print("Processed boxplot batch with 0 rows")
            continue
        grouped = boxplot_df.groupby(group_columns, dropna=False)[COLUMN_NAMES['value']]

        # Accumulate raw value-per-cost values by group so quantiles can be computed after all batches are read.
        # Smuggled-chip rows are already batch-level sums, so they must be summed across batches instead.
        for group_key, values in grouped:
            group_key = tuple(group_key)
            if group_key[0] == SMUGGLED_CHIPS_METRIC_FAMILY:
                smuggled_chip_totals[group_key] = smuggled_chip_totals.get(group_key, 0.0) + float(values.sum())
            else:
                grouped_values.setdefault(group_key, []).extend(values.tolist())
        print(f"Accumulated {len(boxplot_df)} boxplot rows from current batch")

    for group_key, value in smuggled_chip_totals.items():
        grouped_values[group_key] = [value]

    if not grouped_values:
        empty_summary = _summarize_relationship_boxplot_dataframe(
            pd.DataFrame(columns=[*group_columns, COLUMN_NAMES['value']])
        )
        empty_summary.to_csv(output_csv_path, index=False)
        print(f"Wrote empty relationship boxplot summary to {output_csv_path}")
        return empty_summary

    summary_rows = []

    # Convert accumulated raw values into the distribution statistics used by the CSV and plots.
    for group_key, values in grouped_values.items():
        value_series = pd.Series(values, dtype="float64")
        summary_rows.append(
            {
                COLUMN_NAMES['metric_family']: group_key[0],
                COLUMN_NAMES['plv_type']: group_key[1],
                COLUMN_NAMES['relationship_code']: group_key[2],
                COLUMN_NAMES['relationship_description']: group_key[3],
                COLUMN_NAMES['scenario_group']: group_key[4],
                COLUMN_NAMES['scenario_variant']: group_key[5],
                COLUMN_NAMES['scenario_type']: group_key[6],
                COLUMN_NAMES['scenario_count']: int(value_series.count()),
                "Min": float(value_series.min()),
                "Q1": float(value_series.quantile(0.25)),
                "Median": float(value_series.median()),
                "Q3": float(value_series.quantile(0.75)),
                "Max": float(value_series.max()),
                "Mean": float(value_series.mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df[COLUMN_NAMES['plv_type']] = pd.Categorical(summary_df[COLUMN_NAMES['plv_type']], categories=PLV_VARIANT_ORDER, ordered=True)
    summary_df = summary_df.sort_values(group_columns).reset_index(drop=True)
    print(f"Writing relationship boxplot summary CSV to {output_csv_path}")
    summary_df.to_csv(output_csv_path, index=False)
    _render_relationship_boxplot_images(grouped_values, output_images_dir)
    return summary_df
