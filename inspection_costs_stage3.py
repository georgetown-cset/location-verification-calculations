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
    RELATIONSHIP_MODEL_BASE_FEATURE_COLUMNS,
    RELATIONSHIP_MODEL_FEATURE_COLUMNS,
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

def _parse_combo_values(combo_value: object) -> list[int]:
    if pd.isna(combo_value):
        return []
    combo_text = str(combo_value).strip()
    if not combo_text:
        return []
    try:
        return [int(part) for part in combo_text.split("-") if part]
    except ValueError as exc:
        raise ValueError(f"Unable to parse combo value {combo_value!r} into integers") from exc


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
        ],
    ].groupby(COLUMN_NAMES['scenario_id'], sort=False, observed=True).agg(
        component_cluster_size_min=(COLUMN_NAMES['cluster_size'], "min"),
        component_cluster_size_max=(COLUMN_NAMES['cluster_size'], "max"),
        component_cluster_size_mean=(COLUMN_NAMES['cluster_size'], "mean"),
        component_cluster_count_min=(COLUMN_NAMES['number_of_clusters'], "min"),
        component_cluster_count_max=(COLUMN_NAMES['number_of_clusters'], "max"),
        component_cluster_count_mean=(COLUMN_NAMES['number_of_clusters'], "mean"),
    ).reset_index()
    source_df = source_df.merge(component_summary, on=COLUMN_NAMES['scenario_id'], how='left')

    k_values = source_df[COLUMN_NAMES['k_combo']].map(_parse_combo_values)
    n_values = source_df[COLUMN_NAMES['n_combo']].map(_parse_combo_values)
    source_df["K Combo Min"] = k_values.map(lambda values: float(min(values)) if values else np.nan)
    source_df["K Combo Max"] = k_values.map(lambda values: float(max(values)) if values else np.nan)
    source_df["K Combo Sum"] = k_values.map(lambda values: float(sum(values)) if values else np.nan)
    source_df["N Combo Min"] = n_values.map(lambda values: float(min(values)) if values else np.nan)
    source_df["N Combo Max"] = n_values.map(lambda values: float(max(values)) if values else np.nan)
    source_df["N Combo Sum"] = n_values.map(lambda values: float(sum(values)) if values else np.nan)

    return source_df


def _summarize_relationship_rule_bounds(
    source_df: pd.DataFrame,
    *,
    relationship_code_column: str,
    plv_type: str,
) -> pd.DataFrame:
    relationship_labels = _plv_relationship_labels(plv_type)
    feature_groups = {
        COLUMN_NAMES['scenario_component_count']: "Scenario Summary",
        COLUMN_NAMES['total_clusters_in_mix']: "Scenario Summary",
        COLUMN_NAMES['total_tests']: "Scenario Summary",
        COLUMN_NAMES['total_component_chips']: "Scenario Summary",
        COLUMN_NAMES['bad_records']: "Scenario Summary",
        COLUMN_NAMES['number_of_clusters_with_smuggling']: "Scenario Summary",
        COLUMN_NAMES['share_diverted']: "Scenario Summary",
        COLUMN_NAMES['physical_inspection_chip_level_miss_prob']: "Scenario Summary",
        COLUMN_NAMES['plv_chip_level_miss_prob']: "Scenario Summary",
        "K Combo Min": "Combo Derived",
        "K Combo Max": "Combo Derived",
        "K Combo Sum": "Combo Derived",
        "N Combo Min": "Combo Derived",
        "N Combo Max": "Combo Derived",
        "N Combo Sum": "Combo Derived",
        "component_cluster_size_min": "Component Summary",
        "component_cluster_size_max": "Component Summary",
        "component_cluster_size_mean": "Component Summary",
        "component_cluster_count_min": "Component Summary",
        "component_cluster_count_max": "Component Summary",
        "component_cluster_count_mean": "Component Summary",
    }
    feature_columns = list(feature_groups)
    rows: list[dict[str, object]] = []

    relationship_values = source_df[relationship_code_column].astype(str)
    for relationship_code in RELATIONSHIP_CODE_ORDER:
        code_mask = relationship_values == relationship_code
        code_df = source_df.loc[code_mask]
        if code_df.empty:
            continue

        scenario_count = int(len(code_df))
        for feature_name in feature_columns:
            if feature_name not in code_df.columns:
                continue
            values = pd.to_numeric(code_df[feature_name], errors="coerce").dropna()
            if values.empty:
                continue

            observed_min = values.min()
            observed_max = values.max()
            lower_text = _format_rule_value(observed_min)
            upper_text = _format_rule_value(observed_max)
            if lower_text == upper_text:
                rule_text = f"{feature_name} = {lower_text}"
            else:
                rule_text = f"{lower_text} <= {feature_name} <= {upper_text}"

            rows.append(
                {
                    COLUMN_NAMES['plv_type']: plv_type,
                    COLUMN_NAMES['relationship_code']: relationship_code,
                    COLUMN_NAMES['relationship_description']: relationship_labels[relationship_code],
                    "Feature Group": feature_groups[feature_name],
                    "Feature": feature_name,
                    COLUMN_NAMES['scenario_count']: scenario_count,
                    "Distinct Values": int(values.nunique(dropna=True)),
                    "Observed Min": float(observed_min),
                    "Observed Max": float(observed_max),
                    "Rule": rule_text,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                COLUMN_NAMES['plv_type'],
                COLUMN_NAMES['relationship_code'],
                COLUMN_NAMES['relationship_description'],
                "Feature Group",
                "Feature",
                COLUMN_NAMES['scenario_count'],
                "Distinct Values",
                "Observed Min",
                "Observed Max",
                "Rule",
            ]
        )

    result[COLUMN_NAMES['plv_type']] = pd.Categorical(result[COLUMN_NAMES['plv_type']], categories=PLV_VARIANT_ORDER, ordered=True)
    result[COLUMN_NAMES['relationship_code']] = pd.Categorical(
        result[COLUMN_NAMES['relationship_code']],
        categories=RELATIONSHIP_CODE_ORDER,
        ordered=True,
    )
    result = result.sort_values(
        [COLUMN_NAMES['plv_type'], COLUMN_NAMES['relationship_code'], "Feature Group", "Feature"],
        kind="mergesort",
    ).reset_index(drop=True)
    return result


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
                "Feature Group",
                "Feature",
                COLUMN_NAMES['scenario_count'],
                "Distinct Values",
                "Observed Min",
                "Observed Max",
                "Rule",
            ]
        )
        empty_df.to_csv(output_csv_path, index=False)
        print(f"Wrote empty relationship rule summary for {label} to {output_csv_path}")
        return empty_df

    rule_source_df = _build_relationship_rule_frame(scenario_df, scenario_component_df)
    summary_frames: list[pd.DataFrame] = []
    for variant in _plv_variant_specs():
        plv_type = str(variant["plv_type"])
        relationship_code_column = f"Physical vs {plv_type} Benefit Per Dollar Relationship"
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


def _summarize_relationships_from_parquet(
    parquet_path: Path,
    batch_size: int = 65_536,
) -> pd.DataFrame:
    print(f"Summarizing relationship counts from cached costed parquet: {parquet_path}")
    summaries = []
    for variant in _plv_variant_specs():
        plv_type = str(variant["plv_type"])
        bpd_relationship_column = f"Physical vs {plv_type} Benefit Per Dollar Relationship"
        bpd_relationship_labels = _plv_relationship_labels(plv_type)
        benefit_per_dollar_counts: Counter[str] = Counter()

        for batch_df in _iter_parquet_batches(
            parquet_path,
            columns=[bpd_relationship_column],
            batch_size=batch_size,
        ):
            benefit_per_dollar_counts.update(batch_df[bpd_relationship_column].astype(str).tolist())

        benefit_per_dollar_summary = pd.DataFrame(
            {
                COLUMN_NAMES['plv_type']: [plv_type] * len(bpd_relationship_labels),
                COLUMN_NAMES['relationship_code']: list(bpd_relationship_labels.keys()),
                COLUMN_NAMES['benefit_per_dollar_relationship_description']: list(bpd_relationship_labels.values()),
                COLUMN_NAMES['benefit_per_dollar_scenario_count']: [
                    int(benefit_per_dollar_counts.get(code, 0)) for code in bpd_relationship_labels.keys()
                ],
            }
        )
        total_row = pd.DataFrame(
            {
                COLUMN_NAMES['plv_type']: [plv_type],
                COLUMN_NAMES['relationship_code']: ["Total"],
                COLUMN_NAMES['benefit_per_dollar_relationship_description']: ["All scenarios"],
                COLUMN_NAMES['benefit_per_dollar_scenario_count']: [int(sum(benefit_per_dollar_counts.values()))],
            }
        )
        summaries.append(pd.concat([benefit_per_dollar_summary, total_row], ignore_index=True))

    result = pd.concat(summaries, ignore_index=True)
    result[COLUMN_NAMES['plv_type']] = pd.Categorical(result[COLUMN_NAMES['plv_type']], categories=PLV_VARIANT_ORDER, ordered=True)
    relationship_sort_order = RELATIONSHIP_CODE_ORDER + ["Total"]
    result[COLUMN_NAMES['relationship_code']] = pd.Categorical(
        result[COLUMN_NAMES['relationship_code']],
        categories=relationship_sort_order,
        ordered=True,
    )
    return result.sort_values([COLUMN_NAMES['plv_type'], COLUMN_NAMES['relationship_code']]).reset_index(drop=True)


def _build_relationship_boxplot_dataframe(final_df: pd.DataFrame) -> pd.DataFrame:
    """Return tidy raw rows for box plots split by relationship category, PLV variant, and metric family."""

    boxplot_frames: list[pd.DataFrame] = []
    for variant in _plv_variant_specs():
        plv_type = str(variant["plv_type"])
        bpd_relationship_column = f"Physical vs {plv_type} Benefit Per Dollar Relationship"
        bpd_relationship_labels = _plv_relationship_labels(plv_type)
        benefit_per_dollar_source_columns = [
            column
            for column in [
                COLUMN_NAMES['mix_id'],
                COLUMN_NAMES['scenario_id'],
                bpd_relationship_column,
                "Physical - Min Benefit Per Dollar",
                "Physical - Max Benefit Per Dollar",
                f"{plv_type} - Min Benefit Per Dollar",
                f"{plv_type} - Max Benefit Per Dollar",
            ]
            if column in final_df.columns
        ]
        benefit_per_dollar_id_vars = [
            column
            for column in [COLUMN_NAMES['mix_id'], COLUMN_NAMES['scenario_id'], bpd_relationship_column]
            if column in benefit_per_dollar_source_columns
        ]
        benefit_per_dollar_df = final_df[benefit_per_dollar_source_columns].melt(
            id_vars=benefit_per_dollar_id_vars,
            value_vars=[
                "Physical - Min Benefit Per Dollar",
                "Physical - Max Benefit Per Dollar",
                f"{plv_type} - Min Benefit Per Dollar",
                f"{plv_type} - Max Benefit Per Dollar",
            ],
            var_name=COLUMN_NAMES['scenario_type'],
            value_name=COLUMN_NAMES['value'],
        )
        benefit_per_dollar_df[COLUMN_NAMES['metric_family']] = "Benefit Per Dollar"
        benefit_per_dollar_df[COLUMN_NAMES['plv_type']] = plv_type
        benefit_per_dollar_df[COLUMN_NAMES['relationship_code']] = benefit_per_dollar_df[bpd_relationship_column]
        benefit_per_dollar_df[COLUMN_NAMES['relationship_description']] = benefit_per_dollar_df[COLUMN_NAMES['relationship_code']].map(
            bpd_relationship_labels
        )
        benefit_per_dollar_df[COLUMN_NAMES['scenario_group']] = np.where(
            benefit_per_dollar_df[COLUMN_NAMES['scenario_type']].str.startswith("Physical"),
            "Physical Inspection",
            plv_type,
        )
        benefit_per_dollar_df[COLUMN_NAMES['scenario_variant']] = benefit_per_dollar_df[COLUMN_NAMES['scenario_type']].map(
            lambda scenario_type: _scenario_variant_label(scenario_type, plv_type)
        )

        smuggled_chips_df = (
            final_df.groupby(bpd_relationship_column, sort=False, observed=True)[COLUMN_NAMES['bad_records']]
            .sum()
            .reindex(RELATIONSHIP_CODE_ORDER)
            .dropna()
            .reset_index()
            .rename(
                columns={
                    bpd_relationship_column: COLUMN_NAMES['relationship_code'],
                    COLUMN_NAMES['bad_records']: COLUMN_NAMES['value'],
                }
            )
        )
        smuggled_chips_df[COLUMN_NAMES['plv_type']] = plv_type
        smuggled_chips_df[COLUMN_NAMES['metric_family']] = SMUGGLED_CHIPS_METRIC_FAMILY
        smuggled_chips_df[COLUMN_NAMES['relationship_description']] = smuggled_chips_df[COLUMN_NAMES['relationship_code']].map(
            bpd_relationship_labels
        )
        smuggled_chips_df[COLUMN_NAMES['scenario_group']] = SMUGGLED_CHIPS_SCENARIO_GROUP
        smuggled_chips_df[COLUMN_NAMES['scenario_variant']] = SMUGGLED_CHIPS_SCENARIO_VARIANT
        smuggled_chips_df[COLUMN_NAMES['scenario_type']] = SMUGGLED_CHIPS_SCENARIO_TYPE
        smuggled_chips_df[COLUMN_NAMES['mix_id']] = "All Scenarios"
        smuggled_chips_df[COLUMN_NAMES['scenario_id']] = "All Scenarios"
        smuggled_chips_df[COLUMN_NAMES['value']] = smuggled_chips_df[COLUMN_NAMES['value']].astype("float64")

        boxplot_frames.extend([benefit_per_dollar_df, smuggled_chips_df])

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


def _slugify_filename(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "_" for character in value)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _render_relationship_boxplot_images(
    grouped_values: dict[tuple[object, ...], list[float]],
    output_images_dir: Path,
) -> list[Path]:
    output_images_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    if not grouped_values:
        return saved_paths

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
        "Physical vs PLV Renting Benefit Per Dollar Relationship",
        "Physical vs PLV Owning Benefit Per Dollar Relationship",
        COLUMN_NAMES['bad_records'],
        *LONG_SCENARIO_VALUE_VARS,
    ]
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_values: dict[tuple[object, ...], list[float]] = {}
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
        for group_key, values in grouped:
            grouped_values.setdefault(tuple(group_key), []).extend(values.tolist())
        print(f"Accumulated {len(boxplot_df)} boxplot rows from current batch")

    if not grouped_values:
        empty_summary = _summarize_relationship_boxplot_dataframe(
            pd.DataFrame(columns=[*group_columns, COLUMN_NAMES['value']])
        )
        empty_summary.to_csv(output_csv_path, index=False)
        print(f"Wrote empty relationship boxplot summary to {output_csv_path}")
        return empty_summary

    summary_rows = []
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
def _relationship_model_feature_frame(source_df: pd.DataFrame) -> pd.DataFrame:
    """Build numeric predictors and interactions for relationship-code modeling."""

    total_clusters = source_df[COLUMN_NAMES['total_clusters_in_mix']].astype("float64")
    total_tests = source_df[COLUMN_NAMES['total_tests']].astype("float64")
    total_chips = source_df[COLUMN_NAMES['total_component_chips']].astype("float64")
    bad_records = source_df[COLUMN_NAMES['bad_records']].astype("float64")
    physical_detected = source_df[COLUMN_NAMES['physical_inspection_total_diverted_chips_identified']].astype("float64")
    plv_renting_detected = source_df["PLV Renting - Total Diverted Chips Identified"].astype("float64")
    plv_owning_detected = source_df["PLV Owning - Total Diverted Chips Identified"].astype("float64")
    physical_min_cost = source_df["Physical Inspection - Min Total Cost"].astype("float64")
    physical_max_cost = source_df["Physical Inspection - Max Total Cost"].astype("float64")
    plv_renting_min_cost = source_df["PLV Renting - Min Total Cost"].astype("float64")
    plv_owning_min_cost = source_df["PLV Owning - Min Total Cost"].astype("float64")

    features = pd.DataFrame(
        {
            "log_total_clusters_in_mix": np.log1p(total_clusters),
            "scenario_component_count": source_df[COLUMN_NAMES['scenario_component_count']].astype("float64"),
            "log_total_tests": np.log1p(total_tests),
            "log_bad_records": np.log1p(bad_records),
            "share_diverted": source_df[COLUMN_NAMES['share_diverted']].astype("float64"),
            "tests_per_cluster": total_tests / total_clusters.replace(0, np.nan),
            "bad_records_per_test": bad_records / total_tests.replace(0, np.nan),
            "physical_identified_share": physical_detected / bad_records.replace(0, np.nan),
            "plv_renting_identified_share": plv_renting_detected / bad_records.replace(0, np.nan),
            "plv_owning_identified_share": plv_owning_detected / bad_records.replace(0, np.nan),
            "log_physical_detected": np.log1p(physical_detected),
            "log_plv_renting_detected": np.log1p(plv_renting_detected),
            "log_plv_owning_detected": np.log1p(plv_owning_detected),
            "plv_renting_minus_physical_detected_share": (plv_renting_detected - physical_detected) / bad_records.replace(0, np.nan),
            "plv_owning_minus_physical_detected_share": (plv_owning_detected - physical_detected) / bad_records.replace(0, np.nan),
            "physical_to_plv_renting_detected_ratio": physical_detected / plv_renting_detected.replace(0, np.nan),
            "physical_to_plv_owning_detected_ratio": physical_detected / plv_owning_detected.replace(0, np.nan),
            "physical_min_cost_per_test": physical_min_cost / total_tests.replace(0, np.nan),
            "physical_cost_range_per_test": (physical_max_cost - physical_min_cost) / total_tests.replace(0, np.nan),
            "plv_renting_cost_per_chip": plv_renting_min_cost / total_chips.replace(0, np.nan),
            "plv_owning_cost_per_chip": plv_owning_min_cost / total_chips.replace(0, np.nan),
            "plv_cost_spread_per_chip": (plv_owning_min_cost - plv_renting_min_cost) / total_chips.replace(0, np.nan),
        },
        index=source_df.index,
    )
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features[RELATIONSHIP_MODEL_FEATURE_COLUMNS]


def _collect_relationship_model_sample_from_parquet(
    parquet_path: Path,
    target_column: str,
    *,
    max_rows_per_class: int = 5_000,
    batch_size: int = 65_536,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    columns = [*RELATIONSHIP_MODEL_BASE_FEATURE_COLUMNS, target_column]
    class_counts: Counter[str] = Counter()
    sampled_features: dict[str, list[pd.DataFrame]] = {}
    sampled_targets: dict[str, list[pd.Series]] = {}
    sampled_counts: Counter[str] = Counter()

    rng = np.random.default_rng(42)
    for batch_df in _iter_parquet_batches(parquet_path, columns=columns, batch_size=batch_size):
        target_values = batch_df[target_column].astype(str)
        class_counts.update(target_values.tolist())
        feature_df = _relationship_model_feature_frame(batch_df)
        for relationship_code, group_index in target_values.groupby(target_values).groups.items():
            remaining = max_rows_per_class - sampled_counts[relationship_code]
            if remaining <= 0:
                continue

            group_index_array = np.array(list(group_index))
            take = min(remaining, len(group_index_array))
            if take < len(group_index_array):
                group_index_array = rng.choice(group_index_array, size=take, replace=False)

            sampled_features.setdefault(relationship_code, []).append(feature_df.loc[group_index_array])
            sampled_targets.setdefault(relationship_code, []).append(target_values.loc[group_index_array])
            sampled_counts[relationship_code] += take

    if not sampled_features:
        return (
            pd.DataFrame(columns=RELATIONSHIP_MODEL_FEATURE_COLUMNS),
            pd.Series(dtype="object", name=target_column),
            dict(class_counts),
        )

    features = pd.concat(
        [frame for frames in sampled_features.values() for frame in frames],
        ignore_index=True,
    )
    targets = pd.concat(
        [series for series_list in sampled_targets.values() for series in series_list],
        ignore_index=True,
    ).rename(target_column)
    order = rng.permutation(len(targets))
    return features.iloc[order].reset_index(drop=True), targets.iloc[order].reset_index(drop=True), dict(class_counts)


def _fit_multinomial_logistic_regression(
    features: pd.DataFrame,
    targets: pd.Series,
    *,
    l2_penalty: float = 0.01,
    learning_rate: float = 0.2,
    max_iter: int = 400,
    test_fraction: float = 0.25,
) -> dict[str, object]:
    if features.empty or targets.empty:
        raise ValueError("Cannot fit relationship-code model without sampled rows.")

    class_labels = sorted(targets.astype(str).unique().tolist())
    if len(class_labels) < 2:
        raise ValueError("Cannot fit relationship-code model with fewer than two observed classes.")

    rng = np.random.default_rng(43)
    row_count = len(targets)
    shuffled_indices = rng.permutation(row_count)
    test_size = max(len(class_labels), int(row_count * test_fraction))
    test_size = min(test_size, row_count - len(class_labels))
    train_indices = shuffled_indices[test_size:]
    test_indices = shuffled_indices[:test_size]

    feature_matrix = features.to_numpy(dtype="float64")
    feature_mean = feature_matrix[train_indices].mean(axis=0)
    feature_std = feature_matrix[train_indices].std(axis=0)
    feature_std[feature_std == 0] = 1.0
    scaled_features = (feature_matrix - feature_mean) / feature_std
    design_matrix = np.column_stack([np.ones(row_count), scaled_features])

    class_to_index = {label: index for index, label in enumerate(class_labels)}
    y_indices = targets.astype(str).map(class_to_index).to_numpy(dtype="int64")
    train_x = design_matrix[train_indices]
    test_x = design_matrix[test_indices]
    train_y = y_indices[train_indices]
    test_y = y_indices[test_indices]

    weights = np.zeros((design_matrix.shape[1], len(class_labels)), dtype="float64")
    encoded_train_y = np.eye(len(class_labels), dtype="float64")[train_y]
    for _iteration in range(max_iter):
        logits = train_x @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = train_x.T @ (probabilities - encoded_train_y) / len(train_y)
        gradient[1:] += l2_penalty * weights[1:]
        weights -= learning_rate * gradient

    test_logits = test_x @ weights
    predictions = test_logits.argmax(axis=1)
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype="int64")
    for actual, predicted in zip(test_y, predictions):
        confusion[actual, predicted] += 1

    accuracy = float((predictions == test_y).mean())
    baseline_accuracy = float(np.bincount(test_y, minlength=len(class_labels)).max() / len(test_y))
    recalls = []
    precisions = []
    for class_index in range(len(class_labels)):
        true_positive = confusion[class_index, class_index]
        actual_count = confusion[class_index, :].sum()
        predicted_count = confusion[:, class_index].sum()
        recalls.append(float(true_positive / actual_count) if actual_count else 0.0)
        precisions.append(float(true_positive / predicted_count) if predicted_count else 0.0)

    return {
        "class_labels": class_labels,
        "weights": weights,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "train_rows": int(len(train_y)),
        "test_rows": int(len(test_y)),
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "macro_recall": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "confusion": confusion,
    }


def _format_relationship_model_report(
    *,
    target_column: str,
    sampled_features: pd.DataFrame,
    sampled_targets: pd.Series,
    full_class_counts: dict[str, int],
    model_result: dict[str, object],
) -> str:
    class_labels = model_result["class_labels"]
    weights = model_result["weights"]
    lines = [
        f"Target: {target_column}",
        f"Sampled rows: {len(sampled_targets):,}",
        f"Training rows: {model_result['train_rows']:,}",
        f"Holdout rows: {model_result['test_rows']:,}",
        f"Holdout accuracy: {model_result['accuracy']:.3f}",
        f"Majority-class baseline accuracy on holdout: {model_result['baseline_accuracy']:.3f}",
        f"Macro precision: {model_result['macro_precision']:.3f}",
        f"Macro recall: {model_result['macro_recall']:.3f}",
        "",
        "Full-data relationship-code counts:",
    ]
    for code in sorted(full_class_counts):
        lines.append(f"  {code}: {full_class_counts[code]:,}")

    lines.extend(["", "Balanced training-sample counts:"])
    sampled_counts = sampled_targets.value_counts().sort_index()
    for code, count in sampled_counts.items():
        lines.append(f"  {code}: {int(count):,}")

    lines.extend(["", "Holdout confusion matrix (rows=actual, columns=predicted):"])
    lines.append("  " + " ".join(f"{label:>8}" for label in class_labels))
    for label, row in zip(class_labels, model_result["confusion"]):
        lines.append(f"  {label:>8} " + " ".join(f"{int(value):>8}" for value in row))

    lines.extend(["", "Largest standardized coefficients by relationship code:"])
    for class_index, code in enumerate(class_labels):
        coefficients = pd.Series(weights[1:, class_index], index=sampled_features.columns)
        top_positive = coefficients.sort_values(ascending=False).head(5)
        top_negative = coefficients.sort_values(ascending=True).head(5)
        lines.append(f"  Code {code} positive associations:")
        for feature_name, coefficient in top_positive.items():
            lines.append(f"    {feature_name}: {coefficient:.3f}")
        lines.append(f"  Code {code} negative associations:")
        for feature_name, coefficient in top_negative.items():
            lines.append(f"    {feature_name}: {coefficient:.3f}")

    return "\n".join(lines)


def write_relationship_code_model_report(
    parquet_path: Path,
    output_text_path: Path,
    *,
    target_columns: Optional[Iterable[str]] = None,
    max_rows_per_class: int = 5_000,
) -> str:
    if target_columns is None:
        target_columns = [
            "Physical vs PLV Renting Benefit Per Dollar Relationship",
            "Physical vs PLV Owning Benefit Per Dollar Relationship",
        ]

    output_text_path.parent.mkdir(parents=True, exist_ok=True)
    report_sections = [
        "Relationship Code Statistical Model Report",
        "=" * 42,
        "",
        "Model: multinomial logistic regression fitted with NumPy gradient descent.",
        "Predictors: scenario scale, bad-record intensity, detection totals, cost scale, and engineered ratios between these inputs.",
        f"Sampling: up to {max_rows_per_class:,} rows per observed relationship code from the full costed parquet.",
        "",
        "Features:",
        *[f"  {feature}" for feature in RELATIONSHIP_MODEL_FEATURE_COLUMNS],
        "",
    ]

    for target_column in target_columns:
        sampled_features, sampled_targets, full_class_counts = _collect_relationship_model_sample_from_parquet(
            parquet_path,
            target_column,
            max_rows_per_class=max_rows_per_class,
        )
        model_result = _fit_multinomial_logistic_regression(sampled_features, sampled_targets)
        report_sections.append(
            _format_relationship_model_report(
                target_column=target_column,
                sampled_features=sampled_features,
                sampled_targets=sampled_targets,
                full_class_counts=full_class_counts,
                model_result=model_result,
            )
        )
        report_sections.append("")
        report_sections.append("-" * 42)
        report_sections.append("")

    report_text = "\n".join(report_sections).rstrip() + "\n"
    output_text_path.write_text(report_text, encoding="utf-8")
    print(f"Writing relationship code model report to {output_text_path}")
    return report_text
