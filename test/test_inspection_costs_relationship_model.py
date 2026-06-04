from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from inspection_costs import (
    COLUMN_NAMES,
    RELATIONSHIP_MODEL_BASE_FEATURE_COLUMNS,
    RELATIONSHIP_MODEL_FEATURE_COLUMNS,
)
from inspection_costs_stage1 import _iter_parquet_batches


# Build numeric predictors and interactions for relationship-code modeling.
def _relationship_model_feature_frame(source_df: pd.DataFrame) -> pd.DataFrame:
    """Build numeric predictors and interactions for relationship-code modeling."""

    total_clusters = source_df[COLUMN_NAMES["total_clusters_in_mix"]].astype("float64")
    total_tests = source_df[COLUMN_NAMES["total_tests"]].astype("float64")
    total_chips = source_df[COLUMN_NAMES["total_component_chips"]].astype("float64")
    bad_records = source_df[COLUMN_NAMES["bad_records"]].astype("float64")
    physical_detected = source_df[COLUMN_NAMES["physical_inspection_total_diverted_chips_identified"]].astype("float64")
    plv_renting_detected = source_df["PLV Renting - Total Diverted Chips Identified"].astype("float64")
    plv_owning_detected = source_df["PLV Owning - Total Diverted Chips Identified"].astype("float64")
    physical_min_cost = source_df["Physical Inspection - Min Total Cost"].astype("float64")
    physical_max_cost = source_df["Physical Inspection - Max Total Cost"].astype("float64")
    plv_renting_min_cost = source_df["PLV Renting - Min Total Cost"].astype("float64")
    plv_owning_min_cost = source_df["PLV Owning - Min Total Cost"].astype("float64")

    # Features combine scenario scale, smuggling intensity, detected-share ratios, and cost intensity.
    features = pd.DataFrame(
        {
            "log_total_clusters_in_mix": np.log1p(total_clusters),
            "scenario_component_count": source_df[COLUMN_NAMES["scenario_component_count"]].astype("float64"),
            "log_total_tests": np.log1p(total_tests),
            "log_bad_records": np.log1p(bad_records),
            "share_diverted": source_df[COLUMN_NAMES["share_diverted"]].astype("float64"),
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


# Collect a balanced per-class sample of model features and targets from a parquet file.
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

        # Cap each class independently so common relationship codes do not dominate the fitted model.
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


# Fit a simple multinomial logistic-regression classifier using NumPy gradient descent.
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

    # Standardize predictors on the training split and add an intercept column.
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

    # Optimize multinomial cross-entropy with L2 regularization on non-intercept weights.
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

    # Summarize holdout performance with both overall and class-balanced metrics.
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


# Format a fitted relationship-code model, confusion matrix, and top coefficients as text.
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

    # Include both full-data counts and sampled counts so the report makes sampling effects visible.
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
        # Positive and negative coefficients identify predictors associated with each relationship code.
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


# Format a model-report section explaining why fitting was skipped for a target.
def _format_relationship_model_skip_report(
    *,
    target_column: str,
    full_class_counts: dict[str, int],
    sampled_targets: pd.Series,
    reason: str,
) -> str:
    lines = [
        f"Target: {target_column}",
        f"Sampled rows: {len(sampled_targets):,}",
        "",
        f"Model not fit: {reason}",
        "",
        "Full-data relationship-code counts:",
    ]
    for code in sorted(full_class_counts):
        lines.append(f"  {code}: {full_class_counts[code]:,}")

    sampled_counts = sampled_targets.value_counts().sort_index()
    if not sampled_counts.empty:
        lines.extend(["", "Sampled counts:"])
        for code, count in sampled_counts.items():
            lines.append(f"  {code}: {int(count):,}")

    return "\n".join(lines)


# Write the relationship-code statistical model report for each configured target column.
def write_relationship_code_model_report(
    parquet_path: Path,
    output_text_path: Path,
    *,
    target_columns: Optional[Iterable[str]] = None,
    max_rows_per_class: int = 5_000,
) -> str:
    if target_columns is None:
        target_columns = [
            "Physical vs PLV Renting Value Per Cost Relationship",
            "Physical vs PLV Owning Value Per Cost Relationship",
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
        # Each PLV variant gets its own target because the relationship labels are variant-specific.
        sampled_features, sampled_targets, full_class_counts = _collect_relationship_model_sample_from_parquet(
            parquet_path,
            target_column,
            max_rows_per_class=max_rows_per_class,
        )
        class_labels = sorted(sampled_targets.astype(str).unique().tolist())
        if len(class_labels) < 2:
            report_sections.append(
                _format_relationship_model_skip_report(
                    target_column=target_column,
                    full_class_counts=full_class_counts,
                    sampled_targets=sampled_targets,
                    reason="fewer than two observed classes were available after filtering",
                )
            )
        else:
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
