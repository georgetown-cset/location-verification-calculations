from __future__ import annotations

import itertools
import math
from fractions import Fraction
from pathlib import Path
import shutil
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from inspection_costs import (
    ALL_M_VALS,
    COLUMN_NAMES,
    DATA_SAVED_DIR,
    DETECTION_LOOKUP_TABLE_PARQUET_PATH,
    MIN_DIVERTED_CHIPS,
    MIX_STEPS,
    PARQUET_COMPRESSION,
    PHYSICAL_INSPECTION_M,
    PLV_M,
    ALL_OUTPUT_DIR,
    PERFECT_INFORMATION_OUTPUT_DIR,
    MIXES_USED_CSV_PATH,
    SCENARIO_COLUMNS,
    SCENARIOS_PARQUET_PATH,
    SHARE_OF_CLUSTERS_WITH_SMUGGLING_VALS,
    MIN_SCENARIO_DIVERTED_CHIPS,
    TARGET_CHIPS,
    _as_float_list,
    _format_column_preview,
    _workflow_log,
)


# Compute the hypergeometric probability of drawing exactly x diverted chips in a sample of inspected chips.
def _hypergeom_pmf(x: int, cluster_size: int, diverted_chips: int, chips_inspected_per_cluster: int) -> float:
    if x < 0 or x > diverted_chips or x > chips_inspected_per_cluster:
        return 0.0
    if chips_inspected_per_cluster - x > cluster_size - diverted_chips:
        return 0.0
    if (
        cluster_size < 0
        or diverted_chips < 0
        or chips_inspected_per_cluster < 0
        or diverted_chips > cluster_size
        or chips_inspected_per_cluster > cluster_size
    ):
        return 0.0
    return math.comb(diverted_chips, x) * math.comb(cluster_size - diverted_chips, chips_inspected_per_cluster - x) / math.comb(cluster_size, chips_inspected_per_cluster)


# Calculate the chance that inspection detects at least one diverted chip in a cluster.
def p_detect_cluster_diversion(cluster_size: int, chips_inspected_per_cluster: int, diverted_chips: int, miss_prob: float) -> dict[str, float]:
    if diverted_chips == 0 or chips_inspected_per_cluster == 0:
        return {"p_success": 0.0, "p_failure": 1.0}

    x_values = range(
        max(0, chips_inspected_per_cluster - (cluster_size - diverted_chips)),
        min(diverted_chips, chips_inspected_per_cluster) + 1,
    )
    p_failure = sum(
        _hypergeom_pmf(x, cluster_size, diverted_chips, chips_inspected_per_cluster) * (miss_prob**x)
        for x in x_values
    )
    return {"p_success": 1 - p_failure, "p_failure": p_failure}


# Convert a detection probability into expected diverted chips identified for a cluster.
def expected_value_detected_diversion(cluster_size: int, chips_inspected_per_cluster: int, diverted_chips: int, miss_prob: float) -> dict[str, float]:
    p_detect = p_detect_cluster_diversion(
        cluster_size=cluster_size,
        chips_inspected_per_cluster=chips_inspected_per_cluster,
        diverted_chips=diverted_chips,
        miss_prob=miss_prob,
    )["p_success"]
    return {
        "diverted_chips_identified": p_detect * diverted_chips,
        "p_detect": p_detect,
    }


# Build and optionally persist the lookup table of detection outcomes for each cluster size, diverted-chip count, inspected-chip count, and miss-probability combination.
def build_detection_lookup_table(
    cluster_sizes: Iterable[int],
    K_vals: Iterable[int],
    chips_inspected_per_cluster_vals: Iterable[int],
    m_vals: Iterable[float] = ALL_M_VALS,
    parquet_path: Optional[str | Path] = DETECTION_LOOKUP_TABLE_PARQUET_PATH,
) -> pd.DataFrame:
    if parquet_path is not None:
        parquet_path = Path(parquet_path)

    rows = []
    for cluster_size in cluster_sizes:
        for chips_inspected_per_cluster in chips_inspected_per_cluster_vals:
            if chips_inspected_per_cluster > cluster_size:
                continue
            for diverted_chips in K_vals:
                if diverted_chips > cluster_size:
                    continue
                for m in m_vals:
                    physical_result = expected_value_detected_diversion(cluster_size, chips_inspected_per_cluster, diverted_chips, m)
                    plv_result = expected_value_detected_diversion(cluster_size, cluster_size, diverted_chips, m)
                    rows.append(
                        {
                            COLUMN_NAMES['cluster_size']: int(cluster_size),
                            COLUMN_NAMES['chips_inspected_per_cluster']: int(chips_inspected_per_cluster),
                            COLUMN_NAMES['bad_records']: int(diverted_chips),
                            COLUMN_NAMES['share_diverted']: diverted_chips / cluster_size,
                            COLUMN_NAMES['chip_level_miss_prob']: float(m),
                            COLUMN_NAMES['physical_inspection_p_detect']: physical_result["p_detect"],
                            COLUMN_NAMES['physical_inspection_diverted_chips_identified']: physical_result["diverted_chips_identified"],
                            COLUMN_NAMES['plv_p_detect']: plv_result["p_detect"],
                            COLUMN_NAMES['plv_diverted_chips_identified']: plv_result["diverted_chips_identified"],
                        }
                    )
    result = pd.DataFrame(rows)
    if parquet_path is not None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        _workflow_log("Stage 1 / Detection Lookup", f"Writing detection lookup table to {parquet_path}", kind="STEP")
        result.to_parquet(parquet_path, index=False, compression=PARQUET_COMPRESSION)
    return result


# Enumerate valid cluster-size mixes that sum to the target chip count and satisfy minimum constraints.
def build_mix_data(
    cluster_sizes: Iterable[int],
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = None,
    min_clusters_by_size: Optional[dict[int, int]] = None,
) -> list[list[dict[str, int]]]:
    if steps is None:
        steps = MIX_STEPS
    cluster_sizes = list(cluster_sizes)
    steps = list(steps)
    if min_clusters_by_size is None:
        min_clusters_by_size = {}

    # Validate the externally supplied minimums before generating any potentially large mix grid.
    unknown_cluster_sizes = sorted(set(min_clusters_by_size) - set(cluster_sizes))
    if unknown_cluster_sizes:
        raise ValueError(
            "build_mix_data: min_clusters_by_size contains cluster sizes that are not present in cluster_sizes: "
            f"{unknown_cluster_sizes}"
        )
    negative_minimums = {cluster_size: min_count for cluster_size, min_count in min_clusters_by_size.items() if min_count < 0}
    if negative_minimums:
        raise ValueError(f"build_mix_data: minimum cluster counts must be non-negative: {negative_minimums}")
    _workflow_log(
        "Stage 1 / Mix Generation",
        f"Generating mixes for {len(cluster_sizes)} cluster sizes and target {target_chips} chips",
        kind="START",
    )
    if min_clusters_by_size:
        _workflow_log(
            "Stage 1 / Mix Generation",
            f"Applying minimum cluster constraints {min_clusters_by_size}",
            kind="STEP",
        )
    valid_mixes: list[list[dict[str, int]]] = []
    seen_mix_ids: set[str] = set()
    for _mix_index, proportions in _iter_valid_mix_proportions(steps, len(cluster_sizes)):
        active_components = [(cluster_size, proportion) for cluster_size, proportion in zip(cluster_sizes, proportions) if proportion > 0]
        current_mix: list[dict[str, int]] = []

        # Round each active share into cluster counts, then absorb rounding residuals so
        # fractional step grids can still produce mixes with exactly target_chips.
        current_mix = _integer_mix_components_for_proportions(
            target_chips=target_chips,
            active_components=active_components,
        )

        if not current_mix:
            continue

        total_chips_in_mix = sum(
            int(component[COLUMN_NAMES['cluster_size']]) * int(component[COLUMN_NAMES['number_of_clusters']])
            for component in current_mix
        )
        if total_chips_in_mix != target_chips:
            continue

        if not _mix_satisfies_minimum_cluster_counts(current_mix, min_clusters_by_size):
            continue

        # Canonicalize and deduplicate mixes because different proportion paths can describe the same mix.
        current_mix.sort(key=lambda component: component[COLUMN_NAMES['cluster_size']])
        mix_id = build_mix_id(current_mix)
        if mix_id in seen_mix_ids:
            continue
        for component in current_mix:
            component[COLUMN_NAMES['mix_id']] = mix_id
        valid_mixes.append(current_mix)
        seen_mix_ids.add(mix_id)

    _workflow_log(
        "Stage 1 / Mix Generation",
        f"Completed with {len(valid_mixes)} valid mixes",
        kind="DONE",
    )
    return valid_mixes


# Write a flat CSV listing every valid mix and its component counts.
def write_mix_data_csv(
    cluster_sizes: Iterable[int],
    *,
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = None,
    min_clusters_by_size: Optional[dict[int, int]] = None,
    mix_data: Optional[list[list[dict[str, int]]]] = None,
    output_csv_path: str | Path = MIXES_USED_CSV_PATH,
) -> pd.DataFrame:
    cluster_sizes = list(cluster_sizes)
    output_csv_path = Path(output_csv_path)
    if mix_data is None:
        mix_data = build_mix_data(
            cluster_sizes=cluster_sizes,
            target_chips=target_chips,
            steps=steps,
            min_clusters_by_size=min_clusters_by_size,
        )

    rows: list[dict[str, object]] = []
    for mix_components in mix_data:
        mix_row: dict[str, object] = {
            COLUMN_NAMES['mix_id']: mix_components[0][COLUMN_NAMES['mix_id']],
            COLUMN_NAMES['mix_description']: build_mix_description(mix_components),
            COLUMN_NAMES['scenario_component_count']: len(mix_components),
            COLUMN_NAMES['total_clusters_in_mix']: sum(int(component[COLUMN_NAMES['number_of_clusters']]) for component in mix_components),
            COLUMN_NAMES['total_component_chips']: sum(
                int(component[COLUMN_NAMES['cluster_size']]) * int(component[COLUMN_NAMES['number_of_clusters']])
                for component in mix_components
            ),
        }
        for cluster_size in cluster_sizes:
            mix_row[f"Cluster Size {int(cluster_size)} Count"] = 0
        for component in mix_components:
            cluster_size = int(component[COLUMN_NAMES['cluster_size']])
            mix_row[f"Cluster Size {cluster_size} Count"] = int(component[COLUMN_NAMES['number_of_clusters']])
        rows.append(mix_row)

    result = pd.DataFrame(rows)
    if not result.empty:
        ordered_columns = [
            COLUMN_NAMES['mix_id'],
            COLUMN_NAMES['mix_description'],
            COLUMN_NAMES['scenario_component_count'],
            COLUMN_NAMES['total_clusters_in_mix'],
            COLUMN_NAMES['total_component_chips'],
            *[f"Cluster Size {int(cluster_size)} Count" for cluster_size in cluster_sizes],
        ]
        result = result.loc[:, ordered_columns]

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv_path, index=False)
    _workflow_log("Stage 1 / Mix Generation", f"Wrote mix summary CSV to {output_csv_path}", kind="STEP")
    return result


# Convert active mix proportions to integer cluster counts that exactly sum to target_chips.
def _integer_mix_components_for_proportions(
    *,
    target_chips: int,
    active_components: list[tuple[int, float]],
) -> list[dict[str, int]]:
    current_mix: list[dict[str, int]] = []
    for cluster_size, proportion in active_components:
        count = _integer_cluster_count(target_chips, proportion, cluster_size)
        if count > 0:
            current_mix.append(
                {
                    COLUMN_NAMES['cluster_size']: int(cluster_size),
                    COLUMN_NAMES['number_of_clusters']: count,
                }
            )

    if not current_mix:
        return []

    total_chips_in_mix = sum(
        int(component[COLUMN_NAMES['cluster_size']]) * int(component[COLUMN_NAMES['number_of_clusters']])
        for component in current_mix
    )
    residual_chips = int(target_chips) - int(total_chips_in_mix)
    if residual_chips == 0:
        return current_mix

    # Independent rounding can miss the exact target when MIX_STEP_SIZE does not divide
    # target_chips cleanly. Adjust one active bucket, preferring the smallest bucket to
    # keep the proportional distortion as fine-grained as possible.
    for component in sorted(current_mix, key=lambda item: int(item[COLUMN_NAMES['cluster_size']])):
        cluster_size = int(component[COLUMN_NAMES['cluster_size']])
        if residual_chips % cluster_size != 0:
            continue
        cluster_delta = residual_chips // cluster_size
        adjusted_count = int(component[COLUMN_NAMES['number_of_clusters']]) + cluster_delta
        if adjusted_count <= 0:
            continue
        component[COLUMN_NAMES['number_of_clusters']] = int(adjusted_count)
        return current_mix

    return []


# Convert a mix proportion to a rounded cluster count.
def _integer_cluster_count(
    target_chips: int,
    proportion: float,
    cluster_size: int,
) -> int:
    count = (target_chips * proportion) / cluster_size
    return int(round(count))


# Check whether a proposed mix includes enough clusters in each constrained size bucket.
def _mix_satisfies_minimum_cluster_counts(
    mix_components: list[dict[str, int]],
    min_clusters_by_size: dict[int, int],
) -> bool:
    if not min_clusters_by_size:
        return True

    counts_by_size = {
        int(component[COLUMN_NAMES['cluster_size']]): int(component[COLUMN_NAMES['number_of_clusters']])
        for component in mix_components
    }
    return all(
        counts_by_size.get(int(cluster_size), 0) >= int(min_count)
        for cluster_size, min_count in min_clusters_by_size.items()
    )


# Yield proportion vectors whose entries sum to one, using integer step units when possible.
def _iter_valid_mix_proportions(steps: list[float], dimensions: int) -> Iterable[tuple[int, tuple[float, ...]]]:
    if not steps or dimensions <= 0:
        return

    integer_units = _integer_step_units(steps)
    if integer_units is None:
        yield from _iter_valid_mix_proportions_brute_force(steps, dimensions)
        return

    units, target_units, unit_tolerance = integer_units
    if min(units) < 0:
        yield from _iter_valid_mix_proportions_brute_force(steps, dimensions)
        return

    suffix_max = [0] * (dimensions + 1)
    max_unit = max(units)
    for depth in range(dimensions - 1, -1, -1):
        suffix_max[depth] = suffix_max[depth + 1] + max_unit

    base = len(steps)
    index_stack: list[int] = []
    proportion_stack: list[float] = []

    # Recursively prune impossible partial proportion vectors before materializing full products.
    def visit(depth: int, unit_total: int) -> Iterable[tuple[int, tuple[float, ...]]]:
        remaining = dimensions - depth
        if unit_total - unit_tolerance > target_units or unit_total + suffix_max[depth] + unit_tolerance < target_units:
            return
        if depth == dimensions:
            if abs(unit_total - target_units) <= unit_tolerance and np.isclose(sum(proportion_stack), 1.0):
                yield _product_index(index_stack, base), tuple(proportion_stack)
            return

        for step_index, (step, unit) in enumerate(zip(steps, units)):
            if unit_total + unit - unit_tolerance > target_units:
                continue
            if unit_total + unit + max_unit * (remaining - 1) + unit_tolerance < target_units:
                continue
            index_stack.append(step_index)
            proportion_stack.append(step)
            yield from visit(depth + 1, unit_total + unit)
            proportion_stack.pop()
            index_stack.pop()

    yield from visit(0, 0)


# Convert floating-point mix steps into exact integer units when the step grid supports it.
def _integer_step_units(steps: list[float]) -> Optional[tuple[list[int], int, int]]:
    fractions = [Fraction(float(step)).limit_denominator(1_000_000) for step in steps]
    denominators = [fraction.denominator for fraction in fractions]
    scale = math.lcm(*denominators)
    target_units = scale
    unit_tolerance = math.ceil((1e-08 + 1e-05) * scale)
    units = [int(fraction * scale) for fraction in fractions]

    if all(np.isclose(unit / scale, step) for unit, step in zip(units, steps)):
        return units, target_units, unit_tolerance
    return None


# Fall back to brute-force product enumeration for step grids that cannot be represented as integer units.
def _iter_valid_mix_proportions_brute_force(steps: list[float], dimensions: int) -> Iterable[tuple[int, tuple[float, ...]]]:
    for mix_index, proportions in enumerate(itertools.product(steps, repeat=dimensions)):
        if np.isclose(sum(proportions), 1.0):
            yield mix_index, proportions


# Convert a vector of step indices into the same flat index used by itertools.product enumeration.
def _product_index(indices: list[int], base: int) -> int:
    mix_index = 0
    for index in indices:
        mix_index = mix_index * base + index
    return mix_index


# Build a readable description of the cluster sizes and counts in a mix.
def build_mix_description(mix_components: list[dict[str, int]]) -> str:
    return " + ".join(f"{component[COLUMN_NAMES['number_of_clusters']]}x(N={component[COLUMN_NAMES['cluster_size']]})" for component in mix_components)


# Build a stable identifier from the sorted cluster sizes and counts in a mix.
def build_mix_id(mix_components: list[dict[str, int]]) -> str:
    mix_components = sorted(mix_components, key=lambda component: component[COLUMN_NAMES['cluster_size']])
    characteristics = "_".join(
        f"N{component[COLUMN_NAMES['cluster_size']]}-C{component[COLUMN_NAMES['number_of_clusters']]}"
        for component in mix_components
    )
    return f"ClusterMix_{characteristics}"


# Estimate how many clusters in a component contain smuggling activity.
def _number_of_clusters_with_smuggling(number_of_clusters: int, share_of_clusters_with_smuggling: float) -> int:
    if number_of_clusters <= 1:
        return int(number_of_clusters)
    return max(1, int(math.floor(number_of_clusters * share_of_clusters_with_smuggling + 0.5)))


# Reshape the detection lookup table into nested dictionaries for fast scenario generation.
def _group_detection_lookup(detection_lookup_table: pd.DataFrame) -> dict[tuple[int, int], dict[int, dict[float, tuple[float, float, float, float]]]]:
    grouped: dict[tuple[int, int], dict[int, dict[float, tuple[float, float, float, float]]]] = {}
    columns = [
        COLUMN_NAMES['cluster_size'],
        COLUMN_NAMES['bad_records'],
        COLUMN_NAMES['chips_inspected_per_cluster'],
        COLUMN_NAMES['chip_level_miss_prob'],
        COLUMN_NAMES['physical_inspection_p_detect'],
        COLUMN_NAMES['physical_inspection_diverted_chips_identified'],
        COLUMN_NAMES['plv_p_detect'],
        COLUMN_NAMES['plv_diverted_chips_identified'],
    ]
    for row in detection_lookup_table.loc[:, columns].itertuples(index=False, name=None):
        cluster_size, diverted_chips, chips_inspected_per_cluster, m, physical_p_detect, physical_identified, plv_p_detect, plv_identified = row
        grouped.setdefault((int(cluster_size), int(diverted_chips)), {}).setdefault(int(chips_inspected_per_cluster), {})[float(m)] = (
            float(physical_p_detect),
            float(physical_identified),
            float(plv_p_detect),
            float(plv_identified),
        )
    return grouped


# Generate scenario-component rows for every valid mix, smuggling share, diversion amount, and inspected-chip count.
def build_scenarios(
    cluster_sizes: Iterable[int],
    detection_lookup_table: pd.DataFrame,
    k_vals: Iterable[int],
    min_scenario_diverted_chips: int = MIN_SCENARIO_DIVERTED_CHIPS,
    min_clusters_by_size: Optional[dict[int, int]] = None,
    share_of_clusters_with_smuggling_vals: Iterable[float] = SHARE_OF_CLUSTERS_WITH_SMUGGLING_VALS,
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = None,
    mix_data: Optional[list[list[dict[str, int]]]] = None,
    output_parquet_path: Optional[str | Path] = SCENARIOS_PARQUET_PATH,
    return_dataframe: Optional[bool] = None,
) -> pd.DataFrame:
    cluster_sizes = list(cluster_sizes)
    k_vals = list(k_vals)
    min_scenario_diverted_chips = int(min_scenario_diverted_chips)
    if min_scenario_diverted_chips < 0:
        raise ValueError("build_scenarios: min_scenario_diverted_chips must be non-negative")
    share_of_clusters_with_smuggling_vals = _as_float_list(share_of_clusters_with_smuggling_vals)
    _workflow_log(
        "Stage 1 / Scenarios",
        f"Starting with {len(cluster_sizes)} cluster sizes and {len(k_vals)} K values",
        kind="START",
    )
    if mix_data is None:
        mix_data = build_mix_data(
            cluster_sizes=cluster_sizes,
            target_chips=target_chips,
            steps=steps,
            min_clusters_by_size=min_clusters_by_size,
        )
    _workflow_log(
        "Stage 1 / Scenarios",
        f"Received {len(mix_data)} mixes",
        kind="STEP",
    )
    detection_grouped = _group_detection_lookup(detection_lookup_table)

    # Pre-filter K values by cluster size so later Cartesian products never include impossible diversions.
    k_options_by_cluster_size = {
        int(cluster_size): [int(k) for k in k_vals if k <= cluster_size]
        for cluster_size in cluster_sizes
    }

    if return_dataframe is None:
        return_dataframe = output_parquet_path is None

    parquet_path = Path(output_parquet_path) if output_parquet_path is not None else None
    total_scenarios = 0
    total_scenario_component_rows = 0

    if parquet_path is not None:
        component_dataset_path = parquet_path
        _ensure_scenarios_dataset_path(component_dataset_path)
        for mix_number, mix_components in enumerate(mix_data, start=1):
            # Each mix is generated and written independently to keep peak memory use bounded.
            mix_id = mix_components[0][COLUMN_NAMES['mix_id']]
            mix_scenario_component_columns, mix_scenario_count, mix_scenario_component_count = _build_mix_records(
                mix_components=mix_components,
                mix_number=mix_number,
                mix_count=len(mix_data),
                detection_grouped=detection_grouped,
                k_options_by_cluster_size=k_options_by_cluster_size,
                min_scenario_diverted_chips=min_scenario_diverted_chips,
                share_of_clusters_with_smuggling_vals=share_of_clusters_with_smuggling_vals,
                target_chips=target_chips,
            )
            _write_scenario_dataset_to_parquet(
                dataset_path=component_dataset_path,
                mix_id=mix_id,
                columns=mix_scenario_component_columns,
            )
            mix_summary_count = mix_scenario_count
            total_scenarios += mix_summary_count
            total_scenario_component_rows += mix_scenario_component_count

        _workflow_log(
            "Stage 1 / Scenarios",
            f"Completed with {total_scenarios} new scenario rows and {total_scenario_component_rows} new scenario-component rows",
            kind="DONE",
        )

        if return_dataframe:
            # Read the dataset back only when callers need an in-memory frame for downstream processing.
            return pd.read_parquet(component_dataset_path)
        return pd.DataFrame(columns=SCENARIO_COLUMNS)

    # In-memory mode is primarily useful for tests or small parameter grids.
    mix_frames: list[pd.DataFrame] = []
    for mix_number, mix_components in enumerate(mix_data, start=1):
        mix_scenario_component_columns, mix_scenario_count, mix_scenario_component_count = _build_mix_records(
            mix_components=mix_components,
            mix_number=mix_number,
            mix_count=len(mix_data),
            detection_grouped=detection_grouped,
            k_options_by_cluster_size=k_options_by_cluster_size,
            min_scenario_diverted_chips=min_scenario_diverted_chips,
            share_of_clusters_with_smuggling_vals=share_of_clusters_with_smuggling_vals,
            target_chips=target_chips,
        )
        if return_dataframe and mix_scenario_component_columns is not None:
            mix_frames.append(
                pd.DataFrame({column_name: mix_scenario_component_columns[column_name] for column_name in SCENARIO_COLUMNS})
            )
        total_scenarios += mix_scenario_count
        total_scenario_component_rows += mix_scenario_component_count

    _workflow_log(
        "Stage 1 / Scenarios",
        f"Completed with {total_scenarios} new scenario rows and {total_scenario_component_rows} new scenario-component rows",
        kind="DONE",
    )

    if return_dataframe:
        if not mix_frames:
            return pd.DataFrame(columns=SCENARIO_COLUMNS)
        if len(mix_frames) == 1:
            return mix_frames[0]
        return pd.concat(mix_frames, ignore_index=True)
    return pd.DataFrame(columns=SCENARIO_COLUMNS)


# Write one mix's scenario-component columns as a parquet fragment in the scenario dataset.
def _write_scenario_dataset_to_parquet(
    *,
    dataset_path: Path,
    mix_id: str,
    columns: Optional[dict[str, np.ndarray]],
) -> list[Path]:
    pq, pa = _import_pyarrow_parquet()
    _ensure_scenarios_dataset_path(dataset_path)
    mix_parquet_path = _mix_parquet_path(dataset_path, mix_id)
    _clear_existing_mix_files(mix_parquet_path)
    if not columns:
        table = _empty_arrow_table(pa, _final_scenario_component_arrow_schema(pa))
        pq.write_table(table, mix_parquet_path, compression=PARQUET_COMPRESSION)
        return [mix_parquet_path]

    schema = _final_scenario_component_arrow_schema(pa)
    table = _columns_to_arrow_table(columns, pa, schema)
    pq.write_table(table, mix_parquet_path, compression=PARQUET_COMPRESSION)
    return [mix_parquet_path]


# Build cached cross-product arrays for one tuple of per-component inspected-chip option lists.
def _build_inspected_combo_cache(
    signature: tuple[tuple[int, ...], ...],
    *,
    cluster_sizes_arr: np.ndarray,
    num_clusters_arr: np.ndarray,
    num_smug_arr: np.ndarray,
    component_chips_arr: np.ndarray,
) -> dict[str, object]:
    component_count = len(signature)
    option_arrays = [np.asarray(options, dtype=np.int64) for options in signature]

    # Enumerate the inspected-chip cross product in itertools.product order (last component fastest).
    index_grids = np.meshgrid(*[np.arange(options.size) for options in option_arrays], indexing="ij")
    option_index_grid = np.stack([grid.reshape(-1) for grid in index_grids], axis=1)
    combo_count = option_index_grid.shape[0]
    inspected_combos = np.stack(
        [option_arrays[component][option_index_grid[:, component]] for component in range(component_count)],
        axis=1,
    )

    combo_rows = inspected_combos.tolist()
    inspected_combo_strs = ["-".join(map(str, row)) for row in combo_rows]

    inspected_flat = inspected_combos.reshape(-1)
    num_clusters_flat = np.tile(num_clusters_arr, combo_count)
    return {
        "combo_count": combo_count,
        "inspected_combo_tuples": [tuple(row) for row in combo_rows],
        "inspected_combo_strs": inspected_combo_strs,
        "inspected_flat": inspected_flat,
        "option_index_flat": option_index_grid.reshape(-1),
        "component_index_flat": np.tile(np.arange(component_count, dtype=np.int64), combo_count),
        "cluster_size_flat": np.tile(cluster_sizes_arr, combo_count),
        "num_clusters_flat": num_clusters_flat,
        "num_smug_flat": np.tile(num_smug_arr, combo_count),
        "total_component_chips_flat": np.tile(component_chips_arr, combo_count),
        "total_tests_flat": num_clusters_flat * inspected_flat,
        "inspected_combo_str_flat": np.repeat(np.array(inspected_combo_strs, dtype=object), component_count),
    }


# Gather the per-option detection metrics for one component and diverted-chip count.
def _component_detection_vectors(
    detection_grouped: dict[tuple[int, int], dict[int, dict[float, tuple[float, float, float, float]]]],
    cluster_size: int,
    k_val: int,
    options: tuple[int, ...],
    physical_m_val: float,
    plv_m_val: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    detection_info_by_i = detection_grouped[(cluster_size, k_val)]
    physical_p = np.empty(len(options), dtype=np.float64)
    physical_identified = np.empty(len(options), dtype=np.float64)
    plv_p = np.empty(len(options), dtype=np.float64)
    plv_identified = np.empty(len(options), dtype=np.float64)
    for option_index, chips_inspected_per_cluster in enumerate(options):
        detection_info_by_m = detection_info_by_i.get(int(chips_inspected_per_cluster), {})
        physical_info = detection_info_by_m.get(float(physical_m_val))
        if physical_info is None:
            raise ValueError(
                "Missing physical-inspection detection lookup row for "
                f"cluster_size={cluster_size}, k={k_val}, "
                f"chips_inspected_per_cluster={chips_inspected_per_cluster}, m={physical_m_val}"
            )
        plv_info = detection_info_by_m.get(float(plv_m_val))
        if plv_info is None:
            raise ValueError(
                "Missing PLV detection lookup row for "
                f"cluster_size={cluster_size}, k={k_val}, "
                f"chips_inspected_per_cluster={chips_inspected_per_cluster}, m={plv_m_val}"
            )
        physical_p[option_index] = physical_info[0]
        physical_identified[option_index] = physical_info[1]
        plv_p[option_index] = plv_info[2]
        plv_identified[option_index] = plv_info[3]
    return physical_p, physical_identified, plv_p, plv_identified


# Expand one cluster mix into per-column scenario-component arrays and count the scenario rows produced.
def _build_mix_records(
    *,
    mix_components: list[dict[str, int]],
    mix_number: int,
    mix_count: int,
    detection_grouped: dict[tuple[int, int], dict[int, dict[float, tuple[float, float, float, float]]]],
    k_options_by_cluster_size: dict[int, list[int]],
    min_scenario_diverted_chips: int,
    share_of_clusters_with_smuggling_vals: Iterable[float],
    target_chips: int,
) -> tuple[Optional[dict[str, np.ndarray]], int, int]:
    mix_id = mix_components[0][COLUMN_NAMES['mix_id']]
    mix_description = build_mix_description(mix_components)
    total_clusters_in_mix = sum(component[COLUMN_NAMES['number_of_clusters']] for component in mix_components)
    scenario_component_count = len(mix_components)
    share_of_clusters_with_smuggling_vals = list(share_of_clusters_with_smuggling_vals)
    physical_m_val = PHYSICAL_INSPECTION_M
    plv_m_val = PLV_M
    scenario_counter = 0
    total_row_count = 0
    unique_k_combos: set[tuple[int, ...]] = set()
    unique_chips_inspected_per_cluster_combos: set[tuple[int, ...]] = set()
    column_chunks: dict[str, list[np.ndarray]] = {column_name: [] for column_name in SCENARIO_COLUMNS}

    for share_of_clusters_with_smuggling in share_of_clusters_with_smuggling_vals:
        # Component metadata packages cluster size, cluster count, smuggling count, and valid K options.
        component_data = [
            (
                component[COLUMN_NAMES['cluster_size']],
                component[COLUMN_NAMES['number_of_clusters']],
                _number_of_clusters_with_smuggling(
                    component[COLUMN_NAMES['number_of_clusters']],
                    share_of_clusters_with_smuggling,
                ),
                k_options_by_cluster_size[component[COLUMN_NAMES['cluster_size']]],
            )
            for component in mix_components
        ]
        k_options_per_component = [
            [
                k_val
                for k_val in component[3]
                if k_val == 0 or k_val >= MIN_DIVERTED_CHIPS or k_val == component[0]
            ]
            for component in component_data
        ]
        cluster_sizes_arr = np.array([component[0] for component in component_data], dtype=np.int64)
        num_clusters_arr = np.array([component[1] for component in component_data], dtype=np.int64)
        num_smug_arr = np.array([component[2] for component in component_data], dtype=np.int64)
        component_chips_arr = cluster_sizes_arr * num_clusters_arr

        # Inspected-option cross products and detection vectors repeat heavily across K combos, so cache them.
        inspected_cache_by_signature: dict[tuple[tuple[int, ...], ...], dict[str, object]] = {}
        detection_vectors_by_component_k: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        # A scenario chooses one diverted-chip count for each component in the mix.
        for k_combo in itertools.product(*k_options_per_component):
            scenario_diverted_chips = sum(
                int(k_val) * int(num_clusters_with_smuggling)
                for (_cluster_size_comp, _num_clusters_comp, num_clusters_with_smuggling, _), k_val
                in zip(component_data, k_combo)
            )
            if scenario_diverted_chips < min_scenario_diverted_chips:
                continue
            k_combo = tuple(int(k_val) for k_val in k_combo)

            chips_inspected_per_cluster_options_per_component: list[tuple[int, ...]] = []
            for (cluster_size_comp, _num_clusters_comp, _num_clusters_with_smuggling, _), k_val in zip(component_data, k_combo):
                # Inspected-chip options come from the detection table and are keyed by the selected cluster size and K.
                chips_inspected_per_cluster_options = detection_grouped.get((cluster_size_comp, int(k_val)), {})
                if not chips_inspected_per_cluster_options:
                    break
                chips_inspected_per_cluster_options_per_component.append(tuple(sorted(chips_inspected_per_cluster_options)))
            else:
                unique_k_combos.add(k_combo)
                signature = tuple(chips_inspected_per_cluster_options_per_component)
                inspected_cache = inspected_cache_by_signature.get(signature)
                if inspected_cache is None:
                    inspected_cache = _build_inspected_combo_cache(
                        signature,
                        cluster_sizes_arr=cluster_sizes_arr,
                        num_clusters_arr=num_clusters_arr,
                        num_smug_arr=num_smug_arr,
                        component_chips_arr=component_chips_arr,
                    )
                    inspected_cache_by_signature[signature] = inspected_cache
                    unique_chips_inspected_per_cluster_combos.update(inspected_cache["inspected_combo_tuples"])

                # Look up the four detection metrics for every component option, padded into one matrix per metric.
                max_option_count = max(len(options) for options in signature)
                physical_p_matrix = np.zeros((scenario_component_count, max_option_count), dtype=np.float64)
                physical_identified_matrix = np.zeros_like(physical_p_matrix)
                plv_p_matrix = np.zeros_like(physical_p_matrix)
                plv_identified_matrix = np.zeros_like(physical_p_matrix)
                for component_index, ((cluster_size_comp, _num_clusters_comp, _num_smug_comp, _), k_val, options) in enumerate(
                    zip(component_data, k_combo, signature)
                ):
                    vectors = detection_vectors_by_component_k.get((component_index, k_val))
                    if vectors is None:
                        vectors = _component_detection_vectors(
                            detection_grouped,
                            cluster_size_comp,
                            k_val,
                            options,
                            physical_m_val,
                            plv_m_val,
                        )
                        detection_vectors_by_component_k[(component_index, k_val)] = vectors
                    physical_p_matrix[component_index, : len(options)] = vectors[0]
                    physical_identified_matrix[component_index, : len(options)] = vectors[1]
                    plv_p_matrix[component_index, : len(options)] = vectors[2]
                    plv_identified_matrix[component_index, : len(options)] = vectors[3]

                component_index_flat = inspected_cache["component_index_flat"]
                option_index_flat = inspected_cache["option_index_flat"]
                physical_p_flat = physical_p_matrix[component_index_flat, option_index_flat]
                physical_identified_flat = physical_identified_matrix[component_index_flat, option_index_flat]
                plv_p_flat = plv_p_matrix[component_index_flat, option_index_flat]
                plv_identified_flat = plv_identified_matrix[component_index_flat, option_index_flat]

                combo_count = inspected_cache["combo_count"]
                row_count = combo_count * scenario_component_count
                num_smug_flat = inspected_cache["num_smug_flat"]
                k_combo_str = "-".join(map(str, k_combo))
                k_arr = np.array(k_combo, dtype=np.int64)
                scenario_id_prefix = f"{mix_id}_s{share_of_clusters_with_smuggling}_K{k_combo_str}"
                scenario_ids = np.array(
                    [f"{scenario_id_prefix}_i{inspected_str}" for inspected_str in inspected_cache["inspected_combo_strs"]],
                    dtype=object,
                )

                chunk = {
                    COLUMN_NAMES['mix_id']: np.full(row_count, mix_id, dtype=object),
                    COLUMN_NAMES['scenario_id']: np.repeat(scenario_ids, scenario_component_count),
                    COLUMN_NAMES['mix_description']: np.full(row_count, mix_description, dtype=object),
                    COLUMN_NAMES['total_clusters_in_mix']: np.full(row_count, total_clusters_in_mix, dtype=np.int64),
                    COLUMN_NAMES['total_tests']: inspected_cache["total_tests_flat"],
                    COLUMN_NAMES['total_component_chips']: inspected_cache["total_component_chips_flat"],
                    COLUMN_NAMES['scenario_component_count']: np.full(row_count, scenario_component_count, dtype=np.int64),
                    COLUMN_NAMES['k_combo']: np.full(row_count, k_combo_str, dtype=object),
                    COLUMN_NAMES['chips_inspected_per_cluster_combo']: inspected_cache["inspected_combo_str_flat"],
                    COLUMN_NAMES['share_of_clusters_with_smuggling']: np.full(row_count, share_of_clusters_with_smuggling, dtype=np.float64),
                    COLUMN_NAMES['cluster_size']: inspected_cache["cluster_size_flat"],
                    COLUMN_NAMES['bad_records']: np.tile(k_arr, combo_count),
                    COLUMN_NAMES['total_bad_records']: np.tile(k_arr * num_smug_arr, combo_count),
                    COLUMN_NAMES['number_of_clusters']: inspected_cache["num_clusters_flat"],
                    COLUMN_NAMES['number_of_clusters_with_smuggling']: num_smug_flat,
                    COLUMN_NAMES['chips_inspected_per_cluster']: inspected_cache["inspected_flat"],
                    COLUMN_NAMES['physical_inspection_chip_level_miss_prob']: np.full(row_count, physical_m_val, dtype=np.float64),
                    COLUMN_NAMES['physical_inspection_p_detect']: physical_p_flat,
                    COLUMN_NAMES['physical_inspection_diverted_chips_identified']: physical_identified_flat,
                    COLUMN_NAMES['plv_chip_level_miss_prob']: np.full(row_count, plv_m_val, dtype=np.float64),
                    COLUMN_NAMES['plv_p_detect']: plv_p_flat,
                    COLUMN_NAMES['plv_diverted_chips_identified']: plv_identified_flat,
                    COLUMN_NAMES['physical_inspection_total_diverted_chips_identified']: physical_identified_flat * num_smug_flat,
                    COLUMN_NAMES['plv_total_diverted_chips_identified']: plv_identified_flat * num_smug_flat,
                }
                for column_name, values in chunk.items():
                    column_chunks[column_name].append(values)

                scenario_counter += combo_count
                total_row_count += row_count

    _workflow_log(
        "Stage 1 / Scenarios",
        (
            f"{mix_id} ({mix_number} / {mix_count}):\n"
            f"  unique K combinations={len(unique_k_combos)}\n"
            f"  unique chips_inspected_per_cluster combinations={len(unique_chips_inspected_per_cluster_combos)}\n"
            f"  total scenario component rows={total_row_count}\n"
            f"  total scenarios={scenario_counter}"
        ),
        kind="STEP",
    )

    if total_row_count == 0:
        return None, scenario_counter, 0
    mix_columns = {
        column_name: np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        for column_name, chunks in column_chunks.items()
    }
    return mix_columns, scenario_counter, total_row_count


# Remove any existing parquet fragments for a mix before writing replacement records.
def _clear_existing_mix_files(mix_parquet_path: Path) -> None:
    mix_dir = mix_parquet_path.parent
    fragment_stem = mix_parquet_path.stem
    for existing_path in mix_dir.rglob(f"{fragment_stem}*.parquet"):
        if existing_path.is_file():
            existing_path.unlink()


# Delete a workflow artifact directory if it exists.
def _clear_directory(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


# Clear cached data and output directories so each workflow run starts from fresh artifacts.
def _reset_workflow_artifacts() -> None:
    for path in (
        Path(DATA_SAVED_DIR),
        Path(SCENARIOS_PARQUET_PATH),
        Path(ALL_OUTPUT_DIR),
        Path(PERFECT_INFORMATION_OUTPUT_DIR),
    ):
        _clear_directory(path)


# Create an empty Arrow table with the scenario-component schema.
def _empty_arrow_table(pa, schema):
    return pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)

# Ensure the scenario output location is a parquet dataset directory, not a single file.
def _ensure_scenarios_dataset_path(parquet_path: Path) -> None:
    if parquet_path.exists() and not parquet_path.is_dir():
        raise ValueError(
            f"{parquet_path} is a file, but chunked final scenario output now uses a parquet dataset directory. "
            "Move or remove the file, then rerun."
        )
    parquet_path.mkdir(parents=True, exist_ok=True)


# Build the shard path and parquet filename for a mix-specific scenario fragment.
def _mix_parquet_path(parquet_path: Path, mix_id: str) -> Path:
    safe_mix_id = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in mix_id)
    shard = _mix_id_shard(safe_mix_id)
    mix_dir = parquet_path / shard
    mix_dir.mkdir(parents=True, exist_ok=True)
    return mix_dir / f"{safe_mix_id}.parquet"


# Choose a simple shard directory from the first mix component to avoid too many files in one folder.
def _mix_id_shard(safe_mix_id: str) -> str:
    if safe_mix_id.startswith("ClusterMix_"):
        safe_mix_id = safe_mix_id.removeprefix("ClusterMix_")
    first_component = safe_mix_id.split("_", 1)[0]
    return first_component or "unknown"


# Convert per-column numpy arrays into an Arrow table using the provided schema.
def _columns_to_arrow_table(columns: dict[str, np.ndarray], pa, schema):
    return pa.Table.from_arrays(
        [pa.array(columns[field.name], type=field.type) for field in schema],
        schema=schema,
    )


# Import pyarrow lazily so callers using only in-memory paths do not need it at import time.
def _import_pyarrow_parquet():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Writing final scenarios to parquet requires pyarrow. "
            "Install pyarrow or call build_scenarios(..., output_parquet_path=None) "
            "to use the in-memory DataFrame path."
        ) from exc
    return pq, pa


# Stream a parquet file in pandas DataFrame batches for memory-bounded downstream summaries.
def _iter_parquet_batches(
    parquet_path: Path,
    columns: Iterable[str],
    batch_size: int = 65_536,
) -> Iterable[pd.DataFrame]:
    pq, _pa = _import_pyarrow_parquet()
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(columns=list(columns), batch_size=batch_size):
        yield batch.to_pandas()


# Print a compact row, column, and preview summary for a DataFrame.
def _overview_dataframe(name: str, df: pd.DataFrame, max_rows: int = 5) -> None:
    _workflow_log(name, f"rows={len(df):,}; columns={len(df.columns):,}", kind="INFO")
    _workflow_log(name, f"Column sample: {_format_column_preview(df.columns, max_rows)}", kind="INFO")
    if df.empty:
        _workflow_log(name, "DataFrame is empty", kind="INFO")


# Print row, column, and preview details for a single parquet file.
def _overview_parquet_file(path: Path, name: str, max_rows: int = 5) -> None:
    _workflow_log(name, f"Inspecting parquet file at {path}", kind="INFO")
    pq, _pa = _import_pyarrow_parquet()
    parquet_file = pq.ParquetFile(path)
    _workflow_log(
        name,
        f"rows={parquet_file.metadata.num_rows:,}; columns={len(parquet_file.schema.names):,}; column sample: {_format_column_preview(parquet_file.schema.names, max_rows)}",
        kind="INFO",
    )


# Print aggregate row, file, column, and preview details for a parquet dataset directory.
def _overview_parquet_dataset(path: Path, name: str, max_rows: int = 5) -> None:
    _workflow_log(name, f"Inspecting parquet dataset at {path}", kind="INFO")
    parquet_paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not parquet_paths:
        _workflow_log(name, "No parquet files", kind="INFO")
        return

    pq, _pa = _import_pyarrow_parquet()
    row_count = 0
    column_count = 0
    column_names: list[str] = []
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        row_count += parquet_file.metadata.num_rows
        column_count = max(column_count, len(parquet_file.schema.names))
        if not column_names:
            column_names = list(parquet_file.schema.names)

    _workflow_log(name, f"files={len(parquet_paths):,}; rows={row_count:,}; columns={column_count:,}", kind="INFO")
    _workflow_log(name, f"Column sample: {_format_column_preview(column_names, max_rows)}", kind="INFO")


# Print row, column, and preview details for a CSV output file.
def _overview_csv_file(path: Path, name: str, max_rows: int = 5) -> None:
    _workflow_log(name, f"Inspecting CSV file at {path}", kind="INFO")
    preview_df = pd.read_csv(path, nrows=0)
    with path.open(encoding="utf-8") as csv_file:
        row_count = sum(1 for _line in csv_file)
    row_count = max(0, row_count - 1)
    _workflow_log(name, f"rows={row_count:,}; columns={len(preview_df.columns):,}", kind="INFO")
    _workflow_log(name, f"Column sample: {_format_column_preview(preview_df.columns, max_rows)}", kind="INFO")

# Define the Arrow schema used for scenario-component parquet fragments.
def _final_scenario_component_arrow_schema(pa):
    return pa.schema(
        [
            (COLUMN_NAMES['mix_id'], pa.string()),
            (COLUMN_NAMES['scenario_id'], pa.string()),
            (COLUMN_NAMES['mix_description'], pa.string()),
            (COLUMN_NAMES['total_clusters_in_mix'], pa.int64()),
            (COLUMN_NAMES['total_tests'], pa.int64()),
            (COLUMN_NAMES['total_component_chips'], pa.int64()),
            (COLUMN_NAMES['scenario_component_count'], pa.int64()),
            (COLUMN_NAMES['k_combo'], pa.string()),
            (COLUMN_NAMES['chips_inspected_per_cluster_combo'], pa.string()),
            (COLUMN_NAMES['share_of_clusters_with_smuggling'], pa.float64()),
            (COLUMN_NAMES['cluster_size'], pa.int64()),
            (COLUMN_NAMES['bad_records'], pa.int64()),
            (COLUMN_NAMES['total_bad_records'], pa.int64()),
            (COLUMN_NAMES['number_of_clusters'], pa.int64()),
            (COLUMN_NAMES['number_of_clusters_with_smuggling'], pa.int64()),
            (COLUMN_NAMES['chips_inspected_per_cluster'], pa.int64()),
            (COLUMN_NAMES['physical_inspection_chip_level_miss_prob'], pa.float64()),
            (COLUMN_NAMES['physical_inspection_p_detect'], pa.float64()),
            (COLUMN_NAMES['physical_inspection_diverted_chips_identified'], pa.float64()),
            (COLUMN_NAMES['plv_chip_level_miss_prob'], pa.float64()),
            (COLUMN_NAMES['plv_p_detect'], pa.float64()),
            (COLUMN_NAMES['plv_diverted_chips_identified'], pa.float64()),
            (COLUMN_NAMES['physical_inspection_total_diverted_chips_identified'], pa.float64()),
            (COLUMN_NAMES['plv_total_diverted_chips_identified'], pa.float64()),
        ]
    )
