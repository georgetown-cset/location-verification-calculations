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
    PHYSICAL_INSPECTION_M_VALS,
    PLV_M_VALS,
    ALL_OUTPUT_DIR,
    PERFECT_INFORMATION_OUTPUT_DIR,
    SCENARIO_COLUMNS,
    SCENARIOS_PARQUET_PATH,
    SHARE_OF_CLUSTERS_WITH_SMUGGLING,
    MIN_SCENARIO_DIVERTED_CHIPS,
    TARGET_CHIPS,
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
        print(f"build_detection_lookup_table: writing table to {parquet_path}")
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
    print(f"build_mix_data: generating mixes for {len(cluster_sizes)} cluster sizes and target {target_chips} chips")
    if min_clusters_by_size:
        print(f"build_mix_data: applying minimum cluster constraints {min_clusters_by_size}")
    valid_mixes: list[list[dict[str, int]]] = []
    seen_mix_ids: set[str] = set()
    for _mix_index, proportions in _iter_valid_mix_proportions(steps, len(cluster_sizes)):
        active_components = [(cluster_size, proportion) for cluster_size, proportion in zip(cluster_sizes, proportions) if proportion > 0]
        current_mix: list[dict[str, int]] = []

        # Convert each active proportion into an integer cluster count for that cluster size.
        for cluster_size, proportion in active_components:
            count = _integer_cluster_count(target_chips, proportion, cluster_size)
            if count is None:
                current_mix = []
                break
            current_mix.append(
                {
                    COLUMN_NAMES['cluster_size']: int(cluster_size),
                    COLUMN_NAMES['number_of_clusters']: count,
                }
            )

        if not current_mix:
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
        print(
            f"build_mix_data: accepted {mix_id} with {len(current_mix)} components; "
            f"total valid mixes={len(valid_mixes)}"
        )

    print(f"build_mix_data: completed with {len(valid_mixes)} valid mixes")
    return valid_mixes


# Convert a mix proportion to a cluster count, allowing harmless floating-point representation error.
def _integer_cluster_count(
    target_chips: int,
    proportion: float,
    cluster_size: int,
) -> int | None:
    count = (target_chips * proportion) / cluster_size
    rounded_count = round(count)
    if not math.isclose(count, rounded_count, rel_tol=1e-12, abs_tol=1e-9):
        return None
    return int(rounded_count)


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
def _number_of_clusters_with_smuggling(number_of_clusters: int) -> int:
    if number_of_clusters <= 1:
        return int(number_of_clusters)
    return max(1, int(math.floor(number_of_clusters * SHARE_OF_CLUSTERS_WITH_SMUGGLING + 0.5)))


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


# Generate scenario-component rows for every valid mix, diversion amount, inspected-chip count, and miss-probability pair.
def build_scenarios(
    cluster_sizes: Iterable[int],
    detection_lookup_table: pd.DataFrame,
    k_vals: Iterable[int],
    min_scenario_diverted_chips: int = MIN_SCENARIO_DIVERTED_CHIPS,
    min_clusters_by_size: Optional[dict[int, int]] = None,
    physical_m_vals: Iterable[float] = PHYSICAL_INSPECTION_M_VALS,
    plv_m_vals: Iterable[float] = PLV_M_VALS,
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = None,
    output_parquet_path: Optional[str | Path] = SCENARIOS_PARQUET_PATH,
    return_dataframe: Optional[bool] = None,
) -> pd.DataFrame:
    cluster_sizes = list(cluster_sizes)
    k_vals = list(k_vals)
    min_scenario_diverted_chips = int(min_scenario_diverted_chips)
    if min_scenario_diverted_chips < 0:
        raise ValueError("build_scenarios: min_scenario_diverted_chips must be non-negative")
    physical_m_vals = list(physical_m_vals)
    plv_m_vals = list(plv_m_vals)
    print(f"build_scenarios: starting with {len(cluster_sizes)} cluster sizes and {len(k_vals)} K values")
    mix_data = build_mix_data(
        cluster_sizes=cluster_sizes,
        target_chips=target_chips,
        steps=steps,
        min_clusters_by_size=min_clusters_by_size,
    )
    print(f"build_scenarios: received {len(mix_data)} mixes from build_mix_data")
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
        for mix_components in mix_data:
            # Each mix is generated and written independently to keep peak memory use bounded.
            mix_id = mix_components[0][COLUMN_NAMES['mix_id']]
            mix_scenario_component_records, mix_scenario_count, mix_scenario_component_count = _build_mix_records(
                mix_components=mix_components,
                detection_grouped=detection_grouped,
                k_options_by_cluster_size=k_options_by_cluster_size,
                min_scenario_diverted_chips=min_scenario_diverted_chips,
                physical_m_vals=physical_m_vals,
                plv_m_vals=plv_m_vals,
                target_chips=target_chips,
            )
            mix_component_fragments = _write_scenario_dataset_to_parquet(
                dataset_path=component_dataset_path,
                mix_id=mix_id,
                records=mix_scenario_component_records,
            )
            mix_summary_count = mix_scenario_count
            mix_fragment_count = len(mix_component_fragments)
            print(
                f"build_scenarios: wrote {mix_id}; "
                f"scenario rows added={mix_scenario_count}, scenario-component rows added={mix_scenario_component_count}, "
                f"fragment files written={mix_fragment_count}"
            )
            total_scenarios += mix_summary_count
            total_scenario_component_rows += mix_scenario_component_count
            print(
                f"build_scenarios: finished {mix_id}; total scenario rows={total_scenarios}, "
                f"total scenario-component rows={total_scenario_component_rows}"
            )

        print(
            f"build_scenarios: completed with {total_scenarios} new scenario rows "
            f"and {total_scenario_component_rows} new scenario-component rows"
        )

        if return_dataframe:
            # Read the dataset back only when callers need an in-memory frame for downstream processing.
            return pd.read_parquet(component_dataset_path)
        return pd.DataFrame(columns=SCENARIO_COLUMNS)

    # In-memory mode is primarily useful for tests or small parameter grids.
    scenario_component_records: list[tuple[object, ...]] = []
    for mix_components in mix_data:
        mix_scenario_component_records, mix_scenario_count, mix_scenario_component_count = _build_mix_records(
            mix_components=mix_components,
            detection_grouped=detection_grouped,
            k_options_by_cluster_size=k_options_by_cluster_size,
            min_scenario_diverted_chips=min_scenario_diverted_chips,
            physical_m_vals=physical_m_vals,
            plv_m_vals=plv_m_vals,
            target_chips=target_chips,
        )
        if return_dataframe:
            scenario_component_records.extend(mix_scenario_component_records)
        total_scenarios += mix_scenario_count
        total_scenario_component_rows += mix_scenario_component_count
        print(
            f"build_scenarios: finished {mix_components[0][COLUMN_NAMES['mix_id']]}; "
            f"scenario-component rows added={mix_scenario_component_count}, "
            f"total new scenario rows={total_scenarios}, total new scenario-component rows={total_scenario_component_rows}"
        )

    print(
        f"build_scenarios: completed with {total_scenarios} new scenario rows "
        f"and {total_scenario_component_rows} new scenario-component rows"
    )

    if return_dataframe:
        return pd.DataFrame.from_records(scenario_component_records, columns=SCENARIO_COLUMNS)
    return pd.DataFrame(columns=SCENARIO_COLUMNS)


# Write one mix's scenario-component records as a parquet fragment in the scenario dataset.
def _write_scenario_dataset_to_parquet(
    *,
    dataset_path: Path,
    mix_id: str,
    records: list[tuple[object, ...]],
) -> list[Path]:
    pq, pa = _import_pyarrow_parquet()
    _ensure_scenarios_dataset_path(dataset_path)
    mix_parquet_path = _mix_parquet_path(dataset_path, mix_id)
    _clear_existing_mix_files(mix_parquet_path)
    if not records:
        table = _empty_arrow_table(pa, _final_scenario_component_arrow_schema(pa))
        pq.write_table(table, mix_parquet_path, compression=PARQUET_COMPRESSION)
        return [mix_parquet_path]

    schema = _final_scenario_component_arrow_schema(pa)
    table = _records_to_arrow_table(records, pa, schema)
    pq.write_table(table, mix_parquet_path, compression=PARQUET_COMPRESSION)
    return [mix_parquet_path]


# Expand one cluster mix into all scenario-component records and count the scenario rows produced.
def _build_mix_records(
    *,
    mix_components: list[dict[str, int]],
    detection_grouped: dict[tuple[int, int], dict[int, dict[float, tuple[float, float, float, float]]]],
    k_options_by_cluster_size: dict[int, list[int]],
    min_scenario_diverted_chips: int,
    physical_m_vals: Iterable[float],
    plv_m_vals: Iterable[float],
    target_chips: int,
) -> tuple[list[tuple[object, ...]], int, int]:
    scenario_component_records: list[tuple[object, ...]] = []
    mix_id = mix_components[0][COLUMN_NAMES['mix_id']]
    mix_description = build_mix_description(mix_components)
    total_clusters_in_mix = sum(component[COLUMN_NAMES['number_of_clusters']] for component in mix_components)
    scenario_component_count = len(mix_components)
    physical_m_vals = list(physical_m_vals)
    plv_m_vals = list(plv_m_vals)
    print(
        f"build_scenarios: processing {mix_id} ({mix_description}) "
        f"with {len(mix_components)} components and {total_clusters_in_mix} total clusters"
    )

    # Component metadata packages cluster size, cluster count, smuggling count, weight, and valid K options.
    component_data = [
        (
            component[COLUMN_NAMES['cluster_size']],
            component[COLUMN_NAMES['number_of_clusters']],
            _number_of_clusters_with_smuggling(component[COLUMN_NAMES['number_of_clusters']]),
            (component[COLUMN_NAMES['cluster_size']] * component[COLUMN_NAMES['number_of_clusters']] / target_chips) * 100,
            k_options_by_cluster_size[component[COLUMN_NAMES['cluster_size']]],
        )
        for component in mix_components
    ]
    k_options_per_component = [
        [
            k_val
            for k_val in component[4]
            if k_val == 0 or k_val >= MIN_DIVERTED_CHIPS or k_val == component[0]
        ]
        for component in component_data
    ]
    scenario_counter = 0

    # A scenario chooses one diverted-chip count for each component in the mix.
    for combo_count, k_combo in enumerate(itertools.product(*k_options_per_component), start=1):
        scenario_diverted_chips = sum(
            int(k_val) * int(num_clusters_with_smuggling)
            for (_cluster_size_comp, _num_clusters_comp, num_clusters_with_smuggling, _weight_pct, _), k_val
            in zip(component_data, k_combo)
        )
        if scenario_diverted_chips < min_scenario_diverted_chips:
            continue

        chips_inspected_per_cluster_options_per_component: list[tuple[int, ...]] = []
        for (cluster_size_comp, _num_clusters_comp, _num_clusters_with_smuggling, _weight_pct, _), k_val in zip(component_data, k_combo):
            # Inspected-chip options come from the detection table and are keyed by the selected cluster size and K.
            chips_inspected_per_cluster_options = detection_grouped.get((cluster_size_comp, int(k_val)), {})
            if not chips_inspected_per_cluster_options:
                break
            chips_inspected_per_cluster_options_per_component.append(tuple(sorted(chips_inspected_per_cluster_options)))
        else:
            # Once every component has valid K and inspected-chip options, cross product them into concrete scenarios.
            for chips_inspected_per_cluster_combo_count, chips_inspected_per_cluster_combo in enumerate(itertools.product(*chips_inspected_per_cluster_options_per_component), start=1):
                for physical_m_val in physical_m_vals:
                    for plv_m_val in plv_m_vals:
                        scenario_id = (
                            f"{mix_id}_K{'-'.join(map(str, k_combo))}_i{'-'.join(map(str, chips_inspected_per_cluster_combo))}"
                            f"_pm{physical_m_val}_plvm{plv_m_val}"
                        )
                        flat_rows: list[tuple[object, ...]] = []

                        # Store one component row per scenario component; scenario-level totals are computed later.
                        for (cluster_size_comp, num_clusters_comp, num_clusters_with_smuggling, _weight_pct, _), k_val, chips_inspected_per_cluster in zip(component_data, k_combo, chips_inspected_per_cluster_combo):
                            detection_info_by_m = detection_grouped.get((cluster_size_comp, k_val), {}).get(int(chips_inspected_per_cluster), {})
                            physical_info = detection_info_by_m.get(float(physical_m_val))
                            if physical_info is None:
                                raise ValueError(
                                    "Missing physical-inspection detection lookup row for "
                                    f"cluster_size={cluster_size_comp}, k={k_val}, "
                                    f"chips_inspected_per_cluster={chips_inspected_per_cluster}, m={physical_m_val}"
                                )
                            plv_info = detection_info_by_m.get(float(plv_m_val))
                            if plv_info is None:
                                raise ValueError(
                                    "Missing PLV detection lookup row for "
                                    f"cluster_size={cluster_size_comp}, k={k_val}, "
                                    f"chips_inspected_per_cluster={chips_inspected_per_cluster}, m={plv_m_val}"
                                )
                            physical_p_detect, physical_identified_per_cluster, _, _ = physical_info
                            _, _, plv_p_detect, plv_identified_per_cluster = plv_info
                            flat_rows.append(
                                (
                                    mix_id,
                                    scenario_id,
                                    mix_description,
                                    total_clusters_in_mix,
                                    num_clusters_comp * chips_inspected_per_cluster,
                                    num_clusters_comp * cluster_size_comp,
                                    scenario_component_count,
                                    "-".join(map(str, k_combo)),
                                    "-".join(map(str, chips_inspected_per_cluster_combo)),
                                    cluster_size_comp,
                                    k_val,
                                    k_val * num_clusters_with_smuggling,
                                    num_clusters_comp,
                                    num_clusters_with_smuggling,
                                    chips_inspected_per_cluster,
                                    physical_m_val,
                                    physical_p_detect,
                                    physical_identified_per_cluster,
                                    plv_m_val,
                                    plv_p_detect,
                                    plv_identified_per_cluster,
                                    physical_identified_per_cluster * num_clusters_with_smuggling,
                                    plv_identified_per_cluster * num_clusters_with_smuggling,
                                )
                            )

                        scenario_counter += 1
                        scenario_component_records.extend(flat_rows)

                if chips_inspected_per_cluster_combo_count % 1000 == 0:
                    print(
                        f"build_scenarios: {mix_id} processed {combo_count} K combinations and "
                        f"{chips_inspected_per_cluster_combo_count} chips-inspected-per-cluster combinations; scenario rows so far for mix={scenario_counter}, "
                        f"scenario-component rows so far for mix={len(scenario_component_records)}"
                    )

            if combo_count % 1000 == 0:
                print(
                    f"build_scenarios: {mix_id} processed {combo_count} K combinations; "
                    f"scenario rows so far for mix={scenario_counter}, "
                    f"scenario-component rows so far for mix={len(scenario_component_records)}"
                )

    return scenario_component_records, scenario_counter, len(scenario_component_records)


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


# Convert tuple records into an Arrow table using the provided schema.
def _records_to_arrow_table(records: list[tuple[object, ...]], pa, schema):
    return pa.Table.from_arrays([pa.array(values) for values in zip(*records)], schema=schema)


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
    print(f"\n{name}")
    print(f"Rows: {len(df):,}; columns: {len(df.columns):,}")
    if df.empty:
        print("(empty)")
    else:
        print(df.head(max_rows))


# Print row, column, and preview details for a single parquet file.
def _overview_parquet_file(path: Path, name: str, max_rows: int = 5) -> None:
    print(f"\n{name}: {path}")
    pq, _pa = _import_pyarrow_parquet()
    parquet_file = pq.ParquetFile(path)
    print(f"Rows: {parquet_file.metadata.num_rows:,}; columns: {len(parquet_file.schema.names):,}")
    for batch in parquet_file.iter_batches(batch_size=max_rows):
        preview_df = batch.to_pandas()
        print(preview_df.head(max_rows) if not preview_df.empty else "(empty)")
        break


# Print aggregate row, file, column, and preview details for a parquet dataset directory.
def _overview_parquet_dataset(path: Path, name: str, max_rows: int = 5) -> None:
    print(f"\n{name}: {path}")
    parquet_paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not parquet_paths:
        print("No parquet files")
        return

    pq, _pa = _import_pyarrow_parquet()
    row_count = 0
    column_count = 0
    preview_df = pd.DataFrame()
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        row_count += parquet_file.metadata.num_rows
        column_count = max(column_count, len(parquet_file.schema.names))
        if preview_df.empty:
            for batch in parquet_file.iter_batches(batch_size=max_rows):
                preview_df = batch.to_pandas()
                break

    print(f"Files: {len(parquet_paths):,}; rows: {row_count:,}; columns: {column_count:,}")
    print(preview_df.head(max_rows) if not preview_df.empty else "(empty)")


# Print row, column, and preview details for a CSV output file.
def _overview_csv_file(path: Path, name: str, max_rows: int = 5) -> None:
    print(f"\n{name}: {path}")
    preview_df = pd.read_csv(path, nrows=max_rows)
    with path.open(encoding="utf-8") as csv_file:
        row_count = sum(1 for _line in csv_file)
    row_count = max(0, row_count - 1)
    print(f"Rows: {row_count:,}; columns: {len(preview_df.columns):,}")
    print(preview_df if not preview_df.empty else "(empty)")

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
