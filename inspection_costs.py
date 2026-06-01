from __future__ import annotations

import itertools
import math
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


TARGET_CHIPS = 3_000_000 # Total number of chips in the scenarios, which is used to determine how many clusters of each size are needed in the mixes
SHARE_OF_CLUSTERS_WITH_SMUGGLING = 0.25 # Share of clusters in a scenario component that are assumed to contain smuggling. This imposes an upper bound on the number of clusters with smuggling in a scenario component.
NUMBER_OF_PHYSICAL_INSPECTIONS_PER_CLUSTER_PER_YEAR = 2
DOLLARS_PER_CHIP_DETECTED = (1000, 60000) # Estimated range of the value of detecting a diverted chip, which is used to convert net benefit estimates from chip counts to dollars

PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP = (8.3, 48.8)
PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION = (2545, 5900)
PLV_RENTING_COST_PER_TOTAL_CHIP = (2_251_688 / TARGET_CHIPS, 72_427_200 / TARGET_CHIPS) # 12 - 500 landmark servers
PLV_OWNING_COST_PER_TOTAL_CHIP = (3_028_862 / TARGET_CHIPS, 28_715_814 / TARGET_CHIPS) # 12 - 500 landmark servers
PLV_DISCOUNT_RATE = 0.5 # Discount rate to apply to PLV benefits to account for it not being a perfect substitute for physical inspections
MIN_SHARE_DIVERTED = 0.1 # Minimum allowed share diverted (K / N) for each scenario component in mix scenarios.

MIX_STEP_SIZE = 0.2
CLUSTER_SIZES = [10, 100, 1000, 10000, 100000] # Sizes of clusters to consider in mixes
K_VALS = [0, 1, 10, 100, 1000, 10000, 100000] # Number of diverted chips in a cluster with smuggling (i.e., the "bad records" in a cluster)
N_VALS = [0, 1, 10, 100, 1000] # Number of tests conducted on a cluster
PHYSICAL_INSPECTION_M_VALS = [0.05] # Probability that a diverted chip is not detected by a physical inspection test (i.e., the "miss" probability)
PLV_M_VALS = [0.1] # Probability that a diverted chip is not detected by a PLV test (i.e., the "miss" probability)

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
    "net_benefit_relationship_description": "Net Benefit Relationship Description",
    "benefit_per_dollar_relationship_description": "Benefit Per Dollar Relationship Description",
    "net_benefit_scenario_count": "Net Benefit Scenario Count",
    "benefit_per_dollar_scenario_count": "Benefit Per Dollar Scenario Count",
}
SCENARIO_COLUMNS = [
    COLUMN_NAMES['mix_id'],
    COLUMN_NAMES['scenario_id'],
    COLUMN_NAMES['mix_description'],
    COLUMN_NAMES['total_clusters_in_mix'],
    COLUMN_NAMES['total_tests'],
    COLUMN_NAMES['total_component_chips'],
    COLUMN_NAMES['scenario_component_count'],
    COLUMN_NAMES['k_combo'],
    COLUMN_NAMES['n_combo'],
    COLUMN_NAMES['cluster_size'],
    COLUMN_NAMES['bad_records'],
    COLUMN_NAMES['total_bad_records'],
    COLUMN_NAMES['number_of_clusters'],
    COLUMN_NAMES['number_of_clusters_with_smuggling'],
    COLUMN_NAMES['tests'],
    COLUMN_NAMES['physical_inspection_chip_level_miss_prob'],
    COLUMN_NAMES['physical_inspection_p_detect'],
    COLUMN_NAMES['physical_inspection_diverted_chips_identified'],
    COLUMN_NAMES['plv_chip_level_miss_prob'],
    COLUMN_NAMES['plv_p_detect'],
    COLUMN_NAMES['plv_diverted_chips_identified'],
    COLUMN_NAMES['physical_inspection_total_diverted_chips_identified'],
    COLUMN_NAMES['plv_total_diverted_chips_identified'],
]
LONG_SCENARIO_VALUE_VARS = (
    "Physical - Min Net Benefit",
    "Physical - Max Net Benefit",
    "PLV Renting - Min Net Benefit",
    "PLV Renting - Max Net Benefit",
    "PLV Owning - Min Net Benefit",
    "PLV Owning - Max Net Benefit",
    "Physical - Min Benefit Per Dollar",
    "Physical - Max Benefit Per Dollar",
    "PLV Renting - Min Benefit Per Dollar",
    "PLV Renting - Max Benefit Per Dollar",
    "PLV Owning - Min Benefit Per Dollar",
    "PLV Owning - Max Benefit Per Dollar",
)
LONG_SCENARIO_BENEFIT_SCENARIOS = {
    "Physical - Min Net Benefit": "Physical (Conservative)",
    "Physical - Max Net Benefit": "Physical (Optimistic)",
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
NET_BENEFIT_RELATIONSHIP_COLUMN = "Physical vs PLV Net Benefit Relationship"
NET_BENEFIT_RELATIONSHIP_LABELS = {
    "a": "Physical max net benefit is less than PLV min net benefit",
    "b": "Physical max net benefit is within PLV net benefit range and Physical min net benefit is below PLV min net benefit",
    "c": "Physical min and max net benefits are both within PLV net benefit range",
    "d": "Physical min net benefit is within PLV net benefit range and Physical max net benefit is above PLV max net benefit",
    "e": "Physical min net benefit is greater than PLV max net benefit",
    "f": "Physical net benefit range spans both sides of PLV net benefit range",
}
BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN = "Physical vs PLV Benefit Per Dollar Relationship"
BENEFIT_PER_DOLLAR_RELATIONSHIP_LABELS = {
    "a": "Physical max benefit per dollar is less than PLV min benefit per dollar",
    "b": "Physical max benefit per dollar is within PLV range and Physical min benefit per dollar is below PLV min benefit per dollar",
    "c": "Physical min and max benefit per dollar are both within PLV range",
    "d": "Physical min benefit per dollar is within PLV range and Physical max benefit per dollar is above PLV max benefit per dollar",
    "e": "Physical min benefit per dollar is greater than PLV max benefit per dollar",
    "f": "Physical benefit per dollar range spans both sides of PLV range",
}
RELATIONSHIP_CODE_ORDER = list(NET_BENEFIT_RELATIONSHIP_LABELS.keys())


def _plv_relationship_labels(plv_type: str, metric_family: str) -> dict[str, str]:
    if metric_family == "net_benefit":
        return {
            "a": f"Physical max net benefit is less than {plv_type} min net benefit",
            "b": f"Physical max net benefit is within {plv_type} net benefit range and Physical min net benefit is below {plv_type} min net benefit",
            "c": f"Physical min and max net benefits are both within {plv_type} net benefit range",
            "d": f"Physical min net benefit is within {plv_type} net benefit range and Physical max net benefit is above {plv_type} max net benefit",
            "e": f"Physical min net benefit is greater than {plv_type} max net benefit",
            "f": f"Physical net benefit range spans both sides of {plv_type} net benefit range",
        }
    if metric_family == "benefit_per_dollar":
        return {
            "a": f"Physical max benefit per dollar is less than {plv_type} min benefit per dollar",
            "b": f"Physical max benefit per dollar is within {plv_type} range and Physical min benefit per dollar is below {plv_type} min benefit per dollar",
            "c": f"Physical min and max benefit per dollar are both within {plv_type} range",
            "d": f"Physical min benefit per dollar is within {plv_type} range and Physical max benefit per dollar is above {plv_type} max benefit per dollar",
            "e": f"Physical min benefit per dollar is greater than {plv_type} max benefit per dollar",
            "f": f"Physical benefit per dollar range spans both sides of {plv_type} range",
        }
    raise ValueError(f"Unknown metric family: {metric_family}")


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


def _hypergeom_pmf(x: int, N: int, K: int, n: int) -> float:
    if x < 0 or x > K or x > n:
        return 0.0
    if n - x > N - K:
        return 0.0
    if N < 0 or K < 0 or n < 0 or K > N or n > N:
        return 0.0
    return math.comb(K, x) * math.comb(N - K, n - x) / math.comb(N, n)


def p_detect_cluster_diversion(N: int, n: int, K: int, m: float) -> dict[str, float]:
    if K == 0 or n == 0:
        return {"p_success": 0.0, "p_failure": 1.0}

    x_values = range(max(0, n - (N - K)), min(K, n) + 1)
    p_failure = sum(_hypergeom_pmf(x, N, K, n) * (m**x) for x in x_values)
    return {"p_success": 1 - p_failure, "p_failure": p_failure}


def expected_value_detected_diversion(N: int, n: int, K: int, m: float) -> dict[str, float]:
    p_detect = p_detect_cluster_diversion(N=N, n=n, K=K, m=m)["p_success"]
    return {
        "diverted_chips_identified": p_detect * K,
        "p_detect": p_detect,
    }


def build_detection_lookup_table(
    cluster_sizes: Iterable[int],
    K_vals: Iterable[int],
    n_vals: Iterable[int],
    m_vals: Iterable[float] = ALL_M_VALS,
    parquet_path: Optional[str | Path] = DETECTION_LOOKUP_TABLE_PARQUET_PATH,
) -> pd.DataFrame:
    if parquet_path is not None:
        parquet_path = Path(parquet_path)
        if parquet_path.exists():
            print(f"build_detection_lookup_table: reading cached table from {parquet_path}")
            return pd.read_parquet(parquet_path)

    rows = []
    for N in cluster_sizes:
        for n in n_vals:
            if n > N:
                continue
            for K in K_vals:
                if K > N:
                    continue
                for m in m_vals:
                    physical_result = expected_value_detected_diversion(N, n, K, m)
                    plv_result = expected_value_detected_diversion(N, N, K, m)
                    rows.append(
                        {
                            COLUMN_NAMES['cluster_size']: int(N),
                            COLUMN_NAMES['tests']: int(n),
                            COLUMN_NAMES['bad_records']: int(K),
                            COLUMN_NAMES['share_diverted']: K / N,
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


def build_mix_data(
    cluster_sizes: Iterable[int],
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = None,
) -> list[list[dict[str, int]]]:
    if steps is None:
        steps = MIX_STEPS

    cluster_sizes = list(cluster_sizes)
    steps = list(steps)
    print(f"build_mix_data: generating mixes for {len(cluster_sizes)} cluster sizes and target {target_chips} chips")
    valid_mixes: list[list[dict[str, int]]] = []
    seen_mix_ids: set[str] = set()
    for _mix_index, proportions in _iter_valid_mix_proportions(steps, len(cluster_sizes)):
        active_components = [(n, p) for n, p in zip(cluster_sizes, proportions) if p > 0]
        current_mix: list[dict[str, int]] = []
        for cluster_size, proportion in active_components:
            count = (target_chips * proportion) / cluster_size
            if count != int(count):
                current_mix = []
                break
            current_mix.append(
                {
                    COLUMN_NAMES['cluster_size']: int(cluster_size),
                    COLUMN_NAMES['number_of_clusters']: int(count),
                }
            )

        if current_mix:
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


def _iter_valid_mix_proportions_brute_force(steps: list[float], dimensions: int) -> Iterable[tuple[int, tuple[float, ...]]]:
    for mix_index, proportions in enumerate(itertools.product(steps, repeat=dimensions)):
        if np.isclose(sum(proportions), 1.0):
            yield mix_index, proportions


def _product_index(indices: list[int], base: int) -> int:
    mix_index = 0
    for index in indices:
        mix_index = mix_index * base + index
    return mix_index


def build_mix_description(mix_components: list[dict[str, int]]) -> str:
    return " + ".join(f"{component[COLUMN_NAMES['number_of_clusters']]}x(N={component[COLUMN_NAMES['cluster_size']]})" for component in mix_components)


def build_mix_id(mix_components: list[dict[str, int]]) -> str:
    mix_components = sorted(mix_components, key=lambda component: component[COLUMN_NAMES['cluster_size']])
    characteristics = "_".join(
        f"N{component[COLUMN_NAMES['cluster_size']]}-C{component[COLUMN_NAMES['number_of_clusters']]}"
        for component in mix_components
    )
    return f"ClusterMix_{characteristics}"


def _number_of_clusters_with_smuggling(number_of_clusters: int) -> int:
    if number_of_clusters <= 1:
        return int(number_of_clusters)
    return max(1, int(math.floor(number_of_clusters * SHARE_OF_CLUSTERS_WITH_SMUGGLING + 0.5)))


def _group_detection_lookup(detection_lookup_table: pd.DataFrame) -> dict[tuple[int, int], dict[int, dict[float, tuple[float, float, float, float]]]]:
    grouped: dict[tuple[int, int], dict[int, dict[float, tuple[float, float, float, float]]]] = {}
    columns = [
        COLUMN_NAMES['cluster_size'],
        COLUMN_NAMES['bad_records'],
        COLUMN_NAMES['tests'],
        COLUMN_NAMES['chip_level_miss_prob'],
        COLUMN_NAMES['physical_inspection_p_detect'],
        COLUMN_NAMES['physical_inspection_diverted_chips_identified'],
        COLUMN_NAMES['plv_p_detect'],
        COLUMN_NAMES['plv_diverted_chips_identified'],
    ]
    for row in detection_lookup_table.loc[:, columns].itertuples(index=False, name=None):
        N, K, n, m, physical_p_detect, physical_identified, plv_p_detect, plv_identified = row
        grouped.setdefault((int(N), int(K)), {}).setdefault(int(n), {})[float(m)] = (
            float(physical_p_detect),
            float(physical_identified),
            float(plv_p_detect),
            float(plv_identified),
        )
    return grouped


def build_scenarios(
    cluster_sizes: Iterable[int],
    detection_lookup_table: pd.DataFrame,
    k_vals: Iterable[int],
    physical_m_vals: Iterable[float] = PHYSICAL_INSPECTION_M_VALS,
    plv_m_vals: Iterable[float] = PLV_M_VALS,
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = None,
    output_parquet_path: Optional[str | Path] = SCENARIOS_PARQUET_PATH,
    return_dataframe: Optional[bool] = None,
) -> pd.DataFrame:
    cluster_sizes = list(cluster_sizes)
    k_vals = list(k_vals)
    physical_m_vals = list(physical_m_vals)
    plv_m_vals = list(plv_m_vals)
    print(f"build_scenarios: starting with {len(cluster_sizes)} cluster sizes and {len(k_vals)} K values")
    mix_data = build_mix_data(cluster_sizes=cluster_sizes, target_chips=target_chips, steps=steps)
    print(f"build_scenarios: received {len(mix_data)} mixes from build_mix_data")
    detection_grouped = _group_detection_lookup(detection_lookup_table)
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
        pq, _pa = _import_pyarrow_parquet()
        for mix_components in mix_data:
            mix_id = mix_components[0][COLUMN_NAMES['mix_id']]
            mix_parquet_path = _mix_parquet_path(component_dataset_path, mix_id)
            if mix_parquet_path.exists():
                mix_scenario_component_count = pq.ParquetFile(mix_parquet_path).metadata.num_rows
                mix_summary_count = _count_scenarios_in_component_file(mix_parquet_path, pq)
                mix_fragment_count = 0
                print(
                    f"build_scenarios: skipping {mix_id} (already written); "
                    f"scenario rows={mix_summary_count}, scenario-component rows={mix_scenario_component_count}"
                )
            else:
                mix_scenario_component_records, mix_scenario_count, mix_scenario_component_count = _build_mix_records(
                    mix_components=mix_components,
                    detection_grouped=detection_grouped,
                    k_options_by_cluster_size=k_options_by_cluster_size,
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
            return pd.read_parquet(component_dataset_path)
        return pd.DataFrame(columns=SCENARIO_COLUMNS)

    scenario_component_records: list[tuple[object, ...]] = []
    for mix_components in mix_data:
        mix_scenario_component_records, mix_scenario_count, mix_scenario_component_count = _build_mix_records(
            mix_components=mix_components,
            detection_grouped=detection_grouped,
            k_options_by_cluster_size=k_options_by_cluster_size,
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


def _count_scenarios_in_component_file(mix_parquet_path: Path, pq) -> int:
    parquet_file = pq.ParquetFile(mix_parquet_path)
    scenario_ids: set[str] = set()
    for batch in parquet_file.iter_batches(columns=[COLUMN_NAMES['scenario_id']]):
        scenario_ids.update(str(scenario_id) for scenario_id in batch.column(0).to_pylist())
    return len(scenario_ids)


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


def _build_mix_records(
    *,
    mix_components: list[dict[str, int]],
    detection_grouped: dict[tuple[int, int], dict[int, dict[float, tuple[float, float, float, float]]]],
    k_options_by_cluster_size: dict[int, list[int]],
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
        [k_val for k_val in component[4] if (k_val / component[0]) >= MIN_SHARE_DIVERTED]
        for component in component_data
    ]
    scenario_counter = 0

    for combo_count, k_combo in enumerate(itertools.product(*k_options_per_component), start=1):
        n_options_per_component: list[tuple[int, ...]] = []
        for (N_comp, _num_clusters_comp, _num_clusters_with_smuggling, _weight_pct, _), k_val in zip(component_data, k_combo):
            n_options = detection_grouped.get((N_comp, int(k_val)), {})
            if not n_options:
                break
            n_options_per_component.append(tuple(sorted(n_options)))
        else:
            for n_combo_count, n_combo in enumerate(itertools.product(*n_options_per_component), start=1):
                for physical_m_val in physical_m_vals:
                    for plv_m_val in plv_m_vals:
                        scenario_counter += 1
                        scenario_id = (
                            f"{mix_id}_K{'-'.join(map(str, k_combo))}_n{'-'.join(map(str, n_combo))}"
                            f"_pm{physical_m_val}_plvm{plv_m_val}"
                        )
                        flat_rows: list[tuple[object, ...]] = []
                        for (N_comp, num_clusters_comp, num_clusters_with_smuggling, _weight_pct, _), k_val, n_val in zip(component_data, k_combo, n_combo):
                            detection_info_by_m = detection_grouped.get((N_comp, k_val), {}).get(int(n_val), {})
                            physical_info = detection_info_by_m.get(float(physical_m_val))
                            if physical_info is None:
                                continue
                            plv_info = detection_info_by_m.get(float(plv_m_val))
                            if plv_info is None:
                                continue
                            physical_p_detect, physical_identified_per_cluster, _, _ = physical_info
                            _, _, plv_p_detect, plv_identified_per_cluster = plv_info
                            flat_rows.append(
                                (
                                    mix_id,
                                    scenario_id,
                                    mix_description,
                                    total_clusters_in_mix,
                                    num_clusters_comp * n_val,
                                    num_clusters_comp * N_comp,
                                    scenario_component_count,
                                    "-".join(map(str, k_combo)),
                                    "-".join(map(str, n_combo)),
                                    N_comp,
                                    k_val,
                                    k_val * num_clusters_with_smuggling,
                                    num_clusters_comp,
                                    num_clusters_with_smuggling,
                                    n_val,
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

                        scenario_component_records.extend(flat_rows)

                if n_combo_count % 1000 == 0:
                    print(
                        f"build_scenarios: {mix_id} processed {combo_count} K combinations and "
                        f"{n_combo_count} n combinations; scenario rows so far for mix={scenario_counter}, "
                        f"scenario-component rows so far for mix={len(scenario_component_records)}"
                    )

            if combo_count % 1000 == 0:
                print(
                    f"build_scenarios: {mix_id} processed {combo_count} K combinations; "
                    f"scenario rows so far for mix={scenario_counter}, "
                    f"scenario-component rows so far for mix={len(scenario_component_records)}"
                )

    return scenario_component_records, scenario_counter, len(scenario_component_records)


def _clear_existing_mix_files(mix_parquet_path: Path) -> None:
    mix_dir = mix_parquet_path.parent
    fragment_stem = mix_parquet_path.stem
    for existing_path in mix_dir.rglob(f"{fragment_stem}*.parquet"):
        if existing_path.is_file():
            existing_path.unlink()


def _empty_arrow_table(pa, schema):
    return pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)

def _ensure_scenarios_dataset_path(parquet_path: Path) -> None:
    if parquet_path.exists() and not parquet_path.is_dir():
        raise ValueError(
            f"{parquet_path} is a file, but chunked final scenario output now uses a parquet dataset directory. "
            "Move or remove the file, then rerun."
        )
    parquet_path.mkdir(parents=True, exist_ok=True)


def _mix_parquet_path(parquet_path: Path, mix_id: str) -> Path:
    safe_mix_id = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in mix_id)
    shard = _mix_id_shard(safe_mix_id)
    mix_dir = parquet_path / shard
    mix_dir.mkdir(parents=True, exist_ok=True)
    return mix_dir / f"{safe_mix_id}.parquet"


def _mix_id_shard(safe_mix_id: str) -> str:
    if safe_mix_id.startswith("ClusterMix_"):
        safe_mix_id = safe_mix_id.removeprefix("ClusterMix_")
    first_component = safe_mix_id.split("_", 1)[0]
    return first_component or "unknown"


def _records_to_arrow_table(records: list[tuple[object, ...]], pa, schema):
    return pa.Table.from_arrays([pa.array(values) for values in zip(*records)], schema=schema)


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


def _iter_parquet_batches(
    parquet_path: Path,
    columns: Iterable[str],
    batch_size: int = 65_536,
) -> Iterable[pd.DataFrame]:
    pq, _pa = _import_pyarrow_parquet()
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(columns=list(columns), batch_size=batch_size):
        yield batch.to_pandas()


def _overview_dataframe(name: str, df: pd.DataFrame, max_rows: int = 5) -> None:
    print(f"\n{name}")
    print(f"Rows: {len(df):,}; columns: {len(df.columns):,}")
    if df.empty:
        print("(empty)")
    else:
        print(df.head(max_rows))


def _overview_parquet_file(path: Path, name: str, max_rows: int = 5) -> None:
    print(f"\n{name}: {path}")
    if not path.exists():
        print("Missing")
        return

    pq, _pa = _import_pyarrow_parquet()
    parquet_file = pq.ParquetFile(path)
    print(f"Rows: {parquet_file.metadata.num_rows:,}; columns: {len(parquet_file.schema.names):,}")
    for batch in parquet_file.iter_batches(batch_size=max_rows):
        preview_df = batch.to_pandas()
        print(preview_df.head(max_rows) if not preview_df.empty else "(empty)")
        break


def _overview_parquet_dataset(path: Path, name: str, max_rows: int = 5) -> None:
    print(f"\n{name}: {path}")
    if not path.exists():
        print("Missing")
        return

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


def _overview_csv_file(path: Path, name: str, max_rows: int = 5) -> None:
    print(f"\n{name}: {path}")
    if not path.exists():
        print("Missing")
        return

    preview_df = pd.read_csv(path, nrows=max_rows)
    with path.open(encoding="utf-8") as csv_file:
        row_count = sum(1 for _line in csv_file)
    row_count = max(0, row_count - 1)
    print(f"Rows: {row_count:,}; columns: {len(preview_df.columns):,}")
    print(preview_df if not preview_df.empty else "(empty)")


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


def _summarize_relationships_from_parquet(
    parquet_path: Path,
    batch_size: int = 65_536,
) -> pd.DataFrame:
    print(f"Summarizing relationship counts from cached costed parquet: {parquet_path}")
    summaries = []
    for variant in _plv_variant_specs():
        plv_type = str(variant["plv_type"])
        net_relationship_column = f"Physical vs {plv_type} Net Benefit Relationship"
        bpd_relationship_column = f"Physical vs {plv_type} Benefit Per Dollar Relationship"
        net_relationship_labels = _plv_relationship_labels(plv_type, "net_benefit")
        bpd_relationship_labels = _plv_relationship_labels(plv_type, "benefit_per_dollar")
        net_benefit_counts: Counter[str] = Counter()
        benefit_per_dollar_counts: Counter[str] = Counter()

        for batch_df in _iter_parquet_batches(
            parquet_path,
            columns=[net_relationship_column, bpd_relationship_column],
            batch_size=batch_size,
        ):
            net_benefit_counts.update(batch_df[net_relationship_column].astype(str).tolist())
            benefit_per_dollar_counts.update(batch_df[bpd_relationship_column].astype(str).tolist())

        net_benefit_summary = pd.DataFrame(
            {
                COLUMN_NAMES['plv_type']: [plv_type] * len(net_relationship_labels),
                COLUMN_NAMES['relationship_code']: list(net_relationship_labels.keys()),
                COLUMN_NAMES['net_benefit_relationship_description']: list(net_relationship_labels.values()),
                COLUMN_NAMES['net_benefit_scenario_count']: [
                    int(net_benefit_counts.get(code, 0)) for code in net_relationship_labels.keys()
                ],
            }
        )
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

        summary = net_benefit_summary.merge(
            benefit_per_dollar_summary,
            on=[COLUMN_NAMES['plv_type'], COLUMN_NAMES['relationship_code']],
            how="outer",
            validate="one_to_one",
        )
        summary = summary[
            [
                COLUMN_NAMES['plv_type'],
                COLUMN_NAMES['relationship_code'],
                COLUMN_NAMES['net_benefit_relationship_description'],
                COLUMN_NAMES['benefit_per_dollar_relationship_description'],
                COLUMN_NAMES['net_benefit_scenario_count'],
                COLUMN_NAMES['benefit_per_dollar_scenario_count'],
            ]
        ]
        total_row = pd.DataFrame(
            {
                COLUMN_NAMES['plv_type']: [plv_type],
                COLUMN_NAMES['relationship_code']: ["Total"],
                COLUMN_NAMES['net_benefit_relationship_description']: ["All scenarios"],
                COLUMN_NAMES['benefit_per_dollar_relationship_description']: ["All scenarios"],
                COLUMN_NAMES['net_benefit_scenario_count']: [int(sum(net_benefit_counts.values()))],
                COLUMN_NAMES['benefit_per_dollar_scenario_count']: [int(sum(benefit_per_dollar_counts.values()))],
            }
        )
        summaries.append(pd.concat([summary, total_row], ignore_index=True))

    result = pd.concat(summaries, ignore_index=True)
    result[COLUMN_NAMES['plv_type']] = pd.Categorical(result[COLUMN_NAMES['plv_type']], categories=PLV_VARIANT_ORDER, ordered=True)
    return result.sort_values([COLUMN_NAMES['plv_type'], COLUMN_NAMES['relationship_code']]).reset_index(drop=True)


def _build_relationship_boxplot_dataframe(final_df: pd.DataFrame) -> pd.DataFrame:
    """Return tidy raw rows for box plots split by relationship category, PLV variant, and metric family."""

    boxplot_frames: list[pd.DataFrame] = []
    for variant in _plv_variant_specs():
        plv_type = str(variant["plv_type"])
        net_relationship_column = f"Physical vs {plv_type} Net Benefit Relationship"
        bpd_relationship_column = f"Physical vs {plv_type} Benefit Per Dollar Relationship"
        net_relationship_labels = _plv_relationship_labels(plv_type, "net_benefit")
        bpd_relationship_labels = _plv_relationship_labels(plv_type, "benefit_per_dollar")
        net_prefix = plv_type

        net_benefit_source_columns = [
            column
            for column in [
                COLUMN_NAMES['mix_id'],
                COLUMN_NAMES['scenario_id'],
                net_relationship_column,
                "Physical - Min Net Benefit",
                "Physical - Max Net Benefit",
                f"{plv_type} - Min Net Benefit",
                f"{plv_type} - Max Net Benefit",
            ]
            if column in final_df.columns
        ]
        net_benefit_id_vars = [
            column
            for column in [COLUMN_NAMES['mix_id'], COLUMN_NAMES['scenario_id'], net_relationship_column]
            if column in net_benefit_source_columns
        ]
        net_benefit_df = final_df[net_benefit_source_columns].melt(
            id_vars=net_benefit_id_vars,
            value_vars=[
                "Physical - Min Net Benefit",
                "Physical - Max Net Benefit",
                f"{plv_type} - Min Net Benefit",
                f"{plv_type} - Max Net Benefit",
            ],
            var_name=COLUMN_NAMES['scenario_type'],
            value_name=COLUMN_NAMES['value'],
        )
        net_benefit_df[COLUMN_NAMES['metric_family']] = "Net Benefit"
        net_benefit_df[COLUMN_NAMES['plv_type']] = plv_type
        net_benefit_df[COLUMN_NAMES['relationship_code']] = net_benefit_df[net_relationship_column]
        net_benefit_df[COLUMN_NAMES['relationship_description']] = net_benefit_df[COLUMN_NAMES['relationship_code']].map(net_relationship_labels)
        net_benefit_df[COLUMN_NAMES['scenario_group']] = np.where(
            net_benefit_df[COLUMN_NAMES['scenario_type']].str.startswith("Physical"),
            "Physical Inspection",
            plv_type,
        )
        net_benefit_df[COLUMN_NAMES['scenario_variant']] = net_benefit_df[COLUMN_NAMES['scenario_type']].map(
            lambda scenario_type: _scenario_variant_label(scenario_type, plv_type)
        )

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
            column for column in [COLUMN_NAMES['mix_id'], COLUMN_NAMES['scenario_id'], bpd_relationship_column] if column in benefit_per_dollar_source_columns
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
            final_df.groupby(net_relationship_column, sort=False, observed=True)[COLUMN_NAMES['bad_records']]
            .sum()
            .reindex(RELATIONSHIP_CODE_ORDER)
            .dropna()
            .reset_index()
            .rename(
                columns={
                    net_relationship_column: COLUMN_NAMES['relationship_code'],
                    COLUMN_NAMES['bad_records']: COLUMN_NAMES['value'],
                }
            )
        )
        smuggled_chips_df[COLUMN_NAMES['plv_type']] = plv_type
        smuggled_chips_df[COLUMN_NAMES['metric_family']] = SMUGGLED_CHIPS_METRIC_FAMILY
        smuggled_chips_df[COLUMN_NAMES['relationship_description']] = smuggled_chips_df[COLUMN_NAMES['relationship_code']].map(
            net_relationship_labels
        )
        smuggled_chips_df[COLUMN_NAMES['scenario_group']] = SMUGGLED_CHIPS_SCENARIO_GROUP
        smuggled_chips_df[COLUMN_NAMES['scenario_variant']] = SMUGGLED_CHIPS_SCENARIO_VARIANT
        smuggled_chips_df[COLUMN_NAMES['scenario_type']] = SMUGGLED_CHIPS_SCENARIO_TYPE
        smuggled_chips_df[COLUMN_NAMES['mix_id']] = "All Scenarios"
        smuggled_chips_df[COLUMN_NAMES['scenario_id']] = "All Scenarios"
        smuggled_chips_df[COLUMN_NAMES['value']] = smuggled_chips_df[COLUMN_NAMES['value']].astype("float64")

        boxplot_frames.extend([net_benefit_df, benefit_per_dollar_df, smuggled_chips_df])

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
        "Physical vs PLV Renting Net Benefit Relationship",
        "Physical vs PLV Renting Benefit Per Dollar Relationship",
        "Physical vs PLV Owning Net Benefit Relationship",
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
RELATIONSHIP_MODEL_BASE_FEATURE_COLUMNS = [
    COLUMN_NAMES['total_clusters_in_mix'],
    COLUMN_NAMES['scenario_component_count'],
    COLUMN_NAMES['total_tests'],
    COLUMN_NAMES['total_component_chips'],
    COLUMN_NAMES['bad_records'],
    COLUMN_NAMES['share_diverted'],
    COLUMN_NAMES['physical_inspection_total_diverted_chips_identified'],
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
            "Physical vs PLV Renting Net Benefit Relationship",
            "Physical vs PLV Renting Benefit Per Dollar Relationship",
            "Physical vs PLV Owning Net Benefit Relationship",
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
            (COLUMN_NAMES['n_combo'], pa.string()),
            (COLUMN_NAMES['cluster_size'], pa.int64()),
            (COLUMN_NAMES['bad_records'], pa.int64()),
            (COLUMN_NAMES['total_bad_records'], pa.int64()),
            (COLUMN_NAMES['number_of_clusters'], pa.int64()),
            (COLUMN_NAMES['number_of_clusters_with_smuggling'], pa.int64()),
            (COLUMN_NAMES['tests'], pa.int64()),
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


def _add_plv_variant_cost_benefit_columns(
    result: pd.DataFrame,
    *,
    plv_type: str,
    plv_cost_per_total_chip: tuple[float, float],
) -> None:
    min_cost_column = f"{plv_type} - Min Total Cost"
    max_cost_column = f"{plv_type} - Max Total Cost"
    min_net_benefit_column = f"{plv_type} - Min Net Benefit"
    max_net_benefit_column = f"{plv_type} - Max Net Benefit"
    min_benefit_per_dollar_column = f"{plv_type} - Min Benefit Per Dollar"
    max_benefit_per_dollar_column = f"{plv_type} - Max Benefit Per Dollar"
    detected_column = f"{plv_type} - Total Diverted Chips Identified"

    result[detected_column] = result[COLUMN_NAMES['plv_total_diverted_chips_identified']]
    result[min_cost_column] = TARGET_CHIPS * plv_cost_per_total_chip[0]
    result[max_cost_column] = TARGET_CHIPS * plv_cost_per_total_chip[1]
    result[min_net_benefit_column] = (
        result[detected_column] * DOLLARS_PER_CHIP_DETECTED[0] * PLV_DISCOUNT_RATE
        - result[max_cost_column]
    )
    result[max_net_benefit_column] = (
        result[detected_column] * DOLLARS_PER_CHIP_DETECTED[1] * PLV_DISCOUNT_RATE
        - result[min_cost_column]
    )
    result[min_benefit_per_dollar_column] = (
        result[detected_column] * PLV_DISCOUNT_RATE / result[max_cost_column]
    )
    result[max_benefit_per_dollar_column] = (
        result[detected_column] * PLV_DISCOUNT_RATE / result[min_cost_column]
    )

    net_relationship_column = f"Physical vs {plv_type} Net Benefit Relationship"
    bpd_relationship_column = f"Physical vs {plv_type} Benefit Per Dollar Relationship"
    result[net_relationship_column] = _classify_interval_relationship(
        result,
        physical_min_column="Physical - Min Net Benefit",
        physical_max_column="Physical - Max Net Benefit",
        compare_min_column=min_net_benefit_column,
        compare_max_column=max_net_benefit_column,
        output_column=net_relationship_column,
        error_message=(
            f"Encountered an unexpected physical-vs-{plv_type} net benefit relationship; "
            "check the interval classification logic."
        ),
    )
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
    result["Physical - Min Net Benefit"] = (
        result[COLUMN_NAMES['physical_inspection_total_diverted_chips_identified']] * DOLLARS_PER_CHIP_DETECTED[0]
        - result["Physical Inspection - Max Total Cost"]
    )
    result["Physical - Max Net Benefit"] = (
        result[COLUMN_NAMES['physical_inspection_total_diverted_chips_identified']] * DOLLARS_PER_CHIP_DETECTED[1]
        - result["Physical Inspection - Min Total Cost"]
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