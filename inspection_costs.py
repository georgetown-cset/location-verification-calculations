from __future__ import annotations

import itertools
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


TARGET_CHIPS = 2_000_000
OUTPUT_DIR = "output"
DATA_SAVED_DIR = "data/saved"
DETECTION_LOOKUP_TABLE_PARQUET_PATH = f"{DATA_SAVED_DIR}/detection_lookup_table.parquet"
SCENARIOS_PARQUET_PATH = "data/scenario_components"
SCENARIOS_COMBINED_PARQUET_PATH = f"{DATA_SAVED_DIR}/scenarios_combined.parquet"
SCENARIOS_COSTED_PARQUET_PATH = f"{DATA_SAVED_DIR}/scenarios_costed.parquet"
RELATIONSHIP_SUMMARY_CSV_PATH = f"{OUTPUT_DIR}/relationship_summary.csv"
RELATIONSHIP_BOXPLOT_VALUES_CSV_PATH = f"{OUTPUT_DIR}/relationship_boxplot_values.csv"
MIX_STEPS = np.arange(0, 1.2, 0.2)
PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP = (11, 125)
PHYSICAL_INSPECTION_TRAVEL_COST_PER_INSPECTION = (2500, 5000)
PLV_COST_TOTAL = (272211, 76944362)
PLV_COST_PER_TOTAL_CHIP = PLV_COST_TOTAL/TARGET_CHIPS
DOLLARS_PER_CHIP_DETECTED = (1000, 60000)
PLV_BASIS_COLUMN = "Cluster Size (N)"
NET_BENEFIT_FEATURE_COLUMNS = ("Total Clusters in Mix", "Tests (n)", "Share Diverted", "Total Expected Value")
BENEFIT_PER_DOLLAR_FEATURE_COLUMNS = ("Total Clusters in Mix", "Tests (n)", "Share Diverted", "Total Expected Value")
CLUSTER_SIZES = [10, 100, 1000, 10000, 100000, 200000]
K_VALS = [1, 10, 100, 1000, 10000, 100000, 200000]
N_VALS = [1, 10, 100, 1000]
M_VALS = [0.05]
LONG_SCENARIO_VALUE_VARS = (
    "Physical - Min Net Benefit",
    "Physical - Max Net Benefit",
    "PLV - Min Net Benefit",
    "PLV - Max Net Benefit",
    "Physical - Min Benefit Per Dollar",
    "Physical - Max Benefit Per Dollar",
    "PLV - Min Benefit Per Dollar",
    "PLV - Max Benefit Per Dollar",
)
LONG_SCENARIO_BENEFIT_SCENARIOS = {
    "Physical - Min Net Benefit": "Physical (Conservative)",
    "Physical - Max Net Benefit": "Physical (Optimistic)",
    "PLV - Min Net Benefit": "PLV (Conservative)",
    "PLV - Max Net Benefit": "PLV (Optimistic)",
    "Physical - Min Benefit Per Dollar": "Physical (Conservative)",
    "Physical - Max Benefit Per Dollar": "Physical (Optimistic)",
    "PLV - Min Benefit Per Dollar": "PLV (Conservative)",
    "PLV - Max Benefit Per Dollar": "PLV (Optimistic)",
}
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
    m_vals: Iterable[float],
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
                            "Cluster Size (N)": int(N),
                            "Tests (n)": int(n),
                            "Bad Records (K)": int(K),
                            "Share Diverted": K / N,
                            "Chip-level Miss Prob (m)": float(m),
                            "Physical Inspection - P(Detect)": physical_result["p_detect"],
                            "Physical Inspection - Diverted Chips Identified": physical_result["diverted_chips_identified"],
                            "PLV - P(Detect)": plv_result["p_detect"],
                            "PLV - Diverted Chips Identified": plv_result["diverted_chips_identified"],
                        }
                    )
    result = pd.DataFrame(rows)
    if parquet_path is not None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"build_detection_lookup_table: writing table to {parquet_path}")
        result.to_parquet(parquet_path, index=False)
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
                    "Cluster Size (N)": int(cluster_size),
                    "Number of Clusters": int(count),
                }
            )

        if current_mix:
            current_mix.sort(key=lambda component: component["Cluster Size (N)"])
            mix_id = build_mix_id(current_mix)
            if mix_id in seen_mix_ids:
                continue
            for component in current_mix:
                component["Mix ID"] = mix_id
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
    return " + ".join(f"{component['Number of Clusters']}x(N={component['Cluster Size (N)']})" for component in mix_components)


def build_mix_id(mix_components: list[dict[str, int]]) -> str:
    mix_components = sorted(mix_components, key=lambda component: component["Cluster Size (N)"])
    characteristics = "_".join(
        f"N{component['Cluster Size (N)']}-C{component['Number of Clusters']}"
        for component in mix_components
    )
    return f"ClusterMix_{characteristics}"


def _group_detection_lookup(detection_lookup_table: pd.DataFrame) -> dict[tuple[int, int], dict[int, list[tuple[float, float, float, float, float]]]]:
    grouped: dict[tuple[int, int], dict[int, list[tuple[float, float, float, float, float]]]] = {}
    columns = [
        "Cluster Size (N)",
        "Bad Records (K)",
        "Tests (n)",
        "Chip-level Miss Prob (m)",
        "Physical Inspection - P(Detect)",
        "Physical Inspection - Diverted Chips Identified",
        "PLV - P(Detect)",
        "PLV - Diverted Chips Identified",
    ]
    for row in detection_lookup_table.loc[:, columns].itertuples(index=False, name=None):
        N, K, n, m, physical_p_detect, physical_identified, plv_p_detect, plv_identified = row
        grouped.setdefault((int(N), int(K)), {}).setdefault(int(n), []).append(
            (
                float(m),
                float(physical_p_detect),
                float(physical_identified),
                float(plv_p_detect),
                float(plv_identified),
            )
        )
    return grouped


def build_scenarios(
    cluster_sizes: Iterable[int],
    detection_lookup_table: pd.DataFrame,
    k_vals: Iterable[int],
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = None,
    output_parquet_path: Optional[str | Path] = SCENARIOS_PARQUET_PATH,
    return_dataframe: Optional[bool] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_sizes = list(cluster_sizes)
    k_vals = list(k_vals)
    print(f"build_scenarios: starting with {len(cluster_sizes)} cluster sizes and {len(k_vals)} K values")
    mix_data = build_mix_data(cluster_sizes=cluster_sizes, target_chips=target_chips, steps=steps)
    print(f"build_scenarios: received {len(mix_data)} mixes from build_mix_data")
    detection_grouped = _group_detection_lookup(detection_lookup_table)
    k_options_by_cluster_size = {
        int(cluster_size): [int(k) for k in k_vals if k <= cluster_size]
        for cluster_size in cluster_sizes
    }

    scenario_columns = [
        "Mix ID",
        "Scenario ID",
        "Mix Description",
        "Total Clusters in Mix",
        "Scenario Component Count",
        "K Combo",
        "N Combo",
    ]

    component_columns = [
        "Mix ID",
        "Scenario ID",
        "K Combo",
        "N Combo",
        "Cluster Size (N)",
        "Bad Records (K)",
        "Number of Clusters",
        "Tests (n)",
        "Physical Inspection - P(Detect)",
        "Physical Inspection - Diverted Chips Identified",
        "PLV - P(Detect)",
        "PLV - Diverted Chips Identified",
        "Physical Inspection - Total Diverted Chips Identified",
        "PLV - Total Diverted Chips Identified",
    ]

    if return_dataframe is None:
        return_dataframe = output_parquet_path is None

    parquet_path = Path(output_parquet_path) if output_parquet_path is not None else None
    total_scenarios = 0
    total_components = 0

    if parquet_path is not None:
        component_dataset_path = parquet_path
        _ensure_scenarios_dataset_path(component_dataset_path)
        pq, _pa = _import_pyarrow_parquet()
        for mix_components in mix_data:
            mix_id = mix_components[0]["Mix ID"]
            mix_parquet_path = _mix_parquet_path(component_dataset_path, mix_id)
            if mix_parquet_path.exists() and _component_mix_file_is_current_version(mix_parquet_path, pq):
                mix_component_count = pq.ParquetFile(mix_parquet_path).metadata.num_rows
                mix_summary_count = _count_scenarios_in_component_file(mix_parquet_path, pq)
                mix_fragment_count = 0
                print(
                    f"build_scenarios: skipping {mix_id} (already written); "
                    f"scenario rows={mix_summary_count}, component rows={mix_component_count}"
                )
            else:
                mix_scenario_records, mix_component_records = _build_mix_records(
                    mix_components=mix_components,
                    detection_grouped=detection_grouped,
                    k_options_by_cluster_size=k_options_by_cluster_size,
                    target_chips=target_chips,
                )
                mix_component_fragments = _write_scenario_dataset_to_parquet(
                    dataset_path=component_dataset_path,
                    mix_id=mix_id,
                    records=mix_component_records,
                )
                mix_summary_count = len(mix_scenario_records)
                mix_component_count = len(mix_component_records)
                mix_fragment_count = len(mix_component_fragments)
                print(
                    f"build_scenarios: wrote {mix_id}; "
                    f"scenario rows added={mix_summary_count}, component rows added={mix_component_count}, "
                    f"fragment files written={mix_fragment_count}"
                )
            total_scenarios += mix_summary_count
            total_components += mix_component_count
            print(
                f"build_scenarios: finished {mix_id}; total scenario rows={total_scenarios}, "
                f"total component rows={total_components}"
            )

        print(
            f"build_scenarios: completed with {total_scenarios} new scenario rows "
            f"and {total_components} new component rows"
        )

        if return_dataframe:
            component_df = pd.read_parquet(component_dataset_path)
            scenario_df = _derive_scenario_summary_from_components(component_df)
            return scenario_df, component_df
        return pd.DataFrame(columns=scenario_columns), pd.DataFrame(columns=component_columns)

    scenario_records: list[tuple[object, ...]] = []
    component_records: list[tuple[object, ...]] = []
    for mix_components in mix_data:
        mix_scenario_records, mix_component_records = _build_mix_records(
            mix_components=mix_components,
            detection_grouped=detection_grouped,
            k_options_by_cluster_size=k_options_by_cluster_size,
            target_chips=target_chips,
        )
        total_scenarios += len(mix_scenario_records)
        total_components += len(mix_component_records)
        if return_dataframe:
            scenario_records.extend(mix_scenario_records)
            component_records.extend(mix_component_records)
        print(
            f"build_scenarios: finished {mix_components[0]['Mix ID']}; "
            f"scenario rows added={len(mix_scenario_records)}, component rows added={len(mix_component_records)}, "
            f"total new scenario rows={total_scenarios}, total new component rows={total_components}"
        )

    print(
        f"build_scenarios: completed with {total_scenarios} new scenario rows "
        f"and {total_components} new component rows"
    )

    if return_dataframe:
        scenario_df = pd.DataFrame.from_records(scenario_records, columns=scenario_columns)
        component_df = pd.DataFrame.from_records(component_records, columns=component_columns)
    else:
        scenario_df = pd.DataFrame(columns=scenario_columns)
        component_df = pd.DataFrame(columns=component_columns)
    return scenario_df, component_df


def merge_scenarios(
    scenario_df: Optional[pd.DataFrame] = None,
    component_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if component_df is None:
        if scenario_df is None:
            return pd.DataFrame()
        component_df = scenario_df
        scenario_df = None

    if scenario_df is None or scenario_df.empty:
        scenario_df = _derive_scenario_summary_from_components(component_df)
    if component_df.empty:
        return scenario_df.copy()

    return component_df.merge(scenario_df, on="Scenario ID", how="left", validate="many_to_one")


def _derive_scenario_summary_from_components(component_df: pd.DataFrame) -> pd.DataFrame:
    if component_df.empty:
        return pd.DataFrame(
            columns=[
                "Mix ID",
                "Scenario ID",
                "Mix Description",
                "Total Clusters in Mix",
                "Scenario Component Count",
                "K Combo",
                "N Combo",
            ]
        )

    summary_source = component_df.loc[:, ["Scenario ID", "Mix ID", "K Combo", "N Combo", "Cluster Size (N)", "Number of Clusters"]].copy()
    summary_source.sort_values(["Scenario ID", "Cluster Size (N)"], inplace=True)
    summary_source["_component_label"] = (
        summary_source["Number of Clusters"].astype(str)
        + "x(N="
        + summary_source["Cluster Size (N)"].astype(str)
        + ")"
    )

    grouped = summary_source.groupby("Scenario ID", sort=False, observed=True)
    total_scenarios = grouped.ngroups
    print(
        f"_derive_scenario_summary_from_components: deriving {total_scenarios} scenario summaries "
        f"from {len(component_df)} component rows"
    )

    scenario_summary = grouped.agg(
        Mix_ID=("Mix ID", "first"),
        Mix_Description=("_component_label", " + ".join),
        Total_Clusters_in_Mix=("Number of Clusters", "sum"),
        Scenario_Component_Count=("Number of Clusters", "size"),
        K_Combo=("K Combo", "first"),
        N_Combo=("N Combo", "first"),
    ).reset_index()
    scenario_summary = scenario_summary.rename(
        columns={
            "Mix_ID": "Mix ID",
            "Mix_Description": "Mix Description",
            "Total_Clusters_in_Mix": "Total Clusters in Mix",
            "Scenario_Component_Count": "Scenario Component Count",
            "K_Combo": "K Combo",
            "N_Combo": "N Combo",
        }
    )
    scenario_summary = scenario_summary[
        [
            "Mix ID",
            "Scenario ID",
            "Mix Description",
            "Total Clusters in Mix",
            "Scenario Component Count",
            "K Combo",
            "N Combo",
        ]
    ]

    print(
        f"_derive_scenario_summary_from_components: completed {len(scenario_summary)} scenario summaries"
    )
    return scenario_summary


def _component_mix_file_is_current_version(mix_parquet_path: Path, pq) -> bool:
    expected_columns = [
        "Mix ID",
        "Scenario ID",
        "K Combo",
        "N Combo",
        "Cluster Size (N)",
        "Bad Records (K)",
        "Number of Clusters",
        "Tests (n)",
        "Physical Inspection - P(Detect)",
        "Physical Inspection - Diverted Chips Identified",
        "PLV - P(Detect)",
        "PLV - Diverted Chips Identified",
        "Physical Inspection - Total Diverted Chips Identified",
        "PLV - Total Diverted Chips Identified",
    ]
    try:
        parquet_file = pq.ParquetFile(mix_parquet_path)
    except Exception:
        return False
    return list(parquet_file.schema.names) == expected_columns


def _count_scenarios_in_component_file(mix_parquet_path: Path, pq) -> int:
    parquet_file = pq.ParquetFile(mix_parquet_path)
    scenario_ids: set[str] = set()
    for batch in parquet_file.iter_batches(columns=["Scenario ID"]):
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
        pq.write_table(table, mix_parquet_path)
        return [mix_parquet_path]

    schema = _final_scenario_component_arrow_schema(pa)
    table = _records_to_arrow_table(records, pa, schema)
    pq.write_table(table, mix_parquet_path)
    return [mix_parquet_path]


def _build_mix_records(
    *,
    mix_components: list[dict[str, int]],
    detection_grouped: dict[tuple[int, int], dict[int, list[tuple[float, float, float, float, float]]]],
    k_options_by_cluster_size: dict[int, list[int]],
    target_chips: int,
    emit_scenario: Optional[callable] = None,
    skip_scenarios: int = 0,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    scenario_records: list[tuple[object, ...]] = []
    component_records: list[tuple[object, ...]] = []
    mix_id = mix_components[0]["Mix ID"]
    mix_description = build_mix_description(mix_components)
    total_clusters_in_mix = sum(component["Number of Clusters"] for component in mix_components)
    print(
        f"build_scenarios: processing {mix_id} ({mix_description}) "
        f"with {len(mix_components)} components and {total_clusters_in_mix} total clusters"
    )

    component_data = [
        (
            component["Cluster Size (N)"],
            component["Number of Clusters"],
            (component["Cluster Size (N)"] * component["Number of Clusters"] / target_chips) * 100,
            k_options_by_cluster_size[component["Cluster Size (N)"]],
        )
        for component in mix_components
    ]
    k_options_per_component = [component[3] for component in component_data]
    scenario_counter = 0

    for combo_count, k_combo in enumerate(itertools.product(*k_options_per_component), start=1):
        n_options_per_component: list[tuple[int, ...]] = []
        for (N_comp, _num_clusters_comp, _weight_pct, _), k_val in zip(component_data, k_combo):
            n_options = detection_grouped.get((N_comp, int(k_val)), {})
            if not n_options:
                break
            n_options_per_component.append(tuple(sorted(n_options)))
        else:
            for n_combo_count, n_combo in enumerate(itertools.product(*n_options_per_component), start=1):
                scenario_counter += 1
                if scenario_counter <= skip_scenarios:
                    continue
                scenario_id = f"{mix_id}_K{'-'.join(map(str, k_combo))}_n{'-'.join(map(str, n_combo))}"

                scenario_row = (
                    mix_id,
                    scenario_id,
                    mix_description,
                    total_clusters_in_mix,
                    len(mix_components),
                    "-".join(map(str, k_combo)),
                    "-".join(map(str, n_combo)),
                )
                component_rows: list[tuple[object, ...]] = []
                for (N_comp, num_clusters_comp, _weight_pct, _), k_val, n_val in zip(component_data, k_combo, n_combo):
                    detection_info_list = detection_grouped.get((N_comp, k_val), {}).get(int(n_val), [])

                    for (
                        _m_val,
                        physical_p_detect,
                        physical_identified_per_cluster,
                        plv_p_detect,
                        plv_identified_per_cluster,
                    ) in detection_info_list:
                        component_rows.append(
                            (
                                mix_id,
                                scenario_id,
                                "-".join(map(str, k_combo)),
                                "-".join(map(str, n_combo)),
                                N_comp,
                                k_val,
                                num_clusters_comp,
                                n_val,
                                physical_p_detect,
                                physical_identified_per_cluster,
                                plv_p_detect,
                                plv_identified_per_cluster,
                                physical_identified_per_cluster * num_clusters_comp,
                                plv_identified_per_cluster * num_clusters_comp,
                            )
                        )

                if emit_scenario is not None:
                    emit_scenario(scenario_row, component_rows)
                else:
                    scenario_records.append(scenario_row)
                    component_records.extend(component_rows)

                if n_combo_count % 1000 == 0:
                    print(
                        f"build_scenarios: {mix_id} processed {combo_count} K combinations and "
                        f"{n_combo_count} n combinations; scenario rows so far for mix={len(scenario_records)}, "
                        f"component rows so far for mix={len(component_records)}"
                    )

            if combo_count % 1000 == 0:
                print(
                    f"build_scenarios: {mix_id} processed {combo_count} K combinations; "
                    f"scenario rows so far for mix={len(scenario_records)}, component rows so far for mix={len(component_records)}"
                )

    return scenario_records, component_records


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


def _parquet_file_has_columns(parquet_path: Path, expected_columns: Iterable[str]) -> bool:
    pq, _pa = _import_pyarrow_parquet()
    try:
        parquet_file = pq.ParquetFile(parquet_path)
    except Exception:
        return False
    schema_names = set(parquet_file.schema.names)
    return all(column in schema_names for column in expected_columns)


def _iter_parquet_batches(
    parquet_path: Path,
    columns: Iterable[str],
    batch_size: int = 65_536,
) -> Iterable[pd.DataFrame]:
    pq, _pa = _import_pyarrow_parquet()
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(columns=list(columns), batch_size=batch_size):
        yield batch.to_pandas()


def _classify_net_benefit_relationship(summary_df: pd.DataFrame) -> pd.Series:
    """Classify the physical-inspection and PLV net benefit intervals.

    The returned value records how the Physical Inspection interval compares to the
    PLV interval for each scenario:
    a. Physical max < PLV min
    b. Physical max is within PLV range and Physical min < PLV min
    c. Physical min and max are both within PLV range
    d. Physical min is within PLV range and Physical max > PLV max
    e. Physical min > PLV max
    f. Physical min < PLV min and Physical max > PLV max
    """

    physical_min = summary_df["Physical - Min Net Benefit"]
    physical_max = summary_df["Physical - Max Net Benefit"]
    plv_min = summary_df["PLV - Min Net Benefit"]
    plv_max = summary_df["PLV - Max Net Benefit"]

    relationship = np.select(
        [
            physical_max < plv_min,
            physical_min > plv_max,
            (physical_min < plv_min) & (physical_max > plv_max),
            (plv_min <= physical_min) & (physical_max <= plv_max),
            (physical_min < plv_min) & (plv_min <= physical_max) & (physical_max <= plv_max),
            (plv_min <= physical_min) & (physical_min <= plv_max) & (plv_max < physical_max),
        ],
        [
            "a",
            "e",
            "f",
            "c",
            "b",
            "d",
        ],
        default="unknown",
    )
    result = pd.Series(relationship, index=summary_df.index, name=NET_BENEFIT_RELATIONSHIP_COLUMN)
    if (result == "unknown").any():
        raise ValueError(
            "Encountered an unexpected physical-vs-PLV net benefit relationship; "
            "check the interval classification logic."
        )
    return result


def _classify_benefit_per_dollar_relationship(summary_df: pd.DataFrame) -> pd.Series:
    """Classify the physical-inspection and PLV benefit-per-dollar intervals."""

    physical_min = summary_df["Physical - Min Benefit Per Dollar"]
    physical_max = summary_df["Physical - Max Benefit Per Dollar"]
    plv_min = summary_df["PLV - Min Benefit Per Dollar"]
    plv_max = summary_df["PLV - Max Benefit Per Dollar"]

    relationship = np.select(
        [
            physical_max < plv_min,
            physical_min > plv_max,
            (physical_min < plv_min) & (physical_max > plv_max),
            (plv_min <= physical_min) & (physical_max <= plv_max),
            (physical_min < plv_min) & (plv_min <= physical_max) & (physical_max <= plv_max),
            (plv_min <= physical_min) & (physical_min <= plv_max) & (plv_max < physical_max),
        ],
        [
            "a",
            "e",
            "f",
            "c",
            "b",
            "d",
        ],
        default="unknown",
    )
    result = pd.Series(
        relationship,
        index=summary_df.index,
        name=BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN,
    )
    if (result == "unknown").any():
        raise ValueError(
            "Encountered an unexpected physical-vs-PLV benefit-per-dollar relationship; "
            "check the interval classification logic."
        )
    return result


def _summarize_relationship_counts(
    final_df: pd.DataFrame,
    relationship_column: str,
    relationship_labels: dict[str, str],
    count_column_name: str,
) -> pd.DataFrame:
    if relationship_column not in final_df.columns:
        raise ValueError(
            f"{relationship_column!r} is missing from the final dataframe; run add_cost_benefit_columns() first."
        )

    summary = (
        final_df.groupby(relationship_column, dropna=False)
        .size()
        .rename(count_column_name)
        .reset_index()
        .rename(columns={relationship_column: "Relationship Code"})
        .sort_values("Relationship Code")
        .reset_index(drop=True)
    )
    summary[f"{count_column_name} Description"] = summary["Relationship Code"].map(relationship_labels)
    return summary


def _summarize_relationships(final_df: pd.DataFrame) -> pd.DataFrame:
    net_benefit_summary = _summarize_relationship_counts(
        final_df,
        NET_BENEFIT_RELATIONSHIP_COLUMN,
        NET_BENEFIT_RELATIONSHIP_LABELS,
        "Net Benefit Scenario Count",
    )
    benefit_per_dollar_summary = _summarize_relationship_counts(
        final_df,
        BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN,
        BENEFIT_PER_DOLLAR_RELATIONSHIP_LABELS,
        "Benefit Per Dollar Scenario Count",
    )

    summary = net_benefit_summary.merge(
        benefit_per_dollar_summary[["Relationship Code", "Benefit Per Dollar Scenario Count"]],
        on="Relationship Code",
        how="outer",
        validate="one_to_one",
    )
    summary["Net Benefit Relationship Description"] = summary["Relationship Code"].map(NET_BENEFIT_RELATIONSHIP_LABELS)
    summary["Benefit Per Dollar Relationship Description"] = summary["Relationship Code"].map(
        BENEFIT_PER_DOLLAR_RELATIONSHIP_LABELS
    )
    summary["Net Benefit Scenario Count"] = summary["Net Benefit Scenario Count"].fillna(0).astype(int)
    summary["Benefit Per Dollar Scenario Count"] = summary["Benefit Per Dollar Scenario Count"].fillna(0).astype(int)
    summary = summary[
        [
            "Relationship Code",
            "Net Benefit Relationship Description",
            "Benefit Per Dollar Relationship Description",
            "Net Benefit Scenario Count",
            "Benefit Per Dollar Scenario Count",
        ]
    ]
    total_row = pd.DataFrame(
        {
            "Relationship Code": ["Total"],
            "Net Benefit Relationship Description": ["All scenarios"],
            "Benefit Per Dollar Relationship Description": ["All scenarios"],
            "Net Benefit Scenario Count": [int(len(final_df))],
            "Benefit Per Dollar Scenario Count": [int(len(final_df))],
        }
    )
    return pd.concat([summary, total_row], ignore_index=True)


def _summarize_relationships_from_parquet(
    parquet_path: Path,
    batch_size: int = 65_536,
) -> pd.DataFrame:
    print(f"Summarizing relationship counts from cached costed parquet: {parquet_path}")
    net_benefit_counts: Counter[str] = Counter()
    benefit_per_dollar_counts: Counter[str] = Counter()

    for batch_df in _iter_parquet_batches(
        parquet_path,
        columns=[NET_BENEFIT_RELATIONSHIP_COLUMN, BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN],
        batch_size=batch_size,
    ):
        net_benefit_counts.update(batch_df[NET_BENEFIT_RELATIONSHIP_COLUMN].astype(str).tolist())
        benefit_per_dollar_counts.update(batch_df[BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN].astype(str).tolist())

    net_benefit_summary = pd.DataFrame(
        {
            "Relationship Code": list(NET_BENEFIT_RELATIONSHIP_LABELS.keys()),
            "Net Benefit Relationship Description": list(NET_BENEFIT_RELATIONSHIP_LABELS.values()),
            "Net Benefit Scenario Count": [
                int(net_benefit_counts.get(code, 0)) for code in NET_BENEFIT_RELATIONSHIP_LABELS.keys()
            ],
        }
    )
    benefit_per_dollar_summary = pd.DataFrame(
        {
            "Relationship Code": list(BENEFIT_PER_DOLLAR_RELATIONSHIP_LABELS.keys()),
            "Benefit Per Dollar Relationship Description": list(BENEFIT_PER_DOLLAR_RELATIONSHIP_LABELS.values()),
            "Benefit Per Dollar Scenario Count": [
                int(benefit_per_dollar_counts.get(code, 0)) for code in BENEFIT_PER_DOLLAR_RELATIONSHIP_LABELS.keys()
            ],
        }
    )

    summary = net_benefit_summary.merge(
        benefit_per_dollar_summary,
        on="Relationship Code",
        how="outer",
        validate="one_to_one",
    )
    summary = summary[
        [
            "Relationship Code",
            "Net Benefit Relationship Description",
            "Benefit Per Dollar Relationship Description",
            "Net Benefit Scenario Count",
            "Benefit Per Dollar Scenario Count",
        ]
    ]
    total_row = pd.DataFrame(
        {
            "Relationship Code": ["Total"],
            "Net Benefit Relationship Description": ["All scenarios"],
            "Benefit Per Dollar Relationship Description": ["All scenarios"],
            "Net Benefit Scenario Count": [int(sum(net_benefit_counts.values()))],
            "Benefit Per Dollar Scenario Count": [int(sum(benefit_per_dollar_counts.values()))],
        }
    )
    return pd.concat([summary, total_row], ignore_index=True)


def _build_relationship_boxplot_dataframe(final_df: pd.DataFrame) -> pd.DataFrame:
    """Return tidy raw rows for box plots split by relationship category and metric family."""

    net_benefit_source_columns = [
        column
        for column in [
            "Mix ID",
            "Scenario ID",
            NET_BENEFIT_RELATIONSHIP_COLUMN,
            "Physical - Min Net Benefit",
            "Physical - Max Net Benefit",
            "PLV - Min Net Benefit",
            "PLV - Max Net Benefit",
        ]
        if column in final_df.columns
    ]
    net_benefit_id_vars = [column for column in ["Mix ID", "Scenario ID", NET_BENEFIT_RELATIONSHIP_COLUMN] if column in net_benefit_source_columns]
    net_benefit_df = final_df[net_benefit_source_columns].melt(
        id_vars=net_benefit_id_vars,
        value_vars=[
            "Physical - Min Net Benefit",
            "Physical - Max Net Benefit",
            "PLV - Min Net Benefit",
            "PLV - Max Net Benefit",
        ],
        var_name="Scenario Type",
        value_name="Value",
    )
    net_benefit_df["Metric Family"] = "Net Benefit"
    net_benefit_df["Relationship Code"] = net_benefit_df[NET_BENEFIT_RELATIONSHIP_COLUMN]
    net_benefit_df["Relationship Description"] = net_benefit_df["Relationship Code"].map(NET_BENEFIT_RELATIONSHIP_LABELS)

    benefit_per_dollar_source_columns = [
        column
        for column in [
            "Mix ID",
            "Scenario ID",
            BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN,
            "Physical - Min Benefit Per Dollar",
            "Physical - Max Benefit Per Dollar",
            "PLV - Min Benefit Per Dollar",
            "PLV - Max Benefit Per Dollar",
        ]
        if column in final_df.columns
    ]
    benefit_per_dollar_id_vars = [
        column for column in ["Mix ID", "Scenario ID", BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN] if column in benefit_per_dollar_source_columns
    ]
    benefit_per_dollar_df = final_df[benefit_per_dollar_source_columns].melt(
        id_vars=benefit_per_dollar_id_vars,
        value_vars=[
            "Physical - Min Benefit Per Dollar",
            "Physical - Max Benefit Per Dollar",
            "PLV - Min Benefit Per Dollar",
            "PLV - Max Benefit Per Dollar",
        ],
        var_name="Scenario Type",
        value_name="Value",
    )
    benefit_per_dollar_df["Metric Family"] = "Benefit Per Dollar"
    benefit_per_dollar_df["Relationship Code"] = benefit_per_dollar_df[BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN]
    benefit_per_dollar_df["Relationship Description"] = benefit_per_dollar_df["Relationship Code"].map(
        BENEFIT_PER_DOLLAR_RELATIONSHIP_LABELS
    )

    boxplot_df = pd.concat([net_benefit_df, benefit_per_dollar_df], ignore_index=True)
    boxplot_df["Scenario Group"] = np.where(
        boxplot_df["Scenario Type"].str.startswith("Physical"),
        "Physical Inspection",
        "PLV",
    )
    boxplot_df["Scenario Variant"] = boxplot_df["Scenario Type"].map(LONG_SCENARIO_BENEFIT_SCENARIOS)
    output_columns = [
        "Metric Family",
        "Relationship Code",
        "Relationship Description",
        "Scenario Group",
        "Scenario Variant",
    ]
    if "Mix ID" in boxplot_df.columns:
        output_columns.append("Mix ID")
    output_columns.extend(["Scenario ID", "Scenario Type", "Value"])
    boxplot_df = boxplot_df[output_columns]
    return boxplot_df


def _summarize_relationship_boxplot_dataframe(boxplot_df: pd.DataFrame) -> pd.DataFrame:
    """Return box-plot summary statistics for each metric family and relationship group."""

    group_columns = [
        "Metric Family",
        "Relationship Code",
        "Relationship Description",
        "Scenario Group",
        "Scenario Variant",
        "Scenario Type",
    ]
    if boxplot_df.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "Scenario Count",
                "Min",
                "Q1",
                "Median",
                "Q3",
                "Max",
                "Mean",
            ]
        )

    grouped = boxplot_df.groupby(group_columns, dropna=False)["Value"]
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
    summary = summary.rename(columns={"Scenario_Count": "Scenario Count"})
    return summary[
        [
            *group_columns,
            "Scenario Count",
            "Min",
            "Q1",
            "Median",
            "Q3",
            "Max",
            "Mean",
        ]
    ].sort_values(group_columns).reset_index(drop=True)


def _write_relationship_boxplot_values_from_parquet(
    parquet_path: Path,
    output_csv_path: Path,
    batch_size: int = 65_536,
) -> pd.DataFrame:
    print(f"Building relationship boxplot summary from cached costed parquet: {parquet_path}")
    file_columns = set(_parquet_file_columns(parquet_path))
    columns = [
        column
        for column in [
            "Mix ID",
            "Scenario ID",
            NET_BENEFIT_RELATIONSHIP_COLUMN,
            BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN,
            *LONG_SCENARIO_VALUE_VARS,
        ]
        if column in file_columns
    ]
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_values: dict[tuple[object, ...], list[float]] = {}
    group_columns = [
        "Metric Family",
        "Relationship Code",
        "Relationship Description",
        "Scenario Group",
        "Scenario Variant",
        "Scenario Type",
    ]
    for batch_df in _iter_parquet_batches(parquet_path, columns=columns, batch_size=batch_size):
        boxplot_df = _build_relationship_boxplot_dataframe(batch_df)
        if boxplot_df.empty:
            print("Processed boxplot batch with 0 rows")
            continue
        grouped = boxplot_df.groupby(group_columns, dropna=False)["Value"]
        for group_key, values in grouped:
            grouped_values.setdefault(tuple(group_key), []).extend(values.tolist())
        print(f"Accumulated {len(boxplot_df)} boxplot rows from current batch")

    if not grouped_values:
        empty_summary = _summarize_relationship_boxplot_dataframe(
            pd.DataFrame(columns=[*group_columns, "Value"])
        )
        empty_summary.to_csv(output_csv_path, index=False)
        print(f"Wrote empty relationship boxplot summary to {output_csv_path}")
        return empty_summary

    summary_rows = []
    for group_key, values in grouped_values.items():
        value_series = pd.Series(values, dtype="float64")
        summary_rows.append(
            {
                "Metric Family": group_key[0],
                "Relationship Code": group_key[1],
                "Relationship Description": group_key[2],
                "Scenario Group": group_key[3],
                "Scenario Variant": group_key[4],
                "Scenario Type": group_key[5],
                "Scenario Count": int(value_series.count()),
                "Min": float(value_series.min()),
                "Q1": float(value_series.quantile(0.25)),
                "Median": float(value_series.median()),
                "Q3": float(value_series.quantile(0.75)),
                "Max": float(value_series.max()),
                "Mean": float(value_series.mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(group_columns).reset_index(drop=True)
    print(f"Writing relationship boxplot summary CSV to {output_csv_path}")
    summary_df.to_csv(output_csv_path, index=False)
    return summary_df


def _parquet_file_columns(parquet_path: Path) -> list[str]:
    pq, _pa = _import_pyarrow_parquet()
    parquet_file = pq.ParquetFile(parquet_path)
    return list(parquet_file.schema.names)

def _final_scenario_component_arrow_schema(pa):
    return pa.schema(
        [
            ("Mix ID", pa.string()),
            ("Scenario ID", pa.string()),
            ("K Combo", pa.string()),
            ("N Combo", pa.string()),
            ("Cluster Size (N)", pa.int64()),
            ("Bad Records (K)", pa.int64()),
            ("Number of Clusters", pa.int64()),
            ("Tests (n)", pa.int64()),
            ("Physical Inspection - P(Detect)", pa.float64()),
            ("Physical Inspection - Diverted Chips Identified", pa.float64()),
            ("PLV - P(Detect)", pa.float64()),
            ("PLV - Diverted Chips Identified", pa.float64()),
            ("Physical Inspection - Total Diverted Chips Identified", pa.float64()),
            ("PLV - Total Diverted Chips Identified", pa.float64()),
        ]
    )


def _scenarios_arrow_schema(pa):
    return pa.schema(
        [
            ("Mix ID", pa.string()),
            ("Scenario ID", pa.string()),
            ("Cluster Size (N)", pa.int64()),
            ("Bad Records (K)", pa.int64()),
            ("Weight (%)", pa.float64()),
            ("Number of Clusters", pa.int64()),
            ("Mix Description", pa.string()),
            ("Total Clusters in Mix", pa.int64()),
            ("Tests (n)", pa.int64()),
            ("Share Diverted", pa.float64()),
            ("Chip-level Miss Prob (m)", pa.float64()),
            ("Physical Inspection - P(Detect)", pa.float64()),
            ("Physical Inspection - Diverted Chips Identified", pa.float64()),
            ("PLV - P(Detect)", pa.float64()),
            ("PLV - Diverted Chips Identified", pa.float64()),
            ("Physical Inspection - Total Diverted Chips Identified", pa.float64()),
            ("PLV - Total Diverted Chips Identified", pa.float64()),
        ]
    )

def add_cost_benefit_columns(
    summary_df: pd.DataFrame,
    phys_inspection_salary_cost_per_tested_chip: tuple[float, float] = PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP,
    phys_inspection_travel_cost_per_inspection: tuple[float, float] = PHYSICAL_INSPECTION_TRAVEL_COST_PER_INSPECTION,
    plv_cost_per_total_chip: tuple[float, float] = PLV_COST_PER_TOTAL_CHIP,
    plv_basis_column: str = PLV_BASIS_COLUMN,
) -> pd.DataFrame:
    result = summary_df.copy()
    result["Share Diverted"] = result["Bad Records (K)"] / result["Cluster Size (N)"]
    plv_cost_basis = result[plv_basis_column] if plv_basis_column in result.columns else result["Total Clusters in Mix"]

    result["Physical Inspection - Min Total Cost"] = (
        result["Total Clusters in Mix"] * phys_inspection_travel_cost_per_inspection[0]
        + result["Total Clusters in Mix"] * result["Tests (n)"] * phys_inspection_salary_cost_per_tested_chip[0]
    )
    result["Physical Inspection - Max Total Cost"] = (
        result["Total Clusters in Mix"] * phys_inspection_travel_cost_per_inspection[1]
        + result["Total Clusters in Mix"] * result["Tests (n)"] * phys_inspection_salary_cost_per_tested_chip[1]
    )
    result["Physical - Min Net Benefit"] = (
        result["Physical Inspection - Total Diverted Chips Identified"] * DOLLARS_PER_CHIP_DETECTED[0]
        - result["Physical Inspection - Max Total Cost"]
    )
    result["Physical - Max Net Benefit"] = (
        result["Physical Inspection - Total Diverted Chips Identified"] * DOLLARS_PER_CHIP_DETECTED[1]
        - result["Physical Inspection - Min Total Cost"]
    )
    result["Physical - Min Benefit Per Dollar"] = (
        result["Physical Inspection - Total Diverted Chips Identified"] / result["Physical Inspection - Max Total Cost"]
    )
    result["Physical - Max Benefit Per Dollar"] = (
        result["Physical Inspection - Total Diverted Chips Identified"] / result["Physical Inspection - Min Total Cost"]
    )

    result["PLV - Min Total Cost"] = plv_cost_basis * plv_cost_per_total_chip[0]
    result["PLV - Max Total Cost"] = plv_cost_basis * plv_cost_per_total_chip[1]
    result["PLV - Min Net Benefit"] = (
        result["PLV - Total Diverted Chips Identified"] * DOLLARS_PER_CHIP_DETECTED[0]
        - result["PLV - Max Total Cost"]
    )
    result["PLV - Max Net Benefit"] = (
        result["PLV - Total Diverted Chips Identified"] * DOLLARS_PER_CHIP_DETECTED[1]
        - result["PLV - Min Total Cost"]
    )
    result["PLV - Min Benefit Per Dollar"] = (
        result["PLV - Total Diverted Chips Identified"] / result["PLV - Max Total Cost"]
    )
    result["PLV - Max Benefit Per Dollar"] = (
        result["PLV - Total Diverted Chips Identified"] / result["PLV - Min Total Cost"]
    )
    result[NET_BENEFIT_RELATIONSHIP_COLUMN] = _classify_net_benefit_relationship(result)
    result[BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN] = _classify_benefit_per_dollar_relationship(result)

    return result

def run_inspection_costs_workflow(
    *,
    cluster_sizes: Iterable[int] = CLUSTER_SIZES,
    k_vals: Iterable[int] = K_VALS,
    n_vals: Iterable[int] = N_VALS,
    m_vals: Iterable[float] = M_VALS,
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = MIX_STEPS,
    phys_inspection_salary_cost_per_tested_chip: tuple[float, float] = PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP,
    phys_inspection_travel_cost_per_inspection: tuple[float, float] = PHYSICAL_INSPECTION_TRAVEL_COST_PER_INSPECTION,
    plv_cost_per_total_chip: tuple[float, float] = PLV_COST_PER_TOTAL_CHIP,
) -> dict[str, object]:
    cluster_sizes = list(cluster_sizes)
    k_vals = list(k_vals)
    n_vals = list(n_vals)
    m_vals = list(m_vals)
    steps = list(steps) if steps is not None else None
    scenarios_combined_path = Path(SCENARIOS_COMBINED_PARQUET_PATH)
    scenarios_costed_path = Path(SCENARIOS_COSTED_PARQUET_PATH)
    relationship_summary_path = Path(RELATIONSHIP_SUMMARY_CSV_PATH)
    relationship_boxplot_values_path = Path(RELATIONSHIP_BOXPLOT_VALUES_CSV_PATH)
    detection_lookup_table = None
    scenario_df = None
    component_df = None
    combined_df = None
    final_df = None
    relationship_summary_df = None
    relationship_boxplot_df = None

    print("Starting inspection costs workflow")
    if scenarios_costed_path.exists():
        print(f"Loading cached costed scenarios from {scenarios_costed_path}")
        cached_costed_columns = set(_parquet_file_columns(scenarios_costed_path))
        if NET_BENEFIT_RELATIONSHIP_COLUMN not in cached_costed_columns or BENEFIT_PER_DOLLAR_RELATIONSHIP_COLUMN not in cached_costed_columns:
            print(
                "Cached costed scenarios are missing one or more relationship columns; "
                "recomputing and saving refreshed output"
            )
            final_df = pd.read_parquet(scenarios_costed_path)
            final_df = add_cost_benefit_columns(
                final_df,
                phys_inspection_salary_cost_per_tested_chip=phys_inspection_salary_cost_per_tested_chip,
                phys_inspection_travel_cost_per_inspection=phys_inspection_travel_cost_per_inspection,
                plv_cost_per_total_chip=plv_cost_per_total_chip,
            )
            scenarios_costed_path.parent.mkdir(parents=True, exist_ok=True)
            final_df.to_parquet(scenarios_costed_path, index=False)

            print(f"Generating relationship summary at {relationship_summary_path}")
            relationship_summary_df = _summarize_relationships(final_df)
            relationship_summary_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Writing relationship summary CSV to {relationship_summary_path}")
            relationship_summary_df.to_csv(relationship_summary_path, index=False)

            print(f"Generating relationship boxplot summary at {relationship_boxplot_values_path}")
            relationship_boxplot_rows_df = _build_relationship_boxplot_dataframe(final_df)
            relationship_boxplot_df = _summarize_relationship_boxplot_dataframe(relationship_boxplot_rows_df)
            relationship_boxplot_values_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Writing relationship boxplot CSV to {relationship_boxplot_values_path}")
            relationship_boxplot_df.to_csv(relationship_boxplot_values_path, index=False)
        else:
            final_df = None
            print(f"Generating relationship summary at {relationship_summary_path}")
            relationship_summary_df = _summarize_relationships_from_parquet(scenarios_costed_path)
            relationship_summary_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Writing relationship summary CSV to {relationship_summary_path}")
            relationship_summary_df.to_csv(relationship_summary_path, index=False)

            print(f"Generating relationship boxplot summary at {relationship_boxplot_values_path}")
            relationship_boxplot_df = _write_relationship_boxplot_values_from_parquet(
                scenarios_costed_path,
                relationship_boxplot_values_path,
            )
    else:
        if not scenarios_combined_path.exists():
            print(f"Building detection lookup table for {len(cluster_sizes)} cluster sizes")
            detection_lookup_table = build_detection_lookup_table(
                cluster_sizes=cluster_sizes,
                K_vals=k_vals,
                n_vals=n_vals,
                m_vals=m_vals,
            )
            print(f"Detection lookup table rows: {len(detection_lookup_table)}")

            print("Building normalized final scenarios")
            scenario_df, component_df = build_scenarios(
                cluster_sizes=cluster_sizes,
                detection_lookup_table=detection_lookup_table,
                k_vals=k_vals,
                target_chips=target_chips,
                steps=steps,
                return_dataframe=True,
            )
            combined_df = merge_scenarios(scenario_df, component_df)
            print(f"Scenario rows: {len(scenario_df)}")
            print(f"Component rows: {len(component_df)}")
            print(combined_df.head())

            scenarios_combined_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving combined scenarios to {scenarios_combined_path}")
            combined_df.to_parquet(scenarios_combined_path, index=False)
        else:
            print(f"Loading cached combined scenarios from {scenarios_combined_path}")
            combined_df = pd.read_parquet(scenarios_combined_path)

        if combined_df is None:
            raise ValueError("combined_df is required when costed scenarios are not cached")
        print("Adding cost and benefit columns")
        final_df = add_cost_benefit_columns(
            combined_df,
            phys_inspection_salary_cost_per_tested_chip=phys_inspection_salary_cost_per_tested_chip,
            phys_inspection_travel_cost_per_inspection=phys_inspection_travel_cost_per_inspection,
            plv_cost_per_total_chip=plv_cost_per_total_chip,
        )
        scenarios_costed_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving final scenarios to {scenarios_costed_path}")
        final_df.to_parquet(scenarios_costed_path, index=False)

        relationship_summary_df = _summarize_relationships(final_df)
        relationship_summary_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing relationship summary CSV to {relationship_summary_path}")
        relationship_summary_df.to_csv(relationship_summary_path, index=False)

        print(f"Generating relationship boxplot summary at {relationship_boxplot_values_path}")
        relationship_boxplot_rows_df = _build_relationship_boxplot_dataframe(final_df)
        relationship_boxplot_df = _summarize_relationship_boxplot_dataframe(relationship_boxplot_rows_df)
        relationship_boxplot_values_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing relationship boxplot CSV to {relationship_boxplot_values_path}")
        relationship_boxplot_df.to_csv(relationship_boxplot_values_path, index=False)
    print("Inspection costs workflow complete")
    return {
        "scenario_df": scenario_df,
        "component_df": component_df,
        "combined_df": combined_df,
        "final_df": final_df,
        "relationship_summary_df": relationship_summary_df,
        "relationship_boxplot_df": relationship_boxplot_df,
    }

if __name__ == "__main__":
    run_inspection_costs_workflow()
