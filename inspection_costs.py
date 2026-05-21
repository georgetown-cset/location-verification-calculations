from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


TARGET_CHIPS = 2_000_000
MIX_STEPS = np.arange(0, 1.1, 0.1)
PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP = (11, 125)
PHYSICAL_INSPECTION_TRAVEL_COST_PER_INSPECTION = (2500, 5000)
PLV_COST_PER_TOTAL_CHIP = (11, 125)
DOLLARS_PER_CHIP_DETECTED = (1000, 60000)
PLV_BASIS_COLUMN = "Cluster Size (N)"
NET_BENEFIT_FEATURE_COLUMNS = ("Total Clusters in Mix", "Tests (n)", "Share Diverted", "Total Expected Value")
CLUSTER_SIZES = [8, 10, 100, 1000, 10000, 100000, 200000]
K_VALS = [1, 8, 10, 100, 1000, 10000, 100000, 200000]
N_VALS = [1, 8, 10, 100, 1000]
M_VALS = [0.05]


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
) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def build_mix_data(
    cluster_sizes: Iterable[int],
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = None,
) -> list[list[dict[str, int]]]:
    if steps is None:
        steps = MIX_STEPS

    cluster_sizes = list(cluster_sizes)
    valid_mixes: list[list[dict[str, int]]] = []
    for mix_index, proportions in enumerate(itertools.product(steps, repeat=len(cluster_sizes))):
        if not np.isclose(sum(proportions), 1.0):
            continue

        mix_id = f"ClusterMix{mix_index + 1}"
        active_components = [(n, p) for n, p in zip(cluster_sizes, proportions) if p > 0]
        current_mix: list[dict[str, int]] = []
        for cluster_size, proportion in active_components:
            count = (target_chips * proportion) / cluster_size
            if count != int(count):
                current_mix = []
                break
            current_mix.append(
                {
                    "Mix ID": mix_id,
                    "Cluster Size (N)": int(cluster_size),
                    "Number of Clusters": int(count),
                }
            )

        if current_mix:
            valid_mixes.append(current_mix)

    return valid_mixes


def build_mix_description(mix_components: list[dict[str, int]]) -> str:
    return " + ".join(f"{component['Number of Clusters']}x(N={component['Cluster Size (N)']})" for component in mix_components)


def _group_detection_lookup_table(detection_lookup_table: pd.DataFrame) -> dict[tuple[int, int], list[tuple[int, float, float, float]]]:
    grouped = (
        detection_lookup_table.groupby(["Cluster Size (N)", "Bad Records (K)"])[
            [
                "Tests (n)",
                "Chip-level Miss Prob (m)",
                "Physical Inspection - Diverted Chips Identified",
                "PLV - Diverted Chips Identified",
            ]
        ]
        .apply(
            lambda frame: list(
                zip(
                    frame["Tests (n)"].astype(int),
                    frame["Chip-level Miss Prob (m)"].astype(float),
                    frame["Physical Inspection - Diverted Chips Identified"].astype(float),
                    frame["PLV - Diverted Chips Identified"].astype(float),
                )
            )
        )
        .to_dict()
    )
    return grouped


def build_final_scenarios(
    cluster_sizes: Iterable[int],
    detection_lookup_table: pd.DataFrame,
    k_vals: Iterable[int],
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    mix_data = build_mix_data(cluster_sizes=cluster_sizes, target_chips=target_chips, steps=steps)
    detection_grouped = _group_detection_lookup_table(detection_lookup_table)

    detailed_records: list[dict[str, object]] = []
    for mix_components in mix_data:
        mix_id = mix_components[0]["Mix ID"]
        mix_description = build_mix_description(mix_components)
        total_clusters_in_mix = sum(component["Number of Clusters"] for component in mix_components)

        k_options_per_component = []
        for component in mix_components:
            valid_ks = [k for k in k_vals if k <= component["Cluster Size (N)"]]
            k_options_per_component.append(valid_ks)

        for k_combo in itertools.product(*k_options_per_component):
            scenario_id = f"{mix_id}_K{'-'.join(map(str, k_combo))}"

            for component, k_val in zip(mix_components, k_combo):
                N_comp = component["Cluster Size (N)"]
                num_clusters_comp = component["Number of Clusters"]
                weight_pct = (N_comp * num_clusters_comp / target_chips) * 100
                detection_info_list = detection_grouped.get((N_comp, k_val), [])

                for n_val, m_val, phys_identified_per_cluster, plv_identified_per_cluster in detection_info_list:
                    detailed_records.append(
                        {
                            "Mix ID": mix_id,
                            "Scenario ID": scenario_id,
                            "Cluster Size (N)": N_comp,
                            "Bad Records (K)": k_val,
                            "Weight (%)": weight_pct,
                            "Number of Clusters": num_clusters_comp,
                            "Mix Description": mix_description,
                            "Total Clusters in Mix": total_clusters_in_mix,
                            "Tests (n)": n_val,
                            "Share Diverted": k_val / N_comp,
                            "Chip-level Miss Prob (m)": m_val,
                            "Physical Inspection - P(Detect)": expected_value_detected_diversion(N_comp, n_val, k_val, m_val)["p_detect"],
                            "Physical Inspection - Diverted Chips Identified": phys_identified_per_cluster,
                            "PLV - P(Detect)": expected_value_detected_diversion(N_comp, N_comp, k_val, m_val)["p_detect"],
                            "PLV - Diverted Chips Identified": plv_identified_per_cluster,
                            "Physical Inspection - Total Diverted Chips Identified": phys_identified_per_cluster * num_clusters_comp,
                            "PLV - Total Diverted Chips Identified": plv_identified_per_cluster * num_clusters_comp,
                        }
                    )

    return pd.DataFrame(detailed_records)


def expand_scenarios(
    cluster_sizes: Iterable[int],
    detection_lookup_table: pd.DataFrame,
    k_vals: Iterable[int],
    target_chips: int = TARGET_CHIPS,
    steps: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    mix_data = build_mix_data(cluster_sizes=cluster_sizes, target_chips=target_chips, steps=steps)
    detection_grouped = _group_detection_lookup_table(detection_lookup_table)

    summary_records = []
    for mix_components in mix_data:
        mix_id = mix_components[0]["Mix ID"]
        total_clusters_in_mix = sum(component["Number of Clusters"] for component in mix_components)

        k_options_per_component = []
        for component in mix_components:
            valid_ks = [k for k in k_vals if k <= component["Cluster Size (N)"]]
            k_options_per_component.append(valid_ks)

        for k_combo in itertools.product(*k_options_per_component):
            scenario_id = f"{mix_id}_K{'-'.join(map(str, k_combo))}"
            scenario_nm_aggregates: dict[tuple[int, float], dict[str, float]] = {}
            total_bad_records = 0

            for component, k_val in zip(mix_components, k_combo):
                N_comp = component["Cluster Size (N)"]
                num_clusters_comp = component["Number of Clusters"]
                total_bad_records += num_clusters_comp * k_val
                detection_info_list = detection_grouped.get((N_comp, k_val), [])

                for n_val, m_val, phys_identified_per_cluster, plv_identified_per_cluster in detection_info_list:
                    agg_key = (n_val, m_val)
                    if agg_key not in scenario_nm_aggregates:
                        scenario_nm_aggregates[agg_key] = {"Phys_EV_sum": 0.0, "PLV_EV_sum": 0.0}

                    scenario_nm_aggregates[agg_key]["Phys_EV_sum"] += phys_identified_per_cluster * num_clusters_comp
                    scenario_nm_aggregates[agg_key]["PLV_EV_sum"] += plv_identified_per_cluster * num_clusters_comp

            for (n_val, m_val), ev_sums in scenario_nm_aggregates.items():
                summary_records.append(
                    {
                        "Scenario ID": scenario_id,
                        "Mix ID": mix_id,
                        "Total Clusters in Mix": total_clusters_in_mix,
                        "Cluster Size (N)": mix_components[0]["Cluster Size (N)"] if len({component["Cluster Size (N)"] for component in mix_components}) == 1 else np.nan,
                        "Tests (n)": n_val,
                        "Chip-level Miss Prob (m)": m_val,
                        "Share Diverted": total_bad_records / target_chips,
                        "Physical Inspection - Total Diverted Chips Identified": ev_sums["Phys_EV_sum"],
                        "PLV - Total Diverted Chips Identified": ev_sums["PLV_EV_sum"],
                    }
                )

    return pd.DataFrame(summary_records)


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
    dollars_per_chip_detected: tuple[float, float] = DOLLARS_PER_CHIP_DETECTED,
) -> dict[str, object]:
    detection_lookup_table = build_detection_lookup_table(
        cluster_sizes=cluster_sizes,
        K_vals=k_vals,
        n_vals=n_vals,
        m_vals=m_vals,
    )
    final_scenarios = build_final_scenarios(
        cluster_sizes=cluster_sizes,
        detection_lookup_table=detection_lookup_table,
        k_vals=k_vals,
        target_chips=target_chips,
        steps=steps,
    )
    summary_df = expand_scenarios(
        cluster_sizes=cluster_sizes,
        detection_lookup_table=detection_lookup_table,
        k_vals=k_vals,
        target_chips=target_chips,
        steps=steps,
    )
    costed_summary_df = add_cost_benefit_columns(
        summary_df,
        phys_inspection_salary_cost_per_tested_chip=phys_inspection_salary_cost_per_tested_chip,
        phys_inspection_travel_cost_per_inspection=phys_inspection_travel_cost_per_inspection,
        plv_cost_per_total_chip=plv_cost_per_total_chip,
        dollars_per_chip_detected=dollars_per_chip_detected,
    )
    efficiency_long_df = build_efficiency_long_df(
        costed_summary_df,
        dollars_per_chip_detected=dollars_per_chip_detected,
    )
    regression_result = fit_net_benefit_model(efficiency_long_df)
    return {
        "detection_lookup_table": detection_lookup_table,
        "final_scenarios": final_scenarios,
        "summary_df": summary_df,
        "costed_summary_df": costed_summary_df,
        "efficiency_long_df": efficiency_long_df,
        "regression_result": regression_result,
    }


def add_cost_benefit_columns(
    summary_df: pd.DataFrame,
    phys_inspection_salary_cost_per_tested_chip: tuple[float, float] = PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP,
    phys_inspection_travel_cost_per_inspection: tuple[float, float] = PHYSICAL_INSPECTION_TRAVEL_COST_PER_INSPECTION,
    plv_cost_per_total_chip: tuple[float, float] = PLV_COST_PER_TOTAL_CHIP,
    dollars_per_chip_detected: tuple[float, float] = DOLLARS_PER_CHIP_DETECTED,
    plv_basis_column: str = PLV_BASIS_COLUMN,
) -> pd.DataFrame:
    result = summary_df.copy()
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
        result["Physical Inspection - Total Diverted Chips Identified"] * dollars_per_chip_detected[0]
        - result["Physical Inspection - Max Total Cost"]
    )
    result["Physical - Max Net Benefit"] = (
        result["Physical Inspection - Total Diverted Chips Identified"] * dollars_per_chip_detected[1]
        - result["Physical Inspection - Min Total Cost"]
    )

    result["PLV - Min Total Cost"] = plv_cost_basis * plv_cost_per_total_chip[0]
    result["PLV - Max Total Cost"] = plv_cost_basis * plv_cost_per_total_chip[1]
    result["PLV - Min Net Benefit"] = (
        result["PLV - Total Diverted Chips Identified"] * dollars_per_chip_detected[0]
        - result["PLV - Max Total Cost"]
    )
    result["PLV - Max Net Benefit"] = (
        result["PLV - Total Diverted Chips Identified"] * dollars_per_chip_detected[1]
        - result["PLV - Min Total Cost"]
    )

    return result


def build_efficiency_long_df(
    summary_df: pd.DataFrame,
    dollars_per_chip_detected: tuple[float, float] = DOLLARS_PER_CHIP_DETECTED,
) -> pd.DataFrame:
    efficiency_long_df = summary_df.melt(
        id_vars=[column for column in summary_df.columns if "Net Benefit" not in column],
        value_vars=[
            "Physical - Min Net Benefit",
            "Physical - Max Net Benefit",
            "PLV - Min Net Benefit",
            "PLV - Max Net Benefit",
        ],
        var_name="Scenario Type",
        value_name="Net Benefit ($)",
    )

    efficiency_long_df["Benefit Scenario"] = efficiency_long_df["Scenario Type"].map(
        {
            "Physical - Min Net Benefit": "Physical (Conservative)",
            "Physical - Max Net Benefit": "Physical (Optimistic)",
            "PLV - Min Net Benefit": "PLV (Conservative)",
            "PLV - Max Net Benefit": "PLV (Optimistic)",
        }
    )
    efficiency_long_df["Total Expected Value"] = np.where(
        efficiency_long_df["Scenario Type"].str.startswith("Physical"),
        efficiency_long_df["Physical Inspection - Total Diverted Chips Identified"],
        efficiency_long_df["PLV - Total Diverted Chips Identified"],
    ) * dollars_per_chip_detected[0]
    return efficiency_long_df


@dataclass(frozen=True)
class LinearRegressionResult:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    r_squared: float

    @property
    def intercept(self) -> float:
        return float(self.coefficients[0])

    def as_dict(self) -> dict[str, float]:
        result = {"Intercept": self.intercept, "R-squared": float(self.r_squared)}
        for name, coefficient in zip(self.feature_names, self.coefficients[1:]):
            result[name] = float(coefficient)
        return result


def fit_net_benefit_model(
    efficiency_long_df: pd.DataFrame,
    feature_columns: Iterable[str] = NET_BENEFIT_FEATURE_COLUMNS,
) -> LinearRegressionResult:
    features = list(feature_columns)
    X = efficiency_long_df.loc[:, features].to_numpy(dtype=float)
    y = efficiency_long_df["Net Benefit ($)"].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    coefficients, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    predictions = X_design @ coefficients
    ss_res = float(np.sum((y - predictions) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)
    return LinearRegressionResult(feature_names=tuple(features), coefficients=coefficients, r_squared=r_squared)

run_inspection_costs_workflow()