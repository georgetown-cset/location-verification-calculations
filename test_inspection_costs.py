"""Unit and end-to-end tests for the inspection_costs workflow.

Run from the repository root with:

    python3 -m pytest test_inspection_costs.py

The end-to-end test runs the full workflow on a tiny parameter grid inside a
temporary directory (every workflow path is relative, so chdir isolates all
artifacts) and stubs out the GPU bucket-count subprocess so no network access
is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import inspection_costs
import inspection_costs_stage1
import inspection_costs_stage3
from inspection_costs import (
    COLUMN_NAMES,
    SCENARIO_COLUMNS,
    _workflow_log,
    _workflow_stage,
    load_min_clusters_by_size_from_csv,
    run_inspection_costs_workflow,
)
from inspection_costs_stage1 import (
    _hypergeom_pmf,
    _number_of_clusters_with_smuggling,
    build_detection_lookup_table,
    build_mix_data,
    build_mix_description,
    build_mix_id,
    build_scenarios,
    expected_value_detected_diversion,
    p_detect_cluster_diversion,
)
from inspection_costs_stage2 import (
    _classify_interval_relationship,
    _derive_scenario_summary_from_components,
    add_cost_value_columns,
    filter_perfect_information_scenarios,
)
from inspection_costs_stage3 import _format_rule_value, _slugify_filename


# ---------------------------------------------------------------------------
# Workflow output tests
# ---------------------------------------------------------------------------


class TestWorkflowOutput:
    def test_miss_probability_assumptions_are_scalar_constants(self):
        assert inspection_costs.PHYSICAL_INSPECTION_M == 0.05
        assert inspection_costs.PLV_M == 0.1
        assert inspection_costs.ALL_M_VALS == [0.05, 0.1]
        assert inspection_costs.SHARE_OF_CLUSTERS_WITH_SMUGGLING_VALS == [0.2, 0.5]

    def test_workflow_constant_report_order_matches_constants_block(self):
        assert inspection_costs.WORKFLOW_CONSTANT_NAMES_THROUGH_ALL_M_VALS == (
            "TARGET_CHIPS",
            "NUMBER_OF_PHYSICAL_INSPECTIONS_PER_CLUSTER_PER_YEAR",
            "PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP",
            "PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION",
            "PLV_RENTING_COST_PER_TOTAL_CHIP",
            "PLV_OWNING_COST_PER_TOTAL_CHIP",
            "MIN_DIVERTED_CHIPS",
            "MIN_SCENARIO_DIVERTED_CHIPS",
            "PLV_DISCOUNT_RATE",
            "PHYSICAL_INSPECTION_M",
            "PLV_M",
            "MIX_STEP_SIZE",
            "CLUSTER_SIZES",
            "K_VALS",
            "SHARE_OF_CLUSTERS_WITH_SMUGGLING_VALS",
            "CHIPS_INSPECTED_PER_CLUSTER_VALS",
            "ADDITIONAL_MIN_CLUSTERS_BY_SIZE",
            "MIX_STEPS",
            "ALL_M_VALS",
        )

    def test_stage_output_uses_pipe_section_format(self, capsys):
        _workflow_stage("Stage 1 / Detection Lookup")

        assert capsys.readouterr().out == "\n=== Stage 1 / Detection Lookup ===\n"

    def test_log_output_uses_pipe_format(self, capsys):
        _workflow_log("Workflow", "Starting inspection costs workflow", kind="START")

        assert capsys.readouterr().out == "[START | Workflow] Starting inspection costs workflow\n"

    def test_workflow_completion_log_is_printed(self, capsys):
        _workflow_log("Workflow", "Inspection costs workflow complete in 1.23 seconds", kind="DONE")

        assert capsys.readouterr().out == "[DONE  | Workflow] Inspection costs workflow complete in 1.23 seconds\n"

    def test_non_workflow_completion_log_is_suppressed(self, capsys):
        _workflow_log("Stage 1 / Detection Lookup", "Complete", kind="DONE")

        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Stage 1 unit tests: probability math
# ---------------------------------------------------------------------------


class TestHypergeomPmf:
    def test_pmf_sums_to_one(self):
        total = sum(_hypergeom_pmf(x, 20, 6, 8) for x in range(0, 9))
        assert total == pytest.approx(1.0)

    def test_known_value(self):
        # P(X=0) drawing 5 from 10 with 2 marked = C(8,5)/C(10,5) = 56/252.
        assert _hypergeom_pmf(0, 10, 2, 5) == pytest.approx(56 / 252)

    def test_out_of_range_is_zero(self):
        assert _hypergeom_pmf(-1, 10, 2, 5) == 0.0
        assert _hypergeom_pmf(3, 10, 2, 5) == 0.0  # x > diverted_chips
        assert _hypergeom_pmf(6, 10, 8, 5) == 0.0  # x > chips inspected
        # Not enough non-diverted chips to fill the rest of the sample.
        assert _hypergeom_pmf(0, 10, 8, 5) == 0.0

    def test_invalid_parameters_are_zero(self):
        assert _hypergeom_pmf(0, 10, 12, 5) == 0.0  # diverted > cluster
        assert _hypergeom_pmf(0, 10, 2, 12) == 0.0  # inspected > cluster


class TestPDetectClusterDiversion:
    def test_zero_diverted_chips(self):
        result = p_detect_cluster_diversion(100, 10, 0, 0.05)
        assert result == {"p_success": 0.0, "p_failure": 1.0}

    def test_zero_chips_inspected(self):
        result = p_detect_cluster_diversion(100, 0, 10, 0.05)
        assert result == {"p_success": 0.0, "p_failure": 1.0}

    def test_perfect_test_full_inspection_always_detects(self):
        result = p_detect_cluster_diversion(10, 10, 3, 0.0)
        assert result["p_success"] == pytest.approx(1.0)

    def test_useless_test_never_detects(self):
        result = p_detect_cluster_diversion(10, 10, 3, 1.0)
        assert result["p_success"] == pytest.approx(0.0)

    def test_full_inspection_failure_is_miss_prob_to_the_k(self):
        # Inspecting every chip guarantees all K diverted chips are sampled,
        # so the only failure mode is missing each one independently.
        result = p_detect_cluster_diversion(10, 10, 3, 0.5)
        assert result["p_failure"] == pytest.approx(0.5**3)
        assert result["p_success"] == pytest.approx(1 - 0.5**3)

    def test_partial_inspection_perfect_test(self):
        # m=0 means failure only when the sample contains zero diverted chips.
        result = p_detect_cluster_diversion(10, 5, 2, 0.0)
        assert result["p_failure"] == pytest.approx(56 / 252)
        assert result["p_success"] == pytest.approx(1 - 56 / 252)

    def test_probabilities_complementary(self):
        result = p_detect_cluster_diversion(1000, 100, 50, 0.05)
        assert result["p_success"] + result["p_failure"] == pytest.approx(1.0)
        assert 0.0 <= result["p_success"] <= 1.0


class TestExpectedValueDetectedDiversion:
    def test_identified_is_p_detect_times_k(self):
        result = expected_value_detected_diversion(100, 100, 10, 0.1)
        assert result["diverted_chips_identified"] == pytest.approx(
            result["p_detect"] * 10
        )

    def test_zero_diversion_identifies_nothing(self):
        result = expected_value_detected_diversion(100, 10, 0, 0.1)
        assert result["p_detect"] == 0.0
        assert result["diverted_chips_identified"] == 0.0


# ---------------------------------------------------------------------------
# Stage 1 unit tests: detection lookup table
# ---------------------------------------------------------------------------


class TestBuildDetectionLookupTable:
    def test_skips_impossible_combinations(self):
        table = build_detection_lookup_table(
            cluster_sizes=[10],
            K_vals=[0, 10, 100],
            chips_inspected_per_cluster_vals=[0, 10, 100],
            m_vals=[0.05],
            parquet_path=None,
        )
        # K=100 and inspected=100 both exceed the cluster size of 10.
        assert set(table[COLUMN_NAMES["bad_records"]]) == {0, 10}
        assert set(table[COLUMN_NAMES["chips_inspected_per_cluster"]]) == {0, 10}
        assert len(table) == 4  # 2 inspected x 2 K x 1 m

    def test_plv_uses_full_cluster_inspection(self):
        table = build_detection_lookup_table(
            cluster_sizes=[10],
            K_vals=[10],
            chips_inspected_per_cluster_vals=[0],
            m_vals=[0.5],
            parquet_path=None,
        )
        row = table.iloc[0]
        # Physical inspects 0 chips -> no detection; PLV inspects all 10.
        assert row[COLUMN_NAMES["physical_inspection_p_detect"]] == 0.0
        assert row[COLUMN_NAMES["plv_p_detect"]] == pytest.approx(1 - 0.5**10)

    def test_parquet_round_trip(self, tmp_path):
        parquet_path = tmp_path / "lookup.parquet"
        table = build_detection_lookup_table(
            cluster_sizes=[10, 100],
            K_vals=[0, 10],
            chips_inspected_per_cluster_vals=[0, 1],
            m_vals=[0.05, 0.1],
            parquet_path=parquet_path,
        )
        assert parquet_path.exists()
        loaded = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(loaded, table)

    def test_no_write_when_path_is_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        build_detection_lookup_table(
            cluster_sizes=[10],
            K_vals=[0],
            chips_inspected_per_cluster_vals=[0],
            m_vals=[0.05],
            parquet_path=None,
        )
        assert not Path("data").exists()


# ---------------------------------------------------------------------------
# Stage 1 unit tests: mixes
# ---------------------------------------------------------------------------


class TestBuildMixData:
    def test_single_cluster_size(self):
        mixes = build_mix_data([10], target_chips=100, steps=[0.0, 1.0])
        assert len(mixes) == 1
        component = mixes[0][0]
        assert component[COLUMN_NAMES["cluster_size"]] == 10
        assert component[COLUMN_NAMES["number_of_clusters"]] == 10

    def test_two_cluster_sizes_enumerates_expected_mixes(self):
        mixes = build_mix_data([10, 100], target_chips=1000, steps=[0.0, 0.5, 1.0])
        mix_ids = {mix[0][COLUMN_NAMES["mix_id"]] for mix in mixes}
        assert mix_ids == {
            "ClusterMix_N10-C100",
            "ClusterMix_N10-C50_N100-C5",
            "ClusterMix_N100-C10",
        }

    def test_every_mix_sums_to_target(self):
        mixes = build_mix_data([10, 100], target_chips=1000, steps=[0.0, 0.5, 1.0])
        for mix in mixes:
            total = sum(
                component[COLUMN_NAMES["cluster_size"]]
                * component[COLUMN_NAMES["number_of_clusters"]]
                for component in mix
            )
            assert total == 1000

    def test_minimum_cluster_constraint_filters_mixes(self):
        mixes = build_mix_data(
            [10, 100],
            target_chips=1000,
            steps=[0.0, 0.5, 1.0],
            min_clusters_by_size={100: 1},
        )
        mix_ids = {mix[0][COLUMN_NAMES["mix_id"]] for mix in mixes}
        # The all-size-10 mix has zero size-100 clusters and must be dropped.
        assert "ClusterMix_N10-C100" not in mix_ids
        assert mix_ids == {"ClusterMix_N10-C50_N100-C5", "ClusterMix_N100-C10"}

    def test_unknown_cluster_size_in_minimums_raises(self):
        with pytest.raises(ValueError, match="not present in cluster_sizes"):
            build_mix_data(
                [10],
                target_chips=100,
                steps=[0.0, 1.0],
                min_clusters_by_size={100_000: 1},
            )

    def test_negative_minimum_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            build_mix_data(
                [10],
                target_chips=100,
                steps=[0.0, 1.0],
                min_clusters_by_size={10: -1},
            )


class TestMixHelpers:
    def _components(self):
        return [
            {COLUMN_NAMES["cluster_size"]: 100, COLUMN_NAMES["number_of_clusters"]: 5},
            {COLUMN_NAMES["cluster_size"]: 10, COLUMN_NAMES["number_of_clusters"]: 50},
        ]

    def test_build_mix_id_sorts_by_cluster_size(self):
        assert build_mix_id(self._components()) == "ClusterMix_N10-C50_N100-C5"

    def test_build_mix_description(self):
        assert build_mix_description(self._components()) == "5x(N=100) + 50x(N=10)"

    @pytest.mark.parametrize(
        ("number_of_clusters", "share_of_clusters_with_smuggling", "expected"),
        [(0, 0.25, 0), (1, 0.25, 1), (2, 0.25, 1), (4, 0.25, 1), (6, 0.25, 2), (10, 0.25, 3), (100, 0.25, 25), (10, 0.5, 5)],
    )
    def test_number_of_clusters_with_smuggling(self, number_of_clusters, share_of_clusters_with_smuggling, expected):
        assert _number_of_clusters_with_smuggling(number_of_clusters, share_of_clusters_with_smuggling) == expected


# ---------------------------------------------------------------------------
# Stage 1 unit tests: scenario generation (in-memory mode)
# ---------------------------------------------------------------------------


class TestBuildScenarios:
    def _build(self, min_scenario_diverted_chips):
        lookup = build_detection_lookup_table(
            cluster_sizes=[10],
            K_vals=[0, 10],
            chips_inspected_per_cluster_vals=[0, 10],
            m_vals=[0.05, 0.1],
            parquet_path=None,
        )
        return build_scenarios(
            cluster_sizes=[10],
            detection_lookup_table=lookup,
            k_vals=[0, 10],
            min_scenario_diverted_chips=min_scenario_diverted_chips,
            target_chips=100,
            steps=[0.0, 1.0],
            output_parquet_path=None,
        )

    def test_in_memory_columns_and_invariants(self):
        df = self._build(min_scenario_diverted_chips=0)
        assert list(df.columns) == SCENARIO_COLUMNS
        # One mix, 2 smuggling shares x 2 K options x 2 inspected options.
        assert df[COLUMN_NAMES["scenario_id"]].nunique() == 8
        # Single-component mix: one row per scenario, totalling target chips.
        assert (df[COLUMN_NAMES["total_component_chips"]] == 100).all()
        assert (
            df[COLUMN_NAMES["total_bad_records"]]
            == df[COLUMN_NAMES["bad_records"]]
            * df[COLUMN_NAMES["number_of_clusters_with_smuggling"]]
        ).all()

    def test_min_scenario_diverted_chips_filters_zero_k(self):
        df = self._build(min_scenario_diverted_chips=1)
        assert df[COLUMN_NAMES["scenario_id"]].nunique() == 4
        assert (df[COLUMN_NAMES["bad_records"]] == 10).all()

    def test_share_options_create_distinct_scenarios(self):
        lookup = build_detection_lookup_table(
            cluster_sizes=[10],
            K_vals=[10],
            chips_inspected_per_cluster_vals=[10],
            m_vals=[0.05, 0.1],
            parquet_path=None,
        )
        df = build_scenarios(
            cluster_sizes=[10],
            detection_lookup_table=lookup,
            k_vals=[10],
            min_scenario_diverted_chips=0,
            share_of_clusters_with_smuggling_vals=[0.1, 0.5],
            target_chips=100,
            steps=[1.0],
            output_parquet_path=None,
        )

        assert df[COLUMN_NAMES["scenario_id"]].nunique() == 2
        assert set(df[COLUMN_NAMES["share_of_clusters_with_smuggling"]]) == {0.1, 0.5}
        assert set(df[COLUMN_NAMES["number_of_clusters_with_smuggling"]]) == {1, 5}
        assert set(df[COLUMN_NAMES["total_bad_records"]]) == {10, 50}

    def test_negative_minimum_raises(self):
        lookup = build_detection_lookup_table(
            cluster_sizes=[10],
            K_vals=[0],
            chips_inspected_per_cluster_vals=[0],
            m_vals=[0.05],
            parquet_path=None,
        )
        with pytest.raises(ValueError, match="non-negative"):
            build_scenarios(
                cluster_sizes=[10],
                detection_lookup_table=lookup,
                k_vals=[0],
                min_scenario_diverted_chips=-1,
                output_parquet_path=None,
            )

    def test_return_dataframe_reads_generated_parquet_dataset(self, tmp_path, monkeypatch):
        lookup = build_detection_lookup_table(
            cluster_sizes=[10],
            K_vals=[0, 10],
            chips_inspected_per_cluster_vals=[0, 10],
            m_vals=[0.05, 0.1],
            parquet_path=None,
        )
        original_read_parquet = pd.read_parquet
        read_parquet_calls = []

        def counting_read_parquet(*args, **kwargs):
            read_parquet_calls.append(args[0])
            return original_read_parquet(*args, **kwargs)

        monkeypatch.setattr(pd, "read_parquet", counting_read_parquet)
        output_parquet_path = tmp_path / "scenario_components"

        df = build_scenarios(
            cluster_sizes=[10],
            detection_lookup_table=lookup,
            k_vals=[0, 10],
            min_scenario_diverted_chips=0,
            target_chips=100,
            steps=[0.0, 1.0],
            output_parquet_path=output_parquet_path,
            return_dataframe=True,
        )

        assert read_parquet_calls == [output_parquet_path]
        assert df[COLUMN_NAMES["scenario_id"]].nunique() == 8


# ---------------------------------------------------------------------------
# Stage 2 unit tests
# ---------------------------------------------------------------------------


def _component_frame():
    """Two-component scenario plus a one-component scenario."""
    rows = [
        # scenario S1: two components.
        {
            COLUMN_NAMES["mix_id"]: "MixA",
            COLUMN_NAMES["scenario_id"]: "S1",
            COLUMN_NAMES["mix_description"]: "desc",
            COLUMN_NAMES["k_combo"]: "10-100",
            COLUMN_NAMES["chips_inspected_per_cluster_combo"]: "1-10",
            COLUMN_NAMES["share_of_clusters_with_smuggling"]: 0.25,
            COLUMN_NAMES["physical_inspection_chip_level_miss_prob"]: 0.05,
            COLUMN_NAMES["plv_chip_level_miss_prob"]: 0.1,
            COLUMN_NAMES["total_tests"]: 50,
            COLUMN_NAMES["total_component_chips"]: 500,
            COLUMN_NAMES["total_bad_records"]: 30,
            COLUMN_NAMES["number_of_clusters"]: 50,
            COLUMN_NAMES["number_of_clusters_with_smuggling"]: 13,
            COLUMN_NAMES["physical_inspection_total_diverted_chips_identified"]: 4.0,
            COLUMN_NAMES["plv_total_diverted_chips_identified"]: 9.0,
        },
        {
            COLUMN_NAMES["mix_id"]: "MixA",
            COLUMN_NAMES["scenario_id"]: "S1",
            COLUMN_NAMES["mix_description"]: "desc",
            COLUMN_NAMES["k_combo"]: "10-100",
            COLUMN_NAMES["chips_inspected_per_cluster_combo"]: "1-10",
            COLUMN_NAMES["share_of_clusters_with_smuggling"]: 0.25,
            COLUMN_NAMES["physical_inspection_chip_level_miss_prob"]: 0.05,
            COLUMN_NAMES["plv_chip_level_miss_prob"]: 0.1,
            COLUMN_NAMES["total_tests"]: 50,
            COLUMN_NAMES["total_component_chips"]: 500,
            COLUMN_NAMES["total_bad_records"]: 200,
            COLUMN_NAMES["number_of_clusters"]: 5,
            COLUMN_NAMES["number_of_clusters_with_smuggling"]: 2,
            COLUMN_NAMES["physical_inspection_total_diverted_chips_identified"]: 6.0,
            COLUMN_NAMES["plv_total_diverted_chips_identified"]: 11.0,
        },
        # scenario S2: one component.
        {
            COLUMN_NAMES["mix_id"]: "MixB",
            COLUMN_NAMES["scenario_id"]: "S2",
            COLUMN_NAMES["mix_description"]: "other",
            COLUMN_NAMES["k_combo"]: "0",
            COLUMN_NAMES["chips_inspected_per_cluster_combo"]: "0",
            COLUMN_NAMES["share_of_clusters_with_smuggling"]: 0.25,
            COLUMN_NAMES["physical_inspection_chip_level_miss_prob"]: 0.05,
            COLUMN_NAMES["plv_chip_level_miss_prob"]: 0.1,
            COLUMN_NAMES["total_tests"]: 0,
            COLUMN_NAMES["total_component_chips"]: 1000,
            COLUMN_NAMES["total_bad_records"]: 0,
            COLUMN_NAMES["number_of_clusters"]: 10,
            COLUMN_NAMES["number_of_clusters_with_smuggling"]: 3,
            COLUMN_NAMES["physical_inspection_total_diverted_chips_identified"]: 0.0,
            COLUMN_NAMES["plv_total_diverted_chips_identified"]: 0.0,
        },
    ]
    return pd.DataFrame(rows)


class TestDeriveScenarioSummary:
    def test_aggregates_components_per_scenario(self):
        summary = _derive_scenario_summary_from_components(_component_frame())
        assert len(summary) == 2
        s1 = summary.loc[summary[COLUMN_NAMES["scenario_id"]] == "S1"].iloc[0]
        assert s1[COLUMN_NAMES["total_clusters_in_mix"]] == 55
        assert s1[COLUMN_NAMES["scenario_component_count"]] == 2
        assert s1[COLUMN_NAMES["total_tests"]] == 100
        assert s1[COLUMN_NAMES["total_component_chips"]] == 1000
        assert s1[COLUMN_NAMES["bad_records"]] == 230
        assert s1[COLUMN_NAMES["share_of_clusters_with_smuggling"]] == 0.25
        assert s1[COLUMN_NAMES["number_of_clusters_with_smuggling"]] == 15
        assert s1[
            COLUMN_NAMES["physical_inspection_total_diverted_chips_identified"]
        ] == pytest.approx(10.0)
        assert s1[
            COLUMN_NAMES["plv_total_diverted_chips_identified"]
        ] == pytest.approx(20.0)

    def test_empty_input_returns_empty_frame_with_schema(self):
        summary = _derive_scenario_summary_from_components(
            pd.DataFrame(columns=SCENARIO_COLUMNS)
        )
        assert summary.empty
        assert COLUMN_NAMES["scenario_id"] in summary.columns
        assert COLUMN_NAMES["share_of_clusters_with_smuggling"] in summary.columns
        assert COLUMN_NAMES["total_tests"] in summary.columns


class TestClassifyIntervalRelationship:
    def _classify(self, physical_intervals):
        df = pd.DataFrame(
            {
                "pmin": [interval[0] for interval in physical_intervals],
                "pmax": [interval[1] for interval in physical_intervals],
                "cmin": [10.0] * len(physical_intervals),
                "cmax": [20.0] * len(physical_intervals),
            }
        )
        return _classify_interval_relationship(
            df,
            physical_min_column="pmin",
            physical_max_column="pmax",
            compare_min_column="cmin",
            compare_max_column="cmax",
            output_column="code",
            error_message="unexpected relationship",
        )

    def test_all_six_codes(self):
        codes = self._classify(
            [
                (1.0, 5.0),  # a: entirely below
                (5.0, 15.0),  # b: straddles compare min
                (12.0, 18.0),  # c: inside
                (15.0, 25.0),  # d: straddles compare max
                (25.0, 30.0),  # e: entirely above
                (5.0, 25.0),  # f: spans both sides
            ]
        )
        assert list(codes) == ["a", "b", "c", "d", "e", "f"]

    def test_identical_intervals_classify_as_c(self):
        assert list(self._classify([(10.0, 20.0)])) == ["c"]

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="unexpected relationship"):
            self._classify([(np.nan, np.nan)])


class TestAddCostValueColumns:
    def _summary(self):
        return pd.DataFrame(
            [
                {
                    COLUMN_NAMES["total_clusters_in_mix"]: 10,
                    COLUMN_NAMES["total_tests"]: 100,
                    COLUMN_NAMES["total_component_chips"]: 1000,
                    COLUMN_NAMES["bad_records"]: 50,
                    COLUMN_NAMES[
                        "physical_inspection_total_diverted_chips_identified"
                    ]: 20.0,
                    COLUMN_NAMES["plv_total_diverted_chips_identified"]: 30.0,
                }
            ]
        )

    def test_hand_computed_costs_and_values(self):
        result = add_cost_value_columns(
            self._summary(),
            phys_inspection_salary_cost_per_tested_chip=(1, 2),
            phys_inspection_fixed_cost_per_inspection=(10, 20),
            plv_renting_cost_per_total_chip=(0.5, 1.0),
            plv_owning_cost_per_total_chip=(2.0, 4.0),
        )
        row = result.iloc[0]
        inspections_per_year = (
            inspection_costs.NUMBER_OF_PHYSICAL_INSPECTIONS_PER_CLUSTER_PER_YEAR
        )
        assert row["Physical Inspection - Min Total Cost"] == pytest.approx(
            (10 * 10 + 100 * 1) * inspections_per_year
        )
        assert row["Physical Inspection - Max Total Cost"] == pytest.approx(
            (10 * 20 + 100 * 2) * inspections_per_year
        )
        assert row["Physical - Min Value Per Cost"] == pytest.approx(
            20.0 / row["Physical Inspection - Max Total Cost"]
        )
        assert row["Physical - Max Value Per Cost"] == pytest.approx(
            20.0 / row["Physical Inspection - Min Total Cost"]
        )
        assert row["PLV Renting - Min Total Cost"] == pytest.approx(1000 * 0.5)
        assert row["PLV Renting - Max Total Cost"] == pytest.approx(1000 * 1.0)
        assert row["PLV Renting - Min Value Per Cost"] == pytest.approx(
            30.0 * inspection_costs.PLV_DISCOUNT_RATE / 1000.0
        )
        assert row["PLV Renting - Max Value Per Cost"] == pytest.approx(
            30.0 * inspection_costs.PLV_DISCOUNT_RATE / 500.0
        )
        assert row[COLUMN_NAMES["share_diverted"]] == pytest.approx(50 / 1000)
        # Physical value-per-cost overlaps the PLV renting range and lies entirely above owning.
        assert row["Physical vs PLV Renting Value Per Cost Relationship"] == "d"
        assert row["Physical vs PLV Owning Value Per Cost Relationship"] == "e"

    def test_does_not_mutate_input(self):
        summary = self._summary()
        original = summary.copy()
        add_cost_value_columns(summary)
        pd.testing.assert_frame_equal(summary, original)


class TestFilterPerfectInformationScenarios:
    def _costed_frame(self):
        rows = []
        for scenario_id, vpc_max in [("S1", 0.05), ("S2", 0.07), ("S3", 0.02)]:
            rows.append(
                {
                    COLUMN_NAMES["mix_id"]: "MixA",
                    COLUMN_NAMES["scenario_id"]: scenario_id,
                    COLUMN_NAMES["k_combo"]: "10",
                    COLUMN_NAMES["share_of_clusters_with_smuggling"]: 0.25,
                    COLUMN_NAMES["physical_inspection_chip_level_miss_prob"]: 0.05,
                    COLUMN_NAMES["plv_chip_level_miss_prob"]: 0.1,
                    "Physical - Max Value Per Cost": vpc_max,
                }
            )
        # A second group with a single scenario.
        rows.append(
            {
                COLUMN_NAMES["mix_id"]: "MixB",
                COLUMN_NAMES["scenario_id"]: "S4",
                COLUMN_NAMES["k_combo"]: "0",
                COLUMN_NAMES["share_of_clusters_with_smuggling"]: 0.25,
                COLUMN_NAMES["physical_inspection_chip_level_miss_prob"]: 0.05,
                COLUMN_NAMES["plv_chip_level_miss_prob"]: 0.1,
                "Physical - Max Value Per Cost": 0.0,
            }
        )
        return pd.DataFrame(rows)

    def test_keeps_best_scenario_per_group(self):
        filtered = filter_perfect_information_scenarios(self._costed_frame())
        assert set(filtered[COLUMN_NAMES["scenario_id"]]) == {"S2", "S4"}

    def test_miss_probabilities_do_not_split_groups(self):
        costed = self._costed_frame()
        duplicate_group = costed.iloc[[0]].copy()
        duplicate_group[COLUMN_NAMES["scenario_id"]] = "S5"
        duplicate_group[COLUMN_NAMES["physical_inspection_chip_level_miss_prob"]] = 0.2
        duplicate_group[COLUMN_NAMES["plv_chip_level_miss_prob"]] = 0.3
        duplicate_group["Physical - Max Value Per Cost"] = 0.09

        filtered = filter_perfect_information_scenarios(
            pd.concat([costed, duplicate_group], ignore_index=True)
        )

        assert set(filtered[COLUMN_NAMES["scenario_id"]]) == {"S5", "S4"}

    def test_empty_input_returns_empty(self):
        empty = self._costed_frame().iloc[0:0]
        filtered = filter_perfect_information_scenarios(empty)
        assert filtered.empty


# ---------------------------------------------------------------------------
# Stage 3 unit tests: small pure helpers
# ---------------------------------------------------------------------------


class TestStage3Helpers:
    def test_slugify_filename(self):
        assert _slugify_filename("Value Per Cost - PLV Renting") == (
            "value_per_cost_plv_renting"
        )

    def test_format_rule_value(self):
        assert _format_rule_value(np.nan) == "NA"
        assert _format_rule_value(5) == "5"
        assert _format_rule_value(5.0) == "5"
        assert _format_rule_value(0.123456789) == "0.123457"


# ---------------------------------------------------------------------------
# Workflow-entry unit tests
# ---------------------------------------------------------------------------


class TestLoadMinClustersBySizeFromCsv:
    def test_loads_bounds(self, tmp_path):
        csv_path = tmp_path / "buckets.csv"
        csv_path.write_text(
            "bucket,lower_bound,upper_bound,unique_name_count\n"
            "a,10,100,3\n"
            "b,100,1000,7\n"
        )
        assert load_min_clusters_by_size_from_csv(csv_path) == {10: 3, 100: 7}

    def test_skips_rows_with_missing_values(self, tmp_path):
        csv_path = tmp_path / "buckets.csv"
        csv_path.write_text(
            "bucket,lower_bound,upper_bound,unique_name_count\n"
            "a,10,100,3\n"
            "b,,1000,7\n"
            "c,1000,10000,\n"
        )
        assert load_min_clusters_by_size_from_csv(csv_path) == {10: 3}

    def test_missing_columns_raise(self, tmp_path):
        csv_path = tmp_path / "buckets.csv"
        csv_path.write_text("bucket,lower_bound\na,10\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_min_clusters_by_size_from_csv(csv_path)


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


@pytest.fixture
def workflow_sandbox(tmp_path, monkeypatch):
    """Run the workflow inside a temp directory with the GPU step stubbed out.

    All workflow paths are relative, so changing the working directory isolates
    every artifact the workflow writes or deletes.
    """
    monkeypatch.chdir(tmp_path)

    def fake_ensure_gpu_cluster_bucket_counts() -> Path:
        csv_path = Path(inspection_costs.GPU_CLUSTER_BUCKET_COUNTS_CSV_PATH)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(
            "bucket,lower_bound,upper_bound,unique_name_count\n"
            "10 <= X < 100,10,100,0\n"
            '"100 <= X < 1,000",100,1000,0\n'
        )
        return csv_path

    monkeypatch.setattr(
        inspection_costs,
        "ensure_gpu_cluster_bucket_counts",
        fake_ensure_gpu_cluster_bucket_counts,
    )
    return tmp_path


class TestEndToEndWorkflow:
    def test_workflow_generates_mixes_once(self, workflow_sandbox, monkeypatch):
        calls = 0
        original_build_mix_data = inspection_costs_stage1.build_mix_data

        def counting_build_mix_data(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_build_mix_data(*args, **kwargs)

        monkeypatch.setattr(inspection_costs_stage1, "build_mix_data", counting_build_mix_data)

        run_inspection_costs_workflow(
            cluster_sizes=[10, 100],
            k_vals=[0, 10, 100],
            min_scenario_diverted_chips=0,
            chips_inspected_per_cluster_vals=[0, 1, 10],
            extra_min_clusters_by_size=None,
            target_chips=1000,
            steps=[0.0, 0.5, 1.0],
        )

        assert calls == 1

    def test_workflow_streams_relationship_parquets_once_per_population(self, workflow_sandbox, monkeypatch):
        calls = 0
        original_iter_parquet_batches = inspection_costs_stage3._iter_parquet_batches

        def counting_iter_parquet_batches(*args, **kwargs):
            nonlocal calls
            calls += 1
            yield from original_iter_parquet_batches(*args, **kwargs)

        monkeypatch.setattr(inspection_costs_stage3, "_iter_parquet_batches", counting_iter_parquet_batches)

        run_inspection_costs_workflow(
            cluster_sizes=[10, 100],
            k_vals=[0, 10, 100],
            min_scenario_diverted_chips=0,
            chips_inspected_per_cluster_vals=[0, 1, 10],
            extra_min_clusters_by_size=None,
            target_chips=1000,
            steps=[0.0, 0.5, 1.0],
        )

        assert calls == 2

    @pytest.fixture(scope="class")
    def workflow_result(self, request, tmp_path_factory):
        """Run the full workflow once on a tiny grid and share it across tests."""
        tmp_path = tmp_path_factory.mktemp("inspection_costs_e2e")
        monkeypatch = pytest.MonkeyPatch()
        request.addfinalizer(monkeypatch.undo)
        monkeypatch.chdir(tmp_path)

        def fake_ensure_gpu_cluster_bucket_counts() -> Path:
            csv_path = Path(inspection_costs.GPU_CLUSTER_BUCKET_COUNTS_CSV_PATH)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text(
                "bucket,lower_bound,upper_bound,unique_name_count\n"
                "10 <= X < 100,10,100,0\n"
                '"100 <= X < 1,000",100,1000,0\n'
            )
            return csv_path

        monkeypatch.setattr(
            inspection_costs,
            "ensure_gpu_cluster_bucket_counts",
            fake_ensure_gpu_cluster_bucket_counts,
        )

        result = run_inspection_costs_workflow(
            cluster_sizes=[10, 100],
            k_vals=[0, 10, 100],
            min_scenario_diverted_chips=0,
            chips_inspected_per_cluster_vals=[0, 1, 10],
            extra_min_clusters_by_size=None,
            target_chips=1000,
            steps=[0.0, 0.5, 1.0],
        )
        return tmp_path, result

    def test_returns_expected_dataframes(self, workflow_result):
        _tmp_path, result = workflow_result
        expected_keys = {
            "scenario_component_df",
            "final_df",
            "perfect_information_final_df",
            "relationship_summary_df",
            "relationship_boxplot_df",
            "relationship_rules_df",
            "perfect_information_relationship_summary_df",
            "perfect_information_relationship_boxplot_df",
            "perfect_information_relationship_rules_df",
        }
        assert set(result) == expected_keys
        for key in expected_keys:
            assert isinstance(result[key], pd.DataFrame), key
            assert not result[key].empty, key

    def test_expected_scenario_counts(self, workflow_result):
        _tmp_path, result = workflow_result
        # 3 mixes x 2 smuggling shares:
        # N10-C100 (2 K x 3 inspected = 6 scenarios per share),
        # N10-C50_N100-C5 (2x2 K x 3x3 inspected = 36 per share),
        # N100-C10 (2 K x 3 inspected = 6 per share).
        final_df = result["final_df"]
        component_df = result["scenario_component_df"]
        assert len(final_df) == 96
        assert component_df[COLUMN_NAMES["scenario_id"]].nunique() == 96
        # 2 shares x (6*1 + 36*2 + 6*1) component rows.
        assert len(component_df) == 168

    def test_mixes_used_csv(self, workflow_result):
        tmp_path, _result = workflow_result
        mixes_df = pd.read_csv(tmp_path / "output" / "mixes_used.csv")
        assert set(mixes_df[COLUMN_NAMES["mix_id"]]) == {
            "ClusterMix_N10-C100",
            "ClusterMix_N10-C50_N100-C5",
            "ClusterMix_N100-C10",
        }
        assert (mixes_df[COLUMN_NAMES["total_component_chips"]] == 1000).all()

    def test_scenarios_conserve_target_chips(self, workflow_result):
        _tmp_path, result = workflow_result
        final_df = result["final_df"]
        assert (final_df[COLUMN_NAMES["total_component_chips"]] == 1000).all()

    def test_relationship_codes_are_valid(self, workflow_result):
        _tmp_path, result = workflow_result
        final_df = result["final_df"]
        valid_codes = set("abcdef")
        for plv_type in ("PLV Renting", "PLV Owning"):
            column = f"Physical vs {plv_type} Value Per Cost Relationship"
            assert set(final_df[column].astype(str)) <= valid_codes

    def test_relationship_summary_totals_match_scenario_count(self, workflow_result):
        _tmp_path, result = workflow_result
        summary_df = result["relationship_summary_df"]
        totals = summary_df.loc[
            summary_df[COLUMN_NAMES["relationship_code"]] == "Total"
        ]
        assert len(totals) == 2  # one per PLV variant
        assert (
            totals[COLUMN_NAMES["value_per_cost_scenario_count"]] == 96
        ).all()

    def test_perfect_information_subset(self, workflow_result):
        _tmp_path, result = workflow_result
        final_df = result["final_df"]
        perfect_df = result["perfect_information_final_df"]
        # One scenario per (mix, K combo, smuggling share): 2 x (2 + 4 + 2) groups.
        assert len(perfect_df) == 16
        assert set(perfect_df[COLUMN_NAMES["scenario_id"]]) <= set(
            final_df[COLUMN_NAMES["scenario_id"]]
        )

    def test_artifact_files_written(self, workflow_result):
        tmp_path, _result = workflow_result
        expected_files = [
            "data/saved/detection_lookup_table.parquet",
            "data/saved/scenarios_combined.parquet",
            "data/saved/scenarios_costed.parquet",
            "data/saved/scenarios_costed_perfect_information.parquet",
            "output/gpu_cluster_bucket_counts.csv",
            "output/mixes_used.csv",
            "output/inspection_cost_assumptions.txt",
            "output/all/relationship_summary.csv",
            "output/all/relationship_boxplot_values.csv",
            "output/all/relationship_code_rules.csv",
            "output/perfect_information/relationship_summary.csv",
            "output/perfect_information/relationship_boxplot_values.csv",
            "output/perfect_information/relationship_code_rules.csv",
        ]
        for relative_path in expected_files:
            assert (tmp_path / relative_path).is_file(), relative_path

        component_fragments = list(
            (tmp_path / "data" / "scenario_components").rglob("*.parquet")
        )
        assert len(component_fragments) == 3  # one fragment per mix

        for images_dir in (
            tmp_path / "output" / "all" / "images",
            tmp_path / "output" / "perfect_information" / "images",
        ):
            assert list(images_dir.glob("*.jpeg")), images_dir

    def test_costed_parquet_round_trips(self, workflow_result):
        tmp_path, result = workflow_result
        loaded = pd.read_parquet(tmp_path / "data/saved/scenarios_costed.parquet")
        assert len(loaded) == len(result["final_df"])
        assert list(loaded.columns) == list(result["final_df"].columns)

    def test_assumptions_report_contains_constants(self, workflow_result):
        tmp_path, _result = workflow_result
        report_text = (
            tmp_path / "output" / "inspection_cost_assumptions.txt"
        ).read_text(encoding="utf-8")
        assert "Workflow Constants" in report_text
        assert "TARGET_CHIPS" in report_text
        assert "PLV_DISCOUNT_RATE" in report_text


class TestWorkflowReset:
    def test_rerun_clears_prior_artifacts(self, workflow_sandbox):
        stale_file = workflow_sandbox / "data" / "saved" / "stale.txt"
        stale_file.parent.mkdir(parents=True, exist_ok=True)
        stale_file.write_text("stale")

        run_inspection_costs_workflow(
            cluster_sizes=[10, 100],
            k_vals=[0, 100],
            min_scenario_diverted_chips=0,
            chips_inspected_per_cluster_vals=[0, 10],
            extra_min_clusters_by_size=None,
            target_chips=1000,
            steps=[0.0, 1.0],
        )
        assert not stale_file.exists()
        assert (workflow_sandbox / "data" / "saved" / "scenarios_costed.parquet").is_file()
