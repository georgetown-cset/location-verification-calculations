from __future__ import annotations

import unittest

import pandas as pd

from inspection_costs import (
    add_cost_benefit_columns,
    build_detection_lookup_table,
    build_efficiency_long_df,
    build_final_scenarios,
    build_mix_data,
    build_mix_description,
    expand_scenarios,
    expected_value_detected_diversion,
    fit_net_benefit_model,
    p_detect_cluster_diversion,
    run_inspection_costs_workflow,
)


class InspectionCostsTests(unittest.TestCase):
    def test_p_detect_cluster_diversion_handles_simple_case(self) -> None:
        result = p_detect_cluster_diversion(N=5, n=2, K=1, m=0.0)
        self.assertAlmostEqual(result["p_success"], 0.4)
        self.assertAlmostEqual(result["p_failure"], 0.6)

    def test_expected_value_detected_diversion_scales_by_detection_probability(self) -> None:
        result = expected_value_detected_diversion(N=5, n=2, K=1, m=0.0)
        self.assertAlmostEqual(result["p_detect"], 0.4)
        self.assertAlmostEqual(result["diverted_chips_identified"], 0.4)

    def test_build_detection_lookup_table_returns_numeric_rows(self) -> None:
        table = build_detection_lookup_table(cluster_sizes=[2], K_vals=[1], n_vals=[1], m_vals=[0.0])
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["Cluster Size (N)"], 2)
        self.assertEqual(table.iloc[0]["Bad Records (K)"], 1)
        self.assertAlmostEqual(table.iloc[0]["Physical Inspection - P(Detect)"], 0.5)
        self.assertAlmostEqual(table.iloc[0]["PLV - P(Detect)"], 1.0)

    def test_build_mix_data_supports_default_steps(self) -> None:
        mixes = build_mix_data(cluster_sizes=[2], target_chips=2)
        self.assertEqual(len(mixes), 1)
        self.assertEqual(mixes[0][0]["Cluster Size (N)"], 2)
        self.assertEqual(mixes[0][0]["Number of Clusters"], 1)

    def test_build_mix_description_formats_components(self) -> None:
        description = build_mix_description(
            [
                {"Cluster Size (N)": 100_000, "Number of Clusters": 2},
                {"Cluster Size (N)": 200_000, "Number of Clusters": 9},
            ]
        )
        self.assertEqual(description, "2x(N=100000) + 9x(N=200000)")

    def test_build_final_scenarios_includes_notebook_columns(self) -> None:
        lookup = build_detection_lookup_table(cluster_sizes=[2], K_vals=[1], n_vals=[1], m_vals=[0.0])
        final_scenarios = build_final_scenarios(cluster_sizes=[2], detection_lookup_table=lookup, k_vals=[1], target_chips=2, steps=[1.0])
        self.assertEqual(len(final_scenarios), 1)
        row = final_scenarios.iloc[0]
        self.assertEqual(row["Scenario ID"], "ClusterMix1_K1")
        self.assertEqual(row["Mix Description"], "1x(N=2)")
        self.assertEqual(row["Weight (%)"], 100.0)
        self.assertAlmostEqual(row["Physical Inspection - Total Diverted Chips Identified"], 0.5)

    def test_expand_scenarios_and_cost_columns(self) -> None:
        lookup = build_detection_lookup_table(cluster_sizes=[2], K_vals=[1], n_vals=[1], m_vals=[0.0])
        summary = expand_scenarios(cluster_sizes=[2], detection_lookup_table=lookup, k_vals=[1], target_chips=2, steps=[1.0])
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["Scenario ID"], "ClusterMix1_K1")

        costed = add_cost_benefit_columns(summary)
        self.assertIn("Physical - Min Net Benefit", costed.columns)
        self.assertIn("PLV - Max Net Benefit", costed.columns)

        self.assertAlmostEqual(costed.iloc[0]["Physical Inspection - Min Total Cost"], 2511.0)
        self.assertAlmostEqual(costed.iloc[0]["PLV - Min Total Cost"], 22.0)

        long_df = build_efficiency_long_df(costed)
        self.assertEqual(set(long_df["Benefit Scenario"]), {"Physical (Conservative)", "Physical (Optimistic)", "PLV (Conservative)", "PLV (Optimistic)"})

    def test_fit_net_benefit_model_returns_coefficients(self) -> None:
        df = pd.DataFrame(
            {
                "Total Clusters in Mix": [1, 2, 3],
                "Tests (n)": [1, 1, 1],
                "Share Diverted": [0.5, 0.5, 0.5],
                "Total Expected Value": [10, 20, 30],
                "Net Benefit ($)": [5, 10, 15],
            }
        )
        model = fit_net_benefit_model(df)
        self.assertGreaterEqual(model.r_squared, 0.99)
        self.assertIn("Total Expected Value", model.as_dict())

    def test_run_inspection_costs_workflow_returns_all_artifacts(self) -> None:
        result = run_inspection_costs_workflow(
            cluster_sizes=[2],
            k_vals=[1],
            n_vals=[1],
            m_vals=[0.0],
            target_chips=2,
            steps=[1.0],
        )

        self.assertEqual(len(result["detection_lookup_table"]), 1)
        self.assertEqual(len(result["final_scenarios"]), 1)
        self.assertEqual(len(result["summary_df"]), 1)
        self.assertIn("Physical - Min Net Benefit", result["costed_summary_df"].columns)
        self.assertIn("Benefit Scenario", result["efficiency_long_df"].columns)
        self.assertIn("R-squared", result["regression_result"].as_dict())


if __name__ == "__main__":
    unittest.main()
