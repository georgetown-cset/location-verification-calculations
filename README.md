# Location Verification Inspection Cost Model

This repository models the cost effectiveness of two approaches for detecting whether export controlled AI chips in a GPU cluster have been smuggled from their expected location:

- Physical inspections, where a limited number of chips in a cluster are tested directly.
- Ping-based location verification (PLV), which uses the time delay for a 

The model generates possible GPU cluster mixes, diversion scenarios, testing strategies, and detection probabilities. It then estimates costs, expected diverted chips identified, and value per cost for physical inspections and two PLV variants: renting landmark servers and owning landmark servers.

## Model Overview

The workflow starts with assumptions about a target chip population, possible cluster sizes, diverted-chip counts, chips inspected per cluster, and chip-level miss probabilities. It uses a hypergeometric detection model to estimate the probability that inspection detects at least one diverted chip in a cluster.

For each valid scenario, the pipeline computes:

- Total clusters and total chips in the mix.
- Number of clusters assumed to contain smuggling.
- Expected diverted chips identified by physical inspection.
- Expected diverted chips identified by PLV.
- Minimum and maximum inspection costs.
- Minimum and maximum PLV costs for renting and owning variants.
- Value per cost for physical inspection and PLV.
- Relationship codes showing how physical-inspection value-per-cost ranges compare with each PLV range.

The top-level entry point is `inspection_costs.py`. Running it executes all stages and writes cached parquet data plus CSV, text, and image outputs. Run from the repository root:

```bash
python3 inspection_costs.py
```

The workflow clears and regenerates the cached model artifacts in `data/saved/`, `data/scenario_components/`, `output/all/`, and `output/perfect_information/`. Keep copies of any prior outputs before rerunning if you need to preserve them.

## Pipeline Stages

### Stage 0: GPU Cluster Bucket Counts

`gpu_cluster_bucket_counts.py` downloads or reuses `data/gpu_clusters.csv` from Epoch AI's GPU cluster dataset, filters it to relevant non-U.S./non-China clusters, and counts unique clusters by chip-quantity bucket. Those counts become minimum constraints when generating possible cluster mixes.

The output is:

- `output/gpu_cluster_bucket_counts.csv`

### Stage 1: Detection Lookup and Scenario Generation

`inspection_costs_stage1.py` builds the detection lookup table and scenario-component rows.

It:

- Computes hypergeometric detection probabilities for each cluster size, diverted-chip count, chips inspected per cluster, and miss probability.
- Generates valid cluster mixes that sum to the target chip population.
- Applies minimum cluster-count constraints from the GPU cluster bucket summary.
- Expands each mix into scenario-component rows across smuggling and testing combinations.
- Writes scenario-component parquet fragments to `data/scenario_components/`.

Key outputs:

- `data/saved/detection_lookup_table.parquet`
- `data/scenario_components/`
- `data/saved/scenarios_combined.parquet`

### Stage 2: Scenario Costing

`inspection_costs_stage2.py` aggregates component rows to scenario-level rows and adds cost and value metrics.

It:

- Collapses scenario components into one row per full scenario.
- Adds physical-inspection cost ranges.
- Adds PLV renting and owning cost ranges.
- Computes value per cost for physical inspection and PLV.
- Classifies physical-vs-PLV value-per-cost relationships with codes `a` through `f`.
- Builds a perfect-information subset that keeps the best physical-inspection scenario within each comparable group.

Key outputs:

- `data/saved/scenarios_costed.parquet`
- `data/saved/scenarios_costed_perfect_information.parquet`

### Stage 3: Summaries and Reports

`inspection_costs_stage3.py` creates user-facing summaries from the costed scenario parquet files.

It:

- Counts relationship-code frequencies.
- Writes boxplot summary values.
- Renders boxplot JPEGs.
- Produces relationship-code rule summaries.

Key outputs:

- `output/all/relationship_summary.csv`
- `output/all/relationship_boxplot_values.csv`
- `output/all/relationship_code_rules.csv`
- `output/all/images/`
- `output/perfect_information/relationship_summary.csv`
- `output/perfect_information/relationship_boxplot_values.csv`
- `output/perfect_information/relationship_code_rules.csv`
- `output/perfect_information/images/`

## Directory and File Guide

```text
.
├── README.md
├── gpu_cluster_bucket_counts.py
├── inspection_costs.py
├── inspection_costs_stage1.py
├── inspection_costs_stage2.py
├── inspection_costs_stage3.py
├── test/
│   ├── test_inspection_costs_relationship_model.py
│   └── test_probability.py
├── data/
│   ├── gpu_clusters.csv
│   ├── saved/
│   │   ├── detection_lookup_table.parquet
│   │   ├── scenarios_combined.parquet
│   │   ├── scenarios_costed.parquet
│   │   └── scenarios_costed_perfect_information.parquet
│   └── scenario_components/
│       └── ... parquet fragments by cluster-mix shard
└── output/
    ├── gpu_cluster_bucket_counts.csv
    ├── all/
    │   ├── relationship_summary.csv
    │   ├── relationship_boxplot_values.csv
    │   ├── relationship_code_rules.csv
    │   └── images/
    └── perfect_information/
        ├── relationship_summary.csv
        ├── relationship_boxplot_values.csv
        ├── relationship_code_rules.csv
        └── images/
```

### Source Files

- `inspection_costs.py`: Main workflow entry point and shared constants. It coordinates all stages and defines output paths.
- `inspection_costs_stage1.py`: Detection lookup, cluster-mix generation, scenario-component generation, parquet writing, and artifact overview helpers.
- `inspection_costs_stage2.py`: Scenario aggregation, perfect-information filtering, cost calculations, value-per-cost calculations, and relationship classification.
- `inspection_costs_stage3.py`: Relationship summaries, rule summaries, boxplot data, and rendered images.
- `test/test_inspection_costs_relationship_model.py`: Standalone relationship-code statistical model report generator, kept out of the main pipeline.
- `gpu_cluster_bucket_counts.py`: Downloads, filters, and buckets the GPU cluster dataset used to constrain generated mixes.
- `test/test_probability.py`: Local probability-check script.

### Data Directories

- `data/gpu_clusters.csv`: Cached source GPU cluster dataset.
- `data/scenario_components/`: Generated parquet dataset containing scenario-component rows. Files are sharded by cluster-mix prefix.
- `data/saved/`: Generated intermediate parquet files used to avoid recomputing each stage from raw components.

### Output Directories

- `output/gpu_cluster_bucket_counts.csv`: Bucket-count summary used by the scenario generator.
- `output/all/`: Reports and plots for all generated costed scenarios.
- `output/perfect_information/`: Reports and plots for the subset where physical inspection is given perfect information about which testing strategy performs best within each comparison group.

## Important Assumptions

The main assumptions are defined near the top of `inspection_costs.py`:

- `TARGET_CHIPS`: Total chip population modeled in each scenario.
- `CLUSTER_SIZES`: Candidate cluster-size buckets.
- `K_VALS`: Candidate diverted-chip counts per cluster.
- `CHIPS_INSPECTED_PER_CLUSTER_VALS`: Candidate physical-inspection chip counts inspected per cluster.
- `PHYSICAL_INSPECTION_M_VALS`: Physical-inspection chip-level miss probabilities.
- `PLV_M_VALS`: PLV chip-level miss probabilities.
- `PHYSICAL_INSPECTION_SALARY_COST_PER_TESTED_CHIP`: Physical-inspection variable cost range.
- `PHYSICAL_INSPECTION_FIXED_COST_PER_INSPECTION`: Physical-inspection fixed cost range.
- `PLV_RENTING_COST_PER_TOTAL_CHIP`: PLV renting cost range.
- `PLV_OWNING_COST_PER_TOTAL_CHIP`: PLV owning cost range.
- `PLV_DISCOUNT_RATE`: Discount applied to PLV value.

## Dependencies

The model uses Python with:

- `numpy`
- `pandas`
- `matplotlib`
- `pyarrow`

`pyarrow` is required for parquet read/write support. The GPU cluster bucket script also uses Python's standard-library `urllib` to download the source CSV when `data/gpu_clusters.csv` is not already present.
