import math


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


print(p_detect_cluster_diversion(cluster_size=1000, chips_inspected_per_cluster=1000, diverted_chips=100, miss_prob=0.99))
