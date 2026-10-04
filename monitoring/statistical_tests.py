"""TransitVision AI - Statistical Drift Tests.
Implements robust mathematical test functions for distribution comparison:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov 2-sample test (KS)
- Jensen-Shannon divergence (JS)
- Wasserstein distance (Earth Mover's Distance)
- Categorical Frequency PSI / Chi-square
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional, Union
import scipy.stats as stats


def calculate_psi(
    ref_values: Union[np.ndarray, List[float], pd.Series],
    curr_values: Union[np.ndarray, List[float], pd.Series],
    num_bins: int = 10,
    bin_edges: Optional[np.ndarray] = None,
    eps: float = 1e-4
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """
    Computes Population Stability Index (PSI) between reference and current distribution
    with zero-inflation support and Dirichlet smoothing.
    
    Formula:
        PSI = sum( (Actual_i - Expected_i) * ln(Actual_i / Expected_i) )
    
    Returns:
        (psi_value, bin_edges, bin_details_dict)
    """
    ref = np.asarray(ref_values, dtype=float)
    curr = np.asarray(curr_values, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0, np.array([]), {}

    # Define bins with zero-inflation awareness if not precomputed
    if bin_edges is None:
        min_val = ref.min()
        zero_mask_ref = (ref == min_val)
        if zero_mask_ref.mean() >= 0.25:
            # Dedicated zero/min bin + positive tail quantiles
            pos_ref = ref[~zero_mask_ref]
            if len(pos_ref) > 0:
                pos_edges = np.percentile(pos_ref, np.linspace(0, 100, num_bins))
                bin_edges = np.unique(np.concatenate([[min_val - 1e-4, min_val + 1e-4], pos_edges]))
            else:
                bin_edges = np.array([min_val - 1.0, min_val + 1.0])
        else:
            quantiles = np.linspace(0, 100, num_bins + 1)
            bin_edges = np.percentile(ref, quantiles)
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.array([ref.min() - 1e-3, ref.max() + 1e-3])

    # Ensure infinite tails for complete support
    edges = np.copy(bin_edges)
    edges[0] = -np.inf
    edges[-1] = np.inf

    # Compute histogram frequencies
    ref_counts, _ = np.histogram(ref, bins=edges)
    curr_counts, _ = np.histogram(curr, bins=edges)
    k_bins = len(ref_counts)

    # Bayesian Laplace smoothing (alpha=1)
    ref_pct = (ref_counts + 1.0) / (len(ref) + k_bins)
    curr_pct = (curr_counts + 1.0) / (len(curr) + k_bins)

    # Compute PSI components
    psi_vector = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
    psi_value = float(np.sum(psi_vector))

    details = {
        "num_bins": k_bins,
        "ref_counts": ref_counts.tolist(),
        "curr_counts": curr_counts.tolist(),
        "ref_proportions": ref_pct.tolist(),
        "curr_proportions": curr_pct.tolist(),
        "psi_components": psi_vector.tolist()
    }
    return max(0.0, psi_value), bin_edges, details



def calculate_ks_test(
    ref_values: Union[np.ndarray, List[float], pd.Series],
    curr_values: Union[np.ndarray, List[float], pd.Series]
) -> Tuple[float, float]:
    """
    Computes Kolmogorov-Smirnov 2-sample test statistic and p-value.
    
    Returns:
        (ks_statistic, p_value)
    """
    ref = np.asarray(ref_values, dtype=float)
    curr = np.asarray(curr_values, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0, 1.0

    res = stats.ks_2samp(ref, curr)
    return float(res.statistic), float(res.pvalue)


def benjamini_hochberg_fdr(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """
    Applies Benjamini-Hochberg False Discovery Rate (FDR) correction to a list of p-values.
    
    Returns:
        (rejected_bool_list, adjusted_p_values_list)
    """
    m = len(p_values)
    if m == 0:
        return [], []

    p_arr = np.asarray(p_values, dtype=float)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    # Adjusted p-values (step-up)
    adj_p = np.zeros(m)
    adj_p[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        adj_p[i] = min(adj_p[i + 1], sorted_p[i] * m / (i + 1))
    adj_p = np.clip(adj_p, 0.0, 1.0)

    # Re-order to original indices
    orig_adj_p = np.zeros(m)
    orig_adj_p[sorted_indices] = adj_p

    rejected = (orig_adj_p <= alpha).tolist()
    return rejected, orig_adj_p.tolist()



def calculate_js_divergence(
    ref_values: Union[np.ndarray, List[float], pd.Series],
    curr_values: Union[np.ndarray, List[float], pd.Series],
    num_bins: int = 10,
    bin_edges: Optional[np.ndarray] = None,
    eps: float = 1e-5
) -> float:
    """
    Computes Jensen-Shannon Divergence (bounded in [0, 1] for log base 2).
    
    Formula:
        M = 0.5 * (P + Q)
        JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    """
    ref = np.asarray(ref_values, dtype=float)
    curr = np.asarray(curr_values, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0

    if bin_edges is None:
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(ref, quantiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([ref.min() - 1e-3, ref.max() + 1e-3])

    edges = np.copy(bin_edges)
    edges[0] = -np.inf
    edges[-1] = np.inf

    p_counts, _ = np.histogram(ref, bins=edges)
    q_counts, _ = np.histogram(curr, bins=edges)

    p = np.maximum(p_counts / max(len(ref), 1), eps)
    q = np.maximum(q_counts / max(len(curr), 1), eps)
    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)
    kl_p_m = np.sum(p * np.log2(p / m))
    kl_q_m = np.sum(q * np.log2(q / m))
    js_val = 0.5 * (kl_p_m + kl_q_m)
    return float(np.clip(np.sqrt(max(0.0, js_val)), 0.0, 1.0))


def calculate_wasserstein_distance(
    ref_values: Union[np.ndarray, List[float], pd.Series],
    curr_values: Union[np.ndarray, List[float], pd.Series]
) -> float:
    """
    Computes first Wasserstein distance (Earth Mover's Distance) between two 1D distributions.
    """
    ref = np.asarray(ref_values, dtype=float)
    curr = np.asarray(curr_values, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0

    return float(stats.wasserstein_distance(ref, curr))


def calculate_categorical_psi(
    ref_counts: Dict[Any, int],
    curr_counts: Dict[Any, int],
    eps: float = 1e-4
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes PSI for discrete/categorical variables based on category frequency tables.
    """
    all_categories = sorted(list(set(ref_counts.keys()).union(set(curr_counts.keys()))))
    total_ref = max(sum(ref_counts.values()), 1)
    total_curr = max(sum(curr_counts.values()), 1)

    ref_pcts = []
    curr_pcts = []

    for cat in all_categories:
        ref_pct = max(ref_counts.get(cat, 0) / total_ref, eps)
        curr_pct = max(curr_counts.get(cat, 0) / total_curr, eps)
        ref_pcts.append(ref_pct)
        curr_pcts.append(curr_pct)

    ref_pcts = np.array(ref_pcts) / sum(ref_pcts)
    curr_pcts = np.array(curr_pcts) / sum(curr_pcts)

    psi_vec = (curr_pcts - ref_pcts) * np.log(curr_pcts / ref_pcts)
    psi_val = float(np.sum(psi_vec))

    details = {
        "categories": [str(c) for c in all_categories],
        "ref_proportions": ref_pcts.tolist(),
        "curr_proportions": curr_pcts.tolist(),
        "psi_components": psi_vec.tolist()
    }
    return max(0.0, psi_val), details
