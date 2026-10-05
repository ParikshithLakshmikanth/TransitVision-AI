"""Unit tests for Phase 8 Statistical Drift Detectors & Algorithms."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from monitoring.statistical_tests import (
    calculate_psi,
    calculate_ks_test,
    calculate_js_divergence,
    calculate_wasserstein_distance,
    calculate_categorical_psi
)
from monitoring.drift_models import DriftConfig, DriftSeverity, DriftType
from monitoring.drift_detectors import ReferenceProfiler, FeatureDriftDetector
from monitoring.performance_monitor import RollingPerformanceMonitor
from monitoring.residual_monitor import PageHinkleyDetector, ADWINDetector, ResidualConceptMonitor


def test_psi_identical_distributions():
    """Verify PSI between identical distributions is zero or near zero."""
    np.random.seed(42)
    dist1 = np.random.normal(100, 15, size=5000)
    dist2 = np.copy(dist1)
    
    psi_val, bin_edges, details = calculate_psi(dist1, dist2, num_bins=10)
    assert psi_val < 0.01
    assert len(bin_edges) > 1


def test_psi_detects_distribution_shift():
    """Verify PSI strongly reacts to significant mean and variance shift."""
    np.random.seed(42)
    ref = np.random.normal(100, 15, size=5000)
    shifted = np.random.normal(140, 25, size=5000)
    
    psi_val, _, _ = calculate_psi(ref, shifted, num_bins=10)
    assert psi_val > 0.25  # High drift


def test_ks_test_detects_shift():
    """Verify KS 2-sample test identifies shifted distribution with high significance."""
    np.random.seed(42)
    ref = np.random.normal(100, 15, size=1000)
    curr_same = np.random.normal(100, 15, size=1000)
    curr_diff = np.random.normal(120, 15, size=1000)

    stat_same, p_same = calculate_ks_test(ref, curr_same)
    stat_diff, p_diff = calculate_ks_test(ref, curr_diff)

    assert p_same > 0.01
    assert p_diff < 1e-10
    assert stat_diff > stat_same


def test_js_divergence_bounded():
    """Verify Jensen-Shannon divergence is bounded in [0, 1]."""
    np.random.seed(42)
    ref = np.random.normal(0, 1, size=1000)
    shifted = np.random.normal(5, 1, size=1000)

    js_val = calculate_js_divergence(ref, shifted)
    assert 0.0 <= js_val <= 1.0
    assert js_val > 0.5


def test_categorical_psi():
    """Verify categorical frequency PSI detects proportion shift."""
    ref_counts = {"A": 500, "B": 300, "C": 200}
    same_counts = {"A": 250, "B": 150, "C": 100}
    shifted_counts = {"A": 50, "B": 100, "C": 850}

    psi_same, _ = calculate_categorical_psi(ref_counts, same_counts)
    psi_shift, _ = calculate_categorical_psi(ref_counts, shifted_counts)

    assert psi_same < 0.01
    assert psi_shift > 0.30


def test_reference_profiler_generation(tmp_path):
    """Verify ReferenceProfiler generates valid reproducible profile JSON."""
    profiler = ReferenceProfiler(output_path=tmp_path / "test_profile.json")
    profile = profiler.build_profile(force_rebuild=True)

    assert "metadata" in profile
    assert "features" in profile
    assert profile["metadata"]["total_samples"] > 100000
    assert "precipitation" in profile["features"]
    assert profile["features"]["precipitation"]["feature_type"] == "numerical"


def test_rolling_performance_monitor_degradation():
    """Verify RollingPerformanceMonitor flags degradation when errors inflate."""
    perf_mon = RollingPerformanceMonitor(
        config=DriftConfig(min_window_size=50, window_size=100, perf_drift_multiplier=1.5),
        reference_mae=40.0,
        reference_rmse=60.0
    )

    # 1. Feed baseline quality predictions
    for i in range(100):
        perf_mon.add_observation(y_true=100.0, y_pred=100.0 + np.random.normal(0, 20), record_idx=i)

    res_base = perf_mon.evaluate_current_window()
    assert res_base is not None
    assert res_base.is_drift is False
    assert res_base.severity == DriftSeverity.NONE

    # 2. Feed degraded predictions
    for i in range(100, 200):
        perf_mon.add_observation(y_true=250.0, y_pred=100.0, record_idx=i)  # 150s error

    res_degraded = perf_mon.evaluate_current_window()
    assert res_degraded is not None
    assert res_degraded.is_drift is True
    assert res_degraded.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]


def test_page_hinkley_detector_concept_drift():
    """Verify Page-Hinkley cumulative sum detector fires on sudden residual shift."""
    ph = PageHinkleyDetector(reference_mae=5.0, reference_std=3.0, delta=0.2, threshold_lambda=20.0)

    # In-control errors (residuals close to 5.0)
    for i in range(100):
        drift, stat, sev = ph.update(residual_error=float(np.random.normal(5, 2)), sample_idx=i)
        assert drift is False

    # Out-of-control sudden concept shift (residuals jump to 60.0)
    drift_detected = False
    for i in range(100, 200):
        drift, stat, sev = ph.update(residual_error=60.0, sample_idx=i)
        if drift:
            drift_detected = True
            break

    assert drift_detected is True
    assert ph.last_drift_index is not None

