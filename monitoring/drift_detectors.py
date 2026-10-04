"""TransitVision AI - Feature / Covariate Drift Detectors.
Builds and maintains the training reference profile and evaluates
streaming feature distributions using PSI, KS-test with FDR control, and Jensen-Shannon Divergence.
Distinguishes Kinematic/Operational, Environmental, and Cyclical Context features.
"""
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config.settings import PROCESSED_DATA_DIR, METADATA_DIR
from ml.preprocessing.pipeline_preprocessor import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)
from monitoring.drift_models import (
    DriftType,
    DriftSeverity,
    FeatureDriftResult,
    DriftConfig
)
from monitoring.statistical_tests import (
    calculate_psi,
    calculate_ks_test,
    calculate_js_divergence,
    calculate_categorical_psi,
    benjamini_hochberg_fdr
)

logger = logging.getLogger("TransitVision.DriftDetectors")

# 1. Operational & Kinematic telemetry features (Primary continuous distribution shift signals)
KINEMATIC_OPERATIONAL_FEATURES = {
    "segment_delay_ratio",
    "previous_segment_run_time",
    "rolling_prev_segment_mean",
    "rolling_prev_segment_std",
    "cumulative_trip_time_sec",
    "historical_segment_time_mean",
    "congestion_proxy"
}

# 2. Environmental & Meteorological features (Weather degradation signals)
ENVIRONMENTAL_FEATURES = {
    "precipitation",
    "rain",
    "relative_humidity_2m",
    "temperature_2m",
    "wind_speed_10m",
    "weather_code"
}

# 3. Contextual Temporal & Topology features
CONTEXT_TOPOLOGY_FEATURES = {
    "hour", "minute", "day_of_week", "is_weekend", "is_peak_period",
    "sin_hour", "cos_hour", "sin_time_of_day", "cos_time_of_day",
    "deviceid", "direction", "segment", "segment_length_km",
    "segments_completed", "segments_remaining", "trip_progress_ratio"
}


class ReferenceProfiler:
    """Computes and serializes baseline statistical reference profiles from training data."""

    def __init__(self, training_path: Optional[Path] = None, output_path: Optional[Path] = None):
        self.training_path = Path(training_path or (PROCESSED_DATA_DIR / "kandy_eta_training.parquet"))
        self.output_path = Path(output_path or (METADATA_DIR / "drift_reference_profile.json"))

    def build_profile(self, num_bins: int = 10, force_rebuild: bool = False) -> Dict[str, Any]:
        """Builds reproducible reference distribution profile."""
        if self.output_path.exists() and not force_rebuild:
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                logger.info(f"Loaded existing drift reference profile from {self.output_path}")
                return profile
            except Exception as e:
                logger.warning(f"Failed loading existing profile, rebuilding: {e}")

        logger.info(f"Building reference drift profile from {self.training_path}...")
        df = pd.read_parquet(self.training_path)
        with open(self.training_path, "rb") as f:
            data_hash = hashlib.sha256(f.read()).hexdigest()

        profile: Dict[str, Any] = {
            "metadata": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_path": str(self.training_path.name),
                "dataset_sha256": data_hash,
                "total_samples": len(df),
                "model_version": "v1.0.0",
                "methodology": "Empirical Quantile Histogram Profiling with Zero-Inflation Awareness",
                "feature_count": len(ALL_FEATURE_COLUMNS)
            },
            "features": {}
        }

        for col in ALL_FEATURE_COLUMNS:
            if col not in df.columns:
                continue
            
            series = df[col]
            missing_count = int(series.isna().sum())
            missing_rate = float(missing_count / max(len(series), 1))

            if col in CATEGORICAL_FEATURES:
                val_counts = series.value_counts(dropna=True).to_dict()
                freq_dict = {str(k): int(v) for k, v in val_counts.items()}
                profile["features"][col] = {
                    "feature_name": col,
                    "feature_type": "categorical",
                    "sample_count": len(series) - missing_count,
                    "missing_count": missing_count,
                    "missing_rate": missing_rate,
                    "unique_values": int(series.nunique()),
                    "category_frequencies": freq_dict
                }
            else:
                clean_vals = series.dropna().astype(float).values
                min_val = float(np.min(clean_vals)) if len(clean_vals) > 0 else 0.0
                zero_mask = (clean_vals == min_val)
                if zero_mask.mean() >= 0.25:
                    pos_vals = clean_vals[~zero_mask]
                    if len(pos_vals) > 0:
                        pos_edges = np.percentile(pos_vals, np.linspace(0, 100, num_bins))
                        bin_edges = np.unique(np.concatenate([[min_val - 1e-4, min_val + 1e-4], pos_edges]))
                    else:
                        bin_edges = np.array([min_val - 1.0, min_val + 1.0])
                else:
                    quantiles = np.linspace(0, 100, num_bins + 1)
                    bin_edges = np.percentile(clean_vals, quantiles)
                    bin_edges = np.unique(bin_edges)
                    if len(bin_edges) < 2:
                        bin_edges = np.array([min_val - 1e-3, float(np.max(clean_vals)) + 1e-3])

                profile["features"][col] = {
                    "feature_name": col,
                    "feature_type": "numerical",
                    "sample_count": len(clean_vals),
                    "missing_count": missing_count,
                    "missing_rate": missing_rate,
                    "mean": float(np.mean(clean_vals)),
                    "std": float(np.std(clean_vals)),
                    "min": float(np.min(clean_vals)),
                    "q25": float(np.percentile(clean_vals, 25)),
                    "median": float(np.median(clean_vals)),
                    "q75": float(np.percentile(clean_vals, 75)),
                    "max": float(np.max(clean_vals)),
                    "bin_edges": bin_edges.tolist()
                }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

        logger.info(f"Reference profile successfully serialized to {self.output_path}")
        return profile


class FeatureDriftDetector:
    """Evaluates covariate drift on streaming feature windows against ReferenceProfile."""

    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        reference_profile: Optional[Dict[str, Any]] = None,
        reference_data_path: Optional[Path] = None
    ):
        self.config = config or DriftConfig()
        if reference_profile is not None:
            self.ref_profile = reference_profile
        else:
            profiler = ReferenceProfiler(training_path=reference_data_path)
            self.ref_profile = profiler.build_profile()

        # Load reference raw dataframe in memory if available for KS test
        self.ref_df: Optional[pd.DataFrame] = None
        ref_path = Path(reference_data_path or (PROCESSED_DATA_DIR / "kandy_eta_training.parquet"))
        if ref_path.exists():
            try:
                self.ref_df = pd.read_parquet(ref_path)
            except Exception as e:
                logger.warning(f"Could not load reference parquet for KS test: {e}")

    def evaluate_window(self, current_features_df: pd.DataFrame) -> List[FeatureDriftResult]:
        """
        Evaluates drift for all 29 features on the provided window dataframe.
        Applies Benjamini-Hochberg False Discovery Rate (FDR) correction across tests.
        """
        results: List[FeatureDriftResult] = []
        if current_features_df.empty:
            return results

        n_curr = len(current_features_df)
        raw_pvals: List[float] = []
        num_cols: List[str] = []

        # First pass: compute statistics
        temp_stats: Dict[str, Dict[str, Any]] = {}

        for col in ALL_FEATURE_COLUMNS:
            if col not in current_features_df.columns or col not in self.ref_profile.get("features", {}):
                continue

            ref_meta = self.ref_profile["features"][col]
            feat_type = ref_meta["feature_type"]
            curr_series = current_features_df[col]

            if feat_type == "numerical":
                bin_edges = np.array(ref_meta.get("bin_edges", []))
                curr_vals = curr_series.dropna().astype(float).values
                
                if self.ref_df is not None and col in self.ref_df.columns:
                    ref_vals = self.ref_df[col].dropna().astype(float).values
                    psi_val, _, _ = calculate_psi(ref_vals, curr_vals, bin_edges=bin_edges)
                    ks_stat, p_val = calculate_ks_test(ref_vals, curr_vals)
                    js_div = calculate_js_divergence(ref_vals, curr_vals, bin_edges=bin_edges)
                else:
                    ref_vals = np.array([ref_meta["min"], ref_meta["q25"], ref_meta["median"], ref_meta["q75"], ref_meta["max"]])
                    psi_val, _, _ = calculate_psi(ref_vals, curr_vals, bin_edges=bin_edges)
                    ks_stat, p_val = 0.0, 1.0
                    js_div = 0.0

                num_cols.append(col)
                raw_pvals.append(p_val)
                temp_stats[col] = {
                    "psi": psi_val,
                    "ks_stat": ks_stat,
                    "raw_p_val": p_val,
                    "js_div": js_div,
                    "curr_mean": float(np.mean(curr_vals)) if len(curr_vals) > 0 else 0.0,
                    "curr_vals": curr_vals,
                    "ref_mean": float(ref_meta.get("mean", 0.0)),
                    "ref_std": float(ref_meta.get("std", 1.0)),
                    "ref_sample_count": ref_meta.get("sample_count", 0),
                    "feat_type": "numerical"
                }

            elif feat_type == "categorical":
                ref_freqs = {k: int(v) for k, v in ref_meta.get("category_frequencies", {}).items()}
                curr_freqs = {str(k): int(v) for k, v in curr_series.value_counts(dropna=True).to_dict().items()}
                psi_cat, cat_details = calculate_categorical_psi(ref_freqs, curr_freqs)
                temp_stats[col] = {
                    "psi": psi_cat,
                    "ref_sample_count": ref_meta.get("sample_count", 0),
                    "cat_details": cat_details,
                    "feat_type": "categorical"
                }

        # Apply Benjamini-Hochberg FDR correction across numerical features
        fdr_rejected, adj_pvals = benjamini_hochberg_fdr(raw_pvals, alpha=self.config.ks_alpha)
        fdr_dict = {num_cols[i]: (fdr_rejected[i], adj_pvals[i]) for i in range(len(num_cols))}

        # Second pass: classify severity with calibrated thresholds
        for col, s in temp_stats.items():
            if s["feat_type"] == "numerical":
                psi_val = s["psi"]
                ks_stat = s["ks_stat"]
                is_fdr_sig, adj_pval = fdr_dict.get(col, (False, 1.0))
                curr_mean = s["curr_mean"]
                curr_vals = s["curr_vals"]
                ref_mean = s["ref_mean"]
                ref_std = s["ref_std"]

                is_drift = False
                severity = DriftSeverity.NONE

                # Category 1: Kinematic & Operational Features
                if col in KINEMATIC_OPERATIONAL_FEATURES:
                    if psi_val >= self.config.psi_critical_threshold or (psi_val >= self.config.psi_drift_threshold and ks_stat >= self.config.ks_critical_statistic and is_fdr_sig):
                        is_drift = True
                        severity = DriftSeverity.CRITICAL if psi_val >= self.config.psi_critical_threshold else DriftSeverity.HIGH
                    elif psi_val >= self.config.psi_drift_threshold or (ks_stat >= self.config.ks_critical_statistic and is_fdr_sig):
                        is_drift = True
                        severity = DriftSeverity.MEDIUM
                    elif psi_val >= self.config.psi_warning_threshold:
                        severity = DriftSeverity.LOW

                # Category 2: Environmental & Meteorological Features
                elif col in ENVIRONMENTAL_FEATURES:
                    if col in ["precipitation", "rain"]:
                        # Severe simulated downpour: >= 10 mm/h (intensity 0.80 -> 12.8 mm/h)
                        if curr_mean >= 10.0:
                            is_drift = True
                            severity = DriftSeverity.CRITICAL
                        elif curr_mean >= 5.0:
                            is_drift = True
                            severity = DriftSeverity.HIGH
                        elif curr_mean >= 3.0:
                            is_drift = True
                            severity = DriftSeverity.MEDIUM
                    elif col == "relative_humidity_2m":
                        if curr_mean >= 95.0:
                            is_drift = True
                            severity = DriftSeverity.HIGH
                        elif curr_mean >= 92.0 and psi_val >= 0.25:
                            is_drift = True
                            severity = DriftSeverity.MEDIUM
                    elif col == "weather_code":
                        heavy_rain_frac = float(np.mean(curr_vals >= 63))
                        if heavy_rain_frac >= 0.5:
                            is_drift = True
                            severity = DriftSeverity.HIGH
                        elif heavy_rain_frac >= 0.25:
                            is_drift = True
                            severity = DriftSeverity.MEDIUM
                    else:
                        z_score = abs(curr_mean - ref_mean) / max(ref_std, 0.1)
                        if z_score >= 4.0:
                            is_drift = True
                            severity = DriftSeverity.HIGH
                        elif z_score >= 3.0:
                            severity = DriftSeverity.LOW

                # Category 3: Contextual / Topology
                else:
                    if col == "is_peak_period" and curr_mean >= 0.8:
                        severity = DriftSeverity.LOW

                res = FeatureDriftResult(
                    feature_name=col,
                    feature_type="numerical",
                    detector_name="PSI_KS_FDR_HYBRID",
                    statistic=round(psi_val, 4),
                    threshold=self.config.psi_drift_threshold,
                    p_value=round(adj_pval, 6),
                    is_drift=is_drift,
                    severity=severity,
                    sample_count=n_curr,
                    ref_sample_count=s["ref_sample_count"],
                    metadata={
                        "category": "kinematic" if col in KINEMATIC_OPERATIONAL_FEATURES else ("environmental" if col in ENVIRONMENTAL_FEATURES else "context"),
                        "ks_statistic": round(ks_stat, 4),
                        "ks_adj_p_value": round(adj_pval, 6),
                        "fdr_significant": is_fdr_sig,
                        "js_divergence": round(s["js_div"], 4),
                        "curr_mean": round(curr_mean, 4),
                        "ref_mean": round(ref_mean, 4)
                    }
                )
                results.append(res)

            elif s["feat_type"] == "categorical":
                psi_cat = s["psi"]
                is_drift = False
                severity = DriftSeverity.NONE

                if psi_cat >= 0.60:
                    severity = DriftSeverity.LOW

                res = FeatureDriftResult(
                    feature_name=col,
                    feature_type="categorical",
                    detector_name="CATEGORICAL_PSI",
                    statistic=round(psi_cat, 4),
                    threshold=self.config.categorical_psi_threshold,
                    p_value=None,
                    is_drift=is_drift,
                    severity=severity,
                    sample_count=n_curr,
                    ref_sample_count=s["ref_sample_count"],
                    metadata={
                        "category": "context",
                        "categories_evaluated": len(s["cat_details"].get("categories", []))
                    }
                )
                results.append(res)

        return results
