"""TransitVision AI - Drift Monitoring Models & Schemas.
Defines structured dataclasses and enums for Covariate, Performance,
and Residual/Concept Drift detection and logging.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone


class DriftType(str, Enum):
    """Categorization of detected distribution shift."""
    DATA_DRIFT = "DATA_DRIFT"              # P(X): Feature distribution shift
    PERFORMANCE_DRIFT = "PERFORMANCE_DRIFT"  # L(Y, \hat{Y}): Metric accuracy degradation
    CONCEPT_DRIFT = "CONCEPT_DRIFT"        # P(Y|X): Residual / conditional relationship shift


class DriftSeverity(str, Enum):
    """Severity classification of detected drift."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class FeatureDriftResult:
    """Statistical test outcome for an individual feature."""
    feature_name: str
    feature_type: str  # "numerical" or "categorical"
    detector_name: str  # "PSI", "KS_TEST", "JS_DIVERGENCE", "CATEGORICAL_PSI"
    statistic: float
    threshold: float
    p_value: Optional[float] = None
    is_drift: bool = False
    severity: DriftSeverity = DriftSeverity.NONE
    sample_count: int = 0
    ref_sample_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class PerformanceDriftResult:
    """Evaluation of rolling predictive accuracy against baseline reference."""
    window_id: int
    window_start_idx: int
    window_end_idx: int
    sample_count: int
    mae: float
    rmse: float
    median_ae: float
    p90_error: float
    p95_error: float
    mean_signed_error: float
    reference_mae: float
    reference_rmse: float
    mae_degradation_ratio: float
    rmse_degradation_ratio: float
    is_drift: bool = False
    severity: DriftSeverity = DriftSeverity.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class ResidualDriftResult:
    """Sequential concept drift test outcome (Page-Hinkley / ADWIN)."""
    detector_name: str  # "PageHinkley", "ADWIN"
    statistic: float
    threshold: float
    in_warning: bool = False
    in_drift: bool = False
    severity: DriftSeverity = DriftSeverity.NONE
    sample_count: int = 0
    mean_residual: float = 0.0
    variance_residual: float = 0.0
    detection_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class DriftEvent:
    """Unified alert event emitted when statistically significant drift is detected."""
    event_id: str
    timestamp_utc: str
    drift_type: DriftType
    severity: DriftSeverity
    detector: str
    scenario_id: str
    scenario_name: str
    synthetic_disturbance: bool
    disturbance_intensity: float
    source_type: str
    model_id: str
    model_version: str
    window_start_idx: int
    window_end_idx: int
    sample_count: int
    statistic: float
    threshold: float
    p_value: Optional[float]
    baseline_metric: float
    current_metric: float
    affected_features: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: str = "OPEN"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["drift_type"] = self.drift_type.value
        d["severity"] = self.severity.value
        return d


@dataclass
class DriftConfig:
    """Configuration parameters and thresholds for drift detection."""
    # Window settings
    window_size: int = 500
    step_size: int = 250
    min_window_size: int = 100
    
    # Data drift thresholds (calibrated against empirical null baseline)
    psi_warning_threshold: float = 0.15
    psi_drift_threshold: float = 0.25
    psi_critical_threshold: float = 0.45
    ks_alpha: float = 0.05
    ks_critical_statistic: float = 0.18
    js_drift_threshold: float = 0.20
    categorical_psi_threshold: float = 0.25

    # Performance drift thresholds (relative multiplier over reference MAE/RMSE)
    perf_warning_multiplier: float = 1.30   # 30% increase over reference
    perf_drift_multiplier: float = 1.60     # 60% increase over reference
    perf_critical_multiplier: float = 2.20  # 120% increase over reference

    # Sequential residual thresholds (Standardized Page-Hinkley)
    ph_delta: float = 0.25                  # Tolerance factor in standard deviations
    ph_lambda: float = 80.0                 # Cumulative sum decision threshold
    ph_alpha: float = 0.999                 # Forgetting/decay factor

    # Sequential residual thresholds (ADWIN)
    adwin_delta: float = 0.001              # Confidence parameter

    # Reference profile path
    reference_profile_path: str = "data/metadata/drift_reference_profile.json"
    reference_data_path: str = "data/processed/kandy_eta_training.parquet"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

