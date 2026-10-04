"""TransitVision AI - Feature Leakage Guard.
Strictly inspects feature schemas and input matrices before model training/inference
to prevent future-information, label leakage, and missing feature errors.
"""
import logging
from typing import List, Set, Optional, Union
import pandas as pd

logger = logging.getLogger("TransitVision.LeakGuard")

# Forbidden target and future-looking column identifiers
FORBIDDEN_COLUMNS: Set[str] = {
    # Target
    "eta_to_next_stop_sec",
    "run_time_in_seconds",
    # Future segment / trip outcomes
    "future_segment_run_time",
    "future_dwell_time",
    "future_arrival_time",
    "future_timestamp",
    "actual_eta_sec",
    "target",
    "y"
}

FORBIDDEN_SUBSTRINGS: List[str] = [
    "future",
    "next_segment_run",
    "next_dwell",
    "next_arrival",
    "actual_run_time"
]


class LeakageGuardError(Exception):
    """Raised when data leakage or forbidden feature is detected in model inputs."""
    pass


class LeakageGuard:
    """Automated zero-leakage validator."""

    def __init__(self, forbidden_columns: Optional[Set[str]] = None):
        self.forbidden_columns = {c.lower().strip() for c in (forbidden_columns or FORBIDDEN_COLUMNS)}
        self.forbidden_substrings = FORBIDDEN_SUBSTRINGS

    def validate_schema(self, expected_features: List[str], dataset_name: str = "Inference Schema") -> bool:
        """
        Validates model feature schema at initialization time.
        Emits one INFO log message upon verification.
        """
        violations = []
        for col in expected_features:
            col_lower = str(col).lower().strip()
            if col_lower in self.forbidden_columns:
                violations.append(f"Forbidden column in schema: '{col}'")
            for sub in self.forbidden_substrings:
                if sub in col_lower:
                    violations.append(f"Forbidden substring '{sub}' in schema column: '{col}'")

        if violations:
            msg = f"[CRITICAL LEAKAGE DETECTED in {dataset_name}]:\n" + "\n".join(violations)
            logger.error(msg)
            raise LeakageGuardError(msg)

        logger.info(f"LeakageGuard schema validation PASSED ({len(expected_features)} inference features).")
        return True

    def validate_features(
        self,
        X: Union[pd.DataFrame, dict, list],
        dataset_name: str = "Feature Matrix",
        log_success: bool = False,
        required_features: Optional[List[str]] = None
    ) -> bool:
        """
        Fast per-request validation:
        1. Ensures no target or future-looking columns are present.
        2. Ensures all required features are present (if required_features specified).
        3. Raises LeakageGuardError or ValueError on violations.
        4. Logs only on violations (or DEBUG / when log_success=True).
        """
        if isinstance(X, dict):
            cols = set(X.keys())
        elif isinstance(X, pd.DataFrame):
            cols = set(X.columns)
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            cols = set(X[0].keys())
        else:
            raise TypeError(f"X must be a pandas DataFrame, dict, or list of dicts, got {type(X)}")

        violations = []

        # 1. Check for forbidden columns
        for col in cols:
            col_lower = str(col).lower().strip()
            if col_lower in self.forbidden_columns:
                violations.append(f"Forbidden column detected: '{col}'")
            for sub in self.forbidden_substrings:
                if sub in col_lower:
                    violations.append(f"Forbidden substring '{sub}' detected in column: '{col}'")

        if violations:
            msg = f"[CRITICAL LEAKAGE DETECTED in {dataset_name}]:\n" + "\n".join(violations)
            logger.error(msg)
            raise LeakageGuardError(msg)

        # 2. Check for missing required features
        if required_features is not None:
            missing = [req for req in required_features if req not in cols]
            if missing:
                msg = f"[MISSING REQUIRED FEATURES in {dataset_name}]: Missing {len(missing)} features: {missing[:5]}"
                logger.error(msg)
                raise ValueError(msg)

        if log_success:
            logger.info(f"LeakageGuard PASSED for {dataset_name} ({len(cols)} columns verified).")
        else:
            logger.debug(f"LeakageGuard PASSED for {dataset_name} ({len(cols)} columns verified).")

        return True
