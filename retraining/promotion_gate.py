"""TransitVision AI - Multi-Tier Promotion Gate.
Implements rigorous, deterministic promotion gating comparing candidate challenger
against active production champion across dual holdout datasets.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import logging

from retraining.validator import EvaluationMetrics, DualHoldoutValidator

logger = logging.getLogger("TransitVision.PromotionGate")

# Configurable Approved Baseline Promotion Thresholds
DEFAULT_MAX_ORIGINAL_MAE_DEGRADATION_RATIO = 1.01  # Gate A: Candidate MAE <= Champion MAE * 1.01
DEFAULT_MAX_ORIGINAL_R2_DROP = 0.01                # Gate A: Candidate R2 >= Champion R2 - 0.01
DEFAULT_MAX_RECENT_MAE_DEGRADATION_RATIO = 1.00    # Gate B: Candidate MAE <= Champion MAE (Adaptation)
DEFAULT_MAX_RMSE_DEGRADATION_RATIO = 1.05          # Gate C: RMSE <= Champion RMSE * 1.05
DEFAULT_MAX_P95_DEGRADATION_RATIO = 1.08           # Gate C: P95 <= Champion P95 * 1.08
DEFAULT_MAX_ALLOWED_NON_POSITIVE_PCT = 0.0         # Gate C: Zero non-positive predictions
DEFAULT_MAX_ALLOWED_EXCESSIVE_COUNT = 0            # Gate C: Zero excessive predictions (>3600s)
DEFAULT_MAX_LATENCY_PER_SAMPLE_MS = 5.0            # Gate C: Latency SLA <= 5.0ms / sample


@dataclass
class PromotionGateResult:
    """Detailed outcome of the multi-tier promotion gate evaluation."""
    passed: bool
    gate_a_passed: bool
    gate_b_passed: bool
    gate_c_passed: bool
    champion_model_id: str
    candidate_model_id: str
    champion_original_metrics: Dict[str, Any]
    candidate_original_metrics: Dict[str, Any]
    champion_recent_metrics: Dict[str, Any]
    candidate_recent_metrics: Dict[str, Any]
    applied_thresholds: Dict[str, Any]
    failure_reasons: List[str] = field(default_factory=list)
    comparison_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "gate_a_passed": self.gate_a_passed,
            "gate_b_passed": self.gate_b_passed,
            "gate_c_passed": self.gate_c_passed,
            "champion_model_id": self.champion_model_id,
            "candidate_model_id": self.candidate_model_id,
            "champion_original_metrics": self.champion_original_metrics,
            "candidate_original_metrics": self.candidate_original_metrics,
            "champion_recent_metrics": self.champion_recent_metrics,
            "candidate_recent_metrics": self.candidate_recent_metrics,
            "applied_thresholds": self.applied_thresholds,
            "failure_reasons": self.failure_reasons,
            "comparison_summary": self.comparison_summary,
        }


class PromotionGate:
    """
    Evaluates candidate challenger vs active champion across dual holdouts and safety rules.
    """

    def __init__(
        self,
        max_original_mae_degradation_ratio: float = DEFAULT_MAX_ORIGINAL_MAE_DEGRADATION_RATIO,
        max_original_r2_drop: float = DEFAULT_MAX_ORIGINAL_R2_DROP,
        max_recent_mae_degradation_ratio: float = DEFAULT_MAX_RECENT_MAE_DEGRADATION_RATIO,
        max_rmse_degradation_ratio: float = DEFAULT_MAX_RMSE_DEGRADATION_RATIO,
        max_p95_degradation_ratio: float = DEFAULT_MAX_P95_DEGRADATION_RATIO,
        max_allowed_non_positive_pct: float = DEFAULT_MAX_ALLOWED_NON_POSITIVE_PCT,
        max_allowed_excessive_count: int = DEFAULT_MAX_ALLOWED_EXCESSIVE_COUNT,
        max_latency_per_sample_ms: float = DEFAULT_MAX_LATENCY_PER_SAMPLE_MS,
    ):
        self.max_original_mae_degradation_ratio = max_original_mae_degradation_ratio
        self.max_original_r2_drop = max_original_r2_drop
        self.max_recent_mae_degradation_ratio = max_recent_mae_degradation_ratio
        self.max_rmse_degradation_ratio = max_rmse_degradation_ratio
        self.max_p95_degradation_ratio = max_p95_degradation_ratio
        self.max_allowed_non_positive_pct = max_allowed_non_positive_pct
        self.max_allowed_excessive_count = max_allowed_excessive_count
        self.max_latency_per_sample_ms = max_latency_per_sample_ms
        self.evaluator = DualHoldoutValidator()

    def get_configured_thresholds(self) -> Dict[str, Any]:
        """Returns dictionary of currently configured promotion gate thresholds."""
        return {
            "max_original_mae_degradation_ratio": self.max_original_mae_degradation_ratio,
            "max_original_r2_drop": self.max_original_r2_drop,
            "max_recent_mae_degradation_ratio": self.max_recent_mae_degradation_ratio,
            "max_rmse_degradation_ratio": self.max_rmse_degradation_ratio,
            "max_p95_degradation_ratio": self.max_p95_degradation_ratio,
            "max_allowed_non_positive_pct": self.max_allowed_non_positive_pct,
            "max_allowed_excessive_count": self.max_allowed_excessive_count,
            "max_latency_per_sample_ms": self.max_latency_per_sample_ms,
        }

    def evaluate_candidate(
        self,
        champion_model,
        champion_preprocessor,
        candidate_model,
        candidate_preprocessor,
        original_holdout_df,
        recent_holdout_df,
        champion_model_id: str = "model_lightgbm_v1",
        candidate_model_id: str = "model_lightgbm_v1.1.0",
    ) -> PromotionGateResult:
        """
        Executes dual-holdout evaluation and applies multi-tier gating rules.
        """
        failure_reasons: List[str] = []
        thresholds = self.get_configured_thresholds()

        # 1. Evaluate Gate A: Original Holdout (Generalization Preservation)
        champ_orig_metrics, _ = self.evaluator.evaluate_model(
            champion_model, champion_preprocessor, original_holdout_df, dataset_name="Original Holdout (Generalization)"
        )
        cand_orig_metrics, _ = self.evaluator.evaluate_model(
            candidate_model, candidate_preprocessor, original_holdout_df, dataset_name="Original Holdout (Generalization)"
        )

        orig_mae_ratio = cand_orig_metrics.mae_sec / champ_orig_metrics.mae_sec if champ_orig_metrics.mae_sec > 0 else 1.0
        r2_drop = champ_orig_metrics.r2 - cand_orig_metrics.r2

        gate_a_mae_ok = orig_mae_ratio <= self.max_original_mae_degradation_ratio
        gate_a_r2_ok = r2_drop <= self.max_original_r2_drop
        gate_a_passed = gate_a_mae_ok and gate_a_r2_ok

        if not gate_a_mae_ok:
            failure_reasons.append(
                f"Gate A (Generalization) Failed: Original Holdout MAE degraded by {((orig_mae_ratio - 1.0)*100):.2f}% "
                f"({cand_orig_metrics.mae_sec:.2f}s vs champion {champ_orig_metrics.mae_sec:.2f}s, threshold <= {self.max_original_mae_degradation_ratio:.2f}x)."
            )
        if not gate_a_r2_ok:
            failure_reasons.append(
                f"Gate A (Generalization) Failed: R2 dropped by {r2_drop:.4f} "
                f"({cand_orig_metrics.r2:.4f} vs champion {champ_orig_metrics.r2:.4f}, max allowed drop <= {self.max_original_r2_drop:.2f})."
            )

        # 2. Evaluate Gate B: Recent Holdout (Operating Regime Adaptation)
        champ_rec_metrics, _ = self.evaluator.evaluate_model(
            champion_model, champion_preprocessor, recent_holdout_df, dataset_name="Recent Holdout (Adaptation)"
        )
        cand_rec_metrics, _ = self.evaluator.evaluate_model(
            candidate_model, candidate_preprocessor, recent_holdout_df, dataset_name="Recent Holdout (Adaptation)"
        )

        rec_mae_ratio = cand_rec_metrics.mae_sec / champ_rec_metrics.mae_sec if champ_rec_metrics.mae_sec > 0 else 1.0
        gate_b_passed = rec_mae_ratio <= self.max_recent_mae_degradation_ratio

        if not gate_b_passed:
            failure_reasons.append(
                f"Gate B (Adaptation) Failed: Recent Holdout MAE did not improve or maintain champion performance "
                f"({cand_rec_metrics.mae_sec:.2f}s vs champion {champ_rec_metrics.mae_sec:.2f}s, ratio {rec_mae_ratio:.4f}x > threshold {self.max_recent_mae_degradation_ratio:.2f}x)."
            )

        # 3. Evaluate Gate C: Safety, Tail Risk & Latency SLA
        gate_c_failures: List[str] = []
        orig_rmse_ratio = cand_orig_metrics.rmse_sec / champ_orig_metrics.rmse_sec if champ_orig_metrics.rmse_sec > 0 else 1.0
        if orig_rmse_ratio > self.max_rmse_degradation_ratio:
            gate_c_failures.append(f"RMSE ratio ({orig_rmse_ratio:.4f}x) exceeded threshold ({self.max_rmse_degradation_ratio:.2f}x).")

        orig_p95_ratio = cand_orig_metrics.p95_error_sec / champ_orig_metrics.p95_error_sec if champ_orig_metrics.p95_error_sec > 0 else 1.0
        if orig_p95_ratio > self.max_p95_degradation_ratio:
            gate_c_failures.append(f"P95 error ratio ({orig_p95_ratio:.4f}x) exceeded threshold ({self.max_p95_degradation_ratio:.2f}x).")

        if cand_orig_metrics.non_positive_predictions_pct > self.max_allowed_non_positive_pct:
            gate_c_failures.append(f"Non-positive predictions ({cand_orig_metrics.non_positive_predictions_pct:.2f}%) exceeded limit ({self.max_allowed_non_positive_pct}%).")

        if cand_orig_metrics.excessive_predictions_count > self.max_allowed_excessive_count:
            gate_c_failures.append(f"Excessive predictions >3600s count ({cand_orig_metrics.excessive_predictions_count}) exceeded limit ({self.max_allowed_excessive_count}).")

        cand_sample_count = cand_orig_metrics.samples
        if cand_sample_count > 0:
            lat_per_sample = cand_orig_metrics.inference_latency_ms / cand_sample_count
            if lat_per_sample > self.max_latency_per_sample_ms:
                gate_c_failures.append(f"Inference latency SLA exceeded: {lat_per_sample:.3f}ms/sample > {self.max_latency_per_sample_ms}ms/sample.")

        gate_c_passed = len(gate_c_failures) == 0
        if not gate_c_passed:
            failure_reasons.extend([f"Gate C (Safety) Failed: {f}" for f in gate_c_failures])

        overall_passed = gate_a_passed and gate_b_passed and gate_c_passed

        summary = {
            "original_mae_delta_sec": round(cand_orig_metrics.mae_sec - champ_orig_metrics.mae_sec, 2),
            "original_mae_ratio": round(orig_mae_ratio, 4),
            "original_r2_drop": round(r2_drop, 4),
            "recent_mae_delta_sec": round(cand_rec_metrics.mae_sec - champ_rec_metrics.mae_sec, 2),
            "recent_mae_ratio": round(rec_mae_ratio, 4),
            "original_rmse_ratio": round(orig_rmse_ratio, 4),
            "original_p95_ratio": round(orig_p95_ratio, 4),
        }

        logger.info(
            f"Promotion Gate Decision for {candidate_model_id}: {'PASS' if overall_passed else 'FAIL'} "
            f"(Gate A: {'PASS' if gate_a_passed else 'FAIL'}, Gate B: {'PASS' if gate_b_passed else 'FAIL'}, Gate C: {'PASS' if gate_c_passed else 'FAIL'})"
        )

        return PromotionGateResult(
            passed=overall_passed,
            gate_a_passed=gate_a_passed,
            gate_b_passed=gate_b_passed,
            gate_c_passed=gate_c_passed,
            champion_model_id=champion_model_id,
            candidate_model_id=candidate_model_id,
            champion_original_metrics=champ_orig_metrics.to_dict(),
            candidate_original_metrics=cand_orig_metrics.to_dict(),
            champion_recent_metrics=champ_rec_metrics.to_dict(),
            candidate_recent_metrics=cand_rec_metrics.to_dict(),
            applied_thresholds=thresholds,
            failure_reasons=failure_reasons,
            comparison_summary=summary,
        )
