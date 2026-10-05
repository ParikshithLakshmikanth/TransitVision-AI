"""TransitVision AI - Unit Tests for Retraining Trigger Policy."""
import pytest
from retraining.trigger_policy import RetrainingTriggerPolicy, TriggerDecision


def test_trigger_policy_nominal_operation():
    """Verify nominal metrics do not trigger retraining."""
    policy = RetrainingTriggerPolicy()
    decision = policy.evaluate(
        drift_events=[],
        current_stream_index=2000,
        resolved_stream_buffer_size=1500,
        rolling_metrics_history=[{"samples": 500, "mae_sec": 38.5}, {"samples": 500, "mae_sec": 39.0}],
        baseline_mae=38.77,
    )
    assert not decision.should_retrain
    assert decision.severity == "NORMAL"


def test_trigger_policy_data_drift_advisory_only():
    """Verify data drift (PSI/KS) produces advisory alert only without triggering retraining."""
    policy = RetrainingTriggerPolicy()
    drift_events = [
        {"drift_type": "DATA_DRIFT", "feature": "cloud_cover_pct", "psi": 0.35, "severity": "WARNING"}
    ]
    decision = policy.evaluate(
        drift_events=drift_events,
        current_stream_index=2000,
        resolved_stream_buffer_size=1500,
        rolling_metrics_history=[{"samples": 500, "mae_sec": 40.0}],
        baseline_mae=38.77,
    )
    assert not decision.should_retrain
    assert decision.severity == "WARNING"
    assert "Data / Covariate drift detected" in decision.trigger_reason


def test_trigger_policy_isolated_spike_suppressed():
    """Verify single-window performance spike does not trigger retraining."""
    policy = RetrainingTriggerPolicy(sustained_window_count=2, mae_degradation_ratio=1.60)
    decision = policy.evaluate(
        drift_events=[],
        current_stream_index=2000,
        resolved_stream_buffer_size=1500,
        rolling_metrics_history=[
            {"samples": 500, "mae_sec": 65.0, "window_index": 1},  # 1.67x spike (isolated)
            {"samples": 500, "mae_sec": 42.0, "window_index": 2},  # Normal window
        ],
        baseline_mae=38.77,
    )
    assert not decision.should_retrain


def test_trigger_policy_sustained_performance_drift_triggers():
    """Verify >= 2 consecutive degraded windows trigger retraining."""
    policy = RetrainingTriggerPolicy(sustained_window_count=2, mae_degradation_ratio=1.60)
    decision = policy.evaluate(
        drift_events=[],
        current_stream_index=2500,
        resolved_stream_buffer_size=1500,
        rolling_metrics_history=[
            {"samples": 500, "mae_sec": 68.0, "window_index": 1},
            {"samples": 500, "mae_sec": 71.5, "window_index": 2},
        ],
        baseline_mae=38.77,
    )
    assert decision.should_retrain
    assert decision.severity == "CRITICAL"
    assert "Sustained Performance Drift" in decision.trigger_reason


def test_trigger_policy_emergency_concept_drift_triggers():
    """Verify >= 3 Page-Hinkley concept drift alarms trigger emergency retraining."""
    policy = RetrainingTriggerPolicy(concept_drift_alarm_threshold=3)
    drift_events = [
        {"drift_type": "CONCEPT_DRIFT", "message": "Page-Hinkley alarm 1"},
        {"drift_type": "CONCEPT_DRIFT", "message": "Page-Hinkley alarm 2"},
        {"drift_type": "CONCEPT_DRIFT", "message": "Page-Hinkley alarm 3"},
    ]
    decision = policy.evaluate(
        drift_events=drift_events,
        current_stream_index=2500,
        resolved_stream_buffer_size=1500,
        rolling_metrics_history=[],
        baseline_mae=38.77,
    )
    assert decision.should_retrain
    assert decision.severity == "CRITICAL"
    assert "Emergency Concept Drift" in decision.trigger_reason


def test_trigger_policy_cooldown_guardrail():
    """Verify retraining is suppressed if cooldown period is active."""
    policy = RetrainingTriggerPolicy(cooldown_records=1500)
    policy.record_retraining_execution(stream_index=1000)

    drift_events = [
        {"drift_type": "CONCEPT_DRIFT", "message": "Page-Hinkley alarm 1"},
        {"drift_type": "CONCEPT_DRIFT", "message": "Page-Hinkley alarm 2"},
        {"drift_type": "CONCEPT_DRIFT", "message": "Page-Hinkley alarm 3"},
    ]
    # Current index is 1800 (only 800 records elapsed < 1500)
    decision = policy.evaluate(
        drift_events=drift_events,
        current_stream_index=1800,
        resolved_stream_buffer_size=1500,
    )
    assert not decision.should_retrain
    assert decision.cooldown_active
    assert "Cooldown active" in decision.trigger_reason


def test_trigger_policy_minimum_data_guardrail():
    """Verify retraining is suppressed if resolved buffer has fewer than min_new_records."""
    policy = RetrainingTriggerPolicy(min_new_records=1000)
    drift_events = [
        {"drift_type": "CONCEPT_DRIFT", "message": "Page-Hinkley alarm 1"},
        {"drift_type": "CONCEPT_DRIFT", "message": "Page-Hinkley alarm 2"},
        {"drift_type": "CONCEPT_DRIFT", "message": "Page-Hinkley alarm 3"},
    ]
    decision = policy.evaluate(
        drift_events=drift_events,
        current_stream_index=2000,
        resolved_stream_buffer_size=500,  # 500 < 1000 required
    )
    assert not decision.should_retrain
    assert "Insufficient resolved stream buffer" in decision.trigger_reason


def test_trigger_policy_alarms_spread_far_apart_do_not_trigger():
    """Verify concept alarms spread sparsely over history do NOT trigger retraining."""
    policy = RetrainingTriggerPolicy(
        concept_drift_alarm_threshold=3,
        recent_concept_window_records=2500
    )
    # Alarms at 2,000, 6,000, 12,000 evaluated at stream index 15,000
    # None of them are within the recent 2,500-record window (12,500 - 15,000)
    drift_events = [
        {"drift_type": "CONCEPT_DRIFT", "event_id": "ALARM_1", "window_end_idx": 2000},
        {"drift_type": "CONCEPT_DRIFT", "event_id": "ALARM_2", "window_end_idx": 6000},
        {"drift_type": "CONCEPT_DRIFT", "event_id": "ALARM_3", "window_end_idx": 12000},
    ]
    decision = policy.evaluate(
        drift_events=drift_events,
        current_stream_index=15000,
        resolved_stream_buffer_size=15000,
    )
    assert not decision.should_retrain
    assert decision.severity == "NORMAL"


def test_trigger_policy_repeated_alarms_inside_recent_window_do_trigger():
    """Verify repeated concept alarms within recent window trigger emergency retraining."""
    policy = RetrainingTriggerPolicy(
        concept_drift_alarm_threshold=3,
        recent_concept_window_records=2500
    )
    # Alarms at 13,000, 14,000, 14,800 evaluated at stream index 15,000 (all within 2,500 records)
    drift_events = [
        {"drift_type": "CONCEPT_DRIFT", "event_id": "ALARM_1", "window_end_idx": 13000},
        {"drift_type": "CONCEPT_DRIFT", "event_id": "ALARM_2", "window_end_idx": 14000},
        {"drift_type": "CONCEPT_DRIFT", "event_id": "ALARM_3", "window_end_idx": 14800},
    ]
    decision = policy.evaluate(
        drift_events=drift_events,
        current_stream_index=15000,
        resolved_stream_buffer_size=15000,
    )
    assert decision.should_retrain
    assert decision.severity == "CRITICAL"
    assert "within recent 2500 records" in decision.trigger_reason


def test_trigger_policy_duplicate_evaluation_adjacent_windows_does_not_retrigger():
    """Verify subsequent evaluation on the same set of alarms does not duplicate trigger."""
    policy = RetrainingTriggerPolicy(
        concept_drift_alarm_threshold=3,
        recent_concept_window_records=2500
    )
    drift_events = [
        {"drift_type": "CONCEPT_DRIFT", "event_id": "ALARM_1", "window_end_idx": 13000},
        {"drift_type": "CONCEPT_DRIFT", "event_id": "ALARM_2", "window_end_idx": 14000},
        {"drift_type": "CONCEPT_DRIFT", "event_id": "ALARM_3", "window_end_idx": 14800},
    ]
    # First evaluation at 15000 -> triggers
    dec1 = policy.evaluate(
        drift_events=drift_events,
        current_stream_index=15000,
        resolved_stream_buffer_size=15000,
    )
    assert dec1.should_retrain

    # Second evaluation at 15500 with identical alarms -> suppressed / acknowledged
    dec2 = policy.evaluate(
        drift_events=drift_events,
        current_stream_index=15500,
        resolved_stream_buffer_size=15500,
    )
    assert not dec2.should_retrain
    assert "already evaluated" in dec2.trigger_reason


def test_trigger_policy_determinism():
    """Verify policy evaluation is 100% deterministic given identical inputs."""
    drift_events = [
        {"drift_type": "CONCEPT_DRIFT", "event_id": "ALARM_1", "window_end_idx": 4000},
        {"drift_type": "CONCEPT_DRIFT", "event_id": "ALARM_2", "window_end_idx": 4500},
    ]
    policy1 = RetrainingTriggerPolicy()
    policy2 = RetrainingTriggerPolicy()
    
    dec1 = policy1.evaluate(drift_events, 5000, 5000)
    dec2 = policy2.evaluate(drift_events, 5000, 5000)
    
    assert dec1.should_retrain == dec2.should_retrain
    assert dec1.severity == dec2.severity
    assert dec1.trigger_reason == dec2.trigger_reason
