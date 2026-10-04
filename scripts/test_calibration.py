"""Calibrate Phase 8 drift engine on baseline stream and test all 7 scenarios."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from simulator.event_models import SimulationConfig
from simulator.transit_simulator import DigitalTransitSimulator
from monitoring.drift_models import DriftConfig, DriftType, DriftSeverity
from monitoring.drift_detectors import ReferenceProfiler, FeatureDriftDetector
from monitoring.drift_engine import DriftEngine

profiler = ReferenceProfiler()
ref_profile = profiler.build_profile()

# 1. Test Baseline with calibrated thresholds:
# ph_lambda = 80.0, ph_delta = 0.25
# Environmental precipitation threshold = 3.0 mm/h
config = DriftConfig(
    window_size=500,
    step_size=250,
    min_window_size=100,
    psi_drift_threshold=0.25,
    psi_critical_threshold=0.45,
    perf_warning_multiplier=1.30,
    perf_drift_multiplier=1.60,
    perf_critical_multiplier=2.20,
    ph_delta=0.25,
    ph_lambda=80.0,
    adwin_delta=0.001
)

scenarios = [
    ("baseline", 0.0, None),
    ("rush_hour", 0.75, None),
    ("heavy_rain", 0.80, None),
    ("congestion", 0.70, None),
    ("incident", 0.85, [8, 9, 10]),
    ("dwell", 0.75, None),
    ("combined", 0.85, None)
]

print("=" * 130)
print("PHASE 8 CALIBRATION BENCHMARK EVALUATION")
print("=" * 130)
header = f"{'Scenario':<28} | {'Windows':>7} | {'Data Drift':>10} | {'Perf Drift':>10} | {'Concept Drift':>13} | {'First Detect':>12} | {'Max Severity':>12} | {'MAE (s)':>8}"
print(header)
print("-" * 130)

for sc_id, intensity, segs in scenarios:
    sim_cfg = SimulationConfig(
        scenario_id=sc_id,
        scenario_intensity=intensity,
        scenario_params={"affected_segments": segs} if segs else {},
        auto_predict=True
    )
    sim = DigitalTransitSimulator(config=sim_cfg)
    engine = DriftEngine(config=config, reference_profile=ref_profile)
    engine.set_scenario_context(
        scenario_id=sim.scenario.scenario_id,
        scenario_name=sim.scenario.name,
        synthetic_disturbance=sim.scenario.is_synthetic,
        intensity=sim.scenario.intensity
    )

    sim.start()
    batch_size = 500
    while sim.cursor < sim.total_records and sim.status == "RUNNING":
        step_results = sim.step(min(batch_size, sim.total_records - sim.cursor))
        for telemetry, pred, outcome, eval_event in step_results:
            if outcome is not None and pred is not None:
                cur_idx = int(telemetry.event_id.split("_")[-1])
                engine.process_record(
                    record_idx=cur_idx,
                    features=telemetry.features,
                    predicted_eta=pred.predicted_eta_sec,
                    actual_eta=outcome.actual_eta_sec,
                    timestamp_utc=telemetry.event_timestamp_utc,
                    model_version=pred.model_version
                )
    engine.flush()
    sim_summary = sim.get_performance_summary()
    drift_summary = engine.get_summary()
    first_det = f"#{drift_summary['first_detection_index']}" if drift_summary['first_detection_index'] else "None"
    mae = sim_summary.get("online_metrics", {}).get("mae_sec", 0.0)

    row = f"{sim.scenario.name:<28} | {drift_summary['windows_evaluated']:>7} | {drift_summary['data_drift_alerts']:>10} | {drift_summary['performance_drift_alerts']:>10} | {drift_summary['concept_drift_alerts']:>13} | {first_det:>12} | {drift_summary['max_severity']:>12} | {mae:>8.2f}"
    print(row)

print("=" * 130)
