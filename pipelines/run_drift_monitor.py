"""TransitVision AI - Drift Monitoring Pipeline Runner.
Executes online statistical drift monitoring over Digital Transit Simulator replay,
evaluating Covariate Drift, Performance Drift, and Concept Drift across scenarios.
"""
import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import PROCESSED_DATA_DIR, METADATA_DIR
from simulator.event_models import SimulationConfig
from simulator.transit_simulator import DigitalTransitSimulator
from monitoring.drift_models import DriftConfig, DriftType, DriftSeverity
from monitoring.drift_detectors import ReferenceProfiler
from monitoring.drift_engine import DriftEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("TransitVision.DriftRunner")


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash for data integrity verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_drift_monitoring_on_scenario(
    scenario_id: str,
    intensity: float = 0.75,
    segments: Optional[List[int]] = None,
    drift_config: Optional[DriftConfig] = None,
    reference_profile: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Executes drift monitoring on a single simulation scenario.
    """
    cfg = drift_config or DriftConfig()
    engine = DriftEngine(config=cfg, reference_profile=reference_profile)

    # Initialize Simulator
    sim_cfg = SimulationConfig(
        scenario_id=scenario_id,
        scenario_intensity=intensity,
        scenario_params={"affected_segments": segments} if segments else {},
        auto_predict=True
    )
    sim = DigitalTransitSimulator(config=sim_cfg)
    
    engine.set_scenario_context(
        scenario_id=sim.scenario.scenario_id,
        scenario_name=sim.scenario.name,
        synthetic_disturbance=sim.scenario.is_synthetic,
        intensity=sim.scenario.intensity
    )

    logger.info(f"--- Running Drift Monitoring for Scenario: {sim.scenario.name} (Intensity: {intensity}) ---")
    sim.start()
    t0 = time.perf_counter()

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
    elapsed = time.perf_counter() - t0
    sim_summary = sim.get_performance_summary()
    drift_summary = engine.get_summary()
    drift_summary["wall_time_sec"] = round(elapsed, 2)
    drift_summary["mae_sec"] = sim_summary.get("online_metrics", {}).get("mae_sec", 0.0)
    drift_summary["rmse_sec"] = sim_summary.get("online_metrics", {}).get("rmse_sec", 0.0)

    events_list = [e.to_dict() for e in engine.events_history]
    return drift_summary, events_list


def run_all_scenarios_drift_benchmark(drift_config: Optional[DriftConfig] = None) -> List[Dict[str, Any]]:
    """
    Executes the full 7-scenario Phase 8 Drift Monitoring Benchmark.
    """
    stream_path = PROCESSED_DATA_DIR / "kandy_eta_stream.parquet"
    sha_before = compute_file_sha256(stream_path)
    logger.info(f"Verified source Parquet SHA-256 before benchmark: {sha_before}")

    # Build / verify reference profile
    profiler = ReferenceProfiler()
    ref_profile = profiler.build_profile()

    scenarios = [
        {"scenario": "baseline", "intensity": 0.0, "segments": None},
        {"scenario": "rush_hour", "intensity": 0.75, "segments": None},
        {"scenario": "heavy_rain", "intensity": 0.80, "segments": None},
        {"scenario": "congestion", "intensity": 0.70, "segments": None},
        {"scenario": "incident", "intensity": 0.85, "segments": [8, 9, 10]},
        {"scenario": "dwell", "intensity": 0.75, "segments": None},
        {"scenario": "combined", "intensity": 0.85, "segments": None},
    ]

    all_summaries = []
    all_events = []
    window_records = []

    for sc in scenarios:
        summary, events = run_drift_monitoring_on_scenario(
            scenario_id=sc["scenario"],
            intensity=sc["intensity"],
            segments=sc["segments"],
            drift_config=drift_config,
            reference_profile=ref_profile
        )
        all_summaries.append(summary)
        all_events.extend(events)

    # Persist benchmark manifest
    summary_path = METADATA_DIR / "drift_monitoring_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    events_path = METADATA_DIR / "drift_events.json"
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2)

    # Create monitoring results parquet
    if all_events:
        results_df = pd.DataFrame(all_events)
        results_parquet = PROCESSED_DATA_DIR / "drift_monitoring_results.parquet"
        results_df.to_parquet(results_parquet, index=False)
        logger.info(f"Saved drift monitoring results to {results_parquet} ({len(results_df)} records)")

    sha_after = compute_file_sha256(stream_path)
    logger.info(f"Verified source Parquet SHA-256 after benchmark: {sha_after}")
    assert sha_before == sha_after, "SOURCE DATA INTEGRITY VIOLATION DETECTED!"

    print("\n" + "=" * 125)
    print("PHASE 8: CONCEPT DRIFT DETECTION & STATISTICAL MONITORING BENCHMARK RESULTS")
    print("=" * 125)
    header = f"{'Scenario':<28} | {'Windows':>7} | {'Data Drift':>10} | {'Perf Drift':>10} | {'Concept Drift':>13} | {'First Detect':>12} | {'Max Severity':>12} | {'MAE (s)':>8}"
    print(header)
    print("-" * 125)
    for s in all_summaries:
        first_det = f"#{s['first_detection_index']}" if s['first_detection_index'] else "None"
        row = f"{s['scenario_name']:<28} | {s['windows_evaluated']:>7} | {s['data_drift_alerts']:>10} | {s['performance_drift_alerts']:>10} | {s['concept_drift_alerts']:>13} | {first_det:>12} | {s['max_severity']:>12} | {s['mae_sec']:>8.2f}"
        print(row)
    print("=" * 125 + "\n")

    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="TransitVision AI - Drift Monitoring Pipeline")
    parser.add_argument("--scenario", type=str, default="all", help="Scenario ID or 'all'")
    parser.add_argument("--intensity", type=float, default=0.75, help="Disturbance intensity [0.0..1.0]")
    parser.add_argument("--segments", type=str, default=None, help="Comma-separated segments (e.g. 8,9,10)")
    parser.add_argument("--all-scenarios", action="store_true", help="Run full 7-scenario benchmark")
    parser.add_argument("--build-profile-only", action="store_true", help="Only build reference profile")

    args = parser.parse_args()

    if args.build_profile_only:
        profiler = ReferenceProfiler()
        profiler.build_profile(force_rebuild=True)
        print("Reference profile built successfully.")
        return

    if args.all_scenarios or args.scenario.lower() == "all":
        run_all_scenarios_drift_benchmark()
    else:
        segs = [int(s.strip()) for s in args.segments.split(",")] if args.segments else None
        summary, events = run_drift_monitoring_on_scenario(
            scenario_id=args.scenario,
            intensity=args.intensity,
            segments=segs
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
