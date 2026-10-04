"""TransitVision AI - Scenario Experiment & Disturbance Simulation Runner.
Executes controlled operational disturbance scenarios on real Kandy stream telemetry,
evaluates online prediction performance degradation, and generates comparison benchmarks.
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import METADATA_DIR
from simulator.event_models import SimulationConfig
from simulator.transit_simulator import DigitalTransitSimulator
from simulator.scenario_engine import ScenarioEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TransitVision.ScenarioRunner")


def run_single_scenario(
    scenario_name: str,
    intensity: float = 0.75,
    segments: str = None,
    limit: int = 0,
    speed: float = 10000.0,
    quiet: bool = False
) -> Dict[str, Any]:
    """Runs a single scenario simulation session and returns metrics summary."""
    scenario_params = {}
    if segments:
        seg_list = [int(s.strip()) for s in segments.split(",") if s.strip()]
        scenario_params["affected_segments"] = seg_list

    config = SimulationConfig(
        replay_speed=speed,
        scenario_id=scenario_name,
        scenario_intensity=intensity if scenario_name.lower() != "baseline" else 0.0,
        scenario_params=scenario_params,
        auto_predict=True
    )

    sim = DigitalTransitSimulator(config=config)
    total_available = sim.total_records
    num_to_run = total_available if (limit <= 0 or limit > total_available) else limit

    if not quiet:
        print(f"\n[RUNNING SCENARIO] {sim.scenario.name} (ID: {sim.scenario.scenario_id}, Intensity: {sim.scenario.intensity})")
        print(f"Total stream records to process: {num_to_run}...")

    t0 = time.perf_counter()
    sim.start()

    # Step in batches
    batch_size = 500
    remaining = num_to_run
    while remaining > 0 and sim.status in ["RUNNING", "INITIALIZED"]:
        step_size = min(batch_size, remaining)
        step_res = sim.step(step_size)
        if not step_res:
            break
        remaining -= len(step_res)

    elapsed = time.perf_counter() - t0
    sim.stop()

    summary = sim.get_performance_summary()
    summary["wall_time_sec"] = round(elapsed, 3)
    summary["throughput_records_per_sec"] = round(sim.records_emitted / max(elapsed, 0.001), 1)

    if not quiet:
        m = summary["online_metrics"]
        print(f"[COMPLETED in {elapsed:.3f}s] MAE: {m['mae_sec']}s | RMSE: {m['rmse_sec']}s | MedAE: {m['median_ae_sec']}s | P90: {m['p90_error_sec']}s | P95: {m['p95_error_sec']}s | Affected: {summary['affected_events']}/{summary['records_emitted']}")

    return summary


def run_benchmark_suite(limit: int = 0, speed: float = 10000.0) -> Dict[str, Any]:
    """Executes all 7 scenarios across the entire stream and generates comparative benchmark."""
    print("\n" + "=" * 80)
    print("        TRANSITVISION AI — PHASE 7 SCENARIO BENCHMARK EXPERIMENT SUITE        ")
    print("=" * 80)
    print(f"Source Dataset: data/processed/kandy_eta_stream.parquet")
    print(f"Replay Records: {'ALL (30,102)' if limit <= 0 else str(limit)}")
    print("=" * 80 + "\n")

    scenario_configs = [
        {"id": "baseline", "intensity": 0.0, "desc": "Baseline (Control Group)", "segments": None},
        {"id": "rush_hour", "intensity": 0.75, "desc": "Rush Hour Congestion", "segments": None},
        {"id": "heavy_rain", "intensity": 0.80, "desc": "Heavy Monsoon Rain", "segments": None},
        {"id": "congestion", "intensity": 0.70, "desc": "Corridor Congestion Surge", "segments": None},
        {"id": "incident", "intensity": 0.85, "desc": "Road Incident (Seg 8,9,10)", "segments": "8,9,10"},
        {"id": "dwell", "intensity": 0.75, "desc": "Passenger Dwell Surge", "segments": None},
        {"id": "combined", "intensity": 0.85, "desc": "Combined Severe Disruption", "segments": None}
    ]

    benchmark_results = []

    for sc in scenario_configs:
        res = run_single_scenario(
            scenario_name=sc["id"],
            intensity=sc["intensity"],
            segments=sc["segments"],
            limit=limit,
            speed=speed,
            quiet=False
        )
        res["scenario_display_name"] = sc["desc"]
        res["configured_intensity"] = sc["intensity"]
        benchmark_results.append(res)

    # Persist benchmark to data/metadata/scenario_experiments.json
    out_meta = {
        "benchmark_generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "data/processed/kandy_eta_stream.parquet",
        "total_stream_records": benchmark_results[0]["records_emitted"],
        "model_id": "model_lightgbm_v1",
        "model_version": "v1.0.0",
        "scenarios": benchmark_results
    }

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = METADATA_DIR / "scenario_experiments.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)

    print(f"\n[BENCHMARK COMPLETE] Saved experiment manifest to: {out_file}\n")

    # Print Formatted Markdown Table
    print("=" * 90)
    print(f"{'Scenario Name':<30} | {'MAE (s)':>8} | {'RMSE (s)':>8} | {'MedAE (s)':>9} | {'P90 (s)':>8} | {'P95 (s)':>8} | {'Affected':>8}")
    print("-" * 90)
    for r in benchmark_results:
        m = r["online_metrics"]
        print(f"{r['scenario_display_name']:<30} | {m['mae_sec']:>8.2f} | {m['rmse_sec']:>8.2f} | {m['median_ae_sec']:>9.2f} | {m['p90_error_sec']:>8.2f} | {m['p95_error_sec']:>8.2f} | {r['affected_events']:>8}")
    print("=" * 90 + "\n")

    return out_meta


def main():
    parser = argparse.ArgumentParser(description="TransitVision AI - Scenario Experiment Runner")
    parser.add_argument("--scenario", type=str, default="baseline", help="Scenario to run (baseline, rush_hour, heavy_rain, congestion, incident, dwell, combined)")
    parser.add_argument("--intensity", type=float, default=0.75, help="Disturbance intensity (0.0 to 1.0)")
    parser.add_argument("--segments", type=str, default=None, help="Comma-separated affected segments (for incident scenario)")
    parser.add_argument("--limit", type=int, default=0, help="Number of records to replay (default 0 for all)")
    parser.add_argument("--speed", type=str, default="max", help="Replay speed multiplier")
    parser.add_argument("--benchmark", action="store_true", help="Run full 7-scenario comparison benchmark")

    args = parser.parse_args()

    speed_val = 10000.0 if args.speed.lower() == "max" else float(args.speed)

    if args.benchmark:
        run_benchmark_suite(limit=args.limit, speed=speed_val)
    else:
        run_single_scenario(
            scenario_name=args.scenario,
            intensity=args.intensity,
            segments=args.segments,
            limit=args.limit,
            speed=speed_val,
            quiet=False
        )


if __name__ == "__main__":
    main()
