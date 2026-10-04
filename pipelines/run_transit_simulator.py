"""TransitVision AI - Digital Transit Simulator Demonstration CLI.
Executes chronological replay of real Kandy bus GPS telemetry with live ML predictions,
ground-truth outcome resolution, and online error tracking.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from simulator.event_models import SimulationConfig
from simulator.transit_simulator import DigitalTransitSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TransitVision.SimulatorCLI")


def main():
    parser = argparse.ArgumentParser(description="TransitVision AI - Digital Transit Simulator")
    parser.add_argument("--speed", type=str, default="100", help="Replay speed factor (e.g., 10, 100, 500, max)")
    parser.add_argument("--limit", type=int, default=100, help="Number of records to replay (default: 100, use 0 for all)")
    parser.add_argument("--trip_id", type=str, default=None, help="Filter by trip ID")
    parser.add_argument("--deviceid", type=str, default=None, help="Filter by device ID")
    parser.add_argument("--direction", type=int, default=None, help="Filter by direction (1 or 2)")
    parser.add_argument("--segment", type=int, default=None, help="Filter by segment ID (1-34)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-record terminal printing")

    args = parser.parse_args()

    # Parse speed
    if args.speed.lower() == "max":
        speed = 10000.0
    else:
        try:
            speed = float(args.speed)
        except ValueError:
            speed = 100.0

    config = SimulationConfig(
        replay_speed=speed,
        filter_trip_id=args.trip_id,
        filter_deviceid=args.deviceid,
        filter_direction=args.direction,
        filter_segment=args.segment,
        auto_predict=True,
        scenario_id="BASELINE"
    )

    print("\n" + "=" * 65)
    print("      TRANSITVISION AI — DIGITAL TRANSIT SIMULATOR (PHASE 6)     ")
    print("=" * 65)
    print(f"Source Dataset   : data/processed/kandy_eta_stream.parquet")
    print(f"Replay Speed     : {speed}x")
    print(f"Filter Trip      : {args.trip_id or 'ALL'}")
    print(f"Filter Device    : {args.deviceid or 'ALL'}")
    print(f"Filter Direction : {args.direction or 'ALL'}")
    print(f"Filter Segment   : {args.segment or 'ALL'}")
    print("=" * 65 + "\n")

    sim = DigitalTransitSimulator(config=config)
    total_available = sim.total_records
    num_to_run = total_available if (args.limit <= 0 or args.limit > total_available) else args.limit

    print(f"[SIMULATION] Session ID: {sim.simulation_id}")
    print(f"[SIMULATION] Starting replay of {num_to_run} / {total_available} real records...\n")

    sim.start()
    t_start = time.perf_counter()

    if args.quiet:
        batch_size = 500
        remaining = num_to_run
        while remaining > 0 and sim.status in ["RUNNING", "INITIALIZED"]:
            step_size = min(batch_size, remaining)
            step_res = sim.step(step_size)
            if not step_res:
                break
            remaining -= len(step_res)
    else:
        for i in range(num_to_run):
            step_res = sim.step(1)
            if not step_res:
                break
            
            telemetry, pred, outcome, evaluation = step_res[0]

            print(f"[REPLAY]     device={telemetry.deviceid:<5} trip={telemetry.trip_id:<6} dir={telemetry.direction} seg={telemetry.segment:<2} time={telemetry.event_timestamp_utc}")
            if pred:
                print(f"[PREDICTION] ETA={pred.predicted_eta_sec:>6.2f}s  model={pred.model_version} ({pred.algorithm})  latency={pred.latency_ms:.2f}ms")
            if outcome:
                print(f"[OUTCOME]    Actual={outcome.actual_eta_sec:>6.2f}s")
            if evaluation:
                print(f"[EVALUATION] Error={evaluation.signed_error_sec:>+6.2f}s  |Err|={evaluation.absolute_error_sec:>6.2f}s  PctErr={evaluation.percentage_error:>5.1f}%\n")

    elapsed = time.perf_counter() - t_start
    sim.stop()

    summary = sim.get_performance_summary()
    state = sim.get_state()

    print("=" * 65)
    print("                 SIMULATION RUN COMPLETED                        ")
    print("=" * 65)
    print(f"Status                  : {state['status']}")
    print(f"Records Emitted         : {sim.records_emitted}")
    print(f"Predictions Generated   : {sim.predictions_generated}")
    print(f"Outcomes Resolved       : {sim.outcomes_resolved}")
    print(f"Active Fleet Size       : {len(sim.active_buses)} buses")
    print(f"Wall-Clock Duration     : {elapsed:.3f} seconds")
    print(f"Throughput              : {sim.records_emitted / max(elapsed, 0.001):.1f} records/sec")
    print(f"Source Hash Verified    : {'YES (IMMUTABLE)' if sim.verify_source_immutability() else 'NO (MODIFIED!)'}")
    print("-" * 65)
    print("ONLINE EVALUATION ERROR METRICS:")
    m = summary.get("online_metrics", {})
    print(f"  MAE                   : {m.get('mae_sec', 0.0)} seconds")
    print(f"  RMSE                  : {m.get('rmse_sec', 0.0)} seconds")
    print(f"  Median Absolute Error : {m.get('median_ae_sec', 0.0)} seconds")
    print(f"  P90 Absolute Error    : {m.get('p90_error_sec', 0.0)} seconds")
    print(f"  P95 Absolute Error    : {m.get('p95_error_sec', 0.0)} seconds")
    print(f"  Avg Inference Latency : {m.get('avg_inference_latency_ms', 0.0)} ms")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
