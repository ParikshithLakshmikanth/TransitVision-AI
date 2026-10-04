"""TransitVision AI - Digital Transit Simulator.
Replays real historical Kandy bus GPS telemetry chronologically as a live stream,
invokes production ETA model inference, resolves actual ground-truth outcomes,
and evaluates streaming prediction errors in real time.
"""
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import PROCESSED_DATA_DIR, SIMULATION_DATA_DIR
from ml.preprocessing.pipeline_preprocessor import ALL_FEATURE_COLUMNS
from ml.inference.predictor import ETAPredictor
from simulator.event_models import (
    TelemetryEvent,
    PredictionEvent,
    OutcomeEvent,
    EvaluationEvent,
    BusState,
    SimulationConfig
)

logger = logging.getLogger("TransitVision.Simulator")


class DigitalTransitSimulator:
    """Event-driven digital transit replay simulator."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        output_dir: Optional[Path] = None,
        source_parquet_path: Optional[Path] = None
    ):
        self.config = config or SimulationConfig()
        self.source_path = Path(source_parquet_path or (ROOT_DIR / self.config.source_dataset_path))
        self.output_dir = Path(output_dir or SIMULATION_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Compute initial source file hash for provenance & immutability verification
        self.initial_source_hash = self._compute_source_hash()

        # Load stream dataset immutably
        self._load_stream_dataset()

        # Initialize ML Predictor
        try:
            self.predictor = ETAPredictor()
            self.model_healthy = True
        except Exception as e:
            logger.warning(f"Production model could not be initialized: {e}")
            self.predictor = None
            self.model_healthy = False

        # Initialize Scenario Engine
        from simulator.scenario_engine import ScenarioEngine, BaseScenario
        if isinstance(self.config.scenario_id, BaseScenario):
            self.scenario = self.config.scenario_id
        else:
            self.scenario = ScenarioEngine.get_scenario(
                self.config.scenario_id,
                intensity=self.config.scenario_intensity,
                **self.config.scenario_params
            )

        # Simulation State
        self.simulation_id = self._generate_simulation_id()
        self.status = "INITIALIZED"  # "INITIALIZED", "RUNNING", "PAUSED", "STOPPED", "COMPLETED"
        self.cursor = 0
        self.replay_speed = self.config.replay_speed
        
        # Timing state
        self.sim_start_wall_time: Optional[float] = None
        self.dataset_start_time: Optional[pd.Timestamp] = None
        self.current_sim_time: Optional[pd.Timestamp] = None
        self.current_dataset_time: Optional[pd.Timestamp] = None

        # Fleet and In-flight states
        self.active_buses: Dict[str, BusState] = {}
        self.recent_events: List[Dict[str, Any]] = []
        
        # Performance Counters
        self.records_emitted = 0
        self.predictions_generated = 0
        self.outcomes_resolved = 0
        self.affected_events_count = 0
        self.unaffected_events_count = 0
        self.absolute_errors: List[float] = []
        self.signed_errors: List[float] = []
        self.inference_latencies_ms: List[float] = []

        # Session Files
        self.events_log_file = self.output_dir / "simulation_events.jsonl"
        self.predictions_log_file = self.output_dir / "prediction_events.jsonl"
        self.outcomes_log_file = self.output_dir / "outcome_events.jsonl"
        self.sessions_meta_file = self.output_dir / "simulation_sessions.json"

        # Record session registration
        self._register_session()

    def _compute_source_hash(self) -> str:
        """Calculates SHA-256 checksum of source dataset."""
        if not self.source_path.exists():
            raise FileNotFoundError(f"Stream parquet not found at: {self.source_path}")
        with open(self.source_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def verify_source_immutability(self) -> bool:
        """Verifies that the source stream parquet remains strictly identical."""
        current_hash = self._compute_source_hash()
        is_identical = current_hash == self.initial_source_hash
        if not is_identical:
            logger.critical("SOURCE PARQUET IMMUTABILITY BREACH: Hash modified during simulation!")
        return is_identical

    def _load_stream_dataset(self) -> None:
        """Loads and filters streaming partition chronologically."""
        logger.info(f"Loading stream dataset from {self.source_path}...")
        df_raw = pd.read_parquet(self.source_path)

        # Verify timestamp column and chronological sort
        if "timestamp_utc" not in df_raw.columns:
            raise KeyError("Stream dataset missing required 'timestamp_utc' column.")
        
        if not pd.api.types.is_datetime64_any_dtype(df_raw["timestamp_utc"]):
            df_raw["timestamp_utc"] = pd.to_datetime(df_raw["timestamp_utc"], utc=True)

        if not df_raw["timestamp_utc"].is_monotonic_increasing:
            logger.warning("Stream records not strictly sorted. Sorting chronologically...")
            df_raw = df_raw.sort_values(by="timestamp_utc").reset_index(drop=True)

        # Apply filters
        df_filtered = df_raw.copy()
        if self.config.filter_trip_id:
            df_filtered = df_filtered[df_filtered["trip_id"].astype(str) == str(self.config.filter_trip_id)]
        if self.config.filter_deviceid:
            df_filtered = df_filtered[df_filtered["deviceid"].astype(str) == str(self.config.filter_deviceid)]
        if self.config.filter_direction is not None:
            df_filtered = df_filtered[df_filtered["direction"].astype(int) == int(self.config.filter_direction)]
        if self.config.filter_segment is not None:
            df_filtered = df_filtered[df_filtered["segment"].astype(int) == int(self.config.filter_segment)]

        self.stream_df = df_filtered.reset_index(drop=True)
        self.total_records = len(self.stream_df)

        if self.total_records > 0:
            self.dataset_start_time = self.stream_df["timestamp_utc"].min()
            self.current_dataset_time = self.dataset_start_time
        else:
            self.dataset_start_time = None
            self.current_dataset_time = None

        logger.info(f"Stream dataset loaded: {self.total_records} records available after filtering.")

    def _generate_simulation_id(self) -> str:
        """Generates unique session ID."""
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rand_suffix = np.random.randint(100, 999)
        return f"SIM_{now_str}_{rand_suffix}"

    def _register_session(self) -> None:
        """Persists session initialization to simulation_sessions.json."""
        sessions = []
        if self.sessions_meta_file.exists():
            try:
                with open(self.sessions_meta_file, "r", encoding="utf-8") as f:
                    sessions = json.load(f)
            except Exception:
                sessions = []

        session_entry = {
            "simulation_id": self.simulation_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": self.status,
            "source_dataset": str(self.source_path.name),
            "source_sha256": self.initial_source_hash,
            "total_records": self.total_records,
            "replay_speed": self.replay_speed,
            "filters": {
                "trip_id": self.config.filter_trip_id,
                "deviceid": self.config.filter_deviceid,
                "direction": self.config.filter_direction,
                "segment": self.config.filter_segment
            },
            "scenario_id": self.config.scenario_id,
            "source_type": "REAL_REPLAY",
            "synthetic_disturbance": False,
            "model_version": self.predictor.metadata.get("version", "v1.0.0") if self.predictor and self.predictor.metadata else "v1.0.0",
            "records_emitted": 0,
            "predictions_generated": 0,
            "outcomes_resolved": 0
        }
        sessions.append(session_entry)
        with open(self.sessions_meta_file, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2)

    def _update_session_metadata(self) -> None:
        """Updates current session metadata in simulation_sessions.json."""
        if not self.sessions_meta_file.exists():
            return
        try:
            with open(self.sessions_meta_file, "r", encoding="utf-8") as f:
                sessions = json.load(f)

            for s in sessions:
                if s["simulation_id"] == self.simulation_id:
                    s["status"] = self.status
                    s["records_emitted"] = self.records_emitted
                    s["predictions_generated"] = self.predictions_generated
                    s["outcomes_resolved"] = self.outcomes_resolved
                    s["cursor"] = self.cursor
                    s["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                    if len(self.absolute_errors) > 0:
                        s["metrics"] = {
                            "mae_sec": round(float(np.mean(self.absolute_errors)), 2),
                            "rmse_sec": round(float(np.sqrt(np.mean(np.array(self.signed_errors) ** 2))), 2),
                            "median_ae_sec": round(float(np.median(self.absolute_errors)), 2),
                            "avg_latency_ms": round(float(np.mean(self.inference_latencies_ms)), 3) if self.inference_latencies_ms else 0.0
                        }

            with open(self.sessions_meta_file, "w", encoding="utf-8") as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed updating session metadata: {e}")

    # -------------------------------------------------------------
    # Programmatic Controls
    # -------------------------------------------------------------
    def start(self) -> None:
        """Starts simulation playback from current cursor."""
        if self.status == "RUNNING":
            return
        self.status = "RUNNING"
        self.sim_start_wall_time = time.time()
        self.current_sim_time = datetime.now(timezone.utc)
        logger.info(f"Simulator {self.simulation_id} started at cursor {self.cursor} (Speed: {self.replay_speed}x).")
        self._update_session_metadata()

    def pause(self) -> None:
        """Pauses simulation replay."""
        if self.status == "RUNNING":
            self.status = "PAUSED"
            logger.info(f"Simulator {self.simulation_id} PAUSED at cursor {self.cursor}.")
            self._update_session_metadata()

    def resume(self) -> None:
        """Resumes simulation replay from pause."""
        if self.status == "PAUSED":
            self.status = "RUNNING"
            self.sim_start_wall_time = time.time()
            logger.info(f"Simulator {self.simulation_id} RESUMED at cursor {self.cursor}.")
            self._update_session_metadata()

    def stop(self) -> None:
        """Terminates current simulation."""
        self.status = "STOPPED"
        logger.info(f"Simulator {self.simulation_id} STOPPED.")
        self._update_session_metadata()

    def reset(self) -> None:
        """Resets replay cursor to beginning."""
        self.cursor = 0
        self.records_emitted = 0
        self.predictions_generated = 0
        self.outcomes_resolved = 0
        self.absolute_errors.clear()
        self.signed_errors.clear()
        self.inference_latencies_ms.clear()
        self.active_buses.clear()
        self.recent_events.clear()
        self.status = "INITIALIZED"
        if self.total_records > 0:
            self.current_dataset_time = self.stream_df.iloc[0]["timestamp_utc"]
        logger.info(f"Simulator {self.simulation_id} RESET to cursor 0.")
        self._update_session_metadata()

    def seek(self, index: int) -> None:
        """Seeks replay to specific record index."""
        if 0 <= index < self.total_records:
            self.cursor = index
            self.current_dataset_time = self.stream_df.iloc[index]["timestamp_utc"]
            logger.info(f"Simulator {self.simulation_id} SEEK to index {index}.")
            self._update_session_metadata()
        else:
            raise IndexError(f"Seek index {index} out of bounds (0..{self.total_records - 1})")

    def set_speed(self, speed: float) -> None:
        """Modifies replay speed factor dynamically."""
        self.replay_speed = max(0.0, float(speed))
        logger.info(f"Simulator {self.simulation_id} replay speed set to {self.replay_speed}x.")

    def is_finished(self) -> bool:
        """Returns True if simulation has replayed all available records."""
        return self.cursor >= self.total_records or self.status == "COMPLETED"

    # -------------------------------------------------------------
    # Step & Replay Execution
    # -------------------------------------------------------------
    def step(self, n: int = 1) -> List[Tuple[TelemetryEvent, Optional[PredictionEvent], Optional[OutcomeEvent], Optional[EvaluationEvent]]]:
        """
        Advances simulation by n records.
        Processes Telemetry -> Prediction -> Outcome -> Evaluation events
        with full event-level provenance and lifecycle preservation.
        """
        results = []
        if self.status not in ["RUNNING", "INITIALIZED"]:
            return results

        if self.status == "INITIALIZED":
            self.start()

        if self.cursor >= self.total_records:
            self.status = "COMPLETED"
            logger.info(f"Simulator {self.simulation_id} COMPLETED stream replay.")
            self._update_session_metadata()
            return results

        actual_n = min(n, self.total_records - self.cursor)
        records_slice = self.stream_df.iloc[self.cursor : self.cursor + actual_n]
        start_cursor = self.cursor
        self.cursor += actual_n
        self.records_emitted += actual_n

        # Transform features and simulated ground truth through active scenario
        raw_features_list = []
        mod_features_list = []
        sim_gt_list = []
        raw_gt_list = []
        dist_applied_list = []

        for i in range(actual_n):
            record = records_slice.iloc[i]
            event_ts_utc = pd.to_datetime(record["timestamp_utc"]).isoformat()
            segment = int(record["segment"])
            ground_truth_eta = float(record["eta_to_next_stop_sec"])
            raw_features = {col: record[col] for col in ALL_FEATURE_COLUMNS if col in record}

            mod_feat, sim_gt, dist_applied = self.scenario.apply(
                raw_features, ground_truth_eta, event_ts_utc, segment
            )
            raw_features_list.append(raw_features)
            mod_features_list.append(mod_feat)
            raw_gt_list.append(ground_truth_eta)
            sim_gt_list.append(sim_gt)
            dist_applied_list.append(dist_applied)
            if dist_applied:
                self.affected_events_count += 1
            else:
                self.unaffected_events_count += 1

        # If auto_predict is enabled, predict on modified features
        pred_results = None
        if self.config.auto_predict and self.predictor is not None:
            features_df = pd.DataFrame(mod_features_list)
            pred_results = self.predictor.predict(features_df)

        events_to_log = []
        preds_to_log = []
        outcomes_to_log = []

        for i in range(actual_n):
            record = records_slice.iloc[i]
            cur_idx = start_cursor + i + 1

            event_ts_utc = pd.to_datetime(record["timestamp_utc"]).isoformat()
            sim_ts_utc = datetime.now(timezone.utc).isoformat()
            trip_id = str(record["trip_id"])
            deviceid = str(record["deviceid"])
            direction = int(record["direction"])
            segment = int(record["segment"])
            ground_truth_eta = sim_gt_list[i]
            baseline_gt_eta = raw_gt_list[i]
            features = mod_features_list[i]
            dist_applied = dist_applied_list[i]

            telemetry_event = TelemetryEvent(
                event_id=f"EVT_{self.simulation_id}_{cur_idx:06d}",
                event_timestamp_utc=event_ts_utc,
                simulation_timestamp_utc=sim_ts_utc,
                trip_id=trip_id,
                deviceid=deviceid,
                direction=direction,
                segment=segment,
                features=features,
                source_type="REAL_REPLAY",
                source_dataset=self.source_path.name,
                simulation_generated=True,
                synthetic_disturbance=self.scenario.is_synthetic,
                scenario_id=self.scenario.scenario_id,
                scenario_name=self.scenario.name,
                disturbance_intensity=self.scenario.intensity,
                disturbance_applied=dist_applied,
                affected_scope=str(self.scenario.affected_segments or "GLOBAL"),
                _ground_truth_eta_sec=ground_truth_eta,
                _baseline_ground_truth_eta_sec=baseline_gt_eta
            )
            events_to_log.append(telemetry_event.to_dict(include_ground_truth=False))

            pred_event: Optional[PredictionEvent] = None
            if pred_results is not None:
                if actual_n == 1:
                    pred_eta_val = float(pred_results["predicted_eta_sec"])
                    raw_pred_val = float(pred_results.get("raw_prediction_sec", pred_eta_val))
                else:
                    pred_eta_val = float(pred_results["predicted_eta_sec"][i])
                    raw_pred_val = float(pred_results.get("raw_prediction_sec", [pred_eta_val])[i])

                latency_ms = float(pred_results.get("latency_ms", 0.0)) / actual_n
                self.inference_latencies_ms.append(latency_ms)

                pred_event = PredictionEvent(
                    prediction_id=f"PRED_{telemetry_event.event_id}",
                    event_id=telemetry_event.event_id,
                    trip_id=trip_id,
                    deviceid=deviceid,
                    direction=direction,
                    segment=segment,
                    predicted_eta_sec=pred_eta_val,
                    raw_prediction_sec=raw_pred_val,
                    prediction_timestamp_utc=sim_ts_utc,
                    model_id=pred_results.get("model_id", "eta_model_v1"),
                    model_version=pred_results.get("model_version", "v1.0.0"),
                    algorithm=pred_results.get("algorithm", "LightGBMRegressor"),
                    latency_ms=latency_ms,
                    scenario_id=self.scenario.scenario_id,
                    synthetic_disturbance=self.scenario.is_synthetic
                )
                self.predictions_generated += 1
                preds_to_log.append(pred_event.to_dict())

            outcome_event: Optional[OutcomeEvent] = None
            eval_event: Optional[EvaluationEvent] = None

            if ground_truth_eta is not None and pred_event is not None:
                outcome_event = OutcomeEvent(
                    outcome_id=f"OUT_{pred_event.prediction_id}",
                    prediction_id=pred_event.prediction_id,
                    event_id=telemetry_event.event_id,
                    trip_id=trip_id,
                    deviceid=deviceid,
                    direction=direction,
                    segment=segment,
                    actual_eta_sec=ground_truth_eta,
                    baseline_actual_eta_sec=baseline_gt_eta,
                    outcome_timestamp_utc=sim_ts_utc,
                    scenario_id=self.scenario.scenario_id,
                    synthetic_disturbance=self.scenario.is_synthetic
                )
                self.outcomes_resolved += 1
                outcomes_to_log.append(outcome_event.to_dict())

                signed_err = round(ground_truth_eta - pred_event.predicted_eta_sec, 2)
                abs_err = round(abs(signed_err), 2)
                pct_err = round((abs_err / max(ground_truth_eta, 10.0)) * 100.0, 2)

                self.signed_errors.append(signed_err)
                self.absolute_errors.append(abs_err)

                eval_event = EvaluationEvent(
                    evaluation_id=f"EVAL_{pred_event.prediction_id}",
                    prediction_id=pred_event.prediction_id,
                    event_id=telemetry_event.event_id,
                    trip_id=trip_id,
                    deviceid=deviceid,
                    direction=direction,
                    segment=segment,
                    predicted_eta_sec=pred_event.predicted_eta_sec,
                    actual_eta_sec=ground_truth_eta,
                    baseline_actual_eta_sec=baseline_gt_eta,
                    signed_error_sec=signed_err,
                    absolute_error_sec=abs_err,
                    percentage_error=pct_err,
                    evaluation_timestamp_utc=sim_ts_utc,
                    model_version=pred_event.model_version,
                    scenario_id=self.scenario.scenario_id,
                    synthetic_disturbance=self.scenario.is_synthetic,
                    disturbance_applied=dist_applied
                )

            # Live Bus Fleet State
            self.active_buses[deviceid] = BusState(
                deviceid=deviceid,
                trip_id=trip_id,
                direction=direction,
                current_segment=segment,
                last_event_timestamp_utc=event_ts_utc,
                last_simulation_timestamp_utc=sim_ts_utc,
                last_predicted_eta_sec=pred_event.predicted_eta_sec if pred_event else None,
                last_actual_eta_sec=ground_truth_eta,
                last_error_sec=eval_event.signed_error_sec if eval_event else None,
                status="ACTIVE"
            )

            summary_item = {
                "cursor": cur_idx,
                "event_id": telemetry_event.event_id,
                "deviceid": deviceid,
                "trip_id": trip_id,
                "direction": direction,
                "segment": segment,
                "event_time_utc": event_ts_utc,
                "predicted_eta_sec": pred_event.predicted_eta_sec if pred_event else None,
                "actual_eta_sec": ground_truth_eta,
                "baseline_actual_eta_sec": baseline_gt_eta,
                "error_sec": eval_event.signed_error_sec if eval_event else None,
                "model_version": pred_event.model_version if pred_event else None,
                "disturbance_applied": dist_applied
            }
            self.recent_events.append(summary_item)
            if len(self.recent_events) > 100:
                self.recent_events.pop(0)

            results.append((telemetry_event, pred_event, outcome_event, eval_event))

        # Bulk write logs
        if events_to_log:
            self._log_batch_jsonl(self.events_log_file, events_to_log)
        if preds_to_log:
            self._log_batch_jsonl(self.predictions_log_file, preds_to_log)
        if outcomes_to_log:
            self._log_batch_jsonl(self.outcomes_log_file, outcomes_to_log)

        self._update_session_metadata()
        return results

    def run_all(self) -> Dict[str, Any]:
        """Runs through the entire filtered stream to completion."""
        self.start()
        logger.info(f"Running full simulation for {self.total_records} records...")
        t0 = time.perf_counter()
        
        # Process in batches for high throughput
        batch_size = 500
        while self.cursor < self.total_records and self.status == "RUNNING":
            self.step(min(batch_size, self.total_records - self.cursor))

        elapsed_sec = time.perf_counter() - t0
        self.status = "COMPLETED"
        self._update_session_metadata()

        summary = self.get_performance_summary()
        summary["simulation_wall_time_sec"] = round(elapsed_sec, 3)
        summary["throughput_records_per_sec"] = round(self.records_emitted / max(elapsed_sec, 0.001), 1)
        return summary

    def _log_jsonl(self, filepath: Path, data: Dict[str, Any]) -> None:
        """Thread-safe append of JSON line."""
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.debug(f"Log append error to {filepath}: {e}")

    def _log_batch_jsonl(self, filepath: Path, data_list: List[Dict[str, Any]]) -> None:
        """Efficient batch append of JSON lines."""
        if not data_list:
            return
        try:
            lines = [json.dumps(d) + "\n" for d in data_list]
            with open(filepath, "a", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception as e:
            logger.debug(f"Batch log append error to {filepath}: {e}")

    # -------------------------------------------------------------
    # Live State & Performance APIs
    # -------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Exposes complete simulator state for future FastAPI/WebSocket consumers."""
        current_ts = self.stream_df.iloc[self.cursor - 1]["timestamp_utc"] if (self.cursor > 0 and self.cursor <= self.total_records) else self.dataset_start_time
        return {
            "simulation_id": self.simulation_id,
            "status": self.status,
            "cursor": self.cursor,
            "total_records": self.total_records,
            "progress_pct": round((self.cursor / max(self.total_records, 1)) * 100, 2),
            "replay_speed": self.replay_speed,
            "current_source_timestamp_utc": str(current_ts) if current_ts is not None else None,
            "simulation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "records_emitted": self.records_emitted,
            "predictions_generated": self.predictions_generated,
            "outcomes_resolved": self.outcomes_resolved,
            "active_buses_count": len(self.active_buses),
            "model_version": self.predictor.metadata.get("version", "v1.0.0") if self.predictor and self.predictor.metadata else "v1.0.0",
            "scenario_id": self.scenario.scenario_id,
            "scenario_name": self.scenario.name,
            "synthetic_disturbance": self.scenario.is_synthetic,
            "disturbance_intensity": self.scenario.intensity,
            "affected_events": self.affected_events_count,
            "unaffected_events": self.unaffected_events_count,
            "source_dataset": self.source_path.name,
            "source_hash_verified": self.verify_source_immutability()
        }

    def get_active_buses(self) -> Dict[str, Dict[str, Any]]:
        """Returns live fleet snapshot."""
        return {dev: bus.to_dict() for dev, bus in self.active_buses.items()}

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns latest event history."""
        return self.recent_events[-limit:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Calculates online error statistics and runtime performance."""
        if len(self.absolute_errors) > 0:
            mae = float(np.mean(self.absolute_errors))
            rmse = float(np.sqrt(np.mean(np.array(self.signed_errors) ** 2)))
            median_ae = float(np.median(self.absolute_errors))
            p90_err = float(np.percentile(self.absolute_errors, 90))
            p95_err = float(np.percentile(self.absolute_errors, 95))
        else:
            mae = rmse = median_ae = p90_err = p95_err = 0.0

        avg_latency = float(np.mean(self.inference_latencies_ms)) if self.inference_latencies_ms else 0.0

        return {
            "simulation_id": self.simulation_id,
            "status": self.status,
            "scenario_id": self.scenario.scenario_id,
            "scenario_name": self.scenario.name,
            "synthetic_disturbance": self.scenario.is_synthetic,
            "disturbance_intensity": self.scenario.intensity,
            "records_emitted": self.records_emitted,
            "predictions_generated": self.predictions_generated,
            "outcomes_resolved": self.outcomes_resolved,
            "affected_events": self.affected_events_count,
            "unaffected_events": self.unaffected_events_count,
            "active_devices": len(self.active_buses),
            "source_hash_verified": self.verify_source_immutability(),
            "online_metrics": {
                "mae_sec": round(mae, 2),
                "rmse_sec": round(rmse, 2),
                "median_ae_sec": round(median_ae, 2),
                "p90_error_sec": round(p90_err, 2),
                "p95_error_sec": round(p95_err, 2),
                "avg_inference_latency_ms": round(avg_latency, 3)
            }
        }
