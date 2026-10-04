"""TransitVision AI - Authoritative Digital Transit Simulator Service."""
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from simulator.transit_simulator import DigitalTransitSimulator
from simulator.event_models import SimulationConfig, BusState
from simulator.scenario_engine import ScenarioEngine
from backend.services.event_service import EventService
from backend.services.monitoring_service import MonitoringService
from backend.schemas import (
    SimulationStatusResponse,
    BusStateResponse,
    PredictionItem,
    MetricsResponse,
    OnlineMetrics,
    MetricsHistoryItem,
    MetricsHistoryResponse,
    SystemEventType,
)

logger = logging.getLogger("TransitVision.SimulatorService")


class SimulatorService:
    """
    Singleton service managing the lifecycle, background playback loop,
    and event distribution of the DigitalTransitSimulator.
    """

    def __init__(
        self,
        event_service: Optional[EventService] = None,
        monitoring_service: Optional[MonitoringService] = None,
        config: Optional[SimulationConfig] = None,
    ):
        self.event_service = event_service or EventService()
        self.monitoring_service = monitoring_service or MonitoringService()
        self.config = config or SimulationConfig(auto_predict=True)
        
        # Core simulator instance
        self.simulator = DigitalTransitSimulator(config=self.config)
        
        # Concurrency & background playback state
        self._playback_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self._recent_predictions_buffer: List[PredictionItem] = []
        self._metrics_history: List[MetricsHistoryItem] = []
        self._history_window_counter = 0

    # -------------------------------------------------------------
    # Simulation Lifecycle Controls
    # -------------------------------------------------------------
    def start(self) -> SimulationStatusResponse:
        """Starts the simulator and launches background loop if not active."""
        with self._lock:
            self.simulator.start()
            self._ensure_background_task()
            status = self.get_status()

        self.event_service.emit_event_sync(
            event_type=SystemEventType.SIMULATION_STATUS,
            component="SimulatorService",
            payload={"status": self.simulator.status, "cursor": self.simulator.cursor},
            model_version=status.model_version,
            scenario_id=status.scenario_id,
        )
        return status

    def pause(self) -> SimulationStatusResponse:
        """Pauses the simulator."""
        with self._lock:
            self.simulator.pause()
            status = self.get_status()

        self.event_service.emit_event_sync(
            event_type=SystemEventType.SIMULATION_STATUS,
            component="SimulatorService",
            payload={"status": self.simulator.status, "cursor": self.simulator.cursor},
            model_version=status.model_version,
            scenario_id=status.scenario_id,
        )
        return status

    def resume(self) -> SimulationStatusResponse:
        """Resumes the simulator from paused state."""
        with self._lock:
            self.simulator.resume()
            self._ensure_background_task()
            status = self.get_status()

        self.event_service.emit_event_sync(
            event_type=SystemEventType.SIMULATION_STATUS,
            component="SimulatorService",
            payload={"status": self.simulator.status, "cursor": self.simulator.cursor},
            model_version=status.model_version,
            scenario_id=status.scenario_id,
        )
        return status

    def stop(self) -> SimulationStatusResponse:
        """Stops the simulator."""
        with self._lock:
            self.simulator.stop()
            if self._playback_task and not self._playback_task.done():
                self._playback_task.cancel()
            status = self.get_status()

        self.event_service.emit_event_sync(
            event_type=SystemEventType.SIMULATION_STATUS,
            component="SimulatorService",
            payload={"status": self.simulator.status, "cursor": self.simulator.cursor},
            model_version=status.model_version,
            scenario_id=status.scenario_id,
        )
        return status

    def reset(self) -> SimulationStatusResponse:
        """Resets the simulator to cursor 0 and clears buffers."""
        with self._lock:
            if self._playback_task and not self._playback_task.done():
                self._playback_task.cancel()
            self.simulator.reset()
            self.monitoring_service.reset()
            self._recent_predictions_buffer.clear()
            self._metrics_history.clear()
            self._history_window_counter = 0
            status = self.get_status()

        self.event_service.emit_event_sync(
            event_type=SystemEventType.SIMULATION_STATUS,
            component="SimulatorService",
            payload={"status": self.simulator.status, "cursor": 0},
            model_version=status.model_version,
            scenario_id=status.scenario_id,
        )
        return status

    def seek(self, index: int) -> SimulationStatusResponse:
        """Seeks replay to specific index."""
        with self._lock:
            self.simulator.seek(index)
            status = self.get_status()
        return status

    def set_speed(self, speed: float) -> SimulationStatusResponse:
        """Updates playback speed."""
        with self._lock:
            self.simulator.set_speed(speed)
            status = self.get_status()
        return status

    def step(self, n: int = 1) -> List[PredictionItem]:
        """Manually steps the simulator by n records."""
        with self._lock:
            step_results = self.simulator.step(n)
            emitted_items = self._process_step_results(step_results)
        return emitted_items

    def activate_scenario(self, scenario_id: str, intensity: float = 0.75) -> Dict[str, Any]:
        """Swaps active scenario in the simulator and updates monitoring context."""
        with self._lock:
            scenario_obj = ScenarioEngine.get_scenario(scenario_id, intensity=intensity)
            self.simulator.scenario = scenario_obj
            self.simulator.config.scenario_id = scenario_id
            self.simulator.config.scenario_intensity = intensity
            
            self.monitoring_service.set_scenario_context(
                scenario_id=scenario_obj.scenario_id,
                scenario_name=scenario_obj.name,
                synthetic_disturbance=scenario_obj.is_synthetic,
                intensity=scenario_obj.intensity,
            )

        logger.info(f"Activated scenario '{scenario_obj.name}' (intensity={intensity})")
        return {
            "scenario_id": scenario_obj.scenario_id,
            "scenario_name": scenario_obj.name,
            "intensity": scenario_obj.intensity,
            "is_synthetic": scenario_obj.is_synthetic,
        }

    # -------------------------------------------------------------
    # Step Processing & Event Propagation
    # -------------------------------------------------------------
    def _process_step_results(self, step_results: List[Any]) -> List[PredictionItem]:
        """Processes step events, feeds drift detector, and broadcasts to EventService."""
        new_predictions: List[PredictionItem] = []

        for telemetry, pred, outcome, eval_event in step_results:
            if pred is None:
                continue

            # Model item
            pred_item = PredictionItem(
                prediction_id=pred.prediction_id,
                event_id=pred.event_id,
                trip_id=pred.trip_id,
                deviceid=pred.deviceid,
                direction=pred.direction,
                segment=pred.segment,
                predicted_eta_sec=pred.predicted_eta_sec,
                raw_prediction_sec=pred.raw_prediction_sec,
                actual_eta_sec=outcome.actual_eta_sec if outcome else None,
                signed_error_sec=eval_event.signed_error_sec if eval_event else None,
                absolute_error_sec=eval_event.absolute_error_sec if eval_event else None,
                percentage_error=eval_event.percentage_error if eval_event else None,
                prediction_timestamp_utc=pred.prediction_timestamp_utc,
                model_id=pred.model_id,
                model_version=pred.model_version,
                algorithm=pred.algorithm,
                latency_ms=pred.latency_ms,
                scenario_id=pred.scenario_id,
                synthetic_disturbance=pred.synthetic_disturbance,
            )
            new_predictions.append(pred_item)
            self._recent_predictions_buffer.append(pred_item)
            if len(self._recent_predictions_buffer) > 1000:
                self._recent_predictions_buffer.pop(0)

            # Emit Prediction Event
            self.event_service.emit_event_sync(
                event_type=SystemEventType.PREDICTION,
                component="ETAPredictor",
                payload=pred_item.model_dump(),
                model_version=pred.model_version,
                scenario_id=pred.scenario_id,
            )

            # If outcome is resolved, process drift monitoring & emit outcome event
            if outcome and eval_event:
                self.event_service.emit_event_sync(
                    event_type=SystemEventType.OUTCOME_RESOLVED,
                    component="DigitalTransitSimulator",
                    payload={
                        "outcome_id": outcome.outcome_id,
                        "prediction_id": pred.prediction_id,
                        "deviceid": outcome.deviceid,
                        "trip_id": outcome.trip_id,
                        "actual_eta_sec": outcome.actual_eta_sec,
                        "signed_error_sec": eval_event.signed_error_sec,
                        "absolute_error_sec": eval_event.absolute_error_sec,
                    },
                    model_version=pred.model_version,
                    scenario_id=pred.scenario_id,
                )

                # Feed Drift Engine
                alerts = self.monitoring_service.process_observation(
                    record_idx=self.simulator.cursor,
                    features=telemetry.features,
                    predicted_eta=pred.predicted_eta_sec,
                    actual_eta=outcome.actual_eta_sec,
                    timestamp_utc=telemetry.event_timestamp_utc,
                    model_version=pred.model_version,
                )

                for alert in alerts:
                    self.event_service.emit_event_sync(
                        event_type=SystemEventType.DRIFT_DETECTED,
                        component="DriftEngine",
                        payload=alert.model_dump(),
                        model_version=pred.model_version,
                        scenario_id=pred.scenario_id,
                    )

        # Periodically record rolling metrics snapshot
        if len(self._recent_predictions_buffer) > 0 and self.simulator.cursor % 50 == 0:
            self._record_metrics_snapshot()

        return new_predictions

    def _record_metrics_snapshot(self) -> None:
        """Captures a metrics history entry."""
        if not self.simulator.absolute_errors:
            return
        
        self._history_window_counter += 1
        summary = self.simulator.get_performance_summary()
        om = summary["online_metrics"]
        
        item = MetricsHistoryItem(
            window_index=self._history_window_counter,
            sample_count=len(self.simulator.absolute_errors),
            mae_sec=om["mae_sec"],
            rmse_sec=om["rmse_sec"],
            median_ae_sec=om["median_ae_sec"],
            p90_error_sec=om["p90_error_sec"],
            scenario_id=self.simulator.scenario.scenario_id,
            model_version=self.simulator.predictor.metadata.get("version", "v1.0.0") if self.simulator.predictor else "v1.0.0",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._metrics_history.append(item)
        if len(self._metrics_history) > 200:
            self._metrics_history.pop(0)

    # -------------------------------------------------------------
    # Background Async Loop
    # -------------------------------------------------------------
    def _ensure_background_task(self) -> None:
        """Launches asyncio background playback runner if loop exists."""
        try:
            loop = asyncio.get_running_loop()
            if self._playback_task is None or self._playback_task.done():
                self._playback_task = loop.create_task(self._background_playback_loop())
        except RuntimeError:
            pass  # No running event loop

    async def _background_playback_loop(self) -> None:
        """Async playback loop advancing the simulator at configured replay speed."""
        logger.info("Background simulation playback worker started.")
        try:
            while self.simulator.status == "RUNNING" and not self.simulator.is_finished():
                # Step 1 record
                with self._lock:
                    step_res = self.simulator.step(1)
                    self._process_step_results(step_res)

                # Control tick delay based on replay speed (default 1.0 = ~0.2s delay)
                delay = 0.2 / max(self.simulator.replay_speed, 0.1)
                await asyncio.sleep(min(delay, 2.0))

            if self.simulator.is_finished():
                self.simulator.status = "COMPLETED"
                logger.info("Background simulation completed stream.")
        except asyncio.CancelledError:
            logger.info("Background simulation worker stopped.")
        except Exception as e:
            logger.error(f"Error in background simulation worker: {e}", exc_info=True)

    # -------------------------------------------------------------
    # State & Query APIs
    # -------------------------------------------------------------
    def get_status(self) -> SimulationStatusResponse:
        """Returns standard simulator status response."""
        state = self.simulator.get_state()
        return SimulationStatusResponse(**state)

    def get_fleet(self) -> List[BusStateResponse]:
        """Returns active bus fleet state."""
        buses = []
        with self._lock:
            for dev_id, bus in self.simulator.active_buses.items():
                buses.append(
                    BusStateResponse(
                        deviceid=bus.deviceid,
                        trip_id=bus.trip_id,
                        direction=bus.direction,
                        current_segment=bus.current_segment,
                        last_event_timestamp_utc=bus.last_event_timestamp_utc,
                        last_simulation_timestamp_utc=bus.last_simulation_timestamp_utc,
                        latitude=None,
                        longitude=None,
                        predicted_eta_sec=bus.last_predicted_eta_sec,
                        actual_eta_sec=bus.last_actual_eta_sec,
                        prediction_error_sec=bus.last_error_sec,
                        model_id=self.simulator.predictor.metadata.get("model_id", "model_lightgbm_v1") if self.simulator.predictor else None,
                        model_version=self.simulator.predictor.metadata.get("version", "v1.0.0") if self.simulator.predictor else None,
                        scenario_id=self.simulator.scenario.scenario_id,
                        scenario_name=self.simulator.scenario.name,
                        disturbance_intensity=self.simulator.scenario.intensity,
                        status=bus.status,
                    )
                )
        return buses

    def get_bus_state(self, bus_id: str) -> Optional[BusStateResponse]:
        """Returns state for specific bus ID."""
        with self._lock:
            bus = self.simulator.active_buses.get(bus_id)
            if not bus:
                return None
            return BusStateResponse(
                deviceid=bus.deviceid,
                trip_id=bus.trip_id,
                direction=bus.direction,
                current_segment=bus.current_segment,
                last_event_timestamp_utc=bus.last_event_timestamp_utc,
                last_simulation_timestamp_utc=bus.last_simulation_timestamp_utc,
                latitude=None,
                longitude=None,
                predicted_eta_sec=bus.last_predicted_eta_sec,
                actual_eta_sec=bus.last_actual_eta_sec,
                prediction_error_sec=bus.last_error_sec,
                model_id=self.simulator.predictor.metadata.get("model_id", "model_lightgbm_v1") if self.simulator.predictor else None,
                model_version=self.simulator.predictor.metadata.get("version", "v1.0.0") if self.simulator.predictor else None,
                scenario_id=self.simulator.scenario.scenario_id,
                scenario_name=self.simulator.scenario.name,
                disturbance_intensity=self.simulator.scenario.intensity,
                status=bus.status,
            )

    def get_predictions(self, limit: int = 50, trip_id: Optional[str] = None, deviceid: Optional[str] = None) -> List[PredictionItem]:
        """Returns recent predictions matching query criteria."""
        with self._lock:
            preds = list(self._recent_predictions_buffer)
        
        if trip_id:
            preds = [p for p in preds if p.trip_id == trip_id]
        if deviceid:
            preds = [p for p in preds if p.deviceid == deviceid]
        
        return preds[-limit:]

    def get_prediction_by_id(self, prediction_id: str) -> Optional[PredictionItem]:
        """Finds prediction by prediction_id."""
        with self._lock:
            for p in reversed(self._recent_predictions_buffer):
                if p.prediction_id == prediction_id:
                    return p
        return None

    def get_metrics(self) -> MetricsResponse:
        """Calculates current online error metrics."""
        with self._lock:
            summary = self.simulator.get_performance_summary()
            om = summary["online_metrics"]
            
            bias = 0.0
            if len(self.simulator.signed_errors) > 0:
                bias = round(float(sum(self.simulator.signed_errors) / len(self.simulator.signed_errors)), 2)

            prod_id = "model_lightgbm_v1"
            prod_ver = "v1.0.0"
            if self.simulator.predictor and self.simulator.predictor.metadata:
                prod_id = self.simulator.predictor.metadata.get("model_id", prod_id)
                prod_ver = self.simulator.predictor.metadata.get("version", prod_ver)

            return MetricsResponse(
                simulation_id=summary["simulation_id"],
                status=summary["status"],
                active_model_id=prod_id,
                active_model_version=prod_ver,
                records_emitted=summary["records_emitted"],
                predictions_generated=summary["predictions_generated"],
                outcomes_resolved=summary["outcomes_resolved"],
                affected_events=summary["affected_events"],
                unaffected_events=summary["unaffected_events"],
                online_metrics=OnlineMetrics(
                    mae_sec=om["mae_sec"],
                    rmse_sec=om["rmse_sec"],
                    median_ae_sec=om["median_ae_sec"],
                    p90_error_sec=om["p90_error_sec"],
                    p95_error_sec=om["p95_error_sec"],
                    mean_signed_bias_sec=bias,
                    avg_inference_latency_ms=om["avg_inference_latency_ms"],
                ),
            )

    def get_metrics_history(self) -> MetricsHistoryResponse:
        """Returns time/window based error metrics history."""
        with self._lock:
            return MetricsHistoryResponse(history=list(self._metrics_history))
