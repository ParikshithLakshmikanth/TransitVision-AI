"""TransitVision AI - Backend API & WebSocket Test Suite.
Tests all FastAPI REST endpoints, WebSocket streaming, simulation controls,
model registry inspection, drift reporting, and error handling.
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.dependencies import get_simulator_service, get_event_service


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient for FastAPI app with lifespan events."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    """Verifies root status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "TransitVision AI"
    assert data["status"] == "ONLINE"
    assert "/docs" in data["docs"]


def test_health_endpoint(client: TestClient):
    """Verifies GET /api/health returns real system status and production model."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["HEALTHY", "DEGRADED"]
    assert data["production_model_id"] == "model_lightgbm_v1"
    assert data["production_model_version"] == "v1.0.0"
    assert "digital_transit_simulator" in data["services"]
    assert "eta_predictor" in data["services"]
    assert "drift_engine" in data["services"]
    assert "model_registry" in data["services"]
    assert data["services"]["digital_transit_simulator"]["available"] is True


def test_fleet_endpoint_and_bus_lookup(client: TestClient):
    """Verifies GET /api/fleet and GET /api/fleet/{bus_id}."""
    # Step simulator first so fleet has active buses
    step_resp = client.post("/api/simulation/step", json={"n": 10})
    assert step_resp.status_code == 200

    response = client.get("/api/fleet")
    assert response.status_code == 200
    data = response.json()
    assert "total_active_buses" in data
    assert "buses" in data
    assert len(data["buses"]) >= 1

    first_bus = data["buses"][0]
    bus_id = first_bus["deviceid"]
    assert bus_id is not None
    assert first_bus["status"] == "ACTIVE"

    # Query specific bus
    bus_resp = client.get(f"/api/fleet/{bus_id}")
    assert bus_resp.status_code == 200
    bus_data = bus_resp.json()
    assert bus_data["deviceid"] == bus_id
    assert bus_data["current_segment"] >= 1

    # Query non-existent bus -> 404
    non_existent = client.get("/api/fleet/NON_EXISTENT_BUS_999")
    assert non_existent.status_code == 404


def test_predictions_endpoint(client: TestClient):
    """Verifies GET /api/predictions and GET /api/predictions/{prediction_id}."""
    response = client.get("/api/predictions?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) > 0

    first_pred = data["predictions"][0]
    pred_id = first_pred["prediction_id"]
    assert first_pred["predicted_eta_sec"] > 0
    assert first_pred["model_id"] == "model_lightgbm_v1"
    assert first_pred["model_version"] == "v1.0.0"

    # Query by ID
    single_resp = client.get(f"/api/predictions/{pred_id}")
    assert single_resp.status_code == 200
    single_data = single_resp.json()
    assert single_data["prediction_id"] == pred_id

    # Non-existent ID -> 404
    missing_resp = client.get("/api/predictions/NON_EXISTENT_PRED_999")
    assert missing_resp.status_code == 404


def test_metrics_endpoint(client: TestClient):
    """Verifies GET /api/metrics and GET /api/metrics/history."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "online_metrics" in data
    assert data["active_model_id"] == "model_lightgbm_v1"
    assert data["online_metrics"]["mae_sec"] >= 0.0

    hist_resp = client.get("/api/metrics/history")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert "history" in hist_data


def test_drift_endpoints(client: TestClient):
    """Verifies GET /api/drift, GET /api/drift/summary, and GET /api/drift/events."""
    resp = client.get("/api/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "is_data_drift_active" in data
    assert "summary" in data

    summary_resp = client.get("/api/drift/summary")
    assert summary_resp.status_code == 200
    summary_data = summary_resp.json()
    assert "total_records_monitored" in summary_data

    events_resp = client.get("/api/drift/events?limit=20")
    assert events_resp.status_code == 200
    assert isinstance(events_resp.json(), list)


def test_models_registry_endpoints(client: TestClient):
    """Verifies GET /api/models, GET /api/models/production, and GET /api/models/{model_id}."""
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["production_model_id"] == "model_lightgbm_v1"
    assert data["total_models"] >= 5

    prod_resp = client.get("/api/models/production")
    assert prod_resp.status_code == 200
    prod_data = prod_resp.json()
    assert prod_data["model_id"] == "model_lightgbm_v1"
    assert prod_data["version"] == "v1.0.0"
    assert prod_data["status"] == "PRODUCTION"

    # Specific model
    cand_resp = client.get("/api/models/model_lightgbm_v1")
    assert cand_resp.status_code == 200
    assert cand_resp.json()["model_id"] == "model_lightgbm_v1"

    # Non-existent model -> 404
    missing = client.get("/api/models/non_existent_model_id")
    assert missing.status_code == 404


def test_simulation_controls(client: TestClient):
    """Verifies simulation control endpoints (start, pause, resume, speed, seek, reset, stop)."""
    # 1. Status
    st_resp = client.get("/api/simulation/status")
    assert st_resp.status_code == 200
    assert st_resp.json()["total_records"] > 0

    # 2. Pause
    p_resp = client.post("/api/simulation/pause")
    assert p_resp.status_code == 200
    assert p_resp.json()["simulator_state"]["status"] == "PAUSED"

    # 3. Resume
    r_resp = client.post("/api/simulation/resume")
    assert r_resp.status_code == 200
    assert r_resp.json()["simulator_state"]["status"] == "RUNNING"

    # 4. Speed
    sp_resp = client.post("/api/simulation/speed", json={"speed": 5.0})
    assert sp_resp.status_code == 200
    assert sp_resp.json()["simulator_state"]["replay_speed"] == 5.0

    # 5. Seek
    seek_resp = client.post("/api/simulation/seek", json={"index": 100})
    assert seek_resp.status_code == 200
    assert seek_resp.json()["simulator_state"]["cursor"] == 100

    # 6. Seek out of bounds -> 400
    bad_seek = client.post("/api/simulation/seek", json={"index": 9999999})
    assert bad_seek.status_code == 400

    # 7. Stop
    stop_resp = client.post("/api/simulation/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["simulator_state"]["status"] == "STOPPED"

    # 8. Reset
    reset_resp = client.post("/api/simulation/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["simulator_state"]["cursor"] == 0


def test_scenarios_endpoint(client: TestClient):
    """Verifies GET /api/scenarios and POST /api/scenarios/activate."""
    resp = client.get("/api/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert "available_scenarios" in data
    scenario_ids = [s["scenario_id"] for s in data["available_scenarios"]]
    assert "BASELINE" in scenario_ids
    assert "RUSH_HOUR" in scenario_ids
    assert "HEAVY_RAIN" in scenario_ids

    # Activate valid scenario
    act_resp = client.post("/api/scenarios/activate", json={"scenario_id": "RUSH_HOUR", "intensity": 0.75})
    assert act_resp.status_code == 200
    assert act_resp.json()["scenario_id"] == "RUSH_HOUR"
    assert act_resp.json()["intensity"] == 0.75

    # Reset to baseline
    reset_act = client.post("/api/scenarios/activate", json={"scenario_id": "BASELINE", "intensity": 0.0})
    assert reset_act.status_code == 200
    assert reset_act.json()["scenario_id"] == "BASELINE"

    # Invalid scenario -> 400
    bad_sc = client.post("/api/scenarios/activate", json={"scenario_id": "INVALID_SCENARIO", "intensity": 0.5})
    assert bad_sc.status_code == 400


def test_lifecycle_and_retraining_endpoints(client: TestClient):
    """Verifies GET /api/lifecycle, GET /api/retraining/status, and GET /api/rollback/status."""
    lc_resp = client.get("/api/lifecycle")
    assert lc_resp.status_code == 200
    lc_data = lc_resp.json()
    assert lc_data["production_model_id"] == "model_lightgbm_v1"
    assert lc_data["production_model_version"] == "v1.0.0"
    assert "recent_lifecycle_events" in lc_data

    rt_resp = client.get("/api/retraining/status")
    assert rt_resp.status_code == 200
    assert rt_resp.json()["active_production_model_id"] == "model_lightgbm_v1"

    hist_resp = client.get("/api/retraining/history")
    assert hist_resp.status_code == 200

    # Retraining trigger dry run
    trig_resp = client.post("/api/retraining/trigger", json={"candidate_version": "v1.2.0-test", "dry_run": True})
    assert trig_resp.status_code == 200
    assert trig_resp.json()["status"] == "EVALUATED_DRY_RUN"

    # Retraining trigger non-dry run -> 403
    forbidden_trig = client.post("/api/retraining/trigger", json={"candidate_version": "v1.2.0-test", "dry_run": False})
    assert forbidden_trig.status_code == 403

    rb_resp = client.get("/api/rollback/status")
    assert rb_resp.status_code == 200
    rb_data = rb_resp.json()
    assert rb_data["max_degradation_ratio"] == 1.25
    assert rb_data["pre_promotion_baseline_mae_sec"] == 38.77


def test_events_endpoint(client: TestClient):
    """Verifies GET /api/events."""
    resp = client.get("/api/events?limit=20")
    assert resp.status_code == 200
    events = resp.json()
    assert isinstance(events, list)


def test_websocket_live_stream(client: TestClient):
    """Verifies WebSocket connection, initial message, ping/pong, and broadcast events."""
    with client.websocket_connect("/ws/live") as websocket:
        # Receive connection message
        data = websocket.receive_json()
        assert data["type"] == "CONNECTION_ESTABLISHED"

        # Test heartbeat
        websocket.send_text("ping")
        pong = websocket.receive_text()
        assert pong == "pong"

        # Trigger a step to generate broadcast events
        client.post("/api/simulation/step", json={"n": 2})

        # Receive streamed prediction / status event
        event_msg = websocket.receive_json()
        assert "event_type" in event_msg
        assert "payload" in event_msg
        assert event_msg["model_version"] == "v1.0.0"


def test_validation_error_handler(client: TestClient):
    """Verifies 422 error on invalid schema input."""
    resp = client.post("/api/simulation/speed", json={"speed": "invalid_number_type"})
    assert resp.status_code == 422
    assert "Validation Error" in resp.json()["error"]
