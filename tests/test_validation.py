"""Tests for configuration and validator."""
import pytest
import pandas as pd
from config.settings import load_config, CONFIG
from ml.data.validator import TransitDataValidator


def test_config_loading():
    """Verify configuration loads correctly with all required keys."""
    assert "PATHS" in CONFIG
    assert "DATA_SOURCES" in CONFIG
    assert "VALIDATION" in CONFIG
    assert "DRIFT" in CONFIG
    assert "RETRAINING" in CONFIG


def test_validator_detects_clean_and_anomalous_data():
    """Verify TransitDataValidator accurately catches coordinate and speed anomalies."""
    validator = TransitDataValidator(CONFIG)

    # Valid data sample (Dublin coordinates)
    valid_data = pd.DataFrame({
        "Timestamp": [1357041600000000, 1357041620000000],
        "LineID": ["15", "15"],
        "Direction": [1, 1],
        "JourneyPatternID": ["00150001", "00150001"],
        "VehicleJourneyID": ["1001", "1001"],
        "Operator": ["RD", "RD"],
        "Congestion": [0, 1],
        "Longitude": [-6.2603, -6.2590],
        "Latitude": [53.3498, 53.3510],
        "Delay": [120.0, 180.0],
        "BlockID": ["101", "101"],
        "VehicleID": ["4001", "4001"],
        "StopID": ["753", "754"],
        "AtStop": [1, 0]
    })

    report_valid = validator.validate_dataset(valid_data, "test_valid")
    assert report_valid["is_valid"] is True
    assert report_valid["summary"]["out_of_bounds_coords_count"] == 0
    assert report_valid["summary"]["invalid_delay_count"] == 0

    # Anomalous data sample (Coordinates out of Dublin bounds)
    anom_data = valid_data.copy()
    anom_data.loc[0, "Latitude"] = 12.34 # Out of bounds
    anom_data.loc[1, "Delay"] = 99999.0  # Impossible delay

    report_anom = validator.validate_dataset(anom_data, "test_anomalous")
    assert report_anom["summary"]["out_of_bounds_coords_count"] == 1
    assert report_anom["summary"]["invalid_delay_count"] == 1
