import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import PROCESSED_DATA_DIR, METADATA_DIR

from ml.preprocessing.pipeline_preprocessor import ALL_FEATURE_COLUMNS
from simulator.event_models import SimulationConfig
from simulator.transit_simulator import DigitalTransitSimulator
from monitoring.drift_models import DriftConfig, DriftType, DriftSeverity
from monitoring.drift_detectors import ReferenceProfiler, FeatureDriftDetector
from monitoring.drift_engine import DriftEngine
from monitoring.statistical_tests import calculate_psi, calculate_ks_test, calculate_js_divergence

train_df = pd.read_parquet(PROCESSED_DATA_DIR / "kandy_eta_training.parquet")
stream_df = pd.read_parquet(PROCESSED_DATA_DIR / "kandy_eta_stream.parquet")

print("--- DATASET DATE RANGES ---")
print("Training records:", len(train_df), "From:", train_df["timestamp_utc"].min(), "To:", train_df["timestamp_utc"].max())
print("Stream   records:", len(stream_df), "From:", stream_df["timestamp_utc"].min(), "To:", stream_df["timestamp_utc"].max())

profiler = ReferenceProfiler()
ref_profile = profiler.build_profile()

# Window-by-window analysis of baseline stream
window_size = 500
step_size = 250
num_windows = (len(stream_df) - window_size) // step_size + 1

feature_drift_tracking = {col: {"psi_values": [], "ks_stats": [], "ks_pvals": [], "js_divs": [], "flagged_count": 0} for col in ALL_FEATURE_COLUMNS}

print(f"\n--- EVALUATING {num_windows} WINDOWS ON RAW BASELINE STREAM ---")
detector = FeatureDriftDetector(reference_profile=ref_profile)

for w_idx in range(num_windows):
    start = w_idx * step_size
    end = start + window_size
    w_df = stream_df.iloc[start:end]
    results = detector.evaluate_window(w_df)

    for r in results:
        feat = r.feature_name
        feature_drift_tracking[feat]["psi_values"].append(r.statistic)
        if "ks_statistic" in r.metadata:
            feature_drift_tracking[feat]["ks_stats"].append(r.metadata["ks_statistic"])
            feature_drift_tracking[feat]["ks_pvals"].append(r.metadata["ks_p_value"] or 1.0)
            feature_drift_tracking[feat]["js_divs"].append(r.metadata["js_divergence"])
        if r.is_drift:
            feature_drift_tracking[feat]["flagged_count"] += 1

# Rank features by max PSI
ranked_features = []
for col, d in feature_drift_tracking.items():
    psis = d["psi_values"]
    ks_s = d["ks_stats"]
    ks_p = d["ks_pvals"]
    js_s = d["js_divs"]
    ranked_features.append({
        "feature": col,
        "psi_max": round(float(np.max(psis)), 4) if psis else 0.0,
        "psi_median": round(float(np.median(psis)), 4) if psis else 0.0,
        "ks_max": round(float(np.max(ks_s)), 4) if ks_s else 0.0,
        "ks_min_pval": float(np.min(ks_p)) if ks_p else 1.0,
        "js_max": round(float(np.max(js_s)), 4) if js_s else 0.0,
        "windows_flagged": d["flagged_count"]
    })

ranked_features.sort(key=lambda x: (x["windows_flagged"], x["psi_max"]), reverse=True)
df_ranked = pd.DataFrame(ranked_features)
print(df_ranked.to_string(index=False))

# Now check Concept Drift on Baseline
print("\n--- BASELINE CONCEPT DRIFT INVESTIGATION ---")
engine = DriftEngine(config=DriftConfig(window_size=500, step_size=250), reference_profile=ref_profile)
sim = DigitalTransitSimulator(config=SimulationConfig(scenario_id="baseline", auto_predict=True))
sim.start()

batch_size = 500
while sim.cursor < sim.total_records and sim.status == "RUNNING":
    step_results = sim.step(min(batch_size, sim.total_records - sim.cursor))
    for telemetry, pred, outcome, eval_event in step_results:
        cur_idx = int(telemetry.event_id.split("_")[-1])
        engine.process_record(
            record_idx=cur_idx,
            features=telemetry.features,
            predicted_eta=pred.predicted_eta_sec,
            actual_eta=outcome.actual_eta_sec,
            timestamp_utc=telemetry.event_timestamp_utc
        )
engine.flush()

concept_events = [e for e in engine.events_history if e.drift_type == DriftType.CONCEPT_DRIFT]
print(f"Total Concept Drift Events on Baseline: {len(concept_events)}")
for i, ce in enumerate(concept_events):
    print(f"Alert {i+1}: EventID={ce.event_id}, Detector={ce.detector}, WindowEnd={ce.window_end_idx}, Statistic={ce.statistic}, Threshold={ce.threshold}, Severity={ce.severity}, Evidence={ce.evidence}")
