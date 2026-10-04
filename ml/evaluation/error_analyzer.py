"""TransitVision AI - Model Error Analyzer.
Performs stratified error analysis across transit segments, directions, time intervals,
weather conditions, congestion regimes, and travel-time buckets.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from ml.evaluation.evaluator import ModelEvaluator

logger = logging.getLogger("TransitVision.ErrorAnalyzer")


class ModelErrorAnalyzer:
    """Detailed segmented error analysis engine."""

    def __init__(self, evaluator: ModelEvaluator = None):
        self.evaluator = evaluator or ModelEvaluator()

    def analyze(self, df_val: pd.DataFrame, y_pred: np.ndarray, model_name: str = "production_model") -> Dict[str, Any]:
        """
        Calculates stratified error metrics across operational transit dimensions.
        """
        logger.info(f"Running comprehensive error analysis for {model_name} on {len(df_val)} records...")
        analysis_df = df_val.copy()
        analysis_df["predicted_eta_sec"] = y_pred
        analysis_df["abs_error_sec"] = np.abs(analysis_df["eta_to_next_stop_sec"] - analysis_df["predicted_eta_sec"])
        analysis_df["pct_error"] = (analysis_df["abs_error_sec"] / analysis_df["eta_to_next_stop_sec"].clip(lower=10.0)) * 100.0

        report: Dict[str, Any] = {
            "model_name": model_name,
            "analysis_generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "overall_metrics": self.evaluator.calculate_metrics(analysis_df["eta_to_next_stop_sec"], y_pred),
            "by_direction": {},
            "by_segment": {},
            "by_hour": {},
            "by_day_of_week": {},
            "by_peak_period": {},
            "by_weather_condition": {},
            "by_precipitation_tier": {},
            "by_congestion_tier": {},
            "by_target_range": {}
        }

        # 1. By Direction
        for dir_val, grp in analysis_df.groupby("direction"):
            dir_name = "Outbound (Kandy -> Digana)" if int(dir_val) == 1 else "Inbound (Digana -> Kandy)"
            report["by_direction"][dir_name] = {
                "records": len(grp),
                "mae_sec": round(float(grp["abs_error_sec"].mean()), 2),
                "rmse_sec": round(float(np.sqrt((grp["abs_error_sec"] ** 2).mean())), 2),
                "median_ae_sec": round(float(grp["abs_error_sec"].median()), 2),
                "p90_error_sec": round(float(np.percentile(grp["abs_error_sec"], 90)), 2),
                "mape_pct": round(float(grp["pct_error"].mean()), 2)
            }

        # 2. By Segment (top segments & all segments)
        for seg_val, grp in analysis_df.groupby("segment"):
            report["by_segment"][f"Segment_{int(seg_val)}"] = {
                "records": len(grp),
                "mae_sec": round(float(grp["abs_error_sec"].mean()), 2),
                "rmse_sec": round(float(np.sqrt((grp["abs_error_sec"] ** 2).mean())), 2),
                "median_ae_sec": round(float(grp["abs_error_sec"].median()), 2),
                "mean_actual_sec": round(float(grp["eta_to_next_stop_sec"].mean()), 2),
                "mean_pred_sec": round(float(grp["predicted_eta_sec"].mean()), 2)
            }

        # 3. By Hour
        for hr_val, grp in analysis_df.groupby("hour"):
            report["by_hour"][f"{int(hr_val):02d}:00"] = {
                "records": len(grp),
                "mae_sec": round(float(grp["abs_error_sec"].mean()), 2),
                "rmse_sec": round(float(np.sqrt((grp["abs_error_sec"] ** 2).mean())), 2),
                "median_ae_sec": round(float(grp["abs_error_sec"].median()), 2),
                "p90_error_sec": round(float(np.percentile(grp["abs_error_sec"], 90)), 2)
            }

        # 4. By Day of Week (0=Monday, 6=Sunday)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for dow_val, grp in analysis_df.groupby("day_of_week"):
            day_name = days[int(dow_val)] if int(dow_val) < len(days) else f"Day_{dow_val}"
            report["by_day_of_week"][day_name] = {
                "records": len(grp),
                "mae_sec": round(float(grp["abs_error_sec"].mean()), 2),
                "rmse_sec": round(float(np.sqrt((grp["abs_error_sec"] ** 2).mean())), 2)
            }

        # 5. Peak vs Off-Peak
        for peak_val, grp in analysis_df.groupby("is_peak_period"):
            label = "Peak Period" if int(peak_val) == 1 else "Off-Peak"
            report["by_peak_period"][label] = {
                "records": len(grp),
                "mae_sec": round(float(grp["abs_error_sec"].mean()), 2),
                "rmse_sec": round(float(np.sqrt((grp["abs_error_sec"] ** 2).mean())), 2),
                "median_ae_sec": round(float(grp["abs_error_sec"].median()), 2)
            }

        # 6. Weather Condition
        def map_weather(code):
            code = int(code) if not pd.isna(code) else 0
            if code == 0:
                return "Clear Sky"
            elif code in [1, 2, 3]:
                return "Mainly Clear / Overcast"
            elif code in [45, 48]:
                return "Fog"
            elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
                return "Rain / Shower"
            elif code in [95, 96, 99]:
                return "Thunderstorm"
            return f"Weather Code {code}"

        analysis_df["weather_name"] = analysis_df["weather_code"].apply(map_weather)
        for w_val, grp in analysis_df.groupby("weather_name"):
            report["by_weather_condition"][w_val] = {
                "records": len(grp),
                "mae_sec": round(float(grp["abs_error_sec"].mean()), 2),
                "rmse_sec": round(float(np.sqrt((grp["abs_error_sec"] ** 2).mean())), 2)
            }

        # 7. Precipitation Tier
        analysis_df["precip_tier"] = pd.cut(
            analysis_df["precipitation"],
            bins=[-0.01, 0.0, 1.0, 5.0, 100.0],
            labels=["No Rain (0 mm)", "Light Rain (0-1 mm)", "Moderate Rain (1-5 mm)", "Heavy Rain (>5 mm)"]
        )
        for p_tier, grp in analysis_df.groupby("precip_tier", observed=False):
            if len(grp) > 0:
                report["by_precipitation_tier"][str(p_tier)] = {
                    "records": len(grp),
                    "mae_sec": round(float(grp["abs_error_sec"].mean()), 2),
                    "rmse_sec": round(float(np.sqrt((grp["abs_error_sec"] ** 2).mean())), 2)
                }

        # 8. Congestion Tier (Delay Ratio: actual / historical mean)
        analysis_df["congestion_tier"] = pd.cut(
            analysis_df["congestion_proxy"],
            bins=[-0.1, 0.9, 1.3, 100.0],
            labels=["Free Flow (Ratio < 0.9)", "Normal (Ratio 0.9-1.3)", "Congested (Ratio > 1.3)"]
        )
        for c_tier, grp in analysis_df.groupby("congestion_tier", observed=False):
            if len(grp) > 0:
                report["by_congestion_tier"][str(c_tier)] = {
                    "records": len(grp),
                    "mae_sec": round(float(grp["abs_error_sec"].mean()), 2),
                    "rmse_sec": round(float(np.sqrt((grp["abs_error_sec"] ** 2).mean())), 2),
                    "median_ae_sec": round(float(grp["abs_error_sec"].median()), 2)
                }

        # 9. Travel-Time Range Buckets
        analysis_df["target_bucket"] = pd.cut(
            analysis_df["eta_to_next_stop_sec"],
            bins=[0, 60, 120, 240, 480, 10000],
            labels=["Short (< 1 min)", "Medium (1-2 min)", "Standard (2-4 min)", "Long (4-8 min)", "Extreme (> 8 min)"]
        )
        for t_bucket, grp in analysis_df.groupby("target_bucket", observed=False):
            if len(grp) > 0:
                report["by_target_range"][str(t_bucket)] = {
                    "records": len(grp),
                    "mae_sec": round(float(grp["abs_error_sec"].mean()), 2),
                    "rmse_sec": round(float(np.sqrt((grp["abs_error_sec"] ** 2).mean())), 2),
                    "median_ae_sec": round(float(grp["abs_error_sec"].median()), 2),
                    "mape_pct": round(float(grp["pct_error"].mean()), 2)
                }

        return report

    def save_analysis(self, report: Dict[str, Any], filepath: Path) -> None:
        """Saves error analysis json report."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Model error analysis successfully written to {filepath}")
