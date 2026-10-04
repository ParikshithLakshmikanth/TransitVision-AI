"""TransitVision AI - Model Preprocessing Pipeline.
Provides standardized, reproducible transformation of tabular bus telemetry features
for linear and tree-based regression models.
"""
import logging
from typing import List, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger("TransitVision.Preprocessor")

TARGET_COLUMN = "eta_to_next_stop_sec"

CATEGORICAL_FEATURES = ["deviceid", "direction", "segment"]
CATEGORICAL_COLUMNS = CATEGORICAL_FEATURES

NUMERICAL_FEATURES = [
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "is_peak_period",
    "sin_hour",
    "cos_hour",
    "sin_time_of_day",
    "cos_time_of_day",
    "segment_length_km",
    "segments_completed",
    "segments_remaining",
    "trip_progress_ratio",
    "previous_segment_run_time",
    "rolling_prev_segment_mean",
    "rolling_prev_segment_std",
    "cumulative_trip_time_sec",
    "historical_segment_time_mean",
    "segment_delay_ratio",
    "congestion_proxy",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "weather_code"
]
NUMERICAL_COLUMNS = NUMERICAL_FEATURES

ALL_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


class TabularDataPreprocessor:
    """Preprocesses transit features with separate pipelines for linear and tree models."""

    def __init__(self, categorical_cols: List[str] = None, numerical_cols: List[str] = None):
        self.categorical_cols = categorical_cols or CATEGORICAL_FEATURES
        self.numerical_cols = numerical_cols or NUMERICAL_FEATURES
        self.all_feature_cols = self.categorical_cols + self.numerical_cols
        
        # Linear/Ridge preprocessor: OneHot + StandardScaler
        self.linear_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols)
            ],
            remainder="drop"
        )
        
        # Tree preprocessor: Native string/category handling or ordinal casting
        self.is_fitted = False
        self.cat_categories_: Dict[str, List[Any]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Fits transformers on training data."""
        X_sub = X[self.all_feature_cols].copy()
        for col in self.categorical_cols:
            X_sub[col] = X_sub[col].astype(str)
            self.cat_categories_[col] = sorted(list(X_sub[col].unique()))
            
        self.linear_transformer.fit(X_sub)
        self.is_fitted = True
        return self

    def transform_for_linear(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms features for linear/ridge models (scaled numeric + one-hot encoded cat)."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted yet.")
        X_sub = X[self.all_feature_cols].copy()
        for col in self.categorical_cols:
            X_sub[col] = X_sub[col].astype(str)
        return self.linear_transformer.transform(X_sub)

    def transform_for_trees(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms features for tree-based models (LightGBM, HistGradientBoosting, RandomForest).
        Converts categoricals to pandas 'category' dtype for native tree splitting.
        """
        X_sub = X[self.all_feature_cols].copy()
        for col in self.categorical_cols:
            cats = self.cat_categories_.get(col, None)
            val_series = X_sub[col].astype(str)
            if cats is not None:
                val_series = val_series.where(val_series.isin(cats), other=None)
                X_sub[col] = pd.Categorical(val_series, categories=cats)
            else:
                X_sub[col] = pd.Categorical(val_series)
        for col in self.numerical_cols:
            X_sub[col] = pd.to_numeric(X_sub[col], errors="coerce").fillna(0.0)
        return X_sub

    def fit_transform(self, X: pd.DataFrame, y=None, is_tree: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        """Fits preprocessor and returns transformed data."""
        self.fit(X, y)
        return self.transform_for_trees(X) if is_tree else self.transform_for_linear(X)

    def transform(self, X: pd.DataFrame, is_tree: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        """Generic transform dispatcher."""
        return self.transform_for_trees(X) if is_tree else self.transform_for_linear(X)

    def get_feature_names_linear(self) -> List[str]:
        """Returns feature column names after one-hot encoding."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted yet.")
        return list(self.linear_transformer.get_feature_names_out())


TabularPreprocessor = TabularDataPreprocessor
