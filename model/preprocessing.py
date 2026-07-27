"""Feature/target configuration and pipeline builder shared by every model.

Centralized here so the training notebook and the Streamlit app can never
disagree about which columns are features, how the target is encoded, or
whether a given model needs its inputs scaled.
"""

from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Target"
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]

# Logistic Regression and kNN are distance/gradient-based and sensitive to
# feature scale; Decision Tree, Naive Bayes, and Random Forest are not.
SCALED_MODELS = {"logistic_regression", "knn"}


def load_dataset(csv_path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COLUMN]


def build_pipeline(model_key: str, classifier) -> Pipeline:
    steps = []
    if model_key in SCALED_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)
