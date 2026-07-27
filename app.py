"""Streamlit app: upload a test CSV, pick a model, see how it does.

Only ever scores the CSV the user uploads (default: test_data.csv from this
repo) - it never touches the training data, so every number on screen is a
live evaluation on whatever file is currently loaded.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR / "model"))
from preprocessing import TARGET_COLUMN, CLASS_NAMES, get_feature_columns  # noqa: E402
from metrics_utils import compute_metrics, confusion_and_report  # noqa: E402

MODEL_DIR = APP_DIR / "model" / "saved"
METRICS_PATH = APP_DIR / "model" / "metrics.json"
SAMPLE_DATA_PATH = APP_DIR / "test_data.csv"

METRIC_LABELS = [
    ("accuracy", "Accuracy"),
    ("auc", "AUC"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("mcc", "MCC"),
]

st.set_page_config(page_title="Student Outcome Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def load_pipeline(model_key: str):
    return joblib.load(MODEL_DIR / f"{model_key}.joblib")


@st.cache_data
def load_benchmark():
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_reference_columns():
    ref = pd.read_csv(SAMPLE_DATA_PATH, nrows=1)
    return get_feature_columns(ref)


benchmark = load_benchmark()
reference_features = load_reference_columns()
model_options = {info["display_name"]: key for key, info in benchmark.items()}

st.title("🎓 Student Outcome Predictor")
st.caption(
    "Predicting whether a student will drop out, stay enrolled, or graduate, "
    "from data available at enrollment plus first/second-semester performance."
)

tab_overview, tab_evaluate = st.tabs(["Overview", "Evaluate a Model"])

with tab_overview:
    st.subheader("About this dataset")
    st.markdown(
        """
        [UCI "Predict Students' Dropout and Academic Success"](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) —
        4,424 students across several undergraduate degrees, 36 features
        (demographics, socio-economic factors, admission data, and first/second-semester
        academic performance), 3-class target: **Dropout**, **Enrolled**, or **Graduate**.

        5 classifiers were trained on an 80/20 stratified split of this dataset:
        Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes,
        and Random Forest. Use the **Evaluate a Model** tab to see how each one performs
        on a held-out test file.
        """
    )
    if SAMPLE_DATA_PATH.exists():
        sample = pd.read_csv(SAMPLE_DATA_PATH)
        st.subheader("Held-out test split shipped with this repo")
        col_counts, col_preview = st.columns([1, 2])
        with col_counts:
            st.bar_chart(sample[TARGET_COLUMN].value_counts().loc[CLASS_NAMES])
        with col_preview:
            st.dataframe(sample.head(5), height=210)

with tab_evaluate:
    col_upload, col_model = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload test data (CSV, same columns as test_data.csv)", type="csv"
        )
    with col_model:
        selected_display_name = st.selectbox("Model", list(model_options.keys()))
    model_key = model_options[selected_display_name]

    if uploaded_file is None:
        st.info("Upload a CSV to evaluate a model — try test_data.csv from this repo.")
    else:
        data = pd.read_csv(uploaded_file)

        missing_target = TARGET_COLUMN not in data.columns
        missing_features = [c for c in reference_features if c not in data.columns]

        if missing_target or missing_features:
            st.error(
                "Uploaded CSV doesn't match the expected format. "
                + (f"Missing target column '{TARGET_COLUMN}'. " if missing_target else "")
                + (f"Missing feature columns: {missing_features}" if missing_features else "")
            )
        else:
            X = data[reference_features]
            y_true = data[TARGET_COLUMN]

            pipeline = load_pipeline(model_key)
            y_pred = pipeline.predict(X)
            y_proba = pipeline.predict_proba(X)
            live_metrics = compute_metrics(y_true, y_pred, y_proba)
            bench_metrics = benchmark[model_key]

            st.subheader(f"{selected_display_name} — results on your uploaded data")
            st.caption(
                f"Computed live on the {len(data)} rows in the file you just uploaded — "
                "not on training data."
            )

            metric_cols = st.columns(6)
            for col, (key, label) in zip(metric_cols, METRIC_LABELS):
                delta = live_metrics[key] - bench_metrics[key]
                col.metric(label, f"{live_metrics[key]:.3f}", delta=f"{delta:+.3f} vs. benchmark")
            st.caption(
                "Delta is against this model's benchmark metrics, computed at training time "
                "on our own held-out split (see model/train_models.ipynb)."
            )

            col_cm, col_report = st.columns(2)
            cm, report = confusion_and_report(y_true, y_pred, CLASS_NAMES)
            with col_cm:
                st.markdown("**Confusion matrix**")
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            with col_report:
                st.markdown("**Classification report**")
                report_df = pd.DataFrame(report).T.round(3)
                st.dataframe(report_df, height=230)
