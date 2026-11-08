"""Interactive Streamlit evaluation: models are never retrained on uploaded test data."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from evaluation import diagnostic_report, evaluate_model, METRIC_COLUMNS

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Classification Model Lab", page_icon="🔬", layout="wide")
st.title("🔬 Classification Model Lab")
st.caption("BITS AIML learning reference | Five pretrained classifiers | Held-out CSV evaluation")
st.info("Educational classification benchmark only, not medical advice.")

@st.cache_resource
def get_models(file_names):
    return {name: joblib.load(ROOT / "model" / filename)
            for name, filename in file_names}

meta_path = ROOT / "metadata.json"
if not meta_path.exists():
    st.error("Missing metadata.json. Run python train_models.py once.")
    st.stop()
metadata = json.loads(meta_path.read_text(encoding="utf-8"))
feature_names = metadata["feature_names"]
model_names = metadata["model_files"]
missing_models = [filename for filename in model_names.values()
                  if not (ROOT / "model" / filename).exists()]
if missing_models:
    st.error("Model artifacts not found: " + ", ".join(missing_models)
             + ". Run python train_models.py first.")
    st.stop()
models = get_models(tuple(model_names.items()))
with st.sidebar:
    st.header("Experiment setup")
    chosen_name = st.selectbox("Select a trained model", list(model_names))
    uploaded_file = st.file_uploader("Upload test CSV (features + target)", type=["csv"])
    st.caption("Upload unseen test samples only. No training occurs in this app.")
    test_file = ROOT / "test_data.csv"
    if test_file.exists():
        st.download_button("Download example test CSV", data=test_file.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    st.markdown("**Label interpretation:** 0 = malignant; 1 = benign")
try:
    frame = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(test_file)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Unable to read CSV: {exc}")
    st.stop()
required = feature_names + ["target"]
missing = [name for name in required if name not in frame.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()
if frame.empty:
    st.error("CSV has no test records.")
    st.stop()
try:
    X_test = frame.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    y_test = pd.to_numeric(frame["target"], errors="raise")
except (ValueError, TypeError) as exc:
    st.error(f"Features and target must be numeric: {exc}")
    st.stop()
if X_test.isna().any().any() or y_test.isna().any():
    st.error("CSV contains missing values.")
    st.stop()
if not set(y_test.unique()).issubset({0, 1}):
    st.error("Target values must be 0 or 1.")
    st.stop()
st.subheader("Test dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Test records", len(frame))
c2.metric("Input features", len(feature_names))
c3.metric("Trained classifiers", len(models))
st.caption("Uploaded data" if uploaded_file else "Bundled held-out test_data.csv")
with st.expander("Preview test rows"):
    st.dataframe(frame.head(15), use_container_width=True)
st.subheader("Comparison across all five classifiers")
comparison = pd.DataFrame([
    {"ML Model Name": name, **evaluate_model(model, X_test, y_test)}
    for name, model in models.items()
]).set_index("ML Model Name")
st.dataframe(comparison.style.format({m: "{:.4f}" for m in METRIC_COLUMNS}),
             use_container_width=True)
st.caption("AUC is N/A when test data contains only one class. Positive class = benign (1).")
st.subheader(f"Selected model: {chosen_name}")
chosen_model = models[chosen_name]
scores = evaluate_model(chosen_model, X_test, y_test)
for col, metric in zip(st.columns(3) + st.columns(3), METRIC_COLUMNS):
    value = scores[metric]
    col.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")
matrix, report, predictions = diagnostic_report(chosen_model, X_test, y_test)
left, right = st.columns([1, 1])
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Malignant (0)", "Benign (1)"])
    ax.set_yticks([0, 1], ["Malignant (0)", "Benign (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("**Classification report**")
    st.code(report, language="text")
result = frame.copy()
result["predicted_target"] = predictions
st.download_button("Download predictions CSV",
                   result.to_csv(index=False).encode("utf-8"),
                   file_name="predictions.csv", mime="text/csv")
st.caption("Pre-fitted joblib models • no fitting on uploaded CSV")
