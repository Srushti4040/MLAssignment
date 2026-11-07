from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
TARGET = "default payment next month"
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib",
}


@st.cache_resource
def load_model(filename):
    # Load only the trusted model files distributed with this project.
    return joblib.load(ROOT / "model" / filename)


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)

    # AUC is undefined if the uploaded test file contains only one target class.
    auc = roc_auc_score(y, scores) if y.nunique() == 2 else float("nan")
    return {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc,
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }, predictions


st.set_page_config(page_title="Credit Card Default Classifier", page_icon="💳", layout="wide")
st.title("Credit Card Default Classification")
st.caption(
    "Upload a labelled test CSV to evaluate the five pre-trained classifiers. "
    "The application does not fit or retrain models on uploaded data."
)

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded is None:
    st.info("Upload the supplied test_data.csv. It must contain all 23 input features and the target column.")
    st.stop()

try:
    test = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Unable to read the CSV file: {exc}")
    st.stop()

required_columns = FEATURES + [TARGET]
missing_columns = [column for column in required_columns if column not in test.columns]
if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

if test.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

if test[required_columns].isnull().any().any():
    st.error("The required feature and target columns contain missing values. Please correct the CSV and upload it again.")
    st.stop()

try:
    target_values = set(pd.to_numeric(test[TARGET], errors="raise").unique())
except Exception:
    st.error("The target column must contain numeric class labels 0 and 1.")
    st.stop()

if not target_values.issubset({0, 1}):
    st.error("The target column must contain only 0 (no default) and 1 (default).")
    st.stop()

X_test = test[FEATURES]
y_test = pd.to_numeric(test[TARGET]).astype(int)

st.write(f"Uploaded test records: **{len(test):,}**")
st.write("ID, if present, is ignored because it is not a predictive input feature.")

# Evaluate all five models on the same uploaded test rows so their results
# are visible together, as requested in the assignment.
all_results = {}
all_predictions = {}
try:
    for name, filename in MODEL_FILES.items():
        fitted_pipeline = load_model(filename)
        metrics, predictions = evaluate_model(fitted_pipeline, X_test, y_test)
        all_results[name] = metrics
        all_predictions[name] = predictions
except Exception as exc:
    st.error(f"Could not load or evaluate the saved models: {exc}")
    st.stop()

st.subheader("Comparison of all five models on the uploaded test data")
comparison = pd.DataFrame.from_dict(all_results, orient="index")
comparison.index.name = "ML Model Name"
st.dataframe(comparison.style.format("{:.4f}", na_rep="N/A"), use_container_width=True)

selected_model = st.selectbox("Select a model for detailed evaluation", list(MODEL_FILES))
selected_predictions = all_predictions[selected_model]
st.subheader(f"Detailed evaluation: {selected_model}")

selected_metrics = all_results[selected_model]
metric_columns = st.columns(3)
for column, (label, value) in zip(metric_columns * 2, selected_metrics.items()):
    column.metric(label, "N/A" if pd.isna(value) else f"{value:.4f}")

st.subheader("Confusion Matrix")
matrix = confusion_matrix(y_test, selected_predictions, labels=[0, 1])
st.dataframe(
    pd.DataFrame(matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]),
    use_container_width=True,
)

st.subheader("Classification Report")
report = classification_report(
    y_test, selected_predictions, output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
