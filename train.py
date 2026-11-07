"""
Credit Card Default Classification - Assignment 2
Reproducible training script for the five required classifiers.
Run: python train.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "credit_card_default_cleaned.csv"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TARGET = "default payment next month"
ID_COL = "ID"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[ID_COL, TARGET])
y = df[TARGET].astype(int)
numeric = [c for c in X.columns if c not in CATEGORICAL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

def preprocessor(scale_numeric=True):
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("numeric", StandardScaler() if scale_numeric else "passthrough", numeric),
        ],
        remainder="drop"
    )

models = {
    "logistic_regression": Pipeline([
        ("preprocess", preprocessor(True)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ]),
    "decision_tree": Pipeline([
        ("preprocess", preprocessor(False)),
        ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]),
    "knn": Pipeline([
        ("preprocess", preprocessor(True)),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "gaussian_nb": Pipeline([
        ("preprocess", preprocessor(True)),
        ("classifier", GaussianNB())
    ]),
    "svm": Pipeline([
        ("preprocess", preprocessor(True)),
        ("classifier", LinearSVC(dual=False, max_iter=3000, random_state=RANDOM_STATE))
    ]),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    else:
        score = model.decision_function(X_test)

    rows.append({
        "Model": {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree", "knn": "K-Nearest Neighbors", "gaussian_nb": "Gaussian Naive Bayes", "svm": "Support Vector Machine"}[name],
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, score),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    pd.DataFrame(confusion_matrix(y_test, pred),
                 index=["Actual 0", "Actual 1"],
                 columns=["Predicted 0", "Predicted 1"]).to_csv(
                     REPORT_DIR / f"{name}_confusion_matrix.csv")
    pd.DataFrame(classification_report(
        y_test, pred, output_dict=True, zero_division=0
    )).T.to_csv(REPORT_DIR / f"{name}_classification_report.csv")

comparison = pd.DataFrame(rows).set_index("Model")
comparison.to_csv(REPORT_DIR / "model_comparison.csv")
(REPORT_DIR / "model_metrics.json").write_text(
    json.dumps(comparison.to_dict(orient="index"), indent=2)
)
print(comparison.round(4))
