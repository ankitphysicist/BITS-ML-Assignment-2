"""Reproducibly train all five models and save a held-out test CSV."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from evaluation import evaluate_model

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
SEED = 73
TEST_RATIO = 0.20

def main():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()
    X.assign(target=y).to_csv(ROOT / "dataset.csv", index=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=SEED, stratify=y)
    algorithms = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED)),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=3, random_state=SEED),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=2.0, probability=True, random_state=SEED)),
    }
    file_names = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib",
    }
    rows = []
    for name, estimator in algorithms.items():
        estimator.fit(X_train, y_train)
        joblib.dump(estimator, MODEL_DIR / file_names[name])
        rows.append({"ML Model Name": name, **evaluate_model(estimator, X_test, y_test)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ROOT / "model_comparison.csv", index=False)
    X_test.assign(target=y_test).to_csv(ROOT / "test_data.csv", index=False)
    metadata = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic), UCI",
        "dataset_url": "https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic",
        "samples": len(X), "feature_count": X.shape[1],
        "train_rows": len(X_train), "test_rows": len(X_test),
        "feature_names": list(X.columns), "target_column": "target",
        "target_mapping": {"0": "malignant", "1": "benign"},
        "positive_class": 1, "test_ratio": TEST_RATIO, "random_state": SEED,
        "model_files": file_names,
    }
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Saved 5 models and dataset/test/metadata/comparison files.")

if __name__ == "__main__":
    main()
