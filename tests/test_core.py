"""Reproducibility checks for the five saved classifiers."""
from pathlib import Path
import json
import joblib
import pandas as pd
from evaluation import evaluate_model, METRIC_COLUMNS, diagnostic_report

ROOT = Path(__file__).resolve().parents[1]
META = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
TEST_FRAME = pd.read_csv(ROOT / "test_data.csv")

def test_dataset_requirements():
    full = pd.read_csv(ROOT / "dataset.csv")
    assert len(full) >= 500
    assert len(META["feature_names"]) >= 12
    assert set(full["target"].unique()) == {0, 1}
    assert len(TEST_FRAME) == META["test_rows"]

def test_all_five_saved_models_evaluate_without_retraining():
    X = TEST_FRAME[META["feature_names"]]
    y = TEST_FRAME["target"]
    assert len(META["model_files"]) == 5
    for path in META["model_files"].values():
        model = joblib.load(ROOT / "model" / path)
        scores = evaluate_model(model, X, y)
        assert set(scores.keys()) == set(METRIC_COLUMNS)
        assert all(-1 <= v <= 1 for v in scores.values())
        matrix, report, predictions = diagnostic_report(model, X, y)
        assert matrix.shape == (2, 2)
        assert len(predictions) == len(y)
        assert "precision" in report

def test_saved_comparison_matches_fresh_scores():
    expected = pd.read_csv(ROOT / "model_comparison.csv").set_index("ML Model Name")
    X = TEST_FRAME[META["feature_names"]]
    y = TEST_FRAME["target"]
    for name, path in META["model_files"].items():
        score = evaluate_model(joblib.load(ROOT / "model" / path), X, y)
        for metric in METRIC_COLUMNS:
            assert abs(score[metric] - expected.loc[name, metric]) < 1e-12
