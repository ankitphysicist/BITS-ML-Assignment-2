# Classification Model Lab - Machine Learning Assignment 2 (Worked Example)

> **Coursework learning reference.** This is a reproducible worked example, not a ready-to-submit assignment. The brief permits AI for learning support only. Run and adapt the experiment yourself, write your own observations, and use only verified links. Do not present these example files or model outputs as independently authored without attribution.

## a. Problem statement

Build and compare five classification models on ONE dataset, compute Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC), and create an interactive Streamlit application. The app must load previously trained models and score a CSV containing both features and true target labels; it must not train on the uploaded test set.

## b. Dataset description

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Original public repository:** UCI Machine Learning Repository
- **Dataset URL:** https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
- **Loading method:** `sklearn.datasets.load_breast_cancer(as_frame=True)`, which bundles this UCI dataset.
- **Instances:** 569; **numeric features:** 30; **task:** binary classification.
- **Target:** `target = 0` (malignant), `target = 1` (benign).
- **Typical inputs:** mean radius, mean texture, mean perimeter, mean area, and related cell-nucleus measurements.
- **Split:** stratified 80% train / 20% test, random state 73; 455 training rows and 114 unseen test rows.
- **Preprocessing:** `StandardScaler` belongs to the respective fit-on-training-only pipelines for Logistic Regression, KNN, Naive Bayes, and SVM. Decision Tree receives unscaled inputs.
- **Data leakage safeguard:** all preprocessing is fitted on training samples only. Held-out test data is used solely for scoring.
- **Class convention for binary metrics:** **benign (1)** is the positive class. This convention affects Precision, Recall and F1. AUC uses the probability of class 1.

## c. GitHub repository link

**GitHub repository URL:** https://github.com/ankitphysicist/BITS-ML-Assignment-2

**Live Streamlit URL:** https://ankit-ml-assignment-2.streamlit.app/

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ankit-ml-assignment-2.streamlit.app/)

> Live link was supplied by the app owner. GitHub's automated frontend test passed; independent verification of the publicly hosted URL is pending.

## d. Models used and comparison table

All five algorithms use the **same dataset** and **same held-out test split**. Metrics below were computed by executing `python train_models.py` with the listed parameters and scikit-learn 1.8.0.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9912 | 0.9987 | 0.9863 | 1.0000 | 0.9931 | 0.9812 |
| Decision Tree | 0.9035 | 0.9221 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| KNN | 0.9561 | 0.9886 | 0.9351 | 1.0000 | 0.9664 | 0.9076 |
| Naive Bayes | 0.9123 | 0.9802 | 0.8974 | 0.9722 | 0.9333 | 0.8113 |
| Support Vector Machine | 0.9825 | 0.9987 | 0.9730 | 1.0000 | 0.9863 | 0.9626 |

### Observations on model performance (reference analysis - rewrite in your own words)

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest accuracy, F1, and MCC on this held-out split. No class-1 false negatives in this split (Recall = 1.0000). |
| Decision Tree | Lowest accuracy and AUC; a single tree may be less stable or may miss complex boundaries. |
| KNN | Good accuracy and very high AUC; Recall for benign is 1.0000, but Precision is lower than Logistic Regression. |
| Naive Bayes | Relatively high AUC but weaker threshold-based precision and accuracy, consistent with the model's simplifying distribution assumptions. |
| Support Vector Machine | Second-best accuracy and MCC with high AUC; performance is close to Logistic Regression. |
| **Overall winner** | **Logistic Regression** by accuracy, F1, and MCC on this specific split. Different splits or tuning can change ranking. |

**Limitations:** This is one random hold-out split. No nested validation or search-based tuning was performed. The example is for a programming assignment, not medical use. Because class 1 is benign, high class-1 recall does not imply high recall for malignant samples.

## Folder structure

```text
project-folder/
  app.py                    # Streamlit UI; does NOT fit models
  train_models.py           # run once to train and persist all five models
  evaluation.py             # shared evaluation functions
  requirements.txt          # Streamlit Community Cloud dependencies
  README.md                 # assignment explanation and instructions
  LEARN_IN_HINGLISH.md      # commented beginner-friendly study guide
  dataset.csv               # complete 569-row data (features + target)
  test_data.csv             # 114 held-out rows (features + target)
  model_comparison.csv      # measured scores from the 5 models
  metadata.json             # feature names, model filenames, label scheme
  model/                    # five pre-fitted .joblib files
  .streamlit/config.toml    # UI theme
  tests/test_core.py        # reproducibility checks
  make_submission_pdf.py    # editable PDF submission generator
  requirements-report.txt   # PDF-only dependency
  submission_template.pdf   # SAMPLE; not submission-ready
```

## How to run locally

1. Use Python 3.12 and install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install pytest
   ```

2. Run `python train_models.py` if you want to reproduce the five saved classifiers.
3. Run `python -m pytest -q` to validate data/model scores.
4. Run `streamlit run app.py` and open the local URL.
5. Upload `test_data.csv` and confirm all six metrics, confusion matrix and report.
6. Review and personalize this learning reference before any academic submission.

## How the Streamlit application works

- `app.py` reads `metadata.json` and opens all 5 saved models with `joblib.load`.
- By default, it reads bundled `test_data.csv`; you may upload a different held-out test CSV.
- It checks for **all 30 feature columns plus `target`**, numeric data, nonmissing values and binary labels.
- It reorders input features to match the training schema.
- It **only calls** `predict()` and `predict_proba()` on existing models; **never** `.fit()`.
- It shows a comparison table for all models, model selection dropdown, 6 scores, confusion matrix, classification report, and downloadable predictions.
- **AUC requires both classes to appear in the test CSV.** If only one is present, AUC is N/A.

## Deploy on Streamlit Community Cloud

1. Sign into GitHub and create a repository for your own project; use meaningful incremental commits.
2. Upload or push all necessary source, metadata, `test_data.csv`, requirements, and saved `model/` files.
3. Go to https://share.streamlit.io/ and sign in with GitHub.
4. Click **Create app**, select your repository, `main` branch, and `app.py` as the entrypoint, then deploy.
5. Open the new `.streamlit.app` URL and test CSV upload, model selection, 6 metrics, confusion matrix/report.
6. Include the app URL above and the repository URL in your report PDF. Test the public deployment yourself before submission.

## Generate the PDF report

The report includes the verified GitHub link, the deployed Streamlit URL (when available) and README text. The BITS Virtual Lab screenshot is intentionally omitted at the user's request; this differs from the original assignment instructions.

```bash
python -m pip install -r requirements-report.txt
python make_submission_pdf.py --github "https://github.com/ankitphysicist/BITS-ML-Assignment-2" --app "https://YOUR-REAL-APP.streamlit.app" --output "final_submission.pdf"
```

If a working app URL is unavailable, generate without `--app`; the report will be marked incomplete. Never invent or guess the live link.

## Metric definitions

- Accuracy: share of correctly classified examples.
- AUC: area under receiver operating characteristic curve using model probability for class 1.
- Precision (benign = 1): TP / (TP + FP).
- Recall (benign = 1): TP / (TP + FN).
- F1: 2 * Precision * Recall / (Precision + Recall).
- MCC: correlation-like measure based on TP, TN, FP, FN, ranging from -1 to +1.

## Reproducibility and academic integrity

This educational reference is generated without invented screenshots or deployment URLs. Your instructor's brief explicitly notes AI is permitted for **learning support only**, checks commit history and originality, and warns copied submissions can receive zero marks. Review, adapt, and execute the code yourself and rewrite the report based on your own tested results.