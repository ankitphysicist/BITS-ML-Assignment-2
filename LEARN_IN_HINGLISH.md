# Machine Learning Assignment 2 - Beginner's Hinglish Guide

## 1. Assignment ka goal

हमें एक ही dataset पर पाँच ML classifiers train करने हैं, उनकी performance measure करनी है, और Streamlit में एक interactive app बनाना है। **सबसे important बात:** training data और test data अलग होने चाहिए। Test CSV upload होने पर model को **दोबारा train नहीं** करना है।

## 2. Dataset और labels समझें

यहाँ UCI Breast Cancer Wisconsin (Diagnostic) में 569 rows और 30 numeric features हैं। `target` column में `0 = malignant`, `1 = benign` है। Example का positive class **1 (benign)** है। इसलिए Recall = 1.0 का मतलब यहाँ सभी *benign* test rows सही पहचानी गई हैं; यह automatically malignant detection की recall नहीं है। यह educational demonstration है, medical decision tool नहीं।

## 3. Train-test split क्यों?

मान लीजिए आप student हैं। अगर exam के सारे questions पहले से answers सहित याद हों, तो exam का score आपका real understanding नहीं बताएगा। वैसे ही test dataset वह data है जो training में model ने नहीं देखा।

`train_test_split(..., test_size=0.20, stratify=y, random_state=73)` से 455 training records और 114 testing records बनते हैं। `stratify=y` दोनों classes का similar proportion रखने में मदद करता है।

## 4. पाँच algorithms की intuition

- **Logistic Regression:** Features के weighted combination से class की probability निकालती है।
- **Decision Tree:** सवाल पूछती है जैसे "feature X > threshold?" और branches से prediction देती है।
- **KNN:** नए record के आस-पास के 7 training neighbors देखकर class चुनती है।
- **Gaussian Naive Bayes:** Features को Gaussian distributions से approximate करके class probabilities calculate करती है; conditional independence assumption लगाती है।
- **SVM:** Classes के बीच अच्छा separating boundary बनाने की कोशिश करती है; यहाँ RBF kernel है।

## 5. StandardScaler और Pipeline

किसी feature की range बहुत बड़ी, किसी की छोटी हो सकती है। `StandardScaler()` numbers को comparable scale पर लाने में मदद करता है। `make_pipeline(StandardScaler(), Model())` ensures scaler सिर्फ training samples पर fit होता है और बाद में उसी fitted scaler से test data transform होता है। इससे leakage avoid होती है।

## 6. Confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | True Negative (TN) | False Positive (FP) |
| Actual 1 | False Negative (FN) | True Positive (TP) |

क्योंकि positive=1=benign, ऊपर TP का मतलब सही तरीके से benign predict करना है। अगर आपकी priority malignant detection है तो अलग positive-class convention चुनकर उसके metrics explicitly calculate करें।

## 7. Metrics की definitions

- **Accuracy** = (TP + TN) / All examples.
- **Precision** = TP / (TP + FP): predicted benign में कितने सच में benign थे?
- **Recall** = TP / (TP + FN): actual benign में कितने सही पकड़े?
- **F1** = 2 * Precision * Recall / (Precision + Recall).
- **AUC** = अलग-अलग classification thresholds पर ROC curve के नीचे का area. AUC calculate करने के लिए `predict_proba` के probabilities चाहिए, सिर्फ final 0/1 labels नहीं।
- **MCC** = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)). MCC = +1 excellent, 0 random-like correlation, -1 perfect inversion.

## 8. Files कैसे connected हैं?

1. `train_models.py`: dataset load करता है, train/test split करता है, five models train करता है, joblib files save करता है।
2. `evaluation.py`: पहले से trained model से six metrics compute करता है।
3. `app.py`: saved `.joblib` files load करता है, CSV upload लेता है, metrics और confusion matrix दिखाता है। यहाँ `.fit()` **नहीं** है।
4. `test_data.csv`: 30 features plus `target`; app के लिए evaluation data।
5. `README.md`: explanation, comparison और observations; PDF में include करना है।

## 9. Commands

```bash
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

App में **Upload test CSV** चुनिए और bundled `test_data.csv` upload करिए। Dropdown से हर model select करिए और six metrics observe करिए। फिर BITS Virtual Lab में execution का वास्तविक screenshot लीजिए।

## 10. Observation लिखते समय अपना analysis कैसे करें?

क्यों Logistic Regression ने इस split पर ज़्यादा accuracy दी? क्या Decision Tree की depth बदलने पर score बदलता है? Class 1 यानी benign recall के बजाय malignant recall निकालें तो क्या conclusion बदलता है? `random_state` बदले और देखें कौन-से scores stable रहते हैं। अपनी experiments and results के अनुसार README update करें।