# Credit Card Default Classification — ML Assignment 2

## a. Problem statement
Implement five machine-learning classification algorithms to predict whether a credit-card client will default on payment next month. The target is `default payment next month` (0 = no default, 1 = default).

## b. Dataset description
- **Source:** UCI Machine Learning Repository — [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
- **Observations:** 30,000.
- **Predictive input features:** 23.
- **Target:** `default payment next month` (binary classification).
- **ID:** Retained in the source/CSV data where present, but excluded from model inputs.
- **Target distribution:** 23,364 non-default (77.88%) and 6,636 default (22.12%).
- The original source file is the Excel dataset obtained from UCI. `credit_card_default_cleaned.csv` is the CSV used by `train.py`.

## c. GitHub Repository Link
(https://github.com/Srushti4040/MLAssignment)

## d. Models used

### Experimental protocol
- One stratified 80/20 train/test split, with `random_state=42`.
- Training rows: 24,000; test rows: 6,000.
- The same holdout observations are used to evaluate all five models.
- `ID` is excluded from predictors.
- `SEX`, `EDUCATION`, and `MARRIAGE` are one-hot encoded.
- The remaining 20 input features are passed as numeric features.
- Numeric features are standardised for Logistic Regression, KNN, Gaussian Naive Bayes and Linear SVM. Decision Tree uses unscaled numeric features.
- Preprocessing is included inside each saved scikit-learn pipeline and is fitted only on training data.
- No resampling or class weighting is used in this baseline experiment.
- Models: Logistic Regression (`max_iter=2000`), Decision Tree (`random_state=42`), KNN (`n_neighbors=5`), Gaussian Naive Bayes, and Linear SVM (`dual=False`, `max_iter=3000`).
- AUC uses predicted probabilities where available and the decision function for Linear SVM. AUC is undefined if an evaluation dataset contains only one target class.
- Precision, recall and F1 use class 1 (default) as the positive class.

### Evaluation metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8088 | 0.7100 | 0.6923 | 0.2442 | 0.3610 | 0.3302 |
| Decision Tree | 0.7113 | 0.5963 | 0.3587 | 0.3873 | 0.3725 | 0.1856 |
| K-Nearest Neighbors | 0.7948 | 0.7038 | 0.5561 | 0.3587 | 0.4361 | 0.3292 |
| Gaussian Naive Bayes | 0.2903 | 0.7240 | 0.2340 | 0.9714 | 0.3771 | 0.1034 |
| Support Vector Machine | 0.8043 | 0.7065 | 0.7297 | 0.1831 | 0.2928 | 0.2970 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest accuracy (0.8088) and highest MCC (0.3302, narrowly above KNN). Precision is 0.6923, but recall is 0.2442, so it misses many actual defaulters at the default threshold. |
| Decision Tree | Lowest AUC (0.5963) and lowest accuracy (0.7113). Recall (0.3873) is higher than Logistic Regression, but precision is low (0.3587). |
| K-Nearest Neighbors | Highest F1 (0.4361). Its precision (0.5561) and recall (0.3587) provide the strongest F1 balance among these baseline models. |
| Gaussian Naive Bayes | Highest AUC (0.7240) and recall (0.9714), but low accuracy (0.2903) and precision (0.2340). It predicts the default class very frequently at the default threshold. |
| Support Vector Machine | Highest precision (0.7297), but lowest recall (0.1831). It identifies fewer actual defaulters at the default threshold. |

### Overall winner for your dataset?
There is no single winner across all metrics. **K-Nearest Neighbors is the winner if F1 score is the selection criterion** (F1 = 0.4361). Logistic Regression has the highest accuracy and MCC (0.3302). Gaussian Naive Bayes has the highest AUC and recall, but its accuracy is low. The preferred model depends on the relative cost of missed defaulters and false alarms. These are baseline results, not a claim that the models are optimally tuned.

## Project files
- `app.py`: Streamlit interface. It evaluates all five saved models on the uploaded labelled test CSV and shows a comparison table, then provides detailed metrics, confusion matrix and classification report for a selected model.
- `train.py`: Reproducible training and evaluation script.
- `requirements.txt`: Python dependencies, including Joblib for saving/loading model pipelines.
- `credit_card_default_cleaned.csv`: CSV input required by `train.py`.
- `train_data.csv`: Training partition supplied with the project.
- `test_data.csv`: Labelled holdout test data for the app; includes the 23 input features and target.
- `model/`: Five fitted model pipelines (`.joblib`), including their preprocessing steps.
- `reports/model_comparison.csv`: Recorded comparison results from the baseline training run.

## Run locally
From this project directory:

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```


