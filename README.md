# Breast Cancer Classification using Machine Learning

A machine learning project for classifying breast tumors as **Benign or Malignant** using diagnostic features. This project explores multiple classification algorithms, compares their performance, analyzes feature importance, and applies probability threshold tuning to investigate false-negative reduction.

## Project Overview

The goal is to build and evaluate machine learning models that classify breast cancer diagnoses using structured tabular data.

The project focuses on model comparison, evaluation metrics, interpretability, and the trade-off between precision and recall, particularly when false negatives are a major concern.

**Target variable:** `diagnosis`

- `B` → `0` — Benign
- `M` → `1` — Malignant

## Machine Learning Models

The following classification algorithms are implemented and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Dataset shape and data types
- Missing-value inspection
- Descriptive statistics
- Target distribution analysis

### 2. Data Cleaning and Preprocessing

- Removal of unnecessary columns, including `id` and `Unnamed: 32`, when present
- Conversion of categorical target labels into binary values
- Stratified train-test split
- Feature scaling with `StandardScaler` for applicable models

### 3. Model Training and Evaluation

Models are compared using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

### 4. Model Comparison and Error Analysis

- Comparison of model performance
- Confusion matrix analysis
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Identification of models with fewer false negatives

### 5. Feature Importance Analysis

Feature importance is examined using:

- Random Forest
- XGBoost
- LightGBM

The project also compares feature importance across these models and visualizes the top 15 features.

### 6. Probability Threshold Tuning

The LightGBM classifier is evaluated using different probability thresholds, ranging from `0.10` to `0.90`.

The analysis investigates how changing the threshold affects:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- Precision
- Recall
- False Negative Rate

A threshold of `0.10` is selected for the final inference workflow to prioritize identifying malignant cases.

### 7. Cross-Validation

Stratified 5-fold cross-validation is used to examine the stability of the selected threshold and its effect on false negatives across different folds.

### 8. Model Explainability with SHAP

SHAP is used to interpret the LightGBM model through:

- Global feature importance
- SHAP summary bar plot
- SHAP beeswarm plot

### 9. Model Serialization and Inference

The final LightGBM model is trained and saved using Joblib.

The inference function returns:

- Predicted class
- Estimated probability of malignancy
- Risk label for review

The serialized model can be loaded for subsequent inference.

## Tech Stack

- **Language:** Python
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Gradient Boosting:** XGBoost, LightGBM, CatBoost
- **Model Explainability:** SHAP
- **Model Serialization:** Joblib
- **Environment:** Jupyter Notebook

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/behrouzyou/ML-Breast-Cancer.git
cd breast-cancer-classification
```

Replace the repository URL with the actual repository URL if its name differs.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm catboost shap joblib jupyter
```

### 4. Prepare the dataset

Place the dataset CSV file in the project directory and update the notebook's data-loading path accordingly.

```python
df = pd.read_csv("data.csv")
```

The notebook currently uses a local Windows file path, so it must be updated before running the project on another machine.

### 5. Run the notebook

```bash
jupyter notebook breast-cancer.ipynb
```

Execute the notebook cells sequentially to reproduce the workflow.

## Model Output

The final inference function produces a DataFrame containing:

| Column                  | Description                                 |
| ----------------------- | ------------------------------------------- |
| `probability_malignant` | Model-estimated probability of malignancy   |
| `prediction`            | `0` for benign, `1` for malignant           |
| `risk_label`            | `Low Risk` or `High Risk (Review Required)` |

The selected threshold is `0.10`, meaning predictions with an estimated malignancy probability of at least 10% are flagged for review.

## Important Limitations

- The project is intended for educational and research purposes, not clinical diagnosis or medical decision-making.
- A low classification threshold can increase false positives; it does not guarantee that every malignant case will be detected.
- Threshold selection should ideally use a separate validation set or nested cross-validation. Selecting a threshold based on test-set performance can bias the reported results.
- Performance metrics must be reported on data that was not used for model selection or threshold tuning.
- The probability output is a model estimate and should not automatically be interpreted as a clinically calibrated probability.

## Future Improvements

- Separate training, validation, and test datasets for more reliable model selection.
- Hyperparameter optimization using cross-validation.
- Probability calibration and decision-curve analysis.
- A reproducible preprocessing and inference pipeline.
- Deployment through a REST API using FastAPI or Django REST Framework.
- Experiment tracking and model versioning.

## Author

**Behrouz**

GitHub: [@behrouzyou](https://github.com/behrouzyou)

---

*This project demonstrates an end-to-end tabular machine learning workflow, from exploratory data analysis and model comparison to explainability, threshold analysis, and model serialization.*
