# Type II Diabetes Prediction with Machine Learning

A machine learning project to predict **Type II diabetes** from health indicators using multiple classification models, with a focus on data cleaning (abnormal zero handling), model comparison, and evaluation under class imbalance.

> This is a **team project**; co-authors are omitted in this public version.

---

## Dataset

- **Pima Indians Diabetes Dataset** (768 samples, 8 features + binary outcome).
- Original collection: **NIDDK**; distributed via Kaggle (see report references).
- Target label: `Outcome` (1 = diabetes positive, 0 = negative).

The dataset file used in this repo:
- `data/diabetes.csv`

---

## Problem & Approach

### Key Data Issue: Abnormal Zeros
Several predictors contain **zero values** that are physiologically implausible (e.g., `Glucose`, `BMI`, `Insulin`), which can act like missing/invalid measurements.
To avoid large data loss, we apply **median imputation** to replace abnormal zeros for selected features.

### Modeling Pipeline
- Train/test split: **80/20**, using **stratified sampling** to preserve class distribution.
- Hyperparameter tuning: **GridSearchCV** with **Stratified 5-fold CV**.
- Main models compared:
  - Logistic Regression (Ridge), including polynomial features variant
  - SVM (RBF / Linear)
  - Random Forest
  - Shallow Neural Network (1 hidden layer, PyTorch)

---

## Results (from report)

On the held-out test set, **Random Forest** achieved the strongest overall balance across metrics:
- **Accuracy:** ~0.753
- **F1:** ~0.612
- **AUC-ROC:** ~0.811

Logistic Regression and Random Forest achieved similar **AUC-ROC (~0.81)** in the report.
See the full write-up and figures in:
- `docs/diabetes_detection_report_public.pdf`

---

## Repository Structure

```text
.
├── notebooks/
│   └── diabetes_detection.ipynb
├── data/
│   └── diabetes.csv
└── docs/
    └── diabetes_detection_report_public.pdf
```

---

## How to Run

### 1) Create environment (recommended)

```text
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# macOS/Linux:
#   source .venv/bin/activate
```

### 2) Install dependencies
If you have `requirements.txt`:

```text
pip install -r requirements.txt
```

Otherwise install the common stack:

```text
pip install numpy pandas scikit-learn matplotlib seaborn notebook ipywidgets
pip install torch
```

### 3) Run the notebook

Open and execute:
- `notebooks/diabetes_detection.ipynb`

The notebook reads the dataset from:
- `data/diabetes.csv`

---

## Notes
- This repository is for **educational / portfolio** demonstration only and is **not medical advice**.

---

## License
MIT.
