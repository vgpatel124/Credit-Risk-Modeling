# Credit Risk Modeling

End-to-end credit risk modeling project for predicting loan defaults, assigning credit scores, and serving predictions through an interactive Streamlit web app. The full machine learning workflow — from data merging and EDA to feature selection, model training, and deployment — lives in `credit_risk_model.ipynb`.

---

## Overview

This project simulates a lending workflow for **Lauki Finance**. It combines customer demographics, loan details, and credit bureau data to:

1. Predict the probability of loan default
2. Convert that probability into a **credit score (300–900)**
3. Assign a **rating** (Poor, Average, Good, Excellent)

The trained **Logistic Regression** model is serialized with `joblib` and exposed via a Streamlit UI for real-time risk assessment.

---

## Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.9+ |
| **Data & Analysis** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn (Logistic Regression, MinMaxScaler, train/test split, RandomizedSearchCV, ROC/AUC) |
| **Class Imbalance** | imbalanced-learn (SMOTETomek) |
| **Hyperparameter Tuning** | Optuna, RandomizedSearchCV |
| **Model Comparison** | XGBoost (evaluated, not selected for production) |
| **Model Persistence** | joblib |
| **Web App** | Streamlit |
| **Development** | Jupyter Notebook |

---

## Project Structure

```
Credit-Risk-Modeling/
├── app/
│   ├── main.py                  # Streamlit web application
│   ├── prediction_helper.py     # Model loading, preprocessing, and inference
│   └── artifacts/
│       └── model1_data.joblib   # Saved model, scaler, and feature metadata
├── dataset/
│   ├── customers.csv            # Customer demographics (50,000 records)
│   ├── loans.csv                # Loan application & outcome data
│   └── bureau_data.csv          # Credit bureau history
├── credit_risk_model.ipynb      # Full ML pipeline (EDA → training → evaluation)
└── README.md
```

---

## Dataset

Three CSV files are joined on `cust_id` to build a unified dataset of **50,000 loan records**.

| File | Key Fields |
|---|---|
| `customers.csv` | age, gender, marital status, employment, income, dependants, residence type, address history, location |
| `loans.csv` | loan amount, tenure, purpose, type, disbursal details, **default** (target) |
| `bureau_data.csv` | open/closed accounts, delinquency history, DPD, enquiry count, credit utilization |

**Target variable:** `default` (binary)  
**Class distribution:** ~91% non-default, ~9% default (imbalanced — handled with SMOTETomek during training)

---

## ML Pipeline

The notebook follows a structured credit modeling workflow:

### 1. Data Preparation
- Merge `customers`, `loans`, and `bureau_data` on `cust_id`
- Stratified **75/25 train–test split** before EDA to prevent data leakage
- Outlier removal (e.g. processing fee > 3% of loan amount)

### 2. Feature Engineering
- `loan_to_income` — loan amount divided by income
- `delinquency_ratio` — delinquent months as a percentage of total loan months
- `avg_dpd_per_delinquency` — average days past due per delinquent month

### 3. Exploratory Data Analysis
- Univariate and bivariate analysis with KDE plots
- Correlation heatmaps
- Distribution analysis by default status

### 4. Feature Selection
- **VIF (Variance Inflation Factor)** — remove multicollinear features
- **WOE/IV (Weight of Evidence / Information Value)** — keep features with IV > 0.02
- **One-hot encoding** for categorical variables (`residence_type`, `loan_purpose`, `loan_type`)
- **MinMaxScaler** on numeric columns

### 5. Model Training & Tuning
- Baseline models: Logistic Regression, Random Forest, XGBoost
- **Logistic Regression** selected for interpretability (comparable performance to XGBoost)
- **SMOTETomek** applied to handle class imbalance
- **Optuna** hyperparameter optimization (maximize macro F1)
- **RandomizedSearchCV** for initial parameter search

### 6. Evaluation
- Classification report (precision, recall, F1)
- ROC curve and **AUC ≈ 0.98**
- Feature coefficient analysis

### 7. Model Export
The final artifact bundles everything needed for inference:

```python
{
    'model': final_model,           # LogisticRegression
    'scaler': scaler,               # MinMaxScaler fitted on training data
    'features': X_train_encoded.columns,
    'cols_to_scale': cols_to_scale
}
```

---

## Model Features

The production model uses **13 features** after encoding:

| Feature | Type |
|---|---|
| `age` | Numeric |
| `loan_tenure_months` | Numeric |
| `number_of_open_accounts` | Numeric |
| `credit_utilization_ratio` | Numeric |
| `loan_to_income` | Numeric (engineered) |
| `delinquency_ratio` | Numeric (engineered) |
| `avg_dpd_per_delinquency` | Numeric (engineered) |
| `residence_type_Owned`, `residence_type_Rented` | One-hot |
| `loan_purpose_Education`, `loan_purpose_Home`, `loan_purpose_Personal` | One-hot |
| `loan_type_Unsecured` | One-hot |

---

## Credit Score & Rating

Default probability is converted to a credit score using a logistic function:

```
credit_score = 300 + (1 - default_probability) × 600
```

| Score Range | Rating |
|---|---|
| 300 – 499 | Poor |
| 500 – 649 | Average |
| 650 – 749 | Good |
| 750 – 900 | Excellent |

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Credit-Risk-Modeling.git
cd Credit-Risk-Modeling

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn \
            imbalanced-learn optuna xgboost joblib streamlit jupyter
```

### Run the Streamlit App

The app must be started from the `app/` directory because model paths are relative:

```bash
cd app
streamlit run main.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

### Reproduce the Model (Optional)

To retrain from scratch or explore the full pipeline:

```bash
jupyter notebook credit_risk_model.ipynb
```

Run all cells from the project root. The notebook saves the updated model to `artifacts/model1_data.joblib` — copy it into `app/artifacts/` for the web app to pick up the new version.

---

## Web App Usage

The Streamlit UI collects applicant inputs and returns a risk assessment on **Calculate Risk**:

| Input | Description |
|---|---|
| Age | Applicant age (18–100) |
| Income | Annual income |
| Loan Amount | Requested loan amount |
| Loan Tenure | Repayment period in months |
| Avg DPD | Average days past due per delinquency |
| Delinquency Ratio | Percentage of delinquent months |
| Credit Utilization Ratio | Percentage of credit limit used |
| Open Loan Accounts | Number of active loan accounts (1–4) |
| Residence Type | Owned / Rented / Mortgage |
| Loan Purpose | Education / Home / Auto / Personal |
| Loan Type | Unsecured / Secured |

**Output:**
- Default probability (%)
- Credit score (300–900)
- Rating category

---

## Example

```python
from prediction_helper import predict

probability, credit_score, rating = predict(
    age=28,
    income=1200000,
    loan_amount=2560000,
    loan_tenure_months=36,
    avg_dpd_per_delinquency=20,
    delinquency_ratio=30,
    credit_utilization_ratio=30,
    num_open_accounts=2,
    residence_type='Owned',
    loan_purpose='Education',
    loan_type='Unsecured'
)

print(f"Default Probability: {probability:.2%}")
print(f"Credit Score: {credit_score}")
print(f"Rating: {rating}")
# Default Probability: 66.77%
# Credit Score: 499
# Rating: Poor
```

---

