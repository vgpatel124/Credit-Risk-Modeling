## Credit Risk Modeling – Streamlit App

### Project Overview

This project implements a Credit Risk Modeling framework for financial institutions to assess the likelihood of default among loan applicants. Using machine learning techniques, the system improves risk prediction and supports better lending decisions.
The model achieved:
- AUC (ROC): Improved from 82% → 88%
- KS Statistic: 46 (above industry benchmark)
- Gini Coefficient: Improved from 78 → 85
- 12% reduction in projected defaults with model deployment

### Tech Stack

- Languages/Frameworks: Python, Streamlit
- Libraries: Pandas, NumPy, Scikit-learn, Optuna, Matplotlib, Seaborn
- ML Models: Logistic Regression, Random Forest, Gradient Boosting
- Deployment: Streamlit

### Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

### Deploy on Streamlit Community Cloud

1. Push this project to GitHub.
2. In Streamlit Cloud, create a new app pointing to your repo.
3. Set the entrypoint to `app/main.py`.
4. No special secrets are required.

Notes:
- The model file is expected at `artifacts/model_data.joblib` (repo root). The code falls back to `app/artifacts/model_data.joblib` if needed.
- `scikit-learn` is pinned to `1.3.0` to match the serialized model.
