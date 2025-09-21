## Credit Risk Modeling – Streamlit App

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
