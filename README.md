# Lightweight Streamlit ML Predictor

This minimal Streamlit app lets you upload a scikit-learn pickled model (`.pkl`) and a CSV of input rows to produce predictions.

Quick start

1. Create a virtual environment and activate it.

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies and run the app.

```bash
pip install -r requirements.txt
streamlit run app.py
```

How to use

- Upload your trained scikit-learn model file (`.pkl`).
- Upload a CSV where each row is a sample (columns correspond to the features the model expects), or paste a single-row CSV in the text box for one prediction.
- Click `Run prediction` to view and download results.

Deployment

- This app is intentionally minimal and works on Streamlit Cloud and most container platforms. Include the files in a git repo and connect to Streamlit Cloud, or build a small Docker image if preferred.

Notes

- The app does not assume feature names; it feeds the uploaded CSV columns directly to the model. Ensure your input columns match the model's training features and order.
- Unpickling models requires the same library versions used when the model was saved. If unpickling fails, try matching scikit-learn versions locally.
