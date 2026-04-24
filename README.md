# Student Satisfaction Predictor (Streamlit)

This Streamlit app uses a local scikit-learn model file (`ai_student_satisfaction_model.pkl`) to predict student satisfaction.

## Quick Start

Use Python 3.11 for local and cloud consistency.

1. Create and activate a virtual environment.

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the app.

```bash
streamlit run app.py
```

## How Input Works

- The app reads expected feature names from the model (`feature_names_in_`).
- The app auto-detects whether each feature accepts text or numeric values by testing model prediction behavior.
- If a feature supports text, the UI shows a text box.
- If a feature requires numeric input, the UI shows a number input.
- For the current model (`RandomForestClassifier` without a preprocessing pipeline), categorical fields are numeric encoded values.
- Batch CSV uploads must contain exactly the expected columns.

## Version Compatibility

The current model was trained with scikit-learn `1.6.1`. For safest compatibility, pin scikit-learn to that version:

```bash
pip install scikit-learn==1.6.1
```

## Deployment

This repo includes `runtime.txt` with `python-3.11` for platforms that read runtime version from that file.
This repo also includes `.python-version` set to `3.11` for platforms that prefer that file.

If Streamlit Cloud still shows Python 3.14 in logs after pushing these files, open app settings and trigger a full reboot/redeploy so the environment is rebuilt with the new runtime.

### Option 1: Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Ensure these files are in the repo root:
	- `app.py`
	- `requirements.txt`
	- `ai_student_satisfaction_model.pkl`
3. Go to Streamlit Community Cloud and create a new app from your repo.
4. Set main file path to `app.py`.
5. Deploy.

### Option 2: Docker (Any Cloud VM/Container Platform)

Create a `Dockerfile` like this:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t student-satisfaction-app .
docker run -p 8501:8501 student-satisfaction-app
```
