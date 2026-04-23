import io
import pickle

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Lightweight ML Predictor", layout="centered")

st.title("Lightweight ML Predictor")

st.markdown(
    "Upload a scikit-learn pickled model (.pkl) and a CSV of input rows to get predictions."
)

# Model uploader
model_file = st.file_uploader("Upload model (.pkl)", type=["pkl", "joblib"])
model = None
if model_file is not None:
    try:
        model = pickle.load(model_file)
        st.success(f"Model loaded: {model.__class__.__name__}")
        if hasattr(model, "n_features_in_"):
            st.info(f"Model expects {model.n_features_in_} input features")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

st.markdown("---")

# Data uploader / manual input
csv_file = st.file_uploader("Upload input CSV (rows of features)", type=["csv"])  
manual_text = st.text_area(
    "Or paste a single-row CSV (comma-separated) if you want one prediction",
    height=80,
)

df = None
if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

elif manual_text.strip():
    try:
        # Parse single-row CSV into DataFrame
        sample = io.StringIO(manual_text.strip())
        df = pd.read_csv(sample, header=None)
        # If single row with no header, treat as one sample with columns 0..n
        st.write("Parsed single-row input:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to parse manual input: {e}")

if st.button("Run prediction"):
    if model is None:
        st.error("Please upload a model (.pkl) first.")
    elif df is None:
        st.error("Please provide input data via CSV upload or manual paste.")
    else:
        try:
            X = df.values
            # If model expects n_features_in_ and shapes mismatch, try to adapt
            if hasattr(model, "n_features_in_"):
                expected = model.n_features_in_
                if X.shape[1] != expected:
                    st.warning(
                        f"Input has {X.shape[1]} columns but model expects {expected}. Trying to proceed anyway."
                    )

            preds = model.predict(X)
            out = pd.DataFrame(preds, columns=["prediction"]) if preds.ndim == 1 else pd.DataFrame(preds)
            result = pd.concat([pd.DataFrame(df).reset_index(drop=True), out.reset_index(drop=True)], axis=1)
            st.write("Predictions:")
            st.dataframe(result)

            # If probabilities available, show them too
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X)
                    probs_df = pd.DataFrame(probs)
                    probs_df.columns = [f"prob_class_{i}" for i in probs_df.columns]
                    result = pd.concat([result, probs_df.reset_index(drop=True)], axis=1)
                    st.write("Predicted probabilities:")
                    st.dataframe(probs_df)
                except Exception:
                    pass

            # Download button
            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", data=csv_bytes, file_name="predictions.csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Designed to be minimal: upload your model and data, then predict.")
