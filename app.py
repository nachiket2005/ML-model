from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Student Satisfaction Predictor", layout="centered")

MODEL_PATH = Path("ai_student_satisfaction_model.pkl")


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def infer_feature_input_modes(model, feature_names: list[str]) -> dict[str, str]:
    """Infer whether each feature accepts text or numeric values at prediction time."""
    if not feature_names:
        return {}

    numeric_defaults = {
        "Age": 20.0,
        "Daily_Usage_Hours": 2.0,
    }
    string_samples = {
        "Gender": "male",
        "Education_Level": "College",
        "City": "Mumbai",
        "AI_Tool_Used": "Chatgpt",
        "Purpose": "Homework",
        "Impact_on_Grades": "good",
    }

    def _predict_accepts_text(feature: str) -> bool:
        # Start from a numeric-safe baseline, then test a text value for one feature.
        row = {f: numeric_defaults.get(f, 0.0) for f in feature_names}
        row[feature] = string_samples.get(feature, "sample_text")

        test_df = pd.DataFrame([row], columns=feature_names)
        try:
            model.predict(test_df)
            return True
        except Exception:
            return False

    modes: dict[str, str] = {}
    for feature in feature_names:
        modes[feature] = "text" if _predict_accepts_text(feature) else "numeric"

    return modes


def validate_input_columns(df: pd.DataFrame, required_columns: list[str]) -> tuple[bool, str]:
    missing = [c for c in required_columns if c not in df.columns]
    extra = [c for c in df.columns if c not in required_columns]

    if missing:
        return False, f"Missing columns: {missing}"

    if extra:
        return False, f"Unexpected columns: {extra}"

    return True, ""


st.title("Student Satisfaction Model App")
st.caption("Uses your provided RandomForest model file directly.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.info("If this is a version issue, install scikit-learn==1.6.1 in your venv.")
    st.stop()

feature_names = list(getattr(model, "feature_names_in_", []))
if not feature_names:
    st.error("Model does not expose feature names. Cannot build a reliable UI.")
    st.stop()

classes = list(getattr(model, "classes_", []))
input_modes = infer_feature_input_modes(model, feature_names)
text_features = [f for f in feature_names if input_modes.get(f) == "text"]
numeric_features = [f for f in feature_names if input_modes.get(f) == "numeric"]

st.success(f"Model loaded: {model.__class__.__name__}")
st.write("Expected feature columns:")
st.code(", ".join(feature_names))

if text_features:
    st.info(
        "Detected mixed input types from model behavior. "
        f"Text features: {', '.join(text_features)}. "
        f"Numeric features: {', '.join(numeric_features)}."
    )
else:
    st.info(
        "Detected numeric-only model input. "
        "Use encoded numeric values for categorical features."
    )

with st.expander("Model input analysis"):
    analysis_df = pd.DataFrame(
        {
            "feature": feature_names,
            "input_mode": [input_modes.get(f, "numeric") for f in feature_names],
        }
    )
    st.dataframe(analysis_df, use_container_width=True)

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction CSV"])

with tab_single:
    st.subheader("Predict One Student")
    if text_features:
        st.caption("Inputs are auto-configured from model behavior (text or numeric per feature).")
    else:
        st.caption("All fields are numeric. Enter encoded numbers for categorical features.")
    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)

        values = {}
        for i, col in enumerate(feature_names):
            target_col = col1 if i % 2 == 0 else col2
            input_mode = input_modes.get(col, "numeric")

            if input_mode == "text":
                values[col] = target_col.text_input(
                    col,
                    value="",
                    help="Enter the text value expected by the model preprocessing pipeline.",
                )
            elif col == "Age":
                values[col] = target_col.number_input(col, min_value=5.0, max_value=100.0, value=20.0, step=1.0)
            elif col == "Daily_Usage_Hours":
                values[col] = target_col.number_input(col, min_value=0.0, max_value=24.0, value=2.0, step=0.1)
            else:
                values[col] = target_col.number_input(
                    f"{col} (encoded)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    help="Enter the encoded numeric value used during training.",
                )

        submitted = st.form_submit_button("Predict")

    if submitted:
        parsed_values = {}
        empty_text_fields = []

        for col in feature_names:
            if input_modes.get(col) == "text":
                text_value = str(values[col]).strip()
                if not text_value:
                    empty_text_fields.append(col)
                parsed_values[col] = text_value
            else:
                parsed_values[col] = float(values[col])

        if empty_text_fields:
            st.error(f"Please fill these text fields: {', '.join(empty_text_fields)}")
            st.stop()

        input_df = pd.DataFrame([parsed_values], columns=feature_names)
        try:
            pred = model.predict(input_df)[0]
        except Exception as exc:
            st.error(f"Prediction failed with current inputs: {exc}")
            st.stop()

        st.metric("Predicted Class", str(pred))

        if hasattr(model, "predict_proba") and classes:
            proba = model.predict_proba(input_df)[0]
            proba_df = pd.DataFrame(
                {
                    "class": classes,
                    "probability": proba,
                }
            ).sort_values("probability", ascending=False)
            st.write("Class probabilities:")
            st.dataframe(proba_df, use_container_width=True)

with tab_batch:
    st.subheader("Predict in Batch")
    st.write("Upload a CSV with exactly these columns and order:")
    st.code("\n".join(feature_names))

    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if csv_file is not None:
        try:
            batch_df = pd.read_csv(csv_file)
        except Exception as exc:
            st.error(f"CSV read error: {exc}")
            st.stop()

        valid, msg = validate_input_columns(batch_df, feature_names)
        if not valid:
            st.error(msg)
            st.stop()

        batch_df = batch_df[feature_names]
        st.write("Input preview:")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Run Batch Prediction"):
            preds = model.predict(batch_df)
            result_df = batch_df.copy()
            result_df["prediction"] = preds

            if hasattr(model, "predict_proba") and classes:
                probas = model.predict_proba(batch_df)
                for idx, cls in enumerate(classes):
                    result_df[f"prob_class_{cls}"] = probas[:, idx]

            st.write("Prediction results:")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "Download results as CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="student_satisfaction_predictions.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption("If predictions seem odd, verify your categorical encoding matches training-time encoding.")
