import json
from pathlib import Path

import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student Health Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "student_health_model.cbm"
INFO_PATH = BASE_DIR / "preprocessing_info.json"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_data
def load_preprocessing_info():
    with open(INFO_PATH, "r") as f:
        return json.load(f)


model = load_model()
preprocessing_info = load_preprocessing_info()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #e5e7eb;
    }

    .result-title {
        font-size: 32px;
        font-weight: 800;
    }

    .confidence {
        font-size: 20px;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🩺 Health Predictor")

    st.markdown("---")

    st.markdown(
        """
        ### About

        This interactive demo uses a tuned **CatBoost
        multiclass classification model** to predict one of
        three student health-condition categories.

        **Model Performance**

        Balanced Accuracy:

        **90.76%**
        """
    )

    st.markdown("---")

    st.caption(
        "Educational machine-learning demonstration. "
        "This is not a medical diagnosis."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🩺 Student Health Risk Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Predict a student's health condition using lifestyle and "
    "health-related information."
    "</div>",
    unsafe_allow_html=True,
)

st.info(
    "Enter the student's information below and click "
    "**Predict Health Condition**."
)


# =========================================================
# INPUTS
# =========================================================

st.header("Student Information")

st.subheader("Health & Lifestyle Metrics")

col1, col2, col3 = st.columns(3)

with col1:

    sleep_duration = st.number_input(
        "Sleep Duration (hours)",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.1,
    )

    heart_rate = st.number_input(
        "Heart Rate (bpm)",
        min_value=30.0,
        max_value=220.0,
        value=72.0,
        step=1.0,
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=22.0,
        step=0.1,
    )


with col2:

    calorie_expenditure = st.number_input(
        "Calorie Expenditure",
        min_value=0.0,
        max_value=10000.0,
        value=2200.0,
        step=10.0,
    )

    step_count = st.number_input(
        "Step Count",
        min_value=0.0,
        max_value=100000.0,
        value=7000.0,
        step=100.0,
    )

    exercise_duration = st.number_input(
        "Exercise Duration (minutes)",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=1.0,
    )


with col3:

    water_intake = st.number_input(
        "Water Intake (liters)",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.1,
    )


# =========================================================
# CATEGORICAL FEATURES
# =========================================================

st.subheader("Lifestyle Information")

col1, col2, col3 = st.columns(3)

with col1:

    diet_type = st.selectbox(
        "Diet Type",
        ["veg", "non-veg"],
    )

    stress_level = st.selectbox(
        "Stress Level",
        ["low", "medium", "high"],
    )


with col2:

    sleep_quality = st.selectbox(
        "Sleep Quality",
        ["poor", "average", "good"],
    )

    physical_activity_level = st.selectbox(
        "Physical Activity Level",
        ["sedentary", "moderate", "active"],
    )


with col3:

    smoking_alcohol = st.selectbox(
        "Smoking / Alcohol",
        ["yes", "occasional", "no"],
    )

    gender = st.selectbox(
        "Gender",
        ["female", "male", "other"],
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Health Condition",
    use_container_width=True,
    type="primary",
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Keep EXACT same feature names/order used during training
    input_data = pd.DataFrame(
        [
            {
                "sleep_duration": sleep_duration,
                "heart_rate": heart_rate,
                "bmi": bmi,
                "calorie_expenditure": calorie_expenditure,
                "step_count": step_count,
                "exercise_duration": exercise_duration,
                "water_intake": water_intake,
                "diet_type": diet_type,
                "stress_level": stress_level,
                "sleep_quality": sleep_quality,
                "physical_activity_level": physical_activity_level,
                "smoking_alcohol": smoking_alcohol,
                "gender": gender,
            }
        ]
    )

    # Make sure columns are in the exact order used by the model
    feature_order = (
        preprocessing_info["numeric_features"]
        + preprocessing_info["categorical_features"]
    )

    input_data = input_data[feature_order]

    # Prediction
    prediction = model.predict(input_data)

    predicted_index = int(prediction[0][0])

    classes = preprocessing_info["label_classes"]

    predicted_class = classes[predicted_index]

    # Prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    confidence = float(max(probabilities)) * 100


    # =====================================================
    # RESULT
    # =====================================================

    st.header("Prediction Result")

    if predicted_class == "fit":

        st.success(
            f"### 🟢 FIT\n\n"
            f"Model confidence: **{confidence:.1f}%**"
        )

    elif predicted_class == "unhealthy":

        st.warning(
            f"### 🟡 UNHEALTHY\n\n"
            f"Model confidence: **{confidence:.1f}%**"
        )

    elif predicted_class == "at-risk":

        st.error(
            f"### 🔴 AT-RISK\n\n"
            f"Model confidence: **{confidence:.1f}%**"
        )

    else:

        st.info(
            f"### {predicted_class.upper()}\n\n"
            f"Model confidence: **{confidence:.1f}%**"
        )


    # =====================================================
    # PROBABILITY BREAKDOWN
    # =====================================================

    st.subheader("Prediction Probabilities")

    probability_data = pd.DataFrame(
        {
            "Health Condition": classes,
            "Probability": probabilities,
        }
    )

    probability_data["Probability"] = (
        probability_data["Probability"] * 100
    )

    probability_data["Probability"] = probability_data[
        "Probability"
    ].round(2)

    st.dataframe(
        probability_data,
        use_container_width=True,
        hide_index=True,
    )


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    with st.expander("View Submitted Information"):

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True,
        )


    st.caption(
        "⚠️ This tool is an educational machine-learning "
        "demonstration and should not be used for medical "
        "diagnosis or treatment decisions."
    )