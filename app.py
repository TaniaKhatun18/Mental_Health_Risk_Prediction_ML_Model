import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# ----------------- Helper to Load Background Image -----------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ----------------- Load Background Image -----------------
img_base64 = get_base64_image("bg.jpg")

# ----------------- Streamlit Page Config and Style -----------------
st.set_page_config(page_title="Mental Health Risk Predictor", layout="centered")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }}
    h1, h2, h3, h4, h5, h6, p, div, span, label {{
        color: white !important;
    }}
    .stTextInput input, .stSelectbox div {{
        background-color: rgba(255, 255, 255, 0.1);
        color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Load Model Components -----------------
model = joblib.load('mental_health_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
model_features = joblib.load('model_features.pkl')  # list of column names

# ----------------- UI -----------------
st.title("🧠 Mental Health Risk Predictor")
st.write("Predict your mental health risk level based on lifestyle and wellness factors.")

with st.form("risk_form"):
    age = st.slider("Age", 15, 100, 30)
    gender = st.selectbox("Gender", ['Male', 'Female', 'Non-binary', 'Prefer not to say'])
    employment_status = st.selectbox("Employment Status", ['Employed', 'Unemployed', 'Student', 'Self-employed'])
    work_environment = st.selectbox("Work Environment", ['Remote', 'On-site', 'Hybrid'])
    mental_health_history = st.selectbox("Mental Health History", ['Yes', 'No'])
    seeks_treatment = st.selectbox("Seeks Treatment", ['Yes', 'No'])
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    physical_activity_days = st.slider("Physical Activity Days per Week", 0, 7, 3)
    depression_score = st.slider("Depression Score (0-30)", 0, 30, 10)
    anxiety_score = st.slider("Anxiety Score (0-30)", 0, 30, 10)
    social_support_score = st.slider("Social Support Score (0-100)", 0, 100, 50)
    productivity_score = st.slider("Productivity Score (0-100)", 0, 100, 50)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_dict = {
        'age': age,
        'gender': gender,
        'employment_status': employment_status,
        'work_environment': work_environment,
        'mental_health_history': mental_health_history,
        'seeks_treatment': seeks_treatment,
        'stress_level': stress_level,
        'sleep_hours': sleep_hours,
        'physical_activity_days': physical_activity_days,
        'depression_score': depression_score,
        'anxiety_score': anxiety_score,
        'social_support_score': social_support_score,
        'productivity_score': productivity_score
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical columns
    for col in label_encoders:
        le = label_encoders[col]
        if input_df[col].iloc[0] not in le.classes_:
            le.classes_ = np.append(le.classes_, input_df[col].iloc[0])
        input_df[col] = le.transform(input_df[col])

    # Scale numerical columns
    num_cols = ['age', 'stress_level', 'sleep_hours', 'physical_activity_days',
                'depression_score', 'anxiety_score', 'social_support_score', 'productivity_score']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # One-hot encoding and align
    input_encoded = pd.get_dummies(input_df)

    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_features]

    # Predict
    pred = model.predict(input_encoded)[0]
    pred_proba = model.predict_proba(input_encoded)[0]
    label = target_encoder.inverse_transform([pred])[0]
    confidence = round(pred_proba[pred] * 100, 2)

    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Mental Health Risk: **{label}** ({confidence}%)")

# ----------------- Footer -----------------
st.caption("© 2025 Tania Khatun. All rights reserved.")
