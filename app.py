import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("job_model.pkl")

st.title("Job Acceptance Prediction System")
st.write("Predict whether a candidate will accept a job offer")

# User inputs
age = st.number_input("Age", min_value=18, max_value=60, value=25)
experience = st.number_input("Experience (years)", min_value=0, max_value=40, value=2)
salary = st.number_input("Offered Salary", min_value=20000, max_value=300000, value=60000)
location = st.selectbox("Job Location", ["Urban", "Remote"])
company_rating = st.slider("Company Rating", 1, 5, 3)
work_from_home = st.selectbox("Work From Home", ["Yes", "No"])
job_type = st.selectbox("Job Type", ["Full-time", "Part-time"])

# Encoding (SAME as training)
location = 1 if location == "Urban" else 0
work_from_home = 1 if work_from_home == "Yes" else 0
job_type = 1 if job_type == "Full-time" else 0

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, experience, salary, location,
                             company_rating, work_from_home, job_type]])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Candidate is likely to ACCEPT the job")
    else:
        st.error("❌ Candidate is likely to REJECT the job")
