import streamlit as st
import joblib
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="🎓",
    layout="centered"
)

# ================= LOAD MODEL =================
model = joblib.load("job_model.pkl")
MODEL_ACCURACY = "85%"

# ================= SIDEBAR =================
st.sidebar.title("⚙️ App Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Accuracy")
st.sidebar.success(MODEL_ACCURACY)

# ================= THEME COLORS =================
if dark_mode:
    bg = "#0e1117"
    text = "#ffffff"
    card = "#161b22"
    accent = "#00c0ff"
else:
    bg = "#f4f6f9"
    text = "#000000"
    card = "#ffffff"
    accent = "#004aad"

# ================= GLOBAL CSS =================
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {bg};
    color: {text};
}}

.card {{
    background-color: {card};
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 6px 25px rgba(0,0,0,0.25);
}}

.title {{
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: {accent};
}}

.subtitle {{
    text-align: center;
    font-size: 15px;
    color: gray;
}}

.logo {{
    display: block;
    margin-left: auto;
    margin-right: auto;
    border-radius: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}}

.footer {{
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 30px;
}}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<img class="logo" src="https://tse2.mm.bing.net/th/id/OIP.MF2VbS9hBke5HMm_qbDiEAHaHa?rs=1&pid=ImgDetMain&o=7&rm=3" width="140">
""", unsafe_allow_html=True)

st.markdown('<div class="title">Job Acceptance Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Project | Sukkur IBA University</div>', unsafe_allow_html=True)

st.divider()

# ================= MAIN CARD =================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📋 Candidate & Job Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 60, 24)
    experience = st.number_input("Experience (Years)", 0, 40, 2)
    salary = st.number_input("Offered Salary (PKR)", 20000, 300000, 60000)
    company_rating = st.slider("Company Rating", 1, 5, 3)

with col2:
    location = st.selectbox("Job Location", ["Urban", "Remote"])
    work_from_home = st.selectbox("Work From Home", ["Yes", "No"])
    job_type = st.selectbox("Job Type", ["Full-time", "Part-time"])

# ================= ENCODING =================
location_val = 1 if location == "Urban" else 0
wfh_val = 1 if work_from_home == "Yes" else 0
job_type_val = 1 if job_type == "Full-time" else 0

st.divider()

# ================= PREDICTION =================
if st.button("🔍 Predict Job Acceptance"):
    data = np.array([[age, experience, salary,
                      location_val, company_rating,
                      wfh_val, job_type_val]])

    result = model.predict(data)[0]

    st.divider()

    if result == 1:
        st.success("✅ Candidate is likely to ACCEPT the job")
        st.progress(85)
    else:
        st.error("❌ Candidate is likely to REJECT the job")
        st.progress(40)

    st.subheader("🧠 Prediction Explanation")
    st.markdown("""
    The prediction is based on:
    - Salary & experience balance  
    - Company reputation  
    - Work-from-home availability  
    - Job type and location  

    The model evaluates these parameters together
    and produces a final decision.
    """)

st.markdown('</div>', unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
    Developed for Academic Use<br>
    Sukkur IBA University | Machine Learning Project
</div>
""", unsafe_allow_html=True)
