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
MODEL_ACCURACY = "85%"   # training ke baad mention ki jati hai

# ================= SIDEBAR =================
st.sidebar.title("⚙️ App Settings")

theme = st.sidebar.toggle("🌙 Dark Mode")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Accuracy")
st.sidebar.success(MODEL_ACCURACY)

# ================= THEME CSS =================
if theme:
    bg = "#0e1117"
    text = "white"
    card = "#161b22"
else:
    bg = "#f4f6f9"
    text = "#000000"
    card = "white"

st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
}}
.main {{
    background-color: {card};
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}}
h1 {{
    text-align: center;
}}
.footer {{
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 25px;
}}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.image(
    "https://crystalpng.com/wp-content/uploads/2022/03/SIBA_Logo.png",
    width=130
)

st.markdown("<h1>Job Acceptance Prediction System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Machine Learning Project | Sukkur IBA University</p>",
    unsafe_allow_html=True
)

st.divider()

# ================= INPUT SECTION =================
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
        st.success("✅ Prediction Result: Candidate will ACCEPT the job")
    else:
        st.error("❌ Prediction Result: Candidate will REJECT the job")

    # ================= EXPLANATION =================
    st.subheader("🧠 Prediction Explanation")
    st.markdown("""
    This prediction is generated using a **Machine Learning classification model**.

    **Key decision factors include:**
    - Offered **salary**
    - Candidate **experience**
    - **Company rating**
    - **Work-from-home availability**
    - **Job type** and **location**

    The model analyzes these parameters together and predicts whether
    a candidate is likely to **accept or reject** a job offer.
    """)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
    Developed for Academic Use <br>
    Sukkur IBA University | Machine Learning Project
</div>
""", unsafe_allow_html=True)
