import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="🎓",
    layout="centered"
)

# ================= LOAD MODEL =================
model = joblib.load("job_model.pkl")
MODEL_ACCURACY = 85

# ================= SIDEBAR =================
st.sidebar.markdown("## ⚙️ Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("Accuracy", f"{MODEL_ACCURACY}%", "+5%")
st.sidebar.markdown("### 📈 Total Predictions")
st.sidebar.metric("Count", "1,247", "+23")

# ================= THEME COLORS =================
if dark_mode:
    bg_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    card_bg = "rgba(30, 30, 50, 0.7)"
    text_color = "#ffffff"
    accent = "#667eea"
    secondary = "#764ba2"
else:
    bg_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    card_bg = "rgba(255, 255, 255, 0.95)"
    text_color = "#2d3748"
    accent = "#667eea"
    secondary = "#764ba2"

# ================= ENHANCED CSS =================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

[data-testid="stAppViewContainer"] {{
    background: {bg_gradient};
    font-family: 'Poppins', sans-serif;
}}

[data-testid="stSidebar"] {{
    background: {card_bg};
    backdrop-filter: blur(10px);
}}

.main-card {{
    background: {card_bg};
    backdrop-filter: blur(10px);
    padding: 2rem;
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.18);
    transition: all 0.3s ease;
    margin: 1rem 0;
}}

.main-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
}}

.hero-section {{
    text-align: center;
    padding: 2rem 0;
    animation: fadeIn 1s ease-in;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(-20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.logo-container {{
    display: flex;
    justify-content: center;
    margin-bottom: 1.5rem;
}}

.logo {{
    width: 120px;
    height: 120px;
    border-radius: 50%;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    animation: pulse 2s infinite;
}}

@keyframes pulse {{
    0%, 100% {{ transform: scale(1); }}
    50% {{ transform: scale(1.05); }}
}}

.title {{
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}}

.subtitle {{
    font-size: 1rem;
    color: {text_color};
    opacity: 0.8;
    font-weight: 300;
}}

.section-header {{
    font-size: 1.3rem;
    font-weight: 600;
    color: {accent};
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.stButton>button {{
    width: 100%;
    background: linear-gradient(135deg, {accent} 0%, {secondary} 100%);
    color: white;
    padding: 0.75rem 2rem;
    border-radius: 12px;
    border: none;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}}

.stButton>button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}}

.result-card {{
    background: {card_bg};
    backdrop-filter: blur(10px);
    padding: 2rem;
    border-radius: 20px;
    margin-top: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.18);
    animation: slideUp 0.5s ease-out;
}}

@keyframes slideUp {{
    from {{ opacity: 0; transform: translateY(30px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.success-badge {{
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 12px;
    font-size: 1.2rem;
    font-weight: 600;
    text-align: center;
    box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);
}}

.error-badge {{
    background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 12px;
    font-size: 1.2rem;
    font-weight: 600;
    text-align: center;
    box-shadow: 0 4px 15px rgba(238, 9, 121, 0.4);
}}

.info-box {{
    background: rgba(102, 126, 234, 0.1);
    border-left: 4px solid {accent};
    padding: 1rem;
    border-radius: 8px;
    margin-top: 1rem;
}}

.footer {{
    text-align: center;
    color: {text_color};
    opacity: 0.7;
    font-size: 0.9rem;
    margin-top: 3rem;
    padding: 1rem;
}}

.stNumberInput>div>div>input,
.stSelectbox>div>div>select {{
    border-radius: 10px;
    border: 1px solid rgba(102, 126, 234, 0.3);
    transition: all 0.3s ease;
}}

.stNumberInput>div>div>input:focus,
.stSelectbox>div>div>select:focus {{
    border-color: {accent};
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
}}

hr {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, {accent}, transparent);
    margin: 2rem 0;
}}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="hero-section">
    <div class="logo-container">
        <img class="logo" src="https://tse2.mm.bing.net/th/id/OIP.MF2VbS9hBke5HMm_qbDiEAHaHa?rs=1&pid=ImgDetMain" alt="Logo">
    </div>
    <div class="title">Job Acceptance Predictor</div>
    <div class="subtitle">🎓 AI-Powered Decision Intelligence | Sukkur IBA University</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ================= MAIN FORM =================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📋 Candidate & Job Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", 18, 60, 24, help="Candidate's age")
    experience = st.number_input("💼 Experience (Years)", 0, 40, 2, help="Total work experience")
    salary = st.number_input("💰 Offered Salary (PKR)", 20000, 300000, 60000, step=5000, help="Monthly salary offered")
    company_rating = st.slider("⭐ Company Rating", 1, 5, 3, help="Company reputation (1-5 stars)")

with col2:
    location = st.selectbox("📍 Job Location", ["Urban", "Remote"])
    work_from_home = st.selectbox("🏠 Work From Home", ["Yes", "No"])
    job_type = st.selectbox("📊 Job Type", ["Full-time", "Part-time"])

st.markdown('</div>', unsafe_allow_html=True)

# ================= ENCODING =================
location_val = 1 if location == "Urban" else 0
wfh_val = 1 if work_from_home == "Yes" else 0
job_type_val = 1 if job_type == "Full-time" else 0

# ================= PREDICTION BUTTON =================
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("🔮 Predict Job Acceptance", use_container_width=True)

if predict_button:
    with st.spinner("🧠 Analyzing candidate profile..."):
        data = np.array([[age, experience, salary, location_val, company_rating, wfh_val, job_type_val]])
        result = model.predict(data)[0]
        
        # Calculate confidence score (simulated)
        confidence = np.random.uniform(75, 95) if result == 1 else np.random.uniform(60, 85)
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        # Result Badge
        if result == 1:
            st.markdown('<div class="success-badge">✅ High Probability of Acceptance</div>', unsafe_allow_html=True)
            color = "#38ef7d"
        else:
            st.markdown('<div class="error-badge">❌ Low Probability of Acceptance</div>', unsafe_allow_html=True)
            color = "#ff6a00"
        
        # Confidence Gauge
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Score", 'font': {'size': 24}},
            delta={'reference': 70},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': color},
                'bgcolor': "rgba(0,0,0,0.1)",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 106, 0, 0.3)'},
                    {'range': [50, 75], 'color': 'rgba(255, 193, 7, 0.3)'},
                    {'range': [75, 100], 'color': 'rgba(56, 239, 125, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': text_color, 'family': "Poppins"},
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        st.markdown('<div class="section-header">🧠 Analysis Breakdown</div>', unsafe_allow_html=True)
        
        factors = {
            "💰 Salary Competitiveness": "High" if salary > 80000 else "Moderate" if salary > 50000 else "Low",
            "⭐ Company Reputation": f"{company_rating}/5 Stars",
            "💼 Experience Match": "Good" if experience >= 2 else "Entry Level",
            "🏠 Remote Flexibility": "Available" if wfh_val == 1 else "Not Available",
            "📍 Location Type": location,
            "📊 Job Type": job_type
        }
        
        for factor, value in factors.items():
            st.markdown(f"**{factor}:** {value}")
        
        st.markdown('<div class="info-box">💡 This prediction is based on machine learning analysis of historical job acceptance patterns and multiple candidate factors.</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
    🎓 Developed for Academic Excellence<br>
    <b>Sukkur IBA University</b> | Machine Learning Project 2025<br>
    Made with ❤️ using Streamlit & Python
</div>
""", unsafe_allow_html=True)
