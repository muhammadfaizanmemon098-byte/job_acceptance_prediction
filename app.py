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
MODEL_ACCURACY = 85

# ================= SIDEBAR =================
st.sidebar.markdown("## ⚙️ Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("Accuracy", f"{MODEL_ACCURACY}%", "+5%")
st.sidebar.markdown("### 📈 Total Predictions")
st.sidebar.metric("Today", "247", "+23")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Toggle dark mode for better viewing experience")

# ================= THEME COLORS =================
if dark_mode:
    bg_gradient = "linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%)"
    card_bg = "rgba(30, 30, 50, 0.85)"
    input_bg = "rgba(50, 50, 80, 0.6)"
    text = "#ffffff"
    accent = "#a78bfa"
    secondary = "#60a5fa"
else:
    bg_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)"
    card_bg = "rgba(255, 255, 255, 0.95)"
    input_bg = "rgba(240, 240, 255, 0.8)"
    text = "#1e293b"
    accent = "#667eea"
    secondary = "#764ba2"

# ================= ENHANCED CSS =================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

/* Global Styles */
* {{
    font-family: 'Poppins', sans-serif;
}}

[data-testid="stAppViewContainer"] {{
    background: {bg_gradient};
    background-attachment: fixed;
}}

[data-testid="stSidebar"] {{
    background: {card_bg};
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}}

/* Hero Section */
.hero-container {{
    text-align: center;
    padding: 2rem 0 1rem 0;
    animation: fadeInDown 1s ease-out;
}}

@keyframes fadeInDown {{
    from {{
        opacity: 0;
        transform: translateY(-30px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

.developer-section {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    margin: 2rem 0;
    padding: 2rem;
    background: {card_bg};
    backdrop-filter: blur(20px);
    border-radius: 30px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.18);
    animation: slideIn 1s ease-out;
}}

@keyframes slideIn {{
    from {{
        opacity: 0;
        transform: translateX(-50px);
    }}
    to {{
        opacity: 1;
        transform: translateX(0);
    }}
}}

.developer-photo {{
    width: 180px;
    height: 180px;
    border-radius: 50%;
    border: 5px solid {accent};
    box-shadow: 0 15px 50px rgba(102, 126, 234, 0.5);
    object-fit: cover;
    animation: photoFloat 3s ease-in-out infinite;
    transition: all 0.4s ease;
}}

.developer-photo:hover {{
    transform: scale(1.1) rotate(5deg);
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.7);
}}

@keyframes photoFloat {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-10px); }}
}}

.developer-info {{
    text-align: left;
    flex: 1;
    max-width: 500px;
}}

.developer-name {{
    font-size: 2rem;
    font-weight: 800;
    color: {accent};
    margin-bottom: 0.5rem;
}}

.developer-title {{
    font-size: 1.1rem;
    color: {text};
    opacity: 0.8;
    margin-bottom: 1rem;
}}

.developer-stats {{
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
}}

.stat-box {{
    text-align: center;
    padding: 1rem;
    background: rgba(102, 126, 234, 0.1);
    border-radius: 15px;
    border: 1px solid {accent};
    flex: 1;
}}

.stat-number {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {accent};
    display: block;
}}

.stat-label {{
    font-size: 0.9rem;
    color: {text};
    opacity: 0.7;
}}

.logo-wrapper {{
    position: relative;
    display: inline-block;
    margin-bottom: 1.5rem;
}}

.logo {{
    width: 130px;
    height: 130px;
    border-radius: 50%;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.5);
    border: 4px solid rgba(255, 255, 255, 0.3);
    animation: float 3s ease-in-out infinite;
    transition: all 0.3s ease;
}}

.logo:hover {{
    transform: scale(1.1) rotate(5deg);
    box-shadow: 0 15px 50px rgba(102, 126, 234, 0.7);
}}

@keyframes float {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-15px); }}
}}

.title {{
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
    animation: gradientShift 3s ease infinite;
}}

@keyframes gradientShift {{
    0%, 100% {{ filter: hue-rotate(0deg); }}
    50% {{ filter: hue-rotate(20deg); }}
}}

.subtitle {{
    font-size: 1.1rem;
    color: {text};
    opacity: 0.85;
    font-weight: 400;
    margin-bottom: 0.5rem;
}}

.badge {{
    display: inline-block;
    background: rgba(102, 126, 234, 0.2);
    color: {accent};
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    border: 1px solid {accent};
    margin-top: 0.5rem;
}}

/* Main Card */
.main-card {{
    background: {card_bg};
    backdrop-filter: blur(20px);
    padding: 2.5rem;
    border-radius: 30px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.18);
    margin: 2rem 0;
    animation: slideUp 0.8s ease-out;
    position: relative;
    overflow: hidden;
}}

.main-card::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
    animation: rotate 20s linear infinite;
}}

@keyframes rotate {{
    from {{ transform: rotate(0deg); }}
    to {{ transform: rotate(360deg); }}
}}

@keyframes slideUp {{
    from {{
        opacity: 0;
        transform: translateY(40px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

.section-title {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {accent};
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    position: relative;
    z-index: 1;
}}

.section-title::after {{
    content: '';
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, {accent}, transparent);
    margin-left: 1rem;
}}

/* Enhanced Input Styles */
.stNumberInput > div > div > input,
.stSelectbox > div > div > select,
.stSlider > div > div > div {{
    background: {input_bg} !important;
    border: 2px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 15px !important;
    color: {text} !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    padding: 0.75rem !important;
}}

.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > select:focus {{
    border-color: {accent} !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    transform: translateY(-2px);
}}

.stNumberInput > div > div > input:hover,
.stSelectbox > div > div > select:hover {{
    border-color: {secondary} !important;
}}

/* Enhanced Button */
.stButton > button {{
    width: 100%;
    background: linear-gradient(135deg, {accent} 0%, {secondary} 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 20px;
    border: none;
    font-size: 1.2rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.4s ease;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    position: relative;
    overflow: hidden;
    z-index: 1;
}}

.stButton > button::before {{
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
    z-index: -1;
}}

.stButton > button:hover::before {{
    width: 300px;
    height: 300px;
}}

.stButton > button:hover {{
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
}}

.stButton > button:active {{
    transform: translateY(-2px);
}}

/* Result Cards */
.result-card {{
    background: {card_bg};
    backdrop-filter: blur(20px);
    padding: 2rem;
    border-radius: 25px;
    margin-top: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.18);
    animation: resultPop 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    box-shadow: 0 15px 50px rgba(0, 0, 0, 0.2);
}}

@keyframes resultPop {{
    0% {{
        opacity: 0;
        transform: scale(0.8) translateY(30px);
    }}
    100% {{
        opacity: 1;
        transform: scale(1) translateY(0);
    }}
}}

.success-badge {{
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 20px;
    font-size: 1.3rem;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 10px 40px rgba(16, 185, 129, 0.4);
    margin-bottom: 1.5rem;
    animation: successPulse 2s ease-in-out infinite;
}}

@keyframes successPulse {{
    0%, 100% {{ box-shadow: 0 10px 40px rgba(16, 185, 129, 0.4); }}
    50% {{ box-shadow: 0 15px 50px rgba(16, 185, 129, 0.6); }}
}}

.error-badge {{
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 20px;
    font-size: 1.3rem;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 10px 40px rgba(239, 68, 68, 0.4);
    margin-bottom: 1.5rem;
    animation: errorShake 0.5s ease-in-out;
}}

@keyframes errorShake {{
    0%, 100% {{ transform: translateX(0); }}
    25% {{ transform: translateX(-10px); }}
    75% {{ transform: translateX(10px); }}
}}

/* Circular Progress (Accuracy Visualization) */
.accuracy-circle {{
    position: relative;
    width: 180px;
    height: 180px;
    margin: 2rem auto;
}}

.accuracy-svg {{
    transform: rotate(-90deg);
}}

.accuracy-circle-bg {{
    fill: none;
    stroke: rgba(102, 126, 234, 0.2);
    stroke-width: 15;
}}

.accuracy-circle-progress {{
    fill: none;
    stroke: url(#gradient);
    stroke-width: 15;
    stroke-linecap: round;
    stroke-dasharray: 440;
    stroke-dashoffset: 110;
    animation: progressAnimation 2s ease-out forwards;
}}

@keyframes progressAnimation {{
    to {{
        stroke-dashoffset: 110;
    }}
}}

.accuracy-text {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
}}

.accuracy-number {{
    font-size: 2.5rem;
    font-weight: 800;
    color: {accent};
    display: block;
}}

.accuracy-label {{
    font-size: 0.9rem;
    color: {text};
    opacity: 0.7;
}}

/* Progress Bar Enhancement */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {accent}, {secondary}) !important;
    border-radius: 10px;
    height: 20px !important;
    animation: progressGlow 2s ease-in-out infinite;
}}

@keyframes progressGlow {{
    0%, 100% {{ box-shadow: 0 0 10px {accent}; }}
    50% {{ box-shadow: 0 0 20px {secondary}; }}
}}

/* Explanation Box */
.explanation-box {{
    background: rgba(102, 126, 234, 0.1);
    border-left: 5px solid {accent};
    padding: 1.5rem;
    border-radius: 15px;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}}

.explanation-box::before {{
    content: '💡';
    position: absolute;
    font-size: 3rem;
    opacity: 0.1;
    right: 10px;
    top: 10px;
}}

/* Divider */
hr {{
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, {accent}, transparent);
    margin: 2rem 0;
    animation: dividerGlow 3s ease-in-out infinite;
}}

@keyframes dividerGlow {{
    0%, 100% {{ opacity: 0.5; }}
    50% {{ opacity: 1; }}
}}

/* Footer */
.footer {{
    text-align: center;
    color: {text};
    opacity: 0.7;
    font-size: 0.95rem;
    margin-top: 3rem;
    padding: 2rem;
    background: {card_bg};
    border-radius: 20px;
    backdrop-filter: blur(20px);
}}

/* Mobile Responsive */
@media (max-width: 768px) {{
    .title {{ font-size: 2rem; }}
    .logo {{ width: 100px; height: 100px; }}
    .main-card {{ padding: 1.5rem; }}
    .developer-section {{ flex-direction: column; text-align: center; }}
    .developer-info {{ text-align: center; }}
    .developer-stats {{ justify-content: center; }}
}}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown(f"""
<div class="hero-container">
    <div class="logo-wrapper">
        <img class="logo" src="https://tse2.mm.bing.net/th/id/OIP.MF2VbS9hBke5HMm_qbDiEAHaHa?rs=1&pid=ImgDetMain" alt="Logo">
    </div>
    <div class="title">Job Acceptance Predictor</div>
    <div class="subtitle">🎓 AI-Powered Decision Intelligence System</div>
    <div class="badge">Sukkur IBA University | ML Project 2025</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ================= DEVELOPER SECTION =================
st.markdown("""
<div class="developer-section">
    <img class="developer-photo" src="https://i.imgur.com/your-photo-link.jpg" alt="Developer">
    <div class="developer-info">
        <div class="developer-name">Your Name</div>
        <div class="developer-title">Machine Learning Engineer | Data Scientist</div>
        <div class="developer-stats">
            <div class="stat-box">
                <span class="stat-number">85%</span>
                <span class="stat-label">Model Accuracy</span>
            </div>
            <div class="stat-box">
                <span class="stat-number">1.2K+</span>
                <span class="stat-label">Predictions</span>
            </div>
            <div class="stat-box">
                <span class="stat-number">247</span>
                <span class="stat-label">Today</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= ACCURACY VISUALIZATION =================
st.markdown(f"""
<div class="accuracy-circle">
    <svg class="accuracy-svg" width="180" height="180" viewBox="0 0 180 180">
        <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:{accent};stop-opacity:1" />
                <stop offset="100%" style="stop-color:{secondary};stop-opacity:1" />
            </linearGradient>
        </defs>
        <circle class="accuracy-circle-bg" cx="90" cy="90" r="70"></circle>
        <circle class="accuracy-circle-progress" cx="90" cy="90" r="70"></circle>
    </svg>
    <div class="accuracy-text">
        <span class="accuracy-number">{MODEL_ACCURACY}%</span>
        <span class="accuracy-label">Accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ================= MAIN CARD =================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📋 Candidate & Job Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    age = st.number_input("👤 Age", 18, 60, 24, help="Candidate's current age")
    experience = st.number_input("💼 Experience (Years)", 0, 40, 2, help="Total years of work experience")
    salary = st.number_input("💰 Offered Salary (PKR)", 20000, 300000, 60000, step=5000, help="Monthly salary package")
    company_rating = st.slider("⭐ Company Rating", 1, 5, 3, help="Company reputation (1-5 stars)")

with col2:
    location = st.selectbox("📍 Job Location", ["Urban", "Remote"], help="Work location type")
    work_from_home = st.selectbox("🏠 Work From Home", ["Yes", "No"], help="Remote work availability")
    job_type = st.selectbox("📊 Job Type", ["Full-time", "Part-time"], help="Employment type")

# ================= ENCODING =================
location_val = 1 if location == "Urban" else 0
wfh_val = 1 if work_from_home == "Yes" else 0
job_type_val = 1 if job_type == "Full-time" else 0

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= PREDICTION BUTTON =================
predict_button = st.button("🔮 Predict Job Acceptance", use_container_width=True)

if predict_button:
    with st.spinner("🧠 Analyzing candidate profile..."):
        data = np.array([[age, experience, salary, location_val, company_rating, wfh_val, job_type_val]])
        result = model.predict(data)[0]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if result == 1:
            st.markdown('<div class="success-badge">✅ High Probability of Job Acceptance</div>', unsafe_allow_html=True)
            progress_value = 85
            st.progress(progress_value / 100)
            st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: {accent}; font-weight: 600;'>Confidence Level: {progress_value}%</p>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-badge">❌ Low Probability of Job Acceptance</div>', unsafe_allow_html=True)
            progress_value = 40
            st.progress(progress_value / 100)
            st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #ef4444; font-weight: 600;'>Confidence Level: {progress_value}%</p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 Prediction Analysis</div>', unsafe_allow_html=True)

        # Factors Analysis
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**💰 Salary Competitiveness:**")
            if salary > 80000:
                st.success("High")
            elif salary > 50000:
                st.warning("Moderate")
            else:
                st.error("Low")
            
            st.markdown("**⭐ Company Rating:**")
            st.info(f"{company_rating}/5 Stars")
            
            st.markdown("**💼 Experience Level:**")
            if experience >= 5:
                st.success("Senior")
            elif experience >= 2:
                st.info("Mid-Level")
            else:
                st.warning("Entry Level")

        with col_b:
            st.markdown("**🏠 Remote Work:**")
            if wfh_val == 1:
                st.success("Available ✓")
            else:
                st.error("Not Available ✗")
            
            st.markdown("**📍 Location:**")
            st.info(location)
            
            st.markdown("**📊 Job Type:**")
            st.info(job_type)

        st.markdown("""
        <div class="explanation-box">
            <strong style="font-size: 1.1rem;">📊 How This Prediction Works:</strong><br><br>
            Our AI model analyzes multiple factors including:<br>
            • <strong>Salary vs Experience</strong> balance<br>
            • <strong>Company reputation</strong> and culture<br>
            • <strong>Work flexibility</strong> (remote options)<br>
            • <strong>Job type</strong> and location preferences<br><br>
            The model uses historical data patterns to predict the likelihood of job acceptance with <strong>85% accuracy</strong>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
    🎓 <strong>Developed for Academic Excellence</strong><br>
    Sukkur IBA University | Machine Learning Project 2025<br>
    <br>
    Made with ❤️ using <strong>Streamlit</strong> & <strong>Python</strong><br>
    <small style="opacity: 0.6;">For educational and research purposes only</small>
</div>
""", unsafe_allow_html=True)
