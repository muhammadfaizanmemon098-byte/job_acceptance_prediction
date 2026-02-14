#  Job Acceptance Prediction System

A Machine Learning web application that predicts whether a candidate will accept or reject a job offer based on key professional attributes.

---

##  Live Application
🔗 https://jobacceptanceprediction-faizan.streamlit.app/

---

##  Project Overview

In many hiring processes, companies invest significant time and resources in candidates who may decline offers at the final stage.  

This project applies Machine Learning to help predict job acceptance decisions in advance.

The system takes candidate and job-related inputs and returns:
- Acceptance / Rejection Prediction
- Model Accuracy
- Explanation of Prediction

---

##  Machine Learning Details

- **Problem Type:** Binary Classification  
- **Algorithm Used:** Random Forest Classifier  
- **Feature Scaling:** StandardScaler  
- **Model Saving:** Joblib (.pkl files)

---

##  Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  

---

##  Key Features

- Interactive Streamlit Web Interface  
- Dark Mode Toggle  
- Real-time Prediction  
- Model Accuracy Display  
- Prediction Explanation Section  
- University Branding  

---

##  Project Structure

```
job-acceptance-prediction/
│
├── app.py
├── job_acceptance_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

##  How to Run Locally

```
pip install -r requirements.txt
streamlit run app.py
```

---

##  Deployment

The application is deployed using Streamlit Cloud.

---

##  Author

Muhammad Faizan  
Machine Learning & Data Science Enthusiast
