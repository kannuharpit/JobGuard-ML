"""
app.py – Streamlit prediction UI for Fake Job Posting Detection
Run: streamlit run app/app.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import streamlit as st
from predict import predict_job_posting

st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
.result-box {
    padding: 1.5rem; border-radius: 10px;
    font-size: 1.4rem; font-weight: bold; text-align: center;
    margin-top: 1rem;
}
.fraud { background-color: #ffe0e0; color: #c0392b; border: 2px solid #c0392b; }
.legit { background-color: #e0ffe0; color: #27ae60; border: 2px solid #27ae60; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Fake Job Posting Detector")
st.markdown(
    "Paste the details of a job posting below and click **Predict** to find out "
    "whether it is likely to be **fraudulent** or **legitimate**."
)
st.divider()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        title          = st.text_input("Job Title", placeholder="e.g. Senior Software Engineer")
        company_profile = st.text_area("Company Profile", height=100,
                                       placeholder="Brief description of the company...")
        description    = st.text_area("Job Description", height=150,
                                      placeholder="Responsibilities and role details...")
        requirements   = st.text_area("Requirements", height=100,
                                      placeholder="Required skills and qualifications...")
        benefits       = st.text_area("Benefits", height=80,
                                      placeholder="Salary, perks, insurance...")

    with col2:
        employment_type   = st.selectbox("Employment Type",
            ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"])
        required_experience = st.selectbox("Required Experience",
            ["", "Not Applicable", "Internship", "Entry level",
             "Mid-Senior level", "Director", "Executive"])
        required_education = st.selectbox("Required Education",
            ["", "Unspecified", "High School or equivalent", "Associate Degree",
             "Bachelor's Degree", "Master's Degree", "Doctorate"])
        industry = st.text_input("Industry", placeholder="e.g. Information Technology")
        function = st.text_input("Function", placeholder="e.g. Engineering")
        salary_range    = st.text_input("Salary Range", placeholder="e.g. 80000-100000")
        department      = st.text_input("Department", placeholder="e.g. Engineering")
        telecommuting   = st.checkbox("Remote / Telecommuting Position")
        has_company_logo = st.checkbox("Company Logo Present")
        has_questions   = st.checkbox("Screening Questions Included")

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

if submitted:
    if not title.strip() and not description.strip():
        st.warning("Please enter at least a job title or description.")
    else:
        job_dict = {
            "title": title,
            "company_profile": company_profile,
            "description": description,
            "requirements": requirements,
            "benefits": benefits,
            "employment_type": employment_type,
            "required_experience": required_experience,
            "required_education": required_education,
            "industry": industry,
            "function": function,
            "salary_range": salary_range,
            "department": department,
            "telecommuting": int(telecommuting),
            "has_company_logo": int(has_company_logo),
            "has_questions": int(has_questions),
        }
        with st.spinner("Analysing posting ..."):
            result = predict_job_posting(job_dict)

        label = result["label"]
        prob  = result["probability"]
        conf  = result["confidence"]

        box_class = "fraud" if label == "FRAUDULENT" else "legit"
        icon      = "🚨" if label == "FRAUDULENT" else "✅"

        st.markdown(
            f'<div class="result-box {box_class}">{icon} {label}<br>'
            f'<small>Fraud Probability: {prob:.1%} &nbsp;|&nbsp; Confidence: {conf}</small></div>',
            unsafe_allow_html=True,
        )

        st.progress(float(prob), text=f"Fraud probability: {prob:.1%}")

        with st.expander("ℹ️ About this prediction"):
            st.markdown(
                "This model was trained on ~18,000 job postings from Kaggle. "
                "It uses a Logistic Regression classifier with TF-IDF text features "
                "and structured metadata. A high fraud probability means the posting "
                "shares characteristics common in known fraudulent job ads."
            )

st.divider()
st.caption("Fake Job Posting Detector · Powered by scikit-learn · Dataset: Kaggle CC0")
