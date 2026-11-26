"""
predict.py
----------
Prediction system: load saved model + vectorizer and make inferences
on new job postings.

Usage:
    python src/predict.py

Or import the function:
    from src.predict import predict_job_posting
"""

import os, sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

_model     = None
_vectorizer = None
_scaler    = None
_struct_cols = None


def _load_artifacts():
    global _model, _vectorizer, _scaler, _struct_cols
    if _model is None:
        _model       = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        _vectorizer  = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        _scaler      = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        _struct_cols = joblib.load(os.path.join(MODELS_DIR, "struct_cols.pkl"))


def predict_job_posting(job_dict: dict) -> dict:
    """
    Predict whether a job posting is fraudulent.

    Parameters
    ----------
    job_dict : dict with keys matching the original dataset columns, e.g.:
        {
          "title": "Software Engineer",
          "company_profile": "...",
          "description": "...",
          "requirements": "...",
          "benefits": "...",
          "telecommuting": 0,
          "has_company_logo": 1,
          "has_questions": 0,
          "salary_range": "",
          "department": "",
          "employment_type": "Full-time",
          "required_experience": "Mid-Senior level",
          "required_education": "Bachelor's Degree",
          "industry": "Information Technology",
          "function": "Engineering",
        }

    Returns
    -------
    dict with keys:
        label       : "FRAUDULENT" or "LEGITIMATE"
        probability : confidence score (0–1) for being fraudulent
        confidence  : human-readable confidence level
    """
    _load_artifacts()

    from scipy.sparse import hstack, csr_matrix
    import pandas as pd

    # ── Text features ──────────────────────────────────────────────────────────
    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    combined_text = " ".join(
        clean_text(str(job_dict.get(c, ""))) for c in text_cols
    )
    X_text = _vectorizer.transform([combined_text])

    # ── Structured features ────────────────────────────────────────────────────
    row = {}
    row["telecommuting"]       = int(job_dict.get("telecommuting", 0))
    row["has_company_logo"]    = int(job_dict.get("has_company_logo", 0))
    row["has_questions"]       = int(job_dict.get("has_questions", 0))
    row["has_salary_range"]    = 1 if job_dict.get("salary_range") else 0
    row["has_department"]      = 1 if job_dict.get("department") else 0
    row["has_employment_type"] = 1 if job_dict.get("employment_type") else 0
    row["has_required_experience"] = 1 if job_dict.get("required_experience") else 0
    row["has_required_education"]  = 1 if job_dict.get("required_education") else 0
    row["has_industry"]        = 1 if job_dict.get("industry") else 0
    row["has_function"]        = 1 if job_dict.get("function") else 0
    for col in text_cols:
        row[f"{col}_len"] = len(str(job_dict.get(col, "")))

    # Filter to only cols the model was trained on
    struct_values = np.array([[row.get(c, 0) for c in _struct_cols]], dtype=np.float32)
    X_struct = _scaler.transform(struct_values)
    X = hstack([X_text, csr_matrix(X_struct)])

    # ── Predict ────────────────────────────────────────────────────────────────
    pred_label = _model.predict(X)[0]
    if hasattr(_model, "predict_proba"):
        prob_fraud = _model.predict_proba(X)[0][1]
    else:
        prob_fraud = float(pred_label)

    label = "FRAUDULENT" if pred_label == 1 else "LEGITIMATE"
    if prob_fraud >= 0.8:
        confidence = "Very High"
    elif prob_fraud >= 0.6:
        confidence = "High"
    elif prob_fraud >= 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "label": label,
        "probability": round(float(prob_fraud), 4),
        "confidence": confidence,
    }


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example 1: Suspicious posting
    suspicious = {
        "title": "Work from home – earn $5000 weekly guaranteed!",
        "company_profile": "",
        "description": "No experience needed. Unlimited earning potential. Join our global network today and start making money immediately.",
        "requirements": "Must be 18+. No skills required.",
        "benefits": "Unlimited income, work from anywhere, no boss",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
        "employment_type": "",
        "required_experience": "",
        "industry": "",
    }

    # Example 2: Legitimate posting
    legit = {
        "title": "Senior Data Scientist",
        "company_profile": "We are a Fortune 500 company specializing in financial analytics with 10,000+ employees.",
        "description": "We are looking for an experienced data scientist to join our AI team. You will build predictive models, work with large datasets, and collaborate cross-functionally.",
        "requirements": "5+ years ML experience, Python, SQL, strong statistics background.",
        "benefits": "Health insurance, 401k matching, stock options, 20 days PTO",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1,
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Master's Degree",
        "industry": "Financial Services",
        "function": "Data Science",
    }

    for name, posting in [("Suspicious", suspicious), ("Legitimate", legit)]:
        result = predict_job_posting(posting)
        print(f"\n[{name}] → {result['label']} (fraud_prob={result['probability']:.3f}, confidence={result['confidence']})")
