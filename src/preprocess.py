"""
preprocess.py
-------------
Text cleaning and feature preprocessing utilities.
Uses a built-in stopword list to avoid NLTK download requirements.
"""

import re
import pandas as pd
import numpy as np

# Built-in English stopwords (no NLTK download needed)
STOP_WORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are",
    "aren","as","at","be","because","been","before","being","below","between","both",
    "but","by","can","cannot","could","couldn","did","didn","do","does","doesn",
    "doing","don","down","during","each","few","for","from","further","get","got",
    "had","hadn","has","hasn","have","haven","having","he","her","here","hers",
    "herself","him","himself","his","how","i","if","in","into","is","isn","it","its",
    "itself","just","ll","m","me","might","more","most","mustn","my","myself","need",
    "needn","no","nor","not","now","o","of","off","on","once","only","or","other",
    "our","ours","ourselves","out","over","own","re","s","same","shan","she","should",
    "shouldn","so","some","such","t","than","that","the","their","theirs","them",
    "themselves","then","there","these","they","this","those","through","to","too",
    "under","until","up","us","ve","very","was","wasn","we","were","weren","what",
    "when","where","which","while","who","whom","why","will","with","won","wouldn",
    "y","you","your","yours","yourself","yourselves","will","would","could","should",
    "may","might","must","shall","can","need","dare","ought","used",
}

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]


def simple_lemmatize(word: str) -> str:
    """Very lightweight rule-based lemmatization (avoids NLTK/spaCy)."""
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("tion") or word.endswith("tions"):
        return word.replace("tions", "tion")
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ness"):
        return word[:-4]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/special chars, remove stopwords, light lemmatize."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation / numbers
    text = re.sub(r"[^a-z\s]", " ", text)
    # Tokenise
    tokens = text.split()
    # Filter stopwords, short words, and lemmatize
    tokens = [
        simple_lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]
    return " ".join(tokens)


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Load raw CSV, clean text, engineer structured features, return clean DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows, {df.shape[1]} columns")

    # ── Structured feature engineering ────────────────────────────────────────
    df["telecommuting"]    = df["telecommuting"].fillna(0).astype(int)
    df["has_company_logo"] = df["has_company_logo"].fillna(0).astype(int)
    df["has_questions"]    = df["has_questions"].fillna(0).astype(int)
    df["fraudulent"]       = df["fraudulent"].astype(int)

    # Binary indicators for missing fields
    for col in ["salary_range", "department", "employment_type",
                "required_experience", "required_education", "industry", "function"]:
        df[f"has_{col}"] = df[col].notna().astype(int)

    # Text-length features (before cleaning)
    for col in TEXT_COLS:
        df[f"{col}_len"] = df[col].fillna("").apply(len)

    # ── Text cleaning ──────────────────────────────────────────────────────────
    print("[INFO] Cleaning text fields ...")
    for col in TEXT_COLS:
        df[f"{col}_clean"] = df[col].apply(clean_text)

    # Combine all cleaned text into one column
    df["processed_text"] = df[[f"{c}_clean" for c in TEXT_COLS]].apply(
        lambda row: " ".join(row.values), axis=1
    )

    print("[INFO] Preprocessing complete.")
    return df


def get_structured_feature_cols(df: pd.DataFrame) -> list:
    """Return list of structured (non-text) feature column names."""
    cols = [
        "telecommuting", "has_company_logo", "has_questions",
        "has_salary_range", "has_department", "has_employment_type",
        "has_required_experience", "has_required_education",
        "has_industry", "has_function",
    ] + [f"{c}_len" for c in TEXT_COLS]
    return [c for c in cols if c in df.columns]
