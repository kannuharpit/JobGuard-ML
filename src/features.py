"""
features.py
-----------
Feature engineering: TF-IDF / CountVectorizer + structured features.
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import joblib, os


def build_feature_matrix(
    df: pd.DataFrame,
    structured_cols: list,
    vectorizer=None,
    scaler=None,
    vectorizer_type: str = "tfidf",
    fit: bool = True,
    max_features: int = 10000,
):
    """
    Combine TF-IDF (or CountVectorizer) text features with structured features.

    Parameters
    ----------
    df              : DataFrame containing 'processed_text' and structured_cols
    structured_cols : list of structured feature column names
    vectorizer      : fitted vectorizer (None → create new one)
    scaler          : fitted StandardScaler (None → create new one)
    vectorizer_type : 'tfidf' or 'count'
    fit             : whether to fit the vectorizer/scaler (False for test set)
    max_features    : maximum vocabulary size

    Returns
    -------
    X_combined  : sparse matrix (text features + scaled structured features)
    vectorizer  : fitted vectorizer
    scaler      : fitted scaler
    """
    # ── Text features ──────────────────────────────────────────────────────────
    if vectorizer is None:
        if vectorizer_type == "tfidf":
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=max_features,
                sublinear_tf=True,
                min_df=2,
            )
        else:
            vectorizer = CountVectorizer(
                ngram_range=(1, 2),
                max_features=max_features,
                min_df=2,
            )

    texts = df["processed_text"].fillna("").values
    if fit:
        X_text = vectorizer.fit_transform(texts)
    else:
        X_text = vectorizer.transform(texts)

    # ── Structured features ────────────────────────────────────────────────────
    X_struct = df[structured_cols].fillna(0).values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler(with_mean=False)
    if fit:
        X_struct = scaler.fit_transform(X_struct)
    else:
        X_struct = scaler.transform(X_struct)

    X_combined = hstack([X_text, csr_matrix(X_struct)])
    return X_combined, vectorizer, scaler


def save_vectorizer(vectorizer, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"[INFO] Vectorizer saved → {path}")


def load_vectorizer(path: str):
    return joblib.load(path)
