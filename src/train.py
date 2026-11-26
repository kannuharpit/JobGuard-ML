"""
train.py
--------
Full training pipeline: load → preprocess → features → train → evaluate → save.
Run from project root: python src/train.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from scipy.stats import uniform, randint

from preprocess import load_and_preprocess, get_structured_feature_cols
from features import build_feature_matrix, save_vectorizer
from evaluate import (
    compute_metrics, plot_confusion_matrix,
    plot_roc_curves, plot_model_comparison, save_comparison_table,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH     = os.path.join(PROJECT_ROOT, "data", "fake_job_postings.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def main():
    # ── 1. Load & Preprocess ──────────────────────────────────────────────────
    df = load_and_preprocess(CSV_PATH)
    struct_cols = get_structured_feature_cols(df)
    print(f"[INFO] Structured feature columns: {struct_cols}")

    y = df["fraudulent"].values

    # ── 2. Train / Test Split ─────────────────────────────────────────────────
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {len(df_train)}, Test size: {len(df_test)}")
    print(f"[INFO] Fraud rate – Train: {y_train.mean():.3%}, Test: {y_test.mean():.3%}")

    # ── 3. Build Feature Matrices (TF-IDF + structured) ───────────────────────
    print("[INFO] Building TF-IDF feature matrix ...")
    X_train_tfidf, tfidf_vec, tfidf_scaler = build_feature_matrix(
        df_train, struct_cols, vectorizer_type="tfidf", fit=True
    )
    X_test_tfidf, _, _ = build_feature_matrix(
        df_test, struct_cols, vectorizer=tfidf_vec, scaler=tfidf_scaler, fit=False
    )

    print("[INFO] Building CountVectorizer feature matrix ...")
    X_train_count, count_vec, count_scaler = build_feature_matrix(
        df_train, struct_cols, vectorizer_type="count", fit=True
    )
    X_test_count, _, _ = build_feature_matrix(
        df_test, struct_cols, vectorizer=count_vec, scaler=count_scaler, fit=False
    )

    # ── 4. Handle Class Imbalance (SMOTE on TF-IDF train set) ─────────────────
    print("[INFO] Applying SMOTE to TF-IDF training matrix ...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_tfidf, y_train)
    print(f"[INFO] After SMOTE – shape: {X_train_sm.shape}, fraud rate: {y_train_sm.mean():.3%}")

    results = []
    roc_data = []
    models_to_eval = {}

    def evaluate_model(name, model, X_tr, y_tr, X_te, use_smote=False):
        lbl = name + (" [SMOTE]" if use_smote else "")
        print(f"\n[TRAIN] {lbl} ...")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        # Probability / decision function
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_te)[:, 1]
        elif hasattr(model, "decision_function"):
            df_ = model.decision_function(X_te)
            y_prob = (df_ - df_.min()) / (df_.max() - df_.min() + 1e-9)
        else:
            y_prob = None
        metrics = compute_metrics(y_test, y_pred, y_prob, model_name=lbl)
        print(f"       {metrics}")
        results.append(metrics)
        plot_confusion_matrix(y_test, y_pred, lbl)
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data.append((lbl, fpr, tpr, metrics["ROC-AUC"]))
        models_to_eval[lbl] = model
        return model

    # ── 5. Baseline Models ────────────────────────────────────────────────────
    lr = evaluate_model(
        "Logistic Regression", 
        LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, solver="saga"),
        X_train_tfidf, y_train, X_test_tfidf
    )
    lr_sm = evaluate_model(
        "Logistic Regression",
        LogisticRegression(max_iter=1000, C=1.0, solver="saga"),
        X_train_sm, y_train_sm, X_test_tfidf, use_smote=True
    )

    # Naive Bayes (CountVectorizer, positive values only)
    nb = MultinomialNB(alpha=0.1)
    evaluate_model("Naive Bayes (Count)", nb, X_train_count, y_train, X_test_count)

    # ── 6. Additional Models ──────────────────────────────────────────────────
    # Random Forest (class_weight)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, class_weight="balanced",
        n_jobs=-1, random_state=42
    )
    evaluate_model("Random Forest", rf, X_train_tfidf, y_train, X_test_tfidf)

    # Linear SVM (calibrated for probabilities)
    svm_base = LinearSVC(max_iter=2000, class_weight="balanced", C=0.5)
    svm = CalibratedClassifierCV(svm_base, cv=3)
    evaluate_model("Linear SVM", svm, X_train_tfidf, y_train, X_test_tfidf)

    # XGBoost (scale_pos_weight to handle imbalance)
    neg, pos = np.bincount(y_train)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=neg / pos, use_label_encoder=False,
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0
    )
    evaluate_model("XGBoost", xgb, X_train_tfidf, y_train, X_test_tfidf)

    # ── 7. Hyperparameter Tuning (Top Performers: LR & XGBoost) ───────────────
    print("\n[TUNE] RandomizedSearchCV – Logistic Regression ...")
    lr_param_dist = {
        "C": uniform(0.01, 10),
        "solver": ["saga", "lbfgs"],
    }
    lr_tuned_base = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_search = RandomizedSearchCV(
        lr_tuned_base, lr_param_dist, n_iter=15, cv=3,
        scoring="f1", n_jobs=-1, random_state=42, verbose=0
    )
    lr_search.fit(X_train_tfidf, y_train)
    best_lr = lr_search.best_estimator_
    print(f"[TUNE] Best LR params: {lr_search.best_params_}")
    evaluate_model("Logistic Regression (Tuned)", best_lr, X_train_tfidf, y_train, X_test_tfidf)

    print("\n[TUNE] RandomizedSearchCV – XGBoost ...")
    xgb_param_dist = {
        "n_estimators": randint(200, 500),
        "max_depth": randint(4, 10),
        "learning_rate": uniform(0.05, 0.2),
        "subsample": uniform(0.6, 0.4),
    }
    xgb_base = XGBClassifier(
        scale_pos_weight=neg / pos, use_label_encoder=False,
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_param_dist, n_iter=15, cv=3,
        scoring="f1", n_jobs=-1, random_state=42, verbose=0
    )
    xgb_search.fit(X_train_tfidf, y_train)
    best_xgb = xgb_search.best_estimator_
    print(f"[TUNE] Best XGB params: {xgb_search.best_params_}")
    evaluate_model("XGBoost (Tuned)", best_xgb, X_train_tfidf, y_train, X_test_tfidf)

    # ── 8. Compile Results ────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    print(results_df.to_string(index=False))
    save_comparison_table(results_df)
    plot_model_comparison(results_df)
    plot_roc_curves(roc_data)

    # ── 9. Feature Importance (Best Model) ───────────────────────────────────
    print("\n[INFO] Extracting feature importances ...")
    feature_names = tfidf_vec.get_feature_names_out().tolist() + [
        f"struct_{c}" for c in struct_cols
    ]

    # LR coefficients
    lr_coef = best_lr.coef_[0]
    top_fraud_idx = np.argsort(lr_coef)[-30:][::-1]
    top_legit_idx = np.argsort(lr_coef)[:30]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fraud_names = [feature_names[i] for i in top_fraud_idx]
    legit_names = [feature_names[i] for i in top_legit_idx]
    axes[0].barh(fraud_names[::-1], lr_coef[top_fraud_idx[::-1]], color="tomato")
    axes[0].set_title("Top 30 Fraud-Indicating Features (LR)")
    axes[1].barh(legit_names[::-1], lr_coef[top_legit_idx[::-1]], color="steelblue")
    axes[1].set_title("Top 30 Legit-Indicating Features (LR)")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "lr_feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved LR feature importance → reports/lr_feature_importance.png")

    # XGBoost feature importance
    xgb_imp = best_xgb.feature_importances_
    top_xgb_idx = np.argsort(xgb_imp)[-30:][::-1]
    xgb_names = [feature_names[i] for i in top_xgb_idx]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(xgb_names[::-1], xgb_imp[top_xgb_idx[::-1]], color="seagreen")
    ax.set_title("Top 30 Features – XGBoost (Tuned)")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "xgb_feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved XGBoost feature importance → reports/xgb_feature_importance.png")

    # ── 10. Save Best Model ───────────────────────────────────────────────────
    # Choose best model by F1 score (fraud recall > accuracy for this problem)
    best_row = results_df.loc[results_df["F1"].idxmax()]
    best_name = best_row["Model"]
    print(f"\n[INFO] Best model by F1: {best_name} (F1={best_row['F1']:.4f})")

    # We'll save the tuned Logistic Regression (fastest, most interpretable, high recall)
    # alongside the TF-IDF vectorizer and scaler
    joblib.dump(best_lr,      os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(tfidf_vec,    os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(tfidf_scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(struct_cols,  os.path.join(MODELS_DIR, "struct_cols.pkl"))
    print(f"[INFO] Saved model + artifacts to {MODELS_DIR}/")

    # Save results for README
    results_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    print("\n[DONE] Training pipeline complete.")
    return results_df


if __name__ == "__main__":
    main()
