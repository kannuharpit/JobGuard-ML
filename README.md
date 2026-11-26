# Fake Job Posting Detection System

**Portfolio maintainer: Arpit Gupta | M.Tech Computer Science and Engineering**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-2563EB)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)

## Overview

A machine learning and natural language processing (NLP) application that classifies job postings as potentially fraudulent or legitimate based on text and structured job information. The Streamlit interface displays a model-estimated fraud probability and classification.

This portfolio project extends my practical experience with an existing open-source ML pipeline, including local model retraining and compatibility troubleshooting.

## Features

- Job posting input form with title, description, requirements and additional details.
- Text preprocessing and TF-IDF feature extraction.
- Binary classification using classical machine learning algorithms.
- Fraud probability and confidence display.
- Model comparison and evaluation reports.

## Dataset

The included demonstration dataset is synthetic and follows the structure of the Kaggle Real or Fake? Fake Job Posting Prediction dataset. The reference dataset contains approximately 17,880 postings, with approximately 866 labeled fraudulent; these figures describe the reference dataset, not a new evaluation of my local model.

[Reference dataset (Kaggle)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

The target column is `fraudulent`: `0` = legitimate and `1` = fraudulent.

The default synthetic dataset is suitable for demonstrating the pipeline, not establishing real-world fraud detection accuracy. The original documentation identifies the reference dataset license as CC0; review `data/DATA_SOURCE.md` and the dataset terms before redistribution.

## Machine Learning Workflow

1. Clean job posting text, including lowercasing and removal of unnecessary text elements.
2. Convert text into numerical features using TF-IDF and combine them with structured features.
3. Split data into stratified training and testing sets.
4. Train and compare models, including Logistic Regression, Naive Bayes, Random Forest, Linear SVM and XGBoost. The pipeline includes class weighting and SMOTE variants for class imbalance.
5. Evaluate precision, recall, F1-score, accuracy, ROC-AUC and confusion matrices.
6. Save the trained model, vectorizer and preprocessing objects for inference.
7. Display the classification in a Streamlit interface.

The original implementation selects tuned Logistic Regression with TF-IDF. The locally retrained model should be checked against the generated comparison report before claiming specific results.

## Technology Stack

Python, Pandas, NumPy, Scikit-learn, NLP, TF-IDF, imbalanced-learn, XGBoost, Joblib and Streamlit.

## Project Structure

```text
app/
  app.py                    # Streamlit interface
src/
  preprocess.py             # Text preprocessing
  features.py               # Feature extraction
  train.py                  # Model training
  evaluate.py               # Evaluation metrics
  predict.py                # Inference
  eda.py                    # Exploratory data analysis
  generate_synthetic_data.py # Demo dataset generation
  download_data.py          # Reference dataset downloader
data/                       # Dataset and source information
models/                     # Serialized model and preprocessing artifacts
reports/                    # Evaluation reports and charts
requirements.txt            # Dependencies
README.md                   # Project documentation
```

## Installation and Execution

Open the project root containing `app/`, `src/` and `requirements.txt` in VS Code. The following commands are for Windows PowerShell.

### One-time environment setup

If the shared `mlenv` environment already exists, skip its creation.

```powershell
python -m venv "$env:USERPROFILE\mlenv"
& "$env:USERPROFILE\mlenv\Scripts\python.exe" -m pip install -r requirements.txt
```

### Train the model locally

Use the included demo dataset, or generate it using `src/generate_synthetic_data.py` if needed. Retrain locally to avoid incompatibilities between saved models and the installed Scikit-learn version.

```powershell
& "$env:USERPROFILE\mlenv\Scripts\python.exe" src/train.py
```

The local environment used Python 3.10 and Scikit-learn 1.7.2. The downloaded model was serialized with a different Scikit-learn version. Only load serialized model files from trusted sources.

### Launch the application

```powershell
& "$env:USERPROFILE\mlenv\Scripts\python.exe" -m streamlit run app/app.py
```

Open <http://localhost:8501>. Press `Ctrl+C` in the terminal to stop the application. Retraining is not required for each launch if compatible model artifacts already exist.

## My Contributions — Arpit Gupta

- Configured the local Python environment and installed NLP and ML dependencies.
- Diagnosed a Scikit-learn version mismatch that caused the original serialized Logistic Regression model to fail during prediction.
- Retrained the model locally and successfully ran the Streamlit prediction interface.
- Tested sample job descriptions and identified a possible false positive on a legitimate-looking software developer posting.
- Updated documentation to explain the setup, compatibility issue and current limitations.

## Limitations and Planned Improvements

- The demonstration model uses synthetic training data and is not validated for real-world job screening.
- False positives and false negatives are possible; the displayed probability is a model estimate, not proof that a job is fraudulent.
- Train and evaluate on the real reference dataset, with a held-out test split.
- Examine precision, recall and confusion matrices, and tune the classification threshold.
- Improve testing across realistic job descriptions and avoid treating remote work or missing fields as proof of fraud.

