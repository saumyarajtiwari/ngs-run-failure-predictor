"""
train_model.py
Trains Random Forest + XGBoost ensemble on synthetic NGS run data.
Saves trained model to models/ folder.
Author: saumyarajtiwari
Project: NGS Run Failure Predictor
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
from xgboost import XGBClassifier

# ── PATHS ────────────────────────────────────────────────────────
DATA_PATH   = Path('data/synthetic/ngs_runs.csv')
MODELS_PATH = Path('models')
MODELS_PATH.mkdir(exist_ok=True)

def load_and_prepare(path):
    """Load CSV and prepare features for ML."""
    print("📂 Loading data...")
    df = pd.read_csv(path)
    print(f"   └── {len(df)} runs loaded")

    # Drop columns the model shouldn't use
    # run_id is just a label, failure_prob is what we used to generate data
    df = df.drop(columns=['run_id', 'failure_prob'])

    # Encode categorical columns into numbers
    # ML models only understand numbers, not text
    le_method = LabelEncoder()
    le_calib  = LabelEncoder()

    df['lib_method']    = le_method.fit_transform(df['lib_method'])
    df['calib_status']  = le_calib.fit_transform(df['calib_status'])

    # Save encoders — needed later when predicting new runs
    joblib.dump(le_method, MODELS_PATH / 'encoder_lib_method.pkl')
    joblib.dump(le_calib,  MODELS_PATH / 'encoder_calib_status.pkl')

    # Split into features (X) and target (y)
    X = df.drop(columns=['run_failed'])
    y = df['run_failed']

    print(f"   ├── Features : {list(X.columns)}")
    print(f"   ├── Passed   : {(y==0).sum()}")
    print(f"   └── Failed   : {(y==1).sum()}")

    return X, y, list(X.columns)


def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight='balanced',  # handles imbalanced data
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("   └── Done")
    return rf


def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier."""
    print("\n⚡ Training XGBoost...")

    # Calculate scale for imbalanced classes
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale,  # handles imbalanced data
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    print("   └── Done")
    return xgb


def evaluate(model, X_test, y_test, name):
    """Print evaluation metrics for a model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n📊 {name} Results:")
    print(f"   ├── Accuracy : {acc*100:.1f}%")
    print(f"   ├── ROC-AUC  : {auc:.3f}")
    print(f"   └── Report:")
    report = classification_report(y_test, y_pred,
                                target_names=['Passed','Failed'])
    for line in report.split('\n'):
        print('        ' + line)
    return acc, auc


def ensemble_predict(rf, xgb, X):
    """
    Combine RF and XGBoost predictions by averaging probabilities.
    This is more accurate than either model alone.
    """
    rf_proba  = rf.predict_proba(X)[:, 1]
    xgb_proba = xgb.predict_proba(X)[:, 1]
    return (rf_proba + xgb_proba) / 2


if __name__ == '__main__':
    print("=" * 55)
    print("  NGS Run Failure Predictor — Model Training")
    print("=" * 55)

    # 1. Load data
    X, y, feature_names = load_and_prepare(DATA_PATH)

    # 2. Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✂️  Train: {len(X_train)} runs | Test: {len(X_test)} runs")

    # 3. Train both models
    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    # 4. Evaluate individually
    rf_acc,  rf_auc  = evaluate(rf,  X_test, y_test, "Random Forest")
    xgb_acc, xgb_auc = evaluate(xgb, X_test, y_test, "XGBoost")

    # 5. Evaluate ensemble
    print("\n🤝 Ensemble (RF + XGBoost averaged):")
    ens_proba = ensemble_predict(rf, xgb, X_test)
    ens_pred  = (ens_proba >= 0.5).astype(int)
    ens_acc   = accuracy_score(y_test, ens_pred)
    ens_auc   = roc_auc_score(y_test, ens_proba)
    print(f"   ├── Accuracy : {ens_acc*100:.1f}%")
    print(f"   └── ROC-AUC  : {ens_auc:.3f}")

    # 6. Feature importance from Random Forest
    importances = dict(zip(feature_names,
                           rf.feature_importances_.round(4)))
    importances = dict(sorted(importances.items(),
                               key=lambda x: x[1], reverse=True))
    print("\n🔍 Feature Importance (what matters most):")
    for feat, imp in importances.items():
        bar = '█' * int(imp * 50)
        print(f"   {feat:<30} {bar} {imp:.4f}")

    # 7. Save models
    print("\n💾 Saving models...")
    joblib.dump(rf,  MODELS_PATH / 'random_forest.pkl')
    joblib.dump(xgb, MODELS_PATH / 'xgboost.pkl')

    # Save metadata
    metadata = {
        'feature_names'  : feature_names,
        'rf_accuracy'    : round(rf_acc, 4),
        'rf_auc'         : round(rf_auc, 4),
        'xgb_accuracy'   : round(xgb_acc, 4),
        'xgb_auc'        : round(xgb_auc, 4),
        'ensemble_accuracy': round(ens_acc, 4),
        'ensemble_auc'   : round(ens_auc, 4),
        'n_training_runs': len(X_train),
        'n_test_runs'    : len(X_test),
        'data_source'    : 'synthetic'
    }
    with open(MODELS_PATH / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ├── models/random_forest.pkl")
    print(f"   ├── models/xgboost.pkl")
    print(f"   ├── models/encoder_lib_method.pkl")
    print(f"   ├── models/encoder_calib_status.pkl")
    print(f"   └── models/model_metadata.json")
    print("\n✅ Training complete. Models ready for deployment.")
