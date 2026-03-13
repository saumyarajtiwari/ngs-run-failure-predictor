"""
api/app.py
FastAPI backend — serves NGS run failure predictions.
Run with: uvicorn api.app:app --reload
Author: saumyarajtiwari
Project: NGS Run Failure Predictor
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── PATHS ────────────────────────────────────────────────────────
MODELS_PATH = Path('models')

# ── LOAD MODELS AT STARTUP ───────────────────────────────────────
print("🔄 Loading models...")
rf        = joblib.load(MODELS_PATH / 'random_forest.pkl')
xgb       = joblib.load(MODELS_PATH / 'xgboost.pkl')
le_method = joblib.load(MODELS_PATH / 'encoder_lib_method.pkl')
le_calib  = joblib.load(MODELS_PATH / 'encoder_calib_status.pkl')

with open(MODELS_PATH / 'model_metadata.json') as f:
    metadata = json.load(f)

print("✅ Models loaded and ready")

# ── FASTAPI APP ──────────────────────────────────────────────────
app = FastAPI(
    title="NGS Run Failure Predictor",
    description="Predicts MGI sequencing run failure from pre-QC parameters",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── INPUT SCHEMA ─────────────────────────────────────────────────
class RunParameters(BaseModel):
    lib_conc_ng_ul        : float = Field(..., example=2.4)
    frag_size_bp          : int   = Field(..., example=350)
    load_conc_pm          : float = Field(..., example=1.6)
    cluster_density_k_mm2 : int   = Field(..., example=850)
    din_score             : float = Field(..., example=7.0)
    undetermined_pct      : float = Field(..., example=5.0)
    kit_age_days          : int   = Field(..., example=45)
    room_temp_c           : float = Field(..., example=22.0)
    lib_method            : str   = Field(..., example="pcr_free")
    calib_status          : str   = Field(..., example="within_month")
    operator_exp_years    : int   = Field(..., example=3)

# ── HELPER FUNCTIONS ─────────────────────────────────────────────
def encode_inputs(params: RunParameters):
    """Convert input parameters into model-ready DataFrame."""

    try:
        method_encoded = int(le_method.transform([params.lib_method])[0])
    except ValueError:
        method_encoded = 0

    try:
        calib_encoded = int(le_calib.transform([params.calib_status])[0])
    except ValueError:
        calib_encoded = 0

    data = {
        'lib_conc_ng_ul'        : [float(params.lib_conc_ng_ul)],
        'frag_size_bp'          : [int(params.frag_size_bp)],
        'load_conc_pm'          : [float(params.load_conc_pm)],
        'cluster_density_k_mm2' : [int(params.cluster_density_k_mm2)],
        'din_score'             : [float(params.din_score)],
        'undetermined_pct'      : [float(params.undetermined_pct)],
        'kit_age_days'          : [int(params.kit_age_days)],
        'room_temp_c'           : [float(params.room_temp_c)],
        'lib_method'            : [method_encoded],
        'calib_status'          : [calib_encoded],
        'operator_exp_years'    : [int(params.operator_exp_years)],
    }

    return pd.DataFrame(data, columns=metadata['feature_names'])


def get_risk_level(score: float):
    if score < 30:
        return "LOW",      "✓ Proceed",              "All parameters within acceptable range."
    elif score < 65:
        return "MODERATE", "⚡ Caution",               "One or more parameters borderline. Review before proceeding."
    elif score < 85:
        return "HIGH",     "⚠ Consider reprepping",   "Multiple risk factors detected. Recheck library quality."
    else:
        return "CRITICAL", "✕ Abort run",             "Critical failure indicators. Do not proceed. Re-prep required."


def get_top_factors(params: RunParameters):
    factors = []

    if params.lib_conc_ng_ul < 0.8:
        factors.append({"factor": "Library concentration critically low", "severity": "high"})
    elif params.lib_conc_ng_ul < 1.2:
        factors.append({"factor": "Library concentration borderline low", "severity": "medium"})

    if params.frag_size_bp < 150:
        factors.append({"factor": "Fragment size critically small — degradation", "severity": "high"})
    elif params.frag_size_bp < 220:
        factors.append({"factor": "Fragment size below optimal", "severity": "medium"})

    if params.din_score < 3:
        factors.append({"factor": "DIN score critically low — severe degradation", "severity": "high"})
    elif params.din_score < 5:
        factors.append({"factor": "DIN score low — moderate degradation", "severity": "medium"})

    if params.cluster_density_k_mm2 > 1400:
        factors.append({"factor": "Cluster density too high — overclustering risk", "severity": "high"})
    elif params.cluster_density_k_mm2 < 400:
        factors.append({"factor": "Cluster density too low — underloading", "severity": "medium"})

    if params.undetermined_pct > 20:
        factors.append({"factor": "Undetermined indexes very high", "severity": "high"})
    elif params.undetermined_pct > 10:
        factors.append({"factor": "Undetermined indexes elevated", "severity": "medium"})

    if params.kit_age_days > 270:
        factors.append({"factor": "Reagent kit nearing expiry", "severity": "medium"})

    if params.calib_status == 'overdue':
        factors.append({"factor": "Instrument calibration overdue", "severity": "medium"})

    if params.room_temp_c > 28 or params.room_temp_c < 18:
        factors.append({"factor": "Room temperature outside optimal range", "severity": "medium"})

    if not factors:
        factors.append({"factor": "No significant risk factors detected", "severity": "low"})

    return factors[:5]


# ── ROUTES ───────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status"  : "online",
        "service" : "NGS Run Failure Predictor",
        "version" : "1.0.0",
        "models"  : {
            "random_forest_accuracy" : f"{metadata['rf_accuracy']*100:.1f}%",
            "xgboost_accuracy"       : f"{metadata['xgb_accuracy']*100:.1f}%",
            "training_runs"          : metadata['n_training_runs'],
            "data_source"            : metadata['data_source']
        }
    }


@app.post("/predict")
def predict(params: RunParameters):
    X = encode_inputs(params)

    rf_prob  = float(rf.predict_proba(X)[0][1])
    xgb_prob = float(xgb.predict_proba(X)[0][1])

    ensemble_prob = (rf_prob + xgb_prob) / 2
    risk_score    = round(ensemble_prob * 100, 1)

    risk_level, recommendation, explanation = get_risk_level(risk_score)
    top_factors = get_top_factors(params)

    return {
        "risk_score"      : risk_score,
        "risk_level"      : risk_level,
        "recommendation"  : recommendation,
        "explanation"     : explanation,
        "top_factors"     : top_factors,
        "model_breakdown" : {
            "random_forest_prob" : round(rf_prob * 100, 1),
            "xgboost_prob"       : round(xgb_prob * 100, 1),
            "ensemble_prob"      : round(ensemble_prob * 100, 1)
        }
    }


@app.get("/model-info")
def model_info():
    return metadata
