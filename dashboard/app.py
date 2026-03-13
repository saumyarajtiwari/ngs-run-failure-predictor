"""
dashboard/app.py
Streamlit dashboard — cloud deployment version.
API logic is embedded directly (no separate FastAPI server needed).
Author: saumyarajtiwari
Project: NGS Run Failure Predictor
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="NGS Run Failure Predictor",
    page_icon=None,
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #f5f7fa; }
    [data-testid="stSidebar"] { background-color: #0a1628; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color: #c8d8ee !important; font-size: 12px !important; }
    [data-testid="stSidebar"] h2 { color: #ffffff !important; font-size: 13px !important; font-weight: 600 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; border-bottom: 1px solid #1e3a5f; padding-bottom: 10px; margin-bottom: 16px; }
    .stButton > button { background-color: #1a56db; color: white; border: none; border-radius: 3px; font-size: 12px; font-weight: 500; letter-spacing: 0.8px; padding: 10px 0; width: 100%; }
    .metric-block { background: white; border: 1px solid #dde3ed; border-top: 3px solid #1a56db; padding: 20px 24px; border-radius: 2px; }
    .metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 36px; font-weight: 500; color: #0a1628; line-height: 1; margin-bottom: 4px; }
    .metric-label { font-size: 11px; color: #6b7a99; letter-spacing: 1px; text-transform: uppercase; font-weight: 500; }
    .result-card { background: white; border: 1px solid #dde3ed; border-radius: 2px; padding: 28px 32px; margin-bottom: 16px; }
    .section-title { font-size: 11px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #6b7a99; margin-bottom: 16px; border-bottom: 1px solid #eef0f5; padding-bottom: 8px; }
    .factor-row { display: flex; align-items: center; padding: 10px 0; border-bottom: 1px solid #f0f2f7; font-size: 13px; color: #1e2d4a; }
    .severity-dot { width: 8px; height: 8px; border-radius: 50%; margin-right: 12px; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS ──────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models_path = Path('models')
    rf        = joblib.load(models_path / 'random_forest.pkl')
    xgb       = joblib.load(models_path / 'xgboost.pkl')
    le_method = joblib.load(models_path / 'encoder_lib_method.pkl')
    le_calib  = joblib.load(models_path / 'encoder_calib_status.pkl')
    with open(models_path / 'model_metadata.json') as f:
        metadata = json.load(f)
    return rf, xgb, le_method, le_calib, metadata

rf, xgb, le_method, le_calib, metadata = load_models()

# ── PREDICTION LOGIC ─────────────────────────────────────────────
def predict(lib_conc, frag_size, load_conc, cluster_density, din_score,
            undetermined_pct, kit_age, room_temp, lib_method, calib_status, operator_exp):

    try:
        method_encoded = int(le_method.transform([lib_method])[0])
    except:
        method_encoded = 0
    try:
        calib_encoded = int(le_calib.transform([calib_status])[0])
    except:
        calib_encoded = 0

    X = pd.DataFrame([[
        float(lib_conc), int(frag_size), float(load_conc),
        int(cluster_density), float(din_score), float(undetermined_pct),
        int(kit_age), float(room_temp), method_encoded,
        calib_encoded, int(operator_exp)
    ]], columns=metadata['feature_names'])

    rf_prob  = float(rf.predict_proba(X)[0][1])
    xgb_prob = float(xgb.predict_proba(X)[0][1])
    ensemble = (rf_prob + xgb_prob) / 2
    score    = round(ensemble * 100, 1)

    if score < 30:
        level = "LOW";      rec = "Proceed with confidence"
        exp   = "All parameters within acceptable range."
    elif score < 65:
        level = "MODERATE"; rec = "Review flagged parameters"
        exp   = "One or more parameters borderline. Review before proceeding."
    elif score < 85:
        level = "HIGH";     rec = "Consider re-prepping library"
        exp   = "Multiple risk factors detected. Recheck library quality."
    else:
        level = "CRITICAL"; rec = "Abort run"
        exp   = "Critical failure indicators. Do not proceed. Re-prep required."

    factors = []
    if lib_conc < 0.8:       factors.append({"factor": "Library concentration critically low", "severity": "high"})
    elif lib_conc < 1.2:     factors.append({"factor": "Library concentration borderline low", "severity": "medium"})
    if frag_size < 150:      factors.append({"factor": "Fragment size critically small", "severity": "high"})
    elif frag_size < 220:    factors.append({"factor": "Fragment size below optimal", "severity": "medium"})
    if din_score < 3:        factors.append({"factor": "DIN score critically low", "severity": "high"})
    elif din_score < 5:      factors.append({"factor": "DIN score low — moderate degradation", "severity": "medium"})
    if cluster_density > 1400: factors.append({"factor": "Cluster density too high", "severity": "high"})
    elif cluster_density < 400: factors.append({"factor": "Cluster density too low", "severity": "medium"})
    if undetermined_pct > 20: factors.append({"factor": "Undetermined indexes very high", "severity": "high"})
    elif undetermined_pct > 10: factors.append({"factor": "Undetermined indexes elevated", "severity": "medium"})
    if kit_age > 270:        factors.append({"factor": "Reagent kit nearing expiry", "severity": "medium"})
    if calib_status == 'overdue': factors.append({"factor": "Instrument calibration overdue", "severity": "medium"})
    if room_temp > 28 or room_temp < 18: factors.append({"factor": "Room temperature outside optimal range", "severity": "medium"})
    if not factors:          factors.append({"factor": "No significant risk factors detected", "severity": "low"})

    return {
        "score": score, "level": level, "rec": rec, "exp": exp,
        "factors": factors[:5],
        "rf_prob": round(rf_prob * 100, 1),
        "xgb_prob": round(xgb_prob * 100, 1),
        "ensemble": round(ensemble * 100, 1)
    }

# ── HEADER ───────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #dde3ed;">
    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:22px; font-weight:300; color:#0a1628;">NGS Run Failure Predictor</div>
    <div style="font-size:12px; color:#6b7a99; letter-spacing:0.5px;">MGI Platform &nbsp;|&nbsp; Pre-run QC Analysis &nbsp;|&nbsp; Ensemble ML Model</div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────
st.sidebar.markdown("## Input Parameters")

lib_conc         = st.sidebar.number_input("Library Concentration (ng/uL)", min_value=0.1, max_value=20.0, value=2.4, step=0.1)
frag_size        = st.sidebar.number_input("Fragment Size (bp)", min_value=50, max_value=1500, value=350, step=10)
load_conc        = st.sidebar.number_input("Loading Concentration (pM)", min_value=0.5, max_value=5.0, value=1.6, step=0.1)
cluster_density  = st.sidebar.number_input("Cluster Density (K/mm2)", min_value=100, max_value=2000, value=850, step=50)
din_score        = st.sidebar.slider("DIN Score", min_value=1.0, max_value=10.0, value=7.0, step=0.1)
undetermined_pct = st.sidebar.slider("Undetermined Indexes (%)", min_value=0.0, max_value=35.0, value=5.0, step=0.5)
kit_age          = st.sidebar.number_input("Reagent Kit Age (days)", min_value=1, max_value=365, value=45)
room_temp        = st.sidebar.number_input("Room Temperature (C)", min_value=15.0, max_value=35.0, value=22.0, step=0.5)
lib_method       = st.sidebar.selectbox("Library Prep Method", ["pcr_free", "pcr_based", "amplicon", "wes", "rna_seq"])
calib_status     = st.sidebar.selectbox("Calibration Status", ["today", "within_week", "within_month", "overdue"])
operator_exp     = st.sidebar.selectbox("Operator Experience (years)", [1, 2, 3, 5, 8, 10], index=2)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Run Prediction")

# ── MAIN PANEL ───────────────────────────────────────────────────
if not predict_btn:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-block"><div class="metric-value">1,000</div><div class="metric-label">Training Runs</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-block"><div class="metric-value">82.5%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-block"><div class="metric-value">2</div><div class="metric-label">Models in Ensemble</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-block"><div class="metric-value">11</div><div class="metric-label">QC Parameters</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="result-card">
        <div class="section-title">About this system</div>
        <p style="font-size:14px;color:#3a4a6b;line-height:1.8;margin-bottom:16px;">
        This system analyses pre-run QC parameters from your MGI sequencing setup and predicts
        failure probability before the run begins. Enter values in the left panel and click
        Run Prediction to receive a risk score with contributing factor analysis.
        </p>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead><tr style="border-bottom:1px solid #dde3ed;">
                <th style="text-align:left;padding:8px 0;color:#6b7a99;font-weight:500;font-size:11px;text-transform:uppercase;">Risk Level</th>
                <th style="text-align:left;padding:8px 0;color:#6b7a99;font-weight:500;font-size:11px;text-transform:uppercase;">Score</th>
                <th style="text-align:left;padding:8px 0;color:#6b7a99;font-weight:500;font-size:11px;text-transform:uppercase;">Action</th>
            </tr></thead>
            <tbody>
                <tr style="border-bottom:1px solid #f0f2f7;"><td style="padding:10px 0;color:#1a6b3c;font-weight:500;">Low</td><td style="padding:10px 0;color:#3a4a6b;font-family:'IBM Plex Mono',monospace;">0 - 30</td><td style="padding:10px 0;color:#3a4a6b;">Proceed with confidence</td></tr>
                <tr style="border-bottom:1px solid #f0f2f7;"><td style="padding:10px 0;color:#7a5500;font-weight:500;">Moderate</td><td style="padding:10px 0;color:#3a4a6b;font-family:'IBM Plex Mono',monospace;">30 - 65</td><td style="padding:10px 0;color:#3a4a6b;">Review flagged parameters</td></tr>
                <tr style="border-bottom:1px solid #f0f2f7;"><td style="padding:10px 0;color:#8a3500;font-weight:500;">High</td><td style="padding:10px 0;color:#3a4a6b;font-family:'IBM Plex Mono',monospace;">65 - 85</td><td style="padding:10px 0;color:#3a4a6b;">Consider re-prepping library</td></tr>
                <tr><td style="padding:10px 0;color:#8a1a1a;font-weight:500;">Critical</td><td style="padding:10px 0;color:#3a4a6b;font-family:'IBM Plex Mono',monospace;">85 - 100</td><td style="padding:10px 0;color:#3a4a6b;">Abort run — re-prep required</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

else:
    result = predict(lib_conc, frag_size, load_conc, cluster_density, din_score,
                     undetermined_pct, kit_age, room_temp, lib_method, calib_status, operator_exp)

    score = result['score']
    if score < 30:
        score_color = "#1a6b3c"; bar_color = "#2d9e5f"; bg_color = "#f0faf5"; border_color = "#2d9e5f"
    elif score < 65:
        score_color = "#7a5500"; bar_color = "#c48a00"; bg_color = "#fdf8ed"; border_color = "#c48a00"
    elif score < 85:
        score_color = "#8a3500"; bar_color = "#d4580a"; bg_color = "#fdf3ec"; border_color = "#d4580a"
    else:
        score_color = "#8a1a1a"; bar_color = "#c0392b"; bg_color = "#fdf0f0"; border_color = "#c0392b"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div style="background:{bg_color};border:1px solid {border_color};border-radius:2px;padding:32px;text-align:center;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:64px;font-weight:500;color:{score_color};line-height:1;">{score}</div>
            <div style="font-size:11px;color:{score_color};letter-spacing:2px;text-transform:uppercase;font-weight:600;margin-top:8px;">{result['level']} RISK</div>
            <div style="margin:20px 0 8px;height:6px;background:#e0e5ef;border-radius:1px;overflow:hidden;">
                <div style="width:{score}%;height:100%;background:{bar_color};border-radius:1px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:10px;color:#6b7a99;font-family:'IBM Plex Mono',monospace;">
                <span>0</span><span>50</span><span>100</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:white;border:1px solid #dde3ed;border-left:3px solid {border_color};border-radius:2px;padding:28px 32px;">
            <div style="font-size:11px;color:#6b7a99;letter-spacing:1.5px;text-transform:uppercase;font-weight:600;margin-bottom:12px;">Recommendation</div>
            <div style="font-size:18px;font-weight:500;color:{score_color};margin-bottom:12px;">{result['rec']}</div>
            <div style="font-size:13px;color:#3a4a6b;line-height:1.7;">{result['exp']}</div>
            <div style="margin-top:20px;padding-top:16px;border-top:1px solid #eef0f5;display:flex;gap:32px;">
                <div>
                    <div style="font-size:10px;color:#6b7a99;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Random Forest</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;color:#0a1628;font-weight:500;">{result['rf_prob']}%</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#6b7a99;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">XGBoost</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;color:#0a1628;font-weight:500;">{result['xgb_prob']}%</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#6b7a99;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Ensemble</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;color:{score_color};font-weight:600;">{result['ensemble']}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="result-card"><div class="section-title">Contributing Factors</div>', unsafe_allow_html=True)
        for factor in result['factors']:
            sev = factor['severity']
            dot_color = "#c0392b" if sev == "high" else "#c48a00" if sev == "medium" else "#2d9e5f"
            st.markdown(f"""
            <div class="factor-row">
                <div class="severity-dot" style="background:{dot_color};"></div>
                <div>{factor['factor']}</div>
                <div style="margin-left:auto;font-size:11px;color:#6b7a99;text-transform:uppercase;">{sev}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="result-card"><div class="section-title">Parameter Review</div>', unsafe_allow_html=True)
        params_data = [
            ("Library Conc (ng/uL)", lib_conc,         1.0 <= lib_conc <= 5.0),
            ("Fragment Size (bp)",   frag_size,         250 <= frag_size <= 500),
            ("Loading Conc (pM)",    load_conc,         1.2 <= load_conc <= 2.0),
            ("Cluster Density",      cluster_density,   600 <= cluster_density <= 1200),
            ("DIN Score",            din_score,         din_score >= 7.0),
            ("Undetermined (%)",     undetermined_pct,  undetermined_pct < 10),
            ("Kit Age (days)",       kit_age,           kit_age < 180),
            ("Room Temp (C)",        room_temp,         18 <= room_temp <= 26),
        ]
        for name, val, ok in params_data:
            status_color = "#2d9e5f" if ok else "#c0392b"
            status_text  = "Within range" if ok else "Out of range"
            st.markdown(f"""
            <div style="display:flex;align-items:center;padding:8px 0;border-bottom:1px solid #f0f2f7;font-size:12px;">
                <div style="color:#3a4a6b;flex:2;">{name}</div>
                <div style="font-family:'IBM Plex Mono',monospace;color:#0a1628;font-weight:500;flex:1;text-align:right;">{val}</div>
                <div style="color:{status_color};flex:1;text-align:right;font-size:11px;">{status_text}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#9aa5bc;text-align:center;padding-top:8px;border-top:1px solid #eef0f5;">
        NGS Run Failure Predictor v1.0 &nbsp;|&nbsp; saumyarajtiwari &nbsp;|&nbsp; Random Forest + XGBoost Ensemble
    </div>""", unsafe_allow_html=True)
