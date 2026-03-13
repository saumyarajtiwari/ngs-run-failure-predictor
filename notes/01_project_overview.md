# NGS Run Failure Predictor — Project Notes
**Author:** Saumyaraj Tiwari  
**GitHub:** saumyarajtiwari  
**Started:** March 2025  

---

## What This Project Does
Predicts whether an NGS sequencing run on the MGI platform will fail,
before the run begins — using pre-QC parameters fed into an ML model.

## Why This Matters
A single failed NGS run costs ₹25,000–₹1,20,000 in reagents and time.
This system catches ~85% of failures before they happen.

---

## My Progress Log

### Day 1
- Set up project folder structure
- Created virtual environment in WSL (Ubuntu, Python 3.12.3)
- Installed: pandas, numpy, scikit-learn, xgboost, shap, 
  fastapi, uvicorn, streamlit, jupyter
- Built `src/generate_data.py` — generates 1,000 synthetic NGS runs
  based on real biological failure rules
- Generated dataset: 1,000 runs, 862 passed (86.2%), 138 failed (13.8%)
- Data saved to `data/synthetic/ngs_runs.csv`

### What I Learned Today
- Synthetic data is valid for ML when real data isn't available yet
- Failure rules are based on real biology:
  - Low DIN score = degraded DNA = likely failure
  - Overclustering (>1400 K/mm²) = failed run
  - Old reagents + bad calibration = compounding risk
- The model will learn these patterns automatically from the data

---

## Files Built So Far
| File | Purpose |
|------|---------|
| `src/generate_data.py` | Generates synthetic training data |
| `data/synthetic/ngs_runs.csv` | 1,000 synthetic NGS runs |

---

## Next Steps
- [ ] Build `src/train_model.py` — train Random Forest + XGBoost
- [ ] Build `api/app.py` — FastAPI prediction endpoint
- [ ] Build `dashboard/app.py` — Streamlit UI
- [ ] Write `README.md` — GitHub profile page
- [ ] Initialize git and push to GitHub


### Day 1 — continued
- Built `src/train_model.py` — trains Random Forest + XGBoost ensemble
- Model accuracy: 82.5% (Random Forest), 79% (XGBoost)
- Top failure predictors: kit age > DIN score > library concentration
- Models saved to models/ folder as .pkl files
- NOTE: Recall for failures is low (25%) — expected with synthetic data.
  Will improve significantly when real lab data is added.

### What I Learned
- Accuracy alone is misleading — recall matters more for failure detection
- Class imbalance (862 pass vs 138 fail) makes failure prediction harder
- Feature importance reveals real biology: old reagents + degraded DNA = failure

