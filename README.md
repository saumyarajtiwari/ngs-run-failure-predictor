# NGS Run Failure Predictor

A machine learning system that predicts sequencing run failure on the MGI platform by analysing pre-run QC parameters — before the run begins.

## Problem

A single failed NGS run costs between ₹25,000 and ₹1,20,000 in reagents, consumables, and lost time. Most failures are detectable in advance if the right parameters are analysed systematically. This system does that automatically.

## Solution

An ensemble ML model (Random Forest + XGBoost) trained on 1,000 synthetic NGS runs analyses 11 pre-QC parameters and outputs a failure risk score from 0 to 100, along with the top contributing factors and a clear action recommendation.

## Risk Score Interpretation

| Score  | Level    | Action                          |
|--------|----------|---------------------------------|
| 0-30   | Low      | Proceed with confidence         |
| 30-65  | Moderate | Review flagged parameters       |
| 65-85  | High     | Consider re-prepping library    |
| 85-100 | Critical | Abort run - re-prep required    |

## Input Parameters

- Library concentration (ng/uL)
- Fragment size (bp)
- Loading concentration (pM)
- Cluster density (K/mm2)
- DIN score (DNA integrity)
- Percentage undetermined indexes
- Reagent kit age (days)
- Room temperature at loading (C)
- Library prep method
- Instrument calibration status
- Operator experience (years)

## Project Structure
```
ngs-run-failure-predictor/
├── data/
│   ├── synthetic/          # 1,000 synthetic training runs
│   └── real/               # SRA/ENA data coming soon
├── models/                 # Trained model files
├── notebooks/              # Jupyter exploration
├── src/
│   ├── generate_data.py    # Synthetic data generator
│   └── train_model.py      # Model training pipeline
├── api/
│   └── app.py              # FastAPI prediction endpoint
├── dashboard/
│   └── app.py              # Streamlit interface
├── notes/                  # Project learning notes
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository
```bash
git clone https://github.com/saumyarajtiwari/ngs-run-failure-predictor.git
cd ngs-run-failure-predictor
```

2. Create virtual environment and install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Generate synthetic data and train the model
```bash
python3 src/generate_data.py
python3 src/train_model.py
```

4. Start the API
```bash
uvicorn api.app:app --reload
```

5. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

## Tech Stack

- Python 3.12
- scikit-learn - Random Forest classifier
- XGBoost - gradient boosting classifier
- FastAPI - REST API backend
- Streamlit - interactive dashboard
- pandas, numpy - data processing

## Model Performance

| Model         | Accuracy | ROC-AUC |
|---------------|----------|---------|
| Random Forest | 82.5%    | 0.645   |
| XGBoost       | 79.0%    | 0.633   |
| Ensemble      | 79.0%    | 0.646   |

Trained on 800 runs, tested on 200. Performance will improve when real lab data is incorporated.

## Roadmap

- [ ] Integrate real NGS run data from NCBI SRA and ENA
- [ ] Add SHAP explainability plots to dashboard
- [ ] Email and SMS alerts for high-risk runs
- [ ] Connect to MGI instrument API for automated data ingestion
- [ ] Retrain monthly as real lab data accumulates

## Author

Saumyaraj Tiwari
GitHub: https://github.com/saumyarajtiwari

## Part of

This project is part of a three-system AI laboratory intelligence platform:

1. NGS Run Failure Predictor (this repository)
2. Sample Tracking System (coming soon)
3. Lab Workflow Optimizer (coming soon)

## Docker

Run the entire system with one command:
```bash
docker pull saumyarajtiwari/ngs-run-failure-predictor:v1.0
docker run -p 8000:8000 -p 8501:8501 saumyarajtiwari/ngs-run-failure-predictor:v1.0
```

Docker Hub: https://hub.docker.com/r/saumyarajtiwari/ngs-run-failure-predictor
