"""
generate_data.py
Generates synthetic NGS run data for model training.
Author: saumyarajtiwari
Project: NGS Run Failure Predictor
"""

import pandas as pd
import numpy as np
from pathlib import Path

# So we get same results every time we run this
np.random.seed(42)

N_RUNS = 1000  # number of synthetic runs to generate

def generate_ngs_runs(n=N_RUNS):
    """
    Generate realistic synthetic NGS run data.
    Each row = one sequencing run with pre-QC parameters and outcome.
    """

    # ── LIBRARY CONCENTRATION (ng/μL) ──────────────────────────
    # Normal range: 1.0 – 5.0 ng/μL
    # Too low (<0.8) or too high (>10) increases failure risk
    lib_conc = np.random.lognormal(mean=0.8, sigma=0.5, size=n)
    lib_conc = np.clip(lib_conc, 0.1, 20.0).round(2)

    # ── FRAGMENT SIZE (bp) ──────────────────────────────────────
    # Normal range: 250 – 500 bp
    # Too small (<150) means degradation → likely failure
    frag_size = np.random.normal(loc=350, scale=80, size=n)
    frag_size = np.clip(frag_size, 80, 900).round(0).astype(int)

    # ── LOADING CONCENTRATION (pM) ──────────────────────────────
    # Normal range: 1.2 – 2.0 pM
    load_conc = np.random.normal(loc=1.6, scale=0.4, size=n)
    load_conc = np.clip(load_conc, 0.5, 4.0).round(2)

    # ── CLUSTER DENSITY (K/mm²) ─────────────────────────────────
    # Normal range: 600 – 1200 K/mm²
    # Too high (>1400) = overclustering → failure
    cluster_density = np.random.normal(loc=850, scale=180, size=n)
    cluster_density = np.clip(cluster_density, 200, 1800).round(0).astype(int)

    # ── DIN SCORE (DNA Integrity Number, 1–10) ──────────────────
    # Higher is better. Below 4 = severe degradation
    din_score = np.random.normal(loc=7.0, scale=1.8, size=n)
    din_score = np.clip(din_score, 1.0, 10.0).round(1)

    # ── % UNDETERMINED INDEXES ──────────────────────────────────
    # Should be <10%. Above 20% = serious problem
    undetermined_pct = np.random.exponential(scale=5.0, size=n)
    undetermined_pct = np.clip(undetermined_pct, 0.1, 35.0).round(2)

    # ── REAGENT KIT AGE (days since manufacture) ────────────────
    # Fresh kits (<90 days) perform best
    kit_age_days = np.random.randint(1, 365, size=n)

    # ── ROOM TEMPERATURE (°C at time of loading) ────────────────
    # Optimal: 18–26°C. Outside this = risk
    room_temp = np.random.normal(loc=22, scale=3, size=n)
    room_temp = np.clip(room_temp, 15, 35).round(1)

    # ── LIBRARY PREP METHOD ─────────────────────────────────────
    # Different methods have different baseline failure rates
    lib_methods = ['pcr_free', 'pcr_based', 'amplicon', 'wes', 'rna_seq']
    lib_method = np.random.choice(lib_methods, size=n,
                                   p=[0.25, 0.30, 0.20, 0.15, 0.10])

    # ── CALIBRATION STATUS ──────────────────────────────────────
    calib_options = ['today', 'within_week', 'within_month', 'overdue']
    calib_status = np.random.choice(calib_options, size=n,
                                     p=[0.15, 0.30, 0.40, 0.15])

    # ── OPERATOR EXPERIENCE (years) ─────────────────────────────
    # More experienced operators → fewer failures
    operator_exp = np.random.choice([1, 2, 3, 5, 8, 10], size=n,
                                     p=[0.10, 0.15, 0.25, 0.25, 0.15, 0.10])

    # ────────────────────────────────────────────────────────────
    # FAILURE LABEL — this is the "answer" the model learns from
    # Based on real biological/technical rules
    # ────────────────────────────────────────────────────────────
    failure_prob = np.zeros(n)

    # Low library concentration → high failure risk
    failure_prob += np.where(lib_conc < 0.8, 0.35, 0.0)
    failure_prob += np.where((lib_conc >= 0.8) & (lib_conc < 1.2), 0.15, 0.0)
    failure_prob += np.where(lib_conc > 10.0, 0.20, 0.0)

    # Small fragments → degraded DNA → failure
    failure_prob += np.where(frag_size < 150, 0.30, 0.0)
    failure_prob += np.where((frag_size >= 150) & (frag_size < 220), 0.15, 0.0)

    # Low DIN → failure
    failure_prob += np.where(din_score < 3.0, 0.30, 0.0)
    failure_prob += np.where((din_score >= 3.0) & (din_score < 5.0), 0.15, 0.0)

    # Overclustering → failure
    failure_prob += np.where(cluster_density > 1400, 0.25, 0.0)
    failure_prob += np.where(cluster_density < 400, 0.20, 0.0)

    # High undetermined indexes → failure
    failure_prob += np.where(undetermined_pct > 20, 0.20, 0.0)
    failure_prob += np.where((undetermined_pct > 10) & (undetermined_pct <= 20), 0.10, 0.0)

    # Old reagents → failure
    failure_prob += np.where(kit_age_days > 270, 0.15, 0.0)

    # Bad calibration → failure
    failure_prob += np.where(calib_status == 'overdue', 0.15, 0.0)

    # Extreme temperature → failure
    failure_prob += np.where((room_temp > 28) | (room_temp < 18), 0.10, 0.0)

    # Less experienced operator → slightly more failures
    failure_prob += np.where(operator_exp <= 2, 0.08, 0.0)

    # Clip probability between 0 and 1
    failure_prob = np.clip(failure_prob, 0.0, 1.0)

    # Convert probability to actual outcome (1 = failed, 0 = passed)
    outcome = (np.random.random(n) < failure_prob).astype(int)

    # ── BUILD DATAFRAME ─────────────────────────────────────────
    df = pd.DataFrame({
        'run_id': [f'MGI-SYN-{i:04d}' for i in range(1, n+1)],
        'lib_conc_ng_ul': lib_conc,
        'frag_size_bp': frag_size,
        'load_conc_pm': load_conc,
        'cluster_density_k_mm2': cluster_density,
        'din_score': din_score,
        'undetermined_pct': undetermined_pct,
        'kit_age_days': kit_age_days,
        'room_temp_c': room_temp,
        'lib_method': lib_method,
        'calib_status': calib_status,
        'operator_exp_years': operator_exp,
        'failure_prob': failure_prob.round(3),  # keeping for reference
        'run_failed': outcome                    # ← this is what model predicts
    })

    return df


if __name__ == '__main__':
    print("🧬 Generating synthetic NGS run data...")

    df = generate_ngs_runs(N_RUNS)

    # Save to CSV
    output_path = Path('data/synthetic/ngs_runs.csv')
    df.to_csv(output_path, index=False)

    # Summary
    total = len(df)
    failed = df['run_failed'].sum()
    passed = total - failed

    print(f"✅ Generated {total} runs")
    print(f"   ├── Passed : {passed} ({passed/total*100:.1f}%)")
    print(f"   └── Failed : {failed} ({failed/total*100:.1f}%)")
    print(f"📁 Saved to: {output_path}")
    print("\nFirst 5 rows:")
    print(df.head())
