"""
03_fusion_score.py
==================
Computes the Fusion Score and assigns final rating categories.

Pipeline:
  1. Rank clusters by quality metric Q_c = 0.5*SR + 0.3*ROI - 0.2*MDD
  2. Individual Composite Score (CPS): weighted percentile sum across 8 metrics
  3. GMM soft signal: normalised [P(best_component) - P(worst_component)]
  4. DBSCAN penalty: binary flag for noise points
  5. Fusion Score: FS = 0.65*CPS + 0.25*GMM_soft - 0.10*DBSCAN_penalty
  6. Rating: quartile-based assignment (pd.qcut) → 4 balanced tiers

Inputs:  results/data_final.csv, results/model_labels.csv, results/scaler_params.json
Outputs: results/traders_rated.csv, results/fusion_results.json
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler

RESULTS_DIR = 'results'
K_DESIGN    = 4
W_CPS       = 0.65
W_GMM       = 0.25
W_DBSCAN    = 0.10
LABEL_MAP   = {1: 'Very bad', 2: 'Bad', 3: 'Good', 4: 'Very good'}

NUMERIC_COLUMNS = [
    'ROI', 'Pnl', 'Sharpe Ratio', 'MDD',
    'Proporción de ganancias', 'Días con ganancias',
    'Pnl del copiador', 'Días de trading'
]

# Direction: True = higher percentile → better
METRIC_DIRECTION = {
    'ROI': True, 'Pnl': True, 'Sharpe Ratio': True,
    'MDD': False,   # lower drawdown = better → complement percentile
    'Proporción de ganancias': True, 'Días con ganancias': True,
    'Pnl del copiador': True, 'Días de trading': True,
}

# Financially justified weights (sum = 1.00)
METRIC_WEIGHTS = {
    'Sharpe Ratio': 0.25,           # primary risk-adjusted metric
    'ROI': 0.20,                    # gross profitability
    'MDD': 0.20,                    # capital preservation
    'Proporción de ganancias': 0.15, # consistency
    'Días con ganancias': 0.10,     # operational regularity
    'Pnl del copiador': 0.05,       # social credibility
    'Días de trading': 0.03,        # activity level
    'Pnl': 0.02,                    # absolute profit (least informative standalone)
}

# Q_c weights for cluster quality ranking (Sharpe, ROI, MDD)
W_QC = (0.5, 0.3, 0.2)


def cluster_quality(df: pd.DataFrame, labels: np.ndarray,
                    w: tuple = W_QC) -> dict:
    """Score each cluster by Q_c = w1*SR + w2*ROI - w3*MDD."""
    w1, w2, w3 = w
    q = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        sub = df[labels == c]
        q[c] = (w1 * sub['Sharpe Ratio'].mean()
              + w2 * sub['ROI'].mean()
              - w3 * sub['MDD'].mean())
    return q


def compute_cps(df: pd.DataFrame) -> pd.Series:
    """
    Individual Composite Score: weighted sum of population percentiles.
    Each metric is rank-transformed to [0,1]; MDD is complemented so that
    higher always means better.
    """
    cps = pd.Series(0.0, index=df.index)
    for col, higher_better in METRIC_DIRECTION.items():
        pct = df[col].rank(pct=True)
        cps += METRIC_WEIGHTS[col] * (pct if higher_better else (1 - pct))
    return cps


def compute_gmm_soft(df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.Series:
    """
    GMM soft signal: normalised [P(best_component) - P(worst_component)].
    best/worst identified by Q_c on GMM clusters.
    Stability: best component is 100% stable across Q_c weight configurations.
    """
    gmm_labels = labels_df['GMM_label'].values
    q          = cluster_quality(df, gmm_labels, W_QC)
    best_c     = max(q, key=q.get)
    worst_c    = min(q, key=q.get)

    prob_best  = labels_df[f'GMM_prob_{best_c}'].values
    prob_worst = labels_df[f'GMM_prob_{worst_c}'].values
    raw        = prob_best - prob_worst
    eps        = 1e-9
    normalised = (raw - raw.min()) / (raw.max() - raw.min() + eps)
    return pd.Series(normalised, index=df.index), int(best_c), int(worst_c)


if __name__ == '__main__':
    # Load data
    df         = pd.read_csv(f'{RESULTS_DIR}/data_final.csv')
    labels_df  = pd.read_csv(f'{RESULTS_DIR}/model_labels.csv')

    print(f"[03] N={len(df)}")

    # 1. CPS
    df['CPS'] = compute_cps(df)
    print(f"[03] CPS: [{df['CPS'].min():.3f}, {df['CPS'].max():.3f}]  "
          f"μ={df['CPS'].mean():.3f}  σ={df['CPS'].std():.3f}")

    # 2. GMM soft signal
    df['GMM_soft_norm'], best_c, worst_c = compute_gmm_soft(df, labels_df)
    print(f"[03] GMM soft: best_component={best_c}, worst_component={worst_c}")

    # 3. DBSCAN penalty
    df['DBSCAN_penalty'] = (labels_df['DB_label'].values == -1).astype(float)
    n_penalised = int(df['DBSCAN_penalty'].sum())
    print(f"[03] DBSCAN penalty applied to {n_penalised} noise traders")

    # 4. Fusion Score
    df['Fusion_score'] = (
        W_CPS    * df['CPS']
      + W_GMM    * df['GMM_soft_norm']
      - W_DBSCAN * df['DBSCAN_penalty']
    )
    print(f"[03] Fusion Score: [{df['Fusion_score'].min():.3f}, "
          f"{df['Fusion_score'].max():.3f}]  unique={df['Fusion_score'].nunique()}")

    # 5. Rating assignment (quartile-based, ~21-22 traders per tier)
    df['Final_Rating'] = pd.qcut(
        df['Fusion_score'], q=4, labels=[1, 2, 3, 4], duplicates='drop'
    ).astype(int)
    df['Final_Label'] = df['Final_Rating'].map(LABEL_MAP)

    dist = df['Final_Rating'].value_counts().sort_index()
    print(f"[03] Rating distribution:\n{dist.to_string()}")

    # 6. Save
    df.to_csv(f'{RESULTS_DIR}/traders_rated.csv', index=False)

    results = {
        'weights': {'W_CPS': W_CPS, 'W_GMM': W_GMM, 'W_DBSCAN': W_DBSCAN},
        'metric_weights': METRIC_WEIGHTS,
        'qc_weights': {'w_SR': W_QC[0], 'w_ROI': W_QC[1], 'w_MDD': W_QC[2]},
        'gmm_best_component': best_c,
        'gmm_worst_component': worst_c,
        'n_penalised_dbscan': n_penalised,
        'fs_range': [round(float(df['Fusion_score'].min()), 3),
                     round(float(df['Fusion_score'].max()), 3)],
        'fs_unique_values': int(df['Fusion_score'].nunique()),
        'rating_distribution': dist.to_dict(),
    }
    with open(f'{RESULTS_DIR}/fusion_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[03] Saved: traders_rated.csv, fusion_results.json")
