"""
01_preprocessing.py
===================
Data loading, cleaning, scaling, and outlier removal.

Pipeline:
  Raw CSV (N=100)
  -> numeric parsing + listwise deletion (N=88)
  -> RobustScaler (justified by Shapiro-Wilk rejection of normality)
  -> Mahalanobis outlier removal at p97.5 (N=85)

Outputs:
  data_clean   : pd.DataFrame, N=88, parsed numeric columns
  data_final   : pd.DataFrame, N=85, after outlier removal
  scaled_final : np.ndarray, N=85 x 8, RobustScaler applied
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import RobustScaler
import json
import os

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = 'data/traders_data.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

NUMERIC_COLUMNS = [
    'ROI', 'Pnl', 'Sharpe Ratio', 'MDD',
    'Proporción de ganancias', 'Días con ganancias',
    'Pnl del copiador', 'Días de trading'
]

# ── Load ─────────────────────────────────────────────────────────────────────
def load_and_parse(path: str) -> pd.DataFrame:
    """Load the semicolon-separated CSV with European number formatting."""
    df = pd.read_csv(path, encoding='ISO-8859-1', delimiter=';', header=0)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(
            df[col].astype(str)
            .str.replace('.', '', regex=False)   # thousands separator
            .str.replace(',', '.', regex=False)   # decimal separator
            .str.replace('%', '', regex=False)    # percent symbol
            .str.strip(),
            errors='coerce'
        )
    return df

# ── Normality ─────────────────────────────────────────────────────────────────
def shapiro_wilk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Shapiro-Wilk test for each numeric column. Returns summary DataFrame."""
    rows = []
    for col in NUMERIC_COLUMNS:
        w, p = stats.shapiro(df[col].dropna())
        rows.append({'Variable': col, 'W': round(w, 3),
                     'p_value': round(p, 4), 'Normal': p > 0.05})
    return pd.DataFrame(rows)

# ── Outlier removal ───────────────────────────────────────────────────────────
def mahalanobis_filter(df: pd.DataFrame, X: np.ndarray,
                       threshold_pct: float = 0.975):
    """
    Remove multivariate outliers using Mahalanobis distance.

    Parameters
    ----------
    df : DataFrame aligned with X
    X  : scaled feature matrix (N x p)
    threshold_pct : Chi-squared quantile threshold (default 97.5%)

    Returns
    -------
    df_filtered, X_filtered, distances
    """
    mean_vec = X.mean(axis=0)
    cov_inv  = np.linalg.inv(np.cov(X.T))
    distances = np.array([mahalanobis(X[i], mean_vec, cov_inv)
                          for i in range(len(X))])
    threshold = np.percentile(distances, threshold_pct * 100)
    mask = distances <= threshold
    return df[mask].copy(), X[mask], distances

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1. Load and parse
    raw = load_and_parse(DATA_PATH)
    print(f"[01] Loaded: N={len(raw)}")

    # 2. Remove missing values
    data_clean = raw.dropna(subset=NUMERIC_COLUMNS).copy()
    n_missing  = len(raw) - len(data_clean)
    print(f"[01] After missing removal: N={len(data_clean)} (removed {n_missing})")

    # 3. Normality tests (run on N=88)
    sw_table = shapiro_wilk_table(data_clean)
    print(f"[01] Shapiro-Wilk (N={len(data_clean)}):")
    print(sw_table.to_string(index=False))

    # 4. Robust scaling (justified by non-normality)
    scaler       = RobustScaler()
    X_scaled     = scaler.fit_transform(data_clean[NUMERIC_COLUMNS])

    # 5. Mahalanobis outlier removal
    data_final, scaled_final, maha_dists = mahalanobis_filter(
        data_clean, X_scaled, threshold_pct=0.975
    )
    n_outliers = len(data_clean) - len(data_final)
    print(f"[01] After Mahalanobis (p97.5): N={len(data_final)} "
          f"(removed {n_outliers}, {n_outliers/len(data_clean):.1%})")

    # 6. Save outputs
    data_final.to_csv(f'{RESULTS_DIR}/data_final.csv', index=False)
    sw_table.to_csv(f'{RESULTS_DIR}/shapiro_wilk.csv', index=False)

    # Save scaling params for reproducibility
    scaler_params = {
        'center': scaler.center_.tolist(),
        'scale':  scaler.scale_.tolist(),
        'columns': NUMERIC_COLUMNS
    }
    with open(f'{RESULTS_DIR}/scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)

    print(f"[01] Saved: {RESULTS_DIR}/data_final.csv, shapiro_wilk.csv, scaler_params.json")
