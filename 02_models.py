"""
02_models.py
============
Applies the four base clustering models to the analytic sample (N=85).

Models and roles:
  K-Means (k=4)         — structural exploration, cluster quality ordering
  GMM (k=4, full cov.)  — probabilistic soft signal → contributes to FS
  Autoencoder + K-Means — structural exploration, validates linearity claim
  PCA(4) + K-Means      — equivalent to AE, recommended for future use
  DBSCAN                — anomaly detection → contributes to FS as penalty

Inputs:  results/data_final.csv, results/scaler_params.json
Outputs: results/model_labels.csv, results/model_metrics.json
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from scipy.spatial.distance import mahalanobis

warnings.filterwarnings('ignore')

RESULTS_DIR    = 'results'
K_DESIGN       = 4
RANDOM_STATE   = 0      # seed=0 used as reference (stability tested over seeds 0-9)
AE_EPOCHS      = 200
AE_BATCH_SIZE  = 16
ENCODING_DIM   = 4      # aligned with K_DESIGN
DBSCAN_MINPTS  = 5      # pragmatic choice for N=85; theoretical d+1=9 too restrictive
DBSCAN_EPS_PCT = 90     # k-distance graph percentile for epsilon selection

NUMERIC_COLUMNS = [
    'ROI', 'Pnl', 'Sharpe Ratio', 'MDD',
    'Proporción de ganancias', 'Días con ganancias',
    'Pnl del copiador', 'Días de trading'
]


def load_and_scale(results_dir: str):
    """Load data_final and apply RobustScaler using saved params."""
    df = pd.read_csv(f'{results_dir}/data_final.csv')
    with open(f'{results_dir}/scaler_params.json') as f:
        params = json.load(f)
    scaler = RobustScaler()
    scaler.center_ = np.array(params['center'])
    scaler.scale_  = np.array(params['scale'])
    X = scaler.transform(df[NUMERIC_COLUMNS].values)
    return df, X


def run_kmeans(X: np.ndarray, k: int = K_DESIGN) -> tuple:
    """K-Means clustering with fixed seed for reproducibility."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    S  = silhouette_score(X, labels)
    DB = davies_bouldin_score(X, labels)
    return labels, S, DB


def run_gmm(X: np.ndarray, k: int = K_DESIGN) -> tuple:
    """GMM with full covariance matrices. Returns labels and probability matrix."""
    gmm = GaussianMixture(
        n_components=k, covariance_type='full',
        random_state=RANDOM_STATE
    )
    gmm.fit(X)
    labels = gmm.predict(X)
    proba  = gmm.predict_proba(X)   # shape (N, k) — used for GMM soft signal
    S  = silhouette_score(X, labels)
    DB = davies_bouldin_score(X, labels)
    return labels, proba, S, DB, gmm


def run_autoencoder(X: np.ndarray, encoding_dim: int = ENCODING_DIM,
                    k: int = K_DESIGN) -> tuple:
    """
    Symmetric autoencoder 8->16->4->16->8 + K-Means on latent space.

    Note: PCA(4)+K-Means achieves identical Silhouette (0.881) with zero
    stochastic variability. PCA is recommended for production use.
    The Autoencoder is retained as a structural explorer confirming that
    the data is dominated by a single linear quality gradient (PC1 = 97.4%
    of variance).
    """
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)

    inp = Input(shape=(X.shape[1],))
    h1  = Dense(16, activation='relu')(inp)
    enc = Dense(encoding_dim, activation='relu')(h1)
    h2  = Dense(16, activation='relu')(enc)
    out = Dense(X.shape[1], activation='linear')(h2)

    ae = Model(inp, out)
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(X, X, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE,
           shuffle=True, verbose=0)

    encoder = Model(inp, enc)
    Z       = encoder.predict(X, verbose=0)

    km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(Z)
    S      = silhouette_score(Z, labels)
    DB     = davies_bouldin_score(Z, labels)
    return labels, S, DB


def run_pca(X: np.ndarray, n_components: int = ENCODING_DIM,
            k: int = K_DESIGN) -> tuple:
    """
    PCA(4) + K-Means. Recommended over Autoencoder: identical performance,
    deterministic, no stochastic variability.
    """
    pca    = PCA(n_components=n_components, random_state=RANDOM_STATE)
    Z      = pca.fit_transform(X)
    var    = pca.explained_variance_ratio_

    km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(Z)
    S      = silhouette_score(Z, labels)
    DB     = davies_bouldin_score(Z, labels)
    return labels, S, DB, var


def run_dbscan(X: np.ndarray,
               min_pts: int = DBSCAN_MINPTS,
               eps_pct: float = DBSCAN_EPS_PCT) -> tuple:
    """
    DBSCAN with k-distance graph for epsilon selection.
    Typically returns 1 cluster + noise points (anomaly detector role).
    """
    nn    = NearestNeighbors(n_neighbors=min_pts)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    eps   = np.percentile(dists[:, min_pts - 1], eps_pct)

    db     = DBSCAN(eps=eps, min_samples=min_pts)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    return labels, n_clusters, n_noise, eps


if __name__ == '__main__':
    print("[02] Loading data...")
    df, X = load_and_scale(RESULTS_DIR)
    N = len(df)
    print(f"[02] N={N}, features={X.shape[1]}")

    # ── K-Means ──────────────────────────────────────────────────────────────
    print("[02] Running K-Means...")
    km_labels, km_S, km_DB = run_kmeans(X)
    print(f"     K-Means  | S={km_S:.3f} | DB={km_DB:.3f}")

    # ── GMM ──────────────────────────────────────────────────────────────────
    print("[02] Running GMM...")
    gmm_labels, gmm_proba, gmm_S, gmm_DB, gmm_model = run_gmm(X)
    print(f"     GMM      | S={gmm_S:.3f} | DB={gmm_DB:.3f}")

    # ── Autoencoder ───────────────────────────────────────────────────────────
    print("[02] Running Autoencoder + K-Means (seed=0)...")
    ae_labels, ae_S, ae_DB = run_autoencoder(X)
    print(f"     AE+KM    | S={ae_S:.3f} | DB={ae_DB:.3f}")

    # ── PCA ──────────────────────────────────────────────────────────────────
    print("[02] Running PCA(4) + K-Means...")
    pca_labels, pca_S, pca_DB, pca_var = run_pca(X)
    print(f"     PCA+KM   | S={pca_S:.3f} | DB={pca_DB:.3f}")
    print(f"              | Var explained: {pca_var.round(3)}")
    print(f"              | Cumulative:    {pca_var.cumsum().round(3)}")

    # ── DBSCAN ───────────────────────────────────────────────────────────────
    print("[02] Running DBSCAN...")
    db_labels, n_clusters, n_noise, eps = run_dbscan(X)
    print(f"     DBSCAN   | clusters={n_clusters} | noise={n_noise} | eps={eps:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    labels_df = df[['ROI', 'Sharpe Ratio', 'MDD', 'Proporción de ganancias']].copy()
    labels_df['KM_label']  = km_labels
    labels_df['GMM_label'] = gmm_labels
    labels_df['AE_label']  = ae_labels
    labels_df['PCA_label'] = pca_labels
    labels_df['DB_label']  = db_labels
    # Save GMM probabilities as well
    for k in range(K_DESIGN):
        labels_df[f'GMM_prob_{k}'] = gmm_proba[:, k]
    labels_df.to_csv(f'{RESULTS_DIR}/model_labels.csv', index=False)

    metrics = {
        'kmeans':  {'S': km_S,  'DB': km_DB},
        'gmm':     {'S': gmm_S, 'DB': gmm_DB},
        'ae_km':   {'S': ae_S,  'DB': ae_DB, 'seed': RANDOM_STATE},
        'pca_km':  {'S': pca_S, 'DB': pca_DB,
                    'var_explained': pca_var.tolist(),
                    'cumulative':    pca_var.cumsum().tolist()},
        'dbscan':  {'n_clusters': int(n_clusters), 'n_noise': int(n_noise),
                    'eps': float(eps), 'min_pts': DBSCAN_MINPTS},
    }
    with open(f'{RESULTS_DIR}/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[02] Saved: model_labels.csv, model_metrics.json")
