"""
04_validation.py
================
Statistical validation of the rating system (within-sample, PoC scope).

Analyses:
  Level 1  — Spearman rank correlations (ordinal coherence, partial circularity)
  Level 2  — Kruskal-Wallis + Benjamini-Hochberg FDR (group separation)
           — Cohen's d with 95% bootstrap CI (effect sizes, 2000 resamples)
           — Dunn post-hoc pairwise comparisons (FDR-BH)

Note: All analyses are within the observed sample (N=85). Out-of-sample
temporal validation (May-Jun 2025) is the primary planned extension.

Inputs:  results/traders_rated.csv
Outputs: results/validation_results.json
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import json

RESULTS_DIR    = 'results'
N_BOOTSTRAP    = 2000
RANDOM_SEED    = 42
VALIDATION_METRICS = ['Sharpe Ratio', 'MDD', 'Proporción de ganancias', 'ROI']
DUNN_METRICS       = ['Sharpe Ratio', 'MDD', 'Proporción de ganancias']


def spearman(df: pd.DataFrame, score_col: str, metric: str) -> dict:
    rho, p = stats.spearmanr(df[score_col], df[metric])
    return {'rho': round(float(rho), 3), 'p': round(float(p), 4)}


def cohen_d_bootstrap(a: np.ndarray, b: np.ndarray,
                      n_boot: int = N_BOOTSTRAP,
                      ci: float = 0.95,
                      seed: int = RANDOM_SEED) -> dict:
    """Bootstrap CI for Cohen's d (Q4 vs Q1). Uses sample std (ddof=1)."""
    rng = np.random.default_rng(seed)

    def _d(x, y):
        pooled = np.sqrt((x.std(ddof=1)**2 + y.std(ddof=1)**2) / 2)
        return (x.mean() - y.mean()) / pooled if pooled > 0 else 0.0

    obs = _d(b, a)
    boots = [_d(rng.choice(b, len(b), replace=True),
                rng.choice(a, len(a), replace=True))
             for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boots, [alpha * 100, (1 - alpha) * 100])
    return {'d': round(float(obs), 3),
            'ci_lo': round(float(lo), 3),
            'ci_hi': round(float(hi), 3)}


def kruskal_wallis(df: pd.DataFrame, metric: str) -> dict:
    groups = [df[df['Final_Rating'] == r][metric].dropna().values
              for r in [1, 2, 3, 4]]
    H, p = stats.kruskal(*groups)
    return {'H': round(float(H), 3), 'p': round(float(p), 4)}


def dunn_test(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return sp.posthoc_dunn(df, val_col=metric,
                           group_col='Final_Rating', p_adjust='fdr_bh')


if __name__ == '__main__':
    df = pd.read_csv(f'{RESULTS_DIR}/traders_rated.csv')
    print(f"[04] N={len(df)}")

    q1 = df[df['Final_Rating'] == 1]
    q4 = df[df['Final_Rating'] == 4]

    results = {'spearman': {}, 'kruskal_wallis': {}, 'cohen_d': {}, 'dunn': {}}
    kw_pvals = []

    for metric in VALIDATION_METRICS:
        sp_r  = spearman(df, 'Fusion_score', metric)
        kw_r  = kruskal_wallis(df, metric)
        cd_r  = cohen_d_bootstrap(q1[metric].values, q4[metric].values)
        results['spearman'][metric]       = sp_r
        results['kruskal_wallis'][metric] = kw_r
        results['cohen_d'][metric]        = cd_r
        kw_pvals.append(kw_r['p'])
        print(f"  {metric:<32} ρ={sp_r['rho']:+.3f}  "
              f"d={cd_r['d']:+.3f} [{cd_r['ci_lo']:.2f},{cd_r['ci_hi']:.2f}]  "
              f"H={kw_r['H']:.2f}")

    # FDR correction (Benjamini-Hochberg)
    reject, q_adj, _, _ = multipletests(kw_pvals, method='fdr_bh')
    for i, metric in enumerate(VALIDATION_METRICS):
        results['kruskal_wallis'][metric]['q_BH'] = round(float(q_adj[i]), 4)
        results['kruskal_wallis'][metric]['reject_H0'] = bool(reject[i])
    print("\n[04] FDR-adjusted q-values (BH):")
    for metric, q in zip(VALIDATION_METRICS, q_adj):
        print(f"  {metric:<32} q={q:.4f}")

    # Dunn post-hoc
    print("\n[04] Dunn post-hoc pairwise (FDR-BH):")
    for metric in DUNN_METRICS:
        dunn = dunn_test(df, metric)
        results['dunn'][metric] = dunn.round(4).to_dict()
        print(f"\n  {metric}:")
        print(dunn.round(3).to_string())

    with open(f'{RESULTS_DIR}/validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[04] Saved: validation_results.json")
