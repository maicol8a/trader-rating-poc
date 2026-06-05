"""
05_baseline.py
==============
Compares the hybrid system against a Sharpe Ratio quartile-ranking baseline.

The baseline assigns traders to four equal-sized tiers by Sharpe Ratio alone.
Key finding: hybrid achieves 2.33× greater MDD discrimination (KW H=26.6 vs 11.4),
its primary advantage, while the baseline wins on Sharpe Ratio self-correlation
(by construction — it was optimised for exactly that metric).

Inputs:  results/traders_rated.csv
Outputs: results/baseline_results.json
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json

RESULTS_DIR = 'results'
VALIDATION_METRICS = ['Sharpe Ratio', 'MDD', 'Proporción de ganancias', 'ROI']
N_BOOTSTRAP = 2000


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0


if __name__ == '__main__':
    df = pd.read_csv(f'{RESULTS_DIR}/traders_rated.csv')
    print(f"[05] N={len(df)}")

    # Baseline: Sharpe Ratio quartiles
    df['Sharpe_Quartile'] = pd.qcut(
        df['Sharpe Ratio'], q=4, labels=[1, 2, 3, 4], duplicates='drop'
    ).astype(int)

    results = {'hybrid': {}, 'baseline': {}, 'comparison': {}}

    h_kw_pvals = []; b_kw_pvals = []

    print(f"\n{'Metric':<32} {'Hybrid d':>10} {'Hybrid H':>10} "
          f"{'Base d':>10} {'Base H':>10} {'MDD advantage':>14}")
    print("-" * 88)

    for metric in VALIDATION_METRICS:
        # Hybrid
        hq1   = df[df['Final_Rating'] == 1][metric].values
        hq4   = df[df['Final_Rating'] == 4][metric].values
        h_d   = cohen_d(hq1, hq4)
        h_H, h_p = stats.kruskal(*[df[df['Final_Rating']==r][metric].values
                                    for r in [1,2,3,4]])
        # Baseline
        bq1   = df[df['Sharpe_Quartile'] == 1][metric].values
        bq4   = df[df['Sharpe_Quartile'] == 4][metric].values
        b_d   = cohen_d(bq1, bq4)
        b_H, b_p = stats.kruskal(*[df[df['Sharpe_Quartile']==r][metric].values
                                    for r in [1,2,3,4]])

        h_kw_pvals.append(h_p); b_kw_pvals.append(b_p)

        results['hybrid'][metric]   = {'d': round(h_d,3), 'H': round(float(h_H),3), 'p': round(float(h_p),4)}
        results['baseline'][metric] = {'d': round(b_d,3), 'H': round(float(b_H),3), 'p': round(float(b_p),4)}

        adv = f"{h_H/b_H:.2f}x" if metric == 'MDD' else ""
        print(f"{metric:<32} {h_d:>+10.3f} {h_H:>10.1f} {b_d:>+10.3f} {b_H:>10.1f} {adv:>14}")

    # FDR on both
    _, h_q, _, _ = multipletests(h_kw_pvals, method='fdr_bh')
    _, b_q, _, _ = multipletests(b_kw_pvals, method='fdr_bh')
    for i, metric in enumerate(VALIDATION_METRICS):
        results['hybrid'][metric]['q_BH']   = round(float(h_q[i]), 4)
        results['baseline'][metric]['q_BH'] = round(float(b_q[i]), 4)

    mdd_ratio = results['hybrid']['MDD']['H'] / results['baseline']['MDD']['H']
    results['comparison']['MDD_H_ratio'] = round(mdd_ratio, 3)
    results['comparison']['note'] = (
        f"Hybrid achieves {mdd_ratio:.2f}x greater MDD discrimination. "
        "Baseline wins on Sharpe by construction (optimised for that metric)."
    )
    print(f"\n[05] MDD KW H ratio (hybrid/baseline): {mdd_ratio:.2f}x")

    with open(f'{RESULTS_DIR}/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[05] Saved: baseline_results.json")
