# Trader Rating PoC — Hybrid Unsupervised Rating System for Algorithmic Traders

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Proof%20of%20Concept-orange)]()

> **Proof of concept** for a hybrid unsupervised rating system that classifies
> algorithmic copy-traders in cryptocurrency markets across eight financial
> performance dimensions, achieving **2.33× greater MDD discrimination** over a
> Sharpe Ratio quartile baseline.

---

## Overview

This repository contains the full reproducible pipeline for the working paper:

> *A Hybrid Unsupervised Rating System for Algorithmic Traders in Cryptocurrency Markets*
> Tejada Caballero, A. · Vázquez López, L. · Ochoa Arellano, M. (2025/2026)

The system assigns traders to four rating tiers (*Very bad / Bad / Good / Very good*)
using a **Fusion Score** that integrates:

- **Individual Composite Score (CPS)**: weighted percentile ranking across 8 metrics
- **GMM soft signal**: probabilistic confidence from Gaussian Mixture Models
- **DBSCAN anomaly penalty**: density-based outlier flag

**Key empirical finding**: the 8-metric feature space is dominated by a single
linear quality gradient (PC1 explains 97.4% of variance), explaining why
K-Means, GMM, and Autoencoder all converge to near-identical partitions (σ = 0.985).
PCA(4) achieves identical Silhouette (0.881) to the Autoencoder and is the
recommended implementation for future deployments.

---

## Repository structure

```
trader-rating-poc/
├── src/
│   ├── 01_preprocessing.py     # Data loading, cleaning, RobustScaler, Mahalanobis
│   ├── 02_models.py            # K-Means, GMM, Autoencoder+KM, DBSCAN, PCA
│   ├── 03_fusion_score.py      # CPS, GMM soft signal, DBSCAN penalty, FS
│   ├── 04_validation.py        # Spearman, KW+FDR, Cohen's d bootstrap, Dunn
│   ├── 05_baseline.py          # Sharpe quartile baseline comparison
│   ├── 06_sensitivity.py       # CPS weight sensitivity (±0.05/0.10/0.15), Q_c stability
│   └── 07_visualization.py     # All 13 figures (ggplot style, bilingual)
├── notebooks/
│   └── full_pipeline.ipynb     # Complete end-to-end notebook
├── figures/                    # Generated figures (PNG, 200 dpi)
├── results/
│   └── traders_rated.csv       # Output ratings (shareable; no raw input data)
├── paper/
│   └── working_paper_v6.tex    # LaTeX source (PRIMEarxiv template)
├── data/
│   └── README_data.md          # Data collection instructions (Binance API)
├── requirements.txt
├── Makefile
└── README.md
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/yourusername/trader-rating-poc.git
cd trader-rating-poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your data
#    See data/README_data.md for Binance API collection instructions
#    Expected: traders_data.csv (100 rows, 10 columns) in data/

# 4. Run full pipeline
make all

# Or step by step:
python src/01_preprocessing.py    # N=100 → 88 → 85
python src/02_models.py           # K-Means, GMM, AE, DBSCAN, PCA
python src/03_fusion_score.py     # CPS, GMM soft, FS, ratings
python src/04_validation.py       # Statistical validation
python src/05_baseline.py         # Sharpe baseline comparison
python src/06_sensitivity.py      # Robustness analysis
python src/07_visualization.py    # All figures
```

---

## Data

The raw data (`traders_data.csv`) is not included due to Binance Terms of Service.
See `data/README_data.md` for:
- Variables collected (8 financial metrics per trader)
- Binance copy-trading API endpoint
- Collection script template
- Expected format and encoding (ISO-8859-1, semicolon-separated)

The **output file** (`results/traders_rated.csv`) with final ratings is included
and does not contain raw API data.

---

## Main results (N = 85, March–April 2025, Binance)

| Metric | Hybrid system | Sharpe baseline | Advantage |
|--------|--------------|-----------------|-----------|
| Sharpe Ratio (KW H) | 50.0 | 78.8 | Baseline better (by design) |
| **MDD (KW H)** | **26.6** | **11.4** | **Hybrid: 2.33×** |
| Win Rate (KW H) | 45.2 | 47.9 | Comparable |
| ROI (KW H) | 6.8 | 2.9 | Non-significant (both) |

Cohen's *d* (Very bad vs Very good):
- Sharpe Ratio: d = 2.183 [95% CI: 1.81, 3.09]
- MDD:          d = −1.992 [95% CI: −3.18, −1.42]
- Win Rate:     d = 2.711 [95% CI: 1.75, 4.88]

**PCA finding**: PC1 explains 97.4% of variance → the feature space is essentially
one-dimensional. PCA(4)+K-Means achieves S = 0.881, identical to the Autoencoder.

---

## Citation

```bibtex
@misc{tejada2025trader,
  title  = {A Hybrid Unsupervised Rating System for Algorithmic Traders
             in Cryptocurrency Markets},
  author = {Tejada Caballero, Alejandra and V\'azquez L\'opez, Lucas
             and Ochoa Arellano, Maicol},
  year   = {2025},
  note   = {Working paper. UNIR -- Universidad Internacional de La Rioja.}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
The paper (paper/) is under CC BY 4.0.
