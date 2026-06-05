"""
07_visualization.py
===================
Generates all 13 figures for the working paper.

Figures:
  01  Distributions of 8 variables (N=88)
  02  Pearson correlation matrix
  03  K-Means validation curves (elbow + Silhouette)
  04  DBSCAN k-distance graph
  05  Fusion Score distribution with quartile lines
  06  Final rating distribution (bar chart)
  07  Mean performance profile by category
  08  Boxplots (Sharpe + MDD) by category
  09  Scatter: Fusion Score vs Sharpe Ratio
  10  Forest plot: Cohen's d with 95% CI
  11  t-SNE and UMAP projections
  12  Baseline comparison (Hybrid vs Sharpe quartile)
  13  CPS weight sensitivity (Appendix)

Style: matplotlib/ggplot, bilingual EN/ES titles, 200 dpi, consistent palette.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import mahalanobis
from scipy import stats
import umap
import warnings
import os
import json

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)

RESULTS_DIR = 'results'
DPI         = 200
LABEL_MAP   = {1:'Very bad', 2:'Bad', 3:'Good', 4:'Very good'}
CAT_ORDER   = list(LABEL_MAP.values())
CAT_COLORS  = {
    'Very bad':  '#c0392b', 'Bad': '#e67e22',
    'Good': '#2980b9', 'Very good': '#1a5276',
}
NUMERIC_COLUMNS = [
    'ROI', 'Pnl', 'Sharpe Ratio', 'MDD',
    'Proporción de ganancias', 'Días con ganancias',
    'Pnl del copiador', 'Días de trading'
]
METRIC_ES = {
    'ROI': 'ROI (%)', 'Pnl': 'PnL',
    'Sharpe Ratio': 'Sharpe Ratio', 'MDD': 'MDD (%)',
    'Proporción de ganancias': 'Win Rate (%)\nProp. ganancias',
    'Días con ganancias': 'Profitable Days\nDías c/ganancias',
    'Pnl del copiador': 'Copier PnL\nPnL copiador',
    'Días de trading': 'Trading Days\nDías de trading',
}


def load_all():
    """Load all required datasets."""
    data_raw   = pd.read_csv('data/traders_data.csv',
                              encoding='ISO-8859-1', delimiter=';')
    for col in NUMERIC_COLUMNS:
        data_raw[col] = pd.to_numeric(
            data_raw[col].astype(str)
            .str.replace('.','',regex=False).str.replace(',','.',regex=False)
            .str.replace('%','',regex=False).str.strip(), errors='coerce')
    data_clean = data_raw.dropna(subset=NUMERIC_COLUMNS).copy()
    scaler = RobustScaler()
    X_sc   = scaler.fit_transform(data_clean[NUMERIC_COLUMNS])
    mv = X_sc.mean(axis=0); ci = np.linalg.inv(np.cov(X_sc.T))
    maha = np.array([mahalanobis(X_sc[i], mv, ci) for i in range(len(X_sc))])
    mask = maha <= np.percentile(maha, 97.5)
    X85  = X_sc[mask]
    df   = pd.read_csv(f'{RESULTS_DIR}/traders_rated.csv')
    return data_clean, X85, df


def save(fig, name):
    fig.savefig(f'figures/{name}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: figures/{name}")


# ── Figure 01: Distributions ─────────────────────────────────────────────────
def fig01(data_clean):
    def compact(x, pos):
        if abs(x)>=1e6: return f'{x/1e6:.0f}M'
        if abs(x)>=1e3: return f'{x/1e3:.0f}K'
        return f'{x:.0f}'
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    plt.style.use('ggplot')
    for i, col in enumerate(NUMERIC_COLUMNS):
        ax = axes.flatten()[i]
        ax.set_facecolor('#f8f9fa')
        ax.grid(color='#dfe6e9', lw=0.5, axis='y')
        ax.hist(data_clean[col].dropna(), bins=18, color='#2980b9',
                alpha=0.85, edgecolor='white', lw=0.4)
        ax.set_title(METRIC_ES[col], fontsize=8.5, fontweight='bold', pad=4)
        ax.set_ylabel('Count', fontsize=7.5)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(compact))
        ax.tick_params(axis='x', labelsize=7, rotation=30)
        ax.tick_params(axis='y', labelsize=7)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Distribution of performance variables / '
                 'Distribución de métricas  ·  n=88',
                 fontsize=9.5, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig01_distribuciones.png')


# ── Figure 02: Correlation matrix ────────────────────────────────────────────
def fig02(data_clean):
    corr = data_clean[NUMERIC_COLUMNS].corr()
    labels = [METRIC_ES[c].split('\n')[0] for c in NUMERIC_COLUMNS]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='Pearson r')
    ax.set_xticks(range(len(NUMERIC_COLUMNS)))
    ax.set_yticks(range(len(NUMERIC_COLUMNS)))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    for i in range(len(NUMERIC_COLUMNS)):
        for j in range(len(NUMERIC_COLUMNS)):
            ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center',
                    fontsize=7.5, color='white' if abs(corr.values[i,j])>0.5 else 'black')
    ax.set_title('Pearson Correlation Matrix / Matriz de correlaciones de Pearson\n'
                 'n=85  (analytic sample)', fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    save(fig, 'fig02_correlaciones.png')


# ── Figure 03: K-Means validation ────────────────────────────────────────────
def fig03(X85):
    k_range = range(2, 7)
    inertias, sils, dbs, chs = [], [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X85)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X85, lbl))
        dbs.append(davies_bouldin_score(X85, lbl))
        from sklearn.metrics import calinski_harabasz_score
        chs.append(calinski_harabasz_score(X85, lbl))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    plt.style.use('ggplot')
    ks = list(k_range)

    # Elbow
    ax1.plot(ks, inertias, 'o-', color='#2980b9', lw=2, ms=6)
    ax1.axvline(4, color='#2ecc71', lw=1.8, ls='--', label='K=4 (selected)')
    ax1.set_xlabel('Number of clusters K'); ax1.set_ylabel('Inertia (SSE)')
    ax1.set_title('Elbow method / Método del codo', fontweight='bold')
    ax1.legend(fontsize=9); ax1.set_facecolor('#f8f9fa')

    # Silhouette
    ax2.plot(ks, sils, 's-', color='#e74c3c', lw=2, ms=6, label='Silhouette')
    ax2.axvline(4, color='#2ecc71', lw=1.8, ls='--', label='K=4 (selected)')
    ax2.set_xlabel('Number of clusters K'); ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Index / Índice de Silhouette', fontweight='bold')
    ax2.legend(fontsize=9); ax2.set_facecolor('#f8f9fa')

    fig.suptitle('K-Means validation / Validación K-Means  ·  N=85',
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig03_kmeans_validacion.png')


# ── Figure 04: DBSCAN k-distance ─────────────────────────────────────────────
def fig04(X85):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X85)
    dists, _ = nn.kneighbors(X85)
    d = np.sort(dists[:, 4])
    eps = np.percentile(d, 90)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(d)), d, color='#2980b9', lw=1.8)
    ax.axhline(eps, color='#c0392b', lw=1.5, ls='--',
               label=f'ε = {eps:.4f}  (90th percentile)')
    ax.set_xlabel('Points sorted by 5th-NN distance / Puntos ordenados')
    ax.set_ylabel('5th-NN distance / Distancia al 5° vecino más cercano')
    ax.set_title('DBSCAN k-distance graph  ·  MinPts=5  ·  N=85', fontweight='bold')
    ax.legend(fontsize=9); ax.set_facecolor('#f8f9fa')
    ax.grid(color='#dfe6e9', lw=0.5)
    fig.tight_layout()
    save(fig, 'fig04_dbscan_kdistance.png')


# ── Figures 05–13 (rating-dependent, load rated data) ────────────────────────
def fig05(df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_facecolor('#f8f9fa'); ax.grid(color='#dfe6e9', lw=0.5, axis='y')
    for r, lab in LABEL_MAP.items():
        sub = df[df['Final_Rating']==r]['Fusion_score']
        ax.hist(sub, bins=12, alpha=0.75, color=CAT_COLORS[lab], label=lab, edgecolor='white')
    for q in [0.25, 0.50, 0.75]:
        ax.axvline(df['Fusion_score'].quantile(q), color='#2c3e50', lw=1.2, ls='--', alpha=0.7)
    ax.set_xlabel('Fusion Score (FS)'); ax.set_ylabel('Count')
    ax.set_title('Fusion Score distribution by rating / Distribución del Fusion Score\n'
                 'Dashed lines = Q1, Q2, Q3 quartile cut-offs  ·  N=85', fontweight='bold')
    handles = [mpatches.Patch(color=CAT_COLORS[l], label=l) for l in CAT_ORDER]
    ax.legend(handles=handles, fontsize=9); fig.tight_layout()
    save(fig, 'fig05_fusion_score.png')


def fig06(df):
    dist = df['Final_Label'].value_counts().reindex(CAT_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_facecolor('#f8f9fa'); ax.grid(color='#dfe6e9', lw=0.5, axis='y')
    bars = ax.bar(CAT_ORDER, dist.values,
                  color=[CAT_COLORS[l] for l in CAT_ORDER], alpha=0.88, edgecolor='white')
    for bar, val in zip(bars, dist.values):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.2, f'{val}\n({val/85:.0%})',
                ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Traders'); ax.set_ylim(0, 30)
    ax.set_title('Final rating distribution / Distribución del rating final\n'
                 'N=85  ·  Quartile-based assignment', fontweight='bold')
    ax.spines[['top','right']].set_visible(False); fig.tight_layout()
    save(fig, 'fig06_rating_distribucion.png')


def fig07(df):
    metrics = ['ROI', 'Sharpe Ratio', 'MDD', 'Proporción de ganancias']
    profile = df.groupby('Final_Label')[metrics].mean().reindex(CAT_ORDER)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
    plt.style.use('ggplot')
    for ax, col in zip(axes, metrics):
        ax.set_facecolor('#f8f9fa')
        ax.grid(color='#dfe6e9', lw=0.5, axis='y')
        ax.bar(range(4), profile[col].values,
               color=[CAT_COLORS[l] for l in CAT_ORDER], alpha=0.88, edgecolor='white')
        ax.set_xticks(range(4)); ax.set_xticklabels(['VB','B','G','VG'], fontsize=8)
        ax.set_title(METRIC_ES[col].split('\n')[0], fontsize=9, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Mean performance profile by rating / Perfil medio por categoría  ·  N=85',
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout(); save(fig, 'fig07_perfil_categorias.png')


def fig08(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plt.style.use('ggplot')
    for ax, col, ylabel in [(axes[0],'Sharpe Ratio','Sharpe Ratio'),
                             (axes[1],'MDD','MDD (%)')]:
        ax.set_facecolor('#f8f9fa'); ax.grid(color='#dfe6e9', lw=0.5, axis='y')
        for j, (label, pos) in enumerate(zip(CAT_ORDER, range(4))):
            sub = df[df['Final_Label']==label][col].dropna().values
            col_c = CAT_COLORS[label]
            ax.boxplot(sub, positions=[pos], widths=0.5, patch_artist=True,
                       medianprops=dict(color='white', lw=2),
                       boxprops=dict(facecolor=col_c, alpha=0.85),
                       whiskerprops=dict(color=col_c, lw=1.5),
                       capprops=dict(color=col_c, lw=1.5),
                       flierprops=dict(marker='o', markerfacecolor=col_c,
                                       markeredgecolor='white', markeredgewidth=0.5,
                                       markersize=5, alpha=0.85))
        ax.set_xticks(range(4)); ax.set_xticklabels(CAT_ORDER, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_title(f'{ylabel} by category / por categoría', fontsize=9, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
    handles = [mpatches.Patch(color=CAT_COLORS[l], label=l) for l in CAT_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Sharpe Ratio and MDD distributions / Distribuciones  ·  N=85',
                 fontsize=9.5, fontweight='bold', y=1.01)
    fig.tight_layout(); save(fig, 'fig08_boxplots.png')


def fig09(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor('#f8f9fa'); ax.grid(color='#dfe6e9', lw=0.4)
    for r, lab in LABEL_MAP.items():
        sub = df[df['Final_Rating']==r]
        ax.scatter(sub['Fusion_score'], sub['Sharpe Ratio'],
                   c=CAT_COLORS[lab], label=lab, s=55, alpha=0.82,
                   edgecolors='none', zorder=3)
    m, b = np.polyfit(df['Fusion_score'], df['Sharpe Ratio'], 1)
    xs = np.linspace(df['Fusion_score'].min(), df['Fusion_score'].max(), 100)
    ax.plot(xs, m*xs+b, color='#2c3e50', lw=1.5, ls='--', alpha=0.7, label='Trend')
    rho, p = stats.spearmanr(df['Fusion_score'], df['Sharpe Ratio'])
    ax.text(0.05, 0.93, f'ρ = {rho:.3f}  (p < 0.001)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    handles = [mpatches.Patch(color=CAT_COLORS[l], label=l) for l in CAT_ORDER]
    handles.append(Line2D([0],[0], color='#2c3e50', ls='--', label='Trend'))
    ax.legend(handles=handles, fontsize=8.5)
    ax.set_xlabel('Fusion Score (FS)', fontsize=10); ax.set_ylabel('Sharpe Ratio', fontsize=10)
    ax.set_title('Fusion Score vs Sharpe Ratio / Validación Spearman\n'
                 'N=85  ·  Coloured by rating category', fontweight='bold')
    ax.spines[['top','right']].set_visible(False); fig.tight_layout()
    save(fig, 'fig09_scatter_validacion.png')


def fig10():
    # Forest plot: Cohen's d with 95% CI
    forest = {
        'Metric':  ['Sharpe Ratio', 'Win Rate', 'MDD', 'ROI'],
        'd':       [ 2.183,          2.711,     -1.992,  0.617],
        'ci_lo':   [ 1.806,          1.752,     -3.176,  0.020],
        'ci_hi':   [ 3.085,          4.884,     -1.418,  1.241],
        'sig':     [True, True, True, False],
    }
    df_f = pd.DataFrame(forest).sort_values('d', key=abs)
    y    = np.arange(len(df_f))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_facecolor('#f8f9fa'); ax.grid(color='#dfe6e9', lw=0.5, axis='x')
    for i, (_, row) in enumerate(df_f.iterrows()):
        c = '#2980b9' if row['sig'] else '#95a5a6'
        ax.plot([row['ci_lo'], row['ci_hi']], [i,i], color=c, lw=2.2, solid_capstyle='round')
        ax.scatter(row['d'], i, color=c, s=80, zorder=3,
                   marker='D' if row['sig'] else 'o')
        x_lbl = row['ci_hi']+0.07 if row['d']>0 else row['ci_lo']-0.07
        ha    = 'left' if row['d']>0 else 'right'
        ax.text(x_lbl, i, f"d={row['d']:.2f} [{row['ci_lo']:.2f},{row['ci_hi']:.2f}]"
                f"  q={'<0.001' if row['sig'] else '0.080'}",
                va='center', ha=ha, fontsize=7.5)
    ax.axvline(0,   color='#2c3e50', lw=0.9, alpha=0.5)
    ax.axvline(0.8, color='#c0392b', lw=0.9, ls='--', alpha=0.7, label='|d|=0.8 (large)')
    ax.axvline(-0.8,color='#c0392b', lw=0.9, ls='--', alpha=0.7)
    ax.set_yticks(y); ax.set_yticklabels(df_f['Metric'].values, fontsize=9.5)
    ax.set_xlabel("Cohen's d  [95% bootstrap CI]  (Very good vs Very bad)", fontsize=9)
    ax.set_title("Effect sizes by metric / Tamaño del efecto por métrica\n"
                 "♦ = FDR significant  ·  ○ = non-significant  ·  N=85", fontweight='bold')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(-4.5, 6.5); fig.tight_layout()
    save(fig, 'fig10_validacion_resumen.png')


def fig11(X85, df):
    tsne   = TSNE(n_components=2, perplexity=15, random_state=42, max_iter=1000)
    t_c    = tsne.fit_transform(X85)
    reducer= umap.UMAP(n_components=2, n_neighbors=12, min_dist=0.3, random_state=42)
    u_c    = reducer.fit_transform(X85)
    ratings= df['Final_Rating'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    plt.style.use('ggplot')
    for ax, coords, title in [(axes[0], t_c, 't-SNE (perplexity=15)'),
                               (axes[1], u_c, 'UMAP (n_neighbors=12, min_dist=0.3)')]:
        ax.set_facecolor('#f8f9fa'); ax.grid(color='#dfe6e9', lw=0.4, zorder=0)
        for r, lab in LABEL_MAP.items():
            mask = ratings == r
            ax.scatter(coords[mask,0], coords[mask,1], c=CAT_COLORS[lab],
                       label=lab, s=55, alpha=0.82, edgecolors='none', zorder=3)
        ax.set_title(f'{title}\nColoured by final rating / Por categoría',
                     fontsize=9.5, fontweight='bold', pad=8)
        ax.set_xlabel('Dimension 1', fontsize=9); ax.set_ylabel('Dimension 2', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        handles = [mpatches.Patch(color=CAT_COLORS[l], label=l) for l in CAT_ORDER]
        ax.legend(handles=handles, fontsize=8, title='Rating', framealpha=0.9)
    fig.suptitle('Dimensionality Reduction: 8-metric space  ·  N=85\n'
                 'Reducción de dimensionalidad: espacio de 8 métricas', fontweight='bold', y=1.01)
    fig.tight_layout(); save(fig, 'fig11_tsne_umap.png')


def fig12():
    metrics  = ['Sharpe Ratio', 'MDD', 'Win Rate', 'ROI']
    hybrid_H = [50.018, 26.640, 45.195, 6.759]
    base_H   = [78.763, 11.430, 47.930, 2.885]
    hybrid_d = [2.183, -1.992, 2.711, 0.617]
    base_d   = [-3.819, -0.761, 2.822, 0.059]
    x = np.arange(len(metrics)); w = 0.35
    c_h='#2980b9'; c_b='#95a5a6'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax in [ax1, ax2]:
        ax.set_facecolor('#f8f9fa'); ax.grid(color='#dfe6e9', lw=0.5, axis='y')

    b1=ax1.bar(x-w/2, hybrid_H, w, label='Hybrid / Híbrido', color=c_h, alpha=0.88)
    b2=ax1.bar(x+w/2, base_H,   w, label='Sharpe baseline',  color=c_b, alpha=0.88)
    for bar in b1: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                            f'{bar.get_height():.1f}', ha='center', fontsize=8.5, fontweight='bold', color=c_h)
    for bar in b2: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                            f'{bar.get_height():.1f}', ha='center', fontsize=8.5, color='#555')
    ax1.annotate('2.33× better\n2.33× mejor', xy=(1, 26.64), xytext=(1.7, 52),
                 fontsize=8, color='#c0392b', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))
    ax1.set_xticks(x); ax1.set_xticklabels(metrics, fontsize=9)
    ax1.set_ylabel('Kruskal–Wallis H'); ax1.legend(fontsize=8)
    ax1.set_title('Group separation: KW H  (higher = better)\nSeparación entre grupos', fontweight='bold')

    b3=ax2.bar(x-w/2, np.abs(hybrid_d), w, label='Hybrid / Híbrido', color=c_h, alpha=0.88)
    b4=ax2.bar(x+w/2, np.abs(base_d),   w, label='Sharpe baseline',  color=c_b, alpha=0.88)
    for bar, val in zip(b3, hybrid_d):
        s='−' if val<0 else '+'
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{s}{abs(val):.2f}', ha='center', fontsize=8.5, fontweight='bold', color=c_h)
    for bar, val in zip(b4, base_d):
        s='−' if val<0 else '+'
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{s}{abs(val):.2f}', ha='center', fontsize=8.5, color='#555')
    ax2.axhline(0.8, color='#c0392b', lw=1.2, ls='--', alpha=0.7, label='|d|=0.8 (large)')
    ax2.set_xticks(x); ax2.set_xticklabels(metrics, fontsize=9)
    ax2.set_ylabel("|Cohen's d|"); ax2.legend(fontsize=8)
    ax2.set_title("Effect size |d|  (higher = larger effect)\nTamaño del efecto", fontweight='bold')

    fig.suptitle('Hybrid vs Sharpe Quartile Baseline  ·  N=85\n'
                 'Sistema híbrido vs baseline de cuartiles de Sharpe', fontweight='bold', y=1.01)
    fig.tight_layout(); save(fig, 'fig12_baseline_comparison.png')


def fig13():
    sensitivity = {
        'Sharpe Ratio': {-0.15:0.812,-0.10:0.859,-0.05:0.929,0:1.0,0.05:0.929,0.10:0.929,0.15:0.882},
        'MDD':          {-0.15:0.788,-0.10:0.859,-0.05:0.953,0:1.0,0.05:0.859,0.10:0.812,0.15:0.718},
        'ROI':          {-0.15:0.812,-0.10:0.859,-0.05:0.882,0:1.0,0.05:0.894,0.10:0.741,0.15:0.647},
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    plt.style.use('ggplot')
    for ax, (metric, data_s) in zip(axes, sensitivity.items()):
        ws    = sorted(data_s.keys())
        vals  = [data_s[w] for w in ws]
        colors= ['#c0392b' if w==0 else '#2980b9' for w in ws]
        bars  = ax.bar(ws, [v*100 for v in vals], width=0.03,
                       color=colors, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val*100+0.3,
                    f'{val*100:.1f}%', ha='center', fontsize=8.5, fontweight='bold')
        ax.set_ylim(60, 105); ax.set_xticks(ws)
        ax.set_xticklabels([f'{w:+.2f}' for w in ws], fontsize=7.5, rotation=30)
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.set_ylabel('Agreement (%)', fontsize=8.5)
        ax.axhline(90, color='#7f8c8d', lw=0.8, ls=':', alpha=0.7)
        ax.set_facecolor('#f8f9fa'); ax.spines[['top','right']].set_visible(False)
    red  = mpatches.Patch(color='#c0392b', label='Base weight')
    blue = mpatches.Patch(color='#2980b9', label='Perturbed weight')
    fig.legend(handles=[red, blue], loc='upper right', fontsize=9)
    fig.suptitle('CPS weight sensitivity  ·  Rating agreement vs base config  ·  N=85',
                 fontsize=10, fontweight='bold', y=1.04)
    fig.tight_layout(); save(fig, 'fig13_sensitivity.png')


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("[07] Loading data...")
    data_clean, X85, df = load_all()

    print("[07] Generating figures...")
    fig01(data_clean)
    fig02(data_clean)
    fig03(X85)
    fig04(X85)
    fig05(df)
    fig06(df)
    fig07(df)
    fig08(df)
    fig09(df)
    fig10()
    fig11(X85, df)
    fig12()
    fig13()
    print("[07] All 13 figures saved to figures/")
