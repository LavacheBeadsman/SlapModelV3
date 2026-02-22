"""
TE Model Redundancy Check:
Are breakout age (15% threshold) and Rec/TPA measuring the same thing?

1. Correlation between the two non-DC components
2. Multivariate regression: PPG ~ DC + breakout + production
3. Does a 3-component model beat a 2-component model?
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
bt = pd.read_csv('data/te_backtest_master.csv')
bt = bt[bt['nfl_seasons_found'] > 0].copy()
bt['dc_score'] = bt['pick'].apply(lambda p: max(0, min(100, 100 - 2.40 * (p**0.62 - 1))))

# ============================================================
# REBUILD BREAKOUT SCORE AT 15% THRESHOLD (from PFF multi-season)
# ============================================================
pff_file_map = {
    'data/receiving_summary (2).csv': 2015,
    'data/receiving_summary (3).csv': 2016,
    'data/receiving_summary (4).csv': 2017,
    'data/receiving_summary (5).csv': 2018,
    'data/receiving_summary (21).csv': 2019,
    'data/receiving_summary (20).csv': 2020,
    'data/receiving_summary (19).csv': 2021,
    'data/receiving_summary (18).csv': 2022,
    'data/receiving_summary (17).csv': 2023,
    'data/receiving_summary (16).csv': 2024,
    'data/receiving_summary (15).csv': 2025,
}

all_pff = []
for filepath, season in pff_file_map.items():
    pff = pd.read_csv(filepath)
    pff['season'] = season
    all_pff.append(pff)
pff_all = pd.concat(all_pff, ignore_index=True)
receivers = pff_all[pff_all['position'].isin(['TE', 'WR', 'HB'])].copy()
team_totals = receivers.groupby(['team_name', 'season'])['yards'].sum().reset_index()
team_totals.rename(columns={'yards': 'team_rec_yards'}, inplace=True)
te_pff = receivers[receivers['position'] == 'TE'].copy()
te_pff = te_pff.merge(team_totals, on=['team_name', 'season'], how='left')
te_pff['dominator_pct'] = np.where(te_pff['team_rec_yards'] > 0,
    (te_pff['yards'] / te_pff['team_rec_yards']) * 100, 0)

def norm_name(n):
    s = str(n).lower().strip()
    for suf in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
        s = s.replace(suf, '')
    s = s.replace("'", "").replace("\u2019", "").replace("-", "").replace(".", "")
    return s.strip()

te_pff['name_norm'] = te_pff['player'].apply(norm_name)
bt['name_norm2'] = bt['player_name'].apply(norm_name)

# Build multi-season records
msd = []
for _, te in bt.iterrows():
    matches = te_pff[(te_pff['name_norm'] == te['name_norm2']) & (te_pff['season'] < te['draft_year'])]
    for _, pm in matches.iterrows():
        msd.append({
            'player_name': te['player_name'],
            'draft_year': te['draft_year'],
            'draft_age': te['draft_age'],
            'season': pm['season'],
            'season_age': te['draft_age'] - (te['draft_year'] - pm['season']),
            'dominator_pct': pm['dominator_pct'],
        })
msd = pd.DataFrame(msd)

# Calculate breakout score at 15% threshold
THRESH = 15
bo_scores = {}
for te_name in msd['player_name'].unique():
    te_seasons = msd[msd['player_name'] == te_name].sort_values('season')
    peak_dom = te_seasons['dominator_pct'].max()
    hit_seasons = te_seasons[te_seasons['dominator_pct'] >= THRESH]
    if len(hit_seasons) > 0:
        bo_age = hit_seasons.iloc[0]['season_age']
        if bo_age <= 18: base = 100
        elif bo_age <= 19: base = 90
        elif bo_age <= 20: base = 75
        elif bo_age <= 21: base = 60
        elif bo_age <= 22: base = 45
        elif bo_age <= 23: base = 30
        else: base = 20
        bonus = min((peak_dom - THRESH) * 0.5, 9.9)
        score = min(base + bonus, 99.9)
    else:
        score = min(35, 15 + peak_dom)
    bo_scores[te_name] = score

bt['breakout_score_15'] = bt['player_name'].map(bo_scores)

# Build Rec/TPA
bt['rec_per_tpa'] = np.where(
    (bt['cfbd_receptions'].notna()) & (bt['cfbd_team_pass_att'].notna()) & (bt['cfbd_team_pass_att'] > 0),
    bt['cfbd_receptions'] / bt['cfbd_team_pass_att'], np.nan)

# Also build PFF Rec/Game for hybrid fallback
bt['pff_rpg'] = np.where(
    (bt['pff_receptions'].notna()) & (bt['pff_player_game_count'].notna()) & (bt['pff_player_game_count'] > 0),
    bt['pff_receptions'] / bt['pff_player_game_count'], np.nan)

# Normalize Rec/TPA and PFF RPG to 0-100 for hybrid
cfbd_vals = bt['rec_per_tpa'].dropna()
pff_vals = bt['pff_rpg'].dropna()
bt['rec_tpa_norm'] = np.where(bt['rec_per_tpa'].notna(),
    (bt['rec_per_tpa'] - cfbd_vals.min()) / (cfbd_vals.max() - cfbd_vals.min()) * 100, np.nan)
bt['pff_rpg_norm'] = np.where(bt['pff_rpg'].notna(),
    (bt['pff_rpg'] - pff_vals.min()) / (pff_vals.max() - pff_vals.min()) * 100, np.nan)
bt['production_hybrid'] = np.where(bt['rec_tpa_norm'].notna(), bt['rec_tpa_norm'], bt['pff_rpg_norm'])

print(f"TEs with outcomes: {len(bt)}")
print(f"TEs with breakout_score_15: {bt['breakout_score_15'].notna().sum()}")
print(f"TEs with rec_per_tpa: {bt['rec_per_tpa'].notna().sum()}")
print(f"TEs with production_hybrid: {bt['production_hybrid'].notna().sum()}")

# ============================================================
# 1. CORRELATION BETWEEN THE TWO NON-DC COMPONENTS
# ============================================================
print(f"\n{'='*80}")
print("1. CORRELATION: Breakout Score (15%) vs Rec/TPA")
print(f"{'='*80}")

# Using Rec/TPA directly
both = bt[bt['breakout_score_15'].notna() & bt['rec_per_tpa'].notna()].copy()
print(f"\nTEs with BOTH metrics: {len(both)}")
if len(both) >= 10:
    r, p = stats.pearsonr(both['breakout_score_15'], both['rec_per_tpa'])
    print(f"Pearson r = {r:+.3f}, p = {p:.4f}")
    rho, p_rho = stats.spearmanr(both['breakout_score_15'], both['rec_per_tpa'])
    print(f"Spearman rho = {rho:+.3f}, p = {p_rho:.4f}")

# Using hybrid production
both_h = bt[bt['breakout_score_15'].notna() & bt['production_hybrid'].notna()].copy()
print(f"\nTEs with breakout + production_hybrid: {len(both_h)}")
if len(both_h) >= 10:
    r, p = stats.pearsonr(both_h['breakout_score_15'], both_h['production_hybrid'])
    print(f"Pearson r = {r:+.3f}, p = {p:.4f}")

# Also check: how correlated is each with DC?
print(f"\nCorrelation with DC Score:")
for col, name in [('breakout_score_15', 'Breakout Score (15%)'),
                   ('rec_per_tpa', 'Rec/TPA'),
                   ('production_hybrid', 'Production Hybrid')]:
    valid = bt[col].notna() & bt['dc_score'].notna()
    if valid.sum() >= 10:
        r, p = stats.pearsonr(bt.loc[valid, col], bt.loc[valid, 'dc_score'])
        print(f"  {name:30s} vs DC: r={r:+.3f}, p={p:.4f}")

# ============================================================
# 2. MULTIVARIATE REGRESSION
# ============================================================
print(f"\n{'='*80}")
print("2. MULTIVARIATE REGRESSION: first_3yr_ppg ~ DC + Breakout + Production")
print(f"{'='*80}")

from numpy.linalg import lstsq

def ols_regression(X, y, var_names):
    """Simple OLS with p-values."""
    n, k = X.shape
    # Add intercept
    X_int = np.column_stack([np.ones(n), X])
    var_names = ['intercept'] + var_names

    beta, residuals, rank, sv = lstsq(X_int, y, rcond=None)
    y_hat = X_int @ beta
    resid = y - y_hat
    sse = np.sum(resid**2)
    sst = np.sum((y - y.mean())**2)
    r_squared = 1 - sse/sst
    adj_r2 = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

    # Standard errors
    mse = sse / (n - k - 1)
    var_beta = mse * np.linalg.inv(X_int.T @ X_int)
    se = np.sqrt(np.diag(var_beta))
    t_stats = beta / se
    # Two-tailed p-values
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=n-k-1)

    return beta, se, t_stats, p_values, r_squared, adj_r2, n

def print_regression(X, y, var_names, title):
    """Print regression results."""
    beta, se, t_stats, p_values, r2, adj_r2, n = ols_regression(X, y, var_names)
    print(f"\n{title}")
    print(f"  N = {n}, R² = {r2:.4f}, Adjusted R² = {adj_r2:.4f}")
    print(f"  {'Variable':<25s} {'Coef':>8} {'SE':>8} {'t-stat':>8} {'p-value':>8} {'Sig':>5}")
    print(f"  {'-'*65}")
    for i, name in enumerate(['intercept'] + var_names):
        sig = '***' if p_values[i] < 0.01 else ('**' if p_values[i] < 0.05 else ('*' if p_values[i] < 0.10 else ''))
        print(f"  {name:<25s} {beta[i]:>+8.4f} {se[i]:>8.4f} {t_stats[i]:>8.3f} {p_values[i]:>8.4f} {sig:>5}")
    return r2, adj_r2

# Standardize variables for comparable coefficients
def zscore(s):
    return (s - s.mean()) / s.std()

# --- Model A: DC only ---
valid_a = bt['dc_score'].notna() & bt['first_3yr_ppg'].notna()
d_a = bt[valid_a].copy()
X_a = zscore(d_a['dc_score']).values.reshape(-1, 1)
y_a = d_a['first_3yr_ppg'].values
r2_dc, adj_r2_dc = print_regression(X_a, y_a, ['dc_score (z)'], 'MODEL A: DC only')

# --- Model B: DC + Production (Rec/TPA) ---
valid_b = bt['dc_score'].notna() & bt['rec_per_tpa'].notna() & bt['first_3yr_ppg'].notna()
d_b = bt[valid_b].copy()
X_b = np.column_stack([zscore(d_b['dc_score']).values, zscore(d_b['rec_per_tpa']).values])
y_b = d_b['first_3yr_ppg'].values
r2_prod, adj_r2_prod = print_regression(X_b, y_b, ['dc_score (z)', 'rec_per_tpa (z)'],
    'MODEL B: DC + Production (Rec/TPA)')

# --- Model C: DC + Breakout (15%) ---
valid_c = bt['dc_score'].notna() & bt['breakout_score_15'].notna() & bt['first_3yr_ppg'].notna()
d_c = bt[valid_c].copy()
X_c = np.column_stack([zscore(d_c['dc_score']).values, zscore(d_c['breakout_score_15']).values])
y_c = d_c['first_3yr_ppg'].values
r2_bo, adj_r2_bo = print_regression(X_c, y_c, ['dc_score (z)', 'breakout_15 (z)'],
    'MODEL C: DC + Breakout (15% threshold)')

# --- Model D: DC + Production + Breakout (the key test) ---
valid_d = (bt['dc_score'].notna() & bt['rec_per_tpa'].notna() &
           bt['breakout_score_15'].notna() & bt['first_3yr_ppg'].notna())
d_d = bt[valid_d].copy()
X_d = np.column_stack([zscore(d_d['dc_score']).values,
                        zscore(d_d['rec_per_tpa']).values,
                        zscore(d_d['breakout_score_15']).values])
y_d = d_d['first_3yr_ppg'].values
r2_both, adj_r2_both = print_regression(X_d, y_d,
    ['dc_score (z)', 'rec_per_tpa (z)', 'breakout_15 (z)'],
    'MODEL D: DC + Production + Breakout (3-component)')

# --- Model E: DC + Hybrid Production + Breakout ---
valid_e = (bt['dc_score'].notna() & bt['production_hybrid'].notna() &
           bt['breakout_score_15'].notna() & bt['first_3yr_ppg'].notna())
d_e = bt[valid_e].copy()
X_e = np.column_stack([zscore(d_e['dc_score']).values,
                        zscore(d_e['production_hybrid']).values,
                        zscore(d_e['breakout_score_15']).values])
y_e = d_e['first_3yr_ppg'].values
r2_hybrid, adj_r2_hybrid = print_regression(X_e, y_e,
    ['dc_score (z)', 'production_hybrid (z)', 'breakout_15 (z)'],
    'MODEL E: DC + Hybrid Production + Breakout (3-component)')

# ============================================================
# 3. SUMMARY TABLE
# ============================================================
print(f"\n{'='*80}")
print("3. MODEL COMPARISON SUMMARY")
print(f"{'='*80}")

print(f"\n{'Model':<50s} {'N':>5} {'R²':>7} {'Adj R²':>7} {'Improvement':>12}")
print(f"{'-'*85}")
print(f"{'A: DC only':<50s} {len(d_a):>5} {r2_dc:>7.4f} {adj_r2_dc:>7.4f} {'baseline':>12}")
print(f"{'B: DC + Rec/TPA':<50s} {len(d_b):>5} {r2_prod:>7.4f} {adj_r2_prod:>7.4f} {f'+{r2_prod-r2_dc:.4f}':>12}")
print(f"{'C: DC + Breakout (15%)':<50s} {len(d_c):>5} {r2_bo:>7.4f} {adj_r2_bo:>7.4f} {f'+{r2_bo-r2_dc:.4f}':>12}")
print(f"{'D: DC + Rec/TPA + Breakout':<50s} {len(d_d):>5} {r2_both:>7.4f} {adj_r2_both:>7.4f} {f'+{r2_both-r2_dc:.4f}':>12}")
print(f"{'E: DC + Hybrid Prod + Breakout':<50s} {len(d_e):>5} {r2_hybrid:>7.4f} {adj_r2_hybrid:>7.4f} {f'+{r2_hybrid-r2_dc:.4f}':>12}")

# ============================================================
# 4. REPEAT FOR HIT OUTCOMES (logistic-style using OLS for simplicity)
# ============================================================
print(f"\n{'='*80}")
print("4. SAME TESTS FOR HIT24 AND HIT12 OUTCOMES")
print(f"{'='*80}")

for outcome in ['hit24', 'hit12']:
    print(f"\n--- Outcome: {outcome} ---")

    # DC only
    valid = bt['dc_score'].notna() & bt[outcome].notna()
    d_tmp = bt[valid]
    X = zscore(d_tmp['dc_score']).values.reshape(-1, 1)
    y = d_tmp[outcome].values
    r2_a, _ = print_regression(X, y, ['dc_score (z)'], f'  DC only ({outcome})')

    # DC + both (using Rec/TPA)
    valid = (bt['dc_score'].notna() & bt['rec_per_tpa'].notna() &
             bt['breakout_score_15'].notna() & bt[outcome].notna())
    d_tmp = bt[valid]
    X = np.column_stack([zscore(d_tmp['dc_score']).values,
                          zscore(d_tmp['rec_per_tpa']).values,
                          zscore(d_tmp['breakout_score_15']).values])
    y = d_tmp[outcome].values
    r2_d, _ = print_regression(X, y, ['dc_score (z)', 'rec_per_tpa (z)', 'breakout_15 (z)'],
        f'  DC + Rec/TPA + Breakout ({outcome})')

    print(f"\n  R² improvement: {r2_a:.4f} -> {r2_d:.4f} (+{r2_d - r2_a:.4f})")

# ============================================================
# 5. VARIANCE INFLATION FACTOR (VIF) — multicollinearity check
# ============================================================
print(f"\n{'='*80}")
print("5. MULTICOLLINEARITY CHECK (VIF)")
print(f"{'='*80}")

valid = (bt['dc_score'].notna() & bt['rec_per_tpa'].notna() &
         bt['breakout_score_15'].notna() & bt['first_3yr_ppg'].notna())
d_vif = bt[valid].copy()

vars_to_check = ['dc_score', 'rec_per_tpa', 'breakout_score_15']
print(f"\nCorrelation matrix (N={len(d_vif)}):")
corr_matrix = d_vif[vars_to_check].corr()
print(f"{'':25s} {'DC Score':>12} {'Rec/TPA':>12} {'Breakout15':>12}")
for v in vars_to_check:
    vals = [f"{corr_matrix.loc[v, v2]:+.3f}" for v2 in vars_to_check]
    label = {'dc_score': 'DC Score', 'rec_per_tpa': 'Rec/TPA', 'breakout_score_15': 'Breakout (15%)'}[v]
    print(f"  {label:<23s} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

# VIF calculation
print(f"\nVariance Inflation Factors:")
for i, v in enumerate(vars_to_check):
    others = [vars_to_check[j] for j in range(len(vars_to_check)) if j != i]
    X_other = d_vif[others].values
    X_other = np.column_stack([np.ones(len(X_other)), X_other])
    y_v = d_vif[v].values
    beta = lstsq(X_other, y_v, rcond=None)[0]
    y_hat = X_other @ beta
    ss_res = np.sum((y_v - y_hat)**2)
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2_v = 1 - ss_res / ss_tot
    vif = 1 / (1 - r2_v) if r2_v < 1 else float('inf')
    label = {'dc_score': 'DC Score', 'rec_per_tpa': 'Rec/TPA', 'breakout_score_15': 'Breakout (15%)'}[v]
    status = 'OK' if vif < 5 else ('CAUTION' if vif < 10 else 'HIGH')
    print(f"  {label:<23s}: VIF = {vif:.2f} [{status}]")

print(f"\n  VIF < 5 = no multicollinearity concern")
print(f"  VIF 5-10 = moderate, worth monitoring")
print(f"  VIF > 10 = severe, variables are redundant")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
