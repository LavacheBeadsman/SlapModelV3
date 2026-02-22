"""
TE Model Weight Optimization
============================
Priority-weighted methodology matching WR/RB reoptimization:
  40% first_3yr_ppg | 25% hit24 | 20% hit12 | 15% career_ppg

Steps:
  1. Patch missing data (Goedert, Shaheen)
  2. Build all component scores (DC, Breakout 15%, Production hybrid, Early Declare, RAS)
  3. Test 2025 inclusion vs exclusion
  4. DC-only fallback for TEs missing both non-DC components
  5. Grid search 30+ weight configurations
  6. SLSQP optimization with multiple starting points
  7. Full validation (top-decile, tier analysis, disagreements)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('/home/user/SlapModelV3')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def dc_score(pick):
    """Gentler DC curve from CLAUDE.md."""
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

def normalize_0_100(series):
    """Min-max normalize to 0-100."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50, index=series.index)
    return (series - mn) / (mx - mn) * 100

def partial_corr(x, y, z):
    """Partial correlation of x,y controlling for z. Returns (r, p, n)."""
    valid = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    if len(valid) < 15:
        return np.nan, np.nan, len(valid)
    sx, ix, _, _, _ = stats.linregress(valid['z'], valid['x'])
    rx = valid['x'] - (sx * valid['z'] + ix)
    sy, iy, _, _, _ = stats.linregress(valid['z'], valid['y'])
    ry = valid['y'] - (sy * valid['z'] + iy)
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(valid)


# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 120)
print("TE MODEL WEIGHT OPTIMIZATION")
print("Priority weights: first_3yr_ppg=40% | hit24=25% | hit12=20% | career_ppg=15%")
print("=" * 120)

bt = pd.read_csv('data/te_backtest_master.csv')
print(f"\nLoaded te_backtest_master.csv: {len(bt)} TEs")


# ============================================================================
# STEP 0: PATCH GOEDERT AND SHAHEEN
# ============================================================================

print(f"\n{'='*100}")
print("STEP 0: PATCHING MISSING DATA — Goedert and Shaheen")
print(f"{'='*100}")

# Dallas Goedert — South Dakota State, 2018 draft
# Final season (2017): 72 receptions, 1111 receiving yards, 455 team pass attempts
# Source: gojacks.com official stats + school record book
goedert_mask = bt['player_name'] == 'Dallas Goedert'
if goedert_mask.any():
    bt.loc[goedert_mask, 'cfbd_receptions'] = 72
    bt.loc[goedert_mask, 'cfbd_rec_yards'] = 1111
    bt.loc[goedert_mask, 'cfbd_team_pass_att'] = 455
    bt.loc[goedert_mask, 'cfbd_matched'] = True
    print(f"  PATCHED Dallas Goedert: 72 rec, 1111 yds, 455 team pass att (2017 season)")
    print(f"    Source: gojacks.com (official SDSU athletics) + SDSU record book")
else:
    print(f"  WARNING: Dallas Goedert not found in backtest!")

# Adam Shaheen — Ashland University, 2017 draft
# Final season (2016): 57 receptions, 867 receiving yards, ~328 team pass attempts
# Source: goashlandeagles.com (QB Travis Tarnowski 328 att = effectively full team total)
shaheen_mask = bt['player_name'] == 'Adam Shaheen'
if shaheen_mask.any():
    bt.loc[shaheen_mask, 'cfbd_receptions'] = 57
    bt.loc[shaheen_mask, 'cfbd_rec_yards'] = 867
    bt.loc[shaheen_mask, 'cfbd_team_pass_att'] = 328
    bt.loc[shaheen_mask, 'cfbd_matched'] = True
    print(f"  PATCHED Adam Shaheen: 57 rec, 867 yds, 328 team pass att (2016 season)")
    print(f"    Source: goashlandeagles.com (Ashland official) + Wikipedia")
    print(f"    Note: 328 is QB Tarnowski's individual att; ~= team total (no backup QB attempts found)")
else:
    print(f"  WARNING: Adam Shaheen not found in backtest!")


# ============================================================================
# BUILD ALL COMPONENT SCORES
# ============================================================================

print(f"\n{'='*100}")
print("BUILDING COMPONENT SCORES")
print(f"{'='*100}")

# --- 1. Draft Capital Score ---
bt['s_dc'] = bt['pick'].apply(dc_score)
print(f"  DC Score: {bt['s_dc'].notna().sum()}/{len(bt)} have data")

# --- 2. Breakout Score at 15% threshold (from PFF multi-season) ---
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
bt['name_norm_opt'] = bt['player_name'].apply(norm_name)

# Build multi-season data and calculate breakout scores
THRESH = 15
bo_scores = {}
for _, te in bt.iterrows():
    matches = te_pff[(te_pff['name_norm'] == te['name_norm_opt']) & (te_pff['season'] < te['draft_year'])]
    if len(matches) == 0:
        continue
    peak_dom = matches['dominator_pct'].max()
    hit_seasons = []
    for _, pm in matches.sort_values('season').iterrows():
        season_age = te['draft_age'] - (te['draft_year'] - pm['season'])
        if pm['dominator_pct'] >= THRESH:
            hit_seasons.append(season_age)
    if hit_seasons:
        bo_age = hit_seasons[0]
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
    bo_scores[te['player_name']] = score

bt['s_breakout'] = bt['player_name'].map(bo_scores)
print(f"  Breakout Score (15%): {bt['s_breakout'].notna().sum()}/{len(bt)} have data")

# --- 3. Production Hybrid (Rec/TPA + PFF fallback) ---
bt['rec_per_tpa'] = np.where(
    (bt['cfbd_receptions'].notna()) & (bt['cfbd_team_pass_att'].notna()) & (bt['cfbd_team_pass_att'] > 0),
    bt['cfbd_receptions'] / bt['cfbd_team_pass_att'], np.nan)
bt['pff_rpg'] = np.where(
    (bt['pff_receptions'].notna()) & (bt['pff_player_game_count'].notna()) & (bt['pff_player_game_count'] > 0),
    bt['pff_receptions'] / bt['pff_player_game_count'], np.nan)

cfbd_vals = bt['rec_per_tpa'].dropna()
pff_vals = bt['pff_rpg'].dropna()
bt['cfbd_norm'] = np.where(bt['rec_per_tpa'].notna(),
    (bt['rec_per_tpa'] - cfbd_vals.min()) / (cfbd_vals.max() - cfbd_vals.min()) * 100, np.nan)
bt['pff_rpg_norm'] = np.where(bt['pff_rpg'].notna(),
    (bt['pff_rpg'] - pff_vals.min()) / (pff_vals.max() - pff_vals.min()) * 100, np.nan)
bt['s_production'] = np.where(bt['cfbd_norm'].notna(), bt['cfbd_norm'], bt['pff_rpg_norm'])
bt['prod_source'] = np.where(bt['cfbd_norm'].notna(), 'CFBD',
    np.where(bt['pff_rpg_norm'].notna(), 'PFF', 'NONE'))

n_cfbd = (bt['prod_source'] == 'CFBD').sum()
n_pff = (bt['prod_source'] == 'PFF').sum()
n_none = (bt['prod_source'] == 'NONE').sum()
print(f"  Production Hybrid: {bt['s_production'].notna().sum()}/{len(bt)} ({n_cfbd} CFBD + {n_pff} PFF fallback, {n_none} missing)")

# --- 4. Early Declare ---
bt['s_early_dec'] = bt['early_declare'] * 100
print(f"  Early Declare: {bt['early_declare'].notna().sum()}/{len(bt)} have data ({(bt['early_declare']==1).sum()} early)")

# --- 5. RAS ---
bt['s_ras'] = bt['te_ras'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)
print(f"  RAS (0-100): {bt['s_ras'].notna().sum()}/{len(bt)} have data")


# ============================================================================
# STEP 1: DC-ONLY FALLBACK FOR MISSING TEs
# ============================================================================

print(f"\n{'='*100}")
print("STEP 1: DC-ONLY FALLBACK CONFIRMATION")
print("TEs missing BOTH breakout and production get DC-only scores.")
print(f"{'='*100}")

has_bo = bt['s_breakout'].notna()
has_prod = bt['s_production'].notna()
has_both = has_bo & has_prod
has_either = has_bo | has_prod
has_neither = ~has_either

print(f"\n  All 3 components (DC + breakout + production): {has_both.sum()}/160")
print(f"  Has breakout only: {(has_bo & ~has_prod).sum()}")
print(f"  Has production only: {(~has_bo & has_prod).sum()}")
print(f"  Has NEITHER (DC-only fallback): {has_neither.sum()}")

# For imputation: fill missing breakout and production with position averages
# This ensures these TEs get a neutral score, not penalized
avg_bo = bt['s_breakout'].mean()
avg_prod = bt['s_production'].mean()
avg_ras = bt['s_ras'].mean()

print(f"\n  Imputation values (position averages):")
print(f"    Breakout avg: {avg_bo:.1f}")
print(f"    Production avg: {avg_prod:.1f}")
print(f"    RAS avg: {avg_ras:.1f}")
print(f"\n  This means DC-only-fallback TEs get SLAP ≈ DC×w_dc + avg_bo×w_bo + avg_prod×w_prod + ...")
print(f"  Their SLAP is driven almost entirely by DC (the averages just shift all TEs equally).")

bt['s_breakout_filled'] = bt['s_breakout'].fillna(avg_bo)
bt['s_production_filled'] = bt['s_production'].fillna(avg_prod)
bt['s_ras_filled'] = bt['s_ras'].fillna(avg_ras)

# List the DC-only fallback TEs
neither_tes = bt[has_neither].sort_values('pick')
print(f"\n  DC-only fallback TEs ({len(neither_tes)}):")
for _, r in neither_tes.iterrows():
    print(f"    {r['player_name']:<30s} | {int(r['draft_year'])} Rd {int(r['round'])} Pick {int(r['pick']):>3d} | "
          f"DC={r['s_dc']:.1f} | hit24={int(r['hit24'])} | {r['college']}")


# ============================================================================
# STEP 2: TEST 2025 INCLUSION vs EXCLUSION
# ============================================================================

print(f"\n\n{'='*100}")
print("STEP 2: 2025 DRAFT CLASS — INCLUDE OR EXCLUDE?")
print("Testing whether including 2025 TEs (max 1 NFL season) changes optimization results.")
print(f"{'='*100}")

outcome_cols = ['first_3yr_ppg', 'hit24', 'hit12', 'career_ppg']
outcome_weights = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}


def compute_slap(row, weights_dict):
    """Compute SLAP from dict of {col: weight}."""
    total = 0
    for col, w in weights_dict.items():
        total += row[col] * w
    return min(100, max(0, total))


def evaluate_config(df, label, comp_weights):
    """Evaluate a weight configuration against all 4 outcomes."""
    df = df.copy()
    df['slap'] = df.apply(lambda r: compute_slap(r, comp_weights), axis=1)

    results = {}
    pri_sum = 0
    pri_total = 0

    for out in outcome_cols:
        valid = df[['slap', out]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid['slap'], valid[out])
            results[out] = {'r': r, 'p': p, 'n': len(valid)}
            pri_sum += outcome_weights[out] * r
            pri_total += outcome_weights[out]

    pri_avg = pri_sum / pri_total if pri_total > 0 else np.nan

    # Top decile
    n_top = max(1, len(df) // 10)
    top = df.nlargest(n_top, 'slap')
    hit24_rate = top['hit24'].mean() * 100 if top['hit24'].notna().any() else np.nan
    hit12_rate = top['hit12'].mean() * 100 if top['hit12'].notna().any() else np.nan
    top_3yr = top[top['first_3yr_ppg'].notna()]
    ppg_avg = top_3yr['first_3yr_ppg'].mean() if len(top_3yr) > 0 else np.nan

    # Top 20
    top20 = df.nlargest(20, 'slap')
    hit24_top20 = top20['hit24'].mean() * 100

    # Ranking disagreements vs DC
    df['dc_rank'] = df['s_dc'].rank(ascending=False, method='min')
    df['slap_rank'] = df['slap'].rank(ascending=False, method='min')
    df['rank_diff'] = df['dc_rank'] - df['slap_rank']
    disagree_10 = int((df['rank_diff'].abs() >= 10).sum())

    # Boosted vs dinged analysis
    boosted = df[df['rank_diff'] > 5]
    dinged = df[df['rank_diff'] < -5]
    boost_ppg = boosted['first_3yr_ppg'].mean() if len(boosted[boosted['first_3yr_ppg'].notna()]) > 0 else np.nan
    ding_ppg = dinged['first_3yr_ppg'].mean() if len(dinged[dinged['first_3yr_ppg'].notna()]) > 0 else np.nan

    return {
        'label': label,
        'pri_avg': pri_avg,
        'outcomes': results,
        'hit24_top10': hit24_rate,
        'hit12_top10': hit12_rate,
        'ppg_top10': ppg_avg,
        'hit24_top20': hit24_top20,
        'disagree_10': disagree_10,
        'n_top': n_top,
        'n_total': len(df),
        'boost_ppg': boost_ppg,
        'ding_ppg': ding_ppg,
    }


# Build eval samples: with and without 2025
bt_all = bt[bt['hit24'].notna()].copy()
bt_no25 = bt[(bt['hit24'].notna()) & (bt['draft_year'] < 2025)].copy()

print(f"\n  WITH 2025: {len(bt_all)} TEs")
print(f"    Draft years: {int(bt_all['draft_year'].min())}-{int(bt_all['draft_year'].max())}")
print(f"    hit24: {int(bt_all['hit24'].sum())}/{len(bt_all)} ({bt_all['hit24'].mean():.1%})")
print(f"    first_3yr_ppg: {bt_all['first_3yr_ppg'].notna().sum()} with data")
print(f"    career_ppg: {bt_all['career_ppg'].notna().sum()} with data")

print(f"\n  WITHOUT 2025: {len(bt_no25)} TEs")
print(f"    Draft years: {int(bt_no25['draft_year'].min())}-{int(bt_no25['draft_year'].max())}")
print(f"    hit24: {int(bt_no25['hit24'].sum())}/{len(bt_no25)} ({bt_no25['hit24'].mean():.1%})")
print(f"    first_3yr_ppg: {bt_no25['first_3yr_ppg'].notna().sum()} with data")
print(f"    career_ppg: {bt_no25['career_ppg'].notna().sum()} with data")

# Test a few representative configs on both samples
test_configs = [
    ('DC only', {'s_dc': 1.00}),
    ('DC/BO: 70/30', {'s_dc': 0.70, 's_breakout_filled': 0.30}),
    ('DC/Prod: 70/30', {'s_dc': 0.70, 's_production_filled': 0.30}),
    ('DC/BO/Prod: 60/20/20', {'s_dc': 0.60, 's_breakout_filled': 0.20, 's_production_filled': 0.20}),
    ('DC/BO/Prod: 65/20/15', {'s_dc': 0.65, 's_breakout_filled': 0.20, 's_production_filled': 0.15}),
    ('DC/BO/Prod/ED: 60/15/15/10', {'s_dc': 0.60, 's_breakout_filled': 0.15, 's_production_filled': 0.15, 's_early_dec': 0.10}),
]

print(f"\n  {'Config':<40} {'PRI-AVG (w/ 2025)':>18} {'PRI-AVG (w/o 2025)':>18} {'Delta':>8}")
print(f"  {'-'*88}")

for label, weights in test_configs:
    r_all = evaluate_config(bt_all, label, weights)
    r_no25 = evaluate_config(bt_no25, label, weights)
    delta = r_all['pri_avg'] - r_no25['pri_avg'] if not np.isnan(r_all['pri_avg']) and not np.isnan(r_no25['pri_avg']) else np.nan
    delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
    print(f"  {label:<40} {r_all['pri_avg']:>+.4f} (N={r_all['n_total']:>3}) {r_no25['pri_avg']:>+.4f} (N={r_no25['n_total']:>3}) {delta_s:>8}")

# Check if relative RANKING of configs changes
print(f"\n  Checking if relative ranking of configs changes...")
all_results_w25 = [(l, evaluate_config(bt_all, l, w)) for l, w in test_configs]
all_results_wo25 = [(l, evaluate_config(bt_no25, l, w)) for l, w in test_configs]
rank_w25 = [l for l, r in sorted(all_results_w25, key=lambda x: x[1]['pri_avg'], reverse=True)]
rank_wo25 = [l for l, r in sorted(all_results_wo25, key=lambda x: x[1]['pri_avg'], reverse=True)]
print(f"  Ranking WITH 2025:    {' > '.join(rank_w25[:4])}")
print(f"  Ranking WITHOUT 2025: {' > '.join(rank_wo25[:4])}")
if rank_w25[:3] == rank_wo25[:3]:
    print(f"  RESULT: Top 3 configs SAME. 2025 inclusion does NOT change optimal weights.")
else:
    print(f"  RESULT: Rankings differ — excluding 2025 changes optimization results.")


# ============================================================================
# DECISION: Use whichever sample preserves more statistical power
# If rankings are same, use the larger sample (includes 2025)
# ============================================================================

# We'll use bt_no25 for optimization (conservative) and note if it matches bt_all
eval_df = bt_no25.copy()
eval_label = "2015-2024 (excluding 2025)"
print(f"\n  >>> USING {eval_label} for optimization ({len(eval_df)} TEs)")
print(f"      2025 class has max 1 NFL season — first_3yr_ppg is based on incomplete data.")
print(f"      Excluding them is the conservative choice.")


# ============================================================================
# STEP 3: PARTIAL CORRELATIONS (signal check for each component)
# ============================================================================

print(f"\n\n{'='*100}")
print("STEP 3: PARTIAL CORRELATIONS — What adds signal beyond DC?")
print(f"{'='*100}")

test_components = {
    'Breakout Score (15%)': 's_breakout',
    'Production Hybrid': 's_production',
    'Early Declare (binary)': 's_early_dec',
    'RAS (0-100)': 's_ras',
    'Breakout (filled)': 's_breakout_filled',
    'Production (filled)': 's_production_filled',
    'RAS (filled)': 's_ras_filled',
}

print(f"\n  {'Component':<30}", end="")
for out in outcome_cols:
    print(f" {'r('+out+')':>16}", end="")
print(f" {'PRI-AVG':>10} {'N':>6}")
print("  " + "-" * 105)

for name, col in test_components.items():
    row_str = f"  {name:<30}"
    pri_sum = 0
    pri_total = 0
    n_min = 999

    for out in outcome_cols:
        r, p, n = partial_corr(eval_df[col], eval_df[out], eval_df['s_dc'])
        if not np.isnan(r):
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else " "))
            row_str += f" {r:>+.3f}{sig}(N={n:>3})"
            pri_sum += outcome_weights[out] * r
            pri_total += outcome_weights[out]
            n_min = min(n_min, n)
        else:
            row_str += f" {'N/A':>16}"

    pri_avg = pri_sum / pri_total if pri_total > 0 else np.nan
    row_str += f" {pri_avg:>+.4f} {n_min:>6}" if not np.isnan(pri_avg) else f" {'N/A':>10} {'':>6}"
    print(row_str)


# ============================================================================
# STEP 4: GRID SEARCH — 30+ weight configurations
# ============================================================================

print(f"\n\n{'='*120}")
print("STEP 4: WEIGHT CONFIGURATION GRID SEARCH")
print(f"{'='*120}")

configs = []

# Baseline: DC only
configs.append(('DC only (100/0)', {'s_dc': 1.00}))

# ── 2-component: DC + Breakout ──
configs.append(('DC/BO: 80/20', {'s_dc': 0.80, 's_breakout_filled': 0.20}))
configs.append(('DC/BO: 75/25', {'s_dc': 0.75, 's_breakout_filled': 0.25}))
configs.append(('DC/BO: 70/30', {'s_dc': 0.70, 's_breakout_filled': 0.30}))
configs.append(('DC/BO: 65/35', {'s_dc': 0.65, 's_breakout_filled': 0.35}))

# ── 2-component: DC + Production ──
configs.append(('DC/Prod: 80/20', {'s_dc': 0.80, 's_production_filled': 0.20}))
configs.append(('DC/Prod: 75/25', {'s_dc': 0.75, 's_production_filled': 0.25}))
configs.append(('DC/Prod: 70/30', {'s_dc': 0.70, 's_production_filled': 0.30}))
configs.append(('DC/Prod: 65/35', {'s_dc': 0.65, 's_production_filled': 0.35}))

# ── 3-component: DC + Breakout + Production ──
configs.append(('DC/BO/Prod: 75/15/10', {'s_dc': 0.75, 's_breakout_filled': 0.15, 's_production_filled': 0.10}))
configs.append(('DC/BO/Prod: 70/20/10', {'s_dc': 0.70, 's_breakout_filled': 0.20, 's_production_filled': 0.10}))
configs.append(('DC/BO/Prod: 70/15/15', {'s_dc': 0.70, 's_breakout_filled': 0.15, 's_production_filled': 0.15}))
configs.append(('DC/BO/Prod: 65/20/15', {'s_dc': 0.65, 's_breakout_filled': 0.20, 's_production_filled': 0.15}))
configs.append(('DC/BO/Prod: 65/15/20', {'s_dc': 0.65, 's_breakout_filled': 0.15, 's_production_filled': 0.20}))
configs.append(('DC/BO/Prod: 60/25/15', {'s_dc': 0.60, 's_breakout_filled': 0.25, 's_production_filled': 0.15}))
configs.append(('DC/BO/Prod: 60/20/20', {'s_dc': 0.60, 's_breakout_filled': 0.20, 's_production_filled': 0.20}))
configs.append(('DC/BO/Prod: 55/25/20', {'s_dc': 0.55, 's_breakout_filled': 0.25, 's_production_filled': 0.20}))

# ── 3-component: DC + Breakout + Early Declare ──
configs.append(('DC/BO/ED: 70/20/10', {'s_dc': 0.70, 's_breakout_filled': 0.20, 's_early_dec': 0.10}))
configs.append(('DC/BO/ED: 75/15/10', {'s_dc': 0.75, 's_breakout_filled': 0.15, 's_early_dec': 0.10}))
configs.append(('DC/BO/ED: 70/25/5', {'s_dc': 0.70, 's_breakout_filled': 0.25, 's_early_dec': 0.05}))

# ── 3-component: DC + Production + Early Declare ──
configs.append(('DC/Prod/ED: 70/20/10', {'s_dc': 0.70, 's_production_filled': 0.20, 's_early_dec': 0.10}))
configs.append(('DC/Prod/ED: 65/25/10', {'s_dc': 0.65, 's_production_filled': 0.25, 's_early_dec': 0.10}))

# ── 3-component: DC + Breakout + RAS ──
configs.append(('DC/BO/RAS: 70/20/10', {'s_dc': 0.70, 's_breakout_filled': 0.20, 's_ras_filled': 0.10}))
configs.append(('DC/BO/RAS: 65/25/10', {'s_dc': 0.65, 's_breakout_filled': 0.25, 's_ras_filled': 0.10}))

# ── 4-component: DC + Breakout + Production + Early Declare ──
configs.append(('DC/BO/Prod/ED: 65/15/15/5', {'s_dc': 0.65, 's_breakout_filled': 0.15, 's_production_filled': 0.15, 's_early_dec': 0.05}))
configs.append(('DC/BO/Prod/ED: 60/20/15/5', {'s_dc': 0.60, 's_breakout_filled': 0.20, 's_production_filled': 0.15, 's_early_dec': 0.05}))
configs.append(('DC/BO/Prod/ED: 65/20/10/5', {'s_dc': 0.65, 's_breakout_filled': 0.20, 's_production_filled': 0.10, 's_early_dec': 0.05}))
configs.append(('DC/BO/Prod/ED: 60/15/15/10', {'s_dc': 0.60, 's_breakout_filled': 0.15, 's_production_filled': 0.15, 's_early_dec': 0.10}))
configs.append(('DC/BO/Prod/ED: 55/20/15/10', {'s_dc': 0.55, 's_breakout_filled': 0.20, 's_production_filled': 0.15, 's_early_dec': 0.10}))
configs.append(('DC/BO/Prod/ED: 70/15/10/5', {'s_dc': 0.70, 's_breakout_filled': 0.15, 's_production_filled': 0.10, 's_early_dec': 0.05}))

# ── 4-component: DC + Breakout + Production + RAS ──
configs.append(('DC/BO/Prod/RAS: 60/20/15/5', {'s_dc': 0.60, 's_breakout_filled': 0.20, 's_production_filled': 0.15, 's_ras_filled': 0.05}))
configs.append(('DC/BO/Prod/RAS: 65/15/15/5', {'s_dc': 0.65, 's_breakout_filled': 0.15, 's_production_filled': 0.15, 's_ras_filled': 0.05}))
configs.append(('DC/BO/Prod/RAS: 65/20/10/5', {'s_dc': 0.65, 's_breakout_filled': 0.20, 's_production_filled': 0.10, 's_ras_filled': 0.05}))

# ── 5-component: DC + Breakout + Production + Early Declare + RAS ──
configs.append(('DC/BO/Prod/ED/RAS: 60/15/15/5/5', {'s_dc': 0.60, 's_breakout_filled': 0.15, 's_production_filled': 0.15, 's_early_dec': 0.05, 's_ras_filled': 0.05}))
configs.append(('DC/BO/Prod/ED/RAS: 65/15/10/5/5', {'s_dc': 0.65, 's_breakout_filled': 0.15, 's_production_filled': 0.10, 's_early_dec': 0.05, 's_ras_filled': 0.05}))

print(f"\nTesting {len(configs)} weight configurations...\n")

# Run all configs
all_results = []
for label, comp_weights in configs:
    r = evaluate_config(eval_df, label, comp_weights)
    all_results.append(r)

# Sort by PRI-AVG
all_results_sorted = sorted(all_results, key=lambda x: x['pri_avg'] if not np.isnan(x['pri_avg']) else -999, reverse=True)

# Find baseline
dc_only_r = [r for r in all_results if 'DC only' in r['label']][0]['pri_avg']

# Print results
print(f"{'Rank':>4} {'Config':<43} {'PRI-AVG':>8} {'Δ DC':>7} {'r(3yr)':>8} {'r(h24)':>8} {'r(h12)':>8} {'r(cpg)':>8}"
      f" {'Top10%h24':>10} {'Top10%PPG':>10} {'Dis10+':>7}")
print("-" * 135)

for i, r in enumerate(all_results_sorted, 1):
    o = r['outcomes']
    r3 = o.get('first_3yr_ppg', {}).get('r', np.nan)
    rh24 = o.get('hit24', {}).get('r', np.nan)
    rh12 = o.get('hit12', {}).get('r', np.nan)
    rcpg = o.get('career_ppg', {}).get('r', np.nan)
    ppg_s = f"{r['ppg_top10']:.2f}" if not np.isnan(r['ppg_top10']) else "N/A"
    delta = r['pri_avg'] - dc_only_r

    marker = ""
    if 'DC only' in r['label']:
        marker = " BASELINE"
    elif i == 1:
        marker = " <<< BEST"

    print(f"{i:>4}. {r['label']:<43} {r['pri_avg']:>+.4f} {delta:>+.4f} {r3:>+.4f} {rh24:>+.4f} {rh12:>+.4f} {rcpg:>+.4f}"
          f" {r['hit24_top10']:>9.1f}% {ppg_s:>10} {r['disagree_10']:>7}{marker}")


# ============================================================================
# STEP 5: SCIPY OPTIMIZATION (SLSQP with multiple starting points)
# ============================================================================

print(f"\n\n{'='*120}")
print("STEP 5: SCIPY OPTIMIZATION (SLSQP)")
print("Testing 2, 3, and 4-component models with multiple starting points.")
print(f"{'='*120}")


def objective_fn(weights, component_cols, df):
    """Minimize negative PRI-AVG."""
    df = df.copy()
    df['slap'] = 0
    for col, w in zip(component_cols, weights):
        df['slap'] += df[col] * w
    df['slap'] = df['slap'].clip(0, 100)

    pri_sum = 0
    pri_total = 0
    for out in outcome_cols:
        valid = df[['slap', out]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid['slap'], valid[out])
            pri_sum += outcome_weights[out] * r
            pri_total += outcome_weights[out]

    return -(pri_sum / pri_total) if pri_total > 0 else 0


def optimize_model(component_cols, bounds, starting_points, df, label):
    """Run SLSQP with multiple starting points."""
    constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]
    best_result = None
    best_pri = -999

    for x0 in starting_points:
        try:
            res = minimize(
                objective_fn, x0=x0, args=(component_cols, df),
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            if res.success and -res.fun > best_pri:
                best_pri = -res.fun
                best_result = res
        except Exception:
            continue

    if best_result is not None:
        weights = best_result.x
        # Round to nearest 5%
        weights_rounded = [round(w * 20) / 20 for w in weights]
        # Ensure they sum to 1.0
        diff = 1.0 - sum(weights_rounded)
        weights_rounded[0] += diff

        print(f"\n  {label}:")
        print(f"    Optimal (raw):     {' / '.join(f'{w:.3f}' for w in weights)}")
        print(f"    Optimal (rounded): {' / '.join(f'{w:.0%}' for w in weights_rounded)}")
        print(f"    PRI-AVG r (raw):     {best_pri:+.4f}")

        # Evaluate rounded
        comp_weights_rounded = dict(zip(component_cols, weights_rounded))
        r_eval = evaluate_config(df, f"{label} (rounded)", comp_weights_rounded)
        print(f"    PRI-AVG r (rounded): {r_eval['pri_avg']:+.4f}")
        print(f"    Top 10% hit24: {r_eval['hit24_top10']:.1f}%")
        print(f"    Top 10% PPG: {r_eval['ppg_top10']:.2f}" if not np.isnan(r_eval['ppg_top10']) else "    Top 10% PPG: N/A")
        print(f"    Disagree 10+: {r_eval['disagree_10']}")
        return weights, weights_rounded, best_pri, r_eval
    else:
        print(f"\n  {label}: Optimization FAILED")
        return None, None, None, None


# --- 2-component: DC + Breakout ---
opt_2bo = optimize_model(
    ['s_dc', 's_breakout_filled'],
    [(0.50, 0.90), (0.10, 0.50)],
    [[0.70, 0.30], [0.75, 0.25], [0.65, 0.35], [0.80, 0.20]],
    eval_df, "2-comp: DC + Breakout"
)

# --- 2-component: DC + Production ---
opt_2prod = optimize_model(
    ['s_dc', 's_production_filled'],
    [(0.50, 0.90), (0.10, 0.50)],
    [[0.70, 0.30], [0.75, 0.25], [0.65, 0.35], [0.80, 0.20]],
    eval_df, "2-comp: DC + Production"
)

# --- 3-component: DC + Breakout + Production ---
opt_3 = optimize_model(
    ['s_dc', 's_breakout_filled', 's_production_filled'],
    [(0.45, 0.85), (0.05, 0.40), (0.05, 0.40)],
    [
        [0.65, 0.20, 0.15], [0.70, 0.15, 0.15], [0.60, 0.25, 0.15],
        [0.75, 0.15, 0.10], [0.55, 0.25, 0.20], [0.70, 0.20, 0.10],
    ],
    eval_df, "3-comp: DC + Breakout + Production"
)

# --- 3-component: DC + Breakout + Early Declare ---
opt_3ed = optimize_model(
    ['s_dc', 's_breakout_filled', 's_early_dec'],
    [(0.50, 0.85), (0.05, 0.40), (0.00, 0.20)],
    [
        [0.70, 0.20, 0.10], [0.75, 0.15, 0.10], [0.65, 0.25, 0.10],
        [0.70, 0.25, 0.05], [0.80, 0.15, 0.05],
    ],
    eval_df, "3-comp: DC + Breakout + Early Declare"
)

# --- 4-component: DC + Breakout + Production + Early Declare ---
opt_4 = optimize_model(
    ['s_dc', 's_breakout_filled', 's_production_filled', 's_early_dec'],
    [(0.45, 0.80), (0.05, 0.35), (0.05, 0.30), (0.00, 0.15)],
    [
        [0.65, 0.15, 0.15, 0.05], [0.60, 0.20, 0.15, 0.05],
        [0.70, 0.15, 0.10, 0.05], [0.55, 0.20, 0.15, 0.10],
        [0.60, 0.15, 0.15, 0.10], [0.65, 0.20, 0.10, 0.05],
    ],
    eval_df, "4-comp: DC + Breakout + Production + ED"
)

# --- 4-component: DC + Breakout + Production + RAS ---
opt_4ras = optimize_model(
    ['s_dc', 's_breakout_filled', 's_production_filled', 's_ras_filled'],
    [(0.45, 0.80), (0.05, 0.35), (0.05, 0.30), (0.00, 0.15)],
    [
        [0.65, 0.15, 0.15, 0.05], [0.60, 0.20, 0.15, 0.05],
        [0.70, 0.15, 0.10, 0.05], [0.65, 0.20, 0.10, 0.05],
    ],
    eval_df, "4-comp: DC + Breakout + Production + RAS"
)

# --- 5-component: DC + Breakout + Production + Early Declare + RAS ---
opt_5 = optimize_model(
    ['s_dc', 's_breakout_filled', 's_production_filled', 's_early_dec', 's_ras_filled'],
    [(0.45, 0.80), (0.05, 0.30), (0.05, 0.25), (0.00, 0.15), (0.00, 0.10)],
    [
        [0.60, 0.15, 0.15, 0.05, 0.05], [0.65, 0.15, 0.10, 0.05, 0.05],
        [0.55, 0.20, 0.15, 0.05, 0.05], [0.65, 0.15, 0.15, 0.05, 0.00],
    ],
    eval_df, "5-comp: DC + BO + Prod + ED + RAS"
)


# ============================================================================
# STEP 6: TOP 5 FULL VALIDATION
# ============================================================================

print(f"\n\n{'='*120}")
print("STEP 6: FULL VALIDATION — TOP 5 GRID SEARCH CONFIGS + OPTIMIZER WINNERS")
print(f"{'='*120}")

# Collect the top configs from grid search (top 5) + all optimizer results
final_configs = []
for i, r in enumerate(all_results_sorted[:5]):
    for label, weights in configs:
        if label == r['label']:
            final_configs.append((label, weights))
            break

# Add optimizer rounded results
optimizer_results = [
    ('SLSQP 2c: DC+BO', opt_2bo),
    ('SLSQP 2c: DC+Prod', opt_2prod),
    ('SLSQP 3c: DC+BO+Prod', opt_3),
    ('SLSQP 3c: DC+BO+ED', opt_3ed),
    ('SLSQP 4c: DC+BO+Prod+ED', opt_4),
    ('SLSQP 4c: DC+BO+Prod+RAS', opt_4ras),
    ('SLSQP 5c: DC+BO+Prod+ED+RAS', opt_5),
]

comp_cols_map = {
    'SLSQP 2c: DC+BO': ['s_dc', 's_breakout_filled'],
    'SLSQP 2c: DC+Prod': ['s_dc', 's_production_filled'],
    'SLSQP 3c: DC+BO+Prod': ['s_dc', 's_breakout_filled', 's_production_filled'],
    'SLSQP 3c: DC+BO+ED': ['s_dc', 's_breakout_filled', 's_early_dec'],
    'SLSQP 4c: DC+BO+Prod+ED': ['s_dc', 's_breakout_filled', 's_production_filled', 's_early_dec'],
    'SLSQP 4c: DC+BO+Prod+RAS': ['s_dc', 's_breakout_filled', 's_production_filled', 's_ras_filled'],
    'SLSQP 5c: DC+BO+Prod+ED+RAS': ['s_dc', 's_breakout_filled', 's_production_filled', 's_early_dec', 's_ras_filled'],
}

for name, opt_result in optimizer_results:
    if opt_result[1] is not None:  # weights_rounded
        comp_weights = dict(zip(comp_cols_map[name], opt_result[1]))
        pct = '/'.join(f"{w:.0%}" for w in opt_result[1])
        final_configs.append((f"{name} ({pct})", comp_weights))

# Run tier analysis on all final configs
for label, comp_weights in final_configs:
    df_eval = eval_df.copy()
    df_eval['slap'] = df_eval.apply(lambda r: compute_slap(r, comp_weights), axis=1)

    result = evaluate_config(eval_df, label, comp_weights)

    print(f"\n  {'─'*80}")
    print(f"  {label}")
    print(f"  PRI-AVG: {result['pri_avg']:+.4f} (Δ DC: {result['pri_avg'] - dc_only_r:+.4f})")

    o = result['outcomes']
    for out in outcome_cols:
        if out in o:
            r_val = o[out]['r']
            p_val = o[out]['p']
            sig = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.10 else ''))
            print(f"    {out:20s}: r={r_val:+.4f} (p={p_val:.4f}) {sig}")

    print(f"  Top 10% ({result['n_top']}): hit24={result['hit24_top10']:.1f}%, "
          f"PPG={'%.2f' % result['ppg_top10'] if not np.isnan(result['ppg_top10']) else 'N/A'}")
    print(f"  Top 20: hit24={result['hit24_top20']:.1f}%")
    print(f"  Disagree 10+: {result['disagree_10']}")
    if not np.isnan(result['boost_ppg']) and not np.isnan(result['ding_ppg']):
        print(f"  Boosted avg PPG: {result['boost_ppg']:.2f} vs Dinged avg PPG: {result['ding_ppg']:.2f} "
              f"(diff: {result['boost_ppg'] - result['ding_ppg']:+.2f})")

    # Tier analysis
    bins = [(80, 100, 'Elite (80-100)'), (60, 80, 'Good (60-80)'),
            (40, 60, 'Average (40-60)'), (0, 40, 'Below Avg (0-40)')]
    print(f"\n  {'Tier':<20} {'N':>5} {'Hit24':>6} {'Rate':>8} {'Hit12':>6} {'Rate':>8} {'Avg PPG':>10}")
    print(f"  {'-'*70}")
    for lo, hi, tier_name in bins:
        tier = df_eval[(df_eval['slap'] >= lo) & (df_eval['slap'] < hi)]
        if len(tier) == 0:
            continue
        h24 = int(tier['hit24'].sum())
        h12 = int(tier['hit12'].sum())
        r24 = h24 / len(tier) * 100
        r12 = h12 / len(tier) * 100
        tier_ppg = tier[tier['first_3yr_ppg'].notna()]
        ppg = tier_ppg['first_3yr_ppg'].mean() if len(tier_ppg) > 0 else np.nan
        ppg_s = f"{ppg:.2f}" if not np.isnan(ppg) else "N/A"
        print(f"  {tier_name:<20} {len(tier):>5} {h24:>6} {r24:>7.1f}% {h12:>6} {r12:>7.1f}% {ppg_s:>10}")


# ============================================================================
# STEP 7: FINAL COMPARISON TABLE
# ============================================================================

print(f"\n\n{'='*120}")
print("STEP 7: FINAL COMPARISON — ALL CONFIGS RANKED")
print(f"{'='*120}")

# Collect all evaluated results
final_results = []
for label, comp_weights in final_configs:
    r = evaluate_config(eval_df, label, comp_weights)
    final_results.append(r)

# Also add DC-only baseline
r_dc = evaluate_config(eval_df, 'DC only (baseline)', {'s_dc': 1.00})
final_results.append(r_dc)

# Sort
final_sorted = sorted(final_results, key=lambda x: x['pri_avg'] if not np.isnan(x['pri_avg']) else -999, reverse=True)

print(f"\n{'Rank':>4} {'Config':<52} {'PRI-AVG':>8} {'Δ DC':>7} {'Top10%h24':>10} {'Top10%PPG':>10} {'Dis10+':>7} {'Boost-Ding':>12}")
print("-" * 120)

for i, r in enumerate(final_sorted, 1):
    delta = r['pri_avg'] - dc_only_r
    ppg_s = f"{r['ppg_top10']:.2f}" if not np.isnan(r['ppg_top10']) else "N/A"
    bd = f"{r['boost_ppg'] - r['ding_ppg']:+.2f}" if not np.isnan(r.get('boost_ppg', np.nan)) and not np.isnan(r.get('ding_ppg', np.nan)) else "N/A"

    marker = ""
    if 'baseline' in r['label'].lower():
        marker = " BASELINE"
    elif i == 1:
        marker = " <<< BEST"

    print(f"{i:>4}. {r['label']:<52} {r['pri_avg']:>+.4f} {delta:>+.4f} {r['hit24_top10']:>9.1f}% {ppg_s:>10} {r['disagree_10']:>7} {bd:>12}{marker}")


# ============================================================================
# STEP 8: SAVE PATCHED DATA
# ============================================================================

print(f"\n\n{'='*100}")
print("STEP 8: SAVING PATCHED BACKTEST DATA")
print(f"{'='*100}")

# Save patched te_backtest_master.csv (with Goedert + Shaheen data)
# Drop temporary columns we added
save_cols = [c for c in bt.columns if c not in [
    'name_norm_opt', 's_dc', 's_breakout', 's_production', 's_early_dec', 's_ras',
    'rec_per_tpa', 'pff_rpg', 'cfbd_norm', 'pff_rpg_norm', 'prod_source',
    's_breakout_filled', 's_production_filled', 's_ras_filled'
]]
bt[save_cols].to_csv('data/te_backtest_master.csv', index=False)
print(f"  Saved patched te_backtest_master.csv ({len(bt)} rows)")
print(f"  Goedert and Shaheen now have CFBD receiving data.")


print(f"\n{'='*120}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*120}")
print(f"\nReview the results above and choose the best weight configuration.")
print(f"Key metrics to compare:")
print(f"  1. PRI-AVG r (higher = better overall correlation)")
print(f"  2. Top 10% hit24 rate (practical pick accuracy)")
print(f"  3. Disagree 10+ (content value — how many rankings differ from pure DC)")
print(f"  4. Boost-Ding gap (do the model's disagreements turn out to be correct?)")
