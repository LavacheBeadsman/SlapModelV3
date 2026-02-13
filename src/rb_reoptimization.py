"""
RB Model Reoptimization (3 Steps)
==================================
Step 1: Data inventory — coverage for all potential RB components
Step 2: Partial correlations after controlling for DC — what adds signal?
Step 3: Test 15+ weight configurations for best RB model

Uses priority-weighted objective: 40% first_3yr_ppg, 25% hit24, 20% hit12, 15% career_ppg
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('/home/user/SlapModelV3')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def dc_score(pick):
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))

def speed_score(forty, weight):
    """Barnwell Speed Score: (weight * 200) / (40_time)^4"""
    if pd.isna(forty) or pd.isna(weight) or forty <= 0:
        return np.nan
    return (weight * 200) / (forty ** 4)

def normalize_0_100(series):
    """Min-max normalize to 0-100"""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50, index=series.index)
    return (series - mn) / (mx - mn) * 100


# ============================================================================
# DATA LOADING
# ============================================================================

print("=" * 110)
print("RB MODEL REOPTIMIZATION")
print("Priority weights: first_3yr_ppg=40% | hit24=25% | hit12=20% | career_ppg=15%")
print("=" * 110)

# Core backtest
rb = pd.read_csv('data/rb_backtest_with_receiving.csv')
print(f"\nLoaded rb_backtest_with_receiving.csv: {len(rb)} RBs")

# Outcomes
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
out_rb = outcomes[outcomes['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']
].copy()

# Declare status
declare = pd.read_csv('data/rb_backtest_with_declare.csv')
declare_lookup = dict(zip(
    zip(declare['player_name'], declare['draft_year']),
    zip(declare['declare_status'], declare['draft_age'])
))

# PFF data
pff = pd.read_csv('data/rb_pff_corrected.csv')
pff_lookup = {}
for _, row in pff.iterrows():
    key = (row['player_name'], row['draft_year'])
    pff_lookup[key] = {
        'elusive_rating': row['elusive_rating'],
        'rush_yards': row['yards'],
        'rush_attempts': row['attempts'],
        'yco_attempt': row['yco_attempt'],
        'grades_run': row['grades_run'],
        'grades_offense': row['grades_offense'],
        'team_name': row.get('team_name', '')
    }

# ============================================================================
# BUILD MASTER RB TABLE
# ============================================================================

# Merge outcomes
rb = rb.merge(out_rb, on=['player_name', 'draft_year', 'pick'], how='left')

# Add declare status
rb['declare_status'] = rb.apply(
    lambda r: declare_lookup.get((r['player_name'], r['draft_year']), (None, None))[0], axis=1)
rb['draft_age'] = rb.apply(
    lambda r: declare_lookup.get((r['player_name'], r['draft_year']), (None, None))[1], axis=1)
rb['early_declare'] = (rb['declare_status'] == 'EARLY').astype(int)

# Add PFF metrics
for col in ['elusive_rating', 'rush_yards', 'rush_attempts', 'yco_attempt', 'grades_run', 'grades_offense']:
    rb[col] = rb.apply(
        lambda r: pff_lookup.get((r['player_name'], r['draft_year']), {}).get(col, np.nan), axis=1)

# Compute derived metrics
rb['s_dc'] = rb['pick'].apply(dc_score)

# Receiving production score (existing formula)
def rb_production_score(row):
    if pd.isna(row['rec_yards']) or pd.isna(row['team_pass_att']) or row['team_pass_att'] == 0:
        return np.nan
    age = row['age'] if pd.notna(row['age']) else 22
    age_weights = {19: 1.15, 20: 1.10, 21: 1.05, 22: 1.00, 23: 0.95}
    age_w = age_weights.get(int(age), 0.90 if age >= 24 else 1.15)
    raw = (row['rec_yards'] / row['team_pass_att']) * age_w * 100
    return min(99.9, raw / 1.75)

rb['s_rec_prod'] = rb.apply(rb_production_score, axis=1)

# Receiving yards per game (if we had games played — approximate from attempts)
rb['rec_ypg'] = rb['rec_yards']  # total receiving yards (no per-game available)

# RAS normalized to 0-100
rb['s_ras'] = rb['RAS'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)  # RAS is 0-10, scale to 0-100

# Speed Score (need 40 time and weight — check if we have them)
# We don't have raw 40/weight in this file, but RAS encodes athleticism
# Let's compute speed score proxy from PFF data if available

# Rushing metrics normalized
rb['rush_ypg'] = rb['rush_yards']  # total rush yards (proxy)
rb['rush_ypc'] = rb.apply(lambda r: r['rush_yards'] / r['rush_attempts']
                          if pd.notna(r['rush_yards']) and pd.notna(r['rush_attempts']) and r['rush_attempts'] > 0
                          else np.nan, axis=1)

# Rec yards per team pass attempt (raw, not age-weighted)
rb['rec_per_pass_att'] = rb.apply(
    lambda r: r['rec_yards'] / r['team_pass_att']
    if pd.notna(r['rec_yards']) and pd.notna(r['team_pass_att']) and r['team_pass_att'] > 0
    else np.nan, axis=1)

# Receptions per team pass attempt
rb['rec_share'] = rb.apply(
    lambda r: r['receptions'] / r['team_pass_att']
    if pd.notna(r['receptions']) and pd.notna(r['team_pass_att']) and r['team_pass_att'] > 0
    else np.nan, axis=1)

# Build teammate score for RBs (other RBs drafted from same school same year)
# Group by (college, draft_year), sum DC for all RBs from same school
school_year_dc = rb.groupby(['college', 'draft_year'])['s_dc'].apply(list).to_dict()
def rb_teammate_dc(row):
    key = (row['college'], row['draft_year'])
    dcs = school_year_dc.get(key, [])
    # Sum all DC from same school/year EXCEPT this player
    total = sum(dcs) - row['s_dc']
    return max(0, total)

rb['teammate_dc'] = rb.apply(rb_teammate_dc, axis=1)
rb['has_rb_teammate'] = (rb['teammate_dc'] > 0).astype(int)


# ============================================================================
# STEP 1: DATA INVENTORY
# ============================================================================

print(f"\n\n{'=' * 110}")
print("STEP 1: RB DATA INVENTORY")
print(f"Base: {len(rb)} RBs in backtest (2015-2024)")
print("=" * 110)

metrics = {
    'Draft Capital (pick)': 'pick',
    'DC Score': 's_dc',
    'College Rec Yards': 'rec_yards',
    'College Receptions': 'receptions',
    'Team Pass Attempts': 'team_pass_att',
    'Rec Yards / Team Pass Att': 'rec_per_pass_att',
    'Receiving Production Score': 's_rec_prod',
    'RAS (0-10)': 'RAS',
    'RAS Normalized (0-100)': 's_ras',
    'Declare Status': 'declare_status',
    'Early Declare (binary)': 'early_declare',
    'PFF Elusive Rating': 'elusive_rating',
    'PFF Rush Yards': 'rush_yards',
    'PFF Rush Attempts': 'rush_attempts',
    'PFF Yards/Carry (YPC)': 'rush_ypc',
    'PFF Yards After Contact/Att': 'yco_attempt',
    'PFF Run Grade': 'grades_run',
    'PFF Offense Grade': 'grades_offense',
    'RB Teammate DC': 'teammate_dc',
    'Has RB Teammate (binary)': 'has_rb_teammate',
}

print(f"\n  {'Metric':<40} {'Have':>6} {'Miss':>6} {'Coverage':>10} {'Flag':>8}")
print("  " + "-" * 75)

for name, col in metrics.items():
    if col in rb.columns:
        have = rb[col].notna().sum()
        # For declare_status, count non-null
        if col == 'declare_status':
            have = (rb[col].notna() & (rb[col] != '')).sum()
        miss = len(rb) - have
        pct = have / len(rb) * 100
        flag = "⚠ LOW" if pct < 80 else ""
        print(f"  {name:<40} {have:>6} {miss:>6} {pct:>9.1f}% {flag:>8}")
    else:
        print(f"  {name:<40}    N/A    N/A       N/A  MISSING")

# Outcome coverage
print(f"\n  {'OUTCOME DATA':<40} {'Have':>6} {'Miss':>6} {'Coverage':>10}")
print("  " + "-" * 65)
for out in ['hit24', 'hit12', 'best_ppg', 'first_3yr_ppg', 'career_ppg']:
    have = rb[out].notna().sum()
    pct = have / len(rb) * 100
    print(f"  {out:<40} {have:>6} {len(rb)-have:>6} {pct:>9.1f}%")

# Missing data breakdown
print(f"\n  MISSING DATA PATTERNS:")
print(f"  Rec yards missing: {rb['rec_yards'].isna().sum()} players")
no_rec = rb[rb['rec_yards'].isna()]
if len(no_rec) > 0:
    print(f"    Round distribution: {dict(no_rec['round'].value_counts().sort_index())}")
    print(f"    Draft year range: {int(no_rec['draft_year'].min())}-{int(no_rec['draft_year'].max())}")

print(f"\n  RAS missing: {rb['RAS'].isna().sum()} players")
no_ras = rb[rb['RAS'].isna()]
if len(no_ras) > 0:
    print(f"    Round distribution: {dict(no_ras['round'].value_counts().sort_index())}")
    r1_miss = no_ras[no_ras['round'] == 1]
    print(f"    Round 1 missing: {len(r1_miss)} — {list(r1_miss['player_name'].values)}")

print(f"\n  PFF data missing: {rb['elusive_rating'].isna().sum()} players")
no_pff = rb[rb['elusive_rating'].isna()]
if len(no_pff) > 0:
    print(f"    Round distribution: {dict(no_pff['round'].value_counts().sort_index())}")

print(f"\n  Declare status missing: {rb['declare_status'].isna().sum()} players")

# Not available data
print(f"\n  DATA NOT AVAILABLE IN BACKTEST FILES:")
print(f"  ❌ Speed Score (no raw 40-time + weight columns in backtest)")
print(f"  ❌ Breakout Age (no RB breakout age data collected)")
print(f"  ❌ Return production (no kick/punt return data for RBs)")
print(f"  ❌ % of team total yards / rushing yards / touchdowns (not in files)")
print(f"  ❌ PFF pass route grade / receiving grade (not in PFF extract)")
print(f"  ❌ Yards after contact total (only YCO/attempt available)")


# ============================================================================
# STEP 2: PARTIAL CORRELATIONS AFTER CONTROLLING FOR DC
# ============================================================================

print(f"\n\n{'=' * 110}")
print("STEP 2: PARTIAL CORRELATIONS AFTER CONTROLLING FOR DC")
print("Which RB metrics add signal beyond draft capital?")
print("=" * 110)

outcome_cols = ['first_3yr_ppg', 'hit24', 'hit12', 'career_ppg']
outcome_weights = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}

def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    valid = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    if len(valid) < 15:
        return np.nan, np.nan, len(valid)
    # Residualize x on z
    slope_xz, inter_xz, _, _, _ = stats.linregress(valid['z'], valid['x'])
    resid_x = valid['x'] - (slope_xz * valid['z'] + inter_xz)
    # Residualize y on z
    slope_yz, inter_yz, _, _, _ = stats.linregress(valid['z'], valid['y'])
    resid_y = valid['y'] - (slope_yz * valid['z'] + inter_yz)
    # Correlate residuals
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(valid)

# Test metrics
test_metrics = {
    'Rec Production Score (age-weighted)': 's_rec_prod',
    'Rec Yards (raw)': 'rec_yards',
    'Rec Yards / Team Pass Att': 'rec_per_pass_att',
    'Receptions / Team Pass Att': 'rec_share',
    'RAS (0-10)': 'RAS',
    'Early Declare (binary)': 'early_declare',
    'PFF Elusive Rating': 'elusive_rating',
    'PFF Rush Yards': 'rush_yards',
    'PFF YPC': 'rush_ypc',
    'PFF Yards After Contact/Att': 'yco_attempt',
    'PFF Run Grade': 'grades_run',
    'PFF Offense Grade': 'grades_offense',
    'RB Teammate DC': 'teammate_dc',
    'Has RB Teammate (binary)': 'has_rb_teammate',
}

# Eval sample — must have hit24 outcome
rb_eval = rb[rb['hit24'].notna() & rb['draft_year'].between(2015, 2024)].copy()
print(f"\nEval sample: {len(rb_eval)} RBs with hit24 data")
print(f"  hit24: {int(rb_eval['hit24'].sum())}/{len(rb_eval)} ({rb_eval['hit24'].mean()*100:.1f}%)")
print(f"  hit12: {int(rb_eval['hit12'].sum())}/{len(rb_eval)} ({rb_eval['hit12'].mean()*100:.1f}%)")
print(f"  first_3yr_ppg: {rb_eval['first_3yr_ppg'].notna().sum()} with data")
print(f"  career_ppg: {rb_eval['career_ppg'].notna().sum()} with data")

# DC baseline
print(f"\n  DC-only correlations:")
for out in outcome_cols:
    valid = rb_eval[['s_dc', out]].dropna()
    if len(valid) >= 10:
        r, p = stats.pearsonr(valid['s_dc'], valid[out])
        print(f"    DC vs {out:<20}: r={r:+.4f} (p={p:.4f}, N={len(valid)})")

print(f"\n\n  PARTIAL CORRELATIONS (controlling for DC):")
print(f"  {'Metric':<40}", end="")
for out in outcome_cols:
    print(f" {'r('+out+')':>16}", end="")
print(f" {'PRI-AVG':>10} {'N':>6} {'Signal?':>8}")
print("  " + "-" * 110)

metric_results = {}
for name, col in test_metrics.items():
    row_str = f"  {name:<40}"
    pri_sum = 0
    pri_total = 0
    n_min = 999

    for out in outcome_cols:
        r, p, n = partial_corr(rb_eval[col], rb_eval[out], rb_eval['s_dc'])
        if not np.isnan(r):
            sig = "*" if p < 0.05 else " "
            row_str += f" {r:>+.4f}{sig}(N={n:>3})"
            pri_sum += outcome_weights[out] * r
            pri_total += outcome_weights[out]
            n_min = min(n_min, n)
        else:
            row_str += f" {'N/A':>16}"

    pri_avg = pri_sum / pri_total if pri_total > 0 else np.nan
    signal = "YES" if (not np.isnan(pri_avg) and abs(pri_avg) > 0.05) else "weak" if not np.isnan(pri_avg) else ""
    row_str += f" {pri_avg:>+.4f} {n_min:>6} {signal:>8}" if not np.isnan(pri_avg) else f" {'N/A':>10} {'':>6} {'':>8}"
    print(row_str)

    metric_results[name] = {'pri_avg': pri_avg, 'col': col, 'n': n_min if n_min < 999 else 0}

# Rank by partial-r magnitude
print(f"\n  RANKED BY PRIORITY-WEIGHTED PARTIAL CORRELATION (descending |r|):")
ranked = sorted(metric_results.items(), key=lambda x: abs(x[1]['pri_avg']) if not np.isnan(x[1]['pri_avg']) else 0, reverse=True)
print(f"  {'Rank':>4} {'Metric':<40} {'PRI-AVG r':>12} {'N':>6} {'Coverage':>10}")
print("  " + "-" * 75)
for i, (name, info) in enumerate(ranked, 1):
    cov = info['n'] / len(rb_eval) * 100 if info['n'] > 0 else 0
    print(f"  {i:>4}. {name:<40} {info['pri_avg']:>+.4f}     {info['n']:>6}  {cov:>8.1f}%")


# ============================================================================
# STEP 3: WEIGHT CONFIGURATION TESTING
# ============================================================================

print(f"\n\n{'=' * 110}")
print("STEP 3: WEIGHT CONFIGURATION TESTING")
print("=" * 110)

# First, normalize components to 0-100 for fair comparison
# For metrics with missing data, we need to handle imputation

# Receiving Production — already 0-100
# Fill missing rec_prod with position average
avg_rec_prod = rb_eval['s_rec_prod'].mean()
rb_eval['s_rec_prod_filled'] = rb_eval['s_rec_prod'].fillna(avg_rec_prod)

# RAS — scale 0-10 to 0-100
avg_ras = rb_eval['s_ras'].mean()
rb_eval['s_ras_filled'] = rb_eval['s_ras'].fillna(avg_ras)

# Early declare — already 0 or 1, scale to 0-100
rb_eval['s_early_dec'] = rb_eval['early_declare'] * 100

# PFF Elusive Rating — normalize to 0-100
rb_eval['s_elusive'] = np.nan
valid_elusive = rb_eval['elusive_rating'].notna()
if valid_elusive.sum() > 10:
    rb_eval.loc[valid_elusive, 's_elusive'] = normalize_0_100(rb_eval.loc[valid_elusive, 'elusive_rating'])
avg_elusive = rb_eval['s_elusive'].mean()
rb_eval['s_elusive_filled'] = rb_eval['s_elusive'].fillna(avg_elusive)

# PFF Run Grade — normalize to 0-100
rb_eval['s_run_grade'] = np.nan
valid_rg = rb_eval['grades_run'].notna()
if valid_rg.sum() > 10:
    rb_eval.loc[valid_rg, 's_run_grade'] = normalize_0_100(rb_eval.loc[valid_rg, 'grades_run'])
avg_rg = rb_eval['s_run_grade'].mean()
rb_eval['s_run_grade_filled'] = rb_eval['s_run_grade'].fillna(avg_rg)

# PFF YCO/Att — normalize to 0-100
rb_eval['s_yco'] = np.nan
valid_yco = rb_eval['yco_attempt'].notna()
if valid_yco.sum() > 10:
    rb_eval.loc[valid_yco, 's_yco'] = normalize_0_100(rb_eval.loc[valid_yco, 'yco_attempt'])
avg_yco = rb_eval['s_yco'].mean()
rb_eval['s_yco_filled'] = rb_eval['s_yco'].fillna(avg_yco)

# Teammate — raw DC
rb_eval['s_teammate'] = (rb_eval['teammate_dc'] > 50).astype(int) * 100  # binary: had teammate above ~50 DC
# Also test continuous version
rb_eval['s_teammate_cont'] = rb_eval['teammate_dc'].clip(0, 100)

# Rec share (receptions / team pass att) normalized
rb_eval['s_rec_share'] = np.nan
valid_rs = rb_eval['rec_share'].notna()
if valid_rs.sum() > 10:
    rb_eval.loc[valid_rs, 's_rec_share'] = normalize_0_100(rb_eval.loc[valid_rs, 'rec_share'])
avg_rs = rb_eval['s_rec_share'].mean()
rb_eval['s_rec_share_filled'] = rb_eval['s_rec_share'].fillna(avg_rs)

# ──────────────────────────────────────────────────────────────────────
# PHASE A: Test different component SETS (which components to include?)
# ──────────────────────────────────────────────────────────────────────

print(f"\n--- PHASE A: Finding the best component set ---")
print(f"Testing which components add value beyond DC + Rec Production\n")

# Component sets to test (label, components dict: name -> col)
# Start with the current model and systematically add/swap components

def compute_slap(row, weights_dict):
    """Compute SLAP from dict of {col: weight}"""
    total = 0
    for col, w in weights_dict.items():
        total += row[col] * w
    return min(100, max(0, total))

def evaluate_config(df, label, comp_weights):
    """Evaluate a weight configuration against all outcomes."""
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
    }


# ──────────────────────────────────────────────────────────────────────
# Define all configurations to test
# ──────────────────────────────────────────────────────────────────────

configs = []

# Baseline: DC only
configs.append(('DC only (100/0/0/0)', {'s_dc': 1.00}))

# Current model: 50/35/15 (DC/RecProd/RAS)
configs.append(('CURRENT: 50/35/15 (DC/Rec/RAS)', {
    's_dc': 0.50, 's_rec_prod_filled': 0.35, 's_ras_filled': 0.15}))

# ── 2-component: DC + RecProd at various weights ──
configs.append(('DC+Rec: 65/35', {'s_dc': 0.65, 's_rec_prod_filled': 0.35}))
configs.append(('DC+Rec: 60/40', {'s_dc': 0.60, 's_rec_prod_filled': 0.40}))
configs.append(('DC+Rec: 55/45', {'s_dc': 0.55, 's_rec_prod_filled': 0.45}))
configs.append(('DC+Rec: 50/50', {'s_dc': 0.50, 's_rec_prod_filled': 0.50}))

# ── 3-component: DC + RecProd + RAS (current structure, vary weights) ──
configs.append(('DC/Rec/RAS: 55/35/10', {
    's_dc': 0.55, 's_rec_prod_filled': 0.35, 's_ras_filled': 0.10}))
configs.append(('DC/Rec/RAS: 60/30/10', {
    's_dc': 0.60, 's_rec_prod_filled': 0.30, 's_ras_filled': 0.10}))
configs.append(('DC/Rec/RAS: 65/25/10', {
    's_dc': 0.65, 's_rec_prod_filled': 0.25, 's_ras_filled': 0.10}))

# ── 3-component: DC + RecProd + EarlyDeclare ──
configs.append(('DC/Rec/ED: 60/30/10', {
    's_dc': 0.60, 's_rec_prod_filled': 0.30, 's_early_dec': 0.10}))
configs.append(('DC/Rec/ED: 55/35/10', {
    's_dc': 0.55, 's_rec_prod_filled': 0.35, 's_early_dec': 0.10}))
configs.append(('DC/Rec/ED: 50/35/15', {
    's_dc': 0.50, 's_rec_prod_filled': 0.35, 's_early_dec': 0.15}))

# ── 3-component: DC + RecProd + Elusive ──
configs.append(('DC/Rec/Elus: 55/35/10', {
    's_dc': 0.55, 's_rec_prod_filled': 0.35, 's_elusive_filled': 0.10}))
configs.append(('DC/Rec/Elus: 50/35/15', {
    's_dc': 0.50, 's_rec_prod_filled': 0.35, 's_elusive_filled': 0.15}))

# ── 3-component: DC + RecProd + YCO/Att ──
configs.append(('DC/Rec/YCO: 55/35/10', {
    's_dc': 0.55, 's_rec_prod_filled': 0.35, 's_yco_filled': 0.10}))

# ── 4-component: DC + RecProd + RAS + EarlyDeclare ──
configs.append(('DC/Rec/RAS/ED: 50/30/10/10', {
    's_dc': 0.50, 's_rec_prod_filled': 0.30, 's_ras_filled': 0.10, 's_early_dec': 0.10}))
configs.append(('DC/Rec/RAS/ED: 55/25/10/10', {
    's_dc': 0.55, 's_rec_prod_filled': 0.25, 's_ras_filled': 0.10, 's_early_dec': 0.10}))
configs.append(('DC/Rec/RAS/ED: 60/25/10/5', {
    's_dc': 0.60, 's_rec_prod_filled': 0.25, 's_ras_filled': 0.10, 's_early_dec': 0.05}))
configs.append(('DC/Rec/RAS/ED: 55/30/10/5', {
    's_dc': 0.55, 's_rec_prod_filled': 0.30, 's_ras_filled': 0.10, 's_early_dec': 0.05}))
configs.append(('DC/Rec/RAS/ED: 60/20/10/10', {
    's_dc': 0.60, 's_rec_prod_filled': 0.20, 's_ras_filled': 0.10, 's_early_dec': 0.10}))

# ── 4-component: DC + RecProd + Elusive + EarlyDeclare ──
configs.append(('DC/Rec/Elus/ED: 55/30/10/5', {
    's_dc': 0.55, 's_rec_prod_filled': 0.30, 's_elusive_filled': 0.10, 's_early_dec': 0.05}))
configs.append(('DC/Rec/Elus/ED: 50/30/10/10', {
    's_dc': 0.50, 's_rec_prod_filled': 0.30, 's_elusive_filled': 0.10, 's_early_dec': 0.10}))
configs.append(('DC/Rec/Elus/ED: 55/25/10/10', {
    's_dc': 0.55, 's_rec_prod_filled': 0.25, 's_elusive_filled': 0.10, 's_early_dec': 0.10}))

# ── 4-component: DC + RecProd + YCO + EarlyDeclare ──
configs.append(('DC/Rec/YCO/ED: 55/30/10/5', {
    's_dc': 0.55, 's_rec_prod_filled': 0.30, 's_yco_filled': 0.10, 's_early_dec': 0.05}))
configs.append(('DC/Rec/YCO/ED: 50/30/10/10', {
    's_dc': 0.50, 's_rec_prod_filled': 0.30, 's_yco_filled': 0.10, 's_early_dec': 0.10}))

# ── Higher DC configs (following WR finding that DC floor matters) ──
configs.append(('DC/Rec/RAS: 70/20/10', {
    's_dc': 0.70, 's_rec_prod_filled': 0.20, 's_ras_filled': 0.10}))
configs.append(('DC/Rec/ED: 70/20/10', {
    's_dc': 0.70, 's_rec_prod_filled': 0.20, 's_early_dec': 0.10}))
configs.append(('DC/Rec/RAS/ED: 65/20/10/5', {
    's_dc': 0.65, 's_rec_prod_filled': 0.20, 's_ras_filled': 0.10, 's_early_dec': 0.05}))
configs.append(('DC/Rec/Elus/ED: 65/20/10/5', {
    's_dc': 0.65, 's_rec_prod_filled': 0.20, 's_elusive_filled': 0.10, 's_early_dec': 0.05}))


# ──────────────────────────────────────────────────────────────────────
# Run all configurations
# ──────────────────────────────────────────────────────────────────────

all_results = []
for label, comp_weights in configs:
    r = evaluate_config(rb_eval, label, comp_weights)
    all_results.append(r)

# Sort by priority-weighted r
all_results_sorted = sorted(all_results, key=lambda x: x['pri_avg'] if not np.isnan(x['pri_avg']) else -999, reverse=True)

# Print results table
print(f"\n\n{'=' * 130}")
print(f"ALL CONFIGURATIONS RANKED BY PRIORITY-WEIGHTED r")
print(f"{'=' * 130}")

print(f"\n  {'Rank':>4} {'Config':<40} {'PRI-AVG r':>10} {'r(3yr)':>8} {'r(h24)':>8} {'r(h12)':>8} {'r(cpg)':>8}"
      f" {'Top10%h24':>10} {'Top10%PPG':>10} {'Dis10+':>7}")
print("  " + "-" * 120)

# Find baselines
dc_only_r = [r for r in all_results if 'DC only' in r['label']][0]['pri_avg']
current_r = [r for r in all_results if 'CURRENT' in r['label']][0]['pri_avg']

for i, r in enumerate(all_results_sorted, 1):
    o = r['outcomes']
    r3 = o.get('first_3yr_ppg', {}).get('r', np.nan)
    rh24 = o.get('hit24', {}).get('r', np.nan)
    rh12 = o.get('hit12', {}).get('r', np.nan)
    rcpg = o.get('career_ppg', {}).get('r', np.nan)
    ppg_s = f"{r['ppg_top10']:.2f}" if not np.isnan(r['ppg_top10']) else "N/A"

    marker = ""
    if 'CURRENT' in r['label']:
        marker = " ◄ CURRENT"
    elif 'DC only' in r['label']:
        marker = " ◄ BASELINE"
    elif i == 1:
        marker = " ◄ BEST"

    print(f"  {i:>4}. {r['label']:<40} {r['pri_avg']:>+.4f}  {r3:>+.4f} {rh24:>+.4f} {rh12:>+.4f} {rcpg:>+.4f}"
          f" {r['hit24_top10']:>9.1f}% {ppg_s:>10} {r['disagree_10']:>7}{marker}")


# ──────────────────────────────────────────────────────────────────────
# Detailed breakdown of top 10 configs
# ──────────────────────────────────────────────────────────────────────

print(f"\n\n{'=' * 110}")
print("TOP 10 CONFIGURATIONS — DETAILED BREAKDOWN")
print("=" * 110)

for i, r in enumerate(all_results_sorted[:10], 1):
    delta_dc = r['pri_avg'] - dc_only_r
    delta_cur = r['pri_avg'] - current_r
    pct_dc = delta_dc / dc_only_r * 100 if dc_only_r else 0
    pct_cur = delta_cur / current_r * 100 if current_r else 0

    print(f"\n  #{i}: {r['label']}")
    print(f"  Priority-weighted r: {r['pri_avg']:+.4f}")
    print(f"    vs DC-only:  {delta_dc:>+.4f} ({pct_dc:>+.1f}%)")
    print(f"    vs current:  {delta_cur:>+.4f} ({pct_cur:>+.1f}%)")
    print(f"  Top 10% ({r['n_top']} players): hit24={r['hit24_top10']:.1f}%, hit12={r['hit12_top10']:.1f}%, 3yr_ppg={r['ppg_top10']:.2f}" if not np.isnan(r['ppg_top10']) else f"  Top 10%: hit24={r['hit24_top10']:.1f}%")
    print(f"  Top 20: hit24={r['hit24_top20']:.1f}%")
    print(f"  Ranking disagreements vs DC (10+ spots): {r['disagree_10']}")


# ──────────────────────────────────────────────────────────────────────
# Tier analysis for top configs
# ──────────────────────────────────────────────────────────────────────

print(f"\n\n{'=' * 110}")
print("TIER ANALYSIS — Current vs Top Configs")
print("=" * 110)

# Pick: current, DC-only, and top 3 non-baseline configs
show_configs = []
show_configs.append(('DC only (100/0/0/0)', {'s_dc': 1.00}))
show_configs.append(('CURRENT: 50/35/15 (DC/Rec/RAS)', {
    's_dc': 0.50, 's_rec_prod_filled': 0.35, 's_ras_filled': 0.15}))

# Get top 3 that aren't baseline or current
added = 0
for r in all_results_sorted:
    if 'DC only' in r['label'] or 'CURRENT' in r['label']:
        continue
    # Find matching config
    for label, comp_w in configs:
        if label == r['label']:
            show_configs.append((label, comp_w))
            added += 1
            break
    if added >= 3:
        break

for label, comp_weights in show_configs:
    rb_eval_copy = rb_eval.copy()
    rb_eval_copy['slap'] = rb_eval_copy.apply(lambda r: compute_slap(r, comp_weights), axis=1)

    print(f"\n  ── {label} ──")

    # Tier by SLAP score
    bins = [(80, 100, 'Elite (80-100)'), (60, 80, 'Good (60-80)'),
            (40, 60, 'Average (40-60)'), (0, 40, 'Below Avg (0-40)')]

    print(f"  {'Tier':<20} {'N':>5} {'Hit24':>6} {'Rate':>8} {'Hit12':>6} {'Rate':>8} {'Avg PPG':>10}")
    print(f"  {'-' * 70}")

    for lo, hi, tier_name in bins:
        tier = rb_eval_copy[(rb_eval_copy['slap'] >= lo) & (rb_eval_copy['slap'] < hi)]
        if len(tier) == 0:
            continue
        h24 = int(tier['hit24'].sum())
        h12 = int(tier['hit12'].sum())
        r24 = h24 / len(tier) * 100
        r12 = h12 / len(tier) * 100
        tier_3yr = tier[tier['first_3yr_ppg'].notna()]
        ppg = tier_3yr['first_3yr_ppg'].mean() if len(tier_3yr) > 0 else np.nan
        ppg_s = f"{ppg:.2f}" if not np.isnan(ppg) else "N/A"
        print(f"  {tier_name:<20} {len(tier):>5} {h24:>6} {r24:>7.1f}% {h12:>6} {r12:>7.1f}% {ppg_s:>10}")


# ──────────────────────────────────────────────────────────────────────
# Check if RAS actually hurts or helps
# ──────────────────────────────────────────────────────────────────────

print(f"\n\n{'=' * 110}")
print("DOES RAS HELP OR HURT FOR RBs?")
print("Comparing DC+Rec with and without RAS at matched total weights")
print("=" * 110)

comparison_pairs = [
    ('DC+Rec: 65/35', {'s_dc': 0.65, 's_rec_prod_filled': 0.35},
     'DC/Rec/RAS: 55/35/10', {'s_dc': 0.55, 's_rec_prod_filled': 0.35, 's_ras_filled': 0.10}),
    ('DC+Rec: 60/40', {'s_dc': 0.60, 's_rec_prod_filled': 0.40},
     'DC/Rec/RAS: 50/35/15 (CURRENT)', {'s_dc': 0.50, 's_rec_prod_filled': 0.35, 's_ras_filled': 0.15}),
]

for l1, w1, l2, w2 in comparison_pairs:
    r1 = evaluate_config(rb_eval, l1, w1)
    r2 = evaluate_config(rb_eval, l2, w2)
    delta = r2['pri_avg'] - r1['pri_avg']
    print(f"\n  {l1} → PRI-AVG: {r1['pri_avg']:+.4f}, Top10% hit24: {r1['hit24_top10']:.1f}%")
    print(f"  {l2} → PRI-AVG: {r2['pri_avg']:+.4f}, Top10% hit24: {r2['hit24_top10']:.1f}%")
    print(f"  Adding RAS: {delta:>+.4f} ({'helps' if delta > 0 else 'HURTS' if delta < 0 else 'neutral'})")


# ──────────────────────────────────────────────────────────────────────
# Check if Early Declare helps
# ──────────────────────────────────────────────────────────────────────

print(f"\n\n{'=' * 110}")
print("DOES EARLY DECLARE HELP FOR RBs?")
print("=" * 110)

ed_pairs = [
    ('DC+Rec: 65/35', {'s_dc': 0.65, 's_rec_prod_filled': 0.35},
     'DC/Rec/ED: 60/30/10', {'s_dc': 0.60, 's_rec_prod_filled': 0.30, 's_early_dec': 0.10}),
    ('DC/Rec/RAS: 55/35/10', {'s_dc': 0.55, 's_rec_prod_filled': 0.35, 's_ras_filled': 0.10},
     'DC/Rec/RAS/ED: 55/25/10/10', {'s_dc': 0.55, 's_rec_prod_filled': 0.25, 's_ras_filled': 0.10, 's_early_dec': 0.10}),
]

for l1, w1, l2, w2 in ed_pairs:
    r1 = evaluate_config(rb_eval, l1, w1)
    r2 = evaluate_config(rb_eval, l2, w2)
    delta = r2['pri_avg'] - r1['pri_avg']
    print(f"\n  {l1} → PRI-AVG: {r1['pri_avg']:+.4f}, Top10% hit24: {r1['hit24_top10']:.1f}%")
    print(f"  {l2} → PRI-AVG: {r2['pri_avg']:+.4f}, Top10% hit24: {r2['hit24_top10']:.1f}%")
    print(f"  Adding ED: {delta:>+.4f} ({'helps' if delta > 0 else 'HURTS' if delta < 0 else 'neutral'})")


# ──────────────────────────────────────────────────────────────────────
# Performance summary
# ──────────────────────────────────────────────────────────────────────

print(f"\n\n{'=' * 110}")
print("SUMMARY: PERFORMANCE CURVE")
print("=" * 110)

# Text table
print(f"\n  {'Config':<42} {'PRI-AVG':>8} {'Δ DC':>8} {'Δ CUR':>8} {'Top10%h24':>10} {'Top10%PPG':>10} {'Dis10+':>8}")
print("  " + "-" * 100)

for r in all_results_sorted:
    d_dc = r['pri_avg'] - dc_only_r
    d_cur = r['pri_avg'] - current_r
    ppg_s = f"{r['ppg_top10']:.2f}" if not np.isnan(r['ppg_top10']) else "N/A"

    marker = ""
    if 'CURRENT' in r['label']:
        marker = " ◄"
    elif 'DC only' in r['label']:
        marker = " ◄"

    print(f"  {r['label']:<42} {r['pri_avg']:>+.4f} {d_dc:>+.4f} {d_cur:>+.4f}"
          f" {r['hit24_top10']:>9.1f}% {ppg_s:>10} {r['disagree_10']:>8}{marker}")


print(f"\n\n{'=' * 110}")
print("ANALYSIS COMPLETE")
print("=" * 110)
