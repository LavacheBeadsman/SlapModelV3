"""
RB Reoptimization — Deep dive on Size and Early Declare
========================================================
Test 1: Size metrics (weight, height, BMI) — coverage, partial correlations, within-round analysis
Test 2: Early declare deep dive — why does it show negative signal?
Test 3: Head-to-head if either shows promise
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('/home/user/SlapModelV3')

# ============================================================================
# HELPERS
# ============================================================================

def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

def normalize_0_100(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50, index=series.index)
    return (series - mn) / (mx - mn) * 100

def rb_production_score(row):
    if pd.isna(row['rec_yards']) or pd.isna(row['team_pass_att']) or row['team_pass_att'] == 0:
        return np.nan
    age = row['age'] if pd.notna(row['age']) else 22
    age_weights = {19: 1.15, 20: 1.10, 21: 1.05, 22: 1.00, 23: 0.95}
    age_w = age_weights.get(int(age), 0.90 if age >= 24 else 1.15)
    raw = (row['rec_yards'] / row['team_pass_att']) * age_w * 100
    return min(99.9, raw / 1.75)

def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z (single control)."""
    valid = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    if len(valid) < 15:
        return np.nan, np.nan, len(valid)
    slope_xz, inter_xz, _, _, _ = stats.linregress(valid['z'], valid['x'])
    resid_x = valid['x'] - (slope_xz * valid['z'] + inter_xz)
    slope_yz, inter_yz, _, _, _ = stats.linregress(valid['z'], valid['y'])
    resid_y = valid['y'] - (slope_yz * valid['z'] + inter_yz)
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(valid)

def partial_corr_2(x, y, z1, z2):
    """Partial correlation of x and y controlling for z1 AND z2."""
    df = pd.DataFrame({'x': x, 'y': y, 'z1': z1, 'z2': z2}).dropna()
    if len(df) < 20:
        return np.nan, np.nan, len(df)
    # OLS residualize x on z1+z2
    from numpy.linalg import lstsq
    Z = np.column_stack([df['z1'].values, df['z2'].values, np.ones(len(df))])
    coef_x, _, _, _ = lstsq(Z, df['x'].values, rcond=None)
    resid_x = df['x'].values - Z @ coef_x
    coef_y, _, _, _ = lstsq(Z, df['y'].values, rcond=None)
    resid_y = df['y'].values - Z @ coef_y
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(df)

outcome_cols = ['first_3yr_ppg', 'hit24', 'hit12', 'career_ppg']
outcome_weights = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}

# ============================================================================
# DATA LOADING
# ============================================================================

rb = pd.read_csv('data/rb_backtest_with_receiving.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
out_rb = outcomes[outcomes['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
rb = rb.merge(out_rb, on=['player_name', 'draft_year', 'pick'], how='left')
rb['s_dc'] = rb['pick'].apply(dc_score)
rb['s_rec_prod'] = rb.apply(rb_production_score, axis=1)
avg_rec = rb['s_rec_prod'].mean()
rb['s_rec_prod_filled'] = rb['s_rec_prod'].fillna(avg_rec)

# Declare status
declare = pd.read_csv('data/rb_backtest_with_declare.csv')
declare_map = {}
for _, row in declare.iterrows():
    declare_map[(row['player_name'], row['draft_year'])] = row['declare_status']
rb['declare_status'] = rb.apply(
    lambda r: declare_map.get((r['player_name'], r['draft_year']), np.nan), axis=1)
rb['early_declare'] = (rb['declare_status'] == 'EARLY').astype(int)

# Combine data for size
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine_rb = combine[combine['pos'] == 'RB'].copy()

def normalize_name(name):
    if pd.isna(name):
        return ''
    return name.strip().lower().replace('.', '').replace("'", '').replace('-', ' ')

combine_rb['name_norm'] = combine_rb['player_name'].apply(normalize_name)
rb['name_norm'] = rb['player_name'].apply(normalize_name)

# Build lookup keyed on name_norm + draft_year
combine_lookup = {}
for _, row in combine_rb.iterrows():
    dy = row.get('draft_year')
    if pd.isna(dy):
        dy = row.get('season')
    if pd.isna(dy):
        continue
    key = (row['name_norm'], int(dy))
    combine_lookup[key] = {
        'weight': row['wt'],
        'height': row['ht'],
        'forty': row['forty'],
    }

rb['weight'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb['height'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('height', np.nan), axis=1)
rb['forty'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)

# Convert height from string "5-10" to inches (70)
def height_to_inches(h):
    if pd.isna(h):
        return np.nan
    if isinstance(h, (int, float)):
        return float(h)
    h = str(h)
    if '-' in h:
        parts = h.split('-')
        try:
            return int(parts[0]) * 12 + int(parts[1])
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(h)
    except ValueError:
        return np.nan

rb['height'] = rb['height'].apply(height_to_inches)

# BMI (weight in lbs, height in inches): BMI_imperial = 703 * weight / height^2
rb['bmi'] = rb.apply(
    lambda r: 703 * r['weight'] / (r['height'] ** 2)
    if pd.notna(r['weight']) and pd.notna(r['height']) and r['height'] > 0
    else np.nan, axis=1)

rb_eval = rb[rb['hit24'].notna()].copy()

print("=" * 110)
print("RB REOPTIMIZATION — SIZE METRICS & EARLY DECLARE DEEP DIVE")
print("=" * 110)
print(f"Eval sample: {len(rb_eval)} RBs")


# ============================================================================
# TEST 1: SIZE METRICS
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 1: SIZE METRICS FOR RBs")
print("=" * 110)

# Coverage
print(f"\n  COVERAGE (of {len(rb)} total RBs):")
for col, label in [('weight', 'Weight'), ('height', 'Height'), ('bmi', 'BMI')]:
    have = rb[col].notna().sum()
    pct = have / len(rb) * 100
    flag = " ⚠ LOW" if pct < 80 else ""
    print(f"    {label:<12}: {have}/{len(rb)} ({pct:.1f}%){flag}")

# Coverage of eval sample
print(f"\n  COVERAGE (of {len(rb_eval)} eval RBs):")
for col, label in [('weight', 'Weight'), ('height', 'Height'), ('bmi', 'BMI')]:
    have = rb_eval[col].notna().sum()
    pct = have / len(rb_eval) * 100
    flag = " ⚠ LOW" if pct < 80 else ""
    print(f"    {label:<12}: {have}/{len(rb_eval)} ({pct:.1f}%){flag}")

# Summary stats
print(f"\n  DESCRIPTIVE STATS (eval RBs with data):")
for col, label in [('weight', 'Weight (lbs)'), ('height', 'Height (in)'), ('bmi', 'BMI')]:
    valid = rb_eval[col].dropna()
    if len(valid) > 0:
        print(f"    {label:<15}: mean={valid.mean():.1f}, std={valid.std():.1f}, "
              f"min={valid.min():.1f}, max={valid.max():.1f}")

# Coverage by round
print(f"\n  WEIGHT COVERAGE BY ROUND:")
for rnd in sorted(rb_eval['round'].unique()):
    rnd_df = rb_eval[rb_eval['round'] == rnd]
    have = rnd_df['weight'].notna().sum()
    print(f"    Round {int(rnd)}: {have}/{len(rnd_df)} ({have/len(rnd_df)*100:.1f}%)")

# ────────────────────────────────────────────
# Partial correlations: size vs outcomes controlling for DC
# ────────────────────────────────────────────
print(f"\n\n  PARTIAL CORRELATIONS: Size vs outcomes (controlling for DC)")
print(f"  {'Metric':<20}", end="")
for out in outcome_cols:
    print(f" {'r('+out+')':>16}", end="")
print(f" {'PRI-AVG':>10}")
print("  " + "-" * 90)

size_metrics = [('weight', 'Weight'), ('height', 'Height'), ('bmi', 'BMI')]
size_partial_results = {}

for col, label in size_metrics:
    row_str = f"  {label:<20}"
    pri_sum = 0
    pri_total = 0
    for out in outcome_cols:
        r, p, n = partial_corr(rb_eval[col], rb_eval[out], rb_eval['s_dc'])
        if not np.isnan(r):
            sig = '*' if p < 0.05 else ' '
            row_str += f" {r:>+.4f}{sig}(N={n:>3})"
            pri_sum += outcome_weights[out] * r
            pri_total += outcome_weights[out]
        else:
            row_str += f" {'N/A':>16}"
    pri_avg = pri_sum / pri_total if pri_total > 0 else np.nan
    row_str += f" {pri_avg:>+.4f}"
    print(row_str)
    size_partial_results[label] = pri_avg

# ────────────────────────────────────────────
# Partial correlation controlling for DC AND Rec Production
# ────────────────────────────────────────────
print(f"\n\n  PARTIAL CORRELATIONS: Size vs outcomes (controlling for DC AND Rec Production)")
print(f"  Does size add anything beyond the two core components?")
print(f"  {'Metric':<20}", end="")
for out in outcome_cols:
    print(f" {'r('+out+')':>16}", end="")
print(f" {'PRI-AVG':>10}")
print("  " + "-" * 90)

for col, label in size_metrics:
    row_str = f"  {label:<20}"
    pri_sum = 0
    pri_total = 0
    for out in outcome_cols:
        r, p, n = partial_corr_2(rb_eval[col], rb_eval[out], rb_eval['s_dc'], rb_eval['s_rec_prod'])
        if not np.isnan(r):
            sig = '*' if p < 0.05 else ' '
            row_str += f" {r:>+.4f}{sig}(N={n:>3})"
            pri_sum += outcome_weights[out] * r
            pri_total += outcome_weights[out]
        else:
            row_str += f" {'N/A':>16}"
    pri_avg = pri_sum / pri_total if pri_total > 0 else np.nan
    row_str += f" {pri_avg:>+.4f}"
    print(row_str)

# ────────────────────────────────────────────
# Within-round analysis: does size separate hits from misses?
# ────────────────────────────────────────────
print(f"\n\n  WITHIN-ROUND ANALYSIS: Does weight separate hits from misses?")

round_groups = [(1, 1, 'Round 1'), (2, 2, 'Round 2'), (3, 4, 'Rounds 3-4'), (5, 7, 'Rounds 5-7')]

for r_lo, r_hi, label in round_groups:
    rnd = rb_eval[(rb_eval['round'] >= r_lo) & (rb_eval['round'] <= r_hi) & rb_eval['weight'].notna()]
    if len(rnd) < 5:
        continue
    hits = rnd[rnd['hit24'] == 1]
    misses = rnd[rnd['hit24'] == 0]

    print(f"\n  ── {label} ({len(rnd)} RBs with weight data) ──")
    print(f"    {'Group':<15} {'N':>5} {'Avg Wt':>8} {'Med Wt':>8} {'Std':>6}")
    print(f"    {'-' * 45}")
    if len(hits) > 0:
        print(f"    {'Hit24=1':<15} {len(hits):>5} {hits['weight'].mean():>8.1f} {hits['weight'].median():>8.1f} {hits['weight'].std():>6.1f}")
    if len(misses) > 0:
        print(f"    {'Hit24=0':<15} {len(misses):>5} {misses['weight'].mean():>8.1f} {misses['weight'].median():>8.1f} {misses['weight'].std():>6.1f}")

    if len(hits) >= 3 and len(misses) >= 3:
        t_stat, t_p = stats.ttest_ind(hits['weight'], misses['weight'])
        print(f"    t-test: t={t_stat:+.3f}, p={t_p:.4f} {'*significant*' if t_p < 0.05 else '(not significant)'}")

    # Also show first_3yr_ppg
    has_ppg = rnd[rnd['first_3yr_ppg'].notna()]
    if len(has_ppg) >= 10:
        r_wt, p_wt = stats.pearsonr(has_ppg['weight'], has_ppg['first_3yr_ppg'])
        print(f"    Weight vs first_3yr_ppg: r={r_wt:+.3f} (p={p_wt:.4f}, N={len(has_ppg)})")

# ────────────────────────────────────────────
# Weight bucket analysis
# ────────────────────────────────────────────
print(f"\n\n  WEIGHT BUCKET ANALYSIS:")
wb = rb_eval[rb_eval['weight'].notna()].copy()
wb['wt_bucket'] = pd.cut(wb['weight'],
                         bins=[0, 199, 209, 219, 999],
                         labels=['Under 200', '200-209', '210-219', '220+'])

print(f"\n  {'Weight Bucket':<15} {'N':>5} {'Hit24':>6} {'Rate':>8} {'Hit12':>6} {'Rate':>8} {'Avg PPG':>10} {'PPG N':>6} {'Avg Pick':>9}")
print(f"  {'-' * 80}")

for bucket in ['Under 200', '200-209', '210-219', '220+']:
    grp = wb[wb['wt_bucket'] == bucket]
    if len(grp) == 0:
        continue
    h24 = int(grp['hit24'].sum())
    h12 = int(grp['hit12'].sum())
    r24 = h24 / len(grp) * 100
    r12 = h12 / len(grp) * 100
    ppg_grp = grp[grp['first_3yr_ppg'].notna()]
    ppg = ppg_grp['first_3yr_ppg'].mean() if len(ppg_grp) > 0 else np.nan
    ppg_s = f"{ppg:.2f}" if not np.isnan(ppg) else "N/A"
    print(f"  {bucket:<15} {len(grp):>5} {h24:>6} {r24:>7.1f}% {h12:>6} {r12:>7.1f}% {ppg_s:>10} {len(ppg_grp):>6} {grp['pick'].mean():>9.1f}")

# Same but within rounds
print(f"\n  WEIGHT BUCKET × ROUND:")
for r_lo, r_hi, rnd_label in [(1, 2, 'Rounds 1-2'), (3, 4, 'Rounds 3-4'), (5, 7, 'Rounds 5-7')]:
    rnd_grp = wb[(wb['round'] >= r_lo) & (wb['round'] <= r_hi)]
    if len(rnd_grp) < 10:
        continue
    print(f"\n  ── {rnd_label} ({len(rnd_grp)} RBs) ──")
    print(f"  {'Weight Bucket':<15} {'N':>5} {'Hit24':>6} {'Rate':>8} {'Avg PPG':>10} {'PPG N':>6}")
    print(f"  {'-' * 55}")
    for bucket in ['Under 200', '200-209', '210-219', '220+']:
        grp = rnd_grp[rnd_grp['wt_bucket'] == bucket]
        if len(grp) == 0:
            continue
        h24 = int(grp['hit24'].sum())
        r24 = h24 / len(grp) * 100
        ppg_grp = grp[grp['first_3yr_ppg'].notna()]
        ppg = ppg_grp['first_3yr_ppg'].mean() if len(ppg_grp) > 0 else np.nan
        ppg_s = f"{ppg:.2f}" if not np.isnan(ppg) else "N/A"
        print(f"  {bucket:<15} {len(grp):>5} {h24:>6} {r24:>7.1f}% {ppg_s:>10} {len(ppg_grp):>6}")


# ============================================================================
# TEST 2: EARLY DECLARE DEEP DIVE
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 2: EARLY DECLARE DEEP DIVE")
print("Why does early declare show negative signal for RBs?")
print("=" * 110)

# Basic stats by declare group
has_dec = rb_eval[rb_eval['declare_status'].notna()].copy()
print(f"\n  RBs with declare status: {len(has_dec)}/{len(rb_eval)}")

print(f"\n  OVERALL BY DECLARE STATUS:")
print(f"  {'Status':<12} {'N':>5} {'Hit24':>6} {'Rate':>8} {'Hit12':>6} {'Rate':>8} "
      f"{'Avg PPG':>10} {'PPG N':>6} {'Avg Pick':>9} {'Avg DC':>8}")
print(f"  {'-' * 90}")

for status in ['EARLY', 'STANDARD', 'LATE']:
    grp = has_dec[has_dec['declare_status'] == status]
    if len(grp) == 0:
        continue
    h24 = int(grp['hit24'].sum())
    h12 = int(grp['hit12'].sum())
    r24 = h24 / len(grp) * 100
    r12 = h12 / len(grp) * 100
    ppg_grp = grp[grp['first_3yr_ppg'].notna()]
    ppg = ppg_grp['first_3yr_ppg'].mean() if len(ppg_grp) > 0 else np.nan
    ppg_s = f"{ppg:.2f}" if not np.isnan(ppg) else "N/A"
    print(f"  {status:<12} {len(grp):>5} {h24:>6} {r24:>7.1f}% {h12:>6} {r12:>7.1f}% "
          f"{ppg_s:>10} {len(ppg_grp):>6} {grp['pick'].mean():>9.1f} {grp['s_dc'].mean():>8.1f}")

# Within-round breakdowns
print(f"\n\n  WITHIN-ROUND BREAKDOWNS:")

for r_lo, r_hi, rnd_label in [(1, 1, 'Round 1'), (2, 2, 'Round 2'), (3, 4, 'Rounds 3-4'), (5, 7, 'Rounds 5-7')]:
    rnd = has_dec[(has_dec['round'] >= r_lo) & (has_dec['round'] <= r_hi)]
    if len(rnd) < 5:
        continue
    print(f"\n  ── {rnd_label} ({len(rnd)} RBs) ──")
    print(f"  {'Status':<12} {'N':>5} {'Hit24':>6} {'Rate':>8} {'Avg PPG':>10} {'PPG N':>6} {'Avg Pick':>9}")
    print(f"  {'-' * 60}")
    for status in ['EARLY', 'STANDARD', 'LATE']:
        grp = rnd[rnd['declare_status'] == status]
        if len(grp) == 0:
            continue
        h24 = int(grp['hit24'].sum())
        r24 = h24 / len(grp) * 100
        ppg_grp = grp[grp['first_3yr_ppg'].notna()]
        ppg = ppg_grp['first_3yr_ppg'].mean() if len(ppg_grp) > 0 else np.nan
        ppg_s = f"{ppg:.2f}" if not np.isnan(ppg) else "N/A"
        print(f"  {status:<12} {len(grp):>5} {h24:>6} {r24:>7.1f}% {ppg_s:>10} {len(ppg_grp):>6} {grp['pick'].mean():>9.1f}")

# Confounding analysis
print(f"\n\n  CONFOUNDING ANALYSIS: Is early declare just a proxy for DC?")
print(f"  If early declare RBs are drafted higher, the partial correlation removes that overlap.")
print(f"  Let's check if ED adds anything beyond DC position.")

# Average DC by declare status
print(f"\n  Average DC by declare status:")
for status in ['EARLY', 'STANDARD', 'LATE']:
    grp = has_dec[has_dec['declare_status'] == status]
    if len(grp) > 0:
        print(f"    {status:<12}: avg DC = {grp['s_dc'].mean():.1f}, avg pick = {grp['pick'].mean():.0f}")

# What happens if we look at "early declare for their DC"?
# Compare early vs non-early within DC tiers
print(f"\n\n  EARLY DECLARE WITHIN DC TIERS:")
dc_tiers = [(70, 100, 'DC 70-100 (Rds 1-2)'), (50, 70, 'DC 50-70 (Rds 3-4)'), (0, 50, 'DC 0-50 (Rds 5-7)')]

for lo, hi, label in dc_tiers:
    tier = has_dec[(has_dec['s_dc'] >= lo) & (has_dec['s_dc'] < hi)]
    if len(tier) < 5:
        continue
    early = tier[tier['declare_status'] == 'EARLY']
    non_early = tier[tier['declare_status'] != 'EARLY']
    print(f"\n  ── {label} ({len(tier)} RBs) ──")
    for grp, gl in [(early, 'EARLY'), (non_early, 'NOT EARLY')]:
        if len(grp) == 0:
            continue
        h24 = int(grp['hit24'].sum())
        r24 = h24 / len(grp) * 100
        ppg_grp = grp[grp['first_3yr_ppg'].notna()]
        ppg = ppg_grp['first_3yr_ppg'].mean() if len(ppg_grp) > 0 else np.nan
        ppg_s = f"{ppg:.2f}" if not np.isnan(ppg) else "N/A"
        print(f"    {gl:<15} N={len(grp):>3}, hit24={r24:>5.1f}%, avg_ppg={ppg_s}")

# Residual analysis: do early declare RBs outperform or underperform their DC expectation?
print(f"\n\n  RESIDUAL ANALYSIS: Do early declare RBs over/underperform DC expectation?")
has_ppg = has_dec[has_dec['first_3yr_ppg'].notna()].copy()
if len(has_ppg) > 20:
    slope, intercept, _, _, _ = stats.linregress(has_ppg['s_dc'], has_ppg['first_3yr_ppg'])
    has_ppg['ppg_expected'] = slope * has_ppg['s_dc'] + intercept
    has_ppg['ppg_residual'] = has_ppg['first_3yr_ppg'] - has_ppg['ppg_expected']

    print(f"  DC regression: first_3yr_ppg = {slope:.3f} × DC + {intercept:.3f}")
    print(f"\n  {'Status':<12} {'N':>5} {'Avg Residual':>14} {'Median Resid':>14} {'Interpretation':<30}")
    print(f"  {'-' * 80}")
    for status in ['EARLY', 'STANDARD', 'LATE']:
        grp = has_ppg[has_ppg['declare_status'] == status]
        if len(grp) == 0:
            continue
        avg_r = grp['ppg_residual'].mean()
        med_r = grp['ppg_residual'].median()
        interp = "overperforms DC" if avg_r > 0.5 else "underperforms DC" if avg_r < -0.5 else "matches DC"
        print(f"  {status:<12} {len(grp):>5} {avg_r:>+14.2f} {med_r:>+14.2f} {interp:<30}")

# Show specific examples of early declare RBs who underperformed
print(f"\n  EARLY DECLARE RBs — sorted by residual (worst to best):")
early_ppg = has_ppg[has_ppg['declare_status'] == 'EARLY'].sort_values('ppg_residual')
print(f"  {'Player':<25} {'Yr':>4} {'Pick':>4} {'DC':>5} {'Expected':>9} {'Actual':>8} {'Residual':>9} {'Hit24':>6}")
print(f"  {'-' * 75}")
for _, r in early_ppg.iterrows():
    print(f"  {r['player_name']:<25} {int(r['draft_year']):>4} {int(r['pick']):>4} {r['s_dc']:>5.1f} "
          f"{r['ppg_expected']:>9.2f} {r['first_3yr_ppg']:>8.2f} {r['ppg_residual']:>+9.2f} {int(r['hit24']):>6}")


# ============================================================================
# TEST 3: HEAD-TO-HEAD IF EITHER SHOWS PROMISE
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 3: HEAD-TO-HEAD — Adding size and/or early declare to 65/35")
print("=" * 110)

# Normalize weight to 0-100
valid_wt = rb_eval['weight'].notna()
rb_eval['s_weight'] = np.nan
if valid_wt.sum() > 10:
    rb_eval.loc[valid_wt, 's_weight'] = normalize_0_100(rb_eval.loc[valid_wt, 'weight'])
avg_wt = rb_eval['s_weight'].mean() if rb_eval['s_weight'].notna().any() else 50
rb_eval['s_weight_filled'] = rb_eval['s_weight'].fillna(avg_wt)

# Early declare
rb_eval['s_early_dec'] = rb_eval['early_declare'] * 100

# RAS (for current model comparison)
rb_eval['s_ras'] = rb_eval['RAS'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)
avg_ras = rb_eval['s_ras'].mean()
rb_eval['s_ras_filled'] = rb_eval['s_ras'].fillna(avg_ras)

def compute_slap(row, weights_dict):
    total = 0
    for col, w in weights_dict.items():
        total += row[col] * w
    return min(100, max(0, total))

def full_eval(df, label, comp_weights):
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

    n_top = max(1, len(df) // 10)
    top = df.nlargest(n_top, 'slap')
    hit24_rate = top['hit24'].mean() * 100
    hit12_rate = top['hit12'].mean() * 100
    top_3yr = top[top['first_3yr_ppg'].notna()]
    ppg_top = top_3yr['first_3yr_ppg'].mean() if len(top_3yr) > 0 else np.nan

    # Brier
    valid_b = df[df['hit24'].notna()].copy()
    prob = valid_b['slap'] / 100
    brier = ((prob - valid_b['hit24']) ** 2).mean()

    # AUC
    pos = valid_b[valid_b['hit24'] == 1]['slap']
    neg = valid_b[valid_b['hit24'] == 0]['slap']
    auc_sum = 0
    for p_val in pos:
        auc_sum += (neg < p_val).sum() + 0.5 * (neg == p_val).sum()
    auc = auc_sum / (len(pos) * len(neg)) if len(pos) * len(neg) > 0 else np.nan

    # Disagree
    df['dc_rank'] = df['s_dc'].rank(ascending=False, method='min')
    df['slap_rank'] = df['slap'].rank(ascending=False, method='min')
    disagree = int(((df['dc_rank'] - df['slap_rank']).abs() >= 10).sum())

    return {
        'label': label, 'pri_avg': pri_avg, 'outcomes': results,
        'hit24_top10': hit24_rate, 'hit12_top10': hit12_rate,
        'ppg_top10': ppg_top, 'n_top': n_top,
        'brier': brier, 'auc': auc, 'disagree_10': disagree,
    }

configs = [
    ('DC only',                          {'s_dc': 1.00}),
    ('CURRENT: DC/Rec/RAS 50/35/15',    {'s_dc': 0.50, 's_rec_prod_filled': 0.35, 's_ras_filled': 0.15}),
    ('PROPOSED: DC/Rec 65/35',           {'s_dc': 0.65, 's_rec_prod_filled': 0.35}),
    ('DC/Rec/Wt 60/30/10',              {'s_dc': 0.60, 's_rec_prod_filled': 0.30, 's_weight_filled': 0.10}),
    ('DC/Rec/Wt 60/35/5',               {'s_dc': 0.60, 's_rec_prod_filled': 0.35, 's_weight_filled': 0.05}),
    ('DC/Rec/ED 60/35/5',               {'s_dc': 0.60, 's_rec_prod_filled': 0.35, 's_early_dec': 0.05}),
    ('DC/Rec/ED 60/30/10',              {'s_dc': 0.60, 's_rec_prod_filled': 0.30, 's_early_dec': 0.10}),
    ('DC/Rec/Wt/ED 60/30/5/5',          {'s_dc': 0.60, 's_rec_prod_filled': 0.30, 's_weight_filled': 0.05, 's_early_dec': 0.05}),
    ('DC/Rec/Wt/ED 55/30/10/5',         {'s_dc': 0.55, 's_rec_prod_filled': 0.30, 's_weight_filled': 0.10, 's_early_dec': 0.05}),
]

all_results = []
for label, weights in configs:
    r = full_eval(rb_eval, label, weights)
    all_results.append(r)

# Sort by PRI-AVG
all_results_sorted = sorted(all_results, key=lambda x: x['pri_avg'] if not np.isnan(x['pri_avg']) else -999, reverse=True)

print(f"\n  {'Rank':>4} {'Config':<38} {'PRI-AVG':>8} {'r(3yr)':>8} {'r(h24)':>8} {'Top10%h24':>10} {'Top10%PPG':>10} {'AUC':>7} {'Brier':>7} {'Dis10+':>7}")
print("  " + "-" * 115)

proposed_r = [r for r in all_results if 'PROPOSED' in r['label']][0]['pri_avg']

for i, r in enumerate(all_results_sorted, 1):
    o = r['outcomes']
    r3 = o.get('first_3yr_ppg', {}).get('r', np.nan)
    rh24 = o.get('hit24', {}).get('r', np.nan)
    ppg_s = f"{r['ppg_top10']:.2f}" if not np.isnan(r['ppg_top10']) else "N/A"
    delta = r['pri_avg'] - proposed_r
    marker = ""
    if 'PROPOSED' in r['label']:
        marker = " ◄ 65/35"
    elif 'CURRENT' in r['label']:
        marker = " ◄ CURRENT"
    elif 'DC only' in r['label']:
        marker = " ◄ BASE"

    print(f"  {i:>4}. {r['label']:<38} {r['pri_avg']:>+.4f} {r3:>+.4f} {rh24:>+.4f}"
          f" {r['hit24_top10']:>9.1f}% {ppg_s:>10} {r['auc']:>.4f} {r['brier']:>.4f} {r['disagree_10']:>7}{marker}")

# Compare adds vs baseline
print(f"\n\n  DELTA VS PROPOSED DC/Rec 65/35:")
for r in all_results_sorted:
    if 'PROPOSED' in r['label'] or 'DC only' in r['label'] or 'CURRENT' in r['label']:
        continue
    delta = r['pri_avg'] - proposed_r
    delta_auc = r['auc'] - [x for x in all_results if 'PROPOSED' in x['label']][0]['auc']
    delta_brier = r['brier'] - [x for x in all_results if 'PROPOSED' in x['label']][0]['brier']
    print(f"    {r['label']:<38} PRI-AVG: {delta:>+.4f}  AUC: {delta_auc:>+.4f}  Brier: {delta_brier:>+.4f}")

print(f"\n\n{'=' * 110}")
print("ANALYSIS COMPLETE")
print("=" * 110)
