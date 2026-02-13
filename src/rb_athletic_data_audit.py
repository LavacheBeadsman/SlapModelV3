"""
RB Athletic Data Audit — Is a data problem masking real signal?
================================================================
6 audits: coverage bias, raw data, normalization, non-linearity,
round interaction, and data source verification.
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

def normalize_name(name):
    if pd.isna(name):
        return ''
    return str(name).strip().lower().replace('.', '').replace("'", '').replace('-', ' ')

def speed_score_fn(weight, forty):
    if pd.isna(weight) or pd.isna(forty) or forty <= 0 or weight <= 0:
        return np.nan
    return (weight * 200) / (forty ** 4)

def rb_production_score(row):
    if pd.isna(row.get('rec_yards')) or pd.isna(row.get('team_pass_att')) or row.get('team_pass_att', 0) == 0:
        return np.nan
    age = row.get('age', 22)
    if pd.isna(age): age = 22
    age_weights = {19: 1.15, 20: 1.10, 21: 1.05, 22: 1.00, 23: 0.95}
    age_w = age_weights.get(int(age), 0.90 if age >= 24 else 1.15)
    raw = (row['rec_yards'] / row['team_pass_att']) * age_w * 100
    return min(99.9, raw / 1.75)

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
rb['name_norm'] = rb['player_name'].apply(normalize_name)

# Load combine data
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine_rb = combine[combine['pos'] == 'RB'].copy()
combine_rb['name_norm'] = combine_rb['player_name'].apply(normalize_name)

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
        'forty': row['forty'],
        'ht': row['ht'],
    }

rb['weight'] = rb.apply(lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb['forty'] = rb.apply(lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
for col in ['weight', 'forty']:
    rb[col] = pd.to_numeric(rb[col], errors='coerce')

rb['speed_score'] = rb.apply(lambda r: speed_score_fn(r['weight'], r['forty']), axis=1)
rb['ras'] = rb['RAS']

rb_eval = rb[rb['hit24'].notna()].copy()

# ============================================================================
# AUDIT 1: COVERAGE AND BIAS CHECK
# ============================================================================

print("=" * 120)
print("AUDIT 1: COVERAGE AND BIAS CHECK")
print("Are the players missing athletic data systematically different?")
print("=" * 120)

metrics = {
    'Weight': 'weight',
    '40 time': 'forty',
    'Speed Score': 'speed_score',
    'RAS': 'ras',
}

print(f"\n  {'Metric':<15} {'Has Data':>10} {'Missing':>10} {'Avg Pick (has)':>15} {'Avg Pick (miss)':>16} {'Hit24% (has)':>13} {'Hit24% (miss)':>14} {'Bias?':>10}")
print("  " + "-" * 105)

for label, col in metrics.items():
    has = rb_eval[rb_eval[col].notna()]
    miss = rb_eval[rb_eval[col].isna()]
    n_has = len(has)
    n_miss = len(miss)
    avg_pick_has = has['pick'].mean()
    avg_pick_miss = miss['pick'].mean() if len(miss) > 0 else np.nan
    hit24_has = has['hit24'].mean() * 100
    hit24_miss = miss['hit24'].mean() * 100 if len(miss) > 0 else np.nan

    # Is there a significant difference?
    if len(miss) > 5:
        t_pick, p_pick = stats.ttest_ind(has['pick'], miss['pick'])
        bias = "YES" if p_pick < 0.05 else "no"
    else:
        bias = "n/a (few)"

    pick_miss_s = f"{avg_pick_miss:.1f}" if not np.isnan(avg_pick_miss) else "N/A"
    hit_miss_s = f"{hit24_miss:.1f}%" if not np.isnan(hit24_miss) else "N/A"

    print(f"  {label:<15} {n_has:>10} {n_miss:>10} {avg_pick_has:>15.1f} {pick_miss_s:>16} {hit24_has:>12.1f}% {hit_miss_s:>14} {bias:>10}")

# Detailed breakdown: who's missing?
print(f"\n\n  ── WHO IS MISSING WEIGHT DATA? ──")
miss_wt = rb_eval[rb_eval['weight'].isna()].sort_values('pick')
print(f"  {len(miss_wt)} RBs missing weight (of {len(rb_eval)})")
print(f"\n  {'Player':<25} {'Year':>4} {'Pick':>4} {'Rd':>3} {'hit24':>5} {'hit12':>5} {'3yr PPG':>8}")
print("  " + "-" * 60)
for _, row in miss_wt.head(30).iterrows():
    ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "?"
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>4} {int(row['pick']):>4} {int(row['round']):>3} {int(row['hit24']):>5} {int(row['hit12']):>5} {ppg:>8}")

# Same for 40 time
print(f"\n\n  ── WHO IS MISSING 40-YARD DASH? ──")
miss_40 = rb_eval[rb_eval['forty'].isna()].sort_values('pick')
print(f"  {len(miss_40)} RBs missing 40 time (of {len(rb_eval)})")
if len(miss_40) <= 40:
    print(f"\n  {'Player':<25} {'Year':>4} {'Pick':>4} {'Rd':>3} {'Wt':>5} {'hit24':>5} {'3yr PPG':>8}")
    print("  " + "-" * 60)
    for _, row in miss_40.iterrows():
        wt = f"{row['weight']:.0f}" if pd.notna(row['weight']) else "?"
        ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "?"
        print(f"  {row['player_name']:<25} {int(row['draft_year']):>4} {int(row['pick']):>4} {int(row['round']):>3} {wt:>5} {int(row['hit24']):>5} {ppg:>8}")
else:
    print(f"  First 20:")
    print(f"\n  {'Player':<25} {'Year':>4} {'Pick':>4} {'Rd':>3} {'Wt':>5} {'hit24':>5} {'3yr PPG':>8}")
    print("  " + "-" * 60)
    for _, row in miss_40.head(20).iterrows():
        wt = f"{row['weight']:.0f}" if pd.notna(row['weight']) else "?"
        ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "?"
        print(f"  {row['player_name']:<25} {int(row['draft_year']):>4} {int(row['pick']):>4} {int(row['round']):>3} {wt:>5} {int(row['hit24']):>5} {ppg:>8}")

# Hit24 among missing players
for label, col in metrics.items():
    miss = rb_eval[rb_eval[col].isna()]
    if len(miss) > 5:
        hits_missing = miss[miss['hit24'] == 1]
        print(f"\n  Missing {label}: {len(hits_missing)} hits out of {len(miss)} missing players")
        if len(hits_missing) > 0:
            print(f"    Hit players missing {label}:")
            for _, row in hits_missing.sort_values('pick').iterrows():
                ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "?"
                print(f"      {row['player_name']:<25} pick {int(row['pick']):>3} | 3yr PPG: {ppg}")


# ============================================================================
# AUDIT 2: RAW DATA VERIFICATION
# ============================================================================

print(f"\n\n{'=' * 120}")
print("AUDIT 2: RAW DATA VERIFICATION")
print("Are the actual weight and 40-time values correct?")
print("=" * 120)

has_wt = rb_eval[rb_eval['weight'].notna()].copy()

print(f"\n  ── 10 HEAVIEST RBs ──")
print(f"  {'Player':<25} {'Year':>4} {'Pick':>4} {'Wt':>5} {'hit24':>5} {'hit12':>5} {'3yr PPG':>8} {'career PPG':>10}")
print("  " + "-" * 75)
for _, row in has_wt.nlargest(10, 'weight').iterrows():
    ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "?"
    cpg = f"{row['career_ppg']:.1f}" if pd.notna(row['career_ppg']) else "?"
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>4} {int(row['pick']):>4} {row['weight']:>5.0f} {int(row['hit24']):>5} {int(row['hit12']):>5} {ppg:>8} {cpg:>10}")

print(f"\n  ── 10 LIGHTEST RBs ──")
print(f"  {'Player':<25} {'Year':>4} {'Pick':>4} {'Wt':>5} {'hit24':>5} {'hit12':>5} {'3yr PPG':>8} {'career PPG':>10}")
print("  " + "-" * 75)
for _, row in has_wt.nsmallest(10, 'weight').iterrows():
    ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "?"
    cpg = f"{row['career_ppg']:.1f}" if pd.notna(row['career_ppg']) else "?"
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>4} {int(row['pick']):>4} {row['weight']:>5.0f} {int(row['hit24']):>5} {int(row['hit12']):>5} {ppg:>8} {cpg:>10}")

has_40 = rb_eval[rb_eval['forty'].notna()].copy()

print(f"\n  ── 10 FASTEST RBs (lowest 40 time) ──")
print(f"  {'Player':<25} {'Year':>4} {'Pick':>4} {'40':>6} {'Wt':>5} {'hit24':>5} {'hit12':>5} {'3yr PPG':>8}")
print("  " + "-" * 70)
for _, row in has_40.nsmallest(10, 'forty').iterrows():
    wt = f"{row['weight']:.0f}" if pd.notna(row['weight']) else "?"
    ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "?"
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>4} {int(row['pick']):>4} {row['forty']:>6.2f} {wt:>5} {int(row['hit24']):>5} {int(row['hit12']):>5} {ppg:>8}")

print(f"\n  ── 10 SLOWEST RBs (highest 40 time) ──")
print(f"  {'Player':<25} {'Year':>4} {'Pick':>4} {'40':>6} {'Wt':>5} {'hit24':>5} {'hit12':>5} {'3yr PPG':>8}")
print("  " + "-" * 70)
for _, row in has_40.nlargest(10, 'forty').iterrows():
    wt = f"{row['weight']:.0f}" if pd.notna(row['weight']) else "?"
    ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "?"
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>4} {int(row['pick']):>4} {row['forty']:>6.2f} {wt:>5} {int(row['hit24']):>5} {int(row['hit12']):>5} {ppg:>8}")

# Distribution summary
print(f"\n  ── DISTRIBUTION SUMMARY ──")
print(f"  Weight: n={has_wt['weight'].count()}, min={has_wt['weight'].min():.0f}, p25={has_wt['weight'].quantile(0.25):.0f}, median={has_wt['weight'].median():.0f}, p75={has_wt['weight'].quantile(0.75):.0f}, max={has_wt['weight'].max():.0f}, std={has_wt['weight'].std():.1f}")
print(f"  40 time: n={has_40['forty'].count()}, min={has_40['forty'].min():.2f}, p25={has_40['forty'].quantile(0.25):.2f}, median={has_40['forty'].median():.2f}, p75={has_40['forty'].quantile(0.75):.2f}, max={has_40['forty'].max():.2f}, std={has_40['forty'].std():.2f}")

# Flag potential data errors
print(f"\n  ── POTENTIAL DATA ERRORS ──")
sus_wt = has_wt[(has_wt['weight'] < 170) | (has_wt['weight'] > 250)]
if len(sus_wt) > 0:
    print(f"  Weight outside 170-250 range ({len(sus_wt)} players):")
    for _, row in sus_wt.iterrows():
        print(f"    {row['player_name']:<25} {row['weight']:.0f} lbs (pick {int(row['pick'])})")
else:
    print(f"  No suspicious weight values (all between 170-250)")

sus_40 = has_40[(has_40['forty'] < 4.20) | (has_40['forty'] > 5.00)]
if len(sus_40) > 0:
    print(f"  40 time outside 4.20-5.00 range ({len(sus_40)} players):")
    for _, row in sus_40.iterrows():
        print(f"    {row['player_name']:<25} {row['forty']:.2f}s (pick {int(row['pick'])})")
else:
    print(f"  No suspicious 40 times (all between 4.20-5.00)")


# ============================================================================
# AUDIT 3: HOW WAS WEIGHT NORMALIZED?
# ============================================================================

print(f"\n\n{'=' * 120}")
print("AUDIT 3: NORMALIZATION METHOD")
print("How was weight converted to a 0-100 score?")
print("=" * 120)

# Reproduce exactly what the previous script did
wt_valid = rb_eval[rb_eval['weight'].notna()].copy()
wt_min = wt_valid['weight'].min()
wt_max = wt_valid['weight'].max()
wt_range = wt_max - wt_min

print(f"\n  Min weight: {wt_min:.0f} lbs")
print(f"  Max weight: {wt_max:.0f} lbs")
print(f"  Range: {wt_range:.0f} lbs")
print(f"  Method: linear min-max normalization to 0-100")
print(f"  Formula: score = (weight - {wt_min:.0f}) / {wt_range:.0f} * 100")

# Show what this means for specific weights
print(f"\n  ── WHAT 5 LBS MEANS IN SCORE TERMS ──")
points_per_lb = 100 / wt_range
print(f"  Points per pound: {points_per_lb:.2f}")
print(f"  5 lbs = {5 * points_per_lb:.1f} points")
print(f"  10 lbs = {10 * points_per_lb:.1f} points")

print(f"\n  ── WEIGHT → SCORE MAPPING ──")
print(f"  {'Weight':>8} {'Score':>8} {'Players at this weight':>25}")
for wt_check in range(175, 255, 5):
    score = (wt_check - wt_min) / wt_range * 100
    n_at = len(wt_valid[(wt_valid['weight'] >= wt_check - 2) & (wt_valid['weight'] <= wt_check + 2)])
    if score >= 0 and score <= 100:
        print(f"  {wt_check:>6} lbs {score:>7.1f}   ({n_at:>3} RBs within ±2 lbs)")

# Show what imputation does
avg_wt_score = (wt_valid['weight'].mean() - wt_min) / wt_range * 100
print(f"\n  ── IMPUTATION FOR MISSING PLAYERS ──")
print(f"  Average weight: {wt_valid['weight'].mean():.1f} lbs")
print(f"  Imputed score: {avg_wt_score:.1f}")
print(f"  {len(rb_eval) - len(wt_valid)} players receive this imputed score")
print(f"  This means missing players all get the SAME score ({avg_wt_score:.1f})")
print(f"  Which pulls them toward the middle, compressing score variance")

# How much variance is real vs imputed?
n_obs = len(wt_valid)
n_imp = len(rb_eval) - len(wt_valid)
print(f"\n  Real variance: {n_obs} players ({n_obs/len(rb_eval)*100:.0f}%) have actual scores spanning 0-100")
print(f"  Imputed noise: {n_imp} players ({n_imp/len(rb_eval)*100:.0f}%) all get score {avg_wt_score:.1f}")

# Show percentile-based normalization as alternative
print(f"\n  ── ALTERNATIVE: PERCENTILE NORMALIZATION ──")
wt_valid['wt_pctile'] = wt_valid['weight'].rank(pct=True) * 100
print(f"  {'Weight':>8} {'MinMax Score':>13} {'Percentile':>12} {'Difference':>12}")
for wt_check in [190, 200, 205, 210, 215, 220, 225, 230, 235]:
    close = wt_valid[(wt_valid['weight'] >= wt_check - 1) & (wt_valid['weight'] <= wt_check + 1)]
    if len(close) > 0:
        mm_score = (wt_check - wt_min) / wt_range * 100
        pct_score = close['wt_pctile'].mean()
        print(f"  {wt_check:>6} lbs {mm_score:>12.1f} {pct_score:>12.1f} {pct_score - mm_score:>+12.1f}")


# ============================================================================
# AUDIT 4: NON-LINEAR RELATIONSHIP
# ============================================================================

print(f"\n\n{'=' * 120}")
print("AUDIT 4: NON-LINEAR RELATIONSHIP")
print("Is there an inverted U where 210-225 is the sweet spot?")
print("=" * 120)

wt_eval = rb_eval[rb_eval['weight'].notna() & rb_eval['first_3yr_ppg'].notna()].copy()

print(f"\n  ── first_3yr_ppg BY WEIGHT BUCKET (5-lb increments) ──")
print(f"  {'Weight Range':>15} {'n':>4} {'Avg PPG':>8} {'Hit24%':>7} {'Hit12%':>7} {'Avg Pick':>9}")
print("  " + "-" * 55)

buckets = [(185, 195), (195, 200), (200, 205), (205, 210), (210, 215),
           (215, 220), (220, 225), (225, 230), (230, 240), (240, 250)]

for lo, hi in buckets:
    bucket = wt_eval[(wt_eval['weight'] >= lo) & (wt_eval['weight'] < hi)]
    if len(bucket) > 0:
        avg_ppg = bucket['first_3yr_ppg'].mean()
        h24 = bucket['hit24'].mean() * 100
        h12 = bucket['hit12'].mean() * 100
        avg_pick = bucket['pick'].mean()
        print(f"  {lo:>6}-{hi:<6} lbs {len(bucket):>4} {avg_ppg:>8.2f} {h24:>6.1f}% {h12:>6.1f}% {avg_pick:>9.1f}")
    else:
        print(f"  {lo:>6}-{hi:<6} lbs    0      N/A    N/A    N/A       N/A")

# Control for draft capital: within each weight bucket, what's the residual PPG?
print(f"\n  ── RESIDUAL PPG (after removing draft capital effect) BY WEIGHT ──")
from numpy.linalg import lstsq as np_lstsq
X_dc = np.column_stack([wt_eval['s_dc'].values, np.ones(len(wt_eval))])
beta, _, _, _ = np_lstsq(X_dc, wt_eval['first_3yr_ppg'].values, rcond=None)
wt_eval['ppg_resid'] = wt_eval['first_3yr_ppg'].values - X_dc @ beta

print(f"  {'Weight Range':>15} {'n':>4} {'Avg Residual PPG':>18} {'Direction':>10}")
print("  " + "-" * 52)

for lo, hi in buckets:
    bucket = wt_eval[(wt_eval['weight'] >= lo) & (wt_eval['weight'] < hi)]
    if len(bucket) >= 3:
        avg_resid = bucket['ppg_resid'].mean()
        direction = "▲" if avg_resid > 0.5 else "▼" if avg_resid < -0.5 else "─"
        print(f"  {lo:>6}-{hi:<6} lbs {len(bucket):>4} {avg_resid:>+17.2f} {direction:>10}")

# Also check: is there a quadratic relationship?
wt_linear = wt_eval[['weight', 'first_3yr_ppg']].dropna()
r_linear, p_linear = stats.pearsonr(wt_linear['weight'], wt_linear['first_3yr_ppg'])

# Quadratic: use weight and weight^2
wt_eval['weight_sq'] = wt_eval['weight'] ** 2
X_quad = np.column_stack([wt_eval['weight'].values, wt_eval['weight_sq'].values, np.ones(len(wt_eval))])
beta_quad, _, _, _ = np_lstsq(X_quad, wt_eval['first_3yr_ppg'].values, rcond=None)

# After controlling for DC
X_dc_wt = np.column_stack([wt_eval['s_dc'].values, wt_eval['weight'].values, np.ones(len(wt_eval))])
X_dc_wt_sq = np.column_stack([wt_eval['s_dc'].values, wt_eval['weight'].values, wt_eval['weight_sq'].values, np.ones(len(wt_eval))])

from scipy.stats import f as f_dist
# F-test: does adding weight^2 improve over weight alone?
n = len(wt_eval)
rss_linear = np.sum((wt_eval['first_3yr_ppg'].values - X_dc_wt @ np_lstsq(X_dc_wt, wt_eval['first_3yr_ppg'].values, rcond=None)[0]) ** 2)
rss_quad = np.sum((wt_eval['first_3yr_ppg'].values - X_dc_wt_sq @ np_lstsq(X_dc_wt_sq, wt_eval['first_3yr_ppg'].values, rcond=None)[0]) ** 2)
f_stat = ((rss_linear - rss_quad) / 1) / (rss_quad / (n - 4))
p_quad = 1 - f_dist.cdf(f_stat, 1, n - 4)

print(f"\n  Quadratic coefficient (weight²): {beta_quad[1]:.6f}")
print(f"  F-test for quadratic term (controlling for DC): F={f_stat:.3f}, p={p_quad:.4f}")
if p_quad < 0.05:
    # Find the peak
    peak_weight = -beta_quad[0] / (2 * beta_quad[1])
    print(f"  ⚡ Significant quadratic! Peak at {peak_weight:.0f} lbs")
else:
    print(f"  No significant quadratic relationship (p={p_quad:.4f})")


# ============================================================================
# AUDIT 5: INTERACTION WITH DRAFT ROUND
# ============================================================================

print(f"\n\n{'=' * 120}")
print("AUDIT 5: WEIGHT × ROUND INTERACTION")
print("Does weight help in some rounds but hurt in others?")
print("=" * 120)

wt_rd = rb_eval[rb_eval['weight'].notna()].copy()
wt_rd['rd_group'] = wt_rd['round'].apply(
    lambda r: 'Rd 1' if r == 1 else 'Rd 2' if r == 2 else 'Rd 3-4' if r <= 4 else 'Rd 5-7')

print(f"\n  ── AVERAGE WEIGHT: HITS vs MISSES BY ROUND ──")
print(f"  {'Round Group':>12} {'n(hit24=1)':>12} {'Avg Wt(hit)':>12} {'n(hit24=0)':>12} {'Avg Wt(miss)':>13} {'Δ Weight':>10} {'p-value':>10}")
print("  " + "-" * 85)

for rd_group in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5-7']:
    grp = wt_rd[wt_rd['rd_group'] == rd_group]
    hits = grp[grp['hit24'] == 1]
    misses = grp[grp['hit24'] == 0]

    n_h = len(hits)
    n_m = len(misses)
    avg_h = hits['weight'].mean() if n_h > 0 else np.nan
    avg_m = misses['weight'].mean() if n_m > 0 else np.nan
    delta = avg_h - avg_m if not np.isnan(avg_h) and not np.isnan(avg_m) else np.nan

    if n_h >= 3 and n_m >= 3:
        t, p = stats.ttest_ind(hits['weight'], misses['weight'])
        sig = '*' if p < 0.05 else '†' if p < 0.10 else ''
        p_s = f"{p:.4f}{sig}"
    else:
        p_s = "n/a"

    delta_s = f"{delta:+.1f}" if not np.isnan(delta) else "N/A"

    print(f"  {rd_group:>12} {n_h:>12} {avg_h:>12.1f} {n_m:>12} {avg_m:>13.1f} {delta_s:>10} {p_s:>10}")

# Also show hit rates by weight bucket within each round group
print(f"\n\n  ── HIT24 RATE BY WEIGHT × ROUND ──")
print(f"  {'':>12}", end="")
for lo, hi in [(185, 205), (205, 215), (215, 225), (225, 250)]:
    print(f" {lo}-{hi}lb", end="")
print()
print("  " + "-" * 65)

for rd_group in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5-7']:
    grp = wt_rd[wt_rd['rd_group'] == rd_group]
    row = f"  {rd_group:>12}"
    for lo, hi in [(185, 205), (205, 215), (215, 225), (225, 250)]:
        bucket = grp[(grp['weight'] >= lo) & (grp['weight'] < hi)]
        if len(bucket) >= 3:
            rate = bucket['hit24'].mean() * 100
            row += f" {rate:>6.0f}% ({len(bucket):>2})"
        elif len(bucket) > 0:
            rate = bucket['hit24'].mean() * 100
            row += f" {rate:>6.0f}%*({len(bucket):>1})"
        else:
            row += f"    N/A     "
    print(row)

print(f"  * = fewer than 3 players, unreliable")

# Test interaction term statistically
print(f"\n  ── FORMAL INTERACTION TEST ──")
wt_test = wt_rd[wt_rd['first_3yr_ppg'].notna()].copy()
wt_test['wt_x_pick'] = wt_test['weight'] * wt_test['pick']

# Model: PPG ~ DC + weight + DC*weight
X_main = np.column_stack([wt_test['s_dc'].values, wt_test['weight'].values, np.ones(len(wt_test))])
X_inter = np.column_stack([wt_test['s_dc'].values, wt_test['weight'].values,
                            wt_test['s_dc'].values * wt_test['weight'].values, np.ones(len(wt_test))])

rss_main = np.sum((wt_test['first_3yr_ppg'].values - X_main @ np_lstsq(X_main, wt_test['first_3yr_ppg'].values, rcond=None)[0]) ** 2)
rss_inter = np.sum((wt_test['first_3yr_ppg'].values - X_inter @ np_lstsq(X_inter, wt_test['first_3yr_ppg'].values, rcond=None)[0]) ** 2)
n_test = len(wt_test)
f_inter = ((rss_main - rss_inter) / 1) / (rss_inter / (n_test - 4))
p_inter = 1 - f_dist.cdf(f_inter, 1, n_test - 4)

print(f"  Testing: first_3yr_ppg ~ DC + weight + DC×weight")
print(f"  F-statistic for interaction: {f_inter:.3f}")
print(f"  p-value: {p_inter:.4f}")
if p_inter < 0.05:
    print(f"  ⚡ Significant interaction — weight matters differently by round!")
else:
    print(f"  No significant interaction (p={p_inter:.4f})")

# Same test for hit24 (logistic would be better but let's use linear probability)
print(f"\n  Testing: hit24 ~ DC + weight + DC×weight (linear probability)")
rss_main_h = np.sum((wt_test['hit24'].values - X_main @ np_lstsq(X_main, wt_test['hit24'].values, rcond=None)[0]) ** 2)
rss_inter_h = np.sum((wt_test['hit24'].values - X_inter @ np_lstsq(X_inter, wt_test['hit24'].values, rcond=None)[0]) ** 2)
f_inter_h = ((rss_main_h - rss_inter_h) / 1) / (rss_inter_h / (n_test - 4))
p_inter_h = 1 - f_dist.cdf(f_inter_h, 1, n_test - 4)
print(f"  F-statistic: {f_inter_h:.3f}")
print(f"  p-value: {p_inter_h:.4f}")
if p_inter_h < 0.05:
    print(f"  ⚡ Significant interaction for hit24!")
else:
    print(f"  No significant interaction for hit24 (p={p_inter_h:.4f})")


# ============================================================================
# AUDIT 6: DATA SOURCE VERIFICATION
# ============================================================================

print(f"\n\n{'=' * 120}")
print("AUDIT 6: DATA SOURCE VERIFICATION")
print("Where did the weight and 40 data come from?")
print("=" * 120)

print(f"\n  Data source: data/nflverse/combine.parquet")
print(f"  This is NFL Combine measurement data")
print(f"  Weight = official combine weigh-in (wt column)")
print(f"  40 time = official combine 40-yard dash (forty column)")

# Check combine.parquet structure
print(f"\n  ── COMBINE DATA STRUCTURE ──")
print(f"  Total RB records in combine: {len(combine_rb)}")
print(f"  Columns used: wt (weight), forty (40 time)")
print(f"  Year range: {combine_rb['season'].min():.0f} - {combine_rb['season'].max():.0f}")

# Check for duplicates
dupes = combine_rb.groupby(['name_norm', 'season']).size()
dupes_multi = dupes[dupes > 1]
if len(dupes_multi) > 0:
    print(f"  WARNING: {len(dupes_multi)} duplicate combine entries found:")
    for (name, season), count in dupes_multi.items():
        print(f"    {name} ({int(season)}): {count} entries")

# How many RBs are in combine but NOT matched?
matched_names = set(rb_eval[rb_eval['weight'].notna()]['name_norm'].values)
all_names = set(rb_eval['name_norm'].values)
unmatched = all_names - matched_names

# Try to find them in combine with fuzzy matching
print(f"\n  ── UNMATCHED PLAYERS ({len(unmatched)}) ──")
print(f"  These RBs from the backtest did NOT match to combine data:")
unmatched_details = rb_eval[rb_eval['weight'].isna()].sort_values('pick')
for _, row in unmatched_details.iterrows():
    # Search combine for close matches
    dy = int(row['draft_year'])
    name = row['name_norm']
    last = name.split()[-1] if name else ''
    close_matches = combine_rb[
        (combine_rb['season'].between(dy-1, dy+1)) &
        (combine_rb['name_norm'].str.contains(last, na=False))
    ] if last and len(last) > 3 else pd.DataFrame()

    close_str = ""
    if len(close_matches) > 0:
        for _, cm in close_matches.iterrows():
            close_str += f" → maybe '{cm['player_name']}' ({int(cm['season'])})"

    print(f"  {row['player_name']:<25} {dy:>4} pick {int(row['pick']):>3} | search: '{name}'{close_str if close_str else ' → NOT IN COMBINE'}")

# Cross-check a few weights against known combine data
print(f"\n  ── SPOT-CHECK: KNOWN COMBINE WEIGHTS ──")
spot_checks = [
    ('Saquon Barkley', 2018, 233),
    ('Derrick Henry', 2016, 247),
    ('Christian McCaffrey', 2017, 202),
    ('Dalvin Cook', 2017, 210),
    ('Jonathan Taylor', 2020, 226),
    ('Najee Harris', 2021, 232),
    ('Breece Hall', 2022, 217),
    ('Bijan Robinson', 2023, 215),
    ('De\'Von Achane', 2023, 188),
    ('Tarik Cohen', 2017, 179),
]

print(f"  {'Player':<25} {'Expected':>10} {'Our Data':>10} {'Match?':>8}")
print("  " + "-" * 55)
for name, dy, expected_wt in spot_checks:
    nn = normalize_name(name)
    match = rb_eval[(rb_eval['name_norm'] == nn) & (rb_eval['draft_year'] == dy)]
    if len(match) > 0:
        actual = match.iloc[0]['weight']
        ok = "✓" if pd.notna(actual) and abs(actual - expected_wt) <= 2 else "✗"
        actual_s = f"{actual:.0f}" if pd.notna(actual) else "MISSING"
        print(f"  {name:<25} {expected_wt:>10} {actual_s:>10} {ok:>8}")
    else:
        print(f"  {name:<25} {expected_wt:>10} {'NOT FOUND':>10}")


print(f"\n\n{'=' * 120}")
print("AUDIT COMPLETE")
print("=" * 120)
