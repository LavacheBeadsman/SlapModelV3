"""
WR Weight & Size-Speed Composite Analysis
==========================================
Priority-weighted objective: first_3yr_ppg=40%, hit24=25%, hit12=20%, career_ppg=15%

Test 1: Raw weight as standalone variable
Test 2: Size-speed composite = weight × (4.50 / 40_time)
Test 3: 6-component optimization with best size/athletic metric
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
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))

def wr_breakout_score(breakout_age, dominator_pct):
    if pd.isna(breakout_age) or breakout_age >= 99:
        if pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        else:
            return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base_score = age_tiers.get(int(breakout_age), 20)
    if pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
        bonus = max(0, bonus)
    else:
        bonus = 0
    return min(base_score + bonus, 99.9)

def partial_correlation(x, y, covariates):
    """OLS residuals method for partial correlation."""
    data = pd.DataFrame({'x': x.values if hasattr(x, 'values') else x,
                          'y': y.values if hasattr(y, 'values') else y})
    for i, cov in enumerate(covariates):
        data[f'cov_{i}'] = cov.values if hasattr(cov, 'values') else cov
    data = data.dropna()
    n = len(data)
    if n < 10:
        return np.nan, np.nan, n
    cov_cols = [c for c in data.columns if c.startswith('cov_')]
    if not cov_cols:
        r, p = stats.pearsonr(data['x'], data['y'])
        return r, p, n
    X = np.column_stack([np.ones(n), data[cov_cols].values])
    try:
        beta_x = np.linalg.lstsq(X, data['x'].values, rcond=None)[0]
        resid_x = data['x'].values - X @ beta_x
        beta_y = np.linalg.lstsq(X, data['y'].values, rcond=None)[0]
        resid_y = data['y'].values - X @ beta_y
        r, p = stats.pearsonr(resid_x, resid_y)
        return r, p, n
    except:
        return np.nan, np.nan, n


# ============================================================================
# DATA LOADING
# ============================================================================

print("=" * 110)
print("WR WEIGHT & SIZE-SPEED COMPOSITE ANALYSIS")
print("Priority weights: first_3yr_ppg=40% | hit24=25% | hit12=20% | career_ppg=15%")
print("=" * 110)

# Load core data
wr = pd.read_csv('data/wr_backtest_all_components.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
tm = pd.read_csv('data/wr_teammate_scores.csv')

# Load combine data from nflverse parquet
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine_wr = combine[combine['pos'] == 'WR'][['player_name', 'season', 'ht', 'wt', 'forty']].copy()
combine_wr.rename(columns={'season': 'draft_year'}, inplace=True)

# Parse height to inches
def parse_height(h):
    if pd.isna(h):
        return np.nan
    parts = str(h).split('-')
    if len(parts) == 2:
        try:
            return int(parts[0]) * 12 + int(parts[1])
        except:
            return np.nan
    return np.nan

combine_wr['height_inches'] = combine_wr['ht'].apply(parse_height)
combine_wr.drop(columns=['ht'], inplace=True)

# Merge outcomes
out_wr = outcomes[outcomes['position'] == 'WR'][
    ['player_name', 'draft_year', 'first_3yr_ppg', 'career_ppg']
].copy()
wr = wr.merge(out_wr, on=['player_name', 'draft_year'], how='left')

# Merge teammate scores
tm_lookup = dict(zip(zip(tm['player_name'], tm['draft_year']), tm['avg_teammate_dc']))
wr['teammate_dc'] = wr.apply(lambda x: tm_lookup.get((x['player_name'], x['draft_year']), 0), axis=1)

# Merge combine data (weight, forty, height)
# Try exact name + draft_year match first, then fuzzy
wr = wr.merge(combine_wr, on=['player_name', 'draft_year'], how='left')

# Check how many matched
matched = wr['wt'].notna().sum()
print(f"\nDirect name match: {matched}/{len(wr)} WRs got combine data")

# For unmatched, try fuzzy matching (last name + first initial + year)
unmatched = wr[wr['wt'].isna()].copy()
if len(unmatched) > 0:
    import re
    def normalize_name(name):
        n = str(name).lower().strip()
        n = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv)$', '', n).strip()
        return n

    # Known name mappings
    name_map = {
        'Tank Dell': 'Nathaniel Dell',
        'Bub Means': 'Jerrod Means',
        'Hollywood Brown': 'Marquise Brown',
        'Kadarius Toney': 'Kadarius Toney',
        'DK Metcalf': 'D.K. Metcalf',
        'DJ Chark': 'D.J. Chark',
        'DJ Moore': 'D.J. Moore',
        'KJ Osborn': 'K.J. Osborn',
        'JJ Arcega-Whiteside': 'J.J. Arcega-Whiteside',
        'KJ Hamler': 'K.J. Hamler',
        'AJ Brown': 'A.J. Brown',
        'N\'Keal Harry': "N'Keal Harry",
    }

    fuzzy_matches = 0
    for idx, row in unmatched.iterrows():
        pname = row['player_name']
        dy = row['draft_year']

        # Try mapped name
        alt_name = name_map.get(pname, pname)
        cand = combine_wr[combine_wr['draft_year'] == dy]

        # Exact match with mapped name
        match = cand[cand['player_name'] == alt_name]
        if len(match) == 0:
            # Last name + first initial
            pn = normalize_name(pname)
            parts = pn.split()
            if len(parts) >= 2:
                for _, crow in cand.iterrows():
                    cn = normalize_name(crow['player_name'])
                    cparts = cn.split()
                    if len(cparts) >= 2:
                        if parts[-1] == cparts[-1] and parts[0][0] == cparts[0][0]:
                            match = pd.DataFrame([crow])
                            break
        if len(match) > 0:
            wr.loc[idx, 'wt'] = match.iloc[0]['wt']
            wr.loc[idx, 'forty'] = match.iloc[0]['forty']
            wr.loc[idx, 'height_inches'] = match.iloc[0]['height_inches']
            fuzzy_matches += 1

    total_matched = wr['wt'].notna().sum()
    print(f"After fuzzy matching: {total_matched}/{len(wr)} WRs ({fuzzy_matches} fuzzy)")

# Compute component scores
wr['s_dc'] = wr['pick'].apply(dc_score)
wr['s_breakout'] = wr.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)
wr['s_teammate'] = wr['teammate_dc'].clip(0, 100)
wr['s_declare'] = wr['early_declare'] * 100

# Enhanced breakout with +5 rushing bonus
rush_flag = (wr['rush_yards'] >= 20).astype(int)
wr['s_bo_rush'] = (wr['s_breakout'] + rush_flag * 5).clip(0, 99.9)

# Size-speed composite: weight × (4.50 / 40_time)
wr['size_speed'] = np.where(
    wr['wt'].notna() & wr['forty'].notna() & (wr['forty'] > 0),
    wr['wt'] * (4.50 / wr['forty']),
    np.nan)

# HaSS = (weight × 200) / (40_time^4) × (height / 73.0)
wr['hass'] = np.where(
    wr['wt'].notna() & wr['forty'].notna() & wr['height_inches'].notna() & (wr['forty'] > 0),
    (wr['wt'] * 200) / (wr['forty'] ** 4) * (wr['height_inches'] / 73.0),
    np.nan)

# Speed Score = (weight × 200) / (40_time^4)
wr['speed_score'] = np.where(
    wr['wt'].notna() & wr['forty'].notna() & (wr['forty'] > 0),
    (wr['wt'] * 200) / (wr['forty'] ** 4),
    np.nan)

# BMI approximation
wr['bmi'] = np.where(
    wr['wt'].notna() & wr['height_inches'].notna() & (wr['height_inches'] > 0),
    (wr['wt'] * 703) / (wr['height_inches'] ** 2),
    np.nan)

# Eval sample (2015-2024 with outcomes)
wr_eval = wr[wr['hit24'].notna() & wr['draft_year'].between(2015, 2024)].copy()
print(f"\nEval sample: {len(wr_eval)} WRs")

all_outcomes = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
outcome_weights = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}

# Coverage report
print(f"\n{'Metric':<20} {'Available':>10} {'Coverage':>10}")
print("-" * 45)
for col, label in [('wt', 'Weight'), ('forty', '40 time'),
                    ('size_speed', 'Size-Speed'), ('hass', 'HaSS'),
                    ('speed_score', 'Speed Score'), ('RAS', 'RAS'),
                    ('height_inches', 'Height'), ('bmi', 'BMI')]:
    n = wr_eval[col].notna().sum()
    print(f"  {label:<18} {n:>8}/{len(wr_eval)}   {n/len(wr_eval)*100:>5.1f}%")

for out in all_outcomes:
    n = wr_eval[out].notna().sum()
    print(f"  {out:<18} {n:>8}/{len(wr_eval)}   {n/len(wr_eval)*100:>5.1f}%")


# ============================================================================
# TEST 1: RAW WEIGHT AS STANDALONE VARIABLE
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 1: RAW WEIGHT AS STANDALONE VARIABLE")
print("=" * 110)

# -- 1a: Partial correlations at 2 control levels --
print(f"\n{'─' * 80}")
print("1a. PARTIAL CORRELATIONS — Weight vs all 4 outcomes")
print(f"{'─' * 80}")

control_levels = [
    ("DC only", ['s_dc']),
    ("DC + Breakout", ['s_dc', 's_breakout']),
]

for ctrl_label, ctrl_cols in control_levels:
    print(f"\n  Controlling for {ctrl_label}:")
    print(f"  {'Outcome':<18} {'r':>8} {'p':>8} {'N':>6} {'Sig':>6}")
    print("  " + "-" * 50)

    weighted_sum = 0
    total_w = 0
    for out in all_outcomes:
        covs = [wr_eval[c] for c in ctrl_cols]
        r, p, n = partial_correlation(wr_eval['wt'], wr_eval[out], covs)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {out:<18} {r:>+.4f}  {p:>8.4f} {n:>6} {sig:>6}")
        if not np.isnan(r):
            weighted_sum += outcome_weights[out] * r
            total_w += outcome_weights[out]
    pri_avg = weighted_sum / total_w if total_w > 0 else np.nan
    print(f"  {'PRI-WEIGHTED AVG':<18} {pri_avg:>+.4f}")

# -- 1b: Within-round analysis --
print(f"\n\n{'─' * 80}")
print("1b. WITHIN-ROUND ANALYSIS — Does weight separate hits from misses?")
print(f"{'─' * 80}")

round_groups = [
    ("Round 1", [1]),
    ("Round 2", [2]),
    ("Rounds 3-4", [3, 4]),
    ("Rounds 5-7", [5, 6, 7]),
]

for rg_label, rounds in round_groups:
    sub = wr_eval[(wr_eval['round'].isin(rounds)) & wr_eval['wt'].notna()].copy()
    hits = sub[sub['hit24'] == 1]
    misses = sub[sub['hit24'] == 0]

    if len(hits) >= 3 and len(misses) >= 3:
        t, p = stats.ttest_ind(hits['wt'], misses['wt'])
        sig = " ***" if p < 0.05 else ""
        print(f"\n  {rg_label} (N={len(sub)}, hits={len(hits)}, misses={len(misses)}):")
        print(f"    Hit24=1 avg weight: {hits['wt'].mean():.1f} lbs (SD={hits['wt'].std():.1f})")
        print(f"    Hit24=0 avg weight: {misses['wt'].mean():.1f} lbs (SD={misses['wt'].std():.1f})")
        print(f"    Difference: {hits['wt'].mean() - misses['wt'].mean():>+.1f} lbs  (p={p:.3f}){sig}")

        # Also show first_3yr_ppg if available
        sub_3yr = sub[sub['first_3yr_ppg'].notna()]
        if len(sub_3yr) >= 10:
            r, p3 = stats.pearsonr(sub_3yr['wt'], sub_3yr['first_3yr_ppg'])
            print(f"    Weight vs first_3yr_ppg correlation: r={r:+.3f} (p={p3:.3f}, N={len(sub_3yr)})")
    else:
        print(f"\n  {rg_label}: Too few hits or misses for analysis (hits={len(hits)}, misses={len(misses)})")

# -- 1c: Overall hit vs miss weight --
print(f"\n\n{'─' * 80}")
print("1c. OVERALL: Average weight for hit24=1 vs hit24=0")
print(f"{'─' * 80}")

valid_wt = wr_eval[wr_eval['wt'].notna()].copy()
hits = valid_wt[valid_wt['hit24'] == 1]
misses = valid_wt[valid_wt['hit24'] == 0]
t, p = stats.ttest_ind(hits['wt'], misses['wt'])
print(f"\n  Hit24=1: avg weight = {hits['wt'].mean():.1f} lbs (N={len(hits)}, SD={hits['wt'].std():.1f})")
print(f"  Hit24=0: avg weight = {misses['wt'].mean():.1f} lbs (N={len(misses)}, SD={misses['wt'].std():.1f})")
print(f"  Difference: {hits['wt'].mean() - misses['wt'].mean():>+.1f} lbs  (t={t:.2f}, p={p:.4f})")

# -- 1d: Hit24 rate by weight bucket --
print(f"\n\n{'─' * 80}")
print("1d. HIT24 RATE BY WEIGHT BUCKET")
print(f"{'─' * 80}")

bins = [0, 185, 200, 215, 999]
labels = ['Under 185', '185-199', '200-214', '215+']
valid_wt['wt_bucket'] = pd.cut(valid_wt['wt'], bins=bins, labels=labels, right=False)

print(f"\n  {'Bucket':<14} {'Count':>6} {'Hits':>6} {'Hit Rate':>10} {'Avg 3yr PPG':>14} {'N(3yr)':>8}")
print("  " + "-" * 65)

for bucket in labels:
    sub = valid_wt[valid_wt['wt_bucket'] == bucket]
    n = len(sub)
    nh = int(sub['hit24'].sum())
    rate = nh / n * 100 if n > 0 else 0
    sub_3yr = sub[sub['first_3yr_ppg'].notna()]
    avg_ppg = sub_3yr['first_3yr_ppg'].mean() if len(sub_3yr) > 0 else np.nan
    ppg_str = f"{avg_ppg:.2f}" if not np.isnan(avg_ppg) else "N/A"
    print(f"  {bucket:<14} {n:>6} {nh:>6} {rate:>9.1f}% {ppg_str:>14} {len(sub_3yr):>8}")

# -- 1e: Same with first_3yr_ppg buckets --
print(f"\n\n{'─' * 80}")
print("1e. FIRST_3YR_PPG BY WEIGHT BUCKET (full breakdown)")
print(f"{'─' * 80}")

valid_3yr = valid_wt[valid_wt['first_3yr_ppg'].notna()].copy()
valid_3yr['wt_bucket'] = pd.cut(valid_3yr['wt'], bins=bins, labels=labels, right=False)

print(f"\n  {'Bucket':<14} {'N':>6} {'Mean PPG':>10} {'Median PPG':>12} {'Std':>8} {'Min':>8} {'Max':>8}")
print("  " + "-" * 70)

for bucket in labels:
    sub = valid_3yr[valid_3yr['wt_bucket'] == bucket]
    if len(sub) > 0:
        print(f"  {bucket:<14} {len(sub):>6} {sub['first_3yr_ppg'].mean():>10.2f} "
              f"{sub['first_3yr_ppg'].median():>12.2f} {sub['first_3yr_ppg'].std():>8.2f} "
              f"{sub['first_3yr_ppg'].min():>8.2f} {sub['first_3yr_ppg'].max():>8.2f}")

# T-test: 200+ vs under 200
heavy = valid_3yr[valid_3yr['wt'] >= 200]
light = valid_3yr[valid_3yr['wt'] < 200]
if len(heavy) >= 5 and len(light) >= 5:
    t, p = stats.ttest_ind(heavy['first_3yr_ppg'], light['first_3yr_ppg'])
    print(f"\n  200+ lbs ({len(heavy)}) vs Under 200 ({len(light)}):")
    print(f"    Mean PPG: {heavy['first_3yr_ppg'].mean():.2f} vs {light['first_3yr_ppg'].mean():.2f}")
    print(f"    Difference: {heavy['first_3yr_ppg'].mean() - light['first_3yr_ppg'].mean():>+.2f} (p={p:.4f})")


# ============================================================================
# TEST 2: SIZE-SPEED COMPOSITE
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 2: SIZE-SPEED COMPOSITE = weight × (4.50 / 40_time)")
print("=" * 110)

# Coverage
has_ss = wr_eval['size_speed'].notna().sum()
has_ras = wr_eval['RAS'].notna().sum()
has_hass = wr_eval['hass'].notna().sum()
has_wt = wr_eval['wt'].notna().sum()
has_spd = wr_eval['speed_score'].notna().sum()
print(f"\n  Coverage: Size-Speed={has_ss}, RAS={has_ras}, HaSS={has_hass}, "
      f"Weight={has_wt}, Speed Score={has_spd}")

# -- 2a: Partial correlations for size-speed --
print(f"\n{'─' * 80}")
print("2a. PARTIAL CORRELATIONS — Size-Speed vs all 4 outcomes")
print(f"{'─' * 80}")

for ctrl_label, ctrl_cols in control_levels:
    print(f"\n  Controlling for {ctrl_label}:")
    print(f"  {'Outcome':<18} {'r':>8} {'p':>8} {'N':>6} {'Sig':>6}")
    print("  " + "-" * 50)

    weighted_sum = 0
    total_w = 0
    for out in all_outcomes:
        covs = [wr_eval[c] for c in ctrl_cols]
        r, p, n = partial_correlation(wr_eval['size_speed'], wr_eval[out], covs)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {out:<18} {r:>+.4f}  {p:>8.4f} {n:>6} {sig:>6}")
        if not np.isnan(r):
            weighted_sum += outcome_weights[out] * r
            total_w += outcome_weights[out]
    pri_avg = weighted_sum / total_w if total_w > 0 else np.nan
    print(f"  {'PRI-WEIGHTED AVG':<18} {pri_avg:>+.4f}")


# -- 2b: HEAD-TO-HEAD — All 5 athletic metrics --
print(f"\n\n{'─' * 80}")
print("2b. HEAD-TO-HEAD — All athletic/size metrics")
print("    Partial correlations controlling for DC, then DC + Breakout")
print(f"{'─' * 80}")

metrics = [
    ('wt', 'Raw Weight'),
    ('size_speed', 'Size-Speed (wt×4.50/40)'),
    ('RAS', 'RAS (0-10)'),
    ('hass', 'HaSS'),
    ('speed_score', 'Speed Score'),
    ('bmi', 'BMI'),
]

for ctrl_label, ctrl_cols in control_levels:
    print(f"\n  Controlling for {ctrl_label}:")
    print(f"  {'Metric':<28}", end="")
    for out in all_outcomes:
        print(f"  {'r('+out+')':>14}", end="")
    print(f"  {'PRI-AVG':>10} {'N':>6}")
    print("  " + "-" * 100)

    for col, label in metrics:
        weighted_sum = 0
        total_w = 0
        row = f"  {label:<28}"
        for out in all_outcomes:
            covs = [wr_eval[c] for c in ctrl_cols]
            r, p, n = partial_correlation(wr_eval[col], wr_eval[out], covs)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f"  {r:>+.4f}{sig:<3} ({p:.2f})"
            if not np.isnan(r):
                weighted_sum += outcome_weights[out] * r
                total_w += outcome_weights[out]
        pri_avg = weighted_sum / total_w if total_w > 0 else np.nan
        row += f"  {pri_avg:>+.4f}  {n:>4}"
        print(row)


# -- 2c: Focus on first_3yr_ppg (most important outcome) --
print(f"\n\n{'─' * 80}")
print("2c. FIRST_3YR_PPG FOCUS — Which athletic metric predicts 3-year PPG best?")
print(f"{'─' * 80}")

print(f"\n  {'Metric':<28} {'Raw r':>8} {'Partial(DC)':>13} {'Partial(DC+BO)':>16} {'N':>6}")
print("  " + "-" * 75)

for col, label in metrics:
    # Raw correlation
    valid = wr_eval[[col, 'first_3yr_ppg']].dropna()
    raw_r = np.nan
    if len(valid) >= 10:
        raw_r, _ = stats.pearsonr(valid[col], valid['first_3yr_ppg'])

    # Partial (DC)
    r_dc, p_dc, n_dc = partial_correlation(wr_eval[col], wr_eval['first_3yr_ppg'], [wr_eval['s_dc']])
    sig_dc = "*" if p_dc < 0.05 else ""

    # Partial (DC + BO)
    r_dcbo, p_dcbo, n_dcbo = partial_correlation(
        wr_eval[col], wr_eval['first_3yr_ppg'], [wr_eval['s_dc'], wr_eval['s_breakout']])
    sig_dcbo = "*" if p_dcbo < 0.05 else ""

    print(f"  {label:<28} {raw_r:>+.4f}   {r_dc:>+.4f}{sig_dc:<1} (p={p_dc:.3f}) "
          f" {r_dcbo:>+.4f}{sig_dcbo:<1} (p={p_dcbo:.3f})  {n_dcbo:>4}")


# -- 2d: Size-speed distribution for hits vs misses --
print(f"\n\n{'─' * 80}")
print("2d. SIZE-SPEED SCORE — Hits vs Misses")
print(f"{'─' * 80}")

valid_ss = wr_eval[wr_eval['size_speed'].notna()].copy()
hits = valid_ss[valid_ss['hit24'] == 1]
misses = valid_ss[valid_ss['hit24'] == 0]

if len(hits) >= 3 and len(misses) >= 3:
    t, p = stats.ttest_ind(hits['size_speed'], misses['size_speed'])
    print(f"\n  Hit24=1: avg size-speed = {hits['size_speed'].mean():.1f} (N={len(hits)}, SD={hits['size_speed'].std():.1f})")
    print(f"  Hit24=0: avg size-speed = {misses['size_speed'].mean():.1f} (N={len(misses)}, SD={misses['size_speed'].std():.1f})")
    print(f"  Difference: {hits['size_speed'].mean() - misses['size_speed'].mean():>+.1f} (p={p:.4f})")

# Size-speed by round
print(f"\n  Within-round hit vs miss:")
for rg_label, rounds in round_groups:
    sub = valid_ss[valid_ss['round'].isin(rounds)]
    h = sub[sub['hit24'] == 1]
    m = sub[sub['hit24'] == 0]
    if len(h) >= 2 and len(m) >= 2:
        diff = h['size_speed'].mean() - m['size_speed'].mean()
        _, p = stats.ttest_ind(h['size_speed'], m['size_speed'])
        sig = " *" if p < 0.05 else ""
        print(f"    {rg_label:<14}: hits={h['size_speed'].mean():.1f} vs misses={m['size_speed'].mean():.1f} "
              f"(diff={diff:>+.1f}, p={p:.3f}){sig}")


# ============================================================================
# TEST 3: 6-COMPONENT OPTIMIZATION WITH BEST SIZE/ATHLETIC METRIC
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 3: 6-COMPONENT OPTIMIZATION — PRIORITY-WEIGHTED")
print("Which size/athletic metric works best as a 5th (or 6th) component?")
print("=" * 110)

# Normalize each athletic metric to 0-100 scale
for col in ['wt', 'size_speed', 'hass', 'speed_score', 'bmi']:
    valid_vals = wr_eval[col].dropna()
    if len(valid_vals) > 0:
        p1 = valid_vals.quantile(0.01)
        p99 = valid_vals.quantile(0.99)
        if p99 > p1:
            wr_eval[f's_{col}'] = ((wr_eval[col] - p1) / (p99 - p1) * 100).clip(0, 100)
        else:
            wr_eval[f's_{col}'] = 50
        wr_eval[f's_{col}'] = wr_eval[f's_{col}'].fillna(50)  # Impute missing with average

# RAS already on 0-10 scale -> convert to 0-100
wr_eval['s_ras'] = (wr_eval['RAS'] * 10).fillna(50)  # Impute missing with 50


# -- 3a: Try each athletic metric as 5th component in the 4-comp model --
print(f"\n{'─' * 80}")
print("3a. 5-COMPONENT MODELS: Base 4 + each athletic metric")
print("    DC (55%+) / Enhanced Breakout (12%+) / Teammate (2-8%) / ED (2-8%) / Athletic (2-8%)")
print(f"{'─' * 80}")

athletic_options = [
    ('s_wt', 'Raw Weight'),
    ('s_size_speed', 'Size-Speed'),
    ('s_ras', 'RAS'),
    ('s_hass', 'HaSS'),
    ('s_speed_score', 'Speed Score'),
    ('s_bmi', 'BMI'),
]

# Build outcome data for base 4-comp model first
base_cols = ['s_dc', 's_bo_rush', 's_teammate', 's_declare']

# For each model, build outcome data and optimize
model_results = {}

# BASELINE: 4-component model (same constraints as requested for the 4 base components)
print(f"\n  BASELINE: 4-component (DC / BO+Rush / Teammate / ED)")
print(f"  {'─' * 76}")

base_outcome_data = {}
for out in all_outcomes:
    valid = wr_eval[base_cols + [out]].dropna(subset=[out]).copy()
    base_outcome_data[out] = {'X': valid[base_cols].values, 'y': valid[out].values}

def neg_pri_base(weights):
    total = 0
    for out, w in outcome_weights.items():
        X = base_outcome_data[out]['X']
        y = base_outcome_data[out]['y']
        s = X @ weights
        if np.std(s) > 1e-10:
            r = np.corrcoef(s, y)[0, 1]
            total += w * r
    return -total

# Base bounds: DC 55%+, BO 12%+, TM 2-8%, ED 2-8%
bounds_base = [(0.55, 0.90), (0.12, 0.35), (0.02, 0.08), (0.02, 0.08)]
constraints_base = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]

starts_base = [
    [0.75, 0.16, 0.04, 0.05],
    [0.70, 0.20, 0.05, 0.05],
    [0.65, 0.25, 0.05, 0.05],
    [0.80, 0.12, 0.04, 0.04],
    [0.60, 0.30, 0.05, 0.05],
    [0.76, 0.16, 0.04, 0.04],
    [0.55, 0.35, 0.05, 0.05],
]

best_base_r = -999
best_base_w = None
for x0 in starts_base:
    try:
        res = minimize(neg_pri_base, x0, method='SLSQP', bounds=bounds_base,
                       constraints=constraints_base, options={'maxiter': 1000})
        r_opt = -res.fun
        if r_opt > best_base_r:
            best_base_r = r_opt
            best_base_w = res.x
    except:
        pass

print(f"  Weights: DC={best_base_w[0]:.0%} / BO+Rush={best_base_w[1]:.0%} / "
      f"TM={best_base_w[2]:.0%} / ED={best_base_w[3]:.0%}")
print(f"  Priority-weighted avg r: {best_base_r:+.4f}")

# Show individual correlations
print(f"\n  Individual outcome correlations:")
for out in all_outcomes:
    X = base_outcome_data[out]['X']
    y = base_outcome_data[out]['y']
    s = X @ best_base_w
    r = np.corrcoef(s, y)[0, 1]
    tag = f" (weight: {outcome_weights[out]:.0%})" if out in outcome_weights else ""
    print(f"    {out:<15}: r = {r:+.4f}  (N={len(y)}){tag}")

model_results['Baseline: 4-comp'] = {
    'weights': best_base_w,
    'labels': ['DC', 'BO+Rush', 'Teammate', 'EarlyDec'],
    'pri_avg': best_base_r
}

# 5-component models with each athletic metric
for ath_col, ath_label in athletic_options:
    cols_5 = base_cols + [ath_col]
    print(f"\n  5-COMP: Base + {ath_label}")
    print(f"  {'─' * 76}")

    # Build outcome data
    ath_outcome_data = {}
    for out in all_outcomes:
        valid = wr_eval[cols_5 + [out]].dropna(subset=[out]).copy()
        ath_outcome_data[out] = {'X': valid[cols_5].values, 'y': valid[out].values}

    def make_neg_pri(od):
        def fn(weights):
            total = 0
            for out, w in outcome_weights.items():
                X = od[out]['X']
                y = od[out]['y']
                s = X @ weights
                if np.std(s) > 1e-10:
                    r = np.corrcoef(s, y)[0, 1]
                    total += w * r
            return -total
        return fn

    obj_fn = make_neg_pri(ath_outcome_data)

    # Bounds: DC 55%+, BO 12%+, TM 2-8%, ED 2-8%, Athletic 2-8%
    bounds_5 = [(0.55, 0.85), (0.12, 0.30), (0.02, 0.08), (0.02, 0.08), (0.02, 0.08)]
    constraints_5 = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]

    starts_5 = [
        [0.72, 0.15, 0.04, 0.04, 0.05],
        [0.68, 0.18, 0.04, 0.04, 0.06],
        [0.65, 0.20, 0.05, 0.04, 0.06],
        [0.75, 0.13, 0.04, 0.04, 0.04],
        [0.60, 0.25, 0.05, 0.04, 0.06],
        [0.78, 0.12, 0.02, 0.02, 0.06],
        [0.70, 0.16, 0.04, 0.04, 0.06],
    ]

    best_r5 = -999
    best_w5 = None
    for x0 in starts_5:
        try:
            res = minimize(obj_fn, x0, method='SLSQP', bounds=bounds_5,
                           constraints=constraints_5, options={'maxiter': 1000})
            r_opt = -res.fun
            if r_opt > best_r5:
                best_r5 = r_opt
                best_w5 = res.x
        except:
            pass

    print(f"  Weights: DC={best_w5[0]:.0%} / BO+Rush={best_w5[1]:.0%} / "
          f"TM={best_w5[2]:.0%} / ED={best_w5[3]:.0%} / {ath_label}={best_w5[4]:.0%}")
    delta = best_r5 - best_base_r
    print(f"  Priority-weighted avg r: {best_r5:+.4f}  (delta vs 4-comp: {delta:>+.4f})")

    print(f"\n  Individual outcome correlations:")
    for out in all_outcomes:
        X = ath_outcome_data[out]['X']
        y = ath_outcome_data[out]['y']
        s = X @ best_w5
        r = np.corrcoef(s, y)[0, 1]
        tag = f" (weight: {outcome_weights[out]:.0%})" if out in outcome_weights else ""
        print(f"    {out:<15}: r = {r:+.4f}  (N={len(y)}){tag}")

    model_results[f'5-comp + {ath_label}'] = {
        'weights': best_w5,
        'labels': ['DC', 'BO+Rush', 'Teammate', 'EarlyDec', ath_label],
        'pri_avg': best_r5,
        'delta': delta
    }


# -- 3b: Also try without the floor constraints (let optimizer choose freely) --
print(f"\n\n{'─' * 80}")
print("3b. UNCONSTRAINED 5-COMPONENT (athletic min=0%, not 2%)")
print("    If optimizer can drop athletic to 0%, does it?")
print(f"{'─' * 80}")

for ath_col, ath_label in athletic_options:
    cols_5 = base_cols + [ath_col]

    ath_outcome_data = {}
    for out in all_outcomes:
        valid = wr_eval[cols_5 + [out]].dropna(subset=[out]).copy()
        ath_outcome_data[out] = {'X': valid[cols_5].values, 'y': valid[out].values}

    def make_neg_pri2(od):
        def fn(weights):
            total = 0
            for out, w in outcome_weights.items():
                X = od[out]['X']
                y = od[out]['y']
                s = X @ weights
                if np.std(s) > 1e-10:
                    r = np.corrcoef(s, y)[0, 1]
                    total += w * r
            return -total
        return fn

    obj_fn = make_neg_pri2(ath_outcome_data)

    # Unconstrained: athletic can go to 0
    bounds_free = [(0.40, 0.90), (0.05, 0.40), (0.00, 0.15), (0.00, 0.15), (0.00, 0.15)]
    constraints_free = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]

    starts_free = [
        [0.76, 0.16, 0.04, 0.04, 0.00],
        [0.70, 0.18, 0.04, 0.04, 0.04],
        [0.65, 0.20, 0.05, 0.04, 0.06],
        [0.80, 0.12, 0.04, 0.04, 0.00],
        [0.60, 0.25, 0.05, 0.05, 0.05],
    ]

    best_rf = -999
    best_wf = None
    for x0 in starts_free:
        try:
            res = minimize(obj_fn, x0, method='SLSQP', bounds=bounds_free,
                           constraints=constraints_free, options={'maxiter': 1000})
            r_opt = -res.fun
            if r_opt > best_rf:
                best_rf = r_opt
                best_wf = res.x
        except:
            pass

    ath_pct = best_wf[4]
    status = "KEPT" if ath_pct > 0.005 else "DROPPED"
    print(f"  {ath_label:<28}: {ath_pct:>5.1%} ({status})  r={best_rf:+.4f}  "
          f"DC={best_wf[0]:.0%}/BO={best_wf[1]:.0%}/TM={best_wf[2]:.0%}/ED={best_wf[3]:.0%}")


# -- 3c: SUMMARY TABLE --
print(f"\n\n{'=' * 110}")
print("SUMMARY: ALL MODELS RANKED BY PRIORITY-WEIGHTED AVG r")
print("=" * 110)

ranked = sorted(model_results.items(), key=lambda x: x[1]['pri_avg'], reverse=True)
print(f"\n  {'Model':<35} {'PRI-AVG r':>12} {'Delta vs Base':>15} {'Weights':>40}")
print("  " + "-" * 105)

for name, res in ranked:
    delta = res.get('delta', 0)
    delta_str = f"{delta:>+.4f}" if delta != 0 else "    ---"
    wt_str = " / ".join(f"{l}={w:.0%}" for l, w in zip(res['labels'], res['weights']))
    print(f"  {name:<35} {res['pri_avg']:>+.4f}    {delta_str:>12}   {wt_str}")


# -- 3d: Top/bottom 20 by size-speed score with outcomes --
print(f"\n\n{'─' * 80}")
print("TOP 20 WRs BY SIZE-SPEED SCORE (with outcomes)")
print(f"{'─' * 80}")

valid_ss_out = wr_eval[wr_eval['size_speed'].notna()].sort_values('size_speed', ascending=False).head(20)
print(f"\n  {'Player':<25} {'Year':>5} {'Pick':>5} {'Wt':>6} {'40':>6} {'SzSpd':>7} {'Hit24':>6} {'3yr PPG':>9}")
print("  " + "-" * 80)
for _, row in valid_ss_out.iterrows():
    ppg_str = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "N/A"
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} "
          f"{row['wt']:>6.0f} {row['forty']:>6.2f} {row['size_speed']:>7.1f} "
          f"{int(row['hit24']):>6} {ppg_str:>9}")

hits_top20 = int(valid_ss_out['hit24'].sum())
print(f"\n  Top 20 hit rate: {hits_top20}/20 = {hits_top20/20*100:.1f}%")
avg_3yr = valid_ss_out['first_3yr_ppg'].mean()
print(f"  Top 20 avg first_3yr_ppg: {avg_3yr:.2f}" if pd.notna(avg_3yr) else "  Top 20 avg: N/A")

print(f"\n{'─' * 80}")
print("BOTTOM 20 WRs BY SIZE-SPEED SCORE (with outcomes)")
print(f"{'─' * 80}")

valid_ss_bot = wr_eval[wr_eval['size_speed'].notna()].sort_values('size_speed', ascending=True).head(20)
print(f"\n  {'Player':<25} {'Year':>5} {'Pick':>5} {'Wt':>6} {'40':>6} {'SzSpd':>7} {'Hit24':>6} {'3yr PPG':>9}")
print("  " + "-" * 80)
for _, row in valid_ss_bot.iterrows():
    ppg_str = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "N/A"
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} "
          f"{row['wt']:>6.0f} {row['forty']:>6.2f} {row['size_speed']:>7.1f} "
          f"{int(row['hit24']):>6} {ppg_str:>9}")

hits_bot20 = int(valid_ss_bot['hit24'].sum())
print(f"\n  Bottom 20 hit rate: {hits_bot20}/20 = {hits_bot20/20*100:.1f}%")
avg_3yr_bot = valid_ss_bot['first_3yr_ppg'].mean()
print(f"  Bottom 20 avg first_3yr_ppg: {avg_3yr_bot:.2f}" if pd.notna(avg_3yr_bot) else "  Bottom 20 avg: N/A")


print(f"\n\n{'=' * 110}")
print("ANALYSIS COMPLETE")
print("=" * 110)
