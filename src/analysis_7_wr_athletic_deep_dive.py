"""
Analysis 7: WR Athletic Metric Deep Dive
==========================================
Tests every individual combine metric for WRs after controlling for DC.
Uses PARTIAL correlations (DC removed) so we measure athletic signal
independently from draft capital.

Tests:
  1. Individual metric partial correlations (40, vert, broad, cone, shuttle, bench)
  2. Speed Score for WRs
  3. Height-adjusted speed
  4. Agility metrics (3-cone, shuttle)
  5. BMI and size-speed composites
  6. Athletic percentile thresholds (elite/poor flags)
  7. MNAR-aware tiered imputation
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE FORMULAS
# ============================================================================
def normalize_draft_capital(pick):
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))

def wr_breakout_score(breakout_age, dominator_pct):
    if breakout_age is None or pd.isna(breakout_age):
        if dominator_pct is not None and pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base_score = age_tiers.get(int(breakout_age), 20)
    if dominator_pct is not None and pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
        bonus = max(0, bonus)
    else:
        bonus = 0
    return min(base_score + bonus, 99.9)

def partial_corr(x, y, z):
    """Partial correlation of x and y, controlling for z.
    Returns (r_partial, p_value, n).
    """
    df = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    if len(df) < 15:
        return np.nan, np.nan, len(df)
    # Residualize x on z
    from numpy.polynomial.polynomial import polyfit
    coef_xz = np.polyfit(df['z'], df['x'], 1)
    resid_x = df['x'] - np.polyval(coef_xz, df['z'])
    # Residualize y on z
    coef_yz = np.polyfit(df['z'], df['y'], 1)
    resid_y = df['y'] - np.polyval(coef_yz, df['z'])
    r, p = pearsonr(resid_x, resid_y)
    return r, p, len(df)

def corr_safe(x, y):
    valid = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(valid) < 10:
        return np.nan, np.nan, len(valid)
    r, p = pearsonr(valid['x'], valid['y'])
    return r, p, len(valid)

def height_to_inches(ht_str):
    """Convert '6-2' format to inches."""
    if pd.isna(ht_str) or not isinstance(ht_str, str):
        return np.nan
    try:
        parts = ht_str.split('-')
        return int(parts[0]) * 12 + int(parts[1])
    except:
        return np.nan

# ============================================================================
# LOAD AND MERGE DATA
# ============================================================================
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
wr_bt = pd.read_csv('data/wr_backtest_expanded_final.csv')
combine = pd.read_parquet('data/nflverse/combine.parquet')

# Filter combine to WRs 2015-2025
wr_combine = combine[(combine['pos'] == 'WR') &
                      (combine['draft_year'] >= 2015) &
                      (combine['draft_year'] <= 2025)].copy()
wr_combine['height_in'] = wr_combine['ht'].apply(height_to_inches)
wr_combine = wr_combine.rename(columns={'draft_ovr': 'pick_combine'})

# Build master WR dataset
wr = outcomes[outcomes['position'] == 'WR'].merge(
    wr_bt[['player_name', 'pick', 'round', 'RAS', 'breakout_age', 'peak_dominator', 'draft_year']],
    on=['player_name', 'draft_year'], how='inner', suffixes=('', '_bt'))
wr['pick'] = wr['pick'].fillna(wr['pick_bt'])

# Merge combine metrics
wr = wr.merge(
    wr_combine[['player_name', 'draft_year', 'wt', 'forty', 'vertical', 'broad_jump',
                 'cone', 'shuttle', 'bench', 'height_in']],
    on=['player_name', 'draft_year'], how='left')

# Compute core scores
wr['dc_score'] = wr['pick'].apply(normalize_draft_capital)
wr['prod_score'] = wr.apply(lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)
wr['has_ras'] = wr['RAS'].notna()

# Assign rounds
wr['round'] = pd.cut(wr['pick'], bins=[0, 32, 64, 100, 135, 175, 224, 260],
                       labels=[1, 2, 3, 4, 5, 6, 7]).astype(float)

# Baseline: DC + Breakout only (no athletic)
wr['slap_baseline'] = 0.7647 * wr['dc_score'] + 0.2353 * wr['prod_score']

OUTCOMES = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']

print(f"Master WR dataset: {len(wr)} players")
print(f"Combine metrics coverage:")
for c in ['wt', 'forty', 'vertical', 'broad_jump', 'cone', 'shuttle', 'bench', 'height_in', 'RAS']:
    n = wr[c].notna().sum()
    print(f"  {c:>12}: {n:>3}/{len(wr)} ({n/len(wr)*100:.1f}%)")

# ============================================================================
# BASELINE: DC + Breakout only
# ============================================================================
print("\n" + "=" * 130)
print("BASELINE: DC + Breakout (76.5/23.5/0) — No athletic data")
print("=" * 130)
print(f"{'Outcome':<20} {'r':>8} {'p-value':>12} {'N':>5}")
print("-" * 50)
for out in OUTCOMES:
    r, p, n = corr_safe(wr['slap_baseline'], wr[out])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"{out:<20} {r:>8.4f} {p:>12.6f} {n:>5} {sig}")

# ============================================================================
# TEST 1: Individual Metric Partial Correlations
# ============================================================================
print("\n" + "=" * 130)
print("TEST 1: INDIVIDUAL METRIC PARTIAL CORRELATIONS (controlling for DC)")
print("=" * 130)
print("Each metric's correlation with outcomes AFTER removing draft capital's effect.")
print("This isolates the athletic signal from the draft capital signal.\n")

metrics = {
    'forty':      ('40-yard dash (lower=better)', True),   # True = flip sign (lower is better)
    'vertical':   ('Vertical jump (higher=better)', False),
    'broad_jump': ('Broad jump (higher=better)', False),
    'cone':       ('3-cone drill (lower=better)', True),
    'shuttle':    ('Shuttle (lower=better)', True),
    'bench':      ('Bench press (higher=better)', False),
    'RAS':        ('RAS composite (higher=better)', False),
}

print(f"{'Metric':<35} {'N':>5}  {'hit24 r_p':>12} {'p':>8}  {'hit12 r_p':>12} {'p':>8}  "
      f"{'3yr_ppg r_p':>12} {'p':>8}  {'car_ppg r_p':>12} {'p':>8}")
print("-" * 140)

for metric, (desc, flip) in metrics.items():
    valid = wr[wr[metric].notna()].copy()
    n = len(valid)
    if n < 15:
        print(f"{desc:<35} {n:>5}  {'N/A':>12} {'':>8}  {'N/A':>12} {'':>8}  {'N/A':>12} {'':>8}  {'N/A':>12} {'':>8}")
        continue

    vals = valid[metric] * (-1 if flip else 1)  # Flip so positive = better for all

    line = f"{desc:<35} {n:>5}"
    for out in OUTCOMES:
        r, p, nn = partial_corr(vals, valid[out], valid['dc_score'])
        sig = '***' if pd.notna(p) and p < 0.001 else '**' if pd.notna(p) and p < 0.01 else '*' if pd.notna(p) and p < 0.05 else ''
        line += f"  {r:>+11.4f} {p:>7.4f}{sig}"
    print(line)

# Also show raw (non-partial) for comparison
print(f"\n  Raw correlations (NOT controlling for DC) for reference:")
print(f"  {'Metric':<35} {'N':>5}  {'hit24 r':>10}  {'hit12 r':>10}  {'3yr_ppg r':>10}  {'car_ppg r':>10}")
print("  " + "-" * 95)
for metric, (desc, flip) in metrics.items():
    valid = wr[wr[metric].notna()].copy()
    vals = valid[metric] * (-1 if flip else 1)
    n = len(valid)
    if n < 15:
        continue
    line = f"  {desc:<35} {n:>5}"
    for out in OUTCOMES:
        r, p, nn = corr_safe(vals, valid[out])
        sig = '***' if pd.notna(p) and p < 0.001 else '**' if pd.notna(p) and p < 0.01 else '*' if pd.notna(p) and p < 0.05 else ''
        line += f"  {r:>+9.4f}{sig}"
    print(line)

# ============================================================================
# TEST 2: Speed Score for WRs
# ============================================================================
print("\n" + "=" * 130)
print("TEST 2: SPEED SCORE FOR WRs (weight × 200 / forty^4)")
print("=" * 130)

wr['speed_score'] = wr.apply(
    lambda r: (r['wt'] * 200) / (r['forty'] ** 4) if pd.notna(r['wt']) and pd.notna(r['forty']) and r['forty'] > 0 else np.nan,
    axis=1)

n_ss = wr['speed_score'].notna().sum()
print(f"Coverage: {n_ss}/{len(wr)} ({n_ss/len(wr)*100:.1f}%)")
print(f"Range: {wr['speed_score'].min():.1f} - {wr['speed_score'].max():.1f}")
print(f"Mean: {wr['speed_score'].mean():.1f}, Median: {wr['speed_score'].median():.1f}")

print(f"\n  Partial correlation (controlling for DC):")
print(f"  {'Outcome':<20} {'r_partial':>10} {'p':>10} {'N':>5}")
print("  " + "-" * 50)
for out in OUTCOMES:
    r, p, n = partial_corr(wr['speed_score'], wr[out], wr['dc_score'])
    sig = '***' if pd.notna(p) and p < 0.001 else '**' if pd.notna(p) and p < 0.01 else '*' if pd.notna(p) and p < 0.05 else ''
    print(f"  {out:<20} {r:>+10.4f} {p:>10.4f} {n:>5} {sig}")

# Full SLAP with speed score
ss_avg = wr['speed_score'].mean()
ss_min, ss_max = wr['speed_score'].min(), wr['speed_score'].max()
wr['ss_norm'] = wr['speed_score'].apply(
    lambda x: ((x - ss_min) / (ss_max - ss_min)) * 100 if pd.notna(x) else 50)

print(f"\n  Full SLAP with Speed Score (various weights) vs baseline:")
for w_dc, w_prod, w_ath, label in [
    (0.7647, 0.2353, 0.0, "76.5/23.5/0 (no athletic) BASELINE"),
    (0.70, 0.20, 0.10, "70/20/10 with Speed Score"),
    (0.65, 0.20, 0.15, "65/20/15 with Speed Score"),
    (0.72, 0.23, 0.05, "72/23/5 with Speed Score"),
]:
    wr['slap_test'] = w_dc * wr['dc_score'] + w_prod * wr['prod_score'] + w_ath * wr['ss_norm']
    rs = []
    for out in OUTCOMES:
        r, p, n = corr_safe(wr['slap_test'], wr[out])
        rs.append(r)
    avg = np.mean(rs)
    print(f"  {label:<45} hit24={rs[0]:.4f} hit12={rs[1]:.4f} 3yr={rs[2]:.4f} car={rs[3]:.4f} avg={avg:.4f}")

# ============================================================================
# TEST 3: Height-Adjusted Speed
# ============================================================================
print("\n" + "=" * 130)
print("TEST 3: HEIGHT-ADJUSTED SPEED")
print("=" * 130)

# Height-speed interaction: taller + faster = rare and valuable
# Score = height_inches * (1/forty) — bigger number = taller AND faster
wr['height_speed'] = wr.apply(
    lambda r: r['height_in'] * (1.0 / r['forty']) if pd.notna(r['height_in']) and pd.notna(r['forty']) and r['forty'] > 0 else np.nan,
    axis=1)

# Also: 40-time relative to height expectation
# Expected 40 = baseline + height_penalty (taller = slower expected)
# Residual = expected - actual (positive = faster than expected for height)
ht_forty = wr[['height_in', 'forty']].dropna()
if len(ht_forty) > 20:
    coef = np.polyfit(ht_forty['height_in'], ht_forty['forty'], 1)
    wr['forty_vs_height'] = wr.apply(
        lambda r: (np.polyval(coef, r['height_in']) - r['forty']) if pd.notna(r['height_in']) and pd.notna(r['forty']) else np.nan,
        axis=1)
    print(f"Height-40 regression: 40 = {coef[0]:.4f} * height + {coef[1]:.4f}")
    print(f"  (Each inch of height adds {coef[0]*1000:.1f}ms to expected 40-time)")

n_hs = wr['height_speed'].notna().sum()
n_fvh = wr['forty_vs_height'].notna().sum()
print(f"\nCoverage: height_speed={n_hs}, forty_vs_height={n_fvh}")

print(f"\n  Partial correlations (controlling for DC):")
print(f"  {'Metric':<30} {'N':>5}  {'hit24':>10} {'hit12':>10} {'3yr_ppg':>10} {'car_ppg':>10}")
print("  " + "-" * 80)

for metric_name, metric_col in [
    ('Height alone', 'height_in'),
    ('40-time alone (flipped)', None),
    ('Height × Speed', 'height_speed'),
    ('40 vs height expectation', 'forty_vs_height'),
    ('Speed Score', 'speed_score'),
]:
    if metric_col is None:
        # Special handling for flipped forty
        vals = -wr['forty']
        metric_col_use = 'forty'
    else:
        vals = wr[metric_col]
        metric_col_use = metric_col

    valid_mask = vals.notna()
    n = valid_mask.sum()
    line = f"  {metric_name:<30} {n:>5}"
    for out in OUTCOMES:
        if metric_col is None:
            r, p, nn = partial_corr(-wr['forty'], wr[out], wr['dc_score'])
        else:
            r, p, nn = partial_corr(wr[metric_col], wr[out], wr['dc_score'])
        line += f" {r:>+10.4f}" if pd.notna(r) else f" {'N/A':>10}"
    print(line)

# ============================================================================
# TEST 4: Agility Metrics (3-cone, shuttle)
# ============================================================================
print("\n" + "=" * 130)
print("TEST 4: AGILITY METRICS (3-cone, shuttle)")
print("=" * 130)
print("Route running translates to NFL. Do agility drills predict WR success?")

# Agility composite: average of normalized cone + shuttle
cone_valid = wr['cone'].dropna()
shuttle_valid = wr['shuttle'].dropna()

if len(cone_valid) > 10 and len(shuttle_valid) > 10:
    # Normalize each to 0-100 (lower = better, so flip)
    cone_min, cone_max = cone_valid.min(), cone_valid.max()
    shuttle_min, shuttle_max = shuttle_valid.min(), shuttle_valid.max()

    wr['cone_norm'] = wr['cone'].apply(
        lambda x: (1 - (x - cone_min) / (cone_max - cone_min)) * 100 if pd.notna(x) else np.nan)
    wr['shuttle_norm'] = wr['shuttle'].apply(
        lambda x: (1 - (x - shuttle_min) / (shuttle_max - shuttle_min)) * 100 if pd.notna(x) else np.nan)

    # Agility composite (where both available)
    wr['agility'] = wr.apply(
        lambda r: (r['cone_norm'] + r['shuttle_norm']) / 2
                  if pd.notna(r['cone_norm']) and pd.notna(r['shuttle_norm']) else np.nan,
        axis=1)

n_cone = wr['cone'].notna().sum()
n_shuttle = wr['shuttle'].notna().sum()
n_agility = wr['agility'].notna().sum()
print(f"Coverage: 3-cone={n_cone}, shuttle={n_shuttle}, both (agility)={n_agility}")

print(f"\n  Partial correlations (controlling for DC):")
print(f"  {'Metric':<30} {'N':>5}  {'hit24':>10} {'hit12':>10} {'3yr_ppg':>10} {'car_ppg':>10}")
print("  " + "-" * 80)

for metric_name, metric_col, flip in [
    ('3-cone (flipped)', 'cone', True),
    ('Shuttle (flipped)', 'shuttle', True),
    ('Agility composite', 'agility', False),
]:
    vals = -wr[metric_col] if flip else wr[metric_col]
    n = wr[metric_col].notna().sum()
    line = f"  {metric_name:<30} {n:>5}"
    for out in OUTCOMES:
        if flip:
            r, p, nn = partial_corr(-wr[metric_col], wr[out], wr['dc_score'])
        else:
            r, p, nn = partial_corr(wr[metric_col], wr[out], wr['dc_score'])
        sig = '***' if pd.notna(p) and p < 0.001 else '**' if pd.notna(p) and p < 0.01 else '*' if pd.notna(p) and p < 0.05 else ''
        line += f" {r:>+9.4f}{sig}" if pd.notna(r) else f" {'N/A':>10}"
    print(line)

# Test agility in SLAP formula
if n_agility > 50:
    agility_avg = wr['agility'].mean()
    wr['agility_filled'] = wr['agility'].fillna(agility_avg)
    print(f"\n  Full SLAP with Agility composite (various weights) vs baseline:")
    for w_dc, w_prod, w_ath, label in [
        (0.7647, 0.2353, 0.0, "76.5/23.5/0 (no athletic) BASELINE"),
        (0.70, 0.20, 0.10, "70/20/10 with Agility"),
        (0.72, 0.23, 0.05, "72/23/5 with Agility"),
    ]:
        wr['slap_test'] = w_dc * wr['dc_score'] + w_prod * wr['prod_score'] + w_ath * wr['agility_filled']
        rs = []
        for out in OUTCOMES:
            r, p, n = corr_safe(wr['slap_test'], wr[out])
            rs.append(r)
        avg = np.mean(rs)
        print(f"  {label:<45} hit24={rs[0]:.4f} hit12={rs[1]:.4f} 3yr={rs[2]:.4f} car={rs[3]:.4f} avg={avg:.4f}")

# ============================================================================
# TEST 5: BMI and Size-Speed Composite
# ============================================================================
print("\n" + "=" * 130)
print("TEST 5: BMI AND SIZE-SPEED COMPOSITE")
print("=" * 130)

wr['bmi'] = wr.apply(
    lambda r: (r['wt'] / (r['height_in'] ** 2)) * 703 if pd.notna(r['wt']) and pd.notna(r['height_in']) and r['height_in'] > 0 else np.nan,
    axis=1)

# Size-speed composite: BMI * (1/forty) * scaling
wr['size_speed'] = wr.apply(
    lambda r: r['bmi'] * (1.0 / r['forty']) if pd.notna(r['bmi']) and pd.notna(r['forty']) and r['forty'] > 0 else np.nan,
    axis=1)

# SPARQ-like composite: 40 + vertical + broad (standardized, weighted)
# Standardize each metric
for col in ['forty', 'vertical', 'broad_jump']:
    valid = wr[col].dropna()
    if len(valid) > 10:
        wr[f'{col}_z'] = (wr[col] - valid.mean()) / valid.std()

wr['sparq_like'] = wr.apply(
    lambda r: -r['forty_z'] + r['vertical_z'] + r['broad_jump_z']  # negative forty because lower=better
              if all(pd.notna(r[f'{c}_z']) for c in ['forty', 'vertical', 'broad_jump']) else np.nan,
    axis=1)

n_bmi = wr['bmi'].notna().sum()
n_ss2 = wr['size_speed'].notna().sum()
n_sparq = wr['sparq_like'].notna().sum()
print(f"Coverage: BMI={n_bmi}, size-speed={n_ss2}, SPARQ-like={n_sparq}")

print(f"\n  Partial correlations (controlling for DC):")
print(f"  {'Metric':<30} {'N':>5}  {'hit24':>10} {'hit12':>10} {'3yr_ppg':>10} {'car_ppg':>10}")
print("  " + "-" * 80)

for metric_name, metric_col in [
    ('BMI', 'bmi'),
    ('Size-Speed (BMI/forty)', 'size_speed'),
    ('SPARQ-like (40+vert+broad)', 'sparq_like'),
    ('Speed Score (wt*200/40^4)', 'speed_score'),
]:
    n = wr[metric_col].notna().sum()
    line = f"  {metric_name:<30} {n:>5}"
    for out in OUTCOMES:
        r, p, nn = partial_corr(wr[metric_col], wr[out], wr['dc_score'])
        sig = '***' if pd.notna(p) and p < 0.001 else '**' if pd.notna(p) and p < 0.01 else '*' if pd.notna(p) and p < 0.05 else ''
        line += f" {r:>+9.4f}{sig}" if pd.notna(r) else f" {'N/A':>10}"
    print(line)

# ============================================================================
# TEST 6: Athletic Percentile Thresholds
# ============================================================================
print("\n" + "=" * 130)
print("TEST 6: ATHLETIC PERCENTILE THRESHOLDS")
print("=" * 130)
print("Does being elite (>80th pctile) or poor (<20th pctile) in any metric matter?")

# Compute percentiles for each metric
pctile_metrics = ['forty', 'vertical', 'broad_jump', 'cone', 'shuttle', 'speed_score']

for metric in pctile_metrics:
    valid = wr[metric].dropna()
    if len(valid) < 20:
        continue
    p80 = valid.quantile(0.80)
    p20 = valid.quantile(0.20)

    # For forty/cone/shuttle, lower is better — so elite is BELOW p20, poor is ABOVE p80
    if metric in ['forty', 'cone', 'shuttle']:
        wr[f'{metric}_elite'] = (wr[metric] <= p20).astype(float)
        wr[f'{metric}_poor'] = (wr[metric] >= p80).astype(float)
    else:
        wr[f'{metric}_elite'] = (wr[metric] >= p80).astype(float)
        wr[f'{metric}_poor'] = (wr[metric] <= p20).astype(float)

    # Set to NaN where original is NaN
    wr.loc[wr[metric].isna(), f'{metric}_elite'] = np.nan
    wr.loc[wr[metric].isna(), f'{metric}_poor'] = np.nan

# "Any elite" flag: elite in at least one metric
elite_cols = [f'{m}_elite' for m in pctile_metrics if f'{m}_elite' in wr.columns]
poor_cols = [f'{m}_poor' for m in pctile_metrics if f'{m}_poor' in wr.columns]

# Need at least one metric to evaluate
wr['any_elite'] = wr[elite_cols].max(axis=1)  # 1 if elite in any
wr['any_poor'] = wr[poor_cols].max(axis=1)    # 1 if poor in any
wr['no_poor'] = 1 - wr['any_poor']           # 1 if NOT poor in any

print(f"\n  ELITE flags (>80th percentile in metric):")
print(f"  {'Metric':<25} {'N tested':>8} {'N elite':>8} {'Elite Hit24%':>13} {'Non-elite Hit24%':>17} {'Gap':>6}")
print("  " + "-" * 85)

for metric in pctile_metrics:
    col = f'{metric}_elite'
    if col not in wr.columns:
        continue
    tested = wr[wr[col].notna()]
    elite = tested[tested[col] == 1]
    non_elite = tested[tested[col] == 0]
    if len(elite) > 0 and len(non_elite) > 0:
        e_h24 = elite['hit24'].mean() * 100
        ne_h24 = non_elite['hit24'].mean() * 100
        print(f"  {metric:<25} {len(tested):>8} {len(elite):>8} {e_h24:>12.1f}% {ne_h24:>16.1f}% {e_h24-ne_h24:>+5.1f}")

print(f"\n  POOR flags (<20th percentile in metric):")
print(f"  {'Metric':<25} {'N tested':>8} {'N poor':>8} {'Poor Hit24%':>12} {'Non-poor Hit24%':>16} {'Gap':>6}")
print("  " + "-" * 85)

for metric in pctile_metrics:
    col = f'{metric}_poor'
    if col not in wr.columns:
        continue
    tested = wr[wr[col].notna()]
    poor = tested[tested[col] == 1]
    non_poor = tested[tested[col] == 0]
    if len(poor) > 0 and len(non_poor) > 0:
        p_h24 = poor['hit24'].mean() * 100
        np_h24 = non_poor['hit24'].mean() * 100
        print(f"  {metric:<25} {len(tested):>8} {len(poor):>8} {p_h24:>11.1f}% {np_h24:>15.1f}% {p_h24-np_h24:>+5.1f}")

# Composite flags
print(f"\n  Composite flags:")
print(f"  {'Flag':<30} {'N':>5} {'Hit24%':>8} {'Hit12%':>8} {'3yr PPG':>9} {'Car PPG':>9}")
print("  " + "-" * 75)

for flag_name, flag_col in [
    ('Any elite metric', 'any_elite'),
    ('Any poor metric', 'any_poor'),
    ('No poor metrics', 'no_poor'),
]:
    for val in [1, 0]:
        grp = wr[wr[flag_col] == val]
        if len(grp) == 0:
            continue
        label = f"{flag_name}={val}"
        print(f"  {label:<30} {len(grp):>5} {grp['hit24'].mean()*100:>7.1f}% "
              f"{grp['hit12'].mean()*100:>7.1f}% {grp['first_3yr_ppg'].mean():>9.1f} "
              f"{grp['career_ppg'].mean():>9.1f}")

# Partial correlations for threshold flags
print(f"\n  Partial correlations of threshold flags (controlling for DC):")
print(f"  {'Flag':<30} {'N':>5}  {'hit24':>10} {'hit12':>10} {'3yr_ppg':>10} {'car_ppg':>10}")
print("  " + "-" * 80)

for flag_name, flag_col in [
    ('Any elite metric', 'any_elite'),
    ('Any poor metric', 'any_poor'),
    ('No poor metrics', 'no_poor'),
]:
    n = wr[flag_col].notna().sum()
    line = f"  {flag_name:<30} {n:>5}"
    for out in OUTCOMES:
        r, p, nn = partial_corr(wr[flag_col], wr[out], wr['dc_score'])
        sig = '***' if pd.notna(p) and p < 0.001 else '**' if pd.notna(p) and p < 0.01 else '*' if pd.notna(p) and p < 0.05 else ''
        line += f" {r:>+9.4f}{sig}" if pd.notna(r) else f" {'N/A':>10}"
    print(line)

# Full SLAP test with "no poor" flag
wr['no_poor_scaled'] = wr['no_poor'] * 100
print(f"\n  Full SLAP with 'No poor metrics' flag (various weights) vs baseline:")
for w_dc, w_prod, w_ath, label in [
    (0.7647, 0.2353, 0.0, "76.5/23.5/0 (no athletic) BASELINE"),
    (0.70, 0.20, 0.10, "70/20/10 with NoPoor flag"),
    (0.72, 0.23, 0.05, "72/23/5 with NoPoor flag"),
]:
    wr['slap_test'] = w_dc * wr['dc_score'] + w_prod * wr['prod_score'] + w_ath * wr['no_poor_scaled']
    rs = []
    for out in OUTCOMES:
        r, p, n = corr_safe(wr['slap_test'], wr[out])
        rs.append(r)
    avg = np.mean(rs)
    print(f"  {label:<45} hit24={rs[0]:.4f} hit12={rs[1]:.4f} 3yr={rs[2]:.4f} car={rs[3]:.4f} avg={avg:.4f}")

# ============================================================================
# TEST 7: MNAR-Aware Tiered Imputation
# ============================================================================
print("\n" + "=" * 130)
print("TEST 7: MNAR-AWARE TIERED IMPUTATION")
print("=" * 130)
print("If has RAS: use it. If missing Rd 1-2: assign 60th pctile. If missing Rd 3+: assign 40th pctile.")

ras_valid = wr.loc[wr['RAS'].notna(), 'RAS']
p60 = ras_valid.quantile(0.60)
p40 = ras_valid.quantile(0.40)
print(f"RAS 60th percentile: {p60:.2f}, 40th percentile: {p40:.2f}")

def mnar_ras(row):
    if pd.notna(row['RAS']):
        return row['RAS']
    elif row['round'] <= 2:
        return p60  # Assume good athlete who skipped
    else:
        return p40  # Assume below-average who skipped

wr['ras_mnar'] = wr.apply(mnar_ras, axis=1) * 10  # Scale to 0-100

# Also test more aggressive version
p75 = ras_valid.quantile(0.75)
p25 = ras_valid.quantile(0.25)
def mnar_ras_aggressive(row):
    if pd.notna(row['RAS']):
        return row['RAS']
    elif row['round'] <= 2:
        return p75
    else:
        return p25

wr['ras_mnar_agg'] = wr.apply(mnar_ras_aggressive, axis=1) * 10

print(f"\n  Full SLAP with MNAR-aware RAS (various approaches) vs baseline:")
for approach, ras_col in [('MNAR moderate (60th/40th)', 'ras_mnar'),
                           ('MNAR aggressive (75th/25th)', 'ras_mnar_agg')]:
    for w_dc, w_prod, w_ath, label in [
        (0.65, 0.20, 0.15, f"65/20/15 {approach}"),
        (0.70, 0.20, 0.10, f"70/20/10 {approach}"),
        (0.72, 0.23, 0.05, f"72/23/5 {approach}"),
    ]:
        wr['slap_test'] = w_dc * wr['dc_score'] + w_prod * wr['prod_score'] + w_ath * wr[ras_col]
        rs = []
        for out in OUTCOMES:
            r, p, n = corr_safe(wr['slap_test'], wr[out])
            rs.append(r)
        avg = np.mean(rs)
        print(f"  {label:<55} hit24={rs[0]:.4f} hit12={rs[1]:.4f} 3yr={rs[2]:.4f} car={rs[3]:.4f} avg={avg:.4f}")

# Baseline for comparison
wr['slap_test'] = 0.7647 * wr['dc_score'] + 0.2353 * wr['prod_score']
rs_base = [corr_safe(wr['slap_test'], wr[out])[0] for out in OUTCOMES]
avg_base = np.mean(rs_base)
print(f"  {'76.5/23.5/0 (no athletic) BASELINE':<55} hit24={rs_base[0]:.4f} hit12={rs_base[1]:.4f} 3yr={rs_base[2]:.4f} car={rs_base[3]:.4f} avg={avg_base:.4f}")

# ============================================================================
# MASTER SUMMARY
# ============================================================================
print("\n" + "=" * 130)
print("MASTER SUMMARY: DOES ANYTHING ATHLETIC BEAT DC+BREAKOUT ONLY?")
print("=" * 130)

baseline_avg = avg_base
print(f"\nBaseline (DC+Breakout, no athletic): avg r = {baseline_avg:.4f}")
print(f"\n{'Approach':<55} {'Avg r':>8} {'vs Baseline':>12} {'Verdict'}")
print("-" * 85)

tests = []

# Collect all tests
for w_dc, w_prod, w_ath, ras_col, label in [
    (0.70, 0.20, 0.10, 'ss_norm', 'Speed Score (70/20/10)'),
    (0.72, 0.23, 0.05, 'ss_norm', 'Speed Score (72/23/5)'),
    (0.70, 0.20, 0.10, 'agility_filled', 'Agility composite (70/20/10)'),
    (0.72, 0.23, 0.05, 'agility_filled', 'Agility composite (72/23/5)'),
    (0.70, 0.20, 0.10, 'no_poor_scaled', 'No-poor flag (70/20/10)'),
    (0.72, 0.23, 0.05, 'no_poor_scaled', 'No-poor flag (72/23/5)'),
    (0.70, 0.20, 0.10, 'ras_mnar', 'MNAR-aware RAS (70/20/10)'),
    (0.72, 0.23, 0.05, 'ras_mnar', 'MNAR-aware RAS (72/23/5)'),
    (0.70, 0.20, 0.10, 'ras_mnar_agg', 'MNAR-aggressive RAS (70/20/10)'),
    (0.72, 0.23, 0.05, 'ras_mnar_agg', 'MNAR-aggressive RAS (72/23/5)'),
]:
    if ras_col not in wr.columns:
        continue
    wr['slap_test'] = w_dc * wr['dc_score'] + w_prod * wr['prod_score'] + w_ath * wr[ras_col]
    rs = [corr_safe(wr['slap_test'], wr[out])[0] for out in OUTCOMES]
    avg = np.mean(rs)
    tests.append((label, avg))

# Also add RAS current
wr['ras_current'] = wr['RAS'].fillna(wr['RAS'].mean()) * 10
for w_dc, w_prod, w_ath, label in [
    (0.65, 0.20, 0.15, 'Current RAS (65/20/15)'),
    (0.70, 0.20, 0.10, 'Current RAS (70/20/10)'),
]:
    wr['slap_test'] = w_dc * wr['dc_score'] + w_prod * wr['prod_score'] + w_ath * wr['ras_current']
    rs = [corr_safe(wr['slap_test'], wr[out])[0] for out in OUTCOMES]
    avg = np.mean(rs)
    tests.append((label, avg))

# Sort by avg r descending
tests.sort(key=lambda x: -x[1])

for label, avg in tests:
    delta = avg - baseline_avg
    verdict = "BEATS baseline" if delta > 0.001 else "MATCHES baseline" if delta > -0.001 else "LOSES to baseline"
    print(f"{label:<55} {avg:>8.4f} {delta:>+12.4f} {verdict}")

print(f"\n{'DC+Breakout only (BASELINE)':<55} {baseline_avg:>8.4f} {'---':>12}")

print("\n" + "=" * 130)
print("END OF ANALYSIS 7: WR ATHLETIC METRIC DEEP DIVE")
print("=" * 130)
