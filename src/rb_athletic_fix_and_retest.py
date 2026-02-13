"""
RB Athletic Data — Fix Name Matching, Retest Weight, U-shaped Signal, DC-Adjusted Outcomes
============================================================================================
Fix 1: Name matching bug (suffixes, accents, position mismatches)
Fix 2: Retest with corrected data
Test 3: U-shaped weight signal (versions A-D)
Test 4: DC-adjusted outcomes for all metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq as np_lstsq
from scipy.stats import f as f_dist
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

def normalize_name_old(name):
    """Original buggy version - doesn't strip suffixes."""
    if pd.isna(name):
        return ''
    return str(name).strip().lower().replace('.', '').replace("'", '').replace('-', ' ')

def normalize_name(name):
    """Fixed version - strips suffixes, accents, etc."""
    if pd.isna(name):
        return ''
    s = str(name).strip().lower()
    # Remove accents (common: é→e, ñ→n)
    accent_map = {'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
                  'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a',
                  'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
                  'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o',
                  'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
                  'ñ': 'n'}
    for k, v in accent_map.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    # Strip suffixes (must be at end, after a space)
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s.strip()

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

def partial_corr(df, x_col, y_col, control_cols):
    valid = df[[x_col, y_col] + control_cols].dropna()
    if len(valid) < 20:
        return np.nan, np.nan, len(valid)
    X = np.column_stack([valid[c].values for c in control_cols] + [np.ones(len(valid))])
    beta_x, _, _, _ = np_lstsq(X, valid[x_col].values, rcond=None)
    resid_x = valid[x_col].values - X @ beta_x
    beta_y, _, _, _ = np_lstsq(X, valid[y_col].values, rcond=None)
    resid_y = valid[y_col].values - X @ beta_y
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(valid)

def auc_roc(labels, scores):
    valid = pd.DataFrame({'y': labels, 's': scores}).dropna()
    if len(valid) < 10: return np.nan
    pos = valid[valid['y'] == 1]['s']
    neg = valid[valid['y'] == 0]['s']
    if len(pos) == 0 or len(neg) == 0: return np.nan
    auc_sum = sum((neg < p_val).sum() + 0.5 * (neg == p_val).sum() for p_val in pos)
    return auc_sum / (len(pos) * len(neg))

def brier_score(labels, scores_0_100):
    valid = pd.DataFrame({'y': labels, 's': scores_0_100}).dropna()
    if len(valid) < 10: return np.nan
    prob = valid['s'] / 100
    return ((prob - valid['y']) ** 2).mean()

outcome_cols = ['first_3yr_ppg', 'hit24', 'hit12', 'career_ppg']
outcome_weights_map = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}

# ============================================================================
# FIX 1: LOAD DATA WITH FIXED NAME MATCHING
# ============================================================================

print("=" * 120)
print("FIX 1: NAME MATCHING — BEFORE vs AFTER")
print("=" * 120)

rb = pd.read_csv('data/rb_backtest_with_receiving.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
out_rb = outcomes[outcomes['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
rb = rb.merge(out_rb, on=['player_name', 'draft_year', 'pick'], how='left')
rb['s_dc'] = rb['pick'].apply(dc_score)
rb['s_rec_prod'] = rb.apply(rb_production_score, axis=1)

# Load combine — ALL positions (Gibson was listed as WR)
combine = pd.read_parquet('data/nflverse/combine.parquet')

# OLD matching (buggy)
combine_old = combine[combine['pos'] == 'RB'].copy()
combine_old['name_norm'] = combine_old['player_name'].apply(normalize_name_old)

old_lookup = {}
for _, row in combine_old.iterrows():
    dy = row.get('draft_year', row.get('season'))
    if pd.isna(dy): continue
    key = (row['name_norm'], int(dy))
    old_lookup[key] = {'weight': row['wt'], 'forty': row['forty']}

rb['name_norm_old'] = rb['player_name'].apply(normalize_name_old)
rb['weight_old'] = rb.apply(lambda r: old_lookup.get((r['name_norm_old'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb['forty_old'] = rb.apply(lambda r: old_lookup.get((r['name_norm_old'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb['weight_old'] = pd.to_numeric(rb['weight_old'], errors='coerce')
rb['forty_old'] = pd.to_numeric(rb['forty_old'], errors='coerce')

# NEW matching (fixed):
# 1. Use fixed normalize_name with suffix stripping
# 2. Include WR/TE positions (for position converts like Gibson)
# 3. Handle accented characters
combine_new = combine[combine['pos'].isin(['RB', 'WR', 'FB', 'TE'])].copy()
combine_new['name_norm'] = combine_new['player_name'].apply(normalize_name)

new_lookup = {}
# Prefer RB matches over WR matches
for pos_pref in ['RB', 'FB', 'WR', 'TE']:
    sub = combine_new[combine_new['pos'] == pos_pref]
    for _, row in sub.iterrows():
        dy = row.get('draft_year', row.get('season'))
        if pd.isna(dy): continue
        key = (row['name_norm'], int(dy))
        if key not in new_lookup:  # Don't overwrite RB match with WR
            new_lookup[key] = {'weight': row['wt'], 'forty': row['forty'], 'pos': row['pos'],
                               'orig_name': row['player_name']}

rb['name_norm'] = rb['player_name'].apply(normalize_name)
rb['weight_new'] = rb.apply(lambda r: new_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb['forty_new'] = rb.apply(lambda r: new_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb['match_pos'] = rb.apply(lambda r: new_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('pos', ''), axis=1)
rb['match_name'] = rb.apply(lambda r: new_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('orig_name', ''), axis=1)
rb['weight_new'] = pd.to_numeric(rb['weight_new'], errors='coerce')
rb['forty_new'] = pd.to_numeric(rb['forty_new'], errors='coerce')

# Show every fix
fixes = rb[(rb['weight_old'].isna() & rb['weight_new'].notna()) |
           (rb['forty_old'].isna() & rb['forty_new'].notna())].copy()

n_old_wt = rb['weight_old'].notna().sum()
n_new_wt = rb['weight_new'].notna().sum()
n_old_40 = rb['forty_old'].notna().sum()
n_new_40 = rb['forty_new'].notna().sum()

print(f"\n  Weight: {n_old_wt} → {n_new_wt} (gained {n_new_wt - n_old_wt})")
print(f"  40 time: {n_old_40} → {n_new_40} (gained {n_new_40 - n_old_40})")

print(f"\n  ── EVERY FIX ({len(fixes)} players) ──")
print("  %-25s %4s %4s %-30s %6s %6s %6s %6s %5s %8s" % (
    'Player', 'Year', 'Pick', 'Matched To', 'OldWt', 'NewWt', 'Old40', 'New40', 'hit24', '3yr PPG'))
print("  " + "-" * 130)

for _, row in fixes.sort_values('pick').iterrows():
    old_wt = "%.0f" % row['weight_old'] if pd.notna(row['weight_old']) else "MISS"
    new_wt = "%.0f" % row['weight_new'] if pd.notna(row['weight_new']) else "MISS"
    old_40 = "%.2f" % row['forty_old'] if pd.notna(row['forty_old']) else "MISS"
    new_40 = "%.2f" % row['forty_new'] if pd.notna(row['forty_new']) else "MISS"
    ppg = "%.1f" % row['first_3yr_ppg'] if pd.notna(row['first_3yr_ppg']) else "?"
    h24 = "%d" % row['hit24'] if pd.notna(row['hit24']) else "?"
    mn = row['match_name'] if row['match_name'] else ""
    mp = " (%s)" % row['match_pos'] if row['match_pos'] else ""
    print("  %-25s %4d %4d %-30s %6s %6s %6s %6s %5s %8s" % (
        row['player_name'], int(row['draft_year']), int(row['pick']),
        mn + mp, old_wt, new_wt, old_40, new_40, h24, ppg))

# Still missing after fix
still_miss_wt = rb[rb['weight_new'].isna() & rb['hit24'].notna()]
print(f"\n  Still missing weight after fix: {len(still_miss_wt)}")
still_miss_hits = still_miss_wt[still_miss_wt['hit24'] == 1]
if len(still_miss_hits) > 0:
    print(f"  Hit24=1 players STILL missing weight:")
    for _, row in still_miss_hits.sort_values('pick').iterrows():
        ppg = "%.1f" % row['first_3yr_ppg'] if pd.notna(row['first_3yr_ppg']) else "?"
        print(f"    {row['player_name']:<25} pick {int(row['pick']):>3} | 3yr PPG: {ppg}")

# Use the fixed data going forward
rb['weight'] = rb['weight_new']
rb['forty'] = rb['forty_new']
rb['speed_score'] = rb.apply(lambda r: speed_score_fn(r['weight'], r['forty']), axis=1)
rb['ras'] = rb['RAS']

rb_eval = rb[rb['hit24'].notna()].copy()

# ============================================================================
# FIX 2: RETEST WEIGHT WITH CORRECTED DATA
# ============================================================================

print(f"\n\n{'=' * 120}")
print("FIX 2: WEIGHT PARTIAL CORRELATIONS — BEFORE vs AFTER FIX")
print("=" * 120)

print(f"\n  Weight coverage: OLD {n_old_wt}/223 → NEW {rb_eval['weight'].notna().sum()}/223")

# Partial corr: BEFORE (with old data)
print(f"\n  ── BEFORE (buggy name matching, {n_old_wt} observed) ──")
rb_eval['weight_old'] = rb['weight_old'].loc[rb_eval.index]
for out in outcome_cols:
    r, p, n = partial_corr(rb_eval, 'weight_old', out, ['s_dc'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
    print(f"    partial r(weight, {out:>15} | DC): {r:>+.4f}{sig}  p={p:.4f}  n={n}")

# Partial corr: AFTER
print(f"\n  ── AFTER (fixed name matching, {rb_eval['weight'].notna().sum()} observed) ──")
for out in outcome_cols:
    r, p, n = partial_corr(rb_eval, 'weight', out, ['s_dc'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
    print(f"    partial r(weight, {out:>15} | DC): {r:>+.4f}{sig}  p={p:.4f}  n={n}")

# Also controlling for DC + RYPTPA
avg_rec = rb_eval['s_rec_prod'].mean()
rb_eval['s_rec_prod_f'] = rb_eval['s_rec_prod'].fillna(avg_rec)
print(f"\n  ── AFTER (controlling for DC + RYPTPA) ──")
for out in outcome_cols:
    r, p, n = partial_corr(rb_eval, 'weight', out, ['s_dc', 's_rec_prod_f'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
    print(f"    partial r(weight, {out:>15} | DC,rec): {r:>+.4f}{sig}  p={p:.4f}  n={n}")

# Updated bias check
has_wt = rb_eval[rb_eval['weight'].notna()]
miss_wt = rb_eval[rb_eval['weight'].isna()]
print(f"\n  ── UPDATED BIAS CHECK ──")
print(f"  Has weight: {len(has_wt)} | Avg pick: {has_wt['pick'].mean():.1f} | Hit24: {has_wt['hit24'].mean()*100:.1f}%")
print(f"  Missing:    {len(miss_wt)} | Avg pick: {miss_wt['pick'].mean():.1f} | Hit24: {miss_wt['hit24'].mean()*100:.1f}%")


# ============================================================================
# TEST 3: U-SHAPED WEIGHT SIGNAL
# ============================================================================

print(f"\n\n{'=' * 120}")
print("TEST 3: U-SHAPED WEIGHT SIGNAL")
print("=" * 120)

wt_eval = rb_eval[rb_eval['weight'].notna()].copy()
print(f"  Working sample: {len(wt_eval)} RBs with weight data")

# Version A: Distance from 210
wt_eval['v_dist210'] = abs(wt_eval['weight'] - 210)

# Version B: Quadratic (weight - 210)²
wt_eval['v_quad210'] = (wt_eval['weight'] - 210) ** 2

# Version C: Binary flags
wt_eval['v_light'] = (wt_eval['weight'] < 200).astype(float) * 100
wt_eval['v_heavy'] = (wt_eval['weight'] > 225).astype(float) * 100
wt_eval['v_extreme'] = ((wt_eval['weight'] < 200) | (wt_eval['weight'] > 225)).astype(float) * 100

# Version D: BMI extremes (need height)
combine_ht = combine[combine['pos'].isin(['RB', 'WR', 'FB'])].copy()
combine_ht['name_norm'] = combine_ht['player_name'].apply(normalize_name)
ht_lookup = {}
for _, row in combine_ht.iterrows():
    dy = row.get('draft_year', row.get('season'))
    if pd.isna(dy) or pd.isna(row['ht']): continue
    try:
        ht_val = float(row['ht'])
    except (ValueError, TypeError):
        continue
    key = (row['name_norm'], int(dy))
    if key not in ht_lookup:
        ht_lookup[key] = ht_val

wt_eval['height'] = wt_eval.apply(
    lambda r: ht_lookup.get((r['name_norm'], int(r['draft_year'])), np.nan), axis=1)
wt_eval['height'] = pd.to_numeric(wt_eval['height'], errors='coerce')
wt_eval['bmi'] = wt_eval.apply(
    lambda r: (r['weight'] * 703) / (r['height'] ** 2) if pd.notna(r['weight']) and pd.notna(r['height']) and r['height'] > 0 else np.nan,
    axis=1)
bmi_median = wt_eval['bmi'].median()
wt_eval['v_bmi_extreme'] = abs(wt_eval['bmi'] - bmi_median) if bmi_median else np.nan

# Normalize all to 0-100
for col in ['v_dist210', 'v_quad210', 'v_bmi_extreme']:
    valid = wt_eval[col].notna()
    if valid.sum() > 10:
        wt_eval.loc[valid, col + '_n'] = normalize_0_100(wt_eval.loc[valid, col])
    else:
        wt_eval[col + '_n'] = np.nan

versions = {
    'A: Distance from 210':     'v_dist210',
    'B: (Weight-210)²':         'v_quad210',
    'C1: Light (<200)':         'v_light',
    'C2: Heavy (>225)':         'v_heavy',
    'C3: Extreme (either)':     'v_extreme',
    'D: BMI extreme':           'v_bmi_extreme',
    'Raw weight (linear)':      'weight',
}

print(f"\n  ── PARTIAL CORRELATIONS: Controlling for DC ──")
print("  %-25s %12s %8s %12s %8s %12s %8s %12s %8s" % (
    'Version', 'pr(3yr_ppg)', 'p', 'pr(hit24)', 'p', 'pr(hit12)', 'p', 'pr(career)', 'p'))
print("  " + "-" * 110)

version_results = {}
for label, col in versions.items():
    row_str = "  %-25s" % label
    results = {}
    for out in outcome_cols:
        r, p, n = partial_corr(wt_eval, col, out, ['s_dc'])
        results[f'pr_{out}'] = r
        results[f'pp_{out}'] = p
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
        if np.isnan(r):
            row_str += " %12s %8s" % ("N/A", "")
        else:
            row_str += " %+11.4f%s %7.4f" % (r, sig, p)
    print(row_str)
    version_results[label] = results

print(f"\n  ── PARTIAL CORRELATIONS: Controlling for DC + RYPTPA ──")
wt_eval['s_rec_prod_f'] = wt_eval['s_rec_prod'].fillna(wt_eval['s_rec_prod'].mean())
print("  %-25s %12s %8s %12s %8s %12s %8s %12s %8s" % (
    'Version', 'pr(3yr|+rec)', 'p', 'pr(h24|+rec)', 'p', 'pr(h12|+rec)', 'p', 'pr(car|+rec)', 'p'))
print("  " + "-" * 110)

for label, col in versions.items():
    row_str = "  %-25s" % label
    for out in outcome_cols:
        r, p, n = partial_corr(wt_eval, col, out, ['s_dc', 's_rec_prod_f'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
        if np.isnan(r):
            row_str += " %12s %8s" % ("N/A", "")
        else:
            row_str += " %+11.4f%s %7.4f" % (r, sig, p)
    print(row_str)

# Show hit rates for flagged vs unflagged
print(f"\n  ── HIT RATES AND PPG: FLAGGED vs UNFLAGGED ──")
for label, col, threshold in [
    ('Light (<200 lbs)', 'v_light', 50),
    ('Heavy (>225 lbs)', 'v_heavy', 50),
    ('Extreme (either)', 'v_extreme', 50),
]:
    flagged = wt_eval[wt_eval[col] > threshold]
    unflagged = wt_eval[wt_eval[col] <= threshold]
    f_h24 = flagged['hit24'].mean() * 100 if len(flagged) > 0 else 0
    u_h24 = unflagged['hit24'].mean() * 100
    f_ppg = flagged['first_3yr_ppg'].dropna().mean() if len(flagged[flagged['first_3yr_ppg'].notna()]) > 0 else 0
    u_ppg = unflagged['first_3yr_ppg'].dropna().mean()
    f_pick = flagged['pick'].mean() if len(flagged) > 0 else 0
    u_pick = unflagged['pick'].mean()
    print(f"\n  {label}:")
    print(f"    Flagged:   n={len(flagged):>3} | Hit24={f_h24:>5.1f}% | 3yr PPG={f_ppg:>5.2f} | Avg pick={f_pick:>5.1f}")
    print(f"    Unflagged: n={len(unflagged):>3} | Hit24={u_h24:>5.1f}% | 3yr PPG={u_ppg:>5.2f} | Avg pick={u_pick:>5.1f}")

# Test in 65/35 model if any version shows signal
print(f"\n\n  ── MODEL TEST: Add to DC 60 / RYPTPA 35 / U-shape 5 ──")

def full_12_metrics(slap, df):
    res = {}
    for out in outcome_cols:
        valid = pd.DataFrame({'s': slap, 'o': df[out]}).dropna()
        if len(valid) >= 10:
            r, _ = stats.pearsonr(valid['s'], valid['o'])
            res[f'r_{out}'] = r
    pri = sum(outcome_weights_map[o] * res.get(f'r_{o}', 0) for o in outcome_cols
              if not np.isnan(res.get(f'r_{o}', np.nan)))
    pri_w = sum(outcome_weights_map[o] for o in outcome_cols
                if not np.isnan(res.get(f'r_{o}', np.nan)))
    res['pri_avg'] = pri / pri_w if pri_w > 0 else np.nan
    n_top = max(1, len(df) // 10)
    top = df.loc[slap.nlargest(n_top).index]
    res['top10_hit24'] = top['hit24'].mean() * 100
    res['top10_hit12'] = top['hit12'].mean() * 100
    ppg_top = top[top['first_3yr_ppg'].notna()]
    res['top10_ppg'] = ppg_top['first_3yr_ppg'].mean() if len(ppg_top) > 0 else np.nan
    res['auc_hit24'] = auc_roc(df['hit24'], slap)
    res['auc_hit12'] = auc_roc(df['hit12'], slap)
    res['brier_hit24'] = brier_score(df['hit24'], slap)
    res['brier_hit12'] = brier_score(df['hit12'], slap)
    return res

# Need to use rb_eval (all 223) for model test, imputing for missing weight
avg_wt = rb_eval['weight'].mean()
for col_base in ['v_dist210', 'v_quad210', 'v_light', 'v_heavy', 'v_extreme']:
    rb_eval[col_base] = np.nan
rb_eval.loc[rb_eval['weight'].notna(), 'v_dist210'] = abs(rb_eval.loc[rb_eval['weight'].notna(), 'weight'] - 210)
rb_eval.loc[rb_eval['weight'].notna(), 'v_quad210'] = (rb_eval.loc[rb_eval['weight'].notna(), 'weight'] - 210) ** 2
rb_eval.loc[rb_eval['weight'].notna(), 'v_light'] = (rb_eval.loc[rb_eval['weight'].notna(), 'weight'] < 200).astype(float) * 100
rb_eval.loc[rb_eval['weight'].notna(), 'v_heavy'] = (rb_eval.loc[rb_eval['weight'].notna(), 'weight'] > 225).astype(float) * 100
rb_eval.loc[rb_eval['weight'].notna(), 'v_extreme'] = ((rb_eval.loc[rb_eval['weight'].notna(), 'weight'] < 200) | (rb_eval.loc[rb_eval['weight'].notna(), 'weight'] > 225)).astype(float) * 100

# Normalize and impute
for col in ['v_dist210', 'v_quad210']:
    valid = rb_eval[col].notna()
    if valid.sum() > 10:
        rb_eval.loc[valid, col + '_n'] = normalize_0_100(rb_eval.loc[valid, col])
    avg_v = rb_eval[col + '_n'].mean() if rb_eval.get(col + '_n') is not None else 50
    rb_eval[col + '_n_f'] = rb_eval[col + '_n'].fillna(avg_v) if col + '_n' in rb_eval.columns else 50

# For binary flags, impute 0 (average) for missing
for col in ['v_light', 'v_heavy', 'v_extreme']:
    # Impute the base rate for missing
    avg_flag = rb_eval[col].mean()
    rb_eval[col + '_f'] = rb_eval[col].fillna(avg_flag)

# Also normalize raw weight for model test
rb_eval['weight_n'] = np.nan
valid_w = rb_eval['weight'].notna()
if valid_w.sum() > 10:
    rb_eval.loc[valid_w, 'weight_n'] = normalize_0_100(rb_eval.loc[valid_w, 'weight'])
avg_wn = rb_eval['weight_n'].mean() if rb_eval['weight_n'].notna().any() else 50
rb_eval['weight_n_f'] = rb_eval['weight_n'].fillna(avg_wn)

# Baseline: 65/35
base_slap = rb_eval['s_dc'] * 0.65 + rb_eval['s_rec_prod_f'] * 0.35
base_m = full_12_metrics(base_slap, rb_eval)

model_tests = {
    'Baseline 65/35':             ('s_dc', 0.65, 's_rec_prod_f', 0.35, None, 0),
    '+ Linear weight (5%)':       ('s_dc', 0.60, 's_rec_prod_f', 0.35, 'weight_n_f', 0.05),
    '+ Dist from 210 (5%)':       ('s_dc', 0.60, 's_rec_prod_f', 0.35, 'v_dist210_n_f', 0.05),
    '+ (Wt-210)² (5%)':           ('s_dc', 0.60, 's_rec_prod_f', 0.35, 'v_quad210_n_f', 0.05),
    '+ Extreme flag (5%)':        ('s_dc', 0.60, 's_rec_prod_f', 0.35, 'v_extreme_f', 0.05),
    '+ Heavy flag (5%)':          ('s_dc', 0.60, 's_rec_prod_f', 0.35, 'v_heavy_f', 0.05),
    '+ Light flag (5%)':          ('s_dc', 0.60, 's_rec_prod_f', 0.35, 'v_light_f', 0.05),
}

print("  %-30s %8s %8s %8s %8s %8s %8s %8s %8s %8s" % (
    'Config', 'PRI-AVG', 'r(3yr)', 'r(h24)', 'r(h12)', 'T10%h24', 'T10%h12', 'AUC h24', 'AUC h12', 'Brier24'))
print("  " + "-" * 105)

for label, (dc_col, dc_w, rec_col, rec_w, ath_col, ath_w) in model_tests.items():
    if ath_col:
        slap = rb_eval[dc_col] * dc_w + rb_eval[rec_col] * rec_w + rb_eval[ath_col] * ath_w
    else:
        slap = rb_eval[dc_col] * dc_w + rb_eval[rec_col] * rec_w
    m = full_12_metrics(slap, rb_eval)
    d = m['pri_avg'] - base_m['pri_avg']
    marker = " ◄ BASE" if 'Baseline' in label else " (%+.4f)" % d
    print("  %-30s %+.4f %+.4f %+.4f %+.4f %6.1f%% %6.1f%% %.4f  %.4f  %.4f%s" % (
        label, m['pri_avg'], m.get('r_first_3yr_ppg', 0), m.get('r_hit24', 0),
        m.get('r_hit12', 0), m['top10_hit24'], m['top10_hit12'],
        m['auc_hit24'], m['auc_hit12'], m['brier_hit24'], marker))


# ============================================================================
# TEST 4: DC-ADJUSTED OUTCOMES
# ============================================================================

print(f"\n\n{'=' * 120}")
print("TEST 4: DC-ADJUSTED OUTCOMES")
print("Remove draft capital confound entirely — test what predicts OUTPERFORMANCE of slot")
print("=" * 120)

# Calculate expected PPG by round
print(f"\n  ── EXPECTED first_3yr_ppg BY ROUND ──")
ppg_eval = rb_eval[rb_eval['first_3yr_ppg'].notna()].copy()

print("  %8s %5s %10s %8s %10s %8s" % ('Round', 'n', 'Avg PPG', 'Std', 'Hit24%', 'Hit12%'))
print("  " + "-" * 55)

round_expected = {}
for rd in sorted(ppg_eval['round'].unique()):
    grp = ppg_eval[ppg_eval['round'] == rd]
    avg = grp['first_3yr_ppg'].mean()
    std = grp['first_3yr_ppg'].std()
    h24 = grp['hit24'].mean() * 100
    h12 = grp['hit12'].mean() * 100
    round_expected[rd] = avg
    print("  %8d %5d %10.2f %8.2f %9.1f%% %7.1f%%" % (rd, len(grp), avg, std, h24, h12))

# Also by pick ranges for finer granularity
print(f"\n  ── EXPECTED first_3yr_ppg BY PICK RANGE ──")
pick_ranges = [(1, 15), (16, 32), (33, 64), (65, 100), (101, 150), (151, 200), (201, 260)]
pick_expected = {}
for lo, hi in pick_ranges:
    grp = ppg_eval[(ppg_eval['pick'] >= lo) & (ppg_eval['pick'] <= hi)]
    if len(grp) > 0:
        avg = grp['first_3yr_ppg'].mean()
        pick_expected[(lo, hi)] = avg
        print("  Pick %3d-%3d: n=%3d | Avg PPG: %6.2f | Hit24: %5.1f%%" % (
            lo, hi, len(grp), avg, grp['hit24'].mean() * 100))

# Build DC-adjusted PPG (residual from round average)
rb_eval['expected_ppg'] = rb_eval['round'].map(round_expected)
rb_eval['ppg_over_expected'] = rb_eval['first_3yr_ppg'] - rb_eval['expected_ppg']

# Also DC-adjusted hit24: player hit rate vs round hit rate
round_hit24 = {}
for rd in rb_eval['round'].unique():
    grp = rb_eval[rb_eval['round'] == rd]
    round_hit24[rd] = grp['hit24'].mean()
rb_eval['expected_hit24'] = rb_eval['round'].map(round_hit24)
rb_eval['hit24_over_expected'] = rb_eval['hit24'] - rb_eval['expected_hit24']

# And career_ppg adjusted
ppg_eval_career = rb_eval[rb_eval['career_ppg'].notna()].copy()
round_career = {}
for rd in ppg_eval_career['round'].unique():
    grp = ppg_eval_career[ppg_eval_career['round'] == rd]
    round_career[rd] = grp['career_ppg'].mean()
rb_eval['expected_career'] = rb_eval['round'].map(round_career)
rb_eval['career_over_expected'] = rb_eval['career_ppg'] - rb_eval['expected_career']

dc_adj_outcomes = ['ppg_over_expected', 'hit24_over_expected', 'career_over_expected']

print(f"\n  PPG over expectation: mean={rb_eval['ppg_over_expected'].mean():.4f} (should be ~0)")
print(f"  Positive = outperformed draft slot, negative = underperformed")

# ── Test all athletic metrics against DC-adjusted outcomes ──
print(f"\n\n  ── ATHLETIC METRICS vs DC-ADJUSTED OUTCOMES (raw correlation, no DC control needed) ──")

# Prepare weight versions on full rb_eval
rb_eval['v_dist210_raw'] = abs(rb_eval['weight'] - 210)
rb_eval['v_quad210_raw'] = (rb_eval['weight'] - 210) ** 2

# PFF and production metrics — need to load PFF receiving data
# Load from previous analysis or re-match
season_file_map = {
    2015: 'receiving_summary (2).csv', 2016: 'receiving_summary (3).csv',
    2017: 'receiving_summary (4).csv', 2018: 'receiving_summary (5).csv',
    2019: 'receiving_summary (21).csv', 2020: 'receiving_summary (20).csv',
    2021: 'receiving_summary (19).csv', 2022: 'receiving_summary (18).csv',
    2023: 'receiving_summary (17).csv', 2024: 'receiving_summary (16).csv',
}

pff_recv_all = []
for season, fname in season_file_map.items():
    df = pd.read_csv(f'data/{fname}')
    df['college_season'] = season
    hbs = df[df['position'] == 'HB'].copy()
    hbs['name_norm'] = hbs['player'].apply(normalize_name)
    pff_recv_all.append(hbs)
pff_recv = pd.concat(pff_recv_all, ignore_index=True)

# Match PFF to rb_eval
for col in ['yprr', 'grades_pass_route', 'avg_depth_of_target', 'touchdowns', 'yards', 'player_game_count']:
    rb_eval[f'pff_{col}'] = np.nan

for idx, row in rb_eval.iterrows():
    target_season = row['draft_year'] - 1
    name = row['name_norm']
    matches = pff_recv[(pff_recv['college_season'] == target_season) & (pff_recv['name_norm'] == name)]
    if len(matches) == 0:
        last = name.split()[-1] if name else ''
        if last and len(last) > 3:
            matches = pff_recv[(pff_recv['college_season'] == target_season) &
                               (pff_recv['name_norm'].str.contains(last, na=False))]
            if len(matches) > 1: matches = matches.head(1)
    if len(matches) >= 1:
        m = matches.iloc[0]
        for col in ['yprr', 'grades_pass_route', 'avg_depth_of_target', 'touchdowns', 'yards', 'player_game_count']:
            rb_eval.at[idx, f'pff_{col}'] = pd.to_numeric(m.get(col, np.nan), errors='coerce')

# PFF rushing
pff_rush = pd.read_csv('data/rb_pff_corrected.csv')
pff_rush['name_norm'] = pff_rush['player_name'].apply(normalize_name)
for col in ['yards', 'grades_run', 'elusive_rating']:
    rb_eval[f'rush_{col}'] = np.nan
for idx, row in rb_eval.iterrows():
    matches = pff_rush[(pff_rush['name_norm'] == row['name_norm']) & (pff_rush['draft_year'] == row['draft_year'])]
    if len(matches) >= 1:
        m = matches.iloc[0]
        for col in ['yards', 'grades_run', 'elusive_rating']:
            rb_eval.at[idx, f'rush_{col}'] = pd.to_numeric(m.get(col, np.nan), errors='coerce')

# TD per team pass att
rb_eval['td_per_tpa'] = np.nan
for idx, row in rb_eval.iterrows():
    td = row.get('pff_touchdowns', np.nan)
    tpa = row.get('team_pass_att', np.nan)
    if pd.notna(td) and pd.notna(tpa) and tpa > 0:
        rb_eval.at[idx, 'td_per_tpa'] = td / tpa

# Rush YPG
rb_eval['rush_ypg'] = np.nan
for idx, row in rb_eval.iterrows():
    ry = row.get('rush_yards', np.nan)
    gp = row.get('pff_player_game_count', np.nan)
    if pd.notna(ry) and pd.notna(gp) and gp > 0:
        rb_eval.at[idx, 'rush_ypg'] = ry / gp

all_test_metrics = {
    # Athletic
    'Weight (linear)':       'weight',
    'Dist from 210 (A)':     'v_dist210_raw',
    '(Wt-210)² (B)':         'v_quad210_raw',
    'Light flag (<200)':     'v_light',
    'Heavy flag (>225)':     'v_heavy',
    'Extreme flag':          'v_extreme',
    'Speed Score':           'speed_score',
    'RAS':                   'ras',
    '40 time (inv)':         'forty',
    # Production
    'RYPTPA (current)':      's_rec_prod',
    'PFF YPRR':              'pff_yprr',
    'PFF Route Grade':       'pff_grades_pass_route',
    'TD/Team Pass Att':      'td_per_tpa',
    'Rush YPG':              'rush_ypg',
}

print("\n  %-25s %5s %12s %8s %12s %8s %12s %8s" % (
    'Metric', 'n', 'r(PPG_adj)', 'p', 'r(h24_adj)', 'p', 'r(car_adj)', 'p'))
print("  " + "-" * 90)

for label, col in all_test_metrics.items():
    row_str = "  %-25s" % label
    n_valid = rb_eval[col].notna().sum()
    row_str += " %5d" % n_valid

    for out in dc_adj_outcomes:
        valid = rb_eval[[col, out]].dropna()
        if len(valid) >= 10:
            # For 40 time, invert (lower = better)
            if col == 'forty':
                r, p = stats.pearsonr(-valid[col], valid[out])
            else:
                r, p = stats.pearsonr(valid[col], valid[out])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
            row_str += " %+11.4f%s %7.4f" % (r, sig, p)
        else:
            row_str += " %12s %8s" % ("N/A", "")
    print(row_str)

# Sort by PPG over expected correlation
print(f"\n\n  ── RANKED BY r(PPG over expectation) — what predicts outperformance of draft slot? ──")
ranked = []
for label, col in all_test_metrics.items():
    valid = rb_eval[[col, 'ppg_over_expected']].dropna()
    if len(valid) >= 10:
        if col == 'forty':
            r, p = stats.pearsonr(-valid[col], valid['ppg_over_expected'])
        else:
            r, p = stats.pearsonr(valid[col], valid['ppg_over_expected'])
        ranked.append((label, r, p, len(valid)))
ranked.sort(key=lambda x: -x[1])

print("  %4s %-25s %12s %8s %5s" % ('Rank', 'Metric', 'r(PPG_adj)', 'p', 'n'))
print("  " + "-" * 55)
for i, (label, r, p, n) in enumerate(ranked, 1):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
    cat = "PROD" if label in ['RYPTPA (current)', 'PFF YPRR', 'PFF Route Grade', 'TD/Team Pass Att', 'Rush YPG'] else "ATH"
    print("  %4d %-25s %+11.4f%s %7.4f %5d  [%s]" % (i, label, r, sig, p, n, cat))


# ── Residual PPG by weight bucket (using DC-adjusted) ──
print(f"\n\n  ── PPG OVER EXPECTATION BY WEIGHT BUCKET ──")
wt_ppg = rb_eval[rb_eval['weight'].notna() & rb_eval['ppg_over_expected'].notna()].copy()
buckets = [(185, 195), (195, 200), (200, 205), (205, 210), (210, 215),
           (215, 220), (220, 225), (225, 230), (230, 250)]
print("  %15s %4s %12s %12s %12s" % ('Weight Range', 'n', 'Avg PPG_adj', 'Avg h24_adj', 'Avg career_adj'))
print("  " + "-" * 60)
for lo, hi in buckets:
    bucket = wt_ppg[(wt_ppg['weight'] >= lo) & (wt_ppg['weight'] < hi)]
    if len(bucket) >= 2:
        avg_ppg = bucket['ppg_over_expected'].mean()
        avg_h24 = bucket['hit24_over_expected'].mean()
        avg_car = bucket['career_over_expected'].dropna().mean()
        print("  %6d-%d lbs %4d %+11.2f %+11.3f %+11.2f" % (lo, hi, len(bucket), avg_ppg, avg_h24, avg_car))


print(f"\n\n{'=' * 120}")
print("ANALYSIS COMPLETE")
print("=" * 120)
