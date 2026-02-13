"""
RB Reoptimization — Full outcome breakdown for all candidate configs
=====================================================================
12 metrics × 6 configs. Check if any component helps on ANY outcome.
Also test RAS, Speed Score, and 40 time directly.
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

def speed_score_fn(weight, forty):
    if pd.isna(weight) or pd.isna(forty) or forty <= 0 or weight <= 0:
        return np.nan
    return (weight * 200) / (forty ** 4)

def normalize_name(name):
    if pd.isna(name):
        return ''
    return name.strip().lower().replace('.', '').replace("'", '').replace('-', ' ')

def auc_roc(labels, scores):
    """Manual AUC-ROC."""
    valid = pd.DataFrame({'y': labels, 's': scores}).dropna()
    if len(valid) < 10:
        return np.nan
    pos = valid[valid['y'] == 1]['s']
    neg = valid[valid['y'] == 0]['s']
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    auc_sum = 0
    for p_val in pos:
        auc_sum += (neg < p_val).sum() + 0.5 * (neg == p_val).sum()
    return auc_sum / (len(pos) * len(neg))

def brier_score(labels, scores_0_100):
    """Brier score using SLAP/100 as probability."""
    valid = pd.DataFrame({'y': labels, 's': scores_0_100}).dropna()
    if len(valid) < 10:
        return np.nan
    prob = valid['s'] / 100
    return ((prob - valid['y']) ** 2).mean()

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

# RAS
rb['s_ras'] = rb['RAS'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)
avg_ras = rb['s_ras'].mean()
rb['s_ras_filled'] = rb['s_ras'].fillna(avg_ras)

# Declare
declare = pd.read_csv('data/rb_backtest_with_declare.csv')
declare_map = {}
for _, row in declare.iterrows():
    declare_map[(row['player_name'], row['draft_year'])] = row['declare_status']
rb['declare_status'] = rb.apply(
    lambda r: declare_map.get((r['player_name'], r['draft_year']), np.nan), axis=1)
rb['early_declare'] = (rb['declare_status'] == 'EARLY').astype(int)
rb['s_early_dec'] = rb['early_declare'] * 100

# Combine data
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine_rb = combine[combine['pos'] == 'RB'].copy()
combine_rb['name_norm'] = combine_rb['player_name'].apply(normalize_name)
rb['name_norm'] = rb['player_name'].apply(normalize_name)

combine_lookup = {}
for _, row in combine_rb.iterrows():
    dy = row.get('draft_year')
    if pd.isna(dy):
        dy = row.get('season')
    if pd.isna(dy):
        continue
    key = (row['name_norm'], int(dy))
    combine_lookup[key] = {'weight': row['wt'], 'forty': row['forty']}

rb['weight'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb['forty'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb['speed_score_raw'] = rb.apply(
    lambda r: speed_score_fn(r['weight'], r['forty']), axis=1)

# Normalize all to 0-100
for col, raw_col in [('s_weight', 'weight'), ('s_forty_inv', 'forty'), ('s_speed', 'speed_score_raw')]:
    rb[col] = np.nan
    valid = rb[raw_col].notna()
    if valid.sum() > 10:
        if col == 's_forty_inv':
            # Invert 40: lower is better, so negate before normalizing
            rb.loc[valid, col] = normalize_0_100(-rb.loc[valid, raw_col])
        else:
            rb.loc[valid, col] = normalize_0_100(rb.loc[valid, raw_col])
    avg = rb[col].mean() if rb[col].notna().any() else 50
    rb[f'{col}_filled'] = rb[col].fillna(avg)

# Weight filled
valid_wt = rb['weight'].notna()
rb['s_weight'] = np.nan
if valid_wt.sum() > 10:
    rb.loc[valid_wt, 's_weight'] = normalize_0_100(rb.loc[valid_wt, 'weight'])
avg_wt = rb['s_weight'].mean() if rb['s_weight'].notna().any() else 50
rb['s_weight_filled'] = rb['s_weight'].fillna(avg_wt)

rb_eval = rb[rb['hit24'].notna()].copy()

# ============================================================================
# FULL EVALUATION
# ============================================================================

def compute_slap(row, weights_dict):
    total = 0
    for col, w in weights_dict.items():
        total += row[col] * w
    return min(100, max(0, total))

def full_eval_all(df, label, comp_weights):
    """Comprehensive evaluation with all 12 metrics."""
    df = df.copy()
    df['slap'] = df.apply(lambda r: compute_slap(r, comp_weights), axis=1)

    result = {'label': label}

    # Correlations with each outcome
    for out in outcome_cols:
        valid = df[['slap', out]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid['slap'], valid[out])
            result[f'r_{out}'] = r
        else:
            result[f'r_{out}'] = np.nan

    # Priority-weighted average
    pri_sum = sum(outcome_weights[out] * result.get(f'r_{out}', 0)
                  for out in outcome_cols if not np.isnan(result.get(f'r_{out}', np.nan)))
    pri_total = sum(outcome_weights[out]
                    for out in outcome_cols if not np.isnan(result.get(f'r_{out}', np.nan)))
    result['pri_avg'] = pri_sum / pri_total if pri_total > 0 else np.nan

    # Top decile
    n_top = max(1, len(df) // 10)
    top = df.nlargest(n_top, 'slap')

    result['top10_hit24'] = top['hit24'].mean() * 100
    result['top10_hit12'] = top['hit12'].mean() * 100
    top_ppg = top[top['first_3yr_ppg'].notna()]
    result['top10_ppg'] = top_ppg['first_3yr_ppg'].mean() if len(top_ppg) > 0 else np.nan
    result['top10_n'] = n_top

    # AUC-ROC
    result['auc_hit24'] = auc_roc(df['hit24'], df['slap'])
    result['auc_hit12'] = auc_roc(df['hit12'], df['slap'])

    # Brier
    result['brier_hit24'] = brier_score(df['hit24'], df['slap'])
    result['brier_hit12'] = brier_score(df['hit12'], df['slap'])

    return result


# ============================================================================
# CONFIGS TO TEST
# ============================================================================

configs = [
    ('DC only',                          {'s_dc': 1.00}),
    ('CURRENT: DC/Rec/RAS 50/35/15',    {'s_dc': 0.50, 's_rec_prod_filled': 0.35, 's_ras_filled': 0.15}),
    ('DC/Rec 65/35',                     {'s_dc': 0.65, 's_rec_prod_filled': 0.35}),
    ('DC/Rec/Wt 60/30/10',              {'s_dc': 0.60, 's_rec_prod_filled': 0.30, 's_weight_filled': 0.10}),
    ('DC/Rec/Wt 60/35/5',               {'s_dc': 0.60, 's_rec_prod_filled': 0.35, 's_weight_filled': 0.05}),
    ('DC/Rec/ED 60/35/5',               {'s_dc': 0.60, 's_rec_prod_filled': 0.35, 's_early_dec': 0.05}),
    ('DC/Rec/RAS 60/30/10',             {'s_dc': 0.60, 's_rec_prod_filled': 0.30, 's_ras_filled': 0.10}),
    ('DC/Rec/RAS 55/35/10',             {'s_dc': 0.55, 's_rec_prod_filled': 0.35, 's_ras_filled': 0.10}),
    ('DC/Rec/SS 60/30/10',              {'s_dc': 0.60, 's_rec_prod_filled': 0.30, 's_speed_filled': 0.10}),
    ('DC/Rec/SS 55/35/10',              {'s_dc': 0.55, 's_rec_prod_filled': 0.35, 's_speed_filled': 0.10}),
    ('DC/Rec/40 60/30/10',              {'s_dc': 0.60, 's_rec_prod_filled': 0.30, 's_forty_inv_filled': 0.10}),
    ('DC/Rec/40 55/35/10',              {'s_dc': 0.55, 's_rec_prod_filled': 0.35, 's_forty_inv_filled': 0.10}),
]

all_results = []
for label, weights in configs:
    r = full_eval_all(rb_eval, label, weights)
    all_results.append(r)

# ============================================================================
# DISPLAY: Full 12-metric comparison
# ============================================================================

print("=" * 140)
print("RB REOPTIMIZATION — FULL 12-METRIC COMPARISON ACROSS ALL OUTCOMES")
print(f"Eval sample: {len(rb_eval)} RBs | Top decile = {max(1, len(rb_eval) // 10)} players")
print("=" * 140)

# Find the 65/35 baseline for deltas
baseline = [r for r in all_results if r['label'] == 'DC/Rec 65/35'][0]

metrics = [
    ('r_first_3yr_ppg', 'r(first_3yr_ppg)', True, '+.4f'),
    ('r_hit24',         'r(hit24)',          True, '+.4f'),
    ('r_hit12',         'r(hit12)',          True, '+.4f'),
    ('r_career_ppg',    'r(career_ppg)',     True, '+.4f'),
    ('pri_avg',         'PRI-AVG r',         True, '+.4f'),
    ('top10_hit24',     'Top10% hit24',      True, '.1f'),
    ('top10_hit12',     'Top10% hit12',      True, '.1f'),
    ('top10_ppg',       'Top10% 3yr PPG',    True, '.2f'),
    ('auc_hit24',       'AUC-ROC hit24',     True, '.4f'),
    ('auc_hit12',       'AUC-ROC hit12',     True, '.4f'),
    ('brier_hit24',     'Brier hit24',       False, '.4f'),  # lower is better
    ('brier_hit12',     'Brier hit12',       False, '.4f'),  # lower is better
]

# Print header
header = f"  {'Metric':<22}"
for r in all_results:
    short = r['label'].replace('CURRENT: ', '').replace('DC/Rec/', '').replace('DC/Rec ', '')
    if len(short) > 16:
        short = short[:16]
    header += f" {short:>16}"
print(f"\n{header}")
print("  " + "-" * (22 + 17 * len(all_results)))

# Print each metric row
for key, label, higher_better, fmt in metrics:
    row = f"  {label:<22}"
    vals = []
    for r in all_results:
        v = r.get(key, np.nan)
        vals.append(v)
        if np.isnan(v):
            row += f" {'N/A':>16}"
        elif 'Top10%' in label and 'PPG' not in label:
            row += f" {v:>15.1f}%"
        else:
            row += f" {v:>16.{fmt[-2:]}}" if fmt.startswith('+') else f" {v:>16{fmt}}"
    print(row)

# ============================================================================
# DELTA TABLE: Everything vs DC/Rec 65/35
# ============================================================================

print(f"\n\n{'=' * 140}")
print("DELTAS VS DC/Rec 65/35 BASELINE")
print("Green = improvement, Red = worse. For Brier, negative delta = improvement.")
print("=" * 140)

header2 = f"  {'Metric':<22}"
for r in all_results:
    if r['label'] == 'DC/Rec 65/35':
        continue
    short = r['label'].replace('CURRENT: ', '').replace('DC/Rec/', '').replace('DC/Rec ', '')
    if len(short) > 16:
        short = short[:16]
    header2 += f" {short:>16}"
print(f"\n{header2}")
print("  " + "-" * (22 + 17 * (len(all_results) - 1)))

for key, label, higher_better, fmt in metrics:
    row = f"  {label:<22}"
    base_val = baseline.get(key, np.nan)
    for r in all_results:
        if r['label'] == 'DC/Rec 65/35':
            continue
        v = r.get(key, np.nan)
        if np.isnan(v) or np.isnan(base_val):
            row += f" {'N/A':>16}"
        else:
            delta = v - base_val
            if higher_better:
                marker = "+" if delta > 0.001 else "-" if delta < -0.001 else "="
            else:
                marker = "+" if delta < -0.001 else "-" if delta > 0.001 else "="
            row += f" {delta:>+13.4f} {marker:>1}"
        pass
    print(row)

# ============================================================================
# OUTCOME-BY-OUTCOME WINNER ANALYSIS
# ============================================================================

print(f"\n\n{'=' * 110}")
print("OUTCOME-BY-OUTCOME: WHICH CONFIG WINS EACH METRIC?")
print("=" * 110)

for key, label, higher_better, fmt in metrics:
    vals = [(r['label'], r.get(key, np.nan)) for r in all_results]
    vals = [(l, v) for l, v in vals if not np.isnan(v)]
    if not vals:
        continue
    if higher_better:
        winner = max(vals, key=lambda x: x[1])
    else:
        winner = min(vals, key=lambda x: x[1])

    # Is 65/35 the winner?
    base_val = baseline.get(key, np.nan)
    is_65_35_winner = winner[0] == 'DC/Rec 65/35'
    marker = "◄ 65/35 WINS" if is_65_35_winner else f"◄ {winner[0]}"

    print(f"  {label:<22}: best = {winner[1]:>10.4f} ({winner[0]:<30}) {marker}")


# ============================================================================
# FOCUSED: Does ANY component help on hit12 specifically?
# ============================================================================

print(f"\n\n{'=' * 110}")
print("FOCUS: hit12 PERFORMANCE (the outcome where 3rd components might hide signal)")
print("=" * 110)

print(f"\n  {'Config':<38} {'r(hit12)':>10} {'AUC hit12':>10} {'Brier h12':>10} {'Top10% h12':>11}")
print("  " + "-" * 82)

# Sort by r_hit12
by_hit12 = sorted(all_results, key=lambda x: x.get('r_hit12', -999), reverse=True)
for r in by_hit12:
    marker = " ◄ 65/35" if r['label'] == 'DC/Rec 65/35' else \
             " ◄ CURRENT" if 'CURRENT' in r['label'] else \
             " ◄ BASE" if r['label'] == 'DC only' else ""
    print(f"  {r['label']:<38} {r['r_hit12']:>+.4f}   {r['auc_hit12']:>.4f}    {r['brier_hit12']:>.4f}   {r['top10_hit12']:>9.1f}%{marker}")


# ============================================================================
# FOCUSED: Does ANY component help on first_3yr_ppg specifically?
# ============================================================================

print(f"\n\n{'=' * 110}")
print("FOCUS: first_3yr_ppg PERFORMANCE (40% of priority weight)")
print("=" * 110)

print(f"\n  {'Config':<38} {'r(3yr_ppg)':>11} {'Top10% PPG':>11}")
print("  " + "-" * 63)

by_ppg = sorted(all_results, key=lambda x: x.get('r_first_3yr_ppg', -999), reverse=True)
for r in by_ppg:
    ppg_s = f"{r['top10_ppg']:.2f}" if not np.isnan(r['top10_ppg']) else "N/A"
    marker = " ◄ 65/35" if r['label'] == 'DC/Rec 65/35' else \
             " ◄ CURRENT" if 'CURRENT' in r['label'] else \
             " ◄ BASE" if r['label'] == 'DC only' else ""
    print(f"  {r['label']:<38} {r['r_first_3yr_ppg']:>+.4f}      {ppg_s:>10}{marker}")


# ============================================================================
# SCORECARD: Count wins across all 12 metrics
# ============================================================================

print(f"\n\n{'=' * 110}")
print("SCORECARD: HOW MANY OF 12 METRICS DOES EACH CONFIG WIN?")
print("(Win = best value among all configs tested)")
print("=" * 110)

win_counts = {r['label']: 0 for r in all_results}
beat_65_counts = {r['label']: 0 for r in all_results}

for key, label, higher_better, fmt in metrics:
    vals = [(r['label'], r.get(key, np.nan)) for r in all_results]
    vals_valid = [(l, v) for l, v in vals if not np.isnan(v)]
    if not vals_valid:
        continue
    if higher_better:
        best_val = max(v for _, v in vals_valid)
    else:
        best_val = min(v for _, v in vals_valid)

    for l, v in vals_valid:
        if v == best_val:
            win_counts[l] += 1
        # Count how many metrics beat 65/35
        base_val = baseline.get(key, np.nan)
        if not np.isnan(base_val) and l != 'DC/Rec 65/35':
            if higher_better and v > base_val + 0.0001:
                beat_65_counts[l] += 1
            elif not higher_better and v < base_val - 0.0001:
                beat_65_counts[l] += 1

print(f"\n  {'Config':<38} {'Wins (of 12)':>14} {'Beats 65/35':>14}")
print("  " + "-" * 68)
for r in all_results:
    marker = " ◄" if r['label'] == 'DC/Rec 65/35' else ""
    print(f"  {r['label']:<38} {win_counts[r['label']]:>14} {beat_65_counts[r['label']]:>14}{marker}")


# ============================================================================
# NET IMPROVEMENT: For each 3rd component, how many metrics improve vs hurt?
# ============================================================================

print(f"\n\n{'=' * 110}")
print("NET IMPACT: For each 3rd component, how many metrics IMPROVE vs HURT vs DC/Rec 65/35?")
print("=" * 110)

# Group configs by component type
component_groups = {
    'Weight (Wt)': [r for r in all_results if '/Wt ' in r['label']],
    'Early Declare (ED)': [r for r in all_results if '/ED ' in r['label']],
    'RAS': [r for r in all_results if '/RAS ' in r['label']],
    'Speed Score (SS)': [r for r in all_results if '/SS ' in r['label']],
    '40 time': [r for r in all_results if '/40 ' in r['label']],
}

for comp_name, comp_results in component_groups.items():
    if not comp_results:
        continue
    print(f"\n  ── {comp_name} ──")
    for cr in comp_results:
        improve = 0
        hurt = 0
        neutral = 0
        for key, label, higher_better, fmt in metrics:
            v = cr.get(key, np.nan)
            base_v = baseline.get(key, np.nan)
            if np.isnan(v) or np.isnan(base_v):
                continue
            delta = v - base_v
            if higher_better:
                if delta > 0.0001:
                    improve += 1
                elif delta < -0.0001:
                    hurt += 1
                else:
                    neutral += 1
            else:
                if delta < -0.0001:
                    improve += 1
                elif delta > 0.0001:
                    hurt += 1
                else:
                    neutral += 1
        net = improve - hurt
        print(f"    {cr['label']:<38} Improve: {improve:>2} | Hurt: {hurt:>2} | Neutral: {neutral:>2} | Net: {net:>+3}")


print(f"\n\n{'=' * 110}")
print("ANALYSIS COMPLETE")
print("=" * 110)
