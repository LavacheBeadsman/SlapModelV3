"""
RB Reoptimization Validation (4 checks)
=========================================
1. DC floor analysis — where does performance drop?
2. Speed Score from combine.parquet — does it add signal?
3. Validation tests — Brier, AUC-ROC, top decile precision
4. Receiving production variants — which is best for first_3yr_ppg?
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

def speed_score_fn(weight, forty):
    """Barnwell Speed Score: (weight * 200) / (40_time)^4"""
    if pd.isna(weight) or pd.isna(forty) or forty <= 0 or weight <= 0:
        return np.nan
    return (weight * 200) / (forty ** 4)

def rb_production_score(row):
    """Age-weighted receiving production, scaled by 1.75"""
    if pd.isna(row['rec_yards']) or pd.isna(row['team_pass_att']) or row['team_pass_att'] == 0:
        return np.nan
    age = row['age'] if pd.notna(row['age']) else 22
    age_weights = {19: 1.15, 20: 1.10, 21: 1.05, 22: 1.00, 23: 0.95}
    age_w = age_weights.get(int(age), 0.90 if age >= 24 else 1.15)
    raw = (row['rec_yards'] / row['team_pass_att']) * age_w * 100
    return min(99.9, raw / 1.75)


# ============================================================================
# DATA LOADING & MASTER TABLE
# ============================================================================

rb = pd.read_csv('data/rb_backtest_with_receiving.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
out_rb = outcomes[outcomes['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']
].copy()
rb = rb.merge(out_rb, on=['player_name', 'draft_year', 'pick'], how='left')
rb['s_dc'] = rb['pick'].apply(dc_score)

# Receiving production (age-weighted, current formula)
rb['s_rec_prod'] = rb.apply(rb_production_score, axis=1)
avg_rec_prod = rb['s_rec_prod'].mean()
rb['s_rec_prod_filled'] = rb['s_rec_prod'].fillna(avg_rec_prod)

# RAS
rb['s_ras'] = rb['RAS'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)
avg_ras = rb['s_ras'].mean()
rb['s_ras_filled'] = rb['s_ras'].fillna(avg_ras)

# ──────────────────────────────────────────────────────────────────────
# CHECK 2: Speed Score from combine.parquet
# ──────────────────────────────────────────────────────────────────────

print("=" * 110)
print("CHECK 2: SPEED SCORE FROM COMBINE.PARQUET")
print("=" * 110)

combine = pd.read_parquet('data/nflverse/combine.parquet')
combine_rb = combine[combine['pos'] == 'RB'].copy()
print(f"\nCombine RBs total: {len(combine_rb)}")
print(f"  With weight: {combine_rb['wt'].notna().sum()}")
print(f"  With 40 time: {combine_rb['forty'].notna().sum()}")
print(f"  With both: {(combine_rb['wt'].notna() & combine_rb['forty'].notna()).sum()}")

# Compute speed score for combine RBs
combine_rb['speed_score'] = combine_rb.apply(
    lambda r: speed_score_fn(r['wt'], r['forty']), axis=1)

# Match to backtest by player_name + draft_year
# Normalize names for matching
def normalize_name(name):
    if pd.isna(name):
        return ''
    return name.strip().lower().replace('.', '').replace("'", '').replace('-', ' ')

combine_rb['name_norm'] = combine_rb['player_name'].apply(normalize_name)
rb['name_norm'] = rb['player_name'].apply(normalize_name)

# Try exact match on name + draft_year
combine_lookup = {}
for _, row in combine_rb.iterrows():
    if pd.notna(row['draft_year']):
        key = (row['name_norm'], int(row['draft_year']))
        if key not in combine_lookup or pd.notna(row['speed_score']):
            combine_lookup[key] = {
                'weight': row['wt'],
                'forty': row['forty'],
                'speed_score': row['speed_score']
            }

# Also try matching on name + season (combine season = draft_year typically)
for _, row in combine_rb.iterrows():
    if pd.notna(row.get('season')):
        key = (row['name_norm'], int(row['season']))
        if key not in combine_lookup or pd.notna(row['speed_score']):
            combine_lookup[key] = {
                'weight': row['wt'],
                'forty': row['forty'],
                'speed_score': row['speed_score']
            }

rb['weight'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb['forty'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb['speed_score_raw'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('speed_score', np.nan), axis=1)

matched = rb['speed_score_raw'].notna().sum()
print(f"\nMatched to backtest: {matched}/{len(rb)} ({matched/len(rb)*100:.1f}%)")

# Show coverage by round
print(f"\n  Coverage by round:")
for rnd in sorted(rb['round'].unique()):
    rnd_rb = rb[rb['round'] == rnd]
    have = rnd_rb['speed_score_raw'].notna().sum()
    print(f"    Round {int(rnd)}: {have}/{len(rnd_rb)} ({have/len(rnd_rb)*100:.1f}%)")

# Show coverage by year
print(f"\n  Coverage by draft year:")
for yr in sorted(rb['draft_year'].unique()):
    yr_rb = rb[rb['draft_year'] == yr]
    have = yr_rb['speed_score_raw'].notna().sum()
    print(f"    {int(yr)}: {have}/{len(yr_rb)} ({have/len(yr_rb)*100:.1f}%)")

# Normalize speed score to 0-100
valid_ss = rb['speed_score_raw'].notna()
if valid_ss.sum() > 10:
    rb.loc[valid_ss, 's_speed'] = normalize_0_100(rb.loc[valid_ss, 'speed_score_raw'])
else:
    rb['s_speed'] = np.nan

avg_speed = rb['s_speed'].mean() if rb['s_speed'].notna().any() else 50
rb['s_speed_filled'] = rb['s_speed'].fillna(avg_speed)

# Show sample speed scores
print(f"\n  Sample Speed Scores (top 15 by raw speed score):")
top_ss = rb[rb['speed_score_raw'].notna()].nlargest(15, 'speed_score_raw')
print(f"  {'Player':<25} {'Pick':>4} {'Weight':>6} {'40':>6} {'Speed Score':>12} {'Normalized':>10}")
print(f"  {'-' * 70}")
for _, r in top_ss.iterrows():
    print(f"  {r['player_name']:<25} {int(r['pick']):>4} {r['weight']:>6.0f} {r['forty']:>6.2f} {r['speed_score_raw']:>12.1f} {r['s_speed']:>10.1f}")

# Missing speed score pattern — are the missing RBs special?
print(f"\n  MNAR CHECK: Are missing speed scores biased?")
have_ss = rb[rb['speed_score_raw'].notna()]
miss_ss = rb[rb['speed_score_raw'].isna()]
print(f"  {'':>30} {'Have SS':>15} {'Missing SS':>15}")
print(f"  {'Avg DC score':<30} {have_ss['s_dc'].mean():>15.1f} {miss_ss['s_dc'].mean():>15.1f}")
print(f"  {'Avg pick':<30} {have_ss['pick'].mean():>15.1f} {miss_ss['pick'].mean():>15.1f}")
print(f"  {'Hit24 rate':<30} {have_ss['hit24'].mean()*100:>14.1f}% {miss_ss['hit24'].mean()*100:>14.1f}%")
r1_have = have_ss[have_ss['round'] == 1]
r1_miss = miss_ss[miss_ss['round'] == 1]
print(f"  {'Round 1 count':<30} {len(r1_have):>15} {len(r1_miss):>15}")
if len(r1_miss) > 0:
    print(f"  {'Round 1 missing names':<30} {list(r1_miss['player_name'].values)}")

# Partial correlation of speed score vs outcomes controlling for DC
rb_eval = rb[rb['hit24'].notna()].copy()

def partial_corr(x, y, z):
    valid = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    if len(valid) < 15:
        return np.nan, np.nan, len(valid)
    slope_xz, inter_xz, _, _, _ = stats.linregress(valid['z'], valid['x'])
    resid_x = valid['x'] - (slope_xz * valid['z'] + inter_xz)
    slope_yz, inter_yz, _, _, _ = stats.linregress(valid['z'], valid['y'])
    resid_y = valid['y'] - (slope_yz * valid['z'] + inter_yz)
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(valid)

print(f"\n  PARTIAL CORRELATIONS: Speed Score vs outcomes (controlling for DC)")
outcome_cols = ['first_3yr_ppg', 'hit24', 'hit12', 'career_ppg']
outcome_weights = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}
pri_sum = 0
pri_total = 0
for out in outcome_cols:
    r, p, n = partial_corr(rb_eval['speed_score_raw'], rb_eval[out], rb_eval['s_dc'])
    sig = '*' if not np.isnan(p) and p < 0.05 else ' '
    print(f"    Speed Score vs {out:<20}: r={r:+.4f}{sig} (p={p:.4f}, N={n})" if not np.isnan(r) else f"    Speed Score vs {out}: N/A")
    if not np.isnan(r):
        pri_sum += outcome_weights[out] * r
        pri_total += outcome_weights[out]
pri_avg_ss = pri_sum / pri_total if pri_total > 0 else np.nan
print(f"    Priority-weighted partial r: {pri_avg_ss:+.4f}")

# Compare to RAS partial correlation
print(f"\n  Comparison: Speed Score vs RAS (partial r after controlling for DC)")
pri_sum_ras = 0
pri_total_ras = 0
for out in outcome_cols:
    r, p, n = partial_corr(rb_eval['RAS'], rb_eval[out], rb_eval['s_dc'])
    if not np.isnan(r):
        pri_sum_ras += outcome_weights[out] * r
        pri_total_ras += outcome_weights[out]
pri_avg_ras = pri_sum_ras / pri_total_ras if pri_total_ras > 0 else np.nan
print(f"    Speed Score partial r: {pri_avg_ss:+.4f} (N={rb_eval['speed_score_raw'].notna().sum()})")
print(f"    RAS partial r:         {pri_avg_ras:+.4f} (N={rb_eval['RAS'].notna().sum()})")
print(f"    Speed Score {'beats' if abs(pri_avg_ss) > abs(pri_avg_ras) else 'loses to'} RAS by {abs(pri_avg_ss) - abs(pri_avg_ras):+.4f}")


# ============================================================================
# CHECK 1: DC FLOOR ANALYSIS
# ============================================================================

print(f"\n\n{'=' * 110}")
print("CHECK 1: DC FLOOR ANALYSIS")
print("Where does performance meaningfully drop?")
print("=" * 110)

def compute_slap(row, weights_dict):
    total = 0
    for col, w in weights_dict.items():
        total += row[col] * w
    return min(100, max(0, total))

def full_eval(df, label, comp_weights):
    """Full evaluation with all metrics."""
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
    hit24_rate = top['hit24'].mean() * 100
    hit12_rate = top['hit12'].mean() * 100
    top_3yr = top[top['first_3yr_ppg'].notna()]
    ppg_top = top_3yr['first_3yr_ppg'].mean() if len(top_3yr) > 0 else np.nan
    top_3yr_n = len(top_3yr)

    # Top 20 hit rate
    top20 = df.nlargest(20, 'slap')
    hit24_top20 = top20['hit24'].mean() * 100

    # Disagreements
    df['dc_rank'] = df['s_dc'].rank(ascending=False, method='min')
    df['slap_rank'] = df['slap'].rank(ascending=False, method='min')
    disagree_10 = int(((df['dc_rank'] - df['slap_rank']).abs() >= 10).sum())

    # Brier score for hit24
    # Normalize SLAP to probability estimate (slap/100)
    valid_brier = df[df['hit24'].notna()].copy()
    prob = valid_brier['slap'] / 100
    brier = ((prob - valid_brier['hit24']) ** 2).mean()

    # AUC-ROC for hit24
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(valid_brier['hit24'], valid_brier['slap'])
    except:
        # Manual AUC calculation
        pos = valid_brier[valid_brier['hit24'] == 1]['slap']
        neg = valid_brier[valid_brier['hit24'] == 0]['slap']
        auc_sum = 0
        for p_val in pos:
            auc_sum += (neg < p_val).sum() + 0.5 * (neg == p_val).sum()
        auc = auc_sum / (len(pos) * len(neg)) if len(pos) * len(neg) > 0 else np.nan

    return {
        'label': label,
        'pri_avg': pri_avg,
        'outcomes': results,
        'hit24_top10': hit24_rate,
        'hit12_top10': hit12_rate,
        'ppg_top10': ppg_top,
        'ppg_top10_n': top_3yr_n,
        'hit24_top20': hit24_top20,
        'disagree_10': disagree_10,
        'n_top': n_top,
        'brier': brier,
        'auc': auc,
    }

# DC floor configs — systematically test DC from 75% down to 45%
dc_floor_configs = [
    ('DC only (100/0)',        {'s_dc': 1.00}),
    ('DC/Rec: 75/25',         {'s_dc': 0.75, 's_rec_prod_filled': 0.25}),
    ('DC/Rec: 70/30',         {'s_dc': 0.70, 's_rec_prod_filled': 0.30}),
    ('DC/Rec: 65/35',         {'s_dc': 0.65, 's_rec_prod_filled': 0.35}),
    ('DC/Rec: 60/40',         {'s_dc': 0.60, 's_rec_prod_filled': 0.40}),
    ('DC/Rec: 55/45',         {'s_dc': 0.55, 's_rec_prod_filled': 0.45}),
    ('DC/Rec: 50/50',         {'s_dc': 0.50, 's_rec_prod_filled': 0.50}),
    ('DC/Rec: 45/55',         {'s_dc': 0.45, 's_rec_prod_filled': 0.55}),
]

dc_results = []
for label, weights in dc_floor_configs:
    r = full_eval(rb_eval, label, weights)
    dc_results.append(r)

print(f"\n  {'Config':<22} {'PRI-AVG r':>10} {'r(3yr)':>8} {'r(h24)':>8} {'Top10%h24':>10} {'Top10%PPG':>10} {'PPG N':>6} {'Dis10+':>7} {'AUC':>7} {'Brier':>7}")
print("  " + "-" * 105)

best_pri = max(r['pri_avg'] for r in dc_results)
for r in dc_results:
    o = r['outcomes']
    r3 = o.get('first_3yr_ppg', {}).get('r', np.nan)
    rh24 = o.get('hit24', {}).get('r', np.nan)
    ppg_s = f"{r['ppg_top10']:.2f}" if not np.isnan(r['ppg_top10']) else "N/A"
    marker = " ◄ BEST" if r['pri_avg'] == best_pri else ""
    delta = r['pri_avg'] - best_pri
    print(f"  {r['label']:<22} {r['pri_avg']:>+.4f}  {r3:>+.4f} {rh24:>+.4f}"
          f"  {r['hit24_top10']:>8.1f}% {ppg_s:>10} {r['ppg_top10_n']:>6}"
          f" {r['disagree_10']:>7} {r['auc']:>.4f} {r['brier']:>.4f}{marker}")

# Identify the floor
print(f"\n  PERFORMANCE DELTAS (vs best = {best_pri:+.4f}):")
for r in dc_results:
    delta = r['pri_avg'] - best_pri
    pct = delta / best_pri * 100
    bar = "█" * max(0, int(50 + pct * 10))
    marker = " ◄ best" if delta == 0 else " ◄ FLOOR" if -0.005 < delta < 0 else ""
    print(f"  {r['label']:<22} {delta:>+.4f} ({pct:>+.1f}%) {bar}{marker}")


# ============================================================================
# CHECK 2 continued: Test Speed Score in weight configs
# ============================================================================

print(f"\n\n{'=' * 110}")
print("CHECK 2 continued: SPEED SCORE IN WEIGHT CONFIGURATIONS")
print(f"Speed Score coverage: {rb_eval['speed_score_raw'].notna().sum()}/{len(rb_eval)} ({rb_eval['speed_score_raw'].notna().sum()/len(rb_eval)*100:.1f}%)")
print("=" * 110)

# Test configs with speed score
ss_configs = [
    ('DC only',                {'s_dc': 1.00}),
    ('DC/Rec: 65/35',         {'s_dc': 0.65, 's_rec_prod_filled': 0.35}),
    ('DC/Rec/SS: 60/30/10',   {'s_dc': 0.60, 's_rec_prod_filled': 0.30, 's_speed_filled': 0.10}),
    ('DC/Rec/SS: 55/35/10',   {'s_dc': 0.55, 's_rec_prod_filled': 0.35, 's_speed_filled': 0.10}),
    ('DC/Rec/SS: 65/25/10',   {'s_dc': 0.65, 's_rec_prod_filled': 0.25, 's_speed_filled': 0.10}),
    ('DC/Rec/SS: 55/30/15',   {'s_dc': 0.55, 's_rec_prod_filled': 0.30, 's_speed_filled': 0.15}),
    ('DC/Rec/SS: 50/35/15',   {'s_dc': 0.50, 's_rec_prod_filled': 0.35, 's_speed_filled': 0.15}),
    ('CURRENT: DC/Rec/RAS 50/35/15', {'s_dc': 0.50, 's_rec_prod_filled': 0.35, 's_ras_filled': 0.15}),
]

ss_results = []
for label, weights in ss_configs:
    r = full_eval(rb_eval, label, weights)
    ss_results.append(r)

print(f"\n  {'Config':<35} {'PRI-AVG r':>10} {'r(3yr)':>8} {'r(h24)':>8} {'Top10%h24':>10} {'Top10%PPG':>10} {'AUC':>7} {'Brier':>7}")
print("  " + "-" * 100)
for r in ss_results:
    o = r['outcomes']
    r3 = o.get('first_3yr_ppg', {}).get('r', np.nan)
    rh24 = o.get('hit24', {}).get('r', np.nan)
    ppg_s = f"{r['ppg_top10']:.2f}" if not np.isnan(r['ppg_top10']) else "N/A"
    print(f"  {r['label']:<35} {r['pri_avg']:>+.4f}  {r3:>+.4f} {rh24:>+.4f}"
          f"  {r['hit24_top10']:>8.1f}% {ppg_s:>10} {r['auc']:>.4f} {r['brier']:>.4f}")

# Direct comparison: RAS vs Speed Score at same weight
print(f"\n  HEAD-TO-HEAD: RAS vs Speed Score (at 15% weight)")
ras_r = [r for r in ss_results if 'RAS' in r['label']][0]
ss_r = [r for r in ss_results if 'DC/Rec/SS: 50/35/15' in r['label']][0]
print(f"    DC/Rec/RAS 50/35/15:  PRI-AVG={ras_r['pri_avg']:+.4f}, AUC={ras_r['auc']:.4f}, Top10% hit24={ras_r['hit24_top10']:.1f}%")
print(f"    DC/Rec/SS  50/35/15:  PRI-AVG={ss_r['pri_avg']:+.4f}, AUC={ss_r['auc']:.4f}, Top10% hit24={ss_r['hit24_top10']:.1f}%")
delta = ss_r['pri_avg'] - ras_r['pri_avg']
print(f"    Speed Score {'beats' if delta > 0 else 'loses to'} RAS by {delta:+.4f}")

# Compare speed score configs to 65/35 baseline
baseline_65 = [r for r in ss_results if r['label'] == 'DC/Rec: 65/35'][0]
print(f"\n  VS DC/Rec 65/35 BASELINE:")
for r in ss_results:
    if 'SS' in r['label']:
        delta = r['pri_avg'] - baseline_65['pri_avg']
        print(f"    {r['label']:<35} delta={delta:>+.4f} ({'helps' if delta > 0 else 'hurts'})")


# ============================================================================
# CHECK 3: VALIDATION — Brier, AUC, Top Decile for key configs
# ============================================================================

print(f"\n\n{'=' * 110}")
print("CHECK 3: MULTI-METHOD VALIDATION")
print("DC+Rec 65/35 vs Current 50/35/15 vs DC only")
print("=" * 110)

val_configs = [
    ('DC only',                          {'s_dc': 1.00}),
    ('CURRENT: DC/Rec/RAS 50/35/15',    {'s_dc': 0.50, 's_rec_prod_filled': 0.35, 's_ras_filled': 0.15}),
    ('PROPOSED: DC/Rec 65/35',           {'s_dc': 0.65, 's_rec_prod_filled': 0.35}),
]

print(f"\n  {'Metric':<45} {'DC only':>15} {'Current':>15} {'Proposed':>15}")
print("  " + "-" * 92)

val_results = {}
for label, weights in val_configs:
    val_results[label] = full_eval(rb_eval, label, weights)

# Priority-weighted r
print(f"  {'Priority-weighted r':<45}", end="")
for label, _ in val_configs:
    print(f" {val_results[label]['pri_avg']:>+14.4f}", end="")
print()

# Individual correlations
for out in outcome_cols:
    out_label = f"  r({out})"
    print(f"  {out_label:<45}", end="")
    for label, _ in val_configs:
        r = val_results[label]['outcomes'].get(out, {}).get('r', np.nan)
        print(f" {r:>+14.4f}", end="")
    print()

print(f"  {'':<45}{'-'*15}{'-'*15}{'-'*15}")

# AUC-ROC
print(f"  {'AUC-ROC (hit24)':<45}", end="")
for label, _ in val_configs:
    print(f" {val_results[label]['auc']:>14.4f}", end="")
print()

# Brier Score (lower is better)
print(f"  {'Brier Score (hit24, lower=better)':<45}", end="")
for label, _ in val_configs:
    print(f" {val_results[label]['brier']:>14.4f}", end="")
print()

# Top decile hit24
print(f"  {'Top 10% hit24 rate':<45}", end="")
for label, _ in val_configs:
    print(f" {val_results[label]['hit24_top10']:>13.1f}%", end="")
print()

# Top decile hit12
print(f"  {'Top 10% hit12 rate':<45}", end="")
for label, _ in val_configs:
    print(f" {val_results[label]['hit12_top10']:>13.1f}%", end="")
print()

# Top decile PPG
print(f"  {'Top 10% first_3yr_ppg':<45}", end="")
for label, _ in val_configs:
    ppg = val_results[label]['ppg_top10']
    n = val_results[label]['ppg_top10_n']
    print(f" {ppg:>10.2f}(N={n:>2})" if not np.isnan(ppg) else f" {'N/A':>14}", end="")
print()

# Top 20 hit24
print(f"  {'Top 20 hit24 rate':<45}", end="")
for label, _ in val_configs:
    print(f" {val_results[label]['hit24_top20']:>13.1f}%", end="")
print()

# Disagreements
print(f"  {'Ranking disagreements (10+ spots)':<45}", end="")
for label, _ in val_configs:
    print(f" {val_results[label]['disagree_10']:>14}", end="")
print()

# Additional: AUC for hit12
print(f"\n  AUC-ROC for hit12:")
for label, weights in val_configs:
    df_tmp = rb_eval.copy()
    df_tmp['slap'] = df_tmp.apply(lambda r: compute_slap(r, weights), axis=1)
    valid = df_tmp[df_tmp['hit12'].notna()]
    pos = valid[valid['hit12'] == 1]['slap']
    neg = valid[valid['hit12'] == 0]['slap']
    auc_sum = 0
    for p_val in pos:
        auc_sum += (neg < p_val).sum() + 0.5 * (neg == p_val).sum()
    auc12 = auc_sum / (len(pos) * len(neg)) if len(pos) * len(neg) > 0 else np.nan
    print(f"    {label:<40}: {auc12:.4f}")

# Brier score for hit12
print(f"\n  Brier Score for hit12:")
for label, weights in val_configs:
    df_tmp = rb_eval.copy()
    df_tmp['slap'] = df_tmp.apply(lambda r: compute_slap(r, weights), axis=1)
    valid = df_tmp[df_tmp['hit12'].notna()]
    prob = valid['slap'] / 100
    brier12 = ((prob - valid['hit12']) ** 2).mean()
    print(f"    {label:<40}: {brier12:.4f}")

# Tier breakdown for all 3
print(f"\n  TIER BREAKDOWN:")
for label, weights in val_configs:
    df_tmp = rb_eval.copy()
    df_tmp['slap'] = df_tmp.apply(lambda r: compute_slap(r, weights), axis=1)

    print(f"\n  ── {label} ──")
    bins = [(80, 100, 'Elite (80-100)'), (60, 80, 'Good (60-80)'),
            (40, 60, 'Average (40-60)'), (0, 40, 'Below Avg (0-40)')]
    print(f"  {'Tier':<20} {'N':>5} {'H24':>4} {'Rate':>7} {'H12':>4} {'Rate':>7} {'3yr PPG':>8} {'Car PPG':>8}")
    print(f"  {'-' * 65}")
    for lo, hi, tier_name in bins:
        tier = df_tmp[(df_tmp['slap'] >= lo) & (df_tmp['slap'] < hi)]
        if len(tier) == 0:
            continue
        h24 = int(tier['hit24'].sum())
        h12 = int(tier['hit12'].sum())
        r24 = h24 / len(tier) * 100
        r12 = h12 / len(tier) * 100
        t3 = tier[tier['first_3yr_ppg'].notna()]
        ppg3 = t3['first_3yr_ppg'].mean() if len(t3) > 0 else np.nan
        tc = tier[tier['career_ppg'].notna()]
        ppgc = tc['career_ppg'].mean() if len(tc) > 0 else np.nan
        p3s = f"{ppg3:.2f}" if not np.isnan(ppg3) else "N/A"
        pcs = f"{ppgc:.2f}" if not np.isnan(ppgc) else "N/A"
        print(f"  {tier_name:<20} {len(tier):>5} {h24:>4} {r24:>6.1f}% {h12:>4} {r12:>6.1f}% {p3s:>8} {pcs:>8}")


# ============================================================================
# CHECK 4: RECEIVING PRODUCTION VARIANTS
# ============================================================================

print(f"\n\n{'=' * 110}")
print("CHECK 4: RECEIVING PRODUCTION METRIC VARIANTS")
print("Which variant is best for first_3yr_ppg?")
print("=" * 110)

# Compute all variants
rb_eval['prod_age_weighted'] = rb_eval.apply(rb_production_score, axis=1)  # Current: age-weighted, /1.75

# Raw rec_yards / team_pass_att (no age weight, no scaling)
rb_eval['prod_raw_ratio'] = rb_eval.apply(
    lambda r: r['rec_yards'] / r['team_pass_att']
    if pd.notna(r['rec_yards']) and pd.notna(r['team_pass_att']) and r['team_pass_att'] > 0
    else np.nan, axis=1)

# Receptions / team_pass_att
rb_eval['prod_rec_share'] = rb_eval.apply(
    lambda r: r['receptions'] / r['team_pass_att']
    if pd.notna(r['receptions']) and pd.notna(r['team_pass_att']) and r['team_pass_att'] > 0
    else np.nan, axis=1)

# Raw rec_yards (no denominator)
rb_eval['prod_rec_yards'] = rb_eval['rec_yards']

# Age-weighted but NOT scaled by 1.75 (raw age-weighted)
def prod_age_raw(row):
    if pd.isna(row['rec_yards']) or pd.isna(row['team_pass_att']) or row['team_pass_att'] == 0:
        return np.nan
    age = row['age'] if pd.notna(row['age']) else 22
    age_weights = {19: 1.15, 20: 1.10, 21: 1.05, 22: 1.00, 23: 0.95}
    age_w = age_weights.get(int(age), 0.90 if age >= 24 else 1.15)
    return (row['rec_yards'] / row['team_pass_att']) * age_w * 100

rb_eval['prod_age_raw'] = rb_eval.apply(prod_age_raw, axis=1)

# Rec yards / team pass att × 100 (same as age-weighted but age=1.0 for all)
rb_eval['prod_simple_ratio'] = rb_eval.apply(
    lambda r: (r['rec_yards'] / r['team_pass_att']) * 100
    if pd.notna(r['rec_yards']) and pd.notna(r['team_pass_att']) and r['team_pass_att'] > 0
    else np.nan, axis=1)

variants = {
    'Age-weighted, /1.75 (CURRENT)': 'prod_age_weighted',
    'Raw ratio (rec_yds / pass_att)': 'prod_raw_ratio',
    'Receptions / pass_att': 'prod_rec_share',
    'Rec yards (raw total)': 'prod_rec_yards',
    'Age-weighted raw (not /1.75)': 'prod_age_raw',
    'Simple ratio ×100': 'prod_simple_ratio',
}

# Show raw and partial correlations for each variant
print(f"\n  PARTIAL CORRELATIONS (each variant vs each outcome, controlling for DC):")
print(f"\n  {'Variant':<35}", end="")
for out in outcome_cols:
    print(f" {'r('+out+')':>16}", end="")
print(f" {'PRI-AVG':>10}")
print("  " + "-" * 100)

variant_details = {}
for name, col in variants.items():
    row_str = f"  {name:<35}"
    pri_sum = 0
    pri_total = 0
    details = {}
    for out in outcome_cols:
        r, p, n = partial_corr(rb_eval[col], rb_eval[out], rb_eval['s_dc'])
        details[out] = {'r': r, 'p': p, 'n': n}
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
    variant_details[name] = {'pri_avg': pri_avg, 'col': col, 'details': details}

# Focus on first_3yr_ppg (the top-weighted outcome at 40%)
print(f"\n  FOCUS: first_3yr_ppg partial r (the 40%-weighted outcome):")
print(f"  {'Variant':<35} {'Partial r':>12} {'p-value':>12} {'N':>6}")
print("  " + "-" * 70)
for name, col in variants.items():
    d = variant_details[name]['details']['first_3yr_ppg']
    sig = '***' if d['p'] < 0.001 else '**' if d['p'] < 0.01 else '*' if d['p'] < 0.05 else ''
    print(f"  {name:<35} {d['r']:>+.4f}{sig:>4} {d['p']:>12.6f} {d['n']:>6}")

# Now test FULL MODEL with each variant at 65/35
print(f"\n\n  FULL MODEL TEST: DC 65% / Variant 35% — which variant produces best model?")

for name, col in variants.items():
    # Normalize to 0-100
    valid = rb_eval[col].notna()
    if valid.sum() < 10:
        continue
    normalized = pd.Series(np.nan, index=rb_eval.index)
    normalized[valid] = normalize_0_100(rb_eval.loc[valid, col])
    rb_eval[f'{col}_norm'] = normalized.fillna(normalized.mean())

# Test each variant at 65/35
print(f"\n  {'Variant':<35} {'PRI-AVG':>10} {'r(3yr)':>8} {'r(h24)':>8} {'Top10%h24':>10} {'Top10%PPG':>10} {'AUC':>7}")
print("  " + "-" * 95)

for name, col in variants.items():
    norm_col = f'{col}_norm'
    if norm_col not in rb_eval.columns:
        continue
    weights = {'s_dc': 0.65, norm_col: 0.35}
    r = full_eval(rb_eval, name, weights)
    o = r['outcomes']
    r3 = o.get('first_3yr_ppg', {}).get('r', np.nan)
    rh24 = o.get('hit24', {}).get('r', np.nan)
    ppg_s = f"{r['ppg_top10']:.2f}" if not np.isnan(r['ppg_top10']) else "N/A"
    marker = " ◄ CURRENT" if 'CURRENT' in name else ""
    print(f"  {name:<35} {r['pri_avg']:>+.4f}  {r3:>+.4f} {rh24:>+.4f}"
          f"  {r['hit24_top10']:>8.1f}% {ppg_s:>10} {r['auc']:>.4f}{marker}")


# Check if variants differ on top-decile composition
print(f"\n\n  TOP DECILE COMPOSITION: Do different variants select different players?")
# Current age-weighted
w_current = {'s_dc': 0.65, 'prod_age_weighted_norm': 0.35}
w_raw = {'s_dc': 0.65, 'prod_raw_ratio_norm': 0.35}
w_recshr = {'s_dc': 0.65, 'prod_rec_share_norm': 0.35}

df_c = rb_eval.copy()
df_c['slap_current'] = df_c.apply(lambda r: compute_slap(r, w_current), axis=1)
df_c['slap_raw'] = df_c.apply(lambda r: compute_slap(r, w_raw), axis=1)
df_c['slap_recshr'] = df_c.apply(lambda r: compute_slap(r, w_recshr), axis=1)

n_top = max(1, len(df_c) // 10)
top_current = set(df_c.nlargest(n_top, 'slap_current')['player_name'])
top_raw = set(df_c.nlargest(n_top, 'slap_raw')['player_name'])
top_recshr = set(df_c.nlargest(n_top, 'slap_recshr')['player_name'])

print(f"  Top {n_top} overlap:")
print(f"    Age-weighted vs Raw ratio: {len(top_current & top_raw)}/{n_top} shared")
print(f"    Age-weighted vs Rec share: {len(top_current & top_recshr)}/{n_top} shared")
print(f"    Raw ratio vs Rec share:    {len(top_raw & top_recshr)}/{n_top} shared")

# Show where they differ
only_current = top_current - top_raw
only_raw = top_raw - top_current
if only_current or only_raw:
    print(f"\n    In age-weighted top but NOT raw: {only_current}")
    print(f"    In raw top but NOT age-weighted: {only_raw}")


print(f"\n\n{'=' * 110}")
print("ANALYSIS COMPLETE")
print("=" * 110)
