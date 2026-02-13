"""
RB Reoptimization — Athletic component final check
====================================================
1. Rank all athletic metrics at exactly 5% weight
2. Player-level ranking changes for least-costly metric
3. Weight × Receiving interaction term analysis
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
    valid = pd.DataFrame({'y': labels, 's': scores_0_100}).dropna()
    if len(valid) < 10:
        return np.nan
    prob = valid['s'] / 100
    return ((prob - valid['y']) ** 2).mean()

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

# Combine data for athletic metrics
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
    combine_lookup[key] = {
        'weight': row['wt'], 'forty': row['forty'],
        'ht': row['ht'],
    }

rb['weight'] = rb.apply(lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb['forty'] = rb.apply(lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb['height'] = rb.apply(lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('ht', np.nan), axis=1)

# Force numeric types
for col in ['weight', 'forty', 'height']:
    rb[col] = pd.to_numeric(rb[col], errors='coerce')

# Derived metrics
rb['speed_score_raw'] = rb.apply(lambda r: speed_score_fn(r['weight'], r['forty']), axis=1)
def calc_bmi(r):
    try:
        w = float(r['weight'])
        h = float(r['height'])
        if pd.notna(w) and pd.notna(h) and h > 0:
            return (w * 703) / (h ** 2)
    except (ValueError, TypeError):
        pass
    return np.nan
rb['bmi'] = rb.apply(calc_bmi, axis=1)

# RAS
rb['ras_raw'] = rb['RAS'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)

# Normalize each to 0-100 (higher = "more athletic")
athletic_metrics = {
    'weight':      ('weight',          False),  # higher weight, higher score
    'forty_inv':   ('forty',           True),   # lower forty, higher score (invert)
    'speed_score': ('speed_score_raw', False),
    'ras':         ('ras_raw',         False),
    'bmi':         ('bmi',             False),
    'height':      ('height',          False),
}

for name, (raw_col, invert) in athletic_metrics.items():
    score_col = f's_{name}'
    rb[score_col] = np.nan
    valid = rb[raw_col].notna()
    if valid.sum() > 10:
        if invert:
            rb.loc[valid, score_col] = normalize_0_100(-rb.loc[valid, raw_col])
        else:
            rb.loc[valid, score_col] = normalize_0_100(rb.loc[valid, raw_col])
    avg_val = rb[score_col].mean() if rb[score_col].notna().any() else 50
    rb[f'{score_col}_filled'] = rb[score_col].fillna(avg_val)

rb_eval = rb[rb['hit24'].notna()].copy()

# ============================================================================
# PART 1: Rank all athletic metrics at exactly 5% weight
# ============================================================================

print("=" * 120)
print("PART 1: ALL ATHLETIC METRICS AT EXACTLY 5% WEIGHT (DC 60 / Rec 35 / Athletic 5)")
print(f"Eval sample: {len(rb_eval)} RBs")
print("=" * 120)

# Baseline: 65/35
base_slap = rb_eval['s_dc'] * 0.65 + rb_eval['s_rec_prod_filled'] * 0.35

outcomes_list = ['first_3yr_ppg', 'hit24', 'hit12', 'career_ppg']

def full_metrics(slap, df):
    res = {}
    for out in outcomes_list:
        valid = pd.DataFrame({'s': slap, 'o': df[out]}).dropna()
        if len(valid) >= 10:
            r, _ = stats.pearsonr(valid['s'], valid['o'])
            res[f'r_{out}'] = r
        else:
            res[f'r_{out}'] = np.nan
    # PRI-AVG
    pri = sum(outcome_weights[o] * res.get(f'r_{o}', 0) for o in outcomes_list
              if not np.isnan(res.get(f'r_{o}', np.nan)))
    pri_w = sum(outcome_weights[o] for o in outcomes_list
                if not np.isnan(res.get(f'r_{o}', np.nan)))
    res['pri_avg'] = pri / pri_w if pri_w > 0 else np.nan
    # Top decile
    n_top = max(1, len(df) // 10)
    top_idx = slap.nlargest(n_top).index
    top = df.loc[top_idx]
    res['top10_hit24'] = top['hit24'].mean() * 100
    res['top10_hit12'] = top['hit12'].mean() * 100
    ppg_top = top[top['first_3yr_ppg'].notna()]
    res['top10_ppg'] = ppg_top['first_3yr_ppg'].mean() if len(ppg_top) > 0 else np.nan
    # AUC
    res['auc_hit24'] = auc_roc(df['hit24'], slap)
    res['auc_hit12'] = auc_roc(df['hit12'], slap)
    # Brier
    res['brier_hit24'] = brier_score(df['hit24'], slap)
    res['brier_hit12'] = brier_score(df['hit12'], slap)
    return res

base_m = full_metrics(base_slap, rb_eval)

metric_keys = [
    ('r_first_3yr_ppg', True), ('r_hit24', True), ('r_hit12', True), ('r_career_ppg', True),
    ('pri_avg', True), ('top10_hit24', True), ('top10_hit12', True), ('top10_ppg', True),
    ('auc_hit24', True), ('auc_hit12', True), ('brier_hit24', False), ('brier_hit12', False),
]

results_5pct = {}
for name in athletic_metrics:
    score_col = f's_{name}_filled'
    slap = rb_eval['s_dc'] * 0.60 + rb_eval['s_rec_prod_filled'] * 0.35 + rb_eval[score_col] * 0.05
    m = full_metrics(slap, rb_eval)

    # Count metrics improved vs hurt
    improved = 0
    hurt = 0
    total_delta = 0
    for key, higher_better in metric_keys:
        v = m.get(key, np.nan)
        b = base_m.get(key, np.nan)
        if np.isnan(v) or np.isnan(b):
            continue
        delta = v - b
        if higher_better:
            if delta > 0.0001: improved += 1
            elif delta < -0.0001: hurt += 1
            total_delta += delta  # positive = good
        else:
            if delta < -0.0001: improved += 1
            elif delta > 0.0001: hurt += 1
            total_delta -= delta  # flip sign so positive = good

    results_5pct[name] = {
        'metrics': m,
        'improved': improved,
        'hurt': hurt,
        'net': improved - hurt,
        'total_delta': total_delta,  # sum of signed improvements
    }
    # Also compute data coverage
    obs = rb_eval[f's_{name}'].notna().sum()
    results_5pct[name]['coverage'] = obs
    results_5pct[name]['coverage_pct'] = obs / len(rb_eval) * 100

# Sort by net impact (least costly first)
sorted_metrics = sorted(results_5pct.items(), key=lambda x: (-x[1]['net'], -x[1]['total_delta']))

print(f"\n  {'Athletic Metric':<18} {'Coverage':>10} {'Improved':>10} {'Hurt':>8} {'Net':>6} {'PRI-AVG':>10} {'Δ PRI-AVG':>11}")
print("  " + "-" * 78)

for name, info in sorted_metrics:
    m = info['metrics']
    d_pri = m['pri_avg'] - base_m['pri_avg']
    print(f"  {name:<18} {info['coverage']:>6}/{len(rb_eval)} ({info['coverage_pct']:>4.0f}%) {info['improved']:>7} {info['hurt']:>8} {info['net']:>+5} {m['pri_avg']:>+.4f}   {d_pri:>+.4f}")

# Full delta table for all metrics
print(f"\n\n  {'Deltas vs 65/35':<18}", end="")
for name, info in sorted_metrics:
    print(f" {name:>14}", end="")
print(f" {'65/35 base':>14}")
print("  " + "-" * (18 + 15 * (len(sorted_metrics) + 1)))

metric_labels = [
    ('r_first_3yr_ppg', 'r(3yr_ppg)', True),
    ('r_hit24', 'r(hit24)', True),
    ('r_hit12', 'r(hit12)', True),
    ('r_career_ppg', 'r(career_ppg)', True),
    ('pri_avg', 'PRI-AVG', True),
    ('top10_hit24', 'T10% hit24', True),
    ('top10_hit12', 'T10% hit12', True),
    ('top10_ppg', 'T10% PPG', True),
    ('auc_hit24', 'AUC hit24', True),
    ('auc_hit12', 'AUC hit12', True),
    ('brier_hit24', 'Brier h24', False),
    ('brier_hit12', 'Brier h12', False),
]

for key, label, higher_better in metric_labels:
    row = f"  {label:<18}"
    base_v = base_m[key]
    for name, info in sorted_metrics:
        v = info['metrics'][key]
        delta = v - base_v
        if higher_better:
            marker = "+" if delta > 0.001 else "-" if delta < -0.001 else "~"
        else:
            marker = "+" if delta < -0.001 else "-" if delta > 0.001 else "~"
        if 'T10%' in label and 'PPG' not in label:
            row += f" {delta:>+10.1f}% {marker}"
        else:
            row += f" {delta:>+11.4f} {marker}"
    # base value
    if 'T10%' in label and 'PPG' not in label:
        row += f" {base_v:>12.1f}%"
    else:
        row += f" {base_v:>13.4f}"
    print(row)


# ============================================================================
# PART 2: Player-level ranking changes for least-costly metric
# ============================================================================

best_name = sorted_metrics[0][0]
best_info = sorted_metrics[0][1]

print(f"\n\n{'=' * 120}")
print(f"PART 2: PLAYER-LEVEL RANKING CHANGES — {best_name.upper()} AT 5%")
print(f"Comparing DC/Rec 65/35 vs DC/Rec/{best_name} 60/35/5")
print("=" * 120)

rb_eval = rb_eval.copy()
rb_eval['slap_base'] = rb_eval['s_dc'] * 0.65 + rb_eval['s_rec_prod_filled'] * 0.35
rb_eval['rank_base'] = rb_eval['slap_base'].rank(ascending=False, method='min').astype(int)

score_col = f's_{best_name}_filled'
rb_eval['slap_new'] = rb_eval['s_dc'] * 0.60 + rb_eval['s_rec_prod_filled'] * 0.35 + rb_eval[score_col] * 0.05
rb_eval['rank_new'] = rb_eval['slap_new'].rank(ascending=False, method='min').astype(int)
rb_eval['rank_change'] = rb_eval['rank_base'] - rb_eval['rank_new']  # positive = moved up

# Raw athletic score (unimputed)
raw_col = f's_{best_name}'

# Show biggest movers
movers = rb_eval[rb_eval['rank_change'].abs() >= 3].sort_values('rank_change', ascending=False)

print(f"\n  Players who moved 3+ spots:")
print(f"  {'Player':<25} {'Year':>4} {'Pick':>4} {'Wt':>6} {best_name:>8} {'Base SLAP':>10} {'New SLAP':>9} {'Base Rk':>8} {'New Rk':>7} {'Δ Rk':>5} {'hit24':>5} {'hit12':>5} {'3yr PPG':>8}")
print("  " + "-" * 120)

for _, row in movers.iterrows():
    raw_ath = row[raw_col]
    raw_ath_s = f"{raw_ath:.1f}" if pd.notna(raw_ath) else "IMP"
    wt_s = f"{row['weight']:.0f}" if pd.notna(row['weight']) else "?"
    h24 = f"{int(row['hit24'])}" if pd.notna(row['hit24']) else "?"
    h12 = f"{int(row['hit12'])}" if pd.notna(row['hit12']) else "?"
    ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "?"
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>4} {int(row['pick']):>4} {wt_s:>6} {raw_ath_s:>8} {row['slap_base']:>10.1f} {row['slap_new']:>9.1f} {row['rank_base']:>8} {row['rank_new']:>7} {row['rank_change']:>+5} {h24:>5} {h12:>5} {ppg:>8}")

# Summarize: Did promoted players do better?
promoted = rb_eval[rb_eval['rank_change'] >= 3]
demoted = rb_eval[rb_eval['rank_change'] <= -3]

print(f"\n  PROMOTED players (moved up 3+): n={len(promoted)}")
if len(promoted) > 0:
    for out in ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']:
        valid = promoted[out].dropna()
        if len(valid) > 0:
            if out in ['hit24', 'hit12']:
                print(f"    {out}: {valid.mean()*100:.1f}% hit rate")
            else:
                print(f"    {out}: {valid.mean():.2f} avg")

print(f"\n  DEMOTED players (moved down 3+): n={len(demoted)}")
if len(demoted) > 0:
    for out in ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']:
        valid = demoted[out].dropna()
        if len(valid) > 0:
            if out in ['hit24', 'hit12']:
                print(f"    {out}: {valid.mean()*100:.1f}% hit rate")
            else:
                print(f"    {out}: {valid.mean():.2f} avg")

# Sample average for reference
print(f"\n  FULL SAMPLE averages (n={len(rb_eval)}):")
for out in ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']:
    valid = rb_eval[out].dropna()
    if len(valid) > 0:
        if out in ['hit24', 'hit12']:
            print(f"    {out}: {valid.mean()*100:.1f}% hit rate")
        else:
            print(f"    {out}: {valid.mean():.2f} avg")

# Top-20 comparison
print(f"\n\n  TOP 20 COMPARISON: 65/35 vs 60/35/5-{best_name}")
print(f"  {'#':>3} {'65/35 Player':<28} {'SLAP':>5} {'hit24':>5} │ {'#':>3} {'60/35/5 Player':<28} {'SLAP':>5} {'hit24':>5} {'Δ':>4}")
print("  " + "-" * 100)

top20_base = rb_eval.nsmallest(20, 'rank_base').sort_values('rank_base')
top20_new = rb_eval.nsmallest(20, 'rank_new').sort_values('rank_new')

for i in range(20):
    rb = top20_base.iloc[i]
    rn = top20_new.iloc[i]
    h24_b = f"{int(rb['hit24'])}" if pd.notna(rb['hit24']) else "?"
    h24_n = f"{int(rn['hit24'])}" if pd.notna(rn['hit24']) else "?"
    chg = ""
    if rn['player_name'] not in top20_base['player_name'].values:
        chg = " NEW"
    elif rb['player_name'] not in top20_new['player_name'].values:
        pass
    print(f"  {i+1:>3} {rb['player_name']:<28} {rb['slap_base']:>5.1f} {h24_b:>5} │ {i+1:>3} {rn['player_name']:<28} {rn['slap_new']:>5.1f} {h24_n:>5}{chg}")

# Hit rate in top 20
t20b_h24 = top20_base['hit24'].mean() * 100
t20n_h24 = top20_new['hit24'].mean() * 100
t20b_h12 = top20_base['hit12'].mean() * 100
t20n_h12 = top20_new['hit12'].mean() * 100
print(f"\n  Top-20 hit24: 65/35 = {t20b_h24:.1f}%  |  60/35/5-{best_name} = {t20n_h24:.1f}%  |  Δ = {t20n_h24-t20b_h24:+.1f}%")
print(f"  Top-20 hit12: 65/35 = {t20b_h12:.1f}%  |  60/35/5-{best_name} = {t20n_h12:.1f}%  |  Δ = {t20n_h12-t20b_h12:+.1f}%")


# ============================================================================
# PART 2b: Also show weight at 5% specifically (user asked about it)
# ============================================================================

print(f"\n\n{'=' * 120}")
print(f"PART 2b: PLAYER-LEVEL RANKING CHANGES — WEIGHT AT 5%")
print(f"Comparing DC/Rec 65/35 vs DC/Rec/Wt 60/35/5")
print("=" * 120)

rb_eval['slap_wt'] = rb_eval['s_dc'] * 0.60 + rb_eval['s_rec_prod_filled'] * 0.35 + rb_eval['s_weight_filled'] * 0.05
rb_eval['rank_wt'] = rb_eval['slap_wt'].rank(ascending=False, method='min').astype(int)
rb_eval['rank_change_wt'] = rb_eval['rank_base'] - rb_eval['rank_wt']

movers_wt = rb_eval[rb_eval['rank_change_wt'].abs() >= 3].sort_values('rank_change_wt', ascending=False)

print(f"\n  Players who moved 3+ spots with WEIGHT at 5%:")
print(f"  {'Player':<25} {'Year':>4} {'Pick':>4} {'Wt(lb)':>7} {'Wt Score':>9} {'Base SLAP':>10} {'Wt SLAP':>8} {'Δ Rk':>5} {'hit24':>5} {'hit12':>5} {'3yr PPG':>8}")
print("  " + "-" * 100)

for _, row in movers_wt.iterrows():
    wt_s = f"{row['weight']:.0f}" if pd.notna(row['weight']) else "?"
    wt_sc = f"{row['s_weight']:.1f}" if pd.notna(row['s_weight']) else "IMP"
    h24 = f"{int(row['hit24'])}" if pd.notna(row['hit24']) else "?"
    h12 = f"{int(row['hit12'])}" if pd.notna(row['hit12']) else "?"
    ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "?"
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>4} {int(row['pick']):>4} {wt_s:>7} {wt_sc:>9} {row['slap_base']:>10.1f} {row['slap_wt']:>8.1f} {row['rank_change_wt']:>+5} {h24:>5} {h12:>5} {ppg:>8}")

promoted_wt = rb_eval[rb_eval['rank_change_wt'] >= 3]
demoted_wt = rb_eval[rb_eval['rank_change_wt'] <= -3]

print(f"\n  PROMOTED by weight (up 3+): n={len(promoted_wt)}")
if len(promoted_wt) > 0:
    for out in ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']:
        valid = promoted_wt[out].dropna()
        if len(valid) > 0:
            if out in ['hit24', 'hit12']:
                print(f"    {out}: {valid.mean()*100:.1f}% hit rate")
            else:
                print(f"    {out}: {valid.mean():.2f} avg")

print(f"\n  DEMOTED by weight (down 3+): n={len(demoted_wt)}")
if len(demoted_wt) > 0:
    for out in ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']:
        valid = demoted_wt[out].dropna()
        if len(valid) > 0:
            if out in ['hit24', 'hit12']:
                print(f"    {out}: {valid.mean()*100:.1f}% hit rate")
            else:
                print(f"    {out}: {valid.mean():.2f} avg")


# ============================================================================
# PART 3: Weight × Receiving interaction term
# ============================================================================

print(f"\n\n{'=' * 120}")
print("PART 3: INTERACTION TERM — WEIGHT × RECEIVING PRODUCTION")
print("Does being big AND catching passes add signal beyond each alone?")
print("=" * 120)

# Create the interaction term
rb_eval = rb_eval.copy()
# Use actual weight (not normalized) × rec production score
# Normalize the interaction to 0-100
rb_eval['wt_x_rec_raw'] = rb_eval['weight'] * rb_eval['s_rec_prod']
valid_inter = rb_eval['wt_x_rec_raw'].notna()
if valid_inter.sum() > 10:
    rb_eval.loc[valid_inter, 's_wt_x_rec'] = normalize_0_100(rb_eval.loc[valid_inter, 'wt_x_rec_raw'])
avg_inter = rb_eval['s_wt_x_rec'].mean() if rb_eval['s_wt_x_rec'].notna().any() else 50
rb_eval['s_wt_x_rec_filled'] = rb_eval['s_wt_x_rec'].fillna(avg_inter)

print(f"\n  Interaction coverage: {valid_inter.sum()}/{len(rb_eval)} ({valid_inter.sum()/len(rb_eval)*100:.0f}%)")

# 3a: Raw correlation of interaction term with each outcome
print(f"\n  ── Raw correlations (interaction term alone) ──")
for out in outcomes_list:
    valid = rb_eval[['s_wt_x_rec', out]].dropna()
    if len(valid) >= 10:
        r, p = stats.pearsonr(valid['s_wt_x_rec'], valid[out])
        print(f"    r(wt×rec, {out}) = {r:+.4f}  (p={p:.4f})")

# Compare with weight alone and rec alone
print(f"\n  ── For comparison: weight alone, rec alone ──")
for comp, col in [('weight', 's_weight'), ('rec_prod', 's_rec_prod')]:
    for out in outcomes_list:
        valid = rb_eval[[col, out]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid[col], valid[out])
            print(f"    r({comp}, {out}) = {r:+.4f}  (p={p:.4f})")
    print()

# 3b: PARTIAL correlation — does the interaction add value AFTER controlling for DC + rec prod?
print(f"\n  ── Partial correlation: interaction after controlling for DC + Rec Prod ──")
print(f"  (This is the real test: does weight×rec add NEW information?)")

for out in outcomes_list:
    valid = rb_eval[['s_dc', 's_rec_prod_filled', 's_wt_x_rec_filled', out]].dropna()
    if len(valid) < 20:
        print(f"    {out}: insufficient data")
        continue

    # Partial correlation: regress interaction and outcome each on DC + rec, correlate residuals
    from numpy.linalg import lstsq
    X = np.column_stack([valid['s_dc'].values, valid['s_rec_prod_filled'].values, np.ones(len(valid))])

    # Residualize interaction
    beta_int, _, _, _ = lstsq(X, valid['s_wt_x_rec_filled'].values, rcond=None)
    resid_int = valid['s_wt_x_rec_filled'].values - X @ beta_int

    # Residualize outcome
    beta_out, _, _, _ = lstsq(X, valid[out].values, rcond=None)
    resid_out = valid[out].values - X @ beta_out

    r, p = stats.pearsonr(resid_int, resid_out)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else ""
    print(f"    partial r(wt×rec | DC,rec) → {out}: {r:+.4f}  (p={p:.4f}) {sig}")

# 3c: Also do partial correlation for weight alone after controlling for DC + rec
print(f"\n  ── Partial correlation: WEIGHT ALONE after controlling for DC + Rec Prod ──")

for out in outcomes_list:
    valid = rb_eval[['s_dc', 's_rec_prod_filled', 's_weight_filled', out]].dropna()
    if len(valid) < 20:
        print(f"    {out}: insufficient data")
        continue

    X = np.column_stack([valid['s_dc'].values, valid['s_rec_prod_filled'].values, np.ones(len(valid))])

    beta_wt, _, _, _ = lstsq(X, valid['s_weight_filled'].values, rcond=None)
    resid_wt = valid['s_weight_filled'].values - X @ beta_wt

    beta_out, _, _, _ = lstsq(X, valid[out].values, rcond=None)
    resid_out = valid[out].values - X @ beta_out

    r, p = stats.pearsonr(resid_wt, resid_out)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else ""
    print(f"    partial r(weight | DC,rec) → {out}: {r:+.4f}  (p={p:.4f}) {sig}")

# 3d: Test as a SLAP component: DC/Rec/Interaction at various weights
print(f"\n\n  ── Model test: DC/Rec/Wt×Rec interaction at 5% and 10% ──")

for inter_w, dc_w, rec_w in [(0.05, 0.60, 0.35), (0.10, 0.55, 0.35), (0.10, 0.60, 0.30)]:
    label = f"DC/Rec/Wt×Rec {int(dc_w*100)}/{int(rec_w*100)}/{int(inter_w*100)}"
    slap = rb_eval['s_dc'] * dc_w + rb_eval['s_rec_prod_filled'] * rec_w + rb_eval['s_wt_x_rec_filled'] * inter_w
    m = full_metrics(slap, rb_eval)
    d_pri = m['pri_avg'] - base_m['pri_avg']

    # Count improved/hurt
    improved = 0
    hurt = 0
    for key, higher_better in metric_keys:
        v = m.get(key, np.nan)
        b = base_m.get(key, np.nan)
        if np.isnan(v) or np.isnan(b): continue
        delta = v - b
        if higher_better:
            if delta > 0.0001: improved += 1
            elif delta < -0.0001: hurt += 1
        else:
            if delta < -0.0001: improved += 1
            elif delta > 0.0001: hurt += 1

    print(f"    {label:<35} PRI-AVG: {m['pri_avg']:+.4f} (Δ={d_pri:+.4f})  Improved: {improved}  Hurt: {hurt-0}  Net: {improved-hurt:+d}")
    print(f"      r(3yr_ppg)={m['r_first_3yr_ppg']:+.4f}  r(hit24)={m['r_hit24']:+.4f}  r(hit12)={m['r_hit12']:+.4f}  r(career)={m['r_career_ppg']:+.4f}")
    print(f"      T10% hit24={m['top10_hit24']:.1f}%  T10% hit12={m['top10_hit12']:.1f}%  AUC h24={m['auc_hit24']:.4f}  AUC h12={m['auc_hit12']:.4f}")

# 3e: Explore the actual interaction visually — heavy receivers who catch passes
print(f"\n\n  ── Who are the heavy pass-catching RBs? Top-20 by interaction score ──")
print(f"  {'Player':<25} {'Year':>4} {'Pick':>4} {'Wt':>5} {'RecProd':>8} {'Wt×Rec':>7} {'hit24':>5} {'hit12':>5} {'3yr PPG':>8}")
print("  " + "-" * 80)

top_inter = rb_eval[rb_eval['s_wt_x_rec'].notna()].nlargest(20, 's_wt_x_rec')
for _, row in top_inter.iterrows():
    wt_s = f"{row['weight']:.0f}" if pd.notna(row['weight']) else "?"
    rec_s = f"{row['s_rec_prod']:.1f}" if pd.notna(row['s_rec_prod']) else "?"
    inter_s = f"{row['s_wt_x_rec']:.1f}" if pd.notna(row['s_wt_x_rec']) else "?"
    h24 = f"{int(row['hit24'])}" if pd.notna(row['hit24']) else "?"
    h12 = f"{int(row['hit12'])}" if pd.notna(row['hit12']) else "?"
    ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "?"
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>4} {int(row['pick']):>4} {wt_s:>5} {rec_s:>8} {inter_s:>7} {h24:>5} {h12:>5} {ppg:>8}")

h24_rate = top_inter['hit24'].mean() * 100
h12_rate = top_inter['hit12'].mean() * 100
avg_ppg = top_inter['first_3yr_ppg'].dropna().mean()
print(f"\n  Top-20 Wt×Rec: hit24={h24_rate:.1f}%  hit12={h12_rate:.1f}%  3yr_ppg={avg_ppg:.2f}")
print(f"  Sample avg:    hit24={rb_eval['hit24'].mean()*100:.1f}%  hit12={rb_eval['hit12'].mean()*100:.1f}%  3yr_ppg={rb_eval['first_3yr_ppg'].dropna().mean():.2f}")


print(f"\n\n{'=' * 120}")
print("ANALYSIS COMPLETE")
print("=" * 120)
