"""
RB Speed Score — Full recovery, MNAR analysis, gut-check, and model testing
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq as np_lstsq
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
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
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

def auc_roc(labels, scores):
    valid = pd.DataFrame({'y': labels, 's': scores}).dropna()
    if len(valid) < 10: return np.nan
    pos = valid[valid['y'] == 1]['s']
    neg = valid[valid['y'] == 0]['s']
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return sum((neg < p).sum() + 0.5 * (neg == p).sum() for p in pos) / (len(pos) * len(neg))

def brier_score(labels, scores_0_100):
    valid = pd.DataFrame({'y': labels, 's': scores_0_100}).dropna()
    if len(valid) < 10: return np.nan
    return ((valid['s'] / 100 - valid['y']) ** 2).mean()

outcome_cols = ['first_3yr_ppg', 'hit24', 'hit12', 'career_ppg']
outcome_wts = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}

# ============================================================================
# DATA LOADING WITH FIXED MATCHING
# ============================================================================

rb = pd.read_csv('data/rb_backtest_with_receiving.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
out_rb = outcomes[outcomes['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
rb = rb.merge(out_rb, on=['player_name', 'draft_year', 'pick'], how='left')
rb['s_dc'] = rb['pick'].apply(dc_score)
rb['s_rec_prod'] = rb.apply(rb_production_score, axis=1)
rb['name_norm'] = rb['player_name'].apply(normalize_name)

# Fixed combine matching (all positions, fixed names)
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine['name_norm'] = combine['player_name'].apply(normalize_name)

# Build lookup: prefer RB, then FB, then WR
combine_lookup = {}
for pos_pref in ['RB', 'FB', 'WR', 'TE']:
    sub = combine[combine['pos'] == pos_pref]
    for _, row in sub.iterrows():
        dy = row.get('draft_year', row.get('season'))
        if pd.isna(dy): continue
        key = (row['name_norm'], int(dy))
        if key not in combine_lookup:
            combine_lookup[key] = {'weight': row['wt'], 'forty': row['forty'], 'pos': row['pos'],
                                   'orig_name': row['player_name']}

rb['weight'] = rb.apply(lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb['forty'] = rb.apply(lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb['weight'] = pd.to_numeric(rb['weight'], errors='coerce')
rb['forty'] = pd.to_numeric(rb['forty'], errors='coerce')
rb['speed_score'] = rb.apply(lambda r: speed_score_fn(r['weight'], r['forty']), axis=1)

rb_eval = rb[rb['hit24'].notna()].copy()

# ============================================================================
# COVERAGE REPORT
# ============================================================================

print("=" * 120)
print("SPEED SCORE COVERAGE")
print("=" * 120)

n_wt = rb_eval['weight'].notna().sum()
n_40 = rb_eval['forty'].notna().sum()
n_ss = rb_eval['speed_score'].notna().sum()
n_both = rb_eval[rb_eval['weight'].notna() & rb_eval['forty'].notna()].shape[0]

print(f"\n  Weight:      {n_wt}/223 ({n_wt/223*100:.0f}%)")
print(f"  40 time:     {n_40}/223 ({n_40/223*100:.0f}%)")
print(f"  Both:        {n_both}/223 ({n_both/223*100:.0f}%)")
print(f"  Speed Score: {n_ss}/223 ({n_ss/223*100:.0f}%)")

# Every Rd 1-3 RB missing either measurement
print(f"\n\n  ── ROUNDS 1-3 RBs MISSING WEIGHT OR 40 TIME ──")
rd13 = rb_eval[(rb_eval['round'] <= 3) & (rb_eval['weight'].isna() | rb_eval['forty'].isna())].sort_values('pick')
print("  %-25s %4s %4s %3s %6s %6s %5s %8s %6s" % (
    'Player', 'Year', 'Pick', 'Rd', 'Wt', '40', 'hit24', '3yr PPG', 'RAS'))
print("  " + "-" * 85)
for _, row in rd13.iterrows():
    wt = "%.0f" % row['weight'] if pd.notna(row['weight']) else "MISS"
    ft = "%.2f" % row['forty'] if pd.notna(row['forty']) else "MISS"
    ppg = "%.1f" % row['first_3yr_ppg'] if pd.notna(row['first_3yr_ppg']) else "?"
    ras = "%.2f" % row['RAS'] if pd.notna(row['RAS']) else "MISS"
    print("  %-25s %4d %4d %3d %6s %6s %5d %8s %6s" % (
        row['player_name'], int(row['draft_year']), int(row['pick']), int(row['round']),
        wt, ft, int(row['hit24']), ppg, ras))

# ============================================================================
# MNAR ANALYSIS
# ============================================================================

print(f"\n\n{'=' * 120}")
print("MNAR ANALYSIS: Are missing players systematically different?")
print("=" * 120)

print(f"\n  ── SPEED SCORE MISSINGNESS BY ROUND ──")
print("  %8s %5s %8s %8s %10s %10s %10s" % (
    'Round', 'n', 'Has SS', 'Miss SS', 'Hit24(has)', 'Hit24(miss)', 'Pattern'))
print("  " + "-" * 65)

for rd in sorted(rb_eval['round'].unique()):
    grp = rb_eval[rb_eval['round'] == rd]
    has = grp[grp['speed_score'].notna()]
    miss = grp[grp['speed_score'].isna()]
    h_rate = has['hit24'].mean() * 100 if len(has) > 0 else 0
    m_rate = miss['hit24'].mean() * 100 if len(miss) > 0 else 0

    if len(miss) > 2:
        pattern = "MISS BETTER" if m_rate > h_rate + 5 else "MISS WORSE" if m_rate < h_rate - 5 else "SIMILAR"
    else:
        pattern = "n/a (few)"

    print("  %8d %5d %8d %8d %9.1f%% %9.1f%% %10s" % (
        rd, len(grp), len(has), len(miss), h_rate, m_rate, pattern))

# Overall
has_ss = rb_eval[rb_eval['speed_score'].notna()]
miss_ss = rb_eval[rb_eval['speed_score'].isna()]
print(f"\n  Overall: has SS {len(has_ss)} ({has_ss['hit24'].mean()*100:.1f}% hit24) | "
      f"miss SS {len(miss_ss)} ({miss_ss['hit24'].mean()*100:.1f}% hit24)")
print(f"  Avg pick: has SS {has_ss['pick'].mean():.1f} | miss SS {miss_ss['pick'].mean():.1f}")

# Detailed: Rd 1-2 missing vs Rd 3+ missing
rd12_miss = miss_ss[miss_ss['round'] <= 2]
rd3p_miss = miss_ss[miss_ss['round'] >= 3]
print(f"\n  Rd 1-2 missing: n={len(rd12_miss)}, hit24={rd12_miss['hit24'].mean()*100:.1f}%")
if len(rd12_miss) > 0:
    for _, row in rd12_miss.sort_values('pick').iterrows():
        ppg = "%.1f" % row['first_3yr_ppg'] if pd.notna(row['first_3yr_ppg']) else "?"
        print(f"    {row['player_name']:<25} pick {int(row['pick']):>3} hit24={int(row['hit24'])} PPG={ppg}")
print(f"  Rd 3+ missing:  n={len(rd3p_miss)}, hit24={rd3p_miss['hit24'].mean()*100:.1f}%")


# ============================================================================
# SPEED SCORE CALCULATION AND GUT-CHECK
# ============================================================================

print(f"\n\n{'=' * 120}")
print("SPEED SCORE GUT-CHECK")
print("=" * 120)

ss_eval = rb_eval[rb_eval['speed_score'].notna()].copy()
print(f"\n  Speed Score: n={len(ss_eval)}, min={ss_eval['speed_score'].min():.1f}, "
      f"median={ss_eval['speed_score'].median():.1f}, max={ss_eval['speed_score'].max():.1f}")

print(f"\n  ── TOP 20 BY SPEED SCORE ──")
print("  %3s %-25s %4s %4s %5s %6s %8s %5s %5s %8s" % (
    '#', 'Player', 'Year', 'Pick', 'Wt', '40', 'SpScore', 'hit24', 'hit12', '3yr PPG'))
print("  " + "-" * 95)
for i, (_, row) in enumerate(ss_eval.nlargest(20, 'speed_score').iterrows(), 1):
    ppg = "%.1f" % row['first_3yr_ppg'] if pd.notna(row['first_3yr_ppg']) else "?"
    print("  %3d %-25s %4d %4d %5.0f %6.2f %8.1f %5d %5d %8s" % (
        i, row['player_name'], int(row['draft_year']), int(row['pick']),
        row['weight'], row['forty'], row['speed_score'],
        int(row['hit24']), int(row['hit12']), ppg))

print(f"\n  ── BOTTOM 20 BY SPEED SCORE ──")
print("  %3s %-25s %4s %4s %5s %6s %8s %5s %5s %8s" % (
    '#', 'Player', 'Year', 'Pick', 'Wt', '40', 'SpScore', 'hit24', 'hit12', '3yr PPG'))
print("  " + "-" * 95)
for i, (_, row) in enumerate(ss_eval.nsmallest(20, 'speed_score').iterrows(), 1):
    ppg = "%.1f" % row['first_3yr_ppg'] if pd.notna(row['first_3yr_ppg']) else "?"
    print("  %3d %-25s %4d %4d %5.0f %6.2f %8.1f %5d %5d %8s" % (
        i, row['player_name'], int(row['draft_year']), int(row['pick']),
        row['weight'], row['forty'], row['speed_score'],
        int(row['hit24']), int(row['hit12']), ppg))

# Hit rate by speed score quartile
print(f"\n  ── HIT RATES BY SPEED SCORE QUARTILE ──")
ss_eval['ss_quartile'] = pd.qcut(ss_eval['speed_score'], 4, labels=['Q1(slow)', 'Q2', 'Q3', 'Q4(fast)'])
for q in ['Q1(slow)', 'Q2', 'Q3', 'Q4(fast)']:
    grp = ss_eval[ss_eval['ss_quartile'] == q]
    ppg_avg = grp['first_3yr_ppg'].dropna().mean()
    print("  %-10s n=%3d | hit24=%5.1f%% | hit12=%5.1f%% | avg 3yr_ppg=%5.2f | avg pick=%5.1f" % (
        q, len(grp), grp['hit24'].mean()*100, grp['hit12'].mean()*100, ppg_avg, grp['pick'].mean()))


# ============================================================================
# NORMALIZE AND IMPUTE SPEED SCORE
# ============================================================================

# Normalize speed score to 0-100
ss_valid = rb_eval['speed_score'].notna()
rb_eval['ss_norm'] = np.nan
rb_eval.loc[ss_valid, 'ss_norm'] = normalize_0_100(rb_eval.loc[ss_valid, 'speed_score'])

# Imputation Strategy 1: Overall average
avg_ss = rb_eval['ss_norm'].mean()
rb_eval['ss_avg_imp'] = rb_eval['ss_norm'].fillna(avg_ss)

# Imputation Strategy 2: Round average
round_avg_ss = {}
for rd in rb_eval['round'].unique():
    grp = rb_eval[(rb_eval['round'] == rd) & rb_eval['ss_norm'].notna()]
    round_avg_ss[rd] = grp['ss_norm'].mean() if len(grp) > 0 else avg_ss
rb_eval['ss_round_imp'] = rb_eval.apply(
    lambda r: r['ss_norm'] if pd.notna(r['ss_norm']) else round_avg_ss.get(r['round'], avg_ss), axis=1)

# Imputation Strategy 3: MNAR-aware (Rd 1-2 miss → 60th pctile, Rd 3+ miss → 40th pctile)
p60 = rb_eval['ss_norm'].quantile(0.60)
p40 = rb_eval['ss_norm'].quantile(0.40)
def mnar_impute(row):
    if pd.notna(row['ss_norm']):
        return row['ss_norm']
    if row['round'] <= 2:
        return p60
    else:
        return p40
rb_eval['ss_mnar_imp'] = rb_eval.apply(mnar_impute, axis=1)

print(f"\n\n  ── IMPUTATION VALUES ──")
print(f"  Overall average: {avg_ss:.1f}")
print(f"  Round averages: {', '.join('Rd%d=%.1f' % (rd, v) for rd, v in sorted(round_avg_ss.items()))}")
print(f"  MNAR: Rd 1-2 missing → {p60:.1f} (60th pctile), Rd 3+ missing → {p40:.1f} (40th pctile)")

# Rec prod imputed
avg_rec = rb_eval['s_rec_prod'].mean()
rb_eval['s_rec_f'] = rb_eval['s_rec_prod'].fillna(avg_rec)


# ============================================================================
# MODEL TESTING: 5 CONFIGS × 2 IMPUTATION STRATEGIES
# ============================================================================

print(f"\n\n{'=' * 120}")
print("MODEL TESTING: 5 CONFIGS × 2 IMPUTATION STRATEGIES")
print(f"Sample: {len(rb_eval)} RBs | Top decile = {max(1, len(rb_eval)//10)} players")
print("=" * 120)

def full_eval(df, slap):
    res = {}
    for out in outcome_cols:
        valid = pd.DataFrame({'s': slap, 'o': df[out]}).dropna()
        if len(valid) >= 10:
            r, _ = stats.pearsonr(valid['s'], valid['o'])
            res[f'r_{out}'] = r
    pri = sum(outcome_wts[o] * res.get(f'r_{o}', 0) for o in outcome_cols
              if not np.isnan(res.get(f'r_{o}', np.nan)))
    pri_w = sum(outcome_wts[o] for o in outcome_cols
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

configs = [
    ('A: 65/35/0 (baseline)',   0.65, 0.35, 0.00),
    ('B: 62/33/5',              0.62, 0.33, 0.05),
    ('C: 60/35/5',              0.60, 0.35, 0.05),
    ('D: 63/34/3',              0.63, 0.34, 0.03),
    ('E: 60/33/7',              0.60, 0.33, 0.07),
]

imp_strategies = [
    ('Round-avg impute', 'ss_round_imp'),
    ('MNAR-aware impute', 'ss_mnar_imp'),
]

for imp_label, ss_col in imp_strategies:
    print(f"\n  ── {imp_label} ──")
    print("  %-25s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s" % (
        'Config', 'PRI-AVG', 'r(3yr)', 'r(h24)', 'r(h12)', 'r(car)',
        'T10%h24', 'T10%h12', 'T10%PPG', 'AUC h24', 'AUC h12', 'Brier24', 'Brier12'))
    print("  " + "-" * 130)

    base_res = None
    for label, dc_w, rec_w, ss_w in configs:
        if ss_w == 0:
            slap = rb_eval['s_dc'] * dc_w + rb_eval['s_rec_f'] * rec_w
        else:
            slap = rb_eval['s_dc'] * dc_w + rb_eval['s_rec_f'] * rec_w + rb_eval[ss_col] * ss_w
        res = full_eval(rb_eval, slap)
        if base_res is None:
            base_res = res
        d_pri = res['pri_avg'] - base_res['pri_avg']
        marker = " ◄ BASE" if ss_w == 0 else " (%+.4f)" % d_pri
        print("  %-25s %+.4f %+.4f %+.4f %+.4f %+.4f %6.1f%% %6.1f%% %7.2f %.4f  %.4f  %.4f  %.4f%s" % (
            label, res['pri_avg'],
            res.get('r_first_3yr_ppg', 0), res.get('r_hit24', 0),
            res.get('r_hit12', 0), res.get('r_career_ppg', 0),
            res['top10_hit24'], res['top10_hit12'],
            res.get('top10_ppg', 0),
            res['auc_hit24'], res['auc_hit12'],
            res['brier_hit24'], res['brier_hit12'],
            marker))


# ============================================================================
# DELTA TABLE: Every config vs baseline on all 12 metrics
# ============================================================================

print(f"\n\n{'=' * 120}")
print("DELTA TABLE: Every config vs 65/35 baseline")
print("For Brier, negative = improvement. For everything else, positive = improvement.")
print("=" * 120)

base_slap = rb_eval['s_dc'] * 0.65 + rb_eval['s_rec_f'] * 0.35
base_m = full_eval(rb_eval, base_slap)

metric_keys = [
    ('r_first_3yr_ppg', 'r(3yr_ppg)',   True),
    ('r_hit24',         'r(hit24)',      True),
    ('r_hit12',         'r(hit12)',      True),
    ('r_career_ppg',    'r(career)',     True),
    ('pri_avg',         'PRI-AVG',       True),
    ('top10_hit24',     'T10% h24',      True),
    ('top10_hit12',     'T10% h12',      True),
    ('top10_ppg',       'T10% PPG',      True),
    ('auc_hit24',       'AUC h24',       True),
    ('auc_hit12',       'AUC h12',       True),
    ('brier_hit24',     'Brier h24',     False),
    ('brier_hit12',     'Brier h12',     False),
]

for imp_label, ss_col in imp_strategies:
    print(f"\n  ── {imp_label} ──")

    # Collect all results
    all_res = {}
    for label, dc_w, rec_w, ss_w in configs:
        if ss_w == 0:
            slap = rb_eval['s_dc'] * dc_w + rb_eval['s_rec_f'] * rec_w
        else:
            slap = rb_eval['s_dc'] * dc_w + rb_eval['s_rec_f'] * rec_w + rb_eval[ss_col] * ss_w
        all_res[label] = full_eval(rb_eval, slap)

    # Print header
    header = "  %-15s" % 'Metric'
    for label, _, _, ss_w in configs[1:]:  # skip baseline
        short = label.split(':')[1].strip()
        header += " %16s" % short
    print(header)
    print("  " + "-" * (15 + 17 * len(configs[1:])))

    # Print each metric's delta
    improve_counts = {l: 0 for l, _, _, _ in configs[1:]}
    hurt_counts = {l: 0 for l, _, _, _ in configs[1:]}

    for key, mlabel, higher_better in metric_keys:
        row = "  %-15s" % mlabel
        b = base_m[key]
        for label, _, _, _ in configs[1:]:
            v = all_res[label][key]
            delta = v - b
            if higher_better:
                if delta > 0.0005:
                    improve_counts[label] += 1
                    marker = "+"
                elif delta < -0.0005:
                    hurt_counts[label] += 1
                    marker = "-"
                else:
                    marker = "~"
            else:
                if delta < -0.0005:
                    improve_counts[label] += 1
                    marker = "+"
                elif delta > 0.0005:
                    hurt_counts[label] += 1
                    marker = "-"
                else:
                    marker = "~"
            row += " %+13.4f %s " % (delta, marker)
        print(row)

    # Summary
    print()
    row = "  %-15s" % 'Improved'
    for label, _, _, _ in configs[1:]:
        row += " %16d" % improve_counts[label]
    print(row)
    row = "  %-15s" % 'Hurt'
    for label, _, _, _ in configs[1:]:
        row += " %16d" % hurt_counts[label]
    print(row)
    row = "  %-15s" % 'Net'
    for label, _, _, _ in configs[1:]:
        row += " %+15d " % (improve_counts[label] - hurt_counts[label])
    print(row)


# ============================================================================
# SCORECARD: Best config across both imputation strategies
# ============================================================================

print(f"\n\n{'=' * 120}")
print("SCORECARD: BEST APPROACH")
print("=" * 120)

best_pri = base_m['pri_avg']
best_label = "A: 65/35/0 (baseline)"
best_imp = "none"

for imp_label, ss_col in imp_strategies:
    for label, dc_w, rec_w, ss_w in configs:
        if ss_w == 0: continue
        slap = rb_eval['s_dc'] * dc_w + rb_eval['s_rec_f'] * rec_w + rb_eval[ss_col] * ss_w
        res = full_eval(rb_eval, slap)
        if res['pri_avg'] > best_pri:
            best_pri = res['pri_avg']
            best_label = label
            best_imp = imp_label

print(f"\n  Baseline (65/35): PRI-AVG = {base_m['pri_avg']:+.4f}, T10% hit24 = {base_m['top10_hit24']:.1f}%")
print(f"  Best with SS:     {best_label} ({best_imp}): PRI-AVG = {best_pri:+.4f}")
print(f"  Delta PRI-AVG:    {best_pri - base_m['pri_avg']:+.4f}")

# Show what the best SS config does to top-decile
if best_label != "A: 65/35/0 (baseline)":
    for imp_label, ss_col in imp_strategies:
        for label, dc_w, rec_w, ss_w in configs:
            if label == best_label:
                slap = rb_eval['s_dc'] * dc_w + rb_eval['s_rec_f'] * rec_w + rb_eval[ss_col] * ss_w
                res = full_eval(rb_eval, slap)
                print(f"\n  {best_label} with {imp_label}:")
                print(f"    PRI-AVG: {res['pri_avg']:+.4f} (Δ = {res['pri_avg']-base_m['pri_avg']:+.4f})")
                print(f"    T10% hit24: {res['top10_hit24']:.1f}% (Δ = {res['top10_hit24']-base_m['top10_hit24']:+.1f}%)")
                print(f"    T10% hit12: {res['top10_hit12']:.1f}% (Δ = {res['top10_hit12']-base_m['top10_hit12']:+.1f}%)")
                print(f"    AUC hit24: {res['auc_hit24']:.4f} (Δ = {res['auc_hit24']-base_m['auc_hit24']:+.4f})")
                print(f"    AUC hit12: {res['auc_hit12']:.4f} (Δ = {res['auc_hit12']-base_m['auc_hit12']:+.4f})")


print(f"\n\n{'=' * 120}")
print("ANALYSIS COMPLETE")
print("=" * 120)
