"""
DIAGNOSTIC: Percentile-Rank Normalization for Cross-Position Calibration
========================================================================
Rescales all non-DC components to percentile ranks WITHIN position,
using BACKTEST data only as the reference distribution.

2026 prospects are scored against the backtest distribution.

Binary components (Teammate, Early Declare) are LEFT UNCHANGED — they're
already 0/100 by design and work the same across positions.

DOES NOT SAVE ANYTHING — diagnostic output only.
"""

import pandas as pd
import numpy as np
import warnings, os
from scipy import stats as sp_stats
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

# ============================================================================
# HELPERS (copied from build_master_database_v5.py)
# ============================================================================

def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()

def wr_breakout_score(breakout_age, dominator_pct):
    if pd.isna(breakout_age) or breakout_age is None:
        if dominator_pct is not None and pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base = age_tiers.get(int(breakout_age), 20)
    bonus = 0
    if dominator_pct is not None and pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
    return min(base + bonus, 99.9)

def wr_enhanced_breakout(breakout_age, dominator_pct, rush_yards):
    base = wr_breakout_score(breakout_age, dominator_pct)
    rush_bonus = 5 if pd.notna(rush_yards) and rush_yards >= 20 else 0
    return min(base + rush_bonus, 99.9)

def rb_production_score(rec_yards, team_pass_att, age):
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return np.nan
    try:
        age = float(age)
    except (TypeError, ValueError):
        age = 22.0
    if pd.isna(age): age = 22
    season_age = age - 1
    age_w = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
    raw = (rec_yards / team_pass_att) * age_w * 100
    return min(99.9, max(0, raw / 1.75))

def speed_score_fn(weight, forty):
    if pd.isna(weight) or pd.isna(forty) or forty <= 0 or weight <= 0:
        return np.nan
    return (weight * 200) / (forty ** 4)

def normalize_0_100(series):
    mn, mx = series.dropna().min(), series.dropna().max()
    if mx == mn: return pd.Series(50, index=series.index)
    return ((series - mn) / (mx - mn) * 100).clip(0, 100)

def te_breakout_score(breakout_age, peak_dominator, threshold=15):
    if pd.isna(breakout_age) or breakout_age is None:
        if peak_dominator is not None and pd.notna(peak_dominator):
            return min(35, 15 + peak_dominator)
        return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base = age_tiers.get(int(breakout_age), 20)
    bonus = min((peak_dominator - threshold) * 0.5, 9.9) if pd.notna(peak_dominator) and peak_dominator >= threshold else 0
    return min(base + bonus, 99.9)

def te_production_score_fn(rec_yards, team_pass_att, draft_age, draft_year, season_year=None):
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return np.nan
    if season_year is None:
        season_year = draft_year - 1
    season_age = draft_age - (draft_year - season_year) if pd.notna(draft_age) else 22
    if season_age <= 19: aw = 1.15
    elif season_age <= 20: aw = 1.10
    elif season_age <= 21: aw = 1.05
    elif season_age <= 22: aw = 1.00
    elif season_age <= 23: aw = 0.95
    else: aw = 0.90
    return (rec_yards / team_pass_att) * aw * 100


def percentile_rank(series):
    """Convert a series to percentile ranks (0-100). Ties get average rank.
    NaN values stay NaN."""
    valid = series.dropna()
    if len(valid) == 0:
        return series.copy()
    # rankdata gives 1-based ranks, average method for ties
    ranks = sp_stats.rankdata(valid.values, method='average')
    # Convert to percentile: (rank - 0.5) / n * 100
    # This maps the lowest to ~0 and highest to ~100
    pctiles = (ranks - 0.5) / len(ranks) * 100
    result = series.copy().astype(float)
    result.loc[valid.index] = pctiles
    return result


def score_prospect_against_backtest(prospect_val, backtest_values):
    """Score a single prospect value as a percentile against the backtest distribution.
    Returns 0-100 percentile."""
    if pd.isna(prospect_val):
        return np.nan
    bt = backtest_values.dropna().values
    if len(bt) == 0:
        return 50.0
    # What fraction of backtest values is this prospect above?
    below = np.sum(bt < prospect_val)
    tied = np.sum(bt == prospect_val)
    pctile = (below + 0.5 * tied) / len(bt) * 100
    return min(99.9, max(0.1, pctile))


# ============================================================================
# WEIGHTS (unchanged)
# ============================================================================
WR_V5 = {'dc': 0.70, 'breakout': 0.20, 'teammate': 0.05, 'early_declare': 0.05}
RB_V5 = {'dc': 0.65, 'production': 0.30, 'speed_score': 0.05}
TE_V5 = {'dc': 0.60, 'breakout': 0.15, 'production': 0.15, 'ras': 0.10}

print("=" * 120)
print("DIAGNOSTIC: Percentile-Rank Normalization")
print("=" * 120)
print(f"\n  Method: rank all backtest players within position → convert to percentile (0-100)")
print(f"  50 = position median, ~99 = best at position, ~1 = worst at position")
print(f"  Binary components (Teammate, Early Declare) UNCHANGED")
print(f"  Weights UNCHANGED — only component scaling changes")


# ============================================================================
# STEP 1: LOAD AND COMPUTE RAW COMPONENT SCORES (same as build script)
# ============================================================================
print(f"\n\n{'='*120}")
print("STEP 1: Computing raw component scores (same formulas as before)")
print("=" * 120)

# --- WR BACKTEST ---
wr_bt = pd.read_csv('data/wr_backtest_all_components.csv')
wr_tm = pd.read_csv('data/wr_teammate_scores.csv')
wr_bt = wr_bt.merge(wr_tm[['player_name', 'draft_year', 'total_teammate_dc']], on=['player_name', 'draft_year'], how='left')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
wr_out = outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
wr_bt = wr_bt.merge(wr_out, on=['player_name', 'draft_year', 'pick'], how='left')

wr_bt['s_dc'] = wr_bt['pick'].apply(dc_score)
wr_bt['s_breakout_raw'] = wr_bt.apply(
    lambda r: wr_enhanced_breakout(r['breakout_age'], r['peak_dominator'], r['rush_yards']), axis=1)
wr_bt['s_teammate'] = wr_bt['total_teammate_dc'].apply(lambda x: 100 if pd.notna(x) and x > 150 else 0)
wr_bt['s_early_declare'] = wr_bt['early_declare'].apply(lambda x: 100 if x == 1 else 0)
print(f"  WR backtest: {len(wr_bt)} players loaded")

# --- RB BACKTEST ---
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
rb_out = outcomes[outcomes['position'] == 'RB'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
rb_bt = rb_bt.merge(rb_out, on=['player_name', 'draft_year', 'pick'], how='left')

rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)
rb_bt['s_production_raw'] = rb_bt.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb_bt['s_production_raw'] = rb_bt['s_production_raw'].fillna(0)

# Speed Score (same MNAR pipeline)
rb_bt['name_norm'] = rb_bt['player_name'].apply(normalize_name)
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine['name_norm'] = combine['player_name'].apply(normalize_name)
combine_lookup = {}
for pos_pref in ['RB', 'FB', 'WR', 'TE']:
    sub = combine[combine['pos'] == pos_pref]
    for _, row in sub.iterrows():
        dy = row.get('draft_year')
        if pd.isna(dy): dy = row.get('season')
        if pd.isna(dy): continue
        key = (row['name_norm'], int(dy))
        if key not in combine_lookup:
            combine_lookup[key] = {'weight': row['wt'], 'forty': row['forty']}

rb_bt['weight'] = rb_bt.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb_bt['forty'] = rb_bt.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb_bt['weight'] = pd.to_numeric(rb_bt['weight'], errors='coerce')
rb_bt['forty'] = pd.to_numeric(rb_bt['forty'], errors='coerce')

known = rb_bt[rb_bt['weight'].notna() & rb_bt['forty'].notna()].copy()
def wt_bucket(wt):
    if pd.isna(wt): return None
    if wt < 200: return '<200'
    elif wt < 210: return '200-209'
    elif wt < 220: return '210-219'
    else: return '220+'
def rd_bucket(rd):
    if rd <= 1: return 'Rd 1'
    elif rd <= 2: return 'Rd 2'
    elif rd <= 4: return 'Rd 3-4'
    else: return 'Rd 5+'
known['wb'] = known['weight'].apply(wt_bucket)
known['rb_bkt'] = known['round'].apply(rd_bucket)
lookup_40 = {}
for wb in ['<200', '200-209', '210-219', '220+']:
    for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
        sub_k = known[(known['wb'] == wb) & (known['rb_bkt'] == rdb)]
        if len(sub_k) > 0:
            lookup_40[(wb, rdb)] = sub_k['forty'].mean()
    wt_sub = known[known['wb'] == wb]
    if len(wt_sub) > 0:
        for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
            if (wb, rdb) not in lookup_40:
                lookup_40[(wb, rdb)] = wt_sub['forty'].mean()

rb_bt['forty_final'] = rb_bt['forty'].copy()
impute_mask = rb_bt['weight'].notna() & rb_bt['forty'].isna()
for idx in rb_bt[impute_mask].index:
    wb = wt_bucket(rb_bt.loc[idx, 'weight'])
    rdb = rd_bucket(rb_bt.loc[idx, 'round'])
    est = lookup_40.get((wb, rdb))
    if est is not None:
        rb_bt.loc[idx, 'forty_final'] = est

rb_bt['raw_ss'] = rb_bt.apply(lambda r: speed_score_fn(r['weight'], r['forty_final']), axis=1)
real_ss = rb_bt['raw_ss'].dropna()
p60 = real_ss.quantile(0.60)
p40 = real_ss.quantile(0.40)
for idx in rb_bt[rb_bt['raw_ss'].isna()].index:
    rd = rb_bt.loc[idx, 'round']
    rb_bt.loc[idx, 'raw_ss'] = p60 if rd <= 2 else p40
rb_bt['s_speed_raw'] = normalize_0_100(rb_bt['raw_ss'])
print(f"  RB backtest: {len(rb_bt)} players loaded")

# --- TE BACKTEST ---
te_bt = pd.read_csv('data/te_backtest_master.csv')
te_bt['s_dc'] = te_bt['pick'].apply(dc_score)

te_bt['s_breakout_raw'] = te_bt.apply(
    lambda r: te_breakout_score(r['breakout_age'], r['peak_dominator'], threshold=15), axis=1)

te_bt['te_prod_raw'] = te_bt.apply(lambda r: te_production_score_fn(
    r['cfbd_rec_yards'], r['cfbd_team_pass_att'], r['draft_age'], r['draft_year']), axis=1)
# Manual patches
for name, vals in {
    'Dallas Goedert': {'cfbd_rec_yards': 1111, 'cfbd_team_pass_att': 455, 'draft_age': 23.0, 'draft_year': 2018},
    'Adam Shaheen': {'cfbd_rec_yards': 867, 'cfbd_team_pass_att': 328, 'draft_age': 22.3, 'draft_year': 2017},
}.items():
    mask = te_bt['player_name'] == name
    if mask.sum() > 0 and pd.isna(te_bt.loc[mask, 'te_prod_raw'].values[0]):
        te_bt.loc[mask, 'te_prod_raw'] = te_production_score_fn(
            vals['cfbd_rec_yards'], vals['cfbd_team_pass_att'], vals['draft_age'], vals['draft_year'])

# For TE production — first get raw values, then we'll normalize
# Also old min-max for reference
prod_vals = te_bt['te_prod_raw'].dropna()
te_prod_min = prod_vals.min()
te_prod_max = prod_vals.max()
te_bt['s_production_minmax'] = np.where(
    te_bt['te_prod_raw'].notna(),
    ((te_bt['te_prod_raw'] - te_prod_min) / (te_prod_max - te_prod_min) * 99.9).clip(0, 99.9),
    np.nan
)
prod_avg_te_mm = te_bt['s_production_minmax'].mean()
te_bt['s_production_minmax_filled'] = te_bt['s_production_minmax'].fillna(prod_avg_te_mm)

# RAS
te_bt['s_ras_raw'] = te_bt['te_ras'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)
ras_real = te_bt['s_ras_raw'].dropna()
te_ras_p60 = ras_real.quantile(0.60)
te_ras_p40 = ras_real.quantile(0.40)
for idx in te_bt[te_bt['s_ras_raw'].isna()].index:
    rd = te_bt.loc[idx, 'round']
    te_bt.loc[idx, 's_ras_raw'] = te_ras_p60 if rd <= 2 else te_ras_p40

bo_avg_te = te_bt['s_breakout_raw'].mean()
te_bt['s_breakout_raw_filled'] = te_bt['s_breakout_raw'].fillna(bo_avg_te)

print(f"  TE backtest: {len(te_bt)} players loaded")


# ============================================================================
# STEP 2: COMPUTE OLD SLAP (for ranking comparison)
# ============================================================================
wr_bt['slap_old'] = (
    WR_V5['dc'] * wr_bt['s_dc'] +
    WR_V5['breakout'] * wr_bt['s_breakout_raw'] +
    WR_V5['teammate'] * wr_bt['s_teammate'] +
    WR_V5['early_declare'] * wr_bt['s_early_declare']
).round(1)

rb_bt['slap_old'] = (
    RB_V5['dc'] * rb_bt['s_dc'] +
    RB_V5['production'] * rb_bt['s_production_raw'] +
    RB_V5['speed_score'] * rb_bt['s_speed_raw']
).round(1)

te_bt['slap_old'] = (
    TE_V5['dc'] * te_bt['s_dc'] +
    TE_V5['breakout'] * te_bt['s_breakout_raw_filled'] +
    TE_V5['production'] * te_bt['s_production_minmax_filled'] +
    TE_V5['ras'] * te_bt['s_ras_raw']
).round(1)


# ============================================================================
# STEP 3: PERCENTILE-RANK NORMALIZATION (within position, backtest only)
# ============================================================================
print(f"\n\n{'='*120}")
print("STEP 2: Percentile-rank normalization within position")
print("=" * 120)

# --- WR: Percentile-rank the enhanced breakout ---
# Binary components (teammate, early_declare) stay as-is
wr_bt['s_breakout_pctl'] = percentile_rank(wr_bt['s_breakout_raw'])
# Store the backtest raw values for scoring 2026 prospects
wr_bt_breakout_ref = wr_bt['s_breakout_raw'].copy()

print(f"\n  WR Enhanced Breakout:")
print(f"    OLD scale: mean={wr_bt['s_breakout_raw'].mean():.1f}, median={wr_bt['s_breakout_raw'].median():.1f}")
print(f"    NEW pctl:  mean={wr_bt['s_breakout_pctl'].mean():.1f}, median={wr_bt['s_breakout_pctl'].median():.1f}")

# --- RB: Percentile-rank production AND speed score ---
rb_bt['s_production_pctl'] = percentile_rank(rb_bt['s_production_raw'])
rb_bt['s_speed_pctl'] = percentile_rank(rb_bt['s_speed_raw'])
rb_bt_prod_ref = rb_bt['s_production_raw'].copy()
rb_bt_speed_ref = rb_bt['s_speed_raw'].copy()

print(f"\n  RB Production (RYPTPA):")
print(f"    OLD scale: mean={rb_bt['s_production_raw'].mean():.1f}, median={rb_bt['s_production_raw'].median():.1f}")
print(f"    NEW pctl:  mean={rb_bt['s_production_pctl'].mean():.1f}, median={rb_bt['s_production_pctl'].median():.1f}")
print(f"\n  RB Speed Score:")
print(f"    OLD scale: mean={rb_bt['s_speed_raw'].mean():.1f}, median={rb_bt['s_speed_raw'].median():.1f}")
print(f"    NEW pctl:  mean={rb_bt['s_speed_pctl'].mean():.1f}, median={rb_bt['s_speed_pctl'].median():.1f}")

# --- TE: Percentile-rank breakout, production, and RAS ---
te_bt['s_breakout_pctl'] = percentile_rank(te_bt['s_breakout_raw_filled'])
te_bt['s_production_pctl'] = percentile_rank(te_bt['s_production_minmax_filled'])
te_bt['s_ras_pctl'] = percentile_rank(te_bt['s_ras_raw'])
te_bt_bo_ref = te_bt['s_breakout_raw_filled'].copy()
te_bt_prod_ref = te_bt['s_production_minmax_filled'].copy()
te_bt_ras_ref = te_bt['s_ras_raw'].copy()

print(f"\n  TE Breakout:")
print(f"    OLD scale: mean={te_bt['s_breakout_raw_filled'].mean():.1f}, median={te_bt['s_breakout_raw_filled'].median():.1f}")
print(f"    NEW pctl:  mean={te_bt['s_breakout_pctl'].mean():.1f}, median={te_bt['s_breakout_pctl'].median():.1f}")
print(f"\n  TE Production:")
print(f"    OLD scale: mean={te_bt['s_production_minmax_filled'].mean():.1f}, median={te_bt['s_production_minmax_filled'].median():.1f}")
print(f"    NEW pctl:  mean={te_bt['s_production_pctl'].mean():.1f}, median={te_bt['s_production_pctl'].median():.1f}")
print(f"\n  TE RAS:")
print(f"    OLD scale: mean={te_bt['s_ras_raw'].mean():.1f}, median={te_bt['s_ras_raw'].median():.1f}")
print(f"    NEW pctl:  mean={te_bt['s_ras_pctl'].mean():.1f}, median={te_bt['s_ras_pctl'].median():.1f}")


# ============================================================================
# STEP 4: NEW SLAP SCORES
# ============================================================================
print(f"\n\n{'='*120}")
print("STEP 3: Recalculate SLAP with percentile-ranked components (same weights)")
print("=" * 120)

wr_bt['slap_new'] = (
    WR_V5['dc'] * wr_bt['s_dc'] +
    WR_V5['breakout'] * wr_bt['s_breakout_pctl'] +
    WR_V5['teammate'] * wr_bt['s_teammate'] +
    WR_V5['early_declare'] * wr_bt['s_early_declare']
).round(1)

rb_bt['slap_new'] = (
    RB_V5['dc'] * rb_bt['s_dc'] +
    RB_V5['production'] * rb_bt['s_production_pctl'] +
    RB_V5['speed_score'] * rb_bt['s_speed_pctl']
).round(1)

te_bt['slap_new'] = (
    TE_V5['dc'] * te_bt['s_dc'] +
    TE_V5['breakout'] * te_bt['s_breakout_pctl'] +
    TE_V5['production'] * te_bt['s_production_pctl'] +
    TE_V5['ras'] * te_bt['s_ras_pctl']
).round(1)

wr_bt['delta_new'] = (wr_bt['slap_new'] - wr_bt['s_dc']).round(1)
rb_bt['delta_new'] = (rb_bt['slap_new'] - rb_bt['s_dc']).round(1)
te_bt['delta_new'] = (te_bt['slap_new'] - te_bt['s_dc']).round(1)


# ============================================================================
# DIAGNOSTIC A: Average SLAP by draft round (the key table)
# ============================================================================
print(f"\n\n{'='*120}")
print("DIAGNOSTIC A: Average SLAP by draft round × position (OLD vs NEW)")
print("=" * 120)

all_bt = pd.concat([
    wr_bt[['player_name', 'pick', 'round', 'draft_year', 'slap_old', 'slap_new', 's_dc', 'hit24']].assign(position='WR'),
    rb_bt[['player_name', 'pick', 'round', 'draft_year', 'slap_old', 'slap_new', 's_dc', 'hit24']].assign(position='RB'),
    te_bt[['player_name', 'pick', 'round', 'draft_year', 'slap_old', 'slap_new', 's_dc']].assign(position='TE', hit24=te_bt['top12_10g']),
])

print(f"\n  OLD SCORES (current):")
print(f"  {'Round':>5} | {'WR':>7} {'(n)':>4} | {'RB':>7} {'(n)':>4} | {'TE':>7} {'(n)':>4} | {'WR-RB':>6} | {'WR-TE':>6}")
print(f"  {'-'*65}")
for rd in range(1, 8):
    vals = {}
    for pos in ['WR', 'RB', 'TE']:
        sub = all_bt[(all_bt['position'] == pos) & (all_bt['round'] == rd)]
        vals[pos] = (sub['slap_old'].mean(), len(sub))
    wr_a, wr_n = vals['WR']; rb_a, rb_n = vals['RB']; te_a, te_n = vals['TE']
    print(f"  {rd:>5} | {wr_a:>7.1f} {wr_n:>3}  | {rb_a:>7.1f} {rb_n:>3}  | {te_a:>7.1f} {te_n:>3}  | {wr_a-rb_a:>+6.1f} | {wr_a-te_a:>+6.1f}")

print(f"\n  NEW SCORES (percentile-normalized):")
print(f"  {'Round':>5} | {'WR':>7} {'(n)':>4} | {'RB':>7} {'(n)':>4} | {'TE':>7} {'(n)':>4} | {'WR-RB':>6} | {'WR-TE':>6}")
print(f"  {'-'*65}")
for rd in range(1, 8):
    vals = {}
    for pos in ['WR', 'RB', 'TE']:
        sub = all_bt[(all_bt['position'] == pos) & (all_bt['round'] == rd)]
        vals[pos] = (sub['slap_new'].mean(), len(sub))
    wr_a, wr_n = vals['WR']; rb_a, rb_n = vals['RB']; te_a, te_n = vals['TE']
    print(f"  {rd:>5} | {wr_a:>7.1f} {wr_n:>3}  | {rb_a:>7.1f} {rb_n:>3}  | {te_a:>7.1f} {te_n:>3}  | {wr_a-rb_a:>+6.1f} | {wr_a-te_a:>+6.1f}")

print(f"\n  IMPROVEMENT IN WR-RB GAP BY ROUND:")
print(f"  {'Round':>5} | {'Old Gap':>8} | {'New Gap':>8} | {'Change':>8}")
print(f"  {'-'*40}")
for rd in range(1, 8):
    wr_old = all_bt[(all_bt['position'] == 'WR') & (all_bt['round'] == rd)]['slap_old'].mean()
    rb_old = all_bt[(all_bt['position'] == 'RB') & (all_bt['round'] == rd)]['slap_old'].mean()
    wr_new = all_bt[(all_bt['position'] == 'WR') & (all_bt['round'] == rd)]['slap_new'].mean()
    rb_new = all_bt[(all_bt['position'] == 'RB') & (all_bt['round'] == rd)]['slap_new'].mean()
    old_gap = wr_old - rb_old
    new_gap = wr_new - rb_new
    print(f"  {rd:>5} | {old_gap:>+8.1f} | {new_gap:>+8.1f} | {new_gap - old_gap:>+8.1f}")


# ============================================================================
# DIAGNOSTIC B: The Melvin Gordon vs Corey Coleman comparison
# ============================================================================
print(f"\n\n{'='*120}")
print("DIAGNOSTIC B: Melvin Gordon (RB pick 15) vs nearby WRs — OLD vs NEW")
print("=" * 120)

gordon = rb_bt[rb_bt['player_name'] == 'Melvin Gordon'].iloc[0]
print(f"\n  Melvin Gordon (RB, pick {int(gordon['pick'])}, 2015):")
print(f"    DC = {gordon['s_dc']:.1f}")
print(f"    Production: raw={gordon['s_production_raw']:.1f} → pctl={gordon['s_production_pctl']:.1f}")
print(f"    Speed:      raw={gordon['s_speed_raw']:.1f} → pctl={gordon['s_speed_pctl']:.1f}")
print(f"    OLD SLAP = 0.65×{gordon['s_dc']:.1f} + 0.30×{gordon['s_production_raw']:.1f} + 0.05×{gordon['s_speed_raw']:.1f} = {gordon['slap_old']:.1f}")
print(f"    NEW SLAP = 0.65×{gordon['s_dc']:.1f} + 0.30×{gordon['s_production_pctl']:.1f} + 0.05×{gordon['s_speed_pctl']:.1f} = {gordon['slap_new']:.1f}")

nearby_wr = wr_bt[(wr_bt['pick'] >= 13) & (wr_bt['pick'] <= 17)].sort_values('pick')
print(f"\n  WRs drafted picks 13-17:")
for _, r in nearby_wr.iterrows():
    print(f"    {r['player_name']:<25} pick {int(r['pick'])} | BO raw={r['s_breakout_raw']:.1f}→pctl={r['s_breakout_pctl']:.1f} | OLD={r['slap_old']:.1f} → NEW={r['slap_new']:.1f}")


# ============================================================================
# DIAGNOSTIC C: Top 10 at each position
# ============================================================================
print(f"\n\n{'='*120}")
print("DIAGNOSTIC C: Top 10 backtest players at each position (NEW scores)")
print("=" * 120)

for pos, df in [('WR', wr_bt), ('RB', rb_bt), ('TE', te_bt)]:
    top = df.nlargest(10, 'slap_new')
    hit_col = 'hit24' if pos != 'TE' else 'top12_10g'
    print(f"\n  ── {pos} TOP 10 ──")
    print(f"  {'Player':<25} {'Year':>4} {'Pick':>4} {'Old':>6} {'New':>6} {'Chg':>5} {'DC':>5} {'Hit':>4}")
    print(f"  {'-'*65}")
    for _, r in top.iterrows():
        h = int(r[hit_col]) if pd.notna(r.get(hit_col)) else '?'
        print(f"  {r['player_name']:<25} {int(r['draft_year']):>4} {int(r['pick']):>4} {r['slap_old']:>6.1f} {r['slap_new']:>6.1f} {r['slap_new']-r['slap_old']:>+5.1f} {r['s_dc']:>5.1f} {h:>4}")


# ============================================================================
# DIAGNOSTIC D: Ranking preservation check
# ============================================================================
print(f"\n\n{'='*120}")
print("DIAGNOSTIC D: Ranking preservation (within-position ranks should NOT change)")
print("=" * 120)

for pos, df in [('WR', wr_bt), ('RB', rb_bt), ('TE', te_bt)]:
    old_rank = df['slap_old'].rank(ascending=False, method='min')
    new_rank = df['slap_new'].rank(ascending=False, method='min')

    # Spearman correlation between old and new ranks
    corr = old_rank.corr(new_rank, method='spearman')

    # Count how many moved rank
    rank_diff = (old_rank - new_rank).abs()
    moved_0 = (rank_diff == 0).sum()
    moved_1_3 = ((rank_diff >= 1) & (rank_diff <= 3)).sum()
    moved_4_plus = (rank_diff >= 4).sum()
    max_move = rank_diff.max()

    print(f"\n  {pos} (n={len(df)}):")
    print(f"    Spearman rank correlation: {corr:.4f}")
    print(f"    Same rank: {moved_0}  |  Moved 1-3: {moved_1_3}  |  Moved 4+: {moved_4_plus}  |  Max move: {max_move:.0f}")

    # Show biggest rank changers
    df_tmp = df[['player_name', 'draft_year', 'pick', 'slap_old', 'slap_new']].copy()
    df_tmp['old_rank'] = old_rank.values
    df_tmp['new_rank'] = new_rank.values
    df_tmp['rank_change'] = df_tmp['old_rank'] - df_tmp['new_rank']
    biggest = df_tmp.reindex(df_tmp['rank_change'].abs().nlargest(5).index)
    if moved_4_plus > 0:
        print(f"    Biggest rank changes:")
        for _, r in biggest.iterrows():
            direction = "UP" if r['rank_change'] > 0 else "DOWN"
            print(f"      {r['player_name']:<25} pick {int(r['pick']):>3} | rank {int(r['old_rank'])}→{int(r['new_rank'])} ({direction} {abs(r['rank_change']):.0f})")


# ============================================================================
# DIAGNOSTIC E: Picks 1-10 comparison
# ============================================================================
print(f"\n\n{'='*120}")
print("DIAGNOSTIC E: Picks 1-10 average SLAP — OLD vs NEW")
print("=" * 120)

for pos, df in [('WR', wr_bt), ('RB', rb_bt), ('TE', te_bt)]:
    top10 = df[df['pick'] <= 10]
    if len(top10) == 0:
        continue
    old_avg = top10['slap_old'].mean()
    new_avg = top10['slap_new'].mean()
    dc_avg = top10['s_dc'].mean()
    print(f"  {pos} picks 1-10 (n={len(top10)}): DC={dc_avg:.1f} | OLD SLAP={old_avg:.1f} (delta {old_avg-dc_avg:+.1f}) | NEW SLAP={new_avg:.1f} (delta {new_avg-dc_avg:+.1f})")


# ============================================================================
# DIAGNOSTIC F: Component distribution summary (new vs old)
# ============================================================================
print(f"\n\n{'='*120}")
print("DIAGNOSTIC F: Component distributions after percentile normalization")
print("=" * 120)

print(f"\n  {'Component':<25} {'Old Mean':>9} {'New Mean':>9} {'Old Med':>8} {'New Med':>8} {'New Min':>8} {'New Max':>8}")
print(f"  {'-'*80}")
for label, old, new in [
    ('WR breakout', wr_bt['s_breakout_raw'], wr_bt['s_breakout_pctl']),
    ('RB production', rb_bt['s_production_raw'], rb_bt['s_production_pctl']),
    ('RB speed', rb_bt['s_speed_raw'], rb_bt['s_speed_pctl']),
    ('TE breakout', te_bt['s_breakout_raw_filled'], te_bt['s_breakout_pctl']),
    ('TE production', te_bt['s_production_minmax_filled'], te_bt['s_production_pctl']),
    ('TE RAS', te_bt['s_ras_raw'], te_bt['s_ras_pctl']),
]:
    print(f"  {label:<25} {old.mean():>9.1f} {new.mean():>9.1f} {old.median():>8.1f} {new.median():>8.1f} {new.min():>8.1f} {new.max():>8.1f}")


# ============================================================================
# DIAGNOSTIC G: 2026 PROSPECT SCORES (against backtest distribution)
# ============================================================================
print(f"\n\n{'='*120}")
print("DIAGNOSTIC G: 2026 Prospect Scores (scored against backtest distribution)")
print("=" * 120)

# --- WR 2026 ---
wr26 = pd.read_csv('output/slap_v5_wr_2026.csv')
wr26_bo = pd.read_csv('data/wr_breakout_ages_2026.csv')
wr26 = wr26.merge(
    wr26_bo[['player_name', 'breakout_age', 'peak_dominator']].rename(
        columns={'breakout_age': 'bo_age_src', 'peak_dominator': 'pd_src'}),
    on='player_name', how='left')
wr26['breakout_age'] = wr26['breakout_age'].fillna(wr26['bo_age_src'])
wr26['peak_dominator'] = wr26['peak_dominator'].fillna(wr26['pd_src'])

prospects = pd.read_csv('data/prospects_final.csv')
wr_prosp = prospects[prospects['position'] == 'WR'].copy()
wr26 = wr26.merge(wr_prosp[['player_name', 'age', 'weight', 'age_estimated']], on='player_name', how='left')

# Score WR 2026 breakout against backtest
wr26['s_breakout_pctl'] = wr26['enhanced_breakout'].apply(
    lambda v: score_prospect_against_backtest(v, wr_bt_breakout_ref))

wr26['s_dc'] = wr26['projected_pick'].apply(dc_score)
wr26['slap_new'] = (
    WR_V5['dc'] * wr26['s_dc'] +
    WR_V5['breakout'] * wr26['s_breakout_pctl'] +
    WR_V5['teammate'] * wr26['teammate_score'] +
    WR_V5['early_declare'] * wr26['early_declare']
).round(1)

print(f"\n  WR 2026 Top 10:")
wr26_top = wr26.nlargest(10, 'slap_new')
print(f"  {'Player':<25} {'Pick':>4} {'OldSLAP':>7} {'NewSLAP':>7} {'DC':>5} {'BO_raw':>6} {'BO_pctl':>7}")
print(f"  {'-'*65}")
for _, r in wr26_top.iterrows():
    print(f"  {r['player_name']:<25} {int(r['projected_pick']):>4} {r['slap_v5']:>7.1f} {r['slap_new']:>7.1f} {r['s_dc']:>5.1f} {r['enhanced_breakout']:>6.1f} {r['s_breakout_pctl']:>7.1f}")

# --- RB 2026 ---
rb_prosp = prospects[prospects['position'] == 'RB'].copy()
rb_prosp['s_dc'] = rb_prosp['projected_pick'].apply(dc_score)
rb_prosp['s_production_raw'] = rb_prosp.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_attempts'], r['age']), axis=1)
rb_prosp['s_production_raw'] = rb_prosp['s_production_raw'].fillna(0)

rb_prosp['s_production_pctl'] = rb_prosp['s_production_raw'].apply(
    lambda v: score_prospect_against_backtest(v, rb_bt_prod_ref))

# Speed score for 2026 RBs: MNAR imputed → score against backtest
ss_norm = rb_bt['s_speed_raw']
ss_p60_norm = ss_norm.quantile(0.60)
ss_p40_norm = ss_norm.quantile(0.40)
rb_prosp['s_speed_raw'] = rb_prosp['projected_pick'].apply(
    lambda p: ss_p60_norm if p <= 64 else ss_p40_norm)
rb_prosp['s_speed_pctl'] = rb_prosp['s_speed_raw'].apply(
    lambda v: score_prospect_against_backtest(v, rb_bt_speed_ref))

rb_prosp['slap_new'] = (
    RB_V5['dc'] * rb_prosp['s_dc'] +
    RB_V5['production'] * rb_prosp['s_production_pctl'] +
    RB_V5['speed_score'] * rb_prosp['s_speed_pctl']
).round(1)

# Also compute old scores for comparison
rb_prosp['slap_old'] = (
    RB_V5['dc'] * rb_prosp['s_dc'] +
    RB_V5['production'] * rb_prosp['s_production_raw'] +
    RB_V5['speed_score'] * rb_prosp['s_speed_raw']
).round(1)

print(f"\n  RB 2026 Top 10:")
rb26_top = rb_prosp.nlargest(10, 'slap_new')
print(f"  {'Player':<25} {'Pick':>4} {'OldSLAP':>7} {'NewSLAP':>7} {'DC':>5} {'Prod_raw':>8} {'Prod_pctl':>9}")
print(f"  {'-'*70}")
for _, r in rb26_top.iterrows():
    print(f"  {r['player_name']:<25} {int(r['projected_pick']):>4} {r['slap_old']:>7.1f} {r['slap_new']:>7.1f} {r['s_dc']:>5.1f} {r['s_production_raw']:>8.1f} {r['s_production_pctl']:>9.1f}")

# --- TE 2026 ---
te26 = pd.read_csv('data/te_2026_prospects_final.csv')
te26['s_dc'] = te26['projected_pick'].apply(dc_score)

te26['s_breakout_pctl'] = te26['breakout_score_filled'].apply(
    lambda v: score_prospect_against_backtest(v, te_bt_bo_ref))
te26['s_production_pctl'] = te26['production_score_filled'].apply(
    lambda v: score_prospect_against_backtest(v, te_bt_prod_ref))
te26['s_ras_pctl'] = te26['ras_score'].apply(
    lambda v: score_prospect_against_backtest(v, te_bt_ras_ref))

te26['slap_new'] = (
    TE_V5['dc'] * te26['s_dc'] +
    TE_V5['breakout'] * te26['s_breakout_pctl'] +
    TE_V5['production'] * te26['s_production_pctl'] +
    TE_V5['ras'] * te26['s_ras_pctl']
).round(1)

print(f"\n  TE 2026 Top 10:")
te26_top = te26.nlargest(10, 'slap_new')
print(f"  {'Player':<25} {'Pick':>4} {'OldSLAP':>7} {'NewSLAP':>7} {'DC':>5} {'BO_pctl':>7} {'Prod_pctl':>9} {'RAS_pctl':>8}")
print(f"  {'-'*75}")
for _, r in te26_top.iterrows():
    print(f"  {r['player_name']:<25} {int(r['projected_pick']):>4} {r['slap_score']:>7.1f} {r['slap_new']:>7.1f} {r['s_dc']:>5.1f} {r['s_breakout_pctl']:>7.1f} {r['s_production_pctl']:>9.1f} {r['s_ras_pctl']:>8.1f}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n\n{'='*120}")
print("FINAL SUMMARY: Cross-Position Calibration Before vs After")
print("=" * 120)

print(f"\n  Round 1 (picks 1-32):")
print(f"  {'Position':>8} | {'Old SLAP':>9} | {'New SLAP':>9} | {'Old WR-gap':>10} | {'New WR-gap':>10}")
print(f"  {'-'*55}")
wr_r1_old = all_bt[(all_bt['position']=='WR') & (all_bt['round']==1)]['slap_old'].mean()
wr_r1_new = all_bt[(all_bt['position']=='WR') & (all_bt['round']==1)]['slap_new'].mean()
for pos in ['WR', 'RB', 'TE']:
    r1_old = all_bt[(all_bt['position']==pos) & (all_bt['round']==1)]['slap_old'].mean()
    r1_new = all_bt[(all_bt['position']==pos) & (all_bt['round']==1)]['slap_new'].mean()
    old_gap = wr_r1_old - r1_old
    new_gap = wr_r1_new - r1_new
    print(f"  {pos:>8} | {r1_old:>9.1f} | {r1_new:>9.1f} | {old_gap:>+10.1f} | {new_gap:>+10.1f}")

print(f"\n  Overall averages:")
for pos in ['WR', 'RB', 'TE']:
    sub_old = all_bt[all_bt['position']==pos]['slap_old']
    sub_new = all_bt[all_bt['position']==pos]['slap_new']
    print(f"    {pos}: old={sub_old.mean():.1f} → new={sub_new.mean():.1f} ({sub_new.mean()-sub_old.mean():+.1f})")

print(f"\n\n  NOTE: Nothing saved. This is diagnostic only.")
print(f"  Review results above and confirm before implementing.")
