"""
SLAP V5 — Unified Master Database Builder
==========================================
Combines ALL three positions (WR, RB, TE) × (backtest + 2026 prospects)
into a single master CSV with consistent columns.

PERCENTILE NORMALIZATION: All non-DC components are converted to within-position
percentile ranks (0-100) using the backtest distribution as reference. 2026
prospects are scored against that backtest distribution. Binary components
(Teammate, Early Declare) are left as 0/100 — already position-neutral by design.

PROBABILITY-CALIBRATED SCORING: After computing within-position raw SLAP scores,
a logistic regression is fit per position: P(hit) = f(SLAP_raw). The final SLAP
score = predicted probability × 100, so a SLAP of 80 ≈ 80% chance of hitting.
This means cross-position comparisons are meaningful: an RB with SLAP 80 and a WR
with SLAP 80 have roughly similar hit chances. RBs hit more often than WRs at the
same draft slot, so pick-15 RBs will score HIGHER than pick-15 WRs.

Hit definitions:
  WR/RB: hit24 (top-24 fantasy season)
  TE: top12_10g (top-12 TE season with 10+ game minimum)

LOCKED MODELS:
  WR V5: 70% DC / 20% Enhanced Breakout / 5% Teammate / 5% Early Declare
  RB V5: 65% DC / 30% RYPTPA / 5% Speed Score (MNAR imputation)
  TE V5: 60% DC / 15% Breakout / 15% Production / 10% RAS (MNAR imputation)

Outputs:
  output/slap_v5_master_database.csv  — Full master (all positions, all years)
  output/slap_v5_wr.csv              — WR only (backtest + 2026)
  output/slap_v5_rb.csv              — RB only (backtest + 2026)
  output/slap_v5_te.csv              — TE only (backtest + 2026)
  output/slap_v5_2026_all.csv        — 2026 prospects only (all positions)
"""

import pandas as pd
import numpy as np
import warnings, os
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

print("=" * 120)
print("SLAP V5 — UNIFIED MASTER DATABASE BUILDER (probability-calibrated scoring)")
print("=" * 120)

# ============================================================================
# SHARED HELPERS
# ============================================================================

def dc_score(pick):
    """DC = 100 - 2.40 × (pick^0.62 - 1)  (gentler curve)"""
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

def percentile_rank(series):
    """Convert a series to percentile ranks (0-100). Ties get average rank.
    NaN values stay NaN."""
    valid = series.dropna()
    if len(valid) == 0:
        return series.copy()
    ranks = sp_stats.rankdata(valid.values, method='average')
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
    below = np.sum(bt < prospect_val)
    tied = np.sum(bt == prospect_val)
    pctile = (below + 0.5 * tied) / len(bt) * 100
    return min(99.9, max(0.1, pctile))


# ============================================================================
# WR HELPERS
# ============================================================================

def wr_breakout_score(breakout_age, dominator_pct):
    """Continuous breakout scoring: age tier + dominator tiebreaker (20% threshold)."""
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
    """Enhanced breakout = breakout_score + rush_bonus (+5 if 20+ college rush yards)."""
    base = wr_breakout_score(breakout_age, dominator_pct)
    rush_bonus = 5 if pd.notna(rush_yards) and rush_yards >= 20 else 0
    return min(base + rush_bonus, 99.9)


# ============================================================================
# RB HELPERS
# ============================================================================

def rb_production_score(rec_yards, team_pass_att, age):
    """RYPTPA with age weighting, scaled by 1.75."""
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


# ============================================================================
# TE HELPERS
# ============================================================================

def te_breakout_score(breakout_age, peak_dominator, threshold=15):
    """TE breakout scoring: 15% dominator threshold (TE-specific)."""
    if pd.isna(breakout_age) or breakout_age is None:
        if peak_dominator is not None and pd.notna(peak_dominator):
            return min(35, 15 + peak_dominator)
        return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base = age_tiers.get(int(breakout_age), 20)
    bonus = min((peak_dominator - threshold) * 0.5, 9.9) if pd.notna(peak_dominator) and peak_dominator >= threshold else 0
    return min(base + bonus, 99.9)

def te_production_score_fn(rec_yards, team_pass_att, draft_age, draft_year, season_year=None):
    """TE production: rec_yards / team_pass_att × age_weight × 100."""
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


# ============================================================================
# LOCKED V5 WEIGHTS
# ============================================================================
WR_V5 = {'dc': 0.70, 'breakout': 0.20, 'teammate': 0.05, 'early_declare': 0.05}
RB_V5 = {'dc': 0.65, 'production': 0.30, 'speed_score': 0.05}
TE_V5 = {'dc': 0.60, 'breakout': 0.15, 'production': 0.15, 'ras': 0.10}

print(f"\n  LOCKED WEIGHTS:")
print(f"    WR V5: {int(WR_V5['dc']*100)}/{int(WR_V5['breakout']*100)}/{int(WR_V5['teammate']*100)}/{int(WR_V5['early_declare']*100)} (DC/EnhBreakout/Teammate/EarlyDeclare)")
print(f"    RB V5: {int(RB_V5['dc']*100)}/{int(RB_V5['production']*100)}/{int(RB_V5['speed_score']*100)} (DC/RYPTPA/SpeedScore)")
print(f"    TE V5: {int(TE_V5['dc']*100)}/{int(TE_V5['breakout']*100)}/{int(TE_V5['production']*100)}/{int(TE_V5['ras']*100)} (DC/Breakout/Production/RAS)")
print(f"\n  PERCENTILE NORMALIZATION: ON (non-DC components → within-position percentile ranks)")
print(f"  PROBABILITY CALIBRATION: ON (logistic regression per position → SLAP = P(hit) × 100)")


# ============================================================================
# PART 1: WR BACKTEST (2015-2025)
# ============================================================================
print(f"\n\n{'='*120}")
print("PART 1: WR BACKTEST")
print("=" * 120)

wr_bt = pd.read_csv('data/wr_backtest_all_components.csv')
wr_tm = pd.read_csv('data/wr_teammate_scores.csv')
wr_bt = wr_bt.merge(wr_tm[['player_name', 'draft_year', 'total_teammate_dc']], on=['player_name', 'draft_year'], how='left')

# Merge outcomes
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
wr_out = outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
wr_bt = wr_bt.merge(wr_out, on=['player_name', 'draft_year', 'pick'], how='left')

# Calculate RAW components
wr_bt['s_dc'] = wr_bt['pick'].apply(dc_score)
wr_bt['s_breakout_raw'] = wr_bt.apply(
    lambda r: wr_enhanced_breakout(r['breakout_age'], r['peak_dominator'], r['rush_yards']), axis=1)
wr_bt['s_teammate'] = wr_bt['total_teammate_dc'].apply(lambda x: 100 if pd.notna(x) and x > 150 else 0)
wr_bt['s_early_declare'] = wr_bt['early_declare'].apply(lambda x: 100 if x == 1 else 0)

# PERCENTILE NORMALIZATION: breakout → percentile rank within WR backtest
# Binary components (teammate, early_declare) stay as 0/100
wr_bt['s_breakout_pctl'] = percentile_rank(wr_bt['s_breakout_raw'])
wr_bt_breakout_ref = wr_bt['s_breakout_raw'].copy()  # Save for scoring 2026 prospects

# V5 score (using percentile-normalized breakout)
wr_bt['slap_v5_raw'] = (
    WR_V5['dc'] * wr_bt['s_dc'] +
    WR_V5['breakout'] * wr_bt['s_breakout_pctl'] +
    WR_V5['teammate'] * wr_bt['s_teammate'] +
    WR_V5['early_declare'] * wr_bt['s_early_declare']
)

# Data quality flags
wr_bt['breakout_data'] = np.where(wr_bt['breakout_age'].notna(), 'real', 'imputed')

print(f"  WR backtest: {len(wr_bt)} players, draft years {wr_bt['draft_year'].min()}-{wr_bt['draft_year'].max()}")
print(f"  Breakout: raw mean={wr_bt['s_breakout_raw'].mean():.1f} → pctl mean={wr_bt['s_breakout_pctl'].mean():.1f}")


# ============================================================================
# PART 2: RB BACKTEST (2015-2025) — with Speed Score MNAR imputation
# ============================================================================
print(f"\n{'='*120}")
print("PART 2: RB BACKTEST (with Speed Score MNAR imputation)")
print("=" * 120)

rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
rb_out = outcomes[outcomes['position'] == 'RB'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
rb_bt = rb_bt.merge(rb_out, on=['player_name', 'draft_year', 'pick'], how='left')

# DC and Production (RAW)
rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)
rb_bt['s_production_raw'] = rb_bt.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb_bt['s_production_raw_filled'] = rb_bt['s_production_raw'].fillna(0)

# Speed Score — weight recovery from combine.parquet
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

# 40-time estimation from weight × round buckets
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
        sub = known[(known['wb'] == wb) & (known['rb_bkt'] == rdb)]
        if len(sub) > 0:
            lookup_40[(wb, rdb)] = sub['forty'].mean()
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

# MNAR-aware imputation for raw speed score
real_ss = rb_bt['raw_ss'].dropna()
p60 = real_ss.quantile(0.60)
p40 = real_ss.quantile(0.40)
for idx in rb_bt[rb_bt['raw_ss'].isna()].index:
    rd = rb_bt.loc[idx, 'round']
    rb_bt.loc[idx, 'raw_ss'] = p60 if rd <= 2 else p40

# Normalize raw speed score to 0-100 (used as the "raw" scale before percentile)
rb_bt['s_speed_raw'] = normalize_0_100(rb_bt['raw_ss'])

# PERCENTILE NORMALIZATION: production and speed → percentile ranks within RB backtest
rb_bt['s_production_pctl'] = percentile_rank(rb_bt['s_production_raw_filled'])
rb_bt['s_speed_pctl'] = percentile_rank(rb_bt['s_speed_raw'])
rb_bt_prod_ref = rb_bt['s_production_raw_filled'].copy()  # Save for scoring 2026 prospects
rb_bt_speed_ref = rb_bt['s_speed_raw'].copy()

# V5 Score (using percentile-normalized components)
rb_bt['slap_v5_raw'] = (
    RB_V5['dc'] * rb_bt['s_dc'] +
    RB_V5['production'] * rb_bt['s_production_pctl'] +
    RB_V5['speed_score'] * rb_bt['s_speed_pctl']
)

# Athletic data flags
rb_bt['athletic_flag'] = 'real'
rb_bt.loc[impute_mask, 'athletic_flag'] = 'estimated_40'
rb_bt.loc[rb_bt['weight'].isna(), 'athletic_flag'] = 'mnar_imputed'

n_real_ss = rb_bt['forty'].notna().sum()
n_imp40 = impute_mask.sum()
n_mnar = (rb_bt['weight'].isna()).sum()
print(f"  RB backtest: {len(rb_bt)} players, draft years {rb_bt['draft_year'].min()}-{rb_bt['draft_year'].max()}")
print(f"  Speed Score: {n_real_ss} real, {n_imp40} estimated-40, {n_mnar} MNAR-imputed")
print(f"  Production: raw mean={rb_bt['s_production_raw_filled'].mean():.1f} → pctl mean={rb_bt['s_production_pctl'].mean():.1f}")
print(f"  Speed:      raw mean={rb_bt['s_speed_raw'].mean():.1f} → pctl mean={rb_bt['s_speed_pctl'].mean():.1f}")


# ============================================================================
# PART 3: TE BACKTEST (2015-2025) — with RAS from backtest data
# ============================================================================
print(f"\n{'='*120}")
print("PART 3: TE BACKTEST")
print("=" * 120)

te_bt = pd.read_csv('data/te_backtest_master.csv')

# DC
te_bt['s_dc'] = te_bt['pick'].apply(dc_score)

# Breakout (15% dominator threshold, TE-specific) — RAW
te_bt['s_breakout_raw'] = te_bt.apply(
    lambda r: te_breakout_score(r['breakout_age'], r['peak_dominator'], threshold=15), axis=1)
bo_avg_te = te_bt['s_breakout_raw'].mean()
te_bt['s_breakout_raw_filled'] = te_bt['s_breakout_raw'].fillna(bo_avg_te)

# Production — Rec/TPA with age weight, CFBD primary, PFF fallback — RAW
te_bt['te_prod_raw'] = te_bt.apply(lambda r: te_production_score_fn(
    r['cfbd_rec_yards'], r['cfbd_team_pass_att'], r['draft_age'], r['draft_year']), axis=1)

# Manual patches for known missing (Goedert, Shaheen)
for name, vals in {
    'Dallas Goedert': {'cfbd_rec_yards': 1111, 'cfbd_team_pass_att': 455, 'draft_age': 23.0, 'draft_year': 2018},
    'Adam Shaheen': {'cfbd_rec_yards': 867, 'cfbd_team_pass_att': 328, 'draft_age': 22.3, 'draft_year': 2017},
}.items():
    mask = te_bt['player_name'] == name
    if mask.sum() > 0 and pd.isna(te_bt.loc[mask, 'te_prod_raw'].values[0]):
        te_bt.loc[mask, 'te_prod_raw'] = te_production_score_fn(
            vals['cfbd_rec_yards'], vals['cfbd_team_pass_att'], vals['draft_age'], vals['draft_year'])

# PFF fallback for remaining missing
for idx in te_bt[te_bt['te_prod_raw'].isna()].index:
    r = te_bt.loc[idx]
    if pd.notna(r.get('pff_yards')) and pd.notna(r.get('pff_receptions')):
        pass  # Leave as NaN — will be imputed

# Normalize production to 0-99.9 using min-max (this is the "raw" scale before percentile)
prod_vals = te_bt['te_prod_raw'].dropna()
te_prod_min = prod_vals.min()
te_prod_max = prod_vals.max()
te_bt['s_production_minmax'] = np.where(
    te_bt['te_prod_raw'].notna(),
    ((te_bt['te_prod_raw'] - te_prod_min) / (te_prod_max - te_prod_min) * 99.9).clip(0, 99.9),
    np.nan
)
prod_avg_te_mm = te_bt['s_production_minmax'].mean()
te_bt['s_production_raw_filled'] = te_bt['s_production_minmax'].fillna(prod_avg_te_mm)

# RAS — use actual TE RAS where available, MNAR impute rest — RAW
te_bt['s_ras_raw'] = te_bt['te_ras'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)
ras_real = te_bt['s_ras_raw'].dropna()
te_ras_p60 = ras_real.quantile(0.60)
te_ras_p40 = ras_real.quantile(0.40)
for idx in te_bt[te_bt['s_ras_raw'].isna()].index:
    rd = te_bt.loc[idx, 'round']
    te_bt.loc[idx, 's_ras_raw'] = te_ras_p60 if rd <= 2 else te_ras_p40

# PERCENTILE NORMALIZATION: breakout, production, RAS → percentile ranks within TE backtest
te_bt['s_breakout_pctl'] = percentile_rank(te_bt['s_breakout_raw_filled'])
te_bt['s_production_pctl'] = percentile_rank(te_bt['s_production_raw_filled'])
te_bt['s_ras_pctl'] = percentile_rank(te_bt['s_ras_raw'])
te_bt_bo_ref = te_bt['s_breakout_raw_filled'].copy()    # Save for scoring 2026 prospects
te_bt_prod_ref = te_bt['s_production_raw_filled'].copy()
te_bt_ras_ref = te_bt['s_ras_raw'].copy()

# V5 Score (using percentile-normalized components)
te_bt['slap_v5_raw'] = (
    TE_V5['dc'] * te_bt['s_dc'] +
    TE_V5['breakout'] * te_bt['s_breakout_pctl'] +
    TE_V5['production'] * te_bt['s_production_pctl'] +
    TE_V5['ras'] * te_bt['s_ras_pctl']
)

# Data flags
te_bt['breakout_flag'] = np.where(te_bt['breakout_age'].notna() | te_bt['peak_dominator'].notna(), 'real', 'imputed')
te_bt['production_flag'] = np.where(te_bt['te_prod_raw'].notna(), 'real', 'imputed')
te_bt['ras_flag'] = np.where(te_bt['te_ras'].notna(), 'real', 'mnar_imputed')

n_ras_real = (te_bt['te_ras'].notna()).sum()
print(f"  TE backtest: {len(te_bt)} players, draft years {te_bt['draft_year'].min()}-{te_bt['draft_year'].max()}")
n_bo = te_bt['breakout_age'].notna().sum()
n_pff = te_bt['peak_dominator'].notna().sum()
print(f"  Breakout: {n_bo}/{len(te_bt)} broke out, {n_pff}/{len(te_bt)} have PFF data, {len(te_bt)-n_pff} no data (default=25)")
print(f"  Production: {te_bt['te_prod_raw'].notna().sum()}/{len(te_bt)} real, rest imputed (avg={prod_avg_te_mm:.1f})")
print(f"  RAS: {n_ras_real}/{len(te_bt)} real, rest MNAR-imputed (p60={te_ras_p60:.1f}, p40={te_ras_p40:.1f})")
print(f"  Breakout:   raw mean={te_bt['s_breakout_raw_filled'].mean():.1f} → pctl mean={te_bt['s_breakout_pctl'].mean():.1f}")
print(f"  Production: raw mean={te_bt['s_production_raw_filled'].mean():.1f} → pctl mean={te_bt['s_production_pctl'].mean():.1f}")
print(f"  RAS:        raw mean={te_bt['s_ras_raw'].mean():.1f} → pctl mean={te_bt['s_ras_pctl'].mean():.1f}")


# ============================================================================
# PART 4: WR 2026 PROSPECTS — scored against backtest distribution
# ============================================================================
print(f"\n{'='*120}")
print("PART 4: WR 2026 PROSPECTS (scored against WR backtest distribution)")
print("=" * 120)

# Load pre-computed WR 2026 raw data
wr26 = pd.read_csv('output/slap_v5_wr_2026.csv')

# Also load breakout ages for raw data
wr26_bo = pd.read_csv('data/wr_breakout_ages_2026.csv')
wr26 = wr26.merge(
    wr26_bo[['player_name', 'breakout_age', 'peak_dominator']].rename(
        columns={'breakout_age': 'bo_age_src', 'peak_dominator': 'pd_src'}),
    on='player_name', how='left')
wr26['breakout_age'] = wr26['breakout_age'].fillna(wr26['bo_age_src'])
wr26['peak_dominator'] = wr26['peak_dominator'].fillna(wr26['pd_src'])

# Load prospect data for additional fields
prospects = pd.read_csv('data/prospects_final.csv')
wr_prospects = prospects[prospects['position'] == 'WR'].copy()
wr26 = wr26.merge(wr_prospects[['player_name', 'age', 'weight', 'age_estimated']],
                   on='player_name', how='left')

def pick_to_round(pick):
    if pd.isna(pick): return np.nan
    if pick <= 32: return 1
    elif pick <= 64: return 2
    elif pick <= 100: return 3
    elif pick <= 135: return 4
    elif pick <= 176: return 5
    elif pick <= 220: return 6
    else: return 7

# Normalize column names (handle both old and new formats)
if 'projected_pick' not in wr26.columns and 'pick' in wr26.columns:
    wr26['projected_pick'] = wr26['pick']
if 'school' not in wr26.columns and 'college' in wr26.columns:
    wr26['school'] = wr26['college']
if 'early_declare' not in wr26.columns and 'early_declare_score' in wr26.columns:
    wr26['early_declare'] = wr26['early_declare_score']

# DC
wr26['s_dc'] = wr26['projected_pick'].apply(dc_score)

# Score breakout against WR backtest distribution
wr26['s_breakout_pctl'] = wr26['enhanced_breakout'].apply(
    lambda v: score_prospect_against_backtest(v, wr_bt_breakout_ref))

# V5 score (using percentile-normalized breakout)
wr26['slap_v5_raw'] = (
    WR_V5['dc'] * wr26['s_dc'] +
    WR_V5['breakout'] * wr26['s_breakout_pctl'] +
    WR_V5['teammate'] * wr26['teammate_score'] +
    WR_V5['early_declare'] * wr26['early_declare']
)

print(f"  WR 2026 prospects: {len(wr26)} players")
print(f"  Breakout pctl range: {wr26['s_breakout_pctl'].min():.1f} - {wr26['s_breakout_pctl'].max():.1f}")


# ============================================================================
# PART 5: RB 2026 PROSPECTS — scored against backtest distribution
# ============================================================================
print(f"\n{'='*120}")
print("PART 5: RB 2026 PROSPECTS (scored against RB backtest distribution)")
print("=" * 120)

rb_prospects = prospects[prospects['position'] == 'RB'].copy()

rb_prospects['s_dc'] = rb_prospects['projected_pick'].apply(dc_score)
rb_prospects['s_production_raw'] = rb_prospects.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_attempts'], r['age']), axis=1)
rb_prospects['s_production_raw'] = rb_prospects['s_production_raw'].fillna(0)

# Score production against RB backtest distribution
rb_prospects['s_production_pctl'] = rb_prospects['s_production_raw'].apply(
    lambda v: score_prospect_against_backtest(v, rb_bt_prod_ref))

# Speed score for 2026 RBs: MNAR imputed → score against backtest
ss_p60_raw = rb_bt['s_speed_raw'].quantile(0.60)
ss_p40_raw = rb_bt['s_speed_raw'].quantile(0.40)
rb_prospects['s_speed_raw'] = rb_prospects['projected_pick'].apply(
    lambda p: ss_p60_raw if p <= 64 else ss_p40_raw)
rb_prospects['s_speed_pctl'] = rb_prospects['s_speed_raw'].apply(
    lambda v: score_prospect_against_backtest(v, rb_bt_speed_ref))

# V5 Score (using percentile-normalized components)
rb_prospects['slap_v5_raw'] = (
    RB_V5['dc'] * rb_prospects['s_dc'] +
    RB_V5['production'] * rb_prospects['s_production_pctl'] +
    RB_V5['speed_score'] * rb_prospects['s_speed_pctl']
)

print(f"  RB 2026 prospects: {len(rb_prospects)} players")
print(f"  Production pctl range: {rb_prospects['s_production_pctl'].min():.1f} - {rb_prospects['s_production_pctl'].max():.1f}")


# ============================================================================
# PART 6: TE 2026 PROSPECTS — scored against backtest distribution
# ============================================================================
print(f"\n{'='*120}")
print("PART 6: TE 2026 PROSPECTS (scored against TE backtest distribution)")
print("=" * 120)

te26 = pd.read_csv('data/te_2026_prospects_final.csv')

te26['s_dc'] = te26['projected_pick'].apply(dc_score)

# Score components against TE backtest distributions
te26['s_breakout_pctl'] = te26['breakout_score_filled'].apply(
    lambda v: score_prospect_against_backtest(v, te_bt_bo_ref))
te26['s_production_pctl'] = te26['production_score_filled'].apply(
    lambda v: score_prospect_against_backtest(v, te_bt_prod_ref))
te26['s_ras_pctl'] = te26['ras_score'].apply(
    lambda v: score_prospect_against_backtest(v, te_bt_ras_ref))

# V5 Score (using percentile-normalized components)
te26['slap_v5_raw'] = (
    TE_V5['dc'] * te26['s_dc'] +
    TE_V5['breakout'] * te26['s_breakout_pctl'] +
    TE_V5['production'] * te26['s_production_pctl'] +
    TE_V5['ras'] * te26['s_ras_pctl']
)

print(f"  TE 2026 prospects: {len(te26)} players")
print(f"  Breakout pctl range: {te26['s_breakout_pctl'].min():.1f} - {te26['s_breakout_pctl'].max():.1f}")


# ============================================================================
# PROBABILITY-CALIBRATED SCORING — logistic regression per position
# ============================================================================
print(f"\n\n{'='*120}")
print("PROBABILITY-CALIBRATED SCORING: Fitting logistic regression per position")
print("=" * 120)

# Fit logistic regression: P(hit) = f(slap_v5_raw) for each position
# WR/RB use hit24, TE uses top12_10g
logistic_models = {}

# --- WR logistic ---
wr_fit = wr_bt[wr_bt['hit24'].notna()].copy()
X_wr = wr_fit[['slap_v5_raw']].values
y_wr = wr_fit['hit24'].astype(int).values
lr_wr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_wr.fit(X_wr, y_wr)
logistic_models['WR'] = lr_wr
print(f"  WR: coef={lr_wr.coef_[0][0]:.4f}, intercept={lr_wr.intercept_[0]:.4f}, "
      f"n={len(wr_fit)}, hit_rate={y_wr.mean():.3f}")

# --- RB logistic ---
rb_fit = rb_bt[rb_bt['hit24'].notna()].copy()
X_rb = rb_fit[['slap_v5_raw']].values
y_rb = rb_fit['hit24'].astype(int).values
lr_rb = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_rb.fit(X_rb, y_rb)
logistic_models['RB'] = lr_rb
print(f"  RB: coef={lr_rb.coef_[0][0]:.4f}, intercept={lr_rb.intercept_[0]:.4f}, "
      f"n={len(rb_fit)}, hit_rate={y_rb.mean():.3f}")

# --- TE logistic ---
te_fit = te_bt[te_bt['top12_10g'].notna()].copy()
X_te = te_fit[['slap_v5_raw']].values
y_te = te_fit['top12_10g'].astype(int).values
lr_te = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_te.fit(X_te, y_te)
logistic_models['TE'] = lr_te
print(f"  TE: coef={lr_te.coef_[0][0]:.4f}, intercept={lr_te.intercept_[0]:.4f}, "
      f"n={len(te_fit)}, hit_rate={y_te.mean():.3f}")

def prob_calibrate(raw_scores, model):
    """Convert raw SLAP scores to probability-calibrated scores (0-100)."""
    X = raw_scores.values.reshape(-1, 1)
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs * 100, index=raw_scores.index)

# Apply to all backtest
wr_bt['slap_v5'] = prob_calibrate(wr_bt['slap_v5_raw'], lr_wr).round(1)
rb_bt['slap_v5'] = prob_calibrate(rb_bt['slap_v5_raw'], lr_rb).round(1)
te_bt['slap_v5'] = prob_calibrate(te_bt['slap_v5_raw'], lr_te).round(1)

# Apply to all 2026 prospects
wr26['slap_v5'] = prob_calibrate(wr26['slap_v5_raw'], lr_wr).round(1)
rb_prospects['slap_v5'] = prob_calibrate(rb_prospects['slap_v5_raw'], lr_rb).round(1)
te26['slap_v5'] = prob_calibrate(te26['slap_v5_raw'], lr_te).round(1)

# DC-only calibrated scores (for delta calculation)
# For each position, compute what a DC-only raw score would be (non-DC components at 50th pctl)
# Then calibrate that through the same logistic model
def dc_only_raw(dc_val, weights, dc_weight_key='dc'):
    """Compute what raw SLAP would be if all non-DC components were at 50th percentile."""
    non_dc_contribution = sum(w * 50 for k, w in weights.items() if k != dc_weight_key)
    return weights[dc_weight_key] * dc_val + non_dc_contribution

wr_bt['dc_raw_equiv'] = wr_bt['s_dc'].apply(lambda d: dc_only_raw(d, WR_V5))
rb_bt['dc_raw_equiv'] = rb_bt['s_dc'].apply(lambda d: dc_only_raw(d, RB_V5))
te_bt['dc_raw_equiv'] = te_bt['s_dc'].apply(lambda d: dc_only_raw(d, TE_V5))
wr26['dc_raw_equiv'] = wr26['s_dc'].apply(lambda d: dc_only_raw(d, WR_V5))
rb_prospects['dc_raw_equiv'] = rb_prospects['s_dc'].apply(lambda d: dc_only_raw(d, RB_V5))
te26['dc_raw_equiv'] = te26['s_dc'].apply(lambda d: dc_only_raw(d, TE_V5))

# Calibrate DC-only through the same logistic model
wr_bt['dc_score_calibrated'] = prob_calibrate(wr_bt['dc_raw_equiv'], lr_wr).round(1)
rb_bt['dc_score_calibrated'] = prob_calibrate(rb_bt['dc_raw_equiv'], lr_rb).round(1)
te_bt['dc_score_calibrated'] = prob_calibrate(te_bt['dc_raw_equiv'], lr_te).round(1)
wr26['dc_score_calibrated'] = prob_calibrate(wr26['dc_raw_equiv'], lr_wr).round(1)
rb_prospects['dc_score_calibrated'] = prob_calibrate(rb_prospects['dc_raw_equiv'], lr_rb).round(1)
te26['dc_score_calibrated'] = prob_calibrate(te26['dc_raw_equiv'], lr_te).round(1)

# Delta = SLAP - DC-only calibrated (how much does the model disagree with pure DC?)
wr_bt['delta_vs_dc'] = (wr_bt['slap_v5'] - wr_bt['dc_score_calibrated']).round(1)
rb_bt['delta_vs_dc'] = (rb_bt['slap_v5'] - rb_bt['dc_score_calibrated']).round(1)
te_bt['delta_vs_dc'] = (te_bt['slap_v5'] - te_bt['dc_score_calibrated']).round(1)
wr26['delta_vs_dc'] = (wr26['slap_v5'] - wr26['dc_score_calibrated']).round(1)
rb_prospects['delta_vs_dc'] = (rb_prospects['slap_v5'] - rb_prospects['dc_score_calibrated']).round(1)
te26['delta_vs_dc'] = (te26['slap_v5'] - te26['dc_score_calibrated']).round(1)

# Store DC score for output (the raw DC score on 0-100 scale, same as before)
wr_bt['dc_score_final'] = wr_bt['s_dc'].round(1)
rb_bt['dc_score_final'] = rb_bt['s_dc'].round(1)
te_bt['dc_score_final'] = te_bt['s_dc'].round(1)
wr26['dc_score_final'] = wr26['s_dc'].round(1)
rb_prospects['dc_score_final'] = rb_prospects['s_dc'].round(1)
te26['dc_score_final'] = te26['s_dc'].round(1)

# Per-position stats
for pos, df in [('WR', wr_bt), ('RB', rb_bt), ('TE', te_bt)]:
    print(f"  {pos} backtest: {df['slap_v5'].min():.1f} - {df['slap_v5'].max():.1f} (mean {df['slap_v5'].mean():.1f})")


# ============================================================================
# RANKING PRESERVATION CHECK
# ============================================================================
print(f"\n{'='*120}")
print("RANKING PRESERVATION CHECK (logistic transform is monotonic → within-position rankings perfectly preserved)")
print("=" * 120)

# Logistic regression is a monotonic transform, so within-position rankings
# are identical between slap_v5_raw and slap_v5. Verify this:
for pos, df in [('WR', wr_bt), ('RB', rb_bt), ('TE', te_bt)]:
    raw_rank = df['slap_v5_raw'].rank(ascending=False, method='min')
    cal_rank = df['slap_v5'].rank(ascending=False, method='min')
    spearman_r = raw_rank.corr(cal_rank, method='spearman')
    rank_diff = (raw_rank - cal_rank).abs()
    max_move = rank_diff.max()
    print(f"  {pos}: Spearman r={spearman_r:.4f} | Max rank change: {max_move:.0f} (should be 0 — monotonic transform)")


# ============================================================================
# BUILD OUTPUT ROWS
# ============================================================================
print(f"\n{'='*120}")
print("BUILDING OUTPUT ROWS")
print("=" * 120)

# --- WR backtest rows ---
wr_rows = pd.DataFrame({
    'player_name': wr_bt['player_name'],
    'position': 'WR',
    'college': wr_bt['college'],
    'draft_year': wr_bt['draft_year'].astype(int),
    'pick': wr_bt['pick'].astype(int),
    'round': wr_bt['round'].astype(int),
    'slap_v5': wr_bt['slap_v5'],
    'dc_score': wr_bt['dc_score_final'],
    'delta_vs_dc': wr_bt['delta_vs_dc'],
    'data_type': 'backtest',
    'enhanced_breakout': wr_bt['s_breakout_pctl'].round(1),
    'teammate_score': wr_bt['s_teammate'].astype(int),
    'early_declare_score': wr_bt['s_early_declare'].astype(int),
    'breakout_age': wr_bt['breakout_age'],
    'peak_dominator': wr_bt['peak_dominator'].round(1),
    'rush_yards': wr_bt['rush_yards'],
    'production_score': np.nan,
    'speed_score': np.nan,
    'rec_yards': np.nan,
    'team_pass_att': np.nan,
    'te_breakout_score': np.nan,
    'te_production_score': np.nan,
    'ras_score': np.nan,
    'nfl_hit24': wr_bt['hit24'],
    'nfl_hit12': wr_bt['hit12'],
    'nfl_first_3yr_ppg': wr_bt.get('first_3yr_ppg'),
    'nfl_career_ppg': wr_bt.get('career_ppg'),
    'nfl_best_ppr': wr_bt['best_ppr'],
    'nfl_best_ppg': np.nan,
    'breakout_data_flag': wr_bt['breakout_data'],
    'production_data_flag': np.nan,
    'athletic_data_flag': np.nan,
})

# --- RB backtest rows ---
rb_rows = pd.DataFrame({
    'player_name': rb_bt['player_name'],
    'position': 'RB',
    'college': rb_bt['college'],
    'draft_year': rb_bt['draft_year'].astype(int),
    'pick': rb_bt['pick'].astype(int),
    'round': rb_bt['round'].astype(int),
    'slap_v5': rb_bt['slap_v5'],
    'dc_score': rb_bt['dc_score_final'],
    'delta_vs_dc': rb_bt['delta_vs_dc'],
    'data_type': 'backtest',
    'enhanced_breakout': np.nan,
    'teammate_score': np.nan,
    'early_declare_score': np.nan,
    'breakout_age': np.nan,
    'peak_dominator': np.nan,
    'rush_yards': np.nan,
    'production_score': rb_bt['s_production_pctl'].round(1),
    'speed_score': rb_bt['s_speed_pctl'].round(1),
    'rec_yards': rb_bt['rec_yards'],
    'team_pass_att': rb_bt['team_pass_att'],
    'te_breakout_score': np.nan,
    'te_production_score': np.nan,
    'ras_score': np.nan,
    'nfl_hit24': rb_bt['hit24'],
    'nfl_hit12': rb_bt['hit12'],
    'nfl_first_3yr_ppg': rb_bt.get('first_3yr_ppg'),
    'nfl_career_ppg': rb_bt.get('career_ppg'),
    'nfl_best_ppr': rb_bt['best_ppr'],
    'nfl_best_ppg': rb_bt['best_ppg'],
    'breakout_data_flag': np.nan,
    'production_data_flag': np.where(rb_bt['s_production_raw'].notna(), 'real', 'missing'),
    'athletic_data_flag': rb_bt['athletic_flag'],
})

# --- TE backtest rows ---
te_rows = pd.DataFrame({
    'player_name': te_bt['player_name'],
    'position': 'TE',
    'college': te_bt['college'],
    'draft_year': te_bt['draft_year'].astype(int),
    'pick': te_bt['pick'].astype(int),
    'round': te_bt['round'].astype(int),
    'slap_v5': te_bt['slap_v5'],
    'dc_score': te_bt['dc_score_final'],
    'delta_vs_dc': te_bt['delta_vs_dc'],
    'data_type': 'backtest',
    'enhanced_breakout': np.nan,
    'teammate_score': np.nan,
    'early_declare_score': np.nan,
    'breakout_age': te_bt['breakout_age'],
    'peak_dominator': te_bt['peak_dominator'].round(1),
    'rush_yards': np.nan,
    'production_score': np.nan,
    'speed_score': np.nan,
    'rec_yards': te_bt['cfbd_rec_yards'],
    'team_pass_att': te_bt['cfbd_team_pass_att'],
    'te_breakout_score': te_bt['s_breakout_pctl'].round(1),
    'te_production_score': te_bt['s_production_pctl'].round(1),
    'ras_score': te_bt['s_ras_pctl'].round(1),
    'nfl_hit24': te_bt['top12_10g'],
    'nfl_hit12': te_bt['top6_10g'],
    'nfl_first_3yr_ppg': te_bt['best_3yr_ppg_10g'],
    'nfl_career_ppg': te_bt['best_career_ppg_10g'],
    'nfl_best_ppr': te_bt['best_ppr'],
    'nfl_best_ppg': te_bt['best_ppg'],
    'breakout_data_flag': te_bt['breakout_flag'],
    'production_data_flag': te_bt['production_flag'],
    'athletic_data_flag': te_bt['ras_flag'],
})

# --- WR 2026 prospect rows ---
wr26_rows = pd.DataFrame({
    'player_name': wr26['player_name'],
    'position': 'WR',
    'college': wr26['school'],
    'draft_year': 2026,
    'pick': wr26['projected_pick'].astype(int),
    'round': wr26['projected_pick'].apply(pick_to_round).astype(int),
    'slap_v5': wr26['slap_v5'],
    'dc_score': wr26['dc_score_final'],
    'delta_vs_dc': wr26['delta_vs_dc'],
    'data_type': '2026_prospect',
    'enhanced_breakout': wr26['s_breakout_pctl'].round(1),
    'teammate_score': wr26['teammate_score'].astype(int),
    'early_declare_score': wr26['early_declare'].astype(int),
    'breakout_age': wr26['breakout_age'],
    'peak_dominator': wr26['peak_dominator'].round(1) if 'peak_dominator' in wr26.columns else np.nan,
    'rush_yards': wr26['rush_yards'],
    'production_score': np.nan,
    'speed_score': np.nan,
    'rec_yards': np.nan,
    'team_pass_att': np.nan,
    'te_breakout_score': np.nan,
    'te_production_score': np.nan,
    'ras_score': np.nan,
    'nfl_hit24': np.nan,
    'nfl_hit12': np.nan,
    'nfl_first_3yr_ppg': np.nan,
    'nfl_career_ppg': np.nan,
    'nfl_best_ppr': np.nan,
    'nfl_best_ppg': np.nan,
    'breakout_data_flag': np.where(wr26['breakout_age'].notna(), 'real', 'imputed'),
    'production_data_flag': np.nan,
    'athletic_data_flag': np.nan,
})

# --- RB 2026 prospect rows ---
rb26_rows = pd.DataFrame({
    'player_name': rb_prospects['player_name'],
    'position': 'RB',
    'college': rb_prospects['school'],
    'draft_year': 2026,
    'pick': rb_prospects['projected_pick'].astype(int),
    'round': rb_prospects['projected_pick'].apply(pick_to_round).astype(int),
    'slap_v5': rb_prospects['slap_v5'],
    'dc_score': rb_prospects['dc_score_final'],
    'delta_vs_dc': rb_prospects['delta_vs_dc'],
    'data_type': '2026_prospect',
    'enhanced_breakout': np.nan,
    'teammate_score': np.nan,
    'early_declare_score': np.nan,
    'breakout_age': np.nan,
    'peak_dominator': np.nan,
    'rush_yards': np.nan,
    'production_score': rb_prospects['s_production_pctl'].round(1),
    'speed_score': rb_prospects['s_speed_pctl'].round(1),
    'rec_yards': rb_prospects['rec_yards'],
    'team_pass_att': rb_prospects['team_pass_attempts'],
    'te_breakout_score': np.nan,
    'te_production_score': np.nan,
    'ras_score': np.nan,
    'nfl_hit24': np.nan,
    'nfl_hit12': np.nan,
    'nfl_first_3yr_ppg': np.nan,
    'nfl_career_ppg': np.nan,
    'nfl_best_ppr': np.nan,
    'nfl_best_ppg': np.nan,
    'breakout_data_flag': np.nan,
    'production_data_flag': np.where(rb_prospects['s_production_raw'].notna(), 'real', 'missing'),
    'athletic_data_flag': 'mnar_imputed',
})

# --- TE 2026 prospect rows ---
te26_rows = pd.DataFrame({
    'player_name': te26['player_name'],
    'position': 'TE',
    'college': te26['college'],
    'draft_year': 2026,
    'pick': te26['projected_pick'].astype(int),
    'round': te26['projected_pick'].apply(pick_to_round).astype(int),
    'slap_v5': te26['slap_v5'],
    'dc_score': te26['dc_score_final'],
    'delta_vs_dc': te26['delta_vs_dc'],
    'data_type': '2026_prospect',
    'enhanced_breakout': np.nan,
    'teammate_score': np.nan,
    'early_declare_score': np.nan,
    'breakout_age': te26['breakout_age'],
    'peak_dominator': te26['peak_dominator'],
    'rush_yards': np.nan,
    'production_score': np.nan,
    'speed_score': np.nan,
    'rec_yards': te26['cfbd_rec_yards'],
    'team_pass_att': te26['cfbd_team_pass_att'],
    'te_breakout_score': te26['s_breakout_pctl'].round(1),
    'te_production_score': te26['s_production_pctl'].round(1),
    'ras_score': te26['s_ras_pctl'].round(1),
    'nfl_hit24': np.nan,
    'nfl_hit12': np.nan,
    'nfl_first_3yr_ppg': np.nan,
    'nfl_career_ppg': np.nan,
    'nfl_best_ppr': np.nan,
    'nfl_best_ppg': np.nan,
    'breakout_data_flag': np.where(te26['breakout_age'].notna(), 'real', 'imputed'),
    'production_data_flag': np.where(te26['production_score'].notna(), 'real', 'imputed'),
    'athletic_data_flag': 'mnar_imputed',
})


# ============================================================================
# COMBINE INTO MASTER DATABASE
# ============================================================================
print(f"\n\n{'='*120}")
print("COMBINING INTO MASTER DATABASE")
print("=" * 120)

master = pd.concat([wr_rows, rb_rows, te_rows, wr26_rows, rb26_rows, te26_rows], ignore_index=True)

# Sort: position → draft_year → SLAP descending
master = master.sort_values(['position', 'draft_year', 'slap_v5'], ascending=[True, True, False])
master = master.reset_index(drop=True)

# Column order
col_order = [
    'player_name', 'position', 'college', 'draft_year', 'pick', 'round',
    'slap_v5', 'dc_score', 'delta_vs_dc', 'data_type',
    # WR components
    'enhanced_breakout', 'teammate_score', 'early_declare_score',
    # RB components
    'production_score', 'speed_score',
    # TE components
    'te_breakout_score', 'te_production_score', 'ras_score',
    # Shared raw inputs
    'breakout_age', 'peak_dominator', 'rush_yards',
    'rec_yards', 'team_pass_att',
    # NFL outcomes
    'nfl_hit24', 'nfl_hit12', 'nfl_first_3yr_ppg', 'nfl_career_ppg',
    'nfl_best_ppr', 'nfl_best_ppg',
    # Data quality
    'breakout_data_flag', 'production_data_flag', 'athletic_data_flag',
]
master = master[col_order]

# Save master
master.to_csv('output/slap_v5_master_database.csv', index=False)

# Summary
print(f"\n  MASTER DATABASE SUMMARY:")
print(f"  {'='*60}")
for pos in ['WR', 'RB', 'TE']:
    bt = master[(master['position'] == pos) & (master['data_type'] == 'backtest')]
    p26 = master[(master['position'] == pos) & (master['data_type'] == '2026_prospect')]
    print(f"  {pos}: {len(bt)} backtest + {len(p26)} prospects = {len(bt)+len(p26)} total")
total = len(master)
print(f"  {'─'*60}")
print(f"  TOTAL: {total} rows")
print(f"\n  Saved: output/slap_v5_master_database.csv")


# ============================================================================
# SAVE POSITION-SPECIFIC FILES
# ============================================================================
print(f"\n{'='*120}")
print("SAVING POSITION-SPECIFIC FILES")
print("=" * 120)

# WR file — drop TE/RB columns
wr_all = master[master['position'] == 'WR'].copy()
wr_all = wr_all.drop(columns=['production_score', 'speed_score', 'te_breakout_score',
                                'te_production_score', 'ras_score'])
wr_all.to_csv('output/slap_v5_wr.csv', index=False)
print(f"  output/slap_v5_wr.csv: {len(wr_all)} rows ({(wr_all['data_type']=='backtest').sum()} backtest + {(wr_all['data_type']=='2026_prospect').sum()} prospects)")

# RB file
rb_all = master[master['position'] == 'RB'].copy()
rb_all = rb_all.drop(columns=['enhanced_breakout', 'teammate_score', 'early_declare_score',
                                'te_breakout_score', 'te_production_score', 'ras_score',
                                'breakout_age', 'peak_dominator', 'rush_yards'])
rb_all.to_csv('output/slap_v5_rb.csv', index=False)
print(f"  output/slap_v5_rb.csv: {len(rb_all)} rows ({(rb_all['data_type']=='backtest').sum()} backtest + {(rb_all['data_type']=='2026_prospect').sum()} prospects)")

# TE file
te_all = master[master['position'] == 'TE'].copy()
te_all = te_all.drop(columns=['enhanced_breakout', 'teammate_score', 'early_declare_score',
                                'production_score', 'speed_score', 'rush_yards'])
te_all.to_csv('output/slap_v5_te.csv', index=False)
print(f"  output/slap_v5_te.csv: {len(te_all)} rows ({(te_all['data_type']=='backtest').sum()} backtest + {(te_all['data_type']=='2026_prospect').sum()} prospects)")

# 2026 prospects only (all positions)
prospects_2026 = master[master['data_type'] == '2026_prospect'].copy()
prospects_2026 = prospects_2026.sort_values('slap_v5', ascending=False).reset_index(drop=True)
prospects_2026.insert(0, 'overall_rank', range(1, len(prospects_2026) + 1))
prospects_2026.to_csv('output/slap_v5_2026_all.csv', index=False)
print(f"  output/slap_v5_2026_all.csv: {len(prospects_2026)} prospects (all positions)")


# ============================================================================
# FINAL VERIFICATION
# ============================================================================
print(f"\n\n{'='*120}")
print("FINAL VERIFICATION")
print("=" * 120)

# Check score ranges by position
for pos in ['WR', 'RB', 'TE']:
    bt = master[(master['position'] == pos) & (master['data_type'] == 'backtest')]
    p26 = master[(master['position'] == pos) & (master['data_type'] == '2026_prospect')]
    print(f"\n  {pos}:")
    print(f"    Backtest SLAP: {bt['slap_v5'].min():.1f} - {bt['slap_v5'].max():.1f} (mean {bt['slap_v5'].mean():.1f})")
    if len(p26) > 0:
        print(f"    2026 SLAP:     {p26['slap_v5'].min():.1f} - {p26['slap_v5'].max():.1f} (mean {p26['slap_v5'].mean():.1f})")

# Cross-position calibration check: average SLAP by round
print(f"\n  PROBABILITY CALIBRATION CHECK (avg SLAP by round vs actual hit rate):")
print(f"  {'Round':>5} | {'WR SLAP':>7} {'WR Hit%':>7} | {'RB SLAP':>7} {'RB Hit%':>7} | {'TE SLAP':>7} {'TE Hit%':>7}")
print(f"  {'-'*75}")
bt_all = master[master['data_type'] == 'backtest']
for rd in range(1, 8):
    vals = {}
    for pos in ['WR', 'RB', 'TE']:
        sub = bt_all[(bt_all['position'] == pos) & (bt_all['round'] == rd)]
        hit_col = 'nfl_hit24' if pos in ['WR', 'RB'] else 'nfl_hit24'
        avg_slap = sub['slap_v5'].mean() if len(sub) > 0 else float('nan')
        avg_hit = sub[hit_col].mean() * 100 if len(sub) > 0 and sub[hit_col].notna().sum() > 0 else float('nan')
        vals[pos] = (avg_slap, avg_hit)
    print(f"  {rd:>5} | {vals['WR'][0]:>7.1f} {vals['WR'][1]:>6.1f}% | {vals['RB'][0]:>7.1f} {vals['RB'][1]:>6.1f}% | {vals['TE'][0]:>7.1f} {vals['TE'][1]:>6.1f}%")

# Pick 15 comparison: WR vs RB
print(f"\n  PICK 15 COMPARISON (cross-position calibration test):")
for pos in ['WR', 'RB', 'TE']:
    p15 = bt_all[(bt_all['position'] == pos) & (bt_all['pick'].between(10, 20))]
    if len(p15) > 0:
        hit_col = 'nfl_hit24'
        avg_slap = p15['slap_v5'].mean()
        actual_hit = p15[hit_col].mean() * 100
        print(f"    {pos} picks 10-20: avg SLAP={avg_slap:.1f}, actual hit rate={actual_hit:.1f}% (n={len(p15)})")

# Within-position differentiation check
print(f"\n  WITHIN-POSITION DIFFERENTIATION (top vs bottom decile):")
for pos in ['WR', 'RB', 'TE']:
    bt = master[(master['position'] == pos) & (master['data_type'] == 'backtest')]
    top_10pct = bt.nlargest(max(1, len(bt) // 10), 'slap_v5')
    bot_10pct = bt.nsmallest(max(1, len(bt) // 10), 'slap_v5')
    spread = top_10pct['slap_v5'].mean() - bot_10pct['slap_v5'].mean()
    print(f"    {pos}: top decile avg={top_10pct['slap_v5'].mean():.1f}, bottom decile avg={bot_10pct['slap_v5'].mean():.1f}, spread={spread:.1f}")

# Show top 5 2026 prospects per position
print(f"\n\n  TOP 5 2026 PROSPECTS PER POSITION:")
for pos in ['WR', 'RB', 'TE']:
    p26 = prospects_2026[prospects_2026['position'] == pos].head(5)
    print(f"\n  ── {pos} ──")
    print(f"  {'Rk':>3} {'Player':<25} {'School':<18} {'Pick':>4} {'SLAP':>6} {'DC':>5} {'Delta':>6}")
    print(f"  {'-'*70}")
    for _, r in p26.iterrows():
        delta_str = f"+{r['delta_vs_dc']:.1f}" if r['delta_vs_dc'] >= 0 else f"{r['delta_vs_dc']:.1f}"
        print(f"  {int(r['overall_rank']):>3} {r['player_name']:<25} {r['college']:<18} {int(r['pick']):>4} "
              f"{r['slap_v5']:>6.1f} {r['dc_score']:>5.1f} {delta_str:>6}")

# TOP 60 OVERALL (the key cross-position check)
print(f"\n\n  TOP 60 BACKTEST PLAYERS (probability-calibrated — RBs should be well-represented):")
top60 = bt_all.nlargest(60, 'slap_v5')
print(f"  {'#':>3} {'Player':<25} {'Pos':>3} {'Year':>4} {'Pick':>4} {'SLAP':>6} {'DC':>5} {'Delta':>6}")
print(f"  {'-'*60}")
for i, (_, r) in enumerate(top60.iterrows(), 1):
    delta_str = f"+{r['delta_vs_dc']:.1f}" if r['delta_vs_dc'] >= 0 else f"{r['delta_vs_dc']:.1f}"
    print(f"  {i:>3} {r['player_name']:<25} {r['position']:>3} {int(r['draft_year']):>4} {int(r['pick']):>4} "
          f"{r['slap_v5']:>6.1f} {r['dc_score']:>5.1f} {delta_str:>6}")

# Position counts in top 60
pos_counts = top60['position'].value_counts()
print(f"\n  Top 60 position breakdown: {dict(pos_counts)}")

# Data quality summary
print(f"\n\n  DATA QUALITY SUMMARY:")
for pos in ['WR', 'RB', 'TE']:
    bt = master[(master['position'] == pos) & (master['data_type'] == 'backtest')]
    total_n = len(bt)
    if pos == 'WR':
        real_bo = (bt['breakout_data_flag'] == 'real').sum()
        print(f"  WR: {real_bo}/{total_n} real breakout ({total_n - real_bo} imputed)")
    elif pos == 'RB':
        real_prod = (bt['production_data_flag'] == 'real').sum()
        real_ath = (bt['athletic_data_flag'] == 'real').sum()
        print(f"  RB: {real_prod}/{total_n} real production, {real_ath}/{total_n} real speed score")
    elif pos == 'TE':
        real_bo = (bt['breakout_data_flag'] == 'real').sum()
        real_prod = (bt['production_data_flag'] == 'real').sum()
        real_ath = (bt['athletic_data_flag'] == 'real').sum()
        print(f"  TE: {real_bo}/{total_n} real breakout, {real_prod}/{total_n} real production, {real_ath}/{total_n} real RAS")

# Logistic model summary
print(f"\n\n  LOGISTIC REGRESSION MODELS:")
for pos, lr in [('WR', lr_wr), ('RB', lr_rb), ('TE', lr_te)]:
    print(f"    {pos}: P(hit) = 1 / (1 + exp(-({lr.coef_[0][0]:.4f} × SLAP_raw + ({lr.intercept_[0]:.4f}))))")

print(f"\n\n{'='*120}")
print("MASTER DATABASE BUILD COMPLETE (PROBABILITY-CALIBRATED)")
print(f"{'='*120}")
print(f"\n  Files saved:")
print(f"    output/slap_v5_master_database.csv  ({len(master)} rows)")
print(f"    output/slap_v5_wr.csv               ({len(wr_all)} rows)")
print(f"    output/slap_v5_rb.csv               ({len(rb_all)} rows)")
print(f"    output/slap_v5_te.csv               ({len(te_all)} rows)")
print(f"    output/slap_v5_2026_all.csv         ({len(prospects_2026)} rows)")
