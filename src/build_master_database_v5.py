"""
SLAP V5 — Unified Master Database Builder
==========================================
Combines ALL three positions (WR, RB, TE) × (backtest + 2026 prospects)
into a single master CSV with consistent columns.

TWO-LAYER SCORING SYSTEM:
  Layer 1 — slap_raw: Weighted composite using each component's native 0-100 scale.
    NO percentile normalization on components. This is what we validate against.
    - WR: DC (0-100) + Enhanced Breakout (0-99.9) + Teammate (0/100) + Early Declare (0/100)
    - RB: DC (0-100) + Production/1.75 (0-99.9) + Speed Score (0-100)
    - TE: DC (0-100) + Breakout (0-99.9) + Production min-max (0-99.9) + RAS×10 (0-100)
  Layer 2 — slap_display: Per-position 1-99 rescale of slap_raw.
    Best backtest player at each position = 99, worst = 1.
    This is what we publish. 2026 prospects on same scale (clipped 1-99).

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
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

print("=" * 120)
print("SLAP V5 — UNIFIED MASTER DATABASE BUILDER (native-scale components + per-position 1-99 display rescaling)")
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

# percentile_rank and score_prospect_against_backtest REMOVED in favor of native-scale scoring.
# Diagnostic testing showed percentile normalization destroys production signal for all 3 positions.


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
    """RYPTPA with age weighting. Raw score — percentile-normalized downstream."""
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return np.nan
    try:
        age = float(age)
    except (TypeError, ValueError):
        age = 22.0
    if pd.isna(age): age = 22
    season_age = age - 1
    age_w = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
    return (rec_yards / team_pass_att) * age_w * 100

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
print(f"\n  COMPONENT SCORING: NATIVE SCALE (no percentile normalization — /1.75 for RB, min-max for TE prod)")
print(f"  TWO-LAYER OUTPUT: slap_raw (validated) + slap_display (1-99 rescaled per position)")


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
wr_out = outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg', 'seasons_over_10ppg_3yr']].copy()
wr_bt = wr_bt.merge(wr_out, on=['player_name', 'draft_year', 'pick'], how='left')

# Calculate RAW components
wr_bt['s_dc'] = wr_bt['pick'].apply(dc_score)
wr_bt['s_breakout_raw'] = wr_bt.apply(
    lambda r: wr_enhanced_breakout(r['breakout_age'], r['peak_dominator'], r['rush_yards']), axis=1)
wr_bt['s_teammate_binary'] = wr_bt['total_teammate_dc'].apply(lambda x: 1 if pd.notna(x) and x > 150 else 0)
wr_bt['s_early_declare_binary'] = wr_bt['early_declare'].apply(lambda x: 1 if x == 1 else 0)

# NATIVE-SCALE SCORING: breakout is 0-99.9, binaries are 0/100 — all naturally 0-100 scale
# No percentile normalization needed (diagnostic confirmed raw beats percentile on all metrics)
wr_bt['s_teammate'] = np.where(wr_bt['s_teammate_binary'] == 1, 100, 0).astype(float)
wr_bt['s_early_declare'] = np.where(wr_bt['s_early_declare_binary'] == 1, 100, 0).astype(float)

tm_pct = wr_bt['s_teammate_binary'].mean()
ed_pct = wr_bt['s_early_declare_binary'].mean()
print(f"  Teammate: {tm_pct*100:.1f}% have flag (binary 0/100)")
print(f"  Early Declare: {ed_pct*100:.1f}% have flag (binary 0/100)")

# V5 score (native-scale components — this is slap_raw for validation)
wr_bt['slap_v5_raw'] = (
    WR_V5['dc'] * wr_bt['s_dc'] +
    WR_V5['breakout'] * wr_bt['s_breakout_raw'] +
    WR_V5['teammate'] * wr_bt['s_teammate'] +
    WR_V5['early_declare'] * wr_bt['s_early_declare']
)

# Data quality flags
wr_bt['breakout_data'] = np.where(wr_bt['breakout_age'].notna(), 'real', 'imputed')

print(f"  WR backtest: {len(wr_bt)} players, draft years {wr_bt['draft_year'].min()}-{wr_bt['draft_year'].max()}")
print(f"  Breakout: mean={wr_bt['s_breakout_raw'].mean():.1f}, range={wr_bt['s_breakout_raw'].min():.1f}-{wr_bt['s_breakout_raw'].max():.1f}")
print(f"  Teammate mean={wr_bt['s_teammate'].mean():.1f}, Early Declare mean={wr_bt['s_early_declare'].mean():.1f}")


# ============================================================================
# PART 2: RB BACKTEST (2015-2025) — with Speed Score MNAR imputation
# ============================================================================
print(f"\n{'='*120}")
print("PART 2: RB BACKTEST (with Speed Score MNAR imputation)")
print("=" * 120)

rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
rb_out = outcomes[outcomes['position'] == 'RB'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg', 'seasons_over_10ppg_3yr']].copy()
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

# NATIVE-SCALE SCORING: production scaled by /1.75 (0-99.9 range), speed already 0-100
rb_bt['s_production_scaled'] = (rb_bt['s_production_raw_filled'] / 1.75).clip(0, 99.9)

# V5 Score (native-scale components — this is slap_raw for validation)
rb_bt['slap_v5_raw'] = (
    RB_V5['dc'] * rb_bt['s_dc'] +
    RB_V5['production'] * rb_bt['s_production_scaled'] +
    RB_V5['speed_score'] * rb_bt['s_speed_raw']
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
print(f"  Production: raw mean={rb_bt['s_production_raw_filled'].mean():.1f} → scaled/1.75 mean={rb_bt['s_production_scaled'].mean():.1f}")
print(f"  Speed:      raw mean={rb_bt['s_speed_raw'].mean():.1f} (0-100 min-max normalized)")


# ============================================================================
# PART 3: TE BACKTEST (2015-2025) — with RAS from backtest data
# ============================================================================
print(f"\n{'='*120}")
print("PART 3: TE BACKTEST")
print("=" * 120)

te_bt = pd.read_csv('data/te_backtest_master.csv')

# Merge seasons_over_10ppg_3yr from outcomes
te_out_s10 = outcomes[outcomes['position'] == 'TE'][['player_name', 'draft_year', 'pick', 'seasons_over_10ppg_3yr']].copy()
te_bt = te_bt.merge(te_out_s10, on=['player_name', 'draft_year', 'pick'], how='left')

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
    if pd.notna(r.get('pff_yards')) and pd.notna(r.get('pff_pass_plays')) and r['pff_pass_plays'] > 0:
        season_age = (r['draft_age'] - 1) if pd.notna(r['draft_age']) else 22
        if season_age <= 19: aw = 1.15
        elif season_age <= 20: aw = 1.10
        elif season_age <= 21: aw = 1.05
        elif season_age <= 22: aw = 1.00
        elif season_age <= 23: aw = 0.95
        else: aw = 0.90
        te_bt.loc[idx, 'te_prod_raw'] = r['pff_yards'] / (r['pff_pass_plays'] * 1.15) * aw * 100

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

# NATIVE-SCALE SCORING: breakout (0-99.9), production min-max (0-99.9), RAS×10 (0-100)
# No percentile normalization — all components already on ~0-100 native scales

# V5 Score (native-scale components — this is slap_raw for validation)
te_bt['slap_v5_raw'] = (
    TE_V5['dc'] * te_bt['s_dc'] +
    TE_V5['breakout'] * te_bt['s_breakout_raw_filled'] +
    TE_V5['production'] * te_bt['s_production_raw_filled'] +
    TE_V5['ras'] * te_bt['s_ras_raw']
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
print(f"  Breakout:   mean={te_bt['s_breakout_raw_filled'].mean():.1f}, range={te_bt['s_breakout_raw_filled'].min():.1f}-{te_bt['s_breakout_raw_filled'].max():.1f}")
print(f"  Production: mean={te_bt['s_production_raw_filled'].mean():.1f}, range={te_bt['s_production_raw_filled'].min():.1f}-{te_bt['s_production_raw_filled'].max():.1f}")
print(f"  RAS:        mean={te_bt['s_ras_raw'].mean():.1f}, range={te_bt['s_ras_raw'].min():.1f}-{te_bt['s_ras_raw'].max():.1f}")


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

# Native-scale scoring (same as backtest: breakout 0-99.9, binaries 0/100)
wr26['s_breakout_raw'] = wr26['enhanced_breakout']  # Already on native 0-99.9 scale
wr26['s_teammate_binary'] = wr26['teammate_score'].apply(lambda x: 1 if x == 100 else 0)
wr26['s_early_declare_binary'] = wr26['early_declare'].apply(lambda x: 1 if x == 100 or x == 1 else 0)
wr26['s_teammate'] = np.where(wr26['s_teammate_binary'] == 1, 100, 0).astype(float)
wr26['s_early_declare'] = np.where(wr26['s_early_declare_binary'] == 1, 100, 0).astype(float)

# V5 score (native-scale components)
wr26['slap_v5_raw'] = (
    WR_V5['dc'] * wr26['s_dc'] +
    WR_V5['breakout'] * wr26['s_breakout_raw'] +
    WR_V5['teammate'] * wr26['s_teammate'] +
    WR_V5['early_declare'] * wr26['s_early_declare']
)

print(f"  WR 2026 prospects: {len(wr26)} players")
print(f"  Breakout range: {wr26['s_breakout_raw'].min():.1f} - {wr26['s_breakout_raw'].max():.1f}")


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

# Native-scale scoring (same as backtest: production /1.75, speed 0-100)
rb_prospects['s_production_scaled'] = (rb_prospects['s_production_raw'] / 1.75).clip(0, 99.9)

# Speed score for 2026 RBs: MNAR imputed on native 0-100 scale
ss_p60_raw = rb_bt['s_speed_raw'].quantile(0.60)
ss_p40_raw = rb_bt['s_speed_raw'].quantile(0.40)
rb_prospects['s_speed_raw'] = rb_prospects['projected_pick'].apply(
    lambda p: ss_p60_raw if p <= 64 else ss_p40_raw)

# V5 Score (native-scale components)
rb_prospects['slap_v5_raw'] = (
    RB_V5['dc'] * rb_prospects['s_dc'] +
    RB_V5['production'] * rb_prospects['s_production_scaled'] +
    RB_V5['speed_score'] * rb_prospects['s_speed_raw']
)

print(f"  RB 2026 prospects: {len(rb_prospects)} players")
print(f"  Production scaled range: {rb_prospects['s_production_scaled'].min():.1f} - {rb_prospects['s_production_scaled'].max():.1f}")


# ============================================================================
# PART 6: TE 2026 PROSPECTS — scored against backtest distribution
# ============================================================================
print(f"\n{'='*120}")
print("PART 6: TE 2026 PROSPECTS (scored against TE backtest distribution)")
print("=" * 120)

te26 = pd.read_csv('data/te_2026_prospects_final.csv')

te26['s_dc'] = te26['projected_pick'].apply(dc_score)

# Native-scale scoring (same as backtest: breakout 0-99.9, production min-max 0-99.9, RAS×10 0-100)
te26['s_breakout_raw'] = te26['breakout_score_filled']  # Already on native 0-99.9 scale

# Production: need to min-max normalize against the same TE backtest range
te26['s_production_raw'] = np.where(
    te26['production_score_filled'].notna(),
    ((te26['production_score_filled'] - te_prod_min) / (te_prod_max - te_prod_min) * 99.9).clip(0, 99.9),
    prod_avg_te_mm  # imputed with backtest avg
)

# RAS: already on 0-100 scale (RAS × 10)
te26['s_ras_raw'] = te26['ras_score']  # Already 0-100

# V5 Score (native-scale components)
te26['slap_v5_raw'] = (
    TE_V5['dc'] * te26['s_dc'] +
    TE_V5['breakout'] * te26['s_breakout_raw'] +
    TE_V5['production'] * te26['s_production_raw'] +
    TE_V5['ras'] * te26['s_ras_raw']
)

print(f"  TE 2026 prospects: {len(te26)} players")
print(f"  Breakout range: {te26['s_breakout_raw'].min():.1f} - {te26['s_breakout_raw'].max():.1f}")


# ============================================================================
# PER-POSITION RESCALING — slap_raw → slap_display (best backtest = 99, worst = 1)
# ============================================================================
print(f"\n\n{'='*120}")
print("PER-POSITION RESCALING: slap_raw → slap_display (1-99 per position)")
print("=" * 120)

# Store per-position scaling parameters (from backtest only — these define the scale)
pos_scale = {}
for pos, bt_df, p26_df in [('WR', wr_bt, wr26), ('RB', rb_bt, rb_prospects), ('TE', te_bt, te26)]:
    raw_min = bt_df['slap_v5_raw'].min()
    raw_max = bt_df['slap_v5_raw'].max()
    pos_scale[pos] = {'min': raw_min, 'max': raw_max}
    print(f"  {pos} backtest slap_raw range: {raw_min:.2f} - {raw_max:.2f}")

def position_rescale(raw_score, pos):
    """Rescale slap_raw to 1-99 using that position's backtest min/max."""
    mn = pos_scale[pos]['min']
    mx = pos_scale[pos]['max']
    if mx == mn:
        return 50.0
    return 1 + (raw_score - mn) / (mx - mn) * 98  # Maps min→1, max→99

# slap_display: per-position 1-99 rescale of slap_raw (this is what we publish)
wr_bt['slap_v5'] = wr_bt['slap_v5_raw'].apply(lambda x: position_rescale(x, 'WR')).round(1)
rb_bt['slap_v5'] = rb_bt['slap_v5_raw'].apply(lambda x: position_rescale(x, 'RB')).round(1)
te_bt['slap_v5'] = te_bt['slap_v5_raw'].apply(lambda x: position_rescale(x, 'TE')).round(1)

wr26['slap_v5'] = wr26['slap_v5_raw'].apply(lambda x: position_rescale(x, 'WR')).clip(1, 99).round(1)
rb_prospects['slap_v5'] = rb_prospects['slap_v5_raw'].apply(lambda x: position_rescale(x, 'RB')).clip(1, 99).round(1)
te26['slap_v5'] = te26['slap_v5_raw'].apply(lambda x: position_rescale(x, 'TE')).clip(1, 99).round(1)

# DC-only display score (rescaled within position for apples-to-apples delta)
# DC-only raw = DC × dc_weight + avg_non_dc_component × (1 - dc_weight)
# We approximate the "non-DC avg" from the backtest mean of non-DC components
wr_nondc_mean = (WR_V5['breakout'] * wr_bt['s_breakout_raw'].mean() + WR_V5['teammate'] * wr_bt['s_teammate'].mean() + WR_V5['early_declare'] * wr_bt['s_early_declare'].mean()) / (1 - WR_V5['dc'])
rb_nondc_mean = (RB_V5['production'] * rb_bt['s_production_scaled'].mean() + RB_V5['speed_score'] * rb_bt['s_speed_raw'].mean()) / (1 - RB_V5['dc'])
te_nondc_mean = (TE_V5['breakout'] * te_bt['s_breakout_raw_filled'].mean() + TE_V5['production'] * te_bt['s_production_raw_filled'].mean() + TE_V5['ras'] * te_bt['s_ras_raw'].mean()) / (1 - TE_V5['dc'])

wr_bt['dc_score_final'] = wr_bt['s_dc'].apply(lambda x: position_rescale(x * WR_V5['dc'] + wr_nondc_mean * (1 - WR_V5['dc']), 'WR')).round(1)
rb_bt['dc_score_final'] = rb_bt['s_dc'].apply(lambda x: position_rescale(x * RB_V5['dc'] + rb_nondc_mean * (1 - RB_V5['dc']), 'RB')).round(1)
te_bt['dc_score_final'] = te_bt['s_dc'].apply(lambda x: position_rescale(x * TE_V5['dc'] + te_nondc_mean * (1 - TE_V5['dc']), 'TE')).round(1)
wr26['dc_score_final'] = wr26['s_dc'].apply(lambda x: position_rescale(x * WR_V5['dc'] + wr_nondc_mean * (1 - WR_V5['dc']), 'WR')).round(1)
rb_prospects['dc_score_final'] = rb_prospects['s_dc'].apply(lambda x: position_rescale(x * RB_V5['dc'] + rb_nondc_mean * (1 - RB_V5['dc']), 'RB')).round(1)
te26['dc_score_final'] = te26['s_dc'].apply(lambda x: position_rescale(x * TE_V5['dc'] + te_nondc_mean * (1 - TE_V5['dc']), 'TE')).round(1)

# Delta = slap_display minus DC-only display (within same position scale)
wr_bt['delta_vs_dc'] = (wr_bt['slap_v5'] - wr_bt['dc_score_final']).round(1)
rb_bt['delta_vs_dc'] = (rb_bt['slap_v5'] - rb_bt['dc_score_final']).round(1)
te_bt['delta_vs_dc'] = (te_bt['slap_v5'] - te_bt['dc_score_final']).round(1)
wr26['delta_vs_dc'] = (wr26['slap_v5'] - wr26['dc_score_final']).round(1)
rb_prospects['delta_vs_dc'] = (rb_prospects['slap_v5'] - rb_prospects['dc_score_final']).round(1)
te26['delta_vs_dc'] = (te26['slap_v5'] - te26['dc_score_final']).round(1)

# Per-position stats
for pos, df in [('WR', wr_bt), ('RB', rb_bt), ('TE', te_bt)]:
    print(f"  {pos} backtest display: {df['slap_v5'].min():.1f} - {df['slap_v5'].max():.1f} (mean {df['slap_v5'].mean():.1f})")
    print(f"  {pos} backtest raw:     {df['slap_v5_raw'].min():.1f} - {df['slap_v5_raw'].max():.1f} (mean {df['slap_v5_raw'].mean():.1f})")


# ============================================================================
# RANKING CHECK: slap_raw vs slap_display should be perfectly rank-correlated
# ============================================================================
print(f"\n{'='*120}")
print("RANKING CHECK: slap_raw vs slap_display (should be perfectly correlated)")
print("=" * 120)

for pos, df in [('WR', wr_bt), ('RB', rb_bt), ('TE', te_bt)]:
    raw_rank = df['slap_v5_raw'].rank(ascending=False, method='min')
    display_rank = df['slap_v5'].rank(ascending=False, method='min')
    spearman_r = raw_rank.corr(display_rank, method='spearman')
    rank_diff = (raw_rank - display_rank).abs()
    max_move = rank_diff.max()
    print(f"  {pos}: Spearman r={spearman_r:.4f} | Max rank diff: {max_move:.0f} (should be 0)")


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
    'slap_raw': wr_bt['slap_v5_raw'].round(2),
    'enhanced_breakout': wr_bt['s_breakout_raw'].round(1),
    'teammate_score': wr_bt['s_teammate'].round(1),
    'early_declare_score': wr_bt['s_early_declare'].round(1),
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
    'nfl_seasons_10ppg_3yr': wr_bt.get('seasons_over_10ppg_3yr'),
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
    'slap_raw': rb_bt['slap_v5_raw'].round(2),
    'production_score': rb_bt['s_production_scaled'].round(1),
    'speed_score': rb_bt['s_speed_raw'].round(1),
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
    'nfl_seasons_10ppg_3yr': rb_bt.get('seasons_over_10ppg_3yr'),
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
    'slap_raw': te_bt['slap_v5_raw'].round(2),
    'te_breakout_score': te_bt['s_breakout_raw_filled'].round(1),
    'te_production_score': te_bt['s_production_raw_filled'].round(1),
    'ras_score': te_bt['s_ras_raw'].round(1),
    'nfl_hit24': te_bt['top12_8g'] if 'top12_8g' in te_bt.columns else te_bt['top12_10g'],
    'nfl_hit12': te_bt['top6_8g'] if 'top6_8g' in te_bt.columns else te_bt['top6_10g'],
    'nfl_first_3yr_ppg': te_bt['best_3yr_ppg_8g'] if 'best_3yr_ppg_8g' in te_bt.columns else te_bt['best_3yr_ppg_10g'],
    'nfl_career_ppg': te_bt['best_career_ppg_8g'] if 'best_career_ppg_8g' in te_bt.columns else te_bt['best_career_ppg_10g'],
    'nfl_best_ppr': te_bt['best_ppr'],
    'nfl_best_ppg': te_bt['best_ppg'],
    'nfl_seasons_10ppg_3yr': te_bt.get('seasons_over_10ppg_3yr'),
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
    'slap_raw': wr26['slap_v5_raw'].round(2),
    'enhanced_breakout': wr26['s_breakout_raw'].round(1),
    'teammate_score': wr26['s_teammate'].round(1),
    'early_declare_score': wr26['s_early_declare'].round(1),
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
    'nfl_seasons_10ppg_3yr': np.nan,
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
    'slap_raw': rb_prospects['slap_v5_raw'].round(2),
    'production_score': rb_prospects['s_production_scaled'].round(1),
    'speed_score': rb_prospects['s_speed_raw'].round(1),
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
    'nfl_seasons_10ppg_3yr': np.nan,
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
    'slap_raw': te26['slap_v5_raw'].round(2),
    'te_breakout_score': te26['s_breakout_raw'].round(1),
    'te_production_score': te26['s_production_raw'].round(1),
    'ras_score': te26['s_ras_raw'].round(1),
    'nfl_hit24': np.nan,
    'nfl_hit12': np.nan,
    'nfl_first_3yr_ppg': np.nan,
    'nfl_career_ppg': np.nan,
    'nfl_best_ppr': np.nan,
    'nfl_best_ppg': np.nan,
    'nfl_seasons_10ppg_3yr': np.nan,
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
    'slap_v5', 'slap_raw', 'dc_score', 'delta_vs_dc', 'data_type',
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
    'nfl_best_ppr', 'nfl_best_ppg', 'nfl_seasons_10ppg_3yr',
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

# 2026 prospects only (all positions) — ranked within position
prospects_2026 = master[master['data_type'] == '2026_prospect'].copy()
prospects_2026['pos_rank'] = prospects_2026.groupby('position')['slap_v5'].rank(ascending=False, method='min').astype(int)
prospects_2026 = prospects_2026.sort_values(['position', 'pos_rank']).reset_index(drop=True)
prospects_2026.to_csv('output/slap_v5_2026_all.csv', index=False)
print(f"  output/slap_v5_2026_all.csv: {len(prospects_2026)} prospects (all positions, ranked within position)")


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

# Per-position score distribution by round
bt_all = master[master['data_type'] == 'backtest']
print(f"\n  PER-POSITION SLAP BY DRAFT ROUND (each position on its own 1-99 scale):")
for pos in ['WR', 'RB', 'TE']:
    print(f"\n  {pos}:")
    print(f"  {'Round':>5} | {'Count':>5} | {'Mean':>6} | {'Min':>5} | {'Max':>5}")
    print(f"  {'-'*40}")
    for rd in range(1, 8):
        sub = bt_all[(bt_all['position'] == pos) & (bt_all['round'] == rd)]
        if len(sub) > 0:
            print(f"  {rd:>5} | {len(sub):>5} | {sub['slap_v5'].mean():>6.1f} | {sub['slap_v5'].min():>5.1f} | {sub['slap_v5'].max():>5.1f}")

# Show top 20 per position (backtest + 2026 combined, ranked within position)
for pos in ['WR', 'RB', 'TE']:
    pos_all = master[master['position'] == pos].sort_values('slap_v5', ascending=False)
    print(f"\n\n  TOP 20 {pos}s (backtest + 2026, within-position 1-99 scale):")
    print(f"  {'#':>3} {'Player':<25} {'Type':<10} {'Year':>4} {'Pick':>4} {'SLAP':>6} {'DC':>5} {'Delta':>6}")
    print(f"  {'-'*72}")
    for i, (_, r) in enumerate(pos_all.head(20).iterrows(), 1):
        delta_str = f"+{r['delta_vs_dc']:.1f}" if r['delta_vs_dc'] >= 0 else f"{r['delta_vs_dc']:.1f}"
        dtype = 'BT' if r['data_type'] == 'backtest' else '2026'
        print(f"  {i:>3} {r['player_name']:<25} {dtype:<10} {int(r['draft_year']):>4} {int(r['pick']):>4} "
              f"{r['slap_v5']:>6.1f} {r['dc_score']:>5.1f} {delta_str:>6}")

# Show top 20 2026 prospects per position
for pos in ['WR', 'RB', 'TE']:
    p26 = prospects_2026[prospects_2026['position'] == pos].head(20)
    print(f"\n\n  TOP 20 {pos} 2026 PROSPECTS:")
    print(f"  {'Rk':>3} {'Player':<25} {'School':<18} {'Pick':>4} {'SLAP':>6} {'DC':>5} {'Delta':>6}")
    print(f"  {'-'*72}")
    for i, (_, r) in enumerate(p26.iterrows(), 1):
        delta_str = f"+{r['delta_vs_dc']:.1f}" if r['delta_vs_dc'] >= 0 else f"{r['delta_vs_dc']:.1f}"
        print(f"  {i:>3} {r['player_name']:<25} {str(r['college']):<18} {int(r['pick']):>4} "
              f"{r['slap_v5']:>6.1f} {r['dc_score']:>5.1f} {delta_str:>6}")

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

# NON-DC COMPONENT MEANS (native scale)
print(f"\n\n  NON-DC COMPONENT MEANS (backtest, native scale):")
print(f"    WR: breakout mean={wr_bt['s_breakout_raw'].mean():.1f}, teammate mean={wr_bt['s_teammate'].mean():.1f}, early_declare mean={wr_bt['s_early_declare'].mean():.1f}")
print(f"    RB: production/1.75 mean={rb_bt['s_production_scaled'].mean():.1f}, speed mean={rb_bt['s_speed_raw'].mean():.1f}")
print(f"    TE: breakout mean={te_bt['s_breakout_raw_filled'].mean():.1f}, production mean={te_bt['s_production_raw_filled'].mean():.1f}, RAS mean={te_bt['s_ras_raw'].mean():.1f}")

print(f"\n\n{'='*120}")
print("MASTER DATABASE BUILD COMPLETE")
print(f"{'='*120}")
print(f"\n  Files saved:")
print(f"    output/slap_v5_master_database.csv  ({len(master)} rows)")
print(f"    output/slap_v5_wr.csv               ({len(wr_all)} rows)")
print(f"    output/slap_v5_rb.csv               ({len(rb_all)} rows)")
print(f"    output/slap_v5_te.csv               ({len(te_all)} rows)")
print(f"    output/slap_v5_2026_all.csv         ({len(prospects_2026)} rows)")
