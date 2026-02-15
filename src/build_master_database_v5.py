"""
SLAP V5 — Unified Master Database Builder
==========================================
Combines ALL three positions (WR, RB, TE) × (backtest + 2026 prospects)
into a single master CSV with consistent columns.

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
import warnings, os, time
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

print("=" * 120)
print("SLAP V5 — UNIFIED MASTER DATABASE BUILDER")
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

# Calculate components
wr_bt['s_dc'] = wr_bt['pick'].apply(dc_score)
wr_bt['s_breakout'] = wr_bt.apply(
    lambda r: wr_enhanced_breakout(r['breakout_age'], r['peak_dominator'], r['rush_yards']), axis=1)
wr_bt['s_teammate'] = wr_bt['total_teammate_dc'].apply(lambda x: 100 if pd.notna(x) and x > 150 else 0)
wr_bt['s_early_declare'] = wr_bt['early_declare'].apply(lambda x: 100 if x == 1 else 0)

# V5 score
wr_bt['slap_v5'] = (
    WR_V5['dc'] * wr_bt['s_dc'] +
    WR_V5['breakout'] * wr_bt['s_breakout'] +
    WR_V5['teammate'] * wr_bt['s_teammate'] +
    WR_V5['early_declare'] * wr_bt['s_early_declare']
).round(1)
wr_bt['delta_vs_dc'] = (wr_bt['slap_v5'] - wr_bt['s_dc']).round(1)

# Data quality flags
wr_bt['breakout_data'] = np.where(wr_bt['breakout_age'].notna(), 'real', 'imputed')
wr_bt['rush_data'] = np.where(wr_bt['rush_yards'].notna(), 'real', 'missing')
wr_bt['teammate_data'] = 'real'
wr_bt['early_declare_data'] = 'real'

# Build output rows
wr_rows = pd.DataFrame({
    'player_name': wr_bt['player_name'],
    'position': 'WR',
    'college': wr_bt['college'],
    'draft_year': wr_bt['draft_year'].astype(int),
    'pick': wr_bt['pick'].astype(int),
    'round': wr_bt['round'].astype(int),
    'slap_v5': wr_bt['slap_v5'],
    'dc_score': wr_bt['s_dc'].round(1),
    'delta_vs_dc': wr_bt['delta_vs_dc'],
    'data_type': 'backtest',
    # WR-specific components
    'enhanced_breakout': wr_bt['s_breakout'].round(1),
    'teammate_score': wr_bt['s_teammate'].astype(int),
    'early_declare_score': wr_bt['s_early_declare'].astype(int),
    # WR raw inputs
    'breakout_age': wr_bt['breakout_age'],
    'peak_dominator': wr_bt['peak_dominator'].round(1),
    'rush_yards': wr_bt['rush_yards'],
    # RB-specific (blank for WRs)
    'production_score': np.nan,
    'speed_score': np.nan,
    'rec_yards': np.nan,
    'team_pass_att': np.nan,
    # TE-specific (blank for WRs)
    'te_breakout_score': np.nan,
    'te_production_score': np.nan,
    'ras_score': np.nan,
    # NFL outcomes
    'nfl_hit24': wr_bt['hit24'],
    'nfl_hit12': wr_bt['hit12'],
    'nfl_first_3yr_ppg': wr_bt.get('first_3yr_ppg'),
    'nfl_career_ppg': wr_bt.get('career_ppg'),
    'nfl_best_ppr': wr_bt['best_ppr'],
    'nfl_best_ppg': np.nan,
    # Data quality
    'breakout_data_flag': wr_bt['breakout_data'],
    'production_data_flag': np.nan,
    'athletic_data_flag': np.nan,
})

print(f"  WR backtest: {len(wr_rows)} players, draft years {wr_bt['draft_year'].min()}-{wr_bt['draft_year'].max()}")
print(f"  SLAP range: {wr_rows['slap_v5'].min():.1f} - {wr_rows['slap_v5'].max():.1f}")


# ============================================================================
# PART 2: RB BACKTEST (2015-2025) — with Speed Score MNAR imputation
# ============================================================================
print(f"\n{'='*120}")
print("PART 2: RB BACKTEST (with Speed Score MNAR imputation)")
print("=" * 120)

rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
rb_out = outcomes[outcomes['position'] == 'RB'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
rb_bt = rb_bt.merge(rb_out, on=['player_name', 'draft_year', 'pick'], how='left')

# DC and Production
rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)
rb_bt['s_production'] = rb_bt.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb_bt['s_production_final'] = rb_bt['s_production'].fillna(0)

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

# MNAR-aware imputation
real_ss = rb_bt['raw_ss'].dropna()
p60 = real_ss.quantile(0.60)
p40 = real_ss.quantile(0.40)
for idx in rb_bt[rb_bt['raw_ss'].isna()].index:
    rd = rb_bt.loc[idx, 'round']
    rb_bt.loc[idx, 'raw_ss'] = p60 if rd <= 2 else p40

rb_bt['s_speed_score'] = normalize_0_100(rb_bt['raw_ss'])

# V5 Score
rb_bt['slap_v5'] = (
    RB_V5['dc'] * rb_bt['s_dc'] +
    RB_V5['production'] * rb_bt['s_production_final'] +
    RB_V5['speed_score'] * rb_bt['s_speed_score']
).round(1)
rb_bt['delta_vs_dc'] = (rb_bt['slap_v5'] - rb_bt['s_dc']).round(1)

# Athletic data flags
rb_bt['athletic_flag'] = 'real'
rb_bt.loc[impute_mask, 'athletic_flag'] = 'estimated_40'
rb_bt.loc[rb_bt['weight'].isna(), 'athletic_flag'] = 'mnar_imputed'

# Build output rows
rb_rows = pd.DataFrame({
    'player_name': rb_bt['player_name'],
    'position': 'RB',
    'college': rb_bt['college'],
    'draft_year': rb_bt['draft_year'].astype(int),
    'pick': rb_bt['pick'].astype(int),
    'round': rb_bt['round'].astype(int),
    'slap_v5': rb_bt['slap_v5'],
    'dc_score': rb_bt['s_dc'].round(1),
    'delta_vs_dc': rb_bt['delta_vs_dc'],
    'data_type': 'backtest',
    # WR-specific (blank for RBs)
    'enhanced_breakout': np.nan,
    'teammate_score': np.nan,
    'early_declare_score': np.nan,
    'breakout_age': np.nan,
    'peak_dominator': np.nan,
    'rush_yards': np.nan,
    # RB-specific components
    'production_score': rb_bt['s_production_final'].round(1),
    'speed_score': rb_bt['s_speed_score'].round(1),
    'rec_yards': rb_bt['rec_yards'],
    'team_pass_att': rb_bt['team_pass_att'],
    # TE-specific (blank for RBs)
    'te_breakout_score': np.nan,
    'te_production_score': np.nan,
    'ras_score': np.nan,
    # NFL outcomes
    'nfl_hit24': rb_bt['hit24'],
    'nfl_hit12': rb_bt['hit12'],
    'nfl_first_3yr_ppg': rb_bt.get('first_3yr_ppg'),
    'nfl_career_ppg': rb_bt.get('career_ppg'),
    'nfl_best_ppr': rb_bt['best_ppr'],
    'nfl_best_ppg': rb_bt['best_ppg'],
    # Data quality
    'breakout_data_flag': np.nan,
    'production_data_flag': np.where(rb_bt['s_production'].notna(), 'real', 'missing'),
    'athletic_data_flag': rb_bt['athletic_flag'],
})

n_real_ss = rb_bt['forty'].notna().sum()
n_imp40 = impute_mask.sum()
n_mnar = (rb_bt['weight'].isna()).sum()
print(f"  RB backtest: {len(rb_rows)} players, draft years {rb_bt['draft_year'].min()}-{rb_bt['draft_year'].max()}")
print(f"  Speed Score: {n_real_ss} real, {n_imp40} estimated-40, {n_mnar} MNAR-imputed")
print(f"  SLAP range: {rb_rows['slap_v5'].min():.1f} - {rb_rows['slap_v5'].max():.1f}")


# ============================================================================
# PART 3: TE BACKTEST (2015-2025) — with RAS from backtest data
# ============================================================================
print(f"\n{'='*120}")
print("PART 3: TE BACKTEST")
print("=" * 120)

te_bt = pd.read_csv('data/te_backtest_master.csv')

# DC
te_bt['s_dc'] = te_bt['pick'].apply(dc_score)

# Breakout (15% dominator threshold, TE-specific)
te_bt['s_breakout'] = te_bt.apply(
    lambda r: te_breakout_score(r['breakout_age'], r['peak_dominator'], threshold=15), axis=1)
bo_avg_te = te_bt['s_breakout'].mean()
te_bt['s_breakout_filled'] = te_bt['s_breakout'].fillna(bo_avg_te)

# Production — Rec/TPA with age weight, CFBD primary, PFF fallback
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
        # Use PFF rec per game as proxy — not ideal but better than nothing
        # Approximate team_pass_att from PFF data isn't available, so use group average ratio
        pass  # Leave as NaN — will be imputed

# Normalize production to 0-99.9 using min-max
prod_vals = te_bt['te_prod_raw'].dropna()
te_prod_min = prod_vals.min()
te_prod_max = prod_vals.max()
te_bt['s_production'] = np.where(
    te_bt['te_prod_raw'].notna(),
    ((te_bt['te_prod_raw'] - te_prod_min) / (te_prod_max - te_prod_min) * 99.9).clip(0, 99.9),
    np.nan
)
prod_avg_te = te_bt['s_production'].mean()
te_bt['s_production_filled'] = te_bt['s_production'].fillna(prod_avg_te)

# RAS — use actual TE RAS where available, MNAR impute rest
te_bt['s_ras'] = te_bt['te_ras'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)
ras_real = te_bt['s_ras'].dropna()
te_ras_p60 = ras_real.quantile(0.60)
te_ras_p40 = ras_real.quantile(0.40)
for idx in te_bt[te_bt['s_ras'].isna()].index:
    rd = te_bt.loc[idx, 'round']
    te_bt.loc[idx, 's_ras'] = te_ras_p60 if rd <= 2 else te_ras_p40

# V5 Score
te_bt['slap_v5'] = (
    TE_V5['dc'] * te_bt['s_dc'] +
    TE_V5['breakout'] * te_bt['s_breakout_filled'] +
    TE_V5['production'] * te_bt['s_production_filled'] +
    TE_V5['ras'] * te_bt['s_ras']
).round(1)
te_bt['delta_vs_dc'] = (te_bt['slap_v5'] - te_bt['s_dc']).round(1)

# Data flags
te_bt['breakout_flag'] = np.where(te_bt['breakout_age'].notna() | te_bt['peak_dominator'].notna(), 'real', 'imputed')
te_bt['production_flag'] = np.where(te_bt['te_prod_raw'].notna(), 'real', 'imputed')
te_bt['ras_flag'] = np.where(te_bt['te_ras'].notna(), 'real', 'mnar_imputed')

# Build output rows
te_rows = pd.DataFrame({
    'player_name': te_bt['player_name'],
    'position': 'TE',
    'college': te_bt['college'],
    'draft_year': te_bt['draft_year'].astype(int),
    'pick': te_bt['pick'].astype(int),
    'round': te_bt['round'].astype(int),
    'slap_v5': te_bt['slap_v5'],
    'dc_score': te_bt['s_dc'].round(1),
    'delta_vs_dc': te_bt['delta_vs_dc'],
    'data_type': 'backtest',
    # WR-specific (blank for TEs)
    'enhanced_breakout': np.nan,
    'teammate_score': np.nan,
    'early_declare_score': np.nan,
    'breakout_age': te_bt['breakout_age'],
    'peak_dominator': te_bt['peak_dominator'].round(1),
    'rush_yards': np.nan,
    # RB-specific (blank for TEs)
    'production_score': np.nan,
    'speed_score': np.nan,
    'rec_yards': te_bt['cfbd_rec_yards'],
    'team_pass_att': te_bt['cfbd_team_pass_att'],
    # TE-specific components
    'te_breakout_score': te_bt['s_breakout_filled'].round(1),
    'te_production_score': te_bt['s_production_filled'].round(1),
    'ras_score': te_bt['s_ras'].round(1),
    # NFL outcomes — TE uses 10g outcomes
    'nfl_hit24': te_bt['top12_10g'],   # TE "hit" = top12 TE season (10g min)
    'nfl_hit12': te_bt['top6_10g'],    # TE "elite hit" = top6 TE season (10g min)
    'nfl_first_3yr_ppg': te_bt['best_3yr_ppg_10g'],
    'nfl_career_ppg': te_bt['best_career_ppg_10g'],
    'nfl_best_ppr': te_bt['best_ppr'],
    'nfl_best_ppg': te_bt['best_ppg'],
    # Data quality
    'breakout_data_flag': te_bt['breakout_flag'],
    'production_data_flag': te_bt['production_flag'],
    'athletic_data_flag': te_bt['ras_flag'],
})

n_ras_real = (te_bt['te_ras'].notna()).sum()
print(f"  TE backtest: {len(te_rows)} players, draft years {te_bt['draft_year'].min()}-{te_bt['draft_year'].max()}")
n_bo = te_bt['breakout_age'].notna().sum()
n_pff = te_bt['peak_dominator'].notna().sum()
print(f"  Breakout: {n_bo}/{len(te_bt)} broke out, {n_pff}/{len(te_bt)} have PFF data, {len(te_bt)-n_pff} no data (default=25)")
print(f"  Production: {te_bt['te_prod_raw'].notna().sum()}/{len(te_bt)} real, rest imputed (avg={prod_avg_te:.1f})")
print(f"  RAS: {n_ras_real}/{len(te_bt)} real, rest MNAR-imputed (p60={te_ras_p60:.1f}, p40={te_ras_p40:.1f})")
print(f"  SLAP range: {te_rows['slap_v5'].min():.1f} - {te_rows['slap_v5'].max():.1f}")


# ============================================================================
# PART 4: WR 2026 PROSPECTS
# ============================================================================
print(f"\n{'='*120}")
print("PART 4: WR 2026 PROSPECTS")
print("=" * 120)

# Load pre-computed V5 WR 2026 scores (already calculated with correct weights)
wr26 = pd.read_csv('output/slap_v5_wr_2026.csv')

# Also load breakout ages for raw data
wr26_bo = pd.read_csv('data/wr_breakout_ages_2026.csv')
wr26 = wr26.merge(
    wr26_bo[['player_name', 'breakout_age', 'peak_dominator']].rename(
        columns={'breakout_age': 'bo_age_src', 'peak_dominator': 'pd_src'}),
    on='player_name', how='left')
# Use source breakout data where available
wr26['breakout_age'] = wr26['breakout_age'].fillna(wr26['bo_age_src'])
wr26['peak_dominator'] = wr26['peak_dominator'].fillna(wr26['pd_src'])

# Load prospect data for additional fields
prospects = pd.read_csv('data/prospects_final.csv')
wr_prospects = prospects[prospects['position'] == 'WR'].copy()
wr26 = wr26.merge(wr_prospects[['player_name', 'age', 'weight', 'age_estimated']],
                   on='player_name', how='left')

# Determine round from projected pick
def pick_to_round(pick):
    if pd.isna(pick): return np.nan
    if pick <= 32: return 1
    elif pick <= 64: return 2
    elif pick <= 100: return 3
    elif pick <= 135: return 4
    elif pick <= 176: return 5
    elif pick <= 220: return 6
    else: return 7

wr26_rows = pd.DataFrame({
    'player_name': wr26['player_name'],
    'position': 'WR',
    'college': wr26['school'],
    'draft_year': 2026,
    'pick': wr26['projected_pick'].astype(int),
    'round': wr26['projected_pick'].apply(pick_to_round).astype(int),
    'slap_v5': wr26['slap_v5'],
    'dc_score': wr26['dc_score'].round(1),
    'delta_vs_dc': wr26['delta_vs_dc'].round(1),
    'data_type': '2026_prospect',
    # WR-specific
    'enhanced_breakout': wr26['enhanced_breakout'].round(1),
    'teammate_score': wr26['teammate_score'].astype(int),
    'early_declare_score': wr26['early_declare'].astype(int),
    'breakout_age': wr26['breakout_age'],
    'peak_dominator': wr26['peak_dominator'].round(1) if 'peak_dominator' in wr26.columns else np.nan,
    'rush_yards': wr26['rush_yards'],
    # RB-specific (blank)
    'production_score': np.nan,
    'speed_score': np.nan,
    'rec_yards': np.nan,
    'team_pass_att': np.nan,
    # TE-specific (blank)
    'te_breakout_score': np.nan,
    'te_production_score': np.nan,
    'ras_score': np.nan,
    # NFL outcomes (none for prospects)
    'nfl_hit24': np.nan,
    'nfl_hit12': np.nan,
    'nfl_first_3yr_ppg': np.nan,
    'nfl_career_ppg': np.nan,
    'nfl_best_ppr': np.nan,
    'nfl_best_ppg': np.nan,
    # Data quality
    'breakout_data_flag': np.where(wr26['breakout_age'].notna(), 'real', 'imputed'),
    'production_data_flag': np.nan,
    'athletic_data_flag': np.nan,
})

print(f"  WR 2026 prospects: {len(wr26_rows)} players")
print(f"  SLAP range: {wr26_rows['slap_v5'].min():.1f} - {wr26_rows['slap_v5'].max():.1f}")


# ============================================================================
# PART 5: RB 2026 PROSPECTS — Recalculate with V5 weights
# ============================================================================
print(f"\n{'='*120}")
print("PART 5: RB 2026 PROSPECTS (recalculating with V5 weights)")
print("=" * 120)

rb_prospects = prospects[prospects['position'] == 'RB'].copy()

rb_prospects['s_dc'] = rb_prospects['projected_pick'].apply(dc_score)
rb_prospects['s_production'] = rb_prospects.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_attempts'], r['age']), axis=1)
rb_prospects['s_production_final'] = rb_prospects['s_production'].fillna(0)

# Speed Score for 2026 RBs — MNAR imputed (no combine data yet)
# Use the same percentiles from backtest for imputation
rb_prospects['s_speed_score'] = rb_prospects['projected_pick'].apply(
    lambda p: normalize_0_100(real_ss).quantile(0.60) if p <= 64 else normalize_0_100(real_ss).quantile(0.40))
# Actually, use a simpler approach: map the MNAR percentiles through the backtest normalization
# Rd 1-2 → 60th percentile of normalized speed scores, Rd 3+ → 40th percentile
ss_norm = normalize_0_100(rb_bt['raw_ss'])
ss_p60_norm = ss_norm.quantile(0.60)
ss_p40_norm = ss_norm.quantile(0.40)
rb_prospects['s_speed_score'] = rb_prospects['projected_pick'].apply(
    lambda p: ss_p60_norm if p <= 64 else ss_p40_norm)

rb_prospects['slap_v5'] = (
    RB_V5['dc'] * rb_prospects['s_dc'] +
    RB_V5['production'] * rb_prospects['s_production_final'] +
    RB_V5['speed_score'] * rb_prospects['s_speed_score']
).round(1)
rb_prospects['delta_vs_dc'] = (rb_prospects['slap_v5'] - rb_prospects['s_dc']).round(1)

rb26_rows = pd.DataFrame({
    'player_name': rb_prospects['player_name'],
    'position': 'RB',
    'college': rb_prospects['school'],
    'draft_year': 2026,
    'pick': rb_prospects['projected_pick'].astype(int),
    'round': rb_prospects['projected_pick'].apply(pick_to_round).astype(int),
    'slap_v5': rb_prospects['slap_v5'],
    'dc_score': rb_prospects['s_dc'].round(1),
    'delta_vs_dc': rb_prospects['delta_vs_dc'].round(1),
    'data_type': '2026_prospect',
    # WR-specific (blank)
    'enhanced_breakout': np.nan,
    'teammate_score': np.nan,
    'early_declare_score': np.nan,
    'breakout_age': np.nan,
    'peak_dominator': np.nan,
    'rush_yards': np.nan,
    # RB-specific
    'production_score': rb_prospects['s_production_final'].round(1),
    'speed_score': rb_prospects['s_speed_score'].round(1),
    'rec_yards': rb_prospects['rec_yards'],
    'team_pass_att': rb_prospects['team_pass_attempts'],
    # TE-specific (blank)
    'te_breakout_score': np.nan,
    'te_production_score': np.nan,
    'ras_score': np.nan,
    # NFL outcomes (none)
    'nfl_hit24': np.nan,
    'nfl_hit12': np.nan,
    'nfl_first_3yr_ppg': np.nan,
    'nfl_career_ppg': np.nan,
    'nfl_best_ppr': np.nan,
    'nfl_best_ppg': np.nan,
    # Data quality
    'breakout_data_flag': np.nan,
    'production_data_flag': np.where(rb_prospects['s_production'].notna(), 'real', 'missing'),
    'athletic_data_flag': 'mnar_imputed',
})

print(f"  RB 2026 prospects: {len(rb26_rows)} players")
print(f"  SLAP range: {rb26_rows['slap_v5'].min():.1f} - {rb26_rows['slap_v5'].max():.1f}")


# ============================================================================
# PART 6: TE 2026 PROSPECTS — Use pre-calculated scores
# ============================================================================
print(f"\n{'='*120}")
print("PART 6: TE 2026 PROSPECTS")
print("=" * 120)

te26 = pd.read_csv('data/te_2026_prospects_final.csv')

te26_rows = pd.DataFrame({
    'player_name': te26['player_name'],
    'position': 'TE',
    'college': te26['college'],
    'draft_year': 2026,
    'pick': te26['projected_pick'].astype(int),
    'round': te26['projected_pick'].apply(pick_to_round).astype(int),
    'slap_v5': te26['slap_score'].round(1),
    'dc_score': te26['dc_score'].round(1),
    'delta_vs_dc': te26['delta_vs_dc'].round(1),
    'data_type': '2026_prospect',
    # WR-specific (blank)
    'enhanced_breakout': np.nan,
    'teammate_score': np.nan,
    'early_declare_score': np.nan,
    'breakout_age': te26['breakout_age'],
    'peak_dominator': te26['peak_dominator'],
    'rush_yards': np.nan,
    # RB-specific (blank)
    'production_score': np.nan,
    'speed_score': np.nan,
    'rec_yards': te26['cfbd_rec_yards'],
    'team_pass_att': te26['cfbd_team_pass_att'],
    # TE-specific
    'te_breakout_score': te26['breakout_score_filled'].round(1),
    'te_production_score': te26['production_score_filled'].round(1),
    'ras_score': te26['ras_score'].round(1),
    # NFL outcomes (none)
    'nfl_hit24': np.nan,
    'nfl_hit12': np.nan,
    'nfl_first_3yr_ppg': np.nan,
    'nfl_career_ppg': np.nan,
    'nfl_best_ppr': np.nan,
    'nfl_best_ppg': np.nan,
    # Data quality
    'breakout_data_flag': np.where(te26['breakout_age'].notna(), 'real', 'imputed'),
    'production_data_flag': np.where(te26['production_score'].notna(), 'real', 'imputed'),
    'athletic_data_flag': 'mnar_imputed',
})

print(f"  TE 2026 prospects: {len(te26_rows)} players")
print(f"  SLAP range: {te26_rows['slap_v5'].min():.1f} - {te26_rows['slap_v5'].max():.1f}")


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

# WR file — drop TE/RB columns, add rank within position
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
    # Verify formula
    if pos == 'WR' and len(bt) > 0:
        sample = bt.head(1).iloc[0]
        expected = (0.70 * sample['dc_score'] + 0.20 * sample['enhanced_breakout'] +
                    0.05 * sample['teammate_score'] + 0.05 * sample['early_declare_score'])
        print(f"    Formula check: {sample['player_name']}: DC={sample['dc_score']}, BO={sample['enhanced_breakout']}, "
              f"TM={sample['teammate_score']}, ED={sample['early_declare_score']} → expected={expected:.1f}, actual={sample['slap_v5']}")
    elif pos == 'RB' and len(bt) > 0:
        sample = bt.head(1).iloc[0]
        expected = (0.65 * sample['dc_score'] + 0.30 * sample['production_score'] +
                    0.05 * sample['speed_score'])
        print(f"    Formula check: {sample['player_name']}: DC={sample['dc_score']}, Prod={sample['production_score']}, "
              f"SS={sample['speed_score']} → expected={expected:.1f}, actual={sample['slap_v5']}")
    elif pos == 'TE' and len(bt) > 0:
        sample = bt.head(1).iloc[0]
        expected = (0.60 * sample['dc_score'] + 0.15 * sample['te_breakout_score'] +
                    0.15 * sample['te_production_score'] + 0.10 * sample['ras_score'])
        print(f"    Formula check: {sample['player_name']}: DC={sample['dc_score']}, BO={sample['te_breakout_score']}, "
              f"Prod={sample['te_production_score']}, RAS={sample['ras_score']} → expected={expected:.1f}, actual={sample['slap_v5']}")

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

# Data quality summary
print(f"\n\n  DATA QUALITY SUMMARY:")
for pos in ['WR', 'RB', 'TE']:
    bt = master[(master['position'] == pos) & (master['data_type'] == 'backtest')]
    total = len(bt)
    if pos == 'WR':
        real_bo = (bt['breakout_data_flag'] == 'real').sum()
        print(f"  WR: {real_bo}/{total} real breakout ({total - real_bo} imputed)")
    elif pos == 'RB':
        real_prod = (bt['production_data_flag'] == 'real').sum()
        real_ath = (bt['athletic_data_flag'] == 'real').sum()
        print(f"  RB: {real_prod}/{total} real production, {real_ath}/{total} real speed score")
    elif pos == 'TE':
        real_bo = (bt['breakout_data_flag'] == 'real').sum()
        real_prod = (bt['production_data_flag'] == 'real').sum()
        real_ath = (bt['athletic_data_flag'] == 'real').sum()
        print(f"  TE: {real_bo}/{total} real breakout, {real_prod}/{total} real production, {real_ath}/{total} real RAS")

print(f"\n\n{'='*120}")
print("MASTER DATABASE BUILD COMPLETE")
print(f"{'='*120}")
print(f"\n  Files saved:")
print(f"    output/slap_v5_master_database.csv  ({len(master)} rows)")
print(f"    output/slap_v5_wr.csv               ({len(wr_all)} rows)")
print(f"    output/slap_v5_rb.csv               ({len(rb_all)} rows)")
print(f"    output/slap_v5_te.csv               ({len(te_all)} rows)")
print(f"    output/slap_v5_2026_all.csv         ({len(prospects_2026)} rows)")
