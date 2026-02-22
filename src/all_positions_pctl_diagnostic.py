"""
Diagnostic: Percentile vs Raw/Scaled scoring for ALL 3 positions.
Tests whether percentile normalization helps or hurts PRI-AVG for WR, RB, TE.
"""
import pandas as pd
import numpy as np
import warnings, os
from scipy import stats as sp_stats
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

# ─── Shared helpers ──────────────────────────────────────────────────────

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

def percentile_rank(series):
    valid = series.dropna()
    if len(valid) == 0: return series.copy()
    ranks = sp_stats.rankdata(valid.values, method='average')
    pctiles = (ranks - 0.5) / len(ranks) * 100
    result = series.copy().astype(float)
    result.loc[valid.index] = pctiles
    return result

def normalize_0_100(series):
    mn, mx = series.dropna().min(), series.dropna().max()
    if mx == mn: return pd.Series(50, index=series.index)
    return ((series - mn) / (mx - mn) * 100).clip(0, 100)

def wr_breakout_score(breakout_age, dominator_pct):
    if pd.isna(breakout_age) or breakout_age is None:
        if dominator_pct is not None and pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base = age_tiers.get(int(breakout_age), 20)
    bonus = min((dominator_pct - 20) * 0.5, 9.9) if pd.notna(dominator_pct) and dominator_pct >= 20 else 0
    return min(base + bonus, 99.9)

def wr_enhanced_breakout(breakout_age, dominator_pct, rush_yards):
    base = wr_breakout_score(breakout_age, dominator_pct)
    rush_bonus = 5 if pd.notna(rush_yards) and rush_yards >= 20 else 0
    return min(base + rush_bonus, 99.9)

def rb_production_score(rec_yards, team_pass_att, age):
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0: return np.nan
    try: age = float(age)
    except: age = 22.0
    if pd.isna(age): age = 22
    season_age = age - 1
    age_w = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
    return (rec_yards / team_pass_att) * age_w * 100

def speed_score_fn(weight, forty):
    if pd.isna(weight) or pd.isna(forty) or forty <= 0 or weight <= 0: return np.nan
    return (weight * 200) / (forty ** 4)

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
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0: return np.nan
    if season_year is None: season_year = draft_year - 1
    season_age = draft_age - (draft_year - season_year) if pd.notna(draft_age) else 22
    if season_age <= 19: aw = 1.15
    elif season_age <= 20: aw = 1.10
    elif season_age <= 21: aw = 1.05
    elif season_age <= 22: aw = 1.00
    elif season_age <= 23: aw = 0.95
    else: aw = 0.90
    return (rec_yards / team_pass_att) * aw * 100

# ─── PRI-AVG computation ─────────────────────────────────────────────────
def compute_pri_avg(slap_scores, dc_scores, outcomes_df, hit24_col='hit24', hit12_col='hit12'):
    """Compute PRI-AVG for SLAP vs DC. Returns (slap_pri, dc_pri, gap, details)."""
    results = {}
    for outcome, weight in [('first_3yr_ppg', 0.40), ('hit24', 0.25), ('hit12', 0.20), ('career_ppg', 0.15)]:
        col = hit24_col if outcome == 'hit24' else (hit12_col if outcome == 'hit12' else outcome)
        valid = outcomes_df[col].notna()
        if valid.sum() < 10:
            results[outcome] = {'slap_r': np.nan, 'dc_r': np.nan, 'n': 0, 'weight': weight}
            continue
        s = slap_scores[valid]
        d = dc_scores[valid]
        o = outcomes_df.loc[valid, col]
        results[outcome] = {
            'slap_r': s.corr(o),
            'dc_r': d.corr(o),
            'n': int(valid.sum()),
            'weight': weight,
        }
    slap_pri = sum(r['slap_r'] * r['weight'] for r in results.values() if pd.notna(r['slap_r']))
    dc_pri = sum(r['dc_r'] * r['weight'] for r in results.values() if pd.notna(r['dc_r']))
    return slap_pri, dc_pri, slap_pri - dc_pri, results


# ─── Load data ────────────────────────────────────────────────────────────
print("Loading data...")
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')

# === WR ===
wr_bt = pd.read_csv('data/wr_backtest_all_components.csv')
wr_tm = pd.read_csv('data/wr_teammate_scores.csv')
wr_bt = wr_bt.merge(wr_tm[['player_name', 'draft_year', 'total_teammate_dc']], on=['player_name', 'draft_year'], how='left')
wr_out = outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
wr_bt = wr_bt.merge(wr_out, on=['player_name', 'draft_year', 'pick'], how='left')

wr_bt['s_dc'] = wr_bt['pick'].apply(dc_score)
wr_bt['s_breakout_raw'] = wr_bt.apply(
    lambda r: wr_enhanced_breakout(r['breakout_age'], r['peak_dominator'], r['rush_yards']), axis=1)
wr_bt['s_teammate_binary'] = wr_bt['total_teammate_dc'].apply(lambda x: 1 if pd.notna(x) and x > 150 else 0)
wr_bt['s_early_declare_binary'] = wr_bt['early_declare'].apply(lambda x: 1 if x == 1 else 0)

# Percentile versions
wr_bt['s_breakout_pctl'] = percentile_rank(wr_bt['s_breakout_raw'])
tm_pct = wr_bt['s_teammate_binary'].mean()
ed_pct = wr_bt['s_early_declare_binary'].mean()
wr_tm_no  = (1 - tm_pct) / 2 * 100
wr_tm_yes = ((1 - tm_pct) + tm_pct / 2) * 100
wr_ed_no  = (1 - ed_pct) / 2 * 100
wr_ed_yes = ((1 - ed_pct) + ed_pct / 2) * 100
wr_bt['s_teammate_pctl'] = np.where(wr_bt['s_teammate_binary'] == 1, wr_tm_yes, wr_tm_no)
wr_bt['s_early_declare_pctl'] = np.where(wr_bt['s_early_declare_binary'] == 1, wr_ed_yes, wr_ed_no)

# WR SLAP variants
# 1. Current percentile pipeline
wr_bt['slap_pctl'] = (
    0.70 * wr_bt['s_dc'] +
    0.20 * wr_bt['s_breakout_pctl'] +
    0.05 * wr_bt['s_teammate_pctl'] +
    0.05 * wr_bt['s_early_declare_pctl']
)

# 2. Raw/native scale (breakout is 0-99.9, binaries are 0/100)
wr_bt['slap_raw'] = (
    0.70 * wr_bt['s_dc'] +
    0.20 * wr_bt['s_breakout_raw'] +
    0.05 * np.where(wr_bt['s_teammate_binary'] == 1, 100, 0) +
    0.05 * np.where(wr_bt['s_early_declare_binary'] == 1, 100, 0)
)

# DC-only (for comparison)
wr_dc_only = wr_bt['s_dc'].copy()

print(f"  WR: {len(wr_bt)} players")

# === RB ===
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
rb_out = outcomes[outcomes['position'] == 'RB'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
rb_bt = rb_bt.merge(rb_out, on=['player_name', 'draft_year', 'pick'], how='left')

rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)
rb_bt['s_production_raw'] = rb_bt.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb_bt['s_production_raw_filled'] = rb_bt['s_production_raw'].fillna(0)

# Speed score (same as build_master)
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
        sub2 = known[(known['wb'] == wb) & (known['rb_bkt'] == rdb)]
        if len(sub2) > 0: lookup_40[(wb, rdb)] = sub2['forty'].mean()
    wt_sub = known[known['wb'] == wb]
    if len(wt_sub) > 0:
        for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
            if (wb, rdb) not in lookup_40: lookup_40[(wb, rdb)] = wt_sub['forty'].mean()

rb_bt['forty_final'] = rb_bt['forty'].copy()
impute_mask = rb_bt['weight'].notna() & rb_bt['forty'].isna()
for idx in rb_bt[impute_mask].index:
    wb = wt_bucket(rb_bt.loc[idx, 'weight'])
    rdb = rd_bucket(rb_bt.loc[idx, 'round'])
    est = lookup_40.get((wb, rdb))
    if est is not None: rb_bt.loc[idx, 'forty_final'] = est

rb_bt['raw_ss'] = rb_bt.apply(lambda r: speed_score_fn(r['weight'], r['forty_final']), axis=1)
real_ss = rb_bt['raw_ss'].dropna()
p60 = real_ss.quantile(0.60)
p40 = real_ss.quantile(0.40)
for idx in rb_bt[rb_bt['raw_ss'].isna()].index:
    rd = rb_bt.loc[idx, 'round']
    rb_bt.loc[idx, 'raw_ss'] = p60 if rd <= 2 else p40
rb_bt['s_speed_raw'] = normalize_0_100(rb_bt['raw_ss'])

# Percentile versions
rb_bt['s_production_pctl'] = percentile_rank(rb_bt['s_production_raw_filled'])
rb_bt['s_speed_pctl'] = percentile_rank(rb_bt['s_speed_raw'])

# RB scaled production (/1.75)
rb_bt['s_production_scaled'] = (rb_bt['s_production_raw_filled'] / 1.75).clip(0, 99.9)

# RB SLAP variants
rb_bt['slap_pctl'] = 0.65 * rb_bt['s_dc'] + 0.30 * rb_bt['s_production_pctl'] + 0.05 * rb_bt['s_speed_pctl']
rb_bt['slap_raw'] = 0.65 * rb_bt['s_dc'] + 0.30 * rb_bt['s_production_scaled'] + 0.05 * rb_bt['s_speed_raw']
rb_dc_only = rb_bt['s_dc'].copy()

print(f"  RB: {len(rb_bt)} players")

# === TE ===
te_bt = pd.read_csv('data/te_backtest_master.csv')
# TE already has top12_8g/top6_8g and best_3yr_ppg_8g/best_career_ppg_8g
# Use 8g columns for hit24/hit12 and PPG
te_bt['hit24'] = te_bt['top12_8g']
te_bt['hit12'] = te_bt['top6_8g']
te_bt['first_3yr_ppg'] = te_bt['best_3yr_ppg_8g']
te_bt['career_ppg'] = te_bt['best_career_ppg_8g']

te_bt['s_dc'] = te_bt['pick'].apply(dc_score)
te_bt['s_breakout_raw'] = te_bt.apply(
    lambda r: te_breakout_score(r['breakout_age'], r['peak_dominator'], threshold=15), axis=1)
bo_avg_te = te_bt['s_breakout_raw'].mean()
te_bt['s_breakout_raw_filled'] = te_bt['s_breakout_raw'].fillna(bo_avg_te)

# Production
te_bt['te_prod_raw'] = te_bt.apply(lambda r: te_production_score_fn(
    r['cfbd_rec_yards'], r['cfbd_team_pass_att'], r['draft_age'], r['draft_year']), axis=1)
for name, vals in {
    'Dallas Goedert': {'cfbd_rec_yards': 1111, 'cfbd_team_pass_att': 455, 'draft_age': 23.0, 'draft_year': 2018},
    'Adam Shaheen': {'cfbd_rec_yards': 867, 'cfbd_team_pass_att': 328, 'draft_age': 22.3, 'draft_year': 2017},
}.items():
    mask = te_bt['player_name'] == name
    if mask.sum() > 0 and pd.isna(te_bt.loc[mask, 'te_prod_raw'].values[0]):
        te_bt.loc[mask, 'te_prod_raw'] = te_production_score_fn(
            vals['cfbd_rec_yards'], vals['cfbd_team_pass_att'], vals['draft_age'], vals['draft_year'])

prod_vals = te_bt['te_prod_raw'].dropna()
te_prod_min, te_prod_max = prod_vals.min(), prod_vals.max()
te_bt['s_production_minmax'] = np.where(
    te_bt['te_prod_raw'].notna(),
    ((te_bt['te_prod_raw'] - te_prod_min) / (te_prod_max - te_prod_min) * 99.9).clip(0, 99.9),
    np.nan)
prod_avg_te_mm = te_bt['s_production_minmax'].mean()
te_bt['s_production_raw_filled'] = pd.to_numeric(te_bt['s_production_minmax'], errors='coerce').fillna(prod_avg_te_mm)

# RAS
te_bt['s_ras_raw'] = te_bt['te_ras'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)
ras_real = te_bt['s_ras_raw'].dropna()
te_ras_p60 = ras_real.quantile(0.60)
te_ras_p40 = ras_real.quantile(0.40)
for idx in te_bt[te_bt['s_ras_raw'].isna()].index:
    rd = te_bt.loc[idx, 'round']
    te_bt.loc[idx, 's_ras_raw'] = te_ras_p60 if rd <= 2 else te_ras_p40

# Percentile versions
te_bt['s_breakout_pctl'] = percentile_rank(te_bt['s_breakout_raw_filled'])
te_bt['s_production_pctl'] = percentile_rank(te_bt['s_production_raw_filled'])
te_bt['s_ras_pctl'] = percentile_rank(te_bt['s_ras_raw'])

# TE SLAP variants
te_bt['slap_pctl'] = (
    0.60 * te_bt['s_dc'] +
    0.15 * te_bt['s_breakout_pctl'] +
    0.15 * te_bt['s_production_pctl'] +
    0.10 * te_bt['s_ras_pctl']
)
te_bt['slap_raw'] = (
    0.60 * te_bt['s_dc'] +
    0.15 * te_bt['s_breakout_raw_filled'] +
    0.15 * te_bt['s_production_raw_filled'] +
    0.10 * te_bt['s_ras_raw']
)

te_dc_only = te_bt['s_dc'].copy()

print(f"  TE: {len(te_bt)} players")


# ============================================================================
# COMPUTE PRI-AVG FOR ALL 3 POSITIONS × 2 CONFIGS
# ============================================================================
print(f"\n\n{'='*100}")
print("ALL-POSITION PERCENTILE vs RAW/SCALED DIAGNOSTIC")
print("=" * 100)

for pos, df, slap_pctl_col, slap_raw_col, dc_col, h24, h12 in [
    ('WR', wr_bt, 'slap_pctl', 'slap_raw', 's_dc', 'hit24', 'hit12'),
    ('RB', rb_bt, 'slap_pctl', 'slap_raw', 's_dc', 'hit24', 'hit12'),
    ('TE', te_bt, 'slap_pctl', 'slap_raw', 's_dc', 'hit24', 'hit12'),
]:
    print(f"\n{'─'*80}")
    print(f"  {pos} ({len(df)} players)")
    print(f"{'─'*80}")

    for label, slap_col in [('PERCENTILE (current)', slap_pctl_col), ('RAW/SCALED (native)', slap_raw_col)]:
        slap_pri, dc_pri, gap, details = compute_pri_avg(
            df[slap_col], df[dc_col], df, hit24_col=h24, hit12_col=h12)

        print(f"\n  {label}:")
        print(f"  {'Outcome':<20} {'SLAP r':>10} {'DC r':>10} {'Gap':>10} {'N':>6}")
        print(f"  {'-'*58}")
        for outcome in ['first_3yr_ppg', 'hit24', 'hit12', 'career_ppg']:
            r = details[outcome]
            if pd.notna(r['slap_r']):
                print(f"  {outcome:<20} {r['slap_r']:>+10.4f} {r['dc_r']:>+10.4f} {r['slap_r']-r['dc_r']:>+10.4f} {r['n']:>6}")
        print(f"  {'PRI-AVG':<20} {slap_pri:>+10.4f} {dc_pri:>+10.4f} {gap:>+10.4f}")

    # Also show top-decile stats
    for label, slap_col in [('PERCENTILE', slap_pctl_col), ('RAW/SCALED', slap_raw_col)]:
        n10 = max(1, len(df) // 10)
        top10_idx = df[slap_col].nlargest(n10).index
        top10_dc_idx = df[dc_col].nlargest(n10).index
        h24_s = df.loc[top10_idx, h24].mean() * 100
        h24_d = df.loc[top10_dc_idx, h24].mean() * 100
        ppg_col = 'first_3yr_ppg'
        valid_s = df.loc[top10_idx, ppg_col].dropna()
        valid_d = df.loc[top10_dc_idx, ppg_col].dropna()
        ppg_s = valid_s.mean() if len(valid_s) > 0 else np.nan
        ppg_d = valid_d.mean() if len(valid_d) > 0 else np.nan
        print(f"  {label} Top-{n10}: hit24={h24_s:.1f}% (DC:{h24_d:.1f}%), PPG={ppg_s:.1f} (DC:{ppg_d:.1f})")


# ============================================================================
# SIDE-BY-SIDE COMPARISON TABLE
# ============================================================================
print(f"\n\n{'='*100}")
print("SIDE-BY-SIDE COMPARISON TABLE")
print("=" * 100)

print(f"\n  {'Position':<10} {'Current(pctl) PRI':>18} {'Raw/Scaled PRI':>16} {'DC-only PRI':>13} {'Current gap':>13} {'Raw gap':>10}")
print(f"  {'-'*82}")

for pos, df, slap_pctl_col, slap_raw_col, dc_col, h24, h12 in [
    ('WR', wr_bt, 'slap_pctl', 'slap_raw', 's_dc', 'hit24', 'hit12'),
    ('RB', rb_bt, 'slap_pctl', 'slap_raw', 's_dc', 'hit24', 'hit12'),
    ('TE', te_bt, 'slap_pctl', 'slap_raw', 's_dc', 'hit24', 'hit12'),
]:
    pctl_pri, dc_pri, _, _ = compute_pri_avg(df[slap_pctl_col], df[dc_col], df, h24, h12)
    raw_pri, dc_pri2, _, _ = compute_pri_avg(df[slap_raw_col], df[dc_col], df, h24, h12)
    print(f"  {pos:<10} {pctl_pri:>+18.4f} {raw_pri:>+16.4f} {dc_pri:>+13.4f} {pctl_pri - dc_pri:>+13.4f} {raw_pri - dc_pri:>+10.4f}")

print(f"\n  Positive gap = SLAP beats DC-only. Larger = better.")
print(f"\n  RECOMMENDATION: If Raw/Scaled gap > Current gap for a position, switch to Raw/Scaled.")
