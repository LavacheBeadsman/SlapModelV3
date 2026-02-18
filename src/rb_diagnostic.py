"""
RB Diagnostic — Why is PRI-AVG tied with DC-only?
4 tests: raw composite, 65/35, outcome comparison, year-by-year
"""
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import warnings, os
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

OUTCOME_WEIGHTS = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}

def pri_avg(df, slap_col, dc_col):
    """Compute PRI-AVG for SLAP and DC."""
    outcome_map = {'first_3yr_ppg': 'o_first_3yr_ppg', 'career_ppg': 'o_career_ppg',
                   'hit24': 'o_hit24', 'hit12': 'o_hit12'}
    ps, pd_val, pt = 0, 0, 0
    details = {}
    for out_name, out_col in outcome_map.items():
        v = df[[slap_col, dc_col, out_col]].dropna()
        if len(v) < 10: continue
        rs = pearsonr(v[slap_col], v[out_col])[0]
        rd = pearsonr(v[dc_col], v[out_col])[0]
        w = OUTCOME_WEIGHTS[out_name]
        ps += w * rs; pd_val += w * rd; pt += w
        details[out_name] = (rs, rd, len(v))
    return ps/pt, pd_val/pt, details

def full_metrics(df, slap_col, dc_col, label):
    """Print full metrics for one config."""
    outcome_map = {'first_3yr_ppg': 'o_first_3yr_ppg', 'career_ppg': 'o_career_ppg',
                   'hit24': 'o_hit24', 'hit12': 'o_hit12'}
    print(f"\n  {label}:")
    print(f"  {'Outcome':<18} {'SLAP Prs':>9} {'DC Prs':>9} {'Gap':>8} {'N':>5}")
    print(f"  {'-'*55}")
    for out_name, out_col in outcome_map.items():
        v = df[[slap_col, dc_col, out_col]].dropna()
        if len(v) < 10: continue
        rs = pearsonr(v[slap_col], v[out_col])[0]
        rd = pearsonr(v[dc_col], v[out_col])[0]
        print(f"  {out_name:<18} {rs:>+.4f}   {rd:>+.4f}   {rs-rd:>+.4f} {len(v):>5}")

    ps, pd_v, _ = pri_avg(df, slap_col, dc_col)
    print(f"  {'PRI-AVG':<18} {ps:>+.4f}   {pd_v:>+.4f}   {ps-pd_v:>+.4f}")

    # AUC
    for out_name, bin_col in [('hit24', 'o_hit24'), ('hit12', 'o_hit12')]:
        v = df[[slap_col, dc_col, bin_col]].dropna()
        pos_n = int(v[bin_col].sum())
        neg_n = len(v) - pos_n
        if pos_n == 0 or neg_n == 0: continue
        auc_s = roc_auc_score(v[bin_col], v[slap_col])
        auc_d = roc_auc_score(v[bin_col], v[dc_col])
        print(f"  AUC {out_name:<13} {auc_s:>+.4f}   {auc_d:>+.4f}   {auc_s-auc_d:>+.4f}")

    # Brier
    for out_name, bin_col in [('hit24', 'o_hit24'), ('hit12', 'o_hit12')]:
        v = df[[slap_col, dc_col, bin_col]].dropna()
        if len(v) == 0: continue
        bs_s = ((v[slap_col]/100 - v[bin_col])**2).mean()
        bs_d = ((v[dc_col]/100 - v[bin_col])**2).mean()
        print(f"  Brier {out_name:<11} {bs_s:>.4f}    {bs_d:>.4f}    {bs_s-bs_d:>+.4f} (lower=better)")

    # Top decile
    n_top = max(1, len(df) // 10)
    top_s = df.nlargest(n_top, slap_col)
    top_d = df.nlargest(n_top, dc_col)
    h24_s = top_s['o_hit24'].mean()*100
    h24_d = top_d['o_hit24'].mean()*100
    ppg_s = top_s[top_s['o_first_3yr_ppg'].notna()]['o_first_3yr_ppg'].mean()
    ppg_d = top_d[top_d['o_first_3yr_ppg'].notna()]['o_first_3yr_ppg'].mean()
    print(f"  Top10% hit24     {h24_s:>8.1f}%  {h24_d:>8.1f}%  {h24_s-h24_d:>+7.1f}%")
    print(f"  Top10% PPG       {ppg_s:>9.2f}  {ppg_d:>9.2f}  {ppg_s-ppg_d:>+8.2f}")

    return ps, pd_v


# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
outcomes_new = pd.read_csv('data/backtest_outcomes_complete.csv')
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
combine = pd.read_parquet('data/nflverse/combine.parquet')

# DC + Production
rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)
rb_bt['s_prod_raw'] = rb_bt.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb_bt['s_prod_filled'] = rb_bt['s_prod_raw'].fillna(0)

# Scale production to 0-99.9 using /1.75 (the RYPTPA formula from CLAUDE.md)
rb_bt['s_prod_scaled'] = (rb_bt['s_prod_filled'] / 1.75).clip(0, 99.9)

# Speed Score (full MNAR pipeline)
rb_bt['name_norm'] = rb_bt['player_name'].apply(normalize_name)
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
    return '<200' if wt < 200 else '200-209' if wt < 210 else '210-219' if wt < 220 else '220+'
def rd_bucket(rd):
    return 'Rd 1' if rd <= 1 else 'Rd 2' if rd <= 2 else 'Rd 3-4' if rd <= 4 else 'Rd 5+'
known['wb'] = known['weight'].apply(wt_bucket)
known['rb_bkt'] = known['round'].apply(rd_bucket)
lookup_40 = {}
for wb in ['<200', '200-209', '210-219', '220+']:
    for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
        sub = known[(known['wb'] == wb) & (known['rb_bkt'] == rdb)]
        if len(sub) > 0: lookup_40[(wb, rdb)] = sub['forty'].mean()
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
p60 = real_ss.quantile(0.60); p40 = real_ss.quantile(0.40)
for idx in rb_bt[rb_bt['raw_ss'].isna()].index:
    rb_bt.loc[idx, 'raw_ss'] = p60 if rb_bt.loc[idx, 'round'] <= 2 else p40
rb_bt['s_speed_raw'] = normalize_0_100(rb_bt['raw_ss'])

# Percentile versions (what the current pipeline uses)
rb_bt['s_prod_pctl'] = percentile_rank(rb_bt['s_prod_filled'])
rb_bt['s_speed_pctl'] = percentile_rank(rb_bt['s_speed_raw'])

# Merge NEW outcomes (8gm)
rb_out = outcomes_new[outcomes_new['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'hit24', 'hit12',
     'first_3yr_ppg', 'career_ppg', 'seasons_over_10ppg_3yr']].copy()
rb_out.columns = ['player_name', 'draft_year', 'pick', 'o_hit24', 'o_hit12',
                   'o_first_3yr_ppg', 'o_career_ppg', 'o_s10_3yr']
rb_bt = rb_bt.merge(rb_out, on=['player_name', 'draft_year', 'pick'], how='left')
rb_bt['o_hit24'] = rb_bt['o_hit24'].fillna(rb_bt['hit24'])
rb_bt['o_hit12'] = rb_bt['o_hit12'].fillna(rb_bt['hit12'])

# Filter to eval players
eval_df = rb_bt[rb_bt['o_hit24'].notna()].copy()
print(f"Eval players: {len(eval_df)}")


# ============================================================================
# BUILD ALL SCORE VARIANTS
# ============================================================================

# A) Current pipeline: percentile-normalized (what full_validation_8gm.py uses)
eval_df['slap_pctl'] = 0.65 * eval_df['s_dc'] + 0.30 * eval_df['s_prod_pctl'] + 0.05 * eval_df['s_speed_pctl']

# B) RAW 65/30/5: no percentile, uses raw RYPTPA + raw speed 0-100
eval_df['slap_raw_65_30_5'] = 0.65 * eval_df['s_dc'] + 0.30 * eval_df['s_prod_filled'] + 0.05 * eval_df['s_speed_raw']

# C) RAW 65/35: no speed, no normalization
eval_df['slap_raw_65_35'] = 0.65 * eval_df['s_dc'] + 0.35 * eval_df['s_prod_filled']

# D) Scaled 65/30/5: uses /1.75 production scaling + raw speed
eval_df['slap_scaled_65_30_5'] = 0.65 * eval_df['s_dc'] + 0.30 * eval_df['s_prod_scaled'] + 0.05 * eval_df['s_speed_raw']

# E) Scaled 65/35: uses /1.75 production scaling, no speed
eval_df['slap_scaled_65_35'] = 0.65 * eval_df['s_dc'] + 0.35 * eval_df['s_prod_scaled']

eval_df['dc_only'] = eval_df['s_dc']


# ============================================================================
# TEST 1: RAW COMPOSITE (65/30/5, no percentile normalization)
# ============================================================================
print(f"\n\n{'='*100}")
print("TEST 1: RAW 65/30/5 (no percentile normalization)")
print("  SLAP = 0.65*DC + 0.30*raw_production + 0.05*raw_speed_0_100")
print("=" * 100)
full_metrics(eval_df, 'slap_raw_65_30_5', 'dc_only', 'RAW 65/30/5')


# ============================================================================
# TEST 2: RAW 65/35 (no speed, no normalization)
# ============================================================================
print(f"\n\n{'='*100}")
print("TEST 2: RAW 65/35 (no speed score, no normalization)")
print("  SLAP = 0.65*DC + 0.35*raw_production")
print("=" * 100)
full_metrics(eval_df, 'slap_raw_65_35', 'dc_only', 'RAW 65/35')


# ============================================================================
# TEST 2b: SCALED variants
# ============================================================================
print(f"\n\n{'='*100}")
print("TEST 2b: SCALED 65/30/5 (production /1.75, no percentile)")
print("  SLAP = 0.65*DC + 0.30*(prod/1.75) + 0.05*speed_raw")
print("=" * 100)
full_metrics(eval_df, 'slap_scaled_65_30_5', 'dc_only', 'SCALED 65/30/5')

print(f"\n\n{'='*100}")
print("TEST 2c: SCALED 65/35 (production /1.75, no speed)")
print("  SLAP = 0.65*DC + 0.35*(prod/1.75)")
print("=" * 100)
full_metrics(eval_df, 'slap_scaled_65_35', 'dc_only', 'SCALED 65/35')


# ============================================================================
# CURRENT PIPELINE (for comparison)
# ============================================================================
print(f"\n\n{'='*100}")
print("CURRENT PIPELINE: Percentile-normalized 65/30/5")
print("  SLAP = 0.65*DC + 0.30*prod_pctl + 0.05*speed_pctl")
print("=" * 100)
full_metrics(eval_df, 'slap_pctl', 'dc_only', 'PERCENTILE 65/30/5')


# ============================================================================
# COMPARISON TABLE
# ============================================================================
print(f"\n\n{'='*100}")
print("COMPARISON: PRI-AVG across all configs")
print("=" * 100)

configs = [
    ('RAW 65/35 (no speed, no pctl)', 'slap_raw_65_35'),
    ('RAW 65/30/5 (no pctl)', 'slap_raw_65_30_5'),
    ('SCALED 65/35 (prod/1.75)', 'slap_scaled_65_35'),
    ('SCALED 65/30/5 (prod/1.75)', 'slap_scaled_65_30_5'),
    ('PERCENTILE 65/30/5 (current)', 'slap_pctl'),
]

print(f"\n  {'Config':<40} {'SLAP PRI':>10} {'DC PRI':>10} {'Gap':>10} {'first3yr gap':>13}")
print(f"  {'-'*85}")
for label, col in configs:
    ps, pd_v, details = pri_avg(eval_df, col, 'dc_only')
    f3_gap = details['first_3yr_ppg'][0] - details['first_3yr_ppg'][1] if 'first_3yr_ppg' in details else np.nan
    print(f"  {label:<40} {ps:>+10.4f} {pd_v:>+10.4f} {ps-pd_v:>+10.4f} {f3_gap:>+13.4f}")


# ============================================================================
# TEST 3: OUTCOME COMPARISON (old 6gm vs new 8gm)
# ============================================================================
print(f"\n\n{'='*100}")
print("TEST 3: OUTCOME COMPARISON — Old 6gm vs New 8gm")
print("=" * 100)

# Reconstruct old outcomes: re-run with 6gm logic using nflverse data
import unicodedata, re

stats = pd.read_csv('data/nflverse/player_stats_all_years.csv')
parquet_2025 = 'data/nflverse/player_stats_2025.parquet'
if os.path.exists(parquet_2025):
    stats_2025 = pd.read_parquet(parquet_2025)
    if int(stats['season'].max()) < 2025:
        stats = pd.concat([stats, stats_2025], ignore_index=True)
stats_reg = stats[stats['season_type'] == 'REG'].copy()
season_stats = stats_reg.groupby(['player_id', 'season']).agg(
    games=('fantasy_points_ppr', 'count'),
    total_ppr=('fantasy_points_ppr', 'sum'),
).reset_index()
season_stats['ppg'] = season_stats['total_ppr'] / season_stats['games']

draft = pd.read_parquet('data/nflverse/draft_picks.parquet')
draft_rel = draft[(draft['season'] >= 2015) & (draft['season'] <= 2025)]
draft_by_yp = {}
for _, d in draft_rel.iterrows():
    if pd.notna(d['gsis_id']):
        draft_by_yp[(int(d['season']), int(d['pick']))] = d['gsis_id']

rb_bt_full = pd.read_csv('data/rb_backtest_with_receiving.csv')
rb_bt_full['player_id'] = rb_bt_full.apply(
    lambda r: draft_by_yp.get((int(r['draft_year']), int(r['pick']))), axis=1)

def compute_ppg(player_id, draft_year, min_games):
    if pd.isna(player_id): return np.nan
    ps = season_stats[(season_stats['player_id'] == player_id)]
    if ps.empty: return np.nan
    q = ps[ps['games'] >= min_games]
    f3 = q[(q['season'] >= draft_year) & (q['season'] <= draft_year + 2)]
    if f3.empty: return np.nan
    return f3['ppg'].max()

rb_bt_full['ppg_6gm'] = rb_bt_full.apply(lambda r: compute_ppg(r['player_id'], int(r['draft_year']), 6), axis=1)
rb_bt_full['ppg_8gm'] = rb_bt_full.apply(lambda r: compute_ppg(r['player_id'], int(r['draft_year']), 8), axis=1)

# Count differences
both = rb_bt_full[rb_bt_full['ppg_6gm'].notna() | rb_bt_full['ppg_8gm'].notna()].copy()
both['changed'] = (both['ppg_6gm'] != both['ppg_8gm']) & both['ppg_6gm'].notna() & both['ppg_8gm'].notna()
both['lost'] = both['ppg_6gm'].notna() & both['ppg_8gm'].isna()  # had 6gm, lost at 8gm
both['gained'] = both['ppg_6gm'].isna() & both['ppg_8gm'].notna()  # gained at 8gm
both['same'] = (both['ppg_6gm'] == both['ppg_8gm']) & both['ppg_6gm'].notna()

both['s_dc'] = both['pick'].apply(dc_score)

print(f"\n  Players with 6gm outcome: {both['ppg_6gm'].notna().sum()}")
print(f"  Players with 8gm outcome: {both['ppg_8gm'].notna().sum()}")
print(f"  Same value: {both['same'].sum()}")
print(f"  Changed value: {both['changed'].sum()}")
print(f"  Lost (had 6gm, not 8gm): {both['lost'].sum()}")
print(f"  Gained (not 6gm, has 8gm): {both['gained'].sum()}")

# Show lost players (had outcome at 6gm, lost at 8gm)
lost = both[both['lost']]
if len(lost) > 0:
    print(f"\n  LOST PLAYERS (had first_3yr_ppg at 6gm, lost at 8gm):")
    print(f"  {'Player':<25} {'Year':>4} {'Pick':>4} {'DC':>6} {'6gm PPG':>8}")
    print(f"  {'-'*55}")
    for _, r in lost.sort_values('s_dc', ascending=False).iterrows():
        print(f"  {r['player_name']:<25} {int(r['draft_year']):>4} {int(r['pick']):>4} {r['s_dc']:>6.1f} {r['ppg_6gm']:>8.2f}")
    print(f"\n  Lost players avg DC: {lost['s_dc'].mean():.1f}")
    print(f"  Lost players avg 6gm PPG: {lost['ppg_6gm'].mean():.1f}")

# Changed players
changed = both[both['changed']]
if len(changed) > 0:
    changed['ppg_diff'] = changed['ppg_8gm'] - changed['ppg_6gm']
    print(f"\n  CHANGED PLAYERS (value differs between 6gm and 8gm): {len(changed)}")
    print(f"  Avg DC: {changed['s_dc'].mean():.1f}")
    print(f"  Avg PPG change: {changed['ppg_diff'].mean():+.2f}")
    print(f"  Went up (8gm > 6gm): {(changed['ppg_diff'] > 0).sum()}")
    print(f"  Went down (8gm < 6gm): {(changed['ppg_diff'] < 0).sum()}")
    print(f"  No change: {(changed['ppg_diff'] == 0).sum()}")

    # Did changes help high-DC or low-DC players?
    for dc_group, lo, hi in [('High DC (70+)', 70, 101), ('Mid DC (50-70)', 50, 70), ('Low DC (<50)', 0, 50)]:
        sub = changed[(changed['s_dc'] >= lo) & (changed['s_dc'] < hi)]
        if len(sub) > 0:
            print(f"    {dc_group}: N={len(sub)}, avg PPG change={sub['ppg_diff'].mean():+.2f}")

# 2025 draft class
c2025 = rb_bt_full[rb_bt_full['draft_year'] == 2025].copy()
c2025['s_dc'] = c2025['pick'].apply(dc_score)
print(f"\n  2025 DRAFT CLASS: {len(c2025)} RBs")
print(f"  With 8gm outcome: {c2025['ppg_8gm'].notna().sum()}")
print(f"  With 6gm outcome: {c2025['ppg_6gm'].notna().sum()}")
if c2025['ppg_8gm'].notna().sum() > 0:
    has_ppg = c2025[c2025['ppg_8gm'].notna()]
    print(f"  Avg DC: {has_ppg['s_dc'].mean():.1f}, Avg PPG: {has_ppg['ppg_8gm'].mean():.1f}")
    for _, r in has_ppg.sort_values('s_dc', ascending=False).iterrows():
        print(f"    {r['player_name']:<25} pick {int(r['pick']):>3} DC={r['s_dc']:.1f} PPG_8gm={r['ppg_8gm']:.1f}")


# ============================================================================
# TEST 4: YEAR-BY-YEAR SLAP vs DC delta
# ============================================================================
print(f"\n\n{'='*100}")
print("TEST 4: YEAR-BY-YEAR — SLAP vs DC Pearson r gap by draft class")
print("=" * 100)

# Use raw 65/35 as the SLAP score (cleanest signal)
eval_df['slap_test'] = eval_df['slap_raw_65_35']

print(f"\n  Using RAW 65/35 (cleanest comparison):")
print(f"  {'Year':>4} {'N':>4} {'N w/ PPG':>8} {'SLAP PPG r':>11} {'DC PPG r':>9} {'Gap':>8} {'SLAP h24 r':>11} {'DC h24 r':>9} {'Gap':>8}")
print(f"  {'-'*85}")
for yr in range(2015, 2026):
    sub = eval_df[eval_df['draft_year'] == yr]
    if len(sub) < 5: continue
    ppg_sub = sub[sub['o_first_3yr_ppg'].notna()]
    if len(ppg_sub) >= 5:
        rs_ppg = pearsonr(ppg_sub['slap_test'], ppg_sub['o_first_3yr_ppg'])[0]
        rd_ppg = pearsonr(ppg_sub['dc_only'], ppg_sub['o_first_3yr_ppg'])[0]
    else:
        rs_ppg, rd_ppg = np.nan, np.nan
    rs_h24 = pearsonr(sub['slap_test'], sub['o_hit24'])[0]
    rd_h24 = pearsonr(sub['dc_only'], sub['o_hit24'])[0]
    ppg_gap = rs_ppg - rd_ppg if not np.isnan(rs_ppg) else np.nan
    h24_gap = rs_h24 - rd_h24
    ppg_str = f"{rs_ppg:>+.3f}   {rd_ppg:>+.3f}   {ppg_gap:>+.3f}" if not np.isnan(rs_ppg) else "  N/A       N/A      N/A"
    print(f"  {yr:>4} {len(sub):>4} {len(ppg_sub):>8} {ppg_str} {rs_h24:>+11.3f} {rd_h24:>+.3f}   {h24_gap:>+.3f}")

# Same with percentile pipeline
print(f"\n  Using PERCENTILE 65/30/5 (current pipeline):")
print(f"  {'Year':>4} {'N':>4} {'N w/ PPG':>8} {'SLAP PPG r':>11} {'DC PPG r':>9} {'Gap':>8} {'SLAP h24 r':>11} {'DC h24 r':>9} {'Gap':>8}")
print(f"  {'-'*85}")
for yr in range(2015, 2026):
    sub = eval_df[eval_df['draft_year'] == yr]
    if len(sub) < 5: continue
    ppg_sub = sub[sub['o_first_3yr_ppg'].notna()]
    if len(ppg_sub) >= 5:
        rs_ppg = pearsonr(ppg_sub['slap_pctl'], ppg_sub['o_first_3yr_ppg'])[0]
        rd_ppg = pearsonr(ppg_sub['dc_only'], ppg_sub['o_first_3yr_ppg'])[0]
    else:
        rs_ppg, rd_ppg = np.nan, np.nan
    rs_h24 = pearsonr(sub['slap_pctl'], sub['o_hit24'])[0]
    rd_h24 = pearsonr(sub['dc_only'], sub['o_hit24'])[0]
    ppg_gap = rs_ppg - rd_ppg if not np.isnan(rs_ppg) else np.nan
    h24_gap = rs_h24 - rd_h24
    ppg_str = f"{rs_ppg:>+.3f}   {rd_ppg:>+.3f}   {ppg_gap:>+.3f}" if not np.isnan(rs_ppg) else "  N/A       N/A      N/A"
    print(f"  {yr:>4} {len(sub):>4} {len(ppg_sub):>8} {ppg_str} {rs_h24:>+11.3f} {rd_h24:>+.3f}   {h24_gap:>+.3f}")


# ============================================================================
# EXCLUDING 2025 TEST
# ============================================================================
print(f"\n\n{'='*100}")
print("BONUS: EXCLUDE 2025 — Does removing the newest class change results?")
print("=" * 100)

eval_no25 = eval_df[eval_df['draft_year'] < 2025].copy()
print(f"\n  Eval without 2025: {len(eval_no25)} players (removed {len(eval_df) - len(eval_no25)})")

print(f"\n  {'Config':<40} {'SLAP PRI':>10} {'DC PRI':>10} {'Gap':>10} {'first3yr gap':>13}")
print(f"  {'-'*85}")
for label, col in configs:
    ps, pd_v, details = pri_avg(eval_no25, col, 'dc_only')
    f3_gap = details['first_3yr_ppg'][0] - details['first_3yr_ppg'][1] if 'first_3yr_ppg' in details else np.nan
    print(f"  {label:<40} {ps:>+10.4f} {pd_v:>+10.4f} {ps-pd_v:>+10.4f} {f3_gap:>+13.4f}")

print(f"\n\n{'='*100}")
print("DIAGNOSIS COMPLETE")
print("=" * 100)
