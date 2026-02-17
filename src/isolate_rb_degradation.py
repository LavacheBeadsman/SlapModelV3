"""
Isolate RB correlation degradation.
Three changes happened: (1) outcomes rebuilt, (2) percentile normalization, (3) per-position rescaling.
Test each independently.
"""
import pandas as pd
import numpy as np
import warnings, os
from scipy.stats import pearsonr, spearmanr, rankdata
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

# ============================================================================
# LOAD DATA
# ============================================================================
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
rb_out = outcomes[outcomes['position'] == 'RB'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
rb_bt = rb_bt.merge(rb_out, on=['player_name', 'draft_year', 'pick'], how='left')

# ============================================================================
# DC SCORE
# ============================================================================
def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)

# ============================================================================
# RAW PRODUCTION (no /1.75, just RYPTPA * age_weight * 100)
# ============================================================================
def rb_production_raw(rec_yards, team_pass_att, age):
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

rb_bt['prod_raw'] = rb_bt.apply(lambda r: rb_production_raw(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb_bt['prod_raw_filled'] = rb_bt['prod_raw'].fillna(0)

# ============================================================================
# SPEED SCORE (same pipeline as build script)
# ============================================================================
def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()

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

rb_bt['weight'] = rb_bt.apply(lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb_bt['forty'] = rb_bt.apply(lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb_bt['weight'] = pd.to_numeric(rb_bt['weight'], errors='coerce')
rb_bt['forty'] = pd.to_numeric(rb_bt['forty'], errors='coerce')

def speed_score_fn(weight, forty):
    if pd.isna(weight) or pd.isna(forty) or forty <= 0 or weight <= 0:
        return np.nan
    return (weight * 200) / (forty ** 4)

# 40-time estimation from weight x round buckets
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
real_ss = rb_bt['raw_ss'].dropna()
p60 = real_ss.quantile(0.60)
p40 = real_ss.quantile(0.40)
for idx in rb_bt[rb_bt['raw_ss'].isna()].index:
    rd = rb_bt.loc[idx, 'round']
    rb_bt.loc[idx, 'raw_ss'] = p60 if rd <= 2 else p40

mn_ss, mx_ss = rb_bt['raw_ss'].min(), rb_bt['raw_ss'].max()
rb_bt['speed_norm'] = ((rb_bt['raw_ss'] - mn_ss) / (mx_ss - mn_ss) * 100).clip(0, 100)

# ============================================================================
# PERCENTILE NORMALIZATION HELPERS
# ============================================================================
def percentile_rank(series):
    valid = series.dropna()
    if len(valid) == 0:
        return series.copy()
    ranks = rankdata(valid.values, method='average')
    pctiles = (ranks - 0.5) / len(ranks) * 100
    result = series.copy().astype(float)
    result.loc[valid.index] = pctiles
    return result

rb_bt['prod_pctl'] = percentile_rank(rb_bt['prod_raw_filled'])
rb_bt['speed_pctl'] = percentile_rank(rb_bt['speed_norm'])

# ============================================================================
# FILTER TO PLAYERS WITH first_3yr_ppg
# ============================================================================
has_ppg = rb_bt['first_3yr_ppg'].notna()
n_total = len(rb_bt)
n_ppg = has_ppg.sum()

print("=" * 80)
print("RB CORRELATION DEGRADATION ISOLATION")
print("=" * 80)
print(f"\n  Total RB backtest: {n_total}")
print(f"  With first_3yr_ppg: {n_ppg}")
print(f"  With career_ppg: {rb_bt['career_ppg'].notna().sum()}")

# Quick check: what does the outcomes data look like?
print(f"\n  Outcomes data stats:")
ppg_vals = rb_bt.loc[has_ppg, 'first_3yr_ppg']
print(f"    first_3yr_ppg: mean={ppg_vals.mean():.2f}, median={ppg_vals.median():.2f}, min={ppg_vals.min():.2f}, max={ppg_vals.max():.2f}")
print(f"    Draft years with data: {sorted(rb_bt.loc[has_ppg, 'draft_year'].unique())}")

# ============================================================================
# TEST 1: RAW PRODUCTION, 65/30/5, NO PERCENTILE, NO RESCALING
# ============================================================================
print(f"\n{'='*80}")
print("TEST 1: RAW production scores, 65/30/5 weights, NO percentile, NO rescaling")
print("=" * 80)

slap_t1 = 0.65 * rb_bt['s_dc'] + 0.30 * rb_bt['prod_raw_filled'] + 0.05 * rb_bt['speed_norm']
dc_only = rb_bt['s_dc']

mask = has_ppg
for outcome in ['first_3yr_ppg', 'career_ppg']:
    m = rb_bt[outcome].notna()
    r_slap, p_slap = pearsonr(slap_t1[m], rb_bt.loc[m, outcome])
    r_dc, p_dc = pearsonr(dc_only[m], rb_bt.loc[m, outcome])
    sr_slap = spearmanr(slap_t1[m], rb_bt.loc[m, outcome])[0]
    sr_dc = spearmanr(dc_only[m], rb_bt.loc[m, outcome])[0]
    print(f"  {outcome}:")
    print(f"    SLAP: Pearson r={r_slap:.4f} (p={p_slap:.4f}), Spearman r={sr_slap:.4f}")
    print(f"    DC:   Pearson r={r_dc:.4f} (p={p_dc:.4f}), Spearman r={sr_dc:.4f}")
    print(f"    SLAP-DC gap: Pearson {r_slap-r_dc:+.4f}, Spearman {sr_slap-sr_dc:+.4f}")

# Also check hit24, hit12
for outcome in ['hit24', 'hit12']:
    m = rb_bt[outcome].notna()
    r_slap = pearsonr(slap_t1[m], rb_bt.loc[m, outcome])[0]
    r_dc = pearsonr(dc_only[m], rb_bt.loc[m, outcome])[0]
    print(f"  {outcome}: SLAP r={r_slap:.4f}, DC r={r_dc:.4f}, gap={r_slap-r_dc:+.4f}")

# ============================================================================
# TEST 2: PERCENTILE-NORMALIZED production, 65/30/5, NO rescaling
# ============================================================================
print(f"\n{'='*80}")
print("TEST 2: PERCENTILE-NORMALIZED production, 65/30/5, NO rescaling")
print("=" * 80)

slap_t2 = 0.65 * rb_bt['s_dc'] + 0.30 * rb_bt['prod_pctl'] + 0.05 * rb_bt['speed_pctl']

for outcome in ['first_3yr_ppg', 'career_ppg']:
    m = rb_bt[outcome].notna()
    r_slap, p_slap = pearsonr(slap_t2[m], rb_bt.loc[m, outcome])
    r_dc, p_dc = pearsonr(dc_only[m], rb_bt.loc[m, outcome])
    sr_slap = spearmanr(slap_t2[m], rb_bt.loc[m, outcome])[0]
    sr_dc = spearmanr(dc_only[m], rb_bt.loc[m, outcome])[0]
    print(f"  {outcome}:")
    print(f"    SLAP: Pearson r={r_slap:.4f} (p={p_slap:.4f}), Spearman r={sr_slap:.4f}")
    print(f"    DC:   Pearson r={r_dc:.4f} (p={p_dc:.4f}), Spearman r={sr_dc:.4f}")
    print(f"    SLAP-DC gap: Pearson {r_slap-r_dc:+.4f}, Spearman {sr_slap-sr_dc:+.4f}")

for outcome in ['hit24', 'hit12']:
    m = rb_bt[outcome].notna()
    r_slap = pearsonr(slap_t2[m], rb_bt.loc[m, outcome])[0]
    r_dc = pearsonr(dc_only[m], rb_bt.loc[m, outcome])[0]
    print(f"  {outcome}: SLAP r={r_slap:.4f}, DC r={r_dc:.4f}, gap={r_slap-r_dc:+.4f}")

# ============================================================================
# TEST 3: FULL PIPELINE (percentile + per-position rescaling)
# ============================================================================
print(f"\n{'='*80}")
print("TEST 3: FULL PIPELINE (percentile + per-position 1-99 rescaling)")
print("=" * 80)

raw_min = slap_t2.min()
raw_max = slap_t2.max()
slap_t3 = 1 + (slap_t2 - raw_min) / (raw_max - raw_min) * 98
dc_rescaled = 1 + (dc_only - dc_only.min()) / (dc_only.max() - dc_only.min()) * 98

for outcome in ['first_3yr_ppg', 'career_ppg']:
    m = rb_bt[outcome].notna()
    r_slap = pearsonr(slap_t3[m], rb_bt.loc[m, outcome])[0]
    r_dc_r = pearsonr(dc_rescaled[m], rb_bt.loc[m, outcome])[0]
    sr_slap = spearmanr(slap_t3[m], rb_bt.loc[m, outcome])[0]
    sr_dc_r = spearmanr(dc_rescaled[m], rb_bt.loc[m, outcome])[0]
    print(f"  {outcome}:")
    print(f"    SLAP: Pearson r={r_slap:.4f}, Spearman r={sr_slap:.4f}")
    print(f"    DC:   Pearson r={r_dc_r:.4f}, Spearman r={sr_dc_r:.4f}")
    print(f"    SLAP-DC gap: Pearson {r_slap-r_dc_r:+.4f}, Spearman {sr_slap-sr_dc_r:+.4f}")

for outcome in ['hit24', 'hit12']:
    m = rb_bt[outcome].notna()
    r_slap = pearsonr(slap_t3[m], rb_bt.loc[m, outcome])[0]
    r_dc = pearsonr(dc_rescaled[m], rb_bt.loc[m, outcome])[0]
    print(f"  {outcome}: SLAP r={r_slap:.4f}, DC r={r_dc:.4f}, gap={r_slap-r_dc:+.4f}")

# ============================================================================
# TEST 4: RAW production, 65/35 (NO speed score), NO percentile, NO rescaling
# ============================================================================
print(f"\n{'='*80}")
print("TEST 4: RAW production, 65/35 weights (NO speed score), NO percentile, NO rescaling")
print("=" * 80)

slap_t4 = 0.65 * rb_bt['s_dc'] + 0.35 * rb_bt['prod_raw_filled']

for outcome in ['first_3yr_ppg', 'career_ppg']:
    m = rb_bt[outcome].notna()
    r_slap, p_slap = pearsonr(slap_t4[m], rb_bt.loc[m, outcome])
    sr_slap = spearmanr(slap_t4[m], rb_bt.loc[m, outcome])[0]
    r_dc = pearsonr(dc_only[m], rb_bt.loc[m, outcome])[0]
    sr_dc = spearmanr(dc_only[m], rb_bt.loc[m, outcome])[0]
    print(f"  {outcome}:")
    print(f"    SLAP: Pearson r={r_slap:.4f} (p={p_slap:.4f}), Spearman r={sr_slap:.4f}")
    print(f"    DC:   Pearson r={r_dc:.4f}, Spearman r={sr_dc:.4f}")
    print(f"    SLAP-DC gap: Pearson {r_slap-r_dc:+.4f}, Spearman {sr_slap-sr_dc:+.4f}")

for outcome in ['hit24', 'hit12']:
    m = rb_bt[outcome].notna()
    r_slap = pearsonr(slap_t4[m], rb_bt.loc[m, outcome])[0]
    r_dc = pearsonr(dc_only[m], rb_bt.loc[m, outcome])[0]
    print(f"  {outcome}: SLAP r={r_slap:.4f}, DC r={r_dc:.4f}, gap={r_slap-r_dc:+.4f}")

# ============================================================================
# SUMMARY COMPARISON TABLE
# ============================================================================
print(f"\n\n{'='*80}")
print("SUMMARY: Pearson r with first_3yr_ppg")
print("=" * 80)

m = has_ppg
r1 = pearsonr(slap_t1[m], rb_bt.loc[m, 'first_3yr_ppg'])[0]
r2 = pearsonr(slap_t2[m], rb_bt.loc[m, 'first_3yr_ppg'])[0]
r3 = pearsonr(slap_t3[m], rb_bt.loc[m, 'first_3yr_ppg'])[0]
r4 = pearsonr(slap_t4[m], rb_bt.loc[m, 'first_3yr_ppg'])[0]
r_dc = pearsonr(dc_only[m], rb_bt.loc[m, 'first_3yr_ppg'])[0]

print(f"\n  {'Test':<55} {'Pearson r':>10} {'vs DC':>8}")
print(f"  {'-'*75}")
print(f"  {'DC only (baseline)':.<55} {r_dc:>10.4f} {'---':>8}")
print(f"  {'T1: Raw 65/30/5 (no pctl, no rescale)':.<55} {r1:>10.4f} {r1-r_dc:>+8.4f}")
print(f"  {'T2: Percentile 65/30/5 (no rescale)':.<55} {r2:>10.4f} {r2-r_dc:>+8.4f}")
print(f"  {'T3: Percentile + rescale 65/30/5 (full pipeline)':.<55} {r3:>10.4f} {r3-r_dc:>+8.4f}")
print(f"  {'T4: Raw 65/35 (no speed, no pctl, no rescale)':.<55} {r4:>10.4f} {r4-r_dc:>+8.4f}")

print(f"\n  DEGRADATION ISOLATION:")
print(f"    Outcomes data effect: T1 r={r1:.4f} (this is what the current outcomes give us)")
print(f"    Percentile effect:   T2 - T1 = {r2-r1:+.4f}")
print(f"    Rescaling effect:    T3 - T2 = {r3-r2:+.4f}")
print(f"    Speed Score effect:  T4 - T1 = {r4-r1:+.4f} (removing speed score)")

# ============================================================================
# ALSO SHOW: raw production score distribution
# ============================================================================
print(f"\n\n{'='*80}")
print("RAW PRODUCTION SCORE DISTRIBUTION (before any normalization)")
print("=" * 80)
prod = rb_bt['prod_raw_filled']
print(f"  Count: {len(prod)}")
print(f"  Mean:  {prod.mean():.2f}")
print(f"  Std:   {prod.std():.2f}")
print(f"  Min:   {prod.min():.2f}")
print(f"  p25:   {prod.quantile(0.25):.2f}")
print(f"  p50:   {prod.quantile(0.50):.2f}")
print(f"  p75:   {prod.quantile(0.75):.2f}")
print(f"  p90:   {prod.quantile(0.90):.2f}")
print(f"  Max:   {prod.max():.2f}")

# How many zeros?
n_zero = (prod == 0).sum()
print(f"  Zeros (missing data): {n_zero} ({n_zero/len(prod)*100:.1f}%)")

# What is the range of DC vs production in weighted score?
print(f"\n  Scale comparison:")
print(f"    DC range: {rb_bt['s_dc'].min():.1f} - {rb_bt['s_dc'].max():.1f}")
print(f"    DC * 0.65: {rb_bt['s_dc'].min()*0.65:.1f} - {rb_bt['s_dc'].max()*0.65:.1f}")
print(f"    Prod raw range: {prod.min():.1f} - {prod.max():.1f}")
print(f"    Prod * 0.30: {prod.min()*0.30:.1f} - {prod.max()*0.30:.1f}")
print(f"    Prod pctl range: {rb_bt['prod_pctl'].min():.1f} - {rb_bt['prod_pctl'].max():.1f}")
print(f"    Prod pctl * 0.30: {rb_bt['prod_pctl'].min()*0.30:.1f} - {rb_bt['prod_pctl'].max()*0.30:.1f}")

print(f"\n  DONE.")
