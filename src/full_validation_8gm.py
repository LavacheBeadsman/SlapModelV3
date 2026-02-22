"""
Full Validation Suite — All 3 Positions, 8-Game Minimum Outcomes
================================================================
Computes SLAP V5 scores from scratch (matching build_master_database_v5.py exactly),
then validates against 8gm outcomes for all 3 positions.

7 validation categories:
  1. Pearson and Spearman correlations
  2. AUC-ROC for binary outcomes
  3. Brier scores
  4. Bootstrap resampling (1000 iterations)
  5. Top-decile precision
  6. Calibration by tier
  7. Priority-weighted average correlation

seasons_over_10ppg_3yr included as supplementary (NOT used in validation scoring).
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import warnings, os
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

np.random.seed(42)
N_BOOT = 1000

# ============================================================================
# SHARED HELPERS (exact copies from build_master_database_v5.py)
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

def normalize_0_100(series):
    mn, mx = series.dropna().min(), series.dropna().max()
    if mx == mn: return pd.Series(50, index=series.index)
    return ((series - mn) / (mx - mn) * 100).clip(0, 100)

# WR helpers
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

# RB helpers
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

# TE helpers
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


# ============================================================================
# LOCKED V5 WEIGHTS
# ============================================================================
WR_W = {'dc': 0.70, 'breakout': 0.20, 'teammate': 0.05, 'early_declare': 0.05}
RB_W = {'dc': 0.65, 'production': 0.30, 'speed_score': 0.05}
TE_W = {'dc': 0.60, 'breakout': 0.15, 'production': 0.15, 'ras': 0.10}


# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
wr_bt = pd.read_csv('data/wr_backtest_all_components.csv')
wr_tm = pd.read_csv('data/wr_teammate_scores.csv')
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
te_bt = pd.read_csv('data/te_backtest_master.csv')
combine = pd.read_parquet('data/nflverse/combine.parquet')

wr_bt = wr_bt.merge(wr_tm[['player_name', 'draft_year', 'total_teammate_dc']],
                     on=['player_name', 'draft_year'], how='left')


# ============================================================================
# COMPUTE WR SLAP V5
# ============================================================================
print("Computing WR SLAP V5...")
wr_bt['s_dc'] = wr_bt['pick'].apply(dc_score)
wr_bt['s_breakout_raw'] = wr_bt.apply(
    lambda r: wr_enhanced_breakout(r['breakout_age'], r['peak_dominator'], r['rush_yards']), axis=1)
wr_bt['s_teammate_binary'] = wr_bt['total_teammate_dc'].apply(lambda x: 1 if pd.notna(x) and x > 150 else 0)
wr_bt['s_early_declare_binary'] = wr_bt['early_declare'].apply(lambda x: 1 if x == 1 else 0)
# Native-scale: breakout 0-99.9, binaries 0/100 (no percentile normalization)
wr_bt['s_teammate'] = np.where(wr_bt['s_teammate_binary'] == 1, 100, 0).astype(float)
wr_bt['s_early_declare'] = np.where(wr_bt['s_early_declare_binary'] == 1, 100, 0).astype(float)
wr_bt['slap'] = (WR_W['dc'] * wr_bt['s_dc'] + WR_W['breakout'] * wr_bt['s_breakout_raw'] +
                  WR_W['teammate'] * wr_bt['s_teammate'] + WR_W['early_declare'] * wr_bt['s_early_declare'])
wr_bt['dc_only'] = wr_bt['s_dc']

# Merge outcomes
wr_out = outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year', 'pick',
    'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg', 'seasons_over_10ppg_3yr']].copy()
wr_out.columns = ['player_name', 'draft_year', 'pick', 'o_hit24', 'o_hit12',
                   'o_first_3yr_ppg', 'o_career_ppg', 'o_s10_3yr']
wr_bt = wr_bt.merge(wr_out, on=['player_name', 'draft_year', 'pick'], how='left')
# Prefer backtest hit24/hit12 (these are the original well-defined outcomes)
wr_bt['o_hit24'] = wr_bt['o_hit24'].fillna(wr_bt['hit24'])
wr_bt['o_hit12'] = wr_bt['o_hit12'].fillna(wr_bt['hit12'])


# ============================================================================
# COMPUTE RB SLAP V5 (with Speed Score MNAR imputation)
# ============================================================================
print("Computing RB SLAP V5...")
rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)
rb_bt['s_prod_raw'] = rb_bt.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb_bt['s_prod_filled'] = rb_bt['s_prod_raw'].fillna(0)

# Speed Score
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

# 40-time estimation
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
p60 = real_ss.quantile(0.60)
p40 = real_ss.quantile(0.40)
for idx in rb_bt[rb_bt['raw_ss'].isna()].index:
    rb_bt.loc[idx, 'raw_ss'] = p60 if rb_bt.loc[idx, 'round'] <= 2 else p40
rb_bt['s_speed_raw'] = normalize_0_100(rb_bt['raw_ss'])

# Native-scale: production /1.75 (0-99.9), speed already 0-100 (no percentile normalization)
rb_bt['s_prod_scaled'] = (rb_bt['s_prod_filled'] / 1.75).clip(0, 99.9)
rb_bt['slap'] = (RB_W['dc'] * rb_bt['s_dc'] + RB_W['production'] * rb_bt['s_prod_scaled'] +
                  RB_W['speed_score'] * rb_bt['s_speed_raw'])
rb_bt['dc_only'] = rb_bt['s_dc']

# Merge outcomes
rb_out = outcomes[outcomes['position'] == 'RB'][['player_name', 'draft_year', 'pick',
    'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg', 'seasons_over_10ppg_3yr']].copy()
rb_out.columns = ['player_name', 'draft_year', 'pick', 'o_hit24', 'o_hit12',
                   'o_first_3yr_ppg', 'o_career_ppg', 'o_s10_3yr']
rb_bt = rb_bt.merge(rb_out, on=['player_name', 'draft_year', 'pick'], how='left')
rb_bt['o_hit24'] = rb_bt['o_hit24'].fillna(rb_bt['hit24'])
rb_bt['o_hit12'] = rb_bt['o_hit12'].fillna(rb_bt['hit12'])


# ============================================================================
# COMPUTE TE SLAP V5
# ============================================================================
print("Computing TE SLAP V5...")
te_bt['s_dc'] = te_bt['pick'].apply(dc_score)
te_bt['s_breakout_raw'] = te_bt.apply(
    lambda r: te_breakout_score(r['breakout_age'], r['peak_dominator'], threshold=15), axis=1)
te_bt['s_breakout_filled'] = te_bt['s_breakout_raw'].fillna(te_bt['s_breakout_raw'].mean())

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
te_bt['s_prod_minmax'] = np.where(te_bt['te_prod_raw'].notna(),
    ((te_bt['te_prod_raw'] - te_prod_min) / (te_prod_max - te_prod_min) * 99.9).clip(0, 99.9), np.nan)
te_bt['s_prod_filled'] = te_bt['s_prod_minmax'].fillna(te_bt['s_prod_minmax'].mean())

# RAS
te_bt['s_ras_raw'] = te_bt['te_ras'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)
ras_real = te_bt['s_ras_raw'].dropna()
te_ras_p60 = ras_real.quantile(0.60)
te_ras_p40 = ras_real.quantile(0.40)
for idx in te_bt[te_bt['s_ras_raw'].isna()].index:
    te_bt.loc[idx, 's_ras_raw'] = te_ras_p60 if te_bt.loc[idx, 'round'] <= 2 else te_ras_p40

# Native-scale: breakout 0-99.9, production min-max 0-99.9, RAS×10 0-100 (no percentile normalization)
te_bt['slap'] = (TE_W['dc'] * te_bt['s_dc'] + TE_W['breakout'] * te_bt['s_breakout_filled'] +
                  TE_W['production'] * te_bt['s_prod_filled'] + TE_W['ras'] * te_bt['s_ras_raw'])
te_bt['dc_only'] = te_bt['s_dc']

# TE outcomes: use 8gm columns from te_backtest_master + rebuilt outcomes for PPG
te_out = outcomes[outcomes['position'] == 'TE'][['player_name', 'draft_year', 'pick',
    'first_3yr_ppg', 'career_ppg', 'seasons_over_10ppg_3yr']].copy()
te_out.columns = ['player_name', 'draft_year', 'pick', 'o_first_3yr_ppg', 'o_career_ppg', 'o_s10_3yr']
te_bt = te_bt.merge(te_out, on=['player_name', 'draft_year', 'pick'], how='left')

# Use top12_8g/top6_8g for binary outcomes (8gm consistency)
te_bt['o_hit24'] = te_bt['top12_8g'] if 'top12_8g' in te_bt.columns else te_bt['hit24']
te_bt['o_hit12'] = te_bt['top6_8g'] if 'top6_8g' in te_bt.columns else te_bt['hit12']
# Use 8gm PPG from te_backtest if available, otherwise from rebuilt outcomes
if 'best_3yr_ppg_8g' in te_bt.columns:
    te_bt['o_first_3yr_ppg'] = te_bt['o_first_3yr_ppg'].fillna(te_bt['best_3yr_ppg_8g'])
if 'best_career_ppg_8g' in te_bt.columns:
    te_bt['o_career_ppg'] = te_bt['o_career_ppg'].fillna(te_bt['best_career_ppg_8g'])


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

OUTCOME_WEIGHTS = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}

def brier_score(predicted_prob, actual_binary):
    valid = pd.DataFrame({'p': predicted_prob, 'a': actual_binary}).dropna()
    if len(valid) == 0: return np.nan, 0
    return ((valid['p'] - valid['a']) ** 2).mean(), len(valid)


def run_validation(pos_name, df, binary_labels=('hit24', 'hit12')):
    """Run all 7 validation categories for one position."""

    h24_label, h12_label = binary_labels

    # Filter to players with outcomes
    eval_df = df[df['o_hit24'].notna() & (df['draft_year'] <= 2025)].copy()
    n_eval = len(eval_df)

    outcome_map = {
        'first_3yr_ppg': 'o_first_3yr_ppg',
        'career_ppg': 'o_career_ppg',
        'hit24': 'o_hit24',
        'hit12': 'o_hit12',
    }

    print(f"\n\n{'#'*120}")
    print(f"#  {pos_name} VALIDATION (N={n_eval}, 8-game minimum)")
    print(f"#  SLAP V5 vs DC-only baseline")
    print(f"{'#'*120}")

    base_h24 = eval_df['o_hit24'].mean()
    base_h12 = eval_df['o_hit12'].mean()
    has_ppg = eval_df['o_first_3yr_ppg'].notna().sum()
    print(f"\n  {h24_label} base rate: {base_h24:.1%} ({int(eval_df['o_hit24'].sum())}/{n_eval})")
    print(f"  {h12_label} base rate: {base_h12:.1%} ({int(eval_df['o_hit12'].sum())}/{n_eval})")
    print(f"  first_3yr_ppg coverage: {has_ppg}/{n_eval}")

    # ---- 1. CORRELATIONS ----
    print(f"\n{'='*100}")
    print(f"1. CORRELATIONS (Pearson / Spearman)")
    print(f"{'='*100}")
    print(f"\n  {'Outcome':<18} {'SLAP Prs':>9} {'DC Prs':>9} {'Gap':>8} {'SLAP Spr':>9} {'DC Spr':>9} {'Gap':>8} {'N':>5}")
    print(f"  {'-'*82}")

    for out_name, out_col in outcome_map.items():
        v = eval_df[['slap', 'dc_only', out_col]].dropna()
        if len(v) < 10:
            print(f"  {out_name:<18} insufficient data (N={len(v)})")
            continue
        rp_s = pearsonr(v['slap'], v[out_col])[0]
        rp_d = pearsonr(v['dc_only'], v[out_col])[0]
        rs_s = spearmanr(v['slap'], v[out_col])[0]
        rs_d = spearmanr(v['dc_only'], v[out_col])[0]
        disp_name = out_name if out_name not in ('hit24', 'hit12') else (h24_label if out_name == 'hit24' else h12_label)
        print(f"  {disp_name:<18} {rp_s:>+.4f}   {rp_d:>+.4f}   {rp_s-rp_d:>+.4f}   {rs_s:>+.4f}   {rs_d:>+.4f}   {rs_s-rs_d:>+.4f} {len(v):>5}")

    # ---- 2. AUC-ROC ----
    print(f"\n{'='*100}")
    print(f"2. AUC-ROC (higher = better)")
    print(f"{'='*100}")
    print(f"\n  {'Outcome':<18} {'SLAP AUC':>10} {'DC AUC':>10} {'Gap':>10} {'Winner':>8} {'Pos':>5} {'Neg':>5}")
    print(f"  {'-'*65}")

    for out_name, bin_col in [('hit24', 'o_hit24'), ('hit12', 'o_hit12')]:
        v = eval_df[['slap', 'dc_only', bin_col]].dropna()
        pos_n = int(v[bin_col].sum())
        neg_n = len(v) - pos_n
        if pos_n == 0 or neg_n == 0:
            print(f"  {out_name:<18} cannot compute (no variance)")
            continue
        auc_s = roc_auc_score(v[bin_col], v['slap'])
        auc_d = roc_auc_score(v[bin_col], v['dc_only'])
        winner = "SLAP" if auc_s > auc_d else "DC"
        disp = h24_label if out_name == 'hit24' else h12_label
        print(f"  {disp:<18} {auc_s:>10.4f} {auc_d:>10.4f} {auc_s-auc_d:>+10.4f} {winner:>8} {pos_n:>5} {neg_n:>5}")

    # ---- 3. BRIER SCORES ----
    print(f"\n{'='*100}")
    print(f"3. BRIER SCORES (lower = better)")
    print(f"{'='*100}")
    print(f"\n  {'Outcome':<18} {'SLAP Brier':>12} {'DC Brier':>12} {'Gap':>12} {'Winner':>8}")
    print(f"  {'-'*60}")

    eval_df['slap_prob'] = eval_df['slap'] / 100
    eval_df['dc_prob'] = eval_df['dc_only'] / 100

    for out_name, bin_col in [('hit24', 'o_hit24'), ('hit12', 'o_hit12')]:
        bs_s, n_s = brier_score(eval_df['slap_prob'], eval_df[bin_col])
        bs_d, n_d = brier_score(eval_df['dc_prob'], eval_df[bin_col])
        if np.isnan(bs_s): continue
        winner = "SLAP" if bs_s < bs_d else "DC"
        disp = h24_label if out_name == 'hit24' else h12_label
        print(f"  {disp:<18} {bs_s:>12.4f} {bs_d:>12.4f} {bs_s-bs_d:>+12.4f} {winner:>8}")

    # ---- 4. BOOTSTRAP RESAMPLING ----
    print(f"\n{'='*100}")
    print(f"4. BOOTSTRAP CONFIDENCE INTERVALS ({N_BOOT} resamples)")
    print(f"{'='*100}")
    print(f"\n  {'Outcome':<18} {'SLAP r':>8} {'95% CI':>20} {'DC r':>8} {'95% CI':>20} {'SLAP>DC':>8}")
    print(f"  {'-'*88}")

    # Per-outcome bootstrap
    for out_name, out_col in outcome_map.items():
        v = eval_df[['slap', 'dc_only', out_col]].dropna()
        if len(v) < 20: continue
        r_s = pearsonr(v['slap'], v[out_col])[0]
        r_d = pearsonr(v['dc_only'], v[out_col])[0]
        boot_s, boot_d, wins = [], [], 0
        for _ in range(N_BOOT):
            samp = v.sample(n=len(v), replace=True)
            try:
                rs = pearsonr(samp['slap'], samp[out_col])[0]
                rd = pearsonr(samp['dc_only'], samp[out_col])[0]
                boot_s.append(rs); boot_d.append(rd)
                if rs > rd: wins += 1
            except: continue
        ci_s = f"[{np.percentile(boot_s,2.5):+.3f}, {np.percentile(boot_s,97.5):+.3f}]"
        ci_d = f"[{np.percentile(boot_d,2.5):+.3f}, {np.percentile(boot_d,97.5):+.3f}]"
        disp = out_name if out_name not in ('hit24', 'hit12') else (h24_label if out_name == 'hit24' else h12_label)
        print(f"  {disp:<18} {r_s:>+.4f} {ci_s:>20} {r_d:>+.4f} {ci_d:>20} {wins/len(boot_s)*100:>7.1f}%")

    # PRI-AVG bootstrap
    boot_pri_s, boot_pri_d, pri_wins = [], [], 0
    for _ in range(N_BOOT):
        samp = eval_df.sample(n=len(eval_df), replace=True)
        ps, pd_val, pt = 0, 0, 0
        for out_name, out_col in outcome_map.items():
            vv = samp[['slap', 'dc_only', out_col]].dropna()
            if len(vv) >= 10:
                try:
                    rs = pearsonr(vv['slap'], vv[out_col])[0]
                    rd = pearsonr(vv['dc_only'], vv[out_col])[0]
                    w = OUTCOME_WEIGHTS[out_name]
                    ps += w * rs; pd_val += w * rd; pt += w
                except: continue
        if pt > 0:
            boot_pri_s.append(ps/pt); boot_pri_d.append(pd_val/pt)
            if ps/pt > pd_val/pt: pri_wins += 1

    # Point estimates
    pri_s_pt, pri_d_pt, pri_t = 0, 0, 0
    for out_name, out_col in outcome_map.items():
        v = eval_df[['slap', 'dc_only', out_col]].dropna()
        if len(v) >= 10:
            rs = pearsonr(v['slap'], v[out_col])[0]
            rd = pearsonr(v['dc_only'], v[out_col])[0]
            w = OUTCOME_WEIGHTS[out_name]
            pri_s_pt += w * rs; pri_d_pt += w * rd; pri_t += w
    pri_s_pt /= pri_t; pri_d_pt /= pri_t

    ci_ps = f"[{np.percentile(boot_pri_s,2.5):+.3f}, {np.percentile(boot_pri_s,97.5):+.3f}]"
    ci_pd = f"[{np.percentile(boot_pri_d,2.5):+.3f}, {np.percentile(boot_pri_d,97.5):+.3f}]"
    print(f"\n  {'PRI-AVG':<18} {pri_s_pt:>+.4f} {ci_ps:>20} {pri_d_pt:>+.4f} {ci_pd:>20} {pri_wins/len(boot_pri_s)*100:>7.1f}%")
    print(f"  (SLAP > DC in {pri_wins/len(boot_pri_s)*100:.1f}% of {N_BOOT} resamples)")

    # ---- 5. TOP-DECILE PRECISION ----
    print(f"\n{'='*100}")
    print(f"5. TOP-DECILE PRECISION")
    print(f"{'='*100}")

    n_top = max(1, n_eval // 10)
    top_s = eval_df.nlargest(n_top, 'slap')
    top_d = eval_df.nlargest(n_top, 'dc_only')

    print(f"\n  Top 10% = {n_top} players")
    print(f"\n  {'Metric':<35} {'SLAP':>10} {'DC':>10} {'Gap':>10} {'Winner':>8}")
    print(f"  {'-'*75}")

    h24_s = top_s['o_hit24'].mean() * 100
    h24_d = top_d['o_hit24'].mean() * 100
    print(f"  {h24_label + ' rate':<35} {h24_s:>9.1f}% {h24_d:>9.1f}% {h24_s-h24_d:>+9.1f}% {'SLAP' if h24_s > h24_d else 'DC':>7}")

    h12_s = top_s['o_hit12'].mean() * 100
    h12_d = top_d['o_hit12'].mean() * 100
    print(f"  {h12_label + ' rate':<35} {h12_s:>9.1f}% {h12_d:>9.1f}% {h12_s-h12_d:>+9.1f}% {'SLAP' if h12_s > h12_d else 'DC':>7}")

    ppg_s = top_s[top_s['o_first_3yr_ppg'].notna()]['o_first_3yr_ppg'].mean()
    ppg_d = top_d[top_d['o_first_3yr_ppg'].notna()]['o_first_3yr_ppg'].mean()
    print(f"  {'avg first_3yr_ppg':<35} {ppg_s:>10.2f} {ppg_d:>10.2f} {ppg_s-ppg_d:>+10.2f} {'SLAP' if ppg_s > ppg_d else 'DC':>7}")

    # Top 20% too
    n_top20 = max(1, n_eval // 5)
    top20_s = eval_df.nlargest(n_top20, 'slap')
    top20_d = eval_df.nlargest(n_top20, 'dc_only')
    h24_20s = top20_s['o_hit24'].mean() * 100
    h24_20d = top20_d['o_hit24'].mean() * 100
    print(f"  {'Top 20% ' + h24_label + ' rate':<35} {h24_20s:>9.1f}% {h24_20d:>9.1f}% {h24_20s-h24_20d:>+9.1f}% {'SLAP' if h24_20s > h24_20d else 'DC':>7}")

    # ---- 6. CALIBRATION BY TIER ----
    print(f"\n{'='*100}")
    print(f"6. CALIBRATION BY TIER")
    print(f"{'='*100}")

    tiers = [(90, 101, '90+'), (80, 90, '80-89'), (70, 80, '70-79'),
             (60, 70, '60-69'), (50, 60, '50-59'), (0, 50, '<50')]

    for model_name, score_col in [('SLAP', 'slap'), ('DC-Only', 'dc_only')]:
        print(f"\n  {model_name}:")
        print(f"  {'Tier':<10} {'N':>5} {h24_label+' rate':>12} {h12_label+' rate':>12} {'Avg PPG':>10}")
        print(f"  {'-'*55}")
        for lo, hi, label in tiers:
            tier = eval_df[(eval_df[score_col] >= lo) & (eval_df[score_col] < hi)]
            if len(tier) == 0: continue
            r24 = tier['o_hit24'].mean() * 100
            r12 = tier['o_hit12'].mean() * 100
            ppg_v = tier[tier['o_first_3yr_ppg'].notna()]['o_first_3yr_ppg'].mean()
            ppg_str = f"{ppg_v:.2f}" if not np.isnan(ppg_v) else "N/A"
            print(f"  {label:<10} {len(tier):>5} {r24:>11.0f}% {r12:>11.0f}% {ppg_str:>10}")

    # ---- 7. PRIORITY-WEIGHTED AVERAGE CORRELATION ----
    print(f"\n{'='*100}")
    print(f"7. PRIORITY-WEIGHTED AVERAGE (40/25/20/15)")
    print(f"{'='*100}")
    print(f"\n  SLAP PRI-AVG:  {pri_s_pt:+.4f}")
    print(f"  DC   PRI-AVG:  {pri_d_pt:+.4f}")
    print(f"  Gap:           {pri_s_pt - pri_d_pt:+.4f}")
    print(f"  Bootstrap:     SLAP > DC in {pri_wins/len(boot_pri_s)*100:.1f}% of {N_BOOT} resamples")

    # ---- SCORECARD ----
    print(f"\n{'='*100}")
    print(f"SCORECARD: {pos_name}")
    print(f"{'='*100}")

    wins, ties, losses, total_metrics = 0, 0, 0, 0
    scoreboard = []

    # Pearson correlations
    for out_name, out_col in outcome_map.items():
        v = eval_df[['slap', 'dc_only', out_col]].dropna()
        if len(v) < 10: continue
        rp_s = pearsonr(v['slap'], v[out_col])[0]
        rp_d = pearsonr(v['dc_only'], v[out_col])[0]
        disp = out_name if out_name not in ('hit24', 'hit12') else (h24_label if out_name == 'hit24' else h12_label)
        w = "SLAP" if rp_s > rp_d else "DC" if rp_s < rp_d else "TIE"
        scoreboard.append((f"Pearson r ({disp})", rp_s, rp_d, w))
        total_metrics += 1
        if rp_s > rp_d: wins += 1
        elif rp_s < rp_d: losses += 1
        else: ties += 1

    # AUC
    for out_name, bin_col in [('hit24', 'o_hit24'), ('hit12', 'o_hit12')]:
        v = eval_df[['slap', 'dc_only', bin_col]].dropna()
        pos_n = int(v[bin_col].sum())
        neg_n = len(v) - pos_n
        if pos_n == 0 or neg_n == 0: continue
        auc_s = roc_auc_score(v[bin_col], v['slap'])
        auc_d = roc_auc_score(v[bin_col], v['dc_only'])
        disp = h24_label if out_name == 'hit24' else h12_label
        w = "SLAP" if auc_s > auc_d else "DC" if auc_s < auc_d else "TIE"
        scoreboard.append((f"AUC ({disp})", auc_s, auc_d, w))
        total_metrics += 1
        if auc_s > auc_d: wins += 1
        elif auc_s < auc_d: losses += 1
        else: ties += 1

    # Brier
    for out_name, bin_col in [('hit24', 'o_hit24'), ('hit12', 'o_hit12')]:
        bs_s, _ = brier_score(eval_df['slap_prob'], eval_df[bin_col])
        bs_d, _ = brier_score(eval_df['dc_prob'], eval_df[bin_col])
        if np.isnan(bs_s): continue
        disp = h24_label if out_name == 'hit24' else h12_label
        w = "SLAP" if bs_s < bs_d else "DC" if bs_s > bs_d else "TIE"  # lower = better
        scoreboard.append((f"Brier ({disp}) ↓", bs_s, bs_d, w))
        total_metrics += 1
        if bs_s < bs_d: wins += 1
        elif bs_s > bs_d: losses += 1
        else: ties += 1

    # Top decile hit rate
    w = "SLAP" if h24_s > h24_d else "DC" if h24_s < h24_d else "TIE"
    scoreboard.append((f"Top10% {h24_label} rate", h24_s, h24_d, w))
    total_metrics += 1
    if h24_s > h24_d: wins += 1
    elif h24_s < h24_d: losses += 1
    else: ties += 1

    # Top decile PPG
    w = "SLAP" if ppg_s > ppg_d else "DC" if ppg_s < ppg_d else "TIE"
    scoreboard.append(("Top10% avg PPG", ppg_s, ppg_d, w))
    total_metrics += 1
    if ppg_s > ppg_d: wins += 1
    elif ppg_s < ppg_d: losses += 1
    else: ties += 1

    # PRI-AVG
    w = "SLAP" if pri_s_pt > pri_d_pt else "DC"
    scoreboard.append(("PRI-AVG r", pri_s_pt, pri_d_pt, w))
    total_metrics += 1
    if pri_s_pt > pri_d_pt: wins += 1
    elif pri_s_pt < pri_d_pt: losses += 1
    else: ties += 1

    print(f"\n  {'Metric':<35} {'SLAP':>10} {'DC':>10} {'Winner':>8}")
    print(f"  {'-'*65}")
    for metric, val_s, val_d, winner in scoreboard:
        print(f"  {metric:<35} {val_s:>10.4f} {val_d:>10.4f} {winner:>8}")

    print(f"\n  SLAP wins {wins}/{total_metrics} metrics ({losses} DC wins, {ties} ties)")
    print(f"  Bootstrap: SLAP > DC in {pri_wins/len(boot_pri_s)*100:.1f}% of PRI-AVG resamples")

    # ---- SUPPLEMENTARY: seasons_over_10ppg_3yr ----
    print(f"\n{'='*100}")
    print(f"SUPPLEMENTARY: seasons_over_10ppg_3yr (NOT used in validation scoring)")
    print(f"{'='*100}")

    m = eval_df['o_s10_3yr'].notna()
    if m.sum() >= 10:
        r_s10_s = pearsonr(eval_df.loc[m, 'slap'], eval_df.loc[m, 'o_s10_3yr'])[0]
        r_s10_d = pearsonr(eval_df.loc[m, 'dc_only'], eval_df.loc[m, 'o_s10_3yr'])[0]
        sr_s10_s = spearmanr(eval_df.loc[m, 'slap'], eval_df.loc[m, 'o_s10_3yr'])[0]
        sr_s10_d = spearmanr(eval_df.loc[m, 'dc_only'], eval_df.loc[m, 'o_s10_3yr'])[0]
        print(f"  N={m.sum()}")
        print(f"  Distribution: {eval_df.loc[m, 'o_s10_3yr'].value_counts().sort_index().to_dict()}")
        print(f"  SLAP: Pearson {r_s10_s:+.4f}, Spearman {sr_s10_s:+.4f}")
        print(f"  DC:   Pearson {r_s10_d:+.4f}, Spearman {sr_s10_d:+.4f}")
        print(f"  Gap:  Pearson {r_s10_s-r_s10_d:+.4f}, Spearman {sr_s10_s-sr_s10_d:+.4f}")

        binary = (eval_df.loc[m, 'o_s10_3yr'] >= 1).astype(int)
        if binary.sum() > 0 and binary.sum() < m.sum():
            auc_s = roc_auc_score(binary, eval_df.loc[m, 'slap'])
            auc_d = roc_auc_score(binary, eval_df.loc[m, 'dc_only'])
            print(f"  AUC (1+ seasons): SLAP={auc_s:.3f}, DC={auc_d:.3f}, gap={auc_s-auc_d:+.3f}")

        binary2 = (eval_df.loc[m, 'o_s10_3yr'] >= 2).astype(int)
        if binary2.sum() > 0 and binary2.sum() < m.sum():
            auc_s2 = roc_auc_score(binary2, eval_df.loc[m, 'slap'])
            auc_d2 = roc_auc_score(binary2, eval_df.loc[m, 'dc_only'])
            print(f"  AUC (2+ seasons): SLAP={auc_s2:.3f}, DC={auc_d2:.3f}, gap={auc_s2-auc_d2:+.3f}")
    else:
        print(f"  Insufficient data (N={m.sum()})")

    return wins, total_metrics, pri_s_pt, pri_d_pt


# ============================================================================
# RUN VALIDATION FOR ALL 3 POSITIONS
# ============================================================================

print(f"\n\n{'*'*120}")
print(f"*  FULL VALIDATION SUITE — 8-GAME MINIMUM OUTCOMES")
print(f"*  WR V5: 70/20/5/5 | RB V5: 65/30/5 | TE V5: 60/15/15/10")
print(f"{'*'*120}")

results = {}

# WR
w_wr, t_wr, pri_s_wr, pri_d_wr = run_validation("WR", wr_bt, binary_labels=('hit24', 'hit12'))
results['WR'] = (w_wr, t_wr, pri_s_wr, pri_d_wr)

# RB
w_rb, t_rb, pri_s_rb, pri_d_rb = run_validation("RB", rb_bt, binary_labels=('hit24', 'hit12'))
results['RB'] = (w_rb, t_rb, pri_s_rb, pri_d_rb)

# TE
w_te, t_te, pri_s_te, pri_d_te = run_validation("TE", te_bt, binary_labels=('top12', 'top6'))
results['TE'] = (w_te, t_te, pri_s_te, pri_d_te)


# ============================================================================
# CROSS-POSITION SUMMARY
# ============================================================================
print(f"\n\n{'#'*120}")
print(f"#  CROSS-POSITION SUMMARY")
print(f"{'#'*120}")
print(f"\n  {'Position':<8} {'SLAP Wins':>10} {'Total':>6} {'Win%':>6} {'SLAP PRI-AVG':>14} {'DC PRI-AVG':>12} {'Gap':>8}")
print(f"  {'-'*65}")
total_wins = 0
total_total = 0
for pos in ['WR', 'RB', 'TE']:
    w, t, ps, pd = results[pos]
    total_wins += w
    total_total += t
    print(f"  {pos:<8} {w:>10}/{t:<4} {w/t*100:>5.0f}% {ps:>+14.4f} {pd:>+12.4f} {ps-pd:>+8.4f}")

print(f"  {'-'*65}")
print(f"  {'ALL':<8} {total_wins:>10}/{total_total:<4} {total_wins/total_total*100:>5.0f}%")
print(f"\n  SLAP V5 beats DC-only on {total_wins} of {total_total} metrics across all positions.")
print(f"\n{'='*120}")
print(f"VALIDATION COMPLETE")
print(f"{'='*120}")
