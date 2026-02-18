"""
Diagnose WHY the RB outcomes rebuild killed the production signal.
Three tests:
  1. Separate 6-game min vs 2025 data addition
  2. Show the ~27 new RBs added
  3. Check if PPG values changed — and compare old metric (best_ppg) vs new (first_3yr_ppg)
"""
import pandas as pd
import numpy as np
import warnings, os
from scipy.stats import pearsonr, spearmanr, rankdata
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

# ============================================================================
# LOAD EVERYTHING
# ============================================================================
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
rb_out = outcomes[outcomes['position'] == 'RB'].copy()

# Weekly stats for recomputing with different game minimums
stats = pd.read_csv('data/nflverse/player_stats_all_years.csv')
parquet_2025 = 'data/nflverse/player_stats_2025.parquet'
if os.path.exists(parquet_2025):
    stats_2025 = pd.read_parquet(parquet_2025)
    if int(stats['season'].max()) < 2025:
        stats = pd.concat([stats, stats_2025], ignore_index=True)
stats_reg = stats[stats['season_type'] == 'REG'].copy()

# Draft picks for player ID linking
draft = pd.read_parquet('data/nflverse/draft_picks.parquet')
draft_rel = draft[(draft['season'] >= 2015) & (draft['season'] <= 2025)]
draft_by_yp = {}
for _, d in draft_rel.iterrows():
    if pd.notna(d['gsis_id']):
        draft_by_yp[(int(d['season']), int(d['pick']))] = d['gsis_id']

# Link RBs to player IDs
rb_bt['player_id'] = rb_bt.apply(
    lambda r: draft_by_yp.get((int(r['draft_year']), int(r['pick']))), axis=1)

# Season-level stats
season_stats = stats_reg.groupby(['player_id', 'season']).agg(
    games=('fantasy_points_ppr', 'count'),
    total_ppr=('fantasy_points_ppr', 'sum'),
).reset_index()
season_stats['ppg'] = season_stats['total_ppr'] / season_stats['games']

# DC and production scores
def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

def rb_production_raw(rec_yards, team_pass_att, age):
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return np.nan
    try: age = float(age)
    except: age = 22.0
    if pd.isna(age): age = 22
    season_age = age - 1
    age_w = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
    return (rec_yards / team_pass_att) * age_w * 100

rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)
rb_bt['prod_raw'] = rb_bt.apply(lambda r: rb_production_raw(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb_bt['prod_raw_filled'] = rb_bt['prod_raw'].fillna(0)

# Simple SLAP at 65/30/5 with raw production (no speed for simplicity — using 65/35)
rb_bt['slap_raw_65_35'] = 0.65 * rb_bt['s_dc'] + 0.35 * rb_bt['prod_raw_filled']

# Merge current outcomes
rb_bt = rb_bt.merge(rb_out[['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']],
                     on=['player_name', 'draft_year', 'pick'], how='left')

print("=" * 90)
print("DIAGNOSING RB OUTCOMES REBUILD IMPACT")
print("=" * 90)
print(f"  Total RBs: {len(rb_bt)}")
print(f"  With player_id: {rb_bt['player_id'].notna().sum()}")
print(f"  With current first_3yr_ppg: {rb_bt['first_3yr_ppg'].notna().sum()}")

# ============================================================================
# HELPER: compute first_3yr_ppg with different parameters
# ============================================================================
def compute_first_3yr_ppg(player_id, draft_year, min_games=6, max_season=2025):
    """Best single-season PPG in first 3 years, min games threshold."""
    if pd.isna(player_id):
        return np.nan
    ps = season_stats[
        (season_stats['player_id'] == player_id) &
        (season_stats['season'] >= draft_year) &
        (season_stats['season'] <= draft_year + 2) &
        (season_stats['season'] <= max_season) &
        (season_stats['games'] >= min_games)
    ]
    if ps.empty:
        return np.nan
    return ps['ppg'].max()

def compute_old_best_ppg(player_id, min_games=1, max_season=2025):
    """Best single-season total PPR / 17 (original backtest definition)."""
    if pd.isna(player_id):
        return np.nan
    ps = season_stats[
        (season_stats['player_id'] == player_id) &
        (season_stats['season'] <= max_season) &
        (season_stats['games'] >= min_games)
    ]
    if ps.empty:
        return np.nan
    # Original best_ppr / 17 — use best total PPR season
    return ps['total_ppr'].max() / 17

# ============================================================================
# FIRST: What was the OLD metric? best_ppg from backtest file
# ============================================================================
print(f"\n{'='*90}")
print("TEST 0: OLD METRIC (best_ppg from original backtest file)")
print("=" * 90)

# The original backtest has 'best_ppg' column
has_old = rb_bt['best_ppg'].notna() & (rb_bt['best_ppg'] > 0)
n_old = has_old.sum()

r_dc_old, _ = pearsonr(rb_bt.loc[has_old, 's_dc'], rb_bt.loc[has_old, 'best_ppg'])
r_slap_old, _ = pearsonr(rb_bt.loc[has_old, 'slap_raw_65_35'], rb_bt.loc[has_old, 'best_ppg'])
sr_dc_old = spearmanr(rb_bt.loc[has_old, 's_dc'], rb_bt.loc[has_old, 'best_ppg'])[0]
sr_slap_old = spearmanr(rb_bt.loc[has_old, 'slap_raw_65_35'], rb_bt.loc[has_old, 'best_ppg'])[0]

print(f"  N={n_old} RBs with best_ppg > 0")
print(f"  DC-only  Pearson: {r_dc_old:.4f}   Spearman: {sr_dc_old:.4f}")
print(f"  SLAP     Pearson: {r_slap_old:.4f}   Spearman: {sr_slap_old:.4f}")
print(f"  SLAP-DC gap: Pearson {r_slap_old-r_dc_old:+.4f}, Spearman {sr_slap_old-sr_dc_old:+.4f}")

print(f"\n  best_ppg stats: mean={rb_bt.loc[has_old,'best_ppg'].mean():.2f}, "
      f"median={rb_bt.loc[has_old,'best_ppg'].median():.2f}, "
      f"min={rb_bt.loc[has_old,'best_ppg'].min():.2f}, max={rb_bt.loc[has_old,'best_ppg'].max():.2f}")

# Also try with current first_3yr_ppg on SAME players
has_both = has_old & rb_bt['first_3yr_ppg'].notna()
print(f"\n  Players with BOTH metrics: {has_both.sum()}")
if has_both.sum() > 10:
    r_dc_new_subset, _ = pearsonr(rb_bt.loc[has_both, 's_dc'], rb_bt.loc[has_both, 'first_3yr_ppg'])
    r_slap_new_subset, _ = pearsonr(rb_bt.loc[has_both, 'slap_raw_65_35'], rb_bt.loc[has_both, 'first_3yr_ppg'])
    print(f"  Same players, first_3yr_ppg: DC r={r_dc_new_subset:.4f}, SLAP r={r_slap_new_subset:.4f}, gap={r_slap_new_subset-r_dc_new_subset:+.4f}")

    r_dc_old_subset, _ = pearsonr(rb_bt.loc[has_both, 's_dc'], rb_bt.loc[has_both, 'best_ppg'])
    r_slap_old_subset, _ = pearsonr(rb_bt.loc[has_both, 'slap_raw_65_35'], rb_bt.loc[has_both, 'best_ppg'])
    print(f"  Same players, best_ppg:      DC r={r_dc_old_subset:.4f}, SLAP r={r_slap_old_subset:.4f}, gap={r_slap_old_subset-r_dc_old_subset:+.4f}")

# ============================================================================
# TEST 1A: Current outcomes with 8-game minimum (revert 6-game change)
# ============================================================================
print(f"\n{'='*90}")
print("TEST 1A: first_3yr_ppg with 8-GAME MINIMUM (revert 6-game change)")
print("=" * 90)

rb_bt['ppg_8gm'] = rb_bt.apply(
    lambda r: compute_first_3yr_ppg(r['player_id'], int(r['draft_year']), min_games=8), axis=1)

has_8gm = rb_bt['ppg_8gm'].notna()
print(f"  N={has_8gm.sum()} (vs {rb_bt['first_3yr_ppg'].notna().sum()} with 6-game min)")
print(f"  Lost {rb_bt['first_3yr_ppg'].notna().sum() - has_8gm.sum()} players by tightening to 8 games")

if has_8gm.sum() > 10:
    r_dc_8, _ = pearsonr(rb_bt.loc[has_8gm, 's_dc'], rb_bt.loc[has_8gm, 'ppg_8gm'])
    r_slap_8, _ = pearsonr(rb_bt.loc[has_8gm, 'slap_raw_65_35'], rb_bt.loc[has_8gm, 'ppg_8gm'])
    print(f"  DC-only  Pearson: {r_dc_8:.4f}")
    print(f"  SLAP     Pearson: {r_slap_8:.4f}")
    print(f"  Gap: {r_slap_8-r_dc_8:+.4f}")

# ============================================================================
# TEST 1B: Current outcomes WITHOUT 2025 NFL season data
# ============================================================================
print(f"\n{'='*90}")
print("TEST 1B: first_3yr_ppg WITHOUT 2025 NFL data (max_season=2024)")
print("=" * 90)

rb_bt['ppg_no2025'] = rb_bt.apply(
    lambda r: compute_first_3yr_ppg(r['player_id'], int(r['draft_year']), min_games=6, max_season=2024), axis=1)

has_no25 = rb_bt['ppg_no2025'].notna()
print(f"  N={has_no25.sum()} (vs {rb_bt['first_3yr_ppg'].notna().sum()} with 2025)")
print(f"  Lost {rb_bt['first_3yr_ppg'].notna().sum() - has_no25.sum()} players by removing 2025")

if has_no25.sum() > 10:
    r_dc_n25, _ = pearsonr(rb_bt.loc[has_no25, 's_dc'], rb_bt.loc[has_no25, 'ppg_no2025'])
    r_slap_n25, _ = pearsonr(rb_bt.loc[has_no25, 'slap_raw_65_35'], rb_bt.loc[has_no25, 'ppg_no2025'])
    print(f"  DC-only  Pearson: {r_dc_n25:.4f}")
    print(f"  SLAP     Pearson: {r_slap_n25:.4f}")
    print(f"  Gap: {r_slap_n25-r_dc_n25:+.4f}")

# Show who the 2025-only players are
only_2025 = rb_bt['first_3yr_ppg'].notna() & rb_bt['ppg_no2025'].isna()
if only_2025.sum() > 0:
    print(f"\n  Players who ONLY have data because of 2025 ({only_2025.sum()}):")
    for _, r in rb_bt[only_2025].sort_values('pick').iterrows():
        print(f"    {r['player_name']:<25} {int(r['draft_year'])} pick {int(r['pick']):>3} "
              f"first_3yr_ppg={r['first_3yr_ppg']:.1f} prod={r['prod_raw_filled']:.1f}")

# ============================================================================
# TEST 1C: Both changes reverted (8-game min + no 2025)
# ============================================================================
print(f"\n{'='*90}")
print("TEST 1C: 8-GAME MIN + NO 2025 DATA (both reverted)")
print("=" * 90)

rb_bt['ppg_8gm_no25'] = rb_bt.apply(
    lambda r: compute_first_3yr_ppg(r['player_id'], int(r['draft_year']), min_games=8, max_season=2024), axis=1)

has_both_rev = rb_bt['ppg_8gm_no25'].notna()
print(f"  N={has_both_rev.sum()}")

if has_both_rev.sum() > 10:
    r_dc_br, _ = pearsonr(rb_bt.loc[has_both_rev, 's_dc'], rb_bt.loc[has_both_rev, 'ppg_8gm_no25'])
    r_slap_br, _ = pearsonr(rb_bt.loc[has_both_rev, 'slap_raw_65_35'], rb_bt.loc[has_both_rev, 'ppg_8gm_no25'])
    print(f"  DC-only  Pearson: {r_dc_br:.4f}")
    print(f"  SLAP     Pearson: {r_slap_br:.4f}")
    print(f"  Gap: {r_slap_br-r_dc_br:+.4f}")

# ============================================================================
# TEST 2: WHO ARE THE NEW PLAYERS?
# ============================================================================
print(f"\n\n{'='*90}")
print("TEST 2: WHO ARE THE NEW PLAYERS?")
print("=" * 90)

# "Old" = had best_ppg in original backtest. "New" = has first_3yr_ppg in rebuilt outcomes.
had_old_metric = rb_bt['best_ppg'].notna() & (rb_bt['best_ppg'] > 0)
has_new_metric = rb_bt['first_3yr_ppg'].notna()

new_players = has_new_metric & ~had_old_metric
lost_players = had_old_metric & ~has_new_metric

print(f"  Had old best_ppg: {had_old_metric.sum()}")
print(f"  Have new first_3yr_ppg: {has_new_metric.sum()}")
print(f"  NEW players (gained): {new_players.sum()}")
print(f"  LOST players (lost): {lost_players.sum()}")

if new_players.sum() > 0:
    print(f"\n  NEW PLAYERS ADDED ({new_players.sum()}):")
    new_df = rb_bt[new_players].sort_values('pick')

    # For each new player, check if their SLAP rank is consistent with PPG rank
    # among all players with first_3yr_ppg
    all_with_ppg = rb_bt[has_new_metric].copy()
    all_with_ppg['slap_rank'] = all_with_ppg['slap_raw_65_35'].rank(ascending=False, method='min')
    all_with_ppg['ppg_rank'] = all_with_ppg['first_3yr_ppg'].rank(ascending=False, method='min')
    all_with_ppg['rank_diff'] = all_with_ppg['slap_rank'] - all_with_ppg['ppg_rank']

    print(f"  {'Name':<25} {'Year':>4} {'Pick':>4} {'SLAP':>6} {'Prod':>5} {'PPG':>6} {'SlapRk':>7} {'PPGRk':>6} {'RkDiff':>7} {'Impact'}")
    print(f"  {'-'*95}")
    for _, r in new_df.iterrows():
        match = all_with_ppg[(all_with_ppg['player_name'] == r['player_name']) &
                              (all_with_ppg['draft_year'] == r['draft_year'])]
        if not match.empty:
            m = match.iloc[0]
            impact = "HELPS" if abs(m['rank_diff']) < 15 else ("HURTS" if abs(m['rank_diff']) > 30 else "NEUTRAL")
            print(f"  {r['player_name']:<25} {int(r['draft_year']):>4} {int(r['pick']):>4} "
                  f"{r['slap_raw_65_35']:>6.1f} {r['prod_raw_filled']:>5.1f} {r['first_3yr_ppg']:>6.1f} "
                  f"{int(m['slap_rank']):>7} {int(m['ppg_rank']):>6} {int(m['rank_diff']):>+7} {impact}")

if lost_players.sum() > 0:
    print(f"\n  LOST PLAYERS ({lost_players.sum()}):")
    for _, r in rb_bt[lost_players].sort_values('pick').iterrows():
        print(f"    {r['player_name']:<25} {int(r['draft_year'])} pick {int(r['pick']):>3} best_ppg={r['best_ppg']:.1f}")

# ============================================================================
# TEST 3: DID PPG VALUES CHANGE FOR EXISTING PLAYERS?
# ============================================================================
print(f"\n\n{'='*90}")
print("TEST 3: DID PPG VALUES CHANGE FOR EXISTING PLAYERS?")
print("=" * 90)

both_metrics = had_old_metric & has_new_metric
print(f"  Players with both old best_ppg AND new first_3yr_ppg: {both_metrics.sum()}")

# Compare the two metrics
comp = rb_bt[both_metrics][['player_name', 'draft_year', 'pick', 'best_ppg', 'first_3yr_ppg',
                             's_dc', 'prod_raw_filled', 'slap_raw_65_35']].copy()
comp['ppg_diff'] = comp['first_3yr_ppg'] - comp['best_ppg']

# How correlated are the two metrics?
r_metrics, _ = pearsonr(comp['best_ppg'], comp['first_3yr_ppg'])
print(f"\n  Correlation between old best_ppg and new first_3yr_ppg: r={r_metrics:.4f}")

print(f"\n  Summary of differences (first_3yr_ppg - best_ppg):")
print(f"    Mean diff: {comp['ppg_diff'].mean():+.2f}")
print(f"    Std diff:  {comp['ppg_diff'].std():.2f}")
print(f"    Min diff:  {comp['ppg_diff'].min():+.2f}")
print(f"    Max diff:  {comp['ppg_diff'].max():+.2f}")
print(f"    Median:    {comp['ppg_diff'].median():+.2f}")

# How many changed significantly?
big_change = comp[comp['ppg_diff'].abs() > 2]
print(f"    Changed by >2 PPG: {len(big_change)} ({len(big_change)/len(comp)*100:.0f}%)")
big_change5 = comp[comp['ppg_diff'].abs() > 5]
print(f"    Changed by >5 PPG: {len(big_change5)} ({len(big_change5)/len(comp)*100:.0f}%)")

print(f"\n  TOP 15 BIGGEST CHANGES (first_3yr_ppg - best_ppg):")
print(f"  {'Name':<25} {'Year':>4} {'Pick':>4} {'OldPPG':>7} {'NewPPG':>7} {'Diff':>7} {'DC':>5} {'Prod':>5}")
print(f"  {'-'*75}")
for _, r in comp.nlargest(15, 'ppg_diff').iterrows():
    print(f"  {r['player_name']:<25} {int(r['draft_year']):>4} {int(r['pick']):>4} "
          f"{r['best_ppg']:>7.1f} {r['first_3yr_ppg']:>7.1f} {r['ppg_diff']:>+7.1f} "
          f"{r['s_dc']:>5.1f} {r['prod_raw_filled']:>5.1f}")

print(f"\n  BOTTOM 15 BIGGEST CHANGES:")
print(f"  {'Name':<25} {'Year':>4} {'Pick':>4} {'OldPPG':>7} {'NewPPG':>7} {'Diff':>7} {'DC':>5} {'Prod':>5}")
print(f"  {'-'*75}")
for _, r in comp.nsmallest(15, 'ppg_diff').iterrows():
    print(f"  {r['player_name']:<25} {int(r['draft_year']):>4} {int(r['pick']):>4} "
          f"{r['best_ppg']:>7.1f} {r['first_3yr_ppg']:>7.1f} {r['ppg_diff']:>+7.1f} "
          f"{r['s_dc']:>5.1f} {r['prod_raw_filled']:>5.1f}")

# ============================================================================
# EXPLAIN THE DEFINITIONS
# ============================================================================
print(f"\n\n{'='*90}")
print("METRIC DEFINITIONS (THE KEY ISSUE)")
print("=" * 90)
print(f"""
  OLD metric (best_ppg):
    - From original backtest file: best_ppr / 17
    - best_ppr = best single SEASON total PPR points (career, any season)
    - Divided by 17 (fixed denominator = full-season assumption)
    - NO game minimum — counts even 1-game seasons
    - Covers ENTIRE career (not just first 3 years)

  NEW metric (first_3yr_ppg):
    - From rebuild_outcomes.py
    - Best single-season PPG in first 3 NFL seasons
    - PPG = total_ppr / actual_games_played (not /17)
    - 6-game minimum per season
    - ONLY first 3 years (draft year through draft year + 2)

  KEY DIFFERENCES:
    1. Denominator: /17 fixed vs /actual_games (6+ game min)
    2. Time window: career-best vs first-3-years-best
    3. Game minimum: none vs 6 games
    4. Fundamentally different: a player with 200 PPR in 10 games
       gets 200/17=11.8 old vs 200/10=20.0 new
""")

# ============================================================================
# DEFINITIVE TEST: Correlations with EACH metric, same sample
# ============================================================================
print(f"{'='*90}")
print("DEFINITIVE COMPARISON: Same players, different metrics")
print("=" * 90)

# Also recompute old-style best_ppg from weekly data (to verify)
rb_bt['recomputed_best_ppg'] = rb_bt.apply(
    lambda r: compute_old_best_ppg(r['player_id'], min_games=1, max_season=2025), axis=1)

# Compare original best_ppg vs recomputed
orig_ppg_mask = had_old_metric & rb_bt['recomputed_best_ppg'].notna()
if orig_ppg_mask.sum() > 10:
    r_verify = pearsonr(rb_bt.loc[orig_ppg_mask, 'best_ppg'], rb_bt.loc[orig_ppg_mask, 'recomputed_best_ppg'])[0]
    print(f"  Verification: original best_ppg vs recomputed best_ppr/17: r={r_verify:.4f}")
    diff = rb_bt.loc[orig_ppg_mask, 'recomputed_best_ppg'] - rb_bt.loc[orig_ppg_mask, 'best_ppg']
    print(f"  Mean diff: {diff.mean():.2f}, Median: {diff.median():.2f} (recomputed - original)")

# Now compare all metric variants on the SAME set of players
common = both_metrics & rb_bt['recomputed_best_ppg'].notna()
n_common = common.sum()
print(f"\n  Common players for comparison: {n_common}")

metrics_to_test = {
    'best_ppg (old)': rb_bt.loc[common, 'best_ppg'],
    'recomputed best_ppr/17': rb_bt.loc[common, 'recomputed_best_ppg'],
    'first_3yr_ppg (new, 6gm)': rb_bt.loc[common, 'first_3yr_ppg'],
}

# Add 8gm version for common players
rb_bt['ppg_3yr_8gm_common'] = rb_bt.apply(
    lambda r: compute_first_3yr_ppg(r['player_id'], int(r['draft_year']), min_games=8), axis=1)
if rb_bt.loc[common, 'ppg_3yr_8gm_common'].notna().sum() > 10:
    metrics_to_test['first_3yr_ppg (8gm)'] = rb_bt.loc[common, 'ppg_3yr_8gm_common']

# Career best PPG (new style, 6gm)
rb_bt['career_best_ppg_6gm'] = rb_bt.apply(
    lambda r: compute_first_3yr_ppg(r['player_id'], int(r['draft_year']), min_games=6, max_season=2025)
    if pd.isna(r['player_id']) else (
        season_stats[(season_stats['player_id'] == r['player_id']) & (season_stats['games'] >= 6)]['ppg'].max()
        if not season_stats[(season_stats['player_id'] == r['player_id']) & (season_stats['games'] >= 6)].empty
        else np.nan
    ), axis=1)
if rb_bt.loc[common, 'career_best_ppg_6gm'].notna().sum() > 10:
    metrics_to_test['career_best_ppg (6gm)'] = rb_bt.loc[common, 'career_best_ppg_6gm']

print(f"\n  {'Metric':<30} {'DC Pearson':>11} {'SLAP Pearson':>13} {'Gap':>8} {'SLAP wins?':>11}")
print(f"  {'-'*80}")

dc_vals = rb_bt.loc[common, 's_dc']
slap_vals = rb_bt.loc[common, 'slap_raw_65_35']

for label, metric_vals in metrics_to_test.items():
    valid = metric_vals.notna()
    if valid.sum() < 10:
        continue
    r_dc_m = pearsonr(dc_vals[valid], metric_vals[valid])[0]
    r_slap_m = pearsonr(slap_vals[valid], metric_vals[valid])[0]
    gap = r_slap_m - r_dc_m
    wins = "YES" if gap > 0 else "no"
    print(f"  {label:<30} {r_dc_m:>11.4f} {r_slap_m:>13.4f} {gap:>+8.4f} {wins:>11}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print(f"\n\n{'='*90}")
print("FULL SUMMARY TABLE")
print("=" * 90)

# Compile all results
tests = []

# Current 6gm
m = rb_bt['first_3yr_ppg'].notna()
r_dc_cur = pearsonr(rb_bt.loc[m, 's_dc'], rb_bt.loc[m, 'first_3yr_ppg'])[0]
r_slap_cur = pearsonr(rb_bt.loc[m, 'slap_raw_65_35'], rb_bt.loc[m, 'first_3yr_ppg'])[0]
tests.append(('Current (6gm, with 2025, first_3yr)', m.sum(), r_dc_cur, r_slap_cur))

# 8gm
m8 = rb_bt['ppg_8gm'].notna()
if m8.sum() > 10:
    r_dc_8 = pearsonr(rb_bt.loc[m8, 's_dc'], rb_bt.loc[m8, 'ppg_8gm'])[0]
    r_slap_8 = pearsonr(rb_bt.loc[m8, 'slap_raw_65_35'], rb_bt.loc[m8, 'ppg_8gm'])[0]
    tests.append(('Revert: 8-game min (with 2025, first_3yr)', m8.sum(), r_dc_8, r_slap_8))

# No 2025
mn25 = rb_bt['ppg_no2025'].notna()
if mn25.sum() > 10:
    r_dc_n25 = pearsonr(rb_bt.loc[mn25, 's_dc'], rb_bt.loc[mn25, 'ppg_no2025'])[0]
    r_slap_n25 = pearsonr(rb_bt.loc[mn25, 'slap_raw_65_35'], rb_bt.loc[mn25, 'ppg_no2025'])[0]
    tests.append(('Revert: No 2025 data (6gm, first_3yr)', mn25.sum(), r_dc_n25, r_slap_n25))

# Both reverted
mbr = rb_bt['ppg_8gm_no25'].notna()
if mbr.sum() > 10:
    r_dc_br = pearsonr(rb_bt.loc[mbr, 's_dc'], rb_bt.loc[mbr, 'ppg_8gm_no25'])[0]
    r_slap_br = pearsonr(rb_bt.loc[mbr, 'slap_raw_65_35'], rb_bt.loc[mbr, 'ppg_8gm_no25'])[0]
    tests.append(('Revert: 8gm + no 2025 (first_3yr)', mbr.sum(), r_dc_br, r_slap_br))

# Old metric (best_ppg from backtest file)
m_old = has_old
tests.append(('OLD METRIC: best_ppr/17 (from backtest)', m_old.sum(), r_dc_old, r_slap_old))

print(f"\n  {'Config':<45} {'N':>4} {'DC r':>8} {'SLAP r':>8} {'Gap':>8}")
print(f"  {'-'*80}")
for label, n, r_dc, r_slap in tests:
    gap = r_slap - r_dc
    print(f"  {label:<45} {n:>4} {r_dc:>8.4f} {r_slap:>8.4f} {gap:>+8.4f}")

print(f"\n  DONE.")
