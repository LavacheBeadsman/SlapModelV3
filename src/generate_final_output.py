"""
SLAP Score V3 - Final Clean Output Generation (CALIBRATED)
===========================================================

Implements Option B: Percentile Rank Normalization

This ensures WR breakout scores and RB production scores are on the same scale:
- "90" = top 10% of position
- "50" = median for position
- Allows meaningful cross-position comparisons

Output Files:
1. output/slap_complete_database_final.csv - Complete database (all players 2015-2026)
2. output/slap_wr_2026_final.csv - Top 50 2026 WRs
3. output/slap_rb_2026_final.csv - Top 50 2026 RBs
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("SLAP SCORE V3 - CALIBRATED OUTPUT (Option B: Percentile Ranks)")
print("=" * 90)

# ============================================================================
# WEIGHTS (Position-Specific)
# ============================================================================
WR_WEIGHT_DC = 0.65
WR_WEIGHT_PRODUCTION = 0.20
WR_WEIGHT_RAS = 0.15

RB_WEIGHT_DC = 0.50
RB_WEIGHT_PRODUCTION = 0.35
RB_WEIGHT_RAS = 0.15

# Position averages for RAS imputation
WR_AVG_RAS = 68.9
RB_AVG_RAS = 66.5

# ============================================================================
# FORMULAS
# ============================================================================

def normalize_draft_capital(pick):
    """DC = 100 - 2.40 × (pick^0.62 - 1)"""
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))

def wr_breakout_score_raw(breakout_age, dominator_pct):
    """RAW breakout scoring (before percentile conversion)."""
    if breakout_age is None or pd.isna(breakout_age):
        if dominator_pct is not None and pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        else:
            return 25

    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20, 25: 20}
    base_score = age_tiers.get(int(breakout_age), 20)

    if dominator_pct is not None and pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
        bonus = max(0, bonus)
    else:
        bonus = 0

    return min(base_score + bonus, 99.9)

def rb_production_score_raw(rec_yards, team_pass_att, draft_age):
    """RAW RB receiving production score (before percentile conversion)."""
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return None
    if pd.isna(draft_age):
        draft_age = 22

    ratio = rec_yards / team_pass_att
    season_age = draft_age - 1
    age_weight = 1.15 - (0.05 * (season_age - 19))
    age_weight = max(0.85, min(1.15, age_weight))
    raw_score = ratio * age_weight * 100
    scaled_score = raw_score / 1.75
    return min(99.9, max(0, scaled_score))

def get_slap_tier(score):
    """Convert SLAP score to tier label."""
    if score >= 90:
        return "Elite"
    elif score >= 80:
        return "Great"
    elif score >= 70:
        return "Good"
    elif score >= 60:
        return "Average"
    elif score >= 50:
        return "Below Avg"
    else:
        return "Poor"

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

# WR backtest data
wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
wr_dominator = pd.read_csv('data/wr_dominator_complete.csv')
wr_backtest = wr_backtest.merge(
    wr_dominator[['player_name', 'draft_year', 'dominator_pct']],
    on=['player_name', 'draft_year'],
    how='left'
)

# RB backtest data
rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')

# 2026 prospects
prospects = pd.read_csv('data/prospects_final.csv')
wr_2026 = prospects[prospects['position'] == 'WR'].copy()
rb_2026 = prospects[prospects['position'] == 'RB'].copy()

# 2026 WR breakout data (hardcoded from research)
WR_2026_BREAKOUT = {
    'Carnell Tate': 19, 'Jordyn Tyson': 20, 'Makai Lemon': 20, 'Denzel Boston': 20,
    'Kevin Concepcion': 19, 'Chris Bell': 21, 'Elijah Sarratt': 21, 'Zachariah Branch': 19,
    'Germie Bernard': 20, 'Chris Brazzell': 21, "Ja'Kobi Lane": 20, 'Omar Cooper Jr.': 21,
    'Antonio Williams': 21, 'Skyler Bell': 21, 'Malachi Fields': 21, 'C.J. Daniels': 20,
    'Brenen Thompson': 21, 'Deion Burks': 20, 'Ted Hurst': 21, 'Bryce Lance': 21,
    'Kevin Coleman Jr.': 20, 'Eric McAlister': 21, 'Eric Rivers': 21, 'Lewis Bond': 21,
    "De'Zhaun Stribling": 20, 'Keelan Marion': 20, 'Josh Cameron': 21, 'Noah Thomas': 19,
    'Aaron Anderson': 20, 'Dane Key': 20, 'Jordan Hudson': 20, 'Caleb Douglas': 20,
    'Reggie Virgil': 20, 'Vinny Anthony II': 21, 'Caullin Lacy': 21, 'Kendrick Law': 21,
    'Colbie Young': 21, 'Harrison Wallace III': 20, 'Jaden Greathouse': 19, 'Barion Brown': 19,
    'Amare Thomas': 18, 'Hykeem Williams': 20, 'Shelton Sampson Jr.': 19,
    'Hank Beatty': 21, 'Eli Heidenreich': 21, 'Emmanuel Henderson': 22, 'Zavion Thomas': 18,
    'Anthony Evans III': 20, 'Rara Thomas': 23, "Ja'Varrius Johnson": 21,
}

# Load dominator data from CSV if available
try:
    wr_2026_breakout = pd.read_csv('data/wr_breakout_ages_2026.csv')
    WR_2026_DOMINATOR = dict(zip(wr_2026_breakout['player_name'], wr_2026_breakout['peak_dominator']))
except:
    WR_2026_DOMINATOR = {}

# 2026 RB production data
try:
    rb_2026_existing = pd.read_csv('output/slap_2026_rb.csv')
except:
    rb_2026_existing = None

print(f"  WR backtest: {len(wr_backtest)} players")
print(f"  RB backtest: {len(rb_backtest)} players")
print(f"  2026 WR prospects: {len(wr_2026)} players")
print(f"  2026 RB prospects: {len(rb_2026)} players")

# ============================================================================
# STEP 1: Calculate RAW production scores for all players
# ============================================================================
print("\nStep 1: Calculating RAW production scores...")

# WR backtest - raw breakout scores
wr_backtest['dc_score'] = wr_backtest['pick'].apply(normalize_draft_capital)
wr_backtest['breakout_score_raw'] = wr_backtest.apply(
    lambda x: wr_breakout_score_raw(x['breakout_age'], x['dominator_pct']), axis=1
)
wr_backtest['ras_raw'] = wr_backtest['RAS']
wr_backtest['ras_score'] = wr_backtest['RAS'].fillna(WR_AVG_RAS / 10) * 10

# RB backtest - raw production scores
rb_backtest['dc_score'] = rb_backtest['pick'].apply(normalize_draft_capital)
rb_backtest['production_score_raw'] = rb_backtest.apply(
    lambda x: rb_production_score_raw(x['rec_yards'], x['team_pass_att'], x['age']), axis=1
)
# Fill missing with median of non-missing
rb_median_prod = rb_backtest['production_score_raw'].median()
rb_backtest['production_score_raw'] = rb_backtest['production_score_raw'].fillna(rb_median_prod)
rb_backtest['ras_raw'] = rb_backtest['RAS']
rb_backtest['ras_score'] = rb_backtest['RAS'].fillna(RB_AVG_RAS / 10) * 10

# 2026 WRs - raw breakout scores
wr_2026['breakout_age'] = wr_2026['player_name'].map(WR_2026_BREAKOUT)
wr_2026['dominator_pct'] = wr_2026['player_name'].map(WR_2026_DOMINATOR)
wr_2026['dc_score'] = wr_2026['projected_pick'].apply(normalize_draft_capital)
wr_2026['breakout_score_raw'] = wr_2026.apply(
    lambda x: wr_breakout_score_raw(x['breakout_age'], x['dominator_pct']), axis=1
)
wr_2026['ras_raw'] = None
wr_2026['ras_score'] = WR_AVG_RAS

# 2026 RBs - raw production scores
if rb_2026_existing is not None:
    rb_2026 = rb_2026.merge(
        rb_2026_existing[['player_name', 'production_score']],
        on='player_name',
        how='left'
    )
    rb_2026['production_score_raw'] = rb_2026['production_score'].fillna(rb_median_prod)
else:
    rb_2026['production_score_raw'] = rb_median_prod

rb_2026['dc_score'] = rb_2026['projected_pick'].apply(normalize_draft_capital)
rb_2026['ras_raw'] = None
rb_2026['ras_score'] = RB_AVG_RAS

# ============================================================================
# STEP 2: Convert to PERCENTILE RANKS within position
# ============================================================================
print("Step 2: Converting to percentile ranks...")

# Combine all WR raw scores
all_wr_raw = pd.concat([
    wr_backtest[['player_name', 'breakout_score_raw']],
    wr_2026[['player_name', 'breakout_score_raw']]
], ignore_index=True)

# Combine all RB raw scores
all_rb_raw = pd.concat([
    rb_backtest[['player_name', 'production_score_raw']],
    rb_2026[['player_name', 'production_score_raw']]
], ignore_index=True)

# Calculate percentile ranks (0-100 scale)
# rank(pct=True) gives 0-1, multiply by 100 for 0-100 scale
all_wr_raw['breakout_percentile'] = all_wr_raw['breakout_score_raw'].rank(pct=True) * 100
all_rb_raw['production_percentile'] = all_rb_raw['production_score_raw'].rank(pct=True) * 100

# Create lookup dictionaries
wr_percentile_lookup = dict(zip(all_wr_raw['player_name'], all_wr_raw['breakout_percentile']))
rb_percentile_lookup = dict(zip(all_rb_raw['player_name'], all_rb_raw['production_percentile']))

# Apply percentile ranks back to dataframes
wr_backtest['breakout_percentile'] = wr_backtest['player_name'].map(wr_percentile_lookup)
wr_2026['breakout_percentile'] = wr_2026['player_name'].map(wr_percentile_lookup)
rb_backtest['production_percentile'] = rb_backtest['player_name'].map(rb_percentile_lookup)
rb_2026['production_percentile'] = rb_2026['player_name'].map(rb_percentile_lookup)

print(f"  WR breakout: raw mean={all_wr_raw['breakout_score_raw'].mean():.1f} → percentile mean={all_wr_raw['breakout_percentile'].mean():.1f}")
print(f"  RB production: raw mean={all_rb_raw['production_score_raw'].mean():.1f} → percentile mean={all_rb_raw['production_percentile'].mean():.1f}")

# ============================================================================
# STEP 3: Calculate SLAP scores using PERCENTILE ranks
# ============================================================================
print("Step 3: Calculating calibrated SLAP scores...")

# WR SLAP using percentile ranks
wr_backtest['slap_score'] = (
    WR_WEIGHT_DC * wr_backtest['dc_score'] +
    WR_WEIGHT_PRODUCTION * wr_backtest['breakout_percentile'] +
    WR_WEIGHT_RAS * wr_backtest['ras_score']
)
wr_backtest['delta'] = wr_backtest['slap_score'] - wr_backtest['dc_score']
wr_backtest['slap_tier'] = wr_backtest['slap_score'].apply(get_slap_tier)

wr_2026['slap_score'] = (
    WR_WEIGHT_DC * wr_2026['dc_score'] +
    WR_WEIGHT_PRODUCTION * wr_2026['breakout_percentile'] +
    WR_WEIGHT_RAS * wr_2026['ras_score']
)
wr_2026['delta'] = wr_2026['slap_score'] - wr_2026['dc_score']
wr_2026['slap_tier'] = wr_2026['slap_score'].apply(get_slap_tier)

# RB SLAP using percentile ranks
rb_backtest['slap_score'] = (
    RB_WEIGHT_DC * rb_backtest['dc_score'] +
    RB_WEIGHT_PRODUCTION * rb_backtest['production_percentile'] +
    RB_WEIGHT_RAS * rb_backtest['ras_score']
)
rb_backtest['delta'] = rb_backtest['slap_score'] - rb_backtest['dc_score']
rb_backtest['slap_tier'] = rb_backtest['slap_score'].apply(get_slap_tier)

rb_2026['slap_score'] = (
    RB_WEIGHT_DC * rb_2026['dc_score'] +
    RB_WEIGHT_PRODUCTION * rb_2026['production_percentile'] +
    RB_WEIGHT_RAS * rb_2026['ras_score']
)
rb_2026['delta'] = rb_2026['slap_score'] - rb_2026['dc_score']
rb_2026['slap_tier'] = rb_2026['slap_score'].apply(get_slap_tier)

# ============================================================================
# CREATE STANDARDIZED OUTPUT
# ============================================================================
print("\nCreating standardized output...")

wr_backtest_out = pd.DataFrame({
    'player_name': wr_backtest['player_name'],
    'position': 'WR',
    'draft_year': wr_backtest['draft_year'],
    'pick': wr_backtest['pick'],
    'college': wr_backtest['college'],
    'draft_age': wr_backtest.get('age', None),
    'breakout_age': wr_backtest['breakout_age'],
    'dominator_pct': wr_backtest['dominator_pct'],
    'breakout_score_raw': wr_backtest['breakout_score_raw'],
    'production_percentile': wr_backtest['breakout_percentile'],
    'rec_yards': None,
    'team_pass_att': None,
    'production_score_raw': None,
    'ras_raw': wr_backtest['ras_raw'],
    'ras_score': wr_backtest['ras_score'],
    'dc_score': wr_backtest['dc_score'],
    'slap_score': wr_backtest['slap_score'],
    'slap_tier': wr_backtest['slap_tier'],
    'delta': wr_backtest['delta'],
    'nfl_best_ppg': wr_backtest['best_ppr'],
    'hit24': wr_backtest['hit24'],
})

wr_2026_out = pd.DataFrame({
    'player_name': wr_2026['player_name'],
    'position': 'WR',
    'draft_year': 2026,
    'pick': wr_2026['projected_pick'],
    'college': wr_2026['school'],
    'draft_age': wr_2026['age'],
    'breakout_age': wr_2026['breakout_age'],
    'dominator_pct': wr_2026['dominator_pct'],
    'breakout_score_raw': wr_2026['breakout_score_raw'],
    'production_percentile': wr_2026['breakout_percentile'],
    'rec_yards': None,
    'team_pass_att': None,
    'production_score_raw': None,
    'ras_raw': wr_2026['ras_raw'],
    'ras_score': wr_2026['ras_score'],
    'dc_score': wr_2026['dc_score'],
    'slap_score': wr_2026['slap_score'],
    'slap_tier': wr_2026['slap_tier'],
    'delta': wr_2026['delta'],
    'nfl_best_ppg': None,
    'hit24': None,
})

rb_backtest_out = pd.DataFrame({
    'player_name': rb_backtest['player_name'],
    'position': 'RB',
    'draft_year': rb_backtest['draft_year'],
    'pick': rb_backtest['pick'],
    'college': rb_backtest['college'],
    'draft_age': rb_backtest['age'],
    'breakout_age': None,
    'dominator_pct': None,
    'breakout_score_raw': None,
    'production_percentile': rb_backtest['production_percentile'],
    'rec_yards': rb_backtest['rec_yards'],
    'team_pass_att': rb_backtest['team_pass_att'],
    'production_score_raw': rb_backtest['production_score_raw'],
    'ras_raw': rb_backtest['ras_raw'],
    'ras_score': rb_backtest['ras_score'],
    'dc_score': rb_backtest['dc_score'],
    'slap_score': rb_backtest['slap_score'],
    'slap_tier': rb_backtest['slap_tier'],
    'delta': rb_backtest['delta'],
    'nfl_best_ppg': rb_backtest['best_ppg'],
    'hit24': rb_backtest['hit24'],
})

rb_2026_out = pd.DataFrame({
    'player_name': rb_2026['player_name'],
    'position': 'RB',
    'draft_year': 2026,
    'pick': rb_2026['projected_pick'],
    'college': rb_2026['school'],
    'draft_age': rb_2026['age'],
    'breakout_age': None,
    'dominator_pct': None,
    'breakout_score_raw': None,
    'production_percentile': rb_2026['production_percentile'],
    'rec_yards': None,
    'team_pass_att': None,
    'production_score_raw': rb_2026['production_score_raw'],
    'ras_raw': rb_2026['ras_raw'],
    'ras_score': rb_2026['ras_score'],
    'dc_score': rb_2026['dc_score'],
    'slap_score': rb_2026['slap_score'],
    'slap_tier': rb_2026['slap_tier'],
    'delta': rb_2026['delta'],
    'nfl_best_ppg': None,
    'hit24': None,
})

# Combine all data
all_wr = pd.concat([wr_backtest_out, wr_2026_out], ignore_index=True)
all_rb = pd.concat([rb_backtest_out, rb_2026_out], ignore_index=True)
all_players = pd.concat([all_wr, all_rb], ignore_index=True)
all_players = all_players.sort_values(['draft_year', 'slap_score'], ascending=[True, False])

# ============================================================================
# SAVE OUTPUT FILES
# ============================================================================
print("\n" + "=" * 90)
print("SAVING OUTPUT FILES")
print("=" * 90)

all_players.to_csv('output/slap_complete_database_final.csv', index=False)
print(f"\n1. Saved: output/slap_complete_database_final.csv")
print(f"   Total players: {len(all_players)}")
print(f"   - WRs: {len(all_wr)} ({len(wr_backtest_out)} backtest + {len(wr_2026_out)} 2026)")
print(f"   - RBs: {len(all_rb)} ({len(rb_backtest_out)} backtest + {len(rb_2026_out)} 2026)")

wr_2026_final = wr_2026_out.sort_values('slap_score', ascending=False).head(50).reset_index(drop=True)
wr_2026_final.insert(0, 'rank', range(1, len(wr_2026_final) + 1))
wr_2026_final.to_csv('output/slap_wr_2026_final.csv', index=False)
print(f"\n2. Saved: output/slap_wr_2026_final.csv (Top 50)")

rb_2026_final = rb_2026_out.sort_values('slap_score', ascending=False).head(50).reset_index(drop=True)
rb_2026_final.insert(0, 'rank', range(1, len(rb_2026_final) + 1))
rb_2026_final.to_csv('output/slap_rb_2026_final.csv', index=False)
print(f"\n3. Saved: output/slap_rb_2026_final.csv (Top 50)")

# ============================================================================
# CALIBRATION VERIFICATION
# ============================================================================
print("\n" + "=" * 90)
print("CALIBRATION VERIFICATION")
print("=" * 90)

print("\n### BEFORE vs AFTER Calibration")
print("-" * 60)
print(f"\nProduction component (now using percentile ranks):")
print(f"  WR percentile mean: {all_wr['production_percentile'].mean():.1f}")
print(f"  RB percentile mean: {all_rb['production_percentile'].mean():.1f}")
print(f"  Gap: {all_wr['production_percentile'].mean() - all_rb['production_percentile'].mean():.1f} (target: ~0)")

print(f"\nFinal SLAP scores:")
print(f"  WR mean: {all_wr['slap_score'].mean():.1f}")
print(f"  RB mean: {all_rb['slap_score'].mean():.1f}")
print(f"  Gap: {all_wr['slap_score'].mean() - all_rb['slap_score'].mean():.1f} (was 12.9, target: ~2-3)")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 90)
print("SUMMARY STATISTICS")
print("=" * 90)

print("\n### SCORE DISTRIBUTION (CALIBRATED)")
print("-" * 60)
print(f"\n{'Position':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
print("-" * 50)
for pos in ['WR', 'RB']:
    pos_data = all_players[all_players['position'] == pos]['slap_score']
    print(f"{pos:<8} {pos_data.mean():>8.1f} {pos_data.std():>8.1f} {pos_data.min():>8.1f} {pos_data.max():>8.1f} {pos_data.median():>8.1f}")

print("\n### TIER DISTRIBUTION")
print("-" * 60)
tier_order = ['Elite', 'Great', 'Good', 'Average', 'Below Avg', 'Poor']
for pos in ['WR', 'RB']:
    print(f"\n{pos}s:")
    pos_data = all_players[all_players['position'] == pos]
    for tier in tier_order:
        count = (pos_data['slap_tier'] == tier).sum()
        pct = count / len(pos_data) * 100
        print(f"  {tier:<12}: {count:>4} ({pct:>5.1f}%)")

# ============================================================================
# TOP 25 RANKINGS
# ============================================================================
print("\n" + "=" * 90)
print("TOP 25 WRs ALL-TIME BY SLAP SCORE (2015-2026)")
print("=" * 90)

all_wr_ranked = all_wr.sort_values('slap_score', ascending=False).reset_index(drop=True)
print(f"\n{'Rank':>4} {'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'Tier':<10} {'Delta':>7}")
print("-" * 70)
for i, row in all_wr_ranked.head(25).iterrows():
    year = int(row['draft_year'])
    pick = int(row['pick'])
    print(f"{i+1:>4} {row['player_name']:<25} {year:>5} {pick:>5} {row['slap_score']:>6.1f} {row['slap_tier']:<10} {row['delta']:>+7.1f}")

print("\n" + "=" * 90)
print("TOP 25 RBs ALL-TIME BY SLAP SCORE (2015-2026)")
print("=" * 90)

all_rb_ranked = all_rb.sort_values('slap_score', ascending=False).reset_index(drop=True)
print(f"\n{'Rank':>4} {'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'Tier':<10} {'Delta':>7}")
print("-" * 70)
for i, row in all_rb_ranked.head(25).iterrows():
    year = int(row['draft_year'])
    pick = int(row['pick'])
    print(f"{i+1:>4} {row['player_name']:<25} {year:>5} {pick:>5} {row['slap_score']:>6.1f} {row['slap_tier']:<10} {row['delta']:>+7.1f}")

# ============================================================================
# 2026 RANKINGS
# ============================================================================
print("\n" + "=" * 90)
print("TOP 20 WRs FOR 2026 DRAFT")
print("=" * 90)

print(f"\n{'Rank':>4} {'Player':<25} {'School':<20} {'Pick':>5} {'SLAP':>6} {'Tier':<10} {'BO Age':>7}")
print("-" * 85)
for i, row in wr_2026_final.head(20).iterrows():
    bo_age = f"{int(row['breakout_age'])}" if pd.notna(row['breakout_age']) else "N/A"
    print(f"{row['rank']:>4} {row['player_name']:<25} {row['college']:<20} {int(row['pick']):>5} {row['slap_score']:>6.1f} {row['slap_tier']:<10} {bo_age:>7}")

print("\n" + "=" * 90)
print("TOP 20 RBs FOR 2026 DRAFT")
print("=" * 90)

print(f"\n{'Rank':>4} {'Player':<25} {'School':<20} {'Pick':>5} {'SLAP':>6} {'Tier':<10} {'Prod%':>7}")
print("-" * 85)
for i, row in rb_2026_final.head(20).iterrows():
    prod = f"{row['production_percentile']:.0f}" if pd.notna(row['production_percentile']) else "N/A"
    print(f"{row['rank']:>4} {row['player_name']:<25} {row['college']:<20} {int(row['pick']):>5} {row['slap_score']:>6.1f} {row['slap_tier']:<10} {prod:>7}")

print("\n" + "=" * 90)
print("CALIBRATED OUTPUT GENERATION COMPLETE")
print("=" * 90)
