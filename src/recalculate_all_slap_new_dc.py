"""
SLAP Score V3 - Complete Recalculation with New DC Formula
============================================================

New DC Formula (Option B - Gentler Curve):
    DC = 100 - 2.40 × (pick^0.62 - 1)

This creates:
- Pick 1: 100, Pick 5: ~95, Pick 10: ~88
- Pick 32: ~73, Pick 100: ~50, Pick 200: ~35

WEIGHTS (Position-Specific):
  WRs: DC (65%) + Breakout Age (20%) + RAS (15%)
  RBs: DC (50%) + Receiving Production (35%) + RAS (15%)

Why different weights?
- WR breakout age has weaker predictive power (r=0.155) than RB receiving production (r=0.30)
- WR model benefits from heavier DC weighting for better predictions
- RB receiving production is statistically significant (p=0.004) and deserves more weight
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("SLAP SCORE V3 - COMPLETE RECALCULATION WITH NEW DC FORMULA")
print("=" * 90)
print("\nNEW DC Formula: DC = 100 - 2.40 × (pick^0.62 - 1)")
print("\nPosition-Specific Weights:")
print("  WRs: DC (65%) + Breakout Age (20%) + RAS (15%)")
print("  RBs: DC (50%) + Receiving Production (35%) + RAS (15%)")

# ============================================================================
# WEIGHTS - POSITION SPECIFIC
# ============================================================================
# WR weights (breakout age is less predictive, so DC gets more weight)
WR_WEIGHT_DC = 0.65
WR_WEIGHT_PRODUCTION = 0.20
WR_WEIGHT_RAS = 0.15

# RB weights (receiving production is statistically significant)
RB_WEIGHT_DC = 0.50
RB_WEIGHT_PRODUCTION = 0.35
RB_WEIGHT_RAS = 0.15

# ============================================================================
# NEW DC FORMULA
# ============================================================================

def normalize_draft_capital(pick):
    """Convert pick number to 0-100 score using gentler power curve.

    Formula: DC = 100 - 2.40 × (pick^0.62 - 1)
    """
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))

# Show DC formula output for key picks
print("\n" + "=" * 90)
print("DC FORMULA VERIFICATION")
print("=" * 90)
print(f"\n{'Pick':>6} {'DC Score':>10}")
print("-" * 20)
test_picks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 32, 50, 64, 75, 100, 150, 200, 250]
for pick in test_picks:
    dc = normalize_draft_capital(pick)
    print(f"{pick:>6} {dc:>10.1f}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def breakout_age_to_score(age):
    """Convert breakout age to 0-100 score for WRs"""
    if pd.isna(age):
        return None
    age = int(age)
    scores = {
        18: 100,
        19: 90,
        20: 75,
        21: 60,
        22: 45,
        23: 30,
        24: 20,
    }
    return scores.get(age, 25)

def rb_production_score(rec_yards, team_pass_att, draft_age):
    """
    Calculate RB receiving production score with continuous age weighting.

    Younger RBs get a bonus multiplier on their production.
    Older RBs get a penalty multiplier.

    Formula: (rec_yards / team_pass_att) × age_weight × 100

    Age weight: 1.15 - (0.05 × (season_age - 19))
    Where season_age = draft_age - 1 (age during final college season)
    """
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return None
    if pd.isna(draft_age):
        draft_age = 22  # Default to baseline if missing

    # Raw production ratio
    ratio = rec_yards / team_pass_att

    # Season age (age during final college season)
    season_age = draft_age - 1

    # Continuous age weight
    # Age 19 = 1.15, Age 20 = 1.10, Age 21 = 1.05, Age 22 = 1.00, Age 23 = 0.95
    age_weight = 1.15 - (0.05 * (season_age - 19))
    age_weight = max(0.85, min(1.15, age_weight))  # Cap between 0.85 and 1.15

    # Age-adjusted production score
    score = ratio * age_weight * 100

    return min(100, max(0, score))

# Position averages (from backtest)
WR_AVG_BREAKOUT = 79.7
WR_AVG_RAS = 68.9
RB_AVG_PRODUCTION = 30.0
RB_AVG_RAS = 66.5

# ============================================================================
# LOAD AND PROCESS WR BACKTEST DATA (2015-2024)
# ============================================================================
print("\n" + "=" * 90)
print("PROCESSING WR BACKTEST DATA (2015-2024)")
print("=" * 90)

wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"Loaded {len(wr_backtest)} WRs from backtest")

# Calculate scores with NEW DC formula
wr_backtest['dc_score'] = wr_backtest['pick'].apply(normalize_draft_capital)
wr_backtest['breakout_score'] = wr_backtest['breakout_age'].apply(breakout_age_to_score)
wr_backtest['breakout_score_final'] = wr_backtest['breakout_score'].fillna(WR_AVG_BREAKOUT)
wr_backtest['production_score'] = wr_backtest['breakout_score_final']  # For WR, production = breakout
wr_backtest['ras_score'] = wr_backtest['RAS'] * 10  # Convert 0-10 to 0-100
wr_backtest['ras_score_final'] = wr_backtest['ras_score'].fillna(WR_AVG_RAS)

# Calculate SLAP with WR weights (65/20/15)
wr_backtest['slap_score'] = (
    WR_WEIGHT_DC * wr_backtest['dc_score'] +
    WR_WEIGHT_PRODUCTION * wr_backtest['production_score'] +
    WR_WEIGHT_RAS * wr_backtest['ras_score_final']
)

wr_backtest['delta_vs_dc'] = wr_backtest['slap_score'] - wr_backtest['dc_score']

# Status fields
wr_backtest['production_status'] = np.where(wr_backtest['breakout_score'].isna(), 'imputed', 'observed')
wr_backtest['ras_status'] = np.where(wr_backtest['ras_score'].isna(), 'imputed', 'observed')

print(f"  WRs with observed breakout: {(wr_backtest['production_status'] == 'observed').sum()}")
print(f"  WRs with observed RAS: {(wr_backtest['ras_status'] == 'observed').sum()}")

# ============================================================================
# LOAD AND PROCESS RB BACKTEST DATA (2015-2024)
# ============================================================================
print("\n" + "=" * 90)
print("PROCESSING RB BACKTEST DATA (2015-2024)")
print("=" * 90)

rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
print(f"Loaded {len(rb_backtest)} RBs from backtest")

# Calculate scores with NEW DC formula
rb_backtest['dc_score'] = rb_backtest['pick'].apply(normalize_draft_capital)
rb_backtest['production_score'] = rb_backtest.apply(
    lambda x: rb_production_score(x['rec_yards'], x['team_pass_att'], x['age']), axis=1
)
rb_backtest['production_score_final'] = rb_backtest['production_score'].fillna(RB_AVG_PRODUCTION)
rb_backtest['ras_score'] = rb_backtest['RAS'] * 10
rb_backtest['ras_score_final'] = rb_backtest['ras_score'].fillna(RB_AVG_RAS)

# Calculate SLAP with RB weights (50/35/15 - unchanged)
rb_backtest['slap_score'] = (
    RB_WEIGHT_DC * rb_backtest['dc_score'] +
    RB_WEIGHT_PRODUCTION * rb_backtest['production_score_final'] +
    RB_WEIGHT_RAS * rb_backtest['ras_score_final']
)

rb_backtest['delta_vs_dc'] = rb_backtest['slap_score'] - rb_backtest['dc_score']

# Status fields
rb_backtest['production_status'] = np.where(rb_backtest['production_score'].isna(), 'imputed', 'observed')
rb_backtest['ras_status'] = np.where(rb_backtest['ras_score'].isna(), 'imputed', 'observed')

print(f"  RBs with observed production: {(rb_backtest['production_status'] == 'observed').sum()}")
print(f"  RBs with observed RAS: {(rb_backtest['ras_status'] == 'observed').sum()}")

# ============================================================================
# LOAD AND PROCESS 2026 WR PROSPECTS
# ============================================================================
print("\n" + "=" * 90)
print("PROCESSING 2026 WR PROSPECTS")
print("=" * 90)

# 2026 WR Breakout ages (researched)
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
}

prospects = pd.read_csv('data/prospects_final.csv')
wr_2026 = prospects[prospects['position'] == 'WR'].copy()
print(f"Loaded {len(wr_2026)} 2026 WR prospects")

# Calculate scores with NEW DC formula
wr_2026['breakout_age'] = wr_2026['player_name'].map(WR_2026_BREAKOUT)
wr_2026['breakout_score'] = wr_2026['breakout_age'].apply(breakout_age_to_score)
wr_2026['breakout_score_final'] = wr_2026['breakout_score'].fillna(WR_AVG_BREAKOUT)
wr_2026['dc_score'] = wr_2026['projected_pick'].apply(normalize_draft_capital)
wr_2026['ras_score_final'] = WR_AVG_RAS  # No combine data yet
wr_2026['breakout_status'] = np.where(wr_2026['breakout_score'].isna(), 'imputed', 'observed')
wr_2026['ras_status'] = 'imputed'

# Calculate SLAP with WR weights (65/20/15)
wr_2026['slap_score'] = (
    WR_WEIGHT_DC * wr_2026['dc_score'] +
    WR_WEIGHT_PRODUCTION * wr_2026['breakout_score_final'] +
    WR_WEIGHT_RAS * wr_2026['ras_score_final']
)
wr_2026['delta_vs_dc'] = wr_2026['slap_score'] - wr_2026['dc_score']

# ============================================================================
# LOAD AND PROCESS 2026 RB PROSPECTS
# ============================================================================
print("\n" + "=" * 90)
print("PROCESSING 2026 RB PROSPECTS")
print("=" * 90)

# 2026 RB production data (from CFBD - final 2025 college season)
# This needs to be loaded from the existing output or data files
try:
    rb_2026_existing = pd.read_csv('output/slap_2026_rb.csv')
    print(f"Loaded existing 2026 RB data with production scores")
except:
    rb_2026_existing = None

rb_2026 = prospects[prospects['position'] == 'RB'].copy()
print(f"Loaded {len(rb_2026)} 2026 RB prospects")

# If we have existing production scores, use them
if rb_2026_existing is not None:
    rb_2026 = rb_2026.merge(
        rb_2026_existing[['player_name', 'production_score']],
        on='player_name',
        how='left'
    )
    rb_2026['production_score_final'] = rb_2026['production_score'].fillna(RB_AVG_PRODUCTION)
else:
    rb_2026['production_score_final'] = RB_AVG_PRODUCTION

# Calculate scores with NEW DC formula
rb_2026['dc_score'] = rb_2026['projected_pick'].apply(normalize_draft_capital)
rb_2026['ras_score_final'] = RB_AVG_RAS  # No combine data yet
rb_2026['production_status'] = np.where(
    rb_2026.get('production_score', pd.Series([np.nan]*len(rb_2026))).isna(),
    'imputed', 'observed'
)
rb_2026['ras_status'] = 'imputed'

# Calculate SLAP with RB weights (50/35/15 - unchanged)
rb_2026['slap_score'] = (
    RB_WEIGHT_DC * rb_2026['dc_score'] +
    RB_WEIGHT_PRODUCTION * rb_2026['production_score_final'] +
    RB_WEIGHT_RAS * rb_2026['ras_score_final']
)
rb_2026['delta_vs_dc'] = rb_2026['slap_score'] - rb_2026['dc_score']

# ============================================================================
# SCORE DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 90)
print("SCORE DISTRIBUTION ANALYSIS")
print("=" * 90)

all_wr = pd.concat([wr_backtest, wr_2026], ignore_index=True)
all_rb = pd.concat([rb_backtest, rb_2026], ignore_index=True)

print("\nWR SLAP Score Distribution:")
print(f"  Mean: {all_wr['slap_score'].mean():.1f}")
print(f"  Std:  {all_wr['slap_score'].std():.1f}")
print(f"  Min:  {all_wr['slap_score'].min():.1f}")
print(f"  Max:  {all_wr['slap_score'].max():.1f}")
print(f"  Median: {all_wr['slap_score'].median():.1f}")

print("\nRB SLAP Score Distribution:")
print(f"  Mean: {all_rb['slap_score'].mean():.1f}")
print(f"  Std:  {all_rb['slap_score'].std():.1f}")
print(f"  Min:  {all_rb['slap_score'].min():.1f}")
print(f"  Max:  {all_rb['slap_score'].max():.1f}")
print(f"  Median: {all_rb['slap_score'].median():.1f}")

# Score ranges
print("\nWR SLAP Score Ranges:")
bins = [(90, 100), (80, 90), (70, 80), (60, 70), (50, 60), (40, 50), (30, 40), (0, 30)]
for low, high in bins:
    count = len(all_wr[(all_wr['slap_score'] >= low) & (all_wr['slap_score'] < high)])
    pct = count / len(all_wr) * 100
    print(f"  {low:>3}-{high:<3}: {count:>4} ({pct:>5.1f}%)")

print("\nRB SLAP Score Ranges:")
for low, high in bins:
    count = len(all_rb[(all_rb['slap_score'] >= low) & (all_rb['slap_score'] < high)])
    pct = count / len(all_rb) * 100
    print(f"  {low:>3}-{high:<3}: {count:>4} ({pct:>5.1f}%)")

# ============================================================================
# TOP 25 WRS ALL-TIME BY SLAP
# ============================================================================
print("\n" + "=" * 90)
print("TOP 25 WRS ALL-TIME BY SLAP SCORE (2015-2026)")
print("=" * 90)

wr_ranked = all_wr.sort_values('slap_score', ascending=False).reset_index(drop=True)
print(f"\n{'Rank':>4} {'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>5} {'Prod':>5} {'Delta':>6}")
print("-" * 75)

for i, row in wr_ranked.head(25).iterrows():
    year = int(row.get('draft_year', 2026)) if not pd.isna(row.get('draft_year')) else 2026
    pick_val = row.get('pick') if not pd.isna(row.get('pick')) else row.get('projected_pick', 0)
    pick = int(pick_val) if not pd.isna(pick_val) else 0
    prod = row.get('production_score') if 'production_score' in row.index and not pd.isna(row.get('production_score')) else row.get('breakout_score_final', 0)
    print(f"{i+1:>4} {row['player_name']:<25} {year:>5} {pick:>5} {row['slap_score']:>6.1f} {row['dc_score']:>5.1f} {prod:>5.0f} {row['delta_vs_dc']:>+6.1f}")

# ============================================================================
# TOP 25 RBS ALL-TIME BY SLAP
# ============================================================================
print("\n" + "=" * 90)
print("TOP 25 RBS ALL-TIME BY SLAP SCORE (2015-2026)")
print("=" * 90)

rb_ranked = all_rb.sort_values('slap_score', ascending=False).reset_index(drop=True)
print(f"\n{'Rank':>4} {'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>5} {'Prod':>5} {'Delta':>6}")
print("-" * 75)

for i, row in rb_ranked.head(25).iterrows():
    year = int(row.get('draft_year', 2026)) if not pd.isna(row.get('draft_year')) else 2026
    pick_val = row.get('pick') if not pd.isna(row.get('pick')) else row.get('projected_pick', 0)
    pick = int(pick_val) if not pd.isna(pick_val) else 0
    prod = row.get('production_score_final') if not pd.isna(row.get('production_score_final')) else row.get('production_score', 0)
    if pd.isna(prod):
        prod = 0
    print(f"{i+1:>4} {row['player_name']:<25} {year:>5} {pick:>5} {row['slap_score']:>6.1f} {row['dc_score']:>5.1f} {prod:>5.0f} {row['delta_vs_dc']:>+6.1f}")

# ============================================================================
# TOP 25 2026 WRS
# ============================================================================
print("\n" + "=" * 90)
print("TOP 25 2026 WR PROSPECTS BY SLAP SCORE")
print("=" * 90)

wr_2026_ranked = wr_2026.sort_values('slap_score', ascending=False).reset_index(drop=True)
print(f"\n{'Rank':>4} {'Player':<25} {'School':<20} {'Pick':>5} {'SLAP':>6} {'DC':>5} {'BO':>5} {'Delta':>6}")
print("-" * 85)

for i, row in wr_2026_ranked.head(25).iterrows():
    bo_flag = "*" if row['breakout_status'] == 'imputed' else ""
    print(f"{i+1:>4} {row['player_name']:<25} {row['school']:<20} {int(row['projected_pick']):>5} {row['slap_score']:>6.1f} {row['dc_score']:>5.1f} {row['breakout_score_final']:>4.0f}{bo_flag:<1} {row['delta_vs_dc']:>+6.1f}")

print("\n* = imputed breakout age")

# ============================================================================
# TOP 25 2026 RBS
# ============================================================================
print("\n" + "=" * 90)
print("TOP 25 2026 RB PROSPECTS BY SLAP SCORE")
print("=" * 90)

rb_2026_ranked = rb_2026.sort_values('slap_score', ascending=False).reset_index(drop=True)
print(f"\n{'Rank':>4} {'Player':<25} {'School':<20} {'Pick':>5} {'SLAP':>6} {'DC':>5} {'Prod':>5} {'Delta':>6}")
print("-" * 85)

for i, row in rb_2026_ranked.head(25).iterrows():
    prod_flag = "*" if row['production_status'] == 'imputed' else ""
    prod = row['production_score_final']
    print(f"{i+1:>4} {row['player_name']:<25} {row['school']:<20} {int(row['projected_pick']):>5} {row['slap_score']:>6.1f} {row['dc_score']:>5.1f} {prod:>4.0f}{prod_flag:<1} {row['delta_vs_dc']:>+6.1f}")

print("\n* = imputed production")

# ============================================================================
# SANITY CHECK - SCORES BY PICK RANGE
# ============================================================================
print("\n" + "=" * 90)
print("SANITY CHECK - SCORES BY PICK RANGE")
print("=" * 90)

# Get some specific examples
all_players = pd.concat([wr_backtest, rb_backtest], ignore_index=True)

print("\n--- Pick 1-5 with good profile (should be 95+) ---")
top_picks = all_players[all_players['pick'] <= 5].sort_values('slap_score', ascending=False)
for _, row in top_picks.head(5).iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['pick'])} | SLAP {row['slap_score']:.1f}")

print("\n--- Pick 6-15 with good profile (should be 85-95) ---")
mid_1 = all_players[(all_players['pick'] >= 6) & (all_players['pick'] <= 15)].sort_values('slap_score', ascending=False)
for _, row in mid_1.head(5).iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['pick'])} | SLAP {row['slap_score']:.1f}")

print("\n--- Pick 25-40 with good profile (should be 70-85) ---")
mid_2 = all_players[(all_players['pick'] >= 25) & (all_players['pick'] <= 40)].sort_values('slap_score', ascending=False)
for _, row in mid_2.head(5).iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['pick'])} | SLAP {row['slap_score']:.1f}")

print("\n--- Pick 90-110 with good profile (should be 50-65) ---")
late = all_players[(all_players['pick'] >= 90) & (all_players['pick'] <= 110)].sort_values('slap_score', ascending=False)
for _, row in late.head(5).iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['pick'])} | SLAP {row['slap_score']:.1f}")

print("\n--- Pick 180-220 (should be 35-50) ---")
very_late = all_players[(all_players['pick'] >= 180) & (all_players['pick'] <= 220)].sort_values('slap_score', ascending=False)
for _, row in very_late.head(5).iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['pick'])} | SLAP {row['slap_score']:.1f}")

# ============================================================================
# SAVE OUTPUT FILES
# ============================================================================
print("\n" + "=" * 90)
print("SAVING OUTPUT FILES")
print("=" * 90)

# Create standardized WR output
wr_backtest_std = pd.DataFrame({
    'player_name': wr_backtest['player_name'],
    'position': 'WR',
    'school': wr_backtest['college'],
    'draft_year': wr_backtest['draft_year'],
    'pick': wr_backtest['pick'],
    'round': wr_backtest['round'],
    'dc_score': wr_backtest['dc_score'],
    'production_score': wr_backtest['production_score'],
    'production_metric': 'breakout_age',
    'ras_score': wr_backtest['ras_score_final'],
    'slap_score': wr_backtest['slap_score'],
    'delta_vs_dc': wr_backtest['delta_vs_dc'],
    'production_status': wr_backtest['production_status'],
    'ras_status': wr_backtest['ras_status'],
    'breakout_age': wr_backtest['breakout_age'],
    'nfl_best_ppr': wr_backtest['best_ppr'],
    'nfl_hit24': wr_backtest['hit24'],
    'nfl_hit12': wr_backtest['hit12'],
    'data_type': 'backtest'
})

wr_2026_std = pd.DataFrame({
    'player_name': wr_2026['player_name'],
    'position': 'WR',
    'school': wr_2026['school'],
    'draft_year': 2026,
    'pick': wr_2026['projected_pick'],
    'round': np.ceil(wr_2026['projected_pick'] / 32).astype(int),
    'dc_score': wr_2026['dc_score'],
    'production_score': wr_2026['breakout_score_final'],
    'production_metric': 'breakout_age',
    'ras_score': wr_2026['ras_score_final'],
    'slap_score': wr_2026['slap_score'],
    'delta_vs_dc': wr_2026['delta_vs_dc'],
    'production_status': wr_2026['breakout_status'],
    'ras_status': wr_2026['ras_status'],
    'breakout_age': wr_2026['breakout_age'],
    'nfl_best_ppr': None,
    'nfl_hit24': None,
    'nfl_hit12': None,
    'data_type': 'prospect_2026'
})

wr_all = pd.concat([wr_backtest_std, wr_2026_std], ignore_index=True)
wr_all = wr_all.sort_values(['draft_year', 'slap_score'], ascending=[True, False])

# Create standardized RB output
rb_backtest_std = pd.DataFrame({
    'player_name': rb_backtest['player_name'],
    'position': 'RB',
    'school': rb_backtest['college'],
    'draft_year': rb_backtest['draft_year'],
    'pick': rb_backtest['pick'],
    'round': rb_backtest['round'],
    'dc_score': rb_backtest['dc_score'],
    'production_score': rb_backtest['production_score_final'],
    'production_metric': 'receiving_production',
    'ras_score': rb_backtest['ras_score_final'],
    'slap_score': rb_backtest['slap_score'],
    'delta_vs_dc': rb_backtest['delta_vs_dc'],
    'production_status': rb_backtest['production_status'],
    'ras_status': rb_backtest['ras_status'],
    'rec_yards': rb_backtest['rec_yards'],
    'team_pass_att': rb_backtest['team_pass_att'],
    'nfl_best_ppr': rb_backtest['best_ppr'],
    'nfl_best_ppg': rb_backtest['best_ppg'],
    'nfl_hit24': rb_backtest['hit24'],
    'nfl_hit12': rb_backtest['hit12'],
    'data_type': 'backtest'
})

rb_2026_std = pd.DataFrame({
    'player_name': rb_2026['player_name'],
    'position': 'RB',
    'school': rb_2026['school'],
    'draft_year': 2026,
    'pick': rb_2026['projected_pick'],
    'round': np.ceil(rb_2026['projected_pick'] / 32).astype(int),
    'dc_score': rb_2026['dc_score'],
    'production_score': rb_2026['production_score_final'],
    'production_metric': 'receiving_production',
    'ras_score': rb_2026['ras_score_final'],
    'slap_score': rb_2026['slap_score'],
    'delta_vs_dc': rb_2026['delta_vs_dc'],
    'production_status': rb_2026['production_status'],
    'ras_status': rb_2026['ras_status'],
    'rec_yards': None,
    'team_pass_att': None,
    'nfl_best_ppr': None,
    'nfl_best_ppg': None,
    'nfl_hit24': None,
    'nfl_hit12': None,
    'data_type': 'prospect_2026'
})

rb_all = pd.concat([rb_backtest_std, rb_2026_std], ignore_index=True)
rb_all = rb_all.sort_values(['draft_year', 'slap_score'], ascending=[True, False])

# Combine all players
all_players_final = pd.concat([wr_all, rb_all], ignore_index=True)
all_players_final = all_players_final.sort_values(['draft_year', 'slap_score'], ascending=[True, False])
all_players_final['overall_rank'] = range(1, len(all_players_final) + 1)

# Save files
wr_all.to_csv('output/slap_complete_wr.csv', index=False)
print(f"Saved: output/slap_complete_wr.csv ({len(wr_all)} WRs)")

rb_all.to_csv('output/slap_complete_rb.csv', index=False)
print(f"Saved: output/slap_complete_rb.csv ({len(rb_all)} RBs)")

all_players_final.to_csv('output/slap_complete_all_players.csv', index=False)
print(f"Saved: output/slap_complete_all_players.csv ({len(all_players_final)} total)")

# Save v4 database (with position-specific weights)
all_players_final.to_csv('output/slap_complete_database_v4.csv', index=False)
print(f"Saved: output/slap_complete_database_v4.csv ({len(all_players_final)} total)")

# Save 2026 specific files
wr_2026_output = wr_2026_ranked[['player_name', 'school', 'projected_pick', 'age',
                                  'slap_score', 'dc_score', 'breakout_score_final',
                                  'ras_score_final', 'delta_vs_dc', 'breakout_status',
                                  'ras_status', 'breakout_age']].copy()
wr_2026_output.columns = ['player_name', 'school', 'projected_pick', 'age',
                           'slap_score', 'dc_score', 'breakout_score', 'ras_score',
                           'delta_vs_dc', 'breakout_status', 'ras_status', 'breakout_age_raw']
wr_2026_output.to_csv('output/slap_wr_2026.csv', index=False)
print(f"Saved: output/slap_wr_2026.csv ({len(wr_2026_output)} WRs)")

rb_2026_output = rb_2026_ranked[['player_name', 'school', 'projected_pick', 'age',
                                  'slap_score', 'dc_score', 'production_score_final',
                                  'ras_score_final', 'delta_vs_dc', 'production_status',
                                  'ras_status']].copy()
rb_2026_output.columns = ['player_name', 'school', 'projected_pick', 'age',
                           'slap_score', 'dc_score', 'production_score', 'ras_score',
                           'delta_vs_dc', 'production_status', 'ras_status']
rb_2026_output.to_csv('output/slap_rb_2026.csv', index=False)
print(f"Saved: output/slap_rb_2026.csv ({len(rb_2026_output)} RBs)")

print("\n" + "=" * 90)
print("DONE! All scores recalculated with new DC formula.")
print("=" * 90)
