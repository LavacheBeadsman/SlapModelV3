"""
SLAP Score V3 - Master Database Builder
Recalculates ALL historical scores (2015-2024) with 50/35/15 weights
Combines with 2026 prospects for complete database

WEIGHTS: DC (50%) + Production (35%) + RAS (15%)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("SLAP SCORE V3 - MASTER DATABASE BUILDER")
print("Recalculating ALL historical scores with 50/35/15 weights")
print("=" * 90)

# ============================================================================
# WEIGHTS
# ============================================================================
WEIGHT_DC = 0.50
WEIGHT_PRODUCTION = 0.35
WEIGHT_RAS = 0.15

print(f"\nWeight Configuration:")
print(f"  Draft Capital: {WEIGHT_DC*100:.0f}%")
print(f"  Production:    {WEIGHT_PRODUCTION*100:.0f}%")
print(f"  RAS:           {WEIGHT_RAS*100:.0f}%")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_draft_capital(pick, max_pick=262):
    """Convert pick number to 0-100 score using gentler power curve.

    Formula: DC = 100 - 2.40 × (pick^0.62 - 1)

    This creates a gentler decay than 1/sqrt(pick), giving:
    - Pick 1: 100, Pick 5: ~95, Pick 10: ~88, Pick 32: ~73
    - Pick 100: ~50, Pick 200: ~35
    """
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))  # Clamp to 0-100

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

def rb_production_score(rec_yards, team_pass_att):
    """Calculate RB receiving production score"""
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return None
    ratio = rec_yards / team_pass_att
    # Normalize: typical range is 0-1.5, map to 0-100
    # Based on backtest: avg ratio ~0.5, max ~1.5
    score = (ratio / 1.0) * 100  # 1.0 ratio = 100 score
    return min(100, max(0, score))  # Cap at 0-100

# Position averages (from backtest)
WR_AVG_BREAKOUT = 79.7
WR_AVG_RAS = 68.9
RB_AVG_PRODUCTION = 30.0  # Conservative estimate
RB_AVG_RAS = 66.5

# ============================================================================
# LOAD WR BACKTEST DATA
# ============================================================================
print("\n" + "=" * 90)
print("LOADING WR BACKTEST DATA (2015-2024)")
print("=" * 90)

wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"Loaded {len(wr_backtest)} WRs from backtest")

# Calculate scores
wr_backtest['dc_score'] = wr_backtest['pick'].apply(normalize_draft_capital)
wr_backtest['breakout_score'] = wr_backtest['breakout_age'].apply(breakout_age_to_score)
wr_backtest['breakout_score_final'] = wr_backtest['breakout_score'].fillna(WR_AVG_BREAKOUT)
wr_backtest['production_score'] = wr_backtest['breakout_score_final']  # For WR, production = breakout
wr_backtest['ras_score'] = wr_backtest['RAS'] * 10  # Convert 0-10 to 0-100
wr_backtest['ras_score_final'] = wr_backtest['ras_score'].fillna(WR_AVG_RAS)

# Calculate SLAP with 50/35/15 weights
wr_backtest['slap_score'] = (
    WEIGHT_DC * wr_backtest['dc_score'] +
    WEIGHT_PRODUCTION * wr_backtest['production_score'] +
    WEIGHT_RAS * wr_backtest['ras_score_final']
)

wr_backtest['delta_vs_dc'] = wr_backtest['slap_score'] - wr_backtest['dc_score']

# Status fields
wr_backtest['production_status'] = np.where(wr_backtest['breakout_score'].isna(), 'imputed', 'observed')
wr_backtest['ras_status'] = np.where(wr_backtest['ras_score'].isna(), 'imputed', 'observed')

print(f"  WRs with observed breakout: {(wr_backtest['production_status'] == 'observed').sum()}")
print(f"  WRs with observed RAS: {(wr_backtest['ras_status'] == 'observed').sum()}")

# ============================================================================
# LOAD RB BACKTEST DATA
# ============================================================================
print("\n" + "=" * 90)
print("LOADING RB BACKTEST DATA (2015-2024)")
print("=" * 90)

rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
print(f"Loaded {len(rb_backtest)} RBs from backtest")

# Calculate scores
rb_backtest['dc_score'] = rb_backtest['pick'].apply(normalize_draft_capital)
rb_backtest['production_score'] = rb_backtest.apply(
    lambda x: rb_production_score(x['rec_yards'], x['team_pass_att']), axis=1
)
rb_backtest['production_score_final'] = rb_backtest['production_score'].fillna(RB_AVG_PRODUCTION)
rb_backtest['ras_score'] = rb_backtest['RAS'] * 10  # Convert 0-10 to 0-100
rb_backtest['ras_score_final'] = rb_backtest['ras_score'].fillna(RB_AVG_RAS)

# Calculate SLAP with 50/35/15 weights
rb_backtest['slap_score'] = (
    WEIGHT_DC * rb_backtest['dc_score'] +
    WEIGHT_PRODUCTION * rb_backtest['production_score_final'] +
    WEIGHT_RAS * rb_backtest['ras_score_final']
)

rb_backtest['delta_vs_dc'] = rb_backtest['slap_score'] - rb_backtest['dc_score']

# Status fields
rb_backtest['production_status'] = np.where(rb_backtest['production_score'].isna(), 'imputed', 'observed')
rb_backtest['ras_status'] = np.where(rb_backtest['ras_score'].isna(), 'imputed', 'observed')

print(f"  RBs with observed production: {(rb_backtest['production_status'] == 'observed').sum()}")
print(f"  RBs with observed RAS: {(rb_backtest['ras_status'] == 'observed').sum()}")

# ============================================================================
# LOAD 2026 PROSPECT DATA
# ============================================================================
print("\n" + "=" * 90)
print("LOADING 2026 PROSPECT DATA")
print("=" * 90)

wr_2026 = pd.read_csv('output/slap_2026_wr.csv')
rb_2026 = pd.read_csv('output/slap_2026_rb.csv')
print(f"Loaded {len(wr_2026)} 2026 WRs")
print(f"Loaded {len(rb_2026)} 2026 RBs")

# ============================================================================
# CREATE STANDARDIZED WR DATAFRAME
# ============================================================================
print("\n" + "=" * 90)
print("BUILDING STANDARDIZED WR DATABASE")
print("=" * 90)

# WR Backtest
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

# WR 2026
wr_2026_std = pd.DataFrame({
    'player_name': wr_2026['player_name'],
    'position': 'WR',
    'school': wr_2026['school'],
    'draft_year': 2026,
    'pick': wr_2026['projected_pick'],
    'round': np.ceil(wr_2026['projected_pick'] / 32).astype(int),
    'dc_score': wr_2026['dc_score'],
    'production_score': wr_2026['breakout_score'],
    'production_metric': 'breakout_age',
    'ras_score': wr_2026['ras_score'],
    'slap_score': wr_2026['slap_score'],
    'delta_vs_dc': wr_2026['delta_vs_dc'],
    'production_status': wr_2026['breakout_status'],
    'ras_status': wr_2026['ras_status'],
    'breakout_age': wr_2026.get('breakout_age_raw', None),
    'nfl_best_ppr': None,
    'nfl_hit24': None,
    'nfl_hit12': None,
    'data_type': 'prospect_2026'
})

# Combine WRs
wr_all = pd.concat([wr_backtest_std, wr_2026_std], ignore_index=True)
print(f"Total WRs: {len(wr_all)} ({len(wr_backtest_std)} backtest + {len(wr_2026_std)} prospects)")

# ============================================================================
# CREATE STANDARDIZED RB DATAFRAME
# ============================================================================
print("\n" + "=" * 90)
print("BUILDING STANDARDIZED RB DATABASE")
print("=" * 90)

# RB Backtest
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

# RB 2026
rb_2026_std = pd.DataFrame({
    'player_name': rb_2026['player_name'],
    'position': 'RB',
    'school': rb_2026['school'],
    'draft_year': 2026,
    'pick': rb_2026['projected_pick'],
    'round': np.ceil(rb_2026['projected_pick'] / 32).astype(int),
    'dc_score': rb_2026['dc_score'],
    'production_score': rb_2026['production_score'],
    'production_metric': 'receiving_production',
    'ras_score': rb_2026['ras_score'],
    'slap_score': rb_2026['slap_score'],
    'delta_vs_dc': rb_2026['delta_vs_dc'],
    'production_status': rb_2026['production_status'],
    'ras_status': rb_2026['ras_status'],
    'rec_yards': rb_2026.get('rec_yards', None),
    'team_pass_att': rb_2026.get('team_pass_att', None),
    'nfl_best_ppr': None,
    'nfl_best_ppg': None,
    'nfl_hit24': None,
    'nfl_hit12': None,
    'data_type': 'prospect_2026'
})

# Combine RBs
rb_all = pd.concat([rb_backtest_std, rb_2026_std], ignore_index=True)
print(f"Total RBs: {len(rb_all)} ({len(rb_backtest_std)} backtest + {len(rb_2026_std)} prospects)")

# ============================================================================
# COMBINE ALL PLAYERS
# ============================================================================
print("\n" + "=" * 90)
print("COMBINING ALL PLAYERS")
print("=" * 90)

all_players = pd.concat([wr_all, rb_all], ignore_index=True)
all_players = all_players.sort_values(['draft_year', 'slap_score'], ascending=[True, False])
all_players = all_players.reset_index(drop=True)

print(f"\nTOTAL PLAYERS: {len(all_players)}")
print(f"  WRs: {len(wr_all)}")
print(f"  RBs: {len(rb_all)}")
print(f"  Draft years: {sorted(all_players['draft_year'].unique())}")

# ============================================================================
# DISPLAY SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 90)
print("SLAP SCORE SUMMARY BY YEAR")
print("=" * 90)

for year in sorted(all_players['draft_year'].unique()):
    year_data = all_players[all_players['draft_year'] == year]
    wr_data = year_data[year_data['position'] == 'WR']
    rb_data = year_data[year_data['position'] == 'RB']

    print(f"\n{year}:")
    print(f"  WRs: {len(wr_data):>3} | Avg SLAP: {wr_data['slap_score'].mean():>5.1f} | Avg Delta: {wr_data['delta_vs_dc'].mean():>+5.1f}")
    print(f"  RBs: {len(rb_data):>3} | Avg SLAP: {rb_data['slap_score'].mean():>5.1f} | Avg Delta: {rb_data['delta_vs_dc'].mean():>+5.1f}")

# ============================================================================
# DISPLAY TOP HISTORICAL SLEEPERS/BUSTS
# ============================================================================
print("\n" + "=" * 90)
print("TOP HISTORICAL SLEEPERS (Model liked MORE than scouts)")
print("=" * 90)

backtest_only = all_players[all_players['data_type'] == 'backtest'].copy()

# WR Sleepers who hit
print("\nWR SLEEPERS WHO HIT (Delta > +10, hit24=1):")
print("-" * 80)
wr_sleepers_hit = backtest_only[(backtest_only['position'] == 'WR') &
                                (backtest_only['delta_vs_dc'] > 10) &
                                (backtest_only['nfl_hit24'] == 1)].sort_values('delta_vs_dc', ascending=False)
for _, row in wr_sleepers_hit.head(10).iterrows():
    print(f"  {row['player_name']:<25} ({row['draft_year']}) Pick {int(row['pick']):>3} | Delta: {row['delta_vs_dc']:>+6.1f} | PPR: {row['nfl_best_ppr']:.0f}")

# RB Sleepers who hit
print("\nRB SLEEPERS WHO HIT (Delta > +5, hit24=1):")
print("-" * 80)
rb_sleepers_hit = backtest_only[(backtest_only['position'] == 'RB') &
                                (backtest_only['delta_vs_dc'] > 5) &
                                (backtest_only['nfl_hit24'] == 1)].sort_values('delta_vs_dc', ascending=False)
for _, row in rb_sleepers_hit.head(10).iterrows():
    print(f"  {row['player_name']:<25} ({row['draft_year']}) Pick {int(row['pick']):>3} | Delta: {row['delta_vs_dc']:>+6.1f} | PPR: {row['nfl_best_ppr']:.0f}")

print("\n" + "=" * 90)
print("TOP HISTORICAL BUSTS (Model liked LESS than scouts)")
print("=" * 90)

# WR Fades who busted
print("\nWR FADES WHO BUSTED (Delta < -10, hit24=0):")
print("-" * 80)
wr_fades_bust = backtest_only[(backtest_only['position'] == 'WR') &
                              (backtest_only['delta_vs_dc'] < -10) &
                              (backtest_only['nfl_hit24'] == 0)].sort_values('delta_vs_dc', ascending=True)
for _, row in wr_fades_bust.head(10).iterrows():
    print(f"  {row['player_name']:<25} ({row['draft_year']}) Pick {int(row['pick']):>3} | Delta: {row['delta_vs_dc']:>+6.1f} | PPR: {row['nfl_best_ppr']:.0f}")

# RB Fades who busted
print("\nRB FADES WHO BUSTED (Delta < -5, hit24=0):")
print("-" * 80)
rb_fades_bust = backtest_only[(backtest_only['position'] == 'RB') &
                              (backtest_only['delta_vs_dc'] < -5) &
                              (backtest_only['nfl_hit24'] == 0)].sort_values('delta_vs_dc', ascending=True)
for _, row in rb_fades_bust.head(10).iterrows():
    print(f"  {row['player_name']:<25} ({row['draft_year']}) Pick {int(row['pick']):>3} | Delta: {row['delta_vs_dc']:>+6.1f} | PPR: {row['nfl_best_ppr']:.0f}")

# ============================================================================
# CALCULATE EDGE ACCURACY
# ============================================================================
print("\n" + "=" * 90)
print("EDGE-FINDING ACCURACY (50/35/15 WEIGHTS)")
print("=" * 90)

# WR Sleeper accuracy
wr_bt = backtest_only[backtest_only['position'] == 'WR']
wr_sleepers_all = wr_bt[wr_bt['delta_vs_dc'] > 10]
wr_sleeper_hit_rate = wr_sleepers_all['nfl_hit24'].mean() * 100 if len(wr_sleepers_all) > 0 else 0
print(f"\nWR Sleepers (delta > +10): {len(wr_sleepers_all)} players, {wr_sleeper_hit_rate:.1f}% hit rate")

wr_fades_all = wr_bt[wr_bt['delta_vs_dc'] < -10]
wr_fade_bust_rate = (1 - wr_fades_all['nfl_hit24'].mean()) * 100 if len(wr_fades_all) > 0 else 0
print(f"WR Fades (delta < -10): {len(wr_fades_all)} players, {wr_fade_bust_rate:.1f}% bust rate")

# RB Sleeper accuracy
rb_bt = backtest_only[backtest_only['position'] == 'RB']
rb_sleepers_all = rb_bt[rb_bt['delta_vs_dc'] > 5]
rb_sleeper_hit_rate = rb_sleepers_all['nfl_hit24'].mean() * 100 if len(rb_sleepers_all) > 0 else 0
print(f"\nRB Sleepers (delta > +5): {len(rb_sleepers_all)} players, {rb_sleeper_hit_rate:.1f}% hit rate")

rb_fades_all = rb_bt[rb_bt['delta_vs_dc'] < -5]
rb_fade_bust_rate = (1 - rb_fades_all['nfl_hit24'].mean()) * 100 if len(rb_fades_all) > 0 else 0
print(f"RB Fades (delta < -5): {len(rb_fades_all)} players, {rb_fade_bust_rate:.1f}% bust rate")

# ============================================================================
# SAVE OUTPUT FILES
# ============================================================================
print("\n" + "=" * 90)
print("SAVING OUTPUT FILES")
print("=" * 90)

# Save WR master
wr_all = wr_all.sort_values(['draft_year', 'slap_score'], ascending=[True, False])
wr_all.to_csv('output/slap_master_wr_50_35_15.csv', index=False)
print(f"Saved: output/slap_master_wr_50_35_15.csv ({len(wr_all)} WRs)")

# Save RB master
rb_all = rb_all.sort_values(['draft_year', 'slap_score'], ascending=[True, False])
rb_all.to_csv('output/slap_master_rb_50_35_15.csv', index=False)
print(f"Saved: output/slap_master_rb_50_35_15.csv ({len(rb_all)} RBs)")

# Save combined master
all_players = all_players.sort_values(['draft_year', 'slap_score'], ascending=[True, False])
all_players['overall_rank'] = range(1, len(all_players) + 1)
all_players.to_csv('output/slap_master_all_50_35_15.csv', index=False)
print(f"Saved: output/slap_master_all_50_35_15.csv ({len(all_players)} total)")

# Save backtest-only for analysis
backtest_only = backtest_only.sort_values(['draft_year', 'slap_score'], ascending=[True, False])
backtest_only.to_csv('output/slap_backtest_all_50_35_15.csv', index=False)
print(f"Saved: output/slap_backtest_all_50_35_15.csv ({len(backtest_only)} backtest players)")

print("\n" + "=" * 90)
print("DONE! Master database built with 50/35/15 weights.")
print("=" * 90)
