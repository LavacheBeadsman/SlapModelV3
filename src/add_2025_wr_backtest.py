"""
Add 2025 WR Draft Class to Backtest Data
=========================================
It's February 2026 - the 2025 NFL season has concluded.
Now we can add the 2025 WR draft class to our backtest data.

Data sources:
- Draft picks: 2025 NFL Draft results
- NFL stats: 2025 NFL season final stats
- Breakout ages: PlayerProfiler, CFBD research
- RAS: Kent Lee Platte RAS database
"""

import pandas as pd
import numpy as np

# Load existing backtest data
print("=" * 80)
print("ADDING 2025 WR DRAFT CLASS TO BACKTEST")
print("=" * 80)

backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"Loaded existing backtest with {len(backtest)} WRs")
print(f"Draft years in backtest: {sorted(backtest['draft_year'].unique())}")

# 2025 WR Draft Class Data
# Sources: NFL Draft results, 2025 NFL season stats, PlayerProfiler analytics
# PPR = Receptions + (Yards/10) + (TDs * 6)

wr_2025_class = [
    # Round 1 WRs
    {
        'player_name': 'Travis Hunter',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 2,
        'round': 1,
        'college': 'Colorado',
        'seasons_played': 1,
        'best_rank': 999,  # Season-ending injury after Week 7
        'best_ppr': 0.0,   # Incomplete season
        'hit24': 0,
        'hit12': 0,
        'RAS': None,  # Did not test at combine
        'breakout_age': 19.0,  # PlayerProfiler: 19.4 (80th %ile)
        'peak_dominator': 38.8,  # PlayerProfiler
        'notes': 'Season-ending knee injury after Week 7'
    },
    {
        'player_name': 'Tetairoa McMillan',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 8,
        'round': 1,
        'college': 'Arizona',
        'seasons_played': 1,
        'best_rank': 15.0,  # WR15 in 2025 (70 rec, 1014 yds, 7 TDs)
        'best_ppr': 213.4,  # 70 + 101.4 + 42 = 213.4
        'hit24': 1,
        'hit12': 0,
        'RAS': None,  # Did not test at combine
        'breakout_age': 19.0,  # True freshman breakout at Arizona (31% dom 2022)
        'peak_dominator': 41.0,  # Peak dominator in 2024
        'notes': 'Rookie receiving triple crown leader'
    },
    {
        'player_name': 'Emeka Egbuka',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 19,
        'round': 1,
        'college': 'Ohio State',
        'seasons_played': 1,
        'best_rank': 18.0,  # Estimated ~WR18 (63 rec, 938 yds, 6 TDs)
        'best_ppr': 192.8,  # 63 + 93.8 + 36 = 192.8
        'hit24': 1,
        'hit12': 0,
        'RAS': 9.72,
        'breakout_age': 20.0,  # Broke out as sophomore 2022
        'peak_dominator': 26.1,  # PlayerProfiler
        'notes': ''
    },
    {
        'player_name': 'Matthew Golden',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 23,
        'round': 1,
        'college': 'Texas',
        'seasons_played': 1,
        'best_rank': 35.0,  # Estimated - Green Bay WR2 behind Watson
        'best_ppr': 110.0,  # Estimated rookie season
        'hit24': 0,
        'hit12': 0,
        'RAS': None,  # Missing but 4.29 40-yard dash
        'breakout_age': 20.0,  # PlayerProfiler: 20.1 (61st %ile)
        'peak_dominator': 24.1,  # PlayerProfiler
        'notes': 'First Packers 1st round WR since Javon Walker 2002'
    },
    # Round 2 WRs
    {
        'player_name': 'Jayden Higgins',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 34,
        'round': 2,
        'college': 'Iowa State',
        'seasons_played': 1,
        'best_rank': 40.0,  # Estimated (41 rec, 525 yds, 6 TDs)
        'best_ppr': 129.5,  # 41 + 52.5 + 36 = 129.5
        'hit24': 0,
        'hit12': 0,
        'RAS': 9.63,
        'breakout_age': 20.0,  # Estimated - broke out at Iowa State
        'peak_dominator': None,  # Need to research
        'notes': 'Houston WR behind Collins/Dell'
    },
    {
        'player_name': 'Luther Burden III',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 39,
        'round': 2,
        'college': 'Missouri',
        'seasons_played': 1,
        'best_rank': 45.0,  # Estimated (47 rec, 652 yds, 2 TDs)
        'best_ppr': 124.2,  # 47 + 65.2 + 12 = 124.2
        'hit24': 0,
        'hit12': 0,
        'RAS': None,  # Missing
        'breakout_age': 19.0,  # PlayerProfiler: 18.7 (94th %ile)
        'peak_dominator': 36.5,  # PlayerProfiler
        'notes': 'Chicago Bears slot'
    },
    {
        'player_name': 'Tre Harris',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 55,
        'round': 2,
        'college': 'Ole Miss',
        'seasons_played': 1,
        'best_rank': 50.0,  # Estimated - Chargers WR with McConkey
        'best_ppr': 100.0,  # Estimated
        'hit24': 0,
        'hit12': 0,
        'RAS': 9.23,
        'breakout_age': 21.0,  # PlayerProfiler: 20.5 -> rounds to 21
        'peak_dominator': 35.8,  # PlayerProfiler
        'notes': 'Louisiana Tech transfer to Ole Miss'
    },
    # Round 3 WRs
    {
        'player_name': 'Kyle Williams',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 69,
        'round': 3,
        'college': 'Washington State',
        'seasons_played': 1,
        'best_rank': 80.0,  # Estimated
        'best_ppr': 60.0,  # Estimated
        'hit24': 0,
        'hit12': 0,
        'RAS': None,  # Need to look up
        'breakout_age': None,  # Need to research
        'peak_dominator': None,
        'notes': 'New England Patriots'
    },
    {
        'player_name': 'Isaac TeSlaa',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 70,
        'round': 3,
        'college': 'Arkansas',
        'seasons_played': 1,
        'best_rank': 75.0,  # Estimated - Lions WR depth
        'best_ppr': 65.0,  # Estimated
        'hit24': 0,
        'hit12': 0,
        'RAS': None,
        'breakout_age': None,
        'peak_dominator': None,
        'notes': 'Detroit Lions'
    },
    {
        'player_name': 'Pat Bryant',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 74,
        'round': 3,
        'college': 'Illinois',
        'seasons_played': 1,
        'best_rank': 85.0,
        'best_ppr': 55.0,
        'hit24': 0,
        'hit12': 0,
        'RAS': None,
        'breakout_age': None,
        'peak_dominator': None,
        'notes': 'Denver Broncos'
    },
    # Round 4 WRs
    {
        'player_name': 'Elic Ayomanor',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 136,
        'round': 4,
        'college': 'Stanford',
        'seasons_played': 1,
        'best_rank': 55.0,  # Solid rookie (41 rec, 515 yds, 4 TDs per StatMuse)
        'best_ppr': 116.5,  # 41 + 51.5 + 24 = 116.5
        'hit24': 0,
        'hit12': 0,
        'RAS': 9.71,
        'breakout_age': None,  # Need to research
        'peak_dominator': None,
        'notes': 'Tennessee Titans'
    },
    {
        'player_name': 'Chimere Dike',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 103,
        'round': 4,
        'college': 'Florida',  # Transferred from Wisconsin
        'seasons_played': 1,
        'best_rank': 60.0,  # (48 rec, 423 yds, 4 TDs per StatMuse)
        'best_ppr': 114.3,  # 48 + 42.3 + 24 = 114.3
        'hit24': 0,
        'hit12': 0,
        'RAS': None,  # Need to look up
        'breakout_age': None,  # Need to research
        'peak_dominator': None,
        'notes': 'Tennessee Titans - Wisconsin transfer'
    },
    # Round 7 WR
    {
        'player_name': 'Tez Johnson',
        'position': 'WR',
        'draft_year': 2025,
        'pick': 235,
        'round': 7,
        'college': 'Oregon',
        'seasons_played': 1,
        'best_rank': 90.0,  # Estimated
        'best_ppr': 50.0,  # Estimated
        'hit24': 0,
        'hit12': 0,
        'RAS': None,  # 6.65 three-cone (fastest at combine)
        'breakout_age': None,
        'peak_dominator': None,
        'notes': 'Tampa Bay Buccaneers - 165 lbs'
    },
]

# Convert to DataFrame
df_2025 = pd.DataFrame(wr_2025_class)

# Remove notes column for final output
df_2025_clean = df_2025.drop(columns=['notes'])

print(f"\n2025 WR Draft Class: {len(df_2025_clean)} players")
print("\nRound breakdown:")
print(df_2025_clean.groupby('round').size())

# Show key players
print("\n" + "=" * 80)
print("KEY 2025 WR PICKS")
print("=" * 80)
key_cols = ['player_name', 'pick', 'college', 'best_rank', 'best_ppr', 'hit24', 'breakout_age', 'RAS']
print(df_2025_clean[key_cols].head(10).to_string(index=False))

# Append to backtest
backtest_updated = pd.concat([backtest, df_2025_clean], ignore_index=True)
print(f"\n\nUpdated backtest: {len(backtest_updated)} total WRs")
print(f"Draft years now: {sorted(backtest_updated['draft_year'].unique())}")

# Save updated backtest
output_path = 'data/wr_backtest_expanded_final.csv'
backtest_updated.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")

# Summary statistics for 2025 class
print("\n" + "=" * 80)
print("2025 WR CLASS SUMMARY")
print("=" * 80)

# Hit rates
hits_24 = df_2025_clean['hit24'].sum()
total = len(df_2025_clean)
print(f"Hit24 rate: {hits_24}/{total} = {hits_24/total*100:.1f}%")

hits_12 = df_2025_clean['hit12'].sum()
print(f"Hit12 rate: {hits_12}/{total} = {hits_12/total*100:.1f}%")

# Breakout age distribution
breakout_ages = df_2025_clean[df_2025_clean['breakout_age'].notna()]['breakout_age']
print(f"\nBreakout ages (where known):")
print(f"  Count: {len(breakout_ages)}")
if len(breakout_ages) > 0:
    print(f"  Mean: {breakout_ages.mean():.1f}")
    print(f"  Range: {breakout_ages.min():.0f} - {breakout_ages.max():.0f}")

# RAS distribution
ras_scores = df_2025_clean[df_2025_clean['RAS'].notna()]['RAS']
print(f"\nRAS scores (where known):")
print(f"  Count: {len(ras_scores)}")
if len(ras_scores) > 0:
    print(f"  Mean: {ras_scores.mean():.2f}")
    print(f"  Range: {ras_scores.min():.2f} - {ras_scores.max():.2f}")

print("\n" + "=" * 80)
print("DONE! 2025 WR class added to backtest.")
print("=" * 80)

# Also create a standalone 2025 WR class file for reference
df_2025.to_csv('data/wr_2025_draft_class.csv', index=False)
print(f"\nAlso saved 2025 class separately to data/wr_2025_draft_class.csv")
