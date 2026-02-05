"""
Fetch NFL Fantasy Data (2015-2025)
==================================

Uses weekly data to calculate seasonal fantasy points and best PPG.
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FETCHING NFL FANTASY DATA (2015-2025)")
print("=" * 80)

# Fetch weekly data for all years
print("\nFetching weekly fantasy data (this may take a minute)...")
all_weekly = []

for year in range(2015, 2026):
    print(f"  {year}...", end=" ", flush=True)
    try:
        weekly = nfl.import_weekly_data(years=[year])
        weekly = weekly[weekly['position'].isin(['RB', 'WR'])].copy()
        all_weekly.append(weekly)
        print(f"{len(weekly)} records")
    except Exception as e:
        print(f"Error: {e}")

# Combine all years
weekly_df = pd.concat(all_weekly, ignore_index=True)
print(f"\nTotal weekly records: {len(weekly_df)}")

# Calculate seasonal totals
print("\nCalculating seasonal totals...")
seasonal = weekly_df.groupby(['player_id', 'player_name', 'player_display_name', 'position', 'season']).agg({
    'fantasy_points_ppr': 'sum',
    'week': 'count',  # Games played
    'receptions': 'sum',
    'receiving_yards': 'sum',
    'receiving_tds': 'sum',
    'rushing_yards': 'sum',
    'rushing_tds': 'sum'
}).reset_index()
seasonal.columns = ['player_id', 'player_name', 'player_display_name', 'position', 'season',
                    'fantasy_points_ppr', 'games', 'receptions', 'rec_yards', 'rec_tds',
                    'rush_yards', 'rush_tds']

# Calculate PPG
seasonal['ppg'] = seasonal['fantasy_points_ppr'] / seasonal['games']

# Get best season for each player
print("Finding best season for each player...")
best_seasons = seasonal.loc[seasonal.groupby('player_id')['fantasy_points_ppr'].idxmax()].copy()
best_seasons = best_seasons.rename(columns={
    'fantasy_points_ppr': 'best_ppr',
    'ppg': 'best_ppg',
    'season': 'best_season'
})

# Add hit thresholds
best_seasons['hit24'] = (best_seasons['best_ppg'] >= 24).astype(int)
best_seasons['hit12'] = (best_seasons['best_ppg'] >= 12).astype(int)

print(f"\nPlayers with fantasy data: {len(best_seasons)}")
print(f"  WRs: {len(best_seasons[best_seasons['position'] == 'WR'])}")
print(f"  RBs: {len(best_seasons[best_seasons['position'] == 'RB'])}")

# Fetch draft data
print("\nFetching draft data...")
draft = nfl.import_draft_picks(years=list(range(2015, 2026)))
draft = draft[draft['position'].isin(['RB', 'WR'])].copy()
print(f"  Drafted RB/WR: {len(draft)}")

# Match draft to fantasy
# Clean names for matching
draft['name_clean'] = draft['pfr_player_name'].str.lower().str.strip().str.replace('.', '', regex=False).str.replace("'", "", regex=False)
best_seasons['name_clean'] = best_seasons['player_display_name'].str.lower().str.strip().str.replace('.', '', regex=False).str.replace("'", "", regex=False)

# Merge on cleaned name
draft_fantasy = draft.merge(
    best_seasons[['name_clean', 'position', 'best_ppr', 'best_ppg', 'best_season', 'hit24', 'hit12', 'games']],
    on=['name_clean', 'position'],
    how='left',
    suffixes=('', '_fantasy')
)

# Fill missing with 0
draft_fantasy['best_ppr'] = draft_fantasy['best_ppr'].fillna(0)
draft_fantasy['best_ppg'] = draft_fantasy['best_ppg'].fillna(0)
draft_fantasy['hit24'] = draft_fantasy['hit24'].fillna(0).astype(int)
draft_fantasy['hit12'] = draft_fantasy['hit12'].fillna(0).astype(int)

# Summary by draft year
print("\n--- NFL FANTASY DATA BY DRAFT YEAR ---")
print(f"{'Year':<6} {'WRs':<6} {'RBs':<6} {'WR Hit24':<10} {'RB Hit24':<10} {'WR Matched':<12} {'RB Matched':<12}")
print("-" * 75)

for year in range(2015, 2026):
    wr_year = draft_fantasy[(draft_fantasy['season'] == year) & (draft_fantasy['position'] == 'WR')]
    rb_year = draft_fantasy[(draft_fantasy['season'] == year) & (draft_fantasy['position'] == 'RB')]
    wr_hits = wr_year['hit24'].sum()
    rb_hits = rb_year['hit24'].sum()
    wr_matched = (wr_year['best_ppr'] > 0).sum()
    rb_matched = (rb_year['best_ppr'] > 0).sum()
    print(f"{year:<6} {len(wr_year):<6} {len(rb_year):<6} {wr_hits:<10} {rb_hits:<10} {wr_matched:<12} {rb_matched:<12}")

# Show some 2024 and 2025 examples
print("\n--- 2024 DRAFT CLASS EXAMPLES (Top by PPG) ---")
draft_2024 = draft_fantasy[draft_fantasy['season'] == 2024].sort_values('best_ppg', ascending=False)
print(f"{'Player':<25} {'Pos':<4} {'Pick':<5} {'Best PPG':<10} {'Hit24':<6}")
print("-" * 55)
for _, row in draft_2024.head(15).iterrows():
    print(f"{row['pfr_player_name']:<25} {row['position']:<4} {row['pick']:<5} {row['best_ppg']:<10.1f} {row['hit24']:<6}")

print("\n--- 2025 DRAFT CLASS EXAMPLES (Top by PPG) ---")
draft_2025 = draft_fantasy[draft_fantasy['season'] == 2025].sort_values('best_ppg', ascending=False)
print(f"{'Player':<25} {'Pos':<4} {'Pick':<5} {'Best PPG':<10} {'Hit24':<6}")
print("-" * 55)
for _, row in draft_2025.head(15).iterrows():
    print(f"{row['pfr_player_name']:<25} {row['position']:<4} {row['pick']:<5} {row['best_ppg']:<10.1f} {row['hit24']:<6}")

# Save the data
output_path = 'data/nfl_fantasy_outcomes_2015_2025.csv'
draft_fantasy.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")

# Also save a clean version for easy merging
clean_output = draft_fantasy[['pfr_player_name', 'position', 'season', 'pick', 'round', 'team',
                               'best_ppr', 'best_ppg', 'best_season', 'hit24', 'hit12']].copy()
clean_output = clean_output.rename(columns={'season': 'draft_year', 'pfr_player_name': 'player_name'})
clean_output.to_csv('data/nfl_fantasy_clean_2015_2025.csv', index=False)
print(f"Saved: data/nfl_fantasy_clean_2015_2025.csv")

print("\n" + "=" * 80)
print("NFL FANTASY DATA FETCH COMPLETE")
print("=" * 80)
