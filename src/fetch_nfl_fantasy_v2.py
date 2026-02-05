"""
Fetch NFL Fantasy Data (2015-2025) - Fixed Version
===================================================

Uses weekly data to calculate seasonal fantasy points and best PPG.
Correctly calculates hit24/hit12 based on best PPG season.
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FETCHING NFL FANTASY DATA (2015-2025) - FIXED")
print("=" * 80)

# Fetch weekly data for all years
print("\nFetching weekly fantasy data...")
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
}).reset_index()
seasonal.columns = ['player_id', 'player_name', 'player_display_name', 'position', 'season',
                    'fantasy_points_ppr', 'games']

# Calculate PPG for each season
seasonal['ppg'] = seasonal['fantasy_points_ppr'] / seasonal['games']

# Get BEST PPG season for each player (not max total points)
print("Finding best PPG season for each player...")
idx_best_ppg = seasonal.groupby('player_id')['ppg'].idxmax()
best_seasons = seasonal.loc[idx_best_ppg].copy()
best_seasons = best_seasons.rename(columns={
    'fantasy_points_ppr': 'best_ppr',
    'ppg': 'best_ppg',
    'season': 'best_season'
})

# Add hit thresholds based on best PPG
best_seasons['hit24'] = (best_seasons['best_ppg'] >= 24).astype(int)
best_seasons['hit12'] = (best_seasons['best_ppg'] >= 12).astype(int)

print(f"\nPlayers with fantasy data: {len(best_seasons)}")
print(f"  WRs with Hit24: {best_seasons[(best_seasons['position'] == 'WR') & (best_seasons['hit24'] == 1)].shape[0]}")
print(f"  RBs with Hit24: {best_seasons[(best_seasons['position'] == 'RB') & (best_seasons['hit24'] == 1)].shape[0]}")

# Show top WRs by PPG
print("\n--- TOP 15 WRs BY BEST PPG ---")
top_wr = best_seasons[best_seasons['position'] == 'WR'].sort_values('best_ppg', ascending=False).head(15)
print(f"{'Player':<25} {'Best PPG':<10} {'Season':<8} {'Hit24':<6}")
print("-" * 55)
for _, row in top_wr.iterrows():
    print(f"{row['player_display_name']:<25} {row['best_ppg']:<10.1f} {int(row['best_season']):<8} {row['hit24']:<6}")

# Fetch draft data
print("\n\nFetching draft data...")
draft = nfl.import_draft_picks(years=list(range(2015, 2026)))
draft = draft[draft['position'].isin(['RB', 'WR'])].copy()
print(f"  Drafted RB/WR: {len(draft)}")

# Match draft to fantasy using multiple approaches
draft['name_clean'] = draft['pfr_player_name'].str.lower().str.strip().str.replace('.', '', regex=False).str.replace("'", "", regex=False).str.replace('-', ' ', regex=False)
best_seasons['name_clean'] = best_seasons['player_display_name'].str.lower().str.strip().str.replace('.', '', regex=False).str.replace("'", "", regex=False).str.replace('-', ' ', regex=False)

# Merge on cleaned name and position
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
print(f"{'Year':<6} {'WRs':<6} {'RBs':<6} {'WR Hit24':<10} {'RB Hit24':<10} {'WR Hit12':<10} {'RB Hit12':<10}")
print("-" * 75)

for year in range(2015, 2026):
    wr_year = draft_fantasy[(draft_fantasy['season'] == year) & (draft_fantasy['position'] == 'WR')]
    rb_year = draft_fantasy[(draft_fantasy['season'] == year) & (draft_fantasy['position'] == 'RB')]
    wr_hit24 = wr_year['hit24'].sum()
    rb_hit24 = rb_year['hit24'].sum()
    wr_hit12 = wr_year['hit12'].sum()
    rb_hit12 = rb_year['hit12'].sum()
    print(f"{year:<6} {len(wr_year):<6} {len(rb_year):<6} {wr_hit24:<10} {rb_hit24:<10} {wr_hit12:<10} {rb_hit12:<10}")

# Show 2024 draft class
print("\n--- 2024 DRAFT CLASS TOP PERFORMERS ---")
draft_2024 = draft_fantasy[draft_fantasy['season'] == 2024].sort_values('best_ppg', ascending=False)
print(f"{'Player':<25} {'Pos':<4} {'Pick':<5} {'Best PPG':<10} {'Hit24':<6} {'Hit12':<6}")
print("-" * 60)
for _, row in draft_2024.head(15).iterrows():
    print(f"{row['pfr_player_name']:<25} {row['position']:<4} {int(row['pick']):<5} {row['best_ppg']:<10.1f} {row['hit24']:<6} {row['hit12']:<6}")

# Show 2025 draft class
print("\n--- 2025 DRAFT CLASS ---")
draft_2025 = draft_fantasy[draft_fantasy['season'] == 2025].sort_values('best_ppg', ascending=False)
print(f"{'Player':<25} {'Pos':<4} {'Pick':<5} {'Best PPG':<10} {'Hit24':<6}")
print("-" * 55)
for _, row in draft_2025.head(15).iterrows():
    print(f"{row['pfr_player_name']:<25} {row['position']:<4} {int(row['pick']):<5} {row['best_ppg']:<10.1f} {row['hit24']:<6}")

# Save the data
output_path = 'data/nfl_fantasy_outcomes_2015_2025.csv'
draft_fantasy.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")

# Also save a clean version for easy merging
clean_output = draft_fantasy[['pfr_player_name', 'position', 'season', 'pick', 'round', 'team', 'college',
                               'best_ppr', 'best_ppg', 'best_season', 'hit24', 'hit12']].copy()
clean_output = clean_output.rename(columns={'season': 'draft_year', 'pfr_player_name': 'player_name'})
clean_output.to_csv('data/nfl_fantasy_clean_2015_2025.csv', index=False)
print(f"Saved: data/nfl_fantasy_clean_2015_2025.csv")

# Summary stats
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nTotal drafted RB/WR (2015-2025): {len(draft_fantasy)}")
print(f"  WRs: {len(draft_fantasy[draft_fantasy['position'] == 'WR'])}")
print(f"  RBs: {len(draft_fantasy[draft_fantasy['position'] == 'RB'])}")

total_wr_hit24 = draft_fantasy[(draft_fantasy['position'] == 'WR')]['hit24'].sum()
total_rb_hit24 = draft_fantasy[(draft_fantasy['position'] == 'RB')]['hit24'].sum()
total_wr_hit12 = draft_fantasy[(draft_fantasy['position'] == 'WR')]['hit12'].sum()
total_rb_hit12 = draft_fantasy[(draft_fantasy['position'] == 'RB')]['hit12'].sum()

print(f"\nHit Rates (2015-2025):")
print(f"  WR Hit24: {total_wr_hit24}/355 ({total_wr_hit24/355*100:.1f}%)")
print(f"  WR Hit12: {total_wr_hit12}/355 ({total_wr_hit12/355*100:.1f}%)")
print(f"  RB Hit24: {total_rb_hit24}/235 ({total_rb_hit24/235*100:.1f}%)")
print(f"  RB Hit12: {total_rb_hit12}/235 ({total_rb_hit12/235*100:.1f}%)")

print("\n" + "=" * 80)
print("NFL FANTASY DATA FETCH COMPLETE")
print("=" * 80)
