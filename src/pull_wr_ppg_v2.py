"""
Pull WR NFL Fantasy Data - Version 2 with better name matching
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=" * 100)
print("PULLING WR NFL FANTASY DATA FROM NFLVERSE")
print("=" * 100)

# Load the current database
df = pd.read_csv('output/slap_complete_database_v4.csv')
wr_backtest = df[(df['position'] == 'WR') & (df['draft_year'] >= 2015) & (df['draft_year'] <= 2023)].copy()
print(f"\nWRs in backtest: {len(wr_backtest)}")

# Pull NFL fantasy data from nflverse
print("\nFetching NFL fantasy data from nflverse...")

all_seasons = []
for year in range(2015, 2025):
    try:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.csv"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            season_df = pd.read_csv(StringIO(response.text))
            season_df['season'] = year
            all_seasons.append(season_df)
            print(f"   {year}: {len(season_df)} records")
    except Exception as e:
        print(f"   {year}: Error - {e}")

nfl_data = pd.concat(all_seasons, ignore_index=True)
nfl_wr = nfl_data[nfl_data['position'] == 'WR'].copy()
print(f"\nTotal WR records: {len(nfl_wr)}")

# Check name columns
print(f"\nName columns available:")
name_cols = [c for c in nfl_wr.columns if 'name' in c.lower()]
print(f"   {name_cols}")

# Sample names from nflverse
print(f"\nSample player names from nflverse:")
sample_names = nfl_wr['player_display_name'].dropna().unique()[:20]
for name in sample_names:
    print(f"   '{name}'")

# Sample names from our database
print(f"\nSample player names from our database:")
for name in wr_backtest['player_name'].head(20):
    print(f"   '{name}'")

# Use player_display_name for matching
nfl_wr['match_name'] = nfl_wr['player_display_name'].str.lower().str.strip()

# Aggregate by player and season
season_totals = nfl_wr.groupby(['player_display_name', 'season']).agg({
    'fantasy_points_ppr': 'sum',
    'match_name': 'first'
}).reset_index()

# Get games played
games = nfl_wr.groupby(['player_display_name', 'season']).size().reset_index(name='games')
season_totals = season_totals.merge(games, on=['player_display_name', 'season'])

# Filter to 4+ games
season_totals = season_totals[season_totals['games'] >= 4]

# Calculate PPG
season_totals['ppg'] = season_totals['fantasy_points_ppr'] / season_totals['games']

# Get best season for each player
best_seasons = season_totals.loc[season_totals.groupby('player_display_name')['ppg'].idxmax()].copy()
best_seasons = best_seasons[['player_display_name', 'match_name', 'ppg', 'games', 'fantasy_points_ppr']].copy()

print(f"\nUnique WRs with best season: {len(best_seasons)}")

# Create name lookup
name_to_ppg = dict(zip(best_seasons['match_name'], best_seasons['ppg']))

# Also create variations for common name differences
def normalize_name(name):
    """Normalize name for matching"""
    name = name.lower().strip()
    # Remove suffixes
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)$', '', name)
    # Remove special characters
    name = re.sub(r'[^\w\s]', '', name)
    # Normalize whitespace
    name = ' '.join(name.split())
    return name

# Build lookup with normalized names
norm_lookup = {}
for _, row in best_seasons.iterrows():
    norm = normalize_name(row['player_display_name'])
    norm_lookup[norm] = row['ppg']

# Also try last name, first name matching
for _, row in best_seasons.iterrows():
    parts = row['player_display_name'].split()
    if len(parts) >= 2:
        # First Last format
        norm_lookup[normalize_name(row['player_display_name'])] = row['ppg']

print(f"Lookup entries: {len(norm_lookup)}")

# Match function
def find_ppg(player_name):
    # Try exact match first
    norm = normalize_name(player_name)
    if norm in norm_lookup:
        return norm_lookup[norm]

    # Try without middle name/initial
    parts = player_name.split()
    if len(parts) > 2:
        short_name = f"{parts[0]} {parts[-1]}"
        norm = normalize_name(short_name)
        if norm in norm_lookup:
            return norm_lookup[norm]

    return None

# Apply matching
wr_backtest['nfl_best_ppg_new'] = wr_backtest['player_name'].apply(find_ppg)

matched = wr_backtest['nfl_best_ppg_new'].notna().sum()
print(f"\nMatched: {matched}/{len(wr_backtest)} ({matched/len(wr_backtest)*100:.1f}%)")

# Debug: Show some matches and non-matches
print(f"\nSuccessful matches:")
matched_rows = wr_backtest[wr_backtest['nfl_best_ppg_new'].notna()].head(10)
for _, row in matched_rows.iterrows():
    print(f"   {row['player_name']}: {row['nfl_best_ppg_new']:.1f} PPG")

print(f"\nFailed matches (first 20):")
unmatched_rows = wr_backtest[wr_backtest['nfl_best_ppg_new'].isna()].head(20)
for _, row in unmatched_rows.iterrows():
    norm = normalize_name(row['player_name'])
    print(f"   '{row['player_name']}' -> '{norm}'")

# Check specific players
print(f"\n" + "=" * 50)
print("SPECIFIC PLAYER CHECK")
print("=" * 50)

check_players = ["Ja'Marr Chase", "CeeDee Lamb", "Justin Jefferson", "Cooper Kupp", "Tyreek Hill"]
for p in check_players:
    norm = normalize_name(p)
    ppg = norm_lookup.get(norm, "NOT FOUND")
    print(f"   {p} -> '{norm}' -> {ppg}")

# Check what's in nflverse for these players
print(f"\n   In nflverse:")
for p in check_players:
    matches = best_seasons[best_seasons['player_display_name'].str.contains(p.split()[-1], case=False)]
    if len(matches) > 0:
        for _, m in matches.head(2).iterrows():
            print(f"      '{m['player_display_name']}': {m['ppg']:.1f} PPG")
