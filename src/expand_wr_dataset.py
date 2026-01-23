"""
Expand WR Dataset to 2015-2024

Pull all necessary data to extend the backtest from 2020-2023 to 2015-2024.
"""
import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime

# API setup
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE")
CFBD_HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}

print("=" * 90)
print("EXPANDING WR DATASET TO 2015-2024")
print("=" * 90)

# ============================================================================
# STEP 1: GET DRAFT PICKS FROM NFLVERSE
# ============================================================================
print("\n" + "=" * 90)
print("STEP 1: PULLING DRAFT PICKS (2015-2024)")
print("=" * 90)

# nflverse draft picks URL
draft_url = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"

print(f"Fetching from: {draft_url}")
draft_df = pd.read_csv(draft_url)

# Filter to WRs from 2015-2024
wr_draft = draft_df[
    (draft_df['position'] == 'WR') &
    (draft_df['season'].between(2015, 2024))
].copy()

print(f"Found {len(wr_draft)} WRs drafted 2015-2024")
print(f"By year:")
for year in range(2015, 2025):
    count = len(wr_draft[wr_draft['season'] == year])
    print(f"  {year}: {count} WRs")

# Standardize columns
wr_draft = wr_draft.rename(columns={
    'season': 'draft_year',
    'pick': 'pick',
    'pfr_player_name': 'player_name',
    'college': 'college'
})

# Keep relevant columns
wr_draft = wr_draft[['player_name', 'draft_year', 'pick', 'college', 'round']].copy()

# ============================================================================
# STEP 2: GET NFL FANTASY OUTCOMES
# ============================================================================
print("\n" + "=" * 90)
print("STEP 2: PULLING NFL FANTASY OUTCOMES")
print("=" * 90)

# nflverse player stats URL
stats_url = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv"

print(f"Fetching from: {stats_url}")
stats_df = pd.read_csv(stats_url, low_memory=False)

# Filter to WRs and relevant seasons
wr_stats = stats_df[
    (stats_df['position'] == 'WR') &
    (stats_df['season'].between(2015, 2024))
].copy()

print(f"Found {len(wr_stats)} WR season records")

# Calculate fantasy points (PPR)
wr_stats['fantasy_points_ppr'] = (
    wr_stats['receiving_yards'].fillna(0) * 0.1 +
    wr_stats['receiving_tds'].fillna(0) * 6 +
    wr_stats['receptions'].fillna(0) * 1 +  # PPR
    wr_stats['rushing_yards'].fillna(0) * 0.1 +
    wr_stats['rushing_tds'].fillna(0) * 6 +
    wr_stats['receiving_2pt_conversions'].fillna(0) * 2 +
    wr_stats['rushing_2pt_conversions'].fillna(0) * 2 -
    wr_stats['receiving_fumbles_lost'].fillna(0) * 2 -
    wr_stats['rushing_fumbles_lost'].fillna(0) * 2
)

# Get best season for each player
best_seasons = wr_stats.groupby('player_display_name').agg({
    'fantasy_points_ppr': 'max',
    'season': lambda x: x.loc[wr_stats.loc[x.index, 'fantasy_points_ppr'].idxmax()]
}).reset_index()
best_seasons.columns = ['player_name', 'best_ppr', 'best_season']

# Get season rankings to determine Hit24
def get_season_rank(group):
    return group.rank(ascending=False, method='min')

wr_stats['season_rank'] = wr_stats.groupby('season')['fantasy_points_ppr'].transform(get_season_rank)

# Check if player ever finished top-24
hit24 = wr_stats.groupby('player_display_name').agg({
    'season_rank': 'min'  # Best (lowest) rank
}).reset_index()
hit24.columns = ['player_name', 'best_rank']
hit24['hit24'] = (hit24['best_rank'] <= 24).astype(int)
hit24['hit12'] = (hit24['best_rank'] <= 12).astype(int)

# Merge
fantasy_outcomes = best_seasons.merge(hit24[['player_name', 'best_rank', 'hit24', 'hit12']], on='player_name')

print(f"Calculated fantasy outcomes for {len(fantasy_outcomes)} WRs")

# ============================================================================
# STEP 3: MERGE DRAFT + FANTASY
# ============================================================================
print("\n" + "=" * 90)
print("STEP 3: MERGING DRAFT + FANTASY DATA")
print("=" * 90)

# Merge draft picks with fantasy outcomes
# Need to handle name matching carefully
wr_data = wr_draft.merge(
    fantasy_outcomes,
    on='player_name',
    how='left'
)

# Check match rate
matched = wr_data['best_ppr'].notna().sum()
total = len(wr_data)
print(f"Matched {matched}/{total} WRs ({matched/total*100:.1f}%) with fantasy outcomes")

# For unmatched, check for name variations
unmatched = wr_data[wr_data['best_ppr'].isna()]['player_name'].tolist()
if len(unmatched) > 0 and len(unmatched) <= 20:
    print(f"\nUnmatched players (showing up to 20):")
    for name in unmatched[:20]:
        print(f"  - {name}")

# Fill missing with 0 (no NFL production)
wr_data['best_ppr'] = wr_data['best_ppr'].fillna(0)
wr_data['best_rank'] = wr_data['best_rank'].fillna(999)
wr_data['hit24'] = wr_data['hit24'].fillna(0).astype(int)
wr_data['hit12'] = wr_data['hit12'].fillna(0).astype(int)
wr_data['position'] = 'WR'

# ============================================================================
# STEP 4: MERGE RAS DATA
# ============================================================================
print("\n" + "=" * 90)
print("STEP 4: MERGING RAS DATA")
print("=" * 90)

# Load new RAS data
ras_df = pd.read_csv('data/WRRas201502025.csv')
ras_df = ras_df[['Name', 'Year', 'RAS']].rename(columns={
    'Name': 'player_name',
    'Year': 'draft_year'
})

# Merge
wr_data = wr_data.merge(ras_df, on=['player_name', 'draft_year'], how='left')

ras_matched = wr_data['RAS'].notna().sum()
print(f"Matched {ras_matched}/{len(wr_data)} WRs ({ras_matched/len(wr_data)*100:.1f}%) with RAS data")

# ============================================================================
# STEP 5: PULL COLLEGE STATS FOR BREAKOUT AGE (2015-2019 draftees)
# ============================================================================
print("\n" + "=" * 90)
print("STEP 5: PULLING COLLEGE STATS FROM CFBD")
print("=" * 90)

def get_player_usage(season):
    """Get player receiving stats from CFBD for a season."""
    url = f"https://api.collegefootballdata.com/stats/player/season"
    params = {
        'year': season,
        'category': 'receiving',
        'seasonType': 'regular'
    }

    try:
        response = requests.get(url, headers=CFBD_HEADERS, params=params)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            print(f"  Error {response.status_code} for {season}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  Exception for {season}: {e}")
        return pd.DataFrame()

def get_team_stats(season):
    """Get team passing stats from CFBD for a season."""
    url = f"https://api.collegefootballdata.com/stats/season"
    params = {
        'year': season
    }

    try:
        response = requests.get(url, headers=CFBD_HEADERS, params=params)
        if response.status_code == 200:
            data = response.json()
            # Extract pass attempts per team
            team_stats = {}
            for item in data:
                team = item.get('team')
                stat = item.get('statName')
                val = item.get('statValue')
                if stat == 'passAttempts' and team:
                    team_stats[team] = int(val)
            return team_stats
        else:
            print(f"  Error {response.status_code} for team stats {season}")
            return {}
    except Exception as e:
        print(f"  Exception for team stats {season}: {e}")
        return {}

# We need college stats for seasons before 2020
# For a 2019 draftee, we need college seasons 2015-2018
# For a 2015 draftee, we need college seasons 2011-2014

college_seasons_needed = list(range(2011, 2024))  # Cover all possible college years
all_college_stats = []
all_team_stats = {}

print(f"Fetching college stats for seasons {min(college_seasons_needed)}-{max(college_seasons_needed)}...")

for season in college_seasons_needed:
    print(f"  Fetching {season}...", end=" ")

    # Player stats
    player_stats = get_player_usage(season)
    if len(player_stats) > 0:
        player_stats['college_season'] = season
        all_college_stats.append(player_stats)
        print(f"{len(player_stats)} player records", end=" ")

    # Team stats
    team_stats = get_team_stats(season)
    if team_stats:
        all_team_stats[season] = team_stats
        print(f"+ {len(team_stats)} teams")
    else:
        print("")

    time.sleep(0.5)  # Rate limiting

# Combine all college stats
if all_college_stats:
    college_df = pd.concat(all_college_stats, ignore_index=True)
    print(f"\nTotal college receiving records: {len(college_df)}")

    # Filter to receiving yards
    college_df = college_df[college_df['statType'] == 'YDS'].copy()
    college_df = college_df.rename(columns={
        'player': 'college_name',
        'team': 'college',
        'stat': 'rec_yards'
    })
    college_df['rec_yards'] = pd.to_numeric(college_df['rec_yards'], errors='coerce')

    # Add team pass attempts
    def get_team_pass_att(row):
        season = row['college_season']
        team = row['college']
        if season in all_team_stats and team in all_team_stats[season]:
            return all_team_stats[season][team]
        return np.nan

    college_df['team_pass_att'] = college_df.apply(get_team_pass_att, axis=1)

    # Calculate dominator proxy (yards / team pass attempts * 100)
    college_df['dominator'] = (college_df['rec_yards'] / college_df['team_pass_att'] * 100).round(2)

    # Save college stats
    college_df.to_csv('data/college_receiving_2011_2023.csv', index=False)
    print(f"Saved college receiving stats to data/college_receiving_2011_2023.csv")

# ============================================================================
# STEP 6: SAVE EXPANDED DATASET
# ============================================================================
print("\n" + "=" * 90)
print("STEP 6: SAVING EXPANDED DATASET")
print("=" * 90)

# Calculate seasons played (years since draft)
current_year = 2024
wr_data['seasons_played'] = current_year - wr_data['draft_year']

# Reorder columns
wr_data = wr_data[[
    'player_name', 'position', 'draft_year', 'pick', 'round', 'college',
    'seasons_played', 'best_rank', 'best_ppr', 'hit24', 'hit12', 'RAS'
]]

# Save
wr_data.to_csv('data/wr_backtest_2015_2024.csv', index=False)
print(f"Saved {len(wr_data)} WRs to data/wr_backtest_2015_2024.csv")

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"\nDataset: {len(wr_data)} WRs (2015-2024)")
print(f"\nBy year:")
for year in range(2015, 2025):
    subset = wr_data[wr_data['draft_year'] == year]
    hits = subset['hit24'].sum()
    has_ras = subset['RAS'].notna().sum()
    print(f"  {year}: {len(subset):>3} WRs, {hits:>2} hits ({hits/len(subset)*100:>5.1f}%), {has_ras:>3} have RAS ({has_ras/len(subset)*100:>5.1f}%)")

print(f"\nOverall:")
print(f"  Total WRs: {len(wr_data)}")
print(f"  Have RAS: {wr_data['RAS'].notna().sum()} ({wr_data['RAS'].notna().mean()*100:.1f}%)")
print(f"  Total Hits (24): {wr_data['hit24'].sum()} ({wr_data['hit24'].mean()*100:.1f}%)")
print(f"  Total Hits (12): {wr_data['hit12'].sum()} ({wr_data['hit12'].mean()*100:.1f}%)")
