"""
Calculate Breakout Ages for Expanded Dataset (2015-2024)

Use the college receiving stats we pulled to calculate breakout ages
for all WRs in the expanded dataset.
"""
import pandas as pd
import numpy as np
import requests
import time
import os

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE")
CFBD_HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}

print("=" * 90)
print("CALCULATING BREAKOUT AGES FOR EXPANDED DATASET")
print("=" * 90)

# Load expanded WR dataset
wr_df = pd.read_csv('data/wr_backtest_2015_2024.csv')
print(f"Loaded {len(wr_df)} WRs from expanded dataset")

# Load college stats
college_df = pd.read_csv('data/college_receiving_2011_2023.csv')
print(f"Loaded {len(college_df)} college receiving records")

# Load existing breakout ages (2020-2024)
existing_breakout = pd.read_csv('data/wr_breakout_age_scores_fixed.csv')
print(f"Loaded {len(existing_breakout)} existing breakout ages")

# ============================================================================
# GET PLAYER BIRTHDATES FROM CFBD
# ============================================================================
print("\n" + "-" * 60)
print("Fetching player birthdates from CFBD...")
print("-" * 60)

def get_player_info(name):
    """Search for player info from CFBD."""
    url = "https://api.collegefootballdata.com/player/search"
    params = {'searchTerm': name, 'position': 'WR'}

    try:
        response = requests.get(url, headers=CFBD_HEADERS, params=params)
        if response.status_code == 200:
            data = response.json()
            if data:
                # Return first match
                return data[0]
        return None
    except:
        return None

# ============================================================================
# CALCULATE BREAKOUT AGE
# ============================================================================
print("\n" + "-" * 60)
print("Calculating breakout ages...")
print("-" * 60)

def calculate_breakout_age(player_name, college, draft_year, college_df):
    """
    Calculate breakout age for a player.
    Breakout = first season with 20%+ dominator OR 700+ yards
    """
    # Get player's college seasons (up to 5 years before draft)
    player_seasons = college_df[
        (college_df['college_name'].str.lower().str.contains(player_name.split()[-1].lower(), na=False)) &
        (college_df['college_season'].between(draft_year - 5, draft_year - 1))
    ].copy()

    if len(player_seasons) == 0:
        return None, None, None

    # Sort by season
    player_seasons = player_seasons.sort_values('college_season')

    # Find breakout season
    breakout_age = None
    peak_dominator = player_seasons['dominator'].max() if 'dominator' in player_seasons.columns else None
    peak_yards = player_seasons['rec_yards'].max()

    for _, row in player_seasons.iterrows():
        season = row['college_season']
        yards = row['rec_yards']
        dominator = row.get('dominator', 0) or 0

        # Calculate age in that season (rough estimate: draft_year - seasons_left + 18)
        seasons_before_draft = draft_year - season
        age = 23 - seasons_before_draft  # Rough estimate

        # Check breakout criteria
        if dominator >= 20 or yards >= 700:
            breakout_age = age
            break

    return breakout_age, peak_dominator, peak_yards

# Process each player
results = []
for idx, row in wr_df.iterrows():
    player_name = row['player_name']
    college = row['college']
    draft_year = row['draft_year']

    # Check if we already have breakout data
    existing = existing_breakout[
        (existing_breakout['player_name'] == player_name) &
        (existing_breakout['draft_year'] == draft_year)
    ]

    if len(existing) > 0:
        # Use existing data
        results.append({
            'player_name': player_name,
            'draft_year': draft_year,
            'college': college,
            'breakout_age': existing.iloc[0]['breakout_age'],
            'peak_dominator': existing.iloc[0]['peak_dominator'],
            'source': 'existing'
        })
    else:
        # Calculate new
        breakout_age, peak_dom, peak_yards = calculate_breakout_age(
            player_name, college, draft_year, college_df
        )
        results.append({
            'player_name': player_name,
            'draft_year': draft_year,
            'college': college,
            'breakout_age': breakout_age,
            'peak_dominator': peak_dom,
            'source': 'calculated'
        })

    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(wr_df)} players...")

# Create results dataframe
breakout_df = pd.DataFrame(results)

# Summary
print(f"\n" + "=" * 60)
print("BREAKOUT AGE SUMMARY")
print("=" * 60)

has_breakout = breakout_df['breakout_age'].notna().sum()
from_existing = (breakout_df['source'] == 'existing').sum()
from_calc = (breakout_df['source'] == 'calculated').sum()

print(f"\nTotal WRs: {len(breakout_df)}")
print(f"Have breakout age: {has_breakout} ({has_breakout/len(breakout_df)*100:.1f}%)")
print(f"  From existing data: {from_existing}")
print(f"  Newly calculated: {from_calc}")

# Save
breakout_df.to_csv('data/wr_breakout_ages_expanded.csv', index=False)
print(f"\nSaved to data/wr_breakout_ages_expanded.csv")

# ============================================================================
# MERGE WITH MAIN DATASET
# ============================================================================
print("\n" + "=" * 60)
print("CREATING FINAL EXPANDED DATASET")
print("=" * 60)

# Merge breakout ages with main dataset
final_df = wr_df.merge(
    breakout_df[['player_name', 'draft_year', 'breakout_age', 'peak_dominator']],
    on=['player_name', 'draft_year'],
    how='left'
)

# Summary by year
print("\nFinal dataset summary:")
print("-" * 70)
print(f"{'Year':<6} {'WRs':>5} {'Hits':>5} {'RAS':>5} {'Breakout':>8}")
print("-" * 70)

for year in range(2015, 2025):
    subset = final_df[final_df['draft_year'] == year]
    hits = subset['hit24'].sum()
    has_ras = subset['RAS'].notna().sum()
    has_breakout = subset['breakout_age'].notna().sum()
    print(f"{year:<6} {len(subset):>5} {hits:>5} {has_ras:>5} {has_breakout:>8}")

total = len(final_df)
print("-" * 70)
print(f"{'TOTAL':<6} {total:>5} {final_df['hit24'].sum():>5} {final_df['RAS'].notna().sum():>5} {final_df['breakout_age'].notna().sum():>8}")

# Calculate complete data
has_all = (
    final_df['RAS'].notna() &
    final_df['breakout_age'].notna()
).sum()
print(f"\nWRs with COMPLETE data (DC + RAS + Breakout): {has_all} ({has_all/total*100:.1f}%)")

# Save final dataset
final_df.to_csv('data/wr_backtest_expanded_final.csv', index=False)
print(f"\nSaved final dataset to data/wr_backtest_expanded_final.csv")
