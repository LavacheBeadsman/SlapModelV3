"""
Fetch Complete Data for SLAP Model
==================================

Fetches:
1. NFL fantasy outcomes (2015-2025) from NFLVerse
2. College receiving data (2014-2025) from CFBD

Updates all backtest files with complete data.
"""

import pandas as pd
import numpy as np
import requests
import time

print("=" * 80)
print("FETCHING COMPLETE DATA FOR SLAP MODEL")
print("=" * 80)

# ============================================================================
# PART 1: FETCH NFL FANTASY DATA (2015-2025)
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: FETCHING NFL FANTASY DATA (2015-2025)")
print("=" * 80)

try:
    import nfl_data_py as nfl

    # Fetch seasonal fantasy data
    print("\nFetching seasonal fantasy data...")
    seasonal = nfl.import_seasonal_data(years=list(range(2015, 2026)))
    print(f"  Loaded {len(seasonal)} player-seasons")

    # Filter to RB and WR only
    seasonal = seasonal[seasonal['position'].isin(['RB', 'WR'])].copy()
    print(f"  RB/WR records: {len(seasonal)}")

    # Calculate PPR PPG (fantasy_points_ppr / games)
    seasonal['ppg'] = seasonal['fantasy_points_ppr'] / seasonal['games']

    # Get best season PPG for each player
    best_seasons = seasonal.groupby('player_id').agg({
        'player_name': 'first',
        'position': 'first',
        'fantasy_points_ppr': 'max',
        'ppg': 'max',
        'season': lambda x: x.iloc[seasonal.loc[x.index, 'fantasy_points_ppr'].argmax()]
    }).reset_index()
    best_seasons.columns = ['player_id', 'player_name', 'position', 'best_ppr', 'best_ppg', 'best_season']

    # Add hit thresholds
    best_seasons['hit24'] = (best_seasons['best_ppg'] >= 24).astype(int)
    best_seasons['hit12'] = (best_seasons['best_ppg'] >= 12).astype(int)

    print(f"\nPlayers with fantasy data: {len(best_seasons)}")
    print(f"  WRs: {len(best_seasons[best_seasons['position'] == 'WR'])}")
    print(f"  RBs: {len(best_seasons[best_seasons['position'] == 'RB'])}")

    # Fetch draft data to link players
    print("\nFetching draft data...")
    draft = nfl.import_draft_picks(years=list(range(2015, 2026)))
    draft = draft[draft['position'].isin(['RB', 'WR'])].copy()
    print(f"  Drafted RB/WR: {len(draft)}")

    # Merge fantasy data with draft data
    # Clean names for matching
    draft['name_lower'] = draft['pfr_name'].str.lower().str.strip()
    best_seasons['name_lower'] = best_seasons['player_name'].str.lower().str.strip()

    # Merge on name
    draft_with_fantasy = draft.merge(
        best_seasons[['name_lower', 'best_ppr', 'best_ppg', 'best_season', 'hit24', 'hit12']],
        on='name_lower',
        how='left'
    )

    # Fill missing with 0
    draft_with_fantasy['best_ppr'] = draft_with_fantasy['best_ppr'].fillna(0)
    draft_with_fantasy['best_ppg'] = draft_with_fantasy['best_ppg'].fillna(0)
    draft_with_fantasy['hit24'] = draft_with_fantasy['hit24'].fillna(0).astype(int)
    draft_with_fantasy['hit12'] = draft_with_fantasy['hit12'].fillna(0).astype(int)

    # Summary by draft year
    print("\n--- NFL FANTASY DATA BY DRAFT YEAR ---")
    print(f"{'Year':<6} {'WRs':<6} {'RBs':<6} {'WR Hit24':<10} {'RB Hit24':<10}")
    print("-" * 45)

    for year in range(2015, 2026):
        wr_year = draft_with_fantasy[(draft_with_fantasy['season'] == year) & (draft_with_fantasy['position'] == 'WR')]
        rb_year = draft_with_fantasy[(draft_with_fantasy['season'] == year) & (draft_with_fantasy['position'] == 'RB')]
        wr_hits = wr_year['hit24'].sum()
        rb_hits = rb_year['hit24'].sum()
        print(f"{year:<6} {len(wr_year):<6} {len(rb_year):<6} {wr_hits:<10} {rb_hits:<10}")

    # Save NFL fantasy data
    nfl_fantasy_path = 'data/nfl_fantasy_outcomes_2015_2025.csv'
    draft_with_fantasy.to_csv(nfl_fantasy_path, index=False)
    print(f"\nSaved: {nfl_fantasy_path}")

except Exception as e:
    print(f"Error fetching NFL data: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# PART 2: FETCH COLLEGE RECEIVING DATA (2014-2025) FROM CFBD
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: FETCHING COLLEGE RECEIVING DATA (2014-2025)")
print("=" * 80)

CFBD_API_KEY = 'xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE'
headers = {'Authorization': f'Bearer {CFBD_API_KEY}'}

def fetch_cfbd_receiving(year):
    """Fetch receiving stats for a given year from CFBD."""
    url = 'https://api.collegefootballdata.com/stats/player/season'
    params = {'year': year, 'category': 'receiving'}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  Year {year}: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"  Year {year}: Error - {e}")
        return []

def fetch_cfbd_team_stats(year):
    """Fetch team passing attempts for a given year."""
    url = 'https://api.collegefootballdata.com/stats/season'
    params = {'year': year}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # Extract pass attempts per team
            team_pass_att = {}
            for item in data:
                if item.get('statName') == 'passAttempts':
                    team_pass_att[item['team']] = int(item['statValue'])
            return team_pass_att
        else:
            print(f"  Team stats {year}: HTTP {response.status_code}")
            return {}
    except Exception as e:
        print(f"  Team stats {year}: Error - {e}")
        return {}

# Fetch data for all years
all_receiving = []
all_team_pass_att = {}

print("\nFetching receiving stats by year...")
for year in range(2014, 2026):
    print(f"  Fetching {year}...", end=" ")

    # Receiving stats
    receiving = fetch_cfbd_receiving(year)

    # Team pass attempts
    team_stats = fetch_cfbd_team_stats(year)
    all_team_pass_att[year] = team_stats

    # Process receiving data
    for player in receiving:
        # Only keep RBs
        if player.get('category') == 'receiving':
            all_receiving.append({
                'season': year,
                'player_id': player.get('playerId'),
                'player_name': player.get('player'),
                'team': player.get('team'),
                'conference': player.get('conference'),
                'position': player.get('position', 'UNK'),
                'stat_type': player.get('statType'),
                'stat_value': player.get('stat')
            })

    rb_count = len([p for p in receiving if p.get('position') == 'RB'])
    print(f"RBs: {rb_count}, Teams: {len(team_stats)}")
    time.sleep(0.5)  # Rate limiting

# Convert to DataFrame
receiving_df = pd.DataFrame(all_receiving)

# Pivot to get rec_yards per player
if len(receiving_df) > 0:
    receiving_pivot = receiving_df[receiving_df['stat_type'] == 'YDS'].copy()
    receiving_pivot = receiving_pivot.rename(columns={'stat_value': 'rec_yards'})

    # Add team pass attempts
    receiving_pivot['team_pass_att'] = receiving_pivot.apply(
        lambda x: all_team_pass_att.get(x['season'], {}).get(x['team'], np.nan),
        axis=1
    )

    # Filter to RBs only
    rb_receiving = receiving_pivot[receiving_pivot['position'] == 'RB'].copy()

    print(f"\n--- COLLEGE RECEIVING DATA SUMMARY ---")
    print(f"Total RB receiving records: {len(rb_receiving)}")

    print(f"\n{'Year':<6} {'RB Records':<12} {'With Team Pass Att':<20}")
    print("-" * 40)
    for year in range(2014, 2026):
        year_data = rb_receiving[rb_receiving['season'] == year]
        with_att = year_data['team_pass_att'].notna().sum()
        print(f"{year:<6} {len(year_data):<12} {with_att:<20}")

    # Save college receiving data
    college_receiving_path = 'data/college_receiving_2014_2025.csv'
    rb_receiving.to_csv(college_receiving_path, index=False)
    print(f"\nSaved: {college_receiving_path}")
else:
    print("No receiving data fetched")

print("\n" + "=" * 80)
print("DATA FETCH COMPLETE")
print("=" * 80)
