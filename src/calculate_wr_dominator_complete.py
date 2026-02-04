"""
Calculate College Dominator Rating for ALL WRs (2015-2026 draft classes).

DOMINATOR RATING FORMULA:
  Dominator = (Yards Market Share + TD Market Share) / 2

Where:
  - Yards Market Share = Player Rec Yards / Team Rec Yards
  - TD Market Share = Player Rec TDs / Team Rec TDs

This script:
1. Fetches all player receiving stats from CFBD API for seasons 2014-2025
2. Calculates team totals by summing all players per team
3. Calculates dominator rating for each WR
4. Outputs complete coverage for all 433 WRs
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# CFBD API configuration
API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

# School name mappings (nflverse -> CFBD format)
SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State",
    "Penn St.": "Penn State",
    "Michigan St.": "Michigan State",
    "North Dakota St.": "North Dakota State",
    "Oklahoma St.": "Oklahoma State",
    "Miss. St.": "Mississippi State",
    "Mississippi St.": "Mississippi State",
    "San Diego St.": "San Diego State",
    "Florida St.": "Florida State",
    "Fresno St.": "Fresno State",
    "Boise St.": "Boise State",
    "Iowa St.": "Iowa State",
    "Arizona St.": "Arizona State",
    "Kansas St.": "Kansas State",
    "N.C. State": "NC State",
    "N.C.": "North Carolina",
    "North Carolina St.": "NC State",
    "Appalachian St.": "Appalachian State",
    "Colorado St.": "Colorado State",
    "Oregon St.": "Oregon State",
    "Washington St.": "Washington State",
    "S. Carolina": "South Carolina",
    "Miami (FL)": "Miami",
    "Central Mich.": "Central Michigan",
    "Western Mich.": "Western Michigan",
    "Eastern Mich.": "Eastern Michigan",
    "Northern Ill.": "Northern Illinois",
    "Southern Miss.": "Southern Mississippi",
    "Southern Miss": "Southern Mississippi",
    "San Jose St.": "San Jose State",
    "La.-Monroe": "Louisiana Monroe",
    "La.-Lafayette": "Louisiana",
    "Louisiana-Lafayette": "Louisiana",
    "Texas-El Paso": "UTEP",
    "Ala-Birmingham": "UAB",
    "South Dakota St.": "South Dakota State",
    "New Mexico St.": "New Mexico State",
    "Boston Col.": "Boston College",
    "Mississippi": "Ole Miss",
    "Pitt": "Pittsburgh",
    "Central Florida": "UCF",
    "Bowling Green St.": "Bowling Green",
    "Ball St.": "Ball State",
    "Kent St.": "Kent State",
    "Arkansas St.": "Arkansas State",
    "Georgia St.": "Georgia State",
    "Middle Tenn. St.": "Middle Tennessee",
    "Middle Tennessee St.": "Middle Tennessee",
    "Utah St.": "Utah State",
    "UConn": "Connecticut",
    "Connecticut": "Connecticut",
    "UTSA": "UT San Antonio",
    "SMU": "SMU",
    "TCU": "TCU",
    "LSU": "LSU",
    "USC": "USC",
    "UCLA": "UCLA",
    "UCF": "UCF",
    "UAB": "UAB",
    "BYU": "BYU",
    "Buffalo": "Buffalo",
    "Hawaii": "Hawai'i",
    # FCS schools - will return None
    "Weber St.": None,
    "Youngstown St.": None,
    "Illinois St.": None,
    "South Dakota State": None,
    "Samford": None,
    "William & Mary": None,
    "Monmouth": None,
    "Rhode Island": None,
    "Charleston (WV)": None,
    "Princeton": None,
    "SE Missouri St.": None,
    "Eastern Washington": None,
    "Northern Iowa": None,
    "Central Arkansas": None,
    "West Alabama": None,
    "Grambling St.": None,
    "East Central (OK)": None,
    "Pennsylvania": None,
}


def normalize_school(school):
    """Convert school name to CFBD format."""
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def fetch_team_receiving_stats(team, year):
    """
    Fetch all player receiving stats for a team in a season.
    Returns dict with player stats and team totals.
    """
    url = f"{BASE_URL}/stats/player/season"
    params = {
        "year": year,
        "category": "receiving",
        "team": team
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()

            # Parse player stats
            player_stats = {}
            for stat in data:
                player = stat.get("player", "").strip()
                stat_type = stat.get("statType", "")
                value = stat.get("stat", 0)

                if player not in player_stats:
                    player_stats[player] = {"player_name": player, "YDS": 0, "TD": 0, "REC": 0}

                try:
                    if stat_type == "YDS":
                        player_stats[player]["YDS"] = int(float(value))
                    elif stat_type == "TD":
                        player_stats[player]["TD"] = int(float(value))
                    elif stat_type == "REC":
                        player_stats[player]["REC"] = int(float(value))
                except (ValueError, TypeError):
                    pass

            # Calculate team totals
            team_rec_yards = sum(p.get("YDS", 0) for p in player_stats.values())
            team_rec_tds = sum(p.get("TD", 0) for p in player_stats.values())

            return {
                "players": player_stats,
                "team_rec_yards": team_rec_yards,
                "team_rec_tds": team_rec_tds
            }
        elif response.status_code == 429:
            print(f"    Rate limited, waiting 5s...")
            time.sleep(5)
            return fetch_team_receiving_stats(team, year)
        else:
            return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def names_match(name1, name2):
    """Check if two names refer to the same person."""
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    if n1 == n2:
        return True

    # Remove suffixes and punctuation
    parts1 = n1.replace('.', ' ').replace("'", "").replace("-", " ").split()
    parts2 = n2.replace('.', ' ').replace("'", "").replace("-", " ").split()

    suffixes = {'jr', 'jr.', 'ii', 'iii', 'iv', 'sr'}
    clean1 = [p for p in parts1 if p not in suffixes]
    clean2 = [p for p in parts2 if p not in suffixes]

    if not clean1 or not clean2:
        return False

    # Compare last names
    if len(clean1) > 1 and len(clean2) > 1:
        if clean1[-1] != clean2[-1]:
            return False

    # Compare first names (allow partial match)
    first1, first2 = clean1[0], clean2[0]
    if first1 != first2:
        # Allow initial match
        if len(first1) == 1 and first2.startswith(first1):
            return True
        if len(first2) == 1 and first1.startswith(first2):
            return True
        # Allow nickname/full name
        if first1.startswith(first2) or first2.startswith(first1):
            return True
        return False

    return True


def find_player_stats(player_name, team_data):
    """Find a player's stats in team data."""
    if not team_data or "players" not in team_data:
        return None, None, None

    for api_name, stats in team_data["players"].items():
        if names_match(player_name, api_name):
            return stats.get("YDS", 0), stats.get("TD", 0), api_name

    return None, None, None


def calculate_dominator(player_rec_yards, player_rec_tds, team_rec_yards, team_rec_tds):
    """
    Calculate dominator rating using the standard formula.

    Dominator = (Yards Market Share + TD Market Share) / 2
    """
    if team_rec_yards is None or team_rec_yards == 0:
        return None, None, None

    yards_share = player_rec_yards / team_rec_yards

    if team_rec_tds is None or team_rec_tds == 0:
        # If no team TDs, use only yards share
        td_share = yards_share  # Fall back to yards share
    else:
        td_share = player_rec_tds / team_rec_tds

    dominator = (yards_share + td_share) / 2

    return yards_share, td_share, dominator


def main():
    print("=" * 90)
    print("CALCULATING COLLEGE DOMINATOR RATING FOR ALL WRs (2015-2026)")
    print("=" * 90)

    # Load WR data
    wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
    prospects = pd.read_csv('data/prospects_final.csv')
    wr_2026 = prospects[prospects['position'] == 'WR'].copy()

    # Get draft ages from nflverse
    draft = pd.read_parquet('data/nflverse/draft_picks.parquet')
    wr_draft = draft[(draft['position'] == 'WR') & (draft['season'] >= 2015) & (draft['season'] <= 2024)]

    # Create unified WR list
    all_wrs = []

    # Add backtest WRs
    for _, row in wr_backtest.iterrows():
        all_wrs.append({
            'player_name': row['player_name'],
            'draft_year': row['draft_year'],
            'college': row['college'],
            'pick': row['pick']
        })

    # Add 2026 prospects
    for _, row in wr_2026.iterrows():
        all_wrs.append({
            'player_name': row['player_name'],
            'draft_year': 2026,
            'college': row['school'],
            'pick': row['projected_pick']
        })

    all_wrs_df = pd.DataFrame(all_wrs)
    print(f"\nTotal WRs to process: {len(all_wrs_df)}")

    # Add draft age from nflverse (for backtest only)
    all_wrs_df = all_wrs_df.merge(
        wr_draft[['season', 'pfr_player_name', 'age']].rename(
            columns={'season': 'draft_year', 'pfr_player_name': 'player_name', 'age': 'draft_age'}
        ),
        on=['player_name', 'draft_year'],
        how='left'
    )

    # For 2026, get age from prospects
    for idx, row in all_wrs_df.iterrows():
        if row['draft_year'] == 2026 and pd.isna(row['draft_age']):
            match = wr_2026[wr_2026['player_name'] == row['player_name']]
            if len(match) > 0 and 'age' in match.columns:
                all_wrs_df.at[idx, 'draft_age'] = match.iloc[0]['age']

    # Calculate final college season for each WR
    all_wrs_df['final_season'] = all_wrs_df['draft_year'] - 1

    # Cache for team data (to avoid redundant API calls)
    team_cache = {}

    results = []

    # Process each WR
    for idx, row in all_wrs_df.iterrows():
        player_name = row['player_name']
        draft_year = row['draft_year']
        final_season = row['final_season']
        college = row['college']
        pick = row['pick']
        draft_age = row.get('draft_age')

        if pd.isna(college):
            print(f"  [{idx+1}/{len(all_wrs_df)}] {player_name} - No college listed")
            results.append({
                'player_name': player_name,
                'draft_year': draft_year,
                'college': college,
                'pick': pick,
                'season_used': final_season,
                'player_rec_yards': None,
                'player_rec_tds': None,
                'team_rec_yards': None,
                'team_rec_tds': None,
                'yards_market_share': None,
                'td_market_share': None,
                'dominator_rating': None,
                'dominator_pct': None,
                'status': 'no_college'
            })
            continue

        cfbd_school = normalize_school(college)
        if cfbd_school is None:
            print(f"  [{idx+1}/{len(all_wrs_df)}] {player_name} ({college}) - FCS school, skipping")
            results.append({
                'player_name': player_name,
                'draft_year': draft_year,
                'college': college,
                'pick': pick,
                'season_used': final_season,
                'player_rec_yards': None,
                'player_rec_tds': None,
                'team_rec_yards': None,
                'team_rec_tds': None,
                'yards_market_share': None,
                'td_market_share': None,
                'dominator_rating': None,
                'dominator_pct': None,
                'status': 'fcs_school'
            })
            continue

        # Check cache
        cache_key = f"{cfbd_school}_{final_season}"
        if cache_key not in team_cache:
            print(f"  [{idx+1}/{len(all_wrs_df)}] Fetching {cfbd_school} {final_season}...", end=" ")
            team_data = fetch_team_receiving_stats(cfbd_school, final_season)
            team_cache[cache_key] = team_data
            time.sleep(0.25)  # Rate limiting
        else:
            team_data = team_cache[cache_key]
            print(f"  [{idx+1}/{len(all_wrs_df)}] {player_name} ({cfbd_school} {final_season}) - Using cached data...", end=" ")

        if team_data is None:
            print(f"No data")
            results.append({
                'player_name': player_name,
                'draft_year': draft_year,
                'college': college,
                'pick': pick,
                'season_used': final_season,
                'player_rec_yards': None,
                'player_rec_tds': None,
                'team_rec_yards': None,
                'team_rec_tds': None,
                'yards_market_share': None,
                'td_market_share': None,
                'dominator_rating': None,
                'dominator_pct': None,
                'status': 'no_cfbd_data'
            })
            continue

        # Find player stats
        player_yards, player_tds, cfbd_name = find_player_stats(player_name, team_data)

        if player_yards is None:
            print(f"Player not found")
            results.append({
                'player_name': player_name,
                'draft_year': draft_year,
                'college': college,
                'pick': pick,
                'season_used': final_season,
                'player_rec_yards': None,
                'player_rec_tds': None,
                'team_rec_yards': team_data.get('team_rec_yards'),
                'team_rec_tds': team_data.get('team_rec_tds'),
                'yards_market_share': None,
                'td_market_share': None,
                'dominator_rating': None,
                'dominator_pct': None,
                'status': 'player_not_found'
            })
            continue

        # Calculate dominator
        team_yards = team_data.get('team_rec_yards', 0)
        team_tds = team_data.get('team_rec_tds', 0)

        yards_share, td_share, dominator = calculate_dominator(
            player_yards, player_tds, team_yards, team_tds
        )

        if dominator is not None:
            print(f"Found: {player_yards} yds, {player_tds} TDs -> {dominator*100:.1f}%")
        else:
            print(f"Could not calculate")

        results.append({
            'player_name': player_name,
            'draft_year': draft_year,
            'college': college,
            'pick': pick,
            'season_used': final_season,
            'player_rec_yards': player_yards,
            'player_rec_tds': player_tds,
            'team_rec_yards': team_yards,
            'team_rec_tds': team_tds,
            'yards_market_share': yards_share,
            'td_market_share': td_share,
            'dominator_rating': dominator,
            'dominator_pct': dominator * 100 if dominator else None,
            'status': 'calculated'
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_file = 'data/wr_dominator_complete.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nSaved results to {output_file}")

    # =========================================================================
    # SUMMARY AND ANALYSIS
    # =========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    total = len(results_df)
    calculated = results_df['dominator_rating'].notna().sum()

    print(f"\nTotal WRs: {total}")
    print(f"With dominator calculated: {calculated} ({100*calculated/total:.1f}%)")
    print(f"Missing: {total - calculated}")

    # By status
    print("\n--- By Status ---")
    for status, count in results_df['status'].value_counts().items():
        print(f"  {status}: {count}")

    # Distribution
    print("\n--- Dominator Distribution (Calculated Only) ---")
    calc_df = results_df[results_df['dominator_rating'].notna()]

    elite = (calc_df['dominator_pct'] >= 35).sum()
    strong = ((calc_df['dominator_pct'] >= 30) & (calc_df['dominator_pct'] < 35)).sum()
    solid = ((calc_df['dominator_pct'] >= 20) & (calc_df['dominator_pct'] < 30)).sum()
    peripheral = (calc_df['dominator_pct'] < 20).sum()

    print(f"  ≥35% (Elite): {elite} ({100*elite/len(calc_df):.1f}%)")
    print(f"  30-35% (Strong): {strong} ({100*strong/len(calc_df):.1f}%)")
    print(f"  20-29% (Solid): {solid} ({100*solid/len(calc_df):.1f}%)")
    print(f"  <20% (Peripheral): {peripheral} ({100*peripheral/len(calc_df):.1f}%)")

    # Sample calculations
    print("\n--- 20 Sample Calculations ---")
    sample = calc_df.sample(min(20, len(calc_df)), random_state=42)

    print(f"\n{'Player':<25} {'Rec Yds':>8} {'Rec TDs':>8} {'Team Yds':>10} {'Team TDs':>10} {'Yds%':>8} {'TD%':>8} {'DOM%':>8}")
    print("-" * 95)

    for _, row in sample.iterrows():
        print(f"{row['player_name']:<25} {row['player_rec_yards']:>8.0f} {row['player_rec_tds']:>8.0f} "
              f"{row['team_rec_yards']:>10.0f} {row['team_rec_tds']:>10.0f} "
              f"{row['yards_market_share']*100:>7.1f}% {row['td_market_share']*100:>7.1f}% {row['dominator_pct']:>7.1f}%")

    # List missing WRs
    print("\n--- WRs Missing Dominator ---")
    missing = results_df[results_df['dominator_rating'].isna()][['player_name', 'draft_year', 'pick', 'college', 'status']]
    print(missing.to_string(index=False))


if __name__ == "__main__":
    main()
