"""
Fetch WR College Stats from CFBD API for 2014-2017 seasons.

These seasons are missing from our existing data and needed for:
- 2015 draft class (2014 final season)
- 2016 draft class (2015 final season) - partial data exists
- 2017 draft class (2016 final season)
- 2018 draft class (2017 final season)
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
    "Weber St.": None,  # FCS
    "North Dakota St.": None,  # FCS
    "Youngstown St.": None,  # FCS
    "Illinois St.": None,  # FCS
}


def normalize_school(school):
    """Convert school name to CFBD format."""
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def fetch_player_receiving_stats(team, year):
    """Fetch receiving stats for all WRs on a team."""
    url = f"{BASE_URL}/stats/player/season"
    params = {
        "year": year,
        "category": "receiving",
        "team": team
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            stats = {}
            for stat in response.json():
                player = stat.get("player", "").strip()
                stat_type = stat.get("statType", "")
                value = stat.get("stat", 0)

                if player not in stats:
                    stats[player] = {"player_name": player}
                try:
                    stats[player][stat_type] = int(float(value))
                except (ValueError, TypeError):
                    stats[player][stat_type] = 0
            return stats
        elif response.status_code == 429:
            print(f"    Rate limited, waiting 5s...")
            time.sleep(5)
            return fetch_player_receiving_stats(team, year)  # Retry
        else:
            print(f"    Error {response.status_code} for {team} {year}")
            return {}
    except Exception as e:
        print(f"    Error fetching {team} {year}: {e}")
        return {}


def fetch_team_stats(team, year):
    """Fetch team total pass attempts from CFBD."""
    url = f"{BASE_URL}/stats/season"
    params = {
        "year": year,
        "team": team
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            team_stats = {}
            for stat in data:
                stat_name = stat.get("statName")
                stat_value = stat.get("statValue", 0)
                if stat_name in ["passAttempts", "passingYards", "passCompletions"]:
                    try:
                        team_stats[stat_name] = float(stat_value)
                    except:
                        team_stats[stat_name] = 0
            return team_stats
        elif response.status_code == 429:
            print(f"    Rate limited, waiting 5s...")
            time.sleep(5)
            return fetch_team_stats(team, year)  # Retry
        else:
            return {}
    except Exception as e:
        print(f"    Error fetching team stats {team} {year}: {e}")
        return {}


def names_match(name1, name2):
    """Check if two names refer to the same person."""
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    if n1 == n2:
        return True

    # Remove suffixes and compare
    parts1 = n1.replace('.', ' ').replace("'", "").split()
    parts2 = n2.replace('.', ' ').replace("'", "").split()

    suffixes = {'jr', 'jr.', 'ii', 'iii', 'iv', 'sr'}
    clean1 = [p for p in parts1 if p not in suffixes]
    clean2 = [p for p in parts2 if p not in suffixes]

    if not clean1 or not clean2:
        return False

    # Compare first name (allow initial match)
    first1, first2 = clean1[0], clean2[0]
    if first1 != first2:
        if not (len(first1) <= 2 and first2.startswith(first1[0])):
            if not (len(first2) <= 2 and first1.startswith(first2[0])):
                return False

    # Compare last name
    if len(clean1) > 1 and len(clean2) > 1:
        last1, last2 = clean1[-1], clean2[-1]
        if last1 != last2:
            return False

    return True


def find_player_in_team_stats(player_name, team_stats):
    """Find a player's receiving stats in team data."""
    if not team_stats:
        return None, None, None, None

    for api_name, stats in team_stats.items():
        if names_match(player_name, api_name):
            yards = stats.get("YDS", 0)
            receptions = stats.get("REC", 0)
            tds = stats.get("TD", 0)
            return yards, receptions, tds, api_name

    return None, None, None, None


def main():
    print("=" * 90)
    print("FETCHING WR COLLEGE STATS FROM CFBD API (2014-2017)")
    print("=" * 90)

    # Load WR backtest data
    wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
    print(f"\nLoaded {len(wr_backtest)} WRs from backtest")

    # Get draft picks data for ages
    draft = pd.read_parquet('data/nflverse/draft_picks.parquet')
    wr_draft = draft[(draft['position'] == 'WR') & (draft['season'] >= 2015) & (draft['season'] <= 2024)]

    # Filter to WRs who need 2014-2017 seasons
    target_draft_years = [2015, 2016, 2017, 2018]
    target_wrs = wr_backtest[wr_backtest['draft_year'].isin(target_draft_years)].copy()
    print(f"WRs drafted 2015-2018 (needing 2014-2017 data): {len(target_wrs)}")

    # Add draft age from nflverse
    target_wrs = target_wrs.merge(
        wr_draft[['season', 'pfr_player_name', 'age', 'college']].rename(
            columns={'season': 'draft_year', 'pfr_player_name': 'player_name', 'age': 'draft_age', 'college': 'nfl_college'}
        ),
        on=['player_name', 'draft_year'],
        how='left'
    )

    # Calculate final college season
    target_wrs['final_season'] = target_wrs['draft_year'] - 1

    results = []

    # Process each WR
    for idx, row in target_wrs.iterrows():
        player_name = row['player_name']
        draft_year = row['draft_year']
        final_season = row['final_season']
        college = row.get('college') or row.get('nfl_college')
        draft_age = row.get('draft_age')

        if pd.isna(college):
            print(f"  Skipping {player_name} - no college")
            continue

        cfbd_school = normalize_school(college)
        if cfbd_school is None:
            print(f"  Skipping {player_name} ({college}) - FCS school")
            continue

        print(f"  Fetching {player_name} ({cfbd_school} {final_season})...", end=" ")

        # Fetch player receiving stats
        team_receiving = fetch_player_receiving_stats(cfbd_school, final_season)
        time.sleep(0.3)  # Rate limiting

        # Find this player
        rec_yards, receptions, rec_tds, cfbd_name = find_player_in_team_stats(player_name, team_receiving)

        # Fetch team stats
        team_stats = fetch_team_stats(cfbd_school, final_season)
        time.sleep(0.3)  # Rate limiting

        team_pass_att = team_stats.get("passAttempts")

        if rec_yards is not None:
            print(f"Found: {rec_yards} yards, {team_pass_att} team PA")
        else:
            print(f"Not found in CFBD")

        results.append({
            'player_name': player_name,
            'draft_year': draft_year,
            'final_season': final_season,
            'college': college,
            'cfbd_school': cfbd_school,
            'cfbd_name': cfbd_name,
            'draft_age': draft_age,
            'rec_yards': rec_yards,
            'receptions': receptions,
            'rec_tds': rec_tds,
            'team_pass_att': team_pass_att
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_file = 'data/wr_college_stats_2014_2017.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nSaved results to {output_file}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"\nTotal WRs processed: {len(results_df)}")
    print(f"WRs with rec_yards found: {results_df['rec_yards'].notna().sum()}")
    print(f"WRs with team_pass_att found: {results_df['team_pass_att'].notna().sum()}")

    print("\nBy final season:")
    for season in sorted(results_df['final_season'].unique()):
        season_data = results_df[results_df['final_season'] == season]
        found = season_data['rec_yards'].notna().sum()
        print(f"  {season}: {found}/{len(season_data)} found")

    print("\nSample rows (first 10):")
    print(results_df[['player_name', 'final_season', 'draft_age', 'rec_yards', 'team_pass_att']].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
