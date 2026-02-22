"""
SLAP Score V3 - Fetch Backtest Data

Fetches college stats from CFBD API for 2022-2024 draft classes.
Combines with NFLverse data (draft picks, combine, NFL fantasy stats).
"""

import os
import csv
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# School name mappings (NFLverse college names -> CFBD names)
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
    "San Jose St.": "San Jose State",
    "La.-Monroe": "Louisiana Monroe",
    "La.-Lafayette": "Louisiana",
    "Louisiana-Lafayette": "Louisiana",
    "UAB": "UAB",
    "TCU": "TCU",
    "SMU": "SMU",
    "BYU": "BYU",
    "UCF": "UCF",
    "UTSA": "UTSA",
    "UNLV": "UNLV",
    "USC": "USC",
    "LSU": "LSU",
    "UCLA": "UCLA",
    "Pitt": "Pittsburgh",
    "Ole Miss": "Ole Miss",
}


def normalize_school(school):
    """Convert NFLverse school name to CFBD format."""
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def fetch_team_receiving_stats(team, year):
    """Fetch receiving stats for a team from CFBD."""
    url = f"{BASE_URL}/stats/player/season"
    params = {
        "year": year,
        "category": "receiving",
        "team": team
    }

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        stats = {}
        for stat in response.json():
            player = stat.get("player", "").lower()
            stat_type = stat.get("statType", "")
            value = stat.get("stat", 0)

            if player not in stats:
                stats[player] = {}
            try:
                stats[player][stat_type] = int(float(value))
            except (ValueError, TypeError):
                stats[player][stat_type] = 0

        return stats
    elif response.status_code == 429:
        print(f"    Rate limited - waiting 60s...")
        time.sleep(60)
        return fetch_team_receiving_stats(team, year)
    else:
        return {}


def fetch_team_pass_attempts(team, year):
    """Get team pass attempts from CFBD."""
    url = f"{BASE_URL}/stats/season"
    params = {"year": year, "team": team}

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        for stat in response.json():
            if stat.get("statName") == "passAttempts":
                return int(stat.get("statValue", 0))
    return None


def names_match(name1, name2):
    """Check if two names refer to the same person."""
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    if n1 == n2:
        return True

    parts1 = n1.replace('.', ' ').replace("'", "").split()
    parts2 = n2.replace('.', ' ').replace("'", "").split()

    suffixes = {'jr', 'jr.', 'ii', 'iii', 'iv'}
    clean1 = [p for p in parts1 if p not in suffixes]
    clean2 = [p for p in parts2 if p not in suffixes]

    if not clean1 or not clean2:
        return False

    first1, first2 = clean1[0], clean2[0]
    if first1 != first2:
        # Allow initial match
        if not (len(first1) <= 2 and first2.startswith(first1[0])):
            if not (len(first2) <= 2 and first1.startswith(first2[0])):
                return False

    if len(clean1) > 1 and len(clean2) > 1:
        last1, last2 = clean1[-1], clean2[-1]
        if last1 != last2:
            return False

    return True


def find_player_receiving(player_name, team_stats):
    """Find a player's receiving yards in team stats."""
    if not team_stats:
        return None

    for api_name, stats in team_stats.items():
        if names_match(player_name, api_name):
            return stats.get("YDS")

    return None


def main():
    output_path = "data/backtest_college_stats.csv"

    print("=" * 60)
    print("FETCHING COLLEGE STATS FOR BACKTEST")
    print("=" * 60)
    print()

    # Load draft picks
    draft_df = pd.read_parquet("data/nflverse/draft_picks.parquet")
    draft_df = draft_df[(draft_df['season'].isin([2022, 2023, 2024])) &
                        (draft_df['position'].isin(['RB', 'WR']))]

    print(f"Found {len(draft_df)} RB/WR picks from 2022-2024")
    print()

    # Load combine data
    combine_df = pd.read_parquet("data/nflverse/combine.parquet")
    combine_df = combine_df[combine_df['pos'].isin(['RB', 'WR'])]

    # Results list
    results = []

    # Cache for team stats to minimize API calls
    team_cache = {}  # (team, year) -> receiving stats
    pass_att_cache = {}  # (team, year) -> pass attempts

    for draft_year in [2022, 2023, 2024]:
        college_year = draft_year - 1
        year_picks = draft_df[draft_df['season'] == draft_year]

        print(f"\n{'='*60}")
        print(f"DRAFT YEAR {draft_year} (College stats from {college_year})")
        print(f"{'='*60}")
        print(f"Processing {len(year_picks)} players...")

        found = 0
        not_found = 0

        for _, player in year_picks.iterrows():
            name = player['pfr_player_name']
            position = player['position']
            college = normalize_school(player['college'])
            pick = player['pick']
            age = player['age']

            if not college:
                not_found += 1
                continue

            # Get team receiving stats (cached)
            cache_key = (college, college_year)
            if cache_key not in team_cache:
                print(f"  Fetching {college} {college_year}...")
                team_cache[cache_key] = fetch_team_receiving_stats(college, college_year)
                time.sleep(0.1)  # Small delay to avoid rate limits

            team_stats = team_cache[cache_key]

            # Get team pass attempts (cached)
            if cache_key not in pass_att_cache:
                pass_att_cache[cache_key] = fetch_team_pass_attempts(college, college_year)
                time.sleep(0.1)

            team_pass_att = pass_att_cache[cache_key]

            # Find player receiving yards
            rec_yards = find_player_receiving(name, team_stats)

            # Get combine data
            player_combine = combine_df[
                (combine_df['season'] == draft_year) &
                (combine_df['player_name'].str.lower() == name.lower())
            ]

            weight = None
            forty = None

            if len(player_combine) > 0:
                weight = player_combine.iloc[0]['wt']
                forty = player_combine.iloc[0]['forty']

            if rec_yards:
                found += 1
            else:
                not_found += 1

            results.append({
                'draft_year': draft_year,
                'player_name': name,
                'position': position,
                'college': college,
                'pick': pick,
                'age': age,
                'rec_yards': rec_yards if rec_yards else '',
                'team_pass_attempts': team_pass_att if team_pass_att else '',
                'weight': weight if pd.notna(weight) else '',
                'forty': forty if pd.notna(forty) else ''
            })

        print(f"  Found stats: {found}, Missing: {not_found}")

    # Save results
    print(f"\n{'='*60}")
    print(f"SAVING TO {output_path}")
    print(f"{'='*60}")

    fieldnames = ['draft_year', 'player_name', 'position', 'college', 'pick',
                  'age', 'rec_yards', 'team_pass_attempts', 'weight', 'forty']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} players")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    results_df = pd.DataFrame(results)

    for year in [2022, 2023, 2024]:
        year_df = results_df[results_df['draft_year'] == year]
        with_yards = len(year_df[year_df['rec_yards'] != ''])
        with_weight = len(year_df[year_df['weight'] != ''])
        with_forty = len(year_df[year_df['forty'] != ''])
        total = len(year_df)

        print(f"\n{year} Draft Class ({total} players):")
        print(f"  Rec yards:  {with_yards}/{total} ({100*with_yards/total:.0f}%)")
        print(f"  Weight:     {with_weight}/{total} ({100*with_weight/total:.0f}%)")
        print(f"  40 time:    {with_forty}/{total} ({100*with_forty/total:.0f}%)")


if __name__ == "__main__":
    main()
