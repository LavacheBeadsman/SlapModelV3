"""
Fetch team receiving yards from CFBD API and calculate Dominator Rating for WRs.

Dominator Rating = player_rec_yards / team_receiving_yards
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

# Map school names to CFBD names
SCHOOL_NAME_MAP = {
    "Boston Col.": "Boston College",
    "Mississippi": "Ole Miss",
    "Ala-Birmingham": "UAB",
    "UT Martin": "UT Martin",  # FCS - may not be available
    "SE Missouri St.": "Southeast Missouri State",
    "Central Michigan": "Central Michigan",
    "South Dakota St.": "South Dakota State",
    "New Mexico St.": "New Mexico State",
    # FCS schools that may not have data
    "Rhode Island": None,
    "Charleston (WV)": None,
    "North Dakota State": None,
    "Samford": None,
    "Lenoir-Rhyne": None,
    "Princeton": None,
}


def get_cfbd_name(school):
    """Get CFBD-compatible school name."""
    if school in SCHOOL_NAME_MAP:
        return SCHOOL_NAME_MAP[school]
    return school


def get_team_receiving_yards(team, season, retries=3):
    """Fetch team receiving yards (netPassingYards) from CFBD with retries."""
    cfbd_name = get_cfbd_name(team)
    if cfbd_name is None:
        return None  # FCS school without CFBD data

    url = f"{BASE_URL}/stats/season"
    params = {
        "year": season,
        "team": cfbd_name
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for stat in data:
                    if stat.get("statName") == "netPassingYards":
                        return float(stat.get("statValue", 0))
                return None  # No netPassingYards found
            elif response.status_code == 503:
                print(f"  (retry {attempt+1}/{retries})...", end=" ")
                time.sleep(2 ** (attempt + 1))  # Exponential backoff
            else:
                print(f"  Warning: HTTP {response.status_code} for {cfbd_name} {season}")
                return None
        except Exception as e:
            print(f"  Error fetching {cfbd_name} {season}: {e}")
            time.sleep(2)

    return None


def main():
    print("=" * 70)
    print("FETCHING TEAM RECEIVING YARDS FOR DOMINATOR RATING")
    print("=" * 70)
    print()

    # Load backtest data
    bt = pd.read_csv("data/backtest_college_stats.csv")

    # Filter to WRs only (2020-2023 for hit rate analysis)
    wrs = bt[(bt['position'] == 'WR') & (bt['draft_year'] <= 2023)].copy()
    print(f"WRs to process: {len(wrs)}")

    # Get unique school/season combinations
    # Season = draft_year - 1 (player's final college season)
    wrs['season'] = wrs['draft_year'] - 1

    unique_combos = wrs[['college', 'season']].drop_duplicates()
    print(f"Unique school/season combinations: {len(unique_combos)}")
    print()

    # Fetch team receiving yards
    team_rec_yards = {}

    for idx, row in unique_combos.iterrows():
        school = row['college']
        season = int(row['season'])
        key = (school, season)

        cfbd_name = get_cfbd_name(school)
        if cfbd_name is None:
            print(f"Skipping {school} {season} (FCS school)")
            continue

        print(f"Fetching {cfbd_name} {season}...", end=" ")
        rec_yards = get_team_receiving_yards(school, season)

        if rec_yards is not None:
            team_rec_yards[key] = rec_yards
            print(f"{rec_yards}")
        else:
            print("NOT FOUND")

        # Rate limit to avoid hitting API limits
        time.sleep(0.4)

    print()
    print(f"Successfully fetched: {len(team_rec_yards)} / {len(unique_combos)}")

    # Calculate Dominator Rating for each WR
    print()
    print("=" * 70)
    print("CALCULATING DOMINATOR RATINGS")
    print("=" * 70)

    dominator_data = []

    for idx, row in wrs.iterrows():
        school = row['college']
        season = int(row['draft_year'] - 1)
        key = (school, season)

        player_rec_yards = row['rec_yards']
        team_yards = team_rec_yards.get(key)

        if pd.notna(player_rec_yards) and team_yards is not None and team_yards > 0:
            dominator = (player_rec_yards / team_yards) * 100  # As percentage
        else:
            dominator = None

        dominator_data.append({
            'player_name': row['player_name'],
            'draft_year': row['draft_year'],
            'college': school,
            'season': season,
            'player_rec_yards': player_rec_yards,
            'team_rec_yards': team_yards,
            'dominator_rating': dominator
        })

    # Save results
    dom_df = pd.DataFrame(dominator_data)
    dom_df.to_csv("data/wr_dominator_ratings.csv", index=False)

    print()
    print(f"Saved to data/wr_dominator_ratings.csv")

    # Show summary
    valid = dom_df['dominator_rating'].notna().sum()
    print(f"Valid dominator ratings: {valid} / {len(dom_df)}")

    # Show top dominators
    print()
    print("TOP 10 DOMINATOR RATINGS:")
    top = dom_df.nlargest(10, 'dominator_rating')
    for _, r in top.iterrows():
        print(f"  {r['player_name']}: {r['dominator_rating']:.1f}% ({r['player_rec_yards']:.0f} / {r['team_rec_yards']:.0f})")

    # Show missing
    missing = dom_df[dom_df['dominator_rating'].isna()]
    if len(missing) > 0:
        print()
        print(f"MISSING TEAM DATA ({len(missing)} players):")
        for _, r in missing.iterrows():
            print(f"  {r['player_name']} - {r['college']} {r['season']}")


if __name__ == "__main__":
    main()
