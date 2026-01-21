"""
SLAP Score V3 - Data Fetching Script

This script pulls player data from the CollegeFootballData.com API
and combines it with our mock draft projections.

What it does (in plain English):
1. Reads our list of 161 prospects from the mock draft CSV
2. For each player, asks the API for their stats
3. Saves everything into one clean CSV file
"""

import os
import csv
import requests
from dotenv import load_dotenv

# Load the API key from the .env file
# This keeps our secret key out of the code
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")

# The base URL for all API requests
BASE_URL = "https://api.collegefootballdata.com"

# Headers that we send with every API request
# The API requires us to include our key here
HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}


def search_player(player_name):
    """
    Search for a player by name and get their basic info.

    Returns: dictionary with id, team, weight, height, position
    Or None if player not found
    """
    url = f"{BASE_URL}/player/search"
    params = {"searchTerm": player_name}

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        results = response.json()
        if results:
            # Return the first match
            return results[0]

    return None


def get_receiving_yards(player_name, team, year=2025):
    """
    Get a player's receiving yards for a specific season.

    The API returns stats in a weird format - each stat type is a separate row.
    So we need to find the row where statType = "YDS" (yards).

    Returns: receiving yards as an integer, or None if not found
    """
    url = f"{BASE_URL}/stats/player/season"
    params = {
        "year": year,
        "category": "receiving",
        "team": team
    }

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        all_stats = response.json()

        # Loop through all the stats looking for our player's receiving yards
        for stat in all_stats:
            # Check if this is our player AND it's the yards stat
            if stat.get("player", "").lower() == player_name.lower():
                if stat.get("statType") == "YDS":
                    return int(stat.get("stat", 0))

    return None


def get_team_pass_attempts(team, year=2025):
    """
    Get a team's total pass attempts for a season.

    This is used to calculate production rate:
    production = receiving_yards / team_pass_attempts

    A player with 1000 yards on a team that threw 600 times
    is more impressive than 1000 yards on a team that threw 400 times.

    Returns: pass attempts as an integer, or None if not found
    """
    url = f"{BASE_URL}/stats/season"
    params = {
        "year": year,
        "team": team
    }

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        all_stats = response.json()

        # Find the passAttempts stat
        for stat in all_stats:
            if stat.get("statName") == "passAttempts":
                return int(stat.get("statValue", 0))

    return None


def estimate_age(player_name, position, season_year=2025):
    """
    Estimate a player's age during a season based on when they were recruited.

    How it works:
    - Most recruits are 18 years old when they sign
    - If recruited in 2023, they were ~18 in 2023, so ~19 in 2024
    - If recruited in 2022, they were ~18 in 2022, so ~20 in 2024
    - And so on...

    This isn't perfect (some players are older/younger), but it's
    a reasonable estimate when we don't have actual birthdates.

    Returns: estimated age as an integer, or None if not found
    """
    # Search recruiting database for years 2021-2025
    for recruit_year in range(2025, 2020, -1):
        url = f"{BASE_URL}/recruiting/players"
        params = {
            "year": recruit_year,
            "position": position
        }

        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code == 200:
            recruits = response.json()

            # Look for our player
            for recruit in recruits:
                recruit_name = recruit.get("name", "").lower()
                search_name = player_name.lower()

                # Check if names match (handle slight variations)
                if recruit_name == search_name or search_name in recruit_name:
                    # Calculate age: assume 18 at recruitment
                    age_at_recruit = 18
                    years_since = season_year - recruit_year
                    estimated_age = age_at_recruit + years_since
                    return estimated_age

    return None


# ======================
# CACHING: Store data we've already fetched to avoid repeat API calls
# ======================

# Cache for team pass attempts (so we don't fetch same team twice)
_team_pass_attempts_cache = {}

# Cache for recruiting data (loaded once, searched many times)
_recruiting_cache = {}


def get_team_pass_attempts_cached(team, year=2025):
    """Same as get_team_pass_attempts but uses cache to avoid repeat calls."""
    cache_key = f"{team}_{year}"
    if cache_key not in _team_pass_attempts_cache:
        _team_pass_attempts_cache[cache_key] = get_team_pass_attempts(team, year)
    return _team_pass_attempts_cache[cache_key]


def load_recruiting_cache(positions, years=range(2021, 2026)):
    """
    Load all recruiting data upfront so we don't have to make
    separate API calls for each player's age.
    """
    print("Loading recruiting data (this helps speed up age lookups)...")
    for year in years:
        for pos in positions:
            url = f"{BASE_URL}/recruiting/players"
            params = {"year": year, "position": pos}
            response = requests.get(url, headers=HEADERS, params=params)
            if response.status_code == 200:
                for recruit in response.json():
                    name = recruit.get("name", "").lower()
                    _recruiting_cache[name] = year
    print(f"  Loaded {len(_recruiting_cache)} recruits into cache\n")


def estimate_age_cached(player_name, season_year=2025):
    """
    Fast age estimation using the pre-loaded cache.
    """
    search_name = player_name.lower()

    # Try exact match first
    if search_name in _recruiting_cache:
        recruit_year = _recruiting_cache[search_name]
        return 18 + (season_year - recruit_year)

    # Try partial match
    for cached_name, recruit_year in _recruiting_cache.items():
        if search_name in cached_name or cached_name in search_name:
            return 18 + (season_year - recruit_year)

    return None


# ======================
# MAIN FUNCTION: Fetch data for all prospects
# ======================

def fetch_all_prospects(mock_draft_path, output_path, year=2025):
    """
    The main function that does everything:
    1. Reads your mock draft CSV
    2. For each player, fetches their data from the API
    3. Saves everything to a new CSV

    Parameters:
    - mock_draft_path: path to mock_draft_2026.csv
    - output_path: where to save the combined data
    - year: which college season to pull stats from (2025 for 2026 draft prospects)
    """

    # Load recruiting data into cache first (speeds up age lookups)
    load_recruiting_cache(["WR", "RB", "ATH"])

    # Read the mock draft CSV
    print(f"Reading mock draft from: {mock_draft_path}")
    with open(mock_draft_path, 'r') as f:
        reader = csv.DictReader(f)
        prospects = list(reader)
    print(f"  Found {len(prospects)} prospects\n")

    # Prepare output data
    output_data = []
    players_found = 0
    players_missing = 0

    print("Fetching data for each player...")
    print("-" * 50)

    for i, prospect in enumerate(prospects):
        name = prospect['player_name']
        position = prospect['position']
        projected_pick = prospect['projected_pick']

        # Progress indicator (every 10 players)
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(prospects)} players processed")

        # Get player info (team, weight)
        info = search_player(name)

        if info:
            team = info.get('team')
            weight = info.get('weight')

            # Get receiving yards
            rec_yards = get_receiving_yards(name, team, year)

            # Get team pass attempts (cached)
            team_pass_att = get_team_pass_attempts_cached(team, year)

            # Get estimated age (cached)
            age = estimate_age_cached(name, season_year=year)

            # Add to output
            output_data.append({
                'player_name': name,
                'position': position,
                'school': team,
                'projected_pick': projected_pick,
                'rec_yards': rec_yards if rec_yards else '',
                'team_pass_attempts': team_pass_att if team_pass_att else '',
                'age': age if age else '',
                'weight': weight if weight else ''
            })
            players_found += 1
        else:
            # Player not found in API - still include them with empty data
            output_data.append({
                'player_name': name,
                'position': position,
                'school': '',
                'projected_pick': projected_pick,
                'rec_yards': '',
                'team_pass_attempts': '',
                'age': '',
                'weight': ''
            })
            players_missing += 1

    print("-" * 50)
    print(f"\nResults:")
    print(f"  Players found in API: {players_found}")
    print(f"  Players not found: {players_missing}")

    # Save to CSV
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['player_name', 'position', 'school', 'projected_pick',
                      'rec_yards', 'team_pass_attempts', 'age', 'weight']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)

    print("Done!")
    return output_data


# ======================
# RUN THE SCRIPT
# ======================

if __name__ == "__main__":
    import time

    # Paths
    mock_draft_path = "data/mock_draft_2026.csv"
    output_path = "data/prospects_with_stats.csv"

    print("=" * 50)
    print("SLAP Score V3 - Data Fetching")
    print("=" * 50)
    print()

    start_time = time.time()
    fetch_all_prospects(mock_draft_path, output_path)
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.1f} seconds")
