"""
SLAP Score V3 - Fill Missing Ages with Recruiting Class Estimates

This script:
1. Reads the current CSV with birthdates
2. Keeps confirmed birthdates from Wikipedia (age_estimated = FALSE)
3. For MISSING birthdates, estimates age from recruiting class year (age_estimated = TRUE)
4. Adds the 14 birthdates we found from web searches
5. Fixes Chris Brazzell (wrong 1976 birthdate)
"""

import os
import csv
import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Birthdates found from web searches (not on Wikipedia)
WEB_SEARCH_BIRTHDATES = {
    "Omar Cooper Jr.": ("2003-12-14", 20),
    "Noah Thomas": ("2004-01-13", 20),
    "Aaron Anderson": ("2002-12-02", 21),
    "Chris Hilton": ("2002-10-11", 21),
    "Eli Heidenreich": ("2002-05-15", 22),
    "Harrison Wallace III": ("2003-04-23", 21),
    "J.Michael Sturdivant": ("2002-09-06", 22),
    "Emmanuel Henderson": ("2003-03-21", 21),
    "Trebor Pena": ("2002-02-21", 22),
    "Antonio Gates Jr.": ("2003-10-06", 20),
    "Braylin Presley": ("2003-10-24", 20),
    "Dacari Collins": ("2002-11-04", 21),
    "Nick Singleton": ("2004-01-06", 20),
    "E.J. Smith": ("2002-05-15", 22),
}

# Players with wrong Wikipedia matches (need to be marked as MISSING)
WRONG_MATCHES = ["Chris Brazzell"]


def load_recruiting_data():
    """
    Load recruiting class data from CFBD API.
    Returns a dictionary: {player_name_lowercase: recruiting_year}
    """
    print("Loading recruiting data from CFBD API...")
    recruiting_cache = {}

    for year in range(2019, 2025):
        for pos in ["WR", "RB", "ATH"]:
            url = f"{BASE_URL}/recruiting/players"
            params = {"year": year, "position": pos}

            response = requests.get(url, headers=HEADERS, params=params)

            if response.status_code == 200:
                for recruit in response.json():
                    name = recruit.get("name", "").lower()
                    recruiting_cache[name] = year

    print(f"  Loaded {len(recruiting_cache)} recruits\n")
    return recruiting_cache


def estimate_age_from_recruiting(player_name, recruiting_cache, season_year=2024):
    """
    Estimate age based on recruiting class year.
    Assumes player was 18 in their recruiting year.
    """
    search_name = player_name.lower()

    # Try exact match
    if search_name in recruiting_cache:
        recruit_year = recruiting_cache[search_name]
        return 18 + (season_year - recruit_year)

    # Try partial match
    for cached_name, recruit_year in recruiting_cache.items():
        if search_name in cached_name or cached_name in search_name:
            return 18 + (season_year - recruit_year)

    return None


def main():
    input_path = "data/prospects_with_birthdates.csv"
    output_path = "data/prospects_final.csv"

    print("=" * 60)
    print("FILLING MISSING AGES WITH RECRUITING CLASS ESTIMATES")
    print("=" * 60)
    print()

    # Load recruiting data
    recruiting_cache = load_recruiting_data()

    # Read current CSV
    print(f"Reading: {input_path}")
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        prospects = list(reader)
    print(f"  Found {len(prospects)} prospects\n")

    # Process each player
    results = []
    confirmed_count = 0
    estimated_count = 0
    web_search_count = 0
    still_missing_count = 0
    fixed_wrong_match = 0

    print("Processing players...")
    print("-" * 60)

    for prospect in prospects:
        name = prospect['player_name']
        current_birthdate = prospect.get('birthdate', 'MISSING')
        current_age = prospect.get('age', 'MISSING')

        # Start with current data
        new_row = dict(prospect)

        # Check if this is a wrong Wikipedia match that needs fixing
        if name in WRONG_MATCHES:
            new_row['birthdate'] = "MISSING"
            new_row['age'] = "MISSING"
            current_birthdate = "MISSING"
            current_age = "MISSING"
            fixed_wrong_match += 1
            print(f"  Fixed wrong match: {name}")

        # Check if we found birthdate in web search
        if name in WEB_SEARCH_BIRTHDATES:
            birthdate, age = WEB_SEARCH_BIRTHDATES[name]
            new_row['birthdate'] = birthdate
            new_row['age'] = age
            new_row['age_estimated'] = "FALSE"
            web_search_count += 1
            results.append(new_row)
            continue

        # If we have confirmed birthdate from Wikipedia
        if current_birthdate != "MISSING" and current_age != "MISSING":
            new_row['age_estimated'] = "FALSE"
            confirmed_count += 1
            results.append(new_row)
            continue

        # Try to estimate from recruiting class
        estimated_age = estimate_age_from_recruiting(name, recruiting_cache)

        if estimated_age:
            new_row['birthdate'] = "MISSING"  # We don't know actual birthdate
            new_row['age'] = estimated_age
            new_row['age_estimated'] = "TRUE"
            estimated_count += 1
        else:
            new_row['birthdate'] = "MISSING"
            new_row['age'] = "MISSING"
            new_row['age_estimated'] = "MISSING"
            still_missing_count += 1
            print(f"  Could not find recruiting data: {name}")

        results.append(new_row)

    print("-" * 60)
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Confirmed birthdates (Wikipedia): {confirmed_count}")
    print(f"  Confirmed birthdates (Web search): {web_search_count}")
    print(f"  Estimated from recruiting class:  {estimated_count}")
    print(f"  Still missing (no data found):    {still_missing_count}")
    print(f"  Fixed wrong matches:              {fixed_wrong_match}")
    print()
    print(f"  Total with age data: {confirmed_count + web_search_count + estimated_count}")
    print(f"  Total missing:       {still_missing_count}")
    print()

    # Save to CSV
    print(f"Saving to: {output_path}")
    fieldnames = ['player_name', 'position', 'school', 'projected_pick',
                  'rec_yards', 'team_pass_attempts', 'birthdate', 'age',
                  'age_estimated', 'weight']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Done!")
    print()

    # Show sample data
    print("=" * 60)
    print("SAMPLE DATA (first 15 rows)")
    print("=" * 60)
    print(f"{'Player':<25} {'Age':<6} {'Estimated?':<12} {'Birthdate'}")
    print("-" * 60)
    for r in results[:15]:
        age = str(r.get('age', ''))
        estimated = r.get('age_estimated', '')
        birthdate = r.get('birthdate', '')
        print(f"{r['player_name']:<25} {age:<6} {estimated:<12} {birthdate}")


if __name__ == "__main__":
    main()
