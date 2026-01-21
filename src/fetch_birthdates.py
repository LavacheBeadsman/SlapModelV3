"""
SLAP Score V3 - Birthdate Fetching Script

This script pulls REAL birthdates from Wikipedia for our prospects.
No estimation or guessing - only actual data from Wikipedia.

If a player's birthdate is not found, we flag it as missing.
"""

import re
import requests
import time

# Wikipedia API settings
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "SLAPScoreV3/1.0 (Fantasy Football Research)"


def get_wikipedia_birthdate(player_name):
    """
    Fetch a player's birthdate from Wikipedia.

    How it works:
    1. Convert player name to Wikipedia page title format
    2. Call the Wikipedia API to get the page content
    3. Look for the birthdate pattern in the infobox

    Returns:
    - Dictionary with 'birthdate' (YYYY-MM-DD format) and 'source' (Wikipedia URL)
    - Or None if not found
    """

    # Convert name to Wikipedia title format (spaces to underscores)
    wiki_title = player_name.replace(" ", "_")

    # Build the API request
    params = {
        "action": "query",
        "titles": wiki_title,
        "prop": "revisions",
        "rvprop": "content",
        "format": "json",
        "rvslots": "main"
    }

    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(WIKI_API, params=params, headers=headers, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        # Get the page content
        for page_id, page_data in pages.items():
            if page_id == "-1":  # Page not found
                return None

            revisions = page_data.get("revisions", [])
            if not revisions:
                return None

            content = revisions[0].get("slots", {}).get("main", {}).get("*", "")

            # Look for birthdate patterns in Wikipedia markup
            # Pattern 1: {{birth date and age|YYYY|MM|DD|...}}
            pattern1 = r'\{\{birth date and age\|(\d{4})\|(\d{1,2})\|(\d{1,2})'
            match = re.search(pattern1, content, re.IGNORECASE)

            if match:
                year, month, day = match.groups()
                birthdate = f"{year}-{int(month):02d}-{int(day):02d}"
                return {
                    "birthdate": birthdate,
                    "source": f"https://en.wikipedia.org/wiki/{wiki_title}"
                }

            # Pattern 2: {{birth date|YYYY|MM|DD|...}}
            pattern2 = r'\{\{birth date\|(\d{4})\|(\d{1,2})\|(\d{1,2})'
            match = re.search(pattern2, content, re.IGNORECASE)

            if match:
                year, month, day = match.groups()
                birthdate = f"{year}-{int(month):02d}-{int(day):02d}"
                return {
                    "birthdate": birthdate,
                    "source": f"https://en.wikipedia.org/wiki/{wiki_title}"
                }

    except Exception as e:
        print(f"  Error fetching {player_name}: {e}")
        return None

    return None


def calculate_age_at_season(birthdate_str, season_year=2024):
    """
    Calculate a player's age during a college football season.

    We use September 1 of the season year as the reference date,
    since that's roughly when the season starts.

    Returns: age as an integer
    """
    from datetime import date

    # Parse birthdate
    year, month, day = map(int, birthdate_str.split("-"))
    birthdate = date(year, month, day)

    # Season reference date (September 1 of season year)
    season_start = date(season_year, 9, 1)

    # Calculate age
    age = season_start.year - birthdate.year

    # Adjust if birthday hasn't happened yet by season start
    if (season_start.month, season_start.day) < (birthdate.month, birthdate.day):
        age -= 1

    return age


# ======================
# FULL RUN: All prospects
# ======================

if __name__ == "__main__":
    import csv
    from datetime import date

    # File paths
    input_path = "data/prospects_with_stats.csv"
    output_path = "data/prospects_with_birthdates.csv"
    missing_log_path = "data/missing_birthdates.txt"

    print("=" * 60)
    print("WIKIPEDIA BIRTHDATE FETCH - Full Prospect List")
    print("=" * 60)
    print()

    # Read the prospects CSV
    print(f"Reading prospects from: {input_path}")
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        prospects = list(reader)
    print(f"  Found {len(prospects)} prospects")

    # Check for duplicates
    names = [p['player_name'] for p in prospects]
    duplicates = [name for name in names if names.count(name) > 1]
    if duplicates:
        print(f"\n  WARNING: Found duplicate names: {set(duplicates)}")
    else:
        print("  No duplicates found")
    print()

    # Track results
    results = []
    missing_players = []
    found_count = 0
    missing_count = 0

    print("Fetching birthdates from Wikipedia...")
    print("-" * 60)

    for i, prospect in enumerate(prospects):
        name = prospect['player_name']

        # Progress indicator (every 20 players)
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(prospects)} players processed")

        # Small delay to be respectful to Wikipedia's servers
        time.sleep(0.3)

        # Fetch birthdate
        wiki_result = get_wikipedia_birthdate(name)

        if wiki_result:
            birthdate = wiki_result["birthdate"]
            age_2024 = calculate_age_at_season(birthdate, 2024)
            found_count += 1

            # Add birthdate and age to prospect data
            prospect['birthdate'] = birthdate
            prospect['age'] = age_2024  # Replace estimated age with real age
        else:
            # Mark as MISSING - do NOT guess
            missing_count += 1
            missing_players.append(name)
            prospect['birthdate'] = "MISSING"
            prospect['age'] = "MISSING"

        results.append(prospect)

    print("-" * 60)
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total players: {len(prospects)}")
    print(f"  Birthdates found: {found_count} ({100*found_count/len(prospects):.1f}%)")
    print(f"  Birthdates missing: {missing_count} ({100*missing_count/len(prospects):.1f}%)")
    print()

    # Log missing players
    if missing_players:
        print(f"Players with MISSING birthdates ({len(missing_players)}):")
        for name in missing_players:
            print(f"  - {name}")
        print()

        # Save missing players to a separate file
        with open(missing_log_path, 'w') as f:
            f.write("Players with missing birthdates (need manual lookup):\n")
            f.write("=" * 50 + "\n\n")
            for name in missing_players:
                f.write(f"- {name}\n")
        print(f"Missing players log saved to: {missing_log_path}")
        print()

    # Save to CSV
    print(f"Saving results to: {output_path}")
    fieldnames = ['player_name', 'position', 'school', 'projected_pick',
                  'rec_yards', 'team_pass_attempts', 'birthdate', 'age', 'weight']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Done!")
    print()

    # Show first 10 and last 10 rows
    print("=" * 60)
    print("FIRST 10 ROWS")
    print("=" * 60)
    print(f"{'Player':<25} {'Birthdate':<12} {'Age':<6} {'School'}")
    print("-" * 60)
    for r in results[:10]:
        print(f"{r['player_name']:<25} {r['birthdate']:<12} {str(r['age']):<6} {r['school']}")

    print()
    print("=" * 60)
    print("LAST 10 ROWS")
    print("=" * 60)
    print(f"{'Player':<25} {'Birthdate':<12} {'Age':<6} {'School'}")
    print("-" * 60)
    for r in results[-10:]:
        print(f"{r['player_name']:<25} {r['birthdate']:<12} {str(r['age']):<6} {r['school']}")
