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
# TEST: 5 players first
# ======================

if __name__ == "__main__":
    test_players = [
        "Tetairoa McMillan",
        "Jordyn Tyson",
        "Jeremiyah Love",
        "Carnell Tate",
        "Makai Lemon"
    ]

    print("=" * 60)
    print("WIKIPEDIA BIRTHDATE TEST - 5 Players")
    print("=" * 60)
    print()

    results = []

    for name in test_players:
        print(f"Looking up: {name}")

        # Small delay to be respectful to Wikipedia's servers
        time.sleep(0.5)

        result = get_wikipedia_birthdate(name)

        if result:
            birthdate = result["birthdate"]
            age_2024 = calculate_age_at_season(birthdate, 2024)
            source = result["source"]

            print(f"  Birthdate: {birthdate}")
            print(f"  Age during 2024 season: {age_2024}")
            print(f"  Source: {source}")

            results.append({
                "player": name,
                "birthdate": birthdate,
                "age_2024": age_2024,
                "status": "FOUND",
                "source": source
            })
        else:
            print(f"  Status: NOT FOUND on Wikipedia")
            results.append({
                "player": name,
                "birthdate": None,
                "age_2024": None,
                "status": "NOT FOUND",
                "source": None
            })

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    found = sum(1 for r in results if r["status"] == "FOUND")
    print(f"Found: {found}/{len(test_players)}")
    print()

    print("Results table:")
    print(f"{'Player':<25} {'Birthdate':<12} {'Age (2024)':<10} {'Status'}")
    print("-" * 60)
    for r in results:
        birthdate = r["birthdate"] or "—"
        age = str(r["age_2024"]) if r["age_2024"] else "—"
        print(f"{r['player']:<25} {birthdate:<12} {age:<10} {r['status']}")
