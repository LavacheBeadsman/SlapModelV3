"""Find missing receiving yards for backtest players."""

import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# The 18 missing players with their draft year (college year = draft year - 1)
missing_players = [
    # 2022 draft (college 2021)
    ("Christian Watson", "North Dakota State", 2021, "WR"),
    ("Pierre Strong", "South Dakota State", 2021, "RB"),
    ("Snoop Conner", "Ole Miss", 2021, "RB"),
    ("Dareke Young", "Lenoir-Rhyne", 2021, "WR"),  # FCS school
    ("Isiah Pacheco", "Rutgers", 2021, "RB"),
    ("Zander Horvath", "Purdue", 2021, "RB"),

    # 2023 draft (college 2022)
    ("Zay Flowers", "Boston College", 2022, "WR"),
    ("Jonathan Mingo", "Ole Miss", 2022, "WR"),
    ("Tank Dell", "Houston", 2022, "WR"),
    ("Zach Evans", "Ole Miss", 2022, "RB"),
    ("DeWayne McBride", "UAB", 2022, "RB"),

    # 2024 draft (college 2023)
    ("Javon Baker", "UCF", 2023, "WR"),
    ("Bucky Irving", "Oregon", 2023, "RB"),
    ("Ray Davis", "Kentucky", 2023, "RB"),
    ("Bub Means", "Pittsburgh", 2023, "WR"),
    ("Isaiah Davis", "South Dakota State", 2023, "RB"),
    ("Tejhaun Palmer", "UAB", 2023, "WR"),
    ("Ryan Flournoy", "Southeast Missouri State", 2023, "WR"),  # FCS
]

# Alternate names to try
alternate_names = {
    "Tank Dell": ["Nathaniel Dell"],
    "Zay Flowers": ["Xavien Flowers"],
    "Bucky Irving": ["Jordan Irving"],
    "Bub Means": ["Brenden Means"],
}


def fetch_team_receiving(team, year):
    url = f"{BASE_URL}/stats/player/season"
    params = {"year": year, "category": "receiving", "team": team}
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code == 200:
        stats = {}
        for s in resp.json():
            player = s.get("player", "").lower()
            if player not in stats:
                stats[player] = {}
            try:
                stats[player][s.get("statType", "")] = int(float(s.get("stat", 0)))
            except:
                pass
        return stats
    print(f"  API error {resp.status_code} for {team} {year}")
    return {}


def search_player(team, year, name, position):
    """Search for a player in team stats."""
    rec_stats = fetch_team_receiving(team, year)
    time.sleep(0.1)

    # Get alternate names
    names_to_try = [name] + alternate_names.get(name, [])

    for try_name in names_to_try:
        try_lower = try_name.lower()

        # Direct match
        if try_lower in rec_stats:
            return rec_stats[try_lower].get("YDS"), try_name

        # Last name match
        last_name = try_lower.split()[-1]
        for api_name, stats in rec_stats.items():
            api_last = api_name.split()[-1] if api_name else ""
            if last_name == api_last:
                return stats.get("YDS"), api_name

    # Show what players ARE on that team for debugging
    if rec_stats:
        print(f"    Available players at {team}: {list(rec_stats.keys())[:5]}...")

    return None, None


if __name__ == "__main__":
    print("SEARCHING FOR 18 MISSING PLAYERS")
    print("="*70)

    found = []
    not_found = []

    for name, school, year, pos in missing_players:
        yards, matched_name = search_player(school, year, name, pos)

        if yards is not None:
            print(f"✓ {name:<20} {school:<25} {year}: {yards} yards (matched: {matched_name})")
            found.append((name, school, year, yards, matched_name))
        else:
            print(f"✗ {name:<20} {school:<25} {year}: NOT FOUND")
            not_found.append((name, school, year, pos))

    print()
    print(f"Found: {len(found)}/18")
    print(f"Still missing: {len(not_found)}/18")

    if not_found:
        print("\nStill missing:")
        for name, school, year, pos in not_found:
            print(f"  - {name} ({school}, {year}, {pos})")

    if found:
        print("\n" + "="*70)
        print("FOUND DATA TO ADD:")
        print("="*70)
        for name, school, year, yards, matched in found:
            draft_year = year + 1
            print(f'("{name}", {draft_year}, {yards}),  # {matched} at {school}')
