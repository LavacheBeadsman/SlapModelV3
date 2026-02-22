"""
Find transfer players who may have receiving stats at previous schools
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def search_player_receiving(last_name, years=[2024, 2023, 2022, 2021]):
    """Search for player by last name across all teams"""
    results = []

    for year in years:
        url = f"{BASE_URL}/stats/player/season"
        params = {"year": year, "category": "receiving"}

        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                for stat in data:
                    player = stat.get("player", "").lower()
                    if last_name.lower() in player:
                        results.append({
                            'year': year,
                            'player': stat.get("player"),
                            'team': stat.get("team"),
                            'stat_type': stat.get("statType"),
                            'value': stat.get("stat")
                        })
        except Exception as e:
            print(f"Error for {year}: {e}")

        time.sleep(1)  # Rate limit

    return results


# Players not found at their current school
not_found = [
    ("Mike Washington Jr.", "Arkansas", "washington"),
    ("Roman Hemby", "Indiana", "hemby"),
    ("C.J. Donaldson", "Ohio State", "donaldson"),
    ("J'Mari Taylor", "Virginia", "taylor"),
    ("Chip Trayanum", "Toledo", "trayanum"),
    ("Seth McGowan", "Kentucky", "mcgowan"),
    ("Dean Connors", "Houston", "connors"),
    ("Rahsul Faison", "South Carolina", "faison"),
    ("Jaydn Ott", "California", "ott"),
    ("Terion Stewart", "Virginia Tech", "stewart"),
    ("Djay Braswell", "Georgia State", "braswell"),
]

print("=" * 100)
print("SEARCHING FOR TRANSFER PLAYERS")
print("=" * 100)

for player_name, current_school, last_name in not_found:
    print(f"\n--- {player_name} (listed at {current_school}) ---")
    print(f"Searching for '{last_name}'...")

    results = search_player_receiving(last_name)

    # Group by player name
    players = {}
    for r in results:
        key = (r['player'], r['team'], r['year'])
        if key not in players:
            players[key] = {}
        players[key][r['stat_type']] = r['value']

    # Print found players
    if players:
        print(f"  {'Player':<30} {'Team':<25} {'Year':<6} {'REC':>5} {'YDS':>6}")
        print("  " + "-" * 75)
        for (player, team, year), stats in sorted(players.items(), key=lambda x: (-x[0][2], -int(x[1].get('YDS', 0) or 0))):
            rec = stats.get('REC', '-')
            yds = stats.get('YDS', '-')
            # Highlight if this looks like our player
            marker = " <--" if last_name.lower() in player.lower() and (
                player_name.split()[0].lower() in player.lower() or
                player_name.split()[0][:3].lower() in player.lower()
            ) else ""
            print(f"  {player:<30} {team:<25} {year:<6} {rec:>5} {yds:>6}{marker}")
    else:
        print("  No results found")

    print()
