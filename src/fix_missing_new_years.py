"""Fix missing receiving yards for 2020, 2021, 2025 draft classes."""

import os
import requests
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE_URL = "https://api.collegefootballdata.com"


def fetch_team_stats(team, year):
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
    return {}


def fetch_pass_attempts(team, year):
    url = f"{BASE_URL}/stats/season"
    params = {"year": year, "team": team}
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code == 200:
        for stat in resp.json():
            if stat.get("statName") == "passAttempts":
                return int(stat.get("statValue", 0))
    return None


# Players to find with corrected info
# Format: (name, draft_year, correct_school, correct_college_year, reason)
players_to_find = [
    # 2020 draft - some used 2019 stats
    ("Justin Jefferson", 2020, "LSU", 2019, "normal"),
    ("Clyde Edwards-Helaire", 2020, "LSU", 2019, "normal"),
    ("Jonathan Taylor", 2020, "Wisconsin", 2019, "normal"),
    ("Ke'Shawn Vaughn", 2020, "Vanderbilt", 2019, "transfer from Illinois"),
    ("Zack Moss", 2020, "Utah", 2019, "normal"),
    ("Darrynton Evans", 2020, "Appalachian State", 2019, "normal"),
    ("Joshua Kelley", 2020, "UCLA", 2019, "normal"),
    ("Quintez Cephus", 2020, "Wisconsin", 2019, "normal"),
    ("Jason Huntley", 2020, "New Mexico State", 2019, "normal"),
    ("Malcolm Perry", 2020, "Navy", 2019, "QB converted to WR"),

    # 2021 draft - some opted out of 2020
    ("Ja'Marr Chase", 2021, "LSU", 2019, "opted out 2020"),
    ("Tutu Atwell", 2021, "Louisville", 2020, "normal"),
    ("Nico Collins", 2021, "Michigan", 2019, "opted out 2020"),
    ("Kenneth Gainwell", 2021, "Memphis", 2019, "opted out 2020"),
    ("Mike Strachan", 2021, "Charleston", 2019, "D2 school"),

    # 2025 draft
    ("Woody Marks", 2025, "USC", 2024, "normal"),
    ("Jalen Royals", 2025, "Utah State", 2024, "normal"),
    ("KeAndre Lambert-Smith", 2025, "Auburn", 2024, "transfer"),
    ("Jacory Croskey-Merritt", 2025, "Arizona", 2024, "RB"),
]

print("SEARCHING FOR MISSING PLAYERS")
print("=" * 70)

found_data = []

for name, draft_year, school, college_year, reason in players_to_find:
    stats = fetch_team_stats(school, college_year)
    time.sleep(0.1)

    found = None
    last_name = name.split()[-1].lower()
    first_name = name.split()[0].lower()

    for api_name, player_stats in stats.items():
        if last_name in api_name:
            yards = player_stats.get("YDS", 0)
            found = (api_name, yards)
            break

    # Also get team pass attempts
    tpa = fetch_pass_attempts(school, college_year)
    time.sleep(0.1)

    if found:
        print(f"✓ {name:<25} {school:<20} {college_year}: {found[1]} yards, {tpa} PA")
        found_data.append((name, draft_year, found[1], tpa, reason))
    else:
        print(f"✗ {name:<25} {school:<20} {college_year}: NOT FOUND ({reason})")
        if stats:
            print(f"    Available: {list(stats.keys())[:5]}")

# Now update the backtest file
print()
print("=" * 70)
print("UPDATING BACKTEST DATA")
print("=" * 70)

bt = pd.read_csv("data/backtest_college_stats.csv")

for name, draft_year, rec_yards, tpa, reason in found_data:
    mask = (bt['player_name'] == name) & (bt['draft_year'] == draft_year)
    if mask.sum() == 1:
        if rec_yards > 0:
            bt.loc[mask, 'rec_yards'] = rec_yards
        if tpa:
            bt.loc[mask, 'team_pass_attempts'] = tpa
        print(f"✓ Updated {name} ({draft_year}): {rec_yards} yards, {tpa} PA")
    else:
        print(f"✗ Could not find {name} ({draft_year}) in backtest data")

bt.to_csv("data/backtest_college_stats.csv", index=False)

# Final count
print()
print("=" * 70)
print("FINAL COUNTS")
print("=" * 70)

for year in [2020, 2021, 2025]:
    y = bt[bt['draft_year'] == year]
    rec = len(y[(y['rec_yards'].notna()) & (y['rec_yards'] != '')])
    tpa = len(y[(y['team_pass_attempts'].notna()) & (y['team_pass_attempts'] != '')])
    print(f"{year}: rec_yards {rec}/{len(y)}, team_pass_attempts {tpa}/{len(y)}")
