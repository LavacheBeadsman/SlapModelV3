"""
2026 RB Data Verification and Multi-Season Analysis

Problems to fix:
1. May be matching wrong player (same name, different school)
2. Using only one season instead of all seasons
3. Should use EARLIEST breakout season for age, BEST season for production
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State", "Penn St.": "Penn State", "Michigan St.": "Michigan State",
    "Florida St.": "Florida State", "San Diego St.": "San Diego State",
    "Miss. St.": "Mississippi State", "Mississippi": "Ole Miss", "Pitt": "Pittsburgh",
    "Stephen F. Austin": None, "North Dakota State": None, "UT Martin": None,
}


def normalize_school(school):
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def fetch_team_receiving(team, year):
    """Fetch receiving stats for all players on a team for a specific year"""
    url = f"{BASE_URL}/stats/player/season"
    params = {"year": year, "category": "receiving", "team": team}
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []


def fetch_team_pass_attempts(team, year):
    """Fetch team pass attempts"""
    url = f"{BASE_URL}/stats/season"
    params = {"year": year, "team": team}
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            for stat in response.json():
                if stat.get("statName") == "passAttempts":
                    return float(stat.get("statValue", 0))
        return None
    except:
        return None


def search_player_all_teams(player_name, years=[2024, 2023, 2022, 2021]):
    """Search for a player across ALL FBS teams"""
    # First, get list of FBS teams
    url = f"{BASE_URL}/teams/fbs"
    try:
        response = requests.get(url, headers=HEADERS, params={"year": 2024}, timeout=30)
        if response.status_code == 200:
            teams = [t.get("school") for t in response.json()]
        else:
            return []
    except:
        return []

    # This would take too long, so we'll use a different approach
    # Search player stats directly
    results = []
    for year in years:
        url = f"{BASE_URL}/stats/player/season"
        params = {"year": year, "category": "receiving"}
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if response.status_code == 200:
                for stat in response.json():
                    if player_name.lower() in stat.get("player", "").lower():
                        results.append({
                            'year': year,
                            'player': stat.get("player"),
                            'team': stat.get("team"),
                            'stat_type': stat.get("statType"),
                            'stat': stat.get("stat")
                        })
        except:
            pass
        time.sleep(0.5)

    return results


def get_player_all_seasons(player_name, school, years=[2024, 2023, 2022, 2021]):
    """Get all seasons of receiving data for a specific player at a school"""
    cfbd_school = normalize_school(school)
    if cfbd_school is None:
        return None

    all_seasons = []

    for year in years:
        stats = fetch_team_receiving(cfbd_school, year)
        time.sleep(0.3)

        # Group by player
        players = {}
        for s in stats:
            name = s.get("player", "")
            stat_type = s.get("statType", "")
            value = s.get("stat", 0)

            if name not in players:
                players[name] = {'year': year, 'team': cfbd_school}
            try:
                players[name][stat_type] = int(float(value))
            except:
                players[name][stat_type] = 0

        # Find matching player
        for api_name, data in players.items():
            if names_similar(player_name, api_name):
                pass_att = fetch_team_pass_attempts(cfbd_school, year)
                time.sleep(0.2)
                data['pass_att'] = pass_att
                data['api_name'] = api_name
                all_seasons.append(data)
                break

    return all_seasons


def names_similar(name1, name2):
    """Check if names are similar enough to be the same person"""
    n1 = name1.lower().strip().replace('.', ' ').replace("'", "").replace("-", " ")
    n2 = name2.lower().strip().replace('.', ' ').replace("'", "").replace("-", " ")

    if n1 == n2:
        return True

    parts1 = n1.split()
    parts2 = n2.split()

    suffixes = {'jr', 'ii', 'iii', 'iv', 'sr'}
    clean1 = [p for p in parts1 if p not in suffixes]
    clean2 = [p for p in parts2 if p not in suffixes]

    if len(clean1) < 2 or len(clean2) < 2:
        return False

    # Last name must match
    if clean1[-1] != clean2[-1]:
        return False

    # First name - check various patterns
    first1, first2 = clean1[0], clean2[0]

    if first1 == first2:
        return True

    # First 3 chars match
    if len(first1) >= 3 and len(first2) >= 3 and first1[:3] == first2[:3]:
        return True

    # Common nicknames
    nicknames = {
        'nick': 'nicholas', 'nicholas': 'nick',
        'mike': 'michael', 'michael': 'mike',
        'ej': 'e j', 'cj': 'c j', 'dj': 'd j',
    }
    if nicknames.get(first1) == first2 or nicknames.get(first2) == first1:
        return True

    return False


def main():
    print("=" * 100)
    print("2026 RB DATA VERIFICATION")
    print("=" * 100)

    # Load 2026 RB prospects
    prospects = pd.read_csv('data/prospects_final.csv')
    rbs = prospects[prospects['position'] == 'RB'].copy()

    print(f"\nLoaded {len(rbs)} RB prospects for 2026")

    # ==========================================================================
    # STEP 1: VERIFY NAME/COLLEGE MATCHING
    # ==========================================================================
    print("\n" + "=" * 100)
    print("STEP 1: NAME/COLLEGE VERIFICATION")
    print("=" * 100)

    print(f"\n{'Player':<25} {'Our School':<20} {'CFBD School':<20} {'Match?':<8} {'Years Found'}")
    print("-" * 100)

    verification_results = []

    for _, row in rbs.head(30).iterrows():  # Check top 30
        player = row['player_name']
        school = row['school']
        cfbd_school = normalize_school(school)

        if cfbd_school is None:
            print(f"{player:<25} {school:<20} {'(FCS)':<20} {'N/A':<8}")
            verification_results.append({
                'player': player,
                'our_school': school,
                'cfbd_school': None,
                'match': 'FCS',
                'seasons': []
            })
            continue

        # Get all seasons for this player at this school
        seasons = get_player_all_seasons(player, school, [2024, 2023, 2022, 2021])

        if seasons:
            years_found = [s['year'] for s in seasons]
            api_name = seasons[0].get('api_name', '?')
            match_status = "✓" if cfbd_school else "?"
            print(f"{player:<25} {school:<20} {cfbd_school:<20} {match_status:<8} {years_found}")
            verification_results.append({
                'player': player,
                'our_school': school,
                'cfbd_school': cfbd_school,
                'cfbd_name': api_name,
                'match': 'YES',
                'seasons': seasons
            })
        else:
            print(f"{player:<25} {school:<20} {cfbd_school:<20} {'✗ NOT FOUND':<8}")
            verification_results.append({
                'player': player,
                'our_school': school,
                'cfbd_school': cfbd_school,
                'match': 'NOT FOUND',
                'seasons': []
            })

    # ==========================================================================
    # STEP 2: SHOW MULTI-SEASON DATA FOR KEY PLAYERS
    # ==========================================================================
    print("\n" + "=" * 100)
    print("STEP 2: MULTI-SEASON DATA FOR KEY PLAYERS")
    print("=" * 100)

    key_players = [
        ('Jeremiyah Love', 'Notre Dame'),
        ('Emmett Johnson', 'Nebraska'),
        ('Jadarian Price', 'Notre Dame'),
        ('Nick Singleton', 'Penn State'),
        ('Kaytron Allen', 'Penn State'),
        ('Jonah Coleman', 'Washington'),
        ('Desmond Reid', 'Pittsburgh'),
    ]

    for player, school in key_players:
        print(f"\n--- {player} ({school}) ---")
        seasons = get_player_all_seasons(player, school, [2024, 2023, 2022, 2021])

        if not seasons:
            print("  NO DATA FOUND")
            continue

        print(f"  {'Year':<6} {'CFBD Name':<25} {'REC':>5} {'YDS':>6} {'TD':>4} {'PassAtt':>8} {'Yds/PassAtt':>10}")
        print("  " + "-" * 70)

        best_season = None
        best_yds = 0

        for s in seasons:
            yds = s.get('YDS', 0) or 0
            rec = s.get('REC', 0) or 0
            td = s.get('TD', 0) or 0
            pass_att = s.get('pass_att', 0) or 0
            yds_per_pa = yds / pass_att if pass_att > 0 else 0

            print(f"  {s['year']:<6} {s.get('api_name', '?'):<25} {rec:>5} {yds:>6} {td:>4} {pass_att:>8} {yds_per_pa:>10.3f}")

            if yds > best_yds:
                best_yds = yds
                best_season = s

        if best_season:
            print(f"\n  BEST SEASON: {best_season['year']} - {best_yds} receiving yards")

    # ==========================================================================
    # STEP 3: IDENTIFY PROBLEMS
    # ==========================================================================
    print("\n" + "=" * 100)
    print("STEP 3: IDENTIFIED PROBLEMS")
    print("=" * 100)

    not_found = [r for r in verification_results if r['match'] == 'NOT FOUND']
    fcs = [r for r in verification_results if r['match'] == 'FCS']

    print(f"\nPlayers NOT FOUND in CFBD ({len(not_found)}):")
    for r in not_found:
        print(f"  - {r['player']} ({r['our_school']})")

    print(f"\nFCS schools (not in CFBD) ({len(fcs)}):")
    for r in fcs:
        print(f"  - {r['player']} ({r['our_school']})")


if __name__ == "__main__":
    main()
