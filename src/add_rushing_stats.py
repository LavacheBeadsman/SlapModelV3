"""
SLAP Score V3 - Add Rushing Stats for RBs

For RBs who don't have receiving yards, pull their rushing yards
as a fallback production metric.
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


def fetch_team_rushing_stats(team, year=2025):
    """Fetch all rushing stats for a team."""
    url = f"{BASE_URL}/stats/player/season"
    params = {
        "year": year,
        "category": "rushing",
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
        print(f"    Rate limited on {team}")
        return None
    else:
        return {}


def names_match(name1, name2):
    """Check if two names refer to the same person."""
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    if n1 == n2:
        return True

    parts1 = n1.replace('.', ' ').split()
    parts2 = n2.replace('.', ' ').split()

    suffixes = {'jr', 'jr.', 'ii', 'iii', 'iv'}
    clean1 = [p for p in parts1 if p not in suffixes]
    clean2 = [p for p in parts2 if p not in suffixes]

    if not clean1 or not clean2:
        return False

    first1, first2 = clean1[0], clean2[0]
    if first1 != first2:
        if not (len(first1) <= 2 and first2.startswith(first1[0])):
            if not (len(first2) <= 2 and first1.startswith(first2[0])):
                return False

    if len(clean1) > 1 and len(clean2) > 1:
        last1, last2 = clean1[-1], clean2[-1]
        if last1 != last2:
            return False

    return True


def find_player_rushing(player_name, team_stats):
    """Find a player's rushing yards in team stats."""
    if team_stats is None:
        return None, None

    for api_name, stats in team_stats.items():
        if names_match(player_name, api_name):
            yards = stats.get("YDS")
            return yards, api_name

    return None, None


def main():
    input_path = "data/prospects_final.csv"
    output_path = "data/prospects_with_rushing.csv"

    print("=" * 60)
    print("STEP 3: ADDING RUSHING YARDS FOR RBs")
    print("=" * 60)
    print()

    # Read current data
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        prospects = list(reader)

    # Find RBs missing rec_yards
    rbs_missing = [p for p in prospects
                   if p['position'] == 'RB'
                   and not p.get('rec_yards')
                   and p.get('school')]

    print(f"Found {len(rbs_missing)} RBs missing receiving yards (with known school)")
    print()

    # Group by school
    schools = {}
    for p in rbs_missing:
        school = p['school']
        if school not in schools:
            schools[school] = []
        schools[school].append(p)

    print(f"RBs spread across {len(schools)} schools")
    print()

    # Fetch rushing stats and match
    found_count = 0

    print("Fetching rushing stats...")
    print("-" * 60)

    for school, players in schools.items():
        print(f"  Fetching rushing stats for {school}...")
        team_stats = fetch_team_rushing_stats(school)

        if team_stats is None:
            print("    API rate limited - stopping")
            break

        for player in players:
            name = player['player_name']
            yards, matched_name = find_player_rushing(name, team_stats)

            if yards:
                print(f"    FOUND: {name} -> {matched_name} ({yards} rushing yards)")
                # Update in prospects list - store in rush_yards column
                for p in prospects:
                    if p['player_name'] == name:
                        p['rush_yards'] = yards
                        break
                found_count += 1

    print("-" * 60)
    print()
    print(f"RESULTS: Found rushing stats for {found_count} RBs")
    print()

    # Add rush_yards column to fieldnames
    fieldnames = ['player_name', 'position', 'school', 'projected_pick',
                  'rec_yards', 'rush_yards', 'team_pass_attempts', 'birthdate', 'age',
                  'age_estimated', 'weight']

    # Make sure all rows have rush_yards
    for p in prospects:
        if 'rush_yards' not in p:
            p['rush_yards'] = ''

    # Save updated data
    print(f"Saving to: {output_path}")

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prospects)

    print("Done!")

    # Show RBs found
    rbs_with_rush = [p for p in prospects if p['position'] == 'RB' and p.get('rush_yards')]
    print()
    print("=" * 60)
    print(f"RBs WITH RUSHING YARDS ({len(rbs_with_rush)}):")
    print("=" * 60)
    for p in rbs_with_rush[:15]:
        print(f"  {p['player_name']:<25} {p['school']:<20} {p['rush_yards']} rush yds")
    if len(rbs_with_rush) > 15:
        print(f"  ... and {len(rbs_with_rush) - 15} more")


if __name__ == "__main__":
    main()
