"""
Fetch height/weight for 2026 prospects from CFBD roster data (2025 season).

CFBD roster endpoint returns height (inches) and weight (lbs) for each player.
We check the 2025 roster first, then fall back to 2024.

Output: data/prospect_2026_height_weight_cfbd.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import os

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE")
HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}
BASE_URL = "https://api.collegefootballdata.com"

# School name mappings (our names → CFBD names)
SCHOOL_MAPPINGS = {
    "Ohio State": "Ohio State",
    "Penn State": "Penn State",
    "Michigan State": "Michigan State",
    "Oklahoma State": "Oklahoma State",
    "Mississippi State": "Mississippi State",
    "Florida State": "Florida State",
    "Arizona State": "Arizona State",
    "NC State": "NC State",
    "Miami (FL)": "Miami",
    "Miami (OH)": "Miami (OH)",
    "Louisiana-Lafayette": "Louisiana",
    "Central Florida": "UCF",
    "Ala-Birmingham": "UAB",
    "South Dakota State": "South Dakota State",
    "Boston College": "Boston College",
    "Mississippi": "Ole Miss",
    "Pitt": "Pittsburgh",
    "Sam Houston State": "Sam Houston",
    "McNeese State": "McNeese",
    "Stephen F. Austin": "Stephen F. Austin",
    "UT Martin": "UT Martin",
    "North Dakota State": "North Dakota State",
    "Sacramento State": "Sacramento State",
    "Virginia Union": None,   # D2
    "John Carroll": None,     # D3
    "West Alabama": None,     # D2
    "Tarleton State": None,   # FCS/transitioning
    "Delaware State": None,   # FCS
    "Montana": "Montana",
    "Youngstown State": "Youngstown State",
    "Jacksonville State": "Jacksonville State",
    "James Madison": "James Madison",
    "UC Davis": "UC Davis",
    "Georgia State": "Georgia State",
    "New Mexico State": "New Mexico State",
}


def normalize_school(school):
    if pd.isna(school):
        return None
    school = str(school).strip()
    if school in SCHOOL_MAPPINGS:
        return SCHOOL_MAPPINGS[school]
    return school


def normalize_name(name):
    if pd.isna(name):
        return ''
    s = str(name).strip().lower()
    for k, v in {'é': 'e', 'è': 'e', 'ê': 'e', 'á': 'a', 'à': 'a',
                 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s.strip()


def names_match(name1, name2):
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if n1 == n2:
        return True
    parts1 = n1.split()
    parts2 = n2.split()
    if len(parts1) < 2 or len(parts2) < 2:
        return False
    if parts1[-1] != parts2[-1]:
        return False
    first1, first2 = parts1[0], parts2[0]
    if first1 == first2:
        return True
    if len(first1) <= 2 and first2.startswith(first1[0]):
        return True
    if len(first2) <= 2 and first1.startswith(first2[0]):
        return True
    return False


# Cache rosters
roster_cache = {}


def fetch_roster(team, year):
    key = (team, year)
    if key in roster_cache:
        return roster_cache[key]

    url = f"{BASE_URL}/roster"
    params = {"team": team, "year": year}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(5)
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            roster_cache[key] = data
            return data
        else:
            roster_cache[key] = []
            return []
    except Exception as e:
        print(f"    Error fetching {team} {year}: {e}")
        roster_cache[key] = []
        return []


def find_player_in_roster(player_name, roster):
    if not roster:
        return None
    for p in roster:
        cfbd_name = f"{p.get('firstName', '')} {p.get('lastName', '')}"
        if names_match(player_name, cfbd_name):
            ht = p.get('height')
            wt = p.get('weight')
            if ht and wt:
                return {
                    'cfbd_name': cfbd_name,
                    'height': int(ht),
                    'weight': int(wt),
                }
            elif ht:
                return {'cfbd_name': cfbd_name, 'height': int(ht), 'weight': None}
            elif wt:
                return {'cfbd_name': cfbd_name, 'height': None, 'weight': int(wt)}
    return None


def main():
    df = pd.read_csv('output/slap_v5_master_database.csv')
    p26 = df[(df['dataset'] == '2026_prospect') & (df['height_in'].isna())].copy()
    p26 = p26[['player_name', 'position', 'college', 'draft_pick']].copy()
    p26 = p26.sort_values(['position', 'draft_pick']).reset_index(drop=True)

    print(f"Prospects missing height/weight: {len(p26)}")
    print(f"Unique colleges: {p26['college'].nunique()}")

    results = []
    found = 0
    skipped = 0
    not_found = 0

    for idx, row in p26.iterrows():
        player = row['player_name']
        college = row['college']
        position = row['position']

        cfbd_school = normalize_school(college)
        if cfbd_school is None:
            print(f"  [{idx+1}/{len(p26)}] SKIP {player} ({college}) — not in CFBD")
            results.append({
                'player_name': player, 'position': position, 'college': college,
                'status': 'skipped',
            })
            skipped += 1
            continue

        # Try 2025 roster first, then 2024
        match = None
        for year in [2025, 2024]:
            roster = fetch_roster(cfbd_school, year)
            time.sleep(0.25)
            match = find_player_in_roster(player, roster)
            if match:
                break

        if match:
            found += 1
            results.append({
                'player_name': player, 'position': position, 'college': college,
                'cfbd_school': cfbd_school,
                'cfbd_name': match['cfbd_name'],
                'height_in': match['height'],
                'weight': match['weight'],
                'status': 'found',
            })
            print(f"  [{idx+1}/{len(p26)}] FOUND {player:30s} → {match['height']}in, {match['weight']}lbs")
        else:
            not_found += 1
            results.append({
                'player_name': player, 'position': position, 'college': college,
                'cfbd_school': cfbd_school,
                'status': 'not_found',
            })
            print(f"  [{idx+1}/{len(p26)}] MISS  {player:30s} ({cfbd_school})")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total: {len(p26)}")
    print(f"  Found: {found}")
    print(f"  Not found: {not_found}")
    print(f"  Skipped: {skipped}")

    results_df = pd.DataFrame(results)
    output_path = 'data/prospect_2026_height_weight_cfbd.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Show not_found for debugging
    nf = results_df[results_df['status'] == 'not_found']
    if len(nf) > 0:
        print(f"\nNOT FOUND ({len(nf)}):")
        for _, r in nf.iterrows():
            print(f"  {r['player_name']:30s} {r['cfbd_school']}")


if __name__ == '__main__':
    main()
