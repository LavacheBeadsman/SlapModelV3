"""
Fix Rushing Data: Actually call CFBD for every player instead of assuming 0.
Distinguish between "confirmed 0 yards" and "unknown/no data."
"""

import pandas as pd
import requests
import time
import re
import warnings
warnings.filterwarnings('ignore')

API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

# Complete school name mappings — INCLUDING FCS schools (don't skip them!)
SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State", "Penn St.": "Penn State",
    "Michigan St.": "Michigan State", "North Dakota St.": "North Dakota State",
    "Oklahoma St.": "Oklahoma State", "Mississippi St.": "Mississippi State",
    "San Diego St.": "San Diego State", "Florida St.": "Florida State",
    "Fresno St.": "Fresno State", "Boise St.": "Boise State",
    "Iowa St.": "Iowa State", "Arizona St.": "Arizona State",
    "Kansas St.": "Kansas State", "N.C. State": "NC State",
    "Appalachian St.": "Appalachian State", "Colorado St.": "Colorado State",
    "Oregon St.": "Oregon State", "Washington St.": "Washington State",
    "Miami (FL)": "Miami", "Northern Ill.": "Northern Illinois",
    "Southern Miss.": "Southern Mississippi",
    "San Jose St.": "San Jose State", "Ala-Birmingham": "UAB",
    "Central Florida": "UCF", "Mississippi": "Ole Miss",
    "Pitt": "Pittsburgh", "Hawaii": "Hawai'i",
    "Middle Tenn. St.": "Middle Tennessee",
    "East Carolina": "East Carolina", "Georgia Tech": "Georgia Tech",
    "Connecticut": "Connecticut",
    # FCS schools — TRY them in CFBD (many are covered)
    "Eastern Washington": "Eastern Washington",
    "Northern Iowa": "Northern Iowa",
    "North Dakota St.": "North Dakota State",
    "Central Arkansas": "Central Arkansas",
    "Grambling St.": "Grambling",
    "Samford": "Samford",
    "William & Mary": "William & Mary",
    "Monmouth": "Monmouth",
    "Rhode Island": "Rhode Island",
    "Princeton": "Princeton",
    "SE Missouri St.": "Southeast Missouri State",
    "UT Martin": "UT Martin",
    "Lenoir-Rhyne": "Lenoir-Rhyne",
    "Charleston (WV)": "Charleston",
    "Pennsylvania": "Pennsylvania",
}

# Schools where CFBD definitely has NO data (D2, international)
NO_CFBD_SCHOOLS = {"West Alabama", "East Central (OK)"}


def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = re.sub(r'\s+(Jr\.?|Sr\.?|III|II|IV|V)$', '', name, flags=re.IGNORECASE)
    name = name.replace('.', '').replace("'", '').replace("\u2019", '')
    return name.lower().strip()


def normalize_school(school):
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def fetch_and_parse_rushing(team, year, player_name):
    """
    Fetch rushing stats from CFBD and search for the player.
    Returns dict with results or None.
    """
    cfbd_team = normalize_school(team)
    if cfbd_team is None:
        return None

    url = f"{BASE_URL}/stats/player/season"
    params = {'year': year, 'team': cfbd_team, 'category': 'rushing'}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data:
            return {'status': 'team_no_data', 'entries': 0}

        # Search for the player by name
        norm = normalize_name(player_name)
        last_name = player_name.split()[-1].lower()

        # Group by player
        players = {}
        for entry in data:
            p = entry.get('player', '')
            if p not in players:
                players[p] = {}
            players[p][entry.get('statType', '')] = entry.get('stat', '0')
            players[p]['position'] = entry.get('position', '')

        # Try exact normalized match, then last name match
        match = None
        match_name = None
        for p_name, stats in players.items():
            if normalize_name(p_name) == norm:
                match = stats
                match_name = p_name
                break
        if match is None:
            for p_name, stats in players.items():
                if last_name in p_name.lower():
                    match = stats
                    match_name = p_name
                    break

        if match:
            return {
                'status': 'found',
                'cfbd_name': match_name,
                'rush_att': int(float(match.get('CAR', 0))),
                'rush_yards': int(float(match.get('YDS', 0))),
                'rush_tds': int(float(match.get('TD', 0))),
            }
        else:
            # Player NOT in rushing data but team had data
            # → confirmed 0 rushing (team exists, player doesn't appear)
            return {'status': 'confirmed_zero', 'team_players': len(players)}

    except Exception as e:
        return {'status': 'error', 'error': str(e)}


# ============================================================================
# LOAD DATA
# ============================================================================

wr = pd.read_csv('data/wr_backtest_all_components.csv')
print(f"Loaded {len(wr)} WRs")

# Find all players with unreliable rushing data
bad_sources = ['assumed_zero_fcs', 'cfbd_not_found', 'cfbd_no_data']
to_fix = wr[wr['rush_source'].isin(bad_sources)].copy()
print(f"Players to re-check: {len(to_fix)}")


# ============================================================================
# RE-QUERY CFBD FOR EVERY PLAYER
# ============================================================================

print("\n" + "=" * 120)
print("RE-QUERYING CFBD API FOR ALL 32 PLAYERS")
print("=" * 120)

print(f"\n{'#':>3} {'Player':<28} {'Year':>5} {'College':<25} {'Season':>7} "
      f"{'Status':<20} {'ATT':>5} {'YDS':>5} {'TD':>3} {'Notes'}")
print("-" * 130)

results = []

for i, (idx, row) in enumerate(to_fix.iterrows(), 1):
    name = row['player_name']
    year = int(row['draft_year'])
    college = str(row['college'])
    final_season = year - 1
    rd = int(row['round'])

    # Handle international player (Boehringer)
    if pd.isna(row['college']) or college == 'nan':
        results.append({
            'idx': idx, 'player_name': name, 'draft_year': year,
            'rush_att': None, 'rush_yards': None, 'rush_tds': None,
            'rush_source': 'no_college_data',
            'notes': 'International player — no college rushing data exists'
        })
        print(f"{i:>3} {name:<28} {year:>5} {'International':<25} {'N/A':>7} "
              f"{'no_college_data':<20} {'':>5} {'':>5} {'':>3} International player")
        continue

    # Handle D2 schools with no CFBD data
    if college in NO_CFBD_SCHOOLS:
        results.append({
            'idx': idx, 'player_name': name, 'draft_year': year,
            'rush_att': None, 'rush_yards': None, 'rush_tds': None,
            'rush_source': 'unknown_d2',
            'notes': f'D2 school ({college}) — CFBD has no data, true value unknown'
        })
        print(f"{i:>3} {name:<28} {year:>5} {college:<25} {final_season:>7} "
              f"{'unknown_d2':<20} {'?':>5} {'?':>5} {'?':>3} D2 school — no data source")
        continue

    # Call CFBD API
    time.sleep(0.35)
    result = fetch_and_parse_rushing(college, final_season, name)

    if result is None:
        # API call failed entirely
        results.append({
            'idx': idx, 'player_name': name, 'draft_year': year,
            'rush_att': None, 'rush_yards': None, 'rush_tds': None,
            'rush_source': 'unknown_api_fail',
            'notes': f'CFBD API returned nothing for {normalize_school(college)} {final_season}'
        })
        print(f"{i:>3} {name:<28} {year:>5} {college:<25} {final_season:>7} "
              f"{'unknown_api_fail':<20} {'?':>5} {'?':>5} {'?':>3} API returned nothing")

    elif result['status'] == 'found':
        att = result['rush_att']
        yds = result['rush_yards']
        tds = result['rush_tds']
        results.append({
            'idx': idx, 'player_name': name, 'draft_year': year,
            'rush_att': att, 'rush_yards': yds, 'rush_tds': tds,
            'rush_source': 'cfbd',
            'notes': f'Found as "{result["cfbd_name"]}"'
        })
        print(f"{i:>3} {name:<28} {year:>5} {college:<25} {final_season:>7} "
              f"{'FOUND':<20} {att:>5} {yds:>5} {tds:>3} Found as \"{result['cfbd_name']}\"")

    elif result['status'] == 'confirmed_zero':
        results.append({
            'idx': idx, 'player_name': name, 'draft_year': year,
            'rush_att': 0, 'rush_yards': 0, 'rush_tds': 0,
            'rush_source': 'cfbd_confirmed_zero',
            'notes': f'Team had {result["team_players"]} rushers, player not among them → confirmed 0'
        })
        print(f"{i:>3} {name:<28} {year:>5} {college:<25} {final_season:>7} "
              f"{'confirmed_zero':<20} {0:>5} {0:>5} {0:>3} "
              f"Team had {result['team_players']} rushers, player absent")

    elif result['status'] == 'team_no_data':
        results.append({
            'idx': idx, 'player_name': name, 'draft_year': year,
            'rush_att': None, 'rush_yards': None, 'rush_tds': None,
            'rush_source': 'unknown_no_team_data',
            'notes': f'CFBD has no rushing data for {normalize_school(college)} {final_season}'
        })
        print(f"{i:>3} {name:<28} {year:>5} {college:<25} {final_season:>7} "
              f"{'unknown_no_data':<20} {'?':>5} {'?':>5} {'?':>3} CFBD has no data for this team/season")

    else:
        results.append({
            'idx': idx, 'player_name': name, 'draft_year': year,
            'rush_att': None, 'rush_yards': None, 'rush_tds': None,
            'rush_source': 'unknown_error',
            'notes': f'Error: {result.get("error", result["status"])}'
        })
        print(f"{i:>3} {name:<28} {year:>5} {college:<25} {final_season:>7} "
              f"{'error':<20} {'?':>5} {'?':>5} {'?':>3} {result.get('error', '')}")


# ============================================================================
# APPLY FIXES TO DATAFRAME
# ============================================================================

print("\n" + "=" * 120)
print("APPLYING FIXES")
print("=" * 120)

for fix in results:
    idx = fix['idx']
    wr.loc[idx, 'rush_source'] = fix['rush_source']
    wr.loc[idx, 'rush_attempts'] = fix['rush_att']
    wr.loc[idx, 'rush_yards'] = fix['rush_yards']
    wr.loc[idx, 'rush_touchdowns'] = fix['rush_tds']

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 120)
print("RUSHING DATA SUMMARY — ALL SOURCES")
print("=" * 120)

source_counts = wr['rush_source'].value_counts()
print(f"\n{'Source':<30} {'Count':>6} {'Meaning'}")
print("-" * 80)
source_meanings = {
    'pff': 'PFF college data — verified',
    'cfbd': 'CFBD API — real data found',
    'cfbd_confirmed_zero': 'CFBD team had data, player absent → real 0',
    'pff_rushing_summary': 'PFF rushing summary file — verified',
    'unknown_d2': 'D2 school — no database covers this school',
    'unknown_no_team_data': 'FCS school — CFBD has no data for this team/season',
    'unknown_api_fail': 'CFBD API returned nothing — cannot verify',
    'no_college_data': 'International player — no college data exists',
    'assumed_zero_fcs': 'SHOULD NOT EXIST after fix',
    'cfbd_not_found': 'SHOULD NOT EXIST after fix',
}
for src, count in source_counts.items():
    meaning = source_meanings.get(src, '???')
    print(f"  {src:<30} {count:>6} {meaning}")

# List unknowns
unknowns = wr[wr['rush_source'].str.startswith('unknown') | (wr['rush_source'] == 'no_college_data')]
if len(unknowns) > 0:
    print(f"\n\n{'='*120}")
    print(f"PLAYERS WITH UNKNOWN RUSHING (not assumed 0)")
    print(f"{'='*120}")
    print(f"\n{'#':>3} {'Player':<28} {'Year':>5} {'Rd':>3} {'Pick':>5} {'College':<25} {'Source':<25} {'Notes'}")
    print("-" * 120)
    for i, (_, r) in enumerate(unknowns.iterrows(), 1):
        fix = [f for f in results if f['idx'] == _][0] if _ in [f['idx'] for f in results] else {}
        notes = fix.get('notes', '') if fix else ''
        print(f"{i:>3} {r['player_name']:<28} {int(r['draft_year']):>5} {int(r['round']):>3} {int(r['pick']):>5} "
              f"{str(r['college']):<25} {r['rush_source']:<25} {notes}")


# Save
wr.to_csv('data/wr_backtest_all_components.csv', index=False)
print(f"\nSaved: data/wr_backtest_all_components.csv")

# Final coverage
known = wr[~wr['rush_source'].str.startswith('unknown') & (wr['rush_source'] != 'no_college_data')]
unknown = wr[wr['rush_source'].str.startswith('unknown') | (wr['rush_source'] == 'no_college_data')]
print(f"\nFinal rushing coverage:")
print(f"  Known (real data or confirmed 0): {len(known)}/339 = {len(known)/339*100:.1f}%")
print(f"  Unknown (flagged, NOT assumed 0): {len(unknown)}/339 = {len(unknown)/339*100:.1f}%")
