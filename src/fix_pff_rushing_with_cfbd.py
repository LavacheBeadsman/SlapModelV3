"""
Replace unreliable PFF rushing data with CFBD API data for 284 PFF-sourced WRs.

For each PFF-sourced WR:
- Call CFBD API for their final college season rushing stats
- If CFBD has data → use it (replace PFF value), mark source as 'cfbd'
- If CFBD team has data but player not found → confirmed zero, mark 'cfbd_confirmed_zero'
- If CFBD has no data for that team/season → keep PFF value as-is
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

# School name mappings (same as fix_rushing_properly.py)
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
    "Southern Miss.": "Southern Mississippi", "Southern Miss": "Southern Mississippi",
    "San Jose St.": "San Jose State", "Ala-Birmingham": "UAB",
    "Central Florida": "UCF", "Mississippi": "Ole Miss",
    "Pitt": "Pittsburgh", "Hawaii": "Hawai'i",
    "Middle Tenn. St.": "Middle Tennessee",
    "East Carolina": "East Carolina", "Georgia Tech": "Georgia Tech",
    "Connecticut": "Connecticut",
    "Eastern Washington": "Eastern Washington",
    "Northern Iowa": "Northern Iowa",
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
    "North Texas": "North Texas",
    "West Virginia": "West Virginia",
    "Louisville": "Louisville",
    "South Carolina": "South Carolina",
    "North Carolina": "North Carolina",
    "North Carolina St.": "NC State",
}


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


def fetch_rushing(team, year, player_name):
    """Fetch rushing stats from CFBD for a player's final college season."""
    cfbd_team = normalize_school(team)
    if cfbd_team is None:
        return None

    url = f"{BASE_URL}/stats/player/season"
    params = {'year': year, 'team': cfbd_team, 'category': 'rushing'}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return {'status': 'api_error', 'code': resp.status_code}

        data = resp.json()
        if not data:
            return {'status': 'no_team_data'}

        # Group by player
        players = {}
        for entry in data:
            p = entry.get('player', '')
            if p not in players:
                players[p] = {}
            players[p][entry.get('statType', '')] = entry.get('stat', '0')

        # Try exact normalized match first, then last name
        norm = normalize_name(player_name)
        last_name = player_name.split()[-1].lower()

        match = None
        match_name = None
        for p_name, stats in players.items():
            if normalize_name(p_name) == norm:
                match = stats
                match_name = p_name
                break
        if match is None:
            for p_name, stats in players.items():
                if last_name in p_name.lower() and len(last_name) > 3:
                    match = stats
                    match_name = p_name
                    break

        if match:
            return {
                'status': 'found',
                'cfbd_name': match_name,
                'rush_yards': int(float(match.get('YDS', 0))),
                'rush_att': int(float(match.get('CAR', 0))),
                'rush_tds': int(float(match.get('TD', 0))),
            }
        else:
            return {'status': 'confirmed_zero', 'team_players': len(players)}

    except Exception as e:
        return {'status': 'error', 'error': str(e)}


# ============================================================================
# LOAD DATA AND RECORD BEFORE STATE
# ============================================================================

wr = pd.read_csv('data/wr_backtest_all_components.csv')
pff_mask = wr['rush_source'] == 'pff'
pff_wrs = wr[pff_mask].copy()
print(f"Loaded {len(wr)} WRs total, {len(pff_wrs)} PFF-sourced to re-check")

# Save before state for comparison
before = wr[['player_name', 'draft_year', 'rush_yards', 'rush_source']].copy()
before.columns = ['player_name', 'draft_year', 'rush_yards_before', 'rush_source_before']

# ============================================================================
# FETCH CFBD RUSHING DATA FOR ALL 284 PFF-SOURCED WRs
# ============================================================================

print(f"\n{'='*120}")
print("FETCHING CFBD RUSHING DATA FOR ALL PFF-SOURCED WRs")
print("=" * 120)
print(f"\n{'#':>3} {'Player':<30} {'Year':>5} {'College':<25} {'Season':>7} "
      f"{'PFF':>6} {'CFBD':>6} {'Status':<20} {'Notes'}")
print("-" * 140)

cfbd_found = 0
cfbd_confirmed_zero = 0
cfbd_no_data = 0
api_errors = 0
changed = 0

for i, (idx, row) in enumerate(pff_wrs.iterrows(), 1):
    name = row['player_name']
    year = int(row['draft_year'])
    college = str(row['college'])
    final_season = year - 1
    pff_yards = row['rush_yards']

    # Skip if no college
    if pd.isna(row['college']) or college == 'nan':
        cfbd_no_data += 1
        print(f"{i:>3} {name:<30} {year:>5} {'N/A':<25} {'N/A':>7} "
              f"{pff_yards:>6.0f} {'--':>6} {'no_college':<20}")
        continue

    time.sleep(0.3)
    result = fetch_rushing(college, final_season, name)

    if result is None:
        cfbd_no_data += 1
        status = 'no_school_map'
        print(f"{i:>3} {name:<30} {year:>5} {college:<25} {final_season:>7} "
              f"{pff_yards:>6.0f} {'--':>6} {status:<20}")

    elif result['status'] == 'found':
        cfbd_yards = result['rush_yards']
        cfbd_found += 1

        # Update the dataframe
        wr.loc[idx, 'rush_yards'] = cfbd_yards
        wr.loc[idx, 'rush_attempts'] = result['rush_att']
        wr.loc[idx, 'rush_tds'] = result['rush_tds']
        wr.loc[idx, 'rush_source'] = 'cfbd'

        flag = ''
        if (pff_yards < 20 and cfbd_yards >= 20):
            flag = ' *** GAINED BONUS'
            changed += 1
        elif (pff_yards >= 20 and cfbd_yards < 20):
            flag = ' *** LOST BONUS'
            changed += 1

        print(f"{i:>3} {name:<30} {year:>5} {college:<25} {final_season:>7} "
              f"{pff_yards:>6.0f} {cfbd_yards:>6} {'found':<20} as \"{result['cfbd_name']}\"{flag}")

    elif result['status'] == 'confirmed_zero':
        cfbd_confirmed_zero += 1
        # Team had rushing data but player wasn't in it → real zero
        wr.loc[idx, 'rush_yards'] = 0
        wr.loc[idx, 'rush_attempts'] = 0
        wr.loc[idx, 'rush_tds'] = 0
        wr.loc[idx, 'rush_source'] = 'cfbd_confirmed_zero'

        flag = ''
        if pff_yards >= 20:
            flag = ' *** LOST BONUS'
            changed += 1

        print(f"{i:>3} {name:<30} {year:>5} {college:<25} {final_season:>7} "
              f"{pff_yards:>6.0f} {0:>6} {'confirmed_zero':<20} "
              f"team had {result['team_players']} rushers{flag}")

    elif result['status'] == 'no_team_data':
        cfbd_no_data += 1
        # CFBD doesn't have this team/season → keep PFF value
        print(f"{i:>3} {name:<30} {year:>5} {college:<25} {final_season:>7} "
              f"{pff_yards:>6.0f} {'--':>6} {'no_team_data':<20} keeping PFF")

    else:
        api_errors += 1
        print(f"{i:>3} {name:<30} {year:>5} {college:<25} {final_season:>7} "
              f"{pff_yards:>6.0f} {'--':>6} {result['status']:<20} "
              f"{result.get('error', '')}")

    # Progress indicator every 50 players
    if i % 50 == 0:
        print(f"\n  --- Progress: {i}/{len(pff_wrs)} done ---\n")


# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*120}")
print("RESULTS SUMMARY")
print("=" * 120)
print(f"  CFBD data found:          {cfbd_found}")
print(f"  CFBD confirmed zero:      {cfbd_confirmed_zero}")
print(f"  CFBD no team data (kept PFF): {cfbd_no_data}")
print(f"  API errors:               {api_errors}")
print(f"  Bonus changes:            {changed}")

# ============================================================================
# SHOW ALL BONUS CHANGES
# ============================================================================

after = wr[['player_name', 'draft_year', 'rush_yards', 'rush_source']].copy()
after.columns = ['player_name', 'draft_year', 'rush_yards_after', 'rush_source_after']

compare = before.merge(after, on=['player_name', 'draft_year'])
compare['bonus_before'] = compare['rush_yards_before'].apply(lambda x: 1 if pd.notna(x) and x >= 20 else 0)
compare['bonus_after'] = compare['rush_yards_after'].apply(lambda x: 1 if pd.notna(x) and x >= 20 else 0)
compare['bonus_changed'] = compare['bonus_before'] != compare['bonus_after']

bonus_changes = compare[compare['bonus_changed']].sort_values('player_name')

print(f"\n{'='*120}")
print(f"RUSH BONUS CHANGES ({len(bonus_changes)} players)")
print("=" * 120)

if len(bonus_changes) > 0:
    gained = bonus_changes[bonus_changes['bonus_after'] == 1]
    lost = bonus_changes[bonus_changes['bonus_before'] == 1]

    if len(gained) > 0:
        print(f"\n  GAINED +5 rush bonus ({len(gained)} players):")
        print(f"  {'Player':<30} {'Year':>5} {'PFF yards':>10} {'CFBD yards':>11} {'New source':<20}")
        print(f"  {'-'*80}")
        for _, r in gained.iterrows():
            print(f"  {r['player_name']:<30} {int(r['draft_year']):>5} {r['rush_yards_before']:>10.0f} "
                  f"{r['rush_yards_after']:>11.0f} {r['rush_source_after']:<20}")

    if len(lost) > 0:
        print(f"\n  LOST +5 rush bonus ({len(lost)} players):")
        print(f"  {'Player':<30} {'Year':>5} {'PFF yards':>10} {'CFBD yards':>11} {'New source':<20}")
        print(f"  {'-'*80}")
        for _, r in lost.iterrows():
            print(f"  {r['player_name']:<30} {int(r['draft_year']):>5} {r['rush_yards_before']:>10.0f} "
                  f"{r['rush_yards_after']:>11.0f} {r['rush_source_after']:<20}")

# ============================================================================
# HIGH-PROFILE NAME CHECK
# ============================================================================

print(f"\n{'='*120}")
print("HIGH-PROFILE NAME CHECK")
print("=" * 120)

names = ['A.J. Brown', 'Chase Claypool', 'Marquise Brown', "Ja'Marr Chase",
         'Amari Rodgers', 'Brian Thomas']

for name in names:
    row_b = compare[compare['player_name'].str.contains(name, case=False, na=False)]
    if len(row_b) > 0:
        r = row_b.iloc[0]
        bonus_b = '+5' if r['bonus_before'] == 1 else ' 0'
        bonus_a = '+5' if r['bonus_after'] == 1 else ' 0'
        change = 'CHANGED' if r['bonus_changed'] else 'no change'
        print(f"  {r['player_name']:<30} PFF={r['rush_yards_before']:>5.0f} → CFBD={r['rush_yards_after']:>5.0f}  "
              f"bonus: {bonus_b} → {bonus_a}  ({change})")
    else:
        print(f"  {name:<30} NOT FOUND")


# ============================================================================
# FINAL SOURCE DISTRIBUTION
# ============================================================================

print(f"\n{'='*120}")
print("FINAL RUSH_SOURCE DISTRIBUTION")
print("=" * 120)
print(wr['rush_source'].value_counts().to_string())

# ============================================================================
# SAVE
# ============================================================================

wr.to_csv('data/wr_backtest_all_components.csv', index=False)
print(f"\nSaved: data/wr_backtest_all_components.csv")
print(f"\nTotal WRs getting +5 rush bonus: {len(wr[wr['rush_yards'] >= 20])}")
print(f"Total WRs NOT getting bonus: {len(wr[wr['rush_yards'] < 20]) + len(wr[wr['rush_yards'].isna()])}")
