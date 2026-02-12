"""
Task 2: Add games played to all 339 WRs in the backtest.

Sources:
- PFF player_game_count for the 284 matched WRs (from wr_pff_all_2016_2025.csv)
- CFBD API /games/players endpoint for unmatched WRs (2015 class + FCS schools)

Output: data/wr_games_played.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import re
import os

os.chdir("/home/user/SlapModelV3")

CFBD_API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
CFBD_HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}
CFBD_BASE = "https://api.collegefootballdata.com"

# School name mapping for CFBD (backtest name → CFBD name)
SCHOOL_TO_CFBD = {
    'Alabama': 'Alabama',
    'Ala-Birmingham': 'UAB',
    'Arizona': 'Arizona',
    'Arizona St.': 'Arizona State',
    'Arkansas': 'Arkansas',
    'Auburn': 'Auburn',
    'Baylor': 'Baylor',
    'Boise St.': 'Boise State',
    'Boston Col.': 'Boston College',
    'Bowling Green': 'Bowling Green',
    'BYU': 'BYU',
    'California': 'California',
    'Central Arkansas': 'Central Arkansas',
    'Central Florida': 'UCF',
    'Charleston (WV)': None,  # D2
    'Charlotte': 'Charlotte',
    'Cincinnati': 'Cincinnati',
    'Clemson': 'Clemson',
    'Colorado': 'Colorado',
    'Colorado St.': 'Colorado State',
    'Connecticut': 'Connecticut',
    'Duke': 'Duke',
    'East Carolina': 'East Carolina',
    'East Central (OK)': None,  # D2/NAIA
    'Eastern Washington': 'Eastern Washington',
    'Florida': 'Florida',
    'Florida St.': 'Florida State',
    'Fresno St.': 'Fresno State',
    'Georgia': 'Georgia',
    'Georgia St.': 'Georgia State',
    'Georgia Tech': 'Georgia Tech',
    'Grambling St.': 'Grambling',
    'Hawaii': "Hawai'i",
    'Houston': 'Houston',
    'Illinois': 'Illinois',
    'Iowa': 'Iowa',
    'Iowa St.': 'Iowa State',
    'Iowa State': 'Iowa State',
    'Kansas St.': 'Kansas State',
    'Kentucky': 'Kentucky',
    'LSU': 'LSU',
    'La-Monroe': 'Louisiana Monroe',
    'Lenoir-Rhyne': None,  # D2
    'Liberty': 'Liberty',
    'Louisiana Tech': 'Louisiana Tech',
    'Louisville': 'Louisville',
    'Maryland': 'Maryland',
    'Massachusetts': 'Massachusetts',
    'Memphis': 'Memphis',
    'Miami (FL)': 'Miami',
    'Michigan': 'Michigan',
    'Michigan St.': 'Michigan State',
    'Middle Tenn. St.': 'Middle Tennessee',
    'Minnesota': 'Minnesota',
    'Mississippi': 'Ole Miss',
    'Missouri': 'Missouri',
    'Monmouth': 'Monmouth',
    'Nebraska': 'Nebraska',
    'Nevada': 'Nevada',
    'New Mexico St.': 'New Mexico State',
    'North Carolina': 'North Carolina',
    'North Carolina St.': 'NC State',
    'North Dakota St.': 'North Dakota State',
    'North Texas': 'North Texas',
    'Northern Illinois': 'Northern Illinois',
    'Northern Iowa': 'Northern Iowa',
    'Notre Dame': 'Notre Dame',
    'Ohio St.': 'Ohio State',
    'Ohio State': 'Ohio State',
    'Oklahoma': 'Oklahoma',
    'Oklahoma St.': 'Oklahoma State',
    'Old Dominion': 'Old Dominion',
    'Ole Miss': 'Ole Miss',
    'Oregon': 'Oregon',
    'Oregon St.': 'Oregon State',
    'Penn St.': 'Penn State',
    'Pennsylvania': 'Pennsylvania',
    'Pittsburgh': 'Pittsburgh',
    'Princeton': 'Princeton',
    'Purdue': 'Purdue',
    'Rhode Island': 'Rhode Island',
    'Rice': 'Rice',
    'Rutgers': 'Rutgers',
    'SE Missouri St.': 'Southeast Missouri State',
    'SMU': 'SMU',
    'Samford': 'Samford',
    'South Alabama': 'South Alabama',
    'South Carolina': 'South Carolina',
    'South Florida': 'South Florida',
    'Southern Miss': 'Southern Mississippi',
    'Stanford': 'Stanford',
    'TCU': 'TCU',
    'Tennessee': 'Tennessee',
    'Texas': 'Texas',
    'Texas A&M': 'Texas A&M',
    'Texas Tech': 'Texas Tech',
    'Toledo': 'Toledo',
    'Tulane': 'Tulane',
    'UCLA': 'UCLA',
    'USC': 'USC',
    'UT Martin': 'UT Martin',
    'Utah': 'Utah',
    'Virginia': 'Virginia',
    'Virginia Tech': 'Virginia Tech',
    'Wake Forest': 'Wake Forest',
    'Washington': 'Washington',
    'Washington St.': 'Washington State',
    'Washington State': 'Washington State',
    'West Alabama': None,  # D2
    'West Virginia': 'West Virginia',
    'Western Kentucky': 'Western Kentucky',
    'Western Michigan': 'Western Michigan',
    'William & Mary': "William & Mary",
    'Wisconsin': 'Wisconsin',
}


def normalize_name(name):
    """Normalize player name for CFBD matching."""
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    name = name.replace("'", "").replace("'", "").replace(".", "").replace(",", "")
    name = re.sub(r'\s+(jr|sr|ii|iii|iv|v)\s*$', '', name)
    return ' '.join(name.split())


def get_team_games(school, season):
    """Get number of games played by a team in a season via CFBD API."""
    url = f"{CFBD_BASE}/games"
    params = {'year': season, 'team': school, 'seasonType': 'both'}
    try:
        resp = requests.get(url, headers=CFBD_HEADERS, params=params, timeout=10)
        if resp.status_code == 200:
            games = resp.json()
            return len(games)
    except Exception:
        pass
    return None


def get_player_receiving_games(school, season, player_name):
    """
    Try to count games where a player had receiving stats.
    Uses /games/players endpoint with category=receiving.
    """
    url = f"{CFBD_BASE}/games/players"
    params = {'year': season, 'team': school, 'seasonType': 'regular', 'category': 'receiving'}
    try:
        resp = requests.get(url, headers=CFBD_HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()
        norm_target = normalize_name(player_name)
        game_count = 0

        for game in data:
            for team_data in game.get('teams', []):
                for category in team_data.get('categories', []):
                    if category.get('name') == 'receiving':
                        for athlete in category.get('types', [{}])[0].get('athletes', []):
                            norm_athlete = normalize_name(athlete.get('name', ''))
                            if norm_athlete == norm_target:
                                game_count += 1
                                break

        return game_count if game_count > 0 else None
    except Exception as e:
        print(f"      API error: {e}")
        return None


# ===========================================================================
# Load data
# ===========================================================================
print("=" * 80)
print("ADDING GAMES PLAYED TO ALL 339 WRs")
print("=" * 80)

wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
pff_matched = pd.read_csv('data/wr_pff_all_2016_2025.csv')

# Build PFF lookup: (player_name, draft_year) → player_game_count
pff_games = {}
for _, row in pff_matched.iterrows():
    pff_games[(row['player_name'], row['draft_year'])] = row['player_game_count']

print(f"\nPFF games data for: {len(pff_games)} WRs")

# ===========================================================================
# Step 1: Assign PFF game counts to matched WRs
# ===========================================================================
results = []
needs_cfbd = []

for _, wr in wr_backtest.iterrows():
    key = (wr['player_name'], wr['draft_year'])
    if key in pff_games:
        results.append({
            'player_name': wr['player_name'],
            'draft_year': wr['draft_year'],
            'pick': wr['pick'],
            'round': wr['round'],
            'college': wr['college'],
            'games_played': int(pff_games[key]),
            'games_source': 'PFF',
        })
    else:
        needs_cfbd.append(wr)

print(f"Already have games from PFF: {len(results)}")
print(f"Need CFBD lookup: {len(needs_cfbd)}")

# ===========================================================================
# Step 2: Try CFBD API for unmatched WRs
# ===========================================================================
print(f"\n{'='*60}")
print("CFBD API LOOKUPS")
print(f"{'='*60}")

cfbd_hits = 0
cfbd_misses = 0

for wr in needs_cfbd:
    name = wr['player_name']
    college = wr['college']
    draft_year = int(wr['draft_year'])
    season = draft_year - 1  # Final college season

    cfbd_school = SCHOOL_TO_CFBD.get(college)

    if cfbd_school is None:
        print(f"  {name} ({college}, {draft_year}) — D2/NAIA, no CFBD data")
        results.append({
            'player_name': name,
            'draft_year': draft_year,
            'pick': wr['pick'],
            'round': wr['round'],
            'college': college,
            'games_played': np.nan,
            'games_source': 'no_data (D2/NAIA)',
        })
        cfbd_misses += 1
        continue

    # Try to get player-specific game count first
    print(f"  {name} ({cfbd_school}, season {season})...", end=' ')
    time.sleep(0.3)  # Rate limit

    player_games = get_player_receiving_games(cfbd_school, season, name)

    if player_games is not None and player_games > 0:
        print(f"FOUND — {player_games} games with receiving stats")
        results.append({
            'player_name': name,
            'draft_year': draft_year,
            'pick': wr['pick'],
            'round': wr['round'],
            'college': college,
            'games_played': player_games,
            'games_source': 'CFBD (receiving games)',
        })
        cfbd_hits += 1
    else:
        # Fall back to team game count
        time.sleep(0.3)
        team_games = get_team_games(cfbd_school, season)
        if team_games is not None and team_games > 0:
            print(f"TEAM FALLBACK — {team_games} team games")
            results.append({
                'player_name': name,
                'draft_year': draft_year,
                'pick': wr['pick'],
                'round': wr['round'],
                'college': college,
                'games_played': team_games,
                'games_source': 'CFBD (team games)',
            })
            cfbd_hits += 1
        else:
            print("NO DATA")
            results.append({
                'player_name': name,
                'draft_year': draft_year,
                'pick': wr['pick'],
                'round': wr['round'],
                'college': college,
                'games_played': np.nan,
                'games_source': 'no_data',
            })
            cfbd_misses += 1

print(f"\nCFBD results: {cfbd_hits} found, {cfbd_misses} missing")

# ===========================================================================
# Step 3: Flag unreliable CFBD counts
# ===========================================================================
# CFBD has sparse coverage for FCS schools — game counts of 1-2 from CFBD
# for schools outside FBS are likely incomplete data, not real game counts.
# Andre Debose (Florida, 2014) is an exception — he was actually injured.
results_df = pd.DataFrame(results)

FCS_SCHOOLS = {
    'Eastern Washington', 'Grambling St.', 'Northern Iowa', 'Rhode Island',
    'Samford', 'Central Arkansas', 'North Dakota St.', 'Monmouth',
    'William & Mary', 'Princeton', 'UT Martin', 'SE Missouri St.',
    'Charleston (WV)', 'Lenoir-Rhyne', 'West Alabama', 'East Central (OK)',
    'Pennsylvania',
}

for idx, row in results_df.iterrows():
    if (row['games_source'] == 'CFBD (receiving games)' and
            row['games_played'] <= 2 and
            row['college'] in FCS_SCHOOLS):
        results_df.at[idx, 'games_source'] = 'CFBD (FCS, likely incomplete)'

results_df = results_df.sort_values(['draft_year', 'pick']).reset_index(drop=True)
results_df.to_csv('data/wr_games_played.csv', index=False)

print(f"\n{'='*60}")
print("FINAL COVERAGE")
print(f"{'='*60}")
total = len(results_df)
has_games = results_df['games_played'].notna().sum()
print(f"Total WRs: {total}")
print(f"With games played: {has_games}/{total} ({100*has_games/total:.1f}%)")
print()
print("By source:")
print(results_df['games_source'].value_counts().to_string())
print()
print("By draft year:")
for yr in sorted(results_df['draft_year'].unique()):
    yr_df = results_df[results_df['draft_year'] == yr]
    yr_has = yr_df['games_played'].notna().sum()
    print(f"  {yr}: {yr_has}/{len(yr_df)} ({100*yr_has/len(yr_df):.1f}%)")

print()
print("Missing players:")
missing = results_df[results_df['games_played'].isna()]
for _, m in missing.iterrows():
    print(f"  {m['player_name']} ({m['college']}, {m['draft_year']}) — {m['games_source']}")

print(f"\nSaved: data/wr_games_played.csv")
