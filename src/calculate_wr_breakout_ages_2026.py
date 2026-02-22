"""
Calculate WR Breakout Ages for 2026 Prospects
==============================================

Breakout Age = The age at which a WR first achieved a 20%+ dominator rating in college.

For each WR:
1. Pull ALL college seasons from CFBD (2021-2025)
2. Calculate dominator rating for EACH season
3. Find the FIRST season where dominator >= 20%
4. Calculate their age during that season

Dominator Formula: (player_rec_yards/team_rec_yards + player_rec_tds/team_rec_tds) / 2
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import re
from datetime import datetime

# CFBD API key (working key from other scripts)
API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
BASE_URL = "https://api.collegefootballdata.com"

# Load the 64 WRs missing breakout_age
print("=" * 90)
print("CALCULATING BREAKOUT AGES FOR 64 WRs")
print("=" * 90)

wr_missing = pd.read_csv('data/temp_wr_missing_breakout.csv')
print(f"Loaded {len(wr_missing)} WRs missing breakout_age")

# School name mappings (CFBD uses different names)
SCHOOL_MAPPINGS = {
    'Miami (OH)': 'Miami (OH)',
    'Miami': 'Miami',
    'Mississippi State': 'Mississippi State',
    'Florida State': 'Florida State',
    'Michigan State': 'Michigan State',
    'Oklahoma State': 'Oklahoma State',
    'Ohio State': 'Ohio State',
    'Penn State': 'Penn State',
    'Arizona State': 'Arizona State',
    'Kansas State': 'Kansas State',
    'West Virginia': 'West Virginia',
    'Virginia Tech': 'Virginia Tech',
    'Texas A&M': 'Texas A&M',
    'North Carolina': 'North Carolina',
    'South Carolina': 'South Carolina',
    'East Carolina': 'East Carolina',
    'Western Michigan': 'Western Michigan',
    'Northern Illinois': 'Northern Illinois',
}

def normalize_name(name):
    """Normalize player name for matching"""
    # Remove suffixes
    name = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', '', name, flags=re.IGNORECASE)
    # Remove periods
    name = name.replace('.', '')
    # Lowercase
    name = name.lower().strip()
    return name

def get_school_name(school):
    """Get CFBD-compatible school name"""
    return SCHOOL_MAPPINGS.get(school, school)

def names_match(name1, name2):
    """Check if two player names match (flexible matching)"""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)

    # Exact match
    if n1 == n2:
        return True

    # One contains the other
    if n1 in n2 or n2 in n1:
        return True

    # Split into parts
    parts1 = n1.split()
    parts2 = n2.split()

    # Last name match with similar first name
    if len(parts1) >= 2 and len(parts2) >= 2:
        if parts1[-1] == parts2[-1]:  # Same last name
            # Check if first names are similar
            if parts1[0][:3] == parts2[0][:3]:  # First 3 chars match
                return True
            # One first name contains the other
            if parts1[0] in parts2[0] or parts2[0] in parts1[0]:
                return True

    # Just last name match (for unique last names)
    if len(parts1) >= 1 and len(parts2) >= 1:
        if parts1[-1] == parts2[-1] and len(parts1[-1]) > 4:
            return True

    return False

def calculate_age_at_season_start(birthdate, season_year):
    """Calculate player's age at the start of a college season (August 1)"""
    if pd.isna(birthdate) or birthdate == 'MISSING' or birthdate is None:
        return None

    try:
        # Parse birthdate
        if isinstance(birthdate, str):
            birth_dt = datetime.strptime(birthdate.split()[0], '%Y-%m-%d')
        else:
            birth_dt = birthdate

        # Season starts around August 1
        season_start = datetime(season_year, 8, 1)

        age = season_start.year - birth_dt.year
        # Adjust if birthday hasn't occurred yet
        if (season_start.month, season_start.day) < (birth_dt.month, birth_dt.day):
            age -= 1

        return age
    except:
        return None

def fetch_team_season_stats(school, season):
    """Fetch all receiving stats for a team in a season"""
    school_cfbd = get_school_name(school)

    try:
        url = f"{BASE_URL}/stats/player/season"
        params = {
            'year': season,
            'team': school_cfbd,
            'category': 'receiving',
        }

        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data:
            return None

        # Group stats by player
        player_stats = {}
        for stat in data:
            player = stat.get('player', '')
            stat_type = stat.get('statType', '')
            stat_val = float(stat.get('stat', 0)) if stat.get('stat') else 0

            if player not in player_stats:
                player_stats[player] = {'YDS': 0, 'TD': 0, 'REC': 0}

            if stat_type == 'YDS':
                player_stats[player]['YDS'] = stat_val
            elif stat_type == 'TD':
                player_stats[player]['TD'] = stat_val
            elif stat_type == 'REC':
                player_stats[player]['REC'] = stat_val

        # Calculate team totals
        team_yards = sum(p['YDS'] for p in player_stats.values())
        team_tds = sum(p['TD'] for p in player_stats.values())

        return {
            'players': player_stats,
            'team_yards': team_yards,
            'team_tds': team_tds,
        }

    except Exception as e:
        print(f"    Error fetching {school} {season}: {e}")
        return None

def calculate_dominator(player_yards, player_tds, team_yards, team_tds):
    """Calculate dominator rating"""
    if team_yards is None or team_yards == 0:
        return None

    yards_share = player_yards / team_yards

    if team_tds is None or team_tds == 0:
        td_share = yards_share  # Use yards share if no TDs
    else:
        td_share = player_tds / team_tds

    return (yards_share + td_share) / 2

# Process each player
print("\n" + "=" * 90)
print("FETCHING MULTI-SEASON STATS FROM CFBD")
print("=" * 90)

results = []
seasons_to_check = [2021, 2022, 2023, 2024, 2025]

for idx, row in wr_missing.iterrows():
    player_name = row['player_name']
    school = row['school']
    draft_age = row['age']
    birthdate = row['birthdate']

    print(f"\n[{idx+1}/{len(wr_missing)}] {player_name} ({school})")

    player_data = {
        'player_name': player_name,
        'college': school,
        'birthdate': birthdate,
        'draft_age': draft_age,
        'seasons_data': [],
        'breakout_season': None,
        'breakout_age': None,
        'peak_dominator': 0,
        'peak_dominator_season': None,
        'status': 'processing',
    }

    # Check each season
    for season in seasons_to_check:
        team_data = fetch_team_season_stats(school, season)

        if team_data is None:
            continue

        team_yards = team_data['team_yards']
        team_tds = team_data['team_tds']

        if team_yards == 0:
            continue

        # Find the player in the data
        found_player = None
        found_stats = None

        for cfbd_name, stats in team_data['players'].items():
            if names_match(player_name, cfbd_name):
                found_player = cfbd_name
                found_stats = stats
                break

        if found_stats and found_stats['YDS'] > 0:
            player_yards = found_stats['YDS']
            player_tds = found_stats['TD']

            dominator = calculate_dominator(player_yards, player_tds, team_yards, team_tds)
            age_at_season = calculate_age_at_season_start(birthdate, season)

            season_data = {
                'season': season,
                'cfbd_name': found_player,
                'player_rec_yards': player_yards,
                'player_rec_tds': player_tds,
                'team_rec_yards': team_yards,
                'team_rec_tds': team_tds,
                'dominator': dominator,
                'dominator_pct': dominator * 100 if dominator else None,
                'age_at_season': age_at_season,
            }
            player_data['seasons_data'].append(season_data)

            # Track peak dominator
            if dominator and dominator > player_data['peak_dominator']:
                player_data['peak_dominator'] = dominator
                player_data['peak_dominator_season'] = season

            # Check for breakout (first 20%+ dominator)
            if dominator and dominator >= 0.20 and player_data['breakout_season'] is None:
                player_data['breakout_season'] = season
                player_data['breakout_age'] = age_at_season

            bo_marker = " <-- BREAKOUT!" if dominator and dominator >= 0.20 and season == player_data['breakout_season'] else ""
            print(f"  {season}: {player_yards:.0f} yds, {player_tds:.0f} TDs | Dom: {dominator*100:.1f}% | Age: {age_at_season}{bo_marker}")

        time.sleep(0.25)  # Rate limiting

    # Set status
    if len(player_data['seasons_data']) == 0:
        player_data['status'] = 'no_cfbd_data'
    elif player_data['breakout_season']:
        player_data['status'] = 'breakout_found'
    else:
        player_data['status'] = 'never_hit_20pct'

    results.append(player_data)

# Convert to DataFrame
print("\n" + "=" * 90)
print("RESULTS SUMMARY")
print("=" * 90)

# Create output DataFrame
output_rows = []
for r in results:
    output_rows.append({
        'player_name': r['player_name'],
        'college': r['college'],
        'birthdate': r['birthdate'],
        'draft_age': r['draft_age'],
        'seasons_found': len(r['seasons_data']),
        'seasons_data': json.dumps(r['seasons_data']),
        'breakout_season': r['breakout_season'],
        'breakout_age': r['breakout_age'],
        'peak_dominator': r['peak_dominator'] * 100 if r['peak_dominator'] else None,
        'peak_dominator_season': r['peak_dominator_season'],
        'status': r['status'],
    })

output_df = pd.DataFrame(output_rows)

# Status counts
status_counts = output_df['status'].value_counts()
print(f"\nStatus breakdown:")
for status, count in status_counts.items():
    print(f"  {status}: {count}")

# Save to CSV
output_df.to_csv('data/wr_breakout_ages_2026.csv', index=False)
print(f"\nSaved to data/wr_breakout_ages_2026.csv")

# Show breakout age distribution
breakout_found = output_df[output_df['status'] == 'breakout_found']
print(f"\n{'Breakout Age Distribution':}")
if len(breakout_found) > 0:
    print(breakout_found['breakout_age'].value_counts().sort_index())

# Show 10 examples with full math
print("\n" + "=" * 90)
print("10 EXAMPLE CALCULATIONS (with full math)")
print("=" * 90)

examples = [r for r in results if r['status'] == 'breakout_found'][:10]
for i, ex in enumerate(examples, 1):
    print(f"\n--- Example {i}: {ex['player_name']} ({ex['college']}) ---")
    print(f"Birthdate: {ex['birthdate']}")
    print(f"Draft Age: {ex['draft_age']}")
    print(f"\nSeason-by-season:")

    for s in ex['seasons_data']:
        yards_share = s['player_rec_yards'] / s['team_rec_yards'] if s['team_rec_yards'] else 0
        td_share = s['player_rec_tds'] / s['team_rec_tds'] if s['team_rec_tds'] else yards_share
        dom = (yards_share + td_share) / 2

        bo_marker = " <-- BREAKOUT" if s['dominator_pct'] and s['dominator_pct'] >= 20 and s['season'] == ex['breakout_season'] else ""
        print(f"  {s['season']}: {s['player_rec_yards']:.0f} yds / {s['team_rec_yards']:.0f} team = {yards_share*100:.1f}% | "
              f"{s['player_rec_tds']:.0f} TDs / {s['team_rec_tds']:.0f} team = {td_share*100:.1f}% | "
              f"Dom = ({yards_share*100:.1f} + {td_share*100:.1f})/2 = {dom*100:.1f}% | Age: {s['age_at_season']}{bo_marker}")

    print(f"\n=> Breakout Season: {ex['breakout_season']}")
    print(f"=> Breakout Age: {ex['breakout_age']}")
    print(f"=> Peak Dominator: {ex['peak_dominator']*100:.1f}% ({ex['peak_dominator_season']})")

# Also show players who never hit 20%
print("\n" + "=" * 90)
print("PLAYERS WHO NEVER HIT 20% DOMINATOR")
print("=" * 90)

never_hit = [r for r in results if r['status'] == 'never_hit_20pct']
for r in never_hit:
    print(f"\n{r['player_name']} ({r['college']})")
    print(f"  Peak Dominator: {r['peak_dominator']*100:.1f}% ({r['peak_dominator_season']})")
    for s in r['seasons_data']:
        print(f"    {s['season']}: {s['dominator_pct']:.1f}%")

# Show players with no CFBD data
print("\n" + "=" * 90)
print("PLAYERS WITH NO CFBD DATA (need manual research)")
print("=" * 90)

no_data = [r for r in results if r['status'] == 'no_cfbd_data']
for r in no_data:
    print(f"  {r['player_name']} ({r['college']})")

print("\n" + "=" * 90)
print("DONE!")
print("=" * 90)
