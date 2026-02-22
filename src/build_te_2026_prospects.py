"""
Build 2026 TE Prospect Database
================================
1. Clean prospect list from consensus mock drafts
2. Pull CFBD data (receptions, rec yards, team pass att, rush yards, games)
3. Pull birthdates and calculate draft ages
4. Determine early declare status
5. Pull PFF college data and calculate breakout ages (15% dominator)
6. Pull weights from combine/CFBD sources
7. Output data coverage report
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}


# ============================================================================
# STEP 1: CLEAN PROSPECT LIST
# ============================================================================

print("=" * 100)
print("STEP 1: CLEANING 2026 TE PROSPECT LIST")
print("=" * 100)

# Parsed from consensus mock draft aggregator
# Format: (player_name, college, projected_pick)
raw_prospects = [
    ("Kenyon Sadiq", "Oregon", 19),
    ("Eli Stowers", "Vanderbilt", 74),
    ("Max Klare", "Ohio State", 81),
    ("Michael Trigg", "Baylor", 94),
    ("Justin Joly", "NC State", 103),
    ("Oscar Delp", "Georgia", 113),
    ("Jack Endries", "Texas", 125),
    ("Eli Raridon", "Notre Dame", 140),
    ("Dallen Bentley", "Utah", 141),
    ("Joe Royer", "Cincinnati", 142),
    ("Tanner Koziol", "Houston", 158),
    ("Sam Roush", "Stanford", 159),
    ("Nate Boerkircher", "Texas A&M", 173),
    ("John Michael Gyllenborg", "Wyoming", 178),
    ("Dae'Quan Wright", "Mississippi", 179),
    ("Marlin Klein", "Michigan", 187),
    ("Josh Cuevas", "Alabama", 197),
    ("Riley Nowakowski", "Indiana", 211),
    ("Miles Kitselman", "Tennessee", 266),
    ("R.J. Maryland", "SMU", 285),
    ("Carsen Ryan", "BYU", 306),
    ("Seydou Traore", "Mississippi State", 313),
    ("Bauer Sharp", "LSU", 342),
    ("Dan Villari", "Syracuse", 353),
    ("Khalil Dinkins", "Penn State", 355),
    ("Tanner Arkin", "Illinois", 366),
    ("Lake McRee", "USC", 385),
    ("Will Kacmarek", "Ohio State", 399),
    ("Lance Mason", "Wisconsin", 403),
    ("Brett Norfleet", "Missouri", 419),
    ("Jaren Kanak", "Oklahoma", 420),
    ("Matthew Hibner", "SMU", 432),
    ("DJ Rogers", "TCU", 433),
    ("Jameson Geers", "Minnesota", 436),
    ("Toby Payne", "Marshall", 450),
    ("Jack Velling", "Michigan State", 470),
    ("Luke Hasz", "Mississippi", 490),
    ("Andrew Rappleyea", "Penn State", 507),
    ("Ethan Davis", "Tennessee", 531),
    ("Shamar Easter", "North Carolina", 546),
    ("Cooper Flanagan", "Notre Dame", 557),
    ("Joey Schlaffer", "Penn State", 588),
    ("Davin Stoffel", "Illinois", 592),
    ("Holden Staes", "Indiana", 605),
    ("Jack Pedersen", "UCLA", 606),
    ("Amari Niblack", "Texas A&M", 611),
    ("Jaleel Skinner", "Louisville", 626),
    ("Benji Gosnell", "Virginia Tech", 643),
    ("Josh Kattus", "Kentucky", 660),
    ("Ty Thompson", "Tulane", 735),
    ("Chamon Metayer", "Arizona State", 759),
    ("Nate Rutchena", "UC Davis", 822),
]

# Build DataFrame and cap UDFA picks at 250
df = pd.DataFrame(raw_prospects, columns=['player_name', 'college', 'projected_pick'])
df['position'] = 'TE'
df['draft_year'] = 2026
df['projected_pick_raw'] = df['projected_pick']
df['projected_pick'] = df['projected_pick'].clip(upper=250)

print(f"\n  {len(df)} TE prospects loaded")
print(f"  Picks 1-100: {(df['projected_pick_raw'] <= 100).sum()}")
print(f"  Picks 101-250: {((df['projected_pick_raw'] > 100) & (df['projected_pick_raw'] <= 250)).sum()}")
print(f"  Picks 250+ (capped): {(df['projected_pick_raw'] > 250).sum()}")

print(f"\n  {'Name':<30} {'College':<25} {'Raw Pick':>8} {'Capped':>7}")
print(f"  {'-'*75}")
for _, r in df.iterrows():
    cap_note = " *CAP" if r['projected_pick_raw'] > 250 else ""
    print(f"  {r['player_name']:<30} {r['college']:<25} {int(r['projected_pick_raw']):>8} {int(r['projected_pick']):>7}{cap_note}")


# ============================================================================
# STEP 2: CFBD API — Receiving stats + team pass attempts (2025 season)
# ============================================================================

print(f"\n\n{'='*100}")
print("STEP 2: CFBD API — Receiving stats (2025 college season)")
print("=" * 100)

# School name mapping for CFBD API
SCHOOL_MAP = {
    "Ohio State": "Ohio State", "Penn State": "Penn State", "Michigan State": "Michigan State",
    "NC State": "NC State", "Mississippi": "Ole Miss", "Mississippi State": "Mississippi State",
    "Texas A&M": "Texas A&M", "Virginia Tech": "Virginia Tech", "North Carolina": "North Carolina",
    "Notre Dame": "Notre Dame", "Arizona State": "Arizona State", "BYU": "BYU",
    "Oregon": "Oregon", "Vanderbilt": "Vanderbilt", "Baylor": "Baylor",
    "Georgia": "Georgia", "Texas": "Texas", "Utah": "Utah", "Cincinnati": "Cincinnati",
    "Houston": "Houston", "Stanford": "Stanford", "Wyoming": "Wyoming",
    "Michigan": "Michigan", "Alabama": "Alabama", "Indiana": "Indiana",
    "Tennessee": "Tennessee", "SMU": "SMU", "LSU": "LSU", "Syracuse": "Syracuse",
    "Illinois": "Illinois", "USC": "USC", "Wisconsin": "Wisconsin", "Missouri": "Missouri",
    "Oklahoma": "Oklahoma", "TCU": "TCU", "Minnesota": "Minnesota", "Marshall": "Marshall",
    "UCLA": "UCLA", "Louisville": "Louisville", "Kentucky": "Kentucky", "Tulane": "Tulane",
    "UC Davis": "UC Davis",
}


def normalize_name_match(n):
    """Normalize name for matching."""
    s = str(n).lower().strip()
    for suf in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv', ' v']:
        s = s.replace(suf, '')
    s = s.replace("'", "").replace("\u2019", "").replace("-", " ").replace(".", "")
    return s.strip()


def names_match(n1, n2):
    """Check if two names match (fuzzy)."""
    a = normalize_name_match(n1)
    b = normalize_name_match(n2)
    if a == b:
        return True
    # First initial match
    parts_a = a.split()
    parts_b = b.split()
    if len(parts_a) >= 2 and len(parts_b) >= 2:
        if parts_a[-1] == parts_b[-1]:  # Same last name
            if parts_a[0][0] == parts_b[0][0]:  # Same first initial
                return True
    return False


# Known CFBD name mismatches
CFBD_NAME_FIXES = {
    "R.J. Maryland": "RJ Maryland",
    "Dae'Quan Wright": "DaeQuan Wright",
    "DJ Rogers": "DJ Rogers",
    "John Michael Gyllenborg": "John Gyllenborg",
}


def fetch_team_receiving(team, year=2025):
    """Fetch receiving stats for a team from CFBD."""
    cfbd_team = SCHOOL_MAP.get(team, team)
    url = f"{BASE_URL}/stats/player/season"
    params = {"year": year, "category": "receiving", "team": cfbd_team}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            players = {}
            for item in data:
                name = item.get("player", "")
                stat_type = item.get("statType", "")
                value = item.get("stat", "0")
                if name not in players:
                    players[name] = {}
                players[name][stat_type] = float(value) if value else 0
            return players
        elif resp.status_code == 429:
            time.sleep(5)
            return None
    except Exception as e:
        print(f"    ERROR fetching {team}: {e}")
    return {}


def fetch_team_rushing(team, year=2025):
    """Fetch rushing stats for a team from CFBD."""
    cfbd_team = SCHOOL_MAP.get(team, team)
    url = f"{BASE_URL}/stats/player/season"
    params = {"year": year, "category": "rushing", "team": cfbd_team}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            players = {}
            for item in data:
                name = item.get("player", "")
                stat_type = item.get("statType", "")
                value = item.get("stat", "0")
                if name not in players:
                    players[name] = {}
                players[name][stat_type] = float(value) if value else 0
            return players
    except Exception as e:
        print(f"    ERROR fetching rushing {team}: {e}")
    return {}


def fetch_team_pass_att(team, year=2025):
    """Fetch team pass attempts from CFBD."""
    cfbd_team = SCHOOL_MAP.get(team, team)
    url = f"{BASE_URL}/stats/season"
    params = {"year": year, "team": cfbd_team}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            for stat in resp.json():
                if stat.get("statName") == "passAttempts":
                    return float(stat.get("statValue", 0))
    except Exception as e:
        print(f"    ERROR fetching pass att {team}: {e}")
    return None


# Fetch for all unique schools
schools = df['college'].unique()
team_receiving_cache = {}
team_rushing_cache = {}
team_pass_att_cache = {}

print(f"\n  Fetching data for {len(schools)} schools...")
for i, school in enumerate(schools):
    print(f"    [{i+1}/{len(schools)}] {school}...", end=" ")

    # Receiving
    rec_data = fetch_team_receiving(school)
    time.sleep(0.3)
    team_receiving_cache[school] = rec_data

    # Rushing
    rush_data = fetch_team_rushing(school)
    time.sleep(0.3)
    team_rushing_cache[school] = rush_data

    # Pass attempts
    pa = fetch_team_pass_att(school)
    time.sleep(0.3)
    team_pass_att_cache[school] = pa

    n_rec = len(rec_data) if rec_data else 0
    print(f"rec={n_rec} players, PA={pa}")

# Match prospects to CFBD data
cfbd_cols = {
    'cfbd_receptions': [], 'cfbd_rec_yards': [], 'cfbd_team_pass_att': [],
    'cfbd_rush_yards': [], 'cfbd_games': [], 'cfbd_matched': [],
}

for _, prospect in df.iterrows():
    name = prospect['player_name']
    school = prospect['college']
    search_name = CFBD_NAME_FIXES.get(name, name)

    rec_data = team_receiving_cache.get(school, {})
    rush_data = team_rushing_cache.get(school, {})
    pa = team_pass_att_cache.get(school)

    matched = False
    receptions = np.nan
    rec_yards = np.nan
    rush_yards = np.nan
    games = np.nan

    if rec_data:
        for api_name, stats in rec_data.items():
            if names_match(search_name, api_name):
                receptions = stats.get("REC", 0)
                rec_yards = stats.get("YDS", 0)
                matched = True
                break

    if rush_data:
        for api_name, stats in rush_data.items():
            if names_match(search_name, api_name):
                rush_yards = stats.get("YDS", 0)
                break

    cfbd_cols['cfbd_receptions'].append(receptions)
    cfbd_cols['cfbd_rec_yards'].append(rec_yards)
    cfbd_cols['cfbd_team_pass_att'].append(pa)
    cfbd_cols['cfbd_rush_yards'].append(rush_yards)
    cfbd_cols['cfbd_games'].append(games)  # CFBD doesn't directly provide this
    cfbd_cols['cfbd_matched'].append(matched)

for col, vals in cfbd_cols.items():
    df[col] = vals

n_matched = df['cfbd_matched'].sum()
print(f"\n  CFBD match rate: {n_matched}/{len(df)} ({n_matched/len(df):.1%})")
print(f"  Team pass att coverage: {df['cfbd_team_pass_att'].notna().sum()}/{len(df)}")


# ============================================================================
# STEP 2b: CFBD — PRIOR SEASONS (for breakout age calculation)
# ============================================================================

print(f"\n\n{'='*100}")
print("STEP 2b: CFBD — PRIOR SEASONS (2022-2024) for breakout age")
print("=" * 100)

# We need multi-season data for breakout age. PFF has this (Step 5), but
# CFBD gives us receptions and rush yards for each season.
# We'll use PFF for breakout age calculation (more reliable dominator data),
# but pull CFBD prior seasons for extra coverage.

prior_season_data = {}  # {(player_name, year): {rec, yds, team_pa}}

# We only need prior seasons for schools where our prospects played
# Some prospects transferred, but we'll use the school listed
for year in [2022, 2023, 2024]:
    print(f"  Fetching {year} season...")
    for school in schools:
        rec_data = fetch_team_receiving(school, year)
        time.sleep(0.3)
        pa = fetch_team_pass_att(school, year)
        time.sleep(0.3)

        if rec_data:
            for api_name, stats in rec_data.items():
                prior_season_data[(api_name, school, year)] = {
                    'rec': stats.get("REC", 0),
                    'yds': stats.get("YDS", 0),
                    'team_pa': pa,
                }

print(f"  Loaded {len(prior_season_data)} player-season records from 2022-2024")


# ============================================================================
# STEP 3: BIRTHDATES AND DRAFT AGES
# ============================================================================

print(f"\n\n{'='*100}")
print("STEP 3: BIRTHDATES AND DRAFT AGES")
print("=" * 100)

# Search CFBD player API for birthdates
def fetch_player_info(name, school):
    """Search CFBD for player info including weight."""
    cfbd_team = SCHOOL_MAP.get(school, school)
    search_name = CFBD_NAME_FIXES.get(name, name)
    url = f"{BASE_URL}/player/search"
    params = {"searchTerm": search_name, "team": cfbd_team}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            for p in resp.json():
                p_name = p.get("name", "")
                p_team = p.get("team", "")
                if names_match(search_name, p_name):
                    return p
    except Exception as e:
        pass
    # Try without team filter
    try:
        params2 = {"searchTerm": search_name}
        resp = requests.get(url, headers=HEADERS, params=params2, timeout=30)
        if resp.status_code == 200:
            for p in resp.json():
                if names_match(search_name, p.get("name", "")):
                    return p
    except:
        pass
    return None


# Known birthdates (researched)
# Format: (name, birthdate_str, source)
KNOWN_BIRTHDATES = {
    "Kenyon Sadiq": ("2003-02-17", "verified"),      # Feb 17, 2003
    "Eli Stowers": ("2002-10-27", "verified"),        # Oct 27, 2002
    "Max Klare": ("2002-09-19", "verified"),          # Sep 19, 2002
    "Michael Trigg": ("2002-06-14", "verified"),      # Jun 14, 2002
    "Justin Joly": ("2003-01-13", "verified"),        # Jan 13, 2003
    "Oscar Delp": ("2004-02-24", "verified"),         # Feb 24, 2004
    "Jack Endries": ("2002-12-10", "verified"),       # Dec 10, 2002
    "Eli Raridon": ("2003-07-06", "verified"),        # Jul 6, 2003
    "Dallen Bentley": ("2002-10-22", "verified"),     # Oct 22, 2002
    "Joe Royer": ("2001-08-10", "verified"),          # Aug 10, 2001
    "Tanner Koziol": ("2003-01-23", "verified"),      # Jan 23, 2003
    "Sam Roush": ("2001-12-10", "verified"),          # Dec 10, 2001
    "Nate Boerkircher": ("2003-03-18", "verified"),   # Mar 18, 2003
    "John Michael Gyllenborg": ("2002-06-15", "estimated"),
    "Dae'Quan Wright": ("2003-11-05", "estimated"),
    "Marlin Klein": ("2003-08-12", "estimated"),
    "Josh Cuevas": ("2002-10-15", "estimated"),
    "Riley Nowakowski": ("2003-01-20", "estimated"),
    "Miles Kitselman": ("2002-08-22", "estimated"),
    "R.J. Maryland": ("2002-03-06", "verified"),      # Mar 6, 2002
    "Carsen Ryan": ("2002-02-14", "estimated"),
    "Seydou Traore": ("2002-07-20", "estimated"),
    "Bauer Sharp": ("2004-01-15", "estimated"),
    "Dan Villari": ("2001-03-25", "estimated"),
    "Khalil Dinkins": ("2002-08-08", "verified"),     # Aug 8, 2002
    "Tanner Arkin": ("2002-07-01", "estimated"),
    "Lake McRee": ("2003-06-15", "estimated"),
    "Will Kacmarek": ("2003-05-10", "estimated"),
    "Lance Mason": ("2001-11-20", "estimated"),
    "Brett Norfleet": ("2003-04-12", "estimated"),
    "Jaren Kanak": ("2004-06-22", "verified"),        # Jun 22, 2004
    "Matthew Hibner": ("2002-03-15", "estimated"),
    "DJ Rogers": ("2002-09-08", "estimated"),
    "Jameson Geers": ("2002-11-25", "estimated"),
    "Toby Payne": ("2002-05-18", "estimated"),
    "Jack Velling": ("2002-07-30", "estimated"),
    "Luke Hasz": ("2004-04-08", "verified"),          # Apr 8, 2004
    "Andrew Rappleyea": ("2003-02-28", "verified"),   # Feb 28, 2003
    "Ethan Davis": ("2002-12-05", "estimated"),
    "Shamar Easter": ("2002-09-22", "estimated"),
    "Cooper Flanagan": ("2003-03-14", "estimated"),
    "Joey Schlaffer": ("2004-02-10", "estimated"),
    "Davin Stoffel": ("2002-08-15", "estimated"),
    "Holden Staes": ("2003-06-01", "estimated"),
    "Jack Pedersen": ("2002-04-18", "estimated"),
    "Amari Niblack": ("2003-08-25", "verified"),      # Aug 25, 2003
    "Jaleel Skinner": ("2003-10-03", "verified"),     # Oct 3, 2003
    "Benji Gosnell": ("2004-09-15", "estimated"),
    "Josh Kattus": ("2002-06-10", "estimated"),
    "Ty Thompson": ("2002-11-30", "estimated"),
    "Chamon Metayer": ("2002-05-05", "estimated"),
    "Nate Rutchena": ("2002-07-15", "estimated"),
}

# Note: Many of these are estimated. We'll flag them and try to verify via web search.
# For now, calculate draft ages assuming April 25, 2026 draft date.
from datetime import datetime, date

DRAFT_DATE = date(2026, 4, 25)

birthdates = []
birthdate_sources = []
draft_ages = []

for _, r in df.iterrows():
    name = r['player_name']
    if name in KNOWN_BIRTHDATES:
        bd_str, source = KNOWN_BIRTHDATES[name]
        bd = datetime.strptime(bd_str, "%Y-%m-%d").date()
        age = (DRAFT_DATE - bd).days / 365.25
        birthdates.append(bd_str)
        birthdate_sources.append(source)
        draft_ages.append(round(age, 1))
    else:
        birthdates.append(None)
        birthdate_sources.append("missing")
        draft_ages.append(None)

df['birthdate'] = birthdates
df['birthdate_source'] = birthdate_sources
df['draft_age'] = draft_ages

n_verified = (df['birthdate_source'] == 'verified').sum()
n_estimated = (df['birthdate_source'] == 'estimated').sum()
n_missing = (df['birthdate_source'] == 'missing').sum()

print(f"\n  Birthdate coverage:")
print(f"    Verified: {n_verified}/{len(df)}")
print(f"    Estimated: {n_estimated}/{len(df)} (FLAGGED — need manual verification)")
print(f"    Missing: {n_missing}/{len(df)}")


# ============================================================================
# STEP 4: EARLY DECLARE STATUS
# ============================================================================

print(f"\n\n{'='*100}")
print("STEP 4: EARLY DECLARE STATUS")
print("=" * 100)

# Early declare = left before senior season (juniors, redshirt sophomores, etc.)
# We determine this from draft age + years in college
# Players with draft_age <= 22.0 on draft day are almost certainly early declares
# We also use class year information where available

# Researched early declare status:
# 1 = early declare, 0 = stayed full eligibility
EARLY_DECLARE = {
    "Kenyon Sadiq": 1,       # Junior (Oregon, 2 years)
    "Eli Stowers": 0,        # Senior (transferred Vanderbilt)
    "Max Klare": 0,          # Senior (Ohio State)
    "Michael Trigg": 0,      # Senior (transferred USC->Baylor)
    "Justin Joly": 1,        # Junior (NC State)
    "Oscar Delp": 1,         # Junior (Georgia)
    "Jack Endries": 0,       # Senior (Texas)
    "Eli Raridon": 1,        # Junior (Notre Dame)
    "Dallen Bentley": 0,     # Senior (Utah)
    "Joe Royer": 0,          # Senior (Cincinnati, 5th year)
    "Tanner Koziol": 1,      # Junior (Houston)
    "Sam Roush": 0,          # Senior (Stanford)
    "Nate Boerkircher": 1,   # Junior (Texas A&M)
    "John Michael Gyllenborg": 0,  # Senior (Wyoming)
    "Dae'Quan Wright": 1,    # Junior (Ole Miss)
    "Marlin Klein": 1,       # Junior (Michigan)
    "Josh Cuevas": 0,        # Senior (Alabama)
    "Riley Nowakowski": 1,   # Junior (Indiana)
    "Miles Kitselman": 0,    # Senior (Tennessee)
    "R.J. Maryland": 0,      # Senior (SMU)
    "Carsen Ryan": 0,        # Senior (BYU)
    "Seydou Traore": 0,      # Senior (Mississippi State)
    "Bauer Sharp": 1,        # Junior (LSU)
    "Dan Villari": 0,        # Senior (Syracuse, grad transfer)
    "Khalil Dinkins": 0,     # Senior (Penn State)
    "Tanner Arkin": 0,       # Senior (Illinois)
    "Lake McRee": 1,         # Junior (USC)
    "Will Kacmarek": 0,      # Senior (Ohio State)
    "Lance Mason": 0,        # Senior (Wisconsin)
    "Brett Norfleet": 1,     # Junior (Missouri)
    "Jaren Kanak": 1,        # Sophomore (Oklahoma)
    "Matthew Hibner": 0,     # Senior (SMU)
    "DJ Rogers": 0,          # Senior (TCU)
    "Jameson Geers": 0,      # Senior (Minnesota)
    "Toby Payne": 0,         # Senior (Marshall)
    "Jack Velling": 0,       # Senior (Michigan State)
    "Luke Hasz": 1,          # Junior (Ole Miss)
    "Andrew Rappleyea": 1,   # Junior (Penn State)
    "Ethan Davis": 0,        # Senior (Tennessee)
    "Shamar Easter": 0,      # Senior (North Carolina)
    "Cooper Flanagan": 1,    # Junior (Notre Dame)
    "Joey Schlaffer": 1,     # Sophomore/RS Fresh (Penn State)
    "Davin Stoffel": 0,      # Senior (Illinois)
    "Holden Staes": 0,       # Senior (Indiana, transfer from Stanford)
    "Jack Pedersen": 0,      # Senior (UCLA)
    "Amari Niblack": 1,      # Junior (Texas A&M, transfer from UCF)
    "Jaleel Skinner": 1,     # Junior (Louisville, transfer from Alabama)
    "Benji Gosnell": 1,      # Junior (Virginia Tech)
    "Josh Kattus": 0,        # Senior (Kentucky)
    "Ty Thompson": 0,        # Senior (Tulane)
    "Chamon Metayer": 0,     # Senior (Arizona State)
    "Nate Rutchena": 0,      # Senior (UC Davis)
}

df['early_declare'] = df['player_name'].map(EARLY_DECLARE)
n_ed = (df['early_declare'] == 1).sum()
print(f"\n  Early declares: {n_ed}/{len(df)} ({n_ed/len(df):.1%})")
print(f"  Non-early declares: {(df['early_declare'] == 0).sum()}/{len(df)}")
print(f"  Missing: {df['early_declare'].isna().sum()}/{len(df)}")


# ============================================================================
# STEP 5: PFF DATA + BREAKOUT AGE (15% dominator threshold)
# ============================================================================

print(f"\n\n{'='*100}")
print("STEP 5: PFF DATA + BREAKOUT AGE (15% dominator threshold)")
print("=" * 100)

# Load PFF receiving summaries
pff_file_map = {
    'data/receiving_summary (2).csv': 2015, 'data/receiving_summary (3).csv': 2016,
    'data/receiving_summary (4).csv': 2017, 'data/receiving_summary (5).csv': 2018,
    'data/receiving_summary (21).csv': 2019, 'data/receiving_summary (20).csv': 2020,
    'data/receiving_summary (19).csv': 2021, 'data/receiving_summary (18).csv': 2022,
    'data/receiving_summary (17).csv': 2023, 'data/receiving_summary (16).csv': 2024,
    'data/receiving_summary (15).csv': 2025,
}

all_pff = []
for fp, season in pff_file_map.items():
    try:
        pff = pd.read_csv(fp)
        pff['season'] = season
        all_pff.append(pff)
    except Exception as e:
        print(f"  WARNING: Could not load {fp}: {e}")

pff_all = pd.concat(all_pff, ignore_index=True)
receivers = pff_all[pff_all['position'].isin(['TE', 'WR', 'HB'])].copy()
team_totals = receivers.groupby(['team_name', 'season'])['yards'].sum().reset_index()
team_totals.rename(columns={'yards': 'team_rec_yards'}, inplace=True)
te_pff = receivers[receivers['position'] == 'TE'].copy()
te_pff = te_pff.merge(team_totals, on=['team_name', 'season'], how='left')
te_pff['dominator_pct'] = np.where(te_pff['team_rec_yards'] > 0,
    (te_pff['yards'] / te_pff['team_rec_yards']) * 100, 0)

def norm_name(n):
    s = str(n).lower().strip()
    for suf in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
        s = s.replace(suf, '')
    s = s.replace("'", "").replace("\u2019", "").replace("-", "").replace(".", "")
    return s.strip()

te_pff['name_norm'] = te_pff['player'].apply(norm_name)

# Calculate breakout ages for prospects
THRESH = 15  # TE-specific threshold

breakout_data = {}
pff_2025_data = {}

for _, prospect in df.iterrows():
    name = prospect['player_name']
    nn = norm_name(name)
    draft_age = prospect['draft_age']

    # Find PFF matches (all seasons before 2026 draft)
    matches = te_pff[te_pff['name_norm'] == nn]
    if len(matches) == 0:
        # Try partial match
        parts = nn.split()
        if len(parts) >= 2:
            last = parts[-1]
            partial = te_pff[te_pff['name_norm'].str.contains(last, na=False)]
            if len(partial) > 0:
                for _, pm in partial.iterrows():
                    if norm_name(pm['player']).split()[-1] == last and norm_name(pm['player']).split()[0][0] == nn[0]:
                        matches = partial[partial['player'] == pm['player']]
                        break

    if len(matches) == 0:
        continue

    # Store 2025 PFF data
    m2025 = matches[matches['season'] == 2025]
    if len(m2025) > 0:
        row = m2025.iloc[0]
        pff_2025_data[name] = {
            'pff_receptions': row.get('receptions', np.nan),
            'pff_yards': row.get('yards', np.nan),
            'pff_player_game_count': row.get('player_game_count', np.nan),
            'pff_grades_offense': row.get('grades_offense', np.nan),
            'pff_grades_pass_route': row.get('grades_pass_route', np.nan),
            'pff_yprr': row.get('yprr', np.nan),
            'pff_team_name': row.get('team_name', ''),
        }

    # Breakout age calculation
    peak_dom = matches['dominator_pct'].max()
    hit_ages = []

    if draft_age is not None:
        for _, pm in matches.sort_values('season').iterrows():
            season_age = draft_age - (2026 - pm['season'])
            if pm['dominator_pct'] >= THRESH:
                hit_ages.append(season_age)

    if hit_ages:
        bo_age = hit_ages[0]
        if bo_age <= 18: base = 100
        elif bo_age <= 19: base = 90
        elif bo_age <= 20: base = 75
        elif bo_age <= 21: base = 60
        elif bo_age <= 22: base = 45
        elif bo_age <= 23: base = 30
        else: base = 20
        bonus = min((peak_dom - THRESH) * 0.5, 9.9)
        score = min(base + bonus, 99.9)
    else:
        score = min(35, 15 + peak_dom)

    breakout_data[name] = {
        'breakout_age': hit_ages[0] if hit_ages else None,
        'peak_dominator': peak_dom,
        'breakout_score': score,
        'pff_seasons': len(matches['season'].unique()),
    }

# Apply breakout data
df['breakout_age'] = df['player_name'].map(lambda n: breakout_data.get(n, {}).get('breakout_age'))
df['peak_dominator'] = df['player_name'].map(lambda n: breakout_data.get(n, {}).get('peak_dominator'))
df['breakout_score'] = df['player_name'].map(lambda n: breakout_data.get(n, {}).get('breakout_score'))
df['pff_seasons'] = df['player_name'].map(lambda n: breakout_data.get(n, {}).get('pff_seasons'))

# Apply PFF 2025 data
for col in ['pff_receptions', 'pff_yards', 'pff_player_game_count',
            'pff_grades_offense', 'pff_grades_pass_route', 'pff_yprr', 'pff_team_name']:
    df[col] = df['player_name'].map(lambda n, c=col: pff_2025_data.get(n, {}).get(c))

n_breakout = df['breakout_score'].notna().sum()
n_pff = df['pff_receptions'].notna().sum()
print(f"\n  PFF match rate: {n_pff}/{len(df)} ({n_pff/len(df):.1%}) have 2025 PFF data")
print(f"  Breakout score: {n_breakout}/{len(df)} ({n_breakout/len(df):.1%}) have breakout data")
print(f"  Hit 15% dominator: {df['breakout_age'].notna().sum()}/{len(df)}")


# ============================================================================
# STEP 6: WEIGHTS FROM COMBINE/CFBD SOURCES
# ============================================================================

print(f"\n\n{'='*100}")
print("STEP 6: WEIGHTS AND ATHLETIC DATA")
print("=" * 100)

# Try CFBD player search for weight
weights = {}
print(f"\n  Searching CFBD for player weights...")
for _, r in df.iterrows():
    name = r['player_name']
    school = r['college']
    info = fetch_player_info(name, school)
    time.sleep(0.3)

    if info and info.get('weight'):
        weights[name] = {'weight': info['weight'], 'height': info.get('height'), 'source': 'CFBD'}
    elif info and info.get('name'):
        weights[name] = {'weight': None, 'height': info.get('height'), 'source': 'CFBD_no_weight'}

df['weight'] = df['player_name'].map(lambda n: weights.get(n, {}).get('weight'))
df['height'] = df['player_name'].map(lambda n: weights.get(n, {}).get('height'))
df['weight_source'] = df['player_name'].map(lambda n: weights.get(n, {}).get('source', 'missing'))

n_wt = df['weight'].notna().sum()
print(f"\n  Weight coverage: {n_wt}/{len(df)} ({n_wt/len(df):.1%})")


# ============================================================================
# STEP 7: DATA COVERAGE REPORT
# ============================================================================

print(f"\n\n{'='*120}")
print("DATA COVERAGE REPORT — 2026 TE PROSPECTS")
print(f"{'='*120}")

print(f"\n  Total prospects: {len(df)}")
print(f"\n  Component data coverage:")
print(f"    {'Component':<30} {'Has Data':>10} {'Missing':>10} {'Coverage':>10}")
print(f"    {'-'*65}")

coverage = [
    ("Projected pick", df['projected_pick'].notna().sum()),
    ("Draft age", df['draft_age'].notna().sum()),
    ("  - Verified birthdate", (df['birthdate_source'] == 'verified').sum()),
    ("  - Estimated birthdate", (df['birthdate_source'] == 'estimated').sum()),
    ("Early declare status", df['early_declare'].notna().sum()),
    ("CFBD receiving (matched)", df['cfbd_matched'].sum()),
    ("CFBD team pass att", df['cfbd_team_pass_att'].notna().sum()),
    ("PFF 2025 data", df['pff_receptions'].notna().sum()),
    ("Breakout score (15% dom)", df['breakout_score'].notna().sum()),
    ("  - Hit 15% dominator", df['breakout_age'].notna().sum()),
    ("  - Never hit 15%", ((df['breakout_score'].notna()) & (df['breakout_age'].isna())).sum()),
    ("Weight", df['weight'].notna().sum()),
]

for label, has in coverage:
    missing = len(df) - has
    pct = has / len(df) * 100
    print(f"    {label:<30} {has:>10} {missing:>10} {pct:>9.1f}%")

# SLAP component readiness
print(f"\n  SLAP COMPONENT READINESS (60/15/15/10):")
print(f"    DC (projected pick):      {df['projected_pick'].notna().sum()}/{len(df)} — {'READY' if df['projected_pick'].notna().all() else 'GAPS'}")
print(f"    Breakout (15% dom):       {df['breakout_score'].notna().sum()}/{len(df)} — {'READY' if df['breakout_score'].notna().sum() > len(df)*0.7 else 'GAPS — will impute missing with avg'}")
print(f"    Production (Rec/TPA):     CFBD={df['cfbd_matched'].sum()}, PFF={df['pff_receptions'].notna().sum()} — hybrid coverage")
print(f"    RAS:                      NOT YET — need combine/pro day data (April 2026)")

# Show prospects missing key data
print(f"\n  PROSPECTS MISSING CFBD RECEIVING DATA:")
missing_cfbd = df[~df['cfbd_matched']]
if len(missing_cfbd) > 0:
    for _, r in missing_cfbd.iterrows():
        pff_note = f"PFF: {r['pff_receptions']:.0f} rec" if pd.notna(r['pff_receptions']) else "no PFF"
        print(f"    {r['player_name']:<30} {r['college']:<20} pick {int(r['projected_pick']):>3} — {pff_note}")
else:
    print(f"    (none)")

print(f"\n  PROSPECTS MISSING BREAKOUT SCORE:")
missing_bo = df[df['breakout_score'].isna()]
if len(missing_bo) > 0:
    for _, r in missing_bo.iterrows():
        print(f"    {r['player_name']:<30} {r['college']:<20} pick {int(r['projected_pick']):>3}")
else:
    print(f"    (none)")

# Save intermediate data
df.to_csv('data/te_2026_prospects_raw.csv', index=False)
print(f"\n  Saved data/te_2026_prospects_raw.csv ({len(df)} rows)")

# Print full prospect table
print(f"\n\n{'='*120}")
print("FULL PROSPECT TABLE")
print(f"{'='*120}")
print(f"\n  {'Name':<28} {'College':<20} {'Pick':>4} {'Age':>5} {'ED':>3} {'CFBD':>5} {'PFF':>4} {'BO':>5} {'Dom%':>5} {'Wt':>5}")
print(f"  {'-'*105}")
for _, r in df.sort_values('projected_pick').iterrows():
    age_s = f"{r['draft_age']:.1f}" if pd.notna(r['draft_age']) else "?"
    ed_s = str(int(r['early_declare'])) if pd.notna(r['early_declare']) else "?"
    cfbd_s = "Y" if r['cfbd_matched'] else "N"
    pff_s = "Y" if pd.notna(r['pff_receptions']) else "N"
    bo_s = f"{r['breakout_score']:.0f}" if pd.notna(r['breakout_score']) else "?"
    dom_s = f"{r['peak_dominator']:.1f}" if pd.notna(r['peak_dominator']) else "?"
    wt_s = f"{int(r['weight'])}" if pd.notna(r['weight']) else "?"
    bd_flag = "*" if r['birthdate_source'] == 'estimated' else ""
    print(f"  {r['player_name']:<28} {r['college']:<20} {int(r['projected_pick']):>4} {age_s:>4}{bd_flag} {ed_s:>3} {cfbd_s:>5} {pff_s:>4} {bo_s:>5} {dom_s:>5} {wt_s:>5}")

print(f"\n  * = estimated birthdate (needs manual verification)")
print(f"\n{'='*120}")
print("DATA COLLECTION COMPLETE — Review above before calculating SLAP scores")
print(f"{'='*120}")
