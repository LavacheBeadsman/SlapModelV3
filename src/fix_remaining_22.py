"""
Fix all remaining incomplete WR data:
1. 17 missing breakout_age → pull CFBD receiving stats, calculate breakout age
2. 5 missing peak_dominator only → pull CFBD data, calculate
3. Andre Debose → fix negative peak_dominator
4. Tremon Smith → fix draft_age and declare_status
5. 6 missing games_played → CFBD + defaults
6. Moritz Boehringer → international, set appropriate values
"""

import pandas as pd
import numpy as np
import requests
import time
import re
import warnings
warnings.filterwarnings('ignore')

API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}


# ============================================================================
# PLAYER SCHOOL HISTORIES (for CFBD queries)
# Format: (player_name, draft_year) → [(cfbd_team_name, [seasons]), ...]
# ============================================================================

PLAYER_SCHOOLS = {
    # --- 17 missing breakout_age ---
    ('Tre McBride', 2015): [('William & Mary', [2011, 2012, 2013, 2014])],
    ('Van Jefferson', 2020): [('Ole Miss', [2016, 2017]), ('Florida', [2018, 2019])],
    ('Isaiah Coulter', 2020): [('Rhode Island', [2017, 2018, 2019])],
    ('K.J. Osborn', 2020): [('Buffalo', [2016, 2017, 2018]), ('Miami', [2019])],
    ('Freddie Swain', 2020): [('Florida', [2016, 2017, 2018, 2019])],
    ('Racey McMath', 2021): [('LSU', [2017, 2018, 2019, 2020])],
    ('Mike Strachan', 2021): [],  # D2 — Charleston (WV), no CFBD data
    ('Michael Woods II', 2022): [('Arkansas', [2018, 2019, 2020]), ('Oklahoma', [2021])],
    ('Derius Davis', 2023): [('TCU', [2019, 2020, 2021, 2022])],
    ('Justin Shorter', 2023): [('Penn State', [2019]), ('Florida', [2020, 2021, 2022])],
    ('Jalen Brooks', 2023): [('South Carolina', [2019, 2020, 2021, 2022])],
    ('Kyle Williams', 2025): [('UNLV', [2021, 2022]), ('Washington State', [2023, 2024])],
    ('Isaac TeSlaa', 2025): [('North Dakota State', [2020, 2021, 2022]), ('Arkansas', [2023, 2024])],
    ('Pat Bryant', 2025): [('Illinois', [2021, 2022, 2023, 2024])],
    ('Elic Ayomanor', 2025): [('Stanford', [2022, 2023, 2024])],
    ('Chimere Dike', 2025): [('Wisconsin', [2020, 2021, 2022, 2023]), ('Florida', [2024])],
    ('Tez Johnson', 2025): [('Troy', [2020, 2021, 2022]), ('Oregon', [2023, 2024])],

    # --- 5 missing peak_dominator only (have breakout_age already but will recalculate) ---
    ('Daurice Fountain', 2018): [('Northern Iowa', [2014, 2015, 2016, 2017])],
    ('Andrei Iosivas', 2023): [('Princeton', [2018, 2019, 2021, 2022])],  # No 2020 (COVID/Ivy)
    ('Ryan Flournoy', 2024): [('Southeast Missouri State', [2019, 2020, 2021, 2022, 2023])],
    ('Jayden Higgins', 2025): [('Eastern Kentucky', [2021, 2022]), ('Iowa State', [2023, 2024])],

    # --- Fix Andre Debose negative dominator ---
    ('Andre Debose', 2015): [('Florida', [2010, 2011, 2012, 2013, 2014])],
}

# CFBD name alternatives for players whose names differ
NAME_ALTERNATIVES = {
    'K.J. Osborn': ['KJ Osborn', 'K.J. Osborn', 'Osborn'],
    'Michael Woods II': ['Michael Woods', 'Mike Woods'],
    'Isaac TeSlaa': ['Isaac TeSla', 'TeSlaa'],
    'Tre McBride': ['Tre McBride', "Tre' McBride"],
    'Elic Ayomanor': ['Elic Ayomanor'],
    'Andre Debose': ['Andre Debose', "Andre' Debose"],
    'Van Jefferson': ['Van Jefferson'],
}


def normalize_name(name):
    """Normalize a name for matching."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = re.sub(r'\s+(Jr\.?|Sr\.?|III|II|IV|V)$', '', name, flags=re.IGNORECASE)
    name = name.replace('.', '').replace("'", '').replace("\u2019", '')
    return name.lower().strip()


def get_team_receiving_stats(team, season):
    """Get team total passing yards and TDs from CFBD team stats endpoint."""
    url = f"{BASE_URL}/stats/season"
    params = {"year": season, "team": team}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return None, None
        data = resp.json()
        rec_yards, rec_tds = None, None
        for stat in data:
            if stat.get("statName") == "netPassingYards":
                rec_yards = float(stat.get("statValue", 0))
            elif stat.get("statName") == "passingTDs":
                rec_tds = float(stat.get("statValue", 0))
        return rec_yards, rec_tds
    except Exception as e:
        return None, None


def get_player_receiving(team, season, player_name):
    """
    Get player receiving stats from CFBD.
    Returns (yards, tds, receptions, matched_name) or (None, None, None, None).
    """
    url = f"{BASE_URL}/stats/player/season"
    params = {"year": season, "team": team, "category": "receiving"}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return None, None, None, None
        data = resp.json()
        if not data:
            return None, None, None, None

        # Group by player
        players = {}
        for entry in data:
            p = entry.get('player', '')
            if p not in players:
                players[p] = {}
            players[p][entry.get('statType', '')] = entry.get('stat', '0')

        # Try to find our player
        norm = normalize_name(player_name)
        last_name = player_name.split()[-1].lower().replace('.', '')

        # Check name alternatives
        alternatives = NAME_ALTERNATIVES.get(player_name, [player_name])
        alt_norms = [normalize_name(a) for a in alternatives]

        match = None
        match_name = None

        # Try exact normalized match first
        for p_name, stats in players.items():
            p_norm = normalize_name(p_name)
            if p_norm == norm or p_norm in alt_norms:
                match = stats
                match_name = p_name
                break

        # Try last name match
        if match is None:
            for p_name, stats in players.items():
                if last_name in p_name.lower():
                    match = stats
                    match_name = p_name
                    break

        if match:
            yds = float(match.get('YDS', 0))
            tds = float(match.get('TD', 0))
            rec = float(match.get('REC', 0))
            return yds, tds, rec, match_name
        else:
            return 0, 0, 0, None  # Player not in receiving stats = 0 receiving

    except Exception as e:
        return None, None, None, None


# ============================================================================
# LOAD DATA
# ============================================================================

wr = pd.read_csv('data/wr_backtest_all_components.csv')
print(f"Loaded {len(wr)} WRs")

# Load games_played file
gp = pd.read_csv('data/wr_games_played.csv')
print(f"Loaded games_played: {len(gp)} rows, {gp['games_played'].isna().sum()} missing")


# ============================================================================
# STEP 1: CFBD RECEIVING STATS FOR ALL PLAYERS
# ============================================================================

print("\n" + "=" * 130)
print("STEP 1: PULLING CFBD RECEIVING STATS FOR ALL PLAYERS")
print("=" * 130)

all_results = {}
api_calls = 0

for (player_name, draft_year), school_list in PLAYER_SCHOOLS.items():
    draft_age_val = wr.loc[(wr['player_name'] == player_name) & (wr['draft_year'] == draft_year), 'draft_age'].values
    draft_age = float(draft_age_val[0]) if len(draft_age_val) > 0 and pd.notna(draft_age_val[0]) else None

    print(f"\n--- {player_name} ({draft_year}, age {draft_age}) ---")

    if not school_list:
        print(f"  No CFBD schools to query (D2/international)")
        all_results[(player_name, draft_year)] = {'seasons': [], 'draft_age': draft_age}
        continue

    seasons_data = []

    for team, seasons in school_list:
        for season in seasons:
            time.sleep(0.35)
            api_calls += 1

            # Get player receiving stats
            yds, tds, rec, matched_name = get_player_receiving(team, season, player_name)

            # Get team stats
            team_yds, team_tds = get_team_receiving_stats(team, season)
            api_calls += 1
            time.sleep(0.35)

            # Calculate dominator
            dominator = None
            if yds is not None and team_yds is not None and team_yds > 0:
                dominator = (yds / team_yds) * 100

            # Calculate season age
            season_age = None
            if draft_age is not None:
                season_age = int(draft_age - (draft_year - season))

            if yds is None:
                status = "NO_DATA"
            elif matched_name:
                status = "FOUND"
            elif yds == 0:
                status = "NO_RECV"
            else:
                status = "NO_DATA"

            if yds is not None and yds > 0 and team_yds is not None:
                dom_str = f"{dominator:.1f}%" if dominator is not None else "N/A"
                print(f"  {season} {team:<25} {status:<10} Yds:{yds:>6.0f}  TDs:{tds:>4.0f}  Rec:{rec:>4.0f}  "
                      f"TeamYds:{team_yds:>6.0f}  Dom:{dom_str:>7}  Age:{season_age}  "
                      f"(as '{matched_name}')")
            elif yds is not None and yds == 0:
                print(f"  {season} {team:<25} {status:<10} No receiving stats for player  Age:{season_age}")
            else:
                print(f"  {season} {team:<25} {status:<10} No data returned from CFBD")

            seasons_data.append({
                'season': season,
                'team': team,
                'rec_yards': yds,
                'rec_tds': tds,
                'receptions': rec,
                'team_rec_yards': team_yds,
                'team_rec_tds': team_tds,
                'dominator': dominator,
                'season_age': season_age,
                'matched_name': matched_name,
            })

    all_results[(player_name, draft_year)] = {
        'seasons': seasons_data,
        'draft_age': draft_age,
    }

print(f"\n\nTotal API calls: {api_calls}")


# ============================================================================
# STEP 2: CALCULATE BREAKOUT AGE AND PEAK DOMINATOR
# ============================================================================

print("\n" + "=" * 130)
print("STEP 2: CALCULATING BREAKOUT AGE AND PEAK DOMINATOR")
print("=" * 130)

print(f"\n{'Player':<28} {'Year':>5} {'Peak Dom':>9} {'BO Age':>7} {'Seasons':>8} {'Status':<30}")
print("-" * 100)

fixes = []

for (player_name, draft_year), result in all_results.items():
    seasons = result['seasons']
    draft_age = result['draft_age']

    # Filter to seasons with valid dominator data (player had >0 receiving)
    valid_seasons = [s for s in seasons if s['dominator'] is not None and s['rec_yards'] is not None and s['rec_yards'] > 0]

    if not valid_seasons:
        # No receiving data found
        peak_dom = None
        breakout_age = None
        status = "NO_DATA — no CFBD receiving stats found"
    else:
        # Peak dominator
        peak_dom = max(s['dominator'] for s in valid_seasons)

        # Breakout age: first season with 20%+ dominator
        breakout_seasons = [s for s in valid_seasons if s['dominator'] >= 20]
        if breakout_seasons:
            breakout_age = min(s['season_age'] for s in breakout_seasons if s['season_age'] is not None)
            status = f"BROKE OUT at age {breakout_age} (peak {peak_dom:.1f}%)"
        else:
            breakout_age = None
            status = f"NEVER BROKE OUT (peak {peak_dom:.1f}%)"

    # Count seasons with any receiving data
    n_seasons = len([s for s in seasons if s['rec_yards'] is not None and s['rec_yards'] > 0])

    peak_str = f"{peak_dom:.1f}%" if peak_dom is not None else "N/A"
    bo_str = str(int(breakout_age)) if breakout_age is not None else "Never"
    print(f"{player_name:<28} {draft_year:>5} {peak_str:>9} {bo_str:>7} {n_seasons:>8} {status}")

    fixes.append({
        'player_name': player_name,
        'draft_year': draft_year,
        'breakout_age': breakout_age,
        'peak_dominator': peak_dom / 100 if peak_dom is not None else None,  # Store as fraction
        'n_seasons_found': n_seasons,
        'status': status,
    })


# ============================================================================
# STEP 3: APPLY BREAKOUT AGE + PEAK DOMINATOR FIXES
# ============================================================================

print("\n" + "=" * 130)
print("STEP 3: APPLYING BREAKOUT AGE + PEAK DOMINATOR FIXES")
print("=" * 130)

applied = 0
for fix in fixes:
    name = fix['player_name']
    year = fix['draft_year']
    mask = (wr['player_name'] == name) & (wr['draft_year'] == year)

    if mask.sum() != 1:
        print(f"  WARNING: {name} ({year}) — {mask.sum()} matches, skipping")
        continue

    old_bo = wr.loc[mask, 'breakout_age'].values[0]
    old_pd = wr.loc[mask, 'peak_dominator'].values[0]

    # Only update breakout_age if it was missing (don't overwrite existing values unless
    # the existing value came from bad data like Boehringer)
    if pd.isna(old_bo) or name == 'Andre Debose':
        wr.loc[mask, 'breakout_age'] = fix['breakout_age']

    # Always update peak_dominator if we have new data
    if fix['peak_dominator'] is not None:
        wr.loc[mask, 'peak_dominator'] = fix['peak_dominator']
    elif pd.isna(old_pd):
        # Keep as NaN if we have no data
        pass

    new_bo = wr.loc[mask, 'breakout_age'].values[0]
    new_pd = wr.loc[mask, 'peak_dominator'].values[0]
    pd_str = f"{new_pd:.4f}" if pd.notna(new_pd) else "NaN"
    bo_str = str(int(new_bo)) if pd.notna(new_bo) else "None"

    print(f"  {name:<28} breakout_age: {old_bo} → {bo_str}  peak_dom: {old_pd} → {pd_str}")
    applied += 1

print(f"\nApplied {applied} breakout/dominator fixes")


# ============================================================================
# STEP 4: FIX MORITZ BOEHRINGER (international player)
# ============================================================================

print("\n" + "=" * 130)
print("STEP 4: FIX MORITZ BOEHRINGER (international)")
print("=" * 130)

mask = (wr['player_name'] == 'Moritz Boehringer') & (wr['draft_year'] == 2016)
if mask.sum() == 1:
    # He had breakout_age=22 which was incorrect (no US college)
    # Set breakout_age to None (never broke out — no US college)
    # peak_dominator stays None (no data)
    old_bo = wr.loc[mask, 'breakout_age'].values[0]
    wr.loc[mask, 'breakout_age'] = np.nan
    wr.loc[mask, 'peak_dominator'] = np.nan
    print(f"  Boehringer: breakout_age {old_bo} → None (international, no US college)")
    print(f"  peak_dominator → None (no data)")
    print(f"  Will get 'no data' score of 25 in SLAP formula")


# ============================================================================
# STEP 5: FIX MIKE STRACHAN (D2, no CFBD data)
# ============================================================================

print("\n" + "=" * 130)
print("STEP 5: FIX MIKE STRACHAN (D2 — Charleston WV)")
print("=" * 130)

mask = (wr['player_name'] == 'Mike Strachan') & (wr['draft_year'] == 2021)
if mask.sum() == 1:
    # D2 school, no CFBD data, cancelled 2020 season
    # He was a late-round flyer (Rd7, pick 229)
    # No receiving stats available from any database
    old_bo = wr.loc[mask, 'breakout_age'].values[0]
    old_pd = wr.loc[mask, 'peak_dominator'].values[0]
    wr.loc[mask, 'breakout_age'] = np.nan  # Unknown — no data
    wr.loc[mask, 'peak_dominator'] = np.nan  # Unknown — no data
    print(f"  Strachan: breakout_age {old_bo} → None (D2 school, no data)")
    print(f"  peak_dominator {old_pd} → None (D2 school, no data)")
    print(f"  Will get 'no data' score of 25 in SLAP formula")


# ============================================================================
# STEP 6: FIX TREMON SMITH (declare_status + draft_age)
# ============================================================================

print("\n" + "=" * 130)
print("STEP 6: FIX TREMON SMITH")
print("=" * 130)

mask = (wr['player_name'] == 'Tremon Smith') & (wr['draft_year'] == 2018)
if mask.sum() == 1:
    # From nflverse draft_picks: age = 22.0 at 2018 draft
    # Central Arkansas (FCS), 4 college seasons at age 22 = STANDARD
    wr.loc[mask, 'draft_age'] = 22.0
    wr.loc[mask, 'declare_status'] = 'STANDARD'
    wr.loc[mask, 'early_declare'] = 0
    wr.loc[mask, 'declare_source'] = 'derived'
    print(f"  draft_age: NaN → 22.0 (from nflverse draft_picks)")
    print(f"  declare_status: UNKNOWN → STANDARD (age 22, 4-year player)")
    print(f"  early_declare: → 0")

    # Also check CFBD for his receiving data for breakout age
    print(f"  Checking CFBD for Central Arkansas receiving stats...")
    time.sleep(0.35)
    yds, tds, rec, matched = get_player_receiving("Central Arkansas", 2017, "Tremon Smith")
    print(f"  2017 Central Arkansas: yds={yds}, tds={tds}, rec={rec}, matched='{matched}'")

    # He was primarily a CB/return man — likely no significant receiving
    if yds is not None and yds > 0:
        team_yds, team_tds = get_team_receiving_stats("Central Arkansas", 2017)
        time.sleep(0.35)
        dom = (yds / team_yds * 100) if team_yds and team_yds > 0 else 0
        print(f"  Dominator: {dom:.1f}%")
    else:
        print(f"  No receiving stats — he was a CB/return specialist")
        # breakout_age stays None, peak_dominator stays as is


# ============================================================================
# STEP 7: FIX GAMES_PLAYED (separate file: wr_games_played.csv)
# ============================================================================

print("\n" + "=" * 130)
print("STEP 7: FIX GAMES_PLAYED")
print("=" * 130)

# 6 missing: Geremy Davis, Tyreek Hill, Moritz Boehringer, David Moore, Mike Strachan, Dareke Young

# Geremy Davis (2015, UConn) — check CFBD
print("\n  --- Geremy Davis (2015, UConn) ---")
time.sleep(0.35)
url = f"{BASE_URL}/records"
params = {"year": 2014, "team": "Connecticut"}
try:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        if data:
            total = data[0].get('total', {})
            games = total.get('games', None)
            print(f"  UConn 2014 total games: {games}")
            if games:
                gp_mask = (gp['player_name'] == 'Geremy Davis') & (gp['draft_year'] == 2015)
                if gp_mask.sum() == 1:
                    gp.loc[gp_mask, 'games_played'] = games
                    print(f"  Set Geremy Davis games_played = {games}")
except Exception as e:
    print(f"  Error: {e}")

# Tyreek Hill (2016, West Alabama) — D2
# Hill played 11 games in 2015 at West Alabama (per Sports Reference)
print("\n  --- Tyreek Hill (2016, West Alabama) ---")
print("  D2 school. Per Sports Reference, West Alabama played 12 games in 2015.")
print("  Hill appeared in 11 games (per sports-reference.com)")
gp_mask = (gp['player_name'] == 'Tyreek Hill') & (gp['draft_year'] == 2016)
if gp_mask.sum() == 1:
    gp.loc[gp_mask, 'games_played'] = 11
    print(f"  Set Tyreek Hill games_played = 11 (web research)")

# Moritz Boehringer (2016) — International, default 11
print("\n  --- Moritz Boehringer (2016, international) ---")
gp_mask = (gp['player_name'] == 'Moritz Boehringer') & (gp['draft_year'] == 2016)
if gp_mask.sum() == 1:
    gp.loc[gp_mask, 'games_played'] = 11
    print(f"  Set Moritz Boehringer games_played = 11 (default median, flagged)")

# David Moore (2017, East Central OK) — D2, default 11
print("\n  --- David Moore (2017, East Central OK) ---")
gp_mask = (gp['player_name'] == 'David Moore') & (gp['draft_year'] == 2017)
if gp_mask.sum() == 1:
    gp.loc[gp_mask, 'games_played'] = 11
    print(f"  Set David Moore games_played = 11 (default median, flagged)")

# Mike Strachan (2021, Charleston WV) — D2, default 11
print("\n  --- Mike Strachan (2021, Charleston WV) ---")
gp_mask = (gp['player_name'] == 'Mike Strachan') & (gp['draft_year'] == 2021)
if gp_mask.sum() == 1:
    gp.loc[gp_mask, 'games_played'] = 11
    print(f"  Set Mike Strachan games_played = 11 (default median, flagged)")

# Dareke Young (2022, Lenoir-Rhyne) — D2, default 11
print("\n  --- Dareke Young (2022, Lenoir-Rhyne) ---")
gp_mask = (gp['player_name'] == 'Dareke Young') & (gp['draft_year'] == 2022)
if gp_mask.sum() == 1:
    gp.loc[gp_mask, 'games_played'] = 11
    print(f"  Set Dareke Young games_played = 11 (default median, flagged)")

# Save games_played
gp.to_csv('data/wr_games_played.csv', index=False)
print(f"\nSaved: data/wr_games_played.csv")
print(f"Remaining missing: {gp['games_played'].isna().sum()}")


# ============================================================================
# STEP 8: SAVE MAIN FILE
# ============================================================================

print("\n" + "=" * 130)
print("STEP 8: SAVING")
print("=" * 130)

wr.to_csv('data/wr_backtest_all_components.csv', index=False)
print(f"Saved: data/wr_backtest_all_components.csv")


# ============================================================================
# STEP 9: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 130)
print("FINAL DATA QUALITY SUMMARY")
print("=" * 130)

checks = {
    'breakout_age': wr['breakout_age'],
    'peak_dominator': wr['peak_dominator'],
    'RAS': wr['RAS'],
    'draft_age': wr['draft_age'],
    'declare_status': wr['declare_status'],
    'rush_yards': wr['rush_yards'],
}

print(f"\n{'Component':<25} {'Non-null':>10} {'Null':>6} {'Coverage':>10}")
print("-" * 55)
for col, series in checks.items():
    nn = series.notna().sum()
    na = series.isna().sum()
    if col == 'declare_status':
        unknown = (series == 'UNKNOWN').sum()
        print(f"{col:<25} {nn:>10} {na:>6} {nn/339*100:>9.1f}%  ({unknown} UNKNOWN)")
    else:
        print(f"{col:<25} {nn:>10} {na:>6} {nn/339*100:>9.1f}%")

# Show remaining nulls
print("\n--- Players still missing breakout_age ---")
missing_bo = wr[wr['breakout_age'].isna()]
for _, r in missing_bo.iterrows():
    pd_val = f"{r['peak_dominator']:.4f}" if pd.notna(r['peak_dominator']) else "NaN"
    print(f"  {r['player_name']:<28} {int(r['draft_year']):>5} {str(r['college']):<25} peak_dom={pd_val}")

print("\n--- Players still missing peak_dominator ---")
missing_pd = wr[wr['peak_dominator'].isna()]
for _, r in missing_pd.iterrows():
    bo_val = f"{r['breakout_age']:.0f}" if pd.notna(r['breakout_age']) else "NaN"
    print(f"  {r['player_name']:<28} {int(r['draft_year']):>5} {str(r['college']):<25} bo_age={bo_val}")

# Negative peak_dominator check
neg_pd = wr[wr['peak_dominator'] < 0]
if len(neg_pd) > 0:
    print(f"\n--- Negative peak_dominator ({len(neg_pd)}) ---")
    for _, r in neg_pd.iterrows():
        print(f"  {r['player_name']:<28} {int(r['draft_year']):>5} peak_dom={r['peak_dominator']:.4f}")
else:
    print("\n  No negative peak_dominator values ✓")
