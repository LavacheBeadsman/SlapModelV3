"""
Build 2026 Prospect Cards
=========================
Creates comprehensive prospect card datasets for all 2026 WR, RB, and TE prospects.
Combines existing SLAP model data with CFBD college production stats.

Output:
  output/2026_prospect_cards.csv   — All 217 prospects, sorted by position then pick
  output/2026_wr_cards.csv         — 109 WR prospects
  output/2026_rb_cards.csv         — 56 RB prospects
  output/2026_te_cards.csv         — 52 TE prospects

Data sources:
  - SLAP model scores from output/slap_v5_2026_all.csv and output/slap_v5_wr_2026.csv
  - WR breakout data (per-season CFBD receiving) from data/wr_breakout_ages_2026.csv
  - RB/WR prospect data from data/prospects_final.csv
  - TE prospect data from data/te_2026_prospects_final.csv
  - CFBD API for supplementary stats (rushing, receptions, games played)
  - NO PFF data used anywhere
"""

import pandas as pd
import numpy as np
import json
import requests
import time
import os
import re

os.chdir('/home/user/SlapModelV3')

CFBD_API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
CFBD_HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}
CFBD_BASE = "https://api.collegefootballdata.com"

# School name mapping: our data → CFBD API name
# Most school names match directly; only exceptions listed here
SCHOOL_TO_CFBD = {
    'Mississippi': 'Ole Miss',
    'Miami (FL)': 'Miami',
    'Miami (OH)': 'Miami (OH)',
    'UConn': 'Connecticut',
    'Louisiana-Lafayette': 'Louisiana',
    'Florida International': 'FIU',
    # FCS/D2/D3/NAIA schools — CFBD may not have data
    'Virginia Union': None,
    'John Carroll': None,
    'West Alabama': None,
}


def normalize_name(name):
    """Normalize player name for fuzzy matching."""
    if pd.isna(name):
        return ''
    s = str(name).strip().lower()
    for k, v in {'é': 'e', 'è': 'e', 'ê': 'e', 'á': 'a', 'à': 'a',
                  'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace("'", '').replace('-', ' ').replace(',', '')
    s = re.sub(r'\s+(jr|sr|ii|iii|iv|v)\s*$', '', s)
    return ' '.join(s.split())


def get_cfbd_school(school_name):
    """Convert our school name to CFBD API name."""
    if school_name in SCHOOL_TO_CFBD:
        return SCHOOL_TO_CFBD[school_name]
    return school_name  # Most names match directly


def slap_tier(score):
    """Assign tier label based on SLAP display score."""
    if pd.isna(score):
        return ''
    if score >= 80:
        return 'Elite'
    elif score >= 60:
        return 'Strong'
    elif score >= 40:
        return 'Average'
    else:
        return 'Long Shot'


# ============================================================================
print("=" * 90)
print("BUILD 2026 PROSPECT CARDS")
print("=" * 90)


# ============================================================================
# STEP 1: Load all existing data
# ============================================================================
print("\n--- Step 1: Loading existing data files ---")

# SLAP model scores (all positions)
slap_2026 = pd.read_csv('output/slap_v5_2026_all.csv')
# WR-specific SLAP data (has teammate_score, early_declare_score, rush_yards)
slap_wr_2026 = pd.read_csv('output/slap_v5_wr_2026.csv')
# Prospect input files
prospects = pd.read_csv('data/prospects_final.csv')
te_prospects = pd.read_csv('data/te_2026_prospects_final.csv')
# WR breakout data (has per-season CFBD receiving stats in JSON)
wr_breakout = pd.read_csv('data/wr_breakout_ages_2026.csv')

print(f"  SLAP 2026: {len(slap_2026)} prospects (all positions)")
print(f"  WR SLAP:   {len(slap_wr_2026)} WR prospects")
print(f"  Prospects:  {len(prospects)} WR+RB prospects")
print(f"  TE input:   {len(te_prospects)} TE prospects")
print(f"  WR breakout: {len(wr_breakout)} WR breakout records")


# ============================================================================
# STEP 2: Extract WR final season stats from breakout JSON
# ============================================================================
print("\n--- Step 2: Extracting WR final season CFBD data from breakout JSON ---")

wr_final_stats = []
for _, row in wr_breakout.iterrows():
    entry = {'player_name': row['player_name'], 'college': row['college']}
    try:
        seasons = json.loads(row['seasons_data'])
        if seasons:
            # Take the latest season as the "final" season
            latest = max(seasons, key=lambda s: s['season'])
            entry['final_season'] = int(latest['season'])
            entry['wr_rec_yards_json'] = latest.get('player_rec_yards')
            entry['wr_rec_tds_json'] = latest.get('player_rec_tds')
            entry['wr_team_rec_yards_json'] = latest.get('team_rec_yards')
            entry['wr_team_rec_tds_json'] = latest.get('team_rec_tds')
    except (json.JSONDecodeError, TypeError):
        pass
    wr_final_stats.append(entry)

wr_json_df = pd.DataFrame(wr_final_stats)
n_with_data = wr_json_df['wr_rec_yards_json'].notna().sum()
print(f"  Extracted final season receiving from JSON: {n_with_data}/{len(wr_json_df)} WRs")


# ============================================================================
# STEP 3: Identify schools needing CFBD API calls
# ============================================================================
print("\n--- Step 3: Identifying schools for CFBD API ---")

# Build list of (school, season) pairs we need data for
# All 2026 prospects played their final season in 2025 (draft_year - 1)
DEFAULT_SEASON = 2025

# Collect all schools
wr_schools = prospects[prospects['position'] == 'WR']['school'].dropna().unique()
rb_schools = prospects[prospects['position'] == 'RB']['school'].dropna().unique()
te_schools = te_prospects['college'].dropna().unique()
all_schools = sorted(set(list(wr_schools) + list(rb_schools) + list(te_schools)))

# Filter to schools where CFBD might have data
cfbd_schools = []
skipped_schools = []
for s in all_schools:
    cfbd_name = get_cfbd_school(s)
    if cfbd_name is None:
        skipped_schools.append(s)
    else:
        cfbd_schools.append((s, cfbd_name))

print(f"  Total unique schools: {len(all_schools)}")
print(f"  Schools with CFBD mapping: {len(cfbd_schools)}")
if skipped_schools:
    print(f"  Schools without CFBD data: {', '.join(skipped_schools)}")


# ============================================================================
# STEP 4: Fetch CFBD data (receiving + rushing + games) per school
# ============================================================================
print("\n--- Step 4: Fetching CFBD API data ---")
print(f"  Fetching for {len(cfbd_schools)} schools...")

# Cache: cfbd_school → {player_norm → {stat_type: value}}
receiving_cache = {}  # school → [{player, REC, YDS, TD, LONG}]
rushing_cache = {}    # school → [{player, ATT, YDS, TD, LONG}]
team_stats_cache = {} # school → {games, team_rec_yards, team_rec_tds, team_pass_att}

api_calls = 0
api_errors = 0


def cfbd_get(url, params, max_retries=3):
    """Make CFBD API call with retry logic."""
    global api_calls, api_errors
    for attempt in range(max_retries):
        try:
            api_calls += 1
            resp = requests.get(url, headers=CFBD_HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                api_errors += 1
                return None
        except requests.exceptions.RequestException:
            api_errors += 1
            if attempt < max_retries - 1:
                time.sleep(1)
    return None


def parse_player_stats(data):
    """Parse CFBD /stats/player/season response into {player_norm: {stat: val}} dict."""
    players = {}
    if not data:
        return players
    for row in data:
        name = row.get('player', '')
        stat_type = row.get('statType', '')
        try:
            value = float(row.get('stat', 0))
        except (ValueError, TypeError):
            value = 0
        name_n = normalize_name(name)
        if name_n not in players:
            players[name_n] = {'player_raw': name}
        players[name_n][stat_type] = value
    return players


for i, (our_name, cfbd_name) in enumerate(cfbd_schools):
    if (i + 1) % 20 == 0 or i == 0:
        print(f"  [{i+1}/{len(cfbd_schools)}] {cfbd_name}...")

    # Receiving stats
    time.sleep(0.25)
    recv_data = cfbd_get(
        f"{CFBD_BASE}/stats/player/season",
        {'year': DEFAULT_SEASON, 'team': cfbd_name, 'category': 'receiving'})
    receiving_cache[our_name] = parse_player_stats(recv_data)

    # Rushing stats
    time.sleep(0.25)
    rush_data = cfbd_get(
        f"{CFBD_BASE}/stats/player/season",
        {'year': DEFAULT_SEASON, 'team': cfbd_name, 'category': 'rushing'})
    rushing_cache[our_name] = parse_player_stats(rush_data)

    # Team stats (for team_rec_yards, team_rec_tds, team_pass_att)
    time.sleep(0.25)
    team_data = cfbd_get(
        f"{CFBD_BASE}/stats/season",
        {'year': DEFAULT_SEASON, 'team': cfbd_name})
    team_info = {}
    if team_data:
        for stat in team_data:
            sn = stat.get('statName', '')
            sv = stat.get('statValue', 0)
            try:
                sv = float(sv)
            except (ValueError, TypeError):
                sv = 0
            if sn == 'netPassingYards':
                team_info['team_rec_yards_api'] = sv
            elif sn == 'passingTDs':
                team_info['team_rec_tds_api'] = sv
            elif sn == 'passAttempts':
                team_info['team_pass_att_api'] = sv
            elif sn == 'totalGames':
                team_info['team_games'] = int(sv)
            elif sn == 'games':
                team_info['team_games'] = int(sv)
    team_stats_cache[our_name] = team_info

    # Game count (if not in team stats)
    if 'team_games' not in team_stats_cache[our_name]:
        time.sleep(0.25)
        games_data = cfbd_get(
            f"{CFBD_BASE}/games",
            {'year': DEFAULT_SEASON, 'team': cfbd_name, 'seasonType': 'both'})
        if games_data:
            team_stats_cache[our_name]['team_games'] = len(games_data)

print(f"\n  API calls made: {api_calls}")
print(f"  API errors: {api_errors}")
print(f"  Schools with receiving data: {sum(1 for v in receiving_cache.values() if v)}")
print(f"  Schools with rushing data: {sum(1 for v in rushing_cache.values() if v)}")
print(f"  Schools with team stats: {sum(1 for v in team_stats_cache.values() if v)}")


# ============================================================================
# STEP 5: Match each prospect to CFBD data
# ============================================================================
print("\n--- Step 5: Matching prospects to CFBD data ---")


def find_player_in_cache(player_name, school, cache):
    """Find a player in the CFBD cache by normalized name matching."""
    if school not in cache or not cache[school]:
        return {}
    school_players = cache[school]
    target = normalize_name(player_name)

    # Exact match
    if target in school_players:
        return school_players[target]

    # Try last-name match
    target_parts = target.split()
    if len(target_parts) >= 2:
        target_last = target_parts[-1]
        target_first = target_parts[0]
        for pname, pstats in school_players.items():
            parts = pname.split()
            if len(parts) >= 2:
                if parts[-1] == target_last and parts[0][0] == target_first[0]:
                    return pstats

    return {}


# ============================================================================
# STEP 6: Build the prospect card rows
# ============================================================================
print("\n--- Step 6: Building prospect card rows ---")

cards = []

# ----- WR PROSPECTS -----
print("  Building WR cards...")
wr_prospects_df = prospects[prospects['position'] == 'WR'].copy()

for _, p in wr_prospects_df.iterrows():
    name = p['player_name']
    school = p['school']

    # SLAP model data
    slap_row = slap_wr_2026[slap_wr_2026['player_name'] == name]
    slap_all_row = slap_2026[(slap_2026['player_name'] == name) & (slap_2026['position'] == 'WR')]

    slap_score = slap_row['slap_display_score'].values[0] if len(slap_row) > 0 else np.nan
    model_score = slap_row['slap_model_score'].values[0] if len(slap_row) > 0 else np.nan
    dc = slap_all_row['dc_score'].values[0] if len(slap_all_row) > 0 else np.nan
    pick = int(p['projected_pick']) if pd.notna(p['projected_pick']) else np.nan
    breakout_age = slap_row['breakout_age'].values[0] if len(slap_row) > 0 else np.nan
    peak_dom = slap_row['peak_dominator'].values[0] if len(slap_row) > 0 else np.nan
    tm_score = slap_row['teammate_score'].values[0] if len(slap_row) > 0 else np.nan
    ed_score = slap_row['early_declare_score'].values[0] if len(slap_row) > 0 else np.nan
    rush_yds_slap = slap_row['rush_yards'].values[0] if len(slap_row) > 0 else np.nan

    # WR receiving from breakout JSON (CFBD sourced, final season)
    json_row = wr_json_df[wr_json_df['player_name'] == name]
    rec_yards = json_row['wr_rec_yards_json'].values[0] if len(json_row) > 0 else np.nan
    rec_tds = json_row['wr_rec_tds_json'].values[0] if len(json_row) > 0 else np.nan
    team_rec_yards = json_row['wr_team_rec_yards_json'].values[0] if len(json_row) > 0 else np.nan
    team_rec_tds = json_row['wr_team_rec_tds_json'].values[0] if len(json_row) > 0 else np.nan
    team_pass_att = p['team_pass_attempts'] if pd.notna(p.get('team_pass_attempts')) else np.nan

    # CFBD API data (supplementary)
    recv_stats = find_player_in_cache(name, school, receiving_cache)
    rush_stats = find_player_in_cache(name, school, rushing_cache)
    team_info = team_stats_cache.get(school, {})

    receptions = recv_stats.get('REC', np.nan)
    # Use API rec_yards/rec_tds as fallback if JSON is missing
    if pd.isna(rec_yards) and 'YDS' in recv_stats:
        rec_yards = recv_stats['YDS']
    if pd.isna(rec_tds) and 'TD' in recv_stats:
        rec_tds = recv_stats['TD']

    rush_attempts = rush_stats.get('CAR', rush_stats.get('ATT', np.nan))
    rush_yards = rush_stats.get('YDS', np.nan)
    rush_tds = rush_stats.get('TD', np.nan)

    # Use team stats from API if JSON is missing
    if pd.isna(team_rec_yards):
        team_rec_yards = team_info.get('team_rec_yards_api', np.nan)
    if pd.isna(team_rec_tds):
        team_rec_tds = team_info.get('team_rec_tds_api', np.nan)
    if pd.isna(team_pass_att):
        team_pass_att = team_info.get('team_pass_att_api', np.nan)

    games_played = team_info.get('team_games', np.nan)

    # Total touchdowns
    total_tds = np.nan
    if pd.notna(rec_tds) or pd.notna(rush_tds):
        total_tds = (rec_tds if pd.notna(rec_tds) else 0) + (rush_tds if pd.notna(rush_tds) else 0)

    # Market share metrics
    rec_yards_share = (rec_yards / team_rec_yards * 100) if pd.notna(rec_yards) and pd.notna(team_rec_yards) and team_rec_yards > 0 else np.nan
    rec_td_share = (rec_tds / team_rec_tds * 100) if pd.notna(rec_tds) and pd.notna(team_rec_tds) and team_rec_tds > 0 else np.nan
    dominator = ((rec_yards_share + rec_td_share) / 2) if pd.notna(rec_yards_share) and pd.notna(rec_td_share) else np.nan
    rec_per_tpa = (rec_yards / team_pass_att) if pd.notna(rec_yards) and pd.notna(team_pass_att) and team_pass_att > 0 else np.nan

    cards.append({
        # Basic info
        'player_name': name,
        'college': school,
        'position': 'WR',
        'height': np.nan,  # Not available for WRs
        'weight': p['weight'] if pd.notna(p.get('weight')) else np.nan,
        # Combine data
        'forty_yard_dash': np.nan,
        'bench_press': np.nan,
        'vertical_jump': np.nan,
        'broad_jump': np.nan,
        'three_cone': np.nan,
        'shuttle': np.nan,
        'RAS_score': np.nan,
        'speed_score': np.nan,
        # College production (final season, CFBD only)
        'rush_attempts': rush_attempts,
        'rush_yards': rush_yards,
        'rush_tds': rush_tds,
        'receptions': receptions,
        'rec_yards': rec_yards,
        'rec_tds': rec_tds,
        'total_touchdowns': total_tds,
        'games_played': games_played,
        # Receiving market share
        'team_pass_attempts': team_pass_att,
        'team_rec_yards': team_rec_yards,
        'team_rec_tds': team_rec_tds,
        'rec_yards_share': round(rec_yards_share, 1) if pd.notna(rec_yards_share) else np.nan,
        'rec_td_share': round(rec_td_share, 1) if pd.notna(rec_td_share) else np.nan,
        'dominator_rating': round(dominator, 1) if pd.notna(dominator) else np.nan,
        'rec_yards_per_team_pass_att': round(rec_per_tpa, 2) if pd.notna(rec_per_tpa) else np.nan,
        # SLAP model
        'projected_pick': pick,
        'slap_v5_score': round(slap_score, 1) if pd.notna(slap_score) else np.nan,
        'dc_score': round(dc, 1) if pd.notna(dc) else np.nan,
        'delta_vs_dc': round(slap_score - dc, 1) if pd.notna(slap_score) and pd.notna(dc) else np.nan,
        'breakout_age': breakout_age if pd.notna(breakout_age) else np.nan,
        'peak_dominator': round(peak_dom, 1) if pd.notna(peak_dom) else np.nan,
        'teammate_score': tm_score if pd.notna(tm_score) else np.nan,
        'early_declare': int(ed_score / 100) if pd.notna(ed_score) and ed_score > 0 else 0,
        'slap_tier': slap_tier(slap_score),
    })


# ----- RB PROSPECTS -----
print("  Building RB cards...")
rb_prospects_df = prospects[prospects['position'] == 'RB'].copy()

for _, p in rb_prospects_df.iterrows():
    name = p['player_name']
    school = p['school']

    # SLAP model data
    slap_row = slap_2026[(slap_2026['player_name'] == name) & (slap_2026['position'] == 'RB')]

    slap_score = slap_row['slap_display_score'].values[0] if len(slap_row) > 0 else np.nan
    dc = slap_row['dc_score'].values[0] if len(slap_row) > 0 else np.nan
    pick = int(p['projected_pick']) if pd.notna(p['projected_pick']) else np.nan
    prod_score = slap_row['production_score'].values[0] if len(slap_row) > 0 else np.nan
    speed = slap_row['speed_score'].values[0] if len(slap_row) > 0 else np.nan

    # Receiving data from prospects_final.csv (CFBD sourced)
    rec_yards_file = p['rec_yards'] if pd.notna(p.get('rec_yards')) else np.nan
    team_pass_att_file = p['team_pass_attempts'] if pd.notna(p.get('team_pass_attempts')) else np.nan

    # CFBD API data
    recv_stats = find_player_in_cache(name, school, receiving_cache)
    rush_stats = find_player_in_cache(name, school, rushing_cache)
    team_info = team_stats_cache.get(school, {})

    receptions = recv_stats.get('REC', np.nan)
    rec_yards = recv_stats.get('YDS', rec_yards_file)
    rec_tds = recv_stats.get('TD', np.nan)

    rush_attempts = rush_stats.get('CAR', rush_stats.get('ATT', np.nan))
    rush_yards = rush_stats.get('YDS', np.nan)
    rush_tds = rush_stats.get('TD', np.nan)

    team_rec_yards = team_info.get('team_rec_yards_api', np.nan)
    team_rec_tds = team_info.get('team_rec_tds_api', np.nan)
    team_pass_att = team_pass_att_file if pd.notna(team_pass_att_file) else team_info.get('team_pass_att_api', np.nan)

    games_played = team_info.get('team_games', np.nan)

    # Use file rec_yards if API didn't return it
    if pd.isna(rec_yards):
        rec_yards = rec_yards_file

    total_tds = np.nan
    if pd.notna(rec_tds) or pd.notna(rush_tds):
        total_tds = (rec_tds if pd.notna(rec_tds) else 0) + (rush_tds if pd.notna(rush_tds) else 0)

    # Market share metrics
    rec_yards_share = (rec_yards / team_rec_yards * 100) if pd.notna(rec_yards) and pd.notna(team_rec_yards) and team_rec_yards > 0 else np.nan
    rec_td_share = (rec_tds / team_rec_tds * 100) if pd.notna(rec_tds) and pd.notna(team_rec_tds) and team_rec_tds > 0 else np.nan
    dominator = ((rec_yards_share + rec_td_share) / 2) if pd.notna(rec_yards_share) and pd.notna(rec_td_share) else np.nan
    rec_per_tpa = (rec_yards / team_pass_att) if pd.notna(rec_yards) and pd.notna(team_pass_att) and team_pass_att > 0 else np.nan

    cards.append({
        'player_name': name,
        'college': school,
        'position': 'RB',
        'height': np.nan,
        'weight': p['weight'] if pd.notna(p.get('weight')) else np.nan,
        'forty_yard_dash': np.nan,
        'bench_press': np.nan,
        'vertical_jump': np.nan,
        'broad_jump': np.nan,
        'three_cone': np.nan,
        'shuttle': np.nan,
        'RAS_score': np.nan,
        'speed_score': round(speed, 1) if pd.notna(speed) else np.nan,
        'rush_attempts': rush_attempts,
        'rush_yards': rush_yards,
        'rush_tds': rush_tds,
        'receptions': receptions,
        'rec_yards': rec_yards,
        'rec_tds': rec_tds,
        'total_touchdowns': total_tds,
        'games_played': games_played,
        'team_pass_attempts': team_pass_att,
        'team_rec_yards': team_rec_yards,
        'team_rec_tds': team_rec_tds,
        'rec_yards_share': round(rec_yards_share, 1) if pd.notna(rec_yards_share) else np.nan,
        'rec_td_share': round(rec_td_share, 1) if pd.notna(rec_td_share) else np.nan,
        'dominator_rating': round(dominator, 1) if pd.notna(dominator) else np.nan,
        'rec_yards_per_team_pass_att': round(rec_per_tpa, 2) if pd.notna(rec_per_tpa) else np.nan,
        'projected_pick': pick,
        'slap_v5_score': round(slap_score, 1) if pd.notna(slap_score) else np.nan,
        'dc_score': round(dc, 1) if pd.notna(dc) else np.nan,
        'delta_vs_dc': round(slap_score - dc, 1) if pd.notna(slap_score) and pd.notna(dc) else np.nan,
        'breakout_age': np.nan,
        'peak_dominator': np.nan,
        'teammate_score': np.nan,
        'early_declare': np.nan,
        'slap_tier': slap_tier(slap_score),
    })


# ----- TE PROSPECTS -----
print("  Building TE cards...")

for _, t in te_prospects.iterrows():
    name = t['player_name']
    school = t['college']

    # SLAP model data
    slap_row = slap_2026[(slap_2026['player_name'] == name) & (slap_2026['position'] == 'TE')]

    slap_score = slap_row['slap_display_score'].values[0] if len(slap_row) > 0 else np.nan
    dc = slap_row['dc_score'].values[0] if len(slap_row) > 0 else np.nan
    pick = int(t['projected_pick']) if pd.notna(t['projected_pick']) else np.nan
    breakout_age = t['breakout_age'] if pd.notna(t.get('breakout_age')) else np.nan
    peak_dom = t['peak_dominator'] if pd.notna(t.get('peak_dominator')) else np.nan
    ras = t['ras_score'] if pd.notna(t.get('ras_score')) else np.nan

    # TE data from te_2026_prospects_final.csv (CFBD sourced)
    cfbd_receptions = t['cfbd_receptions'] if pd.notna(t.get('cfbd_receptions')) else np.nan
    cfbd_rec_yards = t['cfbd_rec_yards'] if pd.notna(t.get('cfbd_rec_yards')) else np.nan
    cfbd_team_pass_att = t['cfbd_team_pass_att'] if pd.notna(t.get('cfbd_team_pass_att')) else np.nan
    cfbd_rush_yards = t['cfbd_rush_yards'] if pd.notna(t.get('cfbd_rush_yards')) else np.nan

    # CFBD API data (supplementary)
    recv_stats = find_player_in_cache(name, school, receiving_cache)
    rush_stats = find_player_in_cache(name, school, rushing_cache)
    team_info = team_stats_cache.get(school, {})

    receptions = cfbd_receptions if pd.notna(cfbd_receptions) else recv_stats.get('REC', np.nan)
    rec_yards = cfbd_rec_yards if pd.notna(cfbd_rec_yards) else recv_stats.get('YDS', np.nan)
    rec_tds = recv_stats.get('TD', np.nan)

    rush_attempts = rush_stats.get('CAR', rush_stats.get('ATT', np.nan))
    rush_yards = cfbd_rush_yards if pd.notna(cfbd_rush_yards) else rush_stats.get('YDS', np.nan)
    rush_tds = rush_stats.get('TD', np.nan)

    team_pass_att = cfbd_team_pass_att if pd.notna(cfbd_team_pass_att) else team_info.get('team_pass_att_api', np.nan)
    team_rec_yards = team_info.get('team_rec_yards_api', np.nan)
    team_rec_tds = team_info.get('team_rec_tds_api', np.nan)

    games_played = team_info.get('team_games', np.nan)

    total_tds = np.nan
    if pd.notna(rec_tds) or pd.notna(rush_tds):
        total_tds = (rec_tds if pd.notna(rec_tds) else 0) + (rush_tds if pd.notna(rush_tds) else 0)

    # Height: convert from inches to display format
    height = np.nan
    if pd.notna(t.get('height')):
        h = float(t['height'])
        feet = int(h // 12)
        inches = int(h % 12)
        height = f"{feet}'{inches}\""

    # Market share metrics
    rec_yards_share = (rec_yards / team_rec_yards * 100) if pd.notna(rec_yards) and pd.notna(team_rec_yards) and team_rec_yards > 0 else np.nan
    rec_td_share = (rec_tds / team_rec_tds * 100) if pd.notna(rec_tds) and pd.notna(team_rec_tds) and team_rec_tds > 0 else np.nan
    dominator = ((rec_yards_share + rec_td_share) / 2) if pd.notna(rec_yards_share) and pd.notna(rec_td_share) else np.nan
    rec_per_tpa = (rec_yards / team_pass_att) if pd.notna(rec_yards) and pd.notna(team_pass_att) and team_pass_att > 0 else np.nan

    cards.append({
        'player_name': name,
        'college': school,
        'position': 'TE',
        'height': height,
        'weight': t['weight'] if pd.notna(t.get('weight')) else np.nan,
        'forty_yard_dash': np.nan,
        'bench_press': np.nan,
        'vertical_jump': np.nan,
        'broad_jump': np.nan,
        'three_cone': np.nan,
        'shuttle': np.nan,
        'RAS_score': round(ras, 1) if pd.notna(ras) else np.nan,
        'speed_score': np.nan,
        'rush_attempts': rush_attempts,
        'rush_yards': rush_yards,
        'rush_tds': rush_tds,
        'receptions': receptions,
        'rec_yards': rec_yards,
        'rec_tds': rec_tds,
        'total_touchdowns': total_tds,
        'games_played': games_played,
        'team_pass_attempts': team_pass_att,
        'team_rec_yards': team_rec_yards,
        'team_rec_tds': team_rec_tds,
        'rec_yards_share': round(rec_yards_share, 1) if pd.notna(rec_yards_share) else np.nan,
        'rec_td_share': round(rec_td_share, 1) if pd.notna(rec_td_share) else np.nan,
        'dominator_rating': round(dominator, 1) if pd.notna(dominator) else np.nan,
        'rec_yards_per_team_pass_att': round(rec_per_tpa, 2) if pd.notna(rec_per_tpa) else np.nan,
        'projected_pick': pick,
        'slap_v5_score': round(slap_score, 1) if pd.notna(slap_score) else np.nan,
        'dc_score': round(dc, 1) if pd.notna(dc) else np.nan,
        'delta_vs_dc': round(slap_score - dc, 1) if pd.notna(slap_score) and pd.notna(dc) else np.nan,
        'breakout_age': breakout_age if pd.notna(breakout_age) else np.nan,
        'peak_dominator': round(peak_dom, 1) if pd.notna(peak_dom) else np.nan,
        'teammate_score': np.nan,
        'early_declare': np.nan,
        'slap_tier': slap_tier(slap_score),
    })


# ============================================================================
# STEP 7: Build DataFrame and clean up
# ============================================================================
print("\n--- Step 7: Building output DataFrame ---")

df = pd.DataFrame(cards)

# Convert numeric columns — ensure clean types
numeric_cols = [
    'weight', 'forty_yard_dash', 'bench_press', 'vertical_jump', 'broad_jump',
    'three_cone', 'shuttle', 'RAS_score', 'speed_score',
    'rush_attempts', 'rush_yards', 'rush_tds', 'receptions', 'rec_yards', 'rec_tds',
    'total_touchdowns', 'games_played',
    'team_pass_attempts', 'team_rec_yards', 'team_rec_tds',
    'rec_yards_share', 'rec_td_share', 'dominator_rating', 'rec_yards_per_team_pass_att',
    'projected_pick', 'slap_v5_score', 'dc_score', 'delta_vs_dc',
    'breakout_age', 'peak_dominator', 'teammate_score', 'early_declare',
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Sort: position order (RB, TE, WR) → within position by projected_pick
pos_order = {'RB': 0, 'TE': 1, 'WR': 2}
df['pos_sort'] = df['position'].map(pos_order)
df = df.sort_values(['pos_sort', 'projected_pick'], ascending=[True, True]).reset_index(drop=True)
df = df.drop(columns=['pos_sort'])

# Column order matching user specification
col_order = [
    # Basic info
    'player_name', 'college', 'position', 'height', 'weight',
    # Combine data
    'forty_yard_dash', 'bench_press', 'vertical_jump', 'broad_jump',
    'three_cone', 'shuttle', 'RAS_score', 'speed_score',
    # College production
    'rush_attempts', 'rush_yards', 'rush_tds',
    'receptions', 'rec_yards', 'rec_tds', 'total_touchdowns', 'games_played',
    # Receiving market share
    'team_pass_attempts', 'team_rec_yards', 'team_rec_tds',
    'rec_yards_share', 'rec_td_share', 'dominator_rating', 'rec_yards_per_team_pass_att',
    # SLAP model
    'projected_pick', 'slap_v5_score', 'dc_score', 'delta_vs_dc',
    'breakout_age', 'peak_dominator', 'teammate_score', 'early_declare',
    'slap_tier',
]
df = df[col_order]


# ============================================================================
# STEP 8: Save output files
# ============================================================================
print("\n--- Step 8: Saving output files ---")

# All prospects
df.to_csv('output/2026_prospect_cards.csv', index=False)
print(f"  output/2026_prospect_cards.csv: {len(df)} prospects")

# Position-specific files
for pos in ['WR', 'RB', 'TE']:
    pos_df = df[df['position'] == pos].copy()
    pos_df = pos_df.sort_values('projected_pick').reset_index(drop=True)

    # Drop position-irrelevant columns
    if pos == 'WR':
        drop_cols = ['speed_score']
    elif pos == 'RB':
        drop_cols = ['RAS_score', 'breakout_age', 'peak_dominator', 'teammate_score', 'early_declare']
    else:  # TE
        drop_cols = ['speed_score', 'teammate_score', 'early_declare']

    pos_df = pos_df.drop(columns=[c for c in drop_cols if c in pos_df.columns])
    fname = f'output/2026_{pos.lower()}_cards.csv'
    pos_df.to_csv(fname, index=False)
    print(f"  {fname}: {len(pos_df)} {pos} prospects")


# ============================================================================
# STEP 9: Data coverage summary
# ============================================================================
print(f"\n{'='*90}")
print("DATA COVERAGE SUMMARY")
print(f"{'='*90}")

for pos in ['WR', 'RB', 'TE']:
    pos_df = df[df['position'] == pos]
    n = len(pos_df)
    print(f"\n  {pos} ({n} prospects):")

    # Key stats coverage
    stats_to_check = {
        'weight': 'Weight',
        'rush_yards': 'Rush Yards',
        'rush_attempts': 'Rush Attempts',
        'receptions': 'Receptions',
        'rec_yards': 'Rec Yards',
        'rec_tds': 'Rec TDs',
        'games_played': 'Games Played',
        'team_rec_yards': 'Team Rec Yards',
        'rec_yards_share': 'Rec Yards Share',
        'dominator_rating': 'Dominator Rating',
        'slap_v5_score': 'SLAP Score',
    }
    for col, label in stats_to_check.items():
        if col in pos_df.columns:
            have = pos_df[col].notna().sum()
            print(f"    {label:<20}: {have}/{n}")

print(f"\n{'='*90}")
print("PROSPECT CARDS BUILD COMPLETE")
print(f"{'='*90}")
print(f"\nFiles saved:")
print(f"  output/2026_prospect_cards.csv  ({len(df)} prospects, all positions)")
print(f"  output/2026_wr_cards.csv        ({len(df[df['position']=='WR'])} WR)")
print(f"  output/2026_rb_cards.csv        ({len(df[df['position']=='RB'])} RB)")
print(f"  output/2026_te_cards.csv        ({len(df[df['position']=='TE'])} TE)")
print(f"\nNote: Combine data fields are blank (pre-combine). Update after NFL Combine.")
