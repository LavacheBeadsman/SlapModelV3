"""
Fetch RB rushing & receiving stats from CFBD API for backtest RBs (2015-2025).
Targets: 223 backtest RBs missing rush_yards, rush_attempts, rush_tds, rec_tds, games_played.

Fetches per player (final college season = draft_year - 1):
  - rush_yards, rush_attempts, rush_tds (from CFBD rushing category)
  - rec_tds (from CFBD receiving category)
  - games_played (from CFBD player/usage endpoint)

Output: data/cfbd_rb_rushing_stats.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import os

# ─── CFBD API Setup ───
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE")
HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}
BASE_URL = "https://api.collegefootballdata.com"

# ─── School Name Mappings (nflverse → CFBD) ───
SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State",
    "Penn St.": "Penn State",
    "Michigan St.": "Michigan State",
    "North Dakota St.": "North Dakota State",
    "Oklahoma St.": "Oklahoma State",
    "Miss. St.": "Mississippi State",
    "Mississippi St.": "Mississippi State",
    "San Diego St.": "San Diego State",
    "Florida St.": "Florida State",
    "Fresno St.": "Fresno State",
    "Boise St.": "Boise State",
    "Iowa St.": "Iowa State",
    "Arizona St.": "Arizona State",
    "Kansas St.": "Kansas State",
    "N.C. State": "NC State",
    "N.C.": "North Carolina",
    "North Carolina St.": "NC State",
    "North Carolina A&T": "North Carolina A&T",
    "Appalachian St.": "Appalachian State",
    "Colorado St.": "Colorado State",
    "Oregon St.": "Oregon State",
    "Washington St.": "Washington State",
    "S. Carolina": "South Carolina",
    "Miami (FL)": "Miami",
    "Central Mich.": "Central Michigan",
    "Western Mich.": "Western Michigan",
    "Eastern Mich.": "Eastern Michigan",
    "Eastern Michigan": "Eastern Michigan",
    "Northern Ill.": "Northern Illinois",
    "Southern Miss.": "Southern Mississippi",
    "Southern Miss": "Southern Mississippi",
    "San Jose St.": "San Jose State",
    "La-Monroe": "Louisiana Monroe",
    "La.-Monroe": "Louisiana Monroe",
    "La.-Lafayette": "Louisiana",
    "Louisiana-Lafayette": "Louisiana",
    "Central Florida": "UCF",
    "Ala-Birmingham": "UAB",
    "South Dakota St.": "South Dakota State",
    "New Mexico St.": "New Mexico State",
    "Boston Col.": "Boston College",
    "Mississippi": "Ole Miss",
    "Pitt": "Pittsburgh",
    "Middle Tenn. St.": "Middle Tennessee",
    "Grambling St.": "Grambling",
    "Georgia St.": "Georgia State",
    "South Florida": "South Florida",
    "Connecticut": "Connecticut",
    "Texas-El Paso": "UTEP",
    "Utah St.": "Utah State",
    "Virginia St.": "Virginia State",
    "Florida Atlantic": "Florida Atlantic",
    "Coastal Carolina": "Coastal Carolina",
}

# Schools where CFBD likely won't have data (FCS/D2)
SKIP_SCHOOLS = {
    "Virginia St.",         # FCS/D2
    "North Carolina A&T",  # FCS
    "Fordham",             # FCS
    "New Hampshire",       # FCS
}


def normalize_school(school):
    """Convert school name to CFBD format."""
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def normalize_name(name):
    """Normalize player name for matching."""
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
    """Check if two names refer to the same person."""
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


# ─── CFBD API Fetch Functions ───

rushing_stats_cache = {}   # (team, year) -> {player_name: {CAR: x, YDS: y, TD: z}}
receiving_stats_cache = {} # (team, year) -> {player_name: {REC: x, YDS: y, TD: z}}
usage_cache = {}           # (team, year) -> {player_name: games}


def api_get(url, params, retries=3):
    """Make CFBD API request with rate-limit handling."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            return resp
        except Exception as e:
            print(f"    Request error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


def fetch_rushing_stats(team, year):
    """Fetch rushing stats for all players on a team for a given year."""
    cache_key = (team, year)
    if cache_key in rushing_stats_cache:
        return rushing_stats_cache[cache_key]

    resp = api_get(f"{BASE_URL}/stats/player/season",
                   {"year": year, "category": "rushing", "team": team})

    if resp and resp.status_code == 200:
        stats = {}
        for s in resp.json():
            player = s.get("player", "")
            stat_type = s.get("statType", "")
            value = s.get("stat", 0)
            if player not in stats:
                stats[player] = {}
            try:
                stats[player][stat_type] = int(float(value))
            except (ValueError, TypeError):
                pass
        rushing_stats_cache[cache_key] = stats
        return stats

    rushing_stats_cache[cache_key] = {}
    return {}


def fetch_receiving_stats(team, year):
    """Fetch receiving stats for all players on a team for a given year."""
    cache_key = (team, year)
    if cache_key in receiving_stats_cache:
        return receiving_stats_cache[cache_key]

    resp = api_get(f"{BASE_URL}/stats/player/season",
                   {"year": year, "category": "receiving", "team": team})

    if resp and resp.status_code == 200:
        stats = {}
        for s in resp.json():
            player = s.get("player", "")
            stat_type = s.get("statType", "")
            value = s.get("stat", 0)
            if player not in stats:
                stats[player] = {}
            try:
                stats[player][stat_type] = int(float(value))
            except (ValueError, TypeError):
                pass
        receiving_stats_cache[cache_key] = stats
        return stats

    receiving_stats_cache[cache_key] = {}
    return {}


def fetch_player_usage(team, year):
    """Fetch games played per player from CFBD player/usage endpoint."""
    cache_key = (team, year)
    if cache_key in usage_cache:
        return usage_cache[cache_key]

    resp = api_get(f"{BASE_URL}/player/usage",
                   {"year": year, "team": team})

    if resp and resp.status_code == 200:
        games = {}
        for p in resp.json():
            name = p.get("name", "")
            # usage endpoint has overall.games or similar
            usage = p.get("usage", {})
            if isinstance(usage, dict):
                g = usage.get("overall", {})
                if isinstance(g, (int, float)):
                    games[name] = int(g)
            # Some CFBD responses have different structure
        usage_cache[cache_key] = games
        return games

    usage_cache[cache_key] = {}
    return {}


def find_player_rushing(player_name, team_rushing):
    """Find a player's rushing stats from team stats dict."""
    if not team_rushing:
        return None
    for cfbd_player, stats in team_rushing.items():
        if names_match(player_name, cfbd_player):
            return {
                'cfbd_name': cfbd_player,
                'rush_yards': stats.get('YDS', None),
                'rush_attempts': stats.get('CAR', None),
                'rush_tds': stats.get('TD', None),
            }
    return None


def find_player_receiving(player_name, team_receiving):
    """Find a player's receiving TDs from team receiving stats."""
    if not team_receiving:
        return None
    for cfbd_player, stats in team_receiving.items():
        if names_match(player_name, cfbd_player):
            return {
                'rec_tds': stats.get('TD', None),
                'cfbd_rec_yards': stats.get('YDS', None),
                'cfbd_receptions': stats.get('REC', None),
            }
    return None


# ─── Main ───

def main():
    df = pd.read_csv('output/slap_v5_master_database.csv')
    rb_bt = df[(df['position'] == 'RB') & (df['dataset'] == 'backtest')].copy()

    # All backtest RBs need rushing stats
    targets = rb_bt[['player_name', 'college', 'draft_year', 'draft_pick', 'draft_round']].copy()
    targets = targets.sort_values(['draft_year', 'draft_pick']).reset_index(drop=True)

    print(f"RBs to fetch: {len(targets)}")
    print(f"Unique colleges: {targets['college'].nunique()}")
    print(f"Draft years: {int(targets['draft_year'].min())}-{int(targets['draft_year'].max())}")

    results = []
    found = 0
    skipped = 0
    not_found = 0

    for idx, row in targets.iterrows():
        player = row['player_name']
        college = row['college']
        draft_year = int(row['draft_year'])
        season = draft_year - 1  # Final college season

        # Skip schools not in CFBD
        if pd.isna(college) or college in SKIP_SCHOOLS:
            print(f"  [{idx+1}/{len(targets)}] SKIP {player} ({college})")
            results.append({
                'player_name': player, 'college': college, 'draft_year': draft_year,
                'season': season, 'status': 'skipped',
            })
            skipped += 1
            continue

        cfbd_school = normalize_school(college)
        if cfbd_school is None:
            print(f"  [{idx+1}/{len(targets)}] SKIP {player} ({college}) — no mapping")
            results.append({
                'player_name': player, 'college': college, 'draft_year': draft_year,
                'season': season, 'status': 'no_mapping',
            })
            skipped += 1
            continue

        # Fetch rushing stats for that team/season
        team_rushing = fetch_rushing_stats(cfbd_school, season)
        time.sleep(0.3)

        # Find the player in rushing stats
        rush_match = find_player_rushing(player, team_rushing)

        # If not found, try previous season (transfers)
        if rush_match is None and season > 2013:
            prev_rushing = fetch_rushing_stats(cfbd_school, season - 1)
            time.sleep(0.3)
            rush_match = find_player_rushing(player, prev_rushing)
            if rush_match:
                print(f"    Found {player} in {season-1} (prev season)")

        # Fetch receiving stats for rec_tds
        team_receiving = fetch_receiving_stats(cfbd_school, season)
        time.sleep(0.3)
        rec_match = find_player_receiving(player, team_receiving)

        if rec_match is None and season > 2013:
            prev_receiving = fetch_receiving_stats(cfbd_school, season - 1)
            time.sleep(0.3)
            rec_match = find_player_receiving(player, prev_receiving)

        if rush_match:
            found += 1
            result = {
                'player_name': player,
                'college': college,
                'draft_year': draft_year,
                'season': season,
                'cfbd_school': cfbd_school,
                'cfbd_name': rush_match['cfbd_name'],
                'rush_yards': rush_match['rush_yards'],
                'rush_attempts': rush_match['rush_attempts'],
                'rush_tds': rush_match['rush_tds'],
                'rec_tds': rec_match['rec_tds'] if rec_match else None,
                'status': 'found',
            }
            results.append(result)
            rec_td_str = f", {rec_match['rec_tds']} recTDs" if rec_match else ""
            print(f"  [{idx+1}/{len(targets)}] FOUND {player} ({cfbd_school} {season}): "
                  f"{rush_match['rush_yards']} yds, {rush_match['rush_attempts']} att, "
                  f"{rush_match['rush_tds']} TDs{rec_td_str}")
        else:
            not_found += 1
            results.append({
                'player_name': player, 'college': college, 'draft_year': draft_year,
                'season': season, 'cfbd_school': cfbd_school, 'status': 'not_found',
            })
            if team_rushing:
                avail = list(team_rushing.keys())[:5]
                print(f"  [{idx+1}/{len(targets)}] NOT FOUND {player} ({cfbd_school} {season}) — "
                      f"available: {avail}")
            else:
                print(f"  [{idx+1}/{len(targets)}] NOT FOUND {player} ({cfbd_school} {season}) — "
                      f"no team data")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total targets: {len(targets)}")
    print(f"  Found:         {found}")
    print(f"  Not found:     {not_found}")
    print(f"  Skipped:       {skipped}")

    # Save results
    results_df = pd.DataFrame(results)
    output_path = 'data/cfbd_rb_rushing_stats.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Show not_found for manual review
    nf = results_df[results_df['status'] == 'not_found']
    if len(nf) > 0:
        print(f"\n{'='*60}")
        print(f"NOT FOUND ({len(nf)} players):")
        print(f"{'='*60}")
        for _, r in nf.iterrows():
            print(f"  {r['player_name']:30s} {str(r.get('cfbd_school','')):25s} {r['season']}")


if __name__ == '__main__':
    main()
