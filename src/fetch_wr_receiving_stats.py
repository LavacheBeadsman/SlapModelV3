"""
Fetch WR receiving stats from CFBD API for backtest WRs missing rec_yards.
Targets: 162 WRs from 2015-2025 drafts missing final college season production data.

Fetches per player (final college season = draft_year - 1):
  - rec_yards (receiving yards)
  - receptions
  - rec_tds (receiving TDs)
  - team_pass_att (team total pass attempts)

Output: data/wr_backtest_receiving_cfbd.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import re

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
    "Appalachian St.": "Appalachian State",
    "Colorado St.": "Colorado State",
    "Oregon St.": "Oregon State",
    "Washington St.": "Washington State",
    "S. Carolina": "South Carolina",
    "Miami (FL)": "Miami",
    "Central Mich.": "Central Michigan",
    "Western Mich.": "Western Michigan",
    "Eastern Mich.": "Eastern Michigan",
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
}

# Schools where CFBD won't have data (FCS/D2/D3 not in CFBD)
SKIP_SCHOOLS = {
    "Charleston (WV)",      # D2
    "East Central (OK)",    # D2 (NAIA actually)
    "Monmouth",             # FCS (may be in CFBD)
    "Pennsylvania",         # FCS (Ivy League)
    "William & Mary",       # FCS
    "Central Arkansas",     # FCS
    "Eastern Washington",   # FCS
    "Northern Iowa",        # FCS
    "West Alabama",         # D2
    "Grambling St.",        # FCS
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
    # Remove accents
    for k, v in {'é': 'e', 'è': 'e', 'ê': 'e', 'á': 'a', 'à': 'a',
                 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n'}.items():
        s = s.replace(k, v)
    # Remove punctuation
    s = s.replace('.', '').replace("'", '').replace("'", '').replace('-', ' ')
    # Remove suffixes
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

    # Compare last names
    if parts1[-1] != parts2[-1]:
        return False

    # Compare first names (allow initial match, e.g. "D" matches "DeVante")
    first1, first2 = parts1[0], parts2[0]
    if first1 == first2:
        return True
    if len(first1) <= 2 and first2.startswith(first1[0]):
        return True
    if len(first2) <= 2 and first1.startswith(first2[0]):
        return True

    return False


# ─── CFBD API Fetch Functions ───

# Caches to avoid re-fetching the same team/year
player_stats_cache = {}  # (team, year) -> {player_name: {REC: x, YDS: y, TD: z}}
team_pass_att_cache = {}  # (team, year) -> int


def fetch_player_receiving_stats(team, year):
    """Fetch receiving stats for all players on a team for a given year."""
    cache_key = (team, year)
    if cache_key in player_stats_cache:
        return player_stats_cache[cache_key]

    url = f"{BASE_URL}/stats/player/season"
    params = {"year": year, "category": "receiving", "team": team}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 429:
            print(f"    Rate limited, waiting 5s...")
            time.sleep(5)
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)

        if resp.status_code == 200:
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
            player_stats_cache[cache_key] = stats
            return stats
        else:
            print(f"    API error {resp.status_code} for {team} {year}")
            player_stats_cache[cache_key] = {}
            return {}
    except Exception as e:
        print(f"    Exception fetching {team} {year}: {e}")
        player_stats_cache[cache_key] = {}
        return {}


def fetch_team_pass_attempts(team, year):
    """Fetch team total pass attempts from CFBD."""
    cache_key = (team, year)
    if cache_key in team_pass_att_cache:
        return team_pass_att_cache[cache_key]

    url = f"{BASE_URL}/stats/season"
    params = {"year": year, "team": team}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 429:
            print(f"    Rate limited, waiting 5s...")
            time.sleep(5)
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)

        if resp.status_code == 200:
            for s in resp.json():
                if s.get("statName") == "passAttempts":
                    val = float(s.get("statValue", 0))
                    team_pass_att_cache[cache_key] = val
                    return val
            team_pass_att_cache[cache_key] = None
            return None
        else:
            print(f"    API error {resp.status_code} for team stats {team} {year}")
            team_pass_att_cache[cache_key] = None
            return None
    except Exception as e:
        print(f"    Exception fetching team stats {team} {year}: {e}")
        team_pass_att_cache[cache_key] = None
        return None


def find_player_in_stats(player_name, team_stats):
    """Find a player's receiving stats from the team stats dict."""
    if not team_stats:
        return None

    # Try exact normalized match first
    for cfbd_player, stats in team_stats.items():
        if names_match(player_name, cfbd_player):
            return {
                'cfbd_name': cfbd_player,
                'rec_yards': stats.get('YDS', None),
                'receptions': stats.get('REC', None),
                'rec_tds': stats.get('TD', None),
            }

    return None


# ─── Main ───

def main():
    # Load master database
    df = pd.read_csv('output/slap_v5_master_database.csv')
    wr_bt = df[(df['position'] == 'WR') & (df['dataset'] == 'backtest')].copy()

    # Find WRs missing rec_yards
    missing = wr_bt[wr_bt['rec_yards'].isna()][['player_name', 'college', 'draft_year', 'draft_pick']].copy()
    missing = missing.sort_values(['draft_year', 'draft_pick']).reset_index(drop=True)

    print(f"WRs missing rec_yards: {len(missing)}")
    print(f"Unique colleges: {missing['college'].nunique()}")

    # Results storage
    results = []
    found = 0
    skipped = 0
    not_found = 0

    for idx, row in missing.iterrows():
        player = row['player_name']
        college = row['college']
        draft_year = int(row['draft_year'])
        season = draft_year - 1  # Final college season

        # Skip schools not in CFBD
        if pd.isna(college) or college in SKIP_SCHOOLS:
            print(f"  [{idx+1}/{len(missing)}] SKIP {player} ({college}) — not in CFBD")
            results.append({
                'player_name': player,
                'college': college,
                'draft_year': draft_year,
                'season': season,
                'status': 'skipped',
            })
            skipped += 1
            continue

        cfbd_school = normalize_school(college)
        if cfbd_school is None:
            print(f"  [{idx+1}/{len(missing)}] SKIP {player} ({college}) — no mapping")
            results.append({
                'player_name': player,
                'college': college,
                'draft_year': draft_year,
                'season': season,
                'status': 'no_mapping',
            })
            skipped += 1
            continue

        # Fetch team receiving stats for that season
        team_stats = fetch_player_receiving_stats(cfbd_school, season)
        time.sleep(0.3)

        # Find the player
        match = find_player_in_stats(player, team_stats)

        # If not found, try previous season (transfers sometimes)
        if match is None and season > 2013:
            prev_stats = fetch_player_receiving_stats(cfbd_school, season - 1)
            time.sleep(0.3)
            match = find_player_in_stats(player, prev_stats)
            if match:
                print(f"  [{idx+1}/{len(missing)}] FOUND {player} in {season-1} (prev season)")

        # Fetch team pass attempts
        team_pa = fetch_team_pass_attempts(cfbd_school, season)
        time.sleep(0.3)

        if match:
            found += 1
            result = {
                'player_name': player,
                'college': college,
                'draft_year': draft_year,
                'season': season,
                'cfbd_school': cfbd_school,
                'cfbd_name': match['cfbd_name'],
                'rec_yards': match['rec_yards'],
                'receptions': match['receptions'],
                'rec_tds': match['rec_tds'],
                'team_pass_att': team_pa,
                'status': 'found',
            }
            results.append(result)
            print(f"  [{idx+1}/{len(missing)}] FOUND {player} ({cfbd_school} {season}): "
                  f"{match['rec_yards']} yds, {match['receptions']} rec, {match['rec_tds']} TDs, "
                  f"team PA={team_pa}")
        else:
            not_found += 1
            results.append({
                'player_name': player,
                'college': college,
                'draft_year': draft_year,
                'season': season,
                'cfbd_school': cfbd_school,
                'team_pass_att': team_pa,
                'status': 'not_found',
            })
            # Show available players for debugging
            if team_stats:
                avail = list(team_stats.keys())[:5]
                print(f"  [{idx+1}/{len(missing)}] NOT FOUND {player} ({cfbd_school} {season}) — "
                      f"available: {avail}")
            else:
                print(f"  [{idx+1}/{len(missing)}] NOT FOUND {player} ({cfbd_school} {season}) — "
                      f"no team data")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total missing: {len(missing)}")
    print(f"  Found:         {found}")
    print(f"  Not found:     {not_found}")
    print(f"  Skipped:       {skipped}")

    # Save results
    results_df = pd.DataFrame(results)
    output_path = 'data/wr_backtest_receiving_cfbd.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Show not_found for manual review
    nf = results_df[results_df['status'] == 'not_found']
    if len(nf) > 0:
        print(f"\n{'='*60}")
        print(f"NOT FOUND ({len(nf)} players) — may need name fixes:")
        print(f"{'='*60}")
        for _, r in nf.iterrows():
            print(f"  {r['player_name']:30s} {r['cfbd_school']:25s} {r['season']}")


if __name__ == '__main__':
    main()
