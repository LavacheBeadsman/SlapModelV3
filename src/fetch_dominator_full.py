"""
Fetch full Dominator Rating data from CFBD API.

Correct Dominator Rating formula:
  dominator = (player_rec_yards / team_rec_yards + player_rec_tds / team_rec_tds) / 2

Usage:
  python src/fetch_dominator_full.py              # All years
  python src/fetch_dominator_full.py 2020 2021    # Specific year range
"""

import sys
import time
import requests
import pandas as pd

API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

SCHOOL_NAME_MAP = {
    "Boston Col.": "Boston College",
    "Mississippi": "Ole Miss",
    "Ala-Birmingham": "UAB",
    "SE Missouri St.": "Southeast Missouri State",
    "South Dakota St.": "South Dakota State",
    "New Mexico St.": "New Mexico State",
    "Rhode Island": None, "Charleston (WV)": None, "North Dakota State": None,
    "Samford": None, "Lenoir-Rhyne": None, "Princeton": None, "UT Martin": None,
}


def get_cfbd_name(school):
    return SCHOOL_NAME_MAP.get(school, school)


def get_team_stats(team, season, retries=2):
    """Fetch team receiving yards and TDs."""
    cfbd_name = get_cfbd_name(team)
    if cfbd_name is None:
        return None, None

    url = f"{BASE_URL}/stats/season"
    params = {"year": season, "team": cfbd_name}

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                rec_yards, rec_tds = None, None
                for stat in data:
                    if stat.get("statName") == "netPassingYards":
                        rec_yards = float(stat.get("statValue", 0))
                    elif stat.get("statName") == "passingTDs":
                        rec_tds = float(stat.get("statValue", 0))
                return rec_yards, rec_tds
            elif response.status_code == 503:
                time.sleep(2 ** attempt)
        except:
            time.sleep(1)
    return None, None


def get_player_tds_batch(team, season, retries=2):
    """Fetch all player receiving TDs for a team/season."""
    cfbd_name = get_cfbd_name(team)
    if cfbd_name is None:
        return {}

    url = f"{BASE_URL}/stats/player/season"
    params = {"year": season, "team": cfbd_name, "category": "receiving"}

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                # Build dict of player -> TDs
                result = {}
                for stat in data:
                    if stat.get("statType") == "TD":
                        player = stat.get("player", "").lower()
                        result[player] = float(stat.get("stat", 0))
                return result
        except:
            time.sleep(1)
    return {}


def match_player_name(our_name, player_tds_dict):
    """Find player TDs from dict, handling name variations."""
    our_name_lower = our_name.lower().replace(".", "").replace("'", "").replace("-", " ")
    our_parts = our_name_lower.split()

    if len(our_parts) < 2:
        return None

    our_last = our_parts[-1]
    our_first = our_parts[0]

    for cfbd_name, tds in player_tds_dict.items():
        cfbd_clean = cfbd_name.replace(".", "").replace("'", "").replace("-", " ")
        cfbd_parts = cfbd_clean.split()

        if len(cfbd_parts) < 2:
            continue

        cfbd_last = cfbd_parts[-1]
        cfbd_first = cfbd_parts[0]

        # Last name must match, first name or initial
        if our_last == cfbd_last:
            if our_first == cfbd_first or our_first[0] == cfbd_first[0]:
                return tds

    return None


def process_batch(wrs, batch_name):
    """Process a batch of WRs."""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {batch_name} ({len(wrs)} WRs)")
    print(f"{'='*60}")

    if len(wrs) == 0:
        return pd.DataFrame()

    wrs = wrs.copy()
    wrs['season'] = wrs['draft_year'] - 1

    # Get unique school/season combos
    combos = wrs[['college', 'season']].drop_duplicates()
    print(f"Unique teams: {len(combos)}")

    # Fetch team stats and player TDs together
    team_stats = {}
    player_tds_cache = {}

    for _, row in combos.iterrows():
        school = row['college']
        season = int(row['season'])
        key = (school, season)

        cfbd_name = get_cfbd_name(school)
        if cfbd_name is None:
            continue

        print(f"  {cfbd_name} {season}...", end=" ", flush=True)

        # Get team stats
        rec_yards, rec_tds = get_team_stats(school, season)
        if rec_yards and rec_tds:
            team_stats[key] = {"rec_yards": rec_yards, "rec_tds": rec_tds}
            print(f"team OK", end=" ", flush=True)
        else:
            print(f"team MISSING", end=" ", flush=True)

        # Get player TDs
        player_tds = get_player_tds_batch(school, season)
        player_tds_cache[key] = player_tds
        print(f"({len(player_tds)} players)")

        time.sleep(0.2)

    # Calculate Dominator for each WR
    results = []
    for _, row in wrs.iterrows():
        name = row['player_name']
        school = row['college']
        season = int(row['season'])
        draft_year = row['draft_year']
        key = (school, season)

        player_rec_yards = row['rec_yards']

        # Get player TDs from cache
        player_tds_dict = player_tds_cache.get(key, {})
        player_rec_tds = match_player_name(name, player_tds_dict)

        # Get team stats
        team = team_stats.get(key, {})
        team_rec_yards = team.get("rec_yards")
        team_rec_tds = team.get("rec_tds")

        # Calculate shares
        yards_share = None
        tds_share = None
        dominator = None

        if pd.notna(player_rec_yards) and team_rec_yards and team_rec_yards > 0:
            yards_share = player_rec_yards / team_rec_yards

        if player_rec_tds is not None and team_rec_tds and team_rec_tds > 0:
            tds_share = player_rec_tds / team_rec_tds

        if yards_share is not None and tds_share is not None:
            dominator = (yards_share + tds_share) / 2 * 100

        results.append({
            'player_name': name,
            'draft_year': draft_year,
            'college': school,
            'season': season,
            'player_rec_yards': player_rec_yards,
            'player_rec_tds': player_rec_tds,
            'team_rec_yards': team_rec_yards,
            'team_rec_tds': team_rec_tds,
            'yards_share': round(yards_share * 100, 1) if yards_share else None,
            'tds_share': round(tds_share * 100, 1) if tds_share else None,
            'dominator_rating': round(dominator, 1) if dominator else None
        })

    df = pd.DataFrame(results)
    valid = df['dominator_rating'].notna().sum()
    print(f"\nBatch complete: {valid}/{len(df)} valid Dominator Ratings")

    return df


def main():
    # Load backtest data
    bt = pd.read_csv("data/backtest_college_stats.csv")
    wrs = bt[(bt['position'] == 'WR') & (bt['draft_year'] <= 2023)].copy()

    # Check for year range args
    if len(sys.argv) >= 3:
        year_start = int(sys.argv[1])
        year_end = int(sys.argv[2])
        wrs = wrs[(wrs['draft_year'] >= year_start) & (wrs['draft_year'] <= year_end)]
        batch_name = f"Years {year_start}-{year_end}"
        output_file = f"data/wr_dominator_{year_start}_{year_end}.csv"
    else:
        batch_name = "All Years (2020-2023)"
        output_file = "data/wr_dominator_full.csv"

    print(f"Total WRs: {len(wrs)}")

    # Process
    df = process_batch(wrs, batch_name)

    if len(df) > 0:
        df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")

        # Show top 5
        print("\nTOP 5 DOMINATOR RATINGS:")
        top = df.nlargest(5, 'dominator_rating')
        for _, r in top.iterrows():
            if pd.notna(r['dominator_rating']):
                print(f"  {r['player_name']:25s}: {r['dominator_rating']:.1f}% "
                      f"(yards: {r['yards_share']:.1f}%, TDs: {r['tds_share']:.1f}%)")

        # Show missing count
        missing = df[df['dominator_rating'].isna()]
        if len(missing) > 0:
            print(f"\nMissing: {len(missing)} players")
            for _, r in missing.head(5).iterrows():
                print(f"  {r['player_name']} ({r['college']})")


if __name__ == "__main__":
    main()
