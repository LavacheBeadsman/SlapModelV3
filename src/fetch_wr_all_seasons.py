"""
Pull ALL college seasons for each WR to find peak Dominator and breakout age.

For each WR:
1. Search CFBD for all seasons at their school
2. Calculate Dominator Rating for each season
3. Find peak (best) Dominator Rating
4. Find breakout age (first season with 20%+ Dominator)
"""

import time
import requests
import pandas as pd
from collections import defaultdict

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
}

FCS_SCHOOLS = {"Rhode Island", "Charleston (WV)", "North Dakota State", "Samford",
               "Lenoir-Rhyne", "Princeton", "UT Martin"}


def get_cfbd_name(school):
    if school in FCS_SCHOOLS:
        return None
    return SCHOOL_NAME_MAP.get(school, school)


def get_team_stats(team, season):
    """Get team receiving yards and TDs for a season."""
    cfbd_name = get_cfbd_name(team)
    if cfbd_name is None:
        return None, None

    url = f"{BASE_URL}/stats/season"
    params = {"year": season, "team": cfbd_name}

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
    except:
        pass
    return None, None


def get_player_receiving_stats(team, season):
    """Get all player receiving stats for a team/season."""
    cfbd_name = get_cfbd_name(team)
    if cfbd_name is None:
        return {}

    url = f"{BASE_URL}/stats/player/season"
    params = {"year": season, "team": cfbd_name, "category": "receiving"}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # Group by player
            players = defaultdict(dict)
            for stat in data:
                player = stat.get("player", "")
                stat_type = stat.get("statType", "")
                stat_val = stat.get("stat", 0)
                players[player][stat_type] = float(stat_val)
            return dict(players)
    except:
        pass
    return {}


def normalize_name(name):
    """Normalize player name for matching."""
    return name.lower().replace(".", "").replace("'", "").replace("-", " ").strip()


def match_player(our_name, cfbd_players):
    """Find matching player in CFBD data."""
    our_norm = normalize_name(our_name)
    our_parts = our_norm.split()

    if len(our_parts) < 2:
        return None, None

    our_last = our_parts[-1]
    our_first = our_parts[0]

    for cfbd_name, stats in cfbd_players.items():
        cfbd_norm = normalize_name(cfbd_name)
        cfbd_parts = cfbd_norm.split()

        if len(cfbd_parts) < 2:
            continue

        cfbd_last = cfbd_parts[-1]
        cfbd_first = cfbd_parts[0]

        # Match on last name and first initial
        if our_last == cfbd_last and (our_first == cfbd_first or our_first[0] == cfbd_first[0]):
            return cfbd_name, stats

    return None, None


def main():
    print("=" * 70)
    print("PULLING ALL COLLEGE SEASONS FOR WR BREAKOUT AGE ANALYSIS")
    print("=" * 70)
    print()

    # Load backtest data
    bt = pd.read_csv("data/backtest_college_stats.csv")
    wrs = bt[(bt['position'] == 'WR') & (bt['draft_year'] <= 2024)].copy()
    print(f"WRs to process: {len(wrs)}")

    # We'll search seasons from 2017-2023 (covers freshman through senior years for 2020-2024 drafts)
    SEASONS = range(2017, 2024)

    # Cache team stats to avoid redundant API calls
    team_stats_cache = {}
    player_stats_cache = {}

    # Track all seasons found for each WR
    all_wr_seasons = []

    # Process each WR
    for idx, row in wrs.iterrows():
        name = row['player_name']
        school = row['college']
        draft_year = row['draft_year']

        cfbd_school = get_cfbd_name(school)
        if cfbd_school is None:
            print(f"  Skipping {name} - FCS school ({school})")
            continue

        # Search all relevant seasons at this school
        seasons_found = []

        # Search from 4 years before draft through draft year - 1
        search_start = draft_year - 5
        search_end = draft_year - 1

        for season in range(search_start, search_end + 1):
            if season < 2017:  # CFBD data starts ~2017
                continue

            cache_key = (school, season)

            # Get team stats (cached)
            if cache_key not in team_stats_cache:
                team_rec_yards, team_rec_tds = get_team_stats(school, season)
                team_stats_cache[cache_key] = (team_rec_yards, team_rec_tds)
                time.sleep(0.15)
            else:
                team_rec_yards, team_rec_tds = team_stats_cache[cache_key]

            # Get player stats (cached)
            if cache_key not in player_stats_cache:
                players = get_player_receiving_stats(school, season)
                player_stats_cache[cache_key] = players
                time.sleep(0.15)
            else:
                players = player_stats_cache[cache_key]

            # Find this player
            cfbd_name, stats = match_player(name, players)

            if cfbd_name and stats:
                player_yards = stats.get('YDS', 0)
                player_tds = stats.get('TD', 0)
                player_rec = stats.get('REC', 0)

                # Calculate Dominator Rating (yards-only for now)
                if team_rec_yards and team_rec_yards > 0:
                    dominator = (player_yards / team_rec_yards) * 100
                else:
                    dominator = None

                seasons_found.append({
                    'player_name': name,
                    'draft_year': draft_year,
                    'college': school,
                    'season': season,
                    'cfbd_name': cfbd_name,
                    'player_rec_yards': player_yards,
                    'player_rec_tds': player_tds,
                    'player_receptions': player_rec,
                    'team_rec_yards': team_rec_yards,
                    'team_rec_tds': team_rec_tds,
                    'dominator_rating': round(dominator, 1) if dominator else None
                })

        if seasons_found:
            all_wr_seasons.extend(seasons_found)
            print(f"  {name}: {len(seasons_found)} seasons found")
        else:
            print(f"  {name}: NO seasons found at {school}")

    # Save all seasons data
    df = pd.DataFrame(all_wr_seasons)
    df.to_csv("data/wr_all_seasons.csv", index=False)
    print(f"\nSaved {len(df)} season records to data/wr_all_seasons.csv")

    # Show sample analysis
    print()
    print("=" * 70)
    print("SAMPLE: JA'MARR CHASE SEASONS")
    print("=" * 70)
    chase = df[df['player_name'] == "Ja'Marr Chase"].sort_values('season')
    if len(chase) > 0:
        for _, s in chase.iterrows():
            print(f"  {s['season']}: {s['player_rec_yards']:.0f} yds, {s['player_rec_tds']:.0f} TDs, "
                  f"Dominator: {s['dominator_rating']:.1f}%")
    else:
        print("  No data found")

    print()
    print("=" * 70)
    print("SAMPLE: LATE BREAKOUT WR")
    print("=" * 70)
    # Find a WR with multiple seasons where they got better
    multi_season = df.groupby('player_name').size()
    multi_season_wrs = multi_season[multi_season >= 3].index.tolist()

    if multi_season_wrs:
        # Find one where later season has higher dominator
        for wr_name in multi_season_wrs[:10]:
            wr_data = df[df['player_name'] == wr_name].sort_values('season')
            if len(wr_data) >= 2:
                first_dom = wr_data.iloc[0]['dominator_rating']
                last_dom = wr_data.iloc[-1]['dominator_rating']
                if pd.notna(first_dom) and pd.notna(last_dom) and last_dom > first_dom + 10:
                    print(f"\n{wr_name} (peaked later):")
                    for _, s in wr_data.iterrows():
                        dom = s['dominator_rating']
                        if pd.notna(dom):
                            print(f"  {s['season']}: {s['player_rec_yards']:.0f} yds, Dominator: {dom:.1f}%")
                    break

    # Summary stats
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    unique_wrs = df['player_name'].nunique()
    total_seasons = len(df)
    avg_seasons = total_seasons / unique_wrs if unique_wrs > 0 else 0

    print(f"WRs with data: {unique_wrs}")
    print(f"Total season records: {total_seasons}")
    print(f"Average seasons per WR: {avg_seasons:.1f}")

    # Show WRs with only 1 season
    one_season = multi_season[multi_season == 1].index.tolist()
    print(f"\nWRs with only 1 season found: {len(one_season)}")
    for wr in one_season[:10]:
        wr_row = df[df['player_name'] == wr].iloc[0]
        print(f"  {wr} ({wr_row['college']} {wr_row['season']})")

    # Show WRs with no data at all
    wrs_with_data = set(df['player_name'].unique())
    all_wrs = set(wrs['player_name'].unique())
    missing_wrs = all_wrs - wrs_with_data
    print(f"\nWRs with NO data: {len(missing_wrs)}")
    for wr in list(missing_wrs)[:10]:
        wr_row = wrs[wrs['player_name'] == wr].iloc[0]
        print(f"  {wr} ({wr_row['college']})")


if __name__ == "__main__":
    main()
