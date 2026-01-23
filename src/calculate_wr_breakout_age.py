"""
Fix name mismatches and calculate WR breakout scores.

Breakout formula:
  breakout_raw = peak_dominator × age_multiplier

Age multipliers based on breakout age (first 20%+ Dominator season):
  - Age 18: 1.25
  - Age 19: 1.15
  - Age 20: 1.05
  - Age 21: 1.00
  - Age 22: 0.95
  - Age 23+: 0.90
  - No breakout: peak × 0.80
"""

import time
import requests
import pandas as pd
from collections import defaultdict

API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

# Name mappings for mismatched players
NAME_FIXES = {
    ("Tutu Atwell", "Louisville"): "Chatarius Atwell",
    ("Tank Dell", "Houston"): "Nathaniel Dell",
    ("Brian Thomas", "LSU"): "Brian Thomas Jr.",
    ("Michael Woods II", "Oklahoma"): "Michael Woods",
    ("Bub Means", "Pittsburgh"): "Jerrod Means",
}

# Age multipliers for breakout age
BREAKOUT_AGE_MULT = {
    18: 1.25,
    19: 1.15,
    20: 1.05,
    21: 1.00,
    22: 0.95,
    23: 0.90,
    24: 0.90,
    25: 0.90,
}
NO_BREAKOUT_MULT = 0.80


def get_team_stats(team, season):
    """Get team receiving yards and TDs."""
    url = f"{BASE_URL}/stats/season"
    params = {"year": season, "team": team}
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


def get_player_stats(team, season, cfbd_name):
    """Get player receiving stats."""
    url = f"{BASE_URL}/stats/player/season"
    params = {"year": season, "team": team, "category": "receiving"}
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            stats = {}
            for stat in data:
                if cfbd_name.lower() in stat.get("player", "").lower():
                    stats[stat.get("statType")] = float(stat.get("stat", 0))
            if stats:
                return stats.get("YDS", 0), stats.get("TD", 0), stats.get("REC", 0)
    except:
        pass
    return None, None, None


def fetch_missing_players(bt):
    """Fetch data for players with name mismatches."""
    new_records = []

    for (our_name, school), cfbd_name in NAME_FIXES.items():
        print(f"\nFetching {our_name} (as {cfbd_name}) from {school}...")

        # Get draft year from backtest
        player_row = bt[(bt['player_name'] == our_name) & (bt['college'] == school)]
        if len(player_row) == 0:
            print(f"  Not found in backtest")
            continue

        draft_year = int(player_row.iloc[0]['draft_year'])

        # Search multiple seasons
        for season in range(draft_year - 5, draft_year):
            if season < 2017:
                continue

            team_yards, team_tds = get_team_stats(school, season)
            if not team_yards:
                continue

            player_yards, player_tds, player_rec = get_player_stats(school, season, cfbd_name)
            if player_yards is None or player_yards == 0:
                continue

            dominator = (player_yards / team_yards) * 100 if team_yards > 0 else None

            print(f"  {season}: {player_yards:.0f} yds, {player_tds:.0f} TDs, Dom: {dominator:.1f}%")

            new_records.append({
                'player_name': our_name,
                'draft_year': draft_year,
                'college': school,
                'season': season,
                'cfbd_name': cfbd_name,
                'player_rec_yards': player_yards,
                'player_rec_tds': player_tds,
                'player_receptions': player_rec,
                'team_rec_yards': team_yards,
                'team_rec_tds': team_tds,
                'dominator_rating': round(dominator, 1) if dominator else None
            })

            time.sleep(0.2)

    return pd.DataFrame(new_records)


def calculate_breakout_scores(df, bt):
    """Calculate breakout scores for all WRs."""

    # Get ages from backtest
    bt_ages = dict(zip(zip(bt['player_name'], bt['draft_year']), bt['age']))

    results = []

    for name in df['player_name'].unique():
        player_df = df[df['player_name'] == name].sort_values('season')
        draft_year = int(player_df.iloc[0]['draft_year'])
        college = player_df.iloc[0]['college']

        # Get draft age
        draft_age = bt_ages.get((name, draft_year))
        if pd.isna(draft_age):
            draft_age = 22  # Default assumption

        # Calculate season ages
        seasons_data = []
        for _, row in player_df.iterrows():
            season = int(row['season'])
            season_age = int(draft_age - (draft_year - season))
            dom = row['dominator_rating']
            seasons_data.append({
                'season': season,
                'age': season_age,
                'dominator': dom
            })

        # Find peak dominator
        valid_doms = [s['dominator'] for s in seasons_data if pd.notna(s['dominator'])]
        peak_dom = max(valid_doms) if valid_doms else None

        # Find breakout age (first season with 20%+ dominator)
        breakout_seasons = [s for s in seasons_data if pd.notna(s['dominator']) and s['dominator'] >= 20]
        if breakout_seasons:
            breakout_age = min(s['age'] for s in breakout_seasons)
            age_mult = BREAKOUT_AGE_MULT.get(breakout_age, 0.90)
        else:
            breakout_age = None
            age_mult = NO_BREAKOUT_MULT

        # Calculate breakout_raw
        if peak_dom is not None:
            breakout_raw = peak_dom * age_mult
        else:
            breakout_raw = None

        results.append({
            'player_name': name,
            'draft_year': draft_year,
            'college': college,
            'peak_dominator': round(peak_dom, 1) if peak_dom else None,
            'breakout_age': breakout_age,
            'age_multiplier': age_mult,
            'breakout_raw': round(breakout_raw, 1) if breakout_raw else None,
            'seasons_found': len(player_df)
        })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("FIXING NAME MISMATCHES AND CALCULATING BREAKOUT SCORES")
    print("=" * 70)

    # Load data
    df = pd.read_csv("data/wr_all_seasons.csv")
    bt = pd.read_csv("data/backtest_college_stats.csv")

    print(f"\nOriginal records: {len(df)}")
    print(f"Original WRs: {df['player_name'].nunique()}")

    # Fetch missing players
    print("\n" + "=" * 70)
    print("TASK 1: FETCHING MISSING PLAYERS")
    print("=" * 70)

    new_records = fetch_missing_players(bt)
    print(f"\nNew records fetched: {len(new_records)}")

    # Combine with existing data
    df = pd.concat([df, new_records], ignore_index=True)
    df.to_csv("data/wr_all_seasons.csv", index=False)
    print(f"Updated total: {len(df)} records, {df['player_name'].nunique()} WRs")

    # Calculate breakout scores
    print("\n" + "=" * 70)
    print("TASK 2: CALCULATING BREAKOUT SCORES")
    print("=" * 70)

    breakout_df = calculate_breakout_scores(df, bt)
    breakout_df.to_csv("data/wr_breakout_age_scores.csv", index=False)

    print(f"\nBreakout scores calculated for {len(breakout_df)} WRs")

    # Show top 20
    print("\n" + "=" * 70)
    print("TOP 20 WRS BY BREAKOUT_RAW SCORE")
    print("=" * 70)
    print(f"\n{'Rank':<5} {'Player':<25} {'Peak Dom':<10} {'BO Age':<8} {'Mult':<6} {'Raw':<8}")
    print("-" * 70)

    top20 = breakout_df.nlargest(20, 'breakout_raw')
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        bo_age = str(int(row['breakout_age'])) if pd.notna(row['breakout_age']) else "Never"
        print(f"{i:<5} {row['player_name']:<25} {row['peak_dominator']:>7.1f}%  "
              f"{bo_age:<8} {row['age_multiplier']:<6.2f} {row['breakout_raw']:>6.1f}")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    with_breakout = breakout_df[breakout_df['breakout_age'].notna()]
    no_breakout = breakout_df[breakout_df['breakout_age'].isna()]

    print(f"\nWRs with 20%+ breakout: {len(with_breakout)} ({len(with_breakout)/len(breakout_df)*100:.1f}%)")
    print(f"WRs without breakout: {len(no_breakout)} ({len(no_breakout)/len(breakout_df)*100:.1f}%)")

    print("\nBreakout Age Distribution:")
    for age in sorted(with_breakout['breakout_age'].unique()):
        count = len(with_breakout[with_breakout['breakout_age'] == age])
        print(f"  Age {int(age)}: {count} WRs")


if __name__ == "__main__":
    main()
