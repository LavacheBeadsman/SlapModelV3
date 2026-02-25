"""
fix_dominator_games_played.py — Fix dominator ratings for injury-shortened seasons.

Bug: dominator_rating = player_season_yards / team_season_yards. When a player
misses games, their yards are partial but team yards cover the full season.
This deflates dominator for injured players (e.g., Waddle 2020: 5 games = 11.1%
raw, but ~29% per-game).

Fix: normalize to per-game rates.
  adjusted_dom = (player_yards/player_games) / (team_yards/team_games) * share_factor
  where share_factor accounts for both yards and TDs like the original formula.

Data sources:
  - Team games: CFBD /games endpoint (regular + postseason)
  - Player games: CFBD /games/players endpoint (games with receiving stats)
  - Fallback: wr_games_played.csv for final seasons
"""

import time
import requests
import pandas as pd
import numpy as np
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
    "N.C. State": "NC State",
    "Miami (Fla.)": "Miami",
    "Miami (FL)": "Miami",
    "Miami (Ohio)": "Miami (OH)",
    "Pitt": "Pittsburgh",
    "Southern California": "USC",
    "Central Florida": "UCF",
}

FCS_SCHOOLS = {"Rhode Island", "Charleston (WV)", "North Dakota State", "Samford",
               "Lenoir-Rhyne", "Princeton", "UT Martin"}


def get_cfbd_name(school):
    if school in FCS_SCHOOLS:
        return None
    return SCHOOL_NAME_MAP.get(school, school)


def normalize_name(name):
    return name.lower().replace(".", "").replace("'", "").replace("-", " ").replace("'", "").strip()


def fetch_team_games(team, season, retries=3):
    """Get number of games a team played in a season (regular + postseason)."""
    cfbd_name = get_cfbd_name(team)
    if cfbd_name is None:
        return None

    for attempt in range(retries):
        try:
            total = 0
            for st in ["regular", "postseason"]:
                url = f"{BASE_URL}/games"
                params = {"year": season, "team": cfbd_name, "seasonType": st}
                r = requests.get(url, headers=HEADERS, params=params, timeout=15)
                if r.status_code == 200:
                    games = r.json()
                    # Count completed games (have scores)
                    completed = [g for g in games if g.get("homePoints") is not None or g.get("awayPoints") is not None]
                    total += len(completed)
                time.sleep(0.1)
            return total if total > 0 else None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None


def fetch_player_games_for_team(team, season, retries=3):
    """Get per-player game counts for all receivers on a team in a season."""
    cfbd_name = get_cfbd_name(team)
    if cfbd_name is None:
        return {}

    player_game_ids = defaultdict(set)

    for attempt in range(retries):
        try:
            for st in ["regular", "postseason"]:
                url = f"{BASE_URL}/games/players"
                params = {"year": season, "team": cfbd_name, "seasonType": st, "category": "receiving"}
                r = requests.get(url, headers=HEADERS, params=params, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    for game in data:
                        game_id = game.get("id")
                        for team_data in game.get("teams", []):
                            # Only count stats for OUR team
                            if normalize_name(team_data.get("school", "")) != normalize_name(cfbd_name):
                                continue
                            for cat in team_data.get("categories", []):
                                if cat.get("name") == "receiving":
                                    for stat_type in cat.get("types", []):
                                        if stat_type.get("name") == "YDS":
                                            for athlete in stat_type.get("athletes", []):
                                                name = athlete.get("name", "")
                                                stat = athlete.get("stat", "0")
                                                if stat not in ("0", "--", ""):
                                                    player_game_ids[name].add(game_id)
                time.sleep(0.1)

            return {name: len(games) for name, games in player_game_ids.items()}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {}


def match_player_in_games(our_name, player_games_dict):
    """Match a player name from our data to CFBD game data."""
    our_norm = normalize_name(our_name)
    our_parts = our_norm.split()
    if len(our_parts) < 2:
        return None

    our_last = our_parts[-1]
    our_first = our_parts[0]

    for cfbd_name, games in player_games_dict.items():
        cfbd_norm = normalize_name(cfbd_name)
        cfbd_parts = cfbd_norm.split()
        if len(cfbd_parts) < 2:
            continue
        cfbd_last = cfbd_parts[-1]
        cfbd_first = cfbd_parts[0]

        if our_last == cfbd_last and (our_first == cfbd_first or our_first[0] == cfbd_first[0]):
            return games

    return None


def main():
    print("=" * 80)
    print("FIXING DOMINATOR RATINGS FOR INJURY-SHORTENED SEASONS")
    print("=" * 80)

    seasons = pd.read_csv("data/wr_all_seasons.csv")
    games_played = pd.read_csv("data/wr_games_played.csv")

    # Get unique team-seasons
    team_seasons = seasons.groupby(["college", "season"]).size().reset_index()
    team_seasons.columns = ["college", "season", "count"]
    print(f"\n  Unique team-seasons to fetch: {len(team_seasons)}")

    # Phase 1: Fetch team game counts
    print("\n  Phase 1: Fetching team game counts from CFBD...")
    team_games_cache = {}
    fetched = 0
    for _, row in team_seasons.iterrows():
        college = row["college"]
        season = int(row["season"])
        key = (college, season)

        tg = fetch_team_games(college, season)
        team_games_cache[key] = tg
        fetched += 1

        if fetched % 50 == 0:
            print(f"    Fetched {fetched}/{len(team_seasons)} team-seasons...")

    found = sum(1 for v in team_games_cache.values() if v is not None)
    print(f"    Done. Found game counts for {found}/{len(team_seasons)} team-seasons")

    # Phase 2: Fetch player game counts
    print("\n  Phase 2: Fetching player game counts from CFBD...")
    player_games_cache = {}  # (college, season) -> {player_name: games}
    fetched = 0
    for _, row in team_seasons.iterrows():
        college = row["college"]
        season = int(row["season"])
        key = (college, season)

        pg = fetch_player_games_for_team(college, season)
        player_games_cache[key] = pg
        fetched += 1

        if fetched % 50 == 0:
            print(f"    Fetched {fetched}/{len(team_seasons)} team-seasons...")

    print(f"    Done.")

    # Phase 3: Calculate adjusted dominator for each player-season
    print("\n  Phase 3: Recalculating dominator ratings...")

    # Build lookup for wr_games_played (final season fallback)
    gp_lookup = {}
    for _, row in games_played.iterrows():
        gp_lookup[(row["player_name"], int(row["draft_year"]))] = row["games_played"]

    changes = []
    for idx, row in seasons.iterrows():
        name = row["player_name"]
        college = row["college"]
        season = int(row["season"])
        draft_year = int(row["draft_year"])
        old_dom = row["dominator_rating"]

        if pd.isna(old_dom):
            continue

        # Get team games
        tg = team_games_cache.get((college, season))
        if tg is None or tg == 0:
            continue

        # Get player games
        pg_dict = player_games_cache.get((college, season), {})
        pg = match_player_in_games(name, pg_dict)

        # Fallback: if this is the final season, use wr_games_played
        if pg is None and season == draft_year - 1:
            pg = gp_lookup.get((name, draft_year))

        # If still no player games, assume full season
        if pg is None:
            pg = tg

        if pg == 0:
            continue

        # Calculate adjusted dominator
        # Original: dom = (player_yards/team_yards + player_tds/team_tds) / 2 * 100
        # Adjusted: scale by (team_games / player_games) to normalize to full-season rate
        ratio = tg / pg
        adjusted_dom = old_dom * ratio

        # Cap at 100 (can't dominate more than 100%)
        adjusted_dom = min(adjusted_dom, 100.0)
        adjusted_dom = round(adjusted_dom, 1)

        seasons.at[idx, "dominator_rating"] = adjusted_dom
        seasons.at[idx, "player_games"] = pg
        seasons.at[idx, "team_games"] = tg

        if abs(adjusted_dom - old_dom) > 0.5:
            changes.append({
                "name": name,
                "college": college,
                "season": season,
                "draft_year": draft_year,
                "player_games": pg,
                "team_games": tg,
                "old_dom": old_dom,
                "new_dom": adjusted_dom,
            })

    print(f"\n  Dominator changes (>0.5 difference): {len(changes)}")
    changes_sorted = sorted(changes, key=lambda c: abs(c["new_dom"] - c["old_dom"]), reverse=True)

    print(f"\n  {'Player':<25} {'School':<18} {'Year':>4} {'PG':>3}/{' TG':>3} {'Old Dom':>7} {'New Dom':>7} {'Diff':>6}")
    print(f"  {'-' * 80}")
    for c in changes_sorted[:40]:
        diff = c["new_dom"] - c["old_dom"]
        print(f"  {c['name']:<25} {c['college']:<18} {c['season']:>4} {c['player_games']:>3}/{c['team_games']:>3} {c['old_dom']:>7.1f} {c['new_dom']:>7.1f} {diff:>+6.1f}")
    if len(changes_sorted) > 40:
        print(f"  ... and {len(changes_sorted) - 40} more")

    # Save updated seasons
    seasons.to_csv("data/wr_all_seasons.csv", index=False)
    print(f"\n  Saved data/wr_all_seasons.csv with adjusted dominator ratings")

    # Phase 4: Update peak_dominator in backtest
    print(f"\n{'=' * 80}")
    print("UPDATING PEAK DOMINATOR IN BACKTEST")
    print(f"{'=' * 80}")

    bt = pd.read_csv("data/wr_backtest_all_components.csv")
    tm = pd.read_csv("data/wr_teammate_scores.csv")

    # Build career peak lookup from adjusted seasons
    career_peaks = {}
    for name in seasons["player_name"].unique():
        s = seasons[seasons["player_name"] == name]
        dom_vals = s["dominator_rating"].dropna()
        if len(dom_vals) == 0:
            continue
        peak = dom_vals.max()
        dy = int(s["draft_year"].iloc[0])
        career_peaks[(name, dy)] = peak

    peak_changes = []
    for idx_, row in bt.iterrows():
        name = row["player_name"]
        dy = int(row["draft_year"])
        old_pd = row["peak_dominator"]

        key = (name, dy)
        if key not in career_peaks:
            continue

        correct_pd = career_peaks[key]
        if pd.isna(old_pd) and pd.isna(correct_pd):
            continue
        if pd.notna(old_pd) and pd.notna(correct_pd) and abs(old_pd - correct_pd) < 0.1:
            continue

        bt.at[idx_, "peak_dominator"] = correct_pd
        peak_changes.append({
            "name": name,
            "pick": int(row["pick"]),
            "draft_year": dy,
            "old_pd": old_pd,
            "new_pd": correct_pd,
        })

    print(f"\n  Peak dominator changes: {len(peak_changes)}")
    peak_sorted = sorted(peak_changes, key=lambda c: abs(c["new_pd"] - c["old_pd"]) if pd.notna(c["old_pd"]) and pd.notna(c["new_pd"]) else 0, reverse=True)

    print(f"\n  {'Player':<25} {'Pick':>4} {'Old PD':>7} {'New PD':>7} {'Diff':>6}")
    print(f"  {'-' * 55}")
    for c in peak_sorted[:30]:
        old = f"{c['old_pd']:.1f}" if pd.notna(c['old_pd']) else "NaN"
        diff = f"{c['new_pd'] - c['old_pd']:+.1f}" if pd.notna(c['old_pd']) and pd.notna(c['new_pd']) else "N/A"
        print(f"  {c['name']:<25} {c['pick']:>4} {old:>7} {c['new_pd']:>7.1f} {diff:>6}")
    if len(peak_sorted) > 30:
        print(f"  ... and {len(peak_sorted) - 30} more")

    # Show Waddle specifically
    print(f"\n  === WADDLE CHECK ===")
    waddle_seasons = seasons[seasons["player_name"] == "Jaylen Waddle"].sort_values("season")
    for _, r in waddle_seasons.iterrows():
        pg = r.get("player_games", "?")
        tg = r.get("team_games", "?")
        print(f"  {int(r['season'])}: dom={r['dominator_rating']:.1f}%, player_games={pg}, team_games={tg}")
    waddle_bt = bt[bt["player_name"] == "Jaylen Waddle"]
    if len(waddle_bt) > 0:
        print(f"  peak_dominator = {waddle_bt.iloc[0]['peak_dominator']:.1f}%")

    # Save
    bt.to_csv("data/wr_backtest_all_components.csv", index=False)
    print(f"\n  Saved data/wr_backtest_all_components.csv")

    # Update wr_teammate_scores.csv for consistency
    for idx_, row in tm.iterrows():
        key = (row["player_name"], int(row["draft_year"]))
        if key in career_peaks:
            tm.at[idx_, "peak_dominator"] = career_peaks[key]
    tm.to_csv("data/wr_teammate_scores.csv", index=False)
    print(f"  Saved data/wr_teammate_scores.csv")


if __name__ == "__main__":
    main()
