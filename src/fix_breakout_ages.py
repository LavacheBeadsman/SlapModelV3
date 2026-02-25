"""
fix_breakout_ages.py — Recalculate breakout_age from games-adjusted dominator data.

TWO bugs fixed:
1. DOMINATOR BUG: breakout_age was calculated before the dominator-per-game
   adjustment (fix_dominator_games_played.py). Many players' first 20%+ season
   changed. Example: Waddle 2020 went from 11.1% (raw) to 28.9% (adjusted).

2. AGE FORMULA BUG: The old formula (draft_age - (draft_year - season)) gives
   wrong results for young players because draft_age is the floor of actual age
   at draft time. E.g., Rondale Moore (draft_age=20, born June 2000) in 2018
   season: formula gives 17, but he was actually 18.

   Fix: Use birthdate directly to calculate age at season start (September 1).
   This matches CLAUDE.md's intent of integer season ages.

Also adds a 4-game minimum per season to prevent small-sample breakout noise.

For the 181 backtest players NOT in wr_all_seasons.csv (mostly pre-2018 draft
years without CFBD per-season data), breakout_age is unchanged.
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("RECALCULATING BREAKOUT AGES (birthdate method + games-adjusted dominator)")
print("=" * 80)

MIN_GAMES = 4  # Minimum games for a season to count as a "breakout"


def age_at_season_start(birthdate_str, season_year):
    """Calculate integer age at September 1 of the given season year."""
    bd = datetime.strptime(str(birthdate_str), "%Y-%m-%d")
    # Age on September 1 of the season year
    return season_year - bd.year - ((9, 1) < (bd.month, bd.day))


# Load data
seasons = pd.read_csv("data/wr_all_seasons.csv")
bt = pd.read_csv("data/wr_backtest_all_components.csv")

print(f"\n  wr_all_seasons: {len(seasons)} rows, {seasons['player_name'].nunique()} players")
print(f"  wr_backtest: {len(bt)} players")
print(f"  Minimum games for breakout: {MIN_GAMES}")
print(f"  Age method: birthdate-based (age at September 1 of season year)")

# Recalculate breakout_age for each player in wr_all_seasons
changes = []
for name in seasons["player_name"].unique():
    ps = seasons[seasons["player_name"] == name].sort_values("season")
    dy = int(ps["draft_year"].iloc[0])

    bt_row = bt[(bt["player_name"] == name) & (bt["draft_year"] == dy)]
    if len(bt_row) == 0:
        continue

    idx = bt_row.index[0]
    old_ba = bt_row.iloc[0]["breakout_age"]
    old_pd = bt_row.iloc[0]["peak_dominator"]
    pick = int(bt_row.iloc[0]["pick"])
    birthdate = bt_row.iloc[0]["birthdate"]

    if pd.isna(birthdate):
        continue  # Can't calculate age without birthdate

    # Find first season with dominator >= 20% AND player_games >= MIN_GAMES
    new_ba = None
    breakout_season = None
    breakout_dom = None
    for _, row in ps.iterrows():
        dom = row["dominator_rating"]
        pg = row["player_games"]
        season = int(row["season"])

        if pd.notna(dom) and dom >= 20 and pd.notna(pg) and pg >= MIN_GAMES:
            new_ba = age_at_season_start(birthdate, season)
            breakout_season = season
            breakout_dom = dom
            break

    # Check if this is a change
    old_is_nan = pd.isna(old_ba)
    new_is_nan = new_ba is None

    changed = False
    if old_is_nan != new_is_nan:
        changed = True
    elif not old_is_nan and not new_is_nan and int(old_ba) != new_ba:
        changed = True

    if changed:
        # Update the backtest
        if new_ba is not None:
            bt.at[idx, "breakout_age"] = float(new_ba)
        else:
            bt.at[idx, "breakout_age"] = np.nan

        changes.append({
            "name": name,
            "pick": pick,
            "draft_year": dy,
            "old_ba": old_ba,
            "new_ba": new_ba,
            "breakout_season": breakout_season,
            "breakout_dom": breakout_dom,
            "peak_dom": old_pd,
        })

# Report
print(f"\n  Breakout age changes: {len(changes)}")

# Categorize changes
gained = [c for c in changes if pd.isna(c["old_ba"]) and c["new_ba"] is not None]
lost = [c for c in changes if pd.notna(c["old_ba"]) and c["new_ba"] is None]
shifted = [c for c in changes if pd.notna(c["old_ba"]) and c["new_ba"] is not None]

print(f"    Gained breakout (NaN -> age): {len(gained)}")
print(f"    Lost breakout (age -> NaN):   {len(lost)}")
print(f"    Age shifted:                  {len(shifted)}")

# Show all changes sorted by pick
print(f"\n  {'Player':<25} {'Pick':>4} {'DY':>4} {'Old BA':>6} {'New BA':>6} {'BO Szn':>6} {'BO Dom':>6} {'Peak Dom':>8}")
print(f"  {'-' * 80}")
for c in sorted(changes, key=lambda x: x["pick"]):
    old = f"{int(c['old_ba'])}" if pd.notna(c["old_ba"]) else "NaN"
    new = str(c["new_ba"]) if c["new_ba"] is not None else "NaN"
    bos = str(c["breakout_season"]) if c["breakout_season"] else "-"
    bod = f"{c['breakout_dom']:.1f}" if c["breakout_dom"] else "-"
    pd_str = f"{c['peak_dom']:.1f}" if pd.notna(c["peak_dom"]) else "NaN"
    print(f"  {c['name']:<25} {c['pick']:>4} {c['draft_year']:>4} {old:>6} {new:>6} {bos:>6} {bod:>6} {pd_str:>8}")

# Save
bt.to_csv("data/wr_backtest_all_components.csv", index=False)
print(f"\n  Saved data/wr_backtest_all_components.csv")
print("  Done. Run build_master_database_v5.py to recalculate scores.")
