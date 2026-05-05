"""Backfill TE breakout_age from cfbfastR PBP for the 94 of 160 backtest TEs
and 34 of 52 2026 TEs that are currently NaN.

breakout_age = age (integer, season_year - birth_year) at the first college
season where a TE hit the 15% dominator threshold (lower than WR's 20%).

Writes back to data/te_backtest_master.csv (preserving existing breakout_age
values for TEs that already have them) and data/te_2026_prospects_final.csv.
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PBP_DIR = Path("/tmp/cfb_pbp")

sys.path.insert(0, str(ROOT / "scripts"))
from cfbfastr_fill_gaps import (  # type: ignore
    SCHOOL_MAP, PLAYER_NAME_OVERRIDES, PLAYER_TEAM_OVERRIDES,
    names_match, normalize_school,
)

TE_DOMINATOR_THRESHOLD = 15.0  # per CLAUDE.md, TE uses 15% (vs WR's 20%)

_pbp_cache = {}


def load_pbp(year):
    if year in _pbp_cache:
        return _pbp_cache[year]
    p = PBP_DIR / f"ps_{year}.parquet"
    if not p.exists():
        _pbp_cache[year] = None
        return None
    _pbp_cache[year] = pd.read_parquet(p)
    return _pbp_cache[year]


def player_season_dominator(player, team, year):
    """Returns dominator% or None if player not found / team has no data."""
    pbp = load_pbp(year)
    if pbp is None:
        return None
    team_sub = pbp[pbp["team"] == team]
    if team_sub.empty:
        return None
    recv = team_sub[team_sub["reception_player"].notna()]
    matched = recv[recv["reception_player"].apply(lambda x: names_match(x, player))]
    if matched.empty:
        return None
    py = float(matched["reception_yds"].sum())
    pt = int(matched["touchdown_player"].notna().sum())
    ty = float(team_sub["reception_yds"].sum())
    tt = int(((team_sub["reception_player"].notna()) & (team_sub["touchdown_player"].notna())).sum())
    if py == 0:
        return None
    yd_share = (py / ty) if ty > 0 else 0.0
    td_share = (pt / tt) if tt > 0 else 0.0
    return (yd_share + td_share) / 2 * 100


def find_breakout(player, expected_team, draft_year, draft_age):
    """For each season draft_year-5 .. draft_year-1, compute dominator at expected
    school. Return (breakout_age, breakout_season, peak_dominator) where breakout_age
    is age at first season hitting 15%.
    """
    if pd.isna(draft_age):
        return None, None, None
    birth_year = int(round(draft_year - draft_age))
    expected = normalize_school(expected_team)
    overridden = PLAYER_TEAM_OVERRIDES.get(player, (None, None))[0]
    if overridden:
        expected = overridden
    lookup_player = PLAYER_NAME_OVERRIDES.get(player, player)
    if expected is None:
        return None, None, None

    seasons_data = []
    for yr in range(draft_year - 5, draft_year):
        d = player_season_dominator(lookup_player, expected, yr)
        if d is not None:
            seasons_data.append((yr, d))

    if not seasons_data:
        return None, None, None

    peak = max(s[1] for s in seasons_data)
    # Breakout = first season hitting threshold
    breakout = next(((yr, d) for yr, d in seasons_data if d >= TE_DOMINATOR_THRESHOLD), None)
    if breakout is None:
        return None, None, round(peak, 1)
    breakout_year, breakout_dom = breakout
    breakout_age = breakout_year - birth_year
    return breakout_age, breakout_year, round(peak, 1)


def fill_te_backtest():
    print("=" * 80)
    print("TE BACKTEST breakout_age backfill")
    print("=" * 80)
    path = ROOT / "data" / "te_backtest_master.csv"
    df = pd.read_csv(path)
    miss_mask = df["breakout_age"].isna()
    missing = df[miss_mask]
    print(f"Missing breakout_age: {len(missing)}/{len(df)}")

    filled = 0
    for idx, r in missing.iterrows():
        ba, bs, peak = find_breakout(r["player_name"], r["college"],
                                       int(r["draft_year"]), r["draft_age"])
        if ba is not None:
            df.at[idx, "breakout_age"] = ba
            df.at[idx, "breakout_season"] = bs
            # Don't overwrite peak_dominator if already populated; only fill if we found a higher peak
            if pd.isna(r["peak_dominator"]) and peak is not None:
                df.at[idx, "peak_dominator"] = peak
            filled += 1
            print(f"  ✓ {r['player_name']} ({r['college']}, drafted {int(r['draft_year'])}): breakout at age {ba} in {bs}")
        elif peak is not None:
            # Found seasons but never hit 15% — that's valid info, leave breakout_age NaN
            print(f"  – {r['player_name']} ({r['college']}): never hit 15% (peak {peak:.1f}%)")
        else:
            print(f"  ✗ {r['player_name']} ({r['college']}): no PBP data found")

    df.to_csv(path, index=False)
    print(f"\nFilled: {filled}/{len(missing)}")


def fill_te_2026():
    print("\n" + "=" * 80)
    print("TE 2026 breakout_age backfill")
    print("=" * 80)
    path = ROOT / "data" / "te_2026_prospects_final.csv"
    df = pd.read_csv(path)
    miss_mask = df["breakout_age"].isna()
    missing = df[miss_mask]
    print(f"Missing breakout_age: {len(missing)}/{len(df)}")

    filled = 0
    for idx, r in missing.iterrows():
        ba, bs, peak = find_breakout(r["player_name"], r["college"], 2026, r["draft_age"])
        if ba is not None:
            df.at[idx, "breakout_age"] = ba
            if pd.isna(r["peak_dominator"]) and peak is not None:
                df.at[idx, "peak_dominator"] = peak
            filled += 1
            print(f"  ✓ {r['player_name']} ({r['college']}): breakout at age {ba} in {bs}")
        elif peak is not None:
            print(f"  – {r['player_name']} ({r['college']}): never hit 15% (peak {peak:.1f}%)")
        else:
            print(f"  ✗ {r['player_name']} ({r['college']}): no PBP data")

    df.to_csv(path, index=False)
    print(f"\nFilled: {filled}/{len(missing)}")


if __name__ == "__main__":
    fill_te_backtest()
    fill_te_2026()
    print("\nDone. Now run: python src/build_master_database_v5.py")
