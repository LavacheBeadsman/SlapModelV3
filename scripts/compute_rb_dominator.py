"""Compute peak dominator rating for all RBs (and fill the 4 missing 2026 WRs).

Dominator = (player_rec_yds/team_rec_yds + player_rec_tds/team_rec_tds) / 2 × 100

For each player we find their college seasons in cfbfastR PBP, compute dominator
per season, and take the peak. Writes to data/rb_dominator_scores.csv (and updates
data/wr_breakout_ages_2026.csv for the 4 missing WRs).

Data source: pre-downloaded /tmp/cfb_pbp/ps_YYYY.parquet (cfbfastR PBP).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PBP_DIR = Path("/tmp/cfb_pbp")

# Reuse normalization + matching from the prior script. Re-import.
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from cfbfastr_fill_gaps import (  # type: ignore
    SCHOOL_MAP, NICKNAME_TO_FORMAL, PLAYER_NAME_OVERRIDES, PLAYER_TEAM_OVERRIDES,
    names_match, _expand_first, normalize_school,
)


_pbp_cache = {}


def load_pbp(year):
    if year in _pbp_cache:
        return _pbp_cache[year]
    path = PBP_DIR / f"ps_{year}.parquet"
    if not path.exists():
        _pbp_cache[year] = None
        return None
    df = pd.read_parquet(path)
    _pbp_cache[year] = df
    return df


# Pre-aggregate team totals per (year, team) and cache
_team_totals_cache = {}


def team_totals(year, team):
    key = (year, team)
    if key in _team_totals_cache:
        return _team_totals_cache[key]
    pbp = load_pbp(year)
    if pbp is None:
        _team_totals_cache[key] = (0, 0)
        return 0, 0
    sub = pbp[pbp["team"] == team]
    rec_yds = sub["reception_yds"].sum()
    rec_tds = ((sub["reception_player"].notna()) & (sub["touchdown_player"].notna())).sum()
    _team_totals_cache[key] = (float(rec_yds), int(rec_tds))
    return float(rec_yds), int(rec_tds)


def player_season_dominator(player, team, year):
    """Returns (dominator_pct, rec_yds, rec_tds, team_rec_yds, team_rec_tds) or None."""
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
    ty, tt = team_totals(year, team)
    if ty == 0 and tt == 0:
        return None
    yd_share = (py / ty) if ty > 0 else 0.0
    td_share = (pt / tt) if tt > 0 else 0.0
    dom = (yd_share + td_share) / 2 * 100
    return dom, py, pt, ty, tt


def find_player_seasons(player, expected_team, draft_year):
    """Look for the player across all seasons (draft_year-5 .. draft_year-1).
    Use expected_team as primary; if not found there in a year, try cross-team
    in that year using strict name match."""
    expected = normalize_school(expected_team)
    overridden_team = PLAYER_TEAM_OVERRIDES.get(player, (None, None))[0]
    if overridden_team:
        expected = overridden_team
    lookup_player = PLAYER_NAME_OVERRIDES.get(player, player)

    results = []
    if not expected:
        return results
    seasons = list(range(draft_year - 5, draft_year))
    for yr in seasons:
        pbp = load_pbp(yr)
        if pbp is None:
            continue
        # Try expected team first
        d = player_season_dominator(lookup_player, expected, yr)
        if d is not None and d[1] > 0:
            results.append((yr, expected, *d))
            continue
        # Fallback: strict cross-team match this year
        recv = pbp[pbp["reception_player"].notna()]
        cands = recv[recv["reception_player"].apply(lambda x: _strict_match(x, lookup_player))]
        if cands.empty:
            continue
        # Pick team with most receiving yards for the player
        best_team = cands.groupby("team")["reception_yds"].sum().idxmax()
        d2 = player_season_dominator(lookup_player, best_team, yr)
        if d2 is not None and d2[1] > 0:
            results.append((yr, best_team, *d2))
    return results


def _strict_match(api_name, target):
    a = str(api_name).lower().strip().replace(".", " ").replace("'", "")
    b = str(target).lower().strip().replace(".", " ").replace("'", "")
    suf = {"jr", "ii", "iii", "iv", "v", "sr"}
    pa = [p for p in a.split() if p not in suf]
    pb = [p for p in b.split() if p not in suf]
    if not pa or not pb:
        return False
    if pa[-1] != pb[-1]:
        return False
    return _expand_first(pa[0]) == _expand_first(pb[0])


def compute_rb_dominators():
    print("=" * 80)
    print("COMPUTING RB DOMINATOR RATINGS")
    print("=" * 80)

    # Load RB backtest
    rb_bt = pd.read_csv(ROOT / "data" / "rb_backtest_with_receiving.csv")
    # Load RB 2026 prospects
    pros = pd.read_csv(ROOT / "data" / "prospects_final.csv")
    rb_p = pros[pros["position"] == "RB"].copy()

    rows = []

    print(f"\nRB backtest ({len(rb_bt)}):")
    for _, r in rb_bt.iterrows():
        results = find_player_seasons(r["player_name"], r["college"], int(r["draft_year"]))
        if results:
            best = max(results, key=lambda x: x[2])  # x[2] is dominator value
            rows.append({
                "player_name": r["player_name"], "draft_year": int(r["draft_year"]),
                "dataset": "backtest",
                "peak_dominator": round(best[2], 1),
                "peak_dominator_season": best[0],
                "peak_dominator_team": best[1],
                "rec_yds_in_peak": best[3],
                "rec_tds_in_peak": best[4],
                "team_rec_yds_in_peak": best[5],
                "team_rec_tds_in_peak": best[6],
                "n_seasons_found": len(results),
            })
        else:
            rows.append({
                "player_name": r["player_name"], "draft_year": int(r["draft_year"]),
                "dataset": "backtest",
                "peak_dominator": np.nan, "peak_dominator_season": np.nan,
                "peak_dominator_team": np.nan,
                "rec_yds_in_peak": np.nan, "rec_tds_in_peak": np.nan,
                "team_rec_yds_in_peak": np.nan, "team_rec_tds_in_peak": np.nan,
                "n_seasons_found": 0,
            })
    bt_filled = sum(1 for r in rows if not np.isnan(r["peak_dominator"]))
    print(f"  Filled: {bt_filled}/{len(rb_bt)}")

    print(f"\nRB 2026 ({len(rb_p)}):")
    for _, r in rb_p.iterrows():
        results = find_player_seasons(r["player_name"], r["school"], 2026)
        if results:
            best = max(results, key=lambda x: x[2])
            rows.append({
                "player_name": r["player_name"], "draft_year": 2026,
                "dataset": "2026_prospect",
                "peak_dominator": round(best[2], 1),
                "peak_dominator_season": best[0],
                "peak_dominator_team": best[1],
                "rec_yds_in_peak": best[3],
                "rec_tds_in_peak": best[4],
                "team_rec_yds_in_peak": best[5],
                "team_rec_tds_in_peak": best[6],
                "n_seasons_found": len(results),
            })
        else:
            rows.append({
                "player_name": r["player_name"], "draft_year": 2026,
                "dataset": "2026_prospect",
                "peak_dominator": np.nan, "peak_dominator_season": np.nan,
                "peak_dominator_team": np.nan,
                "rec_yds_in_peak": np.nan, "rec_tds_in_peak": np.nan,
                "team_rec_yds_in_peak": np.nan, "team_rec_tds_in_peak": np.nan,
                "n_seasons_found": 0,
            })
    p_filled = sum(1 for r in rows[len(rb_bt):] if not np.isnan(r["peak_dominator"]))
    print(f"  Filled: {p_filled}/{len(rb_p)}")

    out = pd.DataFrame(rows)
    out_path = ROOT / "data" / "rb_dominator_scores.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  Wrote {out_path}")

    print("\nSample top-dominator RBs (backtest):")
    bt = out[out["dataset"] == "backtest"].dropna(subset=["peak_dominator"]).nlargest(10, "peak_dominator")
    print(bt[["player_name", "peak_dominator", "peak_dominator_season", "peak_dominator_team"]].to_string(index=False))

    print("\nSample top-dominator RBs (2026):")
    p = out[out["dataset"] == "2026_prospect"].dropna(subset=["peak_dominator"]).nlargest(10, "peak_dominator")
    print(p[["player_name", "peak_dominator", "peak_dominator_season", "peak_dominator_team"]].to_string(index=False))

    return out


def fill_4_missing_wrs():
    """The 4 WR 2026 prospects without peak_dominator. Try to find via cfbfastR."""
    print("\n" + "=" * 80)
    print("FILLING MISSING WR 2026 peak_dominator")
    print("=" * 80)
    path = ROOT / "data" / "wr_breakout_ages_2026.csv"
    df = pd.read_csv(path)

    # Identify missing
    miss = df[df["peak_dominator"].isna()]
    print(f"Missing in wr_breakout_ages_2026.csv: {len(miss)}")
    targets = ["Tyren Montgomery", "Michael Wortham", "Devin Voisin", "Max Tomzcak"]

    # Also pull from prospects_final to get school
    pros = pd.read_csv(ROOT / "data" / "prospects_final.csv")
    wr_p = pros[pros["position"] == "WR"]

    updated = 0
    for name in targets:
        rows = wr_p[wr_p["player_name"] == name]
        if rows.empty:
            print(f"  ✗ {name}: not in prospects_final.csv")
            continue
        school = rows.iloc[0]["school"]
        results = find_player_seasons(name, school, 2026)
        if results:
            best = max(results, key=lambda x: x[2])
            # Update wr_breakout_ages_2026 if player exists there
            mask = df["player_name"] == name
            if mask.any():
                df.loc[mask, "peak_dominator"] = round(best[2], 1)
                if "breakout_age" in df.columns:
                    pass  # don't overwrite breakout_age — that's a separate calc
                updated += 1
                print(f"  ✓ {name} ({school}): {best[2]:.1f}% in {best[0]} at {best[1]}")
            else:
                print(f"  (would set) {name}: {best[2]:.1f}% in {best[0]} at {best[1]} — but not in wr_breakout_ages_2026.csv")
        else:
            print(f"  ✗ {name} ({school}): no PBP coverage")

    if updated > 0:
        df.to_csv(path, index=False)
        print(f"  Wrote {updated} updates to {path}")


if __name__ == "__main__":
    compute_rb_dominators()
    fill_4_missing_wrs()
    print("\nDone. Now:")
    print("  1. Update src/build_master_database_v5.py to merge data/rb_dominator_scores.csv")
    print("  2. Run python src/build_master_database_v5.py")
