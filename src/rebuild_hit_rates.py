"""
rebuild_hit_rates.py — Rebuild backtest_hit_rates.csv from scratch using NFLVerse data.

For every WR and RB drafted in rounds 1-5 from 2015-2025, calculates:
  - seasons_played: number of NFL seasons since draft (current_year - draft_year)
  - best_rank: best fantasy finish (rank at position) in any season
  - best_ppr: best single-season PPR points total
  - hit24: 1 if player finished top-24 at position in any season, else 0
  - hit12: 1 if player finished top-12 at position in any season, else 0

Data sources:
  - Draft picks: data/nflverse/draft_picks.parquet (NFLVerse)
  - NFL stats 2015-2024: /tmp/player_stats_all.csv (NFLVerse combined player_stats)
  - NFL stats 2025: data/nflverse/player_season_stats_2025.csv (derived from PBP)

Output: data/backtest_hit_rates_rebuilt.csv
"""

import pandas as pd
import numpy as np
import os

# ── Configuration ──────────────────────────────────────────────────────────────
FIRST_DRAFT_YEAR = 2015
LAST_DRAFT_YEAR = 2025
CURRENT_YEAR = 2025  # latest completed NFL season
MAX_ROUND = 5
POSITIONS = ["WR", "RB"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
NFLVERSE_DIR = os.path.join(DATA_DIR, "nflverse")

# ── 1. Load draft picks ───────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading draft picks")
print("=" * 70)

draft = pd.read_parquet(os.path.join(NFLVERSE_DIR, "draft_picks.parquet"))
draft = draft[
    (draft["position"].isin(POSITIONS))
    & (draft["round"] <= MAX_ROUND)
    & (draft["season"] >= FIRST_DRAFT_YEAR)
    & (draft["season"] <= LAST_DRAFT_YEAR)
].copy()

# Drop players with no gsis_id (can't match to NFL stats)
no_gsis = draft[draft["gsis_id"].isna() | (draft["gsis_id"] == "")]
if len(no_gsis) > 0:
    print(f"  WARNING: {len(no_gsis)} players have no gsis_id and will be excluded:")
    for _, r in no_gsis.iterrows():
        print(f"    {r['pfr_player_name']} ({r['position']}, {r['season']} pick {r['pick']})")
    draft = draft[draft["gsis_id"].notna() & (draft["gsis_id"] != "")]

draft = draft.rename(columns={"season": "draft_year", "gsis_id": "player_id"})
print(f"  Loaded {len(draft)} WR/RB drafted in rounds 1-{MAX_ROUND}, {FIRST_DRAFT_YEAR}-{LAST_DRAFT_YEAR}")
print(f"  Per year:\n{draft.groupby('draft_year').size().to_string()}")

# ── 2. Load NFL fantasy stats (2015-2025) ─────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Loading NFL fantasy stats")
print("=" * 70)

# 2a: Combined player_stats (covers 1999-2024, weekly)
combined_path = "/tmp/player_stats_all.csv"
if not os.path.exists(combined_path):
    raise FileNotFoundError(
        f"Missing {combined_path} — download player_stats.csv from "
        "https://github.com/nflverse/nflverse-data/releases/tag/player_stats"
    )

print(f"  Loading combined player_stats from {combined_path}...")
cols_needed = [
    "player_id", "player_display_name", "position", "season",
    "season_type", "fantasy_points_ppr",
]
stats_all = pd.read_csv(combined_path, low_memory=False, usecols=cols_needed)

# Filter to regular season, 2015+, WR/RB only
stats_all = stats_all[
    (stats_all["season_type"] == "REG")
    & (stats_all["season"] >= FIRST_DRAFT_YEAR)
    & (stats_all["position"].isin(POSITIONS))
].copy()

years_in_combined = sorted(stats_all["season"].unique())
print(f"  Combined file covers seasons: {years_in_combined}")

# 2b: 2025 season stats (derived from PBP)
stats_2025_path = os.path.join(NFLVERSE_DIR, "player_season_stats_2025.csv")
if os.path.exists(stats_2025_path):
    print(f"  Loading 2025 season stats from {stats_2025_path}...")
    s25 = pd.read_csv(stats_2025_path)
    s25 = s25[s25["position"].isin(POSITIONS)].copy()
    s25["season_type"] = "REG"
    # Rename to match combined format
    s25 = s25.rename(columns={"player_display_name": "player_display_name"})
    print(f"  2025 stats: {len(s25)} WR/RB season rows")
else:
    print(f"  WARNING: {stats_2025_path} not found — 2025 will be missing!")
    s25 = pd.DataFrame()

# ── 3. Aggregate to season totals ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Aggregating to season totals")
print("=" * 70)

# 3a: Aggregate weekly stats (2015-2024) to season totals
season_totals_weekly = (
    stats_all.groupby(["player_id", "player_display_name", "position", "season"])
    .agg(fantasy_points_ppr=("fantasy_points_ppr", "sum"))
    .reset_index()
)
print(f"  Season totals from weekly data: {len(season_totals_weekly)} player-seasons")

# 3b: 2025 is already season-level
if len(s25) > 0:
    season_2025 = s25[["player_id", "player_display_name", "position", "season", "fantasy_points_ppr"]].copy()
    # Only include 2025 if not already in the combined file
    if 2025 not in years_in_combined:
        season_totals = pd.concat([season_totals_weekly, season_2025], ignore_index=True)
        print(f"  Added 2025 data: {len(season_2025)} player-seasons")
    else:
        season_totals = season_totals_weekly
        print(f"  2025 already in combined file, skipping separate 2025 file")
else:
    season_totals = season_totals_weekly

# Verify year coverage
all_seasons = sorted(season_totals["season"].unique())
print(f"  Final season coverage: {all_seasons}")
expected = set(range(FIRST_DRAFT_YEAR, CURRENT_YEAR + 1))
missing = expected - set(all_seasons)
if missing:
    print(f"  *** MISSING NFL SEASONS: {sorted(missing)} ***")
else:
    print(f"  All seasons {FIRST_DRAFT_YEAR}-{CURRENT_YEAR} present.")

# ── 4. Rank players at each position per season ──────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Ranking players per position per season")
print("=" * 70)

# Rank within position and season (rank 1 = most PPR points)
season_totals["pos_rank"] = (
    season_totals.groupby(["position", "season"])["fantasy_points_ppr"]
    .rank(ascending=False, method="min")
)

print(f"  Total player-seasons ranked: {len(season_totals)}")
for pos in POSITIONS:
    n = len(season_totals[season_totals["position"] == pos])
    print(f"    {pos}: {n} player-seasons")

# ── 5. For each drafted player, find best career stats ────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Matching drafted players to NFL outcomes")
print("=" * 70)

results = []

for _, dp_row in draft.iterrows():
    pid = dp_row["player_id"]
    name = dp_row["pfr_player_name"]
    pos = dp_row["position"]
    draft_year = dp_row["draft_year"]
    pick = dp_row["pick"]

    # Find all NFL seasons for this player AFTER being drafted
    player_seasons = season_totals[
        (season_totals["player_id"] == pid)
        & (season_totals["season"] >= draft_year)
    ]

    seasons_played = CURRENT_YEAR - draft_year

    if len(player_seasons) == 0:
        # Player was drafted but never appeared in NFL stats
        results.append({
            "player_name": name,
            "position": pos,
            "draft_year": draft_year,
            "pick": pick,
            "seasons_played": seasons_played,
            "best_rank": 999,
            "best_ppr": 0.0,
            "hit24": 0,
            "hit12": 0,
        })
    else:
        best_rank = player_seasons["pos_rank"].min()
        best_ppr = player_seasons["fantasy_points_ppr"].max()

        results.append({
            "player_name": name,
            "position": pos,
            "draft_year": draft_year,
            "pick": pick,
            "seasons_played": seasons_played,
            "best_rank": best_rank,
            "best_ppr": best_ppr,
            "hit24": 1 if best_rank <= 24 else 0,
            "hit12": 1 if best_rank <= 12 else 0,
        })

df = pd.DataFrame(results)

# Check for unmatched players
unmatched = df[df["best_rank"] == 999]
print(f"  Matched: {len(df) - len(unmatched)} / {len(df)} players")
if len(unmatched) > 0:
    print(f"  Unmatched ({len(unmatched)} players with no NFL stats):")
    for _, r in unmatched.iterrows():
        print(f"    {r['player_name']} ({r['position']}, {r['draft_year']} pick {int(r['pick'])})")

# ── 6. Save output ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Saving output")
print("=" * 70)

output_path = os.path.join(DATA_DIR, "backtest_hit_rates_rebuilt.csv")
df.to_csv(output_path, index=False)
print(f"  Saved to {output_path}")
print(f"  Total players: {len(df)}")
print(f"  WR: {len(df[df['position'] == 'WR'])}, RB: {len(df[df['position'] == 'RB'])}")

# ── 7. Summary diagnostics ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: Summary diagnostics")
print("=" * 70)

# Per draft year
print("\n  Draft Year | Players | WR | RB | Hit24 Rate | Hit12 Rate | Avg Best PPR")
print("  " + "-" * 75)
for yr in range(FIRST_DRAFT_YEAR, LAST_DRAFT_YEAR + 1):
    sub = df[df["draft_year"] == yr]
    n = len(sub)
    nwr = len(sub[sub["position"] == "WR"])
    nrb = len(sub[sub["position"] == "RB"])
    h24 = sub["hit24"].mean() * 100 if n > 0 else 0
    h12 = sub["hit12"].mean() * 100 if n > 0 else 0
    avg_ppr = sub[sub["best_ppr"] > 0]["best_ppr"].mean() if (sub["best_ppr"] > 0).any() else 0
    flag = ""
    if h24 == 0 and yr <= CURRENT_YEAR - 1:
        flag = " *** SUSPICIOUS: 0% hit rate ***"
    elif h24 < 15 and yr <= CURRENT_YEAR - 2:
        flag = " ** Low hit rate **"
    print(f"  {yr:>10} | {n:>7} | {nwr:>2} | {nrb:>2} | {h24:>9.1f}% | {h12:>9.1f}% | {avg_ppr:>12.1f}{flag}")

# Overall
print(f"\n  Overall: {len(df)} players, "
      f"Hit24={df['hit24'].mean()*100:.1f}%, "
      f"Hit12={df['hit12'].mean()*100:.1f}%")

# Compare to old file
old_path = os.path.join(DATA_DIR, "backtest_hit_rates.csv")
if os.path.exists(old_path):
    old = pd.read_csv(old_path)
    print(f"\n  Comparison to old backtest_hit_rates.csv:")
    print(f"    Old: {len(old)} players, years {sorted(old['draft_year'].unique())}")
    print(f"    New: {len(df)} players, years {sorted(df['draft_year'].unique())}")
    print(f"    Δ: +{len(df) - len(old)} players, +{len(df['draft_year'].unique()) - len(old['draft_year'].unique())} draft years")
