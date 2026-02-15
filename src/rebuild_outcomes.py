"""
Rebuild backtest_outcomes_complete.csv from nflverse weekly data.

Matches ALL 339 WRs + 223 RBs + 160 TEs to nflverse game logs.
Calculates:
  - first_3yr_ppg: BEST single-season PPR PPG in first 3 NFL seasons (min 6 games)
  - career_ppg: BEST single-season PPR PPG across entire career (min 6 games)
  - seasons_over_10ppg: count of seasons averaging 10+ PPR PPG (min 6 games)
Preserves existing hit24/hit12 from backtest files (does NOT recalculate).
"""
import pandas as pd
import numpy as np
import unicodedata
import re

# ─── Load all sources ───────────────────────────────────────────────
print("Loading data files...")
wr_bt = pd.read_csv('data/wr_backtest_all_components.csv')
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
te_bt = pd.read_csv('data/te_backtest_master.csv')

draft = pd.read_parquet('data/nflverse/draft_picks.parquet')

# Load weekly stats — CSV has 1999-2024, append 2025 parquet if available
stats = pd.read_csv('data/nflverse/player_stats_all_years.csv')
max_csv_season = int(stats['season'].max())
print(f"  CSV covers seasons up to {max_csv_season}")

import os
parquet_2025 = 'data/nflverse/player_stats_2025.parquet'
if os.path.exists(parquet_2025):
    stats_2025 = pd.read_parquet(parquet_2025)
    # Only append if 2025 isn't already in the CSV
    if max_csv_season < 2025:
        stats = pd.concat([stats, stats_2025], ignore_index=True)
        print(f"  Appended 2025 parquet ({len(stats_2025)} rows) — now covers through 2025")
    else:
        print(f"  2025 already in CSV, skipping parquet append")
else:
    print(f"  WARNING: {parquet_2025} not found — 2025 draft class will have no stats")

print(f"  WR backtest: {len(wr_bt)} players")
print(f"  RB backtest: {len(rb_bt)} players")
print(f"  TE backtest: {len(te_bt)} players")
print(f"  Draft picks: {len(draft)}")
print(f"  Weekly stats: {len(stats)} rows")

# ─── Build unified player list ─────────────────────────────────────
players = []
for _, r in wr_bt.iterrows():
    players.append({
        'player_name': r['player_name'], 'position': 'WR',
        'draft_year': int(r['draft_year']), 'pick': int(r['pick']),
        'hit24': int(r['hit24']), 'hit12': int(r['hit12']),
    })
for _, r in rb_bt.iterrows():
    players.append({
        'player_name': r['player_name'], 'position': 'RB',
        'draft_year': int(r['draft_year']), 'pick': int(r['pick']),
        'hit24': int(r['hit24']), 'hit12': int(r['hit12']),
    })
for _, r in te_bt.iterrows():
    players.append({
        'player_name': r['player_name'], 'position': 'TE',
        'draft_year': int(r['draft_year']), 'pick': int(r['pick']),
        'hit24': int(r['hit24']), 'hit12': int(r['hit12']),
    })

players_df = pd.DataFrame(players)
print(f"\nTotal players to match: {len(players_df)} (WR={len(wr_bt)}, RB={len(rb_bt)}, TE={len(te_bt)})")

# ─── Name normalization ────────────────────────────────────────────
def normalize_name(name):
    """Strip accents, suffixes, punctuation, collapse whitespace, lowercase."""
    if pd.isna(name):
        return ''
    s = str(name).strip()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = s.lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', s)
    s = re.sub(r"['.()]", '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ─── Build stats-level name→player_id lookup ──────────────────────
# This catches players whose gsis_id is NaN in draft_picks but who DO have stats
print("\nBuilding stats name lookup...")
stats_names = stats[['player_id', 'player_display_name']].drop_duplicates()
stats_by_display_name = {}
stats_by_norm_name = {}
for _, s in stats_names.iterrows():
    if pd.notna(s['player_display_name']) and pd.notna(s['player_id']):
        stats_by_display_name[s['player_display_name']] = s['player_id']
        norm = normalize_name(s['player_display_name'])
        stats_by_norm_name[norm] = s['player_id']

# ─── Manual name mappings ──────────────────────────────────────────
# (backtest_name, draft_year) → nflverse display name for stats matching
MANUAL_NAME_MAP = {
    # WR
    ("Mike Thomas", 2016): "Michael Thomas",
    ("Will Fuller", 2016): "Will Fuller V",
    ("Scotty Miller", 2019): "Scott Miller",
    ("D.K. Metcalf", 2019): "DK Metcalf",
    ("Laviska Shenault", 2020): "Laviska Shenault Jr.",
    ("K.J. Hamler", 2020): "KJ Hamler",
    ("Gabriel Davis", 2020): "Gabe Davis",
    ("K.J. Osborn", 2020): "KJ Osborn",
    ("D'Wayne Eskridge", 2021): "Dee Eskridge",
    ("Demarcus Ayers", 2016): "DeMarcus Ayers",
    # RB
    ("J.K. Dobbins", 2020): "JK Dobbins",
    # TE  (none needed so far)
}

# ─── Link via draft_picks (gsis_id) then fallback to stats display name ─
print("\n=== LINKING PLAYERS TO NFLVERSE ===")

# Step 1: Build draft pick lookup by (name, year, pick)
draft_rel = draft[
    (draft['season'] >= 2015) & (draft['season'] <= 2025)
].copy()
draft_rel['norm_name'] = draft_rel['pfr_player_name'].apply(normalize_name)

# Multiple lookup strategies from draft picks
# Primary: (norm_name, year, pick) for precision
# Fallback: (norm_name, year) for when pick doesn't match exactly
draft_by_nyp = {}  # (norm_name, year, pick) → gsis_id
draft_by_ny = {}   # (norm_name, year) → gsis_id (last wins — ambiguous for dupes)
draft_by_yp = {}   # (year, pick) → gsis_id (most reliable for exact pick match)

for _, d in draft_rel.iterrows():
    gsis = d['gsis_id']
    norm = d['norm_name']
    yr = int(d['season'])
    pk = int(d['pick'])
    if pd.notna(gsis):
        draft_by_nyp[(norm, yr, pk)] = gsis
        draft_by_ny[(norm, yr)] = gsis
        draft_by_yp[(yr, pk)] = gsis

def find_player_id(player_name, draft_year, pick, position):
    """Multi-tier matching. Returns (player_id, match_tier)."""

    # Tier 0: Manual name mapping
    mapped = MANUAL_NAME_MAP.get((player_name, draft_year), player_name)

    # Tier 1: BEST — exact year + pick (unambiguous, handles name dupes)
    yp_key = (draft_year, pick)
    if yp_key in draft_by_yp:
        return draft_by_yp[yp_key], 'draft_pick_exact'

    # Tier 2: Normalized name + year + pick
    for name_variant in [mapped, player_name]:
        norm = normalize_name(name_variant)
        key = (norm, draft_year, pick)
        if key in draft_by_nyp:
            return draft_by_nyp[key], 'draft_nyp'

    # Tier 3: Normalized name + year (fallback for pick mismatches)
    for name_variant in [mapped, player_name]:
        norm = normalize_name(name_variant)
        key = (norm, draft_year)
        if key in draft_by_ny:
            return draft_by_ny[key], 'draft_ny'

    # Tier 3b: No-dots variant (D.K. → DK)
    norm_nodots = normalize_name(mapped).replace('.', '').replace(' ', '')
    for (nname, yr), gsis in draft_by_ny.items():
        if yr == draft_year and nname.replace('.', '').replace(' ', '') == norm_nodots:
            return gsis, 'draft_nodots'

    # Tier 4: Stats display name exact match
    if mapped in stats_by_display_name:
        return stats_by_display_name[mapped], 'stats_exact'
    if player_name in stats_by_display_name:
        return stats_by_display_name[player_name], 'stats_exact_orig'

    # Tier 5: Stats normalized name match
    for name_variant in [mapped, player_name]:
        norm = normalize_name(name_variant)
        if norm in stats_by_norm_name:
            return stats_by_norm_name[norm], 'stats_norm'

    # Tier 6: Draft picks by pick number, NaN gsis → try stats name
    pick_match = draft_rel[
        (draft_rel['season'] == draft_year) & (draft_rel['pick'] == pick)
    ]
    if not pick_match.empty:
        pfr_name = pick_match.iloc[0]['pfr_player_name']
        if pfr_name in stats_by_display_name:
            return stats_by_display_name[pfr_name], 'pick_then_stats'
        norm_pfr = normalize_name(pfr_name)
        if norm_pfr in stats_by_norm_name:
            return stats_by_norm_name[norm_pfr], 'pick_then_stats_norm'

    return None, 'no_match'

# Match all players
match_tiers = {}
player_ids = []
for _, p in players_df.iterrows():
    pid, tier = find_player_id(p['player_name'], p['draft_year'], p['pick'], p['position'])
    player_ids.append(pid)
    match_tiers[tier] = match_tiers.get(tier, 0) + 1

players_df['player_id'] = player_ids

matched = players_df['player_id'].notna().sum()
print(f"  Matched: {matched}/{len(players_df)}")
print(f"  Match tiers: {match_tiers}")

unmatched = players_df[players_df['player_id'].isna()]
if not unmatched.empty:
    print(f"\n  STILL UNMATCHED ({len(unmatched)}):")
    for _, p in unmatched.sort_values(['position', 'draft_year', 'pick']).iterrows():
        print(f"    {p['position']} {p['draft_year']} pick {p['pick']}: {p['player_name']}")

# ─── Calculate PPG metrics from weekly data ────────────────────────
print("\n=== CALCULATING PPG METRICS ===")

# Regular season only
stats_reg = stats[stats['season_type'] == 'REG'].copy()

# Group by player_id + season
season_stats = stats_reg.groupby(['player_id', 'season']).agg(
    games=('fantasy_points_ppr', 'count'),
    total_ppr=('fantasy_points_ppr', 'sum'),
).reset_index()
season_stats['ppg'] = season_stats['total_ppr'] / season_stats['games']

print(f"  Season-level records: {len(season_stats)}")

# For each player, calculate metrics
first_3yr_ppg = []
career_ppg = []
seasons_over_10ppg = []
nfl_games_total = []

for _, p in players_df.iterrows():
    pid = p['player_id']
    dy = p['draft_year']

    if pd.isna(pid) or pid is None:
        first_3yr_ppg.append(np.nan)
        career_ppg.append(np.nan)
        seasons_over_10ppg.append(np.nan)
        nfl_games_total.append(0)
        continue

    player_seasons = season_stats[season_stats['player_id'] == pid]

    if player_seasons.empty:
        first_3yr_ppg.append(np.nan)
        career_ppg.append(np.nan)
        seasons_over_10ppg.append(np.nan)
        nfl_games_total.append(0)
        continue

    total_games = int(player_seasons['games'].sum())
    nfl_games_total.append(total_games)

    # Min 6 games per season to qualify
    qualified = player_seasons[player_seasons['games'] >= 6]

    # First 3 NFL seasons (draft year through draft year + 2)
    first3 = qualified[
        (qualified['season'] >= dy) & (qualified['season'] <= dy + 2)
    ]
    if not first3.empty:
        first_3yr_ppg.append(round(first3['ppg'].max(), 6))
    else:
        first_3yr_ppg.append(np.nan)

    # Career best season
    if not qualified.empty:
        career_ppg.append(round(qualified['ppg'].max(), 6))
    else:
        career_ppg.append(np.nan)

    # Seasons over 10 PPG (min 6 games)
    over_10 = qualified[qualified['ppg'] >= 10.0]
    seasons_over_10ppg.append(len(over_10))

players_df['first_3yr_ppg'] = first_3yr_ppg
players_df['career_ppg'] = career_ppg
players_df['seasons_over_10ppg'] = seasons_over_10ppg
players_df['nfl_games'] = nfl_games_total

# ─── Coverage summary ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("COVERAGE SUMMARY")
print("=" * 70)
for pos in ['WR', 'RB', 'TE']:
    sub = players_df[players_df['position'] == pos]
    has_id = sub['player_id'].notna().sum()
    has_games = (sub['nfl_games'] > 0).sum()
    has_career = sub['career_ppg'].notna().sum()
    has_first3 = sub['first_3yr_ppg'].notna().sum()
    no_games = sub[sub['nfl_games'] == 0]

    print(f"\n{pos} ({len(sub)} total):")
    print(f"  Linked to nflverse:       {has_id}/{len(sub)}")
    print(f"  Have NFL game data:       {has_games}/{len(sub)}")
    print(f"  Have first_3yr_ppg:       {has_first3}/{len(sub)}")
    print(f"  Have career_ppg:          {has_career}/{len(sub)}")

    # Show players with no NFL data
    if len(no_games) > 0:
        print(f"  Players with no NFL stats ({len(no_games)}):")
        for _, r in no_games.sort_values(['draft_year', 'pick']).iterrows():
            tag = "unlinked" if pd.isna(r['player_id']) else "linked-no-stats"
            print(f"    {r['draft_year']} pick {r['pick']:>3}: {r['player_name']:<30} [{tag}]")

# ─── Devin Funchess detail ─────────────────────────────────────────
print("\n" + "=" * 70)
print("DEVIN FUNCHESS CHECK")
print("=" * 70)
funchess = players_df[players_df['player_name'] == 'Devin Funchess']
if not funchess.empty:
    f = funchess.iloc[0]
    print(f"  Name: {f['player_name']} | {f['position']} {f['draft_year']} pick {f['pick']}")
    print(f"  player_id: {f['player_id']}")
    print(f"  hit24={f['hit24']}, hit12={f['hit12']} (preserved from backtest)")
    print(f"  first_3yr_ppg: {f['first_3yr_ppg']} (best single-season PPG, first 3 years, min 8 gm)")
    print(f"  career_ppg:    {f['career_ppg']} (best single-season PPG, career, min 8 gm)")
    print(f"  seasons_over_10ppg: {f['seasons_over_10ppg']}")
    if pd.notna(f['player_id']):
        fs = season_stats[season_stats['player_id'] == f['player_id']].sort_values('season')
        print(f"  Season-by-season:")
        for _, s in fs.iterrows():
            q = "qual" if s['games'] >= 6 else "NOT-qual"
            yr_tag = " <-- first 3yr" if s['season'] <= f['draft_year'] + 2 else ""
            print(f"    {int(s['season'])}: {int(s['games']):>2} games, {s['total_ppr']:>7.1f} PPR, {s['ppg']:>6.2f} PPG [{q}]{yr_tag}")

# ─── Save output ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING")
print("=" * 70)

output = players_df[['player_name', 'position', 'draft_year', 'pick',
                      'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg',
                      'seasons_over_10ppg', 'nfl_games']].copy()
output = output.sort_values(['position', 'draft_year', 'pick']).reset_index(drop=True)
output.to_csv('data/backtest_outcomes_complete.csv', index=False)
print(f"Saved: data/backtest_outcomes_complete.csv ({len(output)} rows)")

# Final totals
print(f"\nFINAL: {len(output)} rows = {len(output[output['position']=='WR'])} WR + "
      f"{len(output[output['position']=='RB'])} RB + {len(output[output['position']=='TE'])} TE")

# ─── Show all unmatched for user review ────────────────────────────
no_stats_all = players_df[players_df['nfl_games'] == 0]
if not no_stats_all.empty:
    print(f"\n{'=' * 70}")
    print(f"ALL PLAYERS WITH NO NFL STATS ({len(no_stats_all)})")
    print(f"{'=' * 70}")
    print("(These should be players who genuinely never played or barely appeared)")
    for _, r in no_stats_all.sort_values(['position', 'draft_year', 'pick']).iterrows():
        linked = "linked" if pd.notna(r['player_id']) else "UNLINKED"
        print(f"  {r['position']} {r['draft_year']} pick {r['pick']:>3}: {r['player_name']:<30} [{linked}]")
