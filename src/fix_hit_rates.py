"""
fix_hit_rates.py — Recalculate hit24/hit12/best_rank/best_ppr from nflverse data
and apply corrections directly to backtest CSV files.

Uses the SAME data sources and matching logic as rebuild_outcomes.py:
  - data/nflverse/player_stats_all_years.csv (through 2024)
  - data/nflverse/player_stats_2025.parquet (2025 season)
  - data/nflverse/draft_picks.parquet (for gsis_id matching)

Hit definitions:
  - Rank players within (position, season) by total regular-season PPR points
  - best_rank = best (lowest) positional rank across all NFL seasons >= draft year
  - best_ppr = highest single-season total PPR points
  - hit24 = 1 if best_rank <= 24
  - hit12 = 1 if best_rank <= 12
"""

import pandas as pd
import numpy as np
import unicodedata
import re
import os

# ─── Load data ────────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

wr_bt = pd.read_csv('data/wr_backtest_all_components.csv')
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
te_bt = pd.read_csv('data/te_backtest_master.csv')

draft = pd.read_parquet('data/nflverse/draft_picks.parquet')

stats = pd.read_csv('data/nflverse/player_stats_all_years.csv')
max_csv_season = int(stats['season'].max())
print(f"  CSV covers seasons up to {max_csv_season}")

parquet_2025 = 'data/nflverse/player_stats_2025.parquet'
if os.path.exists(parquet_2025):
    stats_2025 = pd.read_parquet(parquet_2025)
    if max_csv_season < 2025:
        stats = pd.concat([stats, stats_2025], ignore_index=True)
        print(f"  Appended 2025 parquet ({len(stats_2025)} rows) — now covers through 2025")
    else:
        print(f"  2025 already in CSV, skipping parquet append")
else:
    print(f"  WARNING: {parquet_2025} not found")

print(f"  WR backtest: {len(wr_bt)} players")
print(f"  RB backtest: {len(rb_bt)} players")
print(f"  TE backtest: {len(te_bt)} players")
print(f"  Weekly stats: {len(stats)} rows")

# ─── Name normalization (same as rebuild_outcomes.py) ─────────────────────────
def normalize_name(name):
    if pd.isna(name):
        return ''
    s = str(name).strip()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = s.lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', s)
    s = re.sub(r"['.()]", '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ─── Build lookups ────────────────────────────────────────────────────────────
print("\nBuilding lookups...")

# Stats name lookups
stats_names = stats[['player_id', 'player_display_name']].drop_duplicates()
stats_by_display_name = {}
stats_by_norm_name = {}
for _, s in stats_names.iterrows():
    if pd.notna(s['player_display_name']) and pd.notna(s['player_id']):
        stats_by_display_name[s['player_display_name']] = s['player_id']
        norm = normalize_name(s['player_display_name'])
        stats_by_norm_name[norm] = s['player_id']

# Manual name mappings
MANUAL_NAME_MAP = {
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
    ("J.K. Dobbins", 2020): "JK Dobbins",
}

# Draft pick lookups
draft_rel = draft[(draft['season'] >= 2015) & (draft['season'] <= 2025)].copy()
draft_rel['norm_name'] = draft_rel['pfr_player_name'].apply(normalize_name)

draft_by_nyp = {}
draft_by_ny = {}
draft_by_yp = {}
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
    mapped = MANUAL_NAME_MAP.get((player_name, draft_year), player_name)

    # Tier 1: year + pick (most reliable)
    yp_key = (draft_year, pick)
    if yp_key in draft_by_yp:
        return draft_by_yp[yp_key], 'draft_pick_exact'

    # Tier 2: Normalized name + year + pick
    for name_variant in [mapped, player_name]:
        norm = normalize_name(name_variant)
        key = (norm, draft_year, pick)
        if key in draft_by_nyp:
            return draft_by_nyp[key], 'draft_nyp'

    # Tier 3: Normalized name + year
    for name_variant in [mapped, player_name]:
        norm = normalize_name(name_variant)
        key = (norm, draft_year)
        if key in draft_by_ny:
            return draft_by_ny[key], 'draft_ny'

    # Tier 3b: No-dots variant
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

# ─── Calculate season totals and rankings ─────────────────────────────────────
print("\n" + "=" * 70)
print("CALCULATING SEASON TOTALS AND RANKINGS")
print("=" * 70)

stats_reg = stats[stats['season_type'] == 'REG'].copy()

# Need position from stats for ranking
season_totals = stats_reg.groupby(['player_id', 'position', 'season']).agg(
    total_ppr=('fantasy_points_ppr', 'sum'),
    games=('fantasy_points_ppr', 'count'),
).reset_index()

# Rank within position+season by total PPR (rank 1 = best)
season_totals['pos_rank'] = (
    season_totals.groupby(['position', 'season'])['total_ppr']
    .rank(ascending=False, method='min')
)

seasons_covered = sorted(season_totals['season'].unique())
print(f"  Seasons covered: {min(seasons_covered)}-{max(seasons_covered)}")
print(f"  Total player-seasons: {len(season_totals)}")

# ─── Process each position ────────────────────────────────────────────────────
def process_position(bt_df, position, rank_col='best_rank'):
    """Recalculate hit rates for one position. Returns changes list."""
    print(f"\n{'=' * 70}")
    print(f"PROCESSING {position}")
    print(f"{'=' * 70}")

    changes = []
    match_count = 0
    no_match = 0

    for idx, row in bt_df.iterrows():
        name = row['player_name']
        dy = int(row['draft_year'])
        pk = int(row['pick'])
        old_hit24 = int(row['hit24'])
        old_hit12 = int(row['hit12'])
        old_rank = float(row.get(rank_col, 999))
        old_ppr = float(row.get('best_ppr', 0))

        pid, tier = find_player_id(name, dy, pk, position)

        if pid is None:
            no_match += 1
            continue

        match_count += 1

        # Find all seasons for this player at this EXACT position
        # Do NOT fall back to other positions — a CB ranked #1 among CBs
        # is meaningless for WR hit rates, and a TE ranked #2 among TEs
        # shouldn't count as a WR hit (e.g., Darren Waller drafted as WR)
        player_seasons = season_totals[
            (season_totals['player_id'] == pid)
            & (season_totals['season'] >= dy)
            & (season_totals['position'] == position)
        ]

        if player_seasons.empty:
            new_rank = 999.0
            new_ppr = 0.0
        else:
            new_rank = float(player_seasons['pos_rank'].min())
            new_ppr = round(float(player_seasons['total_ppr'].max()), 1)

        new_hit24 = 1 if new_rank <= 24 else 0
        new_hit12 = 1 if new_rank <= 12 else 0

        # Check for any changes
        rank_changed = abs(new_rank - old_rank) > 0.5
        ppr_changed = abs(new_ppr - old_ppr) > 0.5
        h24_changed = new_hit24 != old_hit24
        h12_changed = new_hit12 != old_hit12

        if rank_changed or ppr_changed or h24_changed or h12_changed:
            changes.append({
                'idx': idx,
                'name': name,
                'draft_year': dy,
                'pick': pk,
                'old_rank': old_rank,
                'new_rank': new_rank,
                'old_ppr': old_ppr,
                'new_ppr': new_ppr,
                'old_hit24': old_hit24,
                'new_hit24': new_hit24,
                'old_hit12': old_hit12,
                'new_hit12': new_hit12,
                'tier': tier,
            })

    print(f"  Matched: {match_count}/{len(bt_df)}, Unmatched: {no_match}")
    print(f"  Changes: {len(changes)}")

    h24_gains = sum(1 for c in changes if c['new_hit24'] > c['old_hit24'])
    h24_losses = sum(1 for c in changes if c['new_hit24'] < c['old_hit24'])
    h12_gains = sum(1 for c in changes if c['new_hit12'] > c['old_hit12'])
    h12_losses = sum(1 for c in changes if c['new_hit12'] < c['old_hit12'])
    print(f"  hit24: +{h24_gains} / -{h24_losses}")
    print(f"  hit12: +{h12_gains} / -{h12_losses}")

    return changes

# ─── Process WR ──────────────────────────────────────────────────────────────
wr_changes = process_position(wr_bt, 'WR', rank_col='best_rank')

print(f"\n  WR CHANGES (sorted by pick):")
for c in sorted(wr_changes, key=lambda x: x['pick']):
    h24_flag = f"  hit24: {c['old_hit24']}→{c['new_hit24']}" if c['old_hit24'] != c['new_hit24'] else ""
    h12_flag = f"  hit12: {c['old_hit12']}→{c['new_hit12']}" if c['old_hit12'] != c['new_hit12'] else ""
    print(f"    {c['name']:25s} ({c['draft_year']} pick {c['pick']:>3d}) "
          f"rank: {c['old_rank']:>5.0f}→{c['new_rank']:>5.0f}  "
          f"ppr: {c['old_ppr']:>6.1f}→{c['new_ppr']:>6.1f}"
          f"{h24_flag}{h12_flag}")

# ─── Process RB ──────────────────────────────────────────────────────────────
rb_changes = process_position(rb_bt, 'RB', rank_col='season_rank')

print(f"\n  RB CHANGES (sorted by pick):")
for c in sorted(rb_changes, key=lambda x: x['pick']):
    h24_flag = f"  hit24: {c['old_hit24']}→{c['new_hit24']}" if c['old_hit24'] != c['new_hit24'] else ""
    h12_flag = f"  hit12: {c['old_hit12']}→{c['new_hit12']}" if c['old_hit12'] != c['new_hit12'] else ""
    print(f"    {c['name']:25s} ({c['draft_year']} pick {c['pick']:>3d}) "
          f"rank: {c['old_rank']:>5.0f}→{c['new_rank']:>5.0f}  "
          f"ppr: {c['old_ppr']:>6.1f}→{c['new_ppr']:>6.1f}"
          f"{h24_flag}{h12_flag}")

# ─── Process TE (just verify, expect 0 changes) ─────────────────────────────
te_changes = process_position(te_bt, 'TE', rank_col='best_rank')

# ─── Apply changes to WR backtest ────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("APPLYING CHANGES TO BACKTEST FILES")
print(f"{'=' * 70}")

print(f"\n  Applying {len(wr_changes)} WR changes...")
for c in wr_changes:
    idx = c['idx']
    wr_bt.at[idx, 'best_rank'] = c['new_rank']
    wr_bt.at[idx, 'best_ppr'] = c['new_ppr']
    wr_bt.at[idx, 'hit24'] = c['new_hit24']
    wr_bt.at[idx, 'hit12'] = c['new_hit12']

# Verify WR hit counts
wr_h24 = wr_bt['hit24'].sum()
wr_h12 = wr_bt['hit12'].sum()
print(f"  WR hit24={wr_h24}, hit12={wr_h12}")

# Save WR
wr_bt.to_csv('data/wr_backtest_all_components.csv', index=False)
print(f"  Saved data/wr_backtest_all_components.csv")

# ─── Apply changes to RB backtest ────────────────────────────────────────────
print(f"\n  Applying {len(rb_changes)} RB changes...")
for c in rb_changes:
    idx = c['idx']
    rb_bt.at[idx, 'season_rank'] = c['new_rank']
    rb_bt.at[idx, 'best_ppr'] = c['new_ppr']
    rb_bt.at[idx, 'hit24'] = c['new_hit24']
    rb_bt.at[idx, 'hit12'] = c['new_hit12']

# Verify RB hit counts
rb_h24 = rb_bt['hit24'].sum()
rb_h12 = rb_bt['hit12'].sum()
print(f"  RB hit24={rb_h24}, hit12={rb_h12}")

# Save RB
rb_bt.to_csv('data/rb_backtest_with_receiving.csv', index=False)
print(f"  Saved data/rb_backtest_with_receiving.csv")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"  WR: {len(wr_changes)} changes applied ({sum(1 for c in wr_changes if c['old_hit24'] != c['new_hit24'])} hit24 flips)")
print(f"  RB: {len(rb_changes)} changes applied ({sum(1 for c in rb_changes if c['old_hit24'] != c['new_hit24'])} hit24 flips)")
print(f"  TE: {len(te_changes)} changes (expected 0)")
print(f"\n  DONE. Next steps:")
print(f"    1. python src/rebuild_outcomes.py       (propagate new hit24/hit12)")
print(f"    2. python src/build_master_database_v5.py  (rebuild master)")
print(f"    3. python src/full_validation_8gm.py       (re-validate)")
