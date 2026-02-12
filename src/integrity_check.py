"""
COMPREHENSIVE INTEGRITY CHECK — All 6 Components, 339 WRs
===========================================================

1. Missing data scan
2. Name consistency check across all files
3. Duplicate check
4. Value sanity check
5. Join test (dry run merge)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LOAD ALL FILES
# ============================================================================

print("=" * 120)
print("LOADING ALL DATA FILES")
print("=" * 120)

files = {}

files['backtest'] = pd.read_csv('data/wr_backtest_expanded_final.csv')
files['components'] = pd.read_csv('data/wr_backtest_all_components.csv')
files['outcomes'] = pd.read_csv('data/backtest_outcomes_complete.csv')
files['pff'] = pd.read_csv('data/wr_pff_all_2016_2025.csv')
files['teammate'] = pd.read_csv('data/wr_teammate_scores.csv')
files['games'] = pd.read_csv('data/wr_games_played.csv')
files['declare'] = pd.read_csv('data/wr_eval_with_declare.csv')
files['ras_big'] = pd.read_csv('data/WRRas201502025.csv', encoding='utf-8-sig')
files['ras_merged'] = pd.read_csv('data/wr_ras_merged.csv')

for name, df in files.items():
    name_col = 'player_name' if 'player_name' in df.columns else ('Name' if 'Name' in df.columns else '???')
    year_col = 'draft_year' if 'draft_year' in df.columns else ('Year' if 'Year' in df.columns else '???')
    print(f"  {name:<15} {len(df):>5} rows  name_col={name_col}  year_col={year_col}")


# ============================================================================
# CHECK 1: MISSING DATA SCAN
# ============================================================================

print("\n\n" + "=" * 120)
print("CHECK 1: MISSING DATA SCAN — All 6 Components")
print("=" * 120)

comp = files['components']
teammate = files['teammate']
games = files['games']

print(f"\nBase file (wr_backtest_all_components.csv): {len(comp)} WRs")

# The 6 components and their required columns:
component_cols = {
    'Draft Capital': ['pick'],
    'Breakout Age': ['breakout_age', 'peak_dominator'],
    'RAS (Athletic)': ['RAS'],
    'Early Declare': ['declare_status', 'early_declare'],
    'Rushing Production': ['rush_yards'],
}

# Check components file
print(f"\n{'Component':<25} {'Column':<20} {'Non-null':>10} {'Null':>6} {'Coverage':>10}")
print("-" * 75)

for component, cols in component_cols.items():
    for col in cols:
        if col in comp.columns:
            n_valid = comp[col].notna().sum()
            n_null = comp[col].isna().sum()
            print(f"{component:<25} {col:<20} {n_valid:>10} {n_null:>6} {n_valid/339*100:>9.1f}%")
        else:
            print(f"{component:<25} {col:<20} {'COLUMN MISSING':>10}")

# Teammate Score is in a separate file
print(f"\n{'Teammate Score':<25} {'avg_teammate_dc':<20}", end="")
if 'avg_teammate_dc' in teammate.columns:
    n_valid = teammate['avg_teammate_dc'].notna().sum()
    print(f" {n_valid:>10} {339-n_valid:>6} {n_valid/339*100:>9.1f}%")
else:
    print(f" {'COLUMN MISSING':>10}")

# Games Played
print(f"{'Games Played':<25} {'games_played':<20}", end="")
if 'games_played' in games.columns:
    n_valid = games['games_played'].notna().sum()
    print(f" {n_valid:>10} {339-n_valid:>6} {n_valid/339*100:>9.1f}%")

# Show EVERY player with ANY null in the 6 components
print(f"\n\n--- Players with ANY missing component data ---\n")

# Build a merged view
merged_check = comp[['player_name', 'draft_year', 'pick', 'round', 'college',
                      'RAS', 'breakout_age', 'peak_dominator',
                      'declare_status', 'early_declare',
                      'rush_yards', 'rush_attempts', 'rush_source']].copy()

# Add teammate score
teammate_lookup = dict(zip(
    zip(teammate['player_name'], teammate['draft_year']),
    teammate['avg_teammate_dc']
))
merged_check['teammate_score'] = merged_check.apply(
    lambda x: teammate_lookup.get((x['player_name'], x['draft_year'])), axis=1
)

# Add games played
games_lookup = dict(zip(
    zip(games['player_name'], games['draft_year']),
    games['games_played']
))
merged_check['games_played'] = merged_check.apply(
    lambda x: games_lookup.get((x['player_name'], x['draft_year'])), axis=1
)

# Check for any nulls in critical columns
critical_cols = ['pick', 'breakout_age', 'peak_dominator', 'RAS',
                 'declare_status', 'rush_yards', 'teammate_score', 'games_played']

any_missing = merged_check[merged_check[critical_cols].isna().any(axis=1)]

if len(any_missing) == 0:
    print("  NONE — all 339 WRs have complete data for all 6 components!")
else:
    print(f"  {len(any_missing)} players with missing data:\n")
    print(f"  {'Player':<28} {'Year':>5} {'Rd':>3} {'Pick':>5} {'Missing Columns'}")
    print("  " + "-" * 100)
    for _, row in any_missing.iterrows():
        missing_cols = [c for c in critical_cols if pd.isna(row[c])]
        print(f"  {row['player_name']:<28} {int(row['draft_year']):>5} {int(row['round']):>3} "
              f"{int(row['pick']):>5} {', '.join(missing_cols)}")


# ============================================================================
# CHECK 2: NAME CONSISTENCY ACROSS ALL FILES
# ============================================================================

print("\n\n" + "=" * 120)
print("CHECK 2: NAME CONSISTENCY ACROSS FILES")
print("=" * 120)

# Master list: all 339 WRs from the backtest
master = set(zip(comp['player_name'], comp['draft_year'].astype(int)))

# Check each file for name matches
file_checks = {
    'backtest_expanded_final': (files['backtest'], 'player_name', 'draft_year'),
    'backtest_outcomes_complete': (files['outcomes'], 'player_name', 'draft_year'),
    'wr_pff_all_2016_2025': (files['pff'], 'player_name', 'draft_year'),
    'wr_teammate_scores': (files['teammate'], 'player_name', 'draft_year'),
    'wr_games_played': (files['games'], 'player_name', 'draft_year'),
    'wr_eval_with_declare': (files['declare'], 'player_name', 'draft_year'),
}

print(f"\nMaster list: {len(master)} unique (name, year) pairs\n")

for file_name, (df, name_col, year_col) in file_checks.items():
    # Get WR-only entries
    if 'position' in df.columns:
        df_wr = df[df['position'] == 'WR'].copy()
    else:
        df_wr = df.copy()

    file_keys = set(zip(df_wr[name_col], df_wr[year_col].astype(int)))

    in_master_not_file = master - file_keys
    in_file_not_master = file_keys - master

    print(f"  {file_name}:")
    print(f"    WR entries: {len(df_wr)}")
    print(f"    In master but NOT in file: {len(in_master_not_file)}")
    print(f"    In file but NOT in master: {len(in_file_not_master)}")

    if len(in_master_not_file) > 0 and len(in_master_not_file) <= 20:
        for name, year in sorted(in_master_not_file, key=lambda x: (x[1], x[0])):
            # Check for close name matches in file
            close = df_wr[df_wr[year_col] == year]
            close_names = close[name_col].tolist()
            possible = [n for n in close_names if name.split()[-1] in n or n.split()[-1] in name]
            if possible:
                print(f"      MISMATCH: '{name}' ({year}) → possible match: {possible}")
            else:
                print(f"      MISSING:  '{name}' ({year})")
    elif len(in_master_not_file) > 20:
        # Show first few + count by year
        by_year = {}
        for name, year in in_master_not_file:
            by_year.setdefault(year, []).append(name)
        for year in sorted(by_year):
            print(f"      {year}: {len(by_year[year])} missing ({', '.join(sorted(by_year[year])[:3])}...)")

    if len(in_file_not_master) > 0 and len(in_file_not_master) <= 10:
        for name, year in sorted(in_file_not_master, key=lambda x: (x[1], x[0])):
            print(f"      EXTRA in file: '{name}' ({year})")
    elif len(in_file_not_master) > 10:
        print(f"      {len(in_file_not_master)} extra entries (non-WR or different draft classes)")
    print()


# ============================================================================
# CHECK 3: DUPLICATE CHECK
# ============================================================================

print("\n" + "=" * 120)
print("CHECK 3: DUPLICATE CHECK")
print("=" * 120)

dup_files = {
    'components': (comp, 'player_name', 'draft_year'),
    'teammate': (teammate, 'player_name', 'draft_year'),
    'games': (games, 'player_name', 'draft_year'),
    'declare': (files['declare'], 'player_name', 'draft_year'),
    'pff': (files['pff'], 'player_name', 'draft_year'),
    'backtest': (files['backtest'], 'player_name', 'draft_year'),
    'outcomes': (files['outcomes'], 'player_name', 'draft_year'),
}

print()
for file_name, (df, nc, yc) in dup_files.items():
    dupes = df.groupby([nc, yc]).size()
    dupes = dupes[dupes > 1]
    if len(dupes) > 0:
        print(f"  {file_name}: {len(dupes)} DUPLICATES FOUND!")
        for (name, year), count in dupes.items():
            rows = df[(df[nc] == name) & (df[yc] == year)]
            print(f"    '{name}' ({int(year)}): {count} entries")
            if 'pick' in df.columns:
                picks = rows['pick'].tolist()
                print(f"      Picks: {picks}")
            if 'college' in df.columns:
                colleges = rows['college'].tolist()
                if len(set(str(c) for c in colleges)) > 1:
                    print(f"      DIFFERENT COLLEGES: {colleges}")
    else:
        print(f"  {file_name}: No duplicates")

# Cross-file consistency: check pick/round/college match
print(f"\n\n--- Cross-file pick/round/college consistency ---\n")
base = comp[['player_name', 'draft_year', 'pick', 'round', 'college']].copy()
for file_name, (df, nc, yc) in [('teammate', (teammate, 'player_name', 'draft_year')),
                                  ('games', (games, 'player_name', 'draft_year'))]:
    if 'pick' in df.columns and 'round' in df.columns:
        merged = base.merge(df[[nc, yc, 'pick', 'round']],
                           on=[nc, yc], how='inner', suffixes=('_base', f'_{file_name}'))
        pick_mismatch = merged[merged['pick_base'] != merged[f'pick_{file_name}']]
        if len(pick_mismatch) > 0:
            print(f"  {file_name}: {len(pick_mismatch)} PICK MISMATCHES!")
            for _, row in pick_mismatch.iterrows():
                print(f"    {row['player_name']} ({int(row['draft_year'])}): "
                      f"base pick={int(row['pick_base'])}, {file_name} pick={int(row[f'pick_{file_name}'])}")
        else:
            print(f"  {file_name}: All picks match")


# ============================================================================
# CHECK 4: VALUE SANITY CHECK
# ============================================================================

print("\n\n" + "=" * 120)
print("CHECK 4: VALUE SANITY CHECK — All Components")
print("=" * 120)

# Component columns with expected ranges
sanity_checks = [
    ('pick', comp, 1, 262, 'Draft pick should be 1-262'),
    ('round', comp, 1, 7, 'Round should be 1-7'),
    ('breakout_age', comp, 17, 24, 'Breakout age should be 17-24'),
    ('peak_dominator', comp, 0, 100, 'Peak dominator should be 0-100'),
    ('RAS', comp, 0, 10.0, 'RAS should be 0-10'),
    ('early_declare', comp, 0, 1, 'Binary 0/1'),
    ('rush_yards', comp, -50, 500, 'Rush yards for WRs typically -50 to 500'),
    ('rush_attempts', comp, 0, 100, 'Rush attempts for WRs typically 0-100'),
    ('avg_teammate_dc', teammate, 0, 100, 'Teammate DC should be 0-100'),
    ('games_played', games, 1, 60, 'Games played should be 1-60'),
]

print(f"\n{'Column':<20} {'File':<15} {'N':>5} {'Min':>8} {'Max':>8} {'Mean':>8} {'StdDev':>8} {'Nulls':>6} {'Outliers':<30}")
print("-" * 120)

outlier_details = []

for col, df, expected_min, expected_max, note in sanity_checks:
    if col not in df.columns:
        print(f"{col:<20} {'???':<15} COLUMN NOT FOUND")
        continue

    vals = df[col].dropna()
    n = len(vals)
    nulls = df[col].isna().sum()

    if n == 0:
        print(f"{col:<20} {'components':<15} {n:>5} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {nulls:>6} All null!")
        continue

    vmin = vals.min()
    vmax = vals.max()
    vmean = vals.mean()
    vstd = vals.std()

    # Find outliers: outside expected range OR > 3 std from mean
    below = df[df[col] < expected_min] if expected_min is not None else pd.DataFrame()
    above = df[df[col] > expected_max] if expected_max is not None else pd.DataFrame()
    n_outliers = len(below) + len(above)

    # Also check 3-sigma outliers
    if vstd > 0:
        sigma3_low = vmean - 3 * vstd
        sigma3_high = vmean + 3 * vstd
        sigma_outliers = df[(df[col] < sigma3_low) | (df[col] > sigma3_high)]
    else:
        sigma_outliers = pd.DataFrame()

    outlier_str = ""
    if n_outliers > 0:
        outlier_str = f"{n_outliers} out of range"
    elif len(sigma_outliers) > 0:
        outlier_str = f"{len(sigma_outliers)} > 3σ"
    else:
        outlier_str = "OK"

    print(f"{col:<20} {'components':<15} {n:>5} {vmin:>8.1f} {vmax:>8.1f} {vmean:>8.1f} {vstd:>8.1f} {nulls:>6} {outlier_str:<30}")

    # Collect outlier details
    if n_outliers > 0:
        for _, row in pd.concat([below, above]).iterrows():
            name = row.get('player_name', 'unknown')
            year = row.get('draft_year', 0)
            val = row[col]
            outlier_details.append(f"    {col}: {name} ({int(year)}) = {val}  [{note}]")

if outlier_details:
    print(f"\n--- Outlier Details ---")
    for detail in outlier_details:
        print(detail)
else:
    print(f"\n  No out-of-range values found.")

# Special checks
print(f"\n--- Special value checks ---")

# Negative rushing yards
neg_rush = comp[comp['rush_yards'] < 0]
if len(neg_rush) > 0:
    print(f"\n  Negative rushing yards ({len(neg_rush)} players):")
    for _, r in neg_rush.iterrows():
        print(f"    {r['player_name']:<28} {int(r['draft_year'])} {r['rush_yards']:>6.0f} yds  "
              f"source={r['rush_source']}")

# RAS values that seem too low for HaSS conversions
low_ras = comp[comp['RAS'] < 2.0]
if len(low_ras) > 0:
    print(f"\n  Very low RAS values < 2.0 ({len(low_ras)} players):")
    for _, r in low_ras.iterrows():
        print(f"    {r['player_name']:<28} {int(r['draft_year'])} RAS={r['RAS']:.2f}")

# Breakout age edge cases
ba_edge = comp[(comp['breakout_age'] > 23) | (comp['breakout_age'] < 18)]
if len(ba_edge) > 0:
    print(f"\n  Edge-case breakout ages ({len(ba_edge)} players):")
    for _, r in ba_edge.iterrows():
        print(f"    {r['player_name']:<28} {int(r['draft_year'])} breakout_age={r['breakout_age']}")

# rush_yards is null but rush_source isn't 'unknown'
rush_null_bad = comp[(comp['rush_yards'].isna()) & (~comp['rush_source'].isin(['unknown_fcs', 'no_final_season_data', 'no_college_data']))]
if len(rush_null_bad) > 0:
    print(f"\n  rush_yards is null but rush_source doesn't explain why ({len(rush_null_bad)}):")
    for _, r in rush_null_bad.iterrows():
        print(f"    {r['player_name']:<28} source={r['rush_source']}")

# declare_status values
print(f"\n  declare_status distribution:")
print(f"    {comp['declare_status'].value_counts().to_dict()}")

# Check for 'notes' column — any flags?
if 'notes' in comp.columns:
    notes_filled = comp[comp['notes'].notna() & (comp['notes'] != '')]
    if len(notes_filled) > 0:
        print(f"\n  Players with notes: {len(notes_filled)}")


# ============================================================================
# CHECK 5: JOIN TEST — Dry Run Merge
# ============================================================================

print("\n\n" + "=" * 120)
print("CHECK 5: JOIN TEST — Dry Run Merge of All 6 Components")
print("=" * 120)

# Start with the components file (339 WRs)
base = comp[['player_name', 'draft_year', 'pick', 'round', 'college',
             'breakout_age', 'peak_dominator', 'RAS',
             'declare_status', 'early_declare',
             'rush_yards', 'rush_attempts', 'rush_touchdowns', 'rush_source',
             'hit24', 'hit12', 'best_ppr']].copy()

print(f"\nStarting with: {len(base)} WRs from wr_backtest_all_components.csv")

# Join 1: Teammate Scores
teammate_join = teammate[['player_name', 'draft_year', 'avg_teammate_dc', 'teammate_count', 'best_teammate_pick']].copy()
merged = base.merge(teammate_join, on=['player_name', 'draft_year'], how='left', indicator=True)
n_matched = (merged['_merge'] == 'both').sum()
n_left = (merged['_merge'] == 'left_only').sum()
print(f"\n  + wr_teammate_scores.csv: {n_matched} matched, {n_left} LEFT ONLY (no teammate data)")
if n_left > 0:
    missing_tm = merged[merged['_merge'] == 'left_only']
    for _, r in missing_tm.iterrows():
        print(f"    DROPPED: {r['player_name']} ({int(r['draft_year'])})")
merged = merged.drop('_merge', axis=1)

# Join 2: Games Played
games_join = games[['player_name', 'draft_year', 'games_played', 'games_source']].copy()
merged = merged.merge(games_join, on=['player_name', 'draft_year'], how='left', indicator=True)
n_matched = (merged['_merge'] == 'both').sum()
n_left = (merged['_merge'] == 'left_only').sum()
print(f"  + wr_games_played.csv:    {n_matched} matched, {n_left} LEFT ONLY (no games data)")
if n_left > 0:
    missing_gp = merged[merged['_merge'] == 'left_only']
    for _, r in missing_gp.iterrows():
        print(f"    DROPPED: {r['player_name']} ({int(r['draft_year'])})")
merged = merged.drop('_merge', axis=1)

# Join 3: Outcomes (for hit rates — needed for optimization)
outcomes = files['outcomes']
outcomes_wr = outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year',
                                                       'first_3yr_ppg', 'career_ppg']].copy()
merged = merged.merge(outcomes_wr, on=['player_name', 'draft_year'], how='left', indicator=True)
n_matched = (merged['_merge'] == 'both').sum()
n_left = (merged['_merge'] == 'left_only').sum()
print(f"  + backtest_outcomes.csv:  {n_matched} matched, {n_left} LEFT ONLY (no outcome data)")
if n_left > 0:
    missing_out = merged[merged['_merge'] == 'left_only']
    for _, r in missing_out.iterrows():
        print(f"    DROPPED: {r['player_name']} ({int(r['draft_year'])})")
merged = merged.drop('_merge', axis=1)

# Final: count complete rows
print(f"\n  Final merged dataset: {len(merged)} rows")

# Count rows with ALL required columns non-null
required_for_scoring = ['pick', 'breakout_age', 'peak_dominator', 'RAS',
                        'declare_status', 'early_declare',
                        'avg_teammate_dc', 'games_played']

# rush_yards can be null for 3 legitimate cases
required_strict = ['pick', 'breakout_age', 'RAS', 'declare_status',
                   'avg_teammate_dc', 'games_played']

complete_strict = merged.dropna(subset=required_strict)
print(f"\n  Complete data (required columns non-null): {len(complete_strict)}/{len(merged)}")

# For rushing: count non-null + legitimately-null
rush_ok = merged[merged['rush_yards'].notna() |
                 merged['rush_source'].isin(['unknown_fcs', 'no_final_season_data', 'no_college_data'])]
print(f"  Rushing: {merged['rush_yards'].notna().sum()} with data + "
      f"{merged['rush_source'].isin(['unknown_fcs', 'no_final_season_data', 'no_college_data']).sum()} legitimately null")

# Show incomplete rows
incomplete = merged[merged[required_strict].isna().any(axis=1)]
if len(incomplete) > 0:
    print(f"\n  INCOMPLETE PLAYERS ({len(incomplete)}):")
    for _, row in incomplete.iterrows():
        missing = [c for c in required_strict if pd.isna(row[c])]
        print(f"    {row['player_name']:<28} {int(row['draft_year'])} — missing: {', '.join(missing)}")
else:
    print(f"\n  ALL 339 WRs have complete data for all 6 components!")

# Final summary
print(f"\n\n{'='*120}")
print("INTEGRITY CHECK SUMMARY")
print("=" * 120)
print(f"\n  Total WRs: {len(merged)}")
print(f"  Complete for all 6 components: {len(complete_strict)}")
print(f"  Ready for weight optimization: {'YES' if len(complete_strict) == 339 else 'NO — see issues above'}")
