"""
Update Backtest Source Files
============================

Merges the fresh NFL fantasy data with the existing backtest files.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("UPDATING BACKTEST SOURCE FILES")
print("=" * 80)

# Load fresh NFL fantasy data
print("\nLoading fresh NFL fantasy data...")
nfl_fantasy = pd.read_csv('data/nfl_fantasy_clean_2015_2025.csv')
print(f"  NFL fantasy records: {len(nfl_fantasy)}")

# ============================================================================
# UPDATE WR BACKTEST FILE
# ============================================================================
print("\n" + "=" * 80)
print("UPDATING WR BACKTEST FILE")
print("=" * 80)

# Load existing WR backtest
wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"Existing WR backtest: {len(wr_backtest)} records")

# Create name matching columns
wr_backtest['name_clean'] = wr_backtest['player_name'].str.lower().str.strip().str.replace('.', '', regex=False).str.replace("'", "", regex=False).str.replace('-', ' ', regex=False)
nfl_fantasy['name_clean'] = nfl_fantasy['player_name'].str.lower().str.strip().str.replace('.', '', regex=False).str.replace("'", "", regex=False).str.replace('-', ' ', regex=False)

# Get WR fantasy data
wr_fantasy = nfl_fantasy[nfl_fantasy['position'] == 'WR'].copy()

# Merge on name and draft year
wr_merged = wr_backtest.merge(
    wr_fantasy[['name_clean', 'draft_year', 'best_ppr', 'best_ppg', 'hit24', 'hit12']],
    on=['name_clean', 'draft_year'],
    how='left',
    suffixes=('_old', '_new')
)

# Update with new data where available
for col in ['best_ppr', 'best_ppg', 'hit24', 'hit12']:
    new_col = f'{col}_new'
    old_col = f'{col}_old' if f'{col}_old' in wr_merged.columns else col
    if new_col in wr_merged.columns:
        # Use new data where available, keep old otherwise
        mask = wr_merged[new_col].notna()
        if old_col in wr_merged.columns:
            wr_merged[col] = np.where(mask, wr_merged[new_col], wr_merged[old_col])
        else:
            wr_merged[col] = wr_merged[new_col]

# Clean up columns
cols_to_drop = [c for c in wr_merged.columns if c.endswith('_old') or c.endswith('_new')]
wr_merged = wr_merged.drop(columns=cols_to_drop + ['name_clean'])

# Show update stats
print(f"\nUpdated WR backtest:")
for year in range(2015, 2026):
    yr = wr_merged[wr_merged['draft_year'] == year]
    hits = yr['hit24'].sum() if 'hit24' in yr.columns else 0
    with_data = (yr['best_ppr'] > 0).sum() if 'best_ppr' in yr.columns else 0
    print(f"  {year}: {len(yr)} WRs, {with_data} with NFL data, {int(hits)} hit24")

# Save updated WR backtest
wr_merged.to_csv('data/wr_backtest_expanded_final.csv', index=False)
print(f"\nSaved: data/wr_backtest_expanded_final.csv")

# ============================================================================
# UPDATE RB BACKTEST FILE
# ============================================================================
print("\n" + "=" * 80)
print("UPDATING RB BACKTEST FILE")
print("=" * 80)

# Load existing RB backtest
rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
print(f"Existing RB backtest: {len(rb_backtest)} records")

# Create name matching columns
rb_backtest['name_clean'] = rb_backtest['player_name'].str.lower().str.strip().str.replace('.', '', regex=False).str.replace("'", "", regex=False).str.replace('-', ' ', regex=False)

# Get RB fantasy data
rb_fantasy = nfl_fantasy[nfl_fantasy['position'] == 'RB'].copy()

# Merge on name and draft year
rb_merged = rb_backtest.merge(
    rb_fantasy[['name_clean', 'draft_year', 'best_ppr', 'best_ppg', 'hit24', 'hit12']],
    on=['name_clean', 'draft_year'],
    how='left',
    suffixes=('_old', '_new')
)

# Update with new data where available
for col in ['best_ppr', 'best_ppg', 'hit24', 'hit12']:
    new_col = f'{col}_new'
    old_col = f'{col}_old' if f'{col}_old' in rb_merged.columns else col
    if new_col in rb_merged.columns:
        mask = rb_merged[new_col].notna()
        if old_col in rb_merged.columns:
            rb_merged[col] = np.where(mask, rb_merged[new_col], rb_merged[old_col])
        else:
            rb_merged[col] = rb_merged[new_col]

# Clean up columns
cols_to_drop = [c for c in rb_merged.columns if c.endswith('_old') or c.endswith('_new')]
rb_merged = rb_merged.drop(columns=cols_to_drop + ['name_clean'])

# Show update stats
print(f"\nUpdated RB backtest:")
for year in range(2015, 2026):
    yr = rb_merged[rb_merged['draft_year'] == year]
    if len(yr) > 0:
        hits = yr['hit24'].sum() if 'hit24' in yr.columns else 0
        with_data = (yr['best_ppr'] > 0).sum() if 'best_ppr' in yr.columns else 0
        print(f"  {year}: {len(yr)} RBs, {with_data} with NFL data, {int(hits)} hit24")

# Save updated RB backtest
rb_merged.to_csv('data/rb_backtest_with_receiving.csv', index=False)
print(f"\nSaved: data/rb_backtest_with_receiving.csv")

# ============================================================================
# ADD 2025 RB DRAFT CLASS (if missing)
# ============================================================================
print("\n" + "=" * 80)
print("CHECKING FOR MISSING 2025 RB DRAFT CLASS")
print("=" * 80)

# Check if 2025 RBs are in the backtest
rb_2025_count = len(rb_merged[rb_merged['draft_year'] == 2025])
print(f"2025 RBs in backtest: {rb_2025_count}")

if rb_2025_count == 0:
    print("\nAdding 2025 RB draft class from NFL fantasy data...")

    # Get 2025 RBs from fantasy data
    rb_2025 = rb_fantasy[rb_fantasy['draft_year'] == 2025].copy()

    if len(rb_2025) > 0:
        # Create matching structure
        rb_2025_backtest = pd.DataFrame({
            'player_name': rb_2025['player_name'],
            'draft_year': rb_2025['draft_year'],
            'pick': rb_2025['pick'],
            'round': rb_2025['round'],
            'college': rb_2025['college'],
            'age': np.nan,  # Will need to fill
            'best_ppr': rb_2025['best_ppr'],
            'best_ppg': rb_2025['best_ppg'],
            'best_season': rb_2025['best_season'] if 'best_season' in rb_2025.columns else np.nan,
            'season_rank': np.nan,
            'hit24': rb_2025['hit24'],
            'hit12': rb_2025['hit12'],
            'RAS': np.nan,
            'rec_yards': np.nan,
            'receptions': np.nan,
            'team_pass_att': np.nan,
            'cfbd_name': rb_2025['player_name'].str.lower()
        })

        # Append to backtest
        rb_final = pd.concat([rb_merged, rb_2025_backtest], ignore_index=True)
        rb_final.to_csv('data/rb_backtest_with_receiving.csv', index=False)
        print(f"Added {len(rb_2025_backtest)} RBs from 2025 draft class")
        print(f"New total: {len(rb_final)} RBs")
    else:
        print("No 2025 RBs found in fantasy data")

print("\n" + "=" * 80)
print("BACKTEST FILES UPDATED")
print("=" * 80)
