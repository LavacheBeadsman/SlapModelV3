"""
RB Model Analysis - Systematic Fix

1. Load and parse RB RAS data
2. Check RB backtest data for fantasy PPG errors
3. Verify known RB star PPG values
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("RB MODEL ANALYSIS")
print("=" * 90)

# ============================================================================
# PART 1: LOAD AND PARSE RB RAS DATA
# ============================================================================
print("\n" + "=" * 90)
print("PART 1: RB RAS DATA")
print("=" * 90)

# Load RB RAS
rb_ras = pd.read_csv('data/RBRas20152025.csv')
print(f"\nRB RAS file columns: {list(rb_ras.columns)}")
print(f"Total records: {len(rb_ras)}")

# Clean up - extract name and RAS
rb_ras = rb_ras.rename(columns={'Name': 'player_name', 'Year': 'draft_year'})
rb_ras['RAS'] = pd.to_numeric(rb_ras['RAS'], errors='coerce')

# Show by year
print("\nRB RAS counts by draft year:")
for year in range(2015, 2026):
    count = len(rb_ras[rb_ras['draft_year'] == year])
    print(f"  {year}: {count} RBs with RAS")

# Show top 10 by RAS
print("\nTop 10 RBs by RAS (all years):")
top10 = rb_ras.nlargest(10, 'RAS')[['player_name', 'draft_year', 'College', 'RAS']]
for _, row in top10.iterrows():
    print(f"  {row['player_name']:<25} {int(row['draft_year'])} {row['College']:<20} RAS: {row['RAS']:.2f}")

# ============================================================================
# PART 2: PULL RB FANTASY DATA FROM NFLVERSE
# ============================================================================
print("\n" + "=" * 90)
print("PART 2: RB FANTASY PPG VERIFICATION")
print("=" * 90)

# Load draft picks to get RBs 2015-2024
draft_url = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
print(f"\nFetching draft picks from nflverse...")
draft_df = pd.read_csv(draft_url)

# Filter to RBs
rb_draft = draft_df[
    (draft_df['position'] == 'RB') &
    (draft_df['season'].between(2015, 2024))
].copy()

print(f"Found {len(rb_draft)} RBs drafted 2015-2024")

# Load fantasy stats
stats_url = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv"
print(f"Fetching fantasy stats from nflverse...")
stats_df = pd.read_csv(stats_url, low_memory=False)

# Filter to RBs
rb_stats = stats_df[
    (stats_df['position'] == 'RB') &
    (stats_df['season'].between(2015, 2024))
].copy()

print(f"Found {len(rb_stats)} RB game records")

# Calculate PPR points per game
rb_stats['ppr'] = (
    rb_stats['receiving_yards'].fillna(0) * 0.1 +
    rb_stats['receiving_tds'].fillna(0) * 6 +
    rb_stats['receptions'].fillna(0) * 1 +
    rb_stats['rushing_yards'].fillna(0) * 0.1 +
    rb_stats['rushing_tds'].fillna(0) * 6 +
    rb_stats['receiving_2pt_conversions'].fillna(0) * 2 +
    rb_stats['rushing_2pt_conversions'].fillna(0) * 2 -
    rb_stats['receiving_fumbles_lost'].fillna(0) * 2 -
    rb_stats['rushing_fumbles_lost'].fillna(0) * 2
)

# STEP 1: Aggregate per player+season to get SEASON TOTALS
season_totals = rb_stats.groupby(['player_display_name', 'season']).agg({
    'ppr': 'sum',
    'week': 'count'
}).reset_index()
season_totals.columns = ['player_name', 'season', 'season_ppr', 'games']

# Calculate PPG per season
season_totals['season_ppg'] = season_totals['season_ppr'] / 17  # Per 17-game season basis

print(f"\nCalculated season totals for {len(season_totals)} RB player-seasons")

# STEP 2: Get best season for each player
best_seasons = season_totals.loc[season_totals.groupby('player_name')['season_ppr'].idxmax()]
best_seasons = best_seasons.rename(columns={
    'season_ppr': 'best_ppr',
    'season': 'best_season'
})

# ============================================================================
# PART 3: VERIFY PPG FOR KNOWN RB STARS
# ============================================================================
print("\n" + "=" * 90)
print("PART 3: VERIFY PPG FOR KNOWN RB STARS")
print("=" * 90)

# Target RBs to verify
target_rbs = [
    'Jonathan Taylor',
    'Derrick Henry',
    'Saquon Barkley',
    'Josh Jacobs',
    'Najee Harris',
    'Breece Hall',
    'Bijan Robinson',
    'Jahmyr Gibbs',
    "De'Von Achane",
    'Kenneth Walker III'
]

print(f"\n{'Player':<25} {'Best Season':>11} {'Season PPR':>11} {'PPG (÷17)':>10} {'Games':>6}")
print("-" * 70)

for name in target_rbs:
    # Find player (try exact match first, then partial)
    match = best_seasons[best_seasons['player_name'] == name]
    if len(match) == 0:
        # Try partial match
        match = best_seasons[best_seasons['player_name'].str.contains(name, case=False, na=False)]

    if len(match) > 0:
        row = match.iloc[0]
        ppg = row['best_ppr'] / 17
        print(f"{row['player_name']:<25} {int(row['best_season']):>11} {row['best_ppr']:>11.1f} {ppg:>10.1f} {row['games']:>6}")
    else:
        print(f"{name:<25} NOT FOUND")

# Show top 20 RBs by best season PPG
print("\n" + "=" * 90)
print("TOP 20 RBs BY BEST SEASON PPG (should be 15-25 range for stars)")
print("=" * 90)

best_seasons['best_ppg'] = best_seasons['best_ppr'] / 17
top20 = best_seasons.nlargest(20, 'best_ppg')[['player_name', 'best_season', 'best_ppr', 'best_ppg', 'games']]

print(f"\n{'Rank':<5} {'Player':<25} {'Season':>7} {'PPR Total':>10} {'PPG':>8} {'Games':>6}")
print("-" * 70)
for i, (_, row) in enumerate(top20.iterrows(), 1):
    print(f"{i:<5} {row['player_name']:<25} {int(row['best_season']):>7} {row['best_ppr']:>10.1f} {row['best_ppg']:>8.1f} {row['games']:>6}")

# ============================================================================
# PART 4: CHECK HIT RATES (Top-24 seasons)
# ============================================================================
print("\n" + "=" * 90)
print("PART 4: RB HIT RATES BY SEASON")
print("=" * 90)

# Rank by season
season_totals['season_rank'] = season_totals.groupby('season')['season_ppr'].rank(ascending=False, method='min')

# Get hit indicators
hits = season_totals.groupby('player_name').agg({
    'season_rank': 'min'  # Best rank
}).reset_index()
hits['hit24'] = (hits['season_rank'] <= 24).astype(int)
hits['hit12'] = (hits['season_rank'] <= 12).astype(int)

# Merge hits to best_seasons
best_seasons = best_seasons.merge(hits[['player_name', 'season_rank', 'hit24', 'hit12']], on='player_name', how='left')

# ============================================================================
# PART 5: MERGE WITH DRAFT DATA TO CREATE RB BACKTEST
# ============================================================================
print("\n" + "=" * 90)
print("PART 5: CREATE RB BACKTEST DATASET")
print("=" * 90)

# Standardize draft data
rb_draft = rb_draft.rename(columns={
    'season': 'draft_year',
    'pfr_player_name': 'player_name'
})[['player_name', 'draft_year', 'pick', 'round', 'college', 'age']].copy()

# Merge draft with fantasy outcomes
rb_backtest = rb_draft.merge(
    best_seasons[['player_name', 'best_ppr', 'best_ppg', 'best_season', 'season_rank', 'hit24', 'hit12']],
    on='player_name',
    how='left'
)

# Fill missing
rb_backtest['best_ppr'] = rb_backtest['best_ppr'].fillna(0)
rb_backtest['best_ppg'] = rb_backtest['best_ppg'].fillna(0)
rb_backtest['hit24'] = rb_backtest['hit24'].fillna(0).astype(int)
rb_backtest['hit12'] = rb_backtest['hit12'].fillna(0).astype(int)

# Merge with RAS
rb_ras_clean = rb_ras[['player_name', 'draft_year', 'RAS']].copy()
rb_backtest = rb_backtest.merge(rb_ras_clean, on=['player_name', 'draft_year'], how='left')

print(f"RB Backtest dataset: {len(rb_backtest)} RBs")
print(f"With fantasy outcomes: {(rb_backtest['best_ppr'] > 0).sum()}")
print(f"With RAS data: {rb_backtest['RAS'].notna().sum()}")

# Summary by year
print(f"\n{'Year':>6} {'RBs':>5} {'Hits24':>7} {'Rate':>7} {'Has RAS':>8}")
print("-" * 40)
for year in range(2015, 2025):
    yr = rb_backtest[rb_backtest['draft_year'] == year]
    n = len(yr)
    hits_yr = yr['hit24'].sum()
    rate = hits_yr / n * 100 if n > 0 else 0
    has_ras = yr['RAS'].notna().sum()
    print(f"{year:>6} {n:>5} {hits_yr:>7} {rate:>6.1f}% {has_ras:>8}")

# Save RB backtest data
rb_backtest.to_csv('data/rb_backtest_2015_2024.csv', index=False)
print(f"\nSaved RB backtest to data/rb_backtest_2015_2024.csv")

# ============================================================================
# PART 6: COMPARE TO EXISTING BACKTEST DATA
# ============================================================================
print("\n" + "=" * 90)
print("PART 6: COMPARE TO EXISTING BACKTEST DATA")
print("=" * 90)

# Load existing backtest hit rates
existing = pd.read_csv('data/backtest_hit_rates.csv')
existing_rbs = existing[existing['position'] == 'RB'].copy()

print("\nChecking if existing backtest has same bug as WR data...")
print("\nExisting backtest PPG values for known RBs:")

for name in ['Jonathan Taylor', 'Bijan Robinson', 'Breece Hall', 'Najee Harris']:
    match = existing_rbs[existing_rbs['player_name'].str.contains(name.split()[0], case=False, na=False)]
    if len(match) > 0:
        row = match.iloc[0]
        best_ppr_old = row['best_ppr'] if 'best_ppr' in row.index else 'N/A'
        # The existing file has 'best_ppr' which might be per-game max, not season total
        print(f"  {row['player_name']:<25} Existing best_ppr: {best_ppr_old}")

    # Compare to new data
    new_match = rb_backtest[rb_backtest['player_name'].str.contains(name.split()[0], case=False, na=False)]
    if len(new_match) > 0:
        row = new_match.iloc[0]
        print(f"  {row['player_name']:<25} NEW best_ppr: {row['best_ppr']:.1f}, PPG: {row['best_ppg']:.1f}")

print("\nDONE! RB backtest data has been created with correct season PPG values.")
