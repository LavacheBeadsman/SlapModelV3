"""
Pull WR NFL Fantasy Data and Complete Validation
Uses nflverse data to get best season PPG for all WRs
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=" * 100)
print("STEP 1: PULL WR NFL FANTASY DATA FROM NFLVERSE")
print("=" * 100)

# Load the current database
df = pd.read_csv('output/slap_complete_database_v4.csv')
wr_backtest = df[(df['position'] == 'WR') & (df['draft_year'] >= 2015) & (df['draft_year'] <= 2023)].copy()
print(f"\nWRs in backtest: {len(wr_backtest)}")

# Get unique WR names for matching
wr_names = set(wr_backtest['player_name'].str.lower().str.strip())
print(f"Unique WR names to match: {len(wr_names)}")

# Pull NFL fantasy data from nflverse
# We need seasonal data to get best season PPR
print("\nFetching NFL fantasy data from nflverse...")

all_seasons = []
for year in range(2015, 2025):
    try:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.csv"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            season_df = pd.read_csv(StringIO(response.text))
            season_df['season'] = year
            all_seasons.append(season_df)
            print(f"   {year}: {len(season_df)} records")
        else:
            print(f"   {year}: Failed (status {response.status_code})")
    except Exception as e:
        print(f"   {year}: Error - {e}")

# Combine all seasons
nfl_data = pd.concat(all_seasons, ignore_index=True)
print(f"\nTotal NFL records: {len(nfl_data)}")

# Filter to WRs only
nfl_wr = nfl_data[nfl_data['position'] == 'WR'].copy()
print(f"WR records: {len(nfl_wr)}")

# Check what columns we have
print(f"\nAvailable columns: {list(nfl_wr.columns)[:20]}...")

# Calculate season PPR points if not already there
if 'fantasy_points_ppr' in nfl_wr.columns:
    ppr_col = 'fantasy_points_ppr'
else:
    # Calculate manually: rec + rec_yards/10 + rec_td*6 + rush_yards/10 + rush_td*6
    nfl_wr['fantasy_points_ppr'] = (
        nfl_wr.get('receptions', 0) * 1 +
        nfl_wr.get('receiving_yards', 0) / 10 +
        nfl_wr.get('receiving_tds', 0) * 6 +
        nfl_wr.get('rushing_yards', 0) / 10 +
        nfl_wr.get('rushing_tds', 0) * 6
    )
    ppr_col = 'fantasy_points_ppr'

print(f"\nUsing PPR column: {ppr_col}")

# Aggregate by player and season to get season totals
season_totals = nfl_wr.groupby(['player_name', 'season']).agg({
    ppr_col: 'sum',
    'player_id': 'first'
}).reset_index()

# Get games played per season
games_played = nfl_wr.groupby(['player_name', 'season']).size().reset_index(name='games')
season_totals = season_totals.merge(games_played, on=['player_name', 'season'])

# Calculate PPG for each season
season_totals['ppg'] = season_totals[ppr_col] / season_totals['games']

# Filter to meaningful seasons (at least 4 games)
season_totals = season_totals[season_totals['games'] >= 4]

print(f"\nSeason totals (4+ games): {len(season_totals)}")

# Get best season PPG for each player
best_seasons = season_totals.loc[season_totals.groupby('player_name')['ppg'].idxmax()]
best_seasons = best_seasons[['player_name', 'season', ppr_col, 'games', 'ppg']].copy()
best_seasons.columns = ['nfl_name', 'best_season', 'best_ppr', 'games', 'best_ppg']

print(f"Unique WRs with best season: {len(best_seasons)}")

# ============================================================================
# NAME MATCHING
# ============================================================================
print("\n" + "=" * 100)
print("NAME MATCHING")
print("=" * 100)

# Create lookup dictionaries
best_seasons['nfl_name_lower'] = best_seasons['nfl_name'].str.lower().str.strip()

# Try exact match first
def find_ppg(player_name):
    name_lower = player_name.lower().strip()
    match = best_seasons[best_seasons['nfl_name_lower'] == name_lower]
    if len(match) > 0:
        return match.iloc[0]['best_ppg']

    # Try partial match (first and last name)
    parts = name_lower.split()
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        for _, row in best_seasons.iterrows():
            nfl_parts = row['nfl_name_lower'].split()
            if len(nfl_parts) >= 2:
                if nfl_parts[0] == first and nfl_parts[-1] == last:
                    return row['best_ppg']

    return None

# Apply matching
wr_backtest['nfl_best_ppg_new'] = wr_backtest['player_name'].apply(find_ppg)

matched = wr_backtest['nfl_best_ppg_new'].notna().sum()
print(f"\nMatched: {matched}/{len(wr_backtest)} ({matched/len(wr_backtest)*100:.1f}%)")

# Show unmatched
unmatched = wr_backtest[wr_backtest['nfl_best_ppg_new'].isna()]['player_name'].tolist()
if len(unmatched) > 0:
    print(f"\nUnmatched WRs ({len(unmatched)}):")
    for name in unmatched[:20]:
        print(f"   {name}")
    if len(unmatched) > 20:
        print(f"   ... and {len(unmatched) - 20} more")

# ============================================================================
# STEP 2: VERIFY DATA
# ============================================================================
print("\n" + "=" * 100)
print("STEP 2: VERIFY DATA")
print("=" * 100)

verify_players = ['Ja\'Marr Chase', 'CeeDee Lamb', 'Justin Jefferson', 'N\'Keal Harry',
                  'Tyreek Hill', 'Davante Adams', 'Cooper Kupp', 'Henry Ruggs III']

print(f"\n{'Player':<25} {'Expected':<12} {'Actual PPG':<12} {'Status'}")
print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}")

for player in verify_players:
    row = wr_backtest[wr_backtest['player_name'] == player]
    if len(row) > 0:
        ppg = row.iloc[0]['nfl_best_ppg_new']
        if player == "Ja'Marr Chase":
            expected = "~20+"
            status = "✓" if ppg and ppg >= 18 else "✗"
        elif player == "CeeDee Lamb":
            expected = "~20+"
            status = "✓" if ppg and ppg >= 18 else "✗"
        elif player == "Justin Jefferson":
            expected = "~22+"
            status = "✓" if ppg and ppg >= 20 else "✗"
        elif player == "N'Keal Harry":
            expected = "~5"
            status = "✓" if ppg and ppg <= 8 else "✗"
        elif player == "Tyreek Hill":
            expected = "~20+"
            status = "✓" if ppg and ppg >= 18 else "✗"
        elif player == "Cooper Kupp":
            expected = "~25+"
            status = "✓" if ppg and ppg >= 20 else "✗"
        else:
            expected = "?"
            status = "?"

        ppg_str = f"{ppg:.1f}" if ppg else "N/A"
        print(f"{player:<25} {expected:<12} {ppg_str:<12} {status}")
    else:
        print(f"{player:<25} {'N/A':<12} {'Not in DB':<12}")

# Show top 10 WRs by PPG
print(f"\nTop 10 WRs by best season PPG:")
top_wr = wr_backtest[wr_backtest['nfl_best_ppg_new'].notna()].nlargest(10, 'nfl_best_ppg_new')
for _, row in top_wr.iterrows():
    print(f"   {row['player_name']}: {row['nfl_best_ppg_new']:.1f} PPG")

# ============================================================================
# STEP 3: UPDATE DATABASE
# ============================================================================
print("\n" + "=" * 100)
print("STEP 3: UPDATE DATABASE")
print("=" * 100)

# Merge the new PPG data into the main dataframe
ppg_lookup = wr_backtest[['player_name', 'draft_year', 'nfl_best_ppg_new']].copy()

# Update the main dataframe
df_updated = df.copy()

# For WRs in backtest, update their nfl_best_ppg
for idx, row in df_updated.iterrows():
    if row['position'] == 'WR' and 2015 <= row['draft_year'] <= 2023:
        match = ppg_lookup[(ppg_lookup['player_name'] == row['player_name']) &
                           (ppg_lookup['draft_year'] == row['draft_year'])]
        if len(match) > 0 and pd.notna(match.iloc[0]['nfl_best_ppg_new']):
            df_updated.at[idx, 'nfl_best_ppg'] = match.iloc[0]['nfl_best_ppg_new']

# Save updated database
df_updated.to_csv('output/slap_complete_database_v4.csv', index=False)
print(f"\n✓ Updated slap_complete_database_v4.csv")

# Verify
wr_check = df_updated[(df_updated['position'] == 'WR') & (df_updated['draft_year'] >= 2015) & (df_updated['draft_year'] <= 2023)]
wr_with_ppg = wr_check['nfl_best_ppg'].notna().sum()
print(f"   WRs with PPG data: {wr_with_ppg}/{len(wr_check)} ({wr_with_ppg/len(wr_check)*100:.1f}%)")

# ============================================================================
# STEP 4: RUN WR VALIDATION
# ============================================================================
print("\n" + "=" * 100)
print("STEP 4: WR VALIDATION ANALYSIS")
print("=" * 100)

# Reload the updated data
df = pd.read_csv('output/slap_complete_database_v4.csv')
wr_valid = df[(df['position'] == 'WR') & (df['draft_year'] >= 2015) & (df['draft_year'] <= 2023) & (df['nfl_best_ppg'].notna())].copy()

print(f"\nWRs with PPG for validation: {len(wr_valid)}")

# 4.1 Overall Correlation
print(f"\n📊 4.1 OVERALL CORRELATION")
slap_corr = wr_valid['slap_score'].corr(wr_valid['nfl_best_ppg'])
dc_corr = wr_valid['dc_score'].corr(wr_valid['nfl_best_ppg'])

print(f"   SLAP vs PPG: r = {slap_corr:.3f}")
print(f"   DC vs PPG:   r = {dc_corr:.3f}")
print(f"   Difference:  {slap_corr - dc_corr:+.3f}")

if slap_corr > dc_corr:
    print(f"   → SLAP beats DC by {(slap_corr - dc_corr)/dc_corr*100:+.1f}%")
else:
    print(f"   → SLAP underperforms DC by {(slap_corr - dc_corr)/dc_corr*100:.1f}%")

# 4.2 By Draft Round
print(f"\n📊 4.2 CORRELATION BY DRAFT ROUND")
print(f"\n   {'Round':<10} {'N':<6} {'DC r':<10} {'SLAP r':<10} {'SLAP Wins?'}")
print(f"   {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*12}")

rounds = [
    (1, 32, 'Round 1'),
    (33, 64, 'Round 2'),
    (65, 128, 'Rounds 3-4'),
    (129, 262, 'Rounds 5-7')
]

for low, high, label in rounds:
    rd_data = wr_valid[(wr_valid['pick'] >= low) & (wr_valid['pick'] <= high)]
    if len(rd_data) >= 5:
        slap_r = rd_data['slap_score'].corr(rd_data['nfl_best_ppg'])
        dc_r = rd_data['dc_score'].corr(rd_data['nfl_best_ppg'])
        wins = "Yes ✓" if slap_r > dc_r + 0.02 else ("No ✗" if dc_r > slap_r + 0.02 else "Tie")
        print(f"   {label:<10} {len(rd_data):<6} {dc_r:<10.3f} {slap_r:<10.3f} {wins}")
    else:
        print(f"   {label:<10} {len(rd_data):<6} {'N/A':<10} {'N/A':<10} {'Too few'}")

# 4.3 Hit Rate by SLAP Tier
print(f"\n📊 4.3 HIT RATE BY SLAP TIER (hit24 = top 24 WR)")

def assign_tier(score):
    if score >= 90: return '1-Elite (90+)'
    elif score >= 80: return '2-Great (80-89)'
    elif score >= 70: return '3-Good (70-79)'
    elif score >= 60: return '4-Average (60-69)'
    elif score >= 50: return '5-Below Avg (50-59)'
    else: return '6-Poor (<50)'

wr_valid['tier'] = wr_valid['slap_score'].apply(assign_tier)

print(f"\n   {'Tier':<25} {'N':<6} {'Hit24 Rate':<12} {'Avg PPG'}")
print(f"   {'-'*25} {'-'*6} {'-'*12} {'-'*10}")

for tier in sorted(wr_valid['tier'].unique()):
    tier_data = wr_valid[wr_valid['tier'] == tier]
    if len(tier_data) > 0:
        hit_rate = tier_data['nfl_hit24'].mean() * 100
        avg_ppg = tier_data['nfl_best_ppg'].mean()
        print(f"   {tier:<25} {len(tier_data):<6} {hit_rate:<12.0f}% {avg_ppg:<10.1f}")

# ============================================================================
# STEP 5: COMPARE WR vs RB MODEL
# ============================================================================
print("\n" + "=" * 100)
print("STEP 5: WR vs RB MODEL COMPARISON")
print("=" * 100)

# Get RB validation data
rb_valid = df[(df['position'] == 'RB') & (df['draft_year'] >= 2015) & (df['draft_year'] <= 2023) & (df['nfl_best_ppg'].notna())].copy()

# RB correlations
rb_slap_corr = rb_valid['slap_score'].corr(rb_valid['nfl_best_ppg'])
rb_dc_corr = rb_valid['dc_score'].corr(rb_valid['nfl_best_ppg'])

# RB tier hit rates
rb_valid['tier'] = rb_valid['slap_score'].apply(assign_tier)
rb_top_tier = rb_valid[rb_valid['slap_score'] >= 80]
rb_top_hit = rb_top_tier['nfl_hit24'].mean() * 100 if len(rb_top_tier) > 0 else 0

# WR top tier hit rate
wr_top_tier = wr_valid[wr_valid['slap_score'] >= 80]
wr_top_hit = wr_top_tier['nfl_hit24'].mean() * 100 if len(wr_top_tier) > 0 else 0

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WR vs RB MODEL COMPARISON                            │
├────────────────────────────┬────────────────────┬───────────────────────────┤
│ Metric                     │ WR Model           │ RB Model                  │
├────────────────────────────┼────────────────────┼───────────────────────────┤
│ Sample Size                │ {len(wr_valid):<18} │ {len(rb_valid):<25} │
│ SLAP vs PPG correlation    │ r = {slap_corr:<13.3f} │ r = {rb_slap_corr:<20.3f} │
│ DC vs PPG correlation      │ r = {dc_corr:<13.3f} │ r = {rb_dc_corr:<20.3f} │
│ SLAP improvement over DC   │ {slap_corr - dc_corr:+.3f}              │ {rb_slap_corr - rb_dc_corr:+.3f}                     │
│ Top tier (80+) hit rate    │ {wr_top_hit:<18.0f}%│ {rb_top_hit:<24.0f}%│
└────────────────────────────┴────────────────────┴───────────────────────────┘
""")

# Summary
print("\n📋 SUMMARY:")
if slap_corr > dc_corr:
    print(f"   ✓ WR model: SLAP beats DC ({slap_corr:.3f} vs {dc_corr:.3f})")
else:
    print(f"   ✗ WR model: SLAP does NOT beat DC ({slap_corr:.3f} vs {dc_corr:.3f})")

if rb_slap_corr > rb_dc_corr:
    print(f"   ✓ RB model: SLAP beats DC ({rb_slap_corr:.3f} vs {rb_dc_corr:.3f})")
else:
    print(f"   ✗ RB model: SLAP does NOT beat DC ({rb_slap_corr:.3f} vs {rb_dc_corr:.3f})")

print("\n" + "=" * 100)
print("VALIDATION COMPLETE")
print("=" * 100)
