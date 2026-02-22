"""
Calculate SLAP Scores for 2026 WR Class

Note: This is a PRE-DRAFT projection using:
- Projected pick from mock drafts (draft capital)
- Breakout age data (needs to be researched for each player)
- RAS data (not available until combine, Feb 2026)

The output will flag which players need additional data collection.
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from calculate_wr_slap import calculate_wr_slap_scores, get_breakout_age_score

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("2026 WR CLASS - SLAP SCORE PROJECTIONS")
print("="*80)

# Load mock draft
mock = pd.read_csv('data/mock_draft_2026.csv')
wr_2026 = mock[mock['position'] == 'WR'].copy()
wr_2026 = wr_2026.rename(columns={'projected_pick': 'pick'})

print(f"\nLoaded {len(wr_2026)} WR prospects")

# ============================================================================
# LOAD BREAKOUT AGE DATA (if available)
# ============================================================================
# For 2026 class, we need to research breakout ages
# This would come from college stats analysis

# Placeholder: You would fill this in with researched data
# Format: {player_name: breakout_age}
BREAKOUT_AGES_2026 = {
    # Top prospects - research these
    # 'Carnell Tate': 19,  # Example - needs verification
    # 'Jordyn Tyson': 20,  # Example - needs verification
}

# Add breakout ages to dataframe
wr_2026['breakout_age'] = wr_2026['player_name'].map(BREAKOUT_AGES_2026)
wr_2026['RAS'] = np.nan  # Not available until combine

print(f"Breakout ages researched: {len(BREAKOUT_AGES_2026)} / {len(wr_2026)}")
print(f"RAS data: Not yet available (combine is Feb 2026)")

# ============================================================================
# CALCULATE SLAP WITH AVAILABLE DATA
# ============================================================================
print("\n" + "="*80)
print("CALCULATING SLAP SCORES (DC + Breakout only)")
print("="*80)

# Use 2020-2024 normalization parameters for consistency
# From backtest: DC mean=0.122616, std=0.074672
#                Breakout mean=66.49, std=25.89
#                RAS mean=7.64, std=1.79

dc_params = (0.122616, 0.074672)
breakout_params = (66.49, 25.89)
ras_params = (7.64, 1.79)

# Calculate SLAP scores
wr_2026_scored, _, _, _ = calculate_wr_slap_scores(
    wr_2026,
    dc_params=dc_params,
    breakout_params=breakout_params,
    ras_params=ras_params
)

# Sort by SLAP score
wr_2026_scored = wr_2026_scored.sort_values('slap_score', ascending=False)

# ============================================================================
# OUTPUT TOP PROSPECTS
# ============================================================================
print("\n" + "-"*90)
print(f"{'Rank':>4} {'Player':<25} {'Pick':>5} {'SLAP':>6} {'DC':>6} {'Delta':>7} {'Status':<15} {'Data Needed'}")
print("-"*90)

for rank, (_, row) in enumerate(wr_2026_scored.head(30).iterrows(), 1):
    # Determine what data is needed
    needs = []
    if pd.isna(row['breakout_age']):
        needs.append("breakout")
    if pd.isna(row['RAS']):
        needs.append("RAS")
    needs_str = ", ".join(needs) if needs else "Complete"

    print(f"{rank:>4} {row['player_name']:<25} {row['pick']:>5.0f} "
          f"{row['slap_score']:>6.1f} {row['dc_norm']:>6.1f} {row['delta']:>+7.1f} "
          f"{row['athletic_status']:<15} {needs_str}")

# ============================================================================
# DATA COLLECTION PRIORITIES
# ============================================================================
print("\n" + "="*80)
print("DATA COLLECTION PRIORITIES")
print("="*80)

# Top 20 by projected pick (most impactful to get right)
top_by_pick = wr_2026_scored.nsmallest(20, 'pick')

print("\nHighest Priority (Top 20 projected picks):")
print("-"*60)

missing_breakout = top_by_pick[top_by_pick['breakout_age'].isna()]
print(f"\n  Need breakout age research ({len(missing_breakout)} players):")
for _, row in missing_breakout.iterrows():
    print(f"    - {row['player_name']} (projected pick {row['pick']:.0f})")

print("\n  All players need RAS data (available after combine)")

# ============================================================================
# SAVE OUTPUT
# ============================================================================
output_cols = ['player_name', 'pick', 'breakout_age', 'RAS', 'dc_norm',
               'breakout_norm', 'slap_score', 'delta', 'athletic_status']
wr_2026_scored[output_cols].to_csv('output/wr_slap_2026_projections.csv', index=False)
print(f"\n\nSaved: output/wr_slap_2026_projections.csv")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Research breakout ages for top 20-30 WRs:
   - Find their college stats by season
   - Calculate dominator rating (receiving yards / team receiving yards)
   - Identify first season with 20%+ dominator
   - Record their age during that season

2. Wait for combine (Feb 2026) for RAS data

3. Update projected picks as draft approaches

4. Re-run this script after data collection
""")
