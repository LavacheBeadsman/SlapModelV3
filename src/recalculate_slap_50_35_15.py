"""
SLAP Score V3 - Recalculate with Edge-Finding Weights
NEW WEIGHTS: DC (50%) + Production/Breakout (35%) + RAS (15%)

Rationale for weight change:
- 85/10/5 created deltas too small to matter (~3-4 points avg)
- 50/35/15 creates meaningful deltas (~8-12 points avg)
- Best combined accuracy for sleeper/bust identification
- Correlation only drops ~0.1 from DC-only baseline
"""

import pandas as pd
import numpy as np

print("=" * 90)
print("SLAP SCORE V3 - RECALCULATING WITH EDGE-FINDING WEIGHTS")
print("NEW WEIGHTS: DC (50%) + Production/Breakout (35%) + RAS (15%)")
print("=" * 90)

# ============================================================================
# NEW WEIGHTS (optimized for edge finding)
# ============================================================================
WEIGHT_DC = 0.50
WEIGHT_PRODUCTION = 0.35  # Breakout for WR, Receiving Production for RB
WEIGHT_RAS = 0.15

print(f"\nWeight Configuration:")
print(f"  Draft Capital: {WEIGHT_DC*100:.0f}%")
print(f"  Production:    {WEIGHT_PRODUCTION*100:.0f}%")
print(f"  RAS:           {WEIGHT_RAS*100:.0f}%")

# ============================================================================
# LOAD EXISTING DATA (preserves production scores already calculated)
# ============================================================================
print("\n" + "=" * 90)
print("LOADING EXISTING DATA")
print("=" * 90)

# Load WR data
wr_data = pd.read_csv('output/slap_2026_wr.csv')
print(f"Loaded {len(wr_data)} WRs")

# Load RB data (already has production_score from receiving metric)
rb_data = pd.read_csv('output/slap_2026_rb.csv')
print(f"Loaded {len(rb_data)} RBs")

# ============================================================================
# RECALCULATE WR SCORES
# ============================================================================
print("\n" + "=" * 90)
print("RECALCULATING WR SLAP SCORES")
print("=" * 90)

# WR uses breakout_score as production metric
wr_data['slap_score_new'] = (
    WEIGHT_DC * wr_data['dc_score'] +
    WEIGHT_PRODUCTION * wr_data['breakout_score'] +
    WEIGHT_RAS * wr_data['ras_score']
)

wr_data['delta_vs_dc_new'] = wr_data['slap_score_new'] - wr_data['dc_score']

# Show change from old to new
wr_data['slap_change'] = wr_data['slap_score_new'] - wr_data['slap_score']
wr_data['delta_change'] = wr_data['delta_vs_dc_new'] - wr_data['delta_vs_dc']

print(f"\nWR Score Changes (85/10/5 -> 50/35/15):")
print(f"  Avg SLAP change: {wr_data['slap_change'].mean():+.1f}")
print(f"  Avg Delta change: {wr_data['delta_change'].mean():+.1f}")
print(f"  Old avg |delta|: {wr_data['delta_vs_dc'].abs().mean():.1f}")
print(f"  New avg |delta|: {wr_data['delta_vs_dc_new'].abs().mean():.1f}")

# Update columns
wr_data['slap_score'] = wr_data['slap_score_new']
wr_data['delta_vs_dc'] = wr_data['delta_vs_dc_new']

# Re-rank
wr_data = wr_data.sort_values('slap_score', ascending=False).reset_index(drop=True)
wr_data['rank'] = range(1, len(wr_data) + 1)

# ============================================================================
# RECALCULATE RB SCORES
# ============================================================================
print("\n" + "=" * 90)
print("RECALCULATING RB SLAP SCORES")
print("=" * 90)

# RB uses production_score (receiving metric) as production
rb_data['slap_score_new'] = (
    WEIGHT_DC * rb_data['dc_score'] +
    WEIGHT_PRODUCTION * rb_data['production_score'] +
    WEIGHT_RAS * rb_data['ras_score']
)

rb_data['delta_vs_dc_new'] = rb_data['slap_score_new'] - rb_data['dc_score']

# Show change
rb_data['slap_change'] = rb_data['slap_score_new'] - rb_data['slap_score']
rb_data['delta_change'] = rb_data['delta_vs_dc_new'] - rb_data['delta_vs_dc']

print(f"\nRB Score Changes (85/10/5 -> 50/35/15):")
print(f"  Avg SLAP change: {rb_data['slap_change'].mean():+.1f}")
print(f"  Avg Delta change: {rb_data['delta_change'].mean():+.1f}")
print(f"  Old avg |delta|: {rb_data['delta_vs_dc'].abs().mean():.1f}")
print(f"  New avg |delta|: {rb_data['delta_vs_dc_new'].abs().mean():.1f}")

# Update columns
rb_data['slap_score'] = rb_data['slap_score_new']
rb_data['delta_vs_dc'] = rb_data['delta_vs_dc_new']

# Re-rank
rb_data = rb_data.sort_values('slap_score', ascending=False).reset_index(drop=True)
rb_data['rank'] = range(1, len(rb_data) + 1)

# ============================================================================
# DISPLAY WR RANKINGS
# ============================================================================
print("\n" + "=" * 90)
print("2026 WR SLAP RANKINGS (50/35/15 WEIGHTS)")
print("=" * 90)

print(f"\n{'Rank':<5} {'Player':<28} {'School':<18} {'Pick':>5} {'SLAP':>6} {'DC':>5} {'BO':>5} {'Delta':>7}")
print("-" * 100)

for _, row in wr_data.head(35).iterrows():
    bo_flag = "*" if row['breakout_status'] == 'imputed' else ""
    print(f"{row['rank']:<5} {row['player_name']:<28} {row['school']:<18} {int(row['projected_pick']):>5} "
          f"{row['slap_score']:>6.1f} {row['dc_score']:>5.1f} {row['breakout_score']:>4.0f}{bo_flag:<1} {row['delta_vs_dc']:>+7.1f}")

print("\n* = imputed breakout age")

# ============================================================================
# DISPLAY RB RANKINGS
# ============================================================================
print("\n" + "=" * 90)
print("2026 RB SLAP RANKINGS (50/35/15 WEIGHTS)")
print("=" * 90)

print(f"\n{'Rank':<5} {'Player':<28} {'School':<18} {'Pick':>5} {'SLAP':>6} {'DC':>5} {'Prod':>5} {'Delta':>7}")
print("-" * 100)

for _, row in rb_data.head(35).iterrows():
    prod_flag = "*" if row['production_status'] == 'imputed' else ""
    print(f"{row['rank']:<5} {row['player_name']:<28} {row['school']:<18} {int(row['projected_pick']):>5} "
          f"{row['slap_score']:>6.1f} {row['dc_score']:>5.1f} {row['production_score']:>4.0f}{prod_flag:<1} {row['delta_vs_dc']:>+7.1f}")

print("\n* = imputed production (no CFBD data)")

# ============================================================================
# HIGHLIGHT EDGE CASES
# ============================================================================
print("\n" + "=" * 90)
print("EDGE CASES: WHERE MODEL DISAGREES WITH SCOUTS")
print("=" * 90)

# WR Sleepers (big positive delta)
print("\n🔥 WR SLEEPERS (Model likes MORE than draft slot):")
print("-" * 70)
wr_sleepers = wr_data[wr_data['delta_vs_dc'] > 10].sort_values('delta_vs_dc', ascending=False)
for _, row in wr_sleepers.head(10).iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['projected_pick']):>3} | Delta: {row['delta_vs_dc']:>+6.1f} | Breakout: {row['breakout_score']:.0f}")

# WR Fades (big negative delta)
print("\n⚠️  WR FADES (Model likes LESS than draft slot):")
print("-" * 70)
wr_fades = wr_data[wr_data['delta_vs_dc'] < -5].sort_values('delta_vs_dc', ascending=True)
for _, row in wr_fades.head(10).iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['projected_pick']):>3} | Delta: {row['delta_vs_dc']:>+6.1f} | Breakout: {row['breakout_score']:.0f}")

# RB Sleepers
print("\n🔥 RB SLEEPERS (Model likes MORE than draft slot):")
print("-" * 70)
rb_sleepers = rb_data[rb_data['delta_vs_dc'] > 5].sort_values('delta_vs_dc', ascending=False)
for _, row in rb_sleepers.head(10).iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['projected_pick']):>3} | Delta: {row['delta_vs_dc']:>+6.1f} | Prod: {row['production_score']:.0f}")

# RB Fades
print("\n⚠️  RB FADES (Model likes LESS than draft slot):")
print("-" * 70)
rb_fades = rb_data[rb_data['delta_vs_dc'] < -5].sort_values('delta_vs_dc', ascending=True)
for _, row in rb_fades.head(10).iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['projected_pick']):>3} | Delta: {row['delta_vs_dc']:>+6.1f} | Prod: {row['production_score']:.0f}")

# ============================================================================
# SAVE OUTPUT
# ============================================================================
print("\n" + "=" * 90)
print("SAVING OUTPUT")
print("=" * 90)

# Clean up temp columns before saving
wr_output = wr_data.drop(columns=['slap_score_new', 'delta_vs_dc_new', 'slap_change', 'delta_change'], errors='ignore')
rb_output = rb_data.drop(columns=['slap_score_new', 'delta_vs_dc_new', 'slap_change', 'delta_change'], errors='ignore')

# Save WR
wr_output.to_csv('output/slap_2026_wr.csv', index=False)
print(f"Saved: output/slap_2026_wr.csv ({len(wr_output)} WRs)")

# Save RB
rb_output.to_csv('output/slap_2026_rb.csv', index=False)
print(f"Saved: output/slap_2026_rb.csv ({len(rb_output)} RBs)")

# Combined
combined = pd.concat([
    wr_output.assign(position='WR'),
    rb_output.assign(position='RB')
], ignore_index=True)
combined = combined.sort_values('slap_score', ascending=False).reset_index(drop=True)
combined['overall_rank'] = range(1, len(combined) + 1)
combined.to_csv('output/slap_2026_combined.csv', index=False)
print(f"Saved: output/slap_2026_combined.csv ({len(combined)} total)")

print("\n" + "=" * 90)
print("DONE! Weights updated to 50/35/15 for edge finding.")
print("=" * 90)
