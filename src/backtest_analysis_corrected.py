"""
SLAP Score V3 Backtest Analysis - CORRECTED
============================================

Evaluate model performance for 2015-2023 drafts ONLY.
(2024 and 2025 draft classes don't have complete NFL data yet)

As of Feb 2026:
- 2015-2023 classes: Have 3-11 NFL seasons of data (complete)
- 2024 class: Just finished Year 2 (data not in source files)
- 2025 class: Just finished Year 1 (data not in source files)
- 2026 class: Not drafted yet (prospects)
"""

import pandas as pd
import numpy as np
from scipy import stats

print("=" * 80)
print("SLAP SCORE V3 BACKTEST ANALYSIS - CORRECTED")
print("Evaluating 2015-2023 draft classes only (complete NFL data)")
print("=" * 80)

# Load the recalculated data
wr_data = pd.read_csv('output/slap_complete_wr.csv')
rb_data = pd.read_csv('output/slap_complete_rb.csv')

# Filter to 2015-2023 ONLY (players with complete NFL outcomes)
wr_backtest = wr_data[(wr_data['data_type'] == 'backtest') &
                       (wr_data['draft_year'] >= 2015) &
                       (wr_data['draft_year'] <= 2023)].copy()
rb_backtest = rb_data[(rb_data['data_type'] == 'backtest') &
                       (rb_data['draft_year'] >= 2015) &
                       (rb_data['draft_year'] <= 2023)].copy()

print(f"\nBacktest Sample (2015-2023 only):")
print(f"  WRs: {len(wr_backtest)} players")
print(f"  RBs: {len(rb_backtest)} players")
print(f"  Total: {len(wr_backtest) + len(rb_backtest)} players")

# ============================================================================
# WR BACKTEST ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("WR BACKTEST ANALYSIS (2015-2023)")
print("Weights: 65% DC + 20% Breakout Age + 15% RAS")
print("=" * 80)

wr_valid = wr_backtest[wr_backtest['nfl_best_ppr'].notna()].copy()
print(f"\nWRs with NFL data: {len(wr_valid)}")

# Correlations
print("\n--- CORRELATION ANALYSIS ---")
print(f"\n{'Metric':<30} {'vs NFL PPG':<15} {'vs Hit24':<15} {'vs Hit12':<15}")
print("-" * 75)

r_ppg, p_ppg = stats.pearsonr(wr_valid['slap_score'], wr_valid['nfl_best_ppr'])
r_hit24, p_hit24 = stats.pointbiserialr(wr_valid['nfl_hit24'], wr_valid['slap_score'])
r_hit12, p_hit12 = stats.pointbiserialr(wr_valid['nfl_hit12'], wr_valid['slap_score'])
print(f"{'SLAP Score':<30} {r_ppg:.3f} (p={p_ppg:.4f})  {r_hit24:.3f} (p={p_hit24:.4f})  {r_hit12:.3f} (p={p_hit12:.4f})")

r_dc_ppg, p_dc_ppg = stats.pearsonr(wr_valid['dc_score'], wr_valid['nfl_best_ppr'])
r_dc_hit24, p_dc_hit24 = stats.pointbiserialr(wr_valid['nfl_hit24'], wr_valid['dc_score'])
r_dc_hit12, p_dc_hit12 = stats.pointbiserialr(wr_valid['nfl_hit12'], wr_valid['dc_score'])
print(f"{'DC Score (baseline)':<30} {r_dc_ppg:.3f} (p={p_dc_ppg:.4f})  {r_dc_hit24:.3f} (p={p_dc_hit24:.4f})  {r_dc_hit12:.3f} (p={p_dc_hit12:.4f})")

r_prod_ppg, p_prod_ppg = stats.pearsonr(wr_valid['production_score'], wr_valid['nfl_best_ppr'])
r_prod_hit24, p_prod_hit24 = stats.pointbiserialr(wr_valid['nfl_hit24'], wr_valid['production_score'])
r_prod_hit12, p_prod_hit12 = stats.pointbiserialr(wr_valid['nfl_hit12'], wr_valid['production_score'])
print(f"{'Breakout Age Score':<30} {r_prod_ppg:.3f} (p={p_prod_ppg:.4f})  {r_prod_hit24:.3f} (p={p_prod_hit24:.4f})  {r_prod_hit12:.3f} (p={p_prod_hit12:.4f})")

# Hit rates by SLAP tier
print("\n--- HIT RATES BY SLAP TIER ---")
print(f"\n{'SLAP Tier':<15} {'Count':<8} {'Hit24 Rate':<12} {'Hit12 Rate':<12} {'Avg NFL PPG':<12}")
print("-" * 60)

tiers = [(80, 100, "80-100 (Elite)"), (70, 80, "70-80 (Good)"), (60, 70, "60-70 (Avg)"),
         (50, 60, "50-60 (Below)"), (0, 50, "0-50 (Low)")]

for low, high, label in tiers:
    tier = wr_valid[(wr_valid['slap_score'] >= low) & (wr_valid['slap_score'] < high)]
    if len(tier) > 0:
        hit24_rate = tier['nfl_hit24'].mean() * 100
        hit12_rate = tier['nfl_hit12'].mean() * 100
        avg_ppg = tier['nfl_best_ppr'].mean() / 17
        print(f"{label:<15} {len(tier):<8} {hit24_rate:>5.1f}%       {hit12_rate:>5.1f}%       {avg_ppg:>5.1f}")

# ============================================================================
# RB BACKTEST ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("RB BACKTEST ANALYSIS (2015-2023)")
print("Weights: 50% DC + 35% Career Avg Production + 15% RAS")
print("=" * 80)

rb_valid = rb_backtest[rb_backtest['nfl_best_ppr'].notna()].copy()
print(f"\nRBs with NFL data: {len(rb_valid)}")

# Correlations
print("\n--- CORRELATION ANALYSIS ---")
print(f"\n{'Metric':<30} {'vs NFL PPG':<15} {'vs Hit24':<15} {'vs Hit12':<15}")
print("-" * 75)

r_ppg, p_ppg = stats.pearsonr(rb_valid['slap_score'], rb_valid['nfl_best_ppr'])
r_hit24, p_hit24 = stats.pointbiserialr(rb_valid['nfl_hit24'], rb_valid['slap_score'])
r_hit12, p_hit12 = stats.pointbiserialr(rb_valid['nfl_hit12'], rb_valid['slap_score'])
print(f"{'SLAP Score':<30} {r_ppg:.3f} (p={p_ppg:.4f})  {r_hit24:.3f} (p={p_hit24:.4f})  {r_hit12:.3f} (p={p_hit12:.4f})")

r_dc_ppg, p_dc_ppg = stats.pearsonr(rb_valid['dc_score'], rb_valid['nfl_best_ppr'])
r_dc_hit24, p_dc_hit24 = stats.pointbiserialr(rb_valid['nfl_hit24'], rb_valid['dc_score'])
r_dc_hit12, p_dc_hit12 = stats.pointbiserialr(rb_valid['nfl_hit12'], rb_valid['dc_score'])
print(f"{'DC Score (baseline)':<30} {r_dc_ppg:.3f} (p={p_dc_ppg:.4f})  {r_dc_hit24:.3f} (p={p_dc_hit24:.4f})  {r_dc_hit12:.3f} (p={p_dc_hit12:.4f})")

rb_valid_prod = rb_valid[rb_valid['production_score'].notna()]
r_prod_ppg, p_prod_ppg = stats.pearsonr(rb_valid_prod['production_score'], rb_valid_prod['nfl_best_ppr'])
r_prod_hit24, p_prod_hit24 = stats.pointbiserialr(rb_valid_prod['nfl_hit24'], rb_valid_prod['production_score'])
r_prod_hit12, p_prod_hit12 = stats.pointbiserialr(rb_valid_prod['nfl_hit12'], rb_valid_prod['production_score'])
print(f"{'Career Avg Production':<30} {r_prod_ppg:.3f} (p={p_prod_ppg:.4f})  {r_prod_hit24:.3f} (p={p_prod_hit24:.4f})  {r_prod_hit12:.3f} (p={p_prod_hit12:.4f})")

# Hit rates by SLAP tier
print("\n--- HIT RATES BY SLAP TIER ---")
print(f"\n{'SLAP Tier':<15} {'Count':<8} {'Hit24 Rate':<12} {'Hit12 Rate':<12} {'Avg NFL PPG':<12}")
print("-" * 60)

for low, high, label in tiers:
    tier = rb_valid[(rb_valid['slap_score'] >= low) & (rb_valid['slap_score'] < high)]
    if len(tier) > 0:
        hit24_rate = tier['nfl_hit24'].mean() * 100
        hit12_rate = tier['nfl_hit12'].mean() * 100
        avg_ppg = tier['nfl_best_ppr'].mean() / 17
        print(f"{label:<15} {len(tier):<8} {hit24_rate:>5.1f}%       {hit12_rate:>5.1f}%       {avg_ppg:>5.1f}")

# ============================================================================
# COMBINED ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("COMBINED ANALYSIS (2015-2023)")
print("=" * 80)

all_valid = pd.concat([wr_valid, rb_valid], ignore_index=True)
print(f"\nTotal players: {len(all_valid)}")

print("\n--- OVERALL CORRELATION ---")
r_ppg, p_ppg = stats.pearsonr(all_valid['slap_score'], all_valid['nfl_best_ppr'])
r_hit24, p_hit24 = stats.pointbiserialr(all_valid['nfl_hit24'], all_valid['slap_score'])
print(f"SLAP vs NFL PPR Points: r={r_ppg:.3f} (p={p_ppg:.6f})")
print(f"SLAP vs Hit24:          r={r_hit24:.3f} (p={p_hit24:.6f})")

r_dc_ppg, p_dc_ppg = stats.pearsonr(all_valid['dc_score'], all_valid['nfl_best_ppr'])
r_dc_hit24, p_dc_hit24 = stats.pointbiserialr(all_valid['nfl_hit24'], all_valid['dc_score'])
print(f"\nDC vs NFL PPR Points:   r={r_dc_ppg:.3f} (p={p_dc_ppg:.6f})")
print(f"DC vs Hit24:            r={r_dc_hit24:.3f} (p={p_dc_hit24:.6f})")

# ============================================================================
# TOP HITS AND MISSES
# ============================================================================
print("\n" + "=" * 80)
print("MODEL VALIDATION: TOP HITS AND MISSES")
print("=" * 80)

print("\n--- TOP 10 BEST PREDICTIONS (High SLAP → Hit24) ---")
hits = all_valid[(all_valid['nfl_hit24'] == 1)].sort_values('slap_score', ascending=False)
print(f"{'Player':<25} {'Pos':<4} {'Year':<6} {'Pick':<5} {'SLAP':<6} {'NFL PPG':<8}")
print("-" * 60)
for _, row in hits.head(10).iterrows():
    ppg = row['nfl_best_ppr'] / 17
    print(f"{row['player_name']:<25} {row['position']:<4} {int(row['draft_year']):<6} {int(row['pick']):<5} {row['slap_score']:<6.1f} {ppg:<8.1f}")

print("\n--- TOP 10 BIGGEST MISSES (High SLAP → No Hit24) ---")
misses = all_valid[(all_valid['nfl_hit24'] == 0)].sort_values('slap_score', ascending=False)
print(f"{'Player':<25} {'Pos':<4} {'Year':<6} {'Pick':<5} {'SLAP':<6} {'NFL PPG':<8}")
print("-" * 60)
for _, row in misses.head(10).iterrows():
    ppg = row['nfl_best_ppr'] / 17
    print(f"{row['player_name']:<25} {row['position']:<4} {int(row['draft_year']):<6} {int(row['pick']):<5} {row['slap_score']:<6.1f} {ppg:<8.1f}")

print("\n--- TOP 10 BEST VALUE FINDS (Low SLAP → Hit24) ---")
values = all_valid[(all_valid['nfl_hit24'] == 1)].sort_values('slap_score', ascending=True)
print(f"{'Player':<25} {'Pos':<4} {'Year':<6} {'Pick':<5} {'SLAP':<6} {'NFL PPG':<8}")
print("-" * 60)
for _, row in values.head(10).iterrows():
    ppg = row['nfl_best_ppr'] / 17
    print(f"{row['player_name']:<25} {row['position']:<4} {int(row['draft_year']):<6} {int(row['pick']):<5} {row['slap_score']:<6.1f} {ppg:<8.1f}")

# ============================================================================
# YEAR BY YEAR BREAKDOWN
# ============================================================================
print("\n" + "=" * 80)
print("YEAR-BY-YEAR HIT RATES (2015-2023)")
print("=" * 80)

print(f"\n{'Year':<6} {'WRs':<6} {'WR Hit24':<12} {'RBs':<6} {'RB Hit24':<12}")
print("-" * 45)

for year in range(2015, 2024):
    wr_year = wr_valid[wr_valid['draft_year'] == year]
    rb_year = rb_valid[rb_valid['draft_year'] == year]

    wr_hit24 = f"{int(wr_year['nfl_hit24'].sum())}/{len(wr_year)}" if len(wr_year) > 0 else "N/A"
    rb_hit24 = f"{int(rb_year['nfl_hit24'].sum())}/{len(rb_year)}" if len(rb_year) > 0 else "N/A"

    print(f"{year:<6} {len(wr_year):<6} {wr_hit24:<12} {len(rb_year):<6} {rb_hit24:<12}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("BACKTEST SUMMARY (2015-2023)")
print("=" * 80)

wr_hit24_total = int(wr_valid['nfl_hit24'].sum())
wr_hit12_total = int(wr_valid['nfl_hit12'].sum())
rb_hit24_total = int(rb_valid['nfl_hit24'].sum())
rb_hit12_total = int(rb_valid['nfl_hit12'].sum())

# Calculate tier hit rates for summary
wr_elite = wr_valid[wr_valid['slap_score'] >= 80]
rb_elite = rb_valid[rb_valid['slap_score'] >= 60]  # Lower threshold for RBs

print(f"""
SAMPLE SIZE:
  WRs: {len(wr_valid)} drafted 2015-2023
  RBs: {len(rb_valid)} drafted 2015-2023
  Total: {len(all_valid)} players

OVERALL HIT RATES:
  WR Hit24: {wr_hit24_total}/{len(wr_valid)} ({wr_hit24_total/len(wr_valid)*100:.1f}%)
  WR Hit12: {wr_hit12_total}/{len(wr_valid)} ({wr_hit12_total/len(wr_valid)*100:.1f}%)
  RB Hit24: {rb_hit24_total}/{len(rb_valid)} ({rb_hit24_total/len(rb_valid)*100:.1f}%)
  RB Hit12: {rb_hit12_total}/{len(rb_valid)} ({rb_hit12_total/len(rb_valid)*100:.1f}%)

MODEL PERFORMANCE:
  WR SLAP vs NFL PPG:  r = {stats.pearsonr(wr_valid['slap_score'], wr_valid['nfl_best_ppr'])[0]:.3f}
  WR SLAP vs Hit24:    r = {stats.pointbiserialr(wr_valid['nfl_hit24'], wr_valid['slap_score'])[0]:.3f}

  RB SLAP vs NFL PPG:  r = {stats.pearsonr(rb_valid['slap_score'], rb_valid['nfl_best_ppr'])[0]:.3f}
  RB SLAP vs Hit24:    r = {stats.pointbiserialr(rb_valid['nfl_hit24'], rb_valid['slap_score'])[0]:.3f}

ELITE TIER PERFORMANCE:
  WR 80+ SLAP: {int(wr_elite['nfl_hit24'].sum())}/{len(wr_elite)} Hit24 ({wr_elite['nfl_hit24'].mean()*100:.1f}%)
  RB 60+ SLAP: {int(rb_elite['nfl_hit24'].sum())}/{len(rb_elite)} Hit24 ({rb_elite['nfl_hit24'].mean()*100:.1f}%)

PRODUCTION COMPONENT VALUE:
  WR Breakout Age vs NFL PPG: r = {stats.pearsonr(wr_valid['production_score'], wr_valid['nfl_best_ppr'])[0]:.3f}
  RB Career Avg vs NFL PPG:   r = {stats.pearsonr(rb_valid_prod['production_score'], rb_valid_prod['nfl_best_ppr'])[0]:.3f}
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
