"""
Year-by-Year SLAP Model Validation
"""

import pandas as pd
import numpy as np

# Load the complete database
db = pd.read_csv('output/slap_complete_database_v4.csv')

print("="*80)
print("PART 5: EXTERNAL VALIDATION - YEAR BY YEAR")
print("="*80)

# Filter to backtest data with outcomes
backtest = db[db['data_type'] == 'backtest'].copy()
wr = backtest[backtest['position'] == 'WR']
rb = backtest[backtest['position'] == 'RB']

print(f"\nBacktest WRs: {len(wr)}")
print(f"Backtest RBs: {len(rb)}")

# Calculate DC-only score for comparison
def dc_score(pick):
    return 100 - 2.40 * (pick**0.62 - 1)

wr['dc_only'] = wr['pick'].apply(dc_score)
rb['dc_only'] = rb['pick'].apply(dc_score)

# ===== WR YEAR BY YEAR =====
print("\n" + "-"*80)
print("WR YEAR-BY-YEAR VALIDATION")
print("-"*80)

print("\n| Year | Top 5 SLAP | Hit Rate | Top 5 DC | Hit Rate | SLAP Better? |")
print("|------|------------|----------|----------|----------|--------------|")

for year in sorted(wr['draft_year'].unique()):
    yr_df = wr[wr['draft_year'] == year].copy()
    if len(yr_df) < 5:
        continue

    # Top 5 by SLAP
    top5_slap = yr_df.nlargest(5, 'slap_score')
    slap_hits = top5_slap['nfl_hit24'].sum()

    # Top 5 by DC (earliest picks)
    top5_dc = yr_df.nsmallest(5, 'pick')
    dc_hits = top5_dc['nfl_hit24'].sum()

    # Get player names
    slap_names = ', '.join(top5_slap['player_name'].str.split().str[0].tolist())
    dc_names = ', '.join(top5_dc['player_name'].str.split().str[0].tolist())

    better = "YES" if slap_hits > dc_hits else ("TIE" if slap_hits == dc_hits else "NO")

    print(f"| {year} | {slap_hits}/5 ({slap_hits*20}%) | {slap_names[:30]:30} | {dc_hits}/5 ({dc_hits*20}%) | {dc_names[:30]:30} | {better} |")

# Show specific years with top 5 players
print("\n\n--- DETAILED TOP 5 BY YEAR ---")
for year in [2019, 2020, 2021, 2022, 2023]:
    yr_df = wr[wr['draft_year'] == year].copy()
    if len(yr_df) < 5:
        continue

    print(f"\n{year} WR CLASS:")
    print("  TOP 5 BY SLAP:")
    top5_slap = yr_df.nlargest(5, 'slap_score')
    for _, row in top5_slap.iterrows():
        hit = "✓ HIT" if row['nfl_hit24'] == 1 else "✗ MISS"
        print(f"    {row['player_name']:25} SLAP={row['slap_score']:.1f}  Pick={row['pick']:3}  {hit}")

    print("  TOP 5 BY DC (Draft Order):")
    top5_dc = yr_df.nsmallest(5, 'pick')
    for _, row in top5_dc.iterrows():
        hit = "✓ HIT" if row['nfl_hit24'] == 1 else "✗ MISS"
        print(f"    {row['player_name']:25} SLAP={row['slap_score']:.1f}  Pick={row['pick']:3}  {hit}")

# ===== RB YEAR BY YEAR =====
print("\n\n" + "-"*80)
print("RB YEAR-BY-YEAR VALIDATION")
print("-"*80)

print("\n| Year | Top 5 SLAP | Hit Rate | Top 5 DC | Hit Rate | SLAP Better? |")
print("|------|------------|----------|----------|----------|--------------|")

for year in sorted(rb['draft_year'].unique()):
    yr_df = rb[rb['draft_year'] == year].copy()
    if len(yr_df) < 5:
        continue

    # Top 5 by SLAP
    top5_slap = yr_df.nlargest(5, 'slap_score')
    slap_hits = top5_slap['nfl_hit24'].sum()

    # Top 5 by DC (earliest picks)
    top5_dc = yr_df.nsmallest(5, 'pick')
    dc_hits = top5_dc['nfl_hit24'].sum()

    better = "YES" if slap_hits > dc_hits else ("TIE" if slap_hits == dc_hits else "NO")

    print(f"| {year} | {slap_hits}/5 ({slap_hits*20}%) | {dc_hits}/5 ({dc_hits*20}%) | {better} |")

# ===== CORRELATION COMPARISON =====
print("\n\n" + "-"*80)
print("CORRELATION ANALYSIS")
print("-"*80)

# WR correlations
wr_valid = wr[wr['nfl_best_ppr'] > 0]
if len(wr_valid) > 10:
    slap_corr = np.corrcoef(wr_valid['slap_score'], wr_valid['nfl_best_ppr'])[0,1]
    dc_corr = np.corrcoef(wr_valid['dc_only'], wr_valid['nfl_best_ppr'])[0,1]

    print(f"\nWR Correlation with Best Season PPR (n={len(wr_valid)}):")
    print(f"  SLAP Score: r = {slap_corr:.3f}")
    print(f"  DC Only:    r = {dc_corr:.3f}")
    print(f"  SLAP Edge:  {slap_corr - dc_corr:+.3f}")

# RB correlations
rb_valid = rb[rb['nfl_best_ppr'] > 0]
if len(rb_valid) > 10:
    slap_corr = np.corrcoef(rb_valid['slap_score'], rb_valid['nfl_best_ppr'])[0,1]
    dc_corr = np.corrcoef(rb_valid['dc_only'], rb_valid['nfl_best_ppr'])[0,1]

    print(f"\nRB Correlation with Best Season PPR (n={len(rb_valid)}):")
    print(f"  SLAP Score: r = {slap_corr:.3f}")
    print(f"  DC Only:    r = {dc_corr:.3f}")
    print(f"  SLAP Edge:  {slap_corr - dc_corr:+.3f}")

# ===== HIT RATE BY SLAP TIER =====
print("\n\n" + "-"*80)
print("HIT RATE BY SLAP TIER")
print("-"*80)

print("\nWR Hit Rates by SLAP Tier:")
for low, high, label in [(80, 100, "80-100"), (70, 80, "70-79"), (60, 70, "60-69"), (50, 60, "50-59"), (0, 50, "<50")]:
    tier = wr[(wr['slap_score'] >= low) & (wr['slap_score'] < high)]
    if len(tier) > 0:
        hit_rate = tier['nfl_hit24'].mean() * 100
        print(f"  SLAP {label}: {tier['nfl_hit24'].sum()}/{len(tier)} ({hit_rate:.1f}%)")

print("\nRB Hit Rates by SLAP Tier:")
for low, high, label in [(80, 100, "80-100"), (70, 80, "70-79"), (60, 70, "60-69"), (50, 60, "50-59"), (0, 50, "<50")]:
    tier = rb[(rb['slap_score'] >= low) & (rb['slap_score'] < high)]
    if len(tier) > 0:
        hit_rate = tier['nfl_hit24'].mean() * 100
        print(f"  SLAP {label}: {tier['nfl_hit24'].sum()}/{len(tier)} ({hit_rate:.1f}%)")

# ===== BIGGEST WINS AND LOSSES =====
print("\n\n" + "-"*80)
print("BIGGEST SLAP WINS (High delta, became hit)")
print("-"*80)

wr_hits = wr[wr['nfl_hit24'] == 1].copy()
wr_hits_sorted = wr_hits.nlargest(10, 'delta_vs_dc')
print("\nWRs where SLAP liked them MORE than DC, and they hit:")
for _, row in wr_hits_sorted.iterrows():
    print(f"  {row['player_name']:25} Pick={row['pick']:3}  SLAP={row['slap_score']:.1f}  Delta={row['delta_vs_dc']:+.1f}")

print("\n" + "-"*80)
print("BIGGEST SLAP MISSES (High SLAP score, busted)")
print("-"*80)

wr_misses = wr[wr['nfl_hit24'] == 0].copy()
wr_misses_sorted = wr_misses.nlargest(10, 'slap_score')
print("\nWRs with highest SLAP scores who MISSED:")
for _, row in wr_misses_sorted.iterrows():
    print(f"  {row['player_name']:25} Pick={row['pick']:3}  SLAP={row['slap_score']:.1f}  Delta={row['delta_vs_dc']:+.1f}")

# ===== SUMMARY =====
print("\n\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

total_wr_years = len(wr['draft_year'].unique())
slap_better_wr = 0
dc_better_wr = 0
tie_wr = 0

for year in wr['draft_year'].unique():
    yr_df = wr[wr['draft_year'] == year]
    if len(yr_df) < 5:
        continue
    top5_slap = yr_df.nlargest(5, 'slap_score')
    top5_dc = yr_df.nsmallest(5, 'pick')
    slap_hits = top5_slap['nfl_hit24'].sum()
    dc_hits = top5_dc['nfl_hit24'].sum()
    if slap_hits > dc_hits:
        slap_better_wr += 1
    elif dc_hits > slap_hits:
        dc_better_wr += 1
    else:
        tie_wr += 1

print(f"\nWR Top-5 Accuracy by Year:")
print(f"  SLAP beats DC: {slap_better_wr} years")
print(f"  DC beats SLAP: {dc_better_wr} years")
print(f"  Tie: {tie_wr} years")
