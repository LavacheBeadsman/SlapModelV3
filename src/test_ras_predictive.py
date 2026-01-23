"""
Test RAS Predictive Value for WRs

Compare RAS (0-10 scale) predictive power vs Speed Score
Handle missing RAS values appropriately
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load data
ras_merged = pd.read_csv('data/wr_ras_merged.csv')
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')

# Filter to WRs with NFL outcomes
wr_nfl = hit_rates[hit_rates['position'] == 'WR'].copy()

# Merge with RAS
analysis = wr_nfl.merge(
    ras_merged[['player_name', 'draft_year', 'RAS']],
    on=['player_name', 'draft_year'],
    how='left'
)

# Calculate PPG
analysis['best_ppg'] = analysis['best_ppr'] / 17

print("="*80)
print("RAS PREDICTIVE VALUE ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: Data completeness
# ============================================================================
print("\n1. DATA COMPLETENESS")
print("-"*50)

total_wrs = len(analysis)
has_ras = analysis['RAS'].notna().sum()
hit24_total = analysis['hit24'].sum()
hit24_with_ras = analysis[analysis['RAS'].notna()]['hit24'].sum()

print(f"   Total WRs: {total_wrs}")
print(f"   With RAS: {has_ras} ({has_ras/total_wrs*100:.1f}%)")
print(f"   Hit24 WRs: {hit24_total}")
print(f"   Hit24 with RAS: {hit24_with_ras} ({hit24_with_ras/hit24_total*100:.1f}%)")

# Show Hit24 WRs missing RAS
print("\n   Hit24 WRs MISSING RAS:")
missing_ras_hits = analysis[(analysis['hit24'] == 1) & (analysis['RAS'].isna())]
for _, row in missing_ras_hits.iterrows():
    print(f"   - {row['player_name']} ({row['draft_year']}, Pick {row['pick']})")

# ============================================================================
# STEP 2: Correlation analysis (RAS only)
# ============================================================================
print("\n" + "="*80)
print("2. RAS CORRELATION WITH NFL SUCCESS")
print("="*80)

valid_ras = analysis[analysis['RAS'].notna()].copy()

# Correlation with PPG
r_ppg, p_ppg = stats.pearsonr(valid_ras['RAS'], valid_ras['best_ppg'])
print(f"\n   RAS vs NFL PPG: r = {r_ppg:.3f}, p = {p_ppg:.4f}")

# Point-biserial with Hit24
r_hit24, p_hit24 = stats.pointbiserialr(valid_ras['hit24'], valid_ras['RAS'])
print(f"   RAS vs Hit24:   r = {r_hit24:.3f}, p = {p_hit24:.4f}")

# For comparison - draft pick
r_pick, _ = stats.pearsonr(valid_ras['pick'], valid_ras['best_ppg'])
print(f"\n   Draft Pick vs NFL PPG: r = {r_pick:.3f} (for comparison)")

# ============================================================================
# STEP 3: Quartile analysis
# ============================================================================
print("\n" + "="*80)
print("3. RAS QUARTILE ANALYSIS")
print("="*80)

q25 = valid_ras['RAS'].quantile(0.25)
q50 = valid_ras['RAS'].quantile(0.50)
q75 = valid_ras['RAS'].quantile(0.75)

print(f"\n   RAS quartile cutoffs: Q1={q25:.2f}, Q2={q50:.2f}, Q3={q75:.2f}")

for label, condition in [
    ("Top 25% RAS (elite)", valid_ras['RAS'] >= q75),
    ("2nd quartile", (valid_ras['RAS'] >= q50) & (valid_ras['RAS'] < q75)),
    ("3rd quartile", (valid_ras['RAS'] >= q25) & (valid_ras['RAS'] < q50)),
    ("Bottom 25% RAS", valid_ras['RAS'] < q25)
]:
    subset = valid_ras[condition]
    if len(subset) > 0:
        print(f"\n   {label} (n={len(subset)}):")
        print(f"     - Avg PPG: {subset['best_ppg'].mean():.1f}")
        print(f"     - Hit24 rate: {subset['hit24'].mean()*100:.1f}%")
        print(f"     - Avg RAS: {subset['RAS'].mean():.2f}")

# ============================================================================
# STEP 4: Hit24 WRs by RAS
# ============================================================================
print("\n" + "="*80)
print("4. HIT24 WRs BY RAS SCORE")
print("="*80)

hit24_wrs = analysis[analysis['hit24'] == 1].sort_values('RAS', ascending=False)

print("\n" + "-"*80)
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'RAS':>6} {'PPG':>6}")
print("-"*80)

for _, row in hit24_wrs.iterrows():
    ras_str = f"{row['RAS']:.2f}" if pd.notna(row['RAS']) else "N/A"
    print(f"{row['player_name']:<25} {row['draft_year']:>5} {row['pick']:>5} "
          f"{ras_str:>6} {row['best_ppg']:>6.1f}")

# Stats
has_ras_hits = hit24_wrs[hit24_wrs['RAS'].notna()]
if len(has_ras_hits) > 0:
    print(f"\n   Hit24 WRs with RAS - Avg RAS: {has_ras_hits['RAS'].mean():.2f}")
    print(f"   All WRs with RAS - Avg RAS: {valid_ras['RAS'].mean():.2f}")

# ============================================================================
# STEP 5: High RAS busts vs Low RAS hits
# ============================================================================
print("\n" + "="*80)
print("5. HIGH RAS BUSTS vs LOW RAS HITS")
print("="*80)

# High RAS (9+) who busted
high_ras_busts = valid_ras[(valid_ras['RAS'] >= 9) & (valid_ras['hit24'] == 0)]
print(f"\n   HIGH RAS (9+) BUSTS (n={len(high_ras_busts)}):")
for _, row in high_ras_busts.nlargest(10, 'RAS').iterrows():
    print(f"   - {row['player_name']:<22} RAS={row['RAS']:.2f}, Pick {row['pick']}, PPG={row['best_ppg']:.1f}")

# Low RAS (<7) who hit
low_ras_hits = valid_ras[(valid_ras['RAS'] < 7) & (valid_ras['hit24'] == 1)]
print(f"\n   LOW RAS (<7) HITS (n={len(low_ras_hits)}):")
for _, row in low_ras_hits.iterrows():
    print(f"   - {row['player_name']:<22} RAS={row['RAS']:.2f}, Pick {row['pick']}, PPG={row['best_ppg']:.1f}")

# ============================================================================
# STEP 6: RAS by draft round
# ============================================================================
print("\n" + "="*80)
print("6. RAS BY DRAFT ROUND")
print("="*80)

valid_ras['round'] = pd.cut(valid_ras['pick'],
                            bins=[0, 32, 64, 100, 150, 300],
                            labels=['Rd 1', 'Rd 2', 'Rd 3', 'Rd 4-5', 'Rd 6-7'])

print("\n" + "-"*60)
print(f"{'Round':<10} {'Count':>6} {'Avg RAS':>10} {'Hit24 %':>10}")
print("-"*60)

for rd in ['Rd 1', 'Rd 2', 'Rd 3', 'Rd 4-5', 'Rd 6-7']:
    subset = valid_ras[valid_ras['round'] == rd]
    if len(subset) > 0:
        print(f"{rd:<10} {len(subset):>6} {subset['RAS'].mean():>10.2f} {subset['hit24'].mean()*100:>9.1f}%")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
RAS Predictive Power:
  - RAS vs NFL PPG: r = {r_ppg:.3f} (p = {p_ppg:.4f})
  - RAS vs Hit24:   r = {r_hit24:.3f} (p = {p_hit24:.4f})

Interpretation:
  - {"POSITIVE" if r_ppg > 0 else "NEGATIVE"} correlation with NFL success
  - {"Statistically significant" if p_ppg < 0.05 else "NOT statistically significant"} (p {"<" if p_ppg < 0.05 else ">"} 0.05)

Missing Data Issue:
  - 4 Hit24 WRs missing RAS (Waddle, Smith, London, J. Williams)
  - All 4 are 1st-round picks who likely skipped combine testing
  - Need strategy to handle missing athletic data

Options for Missing RAS:
  A) Use draft capital as proxy (NFL valued them highly despite no testing)
  B) Assign average RAS (neutral assumption)
  C) Impute based on similar players (school/draft slot)
  D) Exclude athletic component for these players (use DC + Breakout only)
""")
