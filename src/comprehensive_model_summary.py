"""
Comprehensive SLAP Score Model Summary
Generate all validation metrics and analysis for final review
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD ALL DATA
# ============================================================================
print("Loading data...")

# Core data
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')
backtest = pd.read_csv('data/backtest_college_stats.csv')
breakout_ages = pd.read_csv('data/wr_breakout_age_scores.csv')
ras_data = pd.read_csv('data/wr_ras_merged.csv')

# Filter to WRs
wr_nfl = hit_rates[hit_rates['position'] == 'WR'].copy()
rb_nfl = hit_rates[hit_rates['position'] == 'RB'].copy()

# Age score mapping for WRs
AGE_SCORES = {
    18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 15, 25: 10,
}

def get_age_score(breakout_age):
    if pd.isna(breakout_age):
        return 25
    return AGE_SCORES.get(int(breakout_age), 10)

# ============================================================================
# BUILD WR ANALYSIS DATASET
# ============================================================================
print("Building WR analysis dataset...")

# Merge WR data
wr_analysis = wr_nfl.merge(
    breakout_ages[['player_name', 'draft_year', 'breakout_age', 'peak_dominator']],
    on=['player_name', 'draft_year'],
    how='left'
)

wr_analysis = wr_analysis.merge(
    ras_data[['player_name', 'draft_year', 'RAS']],
    on=['player_name', 'draft_year'],
    how='left'
)

# Calculate scores
wr_analysis['best_ppg'] = wr_analysis['best_ppr'] / 17
wr_analysis['breakout_score'] = wr_analysis['breakout_age'].apply(get_age_score)

# Draft capital score
def calc_dc_score(pick):
    if pd.isna(pick) or pick <= 0:
        return np.nan
    return 1 / np.sqrt(pick)

wr_analysis['dc_raw'] = wr_analysis['pick'].apply(calc_dc_score)

# Filter to 2020-2024 with valid data
wr_valid = wr_analysis[
    (wr_analysis['draft_year'].isin([2020, 2021, 2022, 2023, 2024])) &
    (wr_analysis['dc_raw'].notna()) &
    (wr_analysis['breakout_score'].notna())
].copy()

# Fill missing RAS with average
ras_avg = wr_valid['RAS'].mean()
wr_valid['RAS_filled'] = wr_valid['RAS'].fillna(ras_avg)

# Normalize scores (z-score then scale to ~50 mean, ~15 std)
def normalize_score(series):
    z = (series - series.mean()) / series.std()
    return 50 + z * 15

wr_valid['dc_norm'] = normalize_score(wr_valid['dc_raw'])
wr_valid['breakout_norm'] = normalize_score(wr_valid['breakout_score'])
wr_valid['ras_norm'] = normalize_score(wr_valid['RAS_filled'])

# Calculate SLAP score (50% DC, 35% Breakout, 15% Athletic)
wr_valid['slap_score'] = (
    wr_valid['dc_norm'] * 0.50 +
    wr_valid['breakout_norm'] * 0.35 +
    wr_valid['ras_norm'] * 0.15
)

# DC-only baseline
wr_valid['dc_only_score'] = wr_valid['dc_norm']

# Delta (SLAP vs DC-only)
wr_valid['delta'] = wr_valid['slap_score'] - wr_valid['dc_only_score']

print(f"WR sample size: {len(wr_valid)} players (2020-2024)")

# ============================================================================
# PART 3: WR VALIDATION METRICS
# ============================================================================
print("\n" + "="*80)
print("PART 3: WR MODEL VALIDATION RESULTS")
print("="*80)

# Correlations
spearman_slap, p_spearman = stats.spearmanr(wr_valid['slap_score'], wr_valid['best_ppg'])
pearson_slap, p_pearson = stats.pearsonr(wr_valid['slap_score'], wr_valid['best_ppg'])
spearman_dc, _ = stats.spearmanr(wr_valid['dc_norm'], wr_valid['best_ppg'])
pearson_dc, _ = stats.pearsonr(wr_valid['dc_norm'], wr_valid['best_ppg'])

print(f"\nCORRELATION WITH NFL PPG:")
print("-"*60)
print(f"{'Metric':<25} {'SLAP Score':>15} {'DC-Only':>15}")
print("-"*60)
print(f"{'Spearman correlation':<25} {spearman_slap:>15.3f} {spearman_dc:>15.3f}")
print(f"{'Pearson correlation':<25} {pearson_slap:>15.3f} {pearson_dc:>15.3f}")

# Point-biserial with Hit24/Hit12
r_slap_hit24, _ = stats.pointbiserialr(wr_valid['hit24'], wr_valid['slap_score'])
r_dc_hit24, _ = stats.pointbiserialr(wr_valid['hit24'], wr_valid['dc_norm'])
r_slap_hit12, _ = stats.pointbiserialr(wr_valid['hit12'], wr_valid['slap_score'])
r_dc_hit12, _ = stats.pointbiserialr(wr_valid['hit12'], wr_valid['dc_norm'])

print(f"\nCORRELATION WITH HIT THRESHOLDS:")
print("-"*60)
print(f"{'Hit24 correlation':<25} {r_slap_hit24:>15.3f} {r_dc_hit24:>15.3f}")
print(f"{'Hit12 correlation':<25} {r_slap_hit12:>15.3f} {r_dc_hit12:>15.3f}")

# AUC
auc_slap = roc_auc_score(wr_valid['hit24'], wr_valid['slap_score'])
auc_dc = roc_auc_score(wr_valid['hit24'], wr_valid['dc_norm'])

print(f"\nAUC-ROC FOR HIT24 PREDICTION:")
print("-"*60)
print(f"{'AUC-ROC':<25} {auc_slap:>15.3f} {auc_dc:>15.3f}")

# Hit rates by SLAP tier
print(f"\nHIT RATES BY SLAP TIER:")
print("-"*70)

slap_q75 = wr_valid['slap_score'].quantile(0.75)
slap_q50 = wr_valid['slap_score'].quantile(0.50)
slap_q25 = wr_valid['slap_score'].quantile(0.25)

tiers = [
    ("Top 25% SLAP", wr_valid['slap_score'] >= slap_q75),
    ("2nd quartile", (wr_valid['slap_score'] >= slap_q50) & (wr_valid['slap_score'] < slap_q75)),
    ("3rd quartile", (wr_valid['slap_score'] >= slap_q25) & (wr_valid['slap_score'] < slap_q50)),
    ("Bottom 25%", wr_valid['slap_score'] < slap_q25),
]

print(f"{'Tier':<20} {'Count':>6} {'Hit24':>6} {'Hit24%':>8} {'Hit12':>6} {'Hit12%':>8} {'Avg PPG':>10}")
print("-"*70)

for name, condition in tiers:
    subset = wr_valid[condition]
    h24 = subset['hit24'].sum()
    h12 = subset['hit12'].sum()
    h24_pct = subset['hit24'].mean() * 100
    h12_pct = subset['hit12'].mean() * 100
    avg_ppg = subset['best_ppg'].mean()
    print(f"{name:<20} {len(subset):>6} {h24:>6} {h24_pct:>7.1f}% {h12:>6} {h12_pct:>7.1f}% {avg_ppg:>10.1f}")

# Compare to DC-only tiers
print(f"\nHIT RATES BY DC-ONLY TIER (for comparison):")
print("-"*70)

dc_q75 = wr_valid['dc_norm'].quantile(0.75)
dc_q50 = wr_valid['dc_norm'].quantile(0.50)
dc_q25 = wr_valid['dc_norm'].quantile(0.25)

dc_tiers = [
    ("Top 25% DC", wr_valid['dc_norm'] >= dc_q75),
    ("2nd quartile", (wr_valid['dc_norm'] >= dc_q50) & (wr_valid['dc_norm'] < dc_q75)),
    ("3rd quartile", (wr_valid['dc_norm'] >= dc_q25) & (wr_valid['dc_norm'] < dc_q50)),
    ("Bottom 25%", wr_valid['dc_norm'] < dc_q25),
]

print(f"{'Tier':<20} {'Count':>6} {'Hit24':>6} {'Hit24%':>8} {'Hit12':>6} {'Hit12%':>8} {'Avg PPG':>10}")
print("-"*70)

for name, condition in dc_tiers:
    subset = wr_valid[condition]
    h24 = subset['hit24'].sum()
    h12 = subset['hit12'].sum()
    h24_pct = subset['hit24'].mean() * 100
    h12_pct = subset['hit12'].mean() * 100
    avg_ppg = subset['best_ppg'].mean()
    print(f"{name:<20} {len(subset):>6} {h24:>6} {h24_pct:>7.1f}% {h12:>6} {h12_pct:>7.1f}% {avg_ppg:>10.1f}")

# ============================================================================
# PART 4: WHAT THE MODEL DOES WELL
# ============================================================================
print("\n" + "="*80)
print("PART 4: WHAT THE MODEL DOES WELL")
print("="*80)

# Where SLAP adds value: positive delta players who hit
positive_delta_hits = wr_valid[(wr_valid['delta'] > 2) & (wr_valid['hit24'] == 1)].sort_values('delta', ascending=False)

print(f"\nPOSITIVE DELTA HITS (Model liked more than draft slot, and they hit):")
print("-"*90)
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>6} {'Delta':>7} {'PPG':>6}")
print("-"*90)

for _, row in positive_delta_hits.head(10).iterrows():
    delta_str = f"+{row['delta']:.1f}"
    print(f"{row['player_name']:<25} {row['draft_year']:>5} {row['pick']:>5.0f} "
          f"{row['slap_score']:>6.1f} {row['dc_norm']:>6.1f} {delta_str:>7} {row['best_ppg']:>6.1f}")

# Top SLAP hits
print(f"\nTOP 10 SLAP SCORES WHO HIT:")
print("-"*90)
top_slap_hits = wr_valid[wr_valid['hit24'] == 1].nlargest(10, 'slap_score')
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'BrkAge':>7} {'RAS':>6} {'PPG':>6}")
print("-"*90)

for _, row in top_slap_hits.iterrows():
    age_str = f"{row['breakout_age']:.0f}" if pd.notna(row['breakout_age']) else "N/A"
    ras_str = f"{row['RAS']:.1f}" if pd.notna(row['RAS']) else "N/A"
    print(f"{row['player_name']:<25} {row['draft_year']:>5} {row['pick']:>5.0f} "
          f"{row['slap_score']:>6.1f} {age_str:>7} {ras_str:>6} {row['best_ppg']:>6.1f}")

# ============================================================================
# PART 5: KNOWN LIMITATIONS
# ============================================================================
print("\n" + "="*80)
print("PART 5: KNOWN LIMITATIONS")
print("="*80)

# High SLAP busts (top 25% SLAP who didn't hit)
high_slap_busts = wr_valid[(wr_valid['slap_score'] >= slap_q75) & (wr_valid['hit24'] == 0)].sort_values('slap_score', ascending=False)

print(f"\nHIGH SLAP BUSTS (Top 25% SLAP who didn't hit): {len(high_slap_busts)} players")
print("-"*90)
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'BrkAge':>7} {'RAS':>6} {'PPG':>6}")
print("-"*90)

for _, row in high_slap_busts.head(10).iterrows():
    age_str = f"{row['breakout_age']:.0f}" if pd.notna(row['breakout_age']) else "N/A"
    ras_str = f"{row['RAS']:.1f}" if pd.notna(row['RAS']) else "N/A"
    print(f"{row['player_name']:<25} {row['draft_year']:>5} {row['pick']:>5.0f} "
          f"{row['slap_score']:>6.1f} {age_str:>7} {ras_str:>6} {row['best_ppg']:>6.1f}")

# Low SLAP hits (bottom 50% who hit)
low_slap_hits = wr_valid[(wr_valid['slap_score'] < slap_q50) & (wr_valid['hit24'] == 1)].sort_values('slap_score')

print(f"\nLOW SLAP HITS (Bottom 50% SLAP who hit anyway): {len(low_slap_hits)} players")
print("-"*90)

if len(low_slap_hits) > 0:
    print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'BrkAge':>7} {'RAS':>6} {'PPG':>6}")
    print("-"*90)
    for _, row in low_slap_hits.iterrows():
        age_str = f"{row['breakout_age']:.0f}" if pd.notna(row['breakout_age']) else "N/A"
        ras_str = f"{row['RAS']:.1f}" if pd.notna(row['RAS']) else "N/A"
        print(f"{row['player_name']:<25} {row['draft_year']:>5} {row['pick']:>5.0f} "
              f"{row['slap_score']:>6.1f} {age_str:>7} {ras_str:>6} {row['best_ppg']:>6.1f}")
else:
    print("  None found")

# Negative delta misses (model disliked them but they hit anyway)
negative_delta_hits = wr_valid[(wr_valid['delta'] < -2) & (wr_valid['hit24'] == 1)].sort_values('delta')

print(f"\nNEGATIVE DELTA HITS (Model disliked, but they hit anyway): {len(negative_delta_hits)} players")
print("-"*90)

if len(negative_delta_hits) > 0:
    print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>6} {'Delta':>7} {'PPG':>6}")
    print("-"*90)
    for _, row in negative_delta_hits.iterrows():
        delta_str = f"{row['delta']:.1f}"
        print(f"{row['player_name']:<25} {row['draft_year']:>5} {row['pick']:>5.0f} "
              f"{row['slap_score']:>6.1f} {row['dc_norm']:>6.1f} {delta_str:>7} {row['best_ppg']:>6.1f}")
else:
    print("  None found")

# Missing data issues
print(f"\nMISSING DATA ISSUES:")
print("-"*60)
print(f"  WRs missing breakout age data: {wr_valid['breakout_age'].isna().sum()}")
print(f"  WRs missing RAS data: {wr_valid['RAS'].isna().sum()}")

# Show which Hit24 WRs are missing RAS
missing_ras_hits = wr_valid[(wr_valid['RAS'].isna()) & (wr_valid['hit24'] == 1)]
if len(missing_ras_hits) > 0:
    print(f"\n  Hit24 WRs missing RAS ({len(missing_ras_hits)}):")
    for _, row in missing_ras_hits.iterrows():
        print(f"    - {row['player_name']} ({row['draft_year']}, Pick {row['pick']:.0f})")

# ============================================================================
# COMPONENT CONTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("COMPONENT CONTRIBUTION ANALYSIS")
print("="*80)

# Correlation of each component with NFL PPG
r_dc, _ = stats.pearsonr(wr_valid['dc_norm'], wr_valid['best_ppg'])
r_breakout, _ = stats.pearsonr(wr_valid['breakout_norm'], wr_valid['best_ppg'])
r_ras, _ = stats.pearsonr(wr_valid['ras_norm'], wr_valid['best_ppg'])

print(f"\nINDIVIDUAL COMPONENT CORRELATIONS WITH NFL PPG:")
print("-"*50)
print(f"  Draft Capital (50% weight): r = {r_dc:.3f}")
print(f"  Breakout Age (35% weight):  r = {r_breakout:.3f}")
print(f"  RAS (15% weight):           r = {r_ras:.3f}")
print(f"  SLAP Combined:              r = {pearson_slap:.3f}")

# Does SLAP beat sum of parts?
weighted_avg_r = r_dc * 0.50 + r_breakout * 0.35 + r_ras * 0.15
print(f"\n  Weighted avg of components: r = {weighted_avg_r:.3f}")
print(f"  SLAP actually achieves:     r = {pearson_slap:.3f}")

# ============================================================================
# DOES SLAP ADD VALUE OVER DC-ONLY?
# ============================================================================
print("\n" + "="*80)
print("SLAP vs DC-ONLY: DOES THE MODEL ADD VALUE?")
print("="*80)

improvement_spearman = spearman_slap - spearman_dc
improvement_pearson = pearson_slap - pearson_dc
improvement_auc = auc_slap - auc_dc

print(f"\nIMPROVEMENT OVER DRAFT-CAPITAL-ONLY:")
print("-"*50)
print(f"  Spearman improvement: {improvement_spearman:+.3f}")
print(f"  Pearson improvement:  {improvement_pearson:+.3f}")
print(f"  AUC improvement:      {improvement_auc:+.3f}")

if improvement_spearman > 0:
    print(f"\n  ✓ SLAP improves Spearman correlation by {improvement_spearman:.3f}")
else:
    print(f"\n  ✗ SLAP does NOT improve Spearman correlation")

# Compare top quartile hit rates
slap_top_hit = wr_valid[wr_valid['slap_score'] >= slap_q75]['hit24'].mean() * 100
dc_top_hit = wr_valid[wr_valid['dc_norm'] >= dc_q75]['hit24'].mean() * 100

print(f"\n  Top 25% SLAP hit rate: {slap_top_hit:.1f}%")
print(f"  Top 25% DC hit rate:   {dc_top_hit:.1f}%")
print(f"  Difference:            {slap_top_hit - dc_top_hit:+.1f}%")

# ============================================================================
# SAMPLE SIZE AND STATISTICAL POWER
# ============================================================================
print("\n" + "="*80)
print("SAMPLE SIZE AND STATISTICAL POWER")
print("="*80)

print(f"\nSAMPLE BREAKDOWN:")
print("-"*50)
print(f"  Total WRs (2020-2024): {len(wr_valid)}")
print(f"  Hit24 WRs: {wr_valid['hit24'].sum()} ({wr_valid['hit24'].mean()*100:.1f}%)")
print(f"  Hit12 WRs: {wr_valid['hit12'].sum()} ({wr_valid['hit12'].mean()*100:.1f}%)")

print(f"\n  By draft year:")
for year in [2020, 2021, 2022, 2023, 2024]:
    year_data = wr_valid[wr_valid['draft_year'] == year]
    print(f"    {year}: {len(year_data)} WRs, {year_data['hit24'].sum()} hits")

# Statistical significance
print(f"\n  Spearman p-value: {p_spearman:.4f} ({'significant' if p_spearman < 0.05 else 'NOT significant'})")
print(f"  Pearson p-value:  {p_pearson:.4f} ({'significant' if p_pearson < 0.05 else 'NOT significant'})")

# ============================================================================
# FINAL SUMMARY STATS
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY STATISTICS")
print("="*80)

total_hits = wr_valid['hit24'].sum()
total_wrs = len(wr_valid)
base_rate = total_hits / total_wrs

# Top 10 SLAP hit rate
top10_slap = wr_valid.nlargest(10, 'slap_score')
top10_hit_rate = top10_slap['hit24'].mean() * 100

# Top 20 SLAP hit rate
top20_slap = wr_valid.nlargest(20, 'slap_score')
top20_hit_rate = top20_slap['hit24'].mean() * 100

print(f"""
KEY METRICS:
  - Sample size: {total_wrs} WRs (2020-2024)
  - Base hit rate: {base_rate*100:.1f}% (random chance)

  - Top 10 SLAP hit rate: {top10_hit_rate:.1f}%
  - Top 20 SLAP hit rate: {top20_hit_rate:.1f}%
  - Top 25% SLAP hit rate: {slap_top_hit:.1f}%

  - Spearman correlation: {spearman_slap:.3f}
  - AUC-ROC: {auc_slap:.3f}

  - Improvement over DC-only: {improvement_spearman:+.3f} (Spearman)
""")

# Save detailed results
wr_valid.to_csv('output/wr_validation_detailed.csv', index=False)
print("Detailed results saved to: output/wr_validation_detailed.csv")
