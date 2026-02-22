"""
Diagnostic tests to understand why Dominator Rating isn't showing predictive value.
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load data
breakout = pd.read_csv('data/wr_breakout_age_scores.csv')
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')
college_stats = pd.read_csv('data/backtest_college_stats.csv')

# Filter to WRs only in hit_rates
wr_nfl = hit_rates[hit_rates['position'] == 'WR'].copy()

# Merge breakout data with NFL data
merged = wr_nfl.merge(
    breakout[['player_name', 'draft_year', 'peak_dominator', 'breakout_age']],
    on=['player_name', 'draft_year'],
    how='left'
)

# Note: 'pick' column already exists in hit_rates

# Calculate PPG (best_ppr / 17 games)
merged['best_ppg'] = merged['best_ppr'] / 17

# Filter to 2020-2024 drafts only (have NFL data)
merged_with_nfl = merged[merged['draft_year'].isin([2020, 2021, 2022, 2023, 2024])].copy()

print("="*80)
print("DIAGNOSTIC TESTS: Why Isn't Dominator Rating Showing Predictive Value?")
print("="*80)

# ============================================================================
# TEST 4: Data Completeness Check (run first to establish sample size)
# ============================================================================
print("\n" + "="*80)
print("TEST 4: DATA COMPLETENESS CHECK")
print("="*80)

total_wrs = len(merged_with_nfl)
with_dominator = merged_with_nfl['peak_dominator'].notna().sum()
with_nfl_data = merged_with_nfl['best_ppr'].notna().sum()
with_both = merged_with_nfl[
    (merged_with_nfl['peak_dominator'].notna()) &
    (merged_with_nfl['best_ppr'].notna())
].shape[0]

print(f"\nTotal WRs drafted 2020-2024: {total_wrs}")
print(f"WRs with dominator data: {with_dominator} ({with_dominator/total_wrs*100:.1f}%)")
print(f"WRs with NFL fantasy data: {with_nfl_data} ({with_nfl_data/total_wrs*100:.1f}%)")
print(f"WRs with BOTH dominator AND NFL data: {with_both} ({with_both/total_wrs*100:.1f}%)")

# Filter to only WRs with both
analysis_df = merged_with_nfl[
    (merged_with_nfl['peak_dominator'].notna()) &
    (merged_with_nfl['best_ppr'].notna()) &
    (merged_with_nfl['best_ppr'] > 0)  # Must have played
].copy()

print(f"\nUsing {len(analysis_df)} WRs for analysis (have both dominator AND played in NFL)")

# Check if sample is biased
hits_in_sample = analysis_df['hit24'].sum()
total_hits = merged_with_nfl['hit24'].sum()
print(f"\nHit24 in analysis sample: {hits_in_sample}")
print(f"Total Hit24 in full dataset: {total_hits}")
print(f"Are we missing any hits? {'NO - all hits included' if hits_in_sample >= total_hits else 'YES - sample is biased!'}")

# ============================================================================
# TEST 1: Raw Correlations
# ============================================================================
print("\n" + "="*80)
print("TEST 1: RAW CORRELATIONS (no modeling)")
print("="*80)

# Correlation: peak_dominator vs best_ppr
r_dom_ppr, p_dom_ppr = stats.pearsonr(
    analysis_df['peak_dominator'],
    analysis_df['best_ppr']
)
print(f"\n1. peak_dominator vs best_ppr (best season total):")
print(f"   r = {r_dom_ppr:.3f}, p = {p_dom_ppr:.4f}")

# Correlation: peak_dominator vs best_ppg
r_dom_ppg, p_dom_ppg = stats.pearsonr(
    analysis_df['peak_dominator'],
    analysis_df['best_ppg']
)
print(f"\n2. peak_dominator vs best_ppg (best season per-game):")
print(f"   r = {r_dom_ppg:.3f}, p = {p_dom_ppg:.4f}")

# Correlation: breakout_age vs best_ppr
age_df = analysis_df[analysis_df['breakout_age'].notna()]
if len(age_df) > 10:
    r_age_ppr, p_age_ppr = stats.pearsonr(
        age_df['breakout_age'],
        age_df['best_ppr']
    )
    print(f"\n3. breakout_age vs best_ppr:")
    print(f"   r = {r_age_ppr:.3f}, p = {p_age_ppr:.4f}")
    print(f"   (Negative = younger breakouts do better)")

# Correlation: pick vs best_ppr (for comparison)
r_pick_ppr, p_pick_ppr = stats.pearsonr(
    analysis_df['pick'],
    analysis_df['best_ppr']
)
print(f"\n4. pick vs best_ppr (for comparison):")
print(f"   r = {r_pick_ppr:.3f}, p = {p_pick_ppr:.4f}")
print(f"   (Draft capital for reference - should be negative since lower pick = better)")

# ============================================================================
# TEST 2: Scatterplot Data - Top 30 WRs by peak_dominator
# ============================================================================
print("\n" + "="*80)
print("TEST 2: TOP 30 WRs BY PEAK DOMINATOR (eyeball test)")
print("="*80)

top30 = analysis_df.nlargest(30, 'peak_dominator')[
    ['player_name', 'draft_year', 'pick', 'peak_dominator', 'breakout_age', 'best_ppg', 'hit24']
].copy()
top30['hit24'] = top30['hit24'].map({1: 'YES', 0: 'no'})

print("\n" + "-"*95)
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'Dom%':>7} {'BrkAge':>7} {'PPG':>7} {'Hit24':>6}")
print("-"*95)
for _, row in top30.iterrows():
    age_str = f"{row['breakout_age']:.0f}" if pd.notna(row['breakout_age']) else "N/A"
    print(f"{row['player_name']:<25} {row['draft_year']:>5} {row['pick']:>5} "
          f"{row['peak_dominator']:>6.1f}% {age_str:>7} {row['best_ppg']:>7.1f} {row['hit24']:>6}")

hits_in_top30 = (top30['hit24'] == 'YES').sum()
print(f"\nHits in top 30 dominator WRs: {hits_in_top30}/30 ({hits_in_top30/30*100:.1f}%)")

# ============================================================================
# TEST 3: Segment Analysis
# ============================================================================
print("\n" + "="*80)
print("TEST 3: SEGMENT ANALYSIS (quartiles)")
print("="*80)

# Calculate quartiles
q25 = analysis_df['peak_dominator'].quantile(0.25)
q75 = analysis_df['peak_dominator'].quantile(0.75)

top_quartile = analysis_df[analysis_df['peak_dominator'] >= q75]
bottom_quartile = analysis_df[analysis_df['peak_dominator'] <= q25]

print(f"\nDominator cutoffs: Bottom 25% <= {q25:.1f}%, Top 25% >= {q75:.1f}%")

print(f"\nTOP 25% by dominator (n={len(top_quartile)}):")
print(f"  - Average best PPG: {top_quartile['best_ppg'].mean():.1f}")
print(f"  - Hit24 rate: {top_quartile['hit24'].mean()*100:.1f}%")
print(f"  - Average draft pick: {top_quartile['pick'].mean():.0f}")

print(f"\nBOTTOM 25% by dominator (n={len(bottom_quartile)}):")
print(f"  - Average best PPG: {bottom_quartile['best_ppg'].mean():.1f}")
print(f"  - Hit24 rate: {bottom_quartile['hit24'].mean()*100:.1f}%")
print(f"  - Average draft pick: {bottom_quartile['pick'].mean():.0f}")

diff_ppg = top_quartile['best_ppg'].mean() - bottom_quartile['best_ppg'].mean()
diff_hit = (top_quartile['hit24'].mean() - bottom_quartile['hit24'].mean()) * 100
print(f"\nDIFFERENCE (top - bottom):")
print(f"  - PPG difference: {diff_ppg:+.1f}")
print(f"  - Hit rate difference: {diff_hit:+.1f}%")

# Statistical test
t_stat, t_p = stats.ttest_ind(top_quartile['best_ppg'], bottom_quartile['best_ppg'])
print(f"  - T-test p-value: {t_p:.4f}")

# ============================================================================
# TEST 5: Draft Capital Interaction
# ============================================================================
print("\n" + "="*80)
print("TEST 5: DRAFT CAPITAL INTERACTION")
print("="*80)

# Define draft tiers
day1 = analysis_df[analysis_df['pick'] <= 32]
day2 = analysis_df[(analysis_df['pick'] > 32) & (analysis_df['pick'] <= 100)]
day3 = analysis_df[analysis_df['pick'] > 100]

print("\nCorrelation between dominator and NFL PPG by draft tier:")
print("-"*60)

for tier_name, tier_df in [("DAY 1 (picks 1-32)", day1),
                            ("DAY 2 (picks 33-100)", day2),
                            ("DAY 3 (picks 101+)", day3)]:
    if len(tier_df) >= 10:
        r, p = stats.pearsonr(tier_df['peak_dominator'], tier_df['best_ppg'])
        print(f"\n{tier_name}: n={len(tier_df)}")
        print(f"  - Correlation: r = {r:.3f}, p = {p:.4f}")
        print(f"  - Avg dominator: {tier_df['peak_dominator'].mean():.1f}%")
        print(f"  - Avg PPG: {tier_df['best_ppg'].mean():.1f}")
        print(f"  - Hit24 rate: {tier_df['hit24'].mean()*100:.1f}%")
    else:
        print(f"\n{tier_name}: n={len(tier_df)} (too few for analysis)")

# ============================================================================
# BONUS: What DOES predict NFL success?
# ============================================================================
print("\n" + "="*80)
print("BONUS: WHAT ACTUALLY PREDICTS NFL SUCCESS?")
print("="*80)

print("\nCorrelation matrix with best_ppg:")
predictors = ['pick', 'peak_dominator']
if 'breakout_age' in analysis_df.columns and analysis_df['breakout_age'].notna().sum() > 20:
    predictors.append('breakout_age')

for pred in predictors:
    valid = analysis_df[analysis_df[pred].notna()]
    r, p = stats.pearsonr(valid[pred], valid['best_ppg'])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {pred:<20}: r = {r:>6.3f} (p = {p:.4f}) {sig}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: WHERE IS THE BREAKDOWN?")
print("="*80)

if abs(r_dom_ppg) < 0.1:
    print("\n❌ FINDING: Dominator has essentially NO correlation with NFL PPG")
    print("   This means high college dominator WRs don't produce better in the NFL")

if diff_hit < 5:
    print("\n❌ FINDING: Top vs bottom dominator quartiles have similar hit rates")
    print("   Being elite in college doesn't predict being elite in the NFL")

if abs(r_pick_ppr) > abs(r_dom_ppr) * 2:
    print("\n✓ FINDING: Draft capital is MUCH more predictive than dominator")
    print("   NFL scouts already factor in production when drafting")

# Check if there's signal in any draft tier
print("\n\nPOSSIBLE EXPLANATIONS:")
print("1. NFL teams already price in college production when drafting")
print("2. High dominator often comes from weak competition/small schools")
print("3. College production doesn't translate to NFL production")
print("4. Sample size issues with recent draft classes")
