"""
Test threshold approach for WR breakout scoring:
- Dominator as pass/fail (20% threshold)
- Breakout age drives the score
- 0.70 penalty if never hit 20% dominator
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load data
breakout = pd.read_csv('data/wr_breakout_age_scores.csv')
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')

# Filter to WRs only
wr_nfl = hit_rates[hit_rates['position'] == 'WR'].copy()

# Merge breakout data with NFL data
merged = wr_nfl.merge(
    breakout[['player_name', 'draft_year', 'peak_dominator', 'breakout_age']],
    on=['player_name', 'draft_year'],
    how='left'
)

# Filter to 2020-2024 and players who played
analysis = merged[
    (merged['draft_year'].isin([2020, 2021, 2022, 2023, 2024])) &
    (merged['best_ppr'].notna()) &
    (merged['best_ppr'] > 0) &
    (merged['peak_dominator'].notna())
].copy()

print("="*80)
print("TESTING THRESHOLD APPROACH: Dominator Pass/Fail + Age Score")
print("="*80)

# ============================================================================
# STEP 1: Define age-based scoring
# ============================================================================
# Age multipliers (younger = better)
AGE_SCORES = {
    18: 100,  # Best
    19: 90,
    20: 75,
    21: 60,
    22: 45,
    23: 30,
    24: 15,
    25: 10,
}

def get_age_score(breakout_age):
    """Convert breakout age to a 0-100 score"""
    if pd.isna(breakout_age):
        return 25  # Never broke out - low score
    age = int(breakout_age)
    return AGE_SCORES.get(age, 10)

# ============================================================================
# STEP 2: Apply threshold logic
# ============================================================================
DOMINATOR_THRESHOLD = 0.20  # 20%
NO_BREAKOUT_PENALTY = 0.70

analysis['hit_20_pct'] = analysis['peak_dominator'] >= 20  # peak_dominator is already in %
analysis['age_score'] = analysis['breakout_age'].apply(get_age_score)

# Apply penalty if never hit 20%
analysis['breakout_score'] = analysis.apply(
    lambda row: row['age_score'] * NO_BREAKOUT_PENALTY if not row['hit_20_pct'] else row['age_score'],
    axis=1
)

# Calculate PPG for comparison
analysis['best_ppg'] = analysis['best_ppr'] / 17

print(f"\nAnalyzing {len(analysis)} WRs (2020-2024 drafts with dominator data)")

# ============================================================================
# STEP 3: Show the threshold distribution
# ============================================================================
print("\n" + "="*80)
print("THRESHOLD DISTRIBUTION")
print("="*80)

hit_threshold = analysis['hit_20_pct'].sum()
missed_threshold = (~analysis['hit_20_pct']).sum()
print(f"\nWRs who hit 20%+ dominator: {hit_threshold} ({hit_threshold/len(analysis)*100:.1f}%)")
print(f"WRs who NEVER hit 20%: {missed_threshold} ({missed_threshold/len(analysis)*100:.1f}%)")

# Hit rates by threshold
hit_group = analysis[analysis['hit_20_pct']]
miss_group = analysis[~analysis['hit_20_pct']]

print(f"\n--- WRs who HIT 20%+ threshold (n={len(hit_group)}) ---")
print(f"  Hit24 rate: {hit_group['hit24'].mean()*100:.1f}%")
print(f"  Avg PPG: {hit_group['best_ppg'].mean():.1f}")

print(f"\n--- WRs who MISSED 20% threshold (n={len(miss_group)}) ---")
print(f"  Hit24 rate: {miss_group['hit24'].mean()*100:.1f}%")
print(f"  Avg PPG: {miss_group['best_ppg'].mean():.1f}")

# Statistical test
if len(miss_group) > 5:
    t_stat, t_p = stats.ttest_ind(hit_group['best_ppg'], miss_group['best_ppg'])
    print(f"\n  T-test for PPG difference: p = {t_p:.4f}")

# ============================================================================
# STEP 4: Test predictive power of new breakout_score
# ============================================================================
print("\n" + "="*80)
print("PREDICTIVE POWER OF NEW BREAKOUT SCORE")
print("="*80)

# Correlation with NFL PPG
r_new, p_new = stats.pearsonr(analysis['breakout_score'], analysis['best_ppg'])
print(f"\nNEW breakout_score vs NFL PPG:")
print(f"  r = {r_new:.3f}, p = {p_new:.4f}")

# Compare to old approach (peak_dominator alone)
r_old, p_old = stats.pearsonr(analysis['peak_dominator'], analysis['best_ppg'])
print(f"\nOLD peak_dominator vs NFL PPG:")
print(f"  r = {r_old:.3f}, p = {p_old:.4f}")

# Age score alone
r_age, p_age = stats.pearsonr(analysis['age_score'], analysis['best_ppg'])
print(f"\nAge score alone vs NFL PPG:")
print(f"  r = {r_age:.3f}, p = {p_age:.4f}")

# Point-biserial correlation with Hit24
r_hit24, p_hit24 = stats.pointbiserialr(analysis['hit24'], analysis['breakout_score'])
print(f"\nNEW breakout_score vs Hit24 (point-biserial):")
print(f"  r = {r_hit24:.3f}, p = {p_hit24:.4f}")

# ============================================================================
# STEP 5: Show top and bottom scorers
# ============================================================================
print("\n" + "="*80)
print("TOP 20 WRs BY NEW BREAKOUT SCORE")
print("="*80)

top20 = analysis.nlargest(20, 'breakout_score')[
    ['player_name', 'draft_year', 'pick', 'breakout_age', 'peak_dominator',
     'hit_20_pct', 'breakout_score', 'best_ppg', 'hit24']
].copy()

print("\n" + "-"*110)
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'BrkAge':>7} {'Dom%':>6} {'Pass':>5} {'Score':>6} {'PPG':>6} {'Hit24':>6}")
print("-"*110)
for _, row in top20.iterrows():
    age_str = f"{row['breakout_age']:.0f}" if pd.notna(row['breakout_age']) else "N/A"
    pass_str = "✓" if row['hit_20_pct'] else "✗"
    hit_str = "YES" if row['hit24'] == 1 else "no"
    print(f"{row['player_name']:<25} {row['draft_year']:>5} {row['pick']:>5} "
          f"{age_str:>7} {row['peak_dominator']:>5.1f}% {pass_str:>5} "
          f"{row['breakout_score']:>6.0f} {row['best_ppg']:>6.1f} {hit_str:>6}")

hits_top20 = top20['hit24'].sum()
print(f"\nHits in top 20: {hits_top20}/20 ({hits_top20/20*100:.1f}%)")

# ============================================================================
# STEP 6: Show WRs penalized (never hit 20%)
# ============================================================================
print("\n" + "="*80)
print("WRs WHO GOT PENALIZED (never hit 20% dominator)")
print("="*80)

penalized = analysis[~analysis['hit_20_pct']].sort_values('best_ppg', ascending=False)[
    ['player_name', 'draft_year', 'pick', 'peak_dominator', 'breakout_score', 'best_ppg', 'hit24']
].head(15)

print("\n" + "-"*95)
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'MaxDom%':>8} {'Score':>6} {'PPG':>6} {'Hit24':>6}")
print("-"*95)
for _, row in penalized.iterrows():
    hit_str = "YES" if row['hit24'] == 1 else "no"
    print(f"{row['player_name']:<25} {row['draft_year']:>5} {row['pick']:>5} "
          f"{row['peak_dominator']:>7.1f}% {row['breakout_score']:>6.0f} "
          f"{row['best_ppg']:>6.1f} {hit_str:>6}")

# Check if any hits were penalized
penalized_hits = analysis[(~analysis['hit_20_pct']) & (analysis['hit24'] == 1)]
print(f"\n⚠️  NFL hits who got penalized: {len(penalized_hits)}")
if len(penalized_hits) > 0:
    for _, row in penalized_hits.iterrows():
        print(f"   - {row['player_name']} ({row['draft_year']}): {row['peak_dominator']:.1f}% dominator, {row['best_ppg']:.1f} PPG")

# ============================================================================
# STEP 7: Quartile analysis of new score
# ============================================================================
print("\n" + "="*80)
print("QUARTILE ANALYSIS OF NEW BREAKOUT SCORE")
print("="*80)

q25 = analysis['breakout_score'].quantile(0.25)
q50 = analysis['breakout_score'].quantile(0.50)
q75 = analysis['breakout_score'].quantile(0.75)

print(f"\nScore quartile cutoffs: Q1={q25:.0f}, Q2={q50:.0f}, Q3={q75:.0f}")

for label, low, high in [("Top 25%", q75, 999),
                          ("2nd quartile", q50, q75),
                          ("3rd quartile", q25, q50),
                          ("Bottom 25%", 0, q25)]:
    subset = analysis[(analysis['breakout_score'] > low) | (analysis['breakout_score'] >= high)]
    subset = analysis[(analysis['breakout_score'] >= low) & (analysis['breakout_score'] <= high)] if low > 0 else analysis[analysis['breakout_score'] <= high]
    if label == "Top 25%":
        subset = analysis[analysis['breakout_score'] >= q75]
    elif label == "2nd quartile":
        subset = analysis[(analysis['breakout_score'] >= q50) & (analysis['breakout_score'] < q75)]
    elif label == "3rd quartile":
        subset = analysis[(analysis['breakout_score'] >= q25) & (analysis['breakout_score'] < q50)]
    else:
        subset = analysis[analysis['breakout_score'] < q25]

    print(f"\n{label} (n={len(subset)}):")
    print(f"  - Avg PPG: {subset['best_ppg'].mean():.1f}")
    print(f"  - Hit24 rate: {subset['hit24'].mean()*100:.1f}%")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Threshold approach results:
- NEW breakout_score correlation with NFL PPG: r = {r_new:.3f} (p = {p_new:.4f})
- OLD peak_dominator correlation with NFL PPG: r = {r_old:.3f} (p = {p_old:.4f})
- Age score alone correlation with NFL PPG:    r = {r_age:.3f} (p = {p_age:.4f})

Improvement: {'+' if r_new > r_old else ''}{(r_new - r_old):.3f} correlation points

The threshold approach {"IMPROVES" if r_new > r_old else "does NOT improve"} predictive power.
""")
