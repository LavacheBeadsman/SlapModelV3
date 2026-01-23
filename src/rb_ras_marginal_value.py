"""
RB RAS Marginal Value Analysis

We KNOW draft capital dominates. The question is:
Can RAS help find value at the margins?

1. Residual approach - do high-RAS RBs outperform their draft slot?
2. Draft tier analysis - does RAS matter more in certain ranges?
3. Find actual edge cases - sleeper hits and early busts
4. Practical value test - would RAS have helped historically?
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("RB RAS MARGINAL VALUE ANALYSIS")
print("Can RAS help find value at the margins?")
print("=" * 90)

# Load data
rb = pd.read_csv('data/rb_backtest_2015_2024.csv')
rb = rb[rb['draft_year'] <= 2023].copy()  # Exclude 2024 rookies
rb['inv_sqrt_pick'] = 1 / np.sqrt(rb['pick'])

# Filter to those with fantasy production
rb_with_prod = rb[rb['best_ppg'] > 0].copy()
rb_with_ras = rb_with_prod[rb_with_prod['RAS'].notna()].copy()

print(f"\nLoaded {len(rb_with_ras)} RBs with RAS and fantasy production")

# ============================================================================
# ANALYSIS 1: RESIDUAL APPROACH
# ============================================================================
print("\n" + "=" * 90)
print("ANALYSIS 1: RESIDUAL APPROACH")
print("Does RAS correlate with OUTPERFORMANCE vs draft slot?")
print("=" * 90)

# Step 1: Fit DC-only model
X_dc = sm.add_constant(rb_with_ras['inv_sqrt_pick'])
model_dc = sm.OLS(rb_with_ras['best_ppg'], X_dc).fit()

print(f"\nDC-only model: PPG = {model_dc.params['const']:.2f} + {model_dc.params['inv_sqrt_pick']:.2f} × (1/√pick)")
print(f"R² = {model_dc.rsquared:.3f}")

# Step 2: Calculate expected PPG and residual
rb_with_ras['expected_ppg'] = model_dc.predict(X_dc)
rb_with_ras['residual'] = rb_with_ras['best_ppg'] - rb_with_ras['expected_ppg']

print(f"\nResidual stats:")
print(f"  Mean: {rb_with_ras['residual'].mean():.2f} (should be ~0)")
print(f"  Std:  {rb_with_ras['residual'].std():.2f}")
print(f"  Min:  {rb_with_ras['residual'].min():.2f}")
print(f"  Max:  {rb_with_ras['residual'].max():.2f}")

# Step 3: Does RAS correlate with residual?
r_resid, p_resid = stats.pearsonr(rb_with_ras['RAS'], rb_with_ras['residual'])
print(f"\nRAS vs Residual (outperformance):")
print(f"  Pearson r = {r_resid:.3f}, p = {p_resid:.4f}")
print(f"  {'SIGNIFICANT' if p_resid < 0.05 else 'NOT significant'} at p<0.05")

# Show top outperformers and their RAS
print("\n--- Top 10 Outperformers (highest residual) ---")
top_outperformers = rb_with_ras.nlargest(10, 'residual')[
    ['player_name', 'draft_year', 'pick', 'round', 'RAS', 'best_ppg', 'expected_ppg', 'residual']
]
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'RAS':>6} {'PPG':>6} {'Exp':>6} {'Resid':>7}")
print("-" * 70)
for _, row in top_outperformers.iterrows():
    ras = f"{row['RAS']:.1f}" if pd.notna(row['RAS']) else "N/A"
    print(f"{row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} {ras:>6} {row['best_ppg']:>6.1f} {row['expected_ppg']:>6.1f} {row['residual']:>+7.1f}")

# Show biggest underperformers
print("\n--- Top 10 Underperformers (lowest residual) ---")
bottom = rb_with_ras.nsmallest(10, 'residual')[
    ['player_name', 'draft_year', 'pick', 'round', 'RAS', 'best_ppg', 'expected_ppg', 'residual']
]
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'RAS':>6} {'PPG':>6} {'Exp':>6} {'Resid':>7}")
print("-" * 70)
for _, row in bottom.iterrows():
    ras = f"{row['RAS']:.1f}" if pd.notna(row['RAS']) else "N/A"
    print(f"{row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} {ras:>6} {row['best_ppg']:>6.1f} {row['expected_ppg']:>6.1f} {row['residual']:>+7.1f}")

# Average RAS for outperformers vs underperformers
outperformers = rb_with_ras[rb_with_ras['residual'] > 0]
underperformers = rb_with_ras[rb_with_ras['residual'] < 0]

print(f"\n--- RAS Comparison ---")
print(f"Outperformers (n={len(outperformers)}): avg RAS = {outperformers['RAS'].mean():.2f}")
print(f"Underperformers (n={len(underperformers)}): avg RAS = {underperformers['RAS'].mean():.2f}")

t_stat, t_pval = stats.ttest_ind(outperformers['RAS'], underperformers['RAS'])
print(f"T-test: t = {t_stat:.2f}, p = {t_pval:.4f}")

# ============================================================================
# ANALYSIS 2: DRAFT TIER ANALYSIS
# ============================================================================
print("\n" + "=" * 90)
print("ANALYSIS 2: DRAFT TIER ANALYSIS")
print("Does RAS matter more in certain draft ranges?")
print("=" * 90)

tiers = [
    ('Round 1 (1-32)', 1, 32),
    ('Round 2 (33-64)', 33, 64),
    ('Round 3-4 (65-140)', 65, 140),
    ('Day 3 (141+)', 141, 300)
]

print(f"\n{'Tier':<20} {'N':>5} {'RAS→PPG r':>10} {'p-value':>10} {'Significant':>12} {'Hit Rate':>10}")
print("-" * 75)

tier_results = []
for name, lo, hi in tiers:
    tier = rb_with_ras[(rb_with_ras['pick'] >= lo) & (rb_with_ras['pick'] <= hi)]
    if len(tier) >= 10:
        r, p = stats.pearsonr(tier['RAS'], tier['best_ppg'])
        sig = "YES" if p < 0.05 else "no"
        hit_rate = tier['hit24'].mean() * 100
        print(f"{name:<20} {len(tier):>5} {r:>10.3f} {p:>10.4f} {sig:>12} {hit_rate:>9.1f}%")
        tier_results.append((name, len(tier), r, p, hit_rate))
    else:
        print(f"{name:<20} {len(tier):>5} {'(too few)':>10}")

# RAS vs RESIDUAL by tier (does RAS predict outperformance in each tier?)
print("\n--- RAS vs Residual (outperformance) by Tier ---")
print(f"{'Tier':<20} {'N':>5} {'RAS→Resid r':>12} {'p-value':>10} {'Significant':>12}")
print("-" * 65)

for name, lo, hi in tiers:
    tier = rb_with_ras[(rb_with_ras['pick'] >= lo) & (rb_with_ras['pick'] <= hi)]
    if len(tier) >= 10:
        r, p = stats.pearsonr(tier['RAS'], tier['residual'])
        sig = "YES" if p < 0.05 else "no"
        print(f"{name:<20} {len(tier):>5} {r:>12.3f} {p:>10.4f} {sig:>12}")

# ============================================================================
# ANALYSIS 3: FIND THE ACTUAL EDGE CASES
# ============================================================================
print("\n" + "=" * 90)
print("ANALYSIS 3: FIND THE ACTUAL EDGE CASES")
print("=" * 90)

# Day 3 sleeper hits: pick 141+, RAS 8+, hit24=1
print("\n--- DAY 3 SLEEPER HITS (pick 141+, RAS 8+, hit24=1) ---")
day3_sleepers = rb_with_ras[
    (rb_with_ras['pick'] >= 141) &
    (rb_with_ras['RAS'] >= 8) &
    (rb_with_ras['hit24'] == 1)
].sort_values('best_ppg', ascending=False)

if len(day3_sleepers) > 0:
    print(f"\nFound {len(day3_sleepers)} Day 3 high-RAS hits!")
    print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'RAS':>6} {'PPG':>6} {'Hit24':>6}")
    print("-" * 60)
    for _, row in day3_sleepers.iterrows():
        print(f"{row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} {row['RAS']:>6.1f} {row['best_ppg']:>6.1f} {int(row['hit24']):>6}")
else:
    print("No Day 3 high-RAS hits found!")

# Day 3 total for comparison
day3_all = rb_with_ras[rb_with_ras['pick'] >= 141]
day3_high_ras = rb_with_ras[(rb_with_ras['pick'] >= 141) & (rb_with_ras['RAS'] >= 8)]
day3_low_ras = rb_with_ras[(rb_with_ras['pick'] >= 141) & (rb_with_ras['RAS'] < 6)]

print(f"\n--- Day 3 (pick 141+) Hit Rates by RAS ---")
print(f"All Day 3 RBs: {len(day3_all)} total, {day3_all['hit24'].sum()} hits ({day3_all['hit24'].mean()*100:.1f}%)")
print(f"High RAS (8+): {len(day3_high_ras)} total, {day3_high_ras['hit24'].sum()} hits ({day3_high_ras['hit24'].mean()*100:.1f}%)")
print(f"Low RAS (<6):  {len(day3_low_ras)} total, {day3_low_ras['hit24'].sum()} hits ({day3_low_ras['hit24'].mean()*100:.1f}%)")

# Day 1-2 busts: pick <=64, RAS <6, hit24=0
print("\n--- DAY 1-2 BUSTS (pick ≤64, RAS <6, hit24=0) ---")
early_busts = rb_with_ras[
    (rb_with_ras['pick'] <= 64) &
    (rb_with_ras['RAS'] < 6) &
    (rb_with_ras['hit24'] == 0)
].sort_values('pick')

if len(early_busts) > 0:
    print(f"\nFound {len(early_busts)} early low-RAS busts!")
    print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'RAS':>6} {'PPG':>6}")
    print("-" * 55)
    for _, row in early_busts.iterrows():
        print(f"{row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} {row['RAS']:>6.1f} {row['best_ppg']:>6.1f}")
else:
    print("No early low-RAS busts found!")

# Day 1-2 comparison by RAS
early_all = rb_with_ras[rb_with_ras['pick'] <= 64]
early_high_ras = rb_with_ras[(rb_with_ras['pick'] <= 64) & (rb_with_ras['RAS'] >= 8)]
early_low_ras = rb_with_ras[(rb_with_ras['pick'] <= 64) & (rb_with_ras['RAS'] < 6)]

print(f"\n--- Day 1-2 (pick ≤64) Hit Rates by RAS ---")
print(f"All Day 1-2 RBs: {len(early_all)} total, {early_all['hit24'].sum()} hits ({early_all['hit24'].mean()*100:.1f}%)")
if len(early_high_ras) > 0:
    print(f"High RAS (8+):   {len(early_high_ras)} total, {early_high_ras['hit24'].sum()} hits ({early_high_ras['hit24'].mean()*100:.1f}%)")
if len(early_low_ras) > 0:
    print(f"Low RAS (<6):    {len(early_low_ras)} total, {early_low_ras['hit24'].sum()} hits ({early_low_ras['hit24'].mean()*100:.1f}%)")

# ============================================================================
# ANALYSIS 4: PRACTICAL VALUE TEST
# ============================================================================
print("\n" + "=" * 90)
print("ANALYSIS 4: PRACTICAL VALUE TEST")
print("Would 'DC + RAS bonus for Day 3' have helped historically?")
print("=" * 90)

# Create a modified score: DC + RAS bonus for late picks
rb_with_ras['dc_score'] = rb_with_ras['inv_sqrt_pick'] * 100  # Normalize
rb_with_ras['ras_bonus'] = np.where(
    rb_with_ras['pick'] >= 141,  # Day 3 only
    (rb_with_ras['RAS'] - 7) * 2,  # Bonus for RAS above 7
    0
)
rb_with_ras['modified_score'] = rb_with_ras['dc_score'] + rb_with_ras['ras_bonus']

# Find Day 3 players where RAS bonus would have significantly boosted ranking
print("\n--- Day 3 Players with Largest RAS Bonus ---")
day3_bonus = rb_with_ras[rb_with_ras['pick'] >= 141].nlargest(15, 'ras_bonus')[
    ['player_name', 'draft_year', 'pick', 'RAS', 'ras_bonus', 'best_ppg', 'hit24']
]

print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'RAS':>6} {'Bonus':>7} {'PPG':>6} {'Hit':>5}")
print("-" * 70)
for _, row in day3_bonus.iterrows():
    hit = "✓" if row['hit24'] == 1 else ""
    print(f"{row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} {row['RAS']:>6.1f} {row['ras_bonus']:>+7.1f} {row['best_ppg']:>6.1f} {hit:>5}")

# Count how many of the bonus players hit
bonus_players = rb_with_ras[(rb_with_ras['pick'] >= 141) & (rb_with_ras['ras_bonus'] > 2)]
print(f"\nDay 3 players with RAS bonus > 2:")
print(f"  Total: {len(bonus_players)}")
print(f"  Hits: {bonus_players['hit24'].sum()} ({bonus_players['hit24'].mean()*100:.1f}%)")

# Compare to Day 3 baseline
print(f"\nDay 3 baseline hit rate: {day3_all['hit24'].mean()*100:.1f}%")

# Specific case studies
print("\n--- CASE STUDIES: RAS Winners and Losers ---")

# Winners: High RAS, exceeded expectations
winners = rb_with_ras[
    (rb_with_ras['RAS'] >= 9) &
    (rb_with_ras['residual'] > 5)
].sort_values('residual', ascending=False)

print("\nHIGH RAS + OUTPERFORMED (RAS 9+, residual > +5):")
if len(winners) > 0:
    for _, row in winners.head(5).iterrows():
        print(f"  {row['player_name']} ({int(row['draft_year'])}): Pick {int(row['pick'])}, RAS {row['RAS']:.1f}, PPG {row['best_ppg']:.1f} (expected {row['expected_ppg']:.1f}, +{row['residual']:.1f})")
else:
    print("  None found")

# Losers: High RAS, underperformed
losers = rb_with_ras[
    (rb_with_ras['RAS'] >= 9) &
    (rb_with_ras['residual'] < -3)
].sort_values('residual')

print("\nHIGH RAS + UNDERPERFORMED (RAS 9+, residual < -3):")
if len(losers) > 0:
    for _, row in losers.head(5).iterrows():
        print(f"  {row['player_name']} ({int(row['draft_year'])}): Pick {int(row['pick'])}, RAS {row['RAS']:.1f}, PPG {row['best_ppg']:.1f} (expected {row['expected_ppg']:.1f}, {row['residual']:.1f})")
else:
    print("  None found")

# Low RAS surprises
surprises = rb_with_ras[
    (rb_with_ras['RAS'] < 6) &
    (rb_with_ras['hit24'] == 1)
].sort_values('best_ppg', ascending=False)

print("\nLOW RAS SURPRISES (RAS <6, still hit):")
if len(surprises) > 0:
    for _, row in surprises.head(5).iterrows():
        print(f"  {row['player_name']} ({int(row['draft_year'])}): Pick {int(row['pick'])}, RAS {row['RAS']:.1f}, PPG {row['best_ppg']:.1f}")
else:
    print("  None found")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("FINAL SUMMARY: CAN RAS HELP AT THE MARGINS?")
print("=" * 90)

# Calculate key metrics
day3_high_hit = day3_high_ras['hit24'].mean() * 100 if len(day3_high_ras) > 0 else 0
day3_low_hit = day3_low_ras['hit24'].mean() * 100 if len(day3_low_ras) > 0 else 0
day3_spread = day3_high_hit - day3_low_hit

early_high_hit = early_high_ras['hit24'].mean() * 100 if len(early_high_ras) > 0 else 0
early_low_hit = early_low_ras['hit24'].mean() * 100 if len(early_low_ras) > 0 else 0

print(f"""
1. RESIDUAL ANALYSIS:
   - RAS vs outperformance (residual): r = {r_resid:.3f}, p = {p_resid:.4f}
   - Conclusion: RAS does {'NOT ' if p_resid >= 0.05 else ''}predict who outperforms their draft slot

2. DRAFT TIER ANALYSIS:
   - Round 1: RAS correlation = {tier_results[0][2]:.3f} (p={tier_results[0][3]:.3f})
   - Round 2: RAS correlation = {tier_results[1][2]:.3f} (p={tier_results[1][3]:.3f})
   - Round 3-4: RAS correlation = {tier_results[2][2]:.3f} (p={tier_results[2][3]:.3f})
   - Day 3: RAS correlation = {tier_results[3][2]:.3f} (p={tier_results[3][3]:.3f})

3. DAY 3 SLEEPER TEST:
   - Day 3 High RAS (8+): {len(day3_high_ras)} players, {day3_high_hit:.1f}% hit rate
   - Day 3 Low RAS (<6):  {len(day3_low_ras)} players, {day3_low_hit:.1f}% hit rate
   - Spread: {day3_spread:+.1f}%
   - Day 3 sleeper hits found: {len(day3_sleepers)}

4. PRACTICAL VALUE:
   - Would RAS bonus have identified sleepers? {"YES" if len(day3_sleepers) > 0 and day3_spread > 10 else "MARGINAL" if day3_spread > 5 else "NO"}
   - Day 3 bonus players hit rate: {bonus_players['hit24'].mean()*100:.1f}% vs baseline {day3_all['hit24'].mean()*100:.1f}%

BOTTOM LINE: {"RAS provides marginal value for Day 3 RBs" if day3_spread > 10 else "RAS does NOT provide meaningful value even at the margins"}
""")
