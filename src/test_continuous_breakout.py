"""
Test Continuous Breakout Age Scoring Options
=============================================

Current: Discrete tiers (18→100, 19→90, 20→75, etc.)
Testing: Three continuous alternatives

Option A: Linear interpolation (use dominator to spread within years)
Option B: Continuous formula: 100 - 5*(age - 18)
Option C: Yearly buckets with dominator-based spread within each bucket
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=" * 100)
print("TESTING CONTINUOUS BREAKOUT AGE SCORING OPTIONS")
print("=" * 100)

# ============================================================================
# LOAD DATA
# ============================================================================

wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"\nLoaded {len(wr_backtest)} WRs from backtest")

# Filter to those with breakout age and NFL outcomes
valid = wr_backtest[
    (wr_backtest['breakout_age'].notna()) &
    (wr_backtest['best_ppr'].notna()) &
    (wr_backtest['best_ppr'] > 0)
].copy()

print(f"Valid for analysis (has breakout age + NFL PPR): {len(valid)}")

# ============================================================================
# CURRENT SYSTEM (DISCRETE TIERS)
# ============================================================================

def current_breakout_score(age):
    """Current discrete tier system"""
    if pd.isna(age):
        return 79.7  # Average
    age = int(age)
    scores = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    return scores.get(age, 25)

valid['score_current'] = valid['breakout_age'].apply(current_breakout_score)

# ============================================================================
# OPTION A: LINEAR INTERPOLATION WITH DOMINATOR SPREAD
# ============================================================================

def option_a_score(row):
    """
    Linear interpolation using dominator to spread within age years.

    Base scores:
    - Age 18: 100
    - Age 19: 90
    - Age 20: 80
    - Age 21: 70
    - Age 22: 60
    - Age 23: 50
    - No breakout: 40

    Within each age, higher dominator = higher score (up to +9 points)
    """
    age = row['breakout_age']
    dominator = row.get('peak_dominator', np.nan)

    if pd.isna(age):
        return 40  # No breakout

    age = int(age)

    # Base scores for each age
    base_scores = {18: 91, 19: 81, 20: 71, 21: 61, 22: 51, 23: 41}
    base = base_scores.get(age, 31)

    # Add dominator bonus (0-9 points based on dominator percentile within that age)
    if pd.notna(dominator):
        # Normalize dominator to 0-9 range
        # Typical dominator range is roughly 10-40% (100-400 raw)
        # Cap at 400 for max bonus
        dom_bonus = min(9, max(0, (dominator - 100) / 300 * 9))
        return base + dom_bonus
    else:
        return base + 4.5  # Middle of range if missing dominator

valid['score_option_a'] = valid.apply(option_a_score, axis=1)

# ============================================================================
# OPTION B: PURE CONTINUOUS FORMULA
# ============================================================================

def option_b_score(age):
    """
    Pure continuous formula: 100 - 10*(age - 18)

    - Age 18.0: 100
    - Age 18.5: 95
    - Age 19.0: 90
    - Age 19.5: 85
    - Age 20.0: 80
    - Age 20.5: 75
    - Age 21.0: 70
    - etc.
    - No breakout: 40
    """
    if pd.isna(age):
        return 40

    score = 100 - 10 * (age - 18)
    return max(30, min(100, score))  # Floor at 30, cap at 100

valid['score_option_b'] = valid['breakout_age'].apply(option_b_score)

# ============================================================================
# OPTION C: YEARLY BUCKETS WITH DOMINATOR SPREAD
# ============================================================================

def option_c_score(row):
    """
    Yearly buckets with dominator-based spread within each bucket.

    Ranges:
    - Age 18: 95-100 (spread by dominator within age 18)
    - Age 19: 85-94
    - Age 20: 75-84
    - Age 21: 65-74
    - Age 22: 55-64
    - Age 23: 45-54
    - No breakout: 35-44
    """
    age = row['breakout_age']
    dominator = row.get('peak_dominator', np.nan)

    if pd.isna(age):
        return 40  # No breakout middle

    age = int(age)

    # Define ranges for each age
    ranges = {
        18: (95, 100),
        19: (85, 94),
        20: (75, 84),
        21: (65, 74),
        22: (55, 64),
        23: (45, 54),
    }

    low, high = ranges.get(age, (35, 44))

    # Use dominator to place within range
    if pd.notna(dominator):
        # Higher dominator = higher score within range
        # Normalize: assume dominator ranges from 50-400
        dom_pct = min(1.0, max(0.0, (dominator - 50) / 350))
        return low + dom_pct * (high - low)
    else:
        return (low + high) / 2  # Middle of range

valid['score_option_c'] = valid.apply(option_c_score, axis=1)

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("CORRELATION ANALYSIS WITH NFL PPR")
print("=" * 100)

def calc_metrics(scores, outcomes, name):
    """Calculate correlation and stats"""
    r, p = pearsonr(scores, outcomes)
    return {
        'Option': name,
        'Correlation (r)': r,
        'p-value': p,
        'Significant': 'Yes ✓' if p < 0.05 else 'No',
        'Mean': scores.mean(),
        'Std': scores.std(),
        'Min': scores.min(),
        'Max': scores.max(),
        'Unique Values': scores.nunique()
    }

results = []
for score_col, name in [
    ('score_current', 'Current (Discrete Tiers)'),
    ('score_option_a', 'Option A (Linear + Dominator)'),
    ('score_option_b', 'Option B (Pure Continuous)'),
    ('score_option_c', 'Option C (Buckets + Dominator)'),
]:
    results.append(calc_metrics(valid[score_col], valid['best_ppr'], name))

results_df = pd.DataFrame(results)

print(f"\n{'Option':<35} {'r':<10} {'p-value':<12} {'Sig?':<8} {'Mean':<8} {'Std':<8} {'Unique':<8}")
print("-" * 100)
for _, row in results_df.iterrows():
    print(f"{row['Option']:<35} {row['Correlation (r)']:<10.4f} {row['p-value']:<12.4f} {row['Significant']:<8} {row['Mean']:<8.1f} {row['Std']:<8.1f} {row['Unique Values']:<8}")

# ============================================================================
# PARTIAL CORRELATION (After controlling for DC)
# ============================================================================

print("\n" + "=" * 100)
print("PARTIAL CORRELATION (After Controlling for Draft Capital)")
print("=" * 100)

# Add DC score
def calc_dc(pick):
    return 100 - 2.40 * (pick ** 0.62 - 1)

valid['dc_score'] = valid['pick'].apply(calc_dc)

def partial_corr(metric_col, outcome_col, control_col, df):
    """Calculate partial correlation"""
    data = df[[metric_col, outcome_col, control_col]].dropna()
    X = sm.add_constant(data[control_col])

    # Residualize metric
    resid_m = sm.OLS(data[metric_col], X).fit().resid
    # Residualize outcome
    resid_o = sm.OLS(data[outcome_col], X).fit().resid

    r, p = pearsonr(resid_m, resid_o)
    return r, p

print(f"\n{'Option':<35} {'Raw r':<12} {'Partial r':<12} {'p-value':<12} {'Adds Value?':<12}")
print("-" * 100)

partial_results = []
for score_col, name in [
    ('score_current', 'Current (Discrete Tiers)'),
    ('score_option_a', 'Option A (Linear + Dominator)'),
    ('score_option_b', 'Option B (Pure Continuous)'),
    ('score_option_c', 'Option C (Buckets + Dominator)'),
]:
    raw_r, _ = pearsonr(valid[score_col], valid['best_ppr'])
    part_r, part_p = partial_corr(score_col, 'best_ppr', 'dc_score', valid)
    adds = 'Yes ✓' if part_p < 0.10 else 'No'
    partial_results.append({
        'name': name, 'score_col': score_col, 'raw_r': raw_r,
        'part_r': part_r, 'part_p': part_p, 'adds': adds
    })
    print(f"{name:<35} {raw_r:<12.4f} {part_r:<12.4f} {part_p:<12.4f} {adds:<12}")

# ============================================================================
# SCORE DISTRIBUTION COMPARISON
# ============================================================================

print("\n" + "=" * 100)
print("SCORE DISTRIBUTION COMPARISON")
print("=" * 100)

bins = [(90, 101), (80, 90), (70, 80), (60, 70), (50, 60), (40, 50), (30, 40), (0, 30)]

print(f"\n{'Score Range':<15} {'Current':<15} {'Option A':<15} {'Option B':<15} {'Option C':<15}")
print("-" * 75)

for low, high in bins:
    label = f"{low}-{high-1}" if high <= 100 else f"{low}+"
    curr = len(valid[(valid['score_current'] >= low) & (valid['score_current'] < high)])
    opt_a = len(valid[(valid['score_option_a'] >= low) & (valid['score_option_a'] < high)])
    opt_b = len(valid[(valid['score_option_b'] >= low) & (valid['score_option_b'] < high)])
    opt_c = len(valid[(valid['score_option_c'] >= low) & (valid['score_option_c'] < high)])
    print(f"{label:<15} {curr:<15} {opt_a:<15} {opt_b:<15} {opt_c:<15}")

# ============================================================================
# 20 WR COMPARISON TABLE
# ============================================================================

print("\n" + "=" * 100)
print("20 WR COMPARISON - OLD VS NEW SCORES")
print("=" * 100)

# Select diverse sample: mix of ages and draft positions
sample = valid.sort_values(['breakout_age', 'pick']).groupby('breakout_age').head(4).head(20)

print(f"\n{'Player':<25} {'Age':<5} {'Dom':<8} {'Current':<10} {'Opt A':<10} {'Opt B':<10} {'Opt C':<10} {'NFL PPR':<10}")
print("-" * 100)

for _, row in sample.iterrows():
    dom_str = f"{row['peak_dominator']:.0f}" if pd.notna(row['peak_dominator']) else "N/A"
    print(f"{row['player_name']:<25} {int(row['breakout_age']):<5} {dom_str:<8} {row['score_current']:<10.0f} {row['score_option_a']:<10.1f} {row['score_option_b']:<10.0f} {row['score_option_c']:<10.1f} {row['best_ppr']:<10.1f}")

# ============================================================================
# SPECIFIC EXAMPLES OF DIFFERENTIATION
# ============================================================================

print("\n" + "=" * 100)
print("DIFFERENTIATION EXAMPLES - SAME AGE, DIFFERENT SCORES")
print("=" * 100)

for age in [19, 20, 21]:
    print(f"\n--- Age {age} Breakouts (Current all get score {current_breakout_score(age)}) ---")
    age_group = valid[valid['breakout_age'] == age].sort_values('peak_dominator', ascending=False).head(5)

    for _, row in age_group.iterrows():
        dom_str = f"{row['peak_dominator']:.0f}" if pd.notna(row['peak_dominator']) else "N/A"
        print(f"  {row['player_name']:<25} Dom={dom_str:<6} → A:{row['score_option_a']:.1f}  B:{row['score_option_b']:.0f}  C:{row['score_option_c']:.1f}  NFL:{row['best_ppr']:.0f}")

# ============================================================================
# MULTIPLE REGRESSION COMPARISON
# ============================================================================

print("\n" + "=" * 100)
print("MULTIPLE REGRESSION: DC + Breakout Score → NFL PPR")
print("=" * 100)

print(f"\n{'Model':<40} {'DC p-val':<12} {'Score p-val':<12} {'R²':<10} {'Better?'}")
print("-" * 85)

baseline_r2 = None
for score_col, name in [
    ('score_current', 'DC + Current'),
    ('score_option_a', 'DC + Option A'),
    ('score_option_b', 'DC + Option B'),
    ('score_option_c', 'DC + Option C'),
]:
    data = valid[['best_ppr', 'dc_score', score_col]].dropna()
    X = sm.add_constant(data[['dc_score', score_col]])
    y = data['best_ppr']
    model = sm.OLS(y, X).fit()

    dc_p = model.pvalues['dc_score']
    score_p = model.pvalues[score_col]
    r2 = model.rsquared

    if baseline_r2 is None:
        baseline_r2 = r2
        better = 'Baseline'
    else:
        better = 'Yes ✓' if r2 > baseline_r2 else 'No'

    print(f"{name:<40} {dc_p:<12.4f} {score_p:<12.4f} {r2:<10.4f} {better}")

# ============================================================================
# HIT RATE ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("HIT RATE ANALYSIS BY SCORE TIER")
print("=" * 100)

def hit_rate_by_tier(score_col, tiers, df):
    """Calculate hit rates by score tier"""
    results = []
    for low, high in tiers:
        subset = df[(df[score_col] >= low) & (df[score_col] < high)]
        if len(subset) > 0:
            hits = subset['hit24'].sum()
            total = len(subset)
            rate = hits / total * 100
            results.append({'Tier': f"{low}-{high-1}", 'N': total, 'Hits': hits, 'Rate': rate})
    return pd.DataFrame(results)

tiers = [(90, 101), (80, 90), (70, 80), (60, 70), (50, 60), (0, 50)]

for score_col, name in [
    ('score_current', 'Current'),
    ('score_option_c', 'Option C'),  # Most different from current
]:
    print(f"\n{name}:")
    hr = hit_rate_by_tier(score_col, tiers, valid)
    for _, row in hr.iterrows():
        print(f"  {row['Tier']:<10} N={row['N']:<4} Hits={row['Hits']:<3} Rate={row['Rate']:.1f}%")

# ============================================================================
# RECOMMENDATION
# ============================================================================

print("\n" + "=" * 100)
print("RECOMMENDATION")
print("=" * 100)

# Find best option by partial correlation
best = max(partial_results, key=lambda x: x['part_r'])

print(f"""
SUMMARY OF FINDINGS:

1. CORRELATION COMPARISON:
   - Current (Discrete):     r = {partial_results[0]['raw_r']:.4f} (partial: {partial_results[0]['part_r']:.4f})
   - Option A (Linear+Dom):  r = {partial_results[1]['raw_r']:.4f} (partial: {partial_results[1]['part_r']:.4f})
   - Option B (Continuous):  r = {partial_results[2]['raw_r']:.4f} (partial: {partial_results[2]['part_r']:.4f})
   - Option C (Buckets+Dom): r = {partial_results[3]['raw_r']:.4f} (partial: {partial_results[3]['part_r']:.4f})

2. BEST OPTION: {best['name']}
   - Highest partial correlation: {best['part_r']:.4f}
   - p-value: {best['part_p']:.4f}
   - Adds value beyond DC: {best['adds']}

3. KEY INSIGHT:
""")

# Check if continuous options improve on current
current_part_r = partial_results[0]['part_r']
best_alt_r = max(r['part_r'] for r in partial_results[1:])

if best_alt_r > current_part_r:
    improvement = (best_alt_r - current_part_r) / abs(current_part_r) * 100
    print(f"   ✓ Best continuous option improves partial r by {improvement:.1f}%")
else:
    print(f"   ✗ Continuous options do NOT improve on discrete tiers")

print("""
4. RECOMMENDATION:
""")

# Give recommendation based on results
if best['name'] == 'Current (Discrete Tiers)':
    print("   Keep the current discrete tier system.")
    print("   The continuous options don't add predictive value.")
elif best['part_r'] - current_part_r > 0.01:
    print(f"   Switch to {best['name']}:")
    print(f"   - Improves partial correlation from {current_part_r:.4f} to {best['part_r']:.4f}")
    print(f"   - Creates better differentiation between players")
else:
    print("   The improvement is marginal (<0.01 in partial r).")
    print("   Could switch for better differentiation, but won't affect predictions much.")

# ============================================================================
# SAVE RESULTS
# ============================================================================

comparison = valid[['player_name', 'draft_year', 'pick', 'breakout_age', 'peak_dominator',
                    'score_current', 'score_option_a', 'score_option_b', 'score_option_c',
                    'best_ppr', 'hit24', 'dc_score']].copy()
comparison = comparison.sort_values('score_current', ascending=False)
comparison.to_csv('output/breakout_scoring_comparison.csv', index=False)
print(f"\n✓ Saved comparison to output/breakout_scoring_comparison.csv")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
