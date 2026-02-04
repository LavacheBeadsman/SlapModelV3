"""
RB Production Analysis - Age Weighting and Scoring Options
==========================================================

Questions to answer:
1. Is age weighting being applied? (No - we need to fix this)
2. What age data do we have?
3. Does age weighting improve predictions?
4. Should we use draft age or college season age?
5. Should RBs use continuous scoring like WR Option A?
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=" * 100)
print("RB PRODUCTION ANALYSIS - AGE WEIGHTING AND SCORING OPTIONS")
print("=" * 100)

# ============================================================================
# LOAD DATA
# ============================================================================

rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
print(f"\nLoaded {len(rb_backtest)} RBs from backtest")

# Filter to those with production data and NFL outcomes
valid = rb_backtest[
    (rb_backtest['rec_yards'].notna()) &
    (rb_backtest['team_pass_att'].notna()) &
    (rb_backtest['team_pass_att'] > 0) &
    (rb_backtest['best_ppr'].notna()) &
    (rb_backtest['best_ppr'] > 0)
].copy()

print(f"Valid for analysis (has receiving stats + NFL PPR): {len(valid)}")

# ============================================================================
# EXAMINE AGE DATA
# ============================================================================

print("\n" + "=" * 100)
print("AGE DATA EXAMINATION")
print("=" * 100)

print(f"\nAge column (draft age) distribution:")
print(valid['age'].value_counts().sort_index())

print(f"\nAge stats:")
print(f"  Min: {valid['age'].min()}")
print(f"  Max: {valid['age'].max()}")
print(f"  Mean: {valid['age'].mean():.1f}")
print(f"  Missing: {valid['age'].isna().sum()}")

# ============================================================================
# CURRENT SCORING (NO AGE WEIGHT)
# ============================================================================

def current_production_score(rec_yards, team_pass_att):
    """Current scoring - no age weight"""
    ratio = rec_yards / team_pass_att
    return (ratio / 1.0) * 100

valid['raw_production'] = valid['rec_yards'] / valid['team_pass_att']
valid['score_current'] = valid.apply(
    lambda x: current_production_score(x['rec_yards'], x['team_pass_att']), axis=1
)

# ============================================================================
# OPTION 1: AGE WEIGHTING (As documented in CLAUDE.md)
# ============================================================================

def age_weight_documented(draft_age):
    """Age weight as documented in CLAUDE.md (using draft age)"""
    if pd.isna(draft_age):
        return 1.0
    age = int(draft_age)
    weights = {
        19: 1.20,
        20: 1.10,
        21: 1.00,
        22: 0.90,
        23: 0.80,
        24: 0.80,
        25: 0.80,
    }
    return weights.get(age, 0.80)

valid['age_weight_doc'] = valid['age'].apply(age_weight_documented)
valid['score_age_weighted_doc'] = valid['score_current'] * valid['age_weight_doc']

# ============================================================================
# OPTION 2: COLLEGE SEASON AGE (draft_age - 1)
# ============================================================================

def college_season_age_weight(draft_age):
    """Age weight using college season age (draft_age - 1)"""
    if pd.isna(draft_age):
        return 1.0
    college_age = int(draft_age) - 1  # Stats are from year before draft
    weights = {
        18: 1.25,  # True freshman production
        19: 1.15,
        20: 1.05,
        21: 0.95,
        22: 0.85,
        23: 0.75,
    }
    return weights.get(college_age, 0.75)

valid['age_weight_college'] = valid['age'].apply(college_season_age_weight)
valid['score_age_weighted_college'] = valid['score_current'] * valid['age_weight_college']

# ============================================================================
# OPTION 3: CONTINUOUS AGE FORMULA
# ============================================================================

def continuous_age_weight(draft_age):
    """Continuous age weight - linear decay from baseline at 21"""
    if pd.isna(draft_age):
        return 1.0
    # Baseline at draft age 21, +10% per year younger, -10% per year older
    return 1.0 + 0.10 * (21 - draft_age)

valid['age_weight_continuous'] = valid['age'].apply(continuous_age_weight)
valid['score_continuous_age'] = valid['score_current'] * valid['age_weight_continuous']

# ============================================================================
# OPTION 4: CONTINUOUS WITH PRODUCTION SPREAD (Like WR Option A)
# ============================================================================

def option_a_style_rb(row):
    """
    RB production scoring similar to WR Option A:
    - Base score from production tier
    - Bonus/penalty from age
    """
    raw_prod = row['raw_production']
    draft_age = row['age']

    if pd.isna(raw_prod):
        return None

    # Convert raw production to 0-100 scale
    # Typical range: 0.1 (10th percentile) to 1.0 (90th percentile)
    base_score = min(100, max(0, raw_prod * 100))

    # Age adjustment: +/- 10 points based on age
    if pd.notna(draft_age):
        age_adj = (21 - draft_age) * 5  # +5 per year younger than 21
        age_adj = max(-15, min(15, age_adj))  # Cap at +/- 15
    else:
        age_adj = 0

    return base_score + age_adj

valid['score_option_a_style'] = valid.apply(option_a_style_rb, axis=1)

# ============================================================================
# DC SCORE
# ============================================================================

def calc_dc(pick):
    return 100 - 2.40 * (pick ** 0.62 - 1)

valid['dc_score'] = valid['pick'].apply(calc_dc)

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("CORRELATION ANALYSIS WITH NFL PPR")
print("=" * 100)

def calc_metrics(scores, outcomes, name):
    """Calculate correlation and stats"""
    mask = ~(scores.isna() | outcomes.isna())
    r, p = pearsonr(scores[mask], outcomes[mask])
    return {
        'Option': name,
        'r': r,
        'p-value': p,
        'Sig': 'Yes ✓' if p < 0.05 else 'No',
        'Mean': scores.mean(),
        'Std': scores.std(),
    }

results = []
for score_col, name in [
    ('score_current', 'Current (No Age Weight)'),
    ('score_age_weighted_doc', 'CLAUDE.md Age Weight'),
    ('score_age_weighted_college', 'College Season Age Weight'),
    ('score_continuous_age', 'Continuous Age Weight'),
    ('score_option_a_style', 'Option A Style (Add/Sub)'),
]:
    results.append(calc_metrics(valid[score_col], valid['best_ppr'], name))

results_df = pd.DataFrame(results)

print(f"\n{'Option':<35} {'r':<10} {'p-value':<12} {'Sig?':<8} {'Mean':<10} {'Std':<10}")
print("-" * 95)
for _, row in results_df.iterrows():
    print(f"{row['Option']:<35} {row['r']:<10.4f} {row['p-value']:<12.4f} {row['Sig']:<8} {row['Mean']:<10.1f} {row['Std']:<10.1f}")

# ============================================================================
# PARTIAL CORRELATION (After controlling for DC)
# ============================================================================

print("\n" + "=" * 100)
print("PARTIAL CORRELATION (After Controlling for Draft Capital)")
print("=" * 100)

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
print("-" * 95)

partial_results = []
for score_col, name in [
    ('score_current', 'Current (No Age Weight)'),
    ('score_age_weighted_doc', 'CLAUDE.md Age Weight'),
    ('score_age_weighted_college', 'College Season Age Weight'),
    ('score_continuous_age', 'Continuous Age Weight'),
    ('score_option_a_style', 'Option A Style (Add/Sub)'),
]:
    raw_r, _ = pearsonr(valid[score_col].dropna(), valid.loc[valid[score_col].notna(), 'best_ppr'])
    part_r, part_p = partial_corr(score_col, 'best_ppr', 'dc_score', valid)
    adds = 'Yes ✓' if part_p < 0.10 else 'No'
    partial_results.append({
        'name': name, 'score_col': score_col, 'raw_r': raw_r,
        'part_r': part_r, 'part_p': part_p, 'adds': adds
    })
    print(f"{name:<35} {raw_r:<12.4f} {part_r:<12.4f} {part_p:<12.4f} {adds:<12}")

# ============================================================================
# MULTIPLE REGRESSION
# ============================================================================

print("\n" + "=" * 100)
print("MULTIPLE REGRESSION: DC + Production Score → NFL PPR")
print("=" * 100)

print(f"\n{'Model':<40} {'DC p-val':<12} {'Prod p-val':<12} {'R²':<10}")
print("-" * 80)

for score_col, name in [
    ('score_current', 'DC + Current'),
    ('score_age_weighted_doc', 'DC + CLAUDE.md Age'),
    ('score_age_weighted_college', 'DC + College Age'),
    ('score_option_a_style', 'DC + Option A Style'),
]:
    data = valid[['best_ppr', 'dc_score', score_col]].dropna()
    X = sm.add_constant(data[['dc_score', score_col]])
    y = data['best_ppr']
    model = sm.OLS(y, X).fit()

    dc_p = model.pvalues['dc_score']
    score_p = model.pvalues[score_col]
    r2 = model.rsquared

    print(f"{name:<40} {dc_p:<12.4f} {score_p:<12.4f} {r2:<10.4f}")

# ============================================================================
# EXAMINE SPECIFIC EXAMPLES
# ============================================================================

print("\n" + "=" * 100)
print("EXAMPLE PLAYERS - HOW SCORES CHANGE")
print("=" * 100)

# Show players with different ages and production levels
examples = valid.sort_values('best_ppr', ascending=False).head(20)

print(f"\n{'Player':<25} {'Age':<5} {'RecYds':<8} {'Current':<10} {'AgeWt':<10} {'College':<10} {'OptA':<10} {'NFL PPR':<10}")
print("-" * 100)

for _, row in examples.iterrows():
    print(f"{row['player_name']:<25} {int(row['age']):<5} {int(row['rec_yards']):<8} {row['score_current']:<10.1f} {row['score_age_weighted_doc']:<10.1f} {row['score_age_weighted_college']:<10.1f} {row['score_option_a_style']:<10.1f} {row['best_ppr']:<10.1f}")

# ============================================================================
# YOUNG VS OLD ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("YOUNG VS OLD RB ANALYSIS")
print("=" * 100)

young = valid[valid['age'] <= 21]
old = valid[valid['age'] >= 23]

print(f"\nYoung RBs (draft age ≤ 21): {len(young)}")
print(f"  Hit24 rate: {young['hit24'].mean()*100:.1f}%")
print(f"  Avg NFL PPR: {young['best_ppr'].mean():.1f}")
print(f"  Avg raw production: {young['raw_production'].mean():.3f}")

print(f"\nOld RBs (draft age ≥ 23): {len(old)}")
print(f"  Hit24 rate: {old['hit24'].mean()*100:.1f}%")
print(f"  Avg NFL PPR: {old['best_ppr'].mean():.1f}")
print(f"  Avg raw production: {old['raw_production'].mean():.3f}")

# ============================================================================
# SCORE DISTRIBUTION
# ============================================================================

print("\n" + "=" * 100)
print("SCORE DISTRIBUTION COMPARISON")
print("=" * 100)

bins = [(80, 101), (60, 80), (40, 60), (20, 40), (0, 20)]

print(f"\n{'Score Range':<15} {'Current':<15} {'AgeWt Doc':<15} {'College Age':<15} {'Option A':<15}")
print("-" * 75)

for low, high in bins:
    label = f"{low}-{high-1}" if high <= 100 else f"{low}+"
    curr = len(valid[(valid['score_current'] >= low) & (valid['score_current'] < high)])
    age_doc = len(valid[(valid['score_age_weighted_doc'] >= low) & (valid['score_age_weighted_doc'] < high)])
    college = len(valid[(valid['score_age_weighted_college'] >= low) & (valid['score_age_weighted_college'] < high)])
    opt_a = len(valid[(valid['score_option_a_style'] >= low) & (valid['score_option_a_style'] < high)])
    print(f"{label:<15} {curr:<15} {age_doc:<15} {college:<15} {opt_a:<15}")

# ============================================================================
# RECOMMENDATION
# ============================================================================

print("\n" + "=" * 100)
print("FINDINGS AND RECOMMENDATION")
print("=" * 100)

# Find best option by partial correlation
best = max(partial_results, key=lambda x: x['part_r'])
current = partial_results[0]

print(f"""
FINDINGS:

1. CURRENT STATE:
   - Age weighting is NOT being applied (contradicts CLAUDE.md)
   - Current production score correlation: r = {current['raw_r']:.4f}
   - Partial correlation (after DC): {current['part_r']:.4f}, p = {current['part_p']:.4f}
   - Statistically significant: {current['adds']}

2. BEST OPTION: {best['name']}
   - Partial correlation: {best['part_r']:.4f}
   - p-value: {best['part_p']:.4f}
   - Adds value beyond DC: {best['adds']}

3. YOUNG VS OLD:
   - Young RBs (≤21) hit rate: {young['hit24'].mean()*100:.1f}%
   - Old RBs (≥23) hit rate: {old['hit24'].mean()*100:.1f}%
   - This {'supports' if young['hit24'].mean() > old['hit24'].mean() else 'does NOT support'} age weighting

4. RECOMMENDED ACTION:
""")

improvement = best['part_r'] - current['part_r']
if improvement > 0.01 and best['part_p'] < 0.10:
    print(f"   Implement {best['name']}")
    print(f"   - Improves partial r by {improvement:.4f}")
    print(f"   - Adds statistically significant value")
elif improvement > 0:
    print(f"   Consider {best['name']}")
    print(f"   - Slight improvement ({improvement:.4f}) but not statistically significant")
    print(f"   - May not be worth the complexity")
else:
    print(f"   Keep current (no age weight)")
    print(f"   - Age weighting does NOT improve predictions")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
