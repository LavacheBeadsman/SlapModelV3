"""
RB Age Weighting Test - Clear Comparison
=========================================

Question: Does age weighting improve RB production signal?

Using SEASON AGE (draft_age - 1) as documented:
- Season age 19 → 1.20x bonus
- Season age 20 → 1.10x bonus
- Season age 21 → 1.00x baseline
- Season age 22 → 0.90x penalty
- Season age 23+ → 0.80x penalty
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=" * 90)
print("RB AGE WEIGHTING TEST")
print("=" * 90)

# ============================================================================
# LOAD DATA
# ============================================================================

rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')

# Filter to valid data
valid = rb_backtest[
    (rb_backtest['rec_yards'].notna()) &
    (rb_backtest['team_pass_att'].notna()) &
    (rb_backtest['team_pass_att'] > 0) &
    (rb_backtest['best_ppr'].notna()) &
    (rb_backtest['best_ppr'] > 0) &
    (rb_backtest['age'].notna())
].copy()

print(f"\nRBs in backtest with valid data: {len(valid)}")

# ============================================================================
# CLARIFY AGE DATA
# ============================================================================

print("\n" + "=" * 90)
print("CLARIFYING AGE DATA")
print("=" * 90)

print(f"""
AGE IN DATA: Draft Age (age when drafted)
SEASON AGE: draft_age - 1 (age during final college season)

Example:
- Todd Gurley: draft_age = 21 → season_age = 20 (was 20 during final college season)
- Derrick Henry: draft_age = 22 → season_age = 21 (was 21 during final college season)

Age distribution in data (DRAFT AGE):
""")
print(valid['age'].value_counts().sort_index())

valid['season_age'] = valid['age'] - 1

print(f"\nConverted to SEASON AGE (draft_age - 1):")
print(valid['season_age'].value_counts().sort_index())

# ============================================================================
# METHOD 1: WITHOUT AGE WEIGHTING (Current)
# ============================================================================

def production_no_age_weight(rec_yards, team_pass_att):
    """Current method - no age weighting"""
    ratio = rec_yards / team_pass_att
    return (ratio / 1.0) * 100

valid['score_no_age'] = valid.apply(
    lambda x: production_no_age_weight(x['rec_yards'], x['team_pass_att']), axis=1
)

# ============================================================================
# METHOD 2: WITH AGE WEIGHTING (Using Season Age)
# ============================================================================

def production_with_age_weight(rec_yards, team_pass_att, draft_age):
    """With age weighting using SEASON AGE (draft_age - 1)"""
    season_age = int(draft_age) - 1

    age_weights = {
        19: 1.20,  # Sophomore production = 20% bonus
        20: 1.10,  # Junior production = 10% bonus
        21: 1.00,  # Senior baseline
        22: 0.90,  # 5th year = 10% penalty
        23: 0.80,  # 6th year = 20% penalty
    }
    age_weight = age_weights.get(season_age, 0.80)

    ratio = rec_yards / team_pass_att
    return (ratio * age_weight / 1.0) * 100

valid['score_with_age'] = valid.apply(
    lambda x: production_with_age_weight(x['rec_yards'], x['team_pass_att'], x['age']), axis=1
)

# ============================================================================
# CORRELATION COMPARISON
# ============================================================================

print("\n" + "=" * 90)
print("CORRELATION COMPARISON")
print("=" * 90)

# Raw correlation with NFL PPR
r_no_age, p_no_age = pearsonr(valid['score_no_age'], valid['best_ppr'])
r_with_age, p_with_age = pearsonr(valid['score_with_age'], valid['best_ppr'])

print(f"""
RAW CORRELATION WITH NFL PPR:

| Method              | Correlation (r) | p-value    | Significant? |
|---------------------|-----------------|------------|--------------|
| WITHOUT age weight  | {r_no_age:.4f}          | {p_no_age:.6f}   | {'Yes ✓' if p_no_age < 0.05 else 'No'} |
| WITH age weight     | {r_with_age:.4f}          | {p_with_age:.6f}   | {'Yes ✓' if p_with_age < 0.05 else 'No'} |
""")

improvement = (r_with_age - r_no_age) / r_no_age * 100
print(f"Change: {'+' if improvement > 0 else ''}{improvement:.1f}% {'improvement' if improvement > 0 else 'decline'}")

# ============================================================================
# PARTIAL CORRELATION (After controlling for DC)
# ============================================================================

print("\n" + "=" * 90)
print("PARTIAL CORRELATION (What Matters - After Controlling for DC)")
print("=" * 90)

def calc_dc(pick):
    return 100 - 2.40 * (pick ** 0.62 - 1)

valid['dc_score'] = valid['pick'].apply(calc_dc)

def partial_corr(metric_col, df):
    """Calculate partial correlation controlling for DC"""
    data = df[['best_ppr', 'dc_score', metric_col]].dropna()
    X = sm.add_constant(data['dc_score'])

    resid_metric = sm.OLS(data[metric_col], X).fit().resid
    resid_outcome = sm.OLS(data['best_ppr'], X).fit().resid

    r, p = pearsonr(resid_metric, resid_outcome)
    return r, p

part_r_no_age, part_p_no_age = partial_corr('score_no_age', valid)
part_r_with_age, part_p_with_age = partial_corr('score_with_age', valid)

print(f"""
PARTIAL CORRELATION (after controlling for draft capital):

| Method              | Partial r | p-value    | Adds Value Beyond DC? |
|---------------------|-----------|------------|----------------------|
| WITHOUT age weight  | {part_r_no_age:.4f}    | {part_p_no_age:.6f}   | {'Yes ✓' if part_p_no_age < 0.10 else 'No'} |
| WITH age weight     | {part_r_with_age:.4f}    | {part_p_with_age:.6f}   | {'Yes ✓' if part_p_with_age < 0.10 else 'No'} |
""")

part_improvement = (part_r_with_age - part_r_no_age) / part_r_no_age * 100
print(f"Change: {'+' if part_improvement > 0 else ''}{part_improvement:.1f}%")

# ============================================================================
# R-SQUARED COMPARISON
# ============================================================================

print("\n" + "=" * 90)
print("MULTIPLE REGRESSION R² COMPARISON")
print("=" * 90)

# Model 1: DC + production (no age weight)
X1 = sm.add_constant(valid[['dc_score', 'score_no_age']])
model1 = sm.OLS(valid['best_ppr'], X1).fit()

# Model 2: DC + production (with age weight)
X2 = sm.add_constant(valid[['dc_score', 'score_with_age']])
model2 = sm.OLS(valid['best_ppr'], X2).fit()

print(f"""
| Model                        | R²     | Production p-value |
|------------------------------|--------|-------------------|
| DC + Production (no age)     | {model1.rsquared:.4f} | {model1.pvalues['score_no_age']:.4f}            |
| DC + Production (with age)   | {model2.rsquared:.4f} | {model2.pvalues['score_with_age']:.4f}            |
""")

# ============================================================================
# EXAMPLE PLAYERS
# ============================================================================

print("\n" + "=" * 90)
print("HOW AGE WEIGHTING CHANGES SCORES - EXAMPLES")
print("=" * 90)

examples = valid.nlargest(15, 'best_ppr')[['player_name', 'age', 'season_age', 'rec_yards',
                                            'score_no_age', 'score_with_age', 'best_ppr']].copy()
examples['change'] = examples['score_with_age'] - examples['score_no_age']

print(f"\n{'Player':<22} {'Draft':<6} {'Season':<7} {'RecYds':<8} {'NoAge':<8} {'WithAge':<9} {'Change':<8} {'NFL PPR':<10}")
print("-" * 95)

for _, row in examples.iterrows():
    change_str = f"{row['change']:+.1f}"
    print(f"{row['player_name']:<22} {int(row['age']):<6} {int(row['season_age']):<7} {int(row['rec_yards']):<8} {row['score_no_age']:<8.1f} {row['score_with_age']:<9.1f} {change_str:<8} {row['best_ppr']:<10.1f}")

# ============================================================================
# BY AGE GROUP ANALYSIS
# ============================================================================

print("\n" + "=" * 90)
print("HIT RATE BY SEASON AGE")
print("=" * 90)

for season_age in [19, 20, 21, 22, 23]:
    group = valid[valid['season_age'] == season_age]
    if len(group) > 0:
        weight = {19: 1.20, 20: 1.10, 21: 1.00, 22: 0.90, 23: 0.80}.get(season_age, 0.80)
        hit_rate = group['hit24'].mean() * 100
        avg_ppr = group['best_ppr'].mean()
        print(f"Season Age {season_age} (weight {weight}x): N={len(group):>3}, Hit24={hit_rate:>5.1f}%, Avg PPR={avg_ppr:>6.1f}")

# ============================================================================
# VERDICT
# ============================================================================

print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)

print(f"""
QUESTION: Does age weighting improve RB production signal?

RAW CORRELATION:
  Without age: r = {r_no_age:.4f}
  With age:    r = {r_with_age:.4f}
  → Age weighting {'IMPROVES' if r_with_age > r_no_age else 'HURTS'} raw correlation by {abs(improvement):.1f}%

PARTIAL CORRELATION (what matters for SLAP):
  Without age: r = {part_r_no_age:.4f}
  With age:    r = {part_r_with_age:.4f}
  → Age weighting {'IMPROVES' if part_r_with_age > part_r_no_age else 'HURTS'} partial correlation by {abs(part_improvement):.1f}%

R-SQUARED:
  Without age: R² = {model1.rsquared:.4f}
  With age:    R² = {model2.rsquared:.4f}

RECOMMENDATION:
""")

if part_r_with_age > part_r_no_age:
    print("  ✓ IMPLEMENT age weighting - it improves predictions")
else:
    print("  ✗ DO NOT implement age weighting - it doesn't help")
    print("  The current approach (no age weight) performs equally well or better.")
    print("  DC already captures age information (young RBs get drafted higher).")

print("\n" + "=" * 90)
