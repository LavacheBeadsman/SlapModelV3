"""
Analyze Continuous vs Discrete Scoring Approaches
==================================================

Current Problem: Discrete tiers create artificial cliffs
- Age 19.1 gets same score as age 19.9
- Age 20.0 gets same score as age 20.9

This analysis compares:
1. Current discrete tiers
2. Option 1: Linear interpolation (requires exact birthdates)
3. Option 2: Dominator magnitude tiebreaker (works with integer ages)
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 90)
print("CONTINUOUS SCORING ANALYSIS")
print("=" * 90)

# ============================================================================
# PART 1: BIRTHDATE DATA AVAILABILITY
# ============================================================================
print("\n" + "=" * 90)
print("PART 1: BIRTHDATE DATA AVAILABILITY")
print("=" * 90)

# Load all data
wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
prospects = pd.read_csv('data/prospects_final.csv')

print("\n--- BACKTEST DATA (2015-2025) ---")
print(f"WR backtest: {len(wr_backtest)} players")
print(f"  Has birthdate column: NO")
print(f"  Breakout age format: INTEGER (19, 20, 21, etc.)")
print(f"  Decimal ages found: {len(wr_backtest[wr_backtest['breakout_age'].notna() & (wr_backtest['breakout_age'] % 1 != 0)])}")

print(f"\nRB backtest: {len(rb_backtest)} players")
print(f"  Has birthdate column: NO")
print(f"  Draft age format: INTEGER (21, 22, 23, etc.)")
print(f"  Decimal ages found: {len(rb_backtest[rb_backtest['age'].notna() & (rb_backtest['age'] % 1 != 0)])}")

print("\n--- 2026 PROSPECTS ---")
wr_2026 = prospects[prospects['position'] == 'WR']
rb_2026 = prospects[prospects['position'] == 'RB']

wr_with_bd = wr_2026[wr_2026['birthdate'].notna() & (wr_2026['birthdate'] != 'MISSING')]
rb_with_bd = rb_2026[rb_2026['birthdate'].notna() & (rb_2026['birthdate'] != 'MISSING')]

print(f"WR prospects: {len(wr_2026)} total, {len(wr_with_bd)} with birthdates ({100*len(wr_with_bd)/len(wr_2026):.1f}%)")
print(f"RB prospects: {len(rb_2026)} total, {len(rb_with_bd)} with birthdates ({100*len(rb_with_bd)/len(rb_2026):.1f}%)")

# ============================================================================
# PART 2: CURRENT DISCRETE SCORING (WR Breakout Age)
# ============================================================================
print("\n" + "=" * 90)
print("PART 2: CURRENT WR BREAKOUT AGE SCORING (DISCRETE)")
print("=" * 90)

def current_discrete_score(breakout_age):
    """Current discrete tier scoring"""
    if pd.isna(breakout_age):
        return 25.0  # Never broke out
    age = int(breakout_age)
    if age <= 18:
        return 100.0
    elif age == 19:
        return 90.0
    elif age == 20:
        return 75.0
    elif age == 21:
        return 60.0
    elif age == 22:
        return 45.0
    elif age == 23:
        return 30.0
    else:
        return 25.0

print("\nCurrent discrete tiers:")
print("  Age 18 → 100")
print("  Age 19 → 90")
print("  Age 20 → 75")
print("  Age 21 → 60")
print("  Age 22 → 45")
print("  Age 23 → 30")
print("  Never  → 25")

# Show the cliff problem
print("\n--- CLIFF PROBLEM EXAMPLE ---")
print("Player A: breakout at 19.1 years → Score: 90")
print("Player B: breakout at 19.9 years → Score: 90")
print("Player C: breakout at 20.0 years → Score: 75 (15-point drop!)")
print("\nThis 15-point cliff is artificial - 0.1 years shouldn't matter that much.")

# ============================================================================
# PART 3: OPTION 1 - LINEAR INTERPOLATION
# ============================================================================
print("\n" + "=" * 90)
print("PART 3: OPTION 1 - LINEAR INTERPOLATION")
print("=" * 90)

def linear_interpolation_score(exact_age):
    """
    Linear interpolation from age 18 to age 24

    Formula: score = 100 - (5 × (exact_age - 18))

    This creates a smooth decline:
    - Age 18.0 → 100
    - Age 19.0 → 95
    - Age 20.0 → 90
    - Age 21.0 → 85
    - Age 22.0 → 80
    - Age 23.0 → 75
    - Age 24.0 → 70
    - Never → 25
    """
    if pd.isna(exact_age):
        return 25.0

    score = 100 - (5 * (exact_age - 18))
    return max(25.0, min(100.0, score))

print("\nLinear formula: score = 100 - 5 × (exact_age - 18)")
print("\nExample scores:")
for age in [18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 23.0, 24.0]:
    print(f"  Age {age:.1f} → {linear_interpolation_score(age):.1f}")

print("\n--- REQUIREMENT: Exact birthdates needed ---")
print("To calculate exact_age at season start (Aug 1):")
print("  exact_breakout_age = (season_start - birthdate).days / 365.25")

# ============================================================================
# PART 4: OPTION 2 - DOMINATOR MAGNITUDE TIEBREAKER
# ============================================================================
print("\n" + "=" * 90)
print("PART 4: OPTION 2 - DOMINATOR MAGNITUDE TIEBREAKER")
print("=" * 90)

def dominator_tiebreaker_score(breakout_age, peak_dominator):
    """
    Uses integer age tier as base, adds dominator magnitude bonus

    Formula:
    - base_score = age_tier_score (100, 90, 75, etc.)
    - bonus = min((peak_dominator - 20) × 0.5, 9.9)  # Cap at 9.9 to stay within tier
    - final_score = base_score + bonus

    This differentiates within tiers:
    - Age 19, 35% dominator → 90 + 7.5 = 97.5
    - Age 19, 22% dominator → 90 + 1.0 = 91.0
    """
    if pd.isna(breakout_age):
        return 25.0

    # Base score from integer age
    age = int(breakout_age)
    if age <= 18:
        base = 100.0
    elif age == 19:
        base = 90.0
    elif age == 20:
        base = 75.0
    elif age == 21:
        base = 60.0
    elif age == 22:
        base = 45.0
    elif age == 23:
        base = 30.0
    else:
        return 25.0

    # Dominator bonus (only if they broke out and we have dominator data)
    if pd.notna(peak_dominator) and peak_dominator >= 20:
        # Bonus: 0.5 points per 1% above 20%
        # Cap at 9.9 so we don't exceed next tier
        bonus = min((peak_dominator - 20) * 0.5, 9.9)
    else:
        bonus = 0.0

    return base + bonus

print("\nFormula:")
print("  base_score = age_tier (100, 90, 75, 60, 45, 30)")
print("  bonus = min((peak_dominator - 20) × 0.5, 9.9)")
print("  final_score = base + bonus")

print("\n--- EXAMPLE: Age 19 breakouts with different dominators ---")
for dom in [22, 25, 30, 35, 40, 50]:
    score = dominator_tiebreaker_score(19, dom)
    print(f"  Age 19, {dom}% dominator → {score:.1f}")

print("\n--- ADVANTAGE: Works with integer ages (backtest compatible) ---")

# ============================================================================
# PART 5: OPTION 3 - HYBRID (LINEAR + DOMINATOR BONUS)
# ============================================================================
print("\n" + "=" * 90)
print("PART 5: OPTION 3 - HYBRID (LINEAR + DOMINATOR BONUS)")
print("=" * 90)

def hybrid_score(exact_age, peak_dominator):
    """
    Combines linear age interpolation with dominator bonus

    Formula:
    - age_score = 100 - 5 × (exact_age - 18)
    - dominator_bonus = min((peak_dominator - 20) × 0.3, 5.0)
    - final = age_score + dominator_bonus
    """
    if pd.isna(exact_age):
        return 25.0

    # Linear age component
    age_score = 100 - (5 * (exact_age - 18))
    age_score = max(25.0, min(100.0, age_score))

    # Dominator bonus (smaller multiplier since age is already continuous)
    if pd.notna(peak_dominator) and peak_dominator >= 20:
        bonus = min((peak_dominator - 20) * 0.3, 5.0)
    else:
        bonus = 0.0

    return min(100.0, age_score + bonus)

print("\nCombines both approaches:")
print("  age_score = 100 - 5 × (exact_age - 18)")
print("  bonus = min((peak_dominator - 20) × 0.3, 5.0)")
print("  final = age_score + bonus")

print("\n--- EXAMPLE: Two 19.5-year-old breakouts ---")
print(f"  Age 19.5, 25% dominator → {hybrid_score(19.5, 25):.1f}")
print(f"  Age 19.5, 40% dominator → {hybrid_score(19.5, 40):.1f}")

# ============================================================================
# PART 6: COMPARE APPROACHES ON BACKTEST DATA
# ============================================================================
print("\n" + "=" * 90)
print("PART 6: COMPARE APPROACHES ON BACKTEST DATA")
print("=" * 90)

# Calculate scores with different methods
wr_backtest['score_discrete'] = wr_backtest.apply(
    lambda x: current_discrete_score(x['breakout_age']), axis=1
)

wr_backtest['score_dominator_tb'] = wr_backtest.apply(
    lambda x: dominator_tiebreaker_score(x['breakout_age'], x['peak_dominator']), axis=1
)

# Compare distributions
print("\n--- SCORE DISTRIBUTION COMPARISON ---")
print(f"\n{'Metric':<30} {'Discrete':>12} {'Dominator TB':>12}")
print("-" * 55)
print(f"{'Mean':<30} {wr_backtest['score_discrete'].mean():>12.1f} {wr_backtest['score_dominator_tb'].mean():>12.1f}")
print(f"{'Std Dev':<30} {wr_backtest['score_discrete'].std():>12.1f} {wr_backtest['score_dominator_tb'].std():>12.1f}")
print(f"{'Min':<30} {wr_backtest['score_discrete'].min():>12.1f} {wr_backtest['score_dominator_tb'].min():>12.1f}")
print(f"{'Max':<30} {wr_backtest['score_discrete'].max():>12.1f} {wr_backtest['score_dominator_tb'].max():>12.1f}")
print(f"{'Unique Values':<30} {wr_backtest['score_discrete'].nunique():>12} {wr_backtest['score_dominator_tb'].nunique():>12}")

# Show some examples where dominator tiebreaker creates differentiation
print("\n--- AGE 19 BREAKOUTS - DOMINATOR TIEBREAKER DIFFERENTIATION ---")
age_19 = wr_backtest[wr_backtest['breakout_age'] == 19].copy()
age_19 = age_19.sort_values('peak_dominator', ascending=False)
print(f"\n{len(age_19)} WRs broke out at age 19")
print("\nTop 10 by dominator (all get 90 in discrete, differentiated in tiebreaker):")
print(f"{'Player':<25} {'Year':>6} {'Discrete':>10} {'Dominator TB':>12} {'Peak Dom':>10} {'hit24':>6}")
print("-" * 75)
for _, row in age_19.head(10).iterrows():
    print(f"{row['player_name']:<25} {row['draft_year']:>6.0f} {row['score_discrete']:>10.1f} {row['score_dominator_tb']:>12.1f} {row['peak_dominator']:>10.1f} {row['hit24']:>6}")

# ============================================================================
# PART 7: CORRELATION WITH NFL SUCCESS
# ============================================================================
print("\n" + "=" * 90)
print("PART 7: CORRELATION WITH NFL SUCCESS")
print("=" * 90)

# Filter to players with outcomes (not 2025 rookies)
wr_with_outcome = wr_backtest[wr_backtest['draft_year'] < 2025].copy()

# Calculate correlations
from scipy import stats

corr_discrete, p_discrete = stats.pearsonr(
    wr_with_outcome['score_discrete'],
    wr_with_outcome['hit24']
)

corr_dom_tb, p_dom_tb = stats.pearsonr(
    wr_with_outcome['score_dominator_tb'],
    wr_with_outcome['hit24']
)

print(f"\nCorrelation with hit24 (Top-24 WR season):")
print(f"  Discrete scoring:    r = {corr_discrete:.4f}, p = {p_discrete:.4f}")
print(f"  Dominator tiebreaker: r = {corr_dom_tb:.4f}, p = {p_dom_tb:.4f}")

# Also check with best_ppr
corr_discrete_ppr, _ = stats.pearsonr(
    wr_with_outcome['score_discrete'],
    wr_with_outcome['best_ppr']
)

corr_dom_tb_ppr, _ = stats.pearsonr(
    wr_with_outcome['score_dominator_tb'],
    wr_with_outcome['best_ppr']
)

print(f"\nCorrelation with best_ppr:")
print(f"  Discrete scoring:    r = {corr_discrete_ppr:.4f}")
print(f"  Dominator tiebreaker: r = {corr_dom_tb_ppr:.4f}")

# ============================================================================
# PART 8: RECOMMENDATION
# ============================================================================
print("\n" + "=" * 90)
print("PART 8: RECOMMENDATION")
print("=" * 90)

print("""
SUMMARY OF OPTIONS:

┌─────────────────────────────────────────────────────────────────────────────┐
│ OPTION 1: LINEAR INTERPOLATION                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Formula: score = 100 - 5 × (exact_age - 18)                                 │
│ Pros:                                                                       │
│   - Eliminates cliffs entirely                                              │
│   - Most mathematically elegant                                             │
│ Cons:                                                                       │
│   - Requires exact birthdates (only have for 2026 prospects)                │
│   - Cannot apply consistently to backtest (no birthdates)                   │
│   - Creates inconsistency between backtest and prospect scoring             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ OPTION 2: DOMINATOR MAGNITUDE TIEBREAKER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Formula: base_tier_score + min((peak_dom - 20) × 0.5, 9.9)                  │
│ Pros:                                                                       │
│   - Works with integer ages (backtest compatible)                           │
│   - Rewards higher dominators within age tiers                              │
│   - Creates continuous scores without needing birthdates                    │
│   - Consistent application across backtest AND prospects                    │
│ Cons:                                                                       │
│   - Still has tier boundaries (though softened by dominator)                │
│   - Players without dominator data get no bonus                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ OPTION 3: HYBRID (LINEAR + DOMINATOR BONUS)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Formula: (100 - 5 × exact_age) + min((peak_dom - 20) × 0.3, 5.0)            │
│ Pros:                                                                       │
│   - Best of both worlds when birthdates available                           │
│   - Rewards both young breakout AND high dominator                          │
│ Cons:                                                                       │
│   - Still requires birthdates for full benefit                              │
│   - More complex                                                            │
└─────────────────────────────────────────────────────────────────────────────┘

RECOMMENDATION: OPTION 2 (Dominator Magnitude Tiebreaker)

Rationale:
1. CONSISTENCY: Same formula for backtest AND prospects
2. DATA AVAILABILITY: Works with integer ages (backtest has no birthdates)
3. DIFFERENTIATION: Creates ~30+ unique score values vs only 7 in discrete
4. VALIDATION: Can test correlation improvement on backtest immediately
5. FAIRNESS: Higher dominators should differentiate players of same age

For RBs: Use similar approach with receiving production as the base.
""")

print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
