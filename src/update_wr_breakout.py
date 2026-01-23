"""
Update WR Breakout Scoring to Age-Only Approach

This script:
1. Updates backtest WR breakout scores using breakout age (from wr_breakout_age_scores.csv)
2. Updates 2026 WR breakout scores using current age as breakout age
3. Runs logistic regression to verify positive coefficient
4. Shows Waddle/Addison verification
5. Shows 2026 top 10 WRs
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Age score mapping (same as tested)
AGE_SCORES = {
    18: 100,  # Best - broke out as freshman
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

print("="*80)
print("UPDATING WR BREAKOUT SCORING TO AGE-ONLY APPROACH")
print("="*80)

# ============================================================================
# STEP 1: Load all data
# ============================================================================
print("\n1. Loading data...")

# Backtest data
backtest = pd.read_csv('data/backtest_college_stats.csv')
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')
breakout_ages = pd.read_csv('data/wr_breakout_age_scores.csv')

# 2026 prospects
prospects_2026 = pd.read_csv('data/prospects_final.csv')

print(f"   Backtest: {len(backtest)} players")
print(f"   Hit rates: {len(hit_rates)} players")
print(f"   Breakout ages: {len(breakout_ages)} WRs")
print(f"   2026 prospects: {len(prospects_2026)} players")

# ============================================================================
# STEP 2: Update backtest WR breakout scores
# ============================================================================
print("\n2. Updating backtest WR breakout scores...")

# Filter to WRs
backtest_wrs = backtest[backtest['position'] == 'WR'].copy()

# Merge with breakout ages
backtest_wrs = backtest_wrs.merge(
    breakout_ages[['player_name', 'draft_year', 'breakout_age']],
    on=['player_name', 'draft_year'],
    how='left'
)

# Calculate new breakout score (age-only)
backtest_wrs['breakout_score_new'] = backtest_wrs['breakout_age'].apply(get_age_score)

print(f"   WRs with breakout age data: {backtest_wrs['breakout_age'].notna().sum()}")
print(f"   WRs without breakout age (using default 25): {backtest_wrs['breakout_age'].isna().sum()}")

# ============================================================================
# STEP 3: Update 2026 WR breakout scores
# ============================================================================
print("\n3. Updating 2026 WR breakout scores...")

prospects_wrs_2026 = prospects_2026[prospects_2026['position'] == 'WR'].copy()

# For 2026 class: use current age as breakout age
# (If they're producing at age 20, that's their breakout)
prospects_wrs_2026['breakout_age'] = prospects_wrs_2026['age'].apply(
    lambda x: float(x) if pd.notna(x) and x != 'MISSING' else np.nan
)
prospects_wrs_2026['breakout_score_new'] = prospects_wrs_2026['breakout_age'].apply(get_age_score)

print(f"   2026 WRs: {len(prospects_wrs_2026)}")
print(f"   With age data: {prospects_wrs_2026['breakout_age'].notna().sum()}")

# ============================================================================
# STEP 4: Run logistic regression on backtest
# ============================================================================
print("\n4. Running logistic regression with new breakout scores...")

# Merge backtest WRs with NFL outcomes
wr_nfl = hit_rates[hit_rates['position'] == 'WR'].copy()
analysis = wr_nfl.merge(
    backtest_wrs[['player_name', 'draft_year', 'draft_capital_score', 'athletic_score', 'breakout_score_new']],
    on=['player_name', 'draft_year'],
    how='left'
)

# Filter to 2020-2024 with valid data
analysis = analysis[
    (analysis['draft_year'].isin([2020, 2021, 2022, 2023, 2024])) &
    (analysis['draft_capital_score'].notna()) &
    (analysis['breakout_score_new'].notna())
].copy()

print(f"   Analysis sample: {len(analysis)} WRs")

# Prepare features
X = analysis[['draft_capital_score', 'breakout_score_new']].values
y = analysis['hit24'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

print("\n   LOGISTIC REGRESSION COEFFICIENTS:")
print(f"   {'Variable':<25} {'Coefficient':>12} {'Direction':>12}")
print("   " + "-"*50)
for name, coef in zip(['draft_capital_score', 'breakout_score_new'], model.coef_[0]):
    direction = "✓ POSITIVE" if coef > 0 else "✗ NEGATIVE"
    print(f"   {name:<25} {coef:>12.3f} {direction:>12}")

# Calculate correlation
analysis['best_ppg'] = analysis['best_ppr'] / 17
r_new, p_new = stats.pearsonr(analysis['breakout_score_new'], analysis['best_ppg'])
print(f"\n   Correlation (breakout_score_new vs NFL PPG): r = {r_new:.3f}, p = {p_new:.4f}")

# ============================================================================
# STEP 5: Verify Waddle and Addison
# ============================================================================
print("\n" + "="*80)
print("5. WADDLE & ADDISON VERIFICATION")
print("="*80)

for name in ['Jaylen Waddle', 'Jordan Addison']:
    row = analysis[analysis['player_name'] == name]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"\n{r['player_name']} ({r['draft_year']}):")
        print(f"   Draft Pick: {r['pick']}")
        print(f"   Draft Capital Score: {r['draft_capital_score']:.1f}")
        print(f"   NEW Breakout Score: {r['breakout_score_new']:.0f}")
        print(f"   NFL PPG: {r['best_ppr']/17:.1f}")
        print(f"   Hit24: {'YES' if r['hit24'] == 1 else 'no'}")

# ============================================================================
# STEP 6: Calculate full SLAP scores for 2026 WRs
# ============================================================================
print("\n" + "="*80)
print("6. 2026 WR RANKINGS (with Age-Only Breakout)")
print("="*80)

# Calculate draft capital score for 2026 WRs
def calc_dc_score(pick):
    if pd.isna(pick) or pick <= 0:
        return np.nan
    return 1 / np.sqrt(pick)

prospects_wrs_2026['draft_capital_raw'] = prospects_wrs_2026['projected_pick'].apply(
    lambda x: calc_dc_score(float(x)) if pd.notna(x) else np.nan
)

# Normalize draft capital (within 2026 class)
dc_mean = prospects_wrs_2026['draft_capital_raw'].mean()
dc_std = prospects_wrs_2026['draft_capital_raw'].std()
prospects_wrs_2026['draft_capital_score'] = prospects_wrs_2026['draft_capital_raw'].apply(
    lambda x: 50 + ((x - dc_mean) / dc_std) * 15 if pd.notna(x) else np.nan
)

# Normalize breakout score (within 2026 class)
br_mean = prospects_wrs_2026['breakout_score_new'].mean()
br_std = prospects_wrs_2026['breakout_score_new'].std()
prospects_wrs_2026['breakout_score_norm'] = prospects_wrs_2026['breakout_score_new'].apply(
    lambda x: 50 + ((x - br_mean) / br_std) * 15 if pd.notna(x) else np.nan
)

# Calculate SLAP score (50% DC + 35% Breakout, no athletic yet)
# Redistribute weights: DC = 50/(50+35) = 58.8%, BR = 35/(50+35) = 41.2%
prospects_wrs_2026['slap_score'] = (
    prospects_wrs_2026['draft_capital_score'] * 0.588 +
    prospects_wrs_2026['breakout_score_norm'] * 0.412
)

# Calculate delta (difference from draft-only)
prospects_wrs_2026['delta'] = prospects_wrs_2026['slap_score'] - prospects_wrs_2026['draft_capital_score']

# Sort and show top 10
top10 = prospects_wrs_2026.nlargest(10, 'slap_score')[
    ['player_name', 'school', 'projected_pick', 'age', 'breakout_score_new',
     'draft_capital_score', 'breakout_score_norm', 'slap_score', 'delta']
]

print("\n" + "-"*100)
print(f"{'Rank':<5} {'Player':<22} {'School':<15} {'Pick':>5} {'Age':>4} {'BR_Raw':>7} {'DC':>6} {'BR':>6} {'SLAP':>6} {'Delta':>7}")
print("-"*100)

for i, (_, row) in enumerate(top10.iterrows(), 1):
    try:
        age_str = f"{float(row['age']):.0f}" if pd.notna(row['age']) and row['age'] != 'MISSING' else "?"
    except:
        age_str = "?"
    delta_str = f"+{row['delta']:.1f}" if row['delta'] > 0 else f"{row['delta']:.1f}"
    print(f"{i:<5} {row['player_name']:<22} {row['school']:<15} {row['projected_pick']:>5.0f} "
          f"{age_str:>4} {row['breakout_score_new']:>7.0f} {row['draft_capital_score']:>6.1f} "
          f"{row['breakout_score_norm']:>6.1f} {row['slap_score']:>6.1f} {delta_str:>7}")

# ============================================================================
# STEP 7: Show biggest positive deltas (age helps them)
# ============================================================================
print("\n" + "="*80)
print("7. BIGGEST POSITIVE DELTAS (Model likes more than draft slot)")
print("="*80)

pos_delta = prospects_wrs_2026[prospects_wrs_2026['delta'] > 0].nlargest(5, 'delta')
print("\n" + "-"*80)
print(f"{'Player':<25} {'Pick':>5} {'Age':>4} {'SLAP':>6} {'Delta':>7}")
print("-"*80)
for _, row in pos_delta.iterrows():
    try:
        age_str = f"{float(row['age']):.0f}" if pd.notna(row['age']) and row['age'] != 'MISSING' else "?"
    except:
        age_str = "?"
    print(f"{row['player_name']:<25} {row['projected_pick']:>5.0f} {age_str:>4} "
          f"{row['slap_score']:>6.1f} +{row['delta']:.1f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Model Update Complete:
- WR Breakout now uses AGE-ONLY scoring (no dominator input)
- Breakout coefficient is {"POSITIVE ✓" if model.coef_[0][1] > 0 else "NEGATIVE ✗"} ({model.coef_[0][1]:.3f})
- Correlation with NFL PPG: r = {r_new:.3f}

Key Changes:
- Younger breakouts get higher scores (age 18 = 100, age 23 = 30)
- Players who never hit 20% dominator get score = 25 (not penalized further)
- Draft capital still carries 50% weight (NFL evaluation)
""")

# Save updated breakout scores for backtest
backtest_wrs[['player_name', 'draft_year', 'breakout_age', 'breakout_score_new']].to_csv(
    'data/wr_breakout_scores_age_only.csv', index=False
)
print("Saved: data/wr_breakout_scores_age_only.csv")
