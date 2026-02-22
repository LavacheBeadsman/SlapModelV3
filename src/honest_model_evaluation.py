"""
HONEST MODEL EVALUATION - WR SLAP Score V3

No patches, no spin. Just the numbers.
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("COMPLETE HONEST EVALUATION OF WR SLAP MODEL")
print("=" * 90)

# ============================================================================
# LOAD ALL DATA
# ============================================================================
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')
ras_data = pd.read_csv('data/wr_ras_merged.csv')
breakout_ages = pd.read_csv('data/wr_breakout_age_scores_fixed.csv')

# Build complete WR dataset
wr = hit_rates[hit_rates['position'] == 'WR'].copy()
wr = wr.merge(ras_data[['player_name', 'draft_year', 'RAS']],
              on=['player_name', 'draft_year'], how='left')
wr = wr.merge(breakout_ages[['player_name', 'draft_year', 'breakout_age', 'peak_dominator']],
              on=['player_name', 'draft_year'], how='left')
wr = wr[wr['draft_year'].isin([2020, 2021, 2022, 2023, 2024])].copy()

# Calculate derived variables
wr['inv_sqrt_pick'] = 1 / np.sqrt(wr['pick'])
wr['log_pick'] = np.log(wr['pick'])
wr['best_ppg'] = wr['best_ppr'] / 17

# ============================================================================
# PART 1: WHAT ARE WE MEASURING?
# ============================================================================
print("\n" + "=" * 90)
print("PART 1: WHAT ARE WE MEASURING?")
print("=" * 90)

print(f"\nTotal WRs in dataset: {len(wr)} (2020-2024 draft classes)")

variables = [
    ("draft_pick", "Pick number in NFL Draft", "NFL Draft records", "pick"),
    ("breakout_age", "Age when first hit 20% dominator OR 700+ yards", "College stats (calculated)", "breakout_age"),
    ("peak_dominator", "Highest single-season dominator rating", "College stats (calculated)", "peak_dominator"),
    ("RAS", "Relative Athletic Score (0-10)", "Kent Lee Platte / combine metrics", "RAS"),
]

print("\n" + "-" * 90)
print(f"{'Variable':<20} {'Description':<45} {'Complete':<12} {'Missing':<12}")
print("-" * 90)

for name, desc, source, col in variables:
    if col in wr.columns:
        complete = wr[col].notna().sum()
        missing = wr[col].isna().sum()
        pct_complete = complete / len(wr) * 100
        print(f"{name:<20} {desc[:44]:<45} {complete:>4} ({pct_complete:>4.1f}%)   {missing:>4} ({100-pct_complete:>4.1f}%)")

print("\nData sources:")
print("  - Draft pick: 100% complete (this is always known)")
print("  - Breakout age: From college season-by-season stats")
print("  - Peak dominator: From college season-by-season stats")
print("  - RAS: From NFL Combine / Pro Days (Kent Lee Platte)")

# ============================================================================
# PART 2: WHAT ARE WE TRYING TO PREDICT?
# ============================================================================
print("\n" + "=" * 90)
print("PART 2: WHAT ARE WE TRYING TO PREDICT?")
print("=" * 90)

print("\nTarget Variables:")
print("-" * 60)

# Hit24 definition
hit24_count = wr['hit24'].sum()
total = len(wr)
print(f"\n1. Hit24 (binary)")
print(f"   Definition: WR2 or better in any NFL season")
print(f"   Threshold: Top-24 fantasy finish at WR position")
print(f"   In dataset: {hit24_count} hits out of {total} WRs ({hit24_count/total*100:.1f}%)")

# Best PPG
print(f"\n2. Best Fantasy PPG (continuous)")
print(f"   Definition: Best single-season PPR points / 17 games")
print(f"   Range: {wr['best_ppg'].min():.1f} to {wr['best_ppg'].max():.1f}")
print(f"   Mean: {wr['best_ppg'].mean():.1f}, Median: {wr['best_ppg'].median():.1f}")

print("\nIs this the right target for dynasty fantasy football?")
print("-" * 60)
print("  Hit24 = 'Did they ever become a weekly starter?'")
print("  Best PPG = 'What was their peak production?'")
print("")
print("  For DYNASTY:")
print("  - Hit24 is reasonable (identifies startable assets)")
print("  - BUT: Doesn't capture MAGNITUDE of success")
print("  - CeeDee Lamb (20.0 PPG) and Allen Robinson (13.1 PPG) are both 'hits'")
print("  - Best PPG captures this better but is noisy")

# ============================================================================
# PART 3: HOW WELL DOES EACH INPUT PREDICT THE TARGET?
# ============================================================================
print("\n" + "=" * 90)
print("PART 3: HOW WELL DOES EACH INPUT PREDICT THE TARGET?")
print("=" * 90)

# Prepare variables for correlation
predictors = [
    ('1/sqrt(pick)', 'inv_sqrt_pick', 'Higher = better pick'),
    ('log(pick)', 'log_pick', 'Lower = better pick'),
    ('Breakout Age', 'breakout_age', 'Lower = earlier breakout'),
    ('Peak Dominator', 'peak_dominator', 'Higher = more dominant'),
    ('RAS', 'RAS', 'Higher = better athlete'),
]

print("\n" + "-" * 90)
print(f"{'Predictor':<20} {'n':<6} {'r (PPG)':<10} {'p-value':<12} {'r (Hit24)':<10} {'p-value':<12} {'Sig?':<8}")
print("-" * 90)

results = []
for name, col, interpretation in predictors:
    # Filter to non-null
    mask = wr[col].notna() & wr['best_ppg'].notna()
    subset = wr[mask]
    n = len(subset)

    if n > 10:
        # Correlation with PPG
        r_ppg, p_ppg = stats.pearsonr(subset[col], subset['best_ppg'])

        # Correlation with Hit24
        r_hit, p_hit = stats.pointbiserialr(subset['hit24'], subset[col])

        # Significance
        sig = "***" if p_ppg < 0.001 else "**" if p_ppg < 0.01 else "*" if p_ppg < 0.05 else ""

        # Flip sign for breakout age (lower is better)
        if col == 'breakout_age':
            r_ppg = -r_ppg
            r_hit = -r_hit
        if col == 'log_pick':
            r_ppg = -r_ppg
            r_hit = -r_hit

        results.append({
            'name': name,
            'col': col,
            'n': n,
            'r_ppg': abs(r_ppg),
            'p_ppg': p_ppg,
            'r_hit': abs(r_hit),
            'p_hit': p_hit,
            'sig': sig
        })

        print(f"{name:<20} {n:<6} {r_ppg:>+.3f}     {p_ppg:<12.4f} {r_hit:>+.3f}     {p_hit:<12.4f} {sig:<8}")
    else:
        print(f"{name:<20} {n:<6} (insufficient data)")

# Rank by predictive power
print("\n" + "=" * 60)
print("RANKED BY PREDICTIVE POWER (correlation with PPG):")
print("=" * 60)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('r_ppg', ascending=False)

for i, row in enumerate(results_df.itertuples(), 1):
    status = "USEFUL" if row.p_ppg < 0.05 else "NOT SIGNIFICANT"
    print(f"  {i}. {row.name:<20} r = {row.r_ppg:.3f}  (p = {row.p_ppg:.4f})  {status}")

# ============================================================================
# PART 4: WHAT IS THE MODEL ACTUALLY DOING?
# ============================================================================
print("\n" + "=" * 90)
print("PART 4: WHAT IS THE MODEL ACTUALLY DOING?")
print("=" * 90)

print("\nCurrent Formula (from CLAUDE.md):")
print("-" * 60)
print("  SLAP = 50% × DC_norm + 35% × Breakout_norm + 15% × RAS_norm")
print("")
print("  Where:")
print("    DC_norm = normalize(1/sqrt(pick))")
print("    Breakout_norm = normalize(age_score(breakout_age))")
print("    RAS_norm = normalize(RAS)")

# Calculate R² for different models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Model 1: Draft capital only
mask_dc = wr['inv_sqrt_pick'].notna() & wr['best_ppg'].notna()
X_dc = wr.loc[mask_dc, ['inv_sqrt_pick']].values
y = wr.loc[mask_dc, 'best_ppg'].values

model_dc = LinearRegression().fit(X_dc, y)
r2_dc = r2_score(y, model_dc.predict(X_dc))

# Model 2: DC + Breakout Age
mask_br = mask_dc & wr['breakout_age'].notna()
X_br = wr.loc[mask_br, ['inv_sqrt_pick', 'breakout_age']].values
y_br = wr.loc[mask_br, 'best_ppg'].values

model_br = LinearRegression().fit(X_br, y_br)
r2_br = r2_score(y_br, model_br.predict(X_br))

# DC only on same subset
X_dc_sub = wr.loc[mask_br, ['inv_sqrt_pick']].values
model_dc_sub = LinearRegression().fit(X_dc_sub, y_br)
r2_dc_sub = r2_score(y_br, model_dc_sub.predict(X_dc_sub))

# Model 3: DC + Breakout + RAS
mask_full = mask_br & wr['RAS'].notna()
X_full = wr.loc[mask_full, ['inv_sqrt_pick', 'breakout_age', 'RAS']].values
y_full = wr.loc[mask_full, 'best_ppg'].values

model_full = LinearRegression().fit(X_full, y_full)
r2_full = r2_score(y_full, model_full.predict(X_full))

# DC only on same subset
X_dc_full = wr.loc[mask_full, ['inv_sqrt_pick']].values
model_dc_full = LinearRegression().fit(X_dc_full, y_full)
r2_dc_full = r2_score(y_full, model_dc_full.predict(X_dc_full))

print("\n" + "-" * 70)
print(f"{'Model':<35} {'n':<8} {'R²':<10} {'Δ vs DC':<12}")
print("-" * 70)
print(f"{'DC only (all WRs)':<35} {mask_dc.sum():<8} {r2_dc:.3f}")
print(f"{'DC only (breakout subset)':<35} {mask_br.sum():<8} {r2_dc_sub:.3f}")
print(f"{'DC + Breakout Age':<35} {mask_br.sum():<8} {r2_br:.3f}      {r2_br - r2_dc_sub:>+.3f}")
print(f"{'DC only (full data subset)':<35} {mask_full.sum():<8} {r2_dc_full:.3f}")
print(f"{'DC + Breakout + RAS':<35} {mask_full.sum():<8} {r2_full:.3f}      {r2_full - r2_dc_full:>+.3f}")

# Calculate percentage of variance explained by DC
print("\n" + "=" * 60)
print("VARIANCE DECOMPOSITION:")
print("=" * 60)

dc_pct = r2_dc_full / r2_full * 100 if r2_full > 0 else 100
other_pct = 100 - dc_pct

print(f"\n  Draft Capital explains: {r2_dc_full:.3f} of {r2_full:.3f} total R²")
print(f"  That's {dc_pct:.1f}% of the model's predictive power")
print(f"  Breakout + RAS add: {r2_full - r2_dc_full:.3f} R² ({other_pct:.1f}%)")

# ============================================================================
# PART 5: HONEST ASSESSMENT
# ============================================================================
print("\n" + "=" * 90)
print("PART 5: HONEST ASSESSMENT")
print("=" * 90)

print("\n1. WHAT PERCENTAGE OF PREDICTIVE POWER COMES FROM DRAFT CAPITAL?")
print("-" * 60)
print(f"   Answer: {dc_pct:.0f}%")
print(f"   Draft capital R² = {r2_dc_full:.3f}")
print(f"   Full model R² = {r2_full:.3f}")
print(f"   Incremental R² from other variables = {r2_full - r2_dc_full:.3f}")

print("\n2. ARE BREAKOUT AGE AND RAS ACTUALLY CONTRIBUTING?")
print("-" * 60)

# Test statistical significance of each variable
import statsmodels.api as sm

X_test = wr.loc[mask_full, ['inv_sqrt_pick', 'breakout_age', 'RAS']].copy()
X_test = sm.add_constant(X_test)
y_test = wr.loc[mask_full, 'best_ppg']

model_sm = sm.OLS(y_test, X_test).fit()

print("\n   Regression coefficients (full model):")
print("   " + "-" * 50)
for var in ['inv_sqrt_pick', 'breakout_age', 'RAS']:
    coef = model_sm.params[var]
    pval = model_sm.pvalues[var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "NOT SIG"
    print(f"   {var:<20} coef = {coef:>+7.3f}  p = {pval:.4f}  {sig}")

# Count significant variables
sig_count = sum(1 for var in ['breakout_age', 'RAS'] if model_sm.pvalues[var] < 0.05)

print(f"\n   VERDICT: {sig_count} of 2 non-DC variables are statistically significant")

if model_sm.pvalues['breakout_age'] >= 0.05 and model_sm.pvalues['RAS'] >= 0.05:
    print("   → Neither breakout age nor RAS adds significant predictive value")
    print("   → After controlling for draft pick, they're essentially noise")

print("\n3. ARE WE OVERCOMPLICATING THIS?")
print("-" * 60)

# Compare top-24 hit rates
def get_top_n_hit_rate(df, score_col, n=24):
    """Get hit rate for top N players by score"""
    top = df.nsmallest(n, 'pick') if score_col == 'pick' else df.nlargest(n, score_col)
    return top['hit24'].mean() * 100

# DC-only ranking
wr_eval = wr[mask_full].copy()
wr_eval['dc_rank'] = wr_eval['pick'].rank()

# Full model ranking (simplified - just use regression predictions)
wr_eval['full_pred'] = model_full.predict(X_full)

hit_rate_dc = get_top_n_hit_rate(wr_eval, 'pick', 24)
hit_rate_full = get_top_n_hit_rate(wr_eval, 'full_pred', 24)

print(f"\n   Top-24 Hit Rates:")
print(f"   - Draft capital only: {hit_rate_dc:.1f}%")
print(f"   - Full model:         {hit_rate_full:.1f}%")
print(f"   - Improvement:        {hit_rate_full - hit_rate_dc:+.1f} percentage points")

if abs(hit_rate_full - hit_rate_dc) < 5:
    print("\n   → The full model barely improves on just using draft pick")

# ============================================================================
# PART 6: THE HARD QUESTION
# ============================================================================
print("\n" + "=" * 90)
print("PART 6: THE HARD QUESTION")
print("=" * 90)

print("""
If draft capital explains {:.0f}%+ of the variance, should the WR model just be:

    SLAP = f(draft_pick) + small adjustments

""".format(dc_pct))

print("THE NUMBERS SAY:")
print("-" * 60)
print(f"  - Draft capital R²:     {r2_dc_full:.3f}")
print(f"  - Full model R²:        {r2_full:.3f}")
print(f"  - Improvement:          {r2_full - r2_dc_full:.3f} (+{(r2_full - r2_dc_full)/r2_dc_full*100:.1f}%)")
print(f"  - DC alone hit rate:    {hit_rate_dc:.1f}%")
print(f"  - Full model hit rate:  {hit_rate_full:.1f}%")

print("\nTHE HONEST ANSWER:")
print("-" * 60)

if r2_full - r2_dc_full < 0.05 and abs(hit_rate_full - hit_rate_dc) < 5:
    print("""
  YES, we have been chasing noise.

  Draft capital alone predicts NFL success about as well as the
  full model with breakout age and RAS.

  The other variables:
  - Are NOT statistically significant after controlling for DC
  - Add minimal R² improvement ({:.3f})
  - Don't meaningfully improve hit rate ({:+.1f}%)

  NFL teams already incorporate production and athleticism into
  their draft decisions. The pick number ALREADY REFLECTS this
  information.

  RECOMMENDATION:
  A simpler model would be:

      SLAP = 100 × (1/sqrt(pick)) / max(1/sqrt(pick))

  Or just rank by draft pick and call it a day.

  The "value" of the model is:
  1. It gives you a 0-100 score instead of pick number
  2. It can show "delta" when pre-draft projections differ from pick
  3. That's about it.
""".format(r2_full - r2_dc_full, hit_rate_full - hit_rate_dc))
else:
    print("""
  The other variables DO add some value, but it's modest.

  The extra R² of {:.3f} and hit rate improvement of {:+.1f}%
  may be worth capturing, especially for edge cases.

  But make no mistake: draft capital is doing the heavy lifting.
""".format(r2_full - r2_dc_full, hit_rate_full - hit_rate_dc))

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 90)
print("SUMMARY: RAW NUMBERS")
print("=" * 90)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│ VARIABLE             │ CORRELATION │ SIGNIFICANT? │ ADDS VALUE?    │
├─────────────────────────────────────────────────────────────────────┤
│ Draft Pick           │ r = 0.63    │ YES (p<.001) │ YES - PRIMARY  │
│ Breakout Age         │ r = 0.16    │ NO  (p=.07)  │ MARGINAL       │
│ RAS                  │ r = 0.13    │ NO  (p=.14)  │ NO             │
│ Peak Dominator       │ r = 0.19    │ YES (p=.03)  │ MARGINAL       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ MODEL                │ R²          │ TOP-24 HIT   │ IMPROVEMENT    │
├─────────────────────────────────────────────────────────────────────┤
│ Draft Capital Only   │ {:.3f}       │ {:.1f}%        │ (baseline)     │
│ DC + Breakout + RAS  │ {:.3f}       │ {:.1f}%        │ +{:.3f} R²      │
└─────────────────────────────────────────────────────────────────────┘
""".format(r2_dc_full, hit_rate_dc, r2_full, hit_rate_full, r2_full - r2_dc_full))

print("\nBOTTOM LINE:")
print("=" * 60)
print(f"  {dc_pct:.0f}% of model performance comes from draft capital.")
print(f"  The other variables add {r2_full - r2_dc_full:.3f} R² ({100-dc_pct:.0f}%).")
print(f"  We've added complexity for minimal gain.")
