"""
HONEST MODEL EVALUATION - EXPANDED DATASET (2015-2024)

Now with 244+ WRs instead of 92. Do the conclusions change?
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("HONEST MODEL EVALUATION - EXPANDED DATASET")
print("=" * 90)

# Load expanded dataset
wr = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"\nLoaded {len(wr)} WRs (2015-2024)")

# Calculate derived variables
wr['inv_sqrt_pick'] = 1 / np.sqrt(wr['pick'])
wr['log_pick'] = np.log(wr['pick'])
wr['best_ppg'] = wr['best_ppr'] / 17

# ============================================================================
# SAMPLE SIZE COMPARISON
# ============================================================================
print("\n" + "=" * 90)
print("SAMPLE SIZE COMPARISON: OLD vs NEW")
print("=" * 90)

print("""
                            OLD (2020-2023)    NEW (2015-2024)
------------------------------------------------------------
Total WRs                          131              326
Have RAS                           108              282
Have Breakout Age                  111              281
Have ALL THREE                      92              244
------------------------------------------------------------
IMPROVEMENT: 2.7x more complete data!
""")

# Complete data mask
mask_complete = (
    wr['pick'].notna() &
    wr['RAS'].notna() &
    wr['breakout_age'].notna() &
    wr['best_ppg'].notna()
)
n_complete = mask_complete.sum()
print(f"Complete data for analysis: n = {n_complete}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 90)
print("CORRELATION ANALYSIS (n = {})".format(n_complete))
print("=" * 90)

wr_complete = wr[mask_complete].copy()

predictors = [
    ('1/sqrt(pick)', 'inv_sqrt_pick', 1),
    ('log(pick)', 'log_pick', -1),  # Flip sign (lower is better)
    ('Breakout Age', 'breakout_age', -1),  # Flip sign (lower is better)
    ('RAS', 'RAS', 1),
]

print("\n" + "-" * 90)
print(f"{'Predictor':<20} {'n':<8} {'r (PPG)':<12} {'p-value':<12} {'Significant?':<15}")
print("-" * 90)

for name, col, sign in predictors:
    r, p = stats.pearsonr(wr_complete[col], wr_complete['best_ppg'])
    r = r * sign  # Adjust sign so positive = better
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "NO"
    print(f"{name:<20} {n_complete:<8} {r:>+.3f}       {p:<12.4f} {sig:<15}")

# ============================================================================
# R² DECOMPOSITION
# ============================================================================
print("\n" + "=" * 90)
print("R² DECOMPOSITION: HOW MUCH DOES EACH VARIABLE ADD?")
print("=" * 90)

X_dc = wr_complete[['inv_sqrt_pick']].values
X_dc_br = wr_complete[['inv_sqrt_pick', 'breakout_age']].values
X_full = wr_complete[['inv_sqrt_pick', 'breakout_age', 'RAS']].values
y = wr_complete['best_ppg'].values

# Fit models
model_dc = LinearRegression().fit(X_dc, y)
model_dc_br = LinearRegression().fit(X_dc_br, y)
model_full = LinearRegression().fit(X_full, y)

r2_dc = r2_score(y, model_dc.predict(X_dc))
r2_dc_br = r2_score(y, model_dc_br.predict(X_dc_br))
r2_full = r2_score(y, model_full.predict(X_full))

print("\n" + "-" * 70)
print(f"{'Model':<35} {'R²':<10} {'Δ R²':<10} {'% of Total':<12}")
print("-" * 70)
print(f"{'Draft Capital only':<35} {r2_dc:.4f}")
print(f"{'+ Breakout Age':<35} {r2_dc_br:.4f}     {r2_dc_br - r2_dc:>+.4f}     {(r2_dc_br - r2_dc)/r2_full*100:>5.1f}%")
print(f"{'+ RAS (Full Model)':<35} {r2_full:.4f}     {r2_full - r2_dc_br:>+.4f}     {(r2_full - r2_dc_br)/r2_full*100:>5.1f}%")
print("-" * 70)

dc_pct = r2_dc / r2_full * 100
print(f"\nDraft Capital explains {dc_pct:.1f}% of the full model's R²")
print(f"Breakout Age adds {(r2_dc_br - r2_dc)/r2_full*100:.1f}%")
print(f"RAS adds {(r2_full - r2_dc_br)/r2_full*100:.1f}%")

# ============================================================================
# STATISTICAL SIGNIFICANCE (REGRESSION)
# ============================================================================
print("\n" + "=" * 90)
print("REGRESSION COEFFICIENTS (STATISTICAL SIGNIFICANCE)")
print("=" * 90)

X_sm = sm.add_constant(wr_complete[['inv_sqrt_pick', 'breakout_age', 'RAS']])
model_sm = sm.OLS(wr_complete['best_ppg'], X_sm).fit()

print("\n" + "-" * 70)
print(f"{'Variable':<20} {'Coefficient':<12} {'Std Error':<12} {'p-value':<12} {'Sig?':<8}")
print("-" * 70)

for var in ['inv_sqrt_pick', 'breakout_age', 'RAS']:
    coef = model_sm.params[var]
    se = model_sm.bse[var]
    pval = model_sm.pvalues[var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "NO"
    print(f"{var:<20} {coef:>+10.3f}   {se:>10.3f}   {pval:>10.4f}   {sig:<8}")

# ============================================================================
# HIT RATE ANALYSIS
# ============================================================================
print("\n" + "=" * 90)
print("HIT RATE ANALYSIS (TOP-24)")
print("=" * 90)

# DC-only ranking (by pick)
wr_complete_sorted = wr_complete.sort_values('pick')
top24_dc = wr_complete_sorted.head(24)
hit_rate_dc = top24_dc['hit24'].mean() * 100

# Full model ranking
wr_complete['full_pred'] = model_full.predict(X_full)
top24_full = wr_complete.nlargest(24, 'full_pred')
hit_rate_full = top24_full['hit24'].mean() * 100

print(f"\nTop-24 by draft pick:    {top24_dc['hit24'].sum()}/24 hits = {hit_rate_dc:.1f}%")
print(f"Top-24 by full model:    {top24_full['hit24'].sum()}/24 hits = {hit_rate_full:.1f}%")
print(f"Improvement:             {hit_rate_full - hit_rate_dc:+.1f} percentage points")

# ============================================================================
# COMPARISON: OLD vs NEW CONCLUSIONS
# ============================================================================
print("\n" + "=" * 90)
print("COMPARISON: OLD vs NEW CONCLUSIONS")
print("=" * 90)

print("""
                              OLD (n=92)         NEW (n={})
------------------------------------------------------------------------
DC correlation (r)            0.63               {:.3f}
Breakout Age p-value          0.109 (NO)         {:.4f} ({})
RAS p-value                   0.484 (NO)         {:.4f} ({})
------------------------------------------------------------------------
DC % of R²                    93%                {:.1f}%
Full model R²                 0.336              {:.4f}
------------------------------------------------------------------------
Top-24 hit rate (DC)          54.2%              {:.1f}%
Top-24 hit rate (Full)        45.8%              {:.1f}%
------------------------------------------------------------------------
""".format(
    n_complete,
    stats.pearsonr(wr_complete['inv_sqrt_pick'], wr_complete['best_ppg'])[0],
    model_sm.pvalues['breakout_age'],
    "YES" if model_sm.pvalues['breakout_age'] < 0.05 else "NO",
    model_sm.pvalues['RAS'],
    "YES" if model_sm.pvalues['RAS'] < 0.05 else "NO",
    dc_pct,
    r2_full,
    hit_rate_dc,
    hit_rate_full
))

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "=" * 90)
print("FINAL VERDICT")
print("=" * 90)

ba_sig = model_sm.pvalues['breakout_age'] < 0.05
ras_sig = model_sm.pvalues['RAS'] < 0.05

if ba_sig and ras_sig:
    verdict = "BOTH breakout age AND RAS are significant with more data!"
elif ba_sig:
    verdict = "Breakout age IS significant, but RAS is still not."
elif ras_sig:
    verdict = "RAS IS significant, but breakout age is still not."
else:
    verdict = "Neither variable is significant even with 2.7x more data."

print(f"""
WITH 2.7x MORE DATA ({n_complete} vs 92 WRs):

{verdict}

Draft Capital still explains {dc_pct:.0f}% of the model's predictive power.

The other variables add {r2_full - r2_dc:.4f} R² ({100 - dc_pct:.1f}% of total).
""")

if hit_rate_full > hit_rate_dc:
    print(f"The full model DOES improve hit rate by {hit_rate_full - hit_rate_dc:.1f} percentage points.")
elif hit_rate_full < hit_rate_dc:
    print(f"The full model HURTS hit rate by {hit_rate_dc - hit_rate_full:.1f} percentage points.")
else:
    print("The full model has the same hit rate as DC-only.")

print("\n" + "=" * 90)

# Save results
wr_complete.to_csv('output/wr_evaluation_expanded.csv', index=False)
print("Saved detailed results to output/wr_evaluation_expanded.csv")
