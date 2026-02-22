"""
RB RAS Analysis - Does Athleticism Matter for Running Backs?

Same rigorous analysis we did for WRs:
1. Raw correlations
2. Multiple regression (RAS after controlling for DC)
3. Compare RB vs WR results
4. Subgroup analysis
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("RB RAS ANALYSIS - DOES ATHLETICISM MATTER FOR RUNNING BACKS?")
print("=" * 90)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n--- Loading data ---")

rb = pd.read_csv('data/rb_backtest_2015_2024.csv')
print(f"Loaded {len(rb)} RBs from 2015-2024")

# Filter to players with outcomes (exclude 2024 rookies who haven't played enough)
rb = rb[rb['draft_year'] <= 2023].copy()
print(f"After excluding 2024 rookies: {len(rb)} RBs")

# Create key variables
rb['inv_sqrt_pick'] = 1 / np.sqrt(rb['pick'])
rb['has_ras'] = rb['RAS'].notna().astype(int)

# Filter to those with RAS and PPG
rb_with_ras = rb[rb['RAS'].notna() & (rb['best_ppg'] > 0)].copy()
print(f"RBs with RAS and fantasy production: {len(rb_with_ras)}")

# ============================================================================
# PART 1: RAW CORRELATIONS FOR RBs
# ============================================================================
print("\n" + "=" * 90)
print("PART 1: RAW CORRELATIONS FOR RBs")
print("=" * 90)

# Draft capital vs PPG
r_dc, p_dc = stats.pearsonr(rb_with_ras['inv_sqrt_pick'], rb_with_ras['best_ppg'])
print(f"\nDraft Capital (1/√pick) vs NFL PPG:")
print(f"  Pearson r = {r_dc:.3f}, p-value = {p_dc:.4f}")
print(f"  {'SIGNIFICANT' if p_dc < 0.05 else 'NOT significant'} at p<0.05")

# RAS vs PPG
r_ras, p_ras = stats.pearsonr(rb_with_ras['RAS'], rb_with_ras['best_ppg'])
print(f"\nRAS vs NFL PPG:")
print(f"  Pearson r = {r_ras:.3f}, p-value = {p_ras:.4f}")
print(f"  {'SIGNIFICANT' if p_ras < 0.05 else 'NOT significant'} at p<0.05")

# Spearman (rank) correlations
rho_dc, p_rho_dc = stats.spearmanr(rb_with_ras['inv_sqrt_pick'], rb_with_ras['best_ppg'])
rho_ras, p_rho_ras = stats.spearmanr(rb_with_ras['RAS'], rb_with_ras['best_ppg'])

print(f"\nSpearman (rank) correlations:")
print(f"  DC vs PPG:  rho = {rho_dc:.3f}, p = {p_rho_dc:.4f}")
print(f"  RAS vs PPG: rho = {rho_ras:.3f}, p = {p_rho_ras:.4f}")

# ============================================================================
# PART 2: DOES RAS ADD VALUE BEYOND DRAFT CAPITAL?
# ============================================================================
print("\n" + "=" * 90)
print("PART 2: DOES RAS ADD VALUE BEYOND DRAFT CAPITAL?")
print("=" * 90)

try:
    import statsmodels.api as sm

    # Model 1: DC only
    X_dc = sm.add_constant(rb_with_ras['inv_sqrt_pick'])
    model_dc = sm.OLS(rb_with_ras['best_ppg'], X_dc).fit()

    # Model 2: DC + RAS
    X_full = sm.add_constant(rb_with_ras[['inv_sqrt_pick', 'RAS']])
    model_full = sm.OLS(rb_with_ras['best_ppg'], X_full).fit()

    print("\n--- Model 1: DC Only ---")
    print(f"  R² = {model_dc.rsquared:.4f}")
    print(f"  Adj R² = {model_dc.rsquared_adj:.4f}")
    print(f"  DC coefficient: {model_dc.params['inv_sqrt_pick']:.3f}, p = {model_dc.pvalues['inv_sqrt_pick']:.4f}")

    print("\n--- Model 2: DC + RAS ---")
    print(f"  R² = {model_full.rsquared:.4f}")
    print(f"  Adj R² = {model_full.rsquared_adj:.4f}")
    print(f"  DC coefficient:  {model_full.params['inv_sqrt_pick']:.3f}, p = {model_full.pvalues['inv_sqrt_pick']:.4f}")
    print(f"  RAS coefficient: {model_full.params['RAS']:.3f}, p = {model_full.pvalues['RAS']:.4f}")

    # Delta R²
    delta_r2 = model_full.rsquared - model_dc.rsquared
    print(f"\n--- Incremental Value of RAS ---")
    print(f"  ΔR² = {delta_r2:.4f} ({delta_r2*100:.2f}% additional variance explained)")
    print(f"  RAS p-value: {model_full.pvalues['RAS']:.4f}")
    print(f"  RAS {'IS' if model_full.pvalues['RAS'] < 0.05 else 'is NOT'} significant after controlling for DC")

    # Partial correlation
    # Residualize PPG on DC, then correlate with RAS
    resid_ppg = model_dc.resid
    r_partial, p_partial = stats.pearsonr(rb_with_ras['RAS'], resid_ppg)
    print(f"\n--- Partial Correlation (RAS | DC) ---")
    print(f"  r_partial = {r_partial:.3f}, p = {p_partial:.4f}")

    # How much does DC explain?
    dc_pct = model_dc.rsquared / model_full.rsquared * 100 if model_full.rsquared > 0 else 100
    print(f"\n--- Variance Decomposition ---")
    print(f"  DC alone explains: {model_dc.rsquared*100:.1f}% of variance")
    print(f"  DC+RAS explains:   {model_full.rsquared*100:.1f}% of variance")
    print(f"  DC accounts for:   {dc_pct:.1f}% of the full model's explanatory power")

except ImportError:
    print("statsmodels not available, skipping regression analysis")

# ============================================================================
# PART 3: COMPARE RB vs WR
# ============================================================================
print("\n" + "=" * 90)
print("PART 3: COMPARE RB vs WR")
print("=" * 90)

# Load WR data for comparison
wr = pd.read_csv('data/wr_backtest_expanded_final.csv')
wr = wr[wr['draft_year'] <= 2023].copy()
wr['inv_sqrt_pick'] = 1 / np.sqrt(wr['pick'])
wr_with_ras = wr[wr['RAS'].notna() & (wr['best_ppr'] > 0)].copy()
wr_with_ras['best_ppg'] = wr_with_ras['best_ppr'] / 17

# WR correlations
r_dc_wr, p_dc_wr = stats.pearsonr(wr_with_ras['inv_sqrt_pick'], wr_with_ras['best_ppg'])
r_ras_wr, p_ras_wr = stats.pearsonr(wr_with_ras['RAS'], wr_with_ras['best_ppg'])

# WR regression
X_dc_wr = sm.add_constant(wr_with_ras['inv_sqrt_pick'])
model_dc_wr = sm.OLS(wr_with_ras['best_ppg'], X_dc_wr).fit()

X_full_wr = sm.add_constant(wr_with_ras[['inv_sqrt_pick', 'RAS']])
model_full_wr = sm.OLS(wr_with_ras['best_ppg'], X_full_wr).fit()

# Partial correlation for WR
resid_ppg_wr = model_dc_wr.resid
r_partial_wr, p_partial_wr = stats.pearsonr(wr_with_ras['RAS'], resid_ppg_wr)

print("\n" + "-" * 70)
print(f"{'Metric':<45} {'WRs':>12} {'RBs':>12}")
print("-" * 70)
print(f"{'Sample size (with RAS + production)':<45} {len(wr_with_ras):>12} {len(rb_with_ras):>12}")
print(f"{'DC correlation with PPG (r)':<45} {r_dc_wr:>12.3f} {r_dc:>12.3f}")
print(f"{'RAS correlation with PPG (r)':<45} {r_ras_wr:>12.3f} {r_ras:>12.3f}")
print(f"{'RAS p-value (raw)':<45} {p_ras_wr:>12.4f} {p_ras:>12.4f}")
print(f"{'DC R² (alone)':<45} {model_dc_wr.rsquared:>12.3f} {model_dc.rsquared:>12.3f}")
print(f"{'DC+RAS R²':<45} {model_full_wr.rsquared:>12.3f} {model_full.rsquared:>12.3f}")
print(f"{'RAS p-value (after DC control)':<45} {model_full_wr.pvalues['RAS']:>12.4f} {model_full.pvalues['RAS']:>12.4f}")
print(f"{'Partial r (RAS | DC)':<45} {r_partial_wr:>12.3f} {r_partial:>12.3f}")
print(f"{'DC % of model variance':<45} {(model_dc_wr.rsquared/model_full_wr.rsquared*100):>11.1f}% {(model_dc.rsquared/model_full.rsquared*100):>11.1f}%")
print("-" * 70)

# Statistical comparison
print("\n--- Does athleticism matter MORE for RBs? ---")
print(f"  RAS raw correlation: WRs = {r_ras_wr:.3f}, RBs = {r_ras:.3f}")
if abs(r_ras) > abs(r_ras_wr):
    print(f"  RBs show STRONGER raw correlation with RAS ({abs(r_ras) - abs(r_ras_wr):.3f} difference)")
else:
    print(f"  WRs show stronger raw correlation with RAS ({abs(r_ras_wr) - abs(r_ras):.3f} difference)")

print(f"\n  RAS partial correlation: WRs = {r_partial_wr:.3f}, RBs = {r_partial:.3f}")
if abs(r_partial) > abs(r_partial_wr):
    print(f"  RBs show STRONGER partial correlation after DC control")
else:
    print(f"  WRs show stronger partial correlation after DC control")

# ============================================================================
# PART 4: RAS SUBGROUP ANALYSIS FOR RBs
# ============================================================================
print("\n" + "=" * 90)
print("PART 4: RAS SUBGROUP ANALYSIS FOR RBs")
print("=" * 90)

# Elite RAS (9+)
elite_ras = rb_with_ras[rb_with_ras['RAS'] >= 9]
# Good RAS (7-9)
good_ras = rb_with_ras[(rb_with_ras['RAS'] >= 7) & (rb_with_ras['RAS'] < 9)]
# Average RAS (5-7)
avg_ras = rb_with_ras[(rb_with_ras['RAS'] >= 5) & (rb_with_ras['RAS'] < 7)]
# Poor RAS (<5)
poor_ras = rb_with_ras[rb_with_ras['RAS'] < 5]

print(f"\n{'RAS Group':<20} {'N':>6} {'Hit24 Rate':>12} {'Avg PPG':>10} {'Avg Pick':>10}")
print("-" * 65)

for name, group in [('Elite (9+)', elite_ras), ('Good (7-9)', good_ras),
                     ('Average (5-7)', avg_ras), ('Poor (<5)', poor_ras)]:
    if len(group) > 0:
        hit_rate = group['hit24'].mean() * 100
        avg_ppg = group['best_ppg'].mean()
        avg_pick = group['pick'].mean()
        print(f"{name:<20} {len(group):>6} {hit_rate:>11.1f}% {avg_ppg:>10.1f} {avg_pick:>10.1f}")
    else:
        print(f"{name:<20} {0:>6} {'N/A':>12} {'N/A':>10} {'N/A':>10}")

# Compare to WR subgroups
print("\n--- Same analysis for WRs (comparison) ---")

elite_ras_wr = wr_with_ras[wr_with_ras['RAS'] >= 9]
good_ras_wr = wr_with_ras[(wr_with_ras['RAS'] >= 7) & (wr_with_ras['RAS'] < 9)]
avg_ras_wr = wr_with_ras[(wr_with_ras['RAS'] >= 5) & (wr_with_ras['RAS'] < 7)]
poor_ras_wr = wr_with_ras[wr_with_ras['RAS'] < 5]

print(f"\n{'RAS Group':<20} {'N':>6} {'Hit24 Rate':>12} {'Avg PPG':>10} {'Avg Pick':>10}")
print("-" * 65)

for name, group in [('Elite (9+)', elite_ras_wr), ('Good (7-9)', good_ras_wr),
                     ('Average (5-7)', avg_ras_wr), ('Poor (<5)', poor_ras_wr)]:
    if len(group) > 0:
        hit_rate = group['hit24'].mean() * 100
        avg_ppg = group['best_ppg'].mean()
        avg_pick = group['pick'].mean()
        print(f"{name:<20} {len(group):>6} {hit_rate:>11.1f}% {avg_ppg:>10.1f} {avg_pick:>10.1f}")
    else:
        print(f"{name:<20} {0:>6} {'N/A':>12} {'N/A':>10} {'N/A':>10}")

# Spread analysis
print("\n--- RAS Spread Analysis ---")
rb_elite_hit = elite_ras['hit24'].mean() * 100 if len(elite_ras) > 0 else 0
rb_poor_hit = poor_ras['hit24'].mean() * 100 if len(poor_ras) > 0 else 0
rb_spread = rb_elite_hit - rb_poor_hit

wr_elite_hit = elite_ras_wr['hit24'].mean() * 100 if len(elite_ras_wr) > 0 else 0
wr_poor_hit = poor_ras_wr['hit24'].mean() * 100 if len(poor_ras_wr) > 0 else 0
wr_spread = wr_elite_hit - wr_poor_hit

print(f"\nHit rate spread (Elite 9+ vs Poor <5):")
print(f"  RBs: {rb_elite_hit:.1f}% - {rb_poor_hit:.1f}% = {rb_spread:+.1f}%")
print(f"  WRs: {wr_elite_hit:.1f}% - {wr_poor_hit:.1f}% = {wr_spread:+.1f}%")

if rb_spread > wr_spread:
    print(f"\n  RBs show BIGGER spread ({rb_spread - wr_spread:.1f}% larger)")
    print("  → RAS matters MORE for RBs than WRs!")
else:
    print(f"\n  WRs show bigger spread ({wr_spread - rb_spread:.1f}% larger)")
    print("  → RAS matters about the same or less for RBs")

# ============================================================================
# PART 5: DRAFT ROUND ANALYSIS
# ============================================================================
print("\n" + "=" * 90)
print("PART 5: RAS BY DRAFT ROUND (RBs)")
print("=" * 90)

# Does RAS matter more for later picks?
for rnd in [1, 2, 3, '4-7']:
    if rnd == '4-7':
        group = rb_with_ras[rb_with_ras['round'] >= 4]
    else:
        group = rb_with_ras[rb_with_ras['round'] == rnd]

    if len(group) >= 10:
        r, p = stats.pearsonr(group['RAS'], group['best_ppg'])
        hit_rate = group['hit24'].mean() * 100
        print(f"\nRound {rnd}: n={len(group)}, hit rate={hit_rate:.1f}%")
        print(f"  RAS vs PPG: r={r:.3f}, p={p:.4f}")
    else:
        print(f"\nRound {rnd}: n={len(group)} (too few for analysis)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("SUMMARY: DOES RAS MATTER FOR RBs?")
print("=" * 90)

print(f"""
KEY FINDINGS:

1. RAW CORRELATION (RAS vs PPG)
   - RBs: r = {r_ras:.3f}, p = {p_ras:.4f} ({'Significant' if p_ras < 0.05 else 'NOT significant'})
   - WRs: r = {r_ras_wr:.3f}, p = {p_ras_wr:.4f} ({'Significant' if p_ras_wr < 0.05 else 'NOT significant'})

2. AFTER CONTROLLING FOR DRAFT CAPITAL
   - RBs: RAS p = {model_full.pvalues['RAS']:.4f} ({'Significant' if model_full.pvalues['RAS'] < 0.05 else 'NOT significant'})
   - WRs: RAS p = {model_full_wr.pvalues['RAS']:.4f} ({'Significant' if model_full_wr.pvalues['RAS'] < 0.05 else 'NOT significant'})

3. VARIANCE EXPLAINED
   - RBs: DC alone = {model_dc.rsquared*100:.1f}%, DC+RAS = {model_full.rsquared*100:.1f}%
   - WRs: DC alone = {model_dc_wr.rsquared*100:.1f}%, DC+RAS = {model_full_wr.rsquared*100:.1f}%

4. HIT RATE SPREAD (Elite vs Poor RAS)
   - RBs: {rb_spread:+.1f}%
   - WRs: {wr_spread:+.1f}%

CONCLUSION: {"RAS matters MORE for RBs" if (model_full.pvalues['RAS'] < 0.05 or rb_spread > wr_spread + 5) else "RAS does NOT matter more for RBs than WRs"}
""")
