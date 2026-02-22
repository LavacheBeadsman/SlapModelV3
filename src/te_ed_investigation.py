"""
TE Early Declare Investigation
===============================
The user correctly notes that top fantasy TEs (Pitts, Bowers, etc.) were early declares.
If ED is highly correlated with DC, partial correlation controlling for DC would mask ED's value.
This script investigates whether ED's "zero value" finding is real or a statistical artifact.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir('/home/user/SlapModelV3')

bt = pd.read_csv('data/te_backtest_master.csv')

# Eval sample
eval_df = bt[bt['draft_year'] < 2025].copy()
print(f"Eval sample: {len(eval_df)} TEs (2015-2024)\n")

# ============================================================================
# 1. BASIC STATS: Early declare rate and overlap with high DC
# ============================================================================

print("=" * 100)
print("1. EARLY DECLARE BASICS")
print("=" * 100)

ed = eval_df[eval_df['early_declare'] == 1]
non_ed = eval_df[eval_df['early_declare'] == 0]

print(f"\n  Early declares:     {len(ed)}/{len(eval_df)} ({len(ed)/len(eval_df):.1%})")
print(f"  Non-early declares: {len(non_ed)}/{len(eval_df)} ({len(non_ed)/len(eval_df):.1%})")

print(f"\n  Average pick — ED: {ed['pick'].mean():.1f}, Non-ED: {non_ed['pick'].mean():.1f}")
print(f"  Average DC  — ED: {ed['pick'].apply(lambda p: max(0,min(100, 100-2.40*(p**0.62-1)))).mean():.1f}, "
      f"Non-ED: {non_ed['pick'].apply(lambda p: max(0,min(100, 100-2.40*(p**0.62-1)))).mean():.1f}")

# Correlation between ED and DC
dc_scores = eval_df['pick'].apply(lambda p: max(0, min(100, 100 - 2.40 * (p ** 0.62 - 1))))
r_ed_dc, p_ed_dc = stats.pearsonr(eval_df['early_declare'], dc_scores)
print(f"\n  Correlation between ED and DC: r={r_ed_dc:+.3f} (p={p_ed_dc:.4f})")
print(f"  {'HIGH MULTICOLLINEARITY' if abs(r_ed_dc) > 0.40 else 'Moderate'if abs(r_ed_dc) > 0.25 else 'Low'} — ", end="")
if abs(r_ed_dc) > 0.40:
    print("ED is heavily confounded with DC. Partial correlation likely absorbs ED's signal.")
elif abs(r_ed_dc) > 0.25:
    print("ED moderately overlaps with DC. Partial correlation may reduce ED's apparent value.")
else:
    print("ED is relatively independent from DC.")

# ============================================================================
# 2. RAW HIT RATES: ED vs non-ED
# ============================================================================

print(f"\n\n{'='*100}")
print("2. RAW HIT RATES (NOT controlling for DC)")
print("=" * 100)

for label, group in [('Early declares', ed), ('Non-early declares', non_ed)]:
    n = len(group)
    t6 = group['top6_10g'].sum() if 'top6_10g' in group.columns else group['top6_8g'].sum()
    t12 = group['top12_10g'].sum() if 'top12_10g' in group.columns else group['top12_8g'].sum()
    ppg = group['best_3yr_ppg_10g'].dropna() if 'best_3yr_ppg_10g' in group.columns else group['best_3yr_ppg_8g'].dropna()
    s10 = group['seasons_over_10ppg_10g'] if 'seasons_over_10ppg_10g' in group.columns else group['seasons_over_10ppg_8g']

    print(f"\n  {label} (N={n}):")
    print(f"    Top6 rate:  {t6}/{n} ({t6/n:.1%})")
    print(f"    Top12 rate: {t12}/{n} ({t12/n:.1%})")
    print(f"    Avg best 3yr PPG: {ppg.mean():.2f} (N={len(ppg)} with data)")
    print(f"    Avg seasons 10+ PPG: {s10.mean():.2f}")

# ============================================================================
# 3. CONTROLLED COMPARISON: Same DC range
# ============================================================================

print(f"\n\n{'='*100}")
print("3. CONTROLLED COMPARISON — Same draft capital range")
print("Does ED help WITHIN the same draft capital tier?")
print("=" * 100)

dc_scores_series = eval_df['pick'].apply(lambda p: max(0, min(100, 100 - 2.40 * (p ** 0.62 - 1))))
eval_df['s_dc'] = dc_scores_series

# Tier analysis: within each DC tier, compare ED vs non-ED
tiers = [(80, 101, 'Elite DC (80+)'), (60, 80, 'Good DC (60-80)'), (40, 60, 'Mid DC (40-60)'), (0, 40, 'Low DC (0-40)')]

for lo, hi, tier_name in tiers:
    tier = eval_df[(eval_df['s_dc'] >= lo) & (eval_df['s_dc'] < hi)]
    tier_ed = tier[tier['early_declare'] == 1]
    tier_non = tier[tier['early_declare'] == 0]

    print(f"\n  {tier_name} (N={len(tier)})")
    for group_name, group in [('ED=1', tier_ed), ('ED=0', tier_non)]:
        n = len(group)
        if n == 0:
            print(f"    {group_name}: N=0")
            continue
        t12 = group['top12_10g'].sum() if 'top12_10g' in group.columns else 0
        ppg_data = group['best_3yr_ppg_10g'].dropna() if 'best_3yr_ppg_10g' in group.columns else pd.Series()
        ppg = ppg_data.mean() if len(ppg_data) > 0 else float('nan')
        s10 = group['seasons_over_10ppg_10g'].mean() if 'seasons_over_10ppg_10g' in group.columns else 0
        ppg_s = f"{ppg:.2f}" if not np.isnan(ppg) else "N/A"
        print(f"    {group_name}: N={n:>3}, Top12={t12}/{n} ({t12/n:.0%}), Avg3yrPPG={ppg_s}, Szn10+={s10:.1f}")

# ============================================================================
# 4. NAME-BY-NAME: Who are the early declares?
# ============================================================================

print(f"\n\n{'='*100}")
print("4. ALL EARLY DECLARES — Did they hit?")
print("=" * 100)

ed_detail = ed.sort_values('pick').copy()
ed_detail['s_dc'] = ed_detail['pick'].apply(lambda p: max(0, min(100, 100 - 2.40 * (p ** 0.62 - 1))))

print(f"\n  {'Name':<25} {'Year':>4} {'Rd':>3} {'Pick':>5} {'DC':>5} {'Top6':>5} {'Top12':>5} {'3yr PPG':>8} {'Szn10+':>6}")
print(f"  {'-'*75}")

for _, r in ed_detail.iterrows():
    t6 = int(r.get('top6_10g', r.get('top6_8g', 0)))
    t12 = int(r.get('top12_10g', r.get('top12_8g', 0)))
    ppg = r.get('best_3yr_ppg_10g', r.get('best_3yr_ppg_8g', np.nan))
    ppg_s = f"{ppg:.2f}" if not np.isnan(ppg) else "N/A"
    s10 = r.get('seasons_over_10ppg_10g', r.get('seasons_over_10ppg_8g', 0))
    print(f"  {r['player_name']:<25} {int(r['draft_year']):>4} {int(r['round']):>3} {int(r['pick']):>5} "
          f"{r['s_dc']:>5.1f} {t6:>5} {t12:>5} {ppg_s:>8} {s10:>6.0f}")

# Count
n_ed_t12 = ed_detail['top12_10g'].sum() if 'top12_10g' in ed_detail.columns else ed_detail['top12_8g'].sum()
print(f"\n  Summary: {int(n_ed_t12)}/{len(ed_detail)} early declares hit top12 ({n_ed_t12/len(ed_detail):.1%})")

# ============================================================================
# 5. NON-EARLY DECLARES WHO HIT — What would ED have missed?
# ============================================================================

print(f"\n\n{'='*100}")
print("5. NON-EARLY DECLARES WHO HIT TOP-12")
print("=" * 100)

non_ed_t12 = non_ed[non_ed.get('top12_10g', non_ed.get('top12_8g', pd.Series(0))) == 1]
non_ed_t12 = non_ed_t12.sort_values('pick')
non_ed_t12['s_dc'] = non_ed_t12['pick'].apply(lambda p: max(0, min(100, 100 - 2.40 * (p ** 0.62 - 1))))

print(f"\n  {'Name':<25} {'Year':>4} {'Rd':>3} {'Pick':>5} {'DC':>5} {'Top6':>5} {'3yr PPG':>8} {'Szn10+':>6}")
print(f"  {'-'*65}")

for _, r in non_ed_t12.iterrows():
    t6 = int(r.get('top6_10g', r.get('top6_8g', 0)))
    ppg = r.get('best_3yr_ppg_10g', r.get('best_3yr_ppg_8g', np.nan))
    ppg_s = f"{ppg:.2f}" if not np.isnan(ppg) else "N/A"
    s10 = r.get('seasons_over_10ppg_10g', r.get('seasons_over_10ppg_8g', 0))
    print(f"  {r['player_name']:<25} {int(r['draft_year']):>4} {int(r['round']):>3} {int(r['pick']):>5} "
          f"{r['s_dc']:>5.1f} {t6:>5} {ppg_s:>8} {s10:>6.0f}")

print(f"\n  {len(non_ed_t12)} non-early-declares hit top12")

# ============================================================================
# 6. THE KEY QUESTION: Does ED predict success AFTER controlling for DC?
# ============================================================================

print(f"\n\n{'='*100}")
print("6. LOGISTIC REGRESSION: ED predicting top12, controlling for DC")
print("=" * 100)

# Manual logistic regression isn't great, let's use a simpler approach:
# Compare observed vs expected hit rates

# For each DC bin, calculate the expected hit rate and compare ED vs non-ED
print(f"\n  DC Bin       ED=1                  ED=0                 ED advantage?")
print(f"  {'-'*80}")

dc_bins = [(70, 101), (50, 70), (0, 50)]
for lo, hi in dc_bins:
    bin_data = eval_df[(eval_df['s_dc'] >= lo) & (eval_df['s_dc'] < hi)]
    bin_ed = bin_data[bin_data['early_declare'] == 1]
    bin_non = bin_data[bin_data['early_declare'] == 0]

    n_ed = len(bin_ed); n_non = len(bin_non)
    t12_ed = bin_ed['top12_10g'].sum() if n_ed > 0 else 0
    t12_non = bin_non['top12_10g'].sum() if n_non > 0 else 0
    rate_ed = t12_ed/n_ed if n_ed > 0 else 0
    rate_non = t12_non/n_non if n_non > 0 else 0

    adv = "YES" if rate_ed > rate_non else ("TIE" if rate_ed == rate_non else "NO")

    ed_s = f"{int(t12_ed)}/{n_ed} ({rate_ed:.0%})" if n_ed > 0 else "N=0"
    non_s = f"{int(t12_non)}/{n_non} ({rate_non:.0%})" if n_non > 0 else "N=0"

    print(f"  DC {lo:>2}-{hi:<3}    {ed_s:<20}  {non_s:<20}  {adv}")

# ============================================================================
# 7. MULTICOLLINEARITY DIAGNOSIS
# ============================================================================

print(f"\n\n{'='*100}")
print("7. MULTICOLLINEARITY DIAGNOSIS")
print("=" * 100)

# Correlation matrix of all components
comps = {'DC': eval_df['pick'].apply(lambda p: max(0, min(100, 100-2.40*(p**0.62-1)))),
         'ED': eval_df['early_declare'] * 100}
r_mat, p_mat = stats.pearsonr(comps['DC'], comps['ED'])
print(f"\n  Correlation of DC score and ED: r={r_mat:+.3f} (p={p_mat:.6f})")
print(f"  R-squared (how much DC explains ED): {r_mat**2:.1%}")

# VIF-like analysis: regress ED on DC
from scipy.stats import linregress
slope, intercept, r_val, p_val, std_err = linregress(comps['DC'], comps['ED'])
print(f"\n  Regression: ED = {slope:.3f} × DC + {intercept:.1f}")
print(f"  R² = {r_val**2:.3f}")
print(f"  For every 10-point DC increase, ED probability increases by {slope*10:.1f}%")

# What % of Rd 1-2 are ED vs Rd 5-7?
for rds, label in [([1, 2], 'Rd 1-2'), ([3, 4], 'Rd 3-4'), ([5, 6, 7], 'Rd 5-7')]:
    subset = eval_df[eval_df['round'].isin(rds)]
    ed_rate = subset['early_declare'].mean()
    print(f"  {label}: {ed_rate:.0%} early declares ({int(subset['early_declare'].sum())}/{len(subset)})")

print(f"\n  INTERPRETATION:")
print(f"  ED is {'strongly' if abs(r_mat) > 0.4 else 'moderately'} correlated with DC (r={r_mat:+.3f}).")
print(f"  {r_mat**2:.0%} of ED's variance is explained by DC alone.")
if abs(r_mat) > 0.4:
    print(f"  This means the partial correlation (controlling for DC) is likely removing")
    print(f"  most of ED's predictive value because it's already captured by DC.")
    print(f"  ED may genuinely matter, but we can't separate its effect from DC in this sample.")

# ============================================================================
# 8. ALTERNATIVE: What if we test ED only on mid-round TEs?
# ============================================================================

print(f"\n\n{'='*100}")
print("8. ED VALUE IN MID-ROUND TEs (Rd 2-4 only)")
print("These TEs have enough DC variance + meaningful ED variance")
print("=" * 100)

mid = eval_df[eval_df['round'].between(2, 4)].copy()
mid_ed = mid[mid['early_declare'] == 1]
mid_non = mid[mid['early_declare'] == 0]

print(f"\n  Mid-round TEs: {len(mid)} (Rd 2-4)")
print(f"  ED=1: {len(mid_ed)}, ED=0: {len(mid_non)}")

for label, group in [('ED=1', mid_ed), ('ED=0', mid_non)]:
    n = len(group)
    if n < 3:
        print(f"  {label}: N={n} (too small)")
        continue
    t12 = group['top12_10g'].sum()
    ppg = group['best_3yr_ppg_10g'].dropna()
    ppg_m = ppg.mean() if len(ppg) > 0 else float('nan')
    s10 = group['seasons_over_10ppg_10g'].mean()
    ppg_s = f"{ppg_m:.2f}" if not np.isnan(ppg_m) else "N/A"
    print(f"  {label}: N={n:>3}, Top12={int(t12)}/{n} ({t12/n:.0%}), Avg3yrPPG={ppg_s}, Szn10+={s10:.1f}")

# Partial correlation within mid-rounds
mid_dc = mid['pick'].apply(lambda p: max(0, min(100, 100 - 2.40 * (p ** 0.62 - 1))))
for outcome in ['top12_10g', 'best_3yr_ppg_10g', 'seasons_over_10ppg_10g']:
    valid = pd.DataFrame({'x': mid['early_declare']*100, 'y': mid[outcome], 'z': mid_dc}).dropna()
    if len(valid) < 15:
        print(f"  Partial r(ED, {outcome} | DC): N={len(valid)} too small")
        continue
    sx, ix, _, _, _ = linregress(valid['z'], valid['x'])
    rx = valid['x'] - (sx * valid['z'] + ix)
    sy, iy, _, _, _ = linregress(valid['z'], valid['y'])
    ry = valid['y'] - (sy * valid['z'] + iy)
    r, p = stats.pearsonr(rx, ry)
    sig = "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else ""))
    print(f"  Partial r(ED, {outcome} | DC): r={r:+.3f} (p={p:.4f}) {sig}")


print(f"\n{'='*100}")
print("INVESTIGATION COMPLETE")
print("=" * 100)
