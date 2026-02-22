"""
Enhanced Breakout Score — Rushing Bonus Folded Into Production
==============================================================
Instead of rushing as a standalone 6th component, fold it into the
breakout score as a small additive/multiplicative bonus.

Versions tested:
  A: breakout + 3 points if rush_yards >= 20
  B: breakout + 5 points if rush_yards >= 20
  C: breakout × 1.05 if rush_yards >= 20
  D: breakout + min(rush_yards / 50, 5) — sliding scale

Then: 4-component optimization with DC / Enhanced Breakout / Teammate / Early Declare
(RAS dropped since optimizer consistently gave it 0%)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def dc_score(pick):
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))


def wr_breakout_score(breakout_age, dominator_pct):
    """Plain breakout score (no rushing bonus)."""
    if pd.isna(breakout_age) or breakout_age >= 99:
        if pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        else:
            return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base_score = age_tiers.get(int(breakout_age), 20)
    if pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
        bonus = max(0, bonus)
    else:
        bonus = 0
    return min(base_score + bonus, 99.9)


def partial_correlation(x, y, covariates):
    """Partial correlation between x and y, controlling for covariates."""
    data = pd.DataFrame({'x': x, 'y': y})
    for i, cov in enumerate(covariates):
        data[f'cov_{i}'] = cov
    data = data.dropna()
    n = len(data)
    if n < 10:
        return np.nan, np.nan, n
    cov_cols = [c for c in data.columns if c.startswith('cov_')]
    if not cov_cols:
        r, p = stats.pearsonr(data['x'], data['y'])
        return r, p, n
    X = np.column_stack([np.ones(n), data[cov_cols].values])
    try:
        beta_x = np.linalg.lstsq(X, data['x'].values, rcond=None)[0]
        resid_x = data['x'].values - X @ beta_x
        beta_y = np.linalg.lstsq(X, data['y'].values, rcond=None)[0]
        resid_y = data['y'].values - X @ beta_y
        r, p = stats.pearsonr(resid_x, resid_y)
        return r, p, n
    except:
        return np.nan, np.nan, n


# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 100)
print("ENHANCED BREAKOUT SCORE — RUSHING BONUS ANALYSIS")
print("=" * 100)

wr = pd.read_csv('data/wr_backtest_all_components.csv')
tm = pd.read_csv('data/wr_teammate_scores.csv')
gp = pd.read_csv('data/wr_games_played.csv')

print(f"\nLoaded: {len(wr)} WRs")

# Merge teammate scores
tm_lookup = dict(zip(zip(tm['player_name'], tm['draft_year']), tm['avg_teammate_dc']))
wr['teammate_dc'] = wr.apply(lambda x: tm_lookup.get((x['player_name'], x['draft_year']), 0), axis=1)


# ============================================================================
# STEP 1: COMPUTE PLAIN BREAKOUT SCORES
# ============================================================================

print(f"\n{'=' * 100}")
print("STEP 1: PLAIN BREAKOUT SCORES (existing methodology)")
print("=" * 100)

wr['s_breakout_plain'] = wr.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)

print(f"\n  Plain breakout score stats:")
print(f"    Min:    {wr['s_breakout_plain'].min():.1f}")
print(f"    Mean:   {wr['s_breakout_plain'].mean():.1f}")
print(f"    Median: {wr['s_breakout_plain'].median():.1f}")
print(f"    Max:    {wr['s_breakout_plain'].max():.1f}")
print(f"    NaN:    {wr['s_breakout_plain'].isna().sum()}")

# How many qualify for the rushing bonus (20+ yards)?
n_rush_qual = (wr['rush_yards'] >= 20).sum()
n_rush_no = (wr['rush_yards'] < 20).sum()
print(f"\n  Rushing qualifier (20+ yards): {n_rush_qual} WRs ({n_rush_qual/len(wr)*100:.1f}%)")
print(f"  No rushing bonus:              {n_rush_no} WRs ({n_rush_no/len(wr)*100:.1f}%)")


# ============================================================================
# STEP 2: COMPUTE ALL 4 ENHANCED BREAKOUT VERSIONS
# ============================================================================

print(f"\n{'=' * 100}")
print("STEP 2: ENHANCED BREAKOUT SCORES (4 versions)")
print("=" * 100)

rush_flag = (wr['rush_yards'] >= 20).astype(int)

# Version A: +3 points if 20+ yards
wr['s_bo_A'] = wr['s_breakout_plain'] + (rush_flag * 3)
wr['s_bo_A'] = wr['s_bo_A'].clip(0, 99.9)

# Version B: +5 points if 20+ yards
wr['s_bo_B'] = wr['s_breakout_plain'] + (rush_flag * 5)
wr['s_bo_B'] = wr['s_bo_B'].clip(0, 99.9)

# Version C: ×1.05 if 20+ yards
wr['s_bo_C'] = np.where(rush_flag == 1,
                         wr['s_breakout_plain'] * 1.05,
                         wr['s_breakout_plain'])
wr['s_bo_C'] = wr['s_bo_C'].clip(0, 99.9)

# Version D: +min(rush_yards/50, 5) — sliding scale for all WRs
wr['s_bo_D'] = wr['s_breakout_plain'] + np.minimum(wr['rush_yards'].clip(lower=0) / 50, 5)
wr['s_bo_D'] = wr['s_bo_D'].clip(0, 99.9)

bo_versions = {
    'Plain':     's_breakout_plain',
    'A (+3 if 20+ yds)': 's_bo_A',
    'B (+5 if 20+ yds)': 's_bo_B',
    'C (×1.05 if 20+ yds)': 's_bo_C',
    'D (+yds/50, cap 5)': 's_bo_D',
}

print(f"\n{'Version':<25} {'Min':>7} {'Mean':>7} {'Median':>7} {'Max':>7} {'Unique':>7}")
print("-" * 65)
for label, col in bo_versions.items():
    print(f"  {label:<23} {wr[col].min():>7.1f} {wr[col].mean():>7.1f} "
          f"{wr[col].median():>7.1f} {wr[col].max():>7.1f} {wr[col].nunique():>7}")


# Show a few example players to illustrate
print(f"\n\nExample players — how the bonus changes scores:")
print(f"{'Name':<25} {'Pick':>4} {'BO_age':>6} {'Plain':>6} {'A(+3)':>6} {'B(+5)':>6} {'C(×1.05)':>8} {'D(slide)':>8} {'RshYd':>6}")
print("-" * 95)

examples = [
    'CeeDee Lamb', 'Tyreek Hill', 'Garrett Wilson', 'Stefon Diggs',
    'Curtis Samuel', 'Lynn Bowden Jr.', 'Cooper Kupp', 'Ja\'Marr Chase',
    'Justin Jefferson', 'Deebo Samuel', 'Puka Nacua', 'Amon-Ra St. Brown',
    'Amari Cooper', 'DK Metcalf', 'Kadarius Toney', 'Jalen Reagor'
]
for name in examples:
    row = wr[wr['player_name'] == name]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    bo_age = f"{r['breakout_age']:.0f}" if r['breakout_age'] < 99 else "Never"
    print(f"  {name:<23} {r['pick']:>4} {bo_age:>6} {r['s_breakout_plain']:>6.1f} "
          f"{r['s_bo_A']:>6.1f} {r['s_bo_B']:>6.1f} {r['s_bo_C']:>8.1f} {r['s_bo_D']:>8.1f} "
          f"{r['rush_yards']:>6.0f}")


# ============================================================================
# STEP 3: PARTIAL CORRELATIONS — EACH VERSION VS ALL 4 OUTCOMES
# ============================================================================

print(f"\n\n{'=' * 100}")
print("STEP 3: PARTIAL CORRELATIONS (controlling for DC)")
print("=" * 100)
print("Does the rushing bonus improve the breakout component's signal?")

wr['s_dc'] = wr['pick'].apply(dc_score)

# Filter to 2015-2024 with outcomes
wr_eval = wr[wr['hit24'].notna() & wr['draft_year'].between(2015, 2024)].copy()
n_eval = len(wr_eval)
print(f"\nEvaluation sample: {n_eval} WRs (2015-2024 with hit24)")

outcomes = ['hit24', 'hit12', 'best_ppr', 'best_rank']

print(f"\n{'Version':<25}", end="")
for out in outcomes:
    print(f" {'r('+out+')':>12} {'p':>8}", end="")
print(f"  {'N':>5}")
print("-" * 110)

partial_results = {}
for label, col in bo_versions.items():
    row_str = f"  {label:<23}"
    results_for_version = {}
    for out in outcomes:
        r, p, n = partial_correlation(wr_eval[col], wr_eval[out], [wr_eval['s_dc']])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        row_str += f" {r:>+.4f}{sig:<4} {p:>8.4f}"
        results_for_version[out] = (r, p, n)
    row_str += f"  {n:>5}"
    print(row_str)
    partial_results[label] = results_for_version

# Compute deltas vs plain
print(f"\n\nDELTA vs PLAIN (positive = improved signal):")
print(f"{'Version':<25}", end="")
for out in outcomes:
    print(f" {'Δr('+out+')':>12}", end="")
print()
print("-" * 80)

plain_rs = {out: partial_results['Plain'][out][0] for out in outcomes}
for label in list(bo_versions.keys())[1:]:  # Skip Plain
    row_str = f"  {label:<23}"
    for out in outcomes:
        delta = partial_results[label][out][0] - plain_rs[out]
        row_str += f" {delta:>+12.4f}"
    print(row_str)


# ============================================================================
# RAW CORRELATIONS (breakout score vs outcomes, no controls)
# ============================================================================

print(f"\n\n{'=' * 100}")
print("RAW CORRELATIONS (no controls — just breakout version vs outcome)")
print("=" * 100)

print(f"\n{'Version':<25}", end="")
for out in outcomes:
    print(f" {'r('+out+')':>12} {'p':>8}", end="")
print()
print("-" * 110)

for label, col in bo_versions.items():
    row_str = f"  {label:<23}"
    for out in outcomes:
        valid = wr_eval[[col, out]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid[col], valid[out])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row_str += f" {r:>+.4f}{sig:<4} {p:>8.4f}"
        else:
            row_str += f" {'N/A':>12} {'':>8}"
    print(row_str)


# ============================================================================
# STEP 4: FIND THE BEST ENHANCED BREAKOUT VERSION
# ============================================================================

print(f"\n\n{'=' * 100}")
print("STEP 4: DETERMINE BEST ENHANCED BREAKOUT VERSION")
print("=" * 100)

# Score each version by average partial r across hit24 and best_ppr
print(f"\n{'Version':<25} {'partial_r(hit24)':>16} {'partial_r(ppr)':>16} {'avg':>8}")
print("-" * 70)

version_scores = {}
for label, col in bo_versions.items():
    r_hit = partial_results[label]['hit24'][0]
    r_ppr = partial_results[label]['best_ppr'][0]
    avg = (abs(r_hit) + abs(r_ppr)) / 2 if r_ppr > 0 else abs(r_hit)
    # Use signed average for interpretation (both should be positive for good signal)
    signed_avg = (r_hit + r_ppr) / 2
    version_scores[label] = signed_avg
    marker = ""
    print(f"  {label:<23} {r_hit:>+16.4f} {r_ppr:>+16.4f} {signed_avg:>+8.4f}{marker}")

best_version_label = max(version_scores, key=version_scores.get)
best_version_col = bo_versions[best_version_label]
print(f"\n  Best enhanced breakout version: {best_version_label}")
print(f"  (avg partial r = {version_scores[best_version_label]:+.4f} vs Plain = {version_scores['Plain']:+.4f})")

improvement = version_scores[best_version_label] - version_scores['Plain']
print(f"  Improvement over Plain: {improvement:+.4f}")

if improvement > 0.005:
    print(f"  --> Enhanced version IMPROVES signal (by {improvement:+.4f})")
elif improvement > -0.005:
    print(f"  --> Enhanced version is ROUGHLY EQUIVALENT to Plain")
else:
    print(f"  --> Enhanced version is WORSE — rushing bonus hurts the signal")


# ============================================================================
# STEP 5: 4-COMPONENT OPTIMIZATION
# ============================================================================
# Components: DC, Enhanced Breakout, Teammate, Early Declare
# (RAS dropped since optimizer consistently gave it 0%)

print(f"\n\n{'=' * 100}")
print("STEP 5: 4-COMPONENT WEIGHT OPTIMIZATION")
print("=" * 100)
print("Components: DC / Enhanced Breakout / Teammate / Early Declare")
print("(RAS dropped — optimizer consistently gave it 0%)")

# Compute all component scores for eval sample
wr_eval['s_dc'] = wr_eval['pick'].apply(dc_score)
wr_eval['s_teammate'] = wr_eval['teammate_dc'].clip(0, 100)
wr_eval['s_declare'] = wr_eval['early_declare'] * 100

# We'll test two models side by side:
# Model 1: DC + Plain Breakout + Teammate + Early Declare
# Model 2: DC + Best Enhanced Breakout + Teammate + Early Declare

for model_name, bo_col, bo_label in [
    ("PLAIN BREAKOUT", 's_breakout_plain', 'Plain Breakout'),
    (f"ENHANCED BREAKOUT ({best_version_label})", best_version_col, f'Enhanced ({best_version_label})'),
]:
    print(f"\n\n{'─' * 80}")
    print(f"MODEL: {model_name}")
    print(f"{'─' * 80}")

    score_cols = ['s_dc', bo_col, 's_teammate', 's_declare']
    labels = ['Draft Capital', bo_label, 'Teammate Score', 'Early Declare']

    # Check for nulls
    total_na = sum(wr_eval[c].isna().sum() for c in score_cols)
    if total_na > 0:
        print(f"  WARNING: {total_na} NaN values — filling with defaults")
        for c in score_cols:
            wr_eval[c] = wr_eval[c].fillna(0)

    # Component stats
    print(f"\n  {'Component':<25} {'Min':>7} {'Mean':>7} {'Median':>7} {'Max':>7}")
    print(f"  {'-'*55}")
    for col, lab in zip(score_cols, labels):
        print(f"    {lab:<23} {wr_eval[col].min():>7.1f} {wr_eval[col].mean():>7.1f} "
              f"{wr_eval[col].median():>7.1f} {wr_eval[col].max():>7.1f}")

    # GRID SEARCH
    configs_4 = [
        # Current-like baselines
        (65, 20, 0, 15),   # won't sum to 100 — need to adjust
        (65, 20, 10, 5),
        (65, 20, 8, 7),
        (65, 20, 5, 10),
        (65, 15, 10, 10),
        (65, 15, 12, 8),
        (65, 15, 8, 12),
        (65, 15, 15, 5),
        (65, 15, 5, 15),
        (65, 10, 15, 10),
        (65, 10, 10, 15),

        # DC heavy
        (70, 15, 8, 7),
        (70, 15, 5, 10),
        (70, 15, 10, 5),
        (70, 10, 10, 10),
        (70, 10, 12, 8),
        (70, 10, 8, 12),
        (75, 10, 8, 7),
        (75, 10, 5, 10),
        (75, 10, 10, 5),
        (75, 15, 5, 5),
        (80, 10, 5, 5),
        (80, 5, 8, 7),
        (80, 10, 3, 7),
        (80, 5, 5, 10),

        # Breakout heavy
        (55, 25, 10, 10),
        (55, 25, 12, 8),
        (55, 25, 8, 12),
        (55, 25, 15, 5),
        (55, 25, 5, 15),
        (60, 20, 10, 10),
        (60, 20, 12, 8),
        (60, 20, 8, 12),
        (60, 20, 15, 5),
        (60, 20, 5, 15),
        (60, 25, 8, 7),
        (60, 25, 10, 5),
        (60, 25, 5, 10),

        # Teammate emphasis
        (60, 15, 15, 10),
        (60, 15, 20, 5),
        (55, 20, 15, 10),
        (55, 15, 20, 10),
        (65, 10, 20, 5),
    ]

    # Deduplicate and filter to sum=100
    seen = set()
    unique_configs_4 = []
    for c in configs_4:
        if c not in seen and sum(c) == 100:
            seen.add(c)
            unique_configs_4.append(c)

    print(f"\n  Grid search: {len(unique_configs_4)} configurations")
    print(f"\n  {'#':>3} {'Config':>22}  {'hit24':>8} {'ppr':>8}  {'avg':>8}")
    print(f"  {'-'*55}")

    results_4 = []
    for i, (dc_w, bo_w, tm_w, ed_w) in enumerate(unique_configs_4):
        slap = (
            (dc_w / 100) * wr_eval['s_dc'] +
            (bo_w / 100) * wr_eval[bo_col] +
            (tm_w / 100) * wr_eval['s_teammate'] +
            (ed_w / 100) * wr_eval['s_declare']
        )

        label = f"{dc_w}/{bo_w}/{tm_w}/{ed_w}"

        cors = []
        r1, p1 = stats.pearsonr(slap, wr_eval['hit24'])
        cors.append(r1)

        ppr_valid = wr_eval['best_ppr'].notna()
        if ppr_valid.sum() >= 10:
            r2, _ = stats.pearsonr(slap[ppr_valid], wr_eval.loc[ppr_valid, 'best_ppr'])
            cors.append(r2)
        else:
            r2 = np.nan

        avg_r = np.mean([c for c in cors if not np.isnan(c)])
        sig1 = "*" if p1 < 0.05 else ""
        r2_str = f"{r2:>+.3f}" if not np.isnan(r2) else "N/A"

        print(f"  {i+1:>3}  {label:>20}  {r1:>+.3f}{sig1:<2} {r2_str:>7}   {avg_r:>+.3f}")

        results_4.append({
            'config': label, 'dc': dc_w, 'bo': bo_w, 'tm': tm_w, 'ed': ed_w,
            'r_hit24': r1, 'p_hit24': p1, 'r_ppr': r2, 'avg_r': avg_r
        })

    # Sort by avg correlation
    res_df = pd.DataFrame(results_4).sort_values('avg_r', ascending=False)

    print(f"\n  TOP 5 (by avg r):")
    print(f"  {'Rank':>4} {'Config':>22}  {'hit24':>8} {'ppr':>8}  {'avg':>8}")
    print(f"  {'-'*55}")
    for i, (_, row) in enumerate(res_df.head(5).iterrows()):
        ppr_str = f"{row['r_ppr']:>+.3f}" if not np.isnan(row['r_ppr']) else "N/A"
        print(f"  {i+1:>4}  {row['config']:>20}  {row['r_hit24']:>+.3f}   {ppr_str:>7}   {row['avg_r']:>+.3f}")

    best_grid = res_df.iloc[0]
    print(f"\n  Best grid: {best_grid['config']} → avg r = {best_grid['avg_r']:+.4f}")

    # SCIPY OPTIMIZER
    print(f"\n  Scipy optimizer (DC >= 40%, sum = 100%):")

    eval_data = wr_eval[score_cols + ['hit24', 'best_ppr']].dropna(subset=['hit24']).copy()
    X = eval_data[score_cols].values
    y_hit24 = eval_data['hit24'].values
    has_ppr = eval_data['best_ppr'].notna()
    y_ppr = eval_data.loc[has_ppr, 'best_ppr'].values
    X_ppr = eval_data.loc[has_ppr, score_cols].values

    def neg_avg_corr(weights, X_h, y_h, X_p, y_p):
        s1 = X_h @ weights
        r1 = np.corrcoef(s1, y_h)[0, 1] if np.std(s1) > 1e-10 else 0
        s2 = X_p @ weights
        r2 = np.corrcoef(s2, y_p)[0, 1] if np.std(s2) > 1e-10 else 0
        return -(r1 + r2) / 2

    constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]
    bounds_4 = [
        (0.40, 0.85),  # DC
        (0.05, 0.35),  # Breakout
        (0.00, 0.25),  # Teammate
        (0.00, 0.20),  # Early Declare
    ]

    # Multi-start optimization
    best_r_opt = -999
    best_w_opt = None
    starts = [
        [0.65, 0.20, 0.10, 0.05],
        [0.70, 0.15, 0.08, 0.07],
        [0.75, 0.10, 0.08, 0.07],
        [0.60, 0.25, 0.08, 0.07],
        [0.80, 0.10, 0.05, 0.05],
    ]

    for s_idx, x0 in enumerate(starts):
        res = minimize(neg_avg_corr, x0, args=(X, y_hit24, X_ppr, y_ppr),
                       method='SLSQP', bounds=bounds_4, constraints=constraints,
                       options={'maxiter': 1000})
        r_opt = -res.fun
        w = res.x
        tag = " <-- BEST" if r_opt > best_r_opt else ""
        if r_opt > best_r_opt:
            best_r_opt = r_opt
            best_w_opt = w
        print(f"    Start {s_idx+1}: DC={w[0]:.0%}/BO={w[1]:.0%}/TM={w[2]:.0%}/ED={w[3]:.0%} "
              f"→ avg r = {r_opt:+.4f}{tag}")

    print(f"\n  SCIPY BEST: DC={best_w_opt[0]:.0%} / BO={best_w_opt[1]:.0%} / "
          f"TM={best_w_opt[2]:.0%} / ED={best_w_opt[3]:.0%}")
    print(f"  avg r = {best_r_opt:+.4f}")

    # Also compute hit24-only and ppr-only optimized
    def neg_corr(weights, X, y):
        s = X @ weights
        if np.std(s) < 1e-10:
            return 0
        return -np.corrcoef(s, y)[0, 1]

    res_h = minimize(neg_corr, starts[0], args=(X, y_hit24),
                     method='SLSQP', bounds=bounds_4, constraints=constraints,
                     options={'maxiter': 1000})
    print(f"\n  Hit24-only optimized: DC={res_h.x[0]:.0%}/BO={res_h.x[1]:.0%}/"
          f"TM={res_h.x[2]:.0%}/ED={res_h.x[3]:.0%} → r = {-res_h.fun:+.4f}")

    res_p = minimize(neg_corr, starts[0], args=(X_ppr, y_ppr),
                     method='SLSQP', bounds=bounds_4, constraints=constraints,
                     options={'maxiter': 1000})
    print(f"  PPR-only optimized:   DC={res_p.x[0]:.0%}/BO={res_p.x[1]:.0%}/"
          f"TM={res_p.x[2]:.0%}/ED={res_p.x[3]:.0%} → r = {-res_p.fun:+.4f}")

    # Partial correlations for the 4 components
    print(f"\n  Partial correlations (unique contribution, controlling for other 3):")
    print(f"  {'Component':<23} {'hit24':>10} {'best_ppr':>10}")
    print(f"  {'-'*45}")
    for col_i, lab in zip(score_cols, labels):
        others = [c for c in score_cols if c != col_i]
        r_h, p_h, _ = partial_correlation(wr_eval[col_i], wr_eval['hit24'],
                                           [wr_eval[c] for c in others])
        r_p, p_p, _ = partial_correlation(wr_eval[col_i], wr_eval['best_ppr'],
                                           [wr_eval[c] for c in others])
        sig_h = "***" if p_h < 0.001 else "**" if p_h < 0.01 else "*" if p_h < 0.05 else ""
        sig_p = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else ""
        print(f"    {lab:<21} {r_h:>+.3f}{sig_h:<4}    {r_p:>+.3f}{sig_p:<4}")


# ============================================================================
# STEP 6: ALSO TEST 5-COMPONENT (add RAS back) WITH ENHANCED BREAKOUT
# ============================================================================

print(f"\n\n{'=' * 100}")
print("BONUS: 5-COMPONENT MODEL (DC / Enhanced Breakout / Teammate / RAS / Early Declare)")
print("=" * 100)
print("Testing whether RAS adds value when combined with enhanced breakout")

wr_eval['s_ras'] = (wr_eval['RAS'] * 10).clip(0, 100)

score_cols_5 = ['s_dc', best_version_col, 's_teammate', 's_ras', 's_declare']
labels_5 = ['Draft Capital', f'Enhanced ({best_version_label})', 'Teammate', 'RAS', 'Early Declare']

eval_data_5 = wr_eval[score_cols_5 + ['hit24', 'best_ppr']].dropna(subset=['hit24']).copy()
X5 = eval_data_5[score_cols_5].values
y5_hit = eval_data_5['hit24'].values
has_ppr_5 = eval_data_5['best_ppr'].notna()
y5_ppr = eval_data_5.loc[has_ppr_5, 'best_ppr'].values
X5_ppr = eval_data_5.loc[has_ppr_5, score_cols_5].values

bounds_5 = [
    (0.40, 0.85),  # DC
    (0.05, 0.35),  # Breakout
    (0.00, 0.20),  # Teammate
    (0.00, 0.15),  # RAS
    (0.00, 0.15),  # Early Declare
]

def neg_avg_corr_5(weights, X_h, y_h, X_p, y_p):
    s1 = X_h @ weights
    r1 = np.corrcoef(s1, y_h)[0, 1] if np.std(s1) > 1e-10 else 0
    s2 = X_p @ weights
    r2 = np.corrcoef(s2, y_p)[0, 1] if np.std(s2) > 1e-10 else 0
    return -(r1 + r2) / 2

constraints_5 = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]

starts_5 = [
    [0.65, 0.15, 0.08, 0.05, 0.07],
    [0.70, 0.12, 0.06, 0.05, 0.07],
    [0.75, 0.10, 0.05, 0.05, 0.05],
    [0.60, 0.20, 0.08, 0.05, 0.07],
    [0.80, 0.08, 0.04, 0.03, 0.05],
]

best_r_5 = -999
best_w_5 = None
for s_idx, x0 in enumerate(starts_5):
    res = minimize(neg_avg_corr_5, x0, args=(X5, y5_hit, X5_ppr, y5_ppr),
                   method='SLSQP', bounds=bounds_5, constraints=constraints_5,
                   options={'maxiter': 1000})
    r5 = -res.fun
    w5 = res.x
    tag = " <-- BEST" if r5 > best_r_5 else ""
    if r5 > best_r_5:
        best_r_5 = r5
        best_w_5 = w5
    print(f"  Start {s_idx+1}: DC={w5[0]:.0%}/BO={w5[1]:.0%}/TM={w5[2]:.0%}/"
          f"RAS={w5[3]:.0%}/ED={w5[4]:.0%} → avg r = {r5:+.4f}{tag}")

print(f"\n  5-COMPONENT BEST: DC={best_w_5[0]:.0%} / BO={best_w_5[1]:.0%} / "
      f"TM={best_w_5[2]:.0%} / RAS={best_w_5[3]:.0%} / ED={best_w_5[4]:.0%}")
print(f"  avg r = {best_r_5:+.4f}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n\n{'=' * 100}")
print("FINAL COMPARISON — ALL MODEL VARIANTS")
print("=" * 100)

print(f"""
{'Model':<60} {'avg r':>8}
{'─'*70}""")

# Compute current 3-component baseline
slap_current = (0.65 * wr_eval['s_dc'] + 0.20 * wr_eval['s_breakout_plain'] +
                0.15 * wr_eval['s_ras'])
r_cur_h, _ = stats.pearsonr(slap_current, wr_eval['hit24'])
ppr_v = wr_eval['best_ppr'].notna()
r_cur_p, _ = stats.pearsonr(slap_current[ppr_v], wr_eval.loc[ppr_v, 'best_ppr'])
r_cur_avg = (r_cur_h + r_cur_p) / 2
print(f"  {'Current 3-comp (65/20/0/15 DC/BO/TM/RAS)':<58} {r_cur_avg:>+.4f}")

# 4-component with plain breakout (compute from the saved best)
# Need to re-run for plain
score_cols_plain = ['s_dc', 's_breakout_plain', 's_teammate', 's_declare']
eval_plain = wr_eval[score_cols_plain + ['hit24', 'best_ppr']].dropna(subset=['hit24']).copy()
X_plain = eval_plain[score_cols_plain].values
y_plain_hit = eval_plain['hit24'].values
has_ppr_pl = eval_plain['best_ppr'].notna()
y_plain_ppr = eval_plain.loc[has_ppr_pl, 'best_ppr'].values
X_plain_ppr = eval_plain.loc[has_ppr_pl, score_cols_plain].values

res_plain_opt = minimize(neg_avg_corr, [0.65, 0.20, 0.10, 0.05],
                         args=(X_plain, y_plain_hit, X_plain_ppr, y_plain_ppr),
                         method='SLSQP', bounds=bounds_4, constraints=constraints,
                         options={'maxiter': 1000})
r_plain_opt = -res_plain_opt.fun
w_plain_opt = res_plain_opt.x

print(f"  {'4-comp PLAIN (DC/BO/TM/ED optimized)':<58} {r_plain_opt:>+.4f}"
      f"  ({w_plain_opt[0]:.0%}/{w_plain_opt[1]:.0%}/{w_plain_opt[2]:.0%}/{w_plain_opt[3]:.0%})")

print(f"  {'4-comp ENHANCED (DC/BO+rush/TM/ED optimized)':<58} {best_r_opt:>+.4f}"
      f"  ({best_w_opt[0]:.0%}/{best_w_opt[1]:.0%}/{best_w_opt[2]:.0%}/{best_w_opt[3]:.0%})")

print(f"  {'5-comp ENHANCED (DC/BO+rush/TM/RAS/ED optimized)':<58} {best_r_5:>+.4f}"
      f"  ({best_w_5[0]:.0%}/{best_w_5[1]:.0%}/{best_w_5[2]:.0%}/{best_w_5[3]:.0%}/{best_w_5[4]:.0%})")

# Delta
delta_enhanced_vs_plain = best_r_opt - r_plain_opt
delta_enhanced_vs_current = best_r_opt - r_cur_avg
delta_5comp = best_r_5 - best_r_opt

print(f"""
{'─'*70}
Key deltas:
  Enhanced 4-comp vs Plain 4-comp:  {delta_enhanced_vs_plain:+.4f}  {"(enhanced is better)" if delta_enhanced_vs_plain > 0 else "(plain is better)"}
  Enhanced 4-comp vs Current 3-comp: {delta_enhanced_vs_current:+.4f}  {"(new model is better)" if delta_enhanced_vs_current > 0 else "(current is better)"}
  5-comp vs 4-comp enhanced:         {delta_5comp:+.4f}  {"(adding RAS helps)" if delta_5comp > 0.002 else "(RAS adds minimal value)"}
""")
