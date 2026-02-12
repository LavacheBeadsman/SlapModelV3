"""
WR SLAP Model - 6-Component Weight Optimization
=================================================
Uses the clean all-components dataset (339 WRs, zero nulls).

Components:
1. Draft Capital (DC) → 0-100 from pick
2. Breakout Age → 0-100 from breakout_age + peak_dominator
3. Teammate Score → 0-100 from avg_teammate_dc
4. RAS (Athletic) → 0-100 from RAS * 10
5. Early Declare → 0 or 100 from early_declare
6. Rushing Production → 0-100 from rush_yards (normalized)

Outcome variables (2015-2024 classes only):
- hit24: Top-24 WR in first 2 seasons (binary)
- best_ppr: Best PPR season in first 3 years (continuous)
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
    """Breakout score. breakout_age=99 → never broke out sentinel."""
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
print("WR SLAP MODEL — 6-COMPONENT WEIGHT OPTIMIZATION")
print("=" * 100)

wr = pd.read_csv('data/wr_backtest_all_components.csv')
tm = pd.read_csv('data/wr_teammate_scores.csv')
gp = pd.read_csv('data/wr_games_played.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')

print(f"\nLoaded: {len(wr)} WRs from wr_backtest_all_components.csv")

# Merge teammate scores
tm_lookup = dict(zip(zip(tm['player_name'], tm['draft_year']), tm['avg_teammate_dc']))
wr['teammate_dc'] = wr.apply(lambda x: tm_lookup.get((x['player_name'], x['draft_year']), 0), axis=1)

# Merge games played
gp_lookup = dict(zip(zip(gp['player_name'], gp['draft_year']), gp['games_played']))
wr['games_played'] = wr.apply(lambda x: gp_lookup.get((x['player_name'], x['draft_year']), 11), axis=1)

# Merge outcomes
outcomes_wr = outcomes[outcomes['position'] == 'WR'][
    ['player_name', 'draft_year', 'best_ppr', 'best_rank']
].copy()
wr = wr.merge(outcomes_wr, on=['player_name', 'draft_year'], how='left',
              suffixes=('', '_outcome'))

# Use file's own hit24/best_ppr if available, otherwise from outcomes
if 'hit24_outcome' in wr.columns:
    wr['hit24'] = wr['hit24'].fillna(wr.get('hit24_outcome', np.nan))

n_outcomes = wr['hit24'].notna().sum()
n_ppr = wr['best_ppr'].notna().sum()
print(f"Outcomes: {n_outcomes} with hit24, {n_ppr} with best_ppr")


# ============================================================================
# COMPUTE ALL 6 COMPONENT SCORES
# ============================================================================

print(f"\n{'=' * 100}")
print("COMPUTING 6 COMPONENT SCORES (0-100 scale)")
print("=" * 100)

# 1. Draft Capital
wr['s_dc'] = wr['pick'].apply(dc_score)

# 2. Breakout Score
wr['s_breakout'] = wr.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)

# 3. Teammate Score (avg_teammate_dc is already ~0-100 scale)
wr['s_teammate'] = wr['teammate_dc'].clip(0, 100)

# 4. RAS Score (0-10 → 0-100)
wr['s_ras'] = (wr['RAS'] * 10).clip(0, 100)

# 5. Early Declare (binary → 0 or 100)
wr['s_declare'] = wr['early_declare'] * 100

# 6. Rushing Score (rush_yards / games, normalized)
wr['rush_ypg'] = np.where(wr['games_played'] > 0,
                           wr['rush_yards'] / wr['games_played'], 0)
p99 = wr['rush_ypg'].quantile(0.99)
wr['s_rushing'] = (wr['rush_ypg'] / p99 * 100).clip(0, 100)

score_cols = ['s_dc', 's_breakout', 's_teammate', 's_ras', 's_declare', 's_rushing']
score_labels = {
    's_dc': 'Draft Capital',
    's_breakout': 'Breakout Age',
    's_teammate': 'Teammate Score',
    's_ras': 'RAS (Athletic)',
    's_declare': 'Early Declare',
    's_rushing': 'Rushing Prod',
}

print(f"\n{'Component':<20} {'Nulls':>6} {'Min':>7} {'Mean':>7} {'Median':>7} {'Max':>7}")
print("-" * 60)
for col in score_cols:
    na = wr[col].isna().sum()
    print(f"  {score_labels[col]:<18} {na:>6} {wr[col].min():>7.1f} {wr[col].mean():>7.1f} "
          f"{wr[col].median():>7.1f} {wr[col].max():>7.1f}")

total_na = sum(wr[c].isna().sum() for c in score_cols)
print(f"\n  TOTAL NaN across all 6 scores: {total_na}")
assert total_na == 0, "FATAL: Still have NaN in score columns!"
print("  ALL 339 WRs HAVE COMPLETE SCORES ✓")


# ============================================================================
# RAW CORRELATIONS
# ============================================================================

print(f"\n{'=' * 100}")
print("RAW CORRELATIONS (each component vs outcomes)")
print("=" * 100)

outcome_cols = ['hit24', 'best_ppr']
print(f"\n{'Component':<20} {'hit24':>10} {'best_ppr':>10} {'N_hit24':>10} {'N_ppr':>10}")
print("-" * 65)

for col in score_cols:
    row = f"  {score_labels[col]:<18}"
    for out in outcome_cols:
        valid = wr[[col, out]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid[col], valid[out])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f" {r:>+.3f}{sig:<4}"
        else:
            row += f" {'N/A':>10}"
    n1 = wr[[col, 'hit24']].dropna().shape[0]
    n2 = wr[[col, 'best_ppr']].dropna().shape[0]
    row += f" {n1:>10} {n2:>10}"
    print(row)

print("\n  * p<0.05, ** p<0.01, *** p<0.001")


# ============================================================================
# PARTIAL CORRELATIONS (unique contribution of each)
# ============================================================================

print(f"\n{'=' * 100}")
print("PARTIAL CORRELATIONS (unique contribution controlling for other 5 components)")
print("=" * 100)

print(f"\n{'Component':<20} {'hit24':>10} {'best_ppr':>10} {'N':>6}")
print("-" * 50)

for col in score_cols:
    others = [c for c in score_cols if c != col]
    row = f"  {score_labels[col]:<18}"
    for out in outcome_cols:
        covs = [wr[c] for c in others]
        r, p, n = partial_correlation(wr[col], wr[out], covs)
        if not np.isnan(r):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f" {r:>+.3f}{sig:<4}"
        else:
            row += f" {'N/A':>10}"
    n_full = wr[score_cols + [out]].dropna().shape[0]
    row += f" {n_full:>5}"
    print(row)

n_full = wr[score_cols + ['hit24']].dropna().shape[0]
print(f"\n  Full 6-component sample: {n_full} WRs")
print("\n  Key: Positive partial r = this component adds UNIQUE predictive value")
print("       p < 0.05 (*) = statistically significant unique contribution")


# ============================================================================
# MULTICOLLINEARITY CHECK
# ============================================================================

print(f"\n{'=' * 100}")
print("MULTICOLLINEARITY CHECK")
print("=" * 100)
print("Correlation between components (>0.50 = overlap concern, >0.70 = redundant)")

print(f"\n{'':>18}", end="")
for c in score_cols:
    print(f" {score_labels[c]:>13}", end="")
print()
print("-" * 100)

for c1 in score_cols:
    row = f"{score_labels[c1]:<18}"
    for c2 in score_cols:
        if c1 == c2:
            row += f"  {'1.000':>11}"
        else:
            r, p = stats.pearsonr(wr[c1], wr[c2])
            sig = "*" if p < 0.05 else ""
            row += f"  {r:>+.3f}{sig:<7}"
    print(row)


# ============================================================================
# WEIGHT OPTIMIZATION: GRID SEARCH
# ============================================================================

print(f"\n{'=' * 100}")
print("WEIGHT OPTIMIZATION — GRID SEARCH (339 WRs)")
print("=" * 100)
print("Testing weight configurations: DC/Breakout/Teammate/RAS/Declare/Rushing")
print("Constraint: weights sum to 100")

configs = [
    # Current 3-component baseline
    (65, 20, 0, 15, 0, 0),

    # Light 6-component variations
    (60, 18, 5, 7, 5, 5),
    (60, 20, 5, 5, 5, 5),
    (60, 15, 5, 10, 5, 5),
    (60, 15, 8, 7, 5, 5),
    (60, 15, 10, 5, 5, 5),
    (60, 15, 5, 5, 10, 5),
    (60, 15, 5, 5, 5, 10),
    (60, 20, 3, 7, 5, 5),
    (60, 20, 5, 7, 3, 5),
    (60, 20, 5, 5, 3, 7),
    (60, 20, 7, 3, 5, 5),
    (60, 20, 5, 3, 7, 5),

    # DC-heavy
    (65, 15, 5, 5, 5, 5),
    (65, 15, 5, 7, 5, 3),
    (65, 15, 5, 5, 7, 3),
    (65, 15, 3, 7, 5, 5),
    (65, 15, 7, 5, 5, 3),
    (65, 10, 5, 10, 5, 5),
    (65, 10, 10, 5, 5, 5),

    # Higher breakout
    (55, 25, 5, 5, 5, 5),
    (55, 20, 7, 8, 5, 5),
    (55, 20, 5, 8, 7, 5),
    (55, 20, 10, 5, 5, 5),
    (55, 20, 5, 10, 5, 5),
    (55, 20, 5, 5, 10, 5),

    # Lower DC
    (50, 25, 7, 7, 6, 5),
    (50, 20, 10, 10, 5, 5),
    (50, 20, 10, 5, 10, 5),
    (50, 20, 10, 5, 5, 10),
]

# Deduplicate
seen = set()
unique_configs = []
for c in configs:
    if c not in seen and sum(c) == 100:
        seen.add(c)
        unique_configs.append(c)
configs = unique_configs

print(f"\nTesting {len(configs)} configurations\n")

# Filter to WRs with outcome data
wr_eval = wr[wr['hit24'].notna()].copy()
print(f"Evaluation sample: {len(wr_eval)} WRs with hit24 data")
n_ppr = wr_eval['best_ppr'].notna().sum()
print(f"                   {n_ppr} also have best_ppr data")

# Header
print(f"\n{'#':>3} {'Config':>33}  {'hit24':>8} {'ppr':>8}  {'avg':>8}")
print("-" * 70)

results = []
for i, (dc_w, bo_w, tm_w, ras_w, ed_w, rush_w) in enumerate(configs):
    slap = (
        (dc_w / 100) * wr_eval['s_dc'] +
        (bo_w / 100) * wr_eval['s_breakout'] +
        (tm_w / 100) * wr_eval['s_teammate'] +
        (ras_w / 100) * wr_eval['s_ras'] +
        (ed_w / 100) * wr_eval['s_declare'] +
        (rush_w / 100) * wr_eval['s_rushing']
    )

    label = f"{dc_w}/{bo_w}/{tm_w}/{ras_w}/{ed_w}/{rush_w}"

    cors = []
    r1, p1 = stats.pearsonr(slap, wr_eval['hit24'])
    cors.append(r1)

    ppr_valid = wr_eval['best_ppr'].notna()
    if ppr_valid.sum() >= 10:
        r2, p2 = stats.pearsonr(slap[ppr_valid], wr_eval.loc[ppr_valid, 'best_ppr'])
        cors.append(r2)
    else:
        r2 = np.nan

    avg_r = np.mean([c for c in cors if not np.isnan(c)])

    sig1 = "*" if p1 < 0.05 else ""
    r2_str = f"{r2:>+.3f}" if not np.isnan(r2) else "N/A"

    marker = " <-- CURRENT" if (dc_w == 65 and bo_w == 20 and tm_w == 0 and ras_w == 15 and ed_w == 0 and rush_w == 0) else ""
    print(f"{i+1:>3}  {label:>31}  {r1:>+.3f}{sig1:<2} {r2_str:>7}   {avg_r:>+.3f}{marker}")

    results.append({
        'config': label, 'dc': dc_w, 'bo': bo_w, 'tm': tm_w,
        'ras': ras_w, 'ed': ed_w, 'rush': rush_w,
        'r_hit24': r1, 'p_hit24': p1, 'r_ppr': r2, 'avg_r': avg_r
    })

# Sort by avg correlation
results_df = pd.DataFrame(results).sort_values('avg_r', ascending=False)

print(f"\n\n{'=' * 100}")
print("TOP 10 CONFIGURATIONS (by average correlation)")
print("=" * 100)
print(f"\n{'Rank':>4} {'Config':>33}  {'hit24':>8} {'ppr':>8}  {'avg':>8}")
print("-" * 70)

for i, (_, row) in enumerate(results_df.head(10).iterrows()):
    marker = " <-- CURRENT" if row['config'] == "65/20/0/15/0/0" else ""
    ppr_str = f"{row['r_ppr']:>+.3f}" if not np.isnan(row['r_ppr']) else "N/A"
    print(f"{i+1:>4}  {row['config']:>31}  {row['r_hit24']:>+.3f}   {ppr_str:>7}   {row['avg_r']:>+.3f}{marker}")

# Show current model comparison
print(f"\n\nCurrent model (65/20/0/15/0/0):")
current = results_df[results_df['config'] == "65/20/0/15/0/0"]
if len(current) > 0:
    c = current.iloc[0]
    print(f"  hit24 r = {c['r_hit24']:+.3f} (p={c['p_hit24']:.4f})")
    print(f"  ppr r   = {c['r_ppr']:+.3f}")
    print(f"  avg r   = {c['avg_r']:+.3f}")

best = results_df.iloc[0]
print(f"\nBest configuration ({best['config']}):")
print(f"  hit24 r = {best['r_hit24']:+.3f} (p={best['p_hit24']:.4f})")
print(f"  ppr r   = {best['r_ppr']:+.3f}")
print(f"  avg r   = {best['avg_r']:+.3f}")

# Improvement
if len(current) > 0:
    delta_hit = best['r_hit24'] - current.iloc[0]['r_hit24']
    delta_avg = best['avg_r'] - current.iloc[0]['avg_r']
    print(f"\n  Delta vs current: hit24 {delta_hit:+.3f}, avg {delta_avg:+.3f}")
    if delta_avg > 0.01:
        print(f"  --> 6-component model IMPROVES on current 3-component model")
    elif delta_avg > -0.01:
        print(f"  --> 6-component model is ROUGHLY EQUIVALENT to current model")
    else:
        print(f"  --> 6-component model is WORSE — additional components add noise")


# ============================================================================
# SCIPY OPTIMIZER (continuous weights)
# ============================================================================

print(f"\n\n{'=' * 100}")
print("SCIPY OPTIMIZER — CONTINUOUS WEIGHT OPTIMIZATION")
print("=" * 100)
print("Finding optimal continuous weights using scipy.optimize")
print("Constraints: weights sum to 1.0, DC >= 0.40, each component >= 0.0")

eval_data = wr_eval[score_cols + ['hit24', 'best_ppr']].dropna(subset=['hit24']).copy()
X = eval_data[score_cols].values
y_hit24 = eval_data['hit24'].values
has_ppr = eval_data['best_ppr'].notna()
y_ppr = eval_data.loc[has_ppr, 'best_ppr'].values
X_ppr = eval_data.loc[has_ppr, score_cols].values

print(f"\nOptimization sample: {len(X)} WRs (hit24), {len(X_ppr)} WRs (best_ppr)")


def neg_correlation(weights, X, y):
    """Negative Pearson r (to minimize)."""
    scores = X @ weights
    if np.std(scores) < 1e-10:
        return 0
    r = np.corrcoef(scores, y)[0, 1]
    return -r


def neg_avg_correlation(weights, X_hit, y_hit, X_ppr, y_ppr):
    """Negative average correlation across both outcomes."""
    s1 = X_hit @ weights
    r1 = np.corrcoef(s1, y_hit)[0, 1] if np.std(s1) > 1e-10 else 0
    s2 = X_ppr @ weights
    r2 = np.corrcoef(s2, y_ppr)[0, 1] if np.std(s2) > 1e-10 else 0
    return -(r1 + r2) / 2


# Constraints
constraints = [
    {'type': 'eq', 'fun': lambda w: sum(w) - 1.0},  # Sum to 1
]
bounds = [
    (0.40, 0.80),  # DC
    (0.05, 0.35),  # Breakout
    (0.00, 0.15),  # Teammate
    (0.00, 0.20),  # RAS
    (0.00, 0.15),  # Early Declare
    (0.00, 0.15),  # Rushing
]

x0 = [0.65, 0.20, 0.00, 0.15, 0.00, 0.00]  # Current weights

# Optimize for hit24
print("\n--- Optimizing for HIT24 ---")
res_hit = minimize(neg_correlation, x0, args=(X, y_hit24),
                   method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'maxiter': 1000})
w_hit = res_hit.x
r_hit = -res_hit.fun
print(f"  Optimal weights: DC={w_hit[0]:.0%} / BO={w_hit[1]:.0%} / TM={w_hit[2]:.0%} / "
      f"RAS={w_hit[3]:.0%} / ED={w_hit[4]:.0%} / RUSH={w_hit[5]:.0%}")
print(f"  Correlation: r = {r_hit:+.4f}")

# Optimize for best_ppr
print("\n--- Optimizing for BEST_PPR ---")
res_ppr = minimize(neg_correlation, x0, args=(X_ppr, y_ppr),
                   method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'maxiter': 1000})
w_ppr = res_ppr.x
r_ppr = -res_ppr.fun
print(f"  Optimal weights: DC={w_ppr[0]:.0%} / BO={w_ppr[1]:.0%} / TM={w_ppr[2]:.0%} / "
      f"RAS={w_ppr[3]:.0%} / ED={w_ppr[4]:.0%} / RUSH={w_ppr[5]:.0%}")
print(f"  Correlation: r = {r_ppr:+.4f}")

# Optimize for average of both
if len(X_ppr) >= 10:
    print("\n--- Optimizing for AVERAGE (hit24 + best_ppr) ---")
    res_avg = minimize(neg_avg_correlation, x0, args=(X, y_hit24, X_ppr, y_ppr),
                       method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 1000})
    w_avg = res_avg.x
    r_avg = -res_avg.fun
    print(f"  Optimal weights: DC={w_avg[0]:.0%} / BO={w_avg[1]:.0%} / TM={w_avg[2]:.0%} / "
          f"RAS={w_avg[3]:.0%} / ED={w_avg[4]:.0%} / RUSH={w_avg[5]:.0%}")
    print(f"  Avg correlation: r = {r_avg:+.4f}")

    # Compare to current
    current_scores = X @ np.array(x0)
    r_current_hit = np.corrcoef(current_scores, y_hit24)[0, 1]
    current_ppr_scores = X_ppr @ np.array(x0)
    r_current_ppr = np.corrcoef(current_ppr_scores, y_ppr)[0, 1]
    r_current_avg = (r_current_hit + r_current_ppr) / 2

    print(f"\n  Current (65/20/0/15/0/0): avg r = {r_current_avg:+.4f}")
    print(f"  Optimized:               avg r = {r_avg:+.4f}")
    print(f"  Improvement:             {r_avg - r_current_avg:+.4f}")


# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n\n{'=' * 100}")
print("SUMMARY")
print("=" * 100)

print(f"""
Key findings:

1. SAMPLE: {len(wr_eval)} WRs with hit24 outcome data (2015-2024 drafts)

2. PARTIAL CORRELATIONS: Which components add unique value beyond the others?
   (See partial correlations table above)

3. GRID SEARCH: Top configuration from {len(configs)} tested:
   {best['config']} → avg r = {best['avg_r']:+.3f}

4. SCIPY OPTIMIZER: Continuous optimization result:""")

if len(X_ppr) >= 10:
    print(f"   DC={w_avg[0]:.0%} / BO={w_avg[1]:.0%} / TM={w_avg[2]:.0%} / "
          f"RAS={w_avg[3]:.0%} / ED={w_avg[4]:.0%} / RUSH={w_avg[5]:.0%}")
    print(f"   avg r = {r_avg:+.4f} (vs current {r_current_avg:+.4f})")

print(f"""
5. RECOMMENDATION: Compare the optimized weights to the current 3-component
   model. If the improvement is < 0.02, the added complexity of 6 components
   may not be worth it. If > 0.03, the new components add real value.
""")
