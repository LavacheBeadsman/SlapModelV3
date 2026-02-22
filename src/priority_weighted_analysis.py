"""
Priority-Weighted Outcome Analysis
====================================
Outcome weights: first_3yr_ppg=40%, hit24=25%, hit12=20%, career_ppg=15%

1. Partial correlations for rush/return versions at 3 control levels
2. 4-component optimization with priority-weighted objective
3. 5-model head-to-head (Models A-E)
4. Raw first_3yr_ppg averages for rush share and return involvement
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def dc_score(pick):
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))

def wr_breakout_score(breakout_age, dominator_pct):
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
    data = pd.DataFrame({'x': x.values if hasattr(x, 'values') else x,
                          'y': y.values if hasattr(y, 'values') else y})
    for i, cov in enumerate(covariates):
        data[f'cov_{i}'] = cov.values if hasattr(cov, 'values') else cov
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
# LOAD & MERGE
# ============================================================================

print("=" * 110)
print("PRIORITY-WEIGHTED OUTCOME ANALYSIS")
print("Weights: first_3yr_ppg=40% | hit24=25% | hit12=20% | career_ppg=15%")
print("=" * 110)

wr = pd.read_csv('data/wr_backtest_all_components.csv')
ret = pd.read_csv('data/wr_return_stats.csv')
team_rush = pd.read_csv('data/wr_team_rushing.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
tm = pd.read_csv('data/wr_teammate_scores.csv')

# Merge returns
ret_cols = ['player_name', 'draft_year', 'kr_no', 'kr_yds', 'kr_td',
            'pr_no', 'pr_yds', 'pr_td', 'total_returns', 'total_return_yds',
            'total_return_td']
wr = wr.merge(ret[ret_cols], on=['player_name', 'draft_year'], how='left')

# Merge team rushing
tr_cols = ['player_name', 'draft_year', 'team_rush_yds', 'team_pass_yds', 'team_total_yds']
wr = wr.merge(team_rush[tr_cols], on=['player_name', 'draft_year'], how='left')

# Merge outcomes
out_wr = outcomes[outcomes['position'] == 'WR'][
    ['player_name', 'draft_year', 'first_3yr_ppg', 'career_ppg']
].copy()
wr = wr.merge(out_wr, on=['player_name', 'draft_year'], how='left')

# Merge teammate
tm_lookup = dict(zip(zip(tm['player_name'], tm['draft_year']), tm['avg_teammate_dc']))
wr['teammate_dc'] = wr.apply(lambda x: tm_lookup.get((x['player_name'], x['draft_year']), 0), axis=1)

# Fill NaN
for c in ['kr_no', 'kr_yds', 'kr_td', 'pr_no', 'pr_yds', 'pr_td',
          'total_returns', 'total_return_yds', 'total_return_td']:
    wr[c] = wr[c].fillna(0)

# Compute scores
wr['s_dc'] = wr['pick'].apply(dc_score)
wr['s_breakout'] = wr.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)
wr['s_teammate'] = wr['teammate_dc'].clip(0, 100)
wr['s_declare'] = wr['early_declare'] * 100

# Rush share
wr['rush_share'] = np.where(
    wr['team_rush_yds'].notna() & (wr['team_rush_yds'] > 0),
    wr['rush_yards'].clip(lower=0) / wr['team_rush_yds'] * 100, np.nan)

# Eval sample
wr_eval = wr[wr['hit24'].notna() & wr['draft_year'].between(2015, 2024)].copy()
print(f"\nLoaded: {len(wr)} WRs, eval sample: {len(wr_eval)}")

all_outcomes = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
outcome_weights = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}

for out in all_outcomes:
    n = wr_eval[out].notna().sum()
    print(f"  {out:<15}: {n} WRs  (weight: {outcome_weights[out]:.0%})")


# ============================================================================
# BUILD VERSIONS FOR TESTING
# ============================================================================

# H: All-purpose share
wr_eval['all_purpose'] = wr_eval['rush_yards'].clip(lower=0) + wr_eval['total_return_yds']
wr_eval['v_H'] = np.where(
    wr_eval['team_total_yds'].notna() & (wr_eval['team_total_yds'] > 0),
    wr_eval['all_purpose'] / wr_eval['team_total_yds'] * 100, np.nan)
p99_h = wr_eval['v_H'].quantile(0.99)
wr_eval['v_H_norm'] = (wr_eval['v_H'] / p99_h * 100).clip(0, 100) if p99_h > 0 else 0
wr_eval['v_H_norm'] = wr_eval['v_H_norm'].fillna(0)

# G: Return volume
p99_g = wr_eval['total_returns'].quantile(0.99)
wr_eval['v_G'] = (wr_eval['total_returns'] / p99_g * 100).clip(0, 100) if p99_g > 0 else 0

# E: Return yards
p99_e = wr_eval['total_return_yds'].quantile(0.99)
wr_eval['v_E'] = (wr_eval['total_return_yds'] / p99_e * 100).clip(0, 100) if p99_e > 0 else 0

# B5: Rush share >5%
wr_eval['v_B5'] = np.where(
    wr_eval['rush_share'].notna() & (wr_eval['rush_share'] >= 5), 1, 0)

test_versions = {
    'H: All-purpose share':  'v_H_norm',
    'G: Return volume':      'v_G',
    'E: Return yards':       'v_E',
    'B: Rush share >5%':     'v_B5',
}


# ============================================================================
# PART 1: PARTIAL CORRELATIONS AT 3 CONTROL LEVELS
# ============================================================================

control_levels = [
    ("DC only", ['s_dc']),
    ("DC + Breakout", ['s_dc', 's_breakout']),
    ("DC + BO + TM + ED", ['s_dc', 's_breakout', 's_teammate', 's_declare']),
]

for ctrl_label, ctrl_cols in control_levels:
    print(f"\n\n{'=' * 110}")
    print(f"PARTIAL CORRELATIONS — Controlling for {ctrl_label}")
    print("=" * 110)

    print(f"\n{'Version':<28}", end="")
    for out in all_outcomes:
        tag = " <<<" if out == 'first_3yr_ppg' else ""
        print(f"  {'r('+out+')':>9} {'p':>7}{tag}", end="")
    print(f"  {'PRI-AVG':>8}")
    print("-" * 120)

    for vlabel, vcol in test_versions.items():
        row = f"  {vlabel:<26}"
        weighted_sum = 0
        total_weight = 0
        for out in all_outcomes:
            covs = [wr_eval[c] for c in ctrl_cols]
            r, p, n = partial_correlation(wr_eval[vcol], wr_eval[out], covs)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f"  {r:>+.4f}{sig:<3} {p:>7.4f}   "
            if not np.isnan(r):
                weighted_sum += outcome_weights[out] * r
                total_weight += outcome_weights[out]
        pri_avg = weighted_sum / total_weight if total_weight > 0 else np.nan
        row += f"  {pri_avg:>+.4f}"
        print(row)

    # Baselines
    print(f"\n  {'--- BASELINES ---':<26}")
    for bl, bvals in [('Plain breakout', wr_eval['s_breakout']),
                       ('Rush 20+ yds (binary)', (wr_eval['rush_yards'] >= 20).astype(int))]:
        row = f"  {bl:<26}"
        weighted_sum = 0
        total_weight = 0
        for out in all_outcomes:
            covs = [wr_eval[c] for c in ctrl_cols]
            r, p, n = partial_correlation(bvals, wr_eval[out], covs)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f"  {r:>+.4f}{sig:<3} {p:>7.4f}   "
            if not np.isnan(r):
                weighted_sum += outcome_weights[out] * r
                total_weight += outcome_weights[out]
        pri_avg = weighted_sum / total_weight if total_weight > 0 else np.nan
        row += f"  {pri_avg:>+.4f}"
        print(row)


# ============================================================================
# PART 2: 4-COMPONENT OPTIMIZATION — PRIORITY-WEIGHTED
# ============================================================================

print(f"\n\n{'=' * 110}")
print("4-COMPONENT OPTIMIZATION — PRIORITY-WEIGHTED OBJECTIVE")
print("Weights: first_3yr_ppg=40% | hit24=25% | hit12=20% | career_ppg=15%")
print("=" * 110)

score_cols = ['s_dc', 's_breakout', 's_teammate', 's_declare']
labels_4 = ['Draft Capital', 'Breakout Age', 'Teammate', 'Early Declare']

# Build data arrays for each outcome
outcome_data = {}
for out in all_outcomes:
    valid = wr_eval[score_cols + [out]].dropna(subset=[out]).copy()
    outcome_data[out] = {
        'X': valid[score_cols].values,
        'y': valid[out].values,
        'n': len(valid),
    }
# Also best_ppr for reference
valid_ppr = wr_eval[score_cols + ['best_ppr']].dropna(subset=['best_ppr']).copy()
outcome_data['best_ppr'] = {
    'X': valid_ppr[score_cols].values,
    'y': valid_ppr['best_ppr'].values,
    'n': len(valid_ppr),
}

def neg_priority_weighted(weights):
    """Priority-weighted average correlation: 40% 3yr_ppg, 25% hit24, 20% hit12, 15% career_ppg."""
    total = 0
    for out, w in outcome_weights.items():
        X = outcome_data[out]['X']
        y = outcome_data[out]['y']
        s = X @ weights
        if np.std(s) > 1e-10:
            r = np.corrcoef(s, y)[0, 1]
            total += w * r
    return -total

def neg_equal_weighted(weights):
    """Equal-weighted average across all 4."""
    rs = []
    for out in all_outcomes:
        X = outcome_data[out]['X']
        y = outcome_data[out]['y']
        s = X @ weights
        if np.std(s) > 1e-10:
            rs.append(np.corrcoef(s, y)[0, 1])
    return -(sum(rs) / len(rs)) if rs else 0

def neg_original_2(weights):
    """Original: avg(hit24, best_ppr)."""
    rs = []
    for out in ['hit24', 'best_ppr']:
        X = outcome_data[out]['X']
        y = outcome_data[out]['y']
        s = X @ weights
        if np.std(s) > 1e-10:
            rs.append(np.corrcoef(s, y)[0, 1])
    return -(sum(rs) / len(rs)) if rs else 0

constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]
bounds_4 = [
    (0.40, 0.85),  # DC
    (0.05, 0.35),  # Breakout
    (0.00, 0.25),  # Teammate
    (0.00, 0.20),  # Early Declare
]

starts = [
    [0.65, 0.20, 0.10, 0.05],
    [0.70, 0.15, 0.08, 0.07],
    [0.75, 0.10, 0.08, 0.07],
    [0.80, 0.10, 0.05, 0.05],
    [0.60, 0.25, 0.08, 0.07],
    [0.85, 0.05, 0.05, 0.05],
    [0.55, 0.30, 0.08, 0.07],
    [0.76, 0.15, 0.04, 0.05],
]

# Run all 3 objectives
results_by_obj = {}
for obj_label, obj_fn in [
    ("Original (hit24 + best_ppr)", neg_original_2),
    ("Equal-weighted (4 outcomes)", neg_equal_weighted),
    ("Priority-weighted (40/25/20/15)", neg_priority_weighted),
]:
    print(f"\n{'─' * 80}")
    print(f"Objective: {obj_label}")
    print(f"{'─' * 80}")

    best_r = -999
    best_w = None
    for x0 in starts:
        res = minimize(obj_fn, x0, method='SLSQP', bounds=bounds_4,
                       constraints=constraints, options={'maxiter': 1000})
        r_opt = -res.fun
        if r_opt > best_r:
            best_r = r_opt
            best_w = res.x

    print(f"  Weights: DC={best_w[0]:.0%} / BO={best_w[1]:.0%} / "
          f"TM={best_w[2]:.0%} / ED={best_w[3]:.0%}")
    print(f"  Objective value: {best_r:+.4f}")

    # Show all individual correlations
    print(f"\n  Individual correlations:")
    for out in all_outcomes + ['best_ppr']:
        X = outcome_data[out]['X']
        y = outcome_data[out]['y']
        s = X @ best_w
        r = np.corrcoef(s, y)[0, 1]
        tag = f" (weight: {outcome_weights[out]:.0%})" if out in outcome_weights else ""
        print(f"    {out:<15}: r = {r:+.4f}  (N={len(y)}){tag}")

    # Compute priority-weighted avg at these weights
    pri_avg = sum(outcome_weights[out] * np.corrcoef(
        outcome_data[out]['X'] @ best_w, outcome_data[out]['y'])[0, 1]
        for out in all_outcomes)
    print(f"\n  Priority-weighted avg r: {pri_avg:+.4f}")

    results_by_obj[obj_label] = {'weights': best_w, 'obj_val': best_r, 'pri_avg': pri_avg}

# Comparison table
print(f"\n\n{'─' * 80}")
print("WEIGHT COMPARISON ACROSS 3 OBJECTIVES")
print(f"{'─' * 80}")
print(f"\n{'Component':<18} {'Orig (2-out)':>14} {'Equal (4-out)':>14} {'Priority':>14}")
print("-" * 65)
for i, lab in enumerate(labels_4):
    vals = [results_by_obj[k]['weights'][i] for k in results_by_obj]
    print(f"  {lab:<16} {vals[0]:>14.0%} {vals[1]:>14.0%} {vals[2]:>14.0%}")

print(f"\n  {'Priority-wtd avg r':<16}", end="")
for k in results_by_obj:
    print(f" {results_by_obj[k]['pri_avg']:>14.4f}", end="")
print()


# ============================================================================
# PART 3: 5-MODEL HEAD-TO-HEAD — PRIORITY-WEIGHTED
# ============================================================================

print(f"\n\n{'=' * 110}")
print("5-MODEL HEAD-TO-HEAD — PRIORITY-WEIGHTED OBJECTIVE")
print("=" * 110)
print("""
Model A: DC + Breakout + Teammate + Early Declare (4-component, no rushing/returns)
Model B: DC + Breakout(+5 rush bonus) + Teammate + ED (rushing folded into breakout)
Model C: DC + Breakout + Teammate + ED + Rush share 5% (5th component)
Model D: DC + Breakout + Teammate + ED + Return volume (5th component)
Model E: DC + Breakout + Teammate + ED + All-purpose share (5th component)
""")

# Build enhanced breakout for Model B
rush_flag = (wr_eval['rush_yards'] >= 20).astype(int)
wr_eval['s_bo_rush'] = (wr_eval['s_breakout'] + rush_flag * 5).clip(0, 99.9)

# Build 5th component scores
wr_eval['s_rush5pct'] = wr_eval['v_B5'] * 100  # 0 or 100
wr_eval['s_retvol'] = wr_eval['v_G']
wr_eval['s_allpurp'] = wr_eval['v_H_norm']

models = {
    'A: Base 4-comp': {
        'cols': ['s_dc', 's_breakout', 's_teammate', 's_declare'],
        'labels': ['DC', 'Breakout', 'Teammate', 'EarlyDec'],
        'bounds': [(0.40, 0.85), (0.05, 0.35), (0.00, 0.25), (0.00, 0.20)],
    },
    'B: Rush in breakout': {
        'cols': ['s_dc', 's_bo_rush', 's_teammate', 's_declare'],
        'labels': ['DC', 'BO+Rush', 'Teammate', 'EarlyDec'],
        'bounds': [(0.40, 0.85), (0.05, 0.35), (0.00, 0.25), (0.00, 0.20)],
    },
    'C: +Rush share 5%': {
        'cols': ['s_dc', 's_breakout', 's_teammate', 's_declare', 's_rush5pct'],
        'labels': ['DC', 'Breakout', 'Teammate', 'EarlyDec', 'Rush5%'],
        'bounds': [(0.40, 0.80), (0.05, 0.30), (0.00, 0.20), (0.00, 0.15), (0.00, 0.10)],
    },
    'D: +Return volume': {
        'cols': ['s_dc', 's_breakout', 's_teammate', 's_declare', 's_retvol'],
        'labels': ['DC', 'Breakout', 'Teammate', 'EarlyDec', 'RetVol'],
        'bounds': [(0.40, 0.80), (0.05, 0.30), (0.00, 0.20), (0.00, 0.15), (0.00, 0.10)],
    },
    'E: +All-purpose share': {
        'cols': ['s_dc', 's_breakout', 's_teammate', 's_declare', 's_allpurp'],
        'labels': ['DC', 'Breakout', 'Teammate', 'EarlyDec', 'AllPurp'],
        'bounds': [(0.40, 0.80), (0.05, 0.30), (0.00, 0.20), (0.00, 0.15), (0.00, 0.10)],
    },
}

model_results = {}

for model_name, model_cfg in models.items():
    cols = model_cfg['cols']
    lbls = model_cfg['labels']
    bnds = model_cfg['bounds']
    n_comp = len(cols)

    print(f"\n{'─' * 80}")
    print(f"MODEL: {model_name}")
    print(f"{'─' * 80}")

    # Build outcome arrays
    m_outcome_data = {}
    for out in all_outcomes:
        valid = wr_eval[cols + [out]].dropna(subset=[out]).copy()
        m_outcome_data[out] = {'X': valid[cols].values, 'y': valid[out].values}
    valid_ppr = wr_eval[cols + ['best_ppr']].dropna(subset=['best_ppr']).copy()
    m_outcome_data['best_ppr'] = {'X': valid_ppr[cols].values, 'y': valid_ppr['best_ppr'].values}

    def make_obj(od):
        def obj(weights):
            total = 0
            for out, w in outcome_weights.items():
                X = od[out]['X']
                y = od[out]['y']
                s = X @ weights
                if np.std(s) > 1e-10:
                    r = np.corrcoef(s, y)[0, 1]
                    total += w * r
            return -total
        return obj

    obj_fn = make_obj(m_outcome_data)
    m_constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]

    # Generate starts
    m_starts = []
    if n_comp == 4:
        m_starts = [
            [0.65, 0.20, 0.10, 0.05], [0.70, 0.15, 0.08, 0.07],
            [0.75, 0.10, 0.08, 0.07], [0.80, 0.10, 0.05, 0.05],
            [0.60, 0.25, 0.08, 0.07], [0.76, 0.15, 0.04, 0.05],
        ]
    else:
        m_starts = [
            [0.60, 0.18, 0.08, 0.07, 0.07], [0.65, 0.15, 0.05, 0.05, 0.10],
            [0.70, 0.12, 0.05, 0.05, 0.08], [0.75, 0.10, 0.05, 0.05, 0.05],
            [0.55, 0.20, 0.08, 0.07, 0.10], [0.65, 0.20, 0.05, 0.05, 0.05],
        ]

    best_r = -999
    best_w = None
    for x0 in m_starts:
        res = minimize(obj_fn, x0, method='SLSQP', bounds=bnds,
                       constraints=m_constraints, options={'maxiter': 1000})
        r_opt = -res.fun
        if r_opt > best_r:
            best_r = r_opt
            best_w = res.x

    # Show weights
    w_str = " / ".join(f"{l}={best_w[i]:.0%}" for i, l in enumerate(lbls))
    print(f"  Weights: {w_str}")
    print(f"  Priority-weighted avg r: {best_r:+.4f}")

    # Individual correlations
    print(f"\n  {'Outcome':<15} {'r':>8} {'weight':>8} {'weighted_r':>12}")
    print(f"  {'-'*45}")
    for out in all_outcomes:
        X = m_outcome_data[out]['X']
        y = m_outcome_data[out]['y']
        s = X @ best_w
        r = np.corrcoef(s, y)[0, 1]
        wr_r = outcome_weights[out] * r
        tag = "  <<<" if out == 'first_3yr_ppg' else ""
        print(f"  {out:<15} {r:>+8.4f} {outcome_weights[out]:>8.0%} {wr_r:>+12.4f}{tag}")

    # best_ppr for reference
    X_p = m_outcome_data['best_ppr']['X']
    y_p = m_outcome_data['best_ppr']['y']
    s_p = X_p @ best_w
    r_p = np.corrcoef(s_p, y_p)[0, 1]
    print(f"  {'best_ppr':<15} {r_p:>+8.4f} {'(ref)':>8}")

    model_results[model_name] = {'weights': best_w, 'labels': lbls, 'pri_avg': best_r}


# Summary table
print(f"\n\n{'=' * 110}")
print("5-MODEL COMPARISON SUMMARY — PRIORITY-WEIGHTED")
print("=" * 110)

print(f"\n{'Model':<30} {'Weights':>40} {'Pri-Avg r':>12}")
print("-" * 85)
sorted_models = sorted(model_results.items(), key=lambda x: x[1]['pri_avg'], reverse=True)
for model_name, info in sorted_models:
    w_str = "/".join(f"{info['weights'][i]:.0%}" for i in range(len(info['labels'])))
    marker = " <<<" if model_name == sorted_models[0][0] else ""
    print(f"  {model_name:<28} {w_str:>40} {info['pri_avg']:>+12.4f}{marker}")

# Delta from base
base_r = model_results['A: Base 4-comp']['pri_avg']
print(f"\n  Delta from base (Model A):")
for model_name, info in sorted_models:
    if model_name == 'A: Base 4-comp':
        continue
    delta = info['pri_avg'] - base_r
    print(f"    {model_name}: {delta:>+.4f}")


# ============================================================================
# PART 4: RAW first_3yr_ppg AVERAGES
# ============================================================================

print(f"\n\n{'=' * 110}")
print("RAW first_3yr_ppg AVERAGES (NO CONTROLS)")
print("=" * 110)

eval_3yr = wr_eval[wr_eval['first_3yr_ppg'].notna()].copy()
overall_3yr = eval_3yr['first_3yr_ppg'].mean()
overall_career = eval_3yr['career_ppg'].mean()

print(f"\nOverall averages (N={len(eval_3yr)}):")
print(f"  first_3yr_ppg: {overall_3yr:.2f}")
print(f"  career_ppg:    {eval_3yr['career_ppg'].mean():.2f}")

# Rush share 5%+
print(f"\n--- Rush share 5%+ vs rest ---")
rush5_q = eval_3yr[eval_3yr['rush_share'].notna() & (eval_3yr['rush_share'] >= 5)]
rush5_nq = eval_3yr[~(eval_3yr['rush_share'].notna() & (eval_3yr['rush_share'] >= 5))]

print(f"\n  {'Group':<25} {'N':>5} {'first_3yr_ppg':>15} {'career_ppg':>12}")
print(f"  {'-'*60}")
print(f"  {'Rush share >=5%':<25} {len(rush5_q):>5} {rush5_q['first_3yr_ppg'].mean():>15.2f} "
      f"{rush5_q['career_ppg'].mean():>12.2f}")
print(f"  {'Rush share <5%':<25} {len(rush5_nq):>5} {rush5_nq['first_3yr_ppg'].mean():>15.2f} "
      f"{rush5_nq['career_ppg'].mean():>12.2f}")
print(f"  {'Difference':<25} {'':>5} "
      f"{rush5_q['first_3yr_ppg'].mean() - rush5_nq['first_3yr_ppg'].mean():>+15.2f} "
      f"{rush5_q['career_ppg'].mean() - rush5_nq['career_ppg'].mean():>+12.2f}")

# List rush share 5%+ players with their first_3yr_ppg
print(f"\n  Rush share 5%+ players with first_3yr_ppg:")
rs5_detail = rush5_q.sort_values('rush_share', ascending=False)[
    ['player_name', 'draft_year', 'pick', 'college', 'rush_share',
     'first_3yr_ppg', 'career_ppg']
]
print(f"  {'Name':<25} {'Year':>5} {'Pick':>5} {'Share%':>7} {'3yr_ppg':>8} {'car_ppg':>8}")
print(f"  {'-'*65}")
for _, p in rs5_detail.iterrows():
    ppg3 = f"{p['first_3yr_ppg']:.1f}" if pd.notna(p['first_3yr_ppg']) else "—"
    ppgc = f"{p['career_ppg']:.1f}" if pd.notna(p['career_ppg']) else "—"
    print(f"  {p['player_name']:<23} {p['draft_year']:>5} {p['pick']:>5} "
          f"{p['rush_share']:>7.2f} {ppg3:>8} {ppgc:>8}")


# 10+ returns vs rest
print(f"\n\n--- 10+ returns vs rest ---")
ret10_q = eval_3yr[eval_3yr['total_returns'] >= 10]
ret10_nq = eval_3yr[eval_3yr['total_returns'] < 10]

print(f"\n  {'Group':<25} {'N':>5} {'first_3yr_ppg':>15} {'career_ppg':>12}")
print(f"  {'-'*60}")
print(f"  {'10+ returns':<25} {len(ret10_q):>5} {ret10_q['first_3yr_ppg'].mean():>15.2f} "
      f"{ret10_q['career_ppg'].mean():>12.2f}")
print(f"  {'<10 returns':<25} {len(ret10_nq):>5} {ret10_nq['first_3yr_ppg'].mean():>15.2f} "
      f"{ret10_nq['career_ppg'].mean():>12.2f}")
print(f"  {'Difference':<25} {'':>5} "
      f"{ret10_q['first_3yr_ppg'].mean() - ret10_nq['first_3yr_ppg'].mean():>+15.2f} "
      f"{ret10_q['career_ppg'].mean() - ret10_nq['career_ppg'].mean():>+12.2f}")

# T-tests
t_rush, p_rush = stats.ttest_ind(
    rush5_q['first_3yr_ppg'].dropna(), rush5_nq['first_3yr_ppg'].dropna())
t_ret, p_ret = stats.ttest_ind(
    ret10_q['first_3yr_ppg'].dropna(), ret10_nq['first_3yr_ppg'].dropna())
print(f"\n  T-test p-values (first_3yr_ppg):")
print(f"    Rush share 5%+: p = {p_rush:.4f}")
print(f"    10+ returns:    p = {p_ret:.4f}")

print("\n\nDone.")
