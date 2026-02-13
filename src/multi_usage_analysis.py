"""
Multi-Usage & Return Specialist Analysis
==========================================
Tests 8 versions of a "versatility" signal for WR SLAP model:

A: Any rushing OR 10+ returns (binary)
B: Any rushing OR 20+ returns (binary)
C: Both rushing AND returns (binary)
D: Weighted all-purpose continuous score added to breakout
E1: Return specialist (volume + efficiency + TDs), normalized
E2: Return specialist (yards + TDs), capped at 100
E3: Return specialist binary (TD or elite efficiency on 10+ att)

Then: fold best into breakout, compare against plain and rushing-only.
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
# LOAD & MERGE DATA
# ============================================================================

print("=" * 100)
print("MULTI-USAGE & RETURN SPECIALIST ANALYSIS")
print("=" * 100)

wr = pd.read_csv('data/wr_backtest_all_components.csv')
ret = pd.read_csv('data/wr_return_stats.csv')
tm = pd.read_csv('data/wr_teammate_scores.csv')

print(f"Loaded: {len(wr)} WRs, {len(ret)} return records")

# Merge return data
ret_cols = ['player_name', 'draft_year', 'kr_no', 'kr_yds', 'kr_td',
            'pr_no', 'pr_yds', 'pr_td', 'total_returns', 'total_return_yds',
            'total_return_td', 'has_team_data']
wr = wr.merge(ret[ret_cols], on=['player_name', 'draft_year'], how='left')

# Fill NaN return data with 0
for c in ['kr_no', 'kr_yds', 'kr_td', 'pr_no', 'pr_yds', 'pr_td',
          'total_returns', 'total_return_yds', 'total_return_td']:
    wr[c] = wr[c].fillna(0)
wr['has_team_data'] = wr['has_team_data'].fillna(False)

# Compute scores
wr['s_dc'] = wr['pick'].apply(dc_score)
wr['s_breakout_plain'] = wr.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)

# Merge teammate scores
tm_lookup = dict(zip(zip(tm['player_name'], tm['draft_year']), tm['avg_teammate_dc']))
wr['teammate_dc'] = wr.apply(lambda x: tm_lookup.get((x['player_name'], x['draft_year']), 0), axis=1)

# Filter to eval sample (2015-2024 with outcomes)
wr_eval = wr[wr['hit24'].notna() & wr['draft_year'].between(2015, 2024)].copy()
n_eval = len(wr_eval)
print(f"Evaluation sample: {n_eval} WRs with hit24\n")


# ============================================================================
# BUILD ALL VERSION FLAGS
# ============================================================================

print("=" * 100)
print("BUILDING ALL 8 VERSIONS")
print("=" * 100)

rush_flag = (wr['rush_yards'] >= 20).astype(int)
ret_10 = (wr['total_returns'] >= 10).astype(int)
ret_20 = (wr['total_returns'] >= 20).astype(int)
ret_5 = (wr['total_returns'] >= 5).astype(int)

# Version A: 20+ rush yards OR 10+ returns
wr['v_A'] = ((rush_flag == 1) | (ret_10 == 1)).astype(int)

# Version B: 20+ rush yards OR 20+ returns
wr['v_B'] = ((rush_flag == 1) | (ret_20 == 1)).astype(int)

# Version C: 20+ rush yards AND 5+ returns
wr['v_C'] = ((rush_flag == 1) & (ret_5 == 1)).astype(int)

# Version D: continuous all-purpose — (rush_yards + return_yards) / 10, cap 5, added to breakout
wr['d_bonus'] = np.minimum((wr['rush_yards'].clip(lower=0) + wr['total_return_yds']) / 10, 5)
wr['v_D'] = (wr['s_breakout_plain'] + wr['d_bonus']).clip(0, 99.9)

# Version E1: return specialist = (attempts × 0.3) + (avg_yds × 2) + (TDs × 15), normalized
wr['ret_avg'] = np.where(wr['total_returns'] > 0,
                          wr['total_return_yds'] / wr['total_returns'], 0)
wr['e1_raw'] = (wr['total_returns'] * 0.3 + wr['ret_avg'] * 2 + wr['total_return_td'] * 15)
e1_max = wr['e1_raw'].quantile(0.99)
wr['v_E1'] = (wr['e1_raw'] / e1_max * 100).clip(0, 100) if e1_max > 0 else 0

# Version E2: (total_return_yards / 10) + (TDs × 20), capped at 100
wr['v_E2'] = (wr['total_return_yds'] / 10 + wr['total_return_td'] * 20).clip(0, 100)

# Version E3: binary — return TD OR (25+ avg on 10+ attempts)
wr['v_E3'] = ((wr['total_return_td'] > 0) |
              ((wr['ret_avg'] >= 25) & (wr['total_returns'] >= 10))).astype(int)

# Summary
versions = {
    'A (rush OR 10+ ret)': 'v_A',
    'B (rush OR 20+ ret)': 'v_B',
    'C (rush AND ret)': 'v_C',
    'D (continuous bonus)': 'v_D',
    'E1 (ret: vol+eff+TD)': 'v_E1',
    'E2 (ret: yds+TD)': 'v_E2',
    'E3 (ret: TD or elite)': 'v_E3',
}

# For binary versions, show qualifier count
print(f"\n{'Version':<30} {'Type':>10} {'Qual/High':>10} {'Non-Q/Low':>10}")
print("-" * 65)
for label, col in versions.items():
    if col in ['v_D', 'v_E1', 'v_E2']:
        # Continuous — use median split
        med = wr[col].median()
        n_high = (wr[col] > med).sum()
        n_low = (wr[col] <= med).sum()
        print(f"  {label:<28} {'continuous':>10} {n_high:>10} {n_low:>10}  (median={med:.1f})")
    else:
        n1 = (wr[col] == 1).sum()
        n0 = (wr[col] == 0).sum()
        print(f"  {label:<28} {'binary':>10} {n1:>10} {n0:>10}")


# ============================================================================
# HIT24 RATES FOR EACH VERSION
# ============================================================================

print(f"\n\n{'=' * 100}")
print("HIT24 RATES: QUALIFIERS vs NON-QUALIFIERS")
print("=" * 100)

# Refresh eval
for col in versions.values():
    if col not in wr_eval.columns:
        wr_eval = wr[wr['hit24'].notna() & wr['draft_year'].between(2015, 2024)].copy()
        break

print(f"\n{'Version':<30} {'Q Hit':>8} {'Q N':>5} {'NQ Hit':>8} {'NQ N':>6} {'Diff':>8} {'χ² p':>8}")
print("-" * 80)

hit_results = {}
for label, col in versions.items():
    if col in ['v_D', 'v_E1', 'v_E2']:
        # Continuous — median split
        med = wr_eval[col].median()
        high = wr_eval[wr_eval[col] > med]
        low = wr_eval[wr_eval[col] <= med]
        hit_h = high['hit24'].mean()
        hit_l = low['hit24'].mean()
        diff = hit_h - hit_l
        # Chi-square
        ct = pd.crosstab(wr_eval[col] > med, wr_eval['hit24'])
        if ct.shape == (2, 2):
            chi2, pval, _, _ = stats.chi2_contingency(ct)
        else:
            pval = 1.0
        print(f"  {label:<28} {hit_h:>8.3f} {len(high):>5} {hit_l:>8.3f} {len(low):>6} {diff:>+8.3f} {pval:>8.4f}")
        hit_results[label] = {'hit_q': hit_h, 'hit_nq': hit_l, 'diff': diff, 'p': pval}
    else:
        # Binary
        q = wr_eval[wr_eval[col] == 1]
        nq = wr_eval[wr_eval[col] == 0]
        hit_q = q['hit24'].mean() if len(q) > 0 else 0
        hit_nq = nq['hit24'].mean() if len(nq) > 0 else 0
        diff = hit_q - hit_nq
        ct = pd.crosstab(wr_eval[col], wr_eval['hit24'])
        if ct.shape == (2, 2):
            chi2, pval, _, _ = stats.chi2_contingency(ct)
        else:
            pval = 1.0
        print(f"  {label:<28} {hit_q:>8.3f} {len(q):>5} {hit_nq:>8.3f} {len(nq):>6} {diff:>+8.3f} {pval:>8.4f}")
        hit_results[label] = {'hit_q': hit_q, 'hit_nq': hit_nq, 'diff': diff, 'p': pval}


# ============================================================================
# PARTIAL CORRELATIONS: ALL VERSIONS VS ALL 4 OUTCOMES
# ============================================================================

print(f"\n\n{'=' * 100}")
print("PARTIAL CORRELATIONS (controlling for DC)")
print("=" * 100)

outcomes = ['hit24', 'hit12', 'best_ppr', 'best_rank']

print(f"\n{'Version':<30}", end="")
for out in outcomes:
    print(f" {'r('+out+')':>10} {'p':>7}", end="")
print()
print("-" * 105)

partial_results = {}
for label, col in versions.items():
    row_str = f"  {label:<28}"
    pr_dict = {}
    for out in outcomes:
        r, p, n = partial_correlation(wr_eval[col], wr_eval[out], [wr_eval['s_dc']])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        row_str += f" {r:>+.4f}{sig:<3} {p:>7.4f}"
        pr_dict[out] = (r, p, n)
    print(row_str)
    partial_results[label] = pr_dict

# Also show plain breakout and rushing-only for comparison
print(f"\n  {'--- BASELINES ---':<28}")
rush_only_flag = (wr_eval['rush_yards'] >= 20).astype(int)
for baseline_label, baseline_vals in [
    ('Plain breakout', wr_eval['s_breakout_plain']),
    ('Rushing only (20+ yds)', rush_only_flag),
]:
    row_str = f"  {baseline_label:<28}"
    for out in outcomes:
        r, p, n = partial_correlation(baseline_vals, wr_eval[out], [wr_eval['s_dc']])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        row_str += f" {r:>+.4f}{sig:<3} {p:>7.4f}"
    print(row_str)


# ============================================================================
# HEAD-TO-HEAD RANKING
# ============================================================================

print(f"\n\n{'=' * 100}")
print("HEAD-TO-HEAD RANKING — ALL VERSIONS")
print("=" * 100)
print("Ranked by average of |partial r| for hit24 and best_ppr")

rankings = []
for label in versions:
    r_hit = partial_results[label]['hit24'][0]
    r_ppr = partial_results[label]['best_ppr'][0]
    p_hit = partial_results[label]['hit24'][1]
    p_ppr = partial_results[label]['best_ppr'][1]
    # Signed average (both should be positive for "good" components)
    signed_avg = (r_hit + r_ppr) / 2
    rankings.append({
        'label': label, 'r_hit': r_hit, 'p_hit': p_hit,
        'r_ppr': r_ppr, 'p_ppr': p_ppr, 'avg': signed_avg,
        'col': versions[label],
    })

rankings.sort(key=lambda x: x['avg'], reverse=True)

print(f"\n{'Rank':>4} {'Version':<30} {'r(hit24)':>10} {'p':>8} {'r(ppr)':>10} {'p':>8} {'avg':>8}")
print("-" * 85)
for i, r in enumerate(rankings):
    marker = " <-- BEST" if i == 0 else ""
    print(f"  {i+1:>2}  {r['label']:<28} {r['r_hit']:>+10.4f} {r['p_hit']:>8.4f} "
          f"{r['r_ppr']:>+10.4f} {r['p_ppr']:>8.4f} {r['avg']:>+8.4f}{marker}")

best_version = rankings[0]
second_version = rankings[1]

print(f"\n  Best: {best_version['label']}")
print(f"  Second: {second_version['label']}")


# ============================================================================
# QUALIFIER LISTS — TOP 2 VERSIONS
# ============================================================================

for rank_idx, ver in enumerate([best_version, second_version]):
    label = ver['label']
    col = ver['col']

    print(f"\n\n{'=' * 100}")
    print(f"{'QUALIFIER LIST' if rank_idx == 0 else 'SECOND BEST'}: {label}")
    print("=" * 100)

    if col in ['v_D', 'v_E1', 'v_E2']:
        # Continuous — show top 40
        top = wr.nlargest(40, col)[
            ['player_name', 'draft_year', 'pick', 'round', 'college',
             'rush_yards', 'total_returns', 'total_return_yds', 'total_return_td',
             col, 'hit24', 'best_ppr']
        ]
        print(f"\nTop 40 by {col} score:")
        print(f"{'Name':<25} {'Year':>5} {'Pick':>5} {'Rd':>3} {'College':<18} "
              f"{'RshYd':>6} {'Ret':>4} {'RetYd':>6} {'RetTD':>5} {'Score':>6} {'Hit24':>6} {'PPR':>7}")
        print("-" * 115)
        for _, p in top.iterrows():
            hit = f"{p['hit24']:.0f}" if pd.notna(p['hit24']) else "—"
            ppr = f"{p['best_ppr']:.1f}" if pd.notna(p['best_ppr']) else "—"
            print(f"  {p['player_name']:<23} {p['draft_year']:>5} {p['pick']:>5} {p['round']:>3} "
                  f"{p['college']:<18} {p['rush_yards']:>6.0f} {p['total_returns']:>4.0f} "
                  f"{p['total_return_yds']:>6.0f} {p['total_return_td']:>5.0f} "
                  f"{p[col]:>6.1f} {hit:>6} {ppr:>7}")
    else:
        # Binary — show all qualifiers
        quals = wr[wr[col] == 1].sort_values(['draft_year', 'pick'])[
            ['player_name', 'draft_year', 'pick', 'round', 'college',
             'rush_yards', 'total_returns', 'total_return_yds', 'total_return_td',
             'hit24', 'best_ppr']
        ]
        n_q = len(quals)
        n_h = quals['hit24'].notna().sum()
        hits = quals[quals['hit24'].notna()]['hit24'].sum()
        print(f"\nTotal qualifiers: {n_q}")
        print(f"With outcomes: {n_h}, Hits: {hits:.0f} ({hits/n_h*100:.1f}% hit rate)" if n_h > 0 else "")

        print(f"\n{'Name':<25} {'Year':>5} {'Pick':>5} {'Rd':>3} {'College':<18} "
              f"{'RshYd':>6} {'Ret':>4} {'RetYd':>6} {'RetTD':>5} {'Hit24':>6} {'PPR':>7}")
        print("-" * 110)
        for _, p in quals.iterrows():
            hit = f"{p['hit24']:.0f}" if pd.notna(p['hit24']) else "—"
            ppr = f"{p['best_ppr']:.1f}" if pd.notna(p['best_ppr']) else "—"
            print(f"  {p['player_name']:<23} {p['draft_year']:>5} {p['pick']:>5} {p['round']:>3} "
                  f"{p['college']:<18} {p['rush_yards']:>6.0f} {p['total_returns']:>4.0f} "
                  f"{p['total_return_yds']:>6.0f} {p['total_return_td']:>5.0f} "
                  f"{hit:>6} {ppr:>7}")


# ============================================================================
# FOLD BEST INTO BREAKOUT — COMPARE 3 VERSIONS
# ============================================================================

print(f"\n\n{'=' * 100}")
print("BREAKOUT SCORE COMPARISON: PLAIN vs RUSHING-ONLY vs BEST MULTI-USAGE")
print("=" * 100)

# Plain breakout
wr_eval['bo_plain'] = wr_eval['s_breakout_plain']

# Rushing-only enhanced (+5 if 20+ rush yards)
rush_flag_eval = (wr_eval['rush_yards'] >= 20).astype(int)
wr_eval['bo_rush'] = (wr_eval['s_breakout_plain'] + rush_flag_eval * 5).clip(0, 99.9)

# Best version folded into breakout
best_col = best_version['col']
best_label = best_version['label']

if best_col in ['v_D']:
    # D is already a breakout + bonus score
    wr_eval['bo_best'] = wr_eval[best_col]
elif best_col in ['v_E1', 'v_E2']:
    # Continuous — scale to 0-5 bonus
    e_max = wr_eval[best_col].quantile(0.99)
    bonus = (wr_eval[best_col] / e_max * 5).clip(0, 5) if e_max > 0 else 0
    wr_eval['bo_best'] = (wr_eval['s_breakout_plain'] + bonus).clip(0, 99.9)
else:
    # Binary — +5 if flagged
    wr_eval['bo_best'] = (wr_eval['s_breakout_plain'] + wr_eval[best_col] * 5).clip(0, 99.9)

print(f"\nBest version to fold in: {best_label}")
print(f"Method: {'continuous bonus' if best_col in ['v_D', 'v_E1', 'v_E2'] else '+5 points if flagged'}")

bo_versions = {
    'Plain breakout': 'bo_plain',
    'Rush-only (+5 if 20+ yds)': 'bo_rush',
    f'Best: {best_label}': 'bo_best',
}

print(f"\n{'Breakout Version':<35}", end="")
for out in outcomes:
    print(f" {'r('+out+')':>10} {'p':>7}", end="")
print()
print("-" * 110)

for bo_label, bo_col in bo_versions.items():
    row_str = f"  {bo_label:<33}"
    for out in outcomes:
        r, p, n = partial_correlation(wr_eval[bo_col], wr_eval[out], [wr_eval['s_dc']])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        row_str += f" {r:>+.4f}{sig:<3} {p:>7.4f}"
    print(row_str)


# ============================================================================
# FULL MODEL OPTIMIZATION: BEST ENHANCED BREAKOUT
# ============================================================================

print(f"\n\n{'=' * 100}")
print("4-COMPONENT OPTIMIZATION — PLAIN vs RUSH-ENHANCED vs BEST-ENHANCED")
print("=" * 100)
print("Components: DC / Breakout / Teammate / Early Declare")

wr_eval['s_teammate'] = wr_eval['teammate_dc'].clip(0, 100)
wr_eval['s_declare'] = wr_eval['early_declare'] * 100

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

starts = [
    [0.65, 0.20, 0.10, 0.05],
    [0.70, 0.15, 0.08, 0.07],
    [0.75, 0.10, 0.08, 0.07],
    [0.80, 0.10, 0.05, 0.05],
    [0.60, 0.25, 0.08, 0.07],
]

for model_label, bo_col in bo_versions.items():
    print(f"\n{'─' * 80}")
    print(f"MODEL: {model_label}")
    print(f"{'─' * 80}")

    score_cols = ['s_dc', bo_col, 's_teammate', 's_declare']
    eval_data = wr_eval[score_cols + ['hit24', 'best_ppr']].dropna(subset=['hit24']).copy()
    X = eval_data[score_cols].values
    y_hit = eval_data['hit24'].values
    has_ppr = eval_data['best_ppr'].notna()
    y_ppr = eval_data.loc[has_ppr, 'best_ppr'].values
    X_ppr = eval_data.loc[has_ppr, score_cols].values

    best_r = -999
    best_w = None
    for x0 in starts:
        res = minimize(neg_avg_corr, x0, args=(X, y_hit, X_ppr, y_ppr),
                       method='SLSQP', bounds=bounds_4, constraints=constraints,
                       options={'maxiter': 1000})
        r_opt = -res.fun
        if r_opt > best_r:
            best_r = r_opt
            best_w = res.x

    print(f"  Optimized: DC={best_w[0]:.0%} / BO={best_w[1]:.0%} / "
          f"TM={best_w[2]:.0%} / ED={best_w[3]:.0%}")
    print(f"  avg r = {best_r:+.4f}")

    # Also hit24-only
    def neg_corr(w, X, y):
        s = X @ w
        return -np.corrcoef(s, y)[0, 1] if np.std(s) > 1e-10 else 0
    res_h = minimize(neg_corr, starts[0], args=(X, y_hit),
                     method='SLSQP', bounds=bounds_4, constraints=constraints,
                     options={'maxiter': 1000})
    print(f"  Hit24-only: DC={res_h.x[0]:.0%}/BO={res_h.x[1]:.0%}/"
          f"TM={res_h.x[2]:.0%}/ED={res_h.x[3]:.0%} → r = {-res_h.fun:+.4f}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n\n{'=' * 100}")
print("FINAL SUMMARY")
print("=" * 100)

print(f"\n--- HEAD-TO-HEAD RANKING (partial r controlling for DC) ---")
print(f"{'Rank':>4} {'Version':<30} {'r(hit24)':>10} {'r(ppr)':>10} {'avg':>8}")
print("-" * 65)
for i, r in enumerate(rankings):
    marker = " ***" if i == 0 else " **" if i == 1 else ""
    print(f"  {i+1:>2}  {r['label']:<28} {r['r_hit']:>+10.4f} {r['r_ppr']:>+10.4f} {r['avg']:>+8.4f}{marker}")

print(f"\n--- BREAKOUT ENHANCEMENT COMPARISON ---")
for bo_label, bo_col in bo_versions.items():
    r_h, p_h, _ = partial_correlation(wr_eval[bo_col], wr_eval['hit24'], [wr_eval['s_dc']])
    r_p, p_p, _ = partial_correlation(wr_eval[bo_col], wr_eval['best_ppr'], [wr_eval['s_dc']])
    avg = (r_h + r_p) / 2
    print(f"  {bo_label:<35} avg partial r = {avg:+.4f}")

print()
