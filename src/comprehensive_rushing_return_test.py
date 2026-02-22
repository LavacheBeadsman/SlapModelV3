"""
Comprehensive Rushing Share & Return Test — All 4 Outcomes
==========================================================
Tests 8 versions (A-H) against hit24, hit12, first_3yr_ppg, career_ppg.
Also: audit of previous optimizer objective, and rerun with 4-outcome equal weighting.
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
# LOAD & MERGE ALL DATA
# ============================================================================

print("=" * 110)
print("COMPREHENSIVE RUSHING SHARE & RETURN TEST — ALL 4 OUTCOMES")
print("=" * 110)

wr = pd.read_csv('data/wr_backtest_all_components.csv')
ret = pd.read_csv('data/wr_return_stats.csv')
team_rush = pd.read_csv('data/wr_team_rushing.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
tm = pd.read_csv('data/wr_teammate_scores.csv')

# Merge return data
ret_cols = ['player_name', 'draft_year', 'kr_no', 'kr_yds', 'kr_td',
            'pr_no', 'pr_yds', 'pr_td', 'total_returns', 'total_return_yds',
            'total_return_td']
wr = wr.merge(ret[ret_cols], on=['player_name', 'draft_year'], how='left')

# Merge team rushing data
tr_cols = ['player_name', 'draft_year', 'team_rush_yds', 'team_pass_yds', 'team_total_yds']
wr = wr.merge(team_rush[tr_cols], on=['player_name', 'draft_year'], how='left')

# Merge outcomes (first_3yr_ppg, career_ppg)
out_wr = outcomes[outcomes['position'] == 'WR'][
    ['player_name', 'draft_year', 'first_3yr_ppg', 'career_ppg']
].copy()
wr = wr.merge(out_wr, on=['player_name', 'draft_year'], how='left')

# Fill NaN return/team data
for c in ['kr_no', 'kr_yds', 'kr_td', 'pr_no', 'pr_yds', 'pr_td',
          'total_returns', 'total_return_yds', 'total_return_td']:
    wr[c] = wr[c].fillna(0)

# Merge teammate
tm_lookup = dict(zip(zip(tm['player_name'], tm['draft_year']), tm['avg_teammate_dc']))
wr['teammate_dc'] = wr.apply(lambda x: tm_lookup.get((x['player_name'], x['draft_year']), 0), axis=1)

# Compute scores
wr['s_dc'] = wr['pick'].apply(dc_score)
wr['s_breakout'] = wr.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)

print(f"Loaded: {len(wr)} WRs")

# Eval sample
wr_eval = wr[wr['hit24'].notna() & wr['draft_year'].between(2015, 2024)].copy()

# Check outcome coverage
all_outcomes = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
print(f"\nOutcome coverage (eval sample = {len(wr_eval)}):")
for out in all_outcomes:
    n = wr_eval[out].notna().sum()
    print(f"  {out:<15}: {n} WRs ({n/len(wr_eval)*100:.1f}%)")


# ============================================================================
# PART 0: AUDIT OF PREVIOUS OPTIMIZER OBJECTIVE
# ============================================================================

print(f"\n\n{'=' * 110}")
print("PART 0: AUDIT OF PREVIOUS OPTIMIZER OBJECTIVE FUNCTION")
print("=" * 110)

print("""
The previous optimizer (in optimize_wr_6_component.py and all subsequent scripts)
used this objective function:

    def neg_avg_correlation(weights, X_hit, y_hit, X_ppr, y_ppr):
        s1 = X_hit @ weights
        r1 = corrcoef(s1, y_hit)  # correlation with hit24
        s2 = X_ppr @ weights
        r2 = corrcoef(s2, y_ppr)  # correlation with best_ppr
        return -(r1 + r2) / 2

WHAT THIS MEANS:
  - It averaged correlations with ONLY TWO outcomes: hit24 and best_ppr
  - hit12 was NOT included
  - first_3yr_ppg was NOT included
  - career_ppg was NOT included

  This is a 50/50 weighting of:
    50% → hit24 (binary: top-24 WR finish)
    50% → best_ppr (continuous: best single-season PPR total)

  The other two outcomes (first_3yr_ppg, career_ppg) were never part of
  the optimization. The optimizer may have found different weights if all
  four outcomes were included.
""")


# ============================================================================
# BUILD ALL 8 VERSIONS
# ============================================================================

print(f"{'=' * 110}")
print("BUILDING ALL 8 VERSIONS")
print("=" * 110)

# Rush share (need team rushing data)
wr['rush_share'] = np.where(
    wr['team_rush_yds'].notna() & (wr['team_rush_yds'] > 0),
    wr['rush_yards'].clip(lower=0) / wr['team_rush_yds'] * 100,
    np.nan
)

# Receiving yards — use the 'yards' column if it exists, else approximate
# Check what receiving columns we have
if 'yards' not in wr.columns:
    # We don't have WR receiving yards in this file directly
    # Use peak_dominator as a proxy or load from another source
    # Actually we should check — dominator is (rec yds / team rec yds + rec td / team td) / 2
    # For now, flag this limitation
    wr['rec_yards'] = np.nan
    print("  NOTE: WR receiving yards not in main file. Version C and H will be limited.")
else:
    wr['rec_yards'] = wr['yards']

# A: Rush share continuous (WR rush / team rush, normalized 0-100)
p99_rs = wr['rush_share'].quantile(0.99)
wr['v_A'] = (wr['rush_share'] / p99_rs * 100).clip(0, 100) if p99_rs > 0 else 0
wr['v_A'] = wr['v_A'].fillna(0)

# B: Rush share binary — test multiple thresholds
for pct in [1, 2, 3, 5]:
    col = f'v_B_{pct}pct'
    wr[col] = np.where(
        wr['rush_share'].notna() & (wr['rush_share'] >= pct), 1, 0
    )

# C: Offensive touch share — skip if no receiving data
# We can compute this differently: use rush_yards / team_total_yds as a proxy
wr['v_C'] = np.where(
    wr['team_total_yds'].notna() & (wr['team_total_yds'] > 0),
    wr['rush_yards'].clip(lower=0) / wr['team_total_yds'] * 100,
    np.nan
)
p99_c = wr['v_C'].quantile(0.99)
wr['v_C_norm'] = (wr['v_C'] / p99_c * 100).clip(0, 100) if p99_c > 0 else 0
wr['v_C_norm'] = wr['v_C_norm'].fillna(0)

# D: Rush dominator (same as rush_share but kept as raw %)
wr['v_D'] = wr['rush_share'].fillna(0)

# E: Return yards continuous
p99_ret = wr['total_return_yds'].quantile(0.99)
wr['v_E'] = (wr['total_return_yds'] / p99_ret * 100).clip(0, 100) if p99_ret > 0 else 0

# F: Return TD flag
wr['v_F'] = (wr['total_return_td'] > 0).astype(int)

# G: Return volume (total attempts, normalized)
p99_retno = wr['total_returns'].quantile(0.99)
wr['v_G'] = (wr['total_returns'] / p99_retno * 100).clip(0, 100) if p99_retno > 0 else 0

# H: All-purpose share = (rush + return) / team_total
wr['all_purpose'] = wr['rush_yards'].clip(lower=0) + wr['total_return_yds']
wr['v_H'] = np.where(
    wr['team_total_yds'].notna() & (wr['team_total_yds'] > 0),
    wr['all_purpose'] / wr['team_total_yds'] * 100,
    np.nan
)
p99_h = wr['v_H'].quantile(0.99)
wr['v_H_norm'] = (wr['v_H'] / p99_h * 100).clip(0, 100) if p99_h > 0 else 0
wr['v_H_norm'] = wr['v_H_norm'].fillna(0)

# Refresh eval
wr_eval = wr[wr['hit24'].notna() & wr['draft_year'].between(2015, 2024)].copy()

# Define all versions for testing
versions = {
    'A: Rush share (cont)':     ('v_A', 'continuous'),
    'B: Rush share >1%':        ('v_B_1pct', 'binary'),
    'B: Rush share >2%':        ('v_B_2pct', 'binary'),
    'B: Rush share >3%':        ('v_B_3pct', 'binary'),
    'B: Rush share >5%':        ('v_B_5pct', 'binary'),
    'C: Rush/total yds (cont)': ('v_C_norm', 'continuous'),
    'D: Rush dominator %':      ('v_D', 'continuous'),
    'E: Return yds (cont)':     ('v_E', 'continuous'),
    'F: Return TD flag':        ('v_F', 'binary'),
    'G: Return volume (cont)':  ('v_G', 'continuous'),
    'H: All-purpose share':     ('v_H_norm', 'continuous'),
}

# Qualifier counts
print(f"\n{'Version':<30} {'Type':>10} {'Qual/Q1':>10} {'Other':>10}")
print("-" * 65)
for label, (col, vtype) in versions.items():
    if vtype == 'binary':
        n1 = (wr_eval[col] == 1).sum()
        n0 = (wr_eval[col] == 0).sum()
        print(f"  {label:<28} {'binary':>10} {n1:>10} {n0:>10}")
    else:
        q75 = wr_eval[col].quantile(0.75)
        nhi = (wr_eval[col] > q75).sum()
        nlo = (wr_eval[col] <= q75).sum()
        print(f"  {label:<28} {'continuous':>10} {nhi:>10} {nlo:>10}  (Q1 cutoff={q75:.2f})")


# ============================================================================
# PARTIAL CORRELATIONS: CONTROLLING FOR DC ONLY
# ============================================================================

print(f"\n\n{'=' * 110}")
print("PARTIAL CORRELATIONS — CONTROLLING FOR DC ONLY")
print("=" * 110)

print(f"\n{'Version':<30}", end="")
for out in all_outcomes:
    print(f"  {'r('+out+')':>9} {'p':>7}", end="")
print()
print("-" * 110)

partial_dc = {}
for label, (col, vtype) in versions.items():
    row = f"  {label:<28}"
    pr = {}
    for out in all_outcomes:
        r, p, n = partial_correlation(wr_eval[col], wr_eval[out], [wr_eval['s_dc']])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        row += f"  {r:>+.4f}{sig:<3} {p:>7.4f}"
        pr[out] = (r, p, n)
    print(row)
    partial_dc[label] = pr

# Baselines
print(f"\n  {'--- BASELINES ---':<28}")
for bl, bvals in [('Rushing only (20+ yds)', (wr_eval['rush_yards'] >= 20).astype(int)),
                   ('Plain breakout score', wr_eval['s_breakout'])]:
    row = f"  {bl:<28}"
    for out in all_outcomes:
        r, p, n = partial_correlation(bvals, wr_eval[out], [wr_eval['s_dc']])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        row += f"  {r:>+.4f}{sig:<3} {p:>7.4f}"
    print(row)


# ============================================================================
# PARTIAL CORRELATIONS: CONTROLLING FOR DC + BREAKOUT
# ============================================================================

print(f"\n\n{'=' * 110}")
print("PARTIAL CORRELATIONS — CONTROLLING FOR DC AND BREAKOUT AGE")
print("=" * 110)

print(f"\n{'Version':<30}", end="")
for out in all_outcomes:
    print(f"  {'r('+out+')':>9} {'p':>7}", end="")
print()
print("-" * 110)

partial_dc_bo = {}
for label, (col, vtype) in versions.items():
    row = f"  {label:<28}"
    pr = {}
    for out in all_outcomes:
        r, p, n = partial_correlation(wr_eval[col], wr_eval[out],
                                       [wr_eval['s_dc'], wr_eval['s_breakout']])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        row += f"  {r:>+.4f}{sig:<3} {p:>7.4f}"
        pr[out] = (r, p, n)
    print(row)
    partial_dc_bo[label] = pr


# ============================================================================
# AVERAGE OUTCOMES: QUALIFIERS VS NON-QUALIFIERS
# ============================================================================

print(f"\n\n{'=' * 110}")
print("AVERAGE OUTCOMES: QUALIFIERS vs NON-QUALIFIERS")
print("=" * 110)
print("Binary: qualifier vs non-qualifier | Continuous: top quartile vs bottom 3 quartiles")

for label, (col, vtype) in versions.items():
    print(f"\n  {label}:")

    if vtype == 'binary':
        q = wr_eval[wr_eval[col] == 1]
        nq = wr_eval[wr_eval[col] == 0]
        q_label, nq_label = "Qual", "Non-Q"
    else:
        q75 = wr_eval[col].quantile(0.75)
        q = wr_eval[wr_eval[col] > q75]
        nq = wr_eval[wr_eval[col] <= q75]
        q_label, nq_label = "Top Q", "Bot 3Q"

    print(f"    {'Outcome':<15} {q_label+' (N='+str(len(q))+')':>18} {nq_label+' (N='+str(len(nq))+')':>18} {'Diff':>10}")
    print(f"    {'-'*65}")
    for out in all_outcomes:
        q_mean = q[out].mean()
        nq_mean = nq[out].mean()
        diff = q_mean - nq_mean if pd.notna(q_mean) and pd.notna(nq_mean) else np.nan
        q_str = f"{q_mean:.3f}" if pd.notna(q_mean) else "N/A"
        nq_str = f"{nq_mean:.3f}" if pd.notna(nq_mean) else "N/A"
        d_str = f"{diff:+.3f}" if pd.notna(diff) else "N/A"
        print(f"    {out:<15} {q_str:>18} {nq_str:>18} {d_str:>10}")


# ============================================================================
# HEAD-TO-HEAD RANKING
# ============================================================================

print(f"\n\n{'=' * 110}")
print("HEAD-TO-HEAD RANKING — ALL VERSIONS")
print("=" * 110)
print("Ranked by average partial r across ALL 4 outcomes (controlling for DC)")

rankings = []
for label in versions:
    avg_r = np.mean([partial_dc[label][out][0] for out in all_outcomes
                      if not np.isnan(partial_dc[label][out][0])])
    n_sig = sum(1 for out in all_outcomes
                if not np.isnan(partial_dc[label][out][1]) and partial_dc[label][out][1] < 0.05)
    rankings.append({
        'label': label,
        'r_hit24': partial_dc[label]['hit24'][0],
        'r_hit12': partial_dc[label]['hit12'][0],
        'r_3yr': partial_dc[label]['first_3yr_ppg'][0],
        'r_career': partial_dc[label]['career_ppg'][0],
        'avg_r': avg_r,
        'n_sig': n_sig,
    })

rankings.sort(key=lambda x: x['avg_r'], reverse=True)

print(f"\n{'Rank':>4} {'Version':<30} {'hit24':>8} {'hit12':>8} {'3yr_ppg':>8} {'car_ppg':>8} {'avg_r':>8} {'sig':>4}")
print("-" * 90)
for i, r in enumerate(rankings):
    marker = " ***" if i == 0 else " **" if i == 1 else ""
    print(f"  {i+1:>2}  {r['label']:<28} {r['r_hit24']:>+.4f} {r['r_hit12']:>+.4f} "
          f"{r['r_3yr']:>+.4f} {r['r_career']:>+.4f} {r['avg_r']:>+.4f} {r['n_sig']:>4}{marker}")


# ============================================================================
# TOP 20 BY RUSH SHARE
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TOP 20 WRs BY RUSH SHARE (WR rush yards / team rush yards)")
print("=" * 110)

wr_with_rs = wr_eval[wr_eval['rush_share'].notna()].copy()
top_rs = wr_with_rs.nlargest(20, 'rush_share')

print(f"\n{'Name':<25} {'Year':>5} {'Pick':>5} {'Rd':>3} {'College':<16} {'RshYd':>6} {'TmRsh':>6} "
      f"{'Share%':>7} {'Hit24':>6} {'Hit12':>6} {'3yr':>7} {'Career':>7}")
print("-" * 120)
for _, p in top_rs.iterrows():
    h24 = f"{p['hit24']:.0f}" if pd.notna(p['hit24']) else "—"
    h12 = f"{p['hit12']:.0f}" if pd.notna(p['hit12']) else "—"
    ppg3 = f"{p['first_3yr_ppg']:.1f}" if pd.notna(p['first_3yr_ppg']) else "—"
    ppgc = f"{p['career_ppg']:.1f}" if pd.notna(p['career_ppg']) else "—"
    print(f"  {p['player_name']:<23} {p['draft_year']:>5} {p['pick']:>5} {p['round']:>3} "
          f"{p['college']:<16} {p['rush_yards']:>6.0f} {p['team_rush_yds']:>6.0f} "
          f"{p['rush_share']:>7.2f} {h24:>6} {h12:>6} {ppg3:>7} {ppgc:>7}")

# Hit rates
hits = top_rs['hit24'].sum()
n_ev = top_rs['hit24'].notna().sum()
print(f"\n  Top 20 rush share: {hits:.0f}/{n_ev} hit24 ({hits/n_ev*100:.1f}%)" if n_ev > 0 else "")
print(f"  Avg first_3yr_ppg: {top_rs['first_3yr_ppg'].mean():.2f}")
print(f"  Avg career_ppg: {top_rs['career_ppg'].mean():.2f}")


# ============================================================================
# TOP 20 BY RETURN PRODUCTION
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TOP 20 WRs BY RETURN PRODUCTION (total return yards)")
print("=" * 110)

top_ret = wr_eval.nlargest(20, 'total_return_yds')

print(f"\n{'Name':<25} {'Year':>5} {'Pick':>5} {'Rd':>3} {'College':<16} {'KRYd':>5} {'PRYd':>5} "
      f"{'TotYd':>6} {'RetTD':>5} {'Hit24':>6} {'Hit12':>6} {'3yr':>7} {'Career':>7}")
print("-" * 125)
for _, p in top_ret.iterrows():
    h24 = f"{p['hit24']:.0f}" if pd.notna(p['hit24']) else "—"
    h12 = f"{p['hit12']:.0f}" if pd.notna(p['hit12']) else "—"
    ppg3 = f"{p['first_3yr_ppg']:.1f}" if pd.notna(p['first_3yr_ppg']) else "—"
    ppgc = f"{p['career_ppg']:.1f}" if pd.notna(p['career_ppg']) else "—"
    print(f"  {p['player_name']:<23} {p['draft_year']:>5} {p['pick']:>5} {p['round']:>3} "
          f"{p['college']:<16} {p['kr_yds']:>5.0f} {p['pr_yds']:>5.0f} "
          f"{p['total_return_yds']:>6.0f} {p['total_return_td']:>5.0f} "
          f"{h24:>6} {h12:>6} {ppg3:>7} {ppgc:>7}")

hits_r = top_ret['hit24'].sum()
n_ev_r = top_ret['hit24'].notna().sum()
print(f"\n  Top 20 return yards: {hits_r:.0f}/{n_ev_r} hit24 ({hits_r/n_ev_r*100:.1f}%)" if n_ev_r > 0 else "")
print(f"  Avg first_3yr_ppg: {top_ret['first_3yr_ppg'].mean():.2f}")
print(f"  Avg career_ppg: {top_ret['career_ppg'].mean():.2f}")


# ============================================================================
# 4-COMPONENT OPTIMIZATION WITH 4 OUTCOMES EQUALLY WEIGHTED
# ============================================================================

print(f"\n\n{'=' * 110}")
print("4-COMPONENT OPTIMIZATION — ALL 4 OUTCOMES EQUALLY WEIGHTED")
print("=" * 110)
print("Components: DC / Breakout / Teammate / Early Declare")
print("Objective: average of Pearson r across hit24, hit12, first_3yr_ppg, career_ppg")

wr_eval['s_teammate'] = wr_eval['teammate_dc'].clip(0, 100)
wr_eval['s_declare'] = wr_eval['early_declare'] * 100

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
    print(f"  {out}: {len(valid)} WRs with data")

def neg_4outcome_avg(weights):
    """Average correlation across all 4 outcomes (equal weight)."""
    total_r = 0
    count = 0
    for out in all_outcomes:
        X = outcome_data[out]['X']
        y = outcome_data[out]['y']
        s = X @ weights
        if np.std(s) > 1e-10:
            r = np.corrcoef(s, y)[0, 1]
            total_r += r
            count += 1
    return -(total_r / count) if count > 0 else 0

def neg_2outcome_avg(weights):
    """Original: average of hit24 + best_ppr only."""
    total_r = 0
    count = 0
    for out in ['hit24', 'first_3yr_ppg']:  # Use first_3yr_ppg as proxy for best_ppr
        if out in outcome_data:
            X = outcome_data[out]['X']
            y = outcome_data[out]['y']
            s = X @ weights
            if np.std(s) > 1e-10:
                r = np.corrcoef(s, y)[0, 1]
                total_r += r
                count += 1
    return -(total_r / count) if count > 0 else 0

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
]

# Also need best_ppr for the original 2-outcome comparison
valid_ppr = wr_eval[score_cols + ['best_ppr']].dropna(subset=['best_ppr']).copy()
outcome_data['best_ppr'] = {
    'X': valid_ppr[score_cols].values,
    'y': valid_ppr['best_ppr'].values,
    'n': len(valid_ppr),
}

def neg_original_2outcome(weights):
    """Exact original objective: avg(r_hit24, r_best_ppr)."""
    rs = []
    for out in ['hit24', 'best_ppr']:
        X = outcome_data[out]['X']
        y = outcome_data[out]['y']
        s = X @ weights
        if np.std(s) > 1e-10:
            rs.append(np.corrcoef(s, y)[0, 1])
    return -(sum(rs) / len(rs)) if rs else 0


# --- Original 2-outcome optimizer ---
print(f"\n{'─' * 80}")
print("A) ORIGINAL OPTIMIZER: avg(r_hit24, r_best_ppr)")
print(f"{'─' * 80}")

best_r_orig = -999
best_w_orig = None
for x0 in starts:
    res = minimize(neg_original_2outcome, x0,
                   method='SLSQP', bounds=bounds_4, constraints=constraints,
                   options={'maxiter': 1000})
    r_opt = -res.fun
    if r_opt > best_r_orig:
        best_r_orig = r_opt
        best_w_orig = res.x

print(f"  Weights: DC={best_w_orig[0]:.0%} / BO={best_w_orig[1]:.0%} / "
      f"TM={best_w_orig[2]:.0%} / ED={best_w_orig[3]:.0%}")
print(f"  avg r(hit24, best_ppr) = {best_r_orig:+.4f}")

# Show all 4 individual correlations at these weights
print(f"\n  Individual correlations at these weights:")
for out in all_outcomes + ['best_ppr']:
    X = outcome_data[out]['X']
    y = outcome_data[out]['y']
    s = X @ best_w_orig
    r = np.corrcoef(s, y)[0, 1]
    print(f"    {out:<15}: r = {r:+.4f}  (N={len(y)})")


# --- New 4-outcome optimizer ---
print(f"\n{'─' * 80}")
print("B) NEW OPTIMIZER: avg(r_hit24, r_hit12, r_first_3yr_ppg, r_career_ppg)")
print(f"{'─' * 80}")

best_r_new = -999
best_w_new = None
for x0 in starts:
    res = minimize(neg_4outcome_avg, x0,
                   method='SLSQP', bounds=bounds_4, constraints=constraints,
                   options={'maxiter': 1000})
    r_opt = -res.fun
    if r_opt > best_r_new:
        best_r_new = r_opt
        best_w_new = res.x

print(f"  Weights: DC={best_w_new[0]:.0%} / BO={best_w_new[1]:.0%} / "
      f"TM={best_w_new[2]:.0%} / ED={best_w_new[3]:.0%}")
print(f"  avg r(all 4) = {best_r_new:+.4f}")

print(f"\n  Individual correlations at these weights:")
for out in all_outcomes + ['best_ppr']:
    X = outcome_data[out]['X']
    y = outcome_data[out]['y']
    s = X @ best_w_new
    r = np.corrcoef(s, y)[0, 1]
    print(f"    {out:<15}: r = {r:+.4f}  (N={len(y)})")


# --- Comparison ---
print(f"\n{'─' * 80}")
print("COMPARISON: Do weights change with 4-outcome objective?")
print(f"{'─' * 80}")

print(f"\n{'Component':<20} {'2-outcome':>12} {'4-outcome':>12} {'Diff':>10}")
print("-" * 55)
for i, lab in enumerate(labels_4):
    diff = best_w_new[i] - best_w_orig[i]
    print(f"  {lab:<18} {best_w_orig[i]:>12.0%} {best_w_new[i]:>12.0%} {diff:>+10.1%}")

# Cross-evaluate
print(f"\n  Cross-evaluation:")
print(f"  {'Objective':<45} {'2-out wts':>12} {'4-out wts':>12}")
print(f"  {'-'*70}")

# 2-outcome metric at both sets of weights
for wlabel, w in [('2-outcome weights', best_w_orig), ('4-outcome weights', best_w_new)]:
    rs_2 = []
    for out in ['hit24', 'best_ppr']:
        X = outcome_data[out]['X']
        y = outcome_data[out]['y']
        s = X @ w
        rs_2.append(np.corrcoef(s, y)[0, 1])
    avg_2 = np.mean(rs_2)

    rs_4 = []
    for out in all_outcomes:
        X = outcome_data[out]['X']
        y = outcome_data[out]['y']
        s = X @ w
        rs_4.append(np.corrcoef(s, y)[0, 1])
    avg_4 = np.mean(rs_4)

    print(f"  avg r(hit24+ppr) at {wlabel:<25}: {avg_2:+.4f}")
    print(f"  avg r(all 4)     at {wlabel:<25}: {avg_4:+.4f}")
    print()


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'=' * 110}")
print("FINAL SUMMARY")
print("=" * 110)

print(f"""
1. PREVIOUS OPTIMIZER OBJECTIVE:
   - Only used 2 outcomes: hit24 + best_ppr (equal weight)
   - Did NOT include hit12, first_3yr_ppg, or career_ppg

2. HEAD-TO-HEAD (top 3 by avg partial r across all 4 outcomes):""")

for i in range(min(3, len(rankings))):
    r = rankings[i]
    print(f"   #{i+1}: {r['label']:<28} avg_r = {r['avg_r']:+.4f} ({r['n_sig']} significant)")

print(f"""
3. OPTIMIZER COMPARISON:
   Original (2-outcome): DC={best_w_orig[0]:.0%}/BO={best_w_orig[1]:.0%}/TM={best_w_orig[2]:.0%}/ED={best_w_orig[3]:.0%}
   New (4-outcome):      DC={best_w_new[0]:.0%}/BO={best_w_new[1]:.0%}/TM={best_w_new[2]:.0%}/ED={best_w_new[3]:.0%}
""")
