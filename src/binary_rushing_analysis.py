"""
Binary Rushing Analysis — Two Thresholds
=========================================
Version A: 1 if rush_yards >= 20, else 0
Version B: 1 if rush_attempts >= 5, else 0

For each version:
1. Qualifier counts
2. Hit24 rates (qualifiers vs non-qualifiers)
3. Partial correlations controlling for DC
4. Full qualifier list for gut-check

Then: re-run 6-component weight optimization with the better binary flag.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SCORING FUNCTIONS (same as optimize script)
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
# LOAD DATA
# ============================================================================

print("=" * 100)
print("BINARY RUSHING ANALYSIS — TWO THRESHOLD VERSIONS")
print("=" * 100)

wr = pd.read_csv('data/wr_backtest_all_components.csv')
tm = pd.read_csv('data/wr_teammate_scores.csv')
gp = pd.read_csv('data/wr_games_played.csv')

print(f"\nLoaded: {len(wr)} WRs")

# Check rush_attempts coverage
n_att = wr['rush_attempts'].notna().sum()
n_att_null = wr['rush_attempts'].isna().sum()
n_yds = wr['rush_yards'].notna().sum()
n_yds_null = wr['rush_yards'].isna().sum()

print(f"\n{'=' * 100}")
print("DATA AVAILABILITY CHECK")
print("=" * 100)
print(f"  rush_yards:    {n_yds} have data, {n_yds_null} null")
print(f"  rush_attempts: {n_att} have data, {n_att_null} null")

if n_att_null > 0:
    missing_att = wr[wr['rush_attempts'].isna()][['player_name', 'draft_year', 'pick', 'college', 'rush_yards', 'rush_attempts', 'rush_source']]
    print(f"\n  Players MISSING rush_attempts:")
    for _, p in missing_att.iterrows():
        print(f"    {p['player_name']} ({p['draft_year']}) — pick {p['pick']}, {p['college']} — "
              f"rush_yards={p['rush_yards']}, source={p['rush_source']}")
    print(f"\n  ** Version B (5+ attempts) will have {n_att_null} players with unknown status **")

if n_yds_null > 0:
    missing_yds = wr[wr['rush_yards'].isna()][['player_name', 'draft_year', 'pick', 'college', 'rush_yards', 'rush_source']]
    print(f"\n  Players MISSING rush_yards:")
    for _, p in missing_yds.iterrows():
        print(f"    {p['player_name']} ({p['draft_year']}) — pick {p['pick']}, {p['college']}")


# ============================================================================
# BUILD BINARY FLAGS
# ============================================================================

print(f"\n\n{'=' * 100}")
print("BUILDING BINARY FLAGS")
print("=" * 100)

# Version A: 20+ rushing yards
wr['rush_binary_A'] = np.where(wr['rush_yards'] >= 20, 1, 0)
n_a_1 = (wr['rush_binary_A'] == 1).sum()
n_a_0 = (wr['rush_binary_A'] == 0).sum()
print(f"\nVersion A (20+ rushing yards):")
print(f"  Qualifiers (1): {n_a_1} WRs ({n_a_1/len(wr)*100:.1f}%)")
print(f"  Non-qualifiers (0): {n_a_0} WRs ({n_a_0/len(wr)*100:.1f}%)")

# Version B: 5+ rush attempts
wr['rush_binary_B'] = np.where(wr['rush_attempts'] >= 5, 1, 0)
# Handle NaN attempts — if rush_yards=0 and attempts=NaN, safe to assume 0 attempts
wr.loc[wr['rush_attempts'].isna() & (wr['rush_yards'] == 0), 'rush_binary_B'] = 0
n_b_1 = (wr['rush_binary_B'] == 1).sum()
n_b_0 = (wr['rush_binary_B'] == 0).sum()
n_b_unk = wr['rush_binary_B'].isna().sum()
print(f"\nVersion B (5+ rush attempts):")
print(f"  Qualifiers (1): {n_b_1} WRs ({n_b_1/len(wr)*100:.1f}%)")
print(f"  Non-qualifiers (0): {n_b_0} WRs ({n_b_0/len(wr)*100:.1f}%)")
if n_b_unk > 0:
    print(f"  Unknown (NaN attempts): {n_b_unk} WRs")


# ============================================================================
# OUTCOME ANALYSIS — filter to 2015-2024 classes with outcomes
# ============================================================================

wr_eval = wr[wr['draft_year'] <= 2024].copy()
n_eval = len(wr_eval)
n_hit = wr_eval['hit24'].notna().sum()
print(f"\n\n{'=' * 100}")
print(f"OUTCOME ANALYSIS (2015-2024 classes, {n_eval} WRs with hit24 data)")
print("=" * 100)


# --- VERSION A: 20+ rushing yards ---
print(f"\n{'─' * 80}")
print("VERSION A: Binary flag = 1 if rush_yards >= 20")
print(f"{'─' * 80}")

eval_a = wr_eval[wr_eval['hit24'].notna()].copy()

a1 = eval_a[eval_a['rush_binary_A'] == 1]
a0 = eval_a[eval_a['rush_binary_A'] == 0]

hit_a1 = a1['hit24'].mean()
hit_a0 = a0['hit24'].mean()
print(f"\n  Hit24 rates:")
print(f"    Qualifiers (20+ yds):   {hit_a1:.3f}  ({a1['hit24'].sum():.0f}/{len(a1)} WRs)")
print(f"    Non-qualifiers (<20):   {hit_a0:.3f}  ({a0['hit24'].sum():.0f}/{len(a0)} WRs)")
print(f"    Difference:             {hit_a1 - hit_a0:+.3f}")

# Statistical test
ct_a = pd.crosstab(eval_a['rush_binary_A'], eval_a['hit24'])
if ct_a.shape == (2, 2):
    chi2_a, pchi_a, _, _ = stats.chi2_contingency(ct_a)
    print(f"    Chi-squared test: χ²={chi2_a:.3f}, p={pchi_a:.4f}")

# Raw correlations with all 4 outcomes
print(f"\n  Raw correlations:")
for outcome in ['hit24', 'hit12', 'best_ppr', 'best_rank']:
    if outcome in eval_a.columns:
        valid = eval_a[['rush_binary_A', outcome]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid['rush_binary_A'], valid[outcome])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            direction = "(higher → better)" if outcome in ['hit24', 'hit12', 'best_ppr'] else "(higher → worse rank = bad)"
            print(f"    vs {outcome:<10}: r = {r:+.3f} (p={p:.4f}) {sig} {direction}")

# Partial correlations controlling for draft capital
print(f"\n  Partial correlations (controlling for draft pick):")
eval_a['s_dc'] = eval_a['pick'].apply(dc_score)
for outcome in ['hit24', 'hit12', 'best_ppr', 'best_rank']:
    if outcome in eval_a.columns:
        r, p, n = partial_correlation(eval_a['rush_binary_A'], eval_a[outcome], [eval_a['s_dc']])
        if not np.isnan(r):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    vs {outcome:<10}: partial r = {r:+.3f} (p={p:.4f}) {sig}  [N={n}]")


# --- VERSION B: 5+ rush attempts ---
print(f"\n\n{'─' * 80}")
print("VERSION B: Binary flag = 1 if rush_attempts >= 5")
print(f"{'─' * 80}")

eval_b = wr_eval[wr_eval['hit24'].notna() & wr_eval['rush_binary_B'].notna()].copy()

b1 = eval_b[eval_b['rush_binary_B'] == 1]
b0 = eval_b[eval_b['rush_binary_B'] == 0]

hit_b1 = b1['hit24'].mean()
hit_b0 = b0['hit24'].mean()
print(f"\n  Hit24 rates:")
print(f"    Qualifiers (5+ att):    {hit_b1:.3f}  ({b1['hit24'].sum():.0f}/{len(b1)} WRs)")
print(f"    Non-qualifiers (<5):    {hit_b0:.3f}  ({b0['hit24'].sum():.0f}/{len(b0)} WRs)")
print(f"    Difference:             {hit_b1 - hit_b0:+.3f}")

ct_b = pd.crosstab(eval_b['rush_binary_B'], eval_b['hit24'])
if ct_b.shape == (2, 2):
    chi2_b, pchi_b, _, _ = stats.chi2_contingency(ct_b)
    print(f"    Chi-squared test: χ²={chi2_b:.3f}, p={pchi_b:.4f}")

# Raw correlations
print(f"\n  Raw correlations:")
for outcome in ['hit24', 'hit12', 'best_ppr', 'best_rank']:
    if outcome in eval_b.columns:
        valid = eval_b[['rush_binary_B', outcome]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid['rush_binary_B'], valid[outcome])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            direction = "(higher → better)" if outcome in ['hit24', 'hit12', 'best_ppr'] else "(higher → worse rank = bad)"
            print(f"    vs {outcome:<10}: r = {r:+.3f} (p={p:.4f}) {sig} {direction}")

# Partial correlations
print(f"\n  Partial correlations (controlling for draft pick):")
eval_b['s_dc'] = eval_b['pick'].apply(dc_score)
for outcome in ['hit24', 'hit12', 'best_ppr', 'best_rank']:
    if outcome in eval_b.columns:
        r, p, n = partial_correlation(eval_b['rush_binary_B'], eval_b[outcome], [eval_b['s_dc']])
        if not np.isnan(r):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    vs {outcome:<10}: partial r = {r:+.3f} (p={p:.4f}) {sig}  [N={n}]")


# ============================================================================
# HEAD-TO-HEAD COMPARISON
# ============================================================================

print(f"\n\n{'=' * 100}")
print("HEAD-TO-HEAD COMPARISON")
print("=" * 100)

print(f"\n{'Metric':<45} {'Version A':>12} {'Version B':>12}")
print(f"{'':45} {'(20+ yds)':>12} {'(5+ att)':>12}")
print("-" * 70)
print(f"{'Threshold':45} {'20+ yards':>12} {'5+ attempts':>12}")
print(f"{'Qualifiers':45} {n_a_1:>12} {n_b_1:>12}")
print(f"{'Hit24 rate (qualifiers)':45} {hit_a1:>12.3f} {hit_b1:>12.3f}")
print(f"{'Hit24 rate (non-qualifiers)':45} {hit_a0:>12.3f} {hit_b0:>12.3f}")
print(f"{'Difference':45} {hit_a1-hit_a0:>+12.3f} {hit_b1-hit_b0:>+12.3f}")

# Partial r vs hit24 for head-to-head
r_a_h, p_a_h, _ = partial_correlation(eval_a['rush_binary_A'], eval_a['hit24'], [eval_a['s_dc']])
r_b_h, p_b_h, _ = partial_correlation(eval_b['rush_binary_B'], eval_b['hit24'], [eval_b['s_dc']])
print(f"{'Partial r vs hit24 (ctrl DC)':45} {r_a_h:>+12.3f} {r_b_h:>+12.3f}")
print(f"{'p-value':45} {p_a_h:>12.4f} {p_b_h:>12.4f}")

# Partial r vs best_ppr
r_a_p, p_a_p, _ = partial_correlation(eval_a['rush_binary_A'], eval_a['best_ppr'], [eval_a['s_dc']])
r_b_p, p_b_p, _ = partial_correlation(eval_b['rush_binary_B'], eval_b['best_ppr'], [eval_b['s_dc']])
print(f"{'Partial r vs best_ppr (ctrl DC)':45} {r_a_p:>+12.3f} {r_b_p:>+12.3f}")
print(f"{'p-value':45} {p_a_p:>12.4f} {p_b_p:>12.4f}")


# ============================================================================
# QUALIFIER LISTS (for gut-check)
# ============================================================================

print(f"\n\n{'=' * 100}")
print("VERSION A QUALIFIERS: WRs with 20+ rushing yards (full list)")
print("=" * 100)

qual_a = wr[wr['rush_binary_A'] == 1].sort_values(['draft_year', 'pick'])[
    ['player_name', 'draft_year', 'pick', 'round', 'college', 'rush_yards', 'rush_attempts', 'hit24', 'best_ppr']
].copy()

print(f"\n{'Name':<25} {'Year':>5} {'Pick':>5} {'Rd':>3} {'College':<18} {'RshYd':>6} {'RshAtt':>6} {'Hit24':>6} {'PPR':>7}")
print("-" * 95)

for _, p in qual_a.iterrows():
    hit = f"{p['hit24']:.0f}" if pd.notna(p['hit24']) else "—"
    ppr = f"{p['best_ppr']:.1f}" if pd.notna(p['best_ppr']) else "—"
    att = f"{p['rush_attempts']:.0f}" if pd.notna(p['rush_attempts']) else "?"
    print(f"  {p['player_name']:<23} {p['draft_year']:>5} {p['pick']:>5} {p['round']:>3} "
          f"{p['college']:<18} {p['rush_yards']:>6.0f} {att:>6} {hit:>6} {ppr:>7}")

print(f"\n  Total qualifiers: {len(qual_a)}")
n_hit_a = qual_a['hit24'].sum() if qual_a['hit24'].notna().any() else 0
n_eval_a = qual_a['hit24'].notna().sum()
print(f"  With outcomes: {n_eval_a}, Hits: {n_hit_a:.0f} ({n_hit_a/n_eval_a*100:.1f}% hit rate)" if n_eval_a > 0 else "")


print(f"\n\n{'=' * 100}")
print("VERSION B QUALIFIERS: WRs with 5+ rush attempts (full list)")
print("=" * 100)

qual_b = wr[wr['rush_binary_B'] == 1].sort_values(['draft_year', 'pick'])[
    ['player_name', 'draft_year', 'pick', 'round', 'college', 'rush_yards', 'rush_attempts', 'hit24', 'best_ppr']
].copy()

print(f"\n{'Name':<25} {'Year':>5} {'Pick':>5} {'Rd':>3} {'College':<18} {'RshYd':>6} {'RshAtt':>6} {'Hit24':>6} {'PPR':>7}")
print("-" * 95)

for _, p in qual_b.iterrows():
    hit = f"{p['hit24']:.0f}" if pd.notna(p['hit24']) else "—"
    ppr = f"{p['best_ppr']:.1f}" if pd.notna(p['best_ppr']) else "—"
    att = f"{p['rush_attempts']:.0f}" if pd.notna(p['rush_attempts']) else "?"
    print(f"  {p['player_name']:<23} {p['draft_year']:>5} {p['pick']:>5} {p['round']:>3} "
          f"{p['college']:<18} {p['rush_yards']:>6.0f} {att:>6} {hit:>6} {ppr:>7}")

print(f"\n  Total qualifiers: {len(qual_b)}")
n_hit_b = qual_b['hit24'].sum() if qual_b['hit24'].notna().any() else 0
n_eval_b = qual_b['hit24'].notna().sum()
print(f"  With outcomes: {n_eval_b}, Hits: {n_hit_b:.0f} ({n_hit_b/n_eval_b*100:.1f}% hit rate)" if n_eval_b > 0 else "")


# Show players who differ between A and B
print(f"\n\n{'=' * 100}")
print("PLAYERS WHO DIFFER BETWEEN VERSION A AND VERSION B")
print("=" * 100)

a_only = wr[(wr['rush_binary_A'] == 1) & (wr['rush_binary_B'] == 0)][
    ['player_name', 'draft_year', 'pick', 'college', 'rush_yards', 'rush_attempts', 'hit24']
].sort_values(['draft_year', 'pick'])
b_only = wr[(wr['rush_binary_A'] == 0) & (wr['rush_binary_B'] == 1)][
    ['player_name', 'draft_year', 'pick', 'college', 'rush_yards', 'rush_attempts', 'hit24']
].sort_values(['draft_year', 'pick'])

print(f"\nQualify under A (20+ yds) but NOT B (5+ att): {len(a_only)} WRs")
if len(a_only) > 0:
    for _, p in a_only.iterrows():
        hit = f"hit24={p['hit24']:.0f}" if pd.notna(p['hit24']) else "no outcome"
        att = f"{p['rush_attempts']:.0f}" if pd.notna(p['rush_attempts']) else "?"
        print(f"  {p['player_name']:<25} ({p['draft_year']}) pick {p['pick']:>3} — "
              f"{p['rush_yards']:.0f} yds on {att} att — {hit}")

print(f"\nQualify under B (5+ att) but NOT A (20+ yds): {len(b_only)} WRs")
if len(b_only) > 0:
    for _, p in b_only.iterrows():
        hit = f"hit24={p['hit24']:.0f}" if pd.notna(p['hit24']) else "no outcome"
        att = f"{p['rush_attempts']:.0f}" if pd.notna(p['rush_attempts']) else "?"
        print(f"  {p['player_name']:<25} ({p['draft_year']}) pick {p['pick']:>3} — "
              f"{p['rush_yards']:.0f} yds on {att} att — {hit}")


# ============================================================================
# DETERMINE WINNER
# ============================================================================

print(f"\n\n{'=' * 100}")
print("VERDICT: WHICH BINARY VERSION IS BETTER?")
print("=" * 100)

# Use partial r vs hit24 as primary metric
a_score = abs(r_a_h) if not np.isnan(r_a_h) else 0
b_score = abs(r_b_h) if not np.isnan(r_b_h) else 0

if a_score > b_score:
    winner = 'A'
    print(f"\n  Version A (20+ yards) has stronger partial correlation with hit24")
    print(f"    A: partial r = {r_a_h:+.3f} (p={p_a_h:.4f})")
    print(f"    B: partial r = {r_b_h:+.3f} (p={p_b_h:.4f})")
    rush_binary_col = 'rush_binary_A'
    rush_label = '20+ rush yards'
else:
    winner = 'B'
    print(f"\n  Version B (5+ attempts) has stronger partial correlation with hit24")
    print(f"    B: partial r = {r_b_h:+.3f} (p={p_b_h:.4f})")
    print(f"    A: partial r = {r_a_h:+.3f} (p={p_a_h:.4f})")
    rush_binary_col = 'rush_binary_B'
    rush_label = '5+ rush attempts'

print(f"\n  Using Version {winner} ({rush_label}) for weight optimization below.")


# ============================================================================
# 6-COMPONENT WEIGHT OPTIMIZATION WITH BINARY RUSHING
# ============================================================================

print(f"\n\n{'=' * 100}")
print(f"6-COMPONENT WEIGHT OPTIMIZATION — BINARY RUSHING (Version {winner}: {rush_label})")
print("=" * 100)

# Merge teammate scores
tm_lookup = dict(zip(zip(tm['player_name'], tm['draft_year']), tm['avg_teammate_dc']))
wr['teammate_dc'] = wr.apply(lambda x: tm_lookup.get((x['player_name'], x['draft_year']), 0), axis=1)

# Merge games played
gp_lookup = dict(zip(zip(gp['player_name'], gp['draft_year']), gp['games_played']))
wr['games_played'] = wr.apply(lambda x: gp_lookup.get((x['player_name'], x['draft_year']), 11), axis=1)

# Compute scores
wr['s_dc'] = wr['pick'].apply(dc_score)
wr['s_breakout'] = wr.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)
wr['s_teammate'] = wr['teammate_dc'].clip(0, 100)
wr['s_ras'] = (wr['RAS'] * 10).clip(0, 100)
wr['s_declare'] = wr['early_declare'] * 100

# Binary rushing score: 0 or 100 (to match early_declare scale)
wr['s_rushing_bin'] = wr[rush_binary_col] * 100

score_cols = ['s_dc', 's_breakout', 's_teammate', 's_ras', 's_declare', 's_rushing_bin']
score_labels = {
    's_dc': 'Draft Capital',
    's_breakout': 'Breakout Age',
    's_teammate': 'Teammate Score',
    's_ras': 'RAS (Athletic)',
    's_declare': 'Early Declare',
    's_rushing_bin': f'Rushing ({rush_label})',
}

# Check for nulls
total_na = sum(wr[c].isna().sum() for c in score_cols)
print(f"\nTotal NaN across 6 scores: {total_na}")
if total_na > 0:
    print("  WARNING: Some NaN values remain. Filling with 0.")
    for c in score_cols:
        wr[c] = wr[c].fillna(0)

# Component stats
print(f"\n{'Component':<25} {'Nulls':>6} {'Min':>7} {'Mean':>7} {'Median':>7} {'Max':>7}")
print("-" * 65)
for col in score_cols:
    na = wr[col].isna().sum()
    print(f"  {score_labels[col]:<23} {na:>6} {wr[col].min():>7.1f} {wr[col].mean():>7.1f} "
          f"{wr[col].median():>7.1f} {wr[col].max():>7.1f}")


# Filter to evaluation sample
wr_eval = wr[wr['hit24'].notna() & wr['draft_year'].between(2015, 2024)].copy()
print(f"\nEvaluation sample: {len(wr_eval)} WRs with hit24 data")

# GRID SEARCH
print(f"\n{'─' * 80}")
print("GRID SEARCH — 30+ weight configurations")
print(f"{'─' * 80}")
print("Config = DC/Breakout/Teammate/RAS/Declare/Rushing(binary)")

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

    # More rushing emphasis
    (60, 15, 5, 5, 5, 10),
    (60, 10, 5, 5, 5, 15),
    (55, 15, 5, 5, 5, 15),
    (65, 10, 5, 5, 5, 10),
    (65, 15, 0, 5, 5, 10),
    (65, 10, 5, 0, 5, 15),
    (60, 15, 5, 0, 5, 15),
    (55, 20, 5, 0, 5, 15),
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
print(f"{'#':>3} {'Config':>33}  {'hit24':>8} {'ppr':>8}  {'avg':>8}")
print("-" * 70)

results = []
for i, (dc_w, bo_w, tm_w, ras_w, ed_w, rush_w) in enumerate(configs):
    slap = (
        (dc_w / 100) * wr_eval['s_dc'] +
        (bo_w / 100) * wr_eval['s_breakout'] +
        (tm_w / 100) * wr_eval['s_teammate'] +
        (ras_w / 100) * wr_eval['s_ras'] +
        (ed_w / 100) * wr_eval['s_declare'] +
        (rush_w / 100) * wr_eval['s_rushing_bin']
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

# Sort and show top 10
results_df = pd.DataFrame(results).sort_values('avg_r', ascending=False)

print(f"\n\n{'─' * 80}")
print("TOP 10 CONFIGURATIONS (by average correlation)")
print(f"{'─' * 80}")
print(f"\n{'Rank':>4} {'Config':>33}  {'hit24':>8} {'ppr':>8}  {'avg':>8}")
print("-" * 70)

for i, (_, row) in enumerate(results_df.head(10).iterrows()):
    marker = " <-- CURRENT" if row['config'] == "65/20/0/15/0/0" else ""
    ppr_str = f"{row['r_ppr']:>+.3f}" if not np.isnan(row['r_ppr']) else "N/A"
    print(f"{i+1:>4}  {row['config']:>31}  {row['r_hit24']:>+.3f}   {ppr_str:>7}   {row['avg_r']:>+.3f}{marker}")

# Current model comparison
print(f"\n\nCurrent model (65/20/0/15/0/0):")
current = results_df[results_df['config'] == "65/20/0/15/0/0"]
if len(current) > 0:
    c = current.iloc[0]
    print(f"  hit24 r = {c['r_hit24']:+.3f} (p={c['p_hit24']:.4f})")
    print(f"  ppr r   = {c['r_ppr']:+.3f}")
    print(f"  avg r   = {c['avg_r']:+.3f}")

best = results_df.iloc[0]
print(f"\nBest grid search ({best['config']}):")
print(f"  hit24 r = {best['r_hit24']:+.3f} (p={best['p_hit24']:.4f})")
print(f"  ppr r   = {best['r_ppr']:+.3f}")
print(f"  avg r   = {best['avg_r']:+.3f}")


# SCIPY OPTIMIZER
print(f"\n\n{'─' * 80}")
print("SCIPY OPTIMIZER — CONTINUOUS WEIGHT OPTIMIZATION")
print(f"{'─' * 80}")
print("Constraints: weights sum to 1.0, DC >= 0.40, each component >= 0.0")

eval_data = wr_eval[score_cols + ['hit24', 'best_ppr']].dropna(subset=['hit24']).copy()
X = eval_data[score_cols].values
y_hit24 = eval_data['hit24'].values
has_ppr = eval_data['best_ppr'].notna()
y_ppr = eval_data.loc[has_ppr, 'best_ppr'].values
X_ppr = eval_data.loc[has_ppr, score_cols].values

print(f"\nOptimization sample: {len(X)} WRs (hit24), {len(X_ppr)} WRs (best_ppr)")


def neg_correlation(weights, X, y):
    scores = X @ weights
    if np.std(scores) < 1e-10:
        return 0
    r = np.corrcoef(scores, y)[0, 1]
    return -r


def neg_avg_correlation(weights, X_hit, y_hit, X_ppr, y_ppr):
    s1 = X_hit @ weights
    r1 = np.corrcoef(s1, y_hit)[0, 1] if np.std(s1) > 1e-10 else 0
    s2 = X_ppr @ weights
    r2 = np.corrcoef(s2, y_ppr)[0, 1] if np.std(s2) > 1e-10 else 0
    return -(r1 + r2) / 2


constraints = [
    {'type': 'eq', 'fun': lambda w: sum(w) - 1.0},
]
bounds = [
    (0.40, 0.80),  # DC
    (0.05, 0.35),  # Breakout
    (0.00, 0.15),  # Teammate
    (0.00, 0.20),  # RAS
    (0.00, 0.15),  # Early Declare
    (0.00, 0.20),  # Rushing (binary)
]

x0 = [0.65, 0.20, 0.00, 0.15, 0.00, 0.00]

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
r_ppr_val = -res_ppr.fun
print(f"  Optimal weights: DC={w_ppr[0]:.0%} / BO={w_ppr[1]:.0%} / TM={w_ppr[2]:.0%} / "
      f"RAS={w_ppr[3]:.0%} / ED={w_ppr[4]:.0%} / RUSH={w_ppr[5]:.0%}")
print(f"  Correlation: r = {r_ppr_val:+.4f}")

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

# Multi-start optimization
print(f"\n\n{'─' * 80}")
print("MULTI-START OPTIMIZER (5 random starting points)")
print(f"{'─' * 80}")

best_r_ms = -999
best_w_ms = None
np.random.seed(42)

for start_idx in range(5):
    # Random starting point satisfying constraints
    w_rand = np.random.dirichlet(np.ones(6))
    # Ensure DC >= 0.40
    if w_rand[0] < 0.40:
        deficit = 0.40 - w_rand[0]
        w_rand[0] = 0.40
        # Redistribute deficit from other weights
        others_sum = w_rand[1:].sum()
        if others_sum > 0:
            w_rand[1:] *= (1.0 - 0.40) / others_sum
    # Clip to bounds
    for j, (lo, hi) in enumerate(bounds):
        w_rand[j] = max(lo, min(hi, w_rand[j]))
    # Renormalize
    w_rand /= w_rand.sum()

    res_ms = minimize(neg_avg_correlation, w_rand, args=(X, y_hit24, X_ppr, y_ppr),
                      method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000})
    r_ms = -res_ms.fun
    w_ms = res_ms.x

    tag = " <-- BEST" if r_ms > best_r_ms else ""
    if r_ms > best_r_ms:
        best_r_ms = r_ms
        best_w_ms = w_ms

    print(f"  Start {start_idx+1}: DC={w_ms[0]:.0%}/BO={w_ms[1]:.0%}/TM={w_ms[2]:.0%}/"
          f"RAS={w_ms[3]:.0%}/ED={w_ms[4]:.0%}/RUSH={w_ms[5]:.0%} → avg r = {r_ms:+.4f}{tag}")

print(f"\n  Best multi-start result: avg r = {best_r_ms:+.4f}")
print(f"  Weights: DC={best_w_ms[0]:.0%} / BO={best_w_ms[1]:.0%} / TM={best_w_ms[2]:.0%} / "
      f"RAS={best_w_ms[3]:.0%} / ED={best_w_ms[4]:.0%} / RUSH={best_w_ms[5]:.0%}")


# ============================================================================
# PARTIAL CORRELATIONS WITH BINARY RUSHING
# ============================================================================

print(f"\n\n{'=' * 100}")
print("PARTIAL CORRELATIONS — 6 COMPONENTS WITH BINARY RUSHING")
print("=" * 100)
print("Unique contribution of each component controlling for the other 5")

print(f"\n{'Component':<25} {'hit24':>10} {'best_ppr':>10} {'N':>6}")
print("-" * 55)

for col in score_cols:
    others = [c for c in score_cols if c != col]
    row = f"  {score_labels[col]:<23}"
    for out in ['hit24', 'best_ppr']:
        covs = [wr_eval[c] for c in others]
        r, p, n = partial_correlation(wr_eval[col], wr_eval[out], covs)
        if not np.isnan(r):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f" {r:>+.3f}{sig:<4}"
        else:
            row += f" {'N/A':>10}"
    n_full = wr_eval[score_cols + [out]].dropna().shape[0]
    row += f" {n_full:>5}"
    print(row)

print(f"\n  * p<0.05, ** p<0.01, *** p<0.001")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n\n{'=' * 100}")
print("FINAL SUMMARY")
print("=" * 100)

print(f"""
BINARY RUSHING ANALYSIS RESULTS
================================

Version A (20+ rushing yards):
  Qualifiers: {n_a_1} WRs
  Hit24 rate: qualifiers={hit_a1:.3f} vs non-qualifiers={hit_a0:.3f}
  Partial r vs hit24 (ctrl DC): {r_a_h:+.3f} (p={p_a_h:.4f})

Version B (5+ rush attempts):
  Qualifiers: {n_b_1} WRs
  Hit24 rate: qualifiers={hit_b1:.3f} vs non-qualifiers={hit_b0:.3f}
  Partial r vs hit24 (ctrl DC): {r_b_h:+.3f} (p={p_b_h:.4f})

Winner: Version {winner} ({rush_label})

Weight Optimization (with binary rushing):
  Current model (65/20/0/15/0/0): avg r = {r_current_avg:+.4f}
  Best grid search ({best['config']}): avg r = {best['avg_r']:+.4f}""")

if len(X_ppr) >= 10:
    print(f"  Scipy optimized: avg r = {r_avg:+.4f}")
    print(f"    DC={w_avg[0]:.0%} / BO={w_avg[1]:.0%} / TM={w_avg[2]:.0%} / "
          f"RAS={w_avg[3]:.0%} / ED={w_avg[4]:.0%} / RUSH={w_avg[5]:.0%}")
    print(f"  Multi-start best: avg r = {best_r_ms:+.4f}")
    print(f"    DC={best_w_ms[0]:.0%} / BO={best_w_ms[1]:.0%} / TM={best_w_ms[2]:.0%} / "
          f"RAS={best_w_ms[3]:.0%} / ED={best_w_ms[4]:.0%} / RUSH={best_w_ms[5]:.0%}")

print()
