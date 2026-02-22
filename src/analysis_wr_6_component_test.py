"""
WR SLAP Model - Comprehensive 6-Component Analysis
=====================================================

Tests all 6 proposed WR components:
1. Draft Capital (DC) - primary driver
2. Breakout Age - production profile
3. Teammate Score - elite program adjustment
4. RAS (Relative Athletic Score) - athleticism
5. Early Declare status
6. Rushing production - gadget/all-purpose usage

Step 1: Partial correlations (unique contribution of each component)
Step 2: Multicollinearity check (are components redundant?)
Step 3: Weight optimization (15+ configurations)
Step 4: Missing data coverage report
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_draft_capital(pick):
    """DC = 100 - 2.40 * (pick^0.62 - 1)"""
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))

def wr_breakout_score(breakout_age, dominator_pct):
    """Continuous breakout scoring: age tier + dominator tiebreaker."""
    if breakout_age is None or pd.isna(breakout_age):
        if dominator_pct is not None and pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        else:
            return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base_score = age_tiers.get(int(breakout_age), 20)
    if dominator_pct is not None and pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
        bonus = max(0, bonus)
    else:
        bonus = 0
    return min(base_score + bonus, 99.9)

def partial_correlation(x, y, covariates):
    """
    Calculate partial correlation between x and y, controlling for covariates.
    Returns (r, p-value, n).
    Uses OLS residuals method.
    """
    # Drop rows with any NaN
    data = pd.DataFrame({'x': x, 'y': y})
    for i, cov in enumerate(covariates):
        data[f'cov_{i}'] = cov
    data = data.dropna()
    n = len(data)

    if n < 10:
        return np.nan, np.nan, n

    # Regress x on covariates, get residuals
    cov_cols = [c for c in data.columns if c.startswith('cov_')]
    if len(cov_cols) == 0:
        r, p = stats.pearsonr(data['x'], data['y'])
        return r, p, n

    cov_matrix = data[cov_cols].values

    # Add intercept
    X = np.column_stack([np.ones(n), cov_matrix])

    # Residuals for x
    try:
        beta_x = np.linalg.lstsq(X, data['x'].values, rcond=None)[0]
        resid_x = data['x'].values - X @ beta_x

        # Residuals for y
        beta_y = np.linalg.lstsq(X, data['y'].values, rcond=None)[0]
        resid_y = data['y'].values - X @ beta_y

        r, p = stats.pearsonr(resid_x, resid_y)
        return r, p, n
    except:
        return np.nan, np.nan, n


# ============================================================================
# LOAD AND MERGE ALL DATA
# ============================================================================

print("=" * 100)
print("WR SLAP MODEL - COMPREHENSIVE 6-COMPONENT ANALYSIS")
print("=" * 100)

# 1. Main backtest (339 WRs)
wr = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"\n[1] Main backtest: {len(wr)} WRs")

# 2. Dominator data (for breakout scoring)
dom = pd.read_csv('data/wr_dominator_complete.csv')
wr = wr.merge(dom[['player_name', 'draft_year', 'dominator_pct']],
              on=['player_name', 'draft_year'], how='left')

# 3. Teammate scores
teammates = pd.read_csv('data/wr_teammate_scores.csv')
wr = wr.merge(teammates[['player_name', 'draft_year', 'teammate_count',
                           'best_teammate_pick', 'total_teammate_dc', 'avg_teammate_dc']],
              on=['player_name', 'draft_year'], how='left')
print(f"[2] Teammate data merged: {wr['total_teammate_dc'].notna().sum()} with data")

# 4. Early declare status
declare = pd.read_csv('data/wr_eval_with_declare.csv')
# Only keep declare-related columns to avoid column conflicts
declare_cols = declare[['player_name', 'draft_year', 'declare_status', 'early_declare', 'draft_age']].drop_duplicates()
wr = wr.merge(declare_cols, on=['player_name', 'draft_year'], how='left')
print(f"[3] Early declare merged: {wr['declare_status'].notna().sum()} with data")

# 5. Rushing production from PFF college data
pff = pd.read_csv('data/wr_pff_all_2016_2025.csv')
pff_rush = pff[['player_name', 'draft_year', 'rush_attempts', 'rush_yards',
                 'rush_touchdowns', 'player_game_count']].copy()
wr = wr.merge(pff_rush, on=['player_name', 'draft_year'], how='left')
print(f"[4] PFF rushing data merged: {wr['rush_yards'].notna().sum()} with data")

# 6. Outcomes (first_3yr_ppg, career_ppg)
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
outcomes_wr = outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year', 'pick',
                                                       'first_3yr_ppg', 'career_ppg']].copy()
outcomes_wr = outcomes_wr.rename(columns={'pick': 'pick_outcomes'})
wr = wr.merge(outcomes_wr[['player_name', 'draft_year', 'first_3yr_ppg', 'career_ppg']],
              on=['player_name', 'draft_year'], how='left')
print(f"[5] Outcomes merged: {wr['first_3yr_ppg'].notna().sum()} with first_3yr_ppg, "
      f"{wr['career_ppg'].notna().sum()} with career_ppg")

print(f"\nTotal WRs in merged dataset: {len(wr)}")


# ============================================================================
# CALCULATE ALL 6 COMPONENT SCORES (0-100 scale)
# ============================================================================

print("\n" + "=" * 100)
print("CALCULATING 6 COMPONENT SCORES (all on 0-100 scale)")
print("=" * 100)

# --- Component 1: Draft Capital ---
wr['comp_dc'] = wr['pick'].apply(normalize_draft_capital)
print(f"\n1. DRAFT CAPITAL: {wr['comp_dc'].notna().sum()}/{len(wr)} with data")
print(f"   Range: {wr['comp_dc'].min():.1f} to {wr['comp_dc'].max():.1f}, Mean: {wr['comp_dc'].mean():.1f}")

# --- Component 2: Breakout Age ---
wr['comp_breakout'] = wr.apply(
    lambda x: wr_breakout_score(x['breakout_age'], x['dominator_pct']), axis=1
)
print(f"\n2. BREAKOUT AGE: {wr['comp_breakout'].notna().sum()}/{len(wr)} with data")
print(f"   Range: {wr['comp_breakout'].min():.1f} to {wr['comp_breakout'].max():.1f}, Mean: {wr['comp_breakout'].mean():.1f}")

# --- Component 3: Teammate Score ---
# Higher total_teammate_dc = had more/better teammates = harder to dominate
# Score: normalize total_teammate_dc to 0-100
# Players with 0 teammates get 0 (no adjustment)
wr['comp_teammate_raw'] = wr['total_teammate_dc'].fillna(0)
# Normalize: 0 = no teammates, 100 = max teammate DC in dataset
max_tm_dc = wr['comp_teammate_raw'].max()
if max_tm_dc > 0:
    wr['comp_teammate'] = (wr['comp_teammate_raw'] / max_tm_dc) * 100
else:
    wr['comp_teammate'] = 0
print(f"\n3. TEAMMATE SCORE: {(wr['total_teammate_dc'].notna()).sum()}/{len(wr)} with data")
print(f"   Raw total_teammate_dc range: {wr['comp_teammate_raw'].min():.1f} to {wr['comp_teammate_raw'].max():.1f}")
print(f"   Normalized range: {wr['comp_teammate'].min():.1f} to {wr['comp_teammate'].max():.1f}, Mean: {wr['comp_teammate'].mean():.1f}")

# --- Component 4: RAS (Athletic) ---
wr['comp_ras'] = wr['RAS'] * 10  # Convert 0-10 to 0-100
print(f"\n4. RAS (ATHLETIC): {wr['comp_ras'].notna().sum()}/{len(wr)} with data")
ras_valid = wr['comp_ras'].dropna()
print(f"   Range: {ras_valid.min():.1f} to {ras_valid.max():.1f}, Mean: {ras_valid.mean():.1f}")

# --- Component 5: Early Declare ---
# EARLY = 100, STANDARD = 50, LATE = 0
declare_map = {'EARLY': 100, 'STANDARD': 50, 'LATE': 0}
wr['comp_early_declare'] = wr['declare_status'].map(declare_map)
print(f"\n5. EARLY DECLARE: {wr['comp_early_declare'].notna().sum()}/{len(wr)} with data")
print(f"   Distribution:")
for status in ['EARLY', 'STANDARD', 'LATE']:
    count = (wr['declare_status'] == status).sum()
    print(f"     {status}: {count} ({count/len(wr)*100:.1f}%)")
missing_declare = wr['declare_status'].isna().sum()
print(f"     MISSING: {missing_declare} ({missing_declare/len(wr)*100:.1f}%)")

# --- Component 6: Rushing Production ---
# Rush yards per game, normalized to 0-100
# This captures "gadget" WRs who also carry the ball
wr['rush_ypg'] = np.where(
    wr['player_game_count'].notna() & (wr['player_game_count'] > 0),
    wr['rush_yards'].fillna(0) / wr['player_game_count'],
    np.nan
)
# Normalize: max rush_ypg in dataset = 100
rush_valid = wr['rush_ypg'].dropna()
max_rush = rush_valid.max() if len(rush_valid) > 0 else 1
wr['comp_rushing'] = np.where(
    wr['rush_ypg'].notna(),
    (wr['rush_ypg'] / max_rush) * 100,
    np.nan
)
print(f"\n6. RUSHING PRODUCTION: {wr['comp_rushing'].notna().sum()}/{len(wr)} with data")
if len(rush_valid) > 0:
    print(f"   Raw rush_ypg range: {rush_valid.min():.1f} to {rush_valid.max():.1f}")
    rushing_valid = wr['comp_rushing'].dropna()
    print(f"   Normalized range: {rushing_valid.min():.1f} to {rushing_valid.max():.1f}, Mean: {rushing_valid.mean():.1f}")


# ============================================================================
# DEFINE OUTCOMES
# ============================================================================

outcomes_list = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
outcome_labels = {
    'hit24': 'Hit24 (top-24 WR in first 2 years)',
    'hit12': 'Hit12 (top-12 WR in year 1)',
    'first_3yr_ppg': 'First 3yr PPG (PPR points per game)',
    'career_ppg': 'Career PPG (PPR points per game)'
}

print("\n\nOutcome coverage:")
for out in outcomes_list:
    n = wr[out].notna().sum()
    print(f"  {out}: {n}/{len(wr)} WRs with data")


# ============================================================================
# STEP 1: PARTIAL CORRELATIONS
# ============================================================================

print("\n" + "=" * 100)
print("STEP 1: PARTIAL CORRELATIONS")
print("=" * 100)
print("\nFor each component, this shows the UNIQUE contribution to predicting")
print("each outcome AFTER controlling for ALL other 5 components simultaneously.")
print("This is the gold standard test - it tells you what each variable adds")
print("that the others can't already explain.")

components = ['comp_dc', 'comp_breakout', 'comp_teammate', 'comp_ras',
              'comp_early_declare', 'comp_rushing']
comp_labels = {
    'comp_dc': 'Draft Capital',
    'comp_breakout': 'Breakout Age',
    'comp_teammate': 'Teammate Score',
    'comp_ras': 'RAS (Athletic)',
    'comp_early_declare': 'Early Declare',
    'comp_rushing': 'Rushing Prod'
}

# First, show raw (zero-order) correlations for context
print("\n--- RAW CORRELATIONS (no controls, largest sample per component) ---")
print(f"\n{'Component':<20} {'hit24':>12} {'hit12':>12} {'first_3yr_ppg':>14} {'career_ppg':>12} {'N':>6}")
print("-" * 80)

for comp in components:
    row = f"{comp_labels[comp]:<20}"
    for out in outcomes_list:
        valid = wr[[comp, out]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid[comp], valid[out])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f"  {r:>+.3f}{sig:<3}"
        else:
            row += f"  {'N/A':>9}"
    # Show N for the component
    n = wr[comp].notna().sum()
    row += f" {n:>5}"
    print(row)

print("\n  * p<0.05, ** p<0.01, *** p<0.001")

# Now partial correlations
print("\n\n--- PARTIAL CORRELATIONS (controlling for all other 5 components) ---")
print(f"\n{'Component':<20} {'hit24':>12} {'hit12':>12} {'first_3yr_ppg':>14} {'career_ppg':>12} {'N':>6}")
print("-" * 80)

for comp in components:
    other_comps = [c for c in components if c != comp]
    row = f"{comp_labels[comp]:<20}"
    for out in outcomes_list:
        covariates = [wr[c] for c in other_comps]
        r, p, n = partial_correlation(wr[comp], wr[out], covariates)
        if not np.isnan(r):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f"  {r:>+.3f}{sig:<3}"
        else:
            row += f"  {'N/A':>9}"
    # Show N (sample with ALL components present)
    all_cols = components + [out]
    n_all = wr[components + outcomes_list[:1]].dropna().shape[0]
    row += f" {n_all:>5}"
    print(row)

print("\n  * p<0.05, ** p<0.01, *** p<0.001")
print(f"\n  NOTE: Partial correlations require ALL 6 components + outcome to be non-missing.")

# Show the N for the full sample
n_full = wr[components + ['hit24']].dropna().shape[0]
print(f"  Full 6-component sample: {n_full} WRs (limited by PFF rushing coverage)")

# Also run with 5 components (excluding rushing) for larger sample
print("\n\n--- PARTIAL CORRELATIONS WITHOUT RUSHING (5 components, larger sample) ---")
components_5 = ['comp_dc', 'comp_breakout', 'comp_teammate', 'comp_ras', 'comp_early_declare']
print(f"\n{'Component':<20} {'hit24':>12} {'hit12':>12} {'first_3yr_ppg':>14} {'career_ppg':>12} {'N':>6}")
print("-" * 80)

for comp in components_5:
    other_comps = [c for c in components_5 if c != comp]
    row = f"{comp_labels[comp]:<20}"
    for out in outcomes_list:
        covariates = [wr[c] for c in other_comps]
        r, p, n = partial_correlation(wr[comp], wr[out], covariates)
        if not np.isnan(r):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f"  {r:>+.3f}{sig:<3}"
        else:
            row += f"  {'N/A':>9}"
    n_5 = wr[components_5 + [out]].dropna().shape[0]
    row += f" {n_5:>5}"
    print(row)

n_5comp = wr[components_5 + ['hit24']].dropna().shape[0]
print(f"\n  5-component sample: {n_5comp} WRs (no rushing requirement)")

# Also run with just 3 original components for comparison
print("\n\n--- PARTIAL CORRELATIONS - ORIGINAL 3 COMPONENTS (baseline comparison) ---")
components_3 = ['comp_dc', 'comp_breakout', 'comp_ras']
print(f"\n{'Component':<20} {'hit24':>12} {'hit12':>12} {'first_3yr_ppg':>14} {'career_ppg':>12} {'N':>6}")
print("-" * 80)

for comp in components_3:
    other_comps = [c for c in components_3 if c != comp]
    row = f"{comp_labels[comp]:<20}"
    for out in outcomes_list:
        covariates = [wr[c] for c in other_comps]
        r, p, n = partial_correlation(wr[comp], wr[out], covariates)
        if not np.isnan(r):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f"  {r:>+.3f}{sig:<3}"
        else:
            row += f"  {'N/A':>9}"
    n_3 = wr[components_3 + [out]].dropna().shape[0]
    row += f" {n_3:>5}"
    print(row)

n_3comp = wr[components_3 + ['hit24']].dropna().shape[0]
print(f"\n  3-component sample: {n_3comp} WRs (original model baseline)")


# ============================================================================
# STEP 2: MULTICOLLINEARITY CHECK
# ============================================================================

print("\n\n" + "=" * 100)
print("STEP 2: MULTICOLLINEARITY CHECK")
print("=" * 100)
print("\nCorrelation matrix between all 6 components.")
print("If two components correlate >0.50, they may be measuring the same thing.")
print("If >0.70, they are almost certainly redundant.")

# Use pairwise complete observations for each pair
print(f"\n{'':>20}", end="")
for comp in components:
    print(f" {comp_labels[comp]:>14}", end="")
print(f" {'N (pairwise)':>14}")
print("-" * 110)

for comp1 in components:
    row = f"{comp_labels[comp1]:<20}"
    for comp2 in components:
        if comp1 == comp2:
            row += f"  {'1.000':>12}"
        else:
            valid = wr[[comp1, comp2]].dropna()
            if len(valid) >= 10:
                r, p = stats.pearsonr(valid[comp1], valid[comp2])
                sig = "*" if p < 0.05 else ""
                row += f"  {r:>+.3f}{sig:<8}"
            else:
                row += f"  {'N/A':>12}"
    # N for this row (pairwise with worst partner)
    ns = []
    for comp2 in components:
        if comp1 != comp2:
            ns.append(wr[[comp1, comp2]].dropna().shape[0])
    row += f" {min(ns) if ns else 0:>12}"
    print(row)

print("\n  * = statistically significant (p<0.05)")

# Specific checks the user asked about
print("\n--- SPECIFIC REDUNDANCY CHECKS ---")

# Check 1: Is early declare redundant with breakout age?
valid = wr[['comp_early_declare', 'comp_breakout']].dropna()
r, p = stats.pearsonr(valid['comp_early_declare'], valid['comp_breakout'])
print(f"\n  Early Declare vs Breakout Age: r={r:+.3f} (p={p:.4f}, n={len(valid)})")
if abs(r) > 0.5:
    print(f"  --> WARNING: These are moderately correlated - may be measuring similar things")
elif abs(r) > 0.3:
    print(f"  --> MILD overlap - some shared signal but mostly independent")
else:
    print(f"  --> LOW overlap - these measure different things")

# Check 2: Is teammate score correlated with draft capital?
valid = wr[['comp_teammate', 'comp_dc']].dropna()
r, p = stats.pearsonr(valid['comp_teammate'], valid['comp_dc'])
print(f"\n  Teammate Score vs Draft Capital: r={r:+.3f} (p={p:.4f}, n={len(valid)})")
if abs(r) > 0.5:
    print(f"  --> WARNING: These are moderately correlated - DC may already capture teammate effect")
elif abs(r) > 0.3:
    print(f"  --> MILD overlap - some shared signal")
else:
    print(f"  --> LOW overlap - teammate score adds independent information")

# Check 3: Is rushing production correlated with anything?
print(f"\n  Rushing Production correlations with other components:")
for comp in components:
    if comp == 'comp_rushing':
        continue
    valid = wr[['comp_rushing', comp]].dropna()
    if len(valid) >= 10:
        r, p = stats.pearsonr(valid['comp_rushing'], valid[comp])
        sig = "*" if p < 0.05 else ""
        print(f"    vs {comp_labels[comp]:<20}: r={r:>+.3f}{sig} (n={len(valid)})")

# Check 4: Is early declare correlated with draft capital?
valid = wr[['comp_early_declare', 'comp_dc']].dropna()
r, p = stats.pearsonr(valid['comp_early_declare'], valid['comp_dc'])
print(f"\n  Early Declare vs Draft Capital: r={r:+.3f} (p={p:.4f}, n={len(valid)})")
if abs(r) > 0.5:
    print(f"  --> WARNING: Early declares tend to go higher - DC may capture this")
elif abs(r) > 0.3:
    print(f"  --> MILD overlap - some shared signal")
else:
    print(f"  --> LOW overlap - early declare adds independent information")


# ============================================================================
# STEP 3: WEIGHT OPTIMIZATION WITH ALL 6 COMPONENTS
# ============================================================================

print("\n\n" + "=" * 100)
print("STEP 3: WEIGHT OPTIMIZATION (6 Components)")
print("=" * 100)
print("\nConstraints: DC >= 50%, Breakout >= 15%, Others 2-10% each, sum = 100%")
print("Testing 20+ configurations against all 4 outcomes.\n")

# Define weight configurations: (dc, breakout, teammate, ras, early_declare, rushing)
configs = [
    # Baseline (current 3-component mapped to 6)
    (65, 20, 0, 15, 0, 0),    # Current model (no new components)

    # Equal minor weights
    (60, 20, 5, 5, 5, 5),
    (55, 20, 7, 7, 6, 5),
    (65, 15, 5, 5, 5, 5),
    (55, 25, 5, 5, 5, 5),
    (60, 15, 8, 7, 5, 5),

    # DC-heavy
    (65, 15, 5, 5, 5, 5),
    (65, 15, 5, 7, 5, 3),
    (65, 15, 5, 5, 7, 3),
    (65, 15, 3, 7, 5, 5),
    (65, 15, 7, 5, 5, 3),

    # Higher breakout
    (55, 25, 5, 5, 5, 5),
    (50, 25, 7, 7, 6, 5),
    (55, 20, 7, 8, 5, 5),
    (55, 20, 5, 8, 7, 5),

    # Teammate emphasis
    (55, 20, 10, 5, 5, 5),
    (60, 15, 10, 5, 5, 5),
    (55, 20, 10, 7, 5, 3),

    # Athletic emphasis
    (55, 20, 5, 10, 5, 5),
    (60, 15, 5, 10, 5, 5),

    # Early declare emphasis
    (55, 20, 5, 5, 10, 5),
    (60, 15, 5, 5, 10, 5),

    # Rushing emphasis
    (55, 20, 5, 5, 5, 10),
    (60, 15, 5, 5, 5, 10),

    # Minimal minor weights
    (60, 20, 5, 5, 5, 5),
    (60, 20, 3, 7, 5, 5),
    (60, 20, 5, 7, 3, 5),
    (60, 20, 5, 5, 3, 7),
    (60, 20, 7, 3, 5, 5),
    (60, 20, 5, 3, 7, 5),
]

# Remove exact duplicates
seen = set()
unique_configs = []
for c in configs:
    if c not in seen:
        seen.add(c)
        unique_configs.append(c)
configs = unique_configs

# Prepare the data - need complete cases for all 6 components + outcomes
# Use imputed values for missing to maximize sample size
wr_test = wr.copy()
wr_test['comp_ras_imp'] = wr_test['comp_ras'].fillna(wr_test['comp_ras'].mean())
wr_test['comp_early_declare_imp'] = wr_test['comp_early_declare'].fillna(50)  # Default to STANDARD
wr_test['comp_rushing_imp'] = wr_test['comp_rushing'].fillna(wr_test['comp_rushing'].mean())

print("Using FULL sample (339 WRs) with imputed values for missing components.")
print("(Missing RAS → position average, Missing declare → STANDARD, Missing rushing → average)")

# Header
print(f"\n{'Config':>35}  {'hit24':>8} {'hit12':>8} {'3yr_ppg':>8} {'car_ppg':>8}  {'avg_r':>8}")
print("-" * 95)

results = []
for dc_w, bo_w, tm_w, ras_w, ed_w, rush_w in configs:
    # Calculate weighted SLAP score
    slap = (
        (dc_w / 100) * wr_test['comp_dc'] +
        (bo_w / 100) * wr_test['comp_breakout'] +
        (tm_w / 100) * wr_test['comp_teammate'] +
        (ras_w / 100) * wr_test['comp_ras_imp'] +
        (ed_w / 100) * wr_test['comp_early_declare_imp'] +
        (rush_w / 100) * wr_test['comp_rushing_imp']
    )

    label = f"{dc_w}/{bo_w}/{tm_w}/{ras_w}/{ed_w}/{rush_w}"

    row_data = {'config': label, 'dc': dc_w, 'bo': bo_w, 'tm': tm_w,
                'ras': ras_w, 'ed': ed_w, 'rush': rush_w}

    cors = []
    row_str = f"  {label:>33}"
    for out in outcomes_list:
        valid = pd.DataFrame({'slap': slap, 'out': wr_test[out]}).dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid['slap'], valid['out'])
            sig = "*" if p < 0.05 else ""
            row_str += f"  {r:>+.3f}{sig:<2}"
            cors.append(r)
            row_data[out] = r
        else:
            row_str += f"  {'N/A':>8}"
            row_data[out] = np.nan

    avg_r = np.mean(cors) if cors else np.nan
    row_str += f"   {avg_r:>+.3f}"
    row_data['avg_r'] = avg_r
    results.append(row_data)

    # Mark if this is the current model
    marker = " <-- CURRENT" if (dc_w == 65 and bo_w == 20 and tm_w == 0 and ras_w == 15 and ed_w == 0 and rush_w == 0) else ""
    print(row_str + marker)

print("\n  * p<0.05")

# Sort and show top 5 by average correlation
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('avg_r', ascending=False)

print("\n\n--- TOP 10 CONFIGURATIONS (by average correlation across all 4 outcomes) ---")
print(f"\n{'Rank':>4} {'Config':>35}  {'hit24':>8} {'hit12':>8} {'3yr_ppg':>8} {'car_ppg':>8}  {'avg_r':>8}")
print("-" * 95)
for i, (_, row) in enumerate(results_df.head(10).iterrows()):
    marker = " <-- CURRENT" if row['config'] == "65/20/0/15/0/0" else ""
    print(f"{i+1:>4}  {row['config']:>33}  {row.get('hit24', np.nan):>+.3f}   {row.get('hit12', np.nan):>+.3f}   "
          f"{row.get('first_3yr_ppg', np.nan):>+.3f}   {row.get('career_ppg', np.nan):>+.3f}    {row['avg_r']:>+.3f}{marker}")


# Also run weight optimization on the OBSERVED-ONLY sample (no imputation for rushing)
print("\n\n--- WEIGHT OPTIMIZATION ON PFF-COVERED SAMPLE ONLY (no imputation) ---")
print("This uses only WRs with ACTUAL rushing data (2016+ draft classes).\n")

# Filter to rows with all 6 observed
wr_observed = wr.dropna(subset=['comp_dc', 'comp_breakout', 'comp_teammate',
                                 'comp_ras', 'comp_early_declare', 'comp_rushing', 'hit24'])
print(f"Sample size: {len(wr_observed)} WRs with ALL 6 components + hit24 observed")

if len(wr_observed) >= 30:
    print(f"\n{'Config':>35}  {'hit24':>8} {'hit12':>8} {'3yr_ppg':>8} {'car_ppg':>8}  {'avg_r':>8}")
    print("-" * 95)

    obs_results = []
    for dc_w, bo_w, tm_w, ras_w, ed_w, rush_w in configs:
        slap = (
            (dc_w / 100) * wr_observed['comp_dc'] +
            (bo_w / 100) * wr_observed['comp_breakout'] +
            (tm_w / 100) * wr_observed['comp_teammate'] +
            (ras_w / 100) * wr_observed['comp_ras'] +
            (ed_w / 100) * wr_observed['comp_early_declare'] +
            (rush_w / 100) * wr_observed['comp_rushing']
        )

        label = f"{dc_w}/{bo_w}/{tm_w}/{ras_w}/{ed_w}/{rush_w}"
        row_data = {'config': label}

        cors = []
        row_str = f"  {label:>33}"
        for out in outcomes_list:
            valid = pd.DataFrame({'slap': slap, 'out': wr_observed[out]}).dropna()
            if len(valid) >= 10:
                r, p = stats.pearsonr(valid['slap'], valid['out'])
                sig = "*" if p < 0.05 else ""
                row_str += f"  {r:>+.3f}{sig:<2}"
                cors.append(r)
                row_data[out] = r
            else:
                row_str += f"  {'N/A':>8}"

        avg_r = np.mean(cors) if cors else np.nan
        row_str += f"   {avg_r:>+.3f}"
        row_data['avg_r'] = avg_r
        obs_results.append(row_data)

        marker = " <-- CURRENT" if (dc_w == 65 and bo_w == 20 and tm_w == 0 and ras_w == 15 and ed_w == 0 and rush_w == 0) else ""
        print(row_str + marker)
else:
    print(f"  NOT ENOUGH DATA ({len(wr_observed)} WRs). Need at least 30 for reliable correlations.")


# ============================================================================
# STEP 4: MISSING DATA REPORT
# ============================================================================

print("\n\n" + "=" * 100)
print("STEP 4: MISSING DATA REPORT")
print("=" * 100)
print(f"\nTotal WRs in backtest: {len(wr)}")

print(f"\n{'Component':<25} {'Has Data':>10} {'Missing':>10} {'Pct Coverage':>14} {'Notes'}")
print("-" * 100)

# DC
n_dc = wr['comp_dc'].notna().sum()
print(f"{'Draft Capital':<25} {n_dc:>10} {len(wr)-n_dc:>10} {n_dc/len(wr)*100:>13.1f}% {'All players have pick data'}")

# Breakout
n_bo = wr['comp_breakout'].notna().sum()
print(f"{'Breakout Age':<25} {n_bo:>10} {len(wr)-n_bo:>10} {n_bo/len(wr)*100:>13.1f}% {'Continuous scoring handles never-broke-out players'}")

# Teammate
n_tm = wr['total_teammate_dc'].notna().sum()
print(f"{'Teammate Score':<25} {n_tm:>10} {len(wr)-n_tm:>10} {n_tm/len(wr)*100:>13.1f}% {'0 teammates = score of 0 (no adjustment)'}")

# RAS
n_ras = wr['comp_ras'].notna().sum()
n_miss_ras = len(wr) - n_ras
print(f"{'RAS (Athletic)':<25} {n_ras:>10} {n_miss_ras:>10} {n_ras/len(wr)*100:>13.1f}% {'MNAR: elite prospects skip combine'}")

# RAS missing by round
print(f"\n  RAS missing data breakdown by round:")
for rd in range(1, 8):
    rd_players = wr[wr['round'] == rd]
    rd_missing = rd_players['comp_ras'].isna().sum()
    rd_total = len(rd_players)
    if rd_total > 0:
        pct = rd_missing / rd_total * 100
        # Hit rate for missing vs present
        has_ras = rd_players[rd_players['comp_ras'].notna()]
        no_ras = rd_players[rd_players['comp_ras'].isna()]
        hit_has = has_ras['hit24'].mean() if len(has_ras) > 0 else np.nan
        hit_no = no_ras['hit24'].mean() if len(no_ras) > 0 else np.nan
        print(f"    Round {rd}: {rd_missing}/{rd_total} missing ({pct:.0f}%) | "
              f"Hit rate with RAS: {hit_has:.2f}, without: {hit_no:.2f}" if not np.isnan(hit_no)
              else f"    Round {rd}: {rd_missing}/{rd_total} missing ({pct:.0f}%)")

# Early declare
n_ed = wr['comp_early_declare'].notna().sum()
n_miss_ed = len(wr) - n_ed
print(f"\n{'Early Declare':<25} {n_ed:>10} {n_miss_ed:>10} {n_ed/len(wr)*100:>13.1f}% {'Missing: draft_age data unavailable'}")

# Check which draft years are missing declare data
print(f"\n  Early declare missing by draft year:")
for yr in sorted(wr['draft_year'].unique()):
    yr_players = wr[wr['draft_year'] == yr]
    yr_missing = yr_players['comp_early_declare'].isna().sum()
    yr_total = len(yr_players)
    print(f"    {yr}: {yr_missing}/{yr_total} missing")

# Rushing
n_rush = wr['comp_rushing'].notna().sum()
n_miss_rush = len(wr) - n_rush
print(f"\n{'Rushing Production':<25} {n_rush:>10} {n_miss_rush:>10} {n_rush/len(wr)*100:>13.1f}% {'PFF college data: 2016-2025 only (no 2015)'}")

# Check which draft years have rushing
print(f"\n  Rushing data coverage by draft year:")
for yr in sorted(wr['draft_year'].unique()):
    yr_players = wr[wr['draft_year'] == yr]
    yr_has = yr_players['comp_rushing'].notna().sum()
    yr_total = len(yr_players)
    print(f"    {yr}: {yr_has}/{yr_total} have data")

# Outcomes
print(f"\n\n--- OUTCOME DATA COVERAGE ---")
for out in outcomes_list:
    n = wr[out].notna().sum()
    print(f"  {out:<20}: {n}/{len(wr)} ({n/len(wr)*100:.1f}%)")


# ============================================================================
# IMPUTATION RECOMMENDATIONS
# ============================================================================

print("\n\n" + "=" * 100)
print("IMPUTATION RECOMMENDATIONS")
print("=" * 100)

print("""
COMPONENT-BY-COMPONENT RECOMMENDATIONS:

1. DRAFT CAPITAL: No imputation needed (100% coverage)

2. BREAKOUT AGE: No imputation needed (100% coverage)
   - Players who never broke out already get a score (15-35 based on peak dominator)

3. TEAMMATE SCORE: No imputation needed (~100% coverage)
   - Players with 0 drafted teammates → score of 0 (no teammate adjustment)
   - This is the actual data, not a gap

4. RAS (ATHLETIC): NEEDS IMPUTATION""")
n_miss = wr['comp_ras'].isna().sum()
print(f"   - {n_miss} WRs missing ({n_miss/len(wr)*100:.1f}%)")
print(f"""   - CRITICAL: Missing data is NOT random (MNAR)
   - Round 1 missing = elite prospects who skip combine (very high hit rate)
   - Round 5-7 missing = players who didn't get combine invite (low hit rate)
   - CURRENT APPROACH: Use position average RAS for all missing
   - ALTERNATIVE: Use round-specific average RAS (better for MNAR)

5. EARLY DECLARE: NEEDS IMPUTATION""")
n_miss = wr['comp_early_declare'].isna().sum()
print(f"   - {n_miss} WRs missing ({n_miss/len(wr)*100:.1f}%)")
print(f"""   - Can be recovered: if we have draft_age and birthdate, we can calculate
   - Otherwise: impute with draft_age < 22 → EARLY, >= 23 → LATE, else STANDARD
   - OR: Use draft_age as continuous proxy (younger = higher score)

6. RUSHING PRODUCTION: NEEDS IMPUTATION""")
n_miss = wr['comp_rushing'].isna().sum()
print(f"   - {n_miss} WRs missing ({n_miss/len(wr)*100:.1f}%)")
print(f"""   - Missing entirely for 2015 draft class (PFF college data starts 2016)
   - OPTION A: Impute with 0 (assume non-rusher) — simple but may undercount
   - OPTION B: Impute with position average — assumes missing = average
   - OPTION C: Drop rushing component (use 5-component model)
   - RECOMMENDATION: If rushing shows weak partial correlation, consider dropping it
     rather than imputing 16% of the sample
""")

# HaSS note
print("\n--- NOTE ON HaSS (Height-adjusted Speed Score) ---")
print("HaSS requires height + weight + 40-time data for each WR.")
print("Current dataset does NOT have height data.")
print("RAS is available and is a more comprehensive athletic composite")
print("(includes 40, vertical, broad jump, 3-cone, shuttle, bench).")
print("If you want to test HaSS, we would need to source height data for all 339 WRs.")

print("\n\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
