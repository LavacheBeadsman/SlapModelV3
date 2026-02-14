"""
TE Correlation Analysis: Which metrics predict NFL success?

Tests every available metric against 3 NFL outcomes:
  - hit24: finished top-24 TE in a full NFL season (at least once)
  - hit12: finished top-12 TE in a full NFL season (at least once)
  - first_3yr_ppg: PPR points per game in first 3 NFL seasons

Two analyses:
  1. Raw correlations (bivariate)
  2. Partial correlations controlling for draft pick (the key test)

Does NOT pick model weights — just shows the data.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('data/te_backtest_master.csv')

# Only use TEs with NFL outcome data
df_with_outcomes = df[df['nfl_seasons_found'] > 0].copy()
print(f"Total TEs in backtest: {len(df)}")
print(f"TEs with NFL outcomes: {len(df_with_outcomes)}")
print(f"  hit24 rate: {df_with_outcomes['hit24'].mean():.1%} ({int(df_with_outcomes['hit24'].sum())}/{len(df_with_outcomes)})")
print(f"  hit12 rate: {df_with_outcomes['hit12'].mean():.1%} ({int(df_with_outcomes['hit12'].sum())}/{len(df_with_outcomes)})")
print(f"  first_3yr_ppg mean: {df_with_outcomes['first_3yr_ppg'].mean():.2f}")

# ============================================================
# BUILD DERIVED METRICS
# ============================================================
d = df_with_outcomes.copy()

# Draft capital score (gentler curve from CLAUDE.md)
d['dc_score'] = d['pick'].apply(lambda p: max(0, min(100, 100 - 2.40 * (p**0.62 - 1))))

# CFBD-derived metrics
d['ryptpa'] = np.where(
    (d['cfbd_rec_yards'].notna()) & (d['cfbd_team_pass_att'].notna()) & (d['cfbd_team_pass_att'] > 0),
    d['cfbd_rec_yards'] / d['cfbd_team_pass_att'],
    np.nan
)

# Dominator-like: rec_yards / team_rec_yards (if available)
d['cfbd_dominator'] = np.where(
    (d['cfbd_rec_yards'].notna()) & (d['cfbd_team_rec_yards'].notna()) & (d['cfbd_team_rec_yards'] > 0),
    d['cfbd_rec_yards'] / d['cfbd_team_rec_yards'] * 100,
    np.nan
)

# Receptions per team pass attempt
d['rec_per_tpa'] = np.where(
    (d['cfbd_receptions'].notna()) & (d['cfbd_team_pass_att'].notna()) & (d['cfbd_team_pass_att'] > 0),
    d['cfbd_receptions'] / d['cfbd_team_pass_att'],
    np.nan
)

# Speed Score (Barnwell formula) - for TEs with weight + forty
d['speed_score'] = np.where(
    (d['wt'].notna()) & (d['forty'].notna()) & (d['forty'] > 0),
    (d['wt'] * 200) / (d['forty'] ** 4),
    np.nan
)

# BMI-like: weight/height proxy (can't easily parse height string, skip)

# Age-weighted RYPTPA (same concept as RB model)
def age_weight(age):
    if pd.isna(age): return 1.0
    if age <= 19: return 1.15
    elif age <= 20: return 1.10
    elif age <= 21: return 1.05
    elif age <= 22: return 1.00
    elif age <= 23: return 0.95
    else: return 0.90

d['age_weighted_ryptpa'] = d.apply(
    lambda r: r['ryptpa'] * age_weight(r['draft_age']) if pd.notna(r['ryptpa']) else np.nan,
    axis=1
)

# PFF-derived ratios and composites
# Yards per route run is already there as pff_yprr
# Target share proxy: targets / routes
d['pff_target_rate'] = np.where(
    (d['pff_routes'].notna()) & (d['pff_routes'] > 0),
    d['pff_targets'] / d['pff_routes'],
    np.nan
)

# Yards per target
d['pff_yards_per_target'] = np.where(
    (d['pff_targets'].notna()) & (d['pff_targets'] > 0),
    d['pff_yards'] / d['pff_targets'],
    np.nan
)

# Receptions per game
d['pff_rec_per_game'] = np.where(
    (d['pff_player_game_count'].notna()) & (d['pff_player_game_count'] > 0),
    d['pff_receptions'] / d['pff_player_game_count'],
    np.nan
)

# Yards per game
d['pff_yards_per_game'] = np.where(
    (d['pff_player_game_count'].notna()) & (d['pff_player_game_count'] > 0),
    d['pff_yards'] / d['pff_player_game_count'],
    np.nan
)

# TD rate
d['pff_td_rate'] = np.where(
    (d['pff_targets'].notna()) & (d['pff_targets'] > 0),
    d['pff_touchdowns'] / d['pff_targets'],
    np.nan
)

# Contested catch success
# Already have pff_contested_catch_rate

# Non-inline rate (slot + wide)
d['pff_non_inline_rate'] = np.where(
    (d['pff_slot_rate'].notna()) & (d['pff_wide_rate'].notna()),
    d['pff_slot_rate'] + d['pff_wide_rate'],
    np.nan
)

# ============================================================
# DEFINE ALL METRICS TO TEST
# ============================================================

metrics = {
    # --- Draft Capital (baseline) ---
    'dc_score': 'DC Score (gentler curve)',
    'pick': 'Draft Pick (raw, lower=better)',
    'round': 'Draft Round',

    # --- Age ---
    'draft_age': 'Draft Age',
    'early_declare': 'Early Declare (binary)',

    # --- Athletic / Combine ---
    'te_ras': 'RAS (0-10 scale)',
    'wt': 'Weight (lbs)',
    'forty': '40-yard Dash (lower=better)',
    'speed_score': 'Speed Score (Barnwell)',
    'bench': 'Bench Press (reps)',
    'vertical': 'Vertical Jump (inches)',
    'broad_jump': 'Broad Jump (inches)',
    'cone': '3-Cone Drill (lower=better)',
    'shuttle': 'Shuttle (lower=better)',

    # --- Breakout / Dominator ---
    'breakout_age': 'Breakout Age (20% dom threshold)',
    'peak_dominator': 'Peak Dominator Rating (%)',

    # --- CFBD Production ---
    'cfbd_rec_yards': 'CFBD Receiving Yards (final season)',
    'cfbd_receptions': 'CFBD Receptions (final season)',
    'ryptpa': 'Rec Yards / Team Pass Att (RYPTPA)',
    'age_weighted_ryptpa': 'Age-Weighted RYPTPA',
    'cfbd_dominator': 'CFBD Dominator (rec_yds/team_rec_yds %)',
    'rec_per_tpa': 'Receptions / Team Pass Att',
    'cfbd_rush_yards': 'CFBD Rush Yards (final season)',

    # --- PFF Grades ---
    'pff_grades_offense': 'PFF Offense Grade',
    'pff_grades_pass_route': 'PFF Pass Route Grade',
    'pff_grades_pass_block': 'PFF Pass Block Grade',
    'pff_grades_hands_drop': 'PFF Hands/Drop Grade',
    'pff_grades_hands_fumble': 'PFF Hands/Fumble Grade',

    # --- PFF Production ---
    'pff_yprr': 'PFF Yards Per Route Run',
    'pff_yards': 'PFF Receiving Yards',
    'pff_receptions': 'PFF Receptions',
    'pff_targets': 'PFF Targets',
    'pff_touchdowns': 'PFF Touchdowns',
    'pff_first_downs': 'PFF First Downs',
    'pff_yards_per_game': 'PFF Yards Per Game',
    'pff_rec_per_game': 'PFF Receptions Per Game',

    # --- PFF Efficiency ---
    'pff_caught_percent': 'PFF Catch %',
    'pff_drop_rate': 'PFF Drop Rate (lower=better)',
    'pff_drops': 'PFF Drops (lower=better)',
    'pff_contested_catch_rate': 'PFF Contested Catch Rate',
    'pff_yards_per_reception': 'PFF Yards Per Reception',
    'pff_yards_after_catch_per_reception': 'PFF YAC Per Reception',
    'pff_yards_after_catch': 'PFF Yards After Catch',
    'pff_target_rate': 'PFF Target Rate (targets/routes)',
    'pff_yards_per_target': 'PFF Yards Per Target',
    'pff_td_rate': 'PFF TD Rate',
    'pff_targeted_qb_rating': 'PFF Targeted QB Rating',
    'pff_avg_depth_of_target': 'PFF Avg Depth of Target',
    'pff_avoided_tackles': 'PFF Avoided Tackles',
    'pff_longest': 'PFF Longest Reception',

    # --- PFF Alignment ---
    'pff_inline_rate': 'PFF Inline Rate (%)',
    'pff_slot_rate': 'PFF Slot Rate (%)',
    'pff_wide_rate': 'PFF Wide Rate (%)',
    'pff_non_inline_rate': 'PFF Non-Inline Rate (slot+wide)',
    'pff_route_rate': 'PFF Route Rate (%)',
    'pff_pass_block_rate': 'PFF Pass Block Rate (%)',

    # --- PFF Volume ---
    'pff_player_game_count': 'PFF Games Played',
    'pff_routes': 'PFF Total Routes',
    'pff_inline_snaps': 'PFF Inline Snaps',
    'pff_slot_snaps': 'PFF Slot Snaps',
    'pff_wide_snaps': 'PFF Wide Snaps',
    'pff_pass_blocks': 'PFF Pass Blocks',
    'pff_pass_plays': 'PFF Pass Plays',
    'pff_contested_targets': 'PFF Contested Targets',
    'pff_contested_receptions': 'PFF Contested Receptions',
    'pff_fumbles': 'PFF Fumbles (lower=better)',
}

# Metrics where LOWER is better (need to flip sign for interpretation)
lower_is_better = {'pick', 'round', 'draft_age', 'forty', 'cone', 'shuttle',
                    'breakout_age', 'pff_drop_rate', 'pff_drops', 'pff_fumbles'}

outcomes = ['hit24', 'hit12', 'first_3yr_ppg']

# ============================================================
# STEP 1: RAW (BIVARIATE) CORRELATIONS
# ============================================================
print("\n" + "="*100)
print("STEP 1: RAW CORRELATIONS (bivariate, no controls)")
print("="*100)

results = []
for metric_col, metric_name in metrics.items():
    if metric_col not in d.columns:
        continue

    row = {'metric': metric_col, 'name': metric_name}

    # Coverage
    valid = d[metric_col].notna()
    row['n'] = int(valid.sum())
    row['coverage'] = f"{row['n']}/{len(d)} ({100*row['n']/len(d):.0f}%)"

    if row['n'] < 10:
        row['r_hit24'] = np.nan
        row['r_hit12'] = np.nan
        row['r_ppg'] = np.nan
        row['p_ppg'] = np.nan
        results.append(row)
        continue

    subset = d[valid].copy()

    for outcome in outcomes:
        valid_outcome = subset[outcome].notna()
        s = subset[valid_outcome]
        if len(s) < 10:
            row[f'r_{outcome}'] = np.nan
            row[f'p_{outcome}'] = np.nan
            continue
        r, p = stats.pearsonr(s[metric_col].astype(float), s[outcome].astype(float))
        row[f'r_{outcome}'] = r
        row[f'p_{outcome}'] = p

    results.append(row)

raw_df = pd.DataFrame(results)

# Print top metrics by absolute correlation with first_3yr_ppg
print(f"\n{'Metric':<45} {'N':>5} {'Coverage':>12}  {'r(hit24)':>9} {'r(hit12)':>9} {'r(PPG)':>9} {'p(PPG)':>9}")
print("-"*105)

# Sort by absolute correlation with first_3yr_ppg
raw_df['abs_r_ppg'] = raw_df['r_first_3yr_ppg'].abs()
raw_sorted = raw_df.sort_values('abs_r_ppg', ascending=False)

for _, row in raw_sorted.iterrows():
    r_h24 = f"{row['r_hit24']:+.3f}" if pd.notna(row.get('r_hit24')) else '   N/A'
    r_h12 = f"{row['r_hit12']:+.3f}" if pd.notna(row.get('r_hit12')) else '   N/A'
    r_ppg = f"{row['r_first_3yr_ppg']:+.3f}" if pd.notna(row.get('r_first_3yr_ppg')) else '   N/A'
    p_ppg = f"{row['p_first_3yr_ppg']:.4f}" if pd.notna(row.get('p_first_3yr_ppg')) else '   N/A'
    print(f"{row['name']:<45} {row['n']:>5} {row['coverage']:>12}  {r_h24:>9} {r_h12:>9} {r_ppg:>9} {p_ppg:>9}")


# ============================================================
# STEP 2: PARTIAL CORRELATIONS (controlling for draft pick)
# ============================================================
print("\n\n" + "="*100)
print("STEP 2: PARTIAL CORRELATIONS (controlling for draft pick)")
print("This is the KEY test — what predicts success AFTER accounting for where the player was drafted?")
print("="*100)

def partial_correlation(x, y, z):
    """
    Partial correlation between x and y, controlling for z.
    All inputs are arrays/series. Returns (r_partial, p_value, n).
    """
    # Remove any rows with NaN in any of x, y, z
    mask = pd.notna(x) & pd.notna(y) & pd.notna(z)
    x = x[mask].astype(float).values
    y = y[mask].astype(float).values
    z = z[mask].astype(float).values

    n = len(x)
    if n < 10:
        return np.nan, np.nan, n

    # Regress x on z, get residuals
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    resid_x = x - (slope_xz * z + intercept_xz)

    # Regress y on z, get residuals
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    resid_y = y - (slope_yz * z + intercept_yz)

    # Correlate residuals
    r, p = stats.pearsonr(resid_x, resid_y)

    return r, p, n


partial_results = []
for metric_col, metric_name in metrics.items():
    if metric_col not in d.columns:
        continue
    if metric_col in ('pick', 'round', 'dc_score'):
        continue  # Can't partial out DC from itself

    row = {'metric': metric_col, 'name': metric_name}

    for outcome in outcomes:
        r_part, p_part, n_part = partial_correlation(
            d[metric_col], d[outcome], d['pick']
        )
        row[f'r_partial_{outcome}'] = r_part
        row[f'p_partial_{outcome}'] = p_part
        row[f'n_{outcome}'] = n_part

    # Also store coverage info
    valid = d[metric_col].notna()
    row['n_total'] = int(valid.sum())
    row['coverage_pct'] = 100 * row['n_total'] / len(d)

    partial_results.append(row)

partial_df = pd.DataFrame(partial_results)

# ============================================================
# STEP 3: CLEAN REPORT — TOP 15 by partial correlation with PPG
# ============================================================
print(f"\n{'='*120}")
print("TOP 15 METRICS BY PARTIAL CORRELATION WITH first_3yr_ppg (controlling for draft pick)")
print(f"{'='*120}")

partial_df['abs_r_partial_ppg'] = partial_df['r_partial_first_3yr_ppg'].abs()
top15 = partial_df.sort_values('abs_r_partial_ppg', ascending=False).head(15)

print(f"\n{'Rank':<5} {'Metric':<45} {'N':>4} {'Cov%':>5} {'r_part(PPG)':>12} {'p(PPG)':>9} {'r_part(h24)':>12} {'p(h24)':>9} {'r_part(h12)':>12} {'p(h12)':>9} {'70%+':>5}")
print("-"*130)

for rank, (_, row) in enumerate(top15.iterrows(), 1):
    r_ppg = f"{row['r_partial_first_3yr_ppg']:+.3f}" if pd.notna(row.get('r_partial_first_3yr_ppg')) else '  N/A'
    p_ppg = f"{row['p_partial_first_3yr_ppg']:.4f}" if pd.notna(row.get('p_partial_first_3yr_ppg')) else '  N/A'
    r_h24 = f"{row['r_partial_hit24']:+.3f}" if pd.notna(row.get('r_partial_hit24')) else '  N/A'
    p_h24 = f"{row['p_partial_hit24']:.4f}" if pd.notna(row.get('p_partial_hit24')) else '  N/A'
    r_h12 = f"{row['r_partial_hit12']:+.3f}" if pd.notna(row.get('r_partial_hit12')) else '  N/A'
    p_h12 = f"{row['p_partial_hit12']:.4f}" if pd.notna(row.get('p_partial_hit12')) else '  N/A'
    flag = 'YES' if row['coverage_pct'] >= 70 else 'no'
    print(f"{rank:<5} {row['name']:<45} {row.get('n_first_3yr_ppg', 0):>4} {row['coverage_pct']:>4.0f}% {r_ppg:>12} {p_ppg:>9} {r_h24:>12} {p_h24:>9} {r_h12:>12} {p_h12:>9} {flag:>5}")


# ============================================================
# FULL TABLE: ALL metrics sorted by partial correlation
# ============================================================
print(f"\n\n{'='*120}")
print("FULL TABLE: ALL METRICS BY PARTIAL CORRELATION WITH first_3yr_ppg (controlling for draft pick)")
print(f"{'='*120}")

all_sorted = partial_df.sort_values('abs_r_partial_ppg', ascending=False)

print(f"\n{'Rank':<5} {'Metric':<45} {'N':>4} {'Cov%':>5} {'r_part(PPG)':>12} {'p(PPG)':>9} {'Sig?':>5} {'70%+':>5}")
print("-"*95)

for rank, (_, row) in enumerate(all_sorted.iterrows(), 1):
    r_ppg = f"{row['r_partial_first_3yr_ppg']:+.3f}" if pd.notna(row.get('r_partial_first_3yr_ppg')) else '  N/A'
    p_ppg = f"{row['p_partial_first_3yr_ppg']:.4f}" if pd.notna(row.get('p_partial_first_3yr_ppg')) else '  N/A'
    p_val = row.get('p_partial_first_3yr_ppg', 1)
    sig = '***' if pd.notna(p_val) and p_val < 0.01 else ('**' if pd.notna(p_val) and p_val < 0.05 else ('*' if pd.notna(p_val) and p_val < 0.10 else ''))
    flag = 'YES' if row['coverage_pct'] >= 70 else 'no'
    print(f"{rank:<5} {row['name']:<45} {row.get('n_first_3yr_ppg', 0):>4} {row['coverage_pct']:>4.0f}% {r_ppg:>12} {p_ppg:>9} {sig:>5} {flag:>5}")


# ============================================================
# DRAFT CAPITAL BASELINE
# ============================================================
print(f"\n\n{'='*80}")
print("DRAFT CAPITAL BASELINE (for reference)")
print(f"{'='*80}")

for outcome in outcomes:
    valid = d['pick'].notna() & d[outcome].notna()
    s = d[valid]
    r, p = stats.pearsonr(s['dc_score'], s[outcome])
    print(f"  DC Score vs {outcome:15s}: r={r:+.3f}, p={p:.4f}, n={len(s)}")
    r2, p2 = stats.pearsonr(s['pick'], s[outcome])
    print(f"  Raw Pick vs {outcome:15s}: r={r2:+.3f}, p={p2:.4f}, n={len(s)}")

# ============================================================
# ALIGNMENT ANALYSIS (special section for inline vs slot vs wide)
# ============================================================
print(f"\n\n{'='*80}")
print("ALIGNMENT SPLIT ANALYSIS")
print("Does where a TE lined up in college predict NFL success?")
print(f"{'='*80}")

align_metrics = ['pff_inline_rate', 'pff_slot_rate', 'pff_wide_rate', 'pff_non_inline_rate', 'pff_route_rate']
for m in align_metrics:
    if m not in d.columns:
        continue
    valid = d[m].notna() & d['first_3yr_ppg'].notna() & d['pick'].notna()
    s = d[valid]
    if len(s) < 10:
        continue

    # Raw
    r_raw, p_raw = stats.pearsonr(s[m], s['first_3yr_ppg'])
    # Partial
    r_part, p_part, n = partial_correlation(d[m], d['first_3yr_ppg'], d['pick'])

    print(f"  {metrics.get(m, m):<40s}: raw r={r_raw:+.3f} (p={p_raw:.3f}), partial r={r_part:+.3f} (p={p_part:.3f}), n={n}")


# ============================================================
# BREAKOUT AGE ANALYSIS (only 20% have it — special section)
# ============================================================
print(f"\n\n{'='*80}")
print("BREAKOUT AGE ANALYSIS (only 20% of TEs hit 20% dominator)")
print(f"{'='*80}")

has_bo = d['breakout_age'].notna()
print(f"  TEs with breakout age: {has_bo.sum()}/{len(d)}")
if has_bo.sum() >= 10:
    s = d[has_bo & d['first_3yr_ppg'].notna()]
    r, p = stats.pearsonr(s['breakout_age'], s['first_3yr_ppg'])
    print(f"  Raw correlation with PPG: r={r:+.3f}, p={p:.4f}, n={len(s)}")

    # Compare: TEs who broke out vs those who didn't
    broke_out = d[d['breakout_age'].notna() & d['first_3yr_ppg'].notna()]
    never_broke = d[d['breakout_age'].isna() & d['peak_dominator'].notna() & d['first_3yr_ppg'].notna()]
    print(f"\n  TEs who broke out (hit 20% dom): n={len(broke_out)}, avg PPG={broke_out['first_3yr_ppg'].mean():.2f}, hit24={broke_out['hit24'].mean():.1%}")
    print(f"  TEs who never broke out:          n={len(never_broke)}, avg PPG={never_broke['first_3yr_ppg'].mean():.2f}, hit24={never_broke['hit24'].mean():.1%}")

    # Binary: did they break out at all?
    d['broke_out'] = d['breakout_age'].notna().astype(int)
    valid = d['first_3yr_ppg'].notna()
    r, p = stats.pearsonr(d.loc[valid, 'broke_out'], d.loc[valid, 'first_3yr_ppg'])
    print(f"\n  Binary (broke out yes/no) vs PPG: r={r:+.3f}, p={p:.4f}")
    r_part, p_part, n = partial_correlation(d['broke_out'], d['first_3yr_ppg'], d['pick'])
    print(f"  Binary partial (controlling DC):   r={r_part:+.3f}, p={p_part:.4f}, n={n}")


# ============================================================
# PEAK DOMINATOR ANALYSIS (70% coverage — more usable than breakout age)
# ============================================================
print(f"\n\n{'='*80}")
print("PEAK DOMINATOR ANALYSIS (70% coverage)")
print(f"{'='*80}")

has_dom = d['peak_dominator'].notna() & d['first_3yr_ppg'].notna()
s = d[has_dom]
print(f"  TEs with peak dominator: {has_dom.sum()}/{len(d)}")
r_raw, p_raw = stats.pearsonr(s['peak_dominator'], s['first_3yr_ppg'])
print(f"  Raw correlation with PPG: r={r_raw:+.3f}, p={p_raw:.4f}")
r_part, p_part, n = partial_correlation(d['peak_dominator'], d['first_3yr_ppg'], d['pick'])
print(f"  Partial (controlling DC): r={r_part:+.3f}, p={p_part:.4f}, n={n}")

# Dominator bins
print(f"\n  Peak dominator bins:")
bins = [0, 10, 15, 20, 25, 100]
labels = ['0-10%', '10-15%', '15-20%', '20-25%', '25%+']
s_binned = s.copy()
s_binned['dom_bin'] = pd.cut(s_binned['peak_dominator'], bins=bins, labels=labels, right=False)
for label in labels:
    group = s_binned[s_binned['dom_bin'] == label]
    if len(group) > 0:
        print(f"    {label:>8s}: n={len(group):>3}, avg PPG={group['first_3yr_ppg'].mean():.2f}, "
              f"hit24={group['hit24'].mean():.0%}, avg pick={group['pick'].mean():.0f}")


# ============================================================
# COVERAGE SUMMARY
# ============================================================
print(f"\n\n{'='*80}")
print("DATA COVERAGE SUMMARY (for model-building feasibility)")
print(f"{'='*80}")

coverage_groups = {
    'Draft/Age': ['dc_score', 'draft_age', 'early_declare'],
    'Athletic': ['te_ras', 'wt', 'forty', 'speed_score', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle'],
    'CFBD Production': ['cfbd_rec_yards', 'cfbd_receptions', 'ryptpa', 'cfbd_dominator', 'cfbd_rush_yards'],
    'PFF Grades': ['pff_grades_offense', 'pff_grades_pass_route', 'pff_grades_pass_block'],
    'PFF Production': ['pff_yprr', 'pff_yards', 'pff_receptions', 'pff_targets'],
    'PFF Alignment': ['pff_inline_rate', 'pff_slot_rate', 'pff_wide_rate', 'pff_route_rate'],
    'Breakout': ['breakout_age', 'peak_dominator'],
}

for group_name, cols in coverage_groups.items():
    print(f"\n  {group_name}:")
    for col in cols:
        if col in d.columns:
            n = d[col].notna().sum()
            pct = 100 * n / len(d)
            flag = "OK" if pct >= 70 else "LOW"
            print(f"    {metrics.get(col, col):<45s}: {n:>3}/{len(d)} ({pct:>4.0f}%) [{flag}]")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE — Review results above before picking model components.")
print(f"{'='*80}")
