"""
DIAGNOSTIC: RB Production Metric Comparison
============================================
Tests 7 alternative production metrics against NFL outcomes using
partial correlations controlling for draft capital.

Metrics tested:
  1. FINAL season Rec Yards / Team Pass Att (current metric)
  2. BEST season Rec Yards / Team Pass Att
  3. CAREER AVG Rec Yards / Team Pass Att
  4. FINAL season Receptions / Team Pass Att
  5. BEST season Receptions / Team Pass Att (final-season only — no multi-season rec data)
  6. CAREER AVG Receptions / Team Pass Att (final-season only — no multi-season rec data)
  7. EARLIEST receiving "breakout" season (youngest age with significant receiving production)

DOES NOT CHANGE ANYTHING — diagnostic output only.
"""

import pandas as pd
import numpy as np
import warnings, os
from scipy import stats as sp_stats
from datetime import datetime
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')


# ============================================================================
# HELPERS
# ============================================================================
def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()

def partial_corr(x, y, z):
    """Partial correlation of x and y, controlling for z.
    Returns (r, p-value, n)."""
    mask = x.notna() & y.notna() & z.notna()
    x2, y2, z2 = x[mask].values, y[mask].values, z[mask].values
    n = len(x2)
    if n < 10:
        return np.nan, np.nan, n
    # Residualize x on z
    slope_xz = np.polyfit(z2, x2, 1)
    res_x = x2 - np.polyval(slope_xz, z2)
    # Residualize y on z
    slope_yz = np.polyfit(z2, y2, 1)
    res_y = y2 - np.polyval(slope_yz, z2)
    # Correlate residuals
    r, p = sp_stats.pearsonr(res_x, res_y)
    return r, p, n


# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 120)
print("DIAGNOSTIC: RB Production Metric Comparison")
print("=" * 120)

# --- RB backtest (final season data + outcomes) ---
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
# rb_backtest already has hit24, hit12 — only need first_3yr_ppg, career_ppg from outcomes
rb_out = outcomes[outcomes['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']
].copy()
rb_bt = rb_bt.merge(rb_out, on=['player_name', 'draft_year', 'pick'], how='left')
rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)
rb_bt['name_norm'] = rb_bt['player_name'].apply(normalize_name)

print(f"\n  RB backtest: {len(rb_bt)} players (2015-2024 drafts)")
print(f"  Outcomes available: hit24={rb_bt['hit24'].notna().sum()}, hit12={rb_bt['hit12'].notna().sum()}, "
      f"first_3yr_ppg={rb_bt['first_3yr_ppg'].notna().sum()}, career_ppg={rb_bt['career_ppg'].notna().sum()}")

# --- Multi-season college receiving data ---
college = pd.read_csv('data/college_receiving_2011_2023.csv')
college_rb = college[college['position'] == 'RB'].copy()
college_rb['name_norm'] = college_rb['college_name'].apply(normalize_name)
college_rb['rec_yards'] = pd.to_numeric(college_rb['rec_yards'], errors='coerce')
college_rb['team_pass_att'] = pd.to_numeric(college_rb['team_pass_att'], errors='coerce')
college_rb['college_season'] = pd.to_numeric(college_rb['college_season'], errors='coerce')

print(f"  Multi-season college data: {len(college_rb)} RB season-rows")

# --- Birthdates for season age ---
bdays = pd.read_csv('data/nflverse_birthdates_2015_2025.csv')
bdays_rb = bdays[bdays['position'] == 'RB'].copy()
bdays_rb['name_norm'] = bdays_rb['nfl_name'].apply(normalize_name)
bdays_rb['birth_date'] = pd.to_datetime(bdays_rb['birth_date'], errors='coerce')
bday_lookup = dict(zip(bdays_rb['name_norm'], bdays_rb['birth_date']))

print(f"  Birthdates: {len(bdays_rb)} RBs")


# ============================================================================
# MATCH MULTI-SEASON DATA TO BACKTEST RBs
# ============================================================================
print(f"\n\n{'='*120}")
print("STEP 1: Match multi-season college data to backtest RBs")
print("=" * 120)

# For each backtest RB, find all their college seasons
matched_seasons = []
unmatched = []

for _, rb in rb_bt.iterrows():
    name_n = rb['name_norm']
    dy = int(rb['draft_year'])

    # Match by normalized name — seasons must be before draft year
    player_seasons = college_rb[
        (college_rb['name_norm'] == name_n) &
        (college_rb['college_season'] < dy) &
        (college_rb['team_pass_att'].notna()) &
        (college_rb['team_pass_att'] > 0)
    ].copy()

    if len(player_seasons) == 0:
        # Try matching with college name
        cfbd_name = rb.get('cfbd_name', '')
        if pd.notna(cfbd_name) and cfbd_name != '':
            cfbd_norm = normalize_name(cfbd_name)
            player_seasons = college_rb[
                (college_rb['name_norm'] == cfbd_norm) &
                (college_rb['college_season'] < dy) &
                (college_rb['team_pass_att'].notna()) &
                (college_rb['team_pass_att'] > 0)
            ].copy()

    if len(player_seasons) > 0:
        player_seasons = player_seasons.sort_values('college_season')
        player_seasons['player_name'] = rb['player_name']
        player_seasons['draft_year'] = dy
        player_seasons['pick'] = rb['pick']

        # Add season age using birthdate
        bday = bday_lookup.get(name_n)
        if pd.notna(bday):
            # Season age = age as of Sept 1 of that college season
            player_seasons['season_age'] = player_seasons['college_season'].apply(
                lambda s: (datetime(int(s), 9, 1) - bday).days / 365.25
            )
        else:
            # Estimate from draft age
            draft_age = rb['age']
            if pd.notna(draft_age):
                player_seasons['season_age'] = player_seasons['college_season'].apply(
                    lambda s: float(draft_age) - (dy - s)
                )
            else:
                player_seasons['season_age'] = np.nan

        matched_seasons.append(player_seasons)
    else:
        unmatched.append(rb['player_name'])

if matched_seasons:
    all_seasons = pd.concat(matched_seasons, ignore_index=True)
else:
    all_seasons = pd.DataFrame()

# Count matches
matched_players = all_seasons['player_name'].nunique() if len(all_seasons) > 0 else 0
print(f"\n  Matched: {matched_players}/{len(rb_bt)} backtest RBs have multi-season data")
print(f"  Unmatched: {len(unmatched)} RBs")
if len(unmatched) <= 20:
    print(f"  Unmatched names: {', '.join(unmatched[:20])}")

# Show season distribution
if len(all_seasons) > 0:
    seasons_per_player = all_seasons.groupby('player_name').size()
    print(f"\n  Seasons per matched player:")
    print(f"    1 season:  {(seasons_per_player == 1).sum()}")
    print(f"    2 seasons: {(seasons_per_player == 2).sum()}")
    print(f"    3 seasons: {(seasons_per_player == 3).sum()}")
    print(f"    4+ seasons: {(seasons_per_player >= 4).sum()}")


# ============================================================================
# COMPUTE AGE WEIGHT
# ============================================================================
def age_weight(season_age):
    """Same age weight function used in current RB SLAP."""
    if pd.isna(season_age):
        return 1.0
    sa = float(season_age)
    if sa <= 19: return 1.15
    elif sa <= 20: return 1.10
    elif sa <= 21: return 1.05
    elif sa <= 22: return 1.00
    elif sa <= 23: return 0.95
    else: return 0.90


# ============================================================================
# BUILD ALL 7 METRICS
# ============================================================================
print(f"\n\n{'='*120}")
print("STEP 2: Compute all 7 production metrics for backtest RBs")
print("=" * 120)

# Create a metrics dataframe indexed by player
metrics = rb_bt[['player_name', 'draft_year', 'pick', 's_dc', 'age',
                  'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg',
                  'rec_yards', 'receptions', 'team_pass_att']].copy()

# Ensure numeric
metrics['rec_yards'] = pd.to_numeric(metrics['rec_yards'], errors='coerce')
metrics['receptions'] = pd.to_numeric(metrics['receptions'], errors='coerce')
metrics['team_pass_att'] = pd.to_numeric(metrics['team_pass_att'], errors='coerce')
metrics['age'] = pd.to_numeric(metrics['age'], errors='coerce')

# ── METRIC 1: FINAL season Rec Yards / TPA (age-weighted) ──
# This is what we currently use
metrics['m1_final_ryTPA'] = np.where(
    (metrics['rec_yards'].notna()) & (metrics['team_pass_att'].notna()) & (metrics['team_pass_att'] > 0),
    (metrics['rec_yards'] / metrics['team_pass_att']) * metrics['age'].apply(
        lambda a: age_weight(float(a) - 1 if pd.notna(a) else 22)
    ) * 100,
    np.nan
)

# ── METRIC 4: FINAL season Receptions / TPA (age-weighted) ──
metrics['m4_final_recTPA'] = np.where(
    (metrics['receptions'].notna()) & (metrics['team_pass_att'].notna()) & (metrics['team_pass_att'] > 0),
    (metrics['receptions'] / metrics['team_pass_att']) * metrics['age'].apply(
        lambda a: age_weight(float(a) - 1 if pd.notna(a) else 22)
    ) * 100,
    np.nan
)

# For multi-season metrics, compute per player from all_seasons
if len(all_seasons) > 0:
    all_seasons['ry_tpa'] = all_seasons['rec_yards'] / all_seasons['team_pass_att']
    all_seasons['aw'] = all_seasons['season_age'].apply(age_weight)
    all_seasons['ry_tpa_aw'] = all_seasons['ry_tpa'] * all_seasons['aw'] * 100

    # ── METRIC 2: BEST season Rec Yards / TPA (age-weighted) ──
    best_season = all_seasons.groupby('player_name')['ry_tpa_aw'].max().reset_index()
    best_season.columns = ['player_name', 'm2_best_ryTPA']
    metrics = metrics.merge(best_season, on='player_name', how='left')

    # ── METRIC 3: CAREER AVG Rec Yards / TPA ──
    # Total career rec yards / total career team pass att (weighted by age)
    career = all_seasons.groupby('player_name').agg(
        total_ry=('rec_yards', 'sum'),
        total_tpa=('team_pass_att', 'sum'),
        mean_aw=('aw', 'mean')
    ).reset_index()
    career['m3_career_ryTPA'] = (career['total_ry'] / career['total_tpa']) * career['mean_aw'] * 100
    metrics = metrics.merge(career[['player_name', 'm3_career_ryTPA']], on='player_name', how='left')

    # ── METRIC 7: EARLIEST RECEIVING BREAKOUT ──
    # Define breakout: rec_yards / team_pass_att >= 0.05 (5% of team's passes caught as receiving yards)
    # This is roughly the 60th percentile for RBs — meaningful production
    BREAKOUT_THRESHOLD = 0.04  # 4% receiving share
    all_seasons['broke_out'] = all_seasons['ry_tpa'] >= BREAKOUT_THRESHOLD

    breakout_seasons = all_seasons[all_seasons['broke_out']].copy()
    if len(breakout_seasons) > 0:
        # Earliest breakout = youngest season age
        earliest_bo = breakout_seasons.groupby('player_name')['season_age'].min().reset_index()
        earliest_bo.columns = ['player_name', 'm7_breakout_age']
        metrics = metrics.merge(earliest_bo, on='player_name', how='left')
    else:
        metrics['m7_breakout_age'] = np.nan

    # For players with multi-season data but no breakout, mark as "never broke out"
    multi_players = set(all_seasons['player_name'].unique())
    for idx in metrics.index:
        pn = metrics.loc[idx, 'player_name']
        if pn in multi_players and pd.isna(metrics.loc[idx, 'm7_breakout_age']):
            metrics.loc[idx, 'm7_breakout_age'] = 99  # sentinel for "never broke out"
else:
    metrics['m2_best_ryTPA'] = np.nan
    metrics['m3_career_ryTPA'] = np.nan
    metrics['m7_breakout_age'] = np.nan

# ── METRICS 5 & 6: Receptions / TPA (FINAL season only — no multi-season rec data) ──
# Same as metric 4 since we only have final-season receptions
metrics['m5_best_recTPA'] = metrics['m4_final_recTPA']  # same data (final season only)
metrics['m6_career_recTPA'] = metrics['m4_final_recTPA']  # same data (final season only)

# Report coverage
print(f"\n  Metric coverage:")
for col, label in [
    ('m1_final_ryTPA', '1. FINAL Rec Yards / TPA'),
    ('m2_best_ryTPA', '2. BEST Rec Yards / TPA'),
    ('m3_career_ryTPA', '3. CAREER AVG Rec Yards / TPA'),
    ('m4_final_recTPA', '4. FINAL Receptions / TPA'),
    ('m5_best_recTPA', '5. BEST Receptions / TPA (=final, no multi-season rec data)'),
    ('m6_career_recTPA', '6. CAREER AVG Receptions / TPA (=final, no multi-season rec data)'),
    ('m7_breakout_age', '7. Earliest receiving breakout age'),
]:
    valid = metrics[col].notna()
    if col == 'm7_breakout_age':
        # Don't count the sentinel 99
        real_bo = ((metrics[col].notna()) & (metrics[col] < 99)).sum()
        never = (metrics[col] == 99).sum()
        no_data = metrics[col].isna().sum()
        print(f"    {label}: {real_bo} broke out, {never} never broke out, {no_data} no data")
    else:
        print(f"    {label}: {valid.sum()}/{len(metrics)} ({valid.mean()*100:.0f}%)")


# ============================================================================
# STEP 3: DESCRIPTIVE STATS FOR EACH METRIC
# ============================================================================
print(f"\n\n{'='*120}")
print("STEP 3: Descriptive statistics for each metric")
print("=" * 120)

print(f"\n  {'Metric':<45} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>5}")
print(f"  {'-'*90}")
for col, label in [
    ('m1_final_ryTPA', '1. FINAL RY/TPA (current)'),
    ('m2_best_ryTPA', '2. BEST RY/TPA'),
    ('m3_career_ryTPA', '3. CAREER AVG RY/TPA'),
    ('m4_final_recTPA', '4. FINAL Rec/TPA'),
    ('m7_breakout_age', '7. Breakout age (lower=better)'),
]:
    vals = metrics[col].dropna()
    if col == 'm7_breakout_age':
        vals = vals[vals < 99]  # exclude sentinel
    if len(vals) > 0:
        print(f"  {label:<45} {vals.mean():>8.2f} {vals.median():>8.2f} {vals.std():>8.2f} "
              f"{vals.min():>8.2f} {vals.max():>8.2f} {len(vals):>5}")


# ============================================================================
# STEP 4: PARTIAL CORRELATIONS CONTROLLING FOR DC
# ============================================================================
print(f"\n\n{'='*120}")
print("STEP 4: Partial correlations controlling for Draft Capital")
print("=" * 120)
print(f"\n  All correlations are PARTIAL r — the correlation between the metric and NFL outcome")
print(f"  AFTER removing the effect of draft capital. This tells us what the metric adds")
print(f"  beyond where the player was drafted.")

outcome_cols = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
outcome_labels = ['Top-24 PPR', 'Top-12 PPR', 'First 3yr PPG', 'Career PPG']

metric_cols = ['m1_final_ryTPA', 'm2_best_ryTPA', 'm3_career_ryTPA',
               'm4_final_recTPA', 'm7_breakout_age']
metric_labels = [
    '1. FINAL RY/TPA (current)',
    '2. BEST season RY/TPA',
    '3. CAREER AVG RY/TPA',
    '4. FINAL Rec/TPA',
    '7. Breakout age (inverted)',
]

# For breakout age, lower is better — so we invert it for correlation
# (younger breakout → higher score → should positively correlate with outcomes)
metrics['m7_breakout_inv'] = np.where(
    (metrics['m7_breakout_age'].notna()) & (metrics['m7_breakout_age'] < 99),
    -metrics['m7_breakout_age'],  # Invert: lower age = higher value
    np.nan  # NaN for "never broke out" and missing
)
# Also test including "never broke out" as worst
metrics['m7_breakout_inv_full'] = np.where(
    metrics['m7_breakout_age'].notna(),
    np.where(metrics['m7_breakout_age'] == 99, -30, -metrics['m7_breakout_age']),  # Never = age 30 (worst)
    np.nan
)

# Update lists to include both breakout variants
metric_cols_full = ['m1_final_ryTPA', 'm2_best_ryTPA', 'm3_career_ryTPA',
                    'm4_final_recTPA', 'm7_breakout_inv', 'm7_breakout_inv_full']
metric_labels_full = [
    '1. FINAL RY/TPA (current)',
    '2. BEST season RY/TPA',
    '3. CAREER AVG RY/TPA',
    '4. FINAL Rec/TPA',
    '7a. Breakout age (broke-out only)',
    '7b. Breakout age (incl never)',
]

# Print header
header = f"  {'Metric':<38}"
for ol in outcome_labels:
    header += f" | {'r':>6} {'p':>7} {'n':>4}"
print(f"\n{header}")
print(f"  {'':38}" + " | ".join([f"  ── {ol} ──" for ol in outcome_labels]))
print(f"  {'-' * 120}")

for mc, ml in zip(metric_cols_full, metric_labels_full):
    row = f"  {ml:<38}"
    for oc in outcome_cols:
        r, p, n = partial_corr(metrics[mc], metrics[oc], metrics['s_dc'])
        if np.isnan(r):
            row += f" | {'n/a':>6} {'n/a':>7} {n:>4}"
        else:
            p_str = f"{p:.4f}" if p >= 0.0001 else "<.0001"
            sig = " **" if p < 0.01 else " *" if p < 0.05 else "  "
            row += f" | {r:>+6.3f} {p_str:>7}{sig} {n:>3}"
    print(row)

# Also show simple (non-partial) correlations for reference
print(f"\n\n  ── Simple (bivariate) correlations for reference ──")
print(f"  (NOT controlling for DC — includes the DC signal)")
header2 = f"  {'Metric':<38}"
for ol in outcome_labels:
    header2 += f" | {'r':>6} {'p':>7} {'n':>4}"
print(f"\n{header2}")
print(f"  {'':38}" + " | ".join([f"  ── {ol} ──" for ol in outcome_labels]))
print(f"  {'-' * 120}")

for mc, ml in zip(metric_cols_full, metric_labels_full):
    row = f"  {ml:<38}"
    for oc in outcome_cols:
        mask = metrics[mc].notna() & metrics[oc].notna()
        x, y = metrics.loc[mask, mc], metrics.loc[mask, oc]
        n = len(x)
        if n < 10:
            row += f" | {'n/a':>6} {'n/a':>7} {n:>4}"
        else:
            r, p = sp_stats.pearsonr(x, y)
            p_str = f"{p:.4f}" if p >= 0.0001 else "<.0001"
            sig = " **" if p < 0.01 else " *" if p < 0.05 else "  "
            row += f" | {r:>+6.3f} {p_str:>7}{sig} {n:>3}"
    print(row)


# ============================================================================
# STEP 5: HEAD-TO-HEAD COMPARISON (BEST vs FINAL vs CAREER)
# ============================================================================
print(f"\n\n{'='*120}")
print("STEP 5: Head-to-head — How often does BEST season differ from FINAL season?")
print("=" * 120)

if 'm2_best_ryTPA' in metrics.columns:
    both = metrics[metrics['m1_final_ryTPA'].notna() & metrics['m2_best_ryTPA'].notna()].copy()
    both['best_is_final'] = np.isclose(both['m1_final_ryTPA'], both['m2_best_ryTPA'], rtol=0.01)
    both['best_higher'] = both['m2_best_ryTPA'] > both['m1_final_ryTPA'] * 1.01
    both['final_higher'] = both['m1_final_ryTPA'] > both['m2_best_ryTPA'] * 1.01

    print(f"\n  Of {len(both)} RBs with both metrics:")
    print(f"    Best = Final:         {both['best_is_final'].sum()} ({both['best_is_final'].mean()*100:.0f}%)")
    print(f"    Best > Final:         {both['best_higher'].sum()} ({both['best_higher'].mean()*100:.0f}%)")
    print(f"    Final > Best:         {both['final_higher'].sum()} ({both['final_higher'].mean()*100:.0f}%)")

    # When best > final — are those players better or worse?
    if both['best_higher'].sum() >= 5:
        best_won = both[both['best_higher']]
        final_won = both[both['best_is_final'] | both['final_higher']]
        print(f"\n    When BEST > FINAL ({len(best_won)} players):")
        print(f"      hit24 rate: {best_won['hit24'].mean()*100:.1f}%")
        print(f"      avg first_3yr_ppg: {best_won['first_3yr_ppg'].mean():.1f}")
        print(f"    When BEST = FINAL ({len(final_won)} players):")
        print(f"      hit24 rate: {final_won['hit24'].mean()*100:.1f}%")
        print(f"      avg first_3yr_ppg: {final_won['first_3yr_ppg'].mean():.1f}")

    # Show biggest differences
    both['diff'] = both['m2_best_ryTPA'] - both['m1_final_ryTPA']
    biggest_diff = both.nlargest(10, 'diff')
    print(f"\n  Biggest differences (BEST minus FINAL):")
    print(f"  {'Player':<25} {'Year':>4} {'Pick':>4} {'Final':>7} {'Best':>7} {'Diff':>6} {'Hit24':>5}")
    print(f"  {'-'*65}")
    for _, r in biggest_diff.iterrows():
        h = int(r['hit24']) if pd.notna(r['hit24']) else '?'
        print(f"  {r['player_name']:<25} {int(r['draft_year']):>4} {int(r['pick']):>4} "
              f"{r['m1_final_ryTPA']:>7.1f} {r['m2_best_ryTPA']:>7.1f} {r['diff']:>+6.1f} {h:>5}")


# ============================================================================
# STEP 6: CAREER AVG vs FINAL — do career averages add signal?
# ============================================================================
print(f"\n\n{'='*120}")
print("STEP 6: Career AVG vs Final — correlation between the two metrics")
print("=" * 120)

if 'm3_career_ryTPA' in metrics.columns:
    both_c = metrics[metrics['m1_final_ryTPA'].notna() & metrics['m3_career_ryTPA'].notna()].copy()
    if len(both_c) > 10:
        r, p = sp_stats.pearsonr(both_c['m1_final_ryTPA'], both_c['m3_career_ryTPA'])
        print(f"\n  Correlation between FINAL and CAREER AVG: r={r:.3f} (p={p:.4f}), n={len(both_c)}")
        print(f"  → {'High correlation — they measure nearly the same thing' if r > 0.7 else 'Moderate correlation — they differ meaningfully' if r > 0.4 else 'Low correlation — they measure different things'}")

    both_c['diff'] = both_c['m3_career_ryTPA'] - both_c['m1_final_ryTPA']
    biggest_diff_c = both_c.nlargest(10, 'diff')
    print(f"\n  Players where CAREER AVG >> FINAL (improved late-career vs. early):")
    print(f"  {'Player':<25} {'Year':>4} {'Pick':>4} {'Final':>7} {'Career':>7} {'Diff':>6} {'Hit24':>5}")
    print(f"  {'-'*65}")
    for _, r in biggest_diff_c.iterrows():
        h = int(r['hit24']) if pd.notna(r['hit24']) else '?'
        print(f"  {r['player_name']:<25} {int(r['draft_year']):>4} {int(r['pick']):>4} "
              f"{r['m1_final_ryTPA']:>7.1f} {r['m3_career_ryTPA']:>7.1f} {r['diff']:>+6.1f} {h:>5}")


# ============================================================================
# STEP 7: Rec Yards vs Receptions — which denominator works better?
# ============================================================================
print(f"\n\n{'='*120}")
print("STEP 7: Rec Yards / TPA vs Receptions / TPA (final season)")
print("=" * 120)

both_yr = metrics[metrics['m1_final_ryTPA'].notna() & metrics['m4_final_recTPA'].notna()].copy()
if len(both_yr) > 10:
    r, p = sp_stats.pearsonr(both_yr['m1_final_ryTPA'], both_yr['m4_final_recTPA'])
    print(f"\n  Correlation between RY/TPA and Rec/TPA: r={r:.3f} (p={p:.4f}), n={len(both_yr)}")

    print(f"\n  Partial r controlling for DC (comparison):")
    for oc, ol in zip(outcome_cols, outcome_labels):
        r1, p1, n1 = partial_corr(metrics['m1_final_ryTPA'], metrics[oc], metrics['s_dc'])
        r4, p4, n4 = partial_corr(metrics['m4_final_recTPA'], metrics[oc], metrics['s_dc'])
        winner = "RY/TPA" if abs(r1) > abs(r4) else "Rec/TPA" if abs(r4) > abs(r1) else "TIE"
        print(f"    {ol:<15}: RY/TPA r={r1:+.3f} (p={p1:.3f}) vs Rec/TPA r={r4:+.3f} (p={p4:.3f}) → {winner}")


# ============================================================================
# STEP 8: BREAKOUT AGE ANALYSIS
# ============================================================================
print(f"\n\n{'='*120}")
print("STEP 8: RB Receiving Breakout Age Analysis")
print("=" * 120)

if len(all_seasons) > 0:
    # Show breakout threshold sensitivity
    print(f"\n  Testing different breakout thresholds (RY/TPA ratio):")
    print(f"  {'Threshold':>10} {'Broke out':>10} {'Never':>8} {'No data':>8} {'Partial r (hit24)':>18}")
    print(f"  {'-'*60}")

    for thresh in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        # Recompute breakout with this threshold
        all_seasons[f'bo_{thresh}'] = all_seasons['ry_tpa'] >= thresh
        bo_df = all_seasons[all_seasons[f'bo_{thresh}']].groupby('player_name')['season_age'].min().reset_index()
        bo_df.columns = ['player_name', f'bo_age_{thresh}']

        test = metrics[['player_name', 'hit24', 's_dc']].merge(bo_df, on='player_name', how='left')
        # Multi-season players who never broke out
        for idx in test.index:
            pn = test.loc[idx, 'player_name']
            if pn in multi_players and pd.isna(test.loc[idx, f'bo_age_{thresh}']):
                test.loc[idx, f'bo_age_{thresh}'] = 30  # never

        valid = test[test[f'bo_age_{thresh}'].notna()].copy()
        broke = (valid[f'bo_age_{thresh}'] < 30).sum()
        never = (valid[f'bo_age_{thresh}'] == 30).sum()
        no_data = len(metrics) - len(valid)

        if broke >= 10:
            r, p, n = partial_corr(-valid[f'bo_age_{thresh}'], valid['hit24'], valid['s_dc'])
            p_str = f"r={r:+.3f} (p={p:.3f})" if not np.isnan(r) else "n/a"
        else:
            p_str = "too few"

        print(f"  {thresh:>10.2f} {broke:>10} {never:>8} {no_data:>8} {p_str:>18}")


# ============================================================================
# FINAL SUMMARY TABLE
# ============================================================================
print(f"\n\n{'='*120}")
print("FINAL SUMMARY: Which RB production metric is best?")
print("=" * 120)

print(f"""
  KEY: ** = p < 0.01, * = p < 0.05
  Positive partial r = metric adds predictive value beyond draft capital
  Higher |r| = stronger signal

  RECOMMENDATION CRITERIA:
  - Strong partial correlation with hit24 (binary success)
  - Significant partial correlation with first_3yr_ppg (continuous outcome)
  - Good coverage (available for most backtest players)
  - Methodological consistency (no cherry-picking)
""")

# Final clean table
print(f"  {'Metric':<38} {'Coverage':>8} | {'hit24 r':>8} {'p':>7} | {'3yr PPG r':>9} {'p':>7} | {'Career r':>8} {'p':>7}")
print(f"  {'-'*105}")

final_metrics = [
    ('m1_final_ryTPA', '1. FINAL RY/TPA (current)'),
    ('m2_best_ryTPA', '2. BEST season RY/TPA'),
    ('m3_career_ryTPA', '3. CAREER AVG RY/TPA'),
    ('m4_final_recTPA', '4. FINAL Rec/TPA'),
    ('m7_breakout_inv_full', '7. Breakout age (incl never)'),
]

for mc, ml in final_metrics:
    cov = metrics[mc].notna().sum()
    cov_str = f"{cov}/{len(metrics)}"

    results = []
    for oc in ['hit24', 'first_3yr_ppg', 'career_ppg']:
        r, p, n = partial_corr(metrics[mc], metrics[oc], metrics['s_dc'])
        if np.isnan(r):
            results.append(('n/a', 'n/a'))
        else:
            sig = "**" if p < 0.01 else " *" if p < 0.05 else "  "
            results.append((f"{r:+.3f}", f"{p:.4f}{sig}"))

    print(f"  {ml:<38} {cov_str:>8} | {results[0][0]:>8} {results[0][1]:>7} | "
          f"{results[1][0]:>9} {results[1][1]:>7} | {results[2][0]:>8} {results[2][1]:>7}")

print(f"\n  NOTE: Metrics 5 & 6 (BEST/CAREER Receptions/TPA) not shown separately —")
print(f"  multi-season receptions data not available in college_receiving_2011_2023.csv.")
print(f"  Only final-season receptions exist (= same as metric 4).")
print(f"\n  Nothing changed. This is diagnostic only.")
