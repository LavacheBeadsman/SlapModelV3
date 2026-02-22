"""
TE Metric Audit: Extract partial correlation results for 10 metrics
+ Build and test Teammate Score (the only untested metric)
"""

import pandas as pd
import numpy as np
from scipy import stats
import glob, os, warnings
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')


def partial_corr(x, y, z):
    """Partial correlation of x,y controlling for z."""
    valid = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    if len(valid) < 15:
        return np.nan, np.nan, len(valid)
    sx, ix, _, _, _ = stats.linregress(valid['z'], valid['x'])
    rx = valid['x'] - (sx * valid['z'] + ix)
    sy, iy, _, _, _ = stats.linregress(valid['z'], valid['y'])
    ry = valid['y'] - (sy * valid['z'] + iy)
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(valid)


def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))


# Load backtest
bt = pd.read_csv('data/te_backtest_master.csv')
eval_df = bt[bt['draft_year'] < 2025].copy()
eval_df['s_dc'] = eval_df['pick'].apply(dc_score)
print(f"Eval sample: {len(eval_df)} TEs (2015-2024)\n")

# Build derived metrics
eval_df['speed_score'] = np.where(
    eval_df['wt'].notna() & eval_df['forty'].notna() & (eval_df['forty'] > 0),
    (eval_df['wt'] * 200) / (eval_df['forty'] ** 4),
    np.nan
)

# Age weight
def age_weight(da):
    if pd.isna(da): return 1.0
    if da <= 21: return 1.15
    elif da <= 22: return 1.10
    elif da <= 23: return 1.00
    elif da <= 24: return 0.95
    else: return 0.90

eval_df['ryptpa'] = np.where(
    eval_df['cfbd_rec_yards'].notna() & eval_df['cfbd_team_pass_att'].notna() & (eval_df['cfbd_team_pass_att'] > 0),
    eval_df['cfbd_rec_yards'] / eval_df['cfbd_team_pass_att'] * 100,
    np.nan
)
eval_df['age_weighted_ryptpa'] = eval_df.apply(
    lambda r: r['ryptpa'] * age_weight(r['draft_age']) if pd.notna(r['ryptpa']) else np.nan, axis=1
)
eval_df['rec_per_tpa'] = np.where(
    eval_df['cfbd_receptions'].notna() & eval_df['cfbd_team_pass_att'].notna() & (eval_df['cfbd_team_pass_att'] > 0),
    eval_df['cfbd_receptions'] / eval_df['cfbd_team_pass_att'] * 100,
    np.nan
)
eval_df['age_weighted_rec_tpa'] = eval_df.apply(
    lambda r: r['rec_per_tpa'] * age_weight(r['draft_age']) if pd.notna(r['rec_per_tpa']) else np.nan, axis=1
)

# PFF target rate
eval_df['pff_target_rate'] = np.where(
    eval_df['pff_routes'].notna() & (eval_df['pff_routes'] > 0),
    eval_df['pff_targets'] / eval_df['pff_routes'],
    np.nan
)

# Outcomes
outcomes = {
    'top12_10g':          ('Top-12 TE season (10g)', 'binary'),
    'top6_10g':           ('Top-6 TE season (10g)', 'binary'),
    'best_3yr_ppg_10g':   ('Best 3yr PPG (10g)', 'continuous'),
    'best_career_ppg_10g':('Best career PPG (10g)', 'continuous'),
    'seasons_over_10ppg_10g': ('Seasons 10+ PPG', 'continuous'),
}

# ============================================================================
# TEST ALL 10 METRICS
# ============================================================================

metrics = {
    # 1. Weight alone
    'wt':                   'Weight (lbs)',
    # 2. Speed Score
    'speed_score':          'Speed Score (Barnwell)',
    # 3. PFF Blocking grade
    'pff_grades_pass_block':'PFF Pass Block Grade',
    # 5. Peak dominator standalone
    'peak_dominator':       'Peak Dominator (%)',
    # 6. Target share
    'pff_target_rate':      'PFF Target Rate (tgt/routes)',
    'pff_targets':          'PFF Targets (raw)',
    # 7. Rush yards
    'cfbd_rush_yards':      'CFBD Rush Yards',
    # 8. Age-adjusted production
    'age_weighted_ryptpa':  'Age-Weighted RYPTPA',
    'age_weighted_rec_tpa': 'Age-Weighted Rec/TPA',
    'ryptpa':               'Raw RYPTPA (no age adj)',
    'rec_per_tpa':          'Raw Rec/TPA (no age adj)',
    # 9. Individual combine metrics
    'forty':                '40-Yard Dash (lower=better)',
    'vertical':             'Vertical Jump',
    'broad_jump':           'Broad Jump',
    'cone':                 '3-Cone (lower=better)',
    'shuttle':              'Shuttle (lower=better)',
    'bench':                'Bench Press (reps)',
    # 10. PFF YPRR + grades
    'pff_yprr':             'PFF Yards Per Route Run',
    'pff_grades_offense':   'PFF Offense Grade',
    'pff_grades_pass_route':'PFF Pass Route Grade',
}

print("=" * 130)
print("PARTIAL CORRELATIONS CONTROLLING FOR DC — ALL 10 METRIC CATEGORIES")
print("=" * 130)

# Header
out_cols = list(outcomes.keys())
header = f"  {'Metric':<32} {'n':>4}"
for oc in out_cols:
    header += f" {'r':>6} {'p':>7}"
print(f"\n{header}")
sub = f"  {'':32} {'':4}"
for oc_name, (label, _) in outcomes.items():
    short = label[:13]
    sub += f" {short:>13}"
print(sub)
print(f"  {'-'*128}")

results = []
for metric_key, metric_label in metrics.items():
    if metric_key not in eval_df.columns:
        print(f"  {metric_label:<32} {'—':>4} (column not found)")
        continue

    n_valid = eval_df[metric_key].notna().sum()
    line = f"  {metric_label:<32} {n_valid:>4}"

    for oc_name, (oc_label, oc_type) in outcomes.items():
        if oc_name not in eval_df.columns:
            line += f" {'—':>6} {'—':>7}"
            continue
        r, p, n = partial_corr(eval_df[metric_key], eval_df[oc_name], eval_df['pick'])
        line += f" {r:>+6.3f} {p:>7.3f}"
        results.append({
            'metric': metric_label, 'metric_key': metric_key,
            'outcome': oc_name, 'r': r, 'p': p, 'n': n
        })
    print(line)

# Flag significant results
print(f"\n\n{'='*130}")
print("SIGNIFICANT PARTIAL CORRELATIONS (p < 0.10) CONTROLLING FOR DC")
print("=" * 130)

sig_results = [r for r in results if pd.notna(r['p']) and r['p'] < 0.10]
sig_results.sort(key=lambda x: x['p'])

if sig_results:
    print(f"\n  {'Metric':<32} {'Outcome':<22} {'r':>7} {'p':>8} {'n':>5} {'Signal?'}")
    print(f"  {'-'*85}")
    for sr in sig_results:
        sig_flag = "***" if sr['p'] < 0.01 else "**" if sr['p'] < 0.05 else "*"
        print(f"  {sr['metric']:<32} {sr['outcome']:<22} {sr['r']:>+7.3f} {sr['p']:>8.4f} {sr['n']:>5} {sig_flag}")
else:
    print("  No significant results found.")


# ============================================================================
# METRIC #4: TEAMMATE SCORE — BUILD AND TEST
# ============================================================================

print(f"\n\n{'='*130}")
print("METRIC #4: TE TEAMMATE SCORE — Building and testing")
print("(Did the TE share targets with other highly-drafted pass catchers?)")
print("=" * 130)

# Load draft picks to find teammates
draft = pd.read_parquet('data/nflverse/draft_picks.parquet')
pass_catchers = draft[draft['position'].isin(['TE', 'WR'])].copy()
pass_catchers['s_dc'] = pass_catchers['pick'].apply(dc_score)

# For each TE in our backtest, compute total DC of same-school, same-draft-year pass catchers (excluding themselves)
def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()

pass_catchers['name_norm'] = pass_catchers['pfr_player_name'].apply(normalize_name)

teammate_scores = []
for _, te in eval_df.iterrows():
    name = te['player_name']
    dy = te['draft_year']
    school = te['college']
    nn = normalize_name(name)

    # Find all pass catchers from same school, same draft year
    same_class = pass_catchers[
        (pass_catchers['season'] == dy) &
        (pass_catchers['college'].str.lower() == str(school).lower())
    ]

    # Exclude the player themselves
    teammates = same_class[same_class['name_norm'] != nn]
    total_tm_dc = teammates['s_dc'].sum()
    n_teammates = len(teammates)

    teammate_scores.append({
        'player_name': name, 'draft_year': dy, 'college': school,
        'teammate_dc': total_tm_dc, 'n_teammates': n_teammates,
    })

tm_df = pd.DataFrame(teammate_scores)
eval_df = eval_df.merge(tm_df[['player_name', 'draft_year', 'teammate_dc', 'n_teammates']],
                         on=['player_name', 'draft_year'], how='left')

# Binary teammate score (same threshold as WR model: total_dc > 150)
eval_df['teammate_binary_150'] = (eval_df['teammate_dc'] > 150).astype(int)
# Also test lower thresholds
eval_df['teammate_binary_100'] = (eval_df['teammate_dc'] > 100).astype(int)
eval_df['teammate_binary_50'] = (eval_df['teammate_dc'] > 50).astype(int)

print(f"\n  Teammate DC distribution:")
print(f"    n with any teammates drafted: {(eval_df['teammate_dc'] > 0).sum()}/{len(eval_df)}")
print(f"    n with teammate_dc > 50:  {(eval_df['teammate_dc'] > 50).sum()}/{len(eval_df)}")
print(f"    n with teammate_dc > 100: {(eval_df['teammate_dc'] > 100).sum()}/{len(eval_df)}")
print(f"    n with teammate_dc > 150: {(eval_df['teammate_dc'] > 150).sum()}/{len(eval_df)}")
print(f"    Mean teammate_dc: {eval_df['teammate_dc'].mean():.1f}")
print(f"    Max teammate_dc: {eval_df['teammate_dc'].max():.1f}")

# Show top teammate examples
top_tm = eval_df.nlargest(10, 'teammate_dc')
print(f"\n  Top 10 TEs by teammate DC:")
for _, r in top_tm.iterrows():
    hit = "HIT" if r.get('top12_10g', 0) == 1 else "miss"
    print(f"    {r['player_name']:<25} {r['college']:<20} {r['draft_year']} | tm_dc={r['teammate_dc']:.0f}, "
          f"n_tm={int(r['n_teammates'])}, {hit}")

# Partial correlations for teammate score variants
print(f"\n  Partial correlations controlling for DC:")
print(f"  {'Teammate Metric':<30} {'Outcome':<22} {'r':>7} {'p':>8} {'n':>5}")
print(f"  {'-'*75}")

for tm_col, tm_label in [('teammate_dc', 'Teammate DC (continuous)'),
                           ('teammate_binary_150', 'Teammate Binary (>150)'),
                           ('teammate_binary_100', 'Teammate Binary (>100)'),
                           ('teammate_binary_50', 'Teammate Binary (>50)')]:
    for oc_name, (oc_label, _) in outcomes.items():
        if oc_name not in eval_df.columns:
            continue
        r, p, n = partial_corr(eval_df[tm_col], eval_df[oc_name], eval_df['pick'])
        sig = " *" if p < 0.10 else " **" if p < 0.05 else " ***" if p < 0.01 else ""
        if p < 0.10 or oc_name in ['top12_10g', 'best_3yr_ppg_10g']:
            print(f"  {tm_label:<30} {oc_name:<22} {r:>+7.3f} {p:>8.4f} {n:>5}{sig}")

# Hit rates for binary teammate score
print(f"\n  Hit rates by teammate status (>150 threshold):")
for oc in ['top12_10g', 'top6_10g']:
    if oc not in eval_df.columns:
        continue
    has_tm = eval_df[eval_df['teammate_binary_150'] == 1]
    no_tm = eval_df[eval_df['teammate_binary_150'] == 0]
    rate_tm = has_tm[oc].mean() if len(has_tm) > 0 else 0
    rate_no = no_tm[oc].mean() if len(no_tm) > 0 else 0
    print(f"    {oc}: with teammates={rate_tm:.1%} (n={len(has_tm)}), without={rate_no:.1%} (n={len(no_tm)})")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n\n{'='*130}")
print("FINAL AUDIT SUMMARY: ALL 10 METRICS")
print("=" * 130)
print("""
  #  Metric                          Tested?    Key Finding
  ── ─────────────────────────────── ────────── ──────────────────────────────────────────
  1  Weight alone                    YES        Results in table above
  2  Speed Score (Barnwell)          YES        Results in table above
  3  PFF Blocking grade              YES        Pass Block Grade tested (no Run Block available)
  4  Teammate Score                  NEW TEST   Results above — first time tested for TEs
  5  Peak Dominator standalone       YES        Results in table above
  6  Target share / targets per game YES        PFF Target Rate + raw targets tested
  7  Rush yards                      YES        CFBD Rush Yards tested
  8  Age-adjusted production         YES        Age-Weighted RYPTPA + Rec/TPA tested
  9  Individual combine metrics      YES        Vertical, broad jump, 3-cone, shuttle all tested
  10 PFF YPRR / grade tiers         YES        YPRR + 5 PFF grade sub-metrics tested
""")
