"""
FULL RB MODEL AUDIT
====================
Part 1: Model structure, formulas, and code
Part 2: Data coverage for all 223 backtest RBs
Part 3: Unused college metrics — partial correlations vs NFL outcomes
Part 4: Full correlation suite — SLAP vs DC-only on all outcomes

DOES NOT CHANGE ANYTHING — diagnostic output only.
"""

import pandas as pd
import numpy as np
import warnings, os, glob
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score
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

def normalize_team(team):
    if pd.isna(team): return ''
    return str(team).strip().upper().replace('STATE', 'ST').replace('UNIVERSITY', '').strip()

def partial_corr(x, y, z):
    mask = x.notna() & y.notna() & z.notna()
    x2, y2, z2 = x[mask].values, y[mask].values, z[mask].values
    n = len(x2)
    if n < 10:
        return np.nan, np.nan, n
    slope_xz = np.polyfit(z2, x2, 1)
    res_x = x2 - np.polyval(slope_xz, z2)
    slope_yz = np.polyfit(z2, y2, 1)
    res_y = y2 - np.polyval(slope_yz, z2)
    r, p = sp_stats.pearsonr(res_x, res_y)
    return r, p, n

def rb_production_score(rec_yards, team_pass_att, age):
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return np.nan
    try:
        age = float(age)
    except (TypeError, ValueError):
        age = 22.0
    if pd.isna(age): age = 22
    season_age = age - 1
    age_w = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
    raw = (rec_yards / team_pass_att) * age_w * 100
    return min(99.9, max(0, raw / 1.75))

def speed_score_fn(weight, forty):
    if pd.isna(weight) or pd.isna(forty) or forty <= 0 or weight <= 0:
        return np.nan
    return (weight * 200) / (forty ** 4)

def normalize_0_100(series):
    mn, mx = series.dropna().min(), series.dropna().max()
    if mx == mn: return pd.Series(50, index=series.index)
    return ((series - mn) / (mx - mn) * 100).clip(0, 100)


# ============================================================================
# LOAD BASE DATA
# ============================================================================
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
rb_out = outcomes[outcomes['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg',
     'seasons_played', 'first_3yr_games', 'career_games_in_data']
].copy()
rb_bt = rb_bt.merge(rb_out, on=['player_name', 'draft_year', 'pick'], how='left')
rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)
rb_bt['name_norm'] = rb_bt['player_name'].apply(normalize_name)

# Compute SLAP (reproducing build script exactly)
rb_bt['s_production'] = rb_bt.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb_bt['s_production_final'] = rb_bt['s_production'].fillna(0)

# Speed Score with MNAR imputation
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine['name_norm'] = combine['player_name'].apply(normalize_name)
combine_lookup = {}
for pos_pref in ['RB', 'FB', 'WR', 'TE']:
    sub = combine[combine['pos'] == pos_pref]
    for _, row in sub.iterrows():
        dy = row.get('draft_year')
        if pd.isna(dy): dy = row.get('season')
        if pd.isna(dy): continue
        key = (row['name_norm'], int(dy))
        if key not in combine_lookup:
            combine_lookup[key] = {
                'weight': row['wt'], 'forty': row['forty'],
                'vertical': row.get('vertical'), 'broad_jump': row.get('broad_jump'),
                'bench': row.get('bench'), 'cone': row.get('cone'),
                'shuttle': row.get('shuttle'), 'ht': row.get('ht'),
            }

for field in ['weight', 'forty', 'vertical', 'broad_jump', 'bench', 'cone', 'shuttle']:
    rb_bt[field] = rb_bt.apply(
        lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get(field, np.nan), axis=1)
    rb_bt[field] = pd.to_numeric(rb_bt[field], errors='coerce')

known = rb_bt[rb_bt['weight'].notna() & rb_bt['forty'].notna()].copy()
def wt_bucket(wt):
    if pd.isna(wt): return None
    if wt < 200: return '<200'
    elif wt < 210: return '200-209'
    elif wt < 220: return '210-219'
    else: return '220+'
def rd_bucket(rd):
    if rd <= 1: return 'Rd 1'
    elif rd <= 2: return 'Rd 2'
    elif rd <= 4: return 'Rd 3-4'
    else: return 'Rd 5+'

known['wb'] = known['weight'].apply(wt_bucket)
known['rb_bkt'] = known['round'].apply(rd_bucket)
lookup_40 = {}
for wb in ['<200', '200-209', '210-219', '220+']:
    for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
        sub_k = known[(known['wb'] == wb) & (known['rb_bkt'] == rdb)]
        if len(sub_k) > 0:
            lookup_40[(wb, rdb)] = sub_k['forty'].mean()
    wt_sub = known[known['wb'] == wb]
    if len(wt_sub) > 0:
        for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
            if (wb, rdb) not in lookup_40:
                lookup_40[(wb, rdb)] = wt_sub['forty'].mean()

rb_bt['forty_final'] = rb_bt['forty'].copy()
impute_mask = rb_bt['weight'].notna() & rb_bt['forty'].isna()
for idx in rb_bt[impute_mask].index:
    wb = wt_bucket(rb_bt.loc[idx, 'weight'])
    rdb = rd_bucket(rb_bt.loc[idx, 'round'])
    est = lookup_40.get((wb, rdb))
    if est is not None:
        rb_bt.loc[idx, 'forty_final'] = est

rb_bt['raw_ss'] = rb_bt.apply(lambda r: speed_score_fn(r['weight'], r['forty_final']), axis=1)
real_ss = rb_bt['raw_ss'].dropna()
p60, p40 = real_ss.quantile(0.60), real_ss.quantile(0.40)
for idx in rb_bt[rb_bt['raw_ss'].isna()].index:
    rd = rb_bt.loc[idx, 'round']
    rb_bt.loc[idx, 'raw_ss'] = p60 if rd <= 2 else p40
rb_bt['s_speed_score'] = normalize_0_100(rb_bt['raw_ss'])

RB_V5 = {'dc': 0.65, 'production': 0.30, 'speed_score': 0.05}
rb_bt['slap_v5'] = (
    RB_V5['dc'] * rb_bt['s_dc'] +
    RB_V5['production'] * rb_bt['s_production_final'] +
    RB_V5['speed_score'] * rb_bt['s_speed_score']
).round(1)


# ============================================================================
# PART 1: MODEL STRUCTURE
# ============================================================================
print("=" * 120)
print("PART 1: RB V5 MODEL STRUCTURE")
print("=" * 120)

print(f"""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  RB SLAP V5 = DC × 0.65 + RYPTPA × 0.30 + Speed_Score × 0.05        │
  └─────────────────────────────────────────────────────────────────────────┘

  COMPONENT 1: Draft Capital (65% weight)
  ────────────────────────────────────────
    Formula:  DC = 100 - 2.40 × (pick^0.62 - 1)
    Range:    0 to 100 (pick 1 = 100, pick 250 ≈ 29)
    Code:     src/build_master_database_v5.py line 34-36
    Inputs:   pick (actual draft pick number)
    Coverage: 223/223 (100%)

  COMPONENT 2: Receiving Production — RYPTPA (30% weight)
  ────────────────────────────────────────────────────────
    Formula:  raw = (rec_yards / team_pass_att) × age_weight × 100
              RYPTPA = min(99.9, raw / 1.75)
    Code:     src/build_master_database_v5.py lines 77-89

    Age weight function (applied to season_age = draft_age - 1):
      Season age 19: 1.15x    Season age 20: 1.10x
      Season age 21: 1.05x    Season age 22: 1.00x
      Season age 23: 0.95x    Season age 24+: 0.90x
    Code:     line 87: age_w = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))

    Inputs:   rec_yards (FINAL college season receiving yards)
              team_pass_att (FINAL season team pass attempts)
              age (draft age, used for season age weighting)
    Season:   ALWAYS draft_year - 1 (final college season only)
    NaN fill: Missing production → 0 (line 245)

  COMPONENT 3: Speed Score (5% weight)
  ─────────────────────────────────────
    Formula:  raw_ss = (weight × 200) / (forty^4)
              Speed_Score = normalize_0_100(raw_ss)  # min-max within backtest RBs
    Code:     src/build_master_database_v5.py lines 91-94, 307-317

    MNAR Imputation Pipeline:
      Step 1: Weight + 40 from combine.parquet (lines 249-268)
      Step 2: Missing 40 estimated from weight×round bucket averages (lines 270-305)
      Step 3: Still missing → MNAR imputation (lines 309-315):
              Round 1-2 → 60th percentile of known speed scores
              Round 3+  → 40th percentile of known speed scores
      Step 4: All raw speed scores normalized 0-100 (line 317)

  FINAL SLAP CALCULATION (lines 320-324):
    slap_v5 = 0.65 × DC + 0.30 × RYPTPA + 0.05 × Speed_Score
""")


# ============================================================================
# PART 2: DATA COVERAGE AUDIT
# ============================================================================
print(f"\n{'='*120}")
print("PART 2: DATA COVERAGE AUDIT — All 223 Backtest RBs")
print("=" * 120)

# Ensure numeric
rb_bt['rec_yards_num'] = pd.to_numeric(rb_bt['rec_yards'], errors='coerce')
rb_bt['receptions_num'] = pd.to_numeric(rb_bt['receptions'], errors='coerce')
rb_bt['team_pass_att_num'] = pd.to_numeric(rb_bt['team_pass_att'], errors='coerce')

total = len(rb_bt)
print(f"\n  {'Input Field':<40} {'Real':>5} {'Estimated':>9} {'Imputed':>8} {'Missing':>8} {'Coverage':>9}")
print(f"  {'-'*85}")

# Draft capital
print(f"  {'Draft pick':.<40} {total:>5} {'':>9} {'':>8} {0:>8} {'100.0%':>9}")

# Receiving yards
ry_real = rb_bt['rec_yards_num'].notna().sum()
ry_miss = total - ry_real
print(f"  {'Receiving yards (final season)':.<40} {ry_real:>5} {'':>9} {'':>8} {ry_miss:>8} {ry_real/total*100:.1f}%")

# Team pass attempts
tpa_real = rb_bt['team_pass_att_num'].notna().sum()
tpa_miss = total - tpa_real
print(f"  {'Team pass attempts (final season)':.<40} {tpa_real:>5} {'':>9} {'':>8} {tpa_miss:>8} {tpa_real/total*100:.1f}%")

# RYPTPA (computed)
ryptpa_real = rb_bt['s_production'].notna().sum()
ryptpa_filled = total  # NaN → 0
print(f"  {'RYPTPA score (computed)':.<40} {ryptpa_real:>5} {'':>9} {total-ryptpa_real:>8} {'':>8} {ryptpa_real/total*100:.1f}% real")

# Receptions (not used in model but available)
rec_real = rb_bt['receptions_num'].notna().sum()
print(f"  {'Receptions (final season, NOT used)':.<40} {rec_real:>5} {'':>9} {'':>8} {total-rec_real:>8} {rec_real/total*100:.1f}%")

# Weight
wt_real = rb_bt['weight'].notna().sum()
print(f"  {'Weight (combine)':.<40} {wt_real:>5} {'':>9} {'':>8} {total-wt_real:>8} {wt_real/total*100:.1f}%")

# 40 time
forty_real = rb_bt['forty'].notna().sum()
forty_est = impute_mask.sum()
forty_miss = total - forty_real - forty_est
print(f"  {'40-yard dash (real combine time)':.<40} {forty_real:>5} {'':>9} {'':>8} {total-forty_real:>8} {forty_real/total*100:.1f}%")
print(f"  {'40-yard dash (estimated from wt×rd)':.<40} {'':>5} {forty_est:>9} {'':>8} {'':>8}")
print(f"  {'40-yard dash (total with estimates)':.<40} {forty_real+forty_est:>5} {'':>9} {'':>8} {total-forty_real-forty_est:>8} {(forty_real+forty_est)/total*100:.1f}%")

# Speed Score
ss_from_real = rb_bt[rb_bt['forty'].notna() & rb_bt['weight'].notna()].shape[0]
ss_from_est40 = impute_mask.sum()
ss_mnar = (rb_bt['weight'].isna()).sum()
print(f"  {'Speed Score (real wt + real 40)':.<40} {ss_from_real:>5} {'':>9} {'':>8} {'':>8} {ss_from_real/total*100:.1f}%")
print(f"  {'Speed Score (real wt + est 40)':.<40} {'':>5} {ss_from_est40:>9} {'':>8} {'':>8}")
print(f"  {'Speed Score (MNAR imputed)':.<40} {'':>5} {'':>9} {ss_mnar:>8} {'':>8}")
print(f"  {'Speed Score (total = all 223)':.<40} {total:>5} {'':>9} {'':>8} {0:>8} 100.0%")

# Combine extras (not used)
for field, label in [('vertical', 'Vertical jump'), ('broad_jump', 'Broad jump'),
                     ('bench', 'Bench press'), ('cone', '3-cone drill'), ('shuttle', 'Shuttle')]:
    n = rb_bt[field].notna().sum()
    print(f"  {f'{label} (NOT used in model)':.<40} {n:>5} {'':>9} {'':>8} {total-n:>8} {n/total*100:.1f}%")

# Age
age_real = rb_bt['age'].notna().sum()
print(f"  {'Draft age':.<40} {age_real:>5} {'':>9} {'':>8} {total-age_real:>8} {age_real/total*100:.1f}%")

# RAS (from backtest file)
ras_real = rb_bt['RAS'].notna().sum()
print(f"  {'RAS (NOT used in V5)':.<40} {ras_real:>5} {'':>9} {'':>8} {total-ras_real:>8} {ras_real/total*100:.1f}%")

# NFL outcomes
print(f"\n  {'NFL Outcome':<40} {'Available':>9} {'Missing':>8}")
print(f"  {'-'*60}")
for col, label in [('hit24', 'hit24 (top-24 PPR season)'),
                   ('hit12', 'hit12 (top-12 PPR season)'),
                   ('first_3yr_ppg', 'first_3yr_ppg'),
                   ('career_ppg', 'career_ppg'),
                   ('best_ppr', 'best_ppr (season total)')]:
    n = rb_bt[col].notna().sum()
    print(f"  {label:<40} {n:>9} {total-n:>8}")

# Draft year distribution
print(f"\n  Draft year distribution:")
for yr in sorted(rb_bt['draft_year'].unique()):
    n = (rb_bt['draft_year'] == yr).sum()
    print(f"    {int(yr)}: {n} RBs")


# ============================================================================
# PART 3: UNUSED COLLEGE METRICS — PARTIAL CORRELATIONS
# ============================================================================
print(f"\n\n{'='*120}")
print("PART 3: UNUSED COLLEGE METRICS — Partial correlations controlling for DC")
print("=" * 120)
print(f"\n  Loading PFF rushing + receiving summaries for final college season...")

# Map rushing summary files to approximate college seasons
# Based on known player-season matches:
rush_file_seasons = {
    'rushing_summary.csv': 2014,
    'rushing_summary (1).csv': 2015,
    'rushing_summary (2).csv': 2016,
    'rushing_summary (3).csv': 2017,
    'rushing_summary (4).csv': 2018,
    'rushing_summary (5).csv': 2019,
    'rushing_summary (6).csv': 2020,
    'rushing_summary (7).csv': 2021,
    'rushing_summary (8).csv': 2022,
    'rushing_summary (9).csv': 2023,
    'rushing_summary (10).csv': 2024,
    'rushing_summary (11).csv': 2025,
    'rushing_summary (12).csv': 2025,  # possible duplicate
}

recv_file_seasons = {
    'receiving_summary (2).csv': 2014,
    'receiving_summary (3).csv': 2015,
    'receiving_summary (4).csv': 2016,
    'receiving_summary (5).csv': 2017,
    'receiving_summary (13).csv': 2018,
    'receiving_summary (15).csv': 2019,
    'receiving_summary (16).csv': 2020,
    'receiving_summary (17).csv': 2021,
    'receiving_summary (18).csv': 2022,
    'receiving_summary (19).csv': 2023,
    'receiving_summary (20).csv': 2024,
    'receiving_summary (21).csv': 2025,
}

# Load all PFF rushing data with season tags
all_rush = []
for fname, season in rush_file_seasons.items():
    fpath = f'data/{fname}'
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df['pff_season'] = season
        # Filter to HB/RB only
        df = df[df['position'].isin(['HB', 'RB'])]
        df['name_norm'] = df['player'].apply(normalize_name)
        df['team_norm'] = df['team_name'].apply(normalize_team)
        all_rush.append(df)
if all_rush:
    rush_all = pd.concat(all_rush, ignore_index=True)
    print(f"  Loaded {len(rush_all)} RB rushing season-rows from {len(all_rush)} files")
else:
    rush_all = pd.DataFrame()

# Load all PFF receiving data with season tags
all_recv = []
for fname, season in recv_file_seasons.items():
    fpath = f'data/{fname}'
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df['pff_season'] = season
        df = df[df['position'].isin(['HB', 'RB'])]
        df['name_norm'] = df['player'].apply(normalize_name)
        df['team_norm'] = df['team_name'].apply(normalize_team)
        all_recv.append(df)
if all_recv:
    recv_all = pd.concat(all_recv, ignore_index=True)
    print(f"  Loaded {len(recv_all)} RB receiving season-rows from {len(all_recv)} files")
else:
    recv_all = pd.DataFrame()

# Match backtest RBs to their FINAL college season PFF data
# Final season = draft_year - 1
print(f"\n  Matching backtest RBs to final-season PFF data...")

# School name normalization mapping (common PFF → backtest)
school_norm = {
    'OHIO ST': 'OHIO ST', 'ALABAMA': 'ALABAMA', 'GEORGIA': 'GEORGIA',
    'LSU': 'LSU', 'CLEMSON': 'CLEMSON', 'MICHIGAN': 'MICHIGAN',
    'S CAROLINA': 'SOUTH CAROLINA', 'MISS ST': 'MISSISSIPPI ST',
    'N CAROLINA': 'NORTH CAROLINA', 'OLE MISS': 'MISSISSIPPI',
    'OKLA STATE': 'OKLAHOMA ST', 'PENN ST': 'PENN ST',
    'MICH ST': 'MICHIGAN ST', 'WASH ST': 'WASHINGTON ST',
    'S JOSE ST': 'SAN JOSE ST', 'S DIEGO ST': 'SAN DIEGO ST',
    'GA TECH': 'GEORGIA TECH', 'GA SOUTHERN': 'GEORGIA SOUTHERN',
    'NC STATE': 'NC ST', 'BALL ST': 'BALL ST', 'BOISE ST': 'BOISE ST',
    'IOWA ST': 'IOWA ST', 'C MICHIGAN': 'CENTRAL MICHIGAN',
    'W MICHIGAN': 'WESTERN MICHIGAN', 'E MICHIGAN': 'EASTERN MICHIGAN',
    'N ILLINOIS': 'NORTHERN ILLINOIS', 'W KENTUCKY': 'WESTERN KENTUCKY',
    'UNLV': 'UNLV', 'UCLA': 'UCLA', 'USC': 'USC', 'TCU': 'TCU',
    'SMU': 'SMU', 'UCF': 'UCF', 'BYU': 'BYU', 'FAU': 'FAU',
    'FIU': 'FIU', 'UAB': 'UAB', 'UTSA': 'UTSA', 'UTEP': 'UTEP',
}

rush_matched = 0
recv_matched = 0

for idx in rb_bt.index:
    name_n = rb_bt.loc[idx, 'name_norm']
    dy = int(rb_bt.loc[idx, 'draft_year'])
    final_season = dy - 1

    # Try to match in rushing data — look for player in season = final_season ± 1
    for season_try in [final_season, final_season - 1, final_season + 1]:
        candidates = rush_all[(rush_all['name_norm'] == name_n) & (rush_all['pff_season'] == season_try)]
        if len(candidates) == 1:
            row = candidates.iloc[0]
            for col in ['yards', 'attempts', 'ypa', 'yards_after_contact', 'yco_attempt',
                        'avoided_tackles', 'elusive_rating', 'breakaway_yards', 'breakaway_percent',
                        'explosive', 'grades_run', 'grades_offense', 'first_downs',
                        'gap_attempts', 'zone_attempts', 'touchdowns', 'fumbles',
                        'longest', 'player_game_count', 'grades_pass_block',
                        'grades_pass_route', 'elu_rush_mtf']:
                if col in row.index:
                    rb_bt.loc[idx, f'pff_rush_{col}'] = row[col]
            rush_matched += 1
            break

    # Try to match in receiving data
    for season_try in [final_season, final_season - 1, final_season + 1]:
        candidates = recv_all[(recv_all['name_norm'] == name_n) & (recv_all['pff_season'] == season_try)]
        if len(candidates) == 1:
            row = candidates.iloc[0]
            for col in ['yards', 'receptions', 'targets', 'caught_percent', 'drop_rate',
                        'grades_pass_route', 'yards_after_catch', 'yards_after_catch_per_reception',
                        'route_rate', 'grades_pass_block', 'pass_blocks', 'yprr',
                        'targeted_qb_rating', 'avg_depth_of_target', 'grades_offense',
                        'player_game_count']:
                if col in row.index:
                    rb_bt.loc[idx, f'pff_recv_{col}'] = row[col]
            recv_matched += 1
            break

print(f"  Rushing data matched: {rush_matched}/{total} ({rush_matched/total*100:.0f}%)")
print(f"  Receiving data matched: {recv_matched}/{total} ({recv_matched/total*100:.0f}%)")

# Now test partial correlations for all available PFF metrics
print(f"\n  Testing partial correlations controlling for DC against first_3yr_ppg...")
print(f"\n  {'Metric':<45} {'partial r':>9} {'p-value':>9} {'n':>4} {'Sig':>4}")
print(f"  {'-'*75}")

pff_metrics = [
    # Rushing metrics
    ('pff_rush_yards', 'Rush yards (total)'),
    ('pff_rush_attempts', 'Rush attempts'),
    ('pff_rush_ypa', 'Yards per attempt'),
    ('pff_rush_yards_after_contact', 'Yards after contact (total)'),
    ('pff_rush_yco_attempt', 'Yards after contact per attempt'),
    ('pff_rush_avoided_tackles', 'Avoided tackles'),
    ('pff_rush_elusive_rating', 'Elusive rating'),
    ('pff_rush_breakaway_yards', 'Breakaway yards (20+ runs)'),
    ('pff_rush_breakaway_percent', 'Breakaway % of attempts'),
    ('pff_rush_explosive', 'Explosive plays (10+ yards)'),
    ('pff_rush_grades_run', 'PFF Run Grade'),
    ('pff_rush_grades_offense', 'PFF Offense Grade (rushing)'),
    ('pff_rush_first_downs', 'First downs (rushing)'),
    ('pff_rush_touchdowns', 'Rushing TDs'),
    ('pff_rush_longest', 'Longest rush'),
    ('pff_rush_fumbles', 'Fumbles'),
    ('pff_rush_grades_pass_block', 'PFF Pass Block Grade'),
    ('pff_rush_grades_pass_route', 'PFF Pass Route Grade (rush file)'),
    ('pff_rush_elu_rush_mtf', 'Elusiveness moves (spins/jukes)'),
    ('pff_rush_gap_attempts', 'Gap scheme attempts'),
    ('pff_rush_zone_attempts', 'Zone scheme attempts'),
    ('pff_rush_player_game_count', 'Games played'),
    # Receiving metrics
    ('pff_recv_yards', 'PFF Receiving yards'),
    ('pff_recv_receptions', 'PFF Receptions'),
    ('pff_recv_targets', 'Targets'),
    ('pff_recv_caught_percent', 'Catch rate (%)'),
    ('pff_recv_drop_rate', 'Drop rate (%)'),
    ('pff_recv_grades_pass_route', 'PFF Route Grade (recv file)'),
    ('pff_recv_yards_after_catch', 'Receiving YAC (total)'),
    ('pff_recv_yards_after_catch_per_reception', 'YAC per reception'),
    ('pff_recv_route_rate', 'Route rate (% of pass snaps)'),
    ('pff_recv_yprr', 'Yards per route run'),
    ('pff_recv_grades_pass_block', 'PFF Pass Block Grade (recv)'),
    ('pff_recv_avg_depth_of_target', 'Avg depth of target'),
]

# Ensure numeric
for col, _ in pff_metrics:
    if col in rb_bt.columns:
        rb_bt[col] = pd.to_numeric(rb_bt[col], errors='coerce')

# Test against first_3yr_ppg (best continuous outcome with good coverage)
results_3yr = []
for col, label in pff_metrics:
    if col in rb_bt.columns:
        r, p, n = partial_corr(rb_bt[col], rb_bt['first_3yr_ppg'], rb_bt['s_dc'])
        sig = '**' if p < 0.01 else ' *' if p < 0.05 else '  '
        if not np.isnan(r):
            results_3yr.append((label, r, p, n, sig))
            print(f"  {label:<45} {r:>+9.3f} {p:>9.4f} {n:>4} {sig:>4}")
        else:
            print(f"  {label:<45} {'n/a':>9} {'n/a':>9} {n:>4}")
    else:
        print(f"  {label:<45} {'NO DATA':>9}")

# Also test combine metrics
print(f"\n  ── Combine Athletic Metrics ──")
combine_metrics = [
    ('vertical', 'Vertical jump (inches)'),
    ('broad_jump', 'Broad jump (inches)'),
    ('bench', 'Bench press (reps at 225)'),
    ('cone', '3-cone drill (seconds, lower=better)'),
    ('shuttle', 'Shuttle (seconds, lower=better)'),
    ('weight', 'Weight (lbs)'),
]
for col, label in combine_metrics:
    if col in rb_bt.columns:
        # For timed events, negate so positive = better
        if col in ['cone', 'shuttle']:
            vals = -rb_bt[col]
        else:
            vals = rb_bt[col]
        r, p, n = partial_corr(vals, rb_bt['first_3yr_ppg'], rb_bt['s_dc'])
        sig = '**' if p < 0.01 else ' *' if p < 0.05 else '  '
        if not np.isnan(r):
            print(f"  {label:<45} {r:>+9.3f} {p:>9.4f} {n:>4} {sig:>4}")

# Also test existing model components
print(f"\n  ── Current Model Components (for reference) ──")
for col, label in [('s_production_final', 'RYPTPA (current production)'),
                   ('s_speed_score', 'Speed Score (current)')]:
    r, p, n = partial_corr(rb_bt[col], rb_bt['first_3yr_ppg'], rb_bt['s_dc'])
    sig = '**' if p < 0.01 else ' *' if p < 0.05 else '  '
    print(f"  {label:<45} {r:>+9.3f} {p:>9.4f} {n:>4} {sig:>4}")

# Summary: top metrics by |partial r|
print(f"\n\n  ── TOP 10 METRICS BY |PARTIAL r| (controlling for DC, vs first_3yr_ppg) ──")
all_results = []
for col, label in pff_metrics + combine_metrics + [('s_production_final', 'RYPTPA (CURRENT)')]:
    if col in rb_bt.columns:
        vals = -rb_bt[col] if col in ['cone', 'shuttle'] else rb_bt[col]
        r, p, n = partial_corr(vals, rb_bt['first_3yr_ppg'], rb_bt['s_dc'])
        if not np.isnan(r) and n >= 20:
            all_results.append((label, r, p, n))

all_results.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"\n  {'Rank':>4} {'Metric':<45} {'partial r':>9} {'p-value':>9} {'n':>4}")
print(f"  {'-'*75}")
for i, (label, r, p, n) in enumerate(all_results[:15], 1):
    sig = '**' if p < 0.01 else ' *' if p < 0.05 else '  '
    current = ' ← CURRENT' if 'CURRENT' in label else ''
    print(f"  {i:>4} {label:<45} {r:>+9.3f} {p:>9.4f} {n:>4} {sig}{current}")


# ============================================================================
# PART 4: FULL CORRELATION SUITE — SLAP vs DC-only
# ============================================================================
print(f"\n\n{'='*120}")
print("PART 4: FULL CORRELATION SUITE — SLAP V5 vs DC-only")
print("=" * 120)

# Build NFL season-level data for seasons_over_10ppg_3yr metric
# Load NFL season stats to compute this
nfl_stats_file = 'data/nflverse/player_season_stats_2025.csv'
if os.path.exists(nfl_stats_file):
    nfl_stats = pd.read_csv(nfl_stats_file)
    nfl_stats['name_norm'] = nfl_stats['player_display_name'].apply(normalize_name)
    nfl_stats = nfl_stats[nfl_stats['position'] == 'RB'].copy()
    nfl_stats['games'] = pd.to_numeric(nfl_stats.get('games', pd.Series(dtype=float)), errors='coerce')

    # For each backtest RB, count seasons with 10+ PPG in first 3 years
    rb_bt['seasons_10ppg_3yr'] = 0
    for idx in rb_bt.index:
        name_n = rb_bt.loc[idx, 'name_norm']
        dy = int(rb_bt.loc[idx, 'draft_year'])
        first_3_years = [dy, dy + 1, dy + 2]
        player_seasons = nfl_stats[
            (nfl_stats['name_norm'] == name_n) &
            (nfl_stats['season'].isin(first_3_years))
        ]
        count = 0
        for _, ps in player_seasons.iterrows():
            games = ps.get('games', 0)
            ppg = ps.get('fantasy_points_ppr', 0) / max(1, games) if pd.notna(games) and games >= 8 else 0
            if ppg >= 10:
                count += 1
        rb_bt.loc[idx, 'seasons_10ppg_3yr'] = count
else:
    rb_bt['seasons_10ppg_3yr'] = np.nan

# Outcomes to test
print(f"\n  Testing SLAP V5 and DC-only against all outcomes:")

outcomes_list = [
    ('hit24', 'Top-24 PPR season (binary)', True),
    ('hit12', 'Top-12 PPR season (binary)', True),
    ('first_3yr_ppg', 'First 3yr best PPG (continuous)', False),
    ('career_ppg', 'Career best PPG (continuous)', False),
    ('seasons_10ppg_3yr', 'Seasons 10+ PPG in first 3yr (0-3)', False),
]

print(f"\n  {'Outcome':<40} | {'Metric':>10} | {'Pearson r':>9} {'p':>8} | {'AUC':>6} | {'n':>4}")
print(f"  {'-'*90}")

for outcome_col, outcome_label, is_binary in outcomes_list:
    for metric_col, metric_label in [('slap_v5', 'SLAP V5'), ('s_dc', 'DC only')]:
        mask = rb_bt[outcome_col].notna() & rb_bt[metric_col].notna()
        x = rb_bt.loc[mask, metric_col]
        y = rb_bt.loc[mask, outcome_col]
        n = len(x)

        if n < 10:
            print(f"  {outcome_label:<40} | {metric_label:>10} | {'n/a':>9} {'n/a':>8} | {'n/a':>6} | {n:>4}")
            continue

        r, p = sp_stats.pearsonr(x, y)
        p_str = f"{p:.4f}" if p >= 0.0001 else "<.0001"

        auc_str = 'n/a'
        if is_binary and y.nunique() == 2:
            try:
                auc = roc_auc_score(y, x)
                auc_str = f"{auc:.3f}"
            except Exception:
                pass
        elif not is_binary:
            # For continuous, use top-quartile as pseudo-binary for AUC
            threshold = y.quantile(0.75)
            y_bin = (y >= threshold).astype(int)
            if y_bin.nunique() == 2:
                try:
                    auc = roc_auc_score(y_bin, x)
                    auc_str = f"{auc:.3f}"
                except Exception:
                    pass

        sig = '**' if p < 0.01 else ' *' if p < 0.05 else '  '
        print(f"  {outcome_label:<40} | {metric_label:>10} | {r:>+9.3f} {p_str:>8}{sig} | {auc_str:>6} | {n:>4}")

    # Show SLAP advantage
    mask2 = rb_bt[outcome_col].notna()
    x_slap = rb_bt.loc[mask2, 'slap_v5']
    x_dc = rb_bt.loc[mask2, 's_dc']
    y2 = rb_bt.loc[mask2, outcome_col]
    if len(y2) >= 10:
        r_slap, _ = sp_stats.pearsonr(x_slap, y2)
        r_dc, _ = sp_stats.pearsonr(x_dc, y2)
        delta = r_slap - r_dc
        print(f"  {'':40} | {'ΔSLAP-DC':>10} | {delta:>+9.3f} {'':>8}   | {'':>6} |")
    print(f"  {'-'*90}")

# Top-decile hit rates
print(f"\n\n  ── TOP-DECILE HIT RATES ──")
print(f"  (Do top-10% SLAP scores hit more often than top-10% DC-only?)")

for metric_col, metric_label in [('slap_v5', 'SLAP V5'), ('s_dc', 'DC only')]:
    top_10_thresh = rb_bt[metric_col].quantile(0.90)
    top_10 = rb_bt[rb_bt[metric_col] >= top_10_thresh]
    n_top = len(top_10)
    h24 = top_10['hit24'].mean() * 100 if top_10['hit24'].notna().sum() > 0 else 0
    h12 = top_10['hit12'].mean() * 100 if top_10['hit12'].notna().sum() > 0 else 0
    ppg = top_10['first_3yr_ppg'].mean() if top_10['first_3yr_ppg'].notna().sum() > 0 else 0
    print(f"\n  {metric_label} top 10% (n={n_top}, threshold={top_10_thresh:.1f}):")
    print(f"    hit24 rate: {h24:.1f}%")
    print(f"    hit12 rate: {h12:.1f}%")
    print(f"    avg first_3yr_ppg: {ppg:.1f}")

# Spearman (rank) correlations
print(f"\n\n  ── SPEARMAN (RANK) CORRELATIONS ──")
print(f"  {'Outcome':<40} | {'SLAP V5 ρ':>10} | {'DC only ρ':>10} | {'Δ':>6}")
print(f"  {'-'*75}")
for outcome_col, outcome_label, _ in outcomes_list:
    mask = rb_bt[outcome_col].notna()
    if mask.sum() < 10:
        continue
    rho_slap, _ = sp_stats.spearmanr(rb_bt.loc[mask, 'slap_v5'], rb_bt.loc[mask, outcome_col])
    rho_dc, _ = sp_stats.spearmanr(rb_bt.loc[mask, 's_dc'], rb_bt.loc[mask, outcome_col])
    print(f"  {outcome_label:<40} | {rho_slap:>+10.3f} | {rho_dc:>+10.3f} | {rho_slap-rho_dc:>+6.3f}")

print(f"\n\n  Nothing changed. This is diagnostic only.")
