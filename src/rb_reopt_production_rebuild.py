"""
RB Reoptimization — Rebuild Production Component from the Ground Up
=====================================================================
Step 1: Inventory all sub-metrics and partial correlations
Step 2: Build candidate composite production scores
Step 3: Test best composite in full model
"""

import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq
import glob
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('/home/user/SlapModelV3')

# ============================================================================
# HELPERS
# ============================================================================

def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

def normalize_0_100(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50, index=series.index)
    return (series - mn) / (mx - mn) * 100

def normalize_name(name):
    if pd.isna(name):
        return ''
    return str(name).strip().lower().replace('.', '').replace("'", '').replace('-', ' ').replace(' jr', '').replace(' sr', '').replace(' iii', '').replace(' ii', '').strip()

def rb_production_score(row):
    """Current production metric: rec_yards / team_pass_att * age_weight / 1.75"""
    if pd.isna(row.get('rec_yards')) or pd.isna(row.get('team_pass_att')) or row.get('team_pass_att', 0) == 0:
        return np.nan
    age = row.get('age', 22)
    if pd.isna(age):
        age = 22
    age_weights = {19: 1.15, 20: 1.10, 21: 1.05, 22: 1.00, 23: 0.95}
    age_w = age_weights.get(int(age), 0.90 if age >= 24 else 1.15)
    raw = (row['rec_yards'] / row['team_pass_att']) * age_w * 100
    return min(99.9, raw / 1.75)

def partial_corr(df, x_col, y_col, control_cols):
    """Partial correlation of x with y after controlling for controls."""
    valid = df[[x_col, y_col] + control_cols].dropna()
    if len(valid) < 20:
        return np.nan, np.nan, len(valid)
    X = np.column_stack([valid[c].values for c in control_cols] + [np.ones(len(valid))])
    # Residualize x
    beta_x, _, _, _ = lstsq(X, valid[x_col].values, rcond=None)
    resid_x = valid[x_col].values - X @ beta_x
    # Residualize y
    beta_y, _, _, _ = lstsq(X, valid[y_col].values, rcond=None)
    resid_y = valid[y_col].values - X @ beta_y
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(valid)

def auc_roc(labels, scores):
    valid = pd.DataFrame({'y': labels, 's': scores}).dropna()
    if len(valid) < 10:
        return np.nan
    pos = valid[valid['y'] == 1]['s']
    neg = valid[valid['y'] == 0]['s']
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    auc_sum = 0
    for p_val in pos:
        auc_sum += (neg < p_val).sum() + 0.5 * (neg == p_val).sum()
    return auc_sum / (len(pos) * len(neg))

def brier_score(labels, scores_0_100):
    valid = pd.DataFrame({'y': labels, 's': scores_0_100}).dropna()
    if len(valid) < 10:
        return np.nan
    prob = valid['s'] / 100
    return ((prob - valid['y']) ** 2).mean()

outcome_cols = ['first_3yr_ppg', 'hit24', 'hit12', 'career_ppg']
outcome_weights = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}

# ============================================================================
# DATA LOADING
# ============================================================================

print("=" * 120)
print("LOADING DATA SOURCES...")
print("=" * 120)

# 1. Core backtest
rb = pd.read_csv('data/rb_backtest_with_receiving.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
out_rb = outcomes[outcomes['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
rb = rb.merge(out_rb, on=['player_name', 'draft_year', 'pick'], how='left')
rb['s_dc'] = rb['pick'].apply(dc_score)
rb['s_rec_prod'] = rb.apply(rb_production_score, axis=1)
rb['name_norm'] = rb['player_name'].apply(normalize_name)

print(f"  Core backtest: {len(rb)} RBs, {rb['draft_year'].min()}-{rb['draft_year'].max()}")
print(f"  With outcomes (hit24): {rb['hit24'].notna().sum()}")
print(f"  With rec_prod: {rb['s_rec_prod'].notna().sum()}")

# 2. PFF Receiving Data — load all receiving_summary files
season_file_map = {
    2015: 'receiving_summary (2).csv',
    2016: 'receiving_summary (3).csv',
    2017: 'receiving_summary (4).csv',
    2018: 'receiving_summary (5).csv',
    2019: 'receiving_summary (21).csv',
    2020: 'receiving_summary (20).csv',
    2021: 'receiving_summary (19).csv',
    2022: 'receiving_summary (18).csv',
    2023: 'receiving_summary (17).csv',
    2024: 'receiving_summary (16).csv',
}

pff_recv_all = []
for season, fname in season_file_map.items():
    df = pd.read_csv(f'data/{fname}')
    df['college_season'] = season
    hbs = df[df['position'] == 'HB'].copy()
    hbs['name_norm'] = hbs['player'].apply(normalize_name)
    pff_recv_all.append(hbs)

pff_recv = pd.concat(pff_recv_all, ignore_index=True)
print(f"  PFF receiving: {len(pff_recv)} HB-season records across {len(season_file_map)} seasons")

# 3. PFF Rushing Data (already matched)
pff_rush = pd.read_csv('data/rb_pff_corrected.csv')
pff_rush['name_norm'] = pff_rush['player_name'].apply(normalize_name)
print(f"  PFF rushing (pre-matched): {len(pff_rush)} RBs")

# 4. College receiving multi-season
college_recv = pd.read_csv('data/college_receiving_2011_2023.csv')
print(f"  College receiving multi-season: {len(college_recv)} records")

# ============================================================================
# MATCH PFF RECEIVING DATA TO BACKTEST RBs
# ============================================================================

print(f"\n{'=' * 120}")
print("MATCHING PFF RECEIVING DATA TO BACKTEST RBs...")
print("=" * 120)

# For each backtest RB, find their final college season PFF receiving data
# Final season = draft_year - 1

def match_pff_recv(rb_row, pff_recv_df):
    """Match a backtest RB to their PFF receiving summary for final college season."""
    target_season = rb_row['draft_year'] - 1
    name = rb_row['name_norm']
    college = str(rb_row.get('college', '')).upper().strip()

    # Filter to correct season
    season_data = pff_recv_df[pff_recv_df['college_season'] == target_season]
    if len(season_data) == 0:
        return None

    # Try exact name match
    matches = season_data[season_data['name_norm'] == name]
    if len(matches) == 1:
        return matches.iloc[0]
    if len(matches) > 1:
        # Multiple matches — try to match by team/college
        for _, m in matches.iterrows():
            if college and college in str(m.get('team_name', '')).upper():
                return m
        return matches.iloc[0]

    # Try partial name match (last name)
    last_name = name.split()[-1] if name else ''
    if last_name and len(last_name) > 3:
        partial = season_data[season_data['name_norm'].str.contains(last_name, na=False)]
        if len(partial) == 1:
            return partial.iloc[0]
        if len(partial) > 1 and college:
            for _, m in partial.iterrows():
                if college in str(m.get('team_name', '')).upper():
                    return m

    return None

# Match all RBs
pff_recv_cols = ['yprr', 'avg_depth_of_target', 'grades_offense', 'grades_pass_route',
                 'yards', 'receptions', 'targets', 'routes', 'route_rate',
                 'yards_after_catch', 'yards_after_catch_per_reception',
                 'yards_per_reception', 'caught_percent', 'drop_rate',
                 'contested_catch_rate', 'touchdowns', 'avoided_tackles',
                 'grades_hands_drop', 'first_downs', 'player_game_count']

for col in pff_recv_cols:
    rb[f'pff_recv_{col}'] = np.nan

matched_count = 0
for idx, row in rb.iterrows():
    match = match_pff_recv(row, pff_recv)
    if match is not None:
        matched_count += 1
        for col in pff_recv_cols:
            rb.at[idx, f'pff_recv_{col}'] = match.get(col, np.nan)

print(f"  PFF receiving matched: {matched_count}/{len(rb)} ({matched_count/len(rb)*100:.1f}%)")

# Match PFF rushing data
pff_rush_cols = ['elusive_rating', 'yards', 'attempts', 'yco_attempt', 'grades_run',
                 'grades_offense']
for col in pff_rush_cols:
    rb[f'pff_rush_{col}'] = np.nan

rush_matched = 0
for idx, row in rb.iterrows():
    name = row['name_norm']
    dy = row['draft_year']
    matches = pff_rush[(pff_rush['name_norm'] == name) & (pff_rush['draft_year'] == dy)]
    if len(matches) == 0:
        last_name = name.split()[-1] if name else ''
        if last_name and len(last_name) > 3:
            matches = pff_rush[(pff_rush['name_norm'].str.contains(last_name, na=False)) &
                               (pff_rush['draft_year'] == dy)]
    if len(matches) >= 1:
        rush_matched += 1
        m = matches.iloc[0]
        for col in pff_rush_cols:
            rb.at[idx, f'pff_rush_{col}'] = m.get(col, np.nan)

print(f"  PFF rushing matched: {rush_matched}/{len(rb)} ({rush_matched/len(rb)*100:.1f}%)")

# ============================================================================
# BUILD ALL SUB-METRICS
# ============================================================================

print(f"\n{'=' * 120}")
print("BUILDING SUB-METRICS...")
print("=" * 120)

# Force numeric on PFF columns
for col in rb.columns:
    if col.startswith('pff_'):
        rb[col] = pd.to_numeric(rb[col], errors='coerce')

# --- Current metric: Rec yards per team pass attempt (age-weighted) ---
# Already computed as s_rec_prod

# --- Career RYPTPA: all college seasons ---
college_recv['name_norm'] = college_recv['college_name'].apply(normalize_name) if 'college_name' in college_recv.columns else ''
# This needs different approach - use college_receiving_2011_2023.csv
# For now, compute from what we have in the backtest

# --- PFF YPRR ---
rb['m_yprr'] = rb['pff_recv_yprr']

# --- PFF Receiving Grade ---
rb['m_recv_grade'] = rb['pff_recv_grades_offense']

# --- PFF Route Grade ---
rb['m_route_grade'] = rb['pff_recv_grades_pass_route']

# --- PFF ADOT ---
rb['m_adot'] = rb['pff_recv_avg_depth_of_target']

# --- Receiving yard share (dominator for RBs) ---
# rec_yards / (approximate team receiving yards)
# Team receiving yards ≈ team_pass_att * ~7 (avg yards per attempt)
# Actually, use a simpler proxy: rec_yards / team_pass_att already IS a share metric
# Let's also try: PFF receiving yards / total team routes * some factor
rb['m_rec_yard_share'] = np.nan
for idx, row in rb.iterrows():
    if pd.notna(row.get('rec_yards')) and pd.notna(row.get('team_pass_att')) and row['team_pass_att'] > 0:
        # rec_yards as % of estimated team passing yards (team_pass_att * ~7)
        est_team_pass_yards = row['team_pass_att'] * 7
        rb.at[idx, 'm_rec_yard_share'] = (row['rec_yards'] / est_team_pass_yards) * 100

# --- Receiving TD share ---
# We need team pass TDs... approximate from data
# Use PFF touchdowns / team pass attempts as proxy
rb['m_td_per_team_pass'] = np.nan
for idx, row in rb.iterrows():
    td = row.get('pff_recv_touchdowns', np.nan)
    tpa = row.get('team_pass_att', np.nan)
    if pd.notna(td) and pd.notna(tpa) and tpa > 0:
        rb.at[idx, 'm_td_per_team_pass'] = (td / tpa) * 100

# --- Rushing yards per game ---
rb['m_rush_ypg'] = np.nan
for idx, row in rb.iterrows():
    rush_yds = row.get('pff_rush_yards', np.nan)
    games = row.get('pff_recv_player_game_count', np.nan)
    if pd.isna(games):
        # Try to estimate from rushing data
        games_rush = row.get('pff_rush_attempts', np.nan)
        if pd.notna(games_rush) and pd.notna(rush_yds):
            # Can't get games from attempts alone, skip
            pass
    if pd.notna(rush_yds) and pd.notna(games) and games > 0:
        rb.at[idx, 'm_rush_ypg'] = rush_yds / games

# --- Rush + Rec combined per game ---
rb['m_total_ypg'] = np.nan
for idx, row in rb.iterrows():
    rush_yds = row.get('pff_rush_yards', np.nan)
    rec_yds = row.get('pff_recv_yards', np.nan)
    games = row.get('pff_recv_player_game_count', np.nan)
    if pd.notna(rush_yds) and pd.notna(rec_yds) and pd.notna(games) and games > 0:
        rb.at[idx, 'm_total_ypg'] = (rush_yds + rec_yds) / games

# --- Total scrimmage share ---
# (rush + rec yards) / (team rush + pass yards) - would need team rush yards
# Approximate: (rush + rec) / (team_pass_att * 7 + team_rush_att * 4)
# We don't have team rush attempts... use rec yards share for now

# --- PFF rushing grade ---
rb['m_rush_grade'] = rb['pff_rush_grades_run']

# --- PFF elusive rating ---
rb['m_elusive'] = rb['pff_rush_elusive_rating']

# --- PFF yards after contact per attempt ---
rb['m_yco'] = rb['pff_rush_yco_attempt']

# --- Catch rate ---
rb['m_catch_pct'] = rb['pff_recv_caught_percent']

# --- Drop rate (inverse — lower is better) ---
rb['m_drop_rate_inv'] = rb['pff_recv_drop_rate'].apply(lambda x: 100 - x if pd.notna(x) else np.nan)

# --- Route rate (% of pass plays where RB runs a route) ---
rb['m_route_rate'] = rb['pff_recv_route_rate']

# --- YAC per reception ---
rb['m_yac_per_rec'] = rb['pff_recv_yards_after_catch_per_reception']

# --- Avoided tackles per reception ---
rb['m_avoid_tackles'] = np.nan
for idx, row in rb.iterrows():
    at = row.get('pff_recv_avoided_tackles', np.nan)
    rec = row.get('pff_recv_receptions', np.nan)
    if pd.notna(at) and pd.notna(rec) and rec > 0:
        rb.at[idx, 'm_avoid_tackles'] = at / rec

# --- Receptions per game ---
rb['m_rec_per_game'] = np.nan
for idx, row in rb.iterrows():
    rec = row.get('pff_recv_receptions', np.nan)
    games = row.get('pff_recv_player_game_count', np.nan)
    if pd.notna(rec) and pd.notna(games) and games > 0:
        rb.at[idx, 'm_rec_per_game'] = rec / games

# --- Targets per game ---
rb['m_tgt_per_game'] = np.nan
for idx, row in rb.iterrows():
    tgt = row.get('pff_recv_targets', np.nan)
    games = row.get('pff_recv_player_game_count', np.nan)
    if pd.notna(tgt) and pd.notna(games) and games > 0:
        rb.at[idx, 'm_tgt_per_game'] = tgt / games

# --- PFF overall offensive grade (rushing data) ---
rb['m_off_grade_rush'] = rb['pff_rush_grades_offense']

# ============================================================================
# BUILD CAREER (MULTI-SEASON) METRICS
# ============================================================================

# Career RYPTPA from college_receiving_2011_2023.csv
# Parse multi-season data
college_recv_cols = list(college_recv.columns)
print(f"  College receiving columns: {college_recv_cols}")
print(f"  College receiving sample:")
print(college_recv.head(3).to_string())

# Build career RYPTPA: sum all seasons of rec_yards / sum of team_pass_att, age-weighted
# First check if we can match to backtest RBs
college_recv['name_norm'] = college_recv.apply(
    lambda r: normalize_name(r.get('college_name', r.get('player_name', ''))), axis=1)

# Build career stats per player
# Group by player and sum across seasons
career_stats = {}
for name_col in ['college_name', 'player_name']:
    if name_col in college_recv.columns:
        break

# Check structure more carefully
if 'playerId' in college_recv.columns:
    # Group by playerId for accuracy
    for pid, group in college_recv.groupby('playerId'):
        if group['position'].iloc[0] != 'RB' and group['position'].iloc[0] != 'HB':
            # Check if position column exists and filter
            pass
        name = normalize_name(group.iloc[0].get(name_col, ''))
        total_rec = group['rec_yards'].sum() if 'rec_yards' in group.columns else 0
        total_tpa = group['team_pass_att'].sum() if 'team_pass_att' in group.columns else 0
        n_seasons = len(group)
        if total_tpa > 0:
            career_stats[name] = {
                'career_rec_yards': total_rec,
                'career_team_pass_att': total_tpa,
                'career_ryptpa': total_rec / total_tpa,
                'n_seasons': n_seasons,
            }

rb['m_career_ryptpa'] = rb['name_norm'].map(
    lambda n: career_stats.get(n, {}).get('career_ryptpa', np.nan))

print(f"  Career RYPTPA matched: {rb['m_career_ryptpa'].notna().sum()}/{len(rb)}")

# Career dominator from college receiving
rb['m_career_dominator'] = np.nan
if 'dominator' in college_recv.columns:
    # Average dominator across seasons
    dom_stats = {}
    for name_col_check in ['college_name', 'player_name']:
        if name_col_check in college_recv.columns:
            for _, group in college_recv.groupby(college_recv[name_col_check].apply(normalize_name)):
                name = normalize_name(group.iloc[0][name_col_check])
                avg_dom = group['dominator'].mean()
                if pd.notna(avg_dom):
                    dom_stats[name] = avg_dom
            break
    rb['m_career_dominator'] = rb['name_norm'].map(lambda n: dom_stats.get(n, np.nan))
    print(f"  Career dominator matched: {rb['m_career_dominator'].notna().sum()}/{len(rb)}")

# ============================================================================
# STEP 1: COVERAGE & PARTIAL CORRELATIONS
# ============================================================================

rb_eval = rb[rb['hit24'].notna()].copy()
print(f"\n  Eval sample: {len(rb_eval)} RBs with outcomes")

all_metrics = {
    'RYPTPA (current)':      's_rec_prod',
    'PFF YPRR':              'm_yprr',
    'PFF Recv Grade':        'm_recv_grade',
    'PFF Route Grade':       'm_route_grade',
    'PFF ADOT':              'm_adot',
    'Rec Yard Share':        'm_rec_yard_share',
    'TD/Team Pass Att':      'm_td_per_team_pass',
    'Rush YPG':              'm_rush_ypg',
    'Total YPG (R+R)':      'm_total_ypg',
    'PFF Rush Grade':        'm_rush_grade',
    'PFF Elusive Rating':    'm_elusive',
    'PFF YCO/Attempt':       'm_yco',
    'Catch %':               'm_catch_pct',
    'Drop Rate (inv)':       'm_drop_rate_inv',
    'Route Rate':            'm_route_rate',
    'YAC/Rec':               'm_yac_per_rec',
    'Avoid Tackles/Rec':     'm_avoid_tackles',
    'Rec/Game':              'm_rec_per_game',
    'Targets/Game':          'm_tgt_per_game',
    'PFF Off Grade (rush)':  'm_off_grade_rush',
    'Career RYPTPA':         'm_career_ryptpa',
    'Career Dominator':      'm_career_dominator',
}

print(f"\n\n{'=' * 120}")
print("STEP 1: COVERAGE & PARTIAL CORRELATIONS (controlling for DC)")
print(f"{'=' * 120}")

print(f"\n  {'Metric':<25} {'Cover':>6} {'%':>5}  {'part r(3yr)':>12} {'p':>8} {'part r(h24)':>12} {'p':>8} {'part r(h12)':>12} {'p':>8} {'part r(car)':>12} {'p':>8}")
print("  " + "-" * 130)

metric_results = []
for label, col in all_metrics.items():
    coverage = rb_eval[col].notna().sum()
    pct = coverage / len(rb_eval) * 100

    results = {'label': label, 'col': col, 'coverage': coverage, 'pct': pct}

    row_str = f"  {label:<25} {coverage:>4}/{len(rb_eval)} {pct:>4.0f}%"

    for out in outcome_cols:
        r, p, n = partial_corr(rb_eval, col, out, ['s_dc'])
        results[f'pr_{out}'] = r
        results[f'pp_{out}'] = p
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
        if np.isnan(r):
            row_str += f" {'N/A':>12} {'':>8}"
        else:
            row_str += f" {r:>+11.4f}{sig} {p:>7.4f}"

    print(row_str)
    metric_results.append(results)

# Sort by partial r with first_3yr_ppg (40% weight)
print(f"\n\n  ── RANKED BY partial r(first_3yr_ppg) | controlling for DC ──")
sorted_by_ppg = sorted(metric_results, key=lambda x: x.get('pr_first_3yr_ppg', -999), reverse=True)
print(f"  {'Rank':>4} {'Metric':<25} {'Cover':>6} {'pr(3yr_ppg)':>12} {'pr(hit24)':>10} {'pr(hit12)':>10} {'pr(career)':>10}")
print("  " + "-" * 85)
for i, m in enumerate(sorted_by_ppg, 1):
    if m['coverage'] < 20:
        continue
    vals = [m.get(f'pr_{out}', np.nan) for out in outcome_cols]
    avg_pr = np.nanmean(vals)
    row = f"  {i:>4} {m['label']:<25} {m['coverage']:>4}/{len(rb_eval)}"
    for out in outcome_cols:
        v = m.get(f'pr_{out}', np.nan)
        p = m.get(f'pp_{out}', 1)
        sig = '*' if p < 0.05 else '†' if p < 0.10 else ' '
        row += f" {v:>+9.4f}{sig}" if not np.isnan(v) else f" {'N/A':>10}"
    print(row)


# ============================================================================
# STEP 1b: Also test partial correlation controlling for DC + current rec prod
# (Does metric add value BEYOND what we already have?)
# ============================================================================

print(f"\n\n{'=' * 120}")
print("STEP 1b: PARTIAL CORRELATIONS CONTROLLING FOR DC + CURRENT RYPTPA")
print("(Does each metric add value BEYOND what rec production already captures?)")
print("=" * 120)

print(f"\n  {'Metric':<25} {'Cover':>6}  {'pr(3yr|DC,rec)':>15} {'p':>8} {'pr(h24|DC,rec)':>15} {'p':>8} {'pr(h12|DC,rec)':>15} {'p':>8} {'pr(car|DC,rec)':>15} {'p':>8}")
print("  " + "-" * 140)

metric_results_2 = []
avg_rec = rb_eval['s_rec_prod'].mean()
rb_eval['s_rec_prod_f'] = rb_eval['s_rec_prod'].fillna(avg_rec)

for label, col in all_metrics.items():
    if col == 's_rec_prod':
        continue  # Skip the baseline itself
    coverage = rb_eval[col].notna().sum()
    if coverage < 20:
        continue

    results = {'label': label, 'col': col, 'coverage': coverage}
    row_str = f"  {label:<25} {coverage:>4}/{len(rb_eval)}"

    for out in outcome_cols:
        r, p, n = partial_corr(rb_eval, col, out, ['s_dc', 's_rec_prod_f'])
        results[f'pr_{out}'] = r
        results[f'pp_{out}'] = p
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
        row_str += f" {r:>+14.4f}{sig} {p:>7.4f}"

    print(row_str)
    metric_results_2.append(results)

# Rank by average significance
print(f"\n\n  ── TOP METRICS BY INCREMENTAL VALUE (avg partial r after DC + rec prod) ──")
for m in metric_results_2:
    vals = [abs(m.get(f'pr_{out}', 0)) for out in outcome_cols if not np.isnan(m.get(f'pr_{out}', np.nan))]
    m['avg_abs_pr'] = np.mean(vals) if vals else 0
    sig_count = sum(1 for out in outcome_cols if m.get(f'pp_{out}', 1) < 0.10)
    m['sig_count'] = sig_count

sorted_by_value = sorted(metric_results_2, key=lambda x: (-x['sig_count'], -x['avg_abs_pr']))
print(f"  {'Rank':>4} {'Metric':<25} {'Cover':>6} {'Sig(p<.10)':>10} {'Avg|pr|':>10} {'pr(3yr)':>10} {'pr(h24)':>10} {'pr(h12)':>10} {'pr(car)':>10}")
print("  " + "-" * 100)
for i, m in enumerate(sorted_by_value[:15], 1):
    row = f"  {i:>4} {m['label']:<25} {m['coverage']:>4}/{len(rb_eval)} {m['sig_count']:>10} {m['avg_abs_pr']:>10.4f}"
    for out in outcome_cols:
        v = m.get(f'pr_{out}', np.nan)
        p = m.get(f'pp_{out}', 1)
        sig = '*' if p < 0.05 else '†' if p < 0.10 else ' '
        row += f" {v:>+8.4f}{sig}" if not np.isnan(v) else f" {'N/A':>10}"
    print(row)


# ============================================================================
# STEP 2: BUILD CANDIDATE COMPOSITE SCORES
# ============================================================================

print(f"\n\n{'=' * 120}")
print("STEP 2: CANDIDATE COMPOSITE PRODUCTION SCORES")
print("=" * 120)

# Normalize all sub-metrics to 0-100 for compositing
norm_cols = {}
for label, col in all_metrics.items():
    valid = rb_eval[col].notna()
    if valid.sum() < 20:
        continue
    norm_col = f'n_{col}'
    rb_eval[norm_col] = np.nan
    rb_eval.loc[valid, norm_col] = normalize_0_100(rb_eval.loc[valid, col])
    avg_val = rb_eval[norm_col].mean()
    rb_eval[f'{norm_col}_f'] = rb_eval[norm_col].fillna(avg_val)
    norm_cols[col] = f'{norm_col}_f'

# Current baseline: single-metric rec prod
rb_eval['comp_current'] = rb_eval['s_rec_prod'].fillna(rb_eval['s_rec_prod'].mean())

# Composite A: Receiving-only blend (RYPTPA + YPRR + RB dominator)
rb_eval['comp_A'] = (
    rb_eval[norm_cols.get('s_rec_prod', 'comp_current')] * 0.50 +
    rb_eval.get(norm_cols.get('m_yprr', 'comp_current'), rb_eval['comp_current']) * 0.30 +
    rb_eval.get(norm_cols.get('m_rec_yard_share', 'comp_current'), rb_eval['comp_current']) * 0.20
)

# Composite B: Rush & Rec blend (RYPTPA + Rush YPG + YPRR)
rb_eval['comp_B'] = (
    rb_eval[norm_cols.get('s_rec_prod', 'comp_current')] * 0.40 +
    rb_eval.get(norm_cols.get('m_rush_ypg', 'comp_current'), rb_eval['comp_current']) * 0.30 +
    rb_eval.get(norm_cols.get('m_yprr', 'comp_current'), rb_eval['comp_current']) * 0.30
)

# Composite C: Full blend (RYPTPA + YPRR + dominator + rush + TD share)
rb_eval['comp_C'] = (
    rb_eval[norm_cols.get('s_rec_prod', 'comp_current')] * 0.30 +
    rb_eval.get(norm_cols.get('m_yprr', 'comp_current'), rb_eval['comp_current']) * 0.20 +
    rb_eval.get(norm_cols.get('m_rec_yard_share', 'comp_current'), rb_eval['comp_current']) * 0.15 +
    rb_eval.get(norm_cols.get('m_rush_ypg', 'comp_current'), rb_eval['comp_current']) * 0.20 +
    rb_eval.get(norm_cols.get('m_td_per_team_pass', 'comp_current'), rb_eval['comp_current']) * 0.15
)

# Composite D: Career blend (career RYPTPA + career dominator)
if 'm_career_ryptpa' in norm_cols and 'm_career_dominator' in norm_cols:
    rb_eval['comp_D'] = (
        rb_eval[norm_cols['m_career_ryptpa']] * 0.60 +
        rb_eval[norm_cols['m_career_dominator']] * 0.40
    )
elif 'm_career_ryptpa' in norm_cols:
    rb_eval['comp_D'] = rb_eval[norm_cols['m_career_ryptpa']]
else:
    rb_eval['comp_D'] = rb_eval['comp_current']

# Composite E: PFF-heavy blend (YPRR + recv grade + route grade + ADOT)
rb_eval['comp_E'] = (
    rb_eval.get(norm_cols.get('m_yprr', 'comp_current'), rb_eval['comp_current']) * 0.35 +
    rb_eval.get(norm_cols.get('m_recv_grade', 'comp_current'), rb_eval['comp_current']) * 0.25 +
    rb_eval.get(norm_cols.get('m_route_grade', 'comp_current'), rb_eval['comp_current']) * 0.25 +
    rb_eval.get(norm_cols.get('m_adot', 'comp_current'), rb_eval['comp_current']) * 0.15
)

# Composite F: Best signals blend (whatever showed highest partial correlations)
# Will be populated after seeing results
# For now: RYPTPA + whatever has best incremental signal
rb_eval['comp_F'] = rb_eval['comp_current']  # placeholder

# Composite G: RYPTPA + route metrics
rb_eval['comp_G'] = (
    rb_eval[norm_cols.get('s_rec_prod', 'comp_current')] * 0.50 +
    rb_eval.get(norm_cols.get('m_route_rate', 'comp_current'), rb_eval['comp_current']) * 0.25 +
    rb_eval.get(norm_cols.get('m_rec_per_game', 'comp_current'), rb_eval['comp_current']) * 0.25
)

# Composite H: RYPTPA + rushing efficiency
rb_eval['comp_H'] = (
    rb_eval[norm_cols.get('s_rec_prod', 'comp_current')] * 0.50 +
    rb_eval.get(norm_cols.get('m_rush_grade', 'comp_current'), rb_eval['comp_current']) * 0.25 +
    rb_eval.get(norm_cols.get('m_elusive', 'comp_current'), rb_eval['comp_current']) * 0.25
)

composites = {
    'Current (RYPTPA only)': 'comp_current',
    'A: Recv blend (RYPTPA+YPRR+Share)': 'comp_A',
    'B: Rush+Recv (RYPTPA+RushYPG+YPRR)': 'comp_B',
    'C: Full blend (5 metrics)': 'comp_C',
    'D: Career blend (career RYPTPA+dom)': 'comp_D',
    'E: PFF-heavy (YPRR+grades+ADOT)': 'comp_E',
    'G: RYPTPA + route metrics': 'comp_G',
    'H: RYPTPA + rush efficiency': 'comp_H',
}

# Evaluate each composite
def eval_composite(df, comp_col, dc_col='s_dc'):
    """Full evaluation of a production composite in the model."""
    res = {}

    # Partial correlation with DC controlled
    for out in outcome_cols:
        r, p, n = partial_corr(df, comp_col, out, [dc_col])
        res[f'pr_{out}'] = r
        res[f'pp_{out}'] = p

    # As a SLAP component at 65/35
    slap = df[dc_col] * 0.65 + df[comp_col] * 0.35

    for out in outcome_cols:
        valid = pd.DataFrame({'s': slap, 'o': df[out]}).dropna()
        if len(valid) >= 10:
            r, _ = stats.pearsonr(valid['s'], valid['o'])
            res[f'r_{out}'] = r

    # Priority-weighted avg
    pri = sum(outcome_weights[o] * res.get(f'r_{o}', 0) for o in outcome_cols
              if not np.isnan(res.get(f'r_{o}', np.nan)))
    pri_w = sum(outcome_weights[o] for o in outcome_cols
                if not np.isnan(res.get(f'r_{o}', np.nan)))
    res['pri_avg'] = pri / pri_w if pri_w > 0 else np.nan

    # Top decile
    n_top = max(1, len(df) // 10)
    top = df.loc[slap.nlargest(n_top).index]
    res['top10_hit24'] = top['hit24'].mean() * 100
    res['top10_hit12'] = top['hit12'].mean() * 100
    ppg_top = top[top['first_3yr_ppg'].notna()]
    res['top10_ppg'] = ppg_top['first_3yr_ppg'].mean() if len(ppg_top) > 0 else np.nan

    # AUC
    res['auc_hit24'] = auc_roc(df['hit24'], slap)
    res['auc_hit12'] = auc_roc(df['hit12'], slap)

    # Brier
    res['brier_hit24'] = brier_score(df['hit24'], slap)
    res['brier_hit12'] = brier_score(df['hit12'], slap)

    return res

print(f"\n  ── Composite Production Scores vs Current (in DC 65 / Prod 35 model) ──\n")

comp_results = {}
for label, col in composites.items():
    res = eval_composite(rb_eval, col)
    comp_results[label] = res

# Print comparison table
print(f"  {'Composite':<40} {'PRI-AVG':>8} {'r(3yr)':>8} {'r(h24)':>8} {'r(h12)':>8} {'r(car)':>8} {'T10%h24':>8} {'T10%h12':>8} {'AUC h24':>8} {'AUC h12':>8}")
print("  " + "-" * 120)

base_pri = comp_results['Current (RYPTPA only)']['pri_avg']
for label, res in comp_results.items():
    d_pri = res['pri_avg'] - base_pri
    marker = " ◄ BASE" if 'Current' in label else f" ({d_pri:+.4f})"
    print(f"  {label:<40} {res['pri_avg']:>+.4f} {res.get('r_first_3yr_ppg', np.nan):>+.4f} {res.get('r_hit24', np.nan):>+.4f} {res.get('r_hit12', np.nan):>+.4f} {res.get('r_career_ppg', np.nan):>+.4f} {res['top10_hit24']:>7.1f}% {res['top10_hit12']:>7.1f}% {res['auc_hit24']:>.4f}   {res['auc_hit12']:>.4f}  {marker}")

# Print partial correlations (incremental value as production component)
print(f"\n\n  ── Partial Correlations (composite signal after controlling for DC) ──\n")
print(f"  {'Composite':<40} {'pr(3yr)':>10} {'p':>8} {'pr(h24)':>10} {'p':>8} {'pr(h12)':>10} {'p':>8} {'pr(car)':>10} {'p':>8}")
print("  " + "-" * 110)

for label, res in comp_results.items():
    row = f"  {label:<40}"
    for out in outcome_cols:
        r = res.get(f'pr_{out}', np.nan)
        p = res.get(f'pp_{out}', 1)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
        row += f" {r:>+9.4f}{sig} {p:>7.4f}"
    print(row)


# ============================================================================
# STEP 2b: PFF-ONLY SUBSAMPLE (to check if improvements are real)
# ============================================================================

print(f"\n\n{'=' * 120}")
print("STEP 2b: PFF-ONLY SUBSAMPLE (players with actual PFF receiving data)")
print("=" * 120)

pff_mask = rb_eval['pff_recv_yprr'].notna()
rb_pff = rb_eval[pff_mask].copy()
print(f"  PFF subsample: {len(rb_pff)} RBs")

print(f"\n  {'Composite':<40} {'PRI-AVG':>8} {'r(3yr)':>8} {'r(h24)':>8} {'r(h12)':>8} {'T10%h24':>8} {'T10%h12':>8} {'n':>4}")
print("  " + "-" * 90)

for label, col in composites.items():
    res = eval_composite(rb_pff, col)
    d_pri = res['pri_avg'] - comp_results['Current (RYPTPA only)']['pri_avg']
    print(f"  {label:<40} {res['pri_avg']:>+.4f} {res.get('r_first_3yr_ppg', np.nan):>+.4f} {res.get('r_hit24', np.nan):>+.4f} {res.get('r_hit12', np.nan):>+.4f} {res['top10_hit24']:>7.1f}% {res['top10_hit12']:>7.1f}% {len(rb_pff):>4}")


# ============================================================================
# STEP 3: TEST BEST COMPOSITE IN FULL MODEL
# ============================================================================

print(f"\n\n{'=' * 120}")
print("STEP 3: FULL MODEL TEST — BEST COMPOSITE vs CURRENT")
print("=" * 120)

# Find the best composite by PRI-AVG
best_label = max(comp_results.keys(), key=lambda k: comp_results[k]['pri_avg'])
print(f"  Best composite by PRI-AVG: {best_label}")

# Test multiple DC/Production weight splits
def full_model_eval(df, dc_col, prod_col, dc_w, prod_w):
    """Evaluate full model."""
    slap = df[dc_col] * dc_w + df[prod_col] * prod_w
    res = {}
    for out in outcome_cols:
        valid = pd.DataFrame({'s': slap, 'o': df[out]}).dropna()
        if len(valid) >= 10:
            r, _ = stats.pearsonr(valid['s'], valid['o'])
            res[f'r_{out}'] = r
    pri = sum(outcome_weights[o] * res.get(f'r_{o}', 0) for o in outcome_cols
              if not np.isnan(res.get(f'r_{o}', np.nan)))
    pri_w = sum(outcome_weights[o] for o in outcome_cols
                if not np.isnan(res.get(f'r_{o}', np.nan)))
    res['pri_avg'] = pri / pri_w if pri_w > 0 else np.nan
    n_top = max(1, len(df) // 10)
    top = df.loc[slap.nlargest(n_top).index]
    res['top10_hit24'] = top['hit24'].mean() * 100
    res['top10_hit12'] = top['hit12'].mean() * 100
    ppg_top = top[top['first_3yr_ppg'].notna()]
    res['top10_ppg'] = ppg_top['first_3yr_ppg'].mean() if len(ppg_top) > 0 else np.nan
    res['auc_hit24'] = auc_roc(df['hit24'], slap)
    res['auc_hit12'] = auc_roc(df['hit12'], slap)
    res['brier_hit24'] = brier_score(df['hit24'], slap)
    res['brier_hit12'] = brier_score(df['hit12'], slap)
    return res

weight_splits = [
    (1.00, 0.00, 'DC only'),
    (0.75, 0.25, 'DC/Prod 75/25'),
    (0.70, 0.30, 'DC/Prod 70/30'),
    (0.65, 0.35, 'DC/Prod 65/35'),
    (0.60, 0.40, 'DC/Prod 60/40'),
    (0.55, 0.45, 'DC/Prod 55/45'),
    (0.50, 0.50, 'DC/Prod 50/50'),
]

for prod_name, prod_col in [('Current RYPTPA', 'comp_current'), (best_label, composites[best_label])]:
    if prod_name == best_label and prod_col == 'comp_current':
        continue  # Skip if best is current
    print(f"\n  ── {prod_name} across weight splits ──")
    print(f"  {'Config':<25} {'PRI-AVG':>8} {'r(3yr)':>8} {'r(h24)':>8} {'r(h12)':>8} {'r(car)':>8} {'T10%h24':>8} {'AUC h24':>8} {'Brier24':>8}")
    print("  " + "-" * 100)

    for dc_w, prod_w, label in weight_splits:
        res = full_model_eval(rb_eval, 's_dc', prod_col, dc_w, prod_w)
        print(f"  {label:<25} {res['pri_avg']:>+.4f} {res.get('r_first_3yr_ppg', np.nan):>+.4f} {res.get('r_hit24', np.nan):>+.4f} {res.get('r_hit12', np.nan):>+.4f} {res.get('r_career_ppg', np.nan):>+.4f} {res['top10_hit24']:>7.1f}% {res['auc_hit24']:>.4f}   {res['brier_hit24']:>.4f}")


# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n\n{'=' * 120}")
print("SUMMARY")
print("=" * 120)

print(f"\n  Current metric (RYPTPA age-weighted):")
cur = comp_results['Current (RYPTPA only)']
print(f"    PRI-AVG: {cur['pri_avg']:+.4f}")
print(f"    Top-10% hit24: {cur['top10_hit24']:.1f}%")

print(f"\n  Best composite ({best_label}):")
best = comp_results[best_label]
print(f"    PRI-AVG: {best['pri_avg']:+.4f}  (Δ = {best['pri_avg'] - cur['pri_avg']:+.4f})")
print(f"    Top-10% hit24: {best['top10_hit24']:.1f}%  (Δ = {best['top10_hit24'] - cur['top10_hit24']:+.1f}%)")

# Are any composites significantly better?
any_better = False
for label, res in comp_results.items():
    if label == 'Current (RYPTPA only)':
        continue
    if res['pri_avg'] > cur['pri_avg'] + 0.005:
        any_better = True
        print(f"\n  ⚡ {label} beats current by {res['pri_avg'] - cur['pri_avg']:+.4f} PRI-AVG")

if not any_better:
    print(f"\n  No composite meaningfully beats the current single-metric RYPTPA.")
    print(f"  The additional PFF data does not add predictive signal beyond what RYPTPA already captures.")

print(f"\n{'=' * 120}")
print("ANALYSIS COMPLETE")
print("=" * 120)
