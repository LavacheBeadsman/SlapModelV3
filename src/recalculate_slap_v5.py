"""
SLAP Score V5 — Complete Recalculation + Backtest Analysis
==========================================================
WR V5: 70% DC / 20% Enhanced Breakout / 5% Teammate / 5% Early Declare
RB V5: 65% DC / 30% RYPTPA / 5% Speed Score (MNAR-aware imputation)

Outputs:
  1. Hit24 rate by SLAP tier (90+, 80-89, 70-79, 60-69, 50-59, <50)
  2. Top decile precision (hit24, hit12, first_3yr_ppg)
  3. V5 vs V4 vs DC-only comparison (PRI-AVG, AUC, Brier)
  4. 10 biggest WR risers/fallers (V5 - V4) with outcomes
  5. 10 biggest RB risers/fallers (V5 - V4) with outcomes
  6. Save output/slap_v5_database.csv
"""

import pandas as pd
import numpy as np
import warnings, os, time
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

# ============================================================================
# HELPERS
# ============================================================================

def dc_score(pick):
    """DC = 100 - 2.40 × (pick^0.62 - 1)"""
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

def wr_breakout_score(breakout_age, dominator_pct):
    """Continuous breakout scoring: age tier + dominator tiebreaker."""
    if pd.isna(breakout_age) or breakout_age is None:
        if dominator_pct is not None and pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base = age_tiers.get(int(breakout_age), 20)
    bonus = 0
    if dominator_pct is not None and pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
    return min(base + bonus, 99.9)

def wr_enhanced_breakout(breakout_age, dominator_pct, rush_yards):
    """Enhanced breakout = breakout_score + rush_bonus."""
    base = wr_breakout_score(breakout_age, dominator_pct)
    rush_bonus = 5 if pd.notna(rush_yards) and rush_yards >= 20 else 0
    return min(base + rush_bonus, 99.9)

def rb_production_score(rec_yards, team_pass_att, age):
    """RYPTPA with age weighting, scaled by 1.75."""
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return np.nan
    if pd.isna(age): age = 22
    season_age = age - 1  # age during final college season
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

def auc_roc(labels, scores):
    valid = pd.DataFrame({'y': labels, 's': scores}).dropna()
    if len(valid) < 10: return np.nan
    pos, neg = valid[valid['y'] == 1]['s'], valid[valid['y'] == 0]['s']
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return sum((neg < p).sum() + 0.5 * (neg == p).sum() for p in pos) / (len(pos) * len(neg))

def brier_score(labels, scores_0_100):
    valid = pd.DataFrame({'y': labels, 's': scores_0_100}).dropna()
    if len(valid) < 10: return np.nan
    return ((valid['s'] / 100 - valid['y']) ** 2).mean()

# ============================================================================
# V5 WEIGHTS
# ============================================================================
WR_V5 = {'dc': 0.70, 'breakout': 0.20, 'teammate': 0.05, 'early_declare': 0.05}
RB_V5 = {'dc': 0.65, 'production': 0.30, 'speed_score': 0.05}

# V4 WEIGHTS (for comparison)
WR_V4 = {'dc': 0.65, 'breakout': 0.20, 'ras': 0.15}
RB_V4 = {'dc': 0.50, 'production': 0.35, 'ras': 0.15}

print("=" * 120)
print("SLAP SCORE V5 — COMPLETE RECALCULATION")
print("=" * 120)
print(f"\n  WR V5: {int(WR_V5['dc']*100)}/{int(WR_V5['breakout']*100)}/{int(WR_V5['teammate']*100)}/{int(WR_V5['early_declare']*100)} (DC / Enhanced Breakout / Teammate / Early Declare)")
print(f"  RB V5: {int(RB_V5['dc']*100)}/{int(RB_V5['production']*100)}/{int(RB_V5['speed_score']*100)} (DC / RYPTPA / Speed Score)")

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\n{'='*120}")
print("LOADING DATA")
print("=" * 120)

# --- WR data ---
wr = pd.read_csv('data/wr_backtest_all_components.csv')
wr_tm = pd.read_csv('data/wr_teammate_scores.csv')
wr = wr.merge(wr_tm[['player_name', 'draft_year', 'total_teammate_dc']], on=['player_name', 'draft_year'], how='left')
print(f"  WR backtest: {len(wr)} players")

# --- WR outcomes ---
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
wr_out = outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
wr = wr.merge(wr_out, on=['player_name', 'draft_year', 'pick'], how='left')
print(f"  WR outcomes merged: {wr['first_3yr_ppg'].notna().sum()}/{len(wr)} have first_3yr_ppg")

# --- RB data ---
rb = pd.read_csv('data/rb_backtest_with_receiving.csv')
rb_out = outcomes[outcomes['position'] == 'RB'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
rb = rb.merge(rb_out, on=['player_name', 'draft_year', 'pick'], how='left')
print(f"  RB backtest: {len(rb)} players")
print(f"  RB outcomes merged: {rb['first_3yr_ppg'].notna().sum()}/{len(rb)} have first_3yr_ppg")

# ============================================================================
# CALCULATE WR COMPONENT SCORES
# ============================================================================
print(f"\n{'='*120}")
print("CALCULATING WR V5 SCORES")
print("=" * 120)

wr['s_dc'] = wr['pick'].apply(dc_score)

# Enhanced breakout (age tier + dominator + rush bonus)
wr['s_breakout'] = wr.apply(
    lambda r: wr_enhanced_breakout(r['breakout_age'], r['peak_dominator'], r['rush_yards']), axis=1)

# Teammate score (binary: total_teammate_dc > 150 → 100, else 0)
wr['s_teammate'] = wr['total_teammate_dc'].apply(lambda x: 100 if pd.notna(x) and x > 150 else 0)

# Early declare (binary: 100 or 0)
wr['s_early_declare'] = wr['early_declare'].apply(lambda x: 100 if x == 1 else 0)

# V5 SLAP
wr['slap_v5'] = (
    WR_V5['dc'] * wr['s_dc'] +
    WR_V5['breakout'] * wr['s_breakout'] +
    WR_V5['teammate'] * wr['s_teammate'] +
    WR_V5['early_declare'] * wr['s_early_declare']
)

# V4 SLAP (for comparison: 65/20/15 DC/Breakout/RAS)
wr['s_breakout_v4'] = wr.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)
WR_AVG_RAS = 68.9
wr['s_ras'] = wr['RAS'].apply(lambda x: x * 10 if pd.notna(x) else WR_AVG_RAS)
wr['slap_v4'] = (
    WR_V4['dc'] * wr['s_dc'] +
    WR_V4['breakout'] * wr['s_breakout_v4'] +
    WR_V4['ras'] * wr['s_ras']
)

# DC-only
wr['slap_dc'] = wr['s_dc']

wr['delta_v5_v4'] = wr['slap_v5'] - wr['slap_v4']
wr['delta_v5_dc'] = wr['slap_v5'] - wr['s_dc']

n_tm = (wr['s_teammate'] == 100).sum()
n_ed = (wr['s_early_declare'] == 100).sum()
print(f"  Teammate = 100: {n_tm}/{len(wr)} ({n_tm/len(wr)*100:.1f}%)")
print(f"  Early declare = 100: {n_ed}/{len(wr)} ({n_ed/len(wr)*100:.1f}%)")
print(f"  V5 SLAP range: {wr['slap_v5'].min():.1f} - {wr['slap_v5'].max():.1f}")

# ============================================================================
# CALCULATE RB COMPONENT SCORES + SPEED SCORE WITH MNAR IMPUTATION
# ============================================================================
print(f"\n{'='*120}")
print("CALCULATING RB V5 SCORES (with Speed Score MNAR imputation)")
print("=" * 120)

rb['s_dc'] = rb['pick'].apply(dc_score)
rb['s_production'] = rb.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)

# Impute missing production with position average
RB_AVG_PROD = rb['s_production'].mean()
rb['s_production_final'] = rb['s_production'].fillna(RB_AVG_PROD)

# --- Speed Score with weight recovery + MNAR imputation ---
rb['name_norm'] = rb['player_name'].apply(normalize_name)

# Load combine data with fixed 2025 matching
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine['name_norm'] = combine['player_name'].apply(normalize_name)

combine_lookup = {}
for pos_pref in ['RB', 'FB', 'WR', 'TE']:
    sub = combine[combine['pos'] == pos_pref]
    for _, row in sub.iterrows():
        dy = row.get('draft_year')
        if pd.isna(dy):
            dy = row.get('season')
        if pd.isna(dy): continue
        key = (row['name_norm'], int(dy))
        if key not in combine_lookup:
            combine_lookup[key] = {'weight': row['wt'], 'forty': row['forty']}

rb['weight'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb['forty'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb['weight'] = pd.to_numeric(rb['weight'], errors='coerce')
rb['forty'] = pd.to_numeric(rb['forty'], errors='coerce')

# CFBD API weight recovery for missing
missing_wt_mask = rb['weight'].isna()
if missing_wt_mask.sum() > 0:
    try:
        import requests
        CFBD_API_KEY = os.getenv("CFBD_API_KEY", "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE")
        CFBD_HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}
        cfbd_wts = {}
        for _, row in rb[missing_wt_mask].iterrows():
            name = row['player_name']
            try:
                url = f"https://api.collegefootballdata.com/player/search?searchTerm={name.replace(' ', '+')}&position=RB"
                resp = requests.get(url, headers=CFBD_HEADERS, timeout=10)
                if resp.status_code == 200:
                    for p in resp.json():
                        if normalize_name(p.get('name', '')) == row['name_norm'] and p.get('weight'):
                            cfbd_wts[name] = int(p['weight'])
                            break
                time.sleep(0.25)
            except:
                pass
        # Also search FB/WR for position converts
        for name in [n for n in rb[missing_wt_mask]['player_name'] if n not in cfbd_wts]:
            nn = normalize_name(name)
            for pos in ['FB', 'WR']:
                try:
                    url = f"https://api.collegefootballdata.com/player/search?searchTerm={name.replace(' ', '+')}&position={pos}"
                    resp = requests.get(url, headers=CFBD_HEADERS, timeout=10)
                    if resp.status_code == 200:
                        for p in resp.json():
                            if normalize_name(p.get('name', '')) == nn and p.get('weight'):
                                cfbd_wts[name] = int(p['weight'])
                                break
                    time.sleep(0.25)
                except:
                    pass
                if name in cfbd_wts: break
        # Apply
        for name, wt in cfbd_wts.items():
            rb.loc[rb['player_name'] == name, 'weight'] = wt
        print(f"  CFBD recovered weight for {len(cfbd_wts)} players")
    except ImportError:
        print("  WARNING: requests not available for CFBD API")

# 40-time estimation from weight × round buckets
known = rb[rb['weight'].notna() & rb['forty'].notna()].copy()
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
known['rb_bucket'] = known['round'].apply(rd_bucket)
lookup_40 = {}
for wb in ['<200', '200-209', '210-219', '220+']:
    for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
        sub = known[(known['wb'] == wb) & (known['rb_bucket'] == rdb)]
        if len(sub) > 0:
            lookup_40[(wb, rdb)] = sub['forty'].mean()
    # Fallback: weight-bucket average
    wt_sub = known[known['wb'] == wb]
    if len(wt_sub) > 0:
        for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
            if (wb, rdb) not in lookup_40:
                lookup_40[(wb, rdb)] = wt_sub['forty'].mean()

# Apply imputed 40s
rb['forty_final'] = rb['forty'].copy()
impute_mask = rb['weight'].notna() & rb['forty'].isna()
for idx in rb[impute_mask].index:
    wb = wt_bucket(rb.loc[idx, 'weight'])
    rdb = rd_bucket(rb.loc[idx, 'round'])
    est = lookup_40.get((wb, rdb))
    if est is not None:
        rb.loc[idx, 'forty_final'] = est

# Calculate speed scores
rb['raw_ss'] = rb.apply(lambda r: speed_score_fn(r['weight'], r['forty_final']), axis=1)

# MNAR-aware imputation for remaining missing
real_ss = rb['raw_ss'].dropna()
p60 = real_ss.quantile(0.60)
p40 = real_ss.quantile(0.40)
for idx in rb[rb['raw_ss'].isna()].index:
    rd = rb.loc[idx, 'round']
    rb.loc[idx, 'raw_ss'] = p60 if rd <= 2 else p40

# Normalize speed scores 0-100
rb['s_speed_score'] = normalize_0_100(rb['raw_ss'])

n_real_ss = rb['forty'].notna().sum()
n_imputed_40 = impute_mask.sum()
n_mnar = (rb['weight'].isna()).sum()
print(f"  Speed Score: {n_real_ss} real, {n_imputed_40} imputed-40, {n_mnar} MNAR-imputed")

# V5 SLAP
rb['slap_v5'] = (
    RB_V5['dc'] * rb['s_dc'] +
    RB_V5['production'] * rb['s_production_final'] +
    RB_V5['speed_score'] * rb['s_speed_score']
)

# V4 SLAP (for comparison: 50/35/15 DC/Production/RAS)
RB_AVG_RAS = 66.5
rb['s_ras'] = rb['RAS'].apply(lambda x: x * 10 if pd.notna(x) else RB_AVG_RAS)
rb['slap_v4'] = (
    RB_V4['dc'] * rb['s_dc'] +
    RB_V4['production'] * rb['s_production_final'] +
    RB_V4['ras'] * rb['s_ras']
)

# DC-only
rb['slap_dc'] = rb['s_dc']

rb['delta_v5_v4'] = rb['slap_v5'] - rb['slap_v4']
rb['delta_v5_dc'] = rb['slap_v5'] - rb['s_dc']

print(f"  V5 SLAP range: {rb['slap_v5'].min():.1f} - {rb['slap_v5'].max():.1f}")

# ============================================================================
# BACKTEST ANALYSIS
# ============================================================================

# Filter to players with outcomes
wr_eval = wr[wr['hit24'].notna()].copy()
rb_eval = rb[rb['hit24'].notna()].copy()
print(f"\n  WR eval sample: {len(wr_eval)} | RB eval sample: {len(rb_eval)}")

# ============================================================================
# 1. HIT24 RATE BY SLAP TIER
# ============================================================================
print(f"\n\n{'='*120}")
print("1. HIT24 RATE BY SLAP TIER")
print("=" * 120)

tiers = [(90, 100, '90+'), (80, 89.99, '80-89'), (70, 79.99, '70-79'),
         (60, 69.99, '60-69'), (50, 59.99, '50-59'), (0, 49.99, '<50')]

for pos, df, label in [('WR', wr_eval, 'WR V5'), ('RB', rb_eval, 'RB V5')]:
    print(f"\n  ── {label} ──")
    print(f"  {'Tier':<8s} {'n':>4s} {'Hit24':>6s} {'Hit24%':>8s} {'Hit12':>6s} {'Hit12%':>8s} {'Avg PPG':>9s}")
    print(f"  {'-'*55}")
    for lo, hi, name in tiers:
        sub = df[(df['slap_v5'] >= lo) & (df['slap_v5'] <= hi)]
        if len(sub) == 0:
            print(f"  {name:<8s} {0:>4d}      —        —      —        —         —")
            continue
        h24 = sub['hit24'].sum()
        h24_pct = sub['hit24'].mean() * 100
        h12 = sub['hit12'].sum() if 'hit12' in sub.columns else 0
        h12_pct = sub['hit12'].mean() * 100 if 'hit12' in sub.columns else 0
        ppg = sub['first_3yr_ppg'].mean() if sub['first_3yr_ppg'].notna().any() else np.nan
        ppg_str = f"{ppg:.1f}" if pd.notna(ppg) else "—"
        print(f"  {name:<8s} {len(sub):>4d} {h24:>5.0f} {h24_pct:>7.1f}% {h12:>5.0f} {h12_pct:>7.1f}% {ppg_str:>9s}")

# ============================================================================
# 2. TOP DECILE PRECISION
# ============================================================================
print(f"\n\n{'='*120}")
print("2. TOP DECILE PRECISION")
print("=" * 120)

for pos, df, label in [('WR', wr_eval, 'WR V5'), ('RB', rb_eval, 'RB V5')]:
    n10 = max(1, len(df) // 10)
    top = df.nlargest(n10, 'slap_v5')
    h24_rate = top['hit24'].mean() * 100
    h12_rate = top['hit12'].mean() * 100 if 'hit12' in top.columns else np.nan
    ppg = top['first_3yr_ppg'].mean() if top['first_3yr_ppg'].notna().any() else np.nan
    print(f"\n  ── {label} (top {n10} of {len(df)}) ──")
    print(f"  Hit24 rate:   {h24_rate:.1f}% ({int(top['hit24'].sum())}/{n10})")
    print(f"  Hit12 rate:   {h12_rate:.1f}% ({int(top['hit12'].sum())}/{n10})")
    print(f"  Avg 3yr PPG:  {ppg:.2f}" if pd.notna(ppg) else "  Avg 3yr PPG:  —")
    print(f"\n  Top decile roster:")
    print(f"  {'Player':<25s} {'Year':>4s} {'Pick':>4s} {'SLAP':>6s} {'hit24':>5s} {'hit12':>5s} {'PPG':>7s}")
    print(f"  {'-'*60}")
    for _, r in top.sort_values('slap_v5', ascending=False).iterrows():
        ppg_str = f"{r['first_3yr_ppg']:.1f}" if pd.notna(r['first_3yr_ppg']) else "—"
        h12_str = str(int(r['hit12'])) if pd.notna(r.get('hit12')) else "?"
        print(f"  {r['player_name']:<25s} {int(r['draft_year']):>4d} {int(r['pick']):>4d} {r['slap_v5']:>6.1f} {int(r['hit24']):>5d} {h12_str:>5s} {ppg_str:>7s}")

# ============================================================================
# 3. V5 vs V4 vs DC-ONLY COMPARISON
# ============================================================================
print(f"\n\n{'='*120}")
print("3. V5 vs V4 vs DC-ONLY COMPARISON")
print("=" * 120)

outcome_wts = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}

def compute_metrics(df, slap_col, label):
    """Compute all evaluation metrics for a given SLAP column."""
    s = df[slap_col]
    res = {'label': label}
    # Correlations
    for oc, short in [('first_3yr_ppg', '3yr'), ('hit24', 'h24'), ('hit12', 'h12'), ('career_ppg', 'car')]:
        valid = pd.DataFrame({'s': s, 'y': df[oc]}).dropna()
        res[f'r({short})'] = valid['s'].corr(valid['y']) if len(valid) > 5 else np.nan
    # PRI-AVG
    pri = 0
    oc_map = {'first_3yr_ppg': '3yr', 'hit24': 'h24', 'hit12': 'h12', 'career_ppg': 'car'}
    for oc, w in outcome_wts.items():
        r_key = f'r({oc_map[oc]})'
        if r_key in res and pd.notna(res[r_key]):
            pri += res[r_key] * w
    res['PRI-AVG'] = pri
    # Top decile
    n10 = max(1, len(df) // 10)
    top = df.nlargest(n10, slap_col)
    res['T10%h24'] = top['hit24'].mean() * 100
    res['T10%h12'] = top['hit12'].mean() * 100 if 'hit12' in top.columns else np.nan
    ppg_vals = top['first_3yr_ppg'].dropna()
    res['T10%PPG'] = ppg_vals.mean() if len(ppg_vals) > 0 else np.nan
    # AUC
    res['AUC h24'] = auc_roc(df['hit24'], s)
    res['AUC h12'] = auc_roc(df['hit12'], s) if 'hit12' in df.columns else np.nan
    # Brier
    res['Brier24'] = brier_score(df['hit24'], s)
    res['Brier12'] = brier_score(df['hit12'], s) if 'hit12' in df.columns else np.nan
    return res

metric_order = ['PRI-AVG', 'r(3yr)', 'r(h24)', 'r(h12)', 'r(car)',
                'T10%h24', 'T10%h12', 'T10%PPG', 'AUC h24', 'AUC h12', 'Brier24', 'Brier12']

for pos, df, label in [('WR', wr_eval, 'WR'), ('RB', rb_eval, 'RB')]:
    configs = [
        compute_metrics(df, 'slap_v5', f'{label} V5'),
        compute_metrics(df, 'slap_v4', f'{label} V4'),
        compute_metrics(df, 'slap_dc', f'{label} DC-only'),
    ]
    print(f"\n  ── {label} ──")
    header = f"  {'Config':<18s}"
    for m in metric_order:
        header += f" {m:>8s}"
    print(header)
    print(f"  {'-'*145}")
    for cfg in configs:
        line = f"  {cfg['label']:<18s}"
        for m in metric_order:
            v = cfg.get(m, np.nan)
            if 'T10%' in m and 'PPG' not in m:
                line += f" {v:>7.1f}%"
            elif 'T10%PPG' in m:
                line += f" {v:>8.2f}" if pd.notna(v) else f" {'—':>8s}"
            elif 'Brier' in m:
                line += f" {v:>8.4f}"
            elif 'AUC' in m:
                line += f" {v:>8.4f}"
            else:
                line += f" {v:>+8.4f}"
        print(line)

    # Delta summary
    v5 = configs[0]
    v4 = configs[1]
    dc = configs[2]
    print(f"\n  V5 vs V4 deltas:")
    improved, hurt = 0, 0
    for m in metric_order:
        v5_v, v4_v = v5.get(m, np.nan), v4.get(m, np.nan)
        if pd.isna(v5_v) or pd.isna(v4_v): continue
        delta = v5_v - v4_v
        if 'Brier' in m:
            if delta < -0.0001: improved += 1
            elif delta > 0.0001: hurt += 1
        else:
            if delta > 0.0001: improved += 1
            elif delta < -0.0001: hurt += 1
    print(f"    Improved: {improved} | Hurt: {hurt} | Net: {improved - hurt:+d}")

# ============================================================================
# 4. WR RISERS AND FALLERS (V5 - V4)
# ============================================================================
print(f"\n\n{'='*120}")
print("4. WR RISERS AND FALLERS (V5 vs V4)")
print("=" * 120)

wr_with_delta = wr_eval.copy()

print(f"\n  ── TOP 10 WR RISERS (V5 likes more than V4) ──")
risers = wr_with_delta.nlargest(10, 'delta_v5_v4')
print(f"  {'Player':<25s} {'Year':>4s} {'Pick':>4s} {'V5':>6s} {'V4':>6s} {'Delta':>7s} {'hit24':>5s} {'PPG':>7s}")
print(f"  {'-'*75}")
for _, r in risers.iterrows():
    ppg = f"{r['first_3yr_ppg']:.1f}" if pd.notna(r['first_3yr_ppg']) else "—"
    print(f"  {r['player_name']:<25s} {int(r['draft_year']):>4d} {int(r['pick']):>4d} {r['slap_v5']:>6.1f} {r['slap_v4']:>6.1f} {r['delta_v5_v4']:>+7.1f} {int(r['hit24']):>5d} {ppg:>7s}")

riser_hit = risers['hit24'].mean() * 100
riser_ppg = risers['first_3yr_ppg'].mean()
print(f"\n  Risers: {riser_hit:.0f}% hit24 rate, avg {riser_ppg:.1f} PPG" if pd.notna(riser_ppg) else f"\n  Risers: {riser_hit:.0f}% hit24 rate")

print(f"\n  ── TOP 10 WR FALLERS (V5 likes less than V4) ──")
fallers = wr_with_delta.nsmallest(10, 'delta_v5_v4')
print(f"  {'Player':<25s} {'Year':>4s} {'Pick':>4s} {'V5':>6s} {'V4':>6s} {'Delta':>7s} {'hit24':>5s} {'PPG':>7s}")
print(f"  {'-'*75}")
for _, r in fallers.iterrows():
    ppg = f"{r['first_3yr_ppg']:.1f}" if pd.notna(r['first_3yr_ppg']) else "—"
    print(f"  {r['player_name']:<25s} {int(r['draft_year']):>4d} {int(r['pick']):>4d} {r['slap_v5']:>6.1f} {r['slap_v4']:>6.1f} {r['delta_v5_v4']:>+7.1f} {int(r['hit24']):>5d} {ppg:>7s}")

faller_hit = fallers['hit24'].mean() * 100
faller_ppg = fallers['first_3yr_ppg'].mean()
print(f"\n  Fallers: {faller_hit:.0f}% hit24 rate, avg {faller_ppg:.1f} PPG" if pd.notna(faller_ppg) else f"\n  Fallers: {faller_hit:.0f}% hit24 rate")

# ============================================================================
# 5. RB RISERS AND FALLERS (V5 - V4)
# ============================================================================
print(f"\n\n{'='*120}")
print("5. RB RISERS AND FALLERS (V5 vs V4)")
print("=" * 120)

rb_with_delta = rb_eval.copy()

print(f"\n  ── TOP 10 RB RISERS (V5 likes more than V4) ──")
risers = rb_with_delta.nlargest(10, 'delta_v5_v4')
print(f"  {'Player':<25s} {'Year':>4s} {'Pick':>4s} {'V5':>6s} {'V4':>6s} {'Delta':>7s} {'hit24':>5s} {'PPG':>7s}")
print(f"  {'-'*75}")
for _, r in risers.iterrows():
    ppg = f"{r['first_3yr_ppg']:.1f}" if pd.notna(r['first_3yr_ppg']) else "—"
    print(f"  {r['player_name']:<25s} {int(r['draft_year']):>4d} {int(r['pick']):>4d} {r['slap_v5']:>6.1f} {r['slap_v4']:>6.1f} {r['delta_v5_v4']:>+7.1f} {int(r['hit24']):>5d} {ppg:>7s}")

riser_hit = risers['hit24'].mean() * 100
riser_ppg = risers['first_3yr_ppg'].mean()
print(f"\n  Risers: {riser_hit:.0f}% hit24 rate, avg {riser_ppg:.1f} PPG" if pd.notna(riser_ppg) else f"\n  Risers: {riser_hit:.0f}% hit24 rate")

print(f"\n  ── TOP 10 RB FALLERS (V5 likes less than V4) ──")
fallers = rb_with_delta.nsmallest(10, 'delta_v5_v4')
print(f"  {'Player':<25s} {'Year':>4s} {'Pick':>4s} {'V5':>6s} {'V4':>6s} {'Delta':>7s} {'hit24':>5s} {'PPG':>7s}")
print(f"  {'-'*75}")
for _, r in fallers.iterrows():
    ppg = f"{r['first_3yr_ppg']:.1f}" if pd.notna(r['first_3yr_ppg']) else "—"
    print(f"  {r['player_name']:<25s} {int(r['draft_year']):>4d} {int(r['pick']):>4d} {r['slap_v5']:>6.1f} {r['slap_v4']:>6.1f} {r['delta_v5_v4']:>+7.1f} {int(r['hit24']):>5d} {ppg:>7s}")

faller_hit = fallers['hit24'].mean() * 100
faller_ppg = fallers['first_3yr_ppg'].mean()
print(f"\n  Fallers: {faller_hit:.0f}% hit24 rate, avg {faller_ppg:.1f} PPG" if pd.notna(faller_ppg) else f"\n  Fallers: {faller_hit:.0f}% hit24 rate")

# ============================================================================
# 6. SAVE V5 DATABASE
# ============================================================================
print(f"\n\n{'='*120}")
print("6. SAVING V5 DATABASE")
print("=" * 120)

# Build WR output
wr_db = pd.DataFrame({
    'player_name': wr['player_name'],
    'position': 'WR',
    'school': wr['college'],
    'draft_year': wr['draft_year'],
    'pick': wr['pick'],
    'round': wr['round'],
    'slap_v5': wr['slap_v5'].round(1),
    'slap_v4': wr['slap_v4'].round(1),
    'dc_score': wr['s_dc'].round(1),
    'enhanced_breakout': wr['s_breakout'].round(1),
    'teammate_score': wr['s_teammate'].astype(int),
    'early_declare_score': wr['s_early_declare'].astype(int),
    'delta_v5_dc': wr['delta_v5_dc'].round(1),
    'delta_v5_v4': wr['delta_v5_v4'].round(1),
    'breakout_age': wr['breakout_age'],
    'peak_dominator': wr['peak_dominator'],
    'rush_yards': wr['rush_yards'],
    'nfl_hit24': wr['hit24'],
    'nfl_hit12': wr['hit12'],
    'nfl_first_3yr_ppg': wr.get('first_3yr_ppg'),
    'nfl_career_ppg': wr.get('career_ppg'),
    'nfl_best_ppr': wr['best_ppr'],
    'data_type': 'backtest',
})

# Build RB output
rb_db = pd.DataFrame({
    'player_name': rb['player_name'],
    'position': 'RB',
    'school': rb['college'],
    'draft_year': rb['draft_year'],
    'pick': rb['pick'],
    'round': rb['round'],
    'slap_v5': rb['slap_v5'].round(1),
    'slap_v4': rb['slap_v4'].round(1),
    'dc_score': rb['s_dc'].round(1),
    'production_score': rb['s_production_final'].round(1),
    'speed_score_norm': rb['s_speed_score'].round(1),
    'delta_v5_dc': rb['delta_v5_dc'].round(1),
    'delta_v5_v4': rb['delta_v5_v4'].round(1),
    'rec_yards': rb['rec_yards'],
    'team_pass_att': rb['team_pass_att'],
    'weight': rb['weight'],
    'forty': rb['forty'],
    'nfl_hit24': rb['hit24'],
    'nfl_hit12': rb['hit12'],
    'nfl_first_3yr_ppg': rb.get('first_3yr_ppg'),
    'nfl_career_ppg': rb.get('career_ppg'),
    'nfl_best_ppr': rb['best_ppr'],
    'nfl_best_ppg': rb['best_ppg'],
    'data_type': 'backtest',
})

# Combine and save
v5_db = pd.concat([wr_db, rb_db], ignore_index=True)
v5_db = v5_db.sort_values(['position', 'draft_year', 'slap_v5'], ascending=[True, True, False])
v5_db.to_csv('output/slap_v5_database.csv', index=False)
print(f"  Saved: output/slap_v5_database.csv ({len(v5_db)} total: {len(wr_db)} WR + {len(rb_db)} RB)")

print(f"\n{'='*120}")
print("V5 RECALCULATION COMPLETE")
print("=" * 120)
