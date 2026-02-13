"""
RB Speed Score — Weight recovery + 40-time imputation from weight×round
======================================================================
Step 1: Get weight for every RB (combine → CFBD API fallback)
Step 2: Estimate 40 time from weight×round buckets for missing players
Step 3: Show imputed values for gut-check
Step 4: Retest 60/30/10 DC/RYPTPA/SpeedScore vs baselines
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings, os, time, json
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

# ============================================================================
# HELPERS (same as speed_score_test.py)
# ============================================================================

def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

def normalize_0_100(series):
    mn, mx = series.min(), series.max()
    if mx == mn: return pd.Series(50, index=series.index)
    return (series - mn) / (mx - mn) * 100

def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()

def speed_score_fn(weight, forty):
    if pd.isna(weight) or pd.isna(forty) or forty <= 0 or weight <= 0:
        return np.nan
    return (weight * 200) / (forty ** 4)

def rb_production_score(row):
    if pd.isna(row.get('rec_yards')) or pd.isna(row.get('team_pass_att')) or row.get('team_pass_att', 0) == 0:
        return np.nan
    age = row.get('age', 22)
    if pd.isna(age): age = 22
    age_weights = {19: 1.15, 20: 1.10, 21: 1.05, 22: 1.00, 23: 0.95}
    age_w = age_weights.get(int(age), 0.90 if age >= 24 else 1.15)
    raw = (row['rec_yards'] / row['team_pass_att']) * age_w * 100
    return min(99.9, raw / 1.75)

def auc_roc(labels, scores):
    valid = pd.DataFrame({'y': labels, 's': scores}).dropna()
    if len(valid) < 10: return np.nan
    pos = valid[valid['y'] == 1]['s']
    neg = valid[valid['y'] == 0]['s']
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return sum((neg < p).sum() + 0.5 * (neg == p).sum() for p in pos) / (len(pos) * len(neg))

def brier_score(labels, scores_0_100):
    valid = pd.DataFrame({'y': labels, 's': scores_0_100}).dropna()
    if len(valid) < 10: return np.nan
    return ((valid['s'] / 100 - valid['y']) ** 2).mean()

outcome_cols = ['first_3yr_ppg', 'hit24', 'hit12', 'career_ppg']
outcome_wts  = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}

# ============================================================================
# DATA LOADING WITH FIXED NAME MATCHING
# ============================================================================

rb = pd.read_csv('data/rb_backtest_with_receiving.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
out_rb = outcomes[outcomes['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()
rb = rb.merge(out_rb, on=['player_name', 'draft_year', 'pick'], how='left')
rb['s_dc'] = rb['pick'].apply(dc_score)
rb['s_rec_prod'] = rb.apply(rb_production_score, axis=1)
rb['name_norm'] = rb['player_name'].apply(normalize_name)

# ============================================================================
# STEP 1: GET WEIGHT FOR EVERY RB
# ============================================================================

print("=" * 120)
print("STEP 1: WEIGHT RECOVERY FROM ALL SOURCES")
print("=" * 120)

# --- Source 1: combine.parquet (primary) ---
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine['name_norm'] = combine['player_name'].apply(normalize_name)

combine_lookup = {}
for pos_pref in ['RB', 'FB', 'WR', 'TE']:
    sub = combine[combine['pos'] == pos_pref]
    for _, row in sub.iterrows():
        # Use draft_year if available, fall back to season (2025 entries have NaN draft_year)
        dy = row.get('draft_year')
        if pd.isna(dy):
            dy = row.get('season')
        if pd.isna(dy): continue
        key = (row['name_norm'], int(dy))
        if key not in combine_lookup:
            combine_lookup[key] = {
                'weight': row['wt'], 'forty': row['forty'],
                'ht': row.get('ht', np.nan),
                'pos': row['pos'], 'orig_name': row['player_name']
            }

rb['weight_combine'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb['forty_combine'] = rb.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb['weight_combine'] = pd.to_numeric(rb['weight_combine'], errors='coerce')
rb['forty_combine'] = pd.to_numeric(rb['forty_combine'], errors='coerce')

n_wt_combine = rb['weight_combine'].notna().sum()
n_40_combine = rb['forty_combine'].notna().sum()
print(f"\n  Source 1 — combine.parquet:")
print(f"    Weight: {n_wt_combine}/223 ({n_wt_combine/223*100:.0f}%)")
print(f"    40 time: {n_40_combine}/223 ({n_40_combine/223*100:.0f}%)")

# --- Source 2: draft_picks.parquet (check for weight) ---
drafts = pd.read_parquet('data/nflverse/draft_picks.parquet')
has_wt_col = 'weight' in drafts.columns or 'wt' in drafts.columns
print(f"\n  Source 2 — draft_picks.parquet:")
if has_wt_col:
    print(f"    Has weight column: YES")
else:
    print(f"    Has weight column: NO — skipping")

# --- Source 3: PFF files (check for weight) ---
pff_rush = pd.read_csv('data/rushing_summary.csv', nrows=3)
pff_rec = pd.read_csv('data/receiving_summary (2).csv', nrows=3)
wt_cols_rush = [c for c in pff_rush.columns if 'weight' in c.lower() or 'wt' in c.lower() or 'size' in c.lower()]
wt_cols_rec = [c for c in pff_rec.columns if 'weight' in c.lower() or 'wt' in c.lower() or 'size' in c.lower()]
print(f"\n  Source 3 — PFF files:")
print(f"    rushing_summary weight cols: {wt_cols_rush if wt_cols_rush else 'NONE'}")
print(f"    receiving_summary weight cols: {wt_cols_rec if wt_cols_rec else 'NONE'}")

# --- Source 4: CFBD API roster endpoint ---
# Try to get weight for players missing from combine
missing_wt = rb[rb['weight_combine'].isna()][['player_name', 'draft_year', 'college', 'name_norm']].copy()

print(f"\n  Source 4 — CFBD API roster endpoint:")
print(f"    Attempting to recover weight for {len(missing_wt)} players...")

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE")
CFBD_HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}

cfbd_weights = {}
cfbd_errors = []
try:
    import requests
    # Use player search endpoint (returns weight unlike roster endpoint)
    for _, row in missing_wt.iterrows():
        name = row['player_name']
        try:
            url = f"https://api.collegefootballdata.com/player/search?searchTerm={name.replace(' ', '+')}&position=RB"
            resp = requests.get(url, headers=CFBD_HEADERS, timeout=10)
            if resp.status_code == 200:
                results = resp.json()
                for p in results:
                    p_norm = normalize_name(p.get('name', ''))
                    if p_norm == row['name_norm'] and p.get('weight'):
                        cfbd_weights[name] = int(p['weight'])
                        break
            elif resp.status_code == 429:
                time.sleep(2)
            time.sleep(0.25)  # Rate limiting
        except Exception as e:
            cfbd_errors.append(f"{name}: {str(e)[:50]}")
    # Also search FB/WR positions for position converts
    still_missing = [n for n in missing_wt['player_name'] if n not in cfbd_weights]
    for name in still_missing:
        name_norm = normalize_name(name)
        for pos in ['FB', 'WR']:
            try:
                url = f"https://api.collegefootballdata.com/player/search?searchTerm={name.replace(' ', '+')}&position={pos}"
                resp = requests.get(url, headers=CFBD_HEADERS, timeout=10)
                if resp.status_code == 200:
                    for p in resp.json():
                        if normalize_name(p.get('name', '')) == name_norm and p.get('weight'):
                            cfbd_weights[name] = int(p['weight'])
                            break
                time.sleep(0.25)
            except Exception as e:
                cfbd_errors.append(f"{name} ({pos}): {str(e)[:50]}")
            if name in cfbd_weights:
                break
except ImportError:
    print("    WARNING: requests module not available, skipping CFBD API")
    cfbd_errors.append("requests not available")

# Apply CFBD weights
rb['weight_cfbd'] = rb['player_name'].map(cfbd_weights)
rb['weight_cfbd'] = pd.to_numeric(rb['weight_cfbd'], errors='coerce')

n_cfbd = rb['weight_cfbd'].notna().sum()
print(f"    Recovered from CFBD: {n_cfbd} players")
if cfbd_weights:
    for name, wt in sorted(cfbd_weights.items()):
        print(f"      {name}: {wt} lbs")
if cfbd_errors:
    print(f"    API errors: {len(cfbd_errors)}")

# --- Merge: combine weight preferred, CFBD as fallback ---
rb['weight'] = rb['weight_combine'].fillna(rb['weight_cfbd'])
rb['forty'] = rb['forty_combine']  # Only combine has 40 times
rb['weight_source'] = 'missing'
rb.loc[rb['weight_combine'].notna(), 'weight_source'] = 'combine'
rb.loc[(rb['weight_combine'].isna()) & (rb['weight_cfbd'].notna()), 'weight_source'] = 'cfbd'

# --- Summary ---
n_wt_total = rb['weight'].notna().sum()
n_40_total = rb['forty'].notna().sum()
n_both = rb[rb['weight'].notna() & rb['forty'].notna()].shape[0]
n_wt_only = rb[rb['weight'].notna() & rb['forty'].isna()].shape[0]

print(f"\n  ── COVERAGE SUMMARY ──")
print(f"  Total with weight:       {n_wt_total}/223 ({n_wt_total/223*100:.0f}%)")
print(f"  Total with 40 time:      {n_40_total}/223 ({n_40_total/223*100:.0f}%)")
print(f"  Both (real speed score):  {n_both}/223 ({n_both/223*100:.0f}%)")
print(f"  Weight only (need 40):    {n_wt_only} players → will impute 40 time")

# Show who is STILL missing weight after all sources
still_missing = rb[rb['weight'].isna()].sort_values('pick')
if len(still_missing) > 0:
    print(f"\n  ── STILL MISSING WEIGHT ({len(still_missing)} players) ──")
    print(f"  {'Player':<25s} {'Year':>4s} {'Pick':>4s} {'Rd':>3s} {'hit24':>5s} {'3yr PPG':>8s}")
    print(f"  {'-'*55}")
    for _, row in still_missing.iterrows():
        ppg = "%.1f" % row['first_3yr_ppg'] if pd.notna(row['first_3yr_ppg']) else "?"
        h24 = "%d" % int(row['hit24']) if pd.notna(row['hit24']) else "?"
        print(f"  {row['player_name']:<25s} {int(row['draft_year']):>4d} {int(row['pick']):>4d} {int(row['round']):>3d} {h24:>5s} {ppg:>8s}")
else:
    print(f"\n  All 223 RBs have weight data!")

# ============================================================================
# STEP 2: ESTIMATE 40 TIME FROM WEIGHT × ROUND BUCKETS
# ============================================================================

print(f"\n\n{'=' * 120}")
print("STEP 2: 40-TIME ESTIMATION FROM WEIGHT × ROUND BUCKETS")
print("=" * 120)

# Use only the 151 RBs who have BOTH real weight and real 40 time
known = rb[rb['weight'].notna() & rb['forty'].notna()].copy()
print(f"\n  Training data: {len(known)} RBs with both real weight and real 40 time")

# Create buckets
def weight_bucket(wt):
    if pd.isna(wt): return None
    if wt < 200: return '<200'
    elif wt < 210: return '200-209'
    elif wt < 220: return '210-219'
    else: return '220+'

def round_bucket(rd):
    if rd <= 1: return 'Rd 1'
    elif rd <= 2: return 'Rd 2'
    elif rd <= 4: return 'Rd 3-4'
    else: return 'Rd 5+'

known['wt_bucket'] = known['weight'].apply(weight_bucket)
known['rd_bucket'] = known['round'].apply(round_bucket)

# Build lookup table
print(f"\n  ── WEIGHT × ROUND 40-TIME LOOKUP TABLE ──")
print(f"  {'Weight':<12s} {'Round':<8s} {'n':>4s} {'Avg 40':>8s} {'Min':>6s} {'Max':>6s} {'StdDev':>7s}")
print(f"  {'-'*55}")

lookup_table = {}
wt_order = ['<200', '200-209', '210-219', '220+']
rd_order = ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']

for wb in wt_order:
    for rb_bucket in rd_order:
        sub = known[(known['wt_bucket'] == wb) & (known['rd_bucket'] == rb_bucket)]
        if len(sub) >= 2:
            avg40 = sub['forty'].mean()
            lookup_table[(wb, rb_bucket)] = avg40
            print(f"  {wb:<12s} {rb_bucket:<8s} {len(sub):>4d} {avg40:>8.2f} {sub['forty'].min():>6.2f} {sub['forty'].max():>6.2f} {sub['forty'].std():>7.3f}")
        elif len(sub) == 1:
            avg40 = sub['forty'].values[0]
            lookup_table[(wb, rb_bucket)] = avg40
            print(f"  {wb:<12s} {rb_bucket:<8s} {len(sub):>4d} {avg40:>8.2f} {avg40:>6.2f} {avg40:>6.2f}   (n=1)")
        else:
            print(f"  {wb:<12s} {rb_bucket:<8s} {0:>4d}     ──     ──     ──")

# Fill empty cells with weight-bucket average (fallback)
for wb in wt_order:
    wt_sub = known[known['wt_bucket'] == wb]
    if len(wt_sub) > 0:
        wt_avg = wt_sub['forty'].mean()
        for rb_bucket in rd_order:
            if (wb, rb_bucket) not in lookup_table:
                lookup_table[(wb, rb_bucket)] = wt_avg
                print(f"  {wb:<12s} {rb_bucket:<8s}    →  {wt_avg:>8.2f} (fallback: weight-bucket avg)")

# --- Apply imputation to players with weight but no 40 ---
rb['wt_bucket'] = rb['weight'].apply(weight_bucket)
rb['rd_bucket'] = rb['round'].apply(round_bucket)

rb['forty_imputed'] = rb['forty'].copy()
rb['forty_source'] = 'missing'
rb.loc[rb['forty'].notna(), 'forty_source'] = 'observed'

imputed_mask = rb['weight'].notna() & rb['forty'].isna()
for idx in rb[imputed_mask].index:
    wb = rb.loc[idx, 'wt_bucket']
    rdb = rb.loc[idx, 'rd_bucket']
    est40 = lookup_table.get((wb, rdb))
    if est40 is not None:
        rb.loc[idx, 'forty_imputed'] = est40
        rb.loc[idx, 'forty_source'] = f'imputed ({wb}, {rdb})'

# Calculate speed score with imputed 40s
rb['speed_score_imputed'] = rb.apply(
    lambda r: speed_score_fn(r['weight'], r['forty_imputed']), axis=1)

# Also keep the real-only speed score
rb['speed_score_real'] = rb.apply(
    lambda r: speed_score_fn(r['weight'], r['forty']), axis=1)

n_ss_real = rb['speed_score_real'].notna().sum()
n_ss_imputed = rb['speed_score_imputed'].notna().sum()
n_newly_imputed = n_ss_imputed - n_ss_real

print(f"\n  ── IMPUTATION RESULTS ──")
print(f"  Speed scores (real only):      {n_ss_real}/223")
print(f"  Speed scores (with imputed):   {n_ss_imputed}/223")
print(f"  Newly imputed:                 {n_newly_imputed} players")
print(f"  Still missing (no weight):     {223 - n_ss_imputed}")

# ============================================================================
# STEP 3: SHOW IMPUTED VALUES FOR GUT-CHECK
# ============================================================================

print(f"\n\n{'=' * 120}")
print("STEP 3: IMPUTED VALUES — GUT-CHECK")
print("=" * 120)

imputed_players = rb[rb['forty'].isna() & rb['speed_score_imputed'].notna()].sort_values('pick').copy()
if len(imputed_players) > 0:
    print(f"\n  {'Player':<25s} {'Year':>4s} {'Pick':>4s} {'Rd':>3s} {'Weight':>6s} {'Est 40':>7s} {'SpScore':>8s} {'Bucket':<22s} {'hit24':>5s} {'PPG':>7s}")
    print(f"  {'-'*100}")
    for _, row in imputed_players.iterrows():
        ppg = "%.1f" % row['first_3yr_ppg'] if pd.notna(row['first_3yr_ppg']) else "?"
        h24 = "%d" % int(row['hit24']) if pd.notna(row['hit24']) else "?"
        bucket = row['forty_source'].replace('imputed ', '') if 'imputed' in str(row['forty_source']) else '?'
        print(f"  {row['player_name']:<25s} {int(row['draft_year']):>4d} {int(row['pick']):>4d} {int(row['round']):>3d} {row['weight']:>6.0f} {row['forty_imputed']:>7.2f} {row['speed_score_imputed']:>8.1f} {bucket:<22s} {h24:>5s} {ppg:>7s}")
else:
    print("\n  No players were imputed (all have real 40 times or no weight)")

# Compare imputed vs real speed score distributions
print(f"\n  ── DISTRIBUTION CHECK ──")
real_ss = rb[rb['forty_source'] == 'observed']['speed_score_imputed']
imp_ss = rb[rb['forty_source'].str.contains('imputed', na=False)]['speed_score_imputed']
print(f"  Real speed scores:    n={len(real_ss)}, mean={real_ss.mean():.1f}, std={real_ss.std():.1f}, median={real_ss.median():.1f}")
if len(imp_ss) > 0:
    print(f"  Imputed speed scores: n={len(imp_ss)}, mean={imp_ss.mean():.1f}, std={imp_ss.std():.1f}, median={imp_ss.median():.1f}")

# ============================================================================
# STEP 4: MODEL TESTING
# ============================================================================

print(f"\n\n{'=' * 120}")
print("STEP 4: MODEL TESTING — 60/30/10 WITH IMPROVED COVERAGE")
print("=" * 120)

rb_eval = rb[rb['hit24'].notna()].copy()

# Normalize speed scores
# Version A: normalize using imputed scores (full coverage)
ss_imputed = rb_eval['speed_score_imputed'].copy()
ss_real = rb_eval['speed_score_real'].copy()

# For players still missing speed score entirely (no weight at all),
# use position average of imputed scores
ss_avg_imputed = ss_imputed.mean()
ss_imputed_filled = ss_imputed.fillna(ss_avg_imputed)

# Normalize to 0-100
def norm_ss(series):
    mn, mx = series.dropna().min(), series.dropna().max()
    if mx == mn: return pd.Series(50, index=series.index)
    return ((series - mn) / (mx - mn) * 100).clip(0, 100)

rb_eval['s_ss_imputed'] = norm_ss(ss_imputed_filled)

# For MNAR comparison: use the old MNAR-aware imputation
# Rd 1-2 missing → 60th percentile, Rd 3+ → 40th percentile
ss_real_scores = rb_eval['speed_score_real'].copy()
p60 = ss_real_scores.quantile(0.60)
p40 = ss_real_scores.quantile(0.40)
ss_mnar = ss_real_scores.copy()
for idx in rb_eval[ss_real_scores.isna()].index:
    rd = rb_eval.loc[idx, 'round']
    if rd <= 2:
        ss_mnar.loc[idx] = p60
    else:
        ss_mnar.loc[idx] = p40
rb_eval['s_ss_mnar'] = norm_ss(ss_mnar)

# For round-avg comparison
rd_avgs = {}
for rd in range(1, 8):
    sub = rb_eval[rb_eval['round'] == rd]
    rd_avgs[rd] = sub['speed_score_real'].mean() if sub['speed_score_real'].notna().any() else ss_avg_imputed
ss_round_avg = ss_real_scores.copy()
for idx in rb_eval[ss_real_scores.isna()].index:
    rd = rb_eval.loc[idx, 'round']
    ss_round_avg.loc[idx] = rd_avgs.get(rd, ss_avg_imputed)
rb_eval['s_ss_round_avg'] = norm_ss(ss_round_avg)

# --- Define configs ---
configs = {
    'A: 65/35/0 (baseline)':    (0.65, 0.35, 0.00),
    'B: 60/30/10 (imputed 40)': (0.60, 0.30, 0.10),
    'C: 62/33/5 (imputed 40)':  (0.62, 0.33, 0.05),
    'D: 60/30/10 (MNAR)':       (0.60, 0.30, 0.10),
    'E: 62/33/5 (MNAR)':        (0.62, 0.33, 0.05),
    'F: 60/30/10 (round-avg)':  (0.60, 0.30, 0.10),
    'G: 62/33/5 (round-avg)':   (0.62, 0.33, 0.05),
}

# Map config to speed score column
ss_col_map = {
    'A: 65/35/0 (baseline)': 's_ss_imputed',  # doesn't matter, weight=0
    'B: 60/30/10 (imputed 40)': 's_ss_imputed',
    'C: 62/33/5 (imputed 40)':  's_ss_imputed',
    'D: 60/30/10 (MNAR)':       's_ss_mnar',
    'E: 62/33/5 (MNAR)':        's_ss_mnar',
    'F: 60/30/10 (round-avg)':  's_ss_round_avg',
    'G: 62/33/5 (round-avg)':   's_ss_round_avg',
}

def evaluate_config(rb_eval, dc_col, rec_col, ss_col, w_dc, w_rec, w_ss):
    """Calculate all 12 metrics for a config."""
    slap = rb_eval[dc_col] * w_dc + rb_eval[rec_col] * w_rec + rb_eval[ss_col] * w_ss

    results = {}
    # Correlations
    oc_labels = {'first_3yr_ppg': '3yr', 'hit24': 'h24', 'hit12': 'h12', 'career_ppg': 'car'}
    for oc in outcome_cols:
        label = oc_labels[oc]
        valid = pd.DataFrame({'slap': slap, 'y': rb_eval[oc]}).dropna()
        if len(valid) > 5:
            results[f'r({label})'] = valid['slap'].corr(valid['y'])
        else:
            results[f'r({label})'] = np.nan

    # PRI-AVG
    pri = 0
    for oc, w in outcome_wts.items():
        key = f'r({oc[:3] if oc != "first_3yr_ppg" else "3yr"})'
        if key in results and not pd.isna(results[key]):
            pri += results[key] * w
    results['PRI-AVG'] = pri

    # Top-decile
    n10 = max(1, len(rb_eval) // 10)
    top_idx = slap.nlargest(n10).index
    top = rb_eval.loc[top_idx]
    results['T10%h24'] = top['hit24'].mean() * 100 if top['hit24'].notna().any() else np.nan
    results['T10%h12'] = top['hit12'].mean() * 100 if 'hit12' in top.columns and top['hit12'].notna().any() else np.nan
    ppg_vals = top['first_3yr_ppg'].dropna()
    results['T10%PPG'] = ppg_vals.mean() if len(ppg_vals) > 0 else np.nan

    # AUC-ROC
    results['AUC h24'] = auc_roc(rb_eval['hit24'], slap)
    results['AUC h12'] = auc_roc(rb_eval['hit12'], slap) if 'hit12' in rb_eval.columns else np.nan

    # Brier
    results['Brier24'] = brier_score(rb_eval['hit24'], slap)
    results['Brier12'] = brier_score(rb_eval['hit12'], slap) if 'hit12' in rb_eval.columns else np.nan

    return results

# --- Run all configs ---
all_results = {}
for cfg_name, (w_dc, w_rec, w_ss) in configs.items():
    ss_col = ss_col_map[cfg_name]
    res = evaluate_config(rb_eval, 's_dc', 's_rec_prod', ss_col, w_dc, w_rec, w_ss)
    all_results[cfg_name] = res

# --- Print results table ---
metric_order = ['PRI-AVG', 'r(3yr)', 'r(h24)', 'r(h12)', 'r(car)', 'T10%h24', 'T10%h12', 'T10%PPG', 'AUC h24', 'AUC h12', 'Brier24', 'Brier12']

print(f"\n  ── ALL CONFIGS ──")
header = f"  {'Config':<30s}"
for m in metric_order:
    header += f" {m:>8s}"
print(header)
print(f"  {'-'*150}")

baseline_res = all_results['A: 65/35/0 (baseline)']
for cfg_name, res in all_results.items():
    line = f"  {cfg_name:<30s}"
    for m in metric_order:
        v = res.get(m, np.nan)
        if 'T10%' in m and 'PPG' not in m:
            line += f" {v:>7.1f}%"
        elif 'T10%PPG' in m:
            line += f" {v:>8.2f}"
        elif 'Brier' in m:
            line += f" {v:>8.4f}"
        elif 'AUC' in m:
            line += f" {v:>8.4f}"
        else:
            line += f" {v:>+8.4f}"
    if cfg_name == 'A: 65/35/0 (baseline)':
        line += " ◄ BASE"
    else:
        delta_pri = res['PRI-AVG'] - baseline_res['PRI-AVG']
        line += f" ({delta_pri:+.4f})"
    print(line)

# --- Delta table ---
print(f"\n\n  ── DELTA TABLE: Each config vs 65/35 baseline ──")
print(f"  {'Metric':<14s}", end='')
non_base = [k for k in all_results.keys() if 'baseline' not in k]
for cfg in non_base:
    short = cfg.split(':')[0] + cfg.split(')')[-1] if ')' in cfg else cfg[:20]
    print(f" {cfg:<22s}", end='')
print()
print(f"  {'-'*170}")

for m in metric_order:
    print(f"  {m:<14s}", end='')
    for cfg in non_base:
        base_v = baseline_res.get(m, np.nan)
        cfg_v = all_results[cfg].get(m, np.nan)
        if pd.isna(base_v) or pd.isna(cfg_v):
            print(f" {'?':>22s}", end='')
            continue
        delta = cfg_v - base_v
        # For Brier, negative = improvement
        if 'Brier' in m:
            sign = '+' if delta < 0 else '-' if delta > 0 else '~'
        else:
            sign = '+' if delta > 0 else '-' if delta < 0 else '~'
        print(f" {delta:>+10.4f} {sign:<10s}", end='')
    print()

# Count improved/hurt/net
print(f"\n  {'':14s}", end='')
for cfg in non_base:
    improved = 0
    hurt = 0
    for m in metric_order:
        base_v = baseline_res.get(m, np.nan)
        cfg_v = all_results[cfg].get(m, np.nan)
        if pd.isna(base_v) or pd.isna(cfg_v): continue
        delta = cfg_v - base_v
        if 'Brier' in m:
            if delta < -0.0001: improved += 1
            elif delta > 0.0001: hurt += 1
        else:
            if delta > 0.0001: improved += 1
            elif delta < -0.0001: hurt += 1
    print(f" {'Imp %d / Hurt %d / Net %+d' % (improved, hurt, improved - hurt):<22s}", end='')
print()

# --- Top-decile roster comparison ---
print(f"\n\n  ── TOP-DECILE ROSTER: Who's in the top 22 under each approach? ──")
n10 = max(1, len(rb_eval) // 10)

baseline_slap = rb_eval['s_dc'] * 0.65 + rb_eval['s_rec_prod'] * 0.35
baseline_top = set(baseline_slap.nlargest(n10).index)

imputed_slap = rb_eval['s_dc'] * 0.60 + rb_eval['s_rec_prod'] * 0.30 + rb_eval['s_ss_imputed'] * 0.10
imputed_top = set(imputed_slap.nlargest(n10).index)

entered = imputed_top - baseline_top
left = baseline_top - imputed_top

if entered or left:
    print(f"\n  Players ENTERING top decile (60/30/10 imputed vs 65/35):")
    for idx in entered:
        r = rb_eval.loc[idx]
        h24 = int(r['hit24']) if pd.notna(r['hit24']) else '?'
        ppg = "%.1f" % r['first_3yr_ppg'] if pd.notna(r['first_3yr_ppg']) else "?"
        src = r.get('forty_source', '?')
        print(f"    + {r['player_name']:<25s} pick {int(r['pick']):>3d}  SS={r['speed_score_imputed']:.1f}  40src={src}  hit24={h24}  PPG={ppg}")

    print(f"\n  Players LEAVING top decile:")
    for idx in left:
        r = rb_eval.loc[idx]
        h24 = int(r['hit24']) if pd.notna(r['hit24']) else '?'
        ppg = "%.1f" % r['first_3yr_ppg'] if pd.notna(r['first_3yr_ppg']) else "?"
        print(f"    - {r['player_name']:<25s} pick {int(r['pick']):>3d}  hit24={h24}  PPG={ppg}")
else:
    print(f"\n  No changes — same 22 players in top decile")

# --- Final scorecard ---
print(f"\n\n{'=' * 120}")
print("SCORECARD")
print("=" * 120)

best_cfg = max(all_results.items(), key=lambda x: x[1]['PRI-AVG'])
print(f"\n  Baseline (65/35):          PRI-AVG = {baseline_res['PRI-AVG']:+.4f}, T10% hit24 = {baseline_res['T10%h24']:.1f}%")
print(f"  Best config:               {best_cfg[0]}")
print(f"                             PRI-AVG = {best_cfg[1]['PRI-AVG']:+.4f}, T10% hit24 = {best_cfg[1]['T10%h24']:.1f}%")
print(f"  Delta PRI-AVG:             {best_cfg[1]['PRI-AVG'] - baseline_res['PRI-AVG']:+.4f}")

# Coverage summary
print(f"\n  Coverage improvement:")
print(f"    Old (real 40 only):      {n_ss_real}/223 ({n_ss_real/223*100:.0f}%)")
print(f"    New (imputed 40):        {n_ss_imputed}/223 ({n_ss_imputed/223*100:.0f}%)")
print(f"    Still missing (no wt):   {223 - n_ss_imputed}")

print(f"\n{'=' * 120}")
print("ANALYSIS COMPLETE")
print("=" * 120)
