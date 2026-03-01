import csv, os, glob, re
import numpy as np
import pandas as pd
from scipy import stats

###############################################################################
# HELPERS
###############################################################################
def parse_float(val):
    if not val or not val.strip():
        return np.nan
    val = val.strip().replace(',', '').replace('*', '').replace('%', '')
    if val.upper() in ['N/A', 'NA', '#DIV/0!', 'UDFA', '-']:
        return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan

def find_col_exact(headers, target):
    """Find column whose stripped, lowered header EXACTLY matches target."""
    for i, h in enumerate(headers):
        h_clean = h.strip().lower().replace('\n', ' ').replace('\r', ' ')
        if h_clean == target.lower():
            return i
    return None

def find_col_contains(headers, targets, exclude=None):
    """Find column whose header contains any target but NOT any exclude string."""
    for i, h in enumerate(headers):
        h_clean = h.strip().lower().replace('\n', ' ').replace('\r', ' ')
        for t in targets:
            if t.lower() in h_clean:
                if exclude and any(e.lower() in h_clean for e in exclude):
                    continue
                return i
    return None

###############################################################################
# STEP 1: Parse Anatomy files
###############################################################################
wr_rows = []
rb_rows = []

files = sorted(glob.glob("/home/user/SlapModelV3/data/Anatomy of Top WR & RB - *.csv"))
skip = ['2000s', '2025', '2026', '2024 (1)']

for f in files:
    fname = os.path.basename(f)
    if any(s in fname for s in skip):
        continue
    year_match = re.search(r'(\d{4})', fname)
    if not year_match:
        continue
    draft_year = int(year_match.group(1))
    
    with open(f, encoding='utf-8-sig') as fh:
        all_rows = list(csv.reader(fh))
    if not all_rows:
        continue
    
    header = [h.strip().replace('\n', ' ').replace('\r', ' ') for h in all_rows[0]]
    
    # Find WR/RB separator
    sep = None
    for i in range(8, len(header)):
        if not header[i]:
            sep = i
            break
    
    wr_h = header[:sep] if sep else header
    rb_h = header[sep:] if sep else []
    
    # WR column indices - use exact matching for YPR to avoid YPR DOM
    w_bmi = find_col_exact(wr_h, 'bmi')
    w_hand = find_col_contains(wr_h, ['hand size', 'hand'])
    w_arm = find_col_contains(wr_h, ['arm length', 'arm'])
    w_vert = find_col_contains(wr_h, ['standing vert', 'vert'])
    w_ypr = find_col_exact(wr_h, 'ypr')  # EXACT match to avoid YPR DOM
    w_yptpa = find_col_exact(wr_h, 'yptpa')
    w_ppg = find_col_contains(wr_h, ['10+ ppg'])
    w_fin = find_col_contains(wr_h, ['avg top finish'])
    
    # RB column indices (relative to sep)
    r_bmi = find_col_exact(rb_h, 'bmi')
    r_burst = find_col_contains(rb_h, ['burst'])
    r_bdr = find_col_exact(rb_h, 'bdr')
    r_yard = find_col_contains(rb_h, ['peak yard', 'yardage', 'scrimmage'])
    r_rec = find_col_contains(rb_h, ['peak rec', 'recption', '30+ rec'])
    r_ppg = find_col_contains(rb_h, ['10+ ppg'])
    r_fin = find_col_contains(rb_h, ['avg top finish'])
    
    def get_val(row, idx, offset=0):
        abs_idx = idx + offset if idx is not None else None
        if abs_idx is not None and abs_idx < len(row):
            return parse_float(row[abs_idx])
        return np.nan
    
    for row in all_rows[1:]:
        if len(row) < 5:
            continue
        
        # WR
        wr_name = row[0].strip()
        if wr_name and any(c.isalpha() for c in wr_name) and len(wr_name) > 2:
            wr_rows.append({
                'player_name': wr_name,
                'draft_year': draft_year,
                'BMI': get_val(row, w_bmi),
                'Hand_Size': get_val(row, w_hand),
                'Arm_Length': get_val(row, w_arm),
                'Standing_Vert': get_val(row, w_vert),
                'YPR': get_val(row, w_ypr),
                'YPTPA': get_val(row, w_yptpa),
                'ppg_seasons_anat': get_val(row, w_ppg),
                'avg_finish_anat': get_val(row, w_fin),
            })
        
        # RB
        if sep:
            rb_name = None
            for check in [sep, sep+1, sep+2]:
                if check < len(row):
                    v = row[check].strip()
                    if v and any(c.isalpha() for c in v) and len(v) > 2 and v not in ['N/A']:
                        rb_name = v
                        break
            if rb_name:
                rb_rows.append({
                    'player_name': rb_name,
                    'draft_year': draft_year,
                    'BMI': get_val(row, r_bmi, sep),
                    'Burst_Score': get_val(row, r_burst, sep),
                    'BDR': get_val(row, r_bdr, sep),
                    'Peak_Yardage': get_val(row, r_yard, sep),
                    'Peak_REC': get_val(row, r_rec, sep),
                    'ppg_seasons_anat': get_val(row, r_ppg, sep),
                    'avg_finish_anat': get_val(row, r_fin, sep),
                })

wr_anat = pd.DataFrame(wr_rows)
rb_anat = pd.DataFrame(rb_rows)

# Verify YPR fix — Chase should be ~19.6, not 41.8
chase = wr_anat[(wr_anat['player_name'].str.contains('Chase')) & (wr_anat['draft_year']==2021)]
print("YPR column fix verification:")
print(f"  Ja'Marr Chase YPR = {chase['YPR'].values[0] if len(chase) else 'NOT FOUND'} (should be ~19.6)")
jefferson = wr_anat[(wr_anat['player_name'].str.contains('Jefferson')) & (wr_anat['draft_year']==2020)]
print(f"  Justin Jefferson YPR = {jefferson['YPR'].values[0] if len(jefferson) else 'NOT FOUND'} (should be ~14.6)")

print(f"\nExtracted {len(wr_anat)} WR and {len(rb_anat)} RB from Anatomy files")

###############################################################################
# STEP 2: Match to SLAP backtest
###############################################################################
wr_bt = pd.read_csv('data/wr_backtest_all_components.csv')
wr_bt = wr_bt[['player_name', 'draft_year', 'pick', 'hit24', 'hit12', 'best_ppr']].copy()
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
rb_bt = rb_bt[['player_name', 'draft_year', 'pick', 'hit24', 'hit12', 'best_ppr', 'best_ppg']].copy()
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
outcomes = outcomes[['player_name', 'position', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']].copy()

def normalize_name(name):
    return re.sub(r'\s+', ' ', name.strip().replace("\u2019", "'").replace("\u2018", "'"))

for df in [wr_anat, wr_bt, rb_anat, rb_bt, outcomes]:
    df['name_norm'] = df['player_name'].apply(normalize_name)

# Merge
wr_merged = wr_anat.merge(wr_bt, on=['name_norm', 'draft_year'], how='inner', suffixes=('_anat', '_bt'))
wr_outcomes = outcomes[outcomes['position'] == 'WR'][['name_norm', 'draft_year', 'first_3yr_ppg', 'career_ppg']]
wr_merged = wr_merged.merge(wr_outcomes, on=['name_norm', 'draft_year'], how='left')

rb_merged = rb_anat.merge(rb_bt, on=['name_norm', 'draft_year'], how='inner', suffixes=('_anat', '_bt'))
rb_outcomes = outcomes[outcomes['position'] == 'RB'][['name_norm', 'draft_year', 'first_3yr_ppg', 'career_ppg']]
rb_merged = rb_merged.merge(rb_outcomes, on=['name_norm', 'draft_year'], how='left')

def dc_score(pick):
    if pd.isna(pick): return np.nan
    return 100 - 2.40 * (pick ** 0.62 - 1)

wr_merged['dc_score'] = wr_merged['pick'].apply(dc_score)
rb_merged['dc_score'] = rb_merged['pick'].apply(dc_score)

print(f"\nWR matched: {len(wr_merged)}/{len(wr_bt)} ({100*len(wr_merged)/len(wr_bt):.1f}%)")
print(f"RB matched: {len(rb_merged)}/{len(rb_bt)} ({100*len(rb_merged)/len(rb_bt):.1f}%)")

# Coverage after matching
print("\nWR Matched Variable Coverage:")
for var in ['BMI', 'Hand_Size', 'Arm_Length', 'Standing_Vert', 'YPR', 'YPTPA']:
    n = wr_merged[var].notna().sum()
    print(f"  {var:15s}: {n:>4d}/{len(wr_merged)} ({100*n/len(wr_merged):5.1f}%)")

print("\nRB Matched Variable Coverage:")
for var in ['BMI', 'Burst_Score', 'BDR', 'Peak_Yardage', 'Peak_REC']:
    n = rb_merged[var].notna().sum()
    print(f"  {var:15s}: {n:>4d}/{len(rb_merged)} ({100*n/len(rb_merged):5.1f}%)")

###############################################################################
# STEP 3 & 4: Correlations
###############################################################################

def partial_corr(x, y, z):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask.sum() < 15:
        return np.nan, np.nan, mask.sum()
    x, y, z = x[mask], y[mask], z[mask]
    x_r, y_r, z_r = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    def resid(a, b):
        slope, intercept = np.polyfit(b, a, 1)
        return a - (slope * b + intercept)
    r, p = stats.pearsonr(resid(x_r, z_r), resid(y_r, z_r))
    return r, p, mask.sum()

def raw_spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 15:
        return np.nan, np.nan, mask.sum()
    r, p = stats.spearmanr(x[mask], y[mask])
    return r, p, mask.sum()

wr_new_vars = ['BMI', 'Hand_Size', 'Arm_Length', 'Standing_Vert', 'YPR', 'YPTPA']
rb_new_vars = ['BMI', 'Burst_Score', 'BDR', 'Peak_Yardage', 'Peak_REC']
core_outcomes = ['hit24', 'hit12', 'career_ppg', 'first_3yr_ppg']

# Run all correlations
all_results = []

print("\n" + "=" * 90)
print("WR CORRELATIONS (raw Spearman + partial Spearman controlling for DC)")
print("=" * 90)
print(f"{'Variable':15s} | {'Outcome':15s} | {'N':>4s} | {'Raw r':>7s} | {'Raw p':>8s} | {'Part. r':>7s} | {'Part. p':>8s}")
print("-" * 90)

for var in wr_new_vars:
    for out in core_outcomes:
        x = wr_merged[var].values.astype(float)
        y = wr_merged[out].values.astype(float)
        dc = wr_merged['dc_score'].values.astype(float)
        raw_r, raw_p, raw_n = raw_spearman(x, y)
        part_r, part_p, part_n = partial_corr(x, y, dc)
        all_results.append({
            'Position': 'WR', 'Variable': var, 'Outcome': out,
            'N': raw_n, 'Raw_r': raw_r, 'Raw_p': raw_p,
            'Partial_r': part_r, 'Partial_p': part_p,
        })
        sig_raw = "*" if raw_p < 0.05 else " "
        sig_part = "*" if part_p < 0.05 else " "
        print(f"{var:15s} | {out:15s} | {raw_n:4.0f} | {raw_r:+7.3f}{sig_raw}| {raw_p:8.4f} | {part_r:+7.3f}{sig_part}| {part_p:8.4f}")

print("\n" + "=" * 90)
print("RB CORRELATIONS (raw Spearman + partial Spearman controlling for DC)")
print("=" * 90)
print(f"{'Variable':15s} | {'Outcome':15s} | {'N':>4s} | {'Raw r':>7s} | {'Raw p':>8s} | {'Part. r':>7s} | {'Part. p':>8s}")
print("-" * 90)

for var in rb_new_vars:
    for out in core_outcomes:
        x = rb_merged[var].values.astype(float)
        y = rb_merged[out].values.astype(float)
        dc = rb_merged['dc_score'].values.astype(float)
        raw_r, raw_p, raw_n = raw_spearman(x, y)
        part_r, part_p, part_n = partial_corr(x, y, dc)
        all_results.append({
            'Position': 'RB', 'Variable': var, 'Outcome': out,
            'N': raw_n, 'Raw_r': raw_r, 'Raw_p': raw_p,
            'Partial_r': part_r, 'Partial_p': part_p,
        })
        sig_raw = "*" if raw_p < 0.05 else " "
        sig_part = "*" if part_p < 0.05 else " "
        print(f"{var:15s} | {out:15s} | {raw_n:4.0f} | {raw_r:+7.3f}{sig_raw}| {raw_p:8.4f} | {part_r:+7.3f}{sig_part}| {part_p:8.4f}")

###############################################################################
# STEP 5: Summary ranked by predictive value after DC
###############################################################################
print("\n" + "=" * 90)
print("RANKED SUMMARY — All variables by independent signal BEYOND draft capital")
print("=" * 90)
print("Method: Average |partial Spearman r| across hit24, hit12, career_ppg, first_3yr_ppg")
print("'Sig/4' = how many of the 4 outcomes have p < 0.05 for the partial correlation")
print("Direction = average sign of partial r (+positive = higher value → better NFL outcome)\n")

df = pd.DataFrame(all_results)

summary = df.groupby(['Position', 'Variable']).agg(
    avg_abs_partial_r=('Partial_r', lambda x: np.nanmean(np.abs(x))),
    avg_partial_r=('Partial_r', 'mean'),
    avg_raw_r=('Raw_r', 'mean'),
    sig_count=('Partial_p', lambda x: (x < 0.05).sum()),
    min_n=('N', 'min'),
).reset_index()
summary = summary.sort_values('avg_abs_partial_r', ascending=False)

print(f"{'#':>2s} {'Pos':>3s}  {'Variable':15s}  {'Avg|Part.r|':>11s}  {'Direction':>9s}  {'Avg Raw r':>9s}  {'Sig/4':>5s}  {'MinN':>4s}  {'Assessment'}")
print("-" * 95)

for rank, (_, row) in enumerate(summary.iterrows(), 1):
    abs_r = row['avg_abs_partial_r']
    sig = int(row['sig_count'])
    direction = "+" if row['avg_partial_r'] > 0 else "-"
    
    if abs_r >= 0.15 and sig >= 2:
        assess = "REAL SIGNAL — worth exploring"
    elif abs_r >= 0.10 and sig >= 1:
        assess = "Weak signal — borderline"
    elif abs_r >= 0.08:
        assess = "Marginal — probably noise"
    else:
        assess = "No signal"
    
    print(f"{rank:2d} {row['Position']:>3s}  {row['Variable']:15s}  {abs_r:11.4f}  {direction:>4s}{abs(row['avg_partial_r']):+.3f}  {row['avg_raw_r']:+9.4f}  {sig:3d}/4  {int(row['min_n']):4d}  {assess}")

# Detail for promising ones
print("\n" + "=" * 90)
print("DETAIL: Variables with avg |partial r| >= 0.08")
print("=" * 90)

for _, prow in summary[summary['avg_abs_partial_r'] >= 0.08].iterrows():
    pos, var = prow['Position'], prow['Variable']
    print(f"\n--- {pos} {var} (avg |partial r| = {prow['avg_abs_partial_r']:.4f}) ---")
    detail = df[(df['Position'] == pos) & (df['Variable'] == var)]
    for _, d in detail.iterrows():
        sig = " ***" if d['Partial_p'] < 0.01 else " * " if d['Partial_p'] < 0.05 else "   "
        print(f"  vs {d['Outcome']:15s}:  raw r = {d['Raw_r']:+.3f}  →  partial r = {d['Partial_r']:+.3f}  (p = {d['Partial_p']:.4f}){sig}  N = {int(d['N'])}")

# Final interpretation
print("\n" + "=" * 90)
print("BOTTOM LINE")
print("=" * 90)
print("""
WR variables (BMI, Hand_Size, Arm_Length, Standing_Vert, YPR, YPTPA):
  - NONE show meaningful independent signal after controlling for DC
  - YPTPA looks good RAW (r = +0.17 to +0.26) but collapses after DC control
    → DC already captures the production signal YPTPA carries

RB variables with real signal:
  - Peak_REC (peak receptions in a season): consistent +0.18 partial r
    → RBs who catch more passes in college → better NFL outcomes beyond DC
    → This is DIFFERENT from our RYPTPA (receiving yards per team pass attempt)
    → Peak_REC measures raw volume, RYPTPA measures efficiency
  - BDR (backfield dominator rating): consistent +0.16 partial r
    → RBs who dominate their own backfield → better NFL outcomes beyond DC
    → Not currently in our model

RB variables that are marginal:
  - Peak_Yardage: strong raw signal (+0.26) but mostly captured by DC
    → avg partial r of +0.13 but no individual outcome reaches p < 0.05
""")

