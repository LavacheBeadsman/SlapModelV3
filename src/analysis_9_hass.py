"""
Analysis 9: Height-Adjusted Speed Score (HaSS) for WRs
Formula: HaSS = (weight × 200) / (40_time^4) × (height / 73.0)
Developed by Shawn Siegele, builds on Barnwell Speed Score by incorporating height.

Tests:
1. Coverage comparison: HaSS vs RAS
2. Raw correlations with all 4 outcomes
3. Partial correlations controlling for DC
4. Within-round analysis
5. MNAR-aware imputation
6. Head-to-head weight configurations (6 models)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('/home/user/SlapModelV3')

# ============================================================
# DATA LOADING
# ============================================================

wr_bt = pd.read_csv('data/wr_backtest_expanded_final.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
combine = pd.read_parquet('data/nflverse/combine.parquet')

# Merge outcomes
wr_out = outcomes[outcomes.position == 'WR'][['player_name', 'draft_year', 'first_3yr_ppg', 'career_ppg']]
wr = wr_bt.merge(wr_out, on=['player_name', 'draft_year'], how='left')

# Parse combine height to inches
def parse_ht(h):
    if pd.isna(h): return np.nan
    parts = str(h).split('-')
    if len(parts) == 2:
        try: return int(parts[0]) * 12 + int(parts[1])
        except: return np.nan
    return np.nan

combine_wr = combine[combine.pos == 'WR'][['player_name', 'season', 'ht', 'wt', 'forty']].copy()
combine_wr.rename(columns={'season': 'draft_year'}, inplace=True)
combine_wr['height_inches'] = combine_wr['ht'].apply(parse_ht)
combine_wr.drop(columns=['ht'], inplace=True)

wr = wr.merge(combine_wr, on=['player_name', 'draft_year'], how='left')

# DC Score
wr['dc_score'] = wr['pick'].apply(lambda p: 100 - 2.40 * (p ** 0.62 - 1) if pd.notna(p) else np.nan)

# ============================================================
# CALCULATE HaSS
# ============================================================

def calc_hass(row):
    """HaSS = (weight × 200) / (40_time^4) × (height / 73.0)"""
    w, t, h = row.get('wt'), row.get('forty'), row.get('height_inches')
    if pd.isna(w) or pd.isna(t) or pd.isna(h) or t <= 0 or h <= 0:
        return np.nan
    speed_score = (w * 200) / (t ** 4)
    return speed_score * (h / 73.0)

def calc_speed_score(row):
    """Regular Speed Score = (weight × 200) / (40_time^4)"""
    w, t = row.get('wt'), row.get('forty')
    if pd.isna(w) or pd.isna(t) or t <= 0:
        return np.nan
    return (w * 200) / (t ** 4)

wr['hass'] = wr.apply(calc_hass, axis=1)
wr['speed_score'] = wr.apply(calc_speed_score, axis=1)

# WR breakout score (from CLAUDE.md formula)
def wr_breakout_score(breakout_age, dominator_pct):
    if pd.isna(breakout_age):
        if pd.isna(dominator_pct):
            return np.nan
        return min(35, 15 + dominator_pct)

    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20, 25: 20}
    ba_int = int(breakout_age)
    if ba_int < 18: ba_int = 18
    if ba_int > 25: ba_int = 25
    base = age_tiers.get(ba_int, 20)

    if pd.isna(dominator_pct):
        return base

    bonus = min((dominator_pct - 20) * 0.5, 9.9) if dominator_pct > 20 else 0
    return min(base + bonus, 99.9)

wr['breakout_score'] = wr.apply(lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)

# ============================================================
# TEST 1: COVERAGE
# ============================================================

print("=" * 100)
print("TEST 1: COVERAGE — HaSS vs RAS vs Speed Score")
print("=" * 100)

# Only count backtest WRs (2015-2024)
bt = wr[wr.draft_year <= 2024].copy()
total = len(bt)

has_hass = bt['hass'].notna().sum()
has_ras = bt['RAS'].notna().sum()
has_ss = bt['speed_score'].notna().sum()
has_forty = bt['forty'].notna().sum()
has_wt = bt['wt'].notna().sum()
has_ht = bt['height_inches'].notna().sum()

print(f"\nTotal backtest WRs (2015-2024): {total}")
print(f"\nIndividual metric coverage:")
print(f"  Height:     {has_ht:>4}/{total} ({has_ht/total*100:.1f}%)")
print(f"  Weight:     {has_wt:>4}/{total} ({has_wt/total*100:.1f}%)")
print(f"  40 time:    {has_forty:>4}/{total} ({has_forty/total*100:.1f}%)")
print(f"\nComposite score coverage:")
print(f"  HaSS (needs ht+wt+40):  {has_hass:>4}/{total} ({has_hass/total*100:.1f}%)")
print(f"  Speed Score (needs wt+40): {has_ss:>4}/{total} ({has_ss/total*100:.1f}%)")
print(f"  RAS:                    {has_ras:>4}/{total} ({has_ras/total*100:.1f}%)")

# Coverage by round
print(f"\nCoverage by round:")
print(f"  {'Round':<10} {'Total':>6} {'HaSS':>6} {'HaSS%':>7} {'RAS':>6} {'RAS%':>7} {'Diff':>8}")
print(f"  {'-'*55}")
for rnd in [1, 2, 3, 4, 5, 6, 7]:
    sub = bt[bt['round'] == rnd]
    if len(sub) == 0:
        continue
    h = sub['hass'].notna().sum()
    r = sub['RAS'].notna().sum()
    diff = h - r
    print(f"  Round {rnd:<3} {len(sub):>6} {h:>6} {h/len(sub)*100:>6.1f}% {r:>6} {r/len(sub)*100:>6.1f}% {diff:>+7}")

# Who has HaSS but not RAS, and vice versa?
has_hass_no_ras = bt[bt['hass'].notna() & bt['RAS'].isna()]
has_ras_no_hass = bt[bt['RAS'].notna() & bt['hass'].isna()]
has_both = bt[bt['hass'].notna() & bt['RAS'].notna()]
has_neither = bt[bt['hass'].isna() & bt['RAS'].isna()]

print(f"\n  Has both HaSS and RAS:    {len(has_both)}")
print(f"  Has HaSS only (no RAS):   {len(has_hass_no_ras)}")
print(f"  Has RAS only (no HaSS):   {len(has_ras_no_hass)}")
print(f"  Has neither:              {len(has_neither)}")

# MNAR check: missing HaSS by round and hit rate
print(f"\nMNAR check — missing HaSS by round:")
print(f"  {'Round':<10} {'Has HaSS':>10} {'Hit Rate':>10} {'Missing':>10} {'Hit Rate':>10}")
print(f"  {'-'*55}")
for rnd in [1, 2, 3, 4, 5]:
    sub = bt[bt['round'] == rnd]
    if len(sub) < 5:
        continue
    has = sub[sub['hass'].notna()]
    miss = sub[sub['hass'].isna()]
    h_rate = has['hit24'].mean() if len(has) > 0 else 0
    m_rate = miss['hit24'].mean() if len(miss) > 0 else 0
    print(f"  Round {rnd:<3} {len(has):>10} {h_rate:>10.1%} {len(miss):>10} {m_rate:>10.1%}")


# ============================================================
# TEST 2: RAW CORRELATIONS
# ============================================================

print("\n" + "=" * 100)
print("TEST 2: RAW CORRELATIONS — HaSS vs RAS vs Speed Score vs all 4 outcomes")
print("=" * 100)

metrics = [
    ('hass', 'HaSS'),
    ('speed_score', 'Speed Score'),
    ('RAS', 'RAS'),
]

outcomes_list = [
    ('hit24', 'Hit24 (binary)', True),
    ('hit12', 'Hit12 (binary)', True),
    ('first_3yr_ppg', 'First 3yr PPG', False),
    ('career_ppg', 'Career PPG', False),
]

print(f"\n{'Metric':<15} {'Outcome':<20} {'r':>8} {'p-value':>10} {'N':>6} {'Sig':>4}")
print("-" * 67)

for met_col, met_label in metrics:
    for out_col, out_label, is_binary in outcomes_list:
        valid = bt[[met_col, out_col]].dropna()
        if len(valid) < 10:
            print(f"{met_label:<15} {out_label:<20} {'N/A':>8} {'N/A':>10} {len(valid):>6}")
            continue
        if is_binary:
            r, p = stats.pointbiserialr(valid[out_col], valid[met_col])
        else:
            r, p = stats.pearsonr(valid[met_col], valid[out_col])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"{met_label:<15} {out_label:<20} {r:>8.4f} {p:>10.4f} {len(valid):>6} {sig:>4}")
    print()


# ============================================================
# TEST 3: PARTIAL CORRELATIONS (controlling for DC)
# ============================================================

print("=" * 100)
print("TEST 3: PARTIAL CORRELATIONS — controlling for Draft Capital")
print("Does HaSS add signal BEYOND what DC already captures?")
print("=" * 100)

def partial_corr(x_col, y_col, z_col, df):
    """Partial correlation of x and y, controlling for z."""
    valid = df[[x_col, y_col, z_col]].dropna()
    if len(valid) < 15:
        return np.nan, np.nan, len(valid)
    # Residualize x on z
    coef_xz = np.polyfit(valid[z_col], valid[x_col], 1)
    resid_x = valid[x_col] - np.polyval(coef_xz, valid[z_col])
    # Residualize y on z
    coef_yz = np.polyfit(valid[z_col], valid[y_col], 1)
    resid_y = valid[y_col] - np.polyval(coef_yz, valid[z_col])
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(valid)

print(f"\n{'Metric':<15} {'Outcome':<20} {'Partial r':>10} {'p-value':>10} {'N':>6} {'Sig':>4}")
print("-" * 70)

for met_col, met_label in metrics:
    for out_col, out_label, is_binary in outcomes_list:
        r, p, n = partial_corr(met_col, out_col, 'dc_score', bt)
        if np.isnan(r):
            print(f"{met_label:<15} {out_label:<20} {'N/A':>10} {'N/A':>10} {n:>6}")
            continue
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"{met_label:<15} {out_label:<20} {r:>10.4f} {p:>10.4f} {n:>6} {sig:>4}")
    print()

# Also show breakout age partial for reference
print("--- For reference: Breakout Age partial correlations ---")
for out_col, out_label, is_binary in outcomes_list:
    r, p, n = partial_corr('breakout_age', out_col, 'dc_score', bt)
    if np.isnan(r):
        continue
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"{'Breakout Age':<15} {out_label:<20} {r:>10.4f} {p:>10.4f} {n:>6} {sig:>4}")


# ============================================================
# TEST 4: WITHIN-ROUND ANALYSIS
# ============================================================

print("\n" + "=" * 100)
print("TEST 4: WITHIN-ROUND ANALYSIS — Does HaSS separate hits from misses in the SAME round?")
print("=" * 100)

round_groups = [
    ('Round 1', bt[bt['round'] == 1]),
    ('Round 2', bt[bt['round'] == 2]),
    ('Rounds 3-4', bt[bt['round'].isin([3, 4])]),
    ('Rounds 5-7', bt[bt['round'].isin([5, 6, 7])]),
]

for rnd_label, sub in round_groups:
    valid = sub[sub['hass'].notna() & sub['hit24'].notna()].copy()
    hits = valid[valid['hit24'] == 1]
    misses = valid[valid['hit24'] == 0]

    print(f"\n--- {rnd_label} ---")
    print(f"  Total with HaSS: {len(valid)}, Hits: {len(hits)}, Misses: {len(misses)}")

    if len(hits) < 3 or len(misses) < 3:
        print(f"  Too few in one group to analyze.")
        continue

    print(f"  Hit rate: {len(hits)/len(valid)*100:.1f}%")
    print(f"\n  {'Metric':<25} {'Hit Mean':>10} {'Miss Mean':>10} {'Diff':>10} {'Cohen d':>9} {'p-value':>10} {'Sig':>4}")
    print(f"  {'-'*83}")

    for col, label in [('hass', 'HaSS'), ('speed_score', 'Speed Score'), ('RAS', 'RAS')]:
        v = sub[[col, 'hit24']].dropna()
        h = v[v['hit24'] == 1][col]
        m = v[v['hit24'] == 0][col]
        if len(h) < 3 or len(m) < 3:
            continue
        t_stat, p_val = stats.ttest_ind(h, m, equal_var=False)
        pooled_std = np.sqrt((h.std()**2 + m.std()**2) / 2)
        d = (h.mean() - m.mean()) / pooled_std if pooled_std > 0 else 0
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
        print(f"  {label:<25} {h.mean():>10.2f} {m.mean():>10.2f} {h.mean()-m.mean():>10.2f} {d:>9.3f} {p_val:>10.4f} {sig:>4}")

    # Also show within-round correlations with all 4 outcomes
    print(f"\n  Within-round correlations:")
    for out_col, out_label, _ in outcomes_list:
        v = sub[['hass', out_col]].dropna()
        if len(v) < 8:
            continue
        r, p = stats.pearsonr(v['hass'], v[out_col])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"    HaSS vs {out_label:<18} r={r:>7.4f}  p={p:>8.4f}  n={len(v):>3} {sig}")


# ============================================================
# TEST 5: MNAR-AWARE IMPUTATION
# ============================================================

print("\n" + "=" * 100)
print("TEST 5: MNAR-AWARE IMPUTATION for HaSS")
print("If missing & Rd 1-2: assign 60th percentile. If missing & Rd 3+: assign 40th percentile.")
print("=" * 100)

# Calculate percentiles from observed data
hass_observed = bt[bt['hass'].notna()]['hass']
p60 = hass_observed.quantile(0.60)
p40 = hass_observed.quantile(0.40)
p50 = hass_observed.quantile(0.50)

print(f"\nHaSS percentiles (observed):")
print(f"  40th percentile: {p40:.2f}")
print(f"  50th percentile: {p50:.2f}")
print(f"  60th percentile: {p60:.2f}")
print(f"  Mean: {hass_observed.mean():.2f}")
print(f"  Std:  {hass_observed.std():.2f}")

bt['hass_mnar'] = bt['hass'].copy()
n_imputed_high = 0
n_imputed_low = 0
for idx, row in bt.iterrows():
    if pd.isna(row['hass']):
        if row['round'] <= 2:
            bt.at[idx, 'hass_mnar'] = p60
            n_imputed_high += 1
        else:
            bt.at[idx, 'hass_mnar'] = p40
            n_imputed_low += 1

print(f"\nImputation counts:")
print(f"  Rd 1-2 imputed to 60th pctile ({p60:.2f}): {n_imputed_high}")
print(f"  Rd 3+  imputed to 40th pctile ({p40:.2f}): {n_imputed_low}")
print(f"  Total imputed: {n_imputed_high + n_imputed_low}")
print(f"  Total with HaSS after imputation: {bt['hass_mnar'].notna().sum()}/{len(bt)}")

# Compare raw vs imputed correlations
print(f"\nRaw HaSS vs MNAR-imputed HaSS:")
print(f"  {'Version':<20} {'Outcome':<20} {'r':>8} {'p-value':>10} {'N':>6} {'Sig':>4}")
print(f"  {'-'*72}")

for version, col in [('HaSS (raw)', 'hass'), ('HaSS (MNAR)', 'hass_mnar')]:
    for out_col, out_label, is_binary in outcomes_list:
        valid = bt[[col, out_col]].dropna()
        if len(valid) < 10:
            continue
        if is_binary:
            r, p = stats.pointbiserialr(valid[out_col], valid[col])
        else:
            r, p = stats.pearsonr(valid[col], valid[out_col])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  {version:<20} {out_label:<20} {r:>8.4f} {p:>10.4f} {len(valid):>6} {sig:>4}")
    print()

# Partial correlations for MNAR version
print("Partial correlations (controlling for DC):")
print(f"  {'Version':<20} {'Outcome':<20} {'Partial r':>10} {'p-value':>10} {'N':>6} {'Sig':>4}")
print(f"  {'-'*76}")

for version, col in [('HaSS (raw)', 'hass'), ('HaSS (MNAR)', 'hass_mnar')]:
    for out_col, out_label, _ in outcomes_list:
        r, p, n = partial_corr(col, out_col, 'dc_score', bt)
        if np.isnan(r):
            continue
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  {version:<20} {out_label:<20} {r:>10.4f} {p:>10.4f} {n:>6} {sig:>4}")
    print()


# ============================================================
# TEST 6: HEAD-TO-HEAD WEIGHT CONFIGURATIONS
# ============================================================

print("=" * 100)
print("TEST 6: HEAD-TO-HEAD — 6 weight configurations against all 4 outcomes")
print("=" * 100)

def normalize_0_100(series):
    """Normalize a series to 0-100 scale."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50, index=series.index)
    return (series - mn) / (mx - mn) * 100

# Calculate normalized athletic scores
# For HaSS: normalize to 0-100
hass_valid = bt['hass'].dropna()
bt['hass_norm'] = bt['hass'].apply(lambda x: (x - hass_valid.min()) / (hass_valid.max() - hass_valid.min()) * 100 if pd.notna(x) else np.nan)

# For MNAR HaSS
hass_mnar_valid = bt['hass_mnar'].dropna()
bt['hass_mnar_norm'] = bt['hass_mnar'].apply(lambda x: (x - hass_mnar_valid.min()) / (hass_mnar_valid.max() - hass_mnar_valid.min()) * 100 if pd.notna(x) else np.nan)

# For RAS: already 0-10, scale to 0-100
bt['ras_norm'] = bt['RAS'] * 10

# Position average for missing values
avg_hass_norm = bt['hass_norm'].mean()
avg_hass_mnar_norm = bt['hass_mnar_norm'].mean()
avg_ras_norm = bt['ras_norm'].mean()

print(f"\nNormalized athletic score stats:")
print(f"  HaSS norm:      mean={bt['hass_norm'].mean():.1f}, std={bt['hass_norm'].std():.1f}, n={bt['hass_norm'].notna().sum()}")
print(f"  HaSS MNAR norm: mean={bt['hass_mnar_norm'].mean():.1f}, std={bt['hass_mnar_norm'].std():.1f}, n={bt['hass_mnar_norm'].notna().sum()}")
print(f"  RAS norm:       mean={bt['ras_norm'].mean():.1f}, std={bt['ras_norm'].std():.1f}, n={bt['ras_norm'].notna().sum()}")

# Define the 6 models
models = {
    'A: 75/25/0 (no athletic)': {'dc': 0.75, 'bo': 0.25, 'ath': 0.00, 'ath_col': None},
    'B: 70/20/10 (HaSS)':       {'dc': 0.70, 'bo': 0.20, 'ath': 0.10, 'ath_col': 'hass_norm'},
    'C: 65/20/15 (HaSS)':       {'dc': 0.65, 'bo': 0.20, 'ath': 0.15, 'ath_col': 'hass_norm'},
    'D: 72/20/8 (HaSS)':        {'dc': 0.72, 'bo': 0.20, 'ath': 0.08, 'ath_col': 'hass_norm'},
    'E: 70/20/10 (HaSS MNAR)':  {'dc': 0.70, 'bo': 0.20, 'ath': 0.10, 'ath_col': 'hass_mnar_norm'},
    'F: 65/20/15 (RAS)':        {'dc': 0.65, 'bo': 0.20, 'ath': 0.15, 'ath_col': 'ras_norm'},
}

# Calculate SLAP scores for each model
for model_name, params in models.items():
    col_name = f"slap_{model_name.split(':')[0].strip()}"

    if params['ath'] == 0:
        # No athletic component
        bt[col_name] = bt['dc_score'] * params['dc'] + bt['breakout_score'] * params['bo']
    else:
        ath_col = params['ath_col']
        # Use position average for missing
        if 'mnar' in (ath_col or ''):
            avg_ath = avg_hass_mnar_norm
        elif 'ras' in (ath_col or ''):
            avg_ath = avg_ras_norm
        else:
            avg_ath = avg_hass_norm

        bt[col_name] = bt.apply(
            lambda r: r['dc_score'] * params['dc'] +
                      r['breakout_score'] * params['bo'] +
                      (r[ath_col] if pd.notna(r.get(ath_col)) else avg_ath) * params['ath'],
            axis=1
        )

# Compare all models
print(f"\n{'Model':<35} {'Outcome':<20} {'r':>8} {'p-value':>10} {'N':>6} {'Sig':>4}")
print("-" * 87)

model_results = {}
for model_name, params in models.items():
    col_name = f"slap_{model_name.split(':')[0].strip()}"
    model_results[model_name] = {}

    for out_col, out_label, is_binary in outcomes_list:
        valid = bt[[col_name, out_col]].dropna()
        if len(valid) < 10:
            continue
        if is_binary:
            r, p = stats.pointbiserialr(valid[out_col], valid[col_name])
        else:
            r, p = stats.pearsonr(valid[col_name], valid[out_col])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"{model_name:<35} {out_label:<20} {r:>8.4f} {p:>10.4f} {len(valid):>6} {sig:>4}")
        model_results[model_name][out_col] = r
    print()

# Summary comparison table
print("\n" + "=" * 100)
print("SUMMARY: Average correlation across outcomes")
print("=" * 100)

print(f"\n{'Model':<35} {'hit24':>8} {'hit12':>8} {'3yr_ppg':>8} {'car_ppg':>8} {'AVG':>8} {'vs A':>8}")
print("-" * 93)

baseline_avg = None
for model_name in models:
    res = model_results[model_name]
    h24 = res.get('hit24', np.nan)
    h12 = res.get('hit12', np.nan)
    t3 = res.get('first_3yr_ppg', np.nan)
    cp = res.get('career_ppg', np.nan)
    vals = [v for v in [h24, h12, t3, cp] if not np.isnan(v)]
    avg = np.mean(vals) if vals else np.nan

    if baseline_avg is None:
        baseline_avg = avg

    delta = avg - baseline_avg if not np.isnan(avg) and baseline_avg is not None else np.nan
    delta_str = f"{delta:>+8.4f}" if not np.isnan(delta) else "    N/A"

    print(f"{model_name:<35} {h24:>8.4f} {h12:>8.4f} {t3:>8.4f} {cp:>8.4f} {avg:>8.4f} {delta_str}")


# ============================================================
# TIER BREAKDOWNS
# ============================================================

print("\n" + "=" * 100)
print("TIER BREAKDOWNS — Hit rates by SLAP score tier for each model")
print("=" * 100)

tier_bins = [0, 40, 50, 60, 70, 80, 90, 100]
tier_labels = ['0-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']

for model_name in models:
    col_name = f"slap_{model_name.split(':')[0].strip()}"
    valid = bt[[col_name, 'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']].dropna(subset=[col_name, 'hit24'])

    print(f"\n--- {model_name} ---")
    valid['tier'] = pd.cut(valid[col_name], bins=tier_bins, labels=tier_labels, include_lowest=True)

    tier_stats = valid.groupby('tier', observed=False).agg(
        n=('hit24', 'count'),
        hits24=('hit24', 'sum'),
        h24_rate=('hit24', 'mean'),
        hits12=('hit12', 'sum'),
        h12_rate=('hit12', 'mean'),
        avg_3yr=('first_3yr_ppg', 'mean'),
        avg_career=('career_ppg', 'mean'),
    ).reset_index()

    print(f"  {'Tier':<10} {'N':>5} {'H24':>5} {'H24%':>8} {'H12':>5} {'H12%':>8} {'3yr PPG':>10} {'Car PPG':>10}")
    print(f"  {'-'*68}")
    for _, row in tier_stats.iterrows():
        n = row['n']
        if n == 0:
            continue
        t3 = f"{row['avg_3yr']:.1f}" if pd.notna(row['avg_3yr']) else "N/A"
        tc = f"{row['avg_career']:.1f}" if pd.notna(row['avg_career']) else "N/A"
        print(f"  {row['tier']:<10} {n:>5.0f} {row['hits24']:>5.0f} {row['h24_rate']:>8.1%} {row['hits12']:>5.0f} {row['h12_rate']:>8.1%} {t3:>10} {tc:>10}")


# ============================================================
# HASS DESCRIPTIVE STATS
# ============================================================

print("\n" + "=" * 100)
print("APPENDIX: HaSS Descriptive Statistics")
print("=" * 100)

hass_data = bt[bt['hass'].notna()].copy()
print(f"\nHaSS distribution (n={len(hass_data)}):")
print(f"  Min:    {hass_data['hass'].min():.2f}")
print(f"  25th:   {hass_data['hass'].quantile(0.25):.2f}")
print(f"  Median: {hass_data['hass'].median():.2f}")
print(f"  75th:   {hass_data['hass'].quantile(0.75):.2f}")
print(f"  Max:    {hass_data['hass'].max():.2f}")
print(f"  Mean:   {hass_data['hass'].mean():.2f}")
print(f"  Std:    {hass_data['hass'].std():.2f}")

# Show top 15 and bottom 15 HaSS scores
print(f"\nTop 15 HaSS scores:")
top = hass_data.nlargest(15, 'hass')[['player_name', 'draft_year', 'pick', 'round', 'college',
                                       'height_inches', 'wt', 'forty', 'hass', 'hit24']]
print(f"  {'Player':<25} {'Year':>5} {'Pick':>5} {'Rd':>3} {'Ht':>4} {'Wt':>5} {'40':>5} {'HaSS':>7} {'Hit24':>5}")
print(f"  {'-'*73}")
for _, r in top.iterrows():
    print(f"  {r['player_name']:<25} {r['draft_year']:>5.0f} {r['pick']:>5.0f} {r['round']:>3.0f} {r['height_inches']:>4.0f} {r['wt']:>5.0f} {r['forty']:>5.2f} {r['hass']:>7.2f} {r['hit24']:>5.0f}")

print(f"\nBottom 15 HaSS scores:")
bottom = hass_data.nsmallest(15, 'hass')[['player_name', 'draft_year', 'pick', 'round', 'college',
                                           'height_inches', 'wt', 'forty', 'hass', 'hit24']]
print(f"  {'Player':<25} {'Year':>5} {'Pick':>5} {'Rd':>3} {'Ht':>4} {'Wt':>5} {'40':>5} {'HaSS':>7} {'Hit24':>5}")
print(f"  {'-'*73}")
for _, r in bottom.iterrows():
    print(f"  {r['player_name']:<25} {r['draft_year']:>5.0f} {r['pick']:>5.0f} {r['round']:>3.0f} {r['height_inches']:>4.0f} {r['wt']:>5.0f} {r['forty']:>5.2f} {r['hass']:>7.2f} {r['hit24']:>5.0f}")


print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
