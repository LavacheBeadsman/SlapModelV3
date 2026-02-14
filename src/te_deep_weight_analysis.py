"""
TE Deep Weight Analysis
=======================
4 analyses to inform final weight decision:
  1. Outcome weight sensitivity (5 schemes including seasons_over_10ppg)
  2. seasons_over_10ppg as a 5th outcome target
  3. Bootstrap stability test (top configs head-to-head)
  4. Fine-grained config search between top contenders
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import glob, os, warnings
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')


def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()


# ============================================================================
# LOAD AND PREPARE DATA (same as te_recalculate_outcomes_and_validate.py)
# ============================================================================

print("=" * 120)
print("TE DEEP WEIGHT ANALYSIS")
print("=" * 120)

bt = pd.read_csv('data/te_backtest_master.csv')
print(f"Loaded: {len(bt)} TEs")

# Build component scores
def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

bt['s_dc'] = bt['pick'].apply(dc_score)

# Breakout
pff_file_map = {
    'data/receiving_summary (2).csv': 2015, 'data/receiving_summary (3).csv': 2016,
    'data/receiving_summary (4).csv': 2017, 'data/receiving_summary (5).csv': 2018,
    'data/receiving_summary (21).csv': 2019, 'data/receiving_summary (20).csv': 2020,
    'data/receiving_summary (19).csv': 2021, 'data/receiving_summary (18).csv': 2022,
    'data/receiving_summary (17).csv': 2023, 'data/receiving_summary (16).csv': 2024,
    'data/receiving_summary (15).csv': 2025,
}
all_pff = []
for fp, season in pff_file_map.items():
    pff = pd.read_csv(fp); pff['season'] = season; all_pff.append(pff)
pff_all = pd.concat(all_pff, ignore_index=True)
receivers = pff_all[pff_all['position'].isin(['TE', 'WR', 'HB'])].copy()
team_totals_pff = receivers.groupby(['team_name', 'season'])['yards'].sum().reset_index()
team_totals_pff.rename(columns={'yards': 'team_rec_yards'}, inplace=True)
te_pff = receivers[receivers['position'] == 'TE'].copy()
te_pff = te_pff.merge(team_totals_pff, on=['team_name', 'season'], how='left')
te_pff['dominator_pct'] = np.where(te_pff['team_rec_yards'] > 0,
    (te_pff['yards'] / te_pff['team_rec_yards']) * 100, 0)
te_pff['name_norm'] = te_pff['player'].apply(normalize_name)

THRESH = 15
bo_scores = {}
for _, te in bt.iterrows():
    nn = normalize_name(te['player_name'])
    matches = te_pff[(te_pff['name_norm'] == nn) & (te_pff['season'] < te['draft_year'])]
    if len(matches) == 0: continue
    peak_dom = matches['dominator_pct'].max()
    hit_ages = []
    for _, pm in matches.sort_values('season').iterrows():
        sa = te['draft_age'] - (te['draft_year'] - pm['season'])
        if pm['dominator_pct'] >= THRESH: hit_ages.append(sa)
    if hit_ages:
        ba = hit_ages[0]
        base = {18:100, 19:90, 20:75, 21:60, 22:45, 23:30}.get(int(min(ba, 23)), 20)
        if ba > 23: base = 20
        bonus = min((peak_dom - THRESH) * 0.5, 9.9)
        score = min(base + bonus, 99.9)
    else:
        score = min(35, 15 + peak_dom)
    bo_scores[te['player_name']] = score

bt['s_breakout'] = bt['player_name'].map(bo_scores)

# Production hybrid
bt['rec_per_tpa'] = np.where(
    (bt['cfbd_receptions'].notna()) & (bt['cfbd_team_pass_att'].notna()) & (bt['cfbd_team_pass_att'] > 0),
    bt['cfbd_receptions'] / bt['cfbd_team_pass_att'], np.nan)
bt['pff_rpg'] = np.where(
    (bt['pff_receptions'].notna()) & (bt['pff_player_game_count'].notna()) & (bt['pff_player_game_count'] > 0),
    bt['pff_receptions'] / bt['pff_player_game_count'], np.nan)
cv = bt['rec_per_tpa'].dropna(); pv = bt['pff_rpg'].dropna()
bt['cfbd_n'] = np.where(bt['rec_per_tpa'].notna(), (bt['rec_per_tpa']-cv.min())/(cv.max()-cv.min())*100, np.nan)
bt['pff_n'] = np.where(bt['pff_rpg'].notna(), (bt['pff_rpg']-pv.min())/(pv.max()-pv.min())*100, np.nan)
bt['s_production'] = np.where(bt['cfbd_n'].notna(), bt['cfbd_n'], bt['pff_n'])
bt['s_early_dec'] = bt['early_declare'] * 100
bt['s_ras'] = bt['te_ras'].apply(lambda x: x*10 if pd.notna(x) else np.nan)

# Fill missing with averages
bt['s_breakout_f'] = bt['s_breakout'].fillna(bt['s_breakout'].mean())
bt['s_production_f'] = bt['s_production'].fillna(bt['s_production'].mean())
bt['s_ras_f'] = bt['s_ras'].fillna(bt['s_ras'].mean())

# Eval sample: 2015-2024
eval_df = bt[bt['draft_year'] < 2025].copy()
print(f"Eval sample: {len(eval_df)} TEs (2015-2024)")


# ============================================================================
# HELPER: evaluate a config against arbitrary outcomes
# ============================================================================

def evaluate_config(df, comp_weights, outcome_cols, outcome_wts):
    """Evaluate a weight configuration. Returns dict of metrics."""
    df = df.copy()
    df['slap'] = sum(df[c]*w for c,w in comp_weights.items()).clip(0, 100)

    # Correlations
    results = {}; pri_s = 0; pri_t = 0
    for out in outcome_cols:
        v = df[['slap', out]].dropna()
        if len(v) >= 10:
            r,p = stats.pearsonr(v['slap'], v[out])
            results[out] = {'r':r,'p':p,'n':len(v)}
            pri_s += outcome_wts[out]*r; pri_t += outcome_wts[out]
    pri_avg = pri_s/pri_t if pri_t>0 else np.nan

    # Top decile metrics
    n_top = max(1, len(df)//10)
    top = df.nlargest(n_top, 'slap')
    h6 = top['top6_10g'].mean()*100 if 'top6_10g' in df.columns else np.nan
    h12 = top['top12_10g'].mean()*100 if 'top12_10g' in df.columns else np.nan
    top_ppg = top[top['best_3yr_ppg_10g'].notna()]['best_3yr_ppg_10g'].mean() if 'best_3yr_ppg_10g' in df.columns else np.nan

    # Disagreements
    df['dc_rank'] = df['s_dc'].rank(ascending=False, method='min')
    df['slap_rank'] = df['slap'].rank(ascending=False, method='min')
    dis10 = int((df['dc_rank'] - df['slap_rank']).abs().ge(10).sum())

    return {'pri_avg': pri_avg, 'outcomes': results, 'top6_top10': h6,
            'top12_top10': h12, 'ppg_top10': top_ppg, 'dis10': dis10, 'n_top': n_top}


# ============================================================================
# ANALYSIS 1: OUTCOME WEIGHT SENSITIVITY
# ============================================================================

print(f"\n\n{'='*120}")
print("ANALYSIS 1: OUTCOME WEIGHT SENSITIVITY")
print("Testing 5 different priority-weighting schemes for outcomes")
print(f"{'='*120}")

# 10g outcomes
BASE_OUTCOMES = ['best_3yr_ppg_10g', 'top12_10g', 'top6_10g', 'best_career_ppg_10g']

outcome_schemes = {
    'Original (40/25/20/15)': {
        'best_3yr_ppg_10g': 0.40, 'top12_10g': 0.25, 'top6_10g': 0.20, 'best_career_ppg_10g': 0.15
    },
    'Top-heavy top6 (25/25/35/15)': {
        'best_3yr_ppg_10g': 0.25, 'top12_10g': 0.25, 'top6_10g': 0.35, 'best_career_ppg_10g': 0.15
    },
    'Equal weight (25/25/25/25)': {
        'best_3yr_ppg_10g': 0.25, 'top12_10g': 0.25, 'top6_10g': 0.25, 'best_career_ppg_10g': 0.25
    },
    'PPG-focused (45/15/15/25)': {
        'best_3yr_ppg_10g': 0.45, 'top12_10g': 0.15, 'top6_10g': 0.15, 'best_career_ppg_10g': 0.25
    },
    'Top12-focused (25/40/20/15)': {
        'best_3yr_ppg_10g': 0.25, 'top12_10g': 0.40, 'top6_10g': 0.20, 'best_career_ppg_10g': 0.15
    },
}

# Test configs
test_configs = [
    ('DC only', {'s_dc': 1.00}),
    ('65/15/10/5/5 (current)', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.10, 's_early_dec':0.05, 's_ras_f':0.05}),
    ('65/15/15/5 (4c best)', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.15, 's_ras_f':0.05}),
    ('60/15/15/0/10 (SLSQP)', {'s_dc':0.60, 's_breakout_f':0.15, 's_production_f':0.15, 's_ras_f':0.10}),
    ('65/20/10/5 (4c no RAS)', {'s_dc':0.65, 's_breakout_f':0.20, 's_production_f':0.10, 's_early_dec':0.05}),
]

for scheme_name, scheme_wts in outcome_schemes.items():
    print(f"\n  --- {scheme_name} ---")
    print(f"  {'Config':<35} {'PRI-AVG':>8} {'Δ DC':>7} {'Rank':>5}")
    print(f"  {'-'*60}")

    results = []
    for label, cw in test_configs:
        r = evaluate_config(eval_df, cw, BASE_OUTCOMES, scheme_wts)
        results.append((label, r['pri_avg']))

    dc_val = [v for l,v in results if 'DC only' in l][0]
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    for rank, (label, pri) in enumerate(results_sorted, 1):
        d = pri - dc_val
        print(f"  {label:<35} {pri:>+.4f} {d:>+.4f} {rank:>5}")


# ============================================================================
# ANALYSIS 2: INCLUDE seasons_over_10ppg AS 5th OUTCOME
# ============================================================================

print(f"\n\n{'='*120}")
print("ANALYSIS 2: ADDING seasons_over_10ppg AS 5th OUTCOME")
print("Testing whether adding this metric changes optimal weights")
print(f"{'='*120}")

OUTCOMES_5 = ['best_3yr_ppg_10g', 'top12_10g', 'top6_10g', 'best_career_ppg_10g', 'seasons_over_10ppg_10g']

# Different 5-outcome weightings
five_outcome_schemes = {
    'Equal-ish (30/20/15/15/20)': {
        'best_3yr_ppg_10g': 0.30, 'top12_10g': 0.20, 'top6_10g': 0.15,
        'best_career_ppg_10g': 0.15, 'seasons_over_10ppg_10g': 0.20
    },
    'Szn10 light (35/25/15/15/10)': {
        'best_3yr_ppg_10g': 0.35, 'top12_10g': 0.25, 'top6_10g': 0.15,
        'best_career_ppg_10g': 0.15, 'seasons_over_10ppg_10g': 0.10
    },
    'Szn10 medium (30/20/15/15/20)': {
        'best_3yr_ppg_10g': 0.30, 'top12_10g': 0.20, 'top6_10g': 0.15,
        'best_career_ppg_10g': 0.15, 'seasons_over_10ppg_10g': 0.20
    },
    'Szn10 heavy (25/15/10/10/40)': {
        'best_3yr_ppg_10g': 0.25, 'top12_10g': 0.15, 'top6_10g': 0.10,
        'best_career_ppg_10g': 0.10, 'seasons_over_10ppg_10g': 0.40
    },
}

# Check partial correlation of components vs seasons_over_10ppg
print(f"\n  Partial correlations with seasons_over_10ppg (controlling for DC):")
components = {'Breakout':'s_breakout_f', 'Production':'s_production_f',
              'Early Declare':'s_early_dec', 'RAS':'s_ras_f'}
for name, col in components.items():
    valid = pd.DataFrame({'x':eval_df[col],'y':eval_df['seasons_over_10ppg_10g'],'z':eval_df['s_dc']}).dropna()
    if len(valid) < 15: continue
    sx,ix,_,_,_ = stats.linregress(valid['z'], valid['x'])
    rx = valid['x'] - (sx*valid['z']+ix)
    sy,iy,_,_,_ = stats.linregress(valid['z'], valid['y'])
    ry = valid['y'] - (sy*valid['z']+iy)
    r,p = stats.pearsonr(rx, ry)
    sig = "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else " "))
    print(f"    {name:<20} r={r:+.3f} (p={p:.4f}){sig}  N={len(valid)}")

# Raw correlation of DC vs seasons_over_10ppg
v = eval_df[['s_dc','seasons_over_10ppg_10g']].dropna()
r_raw,p_raw = stats.pearsonr(v['s_dc'], v['seasons_over_10ppg_10g'])
print(f"\n    DC raw correlation with seasons_over_10ppg: r={r_raw:+.3f} (p={p_raw:.4f})")

# Extended configs including more granular options
extended_configs = [
    ('DC only', {'s_dc': 1.00}),
    ('65/15/10/5/5 (current)', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.10, 's_early_dec':0.05, 's_ras_f':0.05}),
    ('65/15/15/5 (4c best)', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.15, 's_ras_f':0.05}),
    ('60/15/15/0/10 (SLSQP)', {'s_dc':0.60, 's_breakout_f':0.15, 's_production_f':0.15, 's_ras_f':0.10}),
    ('65/20/10/5 (no RAS)', {'s_dc':0.65, 's_breakout_f':0.20, 's_production_f':0.10, 's_early_dec':0.05}),
    ('70/15/10/5 (3c+ED)', {'s_dc':0.70, 's_breakout_f':0.15, 's_production_f':0.10, 's_early_dec':0.05}),
    ('65/15/15/0/5 (no ED)', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.15, 's_ras_f':0.05}),
]

for scheme_name, scheme_wts in five_outcome_schemes.items():
    print(f"\n  --- {scheme_name} ---")
    print(f"  {'Config':<35} {'PRI-AVG':>8} {'Δ DC':>7} {'Rank':>5}")
    print(f"  {'-'*60}")

    results = []
    for label, cw in extended_configs:
        r = evaluate_config(eval_df, cw, OUTCOMES_5, scheme_wts)
        results.append((label, r['pri_avg']))

    dc_val = [v for l,v in results if 'DC only' in l][0]
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    for rank, (label, pri) in enumerate(results_sorted, 1):
        d = pri - dc_val
        print(f"  {label:<35} {pri:>+.4f} {d:>+.4f} {rank:>5}")


# ============================================================================
# ANALYSIS 3: BOOTSTRAP STABILITY TEST
# ============================================================================

print(f"\n\n{'='*120}")
print("ANALYSIS 3: BOOTSTRAP STABILITY TEST")
print("Head-to-head comparison of top configs (1000 resamples)")
print(f"{'='*120}")

np.random.seed(42)
N_BOOT = 1000

# Outcome scheme for bootstrap
OC = BASE_OUTCOMES
OW = {'best_3yr_ppg_10g': 0.40, 'top12_10g': 0.25, 'top6_10g': 0.20, 'best_career_ppg_10g': 0.15}

boot_configs = [
    ('DC only', {'s_dc': 1.00}),
    ('65/15/10/5/5', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.10, 's_early_dec':0.05, 's_ras_f':0.05}),
    ('65/15/15/0/5', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.15, 's_ras_f':0.05}),
    ('60/15/15/0/10', {'s_dc':0.60, 's_breakout_f':0.15, 's_production_f':0.15, 's_ras_f':0.10}),
    ('65/20/10/0/5', {'s_dc':0.65, 's_breakout_f':0.20, 's_production_f':0.10, 's_ras_f':0.05}),
    ('70/15/10/0/5', {'s_dc':0.70, 's_breakout_f':0.15, 's_production_f':0.10, 's_ras_f':0.05}),
    ('65/15/15/5/0', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.15, 's_early_dec':0.05}),
]

# Bootstrap: for each resample, compute PRI-AVG for each config
boot_results = {label: [] for label, _ in boot_configs}

for b in range(N_BOOT):
    sample = eval_df.sample(n=len(eval_df), replace=True)
    for label, cw in boot_configs:
        sample_copy = sample.copy()
        sample_copy['slap'] = sum(sample_copy[c]*w for c,w in cw.items()).clip(0, 100)
        pri_s = 0; pri_t = 0
        for out in OC:
            v = sample_copy[['slap', out]].dropna()
            if len(v) >= 10:
                try:
                    r,_ = stats.pearsonr(v['slap'], v[out])
                    pri_s += OW[out]*r; pri_t += OW[out]
                except: pass
        if pri_t > 0:
            boot_results[label].append(pri_s/pri_t)
        else:
            boot_results[label].append(np.nan)

# Print bootstrap results
print(f"\n  {'Config':<25} {'Mean PRI':>9} {'Median':>8} {'95% CI':>20} {'Min':>7} {'Max':>7}")
print(f"  {'-'*82}")
for label, vals in boot_results.items():
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) == 0: continue
    mn = np.mean(vals); md = np.median(vals)
    ci_lo = np.percentile(vals, 2.5); ci_hi = np.percentile(vals, 97.5)
    print(f"  {label:<25} {mn:>+.4f}   {md:>+.4f} [{ci_lo:>+.4f}, {ci_hi:>+.4f}] {min(vals):>+.4f} {max(vals):>+.4f}")

# Pairwise head-to-head comparisons
print(f"\n  PAIRWISE HEAD-TO-HEAD (% of bootstrap resamples where row beats column):")
labels = [l for l,_ in boot_configs]
print(f"  {'':>25}", end="")
for l in labels:
    print(f" {l[:10]:>12}", end="")
print()
print(f"  {'-'*(25+12*len(labels))}")

for i, (l1, _) in enumerate(boot_configs):
    v1 = boot_results[l1]
    print(f"  {l1:<25}", end="")
    for j, (l2, _) in enumerate(boot_configs):
        if i == j:
            print(f" {'---':>12}", end="")
        else:
            v2 = boot_results[l2]
            wins = sum(1 for a,b in zip(v1,v2) if not np.isnan(a) and not np.isnan(b) and a>b)
            total = sum(1 for a,b in zip(v1,v2) if not np.isnan(a) and not np.isnan(b))
            pct = wins/total*100 if total > 0 else 0
            print(f" {pct:>10.1f}%", end="")
    print()

# Key comparison: current vs best alternatives
print(f"\n  KEY COMPARISONS:")
current = boot_results['65/15/10/5/5']
for label in ['65/15/15/0/5', '60/15/15/0/10', '65/20/10/0/5']:
    alt = boot_results[label]
    wins_alt = sum(1 for a,b in zip(alt, current) if not np.isnan(a) and not np.isnan(b) and a>b)
    total = sum(1 for a,b in zip(alt, current) if not np.isnan(a) and not np.isnan(b))
    pct = wins_alt/total*100 if total > 0 else 0
    diff = np.mean([a for a in alt if not np.isnan(a)]) - np.mean([c for c in current if not np.isnan(c)])
    print(f"    {label} beats current 65/15/10/5/5 in {pct:.1f}% of resamples (mean diff: {diff:+.4f})")


# ============================================================================
# ANALYSIS 4: FINE-GRAINED CONFIG SEARCH
# ============================================================================

print(f"\n\n{'='*120}")
print("ANALYSIS 4: FINE-GRAINED CONFIG SEARCH")
print("Testing at 1-2% increments between top contenders")
print(f"{'='*120}")

# The top contenders from previous analysis are:
#   1. 65/15/15/5 (DC/BO/Prod/RAS, no ED)
#   2. 60/15/15/0/10 (SLSQP optimal)
#   3. 65/15/10/5/5 (current)
# Key questions:
#   A. What is the best DC weight? (55-70 range)
#   B. Best Breakout weight? (10-25 range)
#   C. Best Production weight? (8-20 range)
#   D. Should ED be included? (0-5)
#   E. Should RAS be included? (0-10)

fine_configs = []

# Systematic: vary DC from 55 to 70 in 5% steps
# For each DC, test combinations of BO/Prod/ED/RAS that sum to (1-DC)
for dc_w in [0.55, 0.60, 0.65, 0.70]:
    remainder = 1.0 - dc_w

    # 3-component (no ED or RAS)
    for bo_w in np.arange(0.10, remainder - 0.04, 0.05):
        prod_w = round(remainder - bo_w, 2)
        if prod_w >= 0.05 and prod_w <= 0.30:
            label = f"DC{dc_w:.0%}/BO{bo_w:.0%}/P{prod_w:.0%}"
            fine_configs.append((label, {'s_dc':dc_w, 's_breakout_f':bo_w, 's_production_f':prod_w}))

    # 4-component with RAS (no ED)
    for bo_w in np.arange(0.10, remainder - 0.09, 0.05):
        for ras_w in [0.05, 0.10]:
            prod_w = round(remainder - bo_w - ras_w, 2)
            if prod_w >= 0.05 and prod_w <= 0.25:
                label = f"DC{dc_w:.0%}/BO{bo_w:.0%}/P{prod_w:.0%}/R{ras_w:.0%}"
                fine_configs.append((label, {'s_dc':dc_w, 's_breakout_f':bo_w, 's_production_f':prod_w, 's_ras_f':ras_w}))

    # 4-component with ED (no RAS)
    for bo_w in np.arange(0.10, remainder - 0.09, 0.05):
        for ed_w in [0.05, 0.10]:
            prod_w = round(remainder - bo_w - ed_w, 2)
            if prod_w >= 0.05 and prod_w <= 0.25:
                label = f"DC{dc_w:.0%}/BO{bo_w:.0%}/P{prod_w:.0%}/E{ed_w:.0%}"
                fine_configs.append((label, {'s_dc':dc_w, 's_breakout_f':bo_w, 's_production_f':prod_w, 's_early_dec':ed_w}))

    # 5-component
    for bo_w in np.arange(0.10, remainder - 0.14, 0.05):
        for ed_w in [0.05]:
            for ras_w in [0.05]:
                prod_w = round(remainder - bo_w - ed_w - ras_w, 2)
                if prod_w >= 0.05 and prod_w <= 0.20:
                    label = f"DC{dc_w:.0%}/BO{bo_w:.0%}/P{prod_w:.0%}/E{ed_w:.0%}/R{ras_w:.0%}"
                    fine_configs.append((label, {'s_dc':dc_w, 's_breakout_f':bo_w, 's_production_f':prod_w, 's_early_dec':ed_w, 's_ras_f':ras_w}))

# Add DC only baseline
fine_configs.append(('DC only', {'s_dc': 1.00}))

print(f"\n  Testing {len(fine_configs)} fine-grained configurations...\n")

# Evaluate all
fine_results = []
for label, cw in fine_configs:
    r = evaluate_config(eval_df, cw, BASE_OUTCOMES, OW)
    fine_results.append((label, r))

# Sort by PRI-AVG
fine_sorted = sorted(fine_results, key=lambda x: x[1]['pri_avg'] if not np.isnan(x[1]['pri_avg']) else -999, reverse=True)
dc_base = [r for l,r in fine_results if l == 'DC only'][0]['pri_avg']

# Print top 30
print(f"{'Rk':>3} {'Config':<45} {'PRI-AVG':>8} {'Δ DC':>7} {'Top10%t12':>10} {'Top10%t6':>9} {'Top10%PPG':>10} {'Dis10+':>7}")
print("-" * 105)
for i, (label, r) in enumerate(fine_sorted[:30], 1):
    d = r['pri_avg'] - dc_base
    ppg_s = f"{r['ppg_top10']:.2f}" if not np.isnan(r['ppg_top10']) else "N/A"
    marker = ""
    if label == 'DC only': marker = " BASE"
    elif '65%/BO15%/P10%/E5%/R5%' in label: marker = " <<<CURRENT"
    print(f"{i:>3}. {label:<45} {r['pri_avg']:>+.4f} {d:>+.4f} {r['top12_top10']:>9.1f}% {r['top6_top10']:>8.1f}% {ppg_s:>10} {r['dis10']:>7}{marker}")

# Find where current config ranks
current_rank = next(i for i, (l,r) in enumerate(fine_sorted, 1) if '65%/BO15%/P10%/E5%/R5%' in l) if any('65%/BO15%/P10%/E5%/R5%' in l for l,r in fine_sorted) else None
print(f"\n  Current config (65/15/10/5/5) ranks: #{current_rank}" if current_rank else "\n  Current config not in this search space")

# Show what the top configs have in common
print(f"\n  PATTERNS IN TOP 10 CONFIGS:")
top10 = fine_sorted[:10]
dc_vals = []; bo_vals = []; prod_vals = []; ras_vals = []; ed_vals = []
for label, r in top10:
    for cw_label, cw in fine_configs:
        if cw_label == label:
            dc_vals.append(cw.get('s_dc', 0))
            bo_vals.append(cw.get('s_breakout_f', 0))
            prod_vals.append(cw.get('s_production_f', 0))
            ras_vals.append(cw.get('s_ras_f', 0))
            ed_vals.append(cw.get('s_early_dec', 0))
            break

if dc_vals:
    print(f"    DC range:         {min(dc_vals)*100:.0f}% - {max(dc_vals)*100:.0f}%  (median: {np.median(dc_vals)*100:.0f}%)")
    print(f"    Breakout range:   {min(bo_vals)*100:.0f}% - {max(bo_vals)*100:.0f}%  (median: {np.median(bo_vals)*100:.0f}%)")
    print(f"    Production range: {min(prod_vals)*100:.0f}% - {max(prod_vals)*100:.0f}%  (median: {np.median(prod_vals)*100:.0f}%)")
    print(f"    RAS range:        {min(ras_vals)*100:.0f}% - {max(ras_vals)*100:.0f}%  (median: {np.median(ras_vals)*100:.0f}%)")
    print(f"    ED range:         {min(ed_vals)*100:.0f}% - {max(ed_vals)*100:.0f}%  (median: {np.median(ed_vals)*100:.0f}%)")

    has_ed = sum(1 for e in ed_vals if e > 0)
    has_ras = sum(1 for r in ras_vals if r > 0)
    print(f"\n    Configs with ED in top 10: {has_ed}/10")
    print(f"    Configs with RAS in top 10: {has_ras}/10")


# ============================================================================
# ANALYSIS 5: ALSO RUN SLSQP WITH seasons_over_10ppg INCLUDED
# ============================================================================

print(f"\n\n{'='*120}")
print("ANALYSIS 5: SLSQP OPTIMIZATION WITH 5 OUTCOMES (including seasons_over_10ppg)")
print(f"{'='*120}")

for scheme_name, scheme_wts in [
    ('4 outcomes (original)', OW),
    ('5 outcomes: szn10 at 10%', {
        'best_3yr_ppg_10g': 0.35, 'top12_10g': 0.22, 'top6_10g': 0.18,
        'best_career_ppg_10g': 0.15, 'seasons_over_10ppg_10g': 0.10
    }),
    ('5 outcomes: szn10 at 20%', {
        'best_3yr_ppg_10g': 0.30, 'top12_10g': 0.20, 'top6_10g': 0.15,
        'best_career_ppg_10g': 0.15, 'seasons_over_10ppg_10g': 0.20
    }),
]:
    oc_list = list(scheme_wts.keys())

    def obj_fn(weights, comp_cols=None, df=None, oc=None, ow=None):
        df = df.copy()
        df['slap'] = sum(df[c]*w for c,w in zip(comp_cols, weights)).clip(0,100)
        ps=0; pt=0
        for out in oc:
            v = df[['slap',out]].dropna()
            if len(v)>=10:
                r,_ = stats.pearsonr(v['slap'],v[out])
                ps += ow[out]*r; pt += ow[out]
        return -(ps/pt) if pt>0 else 0

    comp5 = ['s_dc', 's_breakout_f', 's_production_f', 's_early_dec', 's_ras_f']
    bounds5 = [(0.45,0.80),(0.05,0.30),(0.05,0.25),(0.00,0.15),(0.00,0.15)]
    starts5 = [[0.65,0.15,0.10,0.05,0.05],[0.60,0.15,0.15,0.05,0.05],[0.65,0.15,0.15,0.00,0.05],
               [0.55,0.20,0.15,0.05,0.05],[0.65,0.15,0.15,0.05,0.00],[0.60,0.15,0.15,0.00,0.10]]
    constraints = [{'type':'eq','fun':lambda w: sum(w)-1.0}]

    best_res = None; best_pri = -999
    for x0 in starts5:
        try:
            res = minimize(obj_fn, x0=x0, args=(comp5, eval_df, oc_list, scheme_wts),
                          method='SLSQP', bounds=bounds5, constraints=constraints, options={'maxiter':1000,'ftol':1e-8})
            if res.success and -res.fun > best_pri: best_pri = -res.fun; best_res = res
        except: pass

    if best_res is not None:
        w_raw = best_res.x
        w_rnd = [round(w*20)/20 for w in w_raw]
        w_rnd[0] += 1.0 - sum(w_rnd)
        print(f"\n  {scheme_name}:")
        print(f"    SLSQP raw:     DC={w_raw[0]:.3f} BO={w_raw[1]:.3f} Prod={w_raw[2]:.3f} ED={w_raw[3]:.3f} RAS={w_raw[4]:.3f}")
        print(f"    SLSQP rounded: DC={w_rnd[0]:.0%} BO={w_rnd[1]:.0%} Prod={w_rnd[2]:.0%} ED={w_rnd[3]:.0%} RAS={w_rnd[4]:.0%}")
        print(f"    PRI-AVG: {best_pri:+.4f}")

        # Also do 4-component (no ED)
        comp4 = ['s_dc', 's_breakout_f', 's_production_f', 's_ras_f']
        bounds4 = [(0.50,0.80),(0.05,0.30),(0.05,0.25),(0.00,0.15)]
        starts4 = [[0.65,0.15,0.15,0.05],[0.60,0.20,0.15,0.05],[0.70,0.15,0.10,0.05]]

        best4 = None; best4_pri = -999
        for x0 in starts4:
            try:
                res = minimize(obj_fn, x0=x0, args=(comp4, eval_df, oc_list, scheme_wts),
                              method='SLSQP', bounds=bounds4, constraints=constraints, options={'maxiter':1000,'ftol':1e-8})
                if res.success and -res.fun > best4_pri: best4_pri = -res.fun; best4 = res
            except: pass

        if best4 is not None:
            w4 = best4.x
            w4r = [round(w*20)/20 for w in w4]
            w4r[0] += 1.0 - sum(w4r)
            print(f"    4-comp raw:    DC={w4[0]:.3f} BO={w4[1]:.3f} Prod={w4[2]:.3f} RAS={w4[3]:.3f}")
            print(f"    4-comp rnd:    DC={w4r[0]:.0%} BO={w4r[1]:.0%} Prod={w4r[2]:.0%} RAS={w4r[3]:.0%}")
            print(f"    4-comp PRI:    {best4_pri:+.4f}")


# ============================================================================
# FINAL SUMMARY TABLE
# ============================================================================

print(f"\n\n{'='*120}")
print("FINAL SUMMARY: TOP CONTENDERS ACROSS ALL ANALYSES")
print(f"{'='*120}")

summary_configs = [
    ('DC only (baseline)', {'s_dc': 1.00}),
    ('65/15/10/5/5 (CURRENT)', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.10, 's_early_dec':0.05, 's_ras_f':0.05}),
    ('65/15/15/0/5 (no ED)', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.15, 's_ras_f':0.05}),
    ('65/20/10/0/5 (no ED, hi BO)', {'s_dc':0.65, 's_breakout_f':0.20, 's_production_f':0.10, 's_ras_f':0.05}),
    ('60/15/15/0/10 (SLSQP)', {'s_dc':0.60, 's_breakout_f':0.15, 's_production_f':0.15, 's_ras_f':0.10}),
    ('70/15/10/0/5 (hi DC)', {'s_dc':0.70, 's_breakout_f':0.15, 's_production_f':0.10, 's_ras_f':0.05}),
    ('60/20/15/0/5 (low DC)', {'s_dc':0.60, 's_breakout_f':0.20, 's_production_f':0.15, 's_ras_f':0.05}),
    ('65/15/15/5/0 (ED no RAS)', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.15, 's_early_dec':0.05}),
]

print(f"\n  {'Config':<35} {'4out PRI':>9} {'5out PRI':>9} {'Top10%t12':>10} {'Top10%t6':>9} {'Dis10+':>7}")
print(f"  {'-'*85}")

OW5 = {'best_3yr_ppg_10g': 0.35, 'top12_10g': 0.22, 'top6_10g': 0.18,
       'best_career_ppg_10g': 0.15, 'seasons_over_10ppg_10g': 0.10}

for label, cw in summary_configs:
    r4 = evaluate_config(eval_df, cw, BASE_OUTCOMES, OW)
    r5 = evaluate_config(eval_df, cw, OUTCOMES_5, OW5)
    print(f"  {label:<35} {r4['pri_avg']:>+.4f}   {r5['pri_avg']:>+.4f}   {r4['top12_top10']:>9.1f}% {r4['top6_top10']:>8.1f}% {r4['dis10']:>7}")


print(f"\n{'='*120}")
print("ANALYSIS COMPLETE")
print(f"{'='*120}")
