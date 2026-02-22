"""
WR Model Evaluation: 6 Statistical Tests Beyond Correlation
=============================================================
Compare 5 models:
  A: 4-component (75/17/4/4 — DC/BO+Rush/TM/ED)
  B: Same + 5% RAS (70/17/4/4/5)
  C: Same + 5% weight (70/17/4/4/5)
  D: DC only (100/0/0/0)
  E: Current CLAUDE.md model (65/20/0/0/15 — DC/Breakout/RAS)

Tests:
  1. Brier Score (hit24 probability calibration)
  2. AUC-ROC (hit24 and hit12 separation)
  3. Precision at top decile (when model is most confident)
  4. Log Loss (penalty for confident wrong predictions)
  5. Bootstrap stability (1000 resamples — does RAS/weight get >0% weight?)
  6. Calibration plot (predicted vs actual hit rates by decile)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('/home/user/SlapModelV3')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def dc_score(pick):
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))

def wr_breakout_score(breakout_age, dominator_pct):
    if pd.isna(breakout_age) or breakout_age >= 99:
        if pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        else:
            return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base_score = age_tiers.get(int(breakout_age), 20)
    if pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
        bonus = max(0, bonus)
    else:
        bonus = 0
    return min(base_score + bonus, 99.9)


# ============================================================================
# DATA LOADING
# ============================================================================

print("=" * 110)
print("WR MODEL EVALUATION: 6 STATISTICAL TESTS BEYOND CORRELATION")
print("=" * 110)

wr = pd.read_csv('data/wr_backtest_all_components.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
tm = pd.read_csv('data/wr_teammate_scores.csv')

# Merge outcomes
out_wr = outcomes[outcomes['position'] == 'WR'][
    ['player_name', 'draft_year', 'first_3yr_ppg', 'career_ppg']
].copy()
wr = wr.merge(out_wr, on=['player_name', 'draft_year'], how='left')

# Merge teammate
tm_lookup = dict(zip(zip(tm['player_name'], tm['draft_year']), tm['avg_teammate_dc']))
wr['teammate_dc'] = wr.apply(lambda x: tm_lookup.get((x['player_name'], x['draft_year']), 0), axis=1)

# Merge combine data for weight
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine_wr = combine[combine['pos'] == 'WR'][['player_name', 'season', 'wt']].copy()
combine_wr.rename(columns={'season': 'draft_year'}, inplace=True)
wr = wr.merge(combine_wr, on=['player_name', 'draft_year'], how='left')

# Fuzzy match for unmatched
import re
def normalize_name(name):
    n = str(name).lower().strip()
    n = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv)$', '', n).strip()
    return n

name_map = {
    'Tank Dell': 'Nathaniel Dell', 'Bub Means': 'Jerrod Means',
    'Hollywood Brown': 'Marquise Brown', 'DK Metcalf': 'D.K. Metcalf',
    'DJ Chark': 'D.J. Chark', 'DJ Moore': 'D.J. Moore',
    'KJ Osborn': 'K.J. Osborn', 'KJ Hamler': 'K.J. Hamler',
    'AJ Brown': 'A.J. Brown',
}

unmatched = wr[wr['wt'].isna()].copy()
for idx, row in unmatched.iterrows():
    pname = row['player_name']
    dy = row['draft_year']
    alt_name = name_map.get(pname, pname)
    cand = combine_wr[combine_wr['draft_year'] == dy]
    match = cand[cand['player_name'] == alt_name]
    if len(match) == 0:
        pn = normalize_name(pname)
        parts = pn.split()
        if len(parts) >= 2:
            for _, crow in cand.iterrows():
                cn = normalize_name(crow['player_name'])
                cparts = cn.split()
                if len(cparts) >= 2 and parts[-1] == cparts[-1] and parts[0][0] == cparts[0][0]:
                    match = pd.DataFrame([crow])
                    break
    if len(match) > 0:
        wr.loc[idx, 'wt'] = match.iloc[0]['wt']

# Compute component scores
wr['s_dc'] = wr['pick'].apply(dc_score)
wr['s_breakout'] = wr.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)
wr['s_teammate'] = wr['teammate_dc'].clip(0, 100)
wr['s_declare'] = wr['early_declare'] * 100

# Enhanced breakout with +5 rushing bonus
rush_flag = (wr['rush_yards'] >= 20).astype(int)
wr['s_bo_rush'] = (wr['s_breakout'] + rush_flag * 5).clip(0, 99.9)

# RAS on 0-100 scale
wr['s_ras'] = (wr['RAS'] * 10).fillna(50)

# Weight on 0-100 scale (percentile-based)
wr_eval_pre = wr[wr['hit24'].notna() & wr['draft_year'].between(2015, 2024)].copy()
wt_valid = wr_eval_pre['wt'].dropna()
wt_p1, wt_p99 = wt_valid.quantile(0.01), wt_valid.quantile(0.99)
wr['s_weight'] = ((wr['wt'] - wt_p1) / (wt_p99 - wt_p1) * 100).clip(0, 100)
wr['s_weight'] = wr['s_weight'].fillna(50)  # Impute missing with average

# Eval sample
wr_eval = wr[wr['hit24'].notna() & wr['draft_year'].between(2015, 2024)].copy()
print(f"\nEval sample: {len(wr_eval)} WRs")
print(f"  hit24 rate: {wr_eval['hit24'].mean():.3f} ({int(wr_eval['hit24'].sum())}/{len(wr_eval)})")
print(f"  hit12 rate: {wr_eval['hit12'].mean():.3f} ({int(wr_eval['hit12'].sum())}/{len(wr_eval)})")
print(f"  Weight coverage: {wr_eval['wt'].notna().sum()}/{len(wr_eval)}")


# ============================================================================
# BUILD 5 MODELS — Compute SLAP scores
# ============================================================================

models = {
    'A: 4-comp (75/17/4/4)': {
        'weights': {'s_dc': 0.75, 's_bo_rush': 0.17, 's_teammate': 0.04, 's_declare': 0.04},
        'desc': 'DC/BO+Rush/TM/ED'
    },
    'B: +5% RAS (70/17/4/4/5)': {
        'weights': {'s_dc': 0.70, 's_bo_rush': 0.17, 's_teammate': 0.04, 's_declare': 0.04, 's_ras': 0.05},
        'desc': 'DC/BO+Rush/TM/ED/RAS'
    },
    'C: +5% Weight (70/17/4/4/5)': {
        'weights': {'s_dc': 0.70, 's_bo_rush': 0.17, 's_teammate': 0.04, 's_declare': 0.04, 's_weight': 0.05},
        'desc': 'DC/BO+Rush/TM/ED/Weight'
    },
    'D: DC only (100/0/0/0)': {
        'weights': {'s_dc': 1.00},
        'desc': 'DC only'
    },
    'E: Current (65/20/0/0/15)': {
        'weights': {'s_dc': 0.65, 's_breakout': 0.20, 's_ras': 0.15},
        'desc': 'DC/Breakout/RAS (CLAUDE.md)'
    },
}

# Calculate SLAP score for each model
for mname, mcfg in models.items():
    col = f'slap_{mname[:1]}'
    wr_eval[col] = 0
    for comp, wt in mcfg['weights'].items():
        wr_eval[col] = wr_eval[col] + wr_eval[comp] * wt
    # Clip to 0-100
    wr_eval[col] = wr_eval[col].clip(0, 100)

slap_cols = {k: f'slap_{k[:1]}' for k in models}

print(f"\n{'Model':<35} {'Mean SLAP':>10} {'SD':>8} {'Min':>8} {'Max':>8}")
print("-" * 75)
for mname in models:
    col = slap_cols[mname]
    print(f"  {mname:<33} {wr_eval[col].mean():>10.2f} {wr_eval[col].std():>8.2f} "
          f"{wr_eval[col].min():>8.1f} {wr_eval[col].max():>8.1f}")


# ============================================================================
# CALIBRATION HELPER: Convert SLAP → predicted probability via logistic regression
# ============================================================================

def slap_to_proba(slap_scores, y_true):
    """Fit logistic regression on SLAP scores to get calibrated probabilities."""
    X = slap_scores.values.reshape(-1, 1)
    y = y_true.values
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X, y)
    proba = lr.predict_proba(X)[:, 1]
    return proba


# ============================================================================
# TEST 1: BRIER SCORE
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 1: BRIER SCORE (lower is better)")
print("Measures calibration — how well does the model's confidence match reality?")
print("=" * 110)

print(f"\n{'Model':<35} {'Brier(hit24)':>14} {'Brier(hit12)':>14}")
print("-" * 68)

brier_results = {}
for mname in models:
    col = slap_cols[mname]
    # Convert SLAP to probabilities via logistic regression
    proba_24 = slap_to_proba(wr_eval[col], wr_eval['hit24'])
    proba_12 = slap_to_proba(wr_eval[col], wr_eval['hit12'])

    bs_24 = brier_score_loss(wr_eval['hit24'], proba_24)
    bs_12 = brier_score_loss(wr_eval['hit12'], proba_12)

    print(f"  {mname:<33} {bs_24:>14.4f} {bs_12:>14.4f}")
    brier_results[mname] = {'hit24': bs_24, 'hit12': bs_12}

# Naive baseline (always predict mean)
base_rate_24 = wr_eval['hit24'].mean()
base_rate_12 = wr_eval['hit12'].mean()
bs_naive_24 = brier_score_loss(wr_eval['hit24'], np.full(len(wr_eval), base_rate_24))
bs_naive_12 = brier_score_loss(wr_eval['hit12'], np.full(len(wr_eval), base_rate_12))
print(f"  {'(Naive: always predict mean)':<33} {bs_naive_24:>14.4f} {bs_naive_12:>14.4f}")

# Brier Skill Score (improvement over naive)
print(f"\n  Brier Skill Score (% improvement over naive baseline):")
print(f"  {'Model':<35} {'BSS(hit24)':>14} {'BSS(hit12)':>14}")
print("  " + "-" * 65)
for mname in models:
    bss_24 = 1 - brier_results[mname]['hit24'] / bs_naive_24
    bss_12 = 1 - brier_results[mname]['hit12'] / bs_naive_12
    print(f"  {mname:<33} {bss_24:>13.1%} {bss_12:>13.1%}")


# ============================================================================
# TEST 2: AUC-ROC
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 2: AUC-ROC (higher is better, 0.5 = random)")
print("Measures ability to rank hits above misses")
print("=" * 110)

print(f"\n{'Model':<35} {'AUC(hit24)':>12} {'AUC(hit12)':>12}")
print("-" * 64)

for mname in models:
    col = slap_cols[mname]
    auc_24 = roc_auc_score(wr_eval['hit24'], wr_eval[col])
    auc_12 = roc_auc_score(wr_eval['hit12'], wr_eval[col])
    print(f"  {mname:<33} {auc_24:>12.4f} {auc_12:>12.4f}")


# ============================================================================
# TEST 3: PRECISION AT TOP DECILE
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 3: PRECISION AT TOP DECILE (higher is better)")
print("When the model is most confident, how often is it right?")
print("=" * 110)

n_top = max(1, len(wr_eval) // 10)  # top 10% = ~33 players
print(f"\nTop decile = top {n_top} players by SLAP score")

# Also check top 20 and top 50
for cutoff_label, n_cut in [("Top 10% (~33)", n_top), ("Top 20 players", 20), ("Top 50 players", 50)]:
    print(f"\n  {cutoff_label}:")
    print(f"  {'Model':<35} {'Hit24':>6} {'Rate':>8} {'Hit12':>6} {'Rate':>8} {'Avg 3yr PPG':>14} {'N(3yr)':>8}")
    print("  " + "-" * 90)

    for mname in models:
        col = slap_cols[mname]
        top = wr_eval.nlargest(n_cut, col)
        h24 = int(top['hit24'].sum())
        h12 = int(top['hit12'].sum())
        r24 = h24 / n_cut * 100
        r12 = h12 / n_cut * 100
        top_3yr = top[top['first_3yr_ppg'].notna()]
        avg_ppg = top_3yr['first_3yr_ppg'].mean() if len(top_3yr) > 0 else np.nan
        ppg_str = f"{avg_ppg:.2f}" if not np.isnan(avg_ppg) else "N/A"
        print(f"  {mname:<33} {h24:>6} {r24:>7.1f}% {h12:>6} {r12:>7.1f}% {ppg_str:>14} {len(top_3yr):>8}")

    # Overall base rates for comparison
    print(f"  {'(Overall base rate)':<33} "
          f"{int(wr_eval['hit24'].sum()):>6} {wr_eval['hit24'].mean()*100:>7.1f}% "
          f"{int(wr_eval['hit12'].sum()):>6} {wr_eval['hit12'].mean()*100:>7.1f}% "
          f"{wr_eval[wr_eval['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean():>14.2f} "
          f"{wr_eval['first_3yr_ppg'].notna().sum():>8}")


# ============================================================================
# TEST 4: LOG LOSS
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 4: LOG LOSS (lower is better)")
print("Heavily penalizes confident wrong predictions")
print("=" * 110)

print(f"\n{'Model':<35} {'LogLoss(hit24)':>16} {'LogLoss(hit12)':>16}")
print("-" * 72)

for mname in models:
    col = slap_cols[mname]
    proba_24 = slap_to_proba(wr_eval[col], wr_eval['hit24'])
    proba_12 = slap_to_proba(wr_eval[col], wr_eval['hit12'])

    ll_24 = log_loss(wr_eval['hit24'], proba_24)
    ll_12 = log_loss(wr_eval['hit12'], proba_12)
    print(f"  {mname:<33} {ll_24:>16.4f} {ll_12:>16.4f}")

# Naive baseline
ll_naive_24 = log_loss(wr_eval['hit24'], np.full(len(wr_eval), base_rate_24))
ll_naive_12 = log_loss(wr_eval['hit12'], np.full(len(wr_eval), base_rate_12))
print(f"  {'(Naive: always predict mean)':<33} {ll_naive_24:>16.4f} {ll_naive_12:>16.4f}")


# ============================================================================
# TEST 5: BOOTSTRAP STABILITY (1,000 resamples)
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 5: BOOTSTRAP STABILITY — 1,000 resamples")
print("Does RAS or raw weight get positive weight in a 5-component model?")
print("=" * 110)

outcome_weights = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}
all_outcomes = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']

# Base 4 component columns
base_4 = ['s_dc', 's_bo_rush', 's_teammate', 's_declare']

n_bootstrap = 1000
np.random.seed(42)

# Store results
ras_weights = []
wt_weights = []
ras_positive = 0
wt_positive = 0

# Also track: which model wins (A vs B, A vs C) on priority-weighted r
a_vs_b_wins = 0  # How many times Model A beats Model B
a_vs_c_wins = 0  # How many times Model A beats Model C

print(f"\nRunning {n_bootstrap} bootstrap resamples...")
print("(This tests whether RAS and weight consistently earn positive weight)")

for boot_i in range(n_bootstrap):
    if (boot_i + 1) % 200 == 0:
        print(f"  ... resample {boot_i + 1}/{n_bootstrap}")

    # Resample with replacement
    boot_idx = np.random.choice(len(wr_eval), size=len(wr_eval), replace=True)
    boot = wr_eval.iloc[boot_idx].copy()

    # --- RAS 5-component optimization ---
    cols_ras = base_4 + ['s_ras']
    ras_outcome_data = {}
    for out in all_outcomes:
        valid = boot[cols_ras + [out]].dropna(subset=[out]).copy()
        if len(valid) >= 20:
            ras_outcome_data[out] = {'X': valid[cols_ras].values, 'y': valid[out].values}

    def neg_pri_ras(weights):
        total = 0
        for out, w in outcome_weights.items():
            if out not in ras_outcome_data:
                continue
            X = ras_outcome_data[out]['X']
            y = ras_outcome_data[out]['y']
            s = X @ weights
            if np.std(s) > 1e-10:
                r = np.corrcoef(s, y)[0, 1]
                total += w * r
        return -total

    bounds_ras = [(0.40, 0.90), (0.05, 0.35), (0.00, 0.15), (0.00, 0.15), (0.00, 0.15)]
    constraints_ras = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]

    starts = [
        [0.75, 0.17, 0.04, 0.04, 0.00],
        [0.70, 0.17, 0.04, 0.04, 0.05],
        [0.65, 0.20, 0.05, 0.05, 0.05],
    ]

    best_ras_r = -999
    best_ras_w = None
    for x0 in starts:
        try:
            res = minimize(neg_pri_ras, x0, method='SLSQP', bounds=bounds_ras,
                           constraints=constraints_ras, options={'maxiter': 500})
            if -res.fun > best_ras_r:
                best_ras_r = -res.fun
                best_ras_w = res.x
        except:
            pass

    if best_ras_w is not None:
        ras_wt_pct = best_ras_w[4]
        ras_weights.append(ras_wt_pct)
        if ras_wt_pct > 0.005:
            ras_positive += 1

    # --- Weight 5-component optimization ---
    cols_wt = base_4 + ['s_weight']
    wt_outcome_data = {}
    for out in all_outcomes:
        valid = boot[cols_wt + [out]].dropna(subset=[out]).copy()
        if len(valid) >= 20:
            wt_outcome_data[out] = {'X': valid[cols_wt].values, 'y': valid[out].values}

    def neg_pri_wt(weights):
        total = 0
        for out, w in outcome_weights.items():
            if out not in wt_outcome_data:
                continue
            X = wt_outcome_data[out]['X']
            y = wt_outcome_data[out]['y']
            s = X @ weights
            if np.std(s) > 1e-10:
                r = np.corrcoef(s, y)[0, 1]
                total += w * r
        return -total

    bounds_wt = [(0.40, 0.90), (0.05, 0.35), (0.00, 0.15), (0.00, 0.15), (0.00, 0.15)]
    constraints_wt = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]

    best_wt_r = -999
    best_wt_w = None
    for x0 in starts:
        try:
            res = minimize(neg_pri_wt, x0, method='SLSQP', bounds=bounds_wt,
                           constraints=constraints_wt, options={'maxiter': 500})
            if -res.fun > best_wt_r:
                best_wt_r = -res.fun
                best_wt_w = res.x
        except:
            pass

    if best_wt_w is not None:
        wt_wt_pct = best_wt_w[4]
        wt_weights.append(wt_wt_pct)
        if wt_wt_pct > 0.005:
            wt_positive += 1

    # --- Model A vs B vs C comparison ---
    # Compute priority-weighted r for Models A and B on this bootstrap sample
    for out in all_outcomes:
        if out not in ras_outcome_data:
            continue
    # Model A (4-comp with 75/17/4/4) r on this sample
    w_a = np.array([0.75, 0.17, 0.04, 0.04])
    w_b = np.array([0.70, 0.17, 0.04, 0.04, 0.05])
    w_c = np.array([0.70, 0.17, 0.04, 0.04, 0.05])

    r_a = 0
    r_b = 0
    r_c = 0
    for out, ow in outcome_weights.items():
        # Model A
        valid_a = boot[base_4 + [out]].dropna(subset=[out])
        if len(valid_a) >= 20:
            s_a = valid_a[base_4].values @ w_a
            if np.std(s_a) > 1e-10:
                r_a += ow * np.corrcoef(s_a, valid_a[out].values)[0, 1]

        # Model B (+ RAS)
        valid_b = boot[cols_ras + [out]].dropna(subset=[out])
        if len(valid_b) >= 20:
            s_b = valid_b[cols_ras].values @ w_b
            if np.std(s_b) > 1e-10:
                r_b += ow * np.corrcoef(s_b, valid_b[out].values)[0, 1]

        # Model C (+ Weight)
        valid_c = boot[cols_wt + [out]].dropna(subset=[out])
        if len(valid_c) >= 20:
            s_c = valid_c[cols_wt].values @ w_c
            if np.std(s_c) > 1e-10:
                r_c += ow * np.corrcoef(s_c, valid_c[out].values)[0, 1]

    if r_a >= r_b:
        a_vs_b_wins += 1
    if r_a >= r_c:
        a_vs_c_wins += 1


ras_weights = np.array(ras_weights)
wt_weights = np.array(wt_weights)

print(f"\n{'─' * 80}")
print("BOOTSTRAP RESULTS: RAS as 5th component")
print(f"{'─' * 80}")
print(f"  Resamples where RAS gets >0% weight: {ras_positive}/{n_bootstrap} ({ras_positive/n_bootstrap*100:.1f}%)")
print(f"  RAS weight distribution:")
print(f"    Mean:   {ras_weights.mean():.1%}")
print(f"    Median: {np.median(ras_weights):.1%}")
print(f"    P25:    {np.percentile(ras_weights, 25):.1%}")
print(f"    P75:    {np.percentile(ras_weights, 75):.1%}")
print(f"    Max:    {ras_weights.max():.1%}")
print(f"    At 0%:  {(ras_weights < 0.005).sum()}/{n_bootstrap} ({(ras_weights < 0.005).mean()*100:.1f}%)")

# Histogram
bins_hist = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 1.0]
labels_hist = ['0%', '0-1%', '1-2%', '2-3%', '3-5%', '5-8%', '8-10%', '10-15%', '15%+']
print(f"\n  RAS weight distribution:")
print(f"  {'Bucket':<12} {'Count':>8} {'Pct':>8}")
print("  " + "-" * 30)
for i in range(len(bins_hist) - 1):
    lo, hi = bins_hist[i], bins_hist[i + 1]
    if i == 0:
        count = (ras_weights < hi).sum()
    else:
        count = ((ras_weights >= lo) & (ras_weights < hi)).sum()
    print(f"  {labels_hist[i]:<12} {count:>8} {count/n_bootstrap*100:>7.1f}%")

print(f"\n{'─' * 80}")
print("BOOTSTRAP RESULTS: Raw Weight as 5th component")
print(f"{'─' * 80}")
print(f"  Resamples where Weight gets >0% weight: {wt_positive}/{n_bootstrap} ({wt_positive/n_bootstrap*100:.1f}%)")
print(f"  Weight weight distribution:")
print(f"    Mean:   {wt_weights.mean():.1%}")
print(f"    Median: {np.median(wt_weights):.1%}")
print(f"    P25:    {np.percentile(wt_weights, 25):.1%}")
print(f"    P75:    {np.percentile(wt_weights, 75):.1%}")
print(f"    Max:    {wt_weights.max():.1%}")
print(f"    At 0%:  {(wt_weights < 0.005).sum()}/{n_bootstrap} ({(wt_weights < 0.005).mean()*100:.1f}%)")

print(f"\n  Weight weight distribution:")
print(f"  {'Bucket':<12} {'Count':>8} {'Pct':>8}")
print("  " + "-" * 30)
for i in range(len(bins_hist) - 1):
    lo, hi = bins_hist[i], bins_hist[i + 1]
    if i == 0:
        count = (wt_weights < hi).sum()
    else:
        count = ((wt_weights >= lo) & (wt_weights < hi)).sum()
    print(f"  {labels_hist[i]:<12} {count:>8} {count/n_bootstrap*100:>7.1f}%")

print(f"\n{'─' * 80}")
print("BOOTSTRAP HEAD-TO-HEAD: Model A (4-comp) vs Models B & C")
print(f"{'─' * 80}")
print(f"  Model A beats Model B (+ RAS): {a_vs_b_wins}/{n_bootstrap} ({a_vs_b_wins/n_bootstrap*100:.1f}%)")
print(f"  Model A beats Model C (+ Wt):  {a_vs_c_wins}/{n_bootstrap} ({a_vs_c_wins/n_bootstrap*100:.1f}%)")


# ============================================================================
# TEST 6: CALIBRATION PLOT (by SLAP score decile)
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TEST 6: CALIBRATION BY SLAP SCORE DECILE")
print("For each decile of SLAP scores: predicted hit rate vs actual hit rate")
print("=" * 110)

for mname in ['A: 4-comp (75/17/4/4)', 'B: +5% RAS (70/17/4/4/5)']:
    col = slap_cols[mname]
    proba_24 = slap_to_proba(wr_eval[col], wr_eval['hit24'])
    wr_eval[f'proba_{mname[:1]}'] = proba_24

    # Create deciles based on SLAP score
    wr_eval[f'decile_{mname[:1]}'] = pd.qcut(wr_eval[col], 10, labels=False, duplicates='drop')

    print(f"\n  {mname}:")
    print(f"  {'Decile':<8} {'SLAP Range':<18} {'N':>4} {'Pred Hit%':>11} {'Actual Hit%':>13} {'Actual Hits':>13} {'Avg 3yr PPG':>14}")
    print("  " + "-" * 95)

    for dec in sorted(wr_eval[f'decile_{mname[:1]}'].unique()):
        sub = wr_eval[wr_eval[f'decile_{mname[:1]}'] == dec]
        slap_lo = sub[col].min()
        slap_hi = sub[col].max()
        n = len(sub)
        pred_hit = sub[f'proba_{mname[:1]}'].mean() * 100
        actual_hit = sub['hit24'].mean() * 100
        actual_hits = int(sub['hit24'].sum())
        sub_3yr = sub[sub['first_3yr_ppg'].notna()]
        avg_ppg = sub_3yr['first_3yr_ppg'].mean() if len(sub_3yr) > 0 else np.nan
        ppg_str = f"{avg_ppg:.2f}" if not np.isnan(avg_ppg) else "N/A"
        print(f"  {dec:>6}   {slap_lo:>5.1f} - {slap_hi:>5.1f}     {n:>4} {pred_hit:>10.1f}% {actual_hit:>12.1f}% "
              f"{actual_hits:>11}     {ppg_str:>10}")

    # Calibration error: mean absolute difference between predicted and actual
    cal_error = 0
    n_dec = 0
    for dec in sorted(wr_eval[f'decile_{mname[:1]}'].unique()):
        sub = wr_eval[wr_eval[f'decile_{mname[:1]}'] == dec]
        pred = sub[f'proba_{mname[:1]}'].mean()
        actual = sub['hit24'].mean()
        cal_error += abs(pred - actual)
        n_dec += 1
    cal_error /= n_dec
    print(f"\n  Mean Calibration Error: {cal_error*100:.2f}% (avg |predicted - actual| across deciles)")


# Also show Models D and E for comparison
for mname in ['D: DC only (100/0/0/0)', 'E: Current (65/20/0/0/15)']:
    col = slap_cols[mname]
    proba_24 = slap_to_proba(wr_eval[col], wr_eval['hit24'])
    wr_eval[f'proba_{mname[:1]}'] = proba_24
    wr_eval[f'decile_{mname[:1]}'] = pd.qcut(wr_eval[col], 10, labels=False, duplicates='drop')

    print(f"\n  {mname}:")
    print(f"  {'Decile':<8} {'SLAP Range':<18} {'N':>4} {'Pred Hit%':>11} {'Actual Hit%':>13} {'Actual Hits':>13} {'Avg 3yr PPG':>14}")
    print("  " + "-" * 95)

    for dec in sorted(wr_eval[f'decile_{mname[:1]}'].unique()):
        sub = wr_eval[wr_eval[f'decile_{mname[:1]}'] == dec]
        slap_lo = sub[col].min()
        slap_hi = sub[col].max()
        n = len(sub)
        pred_hit = sub[f'proba_{mname[:1]}'].mean() * 100
        actual_hit = sub['hit24'].mean() * 100
        actual_hits = int(sub['hit24'].sum())
        sub_3yr = sub[sub['first_3yr_ppg'].notna()]
        avg_ppg = sub_3yr['first_3yr_ppg'].mean() if len(sub_3yr) > 0 else np.nan
        ppg_str = f"{avg_ppg:.2f}" if not np.isnan(avg_ppg) else "N/A"
        print(f"  {dec:>6}   {slap_lo:>5.1f} - {slap_hi:>5.1f}     {n:>4} {pred_hit:>10.1f}% {actual_hit:>12.1f}% "
              f"{actual_hits:>11}     {ppg_str:>10}")

    cal_error = 0
    n_dec = 0
    for dec in sorted(wr_eval[f'decile_{mname[:1]}'].unique()):
        sub = wr_eval[wr_eval[f'decile_{mname[:1]}'] == dec]
        pred = sub[f'proba_{mname[:1]}'].mean()
        actual = sub['hit24'].mean()
        cal_error += abs(pred - actual)
        n_dec += 1
    cal_error /= n_dec
    print(f"\n  Mean Calibration Error: {cal_error*100:.2f}% (avg |predicted - actual| across deciles)")


# ============================================================================
# GRAND SUMMARY
# ============================================================================

print(f"\n\n{'=' * 110}")
print("GRAND SUMMARY: ALL 6 TESTS")
print("=" * 110)

print(f"\n{'Model':<35} {'Brier24':>9} {'AUC24':>8} {'AUC12':>8} {'LogL24':>9} {'Top10%Hit':>11} {'Top10%PPG':>11}")
print("-" * 95)

for mname in models:
    col = slap_cols[mname]

    # Brier
    proba_24 = slap_to_proba(wr_eval[col], wr_eval['hit24'])
    bs_24 = brier_score_loss(wr_eval['hit24'], proba_24)

    # AUC
    auc_24 = roc_auc_score(wr_eval['hit24'], wr_eval[col])
    auc_12 = roc_auc_score(wr_eval['hit12'], wr_eval[col])

    # Log Loss
    ll_24 = log_loss(wr_eval['hit24'], proba_24)

    # Top decile
    top = wr_eval.nlargest(n_top, col)
    h24_rate = top['hit24'].mean() * 100
    top_3yr = top[top['first_3yr_ppg'].notna()]
    avg_ppg = top_3yr['first_3yr_ppg'].mean() if len(top_3yr) > 0 else np.nan
    ppg_str = f"{avg_ppg:.2f}" if not np.isnan(avg_ppg) else "N/A"

    print(f"  {mname:<33} {bs_24:>9.4f} {auc_24:>8.4f} {auc_12:>8.4f} {ll_24:>9.4f} {h24_rate:>10.1f}% {ppg_str:>11}")

print(f"\n  Bootstrap: RAS gets >0% in {ras_positive/n_bootstrap*100:.1f}% of resamples (mean weight: {ras_weights.mean():.1%})")
print(f"  Bootstrap: Weight gets >0% in {wt_positive/n_bootstrap*100:.1f}% of resamples (mean weight: {wt_weights.mean():.1%})")
print(f"  Bootstrap: Model A beats Model B in {a_vs_b_wins/n_bootstrap*100:.1f}% of resamples")
print(f"  Bootstrap: Model A beats Model C in {a_vs_c_wins/n_bootstrap*100:.1f}% of resamples")

# Direction arrows
print(f"\n  Legend: Brier/LogLoss = LOWER is better | AUC/Hit%/PPG = HIGHER is better")

print(f"\n\n{'=' * 110}")
print("ANALYSIS COMPLETE")
print("=" * 110)
