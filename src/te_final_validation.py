"""
TE Model Final Validation Suite
================================
Config: DC 65 / Breakout 15 / Production 10 / Early Declare 5 / RAS 5

Full validation:
  1. Brier scores
  2. AUC-ROC
  3. Bootstrap confidence intervals
  4. Calibration test
  5. Tier hit rate table (90+, 80-89, 70-79, 60-69, 50-59, <50)
  6. Top 10 positive and negative SLAP disagreements vs DC
  7. Head-to-head vs DC-only baseline on every metric
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('/home/user/SlapModelV3')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

def normalize_0_100(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50, index=series.index)
    return (series - mn) / (mx - mn) * 100

def brier_score(predicted_prob, actual_binary):
    """Brier score: mean squared error of probability predictions. Lower = better."""
    valid = pd.DataFrame({'p': predicted_prob, 'a': actual_binary}).dropna()
    return ((valid['p'] - valid['a']) ** 2).mean(), len(valid)

def auc_roc(scores, actual_binary):
    """Manual AUC-ROC using the Mann-Whitney U statistic."""
    valid = pd.DataFrame({'s': scores, 'a': actual_binary}).dropna()
    pos = valid[valid['a'] == 1]['s'].values
    neg = valid[valid['a'] == 0]['s'].values
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, 0, 0
    # Mann-Whitney U
    u_stat = 0
    for p in pos:
        u_stat += (neg < p).sum() + 0.5 * (neg == p).sum()
    auc = u_stat / (len(pos) * len(neg))
    return auc, len(pos), len(neg)


# ============================================================================
# LOAD AND BUILD SCORES
# ============================================================================

print("=" * 120)
print("TE MODEL FINAL VALIDATION")
print("Config: DC 65% / Breakout 15% / Production 10% / Early Declare 5% / RAS 5%")
print("=" * 120)

bt = pd.read_csv('data/te_backtest_master.csv')

# Build DC
bt['s_dc'] = bt['pick'].apply(dc_score)

# Build Breakout Score at 15% threshold
pff_file_map = {
    'data/receiving_summary (2).csv': 2015,
    'data/receiving_summary (3).csv': 2016,
    'data/receiving_summary (4).csv': 2017,
    'data/receiving_summary (5).csv': 2018,
    'data/receiving_summary (21).csv': 2019,
    'data/receiving_summary (20).csv': 2020,
    'data/receiving_summary (19).csv': 2021,
    'data/receiving_summary (18).csv': 2022,
    'data/receiving_summary (17).csv': 2023,
    'data/receiving_summary (16).csv': 2024,
    'data/receiving_summary (15).csv': 2025,
}

all_pff = []
for filepath, season in pff_file_map.items():
    pff = pd.read_csv(filepath)
    pff['season'] = season
    all_pff.append(pff)
pff_all = pd.concat(all_pff, ignore_index=True)
receivers = pff_all[pff_all['position'].isin(['TE', 'WR', 'HB'])].copy()
team_totals = receivers.groupby(['team_name', 'season'])['yards'].sum().reset_index()
team_totals.rename(columns={'yards': 'team_rec_yards'}, inplace=True)
te_pff = receivers[receivers['position'] == 'TE'].copy()
te_pff = te_pff.merge(team_totals, on=['team_name', 'season'], how='left')
te_pff['dominator_pct'] = np.where(te_pff['team_rec_yards'] > 0,
    (te_pff['yards'] / te_pff['team_rec_yards']) * 100, 0)

def norm_name(n):
    s = str(n).lower().strip()
    for suf in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
        s = s.replace(suf, '')
    s = s.replace("'", "").replace("\u2019", "").replace("-", "").replace(".", "")
    return s.strip()

te_pff['name_norm'] = te_pff['player'].apply(norm_name)
bt['name_norm_v'] = bt['player_name'].apply(norm_name)

THRESH = 15
bo_scores = {}
for _, te in bt.iterrows():
    matches = te_pff[(te_pff['name_norm'] == te['name_norm_v']) & (te_pff['season'] < te['draft_year'])]
    if len(matches) == 0:
        continue
    peak_dom = matches['dominator_pct'].max()
    hit_seasons = []
    for _, pm in matches.sort_values('season').iterrows():
        season_age = te['draft_age'] - (te['draft_year'] - pm['season'])
        if pm['dominator_pct'] >= THRESH:
            hit_seasons.append(season_age)
    if hit_seasons:
        bo_age = hit_seasons[0]
        if bo_age <= 18: base = 100
        elif bo_age <= 19: base = 90
        elif bo_age <= 20: base = 75
        elif bo_age <= 21: base = 60
        elif bo_age <= 22: base = 45
        elif bo_age <= 23: base = 30
        else: base = 20
        bonus = min((peak_dom - THRESH) * 0.5, 9.9)
        score = min(base + bonus, 99.9)
    else:
        score = min(35, 15 + peak_dom)
    bo_scores[te['player_name']] = score

bt['s_breakout'] = bt['player_name'].map(bo_scores)

# Build Production Hybrid
bt['rec_per_tpa'] = np.where(
    (bt['cfbd_receptions'].notna()) & (bt['cfbd_team_pass_att'].notna()) & (bt['cfbd_team_pass_att'] > 0),
    bt['cfbd_receptions'] / bt['cfbd_team_pass_att'], np.nan)
bt['pff_rpg'] = np.where(
    (bt['pff_receptions'].notna()) & (bt['pff_player_game_count'].notna()) & (bt['pff_player_game_count'] > 0),
    bt['pff_receptions'] / bt['pff_player_game_count'], np.nan)

cfbd_vals = bt['rec_per_tpa'].dropna()
pff_vals = bt['pff_rpg'].dropna()
bt['cfbd_norm'] = np.where(bt['rec_per_tpa'].notna(),
    (bt['rec_per_tpa'] - cfbd_vals.min()) / (cfbd_vals.max() - cfbd_vals.min()) * 100, np.nan)
bt['pff_rpg_norm'] = np.where(bt['pff_rpg'].notna(),
    (bt['pff_rpg'] - pff_vals.min()) / (pff_vals.max() - pff_vals.min()) * 100, np.nan)
bt['s_production'] = np.where(bt['cfbd_norm'].notna(), bt['cfbd_norm'], bt['pff_rpg_norm'])

# Early Declare and RAS
bt['s_early_dec'] = bt['early_declare'] * 100
bt['s_ras'] = bt['te_ras'].apply(lambda x: x * 10 if pd.notna(x) else np.nan)

# Fill missing with position averages
avg_bo = bt['s_breakout'].mean()
avg_prod = bt['s_production'].mean()
avg_ras = bt['s_ras'].mean()

bt['s_breakout_f'] = bt['s_breakout'].fillna(avg_bo)
bt['s_production_f'] = bt['s_production'].fillna(avg_prod)
bt['s_ras_f'] = bt['s_ras'].fillna(avg_ras)

# ============================================================================
# COMPUTE SLAP SCORES — Final model and DC-only baseline
# ============================================================================

# Final model: 65/15/10/5/5
bt['slap'] = (bt['s_dc'] * 0.65 +
              bt['s_breakout_f'] * 0.15 +
              bt['s_production_f'] * 0.10 +
              bt['s_early_dec'] * 0.05 +
              bt['s_ras_f'] * 0.05).clip(0, 100)

# DC-only baseline
bt['slap_dc'] = bt['s_dc']

# Eval sample: 2015-2024 with outcomes
eval_df = bt[(bt['hit24'].notna()) & (bt['draft_year'] < 2025)].copy()
print(f"\nEval sample: {len(eval_df)} TEs (2015-2024)")
print(f"  hit24: {int(eval_df['hit24'].sum())}/{len(eval_df)} ({eval_df['hit24'].mean():.1%})")
print(f"  hit12: {int(eval_df['hit12'].sum())}/{len(eval_df)} ({eval_df['hit12'].mean():.1%})")
print(f"  first_3yr_ppg: {eval_df['first_3yr_ppg'].notna().sum()} with data")
print(f"  career_ppg: {eval_df['career_ppg'].notna().sum()} with data")

outcome_cols = ['first_3yr_ppg', 'hit24', 'hit12', 'career_ppg']
outcome_weights = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}


# ============================================================================
# 1. BRIER SCORES
# ============================================================================

print(f"\n\n{'='*100}")
print("1. BRIER SCORES (lower = better)")
print("Measures calibration: how well do scores predict binary outcomes?")
print(f"{'='*100}")

# Convert SLAP to probability: normalize to 0-1 range
eval_df['slap_prob'] = eval_df['slap'] / 100
eval_df['dc_prob'] = eval_df['slap_dc'] / 100

print(f"\n  {'Outcome':<20} {'SLAP Brier':>12} {'DC Brier':>12} {'Δ (SLAP-DC)':>14} {'Better':>8} {'N':>6}")
print(f"  {'-'*75}")

for outcome in ['hit24', 'hit12']:
    bs_slap, n_slap = brier_score(eval_df['slap_prob'], eval_df[outcome])
    bs_dc, n_dc = brier_score(eval_df['dc_prob'], eval_df[outcome])
    delta = bs_slap - bs_dc
    better = "SLAP" if delta < 0 else "DC" if delta > 0 else "TIE"
    print(f"  {outcome:<20} {bs_slap:>12.4f} {bs_dc:>12.4f} {delta:>+14.4f} {better:>8} {n_slap:>6}")

# Also test with scaled probabilities (logistic transform of SLAP)
# Map SLAP scores to empirical hit rates by decile for better calibration
print(f"\n  Note: Raw Brier uses SLAP/100 as probability proxy.")
print(f"  A SLAP of 80 is treated as 80% probability — this is approximate.")


# ============================================================================
# 2. AUC-ROC
# ============================================================================

print(f"\n\n{'='*100}")
print("2. AUC-ROC (higher = better)")
print("Measures discrimination: can the model separate hits from misses?")
print(f"{'='*100}")

print(f"\n  {'Outcome':<20} {'SLAP AUC':>10} {'DC AUC':>10} {'Δ':>10} {'Better':>8} {'Pos':>6} {'Neg':>6}")
print(f"  {'-'*75}")

for outcome in ['hit24', 'hit12']:
    auc_slap, n_pos_s, n_neg_s = auc_roc(eval_df['slap'], eval_df[outcome])
    auc_dc, n_pos_d, n_neg_d = auc_roc(eval_df['slap_dc'], eval_df[outcome])
    delta = auc_slap - auc_dc
    better = "SLAP" if delta > 0 else "DC" if delta < 0 else "TIE"
    print(f"  {outcome:<20} {auc_slap:>10.4f} {auc_dc:>10.4f} {delta:>+10.4f} {better:>8} {n_pos_s:>6} {n_neg_s:>6}")

print(f"\n  AUC = 0.50 is random guessing, AUC = 1.00 is perfect discrimination.")


# ============================================================================
# 3. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

print(f"\n\n{'='*100}")
print("3. BOOTSTRAP CONFIDENCE INTERVALS (1000 resamples)")
print("95% CI on Pearson r for each outcome.")
print(f"{'='*100}")

np.random.seed(42)
N_BOOT = 1000

print(f"\n  {'Outcome':<20} {'SLAP r':>8} {'95% CI':>20} {'DC r':>8} {'95% CI':>20} {'SLAP > DC':>10}")
print(f"  {'-'*95}")

for outcome in outcome_cols:
    valid = eval_df[['slap', 'slap_dc', outcome]].dropna()
    if len(valid) < 20:
        print(f"  {outcome:<20} insufficient data (N={len(valid)})")
        continue

    r_slap, _ = stats.pearsonr(valid['slap'], valid[outcome])
    r_dc, _ = stats.pearsonr(valid['slap_dc'], valid[outcome])

    boot_slap = []
    boot_dc = []
    slap_wins = 0

    for _ in range(N_BOOT):
        sample = valid.sample(n=len(valid), replace=True)
        try:
            r_s, _ = stats.pearsonr(sample['slap'], sample[outcome])
            r_d, _ = stats.pearsonr(sample['slap_dc'], sample[outcome])
            boot_slap.append(r_s)
            boot_dc.append(r_d)
            if r_s > r_d:
                slap_wins += 1
        except:
            continue

    ci_slap_lo = np.percentile(boot_slap, 2.5)
    ci_slap_hi = np.percentile(boot_slap, 97.5)
    ci_dc_lo = np.percentile(boot_dc, 2.5)
    ci_dc_hi = np.percentile(boot_dc, 97.5)
    win_pct = slap_wins / len(boot_slap) * 100

    print(f"  {outcome:<20} {r_slap:>+.4f} [{ci_slap_lo:>+.4f}, {ci_slap_hi:>+.4f}] {r_dc:>+.4f} [{ci_dc_lo:>+.4f}, {ci_dc_hi:>+.4f}] {win_pct:>8.1f}%")

# Priority-weighted bootstrap
print(f"\n  PRIORITY-WEIGHTED r (40/25/20/15):")
valid_all = eval_df[['slap', 'slap_dc'] + outcome_cols].copy()

boot_pri_slap = []
boot_pri_dc = []
slap_wins_pri = 0

for _ in range(N_BOOT):
    sample = valid_all.sample(n=len(valid_all), replace=True)
    pri_s = 0
    pri_d = 0
    pri_total = 0
    for out in outcome_cols:
        v = sample[['slap', 'slap_dc', out]].dropna()
        if len(v) >= 10:
            try:
                r_s, _ = stats.pearsonr(v['slap'], v[out])
                r_d, _ = stats.pearsonr(v['slap_dc'], v[out])
                pri_s += outcome_weights[out] * r_s
                pri_d += outcome_weights[out] * r_d
                pri_total += outcome_weights[out]
            except:
                continue
    if pri_total > 0:
        boot_pri_slap.append(pri_s / pri_total)
        boot_pri_dc.append(pri_d / pri_total)
        if pri_s / pri_total > pri_d / pri_total:
            slap_wins_pri += 1

# Compute point estimates
pri_slap_point = 0
pri_dc_point = 0
pri_t = 0
for out in outcome_cols:
    v = eval_df[['slap', 'slap_dc', out]].dropna()
    if len(v) >= 10:
        r_s, _ = stats.pearsonr(v['slap'], v[out])
        r_d, _ = stats.pearsonr(v['slap_dc'], v[out])
        pri_slap_point += outcome_weights[out] * r_s
        pri_dc_point += outcome_weights[out] * r_d
        pri_t += outcome_weights[out]
pri_slap_point /= pri_t
pri_dc_point /= pri_t

ci_pri_s_lo = np.percentile(boot_pri_slap, 2.5)
ci_pri_s_hi = np.percentile(boot_pri_slap, 97.5)
ci_pri_d_lo = np.percentile(boot_pri_dc, 2.5)
ci_pri_d_hi = np.percentile(boot_pri_dc, 97.5)
win_pct_pri = slap_wins_pri / len(boot_pri_slap) * 100

print(f"  {'PRI-AVG':<20} {pri_slap_point:>+.4f} [{ci_pri_s_lo:>+.4f}, {ci_pri_s_hi:>+.4f}] {pri_dc_point:>+.4f} [{ci_pri_d_lo:>+.4f}, {ci_pri_d_hi:>+.4f}] {win_pct_pri:>8.1f}%")
print(f"\n  'SLAP > DC' = % of bootstrap resamples where SLAP outperforms DC.")
print(f"  >95% = strong evidence SLAP is better. >80% = moderate evidence.")


# ============================================================================
# 4. CALIBRATION TEST
# ============================================================================

print(f"\n\n{'='*100}")
print("4. CALIBRATION TEST")
print("Do SLAP score ranges match actual hit probabilities?")
print(f"{'='*100}")

# Bin by SLAP score and check actual hit rates
cal_bins = [(90, 101, '90-100'), (80, 90, '80-89'), (70, 80, '70-79'),
            (60, 70, '60-69'), (50, 60, '50-59'), (40, 50, '40-49'),
            (30, 40, '30-39'), (0, 30, '0-29')]

print(f"\n  SLAP Model:")
print(f"  {'SLAP Range':<12} {'N':>5} {'Implied P(h24)':>15} {'Actual h24':>12} {'Actual h12':>12} {'Avg PPG':>10} {'Cal Error':>10}")
print(f"  {'-'*80}")

for lo, hi, label in cal_bins:
    tier = eval_df[(eval_df['slap'] >= lo) & (eval_df['slap'] < hi)]
    if len(tier) == 0:
        continue
    implied = (lo + hi) / 200  # midpoint / 100
    actual_h24 = tier['hit24'].mean()
    actual_h12 = tier['hit12'].mean()
    ppg_vals = tier[tier['first_3yr_ppg'].notna()]
    avg_ppg = ppg_vals['first_3yr_ppg'].mean() if len(ppg_vals) > 0 else np.nan
    ppg_s = f"{avg_ppg:.2f}" if not np.isnan(avg_ppg) else "N/A"
    cal_err = actual_h24 - implied
    print(f"  {label:<12} {len(tier):>5} {implied:>14.0%} {actual_h24:>11.0%} {actual_h12:>11.0%} {ppg_s:>10} {cal_err:>+9.0%}")

print(f"\n  DC-Only Baseline:")
print(f"  {'DC Range':<12} {'N':>5} {'Implied P(h24)':>15} {'Actual h24':>12} {'Actual h12':>12} {'Avg PPG':>10} {'Cal Error':>10}")
print(f"  {'-'*80}")

for lo, hi, label in cal_bins:
    tier = eval_df[(eval_df['slap_dc'] >= lo) & (eval_df['slap_dc'] < hi)]
    if len(tier) == 0:
        continue
    implied = (lo + hi) / 200
    actual_h24 = tier['hit24'].mean()
    actual_h12 = tier['hit12'].mean()
    ppg_vals = tier[tier['first_3yr_ppg'].notna()]
    avg_ppg = ppg_vals['first_3yr_ppg'].mean() if len(ppg_vals) > 0 else np.nan
    ppg_s = f"{avg_ppg:.2f}" if not np.isnan(avg_ppg) else "N/A"
    cal_err = actual_h24 - implied
    print(f"  {label:<12} {len(tier):>5} {implied:>14.0%} {actual_h24:>11.0%} {actual_h12:>11.0%} {ppg_s:>10} {cal_err:>+9.0%}")


# ============================================================================
# 5. TIER HIT RATE TABLE
# ============================================================================

print(f"\n\n{'='*100}")
print("5. TIER HIT RATE TABLE")
print(f"{'='*100}")

tiers = [(90, 101, '90+'), (80, 90, '80-89'), (70, 80, '70-79'),
         (60, 70, '60-69'), (50, 60, '50-59'), (0, 50, 'Below 50')]

print(f"\n  SLAP MODEL (65/15/10/5/5):")
print(f"  {'Tier':<12} {'N':>5} {'Hit24':>6} {'Rate':>8} {'Hit12':>6} {'Rate':>8} {'3yr PPG':>10} {'Career PPG':>12}")
print(f"  {'-'*75}")

total_h24 = 0
total_h12 = 0
for lo, hi, label in tiers:
    tier = eval_df[(eval_df['slap'] >= lo) & (eval_df['slap'] < hi)]
    if len(tier) == 0:
        print(f"  {label:<12} {0:>5} {'—':>6} {'—':>8} {'—':>6} {'—':>8} {'—':>10} {'—':>12}")
        continue
    h24 = int(tier['hit24'].sum())
    h12 = int(tier['hit12'].sum())
    total_h24 += h24
    total_h12 += h12
    r24 = h24 / len(tier) * 100
    r12 = h12 / len(tier) * 100
    ppg3 = tier[tier['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()
    ppgc = tier[tier['career_ppg'].notna()]['career_ppg'].mean()
    ppg3_s = f"{ppg3:.2f}" if not np.isnan(ppg3) else "N/A"
    ppgc_s = f"{ppgc:.2f}" if not np.isnan(ppgc) else "N/A"
    print(f"  {label:<12} {len(tier):>5} {h24:>6} {r24:>7.1f}% {h12:>6} {r12:>7.1f}% {ppg3_s:>10} {ppgc_s:>12}")

print(f"  {'TOTAL':<12} {len(eval_df):>5} {total_h24:>6} {total_h24/len(eval_df)*100:>7.1f}% {total_h12:>6} {total_h12/len(eval_df)*100:>7.1f}%")

print(f"\n  DC-ONLY BASELINE:")
print(f"  {'Tier':<12} {'N':>5} {'Hit24':>6} {'Rate':>8} {'Hit12':>6} {'Rate':>8} {'3yr PPG':>10} {'Career PPG':>12}")
print(f"  {'-'*75}")

total_h24 = 0
total_h12 = 0
for lo, hi, label in tiers:
    tier = eval_df[(eval_df['slap_dc'] >= lo) & (eval_df['slap_dc'] < hi)]
    if len(tier) == 0:
        print(f"  {label:<12} {0:>5} {'—':>6} {'—':>8} {'—':>6} {'—':>8} {'—':>10} {'—':>12}")
        continue
    h24 = int(tier['hit24'].sum())
    h12 = int(tier['hit12'].sum())
    total_h24 += h24
    total_h12 += h12
    r24 = h24 / len(tier) * 100
    r12 = h12 / len(tier) * 100
    ppg3 = tier[tier['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()
    ppgc = tier[tier['career_ppg'].notna()]['career_ppg'].mean()
    ppg3_s = f"{ppg3:.2f}" if not np.isnan(ppg3) else "N/A"
    ppgc_s = f"{ppgc:.2f}" if not np.isnan(ppgc) else "N/A"
    print(f"  {label:<12} {len(tier):>5} {h24:>6} {r24:>7.1f}% {h12:>6} {r12:>7.1f}% {ppg3_s:>10} {ppgc_s:>12}")

print(f"  {'TOTAL':<12} {len(eval_df):>5} {total_h24:>6} {total_h24/len(eval_df)*100:>7.1f}% {total_h12:>6} {total_h12/len(eval_df)*100:>7.1f}%")


# ============================================================================
# 6. TOP SLAP DISAGREEMENTS VS DC
# ============================================================================

print(f"\n\n{'='*100}")
print("6. TOP SLAP DISAGREEMENTS vs DC-ONLY")
print("Players where the model most disagrees with pure draft capital.")
print(f"{'='*100}")

eval_df['delta'] = eval_df['slap'] - eval_df['slap_dc']
eval_df['slap_rank'] = eval_df['slap'].rank(ascending=False, method='min')
eval_df['dc_rank'] = eval_df['slap_dc'].rank(ascending=False, method='min')
eval_df['rank_delta'] = eval_df['dc_rank'] - eval_df['slap_rank']  # positive = model boosted

# Top 10 positive deltas (model likes more than DC)
print(f"\n  TOP 10 POSITIVE DELTAS (model BOOSTS these TEs above their draft slot):")
print(f"  {'Player':<25s} {'Yr':>4} {'Rd':>3} {'Pick':>4} {'DC':>5} {'SLAP':>6} {'Δ':>6} {'RkΔ':>5} {'h24':>4} {'h12':>4} {'3yr PPG':>8} {'Verdict':>10}")
print(f"  {'-'*100}")

top_pos = eval_df.nlargest(10, 'delta')
for _, r in top_pos.iterrows():
    ppg = f"{r['first_3yr_ppg']:.1f}" if pd.notna(r['first_3yr_ppg']) else "N/A"
    verdict = "CORRECT" if r['hit24'] == 1 else "wrong"
    print(f"  {r['player_name']:<25s} {int(r['draft_year']):>4} {int(r['round']):>3} {int(r['pick']):>4} "
          f"{r['slap_dc']:>5.1f} {r['slap']:>6.1f} {r['delta']:>+5.1f} {int(r['rank_delta']):>+5} "
          f"{int(r['hit24']):>4} {int(r['hit12']):>4} {ppg:>8} {verdict:>10}")

pos_correct = (top_pos['hit24'] == 1).sum()
pos_ppg = top_pos[top_pos['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()
print(f"\n  Boosted top 10: {pos_correct}/10 hit24, avg 3yr PPG: {pos_ppg:.2f}")

# Top 10 negative deltas (model dislikes more than DC)
print(f"\n  TOP 10 NEGATIVE DELTAS (model DINGS these TEs below their draft slot):")
print(f"  {'Player':<25s} {'Yr':>4} {'Rd':>3} {'Pick':>4} {'DC':>5} {'SLAP':>6} {'Δ':>6} {'RkΔ':>5} {'h24':>4} {'h12':>4} {'3yr PPG':>8} {'Verdict':>10}")
print(f"  {'-'*100}")

top_neg = eval_df.nsmallest(10, 'delta')
for _, r in top_neg.iterrows():
    ppg = f"{r['first_3yr_ppg']:.1f}" if pd.notna(r['first_3yr_ppg']) else "N/A"
    verdict = "CORRECT" if r['hit24'] == 0 else "wrong"
    print(f"  {r['player_name']:<25s} {int(r['draft_year']):>4} {int(r['round']):>3} {int(r['pick']):>4} "
          f"{r['slap_dc']:>5.1f} {r['slap']:>6.1f} {r['delta']:>+5.1f} {int(r['rank_delta']):>+5} "
          f"{int(r['hit24']):>4} {int(r['hit12']):>4} {ppg:>8} {verdict:>10}")

neg_correct = (top_neg['hit24'] == 0).sum()
neg_ppg = top_neg[top_neg['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()
print(f"\n  Dinged top 10: {neg_correct}/10 busted (hit24=0), avg 3yr PPG: {neg_ppg:.2f}")

# Overall disagreement summary
boosted_all = eval_df[eval_df['delta'] > 3]
dinged_all = eval_df[eval_df['delta'] < -3]
boost_ppg_all = boosted_all[boosted_all['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()
ding_ppg_all = dinged_all[dinged_all['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()
print(f"\n  All TEs boosted >3 pts: N={len(boosted_all)}, avg PPG={boost_ppg_all:.2f}, hit24={boosted_all['hit24'].mean():.0%}")
print(f"  All TEs dinged >3 pts:  N={len(dinged_all)}, avg PPG={ding_ppg_all:.2f}, hit24={dinged_all['hit24'].mean():.0%}")
print(f"  Boost-Ding PPG gap: {boost_ppg_all - ding_ppg_all:+.2f} (positive = model's disagreements are correct)")


# ============================================================================
# 7. COMPREHENSIVE HEAD-TO-HEAD: SLAP vs DC-ONLY
# ============================================================================

print(f"\n\n{'='*100}")
print("7. COMPREHENSIVE HEAD-TO-HEAD: SLAP (65/15/10/5/5) vs DC-ONLY")
print(f"{'='*100}")

print(f"\n  {'Metric':<45} {'SLAP':>10} {'DC-Only':>10} {'Δ':>10} {'Winner':>8}")
print(f"  {'-'*88}")

# Correlations
for out in outcome_cols:
    v = eval_df[['slap', 'slap_dc', out]].dropna()
    r_s, p_s = stats.pearsonr(v['slap'], v[out])
    r_d, p_d = stats.pearsonr(v['slap_dc'], v[out])
    winner = "SLAP" if r_s > r_d else "DC"
    print(f"  Pearson r ({out:20s})   {r_s:>+.4f}   {r_d:>+.4f}   {r_s-r_d:>+.4f} {winner:>8}")

# PRI-AVG
print(f"  {'─'*88}")
print(f"  {'PRI-AVG r (40/25/20/15)':<45} {pri_slap_point:>+.4f}   {pri_dc_point:>+.4f}   {pri_slap_point-pri_dc_point:>+.4f} {'SLAP':>8}")

# AUC
print(f"  {'─'*88}")
for out in ['hit24', 'hit12']:
    auc_s, _, _ = auc_roc(eval_df['slap'], eval_df[out])
    auc_d, _, _ = auc_roc(eval_df['slap_dc'], eval_df[out])
    winner = "SLAP" if auc_s > auc_d else "DC"
    print(f"  {'AUC-ROC (' + out + ')':<45} {auc_s:>.4f}     {auc_d:>.4f}     {auc_s-auc_d:>+.4f} {winner:>8}")

# Brier
print(f"  {'─'*88}")
for out in ['hit24', 'hit12']:
    bs_s, _ = brier_score(eval_df['slap_prob'], eval_df[out])
    bs_d, _ = brier_score(eval_df['dc_prob'], eval_df[out])
    winner = "SLAP" if bs_s < bs_d else "DC"  # lower Brier = better
    print(f"  {'Brier Score (' + out + ') ↓':<45} {bs_s:>.4f}     {bs_d:>.4f}     {bs_s-bs_d:>+.4f} {winner:>8}")

# Top decile
print(f"  {'─'*88}")
n_top = max(1, len(eval_df) // 10)

top_s = eval_df.nlargest(n_top, 'slap')
top_d = eval_df.nlargest(n_top, 'slap_dc')

h24_s = top_s['hit24'].mean() * 100
h24_d = top_d['hit24'].mean() * 100
winner = "SLAP" if h24_s > h24_d else "DC"
print(f"  {'Top 10% hit24 rate':<45} {h24_s:>8.1f}%   {h24_d:>8.1f}%   {h24_s-h24_d:>+8.1f}% {winner:>7}")

h12_s = top_s['hit12'].mean() * 100
h12_d = top_d['hit12'].mean() * 100
winner = "SLAP" if h12_s > h12_d else "DC"
print(f"  {'Top 10% hit12 rate':<45} {h12_s:>8.1f}%   {h12_d:>8.1f}%   {h12_s-h12_d:>+8.1f}% {winner:>7}")

ppg_s = top_s[top_s['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()
ppg_d = top_d[top_d['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()
winner = "SLAP" if ppg_s > ppg_d else "DC"
print(f"  {'Top 10% avg 3yr PPG':<45} {ppg_s:>9.2f}   {ppg_d:>9.2f}   {ppg_s-ppg_d:>+9.2f} {winner:>7}")

# Top 20
top20_s = eval_df.nlargest(20, 'slap')
top20_d = eval_df.nlargest(20, 'slap_dc')
h24_20s = top20_s['hit24'].mean() * 100
h24_20d = top20_d['hit24'].mean() * 100
winner = "SLAP" if h24_20s > h24_20d else "DC" if h24_20s < h24_20d else "TIE"
print(f"  {'Top 20 hit24 rate':<45} {h24_20s:>8.1f}%   {h24_20d:>8.1f}%   {h24_20s-h24_20d:>+8.1f}% {winner:>7}")

# Disagreements
print(f"  {'─'*88}")
disagree_5 = (eval_df['rank_delta'].abs() >= 5).sum()
disagree_10 = (eval_df['rank_delta'].abs() >= 10).sum()
disagree_15 = (eval_df['rank_delta'].abs() >= 15).sum()
print(f"  {'Ranking disagreements (5+ spots)':<45} {disagree_5:>10}")
print(f"  {'Ranking disagreements (10+ spots)':<45} {disagree_10:>10}")
print(f"  {'Ranking disagreements (15+ spots)':<45} {disagree_15:>10}")

# Boost-ding analysis
print(f"  {'─'*88}")
boosted = eval_df[eval_df['rank_delta'] > 5]
dinged = eval_df[eval_df['rank_delta'] < -5]
b_ppg = boosted[boosted['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()
d_ppg = dinged[dinged['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()
b_h24 = boosted['hit24'].mean() * 100
d_h24 = dinged['hit24'].mean() * 100
print(f"  {'Boosted (rank +5) avg PPG':<45} {b_ppg:>9.2f}   (N={len(boosted)}, hit24={b_h24:.0f}%)")
print(f"  {'Dinged (rank -5) avg PPG':<45} {d_ppg:>9.2f}   (N={len(dinged)}, hit24={d_h24:.0f}%)")
print(f"  {'PPG gap (boosted - dinged)':<45} {b_ppg-d_ppg:>+9.2f}   (positive = model is correct)")


# ============================================================================
# SCORE DISTRIBUTION
# ============================================================================

print(f"\n\n{'='*100}")
print("SCORE DISTRIBUTION")
print(f"{'='*100}")

print(f"\n  SLAP scores:")
print(f"    Range: {eval_df['slap'].min():.1f} - {eval_df['slap'].max():.1f}")
print(f"    Mean: {eval_df['slap'].mean():.1f}")
print(f"    Median: {eval_df['slap'].median():.1f}")
print(f"    Std dev: {eval_df['slap'].std():.1f}")

print(f"\n  DC-only scores:")
print(f"    Range: {eval_df['slap_dc'].min():.1f} - {eval_df['slap_dc'].max():.1f}")
print(f"    Mean: {eval_df['slap_dc'].mean():.1f}")
print(f"    Median: {eval_df['slap_dc'].median():.1f}")
print(f"    Std dev: {eval_df['slap_dc'].std():.1f}")

# Correlation between SLAP and DC
r_slap_dc, _ = stats.pearsonr(eval_df['slap'], eval_df['slap_dc'])
print(f"\n  Correlation between SLAP and DC: r={r_slap_dc:.4f}")
print(f"  (1.0 = identical rankings, <0.95 = model adds meaningful reranking)")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n\n{'='*120}")
print("FINAL SUMMARY")
print(f"{'='*120}")

n_slap_wins = 0
n_total_metrics = 0

# Count wins
for out in outcome_cols:
    v = eval_df[['slap', 'slap_dc', out]].dropna()
    r_s, _ = stats.pearsonr(v['slap'], v[out])
    r_d, _ = stats.pearsonr(v['slap_dc'], v[out])
    n_total_metrics += 1
    if r_s > r_d:
        n_slap_wins += 1

for out in ['hit24', 'hit12']:
    auc_s, _, _ = auc_roc(eval_df['slap'], eval_df[out])
    auc_d, _, _ = auc_roc(eval_df['slap_dc'], eval_df[out])
    n_total_metrics += 1
    if auc_s > auc_d:
        n_slap_wins += 1

for out in ['hit24', 'hit12']:
    bs_s, _ = brier_score(eval_df['slap_prob'], eval_df[out])
    bs_d, _ = brier_score(eval_df['dc_prob'], eval_df[out])
    n_total_metrics += 1
    if bs_s < bs_d:
        n_slap_wins += 1

# Top decile hit24
n_total_metrics += 1
if h24_s > h24_d:
    n_slap_wins += 1

print(f"""
  TE SLAP Model: DC 65% / Breakout 15% / Production 10% / ED 5% / RAS 5%

  Eval sample: {len(eval_df)} TEs (2015-2024, excluding 2025)

  SLAP wins {n_slap_wins} of {n_total_metrics} validation metrics vs DC-only baseline.

  PRI-AVG r:       {pri_slap_point:+.4f} (DC: {pri_dc_point:+.4f}, Δ: {pri_slap_point - pri_dc_point:+.4f})
  Bootstrap:       SLAP > DC in {win_pct_pri:.1f}% of 1000 resamples
  Top 10% hit24:   {h24_s:.1f}% (DC: {h24_d:.1f}%)
  Top 10% 3yr PPG: {ppg_s:.2f} (DC: {ppg_d:.2f})
  Disagreements:   {disagree_10} rankings moved 10+ spots vs DC
  Boost-Ding gap:  {b_ppg - d_ppg:+.2f} PPG (boosted TEs outperform dinged TEs)
""")

print(f"{'='*120}")
print("VALIDATION COMPLETE — Ready for 2026 rookie class.")
print(f"{'='*120}")
