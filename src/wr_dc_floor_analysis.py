"""
WR Draft Capital Floor Analysis
================================
How low can DC weight go before performance meaningfully drops?

Tests 6 weight configurations (DC / BO+Rush / Teammate / ED):
  75/17/4/4, 70/20/5/5, 65/22/7/6, 60/25/8/7, 55/28/9/8, 50/30/10/10

For each: priority-weighted r, 4 outcome correlations, top-decile precision,
top-decile PPG, and ranking disagreements vs pure DC.
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
print("WR DRAFT CAPITAL FLOOR ANALYSIS")
print("How low can DC weight go before performance drops?")
print("Priority weights: first_3yr_ppg=40% | hit24=25% | hit12=20% | career_ppg=15%")
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

# Compute component scores
wr['s_dc'] = wr['pick'].apply(dc_score)
wr['s_breakout'] = wr.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)
wr['s_teammate'] = wr['teammate_dc'].clip(0, 100)
wr['s_declare'] = wr['early_declare'] * 100

# Enhanced breakout with +5 rushing bonus
rush_flag = (wr['rush_yards'] >= 20).astype(int)
wr['s_bo_rush'] = (wr['s_breakout'] + rush_flag * 5).clip(0, 99.9)

# Eval sample
wr_eval = wr[wr['hit24'].notna() & wr['draft_year'].between(2015, 2024)].copy()
print(f"\nEval sample: {len(wr_eval)} WRs")
print(f"  hit24: {int(wr_eval['hit24'].sum())}/{len(wr_eval)} ({wr_eval['hit24'].mean()*100:.1f}%)")
print(f"  hit12: {int(wr_eval['hit12'].sum())}/{len(wr_eval)} ({wr_eval['hit12'].mean()*100:.1f}%)")
print(f"  first_3yr_ppg: {wr_eval['first_3yr_ppg'].notna().sum()} with data")

all_outcomes = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
outcome_weights = {'first_3yr_ppg': 0.40, 'hit24': 0.25, 'hit12': 0.20, 'career_ppg': 0.15}


# ============================================================================
# DEFINE 6 WEIGHT CONFIGURATIONS + DC-ONLY BASELINE
# ============================================================================

configs = [
    ('100/0/0/0 (DC only)',  [1.00, 0.00, 0.00, 0.00]),
    ('75/17/4/4 (current)',  [0.75, 0.17, 0.04, 0.04]),
    ('70/20/5/5',            [0.70, 0.20, 0.05, 0.05]),
    ('65/22/7/6',            [0.65, 0.22, 0.07, 0.06]),
    ('60/25/8/7',            [0.60, 0.25, 0.08, 0.07]),
    ('55/28/9/8',            [0.55, 0.28, 0.09, 0.08]),
    ('50/30/10/10',          [0.50, 0.30, 0.10, 0.10]),
]

comp_cols = ['s_dc', 's_bo_rush', 's_teammate', 's_declare']

# Compute DC-only rank (for disagreement counting)
wr_eval['dc_rank'] = wr_eval['s_dc'].rank(ascending=False, method='min')

# Pre-compute all SLAP scores and ranks
for label, weights in configs:
    tag = label.split('/')[0].strip().replace('(', '').replace(')', '').replace(' ', '')
    col = f'slap_{tag}'
    wr_eval[col] = sum(wr_eval[c] * w for c, w in zip(comp_cols, weights))
    wr_eval[col] = wr_eval[col].clip(0, 100)
    wr_eval[f'rank_{tag}'] = wr_eval[col].rank(ascending=False, method='min')


# ============================================================================
# TEST 1-2: CORRELATIONS & PRIORITY-WEIGHTED AVERAGE
# ============================================================================

print(f"\n\n{'=' * 110}")
print("OUTCOME CORRELATIONS BY WEIGHT CONFIGURATION")
print("=" * 110)

print(f"\n{'Config':<24}", end="")
for out in all_outcomes:
    print(f" {'r('+out+')':>16}", end="")
print(f" {'PRI-AVG':>10} {'Delta':>8}")
print("-" * 100)

results = {}
baseline_r = None

for label, weights in configs:
    tag = label.split('/')[0].strip().replace('(', '').replace(')', '').replace(' ', '')
    col = f'slap_{tag}'

    row = f"  {label:<22}"
    pri_sum = 0
    pri_total = 0
    indiv = {}

    for out in all_outcomes:
        valid = wr_eval[[col, out]].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid[col], valid[out])
            sig = "*" if p < 0.05 else ""
            row += f" {r:>+.4f}{sig:<1} (N={len(valid):>3})"
            pri_sum += outcome_weights[out] * r
            pri_total += outcome_weights[out]
            indiv[out] = r
        else:
            row += f" {'N/A':>16}"

    pri_avg = pri_sum / pri_total if pri_total > 0 else np.nan

    if baseline_r is None:
        baseline_r = pri_avg
        delta_str = "   ---"
    else:
        delta = pri_avg - baseline_r
        delta_str = f"{delta:>+.4f}"

    row += f" {pri_avg:>+.4f}  {delta_str}"
    print(row)

    results[label] = {'pri_avg': pri_avg, 'indiv': indiv}


# ============================================================================
# TEST 3-4: TOP DECILE PRECISION & PPG
# ============================================================================

print(f"\n\n{'=' * 110}")
print("TOP DECILE ANALYSIS (top 10% = ~33 players)")
print("=" * 110)

n_top = max(1, len(wr_eval) // 10)  # ~33

print(f"\n  {'Config':<24} {'Hit24':>6} {'Rate':>8} {'Hit12':>6} {'Rate':>8} "
      f"{'Avg 3yr PPG':>13} {'Med 3yr PPG':>13} {'N(3yr)':>8}")
print("  " + "-" * 95)

for label, weights in configs:
    tag = label.split('/')[0].strip().replace('(', '').replace(')', '').replace(' ', '')
    col = f'slap_{tag}'

    top = wr_eval.nlargest(n_top, col)
    h24 = int(top['hit24'].sum())
    h12 = int(top['hit12'].sum())
    r24 = h24 / n_top * 100
    r12 = h12 / n_top * 100
    top_3yr = top[top['first_3yr_ppg'].notna()]
    avg = top_3yr['first_3yr_ppg'].mean() if len(top_3yr) > 0 else np.nan
    med = top_3yr['first_3yr_ppg'].median() if len(top_3yr) > 0 else np.nan
    avg_s = f"{avg:.2f}" if not np.isnan(avg) else "N/A"
    med_s = f"{med:.2f}" if not np.isnan(med) else "N/A"
    print(f"  {label:<24} {h24:>6} {r24:>7.1f}% {h12:>6} {r12:>7.1f}% "
          f"{avg_s:>13} {med_s:>13} {len(top_3yr):>8}")

# Also show top 20
print(f"\n  Top 20 players:")
print(f"  {'Config':<24} {'Hit24':>6} {'Rate':>8} {'Hit12':>6} {'Rate':>8} "
      f"{'Avg 3yr PPG':>13} {'N(3yr)':>8}")
print("  " + "-" * 80)

for label, weights in configs:
    tag = label.split('/')[0].strip().replace('(', '').replace(')', '').replace(' ', '')
    col = f'slap_{tag}'

    top = wr_eval.nlargest(20, col)
    h24 = int(top['hit24'].sum())
    h12 = int(top['hit12'].sum())
    r24 = h24 / 20 * 100
    r12 = h12 / 20 * 100
    top_3yr = top[top['first_3yr_ppg'].notna()]
    avg = top_3yr['first_3yr_ppg'].mean() if len(top_3yr) > 0 else np.nan
    avg_s = f"{avg:.2f}" if not np.isnan(avg) else "N/A"
    print(f"  {label:<24} {h24:>6} {r24:>7.1f}% {h12:>6} {r12:>7.1f}% "
          f"{avg_s:>13} {len(top_3yr):>8}")


# ============================================================================
# TEST 5: RANKING DISAGREEMENTS VS PURE DC
# ============================================================================

print(f"\n\n{'=' * 110}")
print("RANKING DISAGREEMENTS VS PURE DC")
print("Players where SLAP rank differs from DC rank by 10+ spots")
print("=" * 110)

for label, weights in configs:
    if 'DC only' in label:
        continue

    tag = label.split('/')[0].strip().replace('(', '').replace(')', '').replace(' ', '')
    rank_col = f'rank_{tag}'

    wr_eval['rank_diff'] = wr_eval['dc_rank'] - wr_eval[rank_col]  # positive = model likes more
    disagree_10 = wr_eval[wr_eval['rank_diff'].abs() >= 10]
    disagree_20 = wr_eval[wr_eval['rank_diff'].abs() >= 20]
    disagree_30 = wr_eval[wr_eval['rank_diff'].abs() >= 30]
    boosted = wr_eval[wr_eval['rank_diff'] >= 10]  # model ranks higher than DC
    dinged = wr_eval[wr_eval['rank_diff'] <= -10]  # model ranks lower than DC

    print(f"\n  {label}:")
    print(f"    Disagreements of 10+ spots: {len(disagree_10)}/{len(wr_eval)} ({len(disagree_10)/len(wr_eval)*100:.1f}%)")
    print(f"    Disagreements of 20+ spots: {len(disagree_20)}/{len(wr_eval)} ({len(disagree_20)/len(wr_eval)*100:.1f}%)")
    print(f"    Disagreements of 30+ spots: {len(disagree_30)}/{len(wr_eval)} ({len(disagree_30)/len(wr_eval)*100:.1f}%)")
    print(f"    Boosted by model (rank 10+ higher): {len(boosted)}")
    print(f"    Dinged by model (rank 10+ lower):   {len(dinged)}")

    # Are the disagreements CORRECT? Do boosted players outperform, dinged underperform?
    if len(boosted) >= 5 and len(dinged) >= 5:
        boost_hit = boosted['hit24'].mean()
        ding_hit = dinged['hit24'].mean()
        boost_3yr = boosted[boosted['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()
        ding_3yr = dinged[dinged['first_3yr_ppg'].notna()]['first_3yr_ppg'].mean()

        print(f"\n    Do disagreements predict correctly?")
        print(f"    {'Group':<25} {'Hit24 Rate':>12} {'Avg 3yr PPG':>14} {'N':>6}")
        print(f"    {'-' * 60}")
        print(f"    {'Boosted (model > DC)':<25} {boost_hit:>11.1%} {boost_3yr:>14.2f} {len(boosted):>6}")
        print(f"    {'Dinged (model < DC)':<25} {ding_hit:>11.1%} {ding_3yr:>14.2f} {len(dinged):>6}")
        t, p = stats.ttest_ind(
            boosted[boosted['first_3yr_ppg'].notna()]['first_3yr_ppg'],
            dinged[dinged['first_3yr_ppg'].notna()]['first_3yr_ppg'])
        print(f"    3yr PPG difference: {boost_3yr - ding_3yr:>+.2f} (p={p:.3f})")


# ============================================================================
# DETAILED: WHO MOVES MOST AT EACH CONFIG?
# ============================================================================

# Show the biggest movers at the 65/22/7/6 and 55/28/9/8 configs
for show_label in ['65/22/7/6', '55/28/9/8']:
    matching = [l for l, w in configs if l.startswith(show_label)]
    if not matching:
        continue
    label = matching[0]
    tag = label.split('/')[0].strip().replace('(', '').replace(')', '').replace(' ', '')
    rank_col = f'rank_{tag}'
    slap_col = f'slap_{tag}'

    wr_eval['rank_diff'] = wr_eval['dc_rank'] - wr_eval[rank_col]

    print(f"\n\n{'─' * 80}")
    print(f"TOP 15 BIGGEST RISERS — {label}")
    print(f"(Players model likes MORE than DC)")
    print(f"{'─' * 80}")

    risers = wr_eval.nlargest(15, 'rank_diff')
    print(f"\n  {'Player':<25} {'Year':>5} {'Pick':>5} {'DC Rank':>8} {'SLAP Rank':>10} {'Move':>6} "
          f"{'SLAP':>6} {'Hit24':>6} {'3yr PPG':>9}")
    print("  " + "-" * 90)
    for _, row in risers.iterrows():
        ppg_str = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "N/A"
        print(f"  {row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} "
              f"{int(row['dc_rank']):>8} {int(row[rank_col]):>10} {int(row['rank_diff']):>+6} "
              f"{row[slap_col]:>6.1f} {int(row['hit24']):>6} {ppg_str:>9}")

    print(f"\n  Risers hit24 rate: {risers['hit24'].mean()*100:.1f}%")
    risers_3yr = risers[risers['first_3yr_ppg'].notna()]
    if len(risers_3yr) > 0:
        print(f"  Risers avg 3yr PPG: {risers_3yr['first_3yr_ppg'].mean():.2f}")

    print(f"\n{'─' * 80}")
    print(f"TOP 15 BIGGEST FALLERS — {label}")
    print(f"(Players model likes LESS than DC)")
    print(f"{'─' * 80}")

    fallers = wr_eval.nsmallest(15, 'rank_diff')
    print(f"\n  {'Player':<25} {'Year':>5} {'Pick':>5} {'DC Rank':>8} {'SLAP Rank':>10} {'Move':>6} "
          f"{'SLAP':>6} {'Hit24':>6} {'3yr PPG':>9}")
    print("  " + "-" * 90)
    for _, row in fallers.iterrows():
        ppg_str = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "N/A"
        print(f"  {row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} "
              f"{int(row['dc_rank']):>8} {int(row[rank_col]):>10} {int(row['rank_diff']):>+6} "
              f"{row[slap_col]:>6.1f} {int(row['hit24']):>6} {ppg_str:>9}")

    print(f"\n  Fallers hit24 rate: {fallers['hit24'].mean()*100:.1f}%")
    fallers_3yr = fallers[fallers['first_3yr_ppg'].notna()]
    if len(fallers_3yr) > 0:
        print(f"  Fallers avg 3yr PPG: {fallers_3yr['first_3yr_ppg'].mean():.2f}")


# ============================================================================
# PERFORMANCE DROPOFF VISUALIZATION (text-based)
# ============================================================================

print(f"\n\n{'=' * 110}")
print("PERFORMANCE CURVE: How each metric changes as DC weight drops")
print("=" * 110)

dc_pcts = []
pri_avgs = []
hit24_rates = []
ppg_avgs = []
disagree_counts = []

for label, weights in configs:
    tag = label.split('/')[0].strip().replace('(', '').replace(')', '').replace(' ', '')
    col = f'slap_{tag}'
    rank_col = f'rank_{tag}'

    dc_pct = weights[0] * 100
    dc_pcts.append(dc_pct)

    # Priority-weighted r
    pri_sum = 0
    pri_total = 0
    for out in all_outcomes:
        valid = wr_eval[[col, out]].dropna()
        if len(valid) >= 10:
            r, _ = stats.pearsonr(valid[col], valid[out])
            pri_sum += outcome_weights[out] * r
            pri_total += outcome_weights[out]
    pri_avgs.append(pri_sum / pri_total if pri_total > 0 else np.nan)

    # Top decile hit24 rate
    top = wr_eval.nlargest(n_top, col)
    hit24_rates.append(top['hit24'].mean() * 100)

    # Top decile avg 3yr PPG
    top_3yr = top[top['first_3yr_ppg'].notna()]
    ppg_avgs.append(top_3yr['first_3yr_ppg'].mean() if len(top_3yr) > 0 else np.nan)

    # Disagreements (10+ spots)
    if 'DC only' not in label:
        wr_eval['rank_diff'] = wr_eval['dc_rank'] - wr_eval[rank_col]
        disagree_counts.append(int((wr_eval['rank_diff'].abs() >= 10).sum()))
    else:
        disagree_counts.append(0)

# Text table
print(f"\n  {'DC %':<8} {'PRI-AVG r':>12} {'Δ vs DC-only':>14} {'Top10% Hit24':>14} {'Top10% PPG':>12} {'Disagree 10+':>14}")
print("  " + "-" * 80)

for i, (label, _) in enumerate(configs):
    delta = pri_avgs[i] - pri_avgs[0] if i > 0 else 0
    delta_str = f"{delta:>+.4f}" if i > 0 else "    ---"
    ppg_str = f"{ppg_avgs[i]:.2f}" if not np.isnan(ppg_avgs[i]) else "N/A"
    print(f"  {dc_pcts[i]:>5.0f}%  {pri_avgs[i]:>+.4f}      {delta_str}  {hit24_rates[i]:>12.1f}%  {ppg_str:>12} {disagree_counts[i]:>14}")

# Text bar chart for priority-weighted r
print(f"\n  Priority-weighted r (visual):")
max_r = max(pri_avgs)
min_r = min(pri_avgs)
for i, (label, _) in enumerate(configs):
    bar_len = int((pri_avgs[i] - 0.40) / (max_r - 0.40) * 50) if max_r > 0.40 else 0
    bar_len = max(0, bar_len)
    marker = " ◄ BEST" if pri_avgs[i] == max_r else ""
    print(f"  DC {dc_pcts[i]:>3.0f}%: {'█' * bar_len} {pri_avgs[i]:+.4f}{marker}")

# Text bar chart for top decile hit rate
print(f"\n  Top 10% hit24 rate (visual):")
for i, (label, _) in enumerate(configs):
    bar_len = int(hit24_rates[i])
    marker = " ◄ BEST" if hit24_rates[i] == max(hit24_rates) else ""
    print(f"  DC {dc_pcts[i]:>3.0f}%: {'█' * (bar_len // 2)} {hit24_rates[i]:.1f}%{marker}")


# ============================================================================
# GRAND SUMMARY
# ============================================================================

print(f"\n\n{'=' * 110}")
print("SUMMARY & RECOMMENDATION")
print("=" * 110)

# Find the sweet spot
print(f"""
The key tradeoff:
  - Higher DC weight → better raw prediction (correlation, AUC)
  - Lower DC weight → more interesting content (disagreements with draft position)

Performance thresholds:
""")

for i, (label, _) in enumerate(configs):
    if i == 0:
        continue
    delta = pri_avgs[i] - pri_avgs[1]  # vs 75/17/4/4
    pct_drop = delta / pri_avgs[1] * 100 if pri_avgs[1] != 0 else 0
    hit_delta = hit24_rates[i] - hit24_rates[1]
    print(f"  {label:<24}  r delta vs 75/17: {delta:>+.4f} ({pct_drop:>+.1f}%)  "
          f"Hit24 delta: {hit_delta:>+.1f}pp  Disagreements: {disagree_counts[i]:>3}")


print(f"\n\n{'=' * 110}")
print("ANALYSIS COMPLETE")
print("=" * 110)
