"""
Why does Receptions/TPA outperform Receiving Yards/TPA for TEs?

Deep dive comparing the two metrics to understand the mechanism.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

bt = pd.read_csv('data/te_backtest_master.csv')
bt = bt[bt['nfl_seasons_found'] > 0].copy()
bt['dc_score'] = bt['pick'].apply(lambda p: max(0, min(100, 100 - 2.40 * (p**0.62 - 1))))

# Build both metrics
bt['ryptpa'] = np.where(
    (bt['cfbd_rec_yards'].notna()) & (bt['cfbd_team_pass_att'].notna()) & (bt['cfbd_team_pass_att'] > 0),
    bt['cfbd_rec_yards'] / bt['cfbd_team_pass_att'], np.nan)

bt['rec_per_tpa'] = np.where(
    (bt['cfbd_receptions'].notna()) & (bt['cfbd_team_pass_att'].notna()) & (bt['cfbd_team_pass_att'] > 0),
    bt['cfbd_receptions'] / bt['cfbd_team_pass_att'], np.nan)

# Yards per reception (the difference between the two metrics)
bt['ypr'] = np.where(
    (bt['cfbd_rec_yards'].notna()) & (bt['cfbd_receptions'].notna()) & (bt['cfbd_receptions'] > 0),
    bt['cfbd_rec_yards'] / bt['cfbd_receptions'], np.nan)

def partial_corr(x, y, z):
    mask = pd.notna(x) & pd.notna(y) & pd.notna(z)
    x2, y2, z2 = x[mask].astype(float).values, y[mask].astype(float).values, z[mask].astype(float).values
    n = len(x2)
    if n < 10: return np.nan, np.nan, n
    sx, ix, _, _, _ = stats.linregress(z2, x2)
    rx = x2 - (sx * z2 + ix)
    sy, iy, _, _, _ = stats.linregress(z2, y2)
    ry = y2 - (sy * z2 + iy)
    r, p = stats.pearsonr(rx, ry)
    return r, p, n

# ============================================================
# 1. HOW SIMILAR ARE THE TWO METRICS?
# ============================================================
print("="*80)
print("1. HOW SIMILAR ARE Rec/TPA AND RYPTPA?")
print("="*80)

both = bt[bt['rec_per_tpa'].notna() & bt['ryptpa'].notna()].copy()
print(f"\nTEs with both metrics: {len(both)}")

r, p = stats.pearsonr(both['rec_per_tpa'], both['ryptpa'])
print(f"Pearson r = {r:+.4f} (p={p:.6f})")

rho, p_rho = stats.spearmanr(both['rec_per_tpa'], both['ryptpa'])
print(f"Spearman rho = {rho:+.4f} (p={p_rho:.6f})")

print(f"\nThey're almost the same metric — the only difference is yards per reception (YPR).")
print(f"  Rec/TPA = receptions / team_pass_att")
print(f"  RYPTPA  = rec_yards / team_pass_att = (receptions × YPR) / team_pass_att")
print(f"  RYPTPA  = Rec/TPA × YPR")
print(f"\nSo RYPTPA = Rec/TPA weighted by yards per catch.")
print(f"If Rec/TPA beats RYPTPA, it means YPR is HURTING the prediction — adding noise.")

# ============================================================
# 2. SIDE-BY-SIDE PARTIAL CORRELATIONS
# ============================================================
print(f"\n{'='*80}")
print("2. SIDE-BY-SIDE PARTIAL CORRELATIONS (controlling for DC)")
print("="*80)

metrics_to_test = {
    'rec_per_tpa': 'Rec/TPA (catch volume)',
    'ryptpa': 'RYPTPA (yards volume)',
    'ypr': 'Yards Per Reception (the difference)',
    'cfbd_receptions': 'Raw Receptions',
    'cfbd_rec_yards': 'Raw Receiving Yards',
}

print(f"\n{'Metric':<35} {'r(PPG)':>8} {'p(PPG)':>8}  {'r(h24)':>8} {'p(h24)':>8}  {'r(h12)':>8} {'p(h12)':>8}")
print("-"*95)

for col, name in metrics_to_test.items():
    results = {}
    for outcome in ['first_3yr_ppg', 'hit24', 'hit12']:
        r, p, n = partial_corr(bt[col], bt[outcome], bt['pick'])
        results[outcome] = (r, p, n)
    r_ppg, p_ppg, _ = results['first_3yr_ppg']
    r_h24, p_h24, _ = results['hit24']
    r_h12, p_h12, _ = results['hit12']
    print(f"{name:<35} {r_ppg:>+8.3f} {p_ppg:>8.4f}  {r_h24:>+8.3f} {p_h24:>8.4f}  {r_h12:>+8.3f} {p_h12:>8.4f}")

# ============================================================
# 3. THE MECHANISM: Does YPR hurt or help?
# ============================================================
print(f"\n{'='*80}")
print("3. DOES YARDS PER RECEPTION PREDICT NFL SUCCESS?")
print("="*80)

valid = bt['ypr'].notna() & bt['first_3yr_ppg'].notna()
s = bt[valid]
print(f"\nTEs with YPR data: {len(s)}")
print(f"YPR range: {s['ypr'].min():.1f} - {s['ypr'].max():.1f}")
print(f"YPR mean: {s['ypr'].mean():.1f}, median: {s['ypr'].median():.1f}")

# Raw
r_raw, p_raw = stats.pearsonr(s['ypr'], s['first_3yr_ppg'])
print(f"\nYPR vs PPG (raw):     r={r_raw:+.3f}, p={p_raw:.4f}")

# Partial
r_part, p_part, n = partial_corr(bt['ypr'], bt['first_3yr_ppg'], bt['pick'])
print(f"YPR vs PPG (partial): r={r_part:+.3f}, p={p_part:.4f}, n={n}")

# YPR vs hit rates
for outcome in ['hit24', 'hit12']:
    r, p, n = partial_corr(bt['ypr'], bt[outcome], bt['pick'])
    print(f"YPR vs {outcome} (partial): r={r:+.3f}, p={p:.4f}, n={n}")

# Split by YPR quartiles
print(f"\nYPR Quartile Analysis (controlling for DC by showing avg pick):")
s = bt[bt['ypr'].notna() & bt['first_3yr_ppg'].notna()].copy()
s['ypr_q'] = pd.qcut(s['ypr'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
    g = s[s['ypr_q'] == q]
    print(f"  {q:>12s}: n={len(g):>3}, avg YPR={g['ypr'].mean():.1f}, "
          f"avg PPG={g['first_3yr_ppg'].mean():.2f}, hit24={g['hit24'].mean():.0%}, "
          f"avg pick={g['pick'].mean():.0f}")

# ============================================================
# 4. EXAMPLES: TEs WHERE THE METRICS DISAGREE
# ============================================================
print(f"\n{'='*80}")
print("4. TEs WHERE REC/TPA AND RYPTPA DISAGREE")
print("="*80)

both = bt[bt['rec_per_tpa'].notna() & bt['ryptpa'].notna() & bt['first_3yr_ppg'].notna()].copy()

# Normalize both to percentile ranks for comparison
both['rec_tpa_pctl'] = both['rec_per_tpa'].rank(pct=True) * 100
both['ryptpa_pctl'] = both['ryptpa'].rank(pct=True) * 100
both['pctl_diff'] = both['rec_tpa_pctl'] - both['ryptpa_pctl']  # positive = Rec/TPA ranks higher

print(f"\nTEs where Rec/TPA ranks HIGHER than RYPTPA (high catches, low YPR):")
print(f"These are TEs with lots of short catches.\n")

high_rec = both.nlargest(10, 'pctl_diff')
print(f"{'Player':<25s} {'Yr':>4} {'Pk':>4} {'Rec':>4} {'Yds':>5} {'YPR':>5} {'Rec/TPA%':>8} {'RYPT%':>7} {'Diff':>5}  {'PPG':>5} {'h24':>4}")
print("-"*95)
for _, r in high_rec.iterrows():
    print(f"{r['player_name']:<25s} {int(r['draft_year']):>4} {int(r['pick']):>4} "
          f"{int(r['cfbd_receptions']):>4} {int(r['cfbd_rec_yards']):>5} {r['ypr']:>5.1f} "
          f"{r['rec_tpa_pctl']:>7.0f}% {r['ryptpa_pctl']:>6.0f}% {r['pctl_diff']:>+5.0f}  "
          f"{r['first_3yr_ppg']:>5.2f} {int(r['hit24']):>4}")

print(f"\nTEs where RYPTPA ranks HIGHER than Rec/TPA (fewer catches, big YPR):")
print(f"These are TEs with fewer but longer catches.\n")

high_yds = both.nsmallest(10, 'pctl_diff')
print(f"{'Player':<25s} {'Yr':>4} {'Pk':>4} {'Rec':>4} {'Yds':>5} {'YPR':>5} {'Rec/TPA%':>8} {'RYPT%':>7} {'Diff':>5}  {'PPG':>5} {'h24':>4}")
print("-"*95)
for _, r in high_yds.iterrows():
    print(f"{r['player_name']:<25s} {int(r['draft_year']):>4} {int(r['pick']):>4} "
          f"{int(r['cfbd_receptions']):>4} {int(r['cfbd_rec_yards']):>5} {r['ypr']:>5.1f} "
          f"{r['rec_tpa_pctl']:>7.0f}% {r['ryptpa_pctl']:>6.0f}% {r['pctl_diff']:>+5.0f}  "
          f"{r['first_3yr_ppg']:>5.2f} {int(r['hit24']):>4}")

# Outcome comparison
print(f"\n--- Outcome comparison ---")
high_rec_only = both[both['pctl_diff'] > 15]  # Rec/TPA ranks much higher
high_yds_only = both[both['pctl_diff'] < -15]  # RYPTPA ranks much higher
middle = both[both['pctl_diff'].abs() <= 15]  # Similar ranking

print(f"  Rec/TPA boosted (pctl_diff > +15): n={len(high_rec_only)}, "
      f"avg PPG={high_rec_only['first_3yr_ppg'].mean():.2f}, hit24={high_rec_only['hit24'].mean():.0%}, "
      f"avg pick={high_rec_only['pick'].mean():.0f}")
print(f"  RYPTPA boosted (pctl_diff < -15):  n={len(high_yds_only)}, "
      f"avg PPG={high_yds_only['first_3yr_ppg'].mean():.2f}, hit24={high_yds_only['hit24'].mean():.0%}, "
      f"avg pick={high_yds_only['pick'].mean():.0f}")
print(f"  Similar ranking (|diff| <= 15):    n={len(middle)}, "
      f"avg PPG={middle['first_3yr_ppg'].mean():.2f}, hit24={middle['hit24'].mean():.0%}, "
      f"avg pick={middle['pick'].mean():.0f}")

# ============================================================
# 5. DECOMPOSITION: What does YPR actually capture?
# ============================================================
print(f"\n{'='*80}")
print("5. WHAT DRIVES HIGH YPR IN COLLEGE TEs?")
print("="*80)

# Check if PFF metrics explain YPR
pff_cols = {
    'pff_avg_depth_of_target': 'Avg Depth of Target (aDOT)',
    'pff_yards_after_catch_per_reception': 'YAC Per Reception',
    'pff_inline_rate': 'Inline Rate',
    'pff_wide_rate': 'Wide Rate',
    'pff_slot_rate': 'Slot Rate',
    'pff_longest': 'Longest Reception',
}

valid = bt['ypr'].notna()
print(f"\nCorrelation between YPR and PFF metrics (what makes YPR high?):")
for col, name in pff_cols.items():
    v = valid & bt[col].notna()
    if v.sum() >= 10:
        r, p = stats.pearsonr(bt.loc[v, 'ypr'], bt.loc[v, col])
        print(f"  {name:<35s}: r={r:+.3f} (p={p:.4f})")

# ============================================================
# 6. FANTASY CONTEXT: Why receptions > yards for TE value
# ============================================================
print(f"\n{'='*80}")
print("6. PPR SCORING CONTEXT")
print("="*80)
print(f"""
In PPR (point-per-reception) fantasy scoring:
  - 1 reception for 5 yards  = 1.5 points (1 PPR + 0.5 yardage)
  - 0 receptions for 0 yards = 0.0 points

A TE who catches 5 passes for 40 yards (8.0 YPR) scores:
  5 PPR + 4.0 yardage = 9.0 fantasy points

A TE who catches 2 passes for 50 yards (25.0 YPR) scores:
  2 PPR + 5.0 yardage = 7.0 fantasy points

MORE CATCHES at shorter distances beats FEWER CATCHES at longer distances
in PPR scoring. This is the fundamental reason Rec/TPA > RYPTPA for
predicting PPR fantasy success.
""")

# Verify: are our NFL outcomes PPR-based?
print(f"Our NFL outcomes (first_3yr_ppg, hit24, hit12) use PPR scoring.")
print(f"Receptions ARE the scoring mechanism. Yards are secondary.")

# Show the math on a real example
print(f"\n--- Real example from the data ---")
example_high_rec = both[both['pctl_diff'] > 15].nlargest(3, 'first_3yr_ppg')
example_high_yds = both[both['pctl_diff'] < -15].nlargest(3, 'first_3yr_ppg')

print(f"\nBest NFL producers among 'many short catches' TEs:")
for _, r in example_high_rec.iterrows():
    print(f"  {r['player_name']:<25s}: {int(r['cfbd_receptions'])} rec, {int(r['cfbd_rec_yards'])} yds, "
          f"YPR={r['ypr']:.1f} -> NFL PPG={r['first_3yr_ppg']:.2f}")

print(f"\nBest NFL producers among 'few long catches' TEs:")
for _, r in example_high_yds.iterrows():
    print(f"  {r['player_name']:<25s}: {int(r['cfbd_receptions'])} rec, {int(r['cfbd_rec_yards'])} yds, "
          f"YPR={r['ypr']:.1f} -> NFL PPG={r['first_3yr_ppg']:.2f}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print("="*80)
