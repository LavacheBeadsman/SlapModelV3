"""
Diagnostic script — DO NOT modify any data or scores.
1. TE production score distribution (raw → minmax → percentile)
2. Hit rates by SLAP tier for all 3 positions
"""
import pandas as pd
import numpy as np
from scipy.stats import rankdata

master = pd.read_csv('output/slap_v5_master_database.csv')
bt = master[master['data_type'] == 'backtest'].copy()

# ============================================================================
# DIAGNOSTIC 1: TE PRODUCTION SCORES — raw, minmax, percentile
# ============================================================================
print("=" * 110)
print("DIAGNOSTIC 1: TE PRODUCTION SCORE DISTRIBUTION")
print("=" * 110)

te = bt[bt['position'] == 'TE'].copy()

# Reconstruct the raw production score from rec_yards and team_pass_att
# Formula: (rec_yards / team_pass_att) * age_weight * 100
def calc_raw_prod(row):
    ry = row.get('rec_yards', np.nan)
    tpa = row.get('team_pass_att', np.nan)
    da = row.get('draft_age', np.nan)
    dy = row.get('draft_year', np.nan)
    if pd.isna(ry) or pd.isna(tpa) or tpa == 0:
        return np.nan
    season_year = dy - 1
    season_age = da - (dy - season_year) if pd.notna(da) else 22
    if season_age <= 19: aw = 1.15
    elif season_age <= 20: aw = 1.10
    elif season_age <= 21: aw = 1.05
    elif season_age <= 22: aw = 1.00
    elif season_age <= 23: aw = 0.95
    else: aw = 0.90
    return (ry / tpa) * aw * 100

te['raw_prod'] = te.apply(calc_raw_prod, axis=1)

# Min-max normalize (same as build script)
prod_vals = te['raw_prod'].dropna()
te_prod_min = prod_vals.min()
te_prod_max = prod_vals.max()
te['minmax_prod'] = np.where(
    te['raw_prod'].notna(),
    ((te['raw_prod'] - te_prod_min) / (te_prod_max - te_prod_min) * 99.9).clip(0, 99.9),
    np.nan
)

# Fill missing with mean (same as build script)
mm_avg = te['minmax_prod'].mean()
te['minmax_filled'] = te['minmax_prod'].fillna(mm_avg)

# Percentile rank (same as build script)
def percentile_rank(series):
    vals = series.values
    ranks = rankdata(vals, method='average')
    return (ranks - 1) / (len(ranks) - 1) * 100

te['pctl_prod'] = percentile_rank(te['minmax_filled'])

# The te_production_score from the CSV should match pctl_prod
print(f"\n  Raw production: min={prod_vals.min():.2f}, max={prod_vals.max():.2f}, "
      f"mean={prod_vals.mean():.2f}, median={prod_vals.median():.2f}")
print(f"  TEs with real production data: {prod_vals.count()} / {len(te)}")
print(f"  TEs with missing production (imputed to mean): {te['raw_prod'].isna().sum()}")

# Distribution of raw scores
print(f"\n  RAW production score distribution:")
for thresh in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]:
    count = (prod_vals >= thresh).sum()
    print(f"    >= {thresh:>3}: {count:>3} TEs")

# Percentile scores 95+
high_pctl = te[te['pctl_prod'] >= 95].sort_values('pctl_prod', ascending=False)
print(f"\n  TEs with percentile production >= 95: {len(high_pctl)}")

# Top 20 TEs by production
print(f"\n  TOP 20 TEs by production score (raw → minmax → percentile):")
print(f"  {'#':>3} {'Name':<25} {'RecYds':>7} {'TPA':>5} {'Rec/TPA':>8} {'AgeWt':>6} {'Raw':>7} {'MinMax':>7} {'Pctl':>7} {'CSV':>7}")
print(f"  {'─'*3} {'─'*25} {'─'*7} {'─'*5} {'─'*8} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

top20_prod = te.nlargest(20, 'pctl_prod')
for i, (_, r) in enumerate(top20_prod.iterrows(), 1):
    ry = r.get('rec_yards', np.nan)
    tpa = r.get('team_pass_att', np.nan)
    ratio = ry / tpa if pd.notna(ry) and pd.notna(tpa) and tpa > 0 else np.nan
    csv_score = r.get('te_production_score', np.nan)

    # Reconstruct age weight
    da = r.get('draft_age', np.nan)
    dy = r.get('draft_year', np.nan)
    season_age = da - 1 if pd.notna(da) else 22
    if season_age <= 19: aw = 1.15
    elif season_age <= 20: aw = 1.10
    elif season_age <= 21: aw = 1.05
    elif season_age <= 22: aw = 1.00
    elif season_age <= 23: aw = 0.95
    else: aw = 0.90

    ry_s = f"{ry:.0f}" if pd.notna(ry) else "N/A"
    tpa_s = f"{tpa:.0f}" if pd.notna(tpa) else "N/A"
    ratio_s = f"{ratio:.4f}" if pd.notna(ratio) else "N/A"
    raw_s = f"{r['raw_prod']:.2f}" if pd.notna(r['raw_prod']) else "imputed"
    mm_s = f"{r['minmax_prod']:.1f}" if pd.notna(r['minmax_prod']) else f"{r['minmax_filled']:.1f}*"
    csv_s = f"{csv_score:.1f}" if pd.notna(csv_score) else "N/A"

    print(f"  {i:>3} {r['player_name']:<25} {ry_s:>7} {tpa_s:>5} {ratio_s:>8} {aw:>6.2f} {raw_s:>7} {mm_s:>7} {r['pctl_prod']:>7.1f} {csv_s:>7}")

# Clustering at top
print(f"\n  CLUSTERING CHECK — how many unique raw values map to top percentile scores:")
top_decile = te[te['pctl_prod'] >= 90]
print(f"  TEs in top decile (pctl >= 90): {len(top_decile)}")
raw_unique = top_decile['raw_prod'].dropna().nunique()
print(f"  Unique raw production values in top decile: {raw_unique}")
print(f"  Raw value range in top decile: {top_decile['raw_prod'].min():.2f} to {top_decile['raw_prod'].max():.2f}")

# Bottom 20 for context
print(f"\n  BOTTOM 10 TEs by production score:")
bot10_prod = te.nsmallest(10, 'pctl_prod')
for i, (_, r) in enumerate(bot10_prod.iterrows(), 1):
    ry = r.get('rec_yards', np.nan)
    tpa = r.get('team_pass_att', np.nan)
    raw_s = f"{r['raw_prod']:.2f}" if pd.notna(r['raw_prod']) else "imputed"
    print(f"  {i:>3} {r['player_name']:<25} raw={raw_s:>7}  minmax={r['minmax_filled']:>5.1f}  pctl={r['pctl_prod']:>5.1f}")


# ============================================================================
# DIAGNOSTIC 2: HIT RATES BY SLAP TIER — all 3 positions
# ============================================================================
print(f"\n\n{'='*110}")
print("DIAGNOSTIC 2: HIT RATES BY SLAP TIER (using pooled-rescaled slap_v5)")
print("=" * 110)

# For WR/RB: hit24 (top 24 at position)
# For TE: nfl_hit24 column — but what does this actually measure?
# Let's check what hit columns exist
hit_cols = [c for c in bt.columns if 'hit' in c.lower() or 'top' in c.lower()]
print(f"\n  Hit/outcome columns available: {hit_cols}")

# Check TE hit definition
te_hits = bt[bt['position'] == 'TE']
print(f"\n  TE nfl_hit24 distribution: {te_hits['nfl_hit24'].value_counts().to_dict()}")
print(f"  TE hit rate: {te_hits['nfl_hit24'].mean():.3f}")

# Show hit rates by SLAP tier for each position
tiers = [(90, 100, '90-100'), (80, 89.9, '80-89'), (70, 79.9, '70-79'),
         (60, 69.9, '60-69'), (50, 59.9, '50-59'), (40, 49.9, '40-49'),
         (30, 39.9, '30-39'), (0, 29.9, '0-29')]

print(f"\n  {'Tier':>7} | {'WR n':>5} {'WR hit24%':>10} | {'RB n':>5} {'RB hit24%':>10} | {'TE n':>5} {'TE hit%':>10}")
print(f"  {'─'*7}-+-{'─'*5}-{'─'*10}-+-{'─'*5}-{'─'*10}-+-{'─'*5}-{'─'*10}")

for lo, hi, label in tiers:
    row_parts = [f"  {label:>7} |"]
    for pos in ['WR', 'RB', 'TE']:
        sub = bt[(bt['position'] == pos) & (bt['slap_v5'] >= lo) & (bt['slap_v5'] <= hi)]
        n = len(sub)
        if n > 0:
            hit_rate = sub['nfl_hit24'].mean() * 100
            row_parts.append(f" {n:>5} {hit_rate:>9.1f}% |")
        else:
            row_parts.append(f" {n:>5} {'---':>9}  |")
    print("".join(row_parts))

# Totals
print(f"  {'─'*7}-+-{'─'*5}-{'─'*10}-+-{'─'*5}-{'─'*10}-+-{'─'*5}-{'─'*10}")
row_parts = [f"  {'TOTAL':>7} |"]
for pos in ['WR', 'RB', 'TE']:
    sub = bt[bt['position'] == pos]
    n = len(sub)
    hit_rate = sub['nfl_hit24'].mean() * 100
    row_parts.append(f" {n:>5} {hit_rate:>9.1f}% |")
print("".join(row_parts))

# Also show: what does a SLAP of 80 "mean" across positions?
print(f"\n\n  WHAT DOES A SLAP OF 80+ MEAN ACROSS POSITIONS?")
print(f"  {'Position':>8} | {'n at 80+':>8} | {'Hit rate':>8} | {'Avg PPG':>8}")
print(f"  {'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}")
for pos in ['WR', 'RB', 'TE']:
    sub = bt[(bt['position'] == pos) & (bt['slap_v5'] >= 80)]
    n = len(sub)
    if n > 0:
        hr = sub['nfl_hit24'].mean() * 100
        ppg_col = [c for c in sub.columns if 'ppg' in c.lower() or 'first_3yr' in c.lower()]
        ppg = sub[ppg_col[0]].mean() if ppg_col else float('nan')
        print(f"  {pos:>8} | {n:>8} | {hr:>7.1f}% | {ppg:>8.2f}")
    else:
        print(f"  {pos:>8} | {n:>8} | {'---':>8} | {'---':>8}")

print(f"\n  WHAT DOES A SLAP OF 70+ MEAN ACROSS POSITIONS?")
print(f"  {'Position':>8} | {'n at 70+':>8} | {'Hit rate':>8}")
print(f"  {'─'*8}-+-{'─'*8}-+-{'─'*8}")
for pos in ['WR', 'RB', 'TE']:
    sub = bt[(bt['position'] == pos) & (bt['slap_v5'] >= 70)]
    n = len(sub)
    if n > 0:
        hr = sub['nfl_hit24'].mean() * 100
        print(f"  {pos:>8} | {n:>8} | {hr:>7.1f}%")
