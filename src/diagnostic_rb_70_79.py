"""
Diagnostic: Show the 12 RBs in the 70-79 SLAP tier with full NFL stats.
"""
import pandas as pd
import numpy as np

master = pd.read_csv('output/slap_v5_master_database.csv')
bt = master[master['data_type'] == 'backtest'].copy()

# Also load the RB backtest file for more detail
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')

# Also load outcomes for PPG data
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')

print("=" * 120)
print("RB HIT24 DEFINITION")
print("=" * 120)
print("  hit24 = 1 if player finished top-24 RB in total PPR points in ANY NFL season")
print("  This is a SEASON-LEVEL finish, not a single-game threshold.")
print(f"\n  Overall RB hit24 rate: {bt[bt['position']=='RB']['nfl_hit24'].mean():.3f} ({bt[bt['position']=='RB']['nfl_hit24'].mean()*100:.1f}%)")
print(f"  Overall WR hit24 rate: {bt[bt['position']=='WR']['nfl_hit24'].mean():.3f} ({bt[bt['position']=='WR']['nfl_hit24'].mean()*100:.1f}%)")
print(f"  Overall TE hit rate (top12_10g): {bt[bt['position']=='TE']['nfl_hit24'].mean():.3f} ({bt[bt['position']=='TE']['nfl_hit24'].mean()*100:.1f}%)")

# RB hit rate by round
print(f"\n  RB hit24 by draft round:")
for rd in range(1, 8):
    sub = bt[(bt['position'] == 'RB') & (bt['round'] == rd)]
    if len(sub) > 0:
        print(f"    Round {rd}: {sub['nfl_hit24'].mean()*100:.1f}% ({int(sub['nfl_hit24'].sum())}/{len(sub)})")

# WR hit rate by round for comparison
print(f"\n  WR hit24 by draft round:")
for rd in range(1, 8):
    sub = bt[(bt['position'] == 'WR') & (bt['round'] == rd)]
    if len(sub) > 0:
        print(f"    Round {rd}: {sub['nfl_hit24'].mean()*100:.1f}% ({int(sub['nfl_hit24'].sum())}/{len(sub)})")

# ============================================================================
# THE 12 RBs IN THE 70-79 SLAP TIER
# ============================================================================
print(f"\n\n{'='*120}")
print("THE 12 RBs IN THE 70-79 SLAP TIER (91.7% hit rate)")
print("=" * 120)

rb_70_79 = bt[(bt['position'] == 'RB') & (bt['slap_v5'] >= 70) & (bt['slap_v5'] < 80)].copy()
rb_70_79 = rb_70_79.sort_values('slap_v5', ascending=False)

# Merge with outcomes for more detail
print(f"\n  Outcome columns in master: {[c for c in bt.columns if 'ppg' in c.lower() or 'game' in c.lower() or 'season' in c.lower() or 'hit' in c.lower()]}")

print(f"\n  {'#':>2} {'Name':<25} {'Year':>4} {'Pk':>3} {'Rd':>2} {'SLAP':>5} {'hit24':>5} {'hit12':>5} {'1st3yr PPG':>10} {'Career PPG':>10} {'Games':>6}")
print(f"  {'─'*2} {'─'*25} {'─'*4} {'─'*3} {'─'*2} {'─'*5} {'─'*5} {'─'*5} {'─'*10} {'─'*10} {'─'*6}")

for i, (_, r) in enumerate(rb_70_79.iterrows(), 1):
    ppg3 = r.get('nfl_first_3yr_ppg', np.nan)
    ppgc = r.get('nfl_career_ppg', np.nan)
    games = r.get('nfl_games', np.nan)
    h24 = int(r['nfl_hit24']) if pd.notna(r['nfl_hit24']) else '?'
    h12 = int(r.get('nfl_hit12', np.nan)) if pd.notna(r.get('nfl_hit12', np.nan)) else '?'
    ppg3_s = f"{ppg3:.1f}" if pd.notna(ppg3) else "N/A"
    ppgc_s = f"{ppgc:.1f}" if pd.notna(ppgc) else "N/A"
    games_s = f"{games:.0f}" if pd.notna(games) else "N/A"
    print(f"  {i:>2} {r['player_name']:<25} {int(r['draft_year']):>4} {int(r['pick']):>3} {int(r['round']):>2} {r['slap_v5']:>5.1f} {h24:>5} {h12:>5} {ppg3_s:>10} {ppgc_s:>10} {games_s:>6}")

hits = rb_70_79['nfl_hit24'].sum()
total = len(rb_70_79)
print(f"\n  Hit rate: {int(hits)}/{total} = {hits/total*100:.1f}%")
miss_names = rb_70_79[rb_70_79['nfl_hit24'] == 0]['player_name'].tolist()
print(f"  Misses: {miss_names}")

# ============================================================================
# FOR CONTEXT: show ALL RBs by SLAP tier with names
# ============================================================================
print(f"\n\n{'='*120}")
print("ALL RBs IN SLAP 60-100 RANGE (for context)")
print("=" * 120)

for lo, hi, label in [(90, 100, '90-100'), (80, 89.9, '80-89'), (70, 79.9, '70-79'), (60, 69.9, '60-69')]:
    tier = bt[(bt['position'] == 'RB') & (bt['slap_v5'] >= lo) & (bt['slap_v5'] <= hi)].sort_values('slap_v5', ascending=False)
    hits = int(tier['nfl_hit24'].sum())
    n = len(tier)
    hr = hits/n*100 if n > 0 else 0
    print(f"\n  TIER {label}: {hits}/{n} hit ({hr:.1f}%)")
    for _, r in tier.iterrows():
        h = "HIT" if r['nfl_hit24'] == 1 else "miss"
        ppg = r.get('nfl_first_3yr_ppg', np.nan)
        ppg_s = f"{ppg:.1f}" if pd.notna(ppg) else "N/A"
        print(f"    {r['player_name']:<25} pk{int(r['pick']):>3} rd{int(r['round'])} SLAP={r['slap_v5']:>5.1f}  {h:<4}  3yr_ppg={ppg_s}")

# ============================================================================
# IS 26.5% TOO HIGH? Compare to expected base rates
# ============================================================================
print(f"\n\n{'='*120}")
print("IS 26.5% RB HIT RATE REASONABLE?")
print("=" * 120)
print(f"  RB hit24 means: finished top-24 RB in PPR points in at least one NFL season")
print(f"  There are 32 NFL teams, each starting 1 RB, but top-24 means roughly top 75% of starters")
print(f"  Plus RBs have short careers — many get 2-3 chances to crack top-24")
print(f"\n  RB sample: {len(bt[bt['position']=='RB'])} drafted RBs (2015-2024)")
print(f"  RB hits: {int(bt[bt['position']=='RB']['nfl_hit24'].sum())}")
print(f"  RB hit rate: {bt[bt['position']=='RB']['nfl_hit24'].mean()*100:.1f}%")

# Compare hit12
print(f"\n  For reference — hit12 rates (stricter threshold):")
for pos in ['WR', 'RB', 'TE']:
    sub = bt[bt['position'] == pos]
    h12_col = 'nfl_hit12'
    if h12_col in sub.columns:
        print(f"    {pos}: {sub[h12_col].mean()*100:.1f}% ({int(sub[h12_col].sum())}/{len(sub)})")

# How many RBs drafted per year vs top-24 slots
print(f"\n  RBs drafted per year (avg): {len(bt[bt['position']=='RB']) / 10:.0f}")
print(f"  Top-24 slots available per year: 24")
print(f"  So roughly {24 / (len(bt[bt['position']=='RB']) / 10) * 100:.0f}% of drafted RBs COULD hit top-24 in any given year")
print(f"  But players get multiple years, so cumulative hit rate is higher")
