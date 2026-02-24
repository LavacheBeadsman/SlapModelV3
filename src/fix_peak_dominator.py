"""
fix_peak_dominator.py — Fix peak_dominator values in WR backtest.

Bug: peak_dominator stored the FINAL college season's dominator_rating
instead of the CAREER PEAK. This affects 69/339 WRs and changes
teammate score tiers for 13+ players.

Fix: for each player, find max(dominator_rating) across all college
seasons in wr_all_seasons.csv and update if different.
"""

import pandas as pd
import numpy as np

wr = pd.read_csv('data/wr_backtest_all_components.csv')
seasons = pd.read_csv('data/wr_all_seasons.csv')
tm = pd.read_csv('data/wr_teammate_scores.csv')

print("=" * 80)
print("FIXING PEAK DOMINATOR VALUES")
print("=" * 80)

# Build career peak lookup from wr_all_seasons
career_peaks = {}
for name in seasons['player_name'].unique():
    s = seasons[seasons['player_name'] == name]
    peak = s['dominator_rating'].max()
    dy = int(s['draft_year'].iloc[0])
    career_peaks[(name, dy)] = peak

# Fix each player
changes = []
for idx, row in wr.iterrows():
    name = row['player_name']
    dy = int(row['draft_year'])
    old_pd = row['peak_dominator']

    key = (name, dy)
    if key not in career_peaks:
        continue

    correct_pd = career_peaks[key]
    if pd.isna(old_pd) and pd.isna(correct_pd):
        continue
    if pd.notna(old_pd) and pd.notna(correct_pd) and abs(old_pd - correct_pd) < 0.1:
        continue

    # Update
    wr.at[idx, 'peak_dominator'] = correct_pd
    changes.append({
        'name': name,
        'pick': int(row['pick']),
        'draft_year': dy,
        'old_pd': old_pd,
        'new_pd': correct_pd,
        'diff': correct_pd - old_pd if pd.notna(old_pd) and pd.notna(correct_pd) else None,
    })

print(f"\n  Fixed {len(changes)} peak_dominator values\n")

# Show changes sorted by magnitude of difference
changes_sorted = sorted(changes, key=lambda c: abs(c['diff']) if c['diff'] is not None else 0, reverse=True)
print(f"  {'Player':<25} {'Pick':>4} {'Old PD':>7} {'New PD':>7} {'Diff':>6}")
print(f"  {'-'*55}")
for c in changes_sorted[:30]:
    old = f"{c['old_pd']:.1f}" if pd.notna(c['old_pd']) else "NaN"
    diff = f"{c['diff']:+.1f}" if c['diff'] is not None else "N/A"
    print(f"  {c['name']:<25} {c['pick']:>4} {old:>7} {c['new_pd']:>7.1f} {diff:>6}")

if len(changes_sorted) > 30:
    print(f"  ... and {len(changes_sorted) - 30} more")

# Now show impact on teammate scores
print(f"\n{'=' * 80}")
print("TEAMMATE SCORE IMPACT (with extreme-DC adjustment: DC>300 lowers tiers by 5%)")
print(f"{'=' * 80}")

def old_tiered_tm(tm_dc, broke_out, peak_dom):
    if pd.isna(tm_dc) or tm_dc <= 150 or not broke_out:
        return 0.0
    if pd.isna(peak_dom) or peak_dom < 20:
        return 0.0
    if peak_dom < 25: return 40.0
    elif peak_dom < 30: return 60.0
    elif peak_dom < 35: return 80.0
    else: return 100.0

def new_tiered_tm(tm_dc, broke_out, peak_dom):
    """Teammate score with extreme-DC adjustment.
    DC > 300: tiers shift down 5% (15/20/25/30 instead of 20/25/30/35)
    """
    if pd.isna(tm_dc) or tm_dc <= 150 or not broke_out:
        return 0.0
    # Extreme DC: lower thresholds by 5%
    floor = 15 if tm_dc > 300 else 20
    t1 = floor + 5
    t2 = floor + 10
    t3 = floor + 15
    if pd.isna(peak_dom) or peak_dom < floor:
        return 0.0
    if peak_dom < t1: return 40.0
    elif peak_dom < t2: return 60.0
    elif peak_dom < t3: return 80.0
    else: return 100.0

# Merge teammate DC
wr_merged = wr.merge(tm[['player_name', 'draft_year', 'total_teammate_dc']],
                      on=['player_name', 'draft_year'], how='left')

print(f"\n  {'Player':<25} {'Pick':>4} {'PD':>5} {'TM_DC':>6}  {'Old TM':>6} {'New TM':>6} {'Chg':>4}")
print(f"  {'-'*65}")

tm_changes = 0
for _, r in wr_merged.iterrows():
    broke_out = pd.notna(r['breakout_age'])
    tm_dc = r['total_teammate_dc']
    pd_val = r['peak_dominator']

    # Old score: uses the ORIGINAL peak_dominator and old formula
    # We need the original PD for the "before" — but we already updated it.
    # So let's use the old formula with the NEW (correct) PD to isolate the formula change.
    old_score = old_tiered_tm(tm_dc, broke_out, pd_val)
    new_score = new_tiered_tm(tm_dc, broke_out, pd_val)

    if old_score != new_score:
        tm_changes += 1
        print(f"  {r['player_name']:<25} {int(r['pick']):>4} {pd_val:>5.1f} {tm_dc:>6.0f}  {old_score:>6.0f} {new_score:>6.0f} {new_score-old_score:>+4.0f}")

print(f"\n  Teammate score changes from extreme-DC adjustment: {tm_changes}")

# Save
wr.to_csv('data/wr_backtest_all_components.csv', index=False)
print(f"\n  Saved data/wr_backtest_all_components.csv")

# Also update wr_teammate_scores.csv peak_dominator for consistency
for idx, row in tm.iterrows():
    key = (row['player_name'], int(row['draft_year']))
    if key in career_peaks:
        tm.at[idx, 'peak_dominator'] = career_peaks[key]

tm.to_csv('data/wr_teammate_scores.csv', index=False)
print(f"  Saved data/wr_teammate_scores.csv (peak_dominator updated for consistency)")
