"""
Pull WR NFL Fantasy Data and Complete Validation - Final Version
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ============================================================================
# STEP 1: PULL DATA
# ============================================================================
print("=" * 100)
print("STEP 1: PULL WR NFL FANTASY DATA FROM NFLVERSE")
print("=" * 100)

df = pd.read_csv('output/slap_complete_database_v4.csv')
wr_backtest = df[(df['position'] == 'WR') & (df['draft_year'] >= 2015) & (df['draft_year'] <= 2023)].copy()
print(f"\nWRs in backtest: {len(wr_backtest)}")

print("\nFetching NFL fantasy data...")
all_seasons = []
for year in range(2015, 2025):
    try:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.csv"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            season_df = pd.read_csv(StringIO(response.text))
            season_df['season'] = year
            all_seasons.append(season_df)
            print(f"   {year}: OK")
    except Exception as e:
        print(f"   {year}: Error")

nfl_data = pd.concat(all_seasons, ignore_index=True)
nfl_wr = nfl_data[nfl_data['position'] == 'WR'].copy()

# Aggregate by player and season
season_totals = nfl_wr.groupby(['player_display_name', 'season']).agg({
    'fantasy_points_ppr': 'sum'
}).reset_index()

games = nfl_wr.groupby(['player_display_name', 'season']).size().reset_index(name='games')
season_totals = season_totals.merge(games, on=['player_display_name', 'season'])
season_totals = season_totals[season_totals['games'] >= 4]
season_totals['ppg'] = season_totals['fantasy_points_ppr'] / season_totals['games']

best_seasons = season_totals.loc[season_totals.groupby('player_display_name')['ppg'].idxmax()].copy()

def normalize_name(name):
    name = name.lower().strip()
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)$', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = ' '.join(name.split())
    return name

norm_lookup = {}
for _, row in best_seasons.iterrows():
    norm = normalize_name(row['player_display_name'])
    norm_lookup[norm] = row['ppg']

def find_ppg(player_name):
    norm = normalize_name(player_name)
    if norm in norm_lookup:
        return norm_lookup[norm]
    parts = player_name.split()
    if len(parts) > 2:
        short_name = f"{parts[0]} {parts[-1]}"
        norm = normalize_name(short_name)
        if norm in norm_lookup:
            return norm_lookup[norm]
    return None

wr_backtest['nfl_best_ppg_new'] = wr_backtest['player_name'].apply(find_ppg)

matched = wr_backtest['nfl_best_ppg_new'].notna().sum()
print(f"\n✓ Matched: {matched}/{len(wr_backtest)} ({matched/len(wr_backtest)*100:.1f}%)")

# ============================================================================
# STEP 2: VERIFY
# ============================================================================
print("\n" + "=" * 100)
print("STEP 2: VERIFY DATA")
print("=" * 100)

verify = [
    ("Ja'Marr Chase", "~23", lambda x: x >= 20),
    ("CeeDee Lamb", "~23", lambda x: x >= 20),
    ("Justin Jefferson", "~21", lambda x: x >= 18),
    ("Cooper Kupp", "~26", lambda x: x >= 22),
    ("N'Keal Harry", "~5", lambda x: x <= 8),
    ("Tyreek Hill", "~23", lambda x: x >= 20),
]

print(f"\n{'Player':<25} {'Expected':<12} {'Actual':<12} {'Status'}")
print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}")

for name, expected, check_fn in verify:
    ppg = find_ppg(name)
    if ppg:
        status = "✓" if check_fn(ppg) else "✗"
        print(f"{name:<25} {expected:<12} {ppg:.1f} PPG      {status}")
    else:
        # Check if in backtest
        in_bt = name in wr_backtest['player_name'].values
        print(f"{name:<25} {expected:<12} {'NOT FOUND':<12} {'(not in backtest)' if not in_bt else '✗'}")

# ============================================================================
# STEP 3: UPDATE DATABASE
# ============================================================================
print("\n" + "=" * 100)
print("STEP 3: UPDATE DATABASE")
print("=" * 100)

# Update main dataframe
df_updated = df.copy()

# Create lookup from backtest results
ppg_update = dict(zip(
    wr_backtest['player_name'] + '_' + wr_backtest['draft_year'].astype(str),
    wr_backtest['nfl_best_ppg_new']
))

# Apply updates
update_count = 0
for idx, row in df_updated.iterrows():
    if row['position'] == 'WR' and 2015 <= row['draft_year'] <= 2023:
        key = f"{row['player_name']}_{int(row['draft_year'])}"
        if key in ppg_update and pd.notna(ppg_update[key]):
            df_updated.at[idx, 'nfl_best_ppg'] = ppg_update[key]
            update_count += 1

# For unmatched WRs who never played, set PPG to 0
for idx, row in df_updated.iterrows():
    if row['position'] == 'WR' and 2015 <= row['draft_year'] <= 2023:
        if pd.isna(row['nfl_best_ppg']):
            df_updated.at[idx, 'nfl_best_ppg'] = 0.0
            update_count += 1

df_updated.to_csv('output/slap_complete_database_v4.csv', index=False)
print(f"\n✓ Updated {update_count} WRs in slap_complete_database_v4.csv")

# ============================================================================
# STEP 4: WR VALIDATION
# ============================================================================
print("\n" + "=" * 100)
print("STEP 4: WR VALIDATION ANALYSIS")
print("=" * 100)

# Reload
df = pd.read_csv('output/slap_complete_database_v4.csv')
wr_valid = df[(df['position'] == 'WR') & (df['draft_year'] >= 2015) & (df['draft_year'] <= 2023)].copy()

# Filter to those who actually played (PPG > 0) for correlation
wr_played = wr_valid[wr_valid['nfl_best_ppg'] > 0].copy()

print(f"\nWRs with any NFL production: {len(wr_played)}")
print(f"WRs who never played (PPG=0): {len(wr_valid) - len(wr_played)}")

# 4.1 Overall Correlation
print(f"\n📊 4.1 OVERALL CORRELATION (WRs who played)")
slap_corr = wr_played['slap_score'].corr(wr_played['nfl_best_ppg'])
dc_corr = wr_played['dc_score'].corr(wr_played['nfl_best_ppg'])

print(f"   SLAP vs PPG: r = {slap_corr:.3f}")
print(f"   DC vs PPG:   r = {dc_corr:.3f}")
print(f"   Difference:  {slap_corr - dc_corr:+.3f}")

if slap_corr > dc_corr:
    pct_better = (slap_corr - dc_corr) / dc_corr * 100
    print(f"   → SLAP beats DC by {pct_better:+.1f}%")
else:
    pct_worse = (slap_corr - dc_corr) / dc_corr * 100
    print(f"   → SLAP underperforms DC by {pct_worse:.1f}%")

# 4.2 By Draft Round
print(f"\n📊 4.2 CORRELATION BY DRAFT ROUND")
print(f"\n   {'Round':<12} {'N':<6} {'DC r':<10} {'SLAP r':<10} {'SLAP Wins?'}")
print(f"   {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*12}")

rounds = [
    (1, 32, 'Round 1'),
    (33, 64, 'Round 2'),
    (65, 128, 'Rounds 3-4'),
    (129, 262, 'Rounds 5-7')
]

for low, high, label in rounds:
    rd_data = wr_played[(wr_played['pick'] >= low) & (wr_played['pick'] <= high)]
    if len(rd_data) >= 8:
        slap_r = rd_data['slap_score'].corr(rd_data['nfl_best_ppg'])
        dc_r = rd_data['dc_score'].corr(rd_data['nfl_best_ppg'])
        diff = slap_r - dc_r
        wins = "Yes ✓" if diff > 0.03 else ("No ✗" if diff < -0.03 else "~Same")
        print(f"   {label:<12} {len(rd_data):<6} {dc_r:<10.3f} {slap_r:<10.3f} {wins}")
    else:
        print(f"   {label:<12} {len(rd_data):<6} {'N/A':<10} {'N/A':<10} {'Too few'}")

# 4.3 Hit Rate by SLAP Tier
print(f"\n📊 4.3 HIT RATE BY SLAP TIER")

def assign_tier(score):
    if score >= 90: return '1-Elite (90+)'
    elif score >= 80: return '2-Great (80-89)'
    elif score >= 70: return '3-Good (70-79)'
    elif score >= 60: return '4-Average (60-69)'
    elif score >= 50: return '5-Below Avg (50-59)'
    else: return '6-Poor (<50)'

wr_valid['tier'] = wr_valid['slap_score'].apply(assign_tier)

print(f"\n   {'Tier':<25} {'N':<6} {'Hit24 Rate':<12} {'Avg PPG'}")
print(f"   {'-'*25} {'-'*6} {'-'*12} {'-'*10}")

for tier in sorted(wr_valid['tier'].unique()):
    tier_data = wr_valid[wr_valid['tier'] == tier]
    hit_rate = tier_data['nfl_hit24'].mean() * 100
    avg_ppg = tier_data['nfl_best_ppg'].mean()
    print(f"   {tier:<25} {len(tier_data):<6} {hit_rate:<12.0f}% {avg_ppg:<10.1f}")

# ============================================================================
# STEP 5: COMPARE WR vs RB
# ============================================================================
print("\n" + "=" * 100)
print("STEP 5: WR vs RB MODEL COMPARISON")
print("=" * 100)

rb_valid = df[(df['position'] == 'RB') & (df['draft_year'] >= 2015) & (df['draft_year'] <= 2023) & (df['nfl_best_ppg'].notna())].copy()

# RB correlations
rb_slap_corr = rb_valid['slap_score'].corr(rb_valid['nfl_best_ppg'])
rb_dc_corr = rb_valid['dc_score'].corr(rb_valid['nfl_best_ppg'])

# Hit rates for top tier
rb_valid['tier'] = rb_valid['slap_score'].apply(assign_tier)
rb_top = rb_valid[rb_valid['slap_score'] >= 80]
rb_top_hit = rb_top['nfl_hit24'].mean() * 100 if len(rb_top) > 0 else 0

wr_top = wr_valid[wr_valid['slap_score'] >= 80]
wr_top_hit = wr_top['nfl_hit24'].mean() * 100 if len(wr_top) > 0 else 0

print(f"""
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          WR vs RB MODEL COMPARISON                              │
├─────────────────────────────┬─────────────────────┬─────────────────────────────┤
│ Metric                      │ WR Model            │ RB Model                    │
├─────────────────────────────┼─────────────────────┼─────────────────────────────┤
│ Sample Size (with PPG)      │ {len(wr_played):<19} │ {len(rb_valid):<27} │
│ SLAP vs PPG correlation     │ r = {slap_corr:<14.3f} │ r = {rb_slap_corr:<22.3f} │
│ DC vs PPG correlation       │ r = {dc_corr:<14.3f} │ r = {rb_dc_corr:<22.3f} │
│ SLAP improvement over DC    │ {slap_corr - dc_corr:+.3f}               │ {rb_slap_corr - rb_dc_corr:+.3f}                       │
│ Top tier (80+) hit24 rate   │ {wr_top_hit:<19.0f}%│ {rb_top_hit:<27.0f}%│
│ Top tier count              │ {len(wr_top):<19} │ {len(rb_top):<27} │
└─────────────────────────────┴─────────────────────┴─────────────────────────────┘
""")

# Final verdict
print("\n📋 FINAL VERDICT:")
print(f"\n   WR Model:")
if slap_corr > dc_corr:
    print(f"   ✓ SLAP beats DC: r={slap_corr:.3f} vs r={dc_corr:.3f} ({slap_corr-dc_corr:+.3f})")
else:
    print(f"   ✗ SLAP does NOT beat DC: r={slap_corr:.3f} vs r={dc_corr:.3f} ({slap_corr-dc_corr:+.3f})")

print(f"\n   RB Model:")
if rb_slap_corr > rb_dc_corr:
    print(f"   ✓ SLAP beats DC: r={rb_slap_corr:.3f} vs r={rb_dc_corr:.3f} ({rb_slap_corr-rb_dc_corr:+.3f})")
else:
    print(f"   ✗ SLAP does NOT beat DC: r={rb_slap_corr:.3f} vs r={rb_dc_corr:.3f} ({rb_slap_corr-rb_dc_corr:+.3f})")

print(f"\n   Tier Hit Rates (80+):")
print(f"   WRs: {wr_top_hit:.0f}% ({int(wr_top['nfl_hit24'].sum())}/{len(wr_top)})")
print(f"   RBs: {rb_top_hit:.0f}% ({int(rb_top['nfl_hit24'].sum())}/{len(rb_top)})")

print("\n" + "=" * 100)
print("VALIDATION COMPLETE")
print("=" * 100)
