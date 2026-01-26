"""
SLAP Model V4 Detailed Evaluation - Part 2
Focus on historical validation with correct column names.
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=" * 100)
print("SLAP MODEL V4 - DETAILED VALIDATION ANALYSIS")
print("=" * 100)

# Load the database
df = pd.read_csv('output/slap_complete_database_v4.csv')

# Check outcome columns
print("\n📊 OUTCOME COLUMNS AVAILABLE:")
outcome_cols = ['nfl_best_ppr', 'nfl_hit24', 'nfl_hit12', 'nfl_best_ppg']
for col in outcome_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"   {col}: {non_null} players with data")

# Use nfl_best_ppg for validation
ppg_col = 'nfl_best_ppg'

# Filter to backtest players with outcomes
backtest = df[(df['draft_year'] <= 2023) & (df[ppg_col].notna())].copy()
print(f"\n📊 BACKTEST DATA: {len(backtest)} players with NFL PPG data")

# Define hit thresholds
# For fantasy: 10+ PPG is a "hit" (roughly top 24-30 at position)
backtest['hit'] = backtest[ppg_col] >= 10
backtest['hit12'] = backtest[ppg_col] >= 12  # Elite

# Tier assignment
def assign_tier(score):
    if score >= 85: return '1-Elite (85+)'
    elif score >= 75: return '2-Great (75-84)'
    elif score >= 65: return '3-Good (65-74)'
    elif score >= 55: return '4-Average (55-64)'
    elif score >= 45: return '5-Below Avg (45-54)'
    else: return '6-Poor (<45)'

backtest['tier'] = backtest['slap_score'].apply(assign_tier)

# ============================================================================
# DELTA ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("DELTA USEFULNESS ANALYSIS")
print("=" * 100)

print(f"\n📈 DELTA DISTRIBUTION:")
print(f"   Range: {backtest['delta_vs_dc'].min():.1f} to {backtest['delta_vs_dc'].max():.1f}")
print(f"   Mean: {backtest['delta_vs_dc'].mean():.1f}")
print(f"   Std: {backtest['delta_vs_dc'].std():.1f}")
print(f"   25th percentile: {backtest['delta_vs_dc'].quantile(0.25):.1f}")
print(f"   75th percentile: {backtest['delta_vs_dc'].quantile(0.75):.1f}")

# Positive delta analysis (sleepers)
print(f"\n📊 POSITIVE DELTA ANALYSIS (Sleepers):")
for threshold in [3, 5, 8, 10]:
    sleepers = backtest[backtest['delta_vs_dc'] >= threshold]
    if len(sleepers) > 0:
        hit_rate = sleepers['hit'].mean() * 100
        avg_ppg = sleepers[ppg_col].mean()
        print(f"   Delta ≥+{threshold}: {len(sleepers)} players, {hit_rate:.0f}% hit rate, {avg_ppg:.1f} avg PPG")

# Negative delta analysis (bust flags)
print(f"\n📊 NEGATIVE DELTA ANALYSIS (Bust Flags):")
for threshold in [-3, -5, -8, -10]:
    busts = backtest[backtest['delta_vs_dc'] <= threshold]
    if len(busts) > 0:
        bust_rate = (1 - busts['hit'].mean()) * 100
        avg_ppg = busts[ppg_col].mean()
        print(f"   Delta ≤{threshold}: {len(busts)} players, {bust_rate:.0f}% bust rate, {avg_ppg:.1f} avg PPG")

# ============================================================================
# SLEEPER EXAMPLES (Positive Delta + Hit)
# ============================================================================
print("\n" + "=" * 100)
print("TOP SLEEPER FINDS (Positive Delta + NFL Hit)")
print("=" * 100)

sleeper_hits = backtest[(backtest['delta_vs_dc'] >= 3) & (backtest['hit'])].sort_values('delta_vs_dc', ascending=False)
print(f"\n   Found {len(sleeper_hits)} sleeper hits (delta ≥+3, PPG ≥10):\n")
print(f"   {'Player':<25} {'Pos':<4} {'Year':<5} {'Pick':<5} {'SLAP':<6} {'Delta':<7} {'PPG':<6}")
print(f"   {'-'*25} {'-'*4} {'-'*5} {'-'*5} {'-'*6} {'-'*7} {'-'*6}")
for _, row in sleeper_hits.head(15).iterrows():
    print(f"   {row['player_name']:<25} {row['position']:<4} {row['draft_year']:<5} {row['pick']:<5} {row['slap_score']:<6.1f} {row['delta_vs_dc']:+6.1f} {row[ppg_col]:<6.1f}")

# ============================================================================
# BUST FLAG EXAMPLES (Negative Delta + Bust)
# ============================================================================
print("\n" + "=" * 100)
print("TOP BUST FLAGS (Negative Delta + NFL Bust)")
print("=" * 100)

bust_flags = backtest[(backtest['delta_vs_dc'] <= -3) & (~backtest['hit'])].sort_values('delta_vs_dc')
print(f"\n   Found {len(bust_flags)} bust flags (delta ≤-3, PPG <10):\n")
print(f"   {'Player':<25} {'Pos':<4} {'Year':<5} {'Pick':<5} {'SLAP':<6} {'Delta':<7} {'PPG':<6}")
print(f"   {'-'*25} {'-'*4} {'-'*5} {'-'*5} {'-'*6} {'-'*7} {'-'*6}")
for _, row in bust_flags.head(15).iterrows():
    print(f"   {row['player_name']:<25} {row['position']:<4} {row['draft_year']:<5} {row['pick']:<5} {row['slap_score']:<6.1f} {row['delta_vs_dc']:+6.1f} {row[ppg_col]:<6.1f}")

# ============================================================================
# HIT RATE BY TIER
# ============================================================================
print("\n" + "=" * 100)
print("HIT RATE BY SLAP TIER")
print("=" * 100)

print(f"\n   {'Tier':<25} {'Players':<10} {'Hits':<8} {'Hit%':<8} {'Avg PPG':<10}")
print(f"   {'-'*25} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

for tier in sorted(backtest['tier'].unique()):
    tier_data = backtest[backtest['tier'] == tier]
    hits = tier_data['hit'].sum()
    hit_rate = tier_data['hit'].mean() * 100
    avg_ppg = tier_data[ppg_col].mean()
    print(f"   {tier:<25} {len(tier_data):<10} {hits:<8} {hit_rate:<8.0f}% {avg_ppg:<10.1f}")

# By position
for pos in ['WR', 'RB']:
    print(f"\n   {pos} Hit Rate by Tier:")
    pos_data = backtest[backtest['position'] == pos]
    for tier in sorted(pos_data['tier'].unique()):
        tier_data = pos_data[pos_data['tier'] == tier]
        if len(tier_data) >= 3:
            hits = tier_data['hit'].sum()
            hit_rate = tier_data['hit'].mean() * 100
            print(f"      {tier}: {hit_rate:.0f}% ({hits}/{len(tier_data)})")

# ============================================================================
# SLAP vs DRAFT CAPITAL COMPARISON
# ============================================================================
print("\n" + "=" * 100)
print("SLAP vs DRAFT CAPITAL BASELINE")
print("=" * 100)

# Overall correlation
slap_corr = backtest['slap_score'].corr(backtest[ppg_col])
dc_corr = backtest['dc_score'].corr(backtest[ppg_col])

print(f"\n   OVERALL CORRELATION WITH NFL PPG:")
print(f"   SLAP Score: r = {slap_corr:.3f}")
print(f"   DC Only:    r = {dc_corr:.3f}")
print(f"   Difference: {slap_corr - dc_corr:+.3f}")

if slap_corr > dc_corr:
    print(f"   → SLAP beats DC-only by {(slap_corr - dc_corr)/dc_corr*100:+.1f}%")
else:
    print(f"   → SLAP underperforms DC-only by {(slap_corr - dc_corr)/dc_corr*100:.1f}%")

# By position
print(f"\n   BY POSITION:")
for pos in ['WR', 'RB']:
    pos_data = backtest[backtest['position'] == pos]
    slap_r = pos_data['slap_score'].corr(pos_data[ppg_col])
    dc_r = pos_data['dc_score'].corr(pos_data[ppg_col])
    diff = slap_r - dc_r
    print(f"   {pos}: SLAP r={slap_r:.3f}, DC r={dc_r:.3f}, diff={diff:+.3f}")

# ============================================================================
# 5 BEST SLAP WINS
# ============================================================================
print("\n" + "=" * 100)
print("5 BEST SLAP WINS")
print("=" * 100)

print("\nThese are cases where SLAP correctly identified value the draft missed:\n")

# Case 1: High positive delta + big hit (found sleeper)
sleeper_wins = backtest[(backtest['delta_vs_dc'] >= 5) & (backtest[ppg_col] >= 12)].copy()
sleeper_wins['value'] = sleeper_wins['delta_vs_dc'] * sleeper_wins[ppg_col]

# Case 2: Negative delta + bust (correctly flagged)
bust_wins = backtest[(backtest['delta_vs_dc'] <= -5) & (backtest[ppg_col] < 8) & (backtest['pick'] <= 100)].copy()

wins = []
for _, row in sleeper_wins.nlargest(5, 'delta_vs_dc').iterrows():
    wins.append({
        'name': row['player_name'],
        'pos': row['position'],
        'year': row['draft_year'],
        'pick': row['pick'],
        'slap': row['slap_score'],
        'delta': row['delta_vs_dc'],
        'ppg': row[ppg_col],
        'type': 'Sleeper Found'
    })

for _, row in bust_wins.nsmallest(5, 'delta_vs_dc').iterrows():
    wins.append({
        'name': row['player_name'],
        'pos': row['position'],
        'year': row['draft_year'],
        'pick': row['pick'],
        'slap': row['slap_score'],
        'delta': row['delta_vs_dc'],
        'ppg': row[ppg_col],
        'type': 'Bust Flagged'
    })

# Sort by impact
wins_sorted = sorted(wins, key=lambda x: abs(x['delta']) * (x['ppg'] if 'Sleeper' in x['type'] else (15 - x['ppg'])), reverse=True)

print(f"   {'Player':<25} {'Pos':<4} {'Pick':<5} {'SLAP':<6} {'Delta':<7} {'PPG':<6} {'Win Type'}")
print(f"   {'-'*25} {'-'*4} {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*15}")
seen = set()
count = 0
for w in wins_sorted:
    if w['name'] not in seen and count < 5:
        print(f"   {w['name']:<25} {w['pos']:<4} {w['pick']:<5} {w['slap']:<6.0f} {w['delta']:+6.0f} {w['ppg']:<6.1f} {w['type']}")
        seen.add(w['name'])
        count += 1

# ============================================================================
# 5 WORST SLAP MISSES
# ============================================================================
print("\n" + "=" * 100)
print("5 WORST SLAP MISSES")
print("=" * 100)

print("\nThese are cases where SLAP got it wrong:\n")

# Case 1: High SLAP + bust
high_slap_busts = backtest[(backtest['slap_score'] >= 70) & (backtest[ppg_col] < 8)].copy()

# Case 2: Low SLAP + big hit (missed sleeper)
missed_sleepers = backtest[(backtest['slap_score'] < 55) & (backtest[ppg_col] >= 12)].copy()

# Case 3: Negative delta but hit (wrong fade)
wrong_fades = backtest[(backtest['delta_vs_dc'] <= -5) & (backtest[ppg_col] >= 10)].copy()

misses = []
for _, row in high_slap_busts.nlargest(5, 'slap_score').iterrows():
    misses.append({
        'name': row['player_name'],
        'pos': row['position'],
        'year': row['draft_year'],
        'pick': row['pick'],
        'slap': row['slap_score'],
        'delta': row['delta_vs_dc'],
        'ppg': row[ppg_col],
        'type': 'High SLAP → Bust'
    })

for _, row in missed_sleepers.nsmallest(5, 'slap_score').iterrows():
    misses.append({
        'name': row['player_name'],
        'pos': row['position'],
        'year': row['draft_year'],
        'pick': row['pick'],
        'slap': row['slap_score'],
        'delta': row['delta_vs_dc'],
        'ppg': row[ppg_col],
        'type': 'Low SLAP → Hit'
    })

for _, row in wrong_fades.nsmallest(3, 'delta_vs_dc').iterrows():
    misses.append({
        'name': row['player_name'],
        'pos': row['position'],
        'year': row['draft_year'],
        'pick': row['pick'],
        'slap': row['slap_score'],
        'delta': row['delta_vs_dc'],
        'ppg': row[ppg_col],
        'type': f'Wrong Fade'
    })

# Sort by severity
misses_sorted = sorted(misses, key=lambda x: x['slap'] if 'High' in x['type'] else (x['ppg'] if 'Low' in x['type'] else abs(x['delta'])), reverse=True)

print(f"   {'Player':<25} {'Pos':<4} {'Pick':<5} {'SLAP':<6} {'Delta':<7} {'PPG':<6} {'Miss Type'}")
print(f"   {'-'*25} {'-'*4} {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*18}")
seen = set()
count = 0
for m in misses_sorted:
    if m['name'] not in seen and count < 5:
        print(f"   {m['name']:<25} {m['pos']:<4} {m['pick']:<5} {m['slap']:<6.0f} {m['delta']:+6.0f} {m['ppg']:<6.1f} {m['type']}")
        seen.add(m['name'])
        count += 1

# ============================================================================
# ACCURACY BY DRAFT RANGE
# ============================================================================
print("\n" + "=" * 100)
print("ACCURACY BY DRAFT RANGE")
print("=" * 100)

ranges = [
    (1, 32, 'Round 1'),
    (33, 64, 'Round 2'),
    (65, 100, 'Round 3'),
    (101, 175, 'Rounds 4-5'),
    (176, 262, 'Rounds 6-7')
]

print(f"\n   {'Range':<15} {'N':<5} {'Hit%':<8} {'SLAP r':<10} {'DC r':<10} {'SLAP vs DC'}")
print(f"   {'-'*15} {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")

for low, high, label in ranges:
    range_data = backtest[(backtest['pick'] >= low) & (backtest['pick'] <= high)]
    if len(range_data) >= 10:
        hit_rate = range_data['hit'].mean() * 100
        slap_r = range_data['slap_score'].corr(range_data[ppg_col])
        dc_r = range_data['dc_score'].corr(range_data[ppg_col])
        diff = slap_r - dc_r
        verdict = "Better" if diff > 0.02 else ("Worse" if diff < -0.02 else "Same")
        print(f"   {label:<15} {len(range_data):<5} {hit_rate:<8.0f}% {slap_r:<10.3f} {dc_r:<10.3f} {verdict}")

# ============================================================================
# 2026 CLASS ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("2026 CLASS SMELL TEST")
print("=" * 100)

wr_2026 = df[(df['position'] == 'WR') & (df['draft_year'] == 2026)].sort_values('slap_score', ascending=False)
rb_2026 = df[(df['position'] == 'RB') & (df['draft_year'] == 2026)].sort_values('slap_score', ascending=False)

print(f"\n📊 2026 WR CLASS:")
print(f"   Total: {len(wr_2026)} WRs")
print(f"   Score range: {wr_2026['slap_score'].min():.1f} to {wr_2026['slap_score'].max():.1f}")
print(f"   Elite tier: {len(wr_2026[wr_2026['tier'] == '1-Elite (85+)'])} players")
print(f"   Poor tier: {len(wr_2026[wr_2026['tier'] == '6-Poor (<45)'])} players")

print(f"\n   TOP 5 WRs:")
for i, (_, row) in enumerate(wr_2026.head(5).iterrows(), 1):
    pick = row.get('projected_pick', row.get('pick', 'N/A'))
    school = row.get('school', 'N/A')
    print(f"   {i}. {row['player_name']} ({school}): SLAP {row['slap_score']:.0f}, Pick {pick}")

print(f"\n📊 2026 RB CLASS:")
print(f"   Total: {len(rb_2026)} RBs")
print(f"   Score range: {rb_2026['slap_score'].min():.1f} to {rb_2026['slap_score'].max():.1f}")
print(f"   Elite tier: {len(rb_2026[rb_2026['tier'] == '1-Elite (85+)'])} players")
print(f"   Great tier: {len(rb_2026[rb_2026['tier'] == '2-Great (75-84)'])} players")

print(f"\n   TOP 5 RBs:")
for i, (_, row) in enumerate(rb_2026.head(5).iterrows(), 1):
    pick = row.get('projected_pick', row.get('pick', 'N/A'))
    school = row.get('school', 'N/A')
    print(f"   {i}. {row['player_name']} ({school}): SLAP {row['slap_score']:.0f}, Pick {pick}")

# Issues to flag
print(f"\n⚠️  POTENTIAL ISSUES TO REVIEW:")

# RBs with negative delta (model doesn't like them)
rb_flags = rb_2026[rb_2026['delta_vs_dc'] <= -10]
if len(rb_flags) > 0:
    print(f"\n   RBs with big negative deltas (SLAP likes less than draft):")
    for _, row in rb_flags.iterrows():
        print(f"      {row['player_name']}: Pick {row['pick']}, Delta {row['delta_vs_dc']:.0f}")

# Late round WRs SLAP likes
wr_sleepers = wr_2026[(wr_2026['pick'] >= 150) & (wr_2026['delta_vs_dc'] >= 10)]
if len(wr_sleepers) > 0:
    print(f"\n   Late WRs with big positive deltas (potential sleepers):")
    for _, row in wr_sleepers.iterrows():
        print(f"      {row['player_name']}: Pick {row['pick']}, SLAP {row['slap_score']:.0f}, Delta +{row['delta_vs_dc']:.0f}")

# ============================================================================
# FINAL RATINGS
# ============================================================================
print("\n" + "=" * 100)
print("FINAL RATINGS SUMMARY")
print("=" * 100)

print(f"""
📊 PREDICTIVE ACCURACY: 6/10
   • Overall r = {slap_corr:.3f} (modest)
   • {'Beats' if slap_corr > dc_corr else 'Does NOT beat'} DC-only baseline
   • WR model: r = {backtest[backtest['position']=='WR']['slap_score'].corr(backtest[backtest['position']=='WR'][ppg_col]):.3f}
   • RB model: r = {backtest[backtest['position']=='RB']['slap_score'].corr(backtest[backtest['position']=='RB'][ppg_col]):.3f}

📺 CONTENT USEFULNESS: 7/10
   • Tier system creates clear groups
   • Delta provides "takes" for content
   • {len(sleeper_hits)} historical sleeper finds to reference
   • {len(bust_flags)} bust flags correctly identified

📖 EASE OF EXPLANATION: 8/10
   • 0-100 scale is intuitive
   • Tier names are clear (Elite, Great, Good, Average, Below Avg, Poor)
   • Delta explains: "SLAP likes/dislikes this player vs where they're drafted"
   • Formula components are understandable

🔮 2026 CONFIDENCE: 6/10
   • No combine data yet (RAS imputed)
   • Draft picks are projections, will change
   • WR rankings look reasonable
   • RB class is weak (only 1 Great tier player)
""")

print("=" * 100)
print("EVALUATION COMPLETE")
print("=" * 100)
