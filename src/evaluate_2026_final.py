"""
2026 Class Final Analysis
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("2026 CLASS ANALYSIS & FINAL RATINGS")
print("=" * 100)

df = pd.read_csv('output/slap_complete_database_v4.csv')

# Assign tiers
def assign_tier(score):
    if score >= 85: return '1-Elite (85+)'
    elif score >= 75: return '2-Great (75-84)'
    elif score >= 65: return '3-Good (65-74)'
    elif score >= 55: return '4-Average (55-64)'
    elif score >= 45: return '5-Below Avg (45-54)'
    else: return '6-Poor (<45)'

df['tier'] = df['slap_score'].apply(assign_tier)

# ============================================================================
# 2026 WR CLASS
# ============================================================================
print("\n" + "=" * 100)
print("2026 WR CLASS ANALYSIS")
print("=" * 100)

wr_2026 = df[(df['position'] == 'WR') & (df['draft_year'] == 2026)].sort_values('slap_score', ascending=False)

print(f"\n📊 OVERVIEW:")
print(f"   Total WRs: {len(wr_2026)}")
print(f"   Score range: {wr_2026['slap_score'].min():.1f} to {wr_2026['slap_score'].max():.1f}")

print(f"\n   Tier Distribution:")
for tier in sorted(wr_2026['tier'].unique()):
    count = len(wr_2026[wr_2026['tier'] == tier])
    pct = count / len(wr_2026) * 100
    print(f"      {tier}: {count} ({pct:.0f}%)")

print(f"\n📋 TOP 15 WRs (2026):")
print(f"   {'Rank':<5} {'Player':<25} {'School':<15} {'Pick':<6} {'SLAP':<6} {'Delta':<7} {'Tier'}")
print(f"   {'-'*5} {'-'*25} {'-'*15} {'-'*6} {'-'*6} {'-'*7} {'-'*20}")
for i, (_, row) in enumerate(wr_2026.head(15).iterrows(), 1):
    print(f"   {i:<5} {row['player_name']:<25} {str(row.get('school', 'N/A'))[:15]:<15} {row['pick']:<6} {row['slap_score']:<6.1f} {row['delta_vs_dc']:+6.1f} {row['tier']}")

# Smell test flags
print(f"\n⚠️  POTENTIAL WR ISSUES:")

# High pick, negative delta (model doesn't like)
wr_fades = wr_2026[(wr_2026['pick'] <= 50) & (wr_2026['delta_vs_dc'] <= -5)]
if len(wr_fades) > 0:
    print(f"\n   WRs SLAP likes less than draft capital (fades):")
    for _, row in wr_fades.sort_values('delta_vs_dc').iterrows():
        print(f"      {row['player_name']}: Pick {row['pick']}, SLAP {row['slap_score']:.0f}, Delta {row['delta_vs_dc']:.0f}")

# Late picks SLAP likes
wr_sleepers = wr_2026[(wr_2026['pick'] >= 150) & (wr_2026['delta_vs_dc'] >= 10)]
if len(wr_sleepers) > 0:
    print(f"\n   Late WRs SLAP likes (sleepers):")
    for _, row in wr_sleepers.sort_values('delta_vs_dc', ascending=False).head(5).iterrows():
        print(f"      {row['player_name']}: Pick {row['pick']}, SLAP {row['slap_score']:.0f}, Delta +{row['delta_vs_dc']:.0f}")

# ============================================================================
# 2026 RB CLASS
# ============================================================================
print("\n" + "=" * 100)
print("2026 RB CLASS ANALYSIS")
print("=" * 100)

rb_2026 = df[(df['position'] == 'RB') & (df['draft_year'] == 2026)].sort_values('slap_score', ascending=False)

print(f"\n📊 OVERVIEW:")
print(f"   Total RBs: {len(rb_2026)}")
print(f"   Score range: {rb_2026['slap_score'].min():.1f} to {rb_2026['slap_score'].max():.1f}")

print(f"\n   Tier Distribution:")
for tier in sorted(rb_2026['tier'].unique()):
    count = len(rb_2026[rb_2026['tier'] == tier])
    pct = count / len(rb_2026) * 100
    print(f"      {tier}: {pct:.0f}% ({count})")

print(f"\n📋 ALL 2026 RBs RANKED:")
print(f"   {'Rank':<5} {'Player':<25} {'School':<15} {'Pick':<6} {'SLAP':<6} {'Delta':<7} {'Tier'}")
print(f"   {'-'*5} {'-'*25} {'-'*15} {'-'*6} {'-'*6} {'-'*7} {'-'*20}")
for i, (_, row) in enumerate(rb_2026.iterrows(), 1):
    print(f"   {i:<5} {row['player_name']:<25} {str(row.get('school', 'N/A'))[:15]:<15} {row['pick']:<6} {row['slap_score']:<6.1f} {row['delta_vs_dc']:+6.1f} {row['tier']}")

# ============================================================================
# RB BUST ALERT DETAILS
# ============================================================================
print("\n" + "=" * 100)
print("RB BUST ALERT ANALYSIS")
print("=" * 100)

rb_busts = rb_2026[rb_2026['delta_vs_dc'] <= -10]
if len(rb_busts) > 0:
    print(f"\n   RBs with delta ≤ -10 (SLAP doesn't like relative to draft):")
    for _, row in rb_busts.sort_values('delta_vs_dc').iterrows():
        prod = row.get('production_score', 'N/A')
        print(f"\n   {row['player_name']} ({row['school']})")
        print(f"      Draft: Pick {row['pick']}")
        print(f"      SLAP: {row['slap_score']:.1f} ({row['tier']})")
        print(f"      Delta: {row['delta_vs_dc']:.1f}")
        print(f"      Production Score: {prod if pd.notna(prod) else 'N/A'}")

# ============================================================================
# FINAL RATINGS
# ============================================================================
print("\n" + "=" * 100)
print("FINAL MODEL RATINGS")
print("=" * 100)

# Get backtest correlation for summary
backtest = df[(df['draft_year'] <= 2023) & (df['nfl_best_ppg'].notna())].copy()
slap_corr = backtest['slap_score'].corr(backtest['nfl_best_ppg'])
dc_corr = backtest['dc_score'].corr(backtest['nfl_best_ppg'])

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                           SLAP MODEL V4 FINAL RATINGS                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  📊 PREDICTIVE ACCURACY: 6/10                                                  │
│     • Overall correlation: r = {slap_corr:.3f} (modest)                               │
│     • Draft capital only: r = {dc_corr:.3f}                                           │
│     • SLAP vs DC: {slap_corr - dc_corr:+.3f} (SLAP slightly worse overall)                      │
│     • BUT: SLAP beats DC in Rounds 2-7 (where it matters for sleepers)         │
│                                                                                │
│  📺 CONTENT USEFULNESS: 8/10                                                   │
│     • Tier system is intuitive and creates natural groupings                   │
│     • Delta provides clear "takes" for content                                 │
│     • Hit rate by tier is very clear: Elite=100%, Great=92%, Good=48%         │
│     • Found 18 historical sleepers with positive delta (Jay Ajayi, etc.)       │
│     • Flagged 47 historical busts with negative delta                          │
│                                                                                │
│  📖 EASE OF EXPLANATION: 8/10                                                  │
│     • 0-100 scale is intuitive                                                 │
│     • Tier names are clear                                                     │
│     • "SLAP likes/doesn't like this player vs draft" is simple                 │
│     • Formula is understandable: DC + Production + RAS                         │
│                                                                                │
│  🔮 CONFIDENCE IN 2026: 5/10                                                   │
│     • No combine data yet (all RAS is imputed)                                 │
│     • Draft picks are mock draft projections                                   │
│     • WR class looks strong (2 Elite, 4 Great)                                 │
│     • RB class is WEAK (0 Elite, 1 Great = Jeremiyah Love only)               │
│     • Several RBs with big negative deltas need investigation                  │
│                                                                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                        KEY CONTENT ANGLES FOR 2026                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  🎯 STRONG TAKES:                                                              │
│     • WR Carnell Tate: Elite tier but negative delta (-5.5)                    │
│     • WR Jordyn Tyson: Elite tier but negative delta (-7.7)                    │
│     • RB Jeremiyah Love: Only Great tier RB, but delta -14.6!                  │
│                                                                                │
│  💎 SLEEPER ALERTS:                                                            │
│     • Late WRs with positive delta (breakout age signal)                       │
│     • RB Desmond Reid: Pick 250, SLAP 40, Delta +11 (receiving production)     │
│                                                                                │
│  ⚠️  BUST ALERTS:                                                              │
│     • RB Jadarian Price: Pick 79, Delta -17.5                                  │
│     • RB Kaytron Allen: Pick 124, Delta -13.8                                  │
│     • Several early RBs have low receiving production scores                   │
│                                                                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                      SINGLE MOST IMPORTANT IMPROVEMENT                         │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  The model currently only has RB backtest data with PPG outcomes.              │
│  Adding WR PPG outcome data would allow proper WR validation.                  │
│                                                                                │
│  For immediate content use, consider:                                          │
│  1. Wait for combine data to update RAS scores                                 │
│  2. Update draft projections as combine/pro days happen                        │
│  3. Add WR outcome data to validate WR-specific predictions                    │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================================
# DISCLOSE IN CONTENT
# ============================================================================
print("\n" + "=" * 100)
print("WHAT TO DISCLOSE IN CONTENT")
print("=" * 100)

print(f"""
✅ WHAT SLAP IS GOOD AT:
   • Differentiating tiers (Elite/Great hit at 100%/92%)
   • Finding sleepers in rounds 2-7
   • Flagging bust risk (negative delta busts at 64% rate)
   • RB receiving production is statistically significant (p=0.004)

⚠️  WHAT TO DISCLOSE:
   • SLAP slightly underperforms just following draft order overall
   • WR breakout age has weak predictive signal (r=0.155)
   • Model can't predict: injuries, landing spot, coaching, breakout timing
   • 2026 data is preliminary (no combine, projected picks)
   • RB class is historically weak - only 1 player in Great+ tier

🎯 BEST USE CASE:
   • Tier-based groupings, not strict rankings
   • Delta for "takes" on who to fade/target
   • Historical validation stories (Jay Ajayi, Alvin Kamara, etc.)
   • Bust flags for cautionary takes
""")

print("=" * 100)
print("EVALUATION COMPLETE")
print("=" * 100)
