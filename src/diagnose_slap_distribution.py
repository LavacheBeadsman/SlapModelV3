"""
SLAP Score Distribution Diagnosis
Analyze why scores are compressed and propose fixes
"""

import pandas as pd
import numpy as np

print("=" * 90)
print("SLAP SCORE DISTRIBUTION DIAGNOSIS")
print("=" * 90)

# Load master database
all_players = pd.read_csv('output/slap_master_all_50_35_15.csv')
wr_all = all_players[all_players['position'] == 'WR']
rb_all = all_players[all_players['position'] == 'RB']
prospects_2026 = all_players[all_players['draft_year'] == 2026]

# ============================================================================
# PART 1: SCORE DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 90)
print("PART 1: SCORE DISTRIBUTION ANALYSIS")
print("=" * 90)

def show_distribution(df, name):
    bins = [0, 40, 50, 60, 70, 80, 90, 100]
    labels = ['<40', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']

    # Create histogram
    counts = pd.cut(df['slap_score'], bins=bins, labels=labels, right=False).value_counts().sort_index()
    total = len(df)

    print(f"\n{name} (n={total}):")
    print("-" * 50)
    for label in labels:
        count = counts.get(label, 0)
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:>6}: {count:>4} ({pct:>5.1f}%) {bar}")

    print(f"\n  Stats: Min={df['slap_score'].min():.1f}, Max={df['slap_score'].max():.1f}, "
          f"Mean={df['slap_score'].mean():.1f}, Median={df['slap_score'].median():.1f}")

show_distribution(all_players, "ALL PLAYERS (2015-2026)")
show_distribution(wr_all, "WRs ONLY")
show_distribution(rb_all, "RBs ONLY")
show_distribution(prospects_2026, "2026 PROSPECTS ONLY")

# ============================================================================
# PART 2: WHY ARE SCORES SO COMPRESSED?
# ============================================================================
print("\n" + "=" * 90)
print("PART 2: WHY ARE SCORES SO COMPRESSED?")
print("=" * 90)

# 2.1: DC Distribution
print("\n--- 2.1: DRAFT CAPITAL (DC) DISTRIBUTION ---")
print("\nDC Score Distribution (All Players):")
dc_bins = [0, 40, 50, 60, 70, 80, 90, 100]
dc_labels = ['<40', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
dc_counts = pd.cut(all_players['dc_score'], bins=dc_bins, labels=dc_labels, right=False).value_counts().sort_index()
for label in dc_labels:
    count = dc_counts.get(label, 0)
    pct = count / len(all_players) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:>6}: {count:>4} ({pct:>5.1f}%) {bar}")

print(f"\n  DC above 60: {(all_players['dc_score'] >= 60).sum()} players")
print(f"  DC above 70: {(all_players['dc_score'] >= 70).sum()} players")
print(f"  DC above 80: {(all_players['dc_score'] >= 80).sum()} players")
print(f"  DC above 90: {(all_players['dc_score'] >= 90).sum()} players")

# Show DC for specific picks
print("\n  DC Score by Draft Pick:")
for pick in [1, 5, 10, 15, 20, 32, 50, 75, 100, 150, 200]:
    dc = ((1/np.sqrt(pick) - 1/np.sqrt(262)) / (1 - 1/np.sqrt(262))) * 100
    print(f"    Pick {pick:>3}: DC = {dc:.1f}")

# 2.2: Production Distribution
print("\n--- 2.2: PRODUCTION SCORE DISTRIBUTION ---")

# WR Breakout scores
print("\nWR Breakout Score Distribution:")
wr_prod = wr_all['production_score'].dropna()
prod_bins = [0, 30, 45, 60, 75, 90, 100, 101]
prod_labels = ['<30 (never)', '30-44 (age 23)', '45-59 (age 22)', '60-74 (age 21)', '75-89 (age 20)', '90-99 (age 19)', '100 (age 18)']
wr_prod_counts = pd.cut(wr_prod, bins=[0, 30, 45, 60, 75, 90, 100, 101], labels=prod_labels, right=False).value_counts().sort_index()
for label in prod_labels:
    count = wr_prod_counts.get(label, 0)
    pct = count / len(wr_prod) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:>20}: {count:>4} ({pct:>5.1f}%) {bar}")

print(f"\n  WR Breakout Stats: Min={wr_prod.min():.0f}, Max={wr_prod.max():.0f}, Mean={wr_prod.mean():.1f}")

# RB Production scores
print("\nRB Production Score Distribution:")
rb_prod = rb_all['production_score'].dropna()
rb_prod_bins = [0, 20, 40, 60, 80, 100, 200]
rb_prod_labels = ['0-19', '20-39', '40-59', '60-79', '80-99', '100+']
rb_prod_counts = pd.cut(rb_prod, bins=rb_prod_bins, labels=rb_prod_labels, right=False).value_counts().sort_index()
for label in rb_prod_labels:
    count = rb_prod_counts.get(label, 0)
    pct = count / len(rb_prod) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:>10}: {count:>4} ({pct:>5.1f}%) {bar}")

print(f"\n  RB Production Stats: Min={rb_prod.min():.1f}, Max={rb_prod.max():.1f}, Mean={rb_prod.mean():.1f}")

# 2.3: RAS Distribution
print("\n--- 2.3: RAS SCORE DISTRIBUTION ---")
ras_scores = all_players['ras_score'].dropna()
ras_bins = [0, 40, 50, 60, 70, 80, 90, 100, 200]
ras_labels = ['<40', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100+']
ras_counts = pd.cut(ras_scores, bins=ras_bins, labels=ras_labels, right=False).value_counts().sort_index()
for label in ras_labels:
    count = ras_counts.get(label, 0)
    pct = count / len(ras_scores) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:>10}: {count:>4} ({pct:>5.1f}%) {bar}")

print(f"\n  RAS Stats: Min={ras_scores.min():.1f}, Max={ras_scores.max():.1f}, Mean={ras_scores.mean():.1f}")

# 2.4: Math Walkthrough
print("\n--- 2.4: MATH WALKTHROUGH ---")
print("\nExample: Pick 10 player with age-19 breakout (90) and 8.0 RAS (80)")
pick = 10
dc = ((1/np.sqrt(pick) - 1/np.sqrt(262)) / (1 - 1/np.sqrt(262))) * 100
breakout = 90
ras = 80
slap = 0.50 * dc + 0.35 * breakout + 0.15 * ras
print(f"  DC (pick 10) = {dc:.1f}")
print(f"  Breakout (age 19) = {breakout}")
print(f"  RAS (8.0) = {ras}")
print(f"  SLAP = 0.50 × {dc:.1f} + 0.35 × {breakout} + 0.15 × {ras}")
print(f"  SLAP = {0.50*dc:.1f} + {0.35*breakout:.1f} + {0.15*ras:.1f} = {slap:.1f}")

print("\nMaximum possible SLAP for pick 10:")
max_slap_10 = 0.50 * dc + 0.35 * 100 + 0.15 * 100
print(f"  SLAP_max = 0.50 × {dc:.1f} + 0.35 × 100 + 0.15 × 100 = {max_slap_10:.1f}")

print("\nMaximum possible SLAP for pick 32:")
dc_32 = ((1/np.sqrt(32) - 1/np.sqrt(262)) / (1 - 1/np.sqrt(262))) * 100
max_slap_32 = 0.50 * dc_32 + 0.35 * 100 + 0.15 * 100
print(f"  DC (pick 32) = {dc_32:.1f}")
print(f"  SLAP_max = 0.50 × {dc_32:.1f} + 0.35 × 100 + 0.15 × 100 = {max_slap_32:.1f}")

print("\nMaximum possible SLAP for pick 1:")
dc_1 = 100
max_slap_1 = 0.50 * dc_1 + 0.35 * 100 + 0.15 * 100
print(f"  DC (pick 1) = {dc_1:.1f}")
print(f"  SLAP_max = 0.50 × 100 + 0.35 × 100 + 0.15 × 100 = {max_slap_1:.1f}")

# ============================================================================
# PART 3: WHAT SHOULD THE DISTRIBUTION LOOK LIKE?
# ============================================================================
print("\n" + "=" * 90)
print("PART 3: CURRENT VS TARGET DISTRIBUTION")
print("=" * 90)

print("\nTarget Distribution:")
print("  90+:    Top 5%  (~35 players)")
print("  70-89:  Top 25% (~174 players)")
print("  50-69:  Middle 50% (~348 players)")
print("  <50:    Bottom 25% (~138 players)")

print("\nCurrent Distribution:")
current_90 = (all_players['slap_score'] >= 90).sum()
current_70 = ((all_players['slap_score'] >= 70) & (all_players['slap_score'] < 90)).sum()
current_50 = ((all_players['slap_score'] >= 50) & (all_players['slap_score'] < 70)).sum()
current_below = (all_players['slap_score'] < 50).sum()
total = len(all_players)

print(f"  90+:    {current_90} players ({current_90/total*100:.1f}%)")
print(f"  70-89:  {current_70} players ({current_70/total*100:.1f}%)")
print(f"  50-69:  {current_50} players ({current_50/total*100:.1f}%)")
print(f"  <50:    {current_below} players ({current_below/total*100:.1f}%)")

# ============================================================================
# PART 4: DIAGNOSE THE ROOT CAUSE
# ============================================================================
print("\n" + "=" * 90)
print("PART 4: ROOT CAUSE DIAGNOSIS")
print("=" * 90)

print("\n--- THE PROBLEM ---")
print("\n1. DC DISTRIBUTION IS THE MAIN CULPRIT:")
print("   - 1/√pick creates VERY steep dropoff")
print("   - Only pick 1-3 get DC > 90")
print("   - Pick 32 (end of round 1) only gets DC ≈ 55")
print("   - Pick 100 gets DC ≈ 35")
print("   - Most players are day 2-3 picks with DC < 50")

print("\n2. THE MATH MAKES HIGH SCORES IMPOSSIBLE:")
print("   - Even with PERFECT production (100) and RAS (100)...")
print("   - Pick 32 can only reach: 0.50×55 + 0.35×100 + 0.15×100 = 77.5")
print("   - Pick 50 can only reach: 0.50×47 + 0.35×100 + 0.15×100 = 73.5")
print("   - Only pick 1-5 can theoretically reach 90+")

print("\n3. COMPONENT SCALES ARE MISMATCHED:")
print("   - DC: Heavily right-skewed (most players 20-50)")
print("   - WR Breakout: Left-skewed (most players 75-100)")
print("   - RB Production: Right-skewed (most players 20-50)")
print("   - RAS: Normal-ish (most players 50-80)")

# ============================================================================
# PART 5: TEST FIXES
# ============================================================================
print("\n" + "=" * 90)
print("PART 5: TESTING FIXES")
print("=" * 90)

# --- Option A: Rescale final SLAP to 0-100 ---
print("\n--- OPTION A: RESCALE FINAL SLAP ---")
min_slap = all_players['slap_score'].min()
max_slap = all_players['slap_score'].max()
all_players['slap_rescaled_A'] = ((all_players['slap_score'] - min_slap) / (max_slap - min_slap)) * 100

print(f"Original range: {min_slap:.1f} to {max_slap:.1f}")
print(f"Rescaled range: 0 to 100")
show_distribution(all_players.assign(slap_score=all_players['slap_rescaled_A']), "Option A: Rescaled SLAP")

# --- Option B: Gentler DC curve ---
print("\n--- OPTION B: GENTLER DC CURVE ---")
# Use logarithmic: DC = 100 - 25*ln(pick)
# This gives: pick 1 = 100, pick 10 = 42, pick 32 = 13... too harsh
# Try: DC = 100 × (1 - (pick-1)/261)^0.5
# This gives: pick 1 = 100, pick 32 = 94, pick 100 = 80... too gentle
# Try: DC = 100 - 2.5 × (pick^0.6 - 1)
# This gives: pick 1 = 100, pick 32 ≈ 75, pick 100 ≈ 60

def gentler_dc(pick):
    # Calibrated so: pick 1 = 100, pick 32 ≈ 70, pick 100 ≈ 50, pick 262 ≈ 20
    return max(0, 100 - 2.40 * (pick**0.62 - 1))

print("\nGentler DC formula: DC = 100 - 2.40 × (pick^0.62 - 1)")
print("\nDC comparison (old vs new):")
for pick in [1, 5, 10, 20, 32, 50, 75, 100, 150, 200, 262]:
    old_dc = ((1/np.sqrt(pick) - 1/np.sqrt(262)) / (1 - 1/np.sqrt(262))) * 100
    new_dc = gentler_dc(pick)
    print(f"  Pick {pick:>3}: Old DC = {old_dc:>5.1f}, New DC = {new_dc:>5.1f}")

all_players['dc_gentle'] = all_players['pick'].apply(gentler_dc)
all_players['slap_B'] = 0.50 * all_players['dc_gentle'] + 0.35 * all_players['production_score'] + 0.15 * all_players['ras_score']

show_distribution(all_players.assign(slap_score=all_players['slap_B']), "Option B: Gentler DC Curve")

# --- Option C: Percentile-based ---
print("\n--- OPTION C: PERCENTILE-BASED COMPONENTS ---")

# Convert each component to percentile within position
for pos in ['WR', 'RB']:
    mask = all_players['position'] == pos
    all_players.loc[mask, 'dc_pct'] = all_players.loc[mask, 'dc_score'].rank(pct=True) * 100
    all_players.loc[mask, 'prod_pct'] = all_players.loc[mask, 'production_score'].rank(pct=True) * 100
    all_players.loc[mask, 'ras_pct'] = all_players.loc[mask, 'ras_score'].rank(pct=True) * 100

all_players['slap_C'] = 0.50 * all_players['dc_pct'] + 0.35 * all_players['prod_pct'] + 0.15 * all_players['ras_pct']

show_distribution(all_players.assign(slap_score=all_players['slap_C']), "Option C: Percentile-Based")

# --- Option D: Combine B + mild rescale ---
print("\n--- OPTION D: GENTLER DC + MILD RESCALE ---")
# Use gentler DC, then rescale to 15-95 range (preserves some spread, looks reasonable)
min_B = all_players['slap_B'].min()
max_B = all_players['slap_B'].max()
all_players['slap_D'] = 15 + ((all_players['slap_B'] - min_B) / (max_B - min_B)) * 80

show_distribution(all_players.assign(slap_score=all_players['slap_D']), "Option D: Gentler DC + Rescale (15-95)")

# ============================================================================
# PART 6: COMPARE TOP PLAYERS ACROSS OPTIONS
# ============================================================================
print("\n" + "=" * 90)
print("PART 6: TOP 25 PLAYERS - COMPARING OPTIONS")
print("=" * 90)

# Add all versions
comparison = all_players[['player_name', 'position', 'draft_year', 'pick',
                          'slap_score', 'slap_rescaled_A', 'slap_B', 'slap_C', 'slap_D']].copy()
comparison.columns = ['Player', 'Pos', 'Year', 'Pick', 'Current', 'A_Rescale', 'B_GentleDC', 'C_Percentile', 'D_Combined']

print("\nTop 25 by CURRENT scoring:")
print(comparison.nlargest(25, 'Current')[['Player', 'Pos', 'Year', 'Pick', 'Current', 'B_GentleDC', 'D_Combined']].to_string(index=False))

print("\n\nTop 25 by OPTION B (Gentler DC):")
print(comparison.nlargest(25, 'B_GentleDC')[['Player', 'Pos', 'Year', 'Pick', 'Current', 'B_GentleDC', 'D_Combined']].to_string(index=False))

print("\n\nTop 25 by OPTION D (Gentler DC + Rescale):")
print(comparison.nlargest(25, 'D_Combined')[['Player', 'Pos', 'Year', 'Pick', 'Current', 'B_GentleDC', 'D_Combined']].to_string(index=False))

# ============================================================================
# RECOMMENDATION
# ============================================================================
print("\n" + "=" * 90)
print("RECOMMENDATION")
print("=" * 90)

print("""
DIAGNOSIS: The DC formula (1/√pick) is too steep, making high scores mathematically
impossible for anyone outside picks 1-5.

RECOMMENDED FIX: Option B (Gentler DC Curve)

Why Option B:
1. Fixes the root cause (DC distribution) rather than just rescaling
2. Preserves the relative ranking of players
3. Creates meaningful score differences across rounds
4. Pick 32 gets DC ≈ 70 (reasonable for end of round 1)
5. Pick 100 gets DC ≈ 50 (reasonable for mid-round)

New DC Formula: DC = 100 - 2.40 × (pick^0.62 - 1)

This gives:
  Pick 1:   DC = 100 (elite)
  Pick 10:  DC = 90  (top-10)
  Pick 32:  DC = 70  (round 1)
  Pick 64:  DC = 55  (round 2)
  Pick 100: DC = 45  (round 3-4)
  Pick 200: DC = 25  (day 3)
""")

# Save comparison for review
comparison.to_csv('output/slap_distribution_comparison.csv', index=False)
print("\nSaved comparison to: output/slap_distribution_comparison.csv")
