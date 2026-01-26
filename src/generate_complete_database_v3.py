"""
SLAP Score V3 - Complete Database Generator
============================================

Generates comprehensive CSV with all players (2015-2026) including:
- All input scores (DC, Production, RAS)
- Output scores (SLAP, tiers, deltas)
- NFL outcomes (for backtest players)

DC Formula: DC = 100 - 2.40 × (pick^0.62 - 1)
Weights: DC (50%) + Production (35%) + RAS (15%)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("SLAP SCORE V3 - COMPLETE DATABASE GENERATOR")
print("=" * 100)
print("\nDC Formula: DC = 100 - 2.40 × (pick^0.62 - 1)")
print("Weights: DC (50%) + Production (35%) + RAS (15%)")

# ============================================================================
# CONFIGURATION
# ============================================================================
WEIGHT_DC = 0.50
WEIGHT_PRODUCTION = 0.35
WEIGHT_RAS = 0.15

# Breakout age scoring for WRs
BREAKOUT_AGE_SCORES = {
    18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20
}

# Position averages
WR_AVG_BREAKOUT = 79.7
WR_AVG_RAS = 68.9
RB_AVG_PRODUCTION = 30.0
RB_AVG_RAS = 66.5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_draft_capital(pick):
    """New DC formula: DC = 100 - 2.40 × (pick^0.62 - 1)"""
    if pd.isna(pick) or pick <= 0:
        return None
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))

def breakout_age_to_score(age):
    """Convert breakout age to 0-100 score"""
    if pd.isna(age):
        return None
    return BREAKOUT_AGE_SCORES.get(int(age), 25)

def rb_production_score(rec_yards, team_pass_att):
    """Calculate RB receiving production score (normalized)"""
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return None
    ratio = rec_yards / team_pass_att
    score = (ratio / 1.0) * 100
    return min(100, max(0, score))

def get_slap_tier(score):
    """Assign tier based on SLAP score"""
    if pd.isna(score):
        return None
    if score >= 90: return "Elite (90+)"
    if score >= 80: return "Great (80-89)"
    if score >= 70: return "Good (70-79)"
    if score >= 60: return "Average (60-69)"
    if score >= 50: return "Below Avg (50-59)"
    return "Poor (<50)"

def get_delta_tier(delta):
    """Assign tier based on delta"""
    if pd.isna(delta):
        return None
    if delta >= 15: return "Big Sleeper (+15)"
    if delta >= 5: return "Sleeper (+5 to +15)"
    if delta >= -5: return "Neutral (-5 to +5)"
    if delta >= -15: return "Bust Risk (-5 to -15)"
    return "Big Bust (<-15)"

def get_outcome_status(row):
    """Determine outcome status based on draft year and hits"""
    year = row.get('draft_year')
    if pd.isna(year):
        return "TBD"
    year = int(year)
    if year >= 2024:
        return "TBD"
    hit24 = row.get('hit24', row.get('nfl_hit24'))
    hit12 = row.get('hit12', row.get('nfl_hit12'))
    if pd.notna(hit24) and hit24 == 1:
        return "Hit"
    if pd.notna(hit12) and hit12 == 1:
        return "Hit"
    return "Miss"

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "=" * 100)
print("LOADING DATA")
print("=" * 100)

# WR backtest data
wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"WR backtest: {len(wr_backtest)} players")

# RB backtest data
rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
print(f"RB backtest: {len(rb_backtest)} players")

# 2026 prospects
prospects = pd.read_csv('data/prospects_final.csv')
wr_2026 = prospects[prospects['position'] == 'WR'].copy()
rb_2026 = prospects[prospects['position'] == 'RB'].copy()
print(f"2026 WR prospects: {len(wr_2026)}")
print(f"2026 RB prospects: {len(rb_2026)}")

# 2026 WR breakout ages (researched)
WR_2026_BREAKOUT = {
    'Carnell Tate': 19, 'Jordyn Tyson': 20, 'Makai Lemon': 20, 'Denzel Boston': 20,
    'Kevin Concepcion': 19, 'Chris Bell': 21, 'Elijah Sarratt': 21, 'Zachariah Branch': 19,
    'Germie Bernard': 20, 'Chris Brazzell': 21, "Ja'Kobi Lane": 20, 'Omar Cooper Jr.': 21,
    'Antonio Williams': 21, 'Skyler Bell': 21, 'Malachi Fields': 21, 'C.J. Daniels': 20,
    'Brenen Thompson': 21, 'Deion Burks': 20, 'Ted Hurst': 21, 'Bryce Lance': 21,
    'Kevin Coleman Jr.': 20, 'Eric McAlister': 21, 'Eric Rivers': 21, 'Lewis Bond': 21,
    "De'Zhaun Stribling": 20, 'Keelan Marion': 20, 'Josh Cameron': 21, 'Noah Thomas': 19,
    'Aaron Anderson': 20, 'Dane Key': 20, 'Jordan Hudson': 20, 'Caleb Douglas': 20,
    'Reggie Virgil': 20, 'Vinny Anthony II': 21, 'Caullin Lacy': 21, 'Kendrick Law': 21,
    'Colbie Young': 21, 'Harrison Wallace III': 20, 'Jaden Greathouse': 19, 'Barion Brown': 19,
    'Amare Thomas': 18, 'Hykeem Williams': 20, 'Shelton Sampson Jr.': 19,
}

# Load 2026 RB production data if available
try:
    rb_2026_prod = pd.read_csv('output/slap_rb_2026.csv')
    rb_2026 = rb_2026.merge(
        rb_2026_prod[['player_name', 'production_score']],
        on='player_name', how='left'
    )
except:
    rb_2026['production_score'] = None

# ============================================================================
# PROCESS WR BACKTEST DATA
# ============================================================================
print("\n" + "=" * 100)
print("PROCESSING WR DATA")
print("=" * 100)

wr_rows = []

for _, row in wr_backtest.iterrows():
    pick = row['pick']
    dc = normalize_draft_capital(pick)
    breakout_age = row.get('breakout_age')
    breakout_score = breakout_age_to_score(breakout_age)
    breakout_final = breakout_score if pd.notna(breakout_score) else WR_AVG_BREAKOUT
    ras_raw = row.get('RAS')
    ras_score = ras_raw * 10 if pd.notna(ras_raw) else None
    ras_final = ras_score if pd.notna(ras_score) else WR_AVG_RAS
    ras_status = 'observed' if pd.notna(ras_score) else 'imputed'

    slap = WEIGHT_DC * dc + WEIGHT_PRODUCTION * breakout_final + WEIGHT_RAS * ras_final
    delta = slap - dc

    wr_rows.append({
        'player_name': row['player_name'],
        'position': 'WR',
        'draft_year': int(row['draft_year']),
        'pick': int(pick),
        'college': row.get('college', ''),
        'dc_score': round(dc, 1),
        'breakout_age': breakout_age if pd.notna(breakout_age) else None,
        'breakout_score': round(breakout_final, 1),
        'production_raw': None,
        'production_score': None,
        'ras_raw': round(ras_raw, 1) if pd.notna(ras_raw) else None,
        'ras_score': round(ras_final, 1),
        'ras_status': ras_status,
        'slap_score': round(slap, 1),
        'slap_tier': get_slap_tier(slap),
        'delta': round(delta, 1),
        'delta_tier': get_delta_tier(delta),
        'best_season_ppg': row.get('best_ppg', row.get('best_ppr')),
        'hit24': row.get('hit24'),
        'hit12': row.get('hit12'),
        'outcome_status': get_outcome_status(row),
        'data_source': 'backtest'
    })

# Process 2026 WRs
for _, row in wr_2026.iterrows():
    pick = row['projected_pick']
    dc = normalize_draft_capital(pick)
    breakout_age = WR_2026_BREAKOUT.get(row['player_name'])
    breakout_score = breakout_age_to_score(breakout_age)
    breakout_final = breakout_score if pd.notna(breakout_score) else WR_AVG_BREAKOUT
    ras_final = WR_AVG_RAS  # No combine data yet

    slap = WEIGHT_DC * dc + WEIGHT_PRODUCTION * breakout_final + WEIGHT_RAS * ras_final
    delta = slap - dc

    wr_rows.append({
        'player_name': row['player_name'],
        'position': 'WR',
        'draft_year': 2026,
        'pick': int(pick),
        'college': row.get('school', ''),
        'dc_score': round(dc, 1),
        'breakout_age': breakout_age,
        'breakout_score': round(breakout_final, 1),
        'production_raw': None,
        'production_score': None,
        'ras_raw': None,
        'ras_score': round(ras_final, 1),
        'ras_status': 'imputed',
        'slap_score': round(slap, 1),
        'slap_tier': get_slap_tier(slap),
        'delta': round(delta, 1),
        'delta_tier': get_delta_tier(delta),
        'best_season_ppg': None,
        'hit24': None,
        'hit12': None,
        'outcome_status': 'TBD',
        'data_source': 'prospect_2026'
    })

print(f"  Processed {len(wr_rows)} WRs total")

# ============================================================================
# PROCESS RB BACKTEST DATA
# ============================================================================
print("\n" + "=" * 100)
print("PROCESSING RB DATA")
print("=" * 100)

rb_rows = []

for _, row in rb_backtest.iterrows():
    pick = row['pick']
    dc = normalize_draft_capital(pick)
    rec_yards = row.get('rec_yards')
    team_pass_att = row.get('team_pass_att')
    production_raw = rec_yards / team_pass_att if pd.notna(rec_yards) and pd.notna(team_pass_att) and team_pass_att > 0 else None
    production_score = rb_production_score(rec_yards, team_pass_att)
    production_final = production_score if pd.notna(production_score) else RB_AVG_PRODUCTION
    ras_raw = row.get('RAS')
    ras_score = ras_raw * 10 if pd.notna(ras_raw) else None
    ras_final = ras_score if pd.notna(ras_score) else RB_AVG_RAS
    ras_status = 'observed' if pd.notna(ras_score) else 'imputed'

    slap = WEIGHT_DC * dc + WEIGHT_PRODUCTION * production_final + WEIGHT_RAS * ras_final
    delta = slap - dc

    rb_rows.append({
        'player_name': row['player_name'],
        'position': 'RB',
        'draft_year': int(row['draft_year']),
        'pick': int(pick),
        'college': row.get('college', ''),
        'dc_score': round(dc, 1),
        'breakout_age': None,
        'breakout_score': None,
        'production_raw': round(production_raw, 3) if pd.notna(production_raw) else None,
        'production_score': round(production_final, 1),
        'ras_raw': round(ras_raw, 1) if pd.notna(ras_raw) else None,
        'ras_score': round(ras_final, 1),
        'ras_status': ras_status,
        'slap_score': round(slap, 1),
        'slap_tier': get_slap_tier(slap),
        'delta': round(delta, 1),
        'delta_tier': get_delta_tier(delta),
        'best_season_ppg': row.get('best_ppg'),
        'hit24': row.get('hit24'),
        'hit12': row.get('hit12'),
        'outcome_status': get_outcome_status(row),
        'data_source': 'backtest'
    })

# Process 2026 RBs
for _, row in rb_2026.iterrows():
    pick = row['projected_pick']
    dc = normalize_draft_capital(pick)
    production_score = row.get('production_score')
    production_final = production_score if pd.notna(production_score) else RB_AVG_PRODUCTION
    ras_final = RB_AVG_RAS

    slap = WEIGHT_DC * dc + WEIGHT_PRODUCTION * production_final + WEIGHT_RAS * ras_final
    delta = slap - dc

    rb_rows.append({
        'player_name': row['player_name'],
        'position': 'RB',
        'draft_year': 2026,
        'pick': int(pick),
        'college': row.get('school', ''),
        'dc_score': round(dc, 1),
        'breakout_age': None,
        'breakout_score': None,
        'production_raw': None,
        'production_score': round(production_final, 1),
        'ras_raw': None,
        'ras_score': round(ras_final, 1),
        'ras_status': 'imputed',
        'slap_score': round(slap, 1),
        'slap_tier': get_slap_tier(slap),
        'delta': round(delta, 1),
        'delta_tier': get_delta_tier(delta),
        'best_season_ppg': None,
        'hit24': None,
        'hit12': None,
        'outcome_status': 'TBD',
        'data_source': 'prospect_2026'
    })

print(f"  Processed {len(rb_rows)} RBs total")

# ============================================================================
# COMBINE AND SAVE
# ============================================================================
print("\n" + "=" * 100)
print("SAVING DATABASE")
print("=" * 100)

all_players = pd.DataFrame(wr_rows + rb_rows)
all_players = all_players.sort_values(['draft_year', 'slap_score'], ascending=[True, False])
all_players.to_csv('output/slap_complete_database_v3.csv', index=False)
print(f"\nSaved: output/slap_complete_database_v3.csv ({len(all_players)} total players)")

# ============================================================================
# ANALYSIS 1: PLAYER COUNT BY POSITION AND YEAR
# ============================================================================
print("\n" + "=" * 100)
print("1. PLAYER COUNT BY POSITION AND YEAR")
print("=" * 100)

pivot = all_players.groupby(['draft_year', 'position']).size().unstack(fill_value=0)
pivot['Total'] = pivot.sum(axis=1)
print(f"\n{'Year':>6} {'WR':>6} {'RB':>6} {'Total':>7}")
print("-" * 30)
for year in sorted(pivot.index):
    wr_count = pivot.loc[year, 'WR'] if 'WR' in pivot.columns else 0
    rb_count = pivot.loc[year, 'RB'] if 'RB' in pivot.columns else 0
    print(f"{year:>6} {wr_count:>6} {rb_count:>6} {pivot.loc[year, 'Total']:>7}")
print("-" * 30)
print(f"{'TOTAL':>6} {all_players[all_players['position']=='WR'].shape[0]:>6} {all_players[all_players['position']=='RB'].shape[0]:>6} {len(all_players):>7}")

# ============================================================================
# ANALYSIS 2: SCORE DISTRIBUTION SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("2. SCORE DISTRIBUTION BY TIER")
print("=" * 100)

tier_order = ["Elite (90+)", "Great (80-89)", "Good (70-79)", "Average (60-69)", "Below Avg (50-59)", "Poor (<50)"]
tier_counts = all_players['slap_tier'].value_counts()

print(f"\n{'Tier':<22} {'Count':>8} {'Percentage':>12}")
print("-" * 45)
for tier in tier_order:
    count = tier_counts.get(tier, 0)
    pct = count / len(all_players) * 100
    print(f"{tier:<22} {count:>8} {pct:>11.1f}%")
print("-" * 45)
print(f"{'TOTAL':<22} {len(all_players):>8} {'100.0%':>12}")

# ============================================================================
# ANALYSIS 3: TOP 30 SLAP SCORES ALL-TIME
# ============================================================================
print("\n" + "=" * 100)
print("3. TOP 30 SLAP SCORES ALL-TIME (2015-2026)")
print("=" * 100)

top30 = all_players.nlargest(30, 'slap_score')
print(f"\n{'Rk':>3} {'Player':<25} {'Pos':>4} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>6} {'Prod':>6} {'RAS':>5} {'Delta':>7} {'Tier':<15}")
print("-" * 100)

for i, (_, row) in enumerate(top30.iterrows(), 1):
    prod = row['breakout_score'] if row['position'] == 'WR' else row['production_score']
    prod_str = f"{prod:.0f}" if pd.notna(prod) else "-"
    print(f"{i:>3} {row['player_name']:<25} {row['position']:>4} {row['draft_year']:>5} {row['pick']:>5} "
          f"{row['slap_score']:>6.1f} {row['dc_score']:>6.1f} {prod_str:>6} {row['ras_score']:>5.0f} "
          f"{row['delta']:>+7.1f} {row['slap_tier']:<15}")

# ============================================================================
# ANALYSIS 4: TOP 20 WRs FOR 2026 CLASS
# ============================================================================
print("\n" + "=" * 100)
print("4. TOP 20 WRS FOR 2026 CLASS (with all inputs)")
print("=" * 100)

wr_2026_df = all_players[(all_players['position'] == 'WR') & (all_players['draft_year'] == 2026)]
wr_2026_df = wr_2026_df.nlargest(20, 'slap_score')

print(f"\n{'Rk':>3} {'Player':<25} {'College':<18} {'Pick':>5} {'DC':>6} {'BO Age':>7} {'BO Sc':>6} {'RAS':>5} {'SLAP':>6} {'Delta':>7} {'Delta Tier':<18}")
print("-" * 125)

for i, (_, row) in enumerate(wr_2026_df.iterrows(), 1):
    bo_age = f"{int(row['breakout_age'])}" if pd.notna(row['breakout_age']) else "imp"
    print(f"{i:>3} {row['player_name']:<25} {row['college']:<18} {row['pick']:>5} {row['dc_score']:>6.1f} "
          f"{bo_age:>7} {row['breakout_score']:>6.0f} {row['ras_score']:>5.0f} {row['slap_score']:>6.1f} "
          f"{row['delta']:>+7.1f} {row['delta_tier']:<18}")

# ============================================================================
# ANALYSIS 5: TOP 20 RBs FOR 2026 CLASS
# ============================================================================
print("\n" + "=" * 100)
print("5. TOP 20 RBS FOR 2026 CLASS (with all inputs)")
print("=" * 100)

rb_2026_df = all_players[(all_players['position'] == 'RB') & (all_players['draft_year'] == 2026)]
rb_2026_df = rb_2026_df.nlargest(20, 'slap_score')

print(f"\n{'Rk':>3} {'Player':<25} {'College':<18} {'Pick':>5} {'DC':>6} {'Prod':>6} {'RAS':>5} {'SLAP':>6} {'Delta':>7} {'Delta Tier':<18}")
print("-" * 115)

for i, (_, row) in enumerate(rb_2026_df.iterrows(), 1):
    prod = f"{row['production_score']:.0f}" if pd.notna(row['production_score']) else "imp"
    print(f"{i:>3} {row['player_name']:<25} {row['college']:<18} {row['pick']:>5} {row['dc_score']:>6.1f} "
          f"{prod:>6} {row['ras_score']:>5.0f} {row['slap_score']:>6.1f} "
          f"{row['delta']:>+7.1f} {row['delta_tier']:<18}")

# ============================================================================
# ANALYSIS 6: BIGGEST POSITIVE DELTAS (SLEEPERS)
# ============================================================================
print("\n" + "=" * 100)
print("6. BIGGEST POSITIVE DELTAS - TOP 15 SLEEPERS")
print("=" * 100)

sleepers = all_players.nlargest(15, 'delta')
print(f"\n{'Rk':>3} {'Player':<25} {'Pos':>4} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>6} {'Delta':>7} {'Why (Input Scores)':<30}")
print("-" * 105)

for i, (_, row) in enumerate(sleepers.iterrows(), 1):
    if row['position'] == 'WR':
        why = f"BO={row['breakout_score']:.0f}, RAS={row['ras_score']:.0f}"
    else:
        why = f"Prod={row['production_score']:.0f}, RAS={row['ras_score']:.0f}"
    print(f"{i:>3} {row['player_name']:<25} {row['position']:>4} {row['draft_year']:>5} {row['pick']:>5} "
          f"{row['slap_score']:>6.1f} {row['dc_score']:>6.1f} {row['delta']:>+7.1f} {why:<30}")

# ============================================================================
# ANALYSIS 7: BIGGEST NEGATIVE DELTAS (BUST RISKS)
# ============================================================================
print("\n" + "=" * 100)
print("7. BIGGEST NEGATIVE DELTAS - TOP 15 BUST RISKS")
print("=" * 100)

busts = all_players.nsmallest(15, 'delta')
print(f"\n{'Rk':>3} {'Player':<25} {'Pos':>4} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>6} {'Delta':>7} {'Why (Input Scores)':<30}")
print("-" * 105)

for i, (_, row) in enumerate(busts.iterrows(), 1):
    if row['position'] == 'WR':
        why = f"BO={row['breakout_score']:.0f}, RAS={row['ras_score']:.0f}"
    else:
        why = f"Prod={row['production_score']:.0f}, RAS={row['ras_score']:.0f}"
    print(f"{i:>3} {row['player_name']:<25} {row['position']:>4} {row['draft_year']:>5} {row['pick']:>5} "
          f"{row['slap_score']:>6.1f} {row['dc_score']:>6.1f} {row['delta']:>+7.1f} {why:<30}")

# ============================================================================
# ANALYSIS 8: HIT RATE BY SLAP TIER (2015-2023 PLAYERS)
# ============================================================================
print("\n" + "=" * 100)
print("8. HIT RATE BY SLAP TIER (2015-2023 Players - Enough NFL Time)")
print("=" * 100)

# Filter to players with enough NFL time to evaluate
backtest_players = all_players[(all_players['draft_year'] >= 2015) & (all_players['draft_year'] <= 2023)]

print(f"\n{'Tier':<22} {'Total':>7} {'Hits':>6} {'Hit Rate':>10} {'Outcome':<25}")
print("-" * 75)

for tier in tier_order:
    tier_players = backtest_players[backtest_players['slap_tier'] == tier]
    total = len(tier_players)
    if total == 0:
        continue

    # Count hits (hit24 or hit12 = 1)
    hits = tier_players[(tier_players['hit24'] == 1) | (tier_players['hit12'] == 1)].shape[0]
    hit_rate = hits / total * 100 if total > 0 else 0

    # Show example players
    hit_examples = tier_players[(tier_players['hit24'] == 1) | (tier_players['hit12'] == 1)]['player_name'].head(3).tolist()
    example_str = ", ".join(hit_examples[:2]) if hit_examples else "None"

    print(f"{tier:<22} {total:>7} {hits:>6} {hit_rate:>9.1f}% {example_str:<25}")

# By position
print("\n--- BY POSITION ---")
for pos in ['WR', 'RB']:
    print(f"\n{pos} Hit Rates:")
    pos_players = backtest_players[backtest_players['position'] == pos]

    print(f"  {'Tier':<22} {'Total':>6} {'Hits':>5} {'Hit%':>7}")
    print(f"  " + "-" * 45)

    for tier in tier_order:
        tier_pos = pos_players[pos_players['slap_tier'] == tier]
        total = len(tier_pos)
        if total == 0:
            continue
        hits = tier_pos[(tier_pos['hit24'] == 1) | (tier_pos['hit12'] == 1)].shape[0]
        hit_rate = hits / total * 100 if total > 0 else 0
        print(f"  {tier:<22} {total:>6} {hits:>5} {hit_rate:>6.1f}%")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

print(f"\nTotal Players: {len(all_players)}")
print(f"  WRs: {len(all_players[all_players['position']=='WR'])}")
print(f"  RBs: {len(all_players[all_players['position']=='RB'])}")

print(f"\nSLAP Score Stats:")
print(f"  Mean: {all_players['slap_score'].mean():.1f}")
print(f"  Median: {all_players['slap_score'].median():.1f}")
print(f"  Std Dev: {all_players['slap_score'].std():.1f}")
print(f"  Min: {all_players['slap_score'].min():.1f}")
print(f"  Max: {all_players['slap_score'].max():.1f}")

print(f"\nDelta Stats:")
print(f"  Mean: {all_players['delta'].mean():.1f}")
print(f"  Std Dev: {all_players['delta'].std():.1f}")
print(f"  Min: {all_players['delta'].min():.1f} ({all_players.loc[all_players['delta'].idxmin(), 'player_name']})")
print(f"  Max: {all_players['delta'].max():.1f} ({all_players.loc[all_players['delta'].idxmax(), 'player_name']})")

print("\n" + "=" * 100)
print("DATABASE GENERATION COMPLETE!")
print("=" * 100)
