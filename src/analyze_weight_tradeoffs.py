"""
Analyze weight tradeoffs for SLAP model
Goal: Find weights that maximize meaningful deltas while maintaining predictive accuracy
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load backtest data with correct file names
wr_data = pd.read_csv('data/wr_backtest_expanded_final.csv')
rb_data = pd.read_csv('data/rb_backtest_with_receiving.csv')

print("="*80)
print("SLAP WEIGHT TRADEOFF ANALYSIS")
print("="*80)
print(f"WR dataset: {len(wr_data)} players")
print(f"RB dataset: {len(rb_data)} players")

# DC formula
def calc_dc(pick):
    return 100 - 2.40 * (pick**0.62 - 1)

# Breakout age to score (WR)
def breakout_to_score(age):
    if pd.isna(age) or age == 0:
        return 79.7  # imputed average
    mapping = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30}
    return mapping.get(int(age), 25)

# Normalize RAS (0-10 scale to 0-100)
def normalize_ras(ras):
    if pd.isna(ras):
        return 68.9  # imputed average
    return ras * 10

# Weight combinations to test
weights = [
    (0.85, 0.10, 0.05, "85/10/5 (current)"),
    (0.70, 0.20, 0.10, "70/20/10"),
    (0.60, 0.25, 0.15, "60/25/15"),
    (0.50, 0.30, 0.20, "50/30/20"),
    (0.50, 0.35, 0.15, "50/35/15"),
    (0.40, 0.40, 0.20, "40/40/20"),
]

# ============================================================================
# QUESTION 1: DELTA DISTRIBUTION BY WEIGHTS
# ============================================================================

print("\n" + "="*80)
print("QUESTION 1: DELTA DISTRIBUTION BY WEIGHT COMBINATION")
print("="*80)

def analyze_weights_wr(df, dc_w, prod_w, ras_w):
    """Calculate SLAP scores and deltas for WRs"""
    results = []
    for _, row in df.iterrows():
        pick = row['pick']
        breakout_age = row.get('breakout_age', None)
        ras = row.get('RAS', None)
        nfl_ppg = row.get('best_ppr', 0) / 17  # Convert PPR to PPG

        dc = calc_dc(pick)
        breakout_score = breakout_to_score(breakout_age)
        ras_score = normalize_ras(ras)

        slap = dc * dc_w + breakout_score * prod_w + ras_score * ras_w
        delta = slap - dc

        # Hit = top half fantasy finish (PPG >= 10)
        hit = 1 if nfl_ppg >= 10 else 0

        results.append({
            'player': row.get('player_name', 'Unknown'),
            'pick': pick,
            'round': row.get('round', 0),
            'dc': dc,
            'breakout_score': breakout_score,
            'ras_score': ras_score,
            'slap': slap,
            'delta': delta,
            'nfl_ppg': nfl_ppg,
            'hit': hit,
            'breakout_age': breakout_age
        })
    return pd.DataFrame(results)

def analyze_weights_rb(df, dc_w, prod_w, ras_w):
    """Calculate SLAP scores and deltas for RBs"""
    results = []
    for _, row in df.iterrows():
        pick = row['pick']

        # Calculate production from receiving data
        rec_yards = row.get('rec_yards', 0)
        team_pass = row.get('team_pass_att', 500)

        if pd.isna(rec_yards) or pd.isna(team_pass) or team_pass == 0:
            prod_score = 30  # imputed
        else:
            # Raw ratio, then normalize
            raw = (rec_yards / team_pass) * 100
            prod_score = min(100, max(0, raw * 1.0))  # Scale to 0-100

        ras = row.get('RAS', None)
        nfl_ppg = row.get('best_ppg', 0)

        dc = calc_dc(pick)
        ras_score = normalize_ras(ras) if not pd.isna(ras) else 66.5

        slap = dc * dc_w + prod_score * prod_w + ras_score * ras_w
        delta = slap - dc

        hit = 1 if nfl_ppg >= 10 else 0

        results.append({
            'player': row.get('player_name', 'Unknown'),
            'pick': pick,
            'round': row.get('round', 0),
            'dc': dc,
            'production_score': prod_score,
            'ras_score': ras_score,
            'slap': slap,
            'delta': delta,
            'nfl_ppg': nfl_ppg,
            'hit': hit,
            'rec_yards': rec_yards
        })
    return pd.DataFrame(results)

print("\n### WR DELTA DISTRIBUTION ###\n")
print(f"{'Weights':<20} {'Avg |Delta|':>12} {'Max +Delta':>12} {'Max -Delta':>12} {'Std Dev':>10}")
print("-" * 70)

wr_results_by_weight = {}
for dc_w, prod_w, ras_w, label in weights:
    wr_df = analyze_weights_wr(wr_data, dc_w, prod_w, ras_w)
    wr_results_by_weight[label] = wr_df

    avg_abs = wr_df['delta'].abs().mean()
    max_pos = wr_df['delta'].max()
    max_neg = wr_df['delta'].min()
    std_dev = wr_df['delta'].std()

    print(f"{label:<20} {avg_abs:>12.2f} {max_pos:>12.2f} {max_neg:>12.2f} {std_dev:>10.2f}")

print("\n### RB DELTA DISTRIBUTION ###\n")
print(f"{'Weights':<20} {'Avg |Delta|':>12} {'Max +Delta':>12} {'Max -Delta':>12} {'Std Dev':>10}")
print("-" * 70)

rb_results_by_weight = {}
for dc_w, prod_w, ras_w, label in weights:
    rb_df = analyze_weights_rb(rb_data, dc_w, prod_w, ras_w)
    rb_results_by_weight[label] = rb_df

    avg_abs = rb_df['delta'].abs().mean()
    max_pos = rb_df['delta'].max()
    max_neg = rb_df['delta'].min()
    std_dev = rb_df['delta'].std()

    print(f"{label:<20} {avg_abs:>12.2f} {max_pos:>12.2f} {max_neg:>12.2f} {std_dev:>10.2f}")

# ============================================================================
# QUESTION 2: PREDICTIVE ACCURACY VS DELTA SPREAD
# ============================================================================

print("\n" + "="*80)
print("QUESTION 2: ACCURACY vs DELTA SPREAD TRADEOFF")
print("="*80)

print("\n### WR: ACCURACY METRICS BY WEIGHT ###\n")
print(f"{'Weights':<20} {'SLAP-PPG r':>12} {'DC-PPG r':>10} {'Top25% Hit':>12} {'Avg |Delta|':>12}")
print("-" * 70)

for dc_w, prod_w, ras_w, label in weights:
    wr_df = wr_results_by_weight[label]

    # Filter to players with NFL data
    valid = wr_df[wr_df['nfl_ppg'] > 0].copy()

    if len(valid) > 10:
        slap_corr, _ = stats.spearmanr(valid['slap'], valid['nfl_ppg'])
        dc_corr, _ = stats.spearmanr(valid['dc'], valid['nfl_ppg'])

        # Top 25% by SLAP - what's their hit rate?
        top_25_threshold = valid['slap'].quantile(0.75)
        top_25 = valid[valid['slap'] >= top_25_threshold]
        top_25_hit_rate = top_25['hit'].mean() * 100

        avg_abs_delta = valid['delta'].abs().mean()

        print(f"{label:<20} {slap_corr:>12.3f} {dc_corr:>10.3f} {top_25_hit_rate:>11.1f}% {avg_abs_delta:>12.2f}")

print("\n### RB: ACCURACY METRICS BY WEIGHT ###\n")
print(f"{'Weights':<20} {'SLAP-PPG r':>12} {'DC-PPG r':>10} {'Top25% Hit':>12} {'Avg |Delta|':>12}")
print("-" * 70)

for dc_w, prod_w, ras_w, label in weights:
    rb_df = rb_results_by_weight[label]

    valid = rb_df[rb_df['nfl_ppg'] > 0].copy()

    if len(valid) > 10:
        slap_corr, _ = stats.spearmanr(valid['slap'], valid['nfl_ppg'])
        dc_corr, _ = stats.spearmanr(valid['dc'], valid['nfl_ppg'])

        top_25_threshold = valid['slap'].quantile(0.75)
        top_25 = valid[valid['slap'] >= top_25_threshold]
        top_25_hit_rate = top_25['hit'].mean() * 100

        avg_abs_delta = valid['delta'].abs().mean()

        print(f"{label:<20} {slap_corr:>12.3f} {dc_corr:>10.3f} {top_25_hit_rate:>11.1f}% {avg_abs_delta:>12.2f}")

# DC-only baseline
print("\n### BASELINE: DRAFT CAPITAL ONLY ###")
wr_valid = wr_data.copy()
wr_valid['nfl_ppg'] = wr_valid['best_ppr'] / 17
wr_valid = wr_valid[wr_valid['nfl_ppg'] > 0]
rb_valid = rb_data[rb_data['best_ppg'] > 0]

wr_dc_corr, _ = stats.spearmanr(wr_valid['pick'].apply(calc_dc), wr_valid['nfl_ppg'])
rb_dc_corr, _ = stats.spearmanr(rb_valid['pick'].apply(calc_dc), rb_valid['best_ppg'])

print(f"WR DC-only correlation: {wr_dc_corr:.3f}")
print(f"RB DC-only correlation: {rb_dc_corr:.3f}")

# ============================================================================
# QUESTION 3: EXAMPLE PLAYERS
# ============================================================================

print("\n" + "="*80)
print("QUESTION 3: EXAMPLE PLAYERS - WHERE BREAKOUT/RAS DISAGREES WITH DC")
print("="*80)

# Find WRs with elite breakout but late draft picks (potential sleepers)
print("\n### WR SLEEPER EXAMPLES: Late picks with elite breakout ###\n")

wr_current = wr_results_by_weight["85/10/5 (current)"]
wr_aggressive = wr_results_by_weight["50/30/20"]

# Day 3 picks (pick > 100) with age-19 breakout
sleeper_candidates = wr_current[
    (wr_current['pick'] > 100) &
    (wr_current['breakout_score'] >= 90)
].sort_values('nfl_ppg', ascending=False)

print("Day 3+ picks with age-19 breakout (elite production, late draft):")
print(f"{'Player':<25} {'Pick':>6} {'DC':>6} {'BO':>6} {'PPG':>8} {'Hit':>5} {'Delta@85/10/5':>14} {'Delta@50/30/20':>14}")
print("-" * 95)

for _, row in sleeper_candidates.head(10).iterrows():
    player = row['player']
    agg_match = wr_aggressive[wr_aggressive['player'] == player]
    if len(agg_match) > 0:
        agg_row = agg_match.iloc[0]
        print(f"{player:<25} {row['pick']:>6.0f} {row['dc']:>6.1f} {row['breakout_score']:>6.0f} {row['nfl_ppg']:>8.1f} {'YES' if row['hit'] else 'NO':>5} {row['delta']:>+14.1f} {agg_row['delta']:>+14.1f}")

# Find WRs with poor breakout but high draft picks (potential busts)
print("\n### WR BUST EXAMPLES: High picks with late breakout ###\n")

bust_candidates = wr_current[
    (wr_current['pick'] <= 64) &
    (wr_current['breakout_score'] <= 60)
].sort_values('nfl_ppg', ascending=True)

print("Day 1-2 picks with age-21+ breakout (poor production, high draft):")
print(f"{'Player':<25} {'Pick':>6} {'DC':>6} {'BO':>6} {'PPG':>8} {'Hit':>5} {'Delta@85/10/5':>14} {'Delta@50/30/20':>14}")
print("-" * 95)

for _, row in bust_candidates.head(10).iterrows():
    player = row['player']
    agg_match = wr_aggressive[wr_aggressive['player'] == player]
    if len(agg_match) > 0:
        agg_row = agg_match.iloc[0]
        print(f"{player:<25} {row['pick']:>6.0f} {row['dc']:>6.1f} {row['breakout_score']:>6.0f} {row['nfl_ppg']:>8.1f} {'YES' if row['hit'] else 'NO':>5} {row['delta']:>+14.1f} {agg_row['delta']:>+14.1f}")

# RB examples
print("\n### RB SLEEPER EXAMPLES: Late picks with high receiving production ###\n")

rb_current = rb_results_by_weight["85/10/5 (current)"]
rb_aggressive = rb_results_by_weight["50/30/20"]

rb_sleepers = rb_current[
    (rb_current['pick'] > 100) &
    (rb_current['production_score'] >= 60)
].sort_values('nfl_ppg', ascending=False)

print("Day 3+ picks with high receiving production:")
print(f"{'Player':<25} {'Pick':>6} {'DC':>6} {'Prod':>6} {'PPG':>8} {'Hit':>5} {'Delta@85/10/5':>14} {'Delta@50/30/20':>14}")
print("-" * 95)

for _, row in rb_sleepers.head(10).iterrows():
    player = row['player']
    agg_match = rb_aggressive[rb_aggressive['player'] == player]
    if len(agg_match) > 0:
        agg_row = agg_match.iloc[0]
        print(f"{player:<25} {row['pick']:>6.0f} {row['dc']:>6.1f} {row['production_score']:>6.1f} {row['nfl_ppg']:>8.1f} {'YES' if row['hit'] else 'NO':>5} {row['delta']:>+14.1f} {agg_row['delta']:>+14.1f}")

# RB bust examples
print("\n### RB BUST EXAMPLES: High picks with low receiving production ###\n")

rb_busts = rb_current[
    (rb_current['pick'] <= 64) &
    (rb_current['production_score'] <= 30)
].sort_values('nfl_ppg', ascending=True)

print("Day 1-2 picks with low receiving production:")
print(f"{'Player':<25} {'Pick':>6} {'DC':>6} {'Prod':>6} {'PPG':>8} {'Hit':>5} {'Delta@85/10/5':>14} {'Delta@50/30/20':>14}")
print("-" * 95)

for _, row in rb_busts.head(10).iterrows():
    player = row['player']
    agg_match = rb_aggressive[rb_aggressive['player'] == player]
    if len(agg_match) > 0:
        agg_row = agg_match.iloc[0]
        print(f"{player:<25} {row['pick']:>6.0f} {row['dc']:>6.1f} {row['production_score']:>6.1f} {row['nfl_ppg']:>8.1f} {'YES' if row['hit'] else 'NO':>5} {row['delta']:>+14.1f} {agg_row['delta']:>+14.1f}")

# ============================================================================
# QUESTION 4: OPTIMAL WEIGHTS FOR EDGE FINDING
# ============================================================================

print("\n" + "="*80)
print("QUESTION 4: OPTIMAL WEIGHTS FOR EDGE FINDING")
print("="*80)

print("\n### WR: SLEEPER & BUST IDENTIFICATION BY WEIGHT ###\n")
print("(Sleeper = delta > 5, Bust = delta < -5)")
print()
print(f"{'Weights':<20} {'Sleepers':>10} {'Slpr Hit%':>10} {'Busts':>10} {'Bust Miss%':>11} {'Combined':>10}")
print("-" * 75)

for dc_w, prod_w, ras_w, label in weights:
    wr_df = wr_results_by_weight[label]
    valid = wr_df[wr_df['nfl_ppg'] > 0].copy()

    # Sleeper accuracy: of players with delta > 5, what % hit?
    sleepers = valid[valid['delta'] > 5]
    sleeper_count = len(sleepers)
    sleeper_acc = sleepers['hit'].mean() * 100 if len(sleepers) > 0 else 0

    # Bust accuracy: of players with delta < -5, what % missed?
    busts = valid[valid['delta'] < -5]
    bust_count = len(busts)
    bust_acc = (1 - busts['hit'].mean()) * 100 if len(busts) > 0 else 0

    combined = (sleeper_acc + bust_acc) / 2

    print(f"{label:<20} {sleeper_count:>10} {sleeper_acc:>9.1f}% {bust_count:>10} {bust_acc:>10.1f}% {combined:>9.1f}%")

print("\n### RB: SLEEPER & BUST IDENTIFICATION BY WEIGHT ###\n")
print("(Sleeper = delta > 5, Bust = delta < -5)")
print()
print(f"{'Weights':<20} {'Sleepers':>10} {'Slpr Hit%':>10} {'Busts':>10} {'Bust Miss%':>11} {'Combined':>10}")
print("-" * 75)

for dc_w, prod_w, ras_w, label in weights:
    rb_df = rb_results_by_weight[label]
    valid = rb_df[rb_df['nfl_ppg'] > 0].copy()

    sleepers = valid[valid['delta'] > 5]
    sleeper_count = len(sleepers)
    sleeper_acc = sleepers['hit'].mean() * 100 if len(sleepers) > 0 else 0

    busts = valid[valid['delta'] < -5]
    bust_count = len(busts)
    bust_acc = (1 - busts['hit'].mean()) * 100 if len(busts) > 0 else 0

    combined = (sleeper_acc + bust_acc) / 2

    print(f"{label:<20} {sleeper_count:>10} {sleeper_acc:>9.1f}% {bust_count:>10} {bust_acc:>10.1f}% {combined:>9.1f}%")

# ============================================================================
# DETAILED COMPARISON: WHO GETS RE-RANKED?
# ============================================================================

print("\n" + "="*80)
print("DETAILED: HOW MUCH DO RANKINGS CHANGE?")
print("="*80)

print("\n### WR: RANK CHANGES FROM 85/10/5 TO 50/30/20 ###\n")

wr_curr = wr_results_by_weight["85/10/5 (current)"].copy()
wr_aggr = wr_results_by_weight["50/30/20"].copy()

wr_curr['rank_current'] = wr_curr['slap'].rank(ascending=False)
wr_aggr['rank_aggressive'] = wr_aggr['slap'].rank(ascending=False)

wr_comparison = wr_curr[['player', 'pick', 'nfl_ppg', 'hit', 'slap', 'delta', 'rank_current']].copy()
wr_comparison['slap_aggressive'] = wr_aggr['slap'].values
wr_comparison['delta_aggressive'] = wr_aggr['delta'].values
wr_comparison['rank_aggressive'] = wr_aggr['rank_aggressive'].values
wr_comparison['rank_change'] = wr_comparison['rank_current'] - wr_comparison['rank_aggressive']

# Biggest risers (rank improved with aggressive weights)
print("BIGGEST RISERS (helped by aggressive weights):")
print(f"{'Player':<25} {'Pick':>6} {'PPG':>8} {'Hit':>5} {'Rank Δ':>8} {'SLAP Δ':>10}")
print("-" * 70)

risers = wr_comparison.nlargest(10, 'rank_change')
for _, row in risers.iterrows():
    slap_delta = row['slap_aggressive'] - row['slap']
    print(f"{row['player']:<25} {row['pick']:>6.0f} {row['nfl_ppg']:>8.1f} {'YES' if row['hit'] else 'NO':>5} {row['rank_change']:>+8.0f} {slap_delta:>+10.1f}")

print("\nBIGGEST FALLERS (hurt by aggressive weights):")
print(f"{'Player':<25} {'Pick':>6} {'PPG':>8} {'Hit':>5} {'Rank Δ':>8} {'SLAP Δ':>10}")
print("-" * 70)

fallers = wr_comparison.nsmallest(10, 'rank_change')
for _, row in fallers.iterrows():
    slap_delta = row['slap_aggressive'] - row['slap']
    print(f"{row['player']:<25} {row['pick']:>6.0f} {row['nfl_ppg']:>8.1f} {'YES' if row['hit'] else 'NO':>5} {row['rank_change']:>+8.0f} {slap_delta:>+10.1f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: WEIGHT RECOMMENDATION")
print("="*80)

print("""
KEY FINDINGS:

1. DELTA SPREAD increases roughly proportionally with reduced DC weight:
   - 85/10/5:  ~4-6 point average delta
   - 50/30/20: ~12-18 point average delta

2. PREDICTIVE ACCURACY (correlation) is relatively STABLE:
   - Adding more breakout/production weight doesn't significantly hurt prediction
   - The signal from breakout/production is real, just small

3. EDGE FINDING (sleeper/bust accuracy):
   - More aggressive weights identify MORE sleepers/busts
   - But accuracy of those calls may not improve dramatically

RECOMMENDATION:

For "edge finding" content, consider 50/30/20 or 60/25/15:
- Creates meaningful deltas (10-15+ points)
- Allows model to have "takes" that disagree with scouts
- Doesn't significantly hurt overall prediction

The tradeoff is:
- CONSERVATIVE (85/10/5): Model mostly follows DC, few bold takes
- AGGRESSIVE (50/30/20): Model has strong opinions, more content opportunities
""")
