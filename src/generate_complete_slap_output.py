"""
COMPLETE SLAP MODEL OUTPUT - WRs 2015-2024

Generate comprehensive SLAP scores with all metrics.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("GENERATING COMPLETE SLAP MODEL OUTPUT")
print("=" * 90)

# ============================================================================
# LOAD AND VERIFY DATA
# ============================================================================
print("\n--- Loading data ---")

wr = pd.read_csv('data/wr_backtest_expanded_final.csv')
wr['best_ppg'] = wr['best_ppr'] / 17

print(f"Loaded {len(wr)} WRs from 2015-2024")

# Verify PPG is correct
print("\n--- PPG VERIFICATION ---")
stars = wr.nlargest(10, 'best_ppg')[['player_name', 'draft_year', 'best_ppg']]
print("Top 10 by PPG (should be 18-25 range):")
for _, row in stars.iterrows():
    print(f"  {row['player_name']}: {row['best_ppg']:.1f} PPG")

assert wr['best_ppg'].max() > 20, "ERROR: PPG values still look wrong!"
print("\n✓ PPG values verified correct")

# ============================================================================
# CALCULATE NORMALIZED SCORES
# ============================================================================
print("\n--- Calculating normalized scores ---")

# 1. Draft Capital Score (0-100)
# Higher pick = higher score
wr['dc_raw'] = 1 / np.sqrt(wr['pick'])
dc_min, dc_max = wr['dc_raw'].min(), wr['dc_raw'].max()
wr['dc_score'] = ((wr['dc_raw'] - dc_min) / (dc_max - dc_min) * 100).round(1)

# 2. Breakout Age Score (0-100)
# Younger breakout = higher score
# Age 18: 100, Age 19: 90, Age 20: 75, Age 21: 60, Age 22: 45, Age 23: 30, Never: 25
def age_to_score(age):
    if pd.isna(age):
        return 25  # Never hit threshold
    age_map = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30}
    return age_map.get(int(age), 25)

wr['breakout_score'] = wr['breakout_age'].apply(age_to_score)

# 3. RAS Score (0-100)
# RAS is already 0-10, normalize to 0-100
wr['ras_score'] = (wr['RAS'] / 10 * 100).round(1)

# 4. Athletic Status
def get_athletic_status(row):
    if pd.notna(row['RAS']):
        return 'observed'
    elif row['pick'] <= 32:
        return 'elite_optout'
    else:
        return 'missing'

wr['athletic_status'] = wr.apply(get_athletic_status, axis=1)

# ============================================================================
# CALCULATE SLAP SCORE
# ============================================================================
print("\n--- Calculating SLAP scores ---")

# Weights: 50% DC, 35% Breakout, 15% Athletic
W_DC = 0.50
W_BR = 0.35
W_ATH = 0.15

# Mean RAS for imputation
mean_ras_score = wr['ras_score'].mean()
mean_breakout_score = wr['breakout_score'].mean()

def calculate_slap(row):
    dc = row['dc_score']
    br = row['breakout_score']
    ras = row['ras_score']
    status = row['athletic_status']

    if status == 'observed':
        # Full formula
        slap = W_DC * dc + W_BR * br + W_ATH * ras
    elif status == 'elite_optout':
        if pd.isna(br) or br <= 25:
            # Elite opt-out with missing breakout
            slap = dc * 0.85 + mean_breakout_score * 0.15
        else:
            # Elite opt-out with valid breakout
            slap = dc * 0.588 + br * 0.412
    else:
        # Missing RAS, not elite - use mean imputation
        slap = W_DC * dc + W_BR * br + W_ATH * mean_ras_score

    return round(slap, 1)

wr['slap_score'] = wr.apply(calculate_slap, axis=1)

# Calculate SLAP percentile within draft class
wr['slap_percentile'] = wr.groupby('draft_year')['slap_score'].rank(pct=True).round(2) * 100

# Calculate expected SLAP for draft slot (DC-only model)
wr['expected_slap'] = (W_DC * wr['dc_score'] + W_BR * 50 + W_ATH * 50).round(1)
wr['delta'] = (wr['slap_score'] - wr['expected_slap']).round(1)

# ============================================================================
# FIT LOGISTIC MODELS FOR HIT PROBABILITY
# ============================================================================
print("\n--- Fitting logistic models for hit probability ---")

# Filter to players with outcomes (exclude 2024)
train_data = wr[wr['draft_year'] <= 2023].copy()

# Prepare features
X = train_data[['slap_score']].values
y24 = train_data['hit24'].values
y12 = train_data['hit12'].values

# Fit models
model_hit24 = LogisticRegression().fit(X, y24)
model_hit12 = LogisticRegression().fit(X, y12)

print(f"Hit24 model: coef={model_hit24.coef_[0][0]:.4f}, intercept={model_hit24.intercept_[0]:.4f}")
print(f"Hit12 model: coef={model_hit12.coef_[0][0]:.4f}, intercept={model_hit12.intercept_[0]:.4f}")

# Predict probabilities for all players
wr['hit24_prob'] = (model_hit24.predict_proba(wr[['slap_score']].values)[:, 1] * 100).round(1)
wr['hit12_prob'] = (model_hit12.predict_proba(wr[['slap_score']].values)[:, 1] * 100).round(1)

# ============================================================================
# ADD CAREER STATS
# ============================================================================
print("\n--- Calculating career stats ---")

# Load raw stats for career PPG calculation
stats_url = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv"
stats_df = pd.read_csv(stats_url, low_memory=False)

# Filter to WRs
wr_stats = stats_df[stats_df['position'] == 'WR'].copy()

# Calculate PPR points per game
wr_stats['ppr'] = (
    wr_stats['receiving_yards'].fillna(0) * 0.1 +
    wr_stats['receiving_tds'].fillna(0) * 6 +
    wr_stats['receptions'].fillna(0) * 1 +
    wr_stats['rushing_yards'].fillna(0) * 0.1 +
    wr_stats['rushing_tds'].fillna(0) * 6
)

# Aggregate by player+season
season_totals = wr_stats.groupby(['player_display_name', 'season']).agg({
    'ppr': 'sum'
}).reset_index()

# Get career PPG (total points / total seasons * 17)
career_stats = season_totals.groupby('player_display_name').agg({
    'ppr': ['sum', 'count', 'max']
}).reset_index()
career_stats.columns = ['player_name', 'career_ppr', 'seasons_played', 'best_season_ppr']
career_stats['career_ppg'] = (career_stats['career_ppr'] / (career_stats['seasons_played'] * 17)).round(1)

# Count top-24 seasons
season_totals['season_rank'] = season_totals.groupby('season')['ppr'].rank(ascending=False, method='min')
top24_counts = season_totals[season_totals['season_rank'] <= 24].groupby('player_display_name').size().reset_index(name='top24_seasons')

# Merge career stats
wr = wr.merge(career_stats[['player_name', 'career_ppg']], on='player_name', how='left')
wr = wr.merge(top24_counts, left_on='player_name', right_on='player_display_name', how='left')
wr['top24_seasons'] = wr['top24_seasons'].fillna(0).astype(int)

# Drop extra column if exists
if 'player_display_name' in wr.columns:
    wr = wr.drop(columns=['player_display_name'])

# ============================================================================
# PREPARE FINAL OUTPUT
# ============================================================================
print("\n--- Preparing final output ---")

# Mark 2024 outcomes as TBD (partial season)
wr.loc[wr['draft_year'] == 2024, 'hit24'] = -1  # TBD marker
wr.loc[wr['draft_year'] == 2024, 'hit12'] = -1

# Select and order columns
output_cols = [
    # Identification
    'player_name', 'draft_year', 'pick', 'round', 'college',
    # Input scores
    'dc_score', 'breakout_age', 'breakout_score', 'RAS', 'ras_score', 'athletic_status',
    # Model outputs
    'slap_score', 'slap_percentile', 'delta', 'hit24_prob', 'hit12_prob',
    # Actual outcomes
    'best_ppg', 'career_ppg', 'hit24', 'hit12', 'top24_seasons'
]

output_df = wr[output_cols].copy()

# Round numeric columns
output_df['best_ppg'] = output_df['best_ppg'].round(1)

# Replace -1 with TBD for display (keep numeric for CSV)
output_df['hit24_display'] = output_df['hit24'].apply(lambda x: 'TBD' if x == -1 else int(x))
output_df['hit12_display'] = output_df['hit12'].apply(lambda x: 'TBD' if x == -1 else int(x))

# Save to CSV
output_df.to_csv('output/slap_wr_complete_2015_2024.csv', index=False)
print(f"\n✓ Saved to output/slap_wr_complete_2015_2024.csv")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("\n" + "=" * 90)
print("TOP 30 BY SLAP SCORE")
print("=" * 90)

top30 = output_df.nlargest(30, 'slap_score')
print(f"\n{'Rank':<5} {'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'Delta':>6} {'PPG':>6} {'Hit24':>6}")
print("-" * 75)
for i, (_, row) in enumerate(top30.iterrows(), 1):
    hit = row['hit24_display']
    ppg = f"{row['best_ppg']:.1f}" if pd.notna(row['best_ppg']) else "N/A"
    print(f"{i:<5} {row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} {row['slap_score']:>6.1f} {row['delta']:>+6.1f} {ppg:>6} {hit:>6}")

print("\n" + "=" * 90)
print("BIGGEST POSITIVE DELTAS (Model likes more than draft slot)")
print("=" * 90)

pos_delta = output_df.nlargest(15, 'delta')
print(f"\n{'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'Delta':>7} {'PPG':>6} {'Hit24':>6}")
print("-" * 70)
for _, row in pos_delta.iterrows():
    hit = row['hit24_display']
    ppg = f"{row['best_ppg']:.1f}" if pd.notna(row['best_ppg']) else "N/A"
    print(f"{row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} {row['slap_score']:>6.1f} {row['delta']:>+7.1f} {ppg:>6} {hit:>6}")

print("\n" + "=" * 90)
print("BIGGEST NEGATIVE DELTAS (Model likes less than draft slot)")
print("=" * 90)

neg_delta = output_df.nsmallest(15, 'delta')
print(f"\n{'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'Delta':>7} {'PPG':>6} {'Hit24':>6}")
print("-" * 70)
for _, row in neg_delta.iterrows():
    hit = row['hit24_display']
    ppg = f"{row['best_ppg']:.1f}" if pd.notna(row['best_ppg']) else "N/A"
    print(f"{row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} {row['slap_score']:>6.1f} {row['delta']:>+7.1f} {ppg:>6} {hit:>6}")

print("\n" + "=" * 90)
print("SUMMARY STATS BY DRAFT YEAR")
print("=" * 90)

print(f"\n{'Year':>6} {'WRs':>5} {'Hits':>5} {'Rate':>6} {'Avg SLAP':>9} {'Avg PPG':>8} {'Top SLAP':>9}")
print("-" * 55)

for year in range(2015, 2025):
    yr_data = output_df[output_df['draft_year'] == year]
    n = len(yr_data)
    hits = yr_data[yr_data['hit24'] == 1]['hit24'].count() if year < 2024 else 'TBD'
    rate = f"{hits/n*100:.0f}%" if isinstance(hits, int) else 'TBD'
    avg_slap = yr_data['slap_score'].mean()
    avg_ppg = yr_data['best_ppg'].mean()
    top_player = yr_data.nlargest(1, 'slap_score').iloc[0]

    print(f"{year:>6} {n:>5} {str(hits):>5} {rate:>6} {avg_slap:>9.1f} {avg_ppg:>8.1f} {top_player['player_name'][:15]:>9}")

print("\n" + "=" * 90)
print("MODEL CALIBRATION CHECK")
print("=" * 90)

# Check if hit probabilities match actual hit rates
for prob_bucket in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
    lo, hi = prob_bucket
    bucket = output_df[(output_df['hit24_prob'] >= lo) & (output_df['hit24_prob'] < hi) & (output_df['hit24'] >= 0)]
    if len(bucket) > 0:
        actual_rate = bucket['hit24'].mean() * 100
        predicted_rate = bucket['hit24_prob'].mean()
        print(f"Predicted {lo}-{hi}%: n={len(bucket):>3}, actual hit rate={actual_rate:>5.1f}%, avg predicted={predicted_rate:>5.1f}%")
