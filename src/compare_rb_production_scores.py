#!/usr/bin/env python3
"""
Compare old vs new RB production scores with age weighting.
"""

import pandas as pd
import numpy as np

# Old formula (no age weighting)
def rb_production_score_old(rec_yards, team_pass_att):
    """Original RB receiving production score"""
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return None
    ratio = rec_yards / team_pass_att
    score = (ratio / 1.0) * 100
    return min(100, max(0, score))

# New formula (with age weighting)
def rb_production_score_new(rec_yards, team_pass_att, draft_age):
    """
    Calculate RB receiving production score with continuous age weighting.
    """
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return None
    if pd.isna(draft_age):
        draft_age = 22  # Default to baseline if missing

    # Raw production ratio
    ratio = rec_yards / team_pass_att

    # Season age (age during final college season)
    season_age = draft_age - 1

    # Continuous age weight
    # Age 19 = 1.15, Age 20 = 1.10, Age 21 = 1.05, Age 22 = 1.00, Age 23 = 0.95
    age_weight = 1.15 - (0.05 * (season_age - 19))
    age_weight = max(0.85, min(1.15, age_weight))  # Cap between 0.85 and 1.15

    # Age-adjusted production score
    score = ratio * age_weight * 100

    return min(100, max(0, score))

# Load RB backtest data (with receiving stats)
rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
print(f"Loaded {len(rb_backtest)} RBs from backtest data\n")

# Calculate both scores
rb_backtest['old_score'] = rb_backtest.apply(
    lambda x: rb_production_score_old(x['rec_yards'], x['team_pass_att']), axis=1
)
rb_backtest['new_score'] = rb_backtest.apply(
    lambda x: rb_production_score_new(x['rec_yards'], x['team_pass_att'], x['age']), axis=1
)

# Calculate age-related fields for display
rb_backtest['season_age'] = rb_backtest['age'] - 1
rb_backtest['age_weight'] = rb_backtest['season_age'].apply(
    lambda x: max(0.85, min(1.15, 1.15 - (0.05 * (x - 19)))) if not pd.isna(x) else 1.0
)
rb_backtest['score_diff'] = rb_backtest['new_score'] - rb_backtest['old_score']

# Filter to rows with valid scores
valid = rb_backtest[rb_backtest['old_score'].notna()].copy()
print(f"RBs with valid production scores: {len(valid)}\n")

# ============================================================================
# SHOW 15 SAMPLE RBs
# ============================================================================
print("=" * 100)
print("15 SAMPLE RBs - OLD vs NEW PRODUCTION SCORES")
print("=" * 100)

# Get a diverse sample: some young, some old, some middle
young = valid.nsmallest(5, 'age')
old = valid.nlargest(5, 'age')
middle = valid[valid['age'].between(21.5, 22.5)].head(5)
sample = pd.concat([young, old, middle]).drop_duplicates()

# If we don't have 15, just take head
if len(sample) < 15:
    sample = valid.head(15)

display_cols = ['player_name', 'rec_yards', 'team_pass_att', 'age', 'season_age',
                'age_weight', 'old_score', 'new_score', 'score_diff']

print(f"\n{'Name':<25} {'Rec Yds':>8} {'Team PA':>8} {'Draft Age':>10} {'Szn Age':>8} {'Age Wt':>8} {'Old':>8} {'New':>8} {'Diff':>8}")
print("-" * 100)

for _, row in sample.head(15).iterrows():
    print(f"{row['player_name']:<25} {row['rec_yards']:>8.0f} {row['team_pass_att']:>8.0f} {row['age']:>10.1f} {row['season_age']:>8.1f} {row['age_weight']:>8.2f} {row['old_score']:>8.1f} {row['new_score']:>8.1f} {row['score_diff']:>+8.1f}")

# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("OVERALL RB PRODUCTION SCORE DISTRIBUTION COMPARISON")
print("=" * 100)

print("\nOLD FORMULA (no age weighting):")
print(f"  Mean:   {valid['old_score'].mean():.2f}")
print(f"  Median: {valid['old_score'].median():.2f}")
print(f"  Std:    {valid['old_score'].std():.2f}")
print(f"  Min:    {valid['old_score'].min():.2f}")
print(f"  Max:    {valid['old_score'].max():.2f}")

print("\nNEW FORMULA (with age weighting):")
print(f"  Mean:   {valid['new_score'].mean():.2f}")
print(f"  Median: {valid['new_score'].median():.2f}")
print(f"  Std:    {valid['new_score'].std():.2f}")
print(f"  Min:    {valid['new_score'].min():.2f}")
print(f"  Max:    {valid['new_score'].max():.2f}")

print("\nSCORE CHANGES:")
print(f"  Mean change:   {valid['score_diff'].mean():+.2f}")
print(f"  Median change: {valid['score_diff'].median():+.2f}")
print(f"  Max increase:  {valid['score_diff'].max():+.2f}")
print(f"  Max decrease:  {valid['score_diff'].min():+.2f}")

# Age breakdown
print("\n" + "=" * 100)
print("SCORE CHANGES BY DRAFT AGE")
print("=" * 100)
age_groups = valid.groupby(valid['age'].round()).agg({
    'old_score': 'mean',
    'new_score': 'mean',
    'score_diff': 'mean',
    'player_name': 'count'
}).rename(columns={'player_name': 'count'})

print(f"\n{'Draft Age':>10} {'Count':>8} {'Old Mean':>10} {'New Mean':>10} {'Avg Diff':>10}")
print("-" * 50)
for age, row in age_groups.iterrows():
    print(f"{age:>10.0f} {row['count']:>8.0f} {row['old_score']:>10.2f} {row['new_score']:>10.2f} {row['score_diff']:>+10.2f}")

print("\n" + "=" * 100)
print("AGE WEIGHT REFERENCE")
print("=" * 100)
print("\nSeason Age -> Age Weight:")
for season_age in range(19, 26):
    weight = max(0.85, min(1.15, 1.15 - (0.05 * (season_age - 19))))
    draft_age = season_age + 1
    print(f"  Season age {season_age} (draft age {draft_age}): {weight:.2f}")
