"""
Explore Alternative WR Production Metrics - Using Local Data
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ============================================================================
# STEP 1: LOAD EXISTING WR DATA
# ============================================================================
print("=" * 100)
print("STEP 1: LOAD EXISTING WR DATA")
print("=" * 100)

# Load WR season data
wr_seasons = pd.read_csv('data/wr_all_seasons.csv')
print(f"\nWR seasons data: {len(wr_seasons)} records")
print(f"Columns: {list(wr_seasons.columns)}")

# Load backtest data with outcomes
wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"\nWR backtest data: {len(wr_backtest)} records")
print(f"Columns: {list(wr_backtest.columns)}")

# Load main database with NFL PPG
df = pd.read_csv('output/slap_complete_database_v4.csv')
wr_main = df[(df['position'] == 'WR') & (df['draft_year'] >= 2015) & (df['draft_year'] <= 2023)].copy()
print(f"\nWR main database: {len(wr_main)} records with NFL PPG")

# ============================================================================
# STEP 2: PREPARE DATA - GET FINAL SEASON STATS
# ============================================================================
print("\n" + "=" * 100)
print("STEP 2: PREPARE DATA - GET FINAL COLLEGE SEASON STATS")
print("=" * 100)

# Get final season for each player (draft_year - 1)
wr_seasons['final_season'] = wr_seasons['draft_year'] - 1

# Filter to final seasons only
final_seasons = wr_seasons[wr_seasons['season'] == wr_seasons['final_season']].copy()
print(f"\nFinal season records: {len(final_seasons)}")

# Sample
print(f"\nSample final season data:")
print(final_seasons[['player_name', 'draft_year', 'season', 'player_rec_yards', 'player_rec_tds', 'player_receptions', 'dominator_rating']].head(10).to_string(index=False))

# ============================================================================
# STEP 3: MERGE WITH OUTCOMES
# ============================================================================
print("\n" + "=" * 100)
print("STEP 3: MERGE WITH NFL OUTCOMES")
print("=" * 100)

# Merge final season stats with main database
merged = wr_main.merge(
    final_seasons[['player_name', 'draft_year', 'player_rec_yards', 'player_rec_tds',
                   'player_receptions', 'team_rec_yards', 'team_rec_tds', 'dominator_rating']],
    on=['player_name', 'draft_year'],
    how='left'
)

print(f"Merged records: {len(merged)}")
matched = merged['player_rec_yards'].notna().sum()
print(f"With receiving stats: {matched} ({matched/len(merged)*100:.1f}%)")

# Also get breakout age from backtest
breakout_lookup = dict(zip(
    wr_backtest['player_name'] + '_' + wr_backtest['draft_year'].astype(str),
    wr_backtest['breakout_age']
))

merged['breakout_age'] = merged.apply(
    lambda x: breakout_lookup.get(f"{x['player_name']}_{int(x['draft_year'])}", np.nan),
    axis=1
)

# Filter to those with stats and NFL outcomes
valid = merged[(merged['player_rec_yards'].notna()) &
               (merged['nfl_best_ppg'].notna()) &
               (merged['nfl_best_ppg'] > 0)].copy()

print(f"Valid for analysis (stats + NFL PPG): {len(valid)}")

# ============================================================================
# STEP 4: CALCULATE PRODUCTION METRICS
# ============================================================================
print("\n" + "=" * 100)
print("STEP 4: CALCULATE PRODUCTION METRICS")
print("=" * 100)

# Convert to numeric
for col in ['player_rec_yards', 'player_rec_tds', 'player_receptions', 'team_rec_yards', 'team_rec_tds']:
    valid[col] = pd.to_numeric(valid[col], errors='coerce')

# Calculate metrics
valid['dominator'] = valid['player_rec_yards'] / valid['team_rec_yards'].replace(0, np.nan) * 100
valid['yards_per_rec'] = valid['player_rec_yards'] / valid['player_receptions'].replace(0, np.nan)
valid['td_rate'] = valid['player_rec_tds'] / valid['player_receptions'].replace(0, np.nan) * 100
valid['weighted_production'] = valid['player_rec_yards'] + (valid['player_rec_tds'] * 20)
valid['yards_per_game'] = valid['player_rec_yards'] / 12.5  # Estimate ~12.5 games
valid['tds_per_game'] = valid['player_rec_tds'] / 12.5
valid['rec_per_game'] = valid['player_receptions'] / 12.5

# Breakout age to score
def breakout_to_score(age):
    if pd.isna(age):
        return 50
    scores = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    return scores.get(int(age), 25)

valid['breakout_score'] = valid['breakout_age'].apply(breakout_to_score)

# Age weight for production
def age_weight(age):
    if pd.isna(age):
        return 1.0
    weights = {18: 1.25, 19: 1.20, 20: 1.10, 21: 1.00, 22: 0.90, 23: 0.80}
    return weights.get(int(age), 0.70)

valid['age_multiplier'] = valid['breakout_age'].apply(age_weight)
valid['age_weighted_yards'] = valid['player_rec_yards'] * valid['age_multiplier']
valid['age_weighted_dominator'] = valid['dominator'] * valid['age_multiplier']

print(f"\nMetrics calculated:")
metrics = ['dominator', 'player_rec_yards', 'yards_per_game', 'player_rec_tds',
           'tds_per_game', 'yards_per_rec', 'td_rate', 'weighted_production',
           'breakout_score', 'age_weighted_yards', 'age_weighted_dominator']

for m in metrics:
    mean = valid[m].mean()
    std = valid[m].std()
    print(f"   {m}: mean={mean:.2f}, std={std:.2f}")

# ============================================================================
# STEP 5: TEST EACH METRIC CORRELATION
# ============================================================================
print("\n" + "=" * 100)
print("STEP 5: TEST EACH METRIC CORRELATION WITH NFL PPG")
print("=" * 100)

def calc_corr(df, metric, outcome='nfl_best_ppg'):
    data = df[[metric, outcome]].dropna()
    if len(data) < 10:
        return np.nan, np.nan, len(data)
    r, p = pearsonr(data[metric], data[outcome])
    return r, p, len(data)

results = []
for metric in metrics:
    r, p, n = calc_corr(valid, metric)
    sig = "Yes ✓" if pd.notna(p) and p < 0.05 else "No"
    results.append({'Metric': metric, 'r': r, 'p': p, 'N': n, 'Sig': sig})

results_df = pd.DataFrame(results).sort_values('r', ascending=False)

print(f"\n{'Metric':<25} {'Correlation (r)':<18} {'p-value':<12} {'N':<6} {'Significant?'}")
print(f"{'-'*25} {'-'*18} {'-'*12} {'-'*6} {'-'*12}")
for _, row in results_df.iterrows():
    r_str = f"{row['r']:.3f}" if pd.notna(row['r']) else "N/A"
    p_str = f"{row['p']:.4f}" if pd.notna(row['p']) else "N/A"
    print(f"{row['Metric']:<25} {r_str:<18} {p_str:<12} {row['N']:<6} {row['Sig']}")

# ============================================================================
# STEP 6: PARTIAL CORRELATIONS (Controlling for DC)
# ============================================================================
print("\n" + "=" * 100)
print("STEP 6: PARTIAL CORRELATIONS (Controlling for Draft Capital)")
print("=" * 100)

print("\nThis shows what value each metric adds BEYOND draft capital.\n")

def partial_corr(df, metric, outcome, control):
    """Calculate partial correlation"""
    data = df[[metric, outcome, control]].dropna()
    if len(data) < 15:
        return np.nan, np.nan, len(data)

    X = sm.add_constant(data[control])

    # Residualize metric
    model_m = sm.OLS(data[metric], X).fit()
    resid_m = model_m.resid

    # Residualize outcome
    model_o = sm.OLS(data[outcome], X).fit()
    resid_o = model_o.resid

    r, p = pearsonr(resid_m, resid_o)
    return r, p, len(data)

print(f"{'Metric':<25} {'Raw r':<10} {'Partial r':<12} {'p-value':<10} {'Adds Value?'}")
print(f"{'-'*25} {'-'*10} {'-'*12} {'-'*10} {'-'*12}")

partial_results = []
for metric in metrics:
    raw_r, raw_p, _ = calc_corr(valid, metric)
    part_r, part_p, n = partial_corr(valid, metric, 'nfl_best_ppg', 'dc_score')

    adds = "Yes ✓" if pd.notna(part_p) and part_p < 0.10 and abs(part_r) > 0.05 else "No"

    partial_results.append({
        'metric': metric,
        'raw_r': raw_r,
        'part_r': part_r,
        'part_p': part_p,
        'adds_value': adds
    })

    raw_str = f"{raw_r:.3f}" if pd.notna(raw_r) else "N/A"
    part_str = f"{part_r:.3f}" if pd.notna(part_r) else "N/A"
    p_str = f"{part_p:.4f}" if pd.notna(part_p) else "N/A"
    print(f"{metric:<25} {raw_str:<10} {part_str:<12} {p_str:<10} {adds}")

# ============================================================================
# STEP 7: MULTIPLE REGRESSION
# ============================================================================
print("\n" + "=" * 100)
print("STEP 7: MULTIPLE REGRESSION (DC + Metric → NFL PPG)")
print("=" * 100)

top_metrics = ['breakout_score', 'dominator', 'player_rec_yards', 'age_weighted_dominator', 'weighted_production']

for metric in top_metrics:
    data = valid[['nfl_best_ppg', 'dc_score', metric]].dropna()
    if len(data) < 20:
        continue

    X = sm.add_constant(data[['dc_score', metric]])
    y = data['nfl_best_ppg']
    model = sm.OLS(y, X).fit()

    dc_p = model.pvalues['dc_score']
    metric_p = model.pvalues[metric]
    r2 = model.rsquared

    sig_text = "SIGNIFICANT ✓" if metric_p < 0.10 else "not significant"
    print(f"\n   {metric}:")
    print(f"      DC p-value: {dc_p:.4f}")
    print(f"      {metric} p-value: {metric_p:.4f} → {sig_text}")
    print(f"      Model R²: {r2:.3f}")

# ============================================================================
# STEP 8: COMPARE VS CURRENT BREAKOUT AGE
# ============================================================================
print("\n" + "=" * 100)
print("STEP 8: COMPARE METRICS VS CURRENT BREAKOUT AGE")
print("=" * 100)

# Sort by partial correlation (what adds value after DC)
partial_df = pd.DataFrame(partial_results)
partial_df = partial_df.sort_values('part_r', ascending=False)

print(f"\nRanked by value added beyond DC (partial correlation):\n")
print(f"{'Rank':<6} {'Metric':<25} {'Partial r':<12} {'Adds Value?'}")
print(f"{'-'*6} {'-'*25} {'-'*12} {'-'*12}")

for i, (_, row) in enumerate(partial_df.iterrows(), 1):
    part_str = f"{row['part_r']:.3f}" if pd.notna(row['part_r']) else "N/A"
    marker = "← CURRENT" if row['metric'] == 'breakout_score' else ""
    print(f"{i:<6} {row['metric']:<25} {part_str:<12} {row['adds_value']:<12} {marker}")

# ============================================================================
# STEP 9: RECOMMENDATION
# ============================================================================
print("\n" + "=" * 100)
print("STEP 9: RECOMMENDATION")
print("=" * 100)

# Find breakout row
breakout_row = partial_df[partial_df['metric'] == 'breakout_score'].iloc[0] if len(partial_df[partial_df['metric'] == 'breakout_score']) > 0 else None

# Find best non-breakout metric
best_alt = partial_df[partial_df['metric'] != 'breakout_score'].iloc[0] if len(partial_df) > 1 else None

print(f"""
📊 ANALYSIS SUMMARY:

1. CURRENT APPROACH (Breakout Age):
   • Raw correlation with NFL PPG: r = {breakout_row['raw_r']:.3f if breakout_row is not None else 'N/A'}
   • Partial correlation (after DC): r = {breakout_row['part_r']:.3f if breakout_row is not None else 'N/A'}
   • Adds value beyond DC? {breakout_row['adds_value'] if breakout_row is not None else 'N/A'}

2. BEST ALTERNATIVE ({best_alt['metric'] if best_alt is not None else 'N/A'}):
   • Raw correlation with NFL PPG: r = {best_alt['raw_r']:.3f if best_alt is not None else 'N/A'}
   • Partial correlation (after DC): r = {best_alt['part_r']:.3f if best_alt is not None else 'N/A'}
   • Adds value beyond DC? {best_alt['adds_value'] if best_alt is not None else 'N/A'}

3. KEY FINDING:
""")

# Check if anything adds value
any_adds = any(row['adds_value'] == 'Yes ✓' for _, row in partial_df.iterrows())

if any_adds:
    best = partial_df[partial_df['adds_value'] == 'Yes ✓'].iloc[0]
    print(f"   ✓ {best['metric']} adds value beyond DC")
    print(f"   → Consider using {best['metric']} instead of/in addition to breakout_score")
else:
    print(f"   ✗ NO metric adds significant value beyond DC")
    print(f"   → Draft capital is doing all the predictive work")
    print(f"   → Consider REMOVING production component entirely")
    print(f"   → Or: Keep breakout age at low weight (20%) for narrative value only")

print(f"""
4. RECOMMENDATION FOR WR MODEL:
""")

if not any_adds:
    print("""   Since no production metric adds value beyond draft capital:

   OPTION A: Remove production entirely
      • New weights: 85% DC / 15% RAS
      • Simpler, potentially more accurate

   OPTION B: Keep breakout age at minimal weight
      • Current weights: 65% DC / 20% Breakout / 15% RAS
      • Provides narrative differentiation
      • Doesn't hurt predictions (at 20% weight)

   OPTION C: Test replacing breakout with raw receiving yards
      • May have slightly better signal
      • More objective than breakout age
""")
else:
    best_metric = partial_df[partial_df['adds_value'] == 'Yes ✓'].iloc[0]['metric']
    print(f"""   {best_metric} shows promise:

   OPTION A: Replace breakout age with {best_metric}
      • Test new weights with this metric

   OPTION B: Combine breakout age + {best_metric}
      • May capture both age and production signals
""")

# Save analysis
valid.to_csv('output/wr_production_analysis.csv', index=False)
print(f"\n✓ Saved analysis data to output/wr_production_analysis.csv")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
