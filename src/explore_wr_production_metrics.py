"""
Explore Alternative WR Production Metrics
Test what predicts NFL fantasy success better than breakout age
"""

import pandas as pd
import numpy as np
import requests
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

CFBD_API_KEY = "lpa+XMVqHxdlMJvBZaFKSL+0MKncHjCGNyduoccLYTYfrRLl/dsorViGWMg2kc1/"

# ============================================================================
# STEP 1: CHECK WHAT WR STATS ARE AVAILABLE IN CFBD
# ============================================================================
print("=" * 100)
print("STEP 1: CHECK CFBD WR STATS AVAILABILITY")
print("=" * 100)

# Test what endpoints are available
headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}

# Check player season stats endpoint
print("\nChecking CFBD player_stats endpoint...")
url = "https://api.collegefootballdata.com/stats/player/season"
params = {"year": 2023, "category": "receiving", "seasonType": "regular"}
response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    sample = response.json()[:5]
    print(f"✓ Receiving stats available")
    if sample:
        print(f"   Sample keys: {list(sample[0].keys())}")
        print(f"\n   Sample data:")
        for s in sample[:3]:
            print(f"      {s.get('player', 'N/A')}: {s.get('statType', 'N/A')} = {s.get('stat', 'N/A')}")
else:
    print(f"✗ Error: {response.status_code}")

# Check what stat types exist
print("\n   All stat types in receiving category:")
all_stats = response.json() if response.status_code == 200 else []
stat_types = set(s.get('statType') for s in all_stats)
for st in sorted(stat_types):
    print(f"      - {st}")

# Check team stats endpoint for team totals
print("\nChecking team stats endpoint...")
url = "https://api.collegefootballdata.com/stats/season"
params = {"year": 2023}
response = requests.get(url, headers=headers, params=params)
if response.status_code == 200:
    team_data = response.json()[:5]
    print(f"✓ Team stats available")
    if team_data:
        stat_names = set()
        for t in response.json():
            stat_names.add(t.get('statName'))
        print(f"   Available team stats ({len(stat_names)}):")
        for sn in sorted(list(stat_names))[:20]:
            print(f"      - {sn}")
else:
    print(f"✗ Error: {response.status_code}")

# ============================================================================
# STEP 2: PULL WR RECEIVING DATA FOR BACKTEST
# ============================================================================
print("\n" + "=" * 100)
print("STEP 2: PULL WR RECEIVING DATA FROM CFBD")
print("=" * 100)

# Load current database to get WR names and draft years
df = pd.read_csv('output/slap_complete_database_v4.csv')
wr_backtest = df[(df['position'] == 'WR') & (df['draft_year'] >= 2015) & (df['draft_year'] <= 2023)].copy()
print(f"\nWRs in backtest: {len(wr_backtest)}")

# For each WR, pull their final college season stats
# Final season = draft_year - 1

def get_wr_season_stats(year):
    """Get all WR receiving stats for a season"""
    url = "https://api.collegefootballdata.com/stats/player/season"
    params = {"year": year, "category": "receiving", "seasonType": "regular"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    return []

def get_team_stats(year):
    """Get team passing stats for a season"""
    url = "https://api.collegefootballdata.com/stats/season"
    params = {"year": year}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        team_dict = {}
        for row in data:
            team = row.get('team')
            stat = row.get('statName')
            val = row.get('statValue')
            if team not in team_dict:
                team_dict[team] = {}
            team_dict[team][stat] = val
        return team_dict
    return {}

# Pull data for all relevant years
print("\nPulling receiving stats by year...")
all_receiving = []
all_team_stats = {}

for year in range(2014, 2023):  # 2014-2022 seasons for 2015-2023 draft classes
    rec_stats = get_wr_season_stats(year)
    team_stats = get_team_stats(year)
    print(f"   {year}: {len(rec_stats)} receiving records, {len(team_stats)} teams")

    # Process receiving stats
    for stat in rec_stats:
        stat['season'] = year
    all_receiving.extend(rec_stats)
    all_team_stats[year] = team_stats

print(f"\nTotal receiving records: {len(all_receiving)}")

# Convert to dataframe and pivot
rec_df = pd.DataFrame(all_receiving)
print(f"\nReceiving stat types available:")
for st in rec_df['statType'].unique():
    count = len(rec_df[rec_df['statType'] == st])
    print(f"   {st}: {count} records")

# Pivot to get one row per player-season
rec_pivot = rec_df.pivot_table(
    index=['player', 'team', 'season'],
    columns='statType',
    values='stat',
    aggfunc='first'
).reset_index()

print(f"\nPivoted stats columns: {list(rec_pivot.columns)}")
print(f"Player-seasons: {len(rec_pivot)}")

# ============================================================================
# STEP 3: MATCH WRs TO THEIR STATS
# ============================================================================
print("\n" + "=" * 100)
print("STEP 3: MATCH WRs TO THEIR FINAL SEASON STATS")
print("=" * 100)

def normalize_name(name):
    """Normalize name for matching"""
    import re
    name = str(name).lower().strip()
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)$', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = ' '.join(name.split())
    return name

# Create lookup by normalized name and season
rec_pivot['norm_name'] = rec_pivot['player'].apply(normalize_name)
rec_pivot['draft_year'] = rec_pivot['season'] + 1  # draft year = college season + 1

# Match WRs
matched_data = []

for _, wr in wr_backtest.iterrows():
    norm = normalize_name(wr['player_name'])
    final_season = int(wr['draft_year']) - 1

    # Try to find match
    matches = rec_pivot[(rec_pivot['norm_name'] == norm) & (rec_pivot['season'] == final_season)]

    if len(matches) == 0:
        # Try partial match
        first_last = norm.split()
        if len(first_last) >= 2:
            first, last = first_last[0], first_last[-1]
            matches = rec_pivot[
                (rec_pivot['norm_name'].str.contains(first)) &
                (rec_pivot['norm_name'].str.contains(last)) &
                (rec_pivot['season'] == final_season)
            ]

    if len(matches) > 0:
        m = matches.iloc[0]
        team = m['team']
        team_data = all_team_stats.get(final_season, {}).get(team, {})

        matched_data.append({
            'player_name': wr['player_name'],
            'draft_year': wr['draft_year'],
            'pick': wr['pick'],
            'slap_score': wr['slap_score'],
            'dc_score': wr['dc_score'],
            'breakout_age': wr.get('breakout_age'),
            'nfl_best_ppg': wr['nfl_best_ppg'],
            'nfl_hit24': wr['nfl_hit24'],
            'college_team': team,
            'rec_yards': m.get('YDS', 0),
            'receptions': m.get('REC', 0),
            'rec_tds': m.get('TD', 0),
            'rec_long': m.get('LONG', 0),
            'team_pass_att': team_data.get('passAttempts', 0),
            'team_pass_yards': team_data.get('netPassingYards', team_data.get('passingYards', 0)),
            'team_pass_tds': team_data.get('passingTDs', 0),
        })

matched_df = pd.DataFrame(matched_data)
print(f"\nMatched WRs: {len(matched_df)}/{len(wr_backtest)} ({len(matched_df)/len(wr_backtest)*100:.1f}%)")

# Convert to numeric
for col in ['rec_yards', 'receptions', 'rec_tds', 'team_pass_att', 'team_pass_yards', 'team_pass_tds']:
    matched_df[col] = pd.to_numeric(matched_df[col], errors='coerce').fillna(0)

# Filter to those with meaningful data
matched_df = matched_df[(matched_df['rec_yards'] > 0) & (matched_df['team_pass_att'] > 0)]
print(f"WRs with valid receiving data: {len(matched_df)}")

# Also need NFL PPG
matched_df = matched_df[matched_df['nfl_best_ppg'].notna()]
print(f"WRs with NFL PPG for correlation: {len(matched_df)}")

# ============================================================================
# STEP 4: CALCULATE PRODUCTION METRICS
# ============================================================================
print("\n" + "=" * 100)
print("STEP 4: CALCULATE PRODUCTION METRICS")
print("=" * 100)

# Calculate various metrics
matched_df['dominator'] = matched_df['rec_yards'] / matched_df['team_pass_yards'].replace(0, np.nan)
matched_df['yards_per_team_att'] = matched_df['rec_yards'] / matched_df['team_pass_att'].replace(0, np.nan)
matched_df['yards_per_rec'] = matched_df['rec_yards'] / matched_df['receptions'].replace(0, np.nan)
matched_df['td_rate'] = matched_df['rec_tds'] / matched_df['receptions'].replace(0, np.nan)
matched_df['target_share_proxy'] = matched_df['receptions'] / matched_df['team_pass_att'].replace(0, np.nan)
matched_df['weighted_production'] = matched_df['rec_yards'] + (matched_df['rec_tds'] * 20)
matched_df['rec_share'] = matched_df['receptions'] / matched_df['team_pass_att'].replace(0, np.nan)

# Estimate yards per game (assume ~12-13 games)
matched_df['yards_per_game'] = matched_df['rec_yards'] / 12.5

# Also convert breakout age to score (current method)
def breakout_age_to_score(age):
    if pd.isna(age):
        return 50  # Default for missing
    scores = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    return scores.get(int(age), 25)

matched_df['breakout_score'] = matched_df['breakout_age'].apply(breakout_age_to_score)

print(f"\nCalculated metrics:")
metrics = ['dominator', 'yards_per_team_att', 'rec_yards', 'yards_per_game',
           'td_rate', 'yards_per_rec', 'target_share_proxy', 'rec_tds',
           'weighted_production', 'breakout_score']
for m in metrics:
    valid = matched_df[m].notna().sum()
    mean = matched_df[m].mean()
    print(f"   {m}: {valid} valid, mean={mean:.2f}")

# ============================================================================
# STEP 5: TEST EACH METRIC CORRELATION
# ============================================================================
print("\n" + "=" * 100)
print("STEP 5: TEST EACH METRIC CORRELATION WITH NFL PPG")
print("=" * 100)

def calc_correlation(df, metric_col, outcome_col='nfl_best_ppg'):
    """Calculate correlation and p-value"""
    valid = df[[metric_col, outcome_col]].dropna()
    if len(valid) < 10:
        return np.nan, np.nan, len(valid)
    r, p = pearsonr(valid[metric_col], valid[outcome_col])
    return r, p, len(valid)

results = []
for metric in metrics:
    r, p, n = calc_correlation(matched_df, metric)
    sig = "Yes" if p < 0.05 else "No"
    results.append({
        'Metric': metric,
        'r': r,
        'p-value': p,
        'N': n,
        'Significant': sig
    })

results_df = pd.DataFrame(results).sort_values('r', ascending=False)

print(f"\n{'Metric':<25} {'r':<10} {'p-value':<12} {'N':<6} {'Significant'}")
print(f"{'-'*25} {'-'*10} {'-'*12} {'-'*6} {'-'*12}")
for _, row in results_df.iterrows():
    r_str = f"{row['r']:.3f}" if pd.notna(row['r']) else "N/A"
    p_str = f"{row['p-value']:.4f}" if pd.notna(row['p-value']) else "N/A"
    print(f"{row['Metric']:<25} {r_str:<10} {p_str:<12} {row['N']:<6} {row['Significant']}")

# ============================================================================
# STEP 6: TEST COMBINATIONS
# ============================================================================
print("\n" + "=" * 100)
print("STEP 6: TEST METRIC COMBINATIONS")
print("=" * 100)

# Normalize metrics to 0-100 scale for combining
def normalize_0_100(series):
    """Normalize to 0-100 scale"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0 + 50
    return (series - min_val) / (max_val - min_val) * 100

matched_df['dominator_norm'] = normalize_0_100(matched_df['dominator'].fillna(matched_df['dominator'].median()))
matched_df['yards_norm'] = normalize_0_100(matched_df['rec_yards'])
matched_df['tds_norm'] = normalize_0_100(matched_df['rec_tds'])
matched_df['ypg_norm'] = normalize_0_100(matched_df['yards_per_game'])

# Age weight (same as RB model)
def age_weight(age):
    if pd.isna(age):
        return 1.0
    age = int(age)
    weights = {18: 1.25, 19: 1.20, 20: 1.10, 21: 1.00, 22: 0.90, 23: 0.80, 24: 0.70}
    return weights.get(age, 0.70)

matched_df['age_multiplier'] = matched_df['breakout_age'].apply(age_weight)

# Combinations
matched_df['combo_dom_age'] = matched_df['dominator_norm'] * matched_df['age_multiplier']
matched_df['combo_ypg_tds'] = (matched_df['ypg_norm'] + matched_df['tds_norm']) / 2
matched_df['combo_market_volume'] = (matched_df['dominator_norm'] + matched_df['yards_norm']) / 2
matched_df['combo_best_of'] = matched_df[['dominator_norm', 'breakout_score']].max(axis=1)

combo_metrics = ['combo_dom_age', 'combo_ypg_tds', 'combo_market_volume', 'combo_best_of']

print(f"\n{'Combination':<25} {'r':<10} {'p-value':<12} {'N':<6} {'Significant'}")
print(f"{'-'*25} {'-'*10} {'-'*12} {'-'*6} {'-'*12}")

for metric in combo_metrics:
    r, p, n = calc_correlation(matched_df, metric)
    sig = "Yes" if p < 0.05 else "No"
    r_str = f"{r:.3f}" if pd.notna(r) else "N/A"
    p_str = f"{p:.4f}" if pd.notna(p) else "N/A"
    print(f"{metric:<25} {r_str:<10} {p_str:<12} {n:<6} {sig}")

# ============================================================================
# STEP 7: CONTROL FOR DRAFT CAPITAL (Partial Correlations)
# ============================================================================
print("\n" + "=" * 100)
print("STEP 7: CONTROL FOR DRAFT CAPITAL")
print("=" * 100)

def partial_correlation(df, metric_col, outcome_col, control_col):
    """Calculate partial correlation controlling for a third variable"""
    valid = df[[metric_col, outcome_col, control_col]].dropna()
    if len(valid) < 15:
        return np.nan, np.nan, len(valid)

    # Residualize both variables against the control
    X = sm.add_constant(valid[control_col])

    # Metric residuals
    model_metric = sm.OLS(valid[metric_col], X).fit()
    metric_resid = model_metric.resid

    # Outcome residuals
    model_outcome = sm.OLS(valid[outcome_col], X).fit()
    outcome_resid = model_outcome.resid

    # Correlation of residuals
    r, p = pearsonr(metric_resid, outcome_resid)
    return r, p, len(valid)

# Test top metrics with partial correlation
top_metrics = ['dominator', 'yards_per_team_att', 'rec_yards', 'yards_per_game',
               'weighted_production', 'breakout_score', 'combo_dom_age']

print(f"\n{'Metric':<25} {'Raw r':<10} {'Partial r':<12} {'p-value':<10} {'Adds Value?'}")
print(f"{'-'*25} {'-'*10} {'-'*12} {'-'*10} {'-'*12}")

for metric in top_metrics:
    raw_r, raw_p, _ = calc_correlation(matched_df, metric)
    part_r, part_p, n = partial_correlation(matched_df, metric, 'nfl_best_ppg', 'dc_score')

    adds_value = "Yes" if part_p < 0.10 and part_r > 0.05 else "No"

    raw_str = f"{raw_r:.3f}" if pd.notna(raw_r) else "N/A"
    part_str = f"{part_r:.3f}" if pd.notna(part_r) else "N/A"
    p_str = f"{part_p:.4f}" if pd.notna(part_p) else "N/A"

    print(f"{metric:<25} {raw_str:<10} {part_str:<12} {p_str:<10} {adds_value}")

# ============================================================================
# STEP 8: MULTIPLE REGRESSION
# ============================================================================
print("\n" + "=" * 100)
print("STEP 8: MULTIPLE REGRESSION ANALYSIS")
print("=" * 100)

# Does metric add to DC in predicting NFL PPG?
print("\nTesting if each metric adds value beyond DC in multiple regression:\n")

for metric in ['breakout_score', 'dominator', 'yards_per_team_att', 'rec_yards', 'combo_dom_age']:
    valid = matched_df[['nfl_best_ppg', 'dc_score', metric]].dropna()
    if len(valid) < 20:
        continue

    X = sm.add_constant(valid[['dc_score', metric]])
    y = valid['nfl_best_ppg']
    model = sm.OLS(y, X).fit()

    dc_coef = model.params['dc_score']
    dc_pval = model.pvalues['dc_score']
    metric_coef = model.params[metric]
    metric_pval = model.pvalues[metric]

    print(f"   {metric}:")
    print(f"      DC coefficient: {dc_coef:.4f} (p={dc_pval:.4f})")
    print(f"      {metric} coefficient: {metric_coef:.4f} (p={metric_pval:.4f})")
    print(f"      Model R²: {model.rsquared:.3f}")
    print(f"      → {metric} {'SIGNIFICANT' if metric_pval < 0.10 else 'not significant'} after controlling for DC\n")

# ============================================================================
# STEP 9: FINAL COMPARISON
# ============================================================================
print("\n" + "=" * 100)
print("STEP 9: FINAL COMPARISON - BREAKOUT AGE vs BEST ALTERNATIVE")
print("=" * 100)

# Find the best metric
best_metrics = results_df.nlargest(5, 'r')

print(f"\nTop 5 metrics by raw correlation with NFL PPG:")
for _, row in best_metrics.iterrows():
    print(f"   {row['Metric']}: r = {row['r']:.3f}")

# Get partial correlations for comparison
print(f"\n{'Metric':<25} {'Raw r':<10} {'Partial r (after DC)':<20} {'Verdict'}")
print(f"{'-'*25} {'-'*10} {'-'*20} {'-'*20}")

comparison_metrics = ['breakout_score', 'dominator', 'yards_per_team_att', 'rec_yards', 'combo_dom_age']
partial_results = []

for metric in comparison_metrics:
    raw_r, _, _ = calc_correlation(matched_df, metric)
    part_r, part_p, _ = partial_correlation(matched_df, metric, 'nfl_best_ppg', 'dc_score')

    if pd.notna(part_r) and pd.notna(part_p):
        verdict = "✓ Adds value" if part_p < 0.10 and part_r > 0.05 else "✗ No value"
    else:
        verdict = "N/A"

    partial_results.append({
        'metric': metric,
        'raw_r': raw_r,
        'part_r': part_r,
        'part_p': part_p,
        'verdict': verdict
    })

    raw_str = f"{raw_r:.3f}" if pd.notna(raw_r) else "N/A"
    part_str = f"{part_r:.3f}" if pd.notna(part_r) else "N/A"
    print(f"{metric:<25} {raw_str:<10} {part_str:<20} {verdict}")

# ============================================================================
# STEP 10: RECOMMENDATION
# ============================================================================
print("\n" + "=" * 100)
print("STEP 10: RECOMMENDATION")
print("=" * 100)

# Find best metric by partial correlation
partial_df = pd.DataFrame(partial_results)
partial_df['abs_part_r'] = partial_df['part_r'].abs()
best_alt = partial_df[partial_df['metric'] != 'breakout_score'].nlargest(1, 'abs_part_r')
breakout_row = partial_df[partial_df['metric'] == 'breakout_score'].iloc[0] if len(partial_df[partial_df['metric'] == 'breakout_score']) > 0 else None

print(f"""
📊 ANALYSIS SUMMARY:

1. RAW CORRELATIONS (without controlling for DC):
   • Best metric: {results_df.iloc[0]['Metric']} (r = {results_df.iloc[0]['r']:.3f})
   • Breakout Score: r = {matched_df['breakout_score'].corr(matched_df['nfl_best_ppg']):.3f}

2. PARTIAL CORRELATIONS (after controlling for DC):
   • This shows what ADDS VALUE beyond draft capital
   • Breakout Score: partial r = {breakout_row['part_r']:.3f if breakout_row is not None else 'N/A'}
   • Best Alternative: {best_alt.iloc[0]['metric'] if len(best_alt) > 0 else 'N/A'} (partial r = {best_alt.iloc[0]['part_r']:.3f if len(best_alt) > 0 else 'N/A'})

3. RECOMMENDATION:
""")

# Determine recommendation
any_adds_value = any(r['part_p'] < 0.10 and r['part_r'] > 0.05 for r in partial_results if pd.notna(r.get('part_p')))

if any_adds_value:
    best = max(partial_results, key=lambda x: x['part_r'] if pd.notna(x['part_r']) else -999)
    print(f"   ✓ {best['metric']} adds value beyond DC (partial r = {best['part_r']:.3f})")
    if best['metric'] != 'breakout_score':
        print(f"   → Consider replacing breakout_score with {best['metric']}")
else:
    print(f"   ✗ NO metric adds significant value beyond DC")
    print(f"   → The WR production component may not be useful")
    print(f"   → Consider increasing DC weight further or removing production entirely")

# Save matched data for future use
matched_df.to_csv('output/wr_production_analysis.csv', index=False)
print(f"\n✓ Saved analysis data to output/wr_production_analysis.csv")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
