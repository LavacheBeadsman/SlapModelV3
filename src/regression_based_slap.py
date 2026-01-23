"""
SLAP Score V3 - Regression-Based Approach

Instead of arbitrary thresholds and special cases, let the regression
learn from the data what matters and how to handle missingness.

Approach:
1. Raw inputs (no manual transformations)
2. Missingness indicators as features
3. Let regression determine weights
4. Use predicted values as SLAP scores
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("="*80)
print("REGRESSION-BASED SLAP MODEL")
print("="*80)

# Load all data sources
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')
ras_data = pd.read_csv('data/wr_ras_merged.csv')
breakout_full = pd.read_csv('data/wr_breakout_dominator_full.csv')
seasons = pd.read_csv('data/wr_all_seasons.csv')
breakout_fixed = pd.read_csv('data/wr_breakout_age_scores_fixed.csv')

# Build WR dataset
wr = hit_rates[hit_rates['position'] == 'WR'].copy()
wr = wr[wr['draft_year'].isin([2020, 2021, 2022, 2023, 2024])].copy()

# Merge RAS
wr = wr.merge(ras_data[['player_name', 'draft_year', 'RAS']],
              on=['player_name', 'draft_year'], how='left')

# Merge breakout data
wr = wr.merge(breakout_fixed[['player_name', 'draft_year', 'breakout_age', 'peak_dominator']],
              on=['player_name', 'draft_year'], how='left')

# Calculate career/best season receiving yards from seasons data
career_yards = seasons.groupby(['player_name', 'draft_year']).agg({
    'player_rec_yards': ['sum', 'max'],
    'dominator_rating': 'max'
}).reset_index()
career_yards.columns = ['player_name', 'draft_year', 'career_yards', 'best_season_yards', 'best_dominator']

wr = wr.merge(career_yards, on=['player_name', 'draft_year'], how='left')

# Calculate team quality proxy: count of WRs drafted from same school in nearby years
def get_team_quality(row, all_wr):
    """Count how many other WRs were drafted from same school within 2 years"""
    school = row.get('college')
    if pd.isna(school):
        return 0
    same_school = all_wr[
        (all_wr['college'] == school) &
        (abs(all_wr['draft_year'] - row['draft_year']) <= 2) &
        (all_wr['player_name'] != row['player_name'])
    ]
    return len(same_school)

# Get college info
college_info = breakout_full[['player_name', 'draft_year', 'college']].drop_duplicates()
wr = wr.merge(college_info, on=['player_name', 'draft_year'], how='left')

wr['team_quality'] = wr.apply(lambda r: get_team_quality(r, wr), axis=1)

# Calculate outcome
wr['best_ppg'] = wr['best_ppr'] / 17

print(f"\nDataset: {len(wr)} WRs (2020-2024)")
print(f"Outcome: best_ppg (best season fantasy PPG)")

# ============================================================================
# CREATE RAW FEATURES AND MISSINGNESS INDICATORS
# ============================================================================
print("\n" + "="*80)
print("STEP 1: RAW FEATURES AND MISSINGNESS INDICATORS")
print("="*80)

# Raw features
wr['log_pick'] = np.log(wr['pick'])  # Log transform for draft pick (diminishing returns)
wr['inv_sqrt_pick'] = 1 / np.sqrt(wr['pick'])  # Our current DC formula

# Missingness indicators
wr['has_ras'] = wr['RAS'].notna().astype(int)
wr['has_breakout_age'] = wr['breakout_age'].notna().astype(int)
wr['has_dominator'] = wr['peak_dominator'].notna().astype(int)
wr['has_career_yards'] = wr['career_yards'].notna().astype(int)

# Fill missing values with mean (for regression) - the missingness indicator captures the info
wr['RAS_filled'] = wr['RAS'].fillna(wr['RAS'].mean())
wr['breakout_age_filled'] = wr['breakout_age'].fillna(wr['breakout_age'].mean())
wr['peak_dom_filled'] = wr['peak_dominator'].fillna(wr['peak_dominator'].mean())
wr['career_yards_filled'] = wr['career_yards'].fillna(wr['career_yards'].mean())
wr['best_yards_filled'] = wr['best_season_yards'].fillna(wr['best_season_yards'].mean())

print("\nFeature summary:")
print(f"  draft_pick: range {wr['pick'].min():.0f}-{wr['pick'].max():.0f}")
print(f"  RAS: {wr['has_ras'].sum()} observed ({wr['has_ras'].mean()*100:.1f}%), mean={wr['RAS'].mean():.2f}")
print(f"  breakout_age: {wr['has_breakout_age'].sum()} observed ({wr['has_breakout_age'].mean()*100:.1f}%)")
print(f"  peak_dominator: {wr['has_dominator'].sum()} observed ({wr['has_dominator'].mean()*100:.1f}%)")
print(f"  team_quality: range {wr['team_quality'].min():.0f}-{wr['team_quality'].max():.0f}")

# ============================================================================
# STEP 2: CORRELATION ANALYSIS (RAW FEATURES)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CORRELATION ANALYSIS")
print("="*80)

features_to_check = [
    ('log_pick', 'Log(draft pick)'),
    ('inv_sqrt_pick', '1/sqrt(pick)'),
    ('RAS_filled', 'RAS'),
    ('breakout_age_filled', 'Breakout age'),
    ('peak_dom_filled', 'Peak dominator'),
    ('career_yards_filled', 'Career yards'),
    ('best_yards_filled', 'Best season yards'),
    ('team_quality', 'Team quality'),
    ('has_ras', 'Has RAS (indicator)'),
    ('has_breakout_age', 'Has breakout age'),
]

print("\nCorrelation with NFL PPG:")
print("-"*60)
print(f"{'Feature':<25} {'r':>10} {'p-value':>12} {'Significant':>12}")
print("-"*60)

for col, name in features_to_check:
    valid = wr[[col, 'best_ppg']].dropna()
    if len(valid) > 5:
        r, p = stats.pearsonr(valid[col], valid['best_ppg'])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{name:<25} {r:>10.3f} {p:>12.4f} {sig:>12}")

# ============================================================================
# STEP 3: BUILD REGRESSION MODELS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: REGRESSION MODELS")
print("="*80)

# Define feature sets to test
feature_sets = {
    'Basic (DC only)': ['inv_sqrt_pick'],

    'DC + Breakout': ['inv_sqrt_pick', 'breakout_age_filled', 'has_breakout_age'],

    'DC + Breakout + RAS': ['inv_sqrt_pick', 'breakout_age_filled', 'has_breakout_age',
                            'RAS_filled', 'has_ras'],

    'Full model': ['inv_sqrt_pick', 'breakout_age_filled', 'has_breakout_age',
                   'RAS_filled', 'has_ras', 'career_yards_filled',
                   'team_quality'],

    'Full + interactions': ['inv_sqrt_pick', 'breakout_age_filled', 'has_breakout_age',
                            'RAS_filled', 'has_ras', 'career_yards_filled',
                            'team_quality'],
}

# Prepare data
y = wr['best_ppg'].values

results = {}

for name, features in feature_sets.items():
    X = wr[features].values

    # Add interactions for full+interactions model
    if name == 'Full + interactions':
        # Add pick × has_ras interaction (does RAS missingness matter more for early picks?)
        pick_x_hasras = wr['inv_sqrt_pick'] * wr['has_ras']
        # Add pick × breakout interaction
        pick_x_breakout = wr['inv_sqrt_pick'] * (1 / wr['breakout_age_filled'])
        X = np.column_stack([X, pick_x_hasras, pick_x_breakout])
        features = features + ['pick×has_ras', 'pick×breakout']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit OLS with statsmodels for detailed output
    X_with_const = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_with_const).fit()

    # Store results
    results[name] = {
        'model': model,
        'features': features,
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
        'aic': model.aic,
        'predictions': model.predict(X_with_const),
        'scaler': scaler,
        'X': X
    }

    # Cross-validation R²
    cv_scores = cross_val_score(LinearRegression(), X_scaled, y, cv=5, scoring='r2')
    results[name]['cv_r2'] = cv_scores.mean()

# Print comparison
print("\nModel Comparison:")
print("-"*70)
print(f"{'Model':<25} {'R²':>8} {'Adj R²':>8} {'CV R²':>8} {'AIC':>10}")
print("-"*70)

for name, res in results.items():
    print(f"{name:<25} {res['r2']:>8.3f} {res['adj_r2']:>8.3f} {res['cv_r2']:>8.3f} {res['aic']:>10.1f}")

# ============================================================================
# STEP 4: DETAILED COEFFICIENTS FOR BEST MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 4: FULL MODEL COEFFICIENTS")
print("="*80)

best_model_name = 'Full model'
best = results[best_model_name]
model = best['model']
features = best['features']

print(f"\n{best_model_name} (R² = {best['r2']:.3f}, CV R² = {best['cv_r2']:.3f})")
print("\n" + "-"*70)
print(f"{'Feature':<25} {'Coef':>10} {'Std Err':>10} {'t':>8} {'P>|t|':>10}")
print("-"*70)

# Intercept
print(f"{'Intercept':<25} {model.params[0]:>10.3f} {model.bse[0]:>10.3f} "
      f"{model.tvalues[0]:>8.2f} {model.pvalues[0]:>10.4f}")

for i, feat in enumerate(features):
    coef = model.params[i+1]
    se = model.bse[i+1]
    t = model.tvalues[i+1]
    p = model.pvalues[i+1]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{feat:<25} {coef:>10.3f} {se:>10.3f} {t:>8.2f} {p:>10.4f} {sig}")

print("\nInterpretation:")
print("  - Positive coef on 'has_ras' means having RAS data is associated with HIGHER PPG")
print("  - Negative coef on 'has_ras' means MISSING RAS is associated with higher PPG")
print("  - (accounting for all other variables)")

# ============================================================================
# STEP 5: PREDICTIONS FOR KEY PLAYERS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: PREDICTIONS FOR KEY PLAYERS")
print("="*80)

# Use full model predictions
wr['predicted_ppg'] = results['Full model']['predictions']

# Normalize to 0-100 scale
pred_min = wr['predicted_ppg'].min()
pred_max = wr['predicted_ppg'].max()
wr['slap_regression'] = 100 * (wr['predicted_ppg'] - pred_min) / (pred_max - pred_min)

# Also calculate rank
wr['slap_rank'] = wr['slap_regression'].rank(ascending=False)

# Key players
key_players = ['Jaylen Waddle', 'Ja\'Marr Chase', 'Justin Jefferson', 'CeeDee Lamb',
               'Jordan Addison', 'Puka Nacua', 'Amon-Ra St. Brown', 'Nico Collins',
               'Drake London', 'DeVonta Smith', 'Garrett Wilson']

print("\n" + "-"*100)
print(f"{'Player':<22} {'Pick':>5} {'BrkAge':>7} {'RAS':>6} {'TmQual':>7} {'Pred PPG':>9} {'SLAP':>6} {'Rank':>5} {'Actual':>7}")
print("-"*100)

for name in key_players:
    player = wr[wr['player_name'] == name]
    if len(player) > 0:
        p = player.iloc[0]
        age_str = f"{p['breakout_age']:.0f}" if pd.notna(p['breakout_age']) else "N/A"
        ras_str = f"{p['RAS']:.1f}" if pd.notna(p['RAS']) else "N/A"
        hit_str = " HIT" if p['hit24'] == 1 else ""
        print(f"{name:<22} {p['pick']:>5.0f} {age_str:>7} {ras_str:>6} {p['team_quality']:>7.0f} "
              f"{p['predicted_ppg']:>9.1f} {p['slap_regression']:>6.1f} {p['slap_rank']:>5.0f} "
              f"{p['best_ppg']:>7.1f}{hit_str}")

# ============================================================================
# STEP 6: MODEL PERFORMANCE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("STEP 6: PERFORMANCE COMPARISON")
print("="*80)

# Calculate our current SLAP for comparison
AGE_SCORES = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 15, 25: 10}
def get_age_score(age):
    if pd.isna(age): return 25
    return AGE_SCORES.get(int(age), 10)

wr['breakout_score'] = wr['breakout_age'].apply(get_age_score)

# Normalize components
dc_mean, dc_std = wr['inv_sqrt_pick'].mean(), wr['inv_sqrt_pick'].std()
wr['dc_norm'] = 50 + (wr['inv_sqrt_pick'] - dc_mean) / dc_std * 15

br_mean, br_std = wr['breakout_score'].mean(), wr['breakout_score'].std()
wr['br_norm'] = 50 + (wr['breakout_score'] - br_mean) / br_std * 15

ras_mean, ras_std = wr['RAS_filled'].mean(), wr['RAS_filled'].std()
wr['ras_norm'] = 50 + (wr['RAS_filled'] - ras_mean) / ras_std * 15

# Current SLAP formula
wr['slap_current'] = wr['dc_norm'] * 0.50 + wr['br_norm'] * 0.35 + wr['ras_norm'] * 0.15

# Compare correlations
r_regression, _ = stats.pearsonr(wr['slap_regression'], wr['best_ppg'])
r_current, _ = stats.pearsonr(wr['slap_current'], wr['best_ppg'])
r_dc_only, _ = stats.pearsonr(wr['dc_norm'], wr['best_ppg'])

sp_regression, _ = stats.spearmanr(wr['slap_regression'], wr['best_ppg'])
sp_current, _ = stats.spearmanr(wr['slap_current'], wr['best_ppg'])
sp_dc_only, _ = stats.spearmanr(wr['dc_norm'], wr['best_ppg'])

# Hit rates
def get_top_quartile_hit_rate(score_col):
    q75 = wr[score_col].quantile(0.75)
    top = wr[wr[score_col] >= q75]
    return top['hit24'].mean() * 100

hit_regression = get_top_quartile_hit_rate('slap_regression')
hit_current = get_top_quartile_hit_rate('slap_current')
hit_dc = get_top_quartile_hit_rate('dc_norm')

print("\n" + "-"*60)
print(f"{'Metric':<25} {'DC-Only':>12} {'Current SLAP':>12} {'Regression':>12}")
print("-"*60)
print(f"{'Pearson r':<25} {r_dc_only:>12.3f} {r_current:>12.3f} {r_regression:>12.3f}")
print(f"{'Spearman r':<25} {sp_dc_only:>12.3f} {sp_current:>12.3f} {sp_regression:>12.3f}")
print(f"{'Top 25% Hit Rate':<25} {hit_dc:>11.1f}% {hit_current:>11.1f}% {hit_regression:>11.1f}%")

# ============================================================================
# STEP 7: TOP 20 BY REGRESSION SLAP
# ============================================================================
print("\n" + "="*80)
print("STEP 7: TOP 20 BY REGRESSION SLAP")
print("="*80)

top20 = wr.nlargest(20, 'slap_regression')

print("\n" + "-"*90)
print(f"{'Rank':>4} {'Player':<22} {'Pick':>5} {'SLAP':>6} {'Pred PPG':>9} {'Actual':>7} {'Hit':>5}")
print("-"*90)

for rank, (_, row) in enumerate(top20.iterrows(), 1):
    hit_str = "YES" if row['hit24'] == 1 else ""
    print(f"{rank:>4} {row['player_name']:<22} {row['pick']:>5.0f} "
          f"{row['slap_regression']:>6.1f} {row['predicted_ppg']:>9.1f} "
          f"{row['best_ppg']:>7.1f} {hit_str:>5}")

top20_hits = top20['hit24'].sum()
print(f"\nTop 20 Regression SLAP: {top20_hits}/20 hits ({top20_hits/20*100:.0f}%)")

# ============================================================================
# STEP 8: WHAT THE MODEL LEARNED ABOUT MISSINGNESS
# ============================================================================
print("\n" + "="*80)
print("STEP 8: WHAT THE MODEL LEARNED ABOUT MISSINGNESS")
print("="*80)

# Extract coefficients for missingness indicators
model = results['Full model']['model']
features = results['Full model']['features']

print("\nMissingness indicator effects (from regression):")
for i, feat in enumerate(features):
    if 'has_' in feat:
        coef = model.params[i+1]
        p = model.pvalues[i+1]
        direction = "POSITIVE" if coef > 0 else "NEGATIVE"
        sig = "significant" if p < 0.05 else "NOT significant"
        print(f"\n  {feat}:")
        print(f"    Coefficient: {coef:+.3f} ({direction})")
        print(f"    p-value: {p:.4f} ({sig})")

        if 'has_ras' in feat:
            if coef < 0:
                print(f"    → MISSING RAS associated with HIGHER NFL PPG (after controlling for other vars)")
                print(f"    → This confirms MNAR: elite opt-outs tend to be better players")
            else:
                print(f"    → HAVING RAS associated with higher NFL PPG")

# Save results
wr[['player_name', 'draft_year', 'pick', 'breakout_age', 'RAS', 'team_quality',
    'predicted_ppg', 'slap_regression', 'slap_current', 'best_ppg', 'hit24']].to_csv(
    'output/wr_slap_regression.csv', index=False)
print("\n\nSaved: output/wr_slap_regression.csv")
