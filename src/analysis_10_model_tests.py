"""
Analysis 10: Test new model components for WR SLAP Score.

Test 1: Enhanced Breakout Score (games-played proration)
Test 2: Teammate adjustment
Test 3: Size component
Test 4: Head-to-head model comparison

Uses all 339 WRs for non-PFF metrics, 284 for PFF-dependent tests.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.chdir("/home/user/SlapModelV3")

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def dc_score(pick):
    """Draft capital: DC = 100 - 2.40 * (pick^0.62 - 1)"""
    return 100 - 2.40 * (pick ** 0.62 - 1)


def wr_breakout_score(breakout_age, dominator_pct):
    """Current WR breakout scoring from CLAUDE.md."""
    if pd.isna(breakout_age):
        # Never broke out
        if pd.isna(dominator_pct):
            return 15.0
        return min(35, 15 + dominator_pct)

    age = int(breakout_age)
    base_scores = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base = base_scores.get(age, 20 if age >= 24 else 100)

    if pd.isna(dominator_pct) or dominator_pct < 20:
        bonus = 0
    else:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)

    return min(base + bonus, 99.9)


def partial_corr(x, y, controls):
    """
    Partial correlation of x and y controlling for one or more variables.
    controls: array or list of arrays
    """
    if isinstance(controls, np.ndarray) and controls.ndim == 1:
        controls = [controls]
    elif isinstance(controls, np.ndarray) and controls.ndim == 2:
        controls = [controls[:, i] for i in range(controls.shape[1])]

    # Build valid mask
    valid = ~(np.isnan(x) | np.isnan(y))
    for c in controls:
        valid = valid & ~np.isnan(c)

    x_v = x[valid]
    y_v = y[valid]
    c_v = np.column_stack([c[valid] for c in controls])
    n = len(x_v)

    if n < 15:
        return np.nan, np.nan, n

    # Residualize x on controls
    X = np.column_stack([np.ones(n), c_v])
    coef_x = np.linalg.lstsq(X, x_v, rcond=None)[0]
    resid_x = x_v - X @ coef_x

    # Residualize y on controls
    coef_y = np.linalg.lstsq(X, y_v, rcond=None)[0]
    resid_y = y_v - X @ coef_y

    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, n


def normalize_0_100(series):
    """Normalize a series to 0-100 scale."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return ((series - mn) / (mx - mn)) * 100


def calc_model_score(df, weights, components):
    """
    Calculate weighted model score.
    weights: dict of component_name -> weight (should sum to 1)
    components: dict of component_name -> column_name in df
    Returns scores with position average imputation for missing values.
    """
    score = pd.Series(0.0, index=df.index)
    for name, weight in weights.items():
        col = components[name]
        vals = df[col].copy()
        # Impute missing with mean
        vals = vals.fillna(vals.mean())
        score += vals * weight
    return score


def corr_table(df, score_col, outcomes):
    """Compute correlations between a score column and outcome columns."""
    results = {}
    for out in outcomes:
        valid = df[[score_col, out]].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid[score_col], valid[out])
            results[out] = (r, p, len(valid))
        else:
            results[out] = (np.nan, np.nan, 0)
    return results


# ===========================================================================
# LOAD ALL DATA
# ===========================================================================
print("=" * 90)
print("ANALYSIS 10: MODEL COMPONENT TESTS")
print("=" * 90)

wr_bt = pd.read_csv('data/wr_backtest_expanded_final.csv')
outcomes_full = pd.read_csv('data/backtest_outcomes_complete.csv')
pff = pd.read_csv('data/wr_pff_all_2016_2025.csv')
games = pd.read_csv('data/wr_games_played.csv')
teammate = pd.read_csv('data/wr_teammate_scores.csv')
combine = pd.read_parquet('data/nflverse/combine.parquet')
dominator = pd.read_csv('data/wr_dominator_complete.csv')

# College receiving for team_pass_att
college_rec = pd.read_csv('data/college_receiving_2011_2023.csv')
# Get team_pass_att per school per season (use max per group since all rows share it)
team_pa = college_rec.groupby(['college', 'season'])['team_pass_att'].first().reset_index()
team_pa.columns = ['cfbd_college', 'season', 'team_pass_att']

# Build master WR analysis dataframe
wr = wr_bt.copy()
wr['dc'] = wr['pick'].apply(dc_score)

# Merge outcomes
out_wr = outcomes_full[outcomes_full['position'] == 'WR'][
    ['player_name', 'draft_year', 'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
]
wr = wr.merge(out_wr, on=['player_name', 'draft_year'], how='left',
              suffixes=('', '_out'))
# Use the outcomes version of hit24/hit12 (more complete)
if 'hit24_out' in wr.columns:
    wr['hit24'] = wr['hit24_out'].fillna(wr['hit24'])
    wr['hit12'] = wr['hit12_out'].fillna(wr['hit12'])
    wr.drop(columns=['hit24_out', 'hit12_out'], errors='ignore', inplace=True)

# Merge PFF data
pff_cols = ['player_name', 'draft_year', 'player_game_count', 'yprr', 'yards', 'receptions',
            'targets', 'routes', 'grades_offense', 'grades_pass_route', 'drop_rate',
            'contested_catch_rate', 'avg_depth_of_target', 'yards_after_catch',
            'yards_after_catch_per_reception', 'slot_rate', 'wide_rate', 'caught_percent',
            'drops', 'rush_attempts', 'rush_yards']
wr = wr.merge(pff[pff_cols], on=['player_name', 'draft_year'], how='left')

# Merge games played
wr = wr.merge(games[['player_name', 'draft_year', 'games_played', 'games_source']],
              on=['player_name', 'draft_year'], how='left')

# Merge teammate scores
wr = wr.merge(teammate[['player_name', 'draft_year', 'teammate_count',
                         'best_teammate_pick', 'total_teammate_dc', 'avg_teammate_dc']],
              on=['player_name', 'draft_year'], how='left')

# Merge combine size
wr_combine = combine[combine['pos'] == 'WR'][['player_name', 'draft_year', 'ht', 'wt']].copy()
wr_combine.columns = ['player_name', 'draft_year', 'height_str', 'weight']

def parse_height(h):
    """Parse '6-2' format to inches."""
    if pd.isna(h):
        return np.nan
    parts = str(h).split('-')
    if len(parts) == 2:
        try:
            return int(parts[0]) * 12 + int(parts[1])
        except ValueError:
            return np.nan
    return np.nan

wr_combine['height'] = wr_combine['height_str'].apply(parse_height)
wr = wr.merge(wr_combine[['player_name', 'draft_year', 'height', 'weight']],
              on=['player_name', 'draft_year'], how='left')

# Merge dominator data (for enhanced breakout)
dom_final = dominator.groupby(['player_name', 'draft_year']).agg({
    'player_rec_yards': 'first',
    'team_rec_yards': 'first',
    'dominator_pct': 'first',
    'season_used': 'first'
}).reset_index()
wr = wr.merge(dom_final[['player_name', 'draft_year', 'player_rec_yards',
                           'team_rec_yards', 'season_used']],
              on=['player_name', 'draft_year'], how='left')

# Calculate BMI
wr['bmi'] = wr['weight'] / (wr['height'] ** 2) * 703  # BMI in imperial

# Calculate current breakout score
wr['breakout_score'] = wr.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1
)

# Calculate current full SLAP score (for reference)
wr['ras_norm'] = normalize_0_100(wr['RAS'].fillna(wr['RAS'].mean()))
wr['current_slap'] = wr['dc'] * 0.65 + wr['breakout_score'] * 0.20 + wr['ras_norm'] * 0.15

print(f"Master dataset: {len(wr)} WRs")
print(f"With outcomes: {wr['hit24'].notna().sum()}")
print(f"With PFF: {wr['yprr'].notna().sum()}")
print(f"With games: {wr['games_played'].notna().sum()}")
print(f"With size: {wr['height'].notna().sum()}")
print(f"With teammates: {(wr['teammate_count'] > 0).sum()}")

OUTCOMES = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']

# ===========================================================================
# TEST 1: ENHANCED BREAKOUT SCORE
# ===========================================================================
print("\n" + "=" * 90)
print("TEST 1: ENHANCED BREAKOUT SCORE (Games-Played Proration)")
print("=" * 90)

# Get team_pass_att for each WR's final college season
# Map backtest college names to CFBD college names
COLLEGE_TO_CFBD = {
    'Alabama': 'Alabama', 'Arizona': 'Arizona', 'Arizona St.': 'Arizona State',
    'Arkansas': 'Arkansas', 'Auburn': 'Auburn', 'Baylor': 'Baylor',
    'Boise St.': 'Boise State', 'BYU': 'BYU', 'California': 'California',
    'Central Florida': 'UCF', 'Cincinnati': 'Cincinnati', 'Clemson': 'Clemson',
    'Colorado': 'Colorado', 'Colorado St.': 'Colorado State', 'Duke': 'Duke',
    'East Carolina': 'East Carolina', 'Florida': 'Florida',
    'Florida St.': 'Florida State', 'Fresno St.': 'Fresno State',
    'Georgia': 'Georgia', 'Georgia Tech': 'Georgia Tech',
    'Houston': 'Houston', 'Illinois': 'Illinois', 'Iowa': 'Iowa',
    'Iowa St.': 'Iowa State', 'Iowa State': 'Iowa State',
    'Kansas St.': 'Kansas State', 'Kentucky': 'Kentucky',
    'LSU': 'LSU', 'Liberty': 'Liberty', 'Louisville': 'Louisville',
    'Maryland': 'Maryland', 'Memphis': 'Memphis', 'Miami (FL)': 'Miami',
    'Michigan': 'Michigan', 'Michigan St.': 'Michigan State',
    'Minnesota': 'Minnesota', 'Mississippi': 'Ole Miss', 'Missouri': 'Missouri',
    'Nebraska': 'Nebraska', 'Nevada': 'Nevada', 'North Carolina': 'North Carolina',
    'North Carolina St.': 'NC State', 'North Texas': 'North Texas',
    'Northern Illinois': 'Northern Illinois', 'Notre Dame': 'Notre Dame',
    'Ohio St.': 'Ohio State', 'Ohio State': 'Ohio State',
    'Oklahoma': 'Oklahoma', 'Oklahoma St.': 'Oklahoma State',
    'Old Dominion': 'Old Dominion', 'Ole Miss': 'Ole Miss',
    'Oregon': 'Oregon', 'Oregon St.': 'Oregon State',
    'Penn St.': 'Penn State', 'Pittsburgh': 'Pittsburgh',
    'Purdue': 'Purdue', 'Rice': 'Rice', 'Rutgers': 'Rutgers',
    'SMU': 'SMU', 'South Alabama': 'South Alabama',
    'South Carolina': 'South Carolina', 'South Florida': 'USF',
    'Southern Miss': 'Southern Miss', 'Stanford': 'Stanford',
    'TCU': 'TCU', 'Tennessee': 'Tennessee', 'Texas': 'Texas',
    'Texas A&M': 'Texas A&M', 'Texas Tech': 'Texas Tech',
    'Toledo': 'Toledo', 'Tulane': 'Tulane', 'UCLA': 'UCLA',
    'USC': 'USC', 'Utah': 'Utah', 'Virginia': 'Virginia',
    'Virginia Tech': 'Virginia Tech', 'Wake Forest': 'Wake Forest',
    'Washington': 'Washington', 'Washington St.': 'Washington State',
    'Washington State': 'Washington State', 'West Virginia': 'West Virginia',
    'Western Kentucky': 'Western Kentucky', 'Western Michigan': 'Western Michigan',
    'Wisconsin': 'Wisconsin', 'Connecticut': 'Connecticut',
    'Massachusetts': 'UMass', 'Ala-Birmingham': 'UAB',
    'Boston Col.': 'Boston College', 'Bowling Green': 'Bowling Green',
    'Charlotte': 'Charlotte', 'Georgia St.': 'Georgia State',
    'Hawaii': "Hawai'i", 'La-Monroe': 'Louisiana Monroe',
    'Louisiana Tech': 'Louisiana Tech', 'Middle Tenn. St.': 'Middle Tennessee',
    'New Mexico St.': 'New Mexico State',
}

# Look up team_pass_att
wr['cfbd_college'] = wr['college'].map(COLLEGE_TO_CFBD)
wr['final_season'] = wr['draft_year'] - 1

# Merge team_pass_att
wr = wr.merge(
    team_pa,
    left_on=['cfbd_college', 'final_season'],
    right_on=['cfbd_college', 'season'],
    how='left'
)
if 'season' in wr.columns:
    wr.drop(columns=['season'], inplace=True)

print(f"\nteam_pass_att coverage: {wr['team_pass_att'].notna().sum()}/{len(wr)}")

# For WRs with PFF data, use PFF yards; for others use dominator data
wr['rec_yards_final'] = wr['yards']  # PFF yards
wr.loc[wr['rec_yards_final'].isna(), 'rec_yards_final'] = wr.loc[
    wr['rec_yards_final'].isna(), 'player_rec_yards']

print(f"rec_yards coverage: {wr['rec_yards_final'].notna().sum()}/{len(wr)}")

# Calculate age at draft (approximate from birthdate)
wr['birth_date'] = pd.to_datetime(wr['birthdate'], errors='coerce')
wr['draft_date'] = pd.to_datetime(wr['draft_year'].astype(str) + '-04-25')
wr['age_at_draft'] = (wr['draft_date'] - wr['birth_date']).dt.days / 365.25

# Use final_season age = age_at_draft - 1 approximately
# Or we can use the season_used year and birthdate
wr['season_age'] = wr['age_at_draft'] - 1  # approximate age during final college season

# Age weight function (same as RB formula from CLAUDE.md)
def age_weight(age):
    if pd.isna(age):
        return 1.0
    age = round(age)
    weights = {19: 1.15, 20: 1.10, 21: 1.05, 22: 1.00, 23: 0.95}
    if age <= 19:
        return 1.15
    if age >= 24:
        return 0.90
    return weights.get(age, 1.00)

wr['age_wt'] = wr['season_age'].apply(age_weight)

# Use games_played (prefer PFF player_game_count, fall back to games_played from merged)
wr['gp'] = wr['player_game_count']
wr.loc[wr['gp'].isna(), 'gp'] = wr.loc[wr['gp'].isna(), 'games_played']

# Assume ~13 games per team season (standard FBS)
TEAM_GAMES_AVG = 13.0

# Enhanced production: prorate for missed games, then normalize by team_pass_att
# prorated_yards = rec_yards * (TEAM_GAMES / games_played)
# enhanced_raw = prorated_yards / team_pass_att * age_weight
wr['prorated_yards'] = np.where(
    (wr['rec_yards_final'].notna()) & (wr['gp'].notna()) & (wr['gp'] > 0),
    wr['rec_yards_final'] * (TEAM_GAMES_AVG / wr['gp']),
    wr['rec_yards_final']
)

wr['enhanced_raw'] = np.where(
    (wr['prorated_yards'].notna()) & (wr['team_pass_att'].notna()) & (wr['team_pass_att'] > 0),
    (wr['prorated_yards'] / wr['team_pass_att']) * wr['age_wt'] * 100,
    np.nan
)

# Also compute the non-prorated version for comparison
wr['raw_production'] = np.where(
    (wr['rec_yards_final'].notna()) & (wr['team_pass_att'].notna()) & (wr['team_pass_att'] > 0),
    (wr['rec_yards_final'] / wr['team_pass_att']) * wr['age_wt'] * 100,
    np.nan
)

# Normalize enhanced scores to 0-100
for col in ['enhanced_raw', 'raw_production']:
    valid = wr[col].notna()
    if valid.sum() > 0:
        mn, mx = wr.loc[valid, col].min(), wr.loc[valid, col].max()
        if mx > mn:
            wr[col + '_norm'] = ((wr[col] - mn) / (mx - mn)) * 100
        else:
            wr[col + '_norm'] = 50.0

print(f"\nenhanced_raw coverage: {wr['enhanced_raw'].notna().sum()}/{len(wr)}")
print(f"games_played used for proration: {(wr['gp'].notna() & wr['rec_yards_final'].notna()).sum()}")

# Show proration effect: biggest differences
wr['proration_factor'] = np.where(
    (wr['gp'].notna()) & (wr['gp'] > 0),
    TEAM_GAMES_AVG / wr['gp'],
    1.0
)

print(f"\n--- Proration Effect (top 15 biggest boosts) ---")
prorated = wr[(wr['enhanced_raw'].notna()) & (wr['hit24'].notna())].copy()
prorated['boost'] = prorated['proration_factor'] - 1.0
top_prorated = prorated.nlargest(15, 'boost')
print(f"{'Player':<25} {'College':<18} {'Year':>5} {'Games':>6} {'Factor':>7} {'Raw':>8} {'Enhanced':>10} {'Hit24':>6}")
for _, r in top_prorated.iterrows():
    h = 'YES' if r['hit24'] == 1 else 'NO'
    print(f"  {r['player_name']:<23} {str(r['college'])[:16]:<18} {int(r['draft_year']):>5} "
          f"{r['gp']:>6.0f} {r['proration_factor']:>6.2f}x {r['raw_production']:>7.1f} "
          f"{r['enhanced_raw']:>9.1f} {h:>6}")

# Correlation comparison
print(f"\n--- Correlation Comparison: Current Breakout vs Enhanced vs Raw Production ---")
print(f"{'Metric':<30} {'hit24':>12} {'hit12':>12} {'3yr_ppg':>12} {'career_ppg':>12} {'N':>5}")

for label, col in [
    ('Current breakout_score', 'breakout_score'),
    ('Raw production (no prorate)', 'raw_production'),
    ('Enhanced (prorated)', 'enhanced_raw'),
]:
    vals = []
    n_val = 0
    for out in OUTCOMES:
        valid = wr[[col, out]].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid[col], valid[out])
            sig = '*' if p < 0.05 else ''
            vals.append(f"{r:+.3f}{sig}")
            n_val = len(valid)
        else:
            vals.append("n/a")
    print(f"  {label:<28} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12} {n_val:>5}")

# Partial correlations controlling for DC
print(f"\n--- Partial Correlations (controlling for DC) ---")
print(f"{'Metric':<30} {'hit24':>12} {'hit12':>12} {'3yr_ppg':>12} {'career_ppg':>12} {'N':>5}")

for label, col in [
    ('Current breakout_score', 'breakout_score'),
    ('Raw production (no prorate)', 'raw_production'),
    ('Enhanced (prorated)', 'enhanced_raw'),
]:
    vals = []
    n_val = 0
    for out in OUTCOMES:
        valid = wr[[col, out, 'dc']].dropna()
        if len(valid) > 15:
            r, p, n = partial_corr(valid[col].values, valid[out].values,
                                   valid['dc'].values)
            sig = '*' if p < 0.05 else ''
            vals.append(f"{r:+.3f}{sig}")
            n_val = n
        else:
            vals.append("n/a")
    print(f"  {label:<28} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12} {n_val:>5}")


# ===========================================================================
# TEST 2: TEAMMATE ADJUSTMENT
# ===========================================================================
print("\n\n" + "=" * 90)
print("TEST 2: TEAMMATE ADJUSTMENT")
print("=" * 90)

# Create binary elite_program flag
wr['elite_program'] = (wr['total_teammate_dc'] > 150).astype(float)

# For best_teammate_pick, lower = better. Invert for scoring.
# Convert to a 0-100 scale where higher = more elite teammate
wr['best_tm_score'] = np.where(
    wr['best_teammate_pick'].notna(),
    dc_score(wr['best_teammate_pick']),
    0.0  # No teammate = 0 score
)

print(f"\nElite program (>150 DC): {int(wr['elite_program'].sum())} WRs")
print(f"Non-elite: {int((wr['elite_program'] == 0).sum())} WRs")

# A: total_teammate_dc continuous
# B: elite_program binary
# C: best_teammate_pick continuous (as best_tm_score)
print(f"\n--- Partial Correlations (controlling for DC + Breakout) ---")
print(f"{'Approach':<35} {'hit24':>12} {'hit12':>12} {'3yr_ppg':>12} {'career_ppg':>12} {'N':>5}")

controls_cols = ['dc', 'breakout_score']
for label, col in [
    ('A: total_teammate_dc (contin.)', 'total_teammate_dc'),
    ('B: elite_program (binary >150)', 'elite_program'),
    ('C: best_tm_score (contin.)', 'best_tm_score'),
]:
    vals = []
    n_val = 0
    for out in OUTCOMES:
        needed = [col, out] + controls_cols
        valid = wr[needed].dropna()
        if len(valid) > 15:
            r, p, n = partial_corr(
                valid[col].values, valid[out].values,
                np.column_stack([valid[c].values for c in controls_cols])
            )
            sig = '*' if p < 0.05 else ''
            vals.append(f"{r:+.3f}{sig}")
            n_val = n
        else:
            vals.append("n/a")
    print(f"  {label:<33} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12} {n_val:>5}")

# Also show hit rates for elite vs non-elite
print(f"\n--- Hit Rates: Elite Program vs Non-Elite ---")
with_out = wr[wr['hit24'].notna()].copy()
for flag_val, label in [(1, 'Elite (>150 DC)'), (0, 'Non-elite')]:
    sub = with_out[with_out['elite_program'] == flag_val]
    if len(sub) > 0:
        print(f"  {label:<20} N={len(sub):>4}  Hit24={sub['hit24'].mean():.1%}  "
              f"Hit12={sub['hit12'].mean():.1%}  3yr_ppg={sub['first_3yr_ppg'].mean():.1f}")


# ===========================================================================
# TEST 3: SIZE COMPONENT
# ===========================================================================
print("\n\n" + "=" * 90)
print("TEST 3: SIZE COMPONENT")
print("=" * 90)

print(f"\nSize data coverage: height={wr['height'].notna().sum()}, "
      f"weight={wr['weight'].notna().sum()}, BMI={wr['bmi'].notna().sum()}")

# Raw correlations
print(f"\n--- Raw Correlations ---")
print(f"{'Metric':<20} {'hit24':>12} {'hit12':>12} {'3yr_ppg':>12} {'career_ppg':>12} {'N':>5}")
for label, col in [('Height', 'height'), ('Weight', 'weight'), ('BMI', 'bmi')]:
    vals = []
    n_val = 0
    for out in OUTCOMES:
        valid = wr[[col, out]].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid[col], valid[out])
            sig = '*' if p < 0.05 else ''
            vals.append(f"{r:+.3f}{sig}")
            n_val = len(valid)
        else:
            vals.append("n/a")
    print(f"  {label:<18} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12} {n_val:>5}")

# Partial correlations controlling for DC + breakout
print(f"\n--- Partial Correlations (controlling for DC + Breakout) ---")
print(f"{'Metric':<20} {'hit24':>12} {'hit12':>12} {'3yr_ppg':>12} {'career_ppg':>12} {'N':>5}")
for label, col in [('Height', 'height'), ('Weight', 'weight'), ('BMI', 'bmi')]:
    vals = []
    n_val = 0
    for out in OUTCOMES:
        needed = [col, out, 'dc', 'breakout_score']
        valid = wr[needed].dropna()
        if len(valid) > 15:
            r, p, n = partial_corr(
                valid[col].values, valid[out].values,
                np.column_stack([valid['dc'].values, valid['breakout_score'].values])
            )
            sig = '*' if p < 0.05 else ''
            vals.append(f"{r:+.3f}{sig}")
            n_val = n
        else:
            vals.append("n/a")
    print(f"  {label:<18} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12} {n_val:>5}")

# Round 2 specifically (where exploratory showed bigger WRs hit more)
print(f"\n--- Round 2 Only: Partial Correlations (controlling for DC + Breakout) ---")
rd2 = wr[wr['round'] == 2].copy()
print(f"Round 2 WRs: {len(rd2)}, with outcomes: {rd2['hit24'].notna().sum()}")
print(f"{'Metric':<20} {'hit24':>12} {'hit12':>12} {'3yr_ppg':>12} {'career_ppg':>12} {'N':>5}")
for label, col in [('Height', 'height'), ('Weight', 'weight'), ('BMI', 'bmi')]:
    vals = []
    n_val = 0
    for out in OUTCOMES:
        needed = [col, out, 'dc', 'breakout_score']
        valid = rd2[needed].dropna()
        if len(valid) > 15:
            r, p, n = partial_corr(
                valid[col].values, valid[out].values,
                np.column_stack([valid['dc'].values, valid['breakout_score'].values])
            )
            sig = '*' if p < 0.05 else ''
            vals.append(f"{r:+.3f}{sig}")
            n_val = n
        else:
            vals.append("n/a")
    print(f"  {label:<18} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12} {n_val:>5}")


# ===========================================================================
# TEST 4: HEAD-TO-HEAD MODEL COMPARISON
# ===========================================================================
print("\n\n" + "=" * 90)
print("TEST 4: HEAD-TO-HEAD MODEL COMPARISON")
print("=" * 90)

# Prepare normalized components
# DC score: already 0-100 ish
# Breakout score: already 0-100
# Enhanced breakout: normalize to 0-100
wr['enhanced_norm'] = wr['enhanced_raw_norm'].copy() if 'enhanced_raw_norm' in wr.columns else np.nan
# Fill missing enhanced with breakout score (graceful fallback)
wr['enhanced_norm_filled'] = wr['enhanced_norm'].fillna(wr['breakout_score'])

# Teammate: normalize total_teammate_dc to 0-100
wr['tm_norm'] = normalize_0_100(wr['total_teammate_dc'].fillna(0))

# Best teammate score: already 0-100ish from dc_score
wr['best_tm_norm'] = wr['best_tm_score']  # Already on DC scale

# Elite program: 0 or 100
wr['elite_norm'] = wr['elite_program'] * 100

# Size: normalize height and weight to 0-100
wr['height_norm'] = normalize_0_100(wr['height'].fillna(wr['height'].mean()))
wr['weight_norm'] = normalize_0_100(wr['weight'].fillna(wr['weight'].mean()))

# Drop rate: normalize (lower = better, so invert)
wr['drop_rate_inv'] = 100 - normalize_0_100(wr['drop_rate'].fillna(wr['drop_rate'].mean()))

# RAS normalized
wr['ras_norm'] = normalize_0_100(wr['RAS'].fillna(wr['RAS'].mean()))

# Define model configurations
models = {}

# A: Current model — 65/20/15 DC/Breakout/RAS
models['A: Current (65/20/15 DC/BA/RAS)'] = {
    'weights': {'dc': 0.65, 'breakout': 0.20, 'ras': 0.15},
    'cols': {'dc': 'dc', 'breakout': 'breakout_score', 'ras': 'ras_norm'},
    'full_sample': True,
}

# B: Two-factor — 75/25 DC/Breakout
models['B: Two-factor (75/25 DC/BA)'] = {
    'weights': {'dc': 0.75, 'breakout': 0.25},
    'cols': {'dc': 'dc', 'breakout': 'breakout_score'},
    'full_sample': True,
}

# C: DC/Enhanced Breakout/Teammate — multiple weight splits
for dc_w, eb_w, tm_w in [
    (0.60, 0.25, 0.15), (0.55, 0.25, 0.20), (0.60, 0.20, 0.20),
    (0.65, 0.20, 0.15), (0.50, 0.30, 0.20), (0.55, 0.30, 0.15),
    (0.70, 0.15, 0.15),
]:
    label = f"C: DC/EnhBA/Tm ({int(dc_w*100)}/{int(eb_w*100)}/{int(tm_w*100)})"
    models[label] = {
        'weights': {'dc': dc_w, 'enhanced': eb_w, 'teammate': tm_w},
        'cols': {'dc': 'dc', 'enhanced': 'enhanced_norm_filled', 'teammate': 'tm_norm'},
        'full_sample': True,
    }

# D: DC/Enhanced Breakout/Teammate/Size
for dc_w, eb_w, tm_w, sz_w in [
    (0.55, 0.20, 0.15, 0.10), (0.50, 0.25, 0.15, 0.10),
    (0.55, 0.25, 0.10, 0.10), (0.60, 0.20, 0.10, 0.10),
    (0.50, 0.20, 0.20, 0.10), (0.45, 0.25, 0.20, 0.10),
]:
    label = f"D: DC/EnhBA/Tm/Sz ({int(dc_w*100)}/{int(eb_w*100)}/{int(tm_w*100)}/{int(sz_w*100)})"
    models[label] = {
        'weights': {'dc': dc_w, 'enhanced': eb_w, 'teammate': tm_w, 'size': sz_w},
        'cols': {'dc': 'dc', 'enhanced': 'enhanced_norm_filled', 'teammate': 'tm_norm', 'size': 'weight_norm'},
        'full_sample': True,
    }

# E: Kitchen sink — DC/Enhanced Breakout/Teammate/Drop Rate/Size (PFF sample only)
for dc_w, eb_w, tm_w, dr_w, sz_w in [
    (0.50, 0.20, 0.10, 0.10, 0.10), (0.45, 0.20, 0.15, 0.10, 0.10),
    (0.50, 0.15, 0.15, 0.10, 0.10), (0.55, 0.15, 0.10, 0.10, 0.10),
    (0.45, 0.25, 0.10, 0.10, 0.10), (0.40, 0.20, 0.15, 0.15, 0.10),
]:
    label = f"E: Sink ({int(dc_w*100)}/{int(eb_w*100)}/{int(tm_w*100)}/{int(dr_w*100)}/{int(sz_w*100)})"
    models[label] = {
        'weights': {'dc': dc_w, 'enhanced': eb_w, 'teammate': tm_w, 'droprate': dr_w, 'size': sz_w},
        'cols': {'dc': 'dc', 'enhanced': 'enhanced_norm_filled', 'teammate': 'tm_norm',
                 'droprate': 'drop_rate_inv', 'size': 'weight_norm'},
        'full_sample': False,  # PFF sample only
    }

# Run all models
print(f"\n{'Model':<48} {'hit24':>8} {'hit12':>8} {'3yr_ppg':>9} {'car_ppg':>9} {'Avg_r':>7} {'N':>5}")
print("-" * 100)

model_results = []

for model_name, config in models.items():
    weights = config['weights']
    cols = config['cols']
    full_sample = config['full_sample']

    # Calculate model score
    score = pd.Series(0.0, index=wr.index)
    for comp_name, weight in weights.items():
        col_name = cols[comp_name]
        vals = wr[col_name].copy()
        vals = vals.fillna(vals.mean())
        score += vals * weight

    wr['_model_score'] = score

    # Select sample
    if full_sample:
        df_eval = wr[wr['hit24'].notna()].copy()
    else:
        # PFF sample: must have drop_rate
        df_eval = wr[(wr['hit24'].notna()) & (wr['drop_rate'].notna())].copy()

    if len(df_eval) < 20:
        continue

    # Correlations
    row_results = {'model': model_name, 'n': len(df_eval)}
    corr_vals = []
    for out in OUTCOMES:
        valid = df_eval[['_model_score', out]].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid['_model_score'], valid[out])
            row_results[out] = r
            corr_vals.append(r)
        else:
            row_results[out] = np.nan
            corr_vals.append(np.nan)

    row_results['avg_r'] = np.nanmean(corr_vals)
    model_results.append(row_results)

    # Print
    vals_str = []
    for out in OUTCOMES:
        r = row_results.get(out, np.nan)
        if not np.isnan(r):
            vals_str.append(f"{r:+.3f}")
        else:
            vals_str.append("  n/a")
    avg_r = row_results['avg_r']
    n = row_results['n']

    # Highlight if this is a baseline model
    prefix = ">>>" if model_name.startswith('A:') or model_name.startswith('B:') else "   "
    print(f"{prefix}{model_name:<45} {vals_str[0]:>8} {vals_str[1]:>8} "
          f"{vals_str[2]:>9} {vals_str[3]:>9} {avg_r:>6.3f} {n:>5}")

    if model_name.startswith('B:') or model_name.startswith('C: DC/EnhBA/Tm (55/25/20)'):
        print("-" * 100)

# Sort and show best models
print(f"\n{'='*60}")
print("TOP 10 MODELS BY AVERAGE CORRELATION")
print(f"{'='*60}")

results_df = pd.DataFrame(model_results)
results_df = results_df.sort_values('avg_r', ascending=False)

for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
    vals = []
    for out in OUTCOMES:
        r = row.get(out, np.nan)
        vals.append(f"{r:+.3f}" if not np.isnan(r) else "n/a")
    print(f"  {i}. {row['model']:<48} avg_r={row['avg_r']:.3f}  "
          f"hit24={vals[0]}  N={int(row['n'])}")

# ===========================================================================
# TIER ANALYSIS for top 3 models
# ===========================================================================
print(f"\n{'='*60}")
print("TIER BREAKDOWN FOR TOP MODELS")
print(f"{'='*60}")

top3_models = results_df.head(3)['model'].tolist()
# Also include baselines
all_to_tier = ['A: Current (65/20/15 DC/BA/RAS)', 'B: Two-factor (75/25 DC/BA)'] + top3_models
all_to_tier = list(dict.fromkeys(all_to_tier))  # dedupe preserving order

for model_name in all_to_tier:
    config = models[model_name]
    weights = config['weights']
    cols = config['cols']

    score = pd.Series(0.0, index=wr.index)
    for comp_name, weight in weights.items():
        col_name = cols[comp_name]
        vals = wr[col_name].copy()
        vals = vals.fillna(vals.mean())
        score += vals * weight

    wr['_tier_score'] = score

    if config['full_sample']:
        df_eval = wr[wr['hit24'].notna()].copy()
    else:
        df_eval = wr[(wr['hit24'].notna()) & (wr['drop_rate'].notna())].copy()

    df_eval['_tier_score'] = wr.loc[df_eval.index, '_tier_score']

    print(f"\n--- {model_name} ---")
    print(f"{'Tier':<20} {'N':>4} {'Hit24%':>8} {'Hit12%':>8} {'3yr PPG':>9} {'Car PPG':>9}")

    # Create quintile tiers
    df_eval['tier'] = pd.qcut(df_eval['_tier_score'], 5,
                               labels=['Bottom 20%', 'Low 20%', 'Mid 20%', 'High 20%', 'Top 20%'],
                               duplicates='drop')

    for tier_name in ['Top 20%', 'High 20%', 'Mid 20%', 'Low 20%', 'Bottom 20%']:
        t = df_eval[df_eval['tier'] == tier_name]
        if len(t) > 0:
            h24 = t['hit24'].mean()
            h12 = t['hit12'].mean()
            ppg3 = t['first_3yr_ppg'].mean()
            ppgc = t['career_ppg'].mean()
            print(f"  {tier_name:<18} {len(t):>4} {h24:>7.1%} {h12:>7.1%} {ppg3:>8.1f} {ppgc:>8.1f}")


# ===========================================================================
# SAVE OUTPUT
# ===========================================================================
output_file = 'output/analysis_10_model_tests_output.txt'
print(f"\n{'='*90}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*90}")
