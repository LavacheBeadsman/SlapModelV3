"""
SLAP Score V3 - Unified Calculator
Weights: DC (85%) + Breakout Age (10%) + RAS (5%)

Key Design Decisions:
- Draft Capital dominates (85%) because it's the strongest predictor
- Breakout Age adds marginal signal (10%) - earlier breakouts slightly predict success
- RAS adds content value (5%) - small but non-zero effect, useful for discussions
- Missing RAS: use position average (neutral impact), flag as imputed
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("SLAP SCORE V3 - UNIFIED CALCULATOR")
print("Weights: DC (85%) + Breakout Age (10%) + RAS (5%)")
print("=" * 90)

# ============================================================================
# CONFIGURATION
# ============================================================================
WEIGHT_DC = 0.85
WEIGHT_BREAKOUT = 0.10
WEIGHT_RAS = 0.05

# Breakout age scoring (same for WRs and RBs)
BREAKOUT_AGE_SCORES = {
    18: 100,  # Freshman breakout = elite
    19: 90,
    20: 75,
    21: 60,
    22: 45,
    23: 30,
    24: 20,
}

# RB Manual breakout ages (from research)
RB_BREAKOUT_AGES = {
    # 2015 class
    'Todd Gurley': 19, 'Melvin Gordon': 20, 'David Johnson': 21,
    'Tevin Coleman': 20, 'Duke Johnson': 18, 'T.J. Yeldon': 18,
    'Ameer Abdullah': 20, 'Jay Ajayi': 20, 'Mike Davis': 19,
    # 2016 class
    'Ezekiel Elliott': 19, 'Derrick Henry': 21, 'Kenyan Drake': 19,
    'Jordan Howard': 20, 'Devontae Booker': 21, 'Paul Perkins': 18,
    # 2017 class
    'Christian McCaffrey': 19, 'Leonard Fournette': 18, 'Dalvin Cook': 18,
    'Joe Mixon': 19, 'Alvin Kamara': 21, 'Kareem Hunt': 19,
    'Aaron Jones': 19, 'James Conner': 20, 'Samaje Perine': 18, 'Marlon Mack': 19,
    # 2018 class
    'Saquon Barkley': 18, 'Nick Chubb': 18, 'Sony Michel': 18,
    'Ronald Jones': 18, 'Rashaad Penny': 21, 'Kerryon Johnson': 20,
    'Royce Freeman': 18, 'Nyheim Hines': 20,
    # 2019 class
    'Josh Jacobs': 20, 'Miles Sanders': 21, 'David Montgomery': 19,
    'Darrell Henderson': 20, 'Devin Singletary': 19, 'Damien Harris': 18,
    'Justice Hill': 18, 'Tony Pollard': 20,
    # 2020 class
    'Clyde Edwards-Helaire': 21, "D'Andre Swift": 19, 'Jonathan Taylor': 18,
    'Cam Akers': 18, 'J.K. Dobbins': 18, 'Zack Moss': 19,
    'Antonio Gibson': 22, 'AJ Dillon': 18, "Ke'Shawn Vaughn": 20, 'Darrynton Evans': 20,
    # 2021 class
    'Najee Harris': 21, 'Travis Etienne': 18, 'Javonte Williams': 21,
    'Michael Carter': 21, 'Trey Sermon': 21, 'Rhamondre Stevenson': 21,
    'Kenneth Gainwell': 19, 'Chuba Hubbard': 19, 'Elijah Mitchell': 20, 'Khalil Herbert': 20,
    # 2022 class
    'Breece Hall': 19, 'Kenneth Walker III': 21, 'James Cook': 21,
    'Rachaad White': 22, 'Brian Robinson Jr.': 22, 'Dameon Pierce': 21,
    'Isiah Pacheco': 19, 'Kyren Williams': 19, 'Tyler Allgeier': 21,
    'Jerome Ford': 21, 'Zamir White': 20,
    # 2023 class
    'Bijan Robinson': 19, 'Jahmyr Gibbs': 19, 'Zach Charbonnet': 18,
    "De'Von Achane": 20, 'Tank Bigsby': 18, 'Tyjae Spears': 21,
    'Chase Brown': 22, 'Kendre Miller': 21, 'Roschon Johnson': 18,
    'Israel Abanikanda': 21, 'Eric Gray': 18, 'Sean Tucker': 18,
    'Deuce Vaughn': 18, 'Chris Rodriguez': 19, 'DeWayne McBride': 20,
    # 2024 class (estimated based on college careers)
    'Jonathan Brooks': 20, 'Trey Benson': 21, 'Blake Corum': 21,
    'MarShawn Lloyd': 21, 'Jaylen Wright': 20, 'Jonathon Brooks': 20,
    'Ray Davis': 23, 'Braelon Allen': 19, 'Audric Estime': 20,
    'Tyrone Tracy Jr.': 22, 'Isaac Guerendo': 22, 'Kimani Vidal': 21,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_draft_capital(pick, max_pick=262):
    """Convert pick number to 0-100 score using gentler power curve.

    Formula: DC = 100 - 2.40 × (pick^0.62 - 1)

    This creates a gentler decay than 1/sqrt(pick), giving:
    - Pick 1: 100, Pick 5: ~95, Pick 10: ~88, Pick 32: ~73
    - Pick 100: ~50, Pick 200: ~35
    """
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))  # Clamp to 0-100

def breakout_age_to_score(age):
    """Convert breakout age to 0-100 score"""
    if pd.isna(age):
        return None
    age = int(age)
    return BREAKOUT_AGE_SCORES.get(age, 25)  # Default to 25 for very late/never

def normalize_ras(ras, mean_ras=5.5, std_ras=2.5):
    """Convert RAS (0-10 scale) to 0-100 score"""
    if pd.isna(ras):
        return None
    # Center at 50, with typical range 20-80
    return 50 + (ras - mean_ras) / std_ras * 25

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "=" * 90)
print("LOADING DATA")
print("=" * 90)

# Load WR backtest
wr = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"WRs loaded: {len(wr)}")

# Load RB backtest
rb = pd.read_csv('data/rb_backtest_2015_2024.csv')
print(f"RBs loaded: {len(rb)}")

# Add RB breakout ages
rb['breakout_age'] = rb['player_name'].map(RB_BREAKOUT_AGES)

# ============================================================================
# CALCULATE COMPONENT SCORES
# ============================================================================
print("\n" + "=" * 90)
print("CALCULATING COMPONENT SCORES")
print("=" * 90)

# --- WR CALCULATIONS ---
wr['dc_score'] = wr['pick'].apply(normalize_draft_capital)
wr['breakout_score'] = wr['breakout_age'].apply(breakout_age_to_score)
wr['ras_score'] = wr['RAS'].apply(normalize_ras)

# Calculate position averages for imputation
wr_avg_breakout = wr['breakout_score'].mean()
wr_avg_ras = wr['ras_score'].mean()
print(f"WR avg breakout score: {wr_avg_breakout:.1f}")
print(f"WR avg RAS score: {wr_avg_ras:.1f}")

# Impute missing values
wr['breakout_score_final'] = wr['breakout_score'].fillna(wr_avg_breakout)
wr['ras_score_final'] = wr['ras_score'].fillna(wr_avg_ras)

# Track imputation status
wr['breakout_status'] = np.where(wr['breakout_score'].isna(), 'imputed', 'observed')
wr['ras_status'] = np.where(wr['ras_score'].isna(), 'imputed', 'observed')

# --- RB CALCULATIONS ---
rb['dc_score'] = rb['pick'].apply(normalize_draft_capital)
rb['breakout_score'] = rb['breakout_age'].apply(breakout_age_to_score)
rb['ras_score'] = rb['RAS'].apply(normalize_ras)

# Calculate position averages for imputation
rb_avg_breakout = rb['breakout_score'].mean()
rb_avg_ras = rb['ras_score'].mean()
print(f"RB avg breakout score: {rb_avg_breakout:.1f}")
print(f"RB avg RAS score: {rb_avg_ras:.1f}")

# Impute missing values
rb['breakout_score_final'] = rb['breakout_score'].fillna(rb_avg_breakout)
rb['ras_score_final'] = rb['ras_score'].fillna(rb_avg_ras)

# Track imputation status
rb['breakout_status'] = np.where(rb['breakout_score'].isna(), 'imputed', 'observed')
rb['ras_status'] = np.where(rb['ras_score'].isna(), 'imputed', 'observed')

# ============================================================================
# CALCULATE SLAP SCORES
# ============================================================================
print("\n" + "=" * 90)
print("CALCULATING SLAP SCORES")
print("=" * 90)

# SLAP = 85% DC + 10% Breakout + 5% RAS
wr['slap_score'] = (
    WEIGHT_DC * wr['dc_score'] +
    WEIGHT_BREAKOUT * wr['breakout_score_final'] +
    WEIGHT_RAS * wr['ras_score_final']
)

rb['slap_score'] = (
    WEIGHT_DC * rb['dc_score'] +
    WEIGHT_BREAKOUT * rb['breakout_score_final'] +
    WEIGHT_RAS * rb['ras_score_final']
)

# Calculate delta vs DC-only baseline
wr['dc_only'] = wr['dc_score']
wr['delta_vs_dc'] = wr['slap_score'] - wr['dc_only']

rb['dc_only'] = rb['dc_score']
rb['delta_vs_dc'] = rb['slap_score'] - rb['dc_only']

# ============================================================================
# WR RANKINGS
# ============================================================================
print("\n" + "=" * 90)
print("WR RANKINGS (2015-2024 Backtest)")
print("=" * 90)

wr_ranked = wr.sort_values('slap_score', ascending=False).reset_index(drop=True)
wr_ranked['rank'] = range(1, len(wr_ranked) + 1)

print(f"\n{'Rank':<5} {'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>5} {'BO':>5} {'RAS':>5} {'Delta':>6} {'Hit24':<5}")
print("-" * 95)

for i, row in wr_ranked.head(30).iterrows():
    bo_flag = "*" if row['breakout_status'] == 'imputed' else ""
    ras_flag = "*" if row['ras_status'] == 'imputed' else ""
    hit = "✓" if row['hit24'] == 1 else ""
    print(f"{row['rank']:<5} {row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} "
          f"{row['slap_score']:>6.1f} {row['dc_score']:>5.1f} {row['breakout_score_final']:>4.0f}{bo_flag:<1} "
          f"{row['ras_score_final']:>4.0f}{ras_flag:<1} {row['delta_vs_dc']:>+6.1f} {hit:<5}")

print("\n* = imputed (position average)")

# Show biggest deltas (model disagrees with draft)
print("\n" + "-" * 50)
print("BIGGEST POSITIVE DELTAS (SLAP > DC - Model likes more)")
print("-" * 50)
for _, row in wr_ranked.nlargest(10, 'delta_vs_dc').iterrows():
    hit = "HIT" if row['hit24'] == 1 else "miss"
    print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} | SLAP {row['slap_score']:.1f} vs DC {row['dc_score']:.1f} | {hit}")

print("\n" + "-" * 50)
print("BIGGEST NEGATIVE DELTAS (SLAP < DC - Model likes less)")
print("-" * 50)
for _, row in wr_ranked.nsmallest(10, 'delta_vs_dc').iterrows():
    hit = "HIT" if row['hit24'] == 1 else "miss"
    print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} | SLAP {row['slap_score']:.1f} vs DC {row['dc_score']:.1f} | {hit}")

# ============================================================================
# RB RANKINGS
# ============================================================================
print("\n" + "=" * 90)
print("RB RANKINGS (2015-2024 Backtest)")
print("=" * 90)

rb_ranked = rb.sort_values('slap_score', ascending=False).reset_index(drop=True)
rb_ranked['rank'] = range(1, len(rb_ranked) + 1)

print(f"\n{'Rank':<5} {'Player':<25} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>5} {'BO':>5} {'RAS':>5} {'Delta':>6} {'Hit24':<5}")
print("-" * 95)

for i, row in rb_ranked.head(30).iterrows():
    bo_flag = "*" if row['breakout_status'] == 'imputed' else ""
    ras_flag = "*" if row['ras_status'] == 'imputed' else ""
    hit = "✓" if row['hit24'] == 1 else ""
    print(f"{row['rank']:<5} {row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} "
          f"{row['slap_score']:>6.1f} {row['dc_score']:>5.1f} {row['breakout_score_final']:>4.0f}{bo_flag:<1} "
          f"{row['ras_score_final']:>4.0f}{ras_flag:<1} {row['delta_vs_dc']:>+6.1f} {hit:<5}")

print("\n* = imputed (position average)")

# Show biggest deltas (model disagrees with draft)
print("\n" + "-" * 50)
print("BIGGEST POSITIVE DELTAS (SLAP > DC - Model likes more)")
print("-" * 50)
for _, row in rb_ranked.nlargest(10, 'delta_vs_dc').iterrows():
    hit = "HIT" if row['hit24'] == 1 else "miss"
    print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} | SLAP {row['slap_score']:.1f} vs DC {row['dc_score']:.1f} | {hit}")

print("\n" + "-" * 50)
print("BIGGEST NEGATIVE DELTAS (SLAP < DC - Model likes less)")
print("-" * 50)
for _, row in rb_ranked.nsmallest(10, 'delta_vs_dc').iterrows():
    hit = "HIT" if row['hit24'] == 1 else "miss"
    print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} | SLAP {row['slap_score']:.1f} vs DC {row['dc_score']:.1f} | {hit}")

# ============================================================================
# MODEL VALIDATION
# ============================================================================
print("\n" + "=" * 90)
print("MODEL VALIDATION")
print("=" * 90)

from scipy.stats import spearmanr

# WR validation
wr_valid = wr_ranked[wr_ranked['best_ppr'] > 0].copy()
wr_corr, wr_p = spearmanr(wr_valid['slap_score'], wr_valid['best_ppr'])
wr_hit_rate = wr_valid[wr_valid['slap_score'] >= wr_valid['slap_score'].median()]['hit24'].mean()

print(f"\nWR Model Performance:")
print(f"  Spearman correlation (SLAP vs best_ppr): {wr_corr:.3f} (p={wr_p:.4f})")
print(f"  Top-half SLAP hit rate: {wr_hit_rate*100:.1f}%")
print(f"  Overall hit rate: {wr_valid['hit24'].mean()*100:.1f}%")

# RB validation
rb_valid = rb_ranked[rb_ranked['best_ppr'] > 0].copy()
rb_corr, rb_p = spearmanr(rb_valid['slap_score'], rb_valid['best_ppr'])
rb_hit_rate = rb_valid[rb_valid['slap_score'] >= rb_valid['slap_score'].median()]['hit24'].mean()

print(f"\nRB Model Performance:")
print(f"  Spearman correlation (SLAP vs best_ppr): {rb_corr:.3f} (p={rb_p:.4f})")
print(f"  Top-half SLAP hit rate: {rb_hit_rate*100:.1f}%")
print(f"  Overall hit rate: {rb_valid['hit24'].mean()*100:.1f}%")

# ============================================================================
# SAVE OUTPUT
# ============================================================================
print("\n" + "=" * 90)
print("SAVING OUTPUT")
print("=" * 90)

# Select columns for output
wr_output = wr_ranked[[
    'rank', 'player_name', 'draft_year', 'pick', 'college',
    'slap_score', 'dc_score', 'breakout_score_final', 'ras_score_final',
    'delta_vs_dc', 'breakout_status', 'ras_status',
    'breakout_age', 'RAS', 'best_ppr', 'hit24', 'hit12'
]].copy()
wr_output.columns = [
    'rank', 'player_name', 'draft_year', 'pick', 'college',
    'slap_score', 'dc_score', 'breakout_score', 'ras_score',
    'delta_vs_dc', 'breakout_status', 'ras_status',
    'breakout_age_raw', 'ras_raw', 'best_ppr', 'hit24', 'hit12'
]
wr_output.to_csv('output/slap_scores_wr_v3.csv', index=False)
print(f"Saved: output/slap_scores_wr_v3.csv ({len(wr_output)} WRs)")

rb_output = rb_ranked[[
    'rank', 'player_name', 'draft_year', 'pick', 'college',
    'slap_score', 'dc_score', 'breakout_score_final', 'ras_score_final',
    'delta_vs_dc', 'breakout_status', 'ras_status',
    'breakout_age', 'RAS', 'best_ppr', 'hit24', 'hit12'
]].copy()
rb_output.columns = [
    'rank', 'player_name', 'draft_year', 'pick', 'college',
    'slap_score', 'dc_score', 'breakout_score', 'ras_score',
    'delta_vs_dc', 'breakout_status', 'ras_status',
    'breakout_age_raw', 'ras_raw', 'best_ppr', 'hit24', 'hit12'
]
rb_output.to_csv('output/slap_scores_rb_v3.csv', index=False)
print(f"Saved: output/slap_scores_rb_v3.csv ({len(rb_output)} RBs)")

# Combined output
combined = pd.concat([
    wr_output.assign(position='WR'),
    rb_output.assign(position='RB')
], ignore_index=True)
combined = combined.sort_values('slap_score', ascending=False).reset_index(drop=True)
combined['overall_rank'] = range(1, len(combined) + 1)
combined.to_csv('output/slap_scores_combined_v3.csv', index=False)
print(f"Saved: output/slap_scores_combined_v3.csv ({len(combined)} total)")

print("\n" + "=" * 90)
print("DONE!")
print("=" * 90)
