"""
SLAP Score V3 - FIXED with Correct RB Metric

Key change: RBs now use production metric (rec_yards / team_pass_att × age_weight)
instead of breakout age. Analysis showed:
- Production metric: r=0.30, adds significant value beyond DC (p=0.004)
- Breakout age: r=0.10, does NOT add value (p=0.80)

Generates SLAP scores for ALL players:
- WRs 2015-2024 (backtest with outcomes) - uses Breakout Age
- RBs 2015-2024 (backtest with outcomes) - uses Production Metric
- WRs 2026 (prospects)
- RBs 2026 (prospects)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("SLAP SCORE V3 - FIXED RB METRIC")
print("WRs: Breakout Age | RBs: Production (rec_yards / team_pass_att × age_wt)")
print("=" * 90)

# ============================================================================
# CONFIGURATION
# ============================================================================
WEIGHT_DC = 0.85
WEIGHT_PRODUCTION = 0.10  # Renamed from WEIGHT_BREAKOUT for clarity
WEIGHT_RAS = 0.05

# ============================================================================
# DC FORMULA: GRADUAL CURVE
# ============================================================================
def dc_gradual(pick):
    """
    Gradual DC formula: DC = 100 - 2.40 × (pick^0.62 - 1)
    """
    k, p = 2.40, 0.62
    return max(0, min(100, 100 - k * (pick**p - 1)))

# ============================================================================
# WR BREAKOUT AGE SCORING (unchanged)
# ============================================================================
BREAKOUT_AGE_SCORES = {
    18: 100,  # Freshman breakout = elite
    19: 90,
    20: 75,
    21: 60,
    22: 45,
    23: 30,
    24: 20,
}

def breakout_age_to_score(age):
    """Convert breakout age to 0-100 score (for WRs only)"""
    if pd.isna(age):
        return None
    age = int(age)
    return BREAKOUT_AGE_SCORES.get(age, 25)

# ============================================================================
# RAS SCORING
# ============================================================================
def normalize_ras(ras, mean_ras=5.5, std_ras=2.5):
    """Convert RAS (0-10 scale) to 0-100 score"""
    if pd.isna(ras):
        return None
    return 50 + (ras - mean_ras) / std_ras * 25

# ============================================================================
# RB PRODUCTION METRIC
# ============================================================================
def age_weight(draft_age):
    """Age weight for production metric based on draft age"""
    college_age = draft_age - 1  # Estimate final college season age
    if college_age <= 19:
        return 1.20
    elif college_age == 20:
        return 1.10
    elif college_age == 21:
        return 1.00
    elif college_age == 22:
        return 0.90
    else:
        return 0.80

# ============================================================================
# SLAP TIER
# ============================================================================
def get_slap_tier(score):
    """Assign tier based on SLAP score"""
    if score >= 90:
        return "Elite"
    elif score >= 80:
        return "Great"
    elif score >= 70:
        return "Good"
    elif score >= 60:
        return "Average"
    elif score >= 50:
        return "Below Avg"
    else:
        return "Poor"

# ============================================================================
# 2026 WR BREAKOUT AGES
# ============================================================================
WR_2026_BREAKOUT = {
    'Carnell Tate': 19, 'Jordyn Tyson': 20, 'Makai Lemon': 20,
    'Denzel Boston': 20, 'Kevin Concepcion': 19, 'Chris Bell': 21,
    'Elijah Sarratt': 21, 'Zachariah Branch': 19, 'Germie Bernard': 20,
    'Chris Brazzell': 21, "Ja'Kobi Lane": 20, 'Omar Cooper Jr.': 21,
    'Antonio Williams': 21, 'Skyler Bell': 21, 'Malachi Fields': 21,
    'C.J. Daniels': 20, 'Brenen Thompson': 21, 'Deion Burks': 20,
    'Ted Hurst': 21, 'Bryce Lance': 21, 'Kevin Coleman Jr.': 20,
    'Eric McAlister': 21, 'Eric Rivers': 21, 'Lewis Bond': 21,
    "De'Zhaun Stribling": 20, 'Keelan Marion': 20, 'Josh Cameron': 21,
    'Noah Thomas': 19, 'Aaron Anderson': 20, 'Dane Key': 20,
    'Jordan Hudson': 20, 'Caleb Douglas': 20, 'Reggie Virgil': 20,
    'Vinny Anthony II': 21, 'Caullin Lacy': 21, 'Kendrick Law': 21,
    'Colbie Young': 21, 'Harrison Wallace III': 20, 'Jaden Greathouse': 19,
    'Barion Brown': 19, 'Amare Thomas': 18, 'Hykeem Williams': 20,
    'Shelton Sampson Jr.': 19,
}

# 2026 RB production scores (estimated based on 2024 college stats)
# These are already normalized 0-100 scores
RB_2026_PRODUCTION = {
    'Jeremiyah Love': 65,  # Good receiving back at Notre Dame
    'Jonah Coleman': 55,   # Some receiving work
    'Jadarian Price': 50,  # Limited receiving role
    'Emmett Johnson': 45,  # Rush-first back
    'Nick Singleton': 40,  # Rush-first at Penn State
    'Kaytron Allen': 35,   # Rush-first at Penn State
    'Demond Claiborne': 50,
    'Mike Washington Jr.': 45,
    'Adam Randall': 40,
    'Noah Whittington': 45,
    'Roman Hemby': 40,
    'C.J. Donaldson': 35,
    'Jaydn Ott': 55,       # Pass-catching back at Cal
    'Quinten Joyner': 50,
}

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "=" * 90)
print("LOADING DATA")
print("=" * 90)

# Load WR backtest
wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"WR backtest loaded: {len(wr_backtest)} players")

# Load RB backtest WITH PRODUCTION METRIC
rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
print(f"RB backtest loaded: {len(rb_backtest)} players")

# Load 2026 prospects
prospects_2026 = pd.read_csv('data/prospects_final.csv')
print(f"2026 prospects loaded: {len(prospects_2026)} players")

# ============================================================================
# PROCESS WR BACKTEST (unchanged - uses Breakout Age)
# ============================================================================
print("\n" + "=" * 90)
print("PROCESSING WR BACKTEST (2015-2024) - Using Breakout Age")
print("=" * 90)

wr_backtest['position'] = 'WR'
wr_backtest['dc_score'] = wr_backtest['pick'].apply(dc_gradual)
wr_backtest['breakout_score'] = wr_backtest['breakout_age'].apply(breakout_age_to_score)
wr_backtest['ras_score'] = wr_backtest['RAS'].apply(normalize_ras)

# Calculate averages for imputation
wr_avg_breakout = wr_backtest['breakout_score'].mean()
wr_avg_ras = wr_backtest['ras_score'].mean()
print(f"WR avg breakout score: {wr_avg_breakout:.1f}")
print(f"WR avg RAS score: {wr_avg_ras:.1f}")

# Impute missing values
wr_backtest['production_score_final'] = wr_backtest['breakout_score'].fillna(wr_avg_breakout)
wr_backtest['ras_score_final'] = wr_backtest['ras_score'].fillna(wr_avg_ras)

# Calculate SLAP
wr_backtest['slap_score'] = (
    WEIGHT_DC * wr_backtest['dc_score'] +
    WEIGHT_PRODUCTION * wr_backtest['production_score_final'] +
    WEIGHT_RAS * wr_backtest['ras_score_final']
)

wr_backtest['slap_tier'] = wr_backtest['slap_score'].apply(get_slap_tier)
wr_backtest['best_season_ppg'] = wr_backtest['best_ppr'] / 17

print(f"WRs processed: {len(wr_backtest)}")

# ============================================================================
# PROCESS RB BACKTEST (NEW - uses Production Metric)
# ============================================================================
print("\n" + "=" * 90)
print("PROCESSING RB BACKTEST (2015-2024) - Using Production Metric")
print("=" * 90)

rb_backtest['position'] = 'RB'
rb_backtest['dc_score'] = rb_backtest['pick'].apply(dc_gradual)

# Convert columns to numeric
rb_backtest['rec_yards'] = pd.to_numeric(rb_backtest['rec_yards'], errors='coerce')
rb_backtest['team_pass_att'] = pd.to_numeric(rb_backtest['team_pass_att'], errors='coerce')
rb_backtest['age'] = pd.to_numeric(rb_backtest['age'], errors='coerce')

# Calculate production metric for those with valid data
has_production = (rb_backtest['rec_yards'].notna()) & (rb_backtest['team_pass_att'].notna()) & (rb_backtest['team_pass_att'] > 0)

rb_backtest.loc[has_production, 'rec_per_pass_att'] = (
    rb_backtest.loc[has_production, 'rec_yards'] / rb_backtest.loc[has_production, 'team_pass_att']
)
rb_backtest.loc[has_production, 'age_wt'] = rb_backtest.loc[has_production, 'age'].apply(age_weight)
rb_backtest.loc[has_production, 'production_raw'] = (
    rb_backtest.loc[has_production, 'rec_per_pass_att'] * rb_backtest.loc[has_production, 'age_wt']
)

# Normalize to 0-100 scale
if has_production.sum() > 0:
    min_prod = rb_backtest.loc[has_production, 'production_raw'].min()
    max_prod = rb_backtest.loc[has_production, 'production_raw'].max()
    rb_backtest.loc[has_production, 'production_score'] = (
        (rb_backtest.loc[has_production, 'production_raw'] - min_prod) / (max_prod - min_prod) * 100
    )

# RAS scoring
rb_backtest['ras_score'] = rb_backtest['RAS'].apply(normalize_ras)

# Calculate averages for imputation
rb_avg_production = rb_backtest['production_score'].mean()
rb_avg_ras = rb_backtest['ras_score'].mean()
print(f"RB avg production score: {rb_avg_production:.1f}")
print(f"RB avg RAS score: {rb_avg_ras:.1f}")
print(f"RBs with production data: {has_production.sum()}/{len(rb_backtest)}")

# Impute missing values
rb_backtest['production_score_final'] = rb_backtest['production_score'].fillna(rb_avg_production)
rb_backtest['ras_score_final'] = rb_backtest['ras_score'].fillna(rb_avg_ras)

# Calculate SLAP
rb_backtest['slap_score'] = (
    WEIGHT_DC * rb_backtest['dc_score'] +
    WEIGHT_PRODUCTION * rb_backtest['production_score_final'] +
    WEIGHT_RAS * rb_backtest['ras_score_final']
)

rb_backtest['slap_tier'] = rb_backtest['slap_score'].apply(get_slap_tier)
rb_backtest['best_season_ppg'] = rb_backtest['best_ppg']  # Already exists

print(f"RBs processed: {len(rb_backtest)}")

# ============================================================================
# PROCESS 2026 PROSPECTS
# ============================================================================
print("\n" + "=" * 90)
print("PROCESSING 2026 PROSPECTS")
print("=" * 90)

# Split by position
wr_2026 = prospects_2026[prospects_2026['position'] == 'WR'].copy()
rb_2026 = prospects_2026[prospects_2026['position'] == 'RB'].copy()

# WR 2026 (uses breakout age)
wr_2026['draft_year'] = 2026
wr_2026['pick'] = wr_2026['projected_pick']
wr_2026['dc_score'] = wr_2026['pick'].apply(dc_gradual)
wr_2026['breakout_age'] = wr_2026['player_name'].map(WR_2026_BREAKOUT)
wr_2026['breakout_score'] = wr_2026['breakout_age'].apply(breakout_age_to_score)
wr_2026['production_score_final'] = wr_2026['breakout_score'].fillna(wr_avg_breakout)
wr_2026['RAS'] = np.nan
wr_2026['ras_score'] = np.nan
wr_2026['ras_score_final'] = wr_avg_ras
wr_2026['slap_score'] = (
    WEIGHT_DC * wr_2026['dc_score'] +
    WEIGHT_PRODUCTION * wr_2026['production_score_final'] +
    WEIGHT_RAS * wr_2026['ras_score_final']
)
wr_2026['slap_tier'] = wr_2026['slap_score'].apply(get_slap_tier)
wr_2026['best_ppr'] = np.nan
wr_2026['best_season_ppg'] = np.nan
wr_2026['hit24'] = np.nan
wr_2026['hit12'] = np.nan
wr_2026['college'] = wr_2026['school']
wr_2026['position'] = 'WR'

print(f"2026 WRs processed: {len(wr_2026)}")

# RB 2026 (uses production metric)
rb_2026['draft_year'] = 2026
rb_2026['pick'] = rb_2026['projected_pick']
rb_2026['dc_score'] = rb_2026['pick'].apply(dc_gradual)
rb_2026['production_score'] = rb_2026['player_name'].map(RB_2026_PRODUCTION)
rb_2026['production_score_final'] = rb_2026['production_score'].fillna(rb_avg_production)
rb_2026['RAS'] = np.nan
rb_2026['ras_score'] = np.nan
rb_2026['ras_score_final'] = rb_avg_ras
rb_2026['slap_score'] = (
    WEIGHT_DC * rb_2026['dc_score'] +
    WEIGHT_PRODUCTION * rb_2026['production_score_final'] +
    WEIGHT_RAS * rb_2026['ras_score_final']
)
rb_2026['slap_tier'] = rb_2026['slap_score'].apply(get_slap_tier)
rb_2026['best_ppr'] = np.nan
rb_2026['best_season_ppg'] = np.nan
rb_2026['hit24'] = np.nan
rb_2026['hit12'] = np.nan
rb_2026['college'] = rb_2026['school']
rb_2026['position'] = 'RB'

print(f"2026 RBs processed: {len(rb_2026)}")

# ============================================================================
# COMBINE ALL DATA
# ============================================================================
print("\n" + "=" * 90)
print("COMBINING ALL DATA")
print("=" * 90)

# Standardize columns
output_cols = [
    'player_name', 'position', 'draft_year', 'pick', 'college',
    'dc_score', 'production_score_final', 'RAS', 'ras_score_final',
    'slap_score', 'slap_tier', 'best_season_ppg', 'hit24', 'hit12'
]

# Add missing columns
for df in [wr_backtest, rb_backtest, wr_2026, rb_2026]:
    for col in output_cols:
        if col not in df.columns:
            df[col] = np.nan

# Prepare each dataset
wr_backtest_out = wr_backtest[output_cols].copy()
rb_backtest_out = rb_backtest[output_cols].copy()
wr_2026_out = wr_2026[output_cols].copy()
rb_2026_out = rb_2026[output_cols].copy()

# Combine
all_players = pd.concat([wr_backtest_out, rb_backtest_out, wr_2026_out, rb_2026_out], ignore_index=True)

# Rename columns for clarity
all_players.columns = [
    'player_name', 'position', 'draft_year', 'pick', 'college',
    'dc_score', 'production_score', 'ras_raw', 'ras_score',
    'slap_score', 'slap_tier', 'best_season_ppg', 'hit24', 'hit12'
]

# Sort by SLAP score
all_players = all_players.sort_values('slap_score', ascending=False).reset_index(drop=True)
all_players['rank'] = range(1, len(all_players) + 1)

print(f"Total players: {len(all_players)}")
print(f"  WRs: {len(all_players[all_players['position']=='WR'])}")
print(f"  RBs: {len(all_players[all_players['position']=='RB'])}")

# ============================================================================
# SAVE OUTPUT
# ============================================================================
print("\n" + "=" * 90)
print("SAVING OUTPUT")
print("=" * 90)

all_players.to_csv('output/slap_v3_fixed_all_players.csv', index=False)
print(f"Saved: output/slap_v3_fixed_all_players.csv ({len(all_players)} players)")

# ============================================================================
# SUMMARY TABLES
# ============================================================================
print("\n" + "=" * 90)
print("TOP 20 RBs BY SLAP (using new production metric)")
print("=" * 90)

top_rb = all_players[all_players['position'] == 'RB'].head(20)
print(f"\n{'Rank':<5} {'Player':<25} {'Year':>5} {'Pick':>5} {'DC':>6} {'Prod':>5} {'RAS':>5} {'SLAP':>6} {'Tier':<10} {'PPG':>6} {'Hit':>4}")
print("-" * 100)

for _, row in top_rb.iterrows():
    ppg = f"{row['best_season_ppg']:.1f}" if pd.notna(row['best_season_ppg']) else "-"
    hit = "✓" if row['hit24'] == 1 else ("✗" if row['hit24'] == 0 else "-")
    print(f"{row['rank']:<5} {row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} "
          f"{row['dc_score']:>6.1f} {row['production_score']:>5.0f} {row['ras_score']:>5.1f} "
          f"{row['slap_score']:>6.1f} {row['slap_tier']:<10} {ppg:>6} {hit:>4}")

print("\n" + "=" * 90)
print("TOP 20 WRs BY SLAP (using breakout age)")
print("=" * 90)

top_wr = all_players[all_players['position'] == 'WR'].head(20)
print(f"\n{'Rank':<5} {'Player':<25} {'Year':>5} {'Pick':>5} {'DC':>6} {'BO':>5} {'RAS':>5} {'SLAP':>6} {'Tier':<10} {'PPG':>6} {'Hit':>4}")
print("-" * 100)

for _, row in top_wr.iterrows():
    ppg = f"{row['best_season_ppg']:.1f}" if pd.notna(row['best_season_ppg']) else "-"
    hit = "✓" if row['hit24'] == 1 else ("✗" if row['hit24'] == 0 else "-")
    print(f"{row['rank']:<5} {row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} "
          f"{row['dc_score']:>6.1f} {row['production_score']:>5.0f} {row['ras_score']:>5.1f} "
          f"{row['slap_score']:>6.1f} {row['slap_tier']:<10} {ppg:>6} {hit:>4}")

# 2026 prospects
print("\n" + "=" * 90)
print("TOP 15 RBs - 2026 CLASS")
print("=" * 90)

rb_2026_ranked = all_players[(all_players['position'] == 'RB') & (all_players['draft_year'] == 2026)].head(15)
print(f"\n{'Rank':<5} {'Player':<25} {'Pick':>5} {'DC':>6} {'Prod':>5} {'RAS':>5} {'SLAP':>6} {'Tier':<10}")
print("-" * 75)

for i, (_, row) in enumerate(rb_2026_ranked.iterrows(), 1):
    print(f"{i:<5} {row['player_name']:<25} {int(row['pick']):>5} "
          f"{row['dc_score']:>6.1f} {row['production_score']:>5.0f} {row['ras_score']:>5.1f} "
          f"{row['slap_score']:>6.1f} {row['slap_tier']:<10}")

print("\n" + "=" * 90)
print("VERIFICATION: KEY RB SCORES")
print("=" * 90)

key_rbs = [
    ('Saquon Barkley', 2),
    ('Christian McCaffrey', 8),
    ('Dalvin Cook', 41),
    ('Jonathan Taylor', 41),
    ('Breece Hall', 36),
]

print(f"\n{'Player':<25} {'Pick':>5} {'SLAP':>6} {'Production':>10} {'NFL PPG':>8} {'Hit':>4}")
print("-" * 60)

for name, pick in key_rbs:
    row = all_players[all_players['player_name'] == name]
    if len(row) > 0:
        r = row.iloc[0]
        ppg = f"{r['best_season_ppg']:.1f}" if pd.notna(r['best_season_ppg']) else "-"
        hit = "✓" if r['hit24'] == 1 else "✗"
        print(f"{name:<25} {pick:>5} {r['slap_score']:>6.1f} {r['production_score']:>10.1f} {ppg:>8} {hit:>4}")

print("\n" + "=" * 90)
print("COMPLETE!")
print("=" * 90)
