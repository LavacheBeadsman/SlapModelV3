"""
SLAP Score V3 - Complete Generation
Using CORRECTED Tiered DC Formula

Generates SLAP scores for ALL players:
- WRs 2015-2024 (backtest with outcomes)
- RBs 2015-2024 (backtest with outcomes)
- WRs 2026 (prospects)
- RBs 2026 (prospects)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("SLAP SCORE V3 - COMPLETE GENERATION")
print("Using CORRECTED Tiered DC Formula")
print("=" * 90)

# ============================================================================
# CONFIGURATION
# ============================================================================
WEIGHT_DC = 0.85
WEIGHT_BREAKOUT = 0.10
WEIGHT_RAS = 0.05

# ============================================================================
# CORRECTED DC FORMULA: GRADUAL CURVE
# ============================================================================
def dc_gradual(pick):
    """
    Gradual DC formula using power function.

    DC = 100 - 2.40 × (pick^0.62 - 1)

    This produces a smooth, gradual curve that:
    - Keeps Round 1 picks close together (spread of ~18 points)
    - Accelerates decay in later rounds where outcomes diverge more

    Key values:
    - Pick 1: 100.0
    - Pick 10: 92.4
    - Pick 32: 81.8 (end of Round 1)
    - Pick 64: 70.8 (end of Round 2)
    - Pick 100: 60.7 (end of Round 3)
    - Pick 150: 48.8
    - Pick 200: 38.3
    - Pick 262: 31.4
    """
    k, p = 2.40, 0.62
    return max(0, min(100, 100 - k * (pick**p - 1)))

print("\n" + "=" * 90)
print("DC FORMULA: GRADUAL CURVE")
print("DC = 100 - 2.40 × (pick^0.62 - 1)")
print("=" * 90)
print("""
Key properties:
  - Pick 1:   100.0 (guaranteed)
  - Pick 10:   92.4
  - Pick 32:   81.8 (end of Round 1)
  - Pick 64:   70.8 (end of Round 2)
  - Pick 100:  60.7 (end of Round 3)
  - Pick 150:  48.8 (mid Day 3)
  - Pick 200:  38.3 (late Day 3)
  - Pick 262:  31.4 (Mr. Irrelevant)

Day spreads:
  Day 1 (Rd 1, picks 1-32):    18.2 points (100.0 → 81.8)
  Day 2 (Rd 2-3, picks 33-100): 21.1 points (81.8 → 60.7)
  Day 3 (Rd 4-7, picks 101-262): 29.3 points (60.7 → 31.4)
""")

# Show sample DC scores
print("Sample DC Scores:")
for pick in [1, 2, 5, 10, 20, 32, 50, 64, 100, 150, 200, 262]:
    print(f"  Pick {pick:>3}: DC = {dc_gradual(pick):.1f}")

# ============================================================================
# BREAKOUT AGE SCORING
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
    """Convert breakout age to 0-100 score"""
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
    # Center at 50, with typical range 20-80
    return 50 + (ras - mean_ras) / std_ras * 25

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
# RB BREAKOUT AGES (from research)
# ============================================================================
RB_BREAKOUT_AGES = {
    # 2015 class
    'Todd Gurley': 19, 'Melvin Gordon': 20, 'David Johnson': 21,
    'Tevin Coleman': 20, 'Duke Johnson': 18, 'T.J. Yeldon': 18,
    'Ameer Abdullah': 20, 'Jay Ajayi': 20, 'Mike Davis': 19, 'Matt Jones': 21,
    # 2016 class
    'Ezekiel Elliott': 19, 'Derrick Henry': 21, 'Kenyan Drake': 19,
    'Jordan Howard': 20, 'Devontae Booker': 21, 'Paul Perkins': 18,
    'C.J. Prosise': 21,
    # 2017 class
    'Christian McCaffrey': 19, 'Leonard Fournette': 18, 'Dalvin Cook': 18,
    'Joe Mixon': 19, 'Alvin Kamara': 21, 'Kareem Hunt': 19,
    'Aaron Jones': 19, 'James Conner': 20, 'Samaje Perine': 18, 'Marlon Mack': 19,
    "D'Onta Foreman": 20,
    # 2018 class
    'Saquon Barkley': 18, 'Nick Chubb': 18, 'Sony Michel': 18,
    'Ronald Jones': 18, 'Ronald Jones II': 18, 'Rashaad Penny': 21, 'Kerryon Johnson': 20,
    'Royce Freeman': 18, 'Nyheim Hines': 20, 'Derrius Guice': 19,
    # 2019 class
    'Josh Jacobs': 20, 'Miles Sanders': 21, 'David Montgomery': 19,
    'Darrell Henderson': 20, 'Devin Singletary': 19, 'Damien Harris': 18,
    'Justice Hill': 18, 'Tony Pollard': 20, 'Alexander Mattison': 19,
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
    'Jerome Ford': 21, 'Zamir White': 20, 'Tyrion Davis-Price': 21,
    # 2023 class
    'Bijan Robinson': 19, 'Jahmyr Gibbs': 19, 'Zach Charbonnet': 18,
    "De'Von Achane": 20, 'Tank Bigsby': 18, 'Tyjae Spears': 21,
    'Chase Brown': 22, 'Kendre Miller': 21, 'Roschon Johnson': 18,
    'Israel Abanikanda': 21, 'Eric Gray': 18, 'Sean Tucker': 18,
    'Deuce Vaughn': 18, 'Chris Rodriguez': 19, 'DeWayne McBride': 20,
    # 2024 class
    'Jonathan Brooks': 20, 'Jonathon Brooks': 20, 'Trey Benson': 21, 'Blake Corum': 21,
    'MarShawn Lloyd': 21, 'Jaylen Wright': 20,
    'Ray Davis': 23, 'Braelon Allen': 19, 'Audric Estime': 20,
    'Tyrone Tracy Jr.': 22, 'Isaac Guerendo': 22, 'Kimani Vidal': 21,
}

# 2026 RB breakout ages
RB_2026_BREAKOUT = {
    'Jeremiyah Love': 19, 'Jonah Coleman': 20, 'Jadarian Price': 20,
    'Emmett Johnson': 20, 'Nick Singleton': 19, 'Kaytron Allen': 19,
    'Demond Claiborne': 20, 'Mike Washington Jr.': 21, 'Adam Randall': 20,
    'Noah Whittington': 21, 'Roman Hemby': 21, 'C.J. Donaldson': 20,
    "J'Mari Taylor": 21, 'Jamarion Miller': 20, "Le'Veon Moss": 20,
    'Dean Connors': 20, 'Desmond Reid': 20, 'Jaydn Ott': 19,
    'Jamal Haynes': 21, 'Quinten Joyner': 19,
}

# 2026 WR breakout ages
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

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "=" * 90)
print("LOADING DATA")
print("=" * 90)

# Load WR backtest
wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"WR backtest loaded: {len(wr_backtest)} players")

# Load RB backtest
rb_backtest = pd.read_csv('data/rb_backtest_2015_2024.csv')
print(f"RB backtest loaded: {len(rb_backtest)} players")

# Load 2026 prospects
prospects_2026 = pd.read_csv('data/prospects_final.csv')
print(f"2026 prospects loaded: {len(prospects_2026)} players")

# ============================================================================
# PROCESS WR BACKTEST
# ============================================================================
print("\n" + "=" * 90)
print("PROCESSING WR BACKTEST (2015-2024)")
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
wr_backtest['breakout_score_final'] = wr_backtest['breakout_score'].fillna(wr_avg_breakout)
wr_backtest['ras_score_final'] = wr_backtest['ras_score'].fillna(wr_avg_ras)

# Calculate SLAP
wr_backtest['slap_score'] = (
    WEIGHT_DC * wr_backtest['dc_score'] +
    WEIGHT_BREAKOUT * wr_backtest['breakout_score_final'] +
    WEIGHT_RAS * wr_backtest['ras_score_final']
)

wr_backtest['slap_tier'] = wr_backtest['slap_score'].apply(get_slap_tier)

# Calculate best PPG
wr_backtest['best_season_ppg'] = wr_backtest['best_ppr'] / 17

print(f"WRs processed: {len(wr_backtest)}")

# ============================================================================
# PROCESS RB BACKTEST
# ============================================================================
print("\n" + "=" * 90)
print("PROCESSING RB BACKTEST (2015-2024)")
print("=" * 90)

rb_backtest['position'] = 'RB'
rb_backtest['dc_score'] = rb_backtest['pick'].apply(dc_gradual)

# Add breakout ages from research
rb_backtest['breakout_age'] = rb_backtest['player_name'].map(RB_BREAKOUT_AGES)
rb_backtest['breakout_score'] = rb_backtest['breakout_age'].apply(breakout_age_to_score)
rb_backtest['ras_score'] = rb_backtest['RAS'].apply(normalize_ras)

# Calculate averages for imputation
rb_avg_breakout = rb_backtest['breakout_score'].mean()
rb_avg_ras = rb_backtest['ras_score'].mean()
print(f"RB avg breakout score: {rb_avg_breakout:.1f}")
print(f"RB avg RAS score: {rb_avg_ras:.1f}")

# Impute missing values
rb_backtest['breakout_score_final'] = rb_backtest['breakout_score'].fillna(rb_avg_breakout)
rb_backtest['ras_score_final'] = rb_backtest['ras_score'].fillna(rb_avg_ras)

# Calculate SLAP
rb_backtest['slap_score'] = (
    WEIGHT_DC * rb_backtest['dc_score'] +
    WEIGHT_BREAKOUT * rb_backtest['breakout_score_final'] +
    WEIGHT_RAS * rb_backtest['ras_score_final']
)

rb_backtest['slap_tier'] = rb_backtest['slap_score'].apply(get_slap_tier)

# Calculate best PPG
rb_backtest['best_season_ppg'] = rb_backtest['best_ppr'] / 17

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

# WR 2026
wr_2026['draft_year'] = 2026
wr_2026['pick'] = wr_2026['projected_pick']
wr_2026['dc_score'] = wr_2026['pick'].apply(dc_gradual)
wr_2026['breakout_age'] = wr_2026['player_name'].map(WR_2026_BREAKOUT)
wr_2026['breakout_score'] = wr_2026['breakout_age'].apply(breakout_age_to_score)
wr_2026['breakout_score_final'] = wr_2026['breakout_score'].fillna(wr_avg_breakout)
wr_2026['RAS'] = np.nan  # No combine data yet
wr_2026['ras_score'] = np.nan
wr_2026['ras_score_final'] = wr_avg_ras  # Impute with average
wr_2026['slap_score'] = (
    WEIGHT_DC * wr_2026['dc_score'] +
    WEIGHT_BREAKOUT * wr_2026['breakout_score_final'] +
    WEIGHT_RAS * wr_2026['ras_score_final']
)
wr_2026['slap_tier'] = wr_2026['slap_score'].apply(get_slap_tier)
wr_2026['best_ppr'] = np.nan
wr_2026['best_season_ppg'] = np.nan
wr_2026['hit24'] = np.nan
wr_2026['hit12'] = np.nan
wr_2026['college'] = wr_2026['school']

print(f"2026 WRs processed: {len(wr_2026)}")

# RB 2026
rb_2026['draft_year'] = 2026
rb_2026['pick'] = rb_2026['projected_pick']
rb_2026['dc_score'] = rb_2026['pick'].apply(dc_gradual)
rb_2026['breakout_age'] = rb_2026['player_name'].map(RB_2026_BREAKOUT)
rb_2026['breakout_score'] = rb_2026['breakout_age'].apply(breakout_age_to_score)
rb_2026['breakout_score_final'] = rb_2026['breakout_score'].fillna(rb_avg_breakout)
rb_2026['RAS'] = np.nan  # No combine data yet
rb_2026['ras_score'] = np.nan
rb_2026['ras_score_final'] = rb_avg_ras  # Impute with average
rb_2026['slap_score'] = (
    WEIGHT_DC * rb_2026['dc_score'] +
    WEIGHT_BREAKOUT * rb_2026['breakout_score_final'] +
    WEIGHT_RAS * rb_2026['ras_score_final']
)
rb_2026['slap_tier'] = rb_2026['slap_score'].apply(get_slap_tier)
rb_2026['best_ppr'] = np.nan
rb_2026['best_season_ppg'] = np.nan
rb_2026['hit24'] = np.nan
rb_2026['hit12'] = np.nan
rb_2026['college'] = rb_2026['school']

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
    'dc_score', 'breakout_age', 'breakout_score_final', 'RAS', 'ras_score_final',
    'slap_score', 'slap_tier', 'best_season_ppg', 'hit24', 'hit12'
]

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
    'dc_score', 'breakout_age', 'breakout_score', 'ras_raw', 'ras_score',
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

all_players.to_csv('output/slap_complete_all_players.csv', index=False)
print(f"Saved: output/slap_complete_all_players.csv ({len(all_players)} players)")

# ============================================================================
# SUMMARY TABLES
# ============================================================================
print("\n" + "=" * 90)
print("TOP 25 WRs ALL-TIME BY SLAP")
print("=" * 90)

top_wr = all_players[all_players['position'] == 'WR'].head(25)
print(f"\n{'Rank':<5} {'Player':<25} {'Year':>5} {'Pick':>5} {'DC':>6} {'BO':>5} {'RAS':>5} {'SLAP':>6} {'Tier':<10} {'PPG':>6} {'Hit':>4}")
print("-" * 100)

for _, row in top_wr.iterrows():
    ppg = f"{row['best_season_ppg']:.1f}" if pd.notna(row['best_season_ppg']) else "-"
    hit = "✓" if row['hit24'] == 1 else ("✗" if row['hit24'] == 0 else "-")
    print(f"{row['rank']:<5} {row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} "
          f"{row['dc_score']:>6.1f} {row['breakout_score']:>5.0f} {row['ras_score']:>5.1f} "
          f"{row['slap_score']:>6.1f} {row['slap_tier']:<10} {ppg:>6} {hit:>4}")

print("\n" + "=" * 90)
print("TOP 25 RBs ALL-TIME BY SLAP")
print("=" * 90)

top_rb = all_players[all_players['position'] == 'RB'].head(25)
print(f"\n{'Rank':<5} {'Player':<25} {'Year':>5} {'Pick':>5} {'DC':>6} {'BO':>5} {'RAS':>5} {'SLAP':>6} {'Tier':<10} {'PPG':>6} {'Hit':>4}")
print("-" * 100)

for _, row in top_rb.iterrows():
    ppg = f"{row['best_season_ppg']:.1f}" if pd.notna(row['best_season_ppg']) else "-"
    hit = "✓" if row['hit24'] == 1 else ("✗" if row['hit24'] == 0 else "-")
    print(f"{row['rank']:<5} {row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} "
          f"{row['dc_score']:>6.1f} {row['breakout_score']:>5.0f} {row['ras_score']:>5.1f} "
          f"{row['slap_score']:>6.1f} {row['slap_tier']:<10} {ppg:>6} {hit:>4}")

print("\n" + "=" * 90)
print("TOP 25 WRs - 2026 CLASS")
print("=" * 90)

wr_2026_ranked = all_players[(all_players['position'] == 'WR') & (all_players['draft_year'] == 2026)].head(25)
print(f"\n{'Rank':<5} {'Player':<25} {'Pick':>5} {'DC':>6} {'BO':>5} {'RAS':>5} {'SLAP':>6} {'Tier':<10}")
print("-" * 75)

for i, (_, row) in enumerate(wr_2026_ranked.iterrows(), 1):
    print(f"{i:<5} {row['player_name']:<25} {int(row['pick']):>5} "
          f"{row['dc_score']:>6.1f} {row['breakout_score']:>5.0f} {row['ras_score']:>5.1f} "
          f"{row['slap_score']:>6.1f} {row['slap_tier']:<10}")

print("\n" + "=" * 90)
print("TOP 25 RBs - 2026 CLASS")
print("=" * 90)

rb_2026_ranked = all_players[(all_players['position'] == 'RB') & (all_players['draft_year'] == 2026)].head(25)
print(f"\n{'Rank':<5} {'Player':<25} {'Pick':>5} {'DC':>6} {'BO':>5} {'RAS':>5} {'SLAP':>6} {'Tier':<10}")
print("-" * 75)

for i, (_, row) in enumerate(rb_2026_ranked.iterrows(), 1):
    print(f"{i:<5} {row['player_name']:<25} {int(row['pick']):>5} "
          f"{row['dc_score']:>6.1f} {row['breakout_score']:>5.0f} {row['ras_score']:>5.1f} "
          f"{row['slap_score']:>6.1f} {row['slap_tier']:<10}")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "=" * 90)
print("VERIFICATION: KEY PLAYER SCORES")
print("=" * 90)

key_players = [
    ('Saquon Barkley', 2, '90+'),
    ("Ja'Marr Chase", 5, '90+'),
    ('Garrett Wilson', 10, '80+'),
    ('Chris Olave', 11, '80+'),
    ('CeeDee Lamb', 17, '80+'),
    ('Justin Jefferson', 22, '75+'),
]

print(f"\n{'Player':<25} {'Pick':>5} {'SLAP':>6} {'Target':>8} {'Status':<10}")
print("-" * 60)

for name, pick, target in key_players:
    row = all_players[all_players['player_name'] == name]
    if len(row) > 0:
        slap = row.iloc[0]['slap_score']
        target_val = int(target.replace('+', ''))
        status = "✓ PASS" if slap >= target_val else "✗ FAIL"
        print(f"{name:<25} {pick:>5} {slap:>6.1f} {target:>8} {status:<10}")
    else:
        print(f"{name:<25} NOT FOUND")

# Additional verification by pick range
print("\n--- Verification by Pick Range ---")
backtest = all_players[all_players['draft_year'] < 2026]

for pick_range, target in [((1, 5), '95+'), ((6, 10), '85+'), ((21, 32), '70+'), ((76, 100), '35+')]:
    subset = backtest[(backtest['pick'] >= pick_range[0]) & (backtest['pick'] <= pick_range[1])]
    if len(subset) > 0:
        avg_slap = subset['slap_score'].mean()
        min_slap = subset['slap_score'].min()
        max_slap = subset['slap_score'].max()
        target_val = int(target.replace('+', ''))
        status = "✓" if avg_slap >= target_val else "✗"
        print(f"Picks {pick_range[0]}-{pick_range[1]}: Avg SLAP = {avg_slap:.1f} (range {min_slap:.1f}-{max_slap:.1f}) | Target: {target} {status}")

print("\n" + "=" * 90)
print("COMPLETE!")
print("=" * 90)
