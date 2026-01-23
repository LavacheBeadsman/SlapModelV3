"""
SLAP Score V3 - 2026 Prospect Calculator
Weights: DC (85%) + Breakout Age (10%) + RAS (5%)

Note: RAS data not available until combine (Feb-March 2026)
      All RAS values are imputed with position average
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("SLAP SCORE V3 - 2026 PROSPECT CLASS")
print("Weights: DC (85%) + Breakout Age (10%) + RAS (5%)")
print("=" * 90)

# ============================================================================
# CONFIGURATION
# ============================================================================
WEIGHT_DC = 0.85
WEIGHT_BREAKOUT = 0.10
WEIGHT_RAS = 0.05

# Breakout age scoring
BREAKOUT_AGE_SCORES = {
    18: 100,  # Freshman breakout = elite
    19: 90,
    20: 75,
    21: 60,
    22: 45,
    23: 30,
    24: 20,
}

# Position average scores (from backtest)
WR_AVG_BREAKOUT = 79.7  # ~age 19.5 equivalent
WR_AVG_RAS = 68.9
RB_AVG_BREAKOUT = 78.1  # ~age 19.5 equivalent
RB_AVG_RAS = 66.5

# ============================================================================
# 2026 WR BREAKOUT AGES (researched from college careers)
# Based on when player first hit significant production
# ============================================================================
WR_2026_BREAKOUT = {
    # Top WRs - researched based on college career
    'Carnell Tate': 19,       # Ohio State - significant production as RS freshman 2024
    'Jordyn Tyson': 20,       # Arizona State - broke out junior year 2024
    'Makai Lemon': 20,        # USC - broke out 2024 after transfer
    'Denzel Boston': 20,      # Washington - broke out junior year 2024
    'Kevin Concepcion': 19,   # Texas A&M - early impact at Texas A&M
    'Chris Bell': 21,         # Louisville - broke out senior year
    'Elijah Sarratt': 21,     # Indiana - broke out 2024 (age 21)
    'Zachariah Branch': 19,   # Georgia (from USC) - early returner impact
    'Germie Bernard': 20,     # Alabama (from Washington) - transfer breakout
    'Chris Brazzell': 21,     # Tennessee - broke out 2024
    "Ja'Kobi Lane": 20,       # USC - breakout 2024
    'Omar Cooper Jr.': 21,    # Indiana - broke out 2024
    'Antonio Williams': 21,   # Clemson - late breakout
    'Skyler Bell': 21,        # UConn - transfer breakout
    'Malachi Fields': 21,     # Notre Dame - limited production
    'C.J. Daniels': 20,       # Miami - transfer breakout
    'Brenen Thompson': 21,    # Mississippi State - broke out 2024
    'Deion Burks': 20,        # Oklahoma (from Purdue) - early starter
    'Ted Hurst': 21,          # Georgia State - broke out 2024
    'Bryce Lance': 21,        # North Dakota State - FCS breakout
    'Kevin Coleman Jr.': 20,  # Missouri (from Arizona State) - early talent
    'Eric McAlister': 21,     # TCU - broke out 2024
    'Eric Rivers': 21,        # Georgia Tech - late breakout
    'Lewis Bond': 21,         # Boston College - broke out 2024
    "De'Zhaun Stribling": 20, # Ole Miss - early production
    'Keelan Marion': 20,      # Miami - breakout 2024
    'Josh Cameron': 21,       # Baylor - late breakout
    'Noah Thomas': 19,        # Georgia - early talent (limited role)
    'Aaron Anderson': 20,     # LSU - showed early
    'Dane Key': 20,           # Nebraska (from Kentucky) - early starter
    'Jordan Hudson': 20,      # SMU - breakout 2024
    'Caleb Douglas': 20,      # Texas Tech - early production
    'Reggie Virgil': 20,      # Texas Tech - showed early
    'Vinny Anthony II': 21,   # Wisconsin - limited production
    'Caullin Lacy': 21,       # Louisville (from South Alabama) - late
    'Kendrick Law': 21,       # Kentucky - late breakout
    'Colbie Young': 21,       # Georgia - limited production
    'Harrison Wallace III': 20, # Ole Miss - early production
    'Jaden Greathouse': 19,   # Notre Dame - talented but limited snaps
    'Barion Brown': 19,       # LSU (from Kentucky) - early return specialist
    'Amare Thomas': 18,       # Houston - freshman breakout 2024
    'Hykeem Williams': 20,    # Colorado (from Florida State) - early talent
    'Shelton Sampson Jr.': 19, # Louisiana (from LSU) - early production
}

# ============================================================================
# 2026 RB BREAKOUT AGES (researched from college careers)
# ============================================================================
RB_2026_BREAKOUT = {
    'Jeremiyah Love': 19,     # Notre Dame - broke out sophomore year 2024
    'Jonah Coleman': 20,      # Washington (from Arizona) - breakout 2024
    'Jadarian Price': 20,     # Notre Dame - backup role
    'Emmett Johnson': 20,     # Nebraska - broke out 2024
    'Nick Singleton': 19,     # Penn State - broke out sophomore 2023
    'Kaytron Allen': 19,      # Penn State - broke out early
    'Demond Claiborne': 20,   # Wake Forest - breakout 2024
    'Mike Washington Jr.': 21, # Arkansas - late breakout
    'Adam Randall': 20,       # Clemson - breakout 2024
    'Noah Whittington': 21,   # Oregon - late breakout
    'Roman Hemby': 21,        # Indiana - late breakout
    'C.J. Donaldson': 20,     # Ohio State - transfer
    "J'Mari Taylor": 21,      # Virginia - late breakout
    'Jamarion Miller': 20,    # Alabama - showed early
    'Le\'Veon Moss': 20,      # Texas A&M - breakout 2024
    'Dean Connors': 20,       # Houston - breakout 2024
    'Desmond Reid': 20,       # Pittsburgh - breakout 2024
    'Jaydn Ott': 19,          # California - broke out early 2022
    'Jamal Haynes': 21,       # Georgia Tech - late breakout
    'Quinten Joyner': 19,     # Texas Tech - early talent
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_draft_capital(pick, max_pick=262):
    """Convert pick number to 0-100 score using 1/sqrt(pick)"""
    raw = 1 / np.sqrt(pick)
    max_raw = 1 / np.sqrt(1)
    min_raw = 1 / np.sqrt(max_pick)
    return ((raw - min_raw) / (max_raw - min_raw)) * 100

def breakout_age_to_score(age):
    """Convert breakout age to 0-100 score"""
    if pd.isna(age):
        return None
    age = int(age)
    return BREAKOUT_AGE_SCORES.get(age, 25)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "=" * 90)
print("LOADING 2026 PROSPECTS")
print("=" * 90)

prospects = pd.read_csv('data/prospects_final.csv')
print(f"Total prospects: {len(prospects)}")

# Split by position
wrs = prospects[prospects['position'] == 'WR'].copy()
rbs = prospects[prospects['position'] == 'RB'].copy()
print(f"WRs: {len(wrs)}")
print(f"RBs: {len(rbs)}")

# ============================================================================
# CALCULATE WR SCORES
# ============================================================================
print("\n" + "=" * 90)
print("CALCULATING WR SLAP SCORES")
print("=" * 90)

# Add breakout ages
wrs['breakout_age'] = wrs['player_name'].map(WR_2026_BREAKOUT)
wrs['breakout_score'] = wrs['breakout_age'].apply(breakout_age_to_score)
wrs['breakout_score_final'] = wrs['breakout_score'].fillna(WR_AVG_BREAKOUT)
wrs['breakout_status'] = np.where(wrs['breakout_score'].isna(), 'imputed', 'observed')

# DC score from projected pick
wrs['dc_score'] = wrs['projected_pick'].apply(normalize_draft_capital)

# RAS - all imputed (no combine data yet)
wrs['ras_score_final'] = WR_AVG_RAS
wrs['ras_status'] = 'imputed'

# Calculate SLAP
wrs['slap_score'] = (
    WEIGHT_DC * wrs['dc_score'] +
    WEIGHT_BREAKOUT * wrs['breakout_score_final'] +
    WEIGHT_RAS * wrs['ras_score_final']
)

# Delta vs DC only
wrs['delta_vs_dc'] = wrs['slap_score'] - wrs['dc_score']

# Rank
wrs = wrs.sort_values('slap_score', ascending=False).reset_index(drop=True)
wrs['rank'] = range(1, len(wrs) + 1)

print(f"\nWRs with researched breakout age: {(wrs['breakout_status'] == 'observed').sum()}")
print(f"WRs with imputed breakout age: {(wrs['breakout_status'] == 'imputed').sum()}")

# ============================================================================
# CALCULATE RB SCORES
# ============================================================================
print("\n" + "=" * 90)
print("CALCULATING RB SLAP SCORES")
print("=" * 90)

# Add breakout ages
rbs['breakout_age'] = rbs['player_name'].map(RB_2026_BREAKOUT)
rbs['breakout_score'] = rbs['breakout_age'].apply(breakout_age_to_score)
rbs['breakout_score_final'] = rbs['breakout_score'].fillna(RB_AVG_BREAKOUT)
rbs['breakout_status'] = np.where(rbs['breakout_score'].isna(), 'imputed', 'observed')

# DC score from projected pick
rbs['dc_score'] = rbs['projected_pick'].apply(normalize_draft_capital)

# RAS - all imputed (no combine data yet)
rbs['ras_score_final'] = RB_AVG_RAS
rbs['ras_status'] = 'imputed'

# Calculate SLAP
rbs['slap_score'] = (
    WEIGHT_DC * rbs['dc_score'] +
    WEIGHT_BREAKOUT * rbs['breakout_score_final'] +
    WEIGHT_RAS * rbs['ras_score_final']
)

# Delta vs DC only
rbs['delta_vs_dc'] = rbs['slap_score'] - rbs['dc_score']

# Rank
rbs = rbs.sort_values('slap_score', ascending=False).reset_index(drop=True)
rbs['rank'] = range(1, len(rbs) + 1)

print(f"\nRBs with researched breakout age: {(rbs['breakout_status'] == 'observed').sum()}")
print(f"RBs with imputed breakout age: {(rbs['breakout_status'] == 'imputed').sum()}")

# ============================================================================
# DISPLAY WR RANKINGS
# ============================================================================
print("\n" + "=" * 90)
print("2026 WR SLAP RANKINGS")
print("=" * 90)

print(f"\n{'Rank':<5} {'Player':<28} {'School':<20} {'Pick':>5} {'SLAP':>6} {'DC':>5} {'BO':>5} {'Delta':>6}")
print("-" * 95)

for _, row in wrs.head(40).iterrows():
    bo_flag = "*" if row['breakout_status'] == 'imputed' else ""
    print(f"{row['rank']:<5} {row['player_name']:<28} {row['school']:<20} {int(row['projected_pick']):>5} "
          f"{row['slap_score']:>6.1f} {row['dc_score']:>5.1f} {row['breakout_score_final']:>4.0f}{bo_flag:<1} {row['delta_vs_dc']:>+6.1f}")

print("\n* = imputed breakout age (position average)")
print("Note: All RAS values imputed (combine not yet occurred)")

# ============================================================================
# DISPLAY RB RANKINGS
# ============================================================================
print("\n" + "=" * 90)
print("2026 RB SLAP RANKINGS")
print("=" * 90)

print(f"\n{'Rank':<5} {'Player':<28} {'School':<20} {'Pick':>5} {'SLAP':>6} {'DC':>5} {'BO':>5} {'Delta':>6}")
print("-" * 95)

for _, row in rbs.iterrows():
    bo_flag = "*" if row['breakout_status'] == 'imputed' else ""
    print(f"{row['rank']:<5} {row['player_name']:<28} {row['school']:<20} {int(row['projected_pick']):>5} "
          f"{row['slap_score']:>6.1f} {row['dc_score']:>5.1f} {row['breakout_score_final']:>4.0f}{bo_flag:<1} {row['delta_vs_dc']:>+6.1f}")

print("\n* = imputed breakout age (position average)")
print("Note: All RAS values imputed (combine not yet occurred)")

# ============================================================================
# HIGHLIGHT INTERESTING CASES
# ============================================================================
print("\n" + "=" * 90)
print("INTERESTING 2026 PROSPECTS")
print("=" * 90)

# WRs with positive delta (model likes more than draft position)
print("\nWRs where SLAP > DC (Model likes more than draft slot):")
print("-" * 60)
for _, row in wrs[wrs['delta_vs_dc'] > 1.0].head(10).iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['projected_pick']):>3} | SLAP {row['slap_score']:.1f} vs DC {row['dc_score']:.1f} | +{row['delta_vs_dc']:.1f}")

print("\nWRs where SLAP < DC (Model likes less than draft slot):")
print("-" * 60)
for _, row in wrs.nsmallest(5, 'delta_vs_dc').iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['projected_pick']):>3} | SLAP {row['slap_score']:.1f} vs DC {row['dc_score']:.1f} | {row['delta_vs_dc']:.1f}")

# RBs with positive delta
print("\nRBs where SLAP > DC (Model likes more than draft slot):")
print("-" * 60)
for _, row in rbs[rbs['delta_vs_dc'] > 1.0].head(10).iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['projected_pick']):>3} | SLAP {row['slap_score']:.1f} vs DC {row['dc_score']:.1f} | +{row['delta_vs_dc']:.1f}")

# ============================================================================
# SAVE OUTPUT
# ============================================================================
print("\n" + "=" * 90)
print("SAVING OUTPUT")
print("=" * 90)

# WR output
wr_output = wrs[[
    'rank', 'player_name', 'school', 'projected_pick', 'age',
    'slap_score', 'dc_score', 'breakout_score_final', 'ras_score_final',
    'delta_vs_dc', 'breakout_status', 'ras_status', 'breakout_age'
]].copy()
wr_output.columns = [
    'rank', 'player_name', 'school', 'projected_pick', 'age',
    'slap_score', 'dc_score', 'breakout_score', 'ras_score',
    'delta_vs_dc', 'breakout_status', 'ras_status', 'breakout_age_raw'
]
wr_output.to_csv('output/slap_2026_wr.csv', index=False)
print(f"Saved: output/slap_2026_wr.csv ({len(wr_output)} WRs)")

# RB output
rb_output = rbs[[
    'rank', 'player_name', 'school', 'projected_pick', 'age',
    'slap_score', 'dc_score', 'breakout_score_final', 'ras_score_final',
    'delta_vs_dc', 'breakout_status', 'ras_status', 'breakout_age'
]].copy()
rb_output.columns = [
    'rank', 'player_name', 'school', 'projected_pick', 'age',
    'slap_score', 'dc_score', 'breakout_score', 'ras_score',
    'delta_vs_dc', 'breakout_status', 'ras_status', 'breakout_age_raw'
]
rb_output.to_csv('output/slap_2026_rb.csv', index=False)
print(f"Saved: output/slap_2026_rb.csv ({len(rb_output)} RBs)")

# Combined
combined = pd.concat([
    wr_output.assign(position='WR'),
    rb_output.assign(position='RB')
], ignore_index=True)
combined = combined.sort_values('slap_score', ascending=False).reset_index(drop=True)
combined['overall_rank'] = range(1, len(combined) + 1)
combined.to_csv('output/slap_2026_combined.csv', index=False)
print(f"Saved: output/slap_2026_combined.csv ({len(combined)} total)")

print("\n" + "=" * 90)
print("DONE!")
print("=" * 90)
