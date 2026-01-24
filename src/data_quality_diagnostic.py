"""
COMPREHENSIVE DATA QUALITY DIAGNOSTIC
Identify all data gaps limiting model predictive ability
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("COMPREHENSIVE DATA QUALITY DIAGNOSTIC")
print("=" * 90)

# ============================================================================
# LOAD ALL DATA
# ============================================================================
print("\n" + "=" * 90)
print("LOADING DATA")
print("=" * 90)

wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
rb_backtest = pd.read_csv('data/rb_backtest_2015_2024.csv')
prospects_2026 = pd.read_csv('data/prospects_final.csv')

# Add position labels
wr_backtest['position'] = 'WR'
rb_backtest['position'] = 'RB'

print(f"WR backtest: {len(wr_backtest)} players")
print(f"RB backtest: {len(rb_backtest)} players")
print(f"2026 prospects: {len(prospects_2026)} players")

# RB breakout ages (from research)
RB_BREAKOUT_AGES = {
    'Saquon Barkley': 18, 'Leonard Fournette': 18, 'Ezekiel Elliott': 19,
    'Christian McCaffrey': 19, 'Dalvin Cook': 18, 'Todd Gurley': 19,
    'Melvin Gordon': 20, 'Bijan Robinson': 19, 'Jahmyr Gibbs': 19,
    'Travis Etienne': 18, 'Najee Harris': 21, 'Jonathan Taylor': 18,
    'Nick Chubb': 18, 'Josh Jacobs': 20, 'Breece Hall': 19,
    'D\'Andre Swift': 19, 'Clyde Edwards-Helaire': 21, 'Cam Akers': 18,
    'J.K. Dobbins': 18, 'Joe Mixon': 19, 'Alvin Kamara': 21,
    'Kareem Hunt': 19, 'Aaron Jones': 19, 'David Johnson': 21,
    'Derrick Henry': 21, 'Kenneth Walker III': 21, 'Javonte Williams': 21,
    'Rashaad Penny': 21, 'Sony Michel': 18, 'Kerryon Johnson': 20,
    'Ronald Jones': 18, 'Royce Freeman': 18, 'Miles Sanders': 21,
    'David Montgomery': 19, 'Devin Singletary': 19, 'Damien Harris': 18,
    'Tony Pollard': 20, 'James Conner': 20, 'Zack Moss': 19,
    'AJ Dillon': 18, 'Antonio Gibson': 22, 'Kenneth Gainwell': 19,
    'Michael Carter': 21, 'Trey Sermon': 21, 'Rhamondre Stevenson': 21,
    'Chuba Hubbard': 19, 'Elijah Mitchell': 20, 'Khalil Herbert': 20,
    'James Cook': 21, 'Rachaad White': 22, 'Brian Robinson Jr.': 22,
    'Dameon Pierce': 21, 'Isiah Pacheco': 19, 'Kyren Williams': 19,
    'Tyler Allgeier': 21, 'Jerome Ford': 21, 'Zamir White': 20,
    'Zach Charbonnet': 18, 'De\'Von Achane': 20, 'Tank Bigsby': 18,
    'Tyjae Spears': 21, 'Chase Brown': 22, 'Kendre Miller': 21,
    'Roschon Johnson': 18, 'Israel Abanikanda': 21, 'T.J. Yeldon': 18,
    'Tevin Coleman': 20, 'Duke Johnson': 18, 'Ameer Abdullah': 20,
    'Jay Ajayi': 20, 'Jordan Howard': 20, 'Kenyan Drake': 19,
    'Paul Perkins': 18, 'Samaje Perine': 18, 'Marlon Mack': 19,
    'Nyheim Hines': 20, 'Justice Hill': 18, 'Darrell Henderson': 20,
}

# Add breakout ages to RB backtest
rb_backtest['breakout_age'] = rb_backtest['player_name'].map(RB_BREAKOUT_AGES)

# ============================================================================
# PART 1: MISSING DATA AUDIT
# ============================================================================
print("\n" + "=" * 90)
print("PART 1: MISSING DATA AUDIT")
print("=" * 90)

print("\n--- 1a. Missing Data by Variable ---\n")

def calc_missing_pct(df, col):
    if col not in df.columns:
        return 100.0
    missing = df[col].isna().sum()
    return (missing / len(df)) * 100

# WR variables
wr_vars = {
    'pick': 'pick',
    'breakout_age': 'breakout_age',
    'RAS': 'RAS',
    'best_ppr': 'best_ppr',
    'college': 'college',
}

# RB variables
rb_vars = {
    'pick': 'pick',
    'breakout_age': 'breakout_age',
    'RAS': 'RAS',
    'best_ppr': 'best_ppr',
    'college': 'college',
}

print(f"{'Variable':<20} {'WRs Missing':>15} {'RBs Missing':>15}")
print("-" * 55)

for var_name in ['pick', 'breakout_age', 'RAS', 'best_ppr', 'college']:
    wr_missing = calc_missing_pct(wr_backtest, var_name)
    rb_missing = calc_missing_pct(rb_backtest, var_name)
    print(f"{var_name:<20} {wr_missing:>14.1f}% {rb_missing:>14.1f}%")

# Count actual missing
print(f"\n--- Actual Missing Counts ---")
print(f"\nWRs ({len(wr_backtest)} total):")
print(f"  Missing breakout_age: {wr_backtest['breakout_age'].isna().sum()}")
print(f"  Missing RAS: {wr_backtest['RAS'].isna().sum()}")
print(f"  Missing best_ppr: {wr_backtest['best_ppr'].isna().sum()}")

print(f"\nRBs ({len(rb_backtest)} total):")
print(f"  Missing breakout_age: {rb_backtest['breakout_age'].isna().sum()}")
print(f"  Missing RAS: {rb_backtest['RAS'].isna().sum()}")
print(f"  Missing best_ppr: {rb_backtest['best_ppr'].isna().sum()}")

# --- 1b. Missing by Draft Year ---
print("\n--- 1b. Missing Data by Draft Year ---\n")

print(f"{'Year':>6} {'WRs Total':>10} {'WRs Complete':>14} {'RBs Total':>10} {'RBs Complete':>14}")
print("-" * 60)

for year in range(2015, 2025):
    wr_year = wr_backtest[wr_backtest['draft_year'] == year]
    rb_year = rb_backtest[rb_backtest['draft_year'] == year]

    # Complete = has breakout_age AND RAS AND best_ppr
    wr_complete = wr_year[
        wr_year['breakout_age'].notna() &
        wr_year['RAS'].notna() &
        wr_year['best_ppr'].notna()
    ]
    rb_complete = rb_year[
        rb_year['breakout_age'].notna() &
        rb_year['RAS'].notna() &
        rb_year['best_ppr'].notna()
    ]

    wr_pct = (len(wr_complete) / len(wr_year) * 100) if len(wr_year) > 0 else 0
    rb_pct = (len(rb_complete) / len(rb_year) * 100) if len(rb_year) > 0 else 0

    print(f"{year:>6} {len(wr_year):>10} {len(wr_complete):>8} ({wr_pct:>4.0f}%) {len(rb_year):>10} {len(rb_complete):>8} ({rb_pct:>4.0f}%)")

# --- 1c. Missing by Draft Round ---
print("\n--- 1c. Missing Data by Draft Round ---\n")

def get_round(pick):
    if pd.isna(pick):
        return None
    return min(7, int((pick - 1) // 32) + 1)

wr_backtest['round'] = wr_backtest['pick'].apply(get_round)
rb_backtest['round'] = rb_backtest['pick'].apply(get_round)

print("WRs:")
print(f"{'Round':>6} {'Total':>8} {'Missing BO':>12} {'Missing RAS':>12}")
print("-" * 45)

for rnd in range(1, 8):
    wr_rnd = wr_backtest[wr_backtest['round'] == rnd]
    if len(wr_rnd) > 0:
        missing_bo = wr_rnd['breakout_age'].isna().sum()
        missing_ras = wr_rnd['RAS'].isna().sum()
        print(f"{rnd:>6} {len(wr_rnd):>8} {missing_bo:>8} ({missing_bo/len(wr_rnd)*100:>4.0f}%) {missing_ras:>8} ({missing_ras/len(wr_rnd)*100:>4.0f}%)")

print("\nRBs:")
print(f"{'Round':>6} {'Total':>8} {'Missing BO':>12} {'Missing RAS':>12}")
print("-" * 45)

for rnd in range(1, 8):
    rb_rnd = rb_backtest[rb_backtest['round'] == rnd]
    if len(rb_rnd) > 0:
        missing_bo = rb_rnd['breakout_age'].isna().sum()
        missing_ras = rb_rnd['RAS'].isna().sum()
        print(f"{rnd:>6} {len(rb_rnd):>8} {missing_bo:>8} ({missing_bo/len(rb_rnd)*100:>4.0f}%) {missing_ras:>8} ({missing_ras/len(rb_rnd)*100:>4.0f}%)")

# --- 1d. Specific Players Missing Critical Data ---
print("\n--- 1d. Day 1-2 Players (pick ≤ 100) Missing Critical Data ---\n")

print("WRs missing breakout age (pick ≤ 100):")
wr_missing_bo = wr_backtest[(wr_backtest['pick'] <= 100) & (wr_backtest['breakout_age'].isna())]
if len(wr_missing_bo) > 0:
    for _, row in wr_missing_bo.head(15).iterrows():
        print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} ({row['draft_year']})")
else:
    print("  None")

print(f"\nWRs missing RAS (pick ≤ 100):")
wr_missing_ras = wr_backtest[(wr_backtest['pick'] <= 100) & (wr_backtest['RAS'].isna())]
if len(wr_missing_ras) > 0:
    for _, row in wr_missing_ras.head(15).iterrows():
        hit = "HIT" if row['hit24'] == 1 else "miss"
        print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} ({row['draft_year']}) - {hit}")
else:
    print("  None")

print(f"\nRBs missing breakout age (pick ≤ 100):")
rb_missing_bo = rb_backtest[(rb_backtest['pick'] <= 100) & (rb_backtest['breakout_age'].isna())]
if len(rb_missing_bo) > 0:
    for _, row in rb_missing_bo.head(15).iterrows():
        hit = "HIT" if row['hit24'] == 1 else "miss"
        print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} ({row['draft_year']}) - {hit}")
else:
    print("  None")

print(f"\nRBs missing RAS (pick ≤ 100):")
rb_missing_ras = rb_backtest[(rb_backtest['pick'] <= 100) & (rb_backtest['RAS'].isna())]
if len(rb_missing_ras) > 0:
    for _, row in rb_missing_ras.head(15).iterrows():
        hit = "HIT" if row['hit24'] == 1 else "miss"
        print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} ({row['draft_year']}) - {hit}")
else:
    print("  None")

# ============================================================================
# PART 2: SUSPICIOUS DATA
# ============================================================================
print("\n" + "=" * 90)
print("PART 2: SUSPICIOUS DATA")
print("=" * 90)

print("\n--- 2a. Breakout Age Out of Range (<18 or >24) ---")
wr_bad_bo = wr_backtest[(wr_backtest['breakout_age'] < 18) | (wr_backtest['breakout_age'] > 24)]
rb_bad_bo = rb_backtest[(rb_backtest['breakout_age'] < 18) | (rb_backtest['breakout_age'] > 24)]
print(f"WRs with suspicious breakout age: {len(wr_bad_bo)}")
print(f"RBs with suspicious breakout age: {len(rb_bad_bo)}")

print("\n--- 2b. RAS Out of Range (<1 or >10) ---")
wr_bad_ras = wr_backtest[(wr_backtest['RAS'] < 1) | (wr_backtest['RAS'] > 10)]
rb_bad_ras = rb_backtest[(rb_backtest['RAS'] < 1) | (rb_backtest['RAS'] > 10)]
print(f"WRs with suspicious RAS: {len(wr_bad_ras)}")
print(f"RBs with suspicious RAS: {len(rb_bad_ras)}")

print("\n--- 2c. PPG Values Out of Range ---")
wr_backtest['best_ppg'] = wr_backtest['best_ppr'] / 17
rb_backtest['best_ppg'] = rb_backtest['best_ppr'] / 17

wr_bad_ppg = wr_backtest[(wr_backtest['best_ppg'] > 30) | ((wr_backtest['best_ppg'] < 1) & (wr_backtest['best_ppg'] > 0))]
rb_bad_ppg = rb_backtest[(rb_backtest['best_ppg'] > 30) | ((rb_backtest['best_ppg'] < 1) & (rb_backtest['best_ppg'] > 0))]

print(f"WRs with suspicious PPG (>30 or <1): {len(wr_bad_ppg)}")
if len(wr_bad_ppg) > 0:
    for _, row in wr_bad_ppg.head(5).iterrows():
        print(f"  {row['player_name']:<25} PPG: {row['best_ppg']:.1f}")

print(f"\nRBs with suspicious PPG (>30 or <1): {len(rb_bad_ppg)}")
if len(rb_bad_ppg) > 0:
    for _, row in rb_bad_ppg.head(5).iterrows():
        print(f"  {row['player_name']:<25} PPG: {row['best_ppg']:.1f}")

print("\n--- 2d. Players with Zero PPR (drafted but no NFL stats) ---")
wr_zero = wr_backtest[wr_backtest['best_ppr'] == 0]
rb_zero = rb_backtest[rb_backtest['best_ppr'] == 0]
print(f"WRs with 0 PPR: {len(wr_zero)} ({len(wr_zero)/len(wr_backtest)*100:.1f}%)")
print(f"RBs with 0 PPR: {len(rb_zero)} ({len(rb_zero)/len(rb_backtest)*100:.1f}%)")

# High picks with zero
wr_zero_high = wr_zero[wr_zero['pick'] <= 50]
rb_zero_high = rb_zero[rb_zero['pick'] <= 50]
print(f"\nHigh picks (≤50) with 0 PPR:")
print("WRs:")
for _, row in wr_zero_high.iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} ({row['draft_year']})")
print("RBs:")
for _, row in rb_zero_high.iterrows():
    print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} ({row['draft_year']})")

# ============================================================================
# PART 3: BACKTEST COMPLETENESS
# ============================================================================
print("\n" + "=" * 90)
print("PART 3: BACKTEST COMPLETENESS (2015-2023)")
print("=" * 90)

# Filter to 2015-2023 only (players with time to produce)
wr_bt = wr_backtest[wr_backtest['draft_year'] <= 2023].copy()
rb_bt = rb_backtest[rb_backtest['draft_year'] <= 2023].copy()

print(f"\nWRs 2015-2023: {len(wr_bt)} total")
wr_complete = wr_bt[
    wr_bt['pick'].notna() &
    wr_bt['breakout_age'].notna() &
    wr_bt['RAS'].notna() &
    wr_bt['best_ppr'].notna()
]
print(f"  Complete data (pick + breakout + RAS + PPR): {len(wr_complete)} ({len(wr_complete)/len(wr_bt)*100:.1f}%)")
print(f"  Have NFL outcomes (PPR > 0): {len(wr_bt[wr_bt['best_ppr'] > 0])} ({len(wr_bt[wr_bt['best_ppr'] > 0])/len(wr_bt)*100:.1f}%)")

print(f"\nRBs 2015-2023: {len(rb_bt)} total")
rb_complete = rb_bt[
    rb_bt['pick'].notna() &
    rb_bt['breakout_age'].notna() &
    rb_bt['RAS'].notna() &
    rb_bt['best_ppr'].notna()
]
print(f"  Complete data (pick + breakout + RAS + PPR): {len(rb_complete)} ({len(rb_complete)/len(rb_bt)*100:.1f}%)")
print(f"  Have NFL outcomes (PPR > 0): {len(rb_bt[rb_bt['best_ppr'] > 0])} ({len(rb_bt[rb_bt['best_ppr'] > 0])/len(rb_bt)*100:.1f}%)")

# Important players excluded
print("\n--- Important Players Potentially Excluded Due to Missing Data ---")
print("\nTop WR picks (≤32) missing breakout or RAS:")
wr_rd1_incomplete = wr_bt[(wr_bt['pick'] <= 32) & (wr_bt['breakout_age'].isna() | wr_bt['RAS'].isna())]
for _, row in wr_rd1_incomplete.iterrows():
    missing = []
    if pd.isna(row['breakout_age']):
        missing.append('breakout')
    if pd.isna(row['RAS']):
        missing.append('RAS')
    hit = "HIT" if row['hit24'] == 1 else "miss"
    print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} ({row['draft_year']}) - missing {', '.join(missing)} - {hit}")

print("\nTop RB picks (≤32) missing breakout or RAS:")
rb_rd1_incomplete = rb_bt[(rb_bt['pick'] <= 32) & (rb_bt['breakout_age'].isna() | rb_bt['RAS'].isna())]
for _, row in rb_rd1_incomplete.iterrows():
    missing = []
    if pd.isna(row['breakout_age']):
        missing.append('breakout')
    if pd.isna(row['RAS']):
        missing.append('RAS')
    hit = "HIT" if row['hit24'] == 1 else "miss"
    print(f"  {row['player_name']:<25} Pick {int(row['pick']):>3} ({row['draft_year']}) - missing {', '.join(missing)} - {hit}")

# ============================================================================
# PART 4: 2024-2026 CLASS ISSUES
# ============================================================================
print("\n" + "=" * 90)
print("PART 4: 2024-2026 CLASS DATA ISSUES")
print("=" * 90)

# 2024 class (in backtest)
wr_2024 = wr_backtest[wr_backtest['draft_year'] == 2024]
rb_2024 = rb_backtest[rb_backtest['draft_year'] == 2024]

print("\n--- 2024 Class ---")
print(f"WRs: {len(wr_2024)} total, {wr_2024['breakout_age'].isna().sum()} missing breakout, {wr_2024['RAS'].isna().sum()} missing RAS")
print(f"RBs: {len(rb_2024)} total, {rb_2024['breakout_age'].isna().sum()} missing breakout, {rb_2024['RAS'].isna().sum()} missing RAS")

# 2026 prospects
wr_2026 = prospects_2026[prospects_2026['position'] == 'WR']
rb_2026 = prospects_2026[prospects_2026['position'] == 'RB']

# 2026 breakout ages (from our research)
WR_2026_BREAKOUT = {
    'Carnell Tate': 19, 'Jordyn Tyson': 20, 'Makai Lemon': 20,
    'Denzel Boston': 20, 'Kevin Concepcion': 19, 'Chris Bell': 21,
    'Elijah Sarratt': 21, 'Zachariah Branch': 19, 'Germie Bernard': 20,
}
RB_2026_BREAKOUT = {
    'Jeremiyah Love': 19, 'Jonah Coleman': 20, 'Jadarian Price': 20,
    'Emmett Johnson': 20, 'Nick Singleton': 19, 'Kaytron Allen': 19,
}

print("\n--- 2026 Class ---")
wr_2026_has_bo = sum(1 for name in wr_2026['player_name'] if name in WR_2026_BREAKOUT)
rb_2026_has_bo = sum(1 for name in rb_2026['player_name'] if name in RB_2026_BREAKOUT)
print(f"WRs: {len(wr_2026)} total, {wr_2026_has_bo} have breakout age researched ({wr_2026_has_bo/len(wr_2026)*100:.1f}%)")
print(f"RBs: {len(rb_2026)} total, {rb_2026_has_bo} have breakout age researched ({rb_2026_has_bo/len(rb_2026)*100:.1f}%)")
print("Note: ALL 2026 prospects missing RAS (combine not yet occurred)")

# List top 2026 prospects missing breakout
print("\n2026 WRs (pick ≤ 100) missing breakout age:")
for _, row in wr_2026[wr_2026['projected_pick'] <= 100].iterrows():
    if row['player_name'] not in WR_2026_BREAKOUT:
        print(f"  {row['player_name']:<25} Pick {int(row['projected_pick']):>3}")

print("\n2026 RBs (pick ≤ 100) missing breakout age:")
for _, row in rb_2026[rb_2026['projected_pick'] <= 100].iterrows():
    if row['player_name'] not in RB_2026_BREAKOUT:
        print(f"  {row['player_name']:<25} Pick {int(row['projected_pick']):>3}")

# ============================================================================
# PART 5: IMPACT ANALYSIS
# ============================================================================
print("\n" + "=" * 90)
print("PART 5: IMPACT ANALYSIS - What's Hurting Predictions Most?")
print("=" * 90)

# Use 2015-2023 backtest with outcomes
def dc_gradual(pick):
    k, p = 2.40, 0.62
    return max(0, min(100, 100 - k * (pick**p - 1)))

BREAKOUT_SCORES = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30}

def breakout_to_score(age):
    if pd.isna(age):
        return None
    return BREAKOUT_SCORES.get(int(age), 25)

def ras_to_score(ras):
    if pd.isna(ras):
        return None
    return 50 + (ras - 5.5) / 2.5 * 25

# Prepare WR data
wr_bt['dc_score'] = wr_bt['pick'].apply(dc_gradual)
wr_bt['bo_score'] = wr_bt['breakout_age'].apply(breakout_to_score)
wr_bt['ras_score'] = wr_bt['RAS'].apply(ras_to_score)

# Prepare RB data
rb_bt['dc_score'] = rb_bt['pick'].apply(dc_gradual)
rb_bt['bo_score'] = rb_bt['breakout_age'].apply(breakout_to_score)
rb_bt['ras_score'] = rb_bt['RAS'].apply(ras_to_score)

# 5a. DC-only baseline
print("\n--- 5a. DC-Only Model (no missing data issues) ---")

wr_valid = wr_bt[wr_bt['best_ppr'] > 0].copy()
rb_valid = rb_bt[rb_bt['best_ppr'] > 0].copy()

r_wr_dc, _ = spearmanr(wr_valid['dc_score'], wr_valid['best_ppr'])
r_rb_dc, _ = spearmanr(rb_valid['dc_score'], rb_valid['best_ppr'])

print(f"WR DC-only correlation: {r_wr_dc:.4f} (n={len(wr_valid)})")
print(f"RB DC-only correlation: {r_rb_dc:.4f} (n={len(rb_valid)})")

# 5b. Adding breakout (where available)
print("\n--- 5b. Impact of Adding Breakout Age ---")

wr_with_bo = wr_valid[wr_valid['bo_score'].notna()].copy()
rb_with_bo = rb_valid[rb_valid['bo_score'].notna()].copy()

wr_with_bo['slap_dc_bo'] = 0.90 * wr_with_bo['dc_score'] + 0.10 * wr_with_bo['bo_score']
rb_with_bo['slap_dc_bo'] = 0.90 * rb_with_bo['dc_score'] + 0.10 * rb_with_bo['bo_score']

r_wr_dc_bo, _ = spearmanr(wr_with_bo['slap_dc_bo'], wr_with_bo['best_ppr'])
r_rb_dc_bo, _ = spearmanr(rb_with_bo['slap_dc_bo'], rb_with_bo['best_ppr'])

print(f"WR DC+Breakout correlation: {r_wr_dc_bo:.4f} (n={len(wr_with_bo)}, delta={r_wr_dc_bo - r_wr_dc:+.4f})")
print(f"RB DC+Breakout correlation: {r_rb_dc_bo:.4f} (n={len(rb_with_bo)}, delta={r_rb_dc_bo - r_rb_dc:+.4f})")
print(f"WRs excluded due to missing breakout: {len(wr_valid) - len(wr_with_bo)}")
print(f"RBs excluded due to missing breakout: {len(rb_valid) - len(rb_with_bo)}")

# 5c. Adding RAS (where available)
print("\n--- 5c. Impact of Adding RAS ---")

wr_with_ras = wr_valid[wr_valid['ras_score'].notna()].copy()
rb_with_ras = rb_valid[rb_valid['ras_score'].notna()].copy()

wr_with_ras['slap_dc_ras'] = 0.95 * wr_with_ras['dc_score'] + 0.05 * wr_with_ras['ras_score']
rb_with_ras['slap_dc_ras'] = 0.95 * rb_with_ras['dc_score'] + 0.05 * rb_with_ras['ras_score']

r_wr_dc_ras, _ = spearmanr(wr_with_ras['slap_dc_ras'], wr_with_ras['best_ppr'])
r_rb_dc_ras, _ = spearmanr(rb_with_ras['slap_dc_ras'], rb_with_ras['best_ppr'])

print(f"WR DC+RAS correlation: {r_wr_dc_ras:.4f} (n={len(wr_with_ras)}, delta={r_wr_dc_ras - r_wr_dc:+.4f})")
print(f"RB DC+RAS correlation: {r_rb_dc_ras:.4f} (n={len(rb_with_ras)}, delta={r_rb_dc_ras - r_rb_dc:+.4f})")
print(f"WRs excluded due to missing RAS: {len(wr_valid) - len(wr_with_ras)}")
print(f"RBs excluded due to missing RAS: {len(rb_valid) - len(rb_with_ras)}")

# 5d. Full model (where all data available)
print("\n--- 5d. Full Model (DC + Breakout + RAS) ---")

wr_full = wr_valid[wr_valid['bo_score'].notna() & wr_valid['ras_score'].notna()].copy()
rb_full = rb_valid[rb_valid['bo_score'].notna() & rb_valid['ras_score'].notna()].copy()

wr_full['slap_full'] = 0.85 * wr_full['dc_score'] + 0.10 * wr_full['bo_score'] + 0.05 * wr_full['ras_score']
rb_full['slap_full'] = 0.85 * rb_full['dc_score'] + 0.10 * rb_full['bo_score'] + 0.05 * rb_full['ras_score']

r_wr_full, _ = spearmanr(wr_full['slap_full'], wr_full['best_ppr'])
r_rb_full, _ = spearmanr(rb_full['slap_full'], rb_full['best_ppr'])

print(f"WR Full model correlation: {r_wr_full:.4f} (n={len(wr_full)}, delta vs DC={r_wr_full - r_wr_dc:+.4f})")
print(f"RB Full model correlation: {r_rb_full:.4f} (n={len(rb_full)}, delta vs DC={r_rb_full - r_rb_dc:+.4f})")
print(f"WRs with complete data: {len(wr_full)}/{len(wr_valid)} ({len(wr_full)/len(wr_valid)*100:.1f}%)")
print(f"RBs with complete data: {len(rb_full)}/{len(rb_valid)} ({len(rb_full)/len(rb_valid)*100:.1f}%)")

# 5e. Priority ranking
print("\n--- 5e. DATA FIX PRIORITY RANKING ---")
print("""
PRIORITY 1: RB Breakout Ages (HIGH IMPACT)
  - Currently missing: {rb_missing} RBs in backtest
  - These are manually researched - need to complete the list
  - Impact: Moderate (breakout only 10% of model)

PRIORITY 2: WR Missing RAS (Day 1-2 picks)
  - Currently missing: {wr_ras_missing} WRs (pick ≤100)
  - These are elite opt-outs (Waddle, Smith, London, Williams)
  - Impact: Low (RAS only 5% of model) but affects top players

PRIORITY 3: 2026 Breakout Ages
  - Currently have: ~40 WRs, ~20 RBs researched
  - Need: Complete coverage for projected Day 1-2 picks
  - Impact: Moderate for 2026 projections

PRIORITY 4: 2024 Class Validation
  - Need to verify PPG data is complete
  - Some 2024 rookies may have incomplete season data

NOTE: Missing RAS for 2026 is EXPECTED (no combine yet).
      This will be filled after Feb/March 2026.
""".format(
    rb_missing=rb_bt['breakout_age'].isna().sum(),
    wr_ras_missing=len(wr_missing_ras)
))

print("\n" + "=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)
