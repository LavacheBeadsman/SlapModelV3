"""
RB Breakout Age Analysis

Test if breakout age predicts NFL success for RBs.
Uses manually researched breakout ages for key RBs.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("RB BREAKOUT AGE ANALYSIS")
print("Does breakout age predict NFL success for RBs?")
print("=" * 90)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n--- Loading data ---")

# Load RB backtest data
rb = pd.read_csv('data/rb_backtest_2015_2024.csv')
rb = rb[rb['draft_year'] <= 2023].copy()  # Exclude 2024 rookies
print(f"Loaded {len(rb)} RBs from backtest (2015-2023)")

# ============================================================================
# STEP 2: ASSIGN BREAKOUT AGES (MANUALLY RESEARCHED)
# ============================================================================
print("\n" + "=" * 90)
print("STEP 2: ASSIGN BREAKOUT AGES")
print("Based on when each RB first became a significant producer in college")
print("=" * 90)

# Breakout age = age at start of first season with major production
# (1000+ rushing yards OR significant receiving role)

manual_breakout = {
    # 2015 class
    'Todd Gurley': 19,  # Broke out as true sophomore at Georgia (1385 rush, 6 rec TD)
    'Melvin Gordon': 20,  # Broke out junior year at Wisconsin (2587 yards)
    'David Johnson': 21,  # Broke out junior year at Northern Iowa
    'Tevin Coleman': 20,  # Broke out sophomore at Indiana (2036 yards)
    'Duke Johnson': 18,  # Broke out freshman at Miami (947 rush + 421 rec)
    'T.J. Yeldon': 18,  # Broke out freshman at Alabama
    'Ameer Abdullah': 20,  # Broke out junior at Nebraska
    'Jay Ajayi': 20,  # Broke out junior at Boise State
    'Mike Davis': 19,  # Broke out sophomore at South Carolina

    # 2016 class
    'Ezekiel Elliott': 19,  # Broke out sophomore at Ohio State (1878 yards)
    'Derrick Henry': 21,  # Broke out senior year (Heisman) at Alabama (2219 yards)
    'Kenyan Drake': 19,  # Broke out sophomore at Alabama
    'Jordan Howard': 20,  # Broke out junior at UAB/Indiana
    'Devontae Booker': 21,  # Broke out senior at Utah
    'Paul Perkins': 18,  # Broke out freshman at UCLA

    # 2017 class
    'Christian McCaffrey': 19,  # Broke out sophomore at Stanford (Heisman finalist)
    'Leonard Fournette': 18,  # Broke out freshman at LSU (1034 yards)
    'Dalvin Cook': 18,  # Broke out freshman at Florida State (1008 yards)
    'Joe Mixon': 19,  # Broke out sophomore at Oklahoma (1274 total)
    'Alvin Kamara': 21,  # Transfer, broke out senior year at Tennessee
    'Kareem Hunt': 19,  # Broke out sophomore at Toledo (1631 yards)
    'Aaron Jones': 19,  # Broke out sophomore at UTEP
    'James Conner': 20,  # Broke out junior year at Pitt (1765 yards)
    'Samaje Perine': 18,  # Broke out freshman at Oklahoma (1713 yards)
    'Marlon Mack': 19,  # Broke out sophomore at USF

    # 2018 class
    'Saquon Barkley': 18,  # Broke out freshman at Penn State (1076 yards)
    'Nick Chubb': 18,  # Broke out freshman at Georgia (1547 yards)
    'Sony Michel': 18,  # Broke out freshman at Georgia
    'Ronald Jones': 18,  # Broke out freshman at USC (1082 yards)
    'Rashaad Penny': 21,  # Broke out senior at San Diego State (2248 yards)
    'Kerryon Johnson': 20,  # Broke out junior at Auburn
    'Royce Freeman': 18,  # Broke out freshman at Oregon
    'Nyheim Hines': 20,  # Broke out junior at NC State

    # 2019 class
    'Josh Jacobs': 20,  # Broke out junior year at Alabama (640 rush + 247 rec)
    'Miles Sanders': 21,  # Broke out senior year at Penn State (1274 yards)
    'David Montgomery': 19,  # Broke out sophomore at Iowa State
    'Darrell Henderson': 20,  # Broke out junior at Memphis (1909 yards)
    'Devin Singletary': 19,  # Broke out sophomore at FAU (1920 yards)
    'Damien Harris': 18,  # Broke out freshman at Alabama
    'Justice Hill': 18,  # Broke out freshman at Oklahoma State
    'Tony Pollard': 20,  # Broke out junior at Memphis

    # 2020 class
    'Clyde Edwards-Helaire': 21,  # Broke out junior year at LSU (1414 rush + 453 rec)
    "D'Andre Swift": 19,  # Broke out sophomore at Georgia (1049 yards)
    'Jonathan Taylor': 18,  # Broke out freshman at Wisconsin (1977 yards!)
    'Cam Akers': 18,  # Broke out freshman at Florida State (1024 yards)
    'J.K. Dobbins': 18,  # Broke out freshman at Ohio State (1403 yards)
    'Zack Moss': 19,  # Broke out sophomore at Utah
    'Antonio Gibson': 22,  # Late bloomer, broke out senior at Memphis
    'AJ Dillon': 18,  # Broke out freshman at Boston College (1108 yards)
    'Ke\'Shawn Vaughn': 20,  # Broke out junior at Vanderbilt
    'Darrynton Evans': 20,  # Broke out junior at Appalachian State

    # 2021 class
    'Najee Harris': 21,  # Broke out senior year at Alabama (1466 yards)
    'Travis Etienne': 18,  # Broke out freshman at Clemson (766 rush + 78 rec)
    'Javonte Williams': 21,  # Broke out junior year at North Carolina
    'Michael Carter': 21,  # Broke out senior at North Carolina
    'Trey Sermon': 21,  # Broke out senior (transfer) at Ohio State
    'Rhamondre Stevenson': 21,  # Broke out senior at Oklahoma
    'Kenneth Gainwell': 19,  # Broke out sophomore at Memphis (1459 yards)
    'Chuba Hubbard': 19,  # Broke out sophomore at Oklahoma State (2094 yards!)
    'Elijah Mitchell': 20,  # Broke out junior at Louisiana
    'Khalil Herbert': 20,  # Broke out junior at Kansas/VT

    # 2022 class
    'Breece Hall': 19,  # Broke out sophomore at Iowa State (1572 yards)
    'Kenneth Walker III': 21,  # Transfer, broke out senior at Michigan State (1636)
    'James Cook': 21,  # Broke out senior at Georgia
    'Rachaad White': 22,  # Late bloomer at Arizona State
    'Brian Robinson Jr.': 22,  # Broke out super senior at Alabama (1343 yards)
    'Dameon Pierce': 21,  # Broke out junior at Florida
    'Isiah Pacheco': 19,  # Broke out sophomore at Rutgers
    'Kyren Williams': 19,  # Broke out sophomore at Notre Dame (1125 yards)
    'Tyler Allgeier': 21,  # Broke out junior at BYU (1601 yards)
    'Jerome Ford': 21,  # Broke out senior at Cincinnati
    'Zamir White': 20,  # Broke out junior at Georgia

    # 2023 class
    'Bijan Robinson': 19,  # Broke out sophomore at Texas (1127 yards)
    'Jahmyr Gibbs': 19,  # Broke out sophomore at Georgia Tech (926 rush + 465 rec)
    'Zach Charbonnet': 18,  # Broke out freshman at Michigan (726 yards)
    "De'Von Achane": 20,  # Broke out junior at Texas A&M
    'Tank Bigsby': 18,  # Broke out freshman at Auburn (834 yards)
    'Tyjae Spears': 21,  # Broke out junior at Tulane (1581 yards)
    'Chase Brown': 22,  # Broke out senior at Illinois (1643 yards)
    'Kendre Miller': 21,  # Broke out junior at TCU
    'Roschon Johnson': 18,  # Played as freshman at Texas (649 yards)
    'Israel Abanikanda': 21,  # Broke out junior at Pittsburgh (1431 yards)
    'Eric Gray': 18,  # Broke out freshman at Tennessee
    'Sean Tucker': 18,  # Broke out freshman at Syracuse (1496 yards)
    'Deuce Vaughn': 18,  # Broke out freshman at Kansas State
    'Chris Rodriguez': 19,  # Broke out sophomore at Kentucky
    'DeWayne McBride': 20,  # Broke out junior at UAB
}

# Apply manual breakout ages
rb['breakout_age'] = rb['player_name'].map(manual_breakout)

# Check coverage
has_breakout = rb['breakout_age'].notna().sum()
print(f"\nRBs with breakout age assigned: {has_breakout}/{len(rb)} ({has_breakout/len(rb)*100:.1f}%)")

# Show distribution
print("\nBreakout age distribution:")
print(rb['breakout_age'].value_counts().sort_index())

# ============================================================================
# STEP 3: TEST PREDICTIVE VALUE
# ============================================================================
print("\n" + "=" * 90)
print("STEP 3: TEST PREDICTIVE VALUE")
print("=" * 90)

# Filter to RBs with breakout age and fantasy production
rb_test = rb[
    (rb['breakout_age'].notna()) &
    (rb['best_ppg'] > 0)
].copy()

print(f"\nRBs with breakout age and fantasy data: {len(rb_test)}")

# Correlation: breakout age vs NFL PPG
r_breakout, p_breakout = stats.pearsonr(rb_test['breakout_age'], rb_test['best_ppg'])
print(f"\nBreakout Age vs NFL PPG:")
print(f"  Pearson r = {r_breakout:.3f}, p = {p_breakout:.4f}")
print(f"  {'SIGNIFICANT' if p_breakout < 0.05 else 'NOT significant'} at p<0.05")

# Note: negative correlation means younger breakout = higher PPG (good)
if r_breakout < 0:
    print(f"  Direction: CORRECT (younger breakout → higher PPG)")
else:
    print(f"  Direction: WRONG (older breakout → higher PPG)")

# Spearman (rank) correlation
rho, p_rho = stats.spearmanr(rb_test['breakout_age'], rb_test['best_ppg'])
print(f"\nSpearman rank correlation:")
print(f"  rho = {rho:.3f}, p = {p_rho:.4f}")

# Compare to DC correlation
rb_test['inv_sqrt_pick'] = 1 / np.sqrt(rb_test['pick'])
r_dc, p_dc = stats.pearsonr(rb_test['inv_sqrt_pick'], rb_test['best_ppg'])
print(f"\nDraft Capital vs NFL PPG (same sample):")
print(f"  Pearson r = {r_dc:.3f}, p = {p_dc:.4f}")

# ============================================================================
# STEP 4: DOES IT ADD VALUE BEYOND DC?
# ============================================================================
print("\n" + "=" * 90)
print("STEP 4: DOES BREAKOUT AGE ADD VALUE BEYOND DC?")
print("=" * 90)

# Model 1: DC only
X_dc = sm.add_constant(rb_test['inv_sqrt_pick'])
model_dc = sm.OLS(rb_test['best_ppg'], X_dc).fit()

# Model 2: DC + breakout age
X_full = sm.add_constant(rb_test[['inv_sqrt_pick', 'breakout_age']])
model_full = sm.OLS(rb_test['best_ppg'], X_full).fit()

print(f"\n--- Model 1: DC Only ---")
print(f"  R² = {model_dc.rsquared:.4f}")
print(f"  DC coefficient: {model_dc.params['inv_sqrt_pick']:.3f}, p = {model_dc.pvalues['inv_sqrt_pick']:.4f}")

print(f"\n--- Model 2: DC + Breakout Age ---")
print(f"  R² = {model_full.rsquared:.4f}")
print(f"  DC coefficient: {model_full.params['inv_sqrt_pick']:.3f}, p = {model_full.pvalues['inv_sqrt_pick']:.4f}")
print(f"  Breakout Age coefficient: {model_full.params['breakout_age']:.3f}, p = {model_full.pvalues['breakout_age']:.4f}")

# Incremental value
delta_r2 = model_full.rsquared - model_dc.rsquared
print(f"\n--- Incremental Value ---")
print(f"  ΔR² = {delta_r2:.4f} ({delta_r2*100:.2f}% additional variance)")
print(f"  Breakout age {'IS' if model_full.pvalues['breakout_age'] < 0.05 else 'is NOT'} significant after controlling for DC")

# Partial correlation
resid_ppg = model_dc.resid
r_partial, p_partial = stats.pearsonr(rb_test['breakout_age'], resid_ppg)
print(f"\n--- Partial Correlation (Breakout Age | DC) ---")
print(f"  r_partial = {r_partial:.3f}, p = {p_partial:.4f}")

# ============================================================================
# STEP 5: COMPARE TO WR BREAKOUT AGE
# ============================================================================
print("\n" + "=" * 90)
print("STEP 5: COMPARE TO WR BREAKOUT AGE")
print("=" * 90)

# Load WR data
wr = pd.read_csv('data/wr_backtest_expanded_final.csv')
wr = wr[wr['draft_year'] <= 2023].copy()
wr['inv_sqrt_pick'] = 1 / np.sqrt(wr['pick'])
wr['best_ppg'] = wr['best_ppr'] / 17

# Filter to WRs with breakout age
wr_test = wr[wr['breakout_age'].notna() & (wr['best_ppg'] > 0)].copy()

r_wr, p_wr = stats.pearsonr(wr_test['breakout_age'], wr_test['best_ppg'])

# WR regression for comparison
X_dc_wr = sm.add_constant(wr_test['inv_sqrt_pick'])
model_dc_wr = sm.OLS(wr_test['best_ppg'], X_dc_wr).fit()

X_full_wr = sm.add_constant(wr_test[['inv_sqrt_pick', 'breakout_age']])
model_full_wr = sm.OLS(wr_test['best_ppg'], X_full_wr).fit()

print(f"\n{'Metric':<40} {'WRs':>12} {'RBs':>12}")
print("-" * 65)
print(f"{'Sample size':<40} {len(wr_test):>12} {len(rb_test):>12}")
print(f"{'Breakout age vs PPG (r)':<40} {r_wr:>12.3f} {r_breakout:>12.3f}")
print(f"{'Breakout age vs PPG (p)':<40} {p_wr:>12.4f} {p_breakout:>12.4f}")
print(f"{'Significant?':<40} {'YES' if p_wr < 0.05 else 'NO':>12} {'YES' if p_breakout < 0.05 else 'NO':>12}")
print(f"{'DC-only R²':<40} {model_dc_wr.rsquared:>12.4f} {model_dc.rsquared:>12.4f}")
print(f"{'DC + Breakout R²':<40} {model_full_wr.rsquared:>12.4f} {model_full.rsquared:>12.4f}")
print(f"{'Breakout p-value (after DC)':<40} {model_full_wr.pvalues['breakout_age']:>12.4f} {model_full.pvalues['breakout_age']:>12.4f}")

# ============================================================================
# STEP 6: SHOW EXAMPLES
# ============================================================================
print("\n" + "=" * 90)
print("STEP 6: EXAMPLES - YOUNGEST AND OLDEST BREAKOUTS")
print("=" * 90)

# Youngest breakouts
print("\n--- 10 YOUNGEST BREAKOUT AGES (Age 18) ---")
youngest = rb_test[rb_test['breakout_age'] == 18].nlargest(10, 'best_ppg')[
    ['player_name', 'draft_year', 'pick', 'breakout_age', 'best_ppg', 'hit24']
]
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'Brk Age':>8} {'PPG':>7} {'Hit':>5}")
print("-" * 60)
for _, row in youngest.iterrows():
    hit = "✓" if row['hit24'] == 1 else ""
    print(f"{row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} {row['breakout_age']:>8.0f} {row['best_ppg']:>7.1f} {hit:>5}")

# Age 18-19 group
young_group = rb_test[rb_test['breakout_age'] <= 19]
avg_ppg_young = young_group['best_ppg'].mean()
hit_rate_young = young_group['hit24'].mean() * 100

# Oldest breakouts
print("\n--- 10 OLDEST BREAKOUT AGES (Age 21-22) ---")
oldest = rb_test[rb_test['breakout_age'] >= 21].nsmallest(10, 'best_ppg')[
    ['player_name', 'draft_year', 'pick', 'breakout_age', 'best_ppg', 'hit24']
]
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'Brk Age':>8} {'PPG':>7} {'Hit':>5}")
print("-" * 60)
for _, row in oldest.iterrows():
    hit = "✓" if row['hit24'] == 1 else ""
    print(f"{row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} {row['breakout_age']:>8.0f} {row['best_ppg']:>7.1f} {hit:>5}")

# Age 21-22 group
old_group = rb_test[rb_test['breakout_age'] >= 21]
avg_ppg_old = old_group['best_ppg'].mean()
hit_rate_old = old_group['hit24'].mean() * 100

# ============================================================================
# STEP 7: HIT RATES BY BREAKOUT AGE GROUP
# ============================================================================
print("\n" + "=" * 90)
print("STEP 7: HIT RATES BY BREAKOUT AGE GROUP")
print("=" * 90)

print(f"\n{'Age':>5} {'N':>6} {'Hits':>6} {'Hit Rate':>10} {'Avg PPG':>10} {'Avg Pick':>10}")
print("-" * 55)

for age in [18, 19, 20, 21, 22]:
    group = rb_test[rb_test['breakout_age'] == age]
    if len(group) >= 3:
        hits = group['hit24'].sum()
        hit_rate = group['hit24'].mean() * 100
        avg_ppg = group['best_ppg'].mean()
        avg_pick = group['pick'].mean()
        print(f"{age:>5} {len(group):>6} {int(hits):>6} {hit_rate:>9.1f}% {avg_ppg:>10.1f} {avg_pick:>10.0f}")

print(f"\n--- Summary ---")
print(f"Young breakout (18-19): n={len(young_group)}, Hit Rate={hit_rate_young:.1f}%, Avg PPG={avg_ppg_young:.1f}")
print(f"Old breakout (21-22):   n={len(old_group)}, Hit Rate={hit_rate_old:.1f}%, Avg PPG={avg_ppg_old:.1f}")
print(f"Spread: {hit_rate_young - hit_rate_old:+.1f}% hit rate, {avg_ppg_young - avg_ppg_old:+.1f} PPG")

# ============================================================================
# STEP 8: PASS THE SMELL TEST?
# ============================================================================
print("\n" + "=" * 90)
print("STEP 8: DOES THIS PASS THE SMELL TEST?")
print("=" * 90)

print("\n--- YOUNG BREAKOUTS WHO HIT (Age 18-19, Hit24=1) ---")
young_hits = rb_test[(rb_test['breakout_age'] <= 19) & (rb_test['hit24'] == 1)].nlargest(10, 'best_ppg')
for _, row in young_hits.iterrows():
    print(f"  {row['player_name']:<25} Age {int(row['breakout_age'])}, Pick {int(row['pick'])}, {row['best_ppg']:.1f} PPG")

print("\n--- OLD BREAKOUTS WHO BUSTED (Age 21-22, Hit24=0) ---")
old_busts = rb_test[(rb_test['breakout_age'] >= 21) & (rb_test['hit24'] == 0)].nsmallest(10, 'best_ppg')
for _, row in old_busts.iterrows():
    print(f"  {row['player_name']:<25} Age {int(row['breakout_age'])}, Pick {int(row['pick'])}, {row['best_ppg']:.1f} PPG")

print("\n--- EXCEPTIONS: YOUNG BREAKOUTS WHO BUSTED ---")
young_busts = rb_test[(rb_test['breakout_age'] <= 19) & (rb_test['hit24'] == 0)].nsmallest(5, 'best_ppg')
if len(young_busts) > 0:
    for _, row in young_busts.iterrows():
        print(f"  {row['player_name']:<25} Age {int(row['breakout_age'])}, Pick {int(row['pick'])}, {row['best_ppg']:.1f} PPG")
else:
    print("  None found!")

print("\n--- EXCEPTIONS: OLD BREAKOUTS WHO HIT ---")
old_hits = rb_test[(rb_test['breakout_age'] >= 21) & (rb_test['hit24'] == 1)].nlargest(5, 'best_ppg')
if len(old_hits) > 0:
    for _, row in old_hits.iterrows():
        print(f"  {row['player_name']:<25} Age {int(row['breakout_age'])}, Pick {int(row['pick'])}, {row['best_ppg']:.1f} PPG")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("SUMMARY: DOES RB BREAKOUT AGE PREDICT NFL SUCCESS?")
print("=" * 90)

print(f"""
FINDINGS:

1. RAW CORRELATION
   - RB Breakout Age vs PPG: r = {r_breakout:.3f}, p = {p_breakout:.4f}
   - Direction: {'CORRECT' if r_breakout < 0 else 'WRONG'} (younger = better)
   - Significant? {'YES' if p_breakout < 0.05 else 'NO'}

2. COMPARISON TO WRs
   - WR Breakout Age correlation: r = {r_wr:.3f} (p = {p_wr:.4f})
   - RB Breakout Age correlation: r = {r_breakout:.3f} (p = {p_breakout:.4f})
   - RBs show {'STRONGER' if abs(r_breakout) > abs(r_wr) else 'WEAKER'} breakout age signal

3. VALUE BEYOND DRAFT CAPITAL
   - DC alone R²: {model_dc.rsquared:.4f}
   - DC + Breakout R²: {model_full.rsquared:.4f}
   - Incremental: {delta_r2:.4f} ({delta_r2*100:.2f}%)
   - Breakout age significant after DC? {'YES' if model_full.pvalues['breakout_age'] < 0.05 else 'NO'} (p = {model_full.pvalues['breakout_age']:.4f})

4. PRACTICAL VALUE
   - Young breakout (18-19) hit rate: {hit_rate_young:.1f}%
   - Old breakout (21-22) hit rate: {hit_rate_old:.1f}%
   - Spread: {hit_rate_young - hit_rate_old:+.1f}%

5. SMELL TEST
   - Young breakouts who hit: Saquon, Dalvin, JT, Chubb, Fournette, etc. ✓
   - Old breakouts who busted: Check results above
   - Makes intuitive sense? {"YES" if r_breakout < 0 and p_breakout < 0.1 else "MIXED"}

CONCLUSION: {"RB breakout age HAS predictive value!" if p_breakout < 0.05 and r_breakout < 0 else "RB breakout age shows WEAK/MARGINAL signal" if r_breakout < 0 else "RB breakout age does NOT predict success"}
""")
