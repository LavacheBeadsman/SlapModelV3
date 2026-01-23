"""
Test Option 2: Age-Only Breakout for WRs
- No dominator input
- Breakout age drives the entire breakout score
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load data
breakout = pd.read_csv('data/wr_breakout_age_scores.csv')
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')

# Filter to WRs only
wr_nfl = hit_rates[hit_rates['position'] == 'WR'].copy()

# Merge breakout data with NFL data
merged = wr_nfl.merge(
    breakout[['player_name', 'draft_year', 'peak_dominator', 'breakout_age']],
    on=['player_name', 'draft_year'],
    how='left'
)

# Filter to 2020-2024 and players who played
analysis = merged[
    (merged['draft_year'].isin([2020, 2021, 2022, 2023, 2024])) &
    (merged['best_ppr'].notna()) &
    (merged['best_ppr'] > 0)
].copy()

# Calculate PPG
analysis['best_ppg'] = analysis['best_ppr'] / 17

print("="*80)
print("TESTING OPTION 2: AGE-ONLY BREAKOUT FOR WRs")
print("="*80)

# ============================================================================
# STEP 1: Define age-based scoring (same as before)
# ============================================================================
AGE_SCORES = {
    18: 100,  # Best - broke out as freshman
    19: 90,
    20: 75,
    21: 60,
    22: 45,
    23: 30,
    24: 15,
    25: 10,
}

def get_age_score(breakout_age):
    """Convert breakout age to a 0-100 score"""
    if pd.isna(breakout_age):
        return 25  # Never broke out (no 20%+ season) - low score
    age = int(breakout_age)
    return AGE_SCORES.get(age, 10)

analysis['breakout_score'] = analysis['breakout_age'].apply(get_age_score)

print(f"\nAnalyzing {len(analysis)} WRs (2020-2024 drafts)")

# ============================================================================
# STEP 2: Test predictive power
# ============================================================================
print("\n" + "="*80)
print("PREDICTIVE POWER COMPARISON")
print("="*80)

# Age-only breakout score
valid_age = analysis[analysis['breakout_age'].notna()]
r_age, p_age = stats.pearsonr(valid_age['breakout_score'], valid_age['best_ppg'])
print(f"\nAGE-ONLY breakout_score vs NFL PPG:")
print(f"  r = {r_age:.3f}, p = {p_age:.4f}")

# For comparison - the threshold approach (from previous test)
# We need to calculate it here
analysis['hit_20_pct'] = analysis['peak_dominator'] >= 20
analysis['threshold_score'] = analysis.apply(
    lambda row: get_age_score(row['breakout_age']) * 0.70
                if (pd.notna(row['peak_dominator']) and not row['hit_20_pct'])
                else get_age_score(row['breakout_age']),
    axis=1
)

valid_threshold = analysis[analysis['peak_dominator'].notna()]
r_threshold, p_threshold = stats.pearsonr(valid_threshold['threshold_score'], valid_threshold['best_ppg'])
print(f"\nTHRESHOLD approach vs NFL PPG (for comparison):")
print(f"  r = {r_threshold:.3f}, p = {p_threshold:.4f}")

# Peak dominator alone
valid_dom = analysis[analysis['peak_dominator'].notna()]
r_dom, p_dom = stats.pearsonr(valid_dom['peak_dominator'], valid_dom['best_ppg'])
print(f"\nPEAK DOMINATOR alone vs NFL PPG:")
print(f"  r = {r_dom:.3f}, p = {p_dom:.4f}")

# Point-biserial with Hit24
r_hit24, p_hit24 = stats.pointbiserialr(valid_age['hit24'], valid_age['breakout_score'])
print(f"\nAGE-ONLY breakout_score vs Hit24:")
print(f"  r = {r_hit24:.3f}, p = {p_hit24:.4f}")

# ============================================================================
# STEP 3: Show how key players score
# ============================================================================
print("\n" + "="*80)
print("KEY PLAYER SCORES (Age-Only vs Threshold)")
print("="*80)

key_players = ['Jaylen Waddle', 'Jordan Addison', "Ja'Marr Chase", 'Amon-Ra St. Brown',
               'Puka Nacua', 'CeeDee Lamb', 'Justin Jefferson', 'Garrett Wilson',
               'Chris Olave', 'George Pickens', 'Nico Collins', 'Tank Dell']

print("\n" + "-"*90)
print(f"{'Player':<22} {'BrkAge':>7} {'AgeOnly':>8} {'Dom%':>7} {'Thresh':>8} {'PPG':>6} {'Hit24':>6}")
print("-"*90)

for name in key_players:
    row = analysis[analysis['player_name'] == name]
    if len(row) > 0:
        r = row.iloc[0]
        age_str = f"{r['breakout_age']:.0f}" if pd.notna(r['breakout_age']) else "Never"
        dom_str = f"{r['peak_dominator']:.1f}%" if pd.notna(r['peak_dominator']) else "N/A"
        hit_str = "YES" if r['hit24'] == 1 else "no"
        print(f"{r['player_name']:<22} {age_str:>7} {r['breakout_score']:>8.0f} "
              f"{dom_str:>7} {r['threshold_score']:>8.0f} {r['best_ppg']:>6.1f} {hit_str:>6}")

# ============================================================================
# STEP 4: Quartile analysis
# ============================================================================
print("\n" + "="*80)
print("QUARTILE ANALYSIS (Age-Only Breakout Score)")
print("="*80)

q25 = analysis['breakout_score'].quantile(0.25)
q50 = analysis['breakout_score'].quantile(0.50)
q75 = analysis['breakout_score'].quantile(0.75)

print(f"\nScore quartile cutoffs: Q1={q25:.0f}, Q2={q50:.0f}, Q3={q75:.0f}")

for label, condition in [
    ("Top 25%", analysis['breakout_score'] >= q75),
    ("2nd quartile", (analysis['breakout_score'] >= q50) & (analysis['breakout_score'] < q75)),
    ("3rd quartile", (analysis['breakout_score'] >= q25) & (analysis['breakout_score'] < q50)),
    ("Bottom 25%", analysis['breakout_score'] < q25)
]:
    subset = analysis[condition]
    print(f"\n{label} (n={len(subset)}):")
    print(f"  - Avg PPG: {subset['best_ppg'].mean():.1f}")
    print(f"  - Hit24 rate: {subset['hit24'].mean()*100:.1f}%")

# ============================================================================
# STEP 5: Top 20 by age-only score
# ============================================================================
print("\n" + "="*80)
print("TOP 20 WRs BY AGE-ONLY BREAKOUT SCORE")
print("="*80)

top20 = analysis.nlargest(20, 'breakout_score')[
    ['player_name', 'draft_year', 'pick', 'breakout_age', 'breakout_score', 'best_ppg', 'hit24']
].copy()

print("\n" + "-"*85)
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'BrkAge':>7} {'Score':>6} {'PPG':>6} {'Hit24':>6}")
print("-"*85)
for _, row in top20.iterrows():
    age_str = f"{row['breakout_age']:.0f}" if pd.notna(row['breakout_age']) else "Never"
    hit_str = "YES" if row['hit24'] == 1 else "no"
    print(f"{row['player_name']:<25} {row['draft_year']:>5} {row['pick']:>5} "
          f"{age_str:>7} {row['breakout_score']:>6.0f} {row['best_ppg']:>6.1f} {hit_str:>6}")

hits_top20 = top20['hit24'].sum()
print(f"\nHits in top 20: {hits_top20}/20 ({hits_top20/20*100:.1f}%)")

# ============================================================================
# STEP 6: Check Waddle and Addison specifically
# ============================================================================
print("\n" + "="*80)
print("WADDLE & ADDISON: Age-Only vs Threshold")
print("="*80)

for name in ['Jaylen Waddle', 'Jordan Addison']:
    row = analysis[analysis['player_name'] == name]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"\n{r['player_name']}:")
        print(f"  Breakout Age: {r['breakout_age']:.0f}" if pd.notna(r['breakout_age']) else "  Breakout Age: Never")
        print(f"  Peak Dominator: {r['peak_dominator']:.1f}%" if pd.notna(r['peak_dominator']) else "  Peak Dominator: N/A")
        print(f"  Age-Only Score: {r['breakout_score']:.0f}")
        print(f"  Threshold Score: {r['threshold_score']:.0f}")
        print(f"  NFL PPG: {r['best_ppg']:.1f}")
        print(f"  Hit24: {'YES' if r['hit24'] == 1 else 'no'}")

        if r['breakout_score'] > r['threshold_score']:
            print(f"  → Age-Only is BETTER (+{r['breakout_score'] - r['threshold_score']:.0f} points)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: AGE-ONLY VS OTHER APPROACHES")
print("="*80)

print(f"""
Correlation with NFL PPG:
  - Age-Only:    r = {r_age:.3f} (p = {p_age:.4f})
  - Threshold:   r = {r_threshold:.3f} (p = {p_threshold:.4f})
  - Dominator:   r = {r_dom:.3f} (p = {p_dom:.4f})

Age-Only vs Threshold difference: {r_age - r_threshold:+.3f}

Key benefits of Age-Only:
  ✓ Waddle scores {analysis[analysis['player_name']=='Jaylen Waddle']['breakout_score'].values[0]:.0f} (not penalized)
  ✓ Addison scores {analysis[analysis['player_name']=='Jordan Addison']['breakout_score'].values[0]:.0f} (not penalized)
  ✓ No teammate/transfer data issues
  ✓ Simpler model
""")
