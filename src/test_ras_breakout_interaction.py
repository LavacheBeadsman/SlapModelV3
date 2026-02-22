"""
Test RAS + Breakout Age Interaction Effects

Does combining young breakout + athleticism create stronger signal than either alone?
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
ras_merged = pd.read_csv('data/wr_ras_merged.csv')
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')
breakout_ages = pd.read_csv('data/wr_breakout_age_scores.csv')

# Age score mapping
AGE_SCORES = {
    18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 15, 25: 10,
}

def get_age_score(breakout_age):
    if pd.isna(breakout_age):
        return 25
    return AGE_SCORES.get(int(breakout_age), 10)

# Merge all data
wr_nfl = hit_rates[hit_rates['position'] == 'WR'].copy()

# Merge with RAS
analysis = wr_nfl.merge(
    ras_merged[['player_name', 'draft_year', 'RAS']],
    on=['player_name', 'draft_year'],
    how='left'
)

# Merge with breakout ages
analysis = analysis.merge(
    breakout_ages[['player_name', 'draft_year', 'breakout_age']],
    on=['player_name', 'draft_year'],
    how='left'
)

# Calculate scores
analysis['best_ppg'] = analysis['best_ppr'] / 17
analysis['breakout_score'] = analysis['breakout_age'].apply(get_age_score)

print("="*80)
print("RAS + BREAKOUT AGE INTERACTION ANALYSIS")
print("="*80)

# Filter to WRs with both RAS and breakout data
valid = analysis[
    (analysis['RAS'].notna()) &
    (analysis['breakout_age'].notna()) &
    (analysis['draft_year'].isin([2020, 2021, 2022, 2023, 2024]))
].copy()

print(f"\nAnalysis sample: {len(valid)} WRs with both RAS and breakout age data")
print(f"Hit24 in sample: {valid['hit24'].sum()}")

# ============================================================================
# TEST 1: INTERACTION EFFECT VARIABLES
# ============================================================================
print("\n" + "="*80)
print("TEST 1: INTERACTION EFFECT VARIABLES")
print("="*80)

# Create combined variables
valid['young_athlete'] = (valid['breakout_score'] / 100) * (valid['RAS'] / 10)
valid['elite_young'] = ((valid['breakout_age'] <= 20) & (valid['RAS'] >= 8)).astype(int)

# Correlations
r_breakout, p_breakout = stats.pearsonr(valid['breakout_score'], valid['best_ppg'])
r_ras, p_ras = stats.pearsonr(valid['RAS'], valid['best_ppg'])
r_young_athlete, p_young_athlete = stats.pearsonr(valid['young_athlete'], valid['best_ppg'])
r_elite_young, p_elite_young = stats.pointbiserialr(valid['elite_young'], valid['best_ppg'])

print("\n   CORRELATIONS WITH NFL PPG:")
print("-"*60)
print(f"   {'Variable':<25} {'r':>8} {'p-value':>12} {'Significant':>12}")
print("-"*60)
print(f"   {'breakout_score':<25} {r_breakout:>8.3f} {p_breakout:>12.4f} {'YES' if p_breakout < 0.05 else 'no':>12}")
print(f"   {'RAS':<25} {r_ras:>8.3f} {p_ras:>12.4f} {'YES' if p_ras < 0.05 else 'no':>12}")
print(f"   {'young_athlete (combo)':<25} {r_young_athlete:>8.3f} {p_young_athlete:>12.4f} {'YES' if p_young_athlete < 0.05 else 'no':>12}")
print(f"   {'elite_young (binary)':<25} {r_elite_young:>8.3f} {p_elite_young:>12.4f} {'YES' if p_elite_young < 0.05 else 'no':>12}")

# Point-biserial with Hit24
r_breakout_hit, _ = stats.pointbiserialr(valid['hit24'], valid['breakout_score'])
r_ras_hit, _ = stats.pointbiserialr(valid['hit24'], valid['RAS'])
r_combo_hit, _ = stats.pointbiserialr(valid['hit24'], valid['young_athlete'])
r_elite_hit, _ = stats.pointbiserialr(valid['hit24'], valid['elite_young'])

print("\n   CORRELATIONS WITH HIT24:")
print("-"*60)
print(f"   {'breakout_score':<25} r = {r_breakout_hit:>7.3f}")
print(f"   {'RAS':<25} r = {r_ras_hit:>7.3f}")
print(f"   {'young_athlete (combo)':<25} r = {r_combo_hit:>7.3f}")
print(f"   {'elite_young (binary)':<25} r = {r_elite_hit:>7.3f}")

# ============================================================================
# TEST 2: SEGMENT ANALYSIS (4 GROUPS)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: SEGMENT ANALYSIS (4 GROUPS)")
print("="*80)

# Define groups
valid['young_breakout'] = valid['breakout_age'] <= 20
valid['high_ras'] = valid['RAS'] >= 7

segments = [
    ("Young (≤20) + High RAS (≥7)", valid['young_breakout'] & valid['high_ras']),
    ("Young (≤20) + Low RAS (<7)", valid['young_breakout'] & ~valid['high_ras']),
    ("Old (>20) + High RAS (≥7)", ~valid['young_breakout'] & valid['high_ras']),
    ("Old (>20) + Low RAS (<7)", ~valid['young_breakout'] & ~valid['high_ras']),
]

print("\n" + "-"*85)
print(f"{'Segment':<35} {'Count':>6} {'Hit24':>6} {'Hit%':>8} {'Avg PPG':>10} {'Avg RAS':>10}")
print("-"*85)

for name, condition in segments:
    subset = valid[condition]
    if len(subset) > 0:
        hit_count = subset['hit24'].sum()
        hit_rate = subset['hit24'].mean() * 100
        avg_ppg = subset['best_ppg'].mean()
        avg_ras = subset['RAS'].mean()
        print(f"{name:<35} {len(subset):>6} {hit_count:>6} {hit_rate:>7.1f}% {avg_ppg:>10.1f} {avg_ras:>10.2f}")

# Show players in each segment
print("\n   YOUNG + HIGH RAS players:")
young_high = valid[valid['young_breakout'] & valid['high_ras']].sort_values('best_ppg', ascending=False)
for _, row in young_high.head(10).iterrows():
    hit_str = "HIT" if row['hit24'] == 1 else ""
    print(f"     {row['player_name']:<22} Age={row['breakout_age']:.0f}, RAS={row['RAS']:.2f}, PPG={row['best_ppg']:.1f} {hit_str}")

print("\n   YOUNG + LOW RAS players (potential value):")
young_low = valid[valid['young_breakout'] & ~valid['high_ras']].sort_values('best_ppg', ascending=False)
for _, row in young_low.head(10).iterrows():
    hit_str = "HIT" if row['hit24'] == 1 else ""
    print(f"     {row['player_name']:<22} Age={row['breakout_age']:.0f}, RAS={row['RAS']:.2f}, PPG={row['best_ppg']:.1f} {hit_str}")

# ============================================================================
# TEST 3: LOGISTIC REGRESSION WITH INTERACTION TERM
# ============================================================================
print("\n" + "="*80)
print("TEST 3: LOGISTIC REGRESSION WITH INTERACTION TERM")
print("="*80)

# Prepare features
valid['interaction'] = valid['breakout_score'] * valid['RAS']

X = valid[['breakout_score', 'RAS', 'interaction']].values
y = valid['hit24'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_scaled, y)

print("\n   LOGISTIC REGRESSION COEFFICIENTS:")
print("-"*60)
print(f"   {'Variable':<25} {'Coefficient':>12} {'Direction':>12}")
print("-"*60)

feature_names = ['breakout_score', 'RAS', 'interaction']
for name, coef in zip(feature_names, model.coef_[0]):
    direction = "POSITIVE" if coef > 0 else "NEGATIVE"
    sig = "***" if abs(coef) > 0.3 else ""
    print(f"   {name:<25} {coef:>12.3f} {direction:>12} {sig}")

# Compare to model without interaction
X_no_int = valid[['breakout_score', 'RAS']].values
X_no_int_scaled = StandardScaler().fit_transform(X_no_int)
model_no_int = LogisticRegression(random_state=42, max_iter=1000)
model_no_int.fit(X_no_int_scaled, y)

print("\n   WITHOUT INTERACTION TERM:")
print("-"*60)
for name, coef in zip(['breakout_score', 'RAS'], model_no_int.coef_[0]):
    direction = "POSITIVE" if coef > 0 else "NEGATIVE"
    print(f"   {name:<25} {coef:>12.3f} {direction:>12}")

# Model accuracy comparison
from sklearn.metrics import accuracy_score
pred_with_int = model.predict(X_scaled)
pred_no_int = model_no_int.predict(X_no_int_scaled)

print(f"\n   Model accuracy WITH interaction:    {accuracy_score(y, pred_with_int)*100:.1f}%")
print(f"   Model accuracy WITHOUT interaction: {accuracy_score(y, pred_no_int)*100:.1f}%")

# ============================================================================
# TEST 4: ALPHA PROTOTYPE FLAGS
# ============================================================================
print("\n" + "="*80)
print("TEST 4: ALPHA PROTOTYPE FLAGS")
print("="*80)

# Alpha prototype: young breakout (≤19) + elite athlete (RAS ≥ 8) + day 1-2 pick (≤64)
valid['alpha_prototype'] = (
    (valid['breakout_age'] <= 19) &
    (valid['RAS'] >= 8) &
    (valid['pick'] <= 64)
).astype(int)

# Also test looser definitions
valid['proto_young_elite'] = ((valid['breakout_age'] <= 19) & (valid['RAS'] >= 8)).astype(int)
valid['proto_young_day12'] = ((valid['breakout_age'] <= 19) & (valid['pick'] <= 64)).astype(int)
valid['proto_elite_day12'] = ((valid['RAS'] >= 8) & (valid['pick'] <= 64)).astype(int)

prototypes = [
    ("Alpha (age≤19 + RAS≥8 + pick≤64)", 'alpha_prototype'),
    ("Young + Elite (age≤19 + RAS≥8)", 'proto_young_elite'),
    ("Young + Day1-2 (age≤19 + pick≤64)", 'proto_young_day12'),
    ("Elite + Day1-2 (RAS≥8 + pick≤64)", 'proto_elite_day12'),
]

print("\n   PROTOTYPE FLAGS:")
print("-"*80)
print(f"   {'Prototype':<40} {'Count':>6} {'Hits':>6} {'Hit%':>8} {'Avg PPG':>10}")
print("-"*80)

for name, col in prototypes:
    subset = valid[valid[col] == 1]
    if len(subset) > 0:
        hit_count = subset['hit24'].sum()
        hit_rate = subset['hit24'].mean() * 100
        avg_ppg = subset['best_ppg'].mean()
        print(f"   {name:<40} {len(subset):>6} {hit_count:>6} {hit_rate:>7.1f}% {avg_ppg:>10.1f}")

# Show Alpha Prototypes
print("\n   ALPHA PROTOTYPE PLAYERS:")
alphas = valid[valid['alpha_prototype'] == 1].sort_values('best_ppg', ascending=False)
if len(alphas) > 0:
    print("-"*80)
    print(f"   {'Player':<25} {'Year':>5} {'Pick':>5} {'Age':>4} {'RAS':>6} {'PPG':>6} {'Hit24':>6}")
    print("-"*80)
    for _, row in alphas.iterrows():
        hit_str = "YES" if row['hit24'] == 1 else "no"
        print(f"   {row['player_name']:<25} {row['draft_year']:>5} {row['pick']:>5.0f} "
              f"{row['breakout_age']:>4.0f} {row['RAS']:>6.2f} {row['best_ppg']:>6.1f} {hit_str:>6}")
else:
    print("   No players meet all three criteria (age≤19 + RAS≥8 + pick≤64)")

# Looser: just young + elite
print("\n   YOUNG + ELITE (age≤19 + RAS≥8) PLAYERS:")
young_elite = valid[valid['proto_young_elite'] == 1].sort_values('best_ppg', ascending=False)
if len(young_elite) > 0:
    print("-"*80)
    for _, row in young_elite.iterrows():
        hit_str = "YES" if row['hit24'] == 1 else "no"
        print(f"   {row['player_name']:<25} Pick {row['pick']:>3.0f}, Age={row['breakout_age']:.0f}, "
              f"RAS={row['RAS']:.2f}, PPG={row['best_ppg']:.1f} {hit_str}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: DOES COMBINING RAS + BREAKOUT AGE ADD VALUE?")
print("="*80)

# Calculate improvement
combo_improves_ppg = r_young_athlete > r_breakout
combo_improves_hit = r_combo_hit > r_breakout_hit

print(f"""
CORRELATION COMPARISON:
  - Breakout score alone:    r = {r_breakout:.3f} (PPG), r = {r_breakout_hit:.3f} (Hit24)
  - RAS alone:               r = {r_ras:.3f} (PPG), r = {r_ras_hit:.3f} (Hit24)
  - Combined young_athlete:  r = {r_young_athlete:.3f} (PPG), r = {r_combo_hit:.3f} (Hit24)

DOES COMBINATION IMPROVE SIGNAL?
  - vs NFL PPG: {"YES" if combo_improves_ppg else "NO"} (combo r={r_young_athlete:.3f} vs breakout r={r_breakout:.3f})
  - vs Hit24:   {"YES" if combo_improves_hit else "NO"} (combo r={r_combo_hit:.3f} vs breakout r={r_breakout_hit:.3f})

SEGMENT ANALYSIS KEY FINDING:
""")

# Get segment stats
young_high_stats = valid[valid['young_breakout'] & valid['high_ras']]
young_low_stats = valid[valid['young_breakout'] & ~valid['high_ras']]

if len(young_high_stats) > 0 and len(young_low_stats) > 0:
    yh_hit = young_high_stats['hit24'].mean() * 100
    yl_hit = young_low_stats['hit24'].mean() * 100
    print(f"  - Young + High RAS: {yh_hit:.1f}% hit rate")
    print(f"  - Young + Low RAS:  {yl_hit:.1f}% hit rate")
    print(f"  - Difference:       {yh_hit - yl_hit:+.1f}%")

    if yh_hit > yl_hit + 10:
        print("\n  → RAS ADDS VALUE within young breakouts (>10% difference)")
    else:
        print("\n  → RAS does NOT meaningfully separate young breakouts")

# Final recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if combo_improves_ppg and combo_improves_hit:
    print("""
✓ KEEP RAS as part of the model
  - Combination improves both PPG and Hit24 correlation
  - Use formula: young_athlete = (breakout_score/100) × (RAS/10)
  - Weight: Consider 35% breakout, 10% athletic (reduced from 15%)
""")
elif r_ras > 0.1 and (yh_hit - yl_hit > 5):
    print("""
⚠️ KEEP RAS with reduced weight
  - RAS adds some signal within young breakouts
  - But improvement is modest
  - Consider reducing athletic weight to 5-10%
""")
else:
    print("""
✗ DROP RAS from WR model
  - Combination does not improve signal
  - Breakout age alone is sufficient
  - Simplify model: 60% Draft Capital, 40% Breakout Age
""")
