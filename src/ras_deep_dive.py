"""
RAS DEEP DIVE - Testing All Possible Ways RAS Might Matter

6 tests to find if there's any context where RAS adds value.
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("RAS DEEP DIVE: FINDING WHERE ATHLETICISM MATTERS")
print("=" * 90)

# Load data
wr = pd.read_csv('data/wr_backtest_expanded_final.csv')
wr['best_ppg'] = wr['best_ppr'] / 17
wr['inv_sqrt_pick'] = 1 / np.sqrt(wr['pick'])

# Filter to those with RAS
wr_ras = wr[wr['RAS'].notna()].copy()
print(f"\nAnalyzing {len(wr_ras)} WRs with RAS data")
print(f"Overall hit rate: {wr_ras['hit24'].mean()*100:.1f}%")
print(f"Overall avg PPG: {wr_ras['best_ppg'].mean():.1f}")

# ============================================================================
# TEST 1: NON-LINEAR EFFECTS (RAS BUCKETS)
# ============================================================================
print("\n" + "=" * 90)
print("TEST 1: NON-LINEAR EFFECTS - Does RAS matter at the extremes?")
print("=" * 90)

# Create RAS buckets
def ras_bucket(ras):
    if ras >= 9:
        return 'Elite (9+)'
    elif ras >= 7:
        return 'Good (7-9)'
    elif ras >= 5:
        return 'Average (5-7)'
    else:
        return 'Poor (<5)'

wr_ras['ras_bucket'] = wr_ras['RAS'].apply(ras_bucket)

# Order for display
bucket_order = ['Elite (9+)', 'Good (7-9)', 'Average (5-7)', 'Poor (<5)']

print("\n" + "-" * 80)
print(f"{'RAS Bucket':<15} {'n':>6} {'Hits':>6} {'Hit Rate':>10} {'Avg PPG':>10} {'Median PPG':>12}")
print("-" * 80)

for bucket in bucket_order:
    subset = wr_ras[wr_ras['ras_bucket'] == bucket]
    n = len(subset)
    hits = subset['hit24'].sum()
    hit_rate = subset['hit24'].mean() * 100
    avg_ppg = subset['best_ppg'].mean()
    med_ppg = subset['best_ppg'].median()
    print(f"{bucket:<15} {n:>6} {hits:>6} {hit_rate:>9.1f}% {avg_ppg:>10.1f} {med_ppg:>12.1f}")

# Statistical test: Elite vs Poor
elite = wr_ras[wr_ras['ras_bucket'] == 'Elite (9+)']['best_ppg']
poor = wr_ras[wr_ras['ras_bucket'] == 'Poor (<5)']['best_ppg']
if len(elite) > 5 and len(poor) > 5:
    t_stat, p_val = stats.ttest_ind(elite, poor)
    print(f"\nElite vs Poor t-test: t={t_stat:.2f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("→ SIGNIFICANT difference between Elite and Poor RAS!")
    else:
        print("→ No significant difference between Elite and Poor RAS")

# ============================================================================
# TEST 2: INTERACTION WITH DRAFT CAPITAL
# ============================================================================
print("\n" + "=" * 90)
print("TEST 2: INTERACTION WITH DRAFT CAPITAL - Does RAS help find sleepers?")
print("=" * 90)

# Create draft tiers
def draft_tier(pick):
    if pick <= 32:
        return 'Day 1 (1-32)'
    elif pick <= 100:
        return 'Day 2 (33-100)'
    else:
        return 'Day 3 (101+)'

wr_ras['draft_tier'] = wr_ras['pick'].apply(draft_tier)

tier_order = ['Day 1 (1-32)', 'Day 2 (33-100)', 'Day 3 (101+)']

print("\n" + "-" * 80)
print(f"{'Draft Tier':<18} {'n':>5} {'RAS-PPG r':>12} {'p-value':>10} {'Significant?':>14}")
print("-" * 80)

for tier in tier_order:
    subset = wr_ras[wr_ras['draft_tier'] == tier]
    n = len(subset)
    if n > 10:
        r, p = stats.pearsonr(subset['RAS'], subset['best_ppg'])
        sig = "YES" if p < 0.05 else "NO"
        print(f"{tier:<18} {n:>5} {r:>+11.3f} {p:>10.4f} {sig:>14}")
    else:
        print(f"{tier:<18} {n:>5} {'(too few)':>12}")

# High RAS late-round picks
print("\nLate-round (Day 3) athletes:")
day3 = wr_ras[wr_ras['draft_tier'] == 'Day 3 (101+)']
day3_high_ras = day3[day3['RAS'] >= 8]
day3_low_ras = day3[day3['RAS'] < 8]

print(f"  High RAS (≥8): n={len(day3_high_ras)}, hit rate={day3_high_ras['hit24'].mean()*100:.1f}%, avg PPG={day3_high_ras['best_ppg'].mean():.1f}")
print(f"  Low RAS (<8):  n={len(day3_low_ras)}, hit rate={day3_low_ras['hit24'].mean()*100:.1f}%, avg PPG={day3_low_ras['best_ppg'].mean():.1f}")

# ============================================================================
# TEST 3: INTERACTION WITH BREAKOUT AGE
# ============================================================================
print("\n" + "=" * 90)
print("TEST 3: INTERACTION WITH BREAKOUT AGE - Young + Athletic = Best?")
print("=" * 90)

# Filter to those with both RAS and breakout age
wr_both = wr_ras[wr_ras['breakout_age'].notna()].copy()

# Create categories
wr_both['young_breakout'] = wr_both['breakout_age'] <= 20
wr_both['high_ras'] = wr_both['RAS'] >= 8

# Create combined category
def combo_category(row):
    if row['young_breakout'] and row['high_ras']:
        return 'Young + High RAS'
    elif row['young_breakout'] and not row['high_ras']:
        return 'Young + Low RAS'
    elif not row['young_breakout'] and row['high_ras']:
        return 'Old + High RAS'
    else:
        return 'Old + Low RAS'

wr_both['combo'] = wr_both.apply(combo_category, axis=1)

combo_order = ['Young + High RAS', 'Young + Low RAS', 'Old + High RAS', 'Old + Low RAS']

print(f"\nAnalyzing {len(wr_both)} WRs with both RAS and breakout age")
print("\n" + "-" * 80)
print(f"{'Combination':<20} {'n':>5} {'Hits':>5} {'Hit Rate':>10} {'Avg PPG':>10} {'Top Performers':>20}")
print("-" * 80)

for combo in combo_order:
    subset = wr_both[wr_both['combo'] == combo]
    n = len(subset)
    hits = subset['hit24'].sum()
    hit_rate = subset['hit24'].mean() * 100
    avg_ppg = subset['best_ppg'].mean()

    # Top performers in category
    top = subset.nlargest(3, 'best_ppg')['player_name'].tolist()
    top_str = ', '.join(top[:2]) if len(top) >= 2 else ', '.join(top)

    print(f"{combo:<20} {n:>5} {hits:>5} {hit_rate:>9.1f}% {avg_ppg:>10.1f} {top_str:>20}")

# Chi-square test for Young+HighRAS vs others
young_high = wr_both[wr_both['combo'] == 'Young + High RAS']['hit24']
others = wr_both[wr_both['combo'] != 'Young + High RAS']['hit24']
if len(young_high) > 5:
    # Fisher's exact test
    contingency = [[young_high.sum(), len(young_high) - young_high.sum()],
                   [others.sum(), len(others) - others.sum()]]
    odds_ratio, p_val = stats.fisher_exact(contingency)
    print(f"\nYoung+High RAS vs Others: OR={odds_ratio:.2f}, p={p_val:.4f}")

# ============================================================================
# TEST 4: RAS AS CEILING INDICATOR
# ============================================================================
print("\n" + "=" * 90)
print("TEST 4: RAS AS CEILING/VARIANCE INDICATOR")
print("=" * 90)

print("\nDoes RAS predict upside potential rather than average outcome?")

# Split by RAS
high_ras = wr_ras[wr_ras['RAS'] >= 8]
low_ras = wr_ras[wr_ras['RAS'] < 8]

print("\n" + "-" * 80)
print(f"{'Metric':<30} {'High RAS (≥8)':>18} {'Low RAS (<8)':>18} {'Difference':>12}")
print("-" * 80)

metrics = [
    ('n', len(high_ras), len(low_ras)),
    ('Mean PPG', high_ras['best_ppg'].mean(), low_ras['best_ppg'].mean()),
    ('Median PPG', high_ras['best_ppg'].median(), low_ras['best_ppg'].median()),
    ('Std Dev PPG', high_ras['best_ppg'].std(), low_ras['best_ppg'].std()),
    ('90th percentile PPG', high_ras['best_ppg'].quantile(0.9), low_ras['best_ppg'].quantile(0.9)),
    ('75th percentile PPG', high_ras['best_ppg'].quantile(0.75), low_ras['best_ppg'].quantile(0.75)),
    ('Max PPG', high_ras['best_ppg'].max(), low_ras['best_ppg'].max()),
    ('Hit Rate (%)', high_ras['hit24'].mean()*100, low_ras['hit24'].mean()*100),
]

for metric, high_val, low_val in metrics:
    diff = high_val - low_val
    print(f"{metric:<30} {high_val:>18.1f} {low_val:>18.1f} {diff:>+11.1f}")

# Levene's test for variance equality
stat, p_var = stats.levene(high_ras['best_ppg'], low_ras['best_ppg'])
print(f"\nLevene's test for variance: stat={stat:.2f}, p={p_var:.4f}")
if p_var < 0.05:
    print("→ HIGH RAS has significantly different variance than LOW RAS")
else:
    print("→ No significant difference in variance")

# ============================================================================
# TEST 5: INDIVIDUAL COMBINE METRICS
# ============================================================================
print("\n" + "=" * 90)
print("TEST 5: INDIVIDUAL COMBINE METRICS")
print("=" * 90)

# Try to get combine data from nflverse
combine_url = "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.csv"

try:
    print("Fetching combine data from nflverse...")
    combine_df = pd.read_csv(combine_url)
    combine_wr = combine_df[combine_df['pos'] == 'WR'].copy()

    # Merge with our WR data
    wr_combine = wr_ras.merge(
        combine_wr[['player_name', 'forty', 'vertical', 'broad_jump', 'cone', 'shuttle', 'bench']],
        on='player_name',
        how='left'
    )

    # Test each metric
    metrics_to_test = [
        ('forty', 'lower'),  # Lower is better
        ('vertical', 'higher'),
        ('broad_jump', 'higher'),
        ('cone', 'lower'),  # Lower is better (3-cone drill)
        ('shuttle', 'lower'),  # Lower is better
        ('bench', 'higher'),
    ]

    print("\n" + "-" * 80)
    print(f"{'Metric':<15} {'n':>6} {'r with PPG':>12} {'p-value':>10} {'Significant?':>14}")
    print("-" * 80)

    for metric, direction in metrics_to_test:
        subset = wr_combine[wr_combine[metric].notna()]
        n = len(subset)
        if n > 20:
            r, p = stats.pearsonr(subset[metric], subset['best_ppg'])
            # Flip sign if lower is better
            if direction == 'lower':
                r = -r
            sig = "YES" if p < 0.05 else "NO"
            print(f"{metric:<15} {n:>6} {r:>+11.3f} {p:>10.4f} {sig:>14}")
        else:
            print(f"{metric:<15} {n:>6} {'(too few)':>12}")

    # Focus on 3-cone (route running proxy)
    print("\n3-Cone Drill Analysis (route running proxy):")
    cone_data = wr_combine[wr_combine['cone'].notna()]
    if len(cone_data) > 20:
        fast_cone = cone_data[cone_data['cone'] <= cone_data['cone'].median()]
        slow_cone = cone_data[cone_data['cone'] > cone_data['cone'].median()]
        print(f"  Fast 3-cone (≤{cone_data['cone'].median():.2f}s): n={len(fast_cone)}, hit rate={fast_cone['hit24'].mean()*100:.1f}%, avg PPG={fast_cone['best_ppg'].mean():.1f}")
        print(f"  Slow 3-cone (>{cone_data['cone'].median():.2f}s): n={len(slow_cone)}, hit rate={slow_cone['hit24'].mean()*100:.1f}%, avg PPG={slow_cone['best_ppg'].mean():.1f}")

except Exception as e:
    print(f"Could not load combine data: {e}")
    print("Skipping individual metric analysis")

# ============================================================================
# TEST 6: RAS RESIDUAL (Outperforming Athletic Expectations)
# ============================================================================
print("\n" + "=" * 90)
print("TEST 6: RAS RESIDUAL - More Athletic Than Draft Slot Suggests")
print("=" * 90)

# Calculate expected RAS for each draft position
from sklearn.linear_model import LinearRegression

X_pick = wr_ras[['inv_sqrt_pick']].values
y_ras = wr_ras['RAS'].values

model_ras = LinearRegression().fit(X_pick, y_ras)
wr_ras['expected_ras'] = model_ras.predict(X_pick)
wr_ras['ras_residual'] = wr_ras['RAS'] - wr_ras['expected_ras']

print(f"\nRAS Residual = Actual RAS - Expected RAS (based on draft pick)")
print(f"Positive = More athletic than draft position suggests")
print(f"Negative = Less athletic than draft position suggests")

# Correlation of residual with performance
r_resid, p_resid = stats.pearsonr(wr_ras['ras_residual'], wr_ras['best_ppg'])
print(f"\nRAS Residual correlation with PPG: r={r_resid:+.3f}, p={p_resid:.4f}")

# Split by residual
high_resid = wr_ras[wr_ras['ras_residual'] > 0]
low_resid = wr_ras[wr_ras['ras_residual'] <= 0]

print("\n" + "-" * 70)
print(f"{'Category':<35} {'n':>5} {'Hit Rate':>10} {'Avg PPG':>10}")
print("-" * 70)
print(f"{'More athletic than expected (resid>0)':<35} {len(high_resid):>5} {high_resid['hit24'].mean()*100:>9.1f}% {high_resid['best_ppg'].mean():>10.1f}")
print(f"{'Less athletic than expected (resid≤0)':<35} {len(low_resid):>5} {low_resid['hit24'].mean()*100:>9.1f}% {low_resid['best_ppg'].mean():>10.1f}")

# Top overperformers (high residual + hit)
print("\nTop 'more athletic than expected' HITS:")
hits_high_resid = wr_ras[(wr_ras['ras_residual'] > 1) & (wr_ras['hit24'] == 1)].nlargest(5, 'ras_residual')
for _, row in hits_high_resid.iterrows():
    print(f"  {row['player_name']}: Pick {row['pick']}, RAS {row['RAS']:.1f} (expected {row['expected_ras']:.1f}), {row['best_ppg']:.1f} PPG")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("SUMMARY: WHERE DOES RAS MATTER?")
print("=" * 90)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ TEST                          │ FINDING                    │ RAS MATTERS?  │
├─────────────────────────────────────────────────────────────────────────────┤""")

# Test 1 result
elite_hr = wr_ras[wr_ras['ras_bucket'] == 'Elite (9+)']['hit24'].mean()*100
poor_hr = wr_ras[wr_ras['ras_bucket'] == 'Poor (<5)']['hit24'].mean()*100
test1 = "YES" if abs(elite_hr - poor_hr) > 5 else "NO"
print(f"│ 1. Non-linear (buckets)       │ Elite {elite_hr:.0f}% vs Poor {poor_hr:.0f}%     │ {test1:<13} │")

# Test 2 result (Day 3)
if len(day3) > 10:
    r_day3, p_day3 = stats.pearsonr(day3['RAS'], day3['best_ppg'])
    test2 = "MAYBE" if p_day3 < 0.10 else "NO"
    print(f"│ 2. Day 3 sleepers             │ r={r_day3:+.2f}, p={p_day3:.2f}             │ {test2:<13} │")

# Test 3 result
yh_hr = wr_both[wr_both['combo'] == 'Young + High RAS']['hit24'].mean()*100
yl_hr = wr_both[wr_both['combo'] == 'Young + Low RAS']['hit24'].mean()*100
test3 = "YES" if yh_hr > yl_hr + 5 else "NO"
print(f"│ 3. Young+Athletic combo       │ Y+High {yh_hr:.0f}% vs Y+Low {yl_hr:.0f}%  │ {test3:<13} │")

# Test 4 result
high_90 = high_ras['best_ppg'].quantile(0.9)
low_90 = low_ras['best_ppg'].quantile(0.9)
test4 = "MAYBE" if high_90 > low_90 + 2 else "NO"
print(f"│ 4. Ceiling indicator          │ 90th pct: {high_90:.1f} vs {low_90:.1f}      │ {test4:<13} │")

# Test 6 result
test6 = "MAYBE" if p_resid < 0.10 else "NO"
print(f"│ 6. RAS residual               │ r={r_resid:+.2f}, p={p_resid:.2f}             │ {test6:<13} │")

print("└─────────────────────────────────────────────────────────────────────────────┘")
