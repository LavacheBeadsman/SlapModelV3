"""
VERIFICATION TEST - Proving calculations are real
"""
import pandas as pd
import numpy as np
from scipy import stats

print("=" * 90)
print("VERIFICATION TEST - RAW DATA AND MANUAL CALCULATIONS")
print("=" * 90)

# Load data
wr = pd.read_csv('data/wr_backtest_expanded_final.csv')
wr['best_ppg'] = wr['best_ppr'] / 17

# ============================================================================
# PART 1: RAW DATA FOR 10 SPECIFIC WRS
# ============================================================================
print("\n" + "=" * 90)
print("PART 1: RAW DATA FOR 10 SPECIFIC WRS")
print("=" * 90)

target_players = [
    "Ja'Marr Chase",
    "Jaylen Waddle",
    "DeVonta Smith",
    "Chris Olave",
    "Garrett Wilson",
    "CeeDee Lamb",
    "Justin Jefferson",
    "Jerry Jeudy",
    "Henry Ruggs III",
    "Jalen Reagor"
]

print("\n" + "-" * 100)
print(f"{'Player':<20} {'Pick':>6} {'RAS':>8} {'Breakout':>10} {'Best PPG':>10} {'Hit24':>6} {'Draft Yr':>10}")
print("-" * 100)

sample_data = []
for name in target_players:
    row = wr[wr['player_name'] == name]
    if len(row) > 0:
        r = row.iloc[0]
        pick = r['pick']
        ras = r['RAS'] if pd.notna(r['RAS']) else 'N/A'
        breakout = r['breakout_age'] if pd.notna(r['breakout_age']) else 'N/A'
        ppg = r['best_ppg']
        hit24 = int(r['hit24'])
        draft_yr = int(r['draft_year'])

        ras_str = f"{ras:.1f}" if isinstance(ras, float) else ras
        breakout_str = f"{breakout:.0f}" if isinstance(breakout, float) else breakout

        print(f"{name:<20} {pick:>6.0f} {ras_str:>8} {breakout_str:>10} {ppg:>10.2f} {hit24:>6} {draft_yr:>10}")

        if isinstance(ras, float) and pd.notna(ras):
            sample_data.append({'name': name, 'ras': ras, 'ppg': ppg})
    else:
        print(f"{name:<20} NOT FOUND IN DATASET")

# ============================================================================
# PART 2: MANUAL CORRELATION CALCULATION
# ============================================================================
print("\n" + "=" * 90)
print("PART 2: MANUAL CORRELATION CALCULATION (10 players with RAS)")
print("=" * 90)

# Filter to players with RAS
sample_df = pd.DataFrame(sample_data)
n = len(sample_df)

print(f"\nPlayers with RAS data: {n}")
print("\nRaw values:")
for i, row in sample_df.iterrows():
    print(f"  {row['name']}: RAS={row['ras']:.2f}, PPG={row['ppg']:.2f}")

# Manual calculation
x = sample_df['ras'].values
y = sample_df['ppg'].values

x_mean = np.mean(x)
y_mean = np.mean(y)

print(f"\nStep 1: Calculate means")
print(f"  Mean RAS (x̄) = {x_mean:.4f}")
print(f"  Mean PPG (ȳ) = {y_mean:.4f}")

print(f"\nStep 2: Calculate deviations from mean")
print(f"  {'Player':<20} {'RAS':>8} {'x-x̄':>10} {'PPG':>8} {'y-ȳ':>10} {'(x-x̄)(y-ȳ)':>12} {'(x-x̄)²':>10} {'(y-ȳ)²':>10}")
print("  " + "-" * 98)

sum_xy = 0
sum_xx = 0
sum_yy = 0

for i, row in sample_df.iterrows():
    xi = row['ras']
    yi = row['ppg']
    x_dev = xi - x_mean
    y_dev = yi - y_mean
    xy_prod = x_dev * y_dev
    xx_prod = x_dev ** 2
    yy_prod = y_dev ** 2

    sum_xy += xy_prod
    sum_xx += xx_prod
    sum_yy += yy_prod

    print(f"  {row['name']:<20} {xi:>8.2f} {x_dev:>+10.4f} {yi:>8.2f} {y_dev:>+10.4f} {xy_prod:>+12.4f} {xx_prod:>10.4f} {yy_prod:>10.4f}")

print("  " + "-" * 98)
print(f"  {'SUM':<20} {'':<8} {'':<10} {'':<8} {'':<10} {sum_xy:>+12.4f} {sum_xx:>10.4f} {sum_yy:>10.4f}")

print(f"\nStep 3: Calculate Pearson r")
print(f"  r = Σ(x-x̄)(y-ȳ) / √[Σ(x-x̄)² × Σ(y-ȳ)²]")
print(f"  r = {sum_xy:.4f} / √[{sum_xx:.4f} × {sum_yy:.4f}]")
print(f"  r = {sum_xy:.4f} / √[{sum_xx * sum_yy:.4f}]")
print(f"  r = {sum_xy:.4f} / {np.sqrt(sum_xx * sum_yy):.4f}")

r_manual = sum_xy / np.sqrt(sum_xx * sum_yy)
print(f"  r = {r_manual:.4f}")

# Verify with scipy
r_scipy, p_scipy = stats.pearsonr(x, y)
print(f"\nVerification with scipy.stats.pearsonr:")
print(f"  r = {r_scipy:.4f}, p = {p_scipy:.4f}")

# ============================================================================
# PART 3: FULL DATASET CORRELATION
# ============================================================================
print("\n" + "=" * 90)
print("PART 3: FULL DATASET CORRELATION")
print("=" * 90)

full_ras = wr[wr['RAS'].notna() & wr['best_ppg'].notna()]
n_full = len(full_ras)

r_full, p_full = stats.pearsonr(full_ras['RAS'], full_ras['best_ppg'])

print(f"\nFull dataset: n = {n_full} WRs with RAS data")
print(f"Pearson r = {r_full:.4f}")
print(f"p-value = {p_full:.4f}")

print(f"\nVerification - first 5 and last 5 data points:")
print(f"  {'Player':<25} {'RAS':>8} {'PPG':>8}")
print("  " + "-" * 43)
for i, (_, row) in enumerate(full_ras.head(5).iterrows()):
    print(f"  {row['player_name']:<25} {row['RAS']:>8.2f} {row['best_ppg']:>8.2f}")
print("  ...")
for i, (_, row) in enumerate(full_ras.tail(5).iterrows()):
    print(f"  {row['player_name']:<25} {row['RAS']:>8.2f} {row['best_ppg']:>8.2f}")

# ============================================================================
# PART 4: TRAIN/TEST SPLIT VERIFICATION
# ============================================================================
print("\n" + "=" * 90)
print("PART 4: TRAIN/TEST SPLIT VERIFICATION")
print("=" * 90)

train = wr[wr['draft_year'].between(2015, 2020)].copy()
test = wr[wr['draft_year'].between(2021, 2024)].copy()

print(f"\nTrain set (2015-2020):")
print(f"  Total WRs: {len(train)}")
print(f"  Draft years: {sorted(train['draft_year'].unique())}")
print(f"  WRs with RAS: {train['RAS'].notna().sum()}")

print(f"\nTest set (2021-2024):")
print(f"  Total WRs: {len(test)}")
print(f"  Draft years: {sorted(test['draft_year'].unique())}")
print(f"  WRs with RAS: {test['RAS'].notna().sum()}")

# Random 5 from each
print(f"\n5 Random WRs from TRAIN set:")
print(f"  {'Player':<25} {'Year':>6} {'Pick':>6} {'RAS':>8} {'PPG':>8} {'Hit24':>6}")
print("  " + "-" * 65)
train_sample = train[train['RAS'].notna()].sample(5, random_state=42)
for _, row in train_sample.iterrows():
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>6} {int(row['pick']):>6} {row['RAS']:>8.2f} {row['best_ppg']:>8.2f} {int(row['hit24']):>6}")

print(f"\n5 Random WRs from TEST set:")
print(f"  {'Player':<25} {'Year':>6} {'Pick':>6} {'RAS':>8} {'PPG':>8} {'Hit24':>6}")
print("  " + "-" * 65)
test_sample = test[test['RAS'].notna()].sample(5, random_state=42)
for _, row in test_sample.iterrows():
    print(f"  {row['player_name']:<25} {int(row['draft_year']):>6} {int(row['pick']):>6} {row['RAS']:>8.2f} {row['best_ppg']:>8.2f} {int(row['hit24']):>6}")

# Train/test correlations
train_ras = train[train['RAS'].notna() & train['best_ppg'].notna()]
test_ras = test[test['RAS'].notna() & test['best_ppg'].notna()]

r_train, p_train = stats.pearsonr(train_ras['RAS'], train_ras['best_ppg'])
r_test, p_test = stats.pearsonr(test_ras['RAS'], test_ras['best_ppg'])

print(f"\nCorrelations:")
print(f"  Train (n={len(train_ras)}): r = {r_train:.4f}, p = {p_train:.4f}")
print(f"  Test (n={len(test_ras)}):  r = {r_test:.4f}, p = {p_test:.4f}")

# ============================================================================
# PART 5: BONFERRONI CORRECTION - ALL 14 TESTS
# ============================================================================
print("\n" + "=" * 90)
print("PART 5: BONFERRONI CORRECTION - ALL 14 TESTS LISTED")
print("=" * 90)

from sklearn.linear_model import LinearRegression

# Collect all tests
all_tests = []

# Prepare data
wr['inv_sqrt_pick'] = 1 / np.sqrt(wr['pick'])
wr_ras = wr[wr['RAS'].notna()].copy()

# Load combine data for individual metrics
combine_url = "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.csv"
combine_df = pd.read_csv(combine_url)
combine_wr = combine_df[combine_df['pos'] == 'WR'].copy()
wr_combine = wr.merge(
    combine_wr[['player_name', 'forty', 'vertical', 'broad_jump', 'cone', 'shuttle', 'bench']],
    on='player_name', how='left'
)

print("\nTest 1: RAS vs PPG (overall)")
r1, p1 = stats.pearsonr(wr_ras['RAS'], wr_ras['best_ppg'])
print(f"  n = {len(wr_ras)}, r = {r1:.4f}, p = {p1:.6f}")
all_tests.append(('1. RAS vs PPG (overall)', p1))

print("\nTest 2: Elite vs Poor RAS (t-test)")
elite = wr_ras[wr_ras['RAS'] >= 9]['best_ppg']
poor = wr_ras[wr_ras['RAS'] < 5]['best_ppg']
t2, p2 = stats.ttest_ind(elite, poor)
print(f"  Elite n = {len(elite)}, Poor n = {len(poor)}")
print(f"  t = {t2:.4f}, p = {p2:.6f}")
all_tests.append(('2. Elite vs Poor RAS', p2))

print("\nTest 3: RAS vs PPG (Day 1 picks)")
day1 = wr_ras[wr_ras['pick'] <= 32]
r3, p3 = stats.pearsonr(day1['RAS'], day1['best_ppg'])
print(f"  n = {len(day1)}, r = {r3:.4f}, p = {p3:.6f}")
all_tests.append(('3. RAS vs PPG (Day 1)', p3))

print("\nTest 4: RAS vs PPG (Day 2 picks)")
day2 = wr_ras[(wr_ras['pick'] > 32) & (wr_ras['pick'] <= 100)]
r4, p4 = stats.pearsonr(day2['RAS'], day2['best_ppg'])
print(f"  n = {len(day2)}, r = {r4:.4f}, p = {p4:.6f}")
all_tests.append(('4. RAS vs PPG (Day 2)', p4))

print("\nTest 5: RAS vs PPG (Day 3 picks)")
day3 = wr_ras[wr_ras['pick'] > 100]
r5, p5 = stats.pearsonr(day3['RAS'], day3['best_ppg'])
print(f"  n = {len(day3)}, r = {r5:.4f}, p = {p5:.6f}")
all_tests.append(('5. RAS vs PPG (Day 3)', p5))

print("\nTest 6: 40 time vs PPG")
forty_data = wr_combine[wr_combine['forty'].notna() & wr_combine['best_ppg'].notna()]
r6, p6 = stats.pearsonr(forty_data['forty'], forty_data['best_ppg'])
print(f"  n = {len(forty_data)}, r = {-r6:.4f} (flipped), p = {p6:.6f}")
all_tests.append(('6. 40 time vs PPG', p6))

print("\nTest 7: Vertical vs PPG")
vert_data = wr_combine[wr_combine['vertical'].notna() & wr_combine['best_ppg'].notna()]
r7, p7 = stats.pearsonr(vert_data['vertical'], vert_data['best_ppg'])
print(f"  n = {len(vert_data)}, r = {r7:.4f}, p = {p7:.6f}")
all_tests.append(('7. Vertical vs PPG', p7))

print("\nTest 8: Broad jump vs PPG")
broad_data = wr_combine[wr_combine['broad_jump'].notna() & wr_combine['best_ppg'].notna()]
r8, p8 = stats.pearsonr(broad_data['broad_jump'], broad_data['best_ppg'])
print(f"  n = {len(broad_data)}, r = {r8:.4f}, p = {p8:.6f}")
all_tests.append(('8. Broad jump vs PPG', p8))

print("\nTest 9: 3-cone vs PPG")
cone_data = wr_combine[wr_combine['cone'].notna() & wr_combine['best_ppg'].notna()]
r9, p9 = stats.pearsonr(cone_data['cone'], cone_data['best_ppg'])
print(f"  n = {len(cone_data)}, r = {-r9:.4f} (flipped), p = {p9:.6f}")
all_tests.append(('9. 3-cone vs PPG', p9))

print("\nTest 10: Shuttle vs PPG")
shuttle_data = wr_combine[wr_combine['shuttle'].notna() & wr_combine['best_ppg'].notna()]
r10, p10 = stats.pearsonr(shuttle_data['shuttle'], shuttle_data['best_ppg'])
print(f"  n = {len(shuttle_data)}, r = {-r10:.4f} (flipped), p = {p10:.6f}")
all_tests.append(('10. Shuttle vs PPG', p10))

print("\nTest 11: Bench vs PPG")
bench_data = wr_combine[wr_combine['bench'].notna() & wr_combine['best_ppg'].notna()]
r11, p11 = stats.pearsonr(bench_data['bench'], bench_data['best_ppg'])
print(f"  n = {len(bench_data)}, r = {r11:.4f}, p = {p11:.6f}")
all_tests.append(('11. Bench vs PPG', p11))

print("\nTest 12: RAS residual vs PPG")
X_dc = wr_ras[['inv_sqrt_pick']].values
model = LinearRegression().fit(X_dc, wr_ras['RAS'].values)
wr_ras['ras_resid'] = wr_ras['RAS'] - model.predict(X_dc)
r12, p12 = stats.pearsonr(wr_ras['ras_resid'], wr_ras['best_ppg'])
print(f"  n = {len(wr_ras)}, r = {r12:.4f}, p = {p12:.6f}")
all_tests.append(('12. RAS residual vs PPG', p12))

print("\nTest 13: Young+HighRAS vs Young+LowRAS (Fisher's exact)")
wr_both = wr_ras[wr_ras['breakout_age'].notna()].copy()
young_high = wr_both[(wr_both['breakout_age'] <= 20) & (wr_both['RAS'] >= 8)]
young_low = wr_both[(wr_both['breakout_age'] <= 20) & (wr_both['RAS'] < 8)]
contingency = [
    [young_high['hit24'].sum(), len(young_high) - young_high['hit24'].sum()],
    [young_low['hit24'].sum(), len(young_low) - young_low['hit24'].sum()]
]
or13, p13 = stats.fisher_exact(contingency)
print(f"  Young+High n = {len(young_high)}, hits = {young_high['hit24'].sum()}")
print(f"  Young+Low n = {len(young_low)}, hits = {young_low['hit24'].sum()}")
print(f"  Odds Ratio = {or13:.4f}, p = {p13:.6f}")
all_tests.append(('13. Young+HighRAS vs Young+LowRAS', p13))

print("\nTest 14: Fast vs Slow 3-cone (Fisher's exact)")
median_cone = cone_data['cone'].median()
fast_cone = cone_data[cone_data['cone'] <= median_cone]
slow_cone = cone_data[cone_data['cone'] > median_cone]
contingency14 = [
    [int(fast_cone['hit24'].sum()), len(fast_cone) - int(fast_cone['hit24'].sum())],
    [int(slow_cone['hit24'].sum()), len(slow_cone) - int(slow_cone['hit24'].sum())]
]
or14, p14 = stats.fisher_exact(contingency14)
print(f"  Fast n = {len(fast_cone)}, hits = {int(fast_cone['hit24'].sum())}")
print(f"  Slow n = {len(slow_cone)}, hits = {int(slow_cone['hit24'].sum())}")
print(f"  Odds Ratio = {or14:.4f}, p = {p14:.6f}")
all_tests.append(('14. Fast vs Slow 3-cone', p14))

# Summary
print("\n" + "=" * 90)
print("BONFERRONI CORRECTION SUMMARY")
print("=" * 90)

n_tests = len(all_tests)
bonf_alpha = 0.05 / n_tests

print(f"\nNumber of tests: {n_tests}")
print(f"Bonferroni-corrected α = 0.05 / {n_tests} = {bonf_alpha:.6f}")

print("\n" + "-" * 70)
print(f"{'Test':<40} {'p-value':>12} {'p < 0.05':>10} {'p < {:.4f}'.format(bonf_alpha):>12}")
print("-" * 70)

sig_nominal = 0
sig_bonf = 0
for name, p in all_tests:
    nom = "YES" if p < 0.05 else "no"
    bonf = "YES" if p < bonf_alpha else "no"
    if p < 0.05:
        sig_nominal += 1
    if p < bonf_alpha:
        sig_bonf += 1
    print(f"{name:<40} {p:>12.6f} {nom:>10} {bonf:>12}")

print("-" * 70)
print(f"\nSignificant at α=0.05: {sig_nominal}/{n_tests}")
print(f"Significant after Bonferroni: {sig_bonf}/{n_tests}")

if sig_bonf == 0:
    print("\n⚠ ZERO tests survive Bonferroni correction")
