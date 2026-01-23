"""
ROUTE A: TRAIN MODEL END-TO-END WITH PROPER REGULARIZATION

Replace arbitrary 50/35/15 weights with data-driven coefficients.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("ROUTE A: END-TO-END MODEL WITH REGULARIZATION")
print("=" * 90)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("\n--- Loading data ---")

wr = pd.read_csv('data/wr_backtest_expanded_final.csv')
wr['best_ppg'] = wr['best_ppr'] / 17

# Exclude 2024 (incomplete outcomes)
wr = wr[wr['draft_year'] <= 2023].copy()
print(f"Loaded {len(wr)} WRs (2015-2023)")

# ============================================================================
# STEP 1: FEATURE PREPARATION
# ============================================================================
print("\n" + "=" * 90)
print("STEP 1: FEATURE PREPARATION")
print("=" * 90)

# Transform draft pick
wr['log_pick'] = np.log(wr['pick'])
wr['inv_sqrt_pick'] = 1 / np.sqrt(wr['pick'])

# Missing data indicators
wr['has_breakout'] = wr['breakout_age'].notna().astype(int)
wr['has_ras'] = wr['RAS'].notna().astype(int)

# Fill missing values with mean (for regression)
wr['breakout_age_filled'] = wr['breakout_age'].fillna(wr['breakout_age'].mean())
wr['ras_filled'] = wr['RAS'].fillna(wr['RAS'].mean())

# Define feature sets
FEATURES_FULL = ['inv_sqrt_pick', 'breakout_age_filled', 'has_breakout', 'ras_filled', 'has_ras']
FEATURES_DC_ONLY = ['inv_sqrt_pick']
FEATURES_DC_BREAKOUT = ['inv_sqrt_pick', 'breakout_age_filled', 'has_breakout']

print("\nFeature sets:")
print(f"  DC Only: {FEATURES_DC_ONLY}")
print(f"  DC + Breakout: {FEATURES_DC_BREAKOUT}")
print(f"  Full Model: {FEATURES_FULL}")

# Summary
print("\nFeature summary:")
for feat in FEATURES_FULL:
    n_valid = wr[feat].notna().sum()
    print(f"  {feat}: mean={wr[feat].mean():.3f}, std={wr[feat].std():.3f}, n={n_valid}")

# ============================================================================
# STEP 2: LEAVE-ONE-YEAR-OUT CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 90)
print("STEP 2: LEAVE-ONE-YEAR-OUT CROSS-VALIDATION")
print("=" * 90)

years = sorted(wr['draft_year'].unique())
print(f"\nYears for LOYO-CV: {years}")

def evaluate_model_loyo(wr_data, features, target, model_type='logistic'):
    """
    Leave-one-year-out cross-validation.
    Returns predictions and metrics for each held-out year.
    """
    results = []
    all_preds = []
    all_true = []
    all_years = []

    for test_year in years:
        # Split
        train = wr_data[wr_data['draft_year'] != test_year].copy()
        test = wr_data[wr_data['draft_year'] == test_year].copy()

        # Prepare features
        X_train = train[features].values
        X_test = test[features].values
        y_train = train[target].values
        y_test = test[target].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_type == 'logistic':
            # Logistic regression with L2 regularization
            model = LogisticRegressionCV(
                cv=5,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                scoring='neg_log_loss'
            )
            model.fit(X_train_scaled, y_train)
            preds = model.predict_proba(X_test_scaled)[:, 1]

            # Metrics
            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, preds)
                ll = log_loss(y_test, preds)
            else:
                auc = np.nan
                ll = np.nan

            results.append({
                'year': test_year,
                'n': len(test),
                'hits': y_test.sum(),
                'auc': auc,
                'log_loss': ll,
                'best_C': model.C_[0]
            })

        else:  # Ridge regression for PPG
            model = RidgeCV(
                alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                cv=5
            )
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

            # Metrics
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds) if len(y_test) > 1 else np.nan

            results.append({
                'year': test_year,
                'n': len(test),
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'best_alpha': model.alpha_
            })

        all_preds.extend(preds)
        all_true.extend(y_test)
        all_years.extend([test_year] * len(test))

    return results, np.array(all_preds), np.array(all_true), np.array(all_years)

# ============================================================================
# MODEL 1: HIT24 PROBABILITY MODEL
# ============================================================================
print("\n" + "=" * 90)
print("MODEL 1: HIT24 PROBABILITY (Logistic Regression)")
print("=" * 90)

print("\n--- DC Only Model ---")
results_dc_log, preds_dc_log, true_dc_log, years_dc_log = evaluate_model_loyo(
    wr, FEATURES_DC_ONLY, 'hit24', 'logistic'
)

print("\n--- Full Model ---")
results_full_log, preds_full_log, true_full_log, years_full_log = evaluate_model_loyo(
    wr, FEATURES_FULL, 'hit24', 'logistic'
)

# Display results by year
print("\n" + "-" * 80)
print("LOYO Results - Hit24 Probability")
print("-" * 80)
print(f"\n{'Year':<6} {'n':>4} {'Hits':>5} | {'DC AUC':>8} {'Full AUC':>9} | {'DC LL':>8} {'Full LL':>9}")
print("-" * 80)

for dc, full in zip(results_dc_log, results_full_log):
    dc_auc = f"{dc['auc']:.3f}" if not np.isnan(dc['auc']) else "N/A"
    full_auc = f"{full['auc']:.3f}" if not np.isnan(full['auc']) else "N/A"
    dc_ll = f"{dc['log_loss']:.3f}" if not np.isnan(dc['log_loss']) else "N/A"
    full_ll = f"{full['log_loss']:.3f}" if not np.isnan(full['log_loss']) else "N/A"
    print(f"{dc['year']:<6} {dc['n']:>4} {dc['hits']:>5} | {dc_auc:>8} {full_auc:>9} | {dc_ll:>8} {full_ll:>9}")

# Overall metrics
valid_mask = ~np.isnan(true_dc_log) & (true_dc_log >= 0)
overall_auc_dc = roc_auc_score(true_dc_log[valid_mask], preds_dc_log[valid_mask])
overall_auc_full = roc_auc_score(true_full_log[valid_mask], preds_full_log[valid_mask])
overall_ll_dc = log_loss(true_dc_log[valid_mask], preds_dc_log[valid_mask])
overall_ll_full = log_loss(true_full_log[valid_mask], preds_full_log[valid_mask])

print("-" * 80)
print(f"{'OVERALL':<6} {'':<4} {'':<5} | {overall_auc_dc:>8.3f} {overall_auc_full:>9.3f} | {overall_ll_dc:>8.3f} {overall_ll_full:>9.3f}")

# ============================================================================
# MODEL 2: PPG PREDICTION MODEL
# ============================================================================
print("\n" + "=" * 90)
print("MODEL 2: PPG PREDICTION (Ridge Regression)")
print("=" * 90)

print("\n--- DC Only Model ---")
results_dc_ppg, preds_dc_ppg, true_dc_ppg, years_dc_ppg = evaluate_model_loyo(
    wr, FEATURES_DC_ONLY, 'best_ppg', 'ridge'
)

print("\n--- Full Model ---")
results_full_ppg, preds_full_ppg, true_full_ppg, years_full_ppg = evaluate_model_loyo(
    wr, FEATURES_FULL, 'best_ppg', 'ridge'
)

# Display results by year
print("\n" + "-" * 80)
print("LOYO Results - PPG Prediction")
print("-" * 80)
print(f"\n{'Year':<6} {'n':>4} | {'DC RMSE':>9} {'Full RMSE':>10} | {'DC R²':>8} {'Full R²':>9}")
print("-" * 80)

for dc, full in zip(results_dc_ppg, results_full_ppg):
    dc_r2 = f"{dc['r2']:.3f}" if not np.isnan(dc['r2']) else "N/A"
    full_r2 = f"{full['r2']:.3f}" if not np.isnan(full['r2']) else "N/A"
    print(f"{dc['year']:<6} {dc['n']:>4} | {dc['rmse']:>9.2f} {full['rmse']:>10.2f} | {dc_r2:>8} {full_r2:>9}")

# Overall metrics
overall_rmse_dc = np.sqrt(mean_squared_error(true_dc_ppg, preds_dc_ppg))
overall_rmse_full = np.sqrt(mean_squared_error(true_full_ppg, preds_full_ppg))
overall_r2_dc = r2_score(true_dc_ppg, preds_dc_ppg)
overall_r2_full = r2_score(true_full_ppg, preds_full_ppg)

print("-" * 80)
print(f"{'OVERALL':<6} {'':<4} | {overall_rmse_dc:>9.2f} {overall_rmse_full:>10.2f} | {overall_r2_dc:>8.3f} {overall_r2_full:>9.3f}")

# ============================================================================
# STEP 3: TRAIN FINAL MODEL & GET COEFFICIENTS
# ============================================================================
print("\n" + "=" * 90)
print("STEP 3: LEARNED COEFFICIENTS (Full Dataset)")
print("=" * 90)

# Prepare full dataset
X_full = wr[FEATURES_FULL].values
y_hit24 = wr['hit24'].values
y_ppg = wr['best_ppg'].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# Train Hit24 model with different regularizers
print("\n--- Hit24 Model (Logistic with L1/L2/ElasticNet) ---\n")

# L2 (Ridge)
model_l2 = LogisticRegressionCV(cv=5, penalty='l2', solver='lbfgs', max_iter=1000)
model_l2.fit(X_scaled, y_hit24)

# L1 (Lasso)
model_l1 = LogisticRegressionCV(cv=5, penalty='l1', solver='saga', max_iter=1000)
model_l1.fit(X_scaled, y_hit24)

# ElasticNet
model_en = LogisticRegressionCV(cv=5, penalty='elasticnet', solver='saga', l1_ratios=[0.5], max_iter=1000)
model_en.fit(X_scaled, y_hit24)

print(f"{'Feature':<25} {'L2 (Ridge)':>12} {'L1 (Lasso)':>12} {'ElasticNet':>12}")
print("-" * 65)
for i, feat in enumerate(FEATURES_FULL):
    print(f"{feat:<25} {model_l2.coef_[0][i]:>+12.4f} {model_l1.coef_[0][i]:>+12.4f} {model_en.coef_[0][i]:>+12.4f}")
print(f"{'Intercept':<25} {model_l2.intercept_[0]:>+12.4f} {model_l1.intercept_[0]:>+12.4f} {model_en.intercept_[0]:>+12.4f}")

print(f"\nBest C (inverse regularization): L2={model_l2.C_[0]:.4f}, L1={model_l1.C_[0]:.4f}")

# Train PPG model
print("\n--- PPG Model (Ridge/Lasso/ElasticNet) ---\n")

# Ridge
model_ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
model_ridge.fit(X_scaled, y_ppg)

# Lasso
model_lasso = LassoCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5, max_iter=10000)
model_lasso.fit(X_scaled, y_ppg)

# ElasticNet
model_enet = ElasticNetCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], l1_ratio=[0.5], cv=5, max_iter=10000)
model_enet.fit(X_scaled, y_ppg)

print(f"{'Feature':<25} {'Ridge':>12} {'Lasso':>12} {'ElasticNet':>12}")
print("-" * 65)
for i, feat in enumerate(FEATURES_FULL):
    print(f"{feat:<25} {model_ridge.coef_[i]:>+12.4f} {model_lasso.coef_[i]:>+12.4f} {model_enet.coef_[i]:>+12.4f}")
print(f"{'Intercept':<25} {model_ridge.intercept_:>+12.4f} {model_lasso.intercept_:>+12.4f} {model_enet.intercept_:>+12.4f}")

print(f"\nBest alpha: Ridge={model_ridge.alpha_:.4f}, Lasso={model_lasso.alpha_:.4f}, ElasticNet={model_enet.alpha_:.4f}")

# ============================================================================
# STEP 4: FEATURE IMPORTANCE (WHICH SURVIVED REGULARIZATION?)
# ============================================================================
print("\n" + "=" * 90)
print("STEP 4: FEATURE IMPORTANCE")
print("=" * 90)

print("\nFeatures that survived L1 regularization (non-zero coefficients):")
print("\nHit24 Model (Lasso):")
for i, feat in enumerate(FEATURES_FULL):
    coef = model_l1.coef_[0][i]
    status = "✓ KEPT" if abs(coef) > 0.001 else "✗ ZEROED"
    print(f"  {feat:<25} {coef:>+10.4f} {status}")

print("\nPPG Model (Lasso):")
for i, feat in enumerate(FEATURES_FULL):
    coef = model_lasso.coef_[i]
    status = "✓ KEPT" if abs(coef) > 0.001 else "✗ ZEROED"
    print(f"  {feat:<25} {coef:>+10.4f} {status}")

# ============================================================================
# STEP 5: COMPARE TO BASELINES
# ============================================================================
print("\n" + "=" * 90)
print("STEP 5: COMPARISON TO BASELINES")
print("=" * 90)

# Baseline: predict base rate for everyone
base_rate = wr['hit24'].mean()
baseline_preds = np.full(len(wr), base_rate)
baseline_ll = log_loss(y_hit24, baseline_preds)

print("\n--- Hit24 Probability ---")
print(f"\nBaseline (predict base rate {base_rate:.1%} for all): Log Loss = {baseline_ll:.4f}")
print(f"DC Only Model:                                       Log Loss = {overall_ll_dc:.4f}")
print(f"Full Model:                                          Log Loss = {overall_ll_full:.4f}")
print(f"\nImprovement DC over baseline:   {(baseline_ll - overall_ll_dc)/baseline_ll*100:+.1f}%")
print(f"Improvement Full over DC:       {(overall_ll_dc - overall_ll_full)/overall_ll_dc*100:+.1f}%")

# For PPG
mean_ppg = wr['best_ppg'].mean()
baseline_ppg_preds = np.full(len(wr), mean_ppg)
baseline_rmse = np.sqrt(mean_squared_error(y_ppg, baseline_ppg_preds))

print("\n--- PPG Prediction ---")
print(f"\nBaseline (predict mean {mean_ppg:.1f} PPG for all): RMSE = {baseline_rmse:.2f}")
print(f"DC Only Model:                                  RMSE = {overall_rmse_dc:.2f}")
print(f"Full Model:                                     RMSE = {overall_rmse_full:.2f}")
print(f"\nImprovement DC over baseline:   {(baseline_rmse - overall_rmse_dc)/baseline_rmse*100:+.1f}%")
print(f"Improvement Full over DC:       {(overall_rmse_dc - overall_rmse_full)/overall_rmse_dc*100:+.1f}%")

# ============================================================================
# STEP 6: FINAL MODEL FORMULA
# ============================================================================
print("\n" + "=" * 90)
print("STEP 6: FINAL MODEL FORMULA (DATA-DRIVEN WEIGHTS)")
print("=" * 90)

# Use Ridge coefficients (more stable)
print("\n--- Hit24 Probability Model ---")
print("\nP(Hit24) = sigmoid(")
print(f"    {model_l2.coef_[0][0]:+.4f} × inv_sqrt_pick (scaled)")
print(f"    {model_l2.coef_[0][1]:+.4f} × breakout_age (scaled)")
print(f"    {model_l2.coef_[0][2]:+.4f} × has_breakout")
print(f"    {model_l2.coef_[0][3]:+.4f} × RAS (scaled)")
print(f"    {model_l2.coef_[0][4]:+.4f} × has_ras")
print(f"    {model_l2.intercept_[0]:+.4f}")
print(")")

# Relative importance (absolute value of standardized coefficients)
coef_abs = np.abs(model_l2.coef_[0])
coef_pct = coef_abs / coef_abs.sum() * 100

print("\nRelative Feature Importance (Hit24):")
for i, feat in enumerate(FEATURES_FULL):
    print(f"  {feat:<25} {coef_pct[i]:>5.1f}%")

print("\n--- PPG Prediction Model ---")
print("\nPPG = ")
print(f"    {model_ridge.coef_[0]:+.4f} × inv_sqrt_pick (scaled)")
print(f"    {model_ridge.coef_[1]:+.4f} × breakout_age (scaled)")
print(f"    {model_ridge.coef_[2]:+.4f} × has_breakout")
print(f"    {model_ridge.coef_[3]:+.4f} × RAS (scaled)")
print(f"    {model_ridge.coef_[4]:+.4f} × has_ras")
print(f"    {model_ridge.intercept_:+.4f}")

# Relative importance for PPG
coef_abs_ppg = np.abs(model_ridge.coef_)
coef_pct_ppg = coef_abs_ppg / coef_abs_ppg.sum() * 100

print("\nRelative Feature Importance (PPG):")
for i, feat in enumerate(FEATURES_FULL):
    print(f"  {feat:<25} {coef_pct_ppg[i]:>5.1f}%")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "=" * 90)
print("FINAL VERDICT: DOES THE FULL MODEL BEAT DC-ONLY?")
print("=" * 90)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                     OUT-OF-SAMPLE PERFORMANCE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HIT24 PROBABILITY (Log Loss, lower = better):                             │
│    Baseline (base rate):  {baseline_ll:.4f}                                        │
│    DC Only:               {overall_ll_dc:.4f}   ({(baseline_ll - overall_ll_dc)/baseline_ll*100:+.1f}% vs baseline)              │
│    Full Model:            {overall_ll_full:.4f}   ({(overall_ll_dc - overall_ll_full)/overall_ll_dc*100:+.1f}% vs DC)                    │
│                                                                             │
│  PPG PREDICTION (RMSE, lower = better):                                    │
│    Baseline (mean):       {baseline_rmse:.2f}                                          │
│    DC Only:               {overall_rmse_dc:.2f}   ({(baseline_rmse - overall_rmse_dc)/baseline_rmse*100:+.1f}% vs baseline)              │
│    Full Model:            {overall_rmse_full:.2f}   ({(overall_rmse_dc - overall_rmse_full)/overall_rmse_dc*100:+.1f}% vs DC)                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                     DATA-DRIVEN FEATURE WEIGHTS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Hit24 Model:                    PPG Model:                                 │
│    inv_sqrt_pick:   {coef_pct[0]:>5.1f}%         inv_sqrt_pick:   {coef_pct_ppg[0]:>5.1f}%              │
│    breakout_age:    {coef_pct[1]:>5.1f}%         breakout_age:    {coef_pct_ppg[1]:>5.1f}%              │
│    has_breakout:    {coef_pct[2]:>5.1f}%         has_breakout:    {coef_pct_ppg[2]:>5.1f}%              │
│    RAS:             {coef_pct[3]:>5.1f}%         RAS:             {coef_pct_ppg[3]:>5.1f}%              │
│    has_ras:         {coef_pct[4]:>5.1f}%         has_ras:         {coef_pct_ppg[4]:>5.1f}%              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

dc_wins_hit24 = overall_ll_dc < overall_ll_full
dc_wins_ppg = overall_rmse_dc < overall_rmse_full

if dc_wins_hit24 and dc_wins_ppg:
    print("CONCLUSION: DC-ONLY wins on BOTH metrics. The extra features hurt performance.")
elif dc_wins_hit24:
    print("CONCLUSION: DC-ONLY wins on Hit24, Full Model wins on PPG. Mixed results.")
elif dc_wins_ppg:
    print("CONCLUSION: Full Model wins on Hit24, DC-ONLY wins on PPG. Mixed results.")
else:
    print("CONCLUSION: Full Model wins on BOTH metrics. The extra features help!")

# What did Lasso keep?
print("\n" + "=" * 90)
print("FEATURES THAT SURVIVED LASSO REGULARIZATION:")
print("=" * 90)
kept_hit24 = [f for i, f in enumerate(FEATURES_FULL) if abs(model_l1.coef_[0][i]) > 0.001]
kept_ppg = [f for i, f in enumerate(FEATURES_FULL) if abs(model_lasso.coef_[i]) > 0.001]

print(f"\nHit24 Model: {kept_hit24}")
print(f"PPG Model:   {kept_ppg}")

if 'inv_sqrt_pick' in kept_hit24 and len(kept_hit24) == 1:
    print("\n→ Lasso chose ONLY draft capital for Hit24. Other features are noise.")
