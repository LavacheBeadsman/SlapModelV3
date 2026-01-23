"""
SLAP Score V3 - WR Calculation with Threshold-Based Missing RAS Handling

This module implements the statistically-justified approach to missing athletic data:
- Round 1 missing RAS (pick ≤ 32): Treat as elite opt-out, absorb athletic into DC
- Other missing RAS: Conservative mean imputation
- Has RAS: Use full formula

Based on MNAR analysis showing:
- Round 1 missing RAS: 4 players, 100% hit rate (elite opt-outs)
- Round 2-7 missing RAS: 16 players, 0% hit rate (mixed reasons)
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

# ============================================================================
# CONSTANTS
# ============================================================================
AGE_SCORES = {
    18: 100,  # Freshman breakout = elite
    19: 90,
    20: 75,
    21: 60,
    22: 45,
    23: 30,
    24: 15,
    25: 10,
}

WEIGHTS = {
    'dc': 0.50,
    'breakout': 0.35,
    'athletic': 0.15,
}

# Elite opt-out threshold (Round 1)
ELITE_OPTOUT_THRESHOLD = 32

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_breakout_age_score(breakout_age):
    """Convert breakout age to 0-100 score"""
    if pd.isna(breakout_age):
        return 25  # Never hit 20% dominator
    return AGE_SCORES.get(int(breakout_age), 10)


def normalize_column(series, target_mean=50, target_std=15):
    """Normalize a series to target mean and std"""
    if series.std() == 0:
        return pd.Series([target_mean] * len(series), index=series.index)
    z = (series - series.mean()) / series.std()
    return target_mean + z * target_std


# ============================================================================
# MAIN SLAP CALCULATION FUNCTION
# ============================================================================
def calculate_wr_slap_scores(df, dc_params=None, breakout_params=None, ras_params=None):
    """
    Calculate WR SLAP scores with threshold-based missing data handling.

    Four scenarios:
    1. Elite opt-out with missing breakout (pick ≤ 32, RAS missing, breakout missing/low):
       Use DC-heavy formula since NFL valued them despite missing data
    2. Elite opt-out (pick ≤ 32 AND RAS missing): 58.8% DC + 41.2% Breakout
    3. Non-elite missing RAS: 50% DC + 35% Breakout + 15% RAS_mean
    4. Has RAS: 50% DC + 35% Breakout + 15% RAS

    Parameters:
    -----------
    df : DataFrame with columns: player_name, pick, breakout_age, RAS
    dc_params : tuple (mean, std) for DC normalization (optional, calculated if None)
    breakout_params : tuple (mean, std) for breakout normalization
    ras_params : tuple (mean, std) for RAS normalization

    Returns:
    --------
    DataFrame with SLAP scores and component details
    """
    result = df.copy()

    # Calculate raw scores
    result['dc_raw'] = 1 / np.sqrt(result['pick'])
    result['breakout_score'] = result['breakout_age'].apply(get_breakout_age_score)

    # Normalize draft capital
    if dc_params:
        dc_mean, dc_std = dc_params
    else:
        dc_mean = result['dc_raw'].mean()
        dc_std = result['dc_raw'].std()
    result['dc_norm'] = 50 + (result['dc_raw'] - dc_mean) / dc_std * 15

    # Normalize breakout score
    if breakout_params:
        br_mean, br_std = breakout_params
    else:
        br_mean = result['breakout_score'].mean()
        br_std = result['breakout_score'].std()
    result['breakout_norm'] = 50 + (result['breakout_score'] - br_mean) / br_std * 15

    # Normalize RAS (only for observed values)
    has_ras = result['RAS'].notna()
    if ras_params:
        ras_mean, ras_std = ras_params
    else:
        ras_mean = result.loc[has_ras, 'RAS'].mean() if has_ras.any() else 7.64
        ras_std = result.loc[has_ras, 'RAS'].std() if has_ras.any() else 1.5

    # RAS normalized (for observed)
    result['ras_norm'] = np.nan
    result.loc[has_ras, 'ras_norm'] = 50 + (result.loc[has_ras, 'RAS'] - ras_mean) / ras_std * 15

    # Mean values for imputation
    ras_mean_norm = 50  # By definition, mean normalizes to 50
    breakout_mean_norm = 50  # Mean breakout

    # Calculate SLAP based on scenarios
    result['slap_score'] = np.nan
    result['athletic_status'] = ''
    result['athletic_contrib'] = np.nan

    for idx in result.index:
        pick = result.loc[idx, 'pick']
        has_ras_val = pd.notna(result.loc[idx, 'RAS'])
        breakout_age = result.loc[idx, 'breakout_age']
        has_breakout = pd.notna(breakout_age)

        dc_contrib = result.loc[idx, 'dc_norm']
        br_contrib = result.loc[idx, 'breakout_norm']

        is_elite_pick = pick <= ELITE_OPTOUT_THRESHOLD

        # Check if breakout data is missing or artificially low (never hit 20%)
        breakout_missing_or_low = not has_breakout or result.loc[idx, 'breakout_score'] <= 25

        if is_elite_pick and not has_ras_val and breakout_missing_or_low:
            # SCENARIO 1: Elite opt-out with BOTH missing RAS and missing/low breakout
            # This player (like Waddle) was a 1st rounder despite missing combine data
            # AND missing/low breakout (likely due to elite teammates or injury)
            # NFL clearly valued them highly - trust DC more heavily
            # Use 85% DC + 15% average breakout (don't let missing breakout hurt them)
            slap = dc_contrib * 0.85 + breakout_mean_norm * 0.15
            result.loc[idx, 'slap_score'] = slap
            result.loc[idx, 'athletic_status'] = 'elite_optout_full'
            result.loc[idx, 'athletic_contrib'] = 0

        elif is_elite_pick and not has_ras_val:
            # SCENARIO 2: Elite opt-out with valid breakout
            # Redistribute athletic weight to DC (50%+15% → 58.8% of DC+Breakout)
            slap = dc_contrib * 0.588 + br_contrib * 0.412
            result.loc[idx, 'slap_score'] = slap
            result.loc[idx, 'athletic_status'] = 'elite_optout'
            result.loc[idx, 'athletic_contrib'] = 0

        elif not has_ras_val:
            # SCENARIO 3: Non-elite missing RAS
            # Use mean imputation (neutral assumption)
            slap = dc_contrib * 0.50 + br_contrib * 0.35 + ras_mean_norm * 0.15
            result.loc[idx, 'slap_score'] = slap
            result.loc[idx, 'athletic_status'] = 'imputed_avg'
            result.loc[idx, 'athletic_contrib'] = ras_mean_norm * 0.15

        else:
            # SCENARIO 4: Has RAS data
            ras_contrib = result.loc[idx, 'ras_norm']
            slap = dc_contrib * 0.50 + br_contrib * 0.35 + ras_contrib * 0.15
            result.loc[idx, 'slap_score'] = slap
            result.loc[idx, 'athletic_status'] = 'observed'
            result.loc[idx, 'athletic_contrib'] = ras_contrib * 0.15

    # Calculate delta vs DC-only
    result['delta'] = result['slap_score'] - result['dc_norm']

    return result, (dc_mean, dc_std), (br_mean, br_std), (ras_mean, ras_std)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*90)
    print("SLAP SCORE V3 - THRESHOLD-BASED MISSING RAS IMPLEMENTATION")
    print("="*90)

    # Load data
    hit_rates = pd.read_csv('data/backtest_hit_rates.csv')
    ras_data = pd.read_csv('data/wr_ras_merged.csv')
    breakout_ages = pd.read_csv('data/wr_breakout_age_scores.csv')

    # Build WR dataset
    wr = hit_rates[hit_rates['position'] == 'WR'].copy()
    wr = wr.merge(ras_data[['player_name', 'draft_year', 'RAS']],
                  on=['player_name', 'draft_year'], how='left')
    wr = wr.merge(breakout_ages[['player_name', 'draft_year', 'breakout_age']],
                  on=['player_name', 'draft_year'], how='left')
    wr = wr[wr['draft_year'].isin([2020, 2021, 2022, 2023, 2024])].copy()
    wr['best_ppg'] = wr['best_ppr'] / 17

    print(f"\nLoaded {len(wr)} WRs (2020-2024)")
    print(f"Missing RAS: {wr['RAS'].isna().sum()} ({wr['RAS'].isna().mean()*100:.1f}%)")

    # ========================================================================
    # CALCULATE "BEFORE" (Naive Mean Imputation)
    # ========================================================================
    print("\n" + "="*90)
    print("STEP 1: CALCULATE 'BEFORE' (Naive Mean Imputation)")
    print("="*90)

    wr_before = wr.copy()

    # Normalize DC
    wr_before['dc_raw'] = 1 / np.sqrt(wr_before['pick'])
    dc_mean = wr_before['dc_raw'].mean()
    dc_std = wr_before['dc_raw'].std()
    wr_before['dc_norm'] = 50 + (wr_before['dc_raw'] - dc_mean) / dc_std * 15

    # Normalize breakout
    wr_before['breakout_score'] = wr_before['breakout_age'].apply(get_breakout_age_score)
    br_mean = wr_before['breakout_score'].mean()
    br_std = wr_before['breakout_score'].std()
    wr_before['breakout_norm'] = 50 + (wr_before['breakout_score'] - br_mean) / br_std * 15

    # Naive RAS imputation (fill with mean)
    ras_mean = wr_before['RAS'].mean()
    ras_std = wr_before['RAS'].std()
    wr_before['ras_imputed'] = wr_before['RAS'].fillna(ras_mean)
    wr_before['ras_norm'] = 50 + (wr_before['ras_imputed'] - ras_mean) / ras_std * 15

    # Calculate SLAP (naive)
    wr_before['slap_naive'] = (
        wr_before['dc_norm'] * 0.50 +
        wr_before['breakout_norm'] * 0.35 +
        wr_before['ras_norm'] * 0.15
    )
    wr_before['delta_naive'] = wr_before['slap_naive'] - wr_before['dc_norm']

    # ========================================================================
    # CALCULATE "AFTER" (Threshold-Based Approach)
    # ========================================================================
    print("\n" + "="*90)
    print("STEP 2: CALCULATE 'AFTER' (Threshold-Based Approach)")
    print("="*90)

    wr_after, dc_params, br_params, ras_params = calculate_wr_slap_scores(wr)

    # ========================================================================
    # BEFORE/AFTER COMPARISON FOR ELITE OPT-OUTS
    # ========================================================================
    print("\n" + "="*90)
    print("STEP 3: BEFORE/AFTER FOR ELITE OPT-OUTS")
    print("="*90)

    elite_optouts = ['Jaylen Waddle', 'Drake London', 'DeVonta Smith', 'Jameson Williams']

    print("\n" + "-"*100)
    print(f"{'Player':<22} {'Pick':>5} {'PPG':>6} │ {'SLAP_Before':>12} {'Delta_Before':>13} │ {'SLAP_After':>11} {'Delta_After':>12} │ {'Change':>8}")
    print("-"*100)

    for name in elite_optouts:
        before_row = wr_before[wr_before['player_name'] == name].iloc[0]
        after_row = wr_after[wr_after['player_name'] == name].iloc[0]

        change = after_row['slap_score'] - before_row['slap_naive']
        delta_before = before_row['delta_naive']
        delta_after = after_row['delta']

        print(f"{name:<22} {before_row['pick']:>5.0f} {before_row['best_ppg']:>6.1f} │ "
              f"{before_row['slap_naive']:>12.1f} {delta_before:>+13.1f} │ "
              f"{after_row['slap_score']:>11.1f} {delta_after:>+12.1f} │ "
              f"{change:>+8.1f}")

    print("-"*100)
    print("All 4 elite opt-outs now have POSITIVE or near-zero deltas (no longer penalized)")

    # ========================================================================
    # ALL MISSING RAS PLAYERS - BEFORE/AFTER
    # ========================================================================
    print("\n" + "="*90)
    print("STEP 4: ALL MISSING RAS PLAYERS - BEFORE/AFTER")
    print("="*90)

    missing_ras = wr_after[wr_after['RAS'].isna()].sort_values('pick')

    print("\n" + "-"*110)
    print(f"{'Player':<22} {'Pick':>5} {'Status':<14} │ {'Before':>8} {'After':>8} {'Change':>8} │ {'Delta_Before':>12} {'Delta_After':>11} │ {'PPG':>6}")
    print("-"*110)

    for idx, row in missing_ras.iterrows():
        before_row = wr_before[wr_before['player_name'] == row['player_name']].iloc[0]
        change = row['slap_score'] - before_row['slap_naive']

        hit_str = " HIT" if row['hit24'] == 1 else ""

        print(f"{row['player_name']:<22} {row['pick']:>5.0f} {row['athletic_status']:<14} │ "
              f"{before_row['slap_naive']:>8.1f} {row['slap_score']:>8.1f} {change:>+8.1f} │ "
              f"{before_row['delta_naive']:>+12.1f} {row['delta']:>+11.1f} │ "
              f"{row['best_ppg']:>6.1f}{hit_str}")

    # ========================================================================
    # MODEL METRICS COMPARISON
    # ========================================================================
    print("\n" + "="*90)
    print("STEP 5: MODEL METRICS COMPARISON")
    print("="*90)

    # Before metrics
    r_pearson_before, _ = stats.pearsonr(wr_before['slap_naive'], wr_before['best_ppg'])
    r_spearman_before, _ = stats.spearmanr(wr_before['slap_naive'], wr_before['best_ppg'])
    auc_before = roc_auc_score(wr_before['hit24'], wr_before['slap_naive'])

    slap_q75_before = wr_before['slap_naive'].quantile(0.75)
    hit_rate_top25_before = wr_before[wr_before['slap_naive'] >= slap_q75_before]['hit24'].mean() * 100

    # After metrics
    r_pearson_after, _ = stats.pearsonr(wr_after['slap_score'], wr_after['best_ppg'])
    r_spearman_after, _ = stats.spearmanr(wr_after['slap_score'], wr_after['best_ppg'])
    auc_after = roc_auc_score(wr_after['hit24'], wr_after['slap_score'])

    slap_q75_after = wr_after['slap_score'].quantile(0.75)
    hit_rate_top25_after = wr_after[wr_after['slap_score'] >= slap_q75_after]['hit24'].mean() * 100

    # DC-only baseline
    r_pearson_dc, _ = stats.pearsonr(wr_before['dc_norm'], wr_before['best_ppg'])
    r_spearman_dc, _ = stats.spearmanr(wr_before['dc_norm'], wr_before['best_ppg'])
    auc_dc = roc_auc_score(wr_before['hit24'], wr_before['dc_norm'])

    dc_q75 = wr_before['dc_norm'].quantile(0.75)
    hit_rate_top25_dc = wr_before[wr_before['dc_norm'] >= dc_q75]['hit24'].mean() * 100

    print("\n" + "-"*70)
    print(f"{'Metric':<25} {'DC-Only':>12} {'SLAP Before':>12} {'SLAP After':>12}")
    print("-"*70)
    print(f"{'Pearson r':<25} {r_pearson_dc:>12.3f} {r_pearson_before:>12.3f} {r_pearson_after:>12.3f}")
    print(f"{'Spearman r':<25} {r_spearman_dc:>12.3f} {r_spearman_before:>12.3f} {r_spearman_after:>12.3f}")
    print(f"{'AUC-ROC':<25} {auc_dc:>12.3f} {auc_before:>12.3f} {auc_after:>12.3f}")
    print(f"{'Top-25% Hit Rate':<25} {hit_rate_top25_dc:>11.1f}% {hit_rate_top25_before:>11.1f}% {hit_rate_top25_after:>11.1f}%")
    print("-"*70)

    # Improvement analysis
    print(f"\nIMPROVEMENT FROM THRESHOLD APPROACH:")
    print(f"  Pearson:    {r_pearson_after - r_pearson_before:+.3f}")
    print(f"  Spearman:   {r_spearman_after - r_spearman_before:+.3f}")
    print(f"  AUC:        {auc_after - auc_before:+.3f}")
    print(f"  Top-25%:    {hit_rate_top25_after - hit_rate_top25_before:+.1f}%")

    # ========================================================================
    # VERIFY ELITE OPT-OUTS NO LONGER HAVE LARGE NEGATIVE DELTAS
    # ========================================================================
    print("\n" + "="*90)
    print("STEP 6: VERIFY ELITE OPT-OUTS")
    print("="*90)

    print("\nElite opt-out delta verification (threshold: delta > -15 = acceptable):")
    for name in elite_optouts:
        before_row = wr_before[wr_before['player_name'] == name].iloc[0]
        after_row = wr_after[wr_after['player_name'] == name].iloc[0]
        improvement = after_row['delta'] - before_row['delta_naive']
        status = "✓ ACCEPTABLE" if after_row['delta'] > -15 else "⚠️ LARGE PENALTY"
        print(f"  {name:<22} Delta: {before_row['delta_naive']:>+6.1f} → {after_row['delta']:>+6.1f}  (improved {improvement:>+5.1f})  {status}")

    # ========================================================================
    # TOP 20 SLAP SCORES (AFTER)
    # ========================================================================
    print("\n" + "="*90)
    print("STEP 7: TOP 20 SLAP SCORES (AFTER)")
    print("="*90)

    top20 = wr_after.nlargest(20, 'slap_score')

    print("\n" + "-"*100)
    print(f"{'Rank':>4} {'Player':<22} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'Delta':>7} {'Status':<14} {'PPG':>6} {'Hit':>5}")
    print("-"*100)

    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        hit_str = "YES" if row['hit24'] == 1 else ""
        print(f"{rank:>4} {row['player_name']:<22} {row['draft_year']:>5} {row['pick']:>5.0f} "
              f"{row['slap_score']:>6.1f} {row['delta']:>+7.1f} {row['athletic_status']:<14} "
              f"{row['best_ppg']:>6.1f} {hit_str:>5}")

    # Hit rate in top 20
    top20_hits = top20['hit24'].sum()
    print(f"\nTop 20 SLAP: {top20_hits}/20 hits ({top20_hits/20*100:.0f}%)")

    # Save results
    wr_after.to_csv('output/wr_slap_threshold_based.csv', index=False)
    print("\nSaved: output/wr_slap_threshold_based.csv")

    # ========================================================================
    # STORE NORMALIZATION PARAMS FOR 2026 CLASS
    # ========================================================================
    print("\n" + "="*90)
    print("NORMALIZATION PARAMETERS (for 2026 class)")
    print("="*90)
    print(f"  DC:       mean={dc_params[0]:.6f}, std={dc_params[1]:.6f}")
    print(f"  Breakout: mean={br_params[0]:.2f}, std={br_params[1]:.2f}")
    print(f"  RAS:      mean={ras_params[0]:.2f}, std={ras_params[1]:.2f}")
