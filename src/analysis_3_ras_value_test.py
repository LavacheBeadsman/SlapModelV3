"""
Analysis Test 3 of 5: RAS Value Test
=====================================
Tests whether RAS (Relative Athletic Score) adds predictive value to the SLAP model,
or if we'd be better off with just DC + Production.

Currently RAS gets 15% weight in both WR and RB formulas.

Models tested:
  A: Current weights (with RAS)
  B: Without RAS (redistribute to DC)
  C: Without RAS (redistribute proportionally)
  D: DC only
  E: RAS alone
  F: Model A on observed-RAS players only
  G: Model A on imputed-RAS players only
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================

outcomes = pd.read_csv('/home/user/SlapModelV3/data/backtest_outcomes_complete.csv')
wr_data = pd.read_csv('/home/user/SlapModelV3/data/wr_backtest_expanded_final.csv')
rb_data = pd.read_csv('/home/user/SlapModelV3/data/rb_backtest_with_receiving.csv')

# =============================================================================
# FORMULA FUNCTIONS
# =============================================================================

def normalize_draft_capital(pick):
    """DC score from pick number using the gentler curve formula."""
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))


def wr_breakout_score(breakout_age, dominator_pct):
    """WR production score based on breakout age + dominator tiebreaker."""
    if breakout_age is None or pd.isna(breakout_age):
        if dominator_pct is not None and pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        else:
            return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base_score = age_tiers.get(int(breakout_age), 20)
    if dominator_pct is not None and pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
        bonus = max(0, bonus)
    else:
        bonus = 0
    return min(base_score + bonus, 99.9)


def rb_production_score(rec_yards, team_pass_att, draft_age):
    """RB production score based on receiving yards / team pass attempts."""
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return None
    if pd.isna(draft_age):
        draft_age = 22
    ratio = rec_yards / team_pass_att
    season_age = draft_age - 1
    age_weight = 1.15 - (0.05 * (season_age - 19))
    age_weight = max(0.85, min(1.15, age_weight))
    raw_score = ratio * age_weight * 100
    return min(99.9, max(0, raw_score / 1.75))


# =============================================================================
# MERGE DATA WITH OUTCOMES
# =============================================================================

# --- WRs ---
wr_merged = pd.merge(
    wr_data[['player_name', 'draft_year', 'pick', 'RAS', 'breakout_age', 'peak_dominator']],
    outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year', 'pick',
        'seasons_played', 'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']],
    on=['player_name', 'draft_year'],
    how='inner',
    suffixes=('', '_out')
)
# Use pick from WR data primarily
wr_merged['pick_final'] = wr_merged['pick'].fillna(wr_merged['pick_out'])
print(f"WR merged: {len(wr_merged)} players")

# --- RBs ---
rb_merged = pd.merge(
    rb_data[['player_name', 'draft_year', 'pick', 'age', 'RAS', 'rec_yards', 'team_pass_att']],
    outcomes[outcomes['position'] == 'RB'][['player_name', 'draft_year', 'pick',
        'seasons_played', 'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']],
    on=['player_name', 'draft_year'],
    how='inner',
    suffixes=('', '_out')
)
rb_merged['pick_final'] = rb_merged['pick'].fillna(rb_merged['pick_out'])
print(f"RB merged: {len(rb_merged)} players")

# =============================================================================
# COMPUTE COMPONENT SCORES
# =============================================================================

# WR components
wr_merged['dc_score'] = wr_merged['pick_final'].apply(normalize_draft_capital)
wr_merged['breakout_score'] = wr_merged.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)

WR_AVG_RAS = 68.9  # Position average for imputation (from CLAUDE.md)
wr_merged['ras_observed'] = wr_merged['RAS'].notna()
wr_merged['ras_imputed'] = wr_merged['RAS'].isna()
wr_merged['ras_score'] = wr_merged['RAS'].apply(lambda x: x * 10 if pd.notna(x) else WR_AVG_RAS)

# RB components
rb_merged['dc_score'] = rb_merged['pick_final'].apply(normalize_draft_capital)
rb_merged['prod_score'] = rb_merged.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)

RB_AVG_RAS = 66.5  # Position average for imputation (from CLAUDE.md)
rb_merged['ras_observed'] = rb_merged['RAS'].notna()
rb_merged['ras_imputed'] = rb_merged['RAS'].isna()
rb_merged['ras_score'] = rb_merged['RAS'].apply(lambda x: x * 10 if pd.notna(x) else RB_AVG_RAS)

# Drop RBs with missing production (can't compute SLAP)
rb_valid = rb_merged[rb_merged['prod_score'].notna()].copy()
print(f"RB with valid production: {len(rb_valid)} of {len(rb_merged)}")

# =============================================================================
# COMPUTE MODEL SCORES
# =============================================================================

# --- Model A: Current weights (with RAS) ---
wr_merged['slap_A'] = 0.65 * wr_merged['dc_score'] + 0.20 * wr_merged['breakout_score'] + 0.15 * wr_merged['ras_score']
rb_valid['slap_A'] = 0.50 * rb_valid['dc_score'] + 0.35 * rb_valid['prod_score'] + 0.15 * rb_valid['ras_score']

# --- Model B: Without RAS (redistribute weight to DC) ---
wr_merged['slap_B'] = 0.765 * wr_merged['dc_score'] + 0.235 * wr_merged['breakout_score']
rb_valid['slap_B'] = 0.588 * rb_valid['dc_score'] + 0.412 * rb_valid['prod_score']

# --- Model C: Without RAS (redistribute proportionally) ---
wr_merged['slap_C'] = 0.7647 * wr_merged['dc_score'] + 0.2353 * wr_merged['breakout_score']
rb_valid['slap_C'] = 0.5882 * rb_valid['dc_score'] + 0.4118 * rb_valid['prod_score']

# --- Model D: DC only ---
wr_merged['slap_D'] = wr_merged['dc_score']
rb_valid['slap_D'] = rb_valid['dc_score']

# --- Model E: RAS alone ---
wr_merged['slap_E'] = wr_merged['ras_score']
rb_valid['slap_E'] = rb_valid['ras_score']


# =============================================================================
# CORRELATION HELPER
# =============================================================================

OUTCOMES = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']

def compute_correlations(df, score_col, label, outcomes=OUTCOMES):
    """Compute Pearson r and p-value for a score column against each outcome."""
    results = {}
    for outcome in outcomes:
        valid = df[[score_col, outcome]].dropna()
        if len(valid) < 5:
            results[outcome] = {'r': np.nan, 'p': np.nan, 'n': len(valid)}
            continue
        r, p = stats.pearsonr(valid[score_col], valid[outcome])
        results[outcome] = {'r': r, 'p': p, 'n': len(valid)}
    return results


def print_model_table(all_results, position):
    """Pretty-print a table of model results."""
    print(f"\n{'='*120}")
    print(f"  {position} MODEL COMPARISON")
    print(f"{'='*120}")
    header = f"{'Model':<45} | {'N':>4} | {'hit24 r':>8} {'p':>8} | {'hit12 r':>8} {'p':>8} | {'3yr_ppg r':>9} {'p':>8} | {'career r':>9} {'p':>8}"
    print(header)
    print('-' * 120)
    for model_name, res in all_results.items():
        # Get N from first outcome that has data
        n = next((v['n'] for v in res.values() if not np.isnan(v.get('n', np.nan))), 0)
        h24 = res.get('hit24', {})
        h12 = res.get('hit12', {})
        ppg3 = res.get('first_3yr_ppg', {})
        cppg = res.get('career_ppg', {})
        
        def fmt(d):
            r_val = d.get('r', np.nan)
            p_val = d.get('p', np.nan)
            if np.isnan(r_val):
                return '     N/A      N/A'
            star = '***' if p_val < 0.001 else '** ' if p_val < 0.01 else '*  ' if p_val < 0.05 else '   '
            return f'{r_val:>8.3f} {p_val:>7.4f}{star}'
        
        print(f"{model_name:<45} | {n:>4} | {fmt(h24)} | {fmt(h12)} | {fmt(ppg3)} | {fmt(cppg)}")
    print()


# =============================================================================
# RUN ALL MODELS
# =============================================================================

print("\n" + "=" * 120)
print("  ANALYSIS 3: RAS VALUE TEST")
print("  Does RAS (Relative Athletic Score) add predictive value to SLAP?")
print("=" * 120)

# --- WR Models ---
wr_results = {}
for model, label in [
    ('slap_A', 'Model A: Current (65/20/15 with RAS)'),
    ('slap_B', 'Model B: No RAS (76.5/23.5/0)'),
    ('slap_C', 'Model C: No RAS proportional (76.47/23.53/0)'),
    ('slap_D', 'Model D: DC only (100/0/0)'),
    ('slap_E', 'Model E: RAS alone'),
]:
    wr_results[label] = compute_correlations(wr_merged, model, label)

# Model F: Observed RAS only
wr_obs = wr_merged[wr_merged['ras_observed']].copy()
wr_results[f'Model F: Observed RAS only (n={len(wr_obs)})'] = compute_correlations(wr_obs, 'slap_A', 'F')

# Model G: Imputed RAS only
wr_imp = wr_merged[wr_merged['ras_imputed']].copy()
wr_results[f'Model G: Imputed RAS only (n={len(wr_imp)})'] = compute_correlations(wr_imp, 'slap_A', 'G')

print_model_table(wr_results, "WR")

# --- RB Models ---
rb_results = {}
for model, label in [
    ('slap_A', 'Model A: Current (50/35/15 with RAS)'),
    ('slap_B', 'Model B: No RAS (58.8/41.2/0)'),
    ('slap_C', 'Model C: No RAS proportional (58.82/41.18/0)'),
    ('slap_D', 'Model D: DC only (100/0/0)'),
    ('slap_E', 'Model E: RAS alone'),
]:
    rb_results[label] = compute_correlations(rb_valid, model, label)

# Model F: Observed RAS only
rb_obs = rb_valid[rb_valid['ras_observed']].copy()
rb_results[f'Model F: Observed RAS only (n={len(rb_obs)})'] = compute_correlations(rb_obs, 'slap_A', 'F')

# Model G: Imputed RAS only
rb_imp = rb_valid[rb_valid['ras_imputed']].copy()
rb_results[f'Model G: Imputed RAS only (n={len(rb_imp)})'] = compute_correlations(rb_imp, 'slap_A', 'G')

print_model_table(rb_results, "RB")

# =============================================================================
# DELTA ANALYSIS: How much does RAS change scores?
# =============================================================================

print("\n" + "=" * 120)
print("  MODEL A vs MODEL B: SCORE DIFFERENCES (effect of including RAS)")
print("=" * 120)

for pos_label, df in [("WR", wr_merged), ("RB", rb_valid)]:
    df['delta_AB'] = df['slap_A'] - df['slap_B']
    print(f"\n--- {pos_label} ---")
    print(f"  Mean absolute delta (A vs B):  {df['delta_AB'].abs().mean():.2f}")
    print(f"  Max delta:                     {df['delta_AB'].max():.2f}")
    print(f"  Min delta:                     {df['delta_AB'].min():.2f}")
    print(f"  Std dev of delta:              {df['delta_AB'].std():.2f}")
    # Show top 5 players most helped/hurt by RAS
    top_helped = df.nlargest(5, 'delta_AB')[['player_name', 'draft_year', 'pick_final', 'ras_score', 'slap_A', 'slap_B', 'delta_AB', 'hit24', 'career_ppg']]
    top_hurt = df.nsmallest(5, 'delta_AB')[['player_name', 'draft_year', 'pick_final', 'ras_score', 'slap_A', 'slap_B', 'delta_AB', 'hit24', 'career_ppg']]
    print(f"\n  Top 5 players HELPED by RAS inclusion:")
    for _, row in top_helped.iterrows():
        ppg_str = f"{row['career_ppg']:.1f}" if pd.notna(row['career_ppg']) else "N/A"
        print(f"    {row['player_name']:<25} ({int(row['draft_year'])}) pick {int(row['pick_final']):>3} | RAS={row['ras_score']:>5.1f} | A={row['slap_A']:.1f} B={row['slap_B']:.1f} delta={row['delta_AB']:+.1f} | hit24={int(row['hit24'])} career_ppg={ppg_str}")
    print(f"\n  Top 5 players HURT by RAS inclusion:")
    for _, row in top_hurt.iterrows():
        ppg_str = f"{row['career_ppg']:.1f}" if pd.notna(row['career_ppg']) else "N/A"
        print(f"    {row['player_name']:<25} ({int(row['draft_year'])}) pick {int(row['pick_final']):>3} | RAS={row['ras_score']:>5.1f} | A={row['slap_A']:.1f} B={row['slap_B']:.1f} delta={row['delta_AB']:+.1f} | hit24={int(row['hit24'])} career_ppg={ppg_str}")


# =============================================================================
# RAS TIER BREAKDOWN
# =============================================================================

def assign_ras_tier(row):
    """Assign RAS tier label based on raw RAS value and observation status."""
    if row['ras_imputed']:
        return '6_Missing/Imputed'
    ras = row['RAS']
    if ras < 3:
        return '1_Poor (0-3)'
    elif ras < 5:
        return '2_Below Avg (3-5)'
    elif ras < 7:
        return '3_Avg (5-7)'
    elif ras < 9:
        return '4_Good (7-9)'
    else:
        return '5_Elite (9-10)'


print("\n" + "=" * 120)
print("  RAS TIER BREAKDOWN BY POSITION")
print("=" * 120)

for pos_label, df in [("WR", wr_merged), ("RB", rb_valid)]:
    df['ras_tier'] = df.apply(assign_ras_tier, axis=1)
    
    print(f"\n{'='*115}")
    print(f"  {pos_label} RAS TIER BREAKDOWN")
    print(f"{'='*115}")
    header = f"{'Tier':<25} | {'N':>4} | {'Hit24%':>7} | {'Hit12%':>7} | {'Mean 3yr PPG':>13} | {'Mean Career PPG':>16} | {'Seasons Range':>15}"
    print(header)
    print('-' * 115)
    
    for tier in sorted(df['ras_tier'].unique()):
        subset = df[df['ras_tier'] == tier]
        n = len(subset)
        hit24_rate = subset['hit24'].mean() * 100
        hit12_rate = subset['hit12'].mean() * 100
        ppg3 = subset['first_3yr_ppg'].mean()
        cppg = subset['career_ppg'].mean()
        sp_min = subset['seasons_played'].min()
        sp_max = subset['seasons_played'].max()
        
        ppg3_str = f"{ppg3:.2f}" if pd.notna(ppg3) else "N/A"
        cppg_str = f"{cppg:.2f}" if pd.notna(cppg) else "N/A"
        
        print(f"{tier:<25} | {n:>4} | {hit24_rate:>6.1f}% | {hit12_rate:>6.1f}% | {ppg3_str:>13} | {cppg_str:>16} | {int(sp_min):>3} - {int(sp_max):>3}")
    
    # Overall row
    n = len(df)
    print('-' * 115)
    hit24_rate = df['hit24'].mean() * 100
    hit12_rate = df['hit12'].mean() * 100
    ppg3 = df['first_3yr_ppg'].mean()
    cppg = df['career_ppg'].mean()
    sp_min = df['seasons_played'].min()
    sp_max = df['seasons_played'].max()
    ppg3_str = f"{ppg3:.2f}" if pd.notna(ppg3) else "N/A"
    cppg_str = f"{cppg:.2f}" if pd.notna(cppg) else "N/A"
    print(f"{'OVERALL':<25} | {n:>4} | {hit24_rate:>6.1f}% | {hit12_rate:>6.1f}% | {ppg3_str:>13} | {cppg_str:>16} | {int(sp_min):>3} - {int(sp_max):>3}")


# =============================================================================
# RAS TIER BREAKDOWN -- DETAIL BY ROUND (to check MNAR pattern)
# =============================================================================

print("\n" + "=" * 120)
print("  RAS x ROUND INTERACTION (MNAR CHECK)")
print("  Do missing-RAS players perform differently by draft round?")
print("=" * 120)

for pos_label, df in [("WR", wr_merged), ("RB", rb_valid)]:
    df['round'] = pd.cut(df['pick_final'], bins=[0, 32, 64, 96, 128, 160, 192, 260],
                         labels=['Rd1', 'Rd2', 'Rd3', 'Rd4', 'Rd5', 'Rd6', 'Rd7'])
    df['ras_status'] = df['ras_observed'].map({True: 'Observed', False: 'Imputed'})
    
    print(f"\n--- {pos_label}: Hit24 rate by Round x RAS Status ---")
    header = f"{'Round':<8} | {'Observed N':>11} {'Hit24%':>8} {'Mean RAS':>9} | {'Imputed N':>10} {'Hit24%':>8}"
    print(header)
    print('-' * 70)
    
    for rd in ['Rd1', 'Rd2', 'Rd3', 'Rd4', 'Rd5', 'Rd6', 'Rd7']:
        rd_df = df[df['round'] == rd]
        obs = rd_df[rd_df['ras_status'] == 'Observed']
        imp = rd_df[rd_df['ras_status'] == 'Imputed']
        
        obs_n = len(obs)
        obs_hit = f"{obs['hit24'].mean()*100:.1f}%" if obs_n > 0 else "N/A"
        obs_ras = f"{obs['RAS'].mean():.1f}" if obs_n > 0 else "N/A"
        
        imp_n = len(imp)
        imp_hit = f"{imp['hit24'].mean()*100:.1f}%" if imp_n > 0 else "N/A"
        
        print(f"{rd:<8} | {obs_n:>11} {obs_hit:>8} {obs_ras:>9} | {imp_n:>10} {imp_hit:>8}")


# =============================================================================
# STATISTICAL SIGNIFICANCE: Does RAS improve over DC+Production?
# =============================================================================

print("\n" + "=" * 120)
print("  STATISTICAL COMPARISON: Model A (with RAS) vs Model B (without RAS)")
print("  Using Steiger's Z-test for dependent correlations")
print("=" * 120)

for pos_label, df in [("WR", wr_merged), ("RB", rb_valid)]:
    for ppg_col in ['career_ppg', 'first_3yr_ppg']:
        valid = df[df[ppg_col].notna()].copy()
        if len(valid) < 10:
            print(f"\n  {pos_label} -> {ppg_col}: Not enough data for paired comparison")
            continue
        
        r_A, _ = stats.pearsonr(valid['slap_A'], valid[ppg_col])
        r_B, _ = stats.pearsonr(valid['slap_B'], valid[ppg_col])
        r_AB, _ = stats.pearsonr(valid['slap_A'], valid['slap_B'])
        n = len(valid)
        
        # Steiger's Z-test for dependent correlations
        z1 = np.arctanh(r_A)
        z2 = np.arctanh(r_B)
        
        # Steiger (1980) formula
        det = 1 - r_A**2 - r_B**2 - r_AB**2 + 2 * r_A * r_B * r_AB
        
        numerator = (z1 - z2) * np.sqrt((n - 3) * (1 + r_AB))
        denominator = np.sqrt(2 * det * (n - 1) / (n - 3)**2 + (z1 - z2)**2 * (1 + r_AB)**3 / (4 * (n - 1)))
        
        if denominator > 0:
            z_stat = numerator / denominator
            p_steiger = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_steiger = 1.0
        
        print(f"\n  --- {pos_label} -> {ppg_col} (n={n}) ---")
        print(f"  Model A (with RAS) r:    {r_A:.4f}")
        print(f"  Model B (without RAS) r: {r_B:.4f}")
        print(f"  Difference (A - B):      {r_A - r_B:+.4f}")
        print(f"  Correlation A vs B:      {r_AB:.4f}")
        print(f"  Steiger's Z:             z = {z_stat:.3f}, p = {p_steiger:.4f}")
        
        if r_A > r_B:
            if p_steiger < 0.05:
                print(f"  --> RAS SIGNIFICANTLY improves prediction (p < 0.05)")
            else:
                print(f"  --> RAS improves prediction, but NOT significantly (p = {p_steiger:.3f})")
        else:
            if p_steiger < 0.05:
                print(f"  --> RAS SIGNIFICANTLY HURTS prediction (p < 0.05)")
            else:
                print(f"  --> RAS slightly hurts prediction, but NOT significantly (p = {p_steiger:.3f})")


# =============================================================================
# INCREMENTAL R-SQUARED: How much variance does RAS add?
# =============================================================================

print("\n" + "=" * 120)
print("  INCREMENTAL R-SQUARED: Variance explained by RAS beyond DC + Production")
print("=" * 120)

try:
    from sklearn.linear_model import LinearRegression
    sklearn_available = True
except ImportError:
    sklearn_available = False

if sklearn_available:
    for pos_label, df, prod_col in [("WR", wr_merged, 'breakout_score'), ("RB", rb_valid, 'prod_score')]:
        for outcome in ['career_ppg', 'first_3yr_ppg', 'hit24']:
            valid = df[['dc_score', prod_col, 'ras_score', outcome]].dropna()
            if len(valid) < 10:
                continue
            
            X_base = valid[['dc_score', prod_col]].values
            X_full = valid[['dc_score', prod_col, 'ras_score']].values
            y = valid[outcome].values
            
            model_base = LinearRegression().fit(X_base, y)
            model_full = LinearRegression().fit(X_full, y)
            
            r2_base = model_base.score(X_base, y)
            r2_full = model_full.score(X_full, y)
            r2_incr = r2_full - r2_base
            
            # F-test for incremental R-squared
            n = len(valid)
            k_base = 2  # DC + production
            k_full = 3  # DC + production + RAS
            df_num = k_full - k_base  # 1
            df_den = n - k_full - 1
            
            if df_den > 0 and (1 - r2_full) > 0:
                f_stat = (r2_incr / df_num) / ((1 - r2_full) / df_den)
                p_f = 1 - stats.f.cdf(f_stat, df_num, df_den)
            else:
                f_stat = 0
                p_f = 1.0
            
            sig = "***" if p_f < 0.001 else "**" if p_f < 0.01 else "*" if p_f < 0.05 else ""
            print(f"  {pos_label} -> {outcome:<15}: R2_base={r2_base:.4f}  R2_full={r2_full:.4f}  R2_incr={r2_incr:+.4f}  F={f_stat:.2f}  p={p_f:.4f} {sig}")
            
            # Show RAS coefficient
            ras_coef = model_full.coef_[2]
            print(f"    RAS coefficient: {ras_coef:.4f} (positive = higher RAS -> better outcome)")
else:
    print("  sklearn not available, skipping incremental R-squared analysis")


# =============================================================================
# SUMMARY / VERDICT
# =============================================================================

print("\n" + "=" * 120)
print("  SUMMARY: RAS VALUE TEST VERDICT")
print("=" * 120)

# Compute key comparison metrics
for pos_label, df in [("WR", wr_merged), ("RB", rb_valid)]:
    valid = df[df['career_ppg'].notna()]
    r_A, p_A = stats.pearsonr(valid['slap_A'], valid['career_ppg'])
    r_B, p_B = stats.pearsonr(valid['slap_B'], valid['career_ppg'])
    r_D, p_D = stats.pearsonr(valid['slap_D'], valid['career_ppg'])
    r_E, p_E = stats.pearsonr(valid['slap_E'], valid['career_ppg'])
    
    print(f"\n  {pos_label}:")
    print(f"    With RAS (Model A):      r = {r_A:.4f} (p = {p_A:.4f})")
    print(f"    Without RAS (Model B):   r = {r_B:.4f} (p = {p_B:.4f})")
    print(f"    DC only (Model D):       r = {r_D:.4f} (p = {p_D:.4f})")
    print(f"    RAS alone (Model E):     r = {r_E:.4f} (p = {p_E:.4f})")
    
    diff = r_A - r_B
    if diff > 0.01:
        print(f"    --> Including RAS HELPS: +{diff:.4f} correlation")
    elif diff < -0.01:
        print(f"    --> Including RAS HURTS: {diff:.4f} correlation")
    else:
        print(f"    --> Including RAS has NEGLIGIBLE effect: {diff:+.4f} correlation")

print("\n" + "=" * 120)
print("  END OF ANALYSIS 3")
print("=" * 120)
