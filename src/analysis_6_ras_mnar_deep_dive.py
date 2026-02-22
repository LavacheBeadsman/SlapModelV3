"""
Analysis 6: RAS MNAR Deep Dive
================================
Tests 6 approaches to handle Missing Not At Random (MNAR) RAS data.

The problem: Elite prospects skip combine testing because they're already valued.
Missing RAS is NOT random — it's correlated with being good.

Approaches tested:
  0. SCOPE: Show the MNAR problem by round
  1. Current model (position-average imputation)
  2. No-RAS baseline (drop athletic entirely)
  3. Round-average imputation
  4. 75th-percentile-of-round imputation
  5. Position-round median imputation
  6. Binary RAS flag (above-avg = 1, below = 0)
  7. Speed Score from 40-time + weight (individual metrics)
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE FORMULAS (unchanged from model)
# ============================================================================
def normalize_draft_capital(pick):
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))

def wr_breakout_score(breakout_age, dominator_pct):
    if breakout_age is None or pd.isna(breakout_age):
        if dominator_pct is not None and pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
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
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return None
    if pd.isna(draft_age): draft_age = 22
    ratio = rec_yards / team_pass_att
    season_age = draft_age - 1
    age_weight = 1.15 - (0.05 * (season_age - 19))
    age_weight = max(0.85, min(1.15, age_weight))
    raw_score = ratio * age_weight * 100
    return min(99.9, max(0, raw_score / 1.75))

def speed_score(forty, weight):
    """Barnwell Speed Score: (weight * 200) / (forty^4)"""
    if pd.isna(forty) or pd.isna(weight) or forty == 0:
        return None
    return (weight * 200) / (forty ** 4)

def corr_safe(x, y):
    """Pearson r with safe handling of NaN and small N."""
    valid = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(valid) < 10:
        return np.nan, np.nan, len(valid)
    r, p = pearsonr(valid['x'], valid['y'])
    return r, p, len(valid)

# ============================================================================
# LOAD ALL DATA
# ============================================================================
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
wr_bt = pd.read_csv('data/wr_backtest_expanded_final.csv')
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')

# Load combine data (has 40-time and weight)
combine = pd.read_csv('data/backtest_college_stats.csv',
                       usecols=['player_name', 'draft_year', 'position', 'weight', 'forty'])
combine2 = pd.read_csv('data/slap_full_breakdown.csv',
                        usecols=['player_name', 'draft_year', 'position', 'weight', 'forty'])
# Merge combine sources (prefer backtest_college_stats, fill from slap_full_breakdown)
combine_all = pd.concat([combine, combine2]).drop_duplicates(subset=['player_name', 'draft_year'], keep='first')

# Build WR dataset
wr = outcomes[outcomes['position'] == 'WR'].merge(
    wr_bt[['player_name', 'pick', 'round', 'RAS', 'breakout_age', 'peak_dominator', 'draft_year']],
    on=['player_name', 'draft_year'], how='inner', suffixes=('', '_bt'))
wr['pick'] = wr['pick'].fillna(wr['pick_bt'])
wr['prod_score'] = wr.apply(lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)
wr['dc_score'] = wr['pick'].apply(normalize_draft_capital)
wr['has_ras'] = wr['RAS'].notna()
wr['ras_raw'] = wr['RAS']  # Keep original

# Merge combine data into WR
wr = wr.merge(combine_all[combine_all['position'] == 'WR'][['player_name', 'draft_year', 'weight', 'forty']],
              on=['player_name', 'draft_year'], how='left')

# Build RB dataset
rb = outcomes[outcomes['position'] == 'RB'].merge(
    rb_bt[['player_name', 'pick', 'round', 'age', 'RAS', 'rec_yards', 'team_pass_att', 'draft_year']],
    on=['player_name', 'draft_year'], how='inner', suffixes=('', '_bt'))
rb['pick'] = rb['pick'].fillna(rb['pick_bt'])
rb['prod_score'] = rb.apply(lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb['prod_score'] = rb['prod_score'].fillna(rb['prod_score'].mean())
rb['dc_score'] = rb['pick'].apply(normalize_draft_capital)
rb['has_ras'] = rb['RAS'].notna()
rb['ras_raw'] = rb['RAS']

# Merge combine data into RB
rb = rb.merge(combine_all[combine_all['position'] == 'RB'][['player_name', 'draft_year', 'weight', 'forty']],
              on=['player_name', 'draft_year'], how='left')

# Assign rounds for those missing
for df in [wr, rb]:
    if 'round' not in df.columns or df['round'].isna().any():
        df['round'] = pd.cut(df['pick'], bins=[0, 32, 64, 100, 135, 175, 224, 260],
                              labels=[1, 2, 3, 4, 5, 6, 7]).astype(float)

print(f"WR: {len(wr)} players ({wr['has_ras'].sum()} with RAS, {(~wr['has_ras']).sum()} missing)")
print(f"RB: {len(rb)} players ({rb['has_ras'].sum()} with RAS, {(~rb['has_ras']).sum()} missing)")
print(f"WR with 40-time: {wr['forty'].notna().sum()}/{len(wr)}")
print(f"RB with 40-time: {rb['forty'].notna().sum()}/{len(rb)}")
print(f"WR with weight: {wr['weight'].notna().sum()}/{len(wr)}")
print(f"RB with weight: {rb['weight'].notna().sum()}/{len(rb)}")

# ============================================================================
# SECTION 0: SCOPE OF THE MNAR PROBLEM
# ============================================================================
print("\n" + "=" * 130)
print("SECTION 0: SCOPE OF THE MNAR PROBLEM — Missing RAS by Round")
print("=" * 130)

OUTCOMES = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']

for pos_name, df in [("WR", wr), ("RB", rb)]:
    print(f"\n--- {pos_name} ---")
    print(f"{'Round':<8} {'N':>4} {'Missing':>8} {'Miss%':>7}  |  "
          f"{'Hit24(has)':>11} {'Hit24(miss)':>12} {'Gap':>6}  |  "
          f"{'Hit12(has)':>11} {'Hit12(miss)':>12} {'Gap':>6}  |  "
          f"{'3yrPPG(has)':>12} {'3yrPPG(miss)':>13} {'Gap':>6}")
    print("-" * 130)

    for rd in [1, 2, 3, 4, 5, 6, 7]:
        tier = df[df['round'] == rd]
        if len(tier) == 0:
            continue
        has = tier[tier['has_ras']]
        miss = tier[~tier['has_ras']]
        n_miss = len(miss)
        pct_miss = n_miss / len(tier) * 100 if len(tier) > 0 else 0

        h24_has = has['hit24'].mean() * 100 if len(has) > 0 else 0
        h24_miss = miss['hit24'].mean() * 100 if len(miss) > 0 else 0
        h24_gap = h24_miss - h24_has

        h12_has = has['hit12'].mean() * 100 if len(has) > 0 else 0
        h12_miss = miss['hit12'].mean() * 100 if len(miss) > 0 else 0
        h12_gap = h12_miss - h12_has

        ppg_has = has['first_3yr_ppg'].mean() if len(has) > 0 else 0
        ppg_miss = miss['first_3yr_ppg'].mean() if len(miss) > 0 else 0
        ppg_gap = ppg_miss - ppg_has

        print(f"Rd {int(rd):<4} {len(tier):>4} {n_miss:>8} {pct_miss:>6.1f}%  |  "
              f"{h24_has:>10.1f}% {h24_miss:>11.1f}% {h24_gap:>+5.1f}  |  "
              f"{h12_has:>10.1f}% {h12_miss:>11.1f}% {h12_gap:>+5.1f}  |  "
              f"{ppg_has:>11.1f} {ppg_miss:>12.1f} {ppg_gap:>+5.1f}")

    # Summary row
    has_all = df[df['has_ras']]
    miss_all = df[~df['has_ras']]
    print(f"{'ALL':<8} {len(df):>4} {len(miss_all):>8} {len(miss_all)/len(df)*100:>6.1f}%  |  "
          f"{has_all['hit24'].mean()*100:>10.1f}% {miss_all['hit24'].mean()*100:>11.1f}% "
          f"{(miss_all['hit24'].mean()-has_all['hit24'].mean())*100:>+5.1f}  |  "
          f"{has_all['hit12'].mean()*100:>10.1f}% {miss_all['hit12'].mean()*100:>11.1f}% "
          f"{(miss_all['hit12'].mean()-has_all['hit12'].mean())*100:>+5.1f}  |  "
          f"{has_all['first_3yr_ppg'].mean():>11.1f} {miss_all['first_3yr_ppg'].mean():>12.1f} "
          f"{miss_all['first_3yr_ppg'].mean()-has_all['first_3yr_ppg'].mean():>+5.1f}")

    # Name the missing players for rounds 1-3
    print(f"\n  Missing RAS players (Rounds 1-3) for {pos_name}:")
    for _, row in df[(~df['has_ras']) & (df['round'] <= 3)].sort_values('pick').iterrows():
        h24_str = "HIT" if row['hit24'] else "miss"
        ppg_str = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "N/A"
        forty_str = f"{row['forty']:.2f}" if pd.notna(row['forty']) else "no 40"
        print(f"    Rd{int(row['round'])} Pick {int(row['pick']):>3}: {row['player_name']:<25} "
              f"{int(row['draft_year'])}  hit24={h24_str}  3yr_ppg={ppg_str:>5}  {forty_str}")

# ============================================================================
# SECTION 1-6: BUILD ALL IMPUTATION APPROACHES
# ============================================================================
print("\n" + "=" * 130)
print("BUILDING ALL IMPUTATION APPROACHES")
print("=" * 130)

def build_ras_variants(df, pos_name):
    """Build all RAS variants for a dataframe."""

    # --- Approach 1: Current (position-avg imputation) ---
    avg_ras = df.loc[df['has_ras'], 'ras_raw'].mean()
    df['ras_current'] = df['ras_raw'].fillna(avg_ras) * 10  # Scale 0-10 -> 0-100

    # --- Approach 2: No RAS (baseline) ---
    # No column needed, we just skip RAS in the SLAP formula

    # --- Approach 3: Round-average imputation ---
    round_avg = df[df['has_ras']].groupby('round')['ras_raw'].mean()
    df['ras_round_avg'] = df.apply(
        lambda r: (r['ras_raw'] if pd.notna(r['ras_raw']) else round_avg.get(r['round'], avg_ras)) * 10,
        axis=1)

    # --- Approach 4: 75th percentile of round ---
    round_p75 = df[df['has_ras']].groupby('round')['ras_raw'].quantile(0.75)
    df['ras_round_p75'] = df.apply(
        lambda r: (r['ras_raw'] if pd.notna(r['ras_raw']) else round_p75.get(r['round'], avg_ras)) * 10,
        axis=1)

    # --- Approach 5: Position-round median ---
    round_med = df[df['has_ras']].groupby('round')['ras_raw'].median()
    df['ras_round_med'] = df.apply(
        lambda r: (r['ras_raw'] if pd.notna(r['ras_raw']) else round_med.get(r['round'], avg_ras)) * 10,
        axis=1)

    # --- Approach 6: Binary RAS flag ---
    # 1 = above-average athlete (RAS > 5.0 or missing from Round 1-2)
    # 0 = below-average or missing from later rounds
    def binary_ras(row):
        if pd.notna(row['ras_raw']):
            return 1.0 if row['ras_raw'] > 5.0 else 0.0
        else:
            # Missing: assume good if early round, bad if late
            return 1.0 if row['round'] <= 2 else 0.0
    df['ras_binary'] = df.apply(binary_ras, axis=1)
    # Scale to 0-100 for SLAP formula compatibility
    df['ras_binary_scaled'] = df['ras_binary'] * 100

    # --- Approach 7: Speed Score from 40-time + weight ---
    df['speed_score_raw'] = df.apply(lambda r: speed_score(r['forty'], r['weight']), axis=1)
    # Normalize speed score to 0-100
    ss_valid = df['speed_score_raw'].dropna()
    if len(ss_valid) > 0:
        ss_min, ss_max = ss_valid.min(), ss_valid.max()
        df['ras_speed_score'] = df['speed_score_raw'].apply(
            lambda x: ((x - ss_min) / (ss_max - ss_min)) * 100 if pd.notna(x) else None)
        # For missing speed score, use position average
        ss_avg = df['ras_speed_score'].mean()
        df['ras_speed_score'] = df['ras_speed_score'].fillna(ss_avg)
    else:
        df['ras_speed_score'] = 50.0  # Fallback

    # Print coverage stats
    n_total = len(df)
    n_miss_ras = (~df['has_ras']).sum()
    n_has_forty = df['forty'].notna().sum()
    n_has_weight = df['weight'].notna().sum()
    n_has_both = ((df['forty'].notna()) & (df['weight'].notna())).sum()
    n_miss_ras_has_forty = ((~df['has_ras']) & (df['forty'].notna())).sum()

    print(f"\n  {pos_name}: {n_total} total, {n_miss_ras} missing RAS")
    print(f"    40-time available: {n_has_forty}/{n_total} ({n_has_forty/n_total*100:.1f}%)")
    print(f"    Weight available:  {n_has_weight}/{n_total} ({n_has_weight/n_total*100:.1f}%)")
    print(f"    Speed Score (both): {n_has_both}/{n_total} ({n_has_both/n_total*100:.1f}%)")
    print(f"    Missing RAS but HAS 40-time: {n_miss_ras_has_forty}/{n_miss_ras}")

    # Show round-level imputation values
    print(f"\n  {pos_name} Round-level RAS stats (observed players only):")
    print(f"  {'Round':>6} {'N_obs':>6} {'Mean':>6} {'Median':>8} {'P75':>6} {'Min':>6} {'Max':>6}")
    for rd in sorted(df[df['has_ras']]['round'].unique()):
        tier = df[(df['has_ras']) & (df['round'] == rd)]['ras_raw']
        if len(tier) > 0:
            print(f"  Rd {int(rd):>3} {len(tier):>6} {tier.mean():>6.2f} {tier.median():>8.2f} "
                  f"{tier.quantile(0.75):>6.2f} {tier.min():>6.2f} {tier.max():>6.2f}")

    return df

wr = build_ras_variants(wr, "WR")
rb = build_ras_variants(rb, "RB")

# ============================================================================
# SECTION 2: WEIGHT OPTIMIZATION FOR EACH APPROACH
# ============================================================================
print("\n" + "=" * 130)
print("SECTION 2: FULL COMPARISON — ALL APPROACHES vs ALL OUTCOMES")
print("=" * 130)

# Weight combos to test for each approach
WR_COMBOS = [
    (0.65, 0.20, 0.15, "65/20/15 (current)"),
    (0.70, 0.20, 0.10, "70/20/10 (Test1 best)"),
    (0.75, 0.15, 0.10, "75/15/10"),
]

RB_COMBOS = [
    (0.50, 0.35, 0.15, "50/35/15 (current)"),
    (0.65, 0.25, 0.10, "65/25/10 (Test1 best)"),
    (0.60, 0.30, 0.10, "60/30/10"),
]

# No-RAS weight combos (redistribute athletic weight)
WR_NO_RAS = [(0.7647, 0.2353, 0.0, "76.5/23.5/0 (no RAS)")]
RB_NO_RAS = [(0.5882, 0.4118, 0.0, "58.8/41.2/0 (no RAS)")]

APPROACHES = {
    'Current (pos-avg)':     'ras_current',
    'Round-average':         'ras_round_avg',
    'Round 75th pctile':     'ras_round_p75',
    'Round median':          'ras_round_med',
    'Binary flag':           'ras_binary_scaled',
    'Speed Score':           'ras_speed_score',
}

def run_comparison(df, pos_name, combos, no_ras_combos):
    """Run all approaches and weight combos for one position."""

    results = []

    for approach_name, ras_col in APPROACHES.items():
        for w_dc, w_prod, w_ath, combo_name in combos:
            df['slap_test'] = w_dc * df['dc_score'] + w_prod * df['prod_score'] + w_ath * df[ras_col]
            for out in OUTCOMES:
                r, p, n = corr_safe(df['slap_test'], df[out])
                results.append({
                    'approach': approach_name,
                    'weights': combo_name,
                    'outcome': out,
                    'r': r, 'p': p, 'n': n
                })

    # No-RAS baseline
    for w_dc, w_prod, w_ath, combo_name in no_ras_combos:
        df['slap_test'] = w_dc * df['dc_score'] + w_prod * df['prod_score']
        for out in OUTCOMES:
            r, p, n = corr_safe(df['slap_test'], df[out])
            results.append({
                'approach': 'NO RAS (dropped)',
                'weights': combo_name,
                'outcome': out,
                'r': r, 'p': p, 'n': n
            })

    return pd.DataFrame(results)

wr_results = run_comparison(wr, "WR", WR_COMBOS, WR_NO_RAS)
rb_results = run_comparison(rb, "RB", RB_COMBOS, RB_NO_RAS)

# ============================================================================
# SECTION 3: DISPLAY RESULTS
# ============================================================================

def display_results(results_df, pos_name, current_weights_label):
    """Display comparison table for one position."""

    print(f"\n{'='*130}")
    print(f"{pos_name} RESULTS: ALL APPROACHES x ALL OUTCOMES")
    print(f"{'='*130}")

    # For each weight combo, show all approaches
    for wt in results_df['weights'].unique():
        wt_df = results_df[results_df['weights'] == wt]
        print(f"\n--- Weights: {wt} ---")
        print(f"{'Approach':<25} {'hit24 r':>10} {'hit12 r':>10} {'3yr_ppg r':>10} {'career_ppg r':>12} {'Avg r':>8}")
        print("-" * 85)

        rows_for_avg = []
        for approach in list(APPROACHES.keys()) + ['NO RAS (dropped)']:
            a_df = wt_df[wt_df['approach'] == approach]
            if len(a_df) == 0:
                continue
            vals = {}
            for _, row in a_df.iterrows():
                vals[row['outcome']] = row['r']

            avg_r = np.mean([v for v in vals.values() if pd.notna(v)])
            tag = " <<<" if 'current' in wt.lower() and 'Current' in approach else ""
            if approach == 'NO RAS (dropped)':
                tag = " (baseline)"

            print(f"{approach:<25} {vals.get('hit24', np.nan):>10.4f} {vals.get('hit12', np.nan):>10.4f} "
                  f"{vals.get('first_3yr_ppg', np.nan):>10.4f} {vals.get('career_ppg', np.nan):>12.4f} "
                  f"{avg_r:>8.4f}{tag}")
            rows_for_avg.append((approach, avg_r, vals))

        # Find best approach for each outcome
        print(f"\n  Best approach per outcome (weights: {wt}):")
        for out in OUTCOMES:
            best_r, best_name = 0, ''
            for approach in list(APPROACHES.keys()) + ['NO RAS (dropped)']:
                a_df = wt_df[(wt_df['approach'] == approach) & (wt_df['outcome'] == out)]
                if len(a_df) > 0 and abs(a_df.iloc[0]['r']) > abs(best_r):
                    best_r = a_df.iloc[0]['r']
                    best_name = approach
            no_ras_row = wt_df[(wt_df['approach'] == 'NO RAS (dropped)') & (wt_df['outcome'] == out)]
            no_ras_r = no_ras_row.iloc[0]['r'] if len(no_ras_row) > 0 else np.nan
            beats_no_ras = "YES" if best_r > no_ras_r else "NO"
            print(f"    {out:<18}: {best_name:<25} r={best_r:.4f}  "
                  f"(beats no-RAS: {beats_no_ras}, no-RAS r={no_ras_r:.4f})")

display_results(wr_results, "WR", "65/20/15 (current)")
display_results(rb_results, "RB", "50/35/15 (current)")

# ============================================================================
# SECTION 4: MASTER COMPARISON — BEST OF EACH APPROACH
# ============================================================================
print("\n" + "=" * 130)
print("MASTER COMPARISON: BEST WEIGHT COMBO FOR EACH APPROACH")
print("=" * 130)

for pos_name, results_df in [("WR", wr_results), ("RB", rb_results)]:
    print(f"\n--- {pos_name}: Best weight combo per approach (by avg r across 4 outcomes) ---")
    print(f"{'Approach':<25} {'Best Weights':<25} {'hit24':>8} {'hit12':>8} {'3yr_ppg':>8} {'car_ppg':>8} {'AVG r':>8} {'vs No-RAS':>10}")
    print("-" * 110)

    # Get no-RAS baseline avg
    no_ras_df = results_df[results_df['approach'] == 'NO RAS (dropped)']
    no_ras_avg = {}
    for _, row in no_ras_df.iterrows():
        no_ras_avg[row['outcome']] = row['r']
    no_ras_mean = np.mean(list(no_ras_avg.values()))

    for approach in list(APPROACHES.keys()) + ['NO RAS (dropped)']:
        a_df = results_df[results_df['approach'] == approach]
        if len(a_df) == 0:
            continue

        # Find best weight combo by avg r
        best_avg, best_wt, best_vals = 0, '', {}
        for wt in a_df['weights'].unique():
            wt_rows = a_df[a_df['weights'] == wt]
            vals = {row['outcome']: row['r'] for _, row in wt_rows.iterrows()}
            avg = np.mean(list(vals.values()))
            if avg > best_avg:
                best_avg = avg
                best_wt = wt
                best_vals = vals

        delta = best_avg - no_ras_mean
        delta_str = f"{delta:>+.4f}" if approach != 'NO RAS (dropped)' else "baseline"

        print(f"{approach:<25} {best_wt:<25} "
              f"{best_vals.get('hit24', np.nan):>8.4f} {best_vals.get('hit12', np.nan):>8.4f} "
              f"{best_vals.get('first_3yr_ppg', np.nan):>8.4f} {best_vals.get('career_ppg', np.nan):>8.4f} "
              f"{best_avg:>8.4f} {delta_str:>10}")

# ============================================================================
# SECTION 5: SPEED SCORE DEEP DIVE
# ============================================================================
print("\n" + "=" * 130)
print("SECTION 5: SPEED SCORE DEEP DIVE — Coverage & Individual Metrics")
print("=" * 130)

for pos_name, df in [("WR", wr), ("RB", rb)]:
    print(f"\n--- {pos_name} ---")

    # Coverage comparison
    n_total = len(df)
    n_ras = df['has_ras'].sum()
    n_forty = df['forty'].notna().sum()
    n_weight = df['weight'].notna().sum()
    n_speed = ((df['forty'].notna()) & (df['weight'].notna())).sum()

    print(f"  Coverage: RAS={n_ras}/{n_total} ({n_ras/n_total*100:.1f}%), "
          f"40-time={n_forty}/{n_total} ({n_forty/n_total*100:.1f}%), "
          f"Speed Score={n_speed}/{n_total} ({n_speed/n_total*100:.1f}%)")

    # Players who have 40-time but NOT RAS
    has_forty_no_ras = df[(df['forty'].notna()) & (~df['has_ras'])]
    print(f"  Have 40-time but NO RAS: {len(has_forty_no_ras)} players")
    if len(has_forty_no_ras) > 0:
        for _, row in has_forty_no_ras.sort_values('pick').head(15).iterrows():
            h = "HIT" if row['hit24'] else "miss"
            ppg = f"{row['first_3yr_ppg']:.1f}" if pd.notna(row['first_3yr_ppg']) else "N/A"
            print(f"    Pick {int(row['pick']):>3}: {row['player_name']:<25} "
                  f"40={row['forty']:.2f}  wt={int(row['weight']) if pd.notna(row['weight']) else 'N/A':>3}  "
                  f"SS={row['speed_score_raw']:.1f}" if pd.notna(row['speed_score_raw']) else f"    Pick {int(row['pick']):>3}: {row['player_name']:<25} 40={row['forty']:.2f}  wt=N/A  SS=N/A"
                  , end="")
            print(f"  hit24={h}  3yr_ppg={ppg}")

    # Raw correlation of individual metrics
    print(f"\n  Raw correlations (individual metrics vs outcomes, {pos_name}):")
    print(f"  {'Metric':<20} {'N':>5} {'hit24 r':>10} {'hit12 r':>10} {'3yr_ppg r':>10} {'career_ppg r':>12}")
    print("  " + "-" * 75)

    for metric_name, metric_col in [('RAS (raw 0-10)', 'ras_raw'),
                                     ('40-time (raw)', 'forty'),
                                     ('Weight', 'weight'),
                                     ('Speed Score (raw)', 'speed_score_raw')]:
        line = f"  {metric_name:<20}"
        valid = df[df[metric_col].notna()]
        line += f" {len(valid):>5}"
        for out in OUTCOMES:
            r, p, n = corr_safe(valid[metric_col], valid[out])
            sig = '***' if pd.notna(p) and p < 0.001 else '**' if pd.notna(p) and p < 0.01 else '*' if pd.notna(p) and p < 0.05 else ''
            # Flip sign for forty (lower = better)
            line += f" {r:>9.4f}{sig}"
        print(line)

    print(f"  Note: For 40-time, NEGATIVE r means faster = better (expected direction)")

# ============================================================================
# SECTION 6: BINARY FLAG DEEP DIVE
# ============================================================================
print("\n" + "=" * 130)
print("SECTION 6: BINARY FLAG BREAKDOWN")
print("=" * 130)

for pos_name, df in [("WR", wr), ("RB", rb)]:
    print(f"\n--- {pos_name} ---")
    for flag_val, label in [(1, "Above-avg / Elite-missing"), (0, "Below-avg / Late-missing")]:
        grp = df[df['ras_binary'] == flag_val]
        print(f"  {label} (flag={flag_val}): N={len(grp)}, "
              f"hit24={grp['hit24'].mean()*100:.1f}%, "
              f"hit12={grp['hit12'].mean()*100:.1f}%, "
              f"3yr_ppg={grp['first_3yr_ppg'].mean():.1f}, "
              f"career_ppg={grp['career_ppg'].mean():.1f}")

    # More nuanced breakdown
    print(f"\n  Detailed: RAS status x Draft tier")
    print(f"  {'Group':<40} {'N':>4} {'Hit24%':>8} {'Hit12%':>8} {'3yr PPG':>9} {'Car PPG':>9}")
    print("  " + "-" * 85)

    conditions = [
        ("Rd 1-2, has RAS, RAS>=7",  (df['round'] <= 2) & (df['has_ras']) & (df['ras_raw'] >= 7)),
        ("Rd 1-2, has RAS, RAS<7",   (df['round'] <= 2) & (df['has_ras']) & (df['ras_raw'] < 7)),
        ("Rd 1-2, missing RAS",      (df['round'] <= 2) & (~df['has_ras'])),
        ("Rd 3-5, has RAS, RAS>=7",  (df['round'].between(3, 5)) & (df['has_ras']) & (df['ras_raw'] >= 7)),
        ("Rd 3-5, has RAS, RAS<7",   (df['round'].between(3, 5)) & (df['has_ras']) & (df['ras_raw'] < 7)),
        ("Rd 3-5, missing RAS",      (df['round'].between(3, 5)) & (~df['has_ras'])),
        ("Rd 6-7, has RAS",          (df['round'] >= 6) & (df['has_ras'])),
        ("Rd 6-7, missing RAS",      (df['round'] >= 6) & (~df['has_ras'])),
    ]

    for label, mask in conditions:
        grp = df[mask]
        if len(grp) == 0:
            continue
        h24 = grp['hit24'].mean() * 100
        h12 = grp['hit12'].mean() * 100
        ppg3 = grp['first_3yr_ppg'].mean()
        ppgc = grp['career_ppg'].mean()
        print(f"  {label:<40} {len(grp):>4} {h24:>7.1f}% {h12:>7.1f}% {ppg3:>9.1f} {ppgc:>9.1f}")

# ============================================================================
# SECTION 7: FINAL RECOMMENDATION
# ============================================================================
print("\n" + "=" * 130)
print("SECTION 7: FINAL RANKING — ALL APPROACHES FOR EACH POSITION")
print("=" * 130)

for pos_name, results_df in [("WR", wr_results), ("RB", rb_results)]:
    print(f"\n--- {pos_name}: Ranked by average r across all 4 outcomes (best weight combo per approach) ---")
    print(f"{'Rank':>4} {'Approach':<25} {'Best Weights':<25} {'Avg r':>8} {'Delta vs No-RAS':>16}")
    print("-" * 85)

    approach_best = []
    for approach in list(APPROACHES.keys()) + ['NO RAS (dropped)']:
        a_df = results_df[results_df['approach'] == approach]
        if len(a_df) == 0:
            continue
        best_avg, best_wt = 0, ''
        for wt in a_df['weights'].unique():
            wt_rows = a_df[a_df['weights'] == wt]
            vals = {row['outcome']: row['r'] for _, row in wt_rows.iterrows()}
            avg = np.mean(list(vals.values()))
            if avg > best_avg:
                best_avg = avg
                best_wt = wt
        approach_best.append((approach, best_wt, best_avg))

    # Sort by avg r descending
    approach_best.sort(key=lambda x: -x[2])
    no_ras_avg = [x[2] for x in approach_best if x[0] == 'NO RAS (dropped)'][0]

    for rank, (approach, wt, avg_r) in enumerate(approach_best, 1):
        delta = avg_r - no_ras_avg
        delta_str = f"{delta:>+.4f}" if approach != 'NO RAS (dropped)' else "baseline"
        marker = " ***" if rank == 1 else ""
        print(f"{rank:>4} {approach:<25} {wt:<25} {avg_r:>8.4f} {delta_str:>16}{marker}")

print("\n" + "=" * 130)
print("END OF ANALYSIS 6: RAS MNAR DEEP DIVE")
print("=" * 130)
