"""
Analysis Test 1: Weight Optimization for SLAP Model V3

Tests different weight combinations for the 3 SLAP components (DC, Production, Athletic)
for both WRs and RBs separately. Correlates each SLAP score against four NFL outcomes:
hit24, hit12, first_3yr_ppg, career_ppg.

Current weights: WR 65/20/15, RB 50/35/15
"""

import pandas as pd
import numpy as np
from scipy import stats

# ─────────────────────────────────────────────
# FORMULAS (copied exactly from CLAUDE.md)
# ─────────────────────────────────────────────

def normalize_draft_capital(pick):
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))


def wr_breakout_score(breakout_age, dominator_pct):
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


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

outcomes = pd.read_csv('/home/user/SlapModelV3/data/backtest_outcomes_complete.csv')
wr_data = pd.read_csv('/home/user/SlapModelV3/data/wr_backtest_expanded_final.csv')
rb_data = pd.read_csv('/home/user/SlapModelV3/data/rb_backtest_with_receiving.csv')

# ─────────────────────────────────────────────
# PREPARE WR DATA
# ─────────────────────────────────────────────

# Merge WR features with outcomes (outcomes has first_3yr_ppg, career_ppg, seasons_played)
wr_outcomes = outcomes[outcomes['position'] == 'WR'][
    ['player_name', 'draft_year', 'pick', 'seasons_played', 'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
]

# Merge on player_name and draft_year
wr_merged = pd.merge(
    wr_data[['player_name', 'draft_year', 'pick', 'RAS', 'breakout_age', 'peak_dominator']],
    wr_outcomes,
    on=['player_name', 'draft_year'],
    how='inner',
    suffixes=('_wr', '_out')
)

# Use pick from outcomes (more reliable), fall back to WR data
wr_merged['pick'] = wr_merged['pick_out'].fillna(wr_merged['pick_wr'])
wr_merged.drop(columns=['pick_wr', 'pick_out'], inplace=True)

# Compute component scores for WRs
wr_merged['dc_score'] = wr_merged['pick'].apply(normalize_draft_capital)

wr_merged['breakout_score'] = wr_merged.apply(
    lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1
)

# Athletic score: RAS * 10, missing = 68.9 (position average)
WR_AVG_ATH = 68.9
wr_merged['athletic_score'] = wr_merged['RAS'].apply(
    lambda x: x * 10 if pd.notna(x) else WR_AVG_ATH
)

print(f"WR merged dataset: {len(wr_merged)} players")
print(f"  Draft years: {sorted(wr_merged['draft_year'].unique())}")
print(f"  Missing breakout_age: {wr_merged['breakout_age'].isna().sum()}")
print(f"  Missing RAS (imputed with avg {WR_AVG_ATH}): {wr_merged['RAS'].isna().sum()}")
print()

# ─────────────────────────────────────────────
# PREPARE RB DATA
# ─────────────────────────────────────────────

rb_outcomes = outcomes[outcomes['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'seasons_played', 'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
]

rb_merged = pd.merge(
    rb_data[['player_name', 'draft_year', 'pick', 'age', 'RAS', 'rec_yards', 'team_pass_att']],
    rb_outcomes,
    on=['player_name', 'draft_year'],
    how='inner',
    suffixes=('_rb', '_out')
)

rb_merged['pick'] = rb_merged['pick_out'].fillna(rb_merged['pick_rb'])
rb_merged.drop(columns=['pick_rb', 'pick_out'], inplace=True)

# Compute component scores for RBs
rb_merged['dc_score'] = rb_merged['pick'].apply(normalize_draft_capital)

rb_merged['production_score'] = rb_merged.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1
)

# Athletic score: RAS * 10, missing = 66.5 (position average)
RB_AVG_ATH = 66.5
rb_merged['athletic_score'] = rb_merged['RAS'].apply(
    lambda x: x * 10 if pd.notna(x) else RB_AVG_ATH
)

# For RBs with missing production, fill with position average
rb_prod_avg = rb_merged['production_score'].mean()
rb_merged['production_score_filled'] = rb_merged['production_score'].fillna(rb_prod_avg)

print(f"RB merged dataset: {len(rb_merged)} players")
print(f"  Draft years: {sorted(rb_merged['draft_year'].unique())}")
print(f"  Missing production (imputed with avg {rb_prod_avg:.1f}): {rb_merged['production_score'].isna().sum()}")
print(f"  Missing RAS (imputed with avg {RB_AVG_ATH}): {rb_merged['RAS'].isna().sum()}")
print()


# ─────────────────────────────────────────────
# WEIGHT COMBOS
# ─────────────────────────────────────────────

wr_combos = [
    (80, 10, 10),
    (75, 15, 10),
    (70, 20, 10),
    (65, 20, 15),   # current
    (60, 25, 15),
    (55, 30, 15),
    (50, 35, 15),
    (70, 15, 15),
    (60, 30, 10),
    (50, 30, 20),
]

rb_combos = [
    (70, 20, 10),
    (65, 25, 10),
    (60, 30, 10),
    (55, 35, 10),
    (50, 35, 15),   # current
    (45, 40, 15),
    (40, 45, 15),
    (50, 30, 20),
    (60, 25, 15),
    (55, 30, 15),
]

OUTCOME_COLS = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
CURRENT_WR = (65, 20, 15)
CURRENT_RB = (50, 35, 15)


def compute_correlations(df, slap_col, outcome_cols):
    """Compute Pearson r and p-value for SLAP vs each outcome."""
    results = {}
    for outcome in outcome_cols:
        valid = df[[slap_col, outcome]].dropna()
        if len(valid) < 3:
            results[outcome] = (np.nan, np.nan, 0)
            continue
        r, p = stats.pearsonr(valid[slap_col], valid[outcome])
        results[outcome] = (r, p, len(valid))
    return results


def run_optimization(df, combos, current_combo, position_label,
                     dc_col='dc_score', prod_col='production_score_filled',
                     ath_col='athletic_score'):
    """Test all weight combos and print results table."""

    all_results = []

    for (w_dc, w_prod, w_ath) in combos:
        # Compute SLAP score
        w_dc_f = w_dc / 100.0
        w_prod_f = w_prod / 100.0
        w_ath_f = w_ath / 100.0

        df['slap'] = (
            w_dc_f * df[dc_col] +
            w_prod_f * df[prod_col] +
            w_ath_f * df[ath_col]
        )

        corrs = compute_correlations(df, 'slap', OUTCOME_COLS)
        row = {
            'combo': f"{w_dc}/{w_prod}/{w_ath}",
            'w_dc': w_dc, 'w_prod': w_prod, 'w_ath': w_ath,
            'is_current': (w_dc, w_prod, w_ath) == current_combo,
        }
        for outcome in OUTCOME_COLS:
            r, p, n = corrs[outcome]
            row[f'{outcome}_r'] = r
            row[f'{outcome}_p'] = p
            row[f'{outcome}_n'] = n
        all_results.append(row)

    results_df = pd.DataFrame(all_results)

    # Find best combo for each outcome
    best_for = {}
    for outcome in OUTCOME_COLS:
        best_idx = results_df[f'{outcome}_r'].idxmax()
        best_for[outcome] = best_idx

    # Print header
    n_players = len(df)
    years = f"{int(df['draft_year'].min())}-{int(df['draft_year'].max())}"
    print("=" * 120)
    print(f"{position_label} WEIGHT OPTIMIZATION (N={n_players} players, draft years {years})")
    print("=" * 120)
    print(f"{'DC/Prod/Ath':<15} | {'hit24 r (p)':<20} | {'hit12 r (p)':<20} | {'first_3yr_ppg r (p)':<24} | {'career_ppg r (p)':<20}")
    print("-" * 120)

    for idx, row in results_df.iterrows():
        combo_str = row['combo']
        parts = []
        for outcome in OUTCOME_COLS:
            r = row[f'{outcome}_r']
            p = row[f'{outcome}_p']
            cell = f"{r:+.4f} ({p:.4f})"
            if idx == best_for[outcome]:
                cell += " *"
            parts.append(cell)

        marker = " <<<" if row['is_current'] else ""
        print(f"{combo_str:<15} | {parts[0]:<20} | {parts[1]:<20} | {parts[2]:<24} | {parts[3]:<20}{marker}")

    print("-" * 120)
    print("  <<< = current weights    * = best for that outcome column")
    print()

    # ─────────────────────────────────────────────
    # RECOMMENDATION
    # ─────────────────────────────────────────────
    print(f"RECOMMENDATION ({position_label}):")
    print("-" * 60)

    current_row = results_df[results_df['is_current']].iloc[0]

    for outcome in OUTCOME_COLS:
        best_idx_val = best_for[outcome]
        best_row = results_df.loc[best_idx_val]
        best_r = best_row[f'{outcome}_r']
        current_r = current_row[f'{outcome}_r']
        diff = best_r - current_r

        if best_row['is_current']:
            print(f"  {outcome}: Current weights ARE the best ({current_r:+.4f})")
        else:
            print(f"  {outcome}: Best = {best_row['combo']} ({best_r:+.4f}) vs current ({current_r:+.4f}), delta = {diff:+.4f}")

    # Overall assessment: average rank across outcomes
    for outcome in OUTCOME_COLS:
        results_df[f'{outcome}_rank'] = results_df[f'{outcome}_r'].rank(ascending=False)

    results_df['avg_rank'] = results_df[[f'{o}_rank' for o in OUTCOME_COLS]].mean(axis=1)
    best_overall_idx = results_df['avg_rank'].idxmin()
    best_overall = results_df.loc[best_overall_idx]

    print()
    print(f"  Overall (average rank across all 4 outcomes):")
    print(f"    Best overall combo: {best_overall['combo']} (avg rank: {best_overall['avg_rank']:.1f})")
    current_avg_rank = results_df[results_df['is_current']]['avg_rank'].iloc[0]
    print(f"    Current combo rank: {current_row['combo']} (avg rank: {current_avg_rank:.1f})")

    if best_overall['is_current']:
        print(f"    -> Current weights are optimal (or tied for best). No change recommended.")
    else:
        # Check if difference is meaningful
        best_overall_r_avg = np.mean([best_overall[f'{o}_r'] for o in OUTCOME_COLS])
        current_r_avg = np.mean([current_row[f'{o}_r'] for o in OUTCOME_COLS])
        r_diff = best_overall_r_avg - current_r_avg
        print(f"    -> Best overall avg r: {best_overall_r_avg:.4f} vs current avg r: {current_r_avg:.4f} (delta: {r_diff:+.4f})")
        if abs(r_diff) < 0.01:
            print(f"    -> Difference is very small (<0.01). Current weights are fine.")
        elif abs(r_diff) < 0.02:
            print(f"    -> Difference is modest. Consider testing {best_overall['combo']} further.")
        else:
            print(f"    -> Meaningful difference. Consider switching to {best_overall['combo']}.")

    print()

    return results_df


# ─────────────────────────────────────────────
# Per-player detail table using CURRENT weights
# ─────────────────────────────────────────────

def print_player_detail(df, position_label, current_combo,
                        dc_col='dc_score', prod_col='production_score_filled',
                        ath_col='athletic_score'):
    """Print a per-player detail table with SLAP components and outcomes."""
    w_dc, w_prod, w_ath = [x / 100.0 for x in current_combo]
    df = df.copy()
    df['slap'] = w_dc * df[dc_col] + w_prod * df[prod_col] + w_ath * df[ath_col]
    df_sorted = df.sort_values('slap', ascending=False)

    print(f"\n{'=' * 140}")
    print(f"{position_label} PER-PLAYER DETAIL (current weights {current_combo[0]}/{current_combo[1]}/{current_combo[2]}), sorted by SLAP")
    print(f"{'=' * 140}")
    print(f"{'Player':<25} {'Year':>4} {'Pick':>4} {'DC':>6} {'Prod':>6} {'Ath':>6} {'SLAP':>6} {'Szns':>4} {'hit24':>5} {'hit12':>5} {'3yr_ppg':>8} {'car_ppg':>8}")
    print("-" * 140)

    for _, r in df_sorted.iterrows():
        name = str(r['player_name'])[:24]
        yr = int(r['draft_year'])
        pick = int(r['pick'])
        dc = r[dc_col]
        prod = r[prod_col]
        ath = r[ath_col]
        slap = r['slap']
        szns = int(r['seasons_played']) if pd.notna(r.get('seasons_played')) else 0
        h24 = int(r['hit24']) if pd.notna(r.get('hit24')) else '-'
        h12 = int(r['hit12']) if pd.notna(r.get('hit12')) else '-'
        ppg3 = f"{r['first_3yr_ppg']:.1f}" if pd.notna(r.get('first_3yr_ppg')) else '-'
        ppgc = f"{r['career_ppg']:.1f}" if pd.notna(r.get('career_ppg')) else '-'
        print(f"{name:<25} {yr:>4} {pick:>4} {dc:>6.1f} {prod:>6.1f} {ath:>6.1f} {slap:>6.1f} {szns:>4} {h24:>5} {h12:>5} {ppg3:>8} {ppgc:>8}")

    print(f"\nTotal: {len(df_sorted)} players")
    print()


# ─────────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────────

print()
print("#" * 120)
print("#  SLAP MODEL V3 - ANALYSIS TEST 1: WEIGHT OPTIMIZATION")
print("#  Testing all weight combos against 4 NFL outcome metrics")
print("#" * 120)
print()

# WR Analysis
wr_results = run_optimization(
    wr_merged, wr_combos, CURRENT_WR, "WR",
    dc_col='dc_score', prod_col='breakout_score', ath_col='athletic_score'
)

# RB Analysis
rb_results = run_optimization(
    rb_merged, rb_combos, CURRENT_RB, "RB",
    dc_col='dc_score', prod_col='production_score_filled', ath_col='athletic_score'
)

# Per-player detail tables
print_player_detail(
    wr_merged, "WR", CURRENT_WR,
    dc_col='dc_score', prod_col='breakout_score', ath_col='athletic_score'
)

print_player_detail(
    rb_merged, "RB", CURRENT_RB,
    dc_col='dc_score', prod_col='production_score_filled', ath_col='athletic_score'
)

print("Analysis complete.")
