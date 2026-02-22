"""
Analysis Test 4 of 5: Hit Threshold Comparison (hit24 vs hit12)
================================================================
Compare hit24 vs hit12 as predictive thresholds.
- Are both useful?
- Does the SLAP model predict one better than the other?
- Which better identifies NFL success?

Uses backtest data 2015-2025 for WRs and RBs.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FORMULAS (matching existing SLAP model exactly)
# ============================================================

def normalize_draft_capital(pick):
    """DC score using gentler curve formula."""
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))


def wr_breakout_score(breakout_age, dominator_pct):
    """WR production component based on breakout age + dominator tiebreaker."""
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
    """RB production component based on receiving yards / team pass attempts."""
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


def compute_slap_wr(pick, breakout_age, dominator_pct, ras, avg_ras=68.9):
    """Compute WR SLAP score: 65% DC / 20% Breakout / 15% RAS."""
    dc = normalize_draft_capital(pick)
    breakout = wr_breakout_score(breakout_age, dominator_pct)
    if pd.isna(ras):
        ath = avg_ras
    else:
        ath = ras * 10  # RAS is 0-10, scale to 0-100
    slap = dc * 0.65 + breakout * 0.20 + ath * 0.15
    return round(slap, 2)


def compute_slap_rb(pick, rec_yards, team_pass_att, draft_age, ras, avg_ras=66.5):
    """Compute RB SLAP score: 50% DC / 35% Production / 15% Speed."""
    dc = normalize_draft_capital(pick)
    prod = rb_production_score(rec_yards, team_pass_att, draft_age)
    if prod is None:
        prod = 0  # fallback for missing data
    if pd.isna(ras):
        ath = avg_ras
    else:
        ath = ras * 10  # RAS is 0-10, scale to 0-100
    slap = dc * 0.50 + prod * 0.35 + ath * 0.15
    return round(slap, 2)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("ANALYSIS 4: HIT THRESHOLD COMPARISON (hit24 vs hit12)")
print("=" * 80)

# Load outcomes (has first_3yr_ppg, career_ppg, seasons_played)
outcomes = pd.read_csv('/home/user/SlapModelV3/data/backtest_outcomes_complete.csv')

# Load WR data
wr_data = pd.read_csv('/home/user/SlapModelV3/data/wr_backtest_expanded_final.csv')

# Load RB data
rb_data = pd.read_csv('/home/user/SlapModelV3/data/rb_backtest_with_receiving.csv')

print(f"\nData loaded:")
print(f"  Outcomes file: {len(outcomes)} players")
print(f"  WR backtest:   {len(wr_data)} players")
print(f"  RB backtest:   {len(rb_data)} players")

# ============================================================
# MERGE & COMPUTE SLAP SCORES
# ============================================================

# --- WRs ---
# Merge WR data with outcomes to get first_3yr_ppg, career_ppg, seasons_played
wr_merged = wr_data.merge(
    outcomes[['player_name', 'draft_year', 'first_3yr_ppg', 'career_ppg',
              'seasons_played', 'first_3yr_games']],
    on=['player_name', 'draft_year'],
    how='left',
    suffixes=('', '_outcomes')
)

# Use seasons_played from outcomes if available, else from wr_data
if 'seasons_played_outcomes' in wr_merged.columns:
    wr_merged['seasons_played'] = wr_merged['seasons_played_outcomes'].fillna(wr_merged['seasons_played'])
    wr_merged.drop(columns=['seasons_played_outcomes'], inplace=True, errors='ignore')

# Compute WR SLAP scores
wr_merged['slap'] = wr_merged.apply(
    lambda r: compute_slap_wr(r['pick'], r['breakout_age'], r['peak_dominator'], r['RAS']),
    axis=1
)
wr_merged['position'] = 'WR'
if 'round' not in wr_merged.columns:
    wr_merged['round'] = np.ceil(wr_merged['pick'] / 32).astype(int)

print(f"\nWR SLAP scores computed: {len(wr_merged)} players")
print(f"  SLAP range: {wr_merged['slap'].min():.1f} - {wr_merged['slap'].max():.1f}")
print(f"  Mean SLAP:  {wr_merged['slap'].mean():.1f}")

# --- RBs ---
# Merge RB data with outcomes
rb_merged = rb_data.merge(
    outcomes[['player_name', 'draft_year', 'first_3yr_ppg', 'career_ppg',
              'seasons_played', 'first_3yr_games']],
    on=['player_name', 'draft_year'],
    how='left',
    suffixes=('', '_outcomes')
)

# Use seasons_played from outcomes if available
if 'seasons_played_outcomes' in rb_merged.columns:
    rb_merged['seasons_played'] = rb_merged['seasons_played_outcomes'].fillna(
        rb_merged.get('seasons_played', np.nan)
    )
    rb_merged.drop(columns=['seasons_played_outcomes'], inplace=True, errors='ignore')

# Compute RB SLAP scores
rb_merged['slap'] = rb_merged.apply(
    lambda r: compute_slap_rb(r['pick'], r['rec_yards'], r['team_pass_att'], r['age'], r['RAS']),
    axis=1
)
rb_merged['position'] = 'RB'
if 'round' not in rb_merged.columns:
    rb_merged['round'] = np.ceil(rb_merged['pick'] / 32).astype(int)

print(f"\nRB SLAP scores computed: {len(rb_merged)} players")
print(f"  SLAP range: {rb_merged['slap'].min():.1f} - {rb_merged['slap'].max():.1f}")
print(f"  Mean SLAP:  {rb_merged['slap'].mean():.1f}")

# Combine for total analyses
all_players = pd.concat([
    wr_merged[['player_name', 'draft_year', 'pick', 'round', 'position', 'slap',
               'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg', 'seasons_played',
               'first_3yr_games']],
    rb_merged[['player_name', 'draft_year', 'pick', 'round', 'position', 'slap',
               'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg', 'seasons_played',
               'first_3yr_games']]
], ignore_index=True)

print(f"\nTotal players for analysis: {len(all_players)}")
print(f"  WRs: {len(all_players[all_players['position']=='WR'])}")
print(f"  RBs: {len(all_players[all_players['position']=='RB'])}")


# ============================================================
# PART 1: BASELINE HIT RATES
# ============================================================

print("\n" + "=" * 80)
print("PART 1: BASELINE HIT RATES")
print("=" * 80)
print("\nHow common are hit24 and hit12 by position, round, and year?")

# --- By Position ---
print("\n--- Hit Rates by Position ---")
print(f"{'Position':<10} {'N':>5} {'hit24':>8} {'hit24%':>8} {'hit12':>8} {'hit12%':>8} {'Avg Seasons':>12}")
print("-" * 65)
for pos in ['WR', 'RB']:
    subset = all_players[all_players['position'] == pos]
    n = len(subset)
    h24 = subset['hit24'].sum()
    h12 = subset['hit12'].sum()
    avg_szn = subset['seasons_played'].mean()
    print(f"{pos:<10} {n:>5} {int(h24):>8} {h24/n*100:>7.1f}% {int(h12):>8} {h12/n*100:>7.1f}% {avg_szn:>11.1f}")

total = all_players
n = len(total)
h24 = total['hit24'].sum()
h12 = total['hit12'].sum()
avg_szn = total['seasons_played'].mean()
print(f"{'TOTAL':<10} {n:>5} {int(h24):>8} {h24/n*100:>7.1f}% {int(h12):>8} {h12/n*100:>7.1f}% {avg_szn:>11.1f}")

# --- By Draft Round ---
print("\n--- Hit Rates by Draft Round (All Positions) ---")
print(f"{'Round':<8} {'N':>5} {'hit24':>8} {'hit24%':>8} {'hit12':>8} {'hit12%':>8} {'Avg Seasons':>12}")
print("-" * 65)
for rd in sorted(all_players['round'].unique()):
    subset = all_players[all_players['round'] == rd]
    n = len(subset)
    h24 = subset['hit24'].sum()
    h12 = subset['hit12'].sum()
    avg_szn = subset['seasons_played'].mean()
    print(f"Rd {int(rd):<4} {n:>5} {int(h24):>8} {h24/n*100:>7.1f}% {int(h12):>8} {h12/n*100:>7.1f}% {avg_szn:>11.1f}")

# --- By Position and Round ---
for pos in ['WR', 'RB']:
    print(f"\n--- {pos} Hit Rates by Draft Round ---")
    print(f"{'Round':<8} {'N':>5} {'hit24':>8} {'hit24%':>8} {'hit12':>8} {'hit12%':>8} {'Avg Seasons':>12}")
    print("-" * 65)
    pos_df = all_players[all_players['position'] == pos]
    for rd in sorted(pos_df['round'].unique()):
        subset = pos_df[pos_df['round'] == rd]
        n = len(subset)
        h24 = subset['hit24'].sum()
        h12 = subset['hit12'].sum()
        avg_szn = subset['seasons_played'].mean()
        print(f"Rd {int(rd):<4} {n:>5} {int(h24):>8} {h24/n*100:>7.1f}% {int(h12):>8} {h12/n*100:>7.1f}% {avg_szn:>11.1f}")

# --- By Draft Year ---
print("\n--- Hit Rates by Draft Year (All Positions) ---")
print(f"{'Year':<8} {'N':>5} {'hit24':>8} {'hit24%':>8} {'hit12':>8} {'hit12%':>8} {'Avg Seasons':>12}")
print("-" * 65)
for yr in sorted(all_players['draft_year'].unique()):
    subset = all_players[all_players['draft_year'] == yr]
    n = len(subset)
    h24 = subset['hit24'].sum()
    h12 = subset['hit12'].sum()
    avg_szn = subset['seasons_played'].mean()
    print(f"{int(yr):<8} {n:>5} {int(h24):>8} {h24/n*100:>7.1f}% {int(h12):>8} {h12/n*100:>7.1f}% {avg_szn:>11.1f}")

# --- By Draft Year and Position ---
for pos in ['WR', 'RB']:
    print(f"\n--- {pos} Hit Rates by Draft Year ---")
    print(f"{'Year':<8} {'N':>5} {'hit24':>8} {'hit24%':>8} {'hit12':>8} {'hit12%':>8} {'Avg Seasons':>12}")
    print("-" * 65)
    pos_df = all_players[all_players['position'] == pos]
    for yr in sorted(pos_df['draft_year'].unique()):
        subset = pos_df[pos_df['draft_year'] == yr]
        n = len(subset)
        h24 = subset['hit24'].sum()
        h12 = subset['hit12'].sum()
        avg_szn = subset['seasons_played'].mean()
        print(f"{int(yr):<8} {n:>5} {int(h24):>8} {h24/n*100:>7.1f}% {int(h12):>8} {h12/n*100:>7.1f}% {avg_szn:>11.1f}")


# ============================================================
# PART 2: SLAP CORRELATION WITH EACH METRIC
# ============================================================

print("\n" + "=" * 80)
print("PART 2: SLAP CORRELATION WITH hit24, hit12, first_3yr_ppg, career_ppg")
print("=" * 80)

def correlation_report(df, label):
    """Print correlations of SLAP with all outcome metrics."""
    print(f"\n--- {label} (N={len(df)}) ---")
    print(f"{'Metric':<20} {'r':>8} {'p-value':>12} {'Interpretation':<30}")
    print("-" * 70)

    metrics = [
        ('hit24', 'Binary: Top-24 at position'),
        ('hit12', 'Binary: Top-12 at position'),
        ('first_3yr_ppg', 'Continuous: First 3yr PPR/game'),
        ('career_ppg', 'Continuous: Career PPR/game'),
    ]

    for col, desc in metrics:
        valid = df[['slap', col]].dropna()
        if len(valid) < 5:
            print(f"{col:<20} {'N/A':>8} {'N/A':>12} Insufficient data (n={len(valid)})")
            continue
        if col in ['hit24', 'hit12']:
            r, p = stats.pointbiserialr(valid[col], valid['slap'])
        else:
            r, p = stats.pearsonr(valid['slap'], valid[col])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak" if abs(r) > 0.1 else "negligible"
        print(f"{col:<20} {r:>8.3f} {p:>12.6f} {strength} ({sig}) - {desc}")

# All players
correlation_report(all_players, "ALL PLAYERS")

# By position
for pos in ['WR', 'RB']:
    subset = all_players[all_players['position'] == pos]
    correlation_report(subset, f"{pos} ONLY")

# Also show DC-only correlation for comparison
print("\n--- COMPARISON: DC-only vs Full SLAP Correlations ---")
print(f"{'Metric':<20} {'Position':<8} {'DC r':>8} {'SLAP r':>8} {'Improvement':>12}")
print("-" * 60)
for pos in ['WR', 'RB']:
    subset = all_players[all_players['position'] == pos].copy()
    subset['dc_score'] = subset['pick'].apply(normalize_draft_capital)
    for col in ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']:
        valid = subset[['dc_score', 'slap', col]].dropna()
        if len(valid) < 5:
            continue
        if col in ['hit24', 'hit12']:
            r_dc, _ = stats.pointbiserialr(valid[col], valid['dc_score'])
            r_slap, _ = stats.pointbiserialr(valid[col], valid['slap'])
        else:
            r_dc, _ = stats.pearsonr(valid['dc_score'], valid[col])
            r_slap, _ = stats.pearsonr(valid['slap'], valid[col])
        diff = r_slap - r_dc
        direction = "+" if diff > 0 else ""
        print(f"{col:<20} {pos:<8} {r_dc:>8.3f} {r_slap:>8.3f} {direction}{diff:>11.3f}")


# ============================================================
# PART 3: SLAP TIER HIT RATES
# ============================================================

print("\n" + "=" * 80)
print("PART 3: SLAP TIER HIT RATES")
print("=" * 80)

def slap_tier(score):
    if score >= 90:
        return '90+'
    elif score >= 80:
        return '80-89'
    elif score >= 70:
        return '70-79'
    elif score >= 60:
        return '60-69'
    elif score >= 50:
        return '50-59'
    elif score >= 40:
        return '40-49'
    elif score >= 30:
        return '30-39'
    else:
        return '<30'

tier_order = ['90+', '80-89', '70-79', '60-69', '50-59', '40-49', '30-39', '<30']

all_players['slap_tier'] = all_players['slap'].apply(slap_tier)

def print_tier_table(df, label):
    print(f"\n--- {label} ---")
    print(f"{'Tier':<10} {'N':>5} {'hit24':>7} {'hit24%':>8} {'hit12':>7} {'hit12%':>8} "
          f"{'Avg 3yr PPG':>12} {'Avg Career PPG':>15} {'Avg Seasons':>12}")
    print("-" * 100)

    for tier in tier_order:
        subset = df[df['slap_tier'] == tier]
        n = len(subset)
        if n == 0:
            continue
        h24 = subset['hit24'].sum()
        h12 = subset['hit12'].sum()
        avg_3yr = subset['first_3yr_ppg'].mean()
        avg_car = subset['career_ppg'].mean()
        avg_szn = subset['seasons_played'].mean()
        print(f"{tier:<10} {n:>5} {int(h24):>7} {h24/n*100:>7.1f}% {int(h12):>7} {h12/n*100:>7.1f}% "
              f"{avg_3yr:>12.2f} {avg_car:>15.2f} {avg_szn:>12.1f}")

    # Totals
    n = len(df)
    h24 = df['hit24'].sum()
    h12 = df['hit12'].sum()
    avg_3yr = df['first_3yr_ppg'].mean()
    avg_car = df['career_ppg'].mean()
    avg_szn = df['seasons_played'].mean()
    print("-" * 100)
    print(f"{'TOTAL':<10} {n:>5} {int(h24):>7} {h24/n*100:>7.1f}% {int(h12):>7} {h12/n*100:>7.1f}% "
          f"{avg_3yr:>12.2f} {avg_car:>15.2f} {avg_szn:>12.1f}")

print_tier_table(all_players, "ALL PLAYERS")
print_tier_table(all_players[all_players['position'] == 'WR'], "WR ONLY")
print_tier_table(all_players[all_players['position'] == 'RB'], "RB ONLY")


# ============================================================
# PART 4: CROSS-TABULATION (hit24 vs hit12)
# ============================================================

print("\n" + "=" * 80)
print("PART 4: CROSS-TABULATION (hit24 vs hit12)")
print("=" * 80)
print("\n2x2 Table: hit24 (rows) vs hit12 (columns)")

def cross_tab(df, label):
    print(f"\n--- {label} (N={len(df)}) ---")

    # Create crosstab
    ct = pd.crosstab(df['hit24'], df['hit12'], margins=True)
    print("\nCounts:")
    print(ct.to_string())

    # Detailed quadrant analysis
    quadrants = [
        (1, 1, "ELITE: hit24=1 AND hit12=1"),
        (0, 1, "GOOD NOT GREAT: hit24=0 AND hit12=1"),
        (1, 0, "ANOMALY: hit24=1 AND hit12=0 (should be rare)"),
        (0, 0, "BUST: hit24=0 AND hit12=0"),
    ]

    print(f"\n{'Quadrant':<50} {'N':>5} {'%':>7} {'Avg 3yr PPG':>12} {'Avg Career PPG':>15} "
          f"{'Avg SLAP':>10} {'Avg Seasons':>12}")
    print("-" * 120)

    for h24, h12, name in quadrants:
        subset = df[(df['hit24'] == h24) & (df['hit12'] == h12)]
        n = len(subset)
        pct = n / len(df) * 100
        avg_3yr = subset['first_3yr_ppg'].mean() if n > 0 else 0
        avg_car = subset['career_ppg'].mean() if n > 0 else 0
        avg_slap = subset['slap'].mean() if n > 0 else 0
        avg_szn = subset['seasons_played'].mean() if n > 0 else 0
        print(f"{name:<50} {n:>5} {pct:>6.1f}% {avg_3yr:>12.2f} {avg_car:>15.2f} "
              f"{avg_slap:>10.1f} {avg_szn:>12.1f}")

    # Show anomaly cases (hit24=1 but hit12=0) if any exist
    anomalies = df[(df['hit24'] == 1) & (df['hit12'] == 0)]
    if len(anomalies) > 0:
        print(f"\n  ** ANOMALY CASES (hit24=1 but hit12=0): **")
        for _, row in anomalies.iterrows():
            print(f"     {row['player_name']} ({row['position']}, {int(row['draft_year'])}, "
                  f"pick {int(row['pick'])}): SLAP={row['slap']:.1f}, "
                  f"3yr PPG={row.get('first_3yr_ppg', 'N/A')}, "
                  f"career PPG={row.get('career_ppg', 'N/A')}, "
                  f"seasons={row.get('seasons_played', 'N/A')}")
    else:
        print(f"\n  No anomalies found (all hit24 players are also hit12). Thresholds are consistent.")

cross_tab(all_players, "ALL PLAYERS")
cross_tab(all_players[all_players['position'] == 'WR'], "WR ONLY")
cross_tab(all_players[all_players['position'] == 'RB'], "RB ONLY")


# ============================================================
# PART 5: PREDICTIVE ACCURACY AT DIFFERENT SLAP CUTOFFS
# ============================================================

print("\n" + "=" * 80)
print("PART 5: PREDICTIVE ACCURACY AT DIFFERENT SLAP CUTOFFS")
print("=" * 80)
print("\nPrecision = What % of players ABOVE the cutoff are hits?")
print("Recall    = What % of ALL hits are ABOVE the cutoff?")
print("F1        = Harmonic mean of precision and recall (balanced metric)")

cutoffs = [90, 80, 70, 60, 50]

def precision_recall_table(df, label):
    print(f"\n--- {label} (N={len(df)}) ---")

    total_hit24 = df['hit24'].sum()
    total_hit12 = df['hit12'].sum()
    print(f"    Total hit24: {int(total_hit24)} ({total_hit24/len(df)*100:.1f}%)")
    print(f"    Total hit12: {int(total_hit12)} ({total_hit12/len(df)*100:.1f}%)")

    print(f"\n{'Cutoff':<10} {'N Above':>8} {'--- hit24 ---':>30}{'':>5}{'--- hit12 ---':>30}")
    print(f"{'':>10} {'':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}{'':>5}{'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 90)

    for cutoff in cutoffs:
        above = df[df['slap'] >= cutoff]
        n_above = len(above)

        if n_above == 0:
            print(f"SLAP>={cutoff:<4} {n_above:>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}{'':>5}{'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue

        # hit24
        tp24 = above['hit24'].sum()
        prec24 = tp24 / n_above if n_above > 0 else 0
        rec24 = tp24 / total_hit24 if total_hit24 > 0 else 0
        f1_24 = 2 * prec24 * rec24 / (prec24 + rec24) if (prec24 + rec24) > 0 else 0

        # hit12
        tp12 = above['hit12'].sum()
        prec12 = tp12 / n_above if n_above > 0 else 0
        rec12 = tp12 / total_hit12 if total_hit12 > 0 else 0
        f1_12 = 2 * prec12 * rec12 / (prec12 + rec12) if (prec12 + rec12) > 0 else 0

        print(f"SLAP>={cutoff:<4} {n_above:>8} {prec24*100:>7.1f}% {rec24*100:>7.1f}% {f1_24:>8.3f}"
              f"{'':>5}{prec12*100:>7.1f}% {rec12*100:>7.1f}% {f1_12:>8.3f}")

    # Also show detailed breakdown for each cutoff
    print(f"\n  Detailed breakdown (counts):")
    print(f"  {'Cutoff':<10} {'N Above':>8} {'hit24 Hits':>11} {'hit24 Misses':>13} "
          f"{'hit12 Hits':>11} {'hit12 Misses':>13} {'Avg Seasons':>12}")
    print("  " + "-" * 80)

    for cutoff in cutoffs:
        above = df[df['slap'] >= cutoff]
        n_above = len(above)
        if n_above == 0:
            continue
        tp24 = int(above['hit24'].sum())
        fp24 = n_above - tp24
        tp12 = int(above['hit12'].sum())
        fp12 = n_above - tp12
        avg_szn = above['seasons_played'].mean()
        print(f"  SLAP>={cutoff:<4} {n_above:>8} {tp24:>11} {fp24:>13} "
              f"{tp12:>11} {fp12:>13} {avg_szn:>12.1f}")

precision_recall_table(all_players, "ALL PLAYERS")
precision_recall_table(all_players[all_players['position'] == 'WR'], "WR ONLY")
precision_recall_table(all_players[all_players['position'] == 'RB'], "RB ONLY")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY: KEY FINDINGS")
print("=" * 80)

# Compute key stats for summary
wr_df = all_players[all_players['position'] == 'WR']
rb_df = all_players[all_players['position'] == 'RB']

# Correlations for summary
for pos, df_pos in [('WR', wr_df), ('RB', rb_df)]:
    valid_24 = df_pos[['slap', 'hit24']].dropna()
    valid_12 = df_pos[['slap', 'hit12']].dropna()
    valid_3yr = df_pos[['slap', 'first_3yr_ppg']].dropna()
    valid_car = df_pos[['slap', 'career_ppg']].dropna()

    r24, p24 = stats.pointbiserialr(valid_24['hit24'], valid_24['slap']) if len(valid_24) > 2 else (0, 1)
    r12, p12 = stats.pointbiserialr(valid_12['hit12'], valid_12['slap']) if len(valid_12) > 2 else (0, 1)
    r3yr, p3yr = stats.pearsonr(valid_3yr['slap'], valid_3yr['first_3yr_ppg']) if len(valid_3yr) > 2 else (0, 1)
    rcar, pcar = stats.pearsonr(valid_car['slap'], valid_car['career_ppg']) if len(valid_car) > 2 else (0, 1)

    print(f"\n{pos} Summary:")
    print(f"  SLAP-hit24 correlation:      r={r24:.3f} (p={p24:.4f})")
    print(f"  SLAP-hit12 correlation:      r={r12:.3f} (p={p12:.4f})")
    print(f"  SLAP-first_3yr_ppg:          r={r3yr:.3f} (p={p3yr:.4f})")
    print(f"  SLAP-career_ppg:             r={rcar:.3f} (p={pcar:.4f})")
    better = "hit24" if abs(r24) > abs(r12) else "hit12"
    print(f"  --> SLAP predicts {better} better for {pos}s")

# Cross-tab summary
all_both = len(all_players[(all_players['hit24']==1) & (all_players['hit12']==1)])
all_12only = len(all_players[(all_players['hit24']==0) & (all_players['hit12']==1)])
all_24only = len(all_players[(all_players['hit24']==1) & (all_players['hit12']==0)])
all_neither = len(all_players[(all_players['hit24']==0) & (all_players['hit12']==0)])

print(f"\nCross-tab Summary (All players, N={len(all_players)}):")
print(f"  Elite (hit24 AND hit12):           {all_both:>4} ({all_both/len(all_players)*100:.1f}%)")
print(f"  Good not great (hit12 only):       {all_12only:>4} ({all_12only/len(all_players)*100:.1f}%)")
print(f"  Anomaly (hit24 only):              {all_24only:>4} ({all_24only/len(all_players)*100:.1f}%)")
print(f"  Bust (neither):                    {all_neither:>4} ({all_neither/len(all_players)*100:.1f}%)")

# Best cutoff recommendation
print(f"\nRecommendation:")
print(f"  hit24 is more selective ({all_players['hit24'].mean()*100:.1f}% base rate) and identifies true difference-makers.")
print(f"  hit12 is more inclusive ({all_players['hit12'].mean()*100:.1f}% base rate) and captures a wider net of useful players.")
print(f"  Both thresholds appear nested: virtually all hit24 players are also hit12.")
print(f"  SLAP model should be evaluated against BOTH to show it identifies elites AND useful starters.")

print("\n" + "=" * 80)
print("END OF ANALYSIS 4")
print("=" * 80)
