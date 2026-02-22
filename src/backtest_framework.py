"""
SLAP Score V3 - Backtesting Framework
======================================

Calculates SLAP scores for all historical WRs (2015-2025) and RBs (2015-2025),
then evaluates how well those scores predicted actual NFL outcomes.

Data sources:
  - wr_backtest_expanded_final.csv   (WR college profiles + breakout ages)
  - wr_dominator_complete.csv        (dominator % for continuous breakout scoring)
  - rb_backtest_with_receiving.csv   (RB college profiles + receiving production)
  - backtest_hit_rates_rebuilt.csv   (authoritative NFL outcomes - hit24/hit12)

NFL outcomes are joined from backtest_hit_rates_rebuilt.csv, which is the
single source of truth.  The individual WR/RB files may have stale hit24 values.

Output:
  Table 1: Hit24 rate by SLAP tier (WR, RB, Combined)
  Table 2: SLAP vs Draft Capital Only — does SLAP beat just using pick order?
  Table 3: Biggest misses — high-SLAP busts and low-SLAP hits
  Table 4: Year-by-year breakdown for 2015-2023 (70+ vs <60)
  Separate "early returns" section for 2024-2025 draft classes.
  File: output/backtest_results.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# FORMULAS — must match recalculate_all_slap_new_dc.py exactly
# ============================================================================

# Weights
WR_W_DC   = 0.65
WR_W_PROD = 0.20
WR_W_RAS  = 0.15

RB_W_DC   = 0.50
RB_W_PROD = 0.35
RB_W_RAS  = 0.15

# Position averages for missing-data imputation
WR_AVG_RAS  = 68.9   # 0-100 scale (RAS * 10)
RB_AVG_PROD = 30.0
RB_AVG_RAS  = 66.5


def dc_score(pick):
    """Draft Capital: DC = 100 - 2.40 * (pick^0.62 - 1), clamped 0-100."""
    return max(0.0, min(100.0, 100 - 2.40 * (pick ** 0.62 - 1)))


def wr_breakout(breakout_age, dominator_pct):
    """Continuous breakout scoring: age-tier base + dominator tiebreaker."""
    if breakout_age is None or pd.isna(breakout_age):
        if dominator_pct is not None and pd.notna(dominator_pct):
            return min(35, 15 + dominator_pct * 1.0)
        return 25.0

    tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base = tiers.get(int(breakout_age), 20)

    bonus = 0.0
    if dominator_pct is not None and pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
    return min(base + bonus, 99.9)


def rb_prod(rec_yards, team_pass_att, draft_age):
    """RB receiving production with continuous age weighting, scaled /1.75."""
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return None
    if pd.isna(draft_age):
        draft_age = 22
    season_age = draft_age - 1
    age_wt = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
    raw = (rec_yards / team_pass_att) * age_wt * 100
    return min(99.9, max(0.0, raw / 1.75))


TIER_ORDER = ['90+', '80-89', '70-79', '60-69', '50-59', '<50']

def assign_tier(score):
    if score >= 90: return '90+'
    if score >= 80: return '80-89'
    if score >= 70: return '70-79'
    if score >= 60: return '60-69'
    if score >= 50: return '50-59'
    return '<50'


# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 90)
print("SLAP SCORE V3 - BACKTESTING FRAMEWORK")
print("=" * 90)

wr = pd.read_csv('data/wr_backtest_expanded_final.csv')
wr_dom = pd.read_csv('data/wr_dominator_complete.csv')
wr = wr.merge(
    wr_dom[['player_name', 'draft_year', 'dominator_pct']],
    on=['player_name', 'draft_year'], how='left'
)

# FIX: When dominator_pct is missing from the merge (e.g. all 2025 WRs, plus
# ~15 others where wr_dominator_complete.csv has no entry), fall back to
# peak_dominator from the WR backtest file itself.  The two columns hold the
# same value when both exist (verified: 304/305 match within 0.01).
missing_dom = wr['dominator_pct'].isna() & wr['peak_dominator'].notna()
wr.loc[missing_dom, 'dominator_pct'] = wr.loc[missing_dom, 'peak_dominator']
print(f"  Dominator fallback: filled {missing_dom.sum()} WRs from peak_dominator")

rb = pd.read_csv('data/rb_backtest_with_receiving.csv')
hr = pd.read_csv('data/backtest_hit_rates_rebuilt.csv')

print(f"\nLoaded: {len(wr)} WRs, {len(rb)} RBs, {len(hr)} hit-rate records")


# ============================================================================
# CALCULATE SLAP SCORES
# ============================================================================

print("Calculating SLAP scores...")

# WR
wr['dc'] = wr['pick'].apply(dc_score)
wr['prod'] = wr.apply(
    lambda r: wr_breakout(r['breakout_age'], r.get('dominator_pct')), axis=1)
wr['ras'] = (wr['RAS'] * 10).fillna(WR_AVG_RAS)
wr['slap'] = WR_W_DC * wr['dc'] + WR_W_PROD * wr['prod'] + WR_W_RAS * wr['ras']
wr['position'] = 'WR'

# RB
rb['dc'] = rb['pick'].apply(dc_score)
rb['prod_raw'] = rb.apply(
    lambda r: rb_prod(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb['prod'] = rb['prod_raw'].fillna(RB_AVG_PROD)
rb['ras'] = (rb['RAS'] * 10).fillna(RB_AVG_RAS)
rb['slap'] = RB_W_DC * rb['dc'] + RB_W_PROD * rb['prod'] + RB_W_RAS * rb['ras']
rb['position'] = 'RB'


# ============================================================================
# JOIN AUTHORITATIVE NFL OUTCOMES
# ============================================================================

# FIX: Normalize known name mismatches so backtest files join to hit_rates.
# "Luther Burden III" in WR backtest vs "Luther Burden" in hit_rates.
NAME_MAP = {
    'Luther Burden III': 'Luther Burden',
}

# Add a join_name column to each dataframe for matching
for df in [wr, rb]:
    df['join_name'] = df['player_name'].map(NAME_MAP).fillna(df['player_name'])
hr['join_name'] = hr['player_name'].map(NAME_MAP).fillna(hr['player_name'])

hr_key = hr.set_index(['join_name', 'draft_year'])

def get_hr(row, field):
    key = (row['join_name'], row['draft_year'])
    if key in hr_key.index:
        return hr_key.at[key, field]
    return np.nan

for df in [wr, rb]:
    df['hit24']    = df.apply(lambda r: get_hr(r, 'hit24'), axis=1)
    df['hit12']    = df.apply(lambda r: get_hr(r, 'hit12'), axis=1)
    df['best_ppr'] = df.apply(lambda r: get_hr(r, 'best_ppr'), axis=1)

wr_matched = wr['hit24'].notna().sum()
rb_matched = rb['hit24'].notna().sum()
print(f"  WR hit-rate joins: {wr_matched}/{len(wr)}")
print(f"  RB hit-rate joins: {rb_matched}/{len(rb)}")
if wr_matched > 230:
    print(f"  (name normalization recovered {wr_matched - 230} previously-dropped WRs)")


# ============================================================================
# BUILD COMBINED FRAME
# ============================================================================

cols = ['player_name', 'position', 'draft_year', 'pick', 'round', 'college',
        'dc', 'prod', 'ras', 'slap', 'hit24', 'hit12', 'best_ppr']
all_df = pd.concat([wr[cols], rb[cols]], ignore_index=True)
all_df['slap_tier'] = all_df['slap'].apply(assign_tier)
all_df['dc_tier']   = all_df['dc'].apply(assign_tier)

# Drop any rows without NFL outcome data (shouldn't happen)
all_df = all_df[all_df['hit24'].notna()].copy()
all_df['hit24'] = all_df['hit24'].astype(int)

# Splits
main  = all_df[all_df['draft_year'] <= 2023].copy()
early = all_df[all_df['draft_year'] >= 2024].copy()

print(f"\nMain backtest  (2015-2023): {len(main)} players  "
      f"({len(main[main['position']=='WR'])} WR, {len(main[main['position']=='RB'])} RB)")
print(f"Early returns  (2024-2025): {len(early)} players")


# ============================================================================
# HELPER: build tier summary
# ============================================================================

def tier_summary(df, tier_col='slap_tier'):
    rows = []
    for tier in TIER_ORDER:
        t = df[df[tier_col] == tier]
        n = len(t)
        hits = int(t['hit24'].sum()) if n > 0 else 0
        rate = hits / n * 100 if n > 0 else 0
        rows.append({'tier': tier, 'players': n, 'hits': hits, 'rate': rate})
    total_n = len(df)
    total_h = int(df['hit24'].sum())
    rows.append({'tier': 'TOTAL', 'players': total_n, 'hits': total_h,
                 'rate': total_h / total_n * 100 if total_n > 0 else 0})
    return rows


def print_tier_table(label, rows):
    print(f"\n  {label}")
    print(f"  {'Tier':<10} {'Players':>8} {'Hits':>6} {'Hit Rate':>10}")
    print(f"  {'-' * 38}")
    for r in rows:
        rate_str = f"{r['rate']:.1f}%" if r['players'] > 0 else '  n/a'
        print(f"  {r['tier']:<10} {r['players']:>8} {r['hits']:>6} {rate_str:>10}")


# ============================================================================
# TABLE 1: Hit24 Rate by SLAP Tier
# ============================================================================

print("\n" + "=" * 90)
print("TABLE 1: HIT24 RATE BY SLAP TIER (2015-2023)")
print("=" * 90)

for label, subset in [('WR', main[main['position']=='WR']),
                       ('RB', main[main['position']=='RB']),
                       ('COMBINED', main)]:
    print_tier_table(f"--- {label} ---", tier_summary(subset))


# ============================================================================
# TABLE 2: SLAP vs Draft Capital Only
# ============================================================================

print("\n" + "=" * 90)
print("TABLE 2: SLAP vs DRAFT CAPITAL ONLY (2015-2023)")
print("=" * 90)
print("\n  Does adding production + athletic data beat just using draft pick order?")

slap_rows = tier_summary(main, 'slap_tier')
dc_rows   = tier_summary(main, 'dc_tier')

print(f"\n  {'Tier':<10}  {'--- SLAP ---':^28}  {'--- DC Only ---':^28}")
print(f"  {'':10}  {'Players':>8} {'Hits':>6} {'Rate':>10}  {'Players':>8} {'Hits':>6} {'Rate':>10}")
print(f"  {'-' * 76}")
for s, d in zip(slap_rows, dc_rows):
    s_rate = f"{s['rate']:.1f}%" if s['players'] > 0 else '  n/a'
    d_rate = f"{d['rate']:.1f}%" if d['players'] > 0 else '  n/a'
    print(f"  {s['tier']:<10}  {s['players']:>8} {s['hits']:>6} {s_rate:>10}  "
          f"{d['players']:>8} {d['hits']:>6} {d_rate:>10}")

# Correlation comparison
valid = main.dropna(subset=['best_ppr'])
valid = valid[valid['best_ppr'] > 0]
if len(valid) > 20:
    r_slap, p_slap = spearmanr(valid['slap'], valid['best_ppr'])
    r_dc, p_dc     = spearmanr(valid['dc'],   valid['best_ppr'])
    print(f"\n  Spearman correlation with best NFL PPR season:")
    print(f"    SLAP:    r = {r_slap:.3f}  (p = {p_slap:.4f})")
    print(f"    DC only: r = {r_dc:.3f}  (p = {p_dc:.4f})")
    better = "SLAP" if r_slap > r_dc else "DC Only"
    print(f"    Advantage: {better}  (delta r = {abs(r_slap - r_dc):.3f})")

    for pos in ['WR', 'RB']:
        v = valid[valid['position'] == pos]
        if len(v) > 15:
            rs, _ = spearmanr(v['slap'], v['best_ppr'])
            rd, _ = spearmanr(v['dc'],   v['best_ppr'])
            print(f"    {pos}: SLAP r={rs:.3f}  vs  DC r={rd:.3f}  (delta {rs - rd:+.3f})")


# ============================================================================
# TABLE 3: Biggest Misses
# ============================================================================

print("\n" + "=" * 90)
print("TABLE 3: BIGGEST MISSES (2015-2023)")
print("=" * 90)

row_fmt = f"  {'Player':<25} {'Pos':>3} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'Best PPR':>9}"
row_sep = f"  {'-' * 60}"

# High-SLAP busts
busts = main[main['hit24'] == 0].nlargest(10, 'slap')
print(f"\n  Top 10 Highest-SLAP Busts (no Hit24)")
print(row_fmt)
print(row_sep)
for _, r in busts.iterrows():
    ppr = f"{r['best_ppr']:.1f}" if pd.notna(r['best_ppr']) else '-'
    print(f"  {r['player_name']:<25} {r['position']:>3} {int(r['draft_year']):>5} "
          f"{int(r['pick']):>5} {r['slap']:>6.1f} {ppr:>9}")

# Low-SLAP hits
sleepers = main[main['hit24'] == 1].nsmallest(10, 'slap')
print(f"\n  Top 10 Lowest-SLAP Hits (achieved Hit24)")
print(row_fmt)
print(row_sep)
for _, r in sleepers.iterrows():
    ppr = f"{r['best_ppr']:.1f}" if pd.notna(r['best_ppr']) else '-'
    print(f"  {r['player_name']:<25} {r['position']:>3} {int(r['draft_year']):>5} "
          f"{int(r['pick']):>5} {r['slap']:>6.1f} {ppr:>9}")


# ============================================================================
# TABLE 4: Year-by-Year Breakdown (2015-2023)
# ============================================================================

print("\n" + "=" * 90)
print("TABLE 4: YEAR-BY-YEAR BREAKDOWN (2015-2023)")
print("=" * 90)
print("\n  SLAP 70+ vs SLAP <60 hit rates by draft class")

print(f"\n  {'Year':>6}  {'--- SLAP 70+ ---':^26}  {'--- SLAP <60 ---':^26}")
print(f"  {'':>6}  {'Players':>8} {'Hits':>6} {'Rate':>8}  {'Players':>8} {'Hits':>6} {'Rate':>8}")
print(f"  {'-' * 68}")

for year in range(2015, 2024):
    yr = main[main['draft_year'] == year]
    top = yr[yr['slap'] >= 70]
    bot = yr[yr['slap'] < 60]

    t_n, t_h = len(top), int(top['hit24'].sum())
    b_n, b_h = len(bot), int(bot['hit24'].sum())
    t_rate = f"{t_h/t_n*100:.0f}%" if t_n > 0 else 'n/a'
    b_rate = f"{b_h/b_n*100:.0f}%" if b_n > 0 else 'n/a'

    print(f"  {year:>6}  {t_n:>8} {t_h:>6} {t_rate:>8}  {b_n:>8} {b_h:>6} {b_rate:>8}")

# Totals
top_all = main[main['slap'] >= 70]
bot_all = main[main['slap'] < 60]
ta_n, ta_h = len(top_all), int(top_all['hit24'].sum())
ba_n, ba_h = len(bot_all), int(bot_all['hit24'].sum())
print(f"  {'-' * 68}")
print(f"  {'TOTAL':>6}  {ta_n:>8} {ta_h:>6} {ta_h/ta_n*100:>7.0f}%  "
      f"{ba_n:>8} {ba_h:>6} {ba_h/ba_n*100:>7.0f}%")


# ============================================================================
# EARLY RETURNS: 2024-2025
# ============================================================================

print("\n" + "=" * 90)
print("EARLY RETURNS: 2024-2025 DRAFT CLASSES (1 NFL season each)")
print("=" * 90)
print("  Included for reference only — too early to judge careers.\n")

for year in [2024, 2025]:
    yr = early[early['draft_year'] == year]
    if len(yr) == 0:
        continue
    print_tier_table(f"--- {year} Draft Class ({len(yr)} players) ---",
                     tier_summary(yr))


# ============================================================================
# SAVE RESULTS
# ============================================================================

export = all_df.copy()
export['delta_vs_dc'] = export['slap'] - export['dc']
export = export.sort_values(['draft_year', 'slap'], ascending=[True, False])
export.to_csv('output/backtest_results.csv', index=False)

print(f"\nSaved: output/backtest_results.csv ({len(export)} players)")
print("\n" + "=" * 90)
print("BACKTEST COMPLETE")
print("=" * 90)
