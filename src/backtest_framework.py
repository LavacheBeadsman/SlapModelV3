"""
SLAP Score V3 — Backtest Framework
====================================
Calculates SLAP scores for every draft class (2015-2024) using the current
formulas, then measures how well those scores predicted actual NFL outcomes.

2025 draft class is shown separately as "early returns" (only one NFL season).

Outputs:
  1. Hit rate by SLAP tier (90+, 80-89, 70-79, 60-69, 50-59, below 50)
  2. SLAP accuracy vs draft-capital-only accuracy
  3. Biggest misses — high SLAP busts and low SLAP hits
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from config import FIRST_SEASON, LAST_COLLEGE_SEASON, CURRENT_DRAFT_CLASS

# ---------------------------------------------------------------------------
# Weights (from CLAUDE.md)
# ---------------------------------------------------------------------------
WR_W_DC   = 0.65
WR_W_PROD = 0.20
WR_W_RAS  = 0.15

RB_W_DC   = 0.50
RB_W_PROD = 0.35
RB_W_RAS  = 0.15

# Position averages used when data is missing (from recalculate_all_slap_new_dc.py)
WR_AVG_RAS  = 68.9
RB_AVG_PROD = 30.0
RB_AVG_RAS  = 66.5

# ---------------------------------------------------------------------------
# Core formulas — identical to recalculate_all_slap_new_dc.py
# ---------------------------------------------------------------------------

def dc_score(pick):
    """Draft Capital: DC = 100 - 2.40 * (pick^0.62 - 1), clamped 0-100."""
    return max(0.0, min(100.0, 100 - 2.40 * (pick ** 0.62 - 1)))


def wr_breakout_score(breakout_age, dominator_pct):
    """Continuous breakout scoring: age tier base + dominator tiebreaker."""
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


def rb_production_score(rec_yards, team_pass_att, draft_age):
    """RB receiving production with continuous age weighting, scaled by 1.75."""
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return None
    if pd.isna(draft_age):
        draft_age = 22

    season_age = draft_age - 1
    age_weight = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
    raw = (rec_yards / team_pass_att) * age_weight * 100
    return min(99.9, max(0.0, raw / 1.75))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

wr = pd.read_csv("data/wr_backtest_expanded_final.csv")
rb = pd.read_csv("data/rb_backtest_with_receiving.csv")

# Merge dominator percentages into WR data
wr_dom = pd.read_csv("data/wr_dominator_complete.csv")
wr = wr.merge(
    wr_dom[["player_name", "draft_year", "dominator_pct"]],
    on=["player_name", "draft_year"],
    how="left",
)

# ---------------------------------------------------------------------------
# Calculate SLAP scores
# ---------------------------------------------------------------------------

# ---- WR scores ----
wr["dc"]   = wr["pick"].apply(dc_score)
wr["prod"] = wr.apply(
    lambda r: wr_breakout_score(r["breakout_age"], r["dominator_pct"]), axis=1
)
wr["ras"]  = (wr["RAS"] * 10).fillna(WR_AVG_RAS)   # 0-10 → 0-100
wr["slap"] = WR_W_DC * wr["dc"] + WR_W_PROD * wr["prod"] + WR_W_RAS * wr["ras"]
wr["dc_only_score"] = wr["dc"]                       # baseline comparison
wr["position"] = "WR"

# ---- RB scores ----
rb["dc"]   = rb["pick"].apply(dc_score)
rb["prod_raw"] = rb.apply(
    lambda r: rb_production_score(r["rec_yards"], r["team_pass_att"], r["age"]),
    axis=1,
)
rb["prod"] = rb["prod_raw"].fillna(RB_AVG_PROD)
rb["ras"]  = (rb["RAS"] * 10).fillna(RB_AVG_RAS)
rb["slap"] = RB_W_DC * rb["dc"] + RB_W_PROD * rb["prod"] + RB_W_RAS * rb["ras"]
rb["dc_only_score"] = rb["dc"]
rb["position"] = "RB"

# ---------------------------------------------------------------------------
# Build a unified frame with the columns we need
# ---------------------------------------------------------------------------
keep = ["player_name", "position", "draft_year", "pick", "dc", "slap",
        "dc_only_score", "hit24", "hit12", "best_ppr"]

# RB file uses best_ppr; WR file also has best_ppr — grab college name too
wr["college_name"] = wr["college"]
rb["college_name"] = rb["college"]
keep_extra = keep + ["college_name"]

all_players = pd.concat(
    [wr[keep_extra], rb[keep_extra]],
    ignore_index=True,
)

# Separate 2025 early returns from the real backtest (2015-2024)
backtest = all_players[all_players["draft_year"] <= 2024].copy()
early_returns = all_players[all_players["draft_year"] == 2025].copy()

print("=" * 90)
print("SLAP SCORE V3 — BACKTEST FRAMEWORK")
print("=" * 90)
print(f"\nBacktest pool : {len(backtest)} players  (2015-2024)")
print(f"  WRs         : {len(backtest[backtest['position']=='WR'])}")
print(f"  RBs         : {len(backtest[backtest['position']=='RB'])}")
print(f"Early returns : {len(early_returns)} players  (2025, one NFL season)")

# ---------------------------------------------------------------------------
# 1.  HIT RATE BY SLAP TIER
# ---------------------------------------------------------------------------

tier_bins   = [0, 50, 60, 70, 80, 90, 101]
tier_labels = ["Below 50", "50-59", "60-69", "70-79", "80-89", "90+"]

def tier_table(df, score_col, label):
    """Print hit-rate table by tier for a given score column."""
    df = df.copy()
    df["tier"] = pd.cut(df[score_col], bins=tier_bins, labels=tier_labels,
                        right=False)
    grp = df.groupby("tier", observed=False).agg(
        players=("hit24", "size"),
        hits=("hit24", "sum"),
    )
    grp["hit_rate"] = (grp["hits"] / grp["players"] * 100).round(1)
    grp["hit_rate_str"] = grp.apply(
        lambda r: f"{r['hit_rate']:.1f}%" if r["players"] > 0 else "n/a", axis=1
    )

    print(f"\n{'':>2}{label}")
    print(f"{'':>2}{'Tier':<12} {'Players':>8} {'Hits':>6} {'Hit Rate':>10}")
    print(f"{'':>2}{'-'*40}")
    for tier in reversed(tier_labels):          # show highest tier first
        row = grp.loc[tier]
        print(f"{'':>2}{tier:<12} {int(row['players']):>8} "
              f"{int(row['hits']):>6} {row['hit_rate_str']:>10}")
    total_hits  = int(grp["hits"].sum())
    total_count = int(grp["players"].sum())
    print(f"{'':>2}{'TOTAL':<12} {total_count:>8} {total_hits:>6} "
          f"{total_hits/total_count*100:>9.1f}%")
    return grp


print("\n" + "=" * 90)
print("SECTION 1 — HIT RATE BY SLAP TIER  (Hit24 = top-24 PPR finish)")
print("=" * 90)

# Combined
grp_slap = tier_table(backtest, "slap", "SLAP Score — All positions (2015-2024)")
grp_dc   = tier_table(backtest, "dc_only_score", "DC-Only Baseline — All positions (2015-2024)")

# By position
print()
for pos in ["WR", "RB"]:
    subset = backtest[backtest["position"] == pos]
    tier_table(subset, "slap", f"SLAP Score — {pos}s (2015-2024)")
    tier_table(subset, "dc_only_score", f"DC-Only Baseline — {pos}s (2015-2024)")

# ---------------------------------------------------------------------------
# 2.  SLAP ACCURACY VS DC-ONLY ACCURACY
# ---------------------------------------------------------------------------

def accuracy_report(df, label):
    """Correlation + predictive stats for SLAP vs DC-only."""
    valid = df.dropna(subset=["slap", "dc_only_score", "hit24", "best_ppr"])

    r_slap, p_slap = pearsonr(valid["slap"], valid["best_ppr"])
    r_dc,   p_dc   = pearsonr(valid["dc_only_score"], valid["best_ppr"])

    # Hit-rate accuracy: does higher-scored player have higher hit rate?
    # Use top-half vs bottom-half split
    median_slap = valid["slap"].median()
    median_dc   = valid["dc_only_score"].median()

    top_slap_hr = valid[valid["slap"] >= median_slap]["hit24"].mean() * 100
    bot_slap_hr = valid[valid["slap"] <  median_slap]["hit24"].mean() * 100
    top_dc_hr   = valid[valid["dc_only_score"] >= median_dc]["hit24"].mean() * 100
    bot_dc_hr   = valid[valid["dc_only_score"] <  median_dc]["hit24"].mean() * 100

    # Year-by-year: for each draft class, does SLAP or DC better predict
    # which players hit?  Count wins.
    slap_wins = 0
    dc_wins   = 0
    ties      = 0
    for yr, ydf in valid.groupby("draft_year"):
        if len(ydf) < 5:
            continue
        rs, _ = pearsonr(ydf["slap"], ydf["best_ppr"])
        rd, _ = pearsonr(ydf["dc_only_score"], ydf["best_ppr"])
        if rs > rd + 0.01:
            slap_wins += 1
        elif rd > rs + 0.01:
            dc_wins += 1
        else:
            ties += 1

    print(f"\n  {label}")
    print(f"  {'─'*60}")
    print(f"  {'Metric':<40} {'SLAP':>10} {'DC-Only':>10}")
    print(f"  {'─'*60}")
    print(f"  {'Pearson r vs Best PPR':<40} {r_slap:>10.3f} {r_dc:>10.3f}")
    print(f"  {'p-value':<40} {p_slap:>10.4f} {p_dc:>10.4f}")
    print(f"  {'Top-half hit rate':<40} {top_slap_hr:>9.1f}% {top_dc_hr:>9.1f}%")
    print(f"  {'Bottom-half hit rate':<40} {bot_slap_hr:>9.1f}% {bot_dc_hr:>9.1f}%")
    print(f"  {'Spread (top - bottom)':<40} "
          f"{top_slap_hr-bot_slap_hr:>9.1f}% {top_dc_hr-bot_dc_hr:>9.1f}%")
    print(f"  {'Year-by-year wins (r > other + 0.01)':<40} "
          f"{slap_wins:>10} {dc_wins:>10}")
    if ties:
        print(f"  {'Ties (within 0.01)':<40} {ties:>10}")
    advantage = "SLAP" if r_slap > r_dc else "DC-Only"
    print(f"\n  → Overall advantage: {advantage} "
          f"(Δr = {abs(r_slap - r_dc):.3f})")


print("\n" + "=" * 90)
print("SECTION 2 — SLAP ACCURACY VS DRAFT-CAPITAL-ONLY")
print("=" * 90)

accuracy_report(backtest, "All positions (2015-2024)")

for pos in ["WR", "RB"]:
    subset = backtest[backtest["position"] == pos]
    accuracy_report(subset, f"{pos}s only (2015-2024)")

# ---------------------------------------------------------------------------
# 3.  BIGGEST MISSES
# ---------------------------------------------------------------------------

print("\n" + "=" * 90)
print("SECTION 3 — BIGGEST MISSES")
print("=" * 90)

# --- High SLAP busts: SLAP >= 70, but never hit24 ---
busts = (backtest[(backtest["slap"] >= 70) & (backtest["hit24"] == 0)]
         .sort_values("slap", ascending=False))

print(f"\n  HIGH-SLAP BUSTS  (SLAP ≥ 70, never Hit24)  —  {len(busts)} players")
print(f"  {'Player':<25} {'Pos':>3} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>5} "
      f"{'BestPPR':>8}")
print(f"  {'─'*65}")
for _, r in busts.head(25).iterrows():
    ppr = f"{r['best_ppr']:.1f}" if pd.notna(r["best_ppr"]) else "n/a"
    print(f"  {r['player_name']:<25} {r['position']:>3} {int(r['draft_year']):>5} "
          f"{int(r['pick']):>5} {r['slap']:>6.1f} {r['dc']:>5.1f} {ppr:>8}")

# --- Low SLAP hits: SLAP < 60, but hit24 ---
sleepers = (backtest[(backtest["slap"] < 60) & (backtest["hit24"] == 1)]
            .sort_values("slap", ascending=True))

print(f"\n  LOW-SLAP HITS  (SLAP < 60, hit24 = 1)  —  {len(sleepers)} players")
print(f"  {'Player':<25} {'Pos':>3} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>5} "
      f"{'BestPPR':>8}")
print(f"  {'─'*65}")
for _, r in sleepers.head(25).iterrows():
    ppr = f"{r['best_ppr']:.1f}" if pd.notna(r["best_ppr"]) else "n/a"
    print(f"  {r['player_name']:<25} {r['position']:>3} {int(r['draft_year']):>5} "
          f"{int(r['pick']):>5} {r['slap']:>6.1f} {r['dc']:>5.1f} {ppr:>8}")

# --- Players where SLAP disagrees most with DC (positive delta = model likes more) ---
backtest_with_outcome = backtest.copy()
backtest_with_outcome["delta"] = backtest_with_outcome["slap"] - backtest_with_outcome["dc"]

# Positive delta busts: model boosted them but they busted
pos_delta_busts = (backtest_with_outcome[
    (backtest_with_outcome["delta"] > 5) & (backtest_with_outcome["hit24"] == 0)]
    .sort_values("delta", ascending=False))

print(f"\n  POSITIVE-DELTA BUSTS  (SLAP boosted > +5 over DC, but busted)  "
      f"—  {len(pos_delta_busts)} players")
print(f"  {'Player':<25} {'Pos':>3} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>5} "
      f"{'Delta':>6} {'BestPPR':>8}")
print(f"  {'─'*72}")
for _, r in pos_delta_busts.head(15).iterrows():
    ppr = f"{r['best_ppr']:.1f}" if pd.notna(r["best_ppr"]) else "n/a"
    print(f"  {r['player_name']:<25} {r['position']:>3} {int(r['draft_year']):>5} "
          f"{int(r['pick']):>5} {r['slap']:>6.1f} {r['dc']:>5.1f} "
          f"{r['delta']:>+6.1f} {ppr:>8}")

# Negative delta hits: model dinged them but they hit anyway
neg_delta_hits = (backtest_with_outcome[
    (backtest_with_outcome["delta"] < -5) & (backtest_with_outcome["hit24"] == 1)]
    .sort_values("delta", ascending=True))

print(f"\n  NEGATIVE-DELTA HITS  (SLAP dinged < -5 vs DC, but they hit)  "
      f"—  {len(neg_delta_hits)} players")
print(f"  {'Player':<25} {'Pos':>3} {'Year':>5} {'Pick':>5} {'SLAP':>6} {'DC':>5} "
      f"{'Delta':>6} {'BestPPR':>8}")
print(f"  {'─'*72}")
for _, r in neg_delta_hits.head(15).iterrows():
    ppr = f"{r['best_ppr']:.1f}" if pd.notna(r["best_ppr"]) else "n/a"
    print(f"  {r['player_name']:<25} {r['position']:>3} {int(r['draft_year']):>5} "
          f"{int(r['pick']):>5} {r['slap']:>6.1f} {r['dc']:>5.1f} "
          f"{r['delta']:>+6.1f} {ppr:>8}")

# ---------------------------------------------------------------------------
# 4.  EARLY RETURNS — 2025 DRAFT CLASS
# ---------------------------------------------------------------------------

print("\n" + "=" * 90)
print("SECTION 4 — 2025 DRAFT CLASS: EARLY RETURNS  (one NFL season)")
print("=" * 90)
print("\n  ⚠  These players have only played one NFL season.")
print("     Hit24 status may change as careers develop.\n")

if len(early_returns) > 0:
    er = early_returns.sort_values("slap", ascending=False).reset_index(drop=True)
    print(f"  {'Rank':>4} {'Player':<25} {'Pos':>3} {'Pick':>5} {'SLAP':>6} {'DC':>5} "
          f"{'Hit24':>6} {'BestPPR':>8}")
    print(f"  {'─'*70}")
    for i, r in er.iterrows():
        ppr = f"{r['best_ppr']:.1f}" if pd.notna(r["best_ppr"]) else "n/a"
        hit = "YES" if r["hit24"] == 1 else "no"
        print(f"  {i+1:>4} {r['player_name']:<25} {r['position']:>3} "
              f"{int(r['pick']):>5} {r['slap']:>6.1f} {r['dc']:>5.1f} "
              f"{hit:>6} {ppr:>8}")

    hits_2025  = int(er["hit24"].sum())
    total_2025 = len(er)
    print(f"\n  2025 summary: {hits_2025}/{total_2025} hit24 after year 1 "
          f"({hits_2025/total_2025*100:.0f}%)")
else:
    print("  No 2025 data found.")

# ---------------------------------------------------------------------------
# 5.  YEAR-BY-YEAR BREAKDOWN
# ---------------------------------------------------------------------------

print("\n" + "=" * 90)
print("SECTION 5 — YEAR-BY-YEAR CORRELATION  (SLAP vs DC-Only → Best PPR)")
print("=" * 90)

print(f"\n  {'Year':>6} {'N':>5} {'Pos':>5} {'r(SLAP)':>10} {'r(DC)':>10} {'Winner':>10}")
print(f"  {'─'*50}")

for yr in sorted(backtest["draft_year"].unique()):
    ydf = backtest[backtest["draft_year"] == yr].dropna(subset=["best_ppr"])
    if len(ydf) < 5:
        continue
    rs, _ = pearsonr(ydf["slap"], ydf["best_ppr"])
    rd, _ = pearsonr(ydf["dc_only_score"], ydf["best_ppr"])
    pos_mix = "/".join(
        f"{v}{k}" for k, v in ydf["position"].value_counts().items()
    )
    winner = "SLAP" if rs > rd + 0.01 else ("DC" if rd > rs + 0.01 else "TIE")
    print(f"  {int(yr):>6} {len(ydf):>5} {pos_mix:>5} {rs:>10.3f} {rd:>10.3f} "
          f"{winner:>10}")

# Totals
valid_all = backtest.dropna(subset=["best_ppr"])
rs_all, _ = pearsonr(valid_all["slap"], valid_all["best_ppr"])
rd_all, _ = pearsonr(valid_all["dc_only_score"], valid_all["best_ppr"])
winner_all = "SLAP" if rs_all > rd_all else "DC-Only"
print(f"  {'─'*50}")
print(f"  {'ALL':>6} {len(valid_all):>5} {'':>5} {rs_all:>10.3f} {rd_all:>10.3f} "
      f"{winner_all:>10}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("BACKTEST COMPLETE")
print("=" * 90)
