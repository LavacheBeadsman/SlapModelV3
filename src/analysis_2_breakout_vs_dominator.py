"""
Analysis Test 2 of 5: Breakout Age vs Dominator Rating for WR Production Metric
================================================================================
Compares breakout age (current approach) vs dominator rating as the WR
production component in the SLAP model. Tests 5 approaches against 4 outcome
metrics to determine which production metric adds the most value.

Approaches tested:
  1. Breakout Age approach (current) - wr_breakout_score() at 65/20/15
  2. Dominator-only approach - peak_dominator normalized 0-100 at 65/20/15
  3. DC-only approach - no production, redistribute 20% proportionally to DC/RAS
  4. Breakout age alone - raw correlation, no DC involved
  5. Dominator alone - raw correlation, no DC involved

Outcome metrics: hit24, hit12, first_3yr_ppg, career_ppg
"""

import pandas as pd
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# FORMULAS
# ---------------------------------------------------------------------------

def normalize_draft_capital(pick):
    """Gentler DC curve: 100 - 2.40 * (pick^0.62 - 1)"""
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))


def wr_breakout_score(breakout_age, dominator_pct):
    """Current breakout-age-based production score for WRs."""
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


def dominator_score_normalized(dominator_pct):
    """Normalize peak dominator to 0-100 scale (50% dominator = 100)."""
    if dominator_pct is None or pd.isna(dominator_pct):
        return 25  # fallback for missing
    return min(99.9, dominator_pct * (100 / 50))


def athletic_score_wr(ras):
    """RAS * 10 for WRs. Missing = 68.9 average."""
    if ras is None or pd.isna(ras):
        return 68.9
    return ras * 10


# ---------------------------------------------------------------------------
# LOAD AND MERGE DATA
# ---------------------------------------------------------------------------

outcomes = pd.read_csv("/home/user/SlapModelV3/data/backtest_outcomes_complete.csv")
wr_data = pd.read_csv("/home/user/SlapModelV3/data/wr_backtest_expanded_final.csv")

# Filter to WRs only in outcomes
outcomes_wr = outcomes[outcomes["position"] == "WR"].copy()

# Merge on player_name and draft_year for safety
merged = outcomes_wr.merge(
    wr_data[["player_name", "draft_year", "pick", "RAS", "breakout_age", "peak_dominator"]],
    on=["player_name", "draft_year"],
    how="inner",
    suffixes=("", "_wr")
)

# Use outcomes pick as primary, fall back to wr_data pick
merged["pick_final"] = merged["pick"].fillna(merged["pick_wr"])

print("=" * 80)
print("ANALYSIS TEST 2: BREAKOUT AGE vs DOMINATOR RATING (WR Production Metric)")
print("=" * 80)
print(f"\nOutcomes WRs: {len(outcomes_wr)}")
print(f"WR backtest data: {len(wr_data)}")
print(f"Merged (inner join): {len(merged)}")

# Check for any outcomes WRs that didn't match
unmatched = outcomes_wr[~outcomes_wr["player_name"].isin(merged["player_name"])]
if len(unmatched) > 0:
    print(f"WARNING: {len(unmatched)} WRs in outcomes did not match WR backtest data:")
    for _, row in unmatched.iterrows():
        print(f"  - {row['player_name']} ({row['draft_year']})")

# Filter to 2015-2025
merged = merged[(merged["draft_year"] >= 2015) & (merged["draft_year"] <= 2025)].copy()
print(f"After filtering 2015-2025: {len(merged)} WRs")

# ---------------------------------------------------------------------------
# COMPUTE COMPONENT SCORES
# ---------------------------------------------------------------------------

merged["dc_score"] = merged["pick_final"].apply(normalize_draft_capital)

merged["breakout_prod_score"] = merged.apply(
    lambda r: wr_breakout_score(r["breakout_age"], r["peak_dominator"]), axis=1
)

merged["dominator_prod_score"] = merged["peak_dominator"].apply(dominator_score_normalized)

merged["ras_score"] = merged["RAS"].apply(athletic_score_wr)

# ---------------------------------------------------------------------------
# COMPUTE 3 SLAP VERSIONS
# ---------------------------------------------------------------------------

# 1. Breakout Age approach (current): 65/20/15
merged["slap_breakout"] = (
    0.65 * merged["dc_score"]
    + 0.20 * merged["breakout_prod_score"]
    + 0.15 * merged["ras_score"]
)

# 2. Dominator-only approach: 65/20/15
merged["slap_dominator"] = (
    0.65 * merged["dc_score"]
    + 0.20 * merged["dominator_prod_score"]
    + 0.15 * merged["ras_score"]
)

# 3. DC-only approach: redistribute 20% proportionally to DC (65%) and RAS (15%)
#    Instructions say: SLAP = 0.765 * DC + 0.0 * production + 0.235 * RAS
merged["slap_dc_only"] = (
    0.765 * merged["dc_score"]
    + 0.235 * merged["ras_score"]
)

# ---------------------------------------------------------------------------
# OUTCOME COLUMNS
# ---------------------------------------------------------------------------

outcome_cols = ["hit24", "hit12", "first_3yr_ppg", "career_ppg"]
outcome_labels = {
    "hit24": "Hit24 (Top-24 PPR)",
    "hit12": "Hit12 (Top-12 PPR)",
    "first_3yr_ppg": "First 3yr PPG",
    "career_ppg": "Career PPG"
}

# ---------------------------------------------------------------------------
# CORRELATION FUNCTION
# ---------------------------------------------------------------------------

def compute_correlations(df, score_col, outcome_cols):
    """Compute Pearson r and p-value for a score column vs each outcome."""
    results = {}
    for oc in outcome_cols:
        valid = df[[score_col, oc]].dropna()
        if len(valid) < 3:
            results[oc] = (np.nan, np.nan, 0)
            continue
        r, p = stats.pearsonr(valid[score_col], valid[oc])
        results[oc] = (r, p, len(valid))
    return results


# ---------------------------------------------------------------------------
# TEST 1-5: ALL APPROACHES
# ---------------------------------------------------------------------------

approaches = {
    "1. Breakout Age (current, 65/20/15)": "slap_breakout",
    "2. Dominator-only (65/20/15)":         "slap_dominator",
    "3. DC-only (76.5/0/23.5)":             "slap_dc_only",
    "4. Breakout score alone (no DC)":       "breakout_prod_score",
    "5. Dominator score alone (no DC)":      "dominator_prod_score",
}

print("\n" + "=" * 80)
print("CORRELATION RESULTS: ALL 5 APPROACHES vs 4 OUTCOMES")
print("=" * 80)

# Build a results table
results_rows = []
for approach_name, col in approaches.items():
    corrs = compute_correlations(merged, col, outcome_cols)
    row = {"Approach": approach_name}
    for oc in outcome_cols:
        r, p, n = corrs[oc]
        row[f"{oc}_r"] = r
        row[f"{oc}_p"] = p
        row[f"{oc}_n"] = n
    results_rows.append(row)

results_df = pd.DataFrame(results_rows)

# Print formatted table
for _, row in results_df.iterrows():
    print(f"\n  {row['Approach']}")
    print(f"  {'Outcome':<25} {'r':>8} {'p-value':>12} {'N':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*6}")
    for oc in outcome_cols:
        r = row[f"{oc}_r"]
        p = row[f"{oc}_p"]
        n = row[f"{oc}_n"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {outcome_labels[oc]:<25} {r:>8.4f} {p:>12.6f} {int(n):>6} {sig}")

# ---------------------------------------------------------------------------
# SUMMARY COMPARISON TABLE
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY COMPARISON TABLE (Pearson r values)")
print("=" * 80)

header = f"  {'Approach':<42}"
for oc in outcome_cols:
    header += f" {oc:>12}"
print(header)
print(f"  {'-'*42}" + f" {'-'*12}" * 4)

for _, row in results_df.iterrows():
    line = f"  {row['Approach']:<42}"
    for oc in outcome_cols:
        r = row[f"{oc}_r"]
        line += f" {r:>12.4f}"
    print(line)

# Highlight winner for each outcome
print(f"\n  {'BEST FULL SLAP APPROACH PER OUTCOME:':<42}")
for oc in outcome_cols:
    # Only compare the 3 full SLAP approaches (indices 0-2)
    best_slap_idx = results_df.iloc[:3][f"{oc}_r"].idxmax()
    best_slap = results_df.iloc[best_slap_idx]
    print(f"    {outcome_labels[oc]:<25}: {best_slap['Approach']} (r={best_slap[f'{oc}_r']:.4f})")

# ---------------------------------------------------------------------------
# BREAKOUT vs DC-ONLY: DOES PRODUCTION ADD VALUE?
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VALUE-ADD ANALYSIS: Does adding production improve on DC-only?")
print("=" * 80)

for oc in outcome_cols:
    r_breakout = results_df.iloc[0][f"{oc}_r"]
    r_dominator = results_df.iloc[1][f"{oc}_r"]
    r_dc_only = results_df.iloc[2][f"{oc}_r"]
    
    delta_breakout = r_breakout - r_dc_only
    delta_dominator = r_dominator - r_dc_only
    
    print(f"\n  {outcome_labels[oc]}:")
    print(f"    DC-only r:           {r_dc_only:.4f}")
    print(f"    + Breakout r:        {r_breakout:.4f}  (delta: {delta_breakout:+.4f})")
    print(f"    + Dominator r:       {r_dominator:.4f}  (delta: {delta_dominator:+.4f})")
    
    if delta_breakout > 0 and delta_breakout > delta_dominator:
        print(f"    --> Breakout WINS (improves DC-only by {delta_breakout:+.4f})")
    elif delta_dominator > 0 and delta_dominator > delta_breakout:
        print(f"    --> Dominator WINS (improves DC-only by {delta_dominator:+.4f})")
    elif delta_breakout <= 0 and delta_dominator <= 0:
        print(f"    --> Neither adds value over DC-only")
    else:
        print(f"    --> Tie or marginal difference")

# ---------------------------------------------------------------------------
# TIER BREAKDOWN: BREAKOUT AGE
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("TIER BREAKDOWN: BREAKOUT AGE")
print("=" * 80)

def breakout_tier(age):
    if pd.isna(age):
        return "Never (no 20%+ dominator)"
    return f"Age {int(age)}"

merged["breakout_tier"] = merged["breakout_age"].apply(breakout_tier)

# Define tier ordering
tier_order = ["Age 18", "Age 19", "Age 20", "Age 21", "Age 22", "Age 23", "Age 24", "Never (no 20%+ dominator)"]
# Only include tiers that exist
existing_tiers = [t for t in tier_order if t in merged["breakout_tier"].values]

print(f"\n  {'Tier':<30} {'N':>5} {'Hit24%':>8} {'Hit12%':>8} {'3yr PPG':>9} {'Car PPG':>9} {'Seasons':>12}")
print(f"  {'-'*30} {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*12}")

for tier in existing_tiers:
    group = merged[merged["breakout_tier"] == tier]
    n = len(group)
    hit24_rate = group["hit24"].mean() * 100 if n > 0 else 0
    hit12_rate = group["hit12"].mean() * 100 if n > 0 else 0
    mean_3yr = group["first_3yr_ppg"].mean() if n > 0 else 0
    mean_career = group["career_ppg"].mean() if n > 0 else 0
    min_seasons = group["seasons_played"].min() if n > 0 else 0
    max_seasons = group["seasons_played"].max() if n > 0 else 0
    seasons_str = f"{min_seasons}-{max_seasons}"
    
    print(f"  {tier:<30} {n:>5} {hit24_rate:>7.1f}% {hit12_rate:>7.1f}% {mean_3yr:>9.2f} {mean_career:>9.2f} {seasons_str:>12}")

# Total row
total_n = len(merged)
print(f"  {'-'*30} {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*12}")
print(f"  {'TOTAL':<30} {total_n:>5} {merged['hit24'].mean()*100:>7.1f}% {merged['hit12'].mean()*100:>7.1f}% {merged['first_3yr_ppg'].mean():>9.2f} {merged['career_ppg'].mean():>9.2f} {merged['seasons_played'].min()}-{merged['seasons_played'].max():>10}")

# Detail table: show every player in each tier
print("\n  DETAIL BY TIER:")
for tier in existing_tiers:
    group = merged[merged["breakout_tier"] == tier].sort_values("pick_final")
    print(f"\n  --- {tier} ({len(group)} players) ---")
    print(f"  {'Player':<25} {'Pick':>5} {'Year':>5} {'Dominator':>10} {'Hit24':>6} {'Hit12':>6} {'3yr PPG':>8} {'Car PPG':>8} {'Seasons':>8}")
    for _, r in group.iterrows():
        dom_str = f"{r['peak_dominator']:.1f}%" if pd.notna(r['peak_dominator']) else "N/A"
        ppg3 = f"{r['first_3yr_ppg']:.1f}" if pd.notna(r['first_3yr_ppg']) else "N/A"
        ppgc = f"{r['career_ppg']:.1f}" if pd.notna(r['career_ppg']) else "N/A"
        print(f"  {r['player_name']:<25} {int(r['pick_final']):>5} {int(r['draft_year']):>5} {dom_str:>10} {int(r['hit24']):>6} {int(r['hit12']):>6} {ppg3:>8} {ppgc:>8} {int(r['seasons_played']):>8}")

# ---------------------------------------------------------------------------
# TIER BREAKDOWN: DOMINATOR QUINTILES
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("TIER BREAKDOWN: DOMINATOR RATING QUINTILES")
print("=" * 80)

def dominator_tier(dom):
    if pd.isna(dom):
        return "Missing"
    if dom < 10:
        return "0-10%"
    elif dom < 20:
        return "10-20%"
    elif dom < 30:
        return "20-30%"
    elif dom < 40:
        return "30-40%"
    else:
        return "40%+"

merged["dominator_tier"] = merged["peak_dominator"].apply(dominator_tier)

dom_tier_order = ["0-10%", "10-20%", "20-30%", "30-40%", "40%+", "Missing"]
existing_dom_tiers = [t for t in dom_tier_order if t in merged["dominator_tier"].values]

print(f"\n  {'Tier':<20} {'N':>5} {'Hit24%':>8} {'Hit12%':>8} {'3yr PPG':>9} {'Car PPG':>9} {'Seasons':>12}")
print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*12}")

for tier in existing_dom_tiers:
    group = merged[merged["dominator_tier"] == tier]
    n = len(group)
    hit24_rate = group["hit24"].mean() * 100 if n > 0 else 0
    hit12_rate = group["hit12"].mean() * 100 if n > 0 else 0
    mean_3yr = group["first_3yr_ppg"].mean() if n > 0 else 0
    mean_career = group["career_ppg"].mean() if n > 0 else 0
    min_seasons = group["seasons_played"].min() if n > 0 else 0
    max_seasons = group["seasons_played"].max() if n > 0 else 0
    seasons_str = f"{min_seasons}-{max_seasons}"
    
    print(f"  {tier:<20} {n:>5} {hit24_rate:>7.1f}% {hit12_rate:>7.1f}% {mean_3yr:>9.2f} {mean_career:>9.2f} {seasons_str:>12}")

# Total row
print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*12}")
print(f"  {'TOTAL':<20} {total_n:>5} {merged['hit24'].mean()*100:>7.1f}% {merged['hit12'].mean()*100:>7.1f}% {merged['first_3yr_ppg'].mean():>9.2f} {merged['career_ppg'].mean():>9.2f} {merged['seasons_played'].min()}-{merged['seasons_played'].max():>10}")

# Detail table: show every player in each dominator tier
print("\n  DETAIL BY DOMINATOR TIER:")
for tier in existing_dom_tiers:
    group = merged[merged["dominator_tier"] == tier].sort_values("pick_final")
    print(f"\n  --- {tier} ({len(group)} players) ---")
    print(f"  {'Player':<25} {'Pick':>5} {'Year':>5} {'Breakout':>10} {'Dominator':>10} {'Hit24':>6} {'Hit12':>6} {'3yr PPG':>8} {'Car PPG':>8} {'Seasons':>8}")
    for _, r in group.iterrows():
        bo_str = f"Age {int(r['breakout_age'])}" if pd.notna(r['breakout_age']) else "Never"
        dom_str = f"{r['peak_dominator']:.1f}%" if pd.notna(r['peak_dominator']) else "N/A"
        ppg3 = f"{r['first_3yr_ppg']:.1f}" if pd.notna(r['first_3yr_ppg']) else "N/A"
        ppgc = f"{r['career_ppg']:.1f}" if pd.notna(r['career_ppg']) else "N/A"
        print(f"  {r['player_name']:<25} {int(r['pick_final']):>5} {int(r['draft_year']):>5} {bo_str:>10} {dom_str:>10} {int(r['hit24']):>6} {int(r['hit12']):>6} {ppg3:>8} {ppgc:>8} {int(r['seasons_played']):>8}")

# ---------------------------------------------------------------------------
# FINAL VERDICT
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

# Count wins across 4 outcomes (among the 3 SLAP approaches)
wins = {"Breakout": 0, "Dominator": 0, "DC-only": 0}
for oc in outcome_cols:
    r_vals = [
        results_df.iloc[0][f"{oc}_r"],
        results_df.iloc[1][f"{oc}_r"],
        results_df.iloc[2][f"{oc}_r"],
    ]
    best_idx = int(np.argmax(r_vals))
    labels = ["Breakout", "Dominator", "DC-only"]
    wins[labels[best_idx]] += 1

print(f"\n  Wins across 4 outcome metrics (among 3 SLAP approaches):")
for label, count in wins.items():
    print(f"    {label}: {count}/4 outcomes")

# Raw component wins
raw_wins = {"Breakout score": 0, "Dominator score": 0}
for oc in outcome_cols:
    r_breakout_raw = results_df.iloc[3][f"{oc}_r"]
    r_dom_raw = results_df.iloc[4][f"{oc}_r"]
    if r_breakout_raw > r_dom_raw:
        raw_wins["Breakout score"] += 1
    else:
        raw_wins["Dominator score"] += 1

print(f"\n  Raw component correlations (no DC):")
for label, count in raw_wins.items():
    print(f"    {label}: {count}/4 outcomes")

# Check if breakout adds value over DC-only
adds_value_count = 0
for oc in outcome_cols:
    r_breakout = results_df.iloc[0][f"{oc}_r"]
    r_dc_only = results_df.iloc[2][f"{oc}_r"]
    if r_breakout > r_dc_only:
        adds_value_count += 1

print(f"\n  Breakout adds value over DC-only in {adds_value_count}/4 outcomes")

adds_value_dom = 0
for oc in outcome_cols:
    r_dom = results_df.iloc[1][f"{oc}_r"]
    r_dc_only = results_df.iloc[2][f"{oc}_r"]
    if r_dom > r_dc_only:
        adds_value_dom += 1

print(f"  Dominator adds value over DC-only in {adds_value_dom}/4 outcomes")

print("\n" + "=" * 80)
print("END OF ANALYSIS TEST 2")
print("=" * 80)
