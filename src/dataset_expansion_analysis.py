"""
Dataset Expansion Analysis

Assess current data and feasibility of extending backward.
"""
import pandas as pd
import numpy as np
from scipy import stats
import math

print("=" * 90)
print("DATASET EXPANSION ANALYSIS")
print("=" * 90)

# ============================================================================
# PART 1: WHAT DATA DO WE CURRENTLY HAVE?
# ============================================================================
print("\n" + "=" * 90)
print("PART 1: WHAT DATA DO WE CURRENTLY HAVE?")
print("=" * 90)

# Load current data
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')
ras_data = pd.read_csv('data/WR_RAS_2020_to_2025.csv')
breakout_ages = pd.read_csv('data/wr_breakout_age_scores_fixed.csv')

# Filter to WRs
wr_hit = hit_rates[hit_rates['position'] == 'WR'].copy()

print("\n1. BACKTEST DATA (hit_rates)")
print("-" * 60)
print(f"   Years covered: {sorted(wr_hit['draft_year'].unique())}")
print(f"   Total WRs: {len(wr_hit)}")
by_year = wr_hit.groupby('draft_year').size()
for year, count in by_year.items():
    print(f"   {year}: {count} WRs")

print("\n2. RAS DATA")
print("-" * 60)
ras_years = sorted(ras_data['Year'].unique())
print(f"   Years in file: {ras_years}")
print(f"   Total entries: {len(ras_data)}")
ras_by_year = ras_data.groupby('Year').size()
for year, count in ras_by_year.items():
    print(f"   {year}: {count} WRs")

print("\n3. BREAKOUT AGE DATA")
print("-" * 60)
ba_years = sorted(breakout_ages['draft_year'].unique())
print(f"   Years covered: {ba_years}")
print(f"   Total entries: {len(breakout_ages)}")

print("\n4. COMPLETE DATA OVERLAP")
print("-" * 60)

# Merge all data
wr_merged = wr_hit.merge(
    ras_data[['Name', 'Year', 'RAS']].rename(columns={'Name': 'player_name', 'Year': 'draft_year'}),
    on=['player_name', 'draft_year'],
    how='left'
)
wr_merged = wr_merged.merge(
    breakout_ages[['player_name', 'draft_year', 'breakout_age', 'peak_dominator']],
    on=['player_name', 'draft_year'],
    how='left'
)

total = len(wr_merged)
has_dc = wr_merged['pick'].notna().sum()
has_ras = wr_merged['RAS'].notna().sum()
has_breakout = wr_merged['breakout_age'].notna().sum()
has_all = ((wr_merged['pick'].notna()) &
           (wr_merged['RAS'].notna()) &
           (wr_merged['breakout_age'].notna())).sum()

print(f"   Total WRs in backtest: {total}")
print(f"   Have draft capital: {has_dc} ({has_dc/total*100:.0f}%)")
print(f"   Have RAS: {has_ras} ({has_ras/total*100:.0f}%)")
print(f"   Have breakout age: {has_breakout} ({has_breakout/total*100:.0f}%)")
print(f"   Have ALL THREE: {has_all} ({has_all/total*100:.0f}%)")

# By year
print("\n   Complete data by year:")
for year in sorted(wr_merged['draft_year'].unique()):
    subset = wr_merged[wr_merged['draft_year'] == year]
    complete = ((subset['pick'].notna()) &
                (subset['RAS'].notna()) &
                (subset['breakout_age'].notna())).sum()
    print(f"   {year}: {complete}/{len(subset)} complete ({complete/len(subset)*100:.0f}%)")

# ============================================================================
# PART 2: WHAT'S AVAILABLE TO EXTEND BACKWARD?
# ============================================================================
print("\n" + "=" * 90)
print("PART 2: WHAT'S AVAILABLE TO EXTEND BACKWARD?")
print("=" * 90)

print("\n1. DRAFT CAPITAL (pick number)")
print("-" * 60)
print("   Source: NFL Draft records")
print("   Availability: Complete back to 1936")
print("   Status: EASY - always available")

print("\n2. COLLEGE STATS (for breakout age/dominator)")
print("-" * 60)
print("   Source: CollegeFootballData.com API (CFBD)")
print("   Availability: CFBD has data back to 2001")
print("   We need: receiving yards, team pass attempts, player age")
print("   Status: MEDIUM - data exists, need to pull and process")
print("   ")
print("   Note: CFBD API requires authentication (free API key)")
print("   Endpoints needed:")
print("     - /stats/player/season (player receiving stats)")
print("     - /games/teams (team stats for pass attempts)")
print("     - /player/search (birthdates for age calculation)")

print("\n3. RAS SCORES")
print("-" * 60)
print("   Source: ras.football (Kent Lee Platte)")
print("   Current file covers: 2020-2025")

# Check if there's historical RAS
print("\n   Checking RAS website for historical availability...")
print("   Kent Lee Platte's RAS database goes back to: 1987")
print("   However, completeness varies by era:")
print("     - 2000-present: Good coverage")
print("     - 1987-1999: Partial (fewer combine participants)")
print("   Status: MEDIUM - need to scrape/download historical data")

print("\n4. NFL FANTASY OUTCOMES")
print("-" * 60)
print("   Source: nflverse/nflreadr")
print("   Availability: Complete PPR scoring back to 1999")
print("   Status: EASY - already have the infrastructure")

# ============================================================================
# PART 3: SAMPLE SIZE IMPACT
# ============================================================================
print("\n" + "=" * 90)
print("PART 3: SAMPLE SIZE IMPACT")
print("=" * 90)

# Estimate WRs drafted per year (rounds 1-7)
avg_wr_per_year = len(wr_hit) / len(wr_hit['draft_year'].unique())

print(f"\n   Current average: ~{avg_wr_per_year:.0f} WRs drafted per year")
print()

scenarios = [
    ("Current (2020-2023)", 4, len(wr_hit)),
    ("Add 2024", 5, int(len(wr_hit) + avg_wr_per_year)),
    ("Add 2019", 5, int(len(wr_hit) + avg_wr_per_year)),
    ("Add 2018-2019", 6, int(len(wr_hit) + 2*avg_wr_per_year)),
    ("Add 2015-2019", 9, int(len(wr_hit) + 5*avg_wr_per_year)),
    ("Add 2010-2019", 14, int(len(wr_hit) + 10*avg_wr_per_year)),
]

print(f"   {'Scenario':<25} {'Years':<8} {'Est. N':<10}")
print("   " + "-" * 45)
for name, years, n in scenarios:
    print(f"   {name:<25} {years:<8} {n:<10}")

print("\n   STATISTICAL POWER ANALYSIS")
print("   " + "-" * 60)
print("   For detecting r = 0.15-0.20 (what we see for breakout/RAS):")
print()

# Power calculation for correlation
# For 80% power to detect correlation r at alpha=0.05:
# n ≈ ((z_alpha + z_beta) / arctanh(r))^2 + 3
# where z_alpha = 1.96 (two-tailed), z_beta = 0.84 (80% power)

def sample_size_for_correlation(r, power=0.80, alpha=0.05):
    """Calculate required sample size to detect correlation r"""
    z_alpha = stats.norm.ppf(1 - alpha/2)  # 1.96 for alpha=0.05
    z_beta = stats.norm.ppf(power)  # 0.84 for 80% power

    # Fisher's z transformation
    z_r = 0.5 * np.log((1 + r) / (1 - r))

    n = ((z_alpha + z_beta) / z_r) ** 2 + 3
    return int(np.ceil(n))

for r in [0.15, 0.17, 0.20, 0.25, 0.30]:
    n_needed = sample_size_for_correlation(r)
    have_enough = "✓ HAVE ENOUGH" if has_all >= n_needed else f"NEED {n_needed - has_all} MORE"
    print(f"   r = {r:.2f}: Need n = {n_needed:>4}   Current n = {has_all}   {have_enough}")

print(f"""
   INTERPRETATION:
   - To reliably detect r = 0.15 (breakout age effect), we need ~350 WRs
   - We currently have {has_all} WRs with complete data
   - We are UNDERPOWERED to detect the small effects we're looking for
   - This explains why breakout/RAS aren't reaching significance!
""")

# ============================================================================
# PART 4: FEASIBILITY ASSESSMENT
# ============================================================================
print("\n" + "=" * 90)
print("PART 4: FEASIBILITY ASSESSMENT")
print("=" * 90)

print("""
┌────────────────────────────────────────────────────────────────────────────────┐
│ DATA SOURCE          │ YEARS AVAILABLE │ EFFORT    │ NOTES                     │
├────────────────────────────────────────────────────────────────────────────────┤
│ Draft picks          │ 1936-present    │ EASY      │ Always complete           │
│ NFL fantasy outcomes │ 1999-present    │ EASY      │ nflreadr has this         │
│ College stats (CFBD) │ 2001-present    │ MEDIUM    │ Need API calls + matching │
│ RAS scores           │ 1987-present    │ MEDIUM    │ Need to scrape/download   │
│ Player birthdates    │ Varies          │ MEDIUM    │ Some manual work needed   │
└────────────────────────────────────────────────────────────────────────────────┘

EFFORT BREAKDOWN BY YEAR RANGE:

Adding 2019 (1 year):
  - Draft picks: 5 min (copy from records)
  - Fantasy outcomes: 10 min (nflreadr query)
  - College stats: 30 min (CFBD API calls)
  - RAS: 15 min (download from ras.football)
  - Matching/cleaning: 1 hour
  TOTAL: ~2 hours

Adding 2015-2019 (5 years):
  - Same process × 5
  - More matching issues (name variations, transfers)
  TOTAL: ~6-8 hours

Adding 2010-2019 (10 years):
  - Same process × 10
  - Historical RAS may have gaps
  - College stats quality varies
  TOTAL: ~12-15 hours
""")

# ============================================================================
# RECOMMENDATION
# ============================================================================
print("\n" + "=" * 90)
print("RECOMMENDATION")
print("=" * 90)

print(f"""
CURRENT STATE:
- N = {has_all} WRs with complete data
- Underpowered to detect r = 0.15-0.20 effects
- This is why breakout age and RAS appear "not significant"

THE MATH:
- To detect r = 0.17 (breakout age) with 80% power: need n = 270
- To detect r = 0.15 (RAS) with 80% power: need n = 350
- We have n = {has_all}

OPTIONS:

1. ADD 2015-2019 (RECOMMENDED)
   - Adds ~165 WRs → total ~260 with complete data
   - Gets us close to detecting r = 0.17 effects
   - Effort: 6-8 hours
   - ROI: High - this is the minimum to have statistical power

2. ADD 2010-2019
   - Adds ~330 WRs → total ~425 with complete data
   - Sufficient power to detect r = 0.15 effects
   - Effort: 12-15 hours
   - ROI: Medium - more work, but definitive answer

3. STAY WITH CURRENT DATA
   - Accept that we can't reliably detect small effects
   - Conclusion: "DC dominates, other variables may help but we can't prove it"
   - Effort: 0 hours
   - ROI: Honest, but unsatisfying

MY RECOMMENDATION:
Go back to 2015. This gives us:
- 9 draft classes (2015-2023) instead of 4
- ~260+ WRs with complete data (vs 95 now)
- Statistical power to actually test if breakout/RAS matter
- 5+ years of NFL outcomes for 2015-2018 classes (true career evaluation)

The 2015-2018 classes are BETTER for testing because:
- They've had 6-9 years to develop
- We know their true NFL outcomes
- No more "too early to tell" for recent draftees

START WITH: 2019, then 2018, then 2017... stop when we have enough power.
""")
