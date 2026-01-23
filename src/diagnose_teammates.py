"""
Diagnose the teammate problem for WRs like Waddle and Addison
"""
import pandas as pd
import numpy as np

# Load data
all_seasons = pd.read_csv('data/wr_all_seasons.csv')
breakout = pd.read_csv('data/wr_breakout_age_scores.csv')
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')

# Get drafted WRs
drafted_wrs = hit_rates[hit_rates['position'] == 'WR'][['player_name', 'draft_year', 'pick']].copy()

print("="*80)
print("DIAGNOSTIC: The Teammate Problem")
print("="*80)

# ============================================================================
# CASE STUDY 1: Jaylen Waddle at Alabama
# ============================================================================
print("\n" + "="*80)
print("CASE STUDY 1: JAYLEN WADDLE (Alabama)")
print("="*80)

# Show Waddle's seasons
waddle = all_seasons[all_seasons['player_name'] == 'Jaylen Waddle']
print("\nWaddle's College Seasons:")
print("-"*70)
for _, row in waddle.iterrows():
    print(f"  {row['season']}: {row['player_rec_yards']:.0f} yards, "
          f"{row['dominator_rating']:.1f}% dominator")

# Show Alabama teammates in same years
print("\nAlabama WRs by Season (showing drafted WRs):")
print("-"*70)

# Find all Alabama seasons
alabama = all_seasons[all_seasons['college'] == 'Alabama']

for year in [2018, 2019, 2020]:
    print(f"\n{year} Alabama Receiving:")
    year_data = alabama[alabama['season'] == year].sort_values('dominator_rating', ascending=False)

    for _, row in year_data.iterrows():
        # Check if this player was drafted
        draft_info = drafted_wrs[drafted_wrs['player_name'] == row['player_name']]
        if len(draft_info) > 0:
            pick = draft_info.iloc[0]['pick']
            dy = draft_info.iloc[0]['draft_year']
            draft_str = f"(Pick {pick}, {dy})"
        else:
            draft_str = ""

        print(f"  {row['player_name']:<20} {row['player_rec_yards']:>5.0f} yds  "
              f"{row['dominator_rating']:>5.1f}%  {draft_str}")

# Calculate what Waddle WOULD have been without elite teammates
print("\n⚠️  WADDLE'S SITUATION:")
print("  Peak dominator: 17.5% (2018)")
print("  BUT in 2018, he was competing with:")
print("    - Jerry Jeudy (27.1%, Pick 15)")
print("    - Henry Ruggs III (15.3%, Pick 12)")
print("    - DeVonta Smith (14.3%, Pick 10)")
print("  That's THREE future 1st-round WRs on the same team!")

# ============================================================================
# CASE STUDY 2: Jordan Addison at Pitt/USC
# ============================================================================
print("\n" + "="*80)
print("CASE STUDY 2: JORDAN ADDISON (Pitt → USC)")
print("="*80)

addison = all_seasons[all_seasons['player_name'] == 'Jordan Addison']
print("\nAddison's College Seasons IN OUR DATA:")
print("-"*70)
for _, row in addison.iterrows():
    print(f"  {row['season']} ({row['college']}): {row['player_rec_yards']:.0f} yards, "
          f"{row['dominator_rating']:.1f}% dominator")

print("\n⚠️  ADDISON'S SITUATION:")
print("  Peak dominator: 18.6% (2022 at USC)")
print("  BUT at USC in 2022, he was competing with:")

# Show USC 2022
usc_2022 = all_seasons[(all_seasons['college'] == 'USC') & (all_seasons['season'] == 2022)]
print("\n  2022 USC Receiving:")
for _, row in usc_2022.sort_values('dominator_rating', ascending=False).iterrows():
    draft_info = drafted_wrs[drafted_wrs['player_name'] == row['player_name']]
    if len(draft_info) > 0:
        pick = draft_info.iloc[0]['pick']
        draft_str = f"(Pick {pick})"
    else:
        draft_str = ""
    print(f"    {row['player_name']:<20} {row['player_rec_yards']:>5.0f} yds  "
          f"{row['dominator_rating']:>5.1f}%  {draft_str}")

print("\n  CRITICAL: At Pitt in 2021, Addison won Biletnikoff Award")
print("  (1,593 yards, 17 TDs) but we DON'T HAVE THIS DATA!")
print("  He transferred to USC for 2022.")

# ============================================================================
# CASE STUDY 3: Other WRs with NFL-caliber teammates
# ============================================================================
print("\n" + "="*80)
print("OTHER WRs WITH NFL-CALIBER TEAMMATES")
print("="*80)

# Find schools with multiple drafted WRs in same year
teammate_cases = []

schools_with_multiple = all_seasons.groupby(['college', 'season']).apply(
    lambda x: list(x['player_name'].unique())
).reset_index(name='players')

for _, row in schools_with_multiple.iterrows():
    drafted_in_group = []
    for player in row['players']:
        draft_info = drafted_wrs[drafted_wrs['player_name'] == player]
        if len(draft_info) > 0:
            drafted_in_group.append({
                'name': player,
                'pick': draft_info.iloc[0]['pick'],
                'draft_year': draft_info.iloc[0]['draft_year']
            })

    if len(drafted_in_group) >= 2:
        teammate_cases.append({
            'school': row['college'],
            'season': row['season'],
            'drafted_wrs': drafted_in_group
        })

# Show cases with day 1-2 picks
print("\nSchools with multiple drafted WRs in same season:")
print("-"*80)

shown = 0
for case in teammate_cases:
    # Only show if at least one was a Day 1-2 pick
    day12_picks = [d for d in case['drafted_wrs'] if d['pick'] <= 100]
    if len(day12_picks) >= 1 and len(case['drafted_wrs']) >= 2:
        print(f"\n{case['school']} {case['season']}:")
        for wr in sorted(case['drafted_wrs'], key=lambda x: x['pick']):
            # Get their dominator for that season
            dom = all_seasons[(all_seasons['player_name'] == wr['name']) &
                             (all_seasons['season'] == case['season'])]['dominator_rating'].values
            dom_str = f"{dom[0]:.1f}%" if len(dom) > 0 else "N/A"
            print(f"  - {wr['name']}: Pick {wr['pick']} ({wr['draft_year']}), {dom_str} dominator")
        shown += 1
        if shown >= 10:
            break

# ============================================================================
# SOLUTION FEASIBILITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SOLUTION FEASIBILITY ANALYSIS")
print("="*80)

print("""
OPTION A: Teammate Draft Capital Adjustment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ FEASIBLE - We have draft pick data for all WRs
✓ Can identify WRs who shared a team with other drafted WRs
✓ Formula: adjusted_dominator = raw × (1 + 0.10 × num_teammates)

PROBLEM: Only catches teammates drafted in OUR dataset (2020-2024)
         Waddle's 2018 teammates Jeudy/Ruggs/Smith were drafted 2020-2021 ✓
         BUT older teammates or ones drafted after wouldn't be captured

OPTION B: Team-Adjusted Dominator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ NOT FEASIBLE - Would need expected production by draft slot
  This requires a separate model or lookup table we don't have

OPTION C: Breakout Age + Draft Capital Only (drop dominator)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ FEASIBLE - Already have this data
✓ Correlation analysis showed age works (r=-0.334)
✓ Draft capital already accounts for NFL evaluation
✓ SIMPLEST SOLUTION

OPTION D: Conference/School Tier Adjustment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  PARTIALLY FEASIBLE - We have school names
  Would need to manually code conference tiers
  Could use SEC/Big Ten vs Group of 5 distinction
""")

# Quick test of Option A
print("\n" + "="*80)
print("QUICK TEST: Option A (Teammate Adjustment)")
print("="*80)

# For each WR, count how many other drafted WRs were on their team in their peak season
adjustments = []

for _, wr in breakout.iterrows():
    player = wr['player_name']

    # Find their peak season
    player_seasons = all_seasons[all_seasons['player_name'] == player]
    if len(player_seasons) == 0:
        continue

    # Skip if all dominator ratings are NaN
    valid_seasons = player_seasons[player_seasons['dominator_rating'].notna()]
    if len(valid_seasons) == 0:
        continue

    peak_season = valid_seasons.loc[valid_seasons['dominator_rating'].idxmax()]

    # Find teammates in that season
    teammates = all_seasons[
        (all_seasons['college'] == peak_season['college']) &
        (all_seasons['season'] == peak_season['season']) &
        (all_seasons['player_name'] != player)
    ]

    # Count drafted teammates
    drafted_teammates = 0
    teammate_picks = []
    for _, tm in teammates.iterrows():
        draft_info = drafted_wrs[drafted_wrs['player_name'] == tm['player_name']]
        if len(draft_info) > 0:
            drafted_teammates += 1
            teammate_picks.append(draft_info.iloc[0]['pick'])

    # Calculate adjustment
    raw_dom = wr['peak_dominator'] if pd.notna(wr['peak_dominator']) else 0
    adjusted_dom = raw_dom * (1 + 0.10 * drafted_teammates)

    adjustments.append({
        'player_name': player,
        'draft_year': wr['draft_year'],
        'raw_dominator': raw_dom,
        'drafted_teammates': drafted_teammates,
        'adjusted_dominator': adjusted_dom,
        'boost': adjusted_dom - raw_dom
    })

adj_df = pd.DataFrame(adjustments)

# Show biggest adjustments
print("\nWRs who would get the BIGGEST boost from teammate adjustment:")
print("-"*80)
top_boost = adj_df.nlargest(15, 'boost')
for _, row in top_boost.iterrows():
    print(f"  {row['player_name']:<25} {row['raw_dominator']:>5.1f}% → {row['adjusted_dominator']:>5.1f}% "
          f"(+{row['boost']:.1f}%, {row['drafted_teammates']:.0f} teammates)")

# Check Waddle and Addison specifically
print("\n\nWaddle and Addison adjustments:")
for name in ['Jaylen Waddle', 'Jordan Addison']:
    row = adj_df[adj_df['player_name'] == name]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"  {r['player_name']}: {r['raw_dominator']:.1f}% → {r['adjusted_dominator']:.1f}% "
              f"(+{r['boost']:.1f}%, {r['drafted_teammates']:.0f} teammates)")
