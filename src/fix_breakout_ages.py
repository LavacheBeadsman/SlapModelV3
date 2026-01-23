"""
Fix Missing Breakout Ages

Problem: Elite players at loaded programs never hit 20% dominator
         because they shared targets with other future NFL stars.

Solution: Add alternative breakout criterion
- Primary: First season with 20%+ dominator rating
- Alternative: First season with 700+ receiving yards (early production signal)

Take the EARLIER of the two as breakout age.

Evidence:
- 700+ yard early producers have 29.2% hit rate vs 16.8% overall
- Captures Waddle, Addison, Jefferson, Lamb, Puka, ARSB
"""
import pandas as pd
import numpy as np

# Load data
seasons = pd.read_csv('data/wr_all_seasons.csv')
breakout_ages = pd.read_csv('data/wr_breakout_age_scores.csv')
hit_rates = pd.read_csv('data/backtest_hit_rates.csv')

print("="*80)
print("FIXING BREAKOUT AGES - ADDING ALTERNATIVE CRITERION")
print("="*80)

# Age mapping (season year to typical age for freshmen)
# Players are typically 18 in their first college season
def estimate_age_at_season(draft_year, season):
    """Estimate player age during a college season"""
    # A player drafted in 2021 who played in 2020 was typically:
    # - If senior: 22
    # - Years since season to draft = draft_year - season
    years_until_draft = draft_year - season
    # Most players are 21-22 when drafted, so work backwards
    return 22 - years_until_draft

# Find breakout age for each player using BOTH criteria
def find_breakout_age(player_name, draft_year, seasons_df):
    """
    Find breakout age using:
    1. First season with 20%+ dominator
    2. First season with 700+ receiving yards

    Return the EARLIER of the two.
    """
    player_seasons = seasons_df[
        (seasons_df['player_name'] == player_name) &
        (seasons_df['draft_year'] == draft_year)
    ].sort_values('season')

    if len(player_seasons) == 0:
        return None, None, None

    breakout_dom = None
    breakout_yards = None

    for _, row in player_seasons.iterrows():
        season = row['season']
        age = estimate_age_at_season(draft_year, season)

        # Check dominator criterion
        if breakout_dom is None and row['dominator_rating'] >= 20:
            breakout_dom = age

        # Check yards criterion (700+ yards)
        if breakout_yards is None and row['player_rec_yards'] >= 700:
            breakout_yards = age

    # Take earlier breakout
    if breakout_dom and breakout_yards:
        breakout_age = min(breakout_dom, breakout_yards)
        method = 'both' if breakout_dom == breakout_yards else ('dominator' if breakout_age == breakout_dom else 'yards')
    elif breakout_dom:
        breakout_age = breakout_dom
        method = 'dominator'
    elif breakout_yards:
        breakout_age = breakout_yards
        method = 'yards'
    else:
        breakout_age = None
        method = None

    return breakout_age, breakout_dom, breakout_yards

# Process all players
print("\nProcessing players...")

results = []
for _, row in breakout_ages.iterrows():
    player = row['player_name']
    year = row['draft_year']
    old_breakout = row['breakout_age']

    new_breakout, dom_age, yards_age = find_breakout_age(player, year, seasons)

    results.append({
        'player_name': player,
        'draft_year': year,
        'college': row['college'],
        'peak_dominator': row['peak_dominator'],
        'old_breakout_age': old_breakout,
        'new_breakout_age': new_breakout,
        'dom_breakout_age': dom_age,
        'yards_breakout_age': yards_age,
        'method': 'dominator' if pd.notna(old_breakout) else ('yards' if new_breakout else None)
    })

results_df = pd.DataFrame(results)

# Show players who got FIXED
print("\n" + "="*80)
print("PLAYERS FIXED (now have breakout age)")
print("="*80)

fixed = results_df[
    (results_df['old_breakout_age'].isna()) &
    (results_df['new_breakout_age'].notna())
]

# Merge with NFL outcomes
fixed = fixed.merge(
    hit_rates[hit_rates['position'] == 'WR'][['player_name', 'draft_year', 'hit24', 'best_ppr']],
    on=['player_name', 'draft_year'],
    how='left'
)
fixed['best_ppg'] = fixed['best_ppr'] / 17

print(f"\nPlayers now have breakout age who didn't before: {len(fixed)}")
print("\n" + "-"*100)
print(f"{'Player':<22} {'Year':>5} {'Old':>5} {'New':>5} {'Method':<10} {'PPG':>6} {'Hit24':>6}")
print("-"*100)

for _, row in fixed.sort_values('draft_year').iterrows():
    old_str = f"{row['old_breakout_age']:.0f}" if pd.notna(row['old_breakout_age']) else "N/A"
    new_str = f"{row['new_breakout_age']:.0f}" if pd.notna(row['new_breakout_age']) else "N/A"
    hit_str = "HIT" if row.get('hit24') == 1 else ""
    ppg = row.get('best_ppg', 0)
    ppg_str = f"{ppg:.1f}" if pd.notna(ppg) else "N/A"

    print(f"{row['player_name']:<22} {row['draft_year']:>5} {old_str:>5} {new_str:>5} "
          f"{row.get('method', 'yards'):<10} {ppg_str:>6} {hit_str:>6}")

# Specifically highlight key players
print("\n" + "="*80)
print("KEY PLAYERS FIXED")
print("="*80)

key_players = ['Jaylen Waddle', 'Jordan Addison', 'Henry Ruggs III', 'Keon Coleman', 'Ladd McConkey']

for player in key_players:
    player_row = results_df[results_df['player_name'] == player]
    if len(player_row) > 0:
        row = player_row.iloc[0]
        old_str = f"{row['old_breakout_age']:.0f}" if pd.notna(row['old_breakout_age']) else "MISSING"
        new_str = f"{row['new_breakout_age']:.0f}" if pd.notna(row['new_breakout_age']) else "STILL MISSING"

        # Get season details
        player_seasons = seasons[
            (seasons['player_name'] == player) &
            (seasons['draft_year'] == row['draft_year'])
        ].sort_values('season')

        print(f"\n{player}:")
        print(f"  Old breakout age: {old_str}")
        print(f"  New breakout age: {new_str}")
        print(f"  Seasons:")
        for _, s in player_seasons.iterrows():
            age = estimate_age_at_season(row['draft_year'], s['season'])
            dom_flag = "✓20%+" if s['dominator_rating'] >= 20 else ""
            yds_flag = "✓700+" if s['player_rec_yards'] >= 700 else ""
            print(f"    {s['season']}: {s['player_rec_yards']:.0f} yds, {s['dominator_rating']:.1f}% dom "
                  f"(age ~{age}) {dom_flag} {yds_flag}")

# Create updated breakout ages file
print("\n" + "="*80)
print("SAVING UPDATED BREAKOUT AGES")
print("="*80)

# Update the breakout ages
updated_breakout = breakout_ages.copy()

for _, row in results_df.iterrows():
    if pd.notna(row['new_breakout_age']):
        mask = (updated_breakout['player_name'] == row['player_name']) & \
               (updated_breakout['draft_year'] == row['draft_year'])
        updated_breakout.loc[mask, 'breakout_age'] = row['new_breakout_age']

# Show before/after counts
old_missing = breakout_ages['breakout_age'].isna().sum()
new_missing = updated_breakout['breakout_age'].isna().sum()

print(f"\nMissing breakout ages: {old_missing} → {new_missing}")
print(f"Fixed: {old_missing - new_missing} players")

# Save
updated_breakout.to_csv('data/wr_breakout_age_scores_fixed.csv', index=False)
print(f"\nSaved: data/wr_breakout_age_scores_fixed.csv")

# Show remaining missing
still_missing = updated_breakout[updated_breakout['breakout_age'].isna()]
if len(still_missing) > 0:
    print(f"\nStill missing breakout age ({len(still_missing)} players):")
    for _, row in still_missing.iterrows():
        print(f"  - {row['player_name']} ({row['draft_year']})")
