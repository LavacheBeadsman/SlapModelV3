"""
Fill missing weight and 40 time data from NFLverse combine data.

Uses nflreadpy to pull official combine data and match to our backtest dataset.
Tracks source of each value (combine, existing, or missing).
"""

import nflreadpy as nfl
import pandas as pd


def names_match(name1, name2):
    """Check if two player names match (handles variations)."""
    if pd.isna(name1) or pd.isna(name2):
        return False

    n1 = str(name1).lower().strip()
    n2 = str(name2).lower().strip()

    if n1 == n2:
        return True

    # Remove suffixes (must be at end of string or followed by space)
    import re
    suffix_pattern = r'\s+(jr\.?|sr\.?|ii|iii|iv)$'
    n1 = re.sub(suffix_pattern, '', n1).strip()
    n2 = re.sub(suffix_pattern, '', n2).strip()

    if n1 == n2:
        return True

    # Try last name match with first initial
    parts1 = n1.split()
    parts2 = n2.split()

    if len(parts1) >= 2 and len(parts2) >= 2:
        # Same last name and first initial matches
        if parts1[-1] == parts2[-1] and parts1[0][0] == parts2[0][0]:
            return True

    return False


# Manual name mappings for known nickname/name variations
NAME_MAPPINGS = {
    'Tank Dell': 'Nathaniel Dell',
    'Bucky Irving': 'Mar\'Keise Irving',
    'Bub Means': 'Jerrod Means',
}


def main():
    print("=" * 70)
    print("FILLING MISSING ATHLETIC DATA FROM NFLVERSE COMBINE")
    print("=" * 70)
    print()

    # Load our backtest data
    bt = pd.read_csv("data/backtest_college_stats.csv")
    print(f"Loaded backtest data: {len(bt)} players")

    # Count current missing values
    bt['weight_num'] = pd.to_numeric(bt['weight'], errors='coerce')
    bt['forty_num'] = pd.to_numeric(bt['forty'], errors='coerce')

    missing_weight_before = bt['weight_num'].isna().sum()
    missing_forty_before = bt['forty_num'].isna().sum()

    print(f"BEFORE: Missing weight: {missing_weight_before}, Missing forty: {missing_forty_before}")
    print()

    # Load combine data from nflreadpy
    print("Loading NFLverse combine data...")
    combine_polars = nfl.load_combine()
    combine = combine_polars.to_pandas()

    # Filter to 2022-2024 RB/WR
    combine = combine[
        (combine['season'].isin([2022, 2023, 2024])) &
        (combine['pos'].isin(['RB', 'WR']))
    ]
    print(f"Combine data: {len(combine)} RB/WR players (2022-2024)")
    print()

    # Initialize source columns
    bt['weight_source'] = ''
    bt['forty_source'] = ''

    # Mark existing values
    bt.loc[bt['weight_num'].notna(), 'weight_source'] = 'existing'
    bt.loc[bt['forty_num'].notna(), 'forty_source'] = 'existing'

    # Match and fill
    print("Matching players and filling gaps...")
    filled_weight = 0
    filled_forty = 0

    for idx, row in bt.iterrows():
        player_name = row['player_name']
        draft_year = row['draft_year']

        # Find matching combine record
        year_combine = combine[combine['season'] == draft_year]

        # Try mapped name first, then original name
        names_to_try = [player_name]
        if player_name in NAME_MAPPINGS:
            names_to_try.insert(0, NAME_MAPPINGS[player_name])

        match = None
        for try_name in names_to_try:
            for _, comb_row in year_combine.iterrows():
                if names_match(try_name, comb_row['player_name']):
                    match = comb_row
                    break
            if match is not None:
                break

        if match is not None:
            # Fill weight if missing
            if pd.isna(row['weight_num']) and pd.notna(match['wt']):
                bt.at[idx, 'weight'] = match['wt']
                bt.at[idx, 'weight_source'] = 'combine'
                filled_weight += 1

            # Fill forty if missing
            if pd.isna(row['forty_num']) and pd.notna(match['forty']):
                bt.at[idx, 'forty'] = match['forty']
                bt.at[idx, 'forty_source'] = 'combine'
                filled_forty += 1

    print(f"Filled from combine: {filled_weight} weights, {filled_forty} forty times")
    print()

    # Recount missing
    bt['weight_num'] = pd.to_numeric(bt['weight'], errors='coerce')
    bt['forty_num'] = pd.to_numeric(bt['forty'], errors='coerce')

    missing_weight_after = bt['weight_num'].isna().sum()
    missing_forty_after = bt['forty_num'].isna().sum()

    # Mark still missing
    bt.loc[(bt['weight_num'].isna()) & (bt['weight_source'] == ''), 'weight_source'] = 'missing'
    bt.loc[(bt['forty_num'].isna()) & (bt['forty_source'] == ''), 'forty_source'] = 'missing'

    # Drop helper columns
    bt = bt.drop(columns=['weight_num', 'forty_num'])

    # Save updated data
    bt.to_csv("data/backtest_college_stats.csv", index=False)

    # Report results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Metric':<20} {'Before':<10} {'After':<10} {'Filled':<10}")
    print("-" * 50)
    print(f"{'Missing weight':<20} {missing_weight_before:<10} {missing_weight_after:<10} {missing_weight_before - missing_weight_after:<10}")
    print(f"{'Missing forty':<20} {missing_forty_before:<10} {missing_forty_after:<10} {missing_forty_before - missing_forty_after:<10}")
    print()

    # Source breakdown
    print("SOURCE BREAKDOWN:")
    print("-" * 50)
    print(f"Weight sources: {bt['weight_source'].value_counts().to_dict()}")
    print(f"Forty sources:  {bt['forty_source'].value_counts().to_dict()}")
    print()

    # List players still missing athletic data
    still_missing_weight = bt[bt['weight_source'] == 'missing']
    still_missing_forty = bt[bt['forty_source'] == 'missing']

    if len(still_missing_weight) > 0:
        print(f"STILL MISSING WEIGHT ({len(still_missing_weight)} players):")
        for _, r in still_missing_weight.iterrows():
            print(f"  {r['player_name']:<25} {r['college']:<20} Pick {r['pick']}")
        print()

    if len(still_missing_forty) > 0:
        print(f"STILL MISSING FORTY ({len(still_missing_forty)} players):")
        for _, r in still_missing_forty.head(20).iterrows():
            print(f"  {r['player_name']:<25} {r['college']:<20} Pick {r['pick']}")
        if len(still_missing_forty) > 20:
            print(f"  ... and {len(still_missing_forty) - 20} more")
        print()


if __name__ == "__main__":
    main()
