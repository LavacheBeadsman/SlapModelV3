"""
Fix 2016-2018 WR breakout ages using peak season age methodology.

The analysis showed that using peak season ages improves SLAP correlation
from r=0.5077 to r=0.5197. This script updates the breakout ages for
2016-2018 classes to use peak season age calculation.

Peak season age = age during final college season (draft_year - 1)
Calculated as: floor(season_midpoint - birthdate)
Where season_midpoint = October 15 of (draft_year - 1)
"""

import pandas as pd
from datetime import datetime
import numpy as np

def calculate_peak_season_age(birthdate_str, draft_year):
    """Calculate age during peak/final college season."""
    if pd.isna(birthdate_str) or birthdate_str == '':
        return None

    try:
        birthdate = pd.to_datetime(birthdate_str)
        # Season midpoint: October 15 of the year before draft
        season_midpoint = datetime(draft_year - 1, 10, 15)
        age = (season_midpoint - birthdate).days / 365.25
        return int(age)  # Floor to integer age
    except:
        return None

def main():
    # Load data
    df = pd.read_csv('data/wr_backtest_expanded_final.csv')

    print("=" * 70)
    print("FIXING 2016-2018 WR BREAKOUT AGES")
    print("=" * 70)

    # Filter to 2016-2018
    mask_2016_2018 = df['draft_year'].isin([2016, 2017, 2018])
    players_2016_2018 = df[mask_2016_2018].copy()

    print(f"\nTotal 2016-2018 WRs: {len(players_2016_2018)}")

    # Calculate peak season ages
    changes = []

    for idx in players_2016_2018.index:
        row = df.loc[idx]
        name = row['player_name']
        draft_year = row['draft_year']
        old_age = row['breakout_age']
        birthdate = row['birthdate']

        new_age = calculate_peak_season_age(birthdate, draft_year)

        if new_age is not None and old_age != new_age:
            changes.append({
                'name': name,
                'draft_year': draft_year,
                'old_age': old_age,
                'new_age': new_age,
                'diff': new_age - old_age if pd.notna(old_age) else None,
                'birthdate': birthdate
            })
            df.loc[idx, 'breakout_age'] = new_age

    # Print changes summary
    print(f"\nPlayers with changed breakout ages: {len(changes)}")
    print("\n" + "-" * 70)
    print(f"{'Player':<30} {'Year':>6} {'Old':>6} {'New':>6} {'Diff':>6}")
    print("-" * 70)

    for c in sorted(changes, key=lambda x: (x['draft_year'], x['name'])):
        old_str = f"{c['old_age']:.0f}" if pd.notna(c['old_age']) else "N/A"
        diff_str = f"{c['diff']:+.0f}" if c['diff'] is not None else "N/A"
        print(f"{c['name']:<30} {c['draft_year']:>6} {old_str:>6} {c['new_age']:>6} {diff_str:>6}")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY BY DRAFT YEAR")
    print("=" * 70)

    for year in [2016, 2017, 2018]:
        year_changes = [c for c in changes if c['draft_year'] == year]
        year_total = len(df[df['draft_year'] == year])
        diffs = [c['diff'] for c in year_changes if c['diff'] is not None]
        avg_diff = np.mean(diffs) if diffs else 0
        print(f"{year}: {len(year_changes)}/{year_total} players changed, avg diff: {avg_diff:+.1f} years")

    # Save updated data
    df.to_csv('data/wr_backtest_expanded_final.csv', index=False)
    print(f"\nSaved updated data to data/wr_backtest_expanded_final.csv")

    return df

if __name__ == "__main__":
    main()
