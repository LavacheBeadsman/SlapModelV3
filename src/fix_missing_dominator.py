"""
Fix missing Dominator Ratings by adding manual data for name mismatches.
"""

import pandas as pd

# Manual data for players with name mismatches (from CFBD lookup)
MANUAL_PLAYER_DATA = {
    # (player_name, draft_year): (player_rec_tds, team_rec_yards, team_rec_tds)
    ("Ja'Marr Chase", 2021): (20, 6024, 61),      # LSU 2019
    ("Tutu Atwell", 2021): (7, 2686, 24),          # Louisville 2020 (Chatarius Atwell)
    ("Tank Dell", 2023): (17, 4082, 36),           # Houston 2022 (Nathaniel Dell)
    ("Michael Woods II", 2022): (2, 3425, 35),    # Oklahoma 2021
    ("Nico Collins", 2021): (None, None, None),   # Opted out 2020, use 2019 data
}

# Nico Collins 2019 Michigan data (before opt-out)
# He had 7 TDs in 2019, team had 23 passing TDs and 3261 yards
MANUAL_PLAYER_DATA[("Nico Collins", 2021)] = (7, 3261, 23)


def main():
    # Load existing data
    df = pd.read_csv("data/wr_dominator_full.csv")
    print(f"Loaded {len(df)} WRs")

    # Load backtest for player yards
    bt = pd.read_csv("data/backtest_college_stats.csv")
    bt_lookup = dict(zip(zip(bt['player_name'], bt['draft_year']), bt['rec_yards']))

    # Fix missing players
    fixed = 0
    for idx, row in df.iterrows():
        key = (row['player_name'], row['draft_year'])
        if key in MANUAL_PLAYER_DATA:
            player_tds, team_yards, team_tds = MANUAL_PLAYER_DATA[key]

            if player_tds is None:
                continue

            player_yards = bt_lookup.get(key, row['player_rec_yards'])

            # Calculate Dominator
            yards_share = player_yards / team_yards if team_yards else None
            tds_share = player_tds / team_tds if team_tds else None

            if yards_share and tds_share:
                dominator = (yards_share + tds_share) / 2 * 100

                df.loc[idx, 'player_rec_tds'] = player_tds
                df.loc[idx, 'team_rec_yards'] = team_yards
                df.loc[idx, 'team_rec_tds'] = team_tds
                df.loc[idx, 'yards_share'] = round(yards_share * 100, 1)
                df.loc[idx, 'tds_share'] = round(tds_share * 100, 1)
                df.loc[idx, 'dominator_rating'] = round(dominator, 1)

                print(f"Fixed: {row['player_name']} ({row['draft_year']}) = {dominator:.1f}%")
                fixed += 1

    print(f"\nFixed {fixed} players")

    # Save
    df.to_csv("data/wr_dominator_full.csv", index=False)
    print(f"Saved to data/wr_dominator_full.csv")

    # Show final summary
    valid = df['dominator_rating'].notna().sum()
    print(f"\nFinal: {valid}/{len(df)} valid ({valid/len(df)*100:.1f}%)")

    # Show top 10
    print("\nTOP 10 DOMINATOR RATINGS:")
    top = df.nlargest(10, 'dominator_rating')
    for i, (_, r) in enumerate(top.iterrows(), 1):
        print(f"{i:2}. {r['player_name']:25s}: {r['dominator_rating']:.1f}% "
              f"(yards: {r['yards_share']:.1f}%, TDs: {r['tds_share']:.1f}%)")

    # Show remaining missing
    missing = df[df['dominator_rating'].isna()]
    print(f"\nStill missing ({len(missing)}):")
    for _, r in missing.iterrows():
        print(f"  {r['player_name']} ({r['college']})")


if __name__ == "__main__":
    main()
