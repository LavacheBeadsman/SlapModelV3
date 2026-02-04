"""
Fix player name mismatches in WR dominator data.

For players not found, try alternative name spellings and manual lookups.
"""

import time
import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# CFBD API configuration
API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

# Manual name mappings for known mismatches
# Format: (our_name, draft_year): (cfbd_name, cfbd_team, season)
MANUAL_MAPPINGS = {
    # 2015 draft
    ("Dorial Green-Beckham", 2015): ("Dorial Green-Beckham", "Missouri", 2013),  # Transferred from Oklahoma, played at Missouri
    ("Rashad Greene", 2015): ("Rashad Greene Sr.", "Florida State", 2014),

    # 2016 draft
    ("Will Fuller", 2016): ("Will Fuller V", "Notre Dame", 2015),
    ("Mike Thomas", 2016): ("Michael Thomas", "Southern Miss", 2015),
    ("Devin Fuller", 2016): ("Devin Fuller", "UCLA", 2015),

    # 2018 draft
    ("Antonio Callaway", 2018): ("Antonio Callaway", "Florida", 2016),  # Suspended in 2017, use 2016

    # 2021 draft
    ("Ja'Marr Chase", 2021): ("Ja'Marr Chase", "LSU", 2019),  # Opted out 2020, use 2019
    ("Tutu Atwell", 2021): ("Chatarius Atwell", "Louisville", 2020),
    ("Nico Collins", 2021): ("Nico Collins", "Michigan", 2019),  # Opted out 2020, use 2019

    # 2022 draft
    ("Christian Watson", 2022): ("Christian Watson", "North Dakota State", 2021),  # FCS but try anyway

    # 2023 draft
    ("Tank Dell", 2023): ("Nathaniel Dell", "Houston", 2022),

    # 2024 draft
    ("Bub Means", 2024): ("Konata Mumpfield", "Pittsburgh", 2023),  # Bub Means is nickname, real name different

    # 2026 draft - try searching by last name
    ("Reggie Virgil", 2026): ("Reggie Virgil", "Texas Tech", 2024),
    ("Raymond Cottrell", 2026): ("Raymond Cottrell", "Texas A&M", 2024),
    ("Cordale Russell", 2026): ("Cordale Russell", "Colorado", 2024),
    ("Braylon James", 2026): ("Braylon James", "Notre Dame", 2024),
    ("Antonio Gates Jr.", 2026): ("Antonio Gates Jr.", "Michigan State", 2024),
    ("Talyn Shettron", 2026): ("Talyn Shettron", "Oklahoma State", 2024),
    ("Jaron Glover", 2026): ("Jaron Glover", "Michigan State", 2024),
    ("Ja'Mori Maclin", 2026): ("Ja'Mori Maclin", "Kentucky", 2024),
    ("Ja'Varrius Johnson", 2026): ("Ja'Varrius Johnson", "Auburn", 2024),
}

# School name mappings
SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State",
    "Penn St.": "Penn State",
    "Michigan St.": "Michigan State",
    "Florida St.": "Florida State",
    "Oklahoma St.": "Oklahoma State",
    "Miss. St.": "Mississippi State",
    "Mississippi": "Ole Miss",
    "Pitt": "Pittsburgh",
    "Central Florida": "UCF",
    "La-Monroe": "Louisiana Monroe",
    "Southern Miss": "Southern Mississippi",
    "UConn": "Connecticut",
    "North Dakota St.": "North Dakota State",
}


def normalize_school(school):
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def fetch_team_receiving_stats(team, year):
    """Fetch all player receiving stats for a team in a season."""
    url = f"{BASE_URL}/stats/player/season"
    params = {
        "year": year,
        "category": "receiving",
        "team": team
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            player_stats = {}
            for stat in data:
                player = stat.get("player", "").strip()
                stat_type = stat.get("statType", "")
                value = stat.get("stat", 0)

                if player not in player_stats:
                    player_stats[player] = {"player_name": player, "YDS": 0, "TD": 0, "REC": 0}

                try:
                    if stat_type == "YDS":
                        player_stats[player]["YDS"] = int(float(value))
                    elif stat_type == "TD":
                        player_stats[player]["TD"] = int(float(value))
                    elif stat_type == "REC":
                        player_stats[player]["REC"] = int(float(value))
                except (ValueError, TypeError):
                    pass

            team_rec_yards = sum(p.get("YDS", 0) for p in player_stats.values())
            team_rec_tds = sum(p.get("TD", 0) for p in player_stats.values())

            return {
                "players": player_stats,
                "team_rec_yards": team_rec_yards,
                "team_rec_tds": team_rec_tds
            }
        elif response.status_code == 429:
            time.sleep(5)
            return fetch_team_receiving_stats(team, year)
        else:
            return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def search_player_fuzzy(player_name, team_data):
    """Search for player with fuzzy matching."""
    if not team_data or "players" not in team_data:
        return None, None, None, None

    # Get last name
    parts = player_name.replace(".", "").replace("'", "").split()
    if len(parts) < 2:
        return None, None, None, None

    last_name = parts[-1].lower()
    first_initial = parts[0][0].lower() if parts[0] else ""

    # Remove common suffixes
    suffixes = ['jr', 'sr', 'ii', 'iii', 'iv', 'v']
    if last_name in suffixes and len(parts) > 2:
        last_name = parts[-2].lower()

    for api_name, stats in team_data["players"].items():
        api_parts = api_name.replace(".", "").replace("'", "").lower().split()
        if len(api_parts) < 2:
            continue

        api_last = api_parts[-1]
        if api_last in suffixes and len(api_parts) > 2:
            api_last = api_parts[-2]
        api_first_initial = api_parts[0][0] if api_parts[0] else ""

        # Match by last name and first initial
        if last_name == api_last and first_initial == api_first_initial:
            return stats.get("YDS", 0), stats.get("TD", 0), api_name, "fuzzy_match"

        # Match by last name only (if unique)
        if last_name == api_last:
            # Check if this is the only player with this last name
            count = sum(1 for n in team_data["players"].keys()
                       if n.replace(".", "").replace("'", "").lower().split()[-1] == last_name)
            if count == 1:
                return stats.get("YDS", 0), stats.get("TD", 0), api_name, "last_name_match"

    return None, None, None, None


def calculate_dominator(player_rec_yards, player_rec_tds, team_rec_yards, team_rec_tds):
    """Calculate dominator rating."""
    if team_rec_yards is None or team_rec_yards == 0:
        return None, None, None

    yards_share = player_rec_yards / team_rec_yards

    if team_rec_tds is None or team_rec_tds == 0:
        td_share = yards_share
    else:
        td_share = player_rec_tds / team_rec_tds

    dominator = (yards_share + td_share) / 2
    return yards_share, td_share, dominator


def main():
    print("=" * 90)
    print("FIXING WR DOMINATOR NAME MISMATCHES")
    print("=" * 90)

    # Load existing results
    df = pd.read_csv('data/wr_dominator_complete.csv')
    not_found = df[df['status'] == 'player_not_found'].copy()

    print(f"\nPlayers to fix: {len(not_found)}")

    fixed_count = 0
    results = []

    for idx, row in not_found.iterrows():
        player_name = row['player_name']
        draft_year = row['draft_year']
        college = row['college']
        original_season = row['season_used']

        print(f"\n[{fixed_count+1}/{len(not_found)}] {player_name} ({college} {original_season})...")

        # Check manual mapping first
        key = (player_name, draft_year)
        if key in MANUAL_MAPPINGS:
            cfbd_name, cfbd_team, season = MANUAL_MAPPINGS[key]
            print(f"  Manual mapping: {cfbd_name} at {cfbd_team} ({season})")

            # Fetch team data
            team_data = fetch_team_receiving_stats(cfbd_team, season)
            time.sleep(0.3)

            if team_data:
                # Search for player
                for api_name, stats in team_data["players"].items():
                    if cfbd_name.lower() in api_name.lower() or api_name.lower() in cfbd_name.lower():
                        player_yards = stats.get("YDS", 0)
                        player_tds = stats.get("TD", 0)
                        team_yards = team_data["team_rec_yards"]
                        team_tds = team_data["team_rec_tds"]

                        yards_share, td_share, dominator = calculate_dominator(
                            player_yards, player_tds, team_yards, team_tds
                        )

                        if dominator is not None:
                            print(f"  FOUND: {player_yards} yds, {player_tds} TDs -> {dominator*100:.1f}%")
                            results.append({
                                'player_name': player_name,
                                'draft_year': draft_year,
                                'college': college,
                                'season_used': season,
                                'player_rec_yards': player_yards,
                                'player_rec_tds': player_tds,
                                'team_rec_yards': team_yards,
                                'team_rec_tds': team_tds,
                                'yards_market_share': yards_share,
                                'td_market_share': td_share,
                                'dominator_rating': dominator,
                                'dominator_pct': dominator * 100,
                                'status': 'fixed_manual',
                                'cfbd_name': api_name
                            })
                            fixed_count += 1
                            break
                else:
                    # Try fuzzy search
                    player_yards, player_tds, api_name, match_type = search_player_fuzzy(cfbd_name, team_data)
                    if player_yards is not None:
                        team_yards = team_data["team_rec_yards"]
                        team_tds = team_data["team_rec_tds"]
                        yards_share, td_share, dominator = calculate_dominator(
                            player_yards, player_tds, team_yards, team_tds
                        )
                        if dominator is not None:
                            print(f"  FOUND (fuzzy): {api_name} - {player_yards} yds, {player_tds} TDs -> {dominator*100:.1f}%")
                            results.append({
                                'player_name': player_name,
                                'draft_year': draft_year,
                                'college': college,
                                'season_used': season,
                                'player_rec_yards': player_yards,
                                'player_rec_tds': player_tds,
                                'team_rec_yards': team_yards,
                                'team_rec_tds': team_tds,
                                'yards_market_share': yards_share,
                                'td_market_share': td_share,
                                'dominator_rating': dominator,
                                'dominator_pct': dominator * 100,
                                'status': f'fixed_{match_type}',
                                'cfbd_name': api_name
                            })
                            fixed_count += 1
                            continue
                    print(f"  NOT FOUND in manual mapping team data")
            else:
                print(f"  Could not fetch team data for {cfbd_team} {season}")
            continue

        # Try fuzzy search on original team/season
        if row['team_rec_yards'] > 0:
            cfbd_school = normalize_school(college)
            if cfbd_school:
                team_data = fetch_team_receiving_stats(cfbd_school, original_season)
                time.sleep(0.3)

                if team_data:
                    player_yards, player_tds, api_name, match_type = search_player_fuzzy(player_name, team_data)
                    if player_yards is not None:
                        team_yards = team_data["team_rec_yards"]
                        team_tds = team_data["team_rec_tds"]
                        yards_share, td_share, dominator = calculate_dominator(
                            player_yards, player_tds, team_yards, team_tds
                        )
                        if dominator is not None:
                            print(f"  FOUND (fuzzy): {api_name} - {player_yards} yds, {player_tds} TDs -> {dominator*100:.1f}%")
                            results.append({
                                'player_name': player_name,
                                'draft_year': draft_year,
                                'college': college,
                                'season_used': original_season,
                                'player_rec_yards': player_yards,
                                'player_rec_tds': player_tds,
                                'team_rec_yards': team_yards,
                                'team_rec_tds': team_tds,
                                'yards_market_share': yards_share,
                                'td_market_share': td_share,
                                'dominator_rating': dominator,
                                'dominator_pct': dominator * 100,
                                'status': f'fixed_{match_type}',
                                'cfbd_name': api_name
                            })
                            fixed_count += 1
                            continue

        print(f"  Still not found")

    # Save fixes
    if results:
        fixes_df = pd.DataFrame(results)
        fixes_df.to_csv('data/wr_dominator_fixes.csv', index=False)
        print(f"\n\nSaved {len(fixes_df)} fixes to data/wr_dominator_fixes.csv")

        # Update main file
        print("\nUpdating main dominator file...")
        for _, fix in fixes_df.iterrows():
            mask = (df['player_name'] == fix['player_name']) & (df['draft_year'] == fix['draft_year'])
            for col in ['season_used', 'player_rec_yards', 'player_rec_tds', 'team_rec_yards',
                       'team_rec_tds', 'yards_market_share', 'td_market_share',
                       'dominator_rating', 'dominator_pct', 'status']:
                df.loc[mask, col] = fix[col]

        df.to_csv('data/wr_dominator_complete.csv', index=False)
        print("Updated data/wr_dominator_complete.csv")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Players attempted: {len(not_found)}")
    print(f"Players fixed: {fixed_count}")
    print(f"Still missing: {len(not_found) - fixed_count}")

    # Show remaining missing
    remaining = df[df['status'].isin(['player_not_found', 'no_cfbd_data'])]
    if len(remaining) > 0:
        print("\n--- Still Missing ---")
        print(remaining[['player_name', 'draft_year', 'college', 'status']].to_string(index=False))


if __name__ == "__main__":
    main()
