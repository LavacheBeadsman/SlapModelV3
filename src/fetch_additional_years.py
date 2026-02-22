"""
Fetch college stats for 2020, 2021, 2025 draft classes.
Adds to existing backtest data.
"""

import os
import re
import time
import requests
import pandas as pd
import nflreadpy as nfl
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# School name mappings (NFLverse -> CFBD)
SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State", "Penn St.": "Penn State", "Michigan St.": "Michigan State",
    "North Dakota St.": "North Dakota State", "Oklahoma St.": "Oklahoma State",
    "Miss. St.": "Mississippi State", "Mississippi St.": "Mississippi State",
    "San Diego St.": "San Diego State", "Florida St.": "Florida State",
    "Fresno St.": "Fresno State", "Boise St.": "Boise State", "Iowa St.": "Iowa State",
    "Arizona St.": "Arizona State", "Kansas St.": "Kansas State", "N.C. State": "NC State",
    "Appalachian St.": "Appalachian State", "Colorado St.": "Colorado State",
    "Oregon St.": "Oregon State", "Washington St.": "Washington State",
    "S. Carolina": "South Carolina", "Miami (FL)": "Miami", "Central Mich.": "Central Michigan",
    "Western Mich.": "Western Michigan", "Eastern Mich.": "Eastern Michigan",
    "Northern Ill.": "Northern Illinois", "Southern Miss.": "Southern Mississippi",
    "San Jose St.": "San Jose State", "La.-Monroe": "Louisiana Monroe",
    "La.-Lafayette": "Louisiana", "Louisiana-Lafayette": "Louisiana",
    "Pitt": "Pittsburgh", "Boston Col.": "Boston College", "Mississippi": "Ole Miss",
    "Ala-Birmingham": "UAB", "South Dakota St.": "South Dakota State",
    "Central Fla.": "UCF", "Central Florida": "UCF",
}


def normalize_school(school):
    if pd.isna(school):
        return None
    return SCHOOL_MAPPINGS.get(str(school).strip(), str(school).strip())


def names_match(name1, name2):
    if pd.isna(name1) or pd.isna(name2):
        return False
    n1 = str(name1).lower().strip()
    n2 = str(name2).lower().strip()
    if n1 == n2:
        return True
    suffix_pattern = r'\s+(jr\.?|sr\.?|ii|iii|iv)$'
    n1 = re.sub(suffix_pattern, '', n1).strip()
    n2 = re.sub(suffix_pattern, '', n2).strip()
    if n1 == n2:
        return True
    parts1 = n1.split()
    parts2 = n2.split()
    if len(parts1) >= 2 and len(parts2) >= 2:
        if parts1[-1] == parts2[-1] and parts1[0][0] == parts2[0][0]:
            return True
    return False


def fetch_team_receiving(team, year):
    url = f"{BASE_URL}/stats/player/season"
    params = {"year": year, "category": "receiving", "team": team}
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code == 200:
        stats = {}
        for s in resp.json():
            player = s.get("player", "").lower()
            if player not in stats:
                stats[player] = {}
            try:
                stats[player][s.get("statType", "")] = int(float(s.get("stat", 0)))
            except:
                pass
        return stats
    return {}


def fetch_team_pass_attempts(team, year):
    url = f"{BASE_URL}/stats/season"
    params = {"year": year, "team": team}
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code == 200:
        for stat in resp.json():
            if stat.get("statName") == "passAttempts":
                return int(stat.get("statValue", 0))
    return None


def find_player_receiving(name, team_stats):
    if not team_stats:
        return None
    for api_name, stats in team_stats.items():
        if names_match(name, api_name):
            return stats.get("YDS")
    return None


def fetch_draft_class(draft_year):
    """Fetch college stats for a single draft class."""
    college_year = draft_year - 1

    print(f"\n{'='*60}")
    print(f"DRAFT YEAR {draft_year} (College stats from {college_year})")
    print(f"{'='*60}")

    # Load draft picks
    draft_df = pd.read_parquet("data/nflverse/draft_picks.parquet")
    picks = draft_df[
        (draft_df['season'] == draft_year) &
        (draft_df['position'].isin(['RB', 'WR']))
    ].copy()

    print(f"Found {len(picks)} RB/WR picks")

    # Load combine data
    combine = nfl.load_combine().to_pandas()
    combine = combine[combine['season'] == draft_year]

    # Cache for team stats
    team_cache = {}
    pass_att_cache = {}

    results = []
    found_rec = 0

    for _, player in picks.iterrows():
        name = player['pfr_player_name']
        position = player['position']
        college = normalize_school(player['college'])
        pick = player['pick']
        age = player['age']

        rec_yards = None
        team_pass_att = None

        if college:
            # Get team receiving stats
            if (college, college_year) not in team_cache:
                team_cache[(college, college_year)] = fetch_team_receiving(college, college_year)
                time.sleep(0.05)

            team_stats = team_cache[(college, college_year)]
            rec_yards = find_player_receiving(name, team_stats)

            # Get team pass attempts
            if (college, college_year) not in pass_att_cache:
                pass_att_cache[(college, college_year)] = fetch_team_pass_attempts(college, college_year)
                time.sleep(0.05)

            team_pass_att = pass_att_cache[(college, college_year)]

        # Get combine data
        player_combine = combine[combine['player_name'].apply(lambda x: names_match(name, x))]
        weight = player_combine.iloc[0]['wt'] if len(player_combine) > 0 else None
        forty = player_combine.iloc[0]['forty'] if len(player_combine) > 0 else None

        if rec_yards:
            found_rec += 1

        # Determine sources
        weight_source = 'combine' if pd.notna(weight) else 'missing'
        forty_source = 'combine' if pd.notna(forty) else 'missing'

        results.append({
            'draft_year': draft_year,
            'player_name': name,
            'position': position,
            'college': college or player['college'],
            'pick': pick,
            'age': age,
            'rec_yards': rec_yards if rec_yards else '',
            'team_pass_attempts': team_pass_att if team_pass_att else '',
            'weight': weight if pd.notna(weight) else '',
            'forty': forty if pd.notna(forty) else '',
            'weight_source': weight_source,
            'forty_source': forty_source,
        })

    print(f"Found rec_yards: {found_rec}/{len(picks)}")
    return pd.DataFrame(results)


def main():
    print("="*60)
    print("FETCHING COLLEGE STATS FOR 2020, 2021, 2025 DRAFT CLASSES")
    print("="*60)

    all_results = []

    for year in [2020, 2021, 2025]:
        df = fetch_draft_class(year)
        all_results.append(df)

    # Combine new data
    new_data = pd.concat(all_results, ignore_index=True)

    # Load existing backtest data
    existing = pd.read_csv("data/backtest_college_stats.csv")
    print(f"\nExisting backtest data: {len(existing)} players")

    # Combine
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.sort_values(['draft_year', 'pick']).reset_index(drop=True)

    # Add full_athletic_data column
    def has_full_athletic(row):
        try:
            w = float(row['weight'])
            f = float(row['forty'])
            return pd.notna(w) and pd.notna(f)
        except (ValueError, TypeError):
            return False

    combined['full_athletic_data'] = combined.apply(has_full_athletic, axis=1)

    # Save
    combined.to_csv("data/backtest_college_stats.csv", index=False)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total players: {len(combined)}")
    print()

    for year in sorted(combined['draft_year'].unique()):
        y = combined[combined['draft_year'] == year]
        rec = len(y[y['rec_yards'] != ''])
        tpa = len(y[y['team_pass_attempts'] != ''])
        wt = len(y[y['weight'] != ''])
        ft = len(y[(y['forty'] != '') & (y['forty'] != 'NONE')])
        print(f"{int(year)}: {len(y)} players | rec: {rec} | tpa: {tpa} | wt: {wt} | forty: {ft}")


if __name__ == "__main__":
    main()
