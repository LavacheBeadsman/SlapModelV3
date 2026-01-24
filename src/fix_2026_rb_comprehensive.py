"""
Comprehensive 2026 RB Fix - Multi-Season + Transfer Handling

Key fixes:
1. Use BEST season production (not just 2024)
2. Handle transfers by checking previous schools
3. Fix name spelling mismatches
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# =============================================================================
# MANUAL CORRECTIONS FOR TRANSFERS AND NAME MISMATCHES
# =============================================================================

# (player_name, current_school) -> [(actual_school, cfbd_name), ...]
TRANSFER_CORRECTIONS = {
    ("Roman Hemby", "Indiana"): [("Maryland", "Roman Hemby")],
    ("C.J. Donaldson", "Ohio State"): [("West Virginia", "CJ Donaldson")],
    ("Jaydn Ott", "California"): [("California", "Jadyn Ott")],
    ("Terion Stewart", "Virginia Tech"): [("Bowling Green", "Terion Stewart")],
    ("Djay Braswell", "Georgia State"): [("South Carolina", "Djay Braswell"), ("Georgia State", "Djay Braswell")],
    ("Chip Trayanum", "Toledo"): [("Ohio State", "Chip Trayanum"), ("Arizona State", "Chip Trayanum")],
    ("Mike Washington Jr.", "Arkansas"): [("Arkansas", "Mike Washington")],
}

# Standard school name mappings
SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State", "Penn St.": "Penn State", "Michigan St.": "Michigan State",
    "Florida St.": "Florida State", "San Diego St.": "San Diego State",
    "Miss. St.": "Mississippi State", "Mississippi": "Ole Miss", "Pitt": "Pittsburgh",
    "Stephen F. Austin": None, "North Dakota State": None, "UT Martin": None,
}

WEIGHT_DC = 0.85
WEIGHT_PRODUCTION = 0.10
WEIGHT_RAS = 0.05


def normalize_school(school):
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def fetch_with_retry(url, params, max_retries=3):
    """Fetch with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                wait_time = 2 ** attempt
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
    return None


def fetch_team_receiving(team, year):
    url = f"{BASE_URL}/stats/player/season"
    params = {"year": year, "category": "receiving", "team": team}
    result = fetch_with_retry(url, params)
    return result if result else []


def fetch_team_pass_attempts(team, year):
    url = f"{BASE_URL}/stats/season"
    params = {"year": year, "team": team}
    result = fetch_with_retry(url, params)
    if result:
        for stat in result:
            if stat.get("statName") == "passAttempts":
                return float(stat.get("statValue", 0))
    return None


def names_match(name1, name2):
    """Flexible name matching"""
    NICKNAMES = {
        'nick': 'nicholas', 'nicholas': 'nick',
        'mike': 'michael', 'michael': 'mike',
        'cj': 'c j', 'ej': 'e j', 'dj': 'd j', 'aj': 'a j',
    }

    n1 = name1.lower().strip().replace('.', ' ').replace("'", "").replace("-", " ")
    n2 = name2.lower().strip().replace('.', ' ').replace("'", "").replace("-", " ")

    if n1 == n2:
        return True

    parts1 = n1.split()
    parts2 = n2.split()

    suffixes = {'jr', 'ii', 'iii', 'iv', 'sr'}
    clean1 = [p for p in parts1 if p not in suffixes]
    clean2 = [p for p in parts2 if p not in suffixes]

    if len(clean1) < 2 or len(clean2) < 2:
        return False

    if clean1[-1] != clean2[-1]:
        return False

    first1, first2 = clean1[0], clean2[0]

    if first1 == first2:
        return True

    if NICKNAMES.get(first1) == first2 or NICKNAMES.get(first2) == first1:
        return True

    if len(first1) >= 3 and len(first2) >= 3 and first1[:3] == first2[:3]:
        return True

    # Handle Jaydn/Jadyn type differences
    if len(first1) >= 4 and len(first2) >= 4:
        if first1[:2] == first2[:2] and first1[-2:] == first2[-2:]:
            return True

    return False


def get_all_seasons_data(player_name, schools_to_check, years=[2025, 2024, 2023, 2022, 2021]):
    """Get all seasons of receiving data for a player, checking multiple schools"""
    all_seasons = []

    for school, cfbd_name in schools_to_check:
        cfbd_school = normalize_school(school)
        if cfbd_school is None:
            continue

        for year in years:
            stats = fetch_team_receiving(cfbd_school, year)
            time.sleep(0.5)

            players = {}
            for s in stats:
                name = s.get("player", "")
                stat_type = s.get("statType", "")
                value = s.get("stat", 0)

                if name not in players:
                    players[name] = {'year': year, 'team': cfbd_school}
                try:
                    players[name][stat_type] = int(float(value))
                except:
                    players[name][stat_type] = 0

            # Find matching player
            for api_name, data in players.items():
                if names_match(cfbd_name or player_name, api_name):
                    pass_att = fetch_team_pass_attempts(cfbd_school, year)
                    time.sleep(0.5)
                    data['pass_att'] = pass_att
                    data['api_name'] = api_name
                    all_seasons.append(data)
                    break

    return all_seasons


def dc_score(pick):
    if pd.isna(pick):
        return 0
    return max(0, min(100, 100 - 2.40 * (pick**0.62 - 1)))


def age_weight(age):
    if pd.isna(age) or str(age) == 'MISSING':
        return 1.0
    try:
        age = float(age)
    except:
        return 1.0

    college_age = age - 1
    if college_age <= 19:
        return 1.20
    elif college_age == 20:
        return 1.10
    elif college_age == 21:
        return 1.00
    elif college_age == 22:
        return 0.90
    else:
        return 0.80


def main():
    print("=" * 100)
    print("COMPREHENSIVE 2026 RB FIX")
    print("Using BEST season + transfer handling")
    print("=" * 100)

    # Load backtest for normalization
    rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
    rb_backtest['rec_yards'] = pd.to_numeric(rb_backtest['rec_yards'], errors='coerce')
    rb_backtest['team_pass_att'] = pd.to_numeric(rb_backtest['team_pass_att'], errors='coerce')
    rb_backtest['age'] = pd.to_numeric(rb_backtest['age'], errors='coerce')

    bt_has_data = (rb_backtest['rec_yards'].notna()) & (rb_backtest['team_pass_att'].notna()) & (rb_backtest['team_pass_att'] > 0)
    rb_backtest.loc[bt_has_data, 'rec_per_pass_att'] = rb_backtest.loc[bt_has_data, 'rec_yards'] / rb_backtest.loc[bt_has_data, 'team_pass_att']
    rb_backtest.loc[bt_has_data, 'age_wt'] = rb_backtest.loc[bt_has_data, 'age'].apply(age_weight)
    rb_backtest.loc[bt_has_data, 'production_raw'] = rb_backtest.loc[bt_has_data, 'rec_per_pass_att'] * rb_backtest.loc[bt_has_data, 'age_wt']

    PROD_MIN = rb_backtest.loc[bt_has_data, 'production_raw'].min()
    PROD_MAX = rb_backtest.loc[bt_has_data, 'production_raw'].max()

    rb_backtest.loc[bt_has_data, 'production_score'] = (
        (rb_backtest.loc[bt_has_data, 'production_raw'] - PROD_MIN) / (PROD_MAX - PROD_MIN) * 100
    )

    AVG_PRODUCTION = rb_backtest['production_score'].mean()
    AVG_RAS = 66.5  # From backtest

    print(f"\nBacktest normalization: min={PROD_MIN:.4f}, max={PROD_MAX:.4f}, avg={AVG_PRODUCTION:.1f}")

    # Load 2026 prospects
    prospects = pd.read_csv('data/prospects_final.csv')
    rb_2026 = prospects[prospects['position'] == 'RB'].copy()
    print(f"Loaded {len(rb_2026)} RB prospects")

    # Process each player
    results = []

    for idx, row in rb_2026.iterrows():
        player = row['player_name']
        school = row['school']
        pick = row['projected_pick']
        age = row['age']

        print(f"\n--- {player} ({school}) ---")

        # Determine schools to check
        key = (player, school)
        if key in TRANSFER_CORRECTIONS:
            schools_to_check = TRANSFER_CORRECTIONS[key]
            print(f"  Using transfer correction: {schools_to_check}")
        else:
            cfbd_school = normalize_school(school)
            if cfbd_school:
                schools_to_check = [(school, player)]
            else:
                schools_to_check = []
                print(f"  FCS school - skipping CFBD lookup")

        # Get all seasons
        if schools_to_check:
            all_seasons = get_all_seasons_data(player, schools_to_check)
        else:
            all_seasons = []

        # Find BEST season (highest yds/pass_att ratio)
        best_season = None
        best_production_raw = 0

        for s in all_seasons:
            yds = s.get('YDS', 0) or 0
            pass_att = s.get('pass_att', 0) or 0
            if pass_att > 0:
                raw = yds / pass_att * age_weight(age)
                if raw > best_production_raw:
                    best_production_raw = raw
                    best_season = s

        # Calculate production score
        if best_season:
            yds = best_season.get('YDS', 0)
            rec = best_season.get('REC', 0)
            pass_att = best_season.get('pass_att', 0)
            year = best_season.get('year')
            team = best_season.get('team')

            production_score = ((best_production_raw - PROD_MIN) / (PROD_MAX - PROD_MIN) * 100)
            production_score = max(0, min(100, production_score))
            production_status = 'observed'

            print(f"  BEST: {year} at {team} - {yds} yds, {rec} rec, {pass_att} pass_att")
            print(f"  Production score: {production_score:.1f}")

            if len(all_seasons) > 1:
                print(f"  All seasons found: {[(s['year'], s.get('YDS', 0)) for s in all_seasons]}")
        else:
            yds = None
            rec = None
            pass_att = None
            year = None
            team = None
            production_score = AVG_PRODUCTION
            production_status = 'imputed'
            print(f"  No data found - using average ({AVG_PRODUCTION:.1f})")

        # Calculate SLAP
        dc = dc_score(pick)
        ras = AVG_RAS

        slap = WEIGHT_DC * dc + WEIGHT_PRODUCTION * production_score + WEIGHT_RAS * ras
        delta = slap - dc

        results.append({
            'player_name': player,
            'school': school,
            'projected_pick': pick,
            'age': age,
            'slap_score': slap,
            'dc_score': dc,
            'production_score': production_score,
            'ras_score': ras,
            'delta_vs_dc': delta,
            'production_status': production_status,
            'ras_status': 'imputed',
            'rec_yards': yds,
            'receptions': rec,
            'team_pass_att': pass_att,
            'best_season_year': year,
            'best_season_team': team,
        })

    # Create output dataframe
    df = pd.DataFrame(results)
    df = df.sort_values('slap_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    # Print results
    print("\n" + "=" * 100)
    print("2026 RB RANKINGS (CORRECTED)")
    print("=" * 100)

    print(f"\n{'Rank':<5} {'Player':<25} {'School':<15} {'Pick':>5} {'DC':>6} {'Prod':>5} {'SLAP':>6} {'Best Season':<20}")
    print("-" * 110)

    for _, row in df.head(30).iterrows():
        prod_str = f"{row['production_score']:.0f}" if row['production_status'] == 'observed' else f"{row['production_score']:.0f}*"
        best = f"{row['best_season_year']} {row['best_season_team']}" if row['best_season_year'] else "imputed"
        print(f"{int(row['rank']):<5} {row['player_name']:<25} {row['school']:<15} "
              f"{int(row['projected_pick']):>5} {row['dc_score']:>6.1f} {prod_str:>5} "
              f"{row['slap_score']:>6.1f} {best:<20}")

    print("\n* = imputed production score")

    # Save
    output_cols = [
        'rank', 'player_name', 'school', 'projected_pick', 'age',
        'slap_score', 'dc_score', 'production_score', 'ras_score',
        'delta_vs_dc', 'production_status', 'ras_status',
        'rec_yards', 'receptions', 'team_pass_att',
        'best_season_year', 'best_season_team'
    ]

    df[output_cols].to_csv('output/slap_2026_rb.csv', index=False)
    print(f"\n✓ Saved: output/slap_2026_rb.csv")

    # Summary
    observed = (df['production_status'] == 'observed').sum()
    imputed = (df['production_status'] == 'imputed').sum()
    print(f"\nProduction scores: {observed} observed, {imputed} imputed")


if __name__ == "__main__":
    main()
