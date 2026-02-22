"""
Fix 2026 RB Production: Use FINAL Season Only

Matches backtest methodology:
- Backtest uses: draft_year - 1 (final college season)
- 2026 prospects should use: 2025 (their final college season)

This ensures consistency between what we validated and what we apply.
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

# School name mappings
SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State", "Penn St.": "Penn State", "Michigan St.": "Michigan State",
    "Florida St.": "Florida State", "San Diego St.": "San Diego State",
    "Miss. St.": "Mississippi State", "Mississippi": "Ole Miss", "Pitt": "Pittsburgh",
    "N.C. State": "NC State", "Boise St.": "Boise State",
}

# Transfer corrections: (listed_name, listed_school) -> (cfbd_school, cfbd_name)
TRANSFER_CORRECTIONS = {
    ("Roman Hemby", "Indiana"): ("Indiana", "Roman Hemby"),  # Now at Indiana for 2025
    ("C.J. Donaldson", "Ohio State"): ("Ohio State", "CJ Donaldson"),  # Now at Ohio State
    ("Jaydn Ott", "California"): ("California", "Jadyn Ott"),  # Name spelling
    ("Terion Stewart", "Virginia Tech"): ("Virginia Tech", "Terion Stewart"),
    ("Chip Trayanum", "Toledo"): ("Toledo", "Chip Trayanum"),
    ("Djay Braswell", "Georgia State"): ("Georgia State", "Djay Braswell"),
}


def normalize_school(school):
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def fetch_with_retry(url, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(2 ** attempt)
        except:
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
    """Flexible name matching with nicknames"""
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

    return False


def get_final_season_data(player_name, school, final_year=2025):
    """Get ONLY the final college season data (matching backtest methodology)"""

    # Check for transfer correction
    key = (player_name, school)
    if key in TRANSFER_CORRECTIONS:
        cfbd_school, cfbd_name = TRANSFER_CORRECTIONS[key]
    else:
        cfbd_school = normalize_school(school)
        cfbd_name = player_name

    if cfbd_school is None:
        return None, None, None, None, "fcs_school"

    # Try final year first (2025)
    stats = fetch_team_receiving(cfbd_school, final_year)
    time.sleep(0.5)

    if stats:
        players = {}
        for s in stats:
            name = s.get("player", "")
            if name not in players:
                players[name] = {}
            try:
                players[name][s.get("statType", "")] = int(float(s.get("stat", 0)))
            except:
                players[name][s.get("statType", "")] = 0

        for api_name, data in players.items():
            if names_match(cfbd_name, api_name):
                pass_att = fetch_team_pass_attempts(cfbd_school, final_year)
                time.sleep(0.3)
                return data.get('YDS', 0), data.get('REC', 0), pass_att, final_year, "final_season"

    # If not found in 2025, try 2024 as fallback (injury, redshirt, etc.)
    stats = fetch_team_receiving(cfbd_school, final_year - 1)
    time.sleep(0.5)

    if stats:
        players = {}
        for s in stats:
            name = s.get("player", "")
            if name not in players:
                players[name] = {}
            try:
                players[name][s.get("statType", "")] = int(float(s.get("stat", 0)))
            except:
                players[name][s.get("statType", "")] = 0

        for api_name, data in players.items():
            if names_match(cfbd_name, api_name):
                pass_att = fetch_team_pass_attempts(cfbd_school, final_year - 1)
                time.sleep(0.3)
                return data.get('YDS', 0), data.get('REC', 0), pass_att, final_year - 1, "no_final_season_data"

    return None, None, None, None, "not_found"


def age_weight(age):
    """Age weight matching backtest methodology"""
    if pd.isna(age) or age == 'MISSING':
        return 1.0
    try:
        age = float(age)
    except:
        return 1.0

    college_age = age - 1  # Estimate college age from draft age
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


def dc_score(pick):
    """Draft capital score"""
    return 100 - 2.40 * (pick ** 0.62 - 1)


def main():
    print("=" * 100)
    print("FIX 2026 RB PRODUCTION: FINAL SEASON ONLY")
    print("Matching backtest methodology (draft_year - 1)")
    print("=" * 100)

    # Load current data to show before/after
    old_data = pd.read_csv('output/slap_2026_rb.csv')

    # Load prospects
    prospects = pd.read_csv('data/prospects_final.csv')
    rb_2026 = prospects[prospects['position'] == 'RB'].copy()
    print(f"\nLoaded {len(rb_2026)} RB prospects")

    # Get backtest normalization parameters
    backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
    valid_bt = backtest[(backtest['rec_yards'].notna()) & (backtest['team_pass_att'].notna()) & (backtest['team_pass_att'] > 0)]
    valid_bt = valid_bt.copy()
    valid_bt['raw_prod'] = valid_bt['rec_yards'] / valid_bt['team_pass_att']
    bt_min = valid_bt['raw_prod'].min()
    bt_max = valid_bt['raw_prod'].max()
    bt_avg_raw = valid_bt['raw_prod'].mean()
    print(f"Backtest normalization: min={bt_min:.4f}, max={bt_max:.4f}")

    # Process each RB - FINAL SEASON ONLY
    results = []
    changes = []

    print("\n" + "=" * 100)
    print("PROCESSING 2026 RBs (FINAL SEASON = 2025)")
    print("=" * 100)

    for idx, row in rb_2026.iterrows():
        player = row['player_name']
        school = row['school']
        pick = row['projected_pick']
        age = row.get('age', 21)
        ras = row.get('RAS', 66.5)
        if pd.isna(ras):
            ras = 66.5

        # Get FINAL season data only
        rec_yds, rec, pass_att, season_year, status = get_final_season_data(player, school, 2025)

        # Calculate production score
        if rec_yds is not None and pass_att and pass_att > 0:
            raw_prod = rec_yds / pass_att
            # Apply age weight
            raw_prod_weighted = raw_prod * age_weight(age)
            # Normalize to 0-100 using backtest parameters
            prod_score = ((raw_prod_weighted - bt_min) / (bt_max - bt_min)) * 100
            prod_score = max(0, min(100, prod_score))  # Clamp
            prod_status = status
        else:
            # Impute average
            prod_score = 30.0  # Backtest average
            prod_status = "imputed"
            rec_yds = None
            rec = None
            pass_att = None
            season_year = None

        # Calculate SLAP score
        dc = dc_score(pick)
        slap = 0.85 * dc + 0.10 * prod_score + 0.05 * ras
        delta = slap - dc

        # Check for changes from old data
        old_row = old_data[old_data['player_name'] == player]
        if len(old_row) > 0:
            old_season = old_row.iloc[0].get('best_season_year')
            old_prod = old_row.iloc[0].get('production_score')
            old_slap = old_row.iloc[0].get('slap_score')

            if pd.notna(old_season) and old_season != 2025 and season_year == 2025:
                changes.append({
                    'player': player,
                    'old_season': int(old_season),
                    'old_prod': old_prod,
                    'old_slap': old_slap,
                    'new_season': 2025,
                    'new_prod': prod_score,
                    'new_slap': slap,
                    'prod_change': prod_score - old_prod,
                    'slap_change': slap - old_slap
                })

        results.append({
            'player_name': player,
            'school': school,
            'projected_pick': pick,
            'age': age,
            'slap_score': slap,
            'dc_score': dc,
            'production_score': prod_score,
            'ras_score': ras,
            'delta_vs_dc': delta,
            'production_status': prod_status,
            'ras_status': 'imputed' if ras == 66.5 else 'observed',
            'rec_yards': rec_yds,
            'receptions': rec,
            'team_pass_att': pass_att,
            'season_year': season_year,
            'school_used': school
        })

        if rec_yds is not None:
            print(f"  {player}: {rec_yds} yds ({season_year}) -> Prod={prod_score:.1f}, Status={status}")
        else:
            print(f"  {player}: No data -> Prod=30.0 (imputed)")

    # Show before/after for changed players
    print("\n" + "=" * 100)
    print("BEFORE/AFTER: PLAYERS WHO WERE USING EARLIER SEASONS")
    print("=" * 100)

    if changes:
        print(f"\n{'Player':<25} {'Old Yr':<8} {'Old Prod':<10} {'New Yr':<8} {'New Prod':<10} {'Δ Prod':<10} {'Δ SLAP':<10}")
        print("-" * 90)
        for c in sorted(changes, key=lambda x: x['prod_change']):
            print(f"{c['player']:<25} {c['old_season']:<8} {c['old_prod']:>8.1f} {c['new_season']:<8} {c['new_prod']:>8.1f} {c['prod_change']:>+9.1f} {c['slap_change']:>+9.2f}")

        print(f"\nTotal players with methodology change: {len(changes)}")
        avg_prod_change = sum(c['prod_change'] for c in changes) / len(changes)
        avg_slap_change = sum(c['slap_change'] for c in changes) / len(changes)
        print(f"Average production change: {avg_prod_change:+.1f}")
        print(f"Average SLAP change: {avg_slap_change:+.2f}")
    else:
        print("No players changed (all were already using 2025)")

    # Create DataFrame and sort by SLAP
    df = pd.DataFrame(results)
    df = df.sort_values('slap_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    # Reorder columns
    cols = ['rank', 'player_name', 'school', 'projected_pick', 'age', 'slap_score', 'dc_score',
            'production_score', 'ras_score', 'delta_vs_dc', 'production_status', 'ras_status',
            'rec_yards', 'receptions', 'team_pass_att', 'season_year', 'school_used']
    df = df[cols]

    # Show top 25
    print("\n" + "=" * 100)
    print("2026 RB RANKINGS (FINAL SEASON METHODOLOGY)")
    print("=" * 100)

    print(f"\n{'Rank':<5} {'Player':<25} {'School':<16} {'Pick':<6} {'DC':<6} {'Prod':<6} {'SLAP':<6} {'Season':<8} {'Status':<15}")
    print("-" * 105)

    for _, row in df.head(30).iterrows():
        season = int(row['season_year']) if pd.notna(row['season_year']) else 'N/A'
        status = row['production_status']
        status_marker = "" if status == "final_season" else f"({status})"
        print(f"{int(row['rank']):<5} {row['player_name']:<25} {row['school']:<16} {int(row['projected_pick']):<6} "
              f"{row['dc_score']:>5.1f} {row['production_score']:>5.1f} {row['slap_score']:>5.1f} {season:<8} {status_marker:<15}")

    # Save
    df.to_csv('output/slap_2026_rb.csv', index=False)
    print(f"\n✓ Saved: output/slap_2026_rb.csv")

    # Summary
    final_count = len(df[df['production_status'] == 'final_season'])
    fallback_count = len(df[df['production_status'] == 'no_final_season_data'])
    imputed_count = len(df[df['production_status'] == 'imputed'])

    print(f"\nMETHODOLOGY SUMMARY:")
    print(f"  Using 2025 (final season): {final_count}")
    print(f"  Using 2024 (no 2025 data): {fallback_count}")
    print(f"  Imputed (no data found): {imputed_count}")


if __name__ == "__main__":
    main()
