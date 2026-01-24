"""
Fix 2026 RB Projections - Use Production Metric Instead of Breakout Age

The evaluation showed 2026 RBs were using breakout_score (wrong metric).
Backtest proved:
- Production metric (rec_yards / team_pass_att x age_weight): p=0.004, adds value
- Breakout age: p=0.80, does NOT add value

This script:
1. Fetches 2024 receiving stats for 2026 RB prospects from CFBD
2. Calculates production score
3. Regenerates SLAP scores with correct metric
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load API key
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# School name mappings (expanded for 2026 prospects)
SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State",
    "Penn St.": "Penn State",
    "Michigan St.": "Michigan State",
    "North Dakota St.": "North Dakota State",
    "Oklahoma St.": "Oklahoma State",
    "Miss. St.": "Mississippi State",
    "Mississippi St.": "Mississippi State",
    "San Diego St.": "San Diego State",
    "Florida St.": "Florida State",
    "Fresno St.": "Fresno State",
    "Boise St.": "Boise State",
    "Iowa St.": "Iowa State",
    "Arizona St.": "Arizona State",
    "Kansas St.": "Kansas State",
    "N.C. State": "NC State",
    "N.C.": "North Carolina",
    "North Carolina St.": "NC State",
    "Appalachian St.": "Appalachian State",
    "Colorado St.": "Colorado State",
    "Oregon St.": "Oregon State",
    "Washington St.": "Washington State",
    "S. Carolina": "South Carolina",
    "Miami (FL)": "Miami",
    "Central Mich.": "Central Michigan",
    "Western Mich.": "Western Michigan",
    "Eastern Mich.": "Eastern Michigan",
    "Northern Ill.": "Northern Illinois",
    "Southern Miss.": "Southern Mississippi",
    "Southern Miss": "Southern Mississippi",
    "San Jose St.": "San Jose State",
    "La.-Monroe": "Louisiana Monroe",
    "La.-Lafayette": "Louisiana",
    "Louisiana-Lafayette": "Louisiana",
    "Louisiana Tech": "Louisiana Tech",
    "Texas-El Paso": "UTEP",
    "Ala-Birmingham": "UAB",
    "Coastal Carolina": "Coastal Carolina",
    "South Dakota St.": "South Dakota State",
    "New Mexico St.": "New Mexico State",
    "Boston Col.": "Boston College",
    "Boston College": "Boston College",
    "Mississippi": "Ole Miss",
    "Pitt": "Pittsburgh",
    "Pittsburgh": "Pittsburgh",
    "Texas Tech": "Texas Tech",
    "Wake Forest": "Wake Forest",
    "Notre Dame": "Notre Dame",
    "Georgia Tech": "Georgia Tech",
    "Virginia Tech": "Virginia Tech",
    "Texas State": "Texas State",
    "Texas A&M": "Texas A&M",
    "Stephen F. Austin": None,  # FCS
    "North Dakota State": None,  # FCS
    "Jacksonville State": "Jacksonville State",
    "UT Martin": None,  # FCS
    "Georgia State": "Georgia State",
    "New Mexico State": "New Mexico State",
}


def normalize_school(school):
    """Convert school name to CFBD format."""
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def fetch_player_receiving_stats(team, year):
    """Fetch receiving stats for all players on a team."""
    url = f"{BASE_URL}/stats/player/season"
    params = {
        "year": year,
        "category": "receiving",
        "team": team
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            stats = {}
            for stat in response.json():
                player = stat.get("player", "").lower()
                stat_type = stat.get("statType", "")
                value = stat.get("stat", 0)

                if player not in stats:
                    stats[player] = {}
                try:
                    stats[player][stat_type] = int(float(value))
                except (ValueError, TypeError):
                    stats[player][stat_type] = 0
            return stats
        elif response.status_code == 429:
            print(f"    Rate limited on {team} {year}")
            time.sleep(5)
            return None
        else:
            return {}
    except Exception as e:
        print(f"    Error fetching {team} {year}: {e}")
        return {}


def fetch_team_pass_attempts(team, year):
    """Fetch team total pass attempts from CFBD."""
    url = f"{BASE_URL}/stats/season"
    params = {
        "year": year,
        "team": team
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            for stat in data:
                if stat.get("statName") == "passAttempts":
                    return float(stat.get("statValue", 0))
            return None
        elif response.status_code == 429:
            time.sleep(5)
            return None
        else:
            return None
    except Exception as e:
        print(f"    Error fetching team stats {team} {year}: {e}")
        return None


def names_match(name1, name2):
    """Check if two names refer to the same person."""
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    if n1 == n2:
        return True

    # Remove suffixes and compare
    parts1 = n1.replace('.', ' ').replace("'", "").split()
    parts2 = n2.replace('.', ' ').replace("'", "").split()

    suffixes = {'jr', 'jr.', 'ii', 'iii', 'iv', 'sr'}
    clean1 = [p for p in parts1 if p not in suffixes]
    clean2 = [p for p in parts2 if p not in suffixes]

    if not clean1 or not clean2:
        return False

    # Compare first name (allow initial match)
    first1, first2 = clean1[0], clean2[0]
    if first1 != first2:
        if not (len(first1) <= 2 and first2.startswith(first1[0])):
            if not (len(first2) <= 2 and first1.startswith(first2[0])):
                return False

    # Compare last name
    if len(clean1) > 1 and len(clean2) > 1:
        last1, last2 = clean1[-1], clean2[-1]
        if last1 != last2:
            return False

    return True


def find_player_receiving(player_name, team_stats):
    """Find a player's receiving yards in team stats."""
    if team_stats is None or not team_stats:
        return None, None, None

    for api_name, stats in team_stats.items():
        if names_match(player_name, api_name):
            yards = stats.get("YDS", 0)
            receptions = stats.get("REC", 0)
            return yards, receptions, api_name

    return None, None, None


def age_weight(age):
    """Age weight for production metric."""
    if pd.isna(age) or str(age) == 'MISSING':
        return 1.0  # Default
    try:
        age = float(age)
    except (ValueError, TypeError):
        return 1.0

    # Draft age - 1 = final college season age
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


def dc_score(pick):
    """Calculate draft capital score from pick number."""
    if pd.isna(pick):
        return 0
    return 100 - 2.40 * (pick**0.62 - 1)


def main():
    print("=" * 90)
    print("FIX 2026 RB PROJECTIONS - USE PRODUCTION METRIC")
    print("=" * 90)

    # Load current 2026 RB projections
    rb_2026 = pd.read_csv('output/slap_2026_rb.csv')
    print(f"\nLoaded {len(rb_2026)} RB prospects for 2026")

    # Add columns for receiving data
    rb_2026['rec_yards'] = None
    rb_2026['receptions'] = None
    rb_2026['team_pass_att'] = None
    rb_2026['cfbd_name'] = None

    # Cache for team stats
    team_cache = {}
    pass_att_cache = {}

    print("\n" + "-" * 60)
    print("Fetching 2024 receiving stats from CFBD...")
    print("-" * 60)

    found = 0
    missing = 0
    fcs_skipped = 0

    for idx, row in rb_2026.iterrows():
        player = row['player_name']
        school = row['school']

        # Normalize school name
        cfbd_school = normalize_school(school)
        if cfbd_school is None:
            print(f"  {player}: School '{school}' is FCS (not in CFBD)")
            fcs_skipped += 1
            continue

        # Try 2024 season first (most recent)
        for season in [2024, 2023]:
            cache_key = (cfbd_school, season)

            if cache_key not in team_cache:
                print(f"  Fetching {cfbd_school} {season}...")
                team_cache[cache_key] = fetch_player_receiving_stats(cfbd_school, season)
                time.sleep(0.3)

            team_stats = team_cache[cache_key]
            yards, receptions, matched_name = find_player_receiving(player, team_stats)

            if yards is not None:
                rb_2026.at[idx, 'rec_yards'] = yards
                rb_2026.at[idx, 'receptions'] = receptions
                rb_2026.at[idx, 'cfbd_name'] = matched_name

                # Get team pass attempts
                if cache_key not in pass_att_cache:
                    pass_att_cache[cache_key] = fetch_team_pass_attempts(cfbd_school, season)
                    time.sleep(0.3)

                rb_2026.at[idx, 'team_pass_att'] = pass_att_cache[cache_key]
                found += 1
                print(f"    ✓ {player}: {yards} rec yds, {receptions} rec ({season})")
                break
        else:
            missing += 1
            print(f"    ✗ {player}: Not found in CFBD")

    print("\n" + "-" * 60)
    print(f"RESULTS: Found receiving data for {found}/{len(rb_2026)} RBs ({found/len(rb_2026)*100:.1f}%)")
    print(f"Missing: {missing} | FCS skipped: {fcs_skipped}")
    print("-" * 60)

    # Calculate production metric
    print("\n" + "=" * 90)
    print("CALCULATING PRODUCTION SCORES")
    print("=" * 90)

    # For players with receiving data
    has_data = (rb_2026['rec_yards'].notna()) & (rb_2026['team_pass_att'].notna()) & (rb_2026['team_pass_att'] > 0)

    if has_data.sum() > 0:
        rb_2026.loc[has_data, 'rec_per_pass_att'] = (
            rb_2026.loc[has_data, 'rec_yards'].astype(float) /
            rb_2026.loc[has_data, 'team_pass_att'].astype(float)
        )
        rb_2026.loc[has_data, 'age_wt'] = rb_2026.loc[has_data, 'age'].apply(age_weight)
        rb_2026.loc[has_data, 'production_raw'] = (
            rb_2026.loc[has_data, 'rec_per_pass_att'] *
            rb_2026.loc[has_data, 'age_wt']
        )

        # Normalize to 0-100 scale using backtest distribution
        # Load backtest to get normalization parameters
        rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
        bt_has_data = (rb_backtest['rec_yards'].notna()) & (rb_backtest['team_pass_att'].notna()) & (rb_backtest['team_pass_att'] > 0)

        rb_backtest.loc[bt_has_data, 'rec_per_pass_att'] = (
            rb_backtest.loc[bt_has_data, 'rec_yards'].astype(float) /
            rb_backtest.loc[bt_has_data, 'team_pass_att'].astype(float)
        )
        rb_backtest.loc[bt_has_data, 'age_wt'] = rb_backtest.loc[bt_has_data, 'age'].apply(age_weight)
        rb_backtest.loc[bt_has_data, 'production_raw'] = (
            rb_backtest.loc[bt_has_data, 'rec_per_pass_att'] *
            rb_backtest.loc[bt_has_data, 'age_wt']
        )

        # Use backtest min/max for normalization
        min_prod = rb_backtest.loc[bt_has_data, 'production_raw'].min()
        max_prod = rb_backtest.loc[bt_has_data, 'production_raw'].max()

        print(f"\nNormalization from backtest: min={min_prod:.4f}, max={max_prod:.4f}")

        # Normalize 2026 production scores
        rb_2026.loc[has_data, 'production_score'] = (
            (rb_2026.loc[has_data, 'production_raw'] - min_prod) / (max_prod - min_prod) * 100
        ).clip(0, 100)  # Clip to 0-100 range

        rb_2026['production_status'] = 'imputed'
        rb_2026.loc[has_data, 'production_status'] = 'observed'

    # For players without data, use average
    avg_production = 50.0  # Default to average
    rb_2026.loc[~has_data, 'production_score'] = avg_production

    # Recalculate SLAP scores with 85/10/5 weights
    print("\n" + "=" * 90)
    print("RECALCULATING SLAP SCORES")
    print("=" * 90)

    # DC score
    rb_2026['dc_score'] = rb_2026['projected_pick'].apply(dc_score)

    # RAS score (keep existing imputed values, scale to 0-100)
    # RAS is already 0-10 scale, multiply by 10
    rb_2026['ras_normalized'] = rb_2026['ras_score'] * 10 if 'ras_score' in rb_2026.columns else 66.5

    # Calculate SLAP: 85% DC + 10% Production + 5% RAS
    rb_2026['slap_score'] = (
        0.85 * rb_2026['dc_score'] +
        0.10 * rb_2026['production_score'] +
        0.05 * rb_2026['ras_normalized']
    )

    # Delta vs DC-only
    rb_2026['delta_vs_dc'] = rb_2026['slap_score'] - rb_2026['dc_score']

    # Sort by SLAP score
    rb_2026 = rb_2026.sort_values('slap_score', ascending=False).reset_index(drop=True)
    rb_2026['rank'] = range(1, len(rb_2026) + 1)

    # Show results
    print("\n--- Top 20 RBs by SLAP Score (with Production Metric) ---")
    print(f"{'Rank':<5} {'Player':<25} {'School':<15} {'Pick':>5} {'DC':>6} {'Prod':>6} {'SLAP':>6} {'Delta':>7}")
    print("-" * 85)
    for _, row in rb_2026.head(20).iterrows():
        prod = f"{row['production_score']:.1f}" if pd.notna(row['production_score']) else "imp"
        print(f"{int(row['rank']):<5} {row['player_name']:<25} {row['school']:<15} "
              f"{int(row['projected_pick']):>5} {row['dc_score']:>6.1f} {prod:>6} "
              f"{row['slap_score']:>6.1f} {row['delta_vs_dc']:>+7.1f}")

    # Save updated projections
    output_cols = [
        'rank', 'player_name', 'school', 'projected_pick', 'age',
        'slap_score', 'dc_score', 'production_score', 'ras_score',
        'delta_vs_dc', 'production_status', 'ras_status',
        'rec_yards', 'receptions', 'team_pass_att'
    ]

    # Ensure all columns exist
    for col in output_cols:
        if col not in rb_2026.columns:
            rb_2026[col] = None

    rb_2026_out = rb_2026[output_cols].copy()
    rb_2026_out.to_csv('output/slap_2026_rb.csv', index=False)
    print(f"\n✓ Saved: output/slap_2026_rb.csv")

    # Also update combined file if it exists
    try:
        combined = pd.read_csv('output/slap_2026_combined.csv')
        # Remove old RB data
        combined = combined[combined['position'] != 'RB'].copy()

        # Add new RB data
        rb_for_combined = rb_2026_out.copy()
        rb_for_combined['position'] = 'RB'

        # Merge
        combined = pd.concat([combined, rb_for_combined], ignore_index=True)
        combined = combined.sort_values('slap_score', ascending=False).reset_index(drop=True)
        combined.to_csv('output/slap_2026_combined.csv', index=False)
        print(f"✓ Saved: output/slap_2026_combined.csv")
    except FileNotFoundError:
        pass

    print("\n" + "=" * 90)
    print("DONE - 2026 RB projections now use PRODUCTION METRIC")
    print("=" * 90)

    # Summary stats
    observed = (rb_2026['production_status'] == 'observed').sum()
    imputed = (rb_2026['production_status'] == 'imputed').sum()
    print(f"\nProduction scores: {observed} observed, {imputed} imputed")


if __name__ == "__main__":
    main()
