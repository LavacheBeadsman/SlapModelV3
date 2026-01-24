"""
Fetch RB Receiving Stats from CFBD

For all RBs in backtest (2015-2024):
1. Fetch receiving yards for their final college season
2. Fetch team pass attempts
3. Calculate production metric: rec_yards / team_pass_att × age_weight
4. Test predictive value vs breakout age
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load API key
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# School name mappings
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
    "Northern Iowa": "Northern Iowa",
    "Texas-El Paso": "UTEP",
    "North Carolina A&T": None,  # FCS
    "Virginia St.": None,  # FCS
    "Ala-Birmingham": "UAB",
    "Coastal Carolina": "Coastal Carolina",
    "South Dakota St.": "South Dakota State",
    "New Mexico St.": "New Mexico State",
    "Boston Col.": "Boston College",
    "Boston College": "Boston College",
    "Mississippi": "Ole Miss",
    "Pitt": "Pittsburgh",
    "Pittsburgh": "Pittsburgh",
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


def main():
    print("=" * 90)
    print("FETCH RB RECEIVING STATS FROM CFBD")
    print("=" * 90)

    # Load RB backtest
    rb = pd.read_csv('data/rb_backtest_2015_2024.csv')
    print(f"\nLoaded {len(rb)} RBs from backtest")

    # Add columns for new data
    rb['rec_yards'] = None
    rb['receptions'] = None
    rb['team_pass_att'] = None
    rb['cfbd_name'] = None

    # Process each RB
    print("\n" + "-" * 60)
    print("Fetching receiving stats...")
    print("-" * 60)

    # Cache for team stats to avoid duplicate requests
    team_cache = {}  # (team, year) -> receiving stats
    pass_att_cache = {}  # (team, year) -> pass attempts

    found = 0
    missing = 0

    for idx, row in rb.iterrows():
        player = row['player_name']
        college = row['college']
        draft_year = int(row['draft_year'])
        season = draft_year - 1  # Final college season

        # Normalize school name
        cfbd_school = normalize_school(college)
        if cfbd_school is None:
            print(f"  {player}: School '{college}' not in CFBD (FCS)")
            missing += 1
            continue

        # Check cache for receiving stats
        cache_key = (cfbd_school, season)
        if cache_key not in team_cache:
            print(f"  Fetching {cfbd_school} {season}...")
            team_cache[cache_key] = fetch_player_receiving_stats(cfbd_school, season)
            time.sleep(0.3)  # Rate limit

        team_stats = team_cache[cache_key]

        # Find player's receiving stats
        yards, receptions, matched_name = find_player_receiving(player, team_stats)

        if yards is not None:
            rb.at[idx, 'rec_yards'] = yards
            rb.at[idx, 'receptions'] = receptions
            rb.at[idx, 'cfbd_name'] = matched_name

            # Get team pass attempts
            if cache_key not in pass_att_cache:
                pass_att_cache[cache_key] = fetch_team_pass_attempts(cfbd_school, season)
                time.sleep(0.3)

            rb.at[idx, 'team_pass_att'] = pass_att_cache[cache_key]
            found += 1
            print(f"    ✓ {player}: {yards} rec yds, {receptions} rec")
        else:
            missing += 1
            # Try previous season for transfers/early declares
            prev_season = season - 1
            prev_key = (cfbd_school, prev_season)
            if prev_key not in team_cache:
                team_cache[prev_key] = fetch_player_receiving_stats(cfbd_school, prev_season)
                time.sleep(0.3)

            prev_stats = team_cache[prev_key]
            yards2, receptions2, matched2 = find_player_receiving(player, prev_stats)

            if yards2 is not None:
                rb.at[idx, 'rec_yards'] = yards2
                rb.at[idx, 'receptions'] = receptions2
                rb.at[idx, 'cfbd_name'] = matched2

                if prev_key not in pass_att_cache:
                    pass_att_cache[prev_key] = fetch_team_pass_attempts(cfbd_school, prev_season)
                    time.sleep(0.3)

                rb.at[idx, 'team_pass_att'] = pass_att_cache[prev_key]
                found += 1
                missing -= 1
                print(f"    ✓ {player}: {yards2} rec yds (from {prev_season})")

    print("\n" + "-" * 60)
    print(f"RESULTS: Found receiving data for {found}/{len(rb)} RBs ({found/len(rb)*100:.1f}%)")
    print(f"Missing: {missing} RBs")
    print("-" * 60)

    # Save intermediate results
    rb.to_csv('data/rb_backtest_with_receiving.csv', index=False)
    print(f"\nSaved: data/rb_backtest_with_receiving.csv")

    # Calculate production metric for those with data
    print("\n" + "=" * 90)
    print("CALCULATING RB PRODUCTION METRIC")
    print("=" * 90)

    # Age weight function
    def age_weight(draft_age):
        """Apply age weight based on draft age (estimate college age as draft_age - 1)."""
        college_age = draft_age - 1  # Rough estimate of final college season age
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

    # Filter to RBs with valid receiving data
    valid = rb[(rb['rec_yards'].notna()) & (rb['team_pass_att'].notna()) & (rb['team_pass_att'] > 0)].copy()
    print(f"\nRBs with valid receiving data: {len(valid)}")

    if len(valid) > 0:
        # Calculate production metric
        valid['rec_per_pass_att'] = valid['rec_yards'] / valid['team_pass_att']
        valid['age_wt'] = valid['age'].apply(age_weight)
        valid['production_score'] = valid['rec_per_pass_att'] * valid['age_wt']

        # Normalize to 0-100 scale
        min_prod = valid['production_score'].min()
        max_prod = valid['production_score'].max()
        valid['production_normalized'] = ((valid['production_score'] - min_prod) / (max_prod - min_prod)) * 100

        # Show top receivers
        print("\n--- Top 20 RBs by Production Metric ---")
        top = valid.nlargest(20, 'production_normalized')
        print(f"{'Player':<25} {'Pick':>5} {'RecYds':>7} {'PassAtt':>8} {'Rec/Att':>8} {'Score':>6} {'NFL PPG':>8}")
        print("-" * 80)
        for _, row in top.iterrows():
            ppg = f"{row['best_ppg']:.1f}" if pd.notna(row['best_ppg']) else "-"
            print(f"{row['player_name']:<25} {int(row['pick']):>5} {int(row['rec_yards']):>7} "
                  f"{int(row['team_pass_att']):>8} {row['rec_per_pass_att']:.3f} "
                  f"{row['production_normalized']:>6.1f} {ppg:>8}")

        # Test predictive value
        print("\n" + "=" * 90)
        print("PREDICTIVE VALUE ANALYSIS")
        print("=" * 90)

        # Filter to those with NFL production
        nfl_valid = valid[(valid['best_ppr'].notna()) & (valid['best_ppr'] > 0)].copy()
        print(f"\nRBs with NFL production data: {len(nfl_valid)}")

        if len(nfl_valid) >= 20:
            # Correlation with NFL PPG
            r_prod, p_prod = spearmanr(nfl_valid['production_normalized'], nfl_valid['best_ppg'])
            print(f"\nProduction metric vs NFL PPG:")
            print(f"  Spearman r = {r_prod:.4f}, p = {p_prod:.4f}")

            # Compare with hit rate
            hit_rate = nfl_valid['hit24'].mean()
            print(f"  Overall hit rate: {hit_rate*100:.1f}%")

            # Hit rate by production quartile
            nfl_valid['prod_quartile'] = pd.qcut(nfl_valid['production_normalized'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            print("\n  Hit rate by production quartile:")
            for q in ['Q4', 'Q3', 'Q2', 'Q1']:
                q_df = nfl_valid[nfl_valid['prod_quartile'] == q]
                q_hit = q_df['hit24'].mean()
                print(f"    {q} (top receiving): {q_hit*100:.1f}% ({len(q_df)} players)")

        # Save enriched data
        valid.to_csv('data/rb_production_analysis.csv', index=False)
        print(f"\nSaved: data/rb_production_analysis.csv")


if __name__ == "__main__":
    main()
