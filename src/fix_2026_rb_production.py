"""
Fix 2026 RB Projections - CORRECTED VERSION

Matches backtest calculation exactly:
- SLAP = 0.85*DC + 0.10*Production + 0.05*RAS
- All components normalized to 0-100 scale
- Production = (rec_yards / team_pass_att) × age_weight, then normalized
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

# =============================================================================
# CONFIGURATION - MUST MATCH BACKTEST
# =============================================================================
WEIGHT_DC = 0.85
WEIGHT_PRODUCTION = 0.10
WEIGHT_RAS = 0.05

# School name mappings
SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State", "Penn St.": "Penn State", "Michigan St.": "Michigan State",
    "North Dakota St.": "North Dakota State", "Oklahoma St.": "Oklahoma State",
    "Miss. St.": "Mississippi State", "Mississippi St.": "Mississippi State",
    "San Diego St.": "San Diego State", "Florida St.": "Florida State",
    "Fresno St.": "Fresno State", "Boise St.": "Boise State", "Iowa St.": "Iowa State",
    "Arizona St.": "Arizona State", "Kansas St.": "Kansas State", "N.C. State": "NC State",
    "Appalachian St.": "Appalachian State", "Colorado St.": "Colorado State",
    "Oregon St.": "Oregon State", "Washington St.": "Washington State",
    "Miami (FL)": "Miami", "Southern Miss.": "Southern Mississippi",
    "New Mexico St.": "New Mexico State", "Boston Col.": "Boston College",
    "Mississippi": "Ole Miss", "Pitt": "Pittsburgh",
    "Stephen F. Austin": None, "North Dakota State": None, "UT Martin": None,  # FCS
}


def normalize_school(school):
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def fetch_player_receiving_stats(team, year):
    url = f"{BASE_URL}/stats/player/season"
    params = {"year": year, "category": "receiving", "team": team}
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
        return {}
    except Exception as e:
        return {}


def fetch_team_pass_attempts(team, year):
    url = f"{BASE_URL}/stats/season"
    params = {"year": year, "team": team}
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            for stat in response.json():
                if stat.get("statName") == "passAttempts":
                    return float(stat.get("statValue", 0))
        return None
    except:
        return None


def names_match(name1, name2):
    """
    Check if two names refer to the same person.
    Requires BOTH first name AND last name to match (at least partially).
    """
    # Common nickname mappings
    NICKNAMES = {
        'nick': 'nicholas', 'nicholas': 'nick',
        'mike': 'michael', 'michael': 'mike',
        'chris': 'christopher', 'christopher': 'chris',
        'dan': 'daniel', 'daniel': 'dan',
        'dave': 'david', 'david': 'dave',
        'rob': 'robert', 'robert': 'rob', 'bob': 'robert',
        'will': 'william', 'william': 'will', 'bill': 'william',
        'tom': 'thomas', 'thomas': 'tom',
        'jim': 'james', 'james': 'jim', 'jimmy': 'james',
        'joe': 'joseph', 'joseph': 'joe',
        'tony': 'anthony', 'anthony': 'tony',
        'alex': 'alexander', 'alexander': 'alex',
        'matt': 'matthew', 'matthew': 'matt',
        'ben': 'benjamin', 'benjamin': 'ben',
        'sam': 'samuel', 'samuel': 'sam',
        'ej': 'e j', 'cj': 'c j', 'dj': 'd j', 'aj': 'a j', 'jj': 'j j',
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

    if not clean1 or not clean2:
        return False

    # MUST have at least 2 parts (first and last name)
    if len(clean1) < 2 or len(clean2) < 2:
        return False

    # Last name MUST match exactly
    if clean1[-1] != clean2[-1]:
        return False

    # First name matching
    first1, first2 = clean1[0], clean2[0]

    # Exact match
    if first1 == first2:
        return True

    # Check nickname mappings
    if NICKNAMES.get(first1) == first2 or NICKNAMES.get(first2) == first1:
        return True

    # First 3 characters match
    if len(first1) >= 3 and len(first2) >= 3 and first1[:3] == first2[:3]:
        return True

    # One name is a prefix of the other (at least 3 chars)
    short, long = (first1, first2) if len(first1) < len(first2) else (first2, first1)
    if len(short) >= 3 and long.startswith(short):
        return True

    return False


def find_player_receiving(player_name, team_stats):
    if team_stats is None or not team_stats:
        return None, None, None

    for api_name, stats in team_stats.items():
        if names_match(player_name, api_name):
            yards = stats.get("YDS", 0)
            receptions = stats.get("REC", 0)
            return yards, receptions, api_name

    return None, None, None


# =============================================================================
# SCORING FUNCTIONS - MUST MATCH BACKTEST
# =============================================================================

def dc_score(pick):
    """DC formula: DC = 100 - 2.40 × (pick^0.62 - 1)"""
    if pd.isna(pick):
        return 0
    return max(0, min(100, 100 - 2.40 * (pick**0.62 - 1)))


def age_weight(age):
    """Age weight for RB production metric"""
    if pd.isna(age) or str(age) == 'MISSING':
        return 1.0
    try:
        age = float(age)
    except (ValueError, TypeError):
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


def normalize_ras(ras, mean_ras=5.5, std_ras=2.5):
    """Convert RAS (0-10 scale) to 0-100 score - same as backtest"""
    if pd.isna(ras):
        return None
    return 50 + (ras - mean_ras) / std_ras * 25


def main():
    print("=" * 90)
    print("FIX 2026 RB PROJECTIONS - CORRECTED VERSION")
    print("=" * 90)

    # =========================================================================
    # STEP 1: Load backtest to get normalization parameters
    # =========================================================================
    print("\n--- Loading backtest for normalization parameters ---")

    rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')

    # Calculate production for backtest
    rb_backtest['rec_yards'] = pd.to_numeric(rb_backtest['rec_yards'], errors='coerce')
    rb_backtest['team_pass_att'] = pd.to_numeric(rb_backtest['team_pass_att'], errors='coerce')
    rb_backtest['age'] = pd.to_numeric(rb_backtest['age'], errors='coerce')

    bt_has_data = (rb_backtest['rec_yards'].notna()) & (rb_backtest['team_pass_att'].notna()) & (rb_backtest['team_pass_att'] > 0)

    rb_backtest.loc[bt_has_data, 'rec_per_pass_att'] = (
        rb_backtest.loc[bt_has_data, 'rec_yards'] / rb_backtest.loc[bt_has_data, 'team_pass_att']
    )
    rb_backtest.loc[bt_has_data, 'age_wt'] = rb_backtest.loc[bt_has_data, 'age'].apply(age_weight)
    rb_backtest.loc[bt_has_data, 'production_raw'] = (
        rb_backtest.loc[bt_has_data, 'rec_per_pass_att'] * rb_backtest.loc[bt_has_data, 'age_wt']
    )

    # Get min/max for normalization
    PROD_MIN = rb_backtest.loc[bt_has_data, 'production_raw'].min()
    PROD_MAX = rb_backtest.loc[bt_has_data, 'production_raw'].max()

    # Calculate normalized production scores for backtest
    rb_backtest.loc[bt_has_data, 'production_score'] = (
        (rb_backtest.loc[bt_has_data, 'production_raw'] - PROD_MIN) / (PROD_MAX - PROD_MIN) * 100
    )

    # Calculate RAS scores for backtest
    rb_backtest['ras_score'] = rb_backtest['RAS'].apply(normalize_ras)

    # Get averages for imputation
    AVG_PRODUCTION = rb_backtest['production_score'].mean()
    AVG_RAS = rb_backtest['ras_score'].mean()

    print(f"Backtest normalization:")
    print(f"  Production raw range: {PROD_MIN:.4f} - {PROD_MAX:.4f}")
    print(f"  Avg production score: {AVG_PRODUCTION:.1f}")
    print(f"  Avg RAS score: {AVG_RAS:.1f}")

    # =========================================================================
    # STEP 2: Load 2026 RB prospects
    # =========================================================================
    print("\n--- Loading 2026 RB prospects ---")

    # Load from prospects file
    prospects = pd.read_csv('data/prospects_final.csv')
    rb_2026 = prospects[prospects['position'] == 'RB'].copy()
    print(f"Loaded {len(rb_2026)} RB prospects")

    # =========================================================================
    # STEP 3: Fetch receiving data from CFBD
    # =========================================================================
    print("\n--- Fetching 2024 receiving stats from CFBD ---")

    rb_2026['rec_yards'] = None
    rb_2026['receptions'] = None
    rb_2026['team_pass_att'] = None

    team_cache = {}
    pass_att_cache = {}

    found = 0

    for idx, row in rb_2026.iterrows():
        player = row['player_name']
        school = row['school']

        cfbd_school = normalize_school(school)
        if cfbd_school is None:
            continue

        for season in [2024, 2023]:
            cache_key = (cfbd_school, season)

            if cache_key not in team_cache:
                team_cache[cache_key] = fetch_player_receiving_stats(cfbd_school, season)
                time.sleep(0.3)

            team_stats = team_cache[cache_key]
            yards, receptions, matched_name = find_player_receiving(player, team_stats)

            if yards is not None and yards > 0:
                rb_2026.at[idx, 'rec_yards'] = yards
                rb_2026.at[idx, 'receptions'] = receptions

                if cache_key not in pass_att_cache:
                    pass_att_cache[cache_key] = fetch_team_pass_attempts(cfbd_school, season)
                    time.sleep(0.3)

                rb_2026.at[idx, 'team_pass_att'] = pass_att_cache[cache_key]
                found += 1
                print(f"  ✓ {player}: {yards} rec yds ({season})")
                break

    print(f"\nFound receiving data for {found}/{len(rb_2026)} RBs")

    # =========================================================================
    # STEP 4: Calculate production scores
    # =========================================================================
    print("\n--- Calculating production scores ---")

    # Convert to numeric
    rb_2026['rec_yards'] = pd.to_numeric(rb_2026['rec_yards'], errors='coerce')
    rb_2026['team_pass_att'] = pd.to_numeric(rb_2026['team_pass_att'], errors='coerce')

    has_data = (rb_2026['rec_yards'].notna()) & (rb_2026['team_pass_att'].notna()) & (rb_2026['team_pass_att'] > 0)

    # Calculate raw production
    rb_2026.loc[has_data, 'rec_per_pass_att'] = (
        rb_2026.loc[has_data, 'rec_yards'] / rb_2026.loc[has_data, 'team_pass_att']
    )
    rb_2026.loc[has_data, 'age_wt'] = rb_2026.loc[has_data, 'age'].apply(age_weight)
    rb_2026.loc[has_data, 'production_raw'] = (
        rb_2026.loc[has_data, 'rec_per_pass_att'] * rb_2026.loc[has_data, 'age_wt']
    )

    # Normalize using backtest min/max (clip to 0-100)
    rb_2026.loc[has_data, 'production_score'] = (
        (rb_2026.loc[has_data, 'production_raw'] - PROD_MIN) / (PROD_MAX - PROD_MIN) * 100
    ).clip(0, 100)

    # Mark status
    rb_2026['production_status'] = 'imputed'
    rb_2026.loc[has_data, 'production_status'] = 'observed'

    # Impute missing with average
    rb_2026['production_score'] = rb_2026['production_score'].fillna(AVG_PRODUCTION)

    print(f"Production scores: {has_data.sum()} observed, {(~has_data).sum()} imputed")

    # =========================================================================
    # STEP 5: Calculate SLAP scores - MATCHING BACKTEST FORMULA
    # =========================================================================
    print("\n--- Calculating SLAP scores ---")

    # DC score
    rb_2026['dc_score'] = rb_2026['projected_pick'].apply(dc_score)

    # RAS score - impute with backtest average (all 2026 prospects missing RAS)
    rb_2026['ras_score'] = AVG_RAS
    rb_2026['ras_status'] = 'imputed'

    # SLAP = 0.85*DC + 0.10*Production + 0.05*RAS
    rb_2026['slap_score'] = (
        WEIGHT_DC * rb_2026['dc_score'] +
        WEIGHT_PRODUCTION * rb_2026['production_score'] +
        WEIGHT_RAS * rb_2026['ras_score']
    )

    # Delta vs DC
    rb_2026['delta_vs_dc'] = rb_2026['slap_score'] - rb_2026['dc_score']

    # Sort by SLAP
    rb_2026 = rb_2026.sort_values('slap_score', ascending=False).reset_index(drop=True)
    rb_2026['rank'] = range(1, len(rb_2026) + 1)

    # =========================================================================
    # STEP 6: Display and save results
    # =========================================================================
    print("\n" + "=" * 90)
    print("2026 RB RANKINGS (CORRECTED)")
    print("=" * 90)

    print(f"\n{'Rank':<5} {'Player':<25} {'School':<15} {'Pick':>5} {'DC':>6} {'Prod':>5} {'RAS':>5} {'SLAP':>6} {'Delta':>6}")
    print("-" * 95)

    for _, row in rb_2026.head(25).iterrows():
        prod_str = f"{row['production_score']:.0f}" if row['production_status'] == 'observed' else f"{row['production_score']:.0f}*"
        print(f"{int(row['rank']):<5} {row['player_name']:<25} {row['school']:<15} "
              f"{int(row['projected_pick']):>5} {row['dc_score']:>6.1f} {prod_str:>5} "
              f"{row['ras_score']:>5.1f} {row['slap_score']:>6.1f} {row['delta_vs_dc']:>+6.1f}")

    print("\n* = imputed production score")

    # Verify formula
    print("\n--- Formula verification ---")
    sample = rb_2026.iloc[0]
    calc = WEIGHT_DC * sample['dc_score'] + WEIGHT_PRODUCTION * sample['production_score'] + WEIGHT_RAS * sample['ras_score']
    print(f"{sample['player_name']}: {WEIGHT_DC}×{sample['dc_score']:.1f} + {WEIGHT_PRODUCTION}×{sample['production_score']:.1f} + {WEIGHT_RAS}×{sample['ras_score']:.1f} = {calc:.1f}")

    # Save
    output_cols = [
        'rank', 'player_name', 'school', 'projected_pick', 'age',
        'slap_score', 'dc_score', 'production_score', 'ras_score',
        'delta_vs_dc', 'production_status', 'ras_status',
        'rec_yards', 'receptions', 'team_pass_att'
    ]

    for col in output_cols:
        if col not in rb_2026.columns:
            rb_2026[col] = None

    rb_2026[output_cols].to_csv('output/slap_2026_rb.csv', index=False)
    print(f"\n✓ Saved: output/slap_2026_rb.csv")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
