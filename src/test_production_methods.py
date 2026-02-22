"""
Test Different Production Calculation Methods

Compare 5 methods for calculating RB production:
1. Final Season Only (no age weight)
2. Final Season + Age Weight
3. Best Season Only (no age weight)
4. Best Season + Age Weight
5. Earliest Breakout Season

Test against NFL outcomes to find the best predictor.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State", "Penn St.": "Penn State", "Michigan St.": "Michigan State",
    "Florida St.": "Florida State", "San Diego St.": "San Diego State",
    "Miss. St.": "Mississippi State", "Mississippi": "Ole Miss", "Pitt": "Pittsburgh",
    "Boise St.": "Boise State", "N.C. State": "NC State", "Miami (FL)": "Miami",
    "Northern Iowa": None, "North Carolina A&T": None, "Virginia St.": None,
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
    if len(first1) >= 3 and len(first2) >= 3 and first1[:3] == first2[:3]:
        return True
    return False


def age_weight(college_age):
    """Age weight based on college age during season"""
    if pd.isna(college_age):
        return 1.0
    try:
        college_age = float(college_age)
    except:
        return 1.0

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


def get_all_seasons_for_player(player_name, school, draft_year):
    """Get all college seasons for a player"""
    cfbd_school = normalize_school(school)
    if cfbd_school is None:
        return []

    seasons = []
    # Check years from draft_year-1 back to draft_year-5
    for year in range(draft_year - 1, draft_year - 6, -1):
        if year < 2014:  # CFBD data limitation
            continue

        stats = fetch_team_receiving(cfbd_school, year)
        time.sleep(0.3)

        if not stats:
            continue

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
            if names_match(player_name, api_name):
                pass_att = fetch_team_pass_attempts(cfbd_school, year)
                time.sleep(0.2)
                if pass_att and pass_att > 0:
                    seasons.append({
                        'year': year,
                        'rec_yards': data.get('YDS', 0),
                        'receptions': data.get('REC', 0),
                        'pass_att': pass_att,
                        'raw_prod': data.get('YDS', 0) / pass_att
                    })
                break

    return seasons


def calculate_methods(seasons, draft_year, draft_age):
    """Calculate production using all 5 methods"""
    results = {
        'method1_final_only': None,
        'method2_final_age': None,
        'method3_best_only': None,
        'method4_best_age': None,
        'method5_earliest': None,
    }

    if not seasons:
        return results

    # Sort by year (most recent first)
    seasons = sorted(seasons, key=lambda x: x['year'], reverse=True)

    # Method 1 & 2: Final Season
    final_season = seasons[0]  # Most recent
    results['method1_final_only'] = final_season['raw_prod']

    # Estimate college age during final season
    years_before_draft = draft_year - final_season['year'] - 1
    college_age_final = draft_age - years_before_draft if draft_age else 21
    results['method2_final_age'] = final_season['raw_prod'] * age_weight(college_age_final)

    # Method 3 & 4: Best Season (highest raw_prod)
    best_season = max(seasons, key=lambda x: x['raw_prod'])
    results['method3_best_only'] = best_season['raw_prod']

    years_before_draft_best = draft_year - best_season['year'] - 1
    college_age_best = draft_age - years_before_draft_best if draft_age else 21
    results['method4_best_age'] = best_season['raw_prod'] * age_weight(college_age_best)

    # Method 5: Earliest Breakout (first season with 200+ rec yards or raw_prod > 0.3)
    seasons_sorted_by_year = sorted(seasons, key=lambda x: x['year'])  # Oldest first
    for s in seasons_sorted_by_year:
        if s['rec_yards'] >= 200 or s['raw_prod'] >= 0.30:
            years_before_draft_early = draft_year - s['year'] - 1
            college_age_early = draft_age - years_before_draft_early if draft_age else 21
            results['method5_earliest'] = s['raw_prod'] * age_weight(college_age_early)
            break

    # If no breakout found, use final season
    if results['method5_earliest'] is None:
        results['method5_earliest'] = results['method2_final_age']

    return results


def main():
    print("=" * 100)
    print("TESTING PRODUCTION CALCULATION METHODS")
    print("=" * 100)

    # Load backtest data
    rb = pd.read_csv('data/rb_backtest_with_receiving.csv')
    print(f"\nLoaded {len(rb)} RBs from backtest")

    # Filter to those with NFL outcomes
    rb_valid = rb[(rb['best_ppg'].notna()) & (rb['best_ppg'] > 0)].copy()
    print(f"RBs with NFL outcome data: {len(rb_valid)}")

    # Fetch multi-season data for each player
    print("\n" + "-" * 60)
    print("FETCHING MULTI-SEASON DATA...")
    print("-" * 60)

    all_methods = []

    for idx, row in rb_valid.iterrows():
        player = row['player_name']
        school = row['college']
        draft_year = int(row['draft_year'])
        draft_age = row['age'] if pd.notna(row['age']) else 21

        print(f"  {player} ({school}, {draft_year})...", end=" ")

        seasons = get_all_seasons_for_player(player, school, draft_year)

        if seasons:
            methods = calculate_methods(seasons, draft_year, draft_age)
            methods['player_name'] = player
            methods['draft_year'] = draft_year
            methods['pick'] = row['pick']
            methods['best_ppg'] = row['best_ppg']
            methods['hit24'] = row['hit24']
            methods['num_seasons'] = len(seasons)

            # Track which season each method used
            seasons_sorted = sorted(seasons, key=lambda x: x['year'], reverse=True)
            methods['final_year'] = seasons_sorted[0]['year']
            methods['final_yards'] = seasons_sorted[0]['rec_yards']

            best = max(seasons, key=lambda x: x['raw_prod'])
            methods['best_year'] = best['year']
            methods['best_yards'] = best['rec_yards']

            all_methods.append(methods)
            print(f"✓ ({len(seasons)} seasons)")
        else:
            print("✗ (no data)")

    # Convert to DataFrame
    df = pd.DataFrame(all_methods)
    print(f"\nRBs with multi-season data: {len(df)}")

    # Save intermediate data
    df.to_csv('data/rb_methods_comparison.csv', index=False)

    # Calculate correlations
    print("\n" + "=" * 100)
    print("METHOD COMPARISON RESULTS")
    print("=" * 100)

    methods = ['method1_final_only', 'method2_final_age', 'method3_best_only',
               'method4_best_age', 'method5_earliest']
    method_names = ['Final Only', 'Final + Age', 'Best Only', 'Best + Age', 'Earliest Breakout']

    results = []

    for method, name in zip(methods, method_names):
        valid = df[df[method].notna()].copy()
        if len(valid) < 20:
            continue

        # Correlation with NFL PPG
        r_spearman, p_spearman = spearmanr(valid[method], valid['best_ppg'])
        r_pearson, p_pearson = pearsonr(valid[method], valid['best_ppg'])

        # R-squared
        X = valid[[method]].values
        y = valid['best_ppg'].values
        reg = LinearRegression().fit(X, y)
        r2 = r2_score(y, reg.predict(X))

        # Test if adds value beyond DC
        valid['dc_score'] = 100 - 2.40 * (valid['pick'] ** 0.62 - 1)
        X_dc = valid[['dc_score']].values
        reg_dc = LinearRegression().fit(X_dc, y)
        r2_dc_only = r2_score(y, reg_dc.predict(X_dc))

        X_both = valid[['dc_score', method]].values
        reg_both = LinearRegression().fit(X_both, y)
        r2_both = r2_score(y, reg_both.predict(X_both))

        adds_value = r2_both > r2_dc_only + 0.005  # At least 0.5% improvement

        results.append({
            'method': name,
            'n': len(valid),
            'r_spearman': r_spearman,
            'p_spearman': p_spearman,
            'r_pearson': r_pearson,
            'r2': r2,
            'r2_dc_only': r2_dc_only,
            'r2_combined': r2_both,
            'r2_improvement': r2_both - r2_dc_only,
            'adds_value': adds_value
        })

    # Display results table
    print(f"\n{'Method':<20} {'N':<5} {'r (Spearman)':<14} {'p-value':<10} {'R²':<8} {'R² w/DC':<10} {'Δ R²':<8} {'Adds Value?':<12}")
    print("-" * 100)

    for r in results:
        adds = "YES ✓" if r['adds_value'] else "no"
        print(f"{r['method']:<20} {r['n']:<5} {r['r_spearman']:>+.4f}       {r['p_spearman']:<10.4f} {r['r2']:<8.4f} {r['r2_combined']:<10.4f} {r['r2_improvement']:>+.4f}  {adds:<12}")

    # Find winner
    best_method = max(results, key=lambda x: x['r_spearman'])
    print(f"\n✓ WINNER (highest correlation): {best_method['method']} (r = {best_method['r_spearman']:.4f})")

    # Out-of-sample test
    print("\n" + "=" * 100)
    print("OUT-OF-SAMPLE TEST: Train 2015-2020, Test 2021-2024")
    print("=" * 100)

    train = df[df['draft_year'] <= 2020].copy()
    test = df[df['draft_year'] >= 2021].copy()

    print(f"\nTrain set: {len(train)} RBs (2015-2020)")
    print(f"Test set: {len(test)} RBs (2021-2024)")

    oos_results = []

    for method, name in zip(methods, method_names):
        train_valid = train[train[method].notna()].copy()
        test_valid = test[test[method].notna()].copy()

        if len(train_valid) < 20 or len(test_valid) < 10:
            continue

        # Train model
        train_valid['dc_score'] = 100 - 2.40 * (train_valid['pick'] ** 0.62 - 1)
        X_train = train_valid[['dc_score', method]].values
        y_train = train_valid['best_ppg'].values
        reg = LinearRegression().fit(X_train, y_train)

        # Test on held-out data
        test_valid['dc_score'] = 100 - 2.40 * (test_valid['pick'] ** 0.62 - 1)
        X_test = test_valid[['dc_score', method]].values
        y_test = test_valid['best_ppg'].values
        y_pred = reg.predict(X_test)

        # Calculate test correlation
        r_test, p_test = spearmanr(y_pred, y_test)

        # Hit rate analysis
        test_valid['pred_ppg'] = y_pred
        test_valid['pred_hit'] = test_valid['pred_ppg'] > test_valid['pred_ppg'].median()
        test_valid['actual_hit'] = test_valid['hit24'] == 1

        # Confusion matrix
        tp = len(test_valid[(test_valid['pred_hit']) & (test_valid['actual_hit'])])
        fp = len(test_valid[(test_valid['pred_hit']) & (~test_valid['actual_hit'])])
        tn = len(test_valid[(~test_valid['pred_hit']) & (~test_valid['actual_hit'])])
        fn = len(test_valid[(~test_valid['pred_hit']) & (test_valid['actual_hit'])])

        accuracy = (tp + tn) / len(test_valid) if len(test_valid) > 0 else 0

        oos_results.append({
            'method': name,
            'n_test': len(test_valid),
            'r_test': r_test,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        })

    print(f"\n{'Method':<20} {'N Test':<8} {'r (Test)':<12} {'Accuracy':<10} {'TP':<5} {'FP':<5} {'TN':<5} {'FN':<5}")
    print("-" * 80)

    for r in oos_results:
        print(f"{r['method']:<20} {r['n_test']:<8} {r['r_test']:>+.4f}     {r['accuracy']:.1%}      {r['tp']:<5} {r['fp']:<5} {r['tn']:<5} {r['fn']:<5}")

    best_oos = max(oos_results, key=lambda x: x['r_test'])
    print(f"\n✓ OUT-OF-SAMPLE WINNER: {best_oos['method']} (r = {best_oos['r_test']:.4f})")

    # Player-level analysis where methods disagree
    print("\n" + "=" * 100)
    print("PLAYER ANALYSIS: WHERE METHODS DISAGREE")
    print("=" * 100)

    # Find players where final vs best differs significantly
    df['final_vs_best_diff'] = abs(df['method1_final_only'] - df['method3_best_only'])
    disagreements = df.nlargest(15, 'final_vs_best_diff')

    print(f"\n{'Player':<25} {'Pick':<6} {'Final Yr':<9} {'Final Yds':<10} {'Best Yr':<8} {'Best Yds':<10} {'NFL PPG':<8} {'Hit?':<5}")
    print("-" * 95)

    for _, row in disagreements.iterrows():
        hit = "YES" if row['hit24'] == 1 else "no"
        print(f"{row['player_name']:<25} {int(row['pick']):<6} {int(row['final_year']):<9} {int(row['final_yards']):<10} "
              f"{int(row['best_year']):<8} {int(row['best_yards']):<10} {row['best_ppg']:<8.1f} {hit:<5}")

    # Analyze which method was right when they disagreed
    print("\n--- When Final ≠ Best, Which Was Right? ---")

    # Players where best > final significantly
    df['best_better_final'] = df['method3_best_only'] > df['method1_final_only'] * 1.2
    best_better = df[df['best_better_final']]

    if len(best_better) > 0:
        hit_rate_best_better = best_better['hit24'].mean()
        avg_ppg_best_better = best_better['best_ppg'].mean()
        print(f"\nPlayers where Best >> Final ({len(best_better)} players):")
        print(f"  Hit rate: {hit_rate_best_better:.1%}")
        print(f"  Avg NFL PPG: {avg_ppg_best_better:.1f}")

    # Players where final >= best
    final_better = df[~df['best_better_final']]
    if len(final_better) > 0:
        hit_rate_final = final_better['hit24'].mean()
        avg_ppg_final = final_better['best_ppg'].mean()
        print(f"\nPlayers where Final >= Best ({len(final_better)} players):")
        print(f"  Hit rate: {hit_rate_final:.1%}")
        print(f"  Avg NFL PPG: {avg_ppg_final:.1f}")

    # Sleeper analysis (late picks who hit)
    print("\n" + "=" * 100)
    print("SLEEPER ANALYSIS: Late Picks (100+) Who Hit")
    print("=" * 100)

    late_picks = df[(df['pick'] >= 100) & (df['hit24'] == 1)]
    print(f"\nLate-round hits: {len(late_picks)} players")

    if len(late_picks) > 0:
        print(f"\n{'Player':<25} {'Pick':<6} {'Final Prod':<12} {'Best Prod':<12} {'NFL PPG':<8}")
        print("-" * 70)
        for _, row in late_picks.iterrows():
            print(f"{row['player_name']:<25} {int(row['pick']):<6} {row['method1_final_only']:.4f}      {row['method3_best_only']:.4f}      {row['best_ppg']:.1f}")

        # Which method ranked them higher?
        print("\nAverage production scores for sleepers:")
        print(f"  Final only: {late_picks['method1_final_only'].mean():.4f}")
        print(f"  Best only: {late_picks['method3_best_only'].mean():.4f}")

    # Final recommendation
    print("\n" + "=" * 100)
    print("FINAL RECOMMENDATION")
    print("=" * 100)

    # Compare all results
    print("\nSUMMARY OF ALL TESTS:")
    print("-" * 60)

    for r in results:
        print(f"  {r['method']}: r={r['r_spearman']:.4f}, adds_value={r['adds_value']}")

    print("\nOUT-OF-SAMPLE:")
    for r in oos_results:
        print(f"  {r['method']}: r_test={r['r_test']:.4f}")


if __name__ == "__main__":
    main()
