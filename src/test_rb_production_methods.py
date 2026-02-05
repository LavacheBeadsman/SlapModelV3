"""
Test which season selection method works best for RB production scores.

Three methods to test:
1. Final Season - use draft_year - 1 (current method)
2. Best Season - use highest rec_yards/team_pass_att season
3. Career Average - average across all college seasons

Output: Correlation analysis showing which method best predicts NFL success
"""

import pandas as pd
import numpy as np
from scipy import stats

# Age weight function from CLAUDE.md
def get_age_weight(age):
    """Weight production by age (younger = bonus, older = penalty)"""
    if age <= 19:
        return 1.15
    elif age == 20:
        return 1.10
    elif age == 21:
        return 1.05
    elif age == 22:
        return 1.00
    elif age == 23:
        return 0.95
    else:  # 24+
        return 0.90

def calculate_production_score(rec_yards, team_pass_att, age_weight=1.0):
    """Calculate production score using the formula from CLAUDE.md"""
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return np.nan
    raw_score = (rec_yards / team_pass_att) * age_weight * 100
    scaled_score = raw_score / 1.75  # Normalize to 0-99.9 range
    return min(99.9, scaled_score)

def main():
    print("=" * 70)
    print("RB PRODUCTION METHOD COMPARISON")
    print("Testing Final Season vs Best Season vs Career Average")
    print("=" * 70)

    # Load backtest data (has NFL outcomes)
    backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')
    print(f"\nBacktest RBs: {len(backtest)} players (2015-2024)")

    # Load multi-season college receiving data
    college = pd.read_csv('data/college_receiving_2011_2023.csv')

    # Filter to RBs only
    college_rb = college[college['position'] == 'RB'].copy()
    print(f"College receiving records (RB): {len(college_rb)} seasons")

    # Clean up names for matching
    college_rb['name_lower'] = college_rb['college_name'].str.lower().str.strip()
    backtest['name_lower'] = backtest['player_name'].str.lower().str.strip()

    # Also try cfbd_name if available
    if 'cfbd_name' in backtest.columns:
        backtest['cfbd_lower'] = backtest['cfbd_name'].str.lower().str.strip()

    # Build multi-season data for each backtest player
    results = []
    matched_count = 0
    multi_season_count = 0

    for idx, row in backtest.iterrows():
        player = row['player_name']
        draft_year = row['draft_year']
        nfl_ppg = row['best_ppg']
        hit24 = row['hit24']

        # Find matching college seasons
        # Try exact name match first
        name_lower = row['name_lower']
        matches = college_rb[college_rb['name_lower'] == name_lower]

        # If no match, try cfbd_name
        if len(matches) == 0 and 'cfbd_lower' in backtest.columns and pd.notna(row.get('cfbd_name')):
            cfbd_lower = row['cfbd_lower']
            matches = college_rb[college_rb['name_lower'] == cfbd_lower]

        # Filter to seasons before draft
        if len(matches) > 0:
            matches = matches[matches['season'] < draft_year]

        if len(matches) == 0:
            # No multi-season data found - use single season from backtest
            final_rec = row['rec_yards'] if pd.notna(row['rec_yards']) else np.nan
            final_team_att = row['team_pass_att'] if pd.notna(row['team_pass_att']) else np.nan

            results.append({
                'player_name': player,
                'draft_year': draft_year,
                'nfl_ppg': nfl_ppg,
                'hit24': hit24,
                'college_seasons': 1,
                'final_rec_yards': final_rec,
                'final_team_att': final_team_att,
                'final_ratio': final_rec / final_team_att if pd.notna(final_rec) and pd.notna(final_team_att) and final_team_att > 0 else np.nan,
                'best_ratio': final_rec / final_team_att if pd.notna(final_rec) and pd.notna(final_team_att) and final_team_att > 0 else np.nan,
                'career_avg_ratio': final_rec / final_team_att if pd.notna(final_rec) and pd.notna(final_team_att) and final_team_att > 0 else np.nan,
                'multi_season_data': False
            })
            continue

        matched_count += 1

        # Calculate ratio for each season
        season_data = []
        for _, s in matches.iterrows():
            rec = s['rec_yards']
            att = s['team_pass_att']
            if pd.notna(rec) and pd.notna(att) and att > 0:
                ratio = rec / att
                season_data.append({
                    'season': s['season'],
                    'rec_yards': rec,
                    'team_pass_att': att,
                    'ratio': ratio
                })

        if len(season_data) == 0:
            # Has matches but no valid receiving data
            results.append({
                'player_name': player,
                'draft_year': draft_year,
                'nfl_ppg': nfl_ppg,
                'hit24': hit24,
                'college_seasons': 0,
                'final_rec_yards': np.nan,
                'final_team_att': np.nan,
                'final_ratio': np.nan,
                'best_ratio': np.nan,
                'career_avg_ratio': np.nan,
                'multi_season_data': False
            })
            continue

        if len(season_data) > 1:
            multi_season_count += 1

        # Sort by season (latest first)
        season_data.sort(key=lambda x: x['season'], reverse=True)

        # Method 1: Final Season (most recent before draft)
        final_season = season_data[0]
        final_ratio = final_season['ratio']

        # Method 2: Best Season (highest ratio)
        best_season = max(season_data, key=lambda x: x['ratio'])
        best_ratio = best_season['ratio']

        # Method 3: Career Average (average of all ratios)
        career_avg_ratio = np.mean([s['ratio'] for s in season_data])

        results.append({
            'player_name': player,
            'draft_year': draft_year,
            'nfl_ppg': nfl_ppg,
            'hit24': hit24,
            'college_seasons': len(season_data),
            'final_rec_yards': final_season['rec_yards'],
            'final_team_att': final_season['team_pass_att'],
            'final_ratio': final_ratio,
            'best_ratio': best_ratio,
            'career_avg_ratio': career_avg_ratio,
            'multi_season_data': len(season_data) > 1,
            'final_season_year': final_season['season'],
            'best_season_year': best_season['season'],
            'best_same_as_final': final_season['season'] == best_season['season']
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    print(f"\nMatching Results:")
    print(f"  Players matched to college data: {matched_count}")
    print(f"  Players with multi-season data: {multi_season_count}")
    print(f"  Players with single season only: {len(df) - multi_season_count}")

    # Calculate production scores for each method
    df['final_production'] = df.apply(lambda r: calculate_production_score(
        r['final_rec_yards'], r['final_team_att'], 1.0), axis=1)
    df['best_production'] = df['best_ratio'] * 100 / 1.75  # Simplified calculation
    df['career_production'] = df['career_avg_ratio'] * 100 / 1.75

    # Filter to players with valid NFL outcomes and production data
    valid = df[df['nfl_ppg'].notna() & df['final_ratio'].notna()].copy()
    print(f"\nPlayers with valid data for correlation: {len(valid)}")

    # CORRELATION ANALYSIS
    print("\n" + "=" * 70)
    print("CORRELATION WITH NFL PPG (Best Season Fantasy PPG)")
    print("=" * 70)

    methods = [
        ('final_ratio', 'Final Season'),
        ('best_ratio', 'Best Season'),
        ('career_avg_ratio', 'Career Average')
    ]

    for col, name in methods:
        r, p = stats.pearsonr(valid[col], valid['nfl_ppg'])
        print(f"\n{name}:")
        print(f"  Correlation (r): {r:.4f}")
        print(f"  P-value: {p:.6f}")

    print("\n" + "=" * 70)
    print("CORRELATION WITH HIT24 (24+ PPG in NFL)")
    print("=" * 70)

    valid_hit = valid[valid['hit24'].notna()]

    for col, name in methods:
        # Point-biserial correlation (continuous vs binary)
        r, p = stats.pointbiserialr(valid_hit['hit24'], valid_hit[col])
        print(f"\n{name}:")
        print(f"  Correlation (r): {r:.4f}")
        print(f"  P-value: {p:.6f}")

    # SUBSET ANALYSIS: Only players with multi-season data
    print("\n" + "=" * 70)
    print("SUBSET: ONLY PLAYERS WITH MULTI-SEASON DATA")
    print("=" * 70)

    multi = valid[valid['multi_season_data'] == True]
    print(f"Players with multi-season data: {len(multi)}")

    if len(multi) > 10:
        print("\nCorrelation with NFL PPG:")
        for col, name in methods:
            r, p = stats.pearsonr(multi[col], multi['nfl_ppg'])
            print(f"  {name}: r={r:.4f}, p={p:.6f}")

    # EXAMPLES WHERE METHODS DISAGREE
    print("\n" + "=" * 70)
    print("EXAMPLES WHERE METHODS DISAGREE")
    print("(Best Season ≠ Final Season)")
    print("=" * 70)

    # Find players where best season is different from final season
    disagree = df[(df['multi_season_data'] == True) &
                  (df.get('best_same_as_final', True) == False) &
                  (df['nfl_ppg'].notna())].copy()

    if len(disagree) > 0:
        # Sort by difference between methods
        disagree['ratio_diff'] = disagree['best_ratio'] - disagree['final_ratio']
        disagree = disagree.sort_values('ratio_diff', ascending=False)

        print(f"\nFound {len(disagree)} players where best ≠ final")
        print("\nTop 10 examples (largest difference):")
        print("-" * 70)

        for idx, row in disagree.head(10).iterrows():
            print(f"\n{row['player_name']} ({row['draft_year']})")
            print(f"  Final Season Ratio: {row['final_ratio']:.4f} (season {row.get('final_season_year', 'N/A')})")
            print(f"  Best Season Ratio:  {row['best_ratio']:.4f} (season {row.get('best_season_year', 'N/A')})")
            print(f"  Career Avg Ratio:   {row['career_avg_ratio']:.4f}")
            print(f"  NFL PPG:            {row['nfl_ppg']:.1f} | Hit24: {int(row['hit24'])}")

    # ADD DC TO THE ANALYSIS (Partial Correlation)
    print("\n" + "=" * 70)
    print("PARTIAL CORRELATION CONTROLLING FOR DRAFT CAPITAL")
    print("(Does production add value BEYOND draft pick?)")
    print("=" * 70)

    # Calculate DC score for each player
    valid['dc_score'] = valid.apply(lambda r: 100 - 2.40 * (r['draft_year'] - 2015 + 1)**0.62
                                    if pd.isna(backtest[backtest['player_name'] == r['player_name']]['pick'].values[0])
                                    else 100 - 2.40 * (backtest[backtest['player_name'] == r['player_name']]['pick'].values[0]**0.62 - 1),
                                    axis=1)

    # Merge pick data
    valid = valid.merge(backtest[['player_name', 'pick']], on='player_name', how='left')
    valid['dc_score'] = 100 - 2.40 * (valid['pick']**0.62 - 1)

    from scipy.stats import pearsonr

    def partial_correlation(df, x, y, control):
        """Calculate partial correlation of x and y, controlling for control variable"""
        # Residualize x on control
        slope_x, intercept_x, _, _, _ = stats.linregress(df[control], df[x])
        resid_x = df[x] - (slope_x * df[control] + intercept_x)

        # Residualize y on control
        slope_y, intercept_y, _, _, _ = stats.linregress(df[control], df[y])
        resid_y = df[y] - (slope_y * df[control] + intercept_y)

        # Correlate residuals
        r, p = pearsonr(resid_x, resid_y)
        return r, p

    print("\nCorrelation with NFL PPG, controlling for DC:")
    valid_dc = valid[valid['dc_score'].notna() & valid['pick'].notna()]

    for col, name in methods:
        r, p = partial_correlation(valid_dc, col, 'nfl_ppg', 'dc_score')
        print(f"  {name}: r={r:.4f}, p={p:.6f}")

    # RECOMMENDATION
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Calculate overall scores for each method
    scores = {}
    for col, name in methods:
        r_ppg, _ = stats.pearsonr(valid[col], valid['nfl_ppg'])
        r_hit, _ = stats.pointbiserialr(valid_hit['hit24'], valid_hit[col])
        r_partial, p_partial = partial_correlation(valid_dc, col, 'nfl_ppg', 'dc_score')

        scores[name] = {
            'ppg_corr': r_ppg,
            'hit_corr': r_hit,
            'partial_corr': r_partial,
            'partial_p': p_partial
        }

    print("\nSummary of correlations:")
    print(f"{'Method':<20} {'NFL PPG':<10} {'Hit24':<10} {'Partial':<10} {'P-value':<10}")
    print("-" * 60)
    for name, s in scores.items():
        print(f"{name:<20} {s['ppg_corr']:.4f}     {s['hit_corr']:.4f}     {s['partial_corr']:.4f}     {s['partial_p']:.6f}")

    # Save detailed results
    output_path = 'output/rb_production_method_comparison.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == '__main__':
    main()
