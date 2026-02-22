"""
Analysis 8: Comprehensive Exploratory Analysis
What do successful NFL fantasy players actually have in common as college prospects?

Part 1: Profile hits vs misses (Hit24) for WRs and RBs across every metric
Part 2: Same for Hit12, first_3yr_ppg, career_ppg
Part 3: Within-round analysis (what separates hits from busts in the SAME round?)
Part 4: Surprise me - creative exploration, interactions, nonlinear patterns
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATA LOADING - merge everything we have
# ============================================================

def load_all_data():
    """Load and merge all data sources into comprehensive WR and RB dataframes."""

    # Core backtest files
    wr_bt = pd.read_csv('data/wr_backtest_expanded_final.csv')
    rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
    outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')

    # PFF data
    wr_pff = pd.read_csv('data/wr_pff_merged.csv')
    rb_pff = pd.read_csv('data/rb_pff_merged.csv')

    # Combine data (individual metrics)
    combine = pd.read_parquet('data/nflverse/combine.parquet')

    # Declare status
    wr_eval = pd.read_csv('data/wr_eval_with_declare.csv')
    rb_eval = pd.read_csv('data/rb_eval_with_declare.csv')

    # WR college seasons (for per-game stats)
    wr_seasons = pd.read_csv('data/wr_all_seasons.csv')

    # PFF receiving summaries (raw) - load all and deduplicate
    import os
    recv_files = sorted([f for f in os.listdir('data') if 'receiving_summary' in f])
    recv_all = []
    for f in recv_files:
        df = pd.read_csv(f'data/{f}')
        recv_all.append(df)
    recv_pff = pd.concat(recv_all, ignore_index=True)
    recv_pff = recv_pff.drop_duplicates(subset=['player', 'team_name'], keep='first')

    # PFF rushing summaries (raw)
    rush_files = sorted([f for f in os.listdir('data') if 'rushing_summary' in f])
    rush_all = []
    for f in rush_files:
        df = pd.read_csv(f'data/{f}')
        rush_all.append(df)
    rush_pff = pd.concat(rush_all, ignore_index=True)
    rush_pff = rush_pff.drop_duplicates(subset=['player', 'team_name'], keep='first')

    # ---- BUILD WR MASTER ----
    wr = wr_bt.copy()

    # Add outcomes (first_3yr_ppg, career_ppg)
    wr_out = outcomes[outcomes.position == 'WR'][['player_name', 'draft_year', 'first_3yr_ppg', 'career_ppg', 'seasons_played', 'career_games_in_data', 'first_3yr_games']]
    wr = wr.merge(wr_out, on=['player_name', 'draft_year'], how='left', suffixes=('', '_out'))
    if 'seasons_played_out' in wr.columns:
        wr['seasons_played'] = wr['seasons_played'].fillna(wr['seasons_played_out'])
        wr.drop(columns=['seasons_played_out'], inplace=True, errors='ignore')

    # Add PFF metrics
    pff_cols = ['player_name', 'draft_year', 'yprr', 'grades_offense', 'grades_pass_route',
                'drop_rate', 'contested_catch_rate', 'targets', 'routes', 'receptions', 'yards']
    wr_pff_sub = wr_pff[pff_cols].copy()
    wr_pff_sub.columns = ['player_name', 'draft_year', 'pff_yprr', 'pff_off_grade', 'pff_route_grade',
                          'pff_drop_rate', 'pff_contested_catch_rate', 'pff_targets', 'pff_routes',
                          'pff_receptions', 'pff_yards']
    wr = wr.merge(wr_pff_sub, on=['player_name', 'draft_year'], how='left')

    # Add combine metrics
    combine_wr = combine[combine.pos == 'WR'][['player_name', 'season', 'ht', 'wt', 'forty',
                                                 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']].copy()
    combine_wr.rename(columns={'season': 'draft_year'}, inplace=True)
    # Parse height to inches
    def parse_ht(h):
        if pd.isna(h): return np.nan
        parts = str(h).split('-')
        if len(parts) == 2:
            try: return int(parts[0]) * 12 + int(parts[1])
            except: return np.nan
        return np.nan
    combine_wr['height_inches'] = combine_wr['ht'].apply(parse_ht)
    combine_wr.drop(columns=['ht'], inplace=True)
    wr = wr.merge(combine_wr, on=['player_name', 'draft_year'], how='left')

    # Add declare status
    decl_cols = ['player_name', 'draft_year', 'declare_status', 'early_declare', 'draft_age']
    wr_decl = wr_eval[decl_cols].copy()
    wr = wr.merge(wr_decl, on=['player_name', 'draft_year'], how='left')

    # Add WR college per-game stats from best season
    # Get final season stats for each WR
    wr_final_season = wr_seasons.sort_values('season').groupby(['player_name', 'draft_year']).last().reset_index()
    wr_final_season['games_proxy'] = 13  # approximate
    wr_final_season['rec_yards_pg'] = wr_final_season['player_rec_yards'] / wr_final_season['games_proxy']
    wr_final_season['rec_tds_season'] = wr_final_season['player_rec_tds']
    wr_final_season['receptions_season'] = wr_final_season['player_receptions']
    wr_final_season['yards_per_rec'] = wr_final_season['player_rec_yards'] / wr_final_season['player_receptions'].replace(0, np.nan)
    wr_final_season['target_share'] = wr_final_season['player_rec_yards'] / wr_final_season['team_rec_yards'].replace(0, np.nan)

    season_cols = ['player_name', 'draft_year', 'rec_yards_pg', 'rec_tds_season', 'receptions_season',
                   'yards_per_rec', 'target_share', 'player_rec_yards', 'player_rec_tds']
    wr_final_season_sub = wr_final_season[season_cols].copy()
    wr_final_season_sub.rename(columns={'player_rec_yards': 'college_rec_yards', 'player_rec_tds': 'college_rec_tds'}, inplace=True)
    wr = wr.merge(wr_final_season_sub, on=['player_name', 'draft_year'], how='left')

    # Calculate BMI and Speed Score for WRs
    wr['bmi'] = (wr['wt'] / (wr['height_inches'] ** 2)) * 703 if 'wt' in wr.columns else np.nan
    wr['bmi'] = wr.apply(lambda r: (r['wt'] / (r['height_inches'] ** 2)) * 703
                          if pd.notna(r.get('wt')) and pd.notna(r.get('height_inches')) and r.get('height_inches', 0) > 0
                          else np.nan, axis=1)
    wr['speed_score'] = wr.apply(lambda r: (r['wt'] * 200) / (r['forty'] ** 4)
                                  if pd.notna(r.get('wt')) and pd.notna(r.get('forty')) and r.get('forty', 0) > 0
                                  else np.nan, axis=1)

    # Calculate round
    if 'round' not in wr.columns:
        wr['round'] = wr['pick'].apply(lambda p: min(7, (p - 1) // 32 + 1) if pd.notna(p) else np.nan)

    # Count college seasons from wr_all_seasons
    season_counts = wr_seasons.groupby(['player_name', 'draft_year']).size().reset_index(name='college_seasons')
    wr = wr.merge(season_counts, on=['player_name', 'draft_year'], how='left')

    # DC score
    wr['dc_score'] = wr['pick'].apply(lambda p: 100 - 2.40 * (p ** 0.62 - 1) if pd.notna(p) else np.nan)


    # ---- BUILD RB MASTER ----
    rb = rb_bt.copy()

    # Add outcomes
    rb_out = outcomes[outcomes.position == 'RB'][['player_name', 'draft_year', 'first_3yr_ppg', 'career_ppg', 'seasons_played', 'career_games_in_data', 'first_3yr_games']]

    # Need to handle: rb_bt might not have draft_year for all, use pick-based matching too
    # rb_bt has: player_name, draft_year, pick, round, ...
    rb = rb.merge(rb_out, on=['player_name', 'draft_year'], how='left', suffixes=('', '_out'))
    if 'seasons_played_out' in rb.columns:
        rb['seasons_played'] = rb.get('seasons_played', pd.Series(dtype=float)).fillna(rb.get('seasons_played_out', pd.Series(dtype=float)))
        rb.drop(columns=['seasons_played_out'], inplace=True, errors='ignore')

    # Add PFF metrics
    rb_pff_cols = ['player_name', 'draft_year', 'yco_attempt', 'elusive_rating', 'grades_run',
                   'grades_offense', 'ypa']
    rb_pff_sub = rb_pff[rb_pff_cols].copy()
    rb_pff_sub.columns = ['player_name', 'draft_year', 'pff_yco', 'pff_elusive', 'pff_rush_grade',
                          'pff_off_grade', 'pff_ypa']
    rb = rb.merge(rb_pff_sub, on=['player_name', 'draft_year'], how='left')

    # Add combine metrics
    combine_rb = combine[combine.pos == 'RB'][['player_name', 'season', 'ht', 'wt', 'forty',
                                                 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']].copy()
    combine_rb.rename(columns={'season': 'draft_year'}, inplace=True)
    combine_rb['height_inches'] = combine_rb['ht'].apply(parse_ht)
    combine_rb.drop(columns=['ht'], inplace=True)
    # Rename wt to avoid conflict with existing weight columns
    combine_rb.rename(columns={'wt': 'combine_wt'}, inplace=True)
    rb = rb.merge(combine_rb, on=['player_name', 'draft_year'], how='left')

    # Add declare status
    rb_decl = rb_eval[['player_name', 'draft_year', 'declare_status', 'early_declare', 'draft_age']].copy()
    rb = rb.merge(rb_decl, on=['player_name', 'draft_year'], how='left')

    # Calculate RB derived metrics
    rb['bmi'] = rb.apply(lambda r: (r['combine_wt'] / (r['height_inches'] ** 2)) * 703
                          if pd.notna(r.get('combine_wt')) and pd.notna(r.get('height_inches')) and r.get('height_inches', 0) > 0
                          else np.nan, axis=1)
    rb['speed_score'] = rb.apply(lambda r: (r['combine_wt'] * 200) / (r['forty'] ** 4)
                                  if pd.notna(r.get('combine_wt')) and pd.notna(r.get('forty')) and r.get('forty', 0) > 0
                                  else np.nan, axis=1)

    # RB receiving per game (approximate)
    rb['rec_yards_pg'] = rb['rec_yards'] / 13  # approximate games
    rb['rec_per_game'] = rb['receptions'] / 13 if 'receptions' in rb.columns else np.nan
    rb['rec_per_game'] = rb.apply(lambda r: r['receptions'] / 13 if pd.notna(r.get('receptions')) else np.nan, axis=1)

    # RB rush stats from PFF merged
    rb['rush_yards_pg'] = rb.apply(lambda r: np.nan, axis=1)  # placeholder if not available directly

    # DC score
    rb['dc_score'] = rb['pick'].apply(lambda p: 100 - 2.40 * (p ** 0.62 - 1) if pd.notna(p) else np.nan)

    # Breakout age for RBs - check if available
    if 'breakout_age' not in rb.columns:
        rb['breakout_age'] = np.nan

    return wr, rb, recv_pff, rush_pff


def effect_size_table(df, metric, group_col, label, continuous_outcome=None):
    """
    For a binary group (0/1), show mean, median, n for each group + effect size + significance.
    For a continuous outcome, show correlation instead.
    """
    valid = df[[metric, group_col]].dropna()
    if len(valid) < 10:
        return None

    if continuous_outcome:
        # Correlation with continuous outcome
        r, p = stats.pearsonr(valid[metric], valid[group_col])
        n = len(valid)
        return {'metric': label, 'r': r, 'p_value': p, 'n': n}

    g1 = valid[valid[group_col] == 1][metric]
    g0 = valid[valid[group_col] == 0][metric]

    if len(g1) < 3 or len(g0) < 3:
        return None

    # t-test
    t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((g1.std()**2 + g0.std()**2) / 2)
    cohens_d = (g1.mean() - g0.mean()) / pooled_std if pooled_std > 0 else 0

    # Mann-Whitney U (non-parametric)
    try:
        u_stat, u_p = stats.mannwhitneyu(g1, g0, alternative='two-sided')
    except:
        u_p = np.nan

    return {
        'metric': label,
        'hit_mean': g1.mean(),
        'hit_median': g1.median(),
        'hit_n': len(g1),
        'miss_mean': g0.mean(),
        'miss_median': g0.median(),
        'miss_n': len(g0),
        'diff_mean': g1.mean() - g0.mean(),
        'cohens_d': cohens_d,
        'p_ttest': p_val,
        'p_mannwhitney': u_p,
    }


# ============================================================
# PART 1: Profile hits vs misses (Hit24)
# ============================================================

def part1_wr(wr):
    """WR Hit24 profiling across every metric."""
    print("=" * 100)
    print("PART 1: WR HIT24 PROFILE — What do WR hits have in common vs misses?")
    print("=" * 100)

    # Define all metrics to test
    metrics = [
        ('pick', 'Draft Pick'),
        ('dc_score', 'DC Score (0-100)'),
        ('breakout_age', 'Breakout Age'),
        ('peak_dominator', 'Peak Dominator %'),
        ('RAS', 'RAS (0-10)'),
        ('college_rec_yards', 'College Rec Yards (final season)'),
        ('rec_yards_pg', 'College Rec Yards/Game (final season)'),
        ('college_rec_tds', 'College Rec TDs (final season)'),
        ('receptions_season', 'College Receptions (final season)'),
        ('yards_per_rec', 'College Yards/Reception'),
        ('target_share', 'Yards Market Share (of team)'),
        ('pff_yprr', 'PFF Yards Per Route Run'),
        ('pff_off_grade', 'PFF Offensive Grade'),
        ('pff_route_grade', 'PFF Route Grade'),
        ('pff_drop_rate', 'PFF Drop Rate'),
        ('pff_contested_catch_rate', 'PFF Contested Catch Rate'),
        ('pff_targets', 'PFF Targets'),
        ('pff_routes', 'PFF Routes Run'),
        ('college_seasons', 'College Seasons Played'),
        ('draft_age', 'Draft Age'),
        ('early_declare', 'Early Declare (1=yes)'),
        ('height_inches', 'Height (inches)'),
        ('wt', 'Weight (lbs)'),
        ('bmi', 'BMI'),
        ('forty', '40-Yard Dash'),
        ('speed_score', 'Speed Score'),
        ('cone', '3-Cone Drill'),
        ('shuttle', 'Shuttle'),
        ('vertical', 'Vertical Jump'),
        ('broad_jump', 'Broad Jump'),
        ('bench', 'Bench Press'),
    ]

    print(f"\nTotal WRs with Hit24 data: {wr['hit24'].notna().sum()}")
    print(f"  Hits (hit24=1): {(wr['hit24']==1).sum()}")
    print(f"  Misses (hit24=0): {(wr['hit24']==0).sum()}")

    results = []
    for col, label in metrics:
        if col not in wr.columns:
            continue
        r = effect_size_table(wr, col, 'hit24', label)
        if r:
            results.append(r)

    if not results:
        print("No results generated.")
        return

    rdf = pd.DataFrame(results)
    rdf['abs_d'] = rdf['cohens_d'].abs()
    rdf = rdf.sort_values('abs_d', ascending=False)

    # Flag significance
    rdf['sig'] = rdf['p_ttest'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))

    print(f"\n{'Metric':<40} {'Hit Mean':>10} {'Hit Med':>10} {'Hit N':>6} {'Miss Mean':>10} {'Miss Med':>10} {'Miss N':>7} {'Diff':>10} {'Cohen d':>9} {'p-value':>10} {'Sig':>4}")
    print("-" * 147)
    for _, row in rdf.iterrows():
        print(f"{row['metric']:<40} {row['hit_mean']:>10.2f} {row['hit_median']:>10.2f} {row['hit_n']:>6.0f} {row['miss_mean']:>10.2f} {row['miss_median']:>10.2f} {row['miss_n']:>7.0f} {row['diff_mean']:>10.2f} {row['cohens_d']:>9.3f} {row['p_ttest']:>10.4f} {row['sig']:>4}")

    print(f"\n*** p<0.001  ** p<0.01  * p<0.05")
    print(f"Cohen's d interpretation: |d|>0.8 = large, |d|>0.5 = medium, |d|>0.2 = small")

    # Top separators
    print(f"\n--- TOP 10 METRICS BY EFFECT SIZE (largest separation hits vs misses) ---")
    for i, (_, row) in enumerate(rdf.head(10).iterrows()):
        print(f"  {i+1}. {row['metric']}: d={row['cohens_d']:.3f}, diff={row['diff_mean']:.2f}, p={row['p_ttest']:.4f} {row['sig']}")


def part1_rb(rb):
    """RB Hit24 profiling across every metric."""
    print("\n" + "=" * 100)
    print("PART 1: RB HIT24 PROFILE — What do RB hits have in common vs misses?")
    print("=" * 100)

    metrics = [
        ('pick', 'Draft Pick'),
        ('dc_score', 'DC Score (0-100)'),
        ('age', 'Age at Draft'),
        ('rec_yards', 'College Receiving Yards (final)'),
        ('receptions', 'College Receptions (final)'),
        ('team_pass_att', 'Team Pass Attempts'),
        ('rec_yards_pg', 'Receiving Yards/Game'),
        ('rec_per_game', 'Receptions/Game'),
        ('RAS', 'RAS (0-10)'),
        ('pff_yco', 'PFF Yards After Contact/Att'),
        ('pff_elusive', 'PFF Elusive Rating'),
        ('pff_rush_grade', 'PFF Rushing Grade'),
        ('pff_off_grade', 'PFF Offensive Grade'),
        ('pff_ypa', 'PFF Yards Per Attempt'),
        ('breakout_age', 'Breakout Age'),
        ('early_declare', 'Early Declare (1=yes)'),
        ('height_inches', 'Height (inches)'),
        ('combine_wt', 'Weight (lbs)'),
        ('bmi', 'BMI'),
        ('forty', '40-Yard Dash'),
        ('speed_score', 'Speed Score'),
        ('cone', '3-Cone Drill'),
        ('shuttle', 'Shuttle'),
        ('vertical', 'Vertical Jump'),
        ('broad_jump', 'Broad Jump'),
        ('bench', 'Bench Press'),
    ]

    print(f"\nTotal RBs with Hit24 data: {rb['hit24'].notna().sum()}")
    print(f"  Hits (hit24=1): {(rb['hit24']==1).sum()}")
    print(f"  Misses (hit24=0): {(rb['hit24']==0).sum()}")

    results = []
    for col, label in metrics:
        if col not in rb.columns:
            continue
        r = effect_size_table(rb, col, 'hit24', label)
        if r:
            results.append(r)

    rdf = pd.DataFrame(results)
    rdf['abs_d'] = rdf['cohens_d'].abs()
    rdf = rdf.sort_values('abs_d', ascending=False)
    rdf['sig'] = rdf['p_ttest'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))

    print(f"\n{'Metric':<40} {'Hit Mean':>10} {'Hit Med':>10} {'Hit N':>6} {'Miss Mean':>10} {'Miss Med':>10} {'Miss N':>7} {'Diff':>10} {'Cohen d':>9} {'p-value':>10} {'Sig':>4}")
    print("-" * 147)
    for _, row in rdf.iterrows():
        print(f"{row['metric']:<40} {row['hit_mean']:>10.2f} {row['hit_median']:>10.2f} {row['hit_n']:>6.0f} {row['miss_mean']:>10.2f} {row['miss_median']:>10.2f} {row['miss_n']:>7.0f} {row['diff_mean']:>10.2f} {row['cohens_d']:>9.3f} {row['p_ttest']:>10.4f} {row['sig']:>4}")

    print(f"\n--- TOP 10 METRICS BY EFFECT SIZE ---")
    for i, (_, row) in enumerate(rdf.head(10).iterrows()):
        print(f"  {i+1}. {row['metric']}: d={row['cohens_d']:.3f}, diff={row['diff_mean']:.2f}, p={row['p_ttest']:.4f} {row['sig']}")


# ============================================================
# PART 2: Multiple outcomes (Hit12, first_3yr_ppg, career_ppg)
# ============================================================

def part2_binary_analysis(df, pos, outcome_col, outcome_label, metrics):
    """Run hit/miss profiling for a binary outcome."""
    valid = df[df[outcome_col].notna()].copy()
    hits = (valid[outcome_col] == 1).sum()
    misses = (valid[outcome_col] == 0).sum()

    if hits < 3 or misses < 3:
        print(f"\n  {outcome_label}: Only {hits} hits and {misses} misses — skipping.")
        return None

    print(f"\n--- {pos} {outcome_label} ---")
    print(f"  Hits: {hits}, Misses: {misses}")

    results = []
    for col, label in metrics:
        if col not in valid.columns:
            continue
        r = effect_size_table(valid, col, outcome_col, label)
        if r:
            results.append(r)

    if not results:
        return None

    rdf = pd.DataFrame(results)
    rdf['abs_d'] = rdf['cohens_d'].abs()
    rdf = rdf.sort_values('abs_d', ascending=False)
    rdf['sig'] = rdf['p_ttest'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))

    print(f"\n  {'Metric':<40} {'Hit Mean':>10} {'Miss Mean':>10} {'Diff':>10} {'Cohen d':>9} {'p-value':>10} {'N':>5} {'Sig':>4}")
    print("  " + "-" * 98)
    for _, row in rdf.iterrows():
        print(f"  {row['metric']:<40} {row['hit_mean']:>10.2f} {row['miss_mean']:>10.2f} {row['diff_mean']:>10.2f} {row['cohens_d']:>9.3f} {row['p_ttest']:>10.4f} {row['hit_n']+row['miss_n']:>5.0f} {row['sig']:>4}")

    return rdf


def part2_continuous_analysis(df, pos, outcome_col, outcome_label, metrics):
    """Correlate metrics against a continuous outcome (ppg)."""
    valid = df[df[outcome_col].notna()].copy()

    if len(valid) < 15:
        print(f"\n  {outcome_label}: Only {len(valid)} players with data — skipping.")
        return None

    print(f"\n--- {pos} {outcome_label} (n={len(valid)}) ---")

    results = []
    for col, label in metrics:
        if col not in valid.columns:
            continue
        subset = valid[[col, outcome_col]].dropna()
        if len(subset) < 10:
            continue
        r, p = stats.pearsonr(subset[col], subset[outcome_col])
        rho, rho_p = stats.spearmanr(subset[col], subset[outcome_col])
        results.append({
            'metric': label,
            'pearson_r': r,
            'p_pearson': p,
            'spearman_rho': rho,
            'p_spearman': rho_p,
            'n': len(subset)
        })

    if not results:
        return None

    rdf = pd.DataFrame(results)
    rdf['abs_r'] = rdf['pearson_r'].abs()
    rdf = rdf.sort_values('abs_r', ascending=False)
    rdf['sig'] = rdf['p_pearson'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))

    print(f"\n  {'Metric':<40} {'Pearson r':>10} {'p-value':>10} {'Spearman':>10} {'p-value':>10} {'N':>5} {'Sig':>4}")
    print("  " + "-" * 93)
    for _, row in rdf.iterrows():
        print(f"  {row['metric']:<40} {row['pearson_r']:>10.4f} {row['p_pearson']:>10.4f} {row['spearman_rho']:>10.4f} {row['p_spearman']:>10.4f} {row['n']:>5.0f} {row['sig']:>4}")

    return rdf


def part2(wr, rb):
    """Part 2: Multiple outcomes analysis."""
    print("\n" + "=" * 100)
    print("PART 2: MULTIPLE OUTCOMES — Hit12, first_3yr_ppg, career_ppg")
    print("=" * 100)

    wr_metrics = [
        ('pick', 'Draft Pick'),
        ('dc_score', 'DC Score'),
        ('breakout_age', 'Breakout Age'),
        ('peak_dominator', 'Peak Dominator %'),
        ('RAS', 'RAS'),
        ('college_rec_yards', 'College Rec Yards'),
        ('rec_yards_pg', 'Rec Yards/Game'),
        ('college_rec_tds', 'College Rec TDs'),
        ('yards_per_rec', 'Yards/Reception'),
        ('target_share', 'Yards Market Share'),
        ('pff_yprr', 'PFF YPRR'),
        ('pff_off_grade', 'PFF Off Grade'),
        ('pff_route_grade', 'PFF Route Grade'),
        ('pff_drop_rate', 'PFF Drop Rate'),
        ('pff_contested_catch_rate', 'PFF Contested Catch'),
        ('college_seasons', 'College Seasons'),
        ('draft_age', 'Draft Age'),
        ('early_declare', 'Early Declare'),
        ('height_inches', 'Height'),
        ('wt', 'Weight'),
        ('bmi', 'BMI'),
        ('forty', '40 Time'),
        ('speed_score', 'Speed Score'),
        ('cone', '3-Cone'),
        ('shuttle', 'Shuttle'),
        ('vertical', 'Vertical'),
        ('broad_jump', 'Broad Jump'),
    ]

    rb_metrics = [
        ('pick', 'Draft Pick'),
        ('dc_score', 'DC Score'),
        ('age', 'Age'),
        ('rec_yards', 'College Rec Yards'),
        ('receptions', 'College Receptions'),
        ('rec_yards_pg', 'Rec Yards/Game'),
        ('rec_per_game', 'Receptions/Game'),
        ('RAS', 'RAS'),
        ('pff_yco', 'PFF YAC/Att'),
        ('pff_elusive', 'PFF Elusive Rating'),
        ('pff_rush_grade', 'PFF Rush Grade'),
        ('pff_off_grade', 'PFF Off Grade'),
        ('pff_ypa', 'PFF YPA'),
        ('early_declare', 'Early Declare'),
        ('height_inches', 'Height'),
        ('combine_wt', 'Weight'),
        ('bmi', 'BMI'),
        ('forty', '40 Time'),
        ('speed_score', 'Speed Score'),
        ('cone', '3-Cone'),
        ('shuttle', 'Shuttle'),
        ('vertical', 'Vertical'),
        ('broad_jump', 'Broad Jump'),
    ]

    # --- WR OUTCOMES ---
    print("\n" + "=" * 80)
    print("WR OUTCOMES")
    print("=" * 80)

    # Hit12 (binary)
    part2_binary_analysis(wr, 'WR', 'hit12', 'Hit12 (top-12 fantasy season)', wr_metrics)

    # Hit24-not-Hit12 vs Hit12 — what separates starters from elites?
    wr_with_hit = wr[(wr['hit24'] == 1)].copy()
    wr_with_hit['elite'] = (wr_with_hit['hit12'] == 1).astype(int)
    print(f"\n--- WR: Among Hit24 players, what separates Hit12 (elite) from Hit24-only (starters)? ---")
    n_elite = (wr_with_hit['elite'] == 1).sum()
    n_starter = (wr_with_hit['elite'] == 0).sum()
    print(f"  Elite (Hit12): {n_elite}, Starters (Hit24 only): {n_starter}")
    if n_elite >= 3 and n_starter >= 3:
        results = []
        for col, label in wr_metrics:
            if col not in wr_with_hit.columns:
                continue
            r = effect_size_table(wr_with_hit, col, 'elite', label)
            if r:
                results.append(r)
        if results:
            rdf = pd.DataFrame(results)
            rdf['abs_d'] = rdf['cohens_d'].abs()
            rdf = rdf.sort_values('abs_d', ascending=False)
            rdf['sig'] = rdf['p_ttest'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))
            print(f"\n  {'Metric':<40} {'Elite Mean':>10} {'Start Mean':>10} {'Diff':>10} {'Cohen d':>9} {'p-value':>10} {'Sig':>4}")
            print("  " + "-" * 93)
            for _, row in rdf.iterrows():
                print(f"  {row['metric']:<40} {row['hit_mean']:>10.2f} {row['miss_mean']:>10.2f} {row['diff_mean']:>10.2f} {row['cohens_d']:>9.3f} {row['p_ttest']:>10.4f} {row['sig']:>4}")

    # Continuous outcomes
    part2_continuous_analysis(wr, 'WR', 'first_3yr_ppg', 'First 3-Year PPG', wr_metrics)
    part2_continuous_analysis(wr, 'WR', 'career_ppg', 'Career PPG', wr_metrics)

    # --- RB OUTCOMES ---
    print("\n" + "=" * 80)
    print("RB OUTCOMES")
    print("=" * 80)

    part2_binary_analysis(rb, 'RB', 'hit12', 'Hit12 (top-12 fantasy season)', rb_metrics)

    # RB elite vs starter
    rb_with_hit = rb[(rb['hit24'] == 1)].copy()
    rb_with_hit['elite'] = (rb_with_hit['hit12'] == 1).astype(int)
    print(f"\n--- RB: Among Hit24 players, what separates Hit12 (elite) from Hit24-only (starters)? ---")
    n_elite = (rb_with_hit['elite'] == 1).sum()
    n_starter = (rb_with_hit['elite'] == 0).sum()
    print(f"  Elite (Hit12): {n_elite}, Starters (Hit24 only): {n_starter}")
    if n_elite >= 3 and n_starter >= 3:
        results = []
        for col, label in rb_metrics:
            if col not in rb_with_hit.columns:
                continue
            r = effect_size_table(rb_with_hit, col, 'elite', label)
            if r:
                results.append(r)
        if results:
            rdf = pd.DataFrame(results)
            rdf['abs_d'] = rdf['cohens_d'].abs()
            rdf = rdf.sort_values('abs_d', ascending=False)
            rdf['sig'] = rdf['p_ttest'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))
            print(f"\n  {'Metric':<40} {'Elite Mean':>10} {'Start Mean':>10} {'Diff':>10} {'Cohen d':>9} {'p-value':>10} {'Sig':>4}")
            print("  " + "-" * 93)
            for _, row in rdf.iterrows():
                print(f"  {row['metric']:<40} {row['hit_mean']:>10.2f} {row['miss_mean']:>10.2f} {row['diff_mean']:>10.2f} {row['cohens_d']:>9.3f} {row['p_ttest']:>10.4f} {row['sig']:>4}")

    part2_continuous_analysis(rb, 'RB', 'first_3yr_ppg', 'First 3-Year PPG', rb_metrics)
    part2_continuous_analysis(rb, 'RB', 'career_ppg', 'Career PPG', rb_metrics)


# ============================================================
# PART 3: Within-Round Analysis
# ============================================================

def part3_within_round(df, pos, metrics, round_groups):
    """What separates hits from busts WITHIN the same draft round?"""
    print(f"\n{'='*100}")
    print(f"PART 3: WITHIN-ROUND ANALYSIS — {pos}")
    print(f"What separates hits from busts among players drafted in the SAME round?")
    print(f"{'='*100}")

    for rnd_label, rnd_filter in round_groups:
        sub = df[rnd_filter(df)].copy()
        hits = (sub['hit24'] == 1).sum()
        misses = (sub['hit24'] == 0).sum()

        print(f"\n{'='*80}")
        print(f"{pos} {rnd_label} (n={len(sub)}, hits={hits}, misses={misses})")
        print(f"{'='*80}")

        if hits < 3 or misses < 3:
            print(f"  Too few hits or misses to analyze.")
            continue

        print(f"  Hit rate: {hits/(hits+misses)*100:.1f}%")

        results = []
        for col, label in metrics:
            if col not in sub.columns:
                continue
            r = effect_size_table(sub, col, 'hit24', label)
            if r:
                results.append(r)

        if not results:
            print("  No metrics with enough data.")
            continue

        rdf = pd.DataFrame(results)
        rdf['abs_d'] = rdf['cohens_d'].abs()
        rdf = rdf.sort_values('abs_d', ascending=False)
        rdf['sig'] = rdf['p_ttest'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))

        print(f"\n  {'Metric':<40} {'Hit Mean':>10} {'Miss Mean':>10} {'Diff':>10} {'Cohen d':>9} {'p-value':>10} {'N':>5} {'Sig':>4}")
        print("  " + "-" * 98)
        for _, row in rdf.iterrows():
            total_n = row['hit_n'] + row['miss_n']
            print(f"  {row['metric']:<40} {row['hit_mean']:>10.2f} {row['miss_mean']:>10.2f} {row['diff_mean']:>10.2f} {row['cohens_d']:>9.3f} {row['p_ttest']:>10.4f} {total_n:>5.0f} {row['sig']:>4}")

        # Show significant metrics
        sig_metrics = rdf[rdf['p_ttest'] < 0.10].copy()
        if len(sig_metrics) > 0:
            print(f"\n  METRICS WITH p < 0.10 within {rnd_label}:")
            for _, row in sig_metrics.iterrows():
                direction = "HIGHER for hits" if row['diff_mean'] > 0 else "LOWER for hits"
                print(f"    → {row['metric']}: {direction} (d={row['cohens_d']:.3f}, p={row['p_ttest']:.4f})")
        else:
            print(f"\n  No metrics reach p < 0.10 within {rnd_label}")

    # Also do within-round for Hit12 and continuous outcomes
    print(f"\n{'='*80}")
    print(f"{pos} WITHIN-ROUND: Correlations with first_3yr_ppg (controlling for round)")
    print(f"{'='*80}")

    for rnd_label, rnd_filter in round_groups:
        sub = df[rnd_filter(df) & df['first_3yr_ppg'].notna()].copy()
        if len(sub) < 10:
            continue

        print(f"\n  {rnd_label} (n={len(sub)}):")
        results = []
        for col, label in metrics:
            if col not in sub.columns:
                continue
            pair = sub[[col, 'first_3yr_ppg']].dropna()
            if len(pair) < 8:
                continue
            r, p = stats.pearsonr(pair[col], pair['first_3yr_ppg'])
            results.append({'metric': label, 'r': r, 'p': p, 'n': len(pair)})

        if results:
            rdf = pd.DataFrame(results)
            rdf['abs_r'] = rdf['r'].abs()
            rdf = rdf.sort_values('abs_r', ascending=False)
            rdf['sig'] = rdf['p'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))
            for _, row in rdf.head(15).iterrows():
                print(f"    {row['metric']:<40} r={row['r']:>7.4f}  p={row['p']:>8.4f}  n={row['n']:>3.0f} {row['sig']}")


# ============================================================
# PART 4: Surprise Me — Creative Exploration
# ============================================================

def part4(wr, rb, recv_pff, rush_pff):
    """Creative exploration: interactions, nonlinear patterns, unexpected findings."""
    print("\n" + "=" * 100)
    print("PART 4: SURPRISE ME — Creative Exploration")
    print("=" * 100)

    # --- 4A: Interaction effects ---
    print("\n" + "=" * 80)
    print("4A: INTERACTION EFFECTS")
    print("=" * 80)

    # WR: Does breakout age matter more for certain draft capital tiers?
    print("\n--- WR: Breakout Age × Draft Capital Interaction ---")
    print("Hit24 rates by DC tier and breakout age:")
    wr_valid = wr[wr['breakout_age'].notna() & wr['hit24'].notna()].copy()
    wr_valid['dc_tier'] = pd.cut(wr_valid['pick'], bins=[0, 32, 64, 128, 300], labels=['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+'])
    wr_valid['ba_tier'] = wr_valid['breakout_age'].apply(lambda x: 'Young (≤19)' if x <= 19 else ('Mid (20-21)' if x <= 21 else 'Old (22+)'))

    ct = wr_valid.groupby(['dc_tier', 'ba_tier']).agg(
        n=('hit24', 'count'),
        hits=('hit24', 'sum'),
        hit_rate=('hit24', 'mean')
    ).reset_index()

    print(f"\n  {'DC Tier':<12} {'BA Tier':<15} {'N':>5} {'Hits':>5} {'Hit Rate':>10}")
    print("  " + "-" * 50)
    for _, row in ct.iterrows():
        print(f"  {row['dc_tier']:<12} {row['ba_tier']:<15} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%}")

    # WR: Does dominator matter more at specific draft capital levels?
    print("\n--- WR: Peak Dominator × Draft Capital ---")
    wr_valid['dom_tier'] = pd.cut(wr_valid['peak_dominator'], bins=[0, 20, 30, 50], labels=['Low (<20%)', 'Mid (20-30%)', 'High (>30%)'])
    ct2 = wr_valid.groupby(['dc_tier', 'dom_tier']).agg(
        n=('hit24', 'count'),
        hits=('hit24', 'sum'),
        hit_rate=('hit24', 'mean')
    ).reset_index()

    print(f"\n  {'DC Tier':<12} {'Dom Tier':<15} {'N':>5} {'Hits':>5} {'Hit Rate':>10}")
    print("  " + "-" * 50)
    for _, row in ct2.iterrows():
        print(f"  {row['dc_tier']:<12} {row['dom_tier']:<15} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%}")

    # WR: PFF Grade × Draft Capital
    print("\n--- WR: PFF Offensive Grade × Draft Capital ---")
    wr_pff_valid = wr[wr['pff_off_grade'].notna() & wr['hit24'].notna()].copy()
    if len(wr_pff_valid) > 20:
        wr_pff_valid['dc_tier'] = pd.cut(wr_pff_valid['pick'], bins=[0, 32, 64, 128, 300], labels=['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+'])
        wr_pff_valid['grade_tier'] = pd.cut(wr_pff_valid['pff_off_grade'], bins=[0, 80, 85, 90, 100], labels=['<80', '80-85', '85-90', '90+'])
        ct3 = wr_pff_valid.groupby(['dc_tier', 'grade_tier']).agg(
            n=('hit24', 'count'),
            hits=('hit24', 'sum'),
            hit_rate=('hit24', 'mean')
        ).reset_index()

        print(f"\n  {'DC Tier':<12} {'PFF Grade':<12} {'N':>5} {'Hits':>5} {'Hit Rate':>10}")
        print("  " + "-" * 48)
        for _, row in ct3.iterrows():
            print(f"  {row['dc_tier']:<12} {row['grade_tier']:<12} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%}")

    # RB: Receiving production × Draft Capital
    print("\n--- RB: Receiving Yards × Draft Capital ---")
    rb_valid = rb[rb['rec_yards'].notna() & rb['hit24'].notna()].copy()
    rb_valid['dc_tier'] = pd.cut(rb_valid['pick'], bins=[0, 32, 64, 128, 300], labels=['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+'])
    rb_valid['rec_tier'] = pd.cut(rb_valid['rec_yards'], bins=[-1, 100, 200, 400, 2000], labels=['Low (<100)', 'Mid (100-200)', 'High (200-400)', 'Elite (400+)'])
    ct4 = rb_valid.groupby(['dc_tier', 'rec_tier']).agg(
        n=('hit24', 'count'),
        hits=('hit24', 'sum'),
        hit_rate=('hit24', 'mean')
    ).reset_index()

    print(f"\n  {'DC Tier':<12} {'Rec Yards':<18} {'N':>5} {'Hits':>5} {'Hit Rate':>10}")
    print("  " + "-" * 54)
    for _, row in ct4.iterrows():
        print(f"  {row['dc_tier']:<12} {row['rec_tier']:<18} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%}")

    # RB: PFF Elusive Rating × Draft Capital
    print("\n--- RB: PFF Elusive Rating × Draft Capital ---")
    rb_pff_valid = rb[rb['pff_elusive'].notna() & rb['hit24'].notna()].copy()
    if len(rb_pff_valid) > 20:
        rb_pff_valid['dc_tier'] = pd.cut(rb_pff_valid['pick'], bins=[0, 32, 64, 128, 300], labels=['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+'])
        rb_pff_valid['elu_tier'] = pd.cut(rb_pff_valid['pff_elusive'], bins=[0, 40, 70, 100, 200], labels=['Low (<40)', 'Mid (40-70)', 'High (70-100)', 'Elite (100+)'])
        ct5 = rb_pff_valid.groupby(['dc_tier', 'elu_tier']).agg(
            n=('hit24', 'count'),
            hits=('hit24', 'sum'),
            hit_rate=('hit24', 'mean')
        ).reset_index()

        print(f"\n  {'DC Tier':<12} {'Elusive':<18} {'N':>5} {'Hits':>5} {'Hit Rate':>10}")
        print("  " + "-" * 54)
        for _, row in ct5.iterrows():
            print(f"  {row['dc_tier']:<12} {row['elu_tier']:<18} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%}")

    # --- 4B: Nonlinear patterns ---
    print("\n" + "=" * 80)
    print("4B: NONLINEAR PATTERNS — Quintile Analysis")
    print("=" * 80)

    # WR: quintile analysis for key metrics
    print("\n--- WR: Hit24 rate by metric quintile ---")
    wr_quint_metrics = ['breakout_age', 'peak_dominator', 'pff_yprr', 'pff_off_grade',
                         'pff_drop_rate', 'college_rec_yards', 'yards_per_rec', 'forty', 'wt']

    for metric in wr_quint_metrics:
        if metric not in wr.columns:
            continue
        valid = wr[[metric, 'hit24']].dropna()
        if len(valid) < 20:
            continue

        try:
            valid['quintile'] = pd.qcut(valid[metric], 5, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4', 'Q5 (high)'], duplicates='drop')
        except:
            continue

        qt = valid.groupby('quintile').agg(
            n=('hit24', 'count'),
            hits=('hit24', 'sum'),
            hit_rate=('hit24', 'mean'),
            metric_min=(metric, 'min'),
            metric_max=(metric, 'max')
        ).reset_index()

        print(f"\n  {metric}:")
        print(f"    {'Quintile':<12} {'Range':<20} {'N':>5} {'Hits':>5} {'Hit Rate':>10}")
        print("    " + "-" * 55)
        for _, row in qt.iterrows():
            print(f"    {row['quintile']:<12} {row['metric_min']:.1f}-{row['metric_max']:.1f}{'':>10} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%}")

    # RB quintiles
    print("\n--- RB: Hit24 rate by metric quintile ---")
    rb_quint_metrics = ['rec_yards', 'pff_yco', 'pff_elusive', 'pff_rush_grade',
                        'pff_off_grade', 'pff_ypa', 'forty', 'combine_wt', 'speed_score']

    for metric in rb_quint_metrics:
        if metric not in rb.columns:
            continue
        valid = rb[[metric, 'hit24']].dropna()
        if len(valid) < 20:
            continue

        try:
            valid['quintile'] = pd.qcut(valid[metric], 5, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4', 'Q5 (high)'], duplicates='drop')
        except:
            continue

        qt = valid.groupby('quintile').agg(
            n=('hit24', 'count'),
            hits=('hit24', 'sum'),
            hit_rate=('hit24', 'mean'),
            metric_min=(metric, 'min'),
            metric_max=(metric, 'max')
        ).reset_index()

        print(f"\n  {metric}:")
        print(f"    {'Quintile':<12} {'Range':<20} {'N':>5} {'Hits':>5} {'Hit Rate':>10}")
        print("    " + "-" * 55)
        for _, row in qt.iterrows():
            print(f"    {row['quintile']:<12} {row['metric_min']:.1f}-{row['metric_max']:.1f}{'':>10} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%}")

    # --- 4C: Early declare analysis ---
    print("\n" + "=" * 80)
    print("4C: EARLY DECLARE ANALYSIS")
    print("=" * 80)

    # WR declare status
    print("\n--- WR: Hit rate by declare status ---")
    wr_decl = wr[wr['declare_status'].notna() & wr['hit24'].notna()].copy()
    if len(wr_decl) > 0:
        ct = wr_decl.groupby('declare_status').agg(
            n=('hit24', 'count'),
            hits=('hit24', 'sum'),
            hit_rate=('hit24', 'mean'),
            avg_pick=('pick', 'mean'),
            avg_breakout=('breakout_age', 'mean')
        ).reset_index()

        print(f"  {'Status':<12} {'N':>5} {'Hits':>5} {'Hit Rate':>10} {'Avg Pick':>10} {'Avg BA':>10}")
        print("  " + "-" * 55)
        for _, row in ct.iterrows():
            ba_str = f"{row['avg_breakout']:.1f}" if pd.notna(row['avg_breakout']) else "N/A"
            print(f"  {row['declare_status']:<12} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%} {row['avg_pick']:>10.1f} {ba_str:>10}")

    # WR declare within rounds
    print("\n--- WR: Hit rate by declare status WITHIN Round 1 ---")
    wr_r1_decl = wr_decl[wr_decl['round'] == 1]
    if len(wr_r1_decl) > 0:
        ct = wr_r1_decl.groupby('declare_status').agg(
            n=('hit24', 'count'),
            hits=('hit24', 'sum'),
            hit_rate=('hit24', 'mean'),
        ).reset_index()
        print(f"  {'Status':<12} {'N':>5} {'Hits':>5} {'Hit Rate':>10}")
        print("  " + "-" * 35)
        for _, row in ct.iterrows():
            print(f"  {row['declare_status']:<12} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%}")

    print("\n--- WR: Hit rate by declare status WITHIN Round 2 ---")
    wr_r2_decl = wr_decl[wr_decl['round'] == 2]
    if len(wr_r2_decl) > 0:
        ct = wr_r2_decl.groupby('declare_status').agg(
            n=('hit24', 'count'),
            hits=('hit24', 'sum'),
            hit_rate=('hit24', 'mean'),
        ).reset_index()
        print(f"  {'Status':<12} {'N':>5} {'Hits':>5} {'Hit Rate':>10}")
        print("  " + "-" * 35)
        for _, row in ct.iterrows():
            print(f"  {row['declare_status']:<12} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%}")

    # RB declare
    print("\n--- RB: Hit rate by declare status ---")
    rb_decl = rb[rb['declare_status'].notna() & rb['hit24'].notna()].copy()
    if len(rb_decl) > 0:
        ct = rb_decl.groupby('declare_status').agg(
            n=('hit24', 'count'),
            hits=('hit24', 'sum'),
            hit_rate=('hit24', 'mean'),
            avg_pick=('pick', 'mean'),
        ).reset_index()

        print(f"  {'Status':<12} {'N':>5} {'Hits':>5} {'Hit Rate':>10} {'Avg Pick':>10}")
        print("  " + "-" * 45)
        for _, row in ct.iterrows():
            print(f"  {row['declare_status']:<12} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%} {row['avg_pick']:>10.1f}")

    # --- 4D: PFF deep dive - metrics not in the model ---
    print("\n" + "=" * 80)
    print("4D: PFF METRICS DEEP DIVE — What metrics are we NOT looking at?")
    print("=" * 80)

    # WR: Additional PFF metrics from raw receiving summary
    # We already have yprr, off_grade, route_grade, drop_rate, contested_catch_rate
    # Let's look at: avg_depth_of_target, yards_after_catch, slot_rate, wide_rate, caught_percent
    print("\n--- WR: Additional PFF receiving metrics (from raw summaries) ---")
    # Need to match raw PFF to WR backtest
    wr_pff_full = pd.read_csv('data/wr_pff_merged.csv')
    # Get additional columns from raw receiving summary that aren't in the merged file
    # The merged file already has the key PFF metrics, but let's check raw for more

    # Let's try to match raw receiving summaries to our players
    wr_names = set(wr_pff_full['player'].str.lower().str.strip())

    matched_recv = recv_pff[recv_pff['player'].str.lower().str.strip().isin(wr_names) &
                            (recv_pff['position'] == 'WR')].copy()

    if len(matched_recv) > 10:
        # Merge additional PFF columns
        matched_recv['name_lower'] = matched_recv['player'].str.lower().str.strip()
        wr_pff_full['name_lower'] = wr_pff_full['player'].str.lower().str.strip()

        # Get columns not already in the merged file
        extra_cols = ['name_lower', 'avg_depth_of_target', 'yards_after_catch',
                      'yards_after_catch_per_reception', 'caught_percent',
                      'slot_rate', 'wide_rate', 'inline_rate',
                      'player_game_count', 'avoided_tackles', 'first_downs']
        extra_cols = [c for c in extra_cols if c in matched_recv.columns]

        extra = matched_recv[extra_cols].drop_duplicates(subset=['name_lower'], keep='first')
        wr_extra = wr_pff_full.merge(extra, on='name_lower', how='left')

        # Now merge with outcomes
        wr_extra2 = wr_extra[['player_name', 'draft_year', 'hit24', 'hit12'] +
                             [c for c in extra.columns if c != 'name_lower']].copy()

        extra_metrics = [
            ('avg_depth_of_target', 'PFF Avg Depth of Target'),
            ('yards_after_catch', 'PFF Total Yards After Catch'),
            ('yards_after_catch_per_reception', 'PFF YAC/Reception'),
            ('caught_percent', 'PFF Catch %'),
            ('slot_rate', 'PFF Slot Snap Rate'),
            ('wide_rate', 'PFF Wide Snap Rate'),
            ('avoided_tackles', 'PFF Avoided Tackles'),
            ('first_downs', 'PFF First Downs'),
            ('player_game_count', 'PFF Games Played'),
        ]

        results = []
        for col, label in extra_metrics:
            if col not in wr_extra2.columns:
                continue
            r = effect_size_table(wr_extra2, col, 'hit24', label)
            if r:
                results.append(r)

        if results:
            rdf = pd.DataFrame(results)
            rdf['abs_d'] = rdf['cohens_d'].abs()
            rdf = rdf.sort_values('abs_d', ascending=False)
            rdf['sig'] = rdf['p_ttest'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))

            print(f"\n  {'Metric':<40} {'Hit Mean':>10} {'Miss Mean':>10} {'Diff':>10} {'Cohen d':>9} {'p-value':>10} {'N':>5} {'Sig':>4}")
            print("  " + "-" * 98)
            for _, row in rdf.iterrows():
                total_n = row['hit_n'] + row['miss_n']
                print(f"  {row['metric']:<40} {row['hit_mean']:>10.2f} {row['miss_mean']:>10.2f} {row['diff_mean']:>10.2f} {row['cohens_d']:>9.3f} {row['p_ttest']:>10.4f} {total_n:>5.0f} {row['sig']:>4}")

    # RB: Additional PFF rushing metrics
    print("\n--- RB: Additional PFF rushing metrics (from raw summaries) ---")
    rb_pff_full = pd.read_csv('data/rb_pff_merged.csv')
    rb_names = set(rb_pff_full['player'].str.lower().str.strip())

    matched_rush = rush_pff[rush_pff['player'].str.lower().str.strip().isin(rb_names) &
                            (rush_pff['position'].isin(['HB', 'RB']))].copy()

    if len(matched_rush) > 10:
        matched_rush['name_lower'] = matched_rush['player'].str.lower().str.strip()
        rb_pff_full['name_lower'] = rb_pff_full['player'].str.lower().str.strip()

        extra_cols_rb = ['name_lower', 'breakaway_percent', 'breakaway_yards', 'breakaway_attempts',
                         'yards_after_contact', 'avoided_tackles', 'gap_attempts', 'zone_attempts',
                         'explosive', 'first_downs', 'total_touches', 'attempts',
                         'grades_pass', 'grades_pass_route', 'grades_pass_block',
                         'rec_yards', 'receptions', 'routes', 'targets']
        extra_cols_rb = [c for c in extra_cols_rb if c in matched_rush.columns]

        extra_rb = matched_rush[extra_cols_rb].drop_duplicates(subset=['name_lower'], keep='first')
        rb_extra = rb_pff_full.merge(extra_rb, on='name_lower', how='left', suffixes=('', '_pff_raw'))

        rb_extra2 = rb_extra.copy()

        extra_rb_metrics = [
            ('breakaway_percent', 'PFF Breakaway %'),
            ('breakaway_yards', 'PFF Breakaway Yards'),
            ('yards_after_contact', 'PFF Total Yards After Contact'),
            ('avoided_tackles', 'PFF Avoided Tackles'),
            ('explosive', 'PFF Explosive Runs'),
            ('first_downs', 'PFF First Downs'),
            ('total_touches', 'PFF Total Touches'),
            ('attempts', 'PFF Rush Attempts'),
            ('grades_pass', 'PFF Passing Grade'),
            ('grades_pass_route', 'PFF Pass Route Grade'),
            ('grades_pass_block', 'PFF Pass Block Grade'),
        ]

        results = []
        for col, label in extra_rb_metrics:
            actual_col = col + '_pff_raw' if col + '_pff_raw' in rb_extra2.columns else col
            if actual_col not in rb_extra2.columns:
                continue
            r = effect_size_table(rb_extra2, actual_col, 'hit24', label)
            if r:
                results.append(r)

        if results:
            rdf = pd.DataFrame(results)
            rdf['abs_d'] = rdf['cohens_d'].abs()
            rdf = rdf.sort_values('abs_d', ascending=False)
            rdf['sig'] = rdf['p_ttest'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))

            print(f"\n  {'Metric':<40} {'Hit Mean':>10} {'Miss Mean':>10} {'Diff':>10} {'Cohen d':>9} {'p-value':>10} {'N':>5} {'Sig':>4}")
            print("  " + "-" * 98)
            for _, row in rdf.iterrows():
                total_n = row['hit_n'] + row['miss_n']
                print(f"  {row['metric']:<40} {row['hit_mean']:>10.2f} {row['miss_mean']:>10.2f} {row['diff_mean']:>10.2f} {row['cohens_d']:>9.3f} {row['p_ttest']:>10.4f} {total_n:>5.0f} {row['sig']:>4}")

    # --- 4E: Conference/School effects ---
    print("\n" + "=" * 80)
    print("4E: CONFERENCE / SCHOOL TIER EFFECTS")
    print("=" * 80)

    power5 = ['Alabama', 'Ohio State', 'Ohio St.', 'Georgia', 'LSU', 'Clemson', 'Oklahoma',
              'USC', 'Texas', 'Michigan', 'Penn State', 'Penn St.', 'Oregon', 'Florida', 'Auburn',
              'Notre Dame', 'Wisconsin', 'Iowa', 'Mississippi', 'Ole Miss', 'Tennessee',
              'Texas A&M', 'Stanford', 'Washington', 'Miami', 'Miami (FL)', 'Virginia Tech',
              'Florida State', 'Florida St.', 'Michigan State', 'Michigan St.', 'NC State',
              'South Carolina', 'Nebraska', 'Arkansas', 'Kentucky', 'Pittsburgh', 'Iowa State',
              'West Virginia', 'TCU', 'Baylor', 'Oklahoma State', 'Oklahoma St.', 'Kansas State',
              'Minnesota', 'Purdue', 'Indiana', 'Maryland', 'Illinois', 'Northwestern',
              'Colorado', 'Arizona State', 'Arizona St.', 'Utah', 'UCLA', 'California', 'Cal',
              'Oregon State', 'Oregon St.', 'Washington State', 'Washington St.', 'Missouri',
              'Vanderbilt', 'Mississippi State', 'Mississippi St.', 'Louisville', 'Duke',
              'North Carolina', 'Wake Forest', 'Syracuse', 'Boston College', 'Georgia Tech',
              'Rutgers', 'Connecticut', 'Cincinnati', 'UCF', 'Houston', 'BYU',
              'SMU', 'Memphis', 'Tulane']

    # WR: Power 5 vs non-Power 5
    wr_school = wr[wr['hit24'].notna()].copy()
    wr_school['power5'] = wr_school['college'].isin(power5).astype(int)

    print("\n--- WR: Power 5 vs Non-Power 5 ---")
    ct = wr_school.groupby('power5').agg(
        n=('hit24', 'count'),
        hits=('hit24', 'sum'),
        hit_rate=('hit24', 'mean'),
        avg_pick=('pick', 'mean')
    ).reset_index()
    ct['label'] = ct['power5'].map({1: 'Power 5', 0: 'Non-Power 5'})

    print(f"  {'School':<15} {'N':>5} {'Hits':>5} {'Hit Rate':>10} {'Avg Pick':>10}")
    print("  " + "-" * 48)
    for _, row in ct.iterrows():
        print(f"  {row['label']:<15} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%} {row['avg_pick']:>10.1f}")

    # Same within rounds
    print("\n--- WR: Power 5 vs Non-Power 5 WITHIN Round 1 ---")
    wr_r1_school = wr_school[wr_school['round'] == 1]
    ct = wr_r1_school.groupby('power5').agg(
        n=('hit24', 'count'), hits=('hit24', 'sum'), hit_rate=('hit24', 'mean')
    ).reset_index()
    ct['label'] = ct['power5'].map({1: 'Power 5', 0: 'Non-Power 5'})
    print(f"  {'School':<15} {'N':>5} {'Hits':>5} {'Hit Rate':>10}")
    print("  " + "-" * 38)
    for _, row in ct.iterrows():
        print(f"  {row['label']:<15} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%}")

    # Does dominator matter MORE from weak schools?
    print("\n--- WR: Does dominator mean more from Power 5? ---")
    wr_dom = wr_school[wr_school['peak_dominator'].notna()].copy()
    for tier, label in [(1, 'Power 5'), (0, 'Non-Power 5')]:
        sub = wr_dom[wr_dom['power5'] == tier]
        if len(sub) > 10:
            r, p = stats.pearsonr(sub['peak_dominator'], sub['hit24'])
            print(f"  {label}: dominator vs hit24 r={r:.4f}, p={p:.4f}, n={len(sub)}")

    # --- 4F: Age + production combos ---
    print("\n" + "=" * 80)
    print("4F: COMBINED TRAIT PROFILES")
    print("=" * 80)

    # WR: Young + High Dominator + Early Declare
    print("\n--- WR: Multi-trait profile hit rates ---")
    wr_prof = wr[wr['hit24'].notna() & wr['breakout_age'].notna()].copy()
    wr_prof['young_breakout'] = (wr_prof['breakout_age'] <= 20).astype(int)
    wr_prof['high_dom'] = (wr_prof['peak_dominator'] >= 25).astype(int)
    wr_prof['early'] = wr_prof['early_declare'].fillna(0).astype(int)
    wr_prof['rd1_2'] = (wr_prof['round'] <= 2).astype(int)

    profiles = [
        ('Young BA + High Dom + Early + Rd 1-2', lambda d: (d['young_breakout']==1) & (d['high_dom']==1) & (d['early']==1) & (d['rd1_2']==1)),
        ('Young BA + High Dom + Rd 1-2', lambda d: (d['young_breakout']==1) & (d['high_dom']==1) & (d['rd1_2']==1)),
        ('Young BA + Rd 1-2', lambda d: (d['young_breakout']==1) & (d['rd1_2']==1)),
        ('High Dom + Rd 1-2', lambda d: (d['high_dom']==1) & (d['rd1_2']==1)),
        ('Old BA + Rd 1-2', lambda d: (d['young_breakout']==0) & (d['rd1_2']==1)),
        ('Young BA + Late Round', lambda d: (d['young_breakout']==1) & (d['rd1_2']==0)),
        ('Old BA + Late Round', lambda d: (d['young_breakout']==0) & (d['rd1_2']==0)),
    ]

    print(f"  {'Profile':<45} {'N':>5} {'Hits':>5} {'Hit Rate':>10} {'Avg PPG':>10}")
    print("  " + "-" * 78)
    for label, filt in profiles:
        sub = wr_prof[filt(wr_prof)]
        if len(sub) > 0:
            avg_ppg = sub['first_3yr_ppg'].mean() if 'first_3yr_ppg' in sub.columns else np.nan
            ppg_str = f"{avg_ppg:.1f}" if pd.notna(avg_ppg) else "N/A"
            print(f"  {label:<45} {len(sub):>5} {sub['hit24'].sum():>5.0f} {sub['hit24'].mean():>10.1%} {ppg_str:>10}")

    # RB: Receiving + Draft Capital combos
    print("\n--- RB: Multi-trait profile hit rates ---")
    rb_prof = rb[rb['hit24'].notna() & rb['rec_yards'].notna()].copy()
    rb_prof['high_rec'] = (rb_prof['rec_yards'] >= 200).astype(int)
    rb_prof['rd1_2'] = (rb_prof['round'] <= 2).astype(int)
    rb_prof['young'] = (rb_prof['age'] <= 21).astype(int) if 'age' in rb_prof.columns else 0

    rb_profiles = [
        ('High Rec + Young + Rd 1-2', lambda d: (d['high_rec']==1) & (d['young']==1) & (d['rd1_2']==1)),
        ('High Rec + Rd 1-2', lambda d: (d['high_rec']==1) & (d['rd1_2']==1)),
        ('Low Rec + Rd 1-2', lambda d: (d['high_rec']==0) & (d['rd1_2']==1)),
        ('High Rec + Late Round', lambda d: (d['high_rec']==1) & (d['rd1_2']==0)),
        ('Low Rec + Late Round', lambda d: (d['high_rec']==0) & (d['rd1_2']==0)),
    ]

    print(f"  {'Profile':<45} {'N':>5} {'Hits':>5} {'Hit Rate':>10} {'Avg PPG':>10}")
    print("  " + "-" * 78)
    for label, filt in rb_profiles:
        sub = rb_prof[filt(rb_prof)]
        if len(sub) > 0:
            avg_ppg = sub['first_3yr_ppg'].mean() if 'first_3yr_ppg' in sub.columns else np.nan
            ppg_str = f"{avg_ppg:.1f}" if pd.notna(avg_ppg) else "N/A"
            print(f"  {label:<45} {len(sub):>5} {sub['hit24'].sum():>5.0f} {sub['hit24'].mean():>10.1%} {ppg_str:>10}")

    # --- 4G: WR YPRR deep dive (strongest PFF metric from Part 1) ---
    print("\n" + "=" * 80)
    print("4G: YPRR DEEP DIVE (if it's the best PFF metric)")
    print("=" * 80)

    wr_yprr = wr[wr['pff_yprr'].notna() & wr['hit24'].notna()].copy()
    if len(wr_yprr) > 20:
        # YPRR by round
        wr_yprr['round_grp'] = pd.cut(wr_yprr['pick'], bins=[0, 32, 64, 128, 300], labels=['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+'])

        print("\n--- WR YPRR: Descriptive stats by round ---")
        for rnd in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
            sub = wr_yprr[wr_yprr['round_grp'] == rnd]
            if len(sub) > 0:
                print(f"  {rnd}: mean={sub['pff_yprr'].mean():.2f}, median={sub['pff_yprr'].median():.2f}, "
                      f"hit_rate={sub['hit24'].mean():.1%}, n={len(sub)}")

        # YPRR threshold analysis
        print("\n--- WR YPRR: Threshold analysis ---")
        for thresh in [2.0, 2.5, 3.0, 3.5]:
            high = wr_yprr[wr_yprr['pff_yprr'] >= thresh]
            low = wr_yprr[wr_yprr['pff_yprr'] < thresh]
            if len(high) > 0 and len(low) > 0:
                print(f"  YPRR >= {thresh}: hit_rate={high['hit24'].mean():.1%} (n={len(high)}) vs < {thresh}: hit_rate={low['hit24'].mean():.1%} (n={len(low)})")

        # YPRR within-round correlations
        print("\n--- WR YPRR: Within-round correlation with hit24 ---")
        for rnd in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
            sub = wr_yprr[wr_yprr['round_grp'] == rnd]
            if len(sub) > 10:
                r, p = stats.pointbiserialr(sub['hit24'], sub['pff_yprr'])
                print(f"  {rnd}: r={r:.4f}, p={p:.4f}, n={len(sub)}")

    # --- 4H: The "sleeper" profile ---
    print("\n" + "=" * 80)
    print("4H: SLEEPER PROFILE — Who hits from Rounds 3+?")
    print("=" * 80)

    # WR late-round hits
    wr_late = wr[(wr['round'] >= 3) & wr['hit24'].notna()].copy()
    wr_hits_late = wr_late[wr_late['hit24'] == 1]
    wr_miss_late = wr_late[wr_late['hit24'] == 0]

    print(f"\n--- WR Rounds 3+: {len(wr_late)} total, {len(wr_hits_late)} hits ({len(wr_hits_late)/len(wr_late)*100:.1f}%) ---")
    if len(wr_hits_late) > 0:
        print(f"\n  WR Late-Round HITS:")
        for _, p in wr_hits_late.sort_values('pick').iterrows():
            ba = f"BA={p['breakout_age']:.0f}" if pd.notna(p.get('breakout_age')) else "BA=?"
            dom = f"Dom={p['peak_dominator']:.1f}" if pd.notna(p.get('peak_dominator')) else "Dom=?"
            ras = f"RAS={p['RAS']:.1f}" if pd.notna(p.get('RAS')) else "RAS=?"
            yprr = f"YPRR={p['pff_yprr']:.2f}" if pd.notna(p.get('pff_yprr')) else ""
            print(f"    {p['player_name']:<25} Pick {p['pick']:>3.0f}  {p.get('college',''):>15}  {ba}  {dom}  {ras}  {yprr}")

    # RB late-round hits
    rb_late = rb[(rb['round'] >= 3) & rb['hit24'].notna()].copy()
    rb_hits_late = rb_late[rb_late['hit24'] == 1]

    print(f"\n--- RB Rounds 3+: {len(rb_late)} total, {len(rb_hits_late)} hits ({len(rb_hits_late)/len(rb_late)*100:.1f}%) ---")
    if len(rb_hits_late) > 0:
        print(f"\n  RB Late-Round HITS:")
        for _, p in rb_hits_late.sort_values('pick').iterrows():
            rec = f"RecYds={p['rec_yards']:.0f}" if pd.notna(p.get('rec_yards')) else "RecYds=?"
            ras = f"RAS={p['RAS']:.1f}" if pd.notna(p.get('RAS')) else "RAS=?"
            elu = f"Elu={p['pff_elusive']:.1f}" if pd.notna(p.get('pff_elusive')) else ""
            print(f"    {p['player_name']:<25} Pick {p['pick']:>3.0f}  {p.get('college',''):>15}  {rec}  {ras}  {elu}")

    # --- 4I: Year-over-year stability ---
    print("\n" + "=" * 80)
    print("4I: YEAR-OVER-YEAR STABILITY — Do patterns hold across draft classes?")
    print("=" * 80)

    # WR hit rates by year
    print("\n--- WR: Hit24 rates by draft year ---")
    wr_yearly = wr[wr['hit24'].notna()].groupby('draft_year').agg(
        n=('hit24', 'count'),
        hits=('hit24', 'sum'),
        hit_rate=('hit24', 'mean'),
        avg_pick=('pick', 'mean'),
    ).reset_index()

    print(f"  {'Year':>6} {'N':>5} {'Hits':>5} {'Hit Rate':>10} {'Avg Pick':>10}")
    print("  " + "-" * 40)
    for _, row in wr_yearly.iterrows():
        print(f"  {row['draft_year']:>6.0f} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%} {row['avg_pick']:>10.1f}")

    # RB hit rates by year
    print("\n--- RB: Hit24 rates by draft year ---")
    rb_yearly = rb[rb['hit24'].notna()].groupby('draft_year').agg(
        n=('hit24', 'count'),
        hits=('hit24', 'sum'),
        hit_rate=('hit24', 'mean'),
        avg_pick=('pick', 'mean'),
    ).reset_index()

    print(f"  {'Year':>6} {'N':>5} {'Hits':>5} {'Hit Rate':>10} {'Avg Pick':>10}")
    print("  " + "-" * 40)
    for _, row in rb_yearly.iterrows():
        print(f"  {row['draft_year']:>6.0f} {row['n']:>5.0f} {row['hits']:>5.0f} {row['hit_rate']:>10.1%} {row['avg_pick']:>10.1f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    import os
    os.chdir('/home/user/SlapModelV3')

    print("Loading and merging all data sources...")
    wr, rb, recv_pff, rush_pff = load_all_data()
    print(f"WR master: {len(wr)} rows, {len(wr.columns)} columns")
    print(f"RB master: {len(rb)} rows, {len(rb.columns)} columns")
    print(f"WR columns: {list(wr.columns)}")
    print(f"RB columns: {list(rb.columns)}")

    # PART 1
    part1_wr(wr)
    part1_rb(rb)

    # PART 2
    part2(wr, rb)

    # PART 3
    wr_metrics = [
        ('pick', 'Draft Pick (within round)'),
        ('breakout_age', 'Breakout Age'),
        ('peak_dominator', 'Peak Dominator %'),
        ('RAS', 'RAS'),
        ('college_rec_yards', 'College Rec Yards'),
        ('rec_yards_pg', 'Rec Yards/Game'),
        ('college_rec_tds', 'College Rec TDs'),
        ('yards_per_rec', 'Yards/Reception'),
        ('target_share', 'Yards Market Share'),
        ('pff_yprr', 'PFF YPRR'),
        ('pff_off_grade', 'PFF Off Grade'),
        ('pff_route_grade', 'PFF Route Grade'),
        ('pff_drop_rate', 'PFF Drop Rate'),
        ('pff_contested_catch_rate', 'PFF Contested Catch'),
        ('college_seasons', 'College Seasons'),
        ('draft_age', 'Draft Age'),
        ('early_declare', 'Early Declare'),
        ('height_inches', 'Height'),
        ('wt', 'Weight'),
        ('forty', '40 Time'),
        ('speed_score', 'Speed Score'),
        ('vertical', 'Vertical'),
        ('broad_jump', 'Broad Jump'),
    ]

    wr_round_groups = [
        ('Round 1', lambda d: d['round'] == 1),
        ('Round 2', lambda d: d['round'] == 2),
        ('Rounds 3-4', lambda d: d['round'].isin([3, 4])),
        ('Rounds 5-7', lambda d: d['round'].isin([5, 6, 7])),
    ]

    rb_metrics = [
        ('pick', 'Draft Pick (within round)'),
        ('age', 'Age'),
        ('rec_yards', 'College Rec Yards'),
        ('receptions', 'College Receptions'),
        ('rec_yards_pg', 'Rec Yards/Game'),
        ('RAS', 'RAS'),
        ('pff_yco', 'PFF YAC/Att'),
        ('pff_elusive', 'PFF Elusive Rating'),
        ('pff_rush_grade', 'PFF Rush Grade'),
        ('pff_off_grade', 'PFF Off Grade'),
        ('pff_ypa', 'PFF YPA'),
        ('early_declare', 'Early Declare'),
        ('height_inches', 'Height'),
        ('combine_wt', 'Weight'),
        ('forty', '40 Time'),
        ('speed_score', 'Speed Score'),
        ('vertical', 'Vertical'),
        ('broad_jump', 'Broad Jump'),
    ]

    rb_round_groups = [
        ('Round 1', lambda d: d['round'] == 1),
        ('Round 2', lambda d: d['round'] == 2),
        ('Rounds 3-4', lambda d: d['round'].isin([3, 4])),
        ('Rounds 5-7', lambda d: d['round'].isin([5, 6, 7])),
    ]

    part3_within_round(wr, 'WR', wr_metrics, wr_round_groups)
    part3_within_round(rb, 'RB', rb_metrics, rb_round_groups)

    # PART 4
    part4(wr, rb, recv_pff, rush_pff)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
