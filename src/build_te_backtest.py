"""
Build TE Backtest Master Dataset
=================================
Pulls ALL available data for TEs drafted 2015-2025 from every source:
  1. Draft picks (nflverse) — pick, round, team, college, age
  2. NFL outcomes — build hit24/hit12/PPG from weekly stats (same method as WR/RB)
  3. Combine data (nflverse) — height, weight, forty, bench, vertical, broad, cone, shuttle
  4. PFF college data — all 45 receiving metrics from receiving_summary files
  5. CFBD API — college receiving yards, team pass attempts, rushing yards, dominator
  6. Birthdates — from nflverse draft data and CFBD
  7. Early declare status — from draft age

Output: data/te_backtest_master.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import warnings, os, glob, json
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE")
CFBD_HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}

def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()

def cfbd_get(url, params=None):
    try:
        resp = requests.get(url, headers=CFBD_HEADERS, params=params, timeout=15)
        time.sleep(0.25)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

# ============================================================================
# STEP 1: DRAFT PICKS — All TEs drafted 2015-2025, Rounds 1-7
# ============================================================================
print("=" * 100)
print("STEP 1: LOADING DRAFT PICKS (TEs 2015-2025)")
print("=" * 100)

draft = pd.read_parquet('data/nflverse/draft_picks.parquet')
te_draft = draft[(draft['position'] == 'TE') & (draft['season'] >= 2015) & (draft['season'] <= 2025)].copy()
te_draft = te_draft.rename(columns={
    'season': 'draft_year', 'pfr_player_name': 'player_name',
    'pick': 'pick', 'college': 'college'
})
te_draft['name_norm'] = te_draft['player_name'].apply(normalize_name)

print(f"  Total TEs drafted: {len(te_draft)}")
print(f"  By round: {te_draft['round'].value_counts().sort_index().to_dict()}")
print(f"  By year: {te_draft['draft_year'].value_counts().sort_index().to_dict()}")

# ============================================================================
# STEP 2: NFL OUTCOMES — Build hit24/hit12/PPG from weekly stats
# ============================================================================
print(f"\n{'='*100}")
print("STEP 2: BUILDING NFL OUTCOMES (from NFLverse weekly stats)")
print("=" * 100)

# Load all weekly stats
stats_frames = []
for f in sorted(glob.glob('data/nflverse/player_stats_*.parquet')):
    df = pd.read_parquet(f)
    te_df = df[df['position'] == 'TE'][['player_id', 'player_display_name', 'position',
                                         'season', 'week', 'season_type', 'fantasy_points_ppr',
                                         'receptions', 'targets', 'receiving_yards', 'receiving_tds',
                                         'recent_team']]
    stats_frames.append(te_df)

# Also load the all-years CSV for older seasons
all_csv = pd.read_csv('data/nflverse/player_stats_all_years.csv')
all_te = all_csv[all_csv['position'] == 'TE'][['player_id', 'player_display_name', 'position',
                                                 'season', 'week', 'season_type', 'fantasy_points_ppr',
                                                 'receptions', 'targets', 'receiving_yards', 'receiving_tds',
                                                 'recent_team']]
stats_frames.append(all_te)

all_stats = pd.concat(stats_frames, ignore_index=True)
# Deduplicate (parquet + CSV may overlap)
all_stats = all_stats.drop_duplicates(subset=['player_id', 'season', 'week', 'season_type'])
# Regular season only
all_stats = all_stats[all_stats['season_type'] == 'REG']
print(f"  Loaded {len(all_stats)} TE regular-season weekly stat rows")

# Aggregate to season totals
season_totals = all_stats.groupby(['player_id', 'player_display_name', 'season']).agg(
    fantasy_points_ppr=('fantasy_points_ppr', 'sum'),
    games=('week', 'nunique'),
    receptions=('receptions', 'sum'),
    targets=('targets', 'sum'),
    receiving_yards=('receiving_yards', 'sum'),
    receiving_tds=('receiving_tds', 'sum'),
).reset_index()

# Rank within position per season
season_totals['pos_rank'] = (
    season_totals.groupby('season')['fantasy_points_ppr']
    .rank(ascending=False, method='min')
)

# PPG (only count seasons with 6+ games to filter out injuries)
season_totals['ppg'] = season_totals['fantasy_points_ppr'] / season_totals['games']

print(f"  Season totals: {len(season_totals)} TE-seasons")

# Match to drafted TEs
results = []
for _, dp in te_draft.iterrows():
    name = dp['player_name']
    draft_year = dp['draft_year']
    pick = dp['pick']
    rd = dp['round']
    college = dp['college']
    draft_age = dp['age'] if pd.notna(dp.get('age')) else np.nan
    gsis_id = dp.get('gsis_id')

    # Try matching by gsis_id first, then by name
    player_seasons = pd.DataFrame()
    if pd.notna(gsis_id):
        player_seasons = season_totals[
            (season_totals['player_id'] == gsis_id) &
            (season_totals['season'] >= draft_year)
        ]
    if len(player_seasons) == 0:
        # Fallback: name match
        nn = normalize_name(name)
        for _, ss in season_totals.iterrows():
            if normalize_name(ss['player_display_name']) == nn and ss['season'] >= draft_year:
                player_seasons = pd.concat([player_seasons, ss.to_frame().T])

    seasons_played = 2025 - draft_year

    if len(player_seasons) == 0:
        results.append({
            'player_name': name, 'draft_year': draft_year, 'pick': pick, 'round': rd,
            'college': college, 'draft_age': draft_age,
            'seasons_played': seasons_played,
            'best_rank': 999, 'best_ppr': 0, 'best_ppg': 0,
            'hit24': 0, 'hit12': 0,
            'first_3yr_ppg': np.nan, 'career_ppg': np.nan,
            'nfl_seasons_found': 0,
        })
    else:
        best_rank = player_seasons['pos_rank'].min()
        best_ppr = player_seasons['fantasy_points_ppr'].max()
        best_ppg_val = player_seasons[player_seasons['games'] >= 6]['ppg'].max() if (player_seasons['games'] >= 6).any() else 0

        # First 3 years PPG
        first_3yr = player_seasons[
            (player_seasons['season'] >= draft_year) &
            (player_seasons['season'] <= draft_year + 2) &
            (player_seasons['games'] >= 6)
        ]
        if len(first_3yr) > 0:
            total_pts = first_3yr['fantasy_points_ppr'].sum()
            total_games = first_3yr['games'].sum()
            first_3yr_ppg = total_pts / total_games if total_games > 0 else np.nan
        else:
            first_3yr_ppg = np.nan

        # Career PPG
        career = player_seasons[player_seasons['games'] >= 6]
        if len(career) > 0:
            career_ppg = career['fantasy_points_ppr'].sum() / career['games'].sum()
        else:
            career_ppg = np.nan

        results.append({
            'player_name': name, 'draft_year': draft_year, 'pick': pick, 'round': rd,
            'college': college, 'draft_age': draft_age,
            'seasons_played': seasons_played,
            'best_rank': best_rank, 'best_ppr': best_ppr, 'best_ppg': best_ppg_val,
            'hit24': 1 if best_rank <= 24 else 0,
            'hit12': 1 if best_rank <= 12 else 0,
            'first_3yr_ppg': first_3yr_ppg, 'career_ppg': career_ppg,
            'nfl_seasons_found': len(player_seasons),
        })

te = pd.DataFrame(results)
te['name_norm'] = te['player_name'].apply(normalize_name)

print(f"  Matched NFL outcomes for {(te['nfl_seasons_found'] > 0).sum()}/{len(te)} TEs")
print(f"  Hit24 rate: {te['hit24'].mean()*100:.1f}% ({te['hit24'].sum()}/{len(te)})")
print(f"  Hit12 rate: {te['hit12'].mean()*100:.1f}% ({te['hit12'].sum()}/{len(te)})")

# ============================================================================
# STEP 3: COMBINE DATA
# ============================================================================
print(f"\n{'='*100}")
print("STEP 3: MERGING COMBINE DATA")
print("=" * 100)

combine = pd.read_parquet('data/nflverse/combine.parquet')
te_combine = combine[combine['pos'] == 'TE'].copy()
te_combine['name_norm'] = te_combine['player_name'].apply(normalize_name)
te_combine['draft_year'] = te_combine['draft_year'].fillna(te_combine['season'])

# Build lookup
combine_lookup = {}
for _, row in te_combine.iterrows():
    dy = int(row['draft_year']) if pd.notna(row['draft_year']) else None
    if dy is None: continue
    key = (row['name_norm'], dy)
    if key not in combine_lookup:
        combine_lookup[key] = row

# Merge
combine_cols = ['ht', 'wt', 'forty', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']
for col in combine_cols:
    if col == 'ht':
        te[col] = ''
    else:
        te[col] = np.nan

for idx in te.index:
    key = (te.loc[idx, 'name_norm'], int(te.loc[idx, 'draft_year']))
    match = combine_lookup.get(key)
    if match is not None:
        for col in combine_cols:
            val = match.get(col)
            if pd.notna(val):
                te.loc[idx, col] = val

for col in combine_cols:
    n = te[col].notna().sum()
    print(f"  {col}: {n}/{len(te)} ({n/len(te)*100:.0f}%)")

# ============================================================================
# STEP 4: PFF COLLEGE DATA
# ============================================================================
print(f"\n{'='*100}")
print("STEP 4: MERGING PFF COLLEGE DATA")
print("=" * 100)

# Map files to college seasons (identified from known players)
pff_file_map = {
    'data/receiving_summary (2).csv': 2015,
    'data/receiving_summary (3).csv': 2016,
    'data/receiving_summary (4).csv': 2017,
    # (13) is duplicate of (4), skip
    'data/receiving_summary (5).csv': 2018,
    'data/receiving_summary (21).csv': 2019,
    'data/receiving_summary (20).csv': 2020,
    'data/receiving_summary (19).csv': 2021,
    'data/receiving_summary (18).csv': 2022,
    'data/receiving_summary (17).csv': 2023,
    'data/receiving_summary (16).csv': 2024,
    'data/receiving_summary (15).csv': 2025,
}

# For each drafted TE, find their FINAL college season PFF data
# Final season = draft_year - 1
pff_all = {}
for fpath, season in pff_file_map.items():
    df = pd.read_csv(fpath)
    te_pff = df[df['position'] == 'TE'].copy()
    te_pff['name_norm'] = te_pff['player'].apply(normalize_name)
    for _, row in te_pff.iterrows():
        key = (row['name_norm'], season)
        pff_all[key] = row

pff_cols = ['player_game_count', 'avg_depth_of_target', 'avoided_tackles', 'caught_percent',
            'contested_catch_rate', 'contested_receptions', 'contested_targets',
            'drop_rate', 'drops', 'first_downs', 'fumbles',
            'grades_hands_drop', 'grades_hands_fumble', 'grades_offense',
            'grades_pass_block', 'grades_pass_route',
            'inline_rate', 'inline_snaps', 'longest',
            'pass_block_rate', 'pass_blocks', 'pass_plays',
            'receptions', 'route_rate', 'routes',
            'slot_rate', 'slot_snaps',
            'targeted_qb_rating', 'targets', 'touchdowns',
            'wide_rate', 'wide_snaps',
            'yards', 'yards_after_catch', 'yards_after_catch_per_reception',
            'yards_per_reception', 'yprr']

# Add PFF columns with pff_ prefix to avoid collision with NFL stats
for col in pff_cols:
    te[f'pff_{col}'] = np.nan

te['pff_team_name'] = ''
te['pff_matched'] = False

for idx in te.index:
    nn = te.loc[idx, 'name_norm']
    dy = int(te.loc[idx, 'draft_year'])
    final_season = dy - 1

    match = pff_all.get((nn, final_season))
    if match is not None:
        te.loc[idx, 'pff_matched'] = True
        te.loc[idx, 'pff_team_name'] = match.get('team_name', '')
        for col in pff_cols:
            val = match.get(col)
            if pd.notna(val):
                te.loc[idx, f'pff_{col}'] = val

n_pff = te['pff_matched'].sum()
print(f"  PFF matched: {n_pff}/{len(te)} ({n_pff/len(te)*100:.0f}%)")

# Show PFF column coverage
for col in ['grades_offense', 'grades_pass_route', 'grades_pass_block', 'yprr',
            'targets', 'receptions', 'yards', 'touchdowns', 'routes',
            'inline_rate', 'slot_rate', 'wide_rate', 'caught_percent', 'drop_rate',
            'contested_catch_rate', 'avg_depth_of_target', 'yards_after_catch']:
    n = te[f'pff_{col}'].notna().sum()
    print(f"  pff_{col}: {n}/{len(te)} ({n/len(te)*100:.0f}%)")

# ============================================================================
# STEP 5: CFBD COLLEGE STATS
# ============================================================================
print(f"\n{'='*100}")
print("STEP 5: PULLING CFBD COLLEGE STATS")
print("=" * 100)

# For each TE, get their final college season receiving stats + team pass attempts from CFBD
# Also get rushing stats and multi-season data for breakout age calculation

# First, get team pass attempts for all teams/seasons we need
team_pass_cache = {}
te['cfbd_rec_yards'] = np.nan
te['cfbd_receptions'] = np.nan
te['cfbd_team_pass_att'] = np.nan
te['cfbd_rush_yards'] = np.nan
te['cfbd_team_rec_yards'] = np.nan
te['cfbd_games'] = np.nan
te['cfbd_matched'] = False

# School name normalization for CFBD
school_fixes = {
    'LSU': 'LSU', 'USC': 'USC', 'UCF': 'UCF', 'SMU': 'SMU', 'BYU': 'BYU',
    'TCU': 'TCU', 'UTSA': 'UTSA', 'UAB': 'UAB', 'UNLV': 'UNLV',
    'Miami': 'Miami', 'Miami (FL)': 'Miami',
    'Ole Miss': 'Mississippi', 'Mississippi': 'Mississippi',
    'Pitt': 'Pittsburgh', 'Pittsburgh': 'Pittsburgh',
}

print(f"  Fetching CFBD data for {len(te)} TEs...")
fetched = 0
for idx in te.index:
    name = te.loc[idx, 'player_name']
    nn = te.loc[idx, 'name_norm']
    college = te.loc[idx, 'college']
    dy = int(te.loc[idx, 'draft_year'])
    final_season = dy - 1

    if pd.isna(college) or college == '':
        continue

    cfbd_school = school_fixes.get(college, college)

    # Get team pass attempts (cache by school+season)
    cache_key = (cfbd_school, final_season)
    if cache_key not in team_pass_cache:
        result = cfbd_get("https://api.collegefootballdata.com/stats/season",
                         {'year': final_season, 'team': cfbd_school})
        if result:
            tpa = None
            for r in result:
                if r.get('statName') == 'passAttempts':
                    tpa = float(r['statValue'])
                    break
            team_pass_cache[cache_key] = tpa
        else:
            team_pass_cache[cache_key] = None

    tpa = team_pass_cache.get(cache_key)
    if tpa is not None:
        te.loc[idx, 'cfbd_team_pass_att'] = tpa

    # Get player receiving stats
    result = cfbd_get("https://api.collegefootballdata.com/stats/player/season",
                     {'year': final_season, 'team': cfbd_school, 'category': 'receiving'})
    if result:
        player_stats = {}
        for r in result:
            if normalize_name(r.get('player', '')) == nn:
                player_stats[r.get('statType', '')] = r.get('stat')
        if 'YDS' in player_stats:
            te.loc[idx, 'cfbd_rec_yards'] = float(player_stats['YDS'])
            te.loc[idx, 'cfbd_matched'] = True
            fetched += 1
        if 'REC' in player_stats:
            te.loc[idx, 'cfbd_receptions'] = float(player_stats['REC'])

    # Get player rushing stats
    result = cfbd_get("https://api.collegefootballdata.com/stats/player/season",
                     {'year': final_season, 'team': cfbd_school, 'category': 'rushing'})
    if result:
        for r in result:
            if normalize_name(r.get('player', '')) == nn and r.get('statType') == 'YDS':
                te.loc[idx, 'cfbd_rush_yards'] = float(r['stat'])
                break

    if fetched % 20 == 0 and fetched > 0:
        print(f"    ...fetched {fetched} TEs so far")

n_cfbd = te['cfbd_matched'].sum()
n_tpa = te['cfbd_team_pass_att'].notna().sum()
print(f"  CFBD receiving matched: {n_cfbd}/{len(te)} ({n_cfbd/len(te)*100:.0f}%)")
print(f"  Team pass attempts: {n_tpa}/{len(te)} ({n_tpa/len(te)*100:.0f}%)")

# ============================================================================
# STEP 6: BREAKOUT AGE (from CFBD multi-season data)
# ============================================================================
print(f"\n{'='*100}")
print("STEP 6: CALCULATING BREAKOUT AGES (multi-season CFBD)")
print("=" * 100)

# For breakout age, need multi-season dominator data
# dominator = player_rec_yards / team_rec_yards
# breakout = first season with 20%+ dominator
# This requires team receiving yards per season + player rec yards per season

te['breakout_age'] = np.nan
te['peak_dominator'] = np.nan
te['breakout_season'] = np.nan

breakout_fetched = 0
for idx in te.index:
    name = te.loc[idx, 'player_name']
    nn = te.loc[idx, 'name_norm']
    college = te.loc[idx, 'college']
    dy = int(te.loc[idx, 'draft_year'])
    draft_age = te.loc[idx, 'draft_age']

    if pd.isna(college) or college == '':
        continue

    cfbd_school = school_fixes.get(college, college)

    # Check up to 5 seasons back
    season_data = []
    for yr in range(dy - 5, dy):
        # Player receiving
        result = cfbd_get("https://api.collegefootballdata.com/stats/player/season",
                         {'year': yr, 'team': cfbd_school, 'category': 'receiving'})
        player_yds = None
        if result:
            for r in result:
                if normalize_name(r.get('player', '')) == nn and r.get('statType') == 'YDS':
                    player_yds = float(r['stat'])
                    break

        if player_yds is None:
            continue

        # Team total receiving yards (sum all players)
        team_rec_yds = 0
        if result:
            for r in result:
                if r.get('statType') == 'YDS':
                    try:
                        team_rec_yds += float(r['stat'])
                    except:
                        pass

        if team_rec_yds > 0:
            dom_pct = (player_yds / team_rec_yds) * 100
            # Calculate age at season
            if pd.notna(draft_age):
                age_at_season = draft_age - (dy - yr)
            else:
                age_at_season = None
            season_data.append({
                'season': yr, 'rec_yards': player_yds,
                'team_rec_yards': team_rec_yds,
                'dominator_pct': dom_pct,
                'age_at_season': age_at_season,
            })

    if season_data:
        # Peak dominator
        peak_dom = max(s['dominator_pct'] for s in season_data)
        te.loc[idx, 'peak_dominator'] = peak_dom

        # Breakout: first season with 20%+ dominator
        breakout_seasons = [s for s in season_data if s['dominator_pct'] >= 20]
        if breakout_seasons:
            first_breakout = min(breakout_seasons, key=lambda s: s['season'])
            te.loc[idx, 'breakout_season'] = first_breakout['season']
            if first_breakout['age_at_season'] is not None:
                te.loc[idx, 'breakout_age'] = first_breakout['age_at_season']

        breakout_fetched += 1

    if breakout_fetched % 20 == 0 and breakout_fetched > 0:
        print(f"    ...processed {breakout_fetched} TEs for breakout age")

n_bo = te['breakout_age'].notna().sum()
n_pd = te['peak_dominator'].notna().sum()
n_hit20 = (te['breakout_season'].notna()).sum()
print(f"  Breakout age calculated: {n_bo}/{len(te)} ({n_bo/len(te)*100:.0f}%)")
print(f"  Peak dominator: {n_pd}/{len(te)} ({n_pd/len(te)*100:.0f}%)")
print(f"  Hit 20% dominator: {n_hit20}/{len(te)} ({n_hit20/len(te)*100:.0f}%)")

# ============================================================================
# STEP 7: EARLY DECLARE STATUS
# ============================================================================
print(f"\n{'='*100}")
print("STEP 7: DETERMINING EARLY DECLARE STATUS")
print("=" * 100)

# If draft_age <= 22, likely early declare (left before senior year)
# Standard 4-year path: draft at ~22. Early = 21 or younger.
te['early_declare'] = 0
for idx in te.index:
    age = te.loc[idx, 'draft_age']
    if pd.notna(age) and float(age) <= 22:
        te.loc[idx, 'early_declare'] = 1

n_ed = te['early_declare'].sum()
n_age_known = te['draft_age'].notna().sum()
print(f"  Draft age known: {n_age_known}/{len(te)} ({n_age_known/len(te)*100:.0f}%)")
print(f"  Early declare (age <= 22): {n_ed}/{len(te)} ({n_ed/len(te)*100:.0f}%)")

# ============================================================================
# STEP 8: SAVE MASTER FILE
# ============================================================================
print(f"\n{'='*100}")
print("STEP 8: SAVING MASTER FILE")
print("=" * 100)

# Reorder columns for clarity
id_cols = ['player_name', 'draft_year', 'pick', 'round', 'college', 'draft_age', 'early_declare']
outcome_cols = ['seasons_played', 'nfl_seasons_found', 'hit24', 'hit12',
                'first_3yr_ppg', 'career_ppg', 'best_ppr', 'best_ppg', 'best_rank']
combine_cols_out = ['ht', 'wt', 'forty', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']
breakout_cols = ['breakout_age', 'peak_dominator', 'breakout_season']
cfbd_cols = ['cfbd_rec_yards', 'cfbd_receptions', 'cfbd_team_pass_att', 'cfbd_rush_yards',
             'cfbd_team_rec_yards', 'cfbd_matched']
pff_col_list = [f'pff_{c}' for c in pff_cols] + ['pff_team_name', 'pff_matched']

all_cols = id_cols + outcome_cols + combine_cols_out + breakout_cols + cfbd_cols + pff_col_list + ['name_norm']
# Only include cols that exist
final_cols = [c for c in all_cols if c in te.columns]
te_out = te[final_cols].copy()

te_out.to_csv('data/te_backtest_master.csv', index=False)
print(f"  Saved: data/te_backtest_master.csv ({len(te_out)} TEs, {len(final_cols)} columns)")

# ============================================================================
# STEP 9: SHOW THE LANDSCAPE
# ============================================================================
print(f"\n\n{'='*100}")
print("TE BACKTEST DATA LANDSCAPE")
print("=" * 100)

print(f"\n  TOTAL TEs: {len(te)}")
print(f"  Draft years: 2015-2025 ({len(te['draft_year'].unique())} years)")

print(f"\n  BY ROUND:")
print(f"  {'Round':<8s} {'Count':>6s} {'Hit24':>6s} {'Hit24%':>7s} {'Hit12':>6s} {'Hit12%':>7s}")
print(f"  {'-'*42}")
for rd in range(1, 8):
    sub = te[te['round'] == rd]
    if len(sub) == 0: continue
    h24 = sub['hit24'].sum()
    h12 = sub['hit12'].sum()
    print(f"  Rd {rd:<4d} {len(sub):>6d} {h24:>6.0f} {h24/len(sub)*100:>6.1f}% {h12:>6.0f} {h12/len(sub)*100:>6.1f}%")
total_h24 = te['hit24'].sum()
total_h12 = te['hit12'].sum()
print(f"  {'TOTAL':<8s} {len(te):>6d} {total_h24:>6.0f} {total_h24/len(te)*100:>6.1f}% {total_h12:>6.0f} {total_h12/len(te)*100:>6.1f}%")

print(f"\n  DATA COVERAGE:")
coverage = {
    'Draft pick/round/college': te['pick'].notna().sum(),
    'Draft age': te['draft_age'].notna().sum(),
    'NFL outcomes (any games)': (te['nfl_seasons_found'] > 0).sum(),
    'First 3yr PPG': te['first_3yr_ppg'].notna().sum(),
    'PFF college data': te['pff_matched'].sum(),
    'CFBD receiving': te['cfbd_matched'].sum(),
    'CFBD team pass att': te['cfbd_team_pass_att'].notna().sum(),
    'Breakout age': te['breakout_age'].notna().sum(),
    'Peak dominator': te['peak_dominator'].notna().sum(),
    'Combine weight': te['wt'].notna().sum(),
    'Combine 40 time': te['forty'].notna().sum(),
    'Combine height': te['ht'].notna().sum(),
    'Combine bench': te['bench'].notna().sum(),
    'Combine vertical': te['vertical'].notna().sum(),
    'Combine broad jump': te['broad_jump'].notna().sum(),
    'Combine 3-cone': te['cone'].notna().sum(),
    'Combine shuttle': te['shuttle'].notna().sum(),
}
for label, n in coverage.items():
    print(f"  {label:<30s}: {n:>4d}/{len(te)} ({n/len(te)*100:.0f}%)")

# Show PFF key metrics coverage
print(f"\n  PFF METRIC COVERAGE (key metrics):")
for col in ['grades_offense', 'grades_pass_route', 'grades_pass_block', 'yprr',
            'targets', 'receptions', 'yards', 'routes', 'inline_rate', 'slot_rate',
            'caught_percent', 'drop_rate', 'contested_catch_rate', 'avg_depth_of_target']:
    n = te[f'pff_{col}'].notna().sum()
    print(f"    pff_{col:<30s}: {n:>4d}/{len(te)} ({n/len(te)*100:.0f}%)")

# ============================================================================
# SAMPLE: 10 WELL-KNOWN TEs
# ============================================================================
print(f"\n\n{'='*100}")
print("SAMPLE: WELL-KNOWN TEs")
print("=" * 100)

known_tes = ['George Kittle', 'Mark Andrews', 'T.J. Hockenson', 'Kyle Pitts',
             'Dallas Goedert', 'Pat Freiermuth', 'Sam LaPorta', 'Brock Bowers',
             'Dalton Kincaid', 'Evan Engram']

for name in known_tes:
    nn = normalize_name(name)
    match = te[te['name_norm'] == nn]
    if len(match) == 0:
        print(f"\n  {name}: NOT FOUND")
        continue
    r = match.iloc[0]
    print(f"\n  {name} ({r['college']}, {int(r['draft_year'])} Rd {int(r['round'])} Pick {int(r['pick'])})")
    print(f"    NFL: hit24={int(r['hit24'])}, hit12={int(r['hit12'])}, "
          f"3yr_ppg={r['first_3yr_ppg']:.1f}" if pd.notna(r['first_3yr_ppg']) else f"    NFL: hit24={int(r['hit24'])}, hit12={int(r['hit12'])}, 3yr_ppg=—", end='')
    print(f", career_ppg={r['career_ppg']:.1f}" if pd.notna(r['career_ppg']) else ", career_ppg=—", end='')
    print(f", best_ppr={r['best_ppr']:.1f}" if pd.notna(r['best_ppr']) else ", best_ppr=—")
    print(f"    Draft age: {r['draft_age']}" if pd.notna(r['draft_age']) else "    Draft age: —", end='')
    print(f", Early declare: {'Yes' if r['early_declare'] == 1 else 'No'}")
    print(f"    Combine: {r['ht']}, {r['wt']}lbs, {r['forty']}s 40" if pd.notna(r['forty']) else f"    Combine: {r.get('ht','—')}, {r['wt']}lbs, no 40" if pd.notna(r['wt']) else "    Combine: no data")
    print(f"    Breakout age: {r['breakout_age']:.0f}" if pd.notna(r['breakout_age']) else "    Breakout age: —", end='')
    print(f", Peak dominator: {r['peak_dominator']:.1f}%" if pd.notna(r['peak_dominator']) else ", Peak dominator: —")
    pff_grade = r.get('pff_grades_offense')
    pff_yprr_val = r.get('pff_yprr')
    pff_yards_val = r.get('pff_yards')
    pff_inline = r.get('pff_inline_rate')
    pff_slot = r.get('pff_slot_rate')
    pff_wide = r.get('pff_wide_rate')
    print(f"    PFF: grade={pff_grade:.1f}" if pd.notna(pff_grade) else "    PFF: grade=—", end='')
    print(f", yprr={pff_yprr_val:.2f}" if pd.notna(pff_yprr_val) else ", yprr=—", end='')
    print(f", yards={pff_yards_val:.0f}" if pd.notna(pff_yards_val) else ", yards=—", end='')
    print(f" | inline={pff_inline:.0f}%/slot={pff_slot:.0f}%/wide={pff_wide:.0f}%" if pd.notna(pff_inline) else "")
    cfbd_rec = r.get('cfbd_rec_yards')
    cfbd_tpa = r.get('cfbd_team_pass_att')
    print(f"    CFBD: rec_yards={cfbd_rec:.0f}" if pd.notna(cfbd_rec) else "    CFBD: rec_yards=—", end='')
    print(f", team_pass_att={cfbd_tpa:.0f}" if pd.notna(cfbd_tpa) else ", team_pass_att=—")

print(f"\n\n{'='*100}")
print("TE BACKTEST BUILD COMPLETE")
print("=" * 100)
