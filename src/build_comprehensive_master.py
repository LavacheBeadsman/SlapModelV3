"""
Build comprehensive master database: one row per player, every column we have.

Backbone: slap_v5_wr.csv, slap_v5_rb.csv, slap_v5_te.csv (SLAP scores preserved as-is)
Enrichment sources:
  - nflverse/draft_picks.parquet → conference, draft_age
  - nflverse_birthdates_2015_2025.csv + 2026_prospect_birthdates.csv → birthdate
  - nflverse/combine.parquet → backtest combine measurements
  - combine_2026.csv → 2026 combine measurements
  - Anatomy of Top WR & RB files → anatomy dataset variables
  - wr_pff_all_2016_2025.csv → WR PFF grades
  - rb_pff_corrected.csv → RB PFF grades
  - te_backtest_master.csv → TE PFF grades + extra college production
  - backtest_outcomes_complete.csv → NFL outcomes (fill gaps)
  - prospect cards → 2026 college production details
  - WRRas201502025.csv, RBRas20152025.csv, AllTERAS.csv → official RAS scores
  - wr_backtest_all_components.csv → WR backtest extra cols
  - rb_backtest_with_receiving.csv → RB backtest extra cols
  - rb_bdr_calculated.csv → RB backtest BDR details
"""

import pandas as pd
import numpy as np
import warnings
import re
import os
from thefuzz import fuzz, process

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────

def normalize_name(name):
    """Normalize player name for matching."""
    if pd.isna(name):
        return ''
    name = str(name).strip()
    # Remove suffixes like Jr., Sr., III, II, IV
    name = re.sub(r'\s+(Jr\.?|Sr\.?|III|II|IV|V)$', '', name, flags=re.IGNORECASE)
    # Normalize special characters
    name = name.replace("'", "'").replace("'", "'").replace("`", "'")
    # Normalize D' prefixes
    name = re.sub(r"^D'", "D'", name)
    return name.strip()

def fuzzy_merge(left, right, left_on, right_on, left_year='draft_year', right_year='draft_year',
                threshold=85, suffix='_right'):
    """
    Merge two DataFrames using fuzzy name matching within the same draft year.
    Returns the merged DataFrame and a list of fuzzy matches for review.
    """
    # Start with exact merge
    merged = left.merge(right, left_on=[left_on, left_year], right_on=[right_on, right_year],
                        how='left', suffixes=('', suffix), indicator=True)

    exact_matched = merged['_merge'] == 'both'
    exact_count = exact_matched.sum()

    # For unmatched rows, try fuzzy matching
    unmatched_left = merged[merged['_merge'] == 'left_only'].copy()
    fuzzy_matches = []

    if len(unmatched_left) > 0 and len(right) > 0:
        # Build lookup by year
        right_by_year = {}
        for _, row in right.iterrows():
            yr = row[right_year]
            if pd.notna(yr):
                yr = int(yr)
                if yr not in right_by_year:
                    right_by_year[yr] = []
                right_by_year[yr].append(row)

        for idx, row in unmatched_left.iterrows():
            left_name = normalize_name(row[left_on])
            yr = row[left_year]
            if pd.isna(yr) or left_name == '':
                continue
            yr = int(yr)

            candidates = right_by_year.get(yr, [])
            if not candidates:
                continue

            best_score = 0
            best_match = None
            for cand in candidates:
                cand_name = normalize_name(cand[right_on])
                score = fuzz.ratio(left_name.lower(), cand_name.lower())
                if score > best_score:
                    best_score = score
                    best_match = cand

            if best_score >= threshold and best_match is not None:
                fuzzy_matches.append({
                    'left_name': row[left_on],
                    'right_name': best_match[right_on],
                    'year': yr,
                    'score': best_score
                })
                # Copy right columns to merged row
                for col in right.columns:
                    if col not in [right_on, right_year] and col in merged.columns:
                        rcol = col if col + suffix not in merged.columns else col + suffix
                    else:
                        rcol = col + suffix if col + suffix in merged.columns else col
                    if rcol in merged.columns and col in best_match.index:
                        merged.at[idx, rcol] = best_match[col]

    if '_merge' in merged.columns:
        merged = merged.drop(columns=['_merge'])

    return merged, fuzzy_matches


def safe_coalesce(df, primary, fallback, drop_fallback=True):
    """Fill NaN in primary column from fallback column, optionally dropping fallback."""
    if primary in df.columns and fallback in df.columns:
        df[primary] = df[primary].fillna(df[fallback])
        if drop_fallback:
            df = df.drop(columns=[fallback])
    elif fallback in df.columns and primary not in df.columns:
        df = df.rename(columns={fallback: primary})
    return df


def clean_numeric(series):
    """Convert a series to numeric, handling commas and strings."""
    if series.dtype == object:
        series = series.str.replace(',', '').str.strip()
    return pd.to_numeric(series, errors='coerce')


# ─────────────────────────────────────────────────────────
# STEP 1: Load backbone files
# ─────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading backbone files (SLAP scores preserved as-is)")
print("=" * 70)

wr = pd.read_csv('output/slap_v5_wr.csv')
rb = pd.read_csv('output/slap_v5_rb.csv')
te = pd.read_csv('output/slap_v5_te.csv')

# Standardize column names across positions
wr['position'] = 'WR'
rb['position'] = 'RB'
te['position'] = 'TE'

# Rename 'pick' to 'draft_pick' and 'round' to 'draft_round' for clarity
for df in [wr, rb, te]:
    if 'pick' in df.columns:
        df.rename(columns={'pick': 'draft_pick'}, inplace=True)
    if 'round' in df.columns:
        df.rename(columns={'round': 'draft_round'}, inplace=True)

print(f"  WR: {wr.shape[0]} rows, {wr.shape[1]} cols")
print(f"  RB: {rb.shape[0]} rows, {rb.shape[1]} cols")
print(f"  TE: {te.shape[0]} rows, {te.shape[1]} cols")

# Tag position-specific score columns
wr.rename(columns={'enhanced_breakout': 'breakout_score'}, inplace=True)
# RB already has production_score, speed_score
# TE already has te_breakout_score, te_production_score, ras_score

# ─────────────────────────────────────────────────────────
# STEP 2: Load enrichment data
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Loading enrichment sources")
print("=" * 70)

# NFLverse draft picks (for conference, draft_age)
draft_picks = pd.read_parquet('data/nflverse/draft_picks.parquet')
draft_picks = draft_picks[draft_picks['position'].isin(['WR', 'RB', 'TE']) & (draft_picks['season'] >= 2015)]
draft_picks = draft_picks.rename(columns={
    'pfr_player_name': 'dp_name', 'season': 'draft_year',
    'college': 'conference_college', 'age': 'draft_age_dp'
})
print(f"  draft_picks: {len(draft_picks)} rows (2015+ WR/RB/TE)")

# Birthdates
bd_backtest = pd.read_csv('data/nflverse_birthdates_2015_2025.csv')
bd_backtest.rename(columns={'nfl_name': 'player_name', 'birth_date': 'birthdate', 'draft_year': 'draft_year'}, inplace=True)
bd_2026 = pd.read_csv('data/2026_prospect_birthdates.csv')
# TE backtest birthdates (from nflverse players dataset)
te_bd_path = 'data/te_backtest_birthdates.csv'
if os.path.exists(te_bd_path):
    te_bd = pd.read_csv(te_bd_path)
    te_bd = te_bd[te_bd['birthdate'].notna()][['player_name', 'draft_year', 'birthdate']].copy()
    te_bd['position'] = 'TE'
    bd_backtest = pd.concat([bd_backtest, te_bd], ignore_index=True)
print(f"  birthdates: {len(bd_backtest)} backtest + {len(bd_2026)} 2026")

# NFLverse combine (backtest)
combine_nfl = pd.read_parquet('data/nflverse/combine.parquet')
combine_nfl = combine_nfl[combine_nfl['pos'].isin(['WR', 'RB', 'TE'])]
combine_nfl = combine_nfl[combine_nfl['draft_year'].between(2015, 2025)]
print(f"  nflverse combine: {len(combine_nfl)} rows")

# 2026 combine
combine_2026 = pd.read_csv('data/combine_2026.csv')
print(f"  combine_2026: {len(combine_2026)} rows")

# Outcomes
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
print(f"  outcomes: {len(outcomes)} rows")

# WR PFF
wr_pff = pd.read_csv('data/wr_pff_all_2016_2025.csv')
print(f"  WR PFF: {len(wr_pff)} rows")

# RB PFF
rb_pff = pd.read_csv('data/rb_pff_corrected.csv')
print(f"  RB PFF: {len(rb_pff)} rows")

# TE backtest (has PFF + extra college production)
te_bt = pd.read_csv('data/te_backtest_master.csv')
print(f"  TE backtest master: {len(te_bt)} rows")

# TE 2026 prospects (has PFF data too)
te_2026 = pd.read_csv('data/te_2026_prospects_final.csv')
print(f"  TE 2026 prospects: {len(te_2026)} rows")

# WR backtest (has birthdate, declare info, rush stats)
wr_bt = pd.read_csv('data/wr_backtest_all_components.csv')
print(f"  WR backtest components: {len(wr_bt)} rows")

# RB backtest (has extra receiving columns)
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
print(f"  RB backtest: {len(rb_bt)} rows")

# RB BDR
rb_bdr = pd.read_csv('data/rb_bdr_calculated.csv')
print(f"  RB BDR: {len(rb_bdr)} rows")

# WR teammate scores
wr_tm = pd.read_csv('data/wr_teammate_scores.csv')
print(f"  WR teammate scores: {len(wr_tm)} rows")

# RAS files
wr_ras = pd.read_csv('data/WRRas201502025.csv')
rb_ras = pd.read_csv('data/RBRas20152025.csv')
te_ras = pd.read_csv('data/AllTERAS.csv')
print(f"  RAS: WR={len(wr_ras)}, RB={len(rb_ras)}, TE={len(te_ras)}")

# Prospect cards (2026)
wr_cards = pd.read_csv('output/2026_wr_cards.csv')
rb_cards = pd.read_csv('output/2026_rb_cards.csv')
te_cards = pd.read_csv('output/2026_te_cards.csv')
print(f"  Prospect cards: WR={len(wr_cards)}, RB={len(rb_cards)}, TE={len(te_cards)}")

# WR breakout ages 2026
wr_bo_2026 = pd.read_csv('data/wr_breakout_ages_2026.csv')
print(f"  WR breakout 2026: {len(wr_bo_2026)} rows")

# Prospects final (has birthdates, age)
prospects_final = pd.read_csv('data/prospects_final.csv')
print(f"  prospects_final: {len(prospects_final)} rows")


# ─────────────────────────────────────────────────────────
# STEP 3: Parse Anatomy dataset files
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Parsing Anatomy dataset files")
print("=" * 70)

anatomy_wr_rows = []
anatomy_rb_rows = []

anatomy_years = list(range(2015, 2027))

for year in anatomy_years:
    # Try both file name patterns
    fname = f'data/Anatomy of Top WR & RB - {year}.csv'
    try:
        df = pd.read_csv(fname)
    except FileNotFoundError:
        fname = f'data/Anatomy of Top WR & RB - {year} (1).csv'
        try:
            df = pd.read_csv(fname)
        except FileNotFoundError:
            print(f"  {year}: NOT FOUND")
            continue

    # WR section: Unnamed: 0 = player name, then WR columns
    wr_name_col = 'Unnamed: 0'
    if wr_name_col not in df.columns:
        print(f"  {year}: No WR name column")
        continue

    # Get WR data
    wr_section = df[[c for c in df.columns if not c.startswith('Unnamed: 1') and
                     not c.startswith('Unnamed: 2') and
                     not c.endswith('.1')]].copy()

    for _, row in df.iterrows():
        name = row.get(wr_name_col)
        if pd.isna(name) or str(name).strip() == '':
            continue

        wr_row = {
            'player_name': str(name).strip(),
            'anatomy_year': year,
            'anatomy_wdom': pd.to_numeric(row.get('WDOM'), errors='coerce') if 'WDOM' in df.columns else np.nan,
            'anatomy_yptpa': pd.to_numeric(row.get('YPTPA'), errors='coerce') if 'YPTPA' in df.columns else np.nan,
            'anatomy_ypr_dom': pd.to_numeric(row.get('YPR DOM'), errors='coerce') if 'YPR DOM' in df.columns else np.nan,
            'anatomy_boy': pd.to_numeric(row.get('BOY', row.get('BOY in year 1 or 2')), errors='coerce'),
            'anatomy_ypr': pd.to_numeric(row.get('YPR'), errors='coerce') if 'YPR' in df.columns else np.nan,
            'anatomy_boa': pd.to_numeric(row.get('BOA', row.get('BOA of 18-19')), errors='coerce'),
            'anatomy_bmi_wr': pd.to_numeric(row.get('BMI'), errors='coerce') if 'BMI' in df.columns else np.nan,
            'anatomy_ras_wr': pd.to_numeric(row.get('RAS'), errors='coerce') if 'RAS' in df.columns else np.nan,
        }
        anatomy_wr_rows.append(wr_row)

    # RB section: depends on year - some files use Unnamed: 20, some use Unnamed: 17
    rb_name_col = None
    for candidate in ['Unnamed: 20', 'Unnamed: 22', 'Unnamed: 17', 'Unnamed: 13', 'Unnamed: 12']:
        if candidate in df.columns:
            names = df[candidate].dropna()
            if len(names) > 0:
                rb_name_col = candidate
                break

    if rb_name_col:
        # Find column names with alternatives for 2018 variant format
        def find_col(candidates, columns):
            """Find the first matching column from a list of candidates."""
            for c in candidates:
                for col in columns:
                    if c.lower().strip() in col.lower().strip():
                        return col
            return None

        speed_col = find_col(['Speed Score', '100+'], df.columns)
        burst_col = find_col(['Burst Score', '121+ Burst'], df.columns)
        peak_yd_col = find_col(['Peak Yardage', 'Scrimmage', '1500'], df.columns)
        peak_rec_col = find_col(['Peak REC', '30+ Recption', 'Recption Season'], df.columns)
        college_ypa_col = find_col(['College YPA'], df.columns)

        for _, row in df.iterrows():
            name = row.get(rb_name_col)
            if pd.isna(name) or str(name).strip() == '':
                continue

            rb_row = {
                'player_name': str(name).strip(),
                'anatomy_year': year,
                'anatomy_speed_score': pd.to_numeric(row.get(speed_col), errors='coerce') if speed_col else np.nan,
                'anatomy_burst_score': pd.to_numeric(row.get(burst_col), errors='coerce') if burst_col else np.nan,
                'anatomy_bdr': pd.to_numeric(row.get('BDR'), errors='coerce') if 'BDR' in df.columns else np.nan,
                'anatomy_bmi_rb': pd.to_numeric(row.get('BMI.1'), errors='coerce') if 'BMI.1' in df.columns else np.nan,
                'anatomy_ras_rb': pd.to_numeric(row.get('RAS.1'), errors='coerce') if 'RAS.1' in df.columns else np.nan,
                'anatomy_ypa_dom': pd.to_numeric(row.get('YPA DOM'), errors='coerce') if 'YPA DOM' in df.columns else np.nan,
            }
            # Peak Yardage may have commas — try multiple column names
            pk_yd = row.get(peak_yd_col) if peak_yd_col else None
            if pk_yd is None:
                pk_yd = row.get('Peak Yardage')
            if pd.notna(pk_yd):
                rb_row['anatomy_peak_yardage'] = pd.to_numeric(str(pk_yd).replace(',', ''), errors='coerce')
            else:
                rb_row['anatomy_peak_yardage'] = np.nan

            pk_rec = row.get(peak_rec_col) if peak_rec_col else row.get('Peak REC')
            rb_row['anatomy_peak_rec'] = pd.to_numeric(pk_rec, errors='coerce') if pk_rec is not None else np.nan

            # College YPA (no College YPC exists in Anatomy files)
            col_ypa = row.get(college_ypa_col) if college_ypa_col else row.get('College YPA')
            rb_row['anatomy_college_ypa'] = pd.to_numeric(col_ypa, errors='coerce') if col_ypa is not None else np.nan

            anatomy_rb_rows.append(rb_row)

    wr_count = len([r for r in anatomy_wr_rows if r['anatomy_year'] == year])
    rb_count = len([r for r in anatomy_rb_rows if r['anatomy_year'] == year])
    print(f"  {year}: {wr_count} WRs, {rb_count} RBs")

anatomy_wr = pd.DataFrame(anatomy_wr_rows)
anatomy_rb = pd.DataFrame(anatomy_rb_rows)
print(f"\n  Total Anatomy: {len(anatomy_wr)} WR rows, {len(anatomy_rb)} RB rows")


# ─────────────────────────────────────────────────────────
# STEP 4: Parse official RAS scores
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Parsing RAS scores")
print("=" * 70)

def parse_ras_file(df, pos_label):
    """Parse RAS file with Name, Year, College, RAS columns."""
    df = df.copy()
    df = df.rename(columns={'Name': 'player_name', 'Year': 'draft_year', 'College': 'ras_college'})
    df['ras_official'] = pd.to_numeric(df['RAS'], errors='coerce')
    df = df[['player_name', 'draft_year', 'ras_college', 'ras_official']].copy()
    df['player_name'] = df['player_name'].str.strip()
    df['draft_year'] = pd.to_numeric(df['draft_year'], errors='coerce')
    df = df.dropna(subset=['player_name', 'draft_year'])
    # Keep best RAS per player+year (some files have duplicates)
    df = df.sort_values('ras_official', ascending=False).drop_duplicates(subset=['player_name', 'draft_year'], keep='first')
    return df

ras_wr = parse_ras_file(wr_ras, 'WR')
ras_rb = parse_ras_file(rb_ras, 'RB')
ras_te = parse_ras_file(te_ras, 'TE')
print(f"  WR RAS: {len(ras_wr)}, RB RAS: {len(ras_rb)}, TE RAS: {len(ras_te)}")


# ─────────────────────────────────────────────────────────
# STEP 5: Enrich WR data
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Enriching WR data")
print("=" * 70)

# 5a. Merge WR backtest extra columns (birthdate, rush_attempts, rush_tds, declare_source)
wr_bt_extra = wr_bt[['player_name', 'draft_year', 'birthdate', 'draft_age',
                       'rush_attempts', 'rush_tds', 'seasons_played']].copy()
wr_bt_extra.rename(columns={'rush_tds': 'rush_touchdowns_bt'}, inplace=True)

wr = wr.merge(wr_bt_extra, on=['player_name', 'draft_year'], how='left', suffixes=('', '_bt'))
print(f"  Merged WR backtest extras: birthdate, draft_age, rush_attempts")

# 5b. Merge WR PFF grades
wr_pff_cols = wr_pff[['player_name', 'draft_year', 'yprr', 'targets', 'receptions',
                       'grades_offense', 'grades_pass_route', 'drop_rate',
                       'contested_catch_rate', 'avg_depth_of_target',
                       'yards_after_catch', 'yards_after_catch_per_reception',
                       'slot_rate', 'wide_rate', 'caught_percent',
                       'rush_attempts', 'rush_yards', 'rush_touchdowns', 'rush_ypa']].copy()
wr_pff_cols.rename(columns={
    'yprr': 'pff_yprr',
    'targets': 'pff_targets',
    'receptions': 'pff_receptions',
    'grades_offense': 'pff_overall_grade',
    'grades_pass_route': 'pff_receiving_grade',
    'drop_rate': 'pff_drop_rate',
    'contested_catch_rate': 'pff_contested_catch_rate',
    'avg_depth_of_target': 'pff_avg_depth_of_target',
    'yards_after_catch': 'pff_yards_after_catch',
    'yards_after_catch_per_reception': 'pff_yac_per_reception',
    'slot_rate': 'pff_slot_rate',
    'wide_rate': 'pff_wide_rate',
    'caught_percent': 'pff_caught_percent',
    'rush_attempts': 'pff_rush_attempts',
    'rush_yards': 'pff_rush_yards',
    'rush_touchdowns': 'pff_rush_touchdowns',
    'rush_ypa': 'pff_rush_ypa',
}, inplace=True)

# Drop duplicates - keep first (best match)
wr_pff_cols = wr_pff_cols.drop_duplicates(subset=['player_name', 'draft_year'], keep='first')

wr = wr.merge(wr_pff_cols, on=['player_name', 'draft_year'], how='left')
print(f"  Merged WR PFF: {wr_pff_cols.shape[1]-2} columns")

# 5c. Merge WR RAS official
wr = wr.merge(ras_wr[['player_name', 'draft_year', 'ras_official']],
              on=['player_name', 'draft_year'], how='left')
# Rename ras_official to ras_score for WR (they don't have one yet in backbone)
wr.rename(columns={'ras_official': 'ras_score'}, inplace=True)
print(f"  Merged WR RAS official")

# 5d. Merge WR teammate details
wr_tm_extra = wr_tm[['player_name', 'draft_year', 'teammate_count', 'best_teammate_pick',
                      'total_teammate_dc', 'teammate_names']].copy()
wr = wr.merge(wr_tm_extra, on=['player_name', 'draft_year'], how='left')
print(f"  Merged WR teammate details")

# 5e. Merge WR 2026 prospect production from cards
wr_cards_prod = wr_cards[['player_name', 'receptions', 'rec_tds', 'rush_attempts', 'rush_yards', 'rush_tds',
                           'games_played', 'total_touchdowns', 'team_rec_yards', 'team_rec_tds',
                           'rec_yards_share', 'rec_td_share', 'dominator_rating',
                           'rec_yards_per_team_pass_att']].copy()
wr_cards_prod.rename(columns={
    'receptions': 'receptions_card',
    'rec_tds': 'rec_tds_card',
    'rush_attempts': 'rush_attempts_card',
    'rush_yards': 'rush_yards_card',
    'rush_tds': 'rush_tds_card',
    'games_played': 'games_played_card',
}, inplace=True)

wr = wr.merge(wr_cards_prod, on='player_name', how='left')
print(f"  Merged WR 2026 card production")

# 5f. Merge WR 2026 breakout season info
wr_bo_extra = wr_bo_2026[['player_name', 'breakout_season', 'seasons_found']].copy()
wr_bo_extra.rename(columns={'seasons_found': 'college_seasons_found'}, inplace=True)
wr = wr.merge(wr_bo_extra, on='player_name', how='left')

# 5g. WR Anatomy data
# Match anatomy to WR rows by player name and year
# For backtest, anatomy_year = draft_year. For 2026, anatomy_year = 2026
if len(anatomy_wr) > 0:
    # Normalize anatomy names
    anatomy_wr['name_norm'] = anatomy_wr['player_name'].apply(normalize_name).str.lower()

    wr_anatomy_matched = 0
    wr_anatomy_failed = []

    for idx, row in wr.iterrows():
        name = normalize_name(row['player_name']).lower()
        yr = row['draft_year']

        # Try exact match first
        match = anatomy_wr[(anatomy_wr['name_norm'] == name) & (anatomy_wr['anatomy_year'] == yr)]

        if len(match) == 0:
            # Try fuzzy within year
            year_candidates = anatomy_wr[anatomy_wr['anatomy_year'] == yr]
            if len(year_candidates) > 0:
                best_score = 0
                best_idx = None
                for cidx, cand in year_candidates.iterrows():
                    score = fuzz.ratio(name, cand['name_norm'])
                    if score > best_score:
                        best_score = score
                        best_idx = cidx
                if best_score >= 85:
                    match = anatomy_wr.loc[[best_idx]]

        if len(match) > 0:
            m = match.iloc[0]
            for col in ['anatomy_wdom', 'anatomy_yptpa', 'anatomy_ypr_dom', 'anatomy_boy',
                        'anatomy_ypr', 'anatomy_boa', 'anatomy_bmi_wr', 'anatomy_ras_wr']:
                if col not in wr.columns:
                    wr[col] = np.nan
                wr.at[idx, col] = m[col]
            wr_anatomy_matched += 1

    print(f"  WR Anatomy matched: {wr_anatomy_matched}/{len(wr)}")

# 5h. Merge WR rec_yards from backtest_college_stats (for backtest) and prospects_final (for 2026)
cs_wr = pd.read_csv('data/backtest_college_stats.csv')
cs_wr = cs_wr[cs_wr['position'] == 'WR'][['player_name', 'draft_year', 'rec_yards', 'team_pass_attempts']].copy()
cs_wr.rename(columns={'rec_yards': 'rec_yards_cs', 'team_pass_attempts': 'team_pass_att_cs'}, inplace=True)
cs_wr = cs_wr.drop_duplicates(subset=['player_name', 'draft_year'], keep='first')
wr = wr.merge(cs_wr, on=['player_name', 'draft_year'], how='left')

# From CFBD API backtest receiving stats (2015-2019 gap fill)
cfbd_wr_path = 'data/wr_backtest_receiving_cfbd.csv'
if os.path.exists(cfbd_wr_path):
    cfbd_wr = pd.read_csv(cfbd_wr_path)
    cfbd_wr = cfbd_wr[cfbd_wr['status'] == 'found'][['player_name', 'draft_year', 'rec_yards', 'receptions', 'rec_tds', 'team_pass_att']].copy()
    cfbd_wr.rename(columns={'rec_yards': 'rec_yards_cfbd', 'receptions': 'receptions_cfbd',
                            'rec_tds': 'rec_tds_cfbd', 'team_pass_att': 'team_pass_att_cfbd'}, inplace=True)
    cfbd_wr = cfbd_wr.drop_duplicates(subset=['player_name', 'draft_year'], keep='first')
    wr = wr.merge(cfbd_wr, on=['player_name', 'draft_year'], how='left')

# From prospects_final for 2026
pf_wr = prospects_final[prospects_final['position'] == 'WR'][['player_name', 'rec_yards', 'team_pass_attempts']].copy()
pf_wr.rename(columns={'rec_yards': 'rec_yards_pf', 'team_pass_attempts': 'team_pass_att_pf'}, inplace=True)
pf_wr = pf_wr.drop_duplicates(subset=['player_name'], keep='first')
wr = wr.merge(pf_wr, on='player_name', how='left')

# From WR cards for 2026
wr_cards_rec = wr_cards[['player_name', 'rec_yards', 'team_pass_attempts']].copy()
wr_cards_rec.rename(columns={'rec_yards': 'rec_yards_wc', 'team_pass_attempts': 'team_pass_att_wc'}, inplace=True)
wr = wr.merge(wr_cards_rec, on='player_name', how='left')

# Fill rec_yards: backbone → college_stats → CFBD → prospects_final → cards
if 'rec_yards' not in wr.columns:
    wr['rec_yards'] = np.nan
wr['rec_yards'] = wr['rec_yards'].fillna(wr.get('rec_yards_cs', pd.Series(dtype=float)))
wr['rec_yards'] = wr['rec_yards'].fillna(wr.get('rec_yards_cfbd', pd.Series(dtype=float)))
wr['rec_yards'] = wr['rec_yards'].fillna(wr.get('rec_yards_pf', pd.Series(dtype=float)))
wr['rec_yards'] = wr['rec_yards'].fillna(wr.get('rec_yards_wc', pd.Series(dtype=float)))

# Fill team_pass_att
if 'team_pass_att' not in wr.columns:
    wr['team_pass_att'] = np.nan
wr['team_pass_att'] = wr['team_pass_att'].fillna(wr.get('team_pass_att_cs', pd.Series(dtype=float)))
wr['team_pass_att'] = wr['team_pass_att'].fillna(wr.get('team_pass_att_cfbd', pd.Series(dtype=float)))
wr['team_pass_att'] = wr['team_pass_att'].fillna(wr.get('team_pass_att_pf', pd.Series(dtype=float)))
wr['team_pass_att'] = wr['team_pass_att'].fillna(wr.get('team_pass_att_wc', pd.Series(dtype=float)))
print(f"  Merged WR rec_yards: {wr['rec_yards'].notna().sum()}, team_pass_att: {wr['team_pass_att'].notna().sum()}")

# 5i. Fill in consolidated columns
# receptions: from CFBD, PFF, or cards
if 'receptions' not in wr.columns:
    wr['receptions'] = np.nan
wr['receptions'] = wr['receptions'].fillna(wr.get('receptions_cfbd', pd.Series(dtype=float)))
wr['receptions'] = wr['receptions'].fillna(wr.get('receptions_card', pd.Series(dtype=float)))
wr['receptions'] = wr['receptions'].fillna(wr.get('pff_receptions', pd.Series(dtype=float)))

# rec_tds: from CFBD or cards
if 'rec_tds' not in wr.columns:
    wr['rec_tds'] = np.nan
wr['rec_tds'] = wr['rec_tds'].fillna(wr.get('rec_tds_cfbd', pd.Series(dtype=float)))
wr['rec_tds'] = wr['rec_tds'].fillna(wr.get('rec_tds_card', pd.Series(dtype=float)))

# rush_tds
if 'rush_tds' not in wr.columns:
    wr['rush_tds'] = np.nan
wr['rush_tds'] = wr['rush_tds'].fillna(wr.get('rush_tds_card', pd.Series(dtype=float)))
wr['rush_tds'] = wr['rush_tds'].fillna(wr.get('rush_touchdowns_bt', pd.Series(dtype=float)))

# games_played
if 'games_played' not in wr.columns:
    wr['games_played'] = np.nan
wr['games_played'] = wr['games_played'].fillna(wr.get('games_played_card', pd.Series(dtype=float)))

# rush_attempts consolidation
if 'rush_attempts' not in wr.columns:
    wr['rush_attempts'] = np.nan
wr['rush_attempts'] = wr['rush_attempts'].fillna(wr.get('rush_attempts_card', pd.Series(dtype=float)))

# rush_yards from cards for 2026
if 'rush_yards_card' in wr.columns:
    mask_2026 = wr['dataset'] == '2026_prospect'
    wr.loc[mask_2026, 'rush_yards'] = wr.loc[mask_2026, 'rush_yards'].fillna(wr.loc[mask_2026, 'rush_yards_card'])

# dominator_rating
if 'dominator_rating' not in wr.columns:
    wr['dominator_rating'] = np.nan

# rec_yards_share, rec_td_share
if 'rec_yards_share' not in wr.columns:
    wr['rec_yards_share'] = np.nan
if 'rec_td_share' not in wr.columns:
    wr['rec_td_share'] = np.nan

# yards_per_reception
wr['yards_per_reception'] = np.where(
    (wr['rec_yards'].notna()) & (wr['receptions'].notna()) & (wr['receptions'] > 0),
    wr['rec_yards'] / wr['receptions'],
    np.nan
)

# pff_yards_per_route_run alias
wr['yards_per_route_run'] = wr.get('pff_yprr', pd.Series(dtype=float))

print(f"\n  WR enriched: {wr.shape}")


# ─────────────────────────────────────────────────────────
# STEP 6: Enrich RB data
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Enriching RB data")
print("=" * 70)

# 6a. RB backtest extras (receptions, age, cfbd_name)
rb_bt_extra = rb_bt[['player_name', 'draft_year', 'age', 'receptions']].copy()
rb_bt_extra.rename(columns={'age': 'draft_age', 'receptions': 'receptions_bt'}, inplace=True)
rb = rb.merge(rb_bt_extra, on=['player_name', 'draft_year'], how='left')
print(f"  Merged RB backtest extras: draft_age, receptions")

# 6b. RB BDR details (the bdr column is already in backbone, but let's get any extras)
rb_bdr_extra = rb_bdr[['player_name', 'draft_year']].copy()
# BDR already in backbone

# 6c. RB PFF grades
rb_pff_cols = rb_pff[['player_name', 'draft_year', 'elusive_rating', 'yards', 'attempts',
                       'yco_attempt', 'grades_run', 'grades_offense']].copy()
rb_pff_cols.rename(columns={
    'elusive_rating': 'pff_elusive_rating',
    'yards': 'pff_rush_yards',
    'attempts': 'pff_rush_attempts',
    'yco_attempt': 'pff_yco_attempt',
    'grades_run': 'pff_rushing_grade',
    'grades_offense': 'pff_overall_grade',
}, inplace=True)
rb_pff_cols = rb_pff_cols.drop_duplicates(subset=['player_name', 'draft_year'], keep='first')
rb = rb.merge(rb_pff_cols, on=['player_name', 'draft_year'], how='left')
print(f"  Merged RB PFF: {rb_pff_cols.shape[1]-2} columns")

# 6d. RB RAS official
rb = rb.merge(ras_rb[['player_name', 'draft_year', 'ras_official']],
              on=['player_name', 'draft_year'], how='left')
rb.rename(columns={'ras_official': 'ras_score'}, inplace=True)
print(f"  Merged RB RAS official")

# 6e. RB 2026 prospect production from cards
rb_cards_prod = rb_cards[['player_name', 'rush_attempts', 'rush_yards', 'rush_tds',
                           'receptions', 'rec_tds', 'total_touchdowns', 'games_played',
                           'team_rec_yards', 'team_rec_tds', 'rec_yards_share', 'rec_td_share',
                           'dominator_rating', 'rec_yards_per_team_pass_att']].copy()
rb_cards_prod.rename(columns={
    'rush_attempts': 'rush_attempts_card',
    'rush_yards': 'rush_yards_card',
    'rush_tds': 'rush_tds_card',
    'receptions': 'receptions_card',
    'rec_tds': 'rec_tds_card',
    'games_played': 'games_played_card',
}, inplace=True)
rb = rb.merge(rb_cards_prod, on='player_name', how='left')
print(f"  Merged RB 2026 card production")

# 6f. RB Anatomy data
if len(anatomy_rb) > 0:
    anatomy_rb['name_norm'] = anatomy_rb['player_name'].apply(normalize_name).str.lower()

    rb_anatomy_matched = 0
    for idx, row in rb.iterrows():
        name = normalize_name(row['player_name']).lower()
        yr = row['draft_year']

        match = anatomy_rb[(anatomy_rb['name_norm'] == name) & (anatomy_rb['anatomy_year'] == yr)]

        if len(match) == 0:
            year_candidates = anatomy_rb[anatomy_rb['anatomy_year'] == yr]
            if len(year_candidates) > 0:
                best_score = 0
                best_idx = None
                for cidx, cand in year_candidates.iterrows():
                    score = fuzz.ratio(name, cand['name_norm'])
                    if score > best_score:
                        best_score = score
                        best_idx = cidx
                if best_score >= 85:
                    match = anatomy_rb.loc[[best_idx]]

        if len(match) > 0:
            m = match.iloc[0]
            for col in ['anatomy_speed_score', 'anatomy_burst_score', 'anatomy_bdr',
                        'anatomy_peak_yardage', 'anatomy_peak_rec',
                        'anatomy_college_ypa', 'anatomy_ypa_dom', 'anatomy_bmi_rb', 'anatomy_ras_rb']:
                if col not in rb.columns:
                    rb[col] = np.nan
                if col in m.index:
                    rb.at[idx, col] = m[col]
            rb_anatomy_matched += 1

    print(f"  RB Anatomy matched: {rb_anatomy_matched}/{len(rb)}")

# 6g. Fill consolidated columns
# receptions
if 'receptions' not in rb.columns:
    rb['receptions'] = np.nan
rb['receptions'] = rb['receptions'].fillna(rb.get('receptions_bt', pd.Series(dtype=float)))
rb['receptions'] = rb['receptions'].fillna(rb.get('receptions_card', pd.Series(dtype=float)))

# rush_yards, rush_attempts, rush_tds from cards for 2026
for col, card_col in [('rush_yards', 'rush_yards_card'), ('rush_attempts', 'rush_attempts_card'),
                       ('rush_tds', 'rush_tds_card')]:
    if col not in rb.columns:
        rb[col] = np.nan
    if card_col in rb.columns:
        rb[col] = rb[col].fillna(rb[card_col])

# rec_tds
if 'rec_tds' not in rb.columns:
    rb['rec_tds'] = np.nan
rb['rec_tds'] = rb['rec_tds'].fillna(rb.get('rec_tds_card', pd.Series(dtype=float)))

# games_played
if 'games_played' not in rb.columns:
    rb['games_played'] = np.nan
rb['games_played'] = rb['games_played'].fillna(rb.get('games_played_card', pd.Series(dtype=float)))

# yards_per_carry
rb['yards_per_carry'] = np.where(
    (rb['rush_yards'].notna()) & (rb['rush_attempts'].notna()) & (rb['rush_attempts'] > 0),
    rb['rush_yards'] / rb['rush_attempts'],
    np.nan
)

# yards_per_reception
rb['yards_per_reception'] = np.where(
    (rb['rec_yards'].notna()) & (rb['receptions'].notna()) & (rb['receptions'] > 0),
    rb['rec_yards'] / rb['receptions'],
    np.nan
)

# rec_yards_per_team_pass_att
if 'rec_yards_per_team_pass_att' not in rb.columns:
    rb['rec_yards_per_team_pass_att'] = np.where(
        (rb['rec_yards'].notna()) & (rb['team_pass_att'].notna()) & (rb['team_pass_att'] > 0),
        rb['rec_yards'] / rb['team_pass_att'],
        np.nan
    )

# dominator_rating
if 'dominator_rating' not in rb.columns:
    rb['dominator_rating'] = np.nan

# rec_yards_share, rec_td_share
if 'rec_yards_share' not in rb.columns:
    rb['rec_yards_share'] = np.nan
if 'rec_td_share' not in rb.columns:
    rb['rec_td_share'] = np.nan

print(f"\n  RB enriched: {rb.shape}")


# ─────────────────────────────────────────────────────────
# STEP 7: Enrich TE data
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: Enriching TE data")
print("=" * 70)

# 7a. TE backtest extras (PFF grades, college production, draft_age)
te_bt_extra = te_bt[['player_name', 'draft_year', 'draft_age', 'early_declare', 'seasons_played',
                      'cfbd_receptions', 'cfbd_rush_yards', 'cfbd_team_rec_yards',
                      'pff_grades_offense', 'pff_grades_pass_route', 'pff_yprr',
                      'pff_targets', 'pff_receptions', 'pff_yards',
                      'pff_avg_depth_of_target', 'pff_yards_after_catch',
                      'pff_yards_after_catch_per_reception', 'pff_slot_rate',
                      'pff_inline_rate', 'pff_wide_rate', 'pff_caught_percent',
                      'pff_drop_rate', 'pff_contested_catch_rate']].copy()
te_bt_extra.rename(columns={
    'pff_grades_offense': 'pff_overall_grade',
    'pff_grades_pass_route': 'pff_receiving_grade',
    'pff_yprr': 'pff_yprr',
    'cfbd_receptions': 'receptions_bt',
    'cfbd_rush_yards': 'rush_yards_bt',
    'cfbd_team_rec_yards': 'team_rec_yards_bt',
    'early_declare': 'early_declare_te',
}, inplace=True)
te = te.merge(te_bt_extra, on=['player_name', 'draft_year'], how='left', suffixes=('', '_bt'))
# If draft_age existed in both, coalesce
if 'draft_age_bt' in te.columns:
    te['draft_age'] = te['draft_age'].fillna(te['draft_age_bt'])
    te.drop(columns=['draft_age_bt'], inplace=True)
print(f"  Merged TE backtest extras: PFF grades, draft_age, college production")

# 7b. TE 2026 extras
te_2026_extra = te_2026[['player_name', 'draft_age', 'birthdate', 'cfbd_receptions', 'cfbd_rec_yards',
                          'cfbd_rush_yards', 'cfbd_team_pass_att',
                          'pff_receptions', 'pff_yards', 'pff_grades_offense', 'pff_grades_pass_route',
                          'pff_yprr']].copy()
te_2026_extra.rename(columns={
    'draft_age': 'draft_age_2026',
    'birthdate': 'birthdate_2026',
    'cfbd_receptions': 'receptions_2026',
    'cfbd_rec_yards': 'rec_yards_2026',
    'cfbd_rush_yards': 'rush_yards_2026',
    'cfbd_team_pass_att': 'team_pass_att_2026',
    'pff_receptions': 'pff_receptions_2026',
    'pff_yards': 'pff_yards_2026',
    'pff_grades_offense': 'pff_overall_grade_2026',
    'pff_grades_pass_route': 'pff_receiving_grade_2026',
    'pff_yprr': 'pff_yprr_2026',
}, inplace=True)
te = te.merge(te_2026_extra, on='player_name', how='left')

# Fill TE columns from backtest and 2026 sources
te['draft_age'] = te['draft_age'].fillna(te.get('draft_age_2026', pd.Series(dtype=float)))
if 'birthdate' not in te.columns:
    te['birthdate'] = pd.Series([None] * len(te), dtype='object')
else:
    te['birthdate'] = te['birthdate'].astype('object')
te['birthdate'] = te['birthdate'].fillna(te.get('birthdate_2026', pd.Series(dtype='object')))

# Receptions
te['receptions'] = te.get('receptions_bt', pd.Series(dtype=float))
te['receptions'] = te['receptions'].fillna(te.get('receptions_2026', pd.Series(dtype=float)))

# Rush yards
te['rush_yards'] = te.get('rush_yards_bt', pd.Series(dtype=float))
te['rush_yards'] = te['rush_yards'].fillna(te.get('rush_yards_2026', pd.Series(dtype=float)))

# PFF grades consolidation
te['pff_overall_grade'] = te['pff_overall_grade'].fillna(te.get('pff_overall_grade_2026', pd.Series(dtype=float)))
te['pff_receiving_grade'] = te['pff_receiving_grade'].fillna(te.get('pff_receiving_grade_2026', pd.Series(dtype=float)))
if 'pff_yprr' not in te.columns:
    te['pff_yprr'] = np.nan
te['pff_yprr'] = te['pff_yprr'].fillna(te.get('pff_yprr_2026', pd.Series(dtype=float)))

print(f"  Merged TE 2026 extras")

# 7c. TE RAS official
te = te.merge(ras_te[['player_name', 'draft_year', 'ras_official']],
              on=['player_name', 'draft_year'], how='left')
# TE already has ras_score from backbone — use official as supplementary
if 'ras_official' in te.columns:
    te['ras_score'] = te['ras_score'].fillna(te['ras_official'])
print(f"  Merged TE RAS official")

# 7d. TE 2026 prospect production from cards
te_cards_prod = te_cards[['player_name', 'rush_attempts', 'rush_yards', 'rush_tds',
                           'receptions', 'rec_tds', 'total_touchdowns', 'games_played',
                           'team_rec_yards', 'team_rec_tds', 'rec_yards_share', 'rec_td_share',
                           'dominator_rating', 'rec_yards_per_team_pass_att']].copy()
te_cards_prod.rename(columns={
    'rush_attempts': 'rush_attempts_card',
    'rush_yards': 'rush_yards_card',
    'rush_tds': 'rush_tds_card',
    'receptions': 'receptions_card',
    'rec_tds': 'rec_tds_card',
    'games_played': 'games_played_card',
}, inplace=True)
te = te.merge(te_cards_prod, on='player_name', how='left')

# Fill consolidated TE columns
te['receptions'] = te['receptions'].fillna(te.get('receptions_card', pd.Series(dtype=float)))
te['rush_yards'] = te['rush_yards'].fillna(te.get('rush_yards_card', pd.Series(dtype=float)))
if 'rush_tds' not in te.columns:
    te['rush_tds'] = np.nan
te['rush_tds'] = te['rush_tds'].fillna(te.get('rush_tds_card', pd.Series(dtype=float)))
if 'rec_tds' not in te.columns:
    te['rec_tds'] = np.nan
te['rec_tds'] = te['rec_tds'].fillna(te.get('rec_tds_card', pd.Series(dtype=float)))
if 'games_played' not in te.columns:
    te['games_played'] = np.nan
te['games_played'] = te['games_played'].fillna(te.get('games_played_card', pd.Series(dtype=float)))
if 'rush_attempts' not in te.columns:
    te['rush_attempts'] = np.nan
te['rush_attempts'] = te['rush_attempts'].fillna(te.get('rush_attempts_card', pd.Series(dtype=float)))
if 'dominator_rating' not in te.columns:
    te['dominator_rating'] = np.nan
te['dominator_rating'] = te['dominator_rating'].fillna(te.get('dominator_rating', pd.Series(dtype=float)))
if 'rec_yards_share' not in te.columns:
    te['rec_yards_share'] = np.nan
if 'rec_td_share' not in te.columns:
    te['rec_td_share'] = np.nan

# yards_per_reception
te['yards_per_reception'] = np.where(
    (te['rec_yards'].notna()) & (te['receptions'].notna()) & (te['receptions'] > 0),
    te['rec_yards'] / te['receptions'],
    np.nan
)

# pff_yards_per_route_run alias
te['yards_per_route_run'] = te.get('pff_yprr', pd.Series(dtype=float))

print(f"\n  TE enriched: {te.shape}")


# ─────────────────────────────────────────────────────────
# STEP 7b: Merge nflverse combine data for backtest players
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7b: Merging nflverse combine data for backtest players")
print("=" * 70)

# Prepare nflverse combine data
combine_nfl_prep = combine_nfl.copy()
combine_nfl_prep = combine_nfl_prep.rename(columns={
    'player_name': 'comb_name',
    'ht': 'comb_height_in',
    'wt': 'comb_weight',
    'forty': 'comb_forty',
    'bench': 'comb_bench',
    'vertical': 'comb_vertical',
    'broad_jump': 'comb_broad_jump',
    'cone': 'comb_cone',
    'shuttle': 'comb_shuttle',
    'pos': 'comb_pos',
})
# Convert height from decimal (6.01 = 6'1") to inches
def height_to_inches(h):
    if pd.isna(h):
        return np.nan
    h_str = str(h)
    # Handle "5-9" format
    if '-' in h_str:
        parts = h_str.split('-')
        try:
            return int(parts[0]) * 12 + int(parts[1])
        except (ValueError, IndexError):
            return np.nan
    # Handle decimal format like 6.01 (6 feet 1 inch) or 5.09 (5 feet 9 inches)
    try:
        h_float = float(h_str)
        feet = int(h_float)
        inches_part = round((h_float - feet) * 100)
        return feet * 12 + inches_part
    except ValueError:
        return np.nan

combine_nfl_prep['comb_height_in'] = combine_nfl_prep['comb_height_in'].apply(height_to_inches)
combine_nfl_prep = combine_nfl_prep.drop_duplicates(subset=['comb_name', 'draft_year'], keep='first')

# Merge into each position for backtest players
for df, pos_label in [(wr, 'WR'), (rb, 'RB'), (te, 'TE')]:
    comb_pos = combine_nfl_prep[combine_nfl_prep['comb_pos'] == pos_label].copy()

    matched = 0
    for idx, row in df.iterrows():
        if row['dataset'] != 'backtest':
            continue

        name = normalize_name(row['player_name']).lower()
        yr = row['draft_year']

        # Exact match
        match = comb_pos[
            (comb_pos['comb_name'].apply(lambda x: normalize_name(x).lower()) == name) &
            (comb_pos['draft_year'] == yr)
        ]

        if len(match) == 0:
            # Fuzzy within year
            year_cands = comb_pos[comb_pos['draft_year'] == yr]
            if len(year_cands) > 0:
                best_score = 0
                best_match_row = None
                for _, cand in year_cands.iterrows():
                    score = fuzz.ratio(name, normalize_name(cand['comb_name']).lower())
                    if score > best_score:
                        best_score = score
                        best_match_row = cand
                if best_score >= 85:
                    match = pd.DataFrame([best_match_row])

        if len(match) > 0:
            m = match.iloc[0]
            # Only fill where backbone has NaN
            col_map = {
                'height_in': 'comb_height_in',
                'weight': 'comb_weight',
                'forty': 'comb_forty',
                'bench_press': 'comb_bench',
                'vertical_jump': 'comb_vertical',
                'broad_jump_inches': 'comb_broad_jump',
                'three_cone': 'comb_cone',
                'twenty_yard_shuttle': 'comb_shuttle',
            }
            for target_col, source_col in col_map.items():
                if pd.isna(df.at[idx, target_col]) and pd.notna(m.get(source_col)):
                    df.at[idx, target_col] = m[source_col]
            matched += 1

    print(f"  {pos_label} nflverse combine matched: {matched}")


# ─────────────────────────────────────────────────────────
# STEP 8: Add identification columns (conference, birthdate, draft_age)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: Adding identification columns")
print("=" * 70)

# Add conference from draft_picks
dp_conf = draft_picks[['dp_name', 'draft_year', 'position', 'conference_college']].copy()
dp_conf = dp_conf.drop_duplicates(subset=['dp_name', 'draft_year'], keep='first')

for df, pos_label in [(wr, 'WR'), (rb, 'RB'), (te, 'TE')]:
    dp_pos = dp_conf[dp_conf['position'] == pos_label].copy()

    conf_matched = 0
    if 'conference' not in df.columns:
        df['conference'] = pd.Series([None] * len(df), dtype='object')

    for idx, row in df.iterrows():
        if pd.notna(row.get('conference')):
            continue

        name = normalize_name(row['player_name']).lower()
        yr = row['draft_year']

        # Try exact match
        match = dp_pos[dp_pos['dp_name'].apply(lambda x: normalize_name(x).lower()) == name]
        match = match[match['draft_year'] == yr]

        if len(match) == 0:
            # Fuzzy
            year_cands = dp_pos[dp_pos['draft_year'] == yr]
            if len(year_cands) > 0:
                best_score = 0
                best_match = None
                for _, cand in year_cands.iterrows():
                    score = fuzz.ratio(name, normalize_name(cand['dp_name']).lower())
                    if score > best_score:
                        best_score = score
                        best_match = cand
                if best_score >= 85:
                    match = pd.DataFrame([best_match])

        if len(match) > 0:
            df.at[idx, 'conference'] = match.iloc[0]['conference_college']
            conf_matched += 1

    print(f"  {pos_label} conference matched: {conf_matched}")

# Fill conference for 2026 prospects via CFBD team->conference mapping
CFBD_CONFERENCE_MAP = {
    'Alabama': 'SEC', 'Arizona': 'Big 12', 'Arizona State': 'Big 12',
    'Arkansas': 'SEC', 'BYU': 'Big 12', 'Baylor': 'Big 12',
    'Boston College': 'ACC', 'California': 'ACC', 'Charlotte': 'American Athletic',
    'Cincinnati': 'Big 12', 'Clemson': 'ACC', 'Colorado': 'Big 12',
    'Delaware State': 'MEAC', 'Duke': 'ACC', 'East Carolina': 'American Athletic',
    'Florida': 'SEC', 'Florida International': 'Conference USA',
    'Florida State': 'ACC', 'Georgia': 'SEC', 'Georgia State': 'Sun Belt',
    'Georgia Tech': 'ACC', 'Houston': 'Big 12', 'Illinois': 'Big Ten',
    'Incarnate Word': 'Southland', 'Indiana': 'Big Ten', 'Iowa': 'Big Ten',
    'Jacksonville State': 'Conference USA', 'James Madison': 'Sun Belt',
    'John Carroll': 'Ohio', 'Kansas': 'Big 12', 'Kansas State': 'Big 12',
    'Kentucky': 'SEC', 'LSU': 'SEC', 'Louisiana-Lafayette': 'Sun Belt',
    'Louisville': 'ACC', 'Marshall': 'Sun Belt', 'Maryland': 'Big Ten',
    'McNeese State': 'Southland', 'Miami (FL)': 'ACC', 'Miami (OH)': 'Mid-American',
    'Michigan': 'Big Ten', 'Michigan State': 'Big Ten', 'Minnesota': 'Big Ten',
    'Mississippi': 'SEC', 'Mississippi State': 'SEC', 'Missouri': 'SEC',
    'Montana': 'Big Sky', 'NC State': 'ACC', 'Navy': 'American Athletic',
    'Nebraska': 'Big Ten', 'New Mexico State': 'Conference USA',
    'North Carolina': 'ACC', 'North Dakota State': 'MVFC',
    'Northwestern': 'Big Ten', 'Notre Dame': 'FBS Independents',
    'Ohio State': 'Big Ten', 'Oklahoma': 'SEC', 'Oklahoma State': 'Big 12',
    'Oregon': 'Big Ten', 'Penn State': 'Big Ten', 'Pittsburgh': 'ACC',
    'SMU': 'ACC', 'Sacramento State': 'Big Sky', 'Sam Houston State': 'Conference USA',
    'San Diego State': 'Mountain West', 'South Alabama': 'Sun Belt',
    'South Carolina': 'SEC', 'Stanford': 'ACC', 'Stephen F. Austin': 'Southland',
    'Syracuse': 'ACC', 'TCU': 'Big 12', 'Tarleton State': 'UAC',
    'Tennessee': 'SEC', 'Texas': 'SEC', 'Texas A&M': 'SEC',
    'Texas Tech': 'Big 12', 'Toledo': 'Mid-American', 'Troy': 'Sun Belt',
    'Tulane': 'American Athletic', 'Tulsa': 'American Athletic',
    'UAB': 'American Athletic', 'UC Davis': 'Big Sky', 'UCF': 'Big 12',
    'UCLA': 'Big Ten', 'UConn': 'FBS Independents', 'UNLV': 'Mountain West',
    'USC': 'Big Ten', 'UT Martin': 'Big South-OVC', 'UTSA': 'American Athletic',
    'Utah': 'Big 12', 'Vanderbilt': 'SEC', 'Virginia': 'ACC',
    'Virginia Tech': 'ACC', 'Virginia Union': 'CIAA',
    'Wake Forest': 'ACC', 'Washington': 'Big Ten',
    'West Alabama': 'Gulf South', 'West Virginia': 'Big 12',
    'Western Michigan': 'Mid-American', 'Wisconsin': 'Big Ten',
    'Wyoming': 'Mountain West', 'Youngstown State': 'MVFC',
}

for df, pos_label in [(wr, 'WR'), (rb, 'RB'), (te, 'TE')]:
    conf_filled = 0
    for idx, row in df.iterrows():
        if pd.notna(row.get('conference')):
            continue
        college = row.get('college')
        if pd.notna(college) and college in CFBD_CONFERENCE_MAP:
            df.at[idx, 'conference'] = CFBD_CONFERENCE_MAP[college]
            conf_filled += 1
    if conf_filled > 0:
        print(f"  {pos_label} 2026 conference filled via CFBD map: {conf_filled}")

# Add birthdates
# For backtest: from nflverse birthdates
for df, pos_label in [(wr, 'WR'), (rb, 'RB'), (te, 'TE')]:
    if 'birthdate' not in df.columns:
        df['birthdate'] = pd.Series([None] * len(df), dtype='object')
    else:
        df['birthdate'] = df['birthdate'].astype('object')

    bd_pos = bd_backtest[bd_backtest['position'] == pos_label].copy()

    bd_matched = 0
    for idx, row in df.iterrows():
        if pd.notna(row.get('birthdate')):
            continue

        name = normalize_name(row['player_name']).lower()
        yr = row['draft_year']

        # Try exact
        match = bd_pos[bd_pos['player_name'].apply(lambda x: normalize_name(x).lower()) == name]
        if 'draft_year' in bd_pos.columns:
            match = match[match['draft_year'] == yr]

        if len(match) == 0:
            # Fuzzy
            year_cands = bd_pos
            if 'draft_year' in bd_pos.columns:
                year_cands = bd_pos[bd_pos['draft_year'] == yr]
            if len(year_cands) > 0:
                best_score = 0
                best_match = None
                for _, cand in year_cands.iterrows():
                    score = fuzz.ratio(name, normalize_name(cand['player_name']).lower())
                    if score > best_score:
                        best_score = score
                        best_match = cand
                if best_score >= 85:
                    match = pd.DataFrame([best_match])

        if len(match) > 0:
            df.at[idx, 'birthdate'] = match.iloc[0]['birthdate']
            bd_matched += 1

    # For 2026: from prospect birthdates
    bd_2026_pos = bd_2026[bd_2026['position'] == pos_label].copy()
    bd_2026_matched = 0
    for idx, row in df.iterrows():
        if pd.notna(row.get('birthdate')):
            continue
        if row['dataset'] != '2026_prospect':
            continue

        name = normalize_name(row['player_name']).lower()

        match = bd_2026_pos[bd_2026_pos['player_name'].apply(lambda x: normalize_name(x).lower()) == name]

        if len(match) == 0 and len(bd_2026_pos) > 0:
            best_score = 0
            best_match = None
            for _, cand in bd_2026_pos.iterrows():
                score = fuzz.ratio(name, normalize_name(cand['player_name']).lower())
                if score > best_score:
                    best_score = score
                    best_match = cand
            if best_score >= 85:
                match = pd.DataFrame([best_match])

        if len(match) > 0:
            df.at[idx, 'birthdate'] = match.iloc[0]['birthdate']
            bd_2026_matched += 1

    # Also check prospects_final for WR/RB birthdates
    if pos_label in ['WR', 'RB']:
        pf_pos = prospects_final[prospects_final['position'] == pos_label]
        for idx, row in df.iterrows():
            if pd.notna(row.get('birthdate')):
                continue
            if row['dataset'] != '2026_prospect':
                continue
            name = normalize_name(row['player_name']).lower()
            match = pf_pos[pf_pos['player_name'].apply(lambda x: normalize_name(x).lower()) == name]
            if len(match) > 0 and pd.notna(match.iloc[0].get('birthdate')):
                df.at[idx, 'birthdate'] = match.iloc[0]['birthdate']
                bd_2026_matched += 1

    print(f"  {pos_label} birthdates: {bd_matched} backtest + {bd_2026_matched} 2026")

# draft_age: calculate from birthdate where missing
for df in [wr, rb, te]:
    if 'draft_age' not in df.columns:
        df['draft_age'] = np.nan

    for idx, row in df.iterrows():
        if pd.notna(row.get('draft_age')):
            continue
        if pd.notna(row.get('birthdate')) and pd.notna(row.get('draft_year')):
            try:
                bd = pd.to_datetime(row['birthdate'])
                # Approximate: draft is typically late April
                draft_date = pd.Timestamp(year=int(row['draft_year']), month=4, day=25)
                age = (draft_date - bd).days / 365.25
                df.at[idx, 'draft_age'] = round(age, 1)
            except:
                pass

# BMI calculation: weight / (height_in)^2 * 703
for df in [wr, rb, te]:
    df['bmi'] = np.where(
        (df['height_in'].notna()) & (df['weight'].notna()) & (df['height_in'] > 0),
        df['weight'] / (df['height_in'] ** 2) * 703,
        np.nan
    )


# ─────────────────────────────────────────────────────────
# STEP 9: Fill NFL outcomes gaps
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 9: Filling NFL outcome gaps")
print("=" * 70)

outcomes_extra = outcomes[['player_name', 'draft_year', 'position', 'seasons_over_10ppg', 'nfl_games']].copy()
outcomes_extra.rename(columns={'seasons_over_10ppg': 'nfl_seasons_10ppg'}, inplace=True)

for df, pos_label in [(wr, 'WR'), (rb, 'RB'), (te, 'TE')]:
    oc_pos = outcomes_extra[outcomes_extra['position'] == pos_label]

    if 'nfl_seasons_10ppg' not in df.columns:
        df['nfl_seasons_10ppg'] = np.nan
    if 'nfl_games' not in df.columns:
        df['nfl_games'] = np.nan

    matched = 0
    for idx, row in df.iterrows():
        if row['dataset'] == '2026_prospect':
            continue

        name = normalize_name(row['player_name']).lower()
        yr = row['draft_year']

        match = oc_pos[
            (oc_pos['player_name'].apply(lambda x: normalize_name(x).lower()) == name) &
            (oc_pos['draft_year'] == yr)
        ]

        if len(match) > 0:
            m = match.iloc[0]
            if pd.isna(df.at[idx, 'nfl_seasons_10ppg']) and pd.notna(m['nfl_seasons_10ppg']):
                df.at[idx, 'nfl_seasons_10ppg'] = m['nfl_seasons_10ppg']
                matched += 1
            if pd.isna(df.at[idx, 'nfl_games']) and pd.notna(m['nfl_games']):
                df.at[idx, 'nfl_games'] = m['nfl_games']

    print(f"  {pos_label} outcomes gaps filled: {matched}")


# ─────────────────────────────────────────────────────────
# STEP 10: Build unified master
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 10: Building unified master database")
print("=" * 70)

# Define the canonical column order
# We'll build a unified column list, then select/reorder

# Define all target columns in order
id_cols = ['player_name', 'position', 'college', 'conference', 'draft_year', 'draft_pick', 'draft_round', 'dataset']
age_cols = ['birthdate', 'draft_age']
slap_cols = ['slap_display_score', 'slap_model_score', 'dc_score', 'prospect_profile',
             'breakout_score', 'production_score', 'speed_score',
             'te_breakout_score', 'te_production_score', 'ras_score',
             'teammate_score', 'early_declare_score']
combine_measure_cols = ['height_in', 'weight', 'hand_size', 'arm_length', 'wingspan', 'bmi']
combine_drill_cols = ['forty', 'ten_yard_split', 'vertical_jump', 'broad_jump_inches',
                      'bench_press', 'three_cone', 'twenty_yard_shuttle']
athletic_cols = ['ras_unofficial']
college_rush_cols = ['rush_yards', 'rush_attempts', 'rush_tds', 'yards_per_carry', 'games_played']
college_rec_cols = ['rec_yards', 'receptions', 'rec_tds', 'yards_per_reception']
college_market_cols = ['dominator_rating', 'rec_yards_share', 'rec_td_share', 'peak_dominator',
                       'breakout_age', 'breakout_season', 'rec_yards_per_team_pass_att', 'team_pass_att', 'bdr']
anatomy_wr_cols = ['anatomy_wdom', 'anatomy_yptpa', 'anatomy_ypr_dom', 'anatomy_boy', 'anatomy_ypr', 'anatomy_boa']
anatomy_rb_cols = ['anatomy_peak_yardage', 'anatomy_peak_rec', 'anatomy_burst_score',
                   'anatomy_college_ypa', 'anatomy_ypa_dom']
pff_cols = ['pff_overall_grade', 'pff_receiving_grade', 'pff_rushing_grade', 'pff_yprr',
            'yards_per_route_run', 'pff_elusive_rating']
nfl_outcome_cols = ['nfl_best_ppr', 'nfl_best_ppg', 'nfl_career_ppg', 'nfl_first_3yr_ppg',
                    'nfl_hit24', 'nfl_hit12', 'nfl_seasons_10ppg_3yr', 'nfl_seasons_10ppg',
                    'nfl_games']
extra_cols = ['teammate_count', 'total_teammate_dc', 'teammate_names',
              'college_seasons_found', 'seasons_played']

all_target_cols = (id_cols + age_cols + slap_cols + combine_measure_cols + combine_drill_cols +
                   athletic_cols + college_rush_cols + college_rec_cols + college_market_cols +
                   anatomy_wr_cols + anatomy_rb_cols + pff_cols + nfl_outcome_cols + extra_cols)

# Ensure all target columns exist in each position df
for df in [wr, rb, te]:
    for col in all_target_cols:
        if col not in df.columns:
            df[col] = np.nan

# Select target columns from each position
wr_final = wr[all_target_cols].copy()
rb_final = rb[all_target_cols].copy()
te_final = te[all_target_cols].copy()

# Stack into master
master = pd.concat([wr_final, rb_final, te_final], ignore_index=True)

# Sort: backtest first (by draft_year, pick), then 2026 prospects (by position, pick)
master['_sort_key'] = master['dataset'].map({'backtest': 0, '2026_prospect': 1})
master = master.sort_values(['_sort_key', 'position', 'draft_year', 'draft_pick'],
                            na_position='last').reset_index(drop=True)
master = master.drop(columns=['_sort_key'])

# Replace empty strings with NaN
master = master.replace('', np.nan)

# Round numeric columns
float_cols = master.select_dtypes(include=['float64', 'float32']).columns
for col in float_cols:
    if col in ['bmi', 'draft_age', 'yards_per_carry', 'yards_per_reception', 'pff_yprr',
               'yards_per_route_run', 'rec_yards_per_team_pass_att', 'pff_elusive_rating',
               'nfl_first_3yr_ppg', 'nfl_career_ppg', 'nfl_best_ppg']:
        master[col] = master[col].round(2)
    elif col in ['anatomy_college_ypa']:
        master[col] = master[col].round(1)

print(f"\n  Master database: {master.shape[0]} rows × {master.shape[1]} columns")
print(f"  Positions: {master['position'].value_counts().to_dict()}")
print(f"  Datasets: {master['dataset'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────
# STEP 11: Coverage report
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 11: Column coverage report")
print("=" * 70)

total_rows = len(master)
print(f"\n{'Column':<35} {'Non-Null':>10} {'Pct':>8}")
print("-" * 55)
for col in all_target_cols:
    non_null = master[col].notna().sum()
    pct = non_null / total_rows * 100
    marker = "" if pct > 50 else " ← LOW"
    print(f"{col:<35} {non_null:>10} {pct:>7.1f}%{marker}")


# ─────────────────────────────────────────────────────────
# STEP 12: Spot checks
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 12: Spot checks")
print("=" * 70)

spot_check_names = ["Ja'Marr Chase", "Justin Jefferson", "Saquon Barkley", "Brock Bowers", "Carnell Tate"]

for name in spot_check_names:
    rows = master[master['player_name'].str.contains(name, case=False, na=False)]
    if len(rows) == 0:
        print(f"\n  *** {name}: NOT FOUND ***")
        continue

    row = rows.iloc[0]
    print(f"\n  {row['player_name']} ({row['position']}, {row['college']}, {int(row['draft_year'])})")
    print(f"    Draft: Rd {row['draft_round']}, Pick {row['draft_pick']}")
    print(f"    SLAP: display={row['slap_display_score']}, model={row['slap_model_score']:.1f}, DC={row['dc_score']:.1f}")
    print(f"    Profile: {row['prospect_profile']}")
    print(f"    Age: {row.get('draft_age', 'N/A')}, Birthdate: {row.get('birthdate', 'N/A')}")
    print(f"    Combine: {row['height_in']}in, {row['weight']}lbs, {row['forty']} 40yd")
    print(f"    College: {row.get('rec_yards', 'N/A')} rec yds, {row.get('receptions', 'N/A')} rec")

    if row['position'] == 'WR':
        print(f"    WR-specific: breakout={row.get('breakout_score', 'N/A')}, teammate={row.get('teammate_score', 'N/A')}, early_declare={row.get('early_declare_score', 'N/A')}")
        print(f"    Anatomy: wdom={row.get('anatomy_wdom', 'N/A')}, yptpa={row.get('anatomy_yptpa', 'N/A')}, ypr={row.get('anatomy_ypr', 'N/A')}")
    elif row['position'] == 'RB':
        print(f"    RB-specific: production={row.get('production_score', 'N/A')}, speed={row.get('speed_score', 'N/A')}, bdr={row.get('bdr', 'N/A')}")
        print(f"    Anatomy: peak_yd={row.get('anatomy_peak_yardage', 'N/A')}, burst={row.get('anatomy_burst_score', 'N/A')}, ypa={row.get('anatomy_college_ypa', 'N/A')}")
    elif row['position'] == 'TE':
        print(f"    TE-specific: breakout={row.get('te_breakout_score', 'N/A')}, production={row.get('te_production_score', 'N/A')}, ras={row.get('ras_score', 'N/A')}")

    if row['dataset'] == 'backtest':
        print(f"    NFL: hit24={row.get('nfl_hit24', 'N/A')}, career_ppg={row.get('nfl_career_ppg', 'N/A')}, best_ppg={row.get('nfl_best_ppg', 'N/A')}")
        print(f"    PFF: overall={row.get('pff_overall_grade', 'N/A')}, receiving={row.get('pff_receiving_grade', 'N/A')}")


# ─────────────────────────────────────────────────────────
# STEP 13: Export
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 13: Exporting files")
print("=" * 70)

# Master database
master.to_csv('output/slap_v5_master_database.csv', index=False)
print(f"  output/slap_v5_master_database.csv: {master.shape[0]} rows × {master.shape[1]} cols")

# Position files
wr_out = master[master['position'] == 'WR'].copy()
rb_out = master[master['position'] == 'RB'].copy()
te_out = master[master['position'] == 'TE'].copy()

# Drop position-irrelevant columns for cleaner position files
wr_drop = [c for c in anatomy_rb_cols if c in wr_out.columns] + ['production_score', 'speed_score', 'te_breakout_score', 'te_production_score', 'bdr', 'yards_per_carry', 'pff_rushing_grade', 'pff_elusive_rating']
rb_drop = [c for c in anatomy_wr_cols if c in rb_out.columns] + ['breakout_score', 'te_breakout_score', 'te_production_score', 'teammate_score', 'early_declare_score', 'breakout_age', 'breakout_season', 'pff_receiving_grade']
te_drop = [c for c in anatomy_wr_cols + anatomy_rb_cols if c in te_out.columns] + ['breakout_score', 'production_score', 'speed_score', 'teammate_score', 'early_declare_score', 'breakout_season', 'bdr', 'yards_per_carry', 'pff_rushing_grade', 'pff_elusive_rating']

for col_list in [wr_drop, rb_drop, te_drop]:
    # Only drop columns that are actually all-NaN for that position
    pass

# Actually, keep all columns for maximum data — just drop truly irrelevant ones
wr_out = wr_out.drop(columns=[c for c in ['production_score', 'speed_score', 'te_breakout_score', 'te_production_score', 'pff_rushing_grade', 'pff_elusive_rating'] if c in wr_out.columns], errors='ignore')
rb_out = rb_out.drop(columns=[c for c in ['breakout_score', 'te_breakout_score', 'te_production_score', 'teammate_score', 'early_declare_score', 'pff_receiving_grade'] if c in rb_out.columns], errors='ignore')
te_out = te_out.drop(columns=[c for c in ['breakout_score', 'production_score', 'speed_score', 'teammate_score', 'early_declare_score', 'pff_rushing_grade', 'pff_elusive_rating'] if c in te_out.columns], errors='ignore')

# Drop columns that are 100% NaN in position file
for df, label in [(wr_out, 'WR'), (rb_out, 'RB'), (te_out, 'TE')]:
    all_null_cols = [c for c in df.columns if df[c].isna().all()]
    df.drop(columns=all_null_cols, inplace=True, errors='ignore')

wr_out.to_csv('output/slap_v5_wr.csv', index=False)
rb_out.to_csv('output/slap_v5_rb.csv', index=False)
te_out.to_csv('output/slap_v5_te.csv', index=False)

print(f"  output/slap_v5_wr.csv: {wr_out.shape[0]} rows × {wr_out.shape[1]} cols")
print(f"  output/slap_v5_rb.csv: {rb_out.shape[0]} rows × {rb_out.shape[1]} cols")
print(f"  output/slap_v5_te.csv: {te_out.shape[0]} rows × {te_out.shape[1]} cols")

print("\n" + "=" * 70)
print("BUILD COMPLETE")
print("=" * 70)
