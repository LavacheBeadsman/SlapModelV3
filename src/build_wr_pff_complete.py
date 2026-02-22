"""
Build complete WR PFF dataset for all drafted WRs 2016-2025.
Matches backtest WRs to PFF receiving_summary and rushing_summary files.

Draft class 2015 has no PFF file (would need 2014 college season).
"""

import pandas as pd
import numpy as np
import re
import os

os.chdir("/home/user/SlapModelV3")

# ===========================================================================
# STEP 1: File mappings
# ===========================================================================
# PFF files cover college seasons. Draft class = college_season + 1.
RECEIVING_FILE_MAP = {
    # draft_year: receiving_summary file (college season = draft_year - 1)
    2016: 'data/receiving_summary (2).csv',   # 2015 college season
    2017: 'data/receiving_summary (3).csv',   # 2016 college season
    2018: 'data/receiving_summary (4).csv',   # 2017 college season
    2019: 'data/receiving_summary (5).csv',   # 2018 college season
    2020: 'data/receiving_summary (21).csv',  # 2019 college season
    2021: 'data/receiving_summary (20).csv',  # 2020 college season
    2022: 'data/receiving_summary (19).csv',  # 2021 college season
    2023: 'data/receiving_summary (18).csv',  # 2022 college season
    2024: 'data/receiving_summary (17).csv',  # 2023 college season
    2025: 'data/receiving_summary (16).csv',  # 2024 college season
}

RUSHING_FILE_MAP = {
    2016: 'data/rushing_summary (1).csv',
    2017: 'data/rushing_summary (2).csv',
    2018: 'data/rushing_summary (3).csv',
    2019: 'data/rushing_summary (4).csv',
    2020: 'data/rushing_summary (5).csv',
    2021: 'data/rushing_summary (6).csv',
    2022: 'data/rushing_summary (7).csv',
    2023: 'data/rushing_summary (8).csv',
    2024: 'data/rushing_summary (9).csv',
    2025: 'data/rushing_summary (10).csv',
}

# ===========================================================================
# STEP 2: School name mapping (backtest → PFF)
# ===========================================================================
# Backtest uses NFLverse-style names, PFF uses ALL CAPS abbreviations
SCHOOL_TO_PFF = {
    'Alabama': 'ALABAMA',
    'Ala-Birmingham': 'UAB',
    'Arizona': 'ARIZONA',
    'Arizona St.': 'ARIZONA ST',
    'Arkansas': 'ARKANSAS',
    'Auburn': 'AUBURN',
    'Baylor': 'BAYLOR',
    'Boise St.': 'BOISE ST',
    'Boston Col.': 'BOSTON COL',
    'Bowling Green': 'BOWL GREEN',
    'BYU': 'BYU',
    'California': 'CAL',
    'Central Florida': 'UCF',
    'Charlotte': 'CHARLOTTE',
    'Cincinnati': 'CINCINNATI',
    'Clemson': 'CLEMSON',
    'Colorado': 'COLORADO',
    'Colorado St.': 'COLO STATE',
    'Connecticut': 'UCONN',
    'Duke': 'DUKE',
    'East Carolina': 'E CAROLINA',
    'Florida': 'FLORIDA',
    'Florida St.': 'FLORIDA ST',
    'Fresno St.': 'FRESNO ST',
    'Georgia': 'GEORGIA',
    'Georgia St.': 'GA STATE',
    'Georgia Tech': 'GA TECH',
    'Hawaii': 'HAWAII',
    'Houston': 'HOUSTON',
    'Illinois': 'ILLINOIS',
    'Iowa': 'IOWA',
    'Iowa St.': 'IOWA STATE',
    'Iowa State': 'IOWA STATE',
    'Kansas St.': 'KANSAS ST',
    'Kentucky': 'KENTUCKY',
    'LSU': 'LSU',
    'La-Monroe': 'LA MONROE',
    'Liberty': 'LIBERTY',
    'Louisiana Tech': 'LA TECH',
    'Louisville': 'LOUISVILLE',
    'Maryland': 'MARYLAND',
    'Massachusetts': 'UMASS',
    'Memphis': 'MEMPHIS',
    'Miami (FL)': 'MIAMI FL',
    'Michigan': 'MICHIGAN',
    'Michigan St.': 'MICH STATE',
    'Middle Tenn. St.': 'MIDDLE TN',
    'Minnesota': 'MINNESOTA',
    'Mississippi': 'OLE MISS',
    'Missouri': 'MISSOURI',
    'Nebraska': 'NEBRASKA',
    'Nevada': 'NEVADA',
    'New Mexico St.': 'NEW MEX ST',
    'North Carolina': 'N CAROLINA',
    'North Carolina St.': 'NC STATE',
    'North Texas': 'N TEXAS',
    'Northern Illinois': 'N ILLINOIS',
    'Notre Dame': 'NOTRE DAME',
    'Ohio St.': 'OHIO STATE',
    'Ohio State': 'OHIO STATE',
    'Oklahoma': 'OKLAHOMA',
    'Oklahoma St.': 'OKLA STATE',
    'Old Dominion': 'DOMINION',
    'Ole Miss': 'OLE MISS',
    'Oregon': 'OREGON',
    'Oregon St.': 'OREGON ST',
    'Penn St.': 'PENN STATE',
    'Pittsburgh': 'PITTSBURGH',
    'Purdue': 'PURDUE',
    'Rice': 'RICE',
    'Rutgers': 'RUTGERS',
    'SMU': 'SMU',
    'South Alabama': 'S ALABAMA',
    'South Carolina': 'S CAROLINA',
    'South Florida': 'USF',
    'Southern Miss': 'SO MISS',
    'Stanford': 'STANFORD',
    'TCU': 'TCU',
    'Tennessee': 'TENNESSEE',
    'Texas': 'TEXAS',
    'Texas A&M': 'TEXAS A&M',
    'Texas Tech': 'TEXAS TECH',
    'Toledo': 'TOLEDO',
    'Tulane': 'TULANE',
    'UCLA': 'UCLA',
    'USC': 'USC',
    'Utah': 'UTAH',
    'Virginia': 'VIRGINIA',
    'Virginia Tech': 'VA TECH',
    'Wake Forest': 'WAKE',
    'Washington': 'WASHINGTON',
    'Washington St.': 'WASH STATE',
    'Washington State': 'WASH STATE',
    'West Virginia': 'W VIRGINIA',
    'Western Kentucky': 'W KENTUCKY',
    'Western Michigan': 'W MICHIGAN',
    'Wisconsin': 'WISCONSIN',
    # FCS / D2 / small schools — NOT in PFF
    'Central Arkansas': None,     # FCS
    'Charleston (WV)': None,      # D2
    'East Central (OK)': None,    # D2
    'Eastern Washington': None,   # FCS
    'Grambling St.': None,        # FCS
    'Lenoir-Rhyne': None,         # D2
    'Monmouth': None,             # FCS
    'North Dakota St.': None,     # FCS
    'Northern Iowa': None,        # FCS
    'Pennsylvania': None,         # Ivy/FCS
    'Princeton': None,            # Ivy/FCS
    'Rhode Island': None,         # FCS
    'SE Missouri St.': None,      # FCS
    'Samford': None,              # FCS
    'UT Martin': None,            # FCS
    'West Alabama': None,         # D2
    'William & Mary': None,       # FCS
}

# ===========================================================================
# STEP 3: Special cases — opt-outs, position mismatches, season overrides
# ===========================================================================

# Players who opted out of their final season.
# Key = (backtest_name, draft_year), Value = fallback draft_year to use for PFF lookup
# These players' last PFF data is from the PRIOR college season.
OPT_OUT_FALLBACK = {
    ("Ja'Marr Chase", 2021): 2020,       # Opted out of 2020 season; last played 2019 → draft 2020 file
    ('Nico Collins', 2021): 2020,         # Opted out of 2020 season; last played 2019 → draft 2020 file
}

# Players dismissed/suspended whose last season is EARLIER than draft_year - 1
SEASON_OVERRIDE = {
    ('Antonio Callaway', 2018): 2017,     # Dismissed before 2017 season; last played 2016 → draft 2017 file
}

# Players PFF lists under a different position than WR.
# We search ALL positions for these players instead of filtering to WR only.
POSITION_OVERRIDES = {
    ('Curtis Samuel', 2017): 'HB',        # Listed as HB at Ohio State
    ('Lynn Bowden Jr.', 2020): 'QB',       # Listed as QB at Kentucky
    ('Travis Hunter', 2025): 'CB',         # Listed as CB at Colorado (two-way player)
}

# Known player name mismatches (backtest name → PFF name)
NAME_OVERRIDES = {
    # Legal name vs nickname
    ('Tutu Atwell', 2021): 'Chatarius Atwell',
    ('Kadarius Toney', 2021): 'Kadarius Toney',
    ('Hollywood Brown', 2019): 'Marquise Brown',
    ('Elijah Moore', 2021): 'Elijah Moore',
    ('KJ Hamler', 2020): 'K.J. Hamler',
    ('DJ Chark', 2018): 'D.J. Chark',
    ('DK Metcalf', 2019): 'D.K. Metcalf',
    ('AJ Brown', 2019): 'A.J. Brown',
    ('DJ Moore', 2018): 'D.J. Moore',
    ('JJ Arcega-Whiteside', 2019): 'J.J. Arcega-Whiteside',
    ('KJ Osborn', 2020): 'K.J. Osborn',
    ('DJ Turner', 2023): 'D.J. Turner',
    ('CeeDee Lamb', 2020): 'CeeDee Lamb',
    ("Ja'Marr Chase", 2021): "Ja'Marr Chase",
    ('DeVonta Smith', 2021): 'DeVonta Smith',
}

# ===========================================================================
# STEP 4: Name normalization
# ===========================================================================
def normalize_name(name):
    """Normalize player name for matching."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    # Lowercase
    name = name.lower()
    # Remove punctuation: apostrophes, periods, commas, hyphens kept as space
    name = name.replace("'", "").replace("'", "").replace(".", "").replace(",", "")
    # Remove suffixes
    name = re.sub(r'\s+(jr|sr|ii|iii|iv|v)\s*$', '', name)
    # Collapse whitespace
    name = ' '.join(name.split())
    return name


def match_player(backtest_name, backtest_college, draft_year, pff_df):
    """
    Try to match a backtest WR to a row in the PFF file.
    pff_df should already be filtered + have name_norm column.
    Returns (pff_row, match_method) or (None, reason).
    """
    # Get PFF team name for this college
    pff_team = SCHOOL_TO_PFF.get(backtest_college)

    # Check for name override first
    override_key = (backtest_name, draft_year)
    if override_key in NAME_OVERRIDES:
        search_name = normalize_name(NAME_OVERRIDES[override_key])
    else:
        search_name = normalize_name(backtest_name)

    # Strategy 1: Exact normalized name + exact school
    if pff_team is not None:
        matches = pff_df[
            (pff_df['name_norm'] == search_name) &
            (pff_df['team_name'] == pff_team)
        ]
        if len(matches) == 1:
            return matches.iloc[0], 'name+school'
        if len(matches) > 1:
            # Multiple matches at same school — take highest routes (most playing time)
            best = matches.sort_values('routes', ascending=False).iloc[0]
            return best, 'name+school (best routes)'

    # Strategy 2: Exact normalized name only (no school filter)
    matches = pff_df[pff_df['name_norm'] == search_name]
    if len(matches) == 1:
        return matches.iloc[0], 'name only'

    # Strategy 3: Last name + first initial + school
    name_parts = search_name.split()
    if len(name_parts) >= 2 and pff_team is not None:
        first_init = name_parts[0][0]
        last_name = name_parts[-1]
        matches = pff_df[
            (pff_df['name_norm'].str.startswith(first_init)) &
            (pff_df['name_norm'].str.endswith(last_name)) &
            (pff_df['team_name'] == pff_team)
        ]
        if len(matches) == 1:
            return matches.iloc[0], 'initial+last+school'

    # Strategy 4: Last name + school (for nickname issues)
    if len(name_parts) >= 2 and pff_team is not None:
        last_name = name_parts[-1]
        matches = pff_df[
            (pff_df['name_norm'].str.contains(last_name)) &
            (pff_df['team_name'] == pff_team)
        ]
        if len(matches) == 1:
            return matches.iloc[0], 'last+school'

    # No match found — build reason
    if pff_team is None:
        reason = f"FCS/D2 school ({backtest_college})"
    elif pff_team is not None:
        # Check if the school has ANY players in this file
        school_players = pff_df[pff_df['team_name'] == pff_team]
        if len(school_players) == 0:
            reason = f"school not in PFF file ({pff_team})"
        else:
            pff_names = school_players['player'].tolist()[:5]
            reason = f"name mismatch at {pff_team}. PFF players: {pff_names}"
    else:
        reason = "unknown"

    return None, reason


def load_pff_files(draft_year):
    """Load PFF receiving and rushing files for a draft year. Returns (recv_df, rush_df) or (None, None)."""
    if draft_year not in RECEIVING_FILE_MAP:
        return None, None
    recv = pd.read_csv(RECEIVING_FILE_MAP[draft_year])
    recv['name_norm'] = recv['player'].apply(normalize_name)
    rush = pd.read_csv(RUSHING_FILE_MAP[draft_year])
    rush['name_norm'] = rush['player'].apply(normalize_name)
    return recv, rush


# ===========================================================================
# STEP 5: Load backtest
# ===========================================================================
print("=" * 80)
print("BUILDING COMPLETE WR PFF DATASET (2016-2025)")
print("=" * 80)

wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"\nTotal WRs in backtest: {len(wr_backtest)}")
print(f"Draft years: {sorted(wr_backtest['draft_year'].unique())}")

# ===========================================================================
# STEP 6: Process each draft year
# ===========================================================================
# Pre-load all PFF files (needed for opt-out fallbacks)
pff_cache = {}
for dy in RECEIVING_FILE_MAP:
    recv, rush = load_pff_files(dy)
    pff_cache[dy] = (recv, rush)

all_merged = []
all_failures = []
match_stats = {}

def build_merged_row(wr, pff_row, method, draft_year, rush_df, note=''):
    """Build a single merged output row from a backtest WR and PFF match."""
    row = {
        'player_name': wr['player_name'],
        'position': 'WR',
        'draft_year': draft_year,
        'pick': wr['pick'],
        'round': wr['round'],
        'college': wr['college'],
        'pff_team': pff_row['team_name'],
        'match_method': method + (f' ({note})' if note else ''),
        # PFF receiving columns
        'player_game_count': pff_row['player_game_count'],
        'yprr': pff_row['yprr'],
        'yards': pff_row['yards'],
        'receptions': pff_row['receptions'],
        'targets': pff_row['targets'],
        'touchdowns': pff_row['touchdowns'],
        'routes': pff_row['routes'],
        'grades_offense': pff_row['grades_offense'],
        'grades_pass_route': pff_row['grades_pass_route'],
        'drop_rate': pff_row['drop_rate'],
        'contested_catch_rate': pff_row['contested_catch_rate'],
        'contested_targets': pff_row['contested_targets'],
        'contested_receptions': pff_row['contested_receptions'],
        'avg_depth_of_target': pff_row['avg_depth_of_target'],
        'yards_after_catch': pff_row['yards_after_catch'],
        'yards_after_catch_per_reception': pff_row['yards_after_catch_per_reception'],
        'slot_rate': pff_row['slot_rate'],
        'wide_rate': pff_row['wide_rate'],
        'caught_percent': pff_row['caught_percent'],
        'drops': pff_row['drops'],
        'first_downs': pff_row['first_downs'],
        'avoided_tackles': pff_row['avoided_tackles'],
    }

    # Match rushing data — search all positions in the rush file
    pff_team = pff_row['team_name']
    search_norm = pff_row['name_norm']
    rush_match = rush_df[
        (rush_df['name_norm'] == search_norm) &
        (rush_df['team_name'] == pff_team)
    ]
    if len(rush_match) == 0:
        rush_match = rush_df[rush_df['name_norm'] == search_norm]

    if len(rush_match) >= 1:
        rush_row = rush_match.iloc[0]
        row['rush_attempts'] = rush_row['attempts']
        row['rush_yards'] = rush_row['yards']
        row['rush_touchdowns'] = rush_row['touchdowns']
        row['rush_ypa'] = rush_row['ypa']
        row['rush_breakaway_pct'] = rush_row['breakaway_percent']
        row['rush_elusive_rating'] = rush_row['elusive_rating']
        row['rush_grades_offense'] = rush_row['grades_offense']
    else:
        row['rush_attempts'] = 0
        row['rush_yards'] = 0
        row['rush_touchdowns'] = 0
        row['rush_ypa'] = 0.0
        row['rush_breakaway_pct'] = 0.0
        row['rush_elusive_rating'] = 0.0
        row['rush_grades_offense'] = np.nan

    return row


for draft_year in sorted(wr_backtest['draft_year'].unique()):
    print(f"\n{'='*60}")
    print(f"DRAFT YEAR {draft_year}")
    print(f"{'='*60}")

    wr_year = wr_backtest[wr_backtest['draft_year'] == draft_year].copy()
    n_wrs = len(wr_year)

    # Check if PFF file exists for this year
    if draft_year not in RECEIVING_FILE_MAP:
        print(f"  NO PFF FILE for {draft_year} (would need {draft_year-1} college season)")
        for _, wr in wr_year.iterrows():
            all_failures.append({
                'player_name': wr['player_name'],
                'draft_year': draft_year,
                'pick': wr['pick'],
                'round': wr['round'],
                'college': wr['college'],
                'reason': 'no PFF file for this draft year'
            })
        match_stats[draft_year] = {'total': n_wrs, 'matched': 0, 'pct': 0.0}
        continue

    pff_recv_full, pff_rush_full = pff_cache[draft_year]
    pff_recv_wr = pff_recv_full[pff_recv_full['position'] == 'WR'].copy()
    print(f"  PFF receiving file: {RECEIVING_FILE_MAP[draft_year]} ({len(pff_recv_wr)} WRs)")

    matched = 0
    for _, wr in wr_year.iterrows():
        player_key = (wr['player_name'], draft_year)

        # -----------------------------------------------------------
        # Special case 1: Opt-out fallback (e.g., Ja'Marr Chase 2021)
        # Player opted out of their final season — use previous year's file
        # -----------------------------------------------------------
        if player_key in OPT_OUT_FALLBACK:
            fallback_yr = OPT_OUT_FALLBACK[player_key]
            fb_recv, fb_rush = pff_cache.get(fallback_yr, (None, None))
            if fb_recv is not None:
                fb_recv_wr = fb_recv[fb_recv['position'] == 'WR'].copy()
                pff_row, method = match_player(
                    wr['player_name'], wr['college'], draft_year, fb_recv_wr
                )
                if pff_row is not None:
                    matched += 1
                    row = build_merged_row(wr, pff_row, method, draft_year, fb_rush,
                                           note=f'opt-out, used {fallback_yr-1} season')
                    all_merged.append(row)
                    print(f"    OPT-OUT: {wr['player_name']} matched via {fallback_yr} file")
                    continue

        # -----------------------------------------------------------
        # Special case 2: Season override (e.g., Callaway dismissed)
        # Player's last season is earlier than draft_year - 1
        # -----------------------------------------------------------
        if player_key in SEASON_OVERRIDE:
            override_yr = SEASON_OVERRIDE[player_key]
            ov_recv, ov_rush = pff_cache.get(override_yr, (None, None))
            if ov_recv is not None:
                ov_recv_wr = ov_recv[ov_recv['position'] == 'WR'].copy()
                pff_row, method = match_player(
                    wr['player_name'], wr['college'], draft_year, ov_recv_wr
                )
                if pff_row is not None:
                    matched += 1
                    row = build_merged_row(wr, pff_row, method, draft_year, ov_rush,
                                           note=f'season override, used {override_yr-1} season')
                    all_merged.append(row)
                    print(f"    SEASON OVERRIDE: {wr['player_name']} matched via {override_yr} file")
                    continue

        # -----------------------------------------------------------
        # Special case 3: Position override (e.g., Curtis Samuel as HB)
        # Player is listed under different position in PFF
        # -----------------------------------------------------------
        if player_key in POSITION_OVERRIDES:
            alt_pos = POSITION_OVERRIDES[player_key]
            # Search in ALL positions for this player
            pff_row, method = match_player(
                wr['player_name'], wr['college'], draft_year, pff_recv_full
            )
            if pff_row is not None:
                matched += 1
                row = build_merged_row(wr, pff_row, method, draft_year, pff_rush_full,
                                       note=f'PFF position={alt_pos}')
                all_merged.append(row)
                print(f"    POSITION OVERRIDE: {wr['player_name']} found as {alt_pos}")
                continue

        # -----------------------------------------------------------
        # Normal matching: search WR-only in the standard file
        # -----------------------------------------------------------
        pff_row, method = match_player(
            wr['player_name'], wr['college'], draft_year, pff_recv_wr
        )

        if pff_row is not None:
            matched += 1
            row = build_merged_row(wr, pff_row, method, draft_year, pff_rush_full)
            all_merged.append(row)
        else:
            all_failures.append({
                'player_name': wr['player_name'],
                'draft_year': draft_year,
                'pick': wr['pick'],
                'round': wr['round'],
                'college': wr['college'],
                'reason': method  # method contains the failure reason
            })

    pct = 100 * matched / n_wrs if n_wrs > 0 else 0
    match_stats[draft_year] = {'total': n_wrs, 'matched': matched, 'pct': pct}
    print(f"  Matched: {matched}/{n_wrs} ({pct:.1f}%)")


# ===========================================================================
# STEP 7: Build final dataframe and save
# ===========================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

merged_df = pd.DataFrame(all_merged)
failures_df = pd.DataFrame(all_failures)

# Column order for output
output_cols = [
    # Backtest identifiers
    'player_name', 'position', 'draft_year', 'pick', 'round', 'college', 'pff_team', 'match_method',
    # PFF receiving
    'player_game_count', 'yprr', 'yards', 'receptions', 'targets', 'touchdowns',
    'routes', 'grades_offense', 'grades_pass_route', 'drop_rate',
    'contested_catch_rate', 'contested_targets', 'contested_receptions',
    'avg_depth_of_target', 'yards_after_catch', 'yards_after_catch_per_reception',
    'slot_rate', 'wide_rate', 'caught_percent', 'drops', 'first_downs', 'avoided_tackles',
    # PFF rushing
    'rush_attempts', 'rush_yards', 'rush_touchdowns', 'rush_ypa',
    'rush_breakaway_pct', 'rush_elusive_rating', 'rush_grades_offense',
]

merged_df = merged_df[output_cols]
merged_df = merged_df.sort_values(['draft_year', 'pick']).reset_index(drop=True)

# Save
output_path = 'data/wr_pff_all_2016_2025.csv'
merged_df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
print(f"Total rows: {len(merged_df)}")

# ===========================================================================
# STEP 8: Coverage reports
# ===========================================================================
print("\n" + "-" * 60)
print("COVERAGE BY DRAFT YEAR")
print("-" * 60)
print(f"{'Year':<8} {'Matched':<10} {'Total':<8} {'Pct':<8}")
total_matched = 0
total_all = 0
total_matched_excl_2015 = 0
total_all_excl_2015 = 0
for yr in sorted(match_stats.keys()):
    s = match_stats[yr]
    print(f"{yr:<8} {s['matched']:<10} {s['total']:<8} {s['pct']:.1f}%")
    total_matched += s['matched']
    total_all += s['total']
    if yr >= 2016:
        total_matched_excl_2015 += s['matched']
        total_all_excl_2015 += s['total']

print(f"\n{'TOTAL':<8} {total_matched:<10} {total_all:<8} {100*total_matched/total_all:.1f}%")
print(f"{'2016-25':<8} {total_matched_excl_2015:<10} {total_all_excl_2015:<8} {100*total_matched_excl_2015/total_all_excl_2015:.1f}%")

print("\n" + "-" * 60)
print("COVERAGE BY ROUND")
print("-" * 60)
for rnd in sorted(merged_df['round'].unique()):
    rnd_merged = len(merged_df[merged_df['round'] == rnd])
    rnd_total = len(wr_backtest[(wr_backtest['round'] >= 1) & (wr_backtest['round'] == rnd)])
    # Count 2015 failures separately
    rnd_2015 = len(wr_backtest[(wr_backtest['draft_year'] == 2015) & (wr_backtest['round'] == rnd)])
    rnd_total_excl = rnd_total - rnd_2015
    pct = 100 * rnd_merged / rnd_total_excl if rnd_total_excl > 0 else 0
    print(f"Round {rnd}: {rnd_merged}/{rnd_total_excl} ({pct:.1f}%) [excl 2015 class]")

# ===========================================================================
# STEP 9: All unmatched round 1-3 WRs
# ===========================================================================
print("\n" + "-" * 60)
print("UNMATCHED ROUND 1-3 WRs (ALL)")
print("-" * 60)
rd13_failures = failures_df[failures_df['round'].isin([1, 2, 3])].copy()
rd13_failures = rd13_failures.sort_values(['draft_year', 'pick'])

if len(rd13_failures) == 0:
    print("  None! All round 1-3 WRs matched.")
else:
    print(f"  Total unmatched Rd 1-3: {len(rd13_failures)}")
    print()
    for _, f in rd13_failures.iterrows():
        print(f"  Rd{f['round']} Pick {f['pick']}: {f['player_name']} ({f['college']}) "
              f"[{f['draft_year']}] — {f['reason']}")

# ===========================================================================
# STEP 10: All unmatched (for completeness)
# ===========================================================================
print("\n" + "-" * 60)
print("ALL UNMATCHED WRs (Rounds 4-7)")
print("-" * 60)
rd47_failures = failures_df[failures_df['round'].isin([4, 5, 6, 7])].copy()
rd47_failures = rd47_failures.sort_values(['draft_year', 'pick'])
for _, f in rd47_failures.iterrows():
    print(f"  Rd{f['round']} Pick {f['pick']}: {f['player_name']} ({f['college']}) "
          f"[{f['draft_year']}] — {f['reason']}")

# ===========================================================================
# STEP 11: Sample rows
# ===========================================================================
print("\n" + "-" * 60)
print("SAMPLE ROWS (first 10)")
print("-" * 60)
sample = merged_df.head(10)
for _, row in sample.iterrows():
    print(f"\n  {row['player_name']} (Rd{row['round']} Pick {row['pick']}, {row['draft_year']}) — {row['college']}")
    print(f"    PFF Team: {row['pff_team']} | Match: {row['match_method']}")
    print(f"    Games: {row['player_game_count']} | YPRR: {row['yprr']} | Grade: {row['grades_offense']} | Route Grade: {row['grades_pass_route']}")
    print(f"    Rec: {row['receptions']}/{row['targets']} tgt, {row['yards']} yds, {row['touchdowns']} TD | Drop Rate: {row['drop_rate']}")
    print(f"    Rushing: {row['rush_attempts']} att, {row['rush_yards']} yds, {row['rush_touchdowns']} TD")

# ===========================================================================
# STEP 12: Match method breakdown
# ===========================================================================
print("\n" + "-" * 60)
print("MATCH METHOD BREAKDOWN")
print("-" * 60)
print(merged_df['match_method'].value_counts().to_string())

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
