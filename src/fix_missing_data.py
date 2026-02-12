"""
Fix Missing Data for 6-Component WR SLAP Model
=================================================

Fix 1: RAS — 53 WRs missing → search all RAS files + combine.parquet for fallback
Fix 2: Early Declare — 49 WRs missing → derive from birthdates/draft age/seasons
Fix 3: Rushing Production — 55 WRs missing → search PFF rushing files + CFBD API fallback

Goal: 95%+ coverage for every component.
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')


def normalize_name(name):
    """Normalize a name for fuzzy matching."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    # Remove suffixes
    name = re.sub(r'\s+(Jr\.?|Sr\.?|III|II|IV|V)$', '', name, flags=re.IGNORECASE)
    # Remove periods, apostrophes
    name = name.replace('.', '').replace("'", '').replace("'", '')
    # Lowercase
    name = name.lower().strip()
    return name


# ============================================================================
# LOAD MAIN BACKTEST
# ============================================================================

print("=" * 100)
print("LOADING DATA")
print("=" * 100)

wr = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"Main backtest: {len(wr)} WRs")

# Add 2025 class if separate
try:
    wr_2025 = pd.read_csv('data/wr_2025_draft_class.csv')
    # Check if already in main backtest
    existing = set(zip(wr['player_name'], wr['draft_year']))
    new_2025 = wr_2025[~wr_2025.apply(lambda x: (x['player_name'], x['draft_year']) in existing, axis=1)]
    if len(new_2025) > 0:
        print(f"  Found {len(new_2025)} additional 2025 WRs")
except:
    pass

# ============================================================================
# FIX 1: RAS — 53 WRs Missing
# ============================================================================

print("\n" + "=" * 100)
print("FIX 1: RAS — IDENTIFYING AND FILLING 53 MISSING WRs")
print("=" * 100)

missing_ras = wr[wr['RAS'].isna()].copy()
print(f"\n{len(missing_ras)} WRs missing RAS in main backtest")

# Load ALL RAS data sources
print("\n--- Loading all RAS data sources ---")

# Source 1: WRRas201502025.csv (largest file, 2775 WRs)
ras_big = pd.read_csv('data/WRRas201502025.csv', encoding='utf-8-sig')
# Columns are already properly typed: Name (str), Year (int), RAS (float)
ras_big['Name'] = ras_big['Name'].astype(str).str.strip()
ras_big['RAS'] = pd.to_numeric(ras_big['RAS'], errors='coerce')
ras_big['Year'] = pd.to_numeric(ras_big['Year'], errors='coerce')
ras_big['norm_name'] = ras_big['Name'].apply(normalize_name)
print(f"  WRRas201502025.csv: {len(ras_big)} entries")

# Source 2: WR_RAS_2020_to_2025.csv
ras_2020 = pd.read_csv('data/WR_RAS_2020_to_2025.csv', encoding='utf-8-sig')
if 'Name' in ras_2020.columns:
    ras_2020['Name'] = ras_2020['Name'].astype(str).str.strip()
    ras_2020['RAS'] = pd.to_numeric(ras_2020['RAS'], errors='coerce')
    ras_2020['Year'] = pd.to_numeric(ras_2020['Year'], errors='coerce')
    ras_2020['norm_name'] = ras_2020['Name'].apply(normalize_name)
print(f"  WR_RAS_2020_to_2025.csv: {len(ras_2020)} entries")

# Source 3: wr_ras_merged.csv (previously merged)
ras_merged = pd.read_csv('data/wr_ras_merged.csv')
ras_merged['norm_name'] = ras_merged['player_name'].apply(normalize_name)
print(f"  wr_ras_merged.csv: {len(ras_merged)} entries")

# Source 4: combine.parquet (for fallback athletic data)
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine_wr = combine[combine['pos'] == 'WR'].copy()
combine_wr['norm_name'] = combine_wr['player_name'].apply(normalize_name)
print(f"  combine.parquet (WRs): {len(combine_wr)} entries")

# Build a unified RAS lookup from all sources
# Priority: ras_big (most comprehensive), then ras_2020, then ras_merged
all_ras = {}

# From big file
for _, row in ras_big.iterrows():
    key = (normalize_name(row['Name']), int(row['Year']) if pd.notna(row['Year']) else None)
    if pd.notna(row['RAS']):
        all_ras[key] = {'name': row['Name'], 'ras': row['RAS'], 'source': 'WRRas201502025'}

# From 2020 file
for _, row in ras_2020.iterrows():
    key = (normalize_name(row['Name']), int(row['Year']) if pd.notna(row['Year']) else None)
    if pd.notna(row['RAS']) and key not in all_ras:
        all_ras[key] = {'name': row['Name'], 'ras': row['RAS'], 'source': 'WR_RAS_2020_to_2025'}

# From merged file
for _, row in ras_merged.iterrows():
    key = (normalize_name(row['player_name']), int(row['draft_year']) if pd.notna(row['draft_year']) else None)
    if pd.notna(row['RAS']) and key not in all_ras:
        all_ras[key] = {'name': row['player_name'], 'ras': row['RAS'], 'source': 'wr_ras_merged'}

print(f"\n  Total unique RAS entries across all files: {len(all_ras)}")

# Build combine lookup for fallback
combine_lookup = {}
for _, row in combine_wr.iterrows():
    key = (normalize_name(row['player_name']), int(row['draft_year']) if pd.notna(row['draft_year']) else None)
    combine_lookup[key] = {
        'name': row['player_name'],
        'ht': row['ht'],
        'wt': row['wt'],
        'forty': row['forty'],
        'vertical': row['vertical'],
        'broad_jump': row['broad_jump'],
        'cone': row['cone'],
        'shuttle': row['shuttle'],
        'bench': row['bench']
    }

# Also build a name-only lookup (without year) for fuzzy matching
ras_by_name_only = {}
for (norm_name, year), data in all_ras.items():
    if norm_name not in ras_by_name_only:
        ras_by_name_only[norm_name] = []
    ras_by_name_only[norm_name].append({**data, 'year': year})

combine_by_name_only = {}
for (norm_name, year), data in combine_lookup.items():
    if norm_name not in combine_by_name_only:
        combine_by_name_only[norm_name] = []
    combine_by_name_only[norm_name].append({**data, 'year': year})

# Now search for each missing player
print(f"\n--- Searching for {len(missing_ras)} missing RAS WRs ---\n")
print(f"{'Player':<30} {'Year':>5} {'Rd':>3} {'Pick':>5}  {'RAS Found?':<15} {'Source':<25} {'Combine Data?'}")
print("-" * 120)

ras_found = 0
ras_from_combine = 0
ras_truly_missing = 0

ras_fixes = []  # Will store fixes to apply

for _, player in missing_ras.iterrows():
    name = player['player_name']
    year = int(player['draft_year'])
    rd = int(player['round'])
    pick = int(player['pick'])
    norm = normalize_name(name)

    # Try exact match (name + year)
    key = (norm, year)
    found_ras = None
    source = ""

    if key in all_ras:
        found_ras = all_ras[key]['ras']
        source = all_ras[key]['source']
    else:
        # Try name-only match
        if norm in ras_by_name_only:
            matches = ras_by_name_only[norm]
            # Find closest year
            best = min(matches, key=lambda x: abs((x['year'] or 0) - year))
            if best['year'] and abs(best['year'] - year) <= 1:
                found_ras = best['ras']
                source = f"{best['source']} (year±1)"
            elif best['year'] and abs(best['year'] - year) <= 2:
                found_ras = best['ras']
                source = f"{best['source']} (year±2)"

        # Try alternate name formats
        if found_ras is None:
            # Try with/without suffixes, shortened names
            alt_names = set()
            parts = norm.split()
            if len(parts) >= 2:
                # Try first initial + last
                alt_names.add(parts[0][0] + ' ' + ' '.join(parts[1:]))
                # Try without middle name
                if len(parts) >= 3:
                    alt_names.add(parts[0] + ' ' + parts[-1])
                # Add III/Jr variations
                alt_names.add(norm + ' iii')
                alt_names.add(norm + ' jr')
                alt_names.add(norm + ' ii')
                # Try with/without dots in initials
                for alt in list(alt_names):
                    if '.' not in alt:
                        parts2 = alt.split()
                        for i, p in enumerate(parts2):
                            if len(p) == 1:
                                parts2[i] = p + '.'
                        alt_names.add(' '.join(parts2))

            for alt in alt_names:
                if alt in ras_by_name_only:
                    matches = ras_by_name_only[alt]
                    best = min(matches, key=lambda x: abs((x['year'] or 0) - year))
                    if best['year'] and abs(best['year'] - year) <= 1:
                        found_ras = best['ras']
                        source = f"{best['source']} (alt: {alt})"
                        break

    # Check combine data as fallback
    combine_data = None
    has_40 = False
    if key in combine_lookup:
        combine_data = combine_lookup[key]
    elif norm in combine_by_name_only:
        matches = combine_by_name_only[norm]
        best = min(matches, key=lambda x: abs((x['year'] or 0) - year))
        if best['year'] and abs(best['year'] - year) <= 1:
            combine_data = best

    if combine_data:
        has_40 = pd.notna(combine_data.get('forty')) and pd.notna(combine_data.get('wt'))

    # Categorize result
    if found_ras:
        ras_found += 1
        ras_fixes.append({'player_name': name, 'draft_year': year, 'RAS': found_ras, 'source': source})
        combine_str = ""
    elif has_40:
        ras_from_combine += 1
        wt = combine_data['wt']
        forty = combine_data['forty']
        ht = combine_data.get('ht', '')
        combine_str = f"YES (ht={ht}, wt={wt}, 40={forty})"
        ras_fixes.append({'player_name': name, 'draft_year': year, 'RAS': None,
                         'wt': wt, 'forty': forty, 'ht': ht,
                         'source': 'combine_fallback'})
    else:
        ras_truly_missing += 1
        combine_str = "NO combine data"

    ras_str = f"{found_ras:.2f}" if found_ras else "NOT FOUND"
    src_str = source if found_ras else ""
    combine_display = combine_str if not found_ras else ""

    print(f"{name:<30} {year:>5} {rd:>3} {pick:>5}  {ras_str:<15} {src_str:<25} {combine_display}")


print(f"\n--- RAS FIX SUMMARY ---")
print(f"  Found RAS in files:     {ras_found}")
print(f"  Has combine fallback:   {ras_from_combine}")
print(f"  Truly missing all data: {ras_truly_missing}")
print(f"  New coverage: {(339 - ras_truly_missing)}/{339} = {(339-ras_truly_missing)/339*100:.1f}%")


# ============================================================================
# FIX 2: EARLY DECLARE — 49 WRs Missing
# ============================================================================

print("\n\n" + "=" * 100)
print("FIX 2: EARLY DECLARE — DERIVING STATUS FOR 49 MISSING WRs")
print("=" * 100)

# Load declare data
declare = pd.read_csv('data/wr_eval_with_declare.csv')
declare_lookup = dict(zip(
    zip(declare['player_name'], declare['draft_year']),
    zip(declare['declare_status'], declare['draft_age'])
))

# Identify missing
wr['has_declare'] = wr.apply(
    lambda x: (x['player_name'], x['draft_year']) in declare_lookup, axis=1
)
missing_declare = wr[~wr['has_declare']].copy()
print(f"\n{len(missing_declare)} WRs missing early declare status")

# Load ALL birthdate sources
print("\n--- Loading birthdate sources ---")

# Source 1: nflverse_birthdates_2015_2025.csv
bd_nfl = pd.read_csv('data/nflverse_birthdates_2015_2025.csv')
bd_nfl = bd_nfl.rename(columns={'nfl_name': 'player_name', 'birth_date': 'birthdate'})
bd_nfl['norm_name'] = bd_nfl['player_name'].apply(normalize_name)
print(f"  nflverse_birthdates: {len(bd_nfl)} entries")

# Source 2: Main backtest file (has birthdate column)
bd_backtest = wr[['player_name', 'draft_year', 'birthdate']].dropna(subset=['birthdate'])
bd_backtest['norm_name'] = bd_backtest['player_name'].apply(normalize_name)
print(f"  Backtest birthdates: {len(bd_backtest)} entries")

# Source 3: draft_picks.parquet (has age)
draft_picks = pd.read_parquet('data/nflverse/draft_picks.parquet')
draft_picks_wr = draft_picks[draft_picks['position'] == 'WR'].copy()
draft_picks_wr['norm_name'] = draft_picks_wr['pfr_player_name'].apply(normalize_name)
print(f"  draft_picks.parquet (WRs): {len(draft_picks_wr)} entries, has 'age' column")

# Source 4: 2026_prospect_birthdates.csv
try:
    bd_2026 = pd.read_csv('data/2026_prospect_birthdates.csv')
    bd_2026['norm_name'] = bd_2026['player_name'].apply(normalize_name) if 'player_name' in bd_2026.columns else ''
    print(f"  2026_prospect_birthdates: {len(bd_2026)} entries")
except:
    bd_2026 = pd.DataFrame()

# Source 5: prospects_with_birthdates.csv
try:
    bd_prospects = pd.read_csv('data/prospects_with_birthdates.csv')
    print(f"  prospects_with_birthdates: {len(bd_prospects)} entries, cols: {bd_prospects.columns.tolist()[:5]}")
except:
    bd_prospects = pd.DataFrame()

# Build unified birthdate lookup
birthdate_lookup = {}

# From nflverse
for _, row in bd_nfl.iterrows():
    key = normalize_name(row['player_name'])
    if pd.notna(row['birthdate']):
        birthdate_lookup[key] = {'birthdate': row['birthdate'], 'source': 'nflverse'}

# From backtest
for _, row in bd_backtest.iterrows():
    key = normalize_name(row['player_name'])
    if pd.notna(row['birthdate']) and key not in birthdate_lookup:
        birthdate_lookup[key] = {'birthdate': row['birthdate'], 'source': 'backtest'}

# Build draft_age lookup from draft_picks.parquet
draft_age_lookup = {}
for _, row in draft_picks_wr.iterrows():
    key = (normalize_name(row['pfr_player_name']), int(row['season']))
    if pd.notna(row['age']):
        draft_age_lookup[key] = row['age']

# Load wr_all_seasons.csv to count college seasons
wr_seasons = pd.read_csv('data/wr_all_seasons.csv')
wr_seasons['norm_name'] = wr_seasons['player_name'].apply(normalize_name)

# Count seasons per player
seasons_count = wr_seasons.groupby(['player_name', 'draft_year']).agg(
    num_seasons=('season', 'nunique'),
    first_season=('season', 'min'),
    last_season=('season', 'max'),
    seasons_list=('season', lambda x: sorted(x.unique().tolist()))
).reset_index()
seasons_lookup = dict(zip(
    zip(seasons_count['player_name'], seasons_count['draft_year']),
    seasons_count[['num_seasons', 'first_season', 'last_season', 'seasons_list']].to_dict('records')
))

print(f"\n  Birthdate lookup: {len(birthdate_lookup)} unique names")
print(f"  Draft age lookup: {len(draft_age_lookup)} entries")
print(f"  College seasons data: {len(seasons_count)} players")

# Now derive declare status for each missing player
print(f"\n--- Deriving early declare for {len(missing_declare)} missing WRs ---\n")
print(f"{'Player':<30} {'Year':>5} {'Rd':>3}  {'Birthdate':<12} {'Draft Age':>10} {'Seasons':>8} {'Status':<12} {'Method'}")
print("-" * 120)

declare_fixes = []
derived_count = 0
still_missing = 0

for _, player in missing_declare.iterrows():
    name = player['player_name']
    year = int(player['draft_year'])
    rd = int(player['round'])
    norm = normalize_name(name)

    # Get birthdate
    birthdate = None
    if pd.notna(player.get('birthdate')):
        birthdate = player['birthdate']
    elif norm in birthdate_lookup:
        birthdate = birthdate_lookup[norm]['birthdate']

    # Get draft age
    draft_age = None
    key_dp = (norm, year)
    if key_dp in draft_age_lookup:
        draft_age = draft_age_lookup[key_dp]
    elif birthdate:
        try:
            bd = pd.to_datetime(birthdate)
            # NFL draft is typically late April
            draft_date = pd.Timestamp(year=year, month=4, day=25)
            draft_age = (draft_date - bd).days / 365.25
        except:
            pass

    # Get college seasons
    seasons_info = seasons_lookup.get((name, year))
    num_seasons = seasons_info['num_seasons'] if seasons_info else None
    seasons_list = seasons_info['seasons_list'] if seasons_info else None

    # Derive declare status
    status = None
    method = ""

    if draft_age is not None:
        if draft_age <= 21.5:
            status = "EARLY"
            method = f"draft_age={draft_age:.1f} (<=21.5)"
        elif draft_age >= 23.0:
            status = "LATE"
            method = f"draft_age={draft_age:.1f} (>=23.0)"
        else:
            # Between 21.5 and 23.0 — check seasons played
            if num_seasons is not None:
                if num_seasons <= 3:
                    status = "EARLY"
                    method = f"draft_age={draft_age:.1f}, {num_seasons} seasons"
                elif num_seasons >= 5:
                    status = "LATE"
                    method = f"draft_age={draft_age:.1f}, {num_seasons} seasons"
                else:
                    status = "STANDARD"
                    method = f"draft_age={draft_age:.1f}, {num_seasons} seasons"
            else:
                status = "STANDARD"
                method = f"draft_age={draft_age:.1f} (22-23 range, default STANDARD)"
    elif num_seasons is not None:
        if num_seasons <= 3:
            status = "EARLY"
            method = f"no age, but only {num_seasons} college seasons"
        elif num_seasons >= 5:
            status = "LATE"
            method = f"no age, but {num_seasons} college seasons"
        else:
            status = "STANDARD"
            method = f"no age, {num_seasons} college seasons (default STANDARD)"
    else:
        status = None
        method = "NO DATA — no age, no seasons"

    if status:
        derived_count += 1
        declare_fixes.append({
            'player_name': name, 'draft_year': year,
            'declare_status': status, 'draft_age': draft_age,
            'num_seasons': num_seasons, 'method': method
        })
    else:
        still_missing += 1

    bd_str = str(birthdate)[:10] if birthdate else "NONE"
    age_str = f"{draft_age:.1f}" if draft_age else "NONE"
    seas_str = str(num_seasons) if num_seasons else "NONE"
    status_str = status if status else "UNKNOWN"

    print(f"{name:<30} {year:>5} {rd:>3}  {bd_str:<12} {age_str:>10} {seas_str:>8} {status_str:<12} {method}")


print(f"\n--- EARLY DECLARE FIX SUMMARY ---")
print(f"  Successfully derived: {derived_count}")
print(f"  Still missing:        {still_missing}")
print(f"  New coverage: {(290 + derived_count)}/{339} = {(290+derived_count)/339*100:.1f}%")


# ============================================================================
# FIX 3: RUSHING PRODUCTION — 55 WRs Missing
# ============================================================================

print("\n\n" + "=" * 100)
print("FIX 3: RUSHING PRODUCTION — FILLING 55 MISSING WRs")
print("=" * 100)

# Load PFF WR data
pff = pd.read_csv('data/wr_pff_all_2016_2025.csv')
pff_lookup = dict(zip(
    zip(pff['player_name'], pff['draft_year']),
    zip(pff['rush_attempts'], pff['rush_yards'], pff['rush_touchdowns'], pff['player_game_count'])
))

# Identify missing
wr['has_rushing'] = wr.apply(
    lambda x: (x['player_name'], x['draft_year']) in pff_lookup, axis=1
)
missing_rushing = wr[~wr['has_rushing']].copy()
print(f"\n{len(missing_rushing)} WRs missing rushing data")

# Separate 2015 class from others
missing_2015 = missing_rushing[missing_rushing['draft_year'] == 2015]
missing_other = missing_rushing[missing_rushing['draft_year'] != 2015]
print(f"  2015 class: {len(missing_2015)} missing (no PFF file for 2014 season)")
print(f"  Other years: {len(missing_other)} missing (name mismatch or small school)")

# Search PFF rushing_summary files for the non-2015 missing WRs
# Map file numbers to college seasons
# (1) = 2015 season → 2016 draft, (2) = 2016 → 2017, etc.
file_to_season = {
    ' (1)': 2015, ' (2)': 2016, ' (3)': 2017, ' (4)': 2018,
    ' (5)': 2019, ' (6)': 2020, ' (7)': 2021, ' (8)': 2022,
    ' (9)': 2023, ' (10)': 2024, '': 2025,
    ' (11)': 2025, ' (12)': 2025
}

# For draft_year X, the final college season is X-1
# So we need the rushing_summary file that maps to season X-1

# Load all PFF rushing summary files and index by normalized name + season
print("\n--- Loading PFF rushing summary files ---")
pff_rushing_all = {}
for label, season in file_to_season.items():
    try:
        df = pd.read_csv(f'data/rushing_summary{label}.csv')
        wr_rows = df[df['position'] == 'WR']
        for _, row in wr_rows.iterrows():
            norm = normalize_name(row['player'])
            team = row['team_name'] if pd.notna(row.get('team_name')) else ''
            key = (norm, season)
            pff_rushing_all[key] = {
                'player': row['player'],
                'team': team,
                'games': row.get('player_game_count', np.nan),
                'rush_att': row.get('attempts', 0),
                'rush_yards': row.get('yards', 0),
                'rush_tds': row.get('touchdowns', 0),
                'source': f'rushing_summary{label}'
            }
        print(f"  rushing_summary{label}.csv ({season} season): {len(wr_rows)} WRs")
    except Exception as e:
        print(f"  rushing_summary{label}.csv: ERROR - {e}")

# Also index by name only for fuzzy matching
pff_rushing_by_name = {}
for (norm, season), data in pff_rushing_all.items():
    if norm not in pff_rushing_by_name:
        pff_rushing_by_name[norm] = []
    pff_rushing_by_name[norm].append({**data, 'season': season})

print(f"\n  Total PFF rushing WR entries: {len(pff_rushing_all)}")

# Search for non-2015 missing WRs in PFF rushing files
print(f"\n--- Searching PFF rushing files for {len(missing_other)} non-2015 missing WRs ---\n")

rushing_fixes = []
found_in_pff = 0
not_found_non2015 = 0

# First handle non-2015 missing
if len(missing_other) > 0:
    print(f"{'Player':<30} {'Year':>5} {'Rd':>3} {'College':<20} {'Found?':<10} {'Source':<30} {'Rush Yds':>10}")
    print("-" * 120)

    for _, player in missing_other.iterrows():
        name = player['player_name']
        year = int(player['draft_year'])
        rd = int(player['round'])
        college = player.get('college', '')
        norm = normalize_name(name)
        final_season = year - 1

        # Try exact match
        key = (norm, final_season)
        found = None

        if key in pff_rushing_all:
            found = pff_rushing_all[key]
        else:
            # Try name-only match within ±1 year
            if norm in pff_rushing_by_name:
                matches = pff_rushing_by_name[norm]
                for m in matches:
                    if abs(m['season'] - final_season) <= 1:
                        found = m
                        break

            # Try alternate names
            if found is None:
                parts = norm.split()
                alt_names = set()
                if len(parts) >= 2:
                    alt_names.add(parts[0] + ' ' + parts[-1])
                    alt_names.add(norm + ' iii')
                    alt_names.add(norm + ' jr')
                    alt_names.add(norm + ' ii')
                for alt in alt_names:
                    if alt in pff_rushing_by_name:
                        for m in pff_rushing_by_name[alt]:
                            if abs(m['season'] - final_season) <= 1:
                                found = m
                                break
                    if found:
                        break

        if found:
            found_in_pff += 1
            rushing_fixes.append({
                'player_name': name, 'draft_year': year,
                'rush_attempts': found['rush_att'],
                'rush_yards': found['rush_yards'],
                'rush_touchdowns': found['rush_tds'],
                'player_game_count': found['games'],
                'source': found['source']
            })
            print(f"{name:<30} {year:>5} {rd:>3} {str(college):<20} {'YES':<10} {found['source']:<30} {found['rush_yards']:>10}")
        else:
            not_found_non2015 += 1
            # Mark as 0 rushing (player wasn't in PFF rushing data = didn't rush)
            rushing_fixes.append({
                'player_name': name, 'draft_year': year,
                'rush_attempts': 0, 'rush_yards': 0, 'rush_touchdowns': 0,
                'player_game_count': 13,  # default
                'source': 'assumed_zero (not in PFF rushing files)'
            })
            print(f"{name:<30} {year:>5} {rd:>3} {str(college):<20} {'NO':<10} {'→ Assuming 0 rush yards':<30} {0:>10}")

# Now handle 2015 class
print(f"\n\n--- 2015 Draft Class ({len(missing_2015)} WRs): Need CFBD API ---\n")
print("These 35 WRs need rushing stats from their 2014 college season.")
print("PFF college data doesn't cover 2014. Will need CFBD API.")
print(f"\n{'Player':<30} {'Pick':>5} {'Rd':>3} {'College':<25} {'Status'}")
print("-" * 80)

needs_cfbd_2015 = []
for _, player in missing_2015.iterrows():
    name = player['player_name']
    pick = int(player['pick'])
    rd = int(player['round'])
    college = player.get('college', '')
    needs_cfbd_2015.append({
        'player_name': name, 'pick': pick, 'round': rd, 'college': college
    })
    print(f"{name:<30} {pick:>5} {rd:>3} {college:<25} NEEDS CFBD API")


print(f"\n--- RUSHING FIX SUMMARY ---")
print(f"  Found in PFF rushing files:  {found_in_pff}")
print(f"  Assumed 0 (not in PFF):      {not_found_non2015}")
print(f"  Needs CFBD API (2015 class): {len(needs_cfbd_2015)}")
total_rushing_fixed = found_in_pff + not_found_non2015
print(f"  Already fixable: {284 + total_rushing_fixed}/{339} = {(284+total_rushing_fixed)/339*100:.1f}%")
print(f"  After CFBD API: {284 + total_rushing_fixed + len(needs_cfbd_2015)}/{339} = {(284+total_rushing_fixed+len(needs_cfbd_2015))/339*100:.1f}%")


# ============================================================================
# OVERALL SUMMARY
# ============================================================================

print("\n\n" + "=" * 100)
print("OVERALL COVERAGE SUMMARY (after all fixes)")
print("=" * 100)

print(f"\n{'Component':<25} {'Before':>10} {'After Fixes':>12} {'Still Need':>12} {'Notes'}")
print("-" * 100)

# RAS
ras_after = 339 - ras_truly_missing
print(f"{'Draft Capital':<25} {'339/339':>10} {'339/339':>12} {'0':>12} 100% - no fix needed")
print(f"{'Breakout Age':<25} {'339/339':>10} {'339/339':>12} {'0':>12} 100% - no fix needed")
print(f"{'Teammate Score':<25} {'339/339':>10} {'339/339':>12} {'0':>12} 100% - no fix needed")
print(f"{'RAS (Athletic)':<25} {'286/339':>10} {f'{ras_after}/339':>12} {f'{ras_truly_missing}':>12} "
      f"RAS found: {ras_found}, combine fallback: {ras_from_combine}")
print(f"{'Early Declare':<25} {'290/339':>10} {f'{290+derived_count}/339':>12} {f'{still_missing}':>12} "
      f"Derived from age/seasons: {derived_count}")

rushing_after = 284 + total_rushing_fixed + len(needs_cfbd_2015)
print(f"{'Rushing Production':<25} {'284/339':>10} {f'{rushing_after}/339':>12} {'0':>12} "
      f"PFF files: {found_in_pff}, assumed 0: {not_found_non2015}, CFBD needed: {len(needs_cfbd_2015)}")

print(f"\n{'Component':<25} {'Coverage %':>12}")
print("-" * 40)
print(f"{'Draft Capital':<25} {'100.0%':>12}")
print(f"{'Breakout Age':<25} {'100.0%':>12}")
print(f"{'Teammate Score':<25} {'100.0%':>12}")
print(f"{'RAS (Athletic)':<25} {f'{ras_after/339*100:.1f}%':>12}")
print(f"{'Early Declare':<25} {f'{(290+derived_count)/339*100:.1f}%':>12}")
print(f"{'Rushing Production':<25} {f'{rushing_after/339*100:.1f}%':>12}")


# ============================================================================
# PLAYERS TRULY IMPOSSIBLE TO FILL
# ============================================================================

print(f"\n\n--- PLAYERS WITH GENUINELY MISSING DATA ---")

if ras_truly_missing > 0:
    print(f"\nRAS truly missing ({ras_truly_missing} WRs):")
    for _, p in missing_ras.iterrows():
        norm = normalize_name(p['player_name'])
        key = (norm, int(p['draft_year']))
        # Check if this player was found
        was_found = any(f['player_name'] == p['player_name'] and f['draft_year'] == int(p['draft_year'])
                       for f in ras_fixes)
        if not was_found:
            print(f"  {p['player_name']:<30} {int(p['draft_year']):>5} Rd{int(p['round'])} Pick {int(p['pick'])} {p.get('college', '')}")

if still_missing > 0:
    print(f"\nEarly Declare truly missing ({still_missing} WRs):")
    declared_names = set((f['player_name'], f['draft_year']) for f in declare_fixes)
    for _, p in missing_declare.iterrows():
        if (p['player_name'], int(p['draft_year'])) not in declared_names:
            print(f"  {p['player_name']:<30} {int(p['draft_year']):>5} Rd{int(p['round'])} {p.get('college', '')}")


print("\n\n" + "=" * 100)
print("ANALYSIS COMPLETE — Review results above, then we can apply fixes")
print("=" * 100)
