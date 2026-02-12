"""
Apply Rushing Corrections — Replace false zeros with real data or 'unknown'.

Sources:
- CFBD API retry (Kevin White)
- Web research: Sports-Reference, ESPN, school records (Watson, Hill, etc.)
- Confirmed: no season played (Strachan)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

wr = pd.read_csv('data/wr_backtest_all_components.csv')
print(f"Loaded {len(wr)} WRs")

# ============================================================================
# CORRECTIONS — Every value here is from verified sources
# ============================================================================

CORRECTIONS = [
    # (player_name, draft_year, rush_att, rush_yards, rush_tds, rush_source, notes)

    # CFBD retry — confirmed 0
    ('Kevin White', 2015, 0, 0, 0, 'cfbd_confirmed_zero',
     'CFBD retry: WVU had 12 rushers in 2014, White not among them'),

    # Web research — HIGH confidence
    ('Christian Watson', 2022, 15, 114, 1, 'web_research',
     'ESPN/NDSU Athletics: 15 car, 114 yds, 1 TD in 2021 season'),

    ('Tyreek Hill', 2016, 25, 237, 1, 'web_research',
     'Sports-Reference: 25 car, 237 yds, 1 TD at West Alabama 2015'),

    ('Justin Watson', 2018, 11, 52, 0, 'web_research',
     'FOX Sports/CBS: 11 car, 52 yds, 0 TD at Penn 2017'),

    ('Geremy Davis', 2015, 23, 203, 1, 'web_research',
     'ESPN UConn 2014 team stats: 23 car, 203 yds, 1 TD (incl 68-yd long)'),

    ('Andrei Iosivas', 2023, 4, 10, 0, 'cfbd',
     'CFBD API: 4 car, 10 yds, 0 TD at Princeton 2022'),

    ('Montrell Washington', 2022, 3, 19, 1, 'cfbd',
     'CFBD API: 3 car, 19 yds, 1 TD at Samford 2021'),

    ('Ryan Flournoy', 2024, 6, 49, 1, 'cfbd',
     'CFBD API: 6 car, 49 yds, 1 TD at SE Missouri State 2023'),

    ('Colton Dowell', 2023, 1, 6, 0, 'cfbd',
     'CFBD API: 1 car, 6 yds, 0 TD at UT Martin 2022'),

    ('Cooper Kupp', 2017, 2, 29, 0, 'cfbd',
     'CFBD API: 2 car, 29 yds, 0 TD at Eastern Washington 2016'),

    ('Dezmin Lewis', 2015, 1, 0, 0, 'cfbd',
     'CFBD API: 1 car, 0 yds, 0 TD at Central Arkansas 2014'),

    ('Tre McBride', 2015, 3, 6, 0, 'cfbd',
     'CFBD API: 3 car, 6 yds, 0 TD at William & Mary 2014 (as Keith McBride)'),

    # Web research — MEDIUM confidence (derived from career totals)
    ('Dareke Young', 2022, 1, 5, 0, 'web_research_derived',
     'Derived from career totals minus 2018+2019 seasons. ~1 car, ~5 yds in 2021'),

    ('David Moore', 2017, None, 74, 1, 'web_research_partial',
     'Magnolia Reporter: 74 yds, 1 TD confirmed at East Central 2016. Carries unknown'),

    # No season / no college data
    ('Mike Strachan', 2021, None, None, None, 'no_final_season_data',
     'Charleston (WV) cancelled 2020 season (COVID). Strachan opted out of spring 2021'),

    ('Moritz Boehringer', 2016, None, None, None, 'no_college_data',
     'International player (Germany). No US college rushing data exists'),

    # FCS schools with no data — flagged as unknown, NOT 0
    ('Neal Sterling', 2015, None, None, None, 'unknown_fcs',
     'Monmouth FCS — no rushing stats found in any source. Likely 0 but unconfirmed'),
]

# ============================================================================
# APPLY CORRECTIONS
# ============================================================================

print("\n" + "=" * 120)
print("APPLYING RUSHING CORRECTIONS")
print("=" * 120)

print(f"\n{'#':>3} {'Player':<28} {'Year':>5} {'ATT':>5} {'YDS':>5} {'TD':>3} {'Source':<25} {'Notes'}")
print("-" * 130)

applied = 0
for i, (name, year, att, yds, tds, source, notes) in enumerate(CORRECTIONS, 1):
    mask = (wr['player_name'] == name) & (wr['draft_year'] == year)
    if mask.sum() != 1:
        print(f"  WARNING: {name} ({year}) — {mask.sum()} matches, skipping")
        continue

    wr.loc[mask, 'rush_attempts'] = att
    wr.loc[mask, 'rush_yards'] = yds
    wr.loc[mask, 'rush_touchdowns'] = tds
    wr.loc[mask, 'rush_source'] = source

    att_str = str(int(att)) if att is not None and pd.notna(att) else '?'
    yds_str = str(int(yds)) if yds is not None and pd.notna(yds) else '?'
    tds_str = str(int(tds)) if tds is not None and pd.notna(tds) else '?'
    print(f"{i:>3} {name:<28} {year:>5} {att_str:>5} {yds_str:>5} {tds_str:>3} {source:<25} {notes}")
    applied += 1

print(f"\nApplied {applied} corrections")


# ============================================================================
# FINAL RUSHING SUMMARY
# ============================================================================

print("\n" + "=" * 120)
print("FINAL RUSHING DATA QUALITY SUMMARY")
print("=" * 120)

source_counts = wr['rush_source'].value_counts()
print(f"\n{'Source':<30} {'Count':>6} {'Type'}")
print("-" * 80)

source_types = {
    'pff': 'VERIFIED — PFF college data',
    'cfbd': 'VERIFIED — CFBD API real data',
    'cfbd_confirmed_zero': 'VERIFIED — CFBD team had data, player absent = real 0',
    'web_research': 'VERIFIED — web sources (ESPN, Sports-Ref, school records)',
    'web_research_derived': 'ESTIMATED — derived from career totals minus known seasons',
    'web_research_partial': 'PARTIAL — yards/TDs confirmed, carries unknown',
    'no_final_season_data': 'N/A — player had no final college season (COVID/injury)',
    'no_college_data': 'N/A — international player, no college data exists',
    'unknown_fcs': 'UNKNOWN — FCS school, no source found, NOT assumed 0',
}

verified = 0
unknown = 0
na_count = 0
for src, count in source_counts.items():
    stype = source_types.get(src, '???')
    print(f"  {src:<30} {count:>6} {stype}")
    if 'VERIFIED' in stype:
        verified += count
    elif 'UNKNOWN' in stype or 'PARTIAL' in stype or 'ESTIMATED' in stype:
        unknown += count
    elif 'N/A' in stype:
        na_count += count

print(f"\n  VERIFIED (real data or confirmed 0): {verified}")
print(f"  ESTIMATED/PARTIAL/UNKNOWN:           {unknown}")
print(f"  N/A (no season / international):     {na_count}")
print(f"  Total:                               {verified + unknown + na_count}")


# ============================================================================
# LIST EVERY NON-PFF RUSHING ENTRY (complete audit trail)
# ============================================================================

print("\n" + "=" * 120)
print("COMPLETE AUDIT: ALL NON-PFF RUSHING DATA")
print("=" * 120)

non_pff = wr[wr['rush_source'] != 'pff'].sort_values(['draft_year', 'pick'])
print(f"\n{'#':>3} {'Player':<28} {'Year':>5} {'Rd':>3} {'Pick':>5} {'ATT':>5} {'YDS':>5} {'TD':>3} "
      f"{'Source':<25} {'College'}")
print("-" * 130)

for i, (_, row) in enumerate(non_pff.iterrows(), 1):
    att = row['rush_attempts']
    yds = row['rush_yards']
    tds = row['rush_touchdowns']
    att_str = str(int(att)) if pd.notna(att) else '?'
    yds_str = str(int(yds)) if pd.notna(yds) else '?'
    tds_str = str(int(tds)) if pd.notna(tds) else '?'
    print(f"{i:>3} {row['player_name']:<28} {int(row['draft_year']):>5} {int(row['round']):>3} "
          f"{int(row['pick']):>5} {att_str:>5} {yds_str:>5} {tds_str:>3} "
          f"{row['rush_source']:<25} {row['college']}")

# Save
wr.to_csv('data/wr_backtest_all_components.csv', index=False)
print(f"\nSaved: data/wr_backtest_all_components.csv")

# Final coverage numbers
total = len(wr)
has_yards = wr['rush_yards'].notna().sum()
unknown_rush = wr[wr['rush_source'].isin(['unknown_fcs', 'web_research_partial'])].shape[0]
no_season = wr[wr['rush_source'].isin(['no_final_season_data', 'no_college_data'])].shape[0]

print(f"\nFinal rushing coverage:")
print(f"  Verified data:        {has_yards - unknown_rush - no_season}/{total} ({(has_yards - unknown_rush - no_season)/total*100:.1f}%)")
print(f"  Partial/unknown:      {unknown_rush}/{total} (flagged, not falsely zeroed)")
print(f"  N/A (no season):      {no_season}/{total}")
