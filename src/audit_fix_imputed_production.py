"""
SLAP V5 Production Data Audit & Fix
====================================
Replaces imputed/missing production values with real data sourced from:
- CFBD API (10 RBs recovered)
- ESPN (2 RBs: Gurley, Duke Johnson)
- Web research (FCS team pass attempts, Trenton Cannon receiving)

Players who remain unfixable are flagged with null production values
instead of being imputed to the position average.

This script modifies:
  - data/rb_backtest_with_receiving.csv (add real data, flag missing)
  - Prints a complete audit report
"""

import pandas as pd
import numpy as np
import os
os.chdir('/home/user/SlapModelV3')

rb = pd.read_csv('data/rb_backtest_with_receiving.csv')
print(f"Total RBs: {len(rb)}")
print("=" * 100)
print("PRODUCTION DATA AUDIT & FIX REPORT")
print("=" * 100)

# ============================================================================
# FIXES: Real data recovered via CFBD API + web research
# ============================================================================

fixes = {
    # CFBD API recoveries (9 players — both rec_yards and team_pass_att)
    ('Tyler Ervin', 2016):        {'rec_yards': 334, 'team_pass_att': 391, 'source': 'CFBD API'},
    ('Devante Mays', 2017):       {'rec_yards': 8,   'team_pass_att': 383, 'source': 'CFBD API'},
    ('Ito Smith', 2018):          {'rec_yards': 396, 'team_pass_att': 438, 'source': 'CFBD API'},
    ('Darwin Thompson', 2019):    {'rec_yards': 351, 'team_pass_att': 464, 'source': 'CFBD API'},
    ("Ke'Shawn Vaughn", 2020):    {'rec_yards': 270, 'team_pass_att': 381, 'source': 'CFBD API'},
    ('Darrynton Evans', 2020):    {'rec_yards': 198, 'team_pass_att': 369, 'source': 'CFBD API'},
    ('Zander Horvath', 2022):     {'rec_yards': 108, 'team_pass_att': 576, 'source': 'CFBD API'},
    ('Ray Davis', 2024):          {'rec_yards': 323, 'team_pass_att': 375, 'source': 'CFBD API'},
    ('Bucky Irving', 2024):       {'rec_yards': 413, 'team_pass_att': 514, 'source': 'CFBD API'},

    # ESPN recoveries (2 players — Georgia & Miami 2014)
    ('Todd Gurley', 2015):        {'rec_yards': 57,  'team_pass_att': 322, 'receptions': 12, 'source': 'ESPN (Georgia 2014 team stats)'},
    ('Duke Johnson', 2015):       {'rec_yards': 421, 'team_pass_att': 391, 'receptions': 38, 'source': 'ESPN (Miami 2014 team stats)'},

    # CFBD API partial + web research (FCS schools: rec_yards existed, team_pass_att found)
    # Tarik Cohen: rec_yards from CFBD=125, NC A&T 2016 QB Lamar Raynard: 209 attempts + other QBs
    # NC A&T official site doesn't have full stats, so Raynard had 209 ATT in 9 games, plus backup QBs
    # Cohen's Wikipedia says "37 catches and 339 yards" in 2016 — more reliable
    ('Tarik Cohen', 2017):        {'rec_yards': 339, 'team_pass_att': None, 'receptions': 37,
                                   'source': 'Wikipedia (season stats verified). TPA: NC A&T FCS partial data only (QB had 209 ATT in 9 games, team total unavailable)'},

    # David Johnson (Northern Iowa 2014): rec_yards=276 already in data, TPA from UNI athletics = 417
    ('David Johnson', 2015):      {'team_pass_att': 417, 'source': 'UNI Athletics (2014 cumulative stats)'},

    # Chase Edmonds (Fordham 2017): rec_yards=21 already in data (11 catches, 129 yards per Wikipedia)
    # BUT the CSV says 21 rec_yards — Wikipedia says 129. Wikipedia career check:
    # Career: 86 receptions, 905 yards. 2017 senior: 11 rec for 129 yards. The 21 in CSV appears wrong.
    # Kevin Anderson (QB) had 245 attempts in 8 games. Team total ~275-290 with backups.
    ('Chase Edmonds', 2018):      {'rec_yards': 129, 'receptions': 11, 'team_pass_att': None,
                                   'source': 'Wikipedia (career 86 rec/905 yds; senior season 11 rec/129 yds). TPA: Fordham FCS QB had 245 ATT in 8 of 11 games, team total unavailable'},

    # Pierre Strong (SDSU 2021): rec_yards=8 already in data, TPA from CFBD = partial (28 ATT in 1 game)
    # Web research: QB Chris Oladokun had 382-384 attempts + Pierre Strong 4 ATT = ~388 team total
    ('Pierre Strong', 2022):      {'team_pass_att': 388, 'source': 'Web research (Chris Oladokun 384 ATT + Pierre Strong 4 ATT = ~388)'},

    # Isaiah Davis (SDSU 2023): rec_yards=190 already in data, TPA from CFBD game-level = 317
    ('Isaiah Davis', 2024):       {'team_pass_att': 317, 'source': 'CFBD game-level stats (14 of 15 games captured)'},

    # Dylan Laube (New Hampshire 2023): rec_yards=699 already in data, TPA from web = Max Brosmer 459 ATT
    ('Dylan Laube', 2024):        {'team_pass_att': 465, 'source': 'CFBD game-level stats (11 of 11 games) + confirmed by web (Brosmer 459 + backups)'},

    # Trenton Cannon (Virginia State 2017): Division II HBCU, not in CFBD
    # Web research found: 21 receptions, 225 receiving yards, 3 receiving TDs
    # Team pass attempts: Virginia State 2017 QB Cordelral Cook stats not available online
    ('Trenton Cannon', 2018):     {'rec_yards': 225, 'receptions': 21, 'team_pass_att': None,
                                   'source': 'Wikipedia (21 rec, 225 yds, 3 TD in 2017). TPA: Virginia State D2, team passing stats unavailable'},

    # De'Angelo Henderson (Coastal Carolina 2016): FCS independent in transition year
    # Web research: Henderson had 40 receptions for 403 receiving yards (2016 per CCU athletics)
    # BUT this may be 2015 stats. The 2016 season is ambiguous in search results.
    # QB Tyler Keane: 105 ATT in 7 of 12 games. Team total unavailable (FCS independent, no conference stats)
    ("De'Angelo Henderson", 2017): {'rec_yards': None, 'team_pass_att': None,
                                    'source': 'UNFIXABLE: Coastal Carolina FCS independent in 2016. Receiving stats ambiguous between 2015/2016 seasons. Team pass attempts unavailable.'},

    # Isiah Pacheco (Rutgers 2021): CFBD API found Isaih Pacheco — 13 rec, 25 yds, Rutgers 394 TPA
    ('Isiah Pacheco', 2022):      {'rec_yards': 25, 'receptions': 13, 'team_pass_att': 394,
                                   'source': 'CFBD API (Rutgers 2021: 13 rec, 25 yds; team 394 pass att)'},
}

# ============================================================================
# APPLY FIXES
# ============================================================================

print("\n" + "=" * 100)
print("APPLYING FIXES")
print("=" * 100)

fixed_count = 0
still_missing = []

for (name, draft_year), fix in fixes.items():
    mask = (rb['player_name'] == name) & (rb['draft_year'] == draft_year)
    if mask.sum() == 0:
        print(f"  WARNING: {name} ({draft_year}) not found in data!")
        continue

    idx = rb[mask].index[0]
    old_rec = rb.loc[idx, 'rec_yards']
    old_tpa = rb.loc[idx, 'team_pass_att']

    changes = []
    for col in ['rec_yards', 'receptions', 'team_pass_att']:
        if col in fix and fix[col] is not None:
            old_val = rb.loc[idx, col]
            rb.loc[idx, col] = fix[col]
            if pd.isna(old_val) or old_val != fix[col]:
                changes.append(f"{col}: {old_val} → {fix[col]}")

    source = fix.get('source', 'unknown')
    if changes:
        print(f"\n  FIXED: {name} ({draft_year})")
        for c in changes:
            print(f"    {c}")
        print(f"    Source: {source}")
        fixed_count += 1
    else:
        # Check if this player still has missing data
        new_rec = rb.loc[idx, 'rec_yards']
        new_tpa = rb.loc[idx, 'team_pass_att']
        if pd.isna(new_rec) or pd.isna(new_tpa):
            still_missing.append({
                'player_name': name,
                'draft_year': draft_year,
                'rec_yards': new_rec,
                'team_pass_att': new_tpa,
                'reason': source
            })

print(f"\n  Total fixes applied: {fixed_count}")

# Pacheco data already hardcoded in fixes dict above (from CFBD API: 13 rec, 25 yds, 394 TPA)

# ============================================================================
# FINAL AUDIT: Who still has missing production data?
# ============================================================================
print("\n" + "=" * 100)
print("FINAL AUDIT: REMAINING MISSING PRODUCTION DATA")
print("=" * 100)

def rb_production_score(rec_yards, team_pass_att, age):
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return np.nan
    if pd.isna(age): age = 22
    season_age = age - 1
    age_w = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
    raw = (rec_yards / team_pass_att) * age_w * 100
    return min(99.9, max(0, raw / 1.75))

rb['s_production'] = rb.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)

still_null = rb[rb['s_production'].isna()]
print(f"\nPlayers with NULL production score AFTER fixes: {len(still_null)}")
for _, r in still_null.iterrows():
    print(f"  {r['player_name']} ({int(r['draft_year'])}, Rd{int(r['round'])}, Pick {int(r['pick'])}, {r['college']})")
    print(f"    rec_yards={r['rec_yards']}, team_pass_att={r['team_pass_att']}, hit24={r['hit24']}")

# How many were calculable before vs after
rb_orig = pd.read_csv('data/rb_backtest_with_receiving.csv')
rb_orig['s_production'] = rb_orig.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
before_null = rb_orig['s_production'].isna().sum()
after_null = rb['s_production'].isna().sum()

print(f"\n  Before fixes: {before_null} players with NULL production ({len(rb) - before_null} real)")
print(f"  After fixes:  {after_null} players with NULL production ({len(rb) - after_null} real)")
print(f"  Recovered:    {before_null - after_null} players now have real production scores")

# ============================================================================
# SHOW IMPACT: What are the new production scores for fixed players?
# ============================================================================
print("\n" + "=" * 100)
print("IMPACT: NEW PRODUCTION SCORES FOR FIXED PLAYERS")
print("=" * 100)

for (name, draft_year), fix in fixes.items():
    mask = (rb['player_name'] == name) & (rb['draft_year'] == draft_year)
    if mask.sum() == 0: continue
    r = rb[mask].iloc[0]
    if pd.notna(r['s_production']):
        print(f"  {name:25s} ({int(draft_year)}) Rd{int(r['round'])} Pick {int(r['pick']):3d}  "
              f"Production: {r['s_production']:.1f}  (rec_yds={r['rec_yards']:.0f}, tpa={r['team_pass_att']:.0f})  "
              f"hit24={int(r['hit24'])}")

# ============================================================================
# SAVE UPDATED DATA
# ============================================================================
print("\n" + "=" * 100)
print("SAVING UPDATED DATA")
print("=" * 100)

rb.drop(columns=['s_production', 'name_norm'], errors='ignore', inplace=True)
rb.to_csv('data/rb_backtest_with_receiving.csv', index=False)
print(f"  Saved: data/rb_backtest_with_receiving.csv ({len(rb)} rows)")

# ============================================================================
# SPEED SCORE AUDIT SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("SPEED SCORE AUDIT SUMMARY")
print("=" * 100)

combine = pd.read_parquet('data/nflverse/combine.parquet')
def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()

combine_lookup = {}
for pos_pref in ['RB', 'FB', 'WR', 'TE']:
    sub = combine[combine['pos'] == pos_pref]
    for _, row in sub.iterrows():
        dy = row.get('draft_year')
        if pd.isna(dy): dy = row.get('season')
        if pd.isna(dy): continue
        key = (normalize_name(row['player_name']), int(dy))
        if key not in combine_lookup:
            combine_lookup[key] = {'weight': row['wt'], 'forty': row['forty']}

rb['name_norm'] = rb['player_name'].apply(normalize_name)
real_both = 0
weight_only = 0
neither = 0
weight_only_players = []
neither_players = []

for _, r in rb.iterrows():
    key = (r['name_norm'], int(r['draft_year']))
    cdata = combine_lookup.get(key, {})
    has_w = pd.notna(cdata.get('weight'))
    has_f = pd.notna(cdata.get('forty'))
    if has_w and has_f:
        real_both += 1
    elif has_w:
        weight_only += 1
        weight_only_players.append(f"  {r['player_name']} ({int(r['draft_year'])}, Rd{int(r['round'])}, {r['college']})")
    else:
        neither += 1
        neither_players.append(f"  {r['player_name']} ({int(r['draft_year'])}, Rd{int(r['round'])}, {r['college']})")

print(f"\n  REAL speed score (weight+40 from combine): {real_both} ({real_both/len(rb)*100:.1f}%)")
print(f"  ESTIMATED 40-time (real weight, bucket-avg 40): {weight_only} ({weight_only/len(rb)*100:.1f}%)")
print(f"  MNAR-imputed (no combine data at all): {neither} ({neither/len(rb)*100:.1f}%)")

print(f"\n  Weight-only players (40-time estimated from weight×round bucket averages):")
for p in weight_only_players:
    print(p)

print(f"\n  MNAR-imputed players (Rd1-2 → 60th pctl, Rd3+ → 40th pctl):")
for p in neither_players:
    print(p)

print(f"\n{'='*100}")
print("AUDIT COMPLETE")
print("=" * 100)
