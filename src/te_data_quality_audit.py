"""
Complete data quality audit of te_backtest_master.csv.
Checks coverage for every field needed for the TE SLAP model.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

bt = pd.read_csv('data/te_backtest_master.csv')

# Also rebuild breakout score at 15% threshold to check PFF multi-season coverage
pff_file_map = {
    'data/receiving_summary (2).csv': 2015,
    'data/receiving_summary (3).csv': 2016,
    'data/receiving_summary (4).csv': 2017,
    'data/receiving_summary (5).csv': 2018,
    'data/receiving_summary (21).csv': 2019,
    'data/receiving_summary (20).csv': 2020,
    'data/receiving_summary (19).csv': 2021,
    'data/receiving_summary (18).csv': 2022,
    'data/receiving_summary (17).csv': 2023,
    'data/receiving_summary (16).csv': 2024,
    'data/receiving_summary (15).csv': 2025,
}

all_pff = []
for filepath, season in pff_file_map.items():
    pff = pd.read_csv(filepath)
    pff['season'] = season
    all_pff.append(pff)
pff_all = pd.concat(all_pff, ignore_index=True)
receivers = pff_all[pff_all['position'].isin(['TE', 'WR', 'HB'])].copy()
team_totals = receivers.groupby(['team_name', 'season'])['yards'].sum().reset_index()
team_totals.rename(columns={'yards': 'team_rec_yards'}, inplace=True)
te_pff = receivers[receivers['position'] == 'TE'].copy()
te_pff = te_pff.merge(team_totals, on=['team_name', 'season'], how='left')
te_pff['dominator_pct'] = np.where(te_pff['team_rec_yards'] > 0,
    (te_pff['yards'] / te_pff['team_rec_yards']) * 100, 0)

def norm_name(n):
    s = str(n).lower().strip()
    for suf in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
        s = s.replace(suf, '')
    s = s.replace("'", "").replace("\u2019", "").replace("-", "").replace(".", "")
    return s.strip()

te_pff['name_norm'] = te_pff['player'].apply(norm_name)
bt['name_norm_check'] = bt['player_name'].apply(norm_name)

# Build multi-season records for each backtest TE
pff_multi = {}  # player_name -> list of season records
for _, te in bt.iterrows():
    matches = te_pff[(te_pff['name_norm'] == te['name_norm_check']) & (te_pff['season'] < te['draft_year'])]
    if len(matches) > 0:
        seasons = []
        for _, pm in matches.iterrows():
            seasons.append({
                'season': pm['season'],
                'dominator_pct': pm['dominator_pct'],
                'season_age': te['draft_age'] - (te['draft_year'] - pm['season']),
            })
        pff_multi[te['player_name']] = seasons

# Calculate breakout score at 15% for each TE
bo_data = {}
for te_name, seasons in pff_multi.items():
    sdf = sorted(seasons, key=lambda x: x['season'])
    peak_dom = max(s['dominator_pct'] for s in sdf)
    hit_seasons = [s for s in sdf if s['dominator_pct'] >= 15]
    if hit_seasons:
        bo_age = hit_seasons[0]['season_age']
        if bo_age <= 18: base = 100
        elif bo_age <= 19: base = 90
        elif bo_age <= 20: base = 75
        elif bo_age <= 21: base = 60
        elif bo_age <= 22: base = 45
        elif bo_age <= 23: base = 30
        else: base = 20
        bonus = min((peak_dom - 15) * 0.5, 9.9)
        score = min(base + bonus, 99.9)
        bo_data[te_name] = {
            'has_pff_multi': True, 'hit_15': True, 'breakout_age_15': bo_age,
            'peak_dom_pff': peak_dom, 'bo_score_15': score,
            'n_seasons': len(sdf),
        }
    else:
        bo_data[te_name] = {
            'has_pff_multi': True, 'hit_15': False, 'breakout_age_15': np.nan,
            'peak_dom_pff': peak_dom, 'bo_score_15': min(35, 15 + peak_dom),
            'n_seasons': len(sdf),
        }

# TEs without PFF multi-season data
for _, te in bt.iterrows():
    if te['player_name'] not in bo_data:
        bo_data[te['player_name']] = {
            'has_pff_multi': False, 'hit_15': None, 'breakout_age_15': np.nan,
            'peak_dom_pff': np.nan, 'bo_score_15': np.nan,
            'n_seasons': 0,
        }

bo_df = pd.DataFrame.from_dict(bo_data, orient='index')
bo_df.index.name = 'player_name'
bo_df.reset_index(inplace=True)
bt = bt.merge(bo_df, on='player_name', how='left')

# Build hybrid production
bt['rec_per_tpa'] = np.where(
    (bt['cfbd_receptions'].notna()) & (bt['cfbd_team_pass_att'].notna()) & (bt['cfbd_team_pass_att'] > 0),
    bt['cfbd_receptions'] / bt['cfbd_team_pass_att'], np.nan)
bt['pff_rpg'] = np.where(
    (bt['pff_receptions'].notna()) & (bt['pff_player_game_count'].notna()) & (bt['pff_player_game_count'] > 0),
    bt['pff_receptions'] / bt['pff_player_game_count'], np.nan)

cfbd_vals = bt['rec_per_tpa'].dropna()
pff_vals = bt['pff_rpg'].dropna()
if len(cfbd_vals) > 1 and len(pff_vals) > 1:
    bt['cfbd_norm'] = np.where(bt['rec_per_tpa'].notna(),
        (bt['rec_per_tpa'] - cfbd_vals.min()) / (cfbd_vals.max() - cfbd_vals.min()) * 100, np.nan)
    bt['pff_rpg_norm'] = np.where(bt['pff_rpg'].notna(),
        (bt['pff_rpg'] - pff_vals.min()) / (pff_vals.max() - pff_vals.min()) * 100, np.nan)
    bt['production_hybrid'] = np.where(bt['cfbd_norm'].notna(), bt['cfbd_norm'], bt['pff_rpg_norm'])
    bt['prod_source'] = np.where(bt['cfbd_norm'].notna(), 'CFBD', np.where(bt['pff_rpg_norm'].notna(), 'PFF', 'NONE'))

total = len(bt)

# ============================================================
print(f"{'='*100}")
print(f"TE BACKTEST DATA QUALITY AUDIT — {total} TEs")
print(f"{'='*100}")

# ============================================================
# 1. DRAFT CAPITAL
# ============================================================
print(f"\n{'='*80}")
print("1. DRAFT CAPITAL")
print(f"{'='*80}")

has_pick = bt['pick'].notna().sum()
print(f"  pick:         {has_pick}/{total} ({100*has_pick/total:.0f}%)")
has_round = bt['round'].notna().sum()
print(f"  round:        {has_round}/{total} ({100*has_round/total:.0f}%)")
has_college = bt['college'].notna().sum()
print(f"  college:      {has_college}/{total} ({100*has_college/total:.0f}%)")
has_dy = bt['draft_year'].notna().sum()
print(f"  draft_year:   {has_dy}/{total} ({100*has_dy/total:.0f}%)")

missing_pick = bt[bt['pick'].isna()]
if len(missing_pick) > 0:
    print(f"\n  MISSING PICK ({len(missing_pick)} TEs):")
    for _, r in missing_pick.iterrows():
        print(f"    {r['player_name']} ({r['draft_year']})")
else:
    print(f"\n  No missing draft capital data. All 160 TEs have pick numbers.")

# Draft year distribution
print(f"\n  Draft year distribution:")
for yr in sorted(bt['draft_year'].unique()):
    n = len(bt[bt['draft_year'] == yr])
    print(f"    {int(yr)}: {n} TEs")

# Round distribution
print(f"\n  Round distribution:")
for rd in sorted(bt['round'].unique()):
    n = len(bt[bt['round'] == rd])
    print(f"    Rd {int(rd)}: {n} TEs")

# ============================================================
# 2. BREAKOUT AGE (15% threshold)
# ============================================================
print(f"\n{'='*80}")
print("2. BREAKOUT AGE (15% threshold)")
print(f"{'='*80}")

has_pff_multi = bt['has_pff_multi'].sum()
hit_15 = bt['hit_15'].sum()
has_bo_score = bt['bo_score_15'].notna().sum()

print(f"  PFF multi-season data:  {has_pff_multi}/{total} ({100*has_pff_multi/total:.0f}%)")
print(f"  Hit 15% threshold:      {hit_15}/{total} ({100*hit_15/total:.0f}%)")
print(f"  Never hit 15%:          {has_pff_multi - hit_15}/{total}")
print(f"  Breakout score calc'd:  {has_bo_score}/{total} ({100*has_bo_score/total:.0f}%)")
print(f"  NO PFF multi-season:    {total - has_pff_multi}/{total}")

# Seasons per TE
season_counts = bt[bt['has_pff_multi'] == True]['n_seasons']
if len(season_counts) > 0:
    print(f"\n  Seasons of PFF data per TE (for those with data):")
    print(f"    Mean: {season_counts.mean():.1f}, Median: {season_counts.median():.0f}")
    print(f"    Range: {season_counts.min():.0f} - {season_counts.max():.0f}")
    for n in sorted(season_counts.unique()):
        cnt = (season_counts == n).sum()
        print(f"    {int(n)} seasons: {cnt} TEs")

# List TEs WITHOUT PFF multi-season data
no_pff = bt[bt['has_pff_multi'] == False]
print(f"\n  TEs WITHOUT PFF multi-season data ({len(no_pff)}):")
for _, r in no_pff.sort_values(['round', 'pick']).iterrows():
    print(f"    {r['player_name']:<30s} | {int(r['draft_year'])} Rd {int(r['round'])} Pick {int(r['pick']):>3d} | {r['college']}")

# Age data quality
print(f"\n  AGE DATA:")
has_age = bt['draft_age'].notna().sum()
print(f"    draft_age available: {has_age}/{total} ({100*has_age/total:.0f}%)")
# Check if ages look estimated (round numbers, etc.)
ages = bt['draft_age'].dropna()
print(f"    Age range: {ages.min():.1f} - {ages.max():.1f}")
print(f"    Mean: {ages.mean():.1f}")
# Check for integer vs float ages (integer = possibly estimated)
int_ages = (ages == ages.astype(int)).sum()
float_ages = (ages != ages.astype(int)).sum()
print(f"    Integer ages: {int_ages}/{len(ages)} (possibly estimated)")
print(f"    Fractional ages: {float_ages}/{len(ages)} (likely exact birthdate)")

# ============================================================
# 3. RECEIVING PRODUCTION (Rec/TPA hybrid)
# ============================================================
print(f"\n{'='*80}")
print("3. RECEIVING PRODUCTION (Rec/TPA + PFF fallback)")
print(f"{'='*80}")

has_cfbd_rec = bt['cfbd_receptions'].notna().sum()
has_cfbd_tpa = bt['cfbd_team_pass_att'].notna().sum()
has_cfbd_both = (bt['cfbd_receptions'].notna() & bt['cfbd_team_pass_att'].notna()).sum()
has_rec_tpa = bt['rec_per_tpa'].notna().sum()
has_pff_rpg = bt['pff_rpg'].notna().sum()
has_hybrid = bt['production_hybrid'].notna().sum()

print(f"  CFBD receptions:        {has_cfbd_rec}/{total} ({100*has_cfbd_rec/total:.0f}%)")
print(f"  CFBD team pass att:     {has_cfbd_tpa}/{total} ({100*has_cfbd_tpa/total:.0f}%)")
print(f"  CFBD Rec/TPA calc'd:    {has_rec_tpa}/{total} ({100*has_rec_tpa/total:.0f}%)")
print(f"  PFF Rec/Game (fallback):{has_pff_rpg}/{total} ({100*has_pff_rpg/total:.0f}%)")
print(f"  Hybrid coverage:        {has_hybrid}/{total} ({100*has_hybrid/total:.0f}%)")

# Source breakdown
cfbd_used = (bt['prod_source'] == 'CFBD').sum()
pff_used = (bt['prod_source'] == 'PFF').sum()
none_used = (bt['prod_source'] == 'NONE').sum()
print(f"\n  Hybrid source breakdown:")
print(f"    CFBD primary:   {cfbd_used}")
print(f"    PFF fallback:   {pff_used}")
print(f"    Neither:        {none_used}")

# List TEs missing BOTH sources
no_prod = bt[bt['production_hybrid'].isna()]
print(f"\n  TEs missing BOTH CFBD and PFF production ({len(no_prod)}):")
for _, r in no_prod.sort_values(['round', 'pick']).iterrows():
    cfbd_status = "has CFBD" if pd.notna(r['cfbd_receptions']) else "no CFBD"
    pff_status = "has PFF" if pd.notna(r['pff_receptions']) else "no PFF"
    print(f"    {r['player_name']:<30s} | {int(r['draft_year'])} Rd {int(r['round'])} Pick {int(r['pick']):>3d} | {r['college']:<25s} | {cfbd_status}, {pff_status}")

# Coverage by round
print(f"\n  Hybrid production coverage by round:")
for rd in sorted(bt['round'].unique()):
    rd_data = bt[bt['round'] == rd]
    has = rd_data['production_hybrid'].notna().sum()
    print(f"    Rd {int(rd)}: {has}/{len(rd_data)} ({100*has/len(rd_data):.0f}%)")

# ============================================================
# 4. NFL OUTCOMES
# ============================================================
print(f"\n{'='*80}")
print("4. NFL OUTCOMES")
print(f"{'='*80}")

has_ppg = bt['first_3yr_ppg'].notna().sum()
has_h24 = bt['hit24'].notna().sum()
has_h12 = bt['hit12'].notna().sum()
has_career = bt['career_ppg'].notna().sum()
has_nfl = bt['nfl_seasons_found'].notna().sum()
has_nfl_gt0 = (bt['nfl_seasons_found'] > 0).sum()

print(f"  nfl_seasons_found > 0:  {has_nfl_gt0}/{total} ({100*has_nfl_gt0/total:.0f}%)")
print(f"  hit24:                  {has_h24}/{total} ({100*has_h24/total:.0f}%)")
print(f"  hit12:                  {has_h12}/{total} ({100*has_h12/total:.0f}%)")
print(f"  first_3yr_ppg:          {has_ppg}/{total} ({100*has_ppg/total:.0f}%)")
print(f"  career_ppg:             {has_career}/{total} ({100*has_career/total:.0f}%)")

# Missing outcomes by draft year
print(f"\n  Outcome coverage by draft year:")
for yr in sorted(bt['draft_year'].unique()):
    yr_data = bt[bt['draft_year'] == yr]
    n = len(yr_data)
    has = (yr_data['nfl_seasons_found'] > 0).sum()
    ppg = yr_data['first_3yr_ppg'].notna().sum()
    h24 = yr_data['hit24'].notna().sum()
    print(f"    {int(yr)}: {n} TEs, outcomes={has}/{n}, ppg={ppg}/{n}, hit24={h24}/{n}" +
          (" *** INCOMPLETE (too recent)" if yr >= 2024 else ""))

# TEs with NO NFL outcomes
no_outcomes = bt[bt['nfl_seasons_found'] == 0]
if len(no_outcomes) > 0:
    print(f"\n  TEs with NO NFL outcome data ({len(no_outcomes)}):")
    for _, r in no_outcomes.sort_values(['draft_year', 'pick']).iterrows():
        seasons = r.get('seasons_played', 0)
        print(f"    {r['player_name']:<30s} | {int(r['draft_year'])} Rd {int(r['round'])} Pick {int(r['pick']):>3d} | seasons_played={seasons}")

# TEs with outcomes but missing first_3yr_ppg
has_nfl_no_ppg = bt[(bt['nfl_seasons_found'] > 0) & (bt['first_3yr_ppg'].isna())]
if len(has_nfl_no_ppg) > 0:
    print(f"\n  TEs with NFL data but NO first_3yr_ppg ({len(has_nfl_no_ppg)}):")
    for _, r in has_nfl_no_ppg.iterrows():
        print(f"    {r['player_name']:<30s} | {int(r['draft_year'])} Rd {int(r['round'])} | nfl_seasons={r['nfl_seasons_found']}")

# ============================================================
# 5. BIRTHDATES AND AGES
# ============================================================
print(f"\n{'='*80}")
print("5. BIRTHDATES AND AGES")
print(f"{'='*80}")

has_age = bt['draft_age'].notna().sum()
print(f"  draft_age available:    {has_age}/{total} ({100*has_age/total:.0f}%)")

# Check for age_estimated column
if 'age_estimated' in bt.columns:
    est = bt['age_estimated'].sum() if bt['age_estimated'].notna().any() else 0
    print(f"  age_estimated flag:     {est} TEs flagged as estimated")

missing_age = bt[bt['draft_age'].isna()]
if len(missing_age) > 0:
    print(f"\n  TEs MISSING draft_age ({len(missing_age)}):")
    for _, r in missing_age.iterrows():
        print(f"    {r['player_name']:<30s} | {int(r['draft_year'])} Rd {int(r['round'])}")
else:
    print(f"  No missing ages.")

# Age distribution
print(f"\n  Age distribution at draft:")
age_counts = bt['draft_age'].dropna().astype(int).value_counts().sort_index()
for age, count in age_counts.items():
    print(f"    Age {age}: {count} TEs")

# ============================================================
# 6. ATHLETIC DATA
# ============================================================
print(f"\n{'='*80}")
print("6. ATHLETIC DATA (for reference — may not be used in model)")
print(f"{'='*80}")

ath_cols = {
    'te_ras': 'RAS (0-10)',
    'wt': 'Weight',
    'forty': '40-yard dash',
    'bench': 'Bench press',
    'vertical': 'Vertical jump',
    'broad_jump': 'Broad jump',
    'cone': '3-cone drill',
    'shuttle': 'Shuttle',
}

for col, name in ath_cols.items():
    has = bt[col].notna().sum()
    print(f"  {name:<20s}: {has}/{total} ({100*has/total:.0f}%)")

# ============================================================
# 7. EARLY DECLARE
# ============================================================
print(f"\n{'='*80}")
print("7. EARLY DECLARE")
print(f"{'='*80}")

has_ed = bt['early_declare'].notna().sum()
ed_yes = (bt['early_declare'] == 1).sum()
ed_no = (bt['early_declare'] == 0).sum()
print(f"  early_declare data:     {has_ed}/{total} ({100*has_ed/total:.0f}%)")
print(f"  Declared early:         {ed_yes}")
print(f"  Did not declare early:  {ed_no}")

# ============================================================
# 8. OVERALL MODEL READINESS — COMBINED COVERAGE
# ============================================================
print(f"\n{'='*80}")
print("8. COMBINED MODEL READINESS")
print(f"{'='*80}")

# For each TE, what components can we calculate?
bt['has_dc'] = bt['pick'].notna()
bt['has_bo'] = bt['bo_score_15'].notna()
bt['has_prod'] = bt['production_hybrid'].notna()
bt['has_outcomes'] = bt['nfl_seasons_found'] > 0
bt['has_ppg'] = bt['first_3yr_ppg'].notna()

# Full coverage for model building
all_3 = bt['has_dc'] & bt['has_bo'] & bt['has_prod']
all_3_outcomes = all_3 & bt['has_outcomes']
all_3_ppg = all_3 & bt['has_ppg']

dc_bo = bt['has_dc'] & bt['has_bo']
dc_prod = bt['has_dc'] & bt['has_prod']

print(f"\n  3-component model (DC + Breakout + Production):")
print(f"    All 3 components:                   {all_3.sum()}/{total} ({100*all_3.sum()/total:.0f}%)")
print(f"    All 3 + NFL outcomes:               {all_3_outcomes.sum()}/{total} ({100*all_3_outcomes.sum()/total:.0f}%)")
print(f"    All 3 + first_3yr_ppg:              {all_3_ppg.sum()}/{total} ({100*all_3_ppg.sum()/total:.0f}%)")

print(f"\n  2-component model options:")
print(f"    DC + Breakout only:                 {dc_bo.sum()}/{total} ({100*dc_bo.sum()/total:.0f}%)")
print(f"    DC + Production only:               {dc_prod.sum()}/{total} ({100*dc_prod.sum()/total:.0f}%)")

# What's missing for the 3-component model?
missing_any = bt[~all_3]
print(f"\n  TEs missing at least one component ({len(missing_any)}):")
missing_breakdown = {
    'Missing breakout only': (~bt['has_bo'] & bt['has_prod']).sum(),
    'Missing production only': (bt['has_bo'] & ~bt['has_prod']).sum(),
    'Missing both': (~bt['has_bo'] & ~bt['has_prod']).sum(),
}
for desc, count in missing_breakdown.items():
    print(f"    {desc}: {count}")

# List the ones missing components
print(f"\n  Detailed list of TEs missing components:")
for _, r in missing_any.sort_values(['round', 'pick']).iterrows():
    bo_status = "has BO" if r['has_bo'] else "NO BO"
    prod_status = "has PROD" if r['has_prod'] else "NO PROD"
    print(f"    {r['player_name']:<30s} | {int(r['draft_year'])} Rd {int(r['round'])} Pick {int(r['pick']):>3d} | {bo_status:>7s} | {prod_status:>8s} | {r['college']}")

# ============================================================
# 9. 2025 DRAFT CLASS SPECIAL CHECK
# ============================================================
print(f"\n{'='*80}")
print("9. 2025 DRAFT CLASS — SPECIAL CHECK (too recent for full outcomes)")
print(f"{'='*80}")

d25 = bt[bt['draft_year'] == 2025]
print(f"  2025 TEs in backtest: {len(d25)}")
print(f"  With NFL outcomes: {(d25['nfl_seasons_found'] > 0).sum()}")
print(f"  With first_3yr_ppg: {d25['first_3yr_ppg'].notna().sum()}")
print(f"  With breakout score: {d25['bo_score_15'].notna().sum()}")
print(f"  With production: {d25['production_hybrid'].notna().sum()}")
print(f"\n  Note: 2025 TEs have at most 1 NFL season. first_3yr_ppg will be")
print(f"  based on incomplete data. Consider whether to include in backtest.")

# Same for 2024
d24 = bt[bt['draft_year'] == 2024]
print(f"\n  2024 TEs in backtest: {len(d24)}")
print(f"  With NFL outcomes: {(d24['nfl_seasons_found'] > 0).sum()}")
print(f"  With first_3yr_ppg: {d24['first_3yr_ppg'].notna().sum()}")
print(f"  Note: 2024 TEs have at most 2 NFL seasons. first_3yr_ppg may be incomplete.")

print(f"\n{'='*80}")
print("AUDIT COMPLETE")
print(f"{'='*80}")
