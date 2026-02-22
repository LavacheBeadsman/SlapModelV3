"""
Generate 2026 RB Prospects V5 SLAP Scores
==========================================
RB V5: 65% DC / 30% RYPTPA / 5% Speed Score
"""

import pandas as pd
import numpy as np
import warnings, os
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

# --- Helpers ---
def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()

def rb_production_score(rec_yards, team_pass_att, age):
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return np.nan
    if pd.isna(age): age = 22
    age = float(age)
    season_age = age - 1
    age_w = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
    raw = (rec_yards / team_pass_att) * age_w * 100
    return min(99.9, max(0, raw / 1.75))

def speed_score_fn(weight, forty):
    if pd.isna(weight) or pd.isna(forty) or forty <= 0 or weight <= 0:
        return np.nan
    return (weight * 200) / (forty ** 4)

# --- Load prospects ---
p = pd.read_csv('data/prospects_final.csv')
rb26 = p[p['position'] == 'RB'].copy()
rb26['age'] = pd.to_numeric(rb26['age'], errors='coerce')
rb26['weight'] = pd.to_numeric(rb26['weight'], errors='coerce')
rb26['rec_yards'] = pd.to_numeric(rb26['rec_yards'], errors='coerce')
rb26['team_pass_attempts'] = pd.to_numeric(rb26['team_pass_attempts'], errors='coerce')

# --- DC score ---
rb26['dc_score'] = rb26['projected_pick'].apply(dc_score)

# --- Production score ---
rb26['production_score'] = rb26.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_attempts'], r['age']), axis=1)

# Backtest average for missing
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
rb_bt['prod'] = rb_bt.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
avg_prod = rb_bt['prod'].mean()
rb26['production_score_final'] = rb26['production_score'].fillna(avg_prod)
rb26['production_status'] = np.where(rb26['production_score'].notna(), 'observed', 'imputed')

# --- Speed Score ---
# Build 40-time lookup from backtest combine data
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine['name_norm'] = combine['player_name'].apply(normalize_name)
combine_lookup = {}
for pos in ['RB', 'FB', 'WR', 'TE']:
    sub = combine[combine['pos'] == pos]
    for _, row in sub.iterrows():
        dy = row.get('draft_year')
        if pd.isna(dy): dy = row.get('season')
        if pd.isna(dy): continue
        key = (row['name_norm'], int(dy))
        if key not in combine_lookup:
            combine_lookup[key] = {'weight': row['wt'], 'forty': row['forty']}

rb_bt['name_norm'] = rb_bt['player_name'].apply(normalize_name)
rb_bt['bt_weight'] = rb_bt.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb_bt['bt_forty'] = rb_bt.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb_bt['bt_weight'] = pd.to_numeric(rb_bt['bt_weight'], errors='coerce')
rb_bt['bt_forty'] = pd.to_numeric(rb_bt['bt_forty'], errors='coerce')
rb_bt['raw_ss'] = rb_bt.apply(lambda r: speed_score_fn(r['bt_weight'], r['bt_forty']), axis=1)
ss_min = rb_bt['raw_ss'].dropna().min()
ss_max = rb_bt['raw_ss'].dropna().max()

# Build weight × round → avg 40 lookup
known = rb_bt[rb_bt['bt_weight'].notna() & rb_bt['bt_forty'].notna()].copy()
def wt_bucket(wt):
    if pd.isna(wt): return None
    if wt < 200: return '<200'
    elif wt < 210: return '200-209'
    elif wt < 220: return '210-219'
    else: return '220+'
def rd_bucket(rd):
    if rd <= 1: return 'Rd 1'
    elif rd <= 2: return 'Rd 2'
    elif rd <= 4: return 'Rd 3-4'
    else: return 'Rd 5+'

known['wb'] = known['bt_weight'].apply(wt_bucket)
known['rb_bucket'] = known['round'].apply(rd_bucket)
lookup_40 = {}
for wb in ['<200', '200-209', '210-219', '220+']:
    for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
        sub = known[(known['wb'] == wb) & (known['rb_bucket'] == rdb)]
        if len(sub) > 0:
            lookup_40[(wb, rdb)] = sub['bt_forty'].mean()
    wt_sub = known[known['wb'] == wb]
    if len(wt_sub) > 0:
        for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
            if (wb, rdb) not in lookup_40:
                lookup_40[(wb, rdb)] = wt_sub['bt_forty'].mean()

# MNAR percentiles
p60 = rb_bt['raw_ss'].quantile(0.60)
p40 = rb_bt['raw_ss'].quantile(0.40)

# Calculate speed scores for 2026 prospects
rb26['projected_round'] = np.ceil(rb26['projected_pick'] / 32).astype(int)
rb26['raw_ss'] = np.nan
rb26['speed_score_status'] = 'imputed_mnar'

for idx in rb26.index:
    wt = rb26.loc[idx, 'weight']
    rd = rb26.loc[idx, 'projected_round']
    if pd.notna(wt):
        wb = wt_bucket(wt)
        rdb = rd_bucket(rd)
        est_40 = lookup_40.get((wb, rdb))
        if est_40 is not None:
            rb26.loc[idx, 'raw_ss'] = speed_score_fn(wt, est_40)
            rb26.loc[idx, 'speed_score_status'] = 'weight_est40'
    if pd.isna(rb26.loc[idx, 'raw_ss']):
        rb26.loc[idx, 'raw_ss'] = p60 if rd <= 2 else p40

rb26['speed_score'] = ((rb26['raw_ss'] - ss_min) / (ss_max - ss_min) * 100).clip(0, 100)

# --- V5 SLAP ---
rb26['slap_v5'] = (
    0.65 * rb26['dc_score'] +
    0.30 * rb26['production_score_final'] +
    0.05 * rb26['speed_score']
)
rb26['delta_vs_dc'] = rb26['slap_v5'] - rb26['dc_score']

# --- Output ---
out = rb26[['player_name', 'school', 'projected_pick', 'age',
            'slap_v5', 'dc_score', 'production_score_final', 'speed_score',
            'delta_vs_dc', 'production_status', 'speed_score_status',
            'rec_yards', 'team_pass_attempts', 'weight']].copy()
out.columns = ['player_name', 'school', 'projected_pick', 'age',
               'slap_v5', 'dc_score', 'production_score', 'speed_score',
               'delta_vs_dc', 'production_status', 'speed_score_status',
               'rec_yards', 'team_pass_attempts', 'weight']
out = out.sort_values('slap_v5', ascending=False).reset_index(drop=True)
out.index = out.index + 1
out.index.name = 'rank'
out['slap_v5'] = out['slap_v5'].round(1)
out['dc_score'] = out['dc_score'].round(1)
out['production_score'] = out['production_score'].round(1)
out['speed_score'] = out['speed_score'].round(1)
out['delta_vs_dc'] = out['delta_vs_dc'].round(1)

out.to_csv('output/slap_v5_rb_2026.csv')
print(f'Saved: output/slap_v5_rb_2026.csv ({len(out)} RBs)\n')

print(f'{"Rank":>4s} {"Player":<25s} {"School":<20s} {"Pick":>4s} {"SLAP":>6s} {"DC":>5s} {"Prod":>5s} {"SS":>5s} {"Delta":>6s}')
print('-' * 85)
for i, r in out.iterrows():
    prod_flag = '*' if r['production_status'] == 'imputed' else ''
    print(f'{i:>4d} {r["player_name"]:<25s} {r["school"]:<20s} {int(r["projected_pick"]):>4d} {r["slap_v5"]:>6.1f} {r["dc_score"]:>5.1f} {r["production_score"]:>4.1f}{prod_flag:<1s} {r["speed_score"]:>5.1f} {r["delta_vs_dc"]:>+6.1f}')

print(f'\n* = imputed production (missing receiving data)')
print(f'Speed scores are estimated (no 2026 combine data yet)')
