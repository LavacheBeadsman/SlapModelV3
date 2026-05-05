"""Run the same cross-team + reliability-filter breakout audit on:
  - TE 2026 prospects (full recompute, same logic as WR 2026)
  - WR backtest (NaN-fill only — don't change locked validation data)
  - TE backtest (NaN-fill only — don't change locked validation data)

Reliability filter: a team-season needs >= 50 receiving plays in cfbfastR
PBP to count toward breakout calculations (filters out incomplete coverage).

WR threshold: 20% dominator
TE threshold: 15% dominator
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from cfbfastr_fill_gaps import _expand_first  # type: ignore

MIN_TEAM_RECV_PLAYS = 50
PBP_DIR = Path('/tmp/cfb_pbp')


def strict_match(api_name, target):
    a = str(api_name).lower().strip().replace(".", " ").replace("'", "")
    b = str(target).lower().strip().replace(".", " ").replace("'", "")
    suf = {"jr","ii","iii","iv","v","sr"}
    pa = [p for p in a.split() if p not in suf]
    pb = [p for p in b.split() if p not in suf]
    if not pa or not pb: return False
    if pa[-1] != pb[-1]: return False
    return _expand_first(pa[0]) == _expand_first(pb[0])


# Load all available PBP years
print("Loading PBP files...")
pbp = {}
for yr in range(2014, 2026):
    p = PBP_DIR / f'ps_{yr}.parquet'
    if p.exists():
        pbp[yr] = pd.read_parquet(p)
print(f"  loaded years: {sorted(pbp.keys())}")

team_totals = {}
for yr, df in pbp.items():
    rec_only = df[df['reception_player'].notna()]
    grp = rec_only.groupby('team').agg(
        team_rec_yds=('reception_yds', 'sum'),
        team_rec_tds=('touchdown_player', lambda s: s.notna().sum()),
        n_recv_plays=('reception_player', 'count'),
    )
    team_totals[yr] = grp


def find_seasons(name, years):
    out = []
    for yr in years:
        if yr not in pbp:
            continue
        df = pbp[yr]
        recv = df[df['reception_player'].notna()]
        matches = recv[recv['reception_player'].apply(lambda x: strict_match(x, name))]
        if matches.empty: continue
        for team, sub in matches.groupby('team'):
            ry = float(sub['reception_yds'].sum())
            rt = int(sub['touchdown_player'].notna().sum())
            tt = team_totals[yr]
            if team not in tt.index: continue
            ty = tt.loc[team, 'team_rec_yds']
            tt_ = tt.loc[team, 'team_rec_tds']
            np_team = int(tt.loc[team, 'n_recv_plays'])
            if ry == 0: continue
            yd_share = ry/ty if ty > 0 else 0
            td_share = rt/tt_ if tt_ > 0 else 0
            dom = (yd_share + td_share)/2 * 100
            out.append({
                'season': yr, 'team': team,
                'rec_yds': float(ry), 'rec_tds': int(rt),
                'team_rec_yds': float(ty), 'team_rec_tds': int(tt_),
                'team_recv_plays': np_team,
                'dominator_pct': round(dom, 2),
                'reliable': np_team >= MIN_TEAM_RECV_PLAYS,
            })
    return sorted(out, key=lambda x: x['season'])


def compute_breakout(seasons, threshold, birth_year):
    reliable = [s for s in seasons if s['reliable']]
    breakout_yr = None
    breakout_age = None
    for s in reliable:
        if s['dominator_pct'] >= threshold:
            breakout_yr = s['season']
            breakout_age = s['season'] - birth_year
            break
    peak = max((s['dominator_pct'] for s in reliable), default=None)
    return breakout_age, breakout_yr, peak, reliable


# ===========================================================================
# 1) TE 2026 — full recompute (mirror of WR 2026 recompute)
# ===========================================================================
print("\n" + "="*78)
print("TE 2026 PROSPECTS — full recompute")
print("="*78)

te26 = pd.read_csv(ROOT / 'data' / 'te_2026_prospects_final.csv')
te_changes = []
te_log_rows = []

for idx, r in te26.iterrows():
    name = r['player_name']
    # Birth year: prefer birthdate column, else 2026 - draft_age
    birth_year = None
    if pd.notna(r.get('birthdate')) and str(r['birthdate']) != 'estimated':
        try:
            birth_year = int(str(r['birthdate'])[:4])
        except: pass
    if birth_year is None and pd.notna(r.get('draft_age')):
        try:
            birth_year = 2026 - int(round(float(r['draft_age'])))
        except: pass
    if birth_year is None:
        continue

    seasons = find_seasons(name, range(2020, 2026))
    if not seasons:
        continue

    new_bo, new_byr, new_peak, reliable = compute_breakout(seasons, 15.0, birth_year)
    old_bo = r.get('breakout_age') if pd.notna(r.get('breakout_age')) else None
    old_peak = r.get('peak_dominator') if pd.notna(r.get('peak_dominator')) else None
    old_n = int(r.get('seasons_found', 0)) if pd.notna(r.get('seasons_found', 0)) else 0

    more_evidence = len(reliable) > old_n + 1
    should_update = more_evidence or old_bo is None or pd.isna(old_bo)

    if should_update:
        te26.at[idx, 'breakout_age'] = new_bo
        te26.at[idx, 'peak_dominator'] = new_peak
        if 'seasons_found' in te26.columns:
            te26.at[idx, 'seasons_found'] = len(reliable)
        if 'breakout_season' in te26.columns:
            te26.at[idx, 'breakout_season'] = new_byr
        te_log_rows.append({
            'name': name, 'birth_year': birth_year,
            'old_bo': old_bo, 'new_bo': new_bo,
            'old_peak': old_peak, 'new_peak': new_peak,
            'seasons': seasons,
        })
        if old_bo != new_bo:
            te_changes.append((name, old_bo, new_bo))

te26.to_csv(ROOT / 'data' / 'te_2026_prospects_final.csv', index=False)
print(f"  Updated {len(te_log_rows)} rows; breakout_age changes: {len(te_changes)}")
for n, ob, nb in te_changes[:15]:
    ob_s = f"{int(ob)}" if ob is not None and not pd.isna(ob) else 'NaN'
    nb_s = f"{int(nb)}" if nb is not None else 'NaN'
    print(f"    {n:<25}  {ob_s} -> {nb_s}")
if len(te_changes) > 15:
    print(f"    ... and {len(te_changes)-15} more")


# ===========================================================================
# 2) WR backtest — NaN fill only (don't change validated values)
# ===========================================================================
print("\n" + "="*78)
print("WR BACKTEST — NaN fill only")
print("="*78)

wr_bt = pd.read_csv(ROOT / 'data' / 'wr_backtest_all_components.csv')
nan_mask = wr_bt['breakout_age'].isna()
to_fill = wr_bt[nan_mask]
print(f"  NaN breakout_age: {len(to_fill)} of {len(wr_bt)}")

filled_wr_bt = 0
no_data = 0
no_break = 0
wr_bt_log = []
for idx, r in to_fill.iterrows():
    name = r['player_name']
    draft_year = int(r['draft_year'])
    draft_age = r.get('draft_age', 22)
    if pd.isna(draft_age):
        draft_age = 22
    birth_year = draft_year - int(round(float(draft_age)))
    seasons = find_seasons(name, range(draft_year - 5, draft_year))
    if not seasons:
        no_data += 1
        continue
    new_bo, new_byr, new_peak, reliable = compute_breakout(seasons, 20.0, birth_year)
    if new_bo is not None:
        wr_bt.at[idx, 'breakout_age'] = new_bo
        # Don't overwrite peak_dominator if already set (it might be from a better source)
        if pd.isna(r.get('peak_dominator')) and new_peak is not None:
            wr_bt.at[idx, 'peak_dominator'] = new_peak
        filled_wr_bt += 1
        wr_bt_log.append({'name': name, 'new_bo': new_bo, 'breakout_yr': new_byr,
                          'team': next(s['team'] for s in reliable if s['season']==new_byr)})
    else:
        no_break += 1

wr_bt.to_csv(ROOT / 'data' / 'wr_backtest_all_components.csv', index=False)
print(f"  Filled: {filled_wr_bt}")
print(f"  Confirmed never hit 20% (correct NaN): {no_break}")
print(f"  No PBP data: {no_data}")
for r in wr_bt_log:
    print(f"    {r['name']:<25}  age {r['new_bo']} @ {r['team']} ({r['breakout_yr']})")


# ===========================================================================
# 3) TE backtest — NaN fill only (already partially done by backfill_te_breakout_age.py)
# ===========================================================================
print("\n" + "="*78)
print("TE BACKTEST — NaN fill (catching cross-team transfers we missed)")
print("="*78)

te_bt = pd.read_csv(ROOT / 'data' / 'te_backtest_master.csv')
nan_mask = te_bt['breakout_age'].isna()
to_fill = te_bt[nan_mask]
print(f"  Currently NaN breakout_age: {len(to_fill)} of {len(te_bt)}")

filled_te_bt = 0
no_data_te = 0
no_break_te = 0
te_bt_log = []
for idx, r in to_fill.iterrows():
    name = r['player_name']
    draft_year = int(r['draft_year'])
    draft_age = r.get('draft_age', 23)
    if pd.isna(draft_age):
        draft_age = 23
    birth_year = draft_year - int(round(float(draft_age)))
    seasons = find_seasons(name, range(draft_year - 5, draft_year))
    if not seasons:
        no_data_te += 1
        continue
    new_bo, new_byr, new_peak, reliable = compute_breakout(seasons, 15.0, birth_year)
    if new_bo is not None:
        te_bt.at[idx, 'breakout_age'] = new_bo
        if 'breakout_season' in te_bt.columns:
            te_bt.at[idx, 'breakout_season'] = new_byr
        if pd.isna(r.get('peak_dominator')) and new_peak is not None:
            te_bt.at[idx, 'peak_dominator'] = new_peak
        filled_te_bt += 1
        te_bt_log.append({'name': name, 'new_bo': new_bo, 'breakout_yr': new_byr,
                          'team': next(s['team'] for s in reliable if s['season']==new_byr)})
    else:
        no_break_te += 1

te_bt.to_csv(ROOT / 'data' / 'te_backtest_master.csv', index=False)
print(f"  Filled: {filled_te_bt}")
print(f"  Confirmed never hit 15% (correct NaN): {no_break_te}")
print(f"  No PBP data: {no_data_te}")
for r in te_bt_log:
    print(f"    {r['name']:<25}  age {r['new_bo']} @ {r['team']} ({r['breakout_yr']})")

# Write detailed audit
audit_path = ROOT / 'output' / 'breakout_audit_all_positions.txt'
with open(audit_path, 'w') as fh:
    fh.write("Breakout audit — all positions\n")
    fh.write("="*78 + "\n\n")
    fh.write("TE 2026 changes:\n")
    for log in te_log_rows:
        old_s = f"{int(log['old_bo'])}" if log['old_bo'] is not None and not pd.isna(log['old_bo']) else 'NaN'
        new_s = f"{int(log['new_bo'])}" if log['new_bo'] is not None else 'NaN'
        if old_s != new_s:
            fh.write(f"\n  {log['name']} (born {log['birth_year']}): bo {old_s} -> {new_s}\n")
            for s in log['seasons']:
                tag = ' <- HIT 15%' if s['dominator_pct'] >= 15 else ''
                rel = '' if s['reliable'] else ' [INCOMPLETE]'
                fh.write(f"    {s['season']} (age {s['season']-log['birth_year']}) | "
                         f"{s['team']:<22} | rec={int(s['rec_yds']):>4}, "
                         f"team_plays={s['team_recv_plays']} | dom={s['dominator_pct']:.1f}%"
                         f"{tag}{rel}\n")
    fh.write(f"\n\nWR backtest fills: {filled_wr_bt}\n")
    for r in wr_bt_log:
        fh.write(f"  {r['name']:<25}  age {r['new_bo']} @ {r['team']} ({r['breakout_yr']})\n")
    fh.write(f"\n\nTE backtest fills: {filled_te_bt}\n")
    for r in te_bt_log:
        fh.write(f"  {r['name']:<25}  age {r['new_bo']} @ {r['team']} ({r['breakout_yr']})\n")
print(f"\nDetailed audit: {audit_path}")
