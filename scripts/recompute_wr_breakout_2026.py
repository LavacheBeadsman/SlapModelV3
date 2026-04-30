"""Recompute 2026 WR breakout_age and peak_dominator using cross-team history.

The original wr_breakout_ages_2026.csv was built without proper transfer
tracking — many players have only 1-2 college seasons recorded when they
actually played 4-5 across multiple schools.

This script:
  1. For each WR in the file, searches cfbfastR PBP across 2020-2025 at any team
  2. Applies a reliability filter: team-seasons with <50 receiving plays are
     flagged as incomplete and excluded from the breakout calculation
  3. Recomputes breakout_age (first reliable season hitting 20%) and
     peak_dominator (max across reliable seasons)
  4. Writes back to data/wr_breakout_ages_2026.csv

Caveat: a "breakout" we confirm requires reliable team data. For players whose
publicly-reported breakout came in a season where cfbfastR PBP coverage is
incomplete (e.g., Stribling's 2021 WSU), we'll show NaN rather than a value
derived from a tiny denominator. Other published sources (PFF, full CFBD) may
catch these breakouts via better data we don't have access to.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from cfbfastr_fill_gaps import _expand_first  # type: ignore

DOM_THRESHOLD_WR = 20.0
MIN_TEAM_RECV_PLAYS = 50


def strict_match(api_name, target):
    a = str(api_name).lower().strip().replace(".", " ").replace("'", "")
    b = str(target).lower().strip().replace(".", " ").replace("'", "")
    suf = {"jr","ii","iii","iv","v","sr"}
    pa = [p for p in a.split() if p not in suf]
    pb = [p for p in b.split() if p not in suf]
    if not pa or not pb: return False
    if pa[-1] != pb[-1]: return False
    return _expand_first(pa[0]) == _expand_first(pb[0])


PBP_DIR = Path('/tmp/cfb_pbp')
pbp = {yr: pd.read_parquet(PBP_DIR / f'ps_{yr}.parquet') for yr in range(2019, 2026)}

# Pre-compute team totals
team_totals = {}
for yr, df in pbp.items():
    rec_only = df[df['reception_player'].notna()]
    grp = rec_only.groupby('team').agg(
        team_rec_yds=('reception_yds', 'sum'),
        team_rec_tds=('touchdown_player', lambda s: s.notna().sum()),
        n_recv_plays=('reception_player', 'count'),
    )
    team_totals[yr] = grp


def find_seasons(name):
    out = []
    for yr in range(2020, 2026):
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


def get_birth_year(name, bo, prospects):
    row = bo[bo['player_name']==name]
    if not row.empty and pd.notna(row.iloc[0].get('birthdate')):
        try: return int(str(row.iloc[0]['birthdate'])[:4])
        except: pass
    row = prospects[prospects['player_name']==name]
    if not row.empty and pd.notna(row.iloc[0].get('birthdate')) and str(row.iloc[0]['birthdate']) != 'MISSING':
        try: return int(str(row.iloc[0]['birthdate'])[:4])
        except: pass
    if not row.empty and pd.notna(row.iloc[0].get('age')):
        try: return 2026 - int(row.iloc[0]['age'])
        except: pass
    return None


def main():
    bo = pd.read_csv(ROOT / 'data' / 'wr_breakout_ages_2026.csv')
    prospects = pd.read_csv(ROOT / 'data' / 'prospects_final.csv')

    print(f"Recomputing breakout for {len(bo)} 2026 WRs...")
    print(f"Reliability filter: team-season needs >= {MIN_TEAM_RECV_PLAYS} receiving plays")
    print()

    changes = {'no_change': 0, 'bo_filled': 0, 'bo_changed': 0, 'bo_removed': 0,
                'pd_changed': 0, 'no_data': 0}
    detailed_changes = []

    for idx, r in bo.iterrows():
        name = r['player_name']
        birth_year = get_birth_year(name, bo, prospects)
        if birth_year is None:
            continue
        seasons = find_seasons(name)
        if not seasons:
            changes['no_data'] += 1
            continue

        reliable = [s for s in seasons if s['reliable']]
        # Compute new values from reliable seasons only
        new_bo = None
        new_breakout_season = None
        for s in reliable:
            if s['dominator_pct'] >= DOM_THRESHOLD_WR:
                new_bo = s['season'] - birth_year
                new_breakout_season = s['season']
                break
        new_pd = max((s['dominator_pct'] for s in reliable), default=None)

        old_bo = r['breakout_age'] if pd.notna(r['breakout_age']) else None
        old_pd = r['peak_dominator'] if pd.notna(r['peak_dominator']) else None

        # Only overwrite if old values look definitely worse (had only 1-2 seasons stored)
        # OR if new value is clearly better-supported (more reliable seasons found)
        old_n = int(r['seasons_found']) if pd.notna(r['seasons_found']) else 0
        more_evidence = len(reliable) > old_n + 1

        # If the existing data only has 1 season but we have more reliable seasons, replace
        # If the existing breakout_age was never set, fill it
        # Otherwise keep existing (it might be from a better source)
        should_update = more_evidence or old_bo is None

        log = {
            'name': name, 'birth_year': birth_year,
            'old_bo': old_bo, 'old_pd': old_pd, 'old_n': old_n,
            'new_bo': new_bo, 'new_pd': new_pd, 'new_n': len(reliable),
            'breakout_season': new_breakout_season,
            'updated': should_update,
            'seasons': seasons,
        }

        if should_update:
            bo.at[idx, 'breakout_age'] = new_bo
            bo.at[idx, 'peak_dominator'] = new_pd
            bo.at[idx, 'breakout_season'] = new_breakout_season
            bo.at[idx, 'peak_dominator_season'] = (
                next((s['season'] for s in reliable
                       if s['dominator_pct'] == new_pd), None) if new_pd else None)
            bo.at[idx, 'seasons_found'] = len(reliable)
            bo.at[idx, 'seasons_data'] = json.dumps(seasons)
            if new_bo is not None and old_bo is None:
                changes['bo_filled'] += 1
            elif new_bo is None and old_bo is not None:
                changes['bo_removed'] += 1
            elif new_bo is not None and old_bo is not None and new_bo != old_bo:
                changes['bo_changed'] += 1
            else:
                changes['no_change'] += 1
            detailed_changes.append(log)
        else:
            changes['no_change'] += 1

    bo.to_csv(ROOT / 'data' / 'wr_breakout_ages_2026.csv', index=False)

    print("Summary:")
    for k, v in changes.items():
        print(f"  {k:<14}: {v}")
    print(f"  total players touched: {sum([changes['bo_filled'], changes['bo_changed'], changes['bo_removed']])}")

    # Save audit log
    audit_path = ROOT / 'output' / 'wr_breakout_audit_2026.txt'
    with open(audit_path, 'w') as fh:
        fh.write("2026 WR breakout_age recomputation audit\n")
        fh.write("=" * 78 + "\n")
        fh.write(f"Reliability filter: team-season needs >= {MIN_TEAM_RECV_PLAYS} receiving plays\n\n")
        for log in detailed_changes:
            fh.write(f"\n{log['name']} (born {log['birth_year']})\n")
            old_str = f"{int(log['old_bo'])}" if log['old_bo'] else 'NaN'
            new_str = f"{int(log['new_bo'])}" if log['new_bo'] else 'NaN'
            fh.write(f"  breakout_age: {old_str} -> {new_str}  "
                     f"(stored {log['old_n']} seasons; we found {log['new_n']} reliable)\n")
            for s in log['seasons']:
                tag = ' <- HIT 20%' if s['dominator_pct'] >= 20 else ''
                rel = '' if s['reliable'] else ' [INCOMPLETE — IGNORED]'
                fh.write(f"  {s['season']} (age {s['season']-log['birth_year']}) | "
                         f"{s['team']:<22} | rec={int(s['rec_yds']):>4}, "
                         f"team_plays={s['team_recv_plays']} | dom={s['dominator_pct']:.1f}%"
                         f"{tag}{rel}\n")
    print(f"\nAudit detail: {audit_path}")


if __name__ == '__main__':
    main()
