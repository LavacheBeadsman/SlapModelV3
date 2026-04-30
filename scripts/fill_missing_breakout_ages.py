"""Fill breakout_age for players where peak_dominator >= threshold but BO is NaN.

These are players who clearly broke out (per their peak dominator) but the
age wasn't recorded. Search PBP at any team across all years for their first
season hitting the threshold. Skip the reliability filter — if peak data
already shows breakout, accept any season's PBP as confirmation.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from cfbfastr_fill_gaps import _expand_first  # type: ignore


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
print("Loading PBP...")
pbp = {yr: pd.read_parquet(PBP_DIR / f'ps_{yr}.parquet')
       for yr in range(2014, 2026) if (PBP_DIR / f'ps_{yr}.parquet').exists()}

team_totals = {}
for yr, df in pbp.items():
    rec_only = df[df['reception_player'].notna()]
    grp = rec_only.groupby('team').agg(
        team_rec_yds=('reception_yds', 'sum'),
        team_rec_tds=('touchdown_player', lambda s: s.notna().sum()),
        n_recv_plays=('reception_player', 'count'),
    )
    team_totals[yr] = grp


def find_first_breakout(name, years, threshold):
    """Return (season, age_offset, dom%) for first season player hits threshold,
    OR closest-to-threshold season if none hit. NO RELIABILITY FILTER."""
    candidates = []
    for yr in years:
        if yr not in pbp:
            continue
        df = pbp[yr]
        recv = df[df['reception_player'].notna()]
        matches = recv[recv['reception_player'].apply(lambda x: strict_match(x, name))]
        if matches.empty:
            continue
        for team, sub in matches.groupby('team'):
            ry = float(sub['reception_yds'].sum())
            rt = int(sub['touchdown_player'].notna().sum())
            tt = team_totals[yr]
            if team not in tt.index: continue
            ty = tt.loc[team, 'team_rec_yds']
            tt_ = tt.loc[team, 'team_rec_tds']
            if ry == 0 or ty == 0:
                continue
            yd_share = ry/ty if ty > 0 else 0
            td_share = rt/tt_ if tt_ > 0 else 0
            dom = (yd_share + td_share)/2 * 100
            candidates.append({'season': yr, 'team': team, 'dom': dom,
                               'rec_yds': ry, 'team_yds': ty})

    if not candidates:
        return None, None
    # Sort chronologically; first season above threshold wins
    candidates.sort(key=lambda x: x['season'])
    for c in candidates:
        if c['dom'] >= threshold:
            return c['season'], c
    # No threshold hit — return the highest dominator season
    best = max(candidates, key=lambda x: x['dom'])
    return best['season'], best


def fill_wr_2026():
    print("\n" + "="*78)
    print("WR 2026 — fill breakout_age where peak_dominator >= 20%")
    print("="*78)
    bo_path = ROOT / 'data' / 'wr_breakout_ages_2026.csv'
    bo = pd.read_csv(bo_path)
    p = pd.read_csv(ROOT / 'data' / 'prospects_final.csv')

    candidates = bo[bo['breakout_age'].isna() & (bo['peak_dominator'] >= 20)]
    print(f"  candidates: {len(candidates)}")

    filled = 0
    for idx, r in candidates.iterrows():
        name = r['player_name']
        # Birth year: try birthdate -> prospects birthdate -> draft_age -> infer from PBP
        birth_year = None
        bd = r.get('birthdate')
        if pd.notna(bd) and str(bd) not in ('MISSING', 'estimated'):
            try: birth_year = int(str(bd)[:4])
            except: pass
        if birth_year is None:
            prow = p[p['player_name']==name]
            if not prow.empty:
                pbd = prow.iloc[0].get('birthdate')
                if pd.notna(pbd) and str(pbd) not in ('MISSING', 'estimated'):
                    try: birth_year = int(str(pbd)[:4])
                    except: pass
                if birth_year is None:
                    page = prow.iloc[0].get('age')
                    if pd.notna(page) and str(page) != 'MISSING':
                        try: birth_year = 2026 - int(page)
                        except: pass
        if birth_year is None and pd.notna(r.get('draft_age')):
            try: birth_year = 2026 - int(round(float(r['draft_age'])))
            except: pass

        season, info = find_first_breakout(name, range(2020, 2026), 20.0)
        if season is None:
            print(f"  ✗ {name}: not found in PBP")
            continue

        # Last fallback: estimate birth_year by assuming player was a freshman (age 18)
        # in their FIRST PBP-appearance season. Widely-used industry default.
        estimated = False
        if birth_year is None:
            # Find earliest season for this player
            first_season = None
            for yr in range(2020, 2026):
                if yr not in pbp:
                    continue
                df = pbp[yr]
                recv = df[df['reception_player'].notna()]
                if recv['reception_player'].apply(lambda x: strict_match(x, name)).any():
                    first_season = yr
                    break
            if first_season is not None:
                birth_year = first_season - 18
                estimated = True
            else:
                print(f"  ✗ {name}: no birth year, no PBP")
                continue

        age = season - birth_year
        bo.at[idx, 'breakout_age'] = age
        if 'breakout_season' in bo.columns:
            bo.at[idx, 'breakout_season'] = season
        filled += 1
        tag = ' [age estimated from earliest PBP year]' if estimated else ''
        print(f"  ✓ {name}: age {age} @ {info['team']} ({season}, {info['dom']:.1f}%){tag}")

    bo.to_csv(bo_path, index=False)
    print(f"  filled {filled}/{len(candidates)}")


def fill_te_backtest():
    print("\n" + "="*78)
    print("TE backtest — fill breakout_age where peak_dominator >= 15%")
    print("="*78)
    te_path = ROOT / 'data' / 'te_backtest_master.csv'
    te = pd.read_csv(te_path)
    candidates = te[te['breakout_age'].isna() & (te['peak_dominator'] >= 15)]
    print(f"  candidates: {len(candidates)}")

    filled = 0
    for idx, r in candidates.iterrows():
        name = r['player_name']
        draft_year = int(r['draft_year'])
        draft_age = r.get('draft_age', 23)
        if pd.isna(draft_age): draft_age = 23
        birth_year = draft_year - int(round(float(draft_age)))

        season, info = find_first_breakout(name, range(draft_year - 5, draft_year), 15.0)
        if season is None:
            print(f"  ✗ {name}: not found in PBP")
            continue
        age = season - birth_year
        te.at[idx, 'breakout_age'] = age
        if 'breakout_season' in te.columns:
            te.at[idx, 'breakout_season'] = season
        filled += 1
        print(f"  ✓ {name}: age {age} @ {info['team']} ({season}, {info['dom']:.1f}%)")

    te.to_csv(te_path, index=False)
    print(f"  filled {filled}/{len(candidates)}")


if __name__ == '__main__':
    fill_wr_2026()
    fill_te_backtest()
    print("\nDone. Now run: python src/build_master_database_v5.py")
