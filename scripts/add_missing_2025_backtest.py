"""Add the 28 missing 2025 WR/RB players to backtest for fuller validation.

For each missing player:
  - College stats (rec_yards, team_pass_att, peak_dominator, breakout_age) from cfbfastR PBP
  - NFL outcomes (career_ppg, hit24, hit12, etc.) from nflverse 2025 stats

Updates:
  data/wr_backtest_all_components.csv
  data/wr_backtest_with_production.csv
  data/rb_backtest_with_receiving.csv
  data/backtest_outcomes_complete.csv
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from cfbfastr_fill_gaps import _expand_first  # type: ignore


def strict_match(api_name, target):
    a = str(api_name).lower().strip().replace(".", " ").replace("'", "").replace("-", " ")
    b = str(target).lower().strip().replace(".", " ").replace("'", "").replace("-", " ")
    suf = {"jr","ii","iii","iv","v","sr"}
    pa = [p for p in a.split() if p not in suf]
    pb = [p for p in b.split() if p not in suf]
    if not pa or not pb: return False
    if pa[-1] != pb[-1]: return False
    return _expand_first(pa[0]) == _expand_first(pb[0])


PBP_DIR = Path('/tmp/cfb_pbp')
print("Loading cfbfastR PBP...")
pbp = {yr: pd.read_parquet(PBP_DIR / f'ps_{yr}.parquet')
       for yr in range(2019, 2026) if (PBP_DIR / f'ps_{yr}.parquet').exists()}

team_totals = {}
for yr, df in pbp.items():
    rec_only = df[df['reception_player'].notna()]
    grp = rec_only.groupby('team').agg(
        team_rec_yds=('reception_yds', 'sum'),
        team_rec_tds=('touchdown_player', lambda s: s.notna().sum()),
        n_recv_plays=('reception_player', 'count'),
    )
    team_totals[yr] = grp
    # Pass attempts
    pa = df[(df['completion_player'].notna()) | (df['incompletion_player'].notna()) |
             (df['interception_thrown_player'].notna())]
    pass_att = pa.groupby('team').size()
    team_totals[yr]['team_pass_att'] = pass_att.reindex(team_totals[yr].index, fill_value=0)


def find_player_seasons(name, draft_year, birth_year):
    """Returns list of season dicts."""
    out = []
    for yr in range(draft_year - 5, draft_year):
        if yr not in pbp: continue
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
            tpa = int(tt.loc[team, 'team_pass_att'])
            if ry == 0: continue
            yd_share = ry/ty if ty > 0 else 0
            td_share = rt/tt_ if tt_ > 0 else 0
            dom = (yd_share + td_share)/2 * 100
            age = yr - birth_year if birth_year else None
            out.append({'season': yr, 'team': team, 'rec_yds': ry, 'rec_tds': rt,
                         'team_yds': ty, 'team_tds': tt_, 'team_pass_att': tpa,
                         'dominator_pct': round(dom, 2), 'age': age,
                         'reliable': tt.loc[team, 'n_recv_plays'] >= 50})
    return sorted(out, key=lambda x: x['season'])


# Load data sources
draft = pd.read_parquet(ROOT / 'data' / 'nflverse' / 'draft_picks.parquet')
nfl_2025 = pd.read_csv(ROOT / 'data' / 'nflverse' / 'player_season_stats_2025.csv')

# Load existing files
wr_bt = pd.read_csv(ROOT / 'data' / 'wr_backtest_all_components.csv')
wr_prod = pd.read_csv(ROOT / 'data' / 'wr_backtest_with_production.csv')
rb_bt = pd.read_csv(ROOT / 'data' / 'rb_backtest_with_receiving.csv')
outcomes = pd.read_csv(ROOT / 'data' / 'backtest_outcomes_complete.csv')

WR_DOM_THRESHOLD = 20.0


def compute_hit_thresholds(pos):
    """Top-24 and Top-12 PPG thresholds for a given position."""
    sub = nfl_2025[nfl_2025['position']==pos].copy()
    # Use season totals: must have played >= 8 games for "stable" season
    sub = sub[sub['games'] >= 8]
    sub = sub.sort_values('fantasy_ppg', ascending=False)
    top24 = sub['fantasy_ppg'].iloc[23] if len(sub) >= 24 else 0
    top12 = sub['fantasy_ppg'].iloc[11] if len(sub) >= 12 else 0
    return top24, top12


wr_t24, wr_t12 = compute_hit_thresholds('WR')
rb_t24, rb_t12 = compute_hit_thresholds('RB')
print(f"\n2025 NFL hit thresholds:")
print(f"  WR: Top-24 PPG = {wr_t24:.2f}, Top-12 PPG = {wr_t12:.2f}")
print(f"  RB: Top-24 PPG = {rb_t24:.2f}, Top-12 PPG = {rb_t12:.2f}")


def get_nfl_outcomes(pfr_name, pos, t24, t12):
    """Find player's 2025 NFL stats."""
    # First try direct match on player_display_name
    sub = nfl_2025[nfl_2025['player_display_name'].str.contains(pfr_name.split()[-1], case=False, na=False)]
    sub = sub[sub['player_display_name'].apply(lambda x: strict_match(x, pfr_name))]
    if sub.empty:
        return None
    r = sub.iloc[0]
    games = r['games']
    ppg = r['fantasy_ppg']
    ppr = r['fantasy_points_ppr']
    if pd.isna(ppg) or games == 0:
        return None
    hit24 = 1 if ppg >= t24 else 0
    hit12 = 1 if ppg >= t12 else 0
    return {'best_ppr': float(ppr), 'best_ppg': float(ppg), 'games': int(games),
             'hit24': hit24, 'hit12': hit12,
             'first_3yr_ppg': float(ppg),  # only 1 year of data
             'career_ppg': float(ppg),
             'seasons_over_10ppg': 1 if ppg >= 10 else 0,
             'seasons_over_10ppg_3yr': 1 if ppg >= 10 else 0,
             'nfl_games': int(games)}


def add_wr(name, pick, college, age):
    """Add a missing WR to all relevant files."""
    draft_year = 2025
    # birthdate from draft data: age is at draft time (April), so birthyear ~= 2003 if age 22
    birth_year = draft_year - int(age) if not pd.isna(age) else 2002

    # College stats from cfbfastR
    seasons = find_player_seasons(name, draft_year, birth_year)
    reliable_seasons = [s for s in seasons if s['reliable']]
    peak_dom = max((s['dominator_pct'] for s in reliable_seasons), default=None)
    breakout_age = None
    breakout_season = None
    for s in reliable_seasons:
        if s['dominator_pct'] >= WR_DOM_THRESHOLD:
            breakout_age = s['season'] - birth_year
            breakout_season = s['season']
            break

    # Final season stats (for production file)
    final_season = max((s['season'] for s in seasons), default=None)
    final = next((s for s in seasons if s['season']==final_season), None)
    rec_yds_final = final['rec_yds'] if final else None
    team_pass_att_final = final['team_pass_att'] if final else None

    # NFL outcomes
    nfl = get_nfl_outcomes(name, 'WR', wr_t24, wr_t12)
    if nfl is None:
        nfl = {'best_ppr': 0.0, 'best_ppg': 0.0, 'games': 0, 'hit24': 0, 'hit12': 0,
                'first_3yr_ppg': 0.0, 'career_ppg': 0.0,
                'seasons_over_10ppg': 0, 'seasons_over_10ppg_3yr': 0, 'nfl_games': 0}

    return {
        'wr_bt_row': {
            'player_name': name, 'position': 'WR', 'draft_year': draft_year,
            'pick': int(pick), 'round': pick_to_round(pick), 'college': college,
            'seasons_played': 1, 'best_rank': np.nan, 'best_ppr': nfl['best_ppr'],
            'hit24': nfl['hit24'], 'hit12': nfl['hit12'], 'RAS': np.nan,
            'breakout_age': breakout_age, 'peak_dominator': peak_dom,
            'birthdate': f'{birth_year}-01-01' if birth_year else None,
            'declare_status': None, 'early_declare': None,
            'draft_age': age, 'declare_source': None,
            'rush_attempts': np.nan, 'rush_yards': np.nan, 'rush_touchdowns': np.nan,
            'rush_games': nfl['games'], 'rush_source': None, 'rush_tds': np.nan,
        },
        'wr_prod_row': {
            'player_name': name, 'draft_year': draft_year, 'pick': int(pick),
            'college': college, 'rec_yards': rec_yds_final, 'team_pass_att': team_pass_att_final,
        },
        'outcomes_row': {
            'player_name': name, 'position': 'WR', 'draft_year': draft_year, 'pick': int(pick),
            **{k: nfl[k] for k in ['hit24','hit12','first_3yr_ppg','career_ppg','seasons_over_10ppg','seasons_over_10ppg_3yr','nfl_games']}
        },
    }


def add_rb(name, pick, college, age):
    draft_year = 2025
    birth_year = draft_year - int(age) if not pd.isna(age) else 2003

    seasons = find_player_seasons(name, draft_year, birth_year)
    reliable_seasons = [s for s in seasons if s['reliable']]
    peak_dom = max((s['dominator_pct'] for s in reliable_seasons), default=None)
    final_season = max((s['season'] for s in seasons), default=None)
    final = next((s for s in seasons if s['season']==final_season), None)
    rec_yds_final = final['rec_yds'] if final else None
    rec_tds_final = final['rec_tds'] if final else None
    team_pass_att_final = final['team_pass_att'] if final else None

    nfl = get_nfl_outcomes(name, 'RB', rb_t24, rb_t12)
    if nfl is None:
        nfl = {'best_ppr': 0.0, 'best_ppg': 0.0, 'games': 0, 'hit24': 0, 'hit12': 0,
                'first_3yr_ppg': 0.0, 'career_ppg': 0.0,
                'seasons_over_10ppg': 0, 'seasons_over_10ppg_3yr': 0, 'nfl_games': 0}

    return {
        'rb_bt_row': {
            'player_name': name, 'draft_year': draft_year, 'pick': int(pick),
            'round': pick_to_round(pick), 'college': college, 'age': age,
            'best_ppr': nfl['best_ppr'], 'best_ppg': nfl['best_ppg'],
            'best_season': 2025.0 if nfl['best_ppg'] > 0 else None,
            'season_rank': np.nan,
            'hit24': nfl['hit24'], 'hit12': nfl['hit12'], 'RAS': np.nan,
            'rec_yards': rec_yds_final, 'receptions': np.nan,
            'team_pass_att': team_pass_att_final, 'cfbd_name': None,
        },
        'outcomes_row': {
            'player_name': name, 'position': 'RB', 'draft_year': draft_year, 'pick': int(pick),
            **{k: nfl[k] for k in ['hit24','hit12','first_3yr_ppg','career_ppg','seasons_over_10ppg','seasons_over_10ppg_3yr','nfl_games']}
        },
    }


def pick_to_round(pick):
    if pick <= 32: return 1
    elif pick <= 64: return 2
    elif pick <= 100: return 3
    elif pick <= 140: return 4
    elif pick <= 180: return 5
    elif pick <= 220: return 6
    else: return 7


# Identify missing
def is_in_backtest(name, pos):
    if pos == 'WR':
        existing = wr_bt[(wr_bt['draft_year']==2025) & (wr_bt['position']=='WR')]['player_name'].tolist()
    else:
        existing = rb_bt[rb_bt['draft_year']==2025]['player_name'].tolist()
    return any(strict_match(name, e) for e in existing)


print("\n" + "="*80)
print("Adding missing 2025 players...")
print("="*80)

new_wr_bt, new_wr_prod, new_outcomes_wr = [], [], []
new_rb_bt, new_outcomes_rb = [], []

for _, r in draft[(draft['season']==2025) & (draft['position'].isin(['WR','RB']))].iterrows():
    name = r['pfr_player_name']
    pos = r['position']
    pick = r['pick']
    college = r['college']
    age = r['age']
    if is_in_backtest(name, pos):
        continue
    print(f"  + {pos} pick {int(pick):>3} {name:<25} ({college})")
    if pos == 'WR':
        rows = add_wr(name, pick, college, age)
        new_wr_bt.append(rows['wr_bt_row'])
        new_wr_prod.append(rows['wr_prod_row'])
        new_outcomes_wr.append(rows['outcomes_row'])
    else:
        rows = add_rb(name, pick, college, age)
        new_rb_bt.append(rows['rb_bt_row'])
        new_outcomes_rb.append(rows['outcomes_row'])

# Append rows and save
if new_wr_bt:
    wr_bt = pd.concat([wr_bt, pd.DataFrame(new_wr_bt)], ignore_index=True)
    wr_bt.to_csv(ROOT / 'data' / 'wr_backtest_all_components.csv', index=False)
    print(f"\nAdded {len(new_wr_bt)} WRs to wr_backtest_all_components.csv")
if new_wr_prod:
    wr_prod = pd.concat([wr_prod, pd.DataFrame(new_wr_prod)], ignore_index=True)
    wr_prod.to_csv(ROOT / 'data' / 'wr_backtest_with_production.csv', index=False)
    print(f"Added {len(new_wr_prod)} rows to wr_backtest_with_production.csv")
if new_rb_bt:
    rb_bt = pd.concat([rb_bt, pd.DataFrame(new_rb_bt)], ignore_index=True)
    rb_bt.to_csv(ROOT / 'data' / 'rb_backtest_with_receiving.csv', index=False)
    print(f"Added {len(new_rb_bt)} RBs to rb_backtest_with_receiving.csv")
all_outcomes = new_outcomes_wr + new_outcomes_rb
if all_outcomes:
    outcomes = pd.concat([outcomes, pd.DataFrame(all_outcomes)], ignore_index=True)
    outcomes.to_csv(ROOT / 'data' / 'backtest_outcomes_complete.csv', index=False)
    print(f"Added {len(all_outcomes)} rows to backtest_outcomes_complete.csv")

print("\nDone. Now run: python src/build_master_database_v5.py")
