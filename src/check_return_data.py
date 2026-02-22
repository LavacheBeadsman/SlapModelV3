"""
Return Data Coverage Check — CFBD API
======================================
For each of the 339 WRs, query CFBD for kickReturns and puntReturns
in their FINAL college season.

Reports:
1. Coverage: how many WRs have return data?
2. How many had meaningful returns (5+)?
3. Top 20 by total return yards with hit24 outcomes
"""

import pandas as pd
import numpy as np
import requests
import time
import json

API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
BASE = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

wr = pd.read_csv('data/wr_backtest_all_components.csv')
print(f"Loaded: {len(wr)} WRs\n")

# Final college season = draft_year - 1
wr['final_season'] = wr['draft_year'] - 1

# CFBD uses specific team names — we need a mapping for common mismatches
TEAM_NAME_MAP = {
    'Mississippi': 'Ole Miss',
    'Ala-Birmingham': 'UAB',
    'Central Florida': 'UCF',
    'Miami (FL)': 'Miami',
    'Miami (Ohio)': 'Miami (OH)',
    'East. Michigan': 'Eastern Michigan',
    'West. Michigan': 'Western Michigan',
    'N.C. State': 'NC State',
    'South Florida': 'South Florida',
    'La-Monroe': 'Louisiana Monroe',
    'Middle Tenn. St.': 'Middle Tennessee',
    'Michigan St.': 'Michigan State',
    'Colorado St.': 'Colorado State',
    'Eastern Washington': 'Eastern Washington',
    'Tenn-Martin': 'UT Martin',
    'Bowling Green St.': 'Bowling Green',
    'Ohio St.': 'Ohio State',
    'Oregon St.': 'Oregon State',
    'Iowa St.': 'Iowa State',
    'Oklahoma St.': 'Oklahoma State',
    'Washington State': 'Washington State',
    'San Diego St.': 'San Diego State',
    'Boise St.': 'Boise State',
    'Arizona St.': 'Arizona State',
    'Fresno St.': 'Fresno State',
    'Northern Illinois': 'Northern Illinois',
    'SE Missouri St.': 'Southeast Missouri State',
    'North Dakota St.': 'North Dakota State',
    'Penn St.': 'Penn State',
    'Boston Col.': 'Boston College',
    'Michigan St.': 'Michigan State',
}

def get_cfbd_team(college):
    """Map our college names to CFBD team names."""
    return TEAM_NAME_MAP.get(college, college)


def fetch_returns(team, year, category):
    """Fetch return stats from CFBD."""
    cfbd_team = get_cfbd_team(team)
    url = f"{BASE}/stats/player/season"
    params = {'year': year, 'team': cfbd_team, 'category': category}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        return []
    except:
        return []


# Batch by unique (team, year) pairs to minimize API calls
team_years = wr.groupby(['college', 'final_season']).size().reset_index(name='count')
print(f"Unique (team, year) pairs to query: {len(team_years)}")
print(f"API calls needed: {len(team_years) * 2} (kick + punt for each)\n")

# Cache results by (team, year)
kick_cache = {}
punt_cache = {}

total_calls = len(team_years) * 2
call_num = 0

for _, row in team_years.iterrows():
    college = row['college']
    year = int(row['final_season'])
    key = (college, year)

    call_num += 1
    kick_data = fetch_returns(college, year, 'kickReturns')
    kick_cache[key] = kick_data
    time.sleep(0.12)

    call_num += 1
    punt_data = fetch_returns(college, year, 'puntReturns')
    punt_cache[key] = punt_data
    time.sleep(0.12)

    if call_num % 50 == 0:
        print(f"  ... {call_num}/{total_calls} API calls done")

print(f"  ... {total_calls}/{total_calls} API calls done\n")


# Parse return data for each WR
results = []

for _, player in wr.iterrows():
    name = player['player_name']
    year = int(player['final_season'])
    college = player['college']
    key = (college, year)

    # Try to match player in kick return data
    kr_no, kr_yds, kr_td, kr_long = 0, 0, 0, 0
    kick_data = kick_cache.get(key, [])
    for entry in kick_data:
        pname = entry.get('player', '')
        # Match by last name + rough first name
        if not pname:
            continue
        # Simple name matching
        entry_parts = pname.lower().split()
        player_parts = name.lower().split()
        # Check last name match
        if len(entry_parts) >= 2 and len(player_parts) >= 2:
            if entry_parts[-1] == player_parts[-1]:
                # Last name matches - check first initial or name
                if (entry_parts[0] == player_parts[0] or
                    entry_parts[0][0] == player_parts[0][0]):
                    stat = entry.get('statType', '')
                    val = entry.get('stat', '0')
                    try:
                        val = float(val)
                    except:
                        val = 0
                    if stat == 'NO':
                        kr_no = int(val)
                    elif stat == 'YDS':
                        kr_yds = int(val)
                    elif stat == 'TD':
                        kr_td = int(val)
                    elif stat == 'LONG':
                        kr_long = int(val)

    # Same for punt returns
    pr_no, pr_yds, pr_td, pr_long = 0, 0, 0, 0
    punt_data = punt_cache.get(key, [])
    for entry in punt_data:
        pname = entry.get('player', '')
        if not pname:
            continue
        entry_parts = pname.lower().split()
        player_parts = name.lower().split()
        if len(entry_parts) >= 2 and len(player_parts) >= 2:
            if entry_parts[-1] == player_parts[-1]:
                if (entry_parts[0] == player_parts[0] or
                    entry_parts[0][0] == player_parts[0][0]):
                    stat = entry.get('statType', '')
                    val = entry.get('stat', '0')
                    try:
                        val = float(val)
                    except:
                        val = 0
                    if stat == 'NO':
                        pr_no = int(val)
                    elif stat == 'YDS':
                        pr_yds = int(val)
                    elif stat == 'TD':
                        pr_td = int(val)
                    elif stat == 'LONG':
                        pr_long = int(val)

    total_ret = kr_no + pr_no
    total_ret_yds = kr_yds + pr_yds
    total_ret_td = kr_td + pr_td

    # Did the API have ANY data for this team/year?
    has_team_data = len(kick_data) > 0 or len(punt_data) > 0

    results.append({
        'player_name': name,
        'draft_year': player['draft_year'],
        'pick': player['pick'],
        'round': player['round'],
        'college': college,
        'final_season': year,
        'hit24': player['hit24'],
        'best_ppr': player['best_ppr'],
        'kr_no': kr_no, 'kr_yds': kr_yds, 'kr_td': kr_td,
        'pr_no': pr_no, 'pr_yds': pr_yds, 'pr_td': pr_td,
        'total_returns': total_ret,
        'total_return_yds': total_ret_yds,
        'total_return_td': total_ret_td,
        'has_team_data': has_team_data,
    })

df = pd.DataFrame(results)

# Save for later use
df.to_csv('data/wr_return_stats.csv', index=False)
print("Saved to data/wr_return_stats.csv\n")


# ============================================================================
# COVERAGE REPORT
# ============================================================================

print("=" * 100)
print("COVERAGE REPORT")
print("=" * 100)

n_team_data = df['has_team_data'].sum()
n_no_team = (~df['has_team_data']).sum()
print(f"\n  WRs where CFBD had team data: {n_team_data} ({n_team_data/len(df)*100:.1f}%)")
print(f"  WRs where CFBD had NO team data: {n_no_team} ({n_no_team/len(df)*100:.1f}%)")

if n_no_team > 0:
    no_data = df[~df['has_team_data']][['player_name', 'draft_year', 'college', 'final_season']]
    print(f"\n  Schools with NO return data in CFBD:")
    for college in no_data['college'].unique():
        players = no_data[no_data['college'] == college]
        names = ", ".join(f"{r['player_name']} ({r['draft_year']})" for _, r in players.iterrows())
        print(f"    {college}: {names}")

n_any_returns = (df['total_returns'] > 0).sum()
n_5plus = (df['total_returns'] >= 5).sum()
n_10plus = (df['total_returns'] >= 10).sum()

print(f"\n  Return involvement:")
print(f"    Any returns (1+):           {n_any_returns} WRs ({n_any_returns/len(df)*100:.1f}%)")
print(f"    Meaningful returns (5+):    {n_5plus} WRs ({n_5plus/len(df)*100:.1f}%)")
print(f"    Significant returns (10+):  {n_10plus} WRs ({n_10plus/len(df)*100:.1f}%)")

# Breakdown by return type
n_kr = (df['kr_no'] > 0).sum()
n_pr = (df['pr_no'] > 0).sum()
n_both = ((df['kr_no'] > 0) & (df['pr_no'] > 0)).sum()
print(f"\n  By return type:")
print(f"    Kick returns only:  {n_kr - n_both}")
print(f"    Punt returns only:  {n_pr - n_both}")
print(f"    Both:               {n_both}")
print(f"    Neither:            {len(df) - n_kr - n_pr + n_both}")


# ============================================================================
# TOP 20 BY TOTAL RETURN YARDS
# ============================================================================

print(f"\n\n{'=' * 100}")
print("TOP 20 WRs BY TOTAL RETURN YARDS (final college season)")
print("=" * 100)

top20 = df.nlargest(20, 'total_return_yds')
print(f"\n{'Name':<25} {'Year':>5} {'Pick':>5} {'Rd':>3} {'College':<18} {'KR':>4} {'KRYd':>5} "
      f"{'PR':>4} {'PRYd':>5} {'TotYd':>6} {'TD':>3} {'Hit24':>6} {'PPR':>7}")
print("-" * 115)

for _, r in top20.iterrows():
    hit = f"{r['hit24']:.0f}" if pd.notna(r['hit24']) else "—"
    ppr = f"{r['best_ppr']:.1f}" if pd.notna(r['best_ppr']) else "—"
    print(f"  {r['player_name']:<23} {r['draft_year']:>5} {r['pick']:>5} {r['round']:>3} "
          f"{r['college']:<18} {r['kr_no']:>4} {r['kr_yds']:>5} {r['pr_no']:>4} {r['pr_yds']:>5} "
          f"{r['total_return_yds']:>6} {r['total_return_td']:>3} {hit:>6} {ppr:>7}")

# Hit rate for top 20
n_hits_top20 = top20[top20['hit24'].notna()]['hit24'].sum()
n_eval_top20 = top20['hit24'].notna().sum()
print(f"\n  Top 20 returners: {n_hits_top20:.0f}/{n_eval_top20} hit24 ({n_hits_top20/n_eval_top20*100:.1f}% hit rate)"
      if n_eval_top20 > 0 else "")


# ============================================================================
# HIT24 RATES BY RETURN INVOLVEMENT
# ============================================================================

print(f"\n\n{'=' * 100}")
print("HIT24 RATES BY RETURN INVOLVEMENT")
print("=" * 100)

eval_df = df[df['hit24'].notna()].copy()

buckets = [
    ('No returns (0)', eval_df[eval_df['total_returns'] == 0]),
    ('1-4 returns', eval_df[eval_df['total_returns'].between(1, 4)]),
    ('5-9 returns', eval_df[eval_df['total_returns'].between(5, 9)]),
    ('10-19 returns', eval_df[eval_df['total_returns'].between(10, 19)]),
    ('20+ returns', eval_df[eval_df['total_returns'] >= 20]),
]

print(f"\n{'Bucket':<20} {'Count':>7} {'Hits':>6} {'Hit Rate':>9}")
print("-" * 45)
for label, subset in buckets:
    n = len(subset)
    hits = subset['hit24'].sum()
    rate = hits / n if n > 0 else 0
    print(f"  {label:<18} {n:>7} {hits:>6.0f} {rate:>9.3f}")

# Overall rate for context
overall = eval_df['hit24'].mean()
print(f"\n  Overall hit24 rate: {overall:.3f} ({eval_df['hit24'].sum():.0f}/{len(eval_df)})")

print("\n\nDone.")
