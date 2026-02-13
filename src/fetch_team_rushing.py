"""
Fetch team rushing totals from CFBD for each WR's college team/final season.
Save to data/wr_team_rushing.csv
"""

import pandas as pd
import requests
import time

API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
BASE = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

TEAM_NAME_MAP = {
    'Mississippi': 'Ole Miss', 'Ala-Birmingham': 'UAB',
    'Central Florida': 'UCF', 'Miami (FL)': 'Miami',
    'Miami (Ohio)': 'Miami (OH)', 'East. Michigan': 'Eastern Michigan',
    'West. Michigan': 'Western Michigan', 'N.C. State': 'NC State',
    'La-Monroe': 'Louisiana Monroe', 'Middle Tenn. St.': 'Middle Tennessee',
    'Michigan St.': 'Michigan State', 'Colorado St.': 'Colorado State',
    'Tenn-Martin': 'UT Martin', 'Bowling Green St.': 'Bowling Green',
    'Ohio St.': 'Ohio State', 'Oregon St.': 'Oregon State',
    'Iowa St.': 'Iowa State', 'Oklahoma St.': 'Oklahoma State',
    'San Diego St.': 'San Diego State', 'Boise St.': 'Boise State',
    'Arizona St.': 'Arizona State', 'Fresno St.': 'Fresno State',
    'SE Missouri St.': 'Southeast Missouri State',
    'North Dakota St.': 'North Dakota State', 'Penn St.': 'Penn State',
    'Boston Col.': 'Boston College', 'Northern Illinois': 'Northern Illinois',
    'Washington State': 'Washington State', 'Kansas St.': 'Kansas State',
    'Florida St.': 'Florida State', 'Georgia St.': 'Georgia State',
    'North Carolina St.': 'NC State', 'New Mexico St.': 'New Mexico State',
    'Grambling St.': 'Grambling',
}

wr = pd.read_csv('data/wr_backtest_all_components.csv')
wr['final_season'] = wr['draft_year'] - 1
print(f"Loaded: {len(wr)} WRs\n")

# Unique team/year pairs
team_years = wr[['college', 'final_season']].drop_duplicates()
print(f"Unique (team, year) pairs: {len(team_years)}")

# Fetch team rushing stats
results = {}
for i, (_, row) in enumerate(team_years.iterrows()):
    college = row['college']
    year = int(row['final_season'])
    cfbd_team = TEAM_NAME_MAP.get(college, college)

    url = f"{BASE}/stats/season"
    params = {'year': year, 'team': cfbd_team}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        data = resp.json() if resp.status_code == 200 else []
    except:
        data = []

    team_rush_yds = None
    team_pass_yds = None
    team_total_yds = None

    for entry in data:
        stat = entry.get('statName', '')
        val = entry.get('statValue', '0')
        try:
            val = float(val)
        except:
            continue
        if stat == 'rushingYards':
            team_rush_yds = val
        elif stat == 'netPassingYards':
            team_pass_yds = val
        elif stat == 'totalYards':
            team_total_yds = val

    results[(college, year)] = {
        'team_rush_yds': team_rush_yds,
        'team_pass_yds': team_pass_yds,
        'team_total_yds': team_total_yds,
        'has_data': team_rush_yds is not None,
    }

    time.sleep(0.12)
    if (i + 1) % 50 == 0:
        print(f"  ... {i+1}/{len(team_years)} done")

print(f"  ... {len(team_years)}/{len(team_years)} done\n")

# Map back to WRs
rows = []
for _, p in wr.iterrows():
    key = (p['college'], int(p['final_season']))
    info = results.get(key, {'team_rush_yds': None, 'team_pass_yds': None,
                              'team_total_yds': None, 'has_data': False})
    rows.append({
        'player_name': p['player_name'],
        'draft_year': p['draft_year'],
        'pick': p['pick'],
        'college': p['college'],
        'final_season': int(p['final_season']),
        'team_rush_yds': info['team_rush_yds'],
        'team_pass_yds': info['team_pass_yds'],
        'team_total_yds': info['team_total_yds'],
        'has_data': info['has_data'],
    })

df = pd.DataFrame(rows)
df.to_csv('data/wr_team_rushing.csv', index=False)

n_data = df['has_data'].sum()
n_no = (~df['has_data']).sum()
print(f"COVERAGE:")
print(f"  With team rushing data: {n_data} ({n_data/len(df)*100:.1f}%)")
print(f"  Missing: {n_no} ({n_no/len(df)*100:.1f}%)")

if n_no > 0:
    missing = df[~df['has_data']][['player_name', 'draft_year', 'college']].drop_duplicates('college')
    print(f"\n  Missing schools:")
    for _, m in missing.iterrows():
        print(f"    {m['college']} — {m['player_name']} ({m['draft_year']})")

# Sanity check some known values
print(f"\n  Sanity checks (team rushing yards):")
for name, yr in [('Amari Cooper', 2015), ('CeeDee Lamb', 2020), ('Ja\'Marr Chase', 2021)]:
    row = df[(df['player_name'] == name) & (df['draft_year'] == yr)]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"    {name} ({yr}): {r['college']} {int(r['final_season'])} — "
              f"rush={r['team_rush_yds']}, pass={r['team_pass_yds']}, total={r['team_total_yds']}")

print("\nDone.")
