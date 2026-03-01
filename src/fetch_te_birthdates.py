"""
Fetch birthdates for 160 backtest TEs (2015-2025 drafts).

Sources (in priority order):
1. nflbirthdays.com (active NFL players, ~69 TEs)
2. Wikipedia (all drafted players, using existing fetch_birthdates.py pattern)

Output: data/te_backtest_birthdates.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import re
import sys
import os

# Add src to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fetch_birthdates import get_wikipedia_birthdate

# ─── Load TE backtest data ───
def load_te_backtest():
    df = pd.read_csv('output/slap_v5_master_database.csv')
    te_bt = df[(df['position'] == 'TE') & (df['dataset'] == 'backtest')].copy()
    te_bt = te_bt[['player_name', 'college', 'draft_year', 'draft_pick', 'draft_age']].copy()
    te_bt = te_bt.sort_values(['draft_year', 'draft_pick']).reset_index(drop=True)
    return te_bt


# ─── Source 1: nflbirthdays.com ───
def fetch_nflbirthdays():
    """Fetch TE birthdates from nflbirthdays.com HTML table."""
    print("Fetching nflbirthdays.com...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'text/html',
            'Accept-Encoding': 'identity',
        }
        resp = requests.get('https://nflbirthdays.com/', headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}")
            return {}

        html = resp.text
        # Parse the HTML table rows: <tr><td>player</td><td>pos</td><td>team</td><td>birthday</td></tr>
        pattern = r'<tr[^>]*>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>(\d{4}-\d{2}-\d{2})</td>'
        matches = re.findall(pattern, html)

        te_births = {}
        for name, pos, team, birthday in matches:
            if pos.strip() == 'TE':
                clean_name = name.strip()
                te_births[clean_name] = birthday.strip()

        print(f"  Found {len(te_births)} TEs on nflbirthdays.com")
        return te_births
    except Exception as e:
        print(f"  Error: {e}")
        return {}


def normalize_for_match(name):
    """Normalize name for fuzzy matching."""
    s = name.lower().strip()
    s = s.replace('.', '').replace("'", '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s.strip()


def match_nflbirthdays(player_name, nfl_births):
    """Try to match a player name against nflbirthdays.com data."""
    norm = normalize_for_match(player_name)

    for nfl_name, birthday in nfl_births.items():
        if normalize_for_match(nfl_name) == norm:
            return birthday

    # Try last-name + first-initial match
    parts = norm.split()
    if len(parts) >= 2:
        for nfl_name, birthday in nfl_births.items():
            nfl_norm = normalize_for_match(nfl_name)
            nfl_parts = nfl_norm.split()
            if len(nfl_parts) >= 2:
                if parts[-1] == nfl_parts[-1] and parts[0][0] == nfl_parts[0][0]:
                    return birthday

    return None


# ─── Source 2: Wikipedia ───
# Wikipedia name overrides for players whose page differs from our data name
WIKI_NAME_OVERRIDES = {
    "MyCole Pruitt": "Mycole Pruitt",
    "C.J. Uzomah": "C. J. Uzomah",
    "Alizé Mack": "Alize Mack",
    "Tre' McKitty": "Tre' McKitty",
    "T.J. Hockenson": "T. J. Hockenson",
    "O.J. Howard": "O. J. Howard",
    "A.J. Derby": "A.J. Derby (American football)",
    "Dalton Keene": "Dalton Keene",
    "Tyler Davis": "Tyler Davis (tight end)",
    "Ian Thomas": "Ian Thomas (American football)",
    "Jordan Thomas": "Jordan Thomas (tight end)",
    "Bucky Hodges": "Bucky Hodges",
    "David Morgan": "David Morgan (American football)",
    "James Mitchell": "James Mitchell (American football)",
    "Jake Ferguson": "Jake Ferguson (American football)",
    "Nick Muse": "Nick Muse",
    "Davis Allen": "Davis Allen",
    "Zach Davidson": "Zach Davidson (American football)",
    "John Bates": "John Bates (American football)",
    "Luke Farrell": "Luke Farrell (American football)",
}


def fetch_wikipedia_birthdate(player_name):
    """Try Wikipedia with name overrides and disambiguation."""
    # Try override name first
    wiki_name = WIKI_NAME_OVERRIDES.get(player_name, player_name)
    result = get_wikipedia_birthdate(wiki_name)
    if result:
        return result['birthdate']

    # Try with "(American football)" suffix
    if "(American football)" not in wiki_name:
        result = get_wikipedia_birthdate(f"{player_name} (American football)")
        if result:
            return result['birthdate']

    # Try with "(tight end)" suffix
    result = get_wikipedia_birthdate(f"{player_name} (tight end)")
    if result:
        return result['birthdate']

    # Try with "(football)" suffix
    result = get_wikipedia_birthdate(f"{player_name} (football)")
    if result:
        return result['birthdate']

    return None


# ─── Main ───
def main():
    te_bt = load_te_backtest()
    print(f"TEs needing birthdates: {len(te_bt)}")

    # Source 1: nflbirthdays.com
    nfl_births = fetch_nflbirthdays()

    # Source 2: Wikipedia (with rate limiting)
    results = []
    found_nfl = 0
    found_wiki = 0
    not_found = 0

    for idx, row in te_bt.iterrows():
        player = row['player_name']
        draft_year = int(row['draft_year'])

        # Try nflbirthdays.com first
        bd = match_nflbirthdays(player, nfl_births)
        source = 'nflbirthdays'

        if bd is None:
            # Try Wikipedia
            time.sleep(0.4)  # Rate limit
            bd = fetch_wikipedia_birthdate(player)
            source = 'wikipedia'

        if bd:
            if source == 'nflbirthdays':
                found_nfl += 1
            else:
                found_wiki += 1
            print(f"  [{idx+1}/{len(te_bt)}] FOUND {player:30s} → {bd}  ({source})")
        else:
            not_found += 1
            print(f"  [{idx+1}/{len(te_bt)}] MISS  {player:30s}  (draft_age={row['draft_age']})")

        results.append({
            'player_name': player,
            'college': row['college'],
            'draft_year': draft_year,
            'draft_pick': int(row['draft_pick']),
            'draft_age': row['draft_age'],
            'birthdate': bd,
            'source': source if bd else None,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total TEs: {len(te_bt)}")
    print(f"  Found (nflbirthdays): {found_nfl}")
    print(f"  Found (Wikipedia):    {found_wiki}")
    print(f"  Not found:            {not_found}")
    print(f"  Total found:          {found_nfl + found_wiki} ({(found_nfl+found_wiki)/len(te_bt)*100:.1f}%)")

    # Save
    results_df = pd.DataFrame(results)
    output_path = 'data/te_backtest_birthdates.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Show missing
    missing = results_df[results_df['birthdate'].isna()]
    if len(missing) > 0:
        print(f"\n{'='*60}")
        print(f"NOT FOUND ({len(missing)} players):")
        print(f"{'='*60}")
        for _, r in missing.iterrows():
            print(f"  {r['player_name']:30s} {str(r['college']):20s} {int(r['draft_year'])}")


if __name__ == '__main__':
    main()
