"""
CFBD Roster Lookup: Determine actual college seasons for ambiguous early-declare WRs.

For each player:
1. Find their teams via /player/search
2. Check rosters across all plausible years at each team
3. Count total roster appearances = total college seasons
4. Compare to current early_declare flag
"""

import requests
import time
import json
import pandas as pd
import sys

API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
BASE = "https://api.collegefootballdata.com"

def cfbd_get(endpoint, params, retries=3):
    """Make CFBD API request with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(f"{BASE}/{endpoint}", headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"  HTTP {resp.status_code} for {endpoint} {params}")
                return []
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)
    return []


def find_player_teams(name, position="WR"):
    """Find all teams a player was associated with in CFBD."""
    data = cfbd_get("player/search", {"searchTerm": name, "position": position})
    teams = set()
    player_id = None
    for p in data:
        if p["name"].lower() == name.lower():
            teams.add(p["team"])
            player_id = p["id"]
    # Also search without position filter for edge cases
    if not teams:
        data = cfbd_get("player/search", {"searchTerm": name})
        for p in data:
            if p["name"].lower() == name.lower():
                teams.add(p["team"])
                player_id = p["id"]
    return list(teams), player_id


def find_roster_years(name, team, year_range):
    """Check which years a player appears on a team's roster."""
    found_years = []
    first_name = name.split()[0].lower()
    last_name = name.split()[-1].lower()

    for year in year_range:
        data = cfbd_get("roster", {"team": team, "year": year})
        for p in data:
            fn = p.get("firstName", "").lower()
            ln = p.get("lastName", "").lower()
            if fn == first_name and ln == last_name:
                found_years.append(year)
                break
            # Also try partial match for names like "Marvin Harrison Jr."
            if last_name in ln.lower() and first_name in fn.lower():
                found_years.append(year)
                break
    return found_years


def count_college_seasons(name, college, draft_year, draft_age):
    """
    Count total college seasons for a player.
    Returns (total_seasons, roster_years_dict, notes)
    """
    # Determine the range of years to check
    # A player drafted in 2024 could have started college as early as 2019 (5-year senior)
    # or as late as 2021 (true junior)
    earliest_start = draft_year - 6  # generous range
    latest_season = draft_year - 1   # last college season
    year_range = list(range(earliest_start, latest_season + 1))

    # Find all teams from CFBD
    teams, player_id = find_player_teams(name)

    # If CFBD didn't find teams, use the college from our data
    if not teams:
        # Map common college name differences
        college_map = {
            "Ohio St.": "Ohio State",
            "Penn St.": "Penn State",
            "Mich. St.": "Michigan State",
            "Okla. St.": "Oklahoma State",
            "Miss. St.": "Mississippi State",
            "Fla. St.": "Florida State",
            "N.C. State": "NC State",
            "Southern Miss": "Southern Mississippi",
            "So. Carolina": "South Carolina",
        }
        mapped = college_map.get(college, college)
        teams = [mapped]

    # Check rosters across all teams and years
    all_roster_years = {}
    for team in teams:
        years_found = find_roster_years(name, team, year_range)
        if years_found:
            all_roster_years[team] = years_found

    # Count total unique seasons
    all_years = set()
    for team_years in all_roster_years.values():
        all_years.update(team_years)

    total_seasons = len(all_years)

    notes = ""
    if len(all_roster_years) > 1:
        notes = f"Transfer: {', '.join(f'{t}({min(y)}-{max(y)})' for t, y in all_roster_years.items())}"
    elif len(all_roster_years) == 1:
        team = list(all_roster_years.keys())[0]
        years = sorted(all_roster_years[team])
        notes = f"{team}: {years}"
    else:
        notes = "NOT FOUND IN CFBD ROSTERS"

    return total_seasons, all_roster_years, notes


def main():
    # Load backtest data
    bt = pd.read_csv("data/wr_backtest_all_components.csv")

    # Identify ambiguous players
    derived = bt[bt["declare_source"] == "derived"].copy()
    existing_ambig = bt[
        (bt["declare_source"] == "existing") &
        (bt["draft_age"] == 21) &
        (bt["early_declare"] == 0)
    ].copy()

    check = pd.concat([derived, existing_ambig]).sort_values("pick")
    print(f"Checking {len(check)} players via CFBD Roster API...")
    print("=" * 100)

    results = []

    for i, (_, row) in enumerate(check.iterrows()):
        name = row["player_name"]
        college = row["college"]
        draft_year = int(row["draft_year"])
        draft_age = row["draft_age"]
        pick = int(row["pick"])
        current_ed = int(row["early_declare"])
        source = row["declare_source"]

        print(f"\n[{i+1}/{len(check)}] {name} (pick {pick}, {college}, {draft_year}, age {draft_age})")

        total_seasons, roster_years, notes = count_college_seasons(name, college, draft_year, draft_age)

        # Determine correct ED
        if total_seasons > 0:
            correct_ed = 1 if total_seasons <= 3 else 0
            status = "RESOLVED"
        else:
            correct_ed = current_ed  # Can't determine, keep current
            status = "UNRESOLVED"

        changed = "CHANGED" if correct_ed != current_ed else "OK"

        print(f"  Roster seasons: {total_seasons} | Current ED={current_ed} -> Correct ED={correct_ed} | {changed}")
        print(f"  {notes}")

        results.append({
            "player_name": name,
            "pick": pick,
            "draft_year": draft_year,
            "draft_age": draft_age,
            "college": college,
            "current_ed": current_ed,
            "source": source,
            "cfbd_seasons": total_seasons,
            "correct_ed": correct_ed,
            "changed": changed,
            "status": status,
            "notes": notes
        })

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("output/cfbd_roster_results.csv", index=False)

    print("\n" + "=" * 100)
    print(f"\nRESULTS SUMMARY:")
    print(f"  Total checked: {len(results)}")
    print(f"  Resolved: {(results_df['status'] == 'RESOLVED').sum()}")
    print(f"  Unresolved: {(results_df['status'] == 'UNRESOLVED').sum()}")
    print(f"  Changes needed: {(results_df['changed'] == 'CHANGED').sum()}")

    # Show all changes
    changes = results_df[results_df["changed"] == "CHANGED"].sort_values("pick")
    if len(changes) > 0:
        print(f"\nCHANGES NEEDED ({len(changes)}):")
        for _, r in changes.iterrows():
            print(f"  {r['player_name']:25s} pick={r['pick']:3d} age={r['draft_age']} "
                  f"ED: {r['current_ed']} -> {r['correct_ed']} ({r['cfbd_seasons']} seasons) | {r['notes']}")

    # Show unresolved
    unresolved = results_df[results_df["status"] == "UNRESOLVED"].sort_values("pick")
    if len(unresolved) > 0:
        print(f"\nUNRESOLVED ({len(unresolved)}):")
        for _, r in unresolved.iterrows():
            print(f"  {r['player_name']:25s} pick={r['pick']:3d} age={r['draft_age']} ED={r['current_ed']} | {r['notes']}")


if __name__ == "__main__":
    main()
