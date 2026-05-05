"""Fill missing rec_yards / team_pass_att for SLAP V5 master database.

Auto-detects players whose master DB row is missing rec_yards or team_pass_att,
fetches the data from CFBD for their final college season (draft_year - 1),
and writes the result into the appropriate source files.

Run locally (sandbox blocks api.collegefootballdata.com):

    pip install python-dotenv requests pandas
    # Make sure .env contains:  CFBD_API_KEY=your_key
    python scripts/fill_missing_receiving_data.py

Then rebuild:
    python src/build_master_database_v5.py

What it touches:
- WR backtest gaps -> appends to data/wr_backtest_with_production.csv
- RB 2026 gaps     -> updates rec_yards / team_pass_attempts in data/prospects_final.csv
- TE 2026 gaps     -> updates cfbd_rec_yards / cfbd_team_pass_att in data/te_2026_prospects_final.csv
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("CFBD_API_KEY")
if not API_KEY:
    raise SystemExit("CFBD_API_KEY not set. Add it to .env at the repo root.")

BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# School name normalization: backtest CSV format -> CFBD format.
# Reuses (and extends) the mapping from src/fetch_rb_receiving_stats.py.
SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State", "Penn St.": "Penn State", "Michigan St.": "Michigan State",
    "North Dakota St.": "North Dakota State", "Oklahoma St.": "Oklahoma State",
    "Miss. St.": "Mississippi State", "Mississippi St.": "Mississippi State",
    "San Diego St.": "San Diego State", "Florida St.": "Florida State",
    "Fresno St.": "Fresno State", "Boise St.": "Boise State", "Iowa St.": "Iowa State",
    "Arizona St.": "Arizona State", "Kansas St.": "Kansas State",
    "N.C. State": "NC State", "N.C.": "North Carolina",
    "North Carolina St.": "NC State", "Appalachian St.": "Appalachian State",
    "Colorado St.": "Colorado State", "Oregon St.": "Oregon State",
    "Washington St.": "Washington State", "S. Carolina": "South Carolina",
    "Miami (FL)": "Miami", "Central Mich.": "Central Michigan",
    "Western Mich.": "Western Michigan", "Eastern Mich.": "Eastern Michigan",
    "Northern Ill.": "Northern Illinois", "Southern Miss.": "Southern Mississippi",
    "Southern Miss": "Southern Mississippi", "San Jose St.": "San Jose State",
    "La.-Monroe": "Louisiana Monroe", "La.-Lafayette": "Louisiana",
    "Louisiana-Lafayette": "Louisiana", "Texas-El Paso": "UTEP",
    "Ala-Birmingham": "UAB", "South Dakota St.": "South Dakota State",
    "New Mexico St.": "New Mexico State", "Boston Col.": "Boston College",
    "Mississippi": "Ole Miss", "Pitt": "Pittsburgh",
    # FCS / non-FBS schools — CFBD typically won't have full receiving stats
    "North Carolina A&T": None, "Virginia St.": None, "William & Mary": None,
    "Monmouth": None, "Grambling St.": None, "East Central (OK)": None,
    "Pennsylvania": None, "Central Arkansas": None, "Hawaii": "Hawai'i",
    "Charleston (WV)": None, "SE Missouri St.": "Southeast Missouri State",
    "West Alabama": None, "John Carroll": None, "Sacramento State": "Sacramento State",
    "McNeese State": "McNeese", "Sam Houston State": "Sam Houston",
    "Virginia Union": None, "UC Davis": "UC Davis", "Tulane": "Tulane",
}


def normalize_school(school):
    if pd.isna(school):
        return None
    s = str(school).strip()
    return SCHOOL_MAPPINGS.get(s, s)


def names_match(name1, name2):
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if n1 == n2:
        return True
    parts1 = n1.replace(".", " ").replace("'", "").split()
    parts2 = n2.replace(".", " ").replace("'", "").split()
    suffixes = {"jr", "ii", "iii", "iv", "sr"}
    clean1 = [p for p in parts1 if p not in suffixes]
    clean2 = [p for p in parts2 if p not in suffixes]
    if not clean1 or not clean2:
        return False
    first1, first2 = clean1[0], clean2[0]
    if first1 != first2:
        if not (len(first1) <= 2 and first2.startswith(first1[0])):
            if not (len(first2) <= 2 and first1.startswith(first2[0])):
                return False
    if len(clean1) > 1 and len(clean2) > 1:
        if clean1[-1] != clean2[-1]:
            return False
    return True


_recv_cache = {}
_pa_cache = {}


def fetch_team_receiving(team, year):
    key = (team, year)
    if key in _recv_cache:
        return _recv_cache[key]
    try:
        r = requests.get(f"{BASE_URL}/stats/player/season", headers=HEADERS,
                         params={"year": year, "category": "receiving", "team": team}, timeout=30)
        if r.status_code == 200:
            stats = {}
            for stat in r.json():
                player = stat.get("player", "").lower()
                stat_type = stat.get("statType", "")
                value = stat.get("stat", 0)
                if player not in stats:
                    stats[player] = {}
                try:
                    stats[player][stat_type] = int(float(value))
                except (TypeError, ValueError):
                    stats[player][stat_type] = 0
            _recv_cache[key] = stats
            return stats
        if r.status_code == 429:
            print(f"    Rate limited on {team} {year}, sleeping 5s")
            time.sleep(5)
            return None
        return {}
    except requests.RequestException as e:
        print(f"    Network error on {team} {year}: {e}")
        return {}


def fetch_team_pass_att(team, year):
    key = (team, year)
    if key in _pa_cache:
        return _pa_cache[key]
    try:
        r = requests.get(f"{BASE_URL}/stats/season", headers=HEADERS,
                         params={"year": year, "team": team}, timeout=30)
        if r.status_code == 200:
            for stat in r.json():
                if stat.get("statName") == "passAttempts":
                    val = float(stat.get("statValue", 0))
                    _pa_cache[key] = val
                    return val
            _pa_cache[key] = None
            return None
        if r.status_code == 429:
            time.sleep(5)
        return None
    except requests.RequestException as e:
        print(f"    Network error on team stats {team} {year}: {e}")
        return None


def find_player(player_name, team_recv):
    if not team_recv:
        return None, None
    for api_name, stats in team_recv.items():
        if names_match(player_name, api_name):
            return stats.get("YDS"), api_name
    return None, None


def fetch_one(player, college, season):
    """Try the given season; on miss, try season-1 (for transfers/early declares)."""
    cfbd_school = normalize_school(college)
    if cfbd_school is None:
        return None, None, "school_not_in_cfbd"

    team_recv = fetch_team_receiving(cfbd_school, season)
    time.sleep(0.3)
    yards, matched = find_player(player, team_recv)
    if yards is not None:
        pa = fetch_team_pass_att(cfbd_school, season)
        time.sleep(0.3)
        return yards, pa, f"matched as '{matched}' in {season}"

    # Fallback: try previous season
    prev = season - 1
    team_recv_prev = fetch_team_receiving(cfbd_school, prev)
    time.sleep(0.3)
    yards2, matched2 = find_player(player, team_recv_prev)
    if yards2 is not None:
        pa = fetch_team_pass_att(cfbd_school, prev)
        time.sleep(0.3)
        return yards2, pa, f"matched as '{matched2}' in {prev} (fallback)"

    return None, None, "player_not_found"


def fill_wr_backtest():
    """Fill WR backtest gaps. Appends to data/wr_backtest_with_production.csv
    so the existing build pipeline picks them up.
    """
    print("\n" + "=" * 80)
    print("WR BACKTEST")
    print("=" * 80)
    bt = pd.read_csv(ROOT / "data" / "wr_backtest_all_components.csv")
    prod = pd.read_csv(ROOT / "data" / "wr_backtest_with_production.csv")

    # Identify missing
    merged = bt.merge(prod[["player_name", "draft_year", "rec_yards", "team_pass_att"]],
                      on=["player_name", "draft_year"], how="left")
    missing = merged[merged["rec_yards"].isna() | merged["team_pass_att"].isna()].copy()
    print(f"Missing: {len(missing)} WRs")

    new_rows = []
    found = 0
    for _, row in missing.iterrows():
        player = row["player_name"]
        college = row["college"]
        draft_year = int(row["draft_year"])
        season = draft_year - 1
        yards, pa, status = fetch_one(player, college, season)
        if yards is not None:
            new_rows.append({
                "player_name": player, "draft_year": draft_year, "pick": row["pick"],
                "college": college, "rec_yards": yards, "team_pass_att": pa,
            })
            found += 1
            print(f"  ✓ {player} ({college}, {season}): {yards} yds / {pa} pa — {status}")
        else:
            print(f"  ✗ {player} ({college}, {season}): {status}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        out_path = ROOT / "data" / "wr_backtest_with_production.csv"
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["player_name", "draft_year"], keep="last")
        combined.to_csv(out_path, index=False)
        print(f"  -> wrote {len(new_rows)} new rows to {out_path}")

    print(f"  Found: {found}/{len(missing)}")


def fill_rb_2026():
    """Fill RB 2026 gaps in data/prospects_final.csv (in place)."""
    print("\n" + "=" * 80)
    print("RB 2026 PROSPECTS")
    print("=" * 80)
    path = ROOT / "data" / "prospects_final.csv"
    df = pd.read_csv(path)
    rb = df[df["position"] == "RB"]
    missing_mask = (df["position"] == "RB") & (df["rec_yards"].isna() | df["team_pass_attempts"].isna())
    missing = df[missing_mask]
    print(f"Missing: {len(missing)} RBs")

    found = 0
    for idx, row in missing.iterrows():
        player = row["player_name"]
        college = row["school"]
        season = 2025  # final college season for 2026 draft
        yards, pa, status = fetch_one(player, college, season)
        if yards is not None:
            df.at[idx, "rec_yards"] = yards
            if pa is not None:
                df.at[idx, "team_pass_attempts"] = pa
            found += 1
            print(f"  ✓ {player} ({college}, {season}): {yards} yds / {pa} pa — {status}")
        else:
            print(f"  ✗ {player} ({college}, {season}): {status}")

    if found > 0:
        df.to_csv(path, index=False)
        print(f"  -> updated {found} rows in {path}")
    print(f"  Found: {found}/{len(missing)}")


def fill_te_2026():
    """Fill TE 2026 gaps in data/te_2026_prospects_final.csv (in place)."""
    print("\n" + "=" * 80)
    print("TE 2026 PROSPECTS")
    print("=" * 80)
    path = ROOT / "data" / "te_2026_prospects_final.csv"
    df = pd.read_csv(path)
    missing_mask = df["cfbd_rec_yards"].isna() | df["cfbd_team_pass_att"].isna()
    missing = df[missing_mask]
    print(f"Missing: {len(missing)} TEs")

    found = 0
    for idx, row in missing.iterrows():
        player = row["player_name"]
        college = row["college"]
        season = 2025
        yards, pa, status = fetch_one(player, college, season)
        if yards is not None:
            df.at[idx, "cfbd_rec_yards"] = yards
            if pa is not None:
                df.at[idx, "cfbd_team_pass_att"] = pa
            found += 1
            print(f"  ✓ {player} ({college}, {season}): {yards} yds / {pa} pa — {status}")
        else:
            print(f"  ✗ {player} ({college}, {season}): {status}")

    if found > 0:
        df.to_csv(path, index=False)
        print(f"  -> updated {found} rows in {path}")
    print(f"  Found: {found}/{len(missing)}")


if __name__ == "__main__":
    fill_wr_backtest()
    fill_rb_2026()
    fill_te_2026()
    print("\nAll done. Now rebuild: python src/build_master_database_v5.py")
