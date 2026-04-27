"""Fill missing rec_yards / team_pass_att using cfbfastR-data PBP parquet files.

Why this and not the CFBD API: api.collegefootballdata.com is not reachable
from the sandbox, but raw.githubusercontent.com is. cfbfastR-data publishes
play-by-play parquet files keyed by season (~3-5 MB each), which we aggregate
to season totals locally.

Run:
    python scripts/cfbfastr_fill_gaps.py

Then rebuild:
    python src/build_master_database_v5.py

Writes:
- WR gaps   -> appends rows to data/wr_backtest_with_production.csv
- RB 2026   -> updates rec_yards / team_pass_attempts in data/prospects_final.csv
- TE 2026   -> updates cfbd_rec_yards / cfbd_team_pass_att in data/te_2026_prospects_final.csv
"""

import os
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PBP_DIR = Path("/tmp/cfb_pbp")
PBP_DIR.mkdir(exist_ok=True)

PBP_BASE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/player_stats/parquet/player_stats_{year}.parquet"

# School-name normalization: SLAP CSV format -> cfbfastR team-name format.
# cfbfastR uses ESPN team names (e.g., "Ohio State", "Hawai'i"). FCS coverage exists
# but is patchy. None means the player's school is unlikely to be in cfbfastR.
SCHOOL_MAP = {
    # Standard FBS abbreviations
    "Ohio St.": "Ohio State", "Penn St.": "Penn State", "Michigan St.": "Michigan State",
    "North Dakota St.": "North Dakota State", "Oklahoma St.": "Oklahoma State",
    "Miss. St.": "Mississippi State", "Mississippi St.": "Mississippi State",
    "San Diego St.": "San Diego State", "Florida St.": "Florida State",
    "Fresno St.": "Fresno State", "Boise St.": "Boise State", "Iowa St.": "Iowa State",
    "Arizona St.": "Arizona State", "Kansas St.": "Kansas State",
    "N.C. State": "NC State", "North Carolina St.": "NC State",
    "Appalachian St.": "Appalachian State", "Colorado St.": "Colorado State",
    "Oregon St.": "Oregon State", "Washington St.": "Washington State",
    "S. Carolina": "South Carolina", "Miami (FL)": "Miami",
    "Central Mich.": "Central Michigan", "Western Mich.": "Western Michigan",
    "Eastern Mich.": "Eastern Michigan", "Northern Ill.": "Northern Illinois",
    "Southern Miss.": "Southern Mississippi", "Southern Miss": "Southern Mississippi",
    "San Jose St.": "San Jose State", "La.-Monroe": "UL Monroe",
    "La-Monroe": "UL Monroe", "La.-Lafayette": "Louisiana",
    "Texas-El Paso": "UTEP", "Ala-Birmingham": "UAB",
    "South Dakota St.": "South Dakota State", "New Mexico St.": "New Mexico State",
    "Boston Col.": "Boston College", "Mississippi": "Ole Miss",
    "Pitt": "Pittsburgh", "Hawaii": "Hawai'i",
    "Connecticut": "UConn",
    # FCS / non-FBS schools — likely missing from cfbfastR
    "North Carolina A&T": "North Carolina A&T", "Virginia St.": None,
    "William & Mary": "William & Mary", "Monmouth": "Monmouth",
    "Grambling St.": "Grambling", "East Central (OK)": None,
    "Pennsylvania": None,  # Penn (Ivy) — not in cfbfastR FCS coverage
    "Central Arkansas": "Central Arkansas",
    "Charleston (WV)": None, "SE Missouri St.": "Southeast Missouri State",
    "West Alabama": None, "John Carroll": None,
    "McNeese State": "McNeese", "Sam Houston State": "Sam Houston",
    "Sacramento State": "Sacramento State", "Virginia Union": None,
    "UC Davis": "UC Davis",
    # 2026 RB prospect schools
    "Texas Tech": "Texas Tech", "Troy": "Troy", "Pittsburgh": "Pittsburgh",
    # 2025 class schools (already in canonical form mostly)
    "Colorado": "Colorado", "Arizona": "Arizona", "Ohio State": "Ohio State",
    "Texas": "Texas", "Iowa State": "Iowa State", "Missouri": "Missouri",
    "Ole Miss": "Ole Miss", "Washington State": "Washington State",
    "Arkansas": "Arkansas", "Illinois": "Illinois", "Stanford": "Stanford",
    "Florida": "Florida", "Oregon": "Oregon", "Notre Dame": "Notre Dame",
    "Tulane": "Tulane",
}

# First-name nickname expansions: short-form -> formal-form. Bidirectional via names_match.
NICKNAME_TO_FORMAL = {
    "mike": "michael", "pat": "patrick", "tom": "thomas", "tony": "anthony",
    "bill": "william", "will": "william", "billy": "william",
    "rob": "robert", "bob": "robert", "robby": "robert",
    "rick": "richard", "rich": "richard", "dick": "richard",
    "joe": "joseph", "joey": "joseph", "jim": "james", "jimmy": "james",
    "dan": "daniel", "danny": "daniel", "andy": "andrew", "drew": "andrew",
    "matt": "matthew", "ted": "theodore", "ed": "edward", "eddie": "edward",
    "alex": "alexander", "chris": "christopher", "nick": "nicholas",
    "tim": "timothy", "ben": "benjamin", "sam": "samuel", "ted": "edward",
    "stan": "stanley", "vince": "vincent", "ron": "ronald", "ronny": "ronald",
    "frank": "francis", "tre": "trevon",  # weak — disambiguate with last name
}

# Manual player-name overrides: SLAP file name -> cfbfastR record name.
# Used for nicknames that aren't reducible by NICKNAME_TO_FORMAL.
PLAYER_NAME_OVERRIDES = {
    "Bub Means": "Jerrod Means",
    "Pat Bryant": "Patrick Bryant",
    "Will Fuller": "Will Fuller V",  # in case it shows up that way
    "Ja'Mori Maclin": "Jay Maclin",  # listed as 'Jay Maclin' in cfbfastR
}

# Manual school overrides: SLAP file player -> actual cfbfastR team for their final season.
# Used when a player transferred and our data lists the wrong school.
PLAYER_TEAM_OVERRIDES = {
    "Quinten Joyner": ("USC", 2024),  # transferred TT -> USC, last meaningful season
    "Trevion Cooley": ("Georgia Tech", 2024),  # transferred to Troy but played at GT
}


def normalize_school(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    return SCHOOL_MAP.get(s, s)


def _expand_first(name):
    """If the first name is a known nickname, return the formal form (else as-is)."""
    return NICKNAME_TO_FORMAL.get(name, name)


def names_match(a, b):
    if pd.isna(a) or pd.isna(b):
        return False
    a = str(a).lower().strip().replace(".", " ").replace("'", "")
    b = str(b).lower().strip().replace(".", " ").replace("'", "")
    if a == b:
        return True
    suf = {"jr", "ii", "iii", "iv", "v", "sr"}
    pa = [p for p in a.split() if p not in suf]
    pb = [p for p in b.split() if p not in suf]
    if not pa or not pb:
        return False
    fa, fb = pa[0], pb[0]
    if fa != fb:
        # Initial match (e.g., "T.J." vs "Tyler")
        if not (len(fa) <= 2 and fb.startswith(fa[0])):
            if not (len(fb) <= 2 and fa.startswith(fb[0])):
                # Nickname match (e.g., "Mike" <-> "Michael")
                if _expand_first(fa) != _expand_first(fb):
                    return False
    if len(pa) > 1 and len(pb) > 1 and pa[-1] != pb[-1]:
        return False
    return True


_pbp_cache = {}


def load_pbp(year):
    if year in _pbp_cache:
        return _pbp_cache[year]
    path = PBP_DIR / f"ps_{year}.parquet"
    if not path.exists():
        url = PBP_BASE.format(year=year)
        print(f"  downloading {url}")
        try:
            urllib.request.urlretrieve(url, path)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                _pbp_cache[year] = None
                return None
            raise
    df = pd.read_parquet(path)
    _pbp_cache[year] = df
    return df


def player_rec_yards(pbp, player_name, team):
    """Sum reception_yds for player on team within this PBP."""
    cands = pbp[pbp["team"] == team]
    cands = cands[cands["reception_player"].notna()]
    matches = cands[cands["reception_player"].apply(lambda x: names_match(x, player_name))]
    if matches.empty:
        return None, None
    matched_name = matches["reception_player"].mode().iloc[0]
    yards = matches["reception_yds"].sum()
    return float(yards), matched_name


def team_pass_attempts(pbp, team):
    """Pass attempts = completions + incompletions + thrown interceptions."""
    sub = pbp[pbp["team"] == team]
    mask = sub["completion_player"].notna() | sub["incompletion_player"].notna() | sub["interception_thrown_player"].notna()
    return int(mask.sum())


def fetch_one(player, college, season):
    """Try given season then season-1. If still not found, search all loaded years
    across any team and pick the most-receiving year."""
    # Apply manual overrides
    lookup_player = PLAYER_NAME_OVERRIDES.get(player, player)
    if player in PLAYER_TEAM_OVERRIDES:
        ov_team, ov_season = PLAYER_TEAM_OVERRIDES[player]
        college = ov_team
        season = ov_season

    norm = normalize_school(college)
    if norm is None:
        return None, None, "school not in cfbfastR"

    pbp = load_pbp(season)
    if pbp is None:
        return None, None, f"no cfbfastR data for {season}"
    yards, matched = player_rec_yards(pbp, lookup_player, norm)
    if yards is not None:
        pa = team_pass_attempts(pbp, norm)
        return yards, pa, f"matched as '{matched}' in {season}"

    # Fallback 1: prior season at expected school
    prev = season - 1
    pbp_prev = load_pbp(prev)
    if pbp_prev is not None:
        yards2, matched2 = player_rec_yards(pbp_prev, lookup_player, norm)
        if yards2 is not None:
            pa = team_pass_attempts(pbp_prev, norm)
            return yards2, pa, f"matched as '{matched2}' in {prev} (fallback)"

    # Fallback 2: cross-team search in season + season-1 (handles transfers).
    # Require STRICT name match (exact first+last after nickname expansion) to
    # avoid false positives where short names like "Ty" match "Tyson".
    def strict_match(api_name, target):
        a = str(api_name).lower().strip().replace(".", " ").replace("'", "")
        b = str(target).lower().strip().replace(".", " ").replace("'", "")
        suf = {"jr", "ii", "iii", "iv", "v", "sr"}
        pa_ = [p for p in a.split() if p not in suf]
        pb_ = [p for p in b.split() if p not in suf]
        if not pa_ or not pb_:
            return False
        if pa_[-1] != pb_[-1]:
            return False
        # First name: exact match OR nickname-expanded equality (no initial-only)
        return _expand_first(pa_[0]) == _expand_first(pb_[0])

    for try_year, try_pbp in [(season, pbp), (prev, pbp_prev)]:
        if try_pbp is None:
            continue
        cands = try_pbp[(try_pbp["reception_player"].notna()) &
                         try_pbp["reception_player"].apply(lambda x: strict_match(x, lookup_player))]
        if not cands.empty:
            best_team = cands.groupby("team")["reception_yds"].sum().idxmax()
            sub = cands[cands["team"] == best_team]
            yards3 = float(sub["reception_yds"].sum())
            matched3 = sub["reception_player"].mode().iloc[0]
            pa = team_pass_attempts(try_pbp, best_team)
            return yards3, pa, f"matched as '{matched3}' at {best_team} in {try_year} (cross-team fallback)"

    return None, None, f"not found in {season}" + (f" or {prev}" if pbp_prev is not None else "")


def fill_wr_backtest():
    print("\n" + "=" * 80)
    print("WR BACKTEST")
    print("=" * 80)
    bt = pd.read_csv(ROOT / "data" / "wr_backtest_all_components.csv")
    prod = pd.read_csv(ROOT / "data" / "wr_backtest_with_production.csv")
    merged = bt.merge(prod[["player_name", "draft_year", "rec_yards", "team_pass_att"]],
                      on=["player_name", "draft_year"], how="left")
    missing = merged[merged["rec_yards"].isna() | merged["team_pass_att"].isna()].copy()
    print(f"Missing: {len(missing)} WRs")

    new_rows = []
    for _, row in missing.iterrows():
        player = row["player_name"]
        college = row["college"]
        if pd.isna(college):
            print(f"  ✗ {player}: no college on file")
            continue
        season = int(row["draft_year"]) - 1
        try:
            yards, pa, status = fetch_one(player, college, season)
        except FileNotFoundError as e:
            print(f"  ✗ {player}: {e}")
            continue
        if yards is not None:
            new_rows.append({
                "player_name": player, "draft_year": int(row["draft_year"]),
                "pick": row["pick"], "college": college,
                "rec_yards": yards, "team_pass_att": pa,
            })
            print(f"  ✓ {player} ({college}, {season}): {yards:.0f} yds / {pa} pa — {status}")
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
    print(f"  Found: {len(new_rows)}/{len(missing)}")


def fill_wr_rb_2026(position):
    print("\n" + "=" * 80)
    print(f"{position} 2026 PROSPECTS")
    print("=" * 80)
    path = ROOT / "data" / "prospects_final.csv"
    df = pd.read_csv(path)
    miss_mask = (df["position"] == position) & (df["rec_yards"].isna() | df["team_pass_attempts"].isna())
    missing = df[miss_mask]
    print(f"Missing: {len(missing)} {position}s")

    found = 0
    for idx, row in missing.iterrows():
        try:
            yards, pa, status = fetch_one(row["player_name"], row["school"], 2025)
        except FileNotFoundError as e:
            print(f"  ✗ {row['player_name']}: {e}")
            continue
        if yards is not None:
            df.at[idx, "rec_yards"] = yards
            if pa is not None:
                df.at[idx, "team_pass_attempts"] = pa
            found += 1
            print(f"  ✓ {row['player_name']} ({row['school']}, 2025): {yards:.0f} yds / {pa} pa — {status}")
        else:
            print(f"  ✗ {row['player_name']} ({row['school']}, 2025): {status}")

    if found > 0:
        df.to_csv(path, index=False)
        print(f"  -> updated {found} rows in {path}")
    print(f"  Found: {found}/{len(missing)}")


def fill_te_2026():
    print("\n" + "=" * 80)
    print("TE 2026 PROSPECTS")
    print("=" * 80)
    path = ROOT / "data" / "te_2026_prospects_final.csv"
    df = pd.read_csv(path)
    miss_mask = df["cfbd_rec_yards"].isna() | df["cfbd_team_pass_att"].isna()
    missing = df[miss_mask]
    print(f"Missing: {len(missing)} TEs")

    found = 0
    for idx, row in missing.iterrows():
        try:
            yards, pa, status = fetch_one(row["player_name"], row["college"], 2025)
        except FileNotFoundError as e:
            print(f"  ✗ {row['player_name']}: {e}")
            continue
        if yards is not None:
            df.at[idx, "cfbd_rec_yards"] = yards
            if pa is not None:
                df.at[idx, "cfbd_team_pass_att"] = pa
            found += 1
            print(f"  ✓ {row['player_name']} ({row['college']}, 2025): {yards:.0f} yds / {pa} pa — {status}")
        else:
            print(f"  ✗ {row['player_name']} ({row['college']}, 2025): {status}")

    if found > 0:
        df.to_csv(path, index=False)
        print(f"  -> updated {found} rows in {path}")
    print(f"  Found: {found}/{len(missing)}")


if __name__ == "__main__":
    fill_wr_backtest()
    fill_wr_rb_2026("WR")
    fill_wr_rb_2026("RB")
    fill_te_2026()
    print("\nAll done.")
