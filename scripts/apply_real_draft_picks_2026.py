"""Apply real 2026 NFL Draft picks to SLAP V5 prospect files.

Logic:
- Rounds 1-7 of the actual 2026 draft are encoded below (WR/RB/TE only).
- Players matched in our prospect files get their real pick number.
- Players in our prospect files NOT mentioned in any round are set to pick=250 (UDFA convention).
- Eli Heidenreich is reclassified WR -> RB (per user direction).
- Max Bredeson and Anthony Smith are skipped (not in our prospect files; user said skip).
"""

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# (pick, file_player_name, position) — file_player_name matches what's in the CSV
DRAFT_PICKS = [
    # Round 1
    (3,   "Jeremiyah Love",       "RB"),
    (4,   "Carnell Tate",         "WR"),
    (8,   "Jordyn Tyson",         "WR"),
    (16,  "Kenyon Sadiq",         "TE"),
    (20,  "Makai Lemon",          "WR"),
    (24,  "Kevin Concepcion",     "WR"),   # listed as "KC Concepcion"
    (30,  "Omar Cooper Jr.",      "WR"),
    (32,  "Jadarian Price",       "RB"),
    # Round 2
    (33,  "De'Zhaun Stribling",   "WR"),
    (39,  "Denzel Boston",        "WR"),
    (47,  "Germie Bernard",       "WR"),
    (54,  "Eli Stowers",          "TE"),
    (56,  "Nate Boerkircher",     "TE"),
    (59,  "Marlin Klein",         "TE"),
    (61,  "Max Klare",            "TE"),
    # Round 3
    (69,  "Sam Roush",            "TE"),
    (71,  "Antonio Williams",     "WR"),
    (73,  "Oscar Delp",           "TE"),
    (74,  "Malachi Fields",       "WR"),
    (75,  "Caleb Douglas",        "WR"),
    (79,  "Zachariah Branch",     "WR"),
    (80,  "Ja'Kobi Lane",         "WR"),
    (83,  "Chris Brazzell",       "WR"),   # listed as "Chris Brazzell II"
    (84,  "Ted Hurst",            "WR"),
    (87,  "Will Kacmarek",        "TE"),
    (89,  "Zavion Thomas",        "WR"),
    (90,  "Kaelon Black",         "RB"),
    (94,  "Chris Bell",           "WR"),
    (95,  "Eli Raridon",          "TE"),
    # Round 4
    (105, "Brenen Thompson",      "WR"),
    (108, "Jonah Coleman",        "RB"),
    (115, "Elijah Sarratt",       "WR"),
    (121, "Kaden Wetjen",         "WR"),
    (122, "Mike Washington Jr.",  "RB"),
    (125, "Skyler Bell",          "WR"),
    (133, "Matthew Hibner",       "TE"),
    (136, "Bryce Lance",          "WR"),
    (140, "Colbie Young",         "WR"),
    # Round 5
    (143, "Reggie Virgil",        "WR"),
    (152, "Justin Joly",          "TE"),
    # 159 Max Bredeson — SKIP (not in file)
    (161, "Emmett Johnson",       "RB"),
    (164, "Tanner Koziol",        "TE"),
    (165, "Nick Singleton",       "RB"),   # listed as "Nicholas Singleton"
    (168, "Kendrick Law",         "WR"),
    (169, "Riley Nowakowski",     "TE"),
    (170, "Joe Royer",            "TE"),
    (173, "Josh Cuevas",          "TE"),
    (174, "Adam Randall",         "RB"),
    (176, "Cyrus Allen",          "WR"),
    (177, "Kevin Coleman Jr.",    "WR"),
    (180, "Seydou Traore",        "TE"),
    # Round 6
    (185, "Bauer Sharp",          "TE"),
    (187, "Kaytron Allen",        "RB"),
    (190, "Barion Brown",         "WR"),
    (191, "Josh Cameron",         "WR"),
    (195, "Malik Benson",         "WR"),
    (197, "C.J. Daniels",         "WR"),   # listed as "CJ Daniels"
    (198, "Demond Claiborne",     "RB"),
    (199, "Emmanuel Henderson",   "WR"),   # listed as "Emmanuel Henderson Jr."
    (203, "C.J. Williams",        "WR"),   # listed as "CJ Williams"
    (204, "Lewis Bond",           "WR"),
    # Round 7
    # 218 Anthony Smith — SKIP (not in file)
    (221, "Jack Endries",         "TE"),
    (225, "Jaren Kanak",          "TE"),
    (230, "Eli Heidenreich",      "RB"),   # reclassify WR -> RB
    (237, "Seth McGowan",         "RB"),
    (245, "Jamarion Miller",      "RB"),   # listed as "Jam Miller"
    (248, "Carsen Ryan",          "TE"),
    (254, "Deion Burks",          "WR"),
]

UDFA_PICK = 250


def pick_to_round(pick: int) -> int:
    """Standard NFL round mapping (32 picks per round, 7th round caps at 257)."""
    if pick <= 32:
        return 1
    if pick <= 64:
        return 2
    if pick <= 100:
        return 3
    if pick <= 140:
        return 4
    if pick <= 181:
        return 5
    if pick <= 216:
        return 6
    return 7


def update_wr_rb_file():
    path = ROOT / "data" / "prospects_final.csv"
    df = pd.read_csv(path)

    pick_lookup = {name: pick for pick, name, pos in DRAFT_PICKS if pos in ("WR", "RB")}

    matched = []
    unmatched_drafted = list(pick_lookup.keys())

    new_picks = []
    new_positions = []
    for _, row in df.iterrows():
        name = row["player_name"]
        if name in pick_lookup:
            new_picks.append(pick_lookup[name])
            matched.append(name)
            unmatched_drafted.remove(name)
            # Heidenreich: reclassify WR -> RB
            if name == "Eli Heidenreich":
                new_positions.append("RB")
            else:
                new_positions.append(row["position"])
        else:
            new_picks.append(UDFA_PICK)
            new_positions.append(row["position"])

    df["projected_pick"] = new_picks
    df["position"] = new_positions
    df.to_csv(path, index=False)

    print(f"[prospects_final.csv] matched {len(matched)} drafted players")
    if unmatched_drafted:
        print(f"  WARN: drafted players NOT FOUND in file: {unmatched_drafted}")
    return matched, unmatched_drafted


def update_te_file():
    path = ROOT / "data" / "te_2026_prospects_final.csv"
    df = pd.read_csv(path)

    pick_lookup = {name: pick for pick, name, pos in DRAFT_PICKS if pos == "TE"}

    matched = []
    unmatched_drafted = list(pick_lookup.keys())

    new_picks = []
    for _, row in df.iterrows():
        name = row["player_name"]
        if name in pick_lookup:
            new_picks.append(pick_lookup[name])
            matched.append(name)
            unmatched_drafted.remove(name)
        else:
            new_picks.append(UDFA_PICK)

    df["projected_pick"] = new_picks
    df.to_csv(path, index=False)

    print(f"[te_2026_prospects_final.csv] matched {len(matched)} drafted TEs")
    if unmatched_drafted:
        print(f"  WARN: drafted TEs NOT FOUND in file: {unmatched_drafted}")
    return matched, unmatched_drafted


def update_wr_pre_file():
    """Update output/slap_v5_wr_2026.csv. Build script reads `pick` column.

    Heidenreich is reclassified to RB, so if he appears in this WR file, set him to UDFA
    so the WR pipeline ignores him (the build script joins on player_name with the WR/RB file).
    """
    path = ROOT / "output" / "slap_v5_wr_2026.csv"
    df = pd.read_csv(path)

    pick_lookup = {name: pick for pick, name, pos in DRAFT_PICKS if pos == "WR"}

    matched = []
    unmatched_drafted = list(pick_lookup.keys())

    new_picks = []
    new_rounds = []
    for _, row in df.iterrows():
        name = row["player_name"]
        if name in pick_lookup:
            p = pick_lookup[name]
            new_picks.append(p)
            new_rounds.append(pick_to_round(p))
            matched.append(name)
            unmatched_drafted.remove(name)
        else:
            new_picks.append(UDFA_PICK)
            new_rounds.append(pick_to_round(UDFA_PICK))

    df["pick"] = new_picks
    df["round"] = new_rounds
    df.to_csv(path, index=False)

    print(f"[slap_v5_wr_2026.csv] matched {len(matched)} drafted WRs")
    if unmatched_drafted:
        print(f"  WARN: drafted WRs NOT FOUND in file: {unmatched_drafted}")
    return matched, unmatched_drafted


if __name__ == "__main__":
    print("Applying real 2026 draft picks to SLAP V5 prospect files...")
    print()
    update_wr_rb_file()
    update_te_file()
    update_wr_pre_file()
    print()
    print("Done. Now run: python src/build_master_database_v5.py")
