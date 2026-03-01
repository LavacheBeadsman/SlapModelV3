"""
Build data/combine_2026.csv from scratch, combining:
1. MathBomb data (from the original file, hardcoded here)
2. Dynasty Nerds data (from user paste)
Dynasty Nerds takes priority for body measurements and RAS.
"""

import pandas as pd
import re
import numpy as np

# ── Helper functions ─────────────────────────────────────────────────────

def parse_height_to_inches(h):
    """Convert height strings like 6′6″, 6′5¾″, 6′0.1″, 5′11⅝″ to inches."""
    if h is None:
        return None
    h = h.strip()
    frac_map = {'¼': 0.25, '½': 0.5, '¾': 0.75, '⅛': 0.125, '⅜': 0.375, '⅝': 0.625, '⅞': 0.875}
    m = re.match(r"(\d+)[′'](\d+\.?\d*)(.*?)\"?$", h.replace("″", "\"").replace("′", "'"))
    if not m:
        return None
    feet = int(m.group(1))
    inches = float(m.group(2))
    remainder = m.group(3)
    frac = 0
    for char, val in frac_map.items():
        if char in remainder:
            frac = val
            break
    return feet * 12 + inches + frac

def parse_broad_to_inches(b):
    """Convert broad jump like 10′6″, 11′3″ to total inches."""
    if b is None:
        return None
    b = b.strip().replace("″", "\"").replace("′", "'")
    m = re.match(r"(\d+)['](\d+)\"?$", b)
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2))

# ── MathBomb data (from original combine_2026.csv) ──────────────────────
# Only players NOT in Dynasty Nerds, or with extra drill data DN doesn't have

mathbomb_only = [
    # WRs that DN doesn't have
    {"player_name": "Brenen Thompson", "position": "WR", "school": "Mississippi State", "forty": 4.26, "ten_yard_split": 1.54},
    {"player_name": "Barion Brown", "position": "WR", "school": "LSU", "forty": 4.40, "ten_yard_split": 1.62},
    {"player_name": "Chris Brazzell", "position": "WR", "school": "Tennessee", "forty": 4.37, "ten_yard_split": 1.52},
    {"player_name": "Harrison Wallace III", "position": "WR", "school": "Mississippi", "forty": 4.54, "ten_yard_split": 1.58},
    {"player_name": "Carnell Tate", "position": "WR", "school": "Ohio State", "forty": 4.53, "ten_yard_split": 1.61},
    {"player_name": "Colbie Young", "position": "WR", "school": "Georgia", "forty": 4.49, "ten_yard_split": 1.59},
    {"player_name": "Denzel Boston", "position": "WR", "school": "Washington", "vertical_jump": 35.0, "twenty_yard_shuttle": 4.28},
    {"player_name": "Aaron Anderson", "position": "WR", "school": "LSU", "vertical_jump": 30.0, "broad_jump_inches": 113},
    # RBs not in DN
    {"player_name": "Jeremiyah Love", "position": "RB", "school": "Notre Dame", "forty": 4.36, "ten_yard_split": 1.55},
    # Jamarion Miller = "Jam Miller" in Dynasty Nerds data (already included there)
    {"player_name": "Rahsul Faison", "position": "RB", "school": "South Carolina", "vertical_jump": 37.5, "broad_jump_inches": 122},
    {"player_name": "J'Mari Taylor", "position": "RB", "school": "Virginia", "vertical_jump": 34.5, "broad_jump_inches": 115},
    # TEs not in DN
    {"player_name": "Khalil Dinkins", "position": "TE", "school": "Penn State", "forty": 4.72, "ten_yard_split": 1.70, "vertical_jump": 32.5, "broad_jump_inches": 119, "bench_press": 25, "three_cone": 7.28, "twenty_yard_shuttle": 4.33},
    {"player_name": "Nate Boerkircher", "position": "TE", "school": "Texas A&M", "twenty_yard_shuttle": 4.40},
    {"player_name": "Lake McRee", "position": "TE", "school": "USC", "bench_press": 22},
]

# ── Dynasty Nerds data (54 players) ─────────────────────────────────────

# TE data (17 players)
te_raw = [
    ("Sam Roush", "Stanford", "TE", 9.93, "6′6″", 267, 10.00, 30.63, 76.38, 4.70, 1.61, 38.5, "10′6″", 25, 7.08, 4.37),
    ("John Michael Gyllenborg", "Wyoming", "TE", 9.92, "6′5¾″", 249, 10.00, 31.00, None, 4.60, 1.56, 35.5, "10′8″", None, None, 4.22),
    ("Eli Raridon", "Notre Dame", "TE", 9.81, "6′6.1″", 245, 10.75, 32.75, None, 4.62, 1.59, 36.0, "10′3″", 20, None, None),
    ("Tanner Koziol", "Houston", "TE", 9.72, "6′6½″", 247, 9.75, 33.75, None, 4.70, 1.62, 36.5, "10′2″", None, None, None),
    ("Matt Hibner", "SMU", "TE", 9.56, "6′4¼″", 251, 9.38, 32.38, None, 4.57, 1.56, 37.0, "9′8″", 28, None, None),
    ("Eli Stowers", "Vanderbilt", "TE", 9.47, "6′3¾″", 239, 9.75, 32.63, 79.75, 4.51, 1.57, 45.5, "11′3″", None, None, None),
    ("Kenyon Sadiq", "Oregon", "TE", 9.44, "6′3.1″", 241, 10.00, 31.50, 78.25, 4.39, 1.54, 43.5, "11′0″", 26, None, None),
    ("Dallen Bentley", "Utah", "TE", 9.17, "6′4.1″", 253, 9.25, 33.13, None, 4.62, 1.62, 35.0, "9′10″", 24, None, 4.42),
    ("Will Kacmarek", "Ohio State", "TE", 9.14, "6′5½″", 261, 9.75, 32.25, None, 4.74, 1.64, 36.0, "9′11″", None, None, None),
    ("Bauer Sharp", "LSU", "TE", 9.09, "6′4⅝″", 249, 9.75, 31.75, None, 4.63, 1.59, 35.0, "10′0″", 20, None, None),
    ("Marlin Klein", "Michigan", "TE", 9.00, "6′6″", 248, 9.25, 32.38, None, 4.61, 1.63, 36.0, "9′9″", None, 7.42, None),
    ("Jack Endries", "Texas", "TE", 8.65, "6′4⅝″", 245, 9.63, 31.13, None, 4.62, 1.59, 36.0, "9′11″", None, None, None),
    ("RJ Maryland", "SMU", "TE", 8.58, "6′4.1″", 236, 9.00, 32.25, None, 4.51, 1.59, 33.0, "10′2″", None, None, None),
    ("Riley Nowakowski", "Indiana", "TE", 7.97, "6′2¼″", 250, 8.75, 31.50, None, 4.66, 1.62, 33.5, "9′11″", None, None, None),
    ("Jaren Kanak", "Oklahoma", "TE", 7.90, "6′2″", 234, 9.38, 30.50, None, 4.52, 1.59, 36.0, "9′11″", 24, None, None),
    ("Josh Cuevas", "Alabama", "TE", 7.56, "6′3⅜″", 245, 9.63, 30.63, None, 4.65, 1.62, 34.0, "9′10″", None, 7.37, 4.38),
    ("Miles Kitselman", "Tennessee", "TE", 5.83, "6′5.1″", 251, 9.88, 31.88, None, 4.90, 1.66, 34.5, "9′8″", None, None, None),
]

# WR data (28 players)
wr_raw = [
    ("Jeff Caldwell", "Cincinnati", "WR", 9.99, "6′5″", 216, 9.75, 32.63, None, 4.31, 1.50, 42.0, "11′2″", None, None, None),
    ("Bryce Lance", "North Dakota State", "WR", 9.98, "6′3″", 204, 9.25, 32.13, None, 4.34, 1.51, 41.5, "11′1″", None, None, None),
    ("J. Michael Sturdivant", "Florida", "WR", 9.96, "6′3″", 207, 9.50, 32.88, None, 4.40, 1.52, 39.0, "10′11″", None, None, None),
    ("Ted Hurst", "Georgia State", "WR", 9.92, "6′3″", 206, 9.75, 32.63, None, 4.42, 1.55, 36.5, "11′3″", None, None, None),
    ("De'Zhaun Stribling", "Ole Miss", "WR", 9.88, "6′2″", 207, 10.00, 31.63, None, 4.36, 1.52, 36.0, "10′7″", None, None, None),
    ("Skyler Bell", "UConn", "WR", 9.71, "5′11⅝″", 192, 10.00, 31.13, None, 4.40, 1.51, 41.0, "11′1″", None, None, None),
    ("Ja'Kobi Lane", "USC", "WR", 9.69, "6′4″", 200, 10.50, 32.63, None, 4.47, 1.51, 40.0, "10′9″", None, None, None),
    ("Dillon Bell", "Georgia", "WR", 9.57, "6′0¾″", 209, 9.50, 30.25, None, 4.50, 1.55, 39.0, "10′6″", None, None, None),
    ("Caleb Douglas", "Texas Tech", "WR", 9.52, "6′3½″", 206, 10.13, 32.50, None, 4.39, 1.53, 31.5, "10′6″", None, None, None),
    ("Kendrick Law", "Kentucky", "WR", 9.45, "5′11″", 203, 9.63, 31.13, None, 4.45, 1.53, 42.0, "10′8″", None, None, None),
    ("Deion Burks", "Oklahoma", "WR", 9.24, "5′9¾″", 180, 9.50, 29.38, 72.75, 4.30, 1.49, 42.5, "10′11″", None, None, None),
    ("Zavion Thomas", "LSU", "WR", 9.12, "5′10″", 190, 9.00, 30.50, None, 4.28, 1.51, 36.0, None, None, None, None),
    ("Antonio Williams", "Clemson", "WR", 9.09, "5′11½″", 187, 9.25, 30.75, 76.00, 4.41, 1.54, 39.5, "10′4″", None, None, None),
    ("Germie Bernard", "Alabama", "WR", 9.03, "6′1¼″", 206, 9.88, 30.38, 74.88, 4.48, 1.52, 32.5, "10′5″", None, 6.71, 4.31),
    ("Omar Cooper Jr.", "Indiana", "WR", 8.79, "6′0.1″", 199, 9.63, 30.25, 75.25, 4.42, 1.54, 37.0, None, None, None, None),
    ("Eric Rivers", "Georgia Tech", "WR", 8.74, "5′10″", 176, 9.00, 30.50, None, 4.35, 1.49, 37.0, "10′7″", None, None, None),
    ("Zachariah Branch", "Georgia", "WR", 8.30, "5′8⅝″", 177, 9.00, 29.38, 72.38, 4.35, 1.52, 38.0, "10′5″", None, None, None),
    ("Malachi Fields", "Notre Dame", "WR", 8.22, "6′4½″", 218, 9.00, 32.13, 79.00, 4.61, 1.58, 38.0, "10′4″", None, 6.98, 4.35),
    ("Kevin Coleman Jr.", "Missouri", "WR", 7.86, "5′10¼″", 179, 9.50, 30.00, None, 4.49, 1.52, 38.5, "10′6″", None, None, None),
    ("Malik Benson", "Oregon", "WR", 7.84, "6′0.1″", 189, 8.63, 31.88, None, 4.37, 1.53, 32.5, "10′2″", None, None, None),
    ("Vinny Anthony II", "Wisconsin", "WR", 7.68, "5′11⅞″", 183, 9.13, 31.38, None, 4.54, 1.52, 34.5, None, None, 6.86, 4.07),
    ("Emmanuel Henderson Jr.", "Kansas", "WR", 7.60, "6′0⅞″", 185, 8.38, 31.00, None, 4.44, 1.53, 35.0, "10′0″", None, None, None),
    ("Chase Roberts", "BYU", "WR", 7.60, "6′3″", 209, 9.75, 31.50, None, 4.64, 1.57, 37.0, "10′6″", None, None, None),
    ("Reggie Virgil", "Texas Tech", "WR", 7.34, "6′3″", 187, 9.25, 31.25, None, 4.57, 1.51, 36.0, "10′7″", None, None, None),
    ("Kaden Wetjen", "Iowa", "WR", 7.29, "5′9″", 193, 8.50, 29.63, None, 4.47, 1.51, 35.5, None, None, None, None),
    ("Jalen Walthall", "Incarnate Word", "WR", 6.78, "6′1″", 191, 9.00, 31.25, None, 4.57, 1.55, 35.0, "10′11″", None, None, None),
    ("Chris Hilton Jr.", "LSU", "WR", 6.50, "6′0″", 188, 9.25, 32.00, None, 4.41, 1.52, 34.5, "10′2″", None, 7.25, 4.41),
    ("Caullin Lacy", "Louisville", "WR", 3.66, "5′9″", 183, 9.88, 29.38, None, 4.55, 1.60, 33.5, "9′8″", None, None, None),
]

# RB data (9 players)
rb_raw = [
    ("Mike Washington Jr.", "Arkansas", "RB", 10.00, "6′1″", 223, 9.25, 33.63, None, 4.33, 1.51, 39.0, "10′8″", None, None, None),
    ("Seth McGowan", "Kentucky", "RB", 9.63, "6′0¼″", 223, 8.75, 31.38, None, 4.49, 1.52, 42.5, "10′11″", None, None, 4.50),
    ("Adam Randall", "Clemson", "RB", 9.42, "6′3⅜″", 232, 9.75, 32.38, None, 4.50, 1.54, 37.0, "10′4″", None, None, 4.53),
    ("Eli Heidenreich", "Navy", "RB", 9.12, "6′0″", 198, 9.50, 29.25, None, 4.44, 1.54, 35.5, "10′0″", None, None, None),
    ("Jadarian Price", "Notre Dame", "RB", 8.76, "5′10⅝″", 203, 9.63, 30.88, None, 4.49, 1.52, 35.0, "10′4″", None, None, None),
    ("Demond Claiborne", "Wake Forest", "RB", 8.48, "5′9¾″", 188, 9.00, 30.13, None, 4.37, 1.54, None, "10′2″", None, None, None),
    ("Robert Henry Jr.", "UTSA", "RB", 7.47, "5′9¼″", 196, 9.13, 30.13, None, 4.52, 1.56, 37.0, "10′4″", None, None, 4.31),
    ("Jam Miller", "Alabama", "RB", 7.04, "5′10¼″", 209, 8.75, 30.38, None, 4.42, 1.53, 30.5, "9′7″", None, None, None),
    ("Emmett Johnson", "Nebraska", "RB", 5.85, "5′10¼″", 202, 9.75, 30.25, None, 4.56, 1.59, 35.5, "10′0″", None, None, 4.29),
]

# ── Build Dynasty Nerds DataFrame ────────────────────────────────────────

all_dn = te_raw + wr_raw + rb_raw
dn_rows = []
for (name, school, pos, ras, ht_str, wt, hands, arms, wing, forty, ten_split,
     vert, broad_str, bench, three_cone, shuttle) in all_dn:
    dn_rows.append({
        "player_name": name,
        "position": pos,
        "school": school,
        "height_in": parse_height_to_inches(ht_str),
        "weight": float(wt) if wt else None,
        "hand_size": hands,
        "arm_length": arms,
        "wingspan": wing,
        "forty": forty,
        "ten_yard_split": ten_split,
        "vertical_jump": vert,
        "broad_jump_inches": parse_broad_to_inches(broad_str),
        "bench_press": float(bench) if bench else None,
        "three_cone": three_cone,
        "twenty_yard_shuttle": shuttle,
        "ras_unofficial": ras,
        "ras_source": "DynastyNerds",
    })

dn_df = pd.DataFrame(dn_rows)
print(f"Dynasty Nerds: {len(dn_df)} players ({len(te_raw)} TE, {len(wr_raw)} WR, {len(rb_raw)} RB)")

# ── Build MathBomb-only DataFrame ────────────────────────────────────────
mb_df = pd.DataFrame(mathbomb_only)
# Fill ras_source for those that have ras_unofficial
for col in ["height_in", "weight", "hand_size", "arm_length", "wingspan",
            "forty", "ten_yard_split", "vertical_jump", "broad_jump_inches",
            "bench_press", "three_cone", "twenty_yard_shuttle",
            "ras_unofficial", "ras_source"]:
    if col not in mb_df.columns:
        mb_df[col] = None
print(f"MathBomb-only: {len(mb_df)} players")

# ── Also add MathBomb RAS for DN players where MathBomb had a value ──────
# The original file had MathBomb RAS for many players. Let's preserve those
# as ras_mathbomb column while keeping Dynasty Nerds as the primary.
# MathBomb RAS values from original combine_2026.csv:
mathbomb_ras = {
    "Zavion Thomas": 9.12, "Deion Burks": 9.24, "Jeff Caldwell": 9.99,
    "Bryce Lance": 9.98, "Zachariah Branch": 8.30, "Eric Rivers": 8.74,
    "De'Zhaun Stribling": 9.88, "Malik Benson": 7.84, "Caleb Douglas": 9.52,
    "Skyler Bell": 9.71, "J.Michael Sturdivant": 9.96, "Chris Hilton": 6.50,
    "Antonio Williams": 9.09, "Omar Cooper Jr.": 8.79, "Ted Hurst": 9.92,
    "Emmanuel Henderson": 7.60, "Kendrick Law": 9.45, "Ja'Kobi Lane": 9.69,
    "Kaden Wetjen": 7.29, "Germie Bernard": 9.03, "Kevin Coleman Jr.": 7.86,
    "Dillon Bell": 9.57, "Vinny Anthony II": 7.68, "Caullin Lacy": 3.66,
    "Reggie Virgil": 7.34, "Jalen Walthall": 6.78, "Malachi Fields": 8.22,
    "Chase Roberts": 7.60,
    # RBs
    "Mike Washington Jr.": 10.00, "Demond Claiborne": 8.48, "Jamarion Miller": 7.04,
    "Eli Heidenreich": 9.12, "Seth McGowan": 9.63, "Jadarian Price": 8.76,
    "Adam Randall": 9.42, "Robert Henry Jr.": 7.47, "Emmett Johnson": 5.85,
    # TEs
    "Kenyon Sadiq": 9.44, "Eli Stowers": 9.47, "R.J. Maryland": 8.58,
    "Jaren Kanak": 7.90, "Matthew Hibner": 9.56, "John Michael Gyllenborg": 9.92,
    "Marlin Klein": 9.00, "Dallen Bentley": 9.17, "Jack Endries": 8.65,
    "Eli Raridon": 9.81, "Bauer Sharp": 9.09, "Josh Cuevas": 7.56,
    "Riley Nowakowski": 7.97, "Tanner Koziol": 9.72, "Sam Roush": 9.93,
    "Will Kacmarek": 9.14, "Miles Kitselman": 5.83,
}

# ── Combine both DataFrames ─────────────────────────────────────────────
combined = pd.concat([dn_df, mb_df], ignore_index=True)

# Sort: position then forty time then name
pos_order = {"WR": 0, "RB": 1, "TE": 2}
combined["_pos_sort"] = combined["position"].map(pos_order).fillna(3)
combined = combined.sort_values(["_pos_sort", "forty", "player_name"]).drop(columns=["_pos_sort"])
combined = combined.reset_index(drop=True)

# ── Column order ─────────────────────────────────────────────────────────
col_order = [
    "player_name", "position", "school",
    "height_in", "weight", "hand_size", "arm_length", "wingspan",
    "forty", "ten_yard_split", "vertical_jump", "broad_jump_inches",
    "bench_press", "three_cone", "twenty_yard_shuttle",
    "ras_unofficial", "ras_source"
]
combined = combined[col_order]

# ── Save ─────────────────────────────────────────────────────────────────
combined.to_csv("data/combine_2026.csv", index=False)
print(f"\nSaved data/combine_2026.csv: {len(combined)} rows, {len(combined.columns)} columns")
print(f"Columns: {list(combined.columns)}")

# ── Summary ──────────────────────────────────────────────────────────────
for pos in ["WR", "RB", "TE"]:
    sub = combined[combined["position"] == pos]
    h = sub["height_in"].notna().sum()
    w = sub["weight"].notna().sum()
    f = sub["forty"].notna().sum()
    r = sub["ras_unofficial"].notna().sum()
    hs = sub["hand_size"].notna().sum()
    al = sub["arm_length"].notna().sum()
    print(f"\n{pos}: {len(sub)} total | height={h} weight={w} forty={f} ras={r} hands={hs} arms={al}")
