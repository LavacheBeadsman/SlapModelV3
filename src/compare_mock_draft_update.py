"""Parse messy mock draft consensus paste and compare to existing mock_draft_2026.csv"""
import pandas as pd
import re
import os
os.chdir('/home/user/SlapModelV3')

# ============================================================================
# RAW PASTED DATA — manually cleaned from the user's paste
# Format: (projected_pick, player_name, position, college)
# ============================================================================
new_data = [
    (7, "Carnell Tate", "WR", "Ohio State"),
    (10, "Jordyn Tyson", "WR", "Arizona State"),
    (13, "Makai Lemon", "WR", "USC"),
    (22, "Denzel Boston", "WR", "Washington"),
    (26, "Kevin Concepcion", "WR", "Texas A&M"),
    (48, "Omar Cooper Jr.", "WR", "Indiana"),
    (49, "Zachariah Branch", "WR", "Georgia"),
    (50, "Malachi Fields", "WR", "Notre Dame"),
    (51, "Chris Bell", "WR", "Louisville"),
    (55, "Germie Bernard", "WR", "Alabama"),
    (57, "Chris Brazzell", "WR", "Tennessee"),
    (59, "Elijah Sarratt", "WR", "Indiana"),
    (68, "Antonio Williams", "WR", "Clemson"),
    (76, "Ja'Kobi Lane", "WR", "USC"),
    (82, "Ted Hurst", "WR", "Georgia State"),
    (99, "Skyler Bell", "WR", "UConn"),
    (118, "Deion Burks", "WR", "Oklahoma"),
    (121, "Kevin Coleman Jr.", "WR", "Missouri"),
    (131, "Brenen Thompson", "WR", "Mississippi State"),
    (132, "Bryce Lance", "WR", "North Dakota State"),
    (136, "Josh Cameron", "WR", "Baylor"),
    (138, "C.J. Daniels", "WR", "Miami (FL)"),
    (152, "Reggie Virgil", "WR", "Texas Tech"),
    (185, "Lewis Bond", "WR", "Boston College"),
    (186, "De'Zhaun Stribling", "WR", "Mississippi"),
    (192, "Caleb Douglas", "WR", "Texas Tech"),
    (193, "Eric McAlister", "WR", "TCU"),
    (198, "Chase Roberts", "WR", "BYU"),
    (211, "Aaron Anderson", "WR", "LSU"),
    (212, "Kaden Wetjen", "WR", "Iowa"),
    (219, "Cyrus Allen", "WR", "Cincinnati"),
    (223, "Eli Heidenreich", "WR", "Navy"),
    (225, "Barion Brown", "WR", "LSU"),
    (227, "Jordan Hudson", "WR", "SMU"),
    (229, "Tyren Montgomery", "WR", "John Carroll"),
    (237, "Dane Key", "WR", "Nebraska"),
    (242, "Eric Rivers", "WR", "Georgia Tech"),
    (250, "Emmanuel Henderson", "WR", "Kansas"),
    (251, "Zavion Thomas", "WR", "LSU"),
    (253, "Colbie Young", "WR", "Georgia"),
    (263, "Chris Hilton", "WR", "LSU"),
    (267, "Jeff Caldwell", "WR", "Cincinnati"),
    (276, "Devin Voisin", "WR", "South Alabama"),
    (278, "Hank Beatty", "WR", "Illinois"),
    (286, "Keelan Marion", "WR", "Miami (FL)"),
    (327, "Malik Benson", "WR", "Oregon"),
    (349, "Michael Wortham", "WR", "Montana"),
    (354, "J.Michael Sturdivant", "WR", "Florida"),
    (356, "Noah Thomas", "WR", "Georgia"),
    (357, "Vinny Anthony II", "WR", "Wisconsin"),
    (364, "Romello Brinson", "WR", "SMU"),
    (372, "Kendrick Law", "WR", "Kentucky"),
    (385, "Dillon Bell", "WR", "Georgia"),
    (386, "Harrison Wallace III", "WR", "Mississippi"),
    (390, "Squirrel White", "WR", "Florida State"),
    (391, "Caullin Lacy", "WR", "Louisville"),
    (435, "Jordan Dwyer", "WR", "TCU"),
    (436, "Griffin Wilde", "WR", "Northwestern"),
    (439, "Jalil Farooq", "WR", "Maryland"),
    (453, "Jalen Walthall", "WR", "Incarnate Word"),
    (456, "Amare Thomas", "WR", "Houston"),
    (465, "Trebor Pena", "WR", "Penn State"),
    (480, "Hykeem Williams", "WR", "Colorado"),
    (486, "Cordale Russell", "WR", "Miami (OH)"),
    (491, "Brandon Inniss", "WR", "Ohio State"),
    (506, "Anthony Evans III", "WR", "Mississippi State"),
    (512, "Raymond Cottrell", "WR", "West Alabama"),
    (522, "Nathan Leacock", "WR", "North Carolina"),
    (527, "Ryan Niblett", "WR", "Texas"),
    (528, "Jaquaize Pettaway", "WR", "East Carolina"),
    (531, "Jalen Brown", "WR", "Arkansas"),
    (535, "Braylon James", "WR", "TCU"),
    (557, "Shelton Sampson Jr.", "WR", "Louisiana-Lafayette"),
    (559, "Ayden Williams", "WR", "Mississippi State"),
    (562, "Mikey Matthews", "WR", "UCLA"),
    (564, "Tyler Brown", "WR", "Clemson"),
    (568, "London Humphreys", "WR", "Georgia"),
    (611, "Kyion Grayes", "WR", "California"),
    (613, "Tobias Merriweather", "WR", "Utah"),
    (624, "C.J. Williams", "WR", "Stanford"),
    (628, "Andre Greene Jr.", "WR", "Virginia"),
    (632, "Antonio Gates Jr.", "WR", "Delaware State"),
    (633, "Talyn Shettron", "WR", "Oklahoma State"),
    (635, "Kobe Prentice", "WR", "Baylor"),
    (641, "Kaleb Brown", "WR", "UAB"),
    (643, "Braylin Presley", "WR", "Tulsa"),
    (662, "Justus Ross-Simmons", "WR", "Syracuse"),
    (664, "Jaron Glover", "WR", "Mississippi State"),
    (674, "Jayden McGowan", "WR", "Charlotte"),
    (708, "Cody Jackson", "WR", "Tarleton State"),
    (713, "Kyron Ware-Hudson", "WR", "Penn State"),
    (714, "Jayden Ballard", "WR", "Wisconsin"),
    (717, "Deion Colzie", "WR", "Miami (OH)"),
    (729, "JoJo Earle", "WR", "UNLV"),
    (731, "Dacari Collins", "WR", "Louisville"),
    (751, "Malik McClain", "WR", "Arizona State"),
    (752, "Christian Leary", "WR", "Western Michigan"),
    (769, "Jerand Bradley", "WR", "Kansas State"),
    (771, "Rara Thomas", "WR", "Troy"),
    (772, "Andrel Anthony", "WR", "Duke"),
    (776, "Jared Brown", "WR", "South Carolina"),
    (778, "Joseph Manjack IV", "WR", "TCU"),
    (780, "Jaden Bray", "WR", "West Virginia"),
    (782, "Jayden Thomas", "WR", "Virginia"),
    (788, "Max Tomzcak", "WR", "Youngstown State"),
    (812, "E.J. Williams", "WR", "Indiana"),
    (827, "Ja'Mori Maclin", "WR", "Kentucky"),
    (840, "Ja'Varrius Johnson", "WR", "UCF"),
    (841, "Donavon Greene", "WR", "Virginia Tech"),
]

new_df = pd.DataFrame(new_data, columns=['projected_pick', 'player_name', 'position', 'college'])
print(f"Parsed {len(new_df)} WR prospects from new data")
print(f"Pick range: {new_df['projected_pick'].min()} to {new_df['projected_pick'].max()}")

# ============================================================================
# LOAD EXISTING DATA
# ============================================================================
old = pd.read_csv('data/mock_draft_2026.csv')
old_wr = old[old['position'] == 'WR'].copy()
old_rb = old[old['position'] == 'RB'].copy()
print(f"\nExisting mock_draft_2026.csv: {len(old_wr)} WRs, {len(old_rb)} RBs")

# ============================================================================
# COMPARISON: Name matching
# ============================================================================

def norm(name):
    return name.strip().lower().replace("'", "").replace(".", "").replace("-", " ")

old_wr_names = {norm(n): n for n in old_wr['player_name']}
new_names = {norm(n): n for n in new_df['player_name']}

# Find matches, new additions, and removals
matched = set(old_wr_names.keys()) & set(new_names.keys())
added = set(new_names.keys()) - set(old_wr_names.keys())
removed = set(old_wr_names.keys()) - set(new_names.keys())

print(f"\n{'='*100}")
print(f"WR COMPARISON SUMMARY")
print(f"{'='*100}")
print(f"  Matched:  {len(matched)} WRs in both old and new")
print(f"  NEW:      {len(added)} WRs only in new data")
print(f"  REMOVED:  {len(removed)} WRs only in old data (not in new)")
print(f"  RBs:      {len(old_rb)} RBs in old file (NOT in this paste — no RB data provided)")

# ============================================================================
# PICK CHANGES (matched players)
# ============================================================================
print(f"\n{'='*100}")
print(f"PICK CHANGES (matched WRs)")
print(f"{'='*100}")

old_lookup = {}
for _, r in old_wr.iterrows():
    old_lookup[norm(r['player_name'])] = int(r['projected_pick'])

new_lookup = {}
for _, r in new_df.iterrows():
    new_lookup[norm(r['player_name'])] = int(r['projected_pick'])

changes = []
unchanged = []
for n in sorted(matched, key=lambda x: new_lookup.get(x, 999)):
    old_pick = old_lookup[n]
    new_pick = new_lookup[n]
    delta = new_pick - old_pick
    if delta != 0:
        changes.append((new_names[n], old_pick, new_pick, delta))
    else:
        unchanged.append((new_names[n], old_pick))

print(f"\n  {'Player':<30s} {'Old Pick':>8s} {'New Pick':>8s} {'Change':>8s}")
print(f"  {'-'*60}")
for name, old_p, new_p, delta in sorted(changes, key=lambda x: x[3]):
    arrow = "↑" if delta < 0 else "↓"
    print(f"  {name:<30s} {old_p:>8d} {new_p:>8d} {delta:>+7d} {arrow}")

print(f"\n  Total changed: {len(changes)}")
print(f"  Unchanged:     {len(unchanged)}")

# Big movers
big_up = [c for c in changes if c[3] <= -10]
big_down = [c for c in changes if c[3] >= 10]
print(f"\n  Big risers (≥10 picks up): {len(big_up)}")
for name, old_p, new_p, delta in sorted(big_up, key=lambda x: x[3]):
    print(f"    {name}: {old_p} → {new_p} ({delta:+d})")
print(f"\n  Big fallers (≥10 picks down): {len(big_down)}")
for name, old_p, new_p, delta in sorted(big_down, key=lambda x: x[3], reverse=True):
    print(f"    {name}: {old_p} → {new_p} ({delta:+d})")

# ============================================================================
# NEW PLAYERS (in new data but not in old)
# ============================================================================
print(f"\n{'='*100}")
print(f"NEW PLAYERS (not in current mock_draft_2026.csv)")
print(f"{'='*100}")

for n in sorted(added, key=lambda x: new_lookup.get(x, 999)):
    r = new_df[new_df['player_name'].apply(norm) == n].iloc[0]
    print(f"  Pick {int(r['projected_pick']):>4d}: {r['player_name']:<30s} {r['college']}")

# ============================================================================
# REMOVED PLAYERS (in old data but not in new)
# ============================================================================
print(f"\n{'='*100}")
print(f"REMOVED PLAYERS (in old file but NOT in new data)")
print(f"{'='*100}")

for n in sorted(removed, key=lambda x: old_lookup.get(x, 999)):
    old_name = old_wr_names[n]
    old_row = old_wr[old_wr['player_name'].apply(norm) == n].iloc[0]
    print(f"  Pick {int(old_row['projected_pick']):>4d}: {old_name:<30s}")

# ============================================================================
# NOTE: RB data not in this paste
# ============================================================================
print(f"\n{'='*100}")
print(f"WARNING: NO RB DATA IN THIS PASTE")
print(f"{'='*100}")
print(f"  The new data contains {len(new_df)} WRs and 0 RBs.")
print(f"  The existing file has {len(old_rb)} RBs that would be unchanged.")
print(f"\n  RBs currently in mock_draft_2026.csv:")
for _, r in old_rb.sort_values('projected_pick').iterrows():
    print(f"    Pick {int(r['projected_pick']):>4d}: {r['player_name']}")
