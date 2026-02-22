"""Parse messy mock draft consensus paste and compare to existing mock_draft_2026.csv
Includes both WR and RB data. All picks capped at 250 for UDFA prospects."""
import pandas as pd
import re
import os
os.chdir('/home/user/SlapModelV3')

UDFA_CAP = 250  # Max projected_pick for undrafted prospects

# ============================================================================
# RAW PASTED DATA — manually cleaned from the user's paste
# Format: (projected_pick, player_name, position, college)
# ============================================================================

# --- WR DATA (109 WRs) ---
wr_data = [
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

# --- RB DATA (56 RBs) ---
rb_data = [
    (9, "Jeremiyah Love", "RB", "Notre Dame"),
    (61, "Jadarian Price", "RB", "Notre Dame"),
    (73, "Jonah Coleman", "RB", "Washington"),
    (74, "Emmett Johnson", "RB", "Nebraska"),
    (100, "Mike Washington Jr.", "RB", "Arkansas"),
    (101, "Kaytron Allen", "RB", "Penn State"),
    (115, "Nick Singleton", "RB", "Penn State"),
    (172, "Demond Claiborne", "RB", "Wake Forest"),
    (182, "Kaelon Black", "RB", "Indiana"),
    (183, "Roman Hemby", "RB", "Indiana"),
    (190, "Adam Randall", "RB", "Clemson"),
    (191, "Seth McGowan", "RB", "Kentucky"),
    (207, "J'Mari Taylor", "RB", "Virginia"),
    (215, "Robert Henry Jr.", "RB", "UTSA"),
    (222, "Jaydn Ott", "RB", "Oklahoma"),
    (226, "Le'Veon Moss", "RB", "Texas A&M"),
    (238, "Noah Whittington", "RB", "Oregon"),
    (245, "Jamarion Miller", "RB", "Alabama"),
    (256, "Kejon Owens", "RB", "Florida International"),
    (263, "Curtis Allen", "RB", "Virginia Union"),
    (266, "C.J. Donaldson", "RB", "Ohio State"),
    (298, "Kentrel Bullock", "RB", "South Alabama"),
    (299, "Max Bredeson", "RB", "Michigan"),
    (346, "Jamal Haynes", "RB", "Georgia Tech"),
    (347, "Chip Trayanum", "RB", "Toledo"),
    (355, "Desmond Reid", "RB", "Pittsburgh"),
    (363, "Dean Connors", "RB", "Houston"),
    (367, "Rahsul Faison", "RB", "South Carolina"),
    (376, "Terion Stewart", "RB", "Virginia Tech"),
    (398, "Rueben Owens", "RB", "Texas A&M"),
    (404, "Samuel Singleton Jr.", "RB", "Florida State"),
    (429, "Djay Braswell", "RB", "Georgia State"),
    (430, "Kedrick Reescano", "RB", "Arizona"),
    (434, "Quinten Joyner", "RB", "Texas Tech"),
    (446, "Sedrick Alexander", "RB", "Vanderbilt"),
    (449, "Kaden Feagin", "RB", "Illinois"),
    (462, "TreVonte Citizen", "RB", "McNeese State"),
    (467, "Jaylon Glover", "RB", "UNLV"),
    (470, "Branson Robinson", "RB", "Georgia State"),
    (509, "Richard Reese", "RB", "Stephen F. Austin"),
    (510, "Andrew Paul", "RB", "Jacksonville State"),
    (534, "Savion Red", "RB", "Sacramento State"),
    (553, "Kadarius Calloway", "RB", "New Mexico State"),
    (582, "Derrick Davis Jr.", "RB", "Pittsburgh"),
    (593, "Armoni Goodwin", "RB", "UT Martin"),
    (627, "Trevion Cooley", "RB", "Troy"),
    (653, "L.J. Johnson Jr.", "RB", "California"),
    (657, "Byron Cardwell", "RB", "San Diego State"),
    (696, "Barika Kpeenu", "RB", "North Dakota State"),
    (710, "Alton McCaskill IV", "RB", "Sam Houston State"),
    (718, "Logan Diggs", "RB", "Mississippi"),
    (744, "E.J. Smith", "RB", "Texas A&M"),
    (763, "Roydell Williams", "RB", "Florida State"),
    (766, "Dominic Richardson", "RB", "Tulsa"),
    (818, "Jalen Berger", "RB", "UCLA"),
    (825, "Cam Porter", "RB", "Northwestern"),
]

# Combine all data
new_data = wr_data + rb_data
new_df = pd.DataFrame(new_data, columns=['projected_pick', 'player_name', 'position', 'college'])

# Apply UDFA cap: max projected_pick = 250
new_df['raw_pick'] = new_df['projected_pick']  # Keep original for reference
new_df['projected_pick'] = new_df['projected_pick'].clip(upper=UDFA_CAP)

n_capped = (new_df['raw_pick'] > UDFA_CAP).sum()
print(f"Parsed {len(wr_data)} WRs + {len(rb_data)} RBs = {len(new_df)} total prospects")
print(f"Applied UDFA cap: {n_capped} players capped from raw pick to {UDFA_CAP}")

# ============================================================================
# LOAD EXISTING DATA
# ============================================================================
old = pd.read_csv('data/mock_draft_2026.csv')
old_wr = old[old['position'] == 'WR'].copy()
old_rb = old[old['position'] == 'RB'].copy()
print(f"\nExisting mock_draft_2026.csv: {len(old_wr)} WRs, {len(old_rb)} RBs, {len(old)} total")

# ============================================================================
# HELPER
# ============================================================================
def norm(name):
    return name.strip().lower().replace("'", "").replace(".", "").replace("-", " ")

# ============================================================================
# POSITION-BY-POSITION COMPARISON
# ============================================================================
for pos in ['WR', 'RB']:
    old_pos = old[old['position'] == pos].copy()
    new_pos = new_df[new_df['position'] == pos].copy()

    old_names = {norm(n): n for n in old_pos['player_name']}
    new_names = {norm(n): n for n in new_pos['player_name']}

    matched = set(old_names.keys()) & set(new_names.keys())
    added = set(new_names.keys()) - set(old_names.keys())
    removed = set(old_names.keys()) - set(new_names.keys())

    old_lookup = {}
    for _, r in old_pos.iterrows():
        old_lookup[norm(r['player_name'])] = int(r['projected_pick'])

    new_lookup = {}
    for _, r in new_pos.iterrows():
        new_lookup[norm(r['player_name'])] = int(r['projected_pick'])

    print(f"\n{'='*100}")
    print(f"{pos} COMPARISON SUMMARY")
    print(f"{'='*100}")
    print(f"  Old count: {len(old_pos):>4d}")
    print(f"  New count: {len(new_pos):>4d}")
    print(f"  Matched:   {len(matched):>4d} (in both old and new)")
    print(f"  NEW:       {len(added):>4d} (only in new data)")
    print(f"  REMOVED:   {len(removed):>4d} (only in old data)")

    # ---- PICK CHANGES ----
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

    if changes:
        print(f"\n  PICK CHANGES (matched {pos}s)")
        print(f"  {'Player':<30s} {'Old Pick':>8s} {'New Pick':>8s} {'Change':>8s}")
        print(f"  {'-'*60}")
        for name, old_p, new_p, delta in sorted(changes, key=lambda x: x[3]):
            arrow = "↑" if delta < 0 else "↓"
            print(f"  {name:<30s} {old_p:>8d} {new_p:>8d} {delta:>+7d} {arrow}")

        print(f"\n  Total changed: {len(changes)}")
        print(f"  Unchanged:     {len(unchanged)}")

        # Big movers
        big_up = [c for c in changes if c[3] <= -10]
        big_down = [c for c in changes if c[3] >= 10]
        if big_up:
            print(f"\n  Big risers (≥10 picks up): {len(big_up)}")
            for name, old_p, new_p, delta in sorted(big_up, key=lambda x: x[3]):
                print(f"    {name}: {old_p} → {new_p} ({delta:+d})")
        if big_down:
            print(f"\n  Big fallers (≥10 picks down): {len(big_down)}")
            for name, old_p, new_p, delta in sorted(big_down, key=lambda x: x[3], reverse=True):
                print(f"    {name}: {old_p} → {new_p} ({delta:+d})")
    else:
        print(f"\n  No pick changes for matched {pos}s")

    # ---- NEW PLAYERS ----
    if added:
        print(f"\n  NEW {pos}s (not in current mock_draft_2026.csv)")
        print(f"  {'-'*60}")
        for n in sorted(added, key=lambda x: new_lookup.get(x, 999)):
            r = new_pos[new_pos['player_name'].apply(norm) == n].iloc[0]
            raw = new_df[(new_df['player_name'].apply(norm) == n) & (new_df['position'] == pos)].iloc[0]['raw_pick']
            cap_note = f" (raw: {int(raw)}, capped to {UDFA_CAP})" if raw > UDFA_CAP else ""
            print(f"    Pick {int(r['projected_pick']):>4d}: {r['player_name']:<30s} {r['college']}{cap_note}")

    # ---- REMOVED PLAYERS ----
    if removed:
        print(f"\n  REMOVED {pos}s (in old file but NOT in new data)")
        print(f"  {'-'*60}")
        for n in sorted(removed, key=lambda x: old_lookup.get(x, 999)):
            old_name = old_names[n]
            old_row = old_pos[old_pos['player_name'].apply(norm) == n].iloc[0]
            print(f"    Pick {int(old_row['projected_pick']):>4d}: {old_name:<30s}")

# ============================================================================
# SCHOOL CHANGES (players in both old and new but with different colleges)
# ============================================================================
print(f"\n{'='*100}")
print(f"SCHOOL CHANGES (players whose college changed)")
print(f"{'='*100}")

old_college = {}
for _, r in old.iterrows():
    old_college[norm(r['player_name'])] = r.get('college', '')

new_college = {}
for _, r in new_df.iterrows():
    new_college[norm(r['player_name'])] = r['college']

school_changes = []
for n in set(old_college.keys()) & set(new_college.keys()):
    old_c = str(old_college.get(n, '')).strip()
    new_c = str(new_college.get(n, '')).strip()
    if old_c and new_c and old_c.lower() != new_c.lower():
        school_changes.append((n, old_c, new_c))

if school_changes:
    for n, old_c, new_c in sorted(school_changes):
        print(f"  {n}: {old_c} → {new_c}")
else:
    print("  (no school changes detected — note: old file may not have college column)")

# ============================================================================
# OVERALL SUMMARY
# ============================================================================
print(f"\n{'='*100}")
print(f"OVERALL SUMMARY")
print(f"{'='*100}")
print(f"  Old file:  {len(old_wr)} WRs + {len(old_rb)} RBs = {len(old)} total")

new_wr_count = len(new_df[new_df['position'] == 'WR'])
new_rb_count = len(new_df[new_df['position'] == 'RB'])
print(f"  New data:  {new_wr_count} WRs + {new_rb_count} RBs = {len(new_df)} total")
print(f"  UDFA cap:  {n_capped} players capped at pick {UDFA_CAP}")
print(f"\n  *** mock_draft_2026.csv has NOT been updated ***")
print(f"  *** Review the comparison above before approving the update ***")
