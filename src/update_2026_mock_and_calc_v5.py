"""
Update 2026 Mock Draft + Recalculate ALL 2026 SLAP V5 Scores
=============================================================
1. Update mock_draft_2026.csv with Feb 2026 consensus (109 WR + 56 RB, 250 cap)
2. Look up CFBD data for 5 new players
3. Update prospects_final.csv (pick changes + new players)
4. Look up breakout ages for 3 new WRs
5. Calculate V5 SLAP for all 2026 WRs and RBs
6. Show rankings and biggest movers
"""

import pandas as pd
import numpy as np
import requests
import time
import warnings, os, json
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

UDFA_CAP = 250
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE")
CFBD_HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}

# ============================================================================
# HELPERS
# ============================================================================
def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()

def wr_breakout_score(breakout_age, dominator_pct):
    if pd.isna(breakout_age) or breakout_age is None:
        if dominator_pct is not None and pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base = age_tiers.get(int(breakout_age), 20)
    bonus = 0
    if dominator_pct is not None and pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
    return min(base + bonus, 99.9)

def wr_enhanced_breakout(breakout_age, dominator_pct, rush_yards):
    base = wr_breakout_score(breakout_age, dominator_pct)
    rush_bonus = 5 if pd.notna(rush_yards) and rush_yards >= 20 else 0
    return min(base + rush_bonus, 99.9)

def rb_production_score(rec_yards, team_pass_att, age):
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return np.nan
    if pd.isna(age): age = 22
    age = float(age)
    season_age = age - 1
    age_w = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
    raw = (rec_yards / team_pass_att) * age_w * 100
    return min(99.9, max(0, raw / 1.75))

def speed_score_fn(weight, forty):
    if pd.isna(weight) or pd.isna(forty) or forty <= 0 or weight <= 0:
        return np.nan
    return (weight * 200) / (forty ** 4)

def cfbd_get(url, params=None):
    try:
        resp = requests.get(url, headers=CFBD_HEADERS, params=params, timeout=15)
        time.sleep(0.3)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None


# ============================================================================
# STEP 1: UPDATE mock_draft_2026.csv
# ============================================================================
print("=" * 100)
print("STEP 1: UPDATING mock_draft_2026.csv")
print("=" * 100)

# New consensus data (from compare script)
wr_data = [
    (7, "Carnell Tate", "WR", "Ohio State"), (10, "Jordyn Tyson", "WR", "Arizona State"),
    (13, "Makai Lemon", "WR", "USC"), (22, "Denzel Boston", "WR", "Washington"),
    (26, "Kevin Concepcion", "WR", "Texas A&M"), (48, "Omar Cooper Jr.", "WR", "Indiana"),
    (49, "Zachariah Branch", "WR", "Georgia"), (50, "Malachi Fields", "WR", "Notre Dame"),
    (51, "Chris Bell", "WR", "Louisville"), (55, "Germie Bernard", "WR", "Alabama"),
    (57, "Chris Brazzell", "WR", "Tennessee"), (59, "Elijah Sarratt", "WR", "Indiana"),
    (68, "Antonio Williams", "WR", "Clemson"), (76, "Ja'Kobi Lane", "WR", "USC"),
    (82, "Ted Hurst", "WR", "Georgia State"), (99, "Skyler Bell", "WR", "UConn"),
    (118, "Deion Burks", "WR", "Oklahoma"), (121, "Kevin Coleman Jr.", "WR", "Missouri"),
    (131, "Brenen Thompson", "WR", "Mississippi State"), (132, "Bryce Lance", "WR", "North Dakota State"),
    (136, "Josh Cameron", "WR", "Baylor"), (138, "C.J. Daniels", "WR", "Miami (FL)"),
    (152, "Reggie Virgil", "WR", "Texas Tech"), (185, "Lewis Bond", "WR", "Boston College"),
    (186, "De'Zhaun Stribling", "WR", "Mississippi"), (192, "Caleb Douglas", "WR", "Texas Tech"),
    (193, "Eric McAlister", "WR", "TCU"), (198, "Chase Roberts", "WR", "BYU"),
    (211, "Aaron Anderson", "WR", "LSU"), (212, "Kaden Wetjen", "WR", "Iowa"),
    (219, "Cyrus Allen", "WR", "Cincinnati"), (223, "Eli Heidenreich", "WR", "Navy"),
    (225, "Barion Brown", "WR", "LSU"), (227, "Jordan Hudson", "WR", "SMU"),
    (229, "Tyren Montgomery", "WR", "John Carroll"), (237, "Dane Key", "WR", "Nebraska"),
    (242, "Eric Rivers", "WR", "Georgia Tech"), (250, "Emmanuel Henderson", "WR", "Kansas"),
    (251, "Zavion Thomas", "WR", "LSU"), (253, "Colbie Young", "WR", "Georgia"),
    (263, "Chris Hilton", "WR", "LSU"), (267, "Jeff Caldwell", "WR", "Cincinnati"),
    (276, "Devin Voisin", "WR", "South Alabama"), (278, "Hank Beatty", "WR", "Illinois"),
    (286, "Keelan Marion", "WR", "Miami (FL)"), (327, "Malik Benson", "WR", "Oregon"),
    (349, "Michael Wortham", "WR", "Montana"), (354, "J.Michael Sturdivant", "WR", "Florida"),
    (356, "Noah Thomas", "WR", "Georgia"), (357, "Vinny Anthony II", "WR", "Wisconsin"),
    (364, "Romello Brinson", "WR", "SMU"), (372, "Kendrick Law", "WR", "Kentucky"),
    (385, "Dillon Bell", "WR", "Georgia"), (386, "Harrison Wallace III", "WR", "Mississippi"),
    (390, "Squirrel White", "WR", "Florida State"), (391, "Caullin Lacy", "WR", "Louisville"),
    (435, "Jordan Dwyer", "WR", "TCU"), (436, "Griffin Wilde", "WR", "Northwestern"),
    (439, "Jalil Farooq", "WR", "Maryland"), (453, "Jalen Walthall", "WR", "Incarnate Word"),
    (456, "Amare Thomas", "WR", "Houston"), (465, "Trebor Pena", "WR", "Penn State"),
    (480, "Hykeem Williams", "WR", "Colorado"), (486, "Cordale Russell", "WR", "Miami (OH)"),
    (491, "Brandon Inniss", "WR", "Ohio State"), (506, "Anthony Evans III", "WR", "Mississippi State"),
    (512, "Raymond Cottrell", "WR", "West Alabama"), (522, "Nathan Leacock", "WR", "North Carolina"),
    (527, "Ryan Niblett", "WR", "Texas"), (528, "Jaquaize Pettaway", "WR", "East Carolina"),
    (531, "Jalen Brown", "WR", "Arkansas"), (535, "Braylon James", "WR", "TCU"),
    (557, "Shelton Sampson Jr.", "WR", "Louisiana-Lafayette"),
    (559, "Ayden Williams", "WR", "Mississippi State"), (562, "Mikey Matthews", "WR", "UCLA"),
    (564, "Tyler Brown", "WR", "Clemson"), (568, "London Humphreys", "WR", "Georgia"),
    (611, "Kyion Grayes", "WR", "California"), (613, "Tobias Merriweather", "WR", "Utah"),
    (624, "C.J. Williams", "WR", "Stanford"), (628, "Andre Greene Jr.", "WR", "Virginia"),
    (632, "Antonio Gates Jr.", "WR", "Delaware State"), (633, "Talyn Shettron", "WR", "Oklahoma State"),
    (635, "Kobe Prentice", "WR", "Baylor"), (641, "Kaleb Brown", "WR", "UAB"),
    (643, "Braylin Presley", "WR", "Tulsa"), (662, "Justus Ross-Simmons", "WR", "Syracuse"),
    (664, "Jaron Glover", "WR", "Mississippi State"), (674, "Jayden McGowan", "WR", "Charlotte"),
    (708, "Cody Jackson", "WR", "Tarleton State"), (713, "Kyron Ware-Hudson", "WR", "Penn State"),
    (714, "Jayden Ballard", "WR", "Wisconsin"), (717, "Deion Colzie", "WR", "Miami (OH)"),
    (729, "JoJo Earle", "WR", "UNLV"), (731, "Dacari Collins", "WR", "Louisville"),
    (751, "Malik McClain", "WR", "Arizona State"), (752, "Christian Leary", "WR", "Western Michigan"),
    (769, "Jerand Bradley", "WR", "Kansas State"), (771, "Rara Thomas", "WR", "Troy"),
    (772, "Andrel Anthony", "WR", "Duke"), (776, "Jared Brown", "WR", "South Carolina"),
    (778, "Joseph Manjack IV", "WR", "TCU"), (780, "Jaden Bray", "WR", "West Virginia"),
    (782, "Jayden Thomas", "WR", "Virginia"), (788, "Max Tomzcak", "WR", "Youngstown State"),
    (812, "E.J. Williams", "WR", "Indiana"), (827, "Ja'Mori Maclin", "WR", "Kentucky"),
    (840, "Ja'Varrius Johnson", "WR", "UCF"), (841, "Donavon Greene", "WR", "Virginia Tech"),
]

rb_data = [
    (9, "Jeremiyah Love", "RB", "Notre Dame"), (61, "Jadarian Price", "RB", "Notre Dame"),
    (73, "Jonah Coleman", "RB", "Washington"), (74, "Emmett Johnson", "RB", "Nebraska"),
    (100, "Mike Washington Jr.", "RB", "Arkansas"), (101, "Kaytron Allen", "RB", "Penn State"),
    (115, "Nick Singleton", "RB", "Penn State"), (172, "Demond Claiborne", "RB", "Wake Forest"),
    (182, "Kaelon Black", "RB", "Indiana"), (183, "Roman Hemby", "RB", "Indiana"),
    (190, "Adam Randall", "RB", "Clemson"), (191, "Seth McGowan", "RB", "Kentucky"),
    (207, "J'Mari Taylor", "RB", "Virginia"), (215, "Robert Henry Jr.", "RB", "UTSA"),
    (222, "Jaydn Ott", "RB", "Oklahoma"), (226, "Le'Veon Moss", "RB", "Texas A&M"),
    (238, "Noah Whittington", "RB", "Oregon"), (245, "Jamarion Miller", "RB", "Alabama"),
    (256, "Kejon Owens", "RB", "Florida International"),
    (263, "Curtis Allen", "RB", "Virginia Union"),
    (266, "C.J. Donaldson", "RB", "Ohio State"),
    (298, "Kentrel Bullock", "RB", "South Alabama"), (299, "Max Bredeson", "RB", "Michigan"),
    (346, "Jamal Haynes", "RB", "Georgia Tech"), (347, "Chip Trayanum", "RB", "Toledo"),
    (355, "Desmond Reid", "RB", "Pittsburgh"), (363, "Dean Connors", "RB", "Houston"),
    (367, "Rahsul Faison", "RB", "South Carolina"), (376, "Terion Stewart", "RB", "Virginia Tech"),
    (398, "Rueben Owens", "RB", "Texas A&M"), (404, "Samuel Singleton Jr.", "RB", "Florida State"),
    (429, "Djay Braswell", "RB", "Georgia State"), (430, "Kedrick Reescano", "RB", "Arizona"),
    (434, "Quinten Joyner", "RB", "Texas Tech"), (446, "Sedrick Alexander", "RB", "Vanderbilt"),
    (449, "Kaden Feagin", "RB", "Illinois"), (462, "TreVonte Citizen", "RB", "McNeese State"),
    (467, "Jaylon Glover", "RB", "UNLV"), (470, "Branson Robinson", "RB", "Georgia State"),
    (509, "Richard Reese", "RB", "Stephen F. Austin"),
    (510, "Andrew Paul", "RB", "Jacksonville State"),
    (534, "Savion Red", "RB", "Sacramento State"),
    (553, "Kadarius Calloway", "RB", "New Mexico State"),
    (582, "Derrick Davis Jr.", "RB", "Pittsburgh"),
    (593, "Armoni Goodwin", "RB", "UT Martin"), (627, "Trevion Cooley", "RB", "Troy"),
    (653, "L.J. Johnson Jr.", "RB", "California"),
    (657, "Byron Cardwell", "RB", "San Diego State"),
    (696, "Barika Kpeenu", "RB", "North Dakota State"),
    (710, "Alton McCaskill IV", "RB", "Sam Houston State"),
    (718, "Logan Diggs", "RB", "Mississippi"), (744, "E.J. Smith", "RB", "Texas A&M"),
    (763, "Roydell Williams", "RB", "Florida State"),
    (766, "Dominic Richardson", "RB", "Tulsa"), (818, "Jalen Berger", "RB", "UCLA"),
    (825, "Cam Porter", "RB", "Northwestern"),
]

all_data = wr_data + rb_data
mock_df = pd.DataFrame(all_data, columns=['projected_pick', 'player_name', 'position', 'college'])
mock_df['projected_pick'] = mock_df['projected_pick'].clip(upper=UDFA_CAP)

# Save updated mock draft
mock_df[['player_name', 'position', 'projected_pick', 'college']].to_csv('data/mock_draft_2026.csv', index=False)
print(f"  Saved: data/mock_draft_2026.csv ({len(mock_df)} players: {len(wr_data)} WR + {len(rb_data)} RB)")
print(f"  Added college column")

# ============================================================================
# STEP 2: CFBD LOOKUPS FOR 5 NEW PLAYERS
# ============================================================================
print(f"\n{'='*100}")
print("STEP 2: CFBD LOOKUPS FOR NEW PLAYERS")
print("=" * 100)

new_players = {
    'WR': [
        ("Tyren Montgomery", "John Carroll", 229),   # D3 — likely not in CFBD
        ("Devin Voisin", "South Alabama", 250),       # Sun Belt — should be in CFBD
        ("Michael Wortham", "Montana", 250),           # FCS Big Sky — may be in CFBD
    ],
    'RB': [
        ("Kejon Owens", "Florida International", 250), # C-USA — should be in CFBD
        ("Curtis Allen", "Virginia Union", 250),        # D2 — likely not in CFBD
    ]
}

# CFBD school name mapping
cfbd_school_names = {
    "South Alabama": "South Alabama",
    "Montana": "Montana",
    "Florida International": "Florida International",
    "John Carroll": None,  # D3 — not in CFBD
    "Virginia Union": None,  # D2 — not in CFBD
}

new_player_data = {}

for pos, players in new_players.items():
    for name, school, pick in players:
        print(f"\n  Looking up: {name} ({school}, {pos})...")
        data = {'player_name': name, 'position': pos, 'school': school,
                'projected_pick': min(pick, UDFA_CAP), 'rec_yards': None,
                'team_pass_attempts': None, 'birthdate': 'MISSING',
                'age': None, 'age_estimated': True, 'weight': None}

        cfbd_school = cfbd_school_names.get(school)
        if cfbd_school is None:
            print(f"    {school} is D2/D3 — not in CFBD. Flagging as MISSING.")
            new_player_data[name] = data
            continue

        # Search for player
        result = cfbd_get(f"https://api.collegefootballdata.com/player/search",
                         {'searchTerm': name, 'position': pos})
        if result:
            for p in result:
                if normalize_name(p.get('name', '')) == normalize_name(name):
                    if p.get('weight'):
                        data['weight'] = int(p['weight'])
                        print(f"    Weight: {data['weight']}")
                    break

        # Get 2025 season receiving/rushing stats
        season = 2025
        if pos == 'WR':
            # Get player receiving yards from the 2025 season
            result = cfbd_get(f"https://api.collegefootballdata.com/stats/player/season",
                             {'year': season, 'team': cfbd_school, 'category': 'receiving'})
            if result:
                for r in result:
                    if normalize_name(r.get('player', '')) == normalize_name(name):
                        if r.get('stat') == 'YDS':
                            data['rec_yards'] = float(r['stat']) if r['stat'] != 'YDS' else None
                        # CFBD returns separate rows per stat type
                        pass

                # Try parsing the structured data
                player_stats = {}
                for r in result:
                    if normalize_name(r.get('player', '')) == normalize_name(name):
                        player_stats[r.get('statType', '')] = r.get('stat')
                if 'YDS' in player_stats:
                    data['rec_yards'] = float(player_stats['YDS'])
                    print(f"    Rec yards: {data['rec_yards']}")

            # Get team pass attempts
            result = cfbd_get(f"https://api.collegefootballdata.com/stats/season",
                             {'year': season, 'team': cfbd_school})
            if result:
                for r in result:
                    if r.get('statName') == 'passAttempts':
                        data['team_pass_attempts'] = float(r['statValue'])
                        print(f"    Team pass attempts: {data['team_pass_attempts']}")
                        break

        elif pos == 'RB':
            # Get receiving stats for RB
            result = cfbd_get(f"https://api.collegefootballdata.com/stats/player/season",
                             {'year': season, 'team': cfbd_school, 'category': 'receiving'})
            if result:
                player_stats = {}
                for r in result:
                    if normalize_name(r.get('player', '')) == normalize_name(name):
                        player_stats[r.get('statType', '')] = r.get('stat')
                if 'YDS' in player_stats:
                    data['rec_yards'] = float(player_stats['YDS'])
                    print(f"    Rec yards: {data['rec_yards']}")

            # Get team pass attempts
            result = cfbd_get(f"https://api.collegefootballdata.com/stats/season",
                             {'year': season, 'team': cfbd_school})
            if result:
                for r in result:
                    if r.get('statName') == 'passAttempts':
                        data['team_pass_attempts'] = float(r['statValue'])
                        print(f"    Team pass attempts: {data['team_pass_attempts']}")
                        break

        new_player_data[name] = data

print(f"\n  Summary of new player lookups:")
for name, d in new_player_data.items():
    rec = d.get('rec_yards', 'MISSING')
    tpa = d.get('team_pass_attempts', 'MISSING')
    wt = d.get('weight', 'MISSING')
    print(f"    {name}: rec_yards={rec}, team_pass_att={tpa}, weight={wt}")


# ============================================================================
# STEP 3: UPDATE prospects_final.csv
# ============================================================================
print(f"\n{'='*100}")
print("STEP 3: UPDATING prospects_final.csv")
print("=" * 100)

prospects = pd.read_csv('data/prospects_final.csv')
print(f"  Original: {len(prospects)} prospects")

# Build lookup from new mock data
norm_map = {}
for _, r in mock_df.iterrows():
    norm_map[normalize_name(r['player_name'])] = r

# Update projected_pick for existing players
updated_picks = 0
for idx in prospects.index:
    nn = normalize_name(prospects.loc[idx, 'player_name'])
    if nn in norm_map:
        new_pick = norm_map[nn]['projected_pick']
        old_pick = prospects.loc[idx, 'projected_pick']
        if old_pick != new_pick:
            prospects.loc[idx, 'projected_pick'] = new_pick
            updated_picks += 1
        # Also update school if it changed in new data
        new_school = norm_map[nn]['college']
        if pd.notna(new_school):
            prospects.loc[idx, 'school'] = new_school

print(f"  Updated picks: {updated_picks}")

# Remove Jaden Greathouse (removed from mock)
greathouse_mask = prospects['player_name'].apply(normalize_name) == normalize_name('Jaden Greathouse')
if greathouse_mask.sum() > 0:
    prospects = prospects[~greathouse_mask].reset_index(drop=True)
    print(f"  Removed: Jaden Greathouse (no longer in mock)")

# Add new players
existing_norms = set(prospects['player_name'].apply(normalize_name))
added = []
for name, d in new_player_data.items():
    if normalize_name(name) not in existing_norms:
        row = {
            'player_name': d['player_name'],
            'position': d['position'],
            'school': d['school'],
            'projected_pick': d['projected_pick'],
            'rec_yards': d.get('rec_yards'),
            'team_pass_attempts': d.get('team_pass_attempts'),
            'birthdate': d.get('birthdate', 'MISSING'),
            'age': d.get('age', 'MISSING'),
            'age_estimated': d.get('age_estimated', 'MISSING'),
            'weight': d.get('weight'),
        }
        prospects = pd.concat([prospects, pd.DataFrame([row])], ignore_index=True)
        added.append(name)
        print(f"  Added: {name} ({d['position']}, {d['school']})")

# Estimate age for new players if missing
for idx in prospects.index:
    if prospects.loc[idx, 'player_name'] in added:
        if pd.isna(prospects.loc[idx, 'age']) or prospects.loc[idx, 'age'] == 'MISSING':
            # Default age estimate based on position
            prospects.loc[idx, 'age'] = 22
            prospects.loc[idx, 'age_estimated'] = True

prospects.to_csv('data/prospects_final.csv', index=False)
print(f"  Saved: data/prospects_final.csv ({len(prospects)} prospects)")


# ============================================================================
# STEP 4: CALCULATE V5 SLAP FOR ALL 2026 PROSPECTS
# ============================================================================
print(f"\n{'='*100}")
print("STEP 4: CALCULATING V5 SLAP SCORES")
print("=" * 100)

# V5 WEIGHTS
WR_V5 = {'dc': 0.70, 'breakout': 0.20, 'teammate': 0.05, 'early_declare': 0.05}
RB_V5 = {'dc': 0.65, 'production': 0.30, 'speed_score': 0.05}

# --- Load WR breakout age data ---
wr_bo = pd.read_csv('data/wr_breakout_ages_2026.csv')
wr_bo_lookup = {}
for _, r in wr_bo.iterrows():
    wr_bo_lookup[normalize_name(r['player_name'])] = {
        'breakout_age': r.get('breakout_age'),
        'peak_dominator': r.get('peak_dominator'),
    }

# --- Load WR backtest for rush yards and early declare data ---
# 2026 WR components file doesn't exist; use wr_breakout_ages for breakout data
# Rush yards and early declare need different sources

# For rush yards: check if CFBD has rushing data for 2026 WR prospects
# For early declare: estimate from age (age <= 21 at draft → likely early declare)
# For teammate score: calculate from mock draft (same school WR/TE with DC > 0)

# --- WR SCORING ---
print(f"\n  --- WR V5 SCORING ---")
wr_prospects = prospects[prospects['position'] == 'WR'].copy()
wr_prospects['age'] = pd.to_numeric(wr_prospects['age'], errors='coerce')
wr_prospects['projected_pick'] = pd.to_numeric(wr_prospects['projected_pick'], errors='coerce')

# DC Score
wr_prospects['dc_score'] = wr_prospects['projected_pick'].apply(dc_score)

# Enhanced Breakout Score
wr_prospects['breakout_age'] = np.nan
wr_prospects['peak_dominator'] = np.nan
wr_prospects['rush_yards'] = np.nan

for idx in wr_prospects.index:
    nn = normalize_name(wr_prospects.loc[idx, 'player_name'])
    bo = wr_bo_lookup.get(nn, {})
    if bo:
        ba = bo.get('breakout_age')
        pd_val = bo.get('peak_dominator')
        if pd.notna(ba): wr_prospects.loc[idx, 'breakout_age'] = float(ba)
        if pd.notna(pd_val): wr_prospects.loc[idx, 'peak_dominator'] = float(pd_val)

# Try to get rush yards from CFBD for ALL WR prospects (batch by school)
print(f"  Looking up WR rush yards from CFBD...")
wr_rush_lookup = {}
schools_checked = set()
for idx in wr_prospects.index:
    school = wr_prospects.loc[idx, 'school']
    name = wr_prospects.loc[idx, 'player_name']
    nn = normalize_name(name)

    if school in schools_checked:
        if nn in wr_rush_lookup:
            wr_prospects.loc[idx, 'rush_yards'] = wr_rush_lookup[nn]
        continue

    # Fetch rushing stats for this school's 2025 season
    result = cfbd_get("https://api.collegefootballdata.com/stats/player/season",
                     {'year': 2025, 'team': school, 'category': 'rushing'})
    if result:
        for r in result:
            pnn = normalize_name(r.get('player', ''))
            if r.get('statType') == 'YDS':
                try:
                    wr_rush_lookup[pnn] = float(r['stat'])
                except:
                    pass
    schools_checked.add(school)

    if nn in wr_rush_lookup:
        wr_prospects.loc[idx, 'rush_yards'] = wr_rush_lookup[nn]

n_rush = wr_prospects['rush_yards'].notna().sum()
print(f"  Rush yards found for {n_rush}/{len(wr_prospects)} WRs")

# Enhanced breakout
wr_prospects['enhanced_breakout'] = wr_prospects.apply(
    lambda r: wr_enhanced_breakout(r['breakout_age'], r['peak_dominator'], r['rush_yards']), axis=1)

# Teammate Score: calculate from mock draft
# For each WR, find other WRs/TEs from same school in mock, sum their DC scores
print(f"  Calculating teammate scores from mock draft...")
mock_wr_te = mock_df[mock_df['position'].isin(['WR', 'TE'])].copy()
mock_wr_te['dc'] = mock_wr_te['projected_pick'].apply(dc_score)
mock_wr_te['nn'] = mock_wr_te['player_name'].apply(normalize_name)
mock_wr_te['school_norm'] = mock_wr_te['college'].apply(lambda x: str(x).strip().lower())

wr_prospects['school_norm'] = wr_prospects['school'].apply(lambda x: str(x).strip().lower())
wr_prospects['nn'] = wr_prospects['player_name'].apply(normalize_name)

wr_prospects['teammate_dc'] = 0.0
for idx in wr_prospects.index:
    nn = wr_prospects.loc[idx, 'nn']
    school = wr_prospects.loc[idx, 'school_norm']
    # Find teammates: same school, different player, WR or TE
    teammates = mock_wr_te[(mock_wr_te['school_norm'] == school) & (mock_wr_te['nn'] != nn)]
    total_dc = teammates['dc'].sum()
    wr_prospects.loc[idx, 'teammate_dc'] = total_dc

wr_prospects['teammate_score'] = wr_prospects['teammate_dc'].apply(
    lambda x: 100 if x > 150 else 0)

n_tm = (wr_prospects['teammate_score'] == 100).sum()
print(f"  Teammate = 100: {n_tm}/{len(wr_prospects)} WRs")

# Early Declare Score: based on age AND college seasons
# Rule: A player is early declare ONLY if they played 3 or fewer college seasons.
# If they played 4+ seasons, early_declare = 0 regardless of age.
# Age is used as a proxy only when college season data is unavailable.

# Build seasons lookup from breakout ages file (seasons_found = CFBD college seasons)
seasons_2026_lookup = {}
for _, r in wr_bo.iterrows():
    if pd.notna(r.get('seasons_found')):
        seasons_2026_lookup[r['player_name']] = int(r['seasons_found'])

wr_prospects['early_declare'] = 0
for idx in wr_prospects.index:
    name = wr_prospects.loc[idx, 'player_name']
    age = wr_prospects.loc[idx, 'age']
    college_seasons = seasons_2026_lookup.get(name)

    # If we have college season data, use it as the definitive check
    if college_seasons is not None and college_seasons >= 4:
        wr_prospects.loc[idx, 'early_declare'] = 0  # 4+ seasons = true senior
        continue

    # Age-based estimate (only if seasons < 4 or unknown)
    if pd.notna(age) and float(age) <= 21:
        wr_prospects.loc[idx, 'early_declare'] = 100
    # Also check birthdate-based age
    bd = wr_prospects.loc[idx, 'birthdate'] if 'birthdate' in wr_prospects.columns else None
    if pd.notna(bd) and bd != 'MISSING':
        try:
            birth = pd.to_datetime(bd)
            draft_date = pd.Timestamp('2026-04-25')
            draft_age = (draft_date - birth).days / 365.25
            if draft_age <= 21.5:
                wr_prospects.loc[idx, 'early_declare'] = 100
        except:
            pass

    # Final override: if we know they played 4+ seasons, force to 0
    if college_seasons is not None and college_seasons >= 4:
        wr_prospects.loc[idx, 'early_declare'] = 0

n_ed = (wr_prospects['early_declare'] == 100).sum()
print(f"  Early declare = 100: {n_ed}/{len(wr_prospects)} WRs")

# V5 SLAP
wr_prospects['slap_v5'] = (
    WR_V5['dc'] * wr_prospects['dc_score'] +
    WR_V5['breakout'] * wr_prospects['enhanced_breakout'] +
    WR_V5['teammate'] * wr_prospects['teammate_score'] +
    WR_V5['early_declare'] * wr_prospects['early_declare']
)
wr_prospects['delta_vs_dc'] = wr_prospects['slap_v5'] - wr_prospects['dc_score']

print(f"  WR V5 SLAP range: {wr_prospects['slap_v5'].min():.1f} - {wr_prospects['slap_v5'].max():.1f}")


# --- RB SCORING ---
print(f"\n  --- RB V5 SCORING ---")
rb_prospects = prospects[prospects['position'] == 'RB'].copy()
rb_prospects['age'] = pd.to_numeric(rb_prospects['age'], errors='coerce')
rb_prospects['projected_pick'] = pd.to_numeric(rb_prospects['projected_pick'], errors='coerce')
rb_prospects['rec_yards'] = pd.to_numeric(rb_prospects['rec_yards'], errors='coerce')
rb_prospects['team_pass_attempts'] = pd.to_numeric(rb_prospects['team_pass_attempts'], errors='coerce')
rb_prospects['weight'] = pd.to_numeric(rb_prospects['weight'], errors='coerce')

# DC Score
rb_prospects['dc_score'] = rb_prospects['projected_pick'].apply(dc_score)

# Production Score
rb_prospects['production_score'] = rb_prospects.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_attempts'], r['age']), axis=1)

# Backtest average for missing production
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
rb_bt['prod'] = rb_bt.apply(
    lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
avg_prod = rb_bt['prod'].dropna().mean()
rb_prospects['production_final'] = rb_prospects['production_score'].fillna(avg_prod)
rb_prospects['production_status'] = np.where(rb_prospects['production_score'].notna(), 'observed', 'imputed')

n_imputed = (rb_prospects['production_status'] == 'imputed').sum()
print(f"  Production: {len(rb_prospects) - n_imputed} observed, {n_imputed} imputed (avg={avg_prod:.1f})")

# Speed Score (from backtest normalization)
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine['name_norm'] = combine['player_name'].apply(normalize_name)
combine_lookup = {}
for pos_pref in ['RB', 'FB', 'WR', 'TE']:
    sub = combine[combine['pos'] == pos_pref]
    for _, row in sub.iterrows():
        dy = row.get('draft_year')
        if pd.isna(dy): dy = row.get('season')
        if pd.isna(dy): continue
        key = (row['name_norm'], int(dy))
        if key not in combine_lookup:
            combine_lookup[key] = {'weight': row['wt'], 'forty': row['forty']}

rb_bt['name_norm'] = rb_bt['player_name'].apply(normalize_name)
rb_bt['bt_weight'] = rb_bt.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight', np.nan), axis=1)
rb_bt['bt_forty'] = rb_bt.apply(
    lambda r: combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty', np.nan), axis=1)
rb_bt['bt_weight'] = pd.to_numeric(rb_bt['bt_weight'], errors='coerce')
rb_bt['bt_forty'] = pd.to_numeric(rb_bt['bt_forty'], errors='coerce')
rb_bt['raw_ss'] = rb_bt.apply(lambda r: speed_score_fn(r['bt_weight'], r['bt_forty']), axis=1)
ss_min = rb_bt['raw_ss'].dropna().min()
ss_max = rb_bt['raw_ss'].dropna().max()

# Build weight × round → avg 40 lookup
known = rb_bt[rb_bt['bt_weight'].notna() & rb_bt['bt_forty'].notna()].copy()
def wt_bucket(wt):
    if pd.isna(wt): return None
    if wt < 200: return '<200'
    elif wt < 210: return '200-209'
    elif wt < 220: return '210-219'
    else: return '220+'
def rd_bucket(rd):
    if rd <= 1: return 'Rd 1'
    elif rd <= 2: return 'Rd 2'
    elif rd <= 4: return 'Rd 3-4'
    else: return 'Rd 5+'

known['wb'] = known['bt_weight'].apply(wt_bucket)
known['rb_bucket'] = known['round'].apply(rd_bucket)
lookup_40 = {}
for wb in ['<200', '200-209', '210-219', '220+']:
    for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
        sub = known[(known['wb'] == wb) & (known['rb_bucket'] == rdb)]
        if len(sub) > 0:
            lookup_40[(wb, rdb)] = sub['bt_forty'].mean()
    wt_sub = known[known['wb'] == wb]
    if len(wt_sub) > 0:
        for rdb in ['Rd 1', 'Rd 2', 'Rd 3-4', 'Rd 5+']:
            if (wb, rdb) not in lookup_40:
                lookup_40[(wb, rdb)] = wt_sub['bt_forty'].mean()

p60 = rb_bt['raw_ss'].quantile(0.60)
p40 = rb_bt['raw_ss'].quantile(0.40)

# Calculate speed scores for 2026 RBs
rb_prospects['projected_round'] = np.ceil(rb_prospects['projected_pick'] / 32).astype(int)
rb_prospects['raw_ss'] = np.nan
rb_prospects['ss_status'] = 'imputed_mnar'

for idx in rb_prospects.index:
    wt = rb_prospects.loc[idx, 'weight']
    rd = rb_prospects.loc[idx, 'projected_round']
    if pd.notna(wt):
        wb = wt_bucket(wt)
        rdb = rd_bucket(rd)
        est_40 = lookup_40.get((wb, rdb))
        if est_40 is not None:
            rb_prospects.loc[idx, 'raw_ss'] = speed_score_fn(wt, est_40)
            rb_prospects.loc[idx, 'ss_status'] = 'weight_est40'
    if pd.isna(rb_prospects.loc[idx, 'raw_ss']):
        rb_prospects.loc[idx, 'raw_ss'] = p60 if rd <= 2 else p40

rb_prospects['speed_score'] = ((rb_prospects['raw_ss'] - ss_min) / (ss_max - ss_min) * 100).clip(0, 100)

# V5 SLAP
rb_prospects['slap_v5'] = (
    RB_V5['dc'] * rb_prospects['dc_score'] +
    RB_V5['production'] * rb_prospects['production_final'] +
    RB_V5['speed_score'] * rb_prospects['speed_score']
)
rb_prospects['delta_vs_dc'] = rb_prospects['slap_v5'] - rb_prospects['dc_score']

print(f"  RB V5 SLAP range: {rb_prospects['slap_v5'].min():.1f} - {rb_prospects['slap_v5'].max():.1f}")


# ============================================================================
# STEP 5: SAVE OLD SCORES BEFORE SHOWING MOVERS
# ============================================================================
# Load old SLAP scores (before pick changes) to calculate movers
# Recompute using OLD projected picks
old_mock = pd.DataFrame({
    'player_name': ['placeholder'],
    'projected_pick': [250],
})
# We'll compute movers by comparing old vs new pick-based DC scores
# Since SLAP is heavily DC-driven, the biggest movers will be those with biggest pick changes


# ============================================================================
# STEP 6: OUTPUT RANKINGS
# ============================================================================
print(f"\n\n{'='*100}")
print("TOP 30 WRs BY SLAP V5 SCORE")
print("=" * 100)

wr_ranked = wr_prospects.sort_values('slap_v5', ascending=False).head(30)
print(f"\n{'Rank':>4s}  {'Player':<28s} {'School':<22s} {'Pick':>4s} {'SLAP':>6s} {'DC':>5s} {'BO':>5s} {'TM':>3s} {'ED':>3s} {'Delta':>6s}")
print(f"{'':>4s}  {'':28s} {'':22s} {'':>4s} {'':>6s} {'':>5s} {'':>5s} {'':>3s} {'':>3s} {'':>6s}")
print("-" * 110)
for rank, (_, r) in enumerate(wr_ranked.iterrows(), 1):
    tm = int(r['teammate_score']) if pd.notna(r['teammate_score']) else 0
    ed = int(r['early_declare']) if pd.notna(r['early_declare']) else 0
    print(f"{rank:>4d}  {r['player_name']:<28s} {str(r['school']):<22s} {int(r['projected_pick']):>4d} "
          f"{r['slap_v5']:>6.1f} {r['dc_score']:>5.1f} {r['enhanced_breakout']:>5.1f} "
          f"{tm//100:>3d} {ed//100:>3d} {r['delta_vs_dc']:>+6.1f}")

print(f"\n\n{'='*100}")
print("TOP 20 RBs BY SLAP V5 SCORE")
print("=" * 100)

rb_ranked = rb_prospects.sort_values('slap_v5', ascending=False).head(20)
print(f"\n{'Rank':>4s}  {'Player':<28s} {'School':<22s} {'Pick':>4s} {'SLAP':>6s} {'DC':>5s} {'Prod':>5s} {'SS':>5s} {'Delta':>6s} {'Status'}")
print("-" * 115)
for rank, (_, r) in enumerate(rb_ranked.iterrows(), 1):
    status = r['production_status']
    flag = '*' if status == 'imputed' else ''
    print(f"{rank:>4d}  {r['player_name']:<28s} {str(r['school']):<22s} {int(r['projected_pick']):>4d} "
          f"{r['slap_v5']:>6.1f} {r['dc_score']:>5.1f} {r['production_final']:>5.1f} {r['speed_score']:>5.1f} "
          f"{r['delta_vs_dc']:>+6.1f} {flag}")


# ============================================================================
# STEP 7: BIGGEST MOVERS (from pick changes)
# ============================================================================
print(f"\n\n{'='*100}")
print("BIGGEST SLAP SCORE MOVERS (from Feb 2026 mock draft update)")
print("=" * 100)

# Load the old mock_draft picks (from git, before update)
# We already saved the new file, so calculate what old DC scores were
# Old pick data is embedded in our comparison — we know the changes
old_picks = {
    # WR changes from comparison output
    'Carnell Tate': 4, 'Jordyn Tyson': 7, 'Makai Lemon': 16, 'Denzel Boston': 24,
    'Kevin Concepcion': 27, 'Omar Cooper Jr.': 76, 'Zachariah Branch': 47,
    'Malachi Fields': 89, 'Chris Bell': 43, 'Germie Bernard': 52,
    'Chris Brazzell': 53, 'Elijah Sarratt': 44, 'Antonio Williams': 85,
    'Ja\'Kobi Lane': 65, 'Ted Hurst': 132, 'Skyler Bell': 88,
    'Deion Burks': 121, 'Kevin Coleman Jr.': 162, 'Brenen Thompson': 113,
    'Bryce Lance': 150, 'Josh Cameron': 194, 'C.J. Daniels': 112,
    'Reggie Virgil': 216, 'Lewis Bond': 185, 'De\'Zhaun Stribling': 186,
    'Caleb Douglas': 212, 'Eric McAlister': 171, 'Chase Roberts': 250,
    'Aaron Anderson': 199, 'Kaden Wetjen': 250, 'Cyrus Allen': 250,
    'Eli Heidenreich': 250, 'Barion Brown': 250, 'Jordan Hudson': 211,
    'Dane Key': 202, 'Eric Rivers': 178, 'Emmanuel Henderson': 250,
    'Zavion Thomas': 250, 'Colbie Young': 250, 'Chris Hilton': 250,
    'Jeff Caldwell': 250, 'Hank Beatty': 250, 'Keelan Marion': 193,
    'Malik Benson': 250, 'J.Michael Sturdivant': 250, 'Noah Thomas': 196,
    'Vinny Anthony II': 224, 'Romello Brinson': 250, 'Kendrick Law': 250,
    'Dillon Bell': 250, 'Harrison Wallace III': 250, 'Squirrel White': 250,
    'Caullin Lacy': 231, 'Jordan Dwyer': 250, 'Griffin Wilde': 250,
    'Jalil Farooq': 250, 'Jalen Walthall': 250, 'Amare Thomas': 250,
    'Trebor Pena': 250, 'Hykeem Williams': 250, 'Cordale Russell': 250,
    'Brandon Inniss': 250, 'Anthony Evans III': 250, 'Raymond Cottrell': 250,
    'Nathan Leacock': 250, 'Ryan Niblett': 250, 'Jaquaize Pettaway': 250,
    'Jalen Brown': 250, 'Braylon James': 250, 'Shelton Sampson Jr.': 250,
    'Ayden Williams': 250, 'Mikey Matthews': 250, 'Tyler Brown': 250,
    'London Humphreys': 250, 'Kyion Grayes': 250, 'Tobias Merriweather': 250,
    'C.J. Williams': 250, 'Andre Greene Jr.': 250, 'Antonio Gates Jr.': 250,
    'Talyn Shettron': 250, 'Kobe Prentice': 250, 'Kaleb Brown': 250,
    'Braylin Presley': 250, 'Justus Ross-Simmons': 250, 'Jaron Glover': 250,
    'Jayden McGowan': 250, 'Cody Jackson': 250, 'Kyron Ware-Hudson': 250,
    'Jayden Ballard': 250, 'Deion Colzie': 250, 'JoJo Earle': 250,
    'Dacari Collins': 250, 'Malik McClain': 250, 'Christian Leary': 250,
    'Jerand Bradley': 250, 'Rara Thomas': 250, 'Andrel Anthony': 250,
    'Jared Brown': 250, 'Joseph Manjack IV': 250, 'Jaden Bray': 250,
    'Jayden Thomas': 250, 'Max Tomzcak': 250, 'E.J. Williams': 250,
    'Ja\'Mori Maclin': 250, 'Ja\'Varrius Johnson': 250, 'Donavon Greene': 250,
    # RB changes
    'Jeremiyah Love': 10, 'Jadarian Price': 79, 'Jonah Coleman': 56,
    'Emmett Johnson': 83, 'Nick Singleton': 118, 'Kaytron Allen': 124,
    'Demond Claiborne': 136, 'Mike Washington Jr.': 163, 'Adam Randall': 180,
    'Noah Whittington': 183, 'Roman Hemby': 197, 'C.J. Donaldson': 242,
    'J\'Mari Taylor': 245, 'Jamarion Miller': 246, 'Robert Henry Jr.': 250,
    'Max Bredeson': 250, 'Kentrel Bullock': 250, 'Chip Trayanum': 250,
    'Seth McGowan': 250, 'Le\'Veon Moss': 250, 'Dean Connors': 250,
    'Rahsul Faison': 250, 'Desmond Reid': 250, 'Kaelon Black': 250,
    'Jaydn Ott': 250, 'Terion Stewart': 250, 'Jamal Haynes': 250,
    'Rueben Owens': 250, 'Kedrick Reescano': 250, 'Djay Braswell': 250,
    'Quinten Joyner': 250, 'Samuel Singleton Jr.': 250,
    'Sedrick Alexander': 250, 'Kaden Feagin': 250, 'TreVonte Citizen': 250,
    'Branson Robinson': 250, 'Jaylon Glover': 250, 'Savion Red': 250,
    'Andrew Paul': 250, 'Richard Reese': 250, 'Armoni Goodwin': 250,
    'L.J. Johnson Jr.': 250, 'Trevion Cooley': 250, 'Derrick Davis Jr.': 250,
    'Kadarius Calloway': 250, 'Byron Cardwell': 250, 'Barika Kpeenu': 250,
    'Alton McCaskill IV': 250, 'Logan Diggs': 250, 'E.J. Smith': 250,
    'Roydell Williams': 250, 'Dominic Richardson': 250, 'Jalen Berger': 250,
    'Cam Porter': 250,
}

# Calculate SLAP score change for each player
movers = []

# WR movers
for idx, r in wr_prospects.iterrows():
    name = r['player_name']
    old_pick = old_picks.get(name)
    new_pick = r['projected_pick']
    if old_pick is None:
        continue  # New player, skip
    if old_pick == new_pick:
        continue
    # Recalculate old SLAP with old DC
    old_dc = dc_score(old_pick)
    old_slap = (WR_V5['dc'] * old_dc +
                WR_V5['breakout'] * r['enhanced_breakout'] +
                WR_V5['teammate'] * r['teammate_score'] +
                WR_V5['early_declare'] * r['early_declare'])
    new_slap = r['slap_v5']
    delta = new_slap - old_slap
    if abs(delta) > 0.1:
        movers.append({
            'player_name': name, 'position': 'WR', 'school': r['school'],
            'old_pick': old_pick, 'new_pick': new_pick,
            'old_slap': old_slap, 'new_slap': new_slap, 'slap_delta': delta
        })

# RB movers
for idx, r in rb_prospects.iterrows():
    name = r['player_name']
    old_pick = old_picks.get(name)
    new_pick = r['projected_pick']
    if old_pick is None:
        continue
    if old_pick == new_pick:
        continue
    old_dc = dc_score(old_pick)
    old_slap = (RB_V5['dc'] * old_dc +
                RB_V5['production'] * r['production_final'] +
                RB_V5['speed_score'] * r['speed_score'])
    new_slap = r['slap_v5']
    delta = new_slap - old_slap
    if abs(delta) > 0.1:
        movers.append({
            'player_name': name, 'position': 'RB', 'school': r['school'],
            'old_pick': old_pick, 'new_pick': new_pick,
            'old_slap': old_slap, 'new_slap': new_slap, 'slap_delta': delta
        })

movers_df = pd.DataFrame(movers)

if len(movers_df) > 0:
    # Top 10 risers
    risers = movers_df.nlargest(10, 'slap_delta')
    print(f"\n  TOP 10 RISERS (SLAP score increase from pick changes)")
    print(f"  {'Player':<28s} {'Pos':>3s} {'School':<20s} {'Old':>4s} {'New':>4s} {'Old SLAP':>9s} {'New SLAP':>9s} {'Delta':>7s}")
    print(f"  {'-'*95}")
    for _, r in risers.iterrows():
        print(f"  {r['player_name']:<28s} {r['position']:>3s} {str(r['school'])[:20]:<20s} "
              f"{int(r['old_pick']):>4d} {int(r['new_pick']):>4d} {r['old_slap']:>9.1f} {r['new_slap']:>9.1f} {r['slap_delta']:>+7.1f}")

    # Top 10 fallers
    fallers = movers_df.nsmallest(10, 'slap_delta')
    print(f"\n  TOP 10 FALLERS (SLAP score decrease from pick changes)")
    print(f"  {'Player':<28s} {'Pos':>3s} {'School':<20s} {'Old':>4s} {'New':>4s} {'Old SLAP':>9s} {'New SLAP':>9s} {'Delta':>7s}")
    print(f"  {'-'*95}")
    for _, r in fallers.iterrows():
        print(f"  {r['player_name']:<28s} {r['position']:>3s} {str(r['school'])[:20]:<20s} "
              f"{int(r['old_pick']):>4d} {int(r['new_pick']):>4d} {r['old_slap']:>9.1f} {r['new_slap']:>9.1f} {r['slap_delta']:>+7.1f}")

# ============================================================================
# STEP 8: SAVE 2026 SLAP V5 OUTPUT
# ============================================================================
print(f"\n\n{'='*100}")
print("SAVING 2026 V5 OUTPUT FILES")
print("=" * 100)

# WR output
wr_out = wr_prospects[['player_name', 'school', 'projected_pick', 'slap_v5', 'dc_score',
                        'enhanced_breakout', 'teammate_score', 'early_declare',
                        'delta_vs_dc', 'breakout_age', 'peak_dominator', 'rush_yards']].copy()
wr_out = wr_out.sort_values('slap_v5', ascending=False).reset_index(drop=True)
wr_out.index = wr_out.index + 1
wr_out.index.name = 'rank'
for c in ['slap_v5', 'dc_score', 'enhanced_breakout', 'delta_vs_dc']:
    wr_out[c] = wr_out[c].round(1)
wr_out.to_csv('output/slap_v5_wr_2026.csv')
print(f"  Saved: output/slap_v5_wr_2026.csv ({len(wr_out)} WRs)")

# RB output
rb_out = rb_prospects[['player_name', 'school', 'projected_pick', 'slap_v5', 'dc_score',
                        'production_final', 'speed_score', 'delta_vs_dc',
                        'production_status', 'ss_status', 'rec_yards', 'team_pass_attempts', 'weight']].copy()
rb_out.columns = ['player_name', 'school', 'projected_pick', 'slap_v5', 'dc_score',
                   'production_score', 'speed_score', 'delta_vs_dc',
                   'production_status', 'speed_score_status', 'rec_yards', 'team_pass_attempts', 'weight']
rb_out = rb_out.sort_values('slap_v5', ascending=False).reset_index(drop=True)
rb_out.index = rb_out.index + 1
rb_out.index.name = 'rank'
for c in ['slap_v5', 'dc_score', 'production_score', 'speed_score', 'delta_vs_dc']:
    rb_out[c] = rb_out[c].round(1)
rb_out.to_csv('output/slap_v5_rb_2026.csv')
print(f"  Saved: output/slap_v5_rb_2026.csv ({len(rb_out)} RBs)")

print(f"\n{'='*100}")
print("2026 V5 SLAP RECALCULATION COMPLETE")
print("=" * 100)
