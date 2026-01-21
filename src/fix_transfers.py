"""
SLAP Score V3 - Fix Transfer Data

Updates players who transferred to their new school and gets their stats.
"""

import os
import csv
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('CFBD_API_KEY')
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Transfer data discovered from API search
TRANSFERS = {
    "Kevin Concepcion": {"school": "Texas A&M", "rec_yards": 919},
    "Chris Bell": {"school": "Louisville", "rec_yards": 917},
    "Antonio Williams": {"school": "Clemson", "rec_yards": 604},
    "C.J. Daniels": {"school": "Miami", "rec_yards": 557},
    "Kevin Coleman Jr.": {"school": "Missouri", "rec_yards": 732},
    "Eric McAlister": {"school": "TCU", "rec_yards": 1190},
    "Eric Rivers": {"school": "Georgia Tech", "rec_yards": 671},
    "De'Zhaun Stribling": {"school": "Ole Miss", "rec_yards": 811},
    "Keelan Marion": {"school": "Miami", "rec_yards": 746},
    "Aaron Anderson": {"school": "LSU", "rec_yards": 398},
    "Dane Key": {"school": "Nebraska", "rec_yards": 452},
    "Caleb Douglas": {"school": "Texas Tech", "rec_yards": 830},
    "Malik Benson": {"school": "Oregon", "rec_yards": 719},
    "J.Michael Sturdivant": {"school": "Florida", "rec_yards": 406},
    "Barion Brown": {"school": "LSU", "rec_yards": 532},
    "Jordan Dwyer": {"school": "TCU", "rec_yards": 730},
    "Romello Brinson": {"school": "SMU", "rec_yards": 638},
    "Jalen Walthall": {"school": "Incarnate Word", "rec_yards": 847},
    "Chase Roberts": {"school": "BYU", "rec_yards": 802},
    "Emmanuel Henderson": {"school": "Kansas", "rec_yards": 766},
    "Anthony Evans III": {"school": "Mississippi State", "rec_yards": 831},
    "Jalen Brown": {"school": "Arkansas", "rec_yards": 167},
    "Tyler Brown": {"school": "Clemson", "rec_yards": 191},
    "Mikey Matthews": {"school": "UCLA", "rec_yards": 348},
    "Andre Greene Jr.": {"school": "Virginia", "rec_yards": 7},
    "Kobe Prentice": {"school": "Baylor", "rec_yards": 380},
    "Braylin Presley": {"school": "Tulsa", "rec_yards": 115},
    "Tobias Merriweather": {"school": "Utah", "rec_yards": 130},
    "Kaleb Brown": {"school": "UAB", "rec_yards": 255},
    "Jayden McGowan": {"school": "Charlotte", "rec_yards": 7},
    "Justus Ross-Simmons": {"school": "Syracuse", "rec_yards": 312},
    "Jayden Ballard": {"school": "Wisconsin", "rec_yards": 150},
    "Christian Leary": {"school": "Western Michigan", "rec_yards": 115},
    "JoJo Earle": {"school": "UNLV", "rec_yards": 258},
    "Dacari Collins": {"school": "Louisville", "rec_yards": 219},
    "Cody Jackson": {"school": "Tarleton State", "rec_yards": 799},
    "Malik McClain": {"school": "Arizona State", "rec_yards": 441},
    "Jerand Bradley": {"school": "Kansas State", "rec_yards": 184},
    "Jayden Thomas": {"school": "Virginia", "rec_yards": 9},
    "Rara Thomas": {"school": "Troy", "rec_yards": 629},
    "Jared Brown": {"school": "South Carolina", "rec_yards": 102},
    "Jaden Bray": {"school": "West Virginia", "rec_yards": 95},
    "E.J. Williams": {"school": "Indiana", "rec_yards": 438},
    "Jonah Coleman": {"school": "Washington", "rec_yards": 354},
    "J'Mari Taylor": {"school": "Virginia", "rec_yards": 253},
    "Kentrel Bullock": {"school": "South Alabama", "rec_yards": 53},
    "Terion Stewart": {"school": "Virginia Tech", "rec_yards": 19},
    "Branson Robinson": {"school": "Georgia State", "rec_yards": 22},
    "Andrew Paul": {"school": "Jacksonville State", "rec_yards": 29},
    "Richard Reese": {"school": "Stephen F. Austin", "rec_yards": 94},
    "Armoni Goodwin": {"school": "UT Martin", "rec_yards": 18},
    "Kadarius Calloway": {"school": "New Mexico State", "rec_yards": 175},
    "Byron Cardwell": {"school": "San Diego State", "rec_yards": 108},
    "Logan Diggs": {"school": "Ole Miss", "rec_yards": 56},
    "E.J. Smith": {"school": "Texas A&M", "rec_yards": 18},
    "Roydell Williams": {"school": "Florida State", "rec_yards": 19},
    "Dominic Richardson": {"school": "Tulsa", "rec_yards": 130},
    "Jalen Berger": {"school": "UCLA", "rec_yards": 70},
}

# Cache for team pass attempts
_team_cache = {}


def get_team_pass_attempts(team, year=2025):
    """Get team pass attempts from CFBD API."""
    if team in _team_cache:
        return _team_cache[team]

    url = f"{BASE_URL}/stats/season"
    params = {"year": year, "team": team}

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        for stat in response.json():
            if stat.get("statName") == "passAttempts":
                value = int(stat.get("statValue", 0))
                _team_cache[team] = value
                return value

    _team_cache[team] = None
    return None


def main():
    input_path = "data/prospects_final.csv"
    output_path = "data/prospects_final.csv"

    print("=" * 60)
    print("FIXING TRANSFER DATA")
    print("=" * 60)
    print()

    # Read current data
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        prospects = list(reader)

    print(f"Processing {len(TRANSFERS)} transfers...")
    print()

    updated = 0

    for p in prospects:
        name = p['player_name']
        if name in TRANSFERS:
            transfer = TRANSFERS[name]
            old_school = p['school']
            new_school = transfer['school']
            new_yards = transfer['rec_yards']

            # Get team pass attempts for new school
            new_pass_att = get_team_pass_attempts(new_school)

            # Update the record
            p['school'] = new_school
            p['rec_yards'] = new_yards
            if new_pass_att:
                p['team_pass_attempts'] = new_pass_att

            print(f"  {name}: {old_school} -> {new_school} ({new_yards} yards)")
            updated += 1

    print()
    print(f"Updated {updated} players")
    print()

    # Count coverage
    with_yards = len([p for p in prospects if p.get('rec_yards')])
    without_yards = len([p for p in prospects if not p.get('rec_yards')])

    print(f"Data coverage after fix:")
    print(f"  With rec_yards: {with_yards}")
    print(f"  Without rec_yards: {without_yards}")
    print()

    # Save
    print(f"Saving to: {output_path}")
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prospects)

    print("Done!")


if __name__ == "__main__":
    main()
