"""
SLAP Score V3 - Fix Missing Stats with Alternate Name Formats

This script tries alternate name formats to find receiving stats for players
who weren't matched by the standard search.
"""

import os
import csv
import re
import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def get_name_variations(name):
    """
    Generate alternate name formats to try matching.
    Returns a list of possible variations.
    """
    variations = [name]

    # Common nickname mappings
    NICKNAMES = {
        'nick': 'nicholas',
        'nicholas': 'nick',
        'mike': 'michael',
        'michael': 'mike',
        'chris': 'christopher',
        'christopher': 'chris',
        'will': 'william',
        'william': 'will',
        'tom': 'thomas',
        'thomas': 'tom',
        'dan': 'daniel',
        'daniel': 'dan',
        'rob': 'robert',
        'robert': 'rob',
        'bob': 'robert',
        'joe': 'joseph',
        'joseph': 'joe',
        'tony': 'anthony',
        'anthony': 'tony',
        'matt': 'matthew',
        'matthew': 'matt',
        'jon': 'jonathan',
        'jonathan': 'jon',
        'dave': 'david',
        'david': 'dave',
        'jim': 'james',
        'james': 'jim',
        'ed': 'edward',
        'edward': 'ed',
        'sam': 'samuel',
        'samuel': 'sam',
        'ben': 'benjamin',
        'benjamin': 'ben',
        'alex': 'alexander',
        'alexander': 'alex',
    }

    # Remove Jr., III, IV, II suffixes
    no_suffix = re.sub(r'\s+(Jr\.|Jr|III|IV|II)$', '', name, flags=re.IGNORECASE)
    if no_suffix != name:
        variations.append(no_suffix)

    # Handle initials with periods: "C.J." -> "CJ", "J.Michael" -> "J Michael"
    # Remove periods from initials
    no_periods = re.sub(r'\.', '', name)
    if no_periods not in variations:
        variations.append(no_periods)

    # Add space after initial periods: "J.Michael" -> "J. Michael"
    spaced = re.sub(r'\.(?=[A-Z])', '. ', name)
    if spaced not in variations:
        variations.append(spaced)

    # Combine: no suffix + no periods
    no_suffix_no_periods = re.sub(r'\.', '', no_suffix)
    if no_suffix_no_periods not in variations:
        variations.append(no_suffix_no_periods)

    # Try nickname variations
    parts = name.split()
    if len(parts) >= 2:
        first_name = parts[0].replace('.', '').lower()
        last_name = ' '.join(parts[1:])

        if first_name in NICKNAMES:
            alt_first = NICKNAMES[first_name].title()
            alt_name = f"{alt_first} {last_name}"
            if alt_name not in variations:
                variations.append(alt_name)
            # Also try with no suffix
            alt_no_suffix = re.sub(r'\s+(Jr\.|Jr|III|IV|II)$', '', alt_name, flags=re.IGNORECASE)
            if alt_no_suffix not in variations:
                variations.append(alt_no_suffix)

    # First name only (for uniquely named players)
    if len(parts) >= 2:
        first_name = parts[0].replace('.', '')
        if len(first_name) > 2:  # Skip if it's just initials
            variations.append(first_name)

    return variations


def fetch_team_receiving_stats(team, year=2025):
    """Fetch all receiving stats for a team in one API call."""
    url = f"{BASE_URL}/stats/player/season"
    params = {
        "year": year,
        "category": "receiving",
        "team": team
    }

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        stats = {}
        for stat in response.json():
            player = stat.get("player", "").lower()
            stat_type = stat.get("statType", "")
            value = stat.get("stat", 0)

            if player not in stats:
                stats[player] = {}
            try:
                stats[player][stat_type] = int(float(value))
            except (ValueError, TypeError):
                stats[player][stat_type] = 0

        return stats
    elif response.status_code == 429:
        print(f"    Rate limited on {team}")
        return None
    else:
        return {}


def names_match(name1, name2):
    """
    Check if two names refer to the same person.
    More strict than simple substring matching.
    """
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    # Exact match
    if n1 == n2:
        return True

    # Split into parts
    parts1 = n1.replace('.', ' ').split()
    parts2 = n2.replace('.', ' ').split()

    # Get first and last names (ignoring suffixes)
    suffixes = {'jr', 'jr.', 'ii', 'iii', 'iv'}
    clean1 = [p for p in parts1 if p not in suffixes]
    clean2 = [p for p in parts2 if p not in suffixes]

    if not clean1 or not clean2:
        return False

    # First name must match (or be initial)
    first1, first2 = clean1[0], clean2[0]
    if first1 != first2:
        # Allow initial match: "j" matches "john"
        if not (len(first1) <= 2 and first2.startswith(first1[0])):
            if not (len(first2) <= 2 and first1.startswith(first2[0])):
                return False

    # Last name must match exactly
    if len(clean1) > 1 and len(clean2) > 1:
        last1, last2 = clean1[-1], clean2[-1]
        if last1 != last2:
            return False

    return True


def find_player_in_team_stats(player_name, team_stats):
    """
    Search for a player in team stats using name variations.
    Returns (yards, matched_name) or (None, None) if not found.
    """
    if team_stats is None:
        return None, None

    # First try exact match on variations
    variations = get_name_variations(player_name)

    for variation in variations:
        var_lower = variation.lower()
        if var_lower in team_stats:
            yards = team_stats[var_lower].get("YDS")
            return yards, variation

    # Then try smart matching
    for api_name, stats in team_stats.items():
        if names_match(player_name, api_name):
            yards = stats.get("YDS")
            return yards, api_name

    return None, None


def main():
    input_path = "data/prospects_final.csv"
    output_path = "data/prospects_fixed_names.csv"

    print("=" * 60)
    print("STEP 2: TRYING ALTERNATE NAME FORMATS")
    print("=" * 60)
    print()

    # Read current data
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        prospects = list(reader)

    # Find players missing rec_yards
    missing = [p for p in prospects if not p.get('rec_yards') and p.get('school')]
    print(f"Found {len(missing)} players missing rec_yards (with known school)")
    print()

    # Group by school to minimize API calls
    schools = {}
    for p in missing:
        school = p['school']
        if school not in schools:
            schools[school] = []
        schools[school].append(p)

    print(f"Players spread across {len(schools)} schools")
    print()

    # Fetch stats by school and try to match
    found_count = 0
    school_cache = {}

    print("Fetching team stats and matching players...")
    print("-" * 60)

    for school, players in schools.items():
        # Fetch team stats (or use cache)
        if school not in school_cache:
            print(f"  Fetching stats for {school}...")
            team_stats = fetch_team_receiving_stats(school)
            school_cache[school] = team_stats

            if team_stats is None:
                print(f"    API rate limited - stopping")
                break
        else:
            team_stats = school_cache[school]

        # Try to match each player
        for player in players:
            name = player['player_name']
            yards, matched_name = find_player_in_team_stats(name, team_stats)

            if yards:
                print(f"    FOUND: {name} -> {matched_name} ({yards} yards)")
                # Update in prospects list
                for p in prospects:
                    if p['player_name'] == name:
                        p['rec_yards'] = yards
                        break
                found_count += 1

    print("-" * 60)
    print()
    print(f"RESULTS: Found stats for {found_count} additional players")
    print()

    # Count current stats
    with_yards = len([p for p in prospects if p.get('rec_yards')])
    without_yards = len([p for p in prospects if not p.get('rec_yards')])

    print(f"Current data coverage:")
    print(f"  With rec_yards: {with_yards}")
    print(f"  Without rec_yards: {without_yards}")
    print()

    # Save updated data
    print(f"Saving to: {output_path}")
    fieldnames = ['player_name', 'position', 'school', 'projected_pick',
                  'rec_yards', 'team_pass_attempts', 'birthdate', 'age',
                  'age_estimated', 'weight']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prospects)

    print("Done!")

    # Show which players still need data
    still_missing = [p for p in prospects if not p.get('rec_yards') and p.get('school')]
    if still_missing:
        print()
        print("=" * 60)
        print(f"PLAYERS STILL MISSING REC_YARDS ({len(still_missing)}):")
        print("=" * 60)
        for p in still_missing[:20]:
            print(f"  {p['player_name']:<25} {p['position']:<4} {p['school']}")
        if len(still_missing) > 20:
            print(f"  ... and {len(still_missing) - 20} more")


if __name__ == "__main__":
    main()
