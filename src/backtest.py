"""
SLAP Score V3 - Backtest Framework

This script validates the SLAP model by:
1. Taking a historical draft class (e.g., 2022)
2. Calculating what their SLAP scores would have been using college stats
3. Comparing to their actual NFL rookie fantasy production
4. Reporting correlation and predictive accuracy

This tells us: Does SLAP actually predict NFL success?
"""

import os
import csv
import math
import requests
import pandas as pd
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# SLAP configuration (must match calculate_slap.py)
WEIGHT_DRAFT_CAPITAL = 0.45
WEIGHT_BREAKOUT = 0.35
WEIGHT_ATHLETIC = 0.20

AGE_WEIGHTS = {
    18: 1.25, 19: 1.20, 20: 1.10, 21: 1.00,
    22: 0.90, 23: 0.80, 24: 0.70, 25: 0.60,
}


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_college_receiving_yards(player_name, team, year):
    """Get receiving yards from CFBD API."""
    url = f"{BASE_URL}/stats/player/season"
    params = {"year": year, "category": "receiving", "team": team}

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        for stat in response.json():
            if stat.get("player", "").lower() == player_name.lower():
                if stat.get("statType") == "YDS":
                    return int(stat.get("stat", 0))
    return None


def get_team_pass_attempts(team, year):
    """Get team pass attempts from CFBD API."""
    url = f"{BASE_URL}/stats/season"
    params = {"year": year, "team": team}

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        for stat in response.json():
            if stat.get("statName") == "passAttempts":
                return int(stat.get("statValue", 0))
    return None


# Cache for team pass attempts
_team_cache = {}

def get_team_pass_attempts_cached(team, year):
    key = f"{team}_{year}"
    if key not in _team_cache:
        _team_cache[key] = get_team_pass_attempts(team, year)
    return _team_cache[key]


# =============================================================================
# SLAP CALCULATION (simplified version for backtest)
# =============================================================================

def calculate_draft_capital_raw(pick):
    if pick is None or pick <= 0:
        return None
    return 1 / math.sqrt(pick)


def calculate_breakout_raw(rec_yards, team_pass_att, age):
    if rec_yards is None or team_pass_att is None or age is None:
        return None
    if team_pass_att <= 0:
        return None

    production_rate = rec_yards / team_pass_att
    age_weight = AGE_WEIGHTS.get(age, 0.70)
    return production_rate * age_weight


def normalize_scores(raw_scores, target_mean=50, target_std=15):
    valid = [s for s in raw_scores if s is not None]
    if len(valid) == 0:
        return [None] * len(raw_scores)

    mean = sum(valid) / len(valid)
    variance = sum((s - mean) ** 2 for s in valid) / len(valid)
    std = math.sqrt(variance) if variance > 0 else 1

    normalized = []
    for score in raw_scores:
        if score is None:
            normalized.append(None)
        else:
            z = (score - mean) / std
            final = target_mean + (z * target_std)
            final = max(0, min(100, final))
            normalized.append(round(final, 1))

    return normalized


def calculate_slap_for_group(players):
    """Calculate SLAP scores for a group of players (same position)."""

    # Calculate raw scores
    draft_raw = [calculate_draft_capital_raw(p['pick']) for p in players]
    breakout_raw = [calculate_breakout_raw(p['rec_yards'], p['team_pass_att'], p['age']) for p in players]

    # Normalize within group
    draft_norm = normalize_scores(draft_raw)
    breakout_norm = normalize_scores(breakout_raw)

    # Calculate final SLAP
    for i, p in enumerate(players):
        p['draft_capital_score'] = draft_norm[i]
        p['breakout_score'] = breakout_norm[i]

        dc = draft_norm[i]
        br = breakout_norm[i]

        if dc is None:
            p['slap_score'] = None
        elif br is None:
            p['slap_score'] = dc  # Draft only
        else:
            # Redistribute weights (no athletic data)
            adj_dc = WEIGHT_DRAFT_CAPITAL / (WEIGHT_DRAFT_CAPITAL + WEIGHT_BREAKOUT)
            adj_br = WEIGHT_BREAKOUT / (WEIGHT_DRAFT_CAPITAL + WEIGHT_BREAKOUT)
            p['slap_score'] = round((dc * adj_dc) + (br * adj_br), 1)

    return players


# =============================================================================
# BACKTEST MAIN FUNCTION
# =============================================================================

def run_backtest(draft_year, college_year):
    """
    Run a full backtest for a draft class.

    Args:
        draft_year: NFL draft year (e.g., 2022)
        college_year: Final college season year (e.g., 2021)
    """

    print("=" * 80)
    print(f"SLAP SCORE BACKTEST: {draft_year} DRAFT CLASS")
    print("=" * 80)
    print()

    # Load draft picks
    print("Loading draft picks...")
    drafts = pd.read_parquet('data/nflverse/draft_picks.parquet')
    rookies = drafts[(drafts['season'] == draft_year) & (drafts['position'].isin(['RB', 'WR']))]
    print(f"  Found {len(rookies)} RB/WR drafted in {draft_year}")

    # Load NFL fantasy stats
    print("Loading NFL fantasy stats...")
    nfl_stats = pd.read_parquet(f'data/nflverse/player_stats_{draft_year}.parquet')
    nfl_stats = nfl_stats[nfl_stats['season_type'] == 'REG']

    # Aggregate to season totals
    season_totals = nfl_stats.groupby(['player_id', 'player_display_name']).agg({
        'fantasy_points_ppr': 'sum',
        'week': 'count'
    }).reset_index()
    season_totals.rename(columns={'week': 'games'}, inplace=True)

    # Build player list with college stats
    print(f"Fetching {college_year} college stats from CFBD API...")
    players = []

    for _, row in rookies.iterrows():
        name = row['pfr_player_name']
        college = row['college']
        pick = int(row['pick'])
        position = row['position']
        age = int(row['age']) if pd.notna(row['age']) else None
        gsis_id = row['gsis_id']

        # Get college stats
        rec_yards = get_college_receiving_yards(name, college, college_year)
        team_pass_att = get_team_pass_attempts_cached(college, college_year) if college else None

        # Get NFL fantasy points
        nfl_match = season_totals[season_totals['player_id'] == gsis_id]
        if len(nfl_match) > 0:
            nfl_ppr = nfl_match.iloc[0]['fantasy_points_ppr']
            nfl_games = nfl_match.iloc[0]['games']
        else:
            nfl_ppr = None
            nfl_games = 0

        players.append({
            'name': name,
            'position': position,
            'college': college,
            'pick': pick,
            'age': age,
            'rec_yards': rec_yards,
            'team_pass_att': team_pass_att,
            'nfl_ppr': nfl_ppr,
            'nfl_games': nfl_games,
        })

    print(f"  Fetched stats for {len(players)} players")
    print()

    # Split by position and calculate SLAP
    rbs = [p for p in players if p['position'] == 'RB']
    wrs = [p for p in players if p['position'] == 'WR']

    print("Calculating SLAP scores (position-split normalization)...")
    rbs = calculate_slap_for_group(rbs)
    wrs = calculate_slap_for_group(wrs)

    all_players = rbs + wrs

    # Filter to players with both SLAP and NFL stats
    valid = [p for p in all_players if p['slap_score'] is not None and p['nfl_ppr'] is not None]
    print(f"  {len(valid)} players have both SLAP scores and NFL stats")
    print()

    # Calculate correlation
    if len(valid) >= 5:
        slap_scores = [p['slap_score'] for p in valid]
        nfl_points = [p['nfl_ppr'] for p in valid]

        # Pearson correlation
        n = len(valid)
        sum_x = sum(slap_scores)
        sum_y = sum(nfl_points)
        sum_xy = sum(s * n for s, n in zip(slap_scores, nfl_points))
        sum_x2 = sum(s ** 2 for s in slap_scores)
        sum_y2 = sum(n ** 2 for n in nfl_points)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        correlation = numerator / denominator if denominator != 0 else 0

        print("=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print()
        print(f"Correlation (SLAP vs NFL PPR): {correlation:.3f}")
        print()

        if correlation > 0.5:
            print("  Interpretation: STRONG positive correlation")
        elif correlation > 0.3:
            print("  Interpretation: MODERATE positive correlation")
        elif correlation > 0.1:
            print("  Interpretation: WEAK positive correlation")
        else:
            print("  Interpretation: NO meaningful correlation")
        print()

    # Show results table
    print("=" * 80)
    print("DETAILED RESULTS (sorted by SLAP score)")
    print("=" * 80)
    print()

    # Sort by SLAP
    valid_sorted = sorted(valid, key=lambda x: -x['slap_score'])

    print(f"{'Rank':<5} {'Player':<25} {'Pos':<4} {'Pick':<5} {'SLAP':<6} {'NFL PPR':<10} {'Games':<6}")
    print("-" * 80)

    for i, p in enumerate(valid_sorted, 1):
        name = p['name'][:24]
        print(f"{i:<5} {name:<25} {p['position']:<4} {p['pick']:<5} {p['slap_score']:<6} {p['nfl_ppr']:<10.1f} {p['nfl_games']:<6}")

    print()

    # Show biggest hits and misses
    print("=" * 80)
    print("BIGGEST HITS (High SLAP, High NFL)")
    print("=" * 80)

    # Players where both SLAP and NFL are above average
    avg_slap = sum(p['slap_score'] for p in valid) / len(valid)
    avg_nfl = sum(p['nfl_ppr'] for p in valid) / len(valid)

    hits = [p for p in valid if p['slap_score'] > avg_slap and p['nfl_ppr'] > avg_nfl]
    hits = sorted(hits, key=lambda x: -x['nfl_ppr'])[:5]

    for p in hits:
        print(f"  {p['name']:<25} SLAP: {p['slap_score']:<6} NFL: {p['nfl_ppr']:.1f}")

    print()
    print("=" * 80)
    print("BIGGEST MISSES (High SLAP, Low NFL)")
    print("=" * 80)

    misses = [p for p in valid if p['slap_score'] > avg_slap and p['nfl_ppr'] < avg_nfl]
    misses = sorted(misses, key=lambda x: x['nfl_ppr'])[:5]

    for p in misses:
        print(f"  {p['name']:<25} SLAP: {p['slap_score']:<6} NFL: {p['nfl_ppr']:.1f}")

    print()
    print("=" * 80)
    print("SURPRISES (Low SLAP, High NFL)")
    print("=" * 80)

    surprises = [p for p in valid if p['slap_score'] < avg_slap and p['nfl_ppr'] > avg_nfl]
    surprises = sorted(surprises, key=lambda x: -x['nfl_ppr'])[:5]

    for p in surprises:
        print(f"  {p['name']:<25} SLAP: {p['slap_score']:<6} NFL: {p['nfl_ppr']:.1f}")

    # Save results to CSV
    output_path = f"output/backtest_{draft_year}.csv"
    print()
    print(f"Saving results to: {output_path}")

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['name', 'position', 'college', 'pick', 'age',
                      'rec_yards', 'team_pass_att', 'slap_score',
                      'draft_capital_score', 'breakout_score',
                      'nfl_ppr', 'nfl_games']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_players)

    return correlation, all_players


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run backtest for 2022 draft class
    correlation, players = run_backtest(draft_year=2022, college_year=2021)
