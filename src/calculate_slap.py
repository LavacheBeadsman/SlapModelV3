"""
SLAP Score V3 Calculator

This is the main scoring script that combines all three components:
1. Draft Capital (45%) - Higher draft picks get higher scores
2. Breakout Score (35%) - Production relative to opportunity, adjusted for age
3. Athletic Modifier (20%) - Size + Speed (placeholder until Combine data available)

The output is a 0-100 score for each prospect, plus a "delta" showing how
much the model disagrees with their draft position.
"""

import csv
import math


# =============================================================================
# CONFIGURATION - These match the decisions in CLAUDE.md
# =============================================================================

# Component weights (must add to 1.0)
WEIGHT_DRAFT_CAPITAL = 0.45
WEIGHT_BREAKOUT = 0.35
WEIGHT_ATHLETIC = 0.20

# Age weight multipliers (moderate adjustment)
AGE_WEIGHTS = {
    18: 1.25,  # Extra young (rare)
    19: 1.20,
    20: 1.10,
    21: 1.00,  # Baseline
    22: 0.90,
    23: 0.80,
    24: 0.70,  # Older (rare)
    25: 0.60,
}


# =============================================================================
# COMPONENT CALCULATIONS
# =============================================================================

def calculate_draft_capital_raw(draft_pick):
    """
    Transform draft pick into a raw score.

    Formula: 1 / sqrt(draft_pick)

    Why this works:
    - Pick 1 → 1.000 (highest)
    - Pick 4 → 0.500
    - Pick 16 → 0.250
    - Pick 100 → 0.100
    - Pick 250 (UDFA) → 0.063 (lowest)

    The square root creates a curve where early picks are
    separated more than late picks (which makes sense - there's
    a bigger difference between pick 1 and 10 than between 200 and 210).
    """
    if draft_pick is None or draft_pick <= 0:
        return None
    return 1 / math.sqrt(draft_pick)


def calculate_breakout_raw(rec_yards, team_pass_attempts, age):
    """
    Calculate age-adjusted production score.

    Formula: (rec_yards / team_pass_attempts) * age_weight

    What this measures:
    - How much of their team's passing game did they capture?
    - Younger players get a bonus (more room to grow)
    - Older players get a penalty (this might be their ceiling)

    Returns None if any data is missing.
    """
    # Check for missing data
    if rec_yards is None or team_pass_attempts is None or age is None:
        return None
    if team_pass_attempts <= 0:
        return None

    # Calculate raw production rate
    production_rate = rec_yards / team_pass_attempts

    # Apply age adjustment
    age_weight = AGE_WEIGHTS.get(age, 0.70)  # Default to 0.70 for very old players

    return production_rate * age_weight


def calculate_athletic_raw(weight, forty_time):
    """
    Calculate athletic score using Barnwell Speed Score formula.

    Formula: (weight * 200) / (forty_time ^ 4)

    This rewards players who are fast FOR THEIR SIZE.
    A 220-lb player running 4.5 scores better than a 180-lb player running 4.5.

    NOTE: 40-yard dash times won't be available until the 2026 NFL Combine.
    For now, this will return None for most players.
    """
    if weight is None or forty_time is None:
        return None
    if forty_time <= 0:
        return None

    speed_score = (weight * 200) / (forty_time ** 4)
    return speed_score


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_scores(raw_scores, target_mean=50, target_std=15):
    """
    Convert raw scores to a 0-100 scale with mean of 50.

    How it works:
    1. Calculate the average and spread of raw scores
    2. Shift and scale so average becomes 50
    3. Cap at 0 and 100

    This makes scores interpretable:
    - 50 = average prospect
    - 65+ = well above average
    - 35- = well below average
    - 80+ = elite
    """
    # Filter out None values
    valid_scores = [s for s in raw_scores if s is not None]

    if len(valid_scores) == 0:
        return [None] * len(raw_scores)

    # Calculate mean and standard deviation
    mean = sum(valid_scores) / len(valid_scores)
    variance = sum((s - mean) ** 2 for s in valid_scores) / len(valid_scores)
    std = math.sqrt(variance) if variance > 0 else 1

    # Normalize each score
    normalized = []
    for score in raw_scores:
        if score is None:
            normalized.append(None)
        else:
            # Z-score transformation
            z = (score - mean) / std
            # Scale to target mean and std
            final = target_mean + (z * target_std)
            # Cap at 0-100
            final = max(0, min(100, final))
            normalized.append(round(final, 1))

    return normalized


# =============================================================================
# MAIN SLAP SCORE CALCULATION
# =============================================================================

def calculate_slap_scores(prospects):
    """
    Calculate SLAP scores for all prospects.

    Steps:
    1. Calculate raw component scores for each player
    2. Normalize each component to 0-100 scale
    3. Combine with weights: 45% Draft + 35% Breakout + 20% Athletic
    4. Calculate delta vs draft-only baseline

    Returns list of prospects with scores added.
    """

    # Step 1: Calculate raw scores for each component
    draft_raw = []
    breakout_raw = []
    athletic_raw = []

    for p in prospects:
        # Parse data (handle missing values)
        draft_pick = safe_int(p.get('projected_pick'))
        rec_yards = safe_int(p.get('rec_yards'))
        team_pass = safe_int(p.get('team_pass_attempts'))
        age = safe_int(p.get('age'))
        weight = safe_int(p.get('weight'))
        forty = safe_float(p.get('forty_time'))  # Not available yet

        draft_raw.append(calculate_draft_capital_raw(draft_pick))
        breakout_raw.append(calculate_breakout_raw(rec_yards, team_pass, age))
        athletic_raw.append(calculate_athletic_raw(weight, forty))

    # Step 2: Normalize each component
    draft_norm = normalize_scores(draft_raw)
    breakout_norm = normalize_scores(breakout_raw)
    athletic_norm = normalize_scores(athletic_raw)

    # Step 3: Calculate final SLAP score
    results = []

    for i, p in enumerate(prospects):
        result = dict(p)  # Copy original data

        # Store component scores
        result['draft_capital_score'] = draft_norm[i]
        result['breakout_score'] = breakout_norm[i]
        result['athletic_score'] = athletic_norm[i]

        # Calculate weighted SLAP score
        # Handle missing components gracefully
        dc = draft_norm[i]
        br = breakout_norm[i]
        ath = athletic_norm[i]

        if dc is None:
            # Can't calculate without draft capital
            result['slap_score'] = None
            result['delta'] = None
        elif br is None and ath is None:
            # Only have draft capital - use it alone
            result['slap_score'] = round(dc, 1)
            result['delta'] = 0.0  # No delta when using draft only
        elif ath is None:
            # Have draft + breakout but no athletic
            # Redistribute athletic weight: new weights = 0.45/0.80, 0.35/0.80
            adj_dc_weight = WEIGHT_DRAFT_CAPITAL / (WEIGHT_DRAFT_CAPITAL + WEIGHT_BREAKOUT)
            adj_br_weight = WEIGHT_BREAKOUT / (WEIGHT_DRAFT_CAPITAL + WEIGHT_BREAKOUT)
            slap = (dc * adj_dc_weight) + (br * adj_br_weight)
            result['slap_score'] = round(slap, 1)
            result['delta'] = round(slap - dc, 1)  # Difference from draft-only
        else:
            # Have all three components
            slap = (dc * WEIGHT_DRAFT_CAPITAL) + (br * WEIGHT_BREAKOUT) + (ath * WEIGHT_ATHLETIC)
            result['slap_score'] = round(slap, 1)
            result['delta'] = round(slap - dc, 1)

        results.append(result)

    return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_int(value):
    """Convert to int, returning None for missing/invalid values."""
    if value is None or value == '' or value == 'MISSING':
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def safe_float(value):
    """Convert to float, returning None for missing/invalid values."""
    if value is None or value == '' or value == 'MISSING':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":
    input_path = "data/prospects_final.csv"
    output_path = "output/slap_scores.csv"

    print("=" * 70)
    print("SLAP SCORE V3 CALCULATOR")
    print("=" * 70)
    print()

    # Load prospects
    print(f"Loading data from: {input_path}")
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        prospects = list(reader)
    print(f"  Found {len(prospects)} prospects")
    print()

    # Calculate scores
    print("Calculating SLAP scores...")
    results = calculate_slap_scores(prospects)

    # Count valid scores
    valid_slap = sum(1 for r in results if r['slap_score'] is not None)
    valid_breakout = sum(1 for r in results if r['breakout_score'] is not None)
    valid_athletic = sum(1 for r in results if r['athletic_score'] is not None)

    print(f"  Draft Capital scores: {len(results)} (all players have draft projections)")
    print(f"  Breakout scores: {valid_breakout} (need rec_yards, team_pass_attempts, age)")
    print(f"  Athletic scores: {valid_athletic} (need weight + 40 time)")
    print(f"  Final SLAP scores: {valid_slap}")
    print()

    # Sort by SLAP score (highest first)
    results_sorted = sorted(results, key=lambda x: (x['slap_score'] is None, -(x['slap_score'] or 0)))

    # Save to CSV
    print(f"Saving results to: {output_path}")

    # Ensure output directory exists
    import os
    os.makedirs("output", exist_ok=True)

    fieldnames = [
        'player_name', 'position', 'school', 'projected_pick',
        'slap_score', 'delta',
        'draft_capital_score', 'breakout_score', 'athletic_score',
        'rec_yards', 'team_pass_attempts', 'age', 'age_estimated', 'weight'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results_sorted)

    print("Done!")
    print()

    # Show top 20 prospects
    print("=" * 70)
    print("TOP 20 PROSPECTS BY SLAP SCORE")
    print("=" * 70)
    print()
    print(f"{'Rank':<5} {'Player':<25} {'Pos':<4} {'Pick':<5} {'SLAP':<6} {'Delta':<7} {'DC':<6} {'BR':<6}")
    print("-" * 70)

    for i, r in enumerate(results_sorted[:20], 1):
        name = r['player_name'][:24]
        pos = r['position']
        pick = r['projected_pick']
        slap = r['slap_score'] if r['slap_score'] else '-'
        delta = r['delta'] if r['delta'] else '-'
        dc = r['draft_capital_score'] if r['draft_capital_score'] else '-'
        br = r['breakout_score'] if r['breakout_score'] else '-'

        # Format delta with + sign for positive
        if isinstance(delta, (int, float)) and delta > 0:
            delta = f"+{delta}"

        print(f"{i:<5} {name:<25} {pos:<4} {pick:<5} {slap:<6} {delta:<7} {dc:<6} {br:<6}")

    print()

    # Show biggest positive deltas (model likes more than draft position)
    print("=" * 70)
    print("BIGGEST POSITIVE DELTAS (Model likes more than draft slot)")
    print("=" * 70)
    print()

    # Filter to players with valid deltas and sort by delta
    with_delta = [r for r in results if r['delta'] is not None and r['delta'] != 0]
    by_delta = sorted(with_delta, key=lambda x: -x['delta'])

    print(f"{'Player':<25} {'Pos':<4} {'Pick':<5} {'SLAP':<6} {'Delta':<7} {'Why?'}")
    print("-" * 70)

    for r in by_delta[:10]:
        name = r['player_name'][:24]
        pos = r['position']
        pick = r['projected_pick']
        slap = r['slap_score']
        delta = f"+{r['delta']}" if r['delta'] > 0 else r['delta']

        # Explain the delta
        dc = r['draft_capital_score'] or 50
        br = r['breakout_score'] or 50
        if br > dc:
            why = "Strong production"
        else:
            why = "Good all-around"

        print(f"{name:<25} {pos:<4} {pick:<5} {slap:<6} {delta:<7} {why}")

    print()

    # Show biggest negative deltas (model likes less than draft position)
    print("=" * 70)
    print("BIGGEST NEGATIVE DELTAS (Model likes less than draft slot)")
    print("=" * 70)
    print()

    by_delta_neg = sorted(with_delta, key=lambda x: x['delta'])

    print(f"{'Player':<25} {'Pos':<4} {'Pick':<5} {'SLAP':<6} {'Delta':<7} {'Why?'}")
    print("-" * 70)

    for r in by_delta_neg[:10]:
        name = r['player_name'][:24]
        pos = r['position']
        pick = r['projected_pick']
        slap = r['slap_score']
        delta = r['delta']

        # Explain the delta
        dc = r['draft_capital_score'] or 50
        br = r['breakout_score'] or 50
        if br < dc:
            why = "Low production"
        else:
            why = "Below expectations"

        print(f"{name:<25} {pos:<4} {pick:<5} {slap:<6} {delta:<7} {why}")

    print()
    print("=" * 70)
    print("NOTES")
    print("=" * 70)
    print("""
1. SLAP Score: 0-100 scale where 50 = average prospect
   - 65+ = Well above average
   - 80+ = Elite prospect

2. Delta: Difference between SLAP score and draft-only baseline
   - Positive = Model likes them MORE than their draft slot suggests
   - Negative = Model likes them LESS than their draft slot suggests

3. Missing Athletic Scores: 40-yard dash times aren't available until
   the 2026 NFL Combine. Once available, athletic scores will be added
   and final SLAP scores will be recalculated.

4. Players with missing breakout data have SLAP = Draft Capital only.
""")
