"""
SLAP Score V3 Calculator (Position-Split Version)

This scoring script calculates scores SEPARATELY for RBs and WRs:
- RBs are compared against other RBs (50 = average RB)
- WRs are compared against other WRs (50 = average WR)

Components:
1. Draft Capital (50%) - Higher draft picks get higher scores
2. Breakout Score (35%) - Production relative to opportunity, adjusted for age
   - RBs: rec_yards / team_pass_attempts * age_weight
   - WRs: Dominator Rating (rec_yards / team_rec_yards) * age_weight
3. Athletic Modifier (15%) - Barnwell Speed Score

The output is a 0-100 score for each prospect, plus a "delta" showing how
much the model disagrees with their draft position.
"""

import csv
import math
import os


# =============================================================================
# CONFIGURATION - These match the decisions in CLAUDE.md
# =============================================================================

# Component weights (must add to 1.0) - optimized via backtest on 2020-2024 classes
WEIGHT_DRAFT_CAPITAL = 0.50
WEIGHT_BREAKOUT = 0.35
WEIGHT_ATHLETIC = 0.15

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


def calculate_breakout_raw_rb(rec_yards, team_pass_attempts, age):
    """
    Calculate age-adjusted production score for RBs.

    Formula: (rec_yards / team_pass_attempts) * age_weight

    What this measures:
    - How much of their team's passing game did they capture?
    - Younger players get a bonus (more room to grow)
    - Older players get a penalty (this might be their ceiling)

    Returns None if any data is missing.
    """
    if rec_yards is None or team_pass_attempts is None or age is None:
        return None
    if team_pass_attempts <= 0:
        return None

    production_rate = rec_yards / team_pass_attempts
    age_weight = AGE_WEIGHTS.get(age, 0.70)

    return production_rate * age_weight


def calculate_breakout_raw_wr(rec_yards, team_rec_yards, team_pass_attempts, age):
    """
    Calculate age-adjusted production score for WRs using Dominator Rating.

    Formula: (rec_yards / team_rec_yards) * age_weight

    Dominator Rating measures what % of team receiving yards the player captured.
    This is more predictive than yards/PA for WRs because it accounts for team
    passing efficiency (completion rate, yards per attempt).

    Falls back to yards/PA if team_rec_yards is not available.
    """
    if rec_yards is None or age is None:
        return None

    age_weight = AGE_WEIGHTS.get(age, 0.70)

    # Prefer Dominator Rating if team receiving yards available
    if team_rec_yards is not None and team_rec_yards > 0:
        dominator = rec_yards / team_rec_yards
        return dominator * age_weight

    # Fall back to yards/PA
    if team_pass_attempts is not None and team_pass_attempts > 0:
        production_rate = rec_yards / team_pass_attempts
        return production_rate * age_weight

    return None


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
# POSITION-SPLIT SLAP SCORE CALCULATION
# =============================================================================

def calculate_slap_scores_by_position(prospects):
    """
    Calculate SLAP scores separately for RBs and WRs.

    Each position is normalized against itself:
    - RB with 50 breakout score = average RB production
    - WR with 50 breakout score = average WR production

    This prevents RBs from being penalized for having fewer receiving yards
    than WRs (which is expected).
    """

    # Separate by position
    rbs = [p for p in prospects if p.get('position') == 'RB']
    wrs = [p for p in prospects if p.get('position') == 'WR']

    print(f"  Splitting by position: {len(rbs)} RBs, {len(wrs)} WRs")

    # Calculate scores for each position group
    rb_results = calculate_position_group(rbs, "RB")
    wr_results = calculate_position_group(wrs, "WR")

    return rb_results, wr_results


def calculate_position_group(prospects, position_name):
    """
    Calculate SLAP scores for a single position group.
    All normalization happens within this group only.
    """
    if len(prospects) == 0:
        return []

    # Step 1: Calculate raw scores for each component
    draft_raw = []
    breakout_raw = []
    athletic_raw = []

    for p in prospects:
        # Parse data (handle missing values)
        draft_pick = safe_int(p.get('projected_pick'))
        rec_yards = safe_int(p.get('rec_yards'))
        team_pass = safe_int(p.get('team_pass_attempts'))
        team_rec = safe_int(p.get('team_rec_yards'))  # For WR Dominator Rating
        age = safe_int(p.get('age'))
        weight = safe_int(p.get('weight'))
        forty = safe_float(p.get('forty_time'))  # Not available yet

        draft_raw.append(calculate_draft_capital_raw(draft_pick))

        # Use position-specific breakout calculation
        if position_name == 'WR':
            breakout_raw.append(calculate_breakout_raw_wr(rec_yards, team_rec, team_pass, age))
        else:
            breakout_raw.append(calculate_breakout_raw_rb(rec_yards, team_pass, age))

        athletic_raw.append(calculate_athletic_raw(weight, forty))

    # Step 2: Normalize WITHIN this position group
    draft_norm = normalize_scores(draft_raw)
    breakout_norm = normalize_scores(breakout_raw)
    athletic_norm = normalize_scores(athletic_raw)

    # Count valid scores
    valid_breakout = sum(1 for b in breakout_norm if b is not None)
    valid_athletic = sum(1 for a in athletic_norm if a is not None)
    print(f"    {position_name}: {len(prospects)} total, {valid_breakout} with breakout, {valid_athletic} with athletic")

    # Step 3: Calculate final SLAP score
    results = []

    for i, p in enumerate(prospects):
        result = dict(p)  # Copy original data

        # Store component scores
        result['draft_capital_score'] = draft_norm[i]
        result['breakout_score'] = breakout_norm[i]
        result['athletic_score'] = athletic_norm[i]

        # Calculate weighted SLAP score
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
            # Redistribute athletic weight
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


def print_rankings(results, position, count=10):
    """Print top N prospects for a position."""
    # Sort by SLAP score (highest first)
    sorted_results = sorted(results, key=lambda x: (x['slap_score'] is None, -(x['slap_score'] or 0)))

    print(f"{'Rank':<5} {'Player':<25} {'School':<15} {'Pick':<5} {'SLAP':<6} {'Delta':<7} {'DC':<6} {'BR':<6}")
    print("-" * 85)

    for i, r in enumerate(sorted_results[:count], 1):
        name = r['player_name'][:24]
        school = (r.get('school') or '')[:14]
        pick = r['projected_pick']
        slap = r['slap_score'] if r['slap_score'] else '-'
        delta = r['delta'] if r['delta'] else '-'
        dc = r['draft_capital_score'] if r['draft_capital_score'] else '-'
        br = r['breakout_score'] if r['breakout_score'] else '-'

        # Format delta with + sign for positive
        if isinstance(delta, (int, float)) and delta > 0:
            delta = f"+{delta}"

        print(f"{i:<5} {name:<25} {school:<15} {pick:<5} {slap:<6} {delta:<7} {dc:<6} {br:<6}")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":
    input_path = "data/prospects_final.csv"
    output_rb_path = "output/slap_scores_rb.csv"
    output_wr_path = "output/slap_scores_wr.csv"
    output_combined_path = "output/slap_scores.csv"

    print("=" * 85)
    print("SLAP SCORE V3 CALCULATOR (POSITION-SPLIT)")
    print("=" * 85)
    print()

    # Load prospects
    print(f"Loading data from: {input_path}")
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        prospects = list(reader)
    print(f"  Found {len(prospects)} prospects")
    print()

    # Calculate scores by position
    print("Calculating SLAP scores by position...")
    rb_results, wr_results = calculate_slap_scores_by_position(prospects)
    print()

    # Sort each by SLAP score
    rb_sorted = sorted(rb_results, key=lambda x: (x['slap_score'] is None, -(x['slap_score'] or 0)))
    wr_sorted = sorted(wr_results, key=lambda x: (x['slap_score'] is None, -(x['slap_score'] or 0)))

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Save to separate CSV files
    fieldnames = [
        'player_name', 'position', 'school', 'projected_pick',
        'slap_score', 'delta',
        'draft_capital_score', 'breakout_score', 'athletic_score',
        'rec_yards', 'team_pass_attempts', 'age', 'age_estimated', 'weight'
    ]

    print(f"Saving RB results to: {output_rb_path}")
    with open(output_rb_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rb_sorted)

    print(f"Saving WR results to: {output_wr_path}")
    with open(output_wr_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(wr_sorted)

    # Also save combined (for backwards compatibility)
    print(f"Saving combined results to: {output_combined_path}")
    all_sorted = sorted(rb_results + wr_results, key=lambda x: (x['slap_score'] is None, -(x['slap_score'] or 0)))
    with open(output_combined_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_sorted)

    print()
    print("Done!")
    print()

    # Show top 10 RBs
    print("=" * 85)
    print("TOP 10 RUNNING BACKS BY SLAP SCORE")
    print("=" * 85)
    print()
    print("(RBs normalized against other RBs - 50 = average RB)")
    print()
    print_rankings(rb_results, "RB", 10)
    print()

    # Show top 10 WRs
    print("=" * 85)
    print("TOP 10 WIDE RECEIVERS BY SLAP SCORE")
    print("=" * 85)
    print()
    print("(WRs normalized against other WRs - 50 = average WR)")
    print()
    print_rankings(wr_results, "WR", 10)
    print()

    # Show biggest positive deltas by position
    print("=" * 85)
    print("BIGGEST POSITIVE DELTAS BY POSITION")
    print("(Model likes more than draft slot)")
    print("=" * 85)
    print()

    # RB positive deltas
    rb_with_delta = [r for r in rb_results if r['delta'] is not None and r['delta'] > 0]
    rb_by_delta = sorted(rb_with_delta, key=lambda x: -x['delta'])[:5]

    print("RBs:")
    print(f"  {'Player':<25} {'Pick':<6} {'SLAP':<6} {'Delta':<7}")
    print("  " + "-" * 50)
    for r in rb_by_delta:
        print(f"  {r['player_name']:<25} {r['projected_pick']:<6} {r['slap_score']:<6} +{r['delta']}")

    print()

    # WR positive deltas
    wr_with_delta = [r for r in wr_results if r['delta'] is not None and r['delta'] > 0]
    wr_by_delta = sorted(wr_with_delta, key=lambda x: -x['delta'])[:5]

    print("WRs:")
    print(f"  {'Player':<25} {'Pick':<6} {'SLAP':<6} {'Delta':<7}")
    print("  " + "-" * 50)
    for r in wr_by_delta:
        print(f"  {r['player_name']:<25} {r['projected_pick']:<6} {r['slap_score']:<6} +{r['delta']}")

    print()

    # Show biggest negative deltas by position
    print("=" * 85)
    print("BIGGEST NEGATIVE DELTAS BY POSITION")
    print("(Model likes less than draft slot)")
    print("=" * 85)
    print()

    # RB negative deltas
    rb_neg_delta = [r for r in rb_results if r['delta'] is not None and r['delta'] < 0]
    rb_by_neg = sorted(rb_neg_delta, key=lambda x: x['delta'])[:5]

    print("RBs:")
    print(f"  {'Player':<25} {'Pick':<6} {'SLAP':<6} {'Delta':<7}")
    print("  " + "-" * 50)
    for r in rb_by_neg:
        print(f"  {r['player_name']:<25} {r['projected_pick']:<6} {r['slap_score']:<6} {r['delta']}")

    print()

    # WR negative deltas
    wr_neg_delta = [r for r in wr_results if r['delta'] is not None and r['delta'] < 0]
    wr_by_neg = sorted(wr_neg_delta, key=lambda x: x['delta'])[:5]

    print("WRs:")
    print(f"  {'Player':<25} {'Pick':<6} {'SLAP':<6} {'Delta':<7}")
    print("  " + "-" * 50)
    for r in wr_by_neg:
        print(f"  {r['player_name']:<25} {r['projected_pick']:<6} {r['slap_score']:<6} {r['delta']}")

    print()
    print("=" * 85)
    print("NOTES")
    print("=" * 85)
    print("""
1. Position-Split Normalization:
   - RBs are compared only to other RBs (50 = average RB)
   - WRs are compared only to other WRs (50 = average WR)
   - This prevents RBs from being penalized for lower receiving yards

2. SLAP Score: 0-100 scale
   - 65+ = Well above average for position
   - 80+ = Elite prospect for position

3. Delta: Difference between SLAP score and draft-only baseline
   - Positive = Model likes them MORE than their draft slot suggests
   - Negative = Model likes them LESS than their draft slot suggests

4. Missing Athletic Scores: 40-yard dash times aren't available until
   the 2026 NFL Combine. Once available, athletic scores will be added.
""")
