"""
Recalculate SLAP scores for backtest data using percentile ranks.

After fixing data errors (DeVonta Smith weight=170, forty=4.48; Dyami Brown weight=189, forty=4.46),
this script recalculates athletic scores and final SLAP percentile ranks.
"""

import math
import pandas as pd

# SLAP configuration (optimized via backtest on 2020-2024 classes)
WEIGHT_DRAFT_CAPITAL = 0.50
WEIGHT_BREAKOUT = 0.35
WEIGHT_ATHLETIC = 0.15

AGE_WEIGHTS = {
    18: 1.25, 19: 1.20, 20: 1.10, 21: 1.00,
    22: 0.90, 23: 0.80, 24: 0.70, 25: 0.60, 26: 0.50,
}


def calculate_draft_capital_raw(pick):
    if pd.isna(pick) or pick <= 0:
        return None
    return 1 / math.sqrt(pick)


def calculate_breakout_raw(rec_yards, team_pass_att, age):
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or pd.isna(age):
        return None
    if team_pass_att <= 0:
        return None
    production_rate = rec_yards / team_pass_att
    age_weight = AGE_WEIGHTS.get(int(age), 0.50)
    return production_rate * age_weight


def calculate_athletic_raw(weight, forty):
    """Calculate Barnwell Speed Score."""
    if pd.isna(weight) or pd.isna(forty):
        return None
    if forty <= 0:
        return None
    return (weight * 200) / (forty ** 4)


def normalize_to_50_scale(values, target_mean=50, target_std=15):
    """Normalize scores to 0-100 scale with mean 50."""
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return [None] * len(values)

    mean = sum(valid) / len(valid)
    variance = sum((v - mean) ** 2 for v in valid) / len(valid)
    std = math.sqrt(variance) if variance > 0 else 1

    normalized = []
    for v in values:
        if v is None:
            normalized.append(None)
        else:
            z = (v - mean) / std
            final = target_mean + (z * target_std)
            final = max(0, min(100, final))
            normalized.append(round(final, 1))

    return normalized


def main():
    print("=" * 70)
    print("RECALCULATING BACKTEST SLAP SCORES")
    print("=" * 70)
    print()

    # Load data
    bt = pd.read_csv("data/backtest_college_stats.csv")
    print(f"Loaded {len(bt)} players")

    # Convert numeric columns
    bt['pick'] = pd.to_numeric(bt['pick'], errors='coerce')
    bt['age'] = pd.to_numeric(bt['age'], errors='coerce')
    bt['rec_yards'] = pd.to_numeric(bt['rec_yards'], errors='coerce')
    bt['team_pass_attempts'] = pd.to_numeric(bt['team_pass_attempts'], errors='coerce')
    bt['weight'] = pd.to_numeric(bt['weight'], errors='coerce')
    bt['forty'] = pd.to_numeric(bt['forty'], errors='coerce')

    # Split by position
    rbs = bt[bt['position'] == 'RB'].copy()
    wrs = bt[bt['position'] == 'WR'].copy()

    print(f"  RBs: {len(rbs)}, WRs: {len(wrs)}")
    print()

    # Process each position group
    for pos_name, df in [('RB', rbs), ('WR', wrs)]:
        print(f"Processing {pos_name}s...")

        # Calculate raw scores
        draft_raw = [calculate_draft_capital_raw(p) for p in df['pick']]
        breakout_raw = []
        athletic_raw = []

        for _, row in df.iterrows():
            # Breakout
            br = calculate_breakout_raw(row['rec_yards'], row['team_pass_attempts'], row['age'])
            breakout_raw.append(br)

            # Athletic
            ath = calculate_athletic_raw(row['weight'], row['forty'])
            athletic_raw.append(ath)

        # Normalize within position group
        draft_norm = normalize_to_50_scale(draft_raw)
        breakout_norm = normalize_to_50_scale(breakout_raw)
        athletic_norm = normalize_to_50_scale(athletic_raw)

        # Store normalized scores
        df['draft_capital_score'] = draft_norm
        df['breakout_score'] = breakout_norm
        df['athletic_score'] = athletic_norm

        # Handle position converts (set breakout to 50.0)
        if 'position_convert' in df.columns:
            convert_mask = df['position_convert'] == True
            df.loc[convert_mask, 'breakout_score'] = 50.0

        # Calculate weighted SLAP (pre-percentile)
        slap_raw = []
        for i, (_, row) in enumerate(df.iterrows()):
            dc = draft_norm[i]
            br = df.iloc[i]['breakout_score']  # Use potentially modified breakout
            ath = athletic_norm[i]

            if dc is None:
                slap_raw.append(None)
            elif br is None and ath is None:
                slap_raw.append(dc)
            elif ath is None:
                # DC + BR only (56/44 split)
                adj_dc = WEIGHT_DRAFT_CAPITAL / (WEIGHT_DRAFT_CAPITAL + WEIGHT_BREAKOUT)
                adj_br = WEIGHT_BREAKOUT / (WEIGHT_DRAFT_CAPITAL + WEIGHT_BREAKOUT)
                slap_raw.append((dc * adj_dc) + (br * adj_br))
            else:
                # All three components
                slap_raw.append((dc * WEIGHT_DRAFT_CAPITAL) + (br * WEIGHT_BREAKOUT) + (ath * WEIGHT_ATHLETIC))

        df['slap_raw'] = slap_raw

        # Count valid
        valid_br = sum(1 for b in breakout_norm if b is not None)
        valid_ath = sum(1 for a in athletic_norm if a is not None)
        print(f"  {pos_name}: {valid_br} with breakout, {valid_ath} with athletic")

    # Combine back
    combined = pd.concat([rbs, wrs], ignore_index=True)

    # Convert SLAP to percentile ranks (across ALL players)
    print()
    print("Converting to percentile ranks...")

    valid_slap = combined['slap_raw'].dropna()
    print(f"  {len(valid_slap)} players with valid SLAP scores")

    def to_percentile(score):
        if pd.isna(score):
            return None
        # Count how many players have lower scores
        rank = (valid_slap < score).sum() + 1
        percentile = (rank / len(valid_slap)) * 100
        return round(percentile, 1)

    combined['slap_score'] = combined['slap_raw'].apply(to_percentile)

    # Drop temp column
    combined = combined.drop(columns=['slap_raw'])

    # Sort by draft year and pick
    combined = combined.sort_values(['draft_year', 'pick']).reset_index(drop=True)

    # Save
    combined.to_csv("data/backtest_college_stats.csv", index=False)
    print()
    print("Saved updated data")

    # Show summary
    print()
    print("=" * 70)
    print("SLAP SCORE DISTRIBUTION")
    print("=" * 70)

    valid_scores = combined['slap_score'].dropna()

    elite = (valid_scores >= 90).sum()
    good = ((valid_scores >= 70) & (valid_scores < 90)).sum()
    avg = ((valid_scores >= 50) & (valid_scores < 70)).sum()
    below = (valid_scores < 50).sum()

    print(f"Elite (90+):      {elite:>3} ({elite/len(valid_scores)*100:.1f}%)")
    print(f"Good (70-89):     {good:>3} ({good/len(valid_scores)*100:.1f}%)")
    print(f"Average (50-69):  {avg:>3} ({avg/len(valid_scores)*100:.1f}%)")
    print(f"Below Avg (<50):  {below:>3} ({below/len(valid_scores)*100:.1f}%)")

    # Show fixed players
    print()
    print("=" * 70)
    print("FIXED PLAYERS - NEW SCORES")
    print("=" * 70)

    for name in ['DeVonta Smith', 'Dyami Brown']:
        row = combined[combined['player_name'] == name]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"\n{name}:")
            print(f"  Weight: {r['weight']}, Forty: {r['forty']}")
            print(f"  DC: {r['draft_capital_score']}, BR: {r['breakout_score']}, ATH: {r['athletic_score']}")
            print(f"  SLAP: {r['slap_score']} (percentile rank)")


if __name__ == "__main__":
    main()
