"""
Update WR breakout scores using Dominator Rating instead of yards/team_pass_attempts.

This fixes the negative breakout coefficient issue discovered in logistic regression.
"""

import math
import pandas as pd
from scipy import stats

AGE_WEIGHTS = {
    18: 1.25, 19: 1.20, 20: 1.10, 21: 1.00,
    22: 0.90, 23: 0.80, 24: 0.70, 25: 0.60, 26: 0.50,
}

WEIGHT_DRAFT_CAPITAL = 0.50
WEIGHT_BREAKOUT = 0.35
WEIGHT_ATHLETIC = 0.15


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
    print("UPDATING WR BREAKOUT SCORES WITH DOMINATOR RATING")
    print("=" * 70)
    print()

    # Load data
    bt = pd.read_csv("data/backtest_college_stats.csv")
    dom = pd.read_csv("data/wr_dominator_ratings.csv")
    hit_rates = pd.read_csv("data/backtest_hit_rates.csv")

    print(f"Backtest players: {len(bt)}")
    print(f"Dominator ratings: {len(dom)}")
    print(f"Hit rate data: {len(hit_rates)}")

    # Merge dominator into backtest (for WRs only, 2020-2023)
    dom_lookup = dict(zip(zip(dom['player_name'], dom['draft_year']), dom['dominator_rating']))

    # Calculate new breakout for WRs using Dominator Rating
    wrs = bt[(bt['position'] == 'WR') & (bt['draft_year'] <= 2023)].copy()
    print(f"WRs for breakout update: {len(wrs)}")

    # Calculate age-adjusted dominator for each WR
    breakout_raw = []
    for idx, row in wrs.iterrows():
        key = (row['player_name'], row['draft_year'])
        dominator = dom_lookup.get(key)
        age = row['age']

        if pd.notna(dominator) and pd.notna(age):
            age_weight = AGE_WEIGHTS.get(int(age), 0.50)
            raw = dominator * age_weight
            breakout_raw.append(raw)
        else:
            breakout_raw.append(None)

    # Normalize within WRs
    breakout_norm = normalize_to_50_scale(breakout_raw)
    wrs['breakout_score_new'] = breakout_norm

    # Now recalculate SLAP for WRs
    slap_raw = []
    for i, (idx, row) in enumerate(wrs.iterrows()):
        dc = row['draft_capital_score']
        br = breakout_norm[i]
        ath = row['athletic_score']

        if pd.isna(dc):
            slap_raw.append(None)
        elif br is None and pd.isna(ath):
            slap_raw.append(dc)
        elif br is None:
            # DC + ATH only (no breakout)
            adj_dc = WEIGHT_DRAFT_CAPITAL / (WEIGHT_DRAFT_CAPITAL + WEIGHT_ATHLETIC)
            adj_ath = WEIGHT_ATHLETIC / (WEIGHT_DRAFT_CAPITAL + WEIGHT_ATHLETIC)
            slap_raw.append((dc * adj_dc) + (ath * adj_ath))
        elif pd.isna(ath):
            # DC + BR only
            adj_dc = WEIGHT_DRAFT_CAPITAL / (WEIGHT_DRAFT_CAPITAL + WEIGHT_BREAKOUT)
            adj_br = WEIGHT_BREAKOUT / (WEIGHT_DRAFT_CAPITAL + WEIGHT_BREAKOUT)
            slap_raw.append((dc * adj_dc) + (br * adj_br))
        else:
            slap_raw.append((dc * WEIGHT_DRAFT_CAPITAL) + (br * WEIGHT_BREAKOUT) + (ath * WEIGHT_ATHLETIC))

    wrs['slap_raw_new'] = slap_raw

    # Convert to percentile ranks (within WRs with valid scores)
    valid_slap = [s for s in slap_raw if s is not None]

    def to_percentile(score):
        if score is None:
            return None
        rank = sum(1 for s in valid_slap if s < score) + 1
        return round((rank / len(valid_slap)) * 100, 1)

    wrs['slap_score_new'] = [to_percentile(s) for s in slap_raw]

    # Merge with hit rates for analysis
    wrs_2023 = wrs[wrs['draft_year'] <= 2023].copy()
    merged = wrs_2023.merge(hit_rates, on=['player_name', 'draft_year'], how='inner')
    print(f"WRs with hit data: {len(merged)}")

    # Show correlation comparison
    print()
    print("=" * 70)
    print("CORRELATION ANALYSIS: OLD vs NEW BREAKOUT")
    print("=" * 70)

    # Old breakout vs Hit24
    old_valid = merged.dropna(subset=['breakout_score', 'hit24'])
    r_old, p_old = stats.pointbiserialr(old_valid['hit24'], old_valid['breakout_score'])
    print(f"\nOLD Breakout (yards/PA) vs Hit24:")
    print(f"  Correlation: r={r_old:.3f}, p={p_old:.3f}")
    print(f"  Hits mean: {old_valid[old_valid['hit24']==1]['breakout_score'].mean():.1f}")
    print(f"  Misses mean: {old_valid[old_valid['hit24']==0]['breakout_score'].mean():.1f}")

    # New breakout vs Hit24
    new_valid = merged.dropna(subset=['breakout_score_new', 'hit24'])
    r_new, p_new = stats.pointbiserialr(new_valid['hit24'], new_valid['breakout_score_new'])
    print(f"\nNEW Breakout (Dominator × age) vs Hit24:")
    print(f"  Correlation: r={r_new:.3f}, p={p_new:.3f}")
    print(f"  Hits mean: {new_valid[new_valid['hit24']==1]['breakout_score_new'].mean():.1f}")
    print(f"  Misses mean: {new_valid[new_valid['hit24']==0]['breakout_score_new'].mean():.1f}")

    # Logistic regression comparison
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print()
    print("=" * 70)
    print("LOGISTIC REGRESSION: OLD vs NEW")
    print("=" * 70)

    features = ['draft_capital_score', 'breakout_score', 'athletic_score']
    features_new = ['draft_capital_score', 'breakout_score_new', 'athletic_score']

    # Old model
    old_data = merged.dropna(subset=features + ['hit24'])
    X_old = old_data[features]
    y_old = old_data['hit24']

    scaler_old = StandardScaler()
    X_old_scaled = scaler_old.fit_transform(X_old)

    model_old = LogisticRegression(random_state=42, max_iter=1000)
    model_old.fit(X_old_scaled, y_old)

    print("\nOLD MODEL (yards/PA):")
    for feat, coef in zip(features, model_old.coef_[0]):
        print(f"  {feat:25s}: {coef:+.3f}")

    # New model
    new_data = merged.dropna(subset=features_new + ['hit24'])
    X_new = new_data[features_new]
    y_new = new_data['hit24']

    scaler_new = StandardScaler()
    X_new_scaled = scaler_new.fit_transform(X_new)

    model_new = LogisticRegression(random_state=42, max_iter=1000)
    model_new.fit(X_new_scaled, y_new)

    print("\nNEW MODEL (Dominator):")
    for feat, coef in zip(features_new, model_new.coef_[0]):
        display_name = feat.replace('_new', '')
        print(f"  {display_name:25s}: {coef:+.3f}")

    # Verify breakout is now positive
    br_coef = model_new.coef_[0][1]  # breakout is second feature
    print()
    if br_coef > 0:
        print("SUCCESS: Breakout coefficient is now POSITIVE!")
    else:
        print("WARNING: Breakout coefficient is still negative")

    # Show top players by new breakout
    print()
    print("=" * 70)
    print("TOP 10 WRS BY NEW BREAKOUT SCORE")
    print("=" * 70)

    top = wrs.nlargest(10, 'breakout_score_new')
    for _, r in top.iterrows():
        dom_val = dom_lookup.get((r['player_name'], r['draft_year']), None)
        print(f"  {r['player_name']:25s}: {r['breakout_score_new']:.1f} (Dominator: {dom_val:.1f}%)")

    # Save updated data
    print()
    print("Saving new breakout scores...")

    # Update the main backtest file
    for idx, row in wrs.iterrows():
        bt.loc[idx, 'breakout_score'] = row['breakout_score_new']
        if pd.notna(row['slap_score_new']):
            bt.loc[idx, 'slap_score'] = row['slap_score_new']

    bt.to_csv("data/backtest_college_stats_dominator.csv", index=False)
    print("Saved to data/backtest_college_stats_dominator.csv")


if __name__ == "__main__":
    main()
