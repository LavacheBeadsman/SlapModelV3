"""
Update WR breakout scores using correct Dominator Rating formula:
  Dominator = (yards_share + tds_share) / 2

Then verify logistic regression coefficients.
"""

import math
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

AGE_WEIGHTS = {
    18: 1.25, 19: 1.20, 20: 1.10, 21: 1.00,
    22: 0.90, 23: 0.80, 24: 0.70, 25: 0.60, 26: 0.50,
}

WEIGHT_DRAFT_CAPITAL = 0.50
WEIGHT_BREAKOUT = 0.35
WEIGHT_ATHLETIC = 0.15


def normalize_to_50_scale(values, target_mean=50, target_std=15):
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
    print("UPDATING WR BREAKOUT WITH CORRECT DOMINATOR RATING")
    print("(yards% + TDs%) / 2 × age_weight")
    print("=" * 70)
    print()

    # Load data
    bt = pd.read_csv("data/backtest_college_stats.csv")
    dom = pd.read_csv("data/wr_dominator_full.csv")
    hit_rates = pd.read_csv("data/backtest_hit_rates.csv")

    # Create dominator lookup
    dom_lookup = dict(zip(
        zip(dom['player_name'], dom['draft_year']),
        dom['dominator_rating']
    ))

    # Filter to WRs with hit rate data
    wrs = bt[(bt['position'] == 'WR') & (bt['draft_year'] <= 2023)].copy()
    print(f"WRs: {len(wrs)}")

    # Calculate age-adjusted dominator for each WR
    breakout_raw = []
    for _, row in wrs.iterrows():
        key = (row['player_name'], row['draft_year'])
        dominator = dom_lookup.get(key)  # Already as percentage
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

    # Merge with hit rates
    merged = wrs.merge(hit_rates, on=['player_name', 'draft_year'], how='inner')
    print(f"WRs with hit data: {len(merged)}")

    # Correlation analysis
    print()
    print("=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Old breakout vs Hit24
    old_valid = merged.dropna(subset=['breakout_score', 'hit24'])
    r_old, p_old = stats.pointbiserialr(old_valid['hit24'], old_valid['breakout_score'])
    print(f"\nOLD Breakout (yards/PA) vs Hit24:")
    print(f"  Correlation: r={r_old:.3f}, p={p_old:.3f}")

    # New breakout vs Hit24
    new_valid = merged.dropna(subset=['breakout_score_new', 'hit24'])
    r_new, p_new = stats.pointbiserialr(new_valid['hit24'], new_valid['breakout_score_new'])
    print(f"\nNEW Breakout (Dominator) vs Hit24:")
    print(f"  Correlation: r={r_new:.3f}, p={p_new:.3f}")
    print(f"  Hits mean:   {new_valid[new_valid['hit24']==1]['breakout_score_new'].mean():.1f}")
    print(f"  Misses mean: {new_valid[new_valid['hit24']==0]['breakout_score_new'].mean():.1f}")

    # Logistic regression
    print()
    print("=" * 70)
    print("LOGISTIC REGRESSION COEFFICIENTS")
    print("=" * 70)

    features_old = ['draft_capital_score', 'breakout_score', 'athletic_score']
    features_new = ['draft_capital_score', 'breakout_score_new', 'athletic_score']

    # Old model
    old_data = merged.dropna(subset=features_old + ['hit24'])
    X_old = old_data[features_old]
    y_old = old_data['hit24']
    scaler_old = StandardScaler()
    X_old_scaled = scaler_old.fit_transform(X_old)
    model_old = LogisticRegression(random_state=42, max_iter=1000)
    model_old.fit(X_old_scaled, y_old)

    print("\nOLD MODEL (yards/PA breakout):")
    for feat, coef in zip(features_old, model_old.coef_[0]):
        sign = "+" if coef > 0 else ""
        print(f"  {feat:25s}: {sign}{coef:.3f}")

    # New model
    new_data = merged.dropna(subset=features_new + ['hit24'])
    X_new = new_data[features_new]
    y_new = new_data['hit24']
    scaler_new = StandardScaler()
    X_new_scaled = scaler_new.fit_transform(X_new)
    model_new = LogisticRegression(random_state=42, max_iter=1000)
    model_new.fit(X_new_scaled, y_new)

    print("\nNEW MODEL (Dominator breakout):")
    for feat, coef in zip(features_new, model_new.coef_[0]):
        display = feat.replace('_new', '')
        sign = "+" if coef > 0 else ""
        print(f"  {display:25s}: {sign}{coef:.3f}")

    # Verdict
    br_coef = model_new.coef_[0][1]
    print()
    if br_coef > 0:
        print("✓ SUCCESS: Breakout coefficient is POSITIVE!")
    else:
        print("✗ WARNING: Breakout coefficient is still negative")

    # Top players by new breakout
    print()
    print("=" * 70)
    print("TOP 15 WRS BY NEW BREAKOUT SCORE")
    print("=" * 70)

    top = wrs.nlargest(15, 'breakout_score_new')
    for i, (_, r) in enumerate(top.iterrows(), 1):
        dom_val = dom_lookup.get((r['player_name'], r['draft_year']))
        if pd.notna(r['breakout_score_new']):
            print(f"{i:2}. {r['player_name']:25s}: {r['breakout_score_new']:.1f} "
                  f"(Dom: {dom_val:.1f}%)")

    # Save
    print()
    print("Saving updated data...")
    wrs.to_csv("data/wr_breakout_dominator_full.csv", index=False)
    print("Saved to data/wr_breakout_dominator_full.csv")


if __name__ == "__main__":
    main()
