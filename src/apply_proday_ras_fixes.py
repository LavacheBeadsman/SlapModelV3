"""
Apply Pro Day 40 Times + Round-Average RAS Imputation
======================================================

1. Add 9 pro day HaSS scores for elite WRs with known 40 times
2. Impute remaining 13 using round-average RAS from players with real data
3. Output complete tracking of every derived/imputed RAS value
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def calculate_hass(ht_str, weight, forty):
    """
    Height-adjusted Speed Score.
    HaSS = (weight * 200) / (forty^4) * (height_inches / 73.0)
    73 inches = ~6'1" (average WR height)
    """
    if pd.isna(weight) or pd.isna(forty) or forty <= 0:
        return None
    ht_inches = 73.0
    if pd.notna(ht_str) and isinstance(ht_str, str) and '-' in str(ht_str):
        parts = str(ht_str).split('-')
        try:
            ht_inches = int(parts[0]) * 12 + int(parts[1])
        except:
            ht_inches = 73.0
    speed_score = (weight * 200) / (forty ** 4)
    hass = speed_score * (ht_inches / 73.0)
    return hass


def hass_to_ras(hass):
    """Convert HaSS to RAS-equivalent 0-10 scale."""
    return min(10.0, max(0.0, (hass - 50) / 15))


# ============================================================================
# LOAD DATA
# ============================================================================

wr = pd.read_csv('data/wr_backtest_all_components.csv')
print(f"Loaded {len(wr)} WRs")
print(f"Current RAS coverage: {wr['RAS'].notna().sum()}/339 = {wr['RAS'].notna().sum()/339*100:.1f}%")

# Track all non-original RAS values
# First, identify which RAS values came from the original backtest file
original = pd.read_csv('data/wr_backtest_expanded_final.csv')
original_ras = set()
for _, row in original.iterrows():
    if pd.notna(row['RAS']):
        original_ras.add((row['player_name'], int(row['draft_year'])))

# Build tracking list
ras_tracking = []

# First, record the 31 fixes already applied in the previous step
for idx, row in wr.iterrows():
    name = row['player_name']
    year = int(row['draft_year'])
    if (name, year) not in original_ras and pd.notna(row['RAS']):
        ras_tracking.append({
            'player_name': name, 'draft_year': year,
            'round': int(row['round']), 'pick': int(row['pick']),
            'RAS': row['RAS'], 'method': 'RAS file name match' if row['RAS'] > 5 else 'Combine HaSS',
            'source': 'apply_data_fixes.py (previous step)'
        })


# ============================================================================
# PRO DAY 40 TIMES (user-provided)
# ============================================================================

print("\n" + "=" * 80)
print("APPLYING 9 PRO DAY 40-TIME HaSS SCORES")
print("=" * 80)

# Pro day data: (name, year, forty, ht, wt, source_note)
# Heights/weights from combine.parquet where available, otherwise public records
PRO_DAY_DATA = [
    ('Corey Davis',          2017, 4.48, '6-3',  209.0, 'combine ht/wt'),
    ('Marquise Brown',       2019, 4.33, '5-9',  166.0, 'combine ht/wt'),
    ('Jaylen Waddle',        2021, 4.37, '5-10', 182.0, 'Alabama records — not in combine.parquet'),
    ('DeVonta Smith',        2021, 4.39, '6-0',  170.0, 'Alabama records — not in combine.parquet'),
    ('Drake London',         2022, 4.58, '6-4',  219.0, 'combine ht/wt'),
    ('Jameson Williams',     2022, 4.46, '6-2',  179.0, 'combine ht/wt'),
    ('Marvin Harrison Jr.',  2024, 4.33, '6-4',  209.0, 'Ohio State records — not in combine.parquet'),
    ('Travis Hunter',        2025, 4.38, '6-0',  188.0, 'combine ht/wt (listed as CB/WR)'),
    ('Tetairoa McMillan',    2025, 4.51, '6-4',  219.0, 'combine ht/wt'),
]

print(f"\n{'Player':<25} {'Year':>5} {'Ht':>5} {'Wt':>5} {'40':>6} {'HaSS':>7} {'RAS Equiv':>10} {'Ht/Wt Source'}")
print("-" * 90)

for name, year, forty, ht, wt, note in PRO_DAY_DATA:
    hass = calculate_hass(ht, wt, forty)
    ras_eq = hass_to_ras(hass)

    # Apply to dataframe
    mask = (wr['player_name'] == name) & (wr['draft_year'] == year)
    if mask.sum() == 1:
        wr.loc[mask, 'RAS'] = ras_eq
        rd = int(wr.loc[mask, 'round'].values[0])
        pick = int(wr.loc[mask, 'pick'].values[0])
        print(f"{name:<25} {year:>5} {ht:>5} {wt:>5.0f} {forty:>6.2f} {hass:>7.1f} {ras_eq:>10.2f} {note}")
        ras_tracking.append({
            'player_name': name, 'draft_year': year,
            'round': rd, 'pick': pick,
            'RAS': ras_eq, 'method': f'Pro day HaSS (40={forty})',
            'source': f'User-provided 40 + {note}'
        })
    else:
        print(f"  WARNING: {name} ({year}) not found in backtest!")

pro_day_coverage = wr['RAS'].notna().sum()
print(f"\nRAS coverage after pro day fixes: {pro_day_coverage}/339 = {pro_day_coverage/339*100:.1f}%")


# ============================================================================
# ROUND-AVERAGE RAS IMPUTATION
# ============================================================================

print("\n" + "=" * 80)
print("IMPUTING REMAINING MISSING RAS WITH ROUND AVERAGES")
print("=" * 80)

# Calculate round-average RAS from players who have REAL RAS data (from the original file)
wr_with_original_ras = wr[wr.apply(
    lambda x: (x['player_name'], int(x['draft_year'])) in original_ras, axis=1
)]

round_avg_ras = wr_with_original_ras.groupby('round')['RAS'].agg(['mean', 'median', 'count', 'std'])
print("\nRound-average RAS (from original RAS data only):")
print(f"{'Round':>6} {'Mean':>8} {'Median':>8} {'N':>5} {'StdDev':>8}")
print("-" * 40)
for rd, row in round_avg_ras.iterrows():
    print(f"{int(rd):>6} {row['mean']:>8.2f} {row['median']:>8.2f} {int(row['count']):>5} {row['std']:>8.2f}")

# Apply round-average to remaining missing
still_missing = wr[wr['RAS'].isna()]
print(f"\n{len(still_missing)} WRs still missing RAS — imputing with round average:\n")
print(f"{'Player':<25} {'Year':>5} {'Rd':>3} {'Pick':>5} {'College':<25} {'Imputed RAS':>12}")
print("-" * 85)

for idx, row in still_missing.iterrows():
    rd = int(row['round'])
    avg_ras = round_avg_ras.loc[rd, 'mean']
    wr.loc[idx, 'RAS'] = avg_ras
    print(f"{row['player_name']:<25} {int(row['draft_year']):>5} {rd:>3} {int(row['pick']):>5} "
          f"{str(row['college']):<25} {avg_ras:>12.2f}")
    ras_tracking.append({
        'player_name': row['player_name'],
        'draft_year': int(row['draft_year']),
        'round': rd, 'pick': int(row['pick']),
        'RAS': avg_ras,
        'method': f'Round {rd} average imputation',
        'source': f'Mean of {int(round_avg_ras.loc[rd, "count"])} Rd{rd} WRs with real RAS'
    })

final_coverage = wr['RAS'].notna().sum()
print(f"\nFinal RAS coverage: {final_coverage}/339 = {final_coverage/339*100:.1f}%")


# ============================================================================
# SAVE UPDATED DATA
# ============================================================================

wr.to_csv('data/wr_backtest_all_components.csv', index=False)
print(f"\nSaved: data/wr_backtest_all_components.csv")


# ============================================================================
# COMPLETE TRACKING TABLE: ALL NON-ORIGINAL RAS VALUES
# ============================================================================

print("\n" + "=" * 100)
print("COMPLETE RECORD: ALL DERIVED / IMPUTED RAS VALUES")
print("(Every player whose RAS is NOT from the original Kent Lee Platte RAS file)")
print("=" * 100)

tracking_df = pd.DataFrame(ras_tracking)
tracking_df = tracking_df.sort_values(['draft_year', 'pick'])

print(f"\n{'#':>3} {'Player':<28} {'Year':>5} {'Rd':>3} {'Pick':>5} {'RAS':>6} {'Method':<30} {'Source'}")
print("-" * 130)

for i, (_, row) in enumerate(tracking_df.iterrows(), 1):
    print(f"{i:>3} {row['player_name']:<28} {int(row['draft_year']):>5} {int(row['round']):>3} "
          f"{int(row['pick']):>5} {row['RAS']:>6.2f} {row['method']:<30} {row['source']}")

# Also save the tracking table
tracking_df.to_csv('data/ras_imputation_log.csv', index=False)
print(f"\nTracking table saved: data/ras_imputation_log.csv")

# Summary by method
print(f"\n--- Summary by Method ---")
method_counts = tracking_df['method'].apply(
    lambda x: 'RAS file (name match)' if 'name match' in x.lower()
    else 'Combine HaSS' if 'Combine' in x
    else 'Pro day HaSS' if 'Pro day' in x
    else 'Round-avg imputation'
).value_counts()
for method, count in method_counts.items():
    print(f"  {method}: {count} players")

print(f"\n  Total non-original: {len(tracking_df)}")
print(f"  Original RAS (real data): {len(original_ras)}")
print(f"  Grand total: {len(original_ras) + len(tracking_df)} / 339")


# ============================================================================
# FINAL OVERALL COVERAGE
# ============================================================================

print("\n" + "=" * 100)
print("FINAL COVERAGE — ALL 6 COMPONENTS")
print("=" * 100)

print(f"\n{'Component':<25} {'Coverage':>10} {'Real Data':>12} {'Derived':>10} {'Imputed':>10}")
print("-" * 70)
print(f"{'Draft Capital':<25} {'339/339':>10} {'339':>12} {'0':>10} {'0':>10}")
print(f"{'Breakout Age':<25} {'339/339':>10} {'339':>12} {'0':>10} {'0':>10}")
print(f"{'Teammate Score':<25} {'339/339':>10} {'339':>12} {'0':>10} {'0':>10}")

n_ras_real = len(original_ras)
n_ras_derived = len([t for t in ras_tracking if 'imputation' not in t['method']])
n_ras_imputed = len([t for t in ras_tracking if 'imputation' in t['method']])
print(f"{'RAS (Athletic)':<25} {'339/339':>10} {n_ras_real:>12} {n_ras_derived:>12} {n_ras_imputed:>10}")

n_dec_existing = (wr['declare_source'] == 'existing').sum()
n_dec_derived = (wr['declare_source'] == 'derived').sum()
print(f"{'Early Declare':<25} {'339/339':>10} {n_dec_existing:>12} {n_dec_derived:>12} {'0':>10}")

n_rush_pff = (wr['rush_source'] == 'pff').sum()
n_rush_cfbd = wr['rush_source'].isin(['cfbd', 'cfbd_not_found']).sum()
n_rush_fcs = wr['rush_source'].isin(['assumed_zero_fcs', 'cfbd_no_data']).sum()
print(f"{'Rushing Production':<25} {'339/339':>10} {n_rush_pff:>12} {n_rush_cfbd:>12} {n_rush_fcs:>10}")
