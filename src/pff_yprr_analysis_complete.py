"""
Complete YPRR Analysis with all PFF receiving files (2016-2025 draft classes)
"""

import pandas as pd
import numpy as np
import re
from numpy.linalg import lstsq

# Season-to-file mapping (verified by player lookup)
SEASON_FILE_MAP = {
    2016: 'data/receiving_summary (2).csv',   # 2015 college season
    2017: 'data/receiving_summary (3).csv',   # 2016 college season
    2018: 'data/receiving_summary (4).csv',   # 2017 college season
    2019: 'data/receiving_summary (5).csv',   # 2018 college season
    2020: 'data/receiving_summary (21).csv',  # 2019 college season
    2021: 'data/receiving_summary (20).csv',  # 2020 college season
    2022: 'data/receiving_summary (19).csv',  # 2021 college season
    2023: 'data/receiving_summary (18).csv',  # 2022 college season
    2024: 'data/receiving_summary (17).csv',  # 2023 college season
    2025: 'data/receiving_summary (16).csv',  # 2024 college season
}

def normalize_name(name):
    """Normalize player names for matching."""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    # Remove punctuation and apostrophes
    name = re.sub(r"['.,-]", "", name)
    # Remove suffixes
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name)
    return " ".join(name.split())

def partial_correlation(x, y, z):
    """Calculate partial correlation of x and y, controlling for z."""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]

    if len(x) < 10:
        return np.nan

    # Regress out z from both x and y
    X = np.column_stack([np.ones(len(z)), z])
    coeffs_x = lstsq(X, x, rcond=None)[0]
    x_resid = x - (coeffs_x[0] + coeffs_x[1] * z)

    coeffs_y = lstsq(X, y, rcond=None)[0]
    y_resid = y - (coeffs_y[0] + coeffs_y[1] * z)

    return np.corrcoef(x_resid, y_resid)[0, 1]

def dc_score(pick):
    """Calculate draft capital score using gentler curve formula."""
    return 100 - 2.40 * (pick**0.62 - 1)

# Load WR backtest data
print("Loading WR backtest data...")
wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
print(f"Total WRs in backtest: {len(wr_backtest)}")

# Filter to draft years with PFF coverage (2016-2024 for evaluation)
# 2025 has no NFL outcomes yet
wr_backtest = wr_backtest[wr_backtest['draft_year'].isin(range(2016, 2025))]
print(f"WRs in draft years 2016-2024: {len(wr_backtest)}")

# Merge with PFF data
merged_data = []

for draft_year in range(2016, 2025):
    if draft_year not in SEASON_FILE_MAP:
        continue

    # Load PFF file for this draft year
    pff_file = SEASON_FILE_MAP[draft_year]
    try:
        pff_df = pd.read_csv(pff_file)
    except FileNotFoundError:
        print(f"File not found: {pff_file}")
        continue

    # Filter to WRs only (and TEs that might have good receiving stats)
    pff_df = pff_df[pff_df['position'].isin(['WR'])]

    # Normalize PFF names
    pff_df['name_normalized'] = pff_df['player'].apply(normalize_name)

    # Get backtest WRs for this draft year
    wr_year = wr_backtest[wr_backtest['draft_year'] == draft_year].copy()
    wr_year['name_normalized'] = wr_year['player_name'].apply(normalize_name)

    # Merge
    for _, wr in wr_year.iterrows():
        wr_name = wr['name_normalized']

        # Find match in PFF
        pff_match = pff_df[pff_df['name_normalized'] == wr_name]

        if len(pff_match) == 1:
            pff_row = pff_match.iloc[0]
            merged_data.append({
                'player_name': wr['player_name'],
                'draft_year': draft_year,
                'pick': wr['pick'],
                'round': wr['round'],
                'college': wr['college'],
                'best_ppr': wr['best_ppr'],
                'hit24': wr['hit24'],
                'yprr': pff_row['yprr'],
                'yards': pff_row['yards'],
                'receptions': pff_row['receptions'],
                'grades_offense': pff_row['grades_offense'],
                'pff_file': pff_file
            })

# Create merged dataframe
merged_df = pd.DataFrame(merged_data)
print(f"\nMerged data: {len(merged_df)} WRs")

# Calculate match rates by draft year
print("\nMatch rates by draft year:")
for draft_year in range(2016, 2025):
    total = len(wr_backtest[wr_backtest['draft_year'] == draft_year])
    matched = len(merged_df[merged_df['draft_year'] == draft_year])
    pct = matched / total * 100 if total > 0 else 0
    print(f"  {draft_year}: {matched}/{total} ({pct:.1f}%)")

# Save merged data
merged_df.to_csv('data/wr_pff_complete.csv', index=False)
print(f"\nSaved merged data to data/wr_pff_complete.csv")

# Calculate DC scores
merged_df['dc_score'] = merged_df['pick'].apply(dc_score)

# Filter to players with valid outcomes (best_ppr > 0 means they played)
valid_df = merged_df[merged_df['best_ppr'] > 0].copy()
print(f"\nPlayers with NFL production: {len(valid_df)}")

# ============================================
# CORRELATION ANALYSIS
# ============================================
print("\n" + "="*60)
print("YPRR CORRELATION ANALYSIS")
print("="*60)

# Raw correlations
yprr_ppr_raw = np.corrcoef(merged_df['yprr'], merged_df['best_ppr'])[0, 1]
dc_ppr_raw = np.corrcoef(merged_df['dc_score'], merged_df['best_ppr'])[0, 1]
yprr_dc_raw = np.corrcoef(merged_df['yprr'], merged_df['dc_score'])[0, 1]

print(f"\nRaw Correlations (n={len(merged_df)}):")
print(f"  YPRR vs Best PPR:      r = {yprr_ppr_raw:.3f}")
print(f"  DC vs Best PPR:        r = {dc_ppr_raw:.3f}")
print(f"  YPRR vs DC:            r = {yprr_dc_raw:.3f}")

# Partial correlation (YPRR vs PPR, controlling for DC)
yprr_values = merged_df['yprr'].values
ppr_values = merged_df['best_ppr'].values
dc_values = merged_df['dc_score'].values

yprr_partial = partial_correlation(yprr_values, ppr_values, dc_values)

print(f"\nPartial Correlations (controlling for DC):")
print(f"  YPRR vs Best PPR | DC:  r = {yprr_partial:.3f}")

# Hit rate analysis
print("\n" + "-"*60)
print("HIT RATE ANALYSIS (Top 24 WR Season)")
print("-"*60)

# YPRR median split
yprr_median = merged_df['yprr'].median()
high_yprr = merged_df[merged_df['yprr'] >= yprr_median]
low_yprr = merged_df[merged_df['yprr'] < yprr_median]

high_yprr_hit = high_yprr['hit24'].mean() * 100
low_yprr_hit = low_yprr['hit24'].mean() * 100

print(f"\nYPRR median: {yprr_median:.2f}")
print(f"  High YPRR hit rate:  {high_yprr_hit:.1f}% ({high_yprr['hit24'].sum()}/{len(high_yprr)})")
print(f"  Low YPRR hit rate:   {low_yprr_hit:.1f}% ({low_yprr['hit24'].sum()}/{len(low_yprr)})")
print(f"  Difference:          {high_yprr_hit - low_yprr_hit:+.1f}%")

# Top 10 vs Bottom 10 YPRR
top10_yprr = merged_df.nlargest(10, 'yprr')
bot10_yprr = merged_df.nsmallest(10, 'yprr')

print(f"\nTop 10 YPRR players:")
for _, row in top10_yprr.iterrows():
    print(f"  {row['player_name']:25} YPRR={row['yprr']:.2f}  Pick={row['pick']:3}  Hit24={row['hit24']}")
print(f"  Hit rate: {top10_yprr['hit24'].mean()*100:.0f}%")

print(f"\nBottom 10 YPRR players:")
for _, row in bot10_yprr.iterrows():
    print(f"  {row['player_name']:25} YPRR={row['yprr']:.2f}  Pick={row['pick']:3}  Hit24={row['hit24']}")
print(f"  Hit rate: {bot10_yprr['hit24'].mean()*100:.0f}%")

# ============================================
# BY ROUND ANALYSIS
# ============================================
print("\n" + "-"*60)
print("YPRR ANALYSIS BY DRAFT ROUND")
print("-"*60)

for rnd_group, rnd_name in [([1, 2], "Rounds 1-2"), ([3, 4], "Rounds 3-4"), ([5, 6, 7], "Rounds 5-7")]:
    rnd_df = merged_df[merged_df['round'].isin(rnd_group)]
    if len(rnd_df) < 10:
        continue

    # Partial correlation
    yprr_partial_rnd = partial_correlation(
        rnd_df['yprr'].values,
        rnd_df['best_ppr'].values,
        rnd_df['dc_score'].values
    )

    # Hit rate by YPRR median
    yprr_med = rnd_df['yprr'].median()
    high = rnd_df[rnd_df['yprr'] >= yprr_med]
    low = rnd_df[rnd_df['yprr'] < yprr_med]

    print(f"\n{rnd_name} (n={len(rnd_df)}):")
    print(f"  YPRR partial corr:   r = {yprr_partial_rnd:.3f}")
    print(f"  High YPRR hit rate:  {high['hit24'].mean()*100:.1f}% ({high['hit24'].sum()}/{len(high)})")
    print(f"  Low YPRR hit rate:   {low['hit24'].mean()*100:.1f}% ({low['hit24'].sum()}/{len(low)})")

# ============================================
# CONCLUSION
# ============================================
print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if yprr_partial > 0.15:
    print(f"\nYPRR partial correlation ({yprr_partial:.3f}) is POSITIVE and meaningful.")
    print("Consider adding YPRR to the SLAP model.")
elif yprr_partial > 0.05:
    print(f"\nYPRR partial correlation ({yprr_partial:.3f}) is weakly positive.")
    print("YPRR might add marginal value, but effect is small.")
elif yprr_partial > -0.05:
    print(f"\nYPRR partial correlation ({yprr_partial:.3f}) is essentially ZERO.")
    print("YPRR adds NO predictive value beyond draft capital.")
else:
    print(f"\nYPRR partial correlation ({yprr_partial:.3f}) is NEGATIVE!")
    print("Higher YPRR is actually associated with WORSE NFL outcomes after controlling for DC.")
    print("DO NOT add YPRR to the SLAP model.")
