"""
Comprehensive SLAP Model Audit
"""

import pandas as pd
import numpy as np
import os
from glob import glob

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# =====================================================
# PART 1: COMPLETE DATA INVENTORY
# =====================================================

print("="*80)
print("PART 1: COMPLETE DATA INVENTORY")
print("="*80)

# Load all data files
wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
rb_backtest = pd.read_csv('data/rb_backtest_with_receiving.csv')

# Check for 2026 prospects
prospects_file = 'data/prospects_final.csv'
if os.path.exists(prospects_file):
    prospects_2026 = pd.read_csv(prospects_file)
else:
    prospects_2026 = pd.DataFrame()

# Load PFF data
wr_pff = pd.read_csv('data/wr_pff_complete.csv')

# Load RAS data
ras_file = 'data/WR_RAS_2020_to_2025.csv'
if os.path.exists(ras_file):
    ras_data = pd.read_csv(ras_file)
else:
    ras_data = pd.DataFrame()

# Load SLAP output files
slap_files = glob('output/slap_*.csv')

print("\n--- 1.1 WR DATA COVERAGE ---\n")

# Define year groups
year_groups = {
    '2015-2018': [2015, 2016, 2017, 2018],
    '2019-2021': [2019, 2020, 2021],
    '2022-2024': [2022, 2023, 2024],
    '2025': [2025],
    '2026': [2026]
}

# WR fields to check
wr_fields = ['pick', 'birthdate', 'breakout_age', 'peak_dominator', 'RAS', 'best_ppr', 'hit24']

print("WR Backtest columns:", list(wr_backtest.columns))
print(f"\nTotal WRs in backtest: {len(wr_backtest)}")
print(f"Draft years in backtest: {sorted(wr_backtest['draft_year'].unique())}")

# Count coverage by year group
print("\nWR DATA COVERAGE BY YEAR GROUP:")
print("-" * 100)

for group_name, years in year_groups.items():
    wr_group = wr_backtest[wr_backtest['draft_year'].isin(years)]
    total = len(wr_group)

    if total == 0:
        print(f"{group_name}: No data")
        continue

    print(f"\n{group_name} (n={total}):")

    for field in wr_fields:
        if field in wr_group.columns:
            non_null = wr_group[field].notna().sum()
            # For best_ppr, check for actual values (not 0 or 999)
            if field == 'best_ppr':
                valid = ((wr_group[field] > 0) & (wr_group[field] < 999)).sum()
                print(f"  {field:20}: {valid}/{total} ({valid/total*100:.1f}%) with valid NFL data")
            else:
                print(f"  {field:20}: {non_null}/{total} ({non_null/total*100:.1f}%)")
        else:
            print(f"  {field:20}: COLUMN MISSING")

# Check PFF coverage for WRs
print("\n\nPFF YPRR COVERAGE FOR WRs:")
print("-" * 60)
for year in sorted(wr_backtest['draft_year'].unique()):
    wr_year = wr_backtest[wr_backtest['draft_year'] == year]
    pff_year = wr_pff[wr_pff['draft_year'] == year] if year in wr_pff['draft_year'].values else pd.DataFrame()
    total = len(wr_year)
    matched = len(pff_year)
    print(f"  {year}: {matched}/{total} ({matched/total*100:.1f}%)" if total > 0 else f"  {year}: No backtest data")

print("\n\n--- 1.2 RB DATA COVERAGE ---\n")

print("RB Backtest columns:", list(rb_backtest.columns))
print(f"\nTotal RBs in backtest: {len(rb_backtest)}")
print(f"Draft years in backtest: {sorted(rb_backtest['draft_year'].unique())}")

# RB fields to check
rb_fields = ['pick', 'age', 'RAS', 'rec_yards', 'team_pass_att', 'best_ppr', 'hit24']

print("\nRB DATA COVERAGE BY YEAR GROUP:")
print("-" * 100)

for group_name, years in year_groups.items():
    rb_group = rb_backtest[rb_backtest['draft_year'].isin(years)]
    total = len(rb_group)

    if total == 0:
        print(f"{group_name}: No data")
        continue

    print(f"\n{group_name} (n={total}):")

    for field in rb_fields:
        if field in rb_group.columns:
            non_null = rb_group[field].notna().sum()
            if field == 'best_ppr':
                valid = ((rb_group[field] > 0) & (rb_group[field] < 999)).sum()
                print(f"  {field:20}: {valid}/{total} ({valid/total*100:.1f}%) with valid NFL data")
            else:
                print(f"  {field:20}: {non_null}/{total} ({non_null/total*100:.1f}%)")
        else:
            print(f"  {field:20}: COLUMN MISSING")

print("\n\n--- 1.3 PLAYERS MISSING CRITICAL DATA (Rounds 1-4) ---\n")

# WRs missing data in rounds 1-4
wr_r14 = wr_backtest[(wr_backtest['round'] <= 4)]
print(f"WRs in Rounds 1-4: {len(wr_r14)}")

wr_missing = []
for _, row in wr_r14.iterrows():
    missing = []
    if pd.isna(row.get('birthdate')) or row.get('birthdate') == 'MISSING':
        missing.append('birthdate')
    if pd.isna(row.get('breakout_age')):
        missing.append('breakout_age')
    if pd.isna(row.get('RAS')):
        missing.append('RAS')
    if pd.isna(row.get('peak_dominator')):
        missing.append('dominator')

    if missing:
        wr_missing.append({
            'player': row['player_name'],
            'year': row['draft_year'],
            'pick': row['pick'],
            'missing': ', '.join(missing)
        })

print(f"\nWRs (Rd 1-4) missing critical data: {len(wr_missing)}")
if wr_missing:
    print("\n| Player | Year | Pick | Missing Fields |")
    print("|--------|------|------|----------------|")
    for p in wr_missing[:30]:  # Show first 30
        print(f"| {p['player'][:25]:25} | {p['year']} | {p['pick']:3} | {p['missing']} |")
    if len(wr_missing) > 30:
        print(f"... and {len(wr_missing) - 30} more")

# RBs missing data in rounds 1-4
rb_r14 = rb_backtest[(rb_backtest['round'] <= 4)]
print(f"\n\nRBs in Rounds 1-4: {len(rb_r14)}")

rb_missing = []
for _, row in rb_r14.iterrows():
    missing = []
    if pd.isna(row.get('age')):
        missing.append('age')
    if pd.isna(row.get('RAS')):
        missing.append('RAS')
    if pd.isna(row.get('rec_yards')):
        missing.append('rec_yards')
    if pd.isna(row.get('team_pass_att')):
        missing.append('team_pass_att')

    if missing:
        rb_missing.append({
            'player': row['player_name'],
            'year': row['draft_year'],
            'pick': row['pick'],
            'missing': ', '.join(missing)
        })

print(f"\nRBs (Rd 1-4) missing critical data: {len(rb_missing)}")
if rb_missing:
    print("\n| Player | Year | Pick | Missing Fields |")
    print("|--------|------|------|----------------|")
    for p in rb_missing[:30]:
        print(f"| {p['player'][:25]:25} | {p['year']} | {p['pick']:3} | {p['missing']} |")
    if len(rb_missing) > 30:
        print(f"... and {len(rb_missing) - 30} more")

# =====================================================
# PART 2: NFL OUTCOME DATA CHECK
# =====================================================

print("\n\n" + "="*80)
print("PART 2: NFL OUTCOME DATA CHECK")
print("="*80)

print("\n--- 2.1 NFL Fantasy PPG Coverage ---\n")

print("WR NFL Outcome Coverage by Draft Year:")
print("-" * 60)
for year in sorted(wr_backtest['draft_year'].unique()):
    wr_year = wr_backtest[wr_backtest['draft_year'] == year]
    total = len(wr_year)
    with_ppg = wr_year[wr_year['best_ppr'] > 0].shape[0]
    without = total - with_ppg
    print(f"  {year}: {with_ppg}/{total} have NFL PPG ({without} without)")

print("\nRB NFL Outcome Coverage by Draft Year:")
print("-" * 60)
for year in sorted(rb_backtest['draft_year'].unique()):
    rb_year = rb_backtest[rb_backtest['draft_year'] == year]
    total = len(rb_year)
    with_ppg = rb_year[rb_year['best_ppr'] > 0].shape[0]
    without = total - with_ppg
    print(f"  {year}: {with_ppg}/{total} have NFL PPG ({without} without)")

print("\n--- 2.2 How is NFL PPG Calculated? ---\n")
print("From column inspection:")
print("  - 'best_ppr': Appears to be best single season PPR total")
print("  - 'best_rank': Best season-end WR/RB ranking")
print("  - 'best_ppg': Best season PPG (points per game)")
print("  - 'hit24': 1 if ever finished top-24 at position")
print("  - 'hit12': 1 if ever finished top-12 at position")

# Check hit rate consistency
print("\n--- 2.3 Hit Rate Statistics ---\n")
print("WR Hit Rates:")
print(f"  Total WRs: {len(wr_backtest)}")
print(f"  Hit24 count: {wr_backtest['hit24'].sum()} ({wr_backtest['hit24'].mean()*100:.1f}%)")
print(f"  Hit12 count: {wr_backtest['hit12'].sum()} ({wr_backtest['hit12'].mean()*100:.1f}%)")

print("\nRB Hit Rates:")
print(f"  Total RBs: {len(rb_backtest)}")
print(f"  Hit24 count: {rb_backtest['hit24'].sum()} ({rb_backtest['hit24'].mean()*100:.1f}%)")
print(f"  Hit12 count: {rb_backtest['hit12'].sum()} ({rb_backtest['hit12'].mean()*100:.1f}%)")

# =====================================================
# PART 3: WHAT DATA EXISTS THAT WE'RE NOT USING
# =====================================================

print("\n\n" + "="*80)
print("PART 3: UNUSED DATA SOURCES")
print("="*80)

print("\n--- 3.1 PFF Columns Available ---\n")

# Load a sample PFF file to see all columns
pff_sample = pd.read_csv('data/receiving_summary (18).csv')
print("PFF Receiving Columns:")
for i, col in enumerate(pff_sample.columns):
    print(f"  {i+1}. {col}")

# Load RB PFF file
pff_rb_files = glob('data/rushing_summary*.csv')
if pff_rb_files:
    pff_rb_sample = pd.read_csv(pff_rb_files[0])
    print("\nPFF Rushing Columns:")
    for i, col in enumerate(pff_rb_sample.columns):
        print(f"  {i+1}. {col}")

print("\n--- 3.2 Columns Currently Used vs Unused ---\n")

used_wr_pff = ['player', 'position', 'yprr', 'yards', 'grades_offense']
unused_wr_pff = [c for c in pff_sample.columns if c not in used_wr_pff]

print("WR PFF - USED:")
for c in used_wr_pff:
    print(f"  ✓ {c}")

print("\nWR PFF - UNUSED (potentially valuable):")
valuable_unused = ['drop_rate', 'contested_catch_rate', 'yards_after_catch', 'avg_depth_of_target',
                   'grades_pass_route', 'avoided_tackles', 'first_downs', 'touchdowns']
for c in valuable_unused:
    if c in pff_sample.columns:
        print(f"  ? {c}")

print("\n--- 3.3 Data Points Worth Testing ---\n")

print("| Data Point | Source | Available? | Likely Impact | Worth Testing? |")
print("|------------|--------|-----------|---------------|----------------|")
print("| Drop rate | PFF | Yes | Medium | YES |")
print("| Contested catch % | PFF | Yes | Medium | YES |")
print("| Avg depth of target | PFF | Yes | Low | Maybe |")
print("| Route grade | PFF | Yes | Medium | YES |")
print("| YAC/reception | PFF | Yes | Medium | Maybe |")
print("| Avoided tackles | PFF | Yes | Low | No |")
print("| 40 time (raw) | RAS/Combine | Yes | Low | No (in RAS) |")
print("| Height | Combine | Partial | Low | No |")
print("| Weight | Combine | Partial | Low | No (in Speed Score) |")
print("| Conference | CFBD | Yes | Low | Maybe |")
print("| Games played | CFBD | Yes | Low | No |")
print("| Elusive rating | PFF RB | Yes | Tested | NO (r=0.055) |")

# =====================================================
# PART 4: WITHIN-MODEL IMPROVEMENTS
# =====================================================

print("\n\n" + "="*80)
print("PART 4: WITHIN-MODEL IMPROVEMENTS")
print("="*80)

# Load SLAP scores if available
slap_all = None
for f in slap_files:
    if 'complete' in f.lower() and 'all' in f.lower():
        slap_all = pd.read_csv(f)
        break
    elif 'complete' in f.lower():
        slap_all = pd.read_csv(f)

if slap_all is not None:
    print("\n--- 4.1 Current SLAP Score Distribution ---\n")

    wr_slap = slap_all[slap_all['position'] == 'WR'] if 'position' in slap_all.columns else None
    rb_slap = slap_all[slap_all['position'] == 'RB'] if 'position' in slap_all.columns else None

    if wr_slap is not None and 'slap_score' in wr_slap.columns:
        print("WR SLAP Score Distribution:")
        print(f"  Mean: {wr_slap['slap_score'].mean():.1f}")
        print(f"  Median: {wr_slap['slap_score'].median():.1f}")
        print(f"  Std: {wr_slap['slap_score'].std():.1f}")
        print(f"  Min: {wr_slap['slap_score'].min():.1f}")
        print(f"  Max: {wr_slap['slap_score'].max():.1f}")

    if rb_slap is not None and 'slap_score' in rb_slap.columns:
        print("\nRB SLAP Score Distribution:")
        print(f"  Mean: {rb_slap['slap_score'].mean():.1f}")
        print(f"  Median: {rb_slap['slap_score'].median():.1f}")
        print(f"  Std: {rb_slap['slap_score'].std():.1f}")
        print(f"  Min: {rb_slap['slap_score'].min():.1f}")
        print(f"  Max: {rb_slap['slap_score'].max():.1f}")

print("\n--- 4.2 Missing Data Imputation Methods ---\n")
print("Current imputation (from CLAUDE.md):")
print("  - Missing RAS: Use position average RAS")
print("  - Missing breakout age: Use worst tier (never broke out)")
print("  - Missing RB production: Cannot calculate - needs rec_yards + team_pass_att")

print("\n--- 4.3 Formula Review ---\n")
print("Current formulas:")
print("  DC: 100 - 2.40 × (pick^0.62 - 1)")
print("  WR Breakout: age_tier + min((dominator-20)*0.5, 9.9)")
print("  RB Production: (rec_yards / team_pass_att) × age_weight × 100 / 1.75")
print("  Athletic: RAS × 10 for WRs, Speed Score for RBs")

# =====================================================
# PART 5: EXTERNAL VALIDATION (Placeholder)
# =====================================================

print("\n\n" + "="*80)
print("PART 5: EXTERNAL VALIDATION")
print("="*80)

print("\n--- 5.1 SLAP vs Draft Order Comparison ---\n")

# Calculate correlation between SLAP and NFL outcomes vs DC-only
if slap_all is not None and 'slap_score' in slap_all.columns:
    # Merge with outcomes
    wr_merged = wr_backtest.merge(
        slap_all[slap_all['position'] == 'WR'][['player_name', 'draft_year', 'slap_score']] if 'position' in slap_all.columns else slap_all[['player_name', 'draft_year', 'slap_score']],
        on=['player_name', 'draft_year'],
        how='inner'
    )

    if len(wr_merged) > 20:
        from numpy import corrcoef

        dc_only_r = corrcoef(1/wr_merged['pick']**0.5, wr_merged['best_ppr'])[0,1]
        slap_r = corrcoef(wr_merged['slap_score'], wr_merged['best_ppr'])[0,1]

        print(f"WR Correlation with Best PPR (n={len(wr_merged)}):")
        print(f"  Draft order only (1/sqrt(pick)): r = {dc_only_r:.3f}")
        print(f"  SLAP score: r = {slap_r:.3f}")
        print(f"  Improvement: {(slap_r - dc_only_r):.3f}")

print("\n--- 5.2 Year-by-Year Performance ---\n")
print("(Would require historical ADP data for comparison)")

# =====================================================
# Summary Stats
# =====================================================

print("\n\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal WRs in database: {len(wr_backtest)}")
print(f"Total RBs in database: {len(rb_backtest)}")
print(f"Draft years covered: 2015-2024")

wr_complete = wr_backtest[
    wr_backtest['pick'].notna() &
    wr_backtest['breakout_age'].notna() &
    wr_backtest['peak_dominator'].notna()
].shape[0]
print(f"\nWRs with complete core data (pick + breakout + dominator): {wr_complete}/{len(wr_backtest)} ({wr_complete/len(wr_backtest)*100:.1f}%)")

rb_complete = rb_backtest[
    rb_backtest['pick'].notna() &
    rb_backtest['rec_yards'].notna() &
    rb_backtest['team_pass_att'].notna()
].shape[0]
print(f"RBs with complete core data (pick + rec_yards + team_pass): {rb_complete}/{len(rb_backtest)} ({rb_complete/len(rb_backtest)*100:.1f}%)")
