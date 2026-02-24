"""
SLAP V5 Master Database — Comprehensive Data Quality Audit
===========================================================
Checks 10 categories of data quality across all 939 rows.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

pd.set_option('display.max_colwidth', 40)
pd.set_option('display.width', 200)

# ============================================================================
# LOAD DATA
# ============================================================================
df = pd.read_csv('/home/user/SlapModelV3/output/slap_v5_master_database.csv')

print("=" * 100)
print("SLAP V5 MASTER DATABASE — COMPREHENSIVE DATA QUALITY AUDIT")
print("=" * 100)
print(f"\nTotal rows: {len(df)}")
print(f"Positions:  {df['position'].value_counts().to_dict()}")
print(f"Datasets:   {df['dataset'].value_counts().to_dict()}")
print(f"Columns:    {list(df.columns)}")

# Subsets
wr = df[df['position'] == 'WR']
rb = df[df['position'] == 'RB']
te = df[df['position'] == 'TE']
backtest = df[df['dataset'] == 'backtest']
prospects = df[df['dataset'] == '2026_prospect']

# Track issues
issues = {
    '1. Breakout Age': {'critical': [], 'warning': [], 'info': []},
    '2. Early Declare': {'critical': [], 'warning': [], 'info': []},
    '3. Component Ranges': {'critical': [], 'warning': [], 'info': []},
    '4. Missing Data': {'critical': [], 'warning': [], 'info': []},
    '5. Duplicates': {'critical': [], 'warning': [], 'info': []},
    '6. DC Consistency': {'critical': [], 'warning': [], 'info': []},
    '7. Cross-Position Contamination': {'critical': [], 'warning': [], 'info': []},
    '8. Dataset Labels': {'critical': [], 'warning': [], 'info': []},
    '9. NFL Outcomes': {'critical': [], 'warning': [], 'info': []},
    '10. Ranking Consistency': {'critical': [], 'warning': [], 'info': []},
}


# ============================================================================
# CHECK 1: BREAKOUT AGE AUDIT
# ============================================================================
print(f"\n\n{'='*100}")
print("CHECK 1: BREAKOUT AGE AUDIT")
print("=" * 100)

for pos_name, pos_df in [('WR', wr), ('TE', te)]:
    print(f"\n  --- {pos_name} breakout_age ---")
    ba = pos_df['breakout_age']
    non_null = ba.dropna()
    null_count = ba.isna().sum()
    print(f"  Total: {len(pos_df)} | Non-null: {len(non_null)} | NaN: {null_count}")

    if len(non_null) > 0:
        print(f"  Range: {non_null.min()} - {non_null.max()}")
        print(f"  Distribution: {non_null.value_counts().sort_index().to_dict()}")

    # Flag values outside 18-23
    outside = pos_df[ba.notna() & ((ba < 18) | (ba > 23))]
    if len(outside) > 0:
        msg = f"{pos_name}: {len(outside)} players with breakout_age outside 18-23 range"
        issues['1. Breakout Age']['critical'].append(msg)
        print(f"  ** CRITICAL: {msg}")
        for _, r in outside.iterrows():
            print(f"     {r['player_name']} ({r['college']}, {int(r['draft_year'])}): breakout_age = {r['breakout_age']}")
    else:
        print(f"  OK: All non-null breakout_age values are in 18-23 range")

    # Flag values <= 18 (suspicious)
    suspicious = pos_df[ba.notna() & (ba <= 18)]
    if len(suspicious) > 0:
        msg = f"{pos_name}: {len(suspicious)} players with breakout_age <= 18 (suspicious)"
        issues['1. Breakout Age']['warning'].append(msg)
        print(f"  ** WARNING: {msg}")
        for _, r in suspicious.iterrows():
            print(f"     {r['player_name']} ({r['college']}, {int(r['draft_year'])}): breakout_age = {r['breakout_age']}")
    else:
        print(f"  OK: No suspiciously low breakout_age values (<= 18)")

print(f"\n  --- RB breakout_age (should not apply) ---")
rb_ba = rb['breakout_age'].dropna()
if len(rb_ba) > 0:
    msg = f"RB: {len(rb_ba)} RBs have breakout_age values (should be NaN for RBs)"
    issues['1. Breakout Age']['warning'].append(msg)
    print(f"  ** WARNING: {msg}")
else:
    print(f"  OK: All RB breakout_age values are NaN (as expected)")


# ============================================================================
# CHECK 2: EARLY DECLARE AUDIT
# ============================================================================
print(f"\n\n{'='*100}")
print("CHECK 2: EARLY DECLARE AUDIT")
print("=" * 100)

# Check that only WRs have early_declare_score
for pos_name, pos_df in [('RB', rb), ('TE', te)]:
    non_null = pos_df['early_declare_score'].dropna()
    if len(non_null) > 0:
        msg = f"{pos_name}: {len(non_null)} players have early_declare_score (should be NaN)"
        issues['2. Early Declare']['critical'].append(msg)
        print(f"  ** CRITICAL: {msg}")
        for _, r in non_null.head(5).iteritems() if hasattr(non_null, 'iteritems') else non_null.head(5).items():
            row = pos_df.loc[r if isinstance(r, int) else non_null.index[0]]
    else:
        print(f"  OK: {pos_name} has no early_declare_score values (all NaN)")

# WR: check values are exactly 0 or 100
wr_ed = wr['early_declare_score'].dropna()
bad_ed = wr[wr['early_declare_score'].notna() & ~wr['early_declare_score'].isin([0.0, 100.0])]
if len(bad_ed) > 0:
    msg = f"WR: {len(bad_ed)} players have early_declare_score not in {{0, 100}}"
    issues['2. Early Declare']['critical'].append(msg)
    print(f"  ** CRITICAL: {msg}")
    for _, r in bad_ed.iterrows():
        print(f"     {r['player_name']}: early_declare_score = {r['early_declare_score']}")
else:
    print(f"  OK: All WR early_declare_score values are exactly 0 or 100 (n={len(wr_ed)})")
    print(f"      Distribution: {wr_ed.value_counts().sort_index().to_dict()}")

# Cross-check: early_declare = 100 AND breakout_age >= 22 (suspicious combo)
suspicious_ed = wr[(wr['early_declare_score'] == 100) & (wr['breakout_age'].notna()) & (wr['breakout_age'] >= 22)]
if len(suspicious_ed) > 0:
    msg = f"WR: {len(suspicious_ed)} players with early_declare=100 AND breakout_age>=22 (unusual combo)"
    issues['2. Early Declare']['warning'].append(msg)
    print(f"  ** WARNING: {msg}")
    for _, r in suspicious_ed.iterrows():
        print(f"     {r['player_name']} ({r['college']}, {int(r['draft_year'])}): early_declare=100, breakout_age={r['breakout_age']}")
else:
    print(f"  OK: No WRs with both early_declare=100 and breakout_age>=22")


# ============================================================================
# CHECK 3: COMPONENT SCORE RANGE AUDIT
# ============================================================================
print(f"\n\n{'='*100}")
print("CHECK 3: COMPONENT SCORE RANGE AUDIT")
print("=" * 100)

def check_range(series, col_name, low, high, severity='critical', subset_label='ALL', exact_vals=None):
    """Check if values fall within expected range. If exact_vals provided, check for exact matches."""
    non_null = series.dropna()
    if len(non_null) == 0:
        print(f"  {col_name} ({subset_label}): no non-null values")
        return

    if exact_vals is not None:
        bad = non_null[~non_null.isin(exact_vals)]
        if len(bad) > 0:
            msg = f"{col_name} ({subset_label}): {len(bad)} values not in {exact_vals}"
            issues['3. Component Ranges'][severity].append(msg)
            print(f"  ** {severity.upper()}: {msg}")
            # Show up to 5 examples
            bad_idx = bad.index[:5]
            for idx in bad_idx:
                row = df.loc[idx]
                print(f"     {row['player_name']}: {col_name}={row[col_name.split(' ')[0] if ' ' in col_name else col_name]}")
        else:
            print(f"  OK: {col_name} ({subset_label}): all {len(non_null)} values in {exact_vals}")
    else:
        below = non_null[non_null < low]
        above = non_null[non_null > high]
        if len(below) > 0 or len(above) > 0:
            msg = f"{col_name} ({subset_label}): {len(below)} below {low}, {len(above)} above {high} (range: {non_null.min():.2f}-{non_null.max():.2f})"
            issues['3. Component Ranges'][severity].append(msg)
            print(f"  ** {severity.upper()}: {msg}")
            for idx in below.index[:3]:
                row = df.loc[idx]
                print(f"     BELOW: {row['player_name']} = {non_null[idx]:.2f}")
            for idx in above.index[:3]:
                row = df.loc[idx]
                print(f"     ABOVE: {row['player_name']} = {non_null[idx]:.2f}")
        else:
            print(f"  OK: {col_name} ({subset_label}): range {non_null.min():.2f}-{non_null.max():.2f} within [{low}, {high}] (n={len(non_null)})")

# Universal columns
check_range(df['slap_display_score'], 'slap_display_score', 1, 99, 'critical', 'ALL')
check_range(df['slap_model_score'], 'slap_model_score', 20, 100, 'warning', 'ALL')
check_range(df['dc_score'], 'dc_score', 1, 99, 'critical', 'ALL')
check_range(df['prospect_profile'], 'prospect_profile', 0, 100, 'warning', 'ALL')

# WR-specific
check_range(wr['enhanced_breakout'], 'enhanced_breakout', 0, 100, 'critical', 'WR')
check_range(wr['teammate_score'], 'teammate_score', 0, 100, 'critical', 'WR', exact_vals=[0.0, 100.0])
check_range(wr['early_declare_score'], 'early_declare_score', 0, 100, 'critical', 'WR', exact_vals=[0.0, 100.0])

# RB-specific
check_range(rb['production_score'], 'production_score', 0, 100, 'critical', 'RB')
check_range(rb['speed_score'], 'speed_score', 0, 100, 'critical', 'RB')

# TE-specific
check_range(te['te_breakout_score'], 'te_breakout_score', 0, 100, 'critical', 'TE')
check_range(te['te_production_score'], 'te_production_score', 0, 100, 'critical', 'TE')
check_range(te['ras_score'], 'ras_score', 0, 100, 'critical', 'TE')

# peak_dominator (WR + TE)
wr_pd = wr['peak_dominator'].dropna()
te_pd = te['peak_dominator'].dropna()
print(f"\n  --- peak_dominator (should be 0-100, >100 = data error per CLAUDE.md) ---")
check_range(wr['peak_dominator'], 'peak_dominator', 0, 100, 'warning', 'WR')
check_range(te['peak_dominator'], 'peak_dominator', 0, 100, 'warning', 'TE')


# ============================================================================
# CHECK 4: MISSING DATA AUDIT
# ============================================================================
print(f"\n\n{'='*100}")
print("CHECK 4: MISSING DATA AUDIT")
print("=" * 100)

# Core columns that should NEVER be missing
for col in ['slap_display_score', 'slap_model_score', 'dc_score']:
    missing = df[col].isna().sum()
    if missing > 0:
        msg = f"{col}: {missing} missing values (should be 0)"
        issues['4. Missing Data']['critical'].append(msg)
        print(f"  ** CRITICAL: {msg}")
        missing_rows = df[df[col].isna()]
        for _, r in missing_rows.head(5).iterrows():
            print(f"     {r['player_name']} ({r['position']}, {r['dataset']})")
    else:
        print(f"  OK: {col}: 0 missing values")

for col in ['pick', 'round']:
    missing = df[col].isna().sum()
    if missing > 0:
        msg = f"{col}: {missing} missing values (should be 0)"
        issues['4. Missing Data']['critical'].append(msg)
        print(f"  ** CRITICAL: {msg}")
    else:
        print(f"  OK: {col}: 0 missing values")

# prospect_profile
pp_missing = df['prospect_profile'].isna().sum()
if pp_missing > 0:
    msg = f"prospect_profile: {pp_missing} missing values"
    issues['4. Missing Data']['warning'].append(msg)
    print(f"  ** WARNING: {msg}")
else:
    print(f"  OK: prospect_profile: 0 missing values")

# WR-specific missing
print(f"\n  --- WR component missing data ---")
for col in ['enhanced_breakout', 'teammate_score', 'early_declare_score']:
    missing = wr[col].isna().sum()
    pct = missing / len(wr) * 100
    if missing > 0:
        msg = f"WR {col}: {missing}/{len(wr)} missing ({pct:.1f}%)"
        sev = 'critical' if pct > 10 else 'warning' if pct > 0 else 'info'
        issues['4. Missing Data'][sev].append(msg)
        print(f"  ** {sev.upper()}: {msg}")
    else:
        print(f"  OK: WR {col}: 0 missing")

# RB-specific missing
print(f"\n  --- RB component missing data ---")
for col in ['production_score', 'speed_score']:
    missing = rb[col].isna().sum()
    pct = missing / len(rb) * 100
    if missing > 0:
        msg = f"RB {col}: {missing}/{len(rb)} missing ({pct:.1f}%)"
        sev = 'critical' if pct > 10 else 'warning' if pct > 0 else 'info'
        issues['4. Missing Data'][sev].append(msg)
        print(f"  ** {sev.upper()}: {msg}")
    else:
        print(f"  OK: RB {col}: 0 missing")

# TE-specific missing
print(f"\n  --- TE component missing data ---")
for col in ['te_breakout_score', 'te_production_score', 'ras_score']:
    missing = te[col].isna().sum()
    pct = missing / len(te) * 100
    if missing > 0:
        msg = f"TE {col}: {missing}/{len(te)} missing ({pct:.1f}%)"
        sev = 'critical' if pct > 10 else 'warning' if pct > 0 else 'info'
        issues['4. Missing Data'][sev].append(msg)
        print(f"  ** {sev.upper()}: {msg}")
    else:
        print(f"  OK: TE {col}: 0 missing")


# ============================================================================
# CHECK 5: DUPLICATE AUDIT
# ============================================================================
print(f"\n\n{'='*100}")
print("CHECK 5: DUPLICATE AUDIT")
print("=" * 100)

# Exact name duplicates
name_dups = df[df.duplicated(subset=['player_name'], keep=False)].sort_values('player_name')
if len(name_dups) > 0:
    dup_names = name_dups['player_name'].unique()
    print(f"\n  Exact name duplicates found: {len(dup_names)} names appearing more than once")
    for name in dup_names:
        rows = name_dups[name_dups['player_name'] == name]
        positions = rows['position'].tolist()
        years = rows['draft_year'].tolist()
        colleges = rows['college'].tolist()
        detail = f"  {name}: "
        for i, (_, r) in enumerate(rows.iterrows()):
            detail += f"[{r['position']}, {r['college']}, {int(r['draft_year'])}, pick {int(r['pick'])}] "
        print(detail)

        # Same name + same college + same year = likely true duplicate (critical)
        if rows.duplicated(subset=['player_name', 'college', 'draft_year'], keep=False).any():
            msg = f"TRUE DUPLICATE: {name} appears multiple times with same college and year"
            issues['5. Duplicates']['critical'].append(msg)
            print(f"     ** CRITICAL: {msg}")
        else:
            msg = f"Same-name, different player: {name}"
            issues['5. Duplicates']['info'].append(msg)
else:
    print(f"  OK: No exact name duplicates found")

# Near-duplicates (same name + same college + same year)
near_dups = df[df.duplicated(subset=['player_name', 'college', 'draft_year'], keep=False)]
if len(near_dups) > 0 and len(near_dups) > len(name_dups[name_dups.duplicated(subset=['player_name', 'college', 'draft_year'], keep=False)]):
    print(f"\n  Additional near-duplicates (same name + college + year): {len(near_dups)}")


# ============================================================================
# CHECK 6: DC SCORE CONSISTENCY
# ============================================================================
print(f"\n\n{'='*100}")
print("CHECK 6: DC SCORE CONSISTENCY")
print("=" * 100)

# 6a: Verify DC formula for random samples
def dc_formula(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

print(f"\n  --- DC formula verification (raw DC, not display score) ---")
# The dc_score in the output is a DISPLAY score (percentile rank), not the raw formula.
# So we can't directly compare. But we CAN check monotonicity.

# 6b: Monotonicity check — within each position, lower pick should give higher dc_score
print(f"\n  --- DC score monotonicity check (lower pick should = higher dc_score) ---")
for pos_name, pos_df in [('WR', wr), ('RB', rb), ('TE', te)]:
    # Get unique picks and their dc_scores
    pick_dc = pos_df[['pick', 'dc_score']].drop_duplicates().sort_values('pick')

    violations = []
    prev_pick = None
    prev_dc = None
    for _, r in pick_dc.iterrows():
        if prev_pick is not None:
            if r['pick'] > prev_pick and r['dc_score'] > prev_dc:
                violations.append((prev_pick, prev_dc, r['pick'], r['dc_score']))
        prev_pick = r['pick']
        prev_dc = r['dc_score']

    if len(violations) > 0:
        msg = f"{pos_name}: {len(violations)} monotonicity violations in dc_score vs pick"
        issues['6. DC Consistency']['critical'].append(msg)
        print(f"  ** CRITICAL: {msg}")
        for v in violations[:5]:
            print(f"     Pick {int(v[0])} (DC={v[1]:.1f}) < Pick {int(v[2])} (DC={v[3]:.1f}) — higher pick has higher DC!")
    else:
        print(f"  OK: {pos_name} DC score is monotonically decreasing with pick ({len(pick_dc)} unique picks)")

# 6c: Verify a few raw DC formula calculations by back-computing
# Since dc_score is percentile-ranked, we verify the RAW formula by checking that
# slap_model_score is consistent (raw DC can be derived from the formula)
print(f"\n  --- DC raw formula spot-check (5 random players per position) ---")
for pos_name, pos_df in [('WR', wr), ('RB', rb), ('TE', te)]:
    sample = pos_df.sample(min(5, len(pos_df)), random_state=42)
    for _, r in sample.iterrows():
        raw_dc = dc_formula(r['pick'])
        print(f"  {pos_name} {r['player_name']} (pick {int(r['pick'])}): raw DC formula = {raw_dc:.2f}, dc_display = {r['dc_score']:.1f}")


# ============================================================================
# CHECK 7: CROSS-POSITION CONTAMINATION
# ============================================================================
print(f"\n\n{'='*100}")
print("CHECK 7: CROSS-POSITION CONTAMINATION")
print("=" * 100)

# WR columns should be NaN for RB and TE
wr_cols = ['enhanced_breakout', 'teammate_score', 'early_declare_score']
for col in wr_cols:
    for pos_name, pos_df in [('RB', rb), ('TE', te)]:
        non_null = pos_df[col].notna().sum()
        if non_null > 0:
            msg = f"{pos_name} has {non_null} non-null {col} values (WR-only column)"
            issues['7. Cross-Position Contamination']['critical'].append(msg)
            print(f"  ** CRITICAL: {msg}")
        else:
            print(f"  OK: {pos_name} {col}: all NaN (correct)")

# RB columns should be NaN for WR and TE
rb_cols = ['production_score', 'speed_score']
for col in rb_cols:
    for pos_name, pos_df in [('WR', wr), ('TE', te)]:
        non_null = pos_df[col].notna().sum()
        if non_null > 0:
            msg = f"{pos_name} has {non_null} non-null {col} values (RB-only column)"
            issues['7. Cross-Position Contamination']['critical'].append(msg)
            print(f"  ** CRITICAL: {msg}")
        else:
            print(f"  OK: {pos_name} {col}: all NaN (correct)")

# TE columns should be NaN for WR and RB
te_cols = ['te_breakout_score', 'te_production_score', 'ras_score']
for col in te_cols:
    for pos_name, pos_df in [('WR', wr), ('RB', rb)]:
        non_null = pos_df[col].notna().sum()
        if non_null > 0:
            msg = f"{pos_name} has {non_null} non-null {col} values (TE-only column)"
            issues['7. Cross-Position Contamination']['critical'].append(msg)
            print(f"  ** CRITICAL: {msg}")
        else:
            print(f"  OK: {pos_name} {col}: all NaN (correct)")


# ============================================================================
# CHECK 8: DATASET LABEL AUDIT
# ============================================================================
print(f"\n\n{'='*100}")
print("CHECK 8: DATASET LABEL AUDIT")
print("=" * 100)

# Check dataset values
valid_datasets = {'backtest', '2026_prospect'}
invalid = df[~df['dataset'].isin(valid_datasets)]
if len(invalid) > 0:
    msg = f"{len(invalid)} rows with invalid dataset values: {invalid['dataset'].unique()}"
    issues['8. Dataset Labels']['critical'].append(msg)
    print(f"  ** CRITICAL: {msg}")
else:
    print(f"  OK: All dataset values are 'backtest' or '2026_prospect'")

# Backtest should be 2015-2025
bt_years = backtest['draft_year'].unique()
bt_bad = backtest[(backtest['draft_year'] < 2015) | (backtest['draft_year'] > 2025)]
if len(bt_bad) > 0:
    msg = f"{len(bt_bad)} backtest rows with draft_year outside 2015-2025"
    issues['8. Dataset Labels']['critical'].append(msg)
    print(f"  ** CRITICAL: {msg}")
    for _, r in bt_bad.head(5).iterrows():
        print(f"     {r['player_name']}: draft_year={int(r['draft_year'])}")
else:
    print(f"  OK: All backtest draft_years in 2015-2025 (actual range: {sorted(bt_years)})")

# Prospects should be 2026
p26_years = prospects['draft_year'].unique()
p26_bad = prospects[prospects['draft_year'] != 2026]
if len(p26_bad) > 0:
    msg = f"{len(p26_bad)} prospect rows with draft_year != 2026"
    issues['8. Dataset Labels']['critical'].append(msg)
    print(f"  ** CRITICAL: {msg}")
else:
    print(f"  OK: All prospect draft_years = 2026 (values: {sorted(p26_years)})")

# Cross-check: any year mismatches?
yr_mismatch = df[((df['dataset'] == 'backtest') & (df['draft_year'] == 2026)) |
                  ((df['dataset'] == '2026_prospect') & (df['draft_year'] != 2026))]
if len(yr_mismatch) > 0:
    msg = f"{len(yr_mismatch)} rows with dataset/draft_year mismatch"
    issues['8. Dataset Labels']['critical'].append(msg)
    print(f"  ** CRITICAL: {msg}")
else:
    print(f"  OK: No dataset/draft_year mismatches")

# Count by position and dataset
print(f"\n  Breakdown:")
for pos in ['WR', 'RB', 'TE']:
    bt_n = len(df[(df['position'] == pos) & (df['dataset'] == 'backtest')])
    p26_n = len(df[(df['position'] == pos) & (df['dataset'] == '2026_prospect')])
    print(f"    {pos}: {bt_n} backtest + {p26_n} prospects = {bt_n + p26_n}")


# ============================================================================
# CHECK 9: NFL OUTCOMES AUDIT (backtest only)
# ============================================================================
print(f"\n\n{'='*100}")
print("CHECK 9: NFL OUTCOMES AUDIT")
print("=" * 100)

nfl_cols = ['nfl_hit24', 'nfl_hit12', 'nfl_first_3yr_ppg', 'nfl_career_ppg',
            'nfl_best_ppr', 'nfl_best_ppg', 'nfl_seasons_10ppg_3yr']

# Check hit columns are binary (0 or 1)
for col in ['nfl_hit24', 'nfl_hit12']:
    bt_vals = backtest[col].dropna()
    bad = bt_vals[~bt_vals.isin([0, 1, 0.0, 1.0])]
    if len(bad) > 0:
        msg = f"Backtest {col}: {len(bad)} values not in {{0, 1}}"
        issues['9. NFL Outcomes']['critical'].append(msg)
        print(f"  ** CRITICAL: {msg}")
        print(f"     Unique non-binary values: {sorted(bad.unique())}")
    else:
        ones = (bt_vals == 1).sum()
        zeros = (bt_vals == 0).sum()
        missing = backtest[col].isna().sum()
        print(f"  OK: Backtest {col}: {ones} hits, {zeros} misses, {missing} missing (all binary)")

# Check PPG columns >= 0
for col in ['nfl_first_3yr_ppg', 'nfl_career_ppg']:
    bt_vals = backtest[col].dropna()
    negatives = bt_vals[bt_vals < 0]
    if len(negatives) > 0:
        msg = f"Backtest {col}: {len(negatives)} negative values"
        issues['9. NFL Outcomes']['critical'].append(msg)
        print(f"  ** CRITICAL: {msg}")
    else:
        missing = backtest[col].isna().sum()
        print(f"  OK: Backtest {col}: range {bt_vals.min():.2f}-{bt_vals.max():.2f}, {missing} missing (all non-negative)")

# Check that prospects have no NFL outcome data
print(f"\n  --- 2026 prospects should have empty NFL outcomes ---")
for col in nfl_cols:
    p26_non_null = prospects[col].notna().sum()
    if p26_non_null > 0:
        msg = f"2026 prospects have {p26_non_null} non-null {col} values"
        issues['9. NFL Outcomes']['critical'].append(msg)
        print(f"  ** CRITICAL: {msg}")
    else:
        print(f"  OK: 2026 prospects {col}: all NaN")

# Check backtest players missing ALL outcome columns
all_missing_mask = True
for col in nfl_cols:
    all_missing_mask = all_missing_mask & backtest[col].isna()
bt_all_missing = backtest[all_missing_mask]
if len(bt_all_missing) > 0:
    msg = f"{len(bt_all_missing)} backtest players missing ALL NFL outcome columns"
    issues['9. NFL Outcomes']['warning'].append(msg)
    print(f"\n  ** WARNING: {msg}")
    for _, r in bt_all_missing.iterrows():
        print(f"     {r['player_name']} ({r['position']}, {r['college']}, {int(r['draft_year'])})")
else:
    print(f"\n  OK: All backtest players have at least one NFL outcome column populated")

# Additional: check missing rates per position
print(f"\n  --- Backtest NFL outcome missing rates by position ---")
for pos in ['WR', 'RB', 'TE']:
    pos_bt = backtest[backtest['position'] == pos]
    print(f"  {pos} (n={len(pos_bt)}):")
    for col in nfl_cols:
        missing = pos_bt[col].isna().sum()
        pct = missing / len(pos_bt) * 100
        status = "OK" if pct < 5 else "NOTE"
        print(f"    {status}: {col}: {missing}/{len(pos_bt)} missing ({pct:.1f}%)")


# ============================================================================
# CHECK 10: RANKING CONSISTENCY
# ============================================================================
print(f"\n\n{'='*100}")
print("CHECK 10: RANKING CONSISTENCY (slap_model_score vs slap_display_score)")
print("=" * 100)

for pos_name in ['WR', 'RB', 'TE']:
    for ds in ['backtest', '2026_prospect']:
        subset = df[(df['position'] == pos_name) & (df['dataset'] == ds)]
        if len(subset) < 3:
            print(f"  {pos_name} {ds}: too few rows ({len(subset)}) to check")
            continue

        model_rank = subset['slap_model_score'].rank(ascending=False, method='min')
        display_rank = subset['slap_display_score'].rank(ascending=False, method='min')

        # Spearman correlation
        corr, pval = spearmanr(model_rank, display_rank)

        # Count rank mismatches
        rank_diff = (model_rank.values - display_rank.values)
        mismatches = (rank_diff != 0).sum()
        max_diff = np.abs(rank_diff).max()

        if corr < 0.9999:
            msg = f"{pos_name} {ds}: Spearman r={corr:.6f} (expected 1.000000), {mismatches} rank mismatches, max diff={max_diff}"
            sev = 'critical' if corr < 0.999 else 'warning'
            issues['10. Ranking Consistency'][sev].append(msg)
            print(f"  ** {sev.upper()}: {msg}")

            # Show rank differences
            if mismatches > 0:
                subset_copy = subset.copy()
                subset_copy['model_rank'] = model_rank.values
                subset_copy['display_rank'] = display_rank.values
                subset_copy['rank_diff'] = rank_diff
                mismatched = subset_copy[subset_copy['rank_diff'] != 0].sort_values('rank_diff', key=abs, ascending=False)
                for _, r in mismatched.head(5).iterrows():
                    print(f"     {r['player_name']}: model_rank={int(r['model_rank'])}, display_rank={int(r['display_rank'])}, diff={int(r['rank_diff'])}")
        else:
            print(f"  OK: {pos_name} {ds}: Spearman r={corr:.6f}, {mismatches} rank mismatches (n={len(subset)})")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n\n{'='*100}")
print("FINAL SUMMARY")
print("=" * 100)

total_critical = 0
total_warning = 0
total_info = 0

for category, severity_dict in issues.items():
    n_crit = len(severity_dict['critical'])
    n_warn = len(severity_dict['warning'])
    n_info = len(severity_dict['info'])
    total_critical += n_crit
    total_warning += n_warn
    total_info += n_info

    status = "PASS" if n_crit == 0 and n_warn == 0 else "FAIL" if n_crit > 0 else "WARN"
    icon = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
    print(f"\n  {icon} {category}: {n_crit} critical, {n_warn} warning, {n_info} info")

    for msg in severity_dict['critical']:
        print(f"       CRITICAL: {msg}")
    for msg in severity_dict['warning']:
        print(f"       WARNING:  {msg}")
    for msg in severity_dict['info']:
        print(f"       INFO:     {msg}")

print(f"\n{'='*100}")
print(f"TOTALS: {total_critical} CRITICAL | {total_warning} WARNING | {total_info} INFO")
if total_critical == 0:
    print("STATUS: ALL CRITICAL CHECKS PASSED")
elif total_critical <= 3:
    print("STATUS: MINOR ISSUES FOUND — review critical items above")
else:
    print("STATUS: SIGNIFICANT ISSUES FOUND — review all critical items above")
print(f"{'='*100}")
