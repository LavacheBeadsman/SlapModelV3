"""
Audit and fix early declare classifications for all 2026 prospects.

RULE: Early declare = 3 or fewer college seasons played.
- Uses CFBD seasons_found (WR) as primary data
- Uses birthdate-based estimation to catch transfers (CFBD undercounts at current school)
- college_seasons = max(seasons_found, estimated_from_age) to handle both cases
"""

import pandas as pd
import numpy as np

pd.set_option('display.width', 200)

print("=" * 120)
print("EARLY DECLARE AUDIT — ALL 2026 PROSPECTS")
print("=" * 120)

# Load all data sources
wr26_pre = pd.read_csv('output/slap_v5_wr_2026.csv')
wr26_bo = pd.read_csv('data/wr_breakout_ages_2026.csv')
prospects = pd.read_csv('data/prospects_final.csv')
te26 = pd.read_csv('data/te_2026_prospects_final.csv')


def estimate_seasons_from_birthdate(birthdate, draft_year=2026):
    """Estimate college seasons from birthdate (enrollment at age 18)."""
    if pd.isna(birthdate) or str(birthdate).strip() in ('MISSING', '', 'nan'):
        return np.nan
    try:
        birth_year = pd.to_datetime(birthdate).year
        return max(1, draft_year - (birth_year + 18))
    except:
        return np.nan


def classify_seasons(college_seasons):
    """Classify based on total college seasons."""
    if pd.isna(college_seasons):
        return 'UNKNOWN'
    if college_seasons <= 3:
        return 'EARLY'
    elif college_seasons == 4:
        return 'STANDARD'
    else:
        return 'LATE'


# ============================================================================
# WR 2026 PROSPECTS
# ============================================================================
print(f"\n{'='*120}")
print("WR 2026 PROSPECTS")
print("=" * 120)

# Merge WR data
wr_prosp = prospects[prospects['position'] == 'WR'].copy()
wr = wr26_pre.merge(
    wr26_bo[['player_name', 'seasons_found']],
    on='player_name', how='left')
wr = wr.merge(
    wr_prosp[['player_name', 'birthdate', 'age', 'age_estimated']],
    on='player_name', how='left')

# Calculate estimated seasons from age
wr['est_from_age'] = wr['birthdate'].apply(estimate_seasons_from_birthdate)

# College seasons = max of CFBD data and age estimate
# This catches transfers (CFBD undercounts) and trusts CFBD when it shows 4+
wr['college_seasons'] = wr.apply(
    lambda r: max(
        r['seasons_found'] if pd.notna(r['seasons_found']) else 0,
        r['est_from_age'] if pd.notna(r['est_from_age']) else 0
    ) if (pd.notna(r['seasons_found']) or pd.notna(r['est_from_age'])) else np.nan,
    axis=1)

wr['new_class'] = wr['college_seasons'].apply(classify_seasons)
wr['new_ed'] = wr['new_class'].apply(lambda x: 100 if x == 'EARLY' else 0)
wr['old_ed'] = wr['early_declare_score']
wr['changed'] = wr['new_ed'] != wr['old_ed']

# Flag transfers (CFBD seasons much lower than age estimate)
wr['transfer_flag'] = wr.apply(
    lambda r: 'TRANSFER?' if (pd.notna(r['est_from_age']) and pd.notna(r['seasons_found']) and
              r['est_from_age'] - r['seasons_found'] >= 2) else '',
    axis=1)

# Print full table
print(f"\n{'Player':<28} {'College':<20} {'SF':>3} {'AgeEst':>6} {'Total':>5} {'Class':<8} {'Old':>4} {'New':>4} {'Chg':>4} {'Flag':<10}")
print("-" * 120)

changes_wr = []
for _, r in wr.sort_values('pick').iterrows():
    sf = f"{int(r['seasons_found'])}" if pd.notna(r['seasons_found']) else '?'
    ae = f"{int(r['est_from_age'])}" if pd.notna(r['est_from_age']) else '?'
    cs = f"{int(r['college_seasons'])}" if pd.notna(r['college_seasons']) else '?'
    chg = '***' if r['changed'] else ''
    if r['changed']:
        changes_wr.append(r['player_name'])
    print(f"{r['player_name']:<28} {r['college']:<20} {sf:>3} {ae:>6} {cs:>5} {r['new_class']:<8} {int(r['old_ed']):>4} {int(r['new_ed']):>4} {chg:>4} {r['transfer_flag']:<10}")

print(f"\n  WR Summary:")
print(f"    Total: {len(wr)}")
print(f"    EARLY: {(wr['new_class'] == 'EARLY').sum()}")
print(f"    STANDARD: {(wr['new_class'] == 'STANDARD').sum()}")
print(f"    LATE: {(wr['new_class'] == 'LATE').sum()}")
print(f"    UNKNOWN: {(wr['new_class'] == 'UNKNOWN').sum()}")
print(f"    Changed from old: {len(changes_wr)}")
print(f"    Potential transfers flagged: {(wr['transfer_flag'] != '').sum()}")

# ============================================================================
# RB 2026 PROSPECTS (no CFBD multi-season data, estimate from age)
# ============================================================================
print(f"\n\n{'='*120}")
print("RB 2026 PROSPECTS (estimated from birthdate — no CFBD multi-season data)")
print("=" * 120)

rb_prosp = prospects[prospects['position'] == 'RB'].copy()
rb_prosp['est_from_age'] = rb_prosp['birthdate'].apply(estimate_seasons_from_birthdate)
rb_prosp['college_seasons'] = rb_prosp['est_from_age']  # No CFBD fallback for RBs
rb_prosp['new_class'] = rb_prosp['college_seasons'].apply(classify_seasons)
rb_prosp['new_ed'] = rb_prosp['new_class'].apply(lambda x: 100 if x == 'EARLY' else 0)

print(f"\n{'Player':<28} {'School':<18} {'Pick':>5} {'Bday':>12} {'AgeEst':>6} {'Class':<8} {'ED':>4} {'Note':<15}")
print("-" * 110)
for _, r in rb_prosp.sort_values('projected_pick').iterrows():
    bd = str(r.get('birthdate', ''))[:10]
    ae = f"{int(r['est_from_age'])}" if pd.notna(r['est_from_age']) else '?'
    note = 'EST-AGE' if str(r.get('age_estimated', '')).upper() in ('TRUE', 'True') else ''
    if bd == 'MISSING':
        note = 'NO-BDAY'
    print(f"{r['player_name']:<28} {r['school']:<18} {int(r['projected_pick']):>5} {bd:>12} {ae:>6} {r['new_class']:<8} {int(r['new_ed']):>4} {note:<15}")

print(f"\n  RB Summary (RB model has 0% early declare weight — informational only):")
print(f"    Total: {len(rb_prosp)}")
print(f"    EARLY: {(rb_prosp['new_class'] == 'EARLY').sum()}")
print(f"    STANDARD: {(rb_prosp['new_class'] == 'STANDARD').sum()}")
print(f"    LATE: {(rb_prosp['new_class'] == 'LATE').sum()}")
print(f"    UNKNOWN: {(rb_prosp['new_class'] == 'UNKNOWN').sum()}")

# ============================================================================
# TE 2026 PROSPECTS (have early_declare already, verify against age)
# ============================================================================
print(f"\n\n{'='*120}")
print("TE 2026 PROSPECTS (verify existing early_declare against birthdate)")
print("=" * 120)

te26['est_from_age'] = te26['birthdate'].apply(estimate_seasons_from_birthdate)
te26['college_seasons'] = te26['est_from_age']  # Use age estimate (no CFBD seasons_found for TEs)
te26['new_class'] = te26['college_seasons'].apply(classify_seasons)
te26['new_ed'] = te26['new_class'].apply(lambda x: 100 if x == 'EARLY' else 0)
te26['old_ed'] = te26['early_declare']
te26['changed_te'] = te26['new_ed'] != te26['old_ed']

print(f"\n{'Player':<28} {'College':<18} {'DraftAge':>8} {'AgeEst':>6} {'Class':<8} {'Old':>4} {'New':>4} {'Chg':>4} {'BdaySrc':<10}")
print("-" * 110)
changes_te = []
for _, r in te26.sort_values('projected_pick').iterrows():
    da = f"{r['draft_age']:.1f}" if pd.notna(r.get('draft_age')) else '?'
    ae = f"{int(r['est_from_age'])}" if pd.notna(r['est_from_age']) else '?'
    chg = '***' if r['changed_te'] else ''
    if r['changed_te']:
        changes_te.append(r['player_name'])
    print(f"{r['player_name']:<28} {r['college']:<18} {da:>8} {ae:>6} {r['new_class']:<8} {int(r['old_ed']):>4} {int(r['new_ed']):>4} {chg:>4} {str(r.get('birthdate_source', '')):<10}")

print(f"\n  TE Summary (TE model has 0% early declare weight — informational only):")
print(f"    Total: {len(te26)}")
print(f"    EARLY: {(te26['new_class'] == 'EARLY').sum()}")
print(f"    STANDARD: {(te26['new_class'] == 'STANDARD').sum()}")
print(f"    LATE: {(te26['new_class'] == 'LATE').sum()}")
print(f"    UNKNOWN: {(te26['new_class'] == 'UNKNOWN').sum()}")
print(f"    Changed from old: {len(changes_te)}")

# ============================================================================
# IMPACT ANALYSIS (WR only — only position where early declare affects scores)
# ============================================================================
print(f"\n\n{'='*120}")
print("IMPACT ON WR SCORES (5% weight × 100-point change = ±5 points per change)")
print("=" * 120)

gained = wr[(wr['changed']) & (wr['new_ed'] > wr['old_ed'])]
lost = wr[(wr['changed']) & (wr['new_ed'] < wr['old_ed'])]

print(f"\n  Players GAINING early declare credit (+5 model points): {len(gained)}")
for _, r in gained.sort_values('pick').iterrows():
    sf = int(r['seasons_found']) if pd.notna(r['seasons_found']) else '?'
    print(f"    {r['player_name']:<28} pick {int(r['pick']):>3}  SF={sf}  age={r.get('age', '?')}")

print(f"\n  Players LOSING early declare credit (-5 model points): {len(lost)}")
for _, r in lost.sort_values('pick').iterrows():
    sf = int(r['seasons_found']) if pd.notna(r['seasons_found']) else '?'
    flag = r['transfer_flag']
    print(f"    {r['player_name']:<28} pick {int(r['pick']):>3}  SF={sf}  age={r.get('age', '?')}  {flag}")

# ============================================================================
# BIRTHDATE ISSUES SUMMARY
# ============================================================================
print(f"\n\n{'='*120}")
print("BIRTHDATE STATUS SUMMARY")
print("=" * 120)

# WR/RB from prospects_final
for pos in ['WR', 'RB']:
    pos_df = prospects[prospects['position'] == pos]
    missing = pos_df[pos_df['birthdate'].astype(str).str.strip() == 'MISSING']
    estimated = pos_df[pos_df['age_estimated'].astype(str).str.upper().isin(['TRUE', 'True'])]
    confirmed = len(pos_df) - len(missing) - len(estimated)
    print(f"\n  {pos}: {len(pos_df)} total | {confirmed} confirmed | {len(estimated)} estimated | {len(missing)} missing")
    if len(missing) > 0:
        top_missing = missing.sort_values('projected_pick').head(10)
        print(f"    Missing (top by pick):")
        for _, r in top_missing.iterrows():
            print(f"      {r['player_name']:<28} pick {int(r['projected_pick']):>3}")

# TE
te_ver = te26[te26['birthdate_source'] == 'verified']
te_est = te26[te26['birthdate_source'] == 'estimated']
print(f"\n  TE: {len(te26)} total | {len(te_ver)} verified | {len(te_est)} estimated | 0 missing")
if len(te_est) > 0:
    top_est = te_est.sort_values('projected_pick').head(10)
    print(f"    Estimated (top by pick):")
    for _, r in top_est.iterrows():
        print(f"      {r['player_name']:<28} pick {int(r['projected_pick']):>3}")

print(f"\n\nDone.")
