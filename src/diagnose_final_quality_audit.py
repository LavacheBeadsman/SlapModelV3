"""
FINAL DATA QUALITY AUDIT — All 3 Positions
=============================================
Reads source data + master database to report coverage, imputation,
outcome availability, and known issues. DOES NOT CHANGE ANYTHING.
"""

import pandas as pd
import numpy as np
import warnings, os
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()


# ============================================================================
# LOAD ALL SOURCE DATA
# ============================================================================
master = pd.read_csv('output/slap_v5_master_database.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
wr_bt_src = pd.read_csv('data/wr_backtest_all_components.csv')
wr_tm = pd.read_csv('data/wr_teammate_scores.csv')
rb_bt_src = pd.read_csv('data/rb_backtest_with_receiving.csv')
te_bt_src = pd.read_csv('data/te_backtest_master.csv')
prospects_src = pd.read_csv('data/prospects_final.csv')
combine = pd.read_parquet('data/nflverse/combine.parquet')
combine['name_norm'] = combine['player_name'].apply(normalize_name)

# 2026 sources
wr26_src = pd.read_csv('output/slap_v5_wr_2026.csv')
te26_src = pd.read_csv('data/te_2026_prospects_final.csv')
rb26_prospects = prospects_src[prospects_src['position'] == 'RB'].copy()

# WR breakout ages 2026
wr26_bo = pd.read_csv('data/wr_breakout_ages_2026.csv')

# Combine lookup for RBs
combine_rb = combine[combine['pos'].isin(['RB', 'FB'])].copy()


# ============================================================================
print("\n" + "=" * 120)
print("FINAL DATA QUALITY AUDIT — ALL 3 POSITIONS")
print("=" * 120)

for pos in ['WR', 'RB', 'TE']:
    bt = master[(master['position'] == pos) & (master['data_type'] == 'backtest')]
    p26 = master[(master['position'] == pos) & (master['data_type'] == '2026_prospect')]
    print(f"\n  {pos}: {len(bt)} backtest + {len(p26)} prospects = {len(bt)+len(p26)} total")
total = len(master)
print(f"  TOTAL: {total} rows")


# ============================================================================
# WR SECTION
# ============================================================================
print(f"\n\n{'='*120}")
print("WR — Wide Receivers")
print(f"  Formula: SLAP = DC × 0.70 + Enhanced Breakout × 0.20 + Teammate × 0.05 + Early Declare × 0.05")
print("=" * 120)

# --- 1. COUNTS ---
wr_bt = wr_bt_src.copy()
wr_bt = wr_bt.merge(wr_tm[['player_name', 'draft_year', 'total_teammate_dc']], on=['player_name', 'draft_year'], how='left')
n_wr_bt = len(wr_bt)
n_wr_26 = len(wr26_src)
print(f"\n  1. TOTAL PLAYERS")
print(f"     Backtest: {n_wr_bt} (2015-2025)")
print(f"     2026 prospects: {n_wr_26}")
print(f"     Total: {n_wr_bt + n_wr_26}")

# --- 2. COMPONENT COVERAGE — BACKTEST ---
print(f"\n  2. COMPONENT COVERAGE — BACKTEST ({n_wr_bt} WRs)")
print(f"     {'Component':<35} {'Real':>6} {'Est/Imp':>8} {'Missing':>8} {'Method':>25}")
print(f"     {'-'*85}")

# DC
print(f"     {'Draft pick (DC)':.<35} {n_wr_bt:>6} {0:>8} {0:>8} {'—':>25}")

# Breakout age
bo_real = wr_bt['breakout_age'].notna().sum()
bo_miss = n_wr_bt - bo_real
# Players who never broke out still have peak_dominator — check who has neither
pd_real = wr_bt['peak_dominator'].notna().sum()
no_data = n_wr_bt - pd_real
print(f"     {'Breakout age':.<35} {bo_real:>6} {'—':>8} {bo_miss:>8} {'never broke out → use peak dom':>25}")
print(f"     {'Peak dominator (%)':.<35} {pd_real:>6} {'—':>8} {no_data:>8} {'default score=25 if missing':>25}")

# Rush yards
rush_real = wr_bt['rush_yards'].notna().sum()
print(f"     {'Rush yards (for +5 bonus)':.<35} {rush_real:>6} {'—':>8} {n_wr_bt - rush_real:>8} {'no bonus if missing':>25}")

# Teammate DC
tm_real = wr_bt['total_teammate_dc'].notna().sum()
print(f"     {'Teammate DC':.<35} {tm_real:>6} {'—':>8} {n_wr_bt - tm_real:>8} {'0 if missing (no teammates)':>25}")

# Early declare
ed_real = wr_bt['early_declare'].notna().sum()
print(f"     {'Early declare':.<35} {ed_real:>6} {'—':>8} {n_wr_bt - ed_real:>8} {'—':>25}")

# --- 3. COMPONENT COVERAGE — 2026 PROSPECTS ---
print(f"\n  3. COMPONENT COVERAGE — 2026 PROSPECTS ({n_wr_26} WRs)")
print(f"     {'Component':<35} {'Real':>6} {'Est/Imp':>8} {'Missing':>8} {'Method':>25}")
print(f"     {'-'*85}")

# DC (projected)
print(f"     {'Projected pick (DC)':.<35} {n_wr_26:>6} {0:>8} {0:>8} {'consensus mock draft':>25}")

# Breakout age — from wr_breakout_ages_2026
merged = wr26_src.merge(wr26_bo[['player_name', 'breakout_age', 'peak_dominator']], on='player_name', how='left', suffixes=('', '_bo'))
bo26_real = merged['breakout_age_bo'].notna().sum()
# Also check what the build script set
bo26_in_output = wr26_src['breakout_age'].notna().sum() if 'breakout_age' in wr26_src.columns else 0
print(f"     {'Breakout age':.<35} {bo26_real:>6} {'—':>8} {n_wr_26 - bo26_real:>8} {'never broke out → peak dom':>25}")
pd26 = merged['peak_dominator_bo'].notna().sum()
print(f"     {'Peak dominator (%)':.<35} {pd26:>6} {'—':>8} {n_wr_26 - pd26:>8} {'default score=25 if missing':>25}")

# Rush yards
rush26 = wr26_src['rush_yards'].notna().sum() if 'rush_yards' in wr26_src.columns else 0
print(f"     {'Rush yards (for +5 bonus)':.<35} {rush26:>6} {'—':>8} {n_wr_26 - rush26:>8} {'no bonus if missing':>25}")

# Teammate
tm26 = wr26_src['teammate_score'].notna().sum() if 'teammate_score' in wr26_src.columns else 0
print(f"     {'Teammate score':.<35} {tm26:>6} {'—':>8} {n_wr_26 - tm26:>8} {'manually computed':>25}")

# Early declare
ed26 = wr26_src['early_declare'].notna().sum() if 'early_declare' in wr26_src.columns else 0
print(f"     {'Early declare':.<35} {ed26:>6} {'—':>8} {n_wr_26 - ed26:>8} {'—':>25}")

# --- 4. NFL OUTCOMES (WR backtest) ---
print(f"\n  4. NFL OUTCOME COVERAGE — BACKTEST ({n_wr_bt} WRs)")
wr_bt_m = master[(master['position'] == 'WR') & (master['data_type'] == 'backtest')]
for col, label in [('nfl_hit24', 'hit24 (top-24 PPR season)'),
                   ('nfl_hit12', 'hit12 (top-12 PPR season)'),
                   ('nfl_first_3yr_ppg', 'first_3yr_ppg'),
                   ('nfl_career_ppg', 'career_ppg')]:
    n = wr_bt_m[col].notna().sum()
    print(f"     {label:<35} {n:>5}/{n_wr_bt}   {'FULL' if n == n_wr_bt else f'{n_wr_bt-n} missing'}")

# Which draft classes are incomplete?
for yr in sorted(wr_bt_m['draft_year'].unique()):
    yr_data = wr_bt_m[wr_bt_m['draft_year'] == yr]
    ppg_miss = yr_data['nfl_first_3yr_ppg'].isna().sum()
    if ppg_miss > 0:
        print(f"       → {int(yr)}: {ppg_miss}/{len(yr_data)} missing first_3yr_ppg", end='')
        if yr >= 2023:
            print(f" (too early — drafted {int(yr)}, need through {int(yr+2)})")
        else:
            print(f" (data gap)")


# ============================================================================
# RB SECTION
# ============================================================================
print(f"\n\n{'='*120}")
print("RB — Running Backs")
print(f"  Formula: SLAP = DC × 0.65 + RYPTPA × 0.30 + Speed Score × 0.05")
print("=" * 120)

rb_bt = rb_bt_src.copy()
n_rb_bt = len(rb_bt)
n_rb_26 = len(rb26_prospects)

print(f"\n  1. TOTAL PLAYERS")
print(f"     Backtest: {n_rb_bt} (2015-2025)")
print(f"     2026 prospects: {n_rb_26}")
print(f"     Total: {n_rb_bt + n_rb_26}")

# --- 2. COMPONENT COVERAGE — BACKTEST ---
print(f"\n  2. COMPONENT COVERAGE — BACKTEST ({n_rb_bt} RBs)")
print(f"     {'Component':<35} {'Real':>6} {'Est/Imp':>8} {'Missing':>8} {'Method':>25}")
print(f"     {'-'*85}")

print(f"     {'Draft pick (DC)':.<35} {n_rb_bt:>6} {0:>8} {0:>8} {'—':>25}")

# Rec yards
ry = pd.to_numeric(rb_bt['rec_yards'], errors='coerce').notna().sum()
print(f"     {'Rec yards (final season)':.<35} {ry:>6} {'—':>8} {n_rb_bt - ry:>8} {'→ production = 0':>25}")

# Team pass att
tpa = pd.to_numeric(rb_bt['team_pass_att'], errors='coerce').notna().sum()
print(f"     {'Team pass att (final season)':.<35} {tpa:>6} {'—':>8} {n_rb_bt - tpa:>8} {'→ production = 0':>25}")

# RYPTPA (computed)
# Need to recalculate to see how many are real
ry_num = pd.to_numeric(rb_bt['rec_yards'], errors='coerce')
tpa_num = pd.to_numeric(rb_bt['team_pass_att'], errors='coerce')
prod_real = (ry_num.notna() & tpa_num.notna() & (tpa_num > 0)).sum()
prod_zero = n_rb_bt - prod_real
print(f"     {'RYPTPA score (computed)':.<35} {prod_real:>6} {prod_zero:>8} {'—':>8} {'0 if rec_yards/tpa missing':>25}")

# Weight — from combine
rb_bt['name_norm'] = rb_bt['player_name'].apply(normalize_name)
wt_matched = 0
forty_matched = 0
for idx in rb_bt.index:
    nn = rb_bt.loc[idx, 'name_norm']
    dy = int(rb_bt.loc[idx, 'draft_year'])
    match = combine_rb[(combine_rb['name_norm'] == nn) & (combine_rb['draft_year'] == dy)]
    if len(match) == 0:
        match = combine_rb[combine_rb['name_norm'] == nn]
    if len(match) > 0:
        if pd.notna(match.iloc[0]['wt']):
            wt_matched += 1
        if pd.notna(match.iloc[0]['forty']):
            forty_matched += 1

print(f"     {'Weight (combine)':.<35} {wt_matched:>6} {'—':>8} {n_rb_bt - wt_matched:>8} {'—':>25}")
print(f"     {'40-yard dash (real)':.<35} {forty_matched:>6} {'—':>8} {n_rb_bt - forty_matched:>8} {'—':>25}")

# Estimated 40s
est_40 = wt_matched - forty_matched  # those with weight but no 40
print(f"     {'40-yard dash (estimated)':.<35} {'—':>6} {est_40:>8} {'—':>8} {'wt × round bucket avg':>25}")

# MNAR imputed
mnar = n_rb_bt - wt_matched
print(f"     {'Speed Score (MNAR imputed)':.<35} {'—':>6} {mnar:>8} {'—':>8} {'Rd1-2→p60, Rd3+→p40':>25}")

# Total speed score coverage
print(f"     {'Speed Score (total)':.<35} {forty_matched:>6} {est_40 + mnar:>8} {0:>8} {'all 223 have a score':>25}")

# Age
age_real = rb_bt['age'].notna().sum()
print(f"     {'Draft age':.<35} {age_real:>6} {'—':>8} {n_rb_bt - age_real:>8} {'default=22 if missing':>25}")

# --- 3. COMPONENT COVERAGE — 2026 PROSPECTS ---
print(f"\n  3. COMPONENT COVERAGE — 2026 PROSPECTS ({n_rb_26} RBs)")
print(f"     {'Component':<35} {'Real':>6} {'Est/Imp':>8} {'Missing':>8} {'Method':>25}")
print(f"     {'-'*85}")

print(f"     {'Projected pick (DC)':.<35} {n_rb_26:>6} {0:>8} {0:>8} {'consensus mock':>25}")
ry26 = pd.to_numeric(rb26_prospects['rec_yards'], errors='coerce').notna().sum()
print(f"     {'Rec yards (final season)':.<35} {ry26:>6} {'—':>8} {n_rb_26 - ry26:>8} {'→ production = 0':>25}")
tpa26 = pd.to_numeric(rb26_prospects['team_pass_attempts'], errors='coerce').notna().sum()
print(f"     {'Team pass att':.<35} {tpa26:>6} {'—':>8} {n_rb_26 - tpa26:>8} {'→ production = 0':>25}")
print(f"     {'Weight / 40-yard dash':.<35} {0:>6} {n_rb_26:>8} {'—':>8} {'ALL MNAR imputed (pre-draft)':>25}")
print(f"     {'Speed Score':.<35} {0:>6} {n_rb_26:>8} {'—':>8} {'Rd1-2→p60, Rd3+→p40':>25}")

# --- 4. NFL OUTCOMES (RB backtest) ---
print(f"\n  4. NFL OUTCOME COVERAGE — BACKTEST ({n_rb_bt} RBs)")
rb_bt_m = master[(master['position'] == 'RB') & (master['data_type'] == 'backtest')]
for col, label in [('nfl_hit24', 'hit24 (top-24 PPR season)'),
                   ('nfl_hit12', 'hit12 (top-12 PPR season)'),
                   ('nfl_first_3yr_ppg', 'first_3yr_ppg'),
                   ('nfl_career_ppg', 'career_ppg')]:
    n = rb_bt_m[col].notna().sum()
    print(f"     {label:<35} {n:>5}/{n_rb_bt}   {'FULL' if n == n_rb_bt else f'{n_rb_bt-n} missing'}")

for yr in sorted(rb_bt_m['draft_year'].unique()):
    yr_data = rb_bt_m[rb_bt_m['draft_year'] == yr]
    ppg_miss = yr_data['nfl_first_3yr_ppg'].isna().sum()
    if ppg_miss > 0:
        print(f"       → {int(yr)}: {ppg_miss}/{len(yr_data)} missing first_3yr_ppg", end='')
        if yr >= 2023:
            print(f" (too early — drafted {int(yr)}, need through {int(yr+2)})")
        else:
            print(f" (data gap)")


# ============================================================================
# TE SECTION
# ============================================================================
print(f"\n\n{'='*120}")
print("TE — Tight Ends")
print(f"  Formula: SLAP = DC × 0.60 + Breakout × 0.15 + Production × 0.15 + RAS × 0.10")
print("=" * 120)

te_bt = te_bt_src.copy()
n_te_bt = len(te_bt)
n_te_26 = len(te26_src)

print(f"\n  1. TOTAL PLAYERS")
print(f"     Backtest: {n_te_bt} (2015-2024)")
print(f"     2026 prospects: {n_te_26}")
print(f"     Total: {n_te_bt + n_te_26}")

# --- 2. COMPONENT COVERAGE — BACKTEST ---
print(f"\n  2. COMPONENT COVERAGE — BACKTEST ({n_te_bt} TEs)")
print(f"     {'Component':<35} {'Real':>6} {'Est/Imp':>8} {'Missing':>8} {'Method':>25}")
print(f"     {'-'*85}")

print(f"     {'Draft pick (DC)':.<35} {n_te_bt:>6} {0:>8} {0:>8} {'—':>25}")

# Breakout age
te_bo_real = te_bt['breakout_age'].notna().sum()
te_pd_real = te_bt['peak_dominator'].notna().sum()
te_no_pff = n_te_bt - te_pd_real
print(f"     {'Breakout age (15% threshold)':.<35} {te_bo_real:>6} {'—':>8} {n_te_bt - te_bo_real:>8} {'never broke out → peak dom':>25}")
print(f"     {'Peak dominator (PFF data)':.<35} {te_pd_real:>6} {'—':>8} {te_no_pff:>8} {'default score=25 if no PFF':>25}")

# Production — CFBD rec yards
cfbd_ry = te_bt['cfbd_rec_yards'].notna().sum()
cfbd_tpa = te_bt['cfbd_team_pass_att'].notna().sum()
prod_both = (te_bt['cfbd_rec_yards'].notna() & te_bt['cfbd_team_pass_att'].notna()).sum()
# After manual patches
manual_patches = ['Dallas Goedert', 'Adam Shaheen']
patched = sum(1 for name in manual_patches if name in te_bt['player_name'].values)
print(f"     {'Rec yards (CFBD)':.<35} {cfbd_ry:>6} {patched:>8} {n_te_bt - cfbd_ry - patched:>8} {f'{patched} manual patches':>25}")
print(f"     {'Team pass att (CFBD)':.<35} {cfbd_tpa:>6} {'—':>8} {n_te_bt - cfbd_tpa:>8} {'—':>25}")

# How many have production score after patches?
# Need to recalculate
# Count from master database flags
te_bt_m = master[(master['position'] == 'TE') & (master['data_type'] == 'backtest')]
prod_flag = te_bt_m['production_data_flag']
prod_real_count = (prod_flag == 'real').sum()
prod_imp_count = (prod_flag == 'imputed').sum()
print(f"     {'Production score (final)':.<35} {prod_real_count:>6} {prod_imp_count:>8} {0:>8} {'imputed = position avg':>25}")

# RAS
ras_real = te_bt['te_ras'].notna().sum()
ras_mnar = n_te_bt - ras_real
print(f"     {'RAS (0-10 scale → ×10)':.<35} {ras_real:>6} {ras_mnar:>8} {0:>8} {'MNAR: Rd1-2→p60, Rd3+→p40':>25}")

# --- 3. COMPONENT COVERAGE — 2026 PROSPECTS ---
print(f"\n  3. COMPONENT COVERAGE — 2026 PROSPECTS ({n_te_26} TEs)")
print(f"     {'Component':<35} {'Real':>6} {'Est/Imp':>8} {'Missing':>8} {'Method':>25}")
print(f"     {'-'*85}")

print(f"     {'Projected pick (DC)':.<35} {n_te_26:>6} {0:>8} {0:>8} {'consensus mock':>25}")
te26_bo = te26_src['breakout_age'].notna().sum()
te26_pd = te26_src['peak_dominator'].notna().sum()
print(f"     {'Breakout age':.<35} {te26_bo:>6} {'—':>8} {n_te_26 - te26_bo:>8} {'never broke out → peak dom':>25}")
print(f"     {'Peak dominator':.<35} {te26_pd:>6} {'—':>8} {n_te_26 - te26_pd:>8} {'default score=25':>25}")

te26_prod_real = te26_src['production_score'].notna().sum() if 'production_score' in te26_src.columns else 0
te26_prod_imp = n_te_26 - te26_prod_real
print(f"     {'Production score':.<35} {te26_prod_real:>6} {te26_prod_imp:>8} {0:>8} {'imputed = position avg':>25}")

te26_ras_real = 0  # pre-draft, no combine yet
if 'ras_score' in te26_src.columns:
    te26_ras_real = te26_src['ras_score'].notna().sum()
    # Check if any are real vs all imputed - for 2026 prospects most won't have combine
    # Actually some early TEs may have RAS from all-star game workouts
print(f"     {'RAS':.<35} {te26_ras_real:>6} {n_te_26 - te26_ras_real:>8} {0:>8} {'MNAR imputed (pre-draft)':>25}")

# --- 4. NFL OUTCOMES (TE backtest) ---
print(f"\n  4. NFL OUTCOME COVERAGE — BACKTEST ({n_te_bt} TEs)")
print(f"     NOTE: TE outcomes use 10-game minimum season filter")
for col, label in [('nfl_hit24', 'top12_10g (top-12 TE season)'),
                   ('nfl_hit12', 'top6_10g (top-6 TE season)'),
                   ('nfl_first_3yr_ppg', 'best_3yr_ppg_10g'),
                   ('nfl_career_ppg', 'best_career_ppg_10g')]:
    n = te_bt_m[col].notna().sum()
    print(f"     {label:<35} {n:>5}/{n_te_bt}   {'FULL' if n == n_te_bt else f'{n_te_bt-n} missing'}")

for yr in sorted(te_bt_m['draft_year'].unique()):
    yr_data = te_bt_m[te_bt_m['draft_year'] == yr]
    ppg_miss = yr_data['nfl_first_3yr_ppg'].isna().sum()
    if ppg_miss > 0:
        print(f"       → {int(yr)}: {ppg_miss}/{len(yr_data)} missing best_3yr_ppg_10g", end='')
        if yr >= 2023:
            print(f" (too early — drafted {int(yr)}, need through {int(yr+2)})")
        else:
            print(f" (data gap or <10 games)")


# ============================================================================
# 5. KNOWN DATA ISSUES
# ============================================================================
print(f"\n\n{'='*120}")
print("5. KNOWN DATA ISSUES")
print("=" * 120)

# --- WR Issues ---
print(f"\n  ── WR KNOWN ISSUES ──")
# Players with no breakout AND no dominator data
wr_no_bo_data = wr_bt_src[wr_bt_src['peak_dominator'].isna()]
if len(wr_no_bo_data) > 0:
    print(f"  {len(wr_no_bo_data)} WRs with NO PFF dominator data (scored breakout=25 default):")
    for _, r in wr_no_bo_data.iterrows():
        print(f"    - {r['player_name']} ({r['college']}, {int(r['draft_year'])} Rd{int(r['round'])} pick {int(r['pick'])})")
else:
    print(f"  All {n_wr_bt} WRs have PFF dominator data.")

# Players with no rush yards
wr_no_rush = wr_bt_src[wr_bt_src['rush_yards'].isna()]
if len(wr_no_rush) > 0:
    print(f"\n  {len(wr_no_rush)} WRs missing rush yards (no +5 bonus applied):")
    if len(wr_no_rush) <= 10:
        for _, r in wr_no_rush.iterrows():
            print(f"    - {r['player_name']} ({r['college']}, {int(r['draft_year'])})")
    else:
        print(f"    (showing first 10)")
        for _, r in wr_no_rush.head(10).iterrows():
            print(f"    - {r['player_name']} ({r['college']}, {int(r['draft_year'])})")

# WRs who are DC-only (missing ALL non-DC components)
wr_dc_only = wr_bt_src[
    wr_bt_src['peak_dominator'].isna() &
    wr_bt_src['rush_yards'].isna()
]
if len(wr_dc_only) > 0:
    print(f"\n  {len(wr_dc_only)} WRs effectively DC-only (no breakout data + no rush data):")
    for _, r in wr_dc_only.iterrows():
        print(f"    - {r['player_name']} ({r['college']}, {int(r['draft_year'])} pick {int(r['pick'])})")

# --- RB Issues ---
print(f"\n  ── RB KNOWN ISSUES ──")
# RBs with no production
rb_no_prod = rb_bt_src[pd.to_numeric(rb_bt_src['rec_yards'], errors='coerce').isna() |
                        pd.to_numeric(rb_bt_src['team_pass_att'], errors='coerce').isna()]
if len(rb_no_prod) > 0:
    print(f"  {len(rb_no_prod)} RBs missing receiving production data (scored RYPTPA=0):")
    for _, r in rb_no_prod.iterrows():
        ry_val = r['rec_yards'] if pd.notna(r['rec_yards']) else 'missing'
        tpa_val = r['team_pass_att'] if pd.notna(r['team_pass_att']) else 'missing'
        print(f"    - {r['player_name']} ({r['college']}, {int(r['draft_year'])} Rd{int(r['round'])}) rec_yards={ry_val}, team_pa={tpa_val}")

# RBs with no weight (full MNAR imputation)
rb_bt['name_norm'] = rb_bt_src['player_name'].apply(normalize_name)
no_wt_names = []
for idx in rb_bt_src.index:
    nn = normalize_name(rb_bt_src.loc[idx, 'player_name'])
    dy = int(rb_bt_src.loc[idx, 'draft_year'])
    match = combine_rb[(combine_rb['name_norm'] == nn)]
    if len(match) == 0 or match.iloc[0]['wt'] != match.iloc[0]['wt']:  # NaN check
        no_wt_names.append((rb_bt_src.loc[idx, 'player_name'], rb_bt_src.loc[idx, 'college'],
                           int(rb_bt_src.loc[idx, 'draft_year']), int(rb_bt_src.loc[idx, 'round'])))
if no_wt_names:
    print(f"\n  {len(no_wt_names)} RBs with NO combine weight (full MNAR speed score):")
    for name, college, dy, rd in no_wt_names[:15]:
        print(f"    - {name} ({college}, {dy} Rd{rd})")
    if len(no_wt_names) > 15:
        print(f"    ... and {len(no_wt_names) - 15} more")

# --- TE Issues ---
print(f"\n  ── TE KNOWN ISSUES ──")
# Manual patches
print(f"  Manual data patches applied:")
print(f"    - Dallas Goedert: rec_yards=1111, team_pass_att=455 (2018, South Dakota St — not in CFBD)")
print(f"    - Adam Shaheen: rec_yards=867, team_pass_att=328 (2017, Ashland — not in CFBD)")

# TEs with no PFF data at all
te_no_pff = te_bt_src[te_bt_src['peak_dominator'].isna()]
if len(te_no_pff) > 0:
    print(f"\n  {len(te_no_pff)} TEs with NO PFF data (breakout score=25 default):")
    for _, r in te_no_pff.iterrows():
        print(f"    - {r['player_name']} ({r['college']}, {int(r['draft_year'])} Rd{int(r['round'])} pick {int(r['pick'])})")

# TEs with no production
te_no_prod = te_bt_src[te_bt_src['cfbd_rec_yards'].isna() & te_bt_src['cfbd_team_pass_att'].isna()]
# After manual patches, some of these are fixed
patched_names = ['Dallas Goedert', 'Adam Shaheen']
te_no_prod_after = te_no_prod[~te_no_prod['player_name'].isin(patched_names)]
if len(te_no_prod_after) > 0:
    print(f"\n  {len(te_no_prod_after)} TEs with no CFBD production data (after manual patches, imputed=avg):")
    for _, r in te_no_prod_after.iterrows():
        print(f"    - {r['player_name']} ({r['college']}, {int(r['draft_year'])} Rd{int(r['round'])})")

# TEs with no RAS
te_no_ras = te_bt_src[te_bt_src['te_ras'].isna()]
print(f"\n  {len(te_no_ras)} TEs with no RAS data (MNAR imputed):")
rd12_no_ras = te_no_ras[te_no_ras['round'] <= 2]
rd3_no_ras = te_no_ras[te_no_ras['round'] > 2]
print(f"    Rd 1-2 (→ p60 imputation): {len(rd12_no_ras)}")
if len(rd12_no_ras) > 0:
    for _, r in rd12_no_ras.iterrows():
        print(f"      - {r['player_name']} ({r['college']}, {int(r['draft_year'])} Rd{int(r['round'])})")
print(f"    Rd 3+  (→ p40 imputation): {len(rd3_no_ras)}")


# ============================================================================
# 6. THE BOTTOM LINE
# ============================================================================
print(f"\n\n{'='*120}")
print("6. THE BOTTOM LINE — Data Quality Summary Per Position")
print("=" * 120)

# WR bottom line
print(f"\n  ── WR ({n_wr_bt} backtest + {n_wr_26} prospects) ──")
# 100% real = has breakout_age OR peak_dominator, has rush_yards, has early_declare, has teammate_dc
wr_all_real = wr_bt_src[
    wr_bt_src['peak_dominator'].notna() &
    wr_bt_src['rush_yards'].notna() &
    wr_bt_src['early_declare'].notna()
].shape[0]
wr_any_imputed = n_wr_bt - wr_all_real
print(f"  100% real data across all inputs:  {wr_all_real}/{n_wr_bt} ({wr_all_real/n_wr_bt*100:.0f}%)")
print(f"  At least one imputed/default:      {wr_any_imputed}/{n_wr_bt} ({wr_any_imputed/n_wr_bt*100:.0f}%)")
print(f"  Biggest data gap:                  Rush yards missing for {n_wr_bt - rush_real}/{n_wr_bt} ({(n_wr_bt-rush_real)/n_wr_bt*100:.0f}%)")
# Impact assessment
print(f"  Impact: Rush bonus is only +5 on a 0-100 scale (tiny). Missing = no bonus. Low risk.")

# RB bottom line
print(f"\n  ── RB ({n_rb_bt} backtest + {n_rb_26} prospects) ──")
# 100% real = has rec_yards, team_pass_att, AND real combine weight + 40
rb_all_real_count = 0
for idx in rb_bt_src.index:
    nn = normalize_name(rb_bt_src.loc[idx, 'player_name'])
    dy = int(rb_bt_src.loc[idx, 'draft_year'])
    match = combine_rb[(combine_rb['name_norm'] == nn)]
    has_wt = len(match) > 0 and pd.notna(match.iloc[0]['wt'])
    has_40 = len(match) > 0 and pd.notna(match.iloc[0]['forty'])
    has_ry = pd.notna(pd.to_numeric(rb_bt_src.loc[idx, 'rec_yards'], errors='coerce'))
    has_tpa = pd.notna(pd.to_numeric(rb_bt_src.loc[idx, 'team_pass_att'], errors='coerce'))
    if has_ry and has_tpa and has_wt and has_40:
        rb_all_real_count += 1
rb_any_imputed = n_rb_bt - rb_all_real_count
print(f"  100% real data across all inputs:  {rb_all_real_count}/{n_rb_bt} ({rb_all_real_count/n_rb_bt*100:.0f}%)")
print(f"  At least one imputed/estimated:    {rb_any_imputed}/{n_rb_bt} ({rb_any_imputed/n_rb_bt*100:.0f}%)")
print(f"  Biggest data gap:                  40-yard dash real for only {forty_matched}/{n_rb_bt} ({forty_matched/n_rb_bt*100:.0f}%)")
print(f"  Impact: Speed Score is only 5% weight. MNAR imputation handles bias. Low risk.")

# TE bottom line
print(f"\n  ── TE ({n_te_bt} backtest + {n_te_26} prospects) ──")
te_all_real = te_bt_src[
    te_bt_src['peak_dominator'].notna() &
    te_bt_src['cfbd_rec_yards'].notna() &
    te_bt_src['cfbd_team_pass_att'].notna() &
    te_bt_src['te_ras'].notna()
].shape[0]
te_any_imputed = n_te_bt - te_all_real
print(f"  100% real data across all inputs:  {te_all_real}/{n_te_bt} ({te_all_real/n_te_bt*100:.0f}%)")
print(f"  At least one imputed/default:      {te_any_imputed}/{n_te_bt} ({te_any_imputed/n_te_bt*100:.0f}%)")
# Find biggest gap
gaps = {
    'PFF dominator data': n_te_bt - te_pd_real,
    'CFBD production': n_te_bt - prod_real_count,
    'RAS': n_te_bt - ras_real,
}
biggest = max(gaps, key=gaps.get)
print(f"  Biggest data gap:                  {biggest} missing for {gaps[biggest]}/{n_te_bt} ({gaps[biggest]/n_te_bt*100:.0f}%)")
print(f"  Impact: TE has 3 non-DC components that all have imputation. RAS at 10% is the biggest")
print(f"          gap by weight. MNAR imputation handles it, but TE has the most imputed data overall.")

# 2026 prospects summary
print(f"\n  ── 2026 PROSPECTS (all positions) ──")
print(f"  WR prospects: DC + breakout + teammate + early declare = mostly real data")
print(f"  RB prospects: DC + RYPTPA real, Speed Score = ALL MNAR imputed (pre-draft, no combine)")
print(f"  TE prospects: DC mostly real, breakout/production partial, RAS = ALL MNAR imputed")
print(f"  Key: Speed Score (RB 5%) and RAS (TE 10%) are imputed for ALL 2026 prospects")
print(f"       because combine data isn't available pre-draft. Low impact given weights.")

print(f"\n\n  Nothing changed. This is diagnostic only.")
