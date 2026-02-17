"""
SLAP V5 — Final Comprehensive Audit
Part 1: Data Quality Audit
Part 2: Full Statistical Validation
Part 3: Practical Usage Guide (printed at end)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir('/home/user/SlapModelV3')

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, brier_score_loss

master = pd.read_csv('output/slap_v5_master_database.csv')

# Load raw source files for data quality audit
wr_bt_src = pd.read_csv('data/wr_backtest_all_components.csv')
rb_bt_src = pd.read_csv('data/rb_backtest_with_receiving.csv')
te_bt_src = pd.read_csv('data/te_backtest_master.csv')
wr26_src = pd.read_csv('output/slap_v5_wr_2026.csv')
rb26_src = pd.read_csv('data/prospects_final.csv')
rb26_src = rb26_src[rb26_src['position'] == 'RB']
te26_src = pd.read_csv('data/te_2026_prospects_final.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
combine = pd.read_parquet('data/nflverse/combine.parquet')


# ============================================================================
# PART 1: DATA QUALITY AUDIT
# ============================================================================
print("=" * 120)
print("PART 1: DATA QUALITY AUDIT")
print("=" * 120)

def pct(n, total):
    return f"{n/total*100:.0f}%" if total > 0 else "N/A"

def flag(n, total, threshold=0.20):
    return " ⚠️ FLAG" if n/total > threshold else ""

# ---- WR BACKTEST ----
n_wr = len(wr_bt_src)
print(f"\n  WR BACKTEST ({n_wr} players, 2015-2025)")
print(f"  {'Input':<35} {'Total':>6} {'Real':>6} {'Estimated':>10} {'Missing':>8} {'Method':<40} {'Flag'}")
print(f"  {'-'*130}")

# DC
print(f"  {'Draft pick':<35} {n_wr:>6} {n_wr:>6} {'0':>10} {'0':>8} {'Actual pick':.<40}")

# Breakout age
bo_real = wr_bt_src['breakout_age'].notna().sum()
bo_miss = n_wr - bo_real
# All have peak_dominator for fallback scoring
pd_real = wr_bt_src['peak_dominator'].notna().sum()
print(f"  {'Breakout age':.<35} {n_wr:>6} {bo_real:>6} {'0':>10} {bo_miss:>8} {'Never broke out → dom-based score':.<40} {flag(bo_miss, n_wr)}")
print(f"  {'Peak dominator %':.<35} {n_wr:>6} {pd_real:>6} {'0':>10} {n_wr-pd_real:>8} {'Default 25 if missing':.<40} {flag(n_wr-pd_real, n_wr)}")

# Rush yards
rush_real = wr_bt_src['rush_yards'].notna().sum()
print(f"  {'College rush yards':.<35} {n_wr:>6} {rush_real:>6} {'0':>10} {n_wr-rush_real:>8} {'Treated as 0 (no bonus)':.<40} {flag(n_wr-rush_real, n_wr)}")

# Teammate DC
wr_tm = pd.read_csv('data/wr_teammate_scores.csv')
tm_merged = wr_bt_src.merge(wr_tm[['player_name','draft_year','total_teammate_dc']], on=['player_name','draft_year'], how='left')
tm_real = tm_merged['total_teammate_dc'].notna().sum()
print(f"  {'Teammate DC':.<35} {n_wr:>6} {tm_real:>6} {'0':>10} {n_wr-tm_real:>8} {'Missing → 0 (no elite teammates)':.<40} {flag(n_wr-tm_real, n_wr)}")

# Early declare
ed_real = wr_bt_src['early_declare'].notna().sum()
print(f"  {'Early declare status':.<35} {n_wr:>6} {ed_real:>6} {'0':>10} {n_wr-ed_real:>8} {'':.<40}")

# NFL outcomes
wr_out = outcomes[outcomes['position'] == 'WR']
h24 = wr_out['hit24'].notna().sum()
h12 = wr_out['hit12'].notna().sum()
ppg3 = wr_out['first_3yr_ppg'].notna().sum()
ppgc = wr_out['career_ppg'].notna().sum()
print(f"  {'NFL hit24':.<35} {n_wr:>6} {h24:>6} {'0':>10} {n_wr-h24:>8} {'Season-level top-24 finish':.<40} {flag(n_wr-h24, n_wr)}")
print(f"  {'NFL hit12':.<35} {n_wr:>6} {h12:>6} {'0':>10} {n_wr-h12:>8} {'Season-level top-12 finish':.<40} {flag(n_wr-h12, n_wr)}")
print(f"  {'NFL first_3yr_ppg':.<35} {n_wr:>6} {ppg3:>6} {'0':>10} {n_wr-ppg3:>8} {'':.<40} {flag(n_wr-ppg3, n_wr)}")
print(f"  {'NFL career_ppg':.<35} {n_wr:>6} {ppgc:>6} {'0':>10} {n_wr-ppgc:>8} {'':.<40} {flag(n_wr-ppgc, n_wr)}")

# ---- WR 2026 ----
n_wr26 = len(wr26_src)
print(f"\n  WR 2026 PROSPECTS ({n_wr26} players)")
print(f"  {'Input':<35} {'Total':>6} {'Real':>6} {'Estimated':>10} {'Missing':>8} {'Method':<40} {'Flag'}")
print(f"  {'-'*130}")
print(f"  {'Projected pick':<35} {n_wr26:>6} {n_wr26:>6} {'0':>10} {'0':>8} {'Consensus mock draft':.<40}")
bo26 = wr26_src['breakout_age'].notna().sum() if 'breakout_age' in wr26_src.columns else 0
print(f"  {'Breakout age':<35} {n_wr26:>6} {bo26:>6} {'0':>10} {n_wr26-bo26:>8} {'Dom-based fallback':.<40} {flag(n_wr26-bo26, n_wr26)}")
ed26 = wr26_src['early_declare_score'].notna().sum() if 'early_declare_score' in wr26_src.columns else wr26_src['early_declare'].notna().sum() if 'early_declare' in wr26_src.columns else 0
print(f"  {'Early declare':<35} {n_wr26:>6} {ed26:>6} {'0':>10} {n_wr26-ed26:>8} {'':.<40}")
tm26 = wr26_src['teammate_score'].notna().sum() if 'teammate_score' in wr26_src.columns else 0
print(f"  {'Teammate score':<35} {n_wr26:>6} {tm26:>6} {'0':>10} {n_wr26-tm26:>8} {'':.<40}")

# ---- RB BACKTEST ----
n_rb = len(rb_bt_src)
print(f"\n\n  RB BACKTEST ({n_rb} players, 2015-2025)")
print(f"  {'Input':<35} {'Total':>6} {'Real':>6} {'Estimated':>10} {'Missing':>8} {'Method':<40} {'Flag'}")
print(f"  {'-'*130}")
print(f"  {'Draft pick':<35} {n_rb:>6} {n_rb:>6} {'0':>10} {'0':>8} {'Actual pick':.<40}")

rec_real = rb_bt_src['rec_yards'].notna().sum()
tpa_real = rb_bt_src['team_pass_att'].notna().sum()
prod_real = min(rec_real, tpa_real)
prod_miss = n_rb - prod_real
print(f"  {'Receiving yards (final season)':.<35} {n_rb:>6} {rec_real:>6} {'0':>10} {n_rb-rec_real:>8} {'CFBD API':.<40} {flag(n_rb-rec_real, n_rb)}")
print(f"  {'Team pass attempts':.<35} {n_rb:>6} {tpa_real:>6} {'0':>10} {n_rb-tpa_real:>8} {'CFBD API':.<40} {flag(n_rb-tpa_real, n_rb)}")
print(f"  {'Production score (combined)':.<35} {n_rb:>6} {prod_real:>6} {'0':>10} {prod_miss:>8} {'Missing → 0':.<40} {flag(prod_miss, n_rb)}")

# Speed score inputs - check combine
from build_master_database_v5_helpers import get_rb_athletic_counts
# Do it manually
def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()

rb_bt_src['name_norm'] = rb_bt_src['player_name'].apply(normalize_name)
combine['name_norm'] = combine['player_name'].apply(normalize_name)
combine_lookup = {}
for pos_pref in ['RB', 'FB', 'WR', 'TE']:
    sub = combine[combine['pos'] == pos_pref]
    for _, row in sub.iterrows():
        dy = row.get('draft_year')
        if pd.isna(dy): dy = row.get('season')
        if pd.isna(dy): continue
        key = (row['name_norm'], int(dy))
        if key not in combine_lookup:
            combine_lookup[key] = {'weight': row['wt'], 'forty': row['forty']}

wt_count = sum(1 for _, r in rb_bt_src.iterrows()
               if pd.notna(combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight')))
ft_count = sum(1 for _, r in rb_bt_src.iterrows()
               if pd.notna(combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty')))
both_count = sum(1 for _, r in rb_bt_src.iterrows()
                 if pd.notna(combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('weight'))
                 and pd.notna(combine_lookup.get((r['name_norm'], int(r['draft_year'])), {}).get('forty')))

wt_no_ft = wt_count - both_count
no_wt = n_rb - wt_count

print(f"  {'Weight (combine)':.<35} {n_rb:>6} {wt_count:>6} {'0':>10} {n_rb-wt_count:>8} {'combine.parquet':.<40} {flag(n_rb-wt_count, n_rb)}")
print(f"  {'40-yard dash':.<35} {n_rb:>6} {ft_count:>6} {'0':>10} {n_rb-ft_count:>8} {'combine.parquet':.<40} {flag(n_rb-ft_count, n_rb)}")
print(f"  {'Speed Score (real wt+40)':.<35} {n_rb:>6} {both_count:>6} {'0':>10} {n_rb-both_count:>8} {'(weight×200)/(40^4)':.<40} {flag(n_rb-both_count, n_rb)}")
print(f"  {'Speed Score (est. 40)':.<35} {n_rb:>6} {'':>6} {wt_no_ft:>10} {'':>8} {'Wt×round bucket avg 40':.<40}")
print(f"  {'Speed Score (MNAR imputed)':.<35} {n_rb:>6} {'':>6} {no_wt:>10} {'':>8} {'Rd1-2→p60, Rd3+→p40':.<40}")

rb_out = outcomes[outcomes['position'] == 'RB']
rh24 = rb_out['hit24'].notna().sum()
rh12 = rb_out['hit12'].notna().sum()
rppg3 = rb_out['first_3yr_ppg'].notna().sum()
rppgc = rb_out['career_ppg'].notna().sum()
print(f"  {'NFL hit24':.<35} {n_rb:>6} {rh24:>6} {'0':>10} {n_rb-rh24:>8} {'Season-level top-24 finish':.<40} {flag(n_rb-rh24, n_rb)}")
print(f"  {'NFL hit12':.<35} {n_rb:>6} {rh12:>6} {'0':>10} {n_rb-rh12:>8} {'Season-level top-12 finish':.<40} {flag(n_rb-rh12, n_rb)}")
print(f"  {'NFL first_3yr_ppg':.<35} {n_rb:>6} {rppg3:>6} {'0':>10} {n_rb-rppg3:>8} {'':.<40} {flag(n_rb-rppg3, n_rb)}")
print(f"  {'NFL career_ppg':.<35} {n_rb:>6} {rppgc:>6} {'0':>10} {n_rb-rppgc:>8} {'':.<40} {flag(n_rb-rppgc, n_rb)}")

# ---- RB 2026 ----
n_rb26 = len(rb26_src)
print(f"\n  RB 2026 PROSPECTS ({n_rb26} players)")
print(f"  {'Input':<35} {'Total':>6} {'Real':>6} {'Estimated':>10} {'Missing':>8} {'Method':<40} {'Flag'}")
print(f"  {'-'*130}")
print(f"  {'Projected pick':<35} {n_rb26:>6} {n_rb26:>6} {'0':>10} {'0':>8} {'Consensus mock':.<40}")
rb26_rec = rb26_src['rec_yards'].notna().sum()
rb26_tpa = rb26_src['team_pass_attempts'].notna().sum()
print(f"  {'Receiving yards':.<35} {n_rb26:>6} {rb26_rec:>6} {'0':>10} {n_rb26-rb26_rec:>8} {'CFBD API':.<40} {flag(n_rb26-rb26_rec, n_rb26)}")
print(f"  {'Team pass attempts':.<35} {n_rb26:>6} {rb26_tpa:>6} {'0':>10} {n_rb26-rb26_tpa:>8} {'CFBD API':.<40} {flag(n_rb26-rb26_tpa, n_rb26)}")
print(f"  {'Speed Score':.<35} {n_rb26:>6} {'0':>6} {n_rb26:>10} {'0':>8} {'All MNAR-imputed (no combine)':.<40}  ⚠️ FLAG")

# ---- TE BACKTEST ----
n_te = len(te_bt_src)
print(f"\n\n  TE BACKTEST ({n_te} players, 2015-2025)")
print(f"  {'Input':<35} {'Total':>6} {'Real':>6} {'Estimated':>10} {'Missing':>8} {'Method':<40} {'Flag'}")
print(f"  {'-'*130}")
print(f"  {'Draft pick':<35} {n_te:>6} {n_te:>6} {'0':>10} {'0':>8} {'Actual pick':.<40}")

te_bo = te_bt_src['breakout_age'].notna().sum()
te_pd = te_bt_src['peak_dominator'].notna().sum()
te_bo_neither = n_te - te_pd  # no PFF data at all
print(f"  {'Breakout age (15% dom)':.<35} {n_te:>6} {te_bo:>6} {'0':>10} {n_te-te_bo:>8} {'Never broke out → dom fallback':.<40}")
print(f"  {'Peak dominator % (PFF)':.<35} {n_te:>6} {te_pd:>6} {'0':>10} {te_bo_neither:>8} {'Default 25 if no PFF data':.<40} {flag(te_bo_neither, n_te)}")

te_rec = te_bt_src['cfbd_rec_yards'].notna().sum()
te_tpa = te_bt_src['cfbd_team_pass_att'].notna().sum()
te_prod = min(te_rec, te_tpa)
print(f"  {'Rec yards (CFBD)':.<35} {n_te:>6} {te_rec:>6} {'0':>10} {n_te-te_rec:>8} {'CFBD primary, PFF fallback':.<40} {flag(n_te-te_rec, n_te)}")
print(f"  {'Team pass att (CFBD)':.<35} {n_te:>6} {te_tpa:>6} {'0':>10} {n_te-te_tpa:>8} {'CFBD primary':.<40} {flag(n_te-te_tpa, n_te)}")
print(f"  {'Production score (combined)':.<35} {n_te:>6} {te_prod:>6} {'0':>10} {n_te-te_prod:>8} {'Missing → position mean':.<40} {flag(n_te-te_prod, n_te)}")

te_ras = te_bt_src['te_ras'].notna().sum()
print(f"  {'RAS (0-10 scale)':.<35} {n_te:>6} {te_ras:>6} {n_te-te_ras:>10} {'0':>8} {'MNAR: Rd1-2→p60, Rd3+→p40':.<40} {flag(n_te-te_ras, n_te)}")

# TE outcomes
te_h24 = te_bt_src['top12_10g'].notna().sum()
te_h12 = te_bt_src['top6_10g'].notna().sum()
te_ppg3 = te_bt_src['best_3yr_ppg_10g'].notna().sum()
print(f"  {'NFL top12_10g (=hit24)':.<35} {n_te:>6} {te_h24:>6} {'0':>10} {n_te-te_h24:>8} {'Top-12 TE by PPG, 10g min':.<40} {flag(n_te-te_h24, n_te)}")
print(f"  {'NFL top6_10g (=hit12)':.<35} {n_te:>6} {te_h12:>6} {'0':>10} {n_te-te_h12:>8} {'Top-6 TE by PPG, 10g min':.<40} {flag(n_te-te_h12, n_te)}")
print(f"  {'NFL best_3yr_ppg_10g':.<35} {n_te:>6} {te_ppg3:>6} {'0':>10} {n_te-te_ppg3:>8} {'':.<40} {flag(n_te-te_ppg3, n_te)}")

# ---- TE 2026 ----
n_te26 = len(te26_src)
print(f"\n  TE 2026 PROSPECTS ({n_te26} players)")
print(f"  {'Input':<35} {'Total':>6} {'Real':>6} {'Estimated':>10} {'Missing':>8} {'Method':<40} {'Flag'}")
print(f"  {'-'*130}")
print(f"  {'Projected pick':<35} {n_te26:>6} {n_te26:>6} {'0':>10} {'0':>8} {'Consensus mock':.<40}")
te26_bo = te26_src['breakout_age'].notna().sum()
te26_prod = te26_src['production_score'].notna().sum() if 'production_score' in te26_src.columns else te26_src['cfbd_rec_yards'].notna().sum()
te26_ras = te26_src['ras_score'].notna().sum() if 'ras_score' in te26_src.columns else 0
# Count how many ras are real vs imputed
if 'ras_score' in te26_src.columns:
    te26_ras_real = (te26_src['ras_score'] != 40.0).sum() & (te26_src['ras_score'] != 60.0).sum()
    # Actually just count non-null
    te26_ras_real = te26_src['ras_score'].notna().sum()
print(f"  {'Breakout age':.<35} {n_te26:>6} {te26_bo:>6} {'0':>10} {n_te26-te26_bo:>8} {'Dom fallback / default 25':.<40} {flag(n_te26-te26_bo, n_te26)}")
print(f"  {'Production':.<35} {n_te26:>6} {te26_prod:>6} {'0':>10} {n_te26-te26_prod:>8} {'CFBD primary':.<40} {flag(n_te26-te26_prod, n_te26)}")


# ============================================================================
# PART 2: FULL STATISTICAL VALIDATION
# ============================================================================
print(f"\n\n{'='*120}")
print("PART 2: FULL STATISTICAL VALIDATION")
print("=" * 120)

def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

# Separate by position
bt = master[master['data_type'] == 'backtest'].copy()

for pos in ['WR', 'RB', 'TE']:
    pos_bt = bt[bt['position'] == pos].copy()
    pos_bt['dc_only'] = pos_bt['pick'].apply(dc_score)

    # Determine outcomes
    if pos == 'TE':
        outcomes_list = [
            ('hit24 (top12_10g)', 'nfl_hit24', 'binary'),
            ('hit12 (top6_10g)', 'nfl_hit12', 'binary'),
            ('first_3yr_ppg', 'nfl_first_3yr_ppg', 'continuous'),
            ('career_ppg', 'nfl_career_ppg', 'continuous'),
        ]
    else:
        outcomes_list = [
            ('hit24', 'nfl_hit24', 'binary'),
            ('hit12', 'nfl_hit12', 'binary'),
            ('first_3yr_ppg', 'nfl_first_3yr_ppg', 'continuous'),
            ('career_ppg', 'nfl_career_ppg', 'continuous'),
        ]

    print(f"\n\n  {'='*100}")
    print(f"  {pos} VALIDATION (n={len(pos_bt)})")
    print(f"  {'='*100}")

    # --- 1 & 2: Pearson and Spearman correlations ---
    print(f"\n  1-2. CORRELATIONS (SLAP vs DC-only)")
    print(f"  {'Outcome':<25} {'SLAP Pearson':>14} {'DC Pearson':>12} {'SLAP Spear':>12} {'DC Spear':>10} {'SLAP wins?':>12}")
    print(f"  {'-'*90}")

    pri_weights = {'hit24': 0.25, 'hit12': 0.20, 'first_3yr_ppg': 0.40, 'career_ppg': 0.15,
                   'hit24 (top12_10g)': 0.25, 'hit12 (top6_10g)': 0.20}
    slap_pri = 0
    dc_pri = 0

    for label, col, otype in outcomes_list:
        valid = pos_bt[pos_bt[col].notna()].copy()
        if len(valid) < 10:
            print(f"  {label:<25} {'N/A':>14} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
            continue

        s_pear = pearsonr(valid['slap_v5'], valid[col])[0]
        d_pear = pearsonr(valid['dc_only'], valid[col])[0]
        s_spear = spearmanr(valid['slap_v5'], valid[col])[0]
        d_spear = spearmanr(valid['dc_only'], valid[col])[0]
        wins = "YES" if abs(s_spear) > abs(d_spear) else "no"

        print(f"  {label:<25} {s_pear:>+14.3f} {d_pear:>+12.3f} {s_spear:>+12.3f} {d_spear:>+10.3f} {wins:>12}")

        # Priority weighted
        short_label = label.split(' (')[0]
        w = pri_weights.get(label, pri_weights.get(short_label, 0))
        slap_pri += abs(s_spear) * w
        dc_pri += abs(d_spear) * w

    print(f"\n  Priority-weighted avg |r| (40/25/20/15):  SLAP={slap_pri:.3f}  DC={dc_pri:.3f}  Δ={slap_pri-dc_pri:+.3f}")

    # --- 4: AUC-ROC ---
    print(f"\n  4. AUC-ROC (binary outcomes)")
    print(f"  {'Outcome':<25} {'SLAP AUC':>10} {'DC AUC':>10} {'SLAP wins?':>12}")
    print(f"  {'-'*60}")

    for label, col, otype in outcomes_list:
        if otype != 'binary': continue
        valid = pos_bt[pos_bt[col].notna()].copy()
        if len(valid) < 10 or valid[col].nunique() < 2: continue

        s_auc = roc_auc_score(valid[col], valid['slap_v5'])
        d_auc = roc_auc_score(valid[col], valid['dc_only'])
        wins = "YES" if s_auc > d_auc else "no"
        print(f"  {label:<25} {s_auc:>10.3f} {d_auc:>10.3f} {wins:>12}")

    # --- 5: Brier scores ---
    print(f"\n  5. BRIER SCORES (lower is better)")
    print(f"  {'Outcome':<25} {'SLAP Brier':>12} {'DC Brier':>10} {'SLAP wins?':>12}")
    print(f"  {'-'*62}")

    for label, col, otype in outcomes_list:
        if otype != 'binary': continue
        valid = pos_bt[pos_bt[col].notna()].copy()
        if len(valid) < 10 or valid[col].nunique() < 2: continue

        # Normalize to 0-1 for Brier
        slap_norm = (valid['slap_v5'] - valid['slap_v5'].min()) / (valid['slap_v5'].max() - valid['slap_v5'].min())
        dc_norm = (valid['dc_only'] - valid['dc_only'].min()) / (valid['dc_only'].max() - valid['dc_only'].min())

        s_brier = brier_score_loss(valid[col], slap_norm)
        d_brier = brier_score_loss(valid[col], dc_norm)
        wins = "YES" if s_brier < d_brier else "no"
        print(f"  {label:<25} {s_brier:>12.4f} {d_brier:>10.4f} {wins:>12}")

    # --- 6: Bootstrap ---
    print(f"\n  6. BOOTSTRAP (1000 iterations, Spearman r)")
    np.random.seed(42)
    n_boot = 1000

    for label, col, otype in outcomes_list:
        valid = pos_bt[pos_bt[col].notna()].copy().reset_index(drop=True)
        if len(valid) < 10: continue

        slap_boots = []
        dc_boots = []
        slap_wins = 0

        for _ in range(n_boot):
            idx = np.random.choice(len(valid), size=len(valid), replace=True)
            sample = valid.iloc[idx]
            if sample[col].nunique() < 2: continue

            sr_slap = spearmanr(sample['slap_v5'], sample[col])[0]
            sr_dc = spearmanr(sample['dc_only'], sample[col])[0]
            slap_boots.append(sr_slap)
            dc_boots.append(sr_dc)
            if abs(sr_slap) > abs(sr_dc):
                slap_wins += 1

        slap_boots = np.array(slap_boots)
        dc_boots = np.array(dc_boots)
        n_valid_boots = len(slap_boots)

        s_lo, s_hi = np.percentile(slap_boots, [2.5, 97.5])
        d_lo, d_hi = np.percentile(dc_boots, [2.5, 97.5])
        pct_wins = slap_wins / n_valid_boots * 100

        print(f"  {label:<25} SLAP r=[{s_lo:+.3f}, {s_hi:+.3f}]  DC r=[{d_lo:+.3f}, {d_hi:+.3f}]  SLAP wins {pct_wins:.1f}%")

    # --- 7: Calibration by SLAP tier ---
    print(f"\n  7. CALIBRATION BY SLAP TIER")
    tiers = [(90, 99, '90-99'), (80, 89.9, '80-89'), (70, 79.9, '70-79'),
             (60, 69.9, '60-69'), (50, 59.9, '50-59'), (40, 49.9, '40-49'), (1, 39.9, '<40')]

    bin_cols = [(l, c) for l, c, t in outcomes_list if t == 'binary']
    cont_cols = [(l, c) for l, c, t in outcomes_list if t == 'continuous']

    header = f"  {'Tier':<8} {'N':>4}"
    for label, _ in bin_cols:
        short = label.split(' (')[0]
        header += f" {short+' %':>10}"
    for label, _ in cont_cols:
        short = label.split(' (')[0][:8]
        header += f" {short:>10}"
    print(header)
    print(f"  {'-'*len(header)}")

    for lo, hi, label in tiers:
        tier = pos_bt[(pos_bt['slap_v5'] >= lo) & (pos_bt['slap_v5'] <= hi)]
        if len(tier) == 0: continue
        row = f"  {label:<8} {len(tier):>4}"
        for _, col in bin_cols:
            valid = tier[tier[col].notna()]
            if len(valid) > 0:
                rate = valid[col].mean() * 100
                row += f" {rate:>9.1f}%"
            else:
                row += f" {'N/A':>10}"
        for _, col in cont_cols:
            valid = tier[tier[col].notna()]
            if len(valid) > 0:
                avg = valid[col].mean()
                row += f" {avg:>10.1f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # --- 8: Top-decile precision ---
    print(f"\n  8. TOP-DECILE PRECISION")
    n_decile = max(1, len(pos_bt) // 10)

    slap_top = pos_bt.nlargest(n_decile, 'slap_v5')
    dc_top = pos_bt.nlargest(n_decile, 'dc_only')

    print(f"  {'Metric':<25} {'SLAP top 10%':>15} {'DC top 10%':>12} {'SLAP wins?':>12}")
    print(f"  {'-'*68}")

    for label, col, otype in outcomes_list:
        s_valid = slap_top[slap_top[col].notna()]
        d_valid = dc_top[dc_top[col].notna()]
        if len(s_valid) == 0: continue

        if otype == 'binary':
            s_val = s_valid[col].mean() * 100
            d_val = d_valid[col].mean() * 100
            wins = "YES" if s_val > d_val else "no"
            print(f"  {label+' rate':<25} {s_val:>14.1f}% {d_val:>11.1f}% {wins:>12}")
        else:
            s_val = s_valid[col].mean()
            d_val = d_valid[col].mean()
            wins = "YES" if s_val > d_val else "no"
            print(f"  {label+' avg':<25} {s_val:>15.2f} {d_val:>12.2f} {wins:>12}")

    print(f"  (Top decile = top {n_decile} players)")


# ============================================================================
# PART 3: PRACTICAL USAGE GUIDE
# ============================================================================
print(f"\n\n{'='*120}")
print("PART 3: PRACTICAL USAGE GUIDE")
print("=" * 120)

# Gather key stats for the guide
for pos in ['WR', 'RB', 'TE']:
    pos_bt = bt[bt['position'] == pos]
    hit_col = 'nfl_hit24'

    print(f"\n  {pos} — Key thresholds:")
    for lo, hi, label in [(80, 99, '80+'), (60, 79.9, '60-79'), (40, 59.9, '40-59'), (1, 39.9, '<40')]:
        tier = pos_bt[(pos_bt['slap_v5'] >= lo) & (pos_bt['slap_v5'] <= hi)]
        valid = tier[tier[hit_col].notna()]
        if len(valid) > 0:
            hr = valid[hit_col].mean() * 100
            ppg = valid['nfl_first_3yr_ppg'].mean() if valid['nfl_first_3yr_ppg'].notna().any() else 0
            print(f"    SLAP {label:<6}: {len(tier):>3} players, {hr:.0f}% hit rate, {ppg:.1f} avg 3yr PPG")

# Delta analysis
print(f"\n  DELTA VALIDATION (do positive deltas outperform negative deltas?):")
for pos in ['WR', 'RB', 'TE']:
    pos_bt = bt[bt['position'] == pos]
    boosted = pos_bt[pos_bt['delta_vs_dc'] >= 5]
    dinged = pos_bt[pos_bt['delta_vs_dc'] <= -5]

    b_ppg = boosted['nfl_first_3yr_ppg'].mean() if boosted['nfl_first_3yr_ppg'].notna().any() else 0
    d_ppg = dinged['nfl_first_3yr_ppg'].mean() if dinged['nfl_first_3yr_ppg'].notna().any() else 0
    b_hit = boosted['nfl_hit24'].mean() * 100 if boosted['nfl_hit24'].notna().any() else 0
    d_hit = dinged['nfl_hit24'].mean() * 100 if dinged['nfl_hit24'].notna().any() else 0

    print(f"    {pos}: Boosted (delta>=+5, n={len(boosted)}): {b_ppg:.1f} PPG, {b_hit:.0f}% hit | "
          f"Dinged (delta<=-5, n={len(dinged)}): {d_ppg:.1f} PPG, {d_hit:.0f}% hit | "
          f"Gap: {b_ppg-d_ppg:+.1f} PPG")

print(f"\n\n  DONE — audit complete.")
