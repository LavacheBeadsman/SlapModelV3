"""
SLAP V5 — Final Comprehensive Audit (3 parts)
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
wr_bt_src = pd.read_csv('data/wr_backtest_all_components.csv')
rb_bt_src = pd.read_csv('data/rb_backtest_with_receiving.csv')
te_bt_src = pd.read_csv('data/te_backtest_master.csv')
wr26_src = pd.read_csv('output/slap_v5_wr_2026.csv')
prospects = pd.read_csv('data/prospects_final.csv')
rb26_src = prospects[prospects['position'] == 'RB']
wr26_p = prospects[prospects['position'] == 'WR']
te26_src = pd.read_csv('data/te_2026_prospects_final.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')

def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()

combine = pd.read_parquet('data/nflverse/combine.parquet')
combine['name_norm'] = combine['player_name'].apply(normalize_name)
rb_bt_src['name_norm'] = rb_bt_src['player_name'].apply(normalize_name)
combine_lookup = {}
for pos_pref in ['RB','FB']:
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

# ============================================================================
print("=" * 120)
print("PART 1: DATA QUALITY AUDIT")
print("=" * 120)

def flag20(n, total):
    return " *** FLAG >20%" if total > 0 and n/total > 0.20 else ""

# ---- WR BACKTEST ----
n = len(wr_bt_src)
wr_tm = pd.read_csv('data/wr_teammate_scores.csv')
tm_m = wr_bt_src.merge(wr_tm[['player_name','draft_year','total_teammate_dc']], on=['player_name','draft_year'], how='left')
wr_out = outcomes[outcomes['position'] == 'WR']

print(f"\n  WR BACKTEST (n={n}, 2015-2025)")
print(f"  {'Input':<30} {'Total':>5} {'Real':>5} {'Est/Imp':>7} {'Miss':>5} {'Method':<35} {'Flag'}")
print(f"  {'-'*100}")
print(f"  {'Draft pick':<30} {n:>5} {n:>5} {'0':>7} {'0':>5} {'Actual draft pick':<35}")
bo = wr_bt_src['breakout_age'].notna().sum()
print(f"  {'Breakout age':<30} {n:>5} {bo:>5} {'0':>7} {n-bo:>5} {'Never broke out=dom fallback':<35}")
pd_r = wr_bt_src['peak_dominator'].notna().sum()
print(f"  {'Peak dominator %':<30} {n:>5} {pd_r:>5} {'0':>7} {n-pd_r:>5} {'Default score=25':<35}{flag20(n-pd_r,n)}")
rush = wr_bt_src['rush_yards'].notna().sum()
print(f"  {'College rush yards':<30} {n:>5} {rush:>5} {'0':>7} {n-rush:>5} {'Treated as 0 (no +5 bonus)':<35}{flag20(n-rush,n)}")
tm_r = tm_m['total_teammate_dc'].notna().sum()
print(f"  {'Teammate DC':<30} {n:>5} {tm_r:>5} {'0':>7} {n-tm_r:>5} {'Missing=0 (no elite mates)':<35}{flag20(n-tm_r,n)}")
ed = wr_bt_src['early_declare'].notna().sum()
print(f"  {'Early declare':<30} {n:>5} {ed:>5} {'0':>7} {n-ed:>5} {'':<35}")
h24 = wr_out['hit24'].notna().sum()
h12 = wr_out['hit12'].notna().sum()
p3 = wr_out['first_3yr_ppg'].notna().sum()
pc = wr_out['career_ppg'].notna().sum()
print(f"  {'NFL hit24':<30} {n:>5} {h24:>5} {'0':>7} {n-h24:>5} {'Top-24 season finish':<35}{flag20(n-h24,n)}")
print(f"  {'NFL hit12':<30} {n:>5} {h12:>5} {'0':>7} {n-h12:>5} {'Top-12 season finish':<35}{flag20(n-h12,n)}")
print(f"  {'NFL first_3yr_ppg':<30} {n:>5} {p3:>5} {'0':>7} {n-p3:>5} {'':<35}{flag20(n-p3,n)}")
print(f"  {'NFL career_ppg':<30} {n:>5} {pc:>5} {'0':>7} {n-pc:>5} {'':<35}{flag20(n-pc,n)}")

# ---- WR 2026 ----
n26 = len(wr26_src)
print(f"\n  WR 2026 PROSPECTS (n={n26})")
print(f"  {'Input':<30} {'Total':>5} {'Real':>5} {'Est/Imp':>7} {'Miss':>5} {'Method':<35} {'Flag'}")
print(f"  {'-'*100}")
print(f"  {'Projected pick':<30} {n26:>5} {n26:>5} {'0':>7} {'0':>5} {'Consensus mock draft':<35}")
bo26 = wr26_src['breakout_age'].notna().sum() if 'breakout_age' in wr26_src.columns else 0
print(f"  {'Breakout age':<30} {n26:>5} {bo26:>5} {'0':>7} {n26-bo26:>5} {'Dom fallback':<35}{flag20(n26-bo26,n26)}")
print(f"  {'Early declare':<30} {n26:>5} {n26:>5} {'0':>7} {'0':>5} {'Manual research':<35}")
print(f"  {'Teammate score':<30} {n26:>5} {n26:>5} {'0':>7} {'0':>5} {'Manual research':<35}")

# ---- RB BACKTEST ----
n = len(rb_bt_src)
rb_out = outcomes[outcomes['position'] == 'RB']
print(f"\n  RB BACKTEST (n={n}, 2015-2025)")
print(f"  {'Input':<30} {'Total':>5} {'Real':>5} {'Est/Imp':>7} {'Miss':>5} {'Method':<35} {'Flag'}")
print(f"  {'-'*100}")
print(f"  {'Draft pick':<30} {n:>5} {n:>5} {'0':>7} {'0':>5} {'Actual draft pick':<35}")
rec_r = rb_bt_src['rec_yards'].notna().sum()
tpa_r = rb_bt_src['team_pass_att'].notna().sum()
prod_miss = n - min(rec_r, tpa_r)
print(f"  {'Receiving yards':<30} {n:>5} {rec_r:>5} {'0':>7} {n-rec_r:>5} {'CFBD API':<35}{flag20(n-rec_r,n)}")
print(f"  {'Team pass attempts':<30} {n:>5} {tpa_r:>5} {'0':>7} {n-tpa_r:>5} {'CFBD API':<35}{flag20(n-tpa_r,n)}")
print(f"  {'Production (combined)':<30} {n:>5} {min(rec_r,tpa_r):>5} {'0':>7} {prod_miss:>5} {'Missing=0':<35}{flag20(prod_miss,n)}")
wt_no_ft = wt_count - both_count
no_wt = n - wt_count
print(f"  {'Weight':<30} {n:>5} {wt_count:>5} {'0':>7} {n-wt_count:>5} {'combine.parquet':<35}{flag20(n-wt_count,n)}")
print(f"  {'40-yard dash':<30} {n:>5} {ft_count:>5} {'0':>7} {n-ft_count:>5} {'combine.parquet':<35}{flag20(n-ft_count,n)}")
print(f"  {'Speed Score (real wt+40)':<30} {n:>5} {both_count:>5} {'0':>7} {n-both_count:>5} {'(wt*200)/(40^4)':<35}{flag20(n-both_count,n)}")
print(f"  {'  + estimated 40':<30} {'':>5} {'':>5} {wt_no_ft:>7} {'':>5} {'Wt x round bucket avg':<35}")
print(f"  {'  + MNAR imputed':<30} {'':>5} {'':>5} {no_wt:>7} {'':>5} {'Rd1-2->p60, Rd3+->p40':<35}")
rh24 = rb_out['hit24'].notna().sum()
rh12 = rb_out['hit12'].notna().sum()
rp3 = rb_out['first_3yr_ppg'].notna().sum()
rpc = rb_out['career_ppg'].notna().sum()
print(f"  {'NFL hit24':<30} {n:>5} {rh24:>5} {'0':>7} {n-rh24:>5} {'Top-24 season finish':<35}{flag20(n-rh24,n)}")
print(f"  {'NFL hit12':<30} {n:>5} {rh12:>5} {'0':>7} {n-rh12:>5} {'Top-12 season finish':<35}{flag20(n-rh12,n)}")
print(f"  {'NFL first_3yr_ppg':<30} {n:>5} {rp3:>5} {'0':>7} {n-rp3:>5} {'':<35}{flag20(n-rp3,n)}")
print(f"  {'NFL career_ppg':<30} {n:>5} {rpc:>5} {'0':>7} {n-rpc:>5} {'':<35}{flag20(n-rpc,n)}")

# ---- RB 2026 ----
n26 = len(rb26_src)
print(f"\n  RB 2026 PROSPECTS (n={n26})")
print(f"  {'Input':<30} {'Total':>5} {'Real':>5} {'Est/Imp':>7} {'Miss':>5} {'Method':<35} {'Flag'}")
print(f"  {'-'*100}")
print(f"  {'Projected pick':<30} {n26:>5} {n26:>5} {'0':>7} {'0':>5} {'Consensus mock':<35}")
r26rec = rb26_src['rec_yards'].notna().sum()
r26tpa = rb26_src['team_pass_attempts'].notna().sum()
print(f"  {'Receiving yards':<30} {n26:>5} {r26rec:>5} {'0':>7} {n26-r26rec:>5} {'CFBD API':<35}{flag20(n26-r26rec,n26)}")
print(f"  {'Team pass attempts':<30} {n26:>5} {r26tpa:>5} {'0':>7} {n26-r26tpa:>5} {'CFBD API':<35}{flag20(n26-r26tpa,n26)}")
print(f"  {'Speed Score':<30} {n26:>5} {'0':>5} {n26:>7} {'0':>5} {'ALL MNAR-imputed':<35} *** FLAG >20%")

# ---- TE BACKTEST ----
n = len(te_bt_src)
print(f"\n  TE BACKTEST (n={n}, 2015-2025)")
print(f"  {'Input':<30} {'Total':>5} {'Real':>5} {'Est/Imp':>7} {'Miss':>5} {'Method':<35} {'Flag'}")
print(f"  {'-'*100}")
print(f"  {'Draft pick':<30} {n:>5} {n:>5} {'0':>7} {'0':>5} {'Actual draft pick':<35}")
tbo = te_bt_src['breakout_age'].notna().sum()
tpd = te_bt_src['peak_dominator'].notna().sum()
tno_pff = n - tpd
print(f"  {'Breakout age (15% dom)':<30} {n:>5} {tbo:>5} {'0':>7} {n-tbo:>5} {'Never broke out=dom fallback':<35}")
print(f"  {'Peak dominator (PFF)':<30} {n:>5} {tpd:>5} {'0':>7} {tno_pff:>5} {'Default score=25 if no PFF':<35}{flag20(tno_pff,n)}")
trec = te_bt_src['cfbd_rec_yards'].notna().sum()
ttpa = te_bt_src['cfbd_team_pass_att'].notna().sum()
tprod = min(trec, ttpa)
print(f"  {'Rec yards (CFBD)':<30} {n:>5} {trec:>5} {'0':>7} {n-trec:>5} {'CFBD primary':<35}{flag20(n-trec,n)}")
print(f"  {'Team pass att (CFBD)':<30} {n:>5} {ttpa:>5} {'0':>7} {n-ttpa:>5} {'CFBD primary':<35}{flag20(n-ttpa,n)}")
tprod_imp = n - tprod
print(f"  {'Production (combined)':<30} {n:>5} {tprod:>5} {tprod_imp:>7} {'0':>5} {'Missing=position mean':<35}{flag20(tprod_imp,n)}")
tras = te_bt_src['te_ras'].notna().sum()
print(f"  {'RAS (0-10 scale)':<30} {n:>5} {tras:>5} {n-tras:>7} {'0':>5} {'MNAR: Rd1-2->p60,Rd3+->p40':<35}{flag20(n-tras,n)}")
th24 = te_bt_src['top12_10g'].notna().sum()
th12 = te_bt_src['top6_10g'].notna().sum()
tp3 = te_bt_src['best_3yr_ppg_10g'].notna().sum()
tpc = te_bt_src['best_career_ppg_10g'].notna().sum() if 'best_career_ppg_10g' in te_bt_src.columns else 0
print(f"  {'NFL top12 (=hit24)':<30} {n:>5} {th24:>5} {'0':>7} {n-th24:>5} {'Top-12 TE PPG, 10g min':<35}{flag20(n-th24,n)}")
print(f"  {'NFL top6 (=hit12)':<30} {n:>5} {th12:>5} {'0':>7} {n-th12:>5} {'Top-6 TE PPG, 10g min':<35}{flag20(n-th12,n)}")
print(f"  {'NFL best_3yr_ppg':<30} {n:>5} {tp3:>5} {'0':>7} {n-tp3:>5} {'':<35}{flag20(n-tp3,n)}")

# ---- TE 2026 ----
n26 = len(te26_src)
print(f"\n  TE 2026 PROSPECTS (n={n26})")
print(f"  {'Input':<30} {'Total':>5} {'Real':>5} {'Est/Imp':>7} {'Miss':>5} {'Method':<35} {'Flag'}")
print(f"  {'-'*100}")
print(f"  {'Projected pick':<30} {n26:>5} {n26:>5} {'0':>7} {'0':>5} {'Consensus mock':<35}")
t26bo = te26_src['breakout_age'].notna().sum()
t26rec = te26_src['cfbd_rec_yards'].notna().sum()
t26tpa = te26_src['cfbd_team_pass_att'].notna().sum()
print(f"  {'Breakout age':<30} {n26:>5} {t26bo:>5} {'0':>7} {n26-t26bo:>5} {'Dom fallback / default 25':<35}{flag20(n26-t26bo,n26)}")
print(f"  {'Rec yards (CFBD)':<30} {n26:>5} {t26rec:>5} {'0':>7} {n26-t26rec:>5} {'CFBD primary':<35}{flag20(n26-t26rec,n26)}")
print(f"  {'Team pass att':<30} {n26:>5} {t26tpa:>5} {'0':>7} {n26-t26tpa:>5} {'CFBD primary':<35}{flag20(n26-t26tpa,n26)}")


# ============================================================================
print(f"\n\n{'='*120}")
print("PART 2: FULL STATISTICAL VALIDATION")
print("=" * 120)

def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

bt = master[master['data_type'] == 'backtest'].copy()

for pos in ['WR', 'RB', 'TE']:
    pos_bt = bt[bt['position'] == pos].copy()
    pos_bt['dc_only'] = pos_bt['pick'].apply(dc_score)

    if pos == 'TE':
        outs = [('hit24 (top12_10g)', 'nfl_hit24', 'binary'),
                ('hit12 (top6_10g)', 'nfl_hit12', 'binary'),
                ('first_3yr_ppg', 'nfl_first_3yr_ppg', 'continuous'),
                ('career_ppg', 'nfl_career_ppg', 'continuous')]
    else:
        outs = [('hit24', 'nfl_hit24', 'binary'),
                ('hit12', 'nfl_hit12', 'binary'),
                ('first_3yr_ppg', 'nfl_first_3yr_ppg', 'continuous'),
                ('career_ppg', 'nfl_career_ppg', 'continuous')]

    print(f"\n\n  {'='*100}")
    print(f"  {pos} VALIDATION (n={len(pos_bt)})")
    print(f"  {'='*100}")

    # 1-2: Correlations
    print(f"\n  1-2. CORRELATIONS")
    print(f"  {'Outcome':<22} {'SLAP Pear':>10} {'DC Pear':>9} {'SLAP Spr':>10} {'DC Spr':>8} {'Wins?':>7}")
    print(f"  {'-'*70}")

    pri_w = {}
    for label, col, ot in outs:
        short = label.split(' (')[0]
        if 'hit24' in short: pri_w[label] = 0.25
        elif 'hit12' in short: pri_w[label] = 0.20
        elif 'first' in short: pri_w[label] = 0.40
        elif 'career' in short: pri_w[label] = 0.15

    slap_pri = dc_pri = 0
    for label, col, ot in outs:
        v = pos_bt[pos_bt[col].notna()]
        if len(v) < 10: continue
        sp = pearsonr(v['slap_v5'], v[col])[0]
        dp = pearsonr(v['dc_only'], v[col])[0]
        ss = spearmanr(v['slap_v5'], v[col])[0]
        ds = spearmanr(v['dc_only'], v[col])[0]
        w = "YES" if abs(ss) > abs(ds) else "no"
        print(f"  {label:<22} {sp:>+10.3f} {dp:>+9.3f} {ss:>+10.3f} {ds:>+8.3f} {w:>7}")
        slap_pri += abs(ss) * pri_w.get(label, 0)
        dc_pri += abs(ds) * pri_w.get(label, 0)

    print(f"\n  9. PRI-WEIGHTED AVG |r|:  SLAP={slap_pri:.3f}  DC={dc_pri:.3f}  Delta={slap_pri-dc_pri:+.3f}")

    # 4: AUC-ROC
    print(f"\n  4. AUC-ROC")
    print(f"  {'Outcome':<22} {'SLAP':>8} {'DC':>8} {'Wins?':>7}")
    print(f"  {'-'*48}")
    for label, col, ot in outs:
        if ot != 'binary': continue
        v = pos_bt[pos_bt[col].notna()]
        if len(v) < 10 or v[col].nunique() < 2: continue
        sa = roc_auc_score(v[col], v['slap_v5'])
        da = roc_auc_score(v[col], v['dc_only'])
        w = "YES" if sa > da else "no"
        print(f"  {label:<22} {sa:>8.3f} {da:>8.3f} {w:>7}")

    # 5: Brier
    print(f"\n  5. BRIER SCORES (lower=better)")
    print(f"  {'Outcome':<22} {'SLAP':>8} {'DC':>8} {'Wins?':>7}")
    print(f"  {'-'*48}")
    for label, col, ot in outs:
        if ot != 'binary': continue
        v = pos_bt[pos_bt[col].notna()]
        if len(v) < 10 or v[col].nunique() < 2: continue
        sn = (v['slap_v5'] - v['slap_v5'].min()) / (v['slap_v5'].max() - v['slap_v5'].min())
        dn = (v['dc_only'] - v['dc_only'].min()) / (v['dc_only'].max() - v['dc_only'].min())
        sb = brier_score_loss(v[col], sn)
        db = brier_score_loss(v[col], dn)
        w = "YES" if sb < db else "no"
        print(f"  {label:<22} {sb:>8.4f} {db:>8.4f} {w:>7}")

    # 6: Bootstrap
    print(f"\n  6. BOOTSTRAP (1000 iter, Spearman)")
    np.random.seed(42)
    for label, col, ot in outs:
        v = pos_bt[pos_bt[col].notna()].reset_index(drop=True)
        if len(v) < 10: continue
        sb, db2 = [], []
        wins = 0
        for _ in range(1000):
            idx = np.random.choice(len(v), len(v), replace=True)
            s = v.iloc[idx]
            if s[col].nunique() < 2: continue
            rs = spearmanr(s['slap_v5'], s[col])[0]
            rd = spearmanr(s['dc_only'], s[col])[0]
            sb.append(rs); db2.append(rd)
            if abs(rs) > abs(rd): wins += 1
        sb = np.array(sb); db2 = np.array(db2)
        pct_w = wins / len(sb) * 100
        print(f"  {label:<22} SLAP=[{np.percentile(sb,2.5):+.3f},{np.percentile(sb,97.5):+.3f}] "
              f"DC=[{np.percentile(db2,2.5):+.3f},{np.percentile(db2,97.5):+.3f}] SLAP wins {pct_w:.1f}%")

    # 7: Calibration
    print(f"\n  7. CALIBRATION BY SLAP TIER")
    tiers = [(90,99,'90-99'),(80,89.9,'80-89'),(70,79.9,'70-79'),
             (60,69.9,'60-69'),(50,59.9,'50-59'),(40,49.9,'40-49'),(1,39.9,'<40')]
    bc = [(l,c) for l,c,t in outs if t=='binary']
    cc = [(l,c) for l,c,t in outs if t=='continuous']
    hdr = f"  {'Tier':<7} {'N':>4}"
    for l,_ in bc: hdr += f" {l.split(' (')[0]+'%':>9}"
    for l,_ in cc: hdr += f" {l[:8]:>9}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")
    for lo,hi,label in tiers:
        tier = pos_bt[(pos_bt['slap_v5'] >= lo) & (pos_bt['slap_v5'] <= hi)]
        if len(tier) == 0: continue
        row = f"  {label:<7} {len(tier):>4}"
        for _,c in bc:
            vv = tier[tier[c].notna()]
            row += f" {vv[c].mean()*100:>8.1f}%" if len(vv) > 0 else f" {'N/A':>9}"
        for _,c in cc:
            vv = tier[tier[c].notna()]
            row += f" {vv[c].mean():>9.1f}" if len(vv) > 0 else f" {'N/A':>9}"
        print(row)

    # 8: Top-decile
    print(f"\n  8. TOP-DECILE PRECISION (top {max(1,len(pos_bt)//10)} players)")
    nd = max(1, len(pos_bt) // 10)
    st = pos_bt.nlargest(nd, 'slap_v5')
    dt = pos_bt.nlargest(nd, 'dc_only')
    print(f"  {'Metric':<22} {'SLAP top10%':>12} {'DC top10%':>12} {'Wins?':>7}")
    print(f"  {'-'*56}")
    for label,col,ot in outs:
        sv = st[st[col].notna()]
        dv = dt[dt[col].notna()]
        if len(sv) == 0: continue
        if ot == 'binary':
            s_val = sv[col].mean()*100; d_val = dv[col].mean()*100
            w = "YES" if s_val > d_val else "no"
            print(f"  {label+' rate':<22} {s_val:>11.1f}% {d_val:>11.1f}% {w:>7}")
        else:
            s_val = sv[col].mean(); d_val = dv[col].mean()
            w = "YES" if s_val > d_val else "no"
            print(f"  {label+' avg':<22} {s_val:>12.2f} {d_val:>12.2f} {w:>7}")


# ============================================================================
print(f"\n\n{'='*120}")
print("PART 3: KEY STATS FOR USAGE GUIDE")
print("=" * 120)

for pos in ['WR', 'RB', 'TE']:
    pos_bt = bt[bt['position'] == pos]
    print(f"\n  {pos} — Threshold hit rates:")
    for lo,hi,label in [(80,99,'80+'),(60,79.9,'60-79'),(40,59.9,'40-59'),(1,39.9,'<40')]:
        tier = pos_bt[(pos_bt['slap_v5'] >= lo) & (pos_bt['slap_v5'] <= hi)]
        v = tier[tier['nfl_hit24'].notna()]
        if len(v) > 0:
            hr = v['nfl_hit24'].mean()*100
            ppg = tier['nfl_first_3yr_ppg'].mean() if tier['nfl_first_3yr_ppg'].notna().any() else 0
            print(f"    {label:<6}: n={len(tier):>3}, {hr:>5.1f}% hit, {ppg:.1f} avg PPG")

print(f"\n  DELTA VALIDATION:")
for pos in ['WR', 'RB', 'TE']:
    pos_bt = bt[bt['position'] == pos]
    bst = pos_bt[pos_bt['delta_vs_dc'] >= 5]
    dng = pos_bt[pos_bt['delta_vs_dc'] <= -5]
    bp = bst['nfl_first_3yr_ppg'].mean() if bst['nfl_first_3yr_ppg'].notna().any() else 0
    dp = dng['nfl_first_3yr_ppg'].mean() if dng['nfl_first_3yr_ppg'].notna().any() else 0
    bh = bst['nfl_hit24'].mean()*100 if bst['nfl_hit24'].notna().any() else 0
    dh = dng['nfl_hit24'].mean()*100 if dng['nfl_hit24'].notna().any() else 0
    print(f"    {pos}: Boosted(d>=+5,n={len(bst):>3}): {bp:.1f}PPG {bh:.0f}%hit | "
          f"Dinged(d<=-5,n={len(dng):>3}): {dp:.1f}PPG {dh:.0f}%hit | Gap={bp-dp:+.1f}PPG")

# Overall SLAP vs DC win tally
print(f"\n  OVERALL SCORECARD (SLAP vs DC-only):")
total_wins = 0
total_tests = 0
for pos in ['WR', 'RB', 'TE']:
    pos_bt = bt[bt['position'] == pos].copy()
    pos_bt['dc_only'] = pos_bt['pick'].apply(dc_score)
    if pos == 'TE':
        outs_s = [('nfl_hit24','b'),('nfl_hit12','b'),('nfl_first_3yr_ppg','c'),('nfl_career_ppg','c')]
    else:
        outs_s = [('nfl_hit24','b'),('nfl_hit12','b'),('nfl_first_3yr_ppg','c'),('nfl_career_ppg','c')]
    wins = 0
    tests = 0
    for col,ot in outs_s:
        v = pos_bt[pos_bt[col].notna()]
        if len(v) < 10: continue
        tests += 1
        total_tests += 1
        ss = abs(spearmanr(v['slap_v5'], v[col])[0])
        ds = abs(spearmanr(v['dc_only'], v[col])[0])
        if ss > ds:
            wins += 1
            total_wins += 1
    print(f"    {pos}: SLAP wins {wins}/{tests} Spearman comparisons")
print(f"    TOTAL: SLAP wins {total_wins}/{total_tests}")

print(f"\n  DONE.")
