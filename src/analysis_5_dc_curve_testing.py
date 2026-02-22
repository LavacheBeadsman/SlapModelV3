"""Analysis 5: DC Curve Testing - Position-specific DC curves"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import math
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DC CURVES
# ============================================================================
def dc_current(pick):
    dc = 100 - 2.40 * (pick ** 0.62 - 1)
    return max(0, min(100, dc))

def dc_sqrt(pick):
    raw = 1.0 / (pick ** 0.5)
    min_val = 1.0 / (260 ** 0.5)
    max_val = 1.0
    return ((raw - min_val) / (max_val - min_val)) * 100

def dc_steep(pick):
    dc = 100 - 1.80 * (pick ** 0.70 - 1)
    return max(0, min(100, dc))

def dc_linear(pick):
    return max(0, 100 - (pick - 1) * (100 / 259))

def dc_log(pick):
    raw = 1.0 - math.log(pick) / math.log(260)
    return max(0, min(100, raw * 100))

def dc_gentler(pick):
    dc = 100 - 3.00 * (pick ** 0.55 - 1)
    return max(0, min(100, dc))

CURVES = {
    'A: Current (gentler)': dc_current,
    'B: Sqrt (original)':   dc_sqrt,
    'C: Steep power':       dc_steep,
    'D: Linear':            dc_linear,
    'E: Log':               dc_log,
    'F: Even gentler':      dc_gentler,
}

# ============================================================================
# PRODUCTION FUNCTIONS
# ============================================================================
def wr_breakout_score(breakout_age, dominator_pct):
    if breakout_age is None or pd.isna(breakout_age):
        if dominator_pct is not None and pd.notna(dominator_pct):
            return min(35, 15 + (dominator_pct * 1.0))
        return 25
    age_tiers = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}
    base_score = age_tiers.get(int(breakout_age), 20)
    if dominator_pct is not None and pd.notna(dominator_pct) and dominator_pct >= 20:
        bonus = min((dominator_pct - 20) * 0.5, 9.9)
        bonus = max(0, bonus)
    else:
        bonus = 0
    return min(base_score + bonus, 99.9)

def rb_production_score(rec_yards, team_pass_att, draft_age):
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return None
    if pd.isna(draft_age): draft_age = 22
    ratio = rec_yards / team_pass_att
    season_age = draft_age - 1
    age_weight = 1.15 - (0.05 * (season_age - 19))
    age_weight = max(0.85, min(1.15, age_weight))
    raw_score = ratio * age_weight * 100
    return min(99.9, max(0, raw_score / 1.75))

# ============================================================================
# LOAD DATA
# ============================================================================
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
wr_bt = pd.read_csv('data/wr_backtest_expanded_final.csv')
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')

wr = outcomes[outcomes['position'] == 'WR'].merge(
    wr_bt[['player_name','pick','RAS','breakout_age','peak_dominator','draft_year']],
    on=['player_name','draft_year'], how='inner', suffixes=('','_bt'))
wr['pick'] = wr['pick'].fillna(wr['pick_bt'])
wr['prod_score'] = wr.apply(lambda r: wr_breakout_score(r['breakout_age'], r['peak_dominator']), axis=1)
wr['ras_score'] = wr['RAS'].apply(lambda x: x * 10 if pd.notna(x) else 68.9)

rb = outcomes[outcomes['position'] == 'RB'].merge(
    rb_bt[['player_name','pick','age','RAS','rec_yards','team_pass_att','draft_year']],
    on=['player_name','draft_year'], how='inner', suffixes=('','_bt'))
rb['pick'] = rb['pick'].fillna(rb['pick_bt'])
rb['prod_score'] = rb.apply(lambda r: rb_production_score(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb['prod_score'] = rb['prod_score'].fillna(rb['prod_score'].mean())
rb['ras_score'] = rb['RAS'].apply(lambda x: x * 10 if pd.notna(x) else 66.5)

print(f"WR: {len(wr)} players, RB: {len(rb)} players")

# ============================================================================
# PART 1: DC CURVE COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 120)
print("PART 1: DC CURVE COMPARISON TABLE")
print("=" * 120)

picks = [1, 5, 10, 15, 20, 32, 50, 64, 75, 100, 128, 150, 200, 250]
header = f"{'Pick':>6}"
for name in CURVES: header += f" {name[:18]:>18}"
print(header)
print("-" * len(header))
for p in picks:
    line = f"{p:>6}"
    for name, func in CURVES.items():
        line += f" {func(p):>18.1f}"
    print(line)

# ============================================================================
# PART 2: DC-Only Correlation
# ============================================================================
print("\n" + "=" * 120)
print("PART 2: DC-ONLY CORRELATION (no production/athletic)")
print("=" * 120)

def corr_table(df, dc_col, outcomes_list=['hit24','hit12','first_3yr_ppg','career_ppg']):
    results = {}
    for out in outcomes_list:
        valid = df[[dc_col, out]].dropna()
        if len(valid) < 10:
            results[out] = (np.nan, np.nan, len(valid))
        else:
            r, p = pearsonr(valid[dc_col], valid[out])
            results[out] = (r, p, len(valid))
    return results

for pos_name, df in [("WR", wr), ("RB", rb)]:
    print(f"\n--- {pos_name} (N={len(df)}) ---")
    header = f"{'Curve':<25}"
    for out in ['hit24','hit12','first_3yr_ppg','career_ppg']:
        header += f" {out:>18}"
    print(header)
    print("-" * len(header))
    
    best = {'hit24': (0, ''), 'hit12': (0, ''), 'first_3yr_ppg': (0, ''), 'career_ppg': (0, '')}
    
    for cname, cfunc in CURVES.items():
        col = f'dc_{cname[:5]}'
        df[col] = df['pick'].apply(cfunc)
        results = corr_table(df, col)
        line = f"{cname:<25}"
        for out in ['hit24','hit12','first_3yr_ppg','career_ppg']:
            r, p, n = results[out]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            marker = ''
            if abs(r) > abs(best[out][0]):
                best[out] = (r, cname)
            line += f" {r:>7.4f} (p={p:.4f}){sig:>3}"
        tag = " <<<" if 'Current' in cname else ""
        print(line + tag)
    
    print(f"\n  Best DC-only curve per outcome for {pos_name}:")
    for out in ['hit24','hit12','first_3yr_ppg','career_ppg']:
        print(f"    {out:>18}: {best[out][1]} (r={best[out][0]:.4f})")

# ============================================================================
# PART 3: Full SLAP Correlation
# ============================================================================
print("\n" + "=" * 120)
print("PART 3: FULL SLAP CORRELATION (DC + Production + Athletic)")
print("=" * 120)

WR_W = (0.65, 0.20, 0.15)
RB_W = (0.50, 0.35, 0.15)

for pos_name, df, weights in [("WR", wr, WR_W), ("RB", rb, RB_W)]:
    w_dc, w_prod, w_ath = weights
    print(f"\n--- {pos_name} (N={len(df)}, weights={int(w_dc*100)}/{int(w_prod*100)}/{int(w_ath*100)}) ---")
    
    header = f"{'Curve':<25}"
    for out in ['hit24','hit12','first_3yr_ppg','career_ppg']:
        header += f" {out:>18}"
    print(header)
    print("-" * len(header))
    
    best = {'hit24': (0, ''), 'hit12': (0, ''), 'first_3yr_ppg': (0, ''), 'career_ppg': (0, '')}
    
    for cname, cfunc in CURVES.items():
        col_dc = f'dc_{cname[:5]}'
        if col_dc not in df.columns:
            df[col_dc] = df['pick'].apply(cfunc)
        slap_col = f'slap_{cname[:5]}'
        df[slap_col] = w_dc * df[col_dc] + w_prod * df['prod_score'] + w_ath * df['ras_score']
        
        results = corr_table(df, slap_col)
        line = f"{cname:<25}"
        for out in ['hit24','hit12','first_3yr_ppg','career_ppg']:
            r, p, n = results[out]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            if abs(r) > abs(best[out][0]):
                best[out] = (r, cname)
            line += f" {r:>7.4f} (p={p:.4f}){sig:>3}"
        tag = " <<<" if 'Current' in cname else ""
        print(line + tag)
    
    print(f"\n  Best full SLAP curve per outcome for {pos_name}:")
    for out in ['hit24','hit12','first_3yr_ppg','career_ppg']:
        print(f"    {out:>18}: {best[out][1]} (r={best[out][0]:.4f})")

# ============================================================================
# PART 4: Draft Round Tier Breakdown (Current Curve)
# ============================================================================
print("\n" + "=" * 120)
print("PART 4: DRAFT ROUND TIER BREAKDOWN (Current DC Curve)")
print("=" * 120)

for pos_name, df in [("WR", wr), ("RB", rb)]:
    df['dc_score'] = df['pick'].apply(dc_current)
    df['round'] = pd.cut(df['pick'], bins=[0,32,64,100,135,175,224,260], 
                          labels=['Rd1','Rd2','Rd3','Rd4','Rd5','Rd6','Rd7'])
    
    print(f"\n--- {pos_name} ---")
    print(f"{'Round':<8} {'N':>4} {'Hit24%':>8} {'Hit12%':>8} {'Avg 3yr PPG':>12} {'Avg Car PPG':>12} "
          f"{'Avg DC':>8} {'Avg Seasons':>12}")
    print("-" * 100)
    
    for rd in ['Rd1','Rd2','Rd3','Rd4','Rd5','Rd6','Rd7']:
        tier = df[df['round'] == rd]
        if len(tier) == 0: continue
        h24 = tier['hit24'].mean() * 100
        h12 = tier['hit12'].mean() * 100
        ppg3 = tier['first_3yr_ppg'].mean()
        ppgc = tier['career_ppg'].mean()
        avgdc = tier['dc_score'].mean()
        avgsn = tier['seasons_played'].mean()
        print(f"{rd:<8} {len(tier):>4} {h24:>7.1f}% {h12:>7.1f}% {ppg3:>12.1f} {ppgc:>12.1f} "
              f"{avgdc:>8.1f} {avgsn:>12.1f}")

# ============================================================================
# PART 5: Position-Specific DC Testing
# ============================================================================
print("\n" + "=" * 120)
print("PART 5: POSITION-SPECIFIC DC CURVE WINNER SUMMARY")
print("=" * 120)

print("\n--- DC-ONLY (which curve best predicts outcomes without production/athletic?) ---")
print(f"{'Outcome':<20} {'Best for WR':<30} {'r':>8} {'Best for RB':<30} {'r':>8} {'Same?':>6}")
print("-" * 110)

for out in ['hit24','hit12','first_3yr_ppg','career_ppg']:
    best_wr = ('', 0)
    best_rb = ('', 0)
    for cname, cfunc in CURVES.items():
        col = f'dc_{cname[:5]}'
        # WR
        valid_wr = wr[[col, out]].dropna()
        if len(valid_wr) >= 10:
            r_wr, _ = pearsonr(valid_wr[col], valid_wr[out])
            if abs(r_wr) > abs(best_wr[1]): best_wr = (cname, r_wr)
        # RB
        valid_rb = rb[[col, out]].dropna()
        if len(valid_rb) >= 10:
            r_rb, _ = pearsonr(valid_rb[col], valid_rb[out])
            if abs(r_rb) > abs(best_rb[1]): best_rb = (cname, r_rb)
    same = "YES" if best_wr[0] == best_rb[0] else "NO"
    print(f"{out:<20} {best_wr[0]:<30} {best_wr[1]:>8.4f} {best_rb[0]:<30} {best_rb[1]:>8.4f} {same:>6}")

print("\n--- FULL SLAP (which curve best with production + athletic included?) ---")
print(f"{'Outcome':<20} {'Best for WR':<30} {'r':>8} {'Best for RB':<30} {'r':>8} {'Same?':>6}")
print("-" * 110)

for out in ['hit24','hit12','first_3yr_ppg','career_ppg']:
    best_wr = ('', 0)
    best_rb = ('', 0)
    for cname, cfunc in CURVES.items():
        slap_col = f'slap_{cname[:5]}'
        # WR
        valid_wr = wr[[slap_col, out]].dropna()
        if len(valid_wr) >= 10:
            r_wr, _ = pearsonr(valid_wr[slap_col], valid_wr[out])
            if abs(r_wr) > abs(best_wr[1]): best_wr = (cname, r_wr)
        # RB
        valid_rb = rb[[slap_col, out]].dropna()
        if len(valid_rb) >= 10:
            r_rb, _ = pearsonr(valid_rb[slap_col], valid_rb[out])
            if abs(r_rb) > abs(best_rb[1]): best_rb = (cname, r_rb)
    same = "YES" if best_wr[0] == best_rb[0] else "NO"
    print(f"{out:<20} {best_wr[0]:<30} {best_wr[1]:>8.4f} {best_rb[0]:<30} {best_rb[1]:>8.4f} {same:>6}")

# ============================================================================
# PART 6: Current curve vs best alternative — magnitude of difference
# ============================================================================
print("\n" + "=" * 120)
print("PART 6: CURRENT CURVE vs BEST ALTERNATIVE — HOW MUCH DOES IT MATTER?")
print("=" * 120)

for pos_name, df in [("WR", wr), ("RB", rb)]:
    print(f"\n--- {pos_name} (Full SLAP) ---")
    current_col = f'slap_A: Cu'
    print(f"{'Outcome':<20} {'Current r':>10} {'Best Curve':<25} {'Best r':>10} {'Delta':>8} {'Verdict'}")
    print("-" * 95)
    for out in ['hit24','hit12','first_3yr_ppg','career_ppg']:
        valid_curr = df[[current_col, out]].dropna()
        r_curr, _ = pearsonr(valid_curr[current_col], valid_curr[out])
        
        best_name, best_r = '', 0
        for cname in CURVES:
            slap_col = f'slap_{cname[:5]}'
            valid = df[[slap_col, out]].dropna()
            if len(valid) >= 10:
                r, _ = pearsonr(valid[slap_col], valid[out])
                if abs(r) > abs(best_r):
                    best_r = r
                    best_name = cname
        
        delta = best_r - r_curr
        verdict = "KEEP current" if abs(delta) < 0.01 else f"Consider {best_name[:10]}"
        print(f"{out:<20} {r_curr:>10.4f} {best_name:<25} {best_r:>10.4f} {delta:>+8.4f} {verdict}")

print("\n" + "=" * 120)
print("END OF ANALYSIS 5")
print("=" * 120)
