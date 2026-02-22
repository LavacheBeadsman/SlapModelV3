"""
TE Outcome Recalculation + Weight Optimization + Full Validation
=================================================================
REPLACES hit24/hit12 with TE-specific thresholds:
  - top6_season: Finished top-6 TE by PPG in ANY first-3 season
  - top12_season: Finished top-12 TE by PPG in ANY first-3 season
  - best_3yr_ppg: BEST single-season PPG in first 3 NFL seasons
  - best_career_ppg: BEST single-season PPG across entire career
  - seasons_over_10ppg: Count of NFL seasons with 10+ PPG

Tests 8-game and 10-game minimums side by side.
Then re-optimizes weights and runs full validation.

LOCKED WEIGHTS (Feb 2026): DC 60% / Breakout 15% / Production 15% / RAS 10%
  - Early Declare dropped: no signal in mid-rounds where rankings matter
  - RAS increased to 10%: appears in all top-10 configs across every analysis
  - 4-component model: DC / Breakout (15% dominator) / Production (Rec/TPA) / RAS
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import glob, os, warnings
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
# STEP 1: LOAD RAW NFL WEEKLY DATA + AGGREGATE TO TE SEASON TOTALS
# ============================================================================

print("=" * 120)
print("TE OUTCOME RECALCULATION — PPG-based rankings with game minimums")
print("=" * 120)

# Load weekly stats
stats_frames = []
for f in sorted(glob.glob('data/nflverse/player_stats_*.parquet')):
    df = pd.read_parquet(f)
    te_df = df[df['position'] == 'TE'][['player_id', 'player_display_name', 'position',
                                         'season', 'week', 'season_type', 'fantasy_points_ppr']].copy()
    stats_frames.append(te_df)

all_csv = pd.read_csv('data/nflverse/player_stats_all_years.csv')
all_te = all_csv[all_csv['position'] == 'TE'][['player_id', 'player_display_name', 'position',
                                                 'season', 'week', 'season_type', 'fantasy_points_ppr']].copy()
stats_frames.append(all_te)

all_stats = pd.concat(stats_frames, ignore_index=True)
all_stats = all_stats.drop_duplicates(subset=['player_id', 'season', 'week', 'season_type'])
all_stats = all_stats[all_stats['season_type'] == 'REG']
print(f"\nLoaded {len(all_stats)} TE regular-season weekly rows")

# Aggregate to season totals
season_totals = all_stats.groupby(['player_id', 'player_display_name', 'season']).agg(
    total_ppr=('fantasy_points_ppr', 'sum'),
    games=('week', 'nunique'),
).reset_index()

season_totals['ppg'] = season_totals['total_ppr'] / season_totals['games']
print(f"TE season-totals: {len(season_totals)} TE-seasons")

# Normalize names for matching
season_totals['name_norm'] = season_totals['player_display_name'].apply(normalize_name)

# ============================================================================
# STEP 2: BUILD PPG RANKINGS AT BOTH GAME MINIMUMS
# ============================================================================

print(f"\n{'='*120}")
print("STEP 2: PPG-BASED POSITIONAL RANKINGS (8-game and 10-game minimums)")
print(f"{'='*120}")

for min_games in [8, 10]:
    qualified = season_totals[season_totals['games'] >= min_games].copy()
    qualified[f'ppg_rank_{min_games}g'] = (
        qualified.groupby('season')['ppg']
        .rank(ascending=False, method='min')
    )
    # Merge back
    season_totals = season_totals.merge(
        qualified[['player_id', 'season', f'ppg_rank_{min_games}g']],
        on=['player_id', 'season'], how='left'
    )
    n_qual = len(qualified)
    avg_per_season = qualified.groupby('season').size().mean()
    print(f"  {min_games}-game minimum: {n_qual} qualified TE-seasons, avg {avg_per_season:.0f}/season")

# Show top-6 and top-12 PPG thresholds by year
print(f"\n  PPG thresholds by season (10-game minimum):")
print(f"  {'Season':>6} {'#6 PPG':>8} {'#12 PPG':>9} {'#24 PPG':>9}")
print(f"  {'-'*35}")
for yr in range(2015, 2026):
    yr_data = season_totals[(season_totals['season'] == yr) & (season_totals['ppg_rank_10g'].notna())]
    if len(yr_data) == 0:
        continue
    r6 = yr_data[yr_data['ppg_rank_10g'] <= 6]['ppg'].min() if (yr_data['ppg_rank_10g'] <= 6).any() else np.nan
    r12 = yr_data[yr_data['ppg_rank_10g'] <= 12]['ppg'].min() if (yr_data['ppg_rank_10g'] <= 12).any() else np.nan
    r24 = yr_data[yr_data['ppg_rank_10g'] <= 24]['ppg'].min() if (yr_data['ppg_rank_10g'] <= 24).any() else np.nan
    print(f"  {yr:>6} {r6:>8.1f} {r12:>9.1f} {r24:>9.1f}")


# ============================================================================
# STEP 3: MATCH TO DRAFTED TEs AND COMPUTE NEW OUTCOMES
# ============================================================================

print(f"\n{'='*120}")
print("STEP 3: COMPUTING NEW OUTCOMES FOR ALL 160 BACKTEST TEs")
print(f"{'='*120}")

bt = pd.read_csv('data/te_backtest_master.csv')
draft = pd.read_parquet('data/nflverse/draft_picks.parquet')
te_draft = draft[(draft['position'] == 'TE') & (draft['season'] >= 2015) & (draft['season'] <= 2025)].copy()
te_draft['name_norm'] = te_draft['pfr_player_name'].apply(normalize_name)
# Build gsis_id lookup
gsis_lookup = dict(zip(zip(te_draft['name_norm'], te_draft['season']), te_draft['gsis_id']))

bt['name_norm_match'] = bt['player_name'].apply(normalize_name)

for min_games in [8, 10]:
    suffix = f'_{min_games}g'
    rank_col = f'ppg_rank_{min_games}g'

    new_cols = {
        f'top6{suffix}': [], f'top12{suffix}': [],
        f'best_3yr_ppg{suffix}': [], f'best_career_ppg{suffix}': [],
        f'seasons_over_10ppg{suffix}': [],
        f'best_3yr_rank{suffix}': [],
    }

    for _, te in bt.iterrows():
        name = te['player_name']
        dy = te['draft_year']
        nn = te['name_norm_match']

        # Match by gsis_id first
        gsis_id = gsis_lookup.get((nn, dy))
        if pd.notna(gsis_id):
            ps = season_totals[(season_totals['player_id'] == gsis_id) & (season_totals['season'] >= dy)]
        else:
            ps = season_totals[(season_totals['name_norm'] == nn) & (season_totals['season'] >= dy)]

        if len(ps) == 0:
            for col in new_cols:
                new_cols[col].append(np.nan if 'ppg' in col or 'seasons' in col else 0)
            continue

        # First 3 NFL seasons
        first3 = ps[(ps['season'] >= dy) & (ps['season'] <= dy + 2)]
        # Qualified first-3 seasons (meet game minimum)
        first3_q = first3[first3['games'] >= min_games]

        # top6: did they finish top-6 by PPG in any first-3 season?
        hit6 = 0
        hit12 = 0
        best_3yr_ppg = np.nan
        best_3yr_rank = 999
        if len(first3_q) > 0 and rank_col in first3_q.columns:
            ranked = first3_q[first3_q[rank_col].notna()]
            if len(ranked) > 0:
                best_3yr_rank = ranked[rank_col].min()
                if best_3yr_rank <= 6:
                    hit6 = 1
                if best_3yr_rank <= 12:
                    hit12 = 1
            best_3yr_ppg = first3_q['ppg'].max()

        # Career: best single-season PPG
        career_q = ps[ps['games'] >= min_games]
        best_career_ppg = career_q['ppg'].max() if len(career_q) > 0 else np.nan

        # Seasons over 10 PPG
        seasons_10 = (career_q['ppg'] >= 10).sum() if len(career_q) > 0 else 0

        new_cols[f'top6{suffix}'].append(hit6)
        new_cols[f'top12{suffix}'].append(hit12)
        new_cols[f'best_3yr_ppg{suffix}'].append(best_3yr_ppg)
        new_cols[f'best_career_ppg{suffix}'].append(best_career_ppg)
        new_cols[f'seasons_over_10ppg{suffix}'].append(seasons_10)
        new_cols[f'best_3yr_rank{suffix}'].append(best_3yr_rank)

    for col, vals in new_cols.items():
        bt[col] = vals

# Print comparison
print(f"\n  OUTCOME COMPARISON: 8-game vs 10-game minimums")
print(f"  {'Metric':<30} {'8-game min':>12} {'10-game min':>12}")
print(f"  {'-'*58}")
for metric in ['top6', 'top12']:
    v8 = bt[f'{metric}_8g'].sum()
    v10 = bt[f'{metric}_10g'].sum()
    print(f"  {metric + ' (first 3yr)':<30} {v8:>8}/{len(bt)} {v10:>8}/{len(bt)}")
print(f"  {'seasons_over_10ppg (career)':<30} {bt['seasons_over_10ppg_8g'].sum():>12} {bt['seasons_over_10ppg_10g'].sum():>12}")
print(f"  {'best_3yr_ppg (has data)':<30} {bt['best_3yr_ppg_8g'].notna().sum():>12} {bt['best_3yr_ppg_10g'].notna().sum():>12}")
print(f"  {'best_career_ppg (has data)':<30} {bt['best_career_ppg_8g'].notna().sum():>12} {bt['best_career_ppg_10g'].notna().sum():>12}")

# Hit rates by round
for min_games in [8, 10]:
    suffix = f'_{min_games}g'
    print(f"\n  BASE HIT RATES BY ROUND ({min_games}-game minimum):")
    print(f"  {'Round':>5} {'N':>5} {'Top6':>6} {'Rate':>8} {'Top12':>6} {'Rate':>8} {'Avg Best3yr':>12} {'Avg 10+PPG':>11}")
    print(f"  {'-'*70}")
    for rd in sorted(bt['round'].unique()):
        rd_data = bt[bt['round'] == rd]
        n = len(rd_data)
        h6 = rd_data[f'top6{suffix}'].sum()
        h12 = rd_data[f'top12{suffix}'].sum()
        ppg = rd_data[f'best_3yr_ppg{suffix}'].mean()
        s10 = rd_data[f'seasons_over_10ppg{suffix}'].mean()
        print(f"  {int(rd):>5} {n:>5} {int(h6):>6} {h6/n*100:>7.0f}% {int(h12):>6} {h12/n*100:>7.0f}% {ppg:>12.2f} {s10:>11.1f}")
    total_h6 = bt[f'top6{suffix}'].sum()
    total_h12 = bt[f'top12{suffix}'].sum()
    print(f"  {'ALL':>5} {len(bt):>5} {int(total_h6):>6} {total_h6/len(bt)*100:>7.0f}% {int(total_h12):>6} {total_h12/len(bt)*100:>7.0f}%")


# ============================================================================
# STEP 4: WEIGHT OPTIMIZATION WITH NEW OUTCOMES
# ============================================================================

print(f"\n\n{'='*120}")
print("STEP 4: WEIGHT OPTIMIZATION WITH NEW TE-SPECIFIC OUTCOMES")
print("Using 10-game minimum (more conservative — show 8g comparison at end)")
print(f"{'='*120}")

# Build component scores
def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

bt['s_dc'] = bt['pick'].apply(dc_score)

# Breakout
pff_file_map = {
    'data/receiving_summary (2).csv': 2015, 'data/receiving_summary (3).csv': 2016,
    'data/receiving_summary (4).csv': 2017, 'data/receiving_summary (5).csv': 2018,
    'data/receiving_summary (21).csv': 2019, 'data/receiving_summary (20).csv': 2020,
    'data/receiving_summary (19).csv': 2021, 'data/receiving_summary (18).csv': 2022,
    'data/receiving_summary (17).csv': 2023, 'data/receiving_summary (16).csv': 2024,
    'data/receiving_summary (15).csv': 2025,
}
all_pff = []
for fp, season in pff_file_map.items():
    pff = pd.read_csv(fp); pff['season'] = season; all_pff.append(pff)
pff_all = pd.concat(all_pff, ignore_index=True)
receivers = pff_all[pff_all['position'].isin(['TE', 'WR', 'HB'])].copy()
team_totals_pff = receivers.groupby(['team_name', 'season'])['yards'].sum().reset_index()
team_totals_pff.rename(columns={'yards': 'team_rec_yards'}, inplace=True)
te_pff = receivers[receivers['position'] == 'TE'].copy()
te_pff = te_pff.merge(team_totals_pff, on=['team_name', 'season'], how='left')
te_pff['dominator_pct'] = np.where(te_pff['team_rec_yards'] > 0,
    (te_pff['yards'] / te_pff['team_rec_yards']) * 100, 0)
te_pff['name_norm'] = te_pff['player'].apply(normalize_name)

THRESH = 15
bo_scores = {}
for _, te in bt.iterrows():
    nn = normalize_name(te['player_name'])
    matches = te_pff[(te_pff['name_norm'] == nn) & (te_pff['season'] < te['draft_year'])]
    if len(matches) == 0: continue
    peak_dom = matches['dominator_pct'].max()
    hit_ages = []
    for _, pm in matches.sort_values('season').iterrows():
        sa = te['draft_age'] - (te['draft_year'] - pm['season'])
        if pm['dominator_pct'] >= THRESH: hit_ages.append(sa)
    if hit_ages:
        ba = hit_ages[0]
        base = {18:100, 19:90, 20:75, 21:60, 22:45, 23:30}.get(int(min(ba, 23)), 20)
        if ba > 23: base = 20
        bonus = min((peak_dom - THRESH) * 0.5, 9.9)
        score = min(base + bonus, 99.9)
    else:
        score = min(35, 15 + peak_dom)
    bo_scores[te['player_name']] = score

bt['s_breakout'] = bt['player_name'].map(bo_scores)

# Production hybrid
bt['rec_per_tpa'] = np.where(
    (bt['cfbd_receptions'].notna()) & (bt['cfbd_team_pass_att'].notna()) & (bt['cfbd_team_pass_att'] > 0),
    bt['cfbd_receptions'] / bt['cfbd_team_pass_att'], np.nan)
bt['pff_rpg'] = np.where(
    (bt['pff_receptions'].notna()) & (bt['pff_player_game_count'].notna()) & (bt['pff_player_game_count'] > 0),
    bt['pff_receptions'] / bt['pff_player_game_count'], np.nan)
cv = bt['rec_per_tpa'].dropna(); pv = bt['pff_rpg'].dropna()
bt['cfbd_n'] = np.where(bt['rec_per_tpa'].notna(), (bt['rec_per_tpa']-cv.min())/(cv.max()-cv.min())*100, np.nan)
bt['pff_n'] = np.where(bt['pff_rpg'].notna(), (bt['pff_rpg']-pv.min())/(pv.max()-pv.min())*100, np.nan)
bt['s_production'] = np.where(bt['cfbd_n'].notna(), bt['cfbd_n'], bt['pff_n'])
bt['s_early_dec'] = bt['early_declare'] * 100
bt['s_ras'] = bt['te_ras'].apply(lambda x: x*10 if pd.notna(x) else np.nan)

# Fill missing
for col, avg_col in [('s_breakout','s_breakout_f'), ('s_production','s_production_f'), ('s_ras','s_ras_f')]:
    bt[avg_col] = bt[col].fillna(bt[col].mean())

# Eval sample: 2015-2024
eval_df = bt[bt['draft_year'] < 2025].copy()
print(f"\nEval sample: {len(eval_df)} TEs (2015-2024)")

# Use 10-game minimum outcomes for optimization
NEW_OUTCOMES = ['best_3yr_ppg_10g', 'top12_10g', 'top6_10g', 'best_career_ppg_10g']
NEW_WEIGHTS = {'best_3yr_ppg_10g': 0.40, 'top12_10g': 0.25, 'top6_10g': 0.20, 'best_career_ppg_10g': 0.15}

print(f"  top6_10g:  {eval_df['top6_10g'].sum():.0f}/{len(eval_df)} ({eval_df['top6_10g'].mean():.1%})")
print(f"  top12_10g: {eval_df['top12_10g'].sum():.0f}/{len(eval_df)} ({eval_df['top12_10g'].mean():.1%})")
print(f"  best_3yr_ppg_10g: {eval_df['best_3yr_ppg_10g'].notna().sum()} with data")
print(f"  best_career_ppg_10g: {eval_df['best_career_ppg_10g'].notna().sum()} with data")
print(f"  seasons_over_10ppg_10g: mean={eval_df['seasons_over_10ppg_10g'].mean():.2f}")


# --- Partial correlations with new outcomes ---
print(f"\n  PARTIAL CORRELATIONS (controlling for DC, new outcomes):")
def partial_corr(x, y, z):
    valid = pd.DataFrame({'x':x,'y':y,'z':z}).dropna()
    if len(valid) < 15: return np.nan, np.nan, len(valid)
    sx,ix,_,_,_ = stats.linregress(valid['z'], valid['x'])
    rx = valid['x'] - (sx*valid['z']+ix)
    sy,iy,_,_,_ = stats.linregress(valid['z'], valid['y'])
    ry = valid['y'] - (sy*valid['z']+iy)
    r,p = stats.pearsonr(rx, ry)
    return r, p, len(valid)

components = {'Breakout (15%)':'s_breakout_f', 'Production':'s_production_f',
              'Early Declare':'s_early_dec', 'RAS':'s_ras_f'}
print(f"  {'Component':<20}", end="")
for out in NEW_OUTCOMES: print(f" {'r('+out.replace('_10g','').replace('best_','')[:12]+')':>16}", end="")
print(f" {'PRI-AVG':>10}")
print(f"  {'-'*90}")
for name, col in components.items():
    s = f"  {name:<20}"
    pri = 0; pt = 0
    for out in NEW_OUTCOMES:
        r,p,n = partial_corr(eval_df[col], eval_df[out], eval_df['s_dc'])
        if not np.isnan(r):
            sig = "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else " "))
            s += f" {r:>+.3f}{sig}(N={n:>3})"
            pri += NEW_WEIGHTS[out]*r; pt += NEW_WEIGHTS[out]
        else: s += f" {'N/A':>16}"
    pavg = pri/pt if pt>0 else np.nan
    s += f" {pavg:>+.4f}" if not np.isnan(pavg) else f" {'N/A':>10}"
    print(s)


def evaluate_config(df, label, comp_weights, outcome_cols, outcome_wts):
    df = df.copy()
    df['slap'] = sum(df[c]*w for c,w in comp_weights.items()).clip(0, 100)
    results = {}; pri_s = 0; pri_t = 0
    for out in outcome_cols:
        v = df[['slap', out]].dropna()
        if len(v) >= 10:
            r,p = stats.pearsonr(v['slap'], v[out])
            results[out] = {'r':r,'p':p,'n':len(v)}
            pri_s += outcome_wts[out]*r; pri_t += outcome_wts[out]
    pri_avg = pri_s/pri_t if pri_t>0 else np.nan
    n_top = max(1, len(df)//10)
    top = df.nlargest(n_top, 'slap')
    h6 = top[f'top6_10g'].mean()*100 if top[f'top6_10g'].notna().any() else np.nan
    h12 = top[f'top12_10g'].mean()*100 if top[f'top12_10g'].notna().any() else np.nan
    top_ppg = top[top['best_3yr_ppg_10g'].notna()]['best_3yr_ppg_10g'].mean()
    df['dc_rank'] = df['s_dc'].rank(ascending=False, method='min')
    df['slap_rank'] = df['slap'].rank(ascending=False, method='min')
    dis10 = int((df['dc_rank'] - df['slap_rank']).abs().ge(10).sum())
    # Boost-ding
    df['rd'] = df['dc_rank'] - df['slap_rank']
    boosted = df[df['rd'] > 5]; dinged = df[df['rd'] < -5]
    bp = boosted[boosted['best_3yr_ppg_10g'].notna()]['best_3yr_ppg_10g'].mean()
    dp_val = dinged[dinged['best_3yr_ppg_10g'].notna()]['best_3yr_ppg_10g'].mean()
    return {'label':label, 'pri_avg':pri_avg, 'outcomes':results, 'top6_top10':h6,
            'top12_top10':h12, 'ppg_top10':top_ppg, 'dis10':dis10, 'n_top':n_top,
            'boost_ppg':bp, 'ding_ppg':dp_val}

# Grid search
configs = [
    ('DC only', {'s_dc':1.00}),
    ('DC/BO: 80/20', {'s_dc':0.80, 's_breakout_f':0.20}),
    ('DC/BO: 75/25', {'s_dc':0.75, 's_breakout_f':0.25}),
    ('DC/BO: 70/30', {'s_dc':0.70, 's_breakout_f':0.30}),
    ('DC/BO/Prod: 75/15/10', {'s_dc':0.75, 's_breakout_f':0.15, 's_production_f':0.10}),
    ('DC/BO/Prod: 70/20/10', {'s_dc':0.70, 's_breakout_f':0.20, 's_production_f':0.10}),
    ('DC/BO/Prod: 70/15/15', {'s_dc':0.70, 's_breakout_f':0.15, 's_production_f':0.15}),
    ('DC/BO/Prod: 65/20/15', {'s_dc':0.65, 's_breakout_f':0.20, 's_production_f':0.15}),
    ('DC/BO/Prod: 65/15/20', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.20}),
    ('DC/BO/Prod: 60/20/20', {'s_dc':0.60, 's_breakout_f':0.20, 's_production_f':0.20}),
    ('LOCKED: 60/15/15/10', {'s_dc':0.60, 's_breakout_f':0.15, 's_production_f':0.15, 's_ras_f':0.10}),
    ('DC/BO/Prod/RAS: 65/15/15/5', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.15, 's_ras_f':0.05}),
    ('DC/BO/Prod/RAS: 65/20/10/5', {'s_dc':0.65, 's_breakout_f':0.20, 's_production_f':0.10, 's_ras_f':0.05}),
    ('DC/BO/Prod/RAS: 65/15/10/10', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.10, 's_ras_f':0.10}),
    ('DC/BO/Prod/RAS: 70/15/10/5', {'s_dc':0.70, 's_breakout_f':0.15, 's_production_f':0.10, 's_ras_f':0.05}),
    ('OLD: 65/15/10/5/5', {'s_dc':0.65, 's_breakout_f':0.15, 's_production_f':0.10, 's_early_dec':0.05, 's_ras_f':0.05}),
]

all_results = []
for label, cw in configs:
    r = evaluate_config(eval_df, label, cw, NEW_OUTCOMES, NEW_WEIGHTS)
    all_results.append(r)

all_sorted = sorted(all_results, key=lambda x: x['pri_avg'] if not np.isnan(x['pri_avg']) else -999, reverse=True)
dc_base = [r for r in all_results if r['label'] == 'DC only'][0]['pri_avg']

print(f"\n{'Rk':>3} {'Config':<40} {'PRI-AVG':>8} {'Δ DC':>7} {'r(3yr)':>8} {'r(t12)':>8} {'r(t6)':>8} {'r(cpg)':>8} {'Top10%t12':>10} {'Top10%t6':>9} {'Dis10+':>7}")
print("-"*130)
for i, r in enumerate(all_sorted, 1):
    o = r['outcomes']
    r3 = o.get('best_3yr_ppg_10g',{}).get('r',np.nan)
    rt12 = o.get('top12_10g',{}).get('r',np.nan)
    rt6 = o.get('top6_10g',{}).get('r',np.nan)
    rcpg = o.get('best_career_ppg_10g',{}).get('r',np.nan)
    d = r['pri_avg'] - dc_base
    mk = ""
    if r['label'] == 'DC only': mk = " BASELINE"
    elif 'CURRENT' in r['label']: mk = " <<<CURRENT"
    elif i == 1: mk = " <<<BEST"
    print(f"{i:>3}. {r['label']:<40} {r['pri_avg']:>+.4f} {d:>+.4f} {r3:>+.4f} {rt12:>+.4f} {rt6:>+.4f} {rcpg:>+.4f} {r['top12_top10']:>9.1f}% {r['top6_top10']:>8.1f}% {r['dis10']:>7}{mk}")


# SLSQP optimization
def objective_fn(weights, comp_cols, df, oc, ow):
    df = df.copy()
    df['slap'] = sum(df[c]*w for c,w in zip(comp_cols, weights)).clip(0,100)
    ps=0; pt=0
    for out in oc:
        v = df[['slap',out]].dropna()
        if len(v)>=10:
            r,_ = stats.pearsonr(v['slap'],v[out])
            ps += ow[out]*r; pt += ow[out]
    return -(ps/pt) if pt>0 else 0

print(f"\n\n--- SLSQP OPTIMIZATION ---")
comp5 = ['s_dc', 's_breakout_f', 's_production_f', 's_early_dec', 's_ras_f']
bounds5 = [(0.45,0.80),(0.05,0.30),(0.05,0.25),(0.00,0.15),(0.00,0.10)]
starts5 = [[0.65,0.15,0.10,0.05,0.05],[0.60,0.20,0.10,0.05,0.05],[0.70,0.15,0.05,0.05,0.05],
           [0.55,0.20,0.15,0.05,0.05],[0.65,0.15,0.15,0.05,0.00],[0.60,0.15,0.15,0.05,0.05]]
constraints = [{'type':'eq','fun':lambda w: sum(w)-1.0}]

best_res = None; best_pri = -999
for x0 in starts5:
    try:
        res = minimize(objective_fn, x0=x0, args=(comp5, eval_df, NEW_OUTCOMES, NEW_WEIGHTS),
                      method='SLSQP', bounds=bounds5, constraints=constraints, options={'maxiter':1000,'ftol':1e-8})
        if res.success and -res.fun > best_pri: best_pri = -res.fun; best_res = res
    except: pass

if best_res is not None:
    w_raw = best_res.x
    w_rnd = [round(w*20)/20 for w in w_raw]
    w_rnd[0] += 1.0 - sum(w_rnd)
    print(f"  SLSQP 5-comp optimal (raw): {' / '.join(f'{w:.3f}' for w in w_raw)}")
    print(f"  SLSQP 5-comp rounded:       {' / '.join(f'{w:.0%}' for w in w_rnd)}")
    print(f"  PRI-AVG (raw): {best_pri:+.4f}")
    cw_opt = dict(zip(comp5, w_rnd))
    r_opt = evaluate_config(eval_df, "SLSQP optimal", cw_opt, NEW_OUTCOMES, NEW_WEIGHTS)
    print(f"  PRI-AVG (rounded): {r_opt['pri_avg']:+.4f}")


# ============================================================================
# STEP 5: FULL VALIDATION WITH NEW OUTCOMES (using CURRENT 65/15/10/5/5)
# ============================================================================

print(f"\n\n{'='*120}")
print("STEP 5: FULL VALIDATION — LOCKED 60/15/15/10 (DC/BO/Prod/RAS) with TE-specific outcomes (10g)")
print(f"{'='*120}")

eval_df['slap'] = (eval_df['s_dc']*0.60 + eval_df['s_breakout_f']*0.15 +
                   eval_df['s_production_f']*0.15 + eval_df['s_ras_f']*0.10).clip(0,100)
eval_df['slap_dc'] = eval_df['s_dc']

# --- Brier Scores ---
def brier(pred, actual):
    v = pd.DataFrame({'p':pred,'a':actual}).dropna()
    return ((v['p']-v['a'])**2).mean(), len(v)

print(f"\n  1. BRIER SCORES (lower=better):")
eval_df['sp'] = eval_df['slap']/100; eval_df['dp'] = eval_df['slap_dc']/100
for out in ['top12_10g', 'top6_10g']:
    bs,n = brier(eval_df['sp'], eval_df[out]); bd,_ = brier(eval_df['dp'], eval_df[out])
    w = "SLAP" if bs<bd else "DC"
    print(f"    {out:<20} SLAP={bs:.4f}  DC={bd:.4f}  Δ={bs-bd:+.4f}  {w}")

# --- AUC-ROC ---
def auc_roc(scores, actual):
    v = pd.DataFrame({'s':scores,'a':actual}).dropna()
    pos = v[v['a']==1]['s'].values; neg = v[v['a']==0]['s'].values
    if len(pos)==0 or len(neg)==0: return np.nan
    u = sum((neg<p).sum()+0.5*(neg==p).sum() for p in pos)
    return u/(len(pos)*len(neg))

print(f"\n  2. AUC-ROC (higher=better):")
for out in ['top12_10g', 'top6_10g']:
    as_ = auc_roc(eval_df['slap'], eval_df[out]); ad = auc_roc(eval_df['slap_dc'], eval_df[out])
    w = "SLAP" if as_>ad else "DC"
    print(f"    {out:<20} SLAP={as_:.4f}  DC={ad:.4f}  Δ={as_-ad:+.4f}  {w}")

# --- Bootstrap ---
print(f"\n  3. BOOTSTRAP (1000 resamples):")
np.random.seed(42); N_BOOT = 1000
for out in NEW_OUTCOMES:
    v = eval_df[['slap','slap_dc',out]].dropna()
    if len(v) < 20: continue
    rs,_=stats.pearsonr(v['slap'],v[out]); rd_,_=stats.pearsonr(v['slap_dc'],v[out])
    wins = 0
    bs_s = []; bs_d = []
    for _ in range(N_BOOT):
        s = v.sample(n=len(v), replace=True)
        try:
            r1,_ = stats.pearsonr(s['slap'],s[out]); r2,_ = stats.pearsonr(s['slap_dc'],s[out])
            bs_s.append(r1); bs_d.append(r2)
            if r1>r2: wins+=1
        except: pass
    ci_s = (np.percentile(bs_s,2.5), np.percentile(bs_s,97.5))
    ci_d = (np.percentile(bs_d,2.5), np.percentile(bs_d,97.5))
    pct = wins/len(bs_s)*100
    olbl = out.replace('_10g','').replace('best_','')
    print(f"    {olbl:<18} SLAP={rs:+.4f} [{ci_s[0]:+.4f},{ci_s[1]:+.4f}]  DC={rd_:+.4f} [{ci_d[0]:+.4f},{ci_d[1]:+.4f}]  SLAP>{pct:.0f}%")

# PRI-AVG bootstrap
pri_s_pt = sum(NEW_WEIGHTS[o]*stats.pearsonr(eval_df[['slap',o]].dropna()['slap'],eval_df[['slap',o]].dropna()[o])[0] for o in NEW_OUTCOMES if eval_df[o].notna().sum()>=10)
pri_d_pt = sum(NEW_WEIGHTS[o]*stats.pearsonr(eval_df[['slap_dc',o]].dropna()['slap_dc'],eval_df[['slap_dc',o]].dropna()[o])[0] for o in NEW_OUTCOMES if eval_df[o].notna().sum()>=10)
bpri_s=[]; bpri_d=[]; pwins=0
for _ in range(N_BOOT):
    s = eval_df.sample(n=len(eval_df), replace=True)
    ps=0;pd_=0;pt=0
    for o in NEW_OUTCOMES:
        v = s[['slap','slap_dc',o]].dropna()
        if len(v)>=10:
            try:
                r1,_=stats.pearsonr(v['slap'],v[o]); r2,_=stats.pearsonr(v['slap_dc'],v[o])
                ps+=NEW_WEIGHTS[o]*r1; pd_+=NEW_WEIGHTS[o]*r2; pt+=NEW_WEIGHTS[o]
            except: pass
    if pt>0: bpri_s.append(ps/pt); bpri_d.append(pd_/pt)
    if pt>0 and ps/pt>pd_/pt: pwins+=1
ci_ps = (np.percentile(bpri_s,2.5), np.percentile(bpri_s,97.5))
ci_pd = (np.percentile(bpri_d,2.5), np.percentile(bpri_d,97.5))
print(f"    {'PRI-AVG':<18} SLAP={pri_s_pt:+.4f} [{ci_ps[0]:+.4f},{ci_ps[1]:+.4f}]  DC={pri_d_pt:+.4f} [{ci_pd[0]:+.4f},{ci_pd[1]:+.4f}]  SLAP>{pwins/len(bpri_s)*100:.0f}%")


# --- Tier Hit Rates ---
print(f"\n  5. TIER HIT RATE TABLE:")
tiers = [(90,101,'90+'),(80,90,'80-89'),(70,80,'70-79'),(60,70,'60-69'),(50,60,'50-59'),(0,50,'Below 50')]

for model, score_col in [('SLAP (60/15/15/10)','slap'), ('DC-only','slap_dc')]:
    print(f"\n    {model}:")
    print(f"    {'Tier':<12} {'N':>4} {'Top6':>5} {'Rate':>7} {'Top12':>5} {'Rate':>7} {'Best3yrPPG':>11} {'Szn10+':>7}")
    print(f"    {'-'*62}")
    for lo,hi,label in tiers:
        t = eval_df[(eval_df[score_col]>=lo)&(eval_df[score_col]<hi)]
        if len(t)==0: continue
        h6=int(t['top6_10g'].sum()); h12=int(t['top12_10g'].sum())
        ppg=t[t['best_3yr_ppg_10g'].notna()]['best_3yr_ppg_10g'].mean()
        s10=t['seasons_over_10ppg_10g'].mean()
        print(f"    {label:<12} {len(t):>4} {h6:>5} {h6/len(t)*100:>6.0f}% {h12:>5} {h12/len(t)*100:>6.0f}% {ppg:>11.2f} {s10:>7.1f}")


# --- Top disagreements ---
print(f"\n  6. TOP DISAGREEMENTS:")
eval_df['delta'] = eval_df['slap'] - eval_df['slap_dc']
eval_df['slap_rank'] = eval_df['slap'].rank(ascending=False, method='min')
eval_df['dc_rank'] = eval_df['slap_dc'].rank(ascending=False, method='min')
eval_df['rank_delta'] = eval_df['dc_rank'] - eval_df['slap_rank']

for direction, n_largest, title in [
    ('positive', True, 'BOOSTED (model likes more than DC)'),
    ('negative', False, 'DINGED (model likes less than DC)'),
]:
    subset = eval_df.nlargest(10, 'delta') if n_largest else eval_df.nsmallest(10, 'delta')
    print(f"\n    {title}:")
    print(f"    {'Player':<25s} {'Yr':>4} {'Rd':>3} {'Pick':>4} {'DC':>5} {'SLAP':>6} {'Δ':>6} {'t6':>3} {'t12':>4} {'3yrPPG':>7} {'Szn10+':>6} {'Verdict':>8}")
    print(f"    {'-'*95}")
    for _,r in subset.iterrows():
        ppg = f"{r['best_3yr_ppg_10g']:.1f}" if pd.notna(r['best_3yr_ppg_10g']) else "N/A"
        t6 = int(r['top6_10g']) if pd.notna(r['top6_10g']) else 0
        t12 = int(r['top12_10g']) if pd.notna(r['top12_10g']) else 0
        szn10 = int(r['seasons_over_10ppg_10g']) if pd.notna(r['seasons_over_10ppg_10g']) else 0
        is_correct = (t12==1) if n_largest else (t12==0)
        verdict = "CORRECT" if is_correct else "wrong"
        print(f"    {r['player_name']:<25s} {int(r['draft_year']):>4} {int(r['round']):>3} {int(r['pick']):>4} "
              f"{r['slap_dc']:>5.1f} {r['slap']:>6.1f} {r['delta']:>+5.1f} {t6:>3} {t12:>4} "
              f"{ppg:>7} {szn10:>6} {verdict:>8}")


# --- Head-to-head summary ---
print(f"\n\n  7. HEAD-TO-HEAD SUMMARY:")
print(f"  {'Metric':<45} {'SLAP':>10} {'DC':>10} {'Winner':>8}")
print(f"  {'-'*78}")
slap_wins = 0; total_m = 0
for out in NEW_OUTCOMES:
    v = eval_df[['slap','slap_dc',out]].dropna()
    if len(v)<10: continue
    rs,_ = stats.pearsonr(v['slap'],v[out]); rd_,_ = stats.pearsonr(v['slap_dc'],v[out])
    w = "SLAP" if rs>rd_ else "DC"; total_m+=1
    if rs>rd_: slap_wins+=1
    olbl = out.replace('_10g','')
    print(f"  Pearson r ({olbl:20s})           {rs:>+.4f}   {rd_:>+.4f} {w:>8}")

for out in ['top12_10g','top6_10g']:
    as_=auc_roc(eval_df['slap'],eval_df[out]); ad=auc_roc(eval_df['slap_dc'],eval_df[out])
    w="SLAP" if as_>ad else "DC"; total_m+=1
    if (w=="SLAP"): slap_wins+=1
    print(f"  AUC-ROC ({out.replace('_10g',''):20s})            {as_:>.4f}    {ad:>.4f} {w:>8}")

for out in ['top12_10g','top6_10g']:
    bs,_=brier(eval_df['sp'],eval_df[out]); bd,_=brier(eval_df['dp'],eval_df[out])
    w="SLAP" if bs<bd else "DC"; total_m+=1
    if (w=="SLAP"): slap_wins+=1
    print(f"  Brier ({out.replace('_10g',''):20s}) lower=better  {bs:>.4f}    {bd:>.4f} {w:>8}")

n_top=max(1,len(eval_df)//10)
ts=eval_df.nlargest(n_top,'slap'); td=eval_df.nlargest(n_top,'slap_dc')
for out in ['top12_10g','top6_10g']:
    hs=ts[out].mean()*100; hd=td[out].mean()*100
    w="SLAP" if hs>hd else "DC" if hs<hd else "TIE"; total_m+=1
    if hs>hd: slap_wins+=1
    print(f"  Top 10% {out.replace('_10g',''):<20} rate         {hs:>8.1f}%   {hd:>8.1f}% {w:>7}")

ppg_s = ts[ts['best_3yr_ppg_10g'].notna()]['best_3yr_ppg_10g'].mean()
ppg_d = td[td['best_3yr_ppg_10g'].notna()]['best_3yr_ppg_10g'].mean()
w = "SLAP" if ppg_s>ppg_d else "DC"; total_m+=1
if ppg_s>ppg_d: slap_wins+=1
print(f"  Top 10% best 3yr PPG                         {ppg_s:>9.2f}   {ppg_d:>9.2f} {w:>7}")

print(f"\n  SLAP wins {slap_wins} of {total_m} metrics vs DC-only.")


# ============================================================================
# STEP 6: 8-GAME vs 10-GAME COMPARISON
# ============================================================================

print(f"\n\n{'='*120}")
print("STEP 6: 8-GAME vs 10-GAME MINIMUM — SIDE BY SIDE")
print("Same LOCKED 60/15/15/10 weights, different game minimums")
print(f"{'='*120}")

for mg in [8, 10]:
    sfx = f'_{mg}g'
    oc = [f'best_3yr_ppg{sfx}', f'top12{sfx}', f'top6{sfx}', f'best_career_ppg{sfx}']
    ow = {oc[0]:0.40, oc[1]:0.25, oc[2]:0.20, oc[3]:0.15}
    r_s = evaluate_config(eval_df, f"SLAP {mg}g", {'s_dc':0.60,'s_breakout_f':0.15,'s_production_f':0.15,'s_ras_f':0.10}, oc, ow)
    r_d = evaluate_config(eval_df, f"DC {mg}g", {'s_dc':1.00}, oc, ow)
    print(f"\n  {mg}-GAME MINIMUM:")
    print(f"    top6 hits: {eval_df[f'top6{sfx}'].sum():.0f}/{len(eval_df)} ({eval_df[f'top6{sfx}'].mean():.1%})")
    print(f"    top12 hits: {eval_df[f'top12{sfx}'].sum():.0f}/{len(eval_df)} ({eval_df[f'top12{sfx}'].mean():.1%})")
    print(f"    SLAP PRI-AVG: {r_s['pri_avg']:+.4f}  |  DC PRI-AVG: {r_d['pri_avg']:+.4f}  |  Δ: {r_s['pri_avg']-r_d['pri_avg']:+.4f}")
    print(f"    Top 10% top12: SLAP={r_s['top12_top10']:.1f}%  DC={r_d['top12_top10']:.1f}%")
    print(f"    Top 10% top6:  SLAP={r_s['top6_top10']:.1f}%  DC={r_d['top6_top10']:.1f}%")


# ============================================================================
# SAVE UPDATED BACKTEST
# ============================================================================

print(f"\n\n{'='*120}")
print("SAVING UPDATED BACKTEST WITH NEW OUTCOMES")
print(f"{'='*120}")

# Drop temp columns, keep new outcome columns
drop_cols = ['name_norm_match','s_dc','s_breakout','s_production','s_early_dec','s_ras',
             'rec_per_tpa','pff_rpg','cfbd_n','pff_n','s_breakout_f','s_production_f','s_ras_f',
             'slap','slap_dc','sp','dp','delta','slap_rank','dc_rank','rank_delta','rd']
save_bt = bt.drop(columns=[c for c in drop_cols if c in bt.columns], errors='ignore')
save_bt.to_csv('data/te_backtest_master.csv', index=False)
print(f"  Saved te_backtest_master.csv with new outcome columns:")
for sfx in ['_8g', '_10g']:
    print(f"    top6{sfx}, top12{sfx}, best_3yr_ppg{sfx}, best_career_ppg{sfx}, seasons_over_10ppg{sfx}, best_3yr_rank{sfx}")

print(f"\n{'='*120}")
print("COMPLETE")
print(f"{'='*120}")
