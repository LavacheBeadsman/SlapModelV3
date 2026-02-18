"""
RB Deep Analysis — Final signal hunting before model lock.
Part 1: New metric seasons_over_10ppg_3yr (all positions)
Part 2: Open-ended RB analysis (interactions, era, nonlinearity, floor effects)
"""
import pandas as pd
import numpy as np
import warnings, os
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

# ============================================================================
# LOAD DATA
# ============================================================================
rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
wr_bt = pd.read_csv('data/wr_backtest_all_components.csv')
te_bt = pd.read_csv('data/te_backtest_master.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')

# Weekly stats
stats = pd.read_csv('data/nflverse/player_stats_all_years.csv')
parquet_2025 = 'data/nflverse/player_stats_2025.parquet'
if os.path.exists(parquet_2025):
    stats_2025 = pd.read_parquet(parquet_2025)
    if int(stats['season'].max()) < 2025:
        stats = pd.concat([stats, stats_2025], ignore_index=True)
stats_reg = stats[stats['season_type'] == 'REG'].copy()

# Draft picks for linking
draft = pd.read_parquet('data/nflverse/draft_picks.parquet')
draft_rel = draft[(draft['season'] >= 2015) & (draft['season'] <= 2025)]
draft_by_yp = {}
for _, d in draft_rel.iterrows():
    if pd.notna(d['gsis_id']):
        draft_by_yp[(int(d['season']), int(d['pick']))] = d['gsis_id']

# Season-level stats
season_stats = stats_reg.groupby(['player_id', 'season']).agg(
    games=('fantasy_points_ppr', 'count'),
    total_ppr=('fantasy_points_ppr', 'sum'),
).reset_index()
season_stats['ppg'] = season_stats['total_ppr'] / season_stats['games']

# Helpers
def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))

def rb_production_raw(rec_yards, team_pass_att, age):
    if pd.isna(rec_yards) or pd.isna(team_pass_att) or team_pass_att == 0:
        return np.nan
    try: age = float(age)
    except: age = 22.0
    if pd.isna(age): age = 22
    season_age = age - 1
    age_w = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
    return (rec_yards / team_pass_att) * age_w * 100


# ============================================================================
# LINK ALL POSITIONS TO PLAYER IDs
# ============================================================================
for df in [rb_bt, wr_bt, te_bt]:
    df['player_id'] = df.apply(
        lambda r: draft_by_yp.get((int(r['draft_year']), int(r['pick']))), axis=1)

# Compute scores
rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)
rb_bt['prod_raw'] = rb_bt.apply(lambda r: rb_production_raw(r['rec_yards'], r['team_pass_att'], r['age']), axis=1)
rb_bt['prod_raw_filled'] = rb_bt['prod_raw'].fillna(0)
rb_bt['slap_65_35'] = 0.65 * rb_bt['s_dc'] + 0.35 * rb_bt['prod_raw_filled']

wr_bt['s_dc'] = wr_bt['pick'].apply(dc_score)

te_bt['s_dc'] = te_bt['pick'].apply(dc_score)

# Merge existing outcomes
rb_out = outcomes[outcomes['position'] == 'RB'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg', 'hit24', 'hit12']].rename(
    columns={'hit24': 'out_hit24', 'hit12': 'out_hit12'})
rb_bt = rb_bt.merge(rb_out[['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']],
                     on=['player_name', 'draft_year', 'pick'], how='left')

wr_out = outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']]
wr_bt = wr_bt.merge(wr_out, on=['player_name', 'draft_year', 'pick'], how='left')

# ============================================================================
# PART 1: NEW METRIC — seasons_over_10ppg_3yr (8-game minimum)
# ============================================================================
print("=" * 100)
print("PART 1: NEW METRIC — seasons_over_10ppg_3yr")
print("  Count of seasons in first 3 NFL years averaging 10+ PPR PPG (min 8 games)")
print("=" * 100)

def compute_seasons_over_10ppg_3yr(player_id, draft_year, min_games=8):
    if pd.isna(player_id):
        return np.nan
    ps = season_stats[
        (season_stats['player_id'] == player_id) &
        (season_stats['season'] >= draft_year) &
        (season_stats['season'] <= draft_year + 2) &
        (season_stats['games'] >= min_games)
    ]
    if ps.empty:
        return np.nan
    return (ps['ppg'] >= 10.0).sum()

def compute_first_3yr_ppg(player_id, draft_year, min_games=8):
    if pd.isna(player_id):
        return np.nan
    ps = season_stats[
        (season_stats['player_id'] == player_id) &
        (season_stats['season'] >= draft_year) &
        (season_stats['season'] <= draft_year + 2) &
        (season_stats['games'] >= min_games)
    ]
    if ps.empty:
        return np.nan
    return ps['ppg'].max()

# Compute for all positions
for df, pos in [(rb_bt, 'RB'), (wr_bt, 'WR'), (te_bt, 'TE')]:
    df['s10_3yr'] = df.apply(lambda r: compute_seasons_over_10ppg_3yr(r['player_id'], int(r['draft_year'])), axis=1)
    df['ppg_8gm'] = df.apply(lambda r: compute_first_3yr_ppg(r['player_id'], int(r['draft_year'])), axis=1)

# RB Results
print(f"\n  RB (n={len(rb_bt)}):")
has_s10 = rb_bt['s10_3yr'].notna()
print(f"    With data: {has_s10.sum()}")
print(f"    Distribution: {rb_bt.loc[has_s10, 's10_3yr'].value_counts().sort_index().to_dict()}")
m = has_s10
r_dc_s10, _ = pearsonr(rb_bt.loc[m, 's_dc'], rb_bt.loc[m, 's10_3yr'])
r_slap_s10, _ = pearsonr(rb_bt.loc[m, 'slap_65_35'], rb_bt.loc[m, 's10_3yr'])
sr_dc_s10 = spearmanr(rb_bt.loc[m, 's_dc'], rb_bt.loc[m, 's10_3yr'])[0]
sr_slap_s10 = spearmanr(rb_bt.loc[m, 'slap_65_35'], rb_bt.loc[m, 's10_3yr'])[0]
print(f"    DC:   Pearson r={r_dc_s10:.4f}, Spearman r={sr_dc_s10:.4f}")
print(f"    SLAP: Pearson r={r_slap_s10:.4f}, Spearman r={sr_slap_s10:.4f}")
print(f"    Gap:  Pearson {r_slap_s10-r_dc_s10:+.4f}, Spearman {sr_slap_s10-sr_dc_s10:+.4f}")

# Also test first_3yr_ppg at 8gm for RBs (for comparison)
m8 = rb_bt['ppg_8gm'].notna()
r_dc_8, _ = pearsonr(rb_bt.loc[m8, 's_dc'], rb_bt.loc[m8, 'ppg_8gm'])
r_slap_8, _ = pearsonr(rb_bt.loc[m8, 'slap_65_35'], rb_bt.loc[m8, 'ppg_8gm'])
print(f"\n    For comparison — first_3yr_ppg at 8gm (n={m8.sum()}):")
print(f"    DC:   Pearson r={r_dc_8:.4f}")
print(f"    SLAP: Pearson r={r_slap_8:.4f}")
print(f"    Gap:  {r_slap_8-r_dc_8:+.4f}")

# WR Results
print(f"\n  WR (n={len(wr_bt)}):")
has_s10_wr = wr_bt['s10_3yr'].notna()
print(f"    With data: {has_s10_wr.sum()}")
print(f"    Distribution: {wr_bt.loc[has_s10_wr, 's10_3yr'].value_counts().sort_index().to_dict()}")
m = has_s10_wr
r_dc_wr = pearsonr(wr_bt.loc[m, 's_dc'], wr_bt.loc[m, 's10_3yr'])[0]
sr_dc_wr = spearmanr(wr_bt.loc[m, 's_dc'], wr_bt.loc[m, 's10_3yr'])[0]
print(f"    DC:   Pearson r={r_dc_wr:.4f}, Spearman r={sr_dc_wr:.4f}")

# WR SLAP (rough: 70% DC + 30% breakout-placeholder)
# Just test DC for WR since we don't have full SLAP computed here
# The audit already shows SLAP vs DC for WRs

# TE Results
print(f"\n  TE (n={len(te_bt)}):")
has_s10_te = te_bt['s10_3yr'].notna()
print(f"    With data: {has_s10_te.sum()}")
print(f"    Distribution: {te_bt.loc[has_s10_te, 's10_3yr'].value_counts().sort_index().to_dict()}")
m = has_s10_te
r_dc_te = pearsonr(te_bt.loc[m, 's_dc'], te_bt.loc[m, 's10_3yr'])[0]
sr_dc_te = spearmanr(te_bt.loc[m, 's_dc'], te_bt.loc[m, 's10_3yr'])[0]
print(f"    DC:   Pearson r={r_dc_te:.4f}, Spearman r={sr_dc_te:.4f}")

# Cross-position comparison table
print(f"\n\n  CROSS-POSITION: seasons_over_10ppg_3yr")
print(f"  {'Position':<6} {'N':>4} {'DC Pearson':>11} {'DC Spearman':>12}")
print(f"  {'-'*35}")
for pos, df in [('RB', rb_bt), ('WR', wr_bt), ('TE', te_bt)]:
    m = df['s10_3yr'].notna()
    rp = pearsonr(df.loc[m, 's_dc'], df.loc[m, 's10_3yr'])[0]
    rs = spearmanr(df.loc[m, 's_dc'], df.loc[m, 's10_3yr'])[0]
    print(f"  {pos:<6} {m.sum():>4} {rp:>11.4f} {rs:>12.4f}")

# AUC for binary version (1+ seasons over 10 PPG)
print(f"\n  AUC-ROC: any season over 10 PPG in first 3 years (binary)")
for pos, df in [('RB', rb_bt), ('WR', wr_bt), ('TE', te_bt)]:
    m = df['s10_3yr'].notna()
    binary = (df.loc[m, 's10_3yr'] >= 1).astype(int)
    if binary.sum() > 0 and binary.sum() < m.sum():
        auc_dc = roc_auc_score(binary, df.loc[m, 's_dc'])
        print(f"  {pos}: DC AUC={auc_dc:.3f} (base rate: {binary.mean()*100:.0f}%)")
        if pos == 'RB':
            auc_slap = roc_auc_score(binary, df.loc[m, 'slap_65_35'])
            print(f"  {pos}: SLAP AUC={auc_slap:.3f}")


# ============================================================================
# PART 2: OPEN-ENDED RB ANALYSIS
# ============================================================================
print(f"\n\n{'='*100}")
print("PART 2: OPEN-ENDED RB ANALYSIS")
print("=" * 100)

# Use 8gm metric for all RB analysis (our best outcome metric)
rb = rb_bt.copy()
rb['outcome'] = rb['ppg_8gm']
rb['hit'] = rb['hit24']
has_outcome = rb['outcome'].notna()
rb_valid = rb[has_outcome].copy()
n_valid = len(rb_valid)
print(f"  Using first_3yr_ppg (8gm) as outcome. N={n_valid}")

# ============================================================================
# 2A: INTERACTION EFFECTS — Does production matter MORE at certain draft slots?
# ============================================================================
print(f"\n\n{'='*100}")
print("2A: INTERACTION EFFECTS — Production × Draft Slot")
print("=" * 100)

# Split into DC tiers
rb_valid['dc_tier'] = pd.cut(rb_valid['s_dc'], bins=[0, 50, 70, 85, 101],
                              labels=['Low DC (<50)', 'Mid DC (50-70)', 'High DC (70-85)', 'Elite DC (85+)'])

print(f"\n  Production → Outcome correlation BY DC tier:")
print(f"  {'DC Tier':<20} {'N':>4} {'Prod→PPG r':>11} {'p-value':>9} {'Meaning'}")
print(f"  {'-'*70}")
for tier in ['Low DC (<50)', 'Mid DC (50-70)', 'High DC (70-85)', 'Elite DC (85+)']:
    sub = rb_valid[rb_valid['dc_tier'] == tier]
    if len(sub) >= 10:
        r, p = pearsonr(sub['prod_raw_filled'], sub['outcome'])
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {tier:<20} {len(sub):>4} {r:>+11.3f} {p:>9.4f} {sig}")

# Interaction term test
from sklearn.linear_model import LinearRegression
X = rb_valid[['s_dc', 'prod_raw_filled']].copy()
X['interaction'] = X['s_dc'] * X['prod_raw_filled']
y = rb_valid['outcome']
model_main = LinearRegression().fit(X[['s_dc', 'prod_raw_filled']], y)
model_int = LinearRegression().fit(X, y)
print(f"\n  Linear model (DC + Prod): R²={model_main.score(X[['s_dc', 'prod_raw_filled']], y):.4f}")
print(f"  With interaction (DC × Prod): R²={model_int.score(X, y):.4f}")
print(f"  Interaction coefficient: {model_int.coef_[2]:.6f}")
print(f"  R² improvement: {model_int.score(X, y) - model_main.score(X[['s_dc', 'prod_raw_filled']], y):+.4f}")

# ============================================================================
# 2B: ERA EFFECTS — Does production matter differently by era?
# ============================================================================
print(f"\n\n{'='*100}")
print("2B: ERA EFFECTS — 2015-2019 vs 2020-2025")
print("=" * 100)

for era, (y1, y2) in [('2015-2019', (2015, 2019)), ('2020-2025', (2020, 2025))]:
    sub = rb_valid[(rb_valid['draft_year'] >= y1) & (rb_valid['draft_year'] <= y2)]
    if len(sub) < 15:
        print(f"  {era}: N={len(sub)} (too few)")
        continue
    r_dc, p_dc = pearsonr(sub['s_dc'], sub['outcome'])
    r_prod, p_prod = pearsonr(sub['prod_raw_filled'], sub['outcome'])
    r_slap, p_slap = pearsonr(sub['slap_65_35'], sub['outcome'])

    # Partial correlation: prod controlling for DC
    from scipy.stats import pearsonr as pr
    resid_prod = sub['prod_raw_filled'] - LinearRegression().fit(sub[['s_dc']], sub['prod_raw_filled']).predict(sub[['s_dc']])
    resid_out = sub['outcome'] - LinearRegression().fit(sub[['s_dc']], sub['outcome']).predict(sub[['s_dc']])
    partial_r, partial_p = pr(resid_prod, resid_out)

    print(f"\n  {era} (N={len(sub)}):")
    print(f"    DC→PPG:   r={r_dc:+.3f} (p={p_dc:.4f})")
    print(f"    Prod→PPG: r={r_prod:+.3f} (p={p_prod:.4f})")
    print(f"    SLAP→PPG: r={r_slap:+.3f} (p={p_slap:.4f})")
    print(f"    Prod|DC (partial): r={partial_r:+.3f} (p={partial_p:.4f})")
    print(f"    Hit24 rate: {sub['hit'].mean()*100:.0f}%")

# ============================================================================
# 2C: NON-LINEAR EFFECTS — Production thresholds
# ============================================================================
print(f"\n\n{'='*100}")
print("2C: NON-LINEAR EFFECTS — Production threshold analysis")
print("=" * 100)

# Group by production quintiles
rb_valid['prod_quintile'] = pd.qcut(rb_valid['prod_raw_filled'], 5, labels=False, duplicates='drop')

print(f"\n  RB outcomes by PRODUCTION QUINTILE (controlling for nothing):")
print(f"  {'Quintile':>8} {'Prod Range':>15} {'N':>4} {'Hit24%':>7} {'PPG':>6} {'DC avg':>7}")
print(f"  {'-'*55}")
for q in sorted(rb_valid['prod_quintile'].unique()):
    sub = rb_valid[rb_valid['prod_quintile'] == q]
    prod_range = f"{sub['prod_raw_filled'].min():.0f}-{sub['prod_raw_filled'].max():.0f}"
    ppg = sub['outcome'].mean()
    hit = sub['hit'].mean() * 100
    dc_avg = sub['s_dc'].mean()
    print(f"  {q:>8} {prod_range:>15} {len(sub):>4} {hit:>6.0f}% {ppg:>6.1f} {dc_avg:>7.1f}")

# Same but controlling for DC (within DC tiers)
print(f"\n  Production quintile effect WITHIN DC tiers:")
print(f"  (This controls for draft capital — does production add signal at each DC level?)")
for tier in ['High DC (70-85)', 'Mid DC (50-70)', 'Low DC (<50)']:
    sub = rb_valid[rb_valid['dc_tier'] == tier]
    if len(sub) < 20:
        continue
    sub_q = pd.qcut(sub['prod_raw_filled'], 3, labels=['Low Prod', 'Mid Prod', 'High Prod'], duplicates='drop')
    print(f"\n  Within {tier} (N={len(sub)}):")
    print(f"    {'Prod Tier':>10} {'N':>4} {'Hit24%':>7} {'PPG':>6}")
    print(f"    {'-'*30}")
    for plabel in ['Low Prod', 'Mid Prod', 'High Prod']:
        psub = sub[sub_q == plabel]
        if len(psub) >= 3:
            ppg = psub['outcome'].mean()
            hit = psub['hit'].mean() * 100
            print(f"    {plabel:>10} {len(psub):>4} {hit:>6.0f}% {ppg:>6.1f}")

# Specific threshold analysis
print(f"\n  Binary threshold analysis (production above vs below):")
print(f"  {'Threshold':>10} {'N_above':>7} {'N_below':>7} {'PPG_above':>10} {'PPG_below':>10} {'Hit_above':>10} {'Hit_below':>10}")
print(f"  {'-'*75}")
for thresh in [20, 40, 60, 80, 100, 120]:
    above = rb_valid[rb_valid['prod_raw_filled'] >= thresh]
    below = rb_valid[rb_valid['prod_raw_filled'] < thresh]
    if len(above) >= 5 and len(below) >= 5:
        print(f"  {thresh:>10} {len(above):>7} {len(below):>7} "
              f"{above['outcome'].mean():>10.1f} {below['outcome'].mean():>10.1f} "
              f"{above['hit'].mean()*100:>9.0f}% {below['hit'].mean()*100:>9.0f}%")

# ============================================================================
# 2D: COLLEGE RUSHING VOLUME — Interaction with receiving
# ============================================================================
print(f"\n\n{'='*100}")
print("2D: COLLEGE RUSHING — Does it add signal when combined with receiving?")
print("=" * 100)

# Check what rushing data we have
# The RB backtest might not have rushing data directly, but we can check
rush_cols = [c for c in rb_bt.columns if 'rush' in c.lower()]
print(f"  Available rushing columns: {rush_cols}")

# Check if there's a proxy: rec_yards implies non-rushing RB; low rec = pure rusher
# We can test if the ratio of rec/rush matters
# But first, check what data exists from CFBD

# Try alternative: use the raw production as a proxy for "pass-catching back"
# and test if there's a dual-threat signal
# High prod + high DC = dual-threat who got drafted high
# Test: does prod/DC interaction matter?

# ============================================================================
# 2E: FLOOR EFFECT — Does high receiving protect against busting?
# ============================================================================
print(f"\n\n{'='*100}")
print("2E: FLOOR EFFECT — Does high receiving production protect against busting?")
print("=" * 100)

# Define "bust" as never hitting top-24 (hit24 = 0)
# For players with outcome data, check bust rate by production tier
print(f"\n  BUST RATE (hit24=0) by production tier:")
for plabel, low, high in [('Bottom 25%', 0, 25), ('25-50%', 25, 50), ('50-75%', 50, 75), ('Top 25%', 75, 101)]:
    pctile = rb_valid['prod_raw_filled'].rank(pct=True) * 100
    sub = rb_valid[(pctile >= low) & (pctile < high)]
    if len(sub) >= 5:
        bust_rate = 1 - sub['hit'].mean()
        min_ppg = sub['outcome'].min()
        p10_ppg = sub['outcome'].quantile(0.10)
        print(f"  {plabel:<15} N={len(sub):>3} Bust rate: {bust_rate*100:>5.0f}% "
              f"  Floor PPG (p10): {p10_ppg:.1f}  Worst: {min_ppg:.1f}  DC avg: {sub['s_dc'].mean():.1f}")

# Same but within DC tiers (controlling for draft capital)
print(f"\n  BUST RATE by production, WITHIN DC tiers (controlling for draft capital):")
for tier in ['Elite DC (85+)', 'High DC (70-85)', 'Mid DC (50-70)', 'Low DC (<50)']:
    sub = rb_valid[rb_valid['dc_tier'] == tier]
    if len(sub) < 10:
        continue
    median_prod = sub['prod_raw_filled'].median()
    high_prod = sub[sub['prod_raw_filled'] >= median_prod]
    low_prod = sub[sub['prod_raw_filled'] < median_prod]
    if len(high_prod) >= 3 and len(low_prod) >= 3:
        print(f"\n  {tier} (N={len(sub)}):")
        print(f"    High prod (≥{median_prod:.0f}): N={len(high_prod)}, Bust={1-high_prod['hit'].mean():.0%}, "
              f"PPG={high_prod['outcome'].mean():.1f}, Floor(p10)={high_prod['outcome'].quantile(0.10):.1f}")
        print(f"    Low prod  (<{median_prod:.0f}): N={len(low_prod)}, Bust={1-low_prod['hit'].mean():.0%}, "
              f"PPG={low_prod['outcome'].mean():.1f}, Floor(p10)={low_prod['outcome'].quantile(0.10):.1f}")

# ============================================================================
# 2F: LOGISTIC REGRESSION — Production as hit24 predictor
# ============================================================================
print(f"\n\n{'='*100}")
print("2F: LOGISTIC REGRESSION — Production predicting hit24 (controlling for DC)")
print("=" * 100)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

has_hit = rb_valid['hit'].notna()
rb_lr = rb_valid[has_hit].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rb_lr[['s_dc', 'prod_raw_filled']])

# DC only
lr_dc = LogisticRegression(random_state=42, max_iter=1000)
lr_dc.fit(X_scaled[:, :1], rb_lr['hit'])
auc_dc = roc_auc_score(rb_lr['hit'], lr_dc.predict_proba(X_scaled[:, :1])[:, 1])

# DC + Production
lr_both = LogisticRegression(random_state=42, max_iter=1000)
lr_both.fit(X_scaled, rb_lr['hit'])
auc_both = roc_auc_score(rb_lr['hit'], lr_both.predict_proba(X_scaled)[:, 1])

print(f"  DC only:    AUC={auc_dc:.4f}")
print(f"  DC + Prod:  AUC={auc_both:.4f}")
print(f"  Improvement: {auc_both-auc_dc:+.4f}")
print(f"  Prod coefficient (scaled): {lr_both.coef_[0][1]:+.4f}")

# With interaction
X_int = np.column_stack([X_scaled, X_scaled[:, 0] * X_scaled[:, 1]])
lr_int = LogisticRegression(random_state=42, max_iter=1000)
lr_int.fit(X_int, rb_lr['hit'])
auc_int = roc_auc_score(rb_lr['hit'], lr_int.predict_proba(X_int)[:, 1])
print(f"  DC + Prod + Interaction: AUC={auc_int:.4f} (vs {auc_both:.4f} without)")

# ============================================================================
# 2G: ALTERNATIVE OUTCOME — seasons_over_10ppg_3yr
# ============================================================================
print(f"\n\n{'='*100}")
print("2G: SLAP vs DC on seasons_over_10ppg_3yr (the new sustained-production metric)")
print("=" * 100)

m = rb_valid['s10_3yr'].notna()
if m.sum() > 10:
    # Correlation
    r_dc = pearsonr(rb_valid.loc[m, 's_dc'], rb_valid.loc[m, 's10_3yr'])[0]
    r_slap = pearsonr(rb_valid.loc[m, 'slap_65_35'], rb_valid.loc[m, 's10_3yr'])[0]
    sr_dc = spearmanr(rb_valid.loc[m, 's_dc'], rb_valid.loc[m, 's10_3yr'])[0]
    sr_slap = spearmanr(rb_valid.loc[m, 'slap_65_35'], rb_valid.loc[m, 's10_3yr'])[0]

    print(f"  N={m.sum()}")
    print(f"  DC:   Pearson r={r_dc:.4f}, Spearman r={sr_dc:.4f}")
    print(f"  SLAP: Pearson r={r_slap:.4f}, Spearman r={sr_slap:.4f}")
    print(f"  Gap:  Pearson {r_slap-r_dc:+.4f}, Spearman {sr_slap-sr_dc:+.4f}")

    # AUC for binary (1+ seasons)
    binary = (rb_valid.loc[m, 's10_3yr'] >= 1).astype(int)
    auc_dc = roc_auc_score(binary, rb_valid.loc[m, 's_dc'])
    auc_slap = roc_auc_score(binary, rb_valid.loc[m, 'slap_65_35'])
    print(f"\n  AUC (1+ seasons over 10 PPG): DC={auc_dc:.3f}, SLAP={auc_slap:.3f}, gap={auc_slap-auc_dc:+.3f}")

    # AUC for binary (2+ seasons) — truly sustained
    binary2 = (rb_valid.loc[m, 's10_3yr'] >= 2).astype(int)
    if binary2.sum() > 0 and binary2.sum() < m.sum():
        auc_dc2 = roc_auc_score(binary2, rb_valid.loc[m, 's_dc'])
        auc_slap2 = roc_auc_score(binary2, rb_valid.loc[m, 'slap_65_35'])
        print(f"  AUC (2+ seasons over 10 PPG): DC={auc_dc2:.3f}, SLAP={auc_slap2:.3f}, gap={auc_slap2-auc_dc2:+.3f}")

    # By tier
    print(f"\n  seasons_over_10ppg_3yr distribution by DC tier:")
    print(f"  {'DC Tier':<20} {'N':>4} {'Mean':>6} {'0 seasons':>10} {'1 season':>10} {'2+ seasons':>11}")
    print(f"  {'-'*65}")
    for tier in ['Elite DC (85+)', 'High DC (70-85)', 'Mid DC (50-70)', 'Low DC (<50)']:
        sub = rb_valid[m & (rb_valid['dc_tier'] == tier)]
        if len(sub) >= 3:
            s0 = (sub['s10_3yr'] == 0).mean() * 100
            s1 = (sub['s10_3yr'] == 1).mean() * 100
            s2 = (sub['s10_3yr'] >= 2).mean() * 100
            print(f"  {tier:<20} {len(sub):>4} {sub['s10_3yr'].mean():>6.2f} {s0:>9.0f}% {s1:>9.0f}% {s2:>10.0f}%")

# ============================================================================
# 2H: WHAT ABOUT AGE? — Does age interact with production or DC?
# ============================================================================
print(f"\n\n{'='*100}")
print("2H: AGE EFFECTS — Does age at draft add signal beyond DC + Production?")
print("=" * 100)

rb_valid['age_clean'] = pd.to_numeric(rb_valid['age'], errors='coerce')
has_age = rb_valid['age_clean'].notna() & has_outcome
if has_age.sum() > 30:
    sub = rb_valid[has_age]
    r_age, p_age = pearsonr(sub['age_clean'], sub['outcome'])
    print(f"  Age → PPG: r={r_age:+.3f} (p={p_age:.4f})")

    # Partial correlation controlling for DC
    resid_age = sub['age_clean'] - LinearRegression().fit(sub[['s_dc']], sub['age_clean']).predict(sub[['s_dc']])
    resid_out = sub['outcome'] - LinearRegression().fit(sub[['s_dc']], sub['outcome']).predict(sub[['s_dc']])
    partial_r_age, partial_p_age = pearsonr(resid_age, resid_out)
    print(f"  Age|DC (partial): r={partial_r_age:+.3f} (p={partial_p_age:.4f})")

    # Does age interact with production?
    r_age_prod = pearsonr(sub['age_clean'] * sub['prod_raw_filled'], sub['outcome'])[0]
    print(f"  Age × Prod interaction: r={r_age_prod:+.3f}")

    # Young vs old RBs: does production matter more for one group?
    young = sub[sub['age_clean'] <= 21]
    old = sub[sub['age_clean'] > 21]
    if len(young) >= 10 and len(old) >= 10:
        r_young = pearsonr(young['prod_raw_filled'], young['outcome'])[0]
        r_old = pearsonr(old['prod_raw_filled'], old['outcome'])[0]
        print(f"\n  Production → PPG for young (≤21) RBs: r={r_young:+.3f} (N={len(young)})")
        print(f"  Production → PPG for old (>21) RBs:   r={r_old:+.3f} (N={len(old)})")

# ============================================================================
# 2I: RECEIVING YARDS RAW — Does the raw number matter separately from TPA?
# ============================================================================
print(f"\n\n{'='*100}")
print("2I: DECOMPOSING PRODUCTION — Receiving yards vs Team pass attempts")
print("=" * 100)

sub = rb_valid[rb_valid['rec_yards'].notna() & rb_valid['team_pass_att'].notna()]
if len(sub) >= 20:
    r_ry, p_ry = pearsonr(sub['rec_yards'], sub['outcome'])
    r_tpa, p_tpa = pearsonr(sub['team_pass_att'], sub['outcome'])
    r_ratio = pearsonr(sub['rec_yards'] / sub['team_pass_att'], sub['outcome'])[0]
    r_prod = pearsonr(sub['prod_raw_filled'], sub['outcome'])[0]

    print(f"  N={len(sub)}")
    print(f"  Rec yards alone → PPG:       r={r_ry:+.3f} (p={p_ry:.4f})")
    print(f"  Team pass att alone → PPG:    r={r_tpa:+.3f} (p={p_tpa:.4f})")
    print(f"  Rec/TPA ratio → PPG:          r={r_ratio:+.3f}")
    print(f"  Full production score → PPG:  r={r_prod:+.3f}")

    # Partial correlations controlling for DC
    for label, col in [('Rec yards', 'rec_yards'), ('Rec/TPA', None)]:
        vals = sub['rec_yards'] / sub['team_pass_att'] if col is None else sub[col]
        resid_x = vals - LinearRegression().fit(sub[['s_dc']], vals).predict(sub[['s_dc']])
        resid_y = sub['outcome'] - LinearRegression().fit(sub[['s_dc']], sub['outcome']).predict(sub[['s_dc']])
        pr_r, pr_p = pearsonr(resid_x, resid_y)
        print(f"  {label}|DC (partial): r={pr_r:+.3f} (p={pr_p:.4f})")

# ============================================================================
# 2J: RECEPTIONS (not just yards) — Does catch count matter?
# ============================================================================
print(f"\n\n{'='*100}")
print("2J: RECEPTIONS — Does catch count add signal beyond receiving yards?")
print("=" * 100)

has_rec = rb_valid['receptions'].notna() & rb_valid['rec_yards'].notna()
if has_rec.sum() >= 20:
    sub = rb_valid[has_rec]
    r_recs = pearsonr(sub['receptions'], sub['outcome'])[0]
    r_yards = pearsonr(sub['rec_yards'], sub['outcome'])[0]
    r_ypc = pearsonr(sub['rec_yards'] / sub['receptions'].clip(1), sub['outcome'])[0]

    print(f"  N={has_rec.sum()}")
    print(f"  Receptions → PPG:        r={r_recs:+.3f}")
    print(f"  Rec yards → PPG:         r={r_yards:+.3f}")
    print(f"  Yards/catch → PPG:       r={r_ypc:+.3f}")

    # Receptions per TPA
    sub_tpa = sub[sub['team_pass_att'].notna() & (sub['team_pass_att'] > 0)]
    if len(sub_tpa) >= 20:
        r_rec_tpa = pearsonr(sub_tpa['receptions'] / sub_tpa['team_pass_att'], sub_tpa['outcome'])[0]
        print(f"  Rec/TPA → PPG:           r={r_rec_tpa:+.3f}")

        # Partial controlling for DC
        vals = sub_tpa['receptions'] / sub_tpa['team_pass_att']
        resid_x = vals - LinearRegression().fit(sub_tpa[['s_dc']], vals).predict(sub_tpa[['s_dc']])
        resid_y = sub_tpa['outcome'] - LinearRegression().fit(sub_tpa[['s_dc']], sub_tpa['outcome']).predict(sub_tpa[['s_dc']])
        pr_r, pr_p = pearsonr(resid_x, resid_y)
        print(f"  Rec/TPA|DC (partial):    r={pr_r:+.3f} (p={pr_p:.4f})")

# ============================================================================
# 2K: COLLEGE CONFERENCE — Big school vs small school effect?
# ============================================================================
print(f"\n\n{'='*100}")
print("2K: COLLEGE EFFECT — Does school/conference add signal?")
print("=" * 100)

# Check if we can identify power conferences
p5_schools = ['Alabama', 'Auburn', 'Arkansas', 'Florida', 'Georgia', 'Kentucky', 'LSU',
              'Mississippi State', 'Ole Miss', 'Missouri', 'South Carolina', 'Tennessee',
              'Texas A&M', 'Vanderbilt', 'Ohio State', 'Michigan', 'Michigan State',
              'Penn State', 'Iowa', 'Wisconsin', 'Minnesota', 'Nebraska', 'Purdue',
              'Indiana', 'Illinois', 'Northwestern', 'Rutgers', 'Maryland',
              'Clemson', 'Florida State', 'Miami', 'NC State', 'North Carolina',
              'Virginia', 'Virginia Tech', 'Duke', 'Wake Forest', 'Pittsburgh',
              'Syracuse', 'Boston College', 'Louisville', 'Georgia Tech',
              'Oklahoma', 'Texas', 'Oklahoma State', 'TCU', 'Baylor', 'Kansas State',
              'Iowa State', 'West Virginia', 'Kansas', 'Texas Tech',
              'USC', 'UCLA', 'Oregon', 'Washington', 'Stanford', 'California',
              'Arizona', 'Arizona State', 'Colorado', 'Utah', 'Oregon State', 'Washington State',
              'Notre Dame']

rb_valid['power5'] = rb_valid['college'].isin(p5_schools).astype(int)
p5_count = rb_valid['power5'].sum()
print(f"  Power 5 (+Notre Dame): {p5_count}/{len(rb_valid)} ({p5_count/len(rb_valid)*100:.0f}%)")

p5_sub = rb_valid[rb_valid['power5'] == 1]
non_p5 = rb_valid[rb_valid['power5'] == 0]
if len(p5_sub) >= 10 and len(non_p5) >= 10:
    print(f"  Power 5: N={len(p5_sub)}, Hit24={p5_sub['hit'].mean()*100:.0f}%, PPG={p5_sub['outcome'].mean():.1f}, DC={p5_sub['s_dc'].mean():.1f}")
    print(f"  Non-P5:  N={len(non_p5)}, Hit24={non_p5['hit'].mean()*100:.0f}%, PPG={non_p5['outcome'].mean():.1f}, DC={non_p5['s_dc'].mean():.1f}")

    # Partial: conference controlling for DC
    resid_p5 = rb_valid['power5'] - LinearRegression().fit(rb_valid[['s_dc']], rb_valid['power5']).predict(rb_valid[['s_dc']])
    resid_out = rb_valid['outcome'] - LinearRegression().fit(rb_valid[['s_dc']], rb_valid['outcome']).predict(rb_valid[['s_dc']])
    pr_r, pr_p = pearsonr(resid_p5, resid_out)
    print(f"  Power5|DC (partial): r={pr_r:+.3f} (p={pr_p:.4f})")

# ============================================================================
# SUMMARY OF ALL FINDINGS
# ============================================================================
print(f"\n\n{'='*100}")
print("SUMMARY OF ALL FINDINGS")
print("=" * 100)

print(f"""
  PART 1: seasons_over_10ppg_3yr
    - RB: DC r={r_dc_s10:.3f}, SLAP r={r_slap_s10:.3f}, gap={r_slap_s10-r_dc_s10:+.3f}
    - New metric captures sustained production vs one-year wonders
    - Distribution is heavily zero-inflated (most RBs never sustain 10+ PPG)

  PART 2A: INTERACTION EFFECTS
    - Production matters at different DC levels? (see tier breakdown above)
    - Formal interaction term adds minimal R² improvement

  PART 2B: ERA EFFECTS
    - Production signal may differ by era (check partial correlations above)

  PART 2C: NON-LINEAR EFFECTS
    - Check threshold analysis for step-function patterns

  PART 2E: FLOOR EFFECT
    - Does high receiving protect against busting? (see bust rate analysis)

  PART 2F: LOGISTIC REGRESSION
    - DC AUC={auc_dc:.3f}, DC+Prod AUC={auc_both:.3f}
    - Production adds {auc_both-auc_dc:+.3f} AUC for hit24 prediction

  PART 2H: AGE
    - Tested whether age adds signal beyond DC + production

  PART 2I-J: DECOMPOSING PRODUCTION
    - Rec yards vs TPA ratio vs receptions — which matters most?

  DONE.
""")
