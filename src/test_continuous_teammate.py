"""
Test 4 continuous teammate score formulas vs current binary (0/100).
All formulas keep the breakout gate: must have 20%+ dominator AND total_teammate_dc > 150.
"""
import pandas as pd
import numpy as np
from scipy import stats

import os
os.chdir("/home/user/SlapModelV3")


# ============================================================================
# Helper functions (same as validation)
# ============================================================================
def dc_score(pick):
    if pd.isna(pick): return np.nan
    return 100 - 2.40 * (pick ** 0.62 - 1)

def wr_enhanced_breakout(breakout_age, peak_dominator, rush_yards, threshold=20):
    if pd.isna(breakout_age):
        fallback_dom = peak_dominator if pd.notna(peak_dominator) else 0
        return min(35, 15 + fallback_dom)
    base = {18: 100, 19: 90, 20: 75, 21: 60, 22: 45, 23: 30, 24: 20}.get(int(breakout_age), 20)
    bonus = min((peak_dominator - threshold) * 0.5, 9.9) if pd.notna(peak_dominator) and peak_dominator >= threshold else 0
    score = base + bonus
    if pd.notna(rush_yards) and rush_yards >= 20:
        score += 5
    return min(score, 99.9)

def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 15:
        return np.nan, np.nan, len(x)
    coef_xz = np.polyfit(z, x, 1)
    resid_x = x - np.polyval(coef_xz, z)
    coef_yz = np.polyfit(z, y, 1)
    resid_y = y - np.polyval(coef_yz, z)
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(x)


# ============================================================================
# Load data
# ============================================================================
print("Loading data...")
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
wr_bt = pd.read_csv('data/wr_backtest_all_components.csv')
wr_tm = pd.read_csv('data/wr_teammate_scores.csv')

wr_bt = wr_bt.merge(wr_tm[['player_name', 'draft_year', 'total_teammate_dc']],
                     on=['player_name', 'draft_year'], how='left')

# Compute base components
wr_bt['s_dc'] = wr_bt['pick'].apply(dc_score)
wr_bt['s_breakout_raw'] = wr_bt.apply(
    lambda r: wr_enhanced_breakout(r['breakout_age'], r['peak_dominator'], r['rush_yards']), axis=1)
wr_bt['s_early_declare_binary'] = wr_bt['early_declare'].apply(lambda x: 1 if x == 1 else 0)
wr_bt['s_early_declare'] = np.where(wr_bt['s_early_declare_binary'] == 1, 100, 0).astype(float)

# Merge outcomes (only columns not already in wr_bt)
wr_out = outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year', 'pick',
    'first_3yr_ppg', 'career_ppg']].copy()
df = wr_bt.merge(wr_out, on=['player_name', 'draft_year', 'pick'], how='left')

# Gate: qualifies for teammate score
df['qualifies'] = (df['total_teammate_dc'].fillna(0) > 150) & (df['breakout_age'].notna())
n_qual = df['qualifies'].sum()
print(f"Players qualifying (DC>150 + breakout): {n_qual}")
print(f"Players NOT qualifying: {len(df) - n_qual}")

WR_W = {'dc': 0.70, 'breakout': 0.20, 'teammate': 0.05, 'early_declare': 0.05}


# ============================================================================
# Define the 5 teammate formulas
# ============================================================================
def calc_binary(row):
    """Current binary: 100 if qualifies, else 0."""
    return 100.0 if row['qualifies'] else 0.0

def calc_simple_dominator(row, scale_min, scale_max):
    """Formula 1: peak_dominator scaled 0-100 within qualifying group."""
    if not row['qualifies']:
        return 0.0
    raw = row['peak_dominator']
    if pd.isna(raw):
        return 0.0
    # Scale to 0-100 within qualifying group
    scaled = (raw - scale_min) / (scale_max - scale_min) * 100
    return max(0, min(100, scaled))

def calc_interaction(row, scale_min, scale_max):
    """Formula 2: (teammate_DC × peak_dominator) / 100, scaled 0-100."""
    if not row['qualifies']:
        return 0.0
    raw = (row['total_teammate_dc'] * row['peak_dominator']) / 100
    scaled = (raw - scale_min) / (scale_max - scale_min) * 100
    return max(0, min(100, scaled))

def calc_dom_above_threshold(row, scale_min, scale_max):
    """Formula 3: (peak_dominator - 20) scaled 0-100."""
    if not row['qualifies']:
        return 0.0
    raw = row['peak_dominator'] - 20
    if pd.isna(raw) or raw < 0:
        return 0.0
    scaled = (raw - scale_min) / (scale_max - scale_min) * 100
    return max(0, min(100, scaled))

def calc_tiered(row):
    """Formula 4: Tiered by dominator range."""
    if not row['qualifies']:
        return 0.0
    dom = row['peak_dominator']
    if pd.isna(dom) or dom < 20:
        return 0.0
    if dom < 25:
        return 40.0
    elif dom < 30:
        return 60.0
    elif dom < 35:
        return 80.0
    else:
        return 100.0


# ============================================================================
# Calculate scaling parameters from qualifying backtest players
# ============================================================================
qual_df = df[df['qualifies']].copy()

# Formula 1: Simple dominator — min/max of peak_dominator in qualifying group
dom_min = qual_df['peak_dominator'].min()
dom_max = qual_df['peak_dominator'].max()
print(f"\nQualifying group peak_dominator: min={dom_min:.1f}%, max={dom_max:.1f}%")

# Formula 2: Interaction — min/max of (teammate_DC × peak_dominator)/100
qual_df['interaction_raw'] = (qual_df['total_teammate_dc'] * qual_df['peak_dominator']) / 100
int_min = qual_df['interaction_raw'].min()
int_max = qual_df['interaction_raw'].max()
print(f"Qualifying group interaction: min={int_min:.1f}, max={int_max:.1f}")

# Formula 3: Dominator above threshold — min/max of (peak_dom - 20)
qual_df['above_thresh_raw'] = qual_df['peak_dominator'] - 20
at_min = qual_df['above_thresh_raw'].min()
at_max = qual_df['above_thresh_raw'].max()
print(f"Qualifying group dom-above-20: min={at_min:.1f}, max={at_max:.1f}")


# ============================================================================
# Apply all formulas
# ============================================================================
print("\nCalculating teammate scores for all 5 formulas...")

df['tm_binary'] = df.apply(calc_binary, axis=1)
df['tm_simple_dom'] = df.apply(lambda r: calc_simple_dominator(r, dom_min, dom_max), axis=1)
df['tm_interaction'] = df.apply(lambda r: calc_interaction(r, int_min, int_max), axis=1)
df['tm_dom_above'] = df.apply(lambda r: calc_dom_above_threshold(r, at_min, at_max), axis=1)
df['tm_tiered'] = df.apply(calc_tiered, axis=1)

formulas = {
    'A) Binary (current)': 'tm_binary',
    'B) Simple Dominator': 'tm_simple_dom',
    'C) Interaction (DC×Dom)': 'tm_interaction',
    'D) Dom Above 20%': 'tm_dom_above',
    'E) Tiered': 'tm_tiered',
}


# ============================================================================
# Show distribution for each formula (qualifying players only)
# ============================================================================
print(f"\n{'='*100}")
print("SCORE DISTRIBUTIONS (qualifying players only, N={})".format(n_qual))
print(f"{'='*100}")
for label, col in formulas.items():
    qual_scores = df.loc[df['qualifies'], col]
    print(f"\n  {label}:")
    print(f"    Range: {qual_scores.min():.1f} - {qual_scores.max():.1f}")
    print(f"    Mean: {qual_scores.mean():.1f}, Median: {qual_scores.median():.1f}")
    if col == 'tm_tiered':
        for val in [40, 60, 80, 100]:
            n = (qual_scores == val).sum()
            print(f"    Score={val}: {n} players ({n/len(qual_scores)*100:.0f}%)")
    else:
        # Show quintiles
        for pct in [10, 25, 50, 75, 90]:
            print(f"    P{pct}: {qual_scores.quantile(pct/100):.1f}")


# ============================================================================
# Partial correlations (teammate score vs outcomes, controlling for DC)
# ============================================================================
print(f"\n{'='*100}")
print("PARTIAL CORRELATIONS (controlling for DC score)")
print(f"{'='*100}")

analysis_df = df.dropna(subset=['hit24']).copy()
print(f"Players with outcomes: {len(analysis_df)}")

print(f"\n  {'Formula':<30} {'hit24':>18} {'first_3yr_ppg':>18} {'career_ppg':>18} {'hit12':>18}")
print(f"  {'-'*102}")
for label, col in formulas.items():
    vals = []
    for outcome in ['hit24', 'first_3yr_ppg', 'career_ppg', 'hit12']:
        subset = analysis_df[[col, outcome, 's_dc']].dropna()
        if len(subset) > 15:
            r, p, n = partial_corr(subset[col].values, subset[outcome].values, subset['s_dc'].values)
            sig = '**' if p < 0.01 else '*' if p < 0.05 else ''
            vals.append(f"r={r:+.4f}{sig}")
        else:
            vals.append("n/a")
    print(f"  {label:<30} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18} {vals[3]:>18}")


# ============================================================================
# Full validation: plug each formula into SLAP V5 and compute all metrics
# ============================================================================
print(f"\n{'='*100}")
print("FULL VALIDATION COMPARISON (WR V5 with each teammate formula)")
print(f"{'='*100}")

val_df = analysis_df.copy()

results = {}
for label, col in formulas.items():
    # Compute SLAP with this formula
    val_df[f'slap_{col}'] = (
        WR_W['dc'] * val_df['s_dc'] +
        WR_W['breakout'] * val_df['s_breakout_raw'] +
        WR_W['teammate'] * val_df[col] +
        WR_W['early_declare'] * val_df['s_early_declare']
    )
    slap_col = f'slap_{col}'

    metrics = {}

    # Correlations
    for outcome in ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']:
        subset = val_df[[slap_col, outcome]].dropna()
        if len(subset) > 10:
            pr, _ = stats.pearsonr(subset[slap_col], subset[outcome])
            sr, _ = stats.spearmanr(subset[slap_col], subset[outcome])
            metrics[f'pearson_{outcome}'] = pr
            metrics[f'spearman_{outcome}'] = sr

    # PRI-AVG (40/25/20/15 weighted average of Spearman correlations)
    weights = {'hit24': 0.40, 'first_3yr_ppg': 0.25, 'career_ppg': 0.20, 'hit12': 0.15}
    pri_avg = sum(weights[o] * metrics.get(f'spearman_{o}', 0) for o in weights)
    metrics['pri_avg'] = pri_avg

    # Brier scores
    for outcome in ['hit24', 'hit12']:
        subset = val_df[[slap_col, outcome]].dropna()
        if len(subset) > 10:
            # Normalize to 0-1 for probability
            s_min, s_max = subset[slap_col].min(), subset[slap_col].max()
            probs = (subset[slap_col] - s_min) / (s_max - s_min)
            brier = ((probs - subset[outcome]) ** 2).mean()
            metrics[f'brier_{outcome}'] = brier

    # Top-decile precision
    n_top = max(1, len(val_df) // 10)
    top_players = val_df.nlargest(n_top, slap_col)
    metrics['top10_hit24'] = top_players['hit24'].mean() * 100
    top_ppg = top_players['first_3yr_ppg'].dropna()
    metrics['top10_ppg'] = top_ppg.mean() if len(top_ppg) > 0 else 0

    results[label] = metrics

# Print comparison table
print(f"\n  {'Metric':<30}", end="")
for label in formulas:
    short = label.split(')')[0] + ')'
    print(f" {short:>14}", end="")
print()
print(f"  {'-'*100}")

metric_display = [
    ('PRI-AVG', 'pri_avg', '+', True),
    ('Spearman hit24', 'spearman_hit24', '+', True),
    ('Spearman first_3yr_ppg', 'spearman_first_3yr_ppg', '+', True),
    ('Spearman career_ppg', 'spearman_career_ppg', '+', True),
    ('Spearman hit12', 'spearman_hit12', '+', True),
    ('Pearson hit24', 'pearson_hit24', '+', True),
    ('Pearson first_3yr_ppg', 'pearson_first_3yr_ppg', '+', True),
    ('Brier hit24', 'brier_hit24', '', False),
    ('Brier hit12', 'brier_hit12', '', False),
    ('Top10% hit24 rate', 'top10_hit24', '', True),
    ('Top10% avg PPG', 'top10_ppg', '', True),
]

for display_name, key, prefix, higher_better in metric_display:
    print(f"  {display_name:<30}", end="")
    vals_for_best = []
    for label in formulas:
        v = results[label].get(key, np.nan)
        vals_for_best.append(v)

    if higher_better:
        best_val = max(vals_for_best)
    else:
        best_val = min(vals_for_best)

    for i, label in enumerate(formulas):
        v = results[label].get(key, np.nan)
        marker = " <--" if v == best_val and not np.isnan(v) else ""
        if key.startswith('top10_hit24'):
            print(f" {v:>10.1f}%{marker:>3}", end="")
        elif key.startswith('top10_ppg'):
            print(f" {v:>11.2f}{marker:>3}", end="")
        elif key.startswith('brier'):
            print(f" {v:>11.4f}{marker:>3}", end="")
        else:
            print(f" {prefix}{v:>10.4f}{marker:>3}", end="")
    print()

# Mark wins
print(f"\n  {'WINS (best on metric)':<30}", end="")
win_counts = {label: 0 for label in formulas}
for _, key, _, higher_better in metric_display:
    vals = [results[label].get(key, np.nan) for label in formulas]
    if higher_better:
        best_idx = np.nanargmax(vals)
    else:
        best_idx = np.nanargmin(vals)
    win_counts[list(formulas.keys())[best_idx]] += 1

for label in formulas:
    short = label.split(')')[0] + ')'
    print(f" {win_counts[label]:>14}", end="")
print()


# ============================================================================
# Carnell Tate's scores under each formula
# ============================================================================
print(f"\n{'='*100}")
print("CARNELL TATE (Ohio State) — SCORES UNDER EACH FORMULA")
print(f"{'='*100}")

# Load 2026 data
tate_dom = 25.287495460597988  # From breakout data
tate_breakout_age = 20.0

# Calculate Tate's total_teammate_dc (from build output)
# We need to know his teammate DC. Let me compute it from the build output
master = pd.read_csv('output/slap_v5_master_database.csv')
wr26 = master[(master['position'] == 'WR') & (master['dataset'] == '2026_prospect')]

# For Carnell Tate, his teammates are Ohio State pass catchers from 2025-2027 window
# From the build: Emeka Egbuka (2025, pick 19) + other 2026 OSU prospects
# Let me get his total_teammate_dc from the build
draft_picks_all = pd.read_parquet('data/nflverse/draft_picks.parquet')
pc_2025 = draft_picks_all[
    (draft_picks_all['position'].isin(['WR', 'TE'])) &
    (draft_picks_all['season'] == 2025)
].copy()

def normalize_college(name):
    if pd.isna(name): return ""
    name = str(name).strip()
    replacements = {
        'Ohio St.': 'Ohio State', 'Michigan St.': 'Michigan State', 'Penn St.': 'Penn State',
        'Arizona St.': 'Arizona State', 'Oklahoma St.': 'Oklahoma State', 'Oregon St.': 'Oregon State',
        'Washington St.': 'Washington State', 'Florida St.': 'Florida State', 'Boise St.': 'Boise State',
        'Colorado St.': 'Colorado State', 'Iowa St.': 'Iowa State', 'Kansas St.': 'Kansas State',
        'Fresno St.': 'Fresno State', 'North Carolina St.': 'NC State', 'Central Florida': 'UCF',
        'Miami (FL)': 'Miami', 'Mississippi': 'Ole Miss', 'Boston Col.': 'Boston College',
        'North Dakota St.': 'North Dakota State',
    }
    for old, new in replacements.items():
        if name == old:
            return new.lower()
    return name.lower()

pc_2025['college_norm'] = pc_2025['college'].apply(normalize_college)
pc_2025['dc'] = pc_2025['pick'].apply(dc_score)
osu_2025 = pc_2025[pc_2025['college_norm'] == 'ohio state']

# Also get 2026 OSU WR/TE prospects (excluding Tate himself)
prospects = pd.read_csv('data/prospects_final.csv')
wr_prosp = prospects[prospects['position'] == 'WR'].copy()
wr_prosp['college_norm'] = wr_prosp['school'].apply(normalize_college)
osu_wr26 = wr_prosp[(wr_prosp['college_norm'] == 'ohio state') & (wr_prosp['player_name'] != 'Carnell Tate')]

te_prosp = pd.read_csv('data/te_2026_prospects_final.csv')
te_prosp['college_norm'] = te_prosp['college'].apply(normalize_college)
osu_te26 = te_prosp[te_prosp['college_norm'] == 'ohio state']

# Calculate total teammate DC for Tate
tate_tm_dc = 0.0
tm_details = []
for _, row in osu_2025.iterrows():
    d = dc_score(row['pick'])
    tate_tm_dc += d
    tm_details.append(f"{row['pfr_player_name']} (2025 Rd{row['round']} #{row['pick']}, DC={d:.1f})")
for _, row in osu_wr26.iterrows():
    d = dc_score(row['projected_pick'])
    tate_tm_dc += d
    tm_details.append(f"{row['player_name']} (2026 WR #{row['projected_pick']}, DC={d:.1f})")
for _, row in osu_te26.iterrows():
    d = dc_score(row['projected_pick'])
    tate_tm_dc += d
    tm_details.append(f"{row['player_name']} (2026 TE #{row['projected_pick']}, DC={d:.1f})")

print(f"  Carnell Tate: Ohio State, projected pick 7")
print(f"  Breakout age: {tate_breakout_age}, Peak dominator: {tate_dom:.1f}%")
print(f"  Total teammate DC: {tate_tm_dc:.1f}")
print(f"  Teammates: {'; '.join(tm_details)}")
print(f"  Qualifies for teammate score: {'YES' if tate_tm_dc > 150 else 'NO'} (DC>150 + broke out)")

# Calculate Tate's scores under each formula
tate_scores = {}

# Binary
tate_scores['A) Binary (current)'] = 100.0 if tate_tm_dc > 150 else 0.0

# Simple Dominator (scaled within qualifying group)
tate_simple = (tate_dom - dom_min) / (dom_max - dom_min) * 100
tate_scores['B) Simple Dominator'] = max(0, min(100, tate_simple))

# Interaction
tate_int_raw = (tate_tm_dc * tate_dom) / 100
tate_int = (tate_int_raw - int_min) / (int_max - int_min) * 100
tate_scores['C) Interaction (DC×Dom)'] = max(0, min(100, tate_int))

# Dom Above Threshold
tate_above = tate_dom - 20
tate_above_scaled = (tate_above - at_min) / (at_max - at_min) * 100
tate_scores['D) Dom Above 20%'] = max(0, min(100, tate_above_scaled))

# Tiered
if tate_dom < 25:
    tate_scores['E) Tiered'] = 40.0
elif tate_dom < 30:
    tate_scores['E) Tiered'] = 60.0
elif tate_dom < 35:
    tate_scores['E) Tiered'] = 80.0
else:
    tate_scores['E) Tiered'] = 100.0

print(f"\n  {'Formula':<30} {'Teammate Score':>15} {'SLAP Impact':>15}")
print(f"  {'-'*60}")
for label, score in tate_scores.items():
    impact = WR_W['teammate'] * score  # 0.05 × score
    print(f"  {label:<30} {score:>14.1f} {impact:>14.2f}")

# Show what this means for SLAP spread
tate_dc = dc_score(7)
tate_breakout_score = wr_enhanced_breakout(tate_breakout_age, tate_dom, 16)
tate_ed = 100  # early declare
print(f"\n  Tate's other components: DC={tate_dc:.1f}, Breakout={tate_breakout_score:.1f}, Early Declare={tate_ed}")
print(f"\n  {'Formula':<30} {'TM Score':>10} {'SLAP Raw':>12}")
print(f"  {'-'*52}")
for label, tm_score in tate_scores.items():
    slap = WR_W['dc'] * tate_dc + WR_W['breakout'] * tate_breakout_score + \
           WR_W['teammate'] * tm_score + WR_W['early_declare'] * tate_ed
    print(f"  {label:<30} {tm_score:>9.1f} {slap:>11.2f}")


# ============================================================================
# 2026 class: show score ranges and specific players
# ============================================================================
print(f"\n{'='*100}")
print("2026 WR PROSPECTS — TEAMMATE SCORES UNDER EACH FORMULA")
print(f"{'='*100}")

# Load 2026 breakout data
wr26_bo = pd.read_csv('data/wr_breakout_ages_2026.csv')
wr26_pre = pd.read_csv('output/slap_v5_wr_2026.csv')

# Get all 2026 WR prospects who might qualify
# Need: total_teammate_dc (from build) and breakout data
# Rebuild the total_teammate_dc for all 2026 WRs
all_wr26 = wr26_pre.copy()
all_wr26 = all_wr26.merge(
    wr26_bo[['player_name', 'breakout_age', 'peak_dominator']].rename(
        columns={'breakout_age': 'bo_age_new', 'peak_dominator': 'pd_new'}),
    on='player_name', how='left'
)
all_wr26['breakout_age_final'] = all_wr26['bo_age_new'].fillna(all_wr26['breakout_age'])
all_wr26['peak_dominator_final'] = all_wr26['pd_new'].fillna(all_wr26['peak_dominator'])

# Load prospects for college info
wr_prosp_all = prospects[prospects['position'] == 'WR'].copy()
wr_prosp_all['college_norm'] = wr_prosp_all['school'].apply(normalize_college)
all_wr26 = all_wr26.merge(wr_prosp_all[['player_name', 'school', 'college_norm']], on='player_name', how='left')
# Fill college_norm from college column if missing
if 'college' in all_wr26.columns:
    all_wr26['college_norm'] = all_wr26['college_norm'].fillna(all_wr26['college'].apply(normalize_college))

# Calculate total_teammate_dc for each 2026 WR
# Build pool: 2025 actual picks + 2026 WR + 2026 TE prospects
pool_2025 = pc_2025[['pfr_player_name', 'college_norm', 'dc']].rename(columns={'pfr_player_name': 'name'})
pool_wr26 = wr_prosp_all[['player_name', 'college_norm']].copy()
pool_wr26['dc'] = wr_prosp_all['projected_pick'].apply(dc_score)
pool_wr26 = pool_wr26.rename(columns={'player_name': 'name'})
pool_te26 = te_prosp[['player_name', 'college_norm']].copy()
pool_te26['dc'] = te_prosp['projected_pick'].apply(dc_score)
pool_te26 = pool_te26.rename(columns={'player_name': 'name'})
pool_all = pd.concat([pool_2025, pool_wr26, pool_te26], ignore_index=True)

tm_dc_2026 = {}
for _, row in all_wr26.iterrows():
    name = row['player_name']
    school = row.get('college_norm', '')
    if pd.isna(school) or school == '':
        tm_dc_2026[name] = 0.0
        continue
    tms = pool_all[(pool_all['college_norm'] == school) & (pool_all['name'] != name)]
    tm_dc_2026[name] = tms['dc'].sum() if len(tms) > 0 else 0.0

all_wr26['total_tm_dc'] = all_wr26['player_name'].map(tm_dc_2026)
all_wr26['qualifies'] = (all_wr26['total_tm_dc'] > 150) & (all_wr26['breakout_age_final'].notna())

# Calculate scores under each formula
all_wr26['tm_binary'] = np.where(all_wr26['qualifies'], 100, 0).astype(float)

all_wr26['tm_simple_dom'] = np.where(
    all_wr26['qualifies'],
    np.clip((all_wr26['peak_dominator_final'] - dom_min) / (dom_max - dom_min) * 100, 0, 100),
    0
)

all_wr26['tm_interaction'] = np.where(
    all_wr26['qualifies'],
    np.clip(((all_wr26['total_tm_dc'] * all_wr26['peak_dominator_final']) / 100 - int_min) / (int_max - int_min) * 100, 0, 100),
    0
)

all_wr26['tm_dom_above'] = np.where(
    all_wr26['qualifies'],
    np.clip((all_wr26['peak_dominator_final'] - 20 - at_min) / (at_max - at_min) * 100, 0, 100),
    0
)

def tiered_val(dom):
    if pd.isna(dom) or dom < 20: return 0
    if dom < 25: return 40
    elif dom < 30: return 60
    elif dom < 35: return 80
    else: return 100

all_wr26['tm_tiered'] = np.where(
    all_wr26['qualifies'],
    all_wr26['peak_dominator_final'].apply(tiered_val),
    0
)

# Show qualifying 2026 WRs
qualifying = all_wr26[all_wr26['qualifies']].sort_values('peak_dominator_final', ascending=False)
print(f"\n  Qualifying 2026 WRs ({len(qualifying)} total):\n")
print(f"  {'Player':<25} {'College':<18} {'Pick':>5} {'PkDom':>7} {'TM DC':>8}  {'Binary':>7} {'SimpDom':>8} {'Intxn':>7} {'DomAbv':>7} {'Tiered':>7}")
print(f"  {'-'*118}")
for _, r in qualifying.iterrows():
    print(f"  {r['player_name']:<25} {str(r.get('college','')):<18} {r['pick']:>5.0f} "
          f"{r['peak_dominator_final']:>6.1f}% {r['total_tm_dc']:>7.1f}  "
          f"{r['tm_binary']:>7.0f} {r['tm_simple_dom']:>8.1f} {r['tm_interaction']:>7.1f} "
          f"{r['tm_dom_above']:>7.1f} {r['tm_tiered']:>7.0f}")

# Summary statistics for 2026
print(f"\n  2026 Score Ranges (qualifying players only):")
for label, col in formulas.items():
    vals = all_wr26.loc[all_wr26['qualifies'], col]
    if len(vals) > 0:
        print(f"    {label:<30}: {vals.min():.1f} - {vals.max():.1f} (mean {vals.mean():.1f})")


# ============================================================================
# EXAMPLE: high-dominators vs low-dominators among qualifying backtest players
# ============================================================================
print(f"\n{'='*100}")
print("BACKTEST: QUALIFYING PLAYERS — OUTCOMES BY DOMINATOR TIER")
print(f"{'='*100}")

qual_with_outcomes = qual_df.merge(
    outcomes[outcomes['position'] == 'WR'][['player_name', 'draft_year', 'pick',
        'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']],
    on=['player_name', 'draft_year', 'pick'], how='left', suffixes=('_orig', '')
).dropna(subset=['hit24'])

# Tier by dominator
qual_with_outcomes['dom_tier'] = pd.cut(
    qual_with_outcomes['peak_dominator'],
    bins=[0, 25, 30, 35, 100],
    labels=['20-25%', '25-30%', '30-35%', '35%+']
)

print(f"\n  {'Dom Tier':<12} {'N':>5} {'hit24%':>8} {'hit12%':>8} {'Avg PPG':>10} {'Avg SLAP':>10}")
print(f"  {'-'*55}")
for tier in ['20-25%', '25-30%', '30-35%', '35%+']:
    g = qual_with_outcomes[qual_with_outcomes['dom_tier'] == tier]
    if len(g) > 0:
        # Compute SLAP with binary for reference
        g_slap = (WR_W['dc'] * g['s_dc'] + WR_W['breakout'] * g['s_breakout_raw'] +
                  WR_W['teammate'] * 100 + WR_W['early_declare'] * g['s_early_declare'])
        h24 = g['hit24'].mean() * 100
        h12 = g['hit12'].mean() * 100
        ppg = g['first_3yr_ppg'].dropna().mean()
        print(f"  {tier:<12} {len(g):>5} {h24:>7.1f}% {h12:>7.1f}% {ppg:>9.1f} {g_slap.mean():>9.1f}")


print(f"\n{'='*100}")
print("ANALYSIS COMPLETE")
print(f"{'='*100}")
