"""
TE Follow-up Analysis: Two Tests

TEST 1: Lower breakout age thresholds (10%, 12%, 15%, 20%)
  - Recalculate breakout age from PFF multi-season data
  - Test partial correlation with NFL outcomes at each threshold

TEST 2: Production metric head-to-head comparison
  - Compare 4 production metrics across all 3 outcomes
  - Test CFBD+PFF hybrid fallback approach
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
bt = pd.read_csv('data/te_backtest_master.csv')
bt_outcomes = bt[bt['nfl_seasons_found'] > 0].copy()
bt_outcomes['dc_score'] = bt_outcomes['pick'].apply(
    lambda p: max(0, min(100, 100 - 2.40 * (p**0.62 - 1)))
)

print(f"Backtest TEs with outcomes: {len(bt_outcomes)}")

def partial_corr(x, y, z):
    """Partial correlation of x,y controlling for z. Returns (r, p, n)."""
    mask = pd.notna(x) & pd.notna(y) & pd.notna(z)
    x2 = x[mask].astype(float).values
    y2 = y[mask].astype(float).values
    z2 = z[mask].astype(float).values
    n = len(x2)
    if n < 10:
        return np.nan, np.nan, n
    sx, ix, _, _, _ = stats.linregress(z2, x2)
    rx = x2 - (sx * z2 + ix)
    sy, iy, _, _, _ = stats.linregress(z2, y2)
    ry = y2 - (sy * z2 + iy)
    r, p = stats.pearsonr(rx, ry)
    return r, p, n


# ============================================================
# TEST 1: LOWER BREAKOUT AGE THRESHOLDS
# ============================================================
print("\n" + "="*100)
print("TEST 1: BREAKOUT AGE AT DIFFERENT DOMINATOR THRESHOLDS")
print("="*100)

# Load ALL PFF files to build multi-season dominator data
pff_file_map = {
    'data/receiving_summary (2).csv': 2015,
    'data/receiving_summary (3).csv': 2016,
    'data/receiving_summary (4).csv': 2017,
    'data/receiving_summary (5).csv': 2018,
    'data/receiving_summary (21).csv': 2019,
    'data/receiving_summary (20).csv': 2020,
    'data/receiving_summary (19).csv': 2021,
    'data/receiving_summary (18).csv': 2022,
    'data/receiving_summary (17).csv': 2023,
    'data/receiving_summary (16).csv': 2024,
    'data/receiving_summary (15).csv': 2025,
}

print("\nLoading PFF multi-season data...")
all_pff = []
for filepath, season in pff_file_map.items():
    try:
        pff = pd.read_csv(filepath)
        # Keep only TEs and WRs (need WRs to calculate team totals)
        pff['season'] = season
        # Standardize column names - the file uses 'player' and 'yards'
        all_pff.append(pff)
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")

pff_all = pd.concat(all_pff, ignore_index=True)
print(f"  Total PFF rows loaded: {len(pff_all)}")
print(f"  Seasons: {sorted(pff_all['season'].unique())}")
print(f"  Positions: {pff_all['position'].value_counts().to_dict()}")

# Filter to receiving positions only (TE, WR, HB/RB for team totals)
receivers = pff_all[pff_all['position'].isin(['TE', 'WR', 'HB'])].copy()

# Calculate team total receiving yards per season
team_totals = receivers.groupby(['team_name', 'season'])['yards'].sum().reset_index()
team_totals.rename(columns={'yards': 'team_rec_yards'}, inplace=True)

# Get TE-only data
te_pff = receivers[receivers['position'] == 'TE'].copy()
te_pff = te_pff.merge(team_totals, on=['team_name', 'season'], how='left')

# Calculate dominator for each TE-season
te_pff['dominator_pct'] = np.where(
    te_pff['team_rec_yards'] > 0,
    (te_pff['yards'] / te_pff['team_rec_yards']) * 100,
    0
)

print(f"\n  TE-season records: {len(te_pff)}")
print(f"  Unique TEs in PFF: {te_pff['player'].nunique()}")

# Now match PFF TEs to our backtest TEs
# Normalize names for matching
def norm_name(n):
    s = str(n).lower().strip()
    for suf in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
        s = s.replace(suf, '')
    s = s.replace("'", "").replace("'", "").replace("-", "").replace(".", "")
    return s.strip()

te_pff['name_norm'] = te_pff['player'].apply(norm_name)
bt_outcomes['name_norm2'] = bt_outcomes['player_name'].apply(norm_name)

# For each backtest TE, find all their PFF seasons
print("\nMatching backtest TEs to PFF multi-season data...")
multi_season_data = []

for _, te in bt_outcomes.iterrows():
    te_name = te['name_norm2']
    draft_yr = te['draft_year']
    draft_age = te['draft_age']

    # Find all PFF seasons for this player (must be before draft year)
    pff_matches = te_pff[
        (te_pff['name_norm'] == te_name) &
        (te_pff['season'] < draft_yr)
    ].copy()

    if len(pff_matches) == 0:
        # Try partial matching on last name + team
        continue

    for _, pm in pff_matches.iterrows():
        season_age = draft_age - (draft_yr - pm['season'])
        multi_season_data.append({
            'player_name': te['player_name'],
            'draft_year': draft_yr,
            'draft_age': draft_age,
            'season': pm['season'],
            'season_age': season_age,
            'yards': pm['yards'],
            'team_rec_yards': pm['team_rec_yards'],
            'dominator_pct': pm['dominator_pct'],
            'team_name': pm['team_name'],
        })

msd = pd.DataFrame(multi_season_data)
matched_tes = msd['player_name'].nunique()
print(f"  Backtest TEs with multi-season PFF data: {matched_tes}/{len(bt_outcomes)}")
print(f"  Total TE-season records: {len(msd)}")

if len(msd) > 0:
    print(f"  Avg seasons per TE: {len(msd)/matched_tes:.1f}")

# Show dominator distribution across all TE-seasons
print(f"\n  TE-season dominator distribution:")
print(f"    >= 10%: {(msd['dominator_pct'] >= 10).sum()}/{len(msd)} seasons "
      f"({(msd['dominator_pct'] >= 10).mean():.0%})")
print(f"    >= 12%: {(msd['dominator_pct'] >= 12).sum()}/{len(msd)} seasons "
      f"({(msd['dominator_pct'] >= 12).mean():.0%})")
print(f"    >= 15%: {(msd['dominator_pct'] >= 15).sum()}/{len(msd)} seasons "
      f"({(msd['dominator_pct'] >= 15).mean():.0%})")
print(f"    >= 20%: {(msd['dominator_pct'] >= 20).sum()}/{len(msd)} seasons "
      f"({(msd['dominator_pct'] >= 20).mean():.0%})")

# Calculate breakout age at each threshold
thresholds = [10, 12, 15, 20]

print(f"\n{'='*100}")
print("BREAKOUT AGE RESULTS BY THRESHOLD")
print(f"{'='*100}")

threshold_results = {}

for thresh in thresholds:
    # For each TE, find first season they hit this threshold
    breakout_data = {}

    for te_name in msd['player_name'].unique():
        te_seasons = msd[msd['player_name'] == te_name].sort_values('season')

        # Peak dominator across all seasons
        peak_dom = te_seasons['dominator_pct'].max()

        # First season hitting threshold
        hit_seasons = te_seasons[te_seasons['dominator_pct'] >= thresh]
        if len(hit_seasons) > 0:
            first_hit = hit_seasons.iloc[0]
            breakout_data[te_name] = {
                'breakout_age': first_hit['season_age'],
                'peak_dominator': peak_dom,
                'breakout_season': first_hit['season'],
                'hit_threshold': True,
            }
        else:
            breakout_data[te_name] = {
                'breakout_age': np.nan,
                'peak_dominator': peak_dom,
                'breakout_season': np.nan,
                'hit_threshold': False,
            }

    bo_df = pd.DataFrame.from_dict(breakout_data, orient='index')
    bo_df.index.name = 'player_name'
    bo_df.reset_index(inplace=True)

    # Merge with outcomes
    merged = bt_outcomes.merge(bo_df, on='player_name', how='left', suffixes=('', f'_t{thresh}'))

    # Coverage stats
    has_pff = merged['player_name'].isin(msd['player_name'].unique())
    hit_thresh = merged[f'hit_threshold'].fillna(False)
    has_bo = merged['breakout_age' + (f'_t{thresh}' if f'breakout_age_t{thresh}' in merged.columns else '')].notna()

    # The breakout_age column may have suffix conflicts - let's handle carefully
    # Since bt_outcomes already has breakout_age from CFBD, the merge adds _t{thresh}
    bo_col = f'breakout_age_t{thresh}' if f'breakout_age_t{thresh}' in merged.columns else 'breakout_age'

    # Actually let's just use the bo_df data directly
    coverage_total = len(merged)
    n_with_pff = has_pff.sum()
    n_hit = hit_thresh.sum()
    n_never = has_pff.sum() - hit_thresh.sum()
    n_no_pff = (~has_pff).sum()

    print(f"\n--- Threshold: {thresh}% dominator ---")
    print(f"  TEs with PFF multi-season data: {n_with_pff}/{coverage_total} ({100*n_with_pff/coverage_total:.0f}%)")
    print(f"  TEs who hit {thresh}%: {n_hit} ({100*n_hit/coverage_total:.0f}%)")
    print(f"  TEs who never hit {thresh}%: {n_never}")
    print(f"  TEs with no PFF data: {n_no_pff}")

    # Breakout age distribution for those who hit
    hit_tes = merged[hit_thresh]
    if len(hit_tes) > 0 and bo_col in merged.columns:
        bo_vals = merged.loc[hit_thresh, bo_col].dropna()
        if len(bo_vals) > 0:
            print(f"  Breakout age range: {bo_vals.min():.0f} - {bo_vals.max():.0f}")
            print(f"  Breakout age mean: {bo_vals.mean():.1f}")

    # NFL outcomes comparison: hit threshold vs never hit (among those with PFF data)
    pff_tes = merged[has_pff & merged['first_3yr_ppg'].notna()].copy()
    hit_mask = pff_tes['hit_threshold'].fillna(False).astype(bool)
    hit_group = pff_tes[hit_mask]
    miss_group = pff_tes[~hit_mask]

    if len(hit_group) > 0 and len(miss_group) > 0:
        print(f"\n  Hit {thresh}% group:   n={len(hit_group):>3}, avg PPG={hit_group['first_3yr_ppg'].mean():.2f}, "
              f"hit24={hit_group['hit24'].mean():.0%}, hit12={hit_group['hit12'].mean():.0%}, "
              f"avg pick={hit_group['pick'].mean():.0f}")
        print(f"  Never hit group:   n={len(miss_group):>3}, avg PPG={miss_group['first_3yr_ppg'].mean():.2f}, "
              f"hit24={miss_group['hit24'].mean():.0%}, hit12={miss_group['hit12'].mean():.0%}, "
              f"avg pick={miss_group['pick'].mean():.0f}")

    # Partial correlations

    # A) Breakout age (continuous, only for those who hit) vs outcomes
    if bo_col in merged.columns:
        bo_valid = merged[bo_col].notna()
        if bo_valid.sum() >= 10:
            for outcome in ['first_3yr_ppg', 'hit24', 'hit12']:
                r, p, n = partial_corr(merged[bo_col], merged[outcome], merged['pick'])
                print(f"  Breakout age (continuous) vs {outcome:15s}: partial r={r:+.3f}, p={p:.4f}, n={n}" if pd.notna(r)
                      else f"  Breakout age (continuous) vs {outcome:15s}: insufficient data (n={n})")

    # B) Binary (hit threshold yes/no) vs outcomes — includes ALL TEs with PFF data
    merged['hit_binary'] = np.where(has_pff, hit_thresh.astype(int), np.nan)
    for outcome in ['first_3yr_ppg', 'hit24', 'hit12']:
        r, p, n = partial_corr(merged['hit_binary'], merged[outcome], merged['pick'])
        if pd.notna(r):
            print(f"  Binary (hit {thresh}% y/n)  vs {outcome:15s}: partial r={r:+.3f}, p={p:.4f}, n={n}")

    # Store for summary
    r_ppg_cont, p_ppg_cont, n_ppg_cont = partial_corr(
        merged[bo_col] if bo_col in merged.columns else pd.Series(dtype=float),
        merged['first_3yr_ppg'], merged['pick']
    )
    r_ppg_bin, p_ppg_bin, n_ppg_bin = partial_corr(
        merged['hit_binary'], merged['first_3yr_ppg'], merged['pick']
    )

    threshold_results[thresh] = {
        'coverage_pct': 100 * n_with_pff / coverage_total,
        'hit_pct': 100 * n_hit / coverage_total,
        'n_hit': n_hit,
        'r_cont_ppg': r_ppg_cont, 'p_cont_ppg': p_ppg_cont, 'n_cont': n_ppg_cont,
        'r_bin_ppg': r_ppg_bin, 'p_bin_ppg': p_ppg_bin, 'n_bin': n_ppg_bin,
    }

# Summary table
print(f"\n\n{'='*100}")
print("THRESHOLD COMPARISON SUMMARY")
print(f"{'='*100}")
print(f"\n{'Threshold':<12} {'Hit%':>6} {'N hit':>6} {'Cont r(PPG)':>13} {'Cont p':>9} {'Cont n':>7} {'Bin r(PPG)':>12} {'Bin p':>9} {'Bin n':>7}")
print("-"*90)
for thresh in thresholds:
    tr = threshold_results[thresh]
    r_cont = f"{tr['r_cont_ppg']:+.3f}" if pd.notna(tr['r_cont_ppg']) else 'N/A'
    p_cont = f"{tr['p_cont_ppg']:.4f}" if pd.notna(tr['p_cont_ppg']) else 'N/A'
    r_bin = f"{tr['r_bin_ppg']:+.3f}" if pd.notna(tr['r_bin_ppg']) else 'N/A'
    p_bin = f"{tr['p_bin_ppg']:.4f}" if pd.notna(tr['p_bin_ppg']) else 'N/A'
    print(f"  {thresh}%{'':<8} {tr['hit_pct']:>5.0f}% {tr['n_hit']:>5} {r_cont:>13} {p_cont:>9} {tr['n_cont']:>6} {r_bin:>12} {p_bin:>9} {tr['n_bin']:>6}")


# ============================================================
# Also test: WR-style continuous scoring using PFF peak dominator
# ============================================================
print(f"\n\n{'='*100}")
print("BONUS: CONTINUOUS BREAKOUT SCORING (WR-style, using PFF peak dominator)")
print("Uses peak dominator as tiebreaker within age tiers, same approach as WR model")
print(f"{'='*100}")

# Build continuous breakout scores at each threshold
for thresh in [10, 12, 15]:
    # Recalculate from msd
    bo_scores = {}
    for te_name in msd['player_name'].unique():
        te_seasons = msd[msd['player_name'] == te_name].sort_values('season')
        peak_dom = te_seasons['dominator_pct'].max()

        hit_seasons = te_seasons[te_seasons['dominator_pct'] >= thresh]
        if len(hit_seasons) > 0:
            bo_age = hit_seasons.iloc[0]['season_age']
            # WR-style age tier scores
            if bo_age <= 18: base = 100
            elif bo_age <= 19: base = 90
            elif bo_age <= 20: base = 75
            elif bo_age <= 21: base = 60
            elif bo_age <= 22: base = 45
            elif bo_age <= 23: base = 30
            else: base = 20
            bonus = min((peak_dom - thresh) * 0.5, 9.9)
            score = min(base + bonus, 99.9)
        else:
            # Never hit threshold
            score = min(35, 15 + peak_dom)

        bo_scores[te_name] = score

    bo_series = pd.Series(bo_scores, name=f'bo_score_{thresh}')
    merged_bo = bt_outcomes.merge(
        bo_series.reset_index().rename(columns={'index': 'player_name', f'bo_score_{thresh}': 'bo_score'}),
        on='player_name', how='left'
    )

    for outcome in ['first_3yr_ppg', 'hit24', 'hit12']:
        r, p, n = partial_corr(merged_bo['bo_score'], merged_bo[outcome], merged_bo['pick'])
        sig = '***' if pd.notna(p) and p < 0.01 else ('**' if pd.notna(p) and p < 0.05 else ('*' if pd.notna(p) and p < 0.10 else ''))
        if pd.notna(r):
            print(f"  Threshold {thresh}%: bo_score vs {outcome:15s}: partial r={r:+.3f}, p={p:.4f}, n={n} {sig}")


# ============================================================
# TEST 2: PRODUCTION METRIC HEAD-TO-HEAD
# ============================================================
print(f"\n\n{'='*100}")
print("TEST 2: PRODUCTION METRIC HEAD-TO-HEAD COMPARISON")
print("Partial correlations with ALL 3 outcomes, controlling for draft pick")
print(f"{'='*100}")

d = bt_outcomes.copy()

# Build all candidate metrics
# RYPTPA (CFBD)
d['ryptpa'] = np.where(
    (d['cfbd_rec_yards'].notna()) & (d['cfbd_team_pass_att'].notna()) & (d['cfbd_team_pass_att'] > 0),
    d['cfbd_rec_yards'] / d['cfbd_team_pass_att'],
    np.nan
)

# Age-weighted RYPTPA
def age_wt(age):
    if pd.isna(age): return 1.0
    if age <= 19: return 1.15
    elif age <= 20: return 1.10
    elif age <= 21: return 1.05
    elif age <= 22: return 1.00
    elif age <= 23: return 0.95
    else: return 0.90

d['age_ryptpa'] = d.apply(
    lambda r: r['ryptpa'] * age_wt(r['draft_age']) if pd.notna(r['ryptpa']) else np.nan,
    axis=1
)

# Receptions / Team Pass Att (CFBD)
d['rec_per_tpa'] = np.where(
    (d['cfbd_receptions'].notna()) & (d['cfbd_team_pass_att'].notna()) & (d['cfbd_team_pass_att'] > 0),
    d['cfbd_receptions'] / d['cfbd_team_pass_att'],
    np.nan
)

# PFF Receptions Per Game
d['pff_rpg'] = np.where(
    (d['pff_receptions'].notna()) & (d['pff_player_game_count'].notna()) & (d['pff_player_game_count'] > 0),
    d['pff_receptions'] / d['pff_player_game_count'],
    np.nan
)

# PFF YPRR already exists as pff_yprr

# PFF Target Rate
d['pff_target_rate'] = np.where(
    (d['pff_targets'].notna()) & (d['pff_routes'].notna()) & (d['pff_routes'] > 0),
    d['pff_targets'] / d['pff_routes'],
    np.nan
)

# PFF Yards Per Game
d['pff_ypg'] = np.where(
    (d['pff_yards'].notna()) & (d['pff_player_game_count'].notna()) & (d['pff_player_game_count'] > 0),
    d['pff_yards'] / d['pff_player_game_count'],
    np.nan
)

candidate_metrics = {
    'rec_per_tpa': ('Receptions / Team Pass Att (CFBD)', 'CFBD'),
    'ryptpa': ('RYPTPA — Rec Yds / Team Pass Att (CFBD)', 'CFBD'),
    'age_ryptpa': ('Age-Weighted RYPTPA (CFBD)', 'CFBD'),
    'pff_rpg': ('PFF Receptions Per Game', 'PFF'),
    'pff_yprr': ('PFF Yards Per Route Run', 'PFF'),
    'pff_target_rate': ('PFF Target Rate (targets/routes)', 'PFF'),
    'pff_ypg': ('PFF Yards Per Game', 'PFF'),
    'pff_grades_offense': ('PFF Offense Grade', 'PFF'),
    'pff_grades_pass_route': ('PFF Pass Route Grade', 'PFF'),
    'pff_avoided_tackles': ('PFF Avoided Tackles', 'PFF'),
}

print(f"\n{'Metric':<45} {'Src':>4} {'N':>4} {'Cov%':>5}  {'r(PPG)':>8} {'p(PPG)':>8}  {'r(h24)':>8} {'p(h24)':>8}  {'r(h12)':>8} {'p(h12)':>8}  {'Avg|r|':>7}")
print("-"*130)

metric_scores = []
for col, (name, src) in candidate_metrics.items():
    n_valid = d[col].notna().sum()
    cov = 100 * n_valid / len(d)

    rs = {}
    for outcome in ['first_3yr_ppg', 'hit24', 'hit12']:
        r, p, n = partial_corr(d[col], d[outcome], d['pick'])
        rs[outcome] = (r, p, n)

    r_ppg, p_ppg, _ = rs['first_3yr_ppg']
    r_h24, p_h24, _ = rs['hit24']
    r_h12, p_h12, _ = rs['hit12']

    avg_abs_r = np.nanmean([abs(r_ppg) if pd.notna(r_ppg) else 0,
                             abs(r_h24) if pd.notna(r_h24) else 0,
                             abs(r_h12) if pd.notna(r_h12) else 0])

    r_ppg_s = f"{r_ppg:+.3f}" if pd.notna(r_ppg) else '  N/A'
    p_ppg_s = f"{p_ppg:.4f}" if pd.notna(p_ppg) else '  N/A'
    r_h24_s = f"{r_h24:+.3f}" if pd.notna(r_h24) else '  N/A'
    p_h24_s = f"{p_h24:.4f}" if pd.notna(p_h24) else '  N/A'
    r_h12_s = f"{r_h12:+.3f}" if pd.notna(r_h12) else '  N/A'
    p_h12_s = f"{p_h12:.4f}" if pd.notna(p_h12) else '  N/A'

    print(f"{name:<45} {src:>4} {n_valid:>4} {cov:>4.0f}%  {r_ppg_s:>8} {p_ppg_s:>8}  {r_h24_s:>8} {p_h24_s:>8}  {r_h12_s:>8} {p_h12_s:>8}  {avg_abs_r:>6.3f}")

    metric_scores.append({
        'col': col, 'name': name, 'src': src,
        'n': n_valid, 'cov': cov,
        'r_ppg': r_ppg, 'p_ppg': p_ppg,
        'r_h24': r_h24, 'p_h24': p_h24,
        'r_h12': r_h12, 'p_h12': p_h12,
        'avg_abs_r': avg_abs_r,
    })


# ============================================================
# TEST 2B: HYBRID CFBD + PFF FALLBACK
# ============================================================
print(f"\n\n{'='*100}")
print("TEST 2B: HYBRID APPROACH — CFBD primary + PFF fallback for missing data")
print(f"{'='*100}")

# Test several hybrid combinations
hybrids = [
    ('rec_per_tpa', 'pff_rpg', 'Rec/TPA (CFBD) + PFF Rec/Game fallback'),
    ('rec_per_tpa', 'pff_yprr', 'Rec/TPA (CFBD) + PFF YPRR fallback'),
    ('ryptpa', 'pff_rpg', 'RYPTPA (CFBD) + PFF Rec/Game fallback'),
    ('ryptpa', 'pff_yprr', 'RYPTPA (CFBD) + PFF YPRR fallback'),
    ('age_ryptpa', 'pff_rpg', 'Age-RYPTPA (CFBD) + PFF Rec/Game fallback'),
    ('age_ryptpa', 'pff_yprr', 'Age-RYPTPA (CFBD) + PFF YPRR fallback'),
]

print(f"\nApproach: Normalize both metrics to 0-100 within their populations,")
print(f"use CFBD metric when available, PFF metric as fallback when CFBD is missing.\n")

print(f"{'Hybrid':<50} {'N':>4} {'Cov%':>5}  {'r(PPG)':>8} {'p(PPG)':>8}  {'r(h24)':>8} {'p(h24)':>8}  {'r(h12)':>8} {'p(h12)':>8}  {'Avg|r|':>7}")
print("-"*130)

for cfbd_col, pff_col, label in hybrids:
    # Normalize each metric to 0-100 using min-max within its population
    cfbd_vals = d[cfbd_col].dropna()
    pff_vals = d[pff_col].dropna()

    if len(cfbd_vals) < 5 or len(pff_vals) < 5:
        continue

    cfbd_min, cfbd_max = cfbd_vals.min(), cfbd_vals.max()
    pff_min, pff_max = pff_vals.min(), pff_vals.max()

    d['cfbd_norm'] = np.where(
        d[cfbd_col].notna(),
        (d[cfbd_col] - cfbd_min) / (cfbd_max - cfbd_min) * 100,
        np.nan
    )
    d['pff_norm'] = np.where(
        d[pff_col].notna(),
        (d[pff_col] - pff_min) / (pff_max - pff_min) * 100,
        np.nan
    )

    # Hybrid: use CFBD when available, PFF when not
    d['hybrid'] = np.where(d['cfbd_norm'].notna(), d['cfbd_norm'], d['pff_norm'])

    n_valid = d['hybrid'].notna().sum()
    cov = 100 * n_valid / len(d)

    # How many used each source?
    n_cfbd = d[cfbd_col].notna().sum()
    n_pff_only = (d[cfbd_col].isna() & d[pff_col].notna()).sum()

    rs = {}
    for outcome in ['first_3yr_ppg', 'hit24', 'hit12']:
        r, p, n = partial_corr(d['hybrid'], d[outcome], d['pick'])
        rs[outcome] = (r, p, n)

    r_ppg, p_ppg, _ = rs['first_3yr_ppg']
    r_h24, p_h24, _ = rs['hit24']
    r_h12, p_h12, _ = rs['hit12']

    avg_abs_r = np.nanmean([abs(r_ppg) if pd.notna(r_ppg) else 0,
                             abs(r_h24) if pd.notna(r_h24) else 0,
                             abs(r_h12) if pd.notna(r_h12) else 0])

    r_ppg_s = f"{r_ppg:+.3f}" if pd.notna(r_ppg) else '  N/A'
    p_ppg_s = f"{p_ppg:.4f}" if pd.notna(p_ppg) else '  N/A'
    r_h24_s = f"{r_h24:+.3f}" if pd.notna(r_h24) else '  N/A'
    p_h24_s = f"{p_h24:.4f}" if pd.notna(p_h24) else '  N/A'
    r_h12_s = f"{r_h12:+.3f}" if pd.notna(r_h12) else '  N/A'
    p_h12_s = f"{p_h12:.4f}" if pd.notna(p_h12) else '  N/A'

    print(f"{label:<50} {n_valid:>4} {cov:>4.0f}%  {r_ppg_s:>8} {p_ppg_s:>8}  {r_h24_s:>8} {p_h24_s:>8}  {r_h12_s:>8} {p_h12_s:>8}  {avg_abs_r:>6.3f}")
    print(f"  {'':50s} ({n_cfbd} CFBD + {n_pff_only} PFF fallback)")


# ============================================================
# TEST 2C: ALSO CHECK — does PFF alone beat CFBD alone (for the TEs that have both)?
# ============================================================
print(f"\n\n{'='*100}")
print("TEST 2C: HEAD-TO-HEAD ON SAME POPULATION (TEs that have BOTH CFBD and PFF)")
print("Apples-to-apples comparison on the ~68% of TEs that have data from both sources")
print(f"{'='*100}")

both_mask = d['ryptpa'].notna() & d['pff_yprr'].notna() & d['first_3yr_ppg'].notna()
print(f"\nTEs with both CFBD and PFF data: {both_mask.sum()}")

if both_mask.sum() >= 10:
    head2head = [
        ('rec_per_tpa', 'Receptions / Team Pass Att (CFBD)'),
        ('ryptpa', 'RYPTPA (CFBD)'),
        ('age_ryptpa', 'Age-Weighted RYPTPA (CFBD)'),
        ('pff_rpg', 'PFF Receptions Per Game'),
        ('pff_yprr', 'PFF Yards Per Route Run'),
        ('pff_target_rate', 'PFF Target Rate'),
    ]

    print(f"\n{'Metric':<45}  {'r(PPG)':>8} {'p(PPG)':>8}  {'r(h24)':>8} {'p(h24)':>8}  {'r(h12)':>8} {'p(h12)':>8}  {'Avg|r|':>7}")
    print("-"*115)

    for col, name in head2head:
        d_both = d[both_mask]
        rs = {}
        for outcome in ['first_3yr_ppg', 'hit24', 'hit12']:
            r, p, n = partial_corr(d_both[col], d_both[outcome], d_both['pick'])
            rs[outcome] = (r, p, n)

        r_ppg, p_ppg, _ = rs['first_3yr_ppg']
        r_h24, p_h24, _ = rs['hit24']
        r_h12, p_h12, _ = rs['hit12']

        avg_abs_r = np.nanmean([abs(r_ppg) if pd.notna(r_ppg) else 0,
                                 abs(r_h24) if pd.notna(r_h24) else 0,
                                 abs(r_h12) if pd.notna(r_h12) else 0])

        r_ppg_s = f"{r_ppg:+.3f}" if pd.notna(r_ppg) else '  N/A'
        p_ppg_s = f"{p_ppg:.4f}" if pd.notna(p_ppg) else '  N/A'
        r_h24_s = f"{r_h24:+.3f}" if pd.notna(r_h24) else '  N/A'
        p_h24_s = f"{p_h24:.4f}" if pd.notna(p_h24) else '  N/A'
        r_h12_s = f"{r_h12:+.3f}" if pd.notna(r_h12) else '  N/A'
        p_h12_s = f"{p_h12:.4f}" if pd.notna(p_h12) else '  N/A'

        print(f"{name:<45}  {r_ppg_s:>8} {p_ppg_s:>8}  {r_h24_s:>8} {p_h24_s:>8}  {r_h12_s:>8} {p_h12_s:>8}  {avg_abs_r:>6.3f}")


print(f"\n\n{'='*80}")
print("ANALYSIS COMPLETE — Review results before picking model structure.")
print(f"{'='*80}")
