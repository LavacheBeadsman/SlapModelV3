"""
TE Teammate Score — Expanded Test
Compare teammate definitions:
  A) WR + TE only (what we tested before)
  B) WR only (the real "competing for targets with elite WRs" signal)
  C) ALL pass catchers: WR + TE + RB
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir('/home/user/SlapModelV3')


def partial_corr(x, y, z):
    valid = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    if len(valid) < 15:
        return np.nan, np.nan, len(valid)
    sx, ix, _, _, _ = stats.linregress(valid['z'], valid['x'])
    rx = valid['x'] - (sx * valid['z'] + ix)
    sy, iy, _, _, _ = stats.linregress(valid['z'], valid['y'])
    ry = valid['y'] - (sy * valid['z'] + iy)
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(valid)


def dc_score(pick):
    return max(0, min(100, 100 - 2.40 * (pick ** 0.62 - 1)))


def normalize_name(name):
    if pd.isna(name): return ''
    s = str(name).strip().lower()
    for k, v in {'é':'e','è':'e','ê':'e','á':'a','à':'a','í':'i','ó':'o','ú':'u','ñ':'n'}.items():
        s = s.replace(k, v)
    s = s.replace('.', '').replace("'", '').replace('-', ' ')
    for suffix in [' iv', ' iii', ' ii', ' jr', ' sr', ' v']:
        if s.endswith(suffix): s = s[:-len(suffix)]
    return s.strip()


# Load data
bt = pd.read_csv('data/te_backtest_master.csv')
eval_df = bt[bt['draft_year'] < 2025].copy()
eval_df['s_dc'] = eval_df['pick'].apply(dc_score)

draft = pd.read_parquet('data/nflverse/draft_picks.parquet')
draft['s_dc'] = draft['pick'].apply(dc_score)
draft['name_norm'] = draft['pfr_player_name'].apply(normalize_name)

# Three teammate pools
pools = {
    'WR+TE':      draft[draft['position'].isin(['WR', 'TE'])].copy(),
    'WR_only':    draft[draft['position'] == 'WR'].copy(),
    'WR+TE+RB':   draft[draft['position'].isin(['WR', 'TE', 'RB'])].copy(),
}

outcomes = {
    'top12_10g':            'Top-12 TE (10g)',
    'top6_10g':             'Top-6 TE (10g)',
    'best_3yr_ppg_10g':     'Best 3yr PPG (10g)',
    'best_career_ppg_10g':  'Best career PPG (10g)',
    'seasons_over_10ppg_10g': 'Seasons 10+ PPG',
}

# Build teammate scores for each pool
for pool_name, pool_df in pools.items():
    dc_col = f'tm_dc_{pool_name}'
    n_col = f'tm_n_{pool_name}'
    tm_list = []

    for _, te in eval_df.iterrows():
        name = te['player_name']
        dy = te['draft_year']
        school = str(te['college']).lower()
        nn = normalize_name(name)

        same_class = pool_df[
            (pool_df['season'] == dy) &
            (pool_df['college'].str.lower() == school)
        ]
        teammates = same_class[same_class['name_norm'] != nn]
        tm_list.append({
            'player_name': name, 'draft_year': dy,
            dc_col: teammates['s_dc'].sum(),
            n_col: len(teammates),
        })

    tm_df = pd.DataFrame(tm_list)
    eval_df = eval_df.merge(tm_df, on=['player_name', 'draft_year'], how='left')

# ============================================================================
# COMPARE ALL THREE POOLS
# ============================================================================

print("=" * 130)
print("TE TEAMMATE SCORE — EXPANDED TEST (3 teammate pool definitions)")
print("=" * 130)

for pool_name in ['WR+TE', 'WR_only', 'WR+TE+RB']:
    dc_col = f'tm_dc_{pool_name}'
    n_col = f'tm_n_{pool_name}'

    print(f"\n{'─' * 100}")
    print(f"  POOL: {pool_name}")
    print(f"{'─' * 100}")

    # Distribution
    has_any = (eval_df[dc_col] > 0).sum()
    gt50 = (eval_df[dc_col] > 50).sum()
    gt100 = (eval_df[dc_col] > 100).sum()
    gt150 = (eval_df[dc_col] > 150).sum()
    print(f"  Distribution: any={has_any}, >50={gt50}, >100={gt100}, >150={gt150} (of {len(eval_df)} TEs)")
    print(f"  Mean={eval_df[dc_col].mean():.1f}, Median={eval_df[dc_col].median():.1f}, Max={eval_df[dc_col].max():.1f}")

    # Continuous partial correlations
    print(f"\n  Continuous teammate DC — partial correlations controlling for pick:")
    print(f"  {'Outcome':<25} {'r':>7} {'p':>8} {'n':>5} {'Sig?':>5}")
    print(f"  {'─'*55}")
    for oc_name, oc_label in outcomes.items():
        if oc_name not in eval_df.columns:
            continue
        r, p, n = partial_corr(eval_df[dc_col], eval_df[oc_name], eval_df['pick'])
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"  {oc_label:<25} {r:>+7.3f} {p:>8.4f} {n:>5} {sig:>5}")

    # Binary at multiple thresholds
    for threshold in [50, 75, 100, 150]:
        bin_col = f'tm_bin_{pool_name}_{threshold}'
        eval_df[bin_col] = (eval_df[dc_col] > threshold).astype(int)
        n_above = eval_df[bin_col].sum()
        if n_above < 3:
            continue

        print(f"\n  Binary (>{threshold}) — n={n_above} TEs above threshold:")
        print(f"  {'Outcome':<25} {'r':>7} {'p':>8} {'n':>5} {'Hit w/':>8} {'Hit w/o':>8}")
        print(f"  {'─'*70}")
        for oc_name, oc_label in outcomes.items():
            if oc_name not in eval_df.columns:
                continue
            r, p, n = partial_corr(eval_df[bin_col], eval_df[oc_name], eval_df['pick'])
            above = eval_df[eval_df[bin_col] == 1]
            below = eval_df[eval_df[bin_col] == 0]
            hit_above = above[oc_name].mean() if len(above) > 0 and oc_name in above.columns else np.nan
            hit_below = below[oc_name].mean() if len(below) > 0 and oc_name in below.columns else np.nan

            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            # Format hit rates depending on binary vs continuous
            if oc_name in ['top12_10g', 'top6_10g']:
                ha = f"{hit_above:.0%}" if pd.notna(hit_above) else "—"
                hb = f"{hit_below:.0%}" if pd.notna(hit_below) else "—"
            else:
                ha = f"{hit_above:.1f}" if pd.notna(hit_above) else "—"
                hb = f"{hit_below:.1f}" if pd.notna(hit_below) else "—"
            print(f"  {oc_label:<25} {r:>+7.3f} {p:>8.4f} {n:>5} {ha:>8} {hb:>8} {sig}")


# ============================================================================
# SHOW SPECIFIC EXAMPLES: TEs with highest WR teammate DC
# ============================================================================

print(f"\n\n{'=' * 130}")
print("TOP 15 TEs BY WR-ONLY TEAMMATE DC")
print("(TEs who produced despite sharing targets with elite drafted WRs)")
print("=" * 130)

top = eval_df.nlargest(15, 'tm_dc_WR_only')
print(f"\n  {'Player':<25} {'School':<20} {'Yr':>4} {'Pick':>5} {'WR tm DC':>9} {'n WRs':>6} "
      f"{'Top12':>6} {'PPG':>6}")
print(f"  {'─'*90}")
for _, r in top.iterrows():
    hit = "YES" if r.get('top12_10g', 0) == 1 else "no"
    ppg = f"{r['best_3yr_ppg_10g']:.1f}" if pd.notna(r.get('best_3yr_ppg_10g')) else "—"
    print(f"  {r['player_name']:<25} {r['college']:<20} {int(r['draft_year']):>4} "
          f"{int(r['pick']):>5} {r['tm_dc_WR_only']:>9.1f} {int(r['tm_n_WR_only']):>6} "
          f"{hit:>6} {ppg:>6}")

# Also show the WR+TE+RB version
print(f"\n\n{'=' * 130}")
print("TOP 15 TEs BY ALL-PASS-CATCHER TEAMMATE DC (WR+TE+RB)")
print("=" * 130)

top2 = eval_df.nlargest(15, 'tm_dc_WR+TE+RB')
print(f"\n  {'Player':<25} {'School':<20} {'Yr':>4} {'Pick':>5} {'All tm DC':>9} {'n tm':>6} "
      f"{'Top12':>6} {'PPG':>6}")
print(f"  {'─'*90}")
for _, r in top2.iterrows():
    hit = "YES" if r.get('top12_10g', 0) == 1 else "no"
    ppg = f"{r['best_3yr_ppg_10g']:.1f}" if pd.notna(r.get('best_3yr_ppg_10g')) else "—"
    print(f"  {r['player_name']:<25} {r['college']:<20} {int(r['draft_year']):>4} "
          f"{int(r['pick']):>5} {r['tm_dc_WR+TE+RB']:>9.1f} {int(r['tm_n_WR+TE+RB']):>6} "
          f"{hit:>6} {ppg:>6}")
