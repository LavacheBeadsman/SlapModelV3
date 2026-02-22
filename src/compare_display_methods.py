"""
Compare three display-score methods side by side.
READ-ONLY: reads slap_v5_master_database.csv, writes NO files.

Methods:
  Current  = min-max rescaling (backtest min→1, max→99)
  Method 1 = percentile rank against backtest population
  Method 2 = z-score → normal CDF against backtest distribution
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, percentileofscore, skew
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('output/slap_v5_master_database.csv')

bt = df[df['data_type'] == 'backtest'].copy()
p26 = df[df['data_type'] == '2026_prospect'].copy()

# ─────────────────────────────────────────────────────────────
# BACKTEST STATS PER POSITION
# ─────────────────────────────────────────────────────────────
stats = {}
for pos in ['WR', 'RB', 'TE']:
    vals = bt[bt['position'] == pos]['slap_raw'].dropna().values
    stats[pos] = {
        'min': vals.min(),
        'max': vals.max(),
        'mean': vals.mean(),
        'std': vals.std(ddof=1),
        'median': np.median(vals),
        'skew': skew(vals),
        'n': len(vals),
        'values': vals,
    }

# ─────────────────────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────────────────────
def score_minmax(raw, pos):
    mn, mx = stats[pos]['min'], stats[pos]['max']
    if mx == mn:
        return 50.0
    return np.clip(1 + (raw - mn) / (mx - mn) * 98, 1, 99)

def score_percentile(raw, pos):
    pctl = percentileofscore(stats[pos]['values'], raw, kind='rank')
    return np.clip(pctl / 100 * 98 + 1, 1, 99)

def score_zscore(raw, pos):
    z = (raw - stats[pos]['mean']) / stats[pos]['std']
    return np.clip(norm.cdf(z) * 98 + 1, 1, 99)

# ─────────────────────────────────────────────────────────────
# APPLY ALL THREE METHODS TO EVERY ROW
# ─────────────────────────────────────────────────────────────
for frame in [bt, p26]:
    frame['score_minmax'] = frame.apply(
        lambda r: round(score_minmax(r['slap_raw'], r['position']), 1), axis=1)
    frame['score_pctl'] = frame.apply(
        lambda r: round(score_percentile(r['slap_raw'], r['position']), 1), axis=1)
    frame['score_zscore'] = frame.apply(
        lambda r: round(score_zscore(r['slap_raw'], r['position']), 1), axis=1)

all_df = pd.concat([bt, p26], ignore_index=True)

# ─────────────────────────────────────────────────────────────
# TABLE PRINTING HELPERS
# ─────────────────────────────────────────────────────────────
def print_header(title):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")

def print_row(vals, widths, aligns):
    parts = []
    for v, w, a in zip(vals, widths, aligns):
        s = str(v)
        if a == 'L':
            parts.append(s.ljust(w))
        else:
            parts.append(s.rjust(w))
    print('  '.join(parts))

def print_comparison_table(subset, title, show_gap=False):
    """Print a comparison table for a set of players."""
    print(f"\n  {title}")
    print(f"  {'-'*94}")

    hdrs = ['#', 'Player', 'Pos', 'Pick', 'Raw', 'MinMax', 'Pctl', 'ZScore']
    widths = [4, 28, 4, 5, 7, 7, 7, 7]
    aligns = ['R', 'L', 'L', 'R', 'R', 'R', 'R', 'R']

    if show_gap:
        hdrs += ['MM-gap', 'P-gap', 'Z-gap']
        widths += [7, 7, 7]
        aligns += ['R', 'R', 'R']

    print_row(hdrs, widths, aligns)
    print(f"  {'-'*94}")

    prev_mm = prev_p = prev_z = None
    for i, (_, row) in enumerate(subset.iterrows(), 1):
        vals = [
            i,
            row['player_name'][:27],
            row['position'],
            int(row['pick']),
            f"{row['slap_raw']:.1f}",
            f"{row['score_minmax']:.1f}",
            f"{row['score_pctl']:.1f}",
            f"{row['score_zscore']:.1f}",
        ]
        if show_gap:
            if prev_mm is not None:
                vals += [
                    f"{row['score_minmax'] - prev_mm:+.1f}",
                    f"{row['score_pctl'] - prev_p:+.1f}",
                    f"{row['score_zscore'] - prev_z:+.1f}",
                ]
            else:
                vals += ['', '', '']
            prev_mm = row['score_minmax']
            prev_p = row['score_pctl']
            prev_z = row['score_zscore']

        print_row(vals, widths, aligns)


# ═════════════════════════════════════════════════════════════
# TABLE 1: McCaffrey vs Cooper TEST
# ═════════════════════════════════════════════════════════════
print_header("TABLE 1: THE McCaffrey vs Cooper TEST")
print("  Do elite players at different positions get comparable scores?\n")

mccaffrey = all_df[all_df['player_name'] == 'Christian McCaffrey']
cooper = all_df[all_df['player_name'] == 'Amari Cooper']
test_df = pd.concat([cooper, mccaffrey]).reset_index(drop=True)

hdrs = ['Player', 'Pos', 'Pick', 'Raw', 'MinMax', 'Pctl', 'ZScore']
widths = [28, 4, 5, 7, 7, 7, 7]
aligns = ['L', 'L', 'R', 'R', 'R', 'R', 'R']
print_row(hdrs, widths, aligns)
print(f"  {'-'*70}")

for _, row in test_df.iterrows():
    print_row([
        row['player_name'],
        row['position'],
        int(row['pick']),
        f"{row['slap_raw']:.2f}",
        f"{row['score_minmax']:.1f}",
        f"{row['score_pctl']:.1f}",
        f"{row['score_zscore']:.1f}",
    ], widths, aligns)

# Delta row
c = cooper.iloc[0]
m = mccaffrey.iloc[0]
print(f"  {'-'*70}")
print_row([
    'DELTA (Cooper - McCaffrey)',
    '',
    '',
    f"{c['slap_raw'] - m['slap_raw']:+.2f}",
    f"{c['score_minmax'] - m['score_minmax']:+.1f}",
    f"{c['score_pctl'] - m['score_pctl']:+.1f}",
    f"{c['score_zscore'] - m['score_zscore']:+.1f}",
], widths, aligns)

print(f"""
  INTERPRETATION:
  - MinMax: {c['score_minmax']:.1f} vs {m['score_minmax']:.1f} = {abs(c['score_minmax'] - m['score_minmax']):.1f} pt gap (reflects different raw ranges per position)
  - Percentile: {c['score_pctl']:.1f} vs {m['score_pctl']:.1f} = {abs(c['score_pctl'] - m['score_pctl']):.1f} pt gap (both near top of their position)
  - Z-Score: {c['score_zscore']:.1f} vs {m['score_zscore']:.1f} = {abs(c['score_zscore'] - m['score_zscore']):.1f} pt gap (similar, CDF-based)
""")


# ═════════════════════════════════════════════════════════════
# TABLE 2: TOP 10 BACKTEST PER POSITION
# ═════════════════════════════════════════════════════════════
print_header("TABLE 2: TOP 10 BACKTEST BY POSITION")

for pos in ['WR', 'RB', 'TE']:
    sub = bt[bt['position'] == pos].nlargest(10, 'slap_raw')
    print_comparison_table(sub, f"TOP 10 {pos} BACKTEST (n={stats[pos]['n']})")
    print()


# ═════════════════════════════════════════════════════════════
# TABLE 3: TOP 10 2026 PROSPECTS PER POSITION
# ═════════════════════════════════════════════════════════════
print_header("TABLE 3: TOP 10 2026 PROSPECTS BY POSITION")

for pos in ['WR', 'RB', 'TE']:
    sub = p26[p26['position'] == pos].nlargest(10, 'slap_raw')
    n26 = len(p26[p26['position'] == pos])
    print_comparison_table(sub, f"TOP 10 {pos} 2026 PROSPECTS (n={n26}, scored vs {stats[pos]['n']} backtest)")
    print()


# ═════════════════════════════════════════════════════════════
# TABLE 4: CROSS-POSITION COMPARISONS
# ═════════════════════════════════════════════════════════════
print_header("TABLE 4: CROSS-POSITION COMPARISONS")
print("  Best player per position at similar draft slots\n")

pick_ranges = [
    ("PICK 4-8", 4, 8),
    ("PICK 20-30", 20, 30),
    ("PICK 50-70", 50, 70),
]

for label, lo, hi in pick_ranges:
    print(f"\n  {label} (BACKTEST)")
    print(f"  {'-'*80}")
    hdrs = ['Player', 'Pos', 'Pick', 'Year', 'Raw', 'MinMax', 'Pctl', 'ZScore']
    widths = [28, 4, 5, 5, 7, 7, 7, 7]
    aligns = ['L', 'L', 'R', 'R', 'R', 'R', 'R', 'R']
    print_row(hdrs, widths, aligns)
    print(f"  {'-'*80}")

    for pos in ['WR', 'RB', 'TE']:
        sub = bt[(bt['position'] == pos) & (bt['pick'] >= lo) & (bt['pick'] <= hi)]
        if len(sub) == 0:
            print(f"  {'(no ' + pos + ' in range)':<28}")
            continue
        row = sub.nlargest(1, 'slap_raw').iloc[0]
        print_row([
            row['player_name'][:27],
            pos,
            int(row['pick']),
            int(row['draft_year']),
            f"{row['slap_raw']:.1f}",
            f"{row['score_minmax']:.1f}",
            f"{row['score_pctl']:.1f}",
            f"{row['score_zscore']:.1f}",
        ], widths, aligns)

# 2026 cross-position: top prospect per position
print(f"\n\n  TOP 2026 PROSPECT PER POSITION (any pick)")
print(f"  {'-'*80}")
hdrs = ['Player', 'Pos', 'Pick', 'Raw', 'MinMax', 'Pctl', 'ZScore']
widths = [28, 4, 5, 7, 7, 7, 7]
aligns = ['L', 'L', 'R', 'R', 'R', 'R', 'R']
print_row(hdrs, widths, aligns)
print(f"  {'-'*80}")

for pos in ['WR', 'RB', 'TE']:
    sub = p26[p26['position'] == pos]
    if len(sub) == 0:
        continue
    row = sub.nlargest(1, 'slap_raw').iloc[0]
    print_row([
        row['player_name'][:27],
        pos,
        int(row['pick']),
        f"{row['slap_raw']:.1f}",
        f"{row['score_minmax']:.1f}",
        f"{row['score_pctl']:.1f}",
        f"{row['score_zscore']:.1f}",
    ], widths, aligns)


# ═════════════════════════════════════════════════════════════
# TABLE 5: DISTRIBUTION STATS
# ═════════════════════════════════════════════════════════════
print_header("TABLE 5: BACKTEST slap_raw DISTRIBUTION STATS")
print("  These stats determine how each method behaves.\n")

hdrs = ['Pos', 'N', 'Mean', 'Std', 'Min', 'Max', 'Median', 'Skew', 'Range']
widths = [4, 5, 7, 7, 7, 7, 7, 7, 7]
aligns = ['L', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R']
print_row(hdrs, widths, aligns)
print(f"  {'-'*66}")

for pos in ['WR', 'RB', 'TE']:
    s = stats[pos]
    print_row([
        pos,
        s['n'],
        f"{s['mean']:.1f}",
        f"{s['std']:.1f}",
        f"{s['min']:.1f}",
        f"{s['max']:.1f}",
        f"{s['median']:.1f}",
        f"{s['skew']:.3f}",
        f"{s['max'] - s['min']:.1f}",
    ], widths, aligns)

print(f"""
  KEY OBSERVATIONS:
  - RB has highest skew ({stats['RB']['skew']:.3f}) — Saquon is a massive outlier (raw={stats['RB']['max']:.1f})
  - RB has widest range ({stats['RB']['max'] - stats['RB']['min']:.1f}) — this compresses all non-Saquon RBs under MinMax
  - TE has narrowest range ({stats['TE']['max'] - stats['TE']['min']:.1f}) — MinMax inflates TE scores relative to other positions
  - All three positions are right-skewed — z-score (normal assumption) will slightly misfit the tails
""")


# ═════════════════════════════════════════════════════════════
# TABLE 6: GRANULARITY CHECK (Ranks #5-#15)
# ═════════════════════════════════════════════════════════════
print_header("TABLE 6: GRANULARITY CHECK — Ranks #5 to #15 with gaps between adjacent players")
print("  Do the methods create enough separation between closely-ranked players?\n")

for pos in ['WR', 'RB', 'TE']:
    sub = bt[bt['position'] == pos].nlargest(15, 'slap_raw').tail(11)  # ranks 5-15
    print_comparison_table(sub, f"{pos} RANKS #5-#15 (BACKTEST)", show_gap=True)
    print()

    # Summary stats for gaps
    mm_gaps = sub['score_minmax'].diff().dropna().abs()
    p_gaps = sub['score_pctl'].diff().dropna().abs()
    z_gaps = sub['score_zscore'].diff().dropna().abs()
    print(f"    Avg gap:  MinMax={mm_gaps.mean():.2f}   Pctl={p_gaps.mean():.2f}   ZScore={z_gaps.mean():.2f}")
    print(f"    Min gap:  MinMax={mm_gaps.min():.2f}   Pctl={p_gaps.min():.2f}   ZScore={z_gaps.min():.2f}")
    print(f"    Max gap:  MinMax={mm_gaps.max():.2f}   Pctl={p_gaps.max():.2f}   ZScore={z_gaps.max():.2f}")
    print()


# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print_header("SUMMARY: METHOD COMPARISON")

print("""
  CURRENT METHOD (Min-Max Rescaling):
    How it works: backtest_min → 1, backtest_max → 99, linear interpolation
    Pros:  Preserves raw spacing; linear; easy to explain
    Cons:  Cross-position scores NOT comparable (TE 82 != RB 82)
           Sensitive to outliers (Saquon at 94.9 compresses all other RBs)
           2026 prospects below backtest min clip to 1

  METHOD 1 (Percentile Rank):
    How it works: "what % of backtest players did this player beat?"
    Pros:  Cross-position scores ARE comparable (90 = top 10% in any position)
           Not sensitive to outliers; robust
           Easy to explain to audience
    Cons:  Destroys raw spacing — tiny raw gaps become equal display gaps
           Compresses elite players (top 5 all score 97-99)
           Equal-spaced ranks lose "how much better" information

  METHOD 2 (Z-Score + Normal CDF):
    How it works: convert to standard deviations from mean, then CDF
    Pros:  Cross-position somewhat comparable
           Preserves relative spacing better than percentile in the middle
    Cons:  Assumes normality — all 3 positions are right-skewed
           Still compresses elites (similar to percentile in tails)
           Harder to explain to a non-technical audience
""")
