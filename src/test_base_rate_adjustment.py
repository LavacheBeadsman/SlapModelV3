"""
Test base rate adjustment on top of percentile + pooled rescaling.

Loads the master database (already built with pooled rescaling), then tests
several adjustment strategies and sizes to shift cross-position rankings
toward positions that historically hit more often.
"""
import pandas as pd
import numpy as np

master = pd.read_csv('output/slap_v5_master_database.csv')
bt = master[master['data_type'] == 'backtest'].copy()

# Position base hit rates (from backtest)
# WR/RB use hit24, TE uses hit24 (nfl_hit24 in master)
hit_rates = {}
for pos in ['WR', 'RB', 'TE']:
    sub = bt[bt['position'] == pos]
    hit_rates[pos] = sub['nfl_hit24'].mean()

avg_hit = np.mean(list(hit_rates.values()))

print("=" * 100)
print("POSITION BASE HIT RATES")
print("=" * 100)
for pos in ['WR', 'RB', 'TE']:
    print(f"  {pos}: {hit_rates[pos]:.3f} ({hit_rates[pos]*100:.1f}%)")
print(f"  Average: {avg_hit:.3f} ({avg_hit*100:.1f}%)")
print(f"  RB/WR ratio: {hit_rates['RB']/hit_rates['WR']:.2f}x")

# ============================================================================
# TEST ADJUSTMENT STRATEGIES
# ============================================================================

def test_adjustment(bt_df, label, adjust_fn):
    """Apply adjustment and show top 60 breakdown + key players."""
    df = bt_df.copy()
    df['slap_adjusted'] = df.apply(lambda r: adjust_fn(r['slap_v5'], r['position']), axis=1)
    df['slap_adjusted'] = df['slap_adjusted'].clip(0, 100).round(1)

    top60 = df.nlargest(60, 'slap_adjusted')
    pos_counts = top60['position'].value_counts().to_dict()

    # Key players
    barkley = df[df['player_name'] == 'Saquon Barkley']
    chase = df[df['player_name'] == "Ja'Marr Chase"]
    hock = df[df['player_name'] == 'T.J. Hockenson']
    mhj = df[df['player_name'] == 'Marvin Harrison Jr.']
    etienne = df[df['player_name'] == 'Travis Etienne']
    pitts = df[df['player_name'] == 'Kyle Pitts']

    print(f"\n{'─'*100}")
    print(f"  {label}")
    print(f"{'─'*100}")
    print(f"  Top 60: WR={pos_counts.get('WR',0)} | RB={pos_counts.get('RB',0)} | TE={pos_counts.get('TE',0)}")

    # Show #1 player
    top1 = top60.iloc[0]
    print(f"  #1: {top1['player_name']} ({top1['position']}) = {top1['slap_adjusted']:.1f}")

    # Key player scores
    for name, label_p, df_p in [
        ('Saquon Barkley', 'Barkley(RB)', barkley),
        ("Ja'Marr Chase", 'Chase(WR)', chase),
        ('Marvin Harrison Jr.', 'MHJ(WR)', mhj),
        ('T.J. Hockenson', 'Hock(TE)', hock),
        ('Kyle Pitts', 'Pitts(TE)', pitts),
        ('Travis Etienne', 'Etienne(RB)', etienne),
    ]:
        if len(df_p) > 0:
            orig = df_p['slap_v5'].values[0]
            adj = df_p['slap_adjusted'].values[0]
            print(f"    {label_p}: {orig:.1f} → {adj:.1f} ({adj-orig:+.1f})")

    # Average by round
    print(f"\n  Avg SLAP by round (adjusted):")
    print(f"  {'Round':>5} | {'WR':>6} | {'RB':>6} | {'TE':>6} | {'RB-WR':>6}")
    for rd in [1, 2, 3]:
        vals = {}
        for pos in ['WR', 'RB', 'TE']:
            sub = df[(df['position'] == pos) & (df['round'] == rd)]
            vals[pos] = sub['slap_adjusted'].mean() if len(sub) > 0 else float('nan')
        print(f"  {rd:>5} | {vals['WR']:>6.1f} | {vals['RB']:>6.1f} | {vals['TE']:>6.1f} | {vals['RB']-vals['WR']:>+6.1f}")

    # Ranking preservation
    for pos in ['WR', 'RB', 'TE']:
        sub = df[df['position'] == pos]
        spearman = sub['slap_v5'].rank().corr(sub['slap_adjusted'].rank(), method='spearman')
        print(f"  {pos} within-position Spearman: {spearman:.4f}")

    # Show top 20
    print(f"\n  Top 20:")
    top20 = df.nlargest(20, 'slap_adjusted')
    for i, (_, r) in enumerate(top20.iterrows(), 1):
        print(f"    {i:>2}. {r['player_name']:<25} {r['position']:>3} pk{int(r['pick']):>3}  "
              f"orig={r['slap_v5']:>5.1f}  adj={r['slap_adjusted']:>5.1f}")

    return pos_counts


# ============================================================================
# STRATEGY 1: ADDITIVE — add (hit_rate - avg_hit_rate) × scale_factor
# ============================================================================
print(f"\n\n{'='*100}")
print("STRATEGY: ADDITIVE — add (position_hit_rate - avg_hit_rate) × scale_factor")
print("=" * 100)

for scale in [20, 30, 40, 50, 60]:
    def make_adj(s):
        def adj(slap, pos):
            offset = (hit_rates[pos] - avg_hit) * s
            return slap + offset
        return adj
    test_adjustment(bt, f"ADDITIVE scale={scale}  (RB gets +{(hit_rates['RB']-avg_hit)*scale:.1f}, WR gets {(hit_rates['WR']-avg_hit)*scale:+.1f}, TE gets {(hit_rates['TE']-avg_hit)*scale:+.1f})", make_adj(scale))


# ============================================================================
# STRATEGY 2: MULTIPLICATIVE — multiply by (hit_rate / avg_hit_rate) ^ power
# ============================================================================
print(f"\n\n{'='*100}")
print("STRATEGY: MULTIPLICATIVE — SLAP × (position_hit_rate / avg_hit_rate) ^ power")
print("=" * 100)

for power in [0.3, 0.5, 0.7, 1.0]:
    def make_mult(p):
        def adj(slap, pos):
            factor = (hit_rates[pos] / avg_hit) ** p
            return slap * factor
        return adj
    factors = {pos: (hit_rates[pos] / avg_hit) ** power for pos in ['WR', 'RB', 'TE']}
    test_adjustment(bt, f"MULTIPLICATIVE power={power}  (RB ×{factors['RB']:.3f}, WR ×{factors['WR']:.3f}, TE ×{factors['TE']:.3f})", make_mult(power))


# ============================================================================
# STRATEGY 3: HYBRID — blend of additive base + multiplicative scaling
# Small additive floor + multiplicative component
# ============================================================================
print(f"\n\n{'='*100}")
print("STRATEGY: HYBRID — additive_offset + slap × mult_factor")
print("  offset = (hit_rate - avg) × additive_scale")
print("  factor = (hit_rate / avg) ^ mult_power")
print("=" * 100)

for add_s, mult_p in [(15, 0.3), (20, 0.3), (10, 0.5), (15, 0.5)]:
    def make_hybrid(a, m):
        def adj(slap, pos):
            offset = (hit_rates[pos] - avg_hit) * a
            factor = (hit_rates[pos] / avg_hit) ** m
            return slap * factor + offset
        return adj
    test_adjustment(bt, f"HYBRID add={add_s} mult={mult_p}", make_hybrid(add_s, mult_p))

print(f"\n\n{'='*100}")
print("DONE — review options above and pick the adjustment that looks right.")
print("=" * 100)
