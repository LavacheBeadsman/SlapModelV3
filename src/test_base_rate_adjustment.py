"""
Test multiplicative base rate adjustment: powers 0.15-0.30 in 0.05 steps.
Also test TE-exempt variant (only adjust WR/RB, leave TEs as-is).
"""
import pandas as pd
import numpy as np

master = pd.read_csv('output/slap_v5_master_database.csv')
bt = master[master['data_type'] == 'backtest'].copy()

# Position base hit rates
hit_rates = {}
for pos in ['WR', 'RB', 'TE']:
    sub = bt[bt['position'] == pos]
    hit_rates[pos] = sub['nfl_hit24'].mean()

avg_hit = np.mean(list(hit_rates.values()))

print("=" * 110)
print("POSITION BASE HIT RATES")
print("=" * 110)
for pos in ['WR', 'RB', 'TE']:
    print(f"  {pos}: {hit_rates[pos]:.3f} ({hit_rates[pos]*100:.1f}%)")
print(f"  Average: {avg_hit:.3f} ({avg_hit*100:.1f}%)")

# ============================================================================
# HELPER
# ============================================================================
def analyze(bt_df, label, adjust_fn):
    """Apply adjustment and show detailed breakdown."""
    df = bt_df.copy()
    df['slap_adjusted'] = df.apply(lambda r: adjust_fn(r['slap_v5'], r['position']), axis=1)
    df['slap_adjusted'] = df['slap_adjusted'].clip(0, 100).round(1)

    top60 = df.nlargest(60, 'slap_adjusted')
    pos_counts = top60['position'].value_counts().to_dict()

    print(f"\n{'─'*110}")
    print(f"  {label}")
    print(f"{'─'*110}")
    print(f"  Top 60: WR={pos_counts.get('WR',0)} | RB={pos_counts.get('RB',0)} | TE={pos_counts.get('TE',0)}")

    # Key player scores
    for name, label_p in [
        ('Saquon Barkley', 'Barkley(RB)'),
        ("Ja'Marr Chase", 'Chase(WR)'),
        ('Marvin Harrison Jr.', 'MHJ(WR)'),
    ]:
        row = df[df['player_name'] == name]
        if len(row) > 0:
            orig = row['slap_v5'].values[0]
            adj = row['slap_adjusted'].values[0]
            print(f"    {label_p}: {orig:.1f} → {adj:.1f} ({adj-orig:+.1f})")

    # Top 5 TEs
    te_df = df[df['position'] == 'TE'].nlargest(5, 'slap_adjusted')
    print(f"\n  Top 5 TEs:")
    for i, (_, r) in enumerate(te_df.iterrows(), 1):
        orig = r['slap_v5']
        adj = r['slap_adjusted']
        # Find overall rank
        rank = (df['slap_adjusted'] > adj).sum() + 1
        print(f"    {i}. {r['player_name']:<25} orig={orig:>5.1f}  adj={adj:>5.1f} ({adj-orig:+.1f})  overall rank #{rank}")

    # How many TEs in top 60
    te_in_60 = top60[top60['position'] == 'TE']
    print(f"  TEs in top 60: {len(te_in_60)}")
    if len(te_in_60) > 0:
        last_te = te_in_60.nsmallest(1, 'slap_adjusted').iloc[0]
        print(f"  Last TE in top 60: {last_te['player_name']} at {last_te['slap_adjusted']:.1f}")

    # Average by round
    print(f"\n  Avg SLAP by round (adjusted):")
    print(f"  {'Round':>5} | {'WR':>6} | {'RB':>6} | {'TE':>6} | {'RB-WR':>6}")
    for rd in [1, 2, 3, 4, 5]:
        vals = {}
        for pos in ['WR', 'RB', 'TE']:
            sub = df[(df['position'] == pos) & (df['round'] == rd)]
            vals[pos] = sub['slap_adjusted'].mean() if len(sub) > 0 else float('nan')
        rb_wr = vals['RB'] - vals['WR'] if not (np.isnan(vals['RB']) or np.isnan(vals['WR'])) else float('nan')
        print(f"  {rd:>5} | {vals['WR']:>6.1f} | {vals['RB']:>6.1f} | {vals['TE']:>6.1f} | {rb_wr:>+6.1f}")

    return pos_counts


# ============================================================================
# STRATEGY 1: PURE MULTIPLICATIVE (all positions) — powers 0.15 to 0.30
# ============================================================================
print(f"\n\n{'='*110}")
print("STRATEGY 1: MULTIPLICATIVE — SLAP × (position_hit_rate / avg_hit_rate) ^ power")
print("  Applied to ALL positions (WR, RB, TE)")
print("=" * 110)

for power in [0.15, 0.20, 0.25, 0.30]:
    factors = {pos: (hit_rates[pos] / avg_hit) ** power for pos in ['WR', 'RB', 'TE']}
    def make_mult(p):
        def adj(slap, pos):
            factor = (hit_rates[pos] / avg_hit) ** p
            return slap * factor
        return adj
    label = (f"MULT power={power:.2f}  "
             f"(RB ×{factors['RB']:.4f}, WR ×{factors['WR']:.4f}, TE ×{factors['TE']:.4f})")
    analyze(bt, label, make_mult(power))


# ============================================================================
# STRATEGY 2: TE-EXEMPT MULTIPLICATIVE — adjust WR/RB only, TEs stay as-is
# ============================================================================
print(f"\n\n{'='*110}")
print("STRATEGY 2: TE-EXEMPT MULTIPLICATIVE — adjust WR and RB only, TEs unchanged")
print("  Rationale: TEs use top12 hit definition (different bar than WR/RB top24)")
print("=" * 110)

# For TE-exempt, recalculate avg using only WR and RB
avg_hit_wr_rb = (hit_rates['WR'] + hit_rates['RB']) / 2
print(f"  WR/RB average hit rate: {avg_hit_wr_rb:.3f} ({avg_hit_wr_rb*100:.1f}%)")
print(f"  (Using WR+RB average as baseline for WR/RB adjustment)")

for power in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    factors_wr_rb = {pos: (hit_rates[pos] / avg_hit_wr_rb) ** power for pos in ['WR', 'RB']}
    def make_te_exempt(p, avg_wr_rb):
        def adj(slap, pos):
            if pos == 'TE':
                return slap  # No adjustment
            factor = (hit_rates[pos] / avg_wr_rb) ** p
            return slap * factor
        return adj
    label = (f"TE-EXEMPT power={power:.2f}  "
             f"(RB ×{factors_wr_rb['RB']:.4f}, WR ×{factors_wr_rb['WR']:.4f}, TE ×1.0000)")
    analyze(bt, label, make_te_exempt(power, avg_hit_wr_rb))


# ============================================================================
# SUMMARY TABLE
# ============================================================================
print(f"\n\n{'='*110}")
print("SUMMARY TABLE")
print("=" * 110)
print(f"  {'Config':<35} | {'WR':>3} | {'RB':>3} | {'TE':>3} | {'Barkley':>8} | {'Chase':>8} | {'MHJ':>8} | {'Pitts':>8} | {'Hock':>8}")
print(f"  {'─'*35}-+-{'─'*3}-+-{'─'*3}-+-{'─'*3}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}")

configs = []

# Baseline
configs.append(('Baseline (no adjust)', lambda s, p: s))

# Pure mult
for power in [0.15, 0.20, 0.25, 0.30]:
    def make_m(pw):
        return lambda s, p: s * (hit_rates[p] / avg_hit) ** pw
    configs.append((f'Mult power={power:.2f}', make_m(power)))

# TE-exempt
for power in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    def make_te(pw, avg_wr_rb=avg_hit_wr_rb):
        def adj(s, p):
            if p == 'TE':
                return s
            return s * (hit_rates[p] / avg_wr_rb) ** pw
        return adj
    configs.append((f'TE-exempt power={power:.2f}', make_te(power)))

for label, fn in configs:
    df = bt.copy()
    df['adj'] = df.apply(lambda r: fn(r['slap_v5'], r['position']), axis=1)
    df['adj'] = df['adj'].clip(0, 100).round(1)
    top60 = df.nlargest(60, 'adj')
    pc = top60['position'].value_counts().to_dict()

    scores = {}
    for name in ['Saquon Barkley', "Ja'Marr Chase", 'Marvin Harrison Jr.', 'Kyle Pitts', 'T.J. Hockenson']:
        row = df[df['player_name'] == name]
        scores[name] = row['adj'].values[0] if len(row) > 0 else float('nan')

    bark = scores['Saquon Barkley']
    chase = scores["Ja'Marr Chase"]
    mhj = scores['Marvin Harrison Jr.']
    pitts = scores['Kyle Pitts']
    hock = scores['T.J. Hockenson']
    print(f"  {label:<35} | {pc.get('WR',0):>3} | {pc.get('RB',0):>3} | {pc.get('TE',0):>3} | "
          f"{bark:>8.1f} | {chase:>8.1f} | {mhj:>8.1f} | {pitts:>8.1f} | {hock:>8.1f}")

print(f"\n  Target: WR 22-26, RB 22-26, TE 8-12")
print(f"  Barkley & Chase both approaching 95+")
