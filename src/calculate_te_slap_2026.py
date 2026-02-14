"""
TE SLAP Score Calculator — 2026 Draft Class
============================================
Weights: 65% DC / 15% Breakout / 10% Production / 5% Early Declare / 5% RAS

Breakout: 15% dominator threshold (TE-specific)
RAS: MNAR-imputed (no combine data yet) — Rd 1-2 → 60th pctl, Rd 3+ → 40th pctl
Production: Receiving yards / team pass attempts (2025 final season), age-weighted
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')

from datetime import datetime, date

DRAFT_DATE = date(2026, 4, 25)

# ============================================================================
# STEP 1: Load raw data and apply verified birthdate corrections
# ============================================================================
print("=" * 120)
print("TE SLAP SCORE CALCULATOR — 2026 DRAFT CLASS")
print("Weights: 65% DC / 15% Breakout / 10% Production / 5% Early Declare / 5% RAS")
print("=" * 120)

df = pd.read_csv('data/te_2026_prospects_raw.csv')

# Verified birthdate corrections (web-searched, high confidence)
VERIFIED_CORRECTIONS = {
    "Kenyon Sadiq":  ("2005-03-04", "verified"),   # Wikipedia, Tankathon — was Feb 17, 2003
    "Eli Stowers":   ("2003-04-15", "verified"),   # Wikipedia, 247Sports — was Oct 27, 2002
    "Max Klare":     ("2003-07-08", "verified"),   # Wikipedia, Baseball-Ref — was Sep 19, 2002
    "Michael Trigg": ("2002-06-27", "verified"),   # Wikipedia, Steelers Depot — was Jun 14, 2002
}

for name, (bd_str, source) in VERIFIED_CORRECTIONS.items():
    mask = df['player_name'] == name
    old_bd = df.loc[mask, 'birthdate'].values[0]
    bd = datetime.strptime(bd_str, "%Y-%m-%d").date()
    new_age = round((DRAFT_DATE - bd).days / 365.25, 1)
    old_age = df.loc[mask, 'draft_age'].values[0]
    df.loc[mask, 'birthdate'] = bd_str
    df.loc[mask, 'birthdate_source'] = source
    df.loc[mask, 'draft_age'] = new_age
    print(f"  CORRECTED: {name} — {old_bd} -> {bd_str} (age {old_age} -> {new_age})")

print()

# ============================================================================
# STEP 2: Recalculate breakout ages with corrected birthdates
# ============================================================================
print("Recalculating breakout ages with corrected birthdates (15% dominator)...")

# Load PFF data for all seasons
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
    try:
        pff = pd.read_csv(fp)
        pff['season'] = season
        all_pff.append(pff)
    except Exception as e:
        print(f"  WARNING: Could not load {fp}: {e}")

pff_all = pd.concat(all_pff, ignore_index=True)
receivers = pff_all[pff_all['position'].isin(['TE', 'WR', 'HB'])].copy()
team_totals = receivers.groupby(['team_name', 'season'])['yards'].sum().reset_index()
team_totals.rename(columns={'yards': 'team_rec_yards'}, inplace=True)
te_pff = receivers[receivers['position'] == 'TE'].copy()
te_pff = te_pff.merge(team_totals, on=['team_name', 'season'], how='left')
te_pff['dominator_pct'] = np.where(te_pff['team_rec_yards'] > 0,
    (te_pff['yards'] / te_pff['team_rec_yards']) * 100, 0)


def norm_name(n):
    s = str(n).lower().strip()
    for suf in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
        s = s.replace(suf, '')
    s = s.replace("'", "").replace("\u2019", "").replace("-", "").replace(".", "")
    return s.strip()

te_pff['name_norm'] = te_pff['player'].apply(norm_name)

THRESH = 15  # TE-specific dominator threshold

# Recalculate breakout for all prospects using corrected ages
breakout_results = {}
pff_season_detail = {}

for _, prospect in df.iterrows():
    name = prospect['player_name']
    nn = norm_name(name)
    draft_age = prospect['draft_age']

    matches = te_pff[te_pff['name_norm'] == nn]
    if len(matches) == 0:
        parts = nn.split()
        if len(parts) >= 2:
            last = parts[-1]
            partial = te_pff[te_pff['name_norm'].str.contains(last, na=False)]
            for _, pm in partial.iterrows():
                pn = norm_name(pm['player'])
                if pn.split()[-1] == last and pn[0] == nn[0]:
                    matches = partial[partial['player'] == pm['player']]
                    break

    if len(matches) == 0:
        breakout_results[name] = {
            'breakout_age': None, 'peak_dominator': None,
            'breakout_score': None, 'pff_seasons': 0,
            'season_detail': []
        }
        continue

    # Compute dominator by season
    season_data = []
    for _, pm in matches.sort_values('season').iterrows():
        season_age = draft_age - (2026 - pm['season'])
        season_data.append({
            'season': int(pm['season']),
            'season_age': round(season_age, 1),
            'dominator_pct': round(pm['dominator_pct'], 2),
            'rec_yards': pm['yards'],
            'receptions': pm['receptions'],
        })

    peak_dom = matches['dominator_pct'].max()
    hit_ages = []
    for sd in season_data:
        if sd['dominator_pct'] >= THRESH:
            hit_ages.append(sd['season_age'])

    if hit_ages:
        bo_age = hit_ages[0]  # First season hitting threshold
        # Age tier scoring
        if bo_age <= 18: base = 100
        elif bo_age <= 19: base = 90
        elif bo_age <= 20: base = 75
        elif bo_age <= 21: base = 60
        elif bo_age <= 22: base = 45
        elif bo_age <= 23: base = 30
        else: base = 20
        bonus = min((peak_dom - THRESH) * 0.5, 9.9)
        score = min(base + bonus, 99.9)
    else:
        bo_age = None
        score = min(35, 15 + peak_dom)

    breakout_results[name] = {
        'breakout_age': bo_age,
        'peak_dominator': peak_dom,
        'breakout_score': score,
        'pff_seasons': len(matches['season'].unique()),
        'season_detail': season_data,
    }

# Apply recalculated breakout data
df['breakout_age'] = df['player_name'].map(lambda n: breakout_results.get(n, {}).get('breakout_age'))
df['peak_dominator'] = df['player_name'].map(lambda n: breakout_results.get(n, {}).get('peak_dominator'))
df['breakout_score'] = df['player_name'].map(lambda n: breakout_results.get(n, {}).get('breakout_score'))
df['pff_seasons'] = df['player_name'].map(lambda n: breakout_results.get(n, {}).get('pff_seasons'))

# Show breakout changes for corrected players
print("\n  Breakout recalculation for birthdate-corrected players:")
for name in VERIFIED_CORRECTIONS:
    br = breakout_results.get(name, {})
    bo = br.get('breakout_age')
    sc = br.get('breakout_score')
    pk = br.get('peak_dominator')
    detail = br.get('season_detail', [])
    print(f"\n  {name} (draft age {df.loc[df['player_name']==name, 'draft_age'].values[0]}):")
    print(f"    Peak dominator: {pk:.1f}%  |  Breakout age: {bo}  |  Score: {sc:.1f}" if bo else
          f"    Peak dominator: {pk:.1f}%  |  Never hit {THRESH}%  |  Score: {sc:.1f}")
    for sd in detail:
        flag = " <-- BREAKOUT" if sd['dominator_pct'] >= THRESH else ""
        print(f"    {sd['season']}: age {sd['season_age']:.1f}, dom {sd['dominator_pct']:.1f}%, "
              f"rec {sd['receptions']:.0f}, yards {sd['rec_yards']:.0f}{flag}")


# ============================================================================
# STEP 3: Calculate all SLAP components
# ============================================================================
print(f"\n\n{'='*120}")
print("CALCULATING SLAP COMPONENTS")
print("=" * 120)

# --- Component 1: Draft Capital (65%) ---
# DC = 100 - 2.40 × (pick^0.62 - 1)
df['dc_score'] = 100 - 2.40 * (df['projected_pick'] ** 0.62 - 1)
df['dc_score'] = df['dc_score'].clip(lower=0, upper=100)

print(f"\n  DC Score (65% weight):")
print(f"    Range: {df['dc_score'].min():.1f} — {df['dc_score'].max():.1f}")
print(f"    Mean: {df['dc_score'].mean():.1f}")


# --- Component 2: Breakout Score (15%) ---
# Already calculated above. Fill missing with group average.
bo_avg = df['breakout_score'].mean()
df['breakout_score_filled'] = df['breakout_score'].fillna(bo_avg)

print(f"\n  Breakout Score (15% weight, 15% dominator threshold):")
print(f"    Range: {df['breakout_score_filled'].min():.1f} — {df['breakout_score_filled'].max():.1f}")
print(f"    Mean: {df['breakout_score_filled'].mean():.1f}")
print(f"    Missing (imputed w/ avg={bo_avg:.1f}): {df['breakout_score'].isna().sum()}")


# --- Component 3: Production — Receiving (10%) ---
# rec_yards / team_pass_att × age_weight × 100, normalized 0-99.9
# Use CFBD data (primary), PFF data (fallback)
def get_age_weight(draft_age, draft_year=2026, season_year=2025):
    """Age weight for final college season."""
    season_age = draft_age - (draft_year - season_year)
    if season_age <= 19: return 1.15
    elif season_age <= 20: return 1.10
    elif season_age <= 21: return 1.05
    elif season_age <= 22: return 1.00
    elif season_age <= 23: return 0.95
    else: return 0.90

production_raw = []
for _, r in df.iterrows():
    rec_yds = r['cfbd_rec_yards']
    team_pa = r['cfbd_team_pass_att']

    # Fallback to PFF if CFBD missing
    if pd.isna(rec_yds) and pd.notna(r.get('pff_yards')):
        rec_yds = r['pff_yards']
    if pd.isna(team_pa):
        team_pa = np.nan

    if pd.notna(rec_yds) and pd.notna(team_pa) and team_pa > 0:
        age_wt = get_age_weight(r['draft_age'])
        raw = (rec_yds / team_pa) * age_wt * 100
        production_raw.append(raw)
    else:
        production_raw.append(np.nan)

df['production_raw'] = production_raw

# Normalize to 0-99.9 using min-max within the group
prod_min = df['production_raw'].min()
prod_max = df['production_raw'].max()
df['production_score'] = np.where(
    df['production_raw'].notna(),
    (df['production_raw'] - prod_min) / (prod_max - prod_min) * 99.9,
    np.nan
)
prod_avg = df['production_score'].mean()
df['production_score_filled'] = df['production_score'].fillna(prod_avg)

print(f"\n  Production Score (10% weight, rec_yds/team_pa × age_wt):")
print(f"    Raw range: {prod_min:.2f} — {prod_max:.2f}")
print(f"    Scaled range: {df['production_score_filled'].min():.1f} — {df['production_score_filled'].max():.1f}")
print(f"    Mean: {df['production_score_filled'].mean():.1f}")
print(f"    Missing (imputed w/ avg): {df['production_score'].isna().sum()}")

# Show top 10 producers
print(f"\n    Top 10 by production:")
top_prod = df.nlargest(10, 'production_score')
for _, r in top_prod.iterrows():
    print(f"      {r['player_name']:<28} {r['production_score']:.1f} (rec_yds={r['cfbd_rec_yards']:.0f}, "
          f"team_pa={r['cfbd_team_pass_att']:.0f})")


# --- Component 4: Early Declare (5%) ---
df['early_declare_score'] = df['early_declare'] * 100

print(f"\n  Early Declare Score (5% weight):")
print(f"    Early declares (100): {(df['early_declare_score']==100).sum()}")
print(f"    Non-early (0): {(df['early_declare_score']==0).sum()}")


# --- Component 5: RAS — MNAR Imputed (5%) ---
# No combine data yet. Rd 1-2 projected → 60, Rd 3+ → 40
def impute_ras(pick):
    if pick <= 64:  # Rd 1-2
        return 60.0
    else:  # Rd 3+
        return 40.0

df['ras_score'] = df['projected_pick'].apply(impute_ras)

print(f"\n  RAS Score (5% weight, MNAR-imputed — no combine data yet):")
print(f"    Rd 1-2 (60th pctl = 60): {(df['ras_score']==60).sum()} prospects")
print(f"    Rd 3+ (40th pctl = 40):  {(df['ras_score']==40).sum()} prospects")


# ============================================================================
# STEP 4: Calculate SLAP Scores
# ============================================================================
print(f"\n\n{'='*120}")
print("CALCULATING SLAP SCORES: 65% DC + 15% Breakout + 10% Production + 5% Early Declare + 5% RAS")
print("=" * 120)

# Weighted formula
df['slap_score'] = (
    df['dc_score'] * 0.65 +
    df['breakout_score_filled'] * 0.15 +
    df['production_score_filled'] * 0.10 +
    df['early_declare_score'] * 0.05 +
    df['ras_score'] * 0.05
)

# DC-only score for delta calculation
df['dc_only_slap'] = df['dc_score']

# Delta = SLAP - DC-only
df['delta_vs_dc'] = df['slap_score'] - df['dc_only_slap']

# ============================================================================
# STEP 5: Age sensitivity analysis
# ============================================================================
print(f"\nAge sensitivity analysis (would breakout score change >3 pts if age off by 1 year?)...\n")

age_sensitive = []
for _, r in df.iterrows():
    name = r['player_name']
    br = breakout_results.get(name, {})
    bo_age = br.get('breakout_age')
    peak_dom = br.get('peak_dominator')
    current_score = br.get('breakout_score')

    if current_score is None or peak_dom is None:
        continue

    # Simulate age +1 year (older)
    if bo_age is not None:
        bo_older = bo_age + 1
        if bo_older <= 18: base_o = 100
        elif bo_older <= 19: base_o = 90
        elif bo_older <= 20: base_o = 75
        elif bo_older <= 21: base_o = 60
        elif bo_older <= 22: base_o = 45
        elif bo_older <= 23: base_o = 30
        else: base_o = 20
        bonus = min((peak_dom - THRESH) * 0.5, 9.9)
        score_older = min(base_o + bonus, 99.9)
    else:
        score_older = current_score  # "never hit" doesn't depend on age

    # Simulate age -1 year (younger)
    if bo_age is not None:
        bo_younger = bo_age - 1
        if bo_younger <= 18: base_y = 100
        elif bo_younger <= 19: base_y = 90
        elif bo_younger <= 20: base_y = 75
        elif bo_younger <= 21: base_y = 60
        elif bo_younger <= 22: base_y = 45
        elif bo_younger <= 23: base_y = 30
        else: base_y = 20
        bonus = min((peak_dom - THRESH) * 0.5, 9.9)
        score_younger = min(base_y + bonus, 99.9)
    else:
        score_younger = current_score

    max_delta = max(abs(score_older - current_score), abs(score_younger - current_score))

    if max_delta > 3:
        age_sensitive.append({
            'name': name, 'birthdate_source': r['birthdate_source'],
            'current_bo_age': bo_age, 'current_score': current_score,
            'score_if_older': score_older, 'score_if_younger': score_younger,
            'max_delta': max_delta,
            'slap_impact': max_delta * 0.15,  # 15% weight on breakout
        })

if age_sensitive:
    print(f"  {'Player':<28} {'BD Source':<10} {'BO Age':>6} {'Score':>6} {'If +1yr':>7} {'If -1yr':>7} {'Max Δ':>6} {'SLAP Δ':>7}")
    print(f"  {'-'*90}")
    for a in sorted(age_sensitive, key=lambda x: -x['max_delta']):
        print(f"  {a['name']:<28} {a['birthdate_source']:<10} {a['current_bo_age']:>6.1f} {a['current_score']:>6.1f} "
              f"{a['score_if_older']:>7.1f} {a['score_if_younger']:>7.1f} {a['max_delta']:>6.1f} {a['slap_impact']:>6.1f}")
else:
    print("  No players with >3 point sensitivity to age estimates.")

# Also check: estimated-birthday players who broke out near a tier boundary
print(f"\n  Players with estimated birthdates whose breakout age is near a tier boundary:")
boundary_risk = []
for _, r in df.iterrows():
    if r['birthdate_source'] != 'estimated':
        continue
    br = breakout_results.get(r['player_name'], {})
    bo = br.get('breakout_age')
    if bo is None:
        continue
    # Check if within 0.5 of a tier boundary (19, 20, 21, 22, 23)
    for boundary in [19, 20, 21, 22, 23]:
        if abs(bo - boundary) < 0.6:
            boundary_risk.append({
                'name': r['player_name'], 'bo_age': bo, 'near_boundary': boundary,
                'current_score': br['breakout_score'],
            })
            break

if boundary_risk:
    for b in boundary_risk:
        print(f"    {b['name']:<28} BO age={b['bo_age']:.1f}, near {b['near_boundary']} boundary, score={b['current_score']:.1f}")


# ============================================================================
# STEP 6: FULL RANKINGS
# ============================================================================
print(f"\n\n{'='*120}")
print("2026 TE SLAP RANKINGS — V5 (65/15/10/5/5)")
print(f"{'='*120}")

df_ranked = df.sort_values('slap_score', ascending=False).reset_index(drop=True)
df_ranked['rank'] = range(1, len(df_ranked) + 1)

print(f"\n  {'Rk':>3} {'Name':<28} {'College':<18} {'Pick':>4} {'Age':>5} {'SLAP':>6} {'DC':>5} "
      f"{'BO':>5} {'Prod':>5} {'ED':>3} {'RAS':>4} {'Δ DC':>6}")
print(f"  {'-'*110}")

for _, r in df_ranked.iterrows():
    bd_flag = "*" if r['birthdate_source'] == 'estimated' else " "
    ed_flag = "Y" if r['early_declare'] == 1 else "N"
    delta_str = f"+{r['delta_vs_dc']:.1f}" if r['delta_vs_dc'] >= 0 else f"{r['delta_vs_dc']:.1f}"
    print(f"  {int(r['rank']):>3} {r['player_name']:<28} {r['college']:<18} {int(r['projected_pick']):>4} "
          f"{r['draft_age']:>4.1f}{bd_flag} {r['slap_score']:>5.1f} {r['dc_score']:>5.1f} "
          f"{r['breakout_score_filled']:>5.1f} {r['production_score_filled']:>5.1f} "
          f"{ed_flag:>3} {r['ras_score']:>4.0f} {delta_str:>6}")

print(f"\n  * = estimated birthdate (verified for top-100 picks)")
print(f"  ED: Y = early declare, N = stayed full eligibility")
print(f"  Δ DC: positive = model likes player MORE than their draft slot")
print(f"        negative = model likes player LESS than their draft slot")

# Summary stats
print(f"\n  SUMMARY:")
print(f"    Prospects: {len(df)}")
print(f"    SLAP range: {df_ranked['slap_score'].min():.1f} — {df_ranked['slap_score'].max():.1f}")
print(f"    SLAP mean: {df_ranked['slap_score'].mean():.1f}")
print(f"    Players model likes more than DC: {(df_ranked['delta_vs_dc'] > 0).sum()}")
print(f"    Players model likes less than DC: {(df_ranked['delta_vs_dc'] < 0).sum()}")
print(f"    Biggest positive delta: {df_ranked.loc[df_ranked['delta_vs_dc'].idxmax(), 'player_name']} "
      f"(+{df_ranked['delta_vs_dc'].max():.1f})")
print(f"    Biggest negative delta: {df_ranked.loc[df_ranked['delta_vs_dc'].idxmin(), 'player_name']} "
      f"({df_ranked['delta_vs_dc'].min():.1f})")

# Top 10 biggest disagreements
print(f"\n  TOP 10 BIGGEST DISAGREEMENTS (|Δ DC| > 0):")
df_ranked['abs_delta'] = df_ranked['delta_vs_dc'].abs()
top_disagree = df_ranked.nlargest(10, 'abs_delta')
for _, r in top_disagree.iterrows():
    direction = "BOOST" if r['delta_vs_dc'] > 0 else "DING"
    delta_str = f"+{r['delta_vs_dc']:.1f}" if r['delta_vs_dc'] >= 0 else f"{r['delta_vs_dc']:.1f}"
    print(f"    {r['player_name']:<28} pick {int(r['projected_pick']):>3} | "
          f"DC {r['dc_score']:.1f} → SLAP {r['slap_score']:.1f} | {direction} {delta_str}")

# Save results
output_cols = [
    'rank', 'player_name', 'position', 'college', 'projected_pick', 'projected_pick_raw',
    'draft_age', 'birthdate', 'birthdate_source', 'early_declare',
    'slap_score', 'dc_score', 'breakout_score_filled', 'production_score_filled',
    'early_declare_score', 'ras_score', 'delta_vs_dc',
    'breakout_age', 'peak_dominator', 'breakout_score',
    'cfbd_receptions', 'cfbd_rec_yards', 'cfbd_team_pass_att', 'cfbd_rush_yards',
    'production_raw', 'production_score',
    'pff_receptions', 'pff_yards', 'pff_player_game_count',
    'pff_grades_offense', 'pff_grades_pass_route', 'pff_yprr', 'pff_team_name',
    'weight', 'height',
]
df_out = df_ranked[[c for c in output_cols if c in df_ranked.columns]]
df_out.to_csv('output/te_slap_2026.csv', index=False)
print(f"\n  Saved: output/te_slap_2026.csv ({len(df_out)} rows)")

df_out.to_csv('data/te_2026_prospects_final.csv', index=False)
print(f"  Saved: data/te_2026_prospects_final.csv ({len(df_out)} rows)")

print(f"\n{'='*120}")
print("DONE")
print(f"{'='*120}")
