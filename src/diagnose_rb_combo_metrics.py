"""
DIAGNOSTIC: RB Composite Production Metric Test
================================================
Tests whether combining multiple receiving metrics beats the single
FINAL RY/TPA metric. Uses multivariate regression controlling for DC.

DOES NOT CHANGE ANYTHING — diagnostic output only.
"""

import pandas as pd
import numpy as np
import warnings, os
from scipy import stats as sp_stats
from datetime import datetime
import statsmodels.api as sm
warnings.filterwarnings('ignore')
os.chdir('/home/user/SlapModelV3')


# ============================================================================
# HELPERS
# ============================================================================
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

def age_weight(season_age):
    if pd.isna(season_age): return 1.0
    sa = float(season_age)
    if sa <= 19: return 1.15
    elif sa <= 20: return 1.10
    elif sa <= 21: return 1.05
    elif sa <= 22: return 1.00
    elif sa <= 23: return 0.95
    else: return 0.90


# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 120)
print("DIAGNOSTIC: RB Composite Production Metric Test")
print("=" * 120)

rb_bt = pd.read_csv('data/rb_backtest_with_receiving.csv')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')
rb_out = outcomes[outcomes['position'] == 'RB'][
    ['player_name', 'draft_year', 'pick', 'first_3yr_ppg', 'career_ppg']
].copy()
rb_bt = rb_bt.merge(rb_out, on=['player_name', 'draft_year', 'pick'], how='left')
rb_bt['s_dc'] = rb_bt['pick'].apply(dc_score)
rb_bt['name_norm'] = rb_bt['player_name'].apply(normalize_name)

# Multi-season college data
college = pd.read_csv('data/college_receiving_2011_2023.csv')
college_rb = college[college['position'] == 'RB'].copy()
college_rb['name_norm'] = college_rb['college_name'].apply(normalize_name)
college_rb['rec_yards'] = pd.to_numeric(college_rb['rec_yards'], errors='coerce')
college_rb['team_pass_att'] = pd.to_numeric(college_rb['team_pass_att'], errors='coerce')
college_rb['college_season'] = pd.to_numeric(college_rb['college_season'], errors='coerce')

# Birthdates
bdays = pd.read_csv('data/nflverse_birthdates_2015_2025.csv')
bdays_rb = bdays[bdays['position'] == 'RB'].copy()
bdays_rb['name_norm'] = bdays_rb['nfl_name'].apply(normalize_name)
bdays_rb['birth_date'] = pd.to_datetime(bdays_rb['birth_date'], errors='coerce')
bday_lookup = dict(zip(bdays_rb['name_norm'], bdays_rb['birth_date']))

rb_bt['rec_yards_num'] = pd.to_numeric(rb_bt['rec_yards'], errors='coerce')
rb_bt['receptions_num'] = pd.to_numeric(rb_bt['receptions'], errors='coerce')
rb_bt['team_pass_att_num'] = pd.to_numeric(rb_bt['team_pass_att'], errors='coerce')
rb_bt['age_num'] = pd.to_numeric(rb_bt['age'], errors='coerce')

# FINAL RY/TPA (current metric)
rb_bt['m_final_ryTPA'] = np.where(
    (rb_bt['rec_yards_num'].notna()) & (rb_bt['team_pass_att_num'].notna()) & (rb_bt['team_pass_att_num'] > 0),
    (rb_bt['rec_yards_num'] / rb_bt['team_pass_att_num']) * rb_bt['age_num'].apply(
        lambda a: age_weight(float(a) - 1 if pd.notna(a) else 22)
    ) * 100,
    np.nan
)

# FINAL Rec/TPA
rb_bt['m_final_recTPA'] = np.where(
    (rb_bt['receptions_num'].notna()) & (rb_bt['team_pass_att_num'].notna()) & (rb_bt['team_pass_att_num'] > 0),
    (rb_bt['receptions_num'] / rb_bt['team_pass_att_num']) * rb_bt['age_num'].apply(
        lambda a: age_weight(float(a) - 1 if pd.notna(a) else 22)
    ) * 100,
    np.nan
)

print(f"\n  RB backtest: {len(rb_bt)} players")


# ============================================================================
# MATCH MULTI-SEASON DATA
# ============================================================================
matched_seasons = []
for _, rb in rb_bt.iterrows():
    name_n = rb['name_norm']
    dy = int(rb['draft_year'])
    player_seasons = college_rb[
        (college_rb['name_norm'] == name_n) &
        (college_rb['college_season'] < dy) &
        (college_rb['team_pass_att'].notna()) &
        (college_rb['team_pass_att'] > 0)
    ].copy()
    if len(player_seasons) == 0:
        cfbd_name = rb.get('cfbd_name', '')
        if pd.notna(cfbd_name) and cfbd_name != '':
            cfbd_norm = normalize_name(cfbd_name)
            player_seasons = college_rb[
                (college_rb['name_norm'] == cfbd_norm) &
                (college_rb['college_season'] < dy) &
                (college_rb['team_pass_att'].notna()) &
                (college_rb['team_pass_att'] > 0)
            ].copy()
    if len(player_seasons) > 0:
        player_seasons = player_seasons.sort_values('college_season')
        player_seasons['player_name'] = rb['player_name']
        player_seasons['draft_year'] = dy
        bday = bday_lookup.get(name_n)
        if pd.notna(bday):
            player_seasons['season_age'] = player_seasons['college_season'].apply(
                lambda s: (datetime(int(s), 9, 1) - bday).days / 365.25)
        else:
            draft_age = rb['age']
            if pd.notna(draft_age):
                player_seasons['season_age'] = player_seasons['college_season'].apply(
                    lambda s: float(draft_age) - (dy - s))
            else:
                player_seasons['season_age'] = np.nan
        matched_seasons.append(player_seasons)

all_seasons = pd.concat(matched_seasons, ignore_index=True) if matched_seasons else pd.DataFrame()
all_seasons['ry_tpa'] = all_seasons['rec_yards'] / all_seasons['team_pass_att']
all_seasons['aw'] = all_seasons['season_age'].apply(age_weight)
all_seasons['ry_tpa_aw'] = all_seasons['ry_tpa'] * all_seasons['aw'] * 100

# CAREER AVG RY/TPA
career = all_seasons.groupby('player_name').agg(
    total_ry=('rec_yards', 'sum'),
    total_tpa=('team_pass_att', 'sum'),
    mean_aw=('aw', 'mean'),
    n_seasons=('college_season', 'count'),
    total_rec_yards_all=('rec_yards', 'sum'),
).reset_index()
career['m_career_ryTPA'] = (career['total_ry'] / career['total_tpa']) * career['mean_aw'] * 100
rb_bt = rb_bt.merge(career[['player_name', 'm_career_ryTPA', 'n_seasons']], on='player_name', how='left')

# BEST season RY/TPA
best = all_seasons.groupby('player_name')['ry_tpa_aw'].max().reset_index()
best.columns = ['player_name', 'm_best_ryTPA']
rb_bt = rb_bt.merge(best, on='player_name', how='left')

# DECLINE FLAG: did production drop from best to final?
rb_bt['decline_flag'] = np.where(
    rb_bt['m_best_ryTPA'].notna() & rb_bt['m_final_ryTPA'].notna(),
    np.where(rb_bt['m_best_ryTPA'] > rb_bt['m_final_ryTPA'] * 1.01, 1, 0),
    np.nan
)

# DECLINE MAGNITUDE: how much did it drop?
rb_bt['decline_pct'] = np.where(
    rb_bt['m_best_ryTPA'].notna() & rb_bt['m_final_ryTPA'].notna() & (rb_bt['m_best_ryTPA'] > 0),
    (rb_bt['m_final_ryTPA'] - rb_bt['m_best_ryTPA']) / rb_bt['m_best_ryTPA'],
    np.nan
)

# SIMPLE AVERAGE of final + career
rb_bt['m_avg_final_career'] = np.where(
    rb_bt['m_final_ryTPA'].notna() & rb_bt['m_career_ryTPA'].notna(),
    (rb_bt['m_final_ryTPA'] + rb_bt['m_career_ryTPA']) / 2,
    rb_bt['m_final_ryTPA']  # fallback to final if no career data
)

# TOTAL CAREER RECEPTIONS (raw volume)
career_rec = all_seasons.groupby('player_name')['rec_yards'].sum().reset_index()
career_rec.columns = ['player_name', 'career_total_rec_yards']
rb_bt = rb_bt.merge(career_rec, on='player_name', how='left')

print(f"  Multi-season matched: {rb_bt['m_career_ryTPA'].notna().sum()}/{len(rb_bt)}")
print(f"  Decline flag: {(rb_bt['decline_flag']==1).sum()} declined, {(rb_bt['decline_flag']==0).sum()} held/improved")


# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================
print(f"\n\n{'='*120}")
print("REGRESSION ANALYSIS: Does combining metrics beat FINAL RY/TPA alone?")
print("=" * 120)

outcome_cols = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
outcome_labels = ['Top-24 PPR', 'Top-12 PPR', 'First 3yr PPG', 'Career PPG']

def run_regression(df, y_col, x_cols, x_labels):
    """Run OLS regression and return results dict."""
    mask = df[y_col].notna()
    for xc in x_cols:
        mask &= df[xc].notna()
    sub = df[mask].copy()
    if len(sub) < 20:
        return None

    X = sm.add_constant(sub[x_cols])
    y = sub[y_col]
    model = sm.OLS(y, X).fit()

    results = {
        'n': len(sub),
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
        'coeffs': {},
    }
    for xc, xl in zip(x_cols, x_labels):
        results['coeffs'][xl] = {
            'coef': model.params[xc],
            'p': model.pvalues[xc],
            'sig': '**' if model.pvalues[xc] < 0.01 else ' *' if model.pvalues[xc] < 0.05 else '  ',
        }
    return results


# ── BASELINE: DC + FINAL RY/TPA ──
print(f"\n  ── BASELINE: DC + FINAL RY/TPA (current model) ──")
print(f"  {'Outcome':<15} {'R²':>6} {'Adj R²':>7} {'n':>4} | {'DC p':>8} {'FINAL p':>8}")
print(f"  {'-'*55}")

baselines = {}
for oc, ol in zip(outcome_cols, outcome_labels):
    res = run_regression(rb_bt, oc, ['s_dc', 'm_final_ryTPA'], ['DC', 'FINAL RY/TPA'])
    if res:
        baselines[oc] = res
        dc_p = res['coeffs']['DC']['p']
        fin_p = res['coeffs']['FINAL RY/TPA']['p']
        print(f"  {ol:<15} {res['r2']:>6.4f} {res['adj_r2']:>7.4f} {res['n']:>4} | "
              f"{dc_p:>8.4f}{res['coeffs']['DC']['sig']} {fin_p:>8.4f}{res['coeffs']['FINAL RY/TPA']['sig']}")


# ── COMBO TESTS ──
combos = [
    {
        'name': '1. FINAL RY/TPA + CAREER AVG RY/TPA',
        'desc': 'Does adding career context help?',
        'x_cols': ['s_dc', 'm_final_ryTPA', 'm_career_ryTPA'],
        'x_labels': ['DC', 'FINAL RY/TPA', 'CAREER AVG'],
        'new_component': 'CAREER AVG',
    },
    {
        'name': '2. FINAL RY/TPA + DECLINE FLAG',
        'desc': 'Does declining production from best→final matter?',
        'x_cols': ['s_dc', 'm_final_ryTPA', 'decline_flag'],
        'x_labels': ['DC', 'FINAL RY/TPA', 'DECLINE FLAG'],
        'new_component': 'DECLINE FLAG',
    },
    {
        'name': '3. FINAL RY/TPA + FINAL Rec/TPA',
        'desc': 'Does adding catch volume (receptions) help?',
        'x_cols': ['s_dc', 'm_final_ryTPA', 'm_final_recTPA'],
        'x_labels': ['DC', 'FINAL RY/TPA', 'FINAL Rec/TPA'],
        'new_component': 'FINAL Rec/TPA',
    },
    {
        'name': '4. FINAL RY/TPA + CAREER TOTAL REC YARDS',
        'desc': 'Does raw career receiving volume help?',
        'x_cols': ['s_dc', 'm_final_ryTPA', 'career_total_rec_yards'],
        'x_labels': ['DC', 'FINAL RY/TPA', 'CAREER REC YDS'],
        'new_component': 'CAREER REC YDS',
    },
]

for combo in combos:
    print(f"\n\n  ── {combo['name']} ──")
    print(f"  Question: {combo['desc']}")
    print(f"  {'Outcome':<15} {'R²':>6} {'ΔR²':>6} | ", end='')
    for xl in combo['x_labels']:
        print(f"{'p('+xl+')':>16}", end=' ')
    print(f"| {'Verdict'}")
    print(f"  {'-'*100}")

    for oc, ol in zip(outcome_cols, outcome_labels):
        res = run_regression(rb_bt, oc, combo['x_cols'], combo['x_labels'])
        if res and oc in baselines:
            delta_r2 = res['r2'] - baselines[oc]['r2']
            # Check if new component is significant
            new_p = res['coeffs'][combo['new_component']]['p']
            new_sig = res['coeffs'][combo['new_component']]['sig']
            final_p = res['coeffs']['FINAL RY/TPA']['p']
            final_sig = res['coeffs']['FINAL RY/TPA']['sig']

            # Verdict
            if new_p < 0.05 and final_p < 0.05:
                verdict = "BOTH significant"
            elif new_p < 0.05 and final_p >= 0.05:
                verdict = "NEW replaces FINAL"
            elif new_p >= 0.05 and final_p < 0.05:
                verdict = "NEW adds nothing"
            else:
                verdict = "Neither significant"

            row = f"  {ol:<15} {res['r2']:>6.4f} {delta_r2:>+6.4f} | "
            for xl in combo['x_labels']:
                p = res['coeffs'][xl]['p']
                sig = res['coeffs'][xl]['sig']
                row += f"{p:>14.4f}{sig} "
            row += f"| {verdict}"
            print(row)
        elif res:
            print(f"  {ol:<15} {res['r2']:>6.4f}  {'n/a':>5} | (no baseline)")


# ── COMBO 5: Simple average ──
print(f"\n\n  ── 5. SIMPLE AVERAGE of FINAL + CAREER (single composite metric) ──")
print(f"  Question: Does averaging final + career beat final alone?")
print(f"  {'Outcome':<15} {'R²(avg)':>8} {'R²(final)':>10} {'ΔR²':>6} | {'p(avg)':>10} | {'Verdict'}")
print(f"  {'-'*70}")

for oc, ol in zip(outcome_cols, outcome_labels):
    res_avg = run_regression(rb_bt, oc, ['s_dc', 'm_avg_final_career'], ['DC', 'AVG(F+C)'])
    if res_avg and oc in baselines:
        delta = res_avg['r2'] - baselines[oc]['r2']
        avg_p = res_avg['coeffs']['AVG(F+C)']['p']
        verdict = "BETTER" if delta > 0.005 else "WORSE" if delta < -0.005 else "~SAME"
        print(f"  {ol:<15} {res_avg['r2']:>8.4f} {baselines[oc]['r2']:>10.4f} {delta:>+6.4f} | "
              f"{avg_p:>10.4f}{res_avg['coeffs']['AVG(F+C)']['sig']} | {verdict}")


# ============================================================================
# MULTICOLLINEARITY CHECK
# ============================================================================
print(f"\n\n{'='*120}")
print("MULTICOLLINEARITY CHECK: How correlated are these metrics with each other?")
print("=" * 120)

corr_cols = ['m_final_ryTPA', 'm_career_ryTPA', 'm_best_ryTPA', 'm_final_recTPA', 'career_total_rec_yards']
corr_labels = ['FINAL RY/TPA', 'CAREER RY/TPA', 'BEST RY/TPA', 'FINAL Rec/TPA', 'CAREER REC YDS']

valid = rb_bt[corr_cols].dropna()
print(f"\n  Pairwise correlations (n={len(valid)}):")
print(f"  {'':>18}", end='')
for cl in corr_labels:
    print(f" {cl[:12]:>12}", end='')
print()
print(f"  {'-'*78}")

for i, (cc, cl) in enumerate(zip(corr_cols, corr_labels)):
    row = f"  {cl:>18}"
    for j, cc2 in enumerate(corr_cols):
        if j <= i:
            mask2 = rb_bt[cc].notna() & rb_bt[cc2].notna()
            if mask2.sum() > 5:
                r = rb_bt.loc[mask2, cc].corr(rb_bt.loc[mask2, cc2])
                row += f" {r:>12.3f}"
            else:
                row += f" {'n/a':>12}"
        else:
            row += f" {'':>12}"
    print(row)


# ============================================================================
# FINAL VERDICT
# ============================================================================
print(f"\n\n{'='*120}")
print("FINAL VERDICT")
print("=" * 120)
print(f"""
  For each combination, the key question is:
  "When BOTH metrics are in the model, is the NEW component significant (p < 0.05)?"
  If not, it doesn't add information beyond what FINAL RY/TPA already captures.

  Also: ΔR² shows how much additional variance the new component explains.
  A ΔR² < 0.005 means less than 0.5% improvement — not worth the complexity.
""")

print(f"  Nothing changed. This is diagnostic only.")
