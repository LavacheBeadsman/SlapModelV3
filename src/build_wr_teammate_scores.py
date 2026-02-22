"""
Task 3: Build teammate score for all WRs 2015-2025.

For each WR in the backtest, find all other drafted pass catchers (WR + TE) from
the same college who were drafted within a 2-year window (1 year before through
1 year after the WR's draft year).

Output: data/wr_teammate_scores.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

os.chdir("/home/user/SlapModelV3")


def dc_score(pick):
    """Calculate draft capital score: DC = 100 - 2.40 * (pick^0.62 - 1)"""
    return 100 - 2.40 * (pick ** 0.62 - 1)


def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 15:
        return np.nan, np.nan, len(x)
    # Residualize x on z
    coef_xz = np.polyfit(z, x, 1)
    resid_x = x - np.polyval(coef_xz, z)
    # Residualize y on z
    coef_yz = np.polyfit(z, y, 1)
    resid_y = y - np.polyval(coef_yz, z)
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(x)


# ===========================================================================
# Step 1: Load data
# ===========================================================================
print("=" * 80)
print("BUILDING TEAMMATE SCORES FOR ALL WRs 2015-2025")
print("=" * 80)

wr_backtest = pd.read_csv('data/wr_backtest_expanded_final.csv')
draft_picks = pd.read_parquet('data/nflverse/draft_picks.parquet')
outcomes = pd.read_csv('data/backtest_outcomes_complete.csv')

print(f"WR backtest: {len(wr_backtest)} players")
print(f"Draft picks: {len(draft_picks)} total, years {draft_picks['season'].min()}-{draft_picks['season'].max()}")
print(f"Outcomes: {len(outcomes)} players")

# Filter draft picks to pass catchers (WR + TE) in relevant years
# Need 2014-2026 window (1 year before earliest 2015 through 1 year after latest 2025)
pass_catchers = draft_picks[
    (draft_picks['position'].isin(['WR', 'TE'])) &
    (draft_picks['season'].between(2014, 2026))
].copy()
pass_catchers['dc_score'] = pass_catchers['pick'].apply(dc_score)
print(f"Pass catchers (WR+TE, 2014-2026): {len(pass_catchers)}")

# Normalize college names for matching
# The backtest uses one format, draft_picks uses another
# Let's check what format draft_picks uses
print(f"\nSample draft_picks colleges: {pass_catchers['college'].dropna().unique()[:20].tolist()}")

# Build a normalized college name for matching
def normalize_college(name):
    """Normalize college name to a canonical form."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    # Common variations
    replacements = {
        'Ohio St.': 'Ohio State',
        'Michigan St.': 'Michigan State',
        'Penn St.': 'Penn State',
        'Arizona St.': 'Arizona State',
        'Oklahoma St.': 'Oklahoma State',
        'Oregon St.': 'Oregon State',
        'Washington St.': 'Washington State',
        'Florida St.': 'Florida State',
        'Boise St.': 'Boise State',
        'Colorado St.': 'Colorado State',
        'Iowa St.': 'Iowa State',
        'Kansas St.': 'Kansas State',
        'Fresno St.': 'Fresno State',
        'Georgia St.': 'Georgia State',
        'Grambling St.': 'Grambling',
        'Middle Tenn. St.': 'Middle Tennessee',
        'North Carolina St.': 'NC State',
        'New Mexico St.': 'New Mexico State',
        'SE Missouri St.': 'Southeast Missouri State',
        'Central Florida': 'UCF',
        'Miami (FL)': 'Miami',
        'Miami (OH)': 'Miami (OH)',
        'Mississippi': 'Ole Miss',
        'Ala-Birmingham': 'UAB',
        'La-Monroe': 'Louisiana Monroe',
        'South Florida': 'South Florida',
        'Connecticut': 'Connecticut',
        'Massachusetts': 'Massachusetts',
        'Boston Col.': 'Boston College',
        'Southern Miss': 'Southern Mississippi',
        'North Carolina': 'North Carolina',
        'Northern Illinois': 'Northern Illinois',
        'Western Michigan': 'Western Michigan',
        'Western Kentucky': 'Western Kentucky',
        'Eastern Washington': 'Eastern Washington',
        'North Dakota St.': 'North Dakota State',
        'Central Arkansas': 'Central Arkansas',
        'Northern Iowa': 'Northern Iowa',
        'Virginia Tech': 'Virginia Tech',
        'Georgia Tech': 'Georgia Tech',
        'Louisiana Tech': 'Louisiana Tech',
        'Texas A&M': 'Texas A&M',
        'Texas Tech': 'Texas Tech',
        'Wake Forest': 'Wake Forest',
        'West Virginia': 'West Virginia',
        'William & Mary': 'William & Mary',
        'East Central (OK)': 'East Central',
        'West Alabama': 'West Alabama',
        'Lenoir-Rhyne': 'Lenoir-Rhyne',
        'Charleston (WV)': 'Charleston',
        'UT Martin': 'UT Martin',
        'SE Missouri St.': 'Southeast Missouri',
    }
    for old, new in replacements.items():
        if name == old:
            return new.lower()
    return name.lower()


# ===========================================================================
# Step 2: Build teammate scores
# ===========================================================================
print(f"\n{'='*60}")
print("CALCULATING TEAMMATE SCORES")
print(f"{'='*60}")

# Create normalized college for both datasets
wr_backtest['college_norm'] = wr_backtest['college'].apply(normalize_college)
pass_catchers['college_norm'] = pass_catchers['college'].apply(normalize_college)

teammate_results = []

for _, wr in wr_backtest.iterrows():
    wr_name = wr['player_name']
    wr_college = wr['college_norm']
    wr_draft_year = wr['draft_year']
    wr_pick = wr['pick']

    # Find pass catchers from same school, within 2-year window
    # Window: draft_year - 1 through draft_year + 1
    teammates = pass_catchers[
        (pass_catchers['college_norm'] == wr_college) &
        (pass_catchers['season'].between(wr_draft_year - 1, wr_draft_year + 1)) &
        # Exclude the player themselves
        (pass_catchers['pick'] != wr_pick) |
        (pass_catchers['season'] != wr_draft_year)
    ].copy()

    # Need to actually exclude the player themselves more carefully
    # A player matches if same college, same year, same pick
    teammates = pass_catchers[
        (pass_catchers['college_norm'] == wr_college) &
        (pass_catchers['season'].between(wr_draft_year - 1, wr_draft_year + 1))
    ].copy()

    # Remove the player themselves (match on name normalization + pick + year)
    wr_name_lower = wr_name.lower().strip()
    teammates = teammates[
        ~((teammates['pfr_player_name'].str.lower().str.strip() == wr_name_lower) &
          (teammates['season'] == wr_draft_year))
    ]

    if len(teammates) == 0:
        teammate_results.append({
            'player_name': wr_name,
            'draft_year': wr_draft_year,
            'pick': wr_pick,
            'round': wr['round'],
            'college': wr['college'],
            'teammate_count': 0,
            'best_teammate_pick': np.nan,
            'total_teammate_dc': 0.0,
            'avg_teammate_dc': 0.0,
            'teammate_names': '',
        })
    else:
        tm_names = []
        for _, tm in teammates.iterrows():
            tm_names.append(f"{tm['pfr_player_name']} (Rd{tm['round']} #{tm['pick']}, {tm['season']})")

        teammate_results.append({
            'player_name': wr_name,
            'draft_year': wr_draft_year,
            'pick': wr_pick,
            'round': wr['round'],
            'college': wr['college'],
            'teammate_count': len(teammates),
            'best_teammate_pick': int(teammates['pick'].min()),
            'total_teammate_dc': round(teammates['dc_score'].sum(), 1),
            'avg_teammate_dc': round(teammates['dc_score'].mean(), 1),
            'teammate_names': '; '.join(tm_names),
        })

teammate_df = pd.DataFrame(teammate_results)
teammate_df = teammate_df.sort_values(['draft_year', 'pick']).reset_index(drop=True)

# Save
teammate_df.to_csv('data/wr_teammate_scores.csv', index=False)
print(f"Saved: data/wr_teammate_scores.csv")
print(f"Total WRs: {len(teammate_df)}")
print(f"WRs with at least 1 teammate: {(teammate_df['teammate_count'] > 0).sum()}")
print(f"WRs with zero teammates: {(teammate_df['teammate_count'] == 0).sum()}")

# ===========================================================================
# Step 3: Top 20 highest total_teammate_dc
# ===========================================================================
print(f"\n{'='*60}")
print("TOP 20 HIGHEST TOTAL_TEAMMATE_DC")
print(f"{'='*60}")

# Merge with outcomes for hit24
outcomes_wr = outcomes[outcomes['position'] == 'WR'].copy()

teammate_with_outcomes = teammate_df.merge(
    outcomes_wr[['player_name', 'draft_year', 'hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']],
    on=['player_name', 'draft_year'],
    how='left'
)

top20 = teammate_with_outcomes.nlargest(20, 'total_teammate_dc')
for i, (_, row) in enumerate(top20.iterrows(), 1):
    hit_str = f"hit24={'YES' if row['hit24']==1 else 'NO' if row['hit24']==0 else '?'}"
    print(f"\n  {i}. {row['player_name']} ({row['college']}, {row['draft_year']}) — "
          f"Rd{row['round']} Pick {row['pick']} | {hit_str}")
    print(f"     Teammate DC: {row['total_teammate_dc']} ({row['teammate_count']} teammates, "
          f"avg DC: {row['avg_teammate_dc']}, best pick: {row['best_teammate_pick']})")
    print(f"     Teammates: {row['teammate_names']}")

# ===========================================================================
# Step 4: Bottom 20 (zero or near-zero)
# ===========================================================================
print(f"\n{'='*60}")
print("BOTTOM 20 (ZERO OR LOWEST TEAMMATE DC)")
print(f"{'='*60}")

# Get players with outcomes who have lowest teammate DC
with_outcomes = teammate_with_outcomes[teammate_with_outcomes['hit24'].notna()].copy()
bottom20 = with_outcomes.nsmallest(20, 'total_teammate_dc')
for i, (_, row) in enumerate(bottom20.iterrows(), 1):
    hit_str = f"hit24={'YES' if row['hit24']==1 else 'NO'}"
    print(f"  {i}. {row['player_name']} ({row['college']}, {row['draft_year']}) — "
          f"Rd{row['round']} Pick {row['pick']} | {hit_str}")
    print(f"     Teammate DC: {row['total_teammate_dc']} ({row['teammate_count']} teammates)")

# ===========================================================================
# Step 5: Correlations
# ===========================================================================
print(f"\n{'='*60}")
print("CORRELATIONS: TEAMMATE DC vs OUTCOMES")
print(f"{'='*60}")

# Only use WRs with outcome data
analysis_df = teammate_with_outcomes.dropna(subset=['hit24']).copy()
analysis_df['dc_score'] = analysis_df['pick'].apply(dc_score)

print(f"\nWRs with outcomes: {len(analysis_df)}")

outcomes_list = ['hit24', 'hit12', 'first_3yr_ppg', 'career_ppg']
teammate_metrics = ['total_teammate_dc', 'teammate_count', 'avg_teammate_dc', 'best_teammate_pick']

print(f"\n--- RAW CORRELATIONS ---")
print(f"{'Metric':<25} {'hit24':>12} {'hit12':>12} {'first_3yr_ppg':>15} {'career_ppg':>12}")
for tm_metric in teammate_metrics:
    vals = []
    for outcome in outcomes_list:
        valid = analysis_df[[tm_metric, outcome]].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid[tm_metric], valid[outcome])
            sig = '*' if p < 0.05 else ''
            vals.append(f"r={r:+.3f}{sig}")
        else:
            vals.append("n/a")
    print(f"  {tm_metric:<23} {vals[0]:>12} {vals[1]:>12} {vals[2]:>15} {vals[3]:>12}")

print(f"\n--- PARTIAL CORRELATIONS (controlling for DC score) ---")
print(f"{'Metric':<25} {'hit24':>12} {'hit12':>12} {'first_3yr_ppg':>15} {'career_ppg':>12}")
for tm_metric in teammate_metrics:
    vals = []
    for outcome in outcomes_list:
        valid = analysis_df[[tm_metric, outcome, 'dc_score']].dropna()
        if len(valid) > 15:
            r, p, n = partial_corr(
                valid[tm_metric].values,
                valid[outcome].values,
                valid['dc_score'].values
            )
            sig = '*' if p < 0.05 else ''
            vals.append(f"r={r:+.3f}{sig}")
        else:
            vals.append("n/a")
    print(f"  {tm_metric:<23} {vals[0]:>12} {vals[1]:>12} {vals[2]:>15} {vals[3]:>12}")

# ===========================================================================
# Step 6: Hit rates by teammate score groups
# ===========================================================================
print(f"\n{'='*60}")
print("HIT RATES BY TEAMMATE DC GROUP")
print(f"{'='*60}")

# Group by total_teammate_dc
analysis_df['tm_group'] = pd.cut(
    analysis_df['total_teammate_dc'],
    bins=[-1, 0, 50, 100, 200, 999],
    labels=['No teammates', 'Low (1-50)', 'Medium (51-100)', 'High (101-200)', 'Elite (200+)']
)

print(f"\n{'Group':<20} {'N':>5} {'Hit24 Rate':>12} {'Hit12 Rate':>12} {'Avg 3yr PPG':>14}")
for group in ['No teammates', 'Low (1-50)', 'Medium (51-100)', 'High (101-200)', 'Elite (200+)']:
    g = analysis_df[analysis_df['tm_group'] == group]
    if len(g) > 0:
        h24 = g['hit24'].mean()
        h12 = g['hit12'].mean()
        ppg = g['first_3yr_ppg'].mean()
        print(f"  {group:<18} {len(g):>5} {h24:>11.1%} {h12:>11.1%} {ppg:>13.1f}")

print(f"\n{'='*80}")
print("DONE")
print(f"{'='*80}")
