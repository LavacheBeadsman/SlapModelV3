"""
Merge AllTERAS.csv (TE Relative Athletic Score) into te_backtest_master.csv.

RAS data: 933 TEs from 2014-2025, scored 0-10 scale.
Backtest: 160 TEs from 2015-2025.

Handles name mismatches with manual mappings for known issues
(Jr., special characters, spelling differences).
"""

import pandas as pd
import numpy as np

# Load both files
bt = pd.read_csv('data/te_backtest_master.csv')
ras = pd.read_csv('data/AllTERAS.csv')

print(f"Backtest TEs: {len(bt)}")
print(f"RAS file TEs: {len(ras)}")
print(f"RAS NaN in source: {ras['RAS'].isna().sum()}")

# Clean up RAS data - drop the HTML Link column, keep useful columns
ras_clean = ras[['Name', 'Year', 'College', 'RAS']].copy()

# Manual name mappings for the 11 unmatched TEs
# backtest name -> RAS file name
name_fixes = {}

# Find what names exist in RAS for these problem cases
problem_names = [
    'Irv Smith', 'Harold Fannin', 'Josiah Deguara', "Tre' McKitty",
    'Seth Devalve', 'MyCole Pruitt', 'Malcolm Johnson',
    'Connor Heyward', 'John FitzPatrick', 'Thomas Fidone', 'Alize Mack'
]

print("\n=== Searching for unmatched TEs in RAS file ===")
for pname in problem_names:
    # Try partial match
    base = pname.split("'")[-1].split(' ')[-1]  # last name
    matches = ras_clean[ras_clean['Name'].str.contains(base, case=False, na=False)]
    if len(matches) > 0:
        for _, m in matches.iterrows():
            print(f"  Backtest: '{pname}' -> RAS candidate: '{m['Name']}' ({m['Year']}, {m['College']}, RAS={m['RAS']})")

# Build manual mapping based on search results
# We'll map backtest player_name -> RAS Name
# First, let's do exact name+year merge
bt_merged = bt.merge(
    ras_clean[['Name', 'Year', 'RAS']].rename(columns={'Name': 'player_name', 'Year': 'draft_year', 'RAS': 'te_ras'}),
    on=['player_name', 'draft_year'],
    how='left'
)

matched_exact = bt_merged['te_ras'].notna().sum()
print(f"\n=== EXACT match (name+year): {matched_exact}/{len(bt)} ===")

# Now handle the unmatched ones with fuzzy logic
unmatched_mask = bt_merged['te_ras'].isna()
unmatched = bt_merged[unmatched_mask][['player_name', 'draft_year', 'round', 'college']].copy()
print(f"\nUnmatched: {len(unmatched)}")

# Try matching unmatched by normalizing names
def normalize_name(name):
    """Normalize name for fuzzy matching."""
    n = str(name).lower().strip()
    # Remove Jr., Sr., III, II, IV suffixes
    for suffix in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
        n = n.replace(suffix, '')
    # Remove special chars
    n = n.replace("'", "").replace("'", "").replace("-", "").replace(".", "")
    return n.strip()

# Create normalized lookup from RAS
ras_clean['name_norm'] = ras_clean['Name'].apply(normalize_name)
bt_merged['name_norm_temp'] = bt_merged['player_name'].apply(normalize_name)

# For each unmatched, try normalized match
fixed_count = 0
for idx in bt_merged[unmatched_mask].index:
    row = bt_merged.loc[idx]
    norm = row['name_norm_temp']
    year = row['draft_year']

    candidates = ras_clean[(ras_clean['name_norm'] == norm) & (ras_clean['Year'] == year)]
    if len(candidates) == 1:
        bt_merged.loc[idx, 'te_ras'] = candidates.iloc[0]['RAS']
        fixed_count += 1
        print(f"  FIXED: '{row['player_name']}' -> '{candidates.iloc[0]['Name']}' (RAS={candidates.iloc[0]['RAS']})")
    elif len(candidates) > 1:
        print(f"  MULTIPLE: '{row['player_name']}' has {len(candidates)} candidates")
    else:
        # Try just last name + year
        last = norm.split()[-1] if norm.split() else norm
        candidates2 = ras_clean[(ras_clean['name_norm'].str.endswith(last)) & (ras_clean['Year'] == year)]
        if len(candidates2) == 1:
            bt_merged.loc[idx, 'te_ras'] = candidates2.iloc[0]['RAS']
            fixed_count += 1
            print(f"  FIXED (last name): '{row['player_name']}' -> '{candidates2.iloc[0]['Name']}' (RAS={candidates2.iloc[0]['RAS']})")
        else:
            print(f"  STILL UNMATCHED: '{row['player_name']}' ({year}, Rd {row['round']})")

print(f"\nFixed via normalization: {fixed_count}")

# Drop temp column
bt_merged.drop(columns=['name_norm_temp'], inplace=True)

# Final stats
total_matched = bt_merged['te_ras'].notna().sum()
total_with_ras_value = bt_merged['te_ras'].notna().sum()  # some matched but RAS=NaN in source

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"Total backtest TEs: {len(bt_merged)}")
print(f"TEs with RAS data: {total_with_ras_value}/{len(bt_merged)} ({100*total_with_ras_value/len(bt_merged):.1f}%)")
print(f"TEs without RAS: {len(bt_merged) - total_with_ras_value}")

# Show RAS distribution for matched TEs
ras_vals = bt_merged['te_ras'].dropna()
print(f"\nRAS stats (0-10 scale) for matched backtest TEs:")
print(f"  Mean: {ras_vals.mean():.2f}")
print(f"  Median: {ras_vals.median():.2f}")
print(f"  Min: {ras_vals.min():.2f}, Max: {ras_vals.max():.2f}")
print(f"  Std: {ras_vals.std():.2f}")

# Show by round
print(f"\nRAS coverage by round:")
for rd in sorted(bt_merged['round'].unique()):
    rd_data = bt_merged[bt_merged['round'] == rd]
    has_ras = rd_data['te_ras'].notna().sum()
    avg_ras = rd_data['te_ras'].mean()
    print(f"  Rd {rd}: {has_ras}/{len(rd_data)} have RAS" + (f", avg={avg_ras:.2f}" if has_ras > 0 else ""))

# Show 10 well-known TEs with their RAS
print(f"\n=== SAMPLE KNOWN TEs ===")
known = ['Kyle Pitts', 'Mark Andrews', 'George Kittle', 'T.J. Hockenson',
         'Dallas Goedert', 'Pat Freiermuth', 'Sam LaPorta', 'Brock Bowers',
         'Evan Engram', 'Cole Kmet', 'Trey McBride', 'Dalton Kincaid',
         'Noah Fant', 'David Njoku']
for name in known:
    row = bt_merged[bt_merged['player_name'] == name]
    if len(row) > 0:
        r = row.iloc[0]
        ras_str = f"{r['te_ras']:.2f}" if pd.notna(r['te_ras']) else "NaN (skipped workouts)"
        hit_str = "HIT24" if r.get('hit24', 0) == 1 else "miss"
        print(f"  {name:25s} | Rd {int(r['round'])} Pick {int(r['pick']):3d} | RAS: {ras_str:8s} | {hit_str}")

# Still unmatched
still_unmatched = bt_merged[bt_merged['te_ras'].isna()][['player_name', 'draft_year', 'round', 'pick', 'college']]
print(f"\n=== STILL NO RAS DATA ({len(still_unmatched)} TEs) ===")
for _, r in still_unmatched.iterrows():
    print(f"  {r['player_name']:25s} | {int(r['draft_year'])} Rd {int(r['round'])} Pick {int(r['pick']):3d} | {r['college']}")

# Save updated file
bt_merged.to_csv('data/te_backtest_master.csv', index=False)
print(f"\nSaved updated te_backtest_master.csv with te_ras column ({len(bt_merged)} rows, {len(bt_merged.columns)} columns)")
