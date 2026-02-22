"""
Identify High-Priority RBs Missing Breakout Ages
Focus on Rounds 1-3 where draft capital is highest
"""
import pandas as pd

# Load RB backtest
rb = pd.read_csv('data/rb_backtest_2015_2024.csv')

# RB breakout ages from generate_complete_slap.py (updated with research)
RB_BREAKOUT_AGES = {
    # 2015 class
    'Todd Gurley': 19, 'Melvin Gordon': 20, 'David Johnson': 21,
    'Tevin Coleman': 20, 'Duke Johnson': 18, 'T.J. Yeldon': 18,
    'Ameer Abdullah': 20, 'Jay Ajayi': 20, 'Mike Davis': 19, 'Matt Jones': 21,
    # 2016 class
    'Ezekiel Elliott': 19, 'Derrick Henry': 21, 'Kenyan Drake': 19,
    'Jordan Howard': 20, 'Devontae Booker': 21, 'Paul Perkins': 18,
    'C.J. Prosise': 21,
    # 2017 class
    'Christian McCaffrey': 19, 'Leonard Fournette': 18, 'Dalvin Cook': 18,
    'Joe Mixon': 19, 'Alvin Kamara': 21, 'Kareem Hunt': 19,
    'Aaron Jones': 19, 'James Conner': 20, 'Samaje Perine': 18, 'Marlon Mack': 19,
    "D'Onta Foreman": 20,
    # 2018 class
    'Saquon Barkley': 18, 'Nick Chubb': 18, 'Sony Michel': 18,
    'Ronald Jones': 18, 'Ronald Jones II': 18, 'Rashaad Penny': 21, 'Kerryon Johnson': 20,
    'Royce Freeman': 18, 'Nyheim Hines': 20, 'Derrius Guice': 19,
    # 2019 class
    'Josh Jacobs': 20, 'Miles Sanders': 21, 'David Montgomery': 19,
    'Darrell Henderson': 20, 'Devin Singletary': 19, 'Damien Harris': 18,
    'Justice Hill': 18, 'Tony Pollard': 20, 'Alexander Mattison': 19,
    # 2020 class
    'Clyde Edwards-Helaire': 21, "D'Andre Swift": 19, 'Jonathan Taylor': 18,
    'Cam Akers': 18, 'J.K. Dobbins': 18, 'Zack Moss': 19,
    'Antonio Gibson': 22, 'AJ Dillon': 18, "Ke'Shawn Vaughn": 20, 'Darrynton Evans': 20,
    # 2021 class
    'Najee Harris': 21, 'Travis Etienne': 18, 'Javonte Williams': 21,
    'Michael Carter': 21, 'Trey Sermon': 21, 'Rhamondre Stevenson': 21,
    'Kenneth Gainwell': 19, 'Chuba Hubbard': 19, 'Elijah Mitchell': 20, 'Khalil Herbert': 20,
    # 2022 class
    'Breece Hall': 19, 'Kenneth Walker III': 21, 'James Cook': 21,
    'Rachaad White': 22, 'Brian Robinson Jr.': 22, 'Dameon Pierce': 21,
    'Isiah Pacheco': 19, 'Kyren Williams': 19, 'Tyler Allgeier': 21,
    'Jerome Ford': 21, 'Zamir White': 20, 'Tyrion Davis-Price': 21,
    # 2023 class
    'Bijan Robinson': 19, 'Jahmyr Gibbs': 19, 'Zach Charbonnet': 18,
    "De'Von Achane": 20, 'Tank Bigsby': 18, 'Tyjae Spears': 21,
    'Chase Brown': 22, 'Kendre Miller': 21, 'Roschon Johnson': 18,
    'Israel Abanikanda': 21, 'Eric Gray': 18, 'Sean Tucker': 18,
    'Deuce Vaughn': 18, 'Chris Rodriguez': 19, 'DeWayne McBride': 20,
    # 2024 class
    'Jonathan Brooks': 20, 'Jonathon Brooks': 20, 'Trey Benson': 21, 'Blake Corum': 21,
    'MarShawn Lloyd': 21, 'Jaylen Wright': 20,
    'Ray Davis': 23, 'Braelon Allen': 19, 'Audric Estime': 20,
    'Tyrone Tracy Jr.': 22, 'Isaac Guerendo': 22, 'Kimani Vidal': 21,
}

# Add breakout age and check which are missing
rb['breakout_age'] = rb['player_name'].map(RB_BREAKOUT_AGES)
rb['has_breakout'] = rb['breakout_age'].notna()

print("=" * 90)
print("RB BREAKOUT AGE COVERAGE ANALYSIS")
print("=" * 90)

# Overall stats
total = len(rb)
has_breakout = rb['has_breakout'].sum()
missing = total - has_breakout
print(f"\nTotal RBs in backtest: {total}")
print(f"With breakout age: {has_breakout} ({has_breakout/total*100:.1f}%)")
print(f"Missing breakout age: {missing} ({missing/total*100:.1f}%)")

# By round
print("\n" + "-" * 60)
print("BY ROUND:")
for rnd in sorted(rb['round'].unique()):
    rnd_df = rb[rb['round'] == rnd]
    rnd_has = rnd_df['has_breakout'].sum()
    rnd_miss = len(rnd_df) - rnd_has
    print(f"  Round {rnd}: {rnd_has}/{len(rnd_df)} have breakout ({rnd_miss} missing)")

# Priority list: Rounds 1-3 missing breakout
print("\n" + "=" * 90)
print("HIGH PRIORITY: ROUNDS 1-3 RBs MISSING BREAKOUT AGES")
print("(These have highest DC and affect model most)")
print("=" * 90)

priority = rb[(rb['round'] <= 3) & (~rb['has_breakout'])]
priority = priority.sort_values(['round', 'pick'])

print(f"\n{len(priority)} RBs in Rounds 1-3 need breakout age research:\n")
print(f"{'Player':<25} {'Year':>5} {'Pick':>5} {'Round':>6} {'College':<25} {'NFL PPG':>8} {'Hit24':>6}")
print("-" * 90)

for _, row in priority.iterrows():
    ppg = f"{row['best_ppg']:.1f}" if pd.notna(row['best_ppg']) and row['best_ppg'] > 0 else "-"
    hit = "✓" if row['hit24'] == 1 else ("✗" if row['hit24'] == 0 else "-")
    print(f"{row['player_name']:<25} {int(row['draft_year']):>5} {int(row['pick']):>5} {int(row['round']):>6} {row['college']:<25} {ppg:>8} {hit:>6}")

# By draft year
print("\n" + "=" * 90)
print("MISSING RBs BY DRAFT YEAR (Rounds 1-3 only)")
print("=" * 90)

for year in sorted(priority['draft_year'].unique()):
    year_df = priority[priority['draft_year'] == year]
    print(f"\n{year} ({len(year_df)} missing):")
    for _, row in year_df.iterrows():
        print(f"  - {row['player_name']} (Pick {int(row['pick'])}, {row['college']})")

# Show NFL outcomes for missing players
print("\n" + "=" * 90)
print("NFL OUTCOMES FOR PRIORITY MISSING RBs")
print("=" * 90)

hits = priority[priority['hit24'] == 1]
misses = priority[priority['hit24'] == 0]

print(f"\nHITS (hit24=1): {len(hits)} players")
for _, row in hits.sort_values('best_ppg', ascending=False).iterrows():
    print(f"  {row['player_name']} (Pick {int(row['pick'])}) - {row['best_ppg']:.1f} PPG")

print(f"\nMISSES (hit24=0): {len(misses)} players")
for _, row in misses.head(15).iterrows():
    print(f"  {row['player_name']} (Pick {int(row['pick'])})")
if len(misses) > 15:
    print(f"  ... and {len(misses) - 15} more")

# Save priority list
priority_export = priority[['player_name', 'draft_year', 'pick', 'round', 'college', 'best_ppg', 'hit24']].copy()
priority_export.to_csv('output/rb_missing_breakout_priority.csv', index=False)
print(f"\nSaved priority list: output/rb_missing_breakout_priority.csv")
