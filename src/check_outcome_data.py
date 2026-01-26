"""
Check WR and RB outcome data availability
"""

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Load the database
df = pd.read_csv('output/slap_complete_database_v4.csv')

print("=" * 80)
print("OUTCOME DATA VERIFICATION")
print("=" * 80)

# Filter to backtest (2015-2023)
backtest = df[(df['draft_year'] >= 2015) & (df['draft_year'] <= 2023)].copy()

print(f"\nAll columns in dataset:")
print(df.columns.tolist())

# Check which outcome columns exist
outcome_cols = ['nfl_best_ppr', 'nfl_best_ppg', 'nfl_hit24', 'nfl_hit12', 'best_season_ppg', 'career_ppr_ppg']
print(f"\nOutcome columns check:")
for col in outcome_cols:
    if col in df.columns:
        print(f"   {col}: EXISTS")
    else:
        print(f"   {col}: NOT FOUND")

# Use whatever columns exist
ppg_col = None
for col in ['nfl_best_ppg', 'best_season_ppg', 'career_ppr_ppg', 'nfl_best_ppr']:
    if col in df.columns:
        ppg_col = col
        break

hit24_col = 'nfl_hit24' if 'nfl_hit24' in df.columns else None
hit12_col = 'nfl_hit12' if 'nfl_hit12' in df.columns else None

print(f"\nUsing: PPG={ppg_col}, Hit24={hit24_col}, Hit12={hit12_col}")

# ============================================================================
# WR OUTCOME DATA
# ============================================================================
print("\n" + "=" * 80)
print("WR OUTCOME DATA (2015-2023)")
print("=" * 80)

wr_backtest = backtest[backtest['position'] == 'WR']
print(f"\nTotal WRs: {len(wr_backtest)}")

if ppg_col:
    wr_ppg_count = wr_backtest[ppg_col].notna().sum()
    print(f"WRs with {ppg_col}: {wr_ppg_count} ({wr_ppg_count/len(wr_backtest)*100:.1f}%)")

if hit24_col:
    wr_hit24_count = wr_backtest[hit24_col].notna().sum()
    print(f"WRs with {hit24_col}: {wr_hit24_count} ({wr_hit24_count/len(wr_backtest)*100:.1f}%)")

if hit12_col:
    wr_hit12_count = wr_backtest[hit12_col].notna().sum()
    print(f"WRs with {hit12_col}: {wr_hit12_count} ({wr_hit12_count/len(wr_backtest)*100:.1f}%)")

# ============================================================================
# RB OUTCOME DATA
# ============================================================================
print("\n" + "=" * 80)
print("RB OUTCOME DATA (2015-2023)")
print("=" * 80)

rb_backtest = backtest[backtest['position'] == 'RB']
print(f"\nTotal RBs: {len(rb_backtest)}")

if ppg_col:
    rb_ppg_count = rb_backtest[ppg_col].notna().sum()
    print(f"RBs with {ppg_col}: {rb_ppg_count} ({rb_ppg_count/len(rb_backtest)*100:.1f}%)")

if hit24_col:
    rb_hit24_count = rb_backtest[hit24_col].notna().sum()
    print(f"RBs with {hit24_col}: {rb_hit24_count} ({rb_hit24_count/len(rb_backtest)*100:.1f}%)")

if hit12_col:
    rb_hit12_count = rb_backtest[hit12_col].notna().sum()
    print(f"RBs with {hit12_col}: {rb_hit12_count} ({rb_hit12_count/len(rb_backtest)*100:.1f}%)")

# ============================================================================
# SAMPLE WRs 2020-2022
# ============================================================================
print("\n" + "=" * 80)
print("SAMPLE WRs (2020-2022)")
print("=" * 80)

wr_sample = wr_backtest[(wr_backtest['draft_year'] >= 2020) & (wr_backtest['draft_year'] <= 2022)]
cols_to_show = ['player_name', 'draft_year', 'pick', 'slap_score']
if ppg_col:
    cols_to_show.append(ppg_col)
if hit24_col:
    cols_to_show.append(hit24_col)
if hit12_col:
    cols_to_show.append(hit12_col)

print(f"\n10 Sample WRs:")
print(wr_sample[cols_to_show].head(10).to_string(index=False))

# ============================================================================
# SAMPLE RBs 2020-2022
# ============================================================================
print("\n" + "=" * 80)
print("SAMPLE RBs (2020-2022)")
print("=" * 80)

rb_sample = rb_backtest[(rb_backtest['draft_year'] >= 2020) & (rb_backtest['draft_year'] <= 2022)]
print(f"\n10 Sample RBs:")
print(rb_sample[cols_to_show].head(10).to_string(index=False))

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if ppg_col:
    wr_has_ppg = wr_backtest[ppg_col].notna().sum()
    rb_has_ppg = rb_backtest[ppg_col].notna().sum()

    if wr_has_ppg > 0 and rb_has_ppg > 0:
        print(f"\n✅ BOTH WRs and RBs have outcome data!")
        print(f"   WRs with PPG: {wr_has_ppg}/{len(wr_backtest)} ({wr_has_ppg/len(wr_backtest)*100:.1f}%)")
        print(f"   RBs with PPG: {rb_has_ppg}/{len(rb_backtest)} ({rb_has_ppg/len(rb_backtest)*100:.1f}%)")
        print(f"\n❌ I MADE AN ERROR in the evaluation - WR outcomes ARE available!")
    elif wr_has_ppg == 0:
        print(f"\n⚠️ WRs are missing PPG data")
    elif rb_has_ppg == 0:
        print(f"\n⚠️ RBs are missing PPG data")
