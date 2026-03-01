# CLAUDE.md - SLAP Score V5 Project Guide

## About the User

- **No coding background** - Explain everything in plain English, avoid jargon
- Fantasy football content creator (YouTube and Patreon)
- Wants to understand what the code does, not just have it work
- **Honesty is critical** - Never claim something is possible if it isn't

## How to Work Together

1. **Check in before major decisions** - Don't assume, ask
2. **Explain code in simple terms** - What does it do and why
3. **Small steps** - Break work into pieces I can follow
4. **Options with tradeoffs** - When deciding something, give me choices with clear pros and cons
5. **NEVER estimate, guess, or make up data** - If data is missing, flag it and ask me how to handle it

## Project Overview

**SLAP Score V5**: Statistical Likelihood of Achieving Production

A draft-capital-anchored prospect model that rates NFL Draft RBs, WRs, and TEs on a 0-100 scale. The model backtests against 722 drafted players (2015-2025) and scores 217 2026 prospects.

## Build Pipeline

### One Command Builds Everything
```bash
python src/build_master_database_v5.py
```
This is the **single source of truth** for all output files. It reads data CSVs, calculates scores, and writes output CSVs.

### Input Files (data/)

| File | Position | Contents | Key Columns |
|------|----------|----------|-------------|
| `data/wr_backtest_all_components.csv` | WR | 339 backtest WRs (2015-2025) | `breakout_age`, `peak_dominator`, `early_declare`, `rush_yards` |
| `data/wr_teammate_scores.csv` | WR | Teammate DC scores | `total_teammate_dc` |
| `data/wr_breakout_ages_2026.csv` | WR | 2026 WR breakout data from CFBD | `breakout_age`, `peak_dominator`, `seasons_found` |
| `data/rb_backtest_with_receiving.csv` | RB | 223 backtest RBs (2015-2025) | `rec_yards`, `team_pass_att`, `age` |
| `data/te_backtest_master.csv` | TE | 160 backtest TEs (2015-2025) | `breakout_age`, `peak_dominator`, `cfbd_rec_yards`, `te_ras`, `pff_yards` |
| `data/prospects_final.csv` | WR/RB | 165 2026 WR+RB prospects | `projected_pick`, `rec_yards`, `team_pass_attempts`, `age`, `weight` |
| `data/te_2026_prospects_final.csv` | TE | 52 2026 TE prospects | `projected_pick`, `breakout_age`, `ras_score`, `cfbd_rec_yards` |
| `output/slap_v5_wr_2026.csv` | WR | Pre-calculated 2026 WR scores | `early_declare_score`, `teammate_score` |
| `data/backtest_outcomes_complete.csv` | WR/RB | NFL outcomes for validation | `hit24`, `hit12`, `first_3yr_ppg`, `career_ppg` |
| `data/nflverse/combine.parquet` | RB | Combine measurements for Speed Score | `weight`, `forty` |

### Output Files (output/)

| File | Contents | Rows |
|------|----------|------|
| `output/slap_v5_master_database.csv` | **Unified master** — all positions, backtest + 2026 | 939 |
| `output/slap_v5_wr.csv` | WR only | 448 (339 backtest + 109 prospects) |
| `output/slap_v5_rb.csv` | RB only | 279 (223 backtest + 56 prospects) |
| `output/slap_v5_te.csv` | TE only | 212 (160 backtest + 52 prospects) |
| `output/slap_v5_2026_all.csv` | 2026 prospects only, ranked within position | 217 |

### Output Column Definitions

| Column | What It Is | Range | Use For |
|--------|-----------|-------|---------|
| `slap_display_score` | Within-position percentile rank vs backtest | 1-99 | **Publication** — what you show on YouTube/Patreon |
| `slap_model_score` | Native-scale weighted composite (the "real" score) | ~22-98 | **Validation** — what the model actually predicts |
| `dc_score` | Draft Capital component (percentile rank vs backtest) | 1-99 | Shows draft capital tier |
| `prospect_profile` | Weighted average of non-DC components | 0-100 | Shows how good the prospect profile is independent of DC |
| `dataset` | `backtest` or `2026_prospect` | — | Distinguishes historical vs current prospects |
| `bdr` | Backfield Dominator Rating (RB only) | 0-100 | SimScores similarity matching — **not used in SLAP scoring** |

## The Two-Layer Scoring System

**Critical design principle: never let display concerns touch the prediction engine.**

### Layer 1: `slap_model_score` (the prediction engine)
- Weighted composite of position-specific components on their native 0-100 scales
- This is what gets validated against NFL outcomes
- Changes here affect whether the model actually predicts well
- **Never normalize, percentile-rank, or rescale components before combining them**

### Layer 2: `slap_display_score` (the publication layer)
- Per-position percentile rank of `slap_model_score` against backtest distribution
- Formula: `percentileofscore(backtest_values, raw, kind='rank') / 100 * 98 + 1`, clipped 1-99
- Backtest median is always ~50, best backtest player is always 99
- 2026 prospects are scored against the same backtest reference
- **Rankings are always identical** between Layer 1 and Layer 2 (Spearman r = 1.000)

### Why percentile rank, not min-max rescaling?
- Min-max creates a false ceiling: if one backtest player is an extreme outlier (e.g., Saquon), it compresses everyone else
- Percentile rank distributes evenly: each point on the 1-99 scale represents ~1% of backtest players
- Within-position (not pooled): WR, RB, TE each have their own 1-99 scale
- Why not pooled? WR formulas produce higher raw scores than RB/TE. Pooling would make all top scores WRs
- Cross-position display differences reflect real formula differences, not a bug

### `prospect_profile` (non-DC quality indicator)
Shows how strong the prospect's non-draft-capital components are, on a 0-100 scale:
- WR: `(Enhanced_Breakout × 0.20 + Teammate × 0.05 + Early_Declare × 0.05) / 0.30`
- RB: `(RYPTPA × 0.30 + Speed_Score × 0.05) / 0.35`
- TE: `(Breakout × 0.15 + Production × 0.15 + RAS × 0.10) / 0.40`

A high `prospect_profile` with a low `dc_score` means the model thinks the player is better than their draft slot (a "riser"). The reverse means the model thinks they were overdrafted.

## V5 Formulas (All Locked)

### WR V5: 70/20/5/5
```
WR SLAP = DC × 0.70 + Enhanced_Breakout × 0.20 + Teammate × 0.05 + Early_Declare × 0.05
```

| Component | Weight | Scale | Source |
|-----------|--------|-------|--------|
| DC | 70% | `100 - 2.40 × (pick^0.62 - 1)` | Draft pick or projected pick |
| Enhanced Breakout | 20% | Age tier (18→100, 19→90, 20→75, 21→60, 22→45, 23→30) + dominator bonus + rush bonus | CFBD multi-season receiving data |
| Teammate | 5% | Tiered: 0/40/60/80/100 based on teammate DC + breakout + dominator tier | Same-year WR/TE drafted teammates |
| Early Declare | 5% | Binary: 100 if 3 or fewer college seasons, else 0 | College season count (NOT age) |

**Early Declare rule**: A player is early declare ONLY if they played **3 or fewer college seasons**. Age is irrelevant. Players who enrolled young (age 17) and played 4 full seasons are NOT early declares, even if drafted at age 21.

**Teammate Score (Tiered)**:
Two gates must be passed, then peak_dominator determines the tier:
1. `total_teammate_dc > 150` (enough drafted pass-catcher teammates)
2. Player must have broken out (hit 20%+ dominator, i.e., `breakout_age` is not NaN)

If either gate fails → **0**. If both pass, score by peak_dominator tier:
| Peak Dominator | Teammate Score |
|----------------|---------------|
| < 20% | 0 |
| 20–25% | 40 |
| 25–30% | 60 |
| 30–35% | 80 |
| 35%+ | 100 |

**Why tiered instead of binary?** Testing showed a clear staircase in NFL outcomes by dominator tier among players with high teammate DC. Tiered won 6/11 validation metrics vs binary's 5/11 and significantly improved Brier scores (hit24: 0.2784→0.2740, hit12: 0.3152→0.3095). Players who dominated despite elite teammates (35%+ dominator) deserve more credit than those who barely broke out (20–25%).

**Backtest distribution**: 288 at 0, 15 at 40, 15 at 60, 14 at 80, 7 at 100 (339 total).

**Breakout Age scoring**:
- Players who hit 20%+ dominator: base score from age tier + `min((dominator - 20) × 0.5, 9.9)` bonus
- Players who never hit 20%: `min(35, 15 + peak_dominator)` (fallback formula)
- Rush bonus: +5 if 20+ college rushing yards (capped at 99.9 total)
- Uses integer ages (season_year - birth_year), not exact birthdates

### RB V5: 65/30/5
```
RB SLAP = DC × 0.65 + RYPTPA × 0.30 + Speed_Score × 0.05
```

| Component | Weight | Scale | Source |
|-----------|--------|-------|--------|
| DC | 65% | `100 - 2.40 × (pick^0.62 - 1)` | Draft pick or projected pick |
| RYPTPA | 30% | `min(99.9, (rec_yards / team_pass_att) × age_weight × 100 / 1.75)` | CFBD API, final college season only |
| Speed Score | 5% | `normalize_0_100((weight × 200) / (forty^4))` with MNAR imputation | combine.parquet + CFBD player search |

**CRITICAL**: Always use **FINAL college season** for RB receiving (draft_year - 1). Never "best season."

**Age weight**: `season_age = draft_age - 1`, then `age_w = max(0.85, min(1.15, 1.15 - 0.05 × (season_age - 19)))`.

**Speed Score MNAR imputation**: 68% have real data. Weight recovery via CFBD gets to 96%. Missing 40 times estimated from weight×round bucket averages. Fully missing players: Rd 1-2 → 60th percentile (elite prospects skip workouts), Rd 3+ → 40th percentile.

**BDR (Backfield Dominator Rating)**: Included in the dataset but **NOT used in SLAP scoring**. BDR = average of 4 market shares (rush yards, rush TDs, rec yards, rec TDs vs all team RBs) × 100, calculated from CFBD position-level stats for the player's final college season. Tested and found redundant with RYPTPA for prediction (partial r collapses to +0.02–0.08 after controlling for DC + RYPTPA, all p > 0.38). Retained for SimScores similarity matching. Coverage: 199/223 backtest, 47/57 2026 prospects.

### TE V5: 60/15/15/10
```
TE SLAP = DC × 0.60 + Breakout × 0.15 + Production × 0.15 + RAS × 0.10
```

| Component | Weight | Scale | Source |
|-----------|--------|-------|--------|
| DC | 60% | `100 - 2.40 × (pick^0.62 - 1)` | Draft pick or projected pick |
| Breakout | 15% | Same age tiers as WR but **15% dominator threshold** (not 20%) | CFBD multi-season data |
| Production | 15% | `rec_yards / team_pass_att × age_weight × 100` (min-max normalized) | CFBD primary, PFF fallback |
| RAS | 10% | Relative Athletic Score × 10 (MNAR-imputed when missing) | combine.parquet |

**Key TE differences from WR/RB**:
- 15% dominator threshold (TEs have lower target shares)
- No Early Declare (no signal for TEs after controlling for DC)
- No Teammate Score (TEs at WR-rich programs get buried, opposite of WR finding)
- RAS kept at 10% (Speed Score and broad jump have real TE signal: r=+0.24, p=0.02)
- Production uses CFBD primary, PFF `pff_yards / pff_pass_plays × age_weight × 100` as fallback

## Validation Results

### Current Performance (Feb 2026, after all data quality fixes)

**WR V5** (339 backtest, validated against hit24, hit12, first_3yr_ppg, career_ppg):
- PRI-AVG: +0.4552 (priority-weighted average of 4 Spearman correlations)
- Top 10% hit24: 63.6% (21/33 top-scored WRs became fantasy-relevant)
- Top 10% PPG: 13.75
- V5 wins 11/11 metrics vs DC-only

**RB V5** (223 backtest):
- PRI-AVG: +0.565
- Top 10% hit24: 86.4% (19/22)
- Top 10% PPG: 17.86
- V5 wins 10/12 metrics vs V4 (1 tie, 1 marginal loss)

**TE V5** (160 backtest):
- AUC-ROC: 0.916 (top12_10g), 0.904 (top6_10g)
- SLAP wins 11/11 metrics vs DC-only

**Full validation**: `python src/full_validation_8gm.py` (all 3 positions, 7 test categories, bootstrap resampling)

## Post-Draft Update Workflow

When 2026 NFL Draft picks are final and combine data is available:

### Step 1: Update 2026 prospect pick numbers

**File: `data/prospects_final.csv`** (WR + RB prospects)
- Column to update: `projected_pick` → replace with actual draft pick
- Also update `weight` if combine data is now available (for RB Speed Score)

**File: `data/te_2026_prospects_final.csv`** (TE prospects)
- Column to update: `projected_pick` → replace with actual draft pick
- Also update: `ras_score` (if new RAS data available), `weight`, `height`

**File: `output/slap_v5_wr_2026.csv`** (WR pre-calculated scores)
- Column to update: `projected_pick` → actual draft pick
- Also verify: `early_declare_score`, `teammate_score` (these are pre-calculated)

### Step 2: Update athletic data (if newly available)

- **RB 40 times**: If new combine/pro day data, update `data/nflverse/combine.parquet` or add 40 times directly. The build script uses weight × 40 time for Speed Score.
- **TE RAS**: Update `ras_score` in `data/te_2026_prospects_final.csv`
- **RB receiving stats**: If a player's final college season stats were missing, update `rec_yards` and `team_pass_attempts` in `data/prospects_final.csv`

### Step 3: Rebuild
```bash
python src/build_master_database_v5.py
```
Backtest scores won't change (same data). Only 2026 prospect scores update.

### Step 4: Validate (optional — backtest metrics won't change)
```bash
python src/full_validation_8gm.py        # Full 3-position validation suite
python src/recalculate_slap_v5.py        # WR/RB V5 vs V4 comparison
```

### What NOT to change
- **Backtest data files** — these are locked. Changing them invalidates the model.
- **Component weights** — WR 70/20/5/5, RB 65/30/5, TE 60/15/15/10 are final.
- **Scoring formulas** — DC curve, breakout scoring, production scaling are all locked.

## Key Design Decisions

### 1. Why percentile rank over min-max for display scores
Min-max rescaling was tested (P5/P95 floor/ceiling) but rejected because:
- 37 players all tied at 99.0 at the top (lost all differentiation)
- Cross-position spread got worse, not better
- Percentile rank spreads scores evenly with no clipping

Pooled percentile (all 722 players regardless of position) was also tested and rejected:
- Top 20 was 19 WR + 1 RB + 0 TE (WR formulas produce higher raw scores)
- Cross-position spread doubled in most rounds

### 2. Why early declare uses college seasons, not age
Original logic: `draft_age <= 21.5 → early declare`. This was wrong because players who enrolled at 17 play 4 full college seasons but are still 21 at draft. Fixed rule: **3 or fewer college seasons = early declare, period**. This fix improved WR validation (PRI-AVG +0.004, all 8 correlation metrics improved). 20 total players corrected, all 0-for-hit24 (corrections directionally correct).

### 3. Why athletic testing has minimal weight at WR/RB but 10% at TE
- **WR (0%)**: RAS removed in V5. DC already prices in athleticism (NFL teams see combine before drafting). RAS had severe missing-data problems and added zero predictive value after controlling for DC.
- **RB (5% Speed Score)**: Speed Score doesn't improve the 12-metric battery, but adds athlete diversity to rankings. User preference for content creation.
- **TE (10% RAS)**: Speed Score and broad jump have real signal for TEs (r=+0.24, p=0.02). Unlike WR/RB, TE athleticism adds independent value beyond what DC captures.

### 4. The two-layer system
The model was nearly broken when percentile normalization was applied to components before combining them. Diagnostic testing showed this destroyed the production signal for all 3 positions. The solution: keep components on their native scales for prediction (Layer 1), then percentile-rank the final composite for publication (Layer 2). The display layer can never affect the prediction engine.

### 5. TE production uses CFBD primary, PFF fallback
CFBD has direct `rec_yards / team_pass_att`. When CFBD data is missing (smaller schools), the build script falls back to PFF: `pff_yards / pff_pass_plays × age_weight × 100`. This recovered production data for 17 TEs that would otherwise have been imputed at the mean.

### 6. Why teammate score uses tiered dominator instead of binary
Original logic: `total_teammate_dc > 150 → 100, else → 0`. This was improved in two steps:
1. **Breakout gate added**: Players who never hit 20% dominator get TM=0 regardless of teammate DC. This removed 3 false positives (Van Jefferson, Racey McMath, Freddie Swain — all 0-for-hit24).
2. **Tiered scoring**: Among players who pass both gates, peak_dominator determines the tier (40/60/80/100). Testing showed a clear staircase in NFL outcomes by dominator level. Four formulas were tested (Simple Dominator, Interaction, Dom Above Threshold, Tiered); Tiered won 6/11 metrics vs binary's 5/11 and improved all Brier scores.

### 7. Breakout age data integrity
- WR breakout_age must be NaN (not a sentinel like 99) when a player never hit 20% dominator. The scoring function uses a different formula path for NaN vs integer ages.
- TE peak_dominator must be capped at 100 (values above 100% indicate CFBD team-receiving-yard calculation errors). The bonus is capped at +9.9 regardless, so values > 100 don't affect scoring, but they shouldn't appear in published data.
- Breakout_age uses integer ages (season_year minus birth_year), not exact birthdates.

## Other Commands

```bash
# Full validation suite (all 3 positions, 8-game minimum)
python src/full_validation_8gm.py

# WR/RB V5 vs V4 vs DC-only comparison
python src/recalculate_slap_v5.py

# Fetch RB receiving stats from CFBD API (if updating data)
python src/fetch_rb_receiving_stats.py

# Calculate WR breakout ages from CFBD data (if updating 2026 prospects)
python src/calculate_wr_breakout_age.py

# Apply data fixes to WR backtest (early declare overrides, rush yards, etc.)
python src/apply_data_fixes.py

# Update 2026 WR/RB mock draft picks and recalculate (standalone)
python src/update_2026_mock_and_calc_v5.py

# TE 2026 prospect scores (standalone, also included in master build)
python src/calculate_te_slap_2026.py

# Test continuous teammate score formulas (analysis script)
python src/test_continuous_teammate.py
```

## Technical Preferences

- **Language**: Python
- **Data Storage**: CSV files (can be opened in Excel)
- **Visualizations**: Clear charts for content creation

## Data Sources

- **CFBD API**: College receiving yards, team receiving yards, team pass attempts, rushing yards (primary for WR/RB/TE)
- **PFF**: TE receiving data (fallback when CFBD missing), 63 sub-metrics tested for TE model
- **NFLVerse**: `combine.parquet` (weight, 40 time for Speed Score), `draft_picks` (draft outcomes)
- **RAS**: Relative Athletic Score (Kent Lee Platte), used for TE only
- **Mock drafts**: Consensus projected picks for 2026 prospects (replaced with actuals post-draft)
