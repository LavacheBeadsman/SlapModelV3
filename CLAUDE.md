# CLAUDE.md - SLAP Score V3 Project Guide

## About the User

- **No coding background** - Explain everything in plain English, avoid jargon
- Fantasy football content creator (YouTube and Patreon)
- Wants to understand what the code does, not just have it work
- **Honesty is critical** - Never claim something is possible if it isn't

## Project Overview

**SLAP Score V3**: Statistical Likelihood of Achieving Production

A draft-capital-anchored prospect model that rates 2026 NFL Draft RBs and WRs on a 0-100 scale.

### Model Outputs
1. A **0-100 SLAP Score** for each prospect
2. A **delta vs draft-only baseline** - Shows when the model disagrees with where a player was drafted (positive = model likes them more than their draft slot, negative = model likes them less)

## The Formula (Core Structure)

### 1. Draft Capital Anchor
Transform draft pick into a strength score using the "gentler curve" formula:
```
DC = 100 - 2.40 × (pick^0.62 - 1)
```

**Why This Formula?**
- Creates a gentler decay than the original 1/sqrt(pick) formula
- Better differentiates between early picks while still penalizing late picks
- Produces more intuitive scores that match expectations

**DC Score Examples:**
| Pick | DC Score |
|------|----------|
| 1    | 100      |
| 5    | ~96      |
| 10   | ~92      |
| 32   | ~82      |
| 64   | ~71      |
| 100  | ~61      |
| 150  | ~49      |
| 200  | ~38      |
| 250  | ~29      |

- Uses actual draft pick OR expected pick from consensus mock drafts (for pre-draft analysis)
- Higher picks = higher scores (pick 1 is best)

### 2. Production Component (Position-Specific)

**For WRs: Enhanced Breakout Age (Continuous Scoring + Rush Bonus)**
```
enhanced_breakout = min(wr_breakout_score(breakout_age, dominator_pct) + rush_bonus, 99.9)
```
Where:
- `breakout_age` = age when player first hit 20%+ dominator rating
- `rush_bonus` = +5 if player had 20+ college rushing yards, 0 otherwise

**Continuous Scoring Formula (Feb 2026 update):**
Uses age tier as base score + dominator magnitude as tiebreaker:
```python
# For players who broke out (hit 20%+ dominator):
base_score = age_tier (100, 90, 75, 60, 45, 30, 20)
bonus = min((dominator_pct - 20) × 0.5, 9.9)
breakout_score = min(base_score + bonus, 99.9)

# For players who never broke out:
breakout_score = min(35, 15 + dominator_pct)  # Maps 0-20% to 15-35

# Then add rush bonus:
rush_bonus = 5 if rush_yards >= 20 else 0
enhanced_breakout = min(breakout_score + rush_bonus, 99.9)
```

**Age Tier Base Scores:**
- Age 18: 100 → final range 99.9 (capped)
- Age 19: 90 → final range 90.0 - 99.9
- Age 20: 75 → final range 75.0 - 84.9
- Age 21: 60 → final range 60.0 - 69.9
- Age 22: 45 → final range 45.0 - 54.9
- Age 23: 30 → final range 30.0 - 39.9
- Never hit 20%: 15-35 (based on peak dominator)

**Why the +5 Rush Bonus?**
- WRs with 20+ college rushing yards show slightly better NFL outcomes
- Signals versatility and scheme usage (jet sweeps, gadget plays, etc.)
- Small bonus (+5 on a 0-100 scale) — a tiebreaker, not a game-changer
- Binary threshold (20 yards) keeps it simple and avoids overfitting to exact yardage

**Why Continuous Scoring?**
- Discrete tiers created artificial cliffs (19.9 years = 90, 20.0 years = 75)
- Dominator tiebreaker differentiates within age tiers
- Creates 180+ unique scores vs 7 discrete tiers
- Works with integer ages (backtest data has no exact birthdates)

**Why Breakout Age for WRs?**
- Dominator Rating alone had weak correlation (r=0.175) with NFL success
- High dominator often came from weak competition (small schools, late-round busts)
- Breakout Age has stronger correlation (r=0.395) - younger breakouts predict NFL success
- Draft capital already prices in production context (teammates, transfers, competition)
- This avoids penalizing players like Jaylen Waddle (competed with 3 future 1st-rounders)

**For RBs: Receiving Production Score (Continuous Scoring)**
```
production = (receiving_yards / team_pass_attempts) × age_weight × 100 / 1.75
```

**Continuous Scoring Formula (Feb 2026 update):**
```python
raw_score = (rec_yards / team_pass_att) × age_weight × 100
scaled_score = raw_score / 1.75  # Normalize to 0-99.9 range
final_score = min(99.9, scaled_score)
```

Where `age_weight` adjusts for when production occurred:
- Season age 19: 1.15x (15% bonus for early production)
- Season age 20: 1.10x (10% bonus)
- Season age 21: 1.05x (5% bonus)
- Season age 22: 1.00x (baseline)
- Season age 23: 0.95x (5% penalty)
- Season age 24+: 0.90x (10% penalty)

**Why Scale by 1.75?**
- Old formula capped 21 RBs at 100, losing differentiation at the top
- Elite producers like Antonio Gibson (175 raw) and Joe Mixon (154 raw) were all 100
- Scaling by 1.75 spreads scores across 0-99.9 range:
  - Antonio Gibson: 99.9 (was 100)
  - Joe Mixon: 88.1 (was 100)
  - Saquon Barkley: 86.7 (was 100)
  - Average RB: ~30 (was ~52)

**Why Receiving Production for RBs (not Breakout Age)?**
- Backtest analysis (2015-2024, 188 RBs) proved receiving production is better:
  - Production metric: r=0.30, **adds significant value beyond DC (p=0.004)**
  - Breakout age: r=0.10, **does NOT add value (p=0.80)**
- Receiving yards / team pass attempts measures a RB's share of the passing game
- Age-weighting rewards younger players who caught passes early
- Creates 189 unique scores (fully continuous)

**CRITICAL: Season Selection for RB Production**
- **ALWAYS use FINAL college season** (draft_year - 1)
- This matches the backtest methodology that validated the metric
- For 2026 draft class: use 2025 college season
- For 2025 draft class: use 2024 college season
- Do NOT use "best season" - this was not validated and inflates scores
- Exception: If player has no final season data (injury, transfer mid-season), use most recent available season and flag as "no_final_season_data"

### 3. Teammate Score (WR only)

**Binary elite-program indicator:**
```
teammate_score = 100 if total_teammate_dc > 150 else 0
```
- `total_teammate_dc` = sum of DC scores for all WR/TE teammates drafted in the same draft year
- Threshold of 150 means the player shared targets with other high-draft-capital receivers
- Binary (0 or 100) — either the player competed for targets with elite teammates, or didn't

**Why Teammate Score?**
- Players who produced despite sharing with other highly-drafted receivers showed better NFL outcomes
- This is the opposite of what you might expect — it signals the player earned targets in a competitive room
- Examples: Jaylen Waddle (Alabama with DeVonta Smith), Chris Olave (Ohio State with Garrett Wilson)
- Only a 5% weight — a small tiebreaker, not a major driver

### 4. Early Declare (WR only)

**Binary declaration status:**
```
early_declare = 100 if player declared early (before senior season) else 0
```
- Players who left college early (juniors, redshirt sophomores) score 100
- Players who stayed for their full eligibility score 0

**Why Early Declare?**
- Early declaration signals NFL-readiness and confidence in draft stock
- Correlates with youth at draft (younger players tend to have longer NFL careers)
- Draft capital already captures some of this, but early declare adds a small independent signal
- Only a 5% weight — another small tiebreaker

### 5. Athletic Modifier (RB only — pending reoptimization)

**For RBs:**
```
athletic = normalize(speed_score(forty_time, weight))
```
- Uses Barnwell Speed Score: (weight × 200) / (40 time)^4
- Rewards players who are fast for their size

**Note on WR RAS (Removed in V5):**
RAS (Relative Athletic Score) was used in previous WR SLAP versions at 15% weight.
Extensive testing during the V5 reoptimization showed RAS does not add predictive value
for WRs after controlling for draft capital. The new 4-component WR model (DC / Enhanced
Breakout / Teammate / Early Declare) outperforms the old 3-component model (DC / Breakout / RAS)
on every metric tested. See Decision #9 for full details.

RB athletic component (Speed Score) is retained for now pending RB model reoptimization.

## Decisions Made

1. **Component Weights**: Position-Specific (Updated Feb 2026 — WR V5)

   **WRs (V5): 70% DC / 20% Enhanced Breakout / 5% Teammate / 5% Early Declare**
   **RBs: 50% DC / 35% Production / 15% Speed Score** *(pending reoptimization)*

   **WR V5 — 4-Component Model (Feb 2026 update):**

   The WR model was reoptimized from a 3-component model (DC/Breakout/RAS) to a
   4-component model (DC/Enhanced Breakout/Teammate/Early Declare). RAS was dropped
   because it does not add predictive value after controlling for draft capital.

   Systematic testing of DC weight from 75% down to 50% found:
   - 75/17/4/4 and 70/20/5/5 perform identically on top-decile precision (62.5% hit24 rate)
   - Below 70% DC, top-decile hit24 drops (59.4% at 65%, 56.2% at 60%)
   - 70% DC chosen as the floor: no performance loss vs 75%, more room for non-DC components

   **WR V5 vs V4 vs DC-only (priority-weighted correlation, 40/25/20/15):**
   | Config | PRI-AVG r | Top 10% Hit24 | Top 10% PPG | Disagree 10+ |
   |--------|-----------|---------------|-------------|--------------|
   | DC only | +0.435 | 53.1% | 10.70 | 0 |
   | V4 (65/20/15 DC/BO/RAS) | ~+0.445 | ~56% | ~10.9 | ~220 |
   | V5 (70/20/5/5 DC/BO+R/TM/ED) | +0.452 | 62.5% | 11.36 | 246 |

   Key improvements in V5:
   - Top-decile hit24: 62.5% (vs ~56% in V4, 53.1% DC-only)
   - 246 ranking disagreements vs pure DC (interesting content)
   - Boosted players average +1.3 PPG over dinged players (disagreements are correct)

   **RB Weights Stay at 50/35/15** *(pending reoptimization)*:
   - RB receiving production is statistically significant (p=0.004)
   - RB model will be reoptimized with the same 4-component approach
   - Do NOT recalculate RB scores until reoptimization is complete

2. **Age Weight Function (RB Production)**: Moderate adjustment
   - Used to weight RB receiving production by college age
   - Age 19: 1.20x (20% bonus for early production)
   - Age 20: 1.10x (10% bonus)
   - Age 21: 1.00x (baseline)
   - Age 22: 0.90x (10% penalty)
   - Age 23+: 0.80x (20% penalty)

3. **Athletic Score Function**: Position-specific
   - RBs: Speed Score (Barnwell formula) = (Weight × 200) / (40 time)^4
   - **WRs: RAS removed in V5** — replaced by Teammate Score and Early Declare
   - See Decision #9 for full rationale on RAS removal

4. **Position Handling**: Position-Split Production Metrics
   - **RBs**: Receiving yards ÷ Team pass attempts × age weight (validated: p=0.004)
   - **WRs**: Breakout Age scoring (age when first hit 20%+ dominator)
   - Both metrics normalized 0-100 within position (50 = average)
   - Data source for RB receiving: CFBD API (193/208 RBs = 92.8% coverage)
   - Outputs separate RB and WR rankings

5. **WR Breakout Age Methodology** (updated after backtest analysis)
   - Originally tested Dominator Rating but it had weak signal (r=0.175)
   - Dominator penalized players with elite teammates (Waddle at Alabama)
   - Breakout Age (when player first hit 20% dominator) has stronger signal (r=0.395)
   - Logistic regression coefficient is positive (+0.388), confirming predictive value
   - Draft capital already accounts for context (teammates, transfers, competition level)

6. **Missing Athletic Data Handling** (MNAR-aware approach)

   **WRs (V5): No longer applicable.** RAS was removed from the WR model. All four
   WR V5 components (DC, Enhanced Breakout, Teammate, Early Declare) have near-complete
   data coverage, eliminating the missing-data problem that plagued the RAS-based model.

   **RBs: Still applies** (pending reoptimization). Missing Speed Score data is handled
   by imputing the position average:
   ```
   IF Speed Score missing:
       RB SLAP = DC × 0.50 + Production × 0.35 + Avg_SpeedScore × 0.15
       Status: "imputed"
   ELSE:
       RB SLAP = DC × 0.50 + Production × 0.35 + SpeedScore × 0.15
       Status: "observed"
   ```

   **Historical context (WR RAS):** Missing RAS data was NOT random — elite prospects
   skipped workouts BECAUSE they were already valued (MNAR pattern). Round 1 WRs with
   missing RAS had 100% hit rate (Waddle, Smith, London, Williams) while Round 2-7 WRs
   with missing RAS had 0% hit rate. This asymmetry was one reason RAS was ultimately
   removed from the WR model — it created more problems than it solved.

7. **Season Selection for RB Production** (CRITICAL for consistency)
   - **ALWAYS use FINAL college season** (draft_year - 1)
   - Backtest validated using final season only - do NOT use "best season"
   - Using "best season" inflates scores and creates methodological inconsistency

   **Season Mapping:**
   | Draft Year | College Season to Use |
   |------------|----------------------|
   | 2026       | 2025                 |
   | 2025       | 2024                 |
   | 2024       | 2023                 |
   | etc.       | draft_year - 1       |

   **Exception Handling:**
   - If player has NO final season data (injury, redshirt, mid-season transfer):
     - Use most recent available season
     - Flag status as "no_final_season_data"
     - Document the exception

   **Why This Matters:**
   - Backtest showed production predicts NFL success using final season
   - Using "best season" cherry-picks data not validated as predictive
   - 19 players had inflated scores before this fix (avg +12.6 production, +1.26 SLAP)

8. **DC Formula Change: Gentler Curve (Option B)** (Jan 2026)
   - Changed from `normalize(1/sqrt(pick))` to `100 - 2.40 × (pick^0.62 - 1)`

   **Why We Changed It:**
   - Old formula created steep drops early (pick 1 to pick 10 dropped ~30 points)
   - Scouts don't see that much difference between pick 1 and pick 10
   - New formula creates gentler, more realistic decay

   **Old vs New DC Scores:**
   | Pick | Old DC | New DC |
   |------|--------|--------|
   | 1    | 100    | 100    |
   | 5    | 81     | 96     |
   | 10   | 69     | 92     |
   | 32   | 44     | 82     |
   | 100  | 24     | 61     |
   | 200  | 13     | 38     |

   **Impact on Final SLAP Scores:**
   - Scores now range from ~21 (late picks, poor profile) to ~99 (early picks, elite profile)
   - WR average SLAP: ~65 (was ~55)
   - RB average SLAP: ~50 (was ~40)
   - Scores better match intuitive expectations (pick 10 player with great profile = ~90+)

9. **WR V5: RAS Removed, 4-Component Model Adopted** (Feb 2026)

   RAS (Relative Athletic Score) was extensively tested and **removed** from the WR model.
   It was replaced by two new components: Teammate Score and Early Declare.

   **Why RAS Was Removed:**
   - RAS does not add predictive value for WRs after controlling for draft capital
   - DC already prices in athleticism — NFL teams see combine results before drafting
   - RAS had severe missing-data problems (MNAR pattern): elite prospects skip workouts,
     creating a biased sample where the best players often had no RAS data
   - Imputing missing RAS introduced noise without improving predictions
   - Removing RAS and adding Teammate + Early Declare improved every metric tested

   **What Replaced RAS (15% total → 5% Teammate + 5% Early Declare):**
   - **Teammate Score (5%)**: Binary — did the player produce despite sharing with other
     highly-drafted receivers? Signals competitive production in elite programs.
   - **Early Declare (5%)**: Binary — did the player leave early? Signals NFL-readiness
     and youth at draft.
   - Both are simple binary flags with near-complete data coverage (no missing-data issues)

   **V5 Formula:**
   ```
   WR SLAP = DC × 0.70 + Enhanced_Breakout × 0.20 + Teammate × 0.05 + Early_Declare × 0.05
   ```

   Where:
   - DC = `100 - 2.40 × (pick^0.62 - 1)` (gentler curve)
   - Enhanced_Breakout = breakout age score + 5 if 20+ college rush yards
   - Teammate = 100 if total_teammate_dc > 150, else 0
   - Early_Declare = 100 if declared early, else 0

   **Status:** WR V5 formula is locked. RB model reoptimization is pending.

## Technical Preferences

- **Language**: Python
- **Data Storage**: CSV files (can be opened in Excel)
- **Visualizations**: Clear charts for content creation

## How to Work Together

1. **Check in before major decisions** - Don't assume, ask
2. **Explain code in simple terms** - What does it do and why
3. **Small steps** - Break work into pieces I can follow
4. **Options with tradeoffs** - When deciding something, give me choices with clear pros and cons
5. **NEVER estimate, guess, or make up data** - If data is missing, flag it and ask me how to handle it

## Project Structure

```
SlapModelV3/
├── CLAUDE.md          # This file
├── README.md          # Project description
├── data/              # CSV files with prospect data (to be created)
├── src/               # Python code (to be created)
└── output/            # Generated scores and charts (to be created)
```

## Data Requirements

**For all prospects:**
- Name, position, school
- Draft pick (actual or projected)
- Age at draft
- Weight, 40-yard dash time (for athletic score)

**For WRs specifically:**
- Breakout age (age when first hit 20%+ dominator rating)
- Requires multi-season college data to calculate
- College rushing yards (for +5 enhanced breakout bonus if 20+ yards)
- Teammate draft capital (total DC of WR/TE teammates drafted same year)
- Early declare status (did player leave before senior season?)

**For RBs specifically:**
- Receiving yards (best college season)
- Team pass attempts (same season)
- Data source: CFBD API (92.8% coverage for 2015-2024 backtest)

## Commands

```bash
# MAIN COMMAND: Recalculate ALL SLAP scores
# WR V5: 70/20/5/5 (DC / Enhanced Breakout / Teammate / Early Declare)
# RB: 50/35/15 (DC / Production / Speed Score) — pending reoptimization
python src/recalculate_all_slap_new_dc.py   # ← needs update for V5 formula

# Fetch RB receiving stats from CFBD API
python src/fetch_rb_receiving_stats.py

# Update WR breakout scores with age-only approach
python src/update_wr_breakout.py

# Refresh data from APIs (birthdates, stats)
python src/fill_missing_ages.py

# Legacy commands (superseded)
# python src/generate_slap_v3_fixed.py
# python src/calculate_2026_slap.py
# python src/calculate_slap_unified.py
# python src/build_master_database_50_35_15.py
```

## Output Files

### Current Output (WR V5: 70/20/5/5, RB: 50/35/15 pending reopt, gentler DC curve)
- `output/slap_complete_database_v4.csv` - Master database with all SLAP scores (WRs + RBs, 2015-2026)
- `output/slap_complete_all_players.csv` - All SLAP scores (WRs + RBs, 2015-2026)
- `output/slap_complete_wr.csv` - All WR SLAP scores (2015-2026)
- `output/slap_complete_rb.csv` - All RB SLAP scores (2015-2026)
- `output/slap_wr_2026.csv` - 2026 WR class projections
- `output/slap_rb_2026.csv` - 2026 RB class projections

### Data Files
- `data/rb_backtest_with_receiving.csv` - RB backtest data with receiving stats from CFBD
- `data/wr_backtest_expanded_final.csv` - WR backtest data with breakout ages
- `data/wr_backtest_all_components.csv` - WR backtest with all V5 components (breakout, rush yards, early declare)
- `data/wr_teammate_scores.csv` - WR teammate DC scores
- `data/prospects_final.csv` - 2026 prospect data

### Legacy Output (superseded)
- `output/slap_v3_fixed_all_players.csv` - Old scores (before DC formula change)
- `output/slap_2026_wr.csv` - Old 2026 WR projections
- `output/slap_2026_rb.csv` - Old 2026 RB projections
- `output/slap_master_*_50_35_15.csv` - Old master databases
