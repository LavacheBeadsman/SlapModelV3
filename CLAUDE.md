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

**For WRs: Breakout Age (Continuous Scoring)**
```
breakout = wr_breakout_score(breakout_age, dominator_pct)
```
Where `breakout_age` = age when player first hit 20%+ dominator rating

**Continuous Scoring Formula (Feb 2026 update):**
Uses age tier as base score + dominator magnitude as tiebreaker:
```python
# For players who broke out (hit 20%+ dominator):
base_score = age_tier (100, 90, 75, 60, 45, 30, 20)
bonus = min((dominator_pct - 20) × 0.5, 9.9)
final_score = min(base_score + bonus, 99.9)

# For players who never broke out:
final_score = min(35, 15 + dominator_pct)  # Maps 0-20% to 15-35
```

**Age Tier Base Scores:**
- Age 18: 100 → final range 99.9 (capped)
- Age 19: 90 → final range 90.0 - 99.9
- Age 20: 75 → final range 75.0 - 84.9
- Age 21: 60 → final range 60.0 - 69.9
- Age 22: 45 → final range 45.0 - 54.9
- Age 23: 30 → final range 30.0 - 39.9
- Never hit 20%: 15-35 (based on peak dominator)

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

**CRITICAL: Season Selection for RB Production (Updated Feb 2026)**
- **USE CAREER AVERAGE** across all college seasons
- This was validated as the best predictor (Feb 2026 analysis of 208 RBs):

| Method | NFL PPG Corr | Hit24 Corr | Partial Corr (controlling for DC) | P-value |
|--------|-------------|------------|-----------------------------------|---------|
| **Career Average** | **0.2415** | **0.1660** | **0.1748** | **0.015** ✓ |
| Final Season | 0.2355 | 0.1469 | 0.1332 | 0.065 |
| Best Season | 0.1905 | 0.1469 | 0.1161 | 0.109 |

**Why Career Average?**
- Only method with statistically significant partial correlation (p=0.015)
- Smooths out single-season noise (injuries, QB changes, scheme shifts)
- Best Season performs WORST because it cherry-picks outliers
- Career Average captures consistent pass-catching ability across multiple years

**Calculation:**
```python
career_avg_ratio = mean(rec_yards / team_pass_att) for all college seasons
production_score = career_avg_ratio × age_weight × 100 / 1.75
```

**Exception:** If player has only 1 season of data, use that season

### 3. Athletic Modifier

**For RBs:**
```
athletic = normalize(speed_score(forty_time, weight))
```
- Uses Barnwell Speed Score: (weight × 200) / (40 time)^4
- Rewards players who are fast for their size

**For WRs:**
```
athletic = normalize(RAS)
```
- Uses Relative Athletic Score (RAS) from Kent Lee Platte (0-10 scale)
- Composite metric combining 40-yard dash, vertical, broad jump, 3-cone, shuttle, and bench
- Data source: `data/WR_RAS_2020_to_2025.csv`
- For missing RAS: threshold-based handling (see Decision #6)

## Decisions Made

1. **Component Weights**: Position-Specific (Updated Jan 2026)

   **WRs: 65% DC / 20% Breakout / 15% RAS**
   **RBs: 50% DC / 35% Production / 15% RAS**

   **Why Different Weights by Position?**

   Testing revealed WR breakout age has weaker predictive power than RB receiving production:
   - WR breakout age correlation: r=0.155 (weak)
   - RB receiving production correlation: r=0.30 (moderate, p=0.004)

   For WRs, adding breakout at 35% weight actually HURT predictions:
   - DC only: r=0.531
   - DC + 35% breakout: r=0.477 (worse!)
   - DC + 20% breakout: r=0.519 (optimal balance)

   **WR Weight Analysis (65/20/15):**
   | Metric | 50/35/15 (old) | 65/20/15 (new) | Change |
   |--------|----------------|----------------|--------|
   | PPR Correlation | 0.477 | 0.519 | +9% |
   | Hit24 Correlation | 0.404 | 0.450 | +11% |
   | Avg Delta | 11.5 pts | 7.7 pts | Smaller but meaningful |

   **RB Weights Stay at 50/35/15:**
   - RB receiving production is statistically significant (p=0.004)
   - Adding production at 35% improves predictions
   - No change needed for RBs

   **Athletic component (15% for both)**:
   - RAS for WRs, Speed Score for RBs
   - Unchanged from previous version

2. **Age Weight Function (RB Production)**: Moderate adjustment
   - Used to weight RB receiving production by college age
   - Age 19: 1.20x (20% bonus for early production)
   - Age 20: 1.10x (10% bonus)
   - Age 21: 1.00x (baseline)
   - Age 22: 0.90x (10% penalty)
   - Age 23+: 0.80x (20% penalty)

3. **Athletic Score Function**: Position-specific
   - RBs: Speed Score (Barnwell formula) = (Weight × 200) / (40 time)^4
   - WRs: RAS (Relative Athletic Score) = composite 0-10 scale from combine metrics
   - RAS tested: Young+HighRAS has 25% hit rate vs 18.2% for Young+LowRAS
   - Combination of breakout age + RAS slightly improves signal (r=0.411 vs r=0.388)

4. **Position Handling**: Position-Split Production Metrics
   - **RBs**: Receiving yards ÷ Team pass attempts × age weight (validated: p=0.004)
   - **WRs**: Breakout Age scoring (age when first hit 20%+ dominator)
   - Both metrics use **PERCENTILE NORMALIZATION** within position (see Decision #9)
   - Data source for RB receiving: CFBD API (193/208 RBs = 92.8% coverage)
   - Outputs separate RB and WR rankings

5. **WR Breakout Age Methodology** (updated after backtest analysis)
   - Originally tested Dominator Rating but it had weak signal (r=0.175)
   - Dominator penalized players with elite teammates (Waddle at Alabama)
   - Breakout Age (when player first hit 20% dominator) has stronger signal (r=0.395)
   - Logistic regression coefficient is positive (+0.388), confirming predictive value
   - Draft capital already accounts for context (teammates, transfers, competition level)

6. **Missing Athletic Data Handling** (MNAR-aware approach)

   Missing RAS data is NOT random - elite prospects skip workouts BECAUSE they're already valued.
   This is "Missing Not At Random" (MNAR). Evidence from backtest:
   - Round 1 WRs with missing RAS: 4 players, 100% hit rate (Waddle, Smith, London, Williams)
   - Round 2-7 WRs with missing RAS: 16 players, 0% hit rate

   **Threshold-based handling:**

   ```
   IF RAS missing:
       # Use position average for RAS
       WR SLAP = DC × 0.65 + Breakout × 0.20 + Avg_RAS × 0.15
       RB SLAP = DC × 0.50 + Production × 0.35 + Avg_RAS × 0.15
       Status: "imputed"

   ELSE:
       # Has RAS data - use full formula
       WR SLAP = DC × 0.65 + Breakout × 0.20 + RAS × 0.15
       RB SLAP = DC × 0.50 + Production × 0.35 + RAS × 0.15
       Status: "observed"
   ```

   **Note**: With WR weights at 65/20/15, DC dominates the score. Elite prospects
   with missing RAS still get fair scores from their strong DC component.

7. **Season Selection for RB Production** (Updated Feb 2026 - Career Average)
   - **USE CAREER AVERAGE** - average rec_yards/team_pass_att across ALL college seasons
   - Feb 2026 analysis tested 3 methods on 208 RBs (2015-2024):
     - Career Average: r=0.2415, partial r=0.1748 (p=0.015) ✓ BEST
     - Final Season: r=0.2355, partial r=0.1332 (p=0.065)
     - Best Season: r=0.1905, partial r=0.1161 (p=0.109) WORST

   **Why Career Average Works Best:**
   - Only method with statistically significant value beyond DC (p=0.015)
   - Smooths out single-season noise (injuries, QB changes, scheme shifts)
   - Best Season cherry-picks outliers and actually hurts predictions
   - Captures consistent pass-catching ability across college career

   **Data Source:**
   - `data/college_receiving_2011_2023.csv` for multi-season data
   - 127 of 208 backtest RBs had 2+ college seasons
   - 184 of 208 were matched to multi-season data (88%)

   **Exception:** Players with only 1 season of data use that season

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
   - Scores now range from ~17 (late picks, poor profile) to ~99 (early picks, elite profile)
   - WR average SLAP: ~53 (after calibration fix)
   - RB average SLAP: ~52 (after calibration fix)
   - Scores better match intuitive expectations (pick 10 player with great profile = ~90+)

9. **Production Score Calibration: Percentile Normalization** (Feb 2026)

   **The Problem:**
   WR breakout scores and RB receiving scores were on completely different scales:
   - WR production scores: Mean=63, Median=66 (most scored 50-90)
   - RB production scores: Mean=24, Median=21 (most scored 10-40)

   This caused WRs to dramatically outscore RBs (WR SLAP mean was 57.8 vs RB mean of 43.6).
   In the top 50 players by SLAP, 48 were WRs and only 2 were RBs.

   **The Fix: Percentile Normalization**
   Production scores are now converted to percentile ranks within each position:
   ```python
   # Convert raw score to percentile (0-100) within position
   production_score = percentile_rank(raw_score, position_reference_distribution) * 100
   ```

   **Result:**
   - A 90th percentile WR production → score of 90
   - A 90th percentile RB production → score of 90
   - Median player in each position → score of ~50

   **Before vs After:**
   | Metric | Before Fix | After Fix |
   |--------|------------|-----------|
   | WR SLAP Mean | 57.8 | 53.3 |
   | RB SLAP Mean | 43.6 | 52.4 |
   | Gap | 14.2 pts | 0.9 pts |
   | Top 50 (WR/RB) | 48/2 | 39/11 |

   **Raw scores preserved:**
   The original raw production scores are stored in `production_score_raw` column
   for reference. The SLAP calculation uses the percentile-normalized `production_score`.

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
- RAS (Relative Athletic Score) from combine/pro day metrics

**For RBs specifically:**
- Receiving yards and team pass attempts for ALL college seasons (career average)
- Data source: CFBD API via `data/college_receiving_2011_2023.csv`
- Coverage: 88% of 2015-2024 backtest RBs matched to multi-season data

## Commands

```bash
# MAIN COMMAND: Recalculate ALL SLAP scores (WR: 65/20/15, RB: 50/35/15)
python src/recalculate_all_slap_new_dc.py

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

### Current Output (WR: 65/20/15, RB: 50/35/15, gentler DC curve)
- `output/slap_complete_database_v4.csv` - Master database with all SLAP scores (WRs + RBs, 2015-2026)
- `output/slap_complete_all_players.csv` - All SLAP scores (WRs + RBs, 2015-2026)
- `output/slap_complete_wr.csv` - All WR SLAP scores (2015-2026)
- `output/slap_complete_rb.csv` - All RB SLAP scores (2015-2026)
- `output/slap_wr_2026.csv` - 2026 WR class projections
- `output/slap_rb_2026.csv` - 2026 RB class projections

### Data Files
- `data/rb_backtest_with_receiving.csv` - RB backtest data with receiving stats from CFBD
- `data/wr_backtest_expanded_final.csv` - WR backtest data with breakout ages
- `data/prospects_final.csv` - 2026 prospect data

### Legacy Output (superseded)
- `output/slap_v3_fixed_all_players.csv` - Old scores (before DC formula change)
- `output/slap_2026_wr.csv` - Old 2026 WR projections
- `output/slap_2026_rb.csv` - Old 2026 RB projections
- `output/slap_master_*_50_35_15.csv` - Old master databases
