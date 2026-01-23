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
Transform draft pick into a strength score:
```
dc = normalize(1 / sqrt(draft_pick))
```
- Uses actual draft pick OR expected pick from consensus mock drafts (for pre-draft analysis)
- Higher picks = higher scores (pick 1 is best)

### 2. Breakout Score (Production Core)

**For RBs:**
```
prod = rec_yards / team_pass_attempts
breakout = normalize(prod * age_weight)
```

**For WRs (Breakout Age):**
```
breakout = age_score(breakout_age)
```
Where `breakout_age` = age when player first hit 20%+ Dominator Rating

Age Score mapping:
- Age 18: 100 (freshman breakout = elite)
- Age 19: 90
- Age 20: 75
- Age 21: 60
- Age 22: 45
- Age 23: 30
- Never hit 20%: 25

**Why Age-Only for WRs?**
- Dominator Rating alone had weak correlation (r=0.175) with NFL success
- High dominator often came from weak competition (small schools, late-round busts)
- Breakout Age has stronger correlation (r=0.395) - younger breakouts predict NFL success
- Draft capital already prices in production context (teammates, transfers, competition)
- This avoids penalizing players like Jaylen Waddle (competed with 3 future 1st-rounders)

- RBs use yards per team pass attempt (measures receiving share of passing game)
- Younger players breaking out earlier signals higher upside
- Scaled 0-100 where 50 = average for that position

### 3. Athletic Modifier
```
athletic = normalize(speed_score(forty_time, weight))
```
- Uses Barnwell Speed Score: (weight × 200) / (40 time)^4
- Rewards players who are fast for their size (weight already factored into speed score)

## Decisions Made

1. **Component Weights**: 50% Draft Capital / 35% Breakout / 15% Athletic
   - Optimized via backtest on 2020-2024 draft classes (207 players)
   - Draft capital is the strongest predictor (50%), breakout adds value for finding sleepers (35%), athleticism is a modest tiebreaker (15%)
   - Spearman correlation: 0.51 | Top-24 hit rate: 56%

2. **Age Weight Function**: Moderate adjustment
   - Age 19: 1.20x (20% bonus)
   - Age 20: 1.10x (10% bonus)
   - Age 21: 1.00x (baseline)
   - Age 22: 0.90x (10% penalty)
   - Age 23: 0.80x (20% penalty)

3. **Speed Score Function**: Classic Speed Score (Barnwell formula)
   - Formula: (Weight × 200) / (40 time)^4
   - Industry standard, rewards players who are fast for their size

4. **Position Handling**: Position-Split Normalization
   - RBs use: Receiving yards ÷ Team pass attempts × age weight
   - WRs use: Breakout Age scoring (age when first hit 20%+ dominator)
   - RBs are normalized against other RBs only (50 = average RB)
   - WRs are normalized against other WRs only (50 = average WR)
   - This prevents RBs from being penalized for lower receiving yards than WRs
   - Outputs separate RB and WR rankings

5. **WR Breakout Age Methodology** (updated after backtest analysis)
   - Originally tested Dominator Rating but it had weak signal (r=0.175)
   - Dominator penalized players with elite teammates (Waddle at Alabama)
   - Breakout Age (when player first hit 20% dominator) has stronger signal (r=0.395)
   - Logistic regression coefficient is positive (+0.388), confirming predictive value
   - Draft capital already accounts for context (teammates, transfers, competition level)

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

For each prospect, we'll need:
- Name, position, school
- Draft pick (actual or projected)
- Receiving yards (or rushing for RBs)
- Team pass attempts (for RBs)
- Age (at time of production)
- Weight, 40-yard dash time

**For WRs specifically:**
- Breakout age (age when first hit 20%+ dominator rating)
- Requires multi-season college data to calculate

## Commands

```bash
# Calculate SLAP scores (outputs to output/ folder)
python src/calculate_slap.py

# Refresh data from APIs (birthdates, stats)
python src/fill_missing_ages.py

# Update WR breakout scores with age-only approach
python src/update_wr_breakout.py
```

## Output Files

- `output/slap_scores_rb.csv` - RB rankings
- `output/slap_scores_wr.csv` - WR rankings
- `output/slap_scores.csv` - Combined rankings
