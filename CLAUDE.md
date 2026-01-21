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
```
prod = rec_yards / team_pass_attempts
prod_age_adj = prod * age_weight(age)
breakout = normalize(prod_age_adj)
```
- Measures production relative to team opportunity
- Rewards younger players who produce
- Scaled 0-100 where 50 = average

### 3. Athletic Modifier
```
athletic = normalize(weight + speed_score(forty_time, weight))
```
- Combines size and speed
- Heavier players with fast times score higher

## Decisions Made

1. **Component Weights**: 45% Draft Capital / 35% Breakout / 20% Athletic
   - "Balanced Approach" - all three factors matter
   - Draft capital sets the floor, production confirms ability, athleticism raises ceiling

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
   - Both use: Receiving yards ÷ Team pass attempts
   - RBs are normalized against other RBs only (50 = average RB)
   - WRs are normalized against other WRs only (50 = average WR)
   - This prevents RBs from being penalized for lower receiving yards than WRs
   - Outputs separate RB and WR rankings

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
- Team pass attempts
- Age (at time of production)
- Weight, 40-yard dash time

## Commands

```bash
# Calculate SLAP scores (outputs to output/ folder)
python src/calculate_slap.py

# Refresh data from APIs (birthdates, stats)
python src/fill_missing_ages.py
```

## Output Files

- `output/slap_scores_rb.csv` - RB rankings
- `output/slap_scores_wr.csv` - WR rankings
- `output/slap_scores.csv` - Combined rankings
