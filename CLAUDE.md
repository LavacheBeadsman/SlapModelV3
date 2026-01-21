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

## Open Questions (Decide Together)

Before writing code, we need to determine:

1. **Component Weights**: How much should each factor matter?
   - Draft capital vs breakout vs athletic
   - Example: 50% draft, 30% breakout, 20% athletic?

2. **Age Weight Function**: How much to reward younger producers?
   - How much bonus for a 19-year-old vs 22-year-old?

3. **Speed Score Function**: How to combine 40 time and weight?
   - What's a good 40 for a 180lb WR vs 220lb RB?

4. **Position Differences**: Should RBs and WRs use different formulas?
   - Different weights? Different production metrics?

## Technical Preferences

- **Language**: Python
- **Data Storage**: CSV files (can be opened in Excel)
- **Visualizations**: Clear charts for content creation

## How to Work Together

1. **Check in before major decisions** - Don't assume, ask
2. **Explain code in simple terms** - What does it do and why
3. **Small steps** - Break work into pieces I can follow
4. **Options with tradeoffs** - When deciding something, give me choices with clear pros and cons

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

(To be added as we build the project)
