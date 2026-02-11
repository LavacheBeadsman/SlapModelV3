# PROJECT_CONTEXT.md

**Last updated:** February 11, 2026

## Timeline & Data Availability

- **College football data available:** 2015–2025 seasons (complete)
- **NFL data available:** 2015–2025 seasons (complete)
- **2026 draft class:** Current prospects — no NFL data yet
- **2025 draft class:** One NFL season completed (2025)
- **Season-to-draft mapping:** YYYY draft class uses YYYY-1 college season data

## Key Files

- **Main SLAP calculation script:** `src/recalculate_all_slap_new_dc.py`
- **Master output:** `output/slap_complete_database_v4.csv`

## Model Weights

| Position | Draft Capital | Production Component | Athletic Component |
|----------|--------------|---------------------|--------------------|
| **WR**   | 65% DC       | 20% Breakout Age    | 15% RAS            |
| **RB**   | 50% DC       | 35% Receiving Production | 15% RAS       |
| **TE**   | TBD          | TBD                 | TBD — not yet built |
