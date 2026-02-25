# SLAP V5 — Post-Combine & Post-Draft Update Guide

**Created:** February 25, 2026
**Model version:** SLAP V5 (locked weights, locked formulas)
**Current prospect count:** 217 (109 WR + 56 RB + 52 TE)

This is a step-by-step checklist for updating SLAP V5 scores after the NFL Combine and NFL Draft. Hand this to Claude Code in a future session and it can follow the steps.

---

## What's Currently Imputed (Pre-Combine/Pre-Draft)

All 217 prospects currently use **projected mock draft picks** instead of real draft picks. Draft capital is 60-70% of each position model, so this is the biggest source of uncertainty.

Additionally:

| Position | Component | Weight | Current State | How Many Affected |
|----------|-----------|--------|---------------|-------------------|
| **RB** | Speed Score | 5% | ALL 56 RBs use MNAR-imputed speed scores (no 2026 combine 40 times yet) | 56/56 |
| **TE** | RAS | 10% | ALL 52 TEs use MNAR-imputed RAS (47 at 40.0, 5 at 60.0 — placeholders based on projected round) | 52/52 |
| **WR** | No athletic component | 0% | Not affected by combine | 0 |

MNAR imputation logic: Projected Rd 1-2 prospects get the 60th percentile of backtest athletic scores. Rd 3+ get the 40th percentile. This is intentional — early-round prospects who skip workouts tend to be more athletic than late-round no-shows.

---

## UPDATE 1: POST-COMBINE (March/April 2026)

**When to run:** After the NFL Combine and major pro days are complete (typically late March / early April).

**What you're updating:** Real 40 times + weights for RBs, real RAS scores for TEs.

**Expected impact:** Small refinements, not revolutionary. RB Speed Score is only 5% of the model, TE RAS is 10%. A few players will move meaningfully if their athletic testing is surprisingly good or bad, but most scores won't change much.

### Checklist

- [ ] **Step 1: Collect RB combine data (40 times + weights)**

  Source: NFL Combine results, pro day results, or updated nflverse combine data.

  You need two values per RB: `weight` (lbs) and `forty` (40-yard dash in seconds).

  **File to update:** `data/nflverse/combine.parquet`

  Add 2026 RB rows with columns: `season`, `draft_year` (2026), `player_name`, `pos` (RB), `school`, `wt`, `forty`

  Alternatively, you can add `forty` directly to `data/prospects_final.csv` — but the build script currently reads 40 times from `combine.parquet`, so updating that file is the cleanest approach.

  **Current state of `data/prospects_final.csv` for RBs:**
  - 54 of 56 RBs already have `weight` populated
  - NO `forty` column exists in this file (40 times come from combine.parquet)
  - Columns: `player_name`, `position`, `school`, `projected_pick`, `rec_yards`, `team_pass_attempts`, `birthdate`, `age`, `age_estimated`, `weight`

- [ ] **Step 2: Collect TE RAS scores**

  Source: Kent Lee Platte's RAS database (mathbomb.com or @MathBomb on Twitter), or calculate from combine measurables.

  **File to update:** `data/te_2026_prospects_final.csv`

  Column to update: `ras_score`

  Current values are all placeholders (40.0 or 60.0). Replace with real RAS scores (0-10 scale, which the build script multiplies by 10 to get 0-100).

  **Important:** If a TE didn't participate in combine/pro day, KEEP the MNAR-imputed value. Don't zero it out — that would unfairly penalize them.

- [ ] **Step 3: Rebuild**

  ```bash
  python src/build_master_database_v5.py
  ```

- [ ] **Step 4: Spot-check changes**

  Compare before/after for a few players you know tested well or poorly. Make sure the direction makes sense (fast RB should get higher speed score, athletic TE should get higher RAS).

- [ ] **Step 5: Validate (optional)**

  ```bash
  python src/full_validation_8gm.py
  ```

  Backtest metrics should be **identical** — you only changed 2026 prospect data, not backtest data. If metrics changed, something went wrong.

### What NOT to touch in this update
- Draft picks (still using mock draft projections)
- WR data (no athletic component)
- Any backtest files
- Component weights or formulas

---

## UPDATE 2: POST-DRAFT (April/May 2026)

**When to run:** After the NFL Draft is complete (all 7 rounds + UDFA signings).

**What you're updating:** Replace ALL projected picks with REAL draft picks. This is the BIG update — draft capital is 60-70% of every position model.

**Expected impact:** Massive. Some players will rise dramatically (fell in the draft → lower pick number → higher DC than projected). Others will crater (reached in the draft or went undrafted). This is the most impactful single update you'll ever make to 2026 scores.

### Pre-Draft: Archive Current Scores

- [ ] **Step 0: Save pre-draft scores for comparison content**

  Before changing anything, copy the current output files so you can show "before vs after" on YouTube/Patreon.

  ```bash
  mkdir -p output/pre_draft_archive
  cp output/slap_v5_2026_all.csv output/pre_draft_archive/slap_v5_2026_all_PRE_DRAFT.csv
  cp output/slap_v5_wr_2026.csv output/pre_draft_archive/slap_v5_wr_2026_PRE_DRAFT.csv
  cp output/slap_v5_rb_2026.csv output/pre_draft_archive/slap_v5_rb_2026_PRE_DRAFT.csv
  cp output/slap_v5_te_2026.csv output/pre_draft_archive/slap_v5_te_2026_PRE_DRAFT.csv
  ```

  This gives you "here's how SLAP changed after we learned where they were actually drafted" content.

### Update Drafted Prospects

- [ ] **Step 1: Update WR + RB draft picks**

  **File:** `data/prospects_final.csv`

  **Column to change:** `projected_pick` → replace with actual draft pick number

  For EACH of the 165 WR/RB prospects:
  - If drafted: replace `projected_pick` with their actual overall pick number
  - If undrafted: see Step 3 below

  **Also update** `weight` if any new combine data became available since Update 1.

- [ ] **Step 2: Update TE draft picks**

  **File:** `data/te_2026_prospects_final.csv`

  **Column to change:** `projected_pick` → replace with actual draft pick number

  **Also update:**
  - `ras_score` (if you haven't already in Update 1, or if new pro day data came in)
  - `weight`, `height` (if updated)

- [ ] **Step 3: Update WR pre-calculated file**

  **File:** `output/slap_v5_wr_2026.csv`

  **Column to change:** `projected_pick` → replace with actual draft pick number

  **Do NOT change** these columns (they're locked college data):
  - `early_declare_score` — based on college seasons played, doesn't change
  - `teammate_score` — based on college teammates, doesn't change
  - `rush_yards` — college rushing stats, doesn't change
  - `breakout_age` — college breakout timing, doesn't change
  - `peak_dominator` — college dominator rating, doesn't change

- [ ] **Step 4: Update RB combine data (if not done in Update 1)**

  **File:** `data/nflverse/combine.parquet`

  Add 2026 combine rows with real `wt` (weight) and `forty` (40-yard dash) for RBs.

  The build script uses this file to calculate Speed Score = `(weight × 200) / (forty^4)`.

  If combine.parquet isn't updated, the build script falls back to MNAR imputation (which is now less accurate since you know the real pick numbers).

- [ ] **Step 5: Handle undrafted prospects (UDFAs)**

  Players in your prospect list who went undrafted need a decision:

  **Option A: Remove them** — Delete their row from the input CSV. They won't appear in output.

  **Option B: Flag as UDFA with a high pick number** — Set their pick to something like 260-280 (after the last real pick). This gives them a very low DC score, which is realistic — UDFAs rarely hit in fantasy. The DC formula `100 - 2.40 × (pick^0.62 - 1)` gives:
  - Pick 260 → DC score ≈ 4.1
  - Pick 280 → DC score ≈ 2.3

  **Recommendation:** Option B is better for content. You can show "here's what SLAP thinks of the undrafted guys" and their prospect_profile still shows if they have good college production despite going undrafted.

- [ ] **Step 6: Add newly drafted prospects NOT in current list**

  Some players drafted in 2026 won't be in your current 165 WR/RB or 52 TE prospect files. You need to add them.

  **For new WRs:**
  1. Add a row to `data/prospects_final.csv` with: `player_name`, `position` (WR), `school`, `projected_pick` (actual pick), `rec_yards`, `team_pass_attempts`, `birthdate`, `age`, `age_estimated`, `weight`
  2. Run `python src/calculate_wr_breakout_age.py` to get their breakout age from CFBD (or manually add to `data/wr_breakout_ages_2026.csv`)
  3. Add a row to `output/slap_v5_wr_2026.csv` with: `player_name`, `position`, `college`, `draft_year` (2026), `projected_pick` (actual pick), `early_declare_score` (100 if ≤3 college seasons, else 0), `teammate_score` (calculate or set to 0 if unsure), `rush_yards` (from CFBD), `breakout_age`, `peak_dominator`

  **For new RBs:**
  1. Add a row to `data/prospects_final.csv` with: `player_name`, `position` (RB), `school`, `projected_pick` (actual pick), `rec_yards` (final college season), `team_pass_attempts` (final college season), `birthdate`, `age`, `weight`
  2. If combine data exists, add to `data/nflverse/combine.parquet`

  **For new TEs:**
  1. Add a row to `data/te_2026_prospects_final.csv` with all required columns
  2. Key data needed: `breakout_age`, `peak_dominator`, `cfbd_rec_yards`, `cfbd_team_pass_att`, `ras_score`
  3. Run `python src/calculate_te_slap_2026.py` to verify their scores

  **Data sources for new prospects:**
  - CFBD API for college stats (rec_yards, team_pass_att, rush_yards, breakout data)
  - Combine.parquet or pro day results for athletic testing
  - Kent Lee Platte / mathbomb.com for TE RAS scores

- [ ] **Step 7: Rebuild**

  ```bash
  python src/build_master_database_v5.py
  ```

- [ ] **Step 8: Validate**

  ```bash
  python src/full_validation_8gm.py
  ```

  **Backtest metrics MUST be identical to pre-draft.** You only changed 2026 prospect data. If any backtest metric moved, something went wrong — you accidentally touched a backtest file.

- [ ] **Step 9: Generate comparison content**

  Compare post-draft scores to the archived pre-draft scores:

  ```python
  import pandas as pd
  pre = pd.read_csv('output/pre_draft_archive/slap_v5_2026_all_PRE_DRAFT.csv')
  post = pd.read_csv('output/slap_v5_2026_all.csv')
  merged = pre.merge(post, on=['player_name', 'position'], suffixes=('_pre', '_post'))
  merged['slap_delta'] = merged['slap_display_score_post'] - merged['slap_display_score_pre']
  merged['dc_delta'] = merged['dc_score_post'] - merged['dc_score_pre']

  # Biggest risers
  print(merged.sort_values('slap_delta', ascending=False).head(20)[
      ['player_name', 'position', 'pick_pre', 'pick_post', 'slap_display_score_pre', 'slap_display_score_post', 'slap_delta']
  ])

  # Biggest fallers
  print(merged.sort_values('slap_delta', ascending=True).head(20)[
      ['player_name', 'position', 'pick_pre', 'pick_post', 'slap_display_score_pre', 'slap_display_score_post', 'slap_delta']
  ])
  ```

  Content angles:
  - "Biggest SLAP risers after the draft" — players who fell and now look like steals
  - "Biggest SLAP fallers after the draft" — players who were reached and the model is skeptical
  - "High profile vs low DC" — first-rounders with weak prospect profiles
  - "Late-round gems" — day 3 picks with strong prospect profiles

---

## Quick Reference: File Map

### Input Files (you edit these)

| File | What to Update | When |
|------|----------------|------|
| `data/prospects_final.csv` | WR/RB: `projected_pick` → real pick, `weight` | Post-draft |
| `data/te_2026_prospects_final.csv` | TE: `projected_pick` → real pick, `ras_score` | Post-combine + post-draft |
| `output/slap_v5_wr_2026.csv` | WR: `projected_pick` → real pick (pre-calc file) | Post-draft |
| `data/nflverse/combine.parquet` | RB: add 2026 rows with `wt` + `forty` | Post-combine |
| `data/wr_breakout_ages_2026.csv` | WR: add new prospects' breakout data | Post-draft (new WRs only) |

### Output Files (the build script creates these)

| File | Contents |
|------|----------|
| `output/slap_v5_master_database.csv` | All 939+ players (backtest + 2026) |
| `output/slap_v5_2026_all.csv` | All 2026 prospects ranked within position |
| `output/slap_v5_wr_2026.csv` | WR 2026 prospects (also an input — has pre-calced scores) |
| `output/slap_v5_rb_2026.csv` | RB 2026 prospects |
| `output/slap_v5_te_2026.csv` | TE 2026 prospects |

### Scripts

| Script | Purpose | When to Run |
|--------|---------|-------------|
| `src/build_master_database_v5.py` | **Master rebuild** — the only script you NEED | Every update |
| `src/full_validation_8gm.py` | Full validation suite (confirm backtest unchanged) | Every update |
| `src/calculate_wr_breakout_age.py` | Fetch WR breakout ages from CFBD | Adding new WR prospects |
| `src/calculate_te_slap_2026.py` | TE 2026 standalone calculator | Adding new TE prospects |
| `src/fetch_rb_receiving_stats.py` | Fetch RB receiving from CFBD | Adding new RB prospects |
| `src/recalculate_slap_v5.py` | V5 vs V4 comparison (optional) | If you want comparison stats |

---

## What NEVER Changes

These are locked. Do not modify them for any update:

- **Backtest data files** — `data/wr_backtest_all_components.csv`, `data/rb_backtest_with_receiving.csv`, `data/te_backtest_master.csv`, `data/backtest_outcomes_complete.csv`
- **Component weights** — WR 70/20/5/5, RB 65/30/5, TE 60/15/15/10
- **Scoring formulas** — DC curve, breakout age tiers, production scaling, speed score formula
- **Backtest validation metrics** — These should be identical before and after any 2026 update. If they change, you touched something you shouldn't have.
