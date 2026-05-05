# SLAP V5 Methodology

**SLAP** = **S**tatistical **L**ikelihood of **A**chieving **P**roduction

A draft-capital-anchored prospect model that rates NFL draft RBs, WRs, and
TEs on a 1–99 scale. Backtested against 750 drafted players (2015–2025) and
applied to 217 prospects in the 2026 class.

**Generated:** 2026-04-29 · **Model version:** V5.0

---

## What the score means

Every drafted player has two scores:

- **`slap_display_score`** (1–99) — the publication-facing number. Higher = better fantasy outlook.
- **`slap_model_score`** (~22–98) — the raw weighted composite. Used internally for validation.

The display score is a **percentile rank vs the historical backtest** within position. So:
- A 99 means "top of the historical distribution at this position"
- A 50 means "median of historical players at this position"
- A 1 means "bottom of the historical distribution"

This means **WR 95 and RB 95 are NOT directly comparable** — both mean "elite for that position relative to history." Cross-position comparisons should use the raw model score.

---

## Position formulas

Each formula is a weighted sum of components on a native 0–100 scale.

### WR
```
SLAP = DC × 0.70 + Enhanced_Breakout × 0.20 + Teammate × 0.05 + Early_Declare × 0.05
```

### RB
```
SLAP = DC × 0.65 + RYPTPA × 0.30 + Speed_Score × 0.05
```

### TE
```
SLAP = DC × 0.60 + Breakout × 0.15 + Production × 0.15 + RAS × 0.10
```

Draft capital anchors all three formulas because, empirically, what NFL teams pay
for a player at the draft is the strongest single predictor of fantasy success.
The other components add refinement.

---

## Component definitions

### Draft Capital (DC)
**Formula:** `DC = 100 − 2.40 × (pick^0.62 − 1)`

| Pick | DC |
|------|----|
| 1 | 99 |
| 16 | 89 |
| 32 | 81 |
| 64 | 70 |
| 100 | 60 |
| 200 | 39 |
| 250 | 25 |
| UDFA (258) | 23 |

The exponential decay matches how NFL outcomes scale with draft pick — the gap
between pick 5 and pick 20 matters more than between pick 105 and pick 120.

### Enhanced Breakout (WR / TE)
**Source:** Multi-season college receiving data from cfbfastR play-by-play.

A player "breaks out" when their season **dominator rating** (share of team
receiving production) crosses a threshold. WR threshold = 20%, TE threshold = 15%.

The breakout score is determined by their age in their first breakout season:

| Breakout age | Score |
|--------------|-------|
| 18 (true freshman) | 100 |
| 19 (sophomore) | 90 |
| 20 (junior) | 75 |
| 21 (senior) | 60 |
| 22 (5th-year) | 45 |
| 23+ | 30 |

For players who never hit the threshold, the score falls back to
`min(35, 15 + peak_dominator)`.

### RYPTPA (RB)
**Formula:** `RYPTPA = (rec_yards / team_pass_att) × age_weight × 100 / 1.75`

Receiving yards per team pass attempt — captures how involved the RB was in
the team's passing game. Higher = more receiving role at NFL level.

`age_weight` rewards production at younger ages (max 1.15 at age 19, min 0.85 at age 24).

### Speed Score (RB)
**Formula:** `(weight × 200) / (forty^4)`, normalized to 0–100.

Combines body size and 40-yard dash. High speed score = NFL-caliber
combine athleticism. Missing 40 times are imputed via a structured rule:
- Round 1–2 prospects who skipped the combine → assigned 60th-percentile (elite picks rarely have weak speed)
- Round 3+ prospects who skipped → 40th percentile

### Production (TE)
**Formula:** `(rec_yards / team_pass_att) × age_weight × 100`, then min-max normalized to 0–99.9.

Same idea as RYPTPA but min-max scaled across the TE position. Falls back to
PFF data (`pff_yards / pff_pass_plays`) when CFBD is missing.

### RAS (TE)
**Source:** Relative Athletic Score (Kent Lee Platte) × 10 → 0–100 scale.

Composite of combine measurables. Missing values imputed by draft round.

### Teammate Score (WR)
Binary 0/100. Set to 100 when the player's team had **another high-DC WR or TE
drafted in the same class**. Counter-intuitive finding: WRs who shared targets
with another good prospect tend to be **better** NFL fantasy producers.

### Early Declare (WR)
Binary 0/100. Set to 100 when the player **played 3 or fewer college seasons**.
Age is irrelevant — what matters is whether they jumped to the NFL early
relative to their college eligibility.

---

## Validation results (backtest 2015–2025, n=750)

We measure how well SLAP predicts NFL fantasy outcomes using **Spearman rank
correlation** (rank-based, robust to outliers) with **bootstrap 95% CIs**.

### Headline correlations vs Career PPG

| Position | n | r (95% CI) | DC alone (for context) |
|----------|---|------------|------------------------|
| WR | 243 | **+0.50** [+0.39, +0.58] | +0.48 [+0.37, +0.56] |
| RB | 156 | **+0.59** [+0.46, +0.69] | +0.55 [+0.41, +0.66] |
| TE | 114 | **+0.60** [+0.46, +0.72] | +0.57 [+0.43, +0.70] |

### Hit rate by SLAP tier (top-24 fantasy season)

| Tier | WR (n=357) | RB (n=233) | TE (n=160) |
|------|-----------|-----------|-----------|
| Elite (80+) | **44.3%** | **65.2%** | **50.0%** |
| Good (60–80) | 17.8% | 34.0% | 12.5% |
| Avg (40–60) | 8.2% | 12.5% | 6.1% |
| Poor (20–40) | 1.4% | 10.6% | 0.0% |
| Bottom (<20) | 0.0% | 4.4% | 0.0% |

Clean monotonic relationship at every position. **An "Elite" WR is ~44× more
likely to be fantasy-relevant than a "Bottom" WR.**

### Stability check

We ran 5-fold cross-validation (each fold held out once) and 5 different random
80/20 splits. Average correlations across folds matched full-sample within
0.05–0.06, indicating the score generalizes — it's not just memorizing the
training data.

### Honest caveat

The V5 weights were tuned on the full backtest, so this validates **prediction
stability**, not true out-of-sample weight generalization. Re-tuning the
weights inside each fold would require the original fitting code. This is
disclosed; it's the standard limitation of any model that publishes its
training-time correlations.

---

## Data sources

| Source | Used for |
|--------|----------|
| **NFLverse** `draft_picks.parquet` | Draft year, pick, round |
| **NFLverse** `player_season_stats_*.csv` | NFL fantasy outcomes (PPG, hit rates) |
| **NFLverse** `combine.parquet` | Weight, 40 time for Speed Score |
| **CFBD API** | Final-season college receiving (rec_yards, team_pass_att) |
| **cfbfastR** play-by-play | Multi-season dominator + breakout age |
| **PFF** | TE production fallback when CFBD missing |
| **RAS** (Kent Lee Platte) | TE athletic composite |

---

## Data quality flags

Each row in the master database has a `data_quality_flag`:

| Flag | Count | Meaning |
|------|-------|---------|
| `complete` | 882 (91%) | All scoring components populated from real source data |
| `partial_data` | 84 (9%) | At least one receiving/dominator field is NaN |
| `outlier_review` | 1 | Known data quirk requiring context (currently: Eli Heidenreich, Navy triple-option offense) |

Two boolean columns expose imputation:
- `peak_dominator_imputed` — currently always False (no fabrication)
- `broke_out` — `True` (hit threshold) / `False` (verified non-breakout) / `NaN` (RB or unknown)

**No imputation in the published CSV.** When data is missing, it is shown as
NaN. The model's internal fallback formulas still produce a valid SLAP score,
but the underlying input is honestly blank.

---

## Known limitations

1. **D2/D3/Ivy schools**: Players from schools not covered by cfbfastR PBP
   (e.g., John Carroll, Princeton, Drake, Ashland) have NaN peak_dominator
   and ryptpa. Their SLAP is computed via fallback formulas. Affects ~15
   players in the 750-row backtest and 3 in the 2026 class.

2. **2014 cfbfastR PBP gaps**: Some major-school players from the 2015 draft
   class (Todd Gurley at Georgia, Duke Johnson at Miami) are missing from
   cfbfastR's 2014 PBP archive despite being top draftees. This is upstream
   data; we leave their fields NaN rather than estimate.

3. **Triple-option / system distortions**: RYPTPA's denominator (team pass
   attempts) is sensitive to offensive scheme. Players from triple-option
   programs (Navy, Army, Air Force) score artificially high on RYPTPA. The
   only such case in the 2026 class — Eli Heidenreich (Navy → drafted as RB
   pick 230) — is flagged via `outlier_review`. **Future model versions should
   replace RYPTPA with a yardage-share metric for system-independence.**

4. **2025 class sample is one NFL season**: Hit rates for 2025 draftees are
   based on rookie-year only. Career outcomes will mature over 2026–2027.
   Don't over-interpret 2025 hit rates as final.

5. **WR model marginally beats DC alone**: For WR, SLAP V5 outperforms a
   draft-capital-only baseline by ~0.02 r. Translation: NFL teams' WR
   evaluations are already strong, and the college signals add modest
   refinement, not transformative new information. RB and TE see larger
   model-vs-DC gaps.

6. **Position converts**: Players whose NFL position differs from their
   college role (e.g., college WR drafted as RB) score on the formula matching
   their drafted position, using their college-position stats. This works
   imperfectly for converts — see Heidenreich. We flag known cases.

---

## Output columns reference

| Column | Type | Definition |
|--------|------|------------|
| `slap_display_score` | float | 1–99 percentile rank vs backtest |
| `slap_model_score` | float | Native-scale weighted composite |
| `dc_score` | float | Draft Capital component (0–99) |
| `prospect_profile` | float | Non-DC components averaged (0–100) |
| `enhanced_breakout` | float | WR Enhanced_Breakout (native 0–99.9) |
| `teammate_score` | float | WR Teammate flag (0 or 100) |
| `early_declare_score` | float | WR Early Declare flag (0 or 100) |
| `production_score` | float | RB RYPTPA scaled (0–99.9) |
| `speed_score` | float | RB Speed Score (0–100) |
| `te_breakout_score` | float | TE Breakout component |
| `te_production_score` | float | TE Production component |
| `ras_score` | float | TE RAS × 10 |
| `breakout_age` | float | Age at first breakout season (NaN if never) |
| `broke_out` | bool | True/False/NaN — explicit breakout flag |
| `peak_dominator` | float | Highest single-season dominator % |
| `rec_yards` | float | Final college season receiving yards |
| `team_pass_att` | float | Final college season team pass attempts |
| `ryptpa` | float | rec_yards / team_pass_att |
| `data_quality_flag` | str | complete / partial_data / outlier_review |
| `outlier_note` | str | Free-text note for outlier_review rows |
| `model_version` | str | V5.0 |
| `data_as_of_date` | str | ISO date when this CSV was built |

---

## Reproducibility

The full pipeline runs in one command:

```bash
python src/build_master_database_v5.py
```

It reads source CSVs from `data/`, applies the formulas above, and writes
five output files to `output/`:

- `slap_v5_master_database.csv` — all 967 rows, all positions, all components
- `slap_v5_2026_all.csv` — 217 prospects only, ranked within position
- `slap_v5_wr.csv` / `slap_v5_rb.csv` / `slap_v5_te.csv` — per-position files

Backtest scores are deterministic (same inputs → same outputs). 2026 prospect
scores update when draft picks (`projected_pick`) change.

---

## Honest framing

SLAP V5 is **a model, not an oracle.** It does well at separating the top
tier from the bottom tier (Elite vs Bottom hit rates differ by 40+ percentage
points at every position). It does less well at differentiating within the
middle tiers, where most fantasy decisions actually happen.

If you're using SLAP for fantasy decisions:
- **Trust the elite tier as a screening tool** — top scorers are 3–5× more
  likely to hit than middle-tier players
- **Don't read fine-grained differences as gospel** — the bootstrap CIs are
  ~±0.10 at the population level, and individual rankings can swing
- **Read the outlier_note column** — it's there because some scores need context
- **Recognize the limits** — D2/D3 prospects, triple-option converts, and
  position-change cases may not score representatively

This document and the underlying code are open. Audit, replicate, critique.
