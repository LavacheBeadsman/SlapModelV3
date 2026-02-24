# SLAP Score V5 — The Definitive Model Guide

*Last updated: February 2026*

---

## 1. WHAT IS SLAP SCORE?

**SLAP Score** is a data-driven prospect model that rates NFL Draft running backs, wide receivers, and tight ends on a 0-99 scale. It stands for **Statistical Likelihood of Achieving Production** — and that name says exactly what it does. It takes what we know about a player before they play a single NFL snap (where they were drafted, what they did in college, and a handful of other measurable traits) and turns it into a single number that estimates how likely they are to become a fantasy-relevant NFL player.

Every SLAP Score comes with two outputs:

1. **The Score (0-99):** Higher is better. A score of 85 means the model sees a strong profile. A score of 42 means the historical odds are against that player.
2. **The Delta vs. Draft Capital:** This is where it gets interesting. The delta shows you when the model disagrees with where a player was drafted. A positive delta means the model likes the player *more* than their draft slot suggests — it sees something extra in their college profile. A negative delta is a red flag — draft pedigree says one thing, but the data says another.

---

## 2. HOW WAS IT BUILT?

SLAP Score was not built on hunches, eye tests, or gut feelings. Every component was tested against real NFL outcomes using 10 years of data (2015-2024 draft classes), and every claim in this document is backed by statistical evidence.

Here's the development process:

### Step 1: Start With Draft Capital

The foundation is draft capital — where a player was picked in the NFL Draft. This is the single strongest predictor of NFL fantasy success at every position, and it's not close. NFL teams spend millions of dollars on scouting, and their draft decisions contain an enormous amount of information. Any model that ignores draft capital is leaving the most important variable on the table.

SLAP uses a "gentler curve" formula that converts a draft pick number into a 0-100 score:

```
DC = 100 - 2.40 x (pick^0.62 - 1)
```

Pick 1 = 100. Pick 10 = 92. Pick 32 = 82. Pick 100 = 61. Pick 200 = 38. The curve is steep enough to reward early picks but gentle enough to avoid absurd cliffs between adjacent picks.

### Step 2: Test Dozens of College Metrics

With draft capital as the baseline, dozens of college metrics were tested to see which ones add **independent** signal beyond what the draft slot already tells you. This is the critical distinction — a metric that correlates with NFL success isn't useful if draft capital already captures that information.

Metrics tested include:
- College rushing stats (yards, YPC, touchdowns, yards after contact)
- College receiving stats (yards, receptions, targets, target share)
- Dominator Rating (share of team receiving production)
- Breakout Age (when a player first dominated in college)
- Combine results (40-yard dash, RAS, Speed Score, bench press, vertical)
- PFF college grades (22 sub-metrics tested for RBs alone)
- Player size and weight
- Early declaration status
- Teammate draft capital
- Age at draft

### Step 3: Use Partial Correlations to Find Independent Signal

The key statistical tool: **partial correlation**. This measures whether a metric predicts NFL outcomes *after controlling for draft capital*. A metric can have a strong raw correlation with NFL success but still add nothing if draft capital already captures that same information.

Example: A player's 40-yard dash time correlates with NFL success — but NFL teams already see the 40 time before they draft the player, so the draft slot already "prices in" the athletic testing. After controlling for draft capital, athletic metrics add almost nothing. This finding held at every position tested.

### Step 4: Optimize Weights Using Multiple Outcomes

The model doesn't optimize for a single outcome. It uses a **priority-weighted objective** across four NFL outcomes:
- 40% weight: First 3 years PPG (career trajectory)
- 25% weight: Hit24 (ever finished as a top-24 starter)
- 20% weight: Hit12 (ever finished as a top-12 elite player)
- 15% weight: Career PPG (long-term value)

Weight configurations were tested systematically — not one or two, but 30+ combinations per position — to find the weights that maximize predictive power across all four outcomes simultaneously.

### Step 5: Validate Rigorously

The final models were validated using:
- **Bootstrap resampling** (1,000 iterations): Does the model beat DC-only consistently, or is it a fluke of the specific sample?
- **AUC-ROC**: Can the model separate hits from misses?
- **Brier scores**: Are the probability estimates well-calibrated?
- **Top-decile precision**: When the model says a player is elite, how often is it right?
- **Disagreement analysis**: When the model disagrees with draft capital, who's right?
- **Year-by-year stability**: Does it work across different draft classes, or only in certain years?

The bottom line: every component earned its place through statistical testing, not guesswork.

---

## 3. THE THREE MODELS

SLAP V5 runs a separate, position-specific model for wide receivers, running backs, and tight ends. Each position has different components because different things predict success at different positions.

---

### Wide Receivers: 70/20/5/5

```
WR SLAP = Draft Capital x 0.70
        + Enhanced Breakout x 0.20
        + Teammate Score x 0.05
        + Early Declare x 0.05
```

**Component 1: Draft Capital (70%)**
Where the player was drafted, converted to a 0-100 score. This is the backbone of the model. Among WRs drafted from 2015-2024, Round 1 WRs hit at 55%, Round 2 at 33%, Round 3 at 16%, Round 4 at 7%, and Round 5+ at under 3%.

**Component 2: Enhanced Breakout (20%)**
This answers the question: *How young was this player when he first dominated in college?*

Breakout Age is the age when a WR first hit a 20% Dominator Rating (meaning he accounted for at least 20% of his team's receiving yards). Younger breakouts predict NFL success far better than raw production numbers.

The scoring works in tiers based on the age of first breakout, with a dominator magnitude bonus as a tiebreaker within each tier:
- Age 18 breakout: base score 100
- Age 19: base 90
- Age 20: base 75
- Age 21: base 60
- Age 22: base 45
- Age 23: base 30
- Never hit 20%: 15-35 (based on peak dominator)

On top of this, WRs who had 20+ college rushing yards get a small +5 bonus. This signals versatility — jet sweeps, gadget plays, coaches trusting the player in multiple roles. It's a tiebreaker, not a game-changer.

**Component 3: Teammate Score (5%)**
A binary flag: Did this player produce *despite sharing targets with other highly-drafted receivers?*

If the total draft capital of a player's WR/TE teammates (drafted the same year) exceeds a threshold of 150, the player scores 100. Otherwise, 0.

This is counterintuitive — you might expect that sharing targets would *hurt* a player's value. But the opposite is true. WRs who produced in elite rooms with other future NFL receivers showed *better* NFL outcomes, not worse. Think Jaylen Waddle producing at Alabama alongside DeVonta Smith, or Chris Olave at Ohio State with Garrett Wilson. These players earned targets against elite competition. At only 5% weight, this is a small tiebreaker for the toughest player evaluations.

**Component 4: Early Declare (5%)**
Another binary flag: Did the player leave college early?

Players who declared before their senior season score 100. Players who stayed their full eligibility score 0. Early declaration signals NFL-readiness, confidence in draft stock, and youth at the time of drafting. At 5% weight, it's a small differentiator.

**What Was Tested and Rejected for WRs:**
- **RAS (Relative Athletic Score)**: Used in V4 at 15% weight. Extensively tested and **removed** in V5. RAS does not add predictive value for WRs after controlling for draft capital. NFL teams already see combine results before drafting — the draft slot prices in athleticism. RAS also had severe missing-data problems: elite prospects (like CeeDee Lamb and Jaylen Waddle) often skip workouts, creating a biased sample where the best players had no data. Removing RAS and adding Teammate + Early Declare improved every metric tested.
- **Dominator Rating (alone)**: Weak correlation with NFL success (r=0.175). High dominator often came from weak competition at small schools, producing late-round busts. Breakout Age (r=0.395) is far more predictive.
- **Raw college production stats**: Yards, receptions, touchdowns — all largely captured by draft capital already.
- **Height, weight, 40-yard dash, Speed Score, HaSS**: All tested individually and in combinations. None add independent signal after DC. Bootstrap testing showed athletic metrics receive 0% optimal weight in 99% of resamples.

**The WR Key Insight:** Youth matters more than volume. *When* a WR dominated in college is more predictive than *how much* he produced. And athletic testing? Already baked into the draft pick.

---

### Running Backs: 65/30/5

```
RB SLAP = Draft Capital x 0.65
        + Receiving Production (RYPTPA) x 0.30
        + Speed Score x 0.05
```

**Component 1: Draft Capital (65%)**
Same formula as WRs. Draft capital is the dominant predictor. Round 1 RBs hit at over 90%, Round 2 at 56%, and the rate drops sharply from there.

**Component 2: Receiving Production — RYPTPA (30%)**
This is the metric that separates SLAP from draft-capital-only rankings for RBs. RYPTPA stands for **Receiving Yards Per Team Pass Attempt** — it measures how much of the passing game the running back captured in his final college season.

The formula:
```
RYPTPA = (receiving yards / team pass attempts) x age weight x 100 / 1.75
```

The age weight gives a bonus to younger producers (a 19-year-old catching passes gets a 15% boost; a 23-year-old gets a 5% penalty). The division by 1.75 scales the score to a 0-99 range.

Why receiving production? Because it was statistically validated. In the backtest (188 RBs, 2015-2024), receiving production had a partial correlation of r=0.30 with NFL outcomes after controlling for draft capital, with a p-value of 0.004. That means there's less than a 1-in-200 chance this relationship is random noise.

**Critical rule:** SLAP always uses the player's **final college season** (the year before the draft), not their "best" season. Using "best season" cherry-picks data and inflates scores — 19 RBs had their scores corrected downward by an average of 12.6 production points when this fix was applied.

**Component 3: Speed Score (5%)**
The Bill Barnwell Speed Score formula: `(weight x 200) / (40 time)^4`. This rewards players who are fast for their size. A 225-pound back who runs a 4.40 scores higher than a 195-pound back with the same 40 time.

Speed Score is normalized 0-100 within the position. For players missing combine data, the model uses MNAR-aware imputation: Round 1-2 players with missing data get the 60th percentile (elite prospects skip workouts by choice), while Round 3+ missing players get the 40th percentile (they likely weren't invited).

At only 5% weight, Speed Score doesn't materially change rankings. It was included as a tiebreaker that adds athletic diversity without hurting predictive accuracy.

**What Was Tested and Rejected for RBs:**
- **Every rushing metric**: Rushing yards, yards per carry, rushing touchdowns, yards after contact, broken tackles — **none of them predict NFL fantasy success** after controlling for draft capital. Zero. This is the single most surprising finding in the entire model. Twenty-two PFF sub-metrics were tested; RYPTPA ranked #1 or #2 on every single NFL outcome.
- **Breakout Age**: Works for WRs but fails for RBs. Partial correlation r=0.10, p=0.80 — no signal whatsoever.
- **RAS at 15%**: Used in V4. Replaced by Speed Score in V5 because RAS is a black-box calculation with worse data coverage. The 65/35 DC/Production model without any athletic component beat every 3-component model that included RAS.
- **Early Declare**: Tested, marginal value for RBs. Not included.
- **Weight alone, 40 time alone, BMI, size metrics**: All tested, none help.

**The RB Key Insight:** No college rushing stat predicts NFL fantasy success. What *does* predict it? Whether the running back caught passes. RBs who were involved in the passing game in college translate to fantasy-relevant roles in the NFL — and that signal holds even after accounting for draft capital. The receiving production metric literally doubles hit rates in the middle rounds.

---

### Tight Ends: 60/15/15/10

```
TE SLAP = Draft Capital x 0.60
        + Breakout Score x 0.15
        + Production Score x 0.15
        + Early Declare x 0.10
```

**Component 1: Draft Capital (60%)**
Same gentler-curve formula. TE draft capital is highly predictive: Round 1 TEs hit at 62%, while Round 5+ TEs hit at under 9%.

**Component 2: Breakout Score (15%)**
Similar concept to the WR breakout, but with a critical difference: the dominator threshold is **15%**, not 20%. TEs don't dominate target shares the way WRs do — a TE capturing 15% of his team's receiving yards is doing something notable. The age-tier scoring is the same (younger breakouts score higher), with a dominator-magnitude bonus as a tiebreaker.

**Component 3: Production Score (15%)**
Receptions per team pass attempt — a catch-volume metric. This is different from what the RB model uses (receiving yards). For TEs, **catch volume matters more than yards per catch**. A TE who catches 60 balls for 700 yards projects better than a TE who catches 30 balls for 700 yards. The yards-per-reception component adds noise; pure catch frequency is the cleaner signal.

Data comes from CFBD (primary) with PFF receptions-per-game as a fallback.

**Component 4: Early Declare (10%)**
Same binary flag as WRs, but at double the weight (10% vs 5%). TEs who declare early carry a stronger youth-at-draft signal because the TE position has a longer NFL development curve — youth matters more when the learning curve is steeper.

**What Was Tested and Rejected for TEs:**
- **RAS**: Tested at 5% weight. Modest signal but 7% of TEs have missing data, and the benefit was marginal. Ultimately deprioritized in favor of cleaner components.
- **20% dominator threshold**: Too restrictive for TEs. Fewer TEs reach 20% dominator, and the signal was weaker than at 15%.
- **10% and 12% thresholds**: Also tested. 15% was the sweet spot.
- **Receiving yards per team pass attempt (RYPTPA)**: The RB metric. Tested for TEs but outperformed by catch-based metrics (Rec/TPA). Yards per reception adds noise for TEs.
- **PFF college grades**: Tested but didn't add independent value after controlling for DC and production.
- **Thirty-plus weight configurations**: Systematically tested in grid search + optimization.

**The TE Key Insight:** Catch volume beats yards-per-catch. A TE who earns a high volume of receptions relative to his team's passing attempts is the one who projects best — it signals that coaches trusted him as a primary target, which translates directly to NFL target share. And the breakout threshold is 15%, not 20% like WRs, because TEs compete for targets differently.

---

## 4. WHAT WE LEARNED (Key Findings)

Building SLAP Score across three positions produced several findings that challenge conventional dynasty and devy wisdom:

### Draft Capital Is King
At every position, draft capital accounts for 60-70% of the model. This isn't because draft capital is a perfect predictor — it isn't — but because it's by far the best single predictor available. NFL teams aggregate tape, testing, interviews, medicals, and scheme fit into a single decision (the pick). That decision contains more predictive information than any individual college metric.

### Athletic Testing Adds Almost Nothing After Controlling for Draft Capital
This is perhaps the most counterintuitive finding. RAS, Speed Score, 40-yard dash, height-weight-speed composites — they all correlate with NFL success in isolation. But once you control for draft capital, the added signal is near zero. Why? Because NFL teams already see combine results before they draft. The athletic information is already "priced in" to the pick. This was tested exhaustively at all three positions with consistent results: bootstrap analysis showed athletic metrics earn 0% optimal weight in the vast majority of resamples.

### Zero Rushing Metrics Predict RB Fantasy Success
Out of 22 PFF sub-metrics tested for running backs — including rushing yards, YPC, yards after contact, broken tackles, and every rushing grade available — not a single one adds predictive value after draft capital. The metric that *does* work? Receiving production. RBs who caught passes in college project as better fantasy assets than RBs who didn't, even at the same draft slot.

### RB Receiving Production Doubles Hit Rates in the Middle Rounds
The RYPTPA metric's biggest impact is in Rounds 2-4, where draft capital alone leaves the most uncertainty. A Round 3 RB with elite receiving production hits at roughly double the rate of a Round 3 RB without it. The metric is statistically significant at p=0.004.

### The Teammate Score Only Works When the Player Actually Broke Out
For WRs, sharing targets with other future NFL receivers is only a positive signal if the player *still produced* despite that competition. Having elite teammates alone doesn't help — it's having elite teammates and still earning targets that matters. This is why teammate score is paired with breakout age, not used independently.

### TE Breakout Threshold Is 15%, Not 20% Like WRs
Tight ends operate in a different target ecosystem. A TE capturing 15% of his team's receiving production is doing something meaningful; using the WR threshold of 20% would eliminate too many productive TEs and weaken the signal.

### Catch Volume > Yards Per Catch for TEs
When projecting TEs, the number of receptions relative to team passing volume is more predictive than receiving yards. The yards-per-reception component adds noise — a TE with 60 catches for 700 yards projects better than one with 30 catches for 700 yards.

### The Model Is a Bust Detector First, Sleeper Finder Second
SLAP's strongest capability is identifying players who are likely to bust despite their draft capital. When the model dings a player (negative delta), it is more often right than wrong. Sleeper identification (positive delta on a late-round player) is useful but inherently less reliable because late-round hit rates are so low that even doubling them still means most players miss.

---

## 5. DOES IT WORK? (Validation Results)

Here are the current validated results after the hit-rate recalculation from nflverse weekly data through 2025.

### Head-to-Head: SLAP V5 vs. Draft Capital Only

| Position | SLAP PRI-AVG r | DC-Only PRI-AVG r | Improvement | Metric Wins | Bootstrap Confidence |
|----------|---------------|-------------------|-------------|-------------|---------------------|
| WR       | +0.4614       | +0.4399           | +0.0215     | 11/11       | 91.3%               |
| RB       | +0.5821       | +0.5393           | +0.0428     | 11/11       | 98.5%               |
| TE       | +0.5208       | +0.4875           | +0.0333     | 9/11        | 95.5%               |
| **Overall** |            |                   |             | **31/33 (94%)** |                  |

*PRI-AVG r = Priority-weighted average correlation across four outcomes (first 3yr PPG at 40%, hit24 at 25%, hit12 at 20%, career PPG at 15%). Higher is better.*

*Metric wins = how many of 11 individual validation metrics (4 correlations, 2 AUC-ROC, 2 Brier scores, 3 top-decile measures) SLAP V5 beats DC-only on.*

*Bootstrap confidence = percentage of 1,000 bootstrap resamples where SLAP V5 outperforms DC-only. Above 95% = strong statistical evidence.*

### What These Numbers Mean in Plain English

- **WR model**: Beats draft-capital-only on every single metric tested (11 for 11). Bootstrap analysis says there's a 91.3% chance this improvement is real, not a statistical fluke.
- **RB model**: Also 11 for 11. Bootstrap confidence of 98.5% — the strongest result of the three positions. RB receiving production is a genuine, reliable signal.
- **TE model**: Wins 9 of 11 metrics with 95.5% bootstrap confidence. The TE model has the smallest sample size (160 TEs), which makes it harder to achieve statistical significance, but the results are still strong.
- **Overall**: 31 of 33 total metrics favor SLAP V5 over draft-capital-only (94%). The two metrics where DC tied or won are within noise range.

### Top-Decile Precision

When the model ranks a player in its top 10% at the position, how often does that player become a fantasy hit?

| Position | SLAP V5 Top-10% Hit Rate | DC-Only Top-10% Hit Rate | Improvement |
|----------|-------------------------|-------------------------|-------------|
| WR       | 62.5%                   | 53.1%                   | +9.4 pts    |
| RB       | 86.4%                   | 81.8%                   | +4.6 pts    |
| TE       | ~54%                    | ~46%                    | +8 pts      |

This is where the model earns its keep. At the top of the rankings — where fantasy managers are making their most important decisions — SLAP V5 identifies hits at a meaningfully higher rate than draft slot alone.

### The Delta Analysis: Boosted vs. Dinged Players

When the model disagrees with draft capital, who's right?

**Across all three positions, players the model boosts (positive delta) average +3.0 to +3.6 PPG more than players the model dings (negative delta).**

This is the ultimate test of whether the non-DC components are adding real information. If the deltas were random noise, boosted and dinged players would perform equally. Instead, there's a consistent, meaningful gap — the model's disagreements are directionally correct.

| Group | Avg First-3yr PPG | Hit24 Rate |
|-------|-------------------|------------|
| Boosted (positive delta) | Higher by +3.0 to +3.6 | Significantly higher |
| Dinged (negative delta) | Lower | Significantly lower |

### What Validation Tells Us (and Doesn't)

**What it tells us:** The non-DC components (breakout age, receiving production, teammate score, early declare, speed score, catch volume) contain real predictive information that draft capital misses. The improvement is modest but consistent and statistically validated.

**What it doesn't tell us:** That the model will correctly predict every player. A SLAP Score of 80 doesn't mean an 80% chance of hitting. The model improves your odds at the margins — it's a better tool than draft capital alone, not a crystal ball.

---

## 6. HOW TO USE IT

### The Tier System

| Tier | SLAP Score | What It Means |
|------|-----------|---------------|
| Elite | 80+ | Premium profile. Strong draft capital + strong college indicators. These players have the highest historical hit rates. |
| Strong | 60-79 | Good profile. Likely drafted in the first few rounds with at least some positive non-DC signal. Worth investing in. |
| Average | 40-59 | Middle-of-the-pack. Often Day 3 picks or Day 2 picks with red flags. Hit rates drop significantly here. |
| Long Shot | Below 40 | Poor historical odds. Late-round picks with weak college profiles. Most players here do not become fantasy-relevant. |

### Hit Rates by Tier (WR / RB)

| SLAP Tier | WR Hit24 Rate | RB Hit24 Rate |
|-----------|--------------|--------------|
| 90+       | ~75%         | ~100%        |
| 80-89     | ~54%         | ~100%        |
| 70-79     | ~19%         | ~90%         |
| 60-69     | ~15%         | ~57%         |
| 50-59     | ~9%          | ~37%         |
| Below 50  | <1%          | ~11%         |

RBs have higher hit rates at every tier because the position is more draft-dependent — a first-round RB is almost guaranteed a starting role, while first-round WRs still have to beat out incumbents and compete for targets.

### How to Read the Delta

The delta (SLAP Score minus DC Score) tells you when the model disagrees with draft capital:

- **Delta of +5 or more:** The model sees something extra — a young breakout age, receiving chops, production in an elite program. Worth a second look.
- **Delta near 0:** The model and draft capital agree. What you see is what you get.
- **Delta of -5 or more:** Red flag. The player was drafted higher than their college profile warrants. Could be an athleticism pick, a scheme fit, or a team reaching. Historically, these players underperform their draft slot.
- **Delta of +10 or more:** Strong positive disagreement. These are your potential sleepers — players whose college profiles suggest more upside than where they were drafted.
- **Delta of -10 or more:** Strong negative disagreement. These are your bust candidates.

### What the Model CAN'T Tell You

SLAP Score is blind to several factors that obviously matter for NFL success:

- **Landing spot**: Scheme fit, offensive line quality, coaching staff, QB quality
- **Injury**: Current health, injury history, durability concerns
- **ADP / dynasty startup cost**: The model doesn't know what the player costs to acquire
- **Character / work ethic**: Intangibles that don't show up in stats
- **Opportunity**: Depth chart, veteran competition, projected snap share
- **Breakout trajectory**: A player trending upward in year 3 might be better than one who plateaued in year 1

### When to Use SLAP Score vs. When to Override It

**Use it when:**
- Comparing players at the same position with similar draft capital (the delta is the tiebreaker)
- Evaluating a dynasty rookie class before landing spot is known
- Identifying bust risk in early-round players (negative deltas)
- Building a "do not draft" list of late-round players with poor profiles
- Looking for mid-round value (positive deltas in Rounds 2-4)

**Override it when:**
- You have strong landing spot information (a Day 3 pick who lands in a starting role)
- A player has a major injury that the model doesn't account for
- The player transferred or had unusual college circumstances that distort the metrics
- You're evaluating a player in a dynasty startup where their NFL track record already exists
- ADP is so far from the SLAP Score that the market has already priced in the information

---

## 7. DATA INTEGRITY

### Sample Size

| Dataset | Count | Description |
|---------|-------|-------------|
| WR backtest | 339 | WRs drafted 2015-2025, all rounds |
| RB backtest | 223 | RBs drafted 2015-2025, all rounds |
| TE backtest | 160 | TEs drafted 2015-2025, all rounds |
| **Total backtest** | **722** | All players with NFL outcome data |
| 2026 WR prospects | 109 | Projected/mock-drafted WRs |
| 2026 RB prospects | 56 | Projected/mock-drafted RBs |
| 2026 TE prospects | 52 | Projected/mock-drafted TEs |
| **Total prospects** | **217** | 2026 draft class |
| **Grand total** | **939** | All players in the database |

### Data Quality by Position

**WR data is the cleanest.** 100% of backtest WRs have real production data (breakout age, dominator rating, rushing yards, early declare status, teammate scores). No imputation needed for any core component.

**RB data is strong but has some imputation in Speed Score.** Receiving production data covers 98%+ of backtest RBs via CFBD API. For Speed Score (only 5% of the model), 68% have real combine weight + 40-yard dash. Another 21% have real weight with estimated 40 times (from weight-by-round group averages). The remaining 4% use MNAR-aware imputation. Because Speed Score is only 5% of the total model, even the imputed values have minimal impact.

**TE data has the most imputation.** About 30% of TEs are missing some PFF production data, requiring fallback to alternative sources or position-average imputation. Breakout age coverage is 53% (many TEs never reached the 15% dominator threshold and are scored with the fallback formula). RAS coverage is 93%.

### NFL Outcomes: How They Were Verified

All NFL outcomes (hit24, hit12, PPG) were calculated from **nflverse weekly data** through the 2025 season. This is the same data source used by the broader fantasy analytics community.

- **Hit24** = player finished as a top-24 scorer at their position in any NFL season (minimum 6 games)
- **Hit12** = player finished as a top-12 scorer at their position in any NFL season (minimum 6 games)
- **First 3yr PPG** = average PPR points per game across the player's first three NFL seasons
- **Career PPG** = average PPR points per game across their entire career to date

**Spot-check results: 20 out of 20 randomly selected players matched external sources exactly.** NFL outcome data passed every verification test.

### Early Declare Status

WR early declare status was verified using a combination of CFBD roster API data (checking roster years vs eligibility) and manual research for ambiguous cases. This is one of the cleanest binary variables in the dataset.

### Known Limitations

1. **TE model has the most uncertainty.** Smaller sample size (160 vs 339 WRs and 223 RBs) and more missing data mean wider confidence intervals on all TE validation metrics.

2. **RB Speed Score uses estimated values for ~25% of players.** Because Speed Score is only 5% of the RB model, the impact on final scores is minimal (less than 1 SLAP point for most players).

3. **2026 prospect data has gaps.** Not all 2026 WR breakout ages have confirmed birthdates (estimated as age 22 when unknown). All 2026 RB 40-yard dash times are estimated (the 2026 Combine data isn't available yet for all players). These gaps are documented and flagged in the database.

4. **The model can't predict landing spot.** A player's NFL opportunity is arguably the biggest factor in fantasy production, and SLAP Score doesn't capture it. This is by design — SLAP is meant to be used *before* the draft, when landing spot is unknown.

5. **Small-school and FCS players** occasionally lack team-level stats (team pass attempts for RBs, full PFF coverage for TEs), which can result in missing production scores. These players are flagged in the database.

6. **The improvement over DC-only is real but modest.** SLAP V5 doesn't turn lead into gold — it improves prediction by 5-10% at the margins. That margin matters over hundreds of decisions, but it won't override a bad landing spot or a torn ACL.

---

*SLAP Score V5 is a product of statistical analysis, not subjective opinion. Every component was tested against real NFL outcomes. Every number in this document is traceable to the data. Use it as one tool in your evaluation process — a powerful one, but not the only one.*
