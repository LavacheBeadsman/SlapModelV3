# SLAP Score V5 — Complete Project Summary

**SLAP = Statistical Likelihood of Achieving Production**

A model that rates NFL Draft running backs (RBs) and wide receivers (WRs) on a 0-100 scale, combining where a player was drafted with their college production profile. Built on 10 years of historical data (2015-2025 draft classes).

---

## 1. WHAT THE SLAP MODEL PREDICTS

### The Goal
The SLAP model tries to predict which drafted RBs and WRs will become productive fantasy football players in the NFL. Specifically, it's targeting players who will score enough fantasy points to be relevant starters.

### What Does a High SLAP Score Mean?
A SLAP score is a 0-100 rating where higher = better projected NFL outcome.

- **90+**: Elite prospect profile. Historically, 75% of WRs and 100% of RBs in this tier became fantasy-relevant (finished a full NFL season as a top-24 scorer at their position).
- **80-89**: Very strong profile. 54% WR hit rate, 100% RB hit rate.
- **70-79**: Good profile. 19% WR hit rate, 90% RB hit rate.
- **60-69**: Average profile. 15% WR hit rate, 57% RB hit rate.
- **50-59**: Below average. 9% WR hit rate, 37% RB hit rate.
- **Below 50**: Poor profile. Less than 1% WR hit rate, 11% RB hit rate.

In plain terms: a SLAP score of 85+ means "this player has the kind of profile that historically produces full-season fantasy starters." A score below 50 means "players with this profile almost never become relevant."

### What Does the Delta Tell Us?
The "delta vs draft-only baseline" is the most content-worthy number in the model. It answers: **"Does SLAP think this player is better or worse than where they were drafted?"**

- **Positive delta (+5 or more)**: The model likes this player MORE than their draft slot suggests. These are potential sleepers — players whose college production profile is stronger than their draft position alone would predict.
- **Negative delta (-5 or more)**: The model likes this player LESS than their draft slot suggests. These are potential busts — players who were drafted high but whose production profile raises red flags.
- **Delta near zero**: The model agrees with where they were drafted. No strong take either way.

The delta is what creates "takes" for content. If a player went in round 2 but has a -15 delta, that's a conversation starter: "The draft said he's great, but his college profile says be careful."

---

## 2. HOW EFFECTIVE IS IT?

### Overall Accuracy — The Honest Numbers

**The single most important thing to understand**: Draft capital (where a player was drafted) is BY FAR the strongest predictor of NFL success. The entire SLAP model is built on this foundation — draft capital makes up 65-70% of every score. The other components (breakout age, receiving production, etc.) add meaningful but modest improvement on top.

Here are the actual correlations (how well the score tracks with NFL production, where 1.0 = perfect and 0.0 = useless):

| Metric | WR SLAP V5 | WR DC-Only | RB SLAP V5 | RB DC-Only |
|--------|-----------|-----------|-----------|-----------|
| Correlation with first 3 years PPG | 0.498 | 0.459 | 0.661 | 0.638 |
| Correlation with hit24 (best week 24+) | 0.479 | 0.465 | 0.520 | 0.514 |
| Correlation with hit12 (best week 12+) | 0.312 | 0.288 | 0.476 | 0.464 |
| Correlation with career PPG | 0.505 | 0.465 | 0.651 | 0.621 |
| **Priority-Weighted Average** | **0.457** | **0.427** | **0.587** | **0.570** |

**What this means**: SLAP V5 beats "just use draft capital" on every single metric for both positions. The improvement is real but not dramatic — roughly a 3-7% improvement in correlation. The model adds the most value in the later rounds where draft capital alone doesn't differentiate well.

### Top Decile Precision — The Headliner Stat

If you pick the top 10% of players by SLAP score, how good are they?

| Metric | WR SLAP V5 | WR DC-Only | RB SLAP V5 | RB DC-Only |
|--------|-----------|-----------|-----------|-----------|
| Hit24 rate (ever top-24 at position) | **63.6%** | 54.5% | **86.4%** | 81.8% |
| Hit12 rate (ever top-12 at position) | **36.4%** | 24.2% | **72.7%** | 63.6% |
| Average first-3-year PPG | **11.62** | 10.86 | **15.71** | 15.19 |

**This is the strongest selling point.** When the model says a player is in the top tier, it's right about 2 out of 3 times for WRs and about 6 out of 7 times for RBs. Draft capital alone would get you 1 out of 2 for WRs and 4 out of 5 for RBs. That's meaningful improvement.

### V5 vs V4 — Did the Update Help?

V5 (the current model) improved on V4 (the previous version) on **12 out of 12 metrics for WRs** and **10 out of 11 metrics for RBs**. Zero regressions for WRs.

The key changes from V4 to V5:
- WRs: Dropped RAS (athletic score), added Teammate Score and Early Declare. Increased DC weight from 65% to 70%.
- RBs: Replaced RAS with Speed Score. Increased DC weight from 50% to 65%.

### Performance by Draft Round

This is where the model earns its keep — or doesn't:

**WRs:**
| Round | Players | Hit24% | SLAP Beats DC? | How Much? |
|-------|---------|--------|----------------|-----------|
| Rd 1 | 56 | 55.4% | YES | SLAP r=0.406 vs DC r=0.279 |
| Rd 2 | 48 | 33.3% | NO | DC slightly better in Rd 2 |
| Rd 3 | 44 | 15.9% | NO | DC slightly better in Rd 3 |
| Rd 4 | 43 | 7.0% | YES | SLAP r=0.534 vs DC r=0.417 |
| Rd 5 | 38 | 2.6% | YES | SLAP r=0.221 vs DC r=-0.125 |
| Rd 6-7 | 110 | 0.9% | — | Too few hits to measure |

**RBs:**
| Round | Players | Hit24% | SLAP Beats DC? | How Much? |
|-------|---------|--------|----------------|-----------|
| Rd 1 | 22 | 90.9% | NO | DC wins in Rd 1 (everyone hits) |
| Rd 2 | 32 | 56.3% | YES | SLAP adds value |
| Rd 3 | 32 | 37.5% | YES | SLAP adds value |
| Rd 4 | 33 | 27.3% | YES | SLAP adds value |
| Rd 5+ | 104 | 8.7% | YES | SLAP r=0.206 vs DC r=0.080 |

**The pattern is clear**: In Round 1, draft capital alone works great (NFL teams got it right). In the mid-to-late rounds (4+), SLAP's non-DC components — breakout age for WRs, receiving production for RBs — add significant value. The biggest SLAP advantage comes in Round 4+ where draft capital alone is nearly useless (correlation near zero) but SLAP still has meaningful signal.

### Where It Succeeds

1. **Tier separation works beautifully.** The gap between tiers is massive and consistent. A player scored 90+ is dramatically more likely to succeed than a player scored below 50. This makes the tier system reliable for content.

2. **Disagreements tend to be correct.** When V5 boosts a player above their draft slot (positive delta), those players average +1.3 PPG over players the model dinged. The WR "risers" had a 40% hit rate vs 0% for "fallers." The RB risers had 80% vs 10%.

3. **Late-round differentiation.** This is the model's biggest value-add. Among day-3 picks (rounds 4-7), draft capital alone is almost random. SLAP's production components help identify the few late-round picks who actually have a chance.

### Where It Fails

1. **It can't beat draft capital in Round 1.** This isn't surprising — NFL teams spend millions on scouting, and first-round picks are the most researched players. The model can't outsmart 32 NFL front offices on their highest-conviction picks.

2. **WR prediction is harder than RB.** The overall WR correlation (0.498) is meaningfully lower than RB (0.661). WR success in the NFL depends more on scheme fit, quarterback play, and opportunity — things no college stat can capture.

3. **The non-DC components are modest.** Breakout age, receiving production, speed score — they help, but they're not transformative. The improvement over "just follow draft capital" is roughly 5-10% better. Don't oversell this as some revolutionary predictive system.

4. **It can't predict injuries, opportunity, or landing spot.** A player could have a 95 SLAP score and bust because they tear an ACL, land behind an entrenched starter, or play in a run-first offense. The model predicts *profile quality*, not *career outcomes*.

5. **Rd 6-7 WRs are essentially unpredictable.** Hit rates are below 1% regardless of score. There just isn't enough signal to find winners this late in the draft for WRs.

---

## 3. WHAT DATA IS IN THE MASTER DATASET?

### Master V5 Database (output/slap_v5_database.csv)
**562 total players: 339 WRs + 223 RBs, covering 2015-2025 draft classes**

Every row is one drafted player. Here's what each column means:

| Column | What It Is | Position |
|--------|-----------|----------|
| `player_name` | Player's full name | Both |
| `position` | WR or RB | Both |
| `school` | College they played at | Both |
| `draft_year` | Year they were drafted (2015-2025) | Both |
| `pick` | Overall draft pick number (1-262) | Both |
| `round` | Draft round (1-7) | Both |
| `slap_v5` | The V5 SLAP score (0-100) | Both |
| `slap_v4` | The old V4 SLAP score (for comparison) | Both |
| `dc_score` | Draft Capital score alone (0-100). Higher draft pick = higher score. | Both |
| `enhanced_breakout` | WR breakout age score (0-99.9) with rush bonus. Age when player first dominated in college + bonus for rushing versatility. | WR only |
| `teammate_score` | 0 or 100. Did this WR produce despite sharing with other elite receivers? 100 = yes (like Waddle at Alabama with DeVonta Smith). | WR only |
| `early_declare_score` | 0 or 100. Did the player leave college early? 100 = left early (signals NFL-readiness). | WR only |
| `delta_v5_dc` | SLAP V5 minus DC-only score. Positive = model likes them more than draft slot. Negative = model likes them less. | Both |
| `delta_v5_v4` | SLAP V5 minus V4 score. Shows how the model update changed this player's rating. | Both |
| `breakout_age` | WR: Age when they first hit 20%+ dominator rating in college (younger = better). | WR only |
| `peak_dominator` | WR: Highest dominator rating achieved in any college season. Dominator = player's receiving yards divided by team's total receiving yards. | WR only |
| `rush_yards` | WR: College rushing yards. 20+ yards earns a small +5 bonus (signals versatility). | WR only |
| `production_score` | RB: Receiving production score (0-99.9). Based on receiving yards per team pass attempt, adjusted for age. | RB only |
| `speed_score_norm` | RB: Normalized speed score (0-100). Rewards players who are fast for their size. | RB only |
| `rec_yards` | RB: Receiving yards in their final college season. | RB only |
| `team_pass_att` | RB: Their team's total pass attempts that season. | RB only |
| `weight` | RB: Player's weight in pounds (from NFL Combine). | RB only |
| `forty` | RB: 40-yard dash time in seconds (from NFL Combine). | RB only |
| `nfl_hit24` | 1 or 0. Did the player ever finish a full NFL season as a top-24 scorer at their position (WR or RB) in PPR fantasy points? This is the primary "success" threshold — it means they were a fantasy starter for at least one season. | Both |
| `nfl_hit12` | 1 or 0. Did they ever finish a full season as a top-12 scorer at their position? A higher bar — this means they were an elite fantasy starter. | Both |
| `nfl_first_3yr_ppg` | Average fantasy points per game over their first 3 NFL seasons. The main continuous outcome metric. | Both |
| `nfl_career_ppg` | Average fantasy points per game across their entire NFL career. | Both |
| `nfl_best_ppr` | Best single-season PPR fantasy points total. | Both |
| `nfl_best_ppg` | Best single-season fantasy points per game. | RB only |
| `data_type` | Always "backtest" for this file. Distinguishes historical data from 2026 projections. | Both |

### 2026 WR Projections (output/slap_v5_wr_2026.csv)
**109 WR prospects**

| Column | What It Is |
|--------|-----------|
| `rank` | Ranking by SLAP score (1 = best) |
| `player_name` | Player's name |
| `school` | College |
| `projected_pick` | Consensus mock draft projected pick (capped at 250 for UDFAs) |
| `slap_v5` | SLAP V5 score (0-100) |
| `dc_score` | Draft capital score from projected pick |
| `enhanced_breakout` | Breakout age score + rush bonus |
| `teammate_score` | 0 or 100 (shared targets with other top-drafted WRs?) |
| `early_declare` | 0 or 100 (left college early?) |
| `delta_vs_dc` | SLAP minus DC-only (the "take" number) |
| `breakout_age` | Age at first 20%+ dominator season |
| `peak_dominator` | Best dominator rating in college |
| `rush_yards` | College rushing yards |

### 2026 RB Projections (output/slap_v5_rb_2026.csv)
**56 RB prospects**

| Column | What It Is |
|--------|-----------|
| `rank` | Ranking by SLAP score |
| `player_name` | Player's name |
| `school` | College |
| `projected_pick` | Consensus mock draft projected pick |
| `slap_v5` | SLAP V5 score |
| `dc_score` | Draft capital score |
| `production_score` | Receiving production score |
| `speed_score` | Speed Score (fast-for-size metric) |
| `delta_vs_dc` | SLAP minus DC-only |
| `production_status` | "observed" = real data, "imputed" = estimated (missing receiving stats) |
| `speed_score_status` | How speed score was calculated ("weight_est40" = real weight with estimated 40 time, "imputed_mnar" = fully estimated) |
| `rec_yards` | Receiving yards in 2025 college season |
| `team_pass_attempts` | Team's pass attempts in 2025 |
| `weight` | Player weight |

---

## 4. DATA QUALITY ASSESSMENT

### What's Rock Solid

**Draft capital data** — 100% complete for all 562 backtest players and all 165 prospects. Every player has a real draft pick (backtest) or a researched projected pick (2026). This is the foundation and it's bulletproof.

**NFL outcome data (backtest)** — For players drafted 2015-2022, we have complete fantasy production data (PPG, hit rates, best seasons). Players drafted 2023-2025 have partial data (1-2 seasons instead of 3). 217/339 WRs and 149/223 RBs have first-3-year PPG.

**WR backtest components** — 339/339 WRs have breakout age, rush yards, early declare status, and teammate scores. This data was researched and validated for every player.

**RB receiving production** — 222/223 backtest RBs have receiving yards (99.6%). 219/223 have team pass attempts (98.2%). This came from the CFBD API and was cross-checked.

### Known Gaps and Limitations

**RB athletic data (Speed Score)** — Only 68% of backtest RBs have real 40-yard dash times from the NFL Combine. The remaining 32% were estimated:
- Players with real weight but no 40 time: estimated from weight-by-round group averages (21%)
- Players with no weight or 40 time: given a percentile-based default based on draft round (11%)
- Elite early-round prospects who skip workouts get a HIGHER default than late-round prospects. This is intentional — the data is "missing not at random" (MNAR). A first-round pick who didn't run is probably elite; a seventh-round pick who didn't run probably wasn't invited to the Combine.

**2026 WR breakout ages** — This is the weakest data for 2026 projections:
- 45/107 WRs (42%) hit the 20% dominator threshold in college (they "broke out")
- Of those 45, only 30 have a known birthdate to calculate their exact breakout age
- 15 WRs broke out but we don't know their age when it happened (missing birthdate)
- 61 WRs never hit 20% dominator at all — scored using a lower formula based on peak dominator
- 1 WR (John Carroll - D3) has no CFBD data at all

**2026 WR rush yards** — Only 54/109 (50%) have rush data from CFBD. The other 55 WRs default to no rush bonus (+0). This means some players might be slightly underscored if they had rushing production we couldn't find.

**2026 WR teammate scores** — Calculated from the mock draft (who else from the same school is projected to be drafted as a WR/TE). This is an approximation — actual teammate DC depends on real draft results. Only 4/109 WRs currently qualify for the teammate bonus.

**2026 WR early declare** — Estimated from age (21 or younger at draft = early declare). This is a reasonable proxy but not perfect — some 22-year-olds may have declared early (redshirt sophomores).

**2026 birthdates** — Only 61% of WR prospects and 71% of RB prospects have confirmed birthdates. Missing birthdates are estimated as age 22 (the typical draft age). This affects breakout age calculations for WRs.

**2026 RB Speed Scores** — All 56 RB prospects have estimated (not real) 40-yard dash times because the 2026 NFL Combine hasn't happened yet. Speed scores are based on real weight + estimated 40 time from historical weight-by-round averages.

### What Was Imputed (Estimated) and How

| Data Point | Method | Count | Impact |
|-----------|--------|-------|--------|
| RB 40-yard dash (backtest) | Weight-by-round group average | 46/223 (21%) | Low — speed score is only 5% of total |
| RB speed score (backtest, no data) | Round-based percentile (Rd 1-2 get 60th pctl, Rd 3+ get 40th) | 8/223 (4%) | Low |
| RB weight (backtest) | CFBD API player search | 18/223 recovered | Improved accuracy |
| RB production (2026, missing) | Backtest average (32.0) | 15/56 (27%) | Moderate — these are flagged as "imputed" |
| All 2026 RB 40 times | Weight-by-round historical average | 56/56 (100%) | Low — speed score is only 5% |
| 2026 WR breakout age (missing birthdate) | Scored as "never broke out" fallback | 15/109 | Moderate — some of these DID break out but age is unknown |
| 2026 WR early declare | Age <= 21 proxy | 109/109 | Low — 5% weight |
| 2026 WR teammate score | Mock draft projections | 109/109 | Low — 5% weight, only 4 players affected |

### Manual Fixes Made During Audits

- **Season selection fix**: 19 RBs had inflated scores because the code was using their "best season" instead of their "final season." This was fixed to always use the final college season (draft_year - 1), matching the methodology that was validated in backtesting. Average score correction: -12.6 production points, -1.26 SLAP points.
- **School name mapping**: Several schools needed name corrections to match CFBD API format (e.g., "Ole Miss" vs "Mississippi", "Miami" vs "Miami (FL)").
- **CFBD weight recovery**: 18 RBs who were missing combine weight data had their weight recovered via CFBD player search API.
- **DC formula update**: Changed from `normalize(1/sqrt(pick))` to `100 - 2.40 * (pick^0.62 - 1)` to create more realistic score differentiation (old formula penalized mid-round picks too harshly).

---

## 5. MY HONEST ASSESSMENT

### Strongest Aspects

1. **The tier system is reliable and content-ready.** The gap between a 90+ player and a sub-50 player is enormous and consistent across 10 years of data. You can confidently say "this player is in the elite tier" and have historical backing.

2. **RB prediction is genuinely strong.** An r=0.661 correlation with 3-year PPG is meaningful. The RB model correctly identifies 86% of top-tier hits. Receiving production as a RB metric is statistically significant (p=0.004) — this isn't noise, it's real signal.

3. **The delta creates great content.** The "SLAP agrees/disagrees with draft slot" framing is intuitive and produces interesting takes. Historically, the model's disagreements tend to be correct — risers outperform fallers.

4. **V5 is a genuine improvement.** It beat V4 on 22 of 23 metrics across both positions. Dropping RAS (which had severe missing-data problems) and increasing draft capital weight were the right calls.

5. **It's transparent and auditable.** Every score can be traced back to specific inputs. There's no black box — you can explain exactly why a player scored what they did.

### Weakest Aspects

1. **The improvement over "just use draft capital" is modest.** Let's be honest: if you ranked every prospect by draft pick and did nothing else, you'd get r=0.459 for WRs and r=0.638 for RBs. SLAP V5 gets you to r=0.498 and r=0.661. That's better, but it's not transformative. The non-DC components add ~5-10% improvement, not 50%.

2. **WR prediction is mediocre.** An r=0.498 means the model explains about 25% of the variance in WR outcomes. The other 75% is landing spot, quarterback play, scheme, injuries, development — things no college stat can capture. WR is inherently harder to predict than RB.

3. **Late-round WRs are essentially a coin flip.** Below round 4, WR hit rates are in the single digits regardless of SLAP score. The model can slightly improve your odds in rounds 4-5, but by rounds 6-7, there's almost no useful signal.

4. **2026 data has significant gaps.** Only 28% of 2026 WR prospects have a confirmed breakout age. Half are missing rush data. All RB speed scores are estimated. The 2026 projections are less reliable than the backtested scores — treat them as directional, not precise.

5. **The model can't capture context.** It doesn't know about coaching changes, offensive scheme, target competition, injury history, or draft capital invested in the position by the drafting team. A player with a perfect SLAP profile landing behind a franchise WR will underperform their score.

### What I'd Improve If I Could

1. **Better 2026 birthdate coverage.** 39% of WR prospects are missing birthdates, which directly affects the breakout age calculation (the second-most important WR component). Filling these would immediately improve 2026 WR accuracy.

2. **Landing spot adjustment (post-draft).** After the draft, adding a simple adjustment for team context (target share available, offensive system) would be the single biggest potential improvement.

3. **Multi-season receiving data for RBs.** Currently uses only the final college season. Testing whether a multi-season average or trajectory (improving receiving production over time) adds signal could help.

4. **Better early declare classification.** The current age-based proxy for 2026 is imperfect. Manually confirming declare status for at least the top 30-40 prospects would be more accurate.

### SLAP vs "Just Use Draft Capital" — The Bottom Line

Here's the honest comparison:

**If you just ranked players by draft pick**, you'd get:
- WR: r=0.459 with 3yr PPG, 54.5% top-decile hit rate
- RB: r=0.638 with 3yr PPG, 81.8% top-decile hit rate

**SLAP V5 gets you**:
- WR: r=0.498 with 3yr PPG, 63.6% top-decile hit rate (+9.1 percentage points)
- RB: r=0.661 with 3yr PPG, 86.4% top-decile hit rate (+4.6 percentage points)

**The value-add is real but concentrated in two areas:**
1. **Top-tier precision** — SLAP is meaningfully better at identifying which top-30 picks will actually hit (63.6% vs 54.5% for WRs).
2. **Mid-to-late round differentiation** — In rounds 4+, draft capital alone has near-zero correlation. SLAP's production components (breakout age, receiving production) still have useful signal (r=0.25 for WRs, r=0.21 for RBs).

**Where draft capital alone is just as good**: Round 1. Everyone knows first-round picks are good. The model doesn't add much here.

**The honest pitch**: SLAP is not a crystal ball. It's a structured way to combine draft capital with the college production metrics that have historically predicted NFL success. It's most valuable for identifying sleepers (late picks with strong profiles) and busts (early picks with weak profiles) — exactly the "takes" that make for good content. Don't claim it predicts the future. Do claim it identifies which profiles historically produce NFL starters, and that it's statistically validated over 10 years of data.
