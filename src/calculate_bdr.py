"""
Calculate Backfield Dominator Rating (BDR) for all RBs.

BDR = average of 4 market shares × 100:
  - Player rush yards / total RB rush yards on team
  - Player rush TDs / total RB rush TDs on team
  - Player rec yards / total RB rec yards on team
  - Player rec TDs / total RB rec TDs on team

Uses CFBD API to get position-level team stats for each RB's final college season.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

API_KEY = "xsMxXYs8bAGu3k1SYY/npc3Io5yplGKqFSHJC4ZLq6/Xo+RkNGMkhuQo/9i1qOIE"
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

SCHOOL_MAPPINGS = {
    "Ohio St.": "Ohio State",
    "Penn St.": "Penn State",
    "Michigan St.": "Michigan State",
    "North Dakota St.": "North Dakota State",
    "Oklahoma St.": "Oklahoma State",
    "Miss. St.": "Mississippi State",
    "Mississippi St.": "Mississippi State",
    "San Diego St.": "San Diego State",
    "Florida St.": "Florida State",
    "Fresno St.": "Fresno State",
    "Boise St.": "Boise State",
    "Iowa St.": "Iowa State",
    "Arizona St.": "Arizona State",
    "Kansas St.": "Kansas State",
    "N.C. State": "NC State",
    "N.C.": "North Carolina",
    "North Carolina St.": "NC State",
    "Appalachian St.": "Appalachian State",
    "Colorado St.": "Colorado State",
    "Oregon St.": "Oregon State",
    "Washington St.": "Washington State",
    "S. Carolina": "South Carolina",
    "Miami (FL)": "Miami",
    "Central Mich.": "Central Michigan",
    "Western Mich.": "Western Michigan",
    "Eastern Mich.": "Eastern Michigan",
    "Northern Ill.": "Northern Illinois",
    "Southern Miss.": "Southern Mississippi",
    "Southern Miss": "Southern Mississippi",
    "San Jose St.": "San Jose State",
    "La.-Monroe": "Louisiana Monroe",
    "La.-Lafayette": "Louisiana",
    "Louisiana-Lafayette": "Louisiana",
    "Louisiana Tech": "Louisiana Tech",
    "Texas-El Paso": "UTEP",
    "North Carolina A&T": None,
    "Virginia St.": None,
    "Ala-Birmingham": "UAB",
    "Coastal Carolina": "Coastal Carolina",
    "South Dakota St.": "South Dakota State",
    "New Mexico St.": "New Mexico State",
    "Boston Col.": "Boston College",
    "Boston College": "Boston College",
    "Mississippi": "Ole Miss",
    "Pitt": "Pittsburgh",
    "Pittsburgh": "Pittsburgh",
    "Sam Houston State": "Sam Houston",
    "Sam Houston St.": "Sam Houston",
    "Stephen F. Austin": "Stephen F. Austin",
    "Jacksonville State": "Jacksonville State",
    "Georgia State": "Georgia State",
    "McNeese State": "McNeese",
    "Sacramento State": "Sacramento State",
    "Virginia Union": None,  # FCS / D2
    "UT Martin": "UT Martin",
    "Florida International": "FIU",
}


def normalize_school(school):
    if pd.isna(school):
        return None
    school = str(school).strip()
    return SCHOOL_MAPPINGS.get(school, school)


def names_match(name1, name2):
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if n1 == n2:
        return True
    parts1 = n1.replace('.', ' ').replace("'", "").replace("'", "").split()
    parts2 = n2.replace('.', ' ').replace("'", "").replace("'", "").split()
    suffixes = {'jr', 'jr.', 'ii', 'iii', 'iv', 'sr'}
    clean1 = [p for p in parts1 if p not in suffixes]
    clean2 = [p for p in parts2 if p not in suffixes]
    if not clean1 or not clean2:
        return False
    if clean1[0] != clean2[0]:
        if not (len(clean1[0]) <= 2 and clean2[0].startswith(clean1[0][0])):
            if not (len(clean2[0]) <= 2 and clean1[0].startswith(clean2[0][0])):
                return False
    if len(clean1) > 1 and len(clean2) > 1:
        if clean1[-1] != clean2[-1]:
            return False
    return True


# Backfield positions: RB, FB, and "?" (unlabeled in older data — usually RBs)
BACKFIELD_POSITIONS = {"RB", "FB", "?", ""}


def fetch_team_category_stats(team, year, category):
    """Fetch all player stats for a team/year/category from CFBD.
    Returns dict: {player_name_lower: {statType: value, ...}, ...}
    Also stores position for each player.
    """
    url = f"{BASE_URL}/stats/player/season"
    params = {"year": year, "category": category, "team": team}

    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if response.status_code == 200:
                players = {}
                positions = {}
                for stat in response.json():
                    player = stat.get("player", "").strip()
                    player_lower = player.lower()
                    pos = stat.get("position", "?")
                    stat_type = stat.get("statType", "")
                    value = stat.get("stat", 0)

                    if player_lower not in players:
                        players[player_lower] = {"_name": player}
                        positions[player_lower] = pos
                    try:
                        players[player_lower][stat_type] = int(float(value))
                    except (ValueError, TypeError):
                        players[player_lower][stat_type] = 0

                return players, positions
            elif response.status_code == 429:
                time.sleep(5)
            else:
                return {}, {}
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  Error fetching {team} {year} {category}: {e}")
                return {}, {}
    return {}, {}


def calculate_bdr_for_team(team, year, cache):
    """Calculate BDR for all backfield players on a team in a given year.
    Returns dict: {player_name_lower: bdr_value, ...}
    """
    cache_key = (team, year)
    if cache_key in cache:
        return cache[cache_key]

    # Fetch rushing stats
    rush_stats, rush_pos = fetch_team_category_stats(team, year, "rushing")
    time.sleep(0.3)

    # Fetch receiving stats
    rec_stats, rec_pos = fetch_team_category_stats(team, year, "receiving")
    time.sleep(0.3)

    # Merge position info (rushing position takes priority)
    all_pos = {**rec_pos, **rush_pos}

    # Identify backfield players
    backfield = set()
    for player, pos in all_pos.items():
        if pos in BACKFIELD_POSITIONS:
            backfield.add(player)

    if not backfield:
        cache[cache_key] = {}
        return {}

    # Calculate team-level backfield totals
    total_rush_yds = sum(rush_stats.get(p, {}).get("YDS", 0) for p in backfield)
    total_rush_td = sum(rush_stats.get(p, {}).get("TD", 0) for p in backfield)
    total_rec_yds = sum(rec_stats.get(p, {}).get("YDS", 0) for p in backfield)
    total_rec_td = sum(rec_stats.get(p, {}).get("TD", 0) for p in backfield)

    # Calculate BDR for each backfield player
    results = {}
    for player in backfield:
        p_rush_yds = rush_stats.get(player, {}).get("YDS", 0)
        p_rush_td = rush_stats.get(player, {}).get("TD", 0)
        p_rec_yds = rec_stats.get(player, {}).get("YDS", 0)
        p_rec_td = rec_stats.get(player, {}).get("TD", 0)

        shares = []
        if total_rush_yds > 0:
            shares.append(p_rush_yds / total_rush_yds)
        if total_rush_td > 0:
            shares.append(p_rush_td / total_rush_td)
        if total_rec_yds > 0:
            shares.append(p_rec_yds / total_rec_yds)
        if total_rec_td > 0:
            shares.append(p_rec_td / total_rec_td)

        if shares:
            bdr = sum(shares) / len(shares) * 100
        else:
            bdr = np.nan

        # Store full name from the data
        display_name = rush_stats.get(player, {}).get("_name",
                       rec_stats.get(player, {}).get("_name", player))
        results[player] = {
            "bdr": bdr,
            "display_name": display_name,
            "rush_yds": p_rush_yds,
            "rush_td": p_rush_td,
            "rec_yds": p_rec_yds,
            "rec_td": p_rec_td,
            "team_rush_yds": total_rush_yds,
            "team_rush_td": total_rush_td,
            "team_rec_yds": total_rec_yds,
            "team_rec_td": total_rec_td,
            "n_shares": len(shares),
        }

    cache[cache_key] = results
    return results


def find_player_bdr(player_name, team_bdr):
    """Find a player's BDR in the team results using fuzzy name matching."""
    if not team_bdr:
        return None

    # Try exact match first
    player_lower = player_name.lower().strip()
    if player_lower in team_bdr:
        return team_bdr[player_lower]

    # Try fuzzy match
    for api_name, data in team_bdr.items():
        if names_match(player_name, api_name):
            return data

    return None


def main():
    print("=" * 70)
    print("CALCULATING BDR FOR ALL RBs")
    print("=" * 70)

    # Load backtest RBs
    bt = pd.read_csv("data/rb_backtest_with_receiving.csv")
    bt["season"] = bt["draft_year"] - 1
    bt["cfbd_school"] = bt["college"].apply(normalize_school)
    print(f"Backtest RBs: {len(bt)}")

    # Load 2026 prospects
    pros = pd.read_csv("data/prospects_final.csv")
    pros = pros[pros["position"] == "RB"].copy()
    pros["season"] = 2025
    pros["cfbd_school"] = pros["school"].apply(normalize_school)
    print(f"2026 RB prospects: {len(pros)}")

    # Build list of all (team, year) combos needed
    combos = set()
    for _, row in bt.iterrows():
        if row["cfbd_school"] and pd.notna(row["cfbd_school"]):
            combos.add((row["cfbd_school"], int(row["season"])))
    for _, row in pros.iterrows():
        if row["cfbd_school"] and pd.notna(row["cfbd_school"]):
            combos.add((row["cfbd_school"], int(row["season"])))

    print(f"Unique (team, year) combos: {len(combos)}")
    print(f"API calls needed: ~{len(combos) * 2}")
    print()

    # Fetch and calculate BDR for all combos
    cache = {}
    done = 0
    total = len(combos)

    for team, year in sorted(combos):
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  Progress: {done}/{total} teams fetched...")
        calculate_bdr_for_team(team, year, cache)

    print(f"\n  Fetched {len(cache)} team-season combos")

    # Match BDR to backtest RBs
    print("\n" + "=" * 70)
    print("STEP 2: MATCHING BDR TO BACKTEST RBs")
    print("=" * 70)

    bt_bdr = []
    bt_matched = 0
    bt_skipped_school = 0
    bt_no_match = 0

    for _, row in bt.iterrows():
        school = row["cfbd_school"]
        season = row["season"]
        player = row["player_name"]

        if not school:
            bt_skipped_school += 1
            bt_bdr.append(np.nan)
            continue

        team_bdr = cache.get((school, season), {})
        result = find_player_bdr(player, team_bdr)

        if result:
            bt_bdr.append(result["bdr"])
            bt_matched += 1
        else:
            # Try cfbd_name if available
            cfbd_name = row.get("cfbd_name", "")
            if pd.notna(cfbd_name) and cfbd_name:
                result = find_player_bdr(cfbd_name, team_bdr)
                if result:
                    bt_bdr.append(result["bdr"])
                    bt_matched += 1
                    continue
            bt_bdr.append(np.nan)
            bt_no_match += 1

    bt["bdr"] = bt_bdr

    print(f"Backtest RBs matched: {bt_matched}/{len(bt)} ({100*bt_matched/len(bt):.1f}%)")
    print(f"  Skipped (FCS/no school): {bt_skipped_school}")
    print(f"  No match found: {bt_no_match}")

    # Match BDR to 2026 prospects
    pro_bdr = []
    pro_matched = 0

    for _, row in pros.iterrows():
        school = row["cfbd_school"]
        season = row["season"]
        player = row["player_name"]

        if not school:
            pro_bdr.append(np.nan)
            continue

        team_bdr = cache.get((school, season), {})
        result = find_player_bdr(player, team_bdr)

        if result:
            pro_bdr.append(result["bdr"])
            pro_matched += 1
        else:
            pro_bdr.append(np.nan)

    pros["bdr"] = pro_bdr

    print(f"2026 prospects matched: {pro_matched}/{len(pros)} ({100*pro_matched/len(pros):.1f}%)")

    # Show unmatched backtest players
    unmatched = bt[bt["bdr"].isna()][["player_name", "college", "season"]].head(15)
    if len(unmatched) > 0:
        print(f"\nSample unmatched backtest RBs:")
        for _, r in unmatched.iterrows():
            print(f"  {r['player_name']} ({r['college']}, {r['season']})")

    # Save results
    bt.to_csv("data/rb_bdr_calculated.csv", index=False)
    pros_out = pros[["player_name", "school", "bdr"]].copy()
    pros_out.to_csv("data/rb_2026_bdr.csv", index=False)

    print(f"\nSaved: data/rb_bdr_calculated.csv ({len(bt)} rows)")
    print(f"Saved: data/rb_2026_bdr.csv ({len(pros)} rows)")

    # BDR distribution
    valid_bdr = bt["bdr"].dropna()
    print(f"\nBDR distribution (backtest, N={len(valid_bdr)}):")
    print(f"  Mean: {valid_bdr.mean():.1f}")
    print(f"  Median: {valid_bdr.median():.1f}")
    print(f"  Min: {valid_bdr.min():.1f}, Max: {valid_bdr.max():.1f}")
    print(f"  P25: {valid_bdr.quantile(0.25):.1f}, P75: {valid_bdr.quantile(0.75):.1f}")

    # =====================================================================
    # STEP 3: Verify against Anatomy dataset
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 3: VERIFICATION vs ANATOMY BDR VALUES")
    print("=" * 70)

    import csv, re, glob

    # Parse Anatomy BDR values
    anatomy_bdr = {}
    files = sorted(glob.glob("data/Anatomy of Top WR & RB - *.csv"))
    skip_files = ['2000s', '2025', '2026', '2024 (1)']

    for f in files:
        fname = os.path.basename(f)
        if any(s in fname for s in skip_files):
            continue
        year_match = re.search(r'(\d{4})', fname)
        if not year_match:
            continue
        draft_year = int(year_match.group(1))

        with open(f, encoding='utf-8-sig') as fh:
            rows = list(csv.reader(fh))

        header = [h.strip().replace('\n', ' ') for h in rows[0]]

        # Find separator
        sep = None
        for i in range(8, len(header)):
            if not header[i]:
                sep = i
                break
        if not sep:
            continue

        rb_h = header[sep:]

        # Find BDR column in RB section
        bdr_idx = None
        for i, h in enumerate(rb_h):
            if 'bdr' in h.lower():
                bdr_idx = i + sep
                break
        if bdr_idx is None:
            continue

        # Find RB name column
        for row in rows[1:]:
            rb_name = None
            for check in [sep, sep+1, sep+2]:
                if check < len(row):
                    v = row[check].strip()
                    if v and any(c.isalpha() for c in v) and len(v) > 2 and v not in ['N/A']:
                        rb_name = v
                        break
            if rb_name and bdr_idx < len(row):
                val = row[bdr_idx].strip().replace(',', '')
                try:
                    anat_val = float(val)
                    anatomy_bdr[(rb_name, draft_year)] = anat_val
                except (ValueError, TypeError):
                    pass

    print(f"Anatomy BDR values loaded: {len(anatomy_bdr)}")

    # Compare
    comparisons = []
    for _, row in bt.iterrows():
        if pd.isna(row["bdr"]):
            continue
        name = row["player_name"]
        dy = row["draft_year"]
        anat = anatomy_bdr.get((name, dy))
        if anat is not None:
            comparisons.append({
                "player": name,
                "year": dy,
                "our_bdr": row["bdr"],
                "anatomy_bdr": anat,
                "diff": abs(row["bdr"] - anat),
            })

    comparisons.sort(key=lambda x: x["diff"])
    print(f"Players with both our BDR and Anatomy BDR: {len(comparisons)}")
    print(f"\n{'Player':25s} {'Year':>5s} {'Our BDR':>8s} {'Anat BDR':>9s} {'Diff':>6s}")
    print("-" * 60)

    # Show 10 — mix of close and far
    if len(comparisons) >= 10:
        # Show 5 closest and 5 furthest
        show = comparisons[:5] + comparisons[-5:]
    else:
        show = comparisons

    for c in show:
        print(f"{c['player']:25s} {c['year']:5d} {c['our_bdr']:8.1f} {c['anatomy_bdr']:9.1f} {c['diff']:6.1f}")

    if comparisons:
        diffs = [c["diff"] for c in comparisons]
        print(f"\nMedian absolute difference: {np.median(diffs):.1f}")
        print(f"Mean absolute difference: {np.mean(diffs):.1f}")
        calc_vals = [c["our_bdr"] for c in comparisons]
        anat_vals = [c["anatomy_bdr"] for c in comparisons]
        r, p = stats.spearmanr(calc_vals, anat_vals)
        print(f"Spearman correlation: r={r:.3f}, p={p:.4f}")

    # =====================================================================
    # STEP 4: Partial correlation with NFL outcomes
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 4: PARTIAL CORRELATIONS (controlling for DC and RYPTPA)")
    print("=" * 70)

    # Load outcomes
    outcomes = pd.read_csv("data/backtest_outcomes_complete.csv")
    outcomes = outcomes[outcomes["position"] == "RB"][
        ["player_name", "draft_year", "first_3yr_ppg", "career_ppg"]
    ].copy()
    bt_full = bt.merge(outcomes, on=["player_name", "draft_year"], how="left")

    # Calculate DC score and RYPTPA
    def dc_score(pick):
        if pd.isna(pick):
            return np.nan
        return 100 - 2.40 * (pick ** 0.62 - 1)

    bt_full["dc_score"] = bt_full["pick"].apply(dc_score)

    # RYPTPA from existing data
    def calc_ryptpa(row):
        if pd.isna(row["rec_yards"]) or pd.isna(row["team_pass_att"]) or row["team_pass_att"] == 0:
            return np.nan
        season_age = row["age"] - 1
        age_w = max(0.85, min(1.15, 1.15 - 0.05 * (season_age - 19)))
        return min(99.9, (row["rec_yards"] / row["team_pass_att"]) * age_w * 100 / 1.75)

    bt_full["ryptpa"] = bt_full.apply(calc_ryptpa, axis=1)

    outcome_cols = ["hit24", "hit12", "career_ppg", "first_3yr_ppg"]

    def partial_corr(x, y, covariates):
        """Partial Spearman correlation of x,y controlling for covariates."""
        mask = np.isfinite(x) & np.isfinite(y)
        for c in covariates:
            mask = mask & np.isfinite(c)
        if mask.sum() < 20:
            return np.nan, np.nan, mask.sum()
        x_m, y_m = x[mask], y[mask]
        covs = [c[mask] for c in covariates]

        # Rank everything
        x_r = stats.rankdata(x_m)
        y_r = stats.rankdata(y_m)
        cov_r = [stats.rankdata(c) for c in covs]

        # Residualize x and y on covariates
        def residualize(a, controls):
            from numpy.linalg import lstsq
            X = np.column_stack(controls)
            X = np.column_stack([np.ones(len(a)), X])
            coef, _, _, _ = lstsq(X, a, rcond=None)
            return a - X @ coef

        x_res = residualize(x_r, cov_r)
        y_res = residualize(y_r, cov_r)
        from scipy.stats import pearsonr
        r, p = pearsonr(x_res, y_res)
        return r, p, mask.sum()

    def raw_spearman(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 20:
            return np.nan, np.nan, mask.sum()
        r, p = stats.spearmanr(x[mask], y[mask])
        return r, p, mask.sum()

    bdr_vals = bt_full["bdr"].values.astype(float)
    dc_vals = bt_full["dc_score"].values.astype(float)
    ryptpa_vals = bt_full["ryptpa"].values.astype(float)

    print(f"\n{'Outcome':15s} | {'N':>4s} | {'Raw r':>7s} | {'r|DC':>7s} | {'r|DC+RYP':>8s} | {'p|DC+RYP':>8s}")
    print("-" * 65)

    for out in outcome_cols:
        y = bt_full[out].values.astype(float)
        raw_r, raw_p, raw_n = raw_spearman(bdr_vals, y)
        dc_r, dc_p, dc_n = partial_corr(bdr_vals, y, [dc_vals])
        full_r, full_p, full_n = partial_corr(bdr_vals, y, [dc_vals, ryptpa_vals])

        sig = "*" if full_p < 0.05 else " "
        print(f"{out:15s} | {raw_n:4.0f} | {raw_r:+7.3f} | {dc_r:+7.3f} | {full_r:+8.3f} | {full_p:8.4f}{sig}")

    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Three columns of correlation to read left-to-right:
  Raw r      = raw Spearman (BDR vs outcome)
  r|DC       = after controlling for draft capital only
  r|DC+RYP   = after controlling for BOTH DC and RYPTPA

The last column is the key question: does BDR add signal
BEYOND what draft capital + receiving production already capture?
If r|DC+RYP is significant (p < 0.05), BDR has independent value.
""")


if __name__ == "__main__":
    main()
