"""
similarity_engine.py — SimScores V5 similarity engine.
Compares 2026 NFL draft prospects against all backtest players (2015-2025)
within the same position using weighted Euclidean distance on z-scored profiles.

V5 changes from V4:
  - Variable set rebuilt around raw production stats + SLAP model components.
  - Position-specific variable additions:
    * WR: breakout_age, early_declare, teammate_score in Production
    * RB: bdr, speed_score in Size/Speed; rec_yards_per_team_pass_att in Production
    * TE: te_breakout_score, te_production_score in Production
  - Reads from single unified master database (output/slap_v5_master_database.csv).
  - Output includes first_3yr_ppg and category_breakdown.

Modes:
  --test        : Test on 5 specific players
  --prospect    : 2026 class vs backtest pool, save to output/sim_scores_2026.csv
  --backtest    : Backtest all 722 players

Usage:
    python src/sim_scores/similarity_engine.py --test
    python src/sim_scores/similarity_engine.py --prospect
    python src/sim_scores/similarity_engine.py --backtest
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger(__name__)

TOP_N = 10
MAX_DISTANCE = 4.0  # z-score distance that maps to similarity = 0

# ---------------------------------------------------------------------------
# Variable definitions per position
# ---------------------------------------------------------------------------

# Base production variables shared by all positions
PRODUCTION_VARS_SHARED = [
    "rec_yards",
    "receptions",
    "rec_tds",
    "rush_yards",
    "rush_att",
    "yards_per_reception",
    "team_pass_att",
    "games_played",
]

# WR adds these to Production (dominator_rating + peak_dominator are WR/TE concepts)
PRODUCTION_VARS_WR_EXTRA = [
    "dominator_rating",
    "peak_dominator",
    "breakout_age",
    "early_declare",
    "teammate_score",
]

# RB adds this to Production (no dominator_rating or peak_dominator — not meaningful for RBs)
PRODUCTION_VARS_RB_EXTRA = [
    "rec_yards_per_team_pass_att",
]

# TE adds these to Production (dominator_rating + peak_dominator apply to TEs too)
PRODUCTION_VARS_TE_EXTRA = [
    "dominator_rating",
    "peak_dominator",
    "te_breakout_score",
    "te_production_score",
]

# Size/Speed base (all positions)
SIZE_SPEED_BASE = [
    "height_in",
    "weight",
    "forty",
    "ras_score",
]

# RB adds these to Size/Speed
SIZE_SPEED_RB_EXTRA = [
    "bdr",
    "speed_score",
]

# Category configs per position: category → (weight, [variable names])
VAR_CONFIG_WR = {
    "draft_capital": (0.10, ["dc_score"]),
    "size_speed":    (0.25, SIZE_SPEED_BASE),
    "production":    (0.45, PRODUCTION_VARS_SHARED + PRODUCTION_VARS_WR_EXTRA),
    "slap_score":    (0.20, ["slap_score"]),
}

VAR_CONFIG_RB = {
    "draft_capital": (0.10, ["dc_score"]),
    "size_speed":    (0.25, SIZE_SPEED_BASE + SIZE_SPEED_RB_EXTRA),
    "production":    (0.45, PRODUCTION_VARS_SHARED + PRODUCTION_VARS_RB_EXTRA),
    "slap_score":    (0.20, ["slap_score"]),
}

VAR_CONFIG_TE = {
    "draft_capital": (0.10, ["dc_score"]),
    "size_speed":    (0.25, SIZE_SPEED_BASE),
    "production":    (0.45, PRODUCTION_VARS_SHARED + PRODUCTION_VARS_TE_EXTRA),
    "slap_score":    (0.20, ["slap_score"]),
}


def get_var_config(position: str) -> dict:
    """Return variable config for a position."""
    if position == "WR":
        return dict(VAR_CONFIG_WR)
    if position == "RB":
        return dict(VAR_CONFIG_RB)
    if position == "TE":
        return dict(VAR_CONFIG_TE)
    return dict(VAR_CONFIG_WR)  # fallback


def all_vars_for_position(position: str) -> list:
    """Return flat list of all variable names for a given position."""
    config = get_var_config(position)
    return [v for _, var_list in config.values() for v in var_list]


# ---------------------------------------------------------------------------
# Column name mapping: engine variable → master database column
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    # Draft capital & SLAP
    "dc_score":                    "dc_score",
    "slap_score":                  "slap_model_score",
    # Size/Speed
    "height_in":                   "height_in",
    "weight":                      "weight",
    "forty":                       "forty",
    "ras_score":                   "ras_score",
    # Production (base)
    "rec_yards":                   "rec_yards",
    "receptions":                  "receptions",
    "rec_tds":                     "rec_tds",
    "rush_yards":                  "rush_yards",
    "rush_att":                    "rush_attempts",
    "dominator_rating":            "dominator_rating",
    "peak_dominator":              "peak_dominator",
    "yards_per_reception":         "yards_per_reception",
    "team_pass_att":               "team_pass_att",
    "games_played":                "games_played",
    # WR extras
    "breakout_age":                "breakout_age",
    "early_declare":               "early_declare_score",
    "teammate_score":              "teammate_score",
    # RB extras
    "speed_score":                 "speed_score",
    "bdr":                         "bdr",
    "rec_yards_per_team_pass_att": "rec_yards_per_team_pass_att",
    # TE extras
    "te_breakout_score":           "te_breakout_score",
    "te_production_score":         "te_production_score",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the unified master database (output/slap_v5_master_database.csv).

    The master database already contains all SLAP scores, college stats,
    combine measurements, Anatomy variables, PFF grades, BDR, and NFL outcomes
    for 944 players (722 backtest + 217 2026 prospects + 5 UDFAs).
    No merging or derived-column computation needed.
    """
    master_path = OUTPUT_DIR / "slap_v5_master_database.csv"
    df = pd.read_csv(master_path)

    log.info(f"Loaded {len(df)} players from {master_path.name}")
    log.info(f"  Datasets: {df['dataset'].value_counts().to_dict()}")
    log.info(f"  Positions: {df['position'].value_counts().to_dict()}")

    # Coverage report per position
    for pos in ["WR", "RB", "TE"]:
        pos_df = df[df["position"] == pos]
        vars_list = all_vars_for_position(pos)
        coverage = {}
        for v in vars_list:
            src = COLUMN_MAP.get(v, v)
            if src in pos_df.columns:
                coverage[v] = pos_df[src].notna().sum()
            else:
                coverage[v] = 0
        log.info(f"  {pos} ({len(pos_df)} players, {len(vars_list)} vars): "
                 f"{', '.join(f'{k}={v}' for k, v in coverage.items())}")

    return df


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
@dataclass
class NormStats:
    """Z-score normalization parameters per variable per position."""
    means: dict = field(default_factory=dict)
    stds: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame):
        """Compute mean/std from backtest players for each (position, variable)."""
        bt = df[df["dataset"] == "backtest"]
        for pos in ["WR", "RB", "TE"]:
            pos_data = bt[bt["position"] == pos]
            for var in all_vars_for_position(pos):
                src = COLUMN_MAP.get(var, var)
                if src in pos_data.columns:
                    vals = pos_data[src].dropna()
                    if len(vals) >= 10:
                        self.means[(pos, var)] = vals.mean()
                        self.stds[(pos, var)] = vals.std(ddof=1)
                        if self.stds[(pos, var)] == 0:
                            self.stds[(pos, var)] = 1.0

    def transform(self, value: float, position: str, var: str) -> float:
        """Convert a raw value to z-score."""
        key = (position, var)
        if key not in self.means or np.isnan(value):
            return np.nan
        return (value - self.means[key]) / self.stds[key]


# ---------------------------------------------------------------------------
# Similarity calculation
# ---------------------------------------------------------------------------
@dataclass
class CompResult:
    """A single comparison result."""
    prospect_name: str
    prospect_position: str
    prospect_college: str
    prospect_pick: int
    prospect_year: int
    comp_name: str
    comp_college: str
    comp_pick: int
    comp_year: int
    similarity_score: float
    comparison_coverage: float
    dominant_category: str
    category_distances: dict
    # NFL outcomes
    nfl_hit24: float
    nfl_hit12: float
    nfl_career_ppg: float
    nfl_first_3yr_ppg: float
    nfl_best_season_ppg: float


def compute_similarity(
    prospect_z: dict,
    comp_z: dict,
    var_config: dict,
) -> tuple:
    """
    Compute weighted Euclidean distance between two z-scored profiles.

    Only uses variables where BOTH players have data.
    Per-variable weight = category_weight / len(full_var_list).

    Returns:
        (similarity_score, comparison_coverage, dominant_category, category_distances)
    """
    total_weight_available = 0.0
    total_weight_used = 0.0
    weighted_sq_distance = 0.0
    category_distances = {}

    for cat, (cat_weight, var_list) in var_config.items():
        total_weight_available += cat_weight

        # Find shared variables (both have data)
        shared = [v for v in var_list
                  if v in prospect_z and v in comp_z
                  and not np.isnan(prospect_z[v])
                  and not np.isnan(comp_z[v])]

        if not shared:
            category_distances[cat] = None
            continue

        per_var_weight = cat_weight / len(var_list)
        cat_sq_dist = 0.0
        cat_weight_used = 0.0

        for v in shared:
            diff = prospect_z[v] - comp_z[v]
            cat_sq_dist += diff ** 2
            cat_weight_used += per_var_weight

        total_weight_used += cat_weight_used
        weighted_sq_distance += cat_sq_dist * per_var_weight

        # Average per-variable distance in this category
        category_distances[cat] = np.sqrt(cat_sq_dist / len(shared))

    if total_weight_used == 0:
        return 0.0, 0.0, "none", category_distances

    distance = np.sqrt(weighted_sq_distance / total_weight_used)
    coverage = total_weight_used / total_weight_available
    score = max(0.0, 100.0 * (1.0 - distance / MAX_DISTANCE))

    # Dominant category = highest average per-var distance
    dominant = max(
        ((cat, d) for cat, d in category_distances.items() if d is not None),
        key=lambda x: x[1],
        default=("none", 0),
    )[0]

    return score, coverage, dominant, category_distances


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class SimScoresEngine:
    """Main similarity engine."""

    def __init__(self, top_n: int = TOP_N):
        self.top_n = top_n
        self.df = None
        self.norm = NormStats()
        self.z_profiles = {}  # (name, pos, year) → {var: z_score}

    def load_and_prepare(self):
        """Load data, fit normalization, compute z-profiles for all players."""
        self.df = load_data()
        self.norm.fit(self.df)

        for _, row in self.df.iterrows():
            pos = row["position"]
            name = row["player_name"]
            year = int(row["draft_year"])
            key = (name, pos, year)

            z_profile = {}
            for var in all_vars_for_position(pos):
                src = COLUMN_MAP.get(var, var)
                if src in row.index and pd.notna(row[src]):
                    z = self.norm.transform(float(row[src]), pos, var)
                    if not np.isnan(z):
                        z_profile[var] = z

            self.z_profiles[key] = z_profile

        log.info(f"Computed z-profiles for {len(self.z_profiles)} players")

        # Report var counts per position
        for pos in ["WR", "RB", "TE"]:
            keys = [k for k in self.z_profiles if k[1] == pos]
            avg_vars = np.mean([len(self.z_profiles[k]) for k in keys]) if keys else 0
            total_vars = len(all_vars_for_position(pos))
            log.info(f"  {pos}: avg {avg_vars:.1f}/{total_vars} variables per player")

    def _get_pick(self, row) -> int:
        """Get draft pick from a row, handling NaN."""
        val = row.get("draft_pick", np.nan)
        if pd.notna(val):
            return int(val)
        return 0

    def _find_comps_for_row(self, player_row, comp_pool,
                             exclude_self=True, exclude_same_class=False) -> list:
        """Find top N most similar players from comp_pool."""
        name = player_row["player_name"]
        pos = player_row["position"]
        p_year = int(player_row["draft_year"])
        p_key = (name, pos, p_year)
        p_z = self.z_profiles.get(p_key, {})

        if not p_z:
            return []

        var_config = get_var_config(pos)
        results = []

        for _, comp_row in comp_pool.iterrows():
            c_name = comp_row["player_name"]
            c_year = int(comp_row["draft_year"])
            if exclude_self and c_name == name and c_year == p_year:
                continue
            if exclude_same_class and c_year == p_year:
                continue

            c_key = (c_name, pos, c_year)
            c_z = self.z_profiles.get(c_key, {})
            if not c_z:
                continue

            score, coverage, dominant, cat_dists = compute_similarity(p_z, c_z, var_config)

            results.append(CompResult(
                prospect_name=name,
                prospect_position=pos,
                prospect_college=player_row.get("college", ""),
                prospect_pick=self._get_pick(player_row),
                prospect_year=p_year,
                comp_name=c_name,
                comp_college=comp_row.get("college", ""),
                comp_pick=self._get_pick(comp_row),
                comp_year=c_year,
                similarity_score=round(score, 1),
                comparison_coverage=round(coverage, 3),
                dominant_category=dominant,
                category_distances=cat_dists,
                nfl_hit24=comp_row.get("nfl_hit24", np.nan),
                nfl_hit12=comp_row.get("nfl_hit12", np.nan),
                nfl_career_ppg=comp_row.get("nfl_career_ppg", np.nan),
                nfl_first_3yr_ppg=comp_row.get("nfl_first_3yr_ppg", np.nan),
                nfl_best_season_ppg=comp_row.get("nfl_best_ppg", np.nan),
            ))

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:self.top_n]

    def find_comps(self, prospect_name: str) -> list:
        """Find top N most similar backtest players for a 2026 prospect."""
        prospect_row = self.df[self.df["player_name"] == prospect_name]
        if prospect_row.empty:
            log.warning(f"Prospect '{prospect_name}' not found")
            return []
        prospect_row = prospect_row.iloc[0]
        pos = prospect_row["position"]
        bt = self.df[(self.df["dataset"] == "backtest") & (self.df["position"] == pos)]
        return self._find_comps_for_row(prospect_row, bt)

    def find_backtest_comps(self, player_name: str, draft_year: int) -> list:
        """Find top N comps for a backtest player, excluding self and same class."""
        mask = (self.df["player_name"] == player_name) & (self.df["draft_year"] == draft_year)
        player_row = self.df[mask]
        if player_row.empty:
            log.warning(f"Player '{player_name}' ({draft_year}) not found")
            return []
        player_row = player_row.iloc[0]
        pos = player_row["position"]
        bt = self.df[(self.df["dataset"] == "backtest") & (self.df["position"] == pos)]
        return self._find_comps_for_row(player_row, bt,
                                         exclude_self=True, exclude_same_class=True)

    def run_all_prospects(self) -> pd.DataFrame:
        """Run SimScores for all 2026 prospects."""
        prospects = self.df[self.df["dataset"] == "2026_prospect"]
        return self._run_batch(prospects, mode="prospect")

    def run_backtest(self) -> pd.DataFrame:
        """Run SimScores for all backtest players."""
        bt = self.df[self.df["dataset"] == "backtest"]
        return self._run_batch(bt, mode="backtest")

    def _run_batch(self, players_df, mode="prospect") -> pd.DataFrame:
        """Run comps for a batch of players and return DataFrame."""
        all_results = []
        total = len(players_df)

        for i, (_, row) in enumerate(players_df.iterrows()):
            name = row["player_name"]
            year = int(row["draft_year"])
            if mode == "backtest":
                comps = self.find_backtest_comps(name, year)
            else:
                comps = self.find_comps(name)

            if (i + 1) % 50 == 0:
                log.info(f"  Processed {i + 1}/{total} players...")

            for rank, comp in enumerate(comps, 1):
                # Format category breakdown
                cat_parts = []
                for cat, dist in comp.category_distances.items():
                    if dist is not None:
                        cat_parts.append(f"{cat}={dist:.2f}")
                cat_breakdown = "; ".join(cat_parts)

                all_results.append({
                    "prospect": comp.prospect_name,
                    "position": comp.prospect_position,
                    "prospect_college": comp.prospect_college,
                    "prospect_pick": comp.prospect_pick,
                    "prospect_year": comp.prospect_year,
                    "comp_rank": rank,
                    "comp_player": comp.comp_name,
                    "comp_college": comp.comp_college,
                    "comp_pick": comp.comp_pick,
                    "comp_year": comp.comp_year,
                    "similarity_score": comp.similarity_score,
                    "comparison_coverage": comp.comparison_coverage,
                    "dominant_category": comp.dominant_category,
                    "category_breakdown": cat_breakdown,
                    "hit24": comp.nfl_hit24,
                    "hit12": comp.nfl_hit12,
                    "career_ppg": comp.nfl_career_ppg,
                    "first_3yr_ppg": comp.nfl_first_3yr_ppg,
                    "nfl_best_season_ppg": comp.nfl_best_season_ppg,
                })

        return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _print_comp_table(comps: list, engine: SimScoresEngine):
    """Print formatted comp table with z-profile and NFL outcomes."""
    if not comps:
        return

    p = comps[0]
    pos = p.prospect_position
    p_key = (p.prospect_name, pos, p.prospect_year)
    p_z = engine.z_profiles.get(p_key, {})

    config = get_var_config(pos)

    print(f"  Z-profile ({len(p_z)} vars):")
    for cat, (wt, var_list) in config.items():
        vals = []
        for v in var_list:
            if v in p_z:
                short = v[:16]
                vals.append(f"{short}={p_z[v]:+.2f}")
        if vals:
            print(f"    {cat} ({wt:.0%}): {', '.join(vals)}")

    print(f"\n  {'#':>2} {'Sim':>5} {'Cov':>5} {'Player':<28} {'College':<20} {'Pick':>5} {'Year':>5} "
          f"{'Hit24':>6} {'CarPPG':>7} {'3yrPPG':>7} {'BestPPG':>8} {'Driver'}")
    print(f"  {'-'*2} {'-'*5} {'-'*5} {'-'*28} {'-'*20} {'-'*5} {'-'*5} "
          f"{'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*20}")

    for i, c in enumerate(comps, 1):
        hit_str = f"{int(c.nfl_hit24)}" if pd.notna(c.nfl_hit24) else "-"
        cppg = f"{c.nfl_career_ppg:.1f}" if pd.notna(c.nfl_career_ppg) else "-"
        tppg = f"{c.nfl_first_3yr_ppg:.1f}" if pd.notna(c.nfl_first_3yr_ppg) else "-"
        bppg = f"{c.nfl_best_season_ppg:.1f}" if pd.notna(c.nfl_best_season_ppg) else "-"
        print(f"  {i:>2} {c.similarity_score:>5.1f} {c.comparison_coverage:>5.0%} "
              f"{c.comp_name:<28} {c.comp_college:<20} {c.comp_pick:>5} {c.comp_year:>5} "
              f"{hit_str:>6} {cppg:>7} {tppg:>7} {bppg:>8} {c.dominant_category}")


def print_comps(engine: SimScoresEngine, prospect_name: str):
    """Print formatted comp results for a 2026 prospect."""
    comps = engine.find_comps(prospect_name)
    if not comps:
        print(f"  No comps found for {prospect_name}")
        return

    p = comps[0]
    print(f"\n{'='*130}")
    print(f"  {prospect_name} ({p.prospect_position}, {p.prospect_college}, "
          f"Pick #{p.prospect_pick}, {p.prospect_year})")
    print(f"{'='*130}")
    _print_comp_table(comps, engine)


def print_comps_backtest(engine: SimScoresEngine, player_name: str, draft_year: int):
    """Print formatted comp results for a backtest player."""
    comps = engine.find_backtest_comps(player_name, draft_year)
    if not comps:
        print(f"  No comps found for {player_name} ({draft_year})")
        return

    p = comps[0]
    row = engine.df[(engine.df["player_name"] == player_name) &
                     (engine.df["draft_year"] == draft_year)].iloc[0]
    hit_str = f"Hit24={int(row['nfl_hit24'])}" if pd.notna(row.get("nfl_hit24")) else "Hit24=?"
    ppg_str = f"CarPPG={row['nfl_career_ppg']:.1f}" if pd.notna(row.get("nfl_career_ppg")) else ""

    print(f"\n{'='*130}")
    print(f"  {player_name} ({p.prospect_position}, Pick #{p.prospect_pick}, {p.prospect_year})  "
          f"[ACTUAL: {hit_str}  {ppg_str}]")
    print(f"{'='*130}")
    _print_comp_table(comps, engine)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SimScores V5 Similarity Engine")
    parser.add_argument("--backtest", action="store_true",
                        help="Backtest mode: all backtest players")
    parser.add_argument("--prospect", action="store_true",
                        help="Prospect mode: 2026 class vs backtest pool")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 5 specific players")
    parser.add_argument("--top", type=int, default=TOP_N,
                        help="Number of comps per prospect")
    args = parser.parse_args()

    engine = SimScoresEngine(top_n=args.top)
    engine.load_and_prepare()

    if args.test:
        test_players = [
            "Carnell Tate",
            "Jordyn Tyson",
            "Jeremiyah Love",
            "Mike Washington Jr.",
            "Kenyon Sadiq",
        ]
        for name in test_players:
            print_comps(engine, name)

    elif args.backtest:
        print("Running full backtest...")
        results = engine.run_backtest()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "sim_scores_backtest_v5.csv"
        results.to_csv(out_path, index=False)
        log.info(f"Saved {len(results)} rows to {out_path}")

    else:
        # Default: prospect mode
        test_players = [
            "Carnell Tate",
            "Jordyn Tyson",
            "Jeremiyah Love",
            "Mike Washington Jr.",
            "Kenyon Sadiq",
        ]
        for name in test_players:
            print_comps(engine, name)

        if args.prospect:
            print(f"\n\n{'='*130}")
            print("Running full 2026 class...")
            results = engine.run_all_prospects()
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / "sim_scores_2026.csv"
            results.to_csv(out_path, index=False)
            log.info(f"Saved {len(results)} rows to {out_path}")

            prospects = results["prospect"].nunique()
            avg_score = results[results["comp_rank"] == 1]["similarity_score"].mean()
            avg_cov = results[results["comp_rank"] == 1]["comparison_coverage"].mean()
            print(f"\nSummary: {prospects} prospects, "
                  f"avg top-1 similarity={avg_score:.1f}, avg coverage={avg_cov:.1%}")


if __name__ == "__main__":
    main()
