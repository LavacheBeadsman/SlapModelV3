"""One-time migration: standardize player-name suffix formatting.

Canonical format:
  - 'Jr.' / 'Sr.' with period
  - 'II' / 'III' / 'IV' / 'V' without period

Applied corrections:
  Kerrith Whyte Jr   -> Kerrith Whyte Jr.
  Gary Jennings Jr   -> Gary Jennings Jr.
  Chris Brazzell     -> Chris Brazzell II         (per 2026 draft sheet)
  Emmanuel Henderson -> Emmanuel Henderson Jr.    (per 2026 draft sheet)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CORRECTIONS = {
    "Kerrith Whyte Jr": "Kerrith Whyte Jr.",
    "Gary Jennings Jr": "Gary Jennings Jr.",
    "Chris Brazzell": "Chris Brazzell II",
    "Emmanuel Henderson": "Emmanuel Henderson Jr.",
}

# All source files that have a player_name column
TARGETS = [
    "data/wr_backtest_all_components.csv",
    "data/wr_backtest_with_production.csv",
    "data/wr_breakout_ages_2026.csv",
    "data/rb_backtest_with_receiving.csv",
    "data/te_backtest_master.csv",
    "data/te_2026_prospects_final.csv",
    "data/prospects_final.csv",
    "data/wr_teammate_scores.csv",
    "data/backtest_outcomes_complete.csv",
    "output/slap_v5_wr_2026.csv",
    "data/rb_dominator_scores.csv",
]


def main():
    total_changes = 0
    for rel in TARGETS:
        path = ROOT / rel
        if not path.exists():
            print(f"  skip (missing): {rel}")
            continue
        df = pd.read_csv(path)
        if "player_name" not in df.columns:
            print(f"  skip (no player_name col): {rel}")
            continue
        before = df["player_name"].copy()
        df["player_name"] = df["player_name"].replace(CORRECTIONS)
        changes = (before != df["player_name"]).sum()
        if changes > 0:
            df.to_csv(path, index=False)
            total_changes += changes
            for old, new in CORRECTIONS.items():
                if (before == old).any():
                    print(f"  {rel}: '{old}' -> '{new}'")
        else:
            print(f"  {rel}: no changes")
    print(f"\nTotal cell updates: {total_changes}")


if __name__ == "__main__":
    main()
