"""
Merge combine_2026.csv measurements into output files:
- output/slap_v5_master_database.csv
- output/slap_v5_wr.csv
- output/slap_v5_rb.csv
- output/slap_v5_te.csv

Only 2026 prospects get combine data merged. Backtest rows are untouched.
No SLAP scores are changed — this adds informational columns only.
"""

import pandas as pd
import numpy as np

# ── Load combine data ────────────────────────────────────────────────────
combine = pd.read_csv("data/combine_2026.csv")
print(f"Combine data: {len(combine)} players")

# Columns to merge (all the body/athletic measurements)
combine_cols = [
    "height_in", "weight", "hand_size", "arm_length", "wingspan",
    "forty", "ten_yard_split", "vertical_jump", "broad_jump_inches",
    "bench_press", "three_cone", "twenty_yard_shuttle",
    "ras_unofficial",
]

# ── Name normalization ──────────────────────────────────────────────────

def normalize_name(n):
    """Normalize for fuzzy matching."""
    n = str(n).strip().lower()
    n = n.replace(".", "").replace("'", "").replace("'", "").replace("'", "")
    n = n.replace("-", " ").replace("  ", " ")
    # Remove suffixes
    for suffix in [" jr", " ii", " iii", " iv"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)].strip()
    return n

# Name aliases: output_name → combine_name (handles spelling differences)
NAME_ALIASES = {
    "jmichael sturdivant": "j michael sturdivant",
    "jamarion miller": "jam miller",
    "matthew hibner": "matt hibner",
}

combine["_norm"] = combine["player_name"].apply(normalize_name)

# ── Merge function ──────────────────────────────────────────────────────

def merge_combine_into(output_path, position_filter=None):
    """Add combine columns to an output file for 2026 prospects."""
    df = pd.read_csv(output_path)
    orig_rows = len(df)
    orig_cols = list(df.columns)

    # Add combine columns (NaN for all initially)
    for col in combine_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Only merge into 2026 prospects
    prospects = df[df["dataset"] == "2026_prospect"].copy()
    prospects["_norm"] = prospects["player_name"].apply(normalize_name)

    matched = 0
    for idx, row in prospects.iterrows():
        norm = row["_norm"]
        # Check aliases first
        lookup = NAME_ALIASES.get(norm, norm)
        # Try exact match
        match = combine[combine["_norm"] == lookup]
        if match.empty and lookup != norm:
            match = combine[combine["_norm"] == norm]
        if match.empty:
            # Try partial match (first + last name substring)
            parts = norm.split()
            if len(parts) >= 2:
                match = combine[combine["_norm"].str.contains(parts[0]) & combine["_norm"].str.contains(parts[-1])]
                if len(match) > 1 and position_filter:
                    match = match[match["position"] == position_filter]
        if match.empty:
            continue

        matched += 1
        match_row = match.iloc[0]
        for col in combine_cols:
            if pd.notna(match_row.get(col)):
                df.loc[idx, col] = match_row[col]

    # Verify row count unchanged
    assert len(df) == orig_rows, f"Row count changed! {orig_rows} -> {len(df)}"

    # Save
    df.to_csv(output_path, index=False)
    total_2026 = len(prospects)
    print(f"{output_path}: {matched}/{total_2026} prospects matched, {len(df)} rows, {len(df.columns)} cols")

# ── Run merges ──────────────────────────────────────────────────────────

merge_combine_into("output/slap_v5_wr.csv", "WR")
merge_combine_into("output/slap_v5_rb.csv", "RB")
merge_combine_into("output/slap_v5_te.csv", "TE")
merge_combine_into("output/slap_v5_master_database.csv")

# ── Verify ──────────────────────────────────────────────────────────────
print("\n── Verification ──")
for f in ["output/slap_v5_wr.csv", "output/slap_v5_rb.csv", "output/slap_v5_te.csv", "output/slap_v5_master_database.csv"]:
    df = pd.read_csv(f)
    prospects = df[df["dataset"] == "2026_prospect"]
    has_height = prospects["height_in"].notna().sum() if "height_in" in df.columns else 0
    has_forty = prospects["forty"].notna().sum() if "forty" in df.columns else 0
    has_weight = prospects["weight"].notna().sum() if "weight" in df.columns else 0
    print(f"  {f}: {len(df)} rows, {len(df.columns)} cols | 2026: height={has_height}, weight={has_weight}, forty={has_forty}")
