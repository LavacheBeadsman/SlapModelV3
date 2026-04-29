"""Rigorous validation with bootstrap 95% CIs and 5-fold CV.

Generates a publication-quality validation report covering:
  1. In-sample correlations (full backtest) with 1000-iteration bootstrap CIs
  2. 5-fold cross-validation: train on 4 folds, evaluate on the held-out fold,
     repeat for all 5 folds, then average
  3. Per-seed sensitivity: also runs 5 different random seeds for a single 80/20
     split so readers can see how much the holdout estimate varies with the
     specific random draw

Caveat: SLAP V5 weights were tuned on the full backtest, so neither the holdout
nor the CV demonstrates true out-of-sample weight generalization (we cannot
refit weights without the original tuning code). What this DOES demonstrate is
prediction stability: if the model's predictions correlate with NFL outcomes
similarly across random subsets, the published correlation is reliable.

Output: writes a clean text report to output/validation_with_ci_and_holdout.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent

N_BOOTSTRAP = 1000
N_FOLDS = 5
SEED_LIST = [42, 7, 13, 99, 2024]  # for sensitivity analysis on 80/20 split
HOLDOUT_FRAC = 0.20

OUTCOMES = [
    ('nfl_first_3yr_ppg', 'First 3-year PPG'),
    ('nfl_career_ppg',    'Career PPG'),
    ('nfl_hit24',         'Top-24 hit rate'),
    ('nfl_hit12',         'Top-12 hit rate'),
    ('nfl_seasons_10ppg_3yr', 'Seasons over 10 PPG (3yr)'),
]

PREDICTORS = [
    ('slap_model_score', 'SLAP V5 model'),
    ('dc_score',         'DC alone'),
]

_rng = np.random.default_rng(42)


def bootstrap_spearman_ci(x, y, n=N_BOOTSTRAP, alpha=0.05):
    """Returns (point_estimate, ci_low, ci_high)."""
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    point = spearmanr(x, y)[0]
    n_samples = len(x)
    rs = np.empty(n)
    for i in range(n):
        idx = _rng.choice(n_samples, size=n_samples, replace=True)
        rs[i] = spearmanr(x[idx], y[idx])[0]
    rs = rs[~np.isnan(rs)]
    return point, np.percentile(rs, 100 * alpha / 2), np.percentile(rs, 100 * (1 - alpha / 2))


def stratified_split(df, frac=HOLDOUT_FRAC, seed=42):
    """80/20 split stratified by draft_year."""
    rng_local = np.random.default_rng(seed)
    test_idx = []
    for _, group in df.groupby('draft_year'):
        n = len(group)
        n_test = max(1, int(round(n * frac)))
        sampled = rng_local.choice(group.index.values, size=n_test, replace=False)
        test_idx.extend(sampled.tolist())
    train_idx = [i for i in df.index if i not in set(test_idx)]
    return train_idx, test_idx


def stratified_kfold(df, n_folds=N_FOLDS, seed=42):
    """K-fold split stratified by draft_year. Returns list of (train_idx, test_idx) tuples."""
    rng_local = np.random.default_rng(seed)
    fold_assignments = pd.Series(-1, index=df.index)
    for _, group in df.groupby('draft_year'):
        idx = group.index.values.copy()
        rng_local.shuffle(idx)
        for k, i in enumerate(idx):
            fold_assignments.loc[i] = k % n_folds
    splits = []
    for k in range(n_folds):
        test = df.index[fold_assignments == k].tolist()
        train = df.index[fold_assignments != k].tolist()
        splits.append((train, test))
    return splits


def evaluate(df, predictor_col, outcome_col):
    sub = df[[predictor_col, outcome_col]].dropna()
    if len(sub) < 5:
        return {'n': len(sub), 'r': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}
    r, lo, hi = bootstrap_spearman_ci(sub[predictor_col].values, sub[outcome_col].values)
    return {'n': len(sub), 'r': r, 'ci_lo': lo, 'ci_hi': hi}


def position_report(df, pos_name, fh):
    fh.write(f"\n{'='*78}\n")
    fh.write(f"  {pos_name}  (n_total = {len(df)})\n")
    fh.write(f"{'='*78}\n")

    # ---- 1) Full-sample correlations with 95% CIs ----
    fh.write(f"\n  1) FULL-SAMPLE correlations with 95% bootstrap CIs (n_iter={N_BOOTSTRAP})\n\n")
    for pred_col, pred_label in PREDICTORS:
        fh.write(f"  --- {pred_label} ---\n")
        fh.write(f"  {'Outcome':<32} {'n':>4}  {'r (95% CI)':>22}\n")
        fh.write(f"  {'-'*64}\n")
        for out_col, out_label in OUTCOMES:
            m = evaluate(df, pred_col, out_col)
            r_str = f"{m['r']:+.3f} [{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}]" if not np.isnan(m['r']) else 'n/a'
            fh.write(f"  {out_label:<32} {m['n']:>4}  {r_str:>22}\n")
        fh.write("\n")

    # ---- 2) 5-fold cross-validation ----
    fh.write(f"\n  2) {N_FOLDS}-FOLD CROSS-VALIDATION (stratified by draft year)\n")
    fh.write(f"     Each fold serves as the held-out test once; reports mean ± std across folds.\n\n")
    splits = stratified_kfold(df)
    for pred_col, pred_label in PREDICTORS:
        fh.write(f"  --- {pred_label} ---\n")
        fh.write(f"  {'Outcome':<32} {'n_test':>7}  {'r_mean':>7}  {'r_std':>7}  {'r_min':>7}  {'r_max':>7}\n")
        fh.write(f"  {'-'*72}\n")
        for out_col, out_label in OUTCOMES:
            fold_rs = []
            test_ns = []
            for _, test_idx in splits:
                test = df.loc[test_idx]
                sub = test[[pred_col, out_col]].dropna()
                if len(sub) >= 5:
                    r = spearmanr(sub[pred_col].values, sub[out_col].values)[0]
                    if not np.isnan(r):
                        fold_rs.append(r)
                        test_ns.append(len(sub))
            if fold_rs:
                avg_n = int(np.mean(test_ns))
                fh.write(f"  {out_label:<32} {avg_n:>7}  {np.mean(fold_rs):+7.3f}  "
                         f"{np.std(fold_rs):>7.3f}  {min(fold_rs):+7.3f}  {max(fold_rs):+7.3f}\n")
            else:
                fh.write(f"  {out_label:<32}  insufficient data\n")
        fh.write("\n")

    # ---- 3) Sensitivity to seed: single 80/20 split with 5 different seeds ----
    fh.write(f"\n  3) SEED SENSITIVITY: single stratified 80/20 split, 5 different random seeds\n")
    fh.write(f"     Wide spread = high sensitivity to specific random sample.\n\n")
    fh.write(f"  Predictor       Outcome          Seed=42  Seed=7   Seed=13  Seed=99  Seed=2024\n")
    fh.write(f"  {'-'*82}\n")
    for pred_col, pred_label in PREDICTORS:
        for out_col, out_label in [('nfl_career_ppg', 'Career PPG'), ('nfl_hit24', 'Top-24 hit rate')]:
            row_vals = []
            for seed in SEED_LIST:
                _, test_idx = stratified_split(df, seed=seed)
                test = df.loc[test_idx]
                sub = test[[pred_col, out_col]].dropna()
                if len(sub) >= 5:
                    r = spearmanr(sub[pred_col].values, sub[out_col].values)[0]
                    row_vals.append(f"{r:+.3f}")
                else:
                    row_vals.append("  n/a ")
            label = pred_label[:14]
            fh.write(f"  {label:<14}  {out_label:<14}   " + "  ".join(f"{v:>7}" for v in row_vals) + "\n")
        fh.write("\n")


def main():
    df = pd.read_csv(ROOT / 'output' / 'slap_v5_master_database.csv')
    bt = df[df['dataset'] == 'backtest'].copy().reset_index(drop=True)
    print(f"Loaded {len(bt)} backtest rows")
    print(f"Bootstrap n={N_BOOTSTRAP}, folds={N_FOLDS}")

    out_path = ROOT / 'output' / 'validation_with_ci_and_holdout.txt'
    with open(out_path, 'w') as fh:
        fh.write("SLAP V5 — Validation Report (Bootstrap CIs + Cross-Validation)\n")
        fh.write("=" * 78 + "\n")
        fh.write(f"Generated:               {pd.Timestamp.now().date().isoformat()}\n")
        fh.write(f"Bootstrap iterations:    {N_BOOTSTRAP}\n")
        fh.write(f"Cross-validation folds:  {N_FOLDS} (stratified by draft year)\n")
        fh.write(f"Holdout fraction:        {HOLDOUT_FRAC}\n")
        fh.write(f"Seeds tested:            {SEED_LIST}\n")
        fh.write("\n")
        fh.write("METHODOLOGY:\n")
        fh.write("- Spearman rank correlation (rank-based, robust to outliers)\n")
        fh.write("- 95% CIs: 1000-iteration percentile bootstrap (resample with replacement)\n")
        fh.write("- 5-fold CV: each fold held out once, mean ± std reported across folds\n")
        fh.write("- Seed sensitivity: 5 different random seeds for a single 80/20 split\n")
        fh.write("\n")
        fh.write("CAVEAT: V5 weights were tuned on the full backtest, so this validates the\n")
        fh.write("STABILITY of the predictions across random subsets — not true out-of-sample\n")
        fh.write("weight generalization (which would require re-tuning weights on each train fold).\n")

        for pos_name in ['WR', 'RB', 'TE']:
            sub = bt[bt['position'] == pos_name].copy().reset_index(drop=True)
            position_report(sub, f'{pos_name} backtest', fh)

    print(f"\nReport written: {out_path}")
    print()
    print(open(out_path).read())


if __name__ == '__main__':
    main()
