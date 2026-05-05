"""Restore breakout values from main where my recent recompute downgraded them.

Policy:
  - Use earliest breakout_age across (main, current PBP-computed)
  - Use highest peak_dominator across (main, current PBP-computed)
  - Never downgrade to NaN if main had a value
  - Keep my recompute when it found an EARLIER breakout (transfer history)

Affects WR 2026 and TE 2026 prospect files.
"""
import json
import subprocess
from pathlib import Path
import io

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def get_main_file(path):
    r = subprocess.run(['git', 'show', f'main:{path}'], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return None
    return pd.read_csv(io.StringIO(r.stdout))


def merge_wr_2026():
    print("="*78)
    print("WR 2026 — merge main with current (no downgrade policy)")
    print("="*78)
    cur_path = ROOT / 'data' / 'wr_breakout_ages_2026.csv'
    cur = pd.read_csv(cur_path)
    main = get_main_file('data/wr_breakout_ages_2026.csv')
    if main is None:
        print("  Could not load main version")
        return

    cur = cur.set_index('player_name')
    main = main.set_index('player_name')
    common = cur.index.intersection(main.index)

    n_kept_main = 0
    n_kept_current = 0
    n_used_main_pd = 0

    for name in common:
        m_bo = main.loc[name, 'breakout_age']
        c_bo = cur.loc[name, 'breakout_age']
        m_pd = main.loc[name, 'peak_dominator']
        c_pd = cur.loc[name, 'peak_dominator']

        # breakout_age: take earliest (lowest) of the two (current may have caught earlier transfer)
        if pd.notna(m_bo) and pd.notna(c_bo):
            new_bo = min(m_bo, c_bo)
        elif pd.notna(m_bo):
            # Current is NaN but main had value — restore main (don't downgrade)
            new_bo = m_bo
            n_kept_main += 1
        elif pd.notna(c_bo):
            new_bo = c_bo
            n_kept_current += 1
        else:
            new_bo = np.nan

        # peak_dominator: take MAX (don't lose info)
        if pd.notna(m_pd) and pd.notna(c_pd):
            new_pd = max(m_pd, c_pd)
        elif pd.notna(m_pd):
            new_pd = m_pd
            n_used_main_pd += 1
        elif pd.notna(c_pd):
            new_pd = c_pd
        else:
            new_pd = np.nan

        if cur.loc[name, 'breakout_age'] != new_bo and not (pd.isna(cur.loc[name, 'breakout_age']) and pd.isna(new_bo)):
            cur.loc[name, 'breakout_age'] = new_bo
        if cur.loc[name, 'peak_dominator'] != new_pd and not (pd.isna(cur.loc[name, 'peak_dominator']) and pd.isna(new_pd)):
            cur.loc[name, 'peak_dominator'] = new_pd

    cur.reset_index().to_csv(cur_path, index=False)
    print(f"  Restored {n_kept_main} breakout_age values from main (current was NaN)")
    print(f"  Restored {n_used_main_pd} peak_dominator values from main")


def merge_te_2026():
    print("\n" + "="*78)
    print("TE 2026 — merge main with current (no downgrade policy)")
    print("="*78)
    cur_path = ROOT / 'data' / 'te_2026_prospects_final.csv'
    cur = pd.read_csv(cur_path)
    main = get_main_file('data/te_2026_prospects_final.csv')
    if main is None:
        print("  Could not load main version")
        return

    cur = cur.set_index('player_name')
    main = main.set_index('player_name')
    common = cur.index.intersection(main.index)

    n_kept_main = 0
    n_kept_current = 0
    n_used_main_pd = 0

    for name in common:
        m_bo = main.loc[name, 'breakout_age']
        c_bo = cur.loc[name, 'breakout_age']
        m_pd = main.loc[name, 'peak_dominator']
        c_pd = cur.loc[name, 'peak_dominator']

        # earliest breakout_age
        if pd.notna(m_bo) and pd.notna(c_bo):
            new_bo = min(m_bo, c_bo)
        elif pd.notna(m_bo):
            new_bo = m_bo
            n_kept_main += 1
        elif pd.notna(c_bo):
            new_bo = c_bo
            n_kept_current += 1
        else:
            new_bo = np.nan

        # highest peak_dominator
        if pd.notna(m_pd) and pd.notna(c_pd):
            new_pd = max(m_pd, c_pd)
        elif pd.notna(m_pd):
            new_pd = m_pd
            n_used_main_pd += 1
        elif pd.notna(c_pd):
            new_pd = c_pd
        else:
            new_pd = np.nan

        cur.loc[name, 'breakout_age'] = new_bo
        cur.loc[name, 'peak_dominator'] = new_pd

    cur.reset_index().to_csv(cur_path, index=False)
    print(f"  Restored {n_kept_main} breakout_age values from main")
    print(f"  Restored {n_used_main_pd} peak_dominator values from main")


def merge_te_backtest():
    print("\n" + "="*78)
    print("TE backtest — merge main with current (no downgrade policy)")
    print("="*78)
    cur_path = ROOT / 'data' / 'te_backtest_master.csv'
    cur = pd.read_csv(cur_path)
    main = get_main_file('data/te_backtest_master.csv')
    if main is None:
        print("  Could not load main version")
        return

    cur = cur.set_index('player_name')
    main = main.set_index('player_name')
    common = cur.index.intersection(main.index)

    n_kept_main = 0
    n_kept_current = 0
    n_used_main_pd = 0

    for name in common:
        m_bo = main.loc[name, 'breakout_age']
        c_bo = cur.loc[name, 'breakout_age']
        m_pd = main.loc[name, 'peak_dominator']
        c_pd = cur.loc[name, 'peak_dominator']

        if pd.notna(m_bo) and pd.notna(c_bo):
            new_bo = min(m_bo, c_bo)
        elif pd.notna(m_bo):
            new_bo = m_bo
            n_kept_main += 1
        elif pd.notna(c_bo):
            new_bo = c_bo
            n_kept_current += 1
        else:
            new_bo = np.nan

        if pd.notna(m_pd) and pd.notna(c_pd):
            new_pd = max(m_pd, c_pd)
        elif pd.notna(m_pd):
            new_pd = m_pd
            n_used_main_pd += 1
        elif pd.notna(c_pd):
            new_pd = c_pd
        else:
            new_pd = np.nan

        cur.loc[name, 'breakout_age'] = new_bo
        cur.loc[name, 'peak_dominator'] = new_pd

    cur.reset_index().to_csv(cur_path, index=False)
    print(f"  Restored {n_kept_main} breakout_age values from main")
    print(f"  Restored {n_used_main_pd} peak_dominator values from main")


if __name__ == '__main__':
    merge_wr_2026()
    merge_te_2026()
    merge_te_backtest()
    print("\nDone. Now run: python src/build_master_database_v5.py")
