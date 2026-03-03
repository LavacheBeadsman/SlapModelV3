"""build_html_explorer.py — Generate SimScores Explorer HTML page.

Reads the master database + sim score files and produces a single
self-contained HTML file with embedded data, search, and interactive
comp tables.  No server or external dependencies required.

Usage:
    python src/sim_scores/build_html_explorer.py
"""

import json
import math
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Columns to include in the player profile JSON
PROFILE_COLS = {
    "common": [
        "player_name", "position", "college", "draft_year", "draft_pick",
        "draft_round", "dataset",
        "slap_display_score", "slap_model_score", "dc_score", "prospect_profile",
        "height_in", "weight", "forty", "ras_score",
        "rec_yards", "receptions", "rec_tds", "rush_yards", "rush_attempts",
        "yards_per_reception", "team_pass_att", "games_played", "draft_age",
        "peak_dominator", "rec_yards_per_team_pass_att",
    ],
    "WR": [
        "breakout_age", "peak_dominator", "dominator_rating",
        "early_declare_score", "teammate_score",
        "breakout_score",
    ],
    "RB": [
        "speed_score", "bdr", "rec_yards_per_team_pass_att",
        "yards_per_carry", "rec_tds",
    ],
    "TE": [
        "te_breakout_score", "te_production_score",
        "breakout_age", "peak_dominator",
    ],
    "nfl": [
        "nfl_hit24", "nfl_hit12",
        "nfl_career_ppg", "nfl_first_3yr_ppg", "nfl_best_ppg",
        "nfl_games", "nfl_seasons_10ppg",
    ],
}


def _safe_val(v):
    """Convert value for JSON: NaN/inf → None, numpy int → int, etc."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (int,)):
        return int(v)
    if isinstance(v, float):
        return round(v, 2)
    return v


def load_players() -> list[dict]:
    """Load player profiles from master database."""
    df = pd.read_csv(OUTPUT_DIR / "slap_v5_master_database.csv")

    players = []
    for _, row in df.iterrows():
        pos = row["position"]
        cols = list(PROFILE_COLS["common"])
        cols += PROFILE_COLS.get(pos, [])
        cols += PROFILE_COLS["nfl"]
        # Deduplicate while preserving order
        seen = set()
        unique_cols = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                unique_cols.append(c)

        player = {}
        for c in unique_cols:
            if c in row.index:
                player[c] = _safe_val(row[c])
            else:
                player[c] = None
        players.append(player)

    print(f"Loaded {len(players)} player profiles")
    return players


def load_comps() -> dict[str, list[dict]]:
    """Load comp data from both sim score files, keyed by 'name|year'."""
    comps = {}

    for csv_name in ["sim_scores_2026.csv", "sim_scores_backtest_v5.csv"]:
        path = OUTPUT_DIR / csv_name
        if not path.exists():
            print(f"  Warning: {csv_name} not found, skipping")
            continue
        df = pd.read_csv(path)
        print(f"  Loaded {len(df)} rows from {csv_name}")

        for _, row in df.iterrows():
            key = f"{row['prospect']}|{int(row['prospect_year'])}"
            comp = {
                "rank": int(row["comp_rank"]),
                "name": row["comp_player"],
                "college": _safe_val(row.get("comp_college", "")),
                "pick": _safe_val(row.get("comp_pick", None)),
                "year": int(row["comp_year"]),
                "sim": _safe_val(row["similarity_score"]),
                "cov": _safe_val(row["comparison_coverage"]),
                "driver": _safe_val(row.get("dominant_category", "")),
                "cat_breakdown": _safe_val(row.get("category_breakdown", "")),
                "hit24": _safe_val(row.get("hit24", None)),
                "hit12": _safe_val(row.get("hit12", None)),
                "career_ppg": _safe_val(row.get("career_ppg", None)),
                "first_3yr_ppg": _safe_val(row.get("first_3yr_ppg", None)),
                "best_ppg": _safe_val(row.get("nfl_best_season_ppg", None)),
            }
            if key not in comps:
                comps[key] = []
            comps[key].append(comp)

    print(f"Loaded comps for {len(comps)} players")
    return comps


def build_html(players_json: str, comps_json: str, player_count: int) -> str:
    """Build the complete HTML string."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SLAP V5 — SimScores Explorer</title>
<style>
/* ================================================================
   CSS — SimScores Explorer
   ================================================================ */
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

:root {{
  --bg: #f1f5f9;
  --card: #ffffff;
  --border: #e2e8f0;
  --text: #1e293b;
  --text-muted: #64748b;
  --wr: #3b82f6;
  --rb: #10b981;
  --te: #f59e0b;
  --hit-yes: #059669;
  --hit-no: #dc2626;
  --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-lg: 0 4px 12px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06);
  --radius: 8px;
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
               Ubuntu, Cantarell, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}}

/* ---- Header ---- */
.header {{
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
  color: white;
  padding: 2rem 1rem 1.5rem;
  text-align: center;
}}
.header h1 {{
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}}
.header .subtitle {{
  color: #94a3b8;
  font-size: 0.9rem;
  margin-top: 0.25rem;
}}

/* ---- Search ---- */
.search-wrap {{
  max-width: 560px;
  margin: 1.25rem auto 0;
  position: relative;
}}
.search-wrap input {{
  width: 100%;
  padding: 0.75rem 1rem 0.75rem 2.75rem;
  font-size: 1rem;
  border: 2px solid transparent;
  border-radius: var(--radius);
  outline: none;
  background: rgba(255,255,255,0.12);
  color: white;
  transition: background 0.2s, border-color 0.2s;
}}
.search-wrap input::placeholder {{ color: #94a3b8; }}
.search-wrap input:focus {{
  background: rgba(255,255,255,0.18);
  border-color: rgba(255,255,255,0.3);
}}
.search-icon {{
  position: absolute;
  left: 0.9rem;
  top: 50%;
  transform: translateY(-50%);
  color: #94a3b8;
  pointer-events: none;
  font-size: 1.1rem;
}}
.suggestions {{
  position: absolute;
  top: calc(100% + 4px);
  left: 0; right: 0;
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow-lg);
  z-index: 100;
  max-height: 360px;
  overflow-y: auto;
  display: none;
}}
.suggestions.open {{ display: block; }}
.sug-item {{
  padding: 0.6rem 1rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--text);
  border-bottom: 1px solid var(--border);
  font-size: 0.9rem;
}}
.sug-item:last-child {{ border-bottom: none; }}
.sug-item:hover, .sug-item.active {{
  background: #f1f5f9;
}}
.sug-item .sug-meta {{
  color: var(--text-muted);
  font-size: 0.8rem;
  margin-left: auto;
  white-space: nowrap;
}}

/* ---- Position badges ---- */
.pos-badge {{
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 700;
  color: white;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}}
.pos-badge.WR {{ background: var(--wr); }}
.pos-badge.RB {{ background: var(--rb); }}
.pos-badge.TE {{ background: var(--te); }}

.dataset-badge {{
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.dataset-badge.backtest {{ background: #e2e8f0; color: #475569; }}
.dataset-badge.prospect {{ background: #dbeafe; color: #1d4ed8; }}

/* ---- Filters ---- */
.filters {{
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 1rem;
}}
.filter-btn {{
  padding: 0.35rem 1rem;
  border-radius: 20px;
  border: 1.5px solid rgba(255,255,255,0.25);
  background: transparent;
  color: #cbd5e1;
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
}}
.filter-btn:hover {{ border-color: rgba(255,255,255,0.5); color: white; }}
.filter-btn.active {{
  background: rgba(255,255,255,0.15);
  border-color: rgba(255,255,255,0.5);
  color: white;
}}

/* ---- Main ---- */
.main {{
  max-width: 1200px;
  margin: 0 auto;
  padding: 1.5rem 1rem;
}}

/* ---- Player card ---- */
.player-card {{
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.5rem;
  margin-bottom: 1.25rem;
  display: none;
}}
.player-card.visible {{ display: block; }}

.pc-header {{
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}}
.pc-name {{
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}}
.pc-meta {{
  color: var(--text-muted);
  font-size: 0.9rem;
}}
.pc-meta span {{ margin-right: 1rem; }}

/* Score boxes row */
.score-row {{
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-bottom: 1.25rem;
}}
.score-box {{
  flex: 1;
  min-width: 100px;
  max-width: 180px;
  background: var(--bg);
  border-radius: var(--radius);
  padding: 0.75rem;
  text-align: center;
}}
.score-box .score-label {{
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 0.25rem;
}}
.score-box .score-val {{
  font-size: 1.6rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}}

/* NFL outcomes row */
.nfl-row {{
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 1.25rem;
  align-items: center;
}}
.nfl-badge {{
  padding: 0.3rem 0.75rem;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 700;
}}
.nfl-badge.hit {{ background: #d1fae5; color: var(--hit-yes); }}
.nfl-badge.miss {{ background: #fee2e2; color: var(--hit-no); }}
.nfl-badge.na {{ background: #f1f5f9; color: var(--text-muted); }}
.nfl-stat {{
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-left: 0.25rem;
}}
.nfl-stat b {{ color: var(--text); font-weight: 600; }}

/* Stats grid */
.stats-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 0.5rem;
}}
.stat-item {{
  display: flex;
  justify-content: space-between;
  padding: 0.35rem 0.5rem;
  background: var(--bg);
  border-radius: 4px;
  font-size: 0.82rem;
}}
.stat-item .stat-label {{ color: var(--text-muted); }}
.stat-item .stat-val {{ font-weight: 600; }}

/* ---- Comp table ---- */
.comp-section {{
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.25rem;
  display: none;
}}
.comp-section.visible {{ display: block; }}
.comp-title {{
  font-size: 1.1rem;
  font-weight: 700;
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}}
.comp-summary {{
  font-size: 0.82rem;
  color: var(--text-muted);
  font-weight: 400;
}}

.comp-table-wrap {{
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}}
table.comp-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
  white-space: nowrap;
}}
table.comp-table th {{
  background: #f8fafc;
  padding: 0.5rem 0.75rem;
  text-align: left;
  font-weight: 700;
  font-size: 0.75rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-bottom: 2px solid var(--border);
  cursor: pointer;
  user-select: none;
  position: relative;
}}
table.comp-table th:hover {{ color: var(--text); }}
table.comp-table th .sort-arrow {{
  margin-left: 3px;
  font-size: 0.65rem;
  opacity: 0.4;
}}
table.comp-table th.sorted .sort-arrow {{ opacity: 1; }}

table.comp-table td {{
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid var(--border);
}}
table.comp-table tr:hover td {{
  background: #f8fafc;
}}
table.comp-table .comp-name {{
  font-weight: 600;
}}
.sim-pill {{
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-weight: 700;
  font-size: 0.8rem;
  color: white;
}}
.hit-dot {{
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 4px;
}}
.hit-dot.yes {{ background: var(--hit-yes); }}
.hit-dot.no {{ background: var(--hit-no); }}
.hit-dot.na {{ background: #cbd5e1; }}

/* ---- Welcome ---- */
.welcome {{
  text-align: center;
  padding: 4rem 1rem;
  color: var(--text-muted);
}}
.welcome h2 {{
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.5rem;
}}
.welcome p {{
  max-width: 480px;
  margin: 0 auto;
  font-size: 0.95rem;
}}

/* ---- Category breakdown tooltip ---- */
.cat-breakdown {{
  position: relative;
  cursor: help;
  text-decoration: underline dotted;
  text-underline-offset: 2px;
}}
.cat-breakdown:hover .cat-tip {{
  display: block;
}}
.cat-tip {{
  display: none;
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  background: #1e293b;
  color: white;
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  font-size: 0.75rem;
  white-space: nowrap;
  z-index: 50;
  box-shadow: var(--shadow-lg);
}}
.cat-tip::after {{
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 5px solid transparent;
  border-top-color: #1e293b;
}}

/* ---- Responsive ---- */
@media (max-width: 640px) {{
  .header h1 {{ font-size: 1.3rem; }}
  .pc-name {{ font-size: 1.2rem; }}
  .score-box {{ min-width: 80px; }}
  .score-box .score-val {{ font-size: 1.3rem; }}
  .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
  table.comp-table {{ font-size: 0.78rem; }}
  table.comp-table th, table.comp-table td {{ padding: 0.4rem 0.5rem; }}
}}
</style>
</head>
<body>

<!-- ============================================================ -->
<!-- HEADER + SEARCH                                              -->
<!-- ============================================================ -->
<div class="header">
  <h1>SLAP V5 &mdash; SimScores Explorer</h1>
  <div class="subtitle">{player_count} Players &bull; 2015&ndash;2026 &bull; WR / RB / TE</div>

  <div class="search-wrap">
    <span class="search-icon">&#128269;</span>
    <input type="text" id="search" placeholder="Search for a player..." autocomplete="off">
    <div class="suggestions" id="suggestions"></div>
  </div>

  <div class="filters" id="filters">
    <button class="filter-btn active" data-filter="ALL">All</button>
    <button class="filter-btn" data-filter="WR">WR</button>
    <button class="filter-btn" data-filter="RB">RB</button>
    <button class="filter-btn" data-filter="TE">TE</button>
  </div>
</div>

<!-- ============================================================ -->
<!-- MAIN CONTENT                                                 -->
<!-- ============================================================ -->
<div class="main">
  <div class="welcome" id="welcome">
    <h2>Search for any player to view their SimScores</h2>
    <p>Type a name above to see prospect profiles and historical comparisons for all 944 drafted players (2015&ndash;2026).</p>
  </div>

  <div class="player-card" id="player-card"></div>
  <div class="comp-section" id="comp-section"></div>
</div>

<!-- ============================================================ -->
<!-- EMBEDDED DATA                                                -->
<!-- ============================================================ -->
<script>
const PLAYERS = {players_json};
const COMPS = {comps_json};
</script>

<!-- ============================================================ -->
<!-- APPLICATION LOGIC                                            -->
<!-- ============================================================ -->
<script>
(function() {{
  'use strict';

  // ---- State ----
  let posFilter = 'ALL';
  let activeIndex = -1;
  let currentPlayer = null;
  let sortCol = null;
  let sortAsc = true;

  // ---- DOM refs ----
  const searchInput = document.getElementById('search');
  const sugBox = document.getElementById('suggestions');
  const cardEl = document.getElementById('player-card');
  const compEl = document.getElementById('comp-section');
  const welcomeEl = document.getElementById('welcome');
  const filterBtns = document.querySelectorAll('.filter-btn');

  // ---- Index for fast search ----
  const playerIndex = PLAYERS.map((p, i) => ({{
    idx: i,
    lower: p.player_name.toLowerCase(),
    name: p.player_name,
    pos: p.position,
    college: p.college || '',
    year: p.draft_year,
    pick: p.draft_pick,
    dataset: p.dataset,
  }}));

  // ---- Player lookup for comp stats ----
  const playerLookup = {{}};
  PLAYERS.forEach(p => {{
    playerLookup[p.player_name + '|' + p.draft_year] = p;
  }});

  // ---- Helpers ----
  function fmt(v, decimals) {{
    if (v === null || v === undefined) return '—';
    if (typeof decimals === 'number') return Number(v).toFixed(decimals);
    return String(v);
  }}

  function scoreColor(s) {{
    if (s === null || s === undefined) return '#94a3b8';
    if (s >= 80) return '#059669';
    if (s >= 60) return '#2563eb';
    if (s >= 40) return '#d97706';
    if (s >= 20) return '#ea580c';
    return '#dc2626';
  }}

  function simColor(s) {{
    if (s >= 90) return '#059669';
    if (s >= 80) return '#0d9488';
    if (s >= 70) return '#2563eb';
    if (s >= 60) return '#7c3aed';
    if (s >= 50) return '#9333ea';
    return '#6b7280';
  }}

  function posClass(pos) {{ return 'pos-badge ' + pos; }}

  // ---- Search ----
  function search(query) {{
    if (!query || query.length < 1) return [];
    const q = query.toLowerCase();
    let results = playerIndex.filter(p => {{
      if (posFilter !== 'ALL' && p.pos !== posFilter) return false;
      return p.lower.includes(q);
    }});
    // Sort: starts-with first, then alphabetical
    results.sort((a, b) => {{
      const aStarts = a.lower.startsWith(q) ? 0 : 1;
      const bStarts = b.lower.startsWith(q) ? 0 : 1;
      if (aStarts !== bStarts) return aStarts - bStarts;
      return a.lower.localeCompare(b.lower);
    }});
    return results.slice(0, 12);
  }}

  function renderSuggestions(results) {{
    if (!results.length) {{
      sugBox.innerHTML = '';
      sugBox.classList.remove('open');
      return;
    }}
    activeIndex = -1;
    sugBox.innerHTML = results.map((r, i) => `
      <div class="sug-item" data-idx="${{r.idx}}">
        <span class="${{posClass(r.pos)}}">${{r.pos}}</span>
        <span>${{r.name}}</span>
        <span class="sug-meta">${{r.college}}, ${{r.dataset === 'backtest' ? '#' + (r.pick || '?') + ' ' + r.year : '2026 prospect'}}</span>
      </div>
    `).join('');
    sugBox.classList.add('open');

    sugBox.querySelectorAll('.sug-item').forEach(el => {{
      el.addEventListener('click', () => {{
        selectPlayer(PLAYERS[parseInt(el.dataset.idx)]);
      }});
    }});
  }}

  function closeSuggestions() {{
    sugBox.classList.remove('open');
    activeIndex = -1;
  }}

  searchInput.addEventListener('input', () => {{
    const results = search(searchInput.value.trim());
    renderSuggestions(results);
  }});

  searchInput.addEventListener('keydown', (e) => {{
    const items = sugBox.querySelectorAll('.sug-item');
    if (!items.length) return;

    if (e.key === 'ArrowDown') {{
      e.preventDefault();
      activeIndex = Math.min(activeIndex + 1, items.length - 1);
      items.forEach((el, i) => el.classList.toggle('active', i === activeIndex));
    }} else if (e.key === 'ArrowUp') {{
      e.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      items.forEach((el, i) => el.classList.toggle('active', i === activeIndex));
    }} else if (e.key === 'Enter') {{
      e.preventDefault();
      if (activeIndex >= 0 && activeIndex < items.length) {{
        selectPlayer(PLAYERS[parseInt(items[activeIndex].dataset.idx)]);
      }} else if (items.length === 1) {{
        selectPlayer(PLAYERS[parseInt(items[0].dataset.idx)]);
      }}
    }} else if (e.key === 'Escape') {{
      closeSuggestions();
    }}
  }});

  document.addEventListener('click', (e) => {{
    if (!sugBox.contains(e.target) && e.target !== searchInput) closeSuggestions();
  }});

  // ---- Filters ----
  filterBtns.forEach(btn => {{
    btn.addEventListener('click', () => {{
      posFilter = btn.dataset.filter;
      filterBtns.forEach(b => b.classList.toggle('active', b === btn));
      // Re-run search
      const results = search(searchInput.value.trim());
      renderSuggestions(results);
    }});
  }});

  // ---- Select player ----
  function selectPlayer(player) {{
    currentPlayer = player;
    searchInput.value = player.player_name;
    closeSuggestions();
    welcomeEl.style.display = 'none';
    renderCard(player);
    renderComps(player);
  }}

  // ---- Render player card ----
  function renderCard(p) {{
    const isBacktest = p.dataset === 'backtest';
    const pickStr = p.draft_pick ? '#' + p.draft_pick : 'Projected';
    const rdStr = p.draft_round ? 'Rd ' + p.draft_round : '';

    // Score boxes
    const scores = [
      {{ label: 'SLAP Display', val: p.slap_display_score }},
      {{ label: 'Model Score', val: p.slap_model_score }},
      {{ label: 'DC Score', val: p.dc_score }},
      {{ label: 'Prospect Profile', val: p.prospect_profile }},
    ];

    // Position-specific stats
    const stats = [];
    // Physical
    if (p.height_in != null) {{
      const ft = Math.floor(p.height_in / 12);
      const inches = Math.round(p.height_in % 12);
      stats.push({{ label: 'Height', val: ft + "'" + inches + '"' }});
    }}
    if (p.weight != null) stats.push({{ label: 'Weight', val: p.weight + ' lbs' }});
    if (p.forty != null) stats.push({{ label: '40 Time', val: fmt(p.forty, 2) + 's' }});
    if (p.draft_age != null) stats.push({{ label: 'Draft Age', val: fmt(p.draft_age, 1) }});
    if (p.ras_score != null) stats.push({{ label: 'RAS', val: fmt(p.ras_score, 1) }});

    // Production
    if (p.rec_yards != null) stats.push({{ label: 'Rec Yards', val: fmt(p.rec_yards, 0) }});
    if (p.receptions != null) stats.push({{ label: 'Receptions', val: fmt(p.receptions, 0) }});
    if (p.rec_tds != null) stats.push({{ label: 'Rec TDs', val: fmt(p.rec_tds, 0) }});
    if (p.yards_per_reception != null) stats.push({{ label: 'Yards/Rec', val: fmt(p.yards_per_reception, 1) }});
    if (p.rush_yards != null) stats.push({{ label: 'Rush Yards', val: fmt(p.rush_yards, 0) }});
    if (p.rush_attempts != null) stats.push({{ label: 'Rush Att', val: fmt(p.rush_attempts, 0) }});
    if (p.team_pass_att != null) stats.push({{ label: 'Team Pass Att', val: fmt(p.team_pass_att, 0) }});
    if (p.games_played != null) stats.push({{ label: 'Games Played', val: fmt(p.games_played, 0) }});

    // Position-specific
    if (p.position === 'WR') {{
      if (p.breakout_age != null) stats.push({{ label: 'Breakout Age', val: fmt(p.breakout_age, 0) }});
      if (p.peak_dominator != null) stats.push({{ label: 'Peak Dom%', val: fmt(p.peak_dominator, 1) + '%' }});
      if (p.breakout_score != null) stats.push({{ label: 'Breakout Score', val: fmt(p.breakout_score, 1) }});
      if (p.early_declare_score != null) stats.push({{ label: 'Early Declare', val: p.early_declare_score >= 50 ? 'Yes (100)' : 'No (0)' }});
      if (p.teammate_score != null) stats.push({{ label: 'Teammate Score', val: fmt(p.teammate_score, 0) }});
    }}
    if (p.position === 'RB') {{
      if (p.speed_score != null) stats.push({{ label: 'Speed Score', val: fmt(p.speed_score, 1) }});
      if (p.bdr != null) stats.push({{ label: 'BDR', val: fmt(p.bdr, 1) }});
      if (p.rec_yards_per_team_pass_att != null) stats.push({{ label: 'RYPTPA', val: fmt(p.rec_yards_per_team_pass_att, 3) }});
      if (p.yards_per_carry != null) stats.push({{ label: 'Yards/Carry', val: fmt(p.yards_per_carry, 1) }});
    }}
    if (p.position === 'TE') {{
      if (p.te_breakout_score != null) stats.push({{ label: 'TE Breakout', val: fmt(p.te_breakout_score, 1) }});
      if (p.te_production_score != null) stats.push({{ label: 'TE Production', val: fmt(p.te_production_score, 1) }});
      if (p.breakout_age != null) stats.push({{ label: 'Breakout Age', val: fmt(p.breakout_age, 0) }});
      if (p.peak_dominator != null) stats.push({{ label: 'Peak Dom%', val: fmt(p.peak_dominator, 1) + '%' }});
    }}

    // NFL outcomes
    let nflHtml = '';
    if (isBacktest) {{
      const h24 = p.nfl_hit24;
      const h12 = p.nfl_hit12;
      const h24Class = h24 === 1 ? 'hit' : h24 === 0 ? 'miss' : 'na';
      const h12Class = h12 === 1 ? 'hit' : h12 === 0 ? 'miss' : 'na';
      const h24Text = h24 === 1 ? 'Hit (Top 24)' : h24 === 0 ? 'Miss (Top 24)' : 'N/A';
      const h12Text = h12 === 1 ? 'Hit (Top 12)' : h12 === 0 ? 'Miss (Top 12)' : 'N/A';

      nflHtml = `
        <div class="nfl-row">
          <span class="nfl-badge ${{h24Class}}">${{h24Text}}</span>
          <span class="nfl-badge ${{h12Class}}">${{h12Text}}</span>
          ${{p.nfl_career_ppg != null ? `<span class="nfl-stat">Career PPG: <b>${{fmt(p.nfl_career_ppg, 1)}}</b></span>` : ''}}
          ${{p.nfl_first_3yr_ppg != null ? `<span class="nfl-stat">First 3yr PPG: <b>${{fmt(p.nfl_first_3yr_ppg, 1)}}</b></span>` : ''}}
          ${{p.nfl_best_ppg != null ? `<span class="nfl-stat">Best Season: <b>${{fmt(p.nfl_best_ppg, 1)}}</b></span>` : ''}}
          ${{p.nfl_games != null ? `<span class="nfl-stat">NFL Games: <b>${{fmt(p.nfl_games, 0)}}</b></span>` : ''}}
        </div>
      `;
    }}

    cardEl.innerHTML = `
      <div class="pc-header">
        <span class="pc-name">${{p.player_name}}</span>
        <span class="${{posClass(p.position)}}">${{p.position}}</span>
        <span class="dataset-badge ${{isBacktest ? 'backtest' : 'prospect'}}">${{isBacktest ? 'Backtest' : '2026 Prospect'}}</span>
      </div>
      <div class="pc-meta">
        <span>${{p.college || '—'}}</span>
        <span>${{pickStr}}${{rdStr ? ' (' + rdStr + ')' : ''}} &bull; ${{p.draft_year}}</span>
      </div>

      <div class="score-row">
        ${{scores.map(s => `
          <div class="score-box">
            <div class="score-label">${{s.label}}</div>
            <div class="score-val" style="color: ${{scoreColor(s.val)}}">${{s.val != null ? fmt(s.val, s.label === 'Model Score' ? 1 : 0) : '—'}}</div>
          </div>
        `).join('')}}
      </div>

      ${{nflHtml}}

      <div class="stats-grid">
        ${{stats.map(s => `
          <div class="stat-item">
            <span class="stat-label">${{s.label}}</span>
            <span class="stat-val">${{s.val}}</span>
          </div>
        `).join('')}}
      </div>
    `;
    cardEl.classList.add('visible');
  }}

  // ---- Comp stat helpers ----
  function getCompPlayer(comp) {{
    return playerLookup[comp.name + '|' + comp.year] || null;
  }}

  function calcYPG(cp, stat) {{
    if (!cp || cp[stat] == null || cp.games_played == null || cp.games_played === 0) return null;
    return cp[stat] / cp.games_played;
  }}

  function calcRYPTPA(cp) {{
    if (!cp) return null;
    if (cp.rec_yards_per_team_pass_att != null) return cp.rec_yards_per_team_pass_att;
    if (cp.rec_yards != null && cp.team_pass_att != null && cp.team_pass_att > 0)
      return cp.rec_yards / cp.team_pass_att;
    return null;
  }}

  function prodColLabel(pos) {{
    return pos === 'RB' ? 'Rush YPG' : 'Rec YPG';
  }}

  function prodColVal(cp, pos) {{
    return pos === 'RB' ? calcYPG(cp, 'rush_yards') : calcYPG(cp, 'rec_yards');
  }}

  function enrichComp(c, pos) {{
    const cp = getCompPlayer(c);
    return {{
      ...c,
      _weight: cp ? cp.weight : null,
      _peak_dom: cp ? cp.peak_dominator : null,
      _prod: prodColVal(cp, pos),
      _ryptpa: calcRYPTPA(cp),
    }};
  }}

  // ---- Render comps table ----
  function renderComps(p) {{
    const key = p.player_name + '|' + p.draft_year;
    const comps = COMPS[key];
    const pos = p.position;

    if (!comps || !comps.length) {{
      compEl.innerHTML = '<div style="padding:1rem;color:var(--text-muted);">No SimScore comps available for this player.</div>';
      compEl.classList.add('visible');
      return;
    }}

    // Enrich comps with player lookup data
    let enriched = comps.map(c => enrichComp(c, pos));

    // Sort
    if (sortCol) {{
      enriched.sort((a, b) => {{
        let va = a[sortCol], vb = b[sortCol];
        if (va === null || va === undefined) va = -Infinity;
        if (vb === null || vb === undefined) vb = -Infinity;
        if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        return sortAsc ? va - vb : vb - va;
      }});
    }}

    // Summary stats
    const avgSim = (comps.reduce((s, c) => s + c.sim, 0) / comps.length).toFixed(1);
    const avgCov = (comps.reduce((s, c) => s + (c.cov || 0), 0) / comps.length * 100).toFixed(0);

    function thArrow(col) {{
      if (sortCol !== col) return '<span class="sort-arrow">&#9650;</span>';
      return `<span class="sort-arrow">${{sortAsc ? '&#9650;' : '&#9660;'}}</span>`;
    }}

    const prodLabel = prodColLabel(pos);

    compEl.innerHTML = `
      <div class="comp-title">
        Top 10 Comps
        <span class="comp-summary">
          Avg Similarity: ${{avgSim}} &bull; Avg Coverage: ${{avgCov}}%
        </span>
      </div>
      <div class="comp-table-wrap">
        <table class="comp-table">
          <thead>
            <tr>
              <th data-col="rank"># ${{thArrow('rank')}}</th>
              <th data-col="sim">Sim ${{thArrow('sim')}}</th>
              <th data-col="cov">Cov ${{thArrow('cov')}}</th>
              <th data-col="name">Player ${{thArrow('name')}}</th>
              <th data-col="college">College ${{thArrow('college')}}</th>
              <th data-col="_weight">Weight ${{thArrow('_weight')}}</th>
              <th data-col="_peak_dom">Peak Dom% ${{thArrow('_peak_dom')}}</th>
              <th data-col="_prod">${{prodLabel}} ${{thArrow('_prod')}}</th>
              <th data-col="_ryptpa">RYPTPA ${{thArrow('_ryptpa')}}</th>
              <th data-col="driver">Driver ${{thArrow('driver')}}</th>
            </tr>
          </thead>
          <tbody>
            ${{enriched.map(c => {{
              const driverLabel = c.driver ? c.driver.replace('_', ' ') : '—';
              return `
              <tr>
                <td>${{c.rank}}</td>
                <td><span class="sim-pill" style="background:${{simColor(c.sim)}}">${{fmt(c.sim, 1)}}</span></td>
                <td>${{c.cov != null ? (c.cov * 100).toFixed(0) + '%' : '—'}}</td>
                <td class="comp-name">${{c.name}}</td>
                <td>${{c.college || '—'}}</td>
                <td>${{c._weight != null ? c._weight : '—'}}</td>
                <td>${{c._peak_dom != null ? fmt(c._peak_dom, 1) + '%' : '—'}}</td>
                <td>${{c._prod != null ? fmt(c._prod, 1) : '—'}}</td>
                <td>${{c._ryptpa != null ? fmt(c._ryptpa, 3) : '—'}}</td>
                <td class="cat-breakdown">
                  ${{driverLabel}}
                  ${{c.cat_breakdown ? '<span class="cat-tip">' + c.cat_breakdown + '</span>' : ''}}
                </td>
              </tr>`;
            }}).join('')}}
          </tbody>
        </table>
      </div>
    `;
    compEl.classList.add('visible');

    // Attach sort handlers
    compEl.querySelectorAll('th[data-col]').forEach(th => {{
      th.addEventListener('click', () => {{
        const col = th.dataset.col;
        if (sortCol === col) {{
          sortAsc = !sortAsc;
        }} else {{
          sortCol = col;
          sortAsc = col === 'rank' || col === 'name' || col === 'college' || col === 'driver';
        }}
        renderComps(currentPlayer);
      }});
    }});
  }}

}})();
</script>

</body>
</html>"""


def main():
    print("Building SimScores Explorer HTML...")

    players = load_players()
    comps = load_comps()

    # Convert to JSON strings
    players_json = json.dumps(players, ensure_ascii=False)
    comps_json = json.dumps(comps, ensure_ascii=False)

    html = build_html(players_json, comps_json, len(players))

    out_path = OUTPUT_DIR / "sim_scores_explorer.html"
    out_path.write_text(html, encoding="utf-8")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved to {out_path} ({size_mb:.1f} MB)")
    print(f"  {len(players)} players, {sum(len(v) for v in comps.values())} comp rows embedded")


if __name__ == "__main__":
    main()
