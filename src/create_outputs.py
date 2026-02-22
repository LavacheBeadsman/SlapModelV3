"""
SLAP Score V3 - Create Final Outputs for 2026 Content

This script generates:
1. Complete RB rankings CSV
2. Complete WR rankings CSV
3. SLAP vs Draft Position scatter plot
4. Backtest summary stats image
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Create output directory
os.makedirs('output', exist_ok=True)

# =============================================================================
# 1 & 2: EXPORT CSV RANKINGS
# =============================================================================

def export_rankings():
    """Export complete RB and WR rankings with all columns."""

    # Read the scored data
    rb_data = []
    wr_data = []

    with open('output/slap_scores_rb.csv', 'r') as f:
        reader = csv.DictReader(f)
        rb_data = list(reader)

    with open('output/slap_scores_wr.csv', 'r') as f:
        reader = csv.DictReader(f)
        wr_data = list(reader)

    # Define columns for export
    columns = [
        'rank', 'player_name', 'school', 'projected_pick',
        'slap_score', 'delta', 'draft_capital_score', 'breakout_score',
        'rec_yards', 'team_pass_attempts', 'age', 'age_estimated', 'weight'
    ]

    # Export RB rankings
    print("Exporting RB rankings...")
    with open('output/2026_rb_rankings_final.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for i, row in enumerate(rb_data, 1):
            writer.writerow([
                i,
                row.get('player_name', ''),
                row.get('school', ''),
                row.get('projected_pick', ''),
                row.get('slap_score', ''),
                row.get('delta', ''),
                row.get('draft_capital_score', ''),
                row.get('breakout_score', ''),
                row.get('rec_yards', ''),
                row.get('team_pass_attempts', ''),
                row.get('age', ''),
                row.get('age_estimated', ''),
                row.get('weight', '')
            ])
    print(f"  Saved: output/2026_rb_rankings_final.csv ({len(rb_data)} players)")

    # Export WR rankings
    print("Exporting WR rankings...")
    with open('output/2026_wr_rankings_final.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for i, row in enumerate(wr_data, 1):
            writer.writerow([
                i,
                row.get('player_name', ''),
                row.get('school', ''),
                row.get('projected_pick', ''),
                row.get('slap_score', ''),
                row.get('delta', ''),
                row.get('draft_capital_score', ''),
                row.get('breakout_score', ''),
                row.get('rec_yards', ''),
                row.get('team_pass_attempts', ''),
                row.get('age', ''),
                row.get('age_estimated', ''),
                row.get('weight', '')
            ])
    print(f"  Saved: output/2026_wr_rankings_final.csv ({len(wr_data)} players)")

    return rb_data, wr_data


# =============================================================================
# 3: SCATTER PLOT - SLAP vs Draft Position
# =============================================================================

def create_scatter_plot(rb_data, wr_data):
    """Create scatter plot of SLAP Score vs Draft Position."""

    print("Creating SLAP vs Draft Position scatter plot...")

    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Prepare data
    rb_picks = []
    rb_slaps = []
    rb_names = []
    rb_deltas = []

    for row in rb_data:
        try:
            pick = int(row.get('projected_pick', 0))
            slap = float(row.get('slap_score', 0))
            delta = float(row.get('delta', 0)) if row.get('delta') else 0
            if pick > 0 and slap > 0:
                rb_picks.append(pick)
                rb_slaps.append(slap)
                rb_names.append(row.get('player_name', ''))
                rb_deltas.append(delta)
        except (ValueError, TypeError):
            pass

    wr_picks = []
    wr_slaps = []
    wr_names = []
    wr_deltas = []

    for row in wr_data:
        try:
            pick = int(row.get('projected_pick', 0))
            slap = float(row.get('slap_score', 0))
            delta = float(row.get('delta', 0)) if row.get('delta') else 0
            if pick > 0 and slap > 0:
                wr_picks.append(pick)
                wr_slaps.append(slap)
                wr_names.append(row.get('player_name', ''))
                wr_deltas.append(delta)
        except (ValueError, TypeError):
            pass

    # Plot RBs and WRs
    ax.scatter(rb_picks, rb_slaps, c='#E63946', s=100, alpha=0.7, label='RB', edgecolors='white', linewidth=1)
    ax.scatter(wr_picks, wr_slaps, c='#457B9D', s=100, alpha=0.7, label='WR', edgecolors='white', linewidth=1)

    # Add trend line (draft capital baseline)
    x_line = np.linspace(1, 260, 100)
    # Approximate the draft capital score curve
    y_line = 50 + 15 * (1/np.sqrt(x_line) - np.mean(1/np.sqrt(np.arange(1, 261)))) / np.std(1/np.sqrt(np.arange(1, 261)))
    ax.plot(x_line, y_line, 'k--', alpha=0.3, linewidth=2, label='Draft Capital Baseline')

    # Label biggest positive deltas (RBs)
    rb_sorted = sorted(zip(rb_deltas, rb_names, rb_picks, rb_slaps), reverse=True)
    for delta, name, pick, slap in rb_sorted[:3]:
        if delta > 5:
            ax.annotate(f'{name}\n(+{delta:.0f})', xy=(pick, slap),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, color='#E63946', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Label biggest negative deltas (RBs)
    rb_sorted_neg = sorted(zip(rb_deltas, rb_names, rb_picks, rb_slaps))
    for delta, name, pick, slap in rb_sorted_neg[:2]:
        if delta < -5:
            ax.annotate(f'{name}\n({delta:.0f})', xy=(pick, slap),
                       xytext=(10, -15), textcoords='offset points',
                       fontsize=9, color='#E63946', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Label biggest positive deltas (WRs)
    wr_sorted = sorted(zip(wr_deltas, wr_names, wr_picks, wr_slaps), reverse=True)
    for delta, name, pick, slap in wr_sorted[:3]:
        if delta > 5:
            ax.annotate(f'{name}\n(+{delta:.0f})', xy=(pick, slap),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, color='#457B9D', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Label biggest negative deltas (WRs)
    wr_sorted_neg = sorted(zip(wr_deltas, wr_names, wr_picks, wr_slaps))
    for delta, name, pick, slap in wr_sorted_neg[:2]:
        if delta < -5:
            ax.annotate(f'{name}\n({delta:.0f})', xy=(pick, slap),
                       xytext=(10, -15), textcoords='offset points',
                       fontsize=9, color='#457B9D', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Formatting
    ax.set_xlabel('Projected Draft Pick', fontsize=14, fontweight='bold')
    ax.set_ylabel('SLAP Score', fontsize=14, fontweight='bold')
    ax.set_title('2026 SLAP Score vs Draft Position\nPlayers Above Line = Model Likes More Than Draft Slot',
                fontsize=16, fontweight='bold', pad=20)

    ax.set_xlim(0, 270)
    ax.set_ylim(30, 100)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add explanation text
    ax.text(0.02, 0.02,
            'Positive Delta = SLAP likes more than draft position\n'
            'Negative Delta = SLAP likes less than draft position',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('output/slap_vs_draft_position.png', dpi=150, bbox_inches='tight')
    print("  Saved: output/slap_vs_draft_position.png")
    plt.close()


# =============================================================================
# 4: BACKTEST SUMMARY STATS IMAGE
# =============================================================================

def create_backtest_summary():
    """Create a summary image showing backtest correlation results."""

    print("Creating backtest summary image...")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'SLAP Score Model Validation',
            transform=ax.transAxes, fontsize=28, fontweight='bold',
            ha='center', va='top')

    ax.text(0.5, 0.88, '3-Year Backtest: 2022-2024 Draft Classes',
            transform=ax.transAxes, fontsize=16, ha='center', va='top', color='gray')

    # Main correlation box
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#2E86AB', alpha=0.9)
    ax.text(0.5, 0.72, '0.566', transform=ax.transAxes, fontsize=72,
            fontweight='bold', ha='center', va='center', color='white',
            bbox=box_props)

    ax.text(0.5, 0.52, 'Combined Correlation\n(SLAP Score vs NFL Rookie PPR)',
            transform=ax.transAxes, fontsize=14, ha='center', va='top', color='#333')

    # Year-by-year results
    years_data = [
        ('2022', '0.648', '47 players'),
        ('2023', '0.631', '46 players'),
        ('2024', '0.449', '47 players'),
    ]

    y_start = 0.38
    for i, (year, corr, players) in enumerate(years_data):
        x = 0.2 + i * 0.3

        # Year box
        ax.text(x, y_start, year, transform=ax.transAxes, fontsize=18,
                fontweight='bold', ha='center', va='center')
        ax.text(x, y_start - 0.06, corr, transform=ax.transAxes, fontsize=24,
                fontweight='bold', ha='center', va='center', color='#E63946')
        ax.text(x, y_start - 0.11, players, transform=ax.transAxes, fontsize=11,
                ha='center', va='center', color='gray')

    # Interpretation
    ax.text(0.5, 0.15, 'STRONG Positive Correlation',
            transform=ax.transAxes, fontsize=20, fontweight='bold',
            ha='center', va='center', color='#28A745')

    ax.text(0.5, 0.08, 'SLAP Score explains ~32% of variance in rookie fantasy production',
            transform=ax.transAxes, fontsize=12, ha='center', va='center', color='#666')

    ax.text(0.5, 0.03, '140 total players analyzed across RB and WR positions',
            transform=ax.transAxes, fontsize=11, ha='center', va='center', color='#999')

    plt.tight_layout()
    plt.savefig('output/backtest_summary.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("  Saved: output/backtest_summary.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CREATING 2026 SLAP SCORE OUTPUTS")
    print("=" * 60)
    print()

    # Export CSVs
    rb_data, wr_data = export_rankings()
    print()

    # Create scatter plot
    create_scatter_plot(rb_data, wr_data)
    print()

    # Create backtest summary
    create_backtest_summary()
    print()

    print("=" * 60)
    print("ALL OUTPUTS CREATED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Files saved to output/ folder:")
    print("  - 2026_rb_rankings_final.csv")
    print("  - 2026_wr_rankings_final.csv")
    print("  - slap_vs_draft_position.png")
    print("  - backtest_summary.png")
