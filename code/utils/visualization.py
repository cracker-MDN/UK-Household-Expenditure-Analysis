"""
visualization.py
----------------
All plotting functions for the UK Household Expenditure Analysis.
Each function saves a figure to the figures/ directory and returns
the matplotlib Figure object for optional further customisation.

Usage:
    from utils.visualization import (
        plot_expenditure_distribution,
        plot_group_means,
        plot_all_variables,
        plot_effect_sizes,
        plot_effect_overlap,
        plot_interaction,
        apply_plot_style,
    )
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Default plot style ────────────────────────────────────────────────────────

def apply_plot_style():
    """Apply consistent rcParams across all figures."""
    plt.rcParams.update({
        "figure.dpi":       150,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "font.size":        10,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "axes.titlepad":    10,
    })


# ── Colour palettes ───────────────────────────────────────────────────────────

PALETTE = {
    "occ":      ["#27AE60", "#3498DB", "#9B59B6", "#E74C3C", "#F39C12"],
    "tenure":   ["#27AE60", "#3498DB", "#E67E22"],
    "adults":   ["#E74C3C", "#3498DB", "#27AE60", "#9B59B6"],
    "children": ["#9B59B6", "#3498DB", "#27AE60"],
    "overlap":  ["#3498DB", "#E74C3C"],
}

FIGURES_DIR = "figures"


def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> plt.Figure:
    _ensure_figures_dir()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return fig


# ── Figure 1: Expenditure distribution ───────────────────────────────────────

def plot_expenditure_distribution(df: pd.DataFrame, exp_col: str,
                                  filename: str = "fig01_expenditure_distribution.png"
                                  ) -> plt.Figure:
    """
    Histogram + boxplot of the weekly expenditure distribution.

    Parameters
    ----------
    df      : pd.DataFrame
    exp_col : str  — name of the expenditure column
    filename: str  — output filename inside figures/
    """
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    exp = df[exp_col]

    # Histogram
    axes[0].hist(exp, bins=40, color="#3498DB", alpha=0.75, edgecolor="white")
    axes[0].axvline(exp.mean(),   color="#E74C3C", linestyle="--", linewidth=2,
                    label=f"Mean: £{exp.mean():.0f}")
    axes[0].axvline(exp.median(), color="#27AE60", linestyle="-",  linewidth=2,
                    label=f"Median: £{exp.median():.0f}")
    axes[0].set_xlabel("Weekly Expenditure (£)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Weekly Household Expenditure")
    axes[0].legend()

    # Boxplot
    bp = axes[1].boxplot(exp, vert=True, patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#3498DB")
    bp["boxes"][0].set_alpha(0.7)
    axes[1].set_ylabel("Weekly Expenditure (£)")
    axes[1].set_title("Expenditure Spread")
    axes[1].set_xticklabels(["All Households"])

    # Annotation box
    stats_text = (
        f"Mean:   £{exp.mean():.0f}\n"
        f"Median: £{exp.median():.0f}\n"
        f"SD:     £{exp.std():.0f}\n"
        f"Range:  £{exp.min():.0f} – £{exp.max():.0f}"
    )
    axes[1].text(1.35, exp.quantile(0.75), stats_text, fontsize=8.5,
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4F8", alpha=0.8))

    plt.tight_layout()
    return _save(fig, filename)


# ── Figures 2-5: Group means per predictor ───────────────────────────────────

def plot_group_means(df: pd.DataFrame, group_col: str, exp_col: str,
                     group_name: str, category_order: list,
                     color_list: list, labels_map: dict = None,
                     filename: str = None) -> plt.Figure:
    """
    Side-by-side bar chart (with 95% CI) and boxplot for one predictor variable.

    Parameters
    ----------
    df             : pd.DataFrame
    group_col      : str   — categorical predictor column
    exp_col        : str   — expenditure column
    group_name     : str   — human-readable variable name for titles
    category_order : list  — controls bar order
    color_list     : list  — one colour per category
    labels_map     : dict, optional — short labels for display
    filename       : str, optional — overrides auto-generated filename
    """
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    means = df.groupby(group_col)[exp_col].mean().reindex(category_order)
    ci    = 1.96 * df.groupby(group_col)[exp_col].sem().reindex(category_order)
    plot_labels = [labels_map[x] if labels_map else x for x in category_order]

    rotate = group_col.startswith("NS")   # rotate labels for occupational class

    # Bar chart
    bars = axes[0].bar(range(len(category_order)), means, yerr=ci, capsize=5,
                       color=color_list[: len(category_order)], alpha=0.85,
                       edgecolor="grey", error_kw={"elinewidth": 1.2})
    axes[0].set_xticks(range(len(category_order)))
    axes[0].set_xticklabels(plot_labels, fontsize=9,
                             rotation=20 if rotate else 0,
                             ha="right" if rotate else "center")
    axes[0].set_ylabel("Mean Weekly Expenditure (£)")
    axes[0].set_title(f"Mean Expenditure by {group_name} (95% CI)")
    axes[0].axhline(df[exp_col].mean(), color="red", linestyle="--",
                    alpha=0.45, linewidth=1, label="Overall mean")
    axes[0].legend(fontsize=8)

    for bar, m in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0,
                     bar.get_height() + (ci.max() * 0.08),
                     f"£{m:.0f}", ha="center", fontsize=10, fontweight="bold")

    # Boxplot
    bp = axes[1].boxplot(
        [df[df[group_col] == cat][exp_col] for cat in category_order],
        labels=plot_labels, patch_artist=True, widths=0.6,
    )
    for patch, c in zip(bp["boxes"], color_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("Weekly Expenditure (£)")
    axes[1].set_title(f"Expenditure Distribution by {group_name}")
    if rotate:
        axes[1].tick_params(axis="x", rotation=20)

    plt.suptitle(f"{group_name} and Household Expenditure", fontsize=13, y=1.01)
    plt.tight_layout()

    fname = filename or f"fig_group_{group_name.lower().replace(' ', '_')}.png"
    return _save(fig, fname)


# ── Figure 6: All four variables comparison ───────────────────────────────────

def plot_all_variables(df: pd.DataFrame, variables: list, exp_col: str,
                       filename: str = "fig06_all_variables_comparison.png"
                       ) -> plt.Figure:
    """
    2×2 subplot showing mean expenditure for all four predictors.

    Parameters
    ----------
    variables : list of tuples —
        (display_name, column, category_order, labels_map_or_None, color_list)
    """
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for idx, (name, col, order, labels, colors) in enumerate(variables):
        ax = axes[idx // 2, idx % 2]
        means = df.groupby(col)[exp_col].mean().reindex(order)
        ci    = 1.96 * df.groupby(col)[exp_col].sem().reindex(order)
        plabels = [
            (labels[x][:16] + "…" if labels and len(labels[x]) > 16
             else (labels[x] if labels else x))
            for x in order
        ]
        rotate = col.startswith("NS")

        bars = ax.bar(range(len(order)), means, yerr=ci, capsize=4,
                      color=colors[:len(order)], alpha=0.85, edgecolor="grey")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(plabels, fontsize=8,
                           rotation=20 if rotate else 0,
                           ha="right" if rotate else "center")
        ax.set_ylabel("Mean Expenditure (£)")
        ax.set_title(f"({chr(97 + idx)}) {name}")
        ax.axhline(df[exp_col].mean(), color="red", linestyle="--",
                   alpha=0.4, linewidth=1)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 12,
                    f"£{m:.0f}", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Mean Expenditure by All Predictor Variables (95% CI)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    return _save(fig, filename)


# ── Figure 7: Effect size comparison ─────────────────────────────────────────

def plot_effect_sizes(eta_values: dict,
                      filename: str = "fig07_effect_size_comparison.png"
                      ) -> plt.Figure:
    """
    Horizontal bar chart of η² values ranked by size.

    Parameters
    ----------
    eta_values : dict — {variable_name: eta_sq_float}
    """
    apply_plot_style()
    eta_sorted = dict(sorted(eta_values.items(), key=lambda x: x[1]))
    bar_colors = ["#9B59B6", "#3498DB", "#27AE60", "#E74C3C"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(list(eta_sorted.keys()), list(eta_sorted.values()),
                   color=bar_colors, alpha=0.85)

    ax.set_xlabel("Eta-squared (η²) — Proportion of Variance Explained", fontsize=11)
    ax.set_title("Effect Size Comparison — Which Variable Best Predicts Expenditure?",
                 fontsize=12)

    for bar, val in zip(bars, eta_sorted.values()):
        effect = "LARGE" if val >= 0.14 else "MEDIUM" if val >= 0.06 else "SMALL"
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2.0,
                f"{val:.3f} ({val * 100:.1f}%) — {effect}",
                va="center", fontsize=9.5, fontweight="bold")

    # Reference lines
    for x, ls, lbl in [(0.01, ":", "Small (0.01)"),
                        (0.06, "--", "Medium (0.06)"),
                        (0.14, "-",  "Large (0.14)")]:
        ax.axvline(x=x, color="grey", linestyle=ls, alpha=0.65, label=lbl)

    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 0.32)
    plt.tight_layout()
    return _save(fig, filename)


# ── Figure 8: Effect overlap ──────────────────────────────────────────────────

def plot_effect_overlap(overlap_df: pd.DataFrame,
                        filename: str = "fig08_effect_overlap.png"
                        ) -> plt.Figure:
    """
    Grouped bar chart comparing individual vs partial η² for each predictor.

    Parameters
    ----------
    overlap_df : pd.DataFrame — must have columns:
                 [variable, individual_eta_sq, partial_eta_sq, reduction_pct]
                 (output of calc_partial_eta_squared from statistical_tests)
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    x     = np.arange(len(overlap_df))
    width = 0.35

    b1 = ax.bar(x - width / 2, overlap_df["individual_eta_sq"], width,
                label="Individual η²", color=PALETTE["overlap"][0], alpha=0.85)
    b2 = ax.bar(x + width / 2, overlap_df["partial_eta_sq"],    width,
                label="Partial η² (controlling for others)",
                color=PALETTE["overlap"][1], alpha=0.85)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)

    # Reduction annotations
    for i, row in overlap_df.reset_index(drop=True).iterrows():
        ax.annotate(
            f"−{row['reduction_pct']:.0f}%",
            xy=(i + width / 2, row["partial_eta_sq"]),
            xytext=(i + width / 2, row["partial_eta_sq"] + 0.022),
            ha="center", fontsize=8, color="#E74C3C", fontweight="bold",
        )

    ax.set_ylabel("Eta-squared")
    ax.set_title("Individual vs. Partial Effect Sizes — Showing Effect Overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(overlap_df["variable"], fontsize=10)
    ax.legend()
    ax.set_ylim(0, 0.30)
    plt.tight_layout()
    return _save(fig, filename)


# ── Interaction plots ─────────────────────────────────────────────────────────

def plot_interaction(df: pd.DataFrame, row_col: str, col_col: str,
                     exp_col: str, row_order: list, col_order: list,
                     row_labels: dict = None, title: str = "",
                     color_list: list = None, filename: str = "fig_interaction.png"
                     ) -> plt.Figure:
    """
    Grouped bar chart showing mean expenditure across two categorical variables.

    Parameters
    ----------
    row_col    : str — primary grouping variable (y-axis grouping)
    col_col    : str — secondary grouping (creates grouped bars)
    row_order  : list — order of row_col categories
    col_order  : list — order of col_col categories
    row_labels : dict, optional — short labels for row_col
    color_list : list, optional — one colour per col_order category
    """
    apply_plot_style()
    pivot = df.groupby([row_col, col_col])[exp_col].mean().unstack()
    if row_labels:
        pivot.index = [row_labels.get(x, x) for x in pivot.index]
    pivot = pivot.reindex(columns=col_order)

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, width=0.78,
               color=(color_list or PALETTE["tenure"]), alpha=0.85)
    ax.set_ylabel("Mean Weekly Expenditure (£)")
    ax.set_xlabel("")
    ax.set_title(title or f"Mean Expenditure by {row_col} and {col_col}")
    ax.legend(title=col_col, loc="upper right")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return _save(fig, filename)
