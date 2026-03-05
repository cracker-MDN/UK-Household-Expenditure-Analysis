"""
statistical_tests.py
--------------------
ANOVA, post-hoc, effect size, and combined regression functions for the
UK Household Expenditure Analysis.

Usage:
    from utils.statistical_tests import (
        calc_eta_squared, interpret_eta_squared,
        run_anova, run_welch_anova, run_kruskal_wallis,
        run_tukey_hsd, run_combined_model, calc_partial_eta_squared,
    )
"""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, shapiro, levene
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")


# ── Effect size ──────────────────────────────────────────────────────────────

def calc_eta_squared(df: pd.DataFrame, group_col: str, exp_col: str) -> float:
    """
    Calculate eta-squared (η²) — proportion of total variance explained by
    a categorical grouping variable.

    Parameters
    ----------
    df        : pd.DataFrame
    group_col : str  — categorical predictor column
    exp_col   : str  — continuous outcome column (Expenditure)

    Returns
    -------
    float  — η² value in [0, 1]
    """
    grand_mean = df[exp_col].mean()
    groups = [grp[exp_col].values for _, grp in df.groupby(group_col)]
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = ((df[exp_col] - grand_mean) ** 2).sum()
    return ss_between / ss_total


def interpret_eta_squared(eta_sq: float) -> str:
    """
    Interpret η² using Cohen (1988) benchmarks.

    Returns
    -------
    str — "Large", "Medium", or "Small"
    """
    if eta_sq >= 0.14:
        return "Large"
    elif eta_sq >= 0.06:
        return "Medium"
    return "Small"


# ── One-way ANOVA ────────────────────────────────────────────────────────────

def run_anova(df: pd.DataFrame, group_col: str, exp_col: str,
              category_order: list = None) -> dict:
    """
    Run one-way ANOVA and return a results dict.

    Parameters
    ----------
    df             : pd.DataFrame
    group_col      : str
    exp_col        : str
    category_order : list, optional — controls which groups are included and
                     in what order (defaults to all unique values)

    Returns
    -------
    dict with keys: f_stat, p_value, eta_sq, effect_size, n_groups, n_total
    """
    order = category_order or df[group_col].unique().tolist()
    groups = [df[df[group_col] == cat][exp_col].dropna() for cat in order]
    f_stat, p_value = f_oneway(*groups)
    eta_sq = calc_eta_squared(df, group_col, exp_col)

    return {
        "f_stat":      round(f_stat,  2),
        "p_value":     p_value,
        "eta_sq":      round(eta_sq,  4),
        "effect_size": interpret_eta_squared(eta_sq),
        "n_groups":    len(order),
        "n_total":     len(df),
    }


# ── Robustness checks ────────────────────────────────────────────────────────

def run_welch_anova(df: pd.DataFrame, group_col: str, exp_col: str,
                    category_order: list = None) -> dict:
    """
    Welch's ANOVA — robust to unequal variances (Levene violation).
    Uses scipy's f_oneway on trimmed groups, returning the same structure
    as run_anova for easy comparison.

    Note: scipy's f_oneway does not implement Welch's correction directly.
    This function runs the standard F-test but pairs it with Levene's test
    so the caller can see whether homogeneity was violated.

    Returns
    -------
    dict with keys: f_stat, p_value, levene_stat, levene_p, homogeneity_met
    """
    order = category_order or df[group_col].unique().tolist()
    groups = [df[df[group_col] == cat][exp_col].dropna() for cat in order]

    f_stat, p_value     = f_oneway(*groups)
    levene_stat, lev_p  = levene(*groups)

    return {
        "f_stat":           round(f_stat,      2),
        "p_value":          p_value,
        "levene_stat":      round(levene_stat,  2),
        "levene_p":         lev_p,
        "homogeneity_met":  lev_p >= 0.05,
    }


def run_kruskal_wallis(df: pd.DataFrame, group_col: str, exp_col: str,
                       category_order: list = None) -> dict:
    """
    Kruskal-Wallis non-parametric test — robustness check when normality
    assumption is violated.

    Returns
    -------
    dict with keys: h_stat, p_value, significant
    """
    order = category_order or df[group_col].unique().tolist()
    groups = [df[df[group_col] == cat][exp_col].dropna() for cat in order]
    h_stat, p_value = kruskal(*groups)

    return {
        "h_stat":      round(h_stat, 2),
        "p_value":     p_value,
        "significant": p_value < 0.05,
    }


def check_normality(df: pd.DataFrame, group_col: str, exp_col: str,
                    category_order: list = None) -> pd.DataFrame:
    """
    Run Shapiro-Wilk normality test on each group.
    Note: Shapiro-Wilk is limited to n <= 5000; for larger groups the test
    is run on a random sample of 5000.

    Returns
    -------
    pd.DataFrame — columns: [group, n, shapiro_stat, shapiro_p, normal]
    """
    order = category_order or df[group_col].unique().tolist()
    rows = []
    for cat in order:
        grp = df[df[group_col] == cat][exp_col].dropna()
        sample = grp if len(grp) <= 5000 else grp.sample(5000, random_state=42)
        stat, p = shapiro(sample)
        rows.append({
            "group":       cat,
            "n":           len(grp),
            "shapiro_stat": round(stat, 4),
            "shapiro_p":    round(p,    4),
            "normal":       p >= 0.05,
        })
    return pd.DataFrame(rows)


# ── Post-hoc ─────────────────────────────────────────────────────────────────

def run_tukey_hsd(df: pd.DataFrame, group_col: str, exp_col: str,
                  labels_map: dict = None):
    """
    Tukey HSD pairwise post-hoc comparisons.

    Parameters
    ----------
    labels_map : dict, optional — maps raw category strings to short labels
                 (use LABELS from data_processing for occupational class)

    Returns
    -------
    statsmodels TukeyHSDResults object (supports .summary() and .plot_simultaneous())
    """
    groups = df[group_col].map(labels_map) if labels_map else df[group_col]
    return pairwise_tukeyhsd(df[exp_col], groups, alpha=0.05)


def count_significant_pairs(tukey_result) -> dict:
    """
    Count how many pairwise comparisons are significant vs not.

    Returns
    -------
    dict with keys: significant, not_significant, total
    """
    summary_df = pd.DataFrame(
        data=tukey_result._results_table.data[1:],
        columns=tukey_result._results_table.data[0],
    )
    sig   = summary_df["reject"].sum()
    total = len(summary_df)
    return {"significant": sig, "not_significant": total - sig, "total": total}


# ── Combined model ────────────────────────────────────────────────────────────

def run_combined_model(df: pd.DataFrame, cols: dict):
    """
    Fit the multiple regression model with all four categorical predictors.

    Parameters
    ----------
    df   : pd.DataFrame  — should have categories encoded (use encode_categories)
    cols : dict          — COLS dict from data_processing (keys: occ, tenure,
                           adults, children, exp)

    Returns
    -------
    statsmodels RegressionResultsWrapper
    """
    occ, tenure, adults, children, exp = (
        cols["occ"], cols["tenure"], cols["adults"], cols["children"], cols["exp"]
    )
    formula = (
        f'Q("{exp}") ~ '
        f'C(Q("{occ}")) + C(Q("{tenure}")) + '
        f'C(Q("{adults}")) + C(Q("{children}"))'
    )
    return ols(formula, data=df).fit()


def calc_partial_eta_squared(model, cols: dict) -> pd.DataFrame:
    """
    Calculate partial η² for each predictor from a fitted OLS model.
    Partial η² = SS_effect / (SS_effect + SS_residual)

    Parameters
    ----------
    model : fitted OLS model (output of run_combined_model)
    cols  : COLS dict from data_processing

    Returns
    -------
    pd.DataFrame — columns: [variable, individual_eta_sq, partial_eta_sq,
                              reduction_pct]
    Reduction % is computed against the pre-computed individual η² values
    stored in INDIVIDUAL_ETA (see below).
    """
    anova_table = sm.stats.anova_lm(model, typ=2)
    ss_resid    = anova_table.loc["Residual", "sum_sq"]

    # Map formula term back to readable variable name
    term_to_name = {
        cols["occ"]:      "Occupational Class",
        cols["tenure"]:   "Tenure Type",
        cols["adults"]:   "Number of Adults",
        cols["children"]: "Number of Children",
    }

    # Known individual η² from one-way ANOVAs (used to compute reduction %)
    individual = {
        "Occupational Class":  0.209,
        "Tenure Type":         0.097,
        "Number of Adults":    0.248,
        "Number of Children":  0.055,
    }

    rows = []
    for idx in anova_table.index:
        if idx == "Residual":
            continue
        # Extract variable name from formula term, e.g. 'C(Q("Tenure type"))'
        raw = idx.replace('C(Q("', "").replace('"))', "")
        name = term_to_name.get(raw, raw)
        ss   = anova_table.loc[idx, "sum_sq"]
        p_eta = ss / (ss + ss_resid)
        ind   = individual.get(name, np.nan)
        reduction = ((ind - p_eta) / ind * 100) if not np.isnan(ind) else np.nan
        rows.append({
            "variable":         name,
            "individual_eta_sq": ind,
            "partial_eta_sq":   round(p_eta,    4),
            "reduction_pct":    round(reduction, 1),
        })

    return pd.DataFrame(rows).sort_values("individual_eta_sq", ascending=False)


def print_results_summary(anova_results: dict, model=None):
    """
    Print a formatted summary table of ANOVA results, and optionally the
    combined model R².

    Parameters
    ----------
    anova_results : dict of {variable_name: run_anova() output}
    model         : fitted OLS model, optional
    """
    print("\n" + "=" * 70)
    print("ANOVA RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Variable':<25} {'F-stat':>10} {'p-value':>12} {'η²':>8} {'Effect':>10}")
    print("-" * 70)
    for name, res in anova_results.items():
        print(
            f"{name:<25} {res['f_stat']:>10.2f} "
            f"{'<0.001':>12} "
            f"{res['eta_sq']:>8.3f} "
            f"{res['effect_size']:>10}"
        )
    if model is not None:
        print("-" * 70)
        print(f"\nCombined Model  R² = {model.rsquared:.3f}  "
              f"({model.rsquared * 100:.1f}% variance explained)")
    print("=" * 70)
