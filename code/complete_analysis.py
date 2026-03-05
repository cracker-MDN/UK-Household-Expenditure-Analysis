"""
============================================================
MAYERFELD DATA ANALYSIS PRACTICUM - COMPLETE CASE STUDY
============================================================
Research Question:
Is There a Relationship Between Occupational Class, Tenure Type, 
Number of Adults, Number of Children and Expenditure of a Household?

Data: LCF_cleaned.csv (Living Costs and Food Survey 2013)
============================================================

INSTRUCTIONS:
1. Place LCF_cleaned.csv in the same folder as this script
2. Run: python complete_analysis.py
3. All figures will be saved as PNG files
4. Statistical results will be printed to console

REQUIRED PACKAGES:
pip install pandas numpy matplotlib seaborn scipy statsmodels
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, shapiro, levene, kruskal
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
# Column names (using cleaned version)
OCC = 'NS - SEC 8 Class of household reference person'
TENURE = 'Tenure type'
ADULTS = 'Number of adults'
CHILDREN = 'Number of children'
EXP = 'Expenditure'

# Category orders
occ_order = [
    'Higher managerial, administrative and professional occupations',
    'Intermediate occupations',
    'Routine and manual occupations',
    'Never worked and long term unemployed, students and occupation not stated',
    'Not classified for other reasons'
]
occ_labels = {
    'Higher managerial, administrative and professional occupations': 'Higher managerial',
    'Intermediate occupations': 'Intermediate',
    'Routine and manual occupations': 'Routine & manual',
    'Never worked and long term unemployed, students and occupation not stated': 'Never worked/unemployed',
    'Not classified for other reasons': 'Not classified'
}
tenure_order = ['Owned', 'Private rented', 'Public rented']
adults_order = ['1 adult', '2 adults', '3 adults', '4 and more adults']
children_order = ['No children', 'One child', 'Two or more children']

# Colors
colors = {
    'occ': ['#27AE60', '#3498DB', '#9B59B6', '#E74C3C', '#F39C12'],
    'tenure': ['#27AE60', '#3498DB', '#E67E22'],
    'adults': ['#E74C3C', '#3498DB', '#27AE60', '#9B59B6'],
    'children': ['#9B59B6', '#3498DB', '#27AE60']
}

# Plot style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
})

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def calc_eta_squared(df, col):
    """Calculate eta-squared effect size for a grouping variable."""
    groups = [df[df[col] == cat][EXP] for cat in df[col].unique()]
    ss_between = sum(len(g) * (g.mean() - df[EXP].mean())**2 for g in groups)
    ss_total = sum((df[EXP] - df[EXP].mean())**2)
    return ss_between / ss_total

def run_anova(df, col):
    """Run one-way ANOVA and return F-statistic and p-value."""
    groups = [df[df[col] == cat][EXP] for cat in df[col].unique()]
    f, p = f_oneway(*groups)
    return f, p

def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)

# ============================================================
# MAIN ANALYSIS
# ============================================================
def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('LCF_cleaned.csv')
    print(f"Dataset shape: {df.shape}")
    
    # ============================================================
    # FIGURE 1: Expenditure Distribution
    # ============================================================
    print_section("DEPENDENT VARIABLE: EXPENDITURE")
    
    print(f"\nDescriptive Statistics:")
    print(f"  Mean:   £{df[EXP].mean():.0f}/week")
    print(f"  Median: £{df[EXP].median():.0f}/week")
    print(f"  SD:     £{df[EXP].std():.0f}")
    print(f"  Min:    £{df[EXP].min():.0f}/week")
    print(f"  Max:    £{df[EXP].max():.0f}/week")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(df[EXP], bins=40, color='#3498DB', alpha=0.7, edgecolor='white')
    axes[0].axvline(df[EXP].mean(), color='#E74C3C', linestyle='--', linewidth=2, 
                    label=f'Mean: £{df[EXP].mean():.0f}')
    axes[0].axvline(df[EXP].median(), color='#27AE60', linestyle='-', linewidth=2, 
                    label=f'Median: £{df[EXP].median():.0f}')
    axes[0].set_xlabel('Weekly Expenditure (£)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Weekly Household Expenditure')
    axes[0].legend()
    
    bp = axes[1].boxplot(df[EXP], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#3498DB')
    bp['boxes'][0].set_alpha(0.7)
    axes[1].set_ylabel('Weekly Expenditure (£)')
    axes[1].set_title('Expenditure Spread')
    axes[1].set_xticklabels(['All Households'])
    
    plt.tight_layout()
    plt.savefig('fig01_expenditure_distribution.png', bbox_inches='tight')
    plt.close()
    print("\nSaved: fig01_expenditure_distribution.png")
    
    # ============================================================
    # ANALYZE EACH VARIABLE
    # ============================================================
    variables = [
        ('Occupational Class', OCC, occ_order, occ_labels, colors['occ'], 'fig02'),
        ('Tenure Type', TENURE, tenure_order, None, colors['tenure'], 'fig03'),
        ('Number of Adults', ADULTS, adults_order, None, colors['adults'], 'fig04'),
        ('Number of Children', CHILDREN, children_order, None, colors['children'], 'fig05')
    ]
    
    results = {}
    
    for name, col, order, labels, color_list, fig_prefix in variables:
        print_section(f"{name.upper()}")
        
        # Descriptive statistics
        print(f"\nDescriptive Statistics:")
        desc = df.groupby(col)[EXP].agg(['count', 'mean', 'std', 'median']).reindex(order)
        if labels:
            desc.index = [labels[x] for x in desc.index]
        desc.columns = ['N', 'Mean (£)', 'SD (£)', 'Median (£)']
        print(desc.round(2))
        
        # ANOVA
        groups = [df[df[col] == cat][EXP] for cat in order]
        f_stat, p_val = f_oneway(*groups)
        eta_sq = calc_eta_squared(df, col)
        
        # Store results
        results[name] = {'f_stat': f_stat, 'p_val': p_val, 'eta_sq': eta_sq}
        
        print(f"\nOne-way ANOVA:")
        print(f"  F-statistic: {f_stat:.2f}")
        print(f"  p-value: {p_val:.2e}")
        print(f"  Eta-squared: {eta_sq:.4f} ({eta_sq*100:.1f}%)")
        effect = 'LARGE' if eta_sq >= 0.14 else 'MEDIUM' if eta_sq >= 0.06 else 'SMALL'
        print(f"  Effect size: {effect}")
        
        # Kruskal-Wallis robustness check
        kw_stat, kw_p = kruskal(*groups)
        print(f"\nKruskal-Wallis (robustness):")
        print(f"  H = {kw_stat:.2f}, p = {kw_p:.2e}")
        
        # Post-hoc
        print(f"\nTukey HSD Post-hoc:")
        if labels:
            tukey = pairwise_tukeyhsd(df[EXP], df[col].map(labels), alpha=0.05)
        else:
            tukey = pairwise_tukeyhsd(df[EXP], df[col], alpha=0.05)
        print(tukey)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        means = df.groupby(col)[EXP].mean().reindex(order)
        ci = 1.96 * df.groupby(col)[EXP].sem().reindex(order)
        plot_labels = [labels[x] if labels else x for x in order]
        
        # Bar chart
        bars = axes[0].bar(range(len(order)), means, yerr=ci, capsize=5,
                           color=color_list, alpha=0.85, edgecolor='gray')
        axes[0].set_xticks(range(len(order)))
        axes[0].set_xticklabels(plot_labels, fontsize=9, rotation=15 if col == OCC else 0, 
                                 ha='right' if col == OCC else 'center')
        axes[0].set_ylabel('Mean Weekly Expenditure (£)')
        axes[0].set_title(f'Mean Expenditure by {name} (95% CI)')
        axes[0].axhline(df[EXP].mean(), color='red', linestyle='--', alpha=0.5)
        
        for bar, m in zip(bars, means):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15,
                         f'£{m:.0f}', ha='center', fontsize=10, fontweight='bold')
        
        # Boxplot
        bp = axes[1].boxplot([df[df[col] == cat][EXP] for cat in order],
                              labels=plot_labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], color_list):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        axes[1].set_ylabel('Weekly Expenditure (£)')
        axes[1].set_title(f'Expenditure Distribution by {name}')
        if col == OCC:
            axes[1].tick_params(axis='x', rotation=15)
        
        plt.suptitle(f'Figure: {name} and Household Expenditure', fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{fig_prefix}_{name.lower().replace(" ", "_")}.png', bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {fig_prefix}_{name.lower().replace(' ', '_')}.png")
    
    # ============================================================
    # COMPARISON FIGURE
    # ============================================================
    print_section("ALL VARIABLES COMPARISON")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    for idx, (name, col, order, labels, color_list, _) in enumerate(variables):
        ax = axes[idx // 2, idx % 2]
        means = df.groupby(col)[EXP].mean().reindex(order)
        ci = 1.96 * df.groupby(col)[EXP].sem().reindex(order)
        plot_labels = [labels[x][:15] + '...' if labels and len(labels[x]) > 15 else (labels[x] if labels else x) for x in order]
        
        bars = ax.bar(range(len(order)), means, yerr=ci, capsize=4,
                      color=color_list, alpha=0.85, edgecolor='gray')
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(plot_labels, fontsize=8, rotation=15 if col == OCC else 0,
                           ha='right' if col == OCC else 'center')
        ax.set_ylabel('Mean Expenditure (£)')
        ax.set_title(f'({chr(97+idx)}) {name}')
        ax.axhline(df[EXP].mean(), color='red', linestyle='--', alpha=0.5)
        
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                    f'£{m:.0f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Mean Expenditure by All Predictor Variables (95% CI)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('fig06_all_variables_comparison.png', bbox_inches='tight')
    plt.close()
    print("Saved: fig06_all_variables_comparison.png")
    
    # ============================================================
    # EFFECT SIZE COMPARISON
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    eta_values = {name: results[name]['eta_sq'] for name, _, _, _, _, _ in variables}
    eta_sorted = dict(sorted(eta_values.items(), key=lambda x: x[1]))
    colors_sorted = ['#9B59B6', '#3498DB', '#27AE60', '#E74C3C']
    
    bars = ax.barh(list(eta_sorted.keys()), list(eta_sorted.values()), 
                   color=colors_sorted, alpha=0.85)
    
    ax.set_xlabel('Eta-squared (η²) — Proportion of Variance Explained', fontsize=11)
    ax.set_title('Effect Size Comparison — Which Variable Best Predicts Expenditure?', fontsize=13)
    
    for bar, val in zip(bars, eta_sorted.values()):
        effect = 'LARGE' if val >= 0.14 else 'MEDIUM' if val >= 0.06 else 'SMALL'
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f} ({val*100:.1f}%) — {effect}', va='center', fontsize=10, fontweight='bold')
    
    ax.axvline(x=0.01, color='gray', linestyle=':', alpha=0.7, label='Small (0.01)')
    ax.axvline(x=0.06, color='gray', linestyle='--', alpha=0.7, label='Medium (0.06)')
    ax.axvline(x=0.14, color='gray', linestyle='-', alpha=0.7, label='Large (0.14)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 0.32)
    
    plt.tight_layout()
    plt.savefig('fig07_effect_size_comparison.png', bbox_inches='tight')
    plt.close()
    print("Saved: fig07_effect_size_comparison.png")
    
    # ============================================================
    # COMBINED MODEL
    # ============================================================
    print_section("COMBINED MODEL")
    
    model = ols(f'{EXP} ~ C(Q("{OCC}")) + C(Q("{TENURE}")) + C(Q("{ADULTS}")) + C(Q("{CHILDREN}"))', 
                data=df).fit()
    
    print(f"\nOverall R² = {model.rsquared:.4f}")
    print(f"Adjusted R² = {model.rsquared_adj:.4f}")
    print(f"\n{model.rsquared*100:.1f}% of expenditure variance explained by all four variables combined")
    
    # Type II ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\nType II ANOVA Table:")
    print(anova_table)
    
    # Partial eta-squared
    ss_resid = anova_table.loc['Residual', 'sum_sq']
    print("\nPartial Eta-squared (unique contribution):")
    partial_eta = {}
    for idx in anova_table.index:
        if idx != 'Residual':
            ss = anova_table.loc[idx, 'sum_sq']
            p_eta = ss / (ss + ss_resid)
            var_name = idx.replace('C(Q("', '').replace('"))', '')
            partial_eta[var_name] = p_eta
            print(f"  {var_name}: {p_eta:.4f}")
    
    # Effect overlap figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    individual_eta = {
        'Number of Adults': results['Number of Adults']['eta_sq'],
        'Occupational Class': results['Occupational Class']['eta_sq'],
        'Tenure Type': results['Tenure Type']['eta_sq'],
        'Number of Children': results['Number of Children']['eta_sq']
    }
    
    # Map partial eta to readable names
    partial_eta_mapped = {
        'Number of Adults': partial_eta.get(ADULTS, 0),
        'Occupational Class': partial_eta.get(OCC, 0),
        'Tenure Type': partial_eta.get(TENURE, 0),
        'Number of Children': partial_eta.get(CHILDREN, 0)
    }
    
    x = np.arange(len(individual_eta))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, list(individual_eta.values()), width, 
                   label='Individual η²', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x + width/2, list(partial_eta_mapped.values()), width,
                   label='Partial η² (controlling for others)', color='#E74C3C', alpha=0.8)
    
    ax.set_ylabel('Eta-squared')
    ax.set_title('Individual vs. Partial Effect Sizes (Effect Overlap)')
    ax.set_xticks(x)
    ax.set_xticklabels(individual_eta.keys(), fontsize=10)
    ax.legend()
    ax.set_ylim(0, 0.3)
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('fig08_effect_overlap.png', bbox_inches='tight')
    plt.close()
    print("\nSaved: fig08_effect_overlap.png")
    
    # ============================================================
    # INTERACTION FIGURES
    # ============================================================
    # Occupational Class x Tenure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot = df.groupby([OCC, TENURE])[EXP].mean().unstack()
    pivot.index = [occ_labels.get(x, x) for x in pivot.index]
    pivot = pivot[tenure_order]
    
    pivot.plot(kind='bar', ax=ax, width=0.8, color=colors['tenure'])
    ax.set_ylabel('Mean Weekly Expenditure (£)')
    ax.set_xlabel('')
    ax.set_title('Mean Expenditure by Occupational Class and Tenure Type')
    ax.legend(title='Tenure Type', loc='upper right')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('fig09_interaction_occ_tenure.png', bbox_inches='tight')
    plt.close()
    print("Saved: fig09_interaction_occ_tenure.png")
    
    # Occupational Class x Adults
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot2 = df.groupby([OCC, ADULTS])[EXP].mean().unstack()
    pivot2.index = [occ_labels.get(x, x) for x in pivot2.index]
    pivot2 = pivot2[adults_order]
    
    pivot2.plot(kind='bar', ax=ax, width=0.8, color=colors['adults'])
    ax.set_ylabel('Mean Weekly Expenditure (£)')
    ax.set_xlabel('')
    ax.set_title('Mean Expenditure by Occupational Class and Number of Adults')
    ax.legend(title='Number of Adults', loc='upper right')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('fig10_interaction_occ_adults.png', bbox_inches='tight')
    plt.close()
    print("Saved: fig10_interaction_occ_adults.png")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print_section("SUMMARY OF ALL RESULTS")
    
    print(f"\n{'Variable':<25} {'F-stat':>10} {'p-value':>12} {'η²':>10} {'Effect':>10}")
    print("-"*70)
    for name, res in results.items():
        effect = 'LARGE' if res['eta_sq'] >= 0.14 else 'MEDIUM' if res['eta_sq'] >= 0.06 else 'SMALL'
        print(f"{name:<25} {res['f_stat']:>10.2f} {'<0.001':>12} {res['eta_sq']:>10.3f} {effect:>10}")
    
    print(f"\nCombined Model R² = {model.rsquared:.3f}")
    print(f"All four variables together explain {model.rsquared*100:.1f}% of expenditure variance")
    
    print("\nRANKING by effect size:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['eta_sq'], reverse=True)
    for i, (name, res) in enumerate(sorted_results, 1):
        print(f"  {i}. {name}: {res['eta_sq']*100:.1f}%")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated figures:")
    for f in sorted([f for f in os.listdir('.') if f.startswith('fig') and f.endswith('.png')]):
        print(f"  - {f}")

if __name__ == "__main__":
    main()
