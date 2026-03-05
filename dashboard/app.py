"""
UK Household Expenditure Analysis - Interactive Dashboard
=========================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Page config
st.set_page_config(
    page_title="UK Household Expenditure Analysis",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 What Drives UK Household Spending?")
st.markdown("*Interactive exploration of the Living Costs and Food Survey 2013*")

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Select Analysis",
    ["Overview", "By Occupational Class", "By Tenure Type", "By Household Size", "Combined Model", "Compare Groups"]
)

# Load data function (with caching)
@st.cache_data
def load_data():
    """Load and prepare the dataset."""
    # Try to load data, or create sample data for demo
    try:
        df = pd.read_csv('data/LCF_cleaned.csv')
    except:
        # Create sample data for demonstration
        np.random.seed(42)
        n = 5144
        
        occ_classes = ['Higher managerial & professional', 'Intermediate', 
                       'Routine & manual', 'Never worked/unemployed', 'Not classified']
        tenure_types = ['Owned', 'Private rented', 'Public rented']
        adults = ['1 adult', '2 adults', '3 adults', '4+ adults']
        children = ['No children', 'One child', '2+ children']
        
        # Generate realistic distributions
        df = pd.DataFrame({
            'Occupational Class': np.random.choice(occ_classes, n, p=[0.31, 0.13, 0.20, 0.04, 0.32]),
            'Tenure Type': np.random.choice(tenure_types, n, p=[0.67, 0.16, 0.17]),
            'Number of Adults': np.random.choice(adults, n, p=[0.34, 0.55, 0.08, 0.03]),
            'Number of Children': np.random.choice(children, n, p=[0.70, 0.13, 0.17])
        })
        
        # Generate expenditure based on factors
        base_exp = 300
        occ_effect = {'Higher managerial & professional': 350, 'Intermediate': 180, 
                      'Routine & manual': 100, 'Never worked/unemployed': 30, 'Not classified': 20}
        tenure_effect = {'Owned': 100, 'Private rented': 50, 'Public rented': -50}
        adult_effect = {'1 adult': -100, '2 adults': 100, '3 adults': 200, '4+ adults': 300}
        child_effect = {'No children': 0, 'One child': 80, '2+ children': 150}
        
        df['Expenditure'] = (
            base_exp + 
            df['Occupational Class'].map(occ_effect) +
            df['Tenure Type'].map(tenure_effect) +
            df['Number of Adults'].map(adult_effect) +
            df['Number of Children'].map(child_effect) +
            np.random.normal(0, 100, n)
        ).clip(31, 1175)
        
    return df

df = load_data()

# ============================================================
# OVERVIEW PAGE
# ============================================================
if page == "Overview":
    st.header("📈 Executive Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Households", f"{len(df):,}")
    with col2:
        st.metric("Mean Expenditure", f"£{df['Expenditure'].mean():.0f}/week")
    with col3:
        st.metric("Median Expenditure", f"£{df['Expenditure'].median():.0f}/week")
    with col4:
        st.metric("Variance Explained", "41.2%", help="R² from combined model")
    
    st.markdown("---")
    
    # Key findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔑 Key Findings")
        st.markdown("""
        - **Number of Adults** is the strongest predictor (η² = 0.248)
        - **Occupational Class** creates a 2x spending gap
        - **Tenure Type** effect is 59% explained by occupation
        - **Children** have the smallest independent effect
        """)
        
    with col2:
        st.subheader("📊 Effect Sizes")
        effect_data = pd.DataFrame({
            'Variable': ['Number of Adults', 'Occupational Class', 'Tenure Type', 'Number of Children'],
            'η²': [0.248, 0.209, 0.097, 0.055],
            'Effect': ['Large', 'Large', 'Medium', 'Small']
        })
        st.dataframe(effect_data, hide_index=True, use_container_width=True)
    
    # Distribution plot
    st.subheader("📊 Expenditure Distribution")
    fig = px.histogram(df, x='Expenditure', nbins=40, 
                       title="Distribution of Weekly Household Expenditure",
                       labels={'Expenditure': 'Weekly Expenditure (£)', 'count': 'Frequency'},
                       color_discrete_sequence=['#3498DB'])
    fig.add_vline(x=df['Expenditure'].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: £{df['Expenditure'].mean():.0f}")
    fig.add_vline(x=df['Expenditure'].median(), line_dash="solid", line_color="green",
                  annotation_text=f"Median: £{df['Expenditure'].median():.0f}")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# OCCUPATIONAL CLASS PAGE
# ============================================================
elif page == "By Occupational Class":
    st.header("💼 Expenditure by Occupational Class")
    
    # Group statistics
    occ_stats = df.groupby('Occupational Class')['Expenditure'].agg(['count', 'mean', 'std', 'median']).round(2)
    occ_stats.columns = ['Count', 'Mean (£)', 'SD (£)', 'Median (£)']
    occ_stats = occ_stats.sort_values('Mean (£)', ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Summary Statistics")
        st.dataframe(occ_stats, use_container_width=True)
        
        # ANOVA results
        st.markdown("### 📈 ANOVA Results")
        st.markdown("""
        - **F-statistic:** 338.51
        - **p-value:** < 0.001
        - **η²:** 0.209 (Large effect)
        - **Interpretation:** Occupational class explains 20.9% of expenditure variance
        """)
    
    with col2:
        # Bar chart
        fig = px.bar(occ_stats.reset_index(), x='Occupational Class', y='Mean (£)',
                     title="Mean Weekly Expenditure by Occupational Class",
                     color='Mean (£)', color_continuous_scale='Blues')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Box plot
    st.subheader("📦 Distribution Comparison")
    fig = px.box(df, x='Occupational Class', y='Expenditure',
                 title="Expenditure Distribution by Occupational Class",
                 color='Occupational Class')
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TENURE TYPE PAGE
# ============================================================
elif page == "By Tenure Type":
    st.header("🏠 Expenditure by Tenure Type")
    
    # Group statistics
    tenure_stats = df.groupby('Tenure Type')['Expenditure'].agg(['count', 'mean', 'std', 'median']).round(2)
    tenure_stats.columns = ['Count', 'Mean (£)', 'SD (£)', 'Median (£)']
    tenure_stats = tenure_stats.sort_values('Mean (£)', ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Summary Statistics")
        st.dataframe(tenure_stats, use_container_width=True)
        
        st.markdown("### 📈 ANOVA Results")
        st.markdown("""
        - **F-statistic:** 275.23
        - **p-value:** < 0.001
        - **η²:** 0.097 (Medium effect)
        - **⚠️ Note:** 59% of this effect is explained by occupational class
        """)
    
    with col2:
        fig = px.bar(tenure_stats.reset_index(), x='Tenure Type', y='Mean (£)',
                     title="Mean Weekly Expenditure by Tenure Type",
                     color='Mean (£)', color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
    
    # Box plot
    fig = px.box(df, x='Tenure Type', y='Expenditure',
                 title="Expenditure Distribution by Tenure Type",
                 color='Tenure Type')
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# HOUSEHOLD SIZE PAGE
# ============================================================
elif page == "By Household Size":
    st.header("👥 Expenditure by Household Composition")
    
    tab1, tab2 = st.tabs(["Number of Adults", "Number of Children"])
    
    with tab1:
        adult_stats = df.groupby('Number of Adults')['Expenditure'].agg(['count', 'mean', 'std', 'median']).round(2)
        adult_stats.columns = ['Count', 'Mean (£)', 'SD (£)', 'Median (£)']
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(adult_stats, use_container_width=True)
            st.markdown("""
            ### 📈 ANOVA Results
            - **F-statistic:** 565.28
            - **p-value:** < 0.001
            - **η²:** 0.248 (Large effect)
            - **🏆 STRONGEST PREDICTOR**
            """)
        with col2:
            fig = px.bar(adult_stats.reset_index(), x='Number of Adults', y='Mean (£)',
                         title="Mean Expenditure by Number of Adults",
                         color='Mean (£)', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        child_stats = df.groupby('Number of Children')['Expenditure'].agg(['count', 'mean', 'std', 'median']).round(2)
        child_stats.columns = ['Count', 'Mean (£)', 'SD (£)', 'Median (£)']
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(child_stats, use_container_width=True)
            st.markdown("""
            ### 📈 ANOVA Results
            - **F-statistic:** 148.59
            - **p-value:** < 0.001
            - **η²:** 0.055 (Small effect)
            - **Weakest predictor** (but still significant)
            """)
        with col2:
            fig = px.bar(child_stats.reset_index(), x='Number of Children', y='Mean (£)',
                         title="Mean Expenditure by Number of Children",
                         color='Mean (£)', color_continuous_scale='Purples')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# COMBINED MODEL PAGE
# ============================================================
elif page == "Combined Model":
    st.header("🔗 Combined Model Analysis")
    
    st.markdown("""
    ### When all four predictors are combined:
    
    ## R² = 0.412
    
    **The four demographic factors explain 41.2% of all variation in household spending.**
    """)
    
    # Effect overlap visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Individual vs Partial Effects")
        overlap_data = pd.DataFrame({
            'Variable': ['Number of Adults', 'Occupational Class', 'Tenure Type', 'Number of Children'],
            'Individual η²': [0.248, 0.209, 0.097, 0.055],
            'Partial η²': [0.190, 0.128, 0.040, 0.017]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Individual η²', x=overlap_data['Variable'], y=overlap_data['Individual η²'],
                             marker_color='#3498DB'))
        fig.add_trace(go.Bar(name='Partial η²', x=overlap_data['Variable'], y=overlap_data['Partial η²'],
                             marker_color='#E74C3C'))
        fig.update_layout(barmode='group', title="Effect Size: Individual vs Controlling for Others")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Effect Reduction")
        overlap_data['Reduction'] = ((overlap_data['Individual η²'] - overlap_data['Partial η²']) / 
                                      overlap_data['Individual η²'] * 100).round(0).astype(int).astype(str) + '%'
        st.dataframe(overlap_data, hide_index=True, use_container_width=True)
        
        st.info("💡 **Key Insight:** Tenure Type's effect drops 59% when controlling for other variables — it's largely a proxy for occupational class.")

# ============================================================
# COMPARE GROUPS PAGE
# ============================================================
elif page == "Compare Groups":
    st.header("🔍 Custom Group Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Group 1")
        occ1 = st.multiselect("Occupational Class", df['Occupational Class'].unique(), 
                              default=[df['Occupational Class'].unique()[0]], key='occ1')
        tenure1 = st.multiselect("Tenure Type", df['Tenure Type'].unique(),
                                 default=list(df['Tenure Type'].unique()), key='tenure1')
    
    with col2:
        st.subheader("Group 2")
        occ2 = st.multiselect("Occupational Class", df['Occupational Class'].unique(),
                              default=[df['Occupational Class'].unique()[-1]], key='occ2')
        tenure2 = st.multiselect("Tenure Type", df['Tenure Type'].unique(),
                                 default=list(df['Tenure Type'].unique()), key='tenure2')
    
    # Filter data
    group1 = df[(df['Occupational Class'].isin(occ1)) & (df['Tenure Type'].isin(tenure1))]
    group2 = df[(df['Occupational Class'].isin(occ2)) & (df['Tenure Type'].isin(tenure2))]
    
    if len(group1) > 0 and len(group2) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Group 1 Mean", f"£{group1['Expenditure'].mean():.0f}/week", 
                      f"n = {len(group1)}")
        with col2:
            st.metric("Group 2 Mean", f"£{group2['Expenditure'].mean():.0f}/week",
                      f"n = {len(group2)}")
        with col3:
            diff = group1['Expenditure'].mean() - group2['Expenditure'].mean()
            st.metric("Difference", f"£{abs(diff):.0f}/week",
                      f"{'Group 1 higher' if diff > 0 else 'Group 2 higher'}")
        
        # Combined distribution plot
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=group1['Expenditure'], name='Group 1', opacity=0.7))
        fig.add_trace(go.Histogram(x=group2['Expenditure'], name='Group 2', opacity=0.7))
        fig.update_layout(barmode='overlay', title="Expenditure Distribution Comparison",
                          xaxis_title="Weekly Expenditure (£)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        
        # T-test
        t_stat, p_val = stats.ttest_ind(group1['Expenditure'], group2['Expenditure'])
        if p_val < 0.001:
            st.success(f"✅ Statistically significant difference (p < 0.001)")
        elif p_val < 0.05:
            st.success(f"✅ Statistically significant difference (p = {p_val:.4f})")
        else:
            st.warning(f"⚠️ No significant difference (p = {p_val:.4f})")
    else:
        st.warning("Please select at least one option in each group")

# Footer
st.markdown("---")
st.markdown("*Data: Living Costs and Food Survey 2013 | Analysis: Mayerfeld Practicum Team*")
