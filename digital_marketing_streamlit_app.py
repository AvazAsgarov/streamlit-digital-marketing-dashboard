import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Digital Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='W')
    regions = ['North', 'South', 'East', 'West']
    products = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch']
    channels = ['Search Engine', 'Social Media', 'Email', 'Display Ads', 'Video Ads']
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    data = []
    for date in dates:
        for _ in range(np.random.randint(1, 4)):  
            row = {
                'Date': date,
                'Year': date.year,
                'Month': date.strftime('%B'),
                'Region': np.random.choice(regions),
                'Product': np.random.choice(products),
                'Marketing_Channel': np.random.choice(channels),
                'Quarter': f'Q{(date.month-1)//3 + 1}',
                'Spend': np.random.uniform(500, 5000),
                'Impressions': np.random.randint(5000, 50000),
                'Clicks': np.random.randint(100, 2000),
                'Conversions': np.random.randint(10, 200),
                'Revenue': np.random.uniform(1000, 15000),
                'CTR': np.random.uniform(0.02, 0.3),
                'CPC': np.random.uniform(1.5, 25),
                'ROI': np.random.uniform(1.5, 12)
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['Cost_per_Conversion'] = df['Spend'] / df['Conversions']
    df['Revenue_per_Click'] = df['Revenue'] / df['Clicks']
    df['Conversion_Rate'] = df['Conversions'] / df['Clicks']
    
    return df

# Load data
df = load_data()

# Sidebar for filters
st.sidebar.markdown("## 🎛️ Filters")

# Date filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Date'].min(), df['Date'].max()),
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

# Region filter
regions = st.sidebar.multiselect(
    "Select Regions",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

# Product filter
products = st.sidebar.multiselect(
    "Select Products",
    options=df['Product'].unique(),
    default=df['Product'].unique()
)

# Marketing Channel filter
channels = st.sidebar.multiselect(
    "Select Marketing Channels",
    options=df['Marketing_Channel'].unique(),
    default=df['Marketing_Channel'].unique()
)

# Apply filters
filtered_df = df[
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1])) &
    (df['Region'].isin(regions)) &
    (df['Product'].isin(products)) &
    (df['Marketing_Channel'].isin(channels))
]

# Main dashboard
st.markdown('<div class="main-header">📊 Digital Marketing Analytics Dashboard</div>', unsafe_allow_html=True)

# Key Metrics Cards
st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_spend = filtered_df['Spend'].sum()
    st.metric("Total Spend", f"${total_spend:,.0f}")

with col2:
    total_revenue = filtered_df['Revenue'].sum()
    st.metric("Total Revenue", f"${total_revenue:,.0f}")

with col3:
    overall_roi = total_revenue / total_spend if total_spend > 0 else 0
    st.metric("Overall ROI", f"{overall_roi:.2f}x")

with col4:
    avg_ctr = filtered_df['CTR'].mean()
    st.metric("Average CTR", f"{avg_ctr:.2f}%")

with col5:
    total_conversions = filtered_df['Conversions'].sum()
    st.metric("Total Conversions", f"{total_conversions:,.0f}")

# Dashboard tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🌍 Regional Analysis", "📱 Product Performance", "📢 Channel Analysis", "📅 Time Series", "🎯 Advanced Analytics"])

with tab1:
    st.markdown('<div class="section-header">Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Performance summary
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue vs Spend scatter plot
        fig_scatter = px.scatter(
            filtered_df, 
            x='Spend', 
            y='Revenue',
            color='Marketing_Channel',
            size='Conversions',
            hover_data=['Region', 'Product', 'ROI'],
            title="Revenue vs Spend by Marketing Channel"
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # ROI distribution
        fig_hist = px.histogram(
            filtered_df, 
            x='ROI', 
            nbins=20,
            title="ROI Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Top performers table
    st.markdown('<div class="section-header">🏆 Top Performing Campaigns</div>', unsafe_allow_html=True)
    
    top_campaigns = filtered_df.nlargest(10, 'ROI')[['Date', 'Region', 'Product', 'Marketing_Channel', 'Spend', 'Revenue', 'ROI', 'Conversions']]
    st.dataframe(top_campaigns.style.format({
        'Spend': '${:,.0f}',
        'Revenue': '${:,.0f}',
        'ROI': '{:.2f}x',
        'Conversions': '{:,.0f}'
    }), use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">🌍 Regional Performance Analysis</div>', unsafe_allow_html=True)
    
    # Regional metrics
    regional_metrics = filtered_df.groupby('Region').agg({
        'Spend': 'sum',
        'Revenue': 'sum',
        'Conversions': 'sum',
        'ROI': 'mean',
        'CTR': 'mean',
        'CPC': 'mean'
    }).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional revenue pie chart
        fig_pie = px.pie(
            values=regional_metrics['Revenue'],
            names=regional_metrics.index,
            title="Revenue Distribution by Region"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Regional ROI bar chart
        fig_bar = px.bar(
            x=regional_metrics.index,
            y=regional_metrics['ROI'],
            title="Average ROI by Region",
            color=regional_metrics['ROI'],
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Regional performance table
    st.markdown("### Regional Performance Summary")
    st.dataframe(regional_metrics.style.format({
        'Spend': '${:,.0f}',
        'Revenue': '${:,.0f}',
        'Conversions': '{:,.0f}',
        'ROI': '{:.2f}x',
        'CTR': '{:.2f}%',
        'CPC': '${:.2f}'
    }), use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">📱 Product Performance Analysis</div>', unsafe_allow_html=True)
    
    # Product metrics
    product_metrics = filtered_df.groupby('Product').agg({
        'Spend': 'sum',
        'Revenue': 'sum',
        'Conversions': 'sum',
        'ROI': 'mean',
        'CTR': 'mean',
        'Conversion_Rate': 'mean'
    }).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Product revenue comparison
        fig_bar = px.bar(
            x=product_metrics.index,
            y=product_metrics['Revenue'],
            title="Total Revenue by Product",
            color=product_metrics['Revenue'],
            color_continuous_scale='blues'
        )
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Product ROI vs Conversions
        fig_scatter = px.scatter(
            x=product_metrics['ROI'],
            y=product_metrics['Conversions'],
            text=product_metrics.index,
            title="ROI vs Conversions by Product",
            size=product_metrics['Revenue']
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Product performance heatmap
    st.markdown("### Product Performance Heatmap")
    
    # Create heatmap data
    heatmap_data = filtered_df.pivot_table(
        values='ROI', 
        index='Product', 
        columns='Marketing_Channel', 
        aggfunc='mean'
    ).round(2)
    
    fig_heatmap = px.imshow(
        heatmap_data,
        title="Average ROI: Product vs Marketing Channel",
        color_continuous_scale='RdYlBu_r',
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">📢 Marketing Channel Analysis</div>', unsafe_allow_html=True)
    
    # Channel metrics
    channel_metrics = filtered_df.groupby('Marketing_Channel').agg({
        'Spend': 'sum',
        'Revenue': 'sum',
        'Clicks': 'sum',
        'Conversions': 'sum',
        'ROI': 'mean',
        'CTR': 'mean',
        'CPC': 'mean'
    }).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Channel efficiency radar chart
        fig_radar = go.Figure()
        
        # Normalize metrics for radar chart
        normalized_metrics = channel_metrics[['ROI', 'CTR', 'Conversions']].copy()
        for col in normalized_metrics.columns:
            normalized_metrics[col] = (normalized_metrics[col] - normalized_metrics[col].min()) / (normalized_metrics[col].max() - normalized_metrics[col].min())
        
        for channel in normalized_metrics.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=[normalized_metrics.loc[channel, 'ROI'],
                   normalized_metrics.loc[channel, 'CTR'],
                   normalized_metrics.loc[channel, 'Conversions']],
                theta=['ROI', 'CTR', 'Conversions'],
                fill='toself',
                name=channel
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Channel Performance Comparison (Normalized)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Channel spend vs revenue
        fig_bubble = px.scatter(
            x=channel_metrics['Spend'],
            y=channel_metrics['Revenue'],
            size=channel_metrics['Conversions'],
            color=channel_metrics['ROI'],
            hover_name=channel_metrics.index,
            title="Spend vs Revenue by Channel",
            labels={'x': 'Total Spend', 'y': 'Total Revenue'},
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    
    # Channel performance comparison
    st.markdown("### Channel Performance Metrics")
    st.dataframe(channel_metrics.style.format({
        'Spend': '${:,.0f}',
        'Revenue': '${:,.0f}',
        'Clicks': '{:,.0f}',
        'Conversions': '{:,.0f}',
        'ROI': '{:.2f}x',
        'CTR': '{:.2f}%',
        'CPC': '${:.2f}'
    }), use_container_width=True)

with tab5:
    st.markdown('<div class="section-header">📅 Time Series Analysis</div>', unsafe_allow_html=True)
    
    # Time series aggregation
    time_series = filtered_df.groupby('Date').agg({
        'Spend': 'sum',
        'Revenue': 'sum',
        'Clicks': 'sum',
        'Conversions': 'sum',
        'ROI': 'mean'
    }).reset_index()
    
    # Multi-line time series
    fig_ts = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Over Time', 'ROI Over Time', 'Clicks Over Time', 'Conversions Over Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Revenue
    fig_ts.add_trace(
        go.Scatter(x=time_series['Date'], y=time_series['Revenue'], name='Revenue', line=dict(color='blue')),
        row=1, col=1
    )
    
    # ROI
    fig_ts.add_trace(
        go.Scatter(x=time_series['Date'], y=time_series['ROI'], name='ROI', line=dict(color='green')),
        row=1, col=2
    )
    
    # Clicks
    fig_ts.add_trace(
        go.Scatter(x=time_series['Date'], y=time_series['Clicks'], name='Clicks', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Conversions
    fig_ts.add_trace(
        go.Scatter(x=time_series['Date'], y=time_series['Conversions'], name='Conversions', line=dict(color='red')),
        row=2, col=2
    )
    
    fig_ts.update_layout(height=600, title_text="Performance Metrics Over Time")
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Monthly trends
    st.markdown("### Monthly Trends")
    
    monthly_data = filtered_df.groupby('Month').agg({
        'Spend': 'sum',
        'Revenue': 'sum',
        'ROI': 'mean',
        'Conversions': 'sum'
    }).round(2)
    
    # Month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_data = monthly_data.reindex([m for m in month_order if m in monthly_data.index])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_monthly_rev = px.bar(
            x=monthly_data.index,
            y=monthly_data['Revenue'],
            title="Monthly Revenue Trend",
            color=monthly_data['Revenue'],
            color_continuous_scale='blues'
        )
        fig_monthly_rev.update_xaxes(tickangle=45)
        st.plotly_chart(fig_monthly_rev, use_container_width=True)
    
    with col2:
        fig_monthly_roi = px.line(
            x=monthly_data.index,
            y=monthly_data['ROI'],
            title="Monthly ROI Trend",
            markers=True
        )
        fig_monthly_roi.update_xaxes(tickangle=45)
        st.plotly_chart(fig_monthly_roi, use_container_width=True)

with tab6:
    st.markdown('<div class="section-header">🎯 Advanced Analytics</div>', unsafe_allow_html=True)
    
    # ROI Segmentation
    st.markdown("### ROI Performance Segmentation")
    
    # Create ROI segments
    filtered_df['ROI_Segment'] = pd.cut(
        filtered_df['ROI'], 
        bins=[-float('inf'), 2, 5, 10, float('inf')], 
        labels=['Low (≤2)', 'Medium (2-5)', 'High (5-10)', 'Very High (>10)']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI segments distribution
        roi_segments = filtered_df['ROI_Segment'].value_counts()
        fig_pie_roi = px.pie(
            values=roi_segments.values,
            names=roi_segments.index,
            title="Campaign Distribution by ROI Segments"
        )
        st.plotly_chart(fig_pie_roi, use_container_width=True)
    
    with col2:
        # Segment performance
        segment_performance = filtered_df.groupby('ROI_Segment').agg({
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum'
        })
        
        fig_segment = px.bar(
            x=segment_performance.index,
            y=segment_performance['Revenue'],
            title="Revenue by ROI Segments",
            color=segment_performance['Revenue'],
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_segment, use_container_width=True)
    
    # Correlation Analysis
    st.markdown("### Correlation Analysis")
    
    correlation_cols = ['Spend', 'Impressions', 'Clicks', 'Conversions', 'Revenue', 'CTR', 'CPC', 'ROI']
    correlation_matrix = filtered_df[correlation_cols].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        title="Correlation Matrix of Key Metrics",
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Outlier Detection
    st.markdown("### Outlier Detection")
    
    # Select metric for outlier analysis
    outlier_metric = st.selectbox(
        "Select metric for outlier analysis:",
        ['Spend', 'Revenue', 'ROI', 'CTR', 'CPC']
    )
    
    # Calculate outliers using IQR method
    Q1 = filtered_df[outlier_metric].quantile(0.25)
    Q3 = filtered_df[outlier_metric].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = filtered_df[(filtered_df[outlier_metric] < lower_bound) | 
                          (filtered_df[outlier_metric] > upper_bound)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot
        fig_box = px.box(
            filtered_df,
            y=outlier_metric,
            title=f"{outlier_metric} Distribution with Outliers"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        st.write(f"**Outlier Analysis for {outlier_metric}:**")
        st.write(f"- Total outliers detected: {len(outliers)}")
        st.write(f"- Percentage of outliers: {(len(outliers)/len(filtered_df))*100:.1f}%")
        
        if len(outliers) > 0:
            st.write(f"- Outlier range: {outliers[outlier_metric].min():.2f} to {outliers[outlier_metric].max():.2f}")
            st.write(f"- Normal range: {lower_bound:.2f} to {upper_bound:.2f}")

# Footer
st.markdown("---")
st.markdown("### 📊 Dashboard Features")
st.markdown("""
- **Interactive Filters**: Use the sidebar to filter data by date, region, product, and marketing channel
- **Real-time Updates**: All visualizations update automatically based on your filter selections
- **Export Data**: You can export filtered data and charts for further analysis
- **Responsive Design**: Dashboard works on desktop and mobile devices
""")

# Data export functionality
st.markdown("### 📥 Data Export")
if st.button("Export Filtered Data to CSV"):
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"marketing_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Dataset info
with st.expander("📋 Dataset Information"):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Records:** {len(filtered_df):,}")
        st.write(f"**Date Range:** {filtered_df['Date'].min().strftime('%Y-%m-%d')} to {filtered_df['Date'].max().strftime('%Y-%m-%d')}")
        st.write(f"**Regions:** {', '.join(filtered_df['Region'].unique())}")
    with col2:
        st.write(f"**Products:** {', '.join(filtered_df['Product'].unique())}")
        st.write(f"**Marketing Channels:** {', '.join(filtered_df['Marketing_Channel'].unique())}")
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
