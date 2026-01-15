# app.py - COMPLETE DATA ANALYSIS APP FOR GITHUB
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ====================
# APP CONFIGURATION
# ====================
st.set_page_config(
    page_title="Data Analysis App | Scrum Project",
    page_icon="📊",
    layout="wide"
)

# ====================
# CUSTOM CSS FOR BETTER UI
# ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #F0F9FF;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ====================
# TITLE AND HEADER
# ====================
st.markdown('<h1 class="main-header">📊 Data Analysis Application</h1>', unsafe_allow_html=True)
st.markdown("**Agile Scrum Project | 5 Sprints Implementation | Team: MA, LJ, M, HZ**")
st.markdown("---")

# ====================
# CREATE DEMO DATA (ALWAYS WORKS)
# ====================
@st.cache_data
def create_demo_data():
    """Create sample commercial data for demo"""
    np.random.seed(42)
    n = 300
    
    # Generate dates
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    # Create dataframe
    data = {
        'Date': dates,
        'Product': np.random.choice(['Laptop', 'Smartphone', 'Tablet', 'Monitor', 'Printer'], n),
        'Category': np.random.choice(['Electronics', 'Office', 'Accessories'], n),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'Sales_Amount': np.round(np.random.uniform(100, 5000, n), 2),
        'Quantity': np.random.randint(1, 20, n),
        'Profit': np.round(np.random.uniform(-50, 500, n), 2),
        'Customer_Rating': np.random.choice([1, 2, 3, 4, 5, np.nan], n, p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1]),
        'Discount': np.round(np.random.uniform(0, 0.4, n), 2)
    }
    
    df = pd.DataFrame(data)
    
    # Add some duplicates
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    
    return df

# Load demo data
df = create_demo_data()

# ====================
# SIDEBAR - CONTROLS
# ====================
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)
    st.title("🎯 Navigation")
    
    st.markdown("### Scrum Sprints")
    sprints = [
        ("✅ Sprint 1", "Data Loading & Validation"),
        ("✅ Sprint 2", "Data Cleaning & Transformation"),
        ("✅ Sprint 3", "Data Visualization"),
        ("✅ Sprint 4", "Dashboard & Export"),
        ("✅ Sprint 5", "Analytical Report")
    ]
    
    for icon, desc in sprints:
        st.markdown(f"**{icon} {desc}**")
    
    st.markdown("---")
    st.markdown("### Dataset Info")
    st.info(f"""
    **Records:** {len(df):,}
    **Columns:** {len(df.columns)}
    **Size:** {df.memory_usage().sum() / 1024:.1f} KB
    """)
    
    # Upload option
    uploaded_file = st.file_uploader("📁 Upload your CSV", type=['csv'])
    if uploaded_file:
        try:
            user_df = pd.read_csv(uploaded_file)
            df = user_df
            st.success(f"✅ Loaded {len(df)} rows")
        except:
            st.error("❌ Error loading file. Using demo data.")

# ====================
# MAIN APP - TABS FOR EACH SPRINT
# ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 SPRINT 1: Load & Validate",
    "🧹 SPRINT 2: Clean & Transform", 
    "📈 SPRINT 3: Visualize",
    "📊 SPRINT 4: Dashboard",
    "📄 SPRINT 5: Report"
])

# ====================
# TAB 1: SPRINT 1 - LOAD & VALIDATE
# ====================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📥 Data Preview</h2>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, height=400)
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Validation Report</h2>', unsafe_allow_html=True)
        
        # Missing values
        missing = df.isnull().sum()
        st.metric("Missing Values", missing.sum())
        
        for col in df.columns:
            miss_count = missing[col]
            if miss_count > 0:
                st.warning(f"**{col}:** {miss_count} missing")
            else:
                st.success(f"**{col}:** ✓ Complete")
        
        # Duplicates
        dup_count = df.duplicated().sum()
        st.metric("Duplicate Rows", dup_count)
        
        # Data types
        st.markdown("**Data Types:**")
        for col in df.columns:
            dtype = str(df[col].dtype)
            st.caption(f"{col}: `{dtype}`")

# ====================
# TAB 2: SPRINT 2 - CLEAN & TRANSFORM
# ====================
with tab2:
    st.markdown('<h2 class="sub-header">🧹 Data Cleaning Tools</h2>', unsafe_allow_html=True)
    
    # Create two columns for cleaning options
    clean_col1, clean_col2 = st.columns(2)
    
    with clean_col1:
        st.markdown("#### Cleaning Actions")
        
        # Initialize cleaned dataframe in session state
        if 'cleaned_df' not in st.session_state:
            st.session_state.cleaned_df = df.copy()
        
        # Cleaning buttons
        if st.button("🔄 Remove Duplicates", use_container_width=True):
            st.session_state.cleaned_df = st.session_state.cleaned_df.drop_duplicates()
            st.success(f"Removed {df.duplicated().sum()} duplicates!")
        
        if st.button("🔧 Fill Missing Values", use_container_width=True):
            numeric_cols = st.session_state.cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if st.session_state.cleaned_df[col].isnull().any():
                    st.session_state.cleaned_df[col] = st.session_state.cleaned_df[col].fillna(
                        st.session_state.cleaned_df[col].mean()
                    )
            st.success("Filled missing numeric values with mean!")
        
        if st.button("📊 Normalize Text Columns", use_container_width=True):
            text_cols = st.session_state.cleaned_df.select_dtypes(include=['object']).columns
            for col in text_cols:
                st.session_state.cleaned_df[col] = st.session_state.cleaned_df[col].astype(str).str.strip().str.title()
            st.success("Text columns normalized!")
    
    with clean_col2:
        st.markdown("#### Data Transformation")
        
        # Filtering
        st.markdown("**Filter Data:**")
        filter_col = st.selectbox("Select column:", df.columns, key="filter_col")
        
        if filter_col:
            unique_values = df[filter_col].dropna().unique()
            selected_values = st.multiselect(
                f"Select values in {filter_col}:",
                options=list(unique_values)[:20],
                key="filter_values"
            )
            
            if selected_values:
                filtered_df = df[df[filter_col].isin(selected_values)]
                st.info(f"**Filtered:** {len(filtered_df)} rows")
                st.dataframe(filtered_df.head(10))
        
        # Simple aggregation
        st.markdown("**Quick Aggregation:**")
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            agg_col = st.selectbox("Column:", num_cols, key="agg_col")
            if agg_col:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sum", f"{df[agg_col].sum():.2f}")
                with col2:
                    st.metric("Mean", f"{df[agg_col].mean():.2f}")
                with col3:
                    st.metric("Min", f"{df[agg_col].min():.2f}")
                with col4:
                    st.metric("Max", f"{df[agg_col].max():.2f}")

# ====================
# TAB 3: SPRINT 3 - VISUALIZATION
# ====================
with tab3:
    st.markdown('<h2 class="sub-header">📈 Data Visualization</h2>', unsafe_allow_html=True)
    
    # Use cleaned data if available
    vis_df = st.session_state.get('cleaned_df', df)
    
    # Chart configuration
    config_col1, config_col2 = st.columns([1, 2])
    
    with config_col1:
        st.markdown("#### Chart Configuration")
        
        # Chart type
        chart_type = st.selectbox(
            "Select Chart Type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Pie Chart"],
            key="vis_chart_type"
        )
        
        # X-axis (for all charts)
        x_axis = st.selectbox("X-axis:", vis_df.columns, key="vis_x_axis")
        
        # Y-axis (for some charts)
        y_axis = None
        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
            num_cols = vis_df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                y_axis = st.selectbox("Y-axis:", num_cols, key="vis_y_axis")
        
        # Grouping
        group_by = st.selectbox(
            "Group by (optional):", 
            ["None"] + list(vis_df.columns),
            key="vis_group"
        )
    
    with config_col2:
        st.markdown("#### Visualization")
        
        # Create chart based on selection
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            if chart_type == "Bar Chart" and y_axis:
                if group_by != "None":
                    # Grouped bar chart
                    pivot_data = vis_df.groupby([x_axis, group_by])[y_axis].mean().unstack()
                    pivot_data.plot(kind='bar', ax=ax, width=0.8)
                    ax.legend(title=group_by)
                else:
                    # Simple bar chart
                    bar_data = vis_df.groupby(x_axis)[y_axis].mean()
                    bar_data.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_ylabel(y_axis)
                ax.set_title(f"Bar Chart: {y_axis} by {x_axis}")
            
            elif chart_type == "Line Chart" and y_axis:
                if group_by != "None":
                    # Multiple lines
                    for group in vis_df[group_by].dropna().unique():
                        group_data = vis_df[vis_df[group_by] == group]
                        line_data = group_data.groupby(x_axis)[y_axis].mean()
                        line_data.plot(label=group, ax=ax, marker='o')
                    ax.legend(title=group_by)
                else:
                    # Single line
                    line_data = vis_df.groupby(x_axis)[y_axis].mean()
                    line_data.plot(ax=ax, marker='o', color='green')
                ax.set_ylabel(y_axis)
                ax.set_title(f"Line Chart: {y_axis} by {x_axis}")
            
            elif chart_type == "Scatter Plot" and y_axis:
                if group_by != "None":
                    sns.scatterplot(data=vis_df, x=x_axis, y=y_axis, hue=group_by, ax=ax)
                else:
                    sns.scatterplot(data=vis_df, x=x_axis, y=y_axis, ax=ax)
                ax.set_title(f"Scatter Plot: {y_axis} vs {x_axis}")
            
            elif chart_type == "Histogram":
                vis_df[x_axis].hist(ax=ax, bins=20, color='purple', edgecolor='black')
                ax.set_xlabel(x_axis)
                ax.set_ylabel('Frequency')
                ax.set_title(f"Histogram of {x_axis}")
            
            elif chart_type == "Pie Chart":
                if x_axis in vis_df.columns:
                    pie_data = vis_df[x_axis].value_counts().head(10)
                    pie_data.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
                    ax.set_ylabel('')
                    ax.set_title(f"Pie Chart: Distribution of {x_axis}")
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download chart button
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            st.download_button(
                label="📥 Download Chart as PNG",
                data=buf,
                file_name=f"chart_{chart_type.replace(' ', '_')}.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Could not generate chart: {str(e)}")
            st.info("Try selecting different columns or chart type.")

# ====================
# TAB 4: SPRINT 4 - DASHBOARD & EXPORT
# ====================
with tab4:
    st.markdown('<h2 class="sub-header">📊 KPI Dashboard</h2>', unsafe_allow_html=True)
    
    # KPI Metrics Row
    st.markdown("### Key Performance Indicators")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with kpi_col2:
        if 'Sales_Amount' in df.columns:
            st.metric("Total Sales", f"${df['Sales_Amount'].sum():,.2f}")
        else:
            st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    
    with kpi_col3:
        if 'Profit' in df.columns:
            profit = df['Profit'].sum()
            st.metric("Total Profit", f"${profit:,.2f}", delta=f"{'Positive' if profit > 0 else 'Negative'}")
    
    with kpi_col4:
        if 'Customer_Rating' in df.columns:
            avg_rating = df['Customer_Rating'].mean()
            st.metric("Avg Rating", f"{avg_rating:.1f}/5")
    
    # Data Export Section
    st.markdown("---")
    st.markdown("### 📤 Export Data")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="commercial_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col2:
        # Excel Export
        @st.cache_data
        def convert_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            return output.getvalue()
        
        excel_data = convert_to_excel(df)
        st.download_button(
            label="📊 Download as Excel",
            data=excel_data,
            file_name="commercial_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with export_col3:
        # JSON Export
        json_data = df.to_json(orient='records')
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name="commercial_data.json",
            mime="application/json",
            use_container_width=True
        )

# ====================
# TAB 5: SPRINT 5 - ANALYTICAL REPORT
# ====================
with tab5:
    st.markdown('<h2 class="sub-header">📄 Analytical Report</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Report Configuration")
        
        report_title = st.text_input("Report Title:", "Commercial Data Analysis Report")
        
        st.markdown("**Include in report:**")
        include_data = st.checkbox("Dataset Overview", value=True)
        include_stats = st.checkbox("Statistical Summary", value=True)
        include_charts = st.checkbox("Key Charts", value=True)
        include_recommendations = st.checkbox("Recommendations", value=True)
        
        # Generate Report
        if st.button("🔄 Generate Full Report", type="primary", use_container_width=True):
            # Build report content
            report_content = f"""
            {'='*60}
            {report_title}
            {'='*60}
            
            Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            """
            
            if include_data:
                report_content += f"""
                DATASET OVERVIEW:
                - Total Records: {len(df):,}
                - Total Columns: {len(df.columns)}
                - Memory Usage: {df.memory_usage().sum() / 1024:.1f} KB
                - Date Range: {df['Date'].min().date() if 'Date' in df.columns else 'N/A'} to {df['Date'].max().date() if 'Date' in df.columns else 'N/A'}
                
                """
            
            if include_stats:
                report_content += """
                STATISTICAL SUMMARY:
                """
                num_cols = df.select_dtypes(include=[np.number]).columns
                for col in num_cols[:5]:  # Limit to 5 columns
                    report_content += f"""
                {col}:
                  - Mean: {df[col].mean():.2f}
                  - Median: {df[col].median():.2f}
                  - Std Dev: {df[col].std():.2f}
                  - Min: {df[col].min():.2f}
                  - Max: {df[col].max():.2f}
                """
            
            if include_recommendations:
                report_content += """
                RECOMMENDATIONS:
                1. Consider implementing automated data validation for new uploads
                2. Add predictive analytics for sales forecasting
                3. Implement user authentication for multi-user access
                4. Add real-time data integration from APIs
                5. Create mobile-responsive version of the dashboard
                """
            
            # Save to session state
            st.session_state.report_content = report_content
            st.success("✅ Report generated successfully!")
    
    with col2:
        st.markdown("#### Download Report")
        
        if 'report_content' in st.session_state:
            # Download as text file
            st.download_button(
                label="📄 Download Report (.txt)",
                data=st.session_state.report_content,
                file_name="data_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Show preview
            with st.expander("📋 Report Preview"):
                st.text(st.session_state.report_content[:500] + "...")
        else:
            st.info("Generate a report first to download")

# ====================
# FOOTER
# ====================
st.markdown("---")
st.markdown("### 🎯 Scrum Project Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown("""
    **✅ All Sprints Completed**
    - Sprint 1: Data Loading & Validation
    - Sprint 2: Data Cleaning & Transformation  
    - Sprint 3: Data Visualization
    - Sprint 4: Dashboard & Export
    - Sprint 5: Analytical Report
    """)

with summary_col2:
    st.markdown("""
    **✅ All User Stories Implemented**
    - US1: Load CSV files ✓
    - US2: Validate data ✓  
    - US3: Clean data ✓
    - US4: Transform data ✓
    - US5: Visualize data ✓
    - US6-US10: All completed ✓
    """)

with summary_col3:
    st.markdown("""
    **✅ Technical Achievements**
    - Python + Streamlit web app
    - Interactive data processing
    - Real-time visualization
    - Multi-format export
    - Agile methodology
    """)

st.markdown("---")
st.caption("**Data Analysis Application** | Built with Streamlit | Agile Scrum Project | 👥 Team: MA, LJ, M, HZ | 📅 " + pd.Timestamp.now().strftime('%Y-%m-%d'))
