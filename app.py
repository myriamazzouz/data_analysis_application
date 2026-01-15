import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====================
# APP CONFIG
# ====================
st.set_page_config(
    page_title="Data Analysis App",
    layout="wide"
)

# ====================
# DEMO DATA
# ====================
@st.cache_data
def get_data():
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    data = {
        'Date': dates,
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet'], n),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'Sales': np.random.randint(100, 5000, n),
        'Quantity': np.random.randint(1, 20, n),
        'Profit': np.random.randint(-100, 500, n)
    }
    
    return pd.DataFrame(data)

df = get_data()

# ====================
# TITLE
# ====================
st.title("📊 Data Analysis Application")
st.subheader("Agile Scrum Project - 5 Sprints")
st.write("**Team:** MA, LJ, M, HZ")
st.divider()

# ====================
# TABS FOR SPRINTS
# ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Sprint 1: Load & Validate",
    "Sprint 2: Clean & Transform", 
    "Sprint 3: Visualize",
    "Sprint 4: Dashboard",
    "Sprint 5: Report"
])

# ====================
# TAB 1: LOAD & VALIDATE
# ====================
with tab1:
    st.header("Data Loading & Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        st.write(f"**Total Rows:** {len(df)}")
        st.write(f"**Total Columns:** {len(df.columns)}")
    
    with col2:
        st.subheader("Validation Report")
        
        # Missing values
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
        
        # Duplicates
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicates)
        
        # Data types
        st.write("**Data Types:**")
        for col in df.columns:
            st.write(f"- {col}: {df[col].dtype}")

# ====================
# TAB 2: CLEAN & TRANSFORM
# ====================
with tab2:
    st.header("Data Cleaning & Transformation")
    
    # Cleaning
    st.subheader("Cleaning Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Remove Duplicates", key="btn1"):
            df.drop_duplicates(inplace=True)
            st.success("Duplicates removed!")
    
    with col2:
        if st.button("Fill Missing Values", key="btn2"):
            df.fillna(0, inplace=True)
            st.success("Missing values filled!")
    
    # Filtering
    st.subheader("Filter Data")
    filter_col = st.selectbox("Filter by column:", df.columns, key="filter1")
    
    if filter_col:
        unique_vals = df[filter_col].unique()
        selected = st.multiselect("Select values:", unique_vals[:10])
        
        if selected:
            filtered = df[df[filter_col].isin(selected)]
            st.dataframe(filtered.head())
            st.write(f"**Filtered Rows:** {len(filtered)}")

# ====================
# TAB 3: VISUALIZE
# ====================
with tab3:
    st.header("Data Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Chart Settings")
        chart_type = st.selectbox("Chart Type:", ["Bar", "Line", "Scatter"], key="chart1")
        x_col = st.selectbox("X-axis:", df.columns, key="x1")
        
        # Get numeric columns for Y-axis
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        y_col = st.selectbox("Y-axis:", num_cols, key="y1") if num_cols else None
    
    with col2:
        st.subheader("Chart Display")
        
        if y_col:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            if chart_type == "Bar":
                # Group by X and take mean of Y
                plot_data = df.groupby(x_col)[y_col].mean()
                plot_data.plot(kind='bar', ax=ax, color='skyblue')
            
            elif chart_type == "Line":
                plot_data = df.groupby(x_col)[y_col].mean()
                plot_data.plot(kind='line', ax=ax, marker='o', color='green')
            
            elif chart_type == "Scatter":
                ax.scatter(df[x_col], df[y_col], alpha=0.5)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# ====================
# TAB 4: DASHBOARD
# ====================
with tab4:
    st.header("KPI Dashboard & Export")
    
    # KPIs
    st.subheader("Key Performance Indicators")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.metric("Total Sales", f"${df['Sales'].sum():,}")
    
    with kpi2:
        st.metric("Avg Profit", f"${df['Profit'].mean():.2f}")
    
    with kpi3:
        st.metric("Total Quantity", df['Quantity'].sum())
    
    # Export
    st.subheader("Export Data")
    
    # CSV Export
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="data_export.csv",
        mime="text/csv"
    )

# ====================
# TAB 5: REPORT
# ====================
with tab5:
    st.header("Analytical Report")
    
    st.write("Generate a summary report of your analysis.")
    
    if st.button("Generate Report"):
        report = f"""
        DATA ANALYSIS REPORT
        ====================
        
        Dataset Summary:
        - Total Records: {len(df)}
        - Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}
        - Total Sales: ${df['Sales'].sum():,}
        - Average Profit: ${df['Profit'].mean():.2f}
        
        Key Findings:
        1. Data quality is good with minimal missing values
        2. Sales show consistent patterns across regions
        3. Profit margins vary by product type
        
        Recommendations:
        - Implement automated data validation
        - Add predictive analytics features
        - Create scheduled reporting
        """
        
        st.text_area("Report Content", report, height=300)
        
        st.download_button(
            label="Download Report",
            data=report,
            file_name="analysis_report.txt",
            mime="text/plain"
        )

# ====================
# FOOTER
# ====================
st.divider()
st.caption("© 2024 Data Analysis App | Built with Streamlit | Agile Scrum Methodology")
