import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Page Configurations Setup ---
st.set_page_config(page_title="PakWheels Data Dashboard", layout="wide",)

# Custom CSS for Beautiful UI
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    div[data-testid="metric-container"] {
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"] label {
        color: #6b7280 !important; /* Muted gray for the label */
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #111827 !important; /* Dark black for the actual number */
    }
    h1, h2, h3 {color: #1f2937;}
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    file_path = "data/pakwheels_cars_processed.csv"
    raw_path = "data/pakwheels_cars_raw.csv"
    
    df = pd.read_csv(file_path) if os.path.exists(file_path) else None
    
    # We load raw data just to calculate missing values for the EDA report
    raw_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    
    return df, raw_df

df, raw_df = load_data()

if df is None:
    st.error("🚨 Processed Data file not found! Please run `pakwheels_data_engineering.py`.")
    st.stop()

# --- Dashboard Header ---
st.title("Exploratory Data Analysis (EDA) Dashboard")
st.markdown("Automated Analysis & Visualizations for the Data Mining Project Dataset")
st.markdown("---")

# --- Key Performance Indicators (KPIs) ---
st.header("1. Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Vehicles Processed", value=f"{len(df):,}")
with col2:
    avg_price = df['price'].mean() if 'price' in df.columns else 0
    st.metric(label="Avg. Price (PKR)", value=f"{int(avg_price):,}" if avg_price else "N/A")
with col3:
    avg_mileage = df['mileage_km'].mean() if 'mileage_km' in df.columns else 0
    st.metric(label="Average Mileage (km)", value=f"{int(avg_mileage):,}" if avg_mileage else "N/A")
with col4:
    avg_age = df['car_age'].mean() if 'car_age' in df.columns else 0
    st.metric(label="Average Car Age", value=f"{round(avg_age, 1)} Years" if avg_age else "N/A")

st.markdown("---")

# --- Visualizations ---
col_charts1, col_charts2 = st.columns(2)

with col_charts1:
    st.subheader("Distribution of Car Prices")
    if 'price' in df.columns:
        fig_price = px.histogram(df, x='price', nbins=50, title="Price Range Frequency", 
                                 color_discrete_sequence=['#3b82f6'])
        fig_price.update_xaxes(title="Price (PKR)")
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.warning("Price column missing.")

with col_charts2:
    st.subheader("Price vs. Manufacturing Year")
    if 'year' in df.columns and 'price' in df.columns:
        fuel_col = 'fuel_type' if 'fuel_type' in df.columns else None
        hover_cols = ["title", "price"] if "title" in df.columns else None
        fig_year = px.scatter(df, x="year", y="price", color=fuel_col, title="Correlation: Year & Price",
                              opacity=0.6, hover_data=hover_cols)
        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.warning("Year or Price column missing.")

st.markdown("---")

col_charts3, col_charts4 = st.columns(2)

with col_charts3:
    st.subheader("Top Brands in Dataset")
    if 'brand_clean' in df.columns:
        brand_counts = df['brand_clean'].value_counts().reset_index()
        brand_counts.columns = ['Brand', 'Count']
        fig_brands = px.bar(brand_counts, x='Brand', y='Count', title="Market Share",
                            color='Count', color_continuous_scale="Blues")
        st.plotly_chart(fig_brands, use_container_width=True)
    else:
        st.warning("Brand column missing for this plot.")

with col_charts4:
    st.subheader("Transmission Type Distribution")
    if 'transmission' in df.columns:
        fig_trans = px.pie(df, names='transmission', title="Automatic vs Manual", hole=0.4, 
                           color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig_trans, use_container_width=True)
    else:
        st.warning("Transmission column missing.")

st.markdown("---")

st.header("2. Missing Value (NA) Resolution Report")
st.markdown("A crucial part of Exploratory Data Analysis (EDA) is handling `NaN` and `null` values. The engineering pipeline mathematically resolves missing vehicle properties to prevent Artificial Neural Network (ANN) breakage.")

if raw_df is not None:
    col_na1, col_na2 = st.columns(2)
    with col_na1:
        st.subheader("⚠️ Missing Values (Before Processing)")
        
        # Calculate NAs in raw dataset
        raw_na_counts = raw_df.isna().sum()
        # Add "Unknown" counts as missing
        for col in raw_df.columns:
            raw_na_counts[col] += (raw_df[col] == "Unknown").sum()
            
        raw_na_counts = raw_na_counts[raw_na_counts > 0].reset_index()
        raw_na_counts.columns = ['Feature', 'Missing Count']
        
        if not raw_na_counts.empty:
            fig_raw_na = px.bar(raw_na_counts, x='Feature', y='Missing Count', 
                                title="Missing Data Profile (Raw Dataset)", color_discrete_sequence=['#ef4444'])
            st.plotly_chart(fig_raw_na, use_container_width=True)
        else:
            st.success("Wow! No missing values in the raw dataset.")
            
    with col_na2:
        st.subheader("✅ Missing Values (After Statistical Imputation)")
        st.markdown("""
        **Imputation Logic Applied:**
        - **Numerical NA (Mileage, CC):** Imputed using the **Median** to prevent extreme outliers skewing results.
        - **Categorical NA (Body, Color):** Imputed using the **Mode** (Most Frequent Attribute).
        - **Target NA (Price):** Dropped strictly (Cannot guess target variable).
        """)
        
        processed_na_counts = df.isna().sum().reset_index()
        processed_na_counts.columns = ['Feature', 'Missing Count']
        processed_na_counts = processed_na_counts[processed_na_counts['Missing Count'] > 0]
        
        if not processed_na_counts.empty:
            fig_proc_na = px.bar(processed_na_counts, x='Feature', y='Missing Count', 
                                title="Remaining NAs (Processed Dataset)", color_discrete_sequence=['#10b981'])
            st.plotly_chart(fig_proc_na, use_container_width=True)
        else:
            st.info("🎯 **0 Missing Values!** The dataset is 100% complete and perfect for Neural Network ingestion.")
else:
    st.warning("Raw dataset file not found. Could not generate missing values report.")

st.markdown("---")

# --- Mathematical Correlation Matrix (Crucial for ANN) ---
st.header("3. Feature Correlation (Mathematical Analysis)")
st.markdown("This heatmap determines which numerical features have the highest impact on **Price**, allowing optimal ANN feature selection.")

# Select only numerical columns for the heatmap
numerical_df = df.select_dtypes(include=['float64', 'int64', 'int32'])
if not numerical_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation_matrix = numerical_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    st.pyplot(fig)
else:
    st.error("No numerical columns found for correlation.")

st.markdown("---")

# --- Raw Data Inspector ---
st.header("3. Processed Data Inspector")
st.markdown("Explore the raw matrix before it is passed to the Neural Network.")
st.dataframe(df.head(100), use_container_width=True)
