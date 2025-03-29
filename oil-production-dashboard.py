import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Page configuration
st.set_page_config(page_title="Oil Production Dashboard", layout="wide", page_icon="🛢️")

# Update the init_db function to return connection
def init_db(df=None):
    """
    Initialize database with default or provided schema
    
    Parameters:
    df (pandas.DataFrame, optional): DataFrame to use as schema template
    
    Returns:
    sqlite3.Connection: Database connection
    """
    conn = sqlite3.connect('production_data.db')
    c = conn.cursor()
    
    # Check if table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='production_data'")
    table_exists = c.fetchone() is not None
    
    if not table_exists:
        # Default schema if no DataFrame is provided
        default_columns = [
            '"Date" TEXT', '"Puits" TEXT', '"Type Puits" TEXT', '"Périmètre" TEXT', 
            '"Réservoir" TEXT', '"Manif" TEXT', '"Unité" TEXT', '"mode de calcul" TEXT', 
            '"Status" TEXT', '"Duse (mm)" REAL', '"Pt (bar)" REAL', '"Pp (bar)" REAL', 
            '"Heures de marche" REAL', '"P_Amont GL (bar)" REAL', 
            '"P_Aval GL (bar)" REAL', '"Heures de marche GL" REAL', 
            '"HW GL (inH20)" REAL', '"Q GL Calc (Sm³/j)" REAL', 
            '"Q GL Corr (Sm³/j)" REAL', '"Q Huile Calc (Sm³/j)" REAL', 
            '"Q Huile Corr (Sm³/j)" REAL', '"Q Gaz Form Calc (Sm³/j)" REAL', 
            '"Q Gaz Tot Calc (Sm³/j)" REAL', '"Q Gaz Form Corr (Sm³/j)" REAL', 
            '"Q Gaz Tot Corr (Sm³/j)" REAL', '"Q Eau Form Calc (m³/j)" REAL', 
            '"Q Eau Tot Calc (m³/j)" REAL', '"Q Eau Form Corr (m³/j)" REAL', 
            '"Pompage dans Tubing (m³/j)" REAL', 
            '"pompage dans Collecte (m³/j)" REAL', 
            '"Eau de Dessalage (m³/j)" REAL', 
            '"Q Eau inj (m³/j)" REAL', '"Q Eau Tot Corr (m³/j)" REAL', 
            '"MAP (Sm³/j)" REAL', '"Date Fermeture" TEXT', 
            '"Observations" TEXT', '"Date Dernier Test" TEXT', 
            '"Coef K" REAL', '"Duse Test (mm)" REAL', 
            '"Pt Test (bar)" REAL', '"Pp Test (bar)" REAL', 
            '"Q Huile Test (Sm³/h)" REAL', '"Q Gaz Tot Test (Sm³/h)" REAL', 
            '"Q GL Test (Sm³/h)" REAL', '"Q Eau Tot Test (m³/h)" REAL', 
            '"Q Eau inj Test (m³/h)" REAL', '"GOR Form Test" REAL', 
            '"GOR Tot Test" REAL', '"WOR Form Test" REAL', 
            '"WOR Tot Test" REAL', '"GLR Form Test" REAL', 
            '"GLR Tot Test" REAL', '"Densité Test" REAL'
        ]
        
        # If a DataFrame is provided, override with its schema
        if df is not None:
            # Create column definitions with proper quoting
            column_defs = []
            for col in df.columns:
                # Determine SQLite data type based on column dtype
                if pd.api.types.is_string_dtype(df[col]):
                    col_type = 'TEXT'
                elif pd.api.types.is_numeric_dtype(df[col]):
                    col_type = 'REAL'
                else:
                    col_type = 'TEXT'
                
                # Always quote column names to handle special characters
                quoted_col = f'"{col}"'
                column_defs.append(f"{quoted_col} {col_type}")
        else:
            # Use default columns
            column_defs = default_columns
        
        # Create table with columns
        create_table_sql = f"""
        CREATE TABLE production_data (
            {', '.join(column_defs)},
            UNIQUE(Date, Puits)  -- Add unique constraint on date and well combination
        )
        """
        
        try:
            c.execute(create_table_sql)
            conn.commit()
        except sqlite3.OperationalError as e:
            st.error(f"Error creating table: {e}")
    
    return conn

# Update the load_from_db function to ensure numeric conversion
def load_from_db():
    """
    Load data from SQLite database with robust type conversion
    
    Returns:
    pandas.DataFrame: Loaded database contents (empty DataFrame if no data exists)
    """
    conn = init_db()
    try:
        df = pd.read_sql_query('SELECT * FROM production_data', conn)
        
        # Convert date back to datetime if needed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # List of numeric columns that need conversion
        numeric_cols = [
            'Pt (bar)', 'Pp (bar)', 'Q Huile Corr (Sm³/j)', 
            'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)',
            'Q Gaz Form Corr (Sm³/j)', 'Q Eau Form Corr (m³/j)'
        ]
        
        # Convert numeric columns, handling various formats
        for col in numeric_cols:
            if col in df.columns:
                # First try direct numeric conversion
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # For columns that might have comma as decimal separator
                if df[col].isna().any():
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '.'), 
                        errors='coerce'
                    )
        
        return df
    except pd.errors.DatabaseError:
        return pd.DataFrame()  # Return empty DataFrame if table doesn't exist
    finally:
        conn.close()

# Add new function to manage saved data by date
def manage_saved_data():
    """
    Show interface for managing saved data with dropdown list
    """
    st.sidebar.header("📅 Manage Saved Data")
    
    # Load all data
    df = load_from_db()
    
    if df.empty:
        st.sidebar.info("No data in database yet")
        return
    
    # Get unique dates
    unique_dates = df['Date'].dt.date.unique()
    
    # Create dropdown for table selection
    date_options = sorted(unique_dates, reverse=True)
    date_strs = [date.strftime('%Y-%m-%d') for date in date_options]
    
    selected_date_str = st.sidebar.selectbox(
        "📋 Select Saved Table",
        options=date_strs,
        key="date_dropdown"
    )
    
    # Load selected table when dropdown value changes
    if selected_date_str:
        selected_date = pd.to_datetime(selected_date_str).date()
        selected_data = df[df['Date'].dt.date == selected_date]
        
        # Store in session state to use in main display
        st.session_state.selected_table = selected_data
        st.session_state.selected_date = selected_date_str
    
    # Show delete interface
    st.sidebar.subheader("Delete Data")
    selected_dates = st.sidebar.multiselect(
        "Select dates to remove",
        sorted(unique_dates, reverse=True),
        format_func=lambda x: x.strftime('%Y-%m-%d')
    )
    
    if selected_dates:
        # Add delete button
        if st.sidebar.button("❌ Delete Selected Dates", type="primary"):
            conn = init_db()
            c = conn.cursor()
            
            # Convert dates to strings in the database format
            date_strs = [d.strftime('%Y-%m-%d') for d in selected_dates]
            
            # Delete records
            placeholders = ','.join(['?'] * len(date_strs))
            c.execute(f"DELETE FROM production_data WHERE date(Date) IN ({placeholders})", date_strs)
            
            deleted_rows = conn.total_changes
            conn.commit()
            conn.close()
            
            st.sidebar.success(f"Deleted {deleted_rows} records from selected dates!")
            st.rerun()
    
    # In the main display area, show selected table if one was clicked
    if hasattr(st.session_state, 'selected_table'):
        st.subheader(f"Data for {st.session_state.selected_date}")
        st.dataframe(st.session_state.selected_table)
        
        # Show summary stats
        st.subheader("Summary Statistics")
        st.write(st.session_state.selected_table.describe())
    
    # Add a button to clear selection
    if hasattr(st.session_state, 'selected_table'):
        if st.sidebar.button("❌ Clear Selection"):
            del st.session_state.selected_table
            del st.session_state.selected_date
            st.rerun()
# Update the reset_database function to be more comprehensive
def reset_database():
    """
    Completely reset the database (use with caution)
    """
    conn = sqlite3.connect('production_data.db')
    c = conn.cursor()
    
    # Get list of all tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = c.fetchall()
    
    # Drop all tables
    for table in tables:
        c.execute(f"DROP TABLE IF EXISTS {table[0]}")
    
    conn.commit()
    conn.close()
    st.success("Database has been completely reset!")

def save_to_db(df):
    """
    Save DataFrame to SQLite database with duplicate checking
    
    Parameters:
    df (pandas.DataFrame): DataFrame to save
    """
    # Ensure date is parsed correctly
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    
    # Save to database with duplicate checking
    conn = sqlite3.connect('production_data.db')
    
    # Get existing dates and wells to check for duplicates
    existing_data = pd.read_sql_query('SELECT Date, Puits FROM production_data', conn)
    
    # Convert new data to same format for comparison
    new_data = df[['Date', 'Puits']].copy()
    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.strftime('%Y-%m-%d')
    
    # Find duplicates (existing records with same date and well)
    duplicates = pd.merge(existing_data, new_data, on=['Date', 'Puits'], how='inner')
    
    if not duplicates.empty:
        st.warning(f"⚠️ Found {len(duplicates)} duplicate records (same date and well). These will be skipped.")
        # Remove duplicates from new data before saving
        df = df[~df.set_index(['Date', 'Puits']).index.isin(duplicates.set_index(['Date', 'Puits']).index)]
    
    # Use pandas to_sql with quoted column names to handle special characters
    if not df.empty:
        df.to_sql('production_data', conn, if_exists='append', index=False)
        st.success(f"✅ Successfully saved {len(df)} new records to database!")
    else:
        st.info("No new records to save after duplicate removal.")
    
    conn.close()

# Update the DataFrame parsing to convert date column
def parse_excel(uploaded_file):
    df = pd.read_excel(uploaded_file, parse_dates=['Date'])
    
    # Ensure consistent date format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    return df

# Initialize database
init_db()

# Anomaly detection model setup
def train_anomaly_model(df):
    # Use WHP (Pt) and flowline pressure (Pp) for anomaly detection
    # Ensure all columns are numeric by converting to float
    df = df.copy()
    numeric_cols = ['Pt (bar)', 'Pp (bar)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    X = df[numeric_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest model
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)
    
    # Save model and scaler
    joblib.dump(model, 'anomaly_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    return model, scaler

def detect_anomalies(df, model, scaler):
    df = df.copy()
    # Ensure all columns are numeric by converting to float
    numeric_cols = ['Pt (bar)', 'Pp (bar)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    X = df[numeric_cols].dropna()
    if len(X) == 0:
        return pd.Series([False]*len(df), index=df.index)
    
    X_scaled = scaler.transform(X)
    anomalies = model.predict(X_scaled)
    anomaly_series = pd.Series([False]*len(df), index=df.index)
    anomaly_series.loc[X.index] = (anomalies == -1)
    return anomaly_series

# Enhanced visualization functions
def plot_production_comparison(df):
    df = df.copy()
    # Ensure all columns are numeric by converting to float
    numeric_cols = ['Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Oil Production", "Gas Production", "Water Production"))
    
    # Oil production by well
    oil_wells = df[df['Q Huile Corr (Sm³/j)'] > 0].sort_values('Q Huile Corr (Sm³/j)', ascending=False)
    fig.add_trace(
        go.Bar(x=oil_wells['Puits'], y=oil_wells['Q Huile Corr (Sm³/j)'], name="Oil"),
        row=1, col=1
    )
    
    # Gas production by well
    gas_wells = df[df['Q Gaz Tot Corr (Sm³/j)'] > 0].sort_values('Q Gaz Tot Corr (Sm³/j)', ascending=False)
    fig.add_trace(
        go.Bar(x=gas_wells['Puits'], y=gas_wells['Q Gaz Tot Corr (Sm³/j)'], name="Gas"),
        row=1, col=2
    )
    
    # Water production by well
    water_wells = df[df['Q Eau Tot Corr (m³/j)'] > 0].sort_values('Q Eau Tot Corr (m³/j)', ascending=False)
    fig.add_trace(
        go.Bar(x=water_wells['Puits'], y=water_wells['Q Eau Tot Corr (m³/j)'], name="Water"),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="Production by Well")
    st.plotly_chart(fig, use_container_width=True)

def plot_reservoir_performance(df):
    df = df.copy()
    # Ensure all columns are numeric by converting to float
    numeric_cols = ['Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)', 'Pp (bar)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    res_stats = df.groupby('Réservoir').agg({
        'Q Huile Corr (Sm³/j)': 'sum',
        'Q Gaz Tot Corr (Sm³/j)': 'sum',
        'Q Eau Tot Corr (m³/j)': 'sum',
        'Pp (bar)': 'mean'
    }).reset_index()
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'pie'}]])
    
    # Bar chart for production by reservoir
    fig.add_trace(
        go.Bar(x=res_stats['Réservoir'], y=res_stats['Q Huile Corr (Sm³/j)'], name="Oil"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=res_stats['Réservoir'], y=res_stats['Q Gaz Tot Corr (Sm³/j)'], name="Gas"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=res_stats['Réservoir'], y=res_stats['Q Eau Tot Corr (m³/j)'], name="Water"),
        row=1, col=1
    )
    
    # Pie chart for reservoir contribution
    fig.add_trace(
        go.Pie(labels=res_stats['Réservoir'], values=res_stats['Q Huile Corr (Sm³/j)'], name="Oil Contribution"),
        row=1, col=2
    )
    
    fig.update_layout(height=400, barmode='group', title_text="Reservoir Performance")
    st.plotly_chart(fig, use_container_width=True)

def plot_pressure_analysis(df):
    df = df.copy()
    # Ensure columns are numeric
    numeric_cols = ['Pt (bar)', 'Pp (bar)', 'Q Huile Corr (Sm³/j)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    fig = px.scatter(df, x='Pt (bar)', y='Pp (bar)', color='Réservoir',
                     hover_data=['Puits', 'Q Huile Corr (Sm³/j)'],
                     title="WHP vs Flowline Pressure Analysis")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_trends(historical_df, selected_well):
    historical_df = historical_df.copy()
    # Ensure columns are numeric
    numeric_cols = ['Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)', 'Pt (bar)', 'Pp (bar)']
    for col in numeric_cols:
        historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce')
        
    well_data = historical_df[historical_df['Puits'] == selected_well]
    if well_data.empty:
        return
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    
    # Production trends
    fig.add_trace(
        go.Scatter(x=well_data['Date'], y=well_data['Q Huile Corr (Sm³/j)'], name="Oil"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=well_data['Date'], y=well_data['Q Gaz Tot Corr (Sm³/j)'], name="Gas"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=well_data['Date'], y=well_data['Q Eau Tot Corr (m³/j)'], name="Water"),
        row=1, col=1
    )
    
    # Pressure trends
    fig.add_trace(
        go.Scatter(x=well_data['Date'], y=well_data['Pt (bar)'], name="WHP"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=well_data['Date'], y=well_data['Pp (bar)'], name="Flowline Pressure"),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title_text=f"Historical Trends for {selected_well}")
    st.plotly_chart(fig, use_container_width=True)

def plot_rate_analysis(historical_df, selected_well):
    historical_df = historical_df.copy()
    # Ensure columns are numeric
    numeric_cols = ['Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)', 
                   'Q Huile Test (Sm³/h)', 'Q Gaz Tot Test (Sm³/h)', 'Q Eau Tot Test (m³/h)']
    for col in numeric_cols:
        historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce')
        
    well_data = historical_df[historical_df['Puits'] == selected_well]
    if well_data.empty:
        return
    
    # Convert test date column to datetime if it's not already
    well_data['Date Dernier Test'] = pd.to_datetime(well_data['Date Dernier Test'], errors='coerce')
    
    # Get the most recent test data (excluding NaT values)
    test_data = well_data[well_data['Date Dernier Test'].notna()].sort_values('Date Dernier Test', ascending=False)
    
    if test_data.empty:
        st.warning(f"No test data available for well {selected_well}")
        return
    
    latest_test = test_data.iloc[0]
    
    # Prepare comparison data
    comparison_data = well_data[['Date', 'Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)']].copy()
    comparison_data['Type'] = 'Daily Production'
    
    # Convert test rates from hourly to daily (assuming test rates are in Sm³/h)
    test_rates = pd.DataFrame({
        'Date': [latest_test['Date Dernier Test']],
        'Q Huile Corr (Sm³/j)': [latest_test['Q Huile Test (Sm³/h)'] * 24 if pd.notna(latest_test['Q Huile Test (Sm³/h)']) else 0],
        'Q Gaz Tot Corr (Sm³/j)': [latest_test['Q Gaz Tot Test (Sm³/h)'] * 24 if pd.notna(latest_test['Q Gaz Tot Test (Sm³/h)']) else 0],
        'Q Eau Tot Corr (m³/j)': [latest_test['Q Eau Tot Test (m³/h)'] * 24 if pd.notna(latest_test['Q Eau Tot Test (m³/h)']) else 0],
        'Type': ['Test Data']
    })
    
    # Combine data for plotting
    plot_data = pd.concat([comparison_data, test_rates])
    
    # Create figure
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Oil Rate", "Gas Rate", "Water Rate"))
    
    # Oil rate comparison
    fig.add_trace(
        go.Bar(x=plot_data['Type'], y=plot_data['Q Huile Corr (Sm³/j)'], name="Oil", marker_color=['blue', 'red']),
        row=1, col=1
    )
    
    # Gas rate comparison
    fig.add_trace(
        go.Bar(x=plot_data['Type'], y=plot_data['Q Gaz Tot Corr (Sm³/j)'], name="Gas", marker_color=['blue', 'red']),
        row=1, col=2
    )
    
    # Water rate comparison
    fig.add_trace(
        go.Bar(x=plot_data['Type'], y=plot_data['Q Eau Tot Corr (m³/j)'], name="Water", marker_color=['blue', 'red']),
        row=1, col=3
    )
    
    # Calculate percentage differences (only if we have both values)
    oil_diff = 0
    gas_diff = 0
    water_diff = 0
    
    if len(plot_data) == 2:
        if plot_data.iloc[0]['Q Huile Corr (Sm³/j)'] != 0:
            oil_diff = ((plot_data.iloc[1]['Q Huile Corr (Sm³/j)'] - plot_data.iloc[0]['Q Huile Corr (Sm³/j)']) / 
                        plot_data.iloc[0]['Q Huile Corr (Sm³/j)']) * 100
        if plot_data.iloc[0]['Q Gaz Tot Corr (Sm³/j)'] != 0:
            gas_diff = ((plot_data.iloc[1]['Q Gaz Tot Corr (Sm³/j)'] - plot_data.iloc[0]['Q Gaz Tot Corr (Sm³/j)']) / 
                        plot_data.iloc[0]['Q Gaz Tot Corr (Sm³/j)']) * 100
        if plot_data.iloc[0]['Q Eau Tot Corr (m³/j)'] != 0:
            water_diff = ((plot_data.iloc[1]['Q Eau Tot Corr (m³/j)'] - plot_data.iloc[0]['Q Eau Tot Corr (m³/j)']) / 
                          plot_data.iloc[0]['Q Eau Tot Corr (m³/j)']) * 100
    
    # Add annotations
    fig.update_layout(
        height=400,
        title_text=f"Rate Analysis for {selected_well} (Test Date: {latest_test['Date Dernier Test'].strftime('%Y-%m-%d') if pd.notna(latest_test['Date Dernier Test']) else 'N/A'})",
        annotations=[
            dict(
                x=0.15, y=0.9,
                xref="paper", yref="paper",
                text=f"Diff: {oil_diff:.1f}%",
                showarrow=False
            ),
            dict(
                x=0.5, y=0.9,
                xref="paper", yref="paper",
                text=f"Diff: {gas_diff:.1f}%",
                showarrow=False
            ),
            dict(
                x=0.85, y=0.9,
                xref="paper", yref="paper",
                text=f"Diff: {water_diff:.1f}%",
                showarrow=False
            )
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show test details
    with st.expander("View Test Details"):
        test_details = latest_test[['Date Dernier Test', 'Duse Test (mm)', 'Pt Test (bar)', 'Pp Test (bar)',
                                  'Q Huile Test (Sm³/h)', 'Q Gaz Tot Test (Sm³/h)', 'Q Eau Tot Test (m³/h)',
                                  'GOR Tot Test', 'WOR Tot Test']]
        st.dataframe(test_details)

# Streamlit app
def main():
    # Initialize database
    conn = init_db()
    conn.close()
    # Clear table selection if new file is uploaded
    if 'selected_table' in st.session_state and st.sidebar.file_uploader("Upload Production Data (Excel)", type=["xlsx", "xls"]):
        del st.session_state.selected_table
        del st.session_state.selected_date
    
    st.title("🛢️ Oil Production Dashboard GTL Region")
    
    # Database management in sidebar
    st.sidebar.header("Database Management")
    
    # Add the new data management feature
    manage_saved_data()
    
    # Keep the reset button but make it more prominent
    if st.sidebar.button("⚠️ Reset Entire Database (Caution!)", type="secondary"):
        reset_database()
        st.rerun()
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Production Data (Excel)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
            
            # Save to database with correct column names
            save_to_db(df)
            
            # Load all historical data
            historical_df = load_from_db()
            
            # Convert date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            historical_df['Date'] = pd.to_datetime(historical_df['Date'])
            
            # Ensure all numeric columns are properly converted
            # Convert numeric columns
            numeric_cols = ['Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)', 
                           'Pt (bar)', 'Pp (bar)']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                if col in historical_df.columns:
                    historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce')
            
            # Display raw data
            with st.expander("View Raw Data"):
                st.dataframe(df)
            
            # Production Analysis
            st.header("📊 Production Analysis")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Oil Production")
                oil_wells = df[df['Q Huile Corr (Sm³/j)'] > 0]
                st.metric("Total Oil Production", f"{oil_wells['Q Huile Corr (Sm³/j)'].sum():,.0f} Sm³/d")
                st.metric("Number of Producing Wells", len(oil_wells))
                st.metric("Average Oil Rate", f"{oil_wells['Q Huile Corr (Sm³/j)'].mean():,.0f} Sm³/d")
                
            with col2:
                st.subheader("Gas Production")
                gas_wells = df[df['Q Gaz Tot Corr (Sm³/j)'] > 0]
                st.metric("Total Gas Production", f"{gas_wells['Q Gaz Tot Corr (Sm³/j)'].sum():,.0f} Sm³/d")
                st.metric("Number of Gas Producing Wells", len(gas_wells))
                if not oil_wells.empty and oil_wells['Q Huile Corr (Sm³/j)'].sum() > 0:
                    st.metric("Average GOR", f"{gas_wells['Q Gaz Tot Corr (Sm³/j)'].sum() / oil_wells['Q Huile Corr (Sm³/j)'].sum():,.0f} Sm³/Sm³")
                
            with col3:
                st.subheader("Water Production")
                water_wells = df[df['Q Eau Tot Corr (m³/j)'] > 0]
                st.metric("Total Water Production", f"{water_wells['Q Eau Tot Corr (m³/j)'].sum():,.0f} m³/d")
                if not oil_wells.empty and not water_wells.empty:
                    total_fluid = oil_wells['Q Huile Corr (Sm³/j)'].sum() + water_wells['Q Eau Tot Corr (m³/j)'].sum()
                    if total_fluid > 0:
                        st.metric("Water Cut (%)", f"{water_wells['Q Eau Tot Corr (m³/j)'].sum() / total_fluid * 100:.1f}%")
                    if oil_wells['Q Huile Corr (Sm³/j)'].sum() > 0:
                        st.metric("Average WOR", f"{water_wells['Q Eau Tot Corr (m³/j)'].sum() / oil_wells['Q Huile Corr (Sm³/j)'].sum():.2f} m³/Sm³")
            
            # Production comparison visualization
            st.subheader("Production Comparison by Well")
            plot_production_comparison(df)
            
            # Reservoir performance
            st.subheader("Reservoir Performance Analysis")
            plot_reservoir_performance(df)
            
            # Well Classification (HP/LP)
            st.header("🔧 Well Classification (HP/LP)")
            
            # Define thresholds (can be adjusted)
            hp_threshold = st.slider("HP Well Threshold (Pp in bar)", 20, 50, 30)
            
            # Ensure Pp column is numeric
            df['Pp (bar)'] = pd.to_numeric(df['Pp (bar)'], errors='coerce')
            df['Well Class'] = np.where(df['Pp (bar)'] >= hp_threshold, 'HP', 'LP')
            
            class_cols = st.columns(2)
            
            with class_cols[0]:
                st.subheader("Well Classification Summary")
                class_summary = df.groupby('Well Class').agg({
                    'Puits': 'count',
                    'Q Huile Corr (Sm³/j)': ['sum', 'mean'],
                    'Pp (bar)': 'mean'
                })
                st.dataframe(class_summary.style.background_gradient(cmap='Blues'))
                
            with class_cols[1]:
                st.subheader("Production by Well Class")
                fig = px.box(df, x='Well Class', y='Q Huile Corr (Sm³/j)', color='Well Class',
                             points="all", hover_data=['Puits'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Pressure Analysis
            st.header("📈 Pressure Analysis")
            plot_pressure_analysis(df)
            
            # Anomaly Detection
            st.header("⚠️ Anomaly Detection in WHP")
            
            # Train or load model
            if os.path.exists('anomaly_model.joblib') and os.path.exists('scaler.joblib'):
                model = joblib.load('anomaly_model.joblib')
                scaler = joblib.load('scaler.joblib')
            else:
                model, scaler = train_anomaly_model(historical_df)
            
            df['Anomaly'] = detect_anomalies(df, model, scaler)
            
            anomaly_cols = st.columns([2, 1])
            
            with anomaly_cols[0]:
                st.subheader("Anomaly Detection Results")
                anomaly_wells = df[df['Anomaly']]
                
                if not anomaly_wells.empty:
                    st.warning(f"⚠️ {len(anomaly_wells)} potential anomalies detected!")
                    
                    # Plot anomalies with Plotly
                    fig = px.scatter(df, x='Pt (bar)', y='Pp (bar)', color='Anomaly',
                                     hover_data=['Puits', 'Status', 'Q Huile Corr (Sm³/j)'],
                                     title="Anomaly Detection in WHP vs Flowline Pressure")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No anomalies detected in today's data.")
            
            with anomaly_cols[1]:
                if not anomaly_wells.empty:
                    st.subheader("Anomaly Details")
                    st.dataframe(anomaly_wells[['Puits', 'Pt (bar)', 'Pp (bar)', 'Status', 'Q Huile Corr (Sm³/j)']])
            
            # Historical Trends and Rate Analysis
            st.header("📅 Historical Analysis")
            
            trend_cols = st.columns(2)
            
            with trend_cols[0]:
                selected_well = st.selectbox("Select Well for Analysis", df['Puits'].unique())
                
            with trend_cols[1]:
                selected_reservoir = st.selectbox("Select Reservoir for Analysis", df['Réservoir'].unique())
                reservoir_wells = df[df['Réservoir'] == selected_reservoir]['Puits'].unique()
                st.selectbox("Select Well from Reservoir", reservoir_wells)
            
            # Tabs for different analysis types
            tab1, tab2 = st.tabs(["Historical Trends", "Rate Analysis"])
            
            with tab1:
                plot_historical_trends(historical_df, selected_well)
            
            with tab2:
                st.subheader(f"Rate Analysis for {selected_well}")
                plot_rate_analysis(historical_df, selected_well)
            
            # Additional reservoir analysis
            st.subheader(f"Reservoir {selected_reservoir} Analysis")
            
            res_cols = st.columns(2)
            
            with res_cols[0]:
                res_wells = historical_df[historical_df['Réservoir'] == selected_reservoir]
                fig = px.line(res_wells, x='Date', y='Q Huile Corr (Sm³/j)', color='Puits',
                              title=f"Oil Production Trend - {selected_reservoir}")
                st.plotly_chart(fig, use_container_width=True)
            
            with res_cols[1]:
                fig = px.line(res_wells, x='Date', y='Pp (bar)', color='Puits',
                              title=f"Flowline Pressure Trend - {selected_reservoir}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Data export
            st.sidebar.header("📤 Data Export")
            if st.sidebar.button("Export Current Analysis"):
                csv = df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='production_analysis.csv',
                    mime='text/csv'
                )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload an Excel file to begin analysis.")
        
        # Show historical data if available
        historical_df = load_from_db()
        if not historical_df.empty:
            st.subheader("Historical Data Summary")
        
        # Handle case where Date column might be missing or invalid
        try:
            date_range = f"from {historical_df['Date'].min().date()} to {historical_df['Date'].max().date()}"
        except (AttributeError, KeyError):
            date_range = "with unknown date range"
        
        st.write(f"Database contains {len(historical_df)} records {date_range}")
        
        # Show some summary stats
        summary_cols = st.columns(2)
        
        with summary_cols[0]:
            st.write("📅 Latest Production")
            
            try:
                latest_date = historical_df['Date'].max()
                latest_data = historical_df[historical_df['Date'] == latest_date]
                
                if not latest_data.empty:
                    st.write(f"Date: {latest_date.date()}")
                    
                    # Safely calculate sums with proper NaN handling
                    def safe_sum(series):
                        try:
                            return series.sum(skipna=True)
                        except (TypeError, ValueError):
                            return float('nan')
                    
                    oil_sum = safe_sum(latest_data.get('Q Huile Corr (Sm³/j)', pd.Series()))
                    gas_sum = safe_sum(latest_data.get('Q Gaz Tot Corr (Sm³/j)', pd.Series()))
                    water_sum = safe_sum(latest_data.get('Q Eau Tot Corr (m³/j)', pd.Series()))
                    
                    st.write(f"Oil: {oil_sum:,.0f} Sm³/d" if pd.notna(oil_sum) else "Oil: Data unavailable")
                    st.write(f"Gas: {gas_sum:,.0f} Sm³/d" if pd.notna(gas_sum) else "Gas: Data unavailable")
                    st.write(f"Water: {water_sum:,.0f} m³/d" if pd.notna(water_sum) else "Water: Data unavailable")
                else:
                    st.write("No data available for latest date")
            except Exception as e:
                st.error(f"Error displaying latest production: {str(e)}")
        
        with summary_cols[1]:
            st.write("🏆 Top Performing Wells (Historical)")
            
            try:
                if 'Q Huile Corr (Sm³/j)' in historical_df.columns:
                    # Filter out NaN values before calculating top wells
                    valid_wells = historical_df[historical_df['Q Huile Corr (Sm³/j)'].notna()]
                    if not valid_wells.empty:
                        top_historical = valid_wells.groupby('Puits')['Q Huile Corr (Sm³/j)'].mean().nlargest(5)
                        st.dataframe(top_historical)
                    else:
                        st.warning("No valid oil production data available")
                else:
                    st.warning("Oil production column not found in data")
            except Exception as e:
                st.error(f"Error calculating top wells: {str(e)}")
            
            st.write("📊 Production Trend (Last 30 Days)")
            try:
                if 'Date' in historical_df.columns and 'Q Huile Corr (Sm³/j)' in historical_df.columns:
                    recent_data = historical_df[
                        (historical_df['Date'] > (historical_df['Date'].max() - pd.Timedelta(days=30))) &
                        (historical_df['Q Huile Corr (Sm³/j)'].notna())
                    ]
                    
                    if not recent_data.empty:
                        trend_data = recent_data.groupby('Date')['Q Huile Corr (Sm³/j)'].sum().reset_index()
                        fig = px.line(trend_data, x='Date', y='Q Huile Corr (Sm³/j)', 
                                     title="Daily Oil Production Trend")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No recent data available for trend analysis")
                else:
                    st.warning("Required columns missing for trend analysis")
            except Exception as e:
                st.error(f"Error generating trend chart: {str(e)}")

if __name__ == "__main__":
    main()
