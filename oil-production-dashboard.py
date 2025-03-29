import streamlit as st
import pandas as pd
import shutil
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

# Enhanced function to manage saved data by date with improved dashboard preview
def manage_saved_data():
    """
    Show interface for managing saved data with dropdown list and dashboard preview
    
    Returns:
    pandas.DataFrame: Selected data (or None if no selection)
    """
    st.sidebar.header("📅 Manage Saved Data")
    
    # Load all data
    df = load_from_db()
    
    if df.empty:
        st.sidebar.info("No data in database yet")
        return None
    
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
    selected_data = None
    if selected_date_str:
        selected_date = pd.to_datetime(selected_date_str).date()
        selected_data = df[df['Date'].dt.date == selected_date].copy()
        
        # Store in session state to use in main display
        st.session_state.selected_table = selected_data
        st.session_state.selected_date = selected_date_str
        
        # Show a quick preview/summary of the selected data in the sidebar
        with st.sidebar.expander("📊 Selected Date Dashboard", expanded=True):
            # Quick metrics
            total_oil = selected_data['Q Huile Corr (Sm³/j)'].sum()
            total_gas = selected_data['Q Gaz Tot Corr (Sm³/j)'].sum()
            total_water = selected_data['Q Eau Tot Corr (m³/j)'].sum()
            well_count = selected_data['Puits'].nunique()
            
            # Display key metrics in sidebar
            st.metric("Date", selected_date_str)
            st.metric("Wells", f"{well_count}")
            st.metric("Oil Production", f"{total_oil:,.0f} Sm³/d")
            st.metric("Gas Production", f"{total_gas:,.0f} Sm³/d")
            st.metric("Water Production", f"{total_water:,.0f} m³/d")
            
            # Mini visualization - Top 5 oil producing wells
            try:
                top_wells = selected_data.sort_values('Q Huile Corr (Sm³/j)', ascending=False).head(5)
                fig = px.bar(
                    top_wells, 
                    x='Puits', 
                    y='Q Huile Corr (Sm³/j)',
                    title="Top 5 Oil Producing Wells",
                    height=200
                )
                fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
                # Reservoir distribution
                res_chart = selected_data.groupby('Réservoir')['Q Huile Corr (Sm³/j)'].sum().reset_index()
                if not res_chart.empty:
                    fig = px.pie(
                        res_chart, 
                        values='Q Huile Corr (Sm³/j)', 
                        names='Réservoir',
                        title="Oil Production by Reservoir",
                        height=200
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate preview charts: {e}")
            
            # Button to analyze this data in the main view
            if st.button("📈 View Full Analysis", key="view_full_analysis"):
                st.session_state.show_analysis = True
                # Will trigger the full analysis in the main area
    
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
    
    # Add a button to clear selection
    if hasattr(st.session_state, 'selected_table'):
        if st.sidebar.button("❌ Clear Selection"):
            del st.session_state.selected_table
            del st.session_state.selected_date
            if 'show_analysis' in st.session_state:
                del st.session_state.show_analysis
            st.rerun()
    
    return selected_data

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
        
def show_complete_analysis(df, historical_df):
    """
    Show comprehensive analysis of the production data
    
    Parameters:
    df (pandas.DataFrame): Current dataset to analyze
    historical_df (pandas.DataFrame): Historical data for comparisons
    """
    # Summary statistics
    st.subheader("📊 Production Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    # Ensure numeric columns are properly formatted
    df['Q Huile Corr (Sm³/j)'] = pd.to_numeric(df['Q Huile Corr (Sm³/j)'], errors='coerce')
    df['Q Gaz Tot Corr (Sm³/j)'] = pd.to_numeric(df['Q Gaz Tot Corr (Sm³/j)'], errors='coerce')
    df['Q Eau Tot Corr (m³/j)'] = pd.to_numeric(df['Q Eau Tot Corr (m³/j)'], errors='coerce')
    
    # Safely calculate sums
    total_oil = df['Q Huile Corr (Sm³/j)'].sum()
    total_gas = df['Q Gaz Tot Corr (Sm³/j)'].sum()
    total_water = df['Q Eau Tot Corr (m³/j)'].sum()
    active_wells = len(df[df['Status'] == 'Ouvert']) if 'Status' in df.columns else "N/A"
    
    col1.metric("Total Oil Production", f"{total_oil:,.0f} Sm³/d")
    col2.metric("Total Gas Production", f"{total_gas:,.0f} Sm³/d")
    col3.metric("Total Water Production", f"{total_water:,.0f} m³/d")
    col4.metric("Active Wells", active_wells)
    
    # Build or load anomaly detection model
    if os.path.exists('anomaly_model.joblib') and os.path.exists('scaler.joblib'):
        model = joblib.load('anomaly_model.joblib')
        scaler = joblib.load('scaler.joblib')
    else:
        if not historical_df.empty:
            model, scaler = train_anomaly_model(historical_df)
        else:
            model, scaler = train_anomaly_model(df)
    
    # Detect anomalies in current data
    df['is_anomaly'] = detect_anomalies(df, model, scaler)
    
    # Well-level anomaly reporting
    if df['is_anomaly'].any():
        st.subheader("⚠️ Well Anomalies Detected")
        anomaly_wells = df[df['is_anomaly']]
        st.write(f"{len(anomaly_wells)} wells have unusual pressure patterns")
        st.dataframe(anomaly_wells[['Puits', 'Réservoir', 'Pt (bar)', 'Pp (bar)', 'Q Huile Corr (Sm³/j)']])
    
    # Production comparison charts
    st.subheader("🛢️ Production by Well")
    plot_production_comparison(df)
    
    # Reservoir analysis
    st.subheader("📈 Reservoir Analysis")
    plot_reservoir_performance(df)
    
    # Pressure analysis
    st.subheader("🔍 Pressure Analysis")
    plot_pressure_analysis(df)
    
    # Well-level detailed analysis
    st.subheader("🔎 Well-Level Analysis")
    well_options = df['Puits'].unique()
    selected_well = st.selectbox("Select Well for Detailed Analysis:", well_options)
    
    # Historical trends for selected well
    st.write("### Historical Trends")
    plot_historical_trends(historical_df, selected_well)
    
    # Rate analysis (comparing to test rates)
    st.write("### Rate Analysis (vs Test Rates)")
    plot_rate_analysis(historical_df, selected_well)
    
    # Well data table
    st.write("### Well Data")
    well_data = df[df['Puits'] == selected_well]
    st.dataframe(well_data)

def upload_saved_tables():
    """
    Feature to upload saved tables from the database for analysis
    """
    st.sidebar.header("📂 Upload Saved Tables")
    
    # Load database to get available dates
    conn = sqlite3.connect('production_data.db')
    
    try:
        # Check if table exists before querying
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='production_data'")
        if c.fetchone() is None:
            st.sidebar.warning("No saved data found in database.")
            conn.close()
            return None
        
        # Query for unique dates
        dates_df = pd.read_sql_query("SELECT DISTINCT date(Date) as date_only FROM production_data ORDER BY date_only DESC", conn)
        
        if dates_df.empty:
            st.sidebar.warning("No saved data found in database.")
            conn.close()
            return None
        
        # Create date selection dropdown
        selected_date = st.sidebar.selectbox(
            "Select date to load:",
            options=dates_df['date_only'].tolist()
        )
        
        if selected_date:
            # Load data for selected date
            query = f"SELECT * FROM production_data WHERE date(Date) = '{selected_date}'"
            df = pd.read_sql_query(query, conn)
            
            # Convert date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Button to load selected data
            if st.sidebar.button("Load Selected Data"):
                st.session_state.selected_table = df
                st.session_state.selected_date = selected_date
                st.sidebar.success(f"Loaded data from {selected_date}")
                st.rerun()
                
            return df
        
    except Exception as e:
        st.sidebar.error(f"Error loading data: {str(e)}")
    finally:
        conn.close()
    
    return None

def export_database():
    """
    Export entire database as SQLite file
    """
    st.sidebar.header("💾 Database Export")
    
    if st.sidebar.button("Export Entire Database"):
        # Copy the database file to a temporary location
        import shutil
        import tempfile
        import os
        
        try:
            # Create temp file
            temp_dir = tempfile.mkdtemp()
            temp_db_path = os.path.join(temp_dir, 'production_data_export.db')
            
            # Copy the database
            shutil.copy2('production_data.db', temp_db_path)
            
            # Read the file as bytes
            with open(temp_db_path, 'rb') as f:
                db_bytes = f.read()
            
            # Offer download
            st.sidebar.download_button(
                label="Download Database File",
                data=db_bytes,
                file_name='production_data_export.db',
                mime='application/octet-stream'
            )
            
            st.sidebar.success("Database prepared for download!")
            
        except Exception as e:
            st.sidebar.error(f"Error exporting database: {str(e)}")
        finally:
            # Clean up temp files
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def import_database():
    """
    Import database from SQLite file
    """
    st.sidebar.header("📥 Database Import")
    
    uploaded_db = st.sidebar.file_uploader("Upload Database File", type=["db"])
    
    if uploaded_db is not None:
        try:
            # Save uploaded file temporarily
            import tempfile
            import os
            
            # Create temp file
            temp_dir = tempfile.mkdtemp()
            temp_db_path = os.path.join(temp_dir, 'uploaded_db.db')
            
            # Save uploaded file
            with open(temp_db_path, 'wb') as f:
                f.write(uploaded_db.getvalue())
            
            # Verify it's a valid SQLite database
            try:
                conn = sqlite3.connect(temp_db_path)
                c = conn.cursor()
                c.execute("PRAGMA integrity_check")
                result = c.fetchone()[0]
                conn.close()
                
                if result != 'ok':
                    st.sidebar.error("Invalid SQLite database file")
                    return
                
                # Check if it has the expected schema
                conn = sqlite3.connect(temp_db_path)
                c = conn.cursor()
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='production_data'")
                if c.fetchone() is None:
                    st.sidebar.error("Database does not contain production_data table")
                    conn.close()
                    return
                
                # Offer merge or replace option
                import_option = st.sidebar.radio(
                    "Import Option:",
                    options=["Merge with existing data", "Replace existing database"]
                )
                
                if st.sidebar.button("Confirm Import"):
                    if import_option == "Replace existing database":
                        # Make backup of existing db
                        if os.path.exists('production_data.db'):
                            backup_path = 'production_data_backup.db'
                            shutil.copy2('production_data.db', backup_path)
                            st.sidebar.info(f"Existing database backed up to {backup_path}")
                        
                        # Replace the database
                        shutil.copy2(temp_db_path, 'production_data.db')
                        st.sidebar.success("Database replaced successfully!")
                        st.rerun()
                    else:
                        # Merge data
                        src_conn = sqlite3.connect(temp_db_path)
                        src_df = pd.read_sql_query("SELECT * FROM production_data", src_conn)
                        src_conn.close()
                        
                        # Save to current database with duplicate checking
                        save_to_db(src_df)
                        st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Error verifying database: {str(e)}")
            finally:
                # Clean up
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                
        except Exception as e:
            st.sidebar.error(f"Error processing uploaded database: {str(e)}")

def main():
    # Initialize database
    conn = init_db()
    conn.close()
    
    # Clear table selection if new file is uploaded
    if 'selected_table' in st.session_state and st.sidebar.file_uploader("Upload Production Data (Excel)", type=["xlsx", "xls"]):
        del st.session_state.selected_table
        del st.session_state.selected_date
    
    st.title("🛢️ Enhanced Oil Production Dashboard")
    
    # Database management in sidebar
    st.sidebar.header("Database Management")
    
    # Add database upload/download features
    upload_saved_tables()
    export_database()
    import_database()
    
    # Add the data management feature
    manage_saved_data()
    
    # Keep the reset button but make it more prominent
    if st.sidebar.button("⚠️ Reset Entire Database (Caution!)", type="secondary"):
        reset_database()
        st.rerun()
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Production Data (Excel)", type=["xlsx", "xls"])
    
    # Logic for handling previously selected data from session state
    if hasattr(st.session_state, 'selected_table') and not uploaded_file:
        df = st.session_state.selected_table
        st.info(f"Currently viewing saved data from: {st.session_state.selected_date}")
        
        # Load all historical data for comparative analysis
        historical_df = load_from_db()
        
        # Show complete analysis for the selected data
        show_complete_analysis(df, historical_df)
        
        # Data export option for selected data
        st.sidebar.header("📤 Data Export")
        if st.sidebar.button("Export Current Analysis"):
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f'production_analysis_{st.session_state.selected_date}.csv',
                mime='text/csv'
            )
    
    elif uploaded_file is not None:
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
            
            # Show complete analysis for the uploaded data
            show_complete_analysis(df, historical_df)
            
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
        st.info("Please upload an Excel file or select saved data to begin analysis.")
        
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

# Run the application
if __name__ == "__main__":
    main()
