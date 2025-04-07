import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import shutil
import tempfile
import os
import time
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Page configuration
st.set_page_config(
    page_title="Oil Production Analytics Dashboard",
    layout="wide",
    page_icon="🛢️",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {font-size: 1.2rem;}
    h1 {color: #1E3A8A;}
    h2 {color: #1E4899;}
    h3 {color: #2563EB;}
    .stAlert {border-radius: 0.5rem;}
    .metric-card {background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.12);}
    .css-1aumxhk {background-color: #f0f2f6;}
</style>
""", unsafe_allow_html=True)

# Initialize database with enhanced connection handling
def init_db(df=None):
    """
    Initialize database with optimized schema and connection handling
    
    Parameters:
    df (pandas.DataFrame, optional): DataFrame to use as schema template
    
    Returns:
    sqlite3.Connection: Database connection
    """
    conn = sqlite3.connect('production_data.db')
    c = conn.cursor()
    
    # Enable foreign keys support
    c.execute("PRAGMA foreign_keys = ON")
    
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
            column_defs = []
            for col in df.columns:
                if pd.api.types.is_string_dtype(df[col]):
                    col_type = 'TEXT'
                elif pd.api.types.is_numeric_dtype(df[col]):
                    col_type = 'REAL'
                else:
                    col_type = 'TEXT'
                quoted_col = f'"{col}"'
                column_defs.append(f"{quoted_col} {col_type}")
        else:
            column_defs = default_columns
        
        # Create table with columns
        create_table_sql = f"""
        CREATE TABLE production_data (
            {', '.join(column_defs)},
            UNIQUE(Date, Puits)
        )
        """
        
        try:
            c.execute(create_table_sql)
            # Create indexes separately
            c.execute("CREATE INDEX IF NOT EXISTS idx_date_well ON production_data(Date, Puits)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_well ON production_data(Puits)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_reservoir ON production_data(Réservoir)")
            conn.commit()
        except sqlite3.OperationalError as e:
            st.error(f"Error creating table: {e}")
            conn.rollback()
    
    return conn

# Improved data loading function with caching for better performance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_from_db(date_filter=None):
    """
    Load data from SQLite database with robust type conversion and optional date filtering
    
    Parameters:
    date_filter (str, optional): ISO format date string to filter by (YYYY-MM-DD)
    
    Returns:
    pandas.DataFrame: Loaded database contents (empty DataFrame if no data exists)
    """
    conn = init_db()
    try:
        # Build query with optional date filter
        query = 'SELECT * FROM production_data'
        params = []
        
        if date_filter:
            query += ' WHERE date(Date) = ?'
            params.append(date_filter)
            
        df = pd.read_sql_query(query, conn, params=params)
        
        # Convert date back to datetime if needed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # List of all columns that should be numeric
        numeric_cols = [
            'Duse (mm)', 'Pt (bar)', 'Pp (bar)', 'Heures de marche',
            'P_Amont GL (bar)', 'P_Aval GL (bar)', 'Heures de marche GL',
            'HW GL (inH20)', 'Q GL Calc (Sm³/j)', 'Q GL Corr (Sm³/j)',
            'Q Huile Calc (Sm³/j)', 'Q Huile Corr (Sm³/j)', 
            'Q Gaz Form Calc (Sm³/j)', 'Q Gaz Tot Calc (Sm³/j)',
            'Q Gaz Form Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)',
            'Q Eau Form Calc (m³/j)', 'Q Eau Tot Calc (m³/j)',
            'Q Eau Form Corr (m³/j)', 'Q Eau Tot Corr (m³/j)',
            'Pompage dans Tubing (m³/j)', 'pompage dans Collecte (m³/j)',
            'Eau de Dessalage (m³/j)', 'Q Eau inj (m³/j)', 'MAP (Sm³/j)',
            'Coef K', 'Duse Test (mm)', 'Pt Test (bar)', 'Pp Test (bar)',
            'Q Huile Test (Sm³/h)', 'Q Gaz Tot Test (Sm³/h)', 'Q GL Test (Sm³/h)',
            'Q Eau Tot Test (m³/h)', 'Q Eau inj Test (m³/h)', 'GOR Form Test',
            'GOR Tot Test', 'WOR Form Test', 'WOR Tot Test', 'GLR Form Test',
            'GLR Tot Test', 'Densité Test'
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

def manage_saved_data():
    """
    Show interface for managing saved data with dropdown list and automatic loading
    """
    st.sidebar.header("📅 Database Management")
    
    # Get all available dates
    available_dates = get_available_dates()
    
    if not available_dates:
        st.sidebar.info("No data in database yet")
        return None
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Date List"):
        st.cache_data.clear()
        st.rerun()
    
    # Create dropdown for date selection
    selected_date = st.sidebar.selectbox(
        "📋 Select Production Date to Load",
        options=available_dates,
        key="date_dropdown"
    )
    
    # Export/Import section
    with st.sidebar.expander("🗄️ Database Transfer"):
        # Export functionality
        export_database()
        
        # Import functionality
        uploaded_db = st.file_uploader(
            "Upload database file", 
            type=['db', 'sqlite', 'sqlite3'],
            accept_multiple_files=False,
            key="db_uploader"
        )
        
        if uploaded_db is not None:
            import_database(uploaded_db)
    
    # Automatically load data for selected date
    if selected_date:
        df = load_from_db(selected_date)
        st.sidebar.success(f"✅ Loaded data for {selected_date}")
        return df
    
    # Show data deletion interface in expandable section
    with st.sidebar.expander("🗑️ Delete Data"):
        selected_dates = st.multiselect(
            "Select dates to remove",
            options=available_dates
        )
        
        if selected_dates:
            if st.button("❌ Delete Selected Dates", type="primary"):
                conn = init_db()
                c = conn.cursor()
                
                placeholders = ','.join(['?'] * len(selected_dates))
                c.execute(f"DELETE FROM production_data WHERE date(Date) IN ({placeholders})", selected_dates)
                
                deleted_rows = conn.total_changes
                conn.commit()
                conn.close()
                
                st.success(f"Deleted {deleted_rows} records from selected dates!")
                st.rerun()
    
    return None

def reset_database():
    """
    Completely reset the database (use with caution)
    """
    # Add confirmation to prevent accidental resets
    confirmation = st.sidebar.text_input("Type 'CONFIRM' to reset database:")
    
    if confirmation == "CONFIRM":
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
        
        # Clear all cached data
        st.cache_data.clear()
        
        # Clear the confirmation input
        st.session_state.pop("reset_confirmation", None)
        
        # Clear the date dropdown selection
        if 'date_dropdown' in st.session_state:
            st.session_state.pop('date_dropdown')
        
        st.sidebar.success("Database has been completely reset!")
        
        # Add small delay before rerun
        time.sleep(1)
        st.rerun()
    elif confirmation and confirmation != "CONFIRM":
        st.sidebar.error("Confirmation text does not match 'CONFIRM'")

def import_database(uploaded_file):
    """
    Import a database file to replace the current database
    
    Parameters:
    uploaded_file: Uploaded file object from Streamlit
    
    Returns:
    bool: True if import was successful, False otherwise
    """
    # Add confirmation to prevent accidental imports
    confirmation = st.checkbox("I confirm I want to replace the current database")
    
    if confirmation and uploaded_file is not None:
        try:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Verify the file is a valid SQLite database
            try:
                test_conn = sqlite3.connect(tmp_path)
                test_conn.cursor().execute("SELECT name FROM sqlite_master WHERE type='table'")
                test_conn.close()
            except sqlite3.DatabaseError:
                st.error("Uploaded file is not a valid SQLite database")
                return False
            
            # Backup current database (just in case)
            backup_path = 'production_data_backup.db'
            if os.path.exists('production_data.db'):
                shutil.copy2('production_data.db', backup_path)
            
            # Replace current database with uploaded one
            shutil.copy2(tmp_path, 'production_data.db')
            
            st.success("Database imported successfully! Backup saved as production_data_backup.db")
            
            # Clear cache to force reload
            st.cache_data.clear()
            
            # Clear the file uploader state
            st.session_state.pop("db_uploader", None)
            
            # Force immediate reload of dates
            get_available_dates(force_refresh=True)
            
            # Set active tab to Overview
            st.session_state['active_tab'] = 0
            
            # Trigger a rerun
            st.rerun()
            
            return True
            
        except Exception as e:
            st.error(f"Error importing database: {str(e)}")
            if os.path.exists(backup_path):
                st.info("Original database has been restored from backup")
                shutil.copy2(backup_path, 'production_data.db')
            return False
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return False

# Modified export_database function for direct download
def export_database():
    """
    Export the SQLite database to a file for backup or transfer
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
        tmp.close()
        tmp_path = tmp.name
    
    try:
        # Connect to source and destination databases
        src_conn = sqlite3.connect('production_data.db')
        dst_conn = sqlite3.connect(tmp_path)
        
        # Backup the database
        src_conn.backup(dst_conn)
        
        # Close connections
        src_conn.close()
        dst_conn.close()
        
        # Read the temporary file as bytes
        with open(tmp_path, 'rb') as f:
            db_bytes = f.read()
        
        # Create download button
        st.download_button(
            label="⬇️ Download Database File",
            data=db_bytes,
            file_name="production_data_export.db",
            mime="application/x-sqlite3",
            help="Download a complete copy of the current database"
        )
        return True
    except Exception as e:
        st.error(f"Error exporting database: {str(e)}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Modified save_to_db function with automatic refresh
def save_to_db(df):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Preparing data for saving...")

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

    conn = init_db(df)
    
    try:
        progress_bar.progress(25)
        status_text.text("Checking for duplicates...")
        
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='production_data'")
        if not c.fetchone():
            raise Exception("Production table could not be created")
        
        existing_data = pd.read_sql_query('SELECT Date, Puits FROM production_data', conn)
        
        progress_bar.progress(50)
        
        if not existing_data.empty:
            new_data = df[['Date', 'Puits']].copy()
            duplicates = pd.merge(existing_data, new_data, on=['Date', 'Puits'], how='inner')
            
            if not duplicates.empty:
                status_text.warning(f"⚠️ Found {len(duplicates)} duplicate records. These will be skipped.")
                df = df[~df.set_index(['Date', 'Puits']).index.isin(duplicates.set_index(['Date', 'Puits']).index)]
        
        progress_bar.progress(75)
        status_text.text("Saving to database...")
        
        if not df.empty:
            df.to_sql('production_data', conn, if_exists='append', index=False)
            conn.commit()
            progress_bar.progress(100)
            status_text.success(f"✅ Successfully saved {len(df)} new records!")
            
            # Set the flag to refresh dashboard
            st.session_state['refresh_dashboard'] = True
            
            # Explicitly clear cache
            st.cache_data.clear()
            
            return True
        else:
            progress_bar.progress(100)
            status_text.info("No new records to save after duplicate removal.")
            return False
            
    except Exception as e:
        status_text.error(f"Error saving to database: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

# Get all available dates in the database with force refresh option
@st.cache_data(ttl=300, show_spinner="Loading available dates...")
def get_available_dates(force_refresh=False):
    """
    Get all available dates in the database with force refresh option
    
    Parameters:
    force_refresh (bool): Flag to force cache refresh (doesn't affect logic, just invalidates cache)
    
    Returns:
    list: List of available dates in database
    """
    conn = init_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT date(Date) FROM production_data ORDER BY Date DESC")
        dates = [row[0] for row in cursor.fetchall()]
        return dates
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        conn.close()

# Add initialization of session state variables
if 'refresh_dashboard' not in st.session_state:
    st.session_state['refresh_dashboard'] = False

if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 0

# Enhanced Excel parsing function with error handling
def parse_excel(uploaded_file):
    try:
        # Create progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Reading Excel file...")
        
        # Read Excel file
        df = pd.read_excel(uploaded_file, parse_dates=['Date'])
        
        # Update progress
        progress_bar.progress(33)
        status_text.text("Processing dates...")
        
        # Ensure consistent date format
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Update progress
        progress_bar.progress(66)
        status_text.text("Validating data...")
        
        # Check for required columns
        required_cols = ['Date', 'Puits', 'Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            progress_bar.progress(100)
            status_text.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
        
        # Validate date format
        invalid_dates = df['Date'].isna().sum()
        if invalid_dates > 0:
            status_text.warning(f"Found {invalid_dates} rows with invalid dates. These may cause issues.")
            
        # Completion
        progress_bar.progress(100)
        status_text.success("Excel file processed successfully!")
        return df
    
    except Exception as e:
        # Replace the nested expander with a simple error display
        status_text.error(f"Error parsing Excel file: {str(e)}")
        st.error("Please check that your Excel file is properly formatted and not corrupt.")
        return None

# Enhanced pressure model training with better scaling and validation
def train_pressure_model(df):
    """
    Train K-means model on pressure data with improved validation and handling
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing pressure data
    
    Returns:
    tuple: (model, scaler) or (None, None) if training fails
    """
    if df.empty:
        return None, None
        
    df = df.copy()
    # Use WHP (Pt) and flowline pressure (Pp) for clustering
    numeric_cols = ['Pt (bar)', 'Pp (bar)']
    
    # Convert to numeric and drop NA
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    X = df[numeric_cols].dropna()
    
    if len(X) < 5:  # Need at least 5 points for reliable K-means
        return None, None
    
    # Scale data with robust scaler to handle outliers better
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters (between 2-5)
    inertia = []
    k_range = range(2, 6)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Find elbow point or use default of 3 clusters
    try:
        # Calculate first differences
        diffs = np.diff(inertia)
        # Find where the difference starts to flatten (elbow)
        elbow = np.argmax(np.diff(diffs)) + 2  # +2 because we start range at 2
        # Ensure we stay within our range
        k_optimal = min(max(elbow, 2), 5)
    except:
        # Default to 3 clusters if determination fails
        k_optimal = 3
    
    # Train K-means with optimal clusters (low, medium, high pressure)
    model = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    model.fit(X_scaled)
    
    # Save model and scaler
    joblib.dump(model, 'pressure_model.joblib')
    joblib.dump(scaler, 'pressure_scaler.joblib')
    
    return model, scaler

# Enhanced pressure classification function with better visualization
def classify_pressures(df, model, scaler):
    """
    Classify wells into pressure clusters with improved handling and visualization
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing well data
    model: Trained KMeans model
    scaler: Fitted StandardScaler
    
    Returns:
    pandas.Series: Cluster labels for each well
    """
    if df.empty or model is None or scaler is None:
        return pd.Series(dtype='int')
        
    df = df.copy()
    numeric_cols = ['Pt (bar)', 'Pp (bar)']
    
    # Convert to numeric
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    X = df[numeric_cols].dropna()
    
    if len(X) == 0:
        return pd.Series([-1]*len(df), index=df.index)
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    clusters = model.predict(X_scaled)
    
    # Create series with -1 for NA values
    cluster_series = pd.Series([-1]*len(df), index=df.index)
    cluster_series.loc[X.index] = clusters
    
    # Determine cluster meaning based on pressure centers
    centers = scaler.inverse_transform(model.cluster_centers_)
    avg_pressures = centers[:, 0] + centers[:, 1]  # Sum of Pt and Pp
    
    # Sort clusters by average pressure: low, medium, high
    pressure_rank = np.argsort(avg_pressures)
    
    # Map the numerical clusters to their pressure rank
    rank_map = {orig: rank for rank, orig in enumerate(pressure_rank)}
    cluster_series = cluster_series.map(lambda x: rank_map.get(x, -1) if x != -1 else -1)
    
    return cluster_series

# New function to train production prediction models
def train_prediction_models(df, target_col='Q Huile Corr (Sm³/j)', days_ahead=7):
    """
    Train machine learning models to predict future production
    
    Parameters:
    df (pandas.DataFrame): Historical production data
    target_col (str): Column to predict (oil, gas, water)
    days_ahead (int): Number of days to predict ahead
    
    Returns:
    dict: Dictionary of trained models and their performance metrics
    """
    if df.empty or target_col not in df.columns:
        return None
    
    # Copy dataframe and ensure date column is datetime
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create features for prediction
    predictor_cols = [
        'Pt (bar)', 'Pp (bar)', 'Heures de marche',
        'Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)'
    ]
    
    # Ensure all columns are numeric
    for col in predictor_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Only use columns that exist in the dataframe
    available_predictors = [col for col in predictor_cols if col in df.columns]
    
    # Group by well and create time-lagged features
    model_results = {}
    
    for well in df['Puits'].unique():
        well_data = df[df['Puits'] == well].sort_values('Date')
        
        if len(well_data) < 30:  # Need enough data for reliable predictions
            continue
            
        # Create lag features (previous values)
        for col in available_predictors:
            for lag in range(1, 8):  # 1-7 day lags
                well_data[f'{col}_lag{lag}'] = well_data[col].shift(lag)
        
        # Create target (future production)
        well_data[f'{target_col}_future{days_ahead}'] = well_data[target_col].shift(-days_ahead)
        
        # Drop rows with NaN values
        well_data = well_data.dropna()
        
        if len(well_data) < 20:  # Still need enough data after creating features
            continue
            
        # Prepare features and target
        feature_cols = [f'{col}_lag{lag}' for col in available_predictors for lag in range(1, 8)]
        X = well_data[feature_cols]
        y = well_data[f'{target_col}_future{days_ahead}']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        model_results[well] = {}
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_predictions = lr_model.predict(X_test)
        lr_metrics = {
            'model': lr_model,
            'mse': mean_squared_error(y_test, lr_predictions),
            'r2': r2_score(y_test, lr_predictions)
        }
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        rf_metrics = {
            'model': rf_model,
            'mse': mean_squared_error(y_test, rf_predictions),
            'r2': r2_score(y_test, rf_predictions)
        }
        
        # Store results
        model_results[well]['LinearRegression'] = lr_metrics
        model_results[well]['RandomForest'] = rf_metrics
        model_results[well]['feature_cols'] = feature_cols
        model_results[well]['last_data'] = well_data.iloc[-7:][available_predictors]
    
    # Save models
    joblib.dump(model_results, f'{target_col.replace(" ", "_")}_prediction_models.joblib')
    
    return model_results

# Function to make production predictions for a specific well
def predict_production(well, prediction_models, days=7, target_col='Q Huile Corr (Sm³/j)'):
    """
    Make production predictions for a specific well
    
    Parameters:
    well (str): Well name
    prediction_models (dict): Dictionary of trained models
    days (int): Number of days to predict
    target_col (str): Target column to predict
    
    Returns:
    dict: Predictions from different models
    """
    if well not in prediction_models:
        return None
    
    well_models = prediction_models[well]
    
    # Get the most recent data for the well
    last_data = well_models['last_data']
    
    # Create features for prediction
    features = {}
    for i in range(min(len(last_data), 7)):
        row = last_data.iloc[i]
        for col in row.index:
            for lag in range(1, 8):
                if i+1 == lag:
                    features[f'{col}_lag{lag}'] = row[col]
    
    # Create feature array in the correct order
    feature_cols = well_models['feature_cols']
    X_pred = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
    
    # Make predictions with each model
    predictions = {}
    
    for model_name in ['LinearRegression', 'RandomForest']:
        if model_name in well_models:
            model = well_models[model_name]['model']
            pred = model.predict(X_pred)[0]
            predictions[model_name] = max(0, pred)  # Ensure non-negative predictions
    
    return predictions

# New function for time series forecasting with ARIMA and Prophet
def forecast_production(historical_df, well, target_col='Q Huile Corr (Sm³/j)', days_ahead=30):
    """
    Generate time series forecasts for well production
    
    Parameters:
    historical_df (pandas.DataFrame): Historical production data
    well (str): Well name
    target_col (str): Column to forecast
    days_ahead (int): Number of days to forecast ahead
    
    Returns:
    dict: Dictionary with forecast results and visualization data
    """
    if historical_df.empty:
        return None
        
    # Filter data for the selected well
    well_data = historical_df[historical_df['Puits'] == well].copy()
    
    if len(well_data) < 30:  # Need enough data for reliable forecasting
        return {'error': 'Insufficient data for forecasting'}
    
    # Ensure columns are properly formatted
    well_data['Date'] = pd.to_datetime(well_data['Date'])
    well_data[target_col] = pd.to_numeric(well_data[target_col], errors='coerce')
    
    # Remove missing values
    well_data = well_data.dropna(subset=['Date', target_col])
    
    if len(well_data) < 3:  # Check again after removing NAs
        return {'error': 'Insufficient data for forecasting after cleaning'}
    
    # Sort by date
    well_data = well_data.sort_values('Date')
    
    # Create time series dataframe
    ts_data = well_data[['Date', target_col]].set_index('Date')
    
    # Initialize result dictionary
    forecast_results = {
        'well': well,
        'target': target_col,
        'historical': ts_data,
        'forecasts': {}
    }
    
    # ARIMA Forecast
    try:
        # Fit ARIMA model (simple configuration)
        arima_model = ARIMA(ts_data, order=(5,1,0))
        arima_results = arima_model.fit()
        
        # Generate forecast
        arima_forecast = arima_results.forecast(steps=days_ahead)
        
        # Store ARIMA results
        forecast_results['forecasts']['ARIMA'] = arima_forecast
    except Exception as e:
        forecast_results['forecasts']['ARIMA'] = {'error': str(e)}
    
    # Prophet Forecast
    try:
        # Prepare data for Prophet
        prophet_data = well_data[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
        
        # Fit Prophet model
        prophet_model = Prophet(daily_seasonality=False, yearly_seasonality=True)
        prophet_model.fit(prophet_data)
        
        # Create future dataframe
        future = prophet_model.make_future_dataframe(periods=days_ahead)
        
        # Generate forecast
        prophet_forecast = prophet_model.predict(future)
        
        # Store Prophet results
        forecast_results['forecasts']['Prophet'] = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-days_ahead:]
    except Exception as e:
        forecast_results['forecasts']['Prophet'] = {'error': str(e)}
    
    return forecast_results

# Function to create dashboard metrics calculation
def calculate_key_metrics(df):
    """
    Calculate key production metrics for the dashboard
    
    Parameters:
    df (pandas.DataFrame): Production data
    
    Returns:
    dict: Dictionary of calculated metrics
    """
    if df.empty:
        return {}
    
    metrics = {}
    
    # Convert columns to numeric to ensure proper calculations
    numeric_cols = [
        'Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 
        'Q Eau Tot Corr (m³/j)', 'Heures de marche'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Total production
    metrics['total_oil'] = df['Q Huile Corr (Sm³/j)'].sum()
    metrics['total_gas'] = df['Q Gaz Tot Corr (Sm³/j)'].sum()
    metrics['total_water'] = df['Q Eau Tot Corr (m³/j)'].sum()
    
    # Active wells
    metrics['active_wells'] = df[df['Q Huile Corr (Sm³/j)'] > 0]['Puits'].nunique()
    metrics['total_wells'] = df['Puits'].nunique()
    
    # Average production per well
    if metrics['active_wells'] > 0:
        metrics['avg_oil_per_well'] = metrics['total_oil'] / metrics['active_wells']
        metrics['avg_gas_per_well'] = metrics['total_gas'] / metrics['active_wells']
        metrics['avg_water_per_well'] = metrics['total_water'] / metrics['active_wells']
    else:
        metrics['avg_oil_per_well'] = 0
        metrics['avg_gas_per_well'] = 0
        metrics['avg_water_per_well'] = 0
    
    # Runtime hours
    if 'Heures de marche' in df.columns:
        metrics['avg_runtime'] = df['Heures de marche'].mean()
        metrics['total_runtime'] = df['Heures de marche'].sum()
    else:
        metrics['avg_runtime'] = 0
        metrics['total_runtime'] = 0
    
    # Water cut
    total_liquid = metrics['total_oil'] + metrics['total_water']
    metrics['water_cut'] = (metrics['total_water'] / total_liquid * 100) if total_liquid > 0 else 0
    
    # Gas-Oil Ratio (GOR)
    metrics['gor'] = (metrics['total_gas'] / metrics['total_oil']) if metrics['total_oil'] > 0 else 0
    
    return metrics

# Function to generate production distribution plots
def create_production_plots(df):
    """
    Create interactive production distribution plots
    
    Parameters:
    df (pandas.DataFrame): Production data
    
    Returns:
    dict: Dictionary of Plotly figures
    """
    if df.empty:
        return {}
    
    # Ensure numeric columns
    numeric_cols = [
        'Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 
        'Q Eau Tot Corr (m³/j)', 'Pt (bar)', 'Pp (bar)'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    plots = {}
    
    # Oil Production Distribution
    if 'Q Huile Corr (Sm³/j)' in df.columns:
        oil_data = df[df['Q Huile Corr (Sm³/j)'] > 0].copy()
        if not oil_data.empty:
            # Create histogram with density curve
            oil_hist = px.histogram(
                oil_data, 
                x='Q Huile Corr (Sm³/j)',
                nbins=20,
                title='Oil Production Distribution',
                color_discrete_sequence=['#1E88E5'],
                marginal='box',
                labels={'Q Huile Corr (Sm³/j)': 'Oil Production (Sm³/j)'}
            )
            oil_hist.update_layout(
                xaxis_title='Oil Production (Sm³/j)',
                yaxis_title='Number of Wells',
                bargap=0.1
            )
            plots['oil_distribution'] = oil_hist
    
    # Gas Production Distribution
    if 'Q Gaz Tot Corr (Sm³/j)' in df.columns:
        gas_data = df[df['Q Gaz Tot Corr (Sm³/j)'] > 0].copy()
        if not gas_data.empty:
            gas_hist = px.histogram(
                gas_data, 
                x='Q Gaz Tot Corr (Sm³/j)',
                nbins=20,
                title='Gas Production Distribution',
                color_discrete_sequence=['#43A047'],
                marginal='box',
                labels={'Q Gaz Tot Corr (Sm³/j)': 'Gas Production (Sm³/j)'}
            )
            gas_hist.update_layout(
                xaxis_title='Gas Production (Sm³/j)',
                yaxis_title='Number of Wells',
                bargap=0.1
            )
            plots['gas_distribution'] = gas_hist
    
    # Water Production Distribution
    if 'Q Eau Tot Corr (m³/j)' in df.columns:
        water_data = df[df['Q Eau Tot Corr (m³/j)'] > 0].copy()
        if not water_data.empty:
            water_hist = px.histogram(
                water_data, 
                x='Q Eau Tot Corr (m³/j)',
                nbins=20,
                title='Water Production Distribution',
                color_discrete_sequence=['#FFA726'],
                marginal='box',
                labels={'Q Eau Tot Corr (m³/j)': 'Water Production (m³/j)'}
            )
            water_hist.update_layout(
                xaxis_title='Water Production (m³/j)',
                yaxis_title='Number of Wells',
                bargap=0.1
            )
            plots['water_distribution'] = water_hist
    
    # Pressure crossplot
    if 'Pt (bar)' in df.columns and 'Pp (bar)' in df.columns:
        pressure_data = df[['Puits', 'Pt (bar)', 'Pp (bar)', 'Q Huile Corr (Sm³/j)']].dropna()
        if not pressure_data.empty:
            pressure_plot = px.scatter(
                pressure_data,
                x='Pt (bar)',
                y='Pp (bar)',
                size='Q Huile Corr (Sm³/j)',
                color='Q Huile Corr (Sm³/j)',
                hover_name='Puits',
                title='Wellhead vs Flowline Pressure',
                color_continuous_scale='Viridis',
                size_max=20
            )
            pressure_plot.update_layout(
                xaxis_title='Wellhead Pressure (bar)',
                yaxis_title='Flowline Pressure (bar)',
                coloraxis_colorbar_title='Oil Rate (Sm³/j)'
            )
            plots['pressure_crossplot'] = pressure_plot
    
    return plots

# Function to create well performance scatter plots
def create_performance_plots(df):
    """
    Generate interactive well performance visualization
    
    Parameters:
    df (pandas.DataFrame): Production data
    
    Returns:
    dict: Dictionary of Plotly figures
    """
    if df.empty:
        return {}
    
    # Ensure numeric columns
    numeric_cols = [
        'Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 
        'Q Eau Tot Corr (m³/j)', 'Heures de marche'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    plots = {}
    
    # Calculate top 15 wells by oil production
    if 'Q Huile Corr (Sm³/j)' in df.columns:
        top_oil_wells = df.groupby('Puits')['Q Huile Corr (Sm³/j)'].sum().sort_values(ascending=False).head(15)
        
        if not top_oil_wells.empty:
            # Create bar chart
            top_wells_data = top_oil_wells.reset_index()
            top_wells_plot = px.bar(
                top_wells_data,
                x='Puits',
                y='Q Huile Corr (Sm³/j)',
                title='Top 15 Wells by Oil Production',
                color='Q Huile Corr (Sm³/j)',
                color_continuous_scale='Blues',
                labels={'Q Huile Corr (Sm³/j)': 'Oil Production (Sm³/j)'}
            )
            top_wells_plot.update_layout(
                xaxis_title='Well',
                yaxis_title='Oil Production (Sm³/j)',
                xaxis={'categoryorder':'total descending'}
            )
            plots['top_wells'] = top_wells_plot
    
    # Gas-Oil Ratio by Well
    if all(col in df.columns for col in ['Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)']):
        # Calculate GOR for each well
        gor_data = df.copy()
        gor_data = gor_data[gor_data['Q Huile Corr (Sm³/j)'] > 0]  # Avoid division by zero
        gor_data['GOR'] = gor_data['Q Gaz Tot Corr (Sm³/j)'] / gor_data['Q Huile Corr (Sm³/j)']
        
        # Group by well and find average GOR
        well_gor = gor_data.groupby('Puits').agg({
            'GOR': 'mean',
            'Q Huile Corr (Sm³/j)': 'sum'
        }).reset_index()
        
        # Filter to top 15 oil producers
        top_wells = top_oil_wells.index.tolist()
        well_gor = well_gor[well_gor['Puits'].isin(top_wells)]
        
        if not well_gor.empty:
            gor_plot = px.scatter(
                well_gor,
                x='Puits',
                y='GOR',
                size='Q Huile Corr (Sm³/j)',
                color='GOR',
                title='Gas-Oil Ratio by Well',
                color_continuous_scale='Viridis',
                labels={'GOR': 'Gas-Oil Ratio'},
                size_max=25
            )
            gor_plot.update_layout(
                xaxis_title='Well',
                yaxis_title='Gas-Oil Ratio',
                xaxis={'categoryorder':'total descending'}
            )
            plots['gor_by_well'] = gor_plot
    
    # Water Cut by Well
    if all(col in df.columns for col in ['Q Huile Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)']):
        # Calculate water cut for each well
        wc_data = df.copy()
        wc_data['Total_Liquid'] = wc_data['Q Huile Corr (Sm³/j)'] + wc_data['Q Eau Tot Corr (m³/j)']
        wc_data = wc_data[wc_data['Total_Liquid'] > 0]  # Avoid division by zero
        wc_data['Water_Cut'] = wc_data['Q Eau Tot Corr (m³/j)'] / wc_data['Total_Liquid'] * 100
        
        # Group by well
        well_wc = wc_data.groupby('Puits').agg({
            'Water_Cut': 'mean',
            'Q Huile Corr (Sm³/j)': 'sum'
        }).reset_index()
        
        # Filter to top 15 oil producers
        well_wc = well_wc[well_wc['Puits'].isin(top_wells)]
        
        if not well_wc.empty:
            wc_plot = px.scatter(
                well_wc,
                x='Puits',
                y='Water_Cut',
                size='Q Huile Corr (Sm³/j)',
                color='Water_Cut',
                title='Water Cut by Well',
                color_continuous_scale='Reds',
                labels={'Water_Cut': 'Water Cut (%)'},
                size_max=25
            )
            wc_plot.update_layout(
                xaxis_title='Well',
                yaxis_title='Water Cut (%)',
                xaxis={'categoryorder':'total descending'}
            )
            plots['watercut_by_well'] = wc_plot
    
    return plots

# Function to generate production decline plots
def create_decline_plots(historical_df, wells=None, n_wells=5):
    """
    Generate production decline plots for selected wells
    
    Parameters:
    historical_df (pandas.DataFrame): Historical production data
    wells (list): List of well names to include (if None, uses top n_wells)
    n_wells (int): Number of top wells to show if wells is None
    
    Returns:
    dict: Dictionary of Plotly figures
    """
    if historical_df.empty:
        return {}
    
    # Copy and convert date
    df = historical_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure numeric columns
    numeric_cols = [
        'Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # If no wells provided, find top producing wells
    if wells is None:
        well_production = df.groupby('Puits')['Q Huile Corr (Sm³/j)'].sum().sort_values(ascending=False)
        wells = well_production.head(n_wells).index.tolist()
    
    plots = {}
    
    # Filter data to selected wells
    wells_data = df[df['Puits'].isin(wells)].copy()
    
    # Ensure we have data
    if wells_data.empty:
        return {}
    
    # Sort by date
    wells_data = wells_data.sort_values('Date')
    
    # Create oil production decline plot
    if 'Q Huile Corr (Sm³/j)' in wells_data.columns:
        oil_decline = px.line(
            wells_data,
            x='Date',
            y='Q Huile Corr (Sm³/j)',
            color='Puits',
            title='Oil Production Decline',
            labels={'Q Huile Corr (Sm³/j)': 'Oil Production (Sm³/j)'}
        )
        oil_decline.update_layout(
            xaxis_title='Date',
            yaxis_title='Oil Production (Sm³/j)',
            legend_title='Well'
        )
        plots['oil_decline'] = oil_decline
    
    # Create gas production decline plot
    if 'Q Gaz Tot Corr (Sm³/j)' in wells_data.columns:
        gas_decline = px.line(
            wells_data,
            x='Date',
            y='Q Gaz Tot Corr (Sm³/j)',
            color='Puits',
            title='Gas Production Decline',
            labels={'Q Gaz Tot Corr (Sm³/j)': 'Gas Production (Sm³/j)'}
        )
        gas_decline.update_layout(
            xaxis_title='Date',
            yaxis_title='Gas Production (Sm³/j)',
            legend_title='Well'
        )
        plots['gas_decline'] = gas_decline
    
    # Create water production decline plot
    if 'Q Eau Tot Corr (m³/j)' in wells_data.columns:
        water_decline = px.line(
            wells_data,
            x='Date',
            y='Q Eau Tot Corr (m³/j)',
            color='Puits',
            title='Water Production Decline',
            labels={'Q Eau Tot Corr (m³/j)': 'Water Production (m³/j)'}
        )
        water_decline.update_layout(
            xaxis_title='Date',
            yaxis_title='Water Production (m³/j)',
            legend_title='Well'
        )
        plots['water_decline'] = water_decline
    
    return plots

# Enhanced feature engineering function
def enhanced_feature_engineering(df):
    """
    Create advanced engineered features for better production prediction
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing well production data
    
    Returns:
    pandas.DataFrame: DataFrame with additional engineered features
    """
    data = df.copy()
    
    # Basic features from current implementation
    if 'Q Huile Test (Sm³/h)' in data.columns:
        data['Test_Production'] = data['Q Huile Test (Sm³/h)'] * 24  # Convert hourly to daily
    
    data['Choke_Change'] = data['Duse (mm)'].diff()
    data['Days_Since_Test'] = (data['Date'] - pd.to_datetime(data['Date Dernier Test'])).dt.days
    
    # New advanced features
    
    # Pressure differential features
    data['Pressure_Differential'] = data['Pt (bar)'] - data['Pp (bar)']
    data['Normalized_Pressure_Diff'] = data['Pressure_Differential'] / data['Pt (bar)']
    
    # Theoretical flow calculation based on choke physics
    if 'Coef K' in data.columns:
        data['K_Factor'] = data['Coef K']
    else:
        # Default K factor if not available
        data['K_Factor'] = 0.1
    
    data['Theoretical_Flow'] = data['K_Factor'] * (data['Duse (mm)']**2) * (data['Pressure_Differential']**0.5)
    data['Flow_Efficiency'] = data['Q Huile Corr (Sm³/j)'] / data['Theoretical_Flow']
    
    # Time-based features
    data['Days_From_First_Production'] = (data['Date'] - data['Date'].min()).dt.days
    
    # Production decline indicators
    data['Production_Change'] = data['Q Huile Corr (Sm³/j)'].pct_change()
    data['Production_Trend'] = data['Q Huile Corr (Sm³/j)'].rolling(window=5, min_periods=1).mean()
    data['Production_Volatility'] = data['Q Huile Corr (Sm³/j)'].rolling(window=5, min_periods=1).std()
    
    # Fluid ratio features if available
    if all(col in data.columns for col in ['Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)']):
        # Gas-Oil Ratio
        data['GOR'] = data['Q Gaz Tot Corr (Sm³/j)'] / data['Q Huile Corr (Sm³/j)']
        data['GOR'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data['GOR'] = data['GOR'].fillna(method='ffill')
        
        # Water-Oil Ratio
        data['WOR'] = data['Q Eau Tot Corr (m³/j)'] / data['Q Huile Corr (Sm³/j)']
        data['WOR'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data['WOR'] = data['WOR'].fillna(method='ffill')
        
        # Total fluid production
        data['Total_Fluid'] = data['Q Huile Corr (Sm³/j)'] + data['Q Eau Tot Corr (m³/j)']
    
    # Operating hours and efficiency
    if 'Heures de marche' in data.columns:
        data['Operating_Hours'] = data['Heures de marche']
        data['Daily_Efficiency'] = data['Operating_Hours'] / 24.0
        data['Production_Per_Hour'] = data['Q Huile Corr (Sm³/j)'] / data['Operating_Hours'].replace(0, np.nan)
        data['Production_Per_Hour'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data['Production_Per_Hour'] = data['Production_Per_Hour'].fillna(method='ffill')
    
    # Handle any NaN values from calculations
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(method='ffill')
    data[numeric_cols] = data[numeric_cols].fillna(0)  # Fill any remaining NaNs
    
    return data

# Enhanced model training with hyperparameter tuning
def enhanced_train_production_model(X, y, model_type="Random Forest", perform_tuning=False):
    """
    Train a production prediction model with advanced techniques
    
    Parameters:
    X (pandas.DataFrame): Feature matrix
    y (pandas.Series): Target variable
    model_type (str): Type of model to train
    perform_tuning (bool): Whether to perform hyperparameter tuning
    
    Returns:
    tuple: (model, metrics, feature_importances)
    """
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.pipeline import Pipeline
    import xgboost as xgb
    
    # Train/test split with stratification based on quartiles if possible
    try:
        y_quartiles = pd.qcut(y, 4, labels=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y_quartiles
        )
    except:
        # Fallback if stratification fails (e.g., not enough data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Identify numerical features for scaling
    X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Model selection with pipelines and hyperparameter tuning
    if model_type == "Random Forest":
        if perform_tuning:
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('model', RandomForestRegressor(random_state=42))
            ])
            
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
            
            model = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        else:
            model = Pipeline([
                ('scaler', RobustScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
    elif model_type == "XGBoost":
        if perform_tuning:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', xgb.XGBRegressor(random_state=42))
            ])
            
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 6, 9],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.8, 0.9, 1.0]
            }
            
            model = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        else:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', xgb.XGBRegressor(n_estimators=100, random_state=42))
            ])
            
    elif model_type == "Gradient Boosting":
        if perform_tuning:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(random_state=42))
            ])
            
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.8, 0.9, 1.0]
            }
            
            model = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        else:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ])
            
    else:  # Neural Network
        if perform_tuning:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPRegressor(max_iter=1000, random_state=42))
            ])
            
            param_grid = {
                'model__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'model__alpha': [0.0001, 0.001, 0.01],
                'model__learning_rate_init': [0.001, 0.01],
                'model__activation': ['relu', 'tanh']
            }
            
            model = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        else:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
            ])
    
    # Train model with progress tracking
    with st.spinner(f"Training {model_type} model..." + 
                   (" with hyperparameter tuning (this may take a while)" if perform_tuning else "")):
        model.fit(X_train, y_train)
    
    # Extract best model if using GridSearchCV
    if perform_tuning:
        st.write(f"Best parameters: {model.best_params_}")
        if hasattr(model, 'best_estimator_'):
            best_model = model.best_estimator_
        else:
            best_model = model
    else:
        best_model = model
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if not np.any(y_test == 0) else np.nan
    }
    
    # Get feature importances if available
    feature_importances = None
    if hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['model'], 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.named_steps['model'].feature_importances_
        }).sort_values('importance', ascending=False)
    elif not perform_tuning and hasattr(model.named_steps['model'], 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.named_steps['model'].feature_importances_
        }).sort_values('importance', ascending=False)
    
    return best_model, metrics, feature_importances

# Improved Choke Performance / ML Tab
def ml_production_tab(historical_df):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import os
    import joblib
    
    st.header("🤖 Advanced Machine Learning Production Predictor")
    
    if historical_df.empty:
        st.warning("No historical data available. Please upload data first.")
        return
        
    # Setup tab structure for organizing content
    ml_tabs = st.tabs(["Model Training", "Production Prediction", "Time Series Forecasting"])
    
    with ml_tabs[0]:  # Model Training tab
        st.subheader("Train Production Model")
        
        # Well selection with metrics
        well_list = historical_df['Puits'].unique().tolist()
        
        # Show well statistics to help selection
        well_stats = []
        for well in well_list:
            well_data = historical_df[historical_df['Puits'] == well]
            stats = {
                'Well': well,
                'Data Points': len(well_data),
                'Latest Production': well_data['Q Huile Corr (Sm³/j)'].iloc[-1] if not well_data.empty else 0,
                'Latest Date': well_data['Date'].max() if not well_data.empty else None
            }
            well_stats.append(stats)
        
        well_stats_df = pd.DataFrame(well_stats).sort_values('Data Points', ascending=False)
        
        # Display wells with enough data for modeling
        valid_wells = well_stats_df[well_stats_df['Data Points'] >= 7]['Well'].tolist()
        
        if not valid_wells:
            st.warning("No wells have sufficient data for reliable modeling (minimum 10 data points needed)")
            return
        
        # Select well with sufficient data
        selected_well = st.selectbox(
            "Select Well for Production Prediction",
            options=valid_wells,
            key="ml_well_select",
            help="Select a well with at least 10 data points for reliable modeling"
        )
        
        # Filter data for selected well
        well_data = historical_df[historical_df['Puits'] == selected_well].copy()
        well_data = well_data.sort_values('Date')
        
        # Display data statistics
        st.write("**Well Data Statistics:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", len(well_data))
        with col2:
            st.metric("Date Range", f"{well_data['Date'].min().strftime('%Y-%m-%d')} to {well_data['Date'].max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("Avg. Production", f"{well_data['Q Huile Corr (Sm³/j)'].mean():.2f} Sm³/j")
        with col4:
            st.metric("Latest Production", f"{well_data['Q Huile Corr (Sm³/j)'].iloc[-1]:.2f} Sm³/j")
        
        # Feature Engineering - check required columns are present
        required_cols = [
            'Date', 'Duse (mm)', 'Pt (bar)', 'Pp (bar)', 
            'Q Huile Corr (Sm³/j)'
        ]
        
        if all(col in well_data.columns for col in required_cols):
            # Apply enhanced feature engineering
            with st.spinner("Engineering features..."):
                well_data = enhanced_feature_engineering(well_data)
            
            # Feature selection
            st.subheader("Feature Selection")
            
            # Identify all potential features
            numeric_cols = well_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            potential_features = [col for col in numeric_cols if col != 'Q Huile Corr (Sm³/j)' 
                                 and not pd.isna(well_data[col]).all()]
            
            # Remove highly correlated features
            if len(potential_features) > 1:
                corr_matrix = well_data[potential_features].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
                potential_features = [col for col in potential_features if col not in to_drop]
            
            # Let user select features
            selected_features = st.multiselect(
                "Select Features for Model Training",
                options=potential_features,
                default=[
                    'Duse (mm)', 'Pt (bar)', 'Pp (bar)', 'Pressure_Differential',
                    'Theoretical_Flow', 'Days_From_First_Production'
                ] if all(f in potential_features for f in [
                    'Duse (mm)', 'Pt (bar)', 'Pp (bar)', 'Pressure_Differential',
                    'Theoretical_Flow', 'Days_From_First_Production'
                ]) else potential_features[:min(6, len(potential_features))],
                help="Select features to include in the prediction model"
            )
            
            if len(selected_features) < 2:
                st.warning("Please select at least 2 features for modeling")
            else:
                # Model configuration
                st.subheader("Model Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    model_type = st.selectbox(
                        "Select Model Type",
                        options=["Random Forest", "XGBoost", "Gradient Boosting", "Neural Network"],
                        index=0,
                        help="Choose the type of machine learning model to use"
                    )
                
                with col2:
                    hyperparameter_tuning = st.checkbox(
                        "Perform Hyperparameter Tuning", 
                        value=False,
                        help="Enables grid search for optimal model parameters (takes longer to train)"
                    )
                
                # Prepare data
                X = well_data[selected_features].copy()
                y = well_data['Q Huile Corr (Sm³/j)'].copy()
                
                # Check for data quality
                if X.isna().any().any():
                    st.warning("Selected features contain missing values. These will be filled with forward fill and zeros.")
                
                # Train model button
                if st.button("Train Production Prediction Model", type="primary"):
                    try:
                        # Train model
                        model, metrics, feature_importances = enhanced_train_production_model(
                            X, y, model_type, hyperparameter_tuning
                        )
                        
                        # Save model
                        model_file = f"prod_pred_model_{selected_well}.joblib"
                        joblib.dump(model, model_file)
                        
                        # Store model info in session state
                        st.session_state['trained_model'] = model
                        st.session_state['model_metrics'] = metrics
                        st.session_state['model_features'] = selected_features
                        st.session_state['feature_importances'] = feature_importances
                        
                        st.success(f"Model trained successfully and saved as {model_file}!")
                        
                        # Show model performance
                        st.subheader("Model Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("R² Score", f"{metrics['r2']:.3f}")
                        
                        with col2:
                            st.metric("MAE", f"{metrics['mae']:.2f} Sm³/j")
                        
                        with col3:
                            st.metric("RMSE", f"{metrics['rmse']:.2f} Sm³/j")
                        
                        with col4:
                            if not np.isnan(metrics.get('mape', np.nan)):
                                st.metric("MAPE", f"{metrics['mape']:.2f}%")
                        
                        # Feature importance
                        if feature_importances is not None:
                            st.subheader("Feature Importance")
                            
                            fig = px.bar(
                                feature_importances,
                                x='feature',
                                y='importance',
                                labels={'feature': 'Features', 'importance': 'Importance'},
                                title='Feature Importance'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show example predictions vs actual
                        st.subheader("Model Validation")
                        
                        # Get predictions for historical data
                        historical_preds = model.predict(X)
                        validation_df = pd.DataFrame({
                            'Date': well_data['Date'],
                            'Actual': y,
                            'Predicted': historical_preds
                        })
                        
                        # Plot actual vs predicted
                        fig = px.line(
                            validation_df, x='Date', 
                            y=['Actual', 'Predicted'],
                            labels={'value': 'Production (Sm³/j)', 'variable': ''},
                            title='Actual vs Predicted Production'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show error distribution
                        validation_df['Error'] = validation_df['Actual'] - validation_df['Predicted']
                        validation_df['Error_Percent'] = (validation_df['Error'] / validation_df['Actual']) * 100
                        
                        fig = px.histogram(
                            validation_df, x='Error',
                            title='Prediction Error Distribution',
                            labels={'Error': 'Error (Sm³/j)', 'count': 'Frequency'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
            
                # Load existing model
                model_file = f"prod_pred_model_{selected_well}.joblib"
                if os.path.exists(model_file) and 'trained_model' not in st.session_state:
                    if st.button("Load Existing Model"):
                        try:
                            model = joblib.load(model_file)
                            st.session_state['trained_model'] = model
                            st.session_state['model_features'] = selected_features
                            st.success("Existing model loaded successfully!")
                        except Exception as e:
                            st.error(f"Error loading model: {str(e)}")
        else:
            missing_cols = [col for col in required_cols if col not in well_data.columns]
            st.error(f"Missing required columns for production prediction: {', '.join(missing_cols)}")
    
    with ml_tabs[1]:  # Production Prediction tab
        st.subheader("Oil Production Prediction")
        
        if 'trained_model' in st.session_state:
            # Get model features
            model_features = st.session_state['model_features']
            
            # Create dynamic form based on model features
            st.write("Enter parameters for prediction:")
            
            # Organize input fields by categories
            choke_features = [f for f in model_features if 'Duse' in f or 'Choke' in f]
            pressure_features = [f for f in model_features if 'bar' in f or 'Pressure' in f]
            time_features = [f for f in model_features if 'Days' in f or 'Time' in f]
            other_features = [f for f in model_features if f not in choke_features + pressure_features + time_features]
            
            # Create input form with better organization
            input_values = {}
            
            # First row - choke features
            if choke_features:
                st.write("**Choke Parameters:**")
                cols = st.columns(min(3, len(choke_features)))
                for i, feature in enumerate(choke_features):
                    try:
                        default_value = float(well_data[feature].iloc[-1])
                    except:
                        default_value = 0.0
                    
                    input_values[feature] = cols[i % len(cols)].number_input(
                        f"{feature}",
                        min_value=0.0,
                        value=default_value,
                        step=0.5
                    )
            
            # Second row - pressure features
            if pressure_features:
                st.write("**Pressure Parameters:**")
                cols = st.columns(min(3, len(pressure_features)))
                for i, feature in enumerate(pressure_features):
                    try:
                        default_value = float(well_data[feature].iloc[-1])
                    except:
                        default_value = 0.0
                    
                    input_values[feature] = cols[i % len(cols)].number_input(
                        f"{feature}",
                        min_value=0.0,
                        value=default_value,
                        step=1.0
                    )
            
            # Third row - time features
            if time_features:
                st.write("**Time Parameters:**")
                cols = st.columns(min(3, len(time_features)))
                for i, feature in enumerate(time_features):
                    try:
                        default_value = float(well_data[feature].iloc[-1])
                    except:
                        default_value = 0.0
                    
                    input_values[feature] = cols[i % len(cols)].number_input(
                        f"{feature}",
                        min_value=0.0,
                        value=default_value,
                        step=1.0
                    )
            
            # Fourth row - other features
            if other_features:
                st.write("**Other Parameters:**")
                cols = st.columns(min(3, len(other_features)))
                for i, feature in enumerate(other_features):
                    try:
                        default_value = float(well_data[feature].iloc[-1])
                    except:
                        default_value = 0.0
                    
                    input_values[feature] = cols[i % len(cols)].number_input(
                        f"{feature}",
                        min_value=0.0,
                        value=default_value,
                        step=1.0
                    )
            
            # Predict button
            if st.button("Predict Production", key="predict_button"):
                try:
                    # Create input dataframe
                    input_df = pd.DataFrame([input_values], columns=model_features)
                    
                    # Make prediction
                    prediction = st.session_state['trained_model'].predict(input_df)[0]
                    
                    # Display results
                    st.success(f"Predicted Production: {prediction:.2f} Sm³/j")
                    
                    # Create gauge chart for visual feedback
                    max_production = well_data['Q Huile Corr (Sm³/j)'].max() * 1.2  # 20% more than max historical
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, max_production]},
                            'bar': {'color': "#2563EB"},
                            'steps': [
                                {'range': [0, max_production * 0.33], 'color': "#FECACA"},
                                {'range': [max_production * 0.33, max_production * 0.66], 'color': "#FEF3C7"},
                                {'range': [max_production * 0.66, max_production], 'color': "#DCFCE7"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': well_data['Q Huile Corr (Sm³/j)'].iloc[-1]
                            }
                        },
                        title={'text': "Predicted Production (Sm³/j)"}
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Compare with theoretical production if applicable
                    if all(f in well_data.columns for f in ['Duse (mm)', 'Pt (bar)', 'Pp (bar)', 'Coef K']):
                        K = well_data['Coef K'].mean()
                        duse = input_values.get('Duse (mm)', 0)
                        pt = input_values.get('Pt (bar)', 0)
                        pp = input_values.get('Pp (bar)', 0)
                        
                        theoretical = K * duse**2 * max(0, (pt - pp))**0.5
                        efficiency = (prediction / theoretical) * 100 if theoretical > 0 else 0
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Theoretical Production", f"{theoretical:.2f} Sm³/j")
                        with col2:
                            st.metric("Production Efficiency", f"{efficiency:.1f}%")
                    
                    # What-if analysis
                    st.subheader("What-If Analysis")
                    st.write("See how changing parameters affects production:")
                    
                    # Allow user to select parameter to vary
                    vary_param = st.selectbox(
                        "Select parameter to vary:",
                        options=model_features,
                        index=0 if model_features else None
                    )
                    
                    if vary_param:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            min_val = st.number_input(
                                "Minimum value", 
                                value=float(input_values[vary_param]) * 0.5
                            )
                        with col2:
                            max_val = st.number_input(
                                "Maximum value", 
                                value=float(input_values[vary_param]) * 1.5
                            )
                        with col3:
                            steps = st.number_input("Number of steps", value=10, min_value=2)
                        
                        if st.button("Generate What-If Analysis"):
                            # Create range of values
                            param_range = np.linspace(min_val, max_val, int(steps))
                            
                            # Generate predictions for each value
                            what_if_results = []
                            for val in param_range:
                                # Copy input values and update varied parameter
                                varied_inputs = input_values.copy()
                                varied_inputs[vary_param] = val
                                
                                # Create input dataframe
                                input_df = pd.DataFrame([varied_inputs], columns=model_features)
                                
                                # Make prediction
                                pred = st.session_state['trained_model'].predict(input_df)[0]
                                
                                # Save results
                                what_if_results.append({
                                    vary_param: val,
                                    'Predicted Production': pred
                                })
                            
                            # Create dataframe from results
                            what_if_df = pd.DataFrame(what_if_results)
                            
                            # Plot results
                            fig = px.line(
                                what_if_df, x=vary_param, 
                                y='Predicted Production',
                                labels={vary_param: vary_param, 'Predicted Production': 'Production (Sm³/j)'},
                                title=f'Production vs {vary_param}'
                            )
                            
                            # Add current value line
                            fig.add_vline(
                                x=input_values[vary_param],
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Current value"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        else:
            st.info("Please train or load a model in the 'Model Training' tab first.")
    
    with ml_tabs[2]:  # Time Series Forecasting tab
        st.subheader("Production Forecasting")
        
        # Well selection
        well_list = historical_df['Puits'].unique().tolist()
        valid_wells = []
        
        for well in well_list:
            well_data = historical_df[historical_df['Puits'] == well]
            if len(well_data) >= 15:  # Minimum data points for time series
                valid_wells.append(well)
        
        if not valid_wells:
            st.warning("No wells have sufficient data for time series forecasting (minimum 15 data points needed)")
        else:
            # Select well with sufficient data
            selected_well = st.selectbox(
                "Select Well for Forecasting",
                options=valid_wells,
                key="forecast_well_select"
            )
            
            # Filter data for selected well
            well_data = historical_df[historical_df['Puits'] == selected_well].copy()
            well_data = well_data.sort_values('Date')
            
            # Prepare time series data
            ts_data = well_data[['Date', 'Q Huile Corr (Sm³/j)']].copy()
            ts_data.set_index('Date', inplace=True)
            
            # Plot historical data
            st.subheader("Historical Production")
            fig = px.line(
                ts_data, y='Q Huile Corr (Sm³/j)',
                labels={'Q Huile Corr (Sm³/j)': 'Production (Sm³/j)'},
                title=f'Historical Production for Well {selected_well}'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast configuration
            st.subheader("Forecast Configuration")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                forecast_days = st.number_input(
                    "Forecast Days",
                    min_value=7,
                    max_value=365,
                    value=30,
                    step=7
                )
            
            with col2:
                forecast_model = st.selectbox(
                    "Forecast Model",
                    options=["SARIMA", "Prophet", "Exponential Smoothing"],
                    index=0
                )
            
            with col3:
                confidence_interval = st.slider(
                    "Confidence Interval (%)",
                    min_value=50,
                    max_value=95,
                    value=80,
                    step=5
                )
            
            # Generate forecast button
            if st.button("Generate Forecast", type="primary"):
                with st.spinner(f"Generating {forecast_days} day forecast using {forecast_model}..."):
                    try:
                        from statsmodels.tsa.statespace.sarimax import SARIMAX
                        from statsmodels.tsa.holtwinters import ExponentialSmoothing
                        
                        # Get production data
                        y = ts_data['Q Huile Corr (Sm³/j)'].values
                        
                        if forecast_model == "SARIMA":
                            # Simple automatic SARIMA
                            model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
                            model_fit = model.fit(disp=False)
                            
                            # Generate forecast
                            forecast_result = model_fit.get_forecast(steps=forecast_days)
                            forecast_mean = forecast_result.predicted_mean
                            forecast_ci = forecast_result.conf_int(alpha=(1-confidence_interval/100))
                            
                            # Create forecast dates
                            last_date = ts_data.index[-1]
                            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                            
                            # Create forecast dataframe
                            forecast_df = pd.DataFrame({
                                'Forecast': forecast_mean,
                                'Lower CI': forecast_ci[:, 0],
                                'Upper CI': forecast_ci[:, 1]
                            }, index=forecast_dates)
                            
                        elif forecast_model == "Prophet":
                            try:
                                from prophet import Prophet
                                
                                # Prepare data for Prophet
                                prophet_data = pd.DataFrame({
                                    'ds': ts_data.index,
                                    'y': ts_data['Q Huile Corr (Sm³/j)'].values
                                })
                                
                                # Create and fit Prophet model
                                prophet_model = Prophet(interval_width=confidence_interval/100)
                                prophet_model.fit(prophet_data)
                                
                                # Create future dataframe
                                future = prophet_model.make_future_dataframe(periods=forecast_days)
                                
                                # Generate forecast
                                forecast = prophet_model.predict(future)
                                
                                # Extract forecast data
                                historical_dates = ts_data.index
                                forecast_dates = pd.date_range(start=historical_dates[-1] + pd.Timedelta(days=1), periods=forecast_days)
                                
                                forecast_df = pd.DataFrame({
                                    'Forecast': forecast['yhat'].iloc[-forecast_days:].values,
                                    'Lower CI': forecast['yhat_lower'].iloc[-forecast_days:].values,
                                    'Upper CI': forecast['yhat_upper'].iloc[-forecast_days:].values
                                }, index=forecast_dates)
                                
                            except ImportError:
                                st.error("Prophet package is not installed. Using Exponential Smoothing instead.")
                                forecast_model = "Exponential Smoothing"
                        
                        if forecast_model == "Exponential Smoothing":
                            # Triple Exponential Smoothing
                            try:
                                # Try to find seasonality
                                season_length = min(7, len(y) // 4)
                                
                                hw_model = ExponentialSmoothing(
                                    y,
                                    trend='add',
                                    seasonal='add',
                                    seasonal_periods=season_length
                                )
                            except:
                                # Fallback to non-seasonal model
                                hw_model = ExponentialSmoothing(y, trend='add')
                            
                            hw_fit = hw_model.fit()
                            
                            # Generate forecast
                            forecast_mean = hw_fit.forecast(forecast_days)
                            
                            # Create forecast dates
                            last_date = ts_data.index[-1]
                            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                            
                            # Create forecast dataframe with simple confidence interval
                            std_error = np.std(hw_fit.resid)
                            z_value = 1.96 if confidence_interval >= 95 else 1.645 if confidence_interval >= 90 else 1.28
                            
                            forecast_df = pd.DataFrame({
                                'Forecast': forecast_mean,
                                'Lower CI': forecast_mean - z_value * std_error,
                                'Upper CI': forecast_mean + z_value * std_error
                            }, index=forecast_dates)
                        
                        # Plot forecast
                        st.subheader(f"{forecast_days} Day Production Forecast")
                        
                        # Combine historical and forecast data
                        full_df = pd.DataFrame({
                            'Historical': ts_data['Q Huile Corr (Sm³/j)']
                        })
                        
                        pd.DataFrame({
                            'Forecast': forecast_df['Forecast'],
                            'Lower CI': forecast_df['Lower CI'],
                            'Upper CI': forecast_df['Upper CI']
                        })
                        
                        # Create plot
                        fig = go.Figure()
                        
                        # Add historical line
                        fig.add_trace(go.Scatter(
                            x=full_df.index,
                            y=full_df['Historical'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Add forecast line
                        fig.add_trace(go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['Forecast'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red')
                        ))
                        
                        # Add confidence interval
                        fig.add_trace(go.Scatter(
                            x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
                            y=forecast_df['Upper CI'].tolist() + forecast_df['Lower CI'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,0,0,0)'),
                            hoverinfo='skip',
                            showlegend=True,
                            name=f'{confidence_interval}% Confidence Interval'
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{forecast_days} Day Production Forecast for Well {selected_well}",
                            xaxis_title="Date",
                            yaxis_title="Production (Sm³/j)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display forecast statistics
                        st.subheader("Forecast Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Average Forecast", 
                                f"{forecast_df['Forecast'].mean():.2f} Sm³/j",
                                delta=f"{forecast_df['Forecast'].mean() - ts_data['Q Huile Corr (Sm³/j)'].iloc[-30:].mean():.2f}"
                            )
                        
                        with col2:
                            st.metric(
                                "End Production", 
                                f"{forecast_df['Forecast'].iloc[-1]:.2f} Sm³/j",
                                delta=f"{forecast_df['Forecast'].iloc[-1] - ts_data['Q Huile Corr (Sm³/j)'].iloc[-1]:.2f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Min Forecast", 
                                f"{forecast_df['Lower CI'].min():.2f} Sm³/j"
                            )
                        
                        with col4:
                            st.metric(
                                "Max Forecast", 
                                f"{forecast_df['Upper CI'].max():.2f} Sm³/j"
                            )
                        
                        # Display forecast data table
                        with st.expander("Show Forecast Data"):
                            forecast_display = forecast_df.copy()
                            forecast_display.index = forecast_display.index.strftime('%Y-%m-%d')
                            forecast_display = forecast_display.round(2)
                            st.dataframe(forecast_display)
                            
                        # Save forecast button
                        if st.button("Save Forecast to Database"):
                            try:
                                # Create forecast records
                                forecast_records = []
                                for date, row in forecast_df.iterrows():
                                    forecast_records.append({
                                        'Puits': selected_well,
                                        'Date': date.strftime('%Y-%m-%d'),
                                        'Q Huile Forecast (Sm³/j)': row['Forecast'],
                                        'Forecast_Lower_CI': row['Lower CI'],
                                        'Forecast_Upper_CI': row['Upper CI'],
                                        'Forecast_Model': forecast_model,
                                        'Forecast_Date': pd.Timestamp.now().strftime('%Y-%m-%d')
                                    })
                                
                                # Save to database
                                conn = init_db()
                                c = conn.cursor()
                                
                                # Create forecast table if not exists
                                c.execute('''
                                    CREATE TABLE IF NOT EXISTS production_forecasts (
                                        puits TEXT,
                                        date TEXT,
                                        oil_forecast REAL,
                                        lower_ci REAL,
                                        upper_ci REAL,
                                        model_type TEXT,
                                        forecast_date TEXT,
                                        PRIMARY KEY (puits, date)
                                    )
                                ''')
                                
                                # Insert records
                                for record in forecast_records:
                                    c.execute('''
                                        INSERT OR REPLACE INTO production_forecasts 
                                        VALUES (:puits, :date, :Q Huile Forecast (Sm³/j), 
                                                :Forecast_Lower_CI, :Forecast_Upper_CI, 
                                                :Forecast_Model, :Forecast_Date)
                                    ''', record)
                                
                                conn.commit()
                                conn.close()
                                
                                st.success(f"Saved {len(forecast_records)} forecast records to database!")
                            except Exception as e:
                                st.error(f"Error saving forecast: {str(e)}")
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")

# Main application function
def main():
    # Sidebar with data management
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Sonatrach_logo.svg/320px-Sonatrach_logo.svg.png", width=200)
        st.title("🛢️ Oil Production Analytics")
        
        # Create tabs for different sidebar sections
        sidebar_tabs = st.tabs(["💾 Database", "⚙️ Settings"])
        
        # Database Management tab
        with sidebar_tabs[0]:
            st.header("Database Management")
            
            # Reorganized layout with clearer sections
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Data Import")
            
            with col2:
                st.subheader("📅 Data Selection")
            
            # Upload section - Left column
            with col1:
                with st.expander("Upload Excel File", expanded=True):
                    uploaded_file = st.file_uploader("Upload production data", type=['xlsx', 'xls'], key="excel_uploader")
                    
                    if uploaded_file is not None:
                        # Parse Excel file
                        df = parse_excel(uploaded_file)
                
                # Preview data in a separate expander
                if 'df' in locals() and df is not None:
                    with st.expander("Preview Data"):
                        st.dataframe(df.head())
                        
                    # Save to database button (now outside both expanders)
                    if st.button("Save to Database", type="primary"):
                        save_to_db(df)
                        st.session_state['refresh_dashboard'] = True
                        new_date = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d').iloc[0]
                        st.session_state['date_dropdown'] = new_date
                        st.rerun()
                
                with st.expander("Database File Transfer"):
                    # Export functionality
                    if st.button("⬇️ Export Database"):
                        export_database()
                    
                    # Import functionality
                    st.subheader("Import Database")
                    uploaded_db = st.file_uploader(
                        "Upload database file (.db, .sqlite)", 
                        type=['db', 'sqlite', 'sqlite3'],
                        accept_multiple_files=False,
                        key="db_uploader"
                    )
                    
                    if uploaded_db is not None:
                        import_database(uploaded_db)
            
            # Data selection - Right column
            with col2:
                # Get all available dates
                available_dates = get_available_dates()
                
                # Refresh button for date list
                if st.button("🔄 Refresh Data List"):
                    # Clear the cache to force reload of dates
                    st.cache_data.clear()
                    st.rerun()
                
                if not available_dates:
                    st.info("No data in database yet")
                else:
                    # Create dropdown for date selection
                    selected_date = st.selectbox(
                        "Select Production Date",
                        options=available_dates,
                        key="date_dropdown"
                    )
                    
                    # Automatically load data for selected date
                    if selected_date:
                        df = load_from_db(selected_date)
                        
                        # Show loading confirmation
                        st.success(f"✅ Loaded data for {selected_date}")
                        
                        # Add button to view this data
                        if st.button("View Selected Data"):
                            # Set active tab to Data View
                            st.session_state['active_tab'] = 5  # Index of Data View tab
                            st.rerun()
            
            # Data management section - Below both columns
            st.subheader("🗂️ Data Management")
            
            # Data deletion interface
            with st.expander("Delete Production Data"):
                if available_dates:
                    selected_dates = st.multiselect(
                        "Select dates to remove",
                        options=available_dates
                    )
                    
                    if selected_dates:
                        # Add delete button
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            delete_confirm = st.checkbox("Confirm deletion")
                        with col2:
                            if delete_confirm:
                                if st.button("❌ Delete Selected Dates", type="primary"):
                                    conn = init_db()
                                    c = conn.cursor()
                                    
                                    # Delete records
                                    placeholders = ','.join(['?'] * len(selected_dates))
                                    c.execute(f"DELETE FROM production_data WHERE date(Date) IN ({placeholders})", selected_dates)
                                    
                                    deleted_rows = conn.total_changes
                                    conn.commit()
                                    conn.close()
                                    
                                    st.success(f"Deleted {deleted_rows} records from selected dates!")
                                    # Clear cache to refresh date list
                                    st.cache_data.clear()
                                    st.rerun()
                else:
                    st.info("No data available to delete")
            
            # Database administration section
            with st.expander("⚠️ Database Administration"):
                st.warning("Danger Zone! These actions cannot be undone.")
                
                # Database reset functionality
                reset_database()
        
        # Settings tab
        with sidebar_tabs[1]:
            st.header("Dashboard Settings")
            
            # Add settings options
            st.checkbox("Enable advanced analytics", value=True, key="enable_advanced")
            st.selectbox("Default plot theme", ["plotly", "plotly_white", "plotly_dark"], key="plot_theme")
            st.slider("Number of top wells to display", min_value=5, max_value=20, value=10, key="n_top_wells")
            
            # Add application info
            st.divider()
            st.info("Oil Production Analytics v1.0.0")
            st.write("© 2023 Petroleum Engineering Team")
    
    # Main interface
    st.title("🛢️ Oil Production Analytics Dashboard")
    
    # Initialize session state for active tab if not exist
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = 0
    
    # Check if data is loaded
    if 'date_dropdown' in st.session_state and st.session_state.date_dropdown:
        current_date = st.session_state.date_dropdown
        df = load_from_db(current_date)
        historical_df = load_from_db()  # Load all historical data
        
        # Create dashboard tabs
        tab_titles = [
            "📊 Overview", 
            "🔍 Well Analysis", 
            "📈 Production Trends", 
            "🔮 Forecasting", 
            "💧 Well Washing",
            "🤖 ML Production Predictor" ,
            "📋 Data View"
        ]
        
        tabs = st.tabs(tab_titles)
        
        # Set active tab based on session state
        # This would be used in the actual tab content sections
        st.session_state['active_tab']
        
        # Clear refresh flag after dashboard is updated
        if 'refresh_dashboard' in st.session_state and st.session_state['refresh_dashboard']:
            st.session_state['refresh_dashboard'] = False
        
        # Overview Tab
        with tabs[0]:
            st.header(f"Production Overview for {current_date}")
            
            # Calculate key metrics
            metrics = calculate_key_metrics(df)
            
            if metrics:
                # Show metrics in styled cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Oil Production", f"{metrics['total_oil']:.1f} Sm³/d")
                    st.metric("Average per well", f"{metrics['avg_oil_per_well']:.1f} Sm³/d")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Gas Production", f"{metrics['total_gas']:.1f} Sm³/d")
                    st.metric("Gas-Oil Ratio", f"{metrics['gor']:.1f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Water Production", f"{metrics['total_water']:.1f} m³/d")
                    st.metric("Water Cut", f"{metrics['water_cut']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Second row of metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Active Wells", f"{metrics['active_wells']}/{metrics['total_wells']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Average Runtime", f"{metrics['avg_runtime']:.1f} hours")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Create production plots
                st.subheader("Production Distribution")
                plots = create_production_plots(df)
                
                if plots:
                    # Show the plots
                    for plot_name, fig in plots.items():
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available to calculate metrics")
        
        # Well Analysis Tab
        with tabs[1]:
            st.header("Well Performance Analysis")
            
            # Pressure classification
            st.subheader("Pressure Classification")
            
            try:
                # Train or load pressure model
                try:
                    model = joblib.load('pressure_model.joblib')
                    scaler = joblib.load('pressure_scaler.joblib')
                except:
                    model, scaler = train_pressure_model(df)
                
                if model is not None and scaler is not None:
                    # Classify wells into pressure clusters
                    df['Pressure_Cluster'] = classify_pressures(df, model, scaler)
                    
                    # Map cluster numbers to labels
                    cluster_names = {
                        0: "Low Pressure",
                        1: "Medium Pressure",
                        2: "High Pressure",
                        -1: "Unknown"
                    }
                    
                    df['Pressure_Group'] = df['Pressure_Cluster'].map(cluster_names)
                    
                    # Create pressure classification plot
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Scatter plot of wells by pressure
                        pressure_scatter = px.scatter(
                            df[df['Pressure_Cluster'] >= 0],
                            x='Pt (bar)',
                            y='Pp (bar)',
                            color='Pressure_Group',
                            hover_name='Puits',
                            title='Well Pressure Classification',
                            size='Q Huile Corr (Sm³/j)',
                            size_max=20
                        )
                        st.plotly_chart(pressure_scatter, use_container_width=True)
                    
                    with col2:
                        # Count wells in each group
                        cluster_counts = df['Pressure_Group'].value_counts()
                        cluster_counts = cluster_counts[cluster_counts.index != 'Unknown']
                        
                        pressure_pie = px.pie(
                            values=cluster_counts.values,
                            names=cluster_counts.index,
                            title='Wells by Pressure Group',
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            hole=0.4
                        )
                        st.plotly_chart(pressure_pie, use_container_width=True)
                else:
                    st.info("Insufficient pressure data for classification")
            except Exception as e:
                st.error(f"Error in pressure classification: {str(e)}")
            
            # New: Well Pressure Trends Analysis
            # Enhanced Well Pressure Trends vs Production section
            st.subheader("Well Pressure Trends vs Production")

            if not historical_df.empty:
                # Select well for analysis
                well_list = historical_df['Puits'].unique().tolist()
                selected_well = st.selectbox("Select Well for Pressure Analysis", well_list)
                
                # Filter data for selected well
                well_data = historical_df[historical_df['Puits'] == selected_well].copy()
                well_data = well_data.sort_values('Date')
                if well_data['Date'].isnull().any():
                    st.warning("Some date values are invalid and will be skipped")
                    well_data = well_data.dropna(subset=['Date'])
                
                # Ensure numeric columns
                numeric_cols = ['Pt (bar)', 'Pp (bar)', 'Q Huile Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)', 
                                'Q Gaz Tot Corr (Sm³/j)', 'Duse (mm)']
                for col in numeric_cols:
                    if col in well_data.columns:
                        well_data[col] = pd.to_numeric(well_data[col], errors='coerce')
                
                # Create integrated pressure and production distribution plot
                fig_combined = go.Figure()

                # Add WHP trace
                fig_combined.add_trace(go.Scatter(
                    x=well_data['Date'],
                    y=well_data['Pt (bar)'],
                    mode='lines',
                    name='WHP (Pt)',
                    line=dict(color='blue')
                ))

                # Add Flowline Pressure trace
                fig_combined.add_trace(go.Scatter(
                    x=well_data['Date'],
                    y=well_data['Pp (bar)'],
                    mode='lines',
                    name='Flowline Pressure (Pp)',
                    line=dict(color='green')
                ))

                # Add Oil Production trace (secondary y-axis)
                fig_combined.add_trace(go.Scatter(
                    x=well_data['Date'],
                    y=well_data['Q Huile Corr (Sm³/j)'],
                    mode='lines',
                    name='Oil Production',
                    line=dict(color='red', dash='dash'),
                    yaxis='y2'
                ))

                # Add Water Production trace (secondary y-axis)
                fig_combined.add_trace(go.Scatter(
                    x=well_data['Date'],
                    y=well_data['Q Eau Tot Corr (m³/j)'],
                    mode='lines',
                    name='Water Production',
                    line=dict(color='cyan', dash='dash'),
                    yaxis='y2'
                ))

                # Add Gas Production trace (secondary y-axis)
                fig_combined.add_trace(go.Scatter(
                    x=well_data['Date'],
                    y=well_data['Q Gaz Tot Corr (Sm³/j)'],
                    mode='lines',
                    name='Gas Production',
                    line=dict(color='orange', dash='dash'),
                    yaxis='y3'
                ))

                # Add Choke Size trace (secondary y-axis)
                fig_combined.add_trace(go.Scatter(
                    x=well_data['Date'],
                    y=well_data['Duse (mm)'],
                    mode='lines',
                    name='Choke Size',
                    line=dict(color='purple', dash='dashdot'),
                    yaxis='y4'
                ))

                # Add vertical line at the present date with error handling
                try:
                    current_date = pd.Timestamp.now().normalize()
                    fig_combined.add_vline(
                        x=current_date, 
                        line_dash="dash", 
                        line_color="gray",
                        annotation_text="Present",
                        annotation_position="top"
                    )
                except Exception as e:
                    st.warning(f"Could not add current date marker: {str(e)}")

                # Update layout for multiple y-axes
                fig_combined.update_layout(
                    title=f'Integrated Pressure and Production Distribution for {selected_well}',
                    xaxis_title='Date',
                    yaxis=dict(title='Pressure (bar)', color='blue'),
                    yaxis2=dict(
                        title='Liquid Production (m³/j)',
                        color='red',
                        overlaying='y',
                        side='right',
                        anchor='free',
                        position=0.85
                    ),
                    yaxis3=dict(
                        title='Gas Production (Sm³/j)',
                        color='orange',
                        overlaying='y',
                        side='right',
                        anchor='x',
                        position=0.95
                    ),
                    yaxis4=dict(
                        title='Choke Size (mm)',
                        color='purple',
                        overlaying='y',
                        side='left',
                        anchor='free',
                        position=0.05
                    ),
                    legend=dict(orientation='h', y=-0.2),
                    margin=dict(l=100, r=100),
                    height=600
                )

                st.plotly_chart(fig_combined, use_container_width=True)
            else:
                st.info("No historical data available for pressure trend analysis")
            
            # Well Performance Analysis
            st.subheader("Well Performance Metrics")
            
            # Generate performance plots
            performance_plots = create_performance_plots(df)
            
            if performance_plots:
                # Display well performance plots
                for plot_name, fig in performance_plots.items():
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for performance analysis")
        
        # Production Trends Tab
        with tabs[2]:
            st.header("Production Trends")
            
            # Check if we have historical data
            if not historical_df.empty:
                # Allow well selection
                all_wells = historical_df['Puits'].unique().tolist()
                
                # Default to top producers if we have too many wells
                n_wells = min(5, len(all_wells))
                top_wells = historical_df.groupby('Puits')['Q Huile Corr (Sm³/j)'].sum().sort_values(ascending=False).head(n_wells).index.tolist()
                
                selected_wells = st.multiselect(
                    "Select wells to display",
                    options=all_wells,
                    default=top_wells
                )
                
                if selected_wells:
                    # Generate trend plots
                    trend_plots = create_decline_plots(historical_df, wells=selected_wells)
                    
                    if trend_plots:
                        # Display trend plots
                        for plot_name, fig in trend_plots.items():
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No trend data available for selected wells")
                else:
                    st.info("Please select at least one well to display trends")
            else:
                st.info("No historical data available for trend analysis")
        
        # Forecasting Tab
        with tabs[3]:
            st.header("Production Forecasting")
            
            # Options for forecasting
            forecast_col1, forecast_col2 = st.columns(2)
            
            with forecast_col1:
                forecast_target = st.selectbox(
                    "Select production target to forecast",
                    options=["Q Huile Corr (Sm³/j)", "Q Gaz Tot Corr (Sm³/j)", "Q Eau Tot Corr (m³/j)"],
                    index=0
                )
            
            with forecast_col2:
                forecast_days = st.slider(
                    "Forecast horizon (days)",
                    min_value=7,
                    max_value=90,
                    value=30,
                    step=7
                )
            
            # Select well for forecasting
            all_wells = historical_df['Puits'].unique().tolist()
            
            if all_wells:
                selected_well = st.selectbox(
                    "Select well for forecasting",
                    options=all_wells
                )
                
                if st.button("Generate Forecast", type="primary"):
                    with st.spinner("Generating forecast..."):
                        # Generate forecast
                        forecast_results = forecast_production(
                            historical_df, 
                            selected_well, 
                            target_col=forecast_target, 
                            days_ahead=forecast_days
                        )
                        
                        if forecast_results and 'error' not in forecast_results:
                            st.success("Forecast generated successfully!")
                            
                            # Prepare data for visualization
                            historical_data = forecast_results['historical']
                            
                            # Create forecast visualization
                            forecast_fig = go.Figure()
                            
                            # Add historical data
                            forecast_fig.add_trace(go.Scatter(
                                x=historical_data.index,
                                y=historical_data[forecast_target],
                                mode='lines',
                                name='Historical',
                                line=dict(color='blue')
                            ))
                            
                            # Add ARIMA forecast if available
                            if 'ARIMA' in forecast_results['forecasts'] and 'error' not in forecast_results['forecasts']['ARIMA']:
                                arima_forecast = forecast_results['forecasts']['ARIMA']
                                
                                # Create future dates
                                last_date = historical_data.index[-1]
                                future_dates = [last_date + timedelta(days=i+1) for i in range(len(arima_forecast))]
                                
                                forecast_fig.add_trace(go.Scatter(
                                    x=future_dates,
                                    y=arima_forecast,
                                    mode='lines',
                                    name='ARIMA Forecast',
                                    line=dict(color='red', dash='dash')
                                ))
                            
                            # Add Prophet forecast if available
                            if 'Prophet' in forecast_results['forecasts'] and 'error' not in forecast_results['forecasts']['Prophet']:
                                prophet_forecast = forecast_results['forecasts']['Prophet']
                                
                                forecast_fig.add_trace(go.Scatter(
                                    x=prophet_forecast['ds'],
                                    y=prophet_forecast['yhat'],
                                    mode='lines',
                                    name='Prophet Forecast',
                                    line=dict(color='green', dash='dash')
                                ))
                                
                                # Add prediction interval
                                forecast_fig.add_trace(go.Scatter(
                                    x=pd.concat([prophet_forecast['ds'], prophet_forecast['ds'][::-1]]),
                                    y=pd.concat([prophet_forecast['yhat_upper'], prophet_forecast['yhat_lower'][::-1]]),
                                    fill='toself',
                                    fillcolor='rgba(0,176,0,0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='Prophet 80% Interval'
                                ))
                            
                            # Update layout
                            forecast_fig.update_layout(
                                title=f'{forecast_target} Forecast for {selected_well}',
                                xaxis_title='Date',
                                yaxis_title=forecast_target,
                                legend_title='Data Series',
                                template=st.session_state.get('plot_theme', 'plotly')
                            )
                            
                            # Display the forecast plot
                            st.plotly_chart(forecast_fig, use_container_width=True)
                            
                            # Display forecast metrics
                            st.subheader("Forecast Details")
                            col1, col2 = st.columns(2)
                            
                            if 'ARIMA' in forecast_results['forecasts'] and 'error' not in forecast_results['forecasts']['ARIMA']:
                                with col1:
                                    st.write("ARIMA Forecast (Last 5 days):")
                                    arima_df = pd.DataFrame({
                                        'Date': future_dates[-5:],
                                        'Forecast': arima_forecast[-5:]
                                    })
                                    st.dataframe(arima_df.style.format({'Forecast': '{:.2f}'}))

                            if 'Prophet' in forecast_results['forecasts'] and 'error' not in forecast_results['forecasts']['Prophet']:
                                with col2:
                                    st.write("Prophet Forecast (Last 5 days):")
                                    prophet_df = prophet_forecast[['ds', 'yhat']].tail(5)
                                    prophet_df.columns = ['Date', 'Forecast']
                                    st.dataframe(prophet_df.style.format({'Forecast': '{:.2f}'}))

                        elif 'error' in forecast_results:
                            st.error(forecast_results['error'])
                        else:
                            st.error("Error generating forecast")
            else:
                st.info("No wells available for forecasting")
        
        # Well Washing Tab
        with tabs[4]:
            st.header("Well Washing Analysis and MAP")
            
            if not historical_df.empty:
                # Add well selection dropdown
                washing_wells = historical_df['Puits'].unique().tolist()
                selected_washing_well = st.selectbox(
                    "Select Well for Washing Analysis",
                    options=washing_wells,
                    key="washing_well_select"
                )
                
                # Filter data for selected well
                well_washing_data = historical_df[historical_df['Puits'] == selected_washing_well].copy()
                well_washing_data = well_washing_data.sort_values('Date')
                
                # Calculate MAP (manque à produire) if necessary columns exist
                if all(col in well_washing_data.columns for col in ['MAP (Sm³/j)', 'Q Huile Corr (Sm³/j)']):
                    # Ensure numeric columns
                    numeric_wash_cols = ['MAP (Sm³/j)', 'Q Huile Corr (Sm³/j)', 'Heures de marche']
                    for col in numeric_wash_cols:
                        if col in well_washing_data.columns:
                            well_washing_data[col] = pd.to_numeric(well_washing_data[col], errors='coerce')
                    
                    # Create two columns for metrics and controls
                    map_col1, map_col2 = st.columns([2, 1])
                    
                    with map_col2:
                        st.subheader("MAP Statistics")
                        # Calculate key MAP metrics
                        avg_map = well_washing_data['MAP (Sm³/j)'].mean()
                        max_map = well_washing_data['MAP (Sm³/j)'].max()
                        total_map = well_washing_data['MAP (Sm³/j)'].sum()
                        
                        # Display metrics
                        st.metric("Average MAP", f"{avg_map:.2f} Sm³/j")
                        st.metric("Maximum MAP", f"{max_map:.2f} Sm³/j")
                        st.metric("Cumulative MAP", f"{total_map:.2f} Sm³")
                        
                        # Calculate production loss percentage
                        avg_production = well_washing_data['Q Huile Corr (Sm³/j)'].mean()
                        if avg_production > 0:
                            loss_percentage = (avg_map / (avg_production + avg_map)) * 100
                            st.metric("Production Loss", f"{loss_percentage:.1f}%")
                        
                        # Add date range filter
                        st.subheader("Date Range")
                        min_date = well_washing_data['Date'].min().date()
                        max_date = well_washing_data['Date'].max().date()
                        
                        start_date = st.date_input(
                            "Start Date",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date
                        )
                        
                        end_date = st.date_input(
                            "End Date",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date
                        )
                    
                    # Filter data based on date range
                    mask = (well_washing_data['Date'].dt.date >= start_date) & (well_washing_data['Date'].dt.date <= end_date)
                    filtered_washing_data = well_washing_data.loc[mask]
                    
                    with map_col1:
                        # Create MAP visualization
                        fig_map = go.Figure()
                        
                        # Add production trace
                        fig_map.add_trace(go.Scatter(
                            x=filtered_washing_data['Date'],
                            y=filtered_washing_data['Q Huile Corr (Sm³/j)'],
                            mode='lines',
                            name='Oil Production',
                            line=dict(color='green')
                        ))
                        
                        # Add MAP trace
                        fig_map.add_trace(go.Scatter(
                            x=filtered_washing_data['Date'],
                            y=filtered_washing_data['MAP (Sm³/j)'],
                            mode='lines',
                            name='MAP (Production Loss)',
                            line=dict(color='red')
                        ))
                        
                        # Add area for total potential production
                        fig_map.add_trace(go.Scatter(
                            x=filtered_washing_data['Date'],
                            y=filtered_washing_data['MAP (Sm³/j)'] + filtered_washing_data['Q Huile Corr (Sm³/j)'],
                            mode='lines',
                            name='Potential Production',
                            line=dict(color='blue', dash='dash')
                        ))
                        
                        # Add runtime hours on secondary axis
                        fig_map.add_trace(go.Scatter(
                            x=filtered_washing_data['Date'],
                            y=filtered_washing_data['Heures de marche'],
                            mode='lines',
                            name='Runtime Hours',
                            line=dict(color='purple'),
                            yaxis='y2'
                        ))
                        
                        # Update layout
                        fig_map.update_layout(
                            title=f'Production Loss Analysis for {selected_washing_well}',
                            xaxis_title='Date',
                            yaxis_title='Production Rate (Sm³/j)',
                            yaxis2=dict(
                                title='Runtime Hours',
                                overlaying='y',
                                side='right',
                                range=[0, 24]
                            ),
                            legend=dict(orientation='h', y=-0.2)
                        )
                        
                        st.plotly_chart(fig_map, use_container_width=True)
                    
                    # Washing events detection
                    st.subheader("Washing Events Detection")
                    
                    # Prepare data for washing events detection
                    well_washing_data['Production_Drop'] = well_washing_data['Q Huile Corr (Sm³/j)'].pct_change().fillna(0)
                    well_washing_data['MAP_Spike'] = well_washing_data['MAP (Sm³/j)'].pct_change().fillna(0)
                    
                    # Detect potential washing events (significant production drop followed by recovery)
                    threshold = -0.15  # 15% drop in production
                    
                    # Find days with significant production drops
                    potential_washing_days = well_washing_data[well_washing_data['Production_Drop'] <= threshold].copy()
                    
                    if not potential_washing_days.empty:
                        # Look for recovery within 7 days after each drop
                        washing_events = []
                        
                        for idx, row in potential_washing_days.iterrows():
                            event_date = row['Date']
                            
                            # Get data for 7 days after the event
                            after_event = well_washing_data[
                                (well_washing_data['Date'] > event_date) & 
                                (well_washing_data['Date'] <= event_date + pd.Timedelta(days=7))
                            ]
                            
                            if not after_event.empty:
                                # Check if production recovered
                                max_recovery = after_event['Q Huile Corr (Sm³/j)'].max()
                                
                                if max_recovery > row['Q Huile Corr (Sm³/j)']:
                                    recovery_pct = (max_recovery - row['Q Huile Corr (Sm³/j)']) / row['Q Huile Corr (Sm³/j)'] * 100
                                    
                                    # Only include events with significant recovery
                                    if recovery_pct > 10:  # 10% recovery
                                        recovery_date = after_event.loc[after_event['Q Huile Corr (Sm³/j)'].idxmax(), 'Date']
                                        recovery_days = (recovery_date - event_date).days
                                        
                                        washing_events.append({
                                            'Event Date': event_date,
                                            'Recovery Date': recovery_date,
                                            'Production Before': row['Q Huile Corr (Sm³/j)'],
                                            'Production After': max_recovery,
                                            'Recovery %': recovery_pct,
                                            'Recovery Days': recovery_days,
                                            'MAP During Event': after_event['MAP (Sm³/j)'].mean()
                                        })
                        
                        if washing_events:
                            # Create DataFrame of washing events
                            washing_df = pd.DataFrame(washing_events)
                            
                            # Display washing events
                            st.write(f"Detected {len(washing_events)} potential washing events:")
                            st.dataframe(washing_df.style.format({
                                'Production Before': '{:.2f}',
                                'Production After': '{:.2f}',
                                'Recovery %': '{:.1f}%',
                                'MAP During Event': '{:.2f}'
                            }))
                            
                            # Create washing events visualization
                            fig_washings = go.Figure()
                            
                            # Add production line
                            fig_washings.add_trace(go.Scatter(
                                x=filtered_washing_data['Date'],
                                y=filtered_washing_data['Q Huile Corr (Sm³/j)'],
                                mode='lines',
                                name='Oil Production',
                                line=dict(color='green')
                            ))
                            
                            # Add markers for washing events
                            event_dates = [event['Event Date'] for event in washing_events 
                                          if start_date <= event['Event Date'].date() <= end_date]
                            event_productions = [filtered_washing_data.loc[filtered_washing_data['Date'] == date, 'Q Huile Corr (Sm³/j)'].values[0] 
                                                if not filtered_washing_data.loc[filtered_washing_data['Date'] == date, 'Q Huile Corr (Sm³/j)'].empty 
                                                else 0 
                                                for date in event_dates]
                            
                            if event_dates:
                                fig_washings.add_trace(go.Scatter(
                                    x=event_dates,
                                    y=event_productions,
                                    mode='markers',
                                    name='Washing Events',
                                    marker=dict(
                                        color='red',
                                        size=12,
                                        symbol='triangle-down'
                                    )
                                ))
                            
                            # Update layout
                            fig_washings.update_layout(
                                title=f'Detected Washing Events for {selected_washing_well}',
                                xaxis_title='Date',
                                yaxis_title='Oil Production (Sm³/j)',
                                legend=dict(orientation='h', y=-0.2)
                            )
                            
                            st.plotly_chart(fig_washings, use_container_width=True)
                            
                            # Calculate washing effectiveness
                            st.subheader("Washing Effectiveness Analysis")
                            
                            # Calculate average metrics
                            avg_recovery_pct = washing_df['Recovery %'].mean()
                            avg_recovery_days = washing_df['Recovery Days'].mean()
                            avg_production_gain = (washing_df['Production After'] - washing_df['Production Before']).mean()
                            
                            # Display metrics in columns
                            eff_col1, eff_col2, eff_col3 = st.columns(3)
                            
                            with eff_col1:
                                st.metric("Average Recovery", f"{avg_recovery_pct:.1f}%")
                            
                            with eff_col2:
                                st.metric("Average Recovery Time", f"{avg_recovery_days:.1f} days")
                            
                            with eff_col3:
                                st.metric("Average Production Gain", f"{avg_production_gain:.2f} Sm³/j")
                            
                            # Calculate estimated production gain
                            total_gain = sum([(event['Production After'] - event['Production Before']) * 
                                              min(30, (end_date - event['Event Date'].date()).days) 
                                              for event in washing_events 
                                              if start_date <= event['Event Date'].date() <= end_date])
                            
                            st.metric("Estimated Total Production Gain", f"{total_gain:.2f} Sm³")
                        else:
                            st.info("No washing events detected based on production patterns")
                    else:
                        st.info("No significant production drops detected for washing analysis")
                else:
                    st.warning("Missing required data columns for MAP and washing analysis")
            else:
                st.info("No historical data available for washing analysis")
        
        # Choke Performance Tab
        with tabs[5]:
            ml_production_tab(historical_df)
        # Data View Tab
        with tabs[6]:
            st.header("Raw Data View")
            
            if not df.empty:
                # Add filtering options
                col1, col2 = st.columns(2)
                
                with col1:
                    wells_filter = st.multiselect(
                        "Filter by Well",
                        options=df['Puits'].unique().tolist(),
                        default=df['Puits'].unique().tolist()
                    )
                
                with col2:
                    columns_filter = st.multiselect(
                        "Select Columns to Display",
                        options=df.columns.tolist(),
                        default=['Puits', 'Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)', 'Pt (bar)', 'Pp (bar)']
                    )
                
                # Apply filters
                filtered_df = df[df['Puits'].isin(wells_filter)][columns_filter]
                
                # Display data
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=400
                )
                
                # Add download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download filtered data as CSV",
                    data=csv,
                    file_name=f"production_data_{current_date}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No data available to display")
    
    else:
        st.info("Please upload data or select a date from the database to begin analysis.")

if __name__ == "__main__":
    main()
