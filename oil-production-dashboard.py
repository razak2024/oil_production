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

# Get all available dates in the database
@st.cache_data(ttl=300, show_spinner="Loading available dates...")
def get_available_dates():
    conn = init_db()
    try:
        cursor = conn.cursor()
        # Force a fresh count to invalidate cache when data changes
        cursor.execute("SELECT COUNT(*) FROM production_data")
        cursor.fetchone()[0]
        
        cursor.execute("SELECT DISTINCT date(Date) FROM production_data ORDER BY Date DESC")
        dates = [row[0] for row in cursor.fetchall()]
        return dates
    except sqlite3.Error:
        return []
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


# Improved database reset function with confirmation
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
        st.sidebar.success("Database has been completely reset!")
        st.rerun()
    elif confirmation and confirmation != "CONFIRM":
        st.sidebar.error("Confirmation text does not match 'CONFIRM'")

# Enhanced data saving function with progress indicator
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
            # Explicitly clear cache and rerun after commit
            st.cache_data.clear()
            st.rerun()
        else:
            progress_bar.progress(100)
            status_text.info("No new records to save after duplicate removal.")
            
    except Exception as e:
        status_text.error(f"Error saving to database: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

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
        st.sidebar.download_button(
            label="⬇️ Export Database",
            data=db_bytes,
            file_name="production_data_export.db",
            mime="application/x-sqlite3",
            help="Download a complete copy of the current database"
        )
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def import_database(uploaded_file):
    """
    Import a database file to replace the current database
    
    Parameters:
    uploaded_file: Uploaded file object from Streamlit
    """
    # Add confirmation to prevent accidental imports
    confirmation = st.sidebar.text_input("Type 'CONFIRM IMPORT' to replace database:")
    
    if confirmation == "CONFIRM IMPORT":
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
                st.sidebar.error("Uploaded file is not a valid SQLite database")
                return
            
            # Backup current database (just in case)
            backup_path = 'production_data_backup.db'
            if os.path.exists('production_data.db'):
                shutil.copy2('production_data.db', backup_path)
            
            # Replace current database with uploaded one
            shutil.copy2(tmp_path, 'production_data.db')
            
            st.sidebar.success("Database imported successfully! Backup saved as production_data_backup.db")
            st.sidebar.info("Please refresh the page to load the new data")
            
            # Clear cache to force reload
            st.cache_data.clear()
            
        except Exception as e:
            st.sidebar.error(f"Error importing database: {str(e)}")
            if os.path.exists(backup_path):
                st.sidebar.info("Original database has been restored from backup")
                shutil.copy2(backup_path, 'production_data.db')
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    elif confirmation and confirmation != "CONFIRM IMPORT":
        st.sidebar.error("Confirmation text does not match 'CONFIRM IMPORT'")

# Enhanced Excel parsing function with error handling
def parse_excel(uploaded_file):
    try:
        # Create progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Reading Excel file...")
        
        # Read Excel file without assuming date column
        df = pd.read_excel(uploaded_file)
        
        # Update progress
        progress_bar.progress(33)
        status_text.text("Processing dates...")
        
        # Try to find date column (case insensitive and with different common names)
        date_col = None
        possible_date_cols = ['Date', 'DATE', 'date', 'Jour', 'jour', 'Day', 'day']
        
        for col in possible_date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            progress_bar.progress(100)
            status_text.error("Could not find a date column in the Excel file. Please ensure your file has a column named 'Date'.")
            return None
        
        # Convert date column to datetime
        try:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            # Drop rows where date couldn't be parsed
            df = df[df['Date'].notna()]
            
            if df.empty:
                progress_bar.progress(100)
                status_text.error("No valid dates found in the date column.")
                return None
        except Exception as e:
            progress_bar.progress(100)
            status_text.error(f"Error processing dates: {str(e)}")
            return None
        
        # Update progress
        progress_bar.progress(66)
        status_text.text("Validating data...")
        
        # Check for required columns (with case-insensitive matching)
        required_cols = ['Puits', 'Q Huile Corr (Sm³/j)', 'Q Gaz Tot Corr (Sm³/j)', 'Q Eau Tot Corr (m³/j)']
        missing_cols = []
        
        for col in required_cols:
            if col not in df.columns:
                # Try case-insensitive match
                matching_cols = [c for c in df.columns if c.lower() == col.lower()]
                if matching_cols:
                    # Rename to standard column name
                    df.rename(columns={matching_cols[0]: col}, inplace=True)
                else:
                    missing_cols.append(col)
        
        if missing_cols:
            progress_bar.progress(100)
            status_text.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
        
        # Completion
        progress_bar.progress(100)
        status_text.success("Excel file processed successfully!")
        return df
    
    except Exception as e:
        status_text.error(f"Error parsing Excel file: {str(e)}")
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

# Main application function
def main():
    # Sidebar with data management
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Sonatrach_logo.svg/320px-Sonatrach_logo.svg.png", width=200)
        st.title("🛢️ Oil Production Analytics")
        
        # Create tabs for different sidebar sections
        sidebar_tabs = st.tabs(["💾 Database Management", "⚙️ Settings"])
        
        # Database Management tab
        with sidebar_tabs[0]:
            st.header("📅 Database Management")
            
            # Upload section - removed nested expander
            uploaded_file = st.file_uploader("📤 Upload Excel file", type=['xlsx', 'xls'])
            
            if uploaded_file is not None:
                # Parse Excel file
                df = parse_excel(uploaded_file)
                
                if df is not None:
                    # Show sample of data in a container instead of expander
                    preview_container = st.container()
                    with preview_container:
                        st.write("Preview Data:")
                        st.dataframe(df.head())
                    
                    # Save to database
                    if st.button("💾 Save to Database", type="primary"):
                        save_to_db(df)
                        # Force refresh after saving
                        st.cache_data.clear()
                        st.rerun()
            
            # Load existing data section
            available_dates = get_available_dates()
            
            if not available_dates:
                st.info("No data in database yet")
            else:
                # Add refresh button
                if st.button("🔄 Refresh Date List"):
                    st.cache_data.clear()
                    st.rerun()
                    
                # Create dropdown for date selection
                selected_date = st.selectbox(
                    "📋 Select Production Date to Load",
                    options=available_dates,
                    key="date_dropdown"
                )
                
                # Automatically load data for selected date
                if selected_date:
                    df = load_from_db(selected_date)
                    st.success(f"✅ Loaded data for {selected_date}")
            
            # Database transfer section
            st.subheader("🗄️ Database Transfer")
            
            # Export functionality
            export_database()
            
            # Import functionality with improved refresh
            uploaded_db = st.file_uploader(
                "Upload database file", 
                type=['db', 'sqlite', 'sqlite3'],
                accept_multiple_files=False,
                key="db_uploader"
            )
            
            if uploaded_db is not None:
                if st.button("Import Database", type="primary"):
                    import_database(uploaded_db)
                    # Force refresh after import
                    st.cache_data.clear()
                    st.rerun()
            
            # Data deletion interface
            st.subheader("🗑️ Delete Data")
            if available_dates:
                selected_dates = st.multiselect(
                    "Select dates to remove",
                    options=available_dates,
                    key="delete_dates"
                )
                
                if selected_dates and st.button("❌ Delete Selected Dates", type="primary"):
                    conn = init_db()
                    c = conn.cursor()
                    
                    placeholders = ','.join(['?'] * len(selected_dates))
                    c.execute(f"DELETE FROM production_data WHERE date(Date) IN ({placeholders})", selected_dates)
                    
                    deleted_rows = conn.total_changes
                    conn.commit()
                    conn.close()
                    
                    st.success(f"Deleted {deleted_rows} records from selected dates!")
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.info("No data available to delete")
            
            # Database administration section
            st.subheader("⚠️ Database Administration")
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
    
    # Check if data is loaded
    if 'date_dropdown' in st.session_state and st.session_state.date_dropdown:
        current_date = st.session_state.date_dropdown
        df = load_from_db(current_date)
        historical_df = load_from_db()  # Load all historical data
        
        # Create dashboard tabs
        tabs = st.tabs([
            "📊 Overview", 
            "🔍 Well Analysis", 
            "📈 Production Trends", 
            "🔮 Forecasting", 
            "💧 Well Washing",
            "📋 Data View"
        ])
        
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
        
        # Data View Tab
        with tabs[5]:
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
