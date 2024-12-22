import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

# Function to detect outliers based on IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Function to detect outliers based on Z-score
def detect_outliers_zscore(df, column, threshold=3):
    df['z_score'] = zscore(df[column])
    outliers = df[(df['z_score'] > threshold) | (df['z_score'] < -threshold)]
    df.drop(columns=['z_score'], inplace=True)
    return outliers

# Function to detect outliers using Isolation Forest (ML approach)
def detect_outliers_isolation_forest(df, column):
    iso_forest = IsolationForest(contamination=0.01)
    df['anomaly'] = iso_forest.fit_predict(df[[column]])
    outliers = df[df['anomaly'] == -1]
    df.drop(columns=['anomaly'], inplace=True)
    return outliers

# Function to visualize outliers
def plot_outliers(df, column, method):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot showing outliers detected using {method}')
    plt.show()

# Generalized Outlier Detection Function
def generalized_outlier_detection(df, column, method='auto'):
    if df[column].dtype not in ['int64', 'float64']:
        print(f"Skipping column '{column}' because it is not numeric.")
        return pd.DataFrame()  # Return empty DataFrame for non-numeric columns

    # Auto method: Choose method based on data distribution
    if method == 'auto':
        skewness = df[column].skew()
        print(f"Skewness of column {column}: {skewness:.2f}")
        
        if abs(skewness) > 1:  # Highly skewed data
            print("Using IQR for outlier detection (due to skewness).")
            return detect_outliers_iqr(df, column)
        else:  # Normally distributed data
            print("Using Z-score for outlier detection (due to normality).")
            return detect_outliers_zscore(df, column)
    
    # Allow user to choose method
    elif method == 'iqr':
        print(f"Using IQR for outlier detection on column '{column}'.")
        return detect_outliers_iqr(df, column)
    elif method == 'zscore':
        print(f"Using Z-score for outlier detection on column '{column}'.")
        return detect_outliers_zscore(df, column)
    elif method == 'isolation_forest':
        print(f"Using Isolation Forest for outlier detection on column '{column}'.")
        return detect_outliers_isolation_forest(df, column)
    else:
        raise ValueError("Invalid method chosen. Choose from 'auto', 'iqr', 'zscore', or 'isolation_forest'.")

# Function to visualize missing values as a heatmap
def plot_missing_values(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

# Function to plot missing value counts as a bar chart
def plot_missing_counts(df):
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    
    if not missing_counts.empty:
        missing_counts.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), color='skyblue')
        plt.title('Missing Values Per Column')
        plt.ylabel('Count')
        plt.xlabel('Columns')
        plt.show()
    else:
        print("No missing values to display.")

# Function to visualize skewness for numeric columns
def plot_skewness(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, color='blue')
        plt.title(f'Distribution of {col} (Skewness: {df[col].skew():.2f})')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

# Function to plot a correlation heatmap
def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_cols.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()
    else:
        print("No numeric columns to display correlations.")

# Enhanced describe_plus with visuals and outlier detection
def describe_plus_with_visuals_and_outliers(df, method='auto'):
    summary = df.describe(include='all').T  # Basic describe

    # Visualize missing values
    print("\nVisualizing missing values...")
    plot_missing_values(df)
    plot_missing_counts(df)

    # Visualize skewness
    print("\nVisualizing skewness...")
    plot_skewness(df)

    # Visualize correlations
    print("\nVisualizing correlations...")
    plot_correlation_heatmap(df)

    # Detect outliers and visualize
    print("\nDetecting and visualizing outliers...")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        outliers = generalized_outlier_detection(df, col, method)
        if not outliers.empty:
            print(f"Outliers detected in column '{col}': {outliers.shape[0]}")
            plot_outliers(df, col, method)
        else:
            print(f"No outliers detected in column '{col}'.")

    return summary

def correct_data_types(df):
    """
    Corrects the data types of columns in a DataFrame based on their content.
    - Converts int-like strings or objects to integers.
    - Converts float-like strings or objects to floats.
    - Leaves categorical or mixed-type columns unchanged.
    """
    def infer_dtype(series):
        if series.dropna().apply(lambda x: str(x).replace('.', '', 1).isdigit() and float(x).is_integer()).all():
            return 'int'
        elif series.dropna().apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
            return 'float'
        else:
            return 'object'

    corrections = []
    for col in df.columns:
        original_dtype = df[col].dtype
        inferred_dtype = infer_dtype(df[col])
        
        if inferred_dtype == 'int':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif inferred_dtype == 'float':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        new_dtype = df[col].dtype
        if original_dtype != new_dtype:
            corrections.append((col, original_dtype, new_dtype))

    return df, corrections

def describe_plus(df):
    """
    Enhanced describe function:
    - Adds skewness for numeric columns.
    - Recommends filling techniques for missing values based on symmetry/skewness.
    - Allows the user to choose whether to apply the recommendations.
    - Checks and optionally corrects data types.
    - Detects and visualizes outliers.
    """
    summary = df.describe(include='all').T
    summary['missing_count'] = df.isnull().sum()
    summary['skewness'] = np.nan
    summary['recommendation'] = 'N/A'

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64', 'Int64']:
            skew_val = df[col].skew()
            summary.at[col, 'skewness'] = skew_val
            summary.at[col, 'recommendation'] = "Fill with Mean (Symmetric)" if abs(skew_val) < 0.5 else "Fill with Median (Skewed)"
        elif df[col].dtype == 'object':
            summary.at[col, 'recommendation'] = "Fill with Mode (Categorical)"
    
    print(summary)
    apply_filling = input("\nDo you want to apply the recommended filling techniques? (yes/no): ").strip().lower()

    if apply_filling == 'yes':
        for col in df.columns:
            if summary.at[col, 'recommendation'].startswith("Fill with Mean"):
                df[col] = df[col].fillna(df[col].mean())
            elif summary.at[col, 'recommendation'].startswith("Fill with Median"):
                df[col] = df[col].fillna(df[col].median())
            elif summary.at[col, 'recommendation'].startswith("Fill with Mode"):
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])

        print("\nMissing values have been filled based on the recommendations.")
    else:
        print("\nNo changes have been applied to missing values.")
    
    check_dtypes = input("Do you want to check and correct the data types? (yes/no): ").strip().lower()
    if check_dtypes == 'yes':
        df, corrections = correct_data_types(df)
        if corrections:
            print("\nData type corrections applied:")
            for col, old_type, new_type in corrections:
                print(f"- Column '{col}': {old_type} -> {new_type}")
        else:
            print("\nAll columns already have correct data types.")
    else:
        print("\nNo changes have been applied to data types.")
    
    viuals = input('Do you want to display visualization for data frame? yes/no: ').strip().lower()
    if viuals == 'yes':
        print('Please wait a sec...')
        describe_plus_with_visuals_and_outliers(df)  # Integrating outlier detection and visuals
    else:
        print('Fine')
    
    return summary
