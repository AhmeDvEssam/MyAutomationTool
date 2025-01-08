import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
from scipy.stats import zscore
from ydata_profiling import ProfileReport
from sklearn.impute import KNNImputer
class analyisisToolkit:
    
    # Dummy function for correcting data types, replace with actual implementation
    
    def correct_data_types(df):
        corrections = []
        # Example: Change a column's type if needed
        # if 'some_column' in df.columns:
        #     old_type = df['some_column'].dtype
        #     df['some_column'] = df['some_column'].astype('desired_type')
        #     corrections.append(('some_column', old_type, 'desired_type'))
        return df, corrections
    
    def generate_business_report(df, report_filename="business_data_report.pdf"):
        from fpdf import FPDF
        import os
    
        # Initialize PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
    
        # Title and Introduction
        pdf.cell(200, 10, txt="Business Data Preparation Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=(
            "This report summarizes the key steps taken to prepare the dataset for analysis. "
            "The objective is to ensure data quality, consistency, and readiness for business insights."
        ))
        pdf.ln(10)
    
        # Data Overview
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="1. Data Overview", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=(
            f"Initial dataset contained {df.shape[0]} rows and {df.shape[1]} columns.\n"
            f"Summary of dataset:\n{df.describe(include='all').T.to_string()}\n"
        ))
        pdf.ln(10)
    
        # Key Cleaning Steps
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="2. Key Cleaning Steps", ln=True)
        pdf.set_font("Arial", size=12)
    
        # Missing Values
        missing = df.isnull().sum()
        missing_columns = missing[missing > 0]
        if not missing_columns.empty:
            pdf.multi_cell(0, 10, txt=(
                f"Missing values were detected in {len(missing_columns)} columns.\n"
                "Appropriate imputation techniques were applied based on data types and distributions."
            ))
        else:
            pdf.multi_cell(0, 10, txt="No missing values detected.")
        pdf.ln(10)
    
        # Duplicate Removal
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            pdf.multi_cell(0, 10, txt=f"Removed {duplicate_count} duplicate rows from the dataset.")
        else:
            pdf.multi_cell(0, 10, txt="No duplicate rows detected.")
        pdf.ln(10)
    
        # Outliers and Transformations
        pdf.cell(200, 10, txt="3. Outliers and Data Transformations", ln=True)
        outlier_columns = []  # Placeholder for any specific handling logic
        if outlier_columns:
            pdf.multi_cell(0, 10, txt=f"Outliers were detected and capped in the following columns: {outlier_columns}")
        else:
            pdf.multi_cell(0, 10, txt="No significant outliers detected.")
        pdf.ln(10)
    
        # Visualization Section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="4. Data Insights and Visualizations", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt="Visualizations of key metrics and distributions are included below.")
        
        # Example: Correlation Heatmap
        correlation = df.select_dtypes(include=[np.number]).corr()
        if not correlation.empty:
            heatmap_path = "correlation_heatmap.png"
            sns.heatmap(correlation, annot=True, cmap="coolwarm")
            plt.savefig(heatmap_path)
            pdf.image(heatmap_path, x=10, w=180)
            plt.close()
            os.remove(heatmap_path)  # Clean up the saved image
    
        # Conclusion and Recommendations
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="5. Conclusion and Recommendations", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=(
            "The data has been cleaned and prepared for analysis. Key steps included handling missing values, "
            "removing duplicates, and normalizing numeric data. Next steps involve further exploration or predictive modeling."
        ))
    
        # Save Report
        pdf.output(report_filename)
        print(f"Report saved as {report_filename}")
    
    def plot_missing_values(df, handle_missing=False):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing.plot(kind='bar', color='skyblue')
            plt.title('Missing Values Per Column', fontsize=14)
            plt.ylabel('Count', fontsize=12)
            plt.xlabel('Columns', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            if handle_missing:
                filling_method = input("Would you like to fill missing values? (mean/median/mode/drop): ").strip().lower()
                if filling_method == "mean":
                    df.fillna(df.mean(), inplace=True)
                elif filling_method == "median":
                    df.fillna(df.median(), inplace=True)
                elif filling_method == "mode":
                    df.fillna(df.mode().iloc[0], inplace=True)
                elif filling_method == "drop":
                    df.dropna(inplace=True)
                else:
                    print("Invalid method chosen. No changes applied.")
        else:
            print("No missing values to visualize.")
    def plot_correlation_heatmap(df):
        numeric_df = df.select_dtypes(include=[np.number])  # Select numeric columns
        numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]  # Drop low-variance columns
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            plt.figure(figsize=(12, 8))  # Increase figure size for readability
            sns.heatmap(
                corr, 
                annot=True, 
                cmap="coolwarm", 
                fmt=".2f", 
                annot_kws={"size": 8},  # Adjust annotation font size
                linewidths=0.5,  # Add grid lines for clarity
                cbar_kws={'shrink': 0.75}  # Shrink color bar for better alignment
            )
            plt.title("Correlation Heatmap", fontsize=16)
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
            plt.tight_layout()  # Prevent labels from being cut off
            plt.show()
        else:
            print("Not enough valid numeric columns for a correlation heatmap.")
    def plot_skewness(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skew_vals = df[numeric_cols].skew().dropna()
        if not skew_vals.empty:
            skew_vals = skew_vals[abs(skew_vals) > 0.5]  # Focus on highly skewed columns
            if not skew_vals.empty:
                skew_vals.plot(kind='bar', color='salmon')
                plt.title("Skewness of Numeric Columns (Skew > 1)")
                plt.ylabel("Skewness")
                plt.show()
            else:
                print("No significant skewness to display.")
        else:
            print("No numeric columns to visualize skewness.")
    def visualize_outliers(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col], color='skyblue')
            plt.title(f"Box Plot for {col} (Outliers Highlighted)")
            plt.xlabel(col)
            plt.show()
            
    def handle_outliers(df, method='zscore', threshold=3):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if method == 'zscore':
                z_scores = np.abs(zscore(df[col].dropna()))
                outliers = df[col][z_scores > threshold]
                print(f"Outliers in {col}: {outliers}")
                df[col] = df[col].where(z_scores <= threshold, np.nan)
            elif method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_condition = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                outliers = df[col][outlier_condition]
                print(f"Outliers in {col}: {outliers}")
                df[col] = df[col].where(~outlier_condition, np.nan)
        return df
        
    def plot_pairplot(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            sns.pairplot(df[numeric_cols])
            plt.show()
            
        # Normalization
    def normalize_data(df, method='minmax'):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if method == 'minmax':
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df
    
    # Imputation
    def knn_impute(df):
        # Select only numeric columns for KNN imputation
        df_numeric = df.select_dtypes(include=['float64', 'int64'])
        # Initialize the KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        # Apply the imputer to the numeric columns
        df_imputed_numeric = imputer.fit_transform(df_numeric)
        # Create a copy of the original dataframe and update the numeric columns with the imputed values
        df_imputed = df.copy()
        df_imputed[df_numeric.columns] = df_imputed_numeric
        return df_imputed
    # Capping Outliers
    
    def correct_data_types(df):
        corrections = []
        for col in df.columns:
            old_type = df[col].dtype
            # If the column is object type
            if old_type == 'object':
                # Check if all non-null values are integer-like
                if df[col].dropna().apply(lambda x: x.isdigit() if isinstance(x, str) else False).all():
                    df[col] = df[col].astype('Int64')  # Nullable integer
                    new_type = df[col].dtype
                    corrections.append((col, old_type, new_type))
                # Check if all non-null values are float-like
                elif df[col].dropna().apply(lambda x: x.replace('.', '', 1).isdigit() if isinstance(x, str) else False).all():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    new_type = df[col].dtype
                    corrections.append((col, old_type, new_type))
            # Leave other object columns unchanged
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Ensure proper handling of missing values for numeric columns
                if df[col].isna().sum() > 0 and old_type != 'float':
                    df[col] = df[col].astype('float')
                    new_type = df[col].dtype
                    corrections.append((col, old_type, new_type))
        return df, corrections
    
    # Function to cap outliers
    def cap_outliers(series, lower_percentile=1, upper_percentile=99):
        if pd.api.types.is_numeric_dtype(series):
            lower_cap = series.quantile(lower_percentile / 100)
            upper_cap = series.quantile(upper_percentile / 100)
            return series.clip(lower=lower_cap, upper=upper_cap)
        else:
            return series
    
    # Function to remove outliers using IQR
    def remove_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Replace outliers with NaN
        series = series.where((series >= lower_bound) & (series <= upper_bound), np.nan)
        return series
    
    # Main describe_pluss function
    def describe_pluss(df):
        """
        Enhanced describe function with a dynamic interactive menu:
        - Users can explore the dataset and perform preprocessing tasks.
        - Allows replacing placeholders with NaN before handling missing values.
        - Implements dynamic menu updates based on used options.
        """
        
        def display_menu(available_options):
            """
            Display the dynamic menu based on available options.
            """
            print("\nChoose an option from the menu:")
            for key, value in available_options.items():
                print(f"{key}. {value}")
        
        def replace_placeholders(df):
            """
            Replace placeholders with NaN in the dataset.
            """
            default_placeholders = ['?', '-', 'None', 'N/A', '']
            print("\nDefault placeholders to replace: ", default_placeholders)
            custom_placeholder = input(
                "Enter any additional placeholder (leave blank to skip): ").strip()
            if custom_placeholder:
                default_placeholders.append(custom_placeholder)
            df.replace(default_placeholders, np.nan, inplace=True)
            print("\nPlaceholders have been replaced with NaN.")
        
        def update_summary(df):
            """
            Recalculate the summary DataFrame with enhanced features, including outlier detection,
            recommendations, and reasoning for the chosen methods.
            """
            summary = df.describe(include='all').T
            summary['missing_count'] = df.isnull().sum()
            summary['duplicate_count'] = df.duplicated().sum()
            summary['mode'] = df.mode().iloc[0]
            summary['data_type'] = df.dtypes
            summary['skewness'] = np.nan
            summary['variance'] = np.nan
            summary['outliers'] = "N/A"
            summary['missing_value_handling'] = "N/A"
            summary['outlier_handling'] = "N/A"
            
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Calculate skewness and variance
                    skew_val = df[col].skew(skipna=True)
                    variance_val = df[col].var(skipna=True)
                    summary.at[col, 'skewness'] = skew_val
                    summary.at[col, 'variance'] = variance_val
                    
                    # Detect outliers using IQR method
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    summary.at[col, 'outliers'] = len(outliers)
                    
                    # Generate outlier handling recommendations
                    if len(outliers) > 0:
                        if abs(skew_val) < 0.5:
                            summary.at[col, 'outlier_handling'] = "Z-Score or Cap Percentiles (Symmetric Data)"
                        else:
                            summary.at[col, 'outlier_handling'] = "IQR Method (Skewed Data)"
                    else:
                        summary.at[col, 'outlier_handling'] = "No Action Needed"
                    
                    # Handle missing values
                    if summary.at[col, 'missing_count'] > 0:
                        if abs(skew_val) < 0.5:
                            summary.at[col, 'missing_value_handling'] = "Fill with Mean (Symmetric Data)"
                        else:
                            summary.at[col, 'missing_value_handling'] = "Fill with Median (Skewed Data)"
                
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    if summary.at[col, 'missing_count'] > 0:
                        summary.at[col, 'missing_value_handling'] = "Fill with Mode (Categorical Data)"
                    else:
                        summary.at[col, 'missing_value_handling'] = "No Action Needed"
                else:
                    summary.at[col, 'missing_value_handling'] = "Manual Inspection Needed"
            
            return summary
        
        def handle_user_choice(choice, df, available_options):
            nonlocal summary
            if choice == 1:  # Display Summary
                summary = update_summary(df)
                display(summary)
            elif choice == 2:  # Display DataFrame Head
                pd.set_option('display.max_columns', None)
                print("\nDisplaying df.head() with all columns:")
                display(df.head())
            elif choice == 3:  # Replace Placeholders with NaN
                replace_placeholders(df)
            elif choice == 4:  # Apply Recommended Filling Techniques
                summary = update_summary(df)  # Ensure the summary is up-to-date
                for col in df.columns:
                    recommendation = summary.at[col, 'missing_value_handling']
                    if "Mean" in recommendation:
                        df[col] = df[col].fillna(df[col].mean())
                    elif "Median" in recommendation:
                        df[col] = df[col].fillna(df[col].median())
                    elif "Mode" in recommendation:
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            df[col] = df[col].fillna(mode_val[0])
                print("\nMissing values have been filled based on the recommendations.")
            elif choice == 5:  # Remove Duplicates
                duplicates_count = df.duplicated().sum()
                df.drop_duplicates(keep='first', inplace=True)
                print(f"\n{duplicates_count} duplicate rows have been removed.")
            elif choice == 6:  # Check and Correct Data Types
                df, corrections = analyisisToolkit.correct_data_types(df)
                if corrections:
                    print("\nData type corrections applied:")
                    for col, old_type, new_type in corrections:
                        print(f"- Column '{col}': {old_type} -> {new_type}")
                else:
                    print("\nAll columns already have correct data types.")
            elif choice == 7:  # Normalize Numeric Columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                print("\nNumeric columns have been normalized.")
            elif choice == 8:  # Impute Missing Values (KNN)
                df_numeric = df.select_dtypes(include=['float64', 'int64'])
                imputer = KNNImputer(n_neighbors=5)
                df_imputed_numeric = imputer.fit_transform(df_numeric)
                df[df_numeric.columns] = df_imputed_numeric
                print("\nMissing values have been imputed using KNN.")
            elif choice == 9:  # Apply Recommended Outlier Handling Techniques
                summary = update_summary(df)
                for col in df.columns:
                    recommendation = summary.at[col, 'outlier_handling']
                    if "Z-Score" in recommendation or "Cap Percentiles" in recommendation:
                        df[col] = analyisisToolkit.cap_outliers(df[col])
                    elif "IQR" in recommendation:
                        df[col] = analyisisToolkit.remove_outliers_iqr(df[col])
                print("\nOutliers have been handled based on the recommendations.")
            elif choice == 10:  # Encode Categorical Variables
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    df = pd.get_dummies(df, columns=[col], drop_first=True)
                print("\nCategorical variables have been encoded.")
            elif choice == 11:  # Visualize Data
                while True:
                    visualize_choice = input(
                        "\nChoose the visualizations you want to display:"
                        "\n1. Missing Values\n2. Correlation Heatmap\n3. Skewness\n4. Outliers\n5. Pairplot\n"
                        "Choose (1-5 or 'exit' to stop): ").strip().lower()
                    if visualize_choice == '1':
                        analyisisToolkit.plot_missing_values(df)
                    elif visualize_choice == '2':
                        analyisisToolkit.plot_correlation_heatmap(df)
                    elif visualize_choice == '3':
                        analyisisToolkit.plot_skewness(df)
                    elif visualize_choice == '4':
                        analyisisToolkit.visualize_outliers(df)
                    elif visualize_choice == '5':
                        analyisisToolkit.plot_pairplot(df)
                    elif visualize_choice == 'exit':
                        break
                    else:
                        print("Invalid choice, please try again.")
            elif choice == 12:  # Generate Interactive HTML Report
                report_filename = input("Enter the filename for the report (default: business_data_report.html): ").strip()
                report_filename = report_filename if report_filename else "business_data_report.html"
                try:
                    profile = ProfileReport(df, title="Business Data Preparation Report", explorative=True)
                    profile.to_file(report_filename)
                    print(f"Interactive HTML report saved as {report_filename}")
                except Exception as e:
                    print(f"An error occurred while generating the HTML report: {e}")
            
            # Remove the used option unless it's always available
            if choice not in [1, 2, 11, 12]:
                available_options.pop(choice, None)
        
        # Initial setup
        summary = update_summary(df)
        always_available = {1: "Display Summary", 2: "Display DataFrame Head", 11: "Visualize Data (Missing Values, Correlation, Outliers, etc.)", 12: "Generate Interactive HTML Report"}
        all_options = {
            1: "Display Summary",
            2: "Display DataFrame Head",
            3: "Replace Placeholders with NaN",
            4: "Apply Recommended Filling Techniques",
            5: "Remove Duplicates",
            6: "Check and Correct Data Types",
            7: "Normalize Numeric Columns",
            8: "Impute Missing Values (KNN)",
            9: "Apply Recommended Outlier Handling Techniques",
            10: "Encode Categorical Variables",
            11: "Visualize Data (Missing Values, Correlation, Outliers, etc.)",
            12: "Generate Interactive HTML Report"
        }
        available_options = all_options.copy()
        
        # Menu loop
        display_menu(available_options)
        while True:
            try:
                choice = int(input("\nEnter your choice: ").strip())
                handle_user_choice(choice, df, available_options)
                
                repeat_choice = int(input("\nEnter your choice (1: Choose again, 2: Redisplay menu, 0: Stop): ").strip())
                if repeat_choice == 1:
                    if not available_options:
                        print("All options have been used. Exiting.")
                        break  # Break the loop if all options are used
                elif repeat_choice == 2:
                    display_menu(available_options)
                elif repeat_choice == 0:
                    print("Exiting...")
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")
