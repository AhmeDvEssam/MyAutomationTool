import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from fpdf import FPDF
from scipy.stats import zscore, kurtosis,probplot,pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
from ydata_profiling import ProfileReport
from sklearn.impute import KNNImputer
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import joypy
import squarify
from wordcloud import WordCloud
from IPython.display import Markdown, display
import missingno as msno
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

class analyisisToolkit:
    # Create an instance of the analyisisToolkit class
    
    
    # Dummy function for correcting data types, replace with actual implementation
    def generate_business_report(df, report_filename="business_data_report.pdf"):
    
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


    
    def plot_pairplot(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            sns.pairplot(df[numeric_cols])
            plt.show()
            
    def plot_distribution(series, title="Distribution Plot", figsize=(10, 6)):
        """
        Plot a distribution plot (histogram + KDE) for a numeric column.
    
        Parameters:
            series (pd.Series): The numeric column to plot.
            title (str): Title of the plot.
            figsize (tuple): Size of the figure (width, height).
        """
        plt.figure(figsize=figsize)
        sns.histplot(series, kde=True, bins=30, color='skyblue')
        plt.title(title, fontsize=14)
        plt.xlabel(series.name, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.show()
        
    def plot_violin_comparison(df, numeric_col, categorical_col, title="Violin Plot"):
        """
        Plot a violin plot to compare distributions across categories.
        """
        plt.figure(figsize=(20, 20))
        sns.violinplot(x=categorical_col, y=numeric_col, data=df, hue=categorical_col, palette="coolwarm", legend=False)
        plt.title(title)
        plt.xlabel(categorical_col)
        plt.ylabel(numeric_col)
        plt.show()
        
    def plot_swarm(df, numeric_col, categorical_col, title="Swarm Plot", marker_size=3, use_strip=False):
        """
        Plot a swarm plot or strip plot to visualize individual data points.
    
        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            numeric_col (str): The name of the numeric column for the y-axis.
            categorical_col (str): The name of the categorical column for the x-axis.
            title (str): The title of the plot (default: "Swarm Plot").
            marker_size (int): The size of the markers in the plot (default: 3).
            use_strip (bool): If True, use a strip plot instead of a swarm plot (default: False).
        """
        plt.figure(figsize=(14, 8))
        if use_strip:
            # Use a strip plot if the data is too dense for a swarm plot
            sns.stripplot(x=categorical_col, y=numeric_col, data=df, hue=categorical_col, 
                          palette="coolwarm", legend=False, size=marker_size, jitter=True)
        else:
            # Use a swarm plot
            sns.swarmplot(x=categorical_col, y=numeric_col, data=df, hue=categorical_col, 
                          palette="coolwarm", legend=False, size=marker_size)
        
        plt.title(title, fontsize=16)
        plt.xlabel(categorical_col, fontsize=14)
        plt.ylabel(numeric_col, fontsize=14)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent overlapping labels
        plt.show()        
    def plot_missing_heatmap(df, title="Missing Values Heatmap"):
        """
        Plot a heatmap to visualize missing values in the dataset.
        """
        plt.figure(figsize=(14, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title(title)
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()
    
    def plot_qq(series, title="QQ Plot"):
        """
        Plot a QQ plot to check for normality.
        """
        # Convert the series to numeric, coercing errors to NaN
        series_numeric = pd.to_numeric(series, errors='coerce')
        # Drop NaN values (non-numeric data)
        series_numeric = series_numeric.dropna()
        if series_numeric.empty:
            print(f"Skipping QQ plot for '{series.name}': No numeric data available.")
            return
        # Plot the QQ plot
        plt.figure(figsize=(10, 8))
        probplot(series_numeric, dist="norm", plot=plt)
        plt.title(title)
        plt.show()    
        
    def plot_scatter_matrix(df, numeric_cols=None, title="Scatterplot Matrix", figsize=(20, 20), fontsize=10):
        """
        Plot a scatterplot matrix for numeric columns with increased size and clear labels.
        
        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            numeric_cols (list): List of numeric column names to include in the scatterplot matrix.
            title (str): Title of the plot.
            figsize (tuple): Size of the figure (width, height).
            fontsize (int): Font size for axis labels.
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Create the scatterplot matrix
        pd.plotting.scatter_matrix(df[numeric_cols], figsize=figsize, diagonal='kde')
        # Adjust layout and font size
        plt.suptitle(title, y=1.02, fontsize=fontsize + 2)  # Increase title font size
        plt.tight_layout()   
        # Set font size for axis labels
        for ax in plt.gcf().axes:
            ax.tick_params(axis='both', labelsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)       
        plt.show()  

    def plot_pie(df, column, title="Pie Plot", figsize=(10, 10), autopct='%1.1f%%', startangle=90):
        """
        Plot a pie chart for a categorical column, including missing values.
    
        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            column (str): The name of the categorical column to plot.
            title (str): Title of the plot.
            figsize (tuple): Size of the figure (width, height).
            autopct (str): Format string for displaying percentages on the pie chart.
            startangle (int): Angle at which the first slice starts (in degrees).
        """
        # Count the occurrences of each category
        counts = df[column].value_counts(dropna=False)  # Include missing values
        
        # Add a label for missing values if they exist
        if df[column].isnull().sum() > 0:
            counts['Missing'] = df[column].isnull().sum()
        
        # Plot the pie chart
        plt.figure(figsize=figsize)
        plt.pie(counts, labels=counts.index, autopct=autopct, startangle=startangle, colors=plt.cm.Paired.colors)
        plt.title(title, fontsize=14)
        plt.show()    
        
    def plot_regression(df, x_col, y_col, title="Regression Plot"):
        """
        Plot a regression plot for two numeric columns.
        """
        plt.figure(figsize=(10, 8))
        sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'color': 'skyblue'}, line_kws={'color': 'red'})
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

    def plot_missing_values(df):
        sns.heatmap(df.isnull(), cbar=False)
        plt.title("Missing Values")
        plt.show()
        
    def plot_histogram(df, title="Histogram"):
        sns.histplot(df, kde=True)
        plt.title(title)
        plt.show()
    
    def plot_bar(df, x_col, y_col, title="Bar Plot"):   ################### 15
        sns.barplot(x=x_col, y=y_col, data=df)
        plt.title(title)
        plt.show()
    
    def plot_count(df, columns, title="Count Plot"):
        """
        Plots count plots for one or more categorical columns in a DataFrame.
    
        Parameters:
            df (DataFrame): The input DataFrame.
            columns (list or str): Column(s) to plot. Can be a single column name or a list of column names.
            title (str): Base title for the plots. Defaults to "Count Plot".
        """
        # Ensure `columns` is a list for consistent handling
        if isinstance(columns, str):
            columns = [columns]  
        # Create a count plot for each column
        for column in columns:
            plt.figure(figsize=(8, 5))  # Set a size for each plot
            sns.countplot(x=column, data=df)
            plt.title(f"{title} - {column}")
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.show()

    def plot_line(df, x_col, y_col, title="Line Plot"): ############ 17
        sns.lineplot(x=x_col, y=y_col, data=df)
        plt.title(title)
        plt.show()
    
    def plot_heatmap(df, title="Heatmap"):
        # Select only numeric columns for the heatmap
        numeric_df = df.select_dtypes(include=[np.number]) 
        if numeric_df.empty:
            print("No numeric columns found for heatmap.")
            return
    
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(title)
        plt.show()    
        
    def plot_box(df, columns, title="Box Plot"):
        """
        Plots a box plot for one or more columns in a DataFrame.
    
        Parameters:
            df (DataFrame): The input DataFrame.
            columns (list or str): Column(s) to plot. Can be a single column name or a list of column names.
            title (str): Title of the plot. Defaults to "Box Plot".
        """
        # Ensure columns is a list for consistent handling
        if isinstance(columns, str):
            columns = [columns]
        # Filter the DataFrame to only include the specified columns
        selected_data = df[columns]
        # Create the box plot
        sns.boxplot(data=selected_data)
        plt.title(title)
        plt.xlabel("Columns")
        plt.ylabel("Values")
        plt.show()

    def plot_area(df, columns, title="Area Plot"):
        """
        Plots an area chart for the specified columns in a DataFrame.
    
        Parameters:
            df (DataFrame): The input DataFrame.
            columns (list or str): Column(s) to plot. Can be a single column name or a list of column names.
            title (str): Title of the plot. Defaults to "Area Plot".
        """
        # Ensure columns is a list for consistent handling
        if isinstance(columns, str):
            columns = [columns]
        # Plot the area chart
        df[columns].plot.area(alpha=0.6)
        plt.title(title)
        plt.ylabel("Values")
        plt.xlabel("Index")
        plt.show()   
    
    def plot_hexbin(df, x_col, y_col, title="Hexbin Plot"):  ################ 21
        sns.jointplot(x=x_col, y=y_col, data=df, kind='hex')
        plt.title(title)
        plt.show()
    
    def plot_kde(df,columns, title="KDE Plot"):
        if isinstance(columns, str):
            columns = [columns]
        sns.kdeplot(df[columns])
        plt.title(title)
        plt.show()
    
    def plot_facet_grid(df, x_col, y_col, title="Facet Grid"):
        g = sns.FacetGrid(df, col=x_col)
        g.map(sns.scatterplot, y_col)
        plt.title(title)
        plt.show()
    
    def plot_ridge(df, columns, title="Ridge Plot"):
        if isinstance(columns, str):
            columns = [columns]
        joypy.joyplot(df[columns])
        plt.title(title)
        plt.show()
    
    def plot_parallel_coordinates(df, class_column=None, title="Parallel Coordinates Plot"):
        # Check if the DataFrame is empty
        if df.empty:
            print("The DataFrame is empty. Cannot create a parallel coordinates plot.")
            return
        # If class_column is not specified, use the first categorical column
        if class_column is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not categorical_cols:
                print("No categorical column found. Please specify a class_column.")
                return
            class_column = categorical_cols[0]
        # Check if the specified class_column exists in the DataFrame
        if class_column not in df.columns:
            print(f"Column '{class_column}' not found in DataFrame.")
            return
        # Check if the class_column is categorical
        if not pd.api.types.is_categorical_dtype(df[class_column]) and not pd.api.types.is_object_dtype(df[class_column]):
            print(f"Column '{class_column}' must be categorical or object type.")
            return
        # Select numeric columns for the parallel coordinates plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("No numeric columns found for parallel coordinates plot.")
            return
        # Combine the class_column and numeric columns
        plot_data = df[[class_column] + numeric_cols]
        # Plot the parallel coordinates
        plt.figure(figsize=(14, 8))
        parallel_coordinates(plot_data, class_column=class_column, colormap='viridis')
        plt.title(title)
        plt.show() 
        
    def plot_radar(df, title="Radar Chart"):
        # Check if the DataFrame is empty
        if df.empty:
            print("The DataFrame is empty. Cannot create a radar chart.")
            return
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Ensure there's at least one numeric and one categorical column
        if not numeric_cols or not categorical_cols:
            print("Not enough data to create a radar chart. Need at least one numeric and one categorical column.")
            return
        # Use the first categorical column for grouping and the first numeric column for values
        group_col = categorical_cols[0]
        value_col = numeric_cols[0]
        # Aggregate data for the radar chart (e.g., mean of numeric column by categorical column)
        radar_data = df.groupby(group_col)[value_col].mean().reset_index()
        radar_data.rename(columns={value_col: 'value', group_col: 'category'}, inplace=True)
        # Plot the radar chart
        fig = px.line_polar(radar_data, r='value', theta='category', line_close=True)
        fig.update_layout(title=title)
        fig.show()
        
    def plot_treemap(df, title="Treemap"):
        # Prompt the user for the sizes and label columns
        sizes_col = input("Enter the column name for sizes (numeric values): ").strip()
        label_col = input("Enter the column name for labels (categories): ").strip() 
        # Check if the columns exist
        if sizes_col not in df.columns or label_col not in df.columns:
            print(f"Error: Columns '{sizes_col}' or '{label_col}' not found in the DataFrame.")
            return
        # Plot the treemap
        squarify.plot(sizes=df[sizes_col], label=df[label_col])
        plt.title(title)
        plt.axis('off')  # Hide the axes
        plt.show()        
    def plot_bubble(df, x_col, y_col, size_col, title="Bubble Plot"):  ######28
        sns.scatterplot(x=x_col, y=y_col, size=size_col, data=df)
        plt.title(title)
        plt.show()
    
    def plot_wordcloud(text, title="Word Cloud"):
        wordcloud = WordCloud().generate(' '.join(text))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')
        plt.show()
    def plot_wordcloud(text, title="Word Cloud"):
        """
        Plots a word cloud for the given text data.
        Parameters:
            text (pd.Series): The text data to visualize.
            title (str): The title of the plot.
        """
        # Convert all elements to strings
        text = text.astype(str)
        # Generate the word cloud
        wordcloud = WordCloud().generate(' '.join(text))
        # Display the word cloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')  # Hide the axes
        plt.show()
    
    def plot_3d_scatter(df, x_col, y_col, z_col, title="3D Scatter Plot"):
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col)
        fig.show()
    
    def plot_time_series_decomposition(series, period=None, title="Time Series Decomposition"):     ############################### 31
        # Check if the input is a pandas Series
        if not isinstance(series, pd.Series):
            print("Input must be a pandas Series.")
            return
        # Check if the Series contains numeric data
        if not pd.api.types.is_numeric_dtype(series):
            print("The Series must contain numeric data.")
            return
        # If the Series does not have a DatetimeIndex or PeriodIndex, create one
        if not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            print("The Series does not have a DatetimeIndex or PeriodIndex. Creating a default DatetimeIndex...")
            series.index = pd.date_range(start='2023-01-01', periods=len(series), freq='D')
        # If the index is a DatetimeIndex but has no frequency, infer it
        if isinstance(series.index, pd.DatetimeIndex) and series.index.freq is None:
            series = series.asfreq('D')  # Set a default frequency (e.g., daily)
            if series.index.freq is None:
                print("Could not infer frequency from the data. Please specify a period.")
                return
        # Perform time series decomposition
        try:
            decomposition = seasonal_decompose(series, model='additive', period=period)
            decomposition.plot()
            plt.title(title)
            plt.show()
        except Exception as e:
            print(f"An error occurred during time series decomposition: {e}")       
        
    def plot_lag(series, title="Lag Plot"): ###########################32
        pd.plotting.lag_plot(series)
        plt.title(title)
        plt.show()

    def plot_autocorrelation(series, title="Autocorrelation Plot"):  ###################### 33
        pd.plotting.autocorrelation_plot(series)
        plt.title(title)
        plt.show()
    
    def plot_pairgrid(df, title="PairGrid"):
        g = sns.PairGrid(df)
        g.map(sns.scatterplot)
        plt.title(title)
        plt.show()
    
    def plot_joint(df, x_col, y_col,title):   #############35
        sns.jointplot(x=x_col, y=y_col, data=df)
        plt.title(title)
        plt.show()
    
    def plot_dendrogram(df, title="Dendrogram"):
        # Select only numeric columns for clustering
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("No numeric columns found for dendrogram.")
            return
        # Perform hierarchical clustering
        Z = linkage(numeric_df, method='ward')
        # Plot the dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title(title)
        plt.xlabel("Data Points")
        plt.ylabel("Distance")
        plt.show()
    # Visualization 37: Contour Plot
    def plot_contour(self, df, x_col, y_col, title):
        """
        Plots a contour plot.
        """
        plt.figure(figsize=(8, 6))
        plt.tricontourf(df[x_col], df[y_col], df[x_col], cmap='viridis')
        plt.colorbar(label='Intensity')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        plt.show()

    # Visualization 38: Surface Plot
    def plot_surface(self, df, x_col, y_col, z_col, title):
        """
        Plots a 3D surface plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(df[x_col], df[y_col], df[z_col], cmap='viridis')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(title)
        plt.show()

    # Visualization 39: Streamgraph
    def plot_streamgraph(self, df, columns, title):
        """
        Plots a streamgraph using multiple numeric columns.
        """
        # Check if all selected columns are numeric
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' is not numeric. Streamgraph requires numeric data.")
        # Plot the streamgraph
        df.set_index(columns[0])[columns[1:]].plot(kind='area', stacked=True, colormap='viridis', figsize=(10, 6))
        plt.title(title)
        plt.xlabel(columns[0])  # Use the first column as the x-axis label
        plt.ylabel('Value')
        plt.show()    
    # Visualization 40: Network Graph
    def plot_network_graph(self, df, source_col, target_col, title):
        """
        Plots a network graph.
        """
        G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)
        plt.figure(figsize=(10, 8))
        nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10)
        plt.title(title)
        plt.show()

    # Visualization 41: 3D Surface Plot
    def plot_3d_surface(self, df, x_col, y_col, z_col, title):
        """
        Plots a 3D surface plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(df[x_col], df[y_col], df[z_col], cmap='viridis')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(title)
        plt.show()

    # Visualization 42: 3D Bar Plot
    def plot_3d_bar(self, df, x_col, y_col, z_col, title):
        """
        Plots a 3D bar plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(df[x_col], df[y_col], df[z_col], dx=0.5, dy=0.5, dz=0.5)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(title)
        plt.show()

    # Visualization 43: 3D Line Plot
    def plot_3d_line(self, df, x_col, y_col, z_col, title):
        """
        Plots a 3D line plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(df[x_col], df[y_col], df[z_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(title)
        plt.show()

    # Visualization 44: Interactive Scatter Plot (Plotly)
    def plot_interactive_scatter(self, df, x_col, y_col, title):
        """
        Plots an interactive scatter plot using Plotly.
        """
        fig = px.scatter(df, x=x_col, y=y_col, title=title)
        fig.show()

    # Visualization 45: Interactive Heatmap (Plotly)
    def plot_interactive_heatmap(self, df, title):
        """
        Plots an interactive heatmap using Plotly.
        """
        # Filter numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        # Plot the heatmap
        fig = px.imshow(numeric_df.corr(), title=title)
        fig.show()
    # Visualization 46: Interactive Line Plot (Plotly)
    def plot_interactive_line(self, df, x_col, y_col, title):
        """
        Plots an interactive line plot using Plotly.
        """
        fig = px.line(df, x=x_col, y=y_col, title=title)
        fig.show()

    
    
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
            
            # If the column is float type but contains integer-like values
            elif pd.api.types.is_float_dtype(df[col]):
                # Check if all non-null values are integer-like
                if df[col].dropna().apply(lambda x: x.is_integer() if not pd.isna(x) else False).all():
                    df[col] = df[col].astype('Int64')  # Nullable integer
                    new_type = df[col].dtype
                    corrections.append((col, old_type, new_type))
            
            # If the column is numeric but not float, ensure proper handling of missing values
            elif pd.api.types.is_numeric_dtype(df[col]):
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
    
    def remove_outliers_iqr(data, column):
        """Remove outliers using IQR."""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    def remove_outliers_zscore(data, column, threshold=3):
        """Remove outliers using Z-score."""
        z_scores = zscore(data[column])
        return data[(np.abs(z_scores) <= threshold)]
    
    def remove_outliers_percentile(data, column, lower_percentile=0.01, upper_percentile=0.99):
        """Remove outliers based on percentiles."""
        lower_bound = data[column].quantile(lower_percentile)
        upper_bound = data[column].quantile(upper_percentile)
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    def cap_outliers(data, column, lower_percentile=0.01, upper_percentile=0.99):
        """
        Cap outliers using percentile thresholds and return a new DataFrame.
        Converts the column to float64 before capping to avoid dtype issues.
        """
        if pd.api.types.is_numeric_dtype(data[column]):
            # Convert the column to float64 to ensure compatibility with clipped values
            data[column] = data[column].astype('float64')
            # Calculate the lower and upper bounds
            lower_bound = data[column].quantile(lower_percentile)
            upper_bound = data[column].quantile(upper_percentile)   
            # Cap the values using np.clip
            data.loc[:, column] = np.clip(data[column], lower_bound, upper_bound)
        return data        
    def log_transform(data, column):
        """Apply log transformation to reduce outlier impact."""
        data.loc[:, column] = np.log1p(data[column])  # Use .loc to avoid SettingWithCopyWarning
        return data
    
    def sqrt_transform(data, column):
        """Apply square root transformation to reduce outlier impact."""
        data.loc[:, column] = np.sqrt(data[column])  # Use .loc to avoid SettingWithCopyWarning
        return data
    
    def robust_scaling(data, column):
        """Scale data robustly, reducing the effect of outliers."""
        median = data[column].median()
        iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
        data.loc[:, column] = (data[column] - median) / iqr  # Use .loc to avoid SettingWithCopyWarning
        return data            
    

    def describe_pluss(df):
        """
        Enhanced describe function with a dynamic interactive menu.
        Returns the modified DataFrame at the end.
        """
        # --- Outlier Handling Functions ---
        Toolkit = analyisisToolkit()
        def bold_and_large(text):
            display(Markdown(f"# *{text}*"))

        def remove_outliers_iqr(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
        def remove_outliers_zscore(data, column, threshold=3):
            z_scores = zscore(data[column])
            return data[(np.abs(z_scores) <= threshold)]
    
        def remove_outliers_percentile(data, column, lower_percentile=0.01, upper_percentile=0.99):
            lower_bound = data[column].quantile(lower_percentile)
            upper_bound = data[column].quantile(upper_percentile)
            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
        def cap_outliers(data, column, lower_percentile=0.01, upper_percentile=0.99):
            """
            Cap outliers using percentile thresholds and return a new DataFrame.
            Converts the column to float64 before capping to avoid dtype issues.
            """
            if pd.api.types.is_numeric_dtype(data[column]):
                # Convert the column to float64 to ensure compatibility with clipped values
                data[column] = data[column].astype('float64')
                # Calculate the lower and upper bounds
                lower_bound = data[column].quantile(lower_percentile)
                upper_bound = data[column].quantile(upper_percentile)
                # Cap the values using np.clip
                data.loc[:, column] = np.clip(data[column], lower_bound, upper_bound)
            return data
    
        def log_transform(data, column):
            data.loc[:, column] = np.log1p(data[column].astype(float))  # Ensure float dtype
            return data
    
        def robust_scaling(data, column):
            median = data[column].median()
            iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
            data.loc[:, column] = (data[column].astype(float) - median) / iqr  # Ensure float dtype
            return data
    
        def detect_outlier_handling_method(data, column):
            skewness = data[column].skew()
            kurtosis_val = kurtosis(data[column], nan_policy='omit')
            n_outliers_iqr = len(data) - len(remove_outliers_iqr(data, column))
            n_outliers_zscore = len(data) - len(remove_outliers_zscore(data, column))
            n_outliers_percentile = len(data) - len(remove_outliers_percentile(data, column))
    
            # Decision Logic
            if abs(skewness) > 1:  # Highly skewed
                if n_outliers_iqr > 0.05 * len(data):  # More than 5% outliers
                    return "Log Transformation"
                else:
                    return "Cap using Percentiles"
            elif abs(skewness) <= 1:  # Symmetric or moderately skewed
                if n_outliers_zscore > 0.05 * len(data):  # Outliers based on normal distribution
                    return "Remove using Z-score"
                elif n_outliers_percentile > 0.05 * len(data):  # Outliers based on percentiles
                    return "Remove using Percentiles"
                else:
                    return "Robust Scaling"
            else:
                return "No Action Needed"       

        def group_by(df):
            """
            Groups the DataFrame by selected columns and applies an aggregation operation on a numeric column.
            """
            # Step 1: Ask the user to select columns to group by
            bold_and_large("Select columns to group by:")
            group_columns = get_column_choice(df, allow_multiple=True)
            if group_columns == "exit":
                bold_and_large("No columns selected. Exiting group_by function.")
                return df
            # Step 2: Ask the user to select the numeric column to aggregate
            bold_and_large("Select the numeric column to aggregate:")
            numeric_column = get_column_choice(df)
            if numeric_column == "exit":
                bold_and_large("No numeric column selected. Exiting group_by function.")
                return df
        
            # Ensure the selected column is numeric
            if not pd.api.types.is_numeric_dtype(df[numeric_column]):
                bold_and_large(f"Column '{numeric_column}' is not numeric. Please select a numeric column.")
                return df
        
            # Step 3: Ask the user to choose the aggregation operation
            bold_and_large("Choose the aggregation operation:")
            print("1. Mean")
            print("2. Sum")
            print("3. Max")
            print("4. Min")
            print("5. Count")
            print("6. Median")
            print("7. Standard Deviation")
            operation_choice = input("Enter the number of the operation: ").strip()
        
            # Map the user's choice to the corresponding aggregation function
            operation_map = {
                '1': 'mean',
                '2': 'sum',
                '3': 'max',
                '4': 'min',
                '5': 'count',
                '6': 'median',
                '7': 'std'
            }
        
            if operation_choice not in operation_map:
                bold_and_large("Invalid operation choice. Exiting group_by function.")
                return df
        
            operation = operation_map[operation_choice]
        
            # Perform the group by and aggregation
            grouped_df = df.groupby(group_columns)[numeric_column].agg(operation).reset_index()
        
            # Display the result
            bold_and_large(f"Grouped DataFrame with {operation} of {numeric_column}:")
            display(grouped_df)
        
            # Return the grouped DataFrame
            return grouped_df

        
    
        def rename_columns_interactive(df):
            """
            Allows the user to rename columns interactively.
            """
            # Display the column menu
            display_column_menu(df)
    
            # Get the user's column choice
            column_choice = get_column_choice(df)
    
            if column_choice == "exit":
                bold_and_large("No columns selected. Exiting.")
                return df
    
            # If the user selects "All Columns", rename all columns
            if isinstance(column_choice, list):
                columns_to_rename = column_choice
            else:
                columns_to_rename = [column_choice]
    
            # Rename the selected columns
            for col in columns_to_rename:
                new_name = input(f"Enter a new name for column '{col}': ").strip()
                if new_name:
                    df.rename(columns={col: new_name}, inplace=True)
                    bold_and_large(f"Column '{col}' renamed to '{new_name}'.")
                else:
                    bold_and_large(f"No new name provided for column '{col}'. Skipping.")
    
            return df
    
        def remove_unnecessary_columns(df):
            """
            Removes unnecessary columns from a DataFrame:
            - Columns with all missing values.
            - Columns with a single unique value.
            - Duplicated columns (columns with identical values).
            Modifies the DataFrame in place.
            """
            # Remove columns with all missing values
            df.dropna(axis=1, how='all', inplace=True)
        
            # Remove columns with a single unique value
            for col in df.columns:
                if df[col].nunique() == 1:
                    df.drop(col, axis=1, inplace=True)
        
            # Remove duplicated columns
            transposed_df = df.T
            unique_transposed_df = transposed_df.drop_duplicates()
            df.drop(df.columns.difference(unique_transposed_df.T.columns), axis=1, inplace=True)
        
            # No need to return the DataFrame since it's modified in place
    
        # --- Correlation Analysis Function ---
        def analyze_correlations(df, significance_level=0.05):
            """
            Analyze correlations between numeric variables in a DataFrame, allowing the user to filter results
            or manually choose columns for analysis.
            """
            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
    
            # Check if there are at least two numeric columns
            if len(numeric_cols) < 2:
                bold_and_large("Not enough numeric columns to calculate correlations.")
                return
    
            # Ask the user if they want to filter the results
            filter_choice = input("Do you want to filter the results? (yes/no): ").strip().lower()
    
            if filter_choice == "yes":
                # Display filtering options
                bold_and_large("Filter Options:")
                print("1. Filter by Correlation Strength")
                print("2. Filter by Significance Level")
                print("3. Filter by Both")
                print("4. Display All Outcomes")
                filter_option = input("Choose an option (1/2/3/4): ").strip()
    
                # Initialize filters
                correlation_filter = None
                significance_filter = None
    
                # If the user chooses "Display All Outcomes", skip filtering
                if filter_option == "4":
                    bold_and_large("Displaying all correlation outcomes without filtering.")
                else:
                    # Filter by Correlation Strength
                    if filter_option in ["1", "3"]:
                        bold_and_large("Correlation Strength Options:")
                        print("1. Strong Positive (Correlation > 0.7)")
                        print("2. Moderate Positive (0.3 < Correlation <= 0.7)")
                        print("3. Weak or No Correlation (-0.3 <= Correlation <= 0.3)")
                        print("4. Moderate Negative (-0.7 < Correlation < -0.3)")
                        print("5. Strong Negative (Correlation <= -0.7)")
                        strength_choice = input("Choose a correlation strength (1/2/3/4/5): ").strip()
    
                        if strength_choice == "1":
                            correlation_filter = lambda x: x > 0.7
                        elif strength_choice == "2":
                            correlation_filter = lambda x: 0.3 < x <= 0.7
                        elif strength_choice == "3":
                            correlation_filter = lambda x: -0.3 <= x <= 0.3
                        elif strength_choice == "4":
                            correlation_filter = lambda x: -0.7 < x < -0.3
                        elif strength_choice == "5":
                            correlation_filter = lambda x: x <= -0.7
                        else:
                            bold_and_large("Invalid choice. No correlation filter applied.")
    
                    # Filter by Significance Level
                    if filter_option in ["2", "3"]:
                        bold_and_large("Significance Level Options:")
                        print("1. Strong evidence (p-value < 0.001)")
                        print("2. Moderate evidence (0.001 <= p-value < 0.05)")
                        print("3. Weak evidence (0.05 <= p-value < 0.1)")
                        print("4. No evidence (p-value >= 0.1)")
                        significance_choice = input("Choose a significance level (1/2/3/4): ").strip()
    
                        if significance_choice == "1":
                            significance_filter = lambda x: x < 0.001
                        elif significance_choice == "2":
                            significance_filter = lambda x: 0.001 <= x < 0.05
                        elif significance_choice == "3":
                            significance_filter = lambda x: 0.05 <= x < 0.1
                        elif significance_choice == "4":
                            significance_filter = lambda x: x >= 0.1
                        else:
                            bold_and_large("Invalid choice. No significance filter applied.")
    
            else:
                # Allow the user to manually choose columns
                bold_and_large("Available numeric columns:")
                for i, col in enumerate(numeric_cols):
                    print(f"{i + 1}. {col}")
                selected_indices = input("Enter the numbers of the columns you want to analyze (comma-separated): ").strip()
                selected_indices = [int(idx.strip()) - 1 for idx in selected_indices.split(",")]
                numeric_cols = [numeric_cols[idx] for idx in selected_indices]
    
            # Check if there are at least two numeric columns after filtering or selection
            if len(numeric_cols) < 2:
                bold_and_large("Not enough numeric columns to calculate correlations. Please adjust your filters or select more columns.")
                return
    
            # Initialize a list to store correlation results
            results_list = []
    
            # Calculate correlations for all pairs of numeric columns
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Avoid duplicate pairs and self-correlations
                        # Calculate Pearson correlation and p-value
                        corr, p_value = pearsonr(df[col1], df[col2])
    
                        # Determine significance
                        if p_value < 0.001:
                            significance = "Strong evidence"
                        elif p_value < 0.05:
                            significance = "Moderate evidence"
                        elif p_value < 0.1:
                            significance = "Weak evidence"
                        else:
                            significance = "No evidence"
    
                        # Determine the meaning of the Pearson Correlation
                        if corr > 0.7:
                            meaning = "Strong positive linear correlation"
                        elif corr > 0.3:
                            meaning = "Moderate positive linear correlation"
                        elif corr > -0.3:
                            meaning = "Weak or no linear correlation"
                        elif corr > -0.7:
                            meaning = "Moderate negative linear correlation"
                        else:
                            meaning = "Strong negative linear correlation"
    
                        # Append the results as a dictionary to the list
                        results_list.append({
                            'Variable 1': col1,
                            'Variable 2': col2,
                            'Pearson Correlation': corr,
                            'P-value': p_value,
                            'Significance': significance,
                            'Correlation Meaning': meaning
                        })
    
            # Convert the list of dictionaries to a DataFrame
            results = pd.DataFrame(results_list)
    
            # Apply filters to the results (if the user didn't choose "Display All Outcomes")
            if filter_choice == "yes" and filter_option != "4":
                if correlation_filter:
                    results = results[results['Pearson Correlation'].apply(correlation_filter)]
                if significance_filter:
                    results = results[results['P-value'].apply(significance_filter)]
    
            # Check if there are any results after filtering
            if len(results) == 0:
                bold_and_large("No results found after applying filters. Please adjust your filters.")
                return
    
            # Display the correlation results as a DataFrame
            bold_and_large("Correlation Analysis Results:")
            display(results)
    
            # Print insights for each pair of columns
            bold_and_large("Insights:")
            for _, row in results.iterrows():
                var1, var2, corr, p_value, significance, meaning = row
                print(f"- {var1} and {var2}:")
                print(f"  - Pearson Correlation: {corr:.2f} ({meaning})")
                print(f"  - P-value: {p_value:.4f} ({significance} of correlation)")
                print(f"  - Interpretation: {get_correlation_interpretation(corr)}")
                print()
    
        def get_correlation_interpretation(corr):
            """
            Provide a textual interpretation of the Pearson Correlation value.
            """
            if corr > 0.7:
                return "A strong positive relationship exists. As one variable increases, the other tends to increase as well."
            elif corr > 0.3:
                return "A moderate positive relationship exists. As one variable increases, the other tends to increase slightly."
            elif corr > -0.3:
                return "A weak or no relationship exists. Changes in one variable do not significantly affect the other."
            elif corr > -0.7:
                return "A moderate negative relationship exists. As one variable increases, the other tends to decrease slightly."
            else:
                return "A strong negative relationship exists. As one variable increases, the other tends to decrease."
    
        # --- Helper Functions ---
        def display_menu(available_options):
            bold_and_large("Choose an option from the menu:")
            for key, value in available_options.items():
                print(f"{key}. {value}")
    
        def display_column_menu(df):
            bold_and_large("Available Columns:")
            for i, col in enumerate(df.columns, start=1):
                print(f"{i}. {col}")
            print(f"{len(df.columns) + 1}. All Columns")
            print(f"{len(df.columns) + 2}. Exit")
    
        def get_column_choice(df, allow_multiple=False):
            """
            Prompts the user to select one or more columns from the DataFrame.
            If allow_multiple is False, returns a single column name (string).
            If allow_multiple is True, returns a list of column names.
            """
            while True:
                choice = input(
                    f"Enter column numbers (1-{len(df.columns)}), "
                    f"'{len(df.columns) + 1}' for All Columns, "
                    f"'{len(df.columns) + 2}' to Exit, "
                    "or column names separated by commas: "
                ).strip()
        
                if not choice:
                    bold_and_large("No columns selected. Please try again.")
                    continue
        
                choices = [ch.strip() for ch in choice.split(',')]
                selected_columns = []
                valid = True
        
                for ch in choices:
                    if ch.isdigit():
                        col_num = int(ch)
                        if 1 <= col_num <= len(df.columns):
                            selected_columns.append(df.columns[col_num - 1])
                        elif col_num == len(df.columns) + 1:
                            return df.columns.tolist() if allow_multiple else df.columns[0]
                        elif col_num == len(df.columns) + 2:
                            return "exit"
                        else:
                            bold_and_large(f"Invalid choice: {ch}. Please try again.")
                            valid = False
                            break
                    else:
                        if ch in df.columns:
                            selected_columns.append(ch)
                        elif ch.lower() == "all":
                            return df.columns.tolist() if allow_multiple else df.columns[0]
                        elif ch.lower() == "exit":
                            return "exit"
                        else:
                            bold_and_large(f"Invalid choice: {ch}. Please try again.")
                            valid = False
                            break
        
                if valid:
                    if allow_multiple:
                        return selected_columns
                    else:
                        return selected_columns[0] if selected_columns else None     
        def slice(df):
            y = get_column_choice(df,allow_multiple=True)
            x = get_column_choice(df)
            return df.loc[0:x,y]
        def get_exact_columns(df, num_columns, prompt):
            """
            Prompts the user to select exactly `num_columns` columns from the DataFrame.
            Returns a list of selected columns or "exit" if the user chooses to exit.
            """
            while True:
                print(prompt)
                choice = input(
                    f"Enter {num_columns} column numbers (1-{len(df.columns)}), "
                    f"'{len(df.columns) + 1}' for All Columns, "
                    f"'{len(df.columns) + 2}' to Exit, "
                    "or column names separated by commas: "
                ).strip()
        
                if not choice:
                    bold_and_large("No columns selected. Please try again.")
                    continue
        
                choices = [ch.strip() for ch in choice.split(',')]
                selected_columns = []
                valid = True
        
                for ch in choices:
                    if ch.isdigit():
                        col_num = int(ch)
                        if 1 <= col_num <= len(df.columns):
                            selected_columns.append(df.columns[col_num - 1])
                        elif col_num == len(df.columns) + 1:
                            return df.columns.tolist()
                        elif col_num == len(df.columns) + 2:
                            return "exit"
                        else:
                            bold_and_large(f"Invalid choice: {ch}. Please try again.")
                            valid = False
                            break
                    else:
                        if ch in df.columns:
                            selected_columns.append(ch)
                        elif ch.lower() == "all":
                            return df.columns.tolist()
                        elif ch.lower() == "exit":
                            return "exit"
                        else:
                            bold_and_large(f"Invalid choice: {ch}. Please try again.")
                            valid = False
                            break
        
                if valid:
                    if len(selected_columns) == num_columns:
                        return selected_columns
                    else:
                        bold_and_large(f"Please select exactly {num_columns} columns.")        
        def replace_placeholders(df):
            default_placeholders = ['?', '-', 'None', 'N/A', '']
            bold_and_large("Default placeholders to replace:")
            print(default_placeholders)
            custom_placeholder = input("Enter any additional placeholder (leave blank to skip): ").strip()
            if custom_placeholder:
                default_placeholders.append(custom_placeholder)
            df.replace(default_placeholders, np.nan, inplace=True)
            bold_and_large("Placeholders have been replaced with NaN.")
    
        def update_summary(df):
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
    
            summary = df.describe(include='all').T
            summary['missing_count'] = df.isnull().sum()
            summary['duplicate_count'] = df.duplicated().sum()
            summary['mode'] = df.mode().iloc[0]
            summary['data_type'] = df.dtypes
            summary['skewness'] = np.nan
            summary['kurtosis'] = np.nan
            summary['variance'] = np.nan
            summary['outliers'] = "N/A"
            summary['missing_value_handling'] = "N/A"
            summary['outlier_handling'] = "N/A"
    
            summary['column_number'] = range(1, len(df.columns) + 1)
    
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    skew_val = df[col].skew(skipna=True)
                    kurtosis_val = kurtosis(df[col], nan_policy='omit')
                    variance_val = df[col].var(skipna=True)
                    summary.at[col, 'skewness'] = skew_val
                    summary.at[col, 'kurtosis'] = kurtosis_val
                    summary.at[col, 'variance'] = variance_val
    
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    summary.at[col, 'outliers'] = len(outliers)
    
                    recommendation = detect_outlier_handling_method(df, col)
                    summary.at[col, 'outlier_handling'] = recommendation
    
                    if summary.at[col, 'missing_count'] > 0:
                        if abs(skew_val) < 0.5:
                            summary.at[col, 'missing_value_handling'] = "Fill with Mean"
                        else:
                            summary.at[col, 'missing_value_handling'] = "Fill with Median"
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    if summary.at[col, 'missing_count'] > 0:
                        summary.at[col, 'missing_value_handling'] = "Fill with Mode"
                    else:
                        summary.at[col, 'missing_value_handling'] = "No Action Needed"
                else:
                    summary.at[col, 'missing_value_handling'] = "Manual Inspection Needed"
    
            return summary
    
        def handle_user_choice(choice, df, available_options):
            nonlocal summary
            if choice == 1:
                summary = update_summary(df)
                display(summary)
            elif choice == 2:
                pd.set_option('display.max_columns', None)
                bold_and_large("Displaying df.head() with all columns:")
                display(df.head())
            elif choice == 3:
                replace_placeholders(df)
            elif choice == 4:
                summary = update_summary(df)
                for col in df.columns:
                    recommendation = summary.at[col, 'missing_value_handling']
                    if "Mean" in recommendation:
                        df[col] = df[col].astype(float)
                        df[col] = df[col].fillna(df[col].mean())
                    elif "Median" in recommendation:
                        df[col] = df[col].astype(float)
                        df[col] = df[col].fillna(df[col].median())
                    elif "Mode" in recommendation:
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            df[col] = df[col].fillna(mode_val[0])
                bold_and_large("Missing values have been filled based on the recommendations.")
            elif choice == 5:
                duplicates_count = df.duplicated().sum()
                df.drop_duplicates(keep='first', inplace=True)
                # Reset the index
                df.reset_index(drop=True, inplace=True)
                bold_and_large(f"{duplicates_count} duplicate rows have been removed.")
                return df
            
            elif choice == 6:
                df, corrections = analyisisToolkit.correct_data_types(df)
                if corrections:
                    bold_and_large("Data type corrections applied:")
                    for col, old_type, new_type in corrections:
                        print(f"- Column '{col}': {old_type} -> {new_type}")
                else:
                    bold_and_large("All columns already have correct data types.")
            elif choice == 7:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if pd.api.types.is_categorical_dtype(df[col]):
                        continue  # Skip categorical columns
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val - min_val != 0:
                        df.loc[:, col] = (df[col] - min_val) / (max_val - min_val)  # Normalize
                    else:
                        bold_and_large(f"Warning: Column '{col}' has no variation; normalization not applied.")
                bold_and_large("Numeric columns have been normalized.")
            elif choice == 8:
                df_numeric = df.select_dtypes(include=['float64', 'int64', 'Int64'])
                imputer = KNNImputer(n_neighbors=5)
                df_imputed_numeric = imputer.fit_transform(df_numeric)
                df.loc[:, df_numeric.columns] = df_imputed_numeric
                bold_and_large("Missing values have been imputed using KNN.")
            elif choice == 9:
                summary = update_summary(df)
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        recommendation = summary.at[col, 'outlier_handling']
                        if recommendation == "Remove using IQR":
                            df = remove_outliers_iqr(df, col)  # Update the DataFrame
                        elif recommendation == "Remove using Z-score":
                            df = remove_outliers_zscore(df, col)  # Update the DataFrame
                        elif recommendation == "Remove using Percentiles":
                            df = remove_outliers_percentile(df, col)  # Update the DataFrame
                        elif recommendation == "Cap using Percentiles":
                            df = cap_outliers(df, col)  # Update the DataFrame
                        elif recommendation == "Log Transformation":
                            df = log_transform(df, col)  # Update the DataFrame
                        elif recommendation == "Robust Scaling":
                            df = robust_scaling(df, col)  # Update the DataFrame
                        elif recommendation == "No Action Needed":
                            bold_and_large(f"No action needed for column '{col}'.")
                        else:
                            bold_and_large(f"Unknown recommendation for column '{col}': {recommendation}")
                bold_and_large("Outliers have been handled based on the recommendations.")
                summary = update_summary(df)
            elif choice == 10:
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    df = pd.get_dummies(df, columns=[col], drop_first=True)
                bold_and_large("Categorical variables have been encoded.")
            elif choice == 11:
                # Define categories and their corresponding visualizations
                categories = {
                    "Data Distribution": [6, 14, 22, 24],
                    "Relationships Between Variables": [5, 11, 12, 21, 35, 34],
                    "Categorical Data": [7, 8, 15, 16, 19, 20, 27, 28],
                    "Time Series": [17, 31, 32, 33],
                    "Advanced Visualizations": [25, 26, 29, 30, 36, 37, 38, 39, 40],  # Added new visuals
                    "Data Quality": [1, 3, 4, 9, 10, 18],
                    "3D Visualizations": [41, 42, 43],  # New category for 3D visuals
                    "Interactive Visualizations": [44, 45, 46]  # New category for interactive visuals
                }
            
                # Map visualization numbers to their names
                visualization_names = {
                    1: "Missing Values",
                    2: "Correlation Heatmap",
                    3: "Skewness",
                    4: "Outliers",
                    5: "Pairplot",
                    6: "Distribution Plot",
                    7: "Violin Plot (Comparison)",
                    8: "Swarm Plot",
                    9: "Missing Values Heatmap",
                    10: "QQ Plot",
                    11: "Scatterplot Matrix",
                    12: "Regression Plot",
                    13: "Pie Plot",
                    14: "Histogram",
                    15: "Bar Plot",
                    16: "Count Plot",
                    17: "Line Plot",
                    18: "Heatmap (General)",
                    19: "Box Plot",
                    20: "Area Plot",
                    21: "Hexbin Plot",
                    22: "KDE Plot",
                    23: "Facet Grid",
                    24: "Ridge Plot",
                    25: "Parallel Coordinates Plot",
                    26: "Radar Chart",
                    27: "Treemap",
                    28: "Bubble Plot",
                    29: "Word Cloud",
                    30: "3D Scatter Plot",
                    31: "Time Series Decomposition Plot",
                    32: "Lag Plot",
                    33: "Autocorrelation Plot",
                    34: "PairGrid",
                    35: "Joint Plot",
                    36: "Dendrogram",
                    37: "Contour Plot",  # New visualization
                    38: "Surface Plot",  # New visualization
                    39: "Streamgraph",  # New visualization
                    40: "Network Graph",  # New visualization
                    41: "3D Surface Plot",  # New 3D visualization
                    42: "3D Bar Plot",  # New 3D visualization
                    43: "3D Line Plot",  # New 3D visualization
                    44: "Interactive Scatter Plot (Plotly)",  # New interactive visualization
                    45: "Interactive Heatmap (Plotly)",  # New interactive visualization
                    46: "Interactive Line Plot (Plotly)"  # New interactive visualization
                }
            
                # Display columns if requested
                display_columns = input("\033[1mDo You Want To display Columns? (y/n):\033[0m ").strip().lower()
                if display_columns == "y":
                    display_column_menu(df)
            
                while True:
                    # Display category menu
                    bold_and_large("Choose a category of visualizations:")
                    for i, category in enumerate(categories.keys(), start=1):
                        print(f"{i}. {category}")
                    print("0. Exit")
            
                    # Get user's category choice
                    category_choice = input("Enter the category number: ").strip()
                    if category_choice == "0":
                        bold_and_large("Exiting visualization menu.")
                        break
            
                    try:
                        category_choice = int(category_choice)
                        if 1 <= category_choice <= len(categories):
                            category_name = list(categories.keys())[category_choice - 1]
                            visualizations = categories[category_name]
            
                            while True:  # Inner loop for visualization menu
                                # Display visualizations in the chosen category with bold text
                                bold_and_large(f"Visualizations in '{category_name}':")
                                for viz in visualizations:
                                    print(f"{viz}. {visualization_names[viz]}")
                                print("Type 'back' to return to the category menu.")
            
                                # Get user's visualization choice
                                viz_choice = input("Enter the visualization number: ").strip()
                                if viz_choice.lower() == "back":
                                    break  # Exit the inner loop to go back to the category menu
            
                                try:
                                    viz_choice = int(viz_choice)
                                    if viz_choice in visualizations:
                                        # Call the corresponding visualization function
                                        bold_and_large(f"Generating {visualization_names[viz_choice]}...")
                                        if viz_choice == 1:
                                            analyisisToolkit.plot_missing_values(df)
                                        elif viz_choice == 2:
                                            analyisisToolkit.plot_correlation_heatmap(df)
                                        elif viz_choice == 3:
                                            analyisisToolkit.plot_skewness(df)
                                        elif viz_choice == 4:
                                            column_choice = get_column_choice(df)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.visualize_outliers(df[column_choice] if isinstance(column_choice, list) else df[[column_choice]])
                                        elif viz_choice == 5:
                                            analyisisToolkit.plot_pairplot(df)
                                        elif viz_choice == 6:
                                            column_choice = get_column_choice(df)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                for col in (column_choice if isinstance(column_choice, list) else [column_choice]):
                                                    analyisisToolkit.plot_distribution(df[col], title=f"Distribution Plot for {col}")
                                        elif viz_choice == 7:
                                            numeric_col = input("Enter numeric column name: ").strip()
                                            categorical_col = input("Enter categorical column name: ").strip()
                                            analyisisToolkit.plot_violin_comparison(df, numeric_col, categorical_col, title=f"Violin Plot: {numeric_col} by {categorical_col}")
                                        elif viz_choice == 8:
                                            numeric_col = input("Enter numeric column name: ").strip()
                                            categorical_col = input("Enter categorical column name: ").strip()
                                            # Check if the columns exist in the DataFrame
                                            if numeric_col not in df.columns or categorical_col not in df.columns:
                                                print("Error: One or both columns do not exist in the DataFrame.")
                                            else:
                                                # Ask the user if they want to use a strip plot instead of a swarm plot
                                                use_strip = input("The data seems dense. Use a strip plot instead? (y/n): ").strip().lower() == 'y'
                                                # Ask the user to adjust the marker size
                                                marker_size = int(input("Enter marker size (default is 3, reduce for dense data): ").strip() or 3)
                                                # Generate the plot
                                                analyisisToolkit.plot_swarm(
                                                    df, 
                                                    numeric_col=numeric_col, 
                                                    categorical_col=categorical_col, 
                                                    title=f"Swarm Plot: {numeric_col} by {categorical_col}", 
                                                    marker_size=marker_size, 
                                                    use_strip=use_strip
                                                )                                            
                                        elif viz_choice == 9:
                                            analyisisToolkit.plot_missing_heatmap(df, title="Missing Values Heatmap")
                                        elif viz_choice == 10:
                                            column_choice = get_column_choice(df)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.plot_qq(df[column_choice], title=f"QQ Plot for {column_choice}")
                                        elif viz_choice == 11:
                                            analyisisToolkit.plot_scatter_matrix(df, title="Scatterplot Matrix")
                                        elif viz_choice == 12:
                                            x_col = get_column_choice(df)
                                            if x_col == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                y_col = get_column_choice(df)
                                                if y_col == "exit":
                                                    bold_and_large("No action taken.")
                                                else:
                                                    analyisisToolkit.plot_regression(df, x_col, y_col, title=f"Regression Plot: {x_col} vs {y_col}")
                                        elif viz_choice == 13:
                                            column_choice = get_column_choice(df)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.plot_pie(df, column=column_choice, title=f"Pie Plot for {column_choice}")
                                        elif viz_choice == 14:
                                            column_choice = get_column_choice(df)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.plot_histogram(df[column_choice], title=f"Histogram for {column_choice}")
                                        elif viz_choice == 15:
                                            x_col = get_column_choice(df)
                                            if x_col == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                y_col = get_column_choice(df)
                                                if y_col == "exit":
                                                    bold_and_large("No action taken.")
                                                else:
                                                    analyisisToolkit.plot_bar(df, x_col, y_col, title=f"Bar Plot: {x_col} vs {y_col}")   ###########################
                                        elif viz_choice == 16:
                                            column_choice = get_column_choice(df, allow_multiple = True)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.plot_count(df, columns=column_choice, title=f"Count Plot for {column_choice}")   #########################
                                        elif viz_choice == 17:
                                            x_col = get_column_choice(df)
                                            if x_col == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                y_col = get_column_choice(df)
                                                if y_col == "exit":
                                                    bold_and_large("No action taken.")
                                                else:
                                                    analyisisToolkit.plot_line(df, x_col, y_col, title=f"Line Plot: {x_col} vs {y_col}")  ####################
                                        elif viz_choice == 18:
                                            analyisisToolkit.plot_heatmap(df, title="General Heatmap")
                                        elif viz_choice == 19:
                                            column_choice = get_column_choice(df,allow_multiple = True)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.plot_box(df, columns=column_choice, title=f"Box Plot for {column_choice}")  ######################
                                        elif viz_choice == 20:
                                            column_choice = get_column_choice(df,allow_multiple = True)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.plot_area(df, columns=column_choice, title=f"Area Plot for {column_choice}")
                                        elif viz_choice == 21:
                                            x_col = get_column_choice(df)
                                            if x_col == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                y_col = get_column_choice(df)
                                                if y_col == "exit":
                                                    bold_and_large("No action taken.")
                                                else:
                                                    # Call the updated plot_hexbin function with explicit keyword arguments
                                                    analyisisToolkit.plot_hexbin(df, x_col=x_col, y_col=y_col, title=f"Hexbin Plot: {x_col} vs {y_col}")                                                        
                                        elif viz_choice == 22:
                                            column_choice = get_column_choice(df,allow_multiple =True)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:    
                                                analyisisToolkit.plot_kde(df,columns = column_choice, title=f"KDE Plot for {column_choice}")
                                        elif viz_choice == 23:
                                            x_col = get_column_choice(df)
                                            if x_col == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                y_col = get_column_choice(df)
                                                if y_col == "exit":
                                                    bold_and_large("No action taken.")
                                                else:
                                                    analyisisToolkit.plot_facet_grid(df, x_col, y_col, title=f"Facet Grid: {x_col} vs {y_col}")
                                        elif viz_choice == 24:
                                            column_choice = get_column_choice(df,allow_multiple = True)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.plot_ridge(df, columns=column_choice, title=f"Ridge Plot for {column_choice}") ################################
                                        elif viz_choice == 25:
                                            class_column = input("Enter the column name to use for grouping (e.g., 'Manufacturer'): ").strip()
                                            analyisisToolkit.plot_parallel_coordinates(df, class_column=class_column, title="Parallel Coordinates Plot")
                                        elif viz_choice == 26:
                                            analyisisToolkit.plot_radar(df, title="Radar Chart")
                                        elif viz_choice == 27:
                                            analyisisToolkit.plot_treemap(df, title="Treemap")
                                        elif viz_choice == 28:
                                            x_col = get_column_choice(df)
                                            if x_col == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                y_col = get_column_choice(df)
                                                if y_col == "exit":
                                                    bold_and_large("No action taken.")
                                                else:
                                                    size_col = get_column_choice(df)
                                                    if size_col == "exit":
                                                        bold_and_large("No action taken.")
                                                    else:
                                                        analyisisToolkit.plot_bubble(df, x_col, y_col, size_col, title=f"Bubble Plot: {x_col} vs {y_col}")    ################28 
                                        elif viz_choice == 29:
                                            column_choice = get_column_choice(df)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.plot_wordcloud(df[column_choice], title=f"Word Cloud for {column_choice}")
                                        elif viz_choice == 30:
                                            x_col = get_column_choice(df)
                                            if x_col == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                y_col = get_column_choice(df)
                                                if y_col == "exit":
                                                    bold_and_large("No action taken.")
                                                else:
                                                    z_col = get_column_choice(df)
                                                    if z_col == "exit":
                                                        bold_and_large("No action taken.")
                                                    else:
                                                        analyisisToolkit.plot_3d_scatter(df, x_col, y_col, z_col, title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}")
                                        elif viz_choice == 31:
                                            column_choice = get_column_choice(df)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.plot_time_series_decomposition(df[column_choice], title=f"Time Series Decomposition for {column_choice}")   #####################
                                        elif viz_choice == 32:
                                            column_choice = get_column_choice(df,allow_multiple = True)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.plot_lag(df[column_choice], title=f"Lag Plot for {column_choice}") ############################
                                        elif viz_choice == 33:
                                            column_choice = get_column_choice(df,allow_multiple = True)
                                            if column_choice == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                analyisisToolkit.plot_autocorrelation(df[column_choice], title=f"Autocorrelation Plot for {column_choice}")   #######################
                                        elif viz_choice == 34:
                                            analyisisToolkit.plot_pairgrid(df, title="PairGrid")
                                        elif viz_choice == 35:
                                            x_col = get_column_choice(df)
                                            if x_col == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                y_col = get_column_choice(df)
                                                if y_col == "exit":
                                                    bold_and_large("No action taken.")
                                                else:
                                                    analyisisToolkit.plot_joint(df, x_col, y_col, title=f"Joint Plot: {x_col} vs {y_col}")   ###################################
                                        elif viz_choice == 36:
                                            analyisisToolkit.plot_dendrogram(df, title="Dendrogram")
                                        # Visualization 37: Contour Plot
                                        elif viz_choice == 37:
                                            columns = get_exact_columns(df, 2, "Select exactly 2 columns for the Contour Plot (x, y):")
                                            if columns == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                x_col, y_col = columns
                                                Toolkit.plot_contour(df, x_col, y_col, title=f"Contour Plot: {x_col} vs {y_col}")
                                        
                                        # Visualization 38: Surface Plot
                                        elif viz_choice == 38:
                                            columns = get_exact_columns(df, 3, "Select exactly 3 columns for the Surface Plot (x, y, z):")
                                            if columns == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                x_col, y_col, z_col = columns
                                                Toolkit.plot_surface(df, x_col, y_col, z_col, title=f"Surface Plot: {x_col} vs {y_col} vs {z_col}")
                                        
                                        # Visualization 39: Streamgraph
                                        elif viz_choice == 39:
                                            bold_and_large("Select multiple numeric columns for the streamgraph.")
                                            columns = get_column_choice(df, allow_multiple=True)
                                            if columns == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                # Ensure at least two columns are selected
                                                if len(columns) >= 2:
                                                    Toolkit.plot_streamgraph(df, columns=columns, title=f"Streamgraph for {', '.join(columns)}")
                                                else:
                                                    bold_and_large("Please select at least two numeric columns for the streamgraph.")                                        
                                        # Visualization 40: Network Graph
                                        elif viz_choice == 40:
                                            source_col = input("Enter the source column name: ").strip()
                                            target_col = input("Enter the target column name: ").strip()
                                            Toolkit.plot_network_graph(df, source_col, target_col, title="Network Graph")
                                        
                                        # Visualization 41: 3D Surface Plot
                                        elif viz_choice == 41:
                                            columns = get_exact_columns(df, 3, "Select exactly 3 columns for the 3D Surface Plot (x, y, z):")
                                            print(f"Selected columns: {columns}")  # Debug statement
                                            if columns == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                try:
                                                    x_col, y_col, z_col = columns
                                                    # Validate that the selected columns exist in the DataFrame
                                                    if x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
                                                        print("Error: One or more selected columns do not exist in the DataFrame.")
                                                    else:
                                                        # Validate that the selected columns are numeric
                                                        if not all(df[col].dtype in ['int64', 'float64'] for col in [x_col, y_col, z_col]):
                                                            print("Error: Selected columns must be numeric.")
                                                        else:
                                                            # Call the method on the toolkit instance
                                                            Toolkit.plot_3d_surface(df, x_col, y_col, z_col, title=f"3D Surface Plot: {x_col} vs {y_col} vs {z_col}")
                                                except ValueError:
                                                    print("Error: Exactly 3 columns must be selected.")                                                    
                                                # Visualization 42: 3D Bar Plot
                                        elif viz_choice == 42:
                                            columns = get_exact_columns(df, 3, "Select exactly 3 columns for the 3D Bar Plot (x, y, z):")
                                            if columns == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                x_col, y_col, z_col = columns
                                                Toolkit.plot_3d_bar(df, x_col, y_col, z_col, title=f"3D Bar Plot: {x_col} vs {y_col} vs {z_col}")
                                        
                                        # Visualization 43: 3D Line Plot
                                        elif viz_choice == 43:
                                            columns = get_exact_columns(df, 3, "Select exactly 3 columns for the 3D Line Plot (x, y, z):")
                                            if columns == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                x_col, y_col, z_col = columns
                                                Toolkit.plot_3d_line(df, x_col, y_col, z_col, title=f"3D Line Plot: {x_col} vs {y_col} vs {z_col}")
                                        
                                        # Visualization 44: Interactive Scatter Plot
                                        elif viz_choice == 44:
                                            columns = get_exact_columns(df, 2, "Select exactly 2 columns for the Interactive Scatter Plot (x, y):")
                                            if columns == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                x_col, y_col = columns
                                                Toolkit.plot_interactive_scatter(df, x_col, y_col, title=f"Interactive Scatter Plot: {x_col} vs {y_col}")
                                        
                                        # Visualization 45: Interactive Heatmap
                                        elif viz_choice == 45:
                                            Toolkit.plot_interactive_heatmap(df, title="Interactive Heatmap")
                                        
                                        # Visualization 46: Interactive Line Plot
                                        elif viz_choice == 46:
                                            columns = get_exact_columns(df, 2, "Select exactly 2 columns for the Interactive Line Plot (x, y):")
                                            if columns == "exit":
                                                bold_and_large("No action taken.")
                                            else:
                                                x_col, y_col = columns
                                                Toolkit.plot_interactive_line(df, x_col, y_col, title=f"Interactive Line Plot: {x_col} vs {y_col}")                                        
                                        # Pause to let the user see the visualization
                                        input("\nPress Enter to continue...")
                                    else:
                                        bold_and_large("Invalid visualization number. Please try again.")
                                except ValueError:
                                    bold_and_large("Invalid input. Please enter a number or 'back'.")
                        else:
                            bold_and_large("Invalid category number. Please try again.")
                    except ValueError:
                        bold_and_large("Invalid input. Please enter a number.")
            elif choice == 12:
                report_filename = input("Enter the filename for the report (default: business_data_report.html): ").strip()
                report_filename = report_filename if report_filename else "business_data_report.html"
                try:
                    profile = ProfileReport(df, title="Business Data Preparation Report", explorative=True)
                    profile.to_file(report_filename)
                    bold_and_large(f"Interactive HTML report saved as {report_filename}")
                except Exception as e:
                    bold_and_large(f"An error occurred while generating the HTML report: {e}")
            elif choice == 13:
                bold_and_large("Outlier Handling Techniques:")
                print("1. Remove using IQR")
                print("2. Remove using Z-score")
                print("3. Remove using Percentiles")
                print("4. Cap using Percentiles")
                print("5. Log Transformation")
                print("6. Robust Scaling")
                outlier_choice_map = {
                    1: "Remove using IQR",
                    2: "Remove using Z-score",
                    3: "Remove using Percentiles",
                    4: "Cap using Percentiles",
                    5: "Log Transformation",
                    6: "Robust Scaling"
                }
                outlier_choice = int(input("Choose an outlier handling technique (1-6): ").strip())
                method = outlier_choice_map.get(outlier_choice, None)
                if method is None:
                    bold_and_large("Invalid choice.")
                    return df
    
                column_choices = get_column_choice(df)
                if column_choices == "exit":
                    bold_and_large("No action taken.")
                else:
                    if isinstance(column_choices, list):
                        for col in column_choices:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                if method == "Remove using IQR":
                                    df = remove_outliers_iqr(df, col)
                                elif method == "Remove using Z-score":
                                    df = remove_outliers_zscore(df, col)
                                elif method == "Remove using Percentiles":
                                    df = remove_outliers_percentile(df, col)
                                elif method == "Cap using Percentiles":
                                    df = cap_outliers(df, col)
                                elif method == "Log Transformation":
                                    df = log_transform(df, col)
                                elif method == "Robust Scaling":
                                    df = robust_scaling(df, col)
                                else:
                                    bold_and_large(f"Invalid method for column '{col}'. No action taken.")
                    else:
                        if pd.api.types.is_numeric_dtype(df[column_choices]):
                            if method == "Remove using IQR":
                                df = remove_outliers_iqr(df, column_choices)
                            elif method == "Remove using Z-score":
                                df = remove_outliers_zscore(df, column_choices)
                            elif method == "Remove using Percentiles":
                                df = remove_outliers_percentile(df, column_choices)
                            elif method == "Cap using Percentiles":
                                df = cap_outliers(df, column_choices)
                            elif method == "Log Transformation":
                                df = log_transform(df, column_choices)
                            elif method == "Robust Scaling":
                                df = robust_scaling(df, column_choices)
                            else:
                                bold_and_large(f"Invalid method for column '{column_choices}'. No action taken.")
                    bold_and_large("Outliers in selected columns have been handled.")
                    summary = update_summary(df)
                    display(summary)
            elif choice == 14:
                analyze_correlations(df)
            elif choice == 15:
                remove_unnecessary_columns(df)
                bold_and_large("Unnecessary columns have been removed.")
                summary = update_summary(df)
                display(summary)
            elif choice == 16:
                df = rename_columns_interactive(df)
                bold_and_large("Columns have been renamed.")
                summary = update_summary(df)
                display(summary)
            elif choice == 17:
                group_by(df)
                return df
            else:
                bold_and_large("Invalid choice. Please try again.")
            if choice not in [1, 2, 6, 11, 12, 13, 14, 16]:
                available_options.pop(choice, None)
    
            return df
    
        # Initial setup
        summary = update_summary(df)
        always_available = {
            1: "Display Summary",
            2: "Display DataFrame Head",
            6: "Check and Correct Data Types",
            11: "Visualize Data (Missing Values, Correlation, Outliers, etc.)",
            12: "Generate Interactive HTML Report",
            13: "Handle Outliers with User Choice",
            14: "Analyze Correlations",
            16: "Rename Columns"
        }
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
            12: "Generate Interactive HTML Report",
            13: "Handle Outliers with User Choice",
            14: "Analyze Correlations",
            15: "Remove Unnecessary columns",
            16: "Rename Columns",
            17: "Group By and Aggregate"
        }
        available_options = all_options.copy()
    
        # Menu loop
        display_menu(available_options)
        while True:
            try:
                choice = int(input("Enter your choice: ").strip())
                df = handle_user_choice(choice, df, available_options)
    
                repeat_choice = int(input("Enter your choice (1: Choose again, 2: Redisplay menu, 0: Stop): ").strip())
                if repeat_choice == 1:
                    if not available_options:
                        bold_and_large("All options have been used. Exiting.")
                        break
                elif repeat_choice == 2:
                    display_menu(available_options)
                elif repeat_choice == 0:
                    bold_and_large("Exiting...")
                    break
            except ValueError:
                bold_and_large("Invalid input. Please enter a number.")
    
        # Return the modified DataFrame
        return df