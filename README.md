# MyAutomationTool
Automating The Data Wrangling process
Enhanced Data Analysis Toolkit in Python
This project provides a comprehensive Python toolkit for data analysis, focusing on preprocessing, outlier detection, visualization, and data cleaning. It combines advanced statistical methods, machine learning techniques, and insightful visualizations to streamline and enhance the data analysis workflow.

Features
Outlier Detection:

IQR Method: Detects outliers based on the Interquartile Range (IQR).
Z-Score Method: Identifies outliers using the Z-score threshold.
Isolation Forest: A machine learning approach for detecting anomalies.
Data Visualizations:

Boxplots to highlight outliers.
Heatmaps to visualize missing values.
Histograms with skewness indicators.
Correlation heatmaps for numeric columns.
Missing Value Handling:

Provides recommendations for filling techniques based on data skewness:
Mean for symmetric distributions.
Median for skewed distributions.
Mode for categorical data.
Option to apply the recommendations interactively.
Data Type Correction:

Automatically identifies and corrects column data types (e.g., converts string-represented integers to actual integers).
Generalized Analysis Workflow:

Combines descriptive statistics with insights on missing values, skewness, and outliers.
Allows for user interaction to apply corrections and visualize data.
Enhanced describe_plus Function:

Adds skewness and missing value counts to the standard .describe() output.
Integrates recommendations for data cleaning and supports visualizations.
Usage
This toolkit is ideal for:

Preprocessing datasets before modeling.
Gaining insights into data quality.
Identifying and handling outliers and missing values.
How to Use
Load your dataset into a Pandas DataFrame.
Use the describe_plus() function for an interactive analysis.
Choose outlier detection methods (iqr, zscore, isolation_forest, or auto).
Visualize missing values, outliers, and distributions to understand your data better.
Prerequisites
Python 3.x
Libraries: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
