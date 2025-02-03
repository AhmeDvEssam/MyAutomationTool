# Data Analysis Toolkit

A comprehensive Python toolkit designed to simplify data analysis, preprocessing, and visualization tasks. This toolkit integrates a variety of functions to handle data cleaning, outlier detection, business reporting, and advanced visualizations, making it an essential resource for data analysts and data scientists.

---

## **Features**

### 1. **Data Cleaning and Preparation**

- **Missing Value Handling:** Supports mean, median, mode imputation, and KNN imputation.
- **Duplicate Removal:** Identifies and removes duplicate rows.
- **Data Type Correction:** Automatically detects and corrects incorrect data types.
- **Placeholder Replacement:** Converts placeholders like '?', '-', 'None', 'N/A' to NaN.

### 2. **Outlier Detection and Handling**

- **Z-Score Method:** Detects outliers using standard deviation thresholds.
- **IQR Method:** Identifies outliers using the interquartile range.
- **Percentile-Based Capping:** Caps extreme values to specified percentiles.
- **Transformations:** Log and square root transformations to mitigate outlier impact.

### 3. **Data Normalization and Scaling**

- **Min-Max Normalization:** Scales data between 0 and 1.
- **Standardization:** Applies Z-score scaling.
- **Robust Scaling:** Reduces the effect of outliers using IQR-based scaling.

### 4. **Business Reporting**

- **Automated PDF Reports:** Generates business data preparation reports summarizing data overview, key cleaning steps, outlier detection, and visual insights.
- **Summary Statistics:** Provides descriptive statistics, variance, skewness, kurtosis, and data types.

### 5. **Advanced Data Visualizations**

- **General Plots:** Histograms, bar plots, line plots, scatter plots, pie charts.
- **Outlier Visualization:** Boxplots, violin plots, swarm plots.
- **Distribution Analysis:** KDE plots, ridge plots, QQ plots.
- **Correlation Analysis:** Heatmaps, pairplots, scatter matrices.
- **Interactive Visualizations:** Plotly-based heatmaps, scatter plots, and line plots.
- **3D Visualizations:** 3D scatter plots, bar plots, surface plots, and network graphs.
- **Time Series Analysis:** Seasonal decomposition, lag plots, autocorrelation plots.
- **Business Charts:** Treemaps, word clouds, bubble plots.

### 6. **Data Aggregation and Transformation**

- **Group By Operations:** Aggregates data using sum, mean, median, max, min, count, and standard deviation.
- **Filtering and Slicing:** Allows complex filter expressions with logical operators and flexible data slicing.
- **Encoding:** Converts categorical variables using one-hot encoding.

### 7. **Automated Insights**

- **Correlation Insights:** Identifies strong, moderate, and weak correlations with significance levels.
- **Outlier Handling Recommendations:** Suggests methods for handling outliers based on skewness and distribution.
- **Missing Value Recommendations:** Provides filling strategies based on data distribution.

### 8. \*\*Enhanced Data Summary: \*\***`describe_pluss`**

- **Comprehensive Summary:** Extends the basic `describe()` functionality with detailed metrics, including variance, skewness, kurtosis, and data type classifications.
- **Outlier Handling Recommendations:** Suggests the best approach for handling outliers based on skewness and kurtosis. Provides strategies such as capping, removing, or transforming outliers.
- **Missing Value Strategies:** Recommends the optimal method for handling missing values based on data type and distribution, such as mean, median, or mode imputation.
- **Interactive Insights:** Allows users to interactively view summaries, detect data quality issues, and apply fixes directly within the toolkit interface.
- **Dynamic Decision-Making:** Analyzes data distribution to offer recommendations on normalization, scaling, and transformation techniques.

---

## **Technologies Used**

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Seaborn, Matplotlib, Plotly, Scipy, Scikit-learn, Joypy, NetworkX, Missingno, FPDF

---

## **Usage Example**

```python
from analyisisToolkit import analyisisToolkit

# Load your data
import pandas as pd
df = pd.read_csv('your_dataset.csv')

# Initialize toolkit
toolkit = analyisisToolkit()

# Clean data
toolkit.remove_unnecessary_columns(df)
toolkit.correct_data_types(df)

# Handle missing values
toolkit.knn_impute(df)

# Visualize data
toolkit.plot_correlation_heatmap(df)
toolkit.plot_violin_comparison(df, 'Price', 'Category')

# Generate business report
toolkit.generate_business_report(df, 'Business_Report.pdf')

# Enhanced Comprehensive Data Summary
toolkit.describe_pluss(df)
```

---

## **Project Structure**

```
Data-Analysis-Toolkit/
├── analyisisToolkit.py
├── data/
│   └── your_dataset.csv
├── reports/
│   └── Business_Report.pdf
└── README.md
```

---

## **How to Run**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/data-analysis-toolkit.git
   cd data-analysis-toolkit
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run your analysis script:
   ```bash
   python analyisisToolkit.py
   ```

---

## **Contributing**

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your enhancements.

---

## **License**

This project is licensed under the MIT License.

---

## **Connect with Me**

- **LinkedIn:** [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
---

**Transform your data effortlessly with the Data Analysis Toolkit! 🚀**

