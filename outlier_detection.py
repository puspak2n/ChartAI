import pandas as pd
import numpy as np

def detect_outliers_iqr(df, metric):
    # Calculate Q1, Q3, and IQR
    Q1 = df[metric].quantile(0.25)
    Q3 = df[metric].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    df['Is_Outlier'] = (df[metric] < lower_bound) | (df[metric] > upper_bound)
    outliers = df[df['Is_Outlier']]
    
    return outliers

# Example usage with your dataset
# df = your_dataframe
# outliers = detect_outliers_iqr(df, 'Quantity')
# outlier_counts = outliers.groupby('Region').size().reset_index(name='Outlier_Count')