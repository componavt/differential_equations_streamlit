import pandas as pd
import numpy as np


def highlight_extreme_values_in_table(df):
    """
        Applies conditional formatting to highlight extreme values in metrics DataFrame.
        Highlights top 2 values in yellow for columns where high values are significant,
        and top 2 minimum values in light blue for columns where low values are significant.
        """
    # Create a copy of the DataFrame with styling
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Define columns where higher values are significant (max values to be highlighted)
    max_columns = ['ftle', 'ftle_r2', 'amp', 'final_dist', 'hurst', 'curv_count_finite', 
                   'path_len', 'max_kappa', 'frac_high_curv', 'anomaly_score']
    
    # Define columns where lower values are significant (min values to be highlighted)
    min_columns = ['curv_radius_mean', 'curv_radius_median', 'curv_radius_std', 'curv_radius_local_zscore',
                   'curv_p10', 'curv_p90']
    
    for col in df.columns:
        if col == 'idx':  # Skip the index column
            continue
            
        if col in max_columns:
            # Highlight top 2 maximum values in yellow
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32'] and not df[col].isna().all():
                # Get top 2 values (excluding NaN)
                valid_values = df[col].dropna()
                if len(valid_values) >= 2:
                    top2_values = valid_values.nlargest(2)
                    styles.loc[valid_values.isin(top2_values), col] = 'background-color: yellow'
                elif len(valid_values) == 1:
                    top1_idx = valid_values.index[0]
                    styles.loc[top1_idx, col] = 'background-color: yellow'
        
        elif col in min_columns:
            # Highlight top 2 minimum values in light blue
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32'] and not df[col].isna().all():
                # Get top 2 minimum values (excluding NaN)
                valid_values = df[col].dropna()
                if len(valid_values) >= 2:
                    bottom2_values = valid_values.nsmallest(2)
                    styles.loc[valid_values.isin(bottom2_values), col] = 'background-color: lightblue'
                elif len(valid_values) == 1:
                    bottom1_idx = valid_values.index[0]
                    styles.loc[bottom1_idx, col] = 'background-color: lightblue'
    
    return styles