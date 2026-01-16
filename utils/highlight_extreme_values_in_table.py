import pandas as pd
import numpy as np


def highlight_extreme_values_in_table(df):
    """
        Applies conditional formatting to highlight extreme values in metrics DataFrame.
        Highlights top 2 values in chocolate color for columns where high values are significant,
        and top 2 minimum values in darkturquoise color for columns where low values are significant.
        Works with both original column names and renamed versions (with line breaks).
        """
    # Create a copy of the DataFrame with styling
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Function to map original column names to their renamed versions (only for long names)
    def get_renamed_col_name(original_name):
        if original_name in ['curv_p10', 'curv_p90']:
            # Don't rename short column names
            return original_name
        else:
            # Rename long column names
            return original_name.replace('curv_', 'curv_\n').replace('_mean', '_\nmean').replace('_median', '_\nmedian').replace('_std', '_\nstd').replace('_local_zscore', '_\nlocal_\nzscore').replace('_count_finite', '_\ncount_\nfinite')
    
    # Define columns where higher values are significant (max values to be highlighted)
    max_columns_original = ['ftle', 'ftle_r2', 'amp', 'final_dist', 'hurst', 'curv_count_finite',
                   'path_len', 'max_kappa', 'frac_high_curv', 'anomaly_score']
    
    # Define columns where lower values are significant (min values to be highlighted)
    min_columns_original = ['curv_radius_mean', 'curv_radius_median', 'curv_radius_std', 'curv_radius_local_zscore',
                   'curv_p10', 'curv_p90']
    
    # Map original names to their possible renamed versions
    max_columns = []
    for col in df.columns:
        if col in max_columns_original or col in [get_renamed_col_name(name) for name in max_columns_original]:
            max_columns.append(col)
    
    min_columns = []
    for col in df.columns:
        if col in min_columns_original or col in [get_renamed_col_name(name) for name in min_columns_original]:
            min_columns.append(col)
    
    for col in df.columns:
        if col == 'idx':  # Skip the index column
            continue
            
        if col in max_columns:
            # Highlight top 2 maximum values in chocolate color
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32'] and not df[col].isna().all():
                # Get top 2 values (excluding NaN)
                valid_values = df[col].dropna()
                if len(valid_values) >= 2:
                    top2_values = valid_values.nlargest(2)
                    styles.loc[valid_values.isin(top2_values), col] = 'background-color: #D2691E'  # chocolate
                elif len(valid_values) == 1:
                    top1_idx = valid_values.index[0]
                    styles.loc[top1_idx, col] = 'background-color: #D2691E'  # chocolate
         
        elif col in min_columns:
            # Highlight top 2 minimum values in darkturquoise color
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32'] and not df[col].isna().all():
                # Get top 2 minimum values (excluding NaN)
                valid_values = df[col].dropna()
                if len(valid_values) >= 2:
                    bottom2_values = valid_values.nsmallest(2)
                    styles.loc[valid_values.isin(bottom2_values), col] = 'background-color: #00CED1'  # darkturquoise
                elif len(valid_values) == 1:
                    bottom1_idx = valid_values.index[0]
                    styles.loc[bottom1_idx, col] = 'background-color: #00CED1'  # darkturquoise
    
    return styles