import pandas as pd
import numpy as np


def highlight_extreme_values_in_table(df):
    """
        Applies conditional formatting to highlight extreme values in metrics DataFrame.
        Highlights top 2 values in chocolate color for columns where high values are significant,
        and top 2 minimum values in darkturquoise color for columns where low values are significant.
        Works with both original and shortened column names.
        """
    # Create a copy of the DataFrame with styling
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Define possible column names for max values (both original and shortened)
    max_columns_possible = [
        ['ftle'], ['ftle_r2'], ['amp'], ['final_dist'], ['hurst'],
        ['curv_count_finite', 'curv_ct_fin'],  # Both original and shortened
        ['path_len'], ['max_kappa'], ['frac_high_curv'], ['anomaly_score']
    ]
    
    # Define possible column names for min values (both original and shortened)
    min_columns_possible = [
        ['curv_radius_mean', 'curv_rad_mn'],  # Both original and shortened
        ['curv_radius_median', 'curv_rad_med'],  # Both original and shortened
        ['curv_radius_std', 'curv_rad_std'],  # Both original and shortened
        ['curv_radius_local_zscore', 'curv_rad_lcl_z'],  # Both original and shortened
        ['curv_p10'], ['curv_p90']
    ]
    
    # Identify actual columns in the DataFrame
    max_columns = []
    for col_group in max_columns_possible:
        for col in col_group:
            if col in df.columns:
                max_columns.append(col)
                break  # Only add the first matching column from the group
    
    min_columns = []
    for col_group in min_columns_possible:
        for col in col_group:
            if col in df.columns:
                min_columns.append(col)
                break  # Only add the first matching column from the group
    
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