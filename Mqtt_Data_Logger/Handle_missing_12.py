import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os

def load_data(filename, value_columns=None):
    """Load CSV file, handle headers and missing values"""
    df = pd.read_csv(filename, header=0)
    # Some files have empty columns, so we'll clean them up
    df = df.dropna(axis=1, how='all')
    
    # Convert time to numeric in case it's read as string
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    
    # If value_columns is None, try to detect numeric columns (excluding time)
    if value_columns is None:
        value_columns = [col for col in df.columns 
                         if col != 'time' and pd.api.types.is_numeric_dtype(df[col])]
    
    # Convert specified columns to numeric
    for col in value_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, value_columns

def simple_linear_interpolation(df, value_columns):
    """Basic linear interpolation for multiple columns"""
    df_filled = df.copy()
    for col in value_columns:
        df_filled[f'{col}_filled'] = df_filled[col].interpolate(method='linear')
    return df_filled

def spline_interpolation(df, value_columns):
    """Cubic spline interpolation for multiple columns"""
    df_filled = df.copy()
    for col in value_columns:
        # Drop NA values for interpolation
        notna = df_filled[col].notna()
        x = df_filled.loc[notna, 'time']
        y = df_filled.loc[notna, col]
        
        # Create spline function if enough points
        if len(x) > 3:
            spline = interpolate.interp1d(x, y, kind='cubic', fill_value='extrapolate')
            df_filled[f'{col}_filled'] = spline(df_filled['time'])
        else:
            # Fall back to linear if not enough points for spline
            df_filled[f'{col}_filled'] = df_filled[col].interpolate(method='linear')
    
    return df_filled

def moving_average_imputation(df, value_columns, window_size=3):
    """Moving average imputation with specified window size for multiple columns"""
    df_filled = df.copy()
    for col in value_columns:
        # First fill with linear interpolation as base
        filled = df_filled[col].interpolate(method='linear')
        
        # Then apply moving average smoothing
        df_filled[f'{col}_filled'] = filled.rolling(
            window=window_size,
            min_periods=1,
            center=True
        ).mean()
    
    return df_filled

def trend_aware_random_interpolation(
    df, 
    value_columns,
    noise_scale=0.5, 
    model_type='tree', 
    model_params=None,
    window_size=25,
    max_gap_to_interpolate=500,
    weight_power=2
):
    """
    Trend-aware interpolation with weighted contributions from nearest chunks for multiple columns
    """
    df_filled = df.copy()
    
    for col in value_columns:
        # Identify chunks of continuous data (separated by NaN sequences)
        is_valid = df_filled[col].notna()
        chunks = []
        current_chunk = []
        
        for i, (time, val, valid) in enumerate(zip(df_filled['time'], df_filled[col], is_valid)):
            if valid:
                current_chunk.append((time, val))
            elif current_chunk:
                chunks.append(np.array(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(np.array(current_chunk))
        
        # If no valid chunks found, skip this column
        if not chunks:
            continue
        
        # Prepare models for each chunk
        chunk_models = []
        print(f"Found {len(chunks)} chunks of continuous data for column '{col}'.")
        for chunk in chunks:
            x_chunk = chunk[:, 0].reshape(-1, 1)
            y_chunk = chunk[:, 1]
            
            # Create appropriate model
            if model_type == 'linear':
                model = LinearRegression(**model_params) if model_params else LinearRegression()
            elif model_type.startswith('poly'):
                degree = int(model_type[4:])
                model = make_pipeline(
                    PolynomialFeatures(degree=degree),
                    LinearRegression(**model_params) if model_params else LinearRegression()
                )
            elif model_type == 'ridge':
                model = Ridge(**model_params) if model_params else Ridge()
            elif model_type == 'knn':
                model = KNeighborsRegressor(**model_params) if model_params else KNeighborsRegressor()
            elif model_type == 'tree':
                model = DecisionTreeRegressor(**model_params) if model_params else DecisionTreeRegressor()
            elif model_type == 'forest':
                model = RandomForestRegressor(**model_params) if model_params else RandomForestRegressor()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.fit(x_chunk, y_chunk)
            chunk_models.append({
                'start': chunk[0, 0],
                'end': chunk[-1, 0],
                'model': model,
                'x_chunk': x_chunk,
                'y_chunk': y_chunk
            })
        
        # Process each missing value in this column
        missing_mask = df_filled[col].isna()
        missing_times = df_filled.loc[missing_mask, 'time'].values
        
        filled_values = df_filled[col].copy()
        
        for time in missing_times:
            # Skip if gap is too large
            if max_gap_to_interpolate is not None:
                prev_valid = df_filled.loc[(df_filled['time'] < time) & (df_filled[col].notna()), col].last_valid_index()
                next_valid = df_filled.loc[(df_filled['time'] > time) & (df_filled[col].notna()), col].first_valid_index()
                
                if prev_valid is not None and next_valid is not None:
                    gap_size = df_filled.loc[next_valid, 'time'] - df_filled.loc[prev_valid, 'time']
                    if gap_size > max_gap_to_interpolate:
                        continue
            
            # Find all chunks that could influence this point
            influencing_chunks = []
            
            for chunk in chunk_models:
                # Calculate distance to chunk
                if time < chunk['start']:
                    distance = chunk['start'] - time
                elif time > chunk['end']:
                    distance = time - chunk['end']
                else:  # Point is within this chunk
                    distance = 0
                
                influencing_chunks.append({
                    'chunk': chunk,
                    'distance': distance
                })
            
            # Sort by distance and take the two nearest chunks
            influencing_chunks.sort(key=lambda x: x['distance'])
            nearest_chunks = influencing_chunks[:2]
            
            # Calculate weights based on inverse distance
            weights = []
            for chunk_info in nearest_chunks:
                if chunk_info['distance'] == 0:
                    # If point is inside a chunk, give it full weight
                    weights.append(1.0)
                    nearest_chunks = [chunk_info]  # Only use this chunk
                    break
                else:
                    weights.append(1.0 / (chunk_info['distance'] ** weight_power))
            
            # Normalize weights if we're using multiple chunks
            if len(nearest_chunks) > 1:
                total_weight = sum(weights)
                weights = [w/total_weight for w in weights]
            
            # Calculate weighted trend and residuals
            weighted_trend = 0
            weighted_residual = 0
            local_vars = []
            
            for weight, chunk_info in zip(weights, nearest_chunks):
                chunk = chunk_info['chunk']
                model = chunk['model']
                x_chunk = chunk['x_chunk']
                y_chunk = chunk['y_chunk']
                
                # Predict trend value
                trend = model.predict(np.array([[time]]))[0]
                weighted_trend += weight * trend
                
                # Calculate residuals in this chunk
                residuals = y_chunk - model.predict(x_chunk)
                
                # Create spline for residuals if we have enough points
                if len(residuals) > 3:
                    spline = interpolate.interp1d(
                        x_chunk.squeeze(),
                        residuals,
                        kind='cubic',
                        fill_value='extrapolate'
                    )
                    residual = spline(time)
                else:
                    residual = np.mean(residuals)
                
                weighted_residual += weight * residual
                
                # Calculate local variance for this chunk
                chunk_local_vars = []
                for i in range(len(y_chunk)):
                    start = max(0, i - window_size)
                    end = min(len(y_chunk), i + window_size + 1)
                    chunk_local_vars.append(np.var(y_chunk[start:end]))
                
                # Get variance for this time point in this chunk
                if len(chunk_local_vars) > 1:
                    var_spline = interpolate.interp1d(
                        x_chunk.squeeze(),
                        chunk_local_vars,
                        kind='linear',
                        fill_value='extrapolate'
                    )
                    local_var = var_spline(time)
                else:
                    local_var = np.mean(chunk_local_vars) if chunk_local_vars else 0
                
                local_vars.append(local_var)
            
            # Calculate weighted average of local variances
            local_var = sum(w * var for w, var in zip(weights, local_vars))
            
            # Add noise scaled by local variance
            noise = np.random.normal(0, np.sqrt(local_var) * noise_scale)
            
            # Update the value
            filled_values.loc[df_filled['time'] == time] = weighted_trend + weighted_residual + noise
        
        df_filled[f'{col}_filled'] = filled_values
    
    return df_filled

def plot_comparison(original, filled, method_name, filename, value_columns):
    """Plot original vs filled data for comparison for multiple columns"""
    for col in value_columns:
        plt.figure(figsize=(12, 6))
        plt.plot(original['time'], original[col], 'o', label='Original', alpha=0.5)
        plt.plot(filled['time'], filled[f'{col}_filled'], '-', label=f'Filled ({method_name})')
        plt.title(f'Missing Data Imputation: {method_name}\n{filename}\nColumn: {col}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

def process_file(input_file, output_dir, methods, save_results=True, show_plots=True, value_columns=None):
    """Process a single file with multiple methods"""
    print(f"\nProcessing file: {input_file}")
    df, detected_value_columns = load_data(input_file, value_columns)
    
    # Use detected columns if none were specified
    if value_columns is None:
        value_columns = detected_value_columns
    
    print(f"Processing columns: {value_columns}")
    
    # Create output directory if it doesn't exist
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {}
    for name, func in methods.items():
        # Pass the value_columns parameter to each function
        if name == 'Moving Average (window=3)':
            filled_df = func(df, value_columns, window_size=3)
        elif 'Trend-Aware' in name:
            filled_df = func(df, value_columns)
        else:
            filled_df = func(df, value_columns)
            
        results[name] = filled_df
        
        if show_plots:
            plot_comparison(df, filled_df, name, os.path.basename(input_file), value_columns)
        
        if save_results:
            # Create output filename
            base_name = os.path.basename(input_file)
            name_clean = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
            output_file = os.path.join(output_dir, f"{name_clean}_{base_name}")
            filled_df.to_csv(output_file, index=False)
            print(f"Saved {name} results to: {output_file}")
    
    return results

# ===== CONFIGURATION SECTION =====
# Define input files (list all files you want to process)
input_files = [
    # "C:/Users/acer/Documents/Accedemic_Folder_E19254/Agrivoltic_project/Data_set/Image_Data/Pixel_Data_Avg/avg_Plot_02_IN_V3_C1_Pixel_Data.csv",
    # "C:/Users/acer/Documents/Accedemic_Folder_E19254/Agrivoltic_project/Data_set/Image_Data/Pixel_Data_Avg/avg_Plot_02_IN_V4_C1_Pixel_Data.csv",
    # "C:/Users/acer/Documents/Accedemic_Folder_E19254/Agrivoltic_project/Data_set/Image_Data/Pixel_Data_Avg/avg_Plot_03_IN_Pixel_Data.csv",
    # "C:/Users/acer/Documents/Accedemic_Folder_E19254/Agrivoltic_project/Data_set/Image_Data/Pixel_Data_Avg/avg_Plot_04_OUT_Pixel_Data.csv",
    # "C:/Users/acer/Documents/Accedemic_Folder_E19254/Agrivoltic_project/Data_set/Image_Data/Pixel_Data_Avg/avg_Plot_05_IN_Pixel_Data.csv",
    # "C:/Users/acer/Documents/Accedemic_Folder_E19254/Agrivoltic_project/Data_set/Image_Data/Pixel_Data_Avg/avg_Plot_05_OUT_Pixel_Data.csv"

    # "H:/11_08_2025_Thevindu_Weight_Lifting/filtered_mpu6050_data_20250811_115723.csv",
    # "H:/11_08_2025_Thevindu_Weight_Lifting/filtered_mpu6050_data_20250811_114725.csv"

    "C:/Users/acer/Documents/Accedemic_Folder_E19254/Wearable_Hardwear_Project/Walk_1/filtered_mpu6050_data_20250811_135448.xlsx"

]

# Define output directory
# output_directory = "C:/Users/acer/Documents/Accedemic_Folder_E19254/Agrivoltic_project/Data_set/Image_Data/Pixel_Data_Avg"
# output_directory = "H:/11_08_2025_Thevindu_Weight_Lifting"
output_directory = "C:/Users/acer/Documents/Accedemic_Folder_E19254/Wearable_Hardwear_Project/Walk_1/Imputed_mpu6050_data_20250811_135448.xlsx"

# Define the methods to compare
methods = {
    # 'Linear Interpolation': simple_linear_interpolation,
    # 'Cubic Spline': spline_interpolation,
    # 'Moving Average (window=3)': moving_average_imputation,
    'Trend-Aware Random': trend_aware_random_interpolation
}

# Specify which columns to process (None means auto-detect numeric columns)
# VALUE_COLUMNS = ['disease_area_ratio', 'shoots_area_ratio', 'canopy_area_ratio']
# VALUE_COLUMNS = ['disease_pixels_count','disease_area_ratio','shoots_pixels_count','shoots_area_ratio','canopy_pixels_count','canopy_area_ratio']
VALUE_COLUMNS = [
    
    'acc_X_1','acc_Y_1','acc_Z_1','gyro_X_1','gyro_Y_1','gyro_Z_1',
    # 'acc_X_2','acc_Y_2','acc_Z_2','gyro_X_2','gyro_Y_2','gyro_Z_2',
    # 'acc_X_3','acc_Y_3','acc_Z_3','gyro_X_3','gyro_Y_3','gyro_Z_3',
    # 'acc_X_4','acc_Y_4','acc_Z_4','gyro_X_4','gyro_Y_4','gyro_Z_4',
    # 'acc_X_5','acc_Y_5','acc_Z_5','gyro_X_5','gyro_Y_5','gyro_Z_5',
    'acc_X_6','acc_Y_6','acc_Z_6','gyro_X_6','gyro_Y_6','gyro_Z_6',
    'acc_X_7','acc_Y_7','acc_Z_7','gyro_X_7','gyro_Y_7','gyro_Z_7',
    'acc_X_8','acc_Y_8','acc_Z_8','gyro_X_8','gyro_Y_8','gyro_Z_8',
    'acc_X_9','acc_Y_9','acc_Z_9','gyro_X_9','gyro_Y_9','gyro_Z_9',
    'acc_X_10','acc_Y_10','acc_Z_10','gyro_X_10','gyro_Y_10','gyro_Z_10',
    'acc_X_11','acc_Y_11','acc_Z_11','gyro_X_11','gyro_Y_11','gyro_Z_11',
    'acc_X_12','acc_Y_12','acc_Z_12','gyro_X_12','gyro_Y_12','gyro_Z_12',
    # 'acc_X_13','acc_Y_13','acc_Z_13','gyro_X_13','gyro_Y_13','gyro_Z_13',
    'acc_X_14','acc_Y_14','acc_Z_14','gyro_X_14','gyro_Y_14','gyro_Z_14',
    'acc_X_15','acc_Y_15','acc_Z_15','gyro_X_15','gyro_Y_15','gyro_Z_15',
    'acc_X_16','acc_Y_16','acc_Z_16','gyro_X_16','gyro_Y_16','gyro_Z_16',
    # 'acc_X_17','acc_Y_17','acc_Z_17','gyro_X_17','gyro_Y_17','gyro_Z_17',
    # 'acc_X_18','acc_Y_18','acc_Z_18','gyro_X_18','gyro_Y_18','gyro_Z_18',
    # 'acc_X_19','acc_Y_19','acc_Z_19','gyro_X_19','gyro_Y_19','gyro_Z_19',
    # 'acc_X_20','acc_Y_20','acc_Z_20','gyro_X_20','gyro_Y_20','gyro_Z_20',
    # 'acc_X_21','acc_Y_21','acc_Z_21','gyro_X_21','gyro_Y_21','gyro_Z_21',
    # 'acc_X_22','acc_Y_22','acc_Z_22','gyro_X_22','gyro_Y_22','gyro_Z_22',
    # 'acc_X_23','acc_Y_23','acc_Z_23','gyro_X_23','gyro_Y_23','gyro_Z_23',
    # 'acc_X_24','acc_Y_24','acc_Z_24','gyro_X_24','gyro_Y_24','gyro_Z_24',
    # 'acc_X_25','acc_Y_25','acc_Z_25','gyro_X_25','gyro_Y_25','gyro_Z_25'
    ]

# Set these flags based on your preferences
SAVE_RESULTS = True      # Set to False if you don't want to save output files
SHOW_PLOTS = True        # Set to False if you don't want to see the plots

# ===== PROCESS FILES =====
for input_file in input_files:
    if os.path.exists(input_file):
        process_file(
            input_file=input_file,
            output_dir=output_directory,
            methods=methods,
            save_results=SAVE_RESULTS,
            show_plots=SHOW_PLOTS,
            value_columns=VALUE_COLUMNS
        )
    else:
        print(f"File not found: {input_file}")