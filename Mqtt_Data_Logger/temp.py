import pandas as pd
import numpy as np

def process_sensor_data(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Define conversion factors
    ACCEL_SCALE = 8192  # for ±4g range (LSM6DSO)
    GYRO_SCALE = 65.5   # for ±500dps range (LSM6DSO)
    
    # Identify accelerometer and gyroscope columns
    accel_cols = [col for col in df.columns if col.startswith('acc_')]
    gyro_cols = [col for col in df.columns if col.startswith('gyro_')]
    
    # Convert accelerometer data to m/s²
    df[accel_cols] = df[accel_cols] / ACCEL_SCALE * 9.80665  # Convert to m/s²
    
    # Convert gyroscope data to degrees/second
    df[gyro_cols] = df[gyro_cols] / GYRO_SCALE
    
    # Remove outliers (values that are physically impossible)
    # For accelerometer: reasonable range ±20 m/s² (allowing for some margin)
    # For gyroscope: reasonable range ±2000 dps (allowing for some margin)
    accel_mask = (df[accel_cols].abs() > 15).any(axis=1)
    gyro_mask = (df[gyro_cols].abs() > 100).any(axis=1)
    
    # Combine masks and filter
    outlier_mask = accel_mask | gyro_mask
    df_clean = df[~outlier_mask].copy()
    
    # Report how many rows were removed
    print(f"Removed {outlier_mask.sum()} rows with outlier values")
    
    return df_clean

# Example usage
if __name__ == "__main__":
    input_file = "H:/Tharinda_Readings/Tharinda_Walking_Pattern_01_20250720_084136.xlsx"
    processed_data = process_sensor_data(input_file)
    
    # Save the processed data to a new file
    output_file = input_file.split(".")[0] + "processed.xlsx"
    processed_data.to_excel(output_file, index=False)
    print(f"Processed data saved to {output_file}")