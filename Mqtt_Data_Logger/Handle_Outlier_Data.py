import pandas as pd
import numpy as np
import os

def filter_outliers_absolute(
    input_file,
    output_folder,
    acc_max_threshold= 15.0 * 835.067,  # Default max threshold for accelerometer (adjust as needed)
    gyro_max_threshold= 150.0 * 65.5, # Default max threshold for gyroscope (adjust as needed)
):
    """
    Filters outliers by replacing values exceeding given thresholds with NaN.
    
    Parameters:
        input_file (str): Path to input CSV file.
        output_folder (str): Path to output folder.
        acc_max_threshold (float): Maximum allowed absolute value for accelerometer data.
        gyro_max_threshold (float): Maximum allowed absolute value for gyroscope data.
    """
    
    # Read the input file
    df = pd.read_excel(input_file)
    
    # Select only accelerometer and gyroscope columns (ignore quaternion data)
    acc_cols = [col for col in df.columns if col.startswith('acc_')]
    gyro_cols = [col for col in df.columns if col.startswith('gyro_')]
    
    # Replace accelerometer outliers (values beyond ±acc_max_threshold)
    for col in acc_cols:
        df[col] = np.where(
            (df[col].abs() > acc_max_threshold),
            np.nan,
            df[col]
        )
    
    # Replace gyroscope outliers (values beyond ±gyro_max_threshold)
    for col in gyro_cols:
        df[col] = np.where(
            (df[col].abs() > gyro_max_threshold),
            np.nan,
            df[col]
        )
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the cleaned data
    input_filename = os.path.basename(input_file)
    output_path = os.path.join(output_folder, f"filtered_{input_filename}")
    df.to_csv(output_path, index=False)
    
    print(f"Filtered data saved to: {output_path}")

# Example usage (modify these parameters as needed)
input_file = "C:/Users/acer/Documents/Accedemic_Folder_E19254/Wearable_Hardwear_Project/Walk_1/mpu6050_data_20250811_135448.xlsx"  # Change this to your input file path
output_folder = "C:/Users/acer/Documents/Accedemic_Folder_E19254/Wearable_Hardwear_Project/Walk_1/"    # Change this to your desired output folder
acc_max_threshold = 15.0 * 835.067  # Adjust based on expected accelerometer range (e.g., ±20 m/s²)
gyro_max_threshold = 15.0 * 65.5  # Adjust based on expected gyroscope range (e.g., ±10 rad/s)

filter_outliers_absolute(input_file, output_folder, acc_max_threshold, gyro_max_threshold)