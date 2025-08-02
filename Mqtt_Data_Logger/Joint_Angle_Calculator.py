import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Constants for scaling factors
ACCEL_SCALE = 1/835.67  # For accelerometer data
GYRO_SCALE = 1/65.5     # For gyroscopic data

def load_and_preprocess_data(filepath):
    """Load IMU data and apply scaling factors"""
    df = pd.read_csv(filepath)
    
    # Apply scaling factors to all accelerometer and gyroscope data
    for col in df.columns:
        if col.startswith('acc_'):
            df[col] = df[col] * ACCEL_SCALE
        elif col.startswith('gyro_'):
            df[col] = df[col] * GYRO_SCALE
    
    return df

def detect_outliers_simple(df, accel_thresh=20.0, gyro_thresh=10.0):
    """
    Simple threshold-based outlier detection
    - accel_thresh: in m/s² (default 20.0)
    - gyro_thresh: in rad/s (default 10.0)
    """
    clean_df = df.copy()
    
    # Find rows where any accelerometer exceeds threshold
    accel_cols = [col for col in df.columns if col.startswith('acc_')]
    accel_outliers = (df[accel_cols].abs() > accel_thresh).any(axis=1)
    
    # Find rows where any gyroscope exceeds threshold
    gyro_cols = [col for col in df.columns if col.startswith('gyro_')]
    gyro_outliers = (df[gyro_cols].abs() > gyro_thresh).any(axis=1)
    
    # Combine outliers
    outlier_rows = accel_outliers | gyro_outliers
    
    # Replace outlier rows with NaN (keeping timestamp and time_s)
    numerical_cols = [col for col in df.columns if col not in ['timestamp', 'time_s']]
    clean_df.loc[outlier_rows, numerical_cols] = np.nan
    
    return clean_df

def interpolate_missing_values(df):
    """Interpolate missing values with linear interpolation"""
    numerical_cols = [col for col in df.columns if col not in ['timestamp', 'time_s']]
    df[numerical_cols] = df[numerical_cols].interpolate(method='linear', limit_direction='both')
    return df

def lowpass_filter(data, cutoff=5.0, fs=100.0, order=4):
    """
    Apply lowpass filter to data with adjustable cutoff
    - cutoff: cutoff frequency in Hz (default 5.0)
    - fs: sampling frequency in Hz (default 100.0)
    - order: filter order (default 4)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def apply_filters(df, cutoff_freq=5.0):
    """Apply filters to all sensor data with adjustable cutoff"""
    filtered_df = df.copy()
    numerical_cols = [col for col in df.columns if col not in ['timestamp', 'time_s']]
    
    for col in numerical_cols:
        filtered_df[col] = lowpass_filter(df[col].values, cutoff=cutoff_freq)
    
    return filtered_df

def get_sensor_orientation(df, sensor_num):
    """Get orientation quaternions for a specific sensor"""
    w = df[f'w_{sensor_num}'].values
    x = df[f'x_{sensor_num}'].values
    y = df[f'y_{sensor_num}'].values
    z = df[f'z_{sensor_num}'].values
    
    # Normalize quaternions
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w_norm = w / norm
    x_norm = x / norm
    y_norm = y / norm
    z_norm = z / norm
    
    return np.column_stack([w_norm, x_norm, y_norm, z_norm])

def get_downward_vector(quaternions):
    """Calculate downward pointing vector from quaternions"""
    # Initial downward vector in global frame (assuming z-up)
    global_down = np.array([0, 0, -1])
    
    # Rotate this vector by each quaternion
    downward_vectors = []
    rot = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])  # scipy uses xyzw order
    
    for r in rot:
        # Rotate the global down vector to sensor frame
        sensor_down = r.apply(global_down)
        downward_vectors.append(sensor_down)
    
    return np.array(downward_vectors)

def calculate_joint_angle(down_vec1, down_vec2):
    """Calculate angle between two downward vectors in degrees"""
    dot_product = np.sum(down_vec1 * down_vec2, axis=1)
    norm_product = np.linalg.norm(down_vec1, axis=1) * np.linalg.norm(down_vec2, axis=1)
    angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return np.degrees(angle_rad)

def process_imu_data(filepath, joint_pairs, cutoff_freq=5.0, accel_thresh=20.0, gyro_thresh=10.0):
    """
    Main function to process IMU data and calculate joint angles
    
    Parameters:
    - filepath: Path to the CSV file
    - joint_pairs: Dictionary of joint names and sensor pairs
    - cutoff_freq: Low-pass filter cutoff frequency in Hz (default 5.0)
    - accel_thresh: Accelerometer outlier threshold in m/s² (default 20.0)
    - gyro_thresh: Gyroscope outlier threshold in rad/s (default 10.0)
    
    Returns:
    - Dictionary with joint angles over time
    """
    # Load and preprocess data
    df = load_and_preprocess_data(filepath)
    df = detect_outliers_simple(df, accel_thresh=accel_thresh, gyro_thresh=gyro_thresh)
    df = interpolate_missing_values(df)
    df = apply_filters(df, cutoff_freq=cutoff_freq)
    
    # Initialize results dictionary
    results = {'time': df['time_s'].values}
    
    # Calculate joint angles for each specified pair
    for joint_name, (sensor1, sensor2) in joint_pairs.items():
        # Get quaternions for both sensors
        quat1 = get_sensor_orientation(df, sensor1)
        quat2 = get_sensor_orientation(df, sensor2)
        
        # Calculate downward vectors
        down_vec1 = get_downward_vector(quat1)
        down_vec2 = get_downward_vector(quat2)
        
        # Calculate joint angle
        joint_angle = calculate_joint_angle(down_vec1, down_vec2)
        results[joint_name] = joint_angle
    
    return results

def plot_joint_angles(results, joints_to_plot=None):
    """Plot the calculated joint angles"""
    time = results['time']
    
    if joints_to_plot is None:
        joints_to_plot = [key for key in results.keys() if key != 'time']
    
    plt.figure(figsize=(12, 6))
    
    for joint in joints_to_plot:
        plt.plot(time, results[joint], label=joint)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle (degrees)')
    plt.title('Joint Angle Variation Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Specify which sensor pairs correspond to which joints
    # Format: {'joint_name': (sensor1_num, sensor2_num)}
    joint_pairs = {
        'left_hip': (1, 7),
        'right_hip': (6, 8),
        'left_knee': (9, 11),
        'right_knee': (10, 12),
        'left_ankle': (14, 16),
        'right_ankle': (15, 17)
    }
    
    # Process the data with adjustable parameters
    results = process_imu_data(
        "G:/Tharinda_Readings/Tharinda_Walking_Pattern_01_20250720_084136_Processed.csv",
        joint_pairs,
        cutoff_freq=1000.0,    # Adjust cutoff frequency as needed
        accel_thresh=15.0,  # Adjust accelerometer threshold (m/s²)
        gyro_thresh=100.0    # Adjust gyroscope threshold (rad/s)
    )
    
    # Plot selected joints
    plot_joint_angles(results, joints_to_plot=['right_knee', 'right_hip'])
    
    # Access results directly:
    # results['time'] - time values
    # results['right_knee'] - knee joint angles
    # results['right_hip'] - hip joint angles
    # etc.