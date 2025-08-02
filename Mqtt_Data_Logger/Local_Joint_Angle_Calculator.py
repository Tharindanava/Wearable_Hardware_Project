import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import os
import matplotlib.pyplot as plt

class IMUJointAngleCalculator:
    def __init__(self, file_path):
        """
        Initialize the IMU joint angle calculator with data from Excel file.
        
        Args:
            file_path (str): Path to the Excel file containing IMU data
        """
        # Load data from Excel
        self.df = pd.read_excel(file_path)
        
        # Extract timestamp and time columns
        self.timestamps = self.df['timestamp'].values
        self.time_s = self.df['time_s'].values
        
        # Get all unique sensor numbers from column names
        sensor_numbers = set()
        for col in self.df.columns:
            if col.startswith('acc_X_'):
                sensor_numbers.add(int(col.split('_')[-1]))
        self.sensor_numbers = sorted(sensor_numbers)
        
        print(f"Found {len(self.sensor_numbers)} sensors: {self.sensor_numbers}")
    
    def get_sensor_data(self, sensor_num):
        """
        Extract data for a specific sensor.
        
        Args:
            sensor_num (int): Sensor number to extract data for
            
        Returns:
            dict: Dictionary containing acc, gyro, and quaternion data for the sensor
        """
        if sensor_num not in self.sensor_numbers:
            raise ValueError(f"Sensor {sensor_num} not found in data")
        
        # Define outlier thresholds (customize as needed)
        acc_threshold = 10.0  # m/s², example threshold for accelerometer
        gyro_threshold = 100.0  # rad/s, example threshold for gyroscope

        # Identify outlier rows for accelerometer (any axis)
        acc_cols = [f'acc_X_{sensor_num}', f'acc_Y_{sensor_num}', f'acc_Z_{sensor_num}']
        acc_outlier = (self.df[acc_cols].abs() > acc_threshold * 835.067).any(axis=1)

        # Identify outlier rows for gyroscope (any axis)
        gyro_cols = [f'gyro_X_{sensor_num}', f'gyro_Y_{sensor_num}', f'gyro_Z_{sensor_num}']
        gyro_outlier = (np.abs(np.deg2rad(self.df[gyro_cols])) > gyro_threshold).any(axis=1)

        # Combine outlier masks (remove row if any sensor has an outlier)
        outlier_mask = acc_outlier | gyro_outlier

        # Filter out outlier rows from the entire dataframe
        df_clean = self.df.loc[~outlier_mask].reset_index(drop=True)
            
        # Extract accelerometer data (in m/s², assuming original data is in milli-g)
        acc_x = df_clean[f'acc_X_{sensor_num}'].values / 835.067
        acc_y = df_clean[f'acc_Y_{sensor_num}'].values / 835.067
        acc_z = df_clean[f'acc_Z_{sensor_num}'].values / 835.067
        
        # Extract gyroscope data (convert to rad/s, assuming original data is in deg/s)
        gyro_x = np.deg2rad(df_clean[f'gyro_X_{sensor_num}'].values / 65.5)
        gyro_y = np.deg2rad(df_clean[f'gyro_Y_{sensor_num}'].values / 65.5)
        gyro_z = np.deg2rad(df_clean[f'gyro_Z_{sensor_num}'].values / 65.5)
        
        # Extract quaternion data if available
        if f'w_{sensor_num}' in self.df.columns:
            w = df_clean[f'w_{sensor_num}'].values
            x = df_clean[f'x_{sensor_num}'].values
            y = df_clean[f'y_{sensor_num}'].values
            z = df_clean[f'z_{sensor_num}'].values
            quats = np.column_stack((w, x, y, z))
        else:
            quats = None
        
        return {
            'acc': np.column_stack((acc_x, acc_y, acc_z)),
            'gyro': np.column_stack((gyro_x, gyro_y, gyro_z)),
            'quat': quats
        }
    
    def calculate_orientation_relative(self, sensor_num, calibration_period=1.0, alpha=0.98):
        """
        Calculate orientation relative to initial orientation using complementary filter.
        
        Args:
            sensor_num (int): Sensor number
            calibration_period (float): Time in seconds for initial calibration
            alpha (float): Complementary filter weighting (0-1)
            
        Returns:
            np.array: Array of quaternions representing orientation over time relative to initial orientation
        """
        data = self.get_sensor_data(sensor_num)
        acc = data['acc']
        gyro = data['gyro']
        
        # If quaternions are already in the data, use them directly
        if data['quat'] is not None:
            initial_quat = data['quat'][0]
            # Calculate relative to initial orientation
            rot_initial = Rotation.from_quat(initial_quat).inv()
            relative_quats = np.zeros_like(data['quat'])
            for i, q in enumerate(data['quat']):
                relative_quats[i] = (Rotation.from_quat(q) * rot_initial).as_quat()
            return relative_quats
        
        num_samples = len(acc)
        dt = np.mean(np.diff(self.time_s))
        
        # Initialize orientation
        quaternions = np.zeros((num_samples, 4))
        quaternions[0] = [1, 0, 0, 0]  # Initial orientation (no rotation)
        
        # Calculate initial orientation from accelerometer (just for first sample)
        acc_norm = acc[0] / np.linalg.norm(acc[0])
        pitch = np.arcsin(acc_norm[0])
        roll = np.arctan2(-acc_norm[1], -acc_norm[2])
        initial_rot = Rotation.from_euler('xyz', [roll, pitch, 0])
        quaternions[0] = initial_rot.as_quat()
        
        # Store initial orientation as reference
        initial_quat = quaternions[0]
        rot_initial = Rotation.from_quat(initial_quat).inv()
        
        # Find calibration period samples
        calib_samples = int(calibration_period / dt)
        if calib_samples >= num_samples:
            calib_samples = num_samples - 1
        
        # Calculate gyro bias during calibration period
        gyro_bias = np.mean(gyro[:calib_samples], axis=0)
        gyro_calibrated = gyro - gyro_bias
        
        # Complementary filter
        for i in range(1, num_samples):
            # Gyro integration
            gyro_quat = Rotation.from_rotvec(gyro_calibrated[i] * dt)
            gyro_orientation = Rotation.from_quat(quaternions[i-1]) * gyro_quat
            
            # Accelerometer measurement
            if np.linalg.norm(acc[i]) > 0:
                acc_norm = acc[i] / np.linalg.norm(acc[i])
                pitch = np.arcsin(acc_norm[0])
                roll = np.arctan2(-acc_norm[1], -acc_norm[2])
                acc_orientation = Rotation.from_euler('xyz', [roll, pitch, 0])
                
                # Complementary filter blend
                orientation = Rotation.slerp(
                    gyro_orientation, acc_orientation, 1-alpha
                )
                quaternions[i] = orientation.as_quat()
            else:
                quaternions[i] = gyro_orientation.as_quat()
        
        # Convert to relative to initial orientation
        relative_quats = np.zeros_like(quaternions)
        for i, q in enumerate(quaternions):
            relative_quats[i] = (Rotation.from_quat(q) * rot_initial).as_quat()
        
        return relative_quats
    
    def calculate_joint_angle_relative(self, sensor1, sensor2, calibration_period=1.0, 
                                     alpha=0.98, angle_type='relative'):
        """
        Calculate joint angle between two sensors relative to their initial orientations.
        
        Args:
            sensor1 (int): First sensor number
            sensor2 (int): Second sensor number
            calibration_period (float): Time in seconds for initial calibration
            alpha (float): Complementary filter weighting (0-1)
            angle_type (str): 'relative' for relative angle, 'euler' for Euler angles
            
        Returns:
            dict: Dictionary containing angle data and time vector
        """
        # Get orientations for both sensors relative to their initial orientations
        quat1 = self.calculate_orientation_relative(sensor1, calibration_period, alpha)
        quat2 = self.calculate_orientation_relative(sensor2, calibration_period, alpha)
        
        # Find the common time range where both sensors have data
        min_len = min(len(quat1), len(quat2))
        quat1 = quat1[:min_len]
        quat2 = quat2[:min_len]
        common_time = self.time_s[:min_len]
        
        # Calculate relative rotation between sensors
        rot1 = Rotation.from_quat(quat1)
        rot2 = Rotation.from_quat(quat2)
        relative_rot = rot2 * rot1.inv()
        
        if angle_type == 'relative':
            # Calculate relative angle (angle of rotation)
            angles = relative_rot.magnitude()  # Returns angle in radians
            angles_deg = np.rad2deg(angles)
            return {
                'time': common_time,
                'angle_deg': angles_deg,
                'angle_rad': angles,
                'type': 'relative_angle'
            }
        elif angle_type == 'euler':
            # Calculate Euler angles (XYZ order)
            euler_angles = relative_rot.as_euler('xyz', degrees=True)
            return {
                'time': common_time,
                'x_deg': euler_angles[:, 0],
                'y_deg': euler_angles[:, 1],
                'z_deg': euler_angles[:, 2],
                'type': 'euler_angles'
            }
        else:
            raise ValueError("angle_type must be 'relative' or 'euler'")
    
    def calculate_joint_angle_3d_relative(self, sensor1, sensor2, calibration_period=1.0, alpha=0.98):
        """
        Calculate 3D joint angles (roll, pitch, yaw) between two sensors relative to their initial orientations.
        
        Args:
            sensor1 (int): First sensor number
            sensor2 (int): Second sensor number
            calibration_period (float): Time in seconds for initial calibration
            alpha (float): Complementary filter weighting (0-1)
            
        Returns:
            dict: Dictionary containing 3D angle data and time vector
        """
        # Get sensor data
        data1 = self.get_sensor_data(sensor1)
        data2 = self.get_sensor_data(sensor2)
        
        # Find common length (since outlier removal might have different lengths)
        min_len = min(len(data1['acc']), len(data2['acc']))
        acc1 = data1['acc'][:min_len]
        gyro1 = data1['gyro'][:min_len]
        acc2 = data2['acc'][:min_len]
        gyro2 = data2['gyro'][:min_len]
        time = self.time_s[:min_len]
        
        # Calculate time step
        dt = np.mean(np.diff(time))
        
        # Initialize orientation angles for both sensors (roll, pitch, yaw)
        angles1 = np.zeros((min_len, 3))  # [roll, pitch, yaw]
        angles2 = np.zeros((min_len, 3))
        
        # Calculate gyro bias during calibration period
        calib_samples = int(calibration_period / dt)
        if calib_samples >= min_len:
            calib_samples = min_len - 1
        
        gyro1_bias = np.mean(gyro1[:calib_samples], axis=0)
        gyro2_bias = np.mean(gyro2[:calib_samples], axis=0)
        
        gyro1_calib = gyro1 - gyro1_bias
        gyro2_calib = gyro2 - gyro2_bias
        
        # Initial orientation from accelerometer (roll and pitch only - yaw requires magnetometer)
        if np.linalg.norm(acc1[0]) > 0 and np.linalg.norm(acc2[0]) > 0:
            # Sensor 1
            acc1_norm = acc1[0] / np.linalg.norm(acc1[0])
            angles1[0, 0] = np.arctan2(acc1_norm[1], acc1_norm[2])  # Roll
            angles1[0, 1] = np.arctan2(-acc1_norm[0], np.sqrt(acc1_norm[1]**2 + acc1_norm[2]**2))  # Pitch
            
            # Sensor 2
            acc2_norm = acc2[0] / np.linalg.norm(acc2[0])
            angles2[0, 0] = np.arctan2(acc2_norm[1], acc2_norm[2])  # Roll
            angles2[0, 1] = np.arctan2(-acc2_norm[0], np.sqrt(acc2_norm[1]**2 + acc2_norm[2]**2))  # Pitch
        
        # Store initial angles as reference
        initial_angles1 = angles1[0].copy()
        initial_angles2 = angles2[0].copy()
        
        # Complementary filter for each sensor
        for i in range(1, min_len):
            # Sensor 1
            if np.linalg.norm(acc1[i]) > 0:
                acc1_norm = acc1[i] / np.linalg.norm(acc1[i])
                acc_roll = np.arctan2(acc1_norm[1], acc1_norm[2])
                acc_pitch = np.arctan2(-acc1_norm[0], np.sqrt(acc1_norm[1]**2 + acc1_norm[2]**2))
                
                # Gyro integration
                gyro_roll = angles1[i-1, 0] + gyro1_calib[i, 0] * dt
                gyro_pitch = angles1[i-1, 1] + gyro1_calib[i, 1] * dt
                gyro_yaw = angles1[i-1, 2] + gyro1_calib[i, 2] * dt
                
                # Complementary filter
                angles1[i, 0] = alpha * gyro_roll + (1-alpha) * acc_roll
                angles1[i, 1] = alpha * gyro_pitch + (1-alpha) * acc_pitch
                angles1[i, 2] = gyro_yaw  # Yaw from gyro only (no accel component)
            else:
                # Only gyro data available
                angles1[i, 0] = angles1[i-1, 0] + gyro1_calib[i, 0] * dt
                angles1[i, 1] = angles1[i-1, 1] + gyro1_calib[i, 1] * dt
                angles1[i, 2] = angles1[i-1, 2] + gyro1_calib[i, 2] * dt
            
            # Sensor 2 (same process)
            if np.linalg.norm(acc2[i]) > 0:
                acc2_norm = acc2[i] / np.linalg.norm(acc2[i])
                acc_roll = np.arctan2(acc2_norm[1], acc2_norm[2])
                acc_pitch = np.arctan2(-acc2_norm[0], np.sqrt(acc2_norm[1]**2 + acc2_norm[2]**2))
                
                # Gyro integration
                gyro_roll = angles2[i-1, 0] + gyro2_calib[i, 0] * dt
                gyro_pitch = angles2[i-1, 1] + gyro2_calib[i, 1] * dt
                gyro_yaw = angles2[i-1, 2] + gyro2_calib[i, 2] * dt
                
                # Complementary filter
                angles2[i, 0] = alpha * gyro_roll + (1-alpha) * acc_roll
                angles2[i, 1] = alpha * gyro_pitch + (1-alpha) * acc_pitch
                angles2[i, 2] = gyro_yaw  # Yaw from gyro only
            else:
                # Only gyro data available
                angles2[i, 0] = angles2[i-1, 0] + gyro2_calib[i, 0] * dt
                angles2[i, 1] = angles2[i-1, 1] + gyro2_calib[i, 1] * dt
                angles2[i, 2] = angles2[i-1, 2] + gyro2_calib[i, 2] * dt
        
        # Calculate angles relative to initial orientation
        angles1_relative = angles1 - initial_angles1
        angles2_relative = angles2 - initial_angles2
        
        # Calculate relative joint angles
        joint_angles_rad = angles2_relative - angles1_relative
        joint_angles_deg = np.rad2deg(joint_angles_rad)
        
        return {
            'time': time,
            'roll_deg': joint_angles_deg[:, 0],
            'pitch_deg': joint_angles_deg[:, 1],
            'yaw_deg': joint_angles_deg[:, 2],
            'roll_rad': joint_angles_rad[:, 0],
            'pitch_rad': joint_angles_rad[:, 1],
            'yaw_rad': joint_angles_rad[:, 2],
            'type': '3d_angles_relative'
        }
    
    def plot_joint_angle(self, angle_data, angle_data_3d, title="Joint Angle"):
        """
        Plot the joint angle results.
        
        Args:
            angle_data (dict): Result from calculate_joint_angle_relative
            angle_data_3d (dict): Result from calculate_joint_angle_3d_relative
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        
        if angle_data['type'] == 'relative_angle':
            plt.plot(angle_data['time'], angle_data['angle_deg'])
            plt.ylabel('Angle (degrees)')
            plt.title(f'{title} - Relative Angle')
        elif angle_data['type'] == 'euler_angles':
            plt.plot(angle_data['time'], angle_data['x_deg'], label='X (Roll)')
            plt.plot(angle_data['time'], angle_data['y_deg'], label='Y (Pitch)')
            plt.plot(angle_data['time'], angle_data['z_deg'], label='Z (Yaw)')
            plt.ylabel('Angle (degrees)')
            plt.legend()
            plt.title(f'{title} - Euler Angles')
        
        plt.xlabel('Time (s)')
        plt.grid(True)
        plt.show()

        # Plot the 3D results
        plt.figure(figsize=(12, 8))
        plt.plot(angle_data_3d['time'], angle_data_3d['roll_deg'], label='Roll')
        plt.plot(angle_data_3d['time'], angle_data_3d['pitch_deg'], label='Pitch')
        plt.plot(angle_data_3d['time'], angle_data_3d['yaw_deg'], label='Yaw')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title(f'{title} - 3D Relative Angles')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_joint_angles_to_csv(self, angle_data, angle_data_3d, file_path, joint_name="joint"):
        """
        Save joint angle data to a CSV file with customizable column headings.
        If file exists, appends new columns while preserving existing data.
        
        Args:
            angle_data (dict): Result from calculate_joint_angle_relative
            angle_data_3d (dict): Result from calculate_joint_angle_3d_relative
            file_path (str): Path to save the CSV file
            joint_name (str): Base name for the joint (e.g., "knee")
        """
        # Create new DataFrame with time and angle columns
        new_data = {'time': angle_data['time']}
        
        # Add relative angle if available
        if angle_data['type'] == 'relative_angle':
            new_data[f'{joint_name}_relative'] = angle_data['angle_deg']
        elif angle_data['type'] == 'euler_angles':
            new_data[f'{joint_name}_x'] = angle_data['x_deg']
            new_data[f'{joint_name}_y'] = angle_data['y_deg']
            new_data[f'{joint_name}_z'] = angle_data['z_deg']
        
        # Add 3D angles
        new_data[f'{joint_name}_roll'] = angle_data_3d['roll_deg']
        new_data[f'{joint_name}_pitch'] = angle_data_3d['pitch_deg']
        new_data[f'{joint_name}_yaw'] = angle_data_3d['yaw_deg']
        
        new_df = pd.DataFrame(new_data)
        
        # If file exists, merge with existing data
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            
            # Ensure time columns match
            if not existing_df['time'].equals(new_df['time']):
                # If time columns don't match, do an outer merge
                merged_df = pd.merge(existing_df, new_df, on='time', how='outer')
            else:
                # If time columns match, just concatenate horizontally
                merged_df = pd.concat([existing_df, new_df.drop(columns=['time'])], axis=1)
        else:
            merged_df = new_df
        
        # Save to CSV
        merged_df.to_csv(file_path, index=False)
        print(f"Joint angle data saved to {file_path}")

# Example usage
if __name__ == "__main__":
    # Initialize with your data file
    calculator = IMUJointAngleCalculator("G:/Tharinda_Readings/Tharinda_Walking_Pattern_01_20250720_084136_Processed.xlsx")
    
    # Define which sensors to use (these are example numbers - adjust based on your data)
    sensor1 = 15 
    sensor2 = 17  
    
    # Calculate joint angle relative to initial orientation
    angle_data = calculator.calculate_joint_angle_relative(
        sensor1, sensor2, 
        calibration_period=5.0, 
        alpha=0.98,
        angle_type='relative'
    )

    # Calculate 3D joint angles relative to initial orientation
    angle_data_3d = calculator.calculate_joint_angle_3d_relative(
        sensor1, sensor2,
        calibration_period=1.0,
        alpha=0.98
    )
    
    # Plot the results
    calculator.plot_joint_angle(angle_data, angle_data_3d, title=f"Joint Angle between Sensor {sensor1} and {sensor2} (Relative to Initial)")

    # Save to CSV with custom joint name
    calculator.save_joint_angles_to_csv(
        angle_data,
        angle_data_3d,
        file_path="G:/Tharida_Walking_Patter_01_relative.csv",
        joint_name="Right_Ankle"
    )