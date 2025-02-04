import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def read_csv_signal(file_path):
    """
    Reads a CSV file containing signal data.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        signal (array): Signal values.
    """
    data = pd.read_csv(file_path, header=None)  # Assuming no headers in the CSV
    signal = data.iloc[:, 0].values  # First column is the signal
    return signal

def normalize_signal(signal):
    """
    Normalizes the signal to have zero mean and unit variance.

    Parameters:
        signal (array): Original signal.

    Returns:
        normalized_signal (array): Normalized signal.
    """
    return (signal - np.mean(signal)) / np.std(signal)

def generate_sample_points(signal):
    """
    Generates sample numbers for a given signal.

    Parameters:
        signal (array): Signal values.

    Returns:
        sample_points (array): Generated sample numbers.
    """
    return np.arange(len(signal))

def stretch_and_shift_signal(signal1, signal2, sample1, sample2):
    """
    Stretches and shifts signal2 to align its peaks with signal1.

    Parameters:
        signal1 (array): First signal (reference).
        signal2 (array): Second signal (to be aligned).
        sample1 (array): Sample numbers for signal1.
        sample2 (array): Sample numbers for signal2.

    Returns:
        aligned_signal2 (array): Stretched and shifted version of signal2.
        new_sample2 (array): New sample numbers for signal2.
    """
    # Ensure signal2 and sample2 have consistent lengths
    if len(sample2) != len(signal2):
        raise ValueError("Sample2 and Signal2 lengths do not match.")
    
    # Resample signal2 to match the length of signal1 (stretching)
    resample_factor = len(sample1) / len(sample2)
    stretched_sample2 = np.linspace(0, len(sample2) - 1, len(sample1))
    interpolation_func = interp1d(np.arange(len(signal2)), signal2, kind='cubic', fill_value="extrapolate")
    stretched_signal2 = interpolation_func(stretched_sample2)

    # Compute cross-correlation to find the lag for alignment
    correlation = correlate(signal1, stretched_signal2, mode='full')
    lag = np.argmax(correlation) - (len(signal1) - 1)

    # Shift the stretched signal by the computed lag
    shifted_sample2 = np.arange(len(sample1)) + lag
    shifted_signal2_func = interp1d(shifted_sample2, stretched_signal2, kind='cubic', fill_value="extrapolate")
    aligned_signal2 = shifted_signal2_func(np.arange(len(sample1)))

    return aligned_signal2, np.arange(len(sample1))

# File paths (replace with your file paths)
file_path1 = "C:/Users/acer/Documents/Accedemic_Folder_E19254/Wearable_Hardwear_Project/Sensor_Data/Validation/Thevindu_01_01_2025/Results/Validation_20250101_trial02.csv"  # CSV file for signal2
file_path2 = "C:/Users/acer/Documents/Accedemic_Folder_E19254/Wearable_Hardwear_Project/Sensor_Data/Validation/Thevindu_01_01_2025/Results/frames_trial02/Angles.csv"  # CSV file for signal1

# Read the signals from the CSV files
signal1 = read_csv_signal(file_path1)
signal2 = read_csv_signal(file_path2)

# Normalize the signals to consider only their shapes
signal1_normalized = normalize_signal(signal1)
signal2_normalized = normalize_signal(signal2)

# Generate sample numbers
sample1 = generate_sample_points(signal1_normalized)
sample2 = generate_sample_points(signal2_normalized)

# Stretch and shift signal2 to align with signal1
aligned_signal2, new_sample2 = stretch_and_shift_signal(signal1_normalized, signal2_normalized, sample1, sample2)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sample1[1500:6000], signal1_normalized[1500:6000], label="Angle from sensor data", linestyle='--')
#plt.plot(sample2, signal2_normalized, label="Signal 2 (Original, Normalized)", alpha=0.7)
plt.plot(new_sample2[1500:6000], aligned_signal2[1500:6000], label="Angle from CV model", linestyle='-')
plt.xlabel("Sample Number")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.title("Trial 02")
plt.grid()
plt.show()
