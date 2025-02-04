# helper_functions.py

import numpy as np

# Converts a Python list with n elements to a numpy nx1 vector.
def list2vec(lst):
    return np.array([lst]).astype(float).transpose()
#enddef

# Min-max normalizes a numpy array which is in range [min_value, max_value] to [-1,+1]
def normalize_dataset(data, old_min, old_max, new_min=-1.0, new_max=+1.0):
    return ((data-old_min)/(old_max-old_min))*(new_max-new_min)+new_min
#enddef

def standardize_dataset(dataset, mean_zero=True, 
                        stddev_unity=False, 
                        minmax_scaling=True, 
                        old_min=-1.0,    old_max=+1.0,
                        target_min=-1.0, target_max=-+1.0):
    
    dataset_preprocessed = np.zeros(dataset.shape) + dataset

    if mean_zero:      dataset_preprocessed = dataset_preprocessed - np.mean(dataset_preprocessed, axis=0)
    if stddev_unity:   dataset_preprocessed = dataset_preprocessed/np.std(dataset_preprocessed, axis=0)
    if minmax_scaling: dataset_preprocessed = normalize_dataset(dataset_preprocessed, old_min, old_max, target_min, target_max)

    return np.array(dataset_preprocessed)
#enddef
