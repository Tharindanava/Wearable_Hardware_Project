# file_io.py
# Contains functions related to opening and writing files
# specific to this application.

import os
import csv
import numpy as np

from settings import *

# Loads a CSV file to a numpy array
def load_dataset_from_csv_to_ndarray(dataset_path,
                                     datapoint_range_start=0,
                                     datapoint_count=1e+9,
                                     csv_skip_rows = (0,),             # Zero-based indices of the CSV rows to skip,
                                     csv_skip_cols = ()) -> np.array:  # Zero-based indices of the CSV columns to skip)
    
    dataset = []
    datapoint_count = int(datapoint_count)

    with open(dataset_path, 'r') as csv_file_handle:
        csv_reader = csv.reader(csv_file_handle, delimiter=',') # CSV reader

        line_count = 0 # Line counter
        valid_line_count = 0 # Valid line counter (rows with datapoints)

        for row in csv_reader: # Iterate through the CSV file
            if ((line_count < datapoint_range_start) or (line_count in csv_skip_rows)): # Skip rows
                line_count += 1
                continue
            #endif
            
            # Process and add values to original_data array
            filtered_row = [cell for j, cell in enumerate(row) if j not in csv_skip_cols]
            filtered_row_values = list(map(float, filtered_row))

            # Further per-line processing
            # filtered_row_values_float = [float(x) for x in filtered_row]
            # filtered_row = "{0:0.6f},{1:0.6f},{2:0.6f},{3:0.6f},{4:0.6f},{5:0.6f}".format(*filtered_row_values_float)
            # filtered_row = filtered_row.split(',')
            
            dataset.append(filtered_row_values)
            
            line_count += 1
            valid_line_count += 1

            if len(dataset) == datapoint_count:
                break
            #endif
        #endfor
    #endwith

    dataset = np.array(dataset)
    print("Dataset size:", dataset.shape)

    return dataset
#enddef