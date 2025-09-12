# Import custom modules
import sys
import os

# Add the package folder itself to sys.path
sys.path.append(r"C:/Users/acer/Documents/GitHub/kalman-universe/ekf-quat-py")

# Now imports inside geometry.py like "import helper_functions" will work
import geometry as geo
import geometry         as geo
import analyze_data     as azd
import helper_functions as hf
import settings         as sts
import file_io          as fio
import presentation     as psn
import euler_pca        as epca
import kalman           as kl
import dataset_obj      as dso

