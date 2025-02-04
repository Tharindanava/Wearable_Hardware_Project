"""
settings.py
20/04/2024

Defines application-specific settings for the analysis.
"""

#region Absolute constants -----------------------------------------------------

N_QUAT_COMPONENTS = 4    # Number of components in a quaternion

#endregion ---------------------------------------------------------------------

# region Dataset properties ----------------------------------------------------
N_AXES_PER_SENSOR = 6    # Number of axes per sensor

# SMP_FREQ = 64.0          # Sampling frequency for datasets
# SMP_INT  = 1/SMP_FREQ    # Sampling interval for datasets

# Minimum and maximum values of the dataset (based on data type)
# TODO: re-adjust these to adhere with the correct units
accel_old_min_value = -32768.0; accel_target_min_value   = -1.0*1e-15 #-1.0*0
accel_old_max_value = +32768.0; accel_target_max_value   = +1.0*1e-15 #+1.0
gyro_old_min_value  = -32768.0; gyro_target_min_value    = -1.0#-1.0*32 #-1.0*0
gyro_old_max_value  = +32768.0; gyro_target_max_value    = +1.0#+1.0*32 #+1.0
#endregion ---------------------------------------------------------------------

#region Kalman -----------------------------------------------------------------
GRAVITY = [+1, 0, 0]#[0.0,0.0,-1.0]

sigma_omega_r_c     = 3e-3 # Standard deviation of angular velocity rate noise (for continuous time)
sigma_omega_w_c     = 80e-5 # Standard deviation of angular velocity bias noise (for continuous time)
sigma_accel_m_c     = 2e-2 # Standard deviation of acceleration measurement noise (for continuous time)
#endregion ---------------------------------------------------------------------
