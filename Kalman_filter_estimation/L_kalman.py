"""
kalman.py
20/04/2024

Implements functions for the Kalman filter.
"""

import numpy as np
from collections import namedtuple
from numpy.linalg import norm

from settings import *
from helper_functions import list2vec as l2v
import geometry as geo

def L_kalman_step(P_k_k, dt, omega_m_kp1, acc_r, q_hat_k_k, b_hat_k_k, sigma_r, sigma_b, omega_hat_k_k):
    print("LKF")
    """
    Applies a step of Kalman filtering

    Parameters
    ---------------
    P_k_k : 6x6 matrix
        State covariance matrix of (k-1)th (previous) step
    dt : float
        Sampling interval
    omega_m_kp1 : vector
        Angular velocity vector
    acc_r : vector
        Acceleration vector
    q_hat_k_k : quaternion
        Quaternion of rotation at (k-1)th (previous) step
    b_hat_k_k : vector
        Gyro noise bias at (k-1)th (previous) step
    sigma_r : float
        Standard deviation of angular velocity noise
    sigma_b : float
        Standard deviation of gyro noise bias
    omega_hat_k_k : vector
        Angular velocity vector of the previous step
    
    Returns
    -------
    namedtuple
        A namedtuple object containing:
            C_kp1         : Rotation matrix at kth (current) sample
            q_kp1_kp1     : Quaternion of rotation at kth (current) sample
            dq_plus       : dq of quaternion of rotation at kth (current) sample
            P_kp1_kp1     : State covariance matrix at kth (current) sample
            b_hat_kp1_kp1 : Gyro noise bias at kth sample (current) sample
    """

    return_list_type = namedtuple("return_list_type", "C_kp1 q_kp1_kp1 dq_plus P_kp1_kp1 b_hat_kp1_kp1 omega_kp1_kp1")
    return_list = return_list_type(C_kp1 = None, q_kp1_kp1 = None, dq_plus = None, 
                                   P_kp1_kp1 = None, b_hat_kp1_kp1 = None, omega_kp1_kp1 = None)
    
    gravity_vector = l2v([0.0, 0.0, -1.0]) # Gravity in the +x direction
                                           # as the sensors are oriented such that +x aligns with g

    # Assume that noise is equally distributed in the three spatial directions
    # Noise covariance matrix becomes and identity matrix
    Nr = (sigma_omega_r_c**2)*geo.eye3 # Noise covariance for angular velocity noise
    Nw = (sigma_omega_w_c**2)*geo.eye3 # Noise covariance matrix for angular velocity bias noise
    R  = (sigma_accel_m_c**2)*geo.eye3 # Noise covariance matrix for acceleration measurement

    b_hat_kp1_k     = b_hat_k_k # Propagate the bias from previous step
    omega_hat_kp1_k = omega_m_kp1 - b_hat_kp1_k

    # Propagate quaternion using the quaternion from previous step q_hat_k_k
    OMEGA = geo.cross_prod_mat_4
    omega_bar   = 0.5*(omega_hat_k_k + omega_hat_kp1_k)
    q_hat_kp1_k = geo.mat_exp(OMEGA(0.5*dt*omega_bar)) @ q_hat_k_k
    q_hat_kp1_k += ((dt**2)/48.0)*(OMEGA(omega_hat_kp1_k)@OMEGA(omega_hat_k_k) - OMEGA(omega_hat_k_k)@OMEGA(omega_hat_kp1_k)) @ q_hat_k_k
    
    # Compute discrete-time process noise covariance matrix
    omega0 = omega_hat_kp1_k
    Q11 = (sigma_omega_r_c**2*dt*geo.eye3) + sigma_omega_w_c**2 * ((geo.eye3 * ((dt**3)/3)) + (((((norm(omega0) * dt)**3)/3) + 2*np.sin(norm(omega0)*dt) - (2*norm(omega0)*dt) )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**5)))
    Q22 = (sigma_omega_w_c**2)*dt*geo.eye3
    Q12 = (sigma_omega_w_c**2) * ( (geo.eye3*((dt**2)/2)) - ((((norm(omega0)*dt) - np.sin(norm(omega0)*dt))/(norm(omega0))**3)*geo.cross_prod_mat_3(omega0)) + (((((norm(omega0) * dt)**2)/2) + np.cos(norm(omega0)*dt) - 1 )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**4)))
    Q21 = Q12.transpose()
    Qd_k = np.block([[Q11, Q12],
                     [Q21, Q22]])

    # System matrix for error state space (continuous-time) (constant)
    Fc = np.block([[-geo.eye3 * (omega_hat_k_k), -geo.eye3            ], 
                   [np.zeros(shape=(3,3)),                np.zeros(shape=(3,3))]])
    
    # State transition matrix for the discrete time error state space
    #Phi_kp1 = geo.mat_exp(Fc*dt)
    Phi_kp1=Fc*dt
    # State covariance matrix
    P_kp1_k = Phi_kp1 @ P_k_k @ Phi_kp1.transpose() + Qd_k

    # Kalman Update --------------------------------------------------------------------------------
    # Given the propagated state estimates q_hat_kp1_k, b_hat_kp1_k, P_kp1_k, z_kp1 (current measurement), H (meas matrix)
    
    # Measurement matrix
    C = geo.quaternion_to_rotation_matrix(q_hat_kp1_k)
    H_kp1 = np.block([geo.eye3*(C @ gravity_vector), np.zeros(shape=(3,3))])

    z_hat_kp1 = C @ gravity_vector

    # Compute residual
    z_kp1 = acc_r
    r_kp1 = z_kp1 - z_hat_kp1

    # Covariance of residual
    S_kp1 = H_kp1 @ P_kp1_k @ H_kp1.transpose() + R

    # Kalman gain
    K_kp1 = P_kp1_k @ H_kp1.transpose() @ np.linalg.inv(S_kp1)

    # Correction
    delta_x_hat_plus = K_kp1 @ r_kp1

    dq_plus = np.block([[0.5*delta_x_hat_plus[0:3]],[np.array([1])]]) # Quaternion delta (correction to be applied)
    delta_b_hat_plus = delta_x_hat_plus[3:6] # Gyro noise bias delta

    # Apply correction
    q_kp1_kp1 = geo.quaternion_multiply((dq_plus)/(norm(dq_plus)), q_hat_kp1_k) # Quaternion update
    b_hat_kp1_kp1 = b_hat_kp1_k + delta_b_hat_plus # Gyro noise bias update

    # Calculate rotation matrix corresponding to q_kp1_kp1
    C_kp1 = geo.quaternion_to_rotation_matrix(q_kp1_kp1)

    # Compute updated covariance matrix
    P_kp1_kp1 = (geo.eye6-K_kp1@H_kp1) @ P_kp1_k @ (geo.eye6-K_kp1@H_kp1).transpose() + K_kp1 @ R @ K_kp1.transpose()

    # Update the estimated angular velocity using new estimation for bias
    omega_hat_kp1_kp1 = omega_m_kp1 - b_hat_kp1_kp1

    # Populate return list
    return_list = return_list_type(C_kp1, q_kp1_kp1, dq_plus, P_kp1_kp1, b_hat_kp1_kp1, omega_hat_kp1_kp1)

    return return_list
#enddef
