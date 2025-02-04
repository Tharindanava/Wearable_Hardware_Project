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
def cross_mul3(w):
    W = np.array([[0, -w[2], w[1]],
                  [w[2], 0, -w[0]],
                  [-w[1], w[0], 0]])
    return W

def cross_mul4(w):
    W = np.array([[0, w[2], -w[1], w[0]],
                  [-w[2], 0, w[0], w[1]],
                  [w[1], -w[0], 0, w[2]],
                  [-w[0], -w[1], -w[2], 0]])
    return W

def quat_to_mat(q):
    q1, q2, q3, q4 = q
    C = ((2 * (q4**2) - 1) * np.eye(3)) - (2 * q4 * cross_mul3([q1, q2, q3])) + 2 * np.outer([q1, q2, q3], [q1, q2, q3])
    return C

def quat_mul(x, y):
    qu = np.array([x[3] * y[0] + x[2] * y[1] - x[1] * y[2] + x[0] * y[3],
                   -x[2] * y[0] + x[3] * y[1] + x[0] * y[2] + x[1] * y[3],
                   x[1] * y[0] - x[0] * y[1] + x[3] * y[2] + x[2] * y[3],
                   -x[0] * y[0] - x[1] * y[1] - x[2] * y[2] + x[3] * y[3]])
    return qu

def U_kalman_step(P_k_k, dt, omega_m_kp1, acc_r, q_hat_k_k, b_hat_k_k, sigma_r, sigma_b, omega_hat_k_k):
    print("UKF")
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
    Nb = (sigma_omega_w_c**2)*geo.eye3 # Noise covariance matrix for angular velocity bias noise
    R  = (sigma_accel_m_c**2)*geo.eye3 # Noise covariance matrix for acceleration measurement

    n =6  # State dimension (quaternion + bias)
    kappa = 0  # Default scaling parameter
    alpha = 1e-3
    beta = 2  # Optimal for Gaussian distributions
    
    lambda_ = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lambda_)
    
     # Initial state and covariance
    X = np.concatenate((omega_hat_k_k, b_hat_k_k))  # Augmented state vector (size 6)
    P_aug = P_k_k  # State covariance matrix (size 6x6)
    
    # Regularization to ensure positive definiteness
    epsilon = 1e-3  # Regularization factor
    P_aug += epsilon * np.eye(P_aug.shape[0])
    
    # Check eigenvalues for positive definiteness
    if np.any(np.linalg.eigvals(P_aug) <= 0):
        raise ValueError('Covariance matrix is not positive definite.')
    
    # Calculate sigma points
    sqrtP = np.linalg.cholesky((n + lambda_) * P_aug)
    sqrtP= np.expand_dims(sqrtP, axis=1)
    sigma_points = np.zeros((n,1, 2 * n + 1))
    #print(sqrtP.shape)
    sigma_points[:,:, 0] = X
    
    for i in range(n):
        sigma_points[:,:, i + 1] = X  + gamma * sqrtP[:,:, i]
        sigma_points[:,:,i + n + 1] = X - gamma * sqrtP[:,:, i]
    
    # Propagate sigma points through the process model
    X_pred = np.zeros((n,1, 2 * n + 1))
    for i in range(2 * n + 1):
        omega_hat_kp1_k = omega_m_kp1 - sigma_points[3:6,:, i]
        # Propagate quaternion using the quaternion from previous step q_hat_k_k
        OMEGA = geo.cross_prod_mat_4
        omega_bar   = 0.5*(omega_hat_k_k + omega_hat_kp1_k)
        #print(omega_m_kp1.shape)
        q_hat_kp1_k = geo.mat_exp(OMEGA(0.5*dt*omega_bar)) @ q_hat_k_k
        q_hat_kp1_k += ((dt**2)/48.0)*(OMEGA(omega_hat_kp1_k)@OMEGA(omega_hat_k_k) - OMEGA(omega_hat_k_k)@OMEGA(omega_hat_kp1_k)) @ q_hat_k_k
        #print(omega_bar.shape)
        X_pred[0:3,:, i] = omega_bar
        X_pred[3:6,:, i] = sigma_points[3:6, :,i]  # Bias remains unchanged
            
    # Predicted state mean
    Wm = np.hstack((lambda_ / (n + lambda_), np.full(2 * n, 0.5 / (n + lambda_))))
    X_mean = np.dot(X_pred, Wm)
    
    # Predicted covariance
    P_kp1_k = np.zeros((n, n))
    for i in range(2 * n + 1):
        dX = X_pred[:, :,i] - X_mean
        P_kp1_k += Wm[i] * np.outer(dX, dX)
    P_kp1_k += np.block([[Nr, np.zeros((3, 3))], [np.zeros((3, 3)), Nb]])  # Process noise
    
    # # Compute discrete-time process noise covariance matrix
    # omega0 = omega_hat_kp1_k
    # Q11 = (sigma_omega_r_c**2*dt*geo.eye3) + sigma_omega_w_c**2 * ((geo.eye3 * ((dt**3)/3)) + (((((norm(omega0) * dt)**3)/3) + 2*np.sin(norm(omega0)*dt) - (2*norm(omega0)*dt) )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**5)))
    # Q22 = (sigma_omega_w_c**2)*dt*geo.eye3
    # Q12 = (sigma_omega_w_c**2) * ( (geo.eye3*((dt**2)/2)) - ((((norm(omega0)*dt) - np.sin(norm(omega0)*dt))/(norm(omega0))**3)*geo.cross_prod_mat_3(omega0)) + (((((norm(omega0) * dt)**2)/2) + np.cos(norm(omega0)*dt) - 1 )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**4)))
    # Q21 = Q12.transpose()
    # Qd_k = np.block([[Q11, Q12],
    #                  [Q21, Q22]])

    # # System matrix for error state space (continuous-time) (constant)
    # Fc = np.block([[-geo.cross_prod_mat_3(omega_hat_k_k), -geo.eye3            ], 
    #                [np.zeros(shape=(3,3)),                np.zeros(shape=(3,3))]])
    
    # # State transition matrix for the discrete time error state space
    # Phi_kp1 = geo.mat_exp(Fc*dt)

    # # State covariance matrix
    # P_kp1_k = Phi_kp1 @ P_k_k @ Phi_kp1.transpose() + Qd_k

    # Kalman Update --------------------------------------------------------------------------------
    # Given the propagated state estimates q_hat_kp1_k, b_hat_kp1_k, P_kp1_k, z_kp1 (current measurement), H (meas matrix)
    
    # Measurement prediction
    z_hat_kp1 = np.zeros((3,1, 2 * n + 1))
    for i in range(2 * n + 1):
        C = geo.quaternion_to_rotation_matrix(q_hat_kp1_k)
        z_hat_kp1[:, :,i] = C @ gravity_vector
    
    Z_mean = np.dot(z_hat_kp1, Wm)  # Predicted measurement mean
    
    # Measurement covariance and cross covariance
    Pzz = np.zeros((3, 3))
    Pxz = np.zeros((n, 3))
    for i in range(2 * n + 1):
        dZ = z_hat_kp1[:,:, i] - Z_mean
        dX = X_pred[:,:, i] - X_mean
        Pzz += Wm[i] * np.outer(dZ, dZ)
        Pxz += Wm[i] * np.outer(dX, dZ)
    Pzz += R  # Add measurement noise
    
    # Kalman gain
    K = Pxz @ np.linalg.inv(Pzz)

    # Compute residual
    z_kp1 = acc_r
    r_kp1 = z_kp1 - Z_mean
    
    dX = K @ r_kp1 
    #print(dX.shape)
    dq = np.concatenate((0.5 * dX[0:3,0], [1]))
    #print(q_hat_kp1_k.shape)
    dq= np.expand_dims(dq, axis=1)
    q_kp1_kp1 = geo.quaternion_multiply((dq)/(norm(dq)), q_hat_kp1_k) # Quaternion update
    b_hat_kp1_kp1 = X_mean[3:6] + dX[3:6]
    
    # Update covariance
    P_kp1_kp1 = P_kp1_k - np.dot(K, np.dot(Pzz, K.T))
    
   

    # Calculate rotation matrix corresponding to q_kp1_kp1
    C_kp1 = geo.quaternion_to_rotation_matrix(q_kp1_kp1)


    # Update the estimated angular velocity using new estimation for bias
    #omega_hat_kp1_kp1 = omega_m_kp1 - b_hat_kp1_kp1
    omega_hat_kp1_kp1 =  X_mean[0:3] + dX[0:3]
    # Populate return list
    return_list = return_list_type(C_kp1, q_kp1_kp1, dq, P_kp1_kp1, b_hat_kp1_kp1, omega_hat_kp1_kp1)

    return return_list
#enddef
