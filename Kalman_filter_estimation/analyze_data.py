"""
analyze_data.py
20/04/2024

Defines functions for data analysis.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

import helper_functions as hf
import geometry as geo
from settings import *
from  helper_functions import list2vec as l2v
import L_kalman as klL
import kalman as kl
import U_kalman as k2
import presentation as psn

def analyze_data(dataset, dt, q0, n_sensors) -> list:
    """
    Analyses a dataset 'dataset'.

    Parameters
    ----------
    dataset : numpy array
        The dataset to analyze.
        Specs for dataset: cols = n_axes_per_sensor x n_datapoints
                           rows = n_datapoints
    
    dt : float
        Sampling time of the dataset.

    q0 : quaternion
        Initial quaternion. (unused)
    """
    filename = "Hip90_EKF_con.csv"
    scalar=0.45##############################################oshada
    T = dt                             # Sampling interval
    n_datapoints = len(dataset[:,0])   # Length of first column
    gravity_vector = l2v([0.0,0.0,-1.0]) #GRAVITY)

    # Quaternion containers
    q_k_list  = np.zeros(shape=(n_sensors, n_datapoints, N_QUAT_COMPONENTS, 1))
    dq_k_list = np.zeros(shape=(n_sensors, n_datapoints, N_QUAT_COMPONENTS, 1))
    R_k_list  = np.zeros(shape=(n_sensors, n_datapoints, 3, 3))
    Qg        = np.zeros(shape=(N_QUAT_COMPONENTS, n_datapoints, n_sensors))

    for j in range(n_sensors):
        # Extract the accel/gyro data of the current sensor into separate matrices -----------------
        accel_slicer = np.array([0,1,2])+6*j
        gyro_slicer  = np.array([3,4,5])+6*j
        accel_data   = dataset[:,accel_slicer]
        gyro_data    = dataset[:,gyro_slicer]

        # Pre-process the dataset ------------------------------------------------------------------
        accel_data = hf.standardize_dataset(accel_data,
                                            mean_zero=True, stddev_unity=False,
                                            minmax_scaling=True,
                                            old_min=accel_old_min_value,
                                            old_max=accel_old_max_value,
                                            target_min=accel_target_min_value,
                                            target_max=accel_target_max_value)
        gyro_data  = hf.standardize_dataset(gyro_data,
                                            mean_zero=True, stddev_unity=False,
                                            minmax_scaling=True,
                                            old_min=gyro_old_min_value, 
                                            old_max=gyro_old_max_value,
                                            target_min=gyro_target_min_value, 
                                            target_max=gyro_target_max_value)
        
        # Compute initial rotation state (matrix/quaternion) ---------------------------------------
        init_accel            = l2v(accel_data[0,:]) # Initial acceleration (3x1)
        init_gyro             = l2v(gyro_data[0,:])  # Initial angular velocity (3x1)

        init_accel_rot_axis   = geo.vector_cross_product(init_accel, gravity_vector)
        init_accel_rot_axis   = geo.normalize_vector(init_accel_rot_axis)
        init_accel_rot_angle  = geo.angle_between(init_accel, gravity_vector)
        init_accel_quaternion = geo.axis_angle_to_quaternion(init_accel_rot_axis, init_accel_rot_angle)
        init_accel_rot_mat    = geo.axis_angle_to_rotation_matrix(init_accel_rot_axis, init_accel_rot_angle)

        # Manual adjustments to the initial quaternion
        init_accel_quaternion = q0#[:,0,n_sensors]
        init_accel_rot_mat    = geo.quaternion_to_rotation_matrix(init_accel_quaternion)
        
        q_k = init_accel_quaternion
        qg0 = init_accel_quaternion # init_gyro_quaternion # TODO: get the initial quaternion from gyro data
        b_k = l2v([0,0,0])          # Initial gyro noise bias
        P_k = np.zeros(shape=(6,6)) # Initial state covariance matrix
        omega_k = l2v([0,0,0])

        theta_list = []
        
        # Iteration through the data points
        for k in range(n_datapoints):
            #[C_kp1, q_kp1_kp1, dq_plus, P_kp1_kp1, b_hat_kp1_kp1, omega_hat_kp1_kp1] = klL.L_kalman_step(P_k, dt, l2v(gyro_data[k,:]), l2v(accel_data[k,:]), q_k, b_k, sigma_omega_r_c, sigma_omega_w_c, omega_k)
            [C_kp1, q_kp1_kp1, dq_plus, P_kp1_kp1, b_hat_kp1_kp1, omega_hat_kp1_kp1] = kl.kalman_step(P_k, dt, l2v(gyro_data[k,:]), l2v(accel_data[k,:]), q_k, b_k, sigma_omega_r_c, sigma_omega_w_c, omega_k)
            #[C_kp1, q_kp1_kp1, dq_plus, P_kp1_kp1, b_hat_kp1_kp1, omega_hat_kp1_kp1] = k2.U_kalman_step(P_k, dt, l2v(gyro_data[k,:]), l2v(accel_data[k,:]), q_k, b_k, sigma_omega_r_c, sigma_omega_w_c, omega_k)
            
            P_k = P_kp1_kp1
            b_k = b_hat_kp1_kp1
            q_k = q_kp1_kp1
            omega_k = omega_hat_kp1_kp1
            R_k = C_kp1
            dq_k = dq_plus

            q_k_list[j,k,:,:]  = q_k
            dq_k_list[j,k,:,:] = dq_k
            R_k_list[j,k,:,:]  = geo.quaternion_to_rotation_matrix(q_k)

        #     [Cg, qg] = only_gyro(gyro[i,:], qg0, dt)
        #     gq0 = qg
        #     Qg[:,i,k] = gq

            # TODO: visualization
            # theta_list.append(2*np.arccos(q_k[3,0])*180/np.pi)
        #endfor

        # fig = plt.figure()
        # ax = fig.gca()
        # ax.plot(range(len(theta_list)), theta_list)
        # ax.grid(True)
        # fig.show()

        # input()

    #endfor

    #region Single-sensor angle calculation and plotting -------------------------------------------
    if n_sensors == 1:
        eul_list = np.zeros(shape=(n_datapoints, 3, 1))
        for k in range(n_datapoints):
            # 4 sensor model
            # R1 = R_k_list[0,k,:,:]
            # R2 = R_k_list[1,k,:,:]
            # R3 = R_k_list[2,k,:,:]
            # R4 = R_k_list[3,k,:,:]
            # R5 = R_k_list[4,k,:,:]
            # R6 = R_k_list[5,k,:,:]

            q = q_k_list[0,k,:,:]
            r = geo.quaternion_to_rotation_matrix(q)
            eul = geo.rotm2eul(r, scheme='xyz')
            eul_list[k,:,:] = eul
        #endfor

        # Unwrap angles
        # eul_list[:,0,0] = np.unwrap(eul_list[:,0,0], period=360.0)
        # eul_list[:,1,0] = np.unwrap(eul_list[:,1,0], period=360.0)
        # eul_list[:,2,0] = np.unwrap(eul_list[:,2,0], period=360.0)

        # Plot the angles
        eul_lists = [eul_list]
        psn.plot_euler_angles(n_datapoints, eul_lists)

        input()

    #endregion -------------------------------------------------------------------------------------

    #region Joint angle calculation and plotting: 2 sensors ----------------------------------------
    if n_sensors == 2:
        eul21_list = np.zeros(shape=(n_datapoints, 3, 1))

        for k in range(n_datapoints):
            with open(filename, 'a', newline='') as csvfile:
                print("HI1")
                q21 = geo.quaternion_multiply(q_k_list[0,k,:,:], geo.quaternion_inverse(q_k_list[1,k,:,:]))
                r21 = geo.quaternion_to_rotation_matrix(q21)

                eul21 = geo.rotm2eul(r21)

                eul21_list[k,:,:] = eul21
                writer = csv.writer(csvfile)
                writer.writerow(eul21_list[k,1,:])
        #endfor

        eul_lists = [eul21_list]
        psn.plot_euler_angles(n_datapoints, eul_lists)

        input()

    #endregion -------------------------------------------------------------------------------------

    #region Joint angle calculation and plotting: 6 sensors ----------------------------------------
    if n_sensors == 6:
        
        print("HI0")
        eul21_list = np.zeros(shape=(n_datapoints, 3, 1))
        eul43_list = np.zeros(shape=(n_datapoints, 3, 1))
        eul65_list = np.zeros(shape=(n_datapoints, 3, 1))

        for k in range(n_datapoints):
            with open(filename, 'a', newline='') as csvfile:
                print("HI1")
                # 6 sensor model
                # R1 = R_k_list[0,k,:,:]
                # R2 = R_k_list[1,k,:,:]
                # R3 = R_k_list[2,k,:,:]
                # R4 = R_k_list[3,k,:,:]
                # R5 = R_k_list[4,k,:,:]
                # R6 = R_k_list[5,k,:,:]

                q21 = geo.quaternion_multiply(q_k_list[0,k,:,:], geo.quaternion_inverse(q_k_list[1,k,:,:]))
                r21 = geo.quaternion_to_rotation_matrix(q21)

                q43 = geo.quaternion_multiply(q_k_list[2,k,:,:], geo.quaternion_inverse(q_k_list[3,k,:,:]))
                r43 = geo.quaternion_to_rotation_matrix(q43)

                q65 = geo.quaternion_multiply(q_k_list[4,k,:,:], geo.quaternion_inverse(q_k_list[5,k,:,:]))
                r65 = geo.quaternion_to_rotation_matrix(q65)

                eul21 = geo.rotm2eul(r21)
                eul43 = geo.rotm2eul(r43)
                eul65 = geo.rotm2eul(r65)

                eul21_list[k,:,:] = eul21*scalar
                eul43_list[k,:,:] = np.pi-eul43*scalar##################oshada
                eul65_list[k,:,:] = np.pi+eul65*scalar
                writer = csv.writer(csvfile)
                writer.writerow(eul65_list[k,1,:])
        #endfor
        print("HI2")
        eul_lists = [eul21_list, eul43_list, eul65_list]
        psn.plot_euler_angles(n_datapoints, eul_lists)
        print("HI3")
        input()

    #endregion -------------------------------------------------------------------------------------

    #region Joint angle calculation and plotting: 4 sensors ----------------------------------------
    if n_sensors == 4:
        eul21_list = np.zeros(shape=(n_datapoints, 3, 1))
        eul32_list = np.zeros(shape=(n_datapoints, 3, 1))
        eul43_list = np.zeros(shape=(n_datapoints, 3, 1))

        for k in range(n_datapoints):
            q21 = geo.quaternion_multiply(q_k_list[0,k,:,:], geo.quaternion_inverse(q_k_list[1,k,:,:]))
            r21 = geo.quaternion_to_rotation_matrix(q21)

            q32 = geo.quaternion_multiply(q_k_list[1,k,:,:], geo.quaternion_inverse(q_k_list[2,k,:,:]))
            r32 = geo.quaternion_to_rotation_matrix(q32)

            q43 = geo.quaternion_multiply(q_k_list[2,k,:,:], geo.quaternion_inverse(q_k_list[3,k,:,:]))
            r43 = geo.quaternion_to_rotation_matrix(q43)

            eul21 = geo.rotm2eul(r21)
            eul32 = geo.rotm2eul(r32)
            eul43 = geo.rotm2eul(r43)

            eul21_list[k,:,:] = eul21
            eul32_list[k,:,:] = eul32
            eul43_list[k,:,:] = eul43
        #endfor
        
        eul_lists = [eul21_list, eul32_list, eul43_list]
        psn.plot_euler_angles(n_datapoints, eul_lists)

        input()

    #endregion -------------------------------------------------------------------------------------

    #region Joint angle calculation and plotting: 3 sensors ----------------------------------------
    if n_sensors == 3:
        eul21_list = np.zeros(shape=(n_datapoints, 3, 1))
        eul32_list = np.zeros(shape=(n_datapoints, 3, 1))

        for k in range(n_datapoints):
            q21 = geo.quaternion_multiply(q_k_list[0,k,:,:], geo.quaternion_inverse(q_k_list[1,k,:,:]))
            r21 = geo.quaternion_to_rotation_matrix(q21)

            q32 = geo.quaternion_multiply(q_k_list[1,k,:,:], geo.quaternion_inverse(q_k_list[2,k,:,:]))
            r32 = geo.quaternion_to_rotation_matrix(q32)

            eul21 = geo.rotm2eul(r21)
            eul32 = geo.rotm2eul(r32)

            eul21_list[k,:,:] = eul21
            eul32_list[k,:,:] = eul32
        #endfor
        
        eul_lists = [eul21_list, eul32_list]
        psn.plot_euler_angles(n_datapoints, eul_lists)

        input()

    #endregion -------------------------------------------------------------------------------------

    #region Plot overall rotation angle ------------------------------------------------------------
    # fig3, ax3 = plt.subplots(1, 1)
    # ax3.plot(eul43_list[:,0,0], eul65_list[:,1,0])
    # fig3.show()
    # input()
    #endregion -------------------------------------------------------------------------------------

    return [eul_lists, R_k_list, q_k_list]
#enddef