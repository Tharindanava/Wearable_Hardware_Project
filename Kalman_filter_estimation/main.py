"""
main.py
20/04/2024

Application entry point
"""

import os
import csv
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np

import geometry as geo
import analyze_data as azd
import helper_functions as hf
from helper_functions import list2vec as l2v
from settings import *
import file_io as fio
from euler_pca import *

def main() -> None:
    """
    Application entry point.

    Parameters: Nothing
    Returns : Nothing
    """

    # 1. Compute the noise to obtain the initial quaternion
    noise_dataset = fio.load_dataset_from_csv_to_ndarray(dataset_path="C:/Users/acer/Documents/Accedemic_Folder_E19254/Wearable_Hardwear_Project/Sensor_Data/Validation/Thevindu_01_01_2025/Validation_trial02_20250101_Ordered.csv",
                                                         datapoint_range_start=0,
                                                         datapoint_count=500)#,
                                                        # csv_skip_rows = (0,1,2),
                                                        # csv_skip_cols = (0,7,8,9))
    print("hi0")
    [noise_eul_lists, noise_R_k_list, noise_q_k_list] = azd.analyze_data(dataset=noise_dataset, 
                                                                         dt=1/64.0, 
                                                                         q0=geo.identity_quaternion, 
                                                                         n_sensors=2)
    print("hi1")
    # Compute the average initial quaternion using initial noise data (initial orientation)
    q0_avg = np.mean(noise_q_k_list[0,:,:,:],axis=0)
    
    # 2. Process the whole dataset
    dataset = fio.load_dataset_from_csv_to_ndarray(dataset_path="C:/Users/acer/Documents/Accedemic_Folder_E19254/Wearable_Hardwear_Project/Sensor_Data/Validation/Thevindu_01_01_2025/Validation_trial02_20250101_Ordered.csv",
                                                   datapoint_range_start=0,
                                                   datapoint_count=6100)#,
                                                #    csv_skip_rows = (0,1,2),
                                                #    csv_skip_cols = (0,7,8,9))

    [eul_lists, R_k_list, q_k_list] = azd.analyze_data(dataset=dataset, 
                                                       dt=1/64.0, 
                                                       q0=q0_avg, 
                                                       n_sensors=2)
    
#     #region PCA for angles -------------------------------------------------------------------------
#     angles = eul_lists[0] # (n x 3 x 1) the Euler angle dataset
#     angles = angles - np.mean(angles, axis=0)
#     angles = np.reshape(angles, (angles.shape[0], 3))
#     angles = angles.transpose()

#     # calculating the covariance matrix of the mean-centered data.
#     cov_mat = np.cov(angles)

#     #Calculating Eigenvalues and Eigenvectors of the covariance matrix
#     eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

#     #sort the eigenvalues in descending order
#     sorted_index = np.argsort(eigen_values)[::-1]
#     eigen_values = eigen_values[sorted_index]

#     #similarly sort the eigenvectors 
#     eigen_vectors = eigen_vectors[:,sorted_index]

#     # select the first n eigenvectors, n is desired dimension
#     # of our final reduced data.
    
#     n_components = 3#2 # Number of components (significant axes)
#     eigenvector_subset = eigen_vectors[:,0:n_components]
#     component_variance_share = eigen_values / np.sum(eigen_values)
#     print(component_variance_share)

#     # Transform the data 
#     X_reduced = np.dot(angles.transpose(), eigenvector_subset)

#     #endregion -------------------------------------------------------------------------------------

#     #region Plot the input data --------------------------------------------------------------------
#     fig_pca1, axs_pca1 = plt.subplots(nrows=3, ncols=1)
#     axs_pca1 = np.reshape(axs_pca1, (3,1))

#     axs_pca1[0,0].plot(range(0,angles.shape[1]), angles[0,:].ravel())
#     axs_pca1[1,0].plot(range(0,angles.shape[1]), angles[1,:].ravel())
#     axs_pca1[2,0].plot(range(0,angles.shape[1]), angles[2,:].ravel())

#     axs_pca1[0,0].grid(True); axs_pca1[1,0].grid(True); axs_pca1[2,0].grid(True)
#     fig_pca1.show()
#     #endregion -------------------------------------------------------------------------------------

#     #region Plot the 3d scatter of input data ------------------------------------------------------
#     fig_pca2 = plt.figure()
#     axs_pca2 = fig_pca2.add_subplot(projection='3d')
    
#     axs_pca2.scatter(angles[0,:], angles[1,:], angles[2,:])
#     axs_pca2.grid(True)
#     axs_pca2.set_xlabel('x')
#     axs_pca2.set_ylabel('y')
#     axs_pca2.set_zlabel('z')

#     fig_pca2.show()
#     #endregion -------------------------------------------------------------------------------------

#     #region Plot the 2d scatter of reduced data ----------------------------------------------------
#     fig_pca3 = plt.figure()
#     axs_pca3 = fig_pca3.add_subplot()
    
#     axs_pca3.scatter(X_reduced[:,0], X_reduced[:,1])
#     axs_pca3.grid(True)
#     axs_pca3.set_xlabel('x1')
#     axs_pca3.set_ylabel('x2')

#     fig_pca3.show()
#     #endregion -------------------------------------------------------------------------------------

#     #region Plot the reduced data with time --------------------------------------------------------
#     fig_pca4, axs_pca4 = plt.subplots(nrows=3, ncols=1)
#     axs_pca4 = np.reshape(axs_pca4, (3,1))
    
#     axs_pca4[0,0].plot(range(0,angles.shape[1]), X_reduced[:,0].ravel())
#     axs_pca4[1,0].plot(range(0,angles.shape[1]), X_reduced[:,1].ravel())
#     axs_pca4[2,0].plot(range(0,angles.shape[1]), X_reduced[:,2].ravel())

#     axs_pca4[0,0].grid(True); axs_pca4[1,0].grid(True); axs_pca4[2,0].grid(True)
#     y_max = 1.1*(np.array([X_reduced[:,0].ravel().max(),X_reduced[:,1].ravel().max(),X_reduced[:,2].ravel().max()]).max())
#     y_min = 1.1*(np.array([X_reduced[:,0].ravel().min(),X_reduced[:,1].ravel().min(),X_reduced[:,2].ravel().min()]).min())
#     axs_pca4[0,0].set_ylim([y_min,y_max])
#     axs_pca4[1,0].set_ylim([y_min,y_max])
#     axs_pca4[2,0].set_ylim([y_min,y_max])
#     fig_pca4.show()
#     #endregion -------------------------------------------------------------------------------------

    
    
#     pca_ax1 = X_reduced[:,0].ravel()
#     pca_ax2 = X_reduced[:,1].ravel()
#     pca_ax3 = X_reduced[:,2].ravel()

#     np.savetxt('data.csv', X_reduced, delimiter=',')

#     print("Done.")
#     input()
# #enddef

if __name__ == "__main__":
    main()
#endif
