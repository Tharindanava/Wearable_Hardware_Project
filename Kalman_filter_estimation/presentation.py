"""
presentation.py
24/04/2024

Presentation layer for the application.
Defines functions for plotting and graphics.
"""

import matplotlib.pyplot as plt

# Plots the nine Euler angled (of each sensor, of each axis)
def plot_euler_angles(n_datapoints, eul_lists):#eul21_list, eul43_list, eul65_list):
    print("HII0")
    n_plot_cols = len(eul_lists)
    fig2, ax2 = plt.subplots(nrows=3+1, ncols=n_plot_cols, gridspec_kw={"height_ratios":[0.01,1,1,1]})

    ax2 = ax2.reshape((3+1,n_plot_cols))
    axes_str_list = ('x', 'y', 'z')
    
    for k_eul_list in range(len(eul_lists)):
        print("HII1")
        eul_list = eul_lists[k_eul_list]

        # Column heading
        ax2[0,k_eul_list].axis("off")
        ax2[0,k_eul_list].set_title("Column {}".format(k_eul_list), fontweight='bold')

        for rot_axis_idx in range(3): # Iterate through each axis
            # Offset the values so that the first value of each axis is 0
            #eul_list[:,rot_axis_idx,0] = eul_list[:,rot_axis_idx,0]-eul_list[0,rot_axis_idx,0]###################oshada

            ax2[rot_axis_idx+1,k_eul_list].plot(range(0,n_datapoints), eul_list[:,rot_axis_idx,0], lw=1)
            ax2[rot_axis_idx+1,k_eul_list].grid(True)
            ax2[rot_axis_idx+1,k_eul_list].set_xlabel('k')
            ax2[rot_axis_idx+1,k_eul_list].set_ylabel(axes_str_list[rot_axis_idx])
        #endfor
    #endfor
    print("HII2")
    # fig2.suptitle('Euler angles', fontsize=14)
    fig2.subplots_adjust(hspace=0.5)#, top=0.9, bottom=0.1, left=0.15, right=0.95)
    fig2.suptitle('UKF_consistent')
    fig2.show()
    
    input()
    print("HII3")
#enddef