import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Since the file is in CSV format, we'll load it as CSV and proceed.
data = pd.read_csv("C:/Users/acer/Documents/Accedemic_Folder_E19254/Agrivoltic_project/Test_data/sensor_data_20241008_04.csv")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

# Extract relevant columns for animation
timestamps = data['timestamp'].values
acc_X = data['acc_X'].values
acc_Y = data['acc_Y'].values
acc_Z = data['acc_Z'].values
quaternions = data[['w', 'i', 'j', 'k']].values

# Create the 3D square (a flat square in the XY plane at z=0)
square_3d = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])

# Function to update the 3D square's position and orientation based on accelerometer and quaternion data
def update_square_3d(i, square, patch):
    # Calculate translation based on accelerometer data (simplified integration)
    translation = np.array([np.sum(acc_X[:i]) * 0.240, np.sum(acc_Y[:i]) * 0.240, np.sum(acc_Z[:i]) * 0.240])
    
    # Calculate rotation based on quaternion
    quat = quaternions[i]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Scipy expects quaternions in [i, j, k, w] order
    rotated_square = r.apply(square)
    
    # Update the patch with the new position
    new_square = rotated_square + translation
    patch[0].set_verts([new_square])
    return patch

# Setup the figure and the initial 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# Create a Poly3DCollection representing the 3D square
patch = [Poly3DCollection([square_3d], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.75)]

# Add patch to the plot
ax.add_collection3d(patch[0])

# Create the 3D animation
anim_3d = FuncAnimation(fig, update_square_3d, frames=len(timestamps), fargs=(square_3d, patch), interval=240)

# Display the 3D animation
plt.show()

