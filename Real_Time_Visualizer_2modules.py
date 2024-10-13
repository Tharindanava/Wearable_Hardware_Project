import paho.mqtt.client as mqtt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import json
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# MQTT Configuration (replace with your MQTT broker details and topics)
BROKER = '192.168.1.103'
PORT = 1883
TOPIC_1 = 'sensor/mpu6050_01'
TOPIC_2 = 'sensor/mpu6050_02'

# Initialize empty lists to store incoming data
timestamps_1, timestamps_2 = [], []
acc_X_1, acc_Y_1, acc_Z_1 = [], [], []
acc_X_2, acc_Y_2, acc_Z_2 = [], [], []
quaternions_1, quaternions_2 = [], []

# Variable for adjustable bar length (you can change this)
bar_length = 10.0

# MQTT callback when a message is received for module 1
def on_message_1(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        timestamps_1.append(float(data.get('timestamp', 0)))
        acc_X_1.append(float(data.get('acc_X', 0)))
        acc_Y_1.append(float(data.get('acc_Y', 0)))
        acc_Z_1.append(float(data.get('acc_Z', 0)))
        quaternions_1.append([float(data.get('w', 0)), float(data.get('i', 0)), float(data.get('j', 0)), float(data.get('k', 0))])
    except (ValueError, KeyError) as e:
        print(f"Error in message for module 1: {e}")

# MQTT callback when a message is received for module 2
def on_message_2(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        timestamps_2.append(float(data.get('timestamp', 0)))
        acc_X_2.append(float(data.get('acc_X', 0)))
        acc_Y_2.append(float(data.get('acc_Y', 0)))
        acc_Z_2.append(float(data.get('acc_Z', 0)))
        quaternions_2.append([float(data.get('w', 0)), float(data.get('i', 0)), float(data.get('j', 0)), float(data.get('k', 0))])
    except (ValueError, KeyError) as e:
        print(f"Error in message for module 2: {e}")

# Setup the MQTT client for module 1
client_1 = mqtt.Client()
client_1.on_message = on_message_1
client_1.connect(BROKER, PORT, 60)
client_1.subscribe(TOPIC_1)

# Setup the MQTT client for module 2
client_2 = mqtt.Client()
client_2.on_message = on_message_2
client_2.connect(BROKER, PORT, 60)
client_2.subscribe(TOPIC_2)

# Create a 3D box (top and bottom faces) for each module
bottom_face = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
top_face = bottom_face + np.array([0, 0, 0.1])

# Function to update the 3D box's position and orientation for both modules
def update_box_3d(i, bottom, top, patch_bottom_1, patch_top_1, patch_bottom_2, patch_top_2, bar):
    if len(acc_X_1) > 0 and len(acc_X_2) > 0:  # Ensure there's data from both modules
        latest_index_1 = len(acc_X_1) - 1
        latest_index_2 = len(acc_X_2) - 1

        # Rotation for module 1
        quat_1 = quaternions_1[latest_index_1]
        r_1 = R.from_quat([quat_1[1], quat_1[2], quat_1[3], quat_1[0]])  # Scipy expects [i, j, k, w]
        rotated_bottom_1 = r_1.apply(bottom)
        rotated_top_1 = r_1.apply(top)

        # Rotation for module 2
        quat_2 = quaternions_2[latest_index_2]
        r_2 = R.from_quat([quat_2[1], quat_2[2], quat_2[3], quat_2[0]])  # Scipy expects [i, j, k, w]
        rotated_bottom_2 = r_2.apply(bottom)
        rotated_top_2 = r_2.apply(top)

        # Update the 3D patches with new positions
        patch_bottom_1.set_verts([rotated_bottom_1])
        patch_top_1.set_verts([rotated_top_1])
        patch_bottom_2.set_verts([rotated_bottom_2])
        patch_top_2.set_verts([rotated_top_2])

        # Update the bar between the two modules using the center positions of their top faces
        center_1 = np.mean(rotated_top_1, axis=0)  # Get center of the top face of module 1
        center_2 = np.mean(rotated_top_2, axis=0)  # Get center of the top face of module 2

        # Update the bar's coordinates based on module 1 and module 2 top faces' centers
        bar.set_data([center_1[0], center_2[0]], [center_1[1], center_2[1]])
        bar.set_3d_properties([center_1[2], center_2[2]])

    return patch_bottom_1, patch_top_1, patch_bottom_2, patch_top_2, bar

# Setup the figure and initial 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# Create Poly3DCollection for the bottom and top faces of the boxes (both modules)
patch_bottom_1 = Poly3DCollection([bottom_face], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.75)
patch_top_1 = Poly3DCollection([top_face], facecolors='magenta', linewidths=1, edgecolors='b', alpha=0.75)
patch_bottom_2 = Poly3DCollection([bottom_face], facecolors='green', linewidths=1, edgecolors='r', alpha=0.75)
patch_top_2 = Poly3DCollection([top_face], facecolors='yellow', linewidths=1, edgecolors='b', alpha=0.75)

# Add patches to the plot
ax.add_collection3d(patch_bottom_1)
ax.add_collection3d(patch_top_1)
ax.add_collection3d(patch_bottom_2)
ax.add_collection3d(patch_top_2)

# Initialize the line (bar) linking the two modules
bar, = ax.plot([0, bar_length], [0, 0], [0, 0], color='blue', linewidth=2)

# Create the 3D animation
anim_3d = FuncAnimation(fig, update_box_3d, fargs=(bottom_face, top_face, patch_bottom_1, patch_top_1, patch_bottom_2, patch_top_2, bar), interval=240, cache_frame_data=False, save_count=50)

# Start the MQTT clients' loops in a non-blocking way
client_1.loop_start()
client_2.loop_start()

# Display the 3D animation
plt.show()

# Stop the MQTT clients' loops after the animation is done (optional)
client_1.loop_stop()
client_2.loop_stop()
