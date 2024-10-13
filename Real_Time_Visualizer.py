import paho.mqtt.client as mqtt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import json
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# MQTT Configuration (replace with your MQTT broker details and topic)
BROKER = '192.168.1.103'
PORT = 1883
TOPIC = 'sensor/mpu6050_02'

# Initialize empty lists to store incoming data
timestamps = []
acc_X, acc_Y, acc_Z = [], [], []
quaternions = []

# MQTT callback when a message is received
def on_message(client, userdata, msg):
    try:
        # Decode the incoming message (assume JSON format)
        data = json.loads(msg.payload.decode())

        # Append new data to the lists, with strict conversion to floats
        timestamps.append(float(data.get('timestamp', 0)))
        acc_X.append(float(data.get('acc_X', 0)))  # Ensure valid float
        acc_Y.append(float(data.get('acc_Y', 0)))  # Ensure valid float
        acc_Z.append(float(data.get('acc_Z', 0)))  # Ensure valid float
        quaternions.append([float(data.get('w', 0)), float(data.get('i', 0)), float(data.get('j', 0)), float(data.get('k', 0))])  # Ensure valid float
    except (ValueError, KeyError) as e:
        print(f"Error in message: {e}")

# Setup the MQTT client
client = mqtt.Client()
client.on_message = on_message

# Connect to the broker and subscribe to the topic
client.connect(BROKER, PORT, 60)
client.subscribe(TOPIC)

# Create a 3D box (top and bottom faces with a height of 1 unit)
bottom_face = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
top_face = bottom_face + np.array([0, 0, 0.1])  # Shift the top face 1 unit up in the z-direction

# Function to update the 3D box's position and orientation based on accelerometer and quaternion data
def update_box_3d(i, bottom, top, patch_bottom, patch_top):
    if len(acc_X) > 0:  # Ensure there's data available before proceeding
        # Get the latest data index
        latest_index = len(acc_X) - 1
        
        # Calculate translation based on accelerometer data (simplified integration)
        # translation = np.array([np.sum(acc_X[:latest_index]) * 0.240, np.sum(acc_Y[:latest_index]) * 0.240, np.sum(acc_Z[:latest_index]) * 0.240])
        
        # Calculate rotation based on the latest quaternion
        quat = quaternions[latest_index]

        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Scipy expects quaternions in [i, j, k, w] order
        rotated_bottom = r.apply(bottom)
        rotated_top = r.apply(top)
        
        # Update the patches with the new position
        new_bottom = rotated_bottom #+ translation
        new_top = rotated_top #+ translation
        patch_bottom.set_verts([new_bottom])
        patch_top.set_verts([new_top])
    return patch_bottom, patch_top

# Setup the figure and the initial 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# Create Poly3DCollection for the bottom and top faces of the box
patch_bottom = Poly3DCollection([bottom_face], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.75)
patch_top = Poly3DCollection([top_face], facecolors='magenta', linewidths=1, edgecolors='b', alpha=0.75)

# Add patches to the plot
ax.add_collection3d(patch_bottom)
ax.add_collection3d(patch_top)

# Create the 3D animation
anim_3d = FuncAnimation(fig, update_box_3d, fargs=(bottom_face, top_face, patch_bottom, patch_top), interval=240, cache_frame_data=False, save_count=50)


# Start the MQTT client loop in a non-blocking way
client.loop_start()

# Display the 3D animation
plt.show()

# Stop the MQTT client loop after the animation is done (optional based on your setup)
client.loop_stop()
