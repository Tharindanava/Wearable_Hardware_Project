import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load joint angles from CSV
joint_angles = pd.read_csv("C:/Users/acer/Documents/GitHub/Wearable_Hardware_Project/Mqtt_Data_Logger/new_joint_angles.csv")

# Define segment lengths
L = {
    'torso': 1.0,
    'upper_leg': 0.7,
    'lower_leg': 0.7,
    'foot': 0.3
}

def get_leg_positions(angles, side='right'):
    # Hip at origin
    x0, y0 = 0, 0
    # Hip angle
    theta_hip = np.deg2rad(angles[f'{side}_hip'])
    x1 = x0 + L['upper_leg'] * np.sin(theta_hip)
    y1 = y0 - L['upper_leg'] * np.cos(theta_hip)
    # Knee angle (relative to hip)
    theta_knee = theta_hip + np.deg2rad(angles[f'{side}_knee'])
    x2 = x1 + L['lower_leg'] * np.sin(theta_knee)
    y2 = y1 - L['lower_leg'] * np.cos(theta_knee)
    # Ankle angle (relative to knee)
    theta_ankle = theta_knee + np.deg2rad(angles.get(f'{side}_ankle', 0))
    x3 = x2 + L['foot'] * np.sin(theta_ankle)
    y3 = y2 - L['foot'] * np.cos(theta_ankle)
    return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]

fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Right leg, left leg, torso
right_leg_line, = ax.plot([], [], 'o-', lw=4, color='blue', label='Right Leg')
left_leg_line, = ax.plot([], [], 'o-', lw=4, color='red', label='Left Leg')
torso_line, = ax.plot([], [], 's-', lw=4, color='black', label='Torso')

def init():
    right_leg_line.set_data([], [])
    left_leg_line.set_data([], [])
    torso_line.set_data([], [])
    return right_leg_line, left_leg_line, torso_line

def update(frame):
    angles = joint_angles.iloc[frame]

    # Right leg
    right_leg = get_leg_positions(angles, 'right')
    rx, ry = zip(*right_leg)
    right_leg_line.set_data(rx, ry)

    # Left leg
    left_leg = get_leg_positions(angles, 'left')
    lx, ly = zip(*left_leg)
    left_leg_line.set_data(lx, ly)

    # Torso (stationary, hip at (0,0), torso top at (0, L['torso']))
    torso_line.set_data([0, 0], [0, L['torso']])

    return right_leg_line, left_leg_line, torso_line

ani = FuncAnimation(fig, update, frames=len(joint_angles),
                    init_func=init, blit=True, interval=50)

plt.legend()
plt.show()
