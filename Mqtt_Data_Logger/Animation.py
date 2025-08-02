import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import os

class WalkingAnimation:
    def __init__(self, joint_angles_file, segment_lengths=None):
        """
        Initialize the walking animation with joint angle data.
        
        Args:
            joint_angles_file (str): Path to CSV file containing joint angles
            segment_lengths (dict): Dictionary of body segment lengths in meters
        """
        # Load joint angle data
        self.df = pd.read_csv(joint_angles_file)
        
        # Set default segment lengths if not provided
        if segment_lengths is None:
            self.segment_lengths = {
                'pelvis': 0.2,      # Width of pelvis
                'thigh': 0.4,       # Length of thigh
                'shank': 0.4,       # Length of shank
                'foot': 0.2,        # Length of foot
                'torso': 0.5,       # Length of torso
                'upper_arm': 0.3,   # Length of upper arm
                'lower_arm': 0.3    # Length of lower arm
            }
        else:
            self.segment_lengths = segment_lengths
        
        # Extract time and angle data
        self.time = self.df['time'].values
        self.dt = np.mean(np.diff(self.time))
        
        # Initialize figure for animation
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set axis limits
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Walking Pattern Animation')
        
        # Initialize body segments
        self.lines = []
        self.points = []
        
        # Colors for different body parts
        self.colors = {
            'pelvis': 'black',
            'torso': 'blue',
            'thigh': 'red',
            'shank': 'green',
            'foot': 'purple',
            'arm': 'orange'
        }
        
        # Create mapping from column names to our expected names
        self.column_mapping = self._create_column_mapping()
    
    def _create_column_mapping(self):
        """Create mapping from available columns to expected angle names."""
        mapping = {}
        available_columns = set(self.df.columns)
        
        # Try to map available columns to expected names
        expected_columns = [
            'Right_hip_roll', 'Right_hip_pitch', 'Right_knee_pitch', 'Right_ankle_pitch',
            'Left_hip_roll', 'Left_hip_pitch', 'Left_knee_pitch', 'Left_ankle_pitch',
            'Torso_pitch', 'Torso_roll',
            'Right_shoulder_pitch', 'Right_elbow_pitch',
            'Left_shoulder_pitch', 'Left_elbow_pitch'
        ]
        
        for col in expected_columns:
            # Try exact match first
            if col in available_columns:
                mapping[col] = col
            else:
                # Try case-insensitive match
                lower_col = col.lower()
                for avail_col in available_columns:
                    if avail_col.lower() == lower_col:
                        mapping[col] = avail_col
                        break
                else:
                    # Try partial match
                    for avail_col in available_columns:
                        if col.split('_')[-1].lower() in avail_col.lower():
                            mapping[col] = avail_col
                            break
        
        return mapping
    
    def _get_angle(self, frame, angle_name, default=0):
        """Get angle value with fallback to default if column not found."""
        if angle_name in self.column_mapping:
            return np.deg2rad(self.df.at[frame, self.column_mapping[angle_name]])
        return np.deg2rad(default)
    
    def create_skeleton(self):
        """Initialize the skeleton structure with lines and points."""
        # Clear any existing lines/points
        for line in self.lines:
            line.remove()
        for point in self.points:
            point.remove()
        self.lines = []
        self.points = []
        
        # Pelvis (a small box)
        pelvis_line1, = self.ax.plot([], [], [], 'o-', color=self.colors['pelvis'], lw=2)
        pelvis_line2, = self.ax.plot([], [], [], 'o-', color=self.colors['pelvis'], lw=2)
        
        # Torso
        torso_line, = self.ax.plot([], [], [], 'o-', color=self.colors['torso'], lw=3)
        
        # Legs (right and left)
        r_thigh_line, = self.ax.plot([], [], [], 'o-', color=self.colors['thigh'], lw=2)
        r_shank_line, = self.ax.plot([], [], [], 'o-', color=self.colors['shank'], lw=2)
        r_foot_line, = self.ax.plot([], [], [], 'o-', color=self.colors['foot'], lw=2)
        
        l_thigh_line, = self.ax.plot([], [], [], 'o-', color=self.colors['thigh'], lw=2)
        l_shank_line, = self.ax.plot([], [], [], 'o-', color=self.colors['shank'], lw=2)
        l_foot_line, = self.ax.plot([], [], [], 'o-', color=self.colors['foot'], lw=2)
        
        # Arms (right and left)
        r_upper_arm_line, = self.ax.plot([], [], [], 'o-', color=self.colors['arm'], lw=2)
        r_lower_arm_line, = self.ax.plot([], [], [], 'o-', color=self.colors['arm'], lw=2)
        
        l_upper_arm_line, = self.ax.plot([], [], [], 'o-', color=self.colors['arm'], lw=2)
        l_lower_arm_line, = self.ax.plot([], [], [], 'o-', color=self.colors['arm'], lw=2)
        
        # Head
        head_point, = self.ax.plot([], [], [], 'o', color='yellow', markersize=10)
        
        # Store all lines and points
        self.lines.extend([
            pelvis_line1, pelvis_line2, torso_line,
            r_thigh_line, r_shank_line, r_foot_line,
            l_thigh_line, l_shank_line, l_foot_line,
            r_upper_arm_line, r_lower_arm_line,
            l_upper_arm_line, l_lower_arm_line,
            head_point
        ])
        
        return self.lines
    
    def update_skeleton(self, frame):
        """Update the skeleton positions based on current frame's angles."""
        # Get current angles using the mapping
        r_hip_roll = self._get_angle(frame, 'Right_hip_roll')
        r_hip_pitch = self._get_angle(frame, 'Right_hip_pitch')
        r_knee_pitch = self._get_angle(frame, 'Right_knee_pitch')
        r_ankle_pitch = self._get_angle(frame, 'Right_ankle_pitch')
        
        l_hip_roll = self._get_angle(frame, 'Left_hip_roll')
        l_hip_pitch = self._get_angle(frame, 'Left_hip_pitch')
        l_knee_pitch = self._get_angle(frame, 'Left_knee_pitch')
        l_ankle_pitch = self._get_angle(frame, 'Left_ankle_pitch')
        
        torso_pitch = self._get_angle(frame, 'Torso_pitch')
        torso_roll = self._get_angle(frame, 'Torso_roll')
        
        r_shoulder_pitch = self._get_angle(frame, 'Right_shoulder_pitch')
        r_elbow_pitch = self._get_angle(frame, 'Right_elbow_pitch')
        
        l_shoulder_pitch = self._get_angle(frame, 'Left_shoulder_pitch')
        l_elbow_pitch = self._get_angle(frame, 'Left_elbow_pitch')
        
        # Pelvis position (center of the body)
        pelvis_pos = np.array([0, 0, self.segment_lengths['thigh'] + self.segment_lengths['shank']])
        
        # Pelvis (a small box)
        pelvis_width = self.segment_lengths['pelvis']
        pelvis_points = np.array([
            [pelvis_pos[0] - pelvis_width/2, pelvis_pos[1], pelvis_pos[2]],
            [pelvis_pos[0] + pelvis_width/2, pelvis_pos[1], pelvis_pos[2]]
        ])
        
        # Torso (from pelvis to shoulders)
        torso_rot = Rotation.from_euler('xyz', [torso_roll, torso_pitch, 0])
        torso_end = pelvis_pos + torso_rot.apply([0, 0, self.segment_lengths['torso']])
        torso_points = np.vstack([pelvis_pos, torso_end])
        
        # Head (on top of torso)
        head_pos = torso_end
        
        # Right leg
        r_hip_rot = Rotation.from_euler('xyz', [r_hip_roll, r_hip_pitch, 0])
        r_thigh_end = pelvis_points[1] + r_hip_rot.apply([0, 0, -self.segment_lengths['thigh']])
        
        r_knee_rot = Rotation.from_euler('x', [r_knee_pitch])
        r_shank_end = r_thigh_end + r_knee_rot.apply([0, 0, -self.segment_lengths['shank']])
        
        r_ankle_rot = Rotation.from_euler('x', [r_ankle_pitch])
        r_foot_end = r_shank_end + r_ankle_rot.apply([0, 0, -self.segment_lengths['foot']])
        
        # Left leg
        l_hip_rot = Rotation.from_euler('xyz', [l_hip_roll, l_hip_pitch, 0])
        l_thigh_end = pelvis_points[0] + l_hip_rot.apply([0, 0, -self.segment_lengths['thigh']])
        
        l_knee_rot = Rotation.from_euler('x', [l_knee_pitch])
        l_shank_end = l_thigh_end + l_knee_rot.apply([0, 0, -self.segment_lengths['shank']])
        
        l_ankle_rot = Rotation.from_euler('x', [l_ankle_pitch])
        l_foot_end = l_shank_end + l_ankle_rot.apply([0, 0, -self.segment_lengths['foot']])
        
        # Right arm
        r_shoulder_rot = Rotation.from_euler('x', [r_shoulder_pitch])
        r_upper_arm_end = torso_end + r_shoulder_rot.apply([0, -self.segment_lengths['upper_arm'], 0])
        
        r_elbow_rot = Rotation.from_euler('x', [r_elbow_pitch])
        r_lower_arm_end = r_upper_arm_end + r_elbow_rot.apply([0, -self.segment_lengths['lower_arm'], 0])
        
        # Left arm
        l_shoulder_rot = Rotation.from_euler('x', [l_shoulder_pitch])
        l_upper_arm_end = torso_end + l_shoulder_rot.apply([0, self.segment_lengths['upper_arm'], 0])
        
        l_elbow_rot = Rotation.from_euler('x', [l_elbow_pitch])
        l_lower_arm_end = l_upper_arm_end + l_elbow_rot.apply([0, self.segment_lengths['lower_arm'], 0])
        
        # Update all lines and points
        # Pelvis
        self.lines[0].set_data(pelvis_points[:, 0], pelvis_points[:, 1])
        self.lines[0].set_3d_properties(pelvis_points[:, 2])
        
        self.lines[1].set_data(pelvis_points[:, 0], pelvis_points[:, 1])
        self.lines[1].set_3d_properties(pelvis_points[:, 2])
        
        # Torso
        self.lines[2].set_data(torso_points[:, 0], torso_points[:, 1])
        self.lines[2].set_3d_properties(torso_points[:, 2])
        
        # Right leg
        thigh_points = np.vstack([pelvis_points[1], r_thigh_end])
        self.lines[3].set_data(thigh_points[:, 0], thigh_points[:, 1])
        self.lines[3].set_3d_properties(thigh_points[:, 2])
        
        shank_points = np.vstack([r_thigh_end, r_shank_end])
        self.lines[4].set_data(shank_points[:, 0], shank_points[:, 1])
        self.lines[4].set_3d_properties(shank_points[:, 2])
        
        foot_points = np.vstack([r_shank_end, r_foot_end])
        self.lines[5].set_data(foot_points[:, 0], foot_points[:, 1])
        self.lines[5].set_3d_properties(foot_points[:, 2])
        
        # Left leg
        thigh_points = np.vstack([pelvis_points[0], l_thigh_end])
        self.lines[6].set_data(thigh_points[:, 0], thigh_points[:, 1])
        self.lines[6].set_3d_properties(thigh_points[:, 2])
        
        shank_points = np.vstack([l_thigh_end, l_shank_end])
        self.lines[7].set_data(shank_points[:, 0], shank_points[:, 1])
        self.lines[7].set_3d_properties(shank_points[:, 2])
        
        foot_points = np.vstack([l_shank_end, l_foot_end])
        self.lines[8].set_data(foot_points[:, 0], foot_points[:, 1])
        self.lines[8].set_3d_properties(foot_points[:, 2])
        
        # Right arm
        upper_arm_points = np.vstack([torso_end, r_upper_arm_end])
        self.lines[9].set_data(upper_arm_points[:, 0], upper_arm_points[:, 1])
        self.lines[9].set_3d_properties(upper_arm_points[:, 2])
        
        lower_arm_points = np.vstack([r_upper_arm_end, r_lower_arm_end])
        self.lines[10].set_data(lower_arm_points[:, 0], lower_arm_points[:, 1])
        self.lines[10].set_3d_properties(lower_arm_points[:, 2])
        
        # Left arm
        upper_arm_points = np.vstack([torso_end, l_upper_arm_end])
        self.lines[11].set_data(upper_arm_points[:, 0], upper_arm_points[:, 1])
        self.lines[11].set_3d_properties(upper_arm_points[:, 2])
        
        lower_arm_points = np.vstack([l_upper_arm_end, l_lower_arm_end])
        self.lines[12].set_data(lower_arm_points[:, 0], lower_arm_points[:, 1])
        self.lines[12].set_3d_properties(lower_arm_points[:, 2])
        
        # Head
        self.lines[13].set_data([head_pos[0]], [head_pos[1]])
        self.lines[13].set_3d_properties([head_pos[2]])
        
        return self.lines
    
    def animate(self, save_path=None, fps=30):
        """Create and run the animation."""
        # Initialize skeleton
        self.create_skeleton()
        
        # Create animation
        frames = len(self.df)
        interval = 1000 / fps  # ms per frame
        
        anim = FuncAnimation(
            self.fig, 
            self.update_skeleton, 
            frames=frames, 
            interval=interval, 
            blit=False  # Changed to False to prevent issues with 3D plotting
        )
        
        # Save animation if path is provided
        if save_path:
            try:
                anim.save(save_path, writer='ffmpeg', fps=fps)
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Could not save animation: {e}")
        
        plt.tight_layout()
        plt.show()
        return anim

# Example usage
if __name__ == "__main__":
    # Initialize with your joint angle data file
    animator = WalkingAnimation(
        joint_angles_file="G:/Tharida_Walking_Patter_01_relative.csv",
        segment_lengths={
            'pelvis': 0.2,
            'thigh': 0.4,
            'shank': 0.4,
            'foot': 0.2,
            'torso': 0.5,
            'upper_arm': 0.3,
            'lower_arm': 0.3
        }
    )
    
    # Create and display animation
    animator.animate(
        save_path="G:/walking_animation.mp4",
        fps=30
    )