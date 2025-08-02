import os
import cv2
import numpy as np
from ultralytics import YOLO
import math
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time

# Load the YOLOv8 pose estimation model
model = YOLO('yolov8x-pose.pt')  # Replace with your preferred model

# Define keypoint names (COCO keypoints)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Define connections between keypoints for drawing
SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7)
]

# Configuration for angle calculations
ANGLE_CONFIG = {
    # Format: 
    # "angle_name": (joint1, joint2, joint3), 
    # where joint2 is the vertex of the angle
    "right_elbow": (10, 8, 6),       # right wrist-elbow-shoulder
    "left_elbow": (9, 7, 5),         # left wrist-elbow-shoulder
    "right_knee": (16, 14, 12),      # right ankle-knee-hip
    "left_knee": (15, 13, 11),       # left ankle-knee-hip
    "right_hip_front": (14, 12, 6),  # right knee-hip-shoulder (front view)
    "left_hip_front": (13, 11, 5),   # left knee-hip-shoulder (front view)
    "right_hip_side": (14, 12, 13),  # right knee-hip-left hip (side view)
    "left_hip_side": (13, 11, 12),   # left knee-hip-right hip (side view)
    "right_shoulder": (8, 6, 12),    # right elbow-shoulder-hip
    "left_shoulder": (7, 5, 11),     # left elbow-shoulder-hip
    "torso_angle_right": (6, 12, 14), # right shoulder-hip-knee (side view)
    "torso_angle_left": (5, 11, 13),  # left shoulder-hip-knee (side view)
    "back_angle_right": (6, 12, 13),  # right shoulder-hip-opposite hip (back arch)
    "back_angle_left": (5, 11, 12),   # left shoulder-hip-opposite hip (back arch)
}

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate the angle between three points (b is the vertex)"""
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Avoid numerical errors
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

def get_angle_config(view: str = 'front', side: str = 'right') -> Dict[str, Tuple[int, int, int]]:
    """Get angle configuration based on view type and side preference"""
    base_angles = {
        'right_elbow': ANGLE_CONFIG['right_elbow'],
        'left_elbow': ANGLE_CONFIG['left_elbow'],
        'right_knee': ANGLE_CONFIG['right_knee'],
        'left_knee': ANGLE_CONFIG['left_knee'],
    }
    
    if view.lower() == 'side':
        if side.lower() == 'right':
            base_angles.update({
                'right_hip': ANGLE_CONFIG['right_hip_side'],
                'torso_angle': ANGLE_CONFIG['torso_angle_right'],
                'back_angle': ANGLE_CONFIG['back_angle_right']
            })
        else:  # left side
            base_angles.update({
                'left_hip': ANGLE_CONFIG['left_hip_side'],
                'torso_angle': ANGLE_CONFIG['torso_angle_left'],
                'back_angle': ANGLE_CONFIG['back_angle_left']
            })
    else:  # front view
        base_angles.update({
            'right_hip': ANGLE_CONFIG['right_hip_front'],
            'left_hip': ANGLE_CONFIG['left_hip_front'],
            'right_shoulder': ANGLE_CONFIG['right_shoulder'],
            'left_shoulder': ANGLE_CONFIG['left_shoulder']
        })
    
    return base_angles

def process_video(
    video_path: str,
    output_folder: str = 'output_frames',  # Changed from output_path to output_folder
    view: str = 'front',
    side: str = 'right',
    selected_angles: Optional[List[str]] = None,
    show_video: bool = False,
    display_scale: float = 1.5,
    show_progress: bool = True,
    save_frames: bool = True,  # Added option to control frame saving
    frame_format: str = 'jpg'  # Added option for image format
) -> pd.DataFrame:
    """
    Process the video to extract selected joint angles and save individual frames
    
    Args:
        video_path: Path to input video
        output_folder: Folder to save output frames
        view: 'front' or 'side' view of athlete
        side: 'left' or 'right' (for side view)
        selected_angles: List of angles to calculate
        show_video: Whether to display video during processing
        display_scale: Scale factor for display window
        show_progress: Whether to show progress bar
        save_frames: Whether to save individual frames
        frame_format: Image format for saved frames ('jpg', 'png', etc.)
    """
    # Initialize model and video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Create output folder if needed
    if save_frames:
        os.makedirs(output_folder, exist_ok=True)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize data storage
    angles_data = {'frame': []}
    angle_config = get_angle_config(view, side)
    if selected_angles is not None:
        angle_config = {k: v for k, v in angle_config.items() if k in selected_angles}
    for angle_name in angle_config.keys():
        angles_data[angle_name] = []
    
    # Create window if displaying
    if show_video:
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
    
    # Initialize progress bar
    if show_progress:
        pbar = tqdm(total=total_frames, desc="Processing Video")
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process frame
        results = model(frame, conf=0.5)
        processed_frame = frame.copy()
        
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            
            # Draw skeleton and keypoints
            for i, j in SKELETON:
                if i < len(keypoints) and j < len(keypoints):
                    start = tuple(map(int, keypoints[i]))
                    end = tuple(map(int, keypoints[j]))
                    cv2.line(processed_frame, start, end, (255, 0, 0), 2)
            for i, kp in enumerate(keypoints):
                x, y = map(int, kp)
                cv2.circle(processed_frame, (x, y), 5, (0, 255, 0), -1)
            
            # Calculate angles
            text_y = 50
            for angle_name, (j1, j2, j3) in angle_config.items():
                if (j1 < len(keypoints) and (j2 < len(keypoints)) and (j3 < len(keypoints))):
                    if all(keypoints[j][0] > 0 and keypoints[j][1] > 0 for j in [j1, j2, j3]):
                        angle = calculate_angle(keypoints[j1], keypoints[j2], keypoints[j3])
                        angles_data[angle_name].append(angle)
                        display_name = angle_name.replace('_', ' ').title()
                        cv2.putText(processed_frame, f"{display_name}: {angle:.1f}°", 
                                   (50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1.5, (0, 255, 255), 2)
                        text_y += 30
        
        # Save individual frame if enabled
        if save_frames:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.{frame_format}")
            cv2.imwrite(frame_filename, processed_frame)
        
        # Optional display
        if show_video:
            display_frame = cv2.resize(processed_frame, None, fx=display_scale, fy=display_scale)
            cv2.imshow('Pose Estimation', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Update progress
        if show_progress:
            pbar.update(1)
    
    # Cleanup
    cap.release()
    if show_video:
        cv2.destroyAllWindows()
    if show_progress:
        pbar.close()
    
    # Calculate processing stats
    processing_time = time.time() - start_time
    if show_progress:
        print(f"\nProcessing complete. {frame_count} frames processed in {processing_time:.2f} seconds")
        if save_frames:
            print(f"Frames saved to: {output_folder}")
    
    # Ensure all angles have same length as frames
    angles_data['frame'] = list(range(1, frame_count + 1))
    for angle_name in angle_config.keys():
        if len(angles_data[angle_name]) < frame_count:
            angles_data[angle_name].extend([np.nan] * (frame_count - len(angles_data[angle_name])))
    
    return pd.DataFrame(angles_data)

def plot_selected_angles(angles_df: pd.DataFrame, selected_angles: List[str]):
    """Plot selected angles over time"""
    if not selected_angles:
        selected_angles = [col for col in angles_df.columns if col != 'frame']
    
    plt.figure(figsize=(12, 6))
    
    for angle_name in selected_angles:
        if angle_name in angles_df.columns:
            plt.plot(angles_df['frame'], angles_df[angle_name], 
                    label=angle_name.replace('_', ' ').title(),
                    linewidth=2)
    
    plt.title('Joint Angles Over Time', fontsize=14)
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example usage with frame saving
if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "C:/Users/acer/Documents/GitHub/Wearable_Hardware_Project/Computer Vision/Trial_3_front.MOV"
    OUTPUT_FOLDER = 'C:/Users/acer/Documents/GitHub/Wearable_Hardware_Project/Computer Vision/Trial_3_front_frames'
    VIEW_TYPE = 'front'  # 'front' or 'side'
    SIDE = 'left'      # 'left' or 'right'
    
    # Processing options
    SHOW_VIDEO = False
    SHOW_PROGRESS = True
    SAVE_FRAMES = True
    FRAME_FORMAT = 'jpg'  # or 'png'
    
    # Select angles based on view
    if VIEW_TYPE == 'side':
        SELECTED_ANGLES = ['right_elbow', 'right_knee','right_shoulder', 'torso_angle'] if SIDE == 'right' else \
                         ['left_elbow', 'left_knee','left_shoulder', 'torso_angle']
    else:
        SELECTED_ANGLES = None  # All front view angles
    
    # Process video
    angles_df = process_video(
        video_path=VIDEO_PATH,
        output_folder=OUTPUT_FOLDER,
        view=VIEW_TYPE,
        side=SIDE,
        selected_angles=SELECTED_ANGLES,
        show_video=SHOW_VIDEO,
        show_progress=SHOW_PROGRESS,
        save_frames=SAVE_FRAMES,
        frame_format=FRAME_FORMAT
    )
    
    # Save and plot results
    angles_df.to_csv('Trial_3_front_joint_angles.csv', index=False)
    plot_selected_angles(angles_df, SELECTED_ANGLES)