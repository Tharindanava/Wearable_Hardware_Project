import cv2
import os
from pathlib import Path

def create_video_from_images(image_dir, output_path='output_video.mp4', fps=10):
    # Convert to Path object for flexibility
    image_dir = Path(image_dir)

    # Get list of image files sorted by name
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not image_files:
        raise ValueError("No image files found in the directory.")

    # Read the first image to get the frame size
    first_image = cv2.imread(str(image_files[0]))
    height, width, _ = first_image.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating video: {output_path}")
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Skipping {img_path} (could not read)")
            continue
        # Resize image if needed to match the first frame
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
        video_writer.write(img)

    video_writer.release()
    print("Video creation completed.")

# Example usage
create_video_from_images("C:/Users/acer/Documents/GitHub/Wearable_Hardware_Project/Computer Vision/Trial_1_side_frames",
                        "C:/Users/acer/Documents/GitHub/Wearable_Hardware_Project/Computer Vision/Trial 1_side_analysis.mp4",
                        fps=30)
