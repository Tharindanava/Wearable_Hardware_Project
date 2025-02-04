import cv2
import numpy as np
import socket
import struct
import math
import csv
import os

def Filter(image):
    # Load the image
    #image = cv2.imread(image)

    # Get the shape of the image
    height, width, _ = image.shape

    # Print the width and height of the image
    #print(width, height)

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the red color in the HSV color space
    # Lower mask (0-10)
    lower_red0 = np.array([10, 130, 140])
    upper_red0 = np.array([25, 205, 255])
    mask0 = cv2.inRange(hsv, lower_red0, upper_red0)

    # Upper mask (170-180)
    lower_red1 = np.array([170, 130, 140])
    upper_red1 = np.array([160, 205, 205])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    # Join the masks
    raw_mask = mask0 | mask1

    # Find contours
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize mask with zeros
    mask = np.zeros_like(raw_mask)
    points = [];

    idx = 0
    for c in contours:
        area = cv2.contourArea(c)  # Find the area of each contour
        if area > 50:  # Ignore small contours (assume noise).
            cv2.drawContours(mask, [c], 0, 255, -1)

            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(mask, center, radius, 255, -1)
            
            # Find the centroid of the red pixels
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])  # Centroid x coordinate
                cy = int(M['m01'] / M['m00'])  # Centroid y coordinate
                points.append((cx,cy))
                #print(f"Centroid of circle {idx}: ({cx}, {cy})")
                # Draw the centroid on the original image
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            else:
                print(f"Contour {idx} has zero area, skipping centroid calculation.")

            tmp_mask = np.zeros_like(mask)
            cv2.circle(tmp_mask, center, radius, 255, -1)
            output = cv2.bitwise_and(image, image, mask=tmp_mask)
            #resized_output = cv2.resize(output, (640, 480))
            cv2.namedWindow(f'output_{idx}', cv2.WINDOW_NORMAL)
            cv2.imshow(f'output_{idx}', output)  # Show output images for testing
            cv2.imwrite(f'output_{idx}.png', output)  # Save output images for testing
            idx += 1

    # Find the centroid of the red pixels
    #moments = cv2.moments(mask, True)
    #if moments['m00'] != 0:
    #    u = int(moments['m10'] / moments['m00'])
    #    v = int(moments['m01'] / moments['m00'])
    #    print("Image coordinates:", u, v)
    
    # Display the masks and the image
    #resized_mask_output = cv2.resize(mask, (640, 480))
    #resized_image = cv2.resize(image, (640, 480))
    """
    cv2.imshow('raw_mask', raw_mask)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.imshow('mask', mask)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows() """
    return points
    
def calculateAngle(x1, y1, x2, y2, x3, y3):
    ABx = x1 - x2;
    ABy = y1 - y2;
    BCx = x3 - x2;
    BCy = y3 - y2;
    dotProduct = (ABx * BCx + ABy * BCy );
    magnitudeAB = math.sqrt(ABx * ABx + ABy * ABy );
    magnitudeBC = math.sqrt(BCx * BCx +BCy * BCy);
    cosangle = dotProduct/(magnitudeAB * magnitudeBC);
    theta_radians = math.acos(cosangle);
    theta_degrees = math.degrees(theta_radians);
    print(round(abs(theta_degrees), 4))
    return theta_radians
    
def save_coordinates_to_csv(coordinates, filename="JointAngles.csv", write_header=True):
    file_exists = os.path.exists(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Optionally, write a header
        # Write the header only if the file does not already exist or is empty
        if write_header and not file_exists:
            writer.writerow(["Angle1", "Angle2"])
        elif write_header and file_exists:
            # Check if the file is empty or not
            file.seek(0, os.SEEK_END)
            if file.tell() == 0:  # File is empty
                writer.writerow(["Angle1", "Angle2"])
                
        
        # Write the coordinate pairs to the file
        #for coordinate in coordinates:
        writer.writerow(coordinates)
    
    print(f"Data has been written to {filename}")
"""
# Example usage
image = cv2.imread("D:/Z_Bio Med Height/Screenshot (38).png")
image_path = r"D:\Z_Bio Med Height\frames\frame_0039.jpg"
points = Filter(image_path)
#print(points)
x1, y1 = points[0];
#print(x1,y1)
    # Points B
x2, y2 = points[1];
    # Points C
x3, y3 = points[2];
    # Function Call
angle1 = calculateAngle(x1, y1, x2, y2, x3, y3);
x1, y1 = points[1];
#print(x1,y1)
    # Points B
x2, y2 = points[2];
    # Points C
x3, y3 = points[3];
    # Function Call
angle2 = calculateAngle(x1, y1, x2, y2, x3, y3);
print((angle1, angle2))
save_coordinates_to_csv((angle1, angle2), "D:/Z_Bio Med Height/Angles.csv")"""





def capture_and_display_frames(video_path, output_folder, frame_interval=30):
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    notenough = 0;
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        points = Filter(frame)
        print(points)
        if len(points) == 3:
            x1, y1 = points[0];
        #print(x1,y1)
        # Points B
            x2, y2 = points[1];
        # Points C
            x3, y3 = points[2];
        # Function Call
            angle1 = calculateAngle(x1, y1, x2, y2, x3, y3);
        
            #x1, y1 = points[1];
        #print(x1,y1)
        # Points B
            #x2, y2 = points[2];
        # Points C
            #x3, y3 = points[3];
        # Function Call
            #angle2 = calculateAngle(x1, y1, x2, y2, x3, y3);
            #print((angle1, angle2))
            
            print((angle1))
            save_coordinates_to_csv((angle1, 0), "C:/Users/acer/Documents/Accedemic_Folder_E19254/Wearable_Hardwear_Project/Sensor_Data/Validation/Thevindu_01_01_2025/frames/Angles.csv")
        else:
            print(f"not enough points in frame {saved_frame_count}")
            notenough += 1

        # Save and display the frame at the specified interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            saved_frame_count += 1

            #Display the frame in a window
            #cv2.imshow('Captured Frame', frame)
            
            # Wait for 1 millisecond to display the frame
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
        print(notenough)
        frame_count += 1

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"Frames have been saved to {output_folder}")

# Example usage:
video_path = r"C:/Users/acer/Documents/Accedemic_Folder_E19254/Wearable_Hardwear_Project/Sensor_Data/Validation/Thevindu_01_01_2025/Videos/ThevinduTrial2.mp4"
output_folder = r"C:/Users/acer/Documents/Accedemic_Folder_E19254/Wearable_Hardwear_Project/Sensor_Data/Validation/Thevindu_01_01_2025/frames"
capture_and_display_frames(video_path, output_folder, frame_interval=28)