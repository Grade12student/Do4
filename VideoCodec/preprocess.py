import cv2
import os

# Load video using OpenCV VideoCapture
video_path = "cutter.mp4"
cap = cv2.VideoCapture(video_path)

# Create a directory to save the frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Extract frames and resize
frame_count = 0
clip_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (832, 448))  # Resize to desired dimensions
    frame_count += 1

    # Divide into clips of 13 frames
    if frame_count % 13 == 0:
        clip_count += 1

    # Save the frames as individual files
    cv2.imwrite(f"frames/frame_{clip_count}_{frame_count}.png", frame)

cap.release()
cv2.destroyAllWindows()

