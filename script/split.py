import cv2
import os
import glob
import numpy as np
import subprocess

input_folder = "videos" 
output_folder = "processed_videos" 
os.makedirs(output_folder, exist_ok=True)

def detect_black_borders(frame):
    """Automatically detect and remove black bars (top and bottom)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    vertical_projection = np.sum(thresh, axis=1)

    top_crop = np.argmax(vertical_projection > 10)
    bottom_crop = len(vertical_projection) - np.argmax(np.flip(vertical_projection) > 10)

    return top_crop, bottom_crop

# join video files together
video_files = glob.glob(os.path.join(input_folder, "*.mp4"))

if not video_files:
    print("No videos found in the folder!")
    exit()

for video_path in video_files:
    print(f"Processing {video_path}...")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    left_video_path = os.path.join(output_folder, f"{base_name}_left.mp4")
    right_video_path = os.path.join(output_folder, f"{base_name}_right.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        continue

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Could not read {video_path}")
        cap.release()
        continue

    top_crop, bottom_crop = detect_black_borders(first_frame)
    print(f"Detected black bars - Top: {top_crop}px, Bottom: {frame_height - bottom_crop}px")

    cropped_height = bottom_crop - top_crop

    # define output
    left_writer = cv2.VideoWriter(left_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width // 2, cropped_height))
    right_writer = cv2.VideoWriter(right_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width // 2, cropped_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        frame_cropped = frame[top_crop:bottom_crop, :]

        # left and right
        left_half = frame_cropped[:, :frame_width // 2]
        right_half = frame_cropped[:, frame_width // 2:]
      
        left_writer.write(left_half)
        right_writer.write(right_half)

    cap.release()
    left_writer.release()
    right_writer.release()

    print(f"Saved cropped videos (without black bars):\n  - {left_video_path}\n  - {right_video_path}")

    # running subprocesses (keypoints.py)
    print("Running keypoint.py on cropped videos...")
    subprocess.run(["python", "keypoints.py", left_video_path])
    subprocess.run(["python", "keypoints.py", right_video_path])

print("All videos processed successfully!")
