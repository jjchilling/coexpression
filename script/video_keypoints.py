import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

KEYPOINT_GROUPS = {
    "eyebrows_left": list(range(52, 59)),
    "eyebrows_right": list(range(282, 291)),
    "chin": list(range(152, 160)),
    "cheeks_left": list(range(205, 209)),
    "cheeks_right": list(range(425, 429)),
    "mouth": list(range(0, 12)),
    "eyes_left": list(range(143, 156)),
    "eyes_right": list(range(373, 386)),
    "nose": list(range(1, 12)) + list(range(168, 187))
}

ALL_KEYPOINT_INDICES = {f"{group}_{i}": idx for group, indices in KEYPOINT_GROUPS.items() for i, idx in enumerate(indices)}

def preprocess_frame(frame):
    """Enhances and upscales the frame for better facial keypoint detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    enhanced_image = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return cv2.resize(enhanced_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

def extract_facial_keypoints(landmarks, indices, width, height):
    """Extracts x, y, z coordinates and visibility for each keypoint."""
    return {
        key: (
            int(landmarks[idx].x * width),
            int(landmarks[idx].y * height),
            landmarks[idx].z,
            getattr(landmarks[idx], 'visibility', 1.0)
        ) for key, idx in indices.items()
    }

def draw_key_points(image, key_points, color=(255, 0, 0)):
    """Draws the detected key points on the frame."""
    for x, y, _, _ in key_points.values():
        cv2.circle(image, (x, y), radius=2, color=color, thickness=-1)

if len(sys.argv) < 2:
    print("Usage: python keypoints.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    sys.exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = max(1, fps // 5)

output_csv = os.path.splitext(os.path.basename(video_path))[0] + "_keypoints.csv"

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["video_filepath", "frame_number", "feature", "x", "y", "z", "visibility"])

    with mp_face_mesh.FaceMesh(
        max_num_faces=10, 
        refine_landmarks=True,
        min_detection_confidence=0.3, 
        min_tracking_confidence=0.3
    ) as face_mesh:

        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            if frame_count % frame_interval == 0:  
                frame = preprocess_frame(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        width, height, _ = frame.shape
                        keypoints = extract_facial_keypoints(face_landmarks.landmark, ALL_KEYPOINT_INDICES, width, height)

                        for feature, (x, y, z, visibility) in keypoints.items():
                            writer.writerow([video_path, frame_count, feature, x, y, z, visibility])

                        draw_key_points(frame, keypoints, color=(0, 255, 0))

            frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"CSV saved as {output_csv}")
