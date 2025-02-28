import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# mediapipe facemesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# all the key facial features (adjustable)
EYEBROWS_LEFT = list(range(52, 59))
EYEBROWS_RIGHT = list(range(282, 291))
CHIN = list(range(152, 160))
CHEEKS_LEFT = list(range(205, 209))
CHEEKS_RIGHT = list(range(425, 429))
MOUTH = list(range(0, 12))
EYES_LEFT = list(range(143, 156))
EYES_RIGHT = list(range(373, 386))
NOSE = list(range(1, 12)) + list(range(168, 187))

ALL_KEYPOINT_INDICES = (
    EYEBROWS_LEFT + EYEBROWS_RIGHT + CHIN +
    CHEEKS_LEFT + CHEEKS_RIGHT + MOUTH +
    EYES_LEFT + EYES_RIGHT + NOSE
)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    enhanced_image = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    upscaled_image = cv2.resize(enhanced_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    return upscaled_image

def extract_facial_keypoints(landmarks, indices):
    return [
        (
            float(landmarks[idx].x),
            float(landmarks[idx].y),
            float(landmarks[idx].z),
            float(getattr(landmarks[idx], 'visibility', 1.0))
        ) for idx in indices
    ]

def draw_key_points(image, key_points, color=(255, 0, 0)):
    height, width, _ = image.shape
    for x, y, _, _ in key_points:
        pixel_x = int(x * width)
        pixel_y = int(y * height)
        cv2.circle(image, (pixel_x, pixel_y), radius=2, color=color, thickness=-1)

if len(sys.argv) < 2:
    print("Usage: python keypoints.py <video_path>")
    sys.exit(1)

# sys arg instead
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    sys.exit(1)

with mp_face_mesh.FaceMesh(
    max_num_faces=10, 
    refine_landmarks=True,
    min_detection_confidence=0.3, 
    min_tracking_confidence=0.3
) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = preprocess_frame(frame)
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        frame.flags.writeable = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                keypoints = extract_facial_keypoints(face_landmarks.landmark, ALL_KEYPOINT_INDICES)
                draw_key_points(frame, keypoints, color=(0, 255, 0)) 

        cv2.imshow(f'Face Keypoints - {os.path.basename(video_path)}', cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
