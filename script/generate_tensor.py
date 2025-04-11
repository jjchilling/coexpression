import os
import torch
import librosa
import numpy as np
import cv2
import mediapipe as mp
from transformers import WhisperProcessor, WhisperModel
from tqdm import tqdm

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperModel.from_pretrained("openai/whisper-base")

mp_face_mesh = mp.solutions.face_mesh
KEYPOINT_INDICES = {
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
ALL_KEYPOINTS = [idx for indices in KEYPOINT_INDICES.values() for idx in indices]

def extract_video_keypoints(video_path, target_fps=6):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(video_fps / target_fps))
    keypoints = []

    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.3, min_tracking_confidence=0.3
    ) as face_mesh:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    w, h = frame.shape[1], frame.shape[0]
                    vec = []
                    for idx in ALL_KEYPOINTS:
                        vec.extend([lm[idx].x * w, lm[idx].y * h, lm[idx].z])
                    keypoints.append(vec)
            frame_idx += 1
    cap.release()
    return torch.tensor(keypoints, dtype=torch.float32)

def extract_audio_features(wav_path, target_fps=6):
    audio, sr = librosa.load(wav_path, sr=16000)
    duration = len(audio) / sr
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        out = model.encoder(input_features)
        # [T, D]
        features = out.last_hidden_state[0] 
    target_frames = int(duration * target_fps)
    resampled = torch.nn.functional.interpolate(
        features.permute(1, 0).unsqueeze(0), size=target_frames, mode='linear'
    )[0].permute(1, 0)
    return resampled

def fuse_modalities(video_tensor, audio_tensor):
    min_len = min(video_tensor.shape[0], audio_tensor.shape[0])
    return torch.cat((video_tensor[:min_len], audio_tensor[:min_len]), dim=1)

def process_video_audio_pairs(video_folder, audio_folder, output_folder="fused_output", target_fps=6):
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
    paired_names = set(os.path.splitext(f)[0] for f in video_files) & set(os.path.splitext(f)[0] for f in audio_files)

    if not paired_names:
        print("No matching .mp4/.wav pairs found.")
        return

    for base in tqdm(sorted(paired_names), desc="Processing pairs"):
        video_path = os.path.join(video_folder, base + ".mp4")
        audio_path = os.path.join(audio_folder, base + ".wav")

        try:
            video_kp = extract_video_keypoints(video_path, target_fps)
            audio_kp = extract_audio_features(audio_path, target_fps)
            fused = fuse_modalities(video_kp, audio_kp)

            output_path = os.path.join(output_folder, base + "_fused.pt")
            torch.save(fused, output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {base}: {e}")

if __name__ == "__main__":
    process_video_audio_pairs(
        video_folder="/Users/julie_chung/Desktop/coexpression/script/processed_videos",   
        audio_folder="/Users/julie_chung/Desktop/coexpression/script/audios",   
        output_folder="fused_tensors", 
        target_fps=6
    )
