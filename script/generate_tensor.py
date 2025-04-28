import os
import torch
import librosa
import numpy as np
import cv2
import mediapipe as mp
import csv
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel
import glob

# Load Whisper model once
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperModel.from_pretrained("openai/whisper-base")

# Mediapipe setup
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
    frames_info = []
    
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
                    row = [video_path, frame_idx]
                    for idx in ALL_KEYPOINTS:
                        x, y, z = lm[idx].x * w, lm[idx].y * h, lm[idx].z
                        vec.extend([x, y, z])
                        row.extend([x, y, z])
                    keypoints.append(vec)
                    frames_info.append(row)
            frame_idx += 1
    cap.release()

    return torch.tensor(keypoints, dtype=torch.float32), frames_info

def extract_audio_features(audio_path, target_fps=6):
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        out = model.encoder(input_features)
        features = out.last_hidden_state[0]

    target_frames = int(duration * target_fps)
    resampled = torch.nn.functional.interpolate(
        features.permute(1, 0).unsqueeze(0), size=target_frames, mode='linear'
    )[0].permute(1, 0)

    audio_scalar = resampled.mean(dim=1, keepdim=True)
    
    frames_info = []
    for idx in range(audio_scalar.shape[0]):
        frames_info.append([audio_path, idx, audio_scalar[idx].item()])

    return audio_scalar, frames_info

def fuse_modalities(video_tensor, audio_tensor):
    min_len = min(video_tensor.shape[0], audio_tensor.shape[0])
    return torch.cat((video_tensor[:min_len], audio_tensor[:min_len]), dim=1)

def save_csv(data, header, save_path):
    with open(save_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)
    print(f"Saved CSV: {save_path}")

def process_split_videos(video_folder, audio_folder, output_folder="fused_output", target_fps=6):
    os.makedirs(output_folder, exist_ok=True)

    
    video_files = [f for f in os.listdir(video_folder) if f.endswith("_left.mp4") or f.endswith("_right.mp4")]

    for video_file in tqdm(sorted(video_files), desc="Processing left/right speakers"):
        video_path = os.path.join(video_folder, video_file)

     
        if "_left" in video_file:
            base_name = video_file.replace("_left.mp4", "")
        elif "_right" in video_file:
            base_name = video_file.replace("_right.mp4", "")
        else:
            base_name = os.path.splitext(video_file)[0]

        audio_path = os.path.join(audio_folder, base_name + ".wav")


        if not os.path.exists(audio_path):
            print(f"Warning: No matching audio found for {video_file}")
            continue

        try:
            # Extract keypoints
            video_kp, video_frames_info = extract_video_keypoints(video_path, target_fps)
            audio_kp, audio_frames_info = extract_audio_features(audio_path, target_fps)
            fused = fuse_modalities(video_kp, audio_kp)

            # Save fused tensor
            fused_save_path = os.path.join(output_folder, video_file.replace(".mp4", "_fused.pt"))
            torch.save(fused, fused_save_path)
            print(f"Saved fused tensor: {fused_save_path}")

            # Save video keypoints CSV
            if video_frames_info:
                video_csv_path = os.path.join(output_folder, video_file.replace(".mp4", "_video_keypoints.csv"))
                video_header = ["video_filepath", "frame_number"] + [f"keypoint_{i}" for i in range(len(video_frames_info[0]) - 2)]
                save_csv(video_frames_info, video_header, video_csv_path)
            else:
                print(f"Warning: No video frames detected for {video_file}.")

            # Save audio keypoints CSV
            if audio_frames_info:
                audio_csv_path = os.path.join(output_folder, video_file.replace(".mp4", "_audio_keypoints.csv"))
                audio_header = ["audio_filepath", "frame_number", "audio_scalar"]
                save_csv(audio_frames_info, audio_header, audio_csv_path)
            else:
                print(f"Warning: No audio frames extracted for {video_file}.")

            print(f"Video shape: {video_kp.shape}, Audio shape: {audio_kp.shape}, Fused shape: {fused.shape}")

        except Exception as e:
            print(f"Error processing {video_file}: {e}")

if __name__ == "__main__":
    process_split_videos(
        video_folder="/Users/julie_chung/Desktop/coexpression/script/processed_videos", 
        audio_folder="/Users/julie_chung/Desktop/coexpression/script/audios",       
        output_folder="fused_tensors",            
        target_fps=6
    )
