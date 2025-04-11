import torch
import librosa
import numpy as np
import os
import csv
from transformers import WhisperProcessor, WhisperModel

def extract_audio_keypoints(audio_path, target_fps=5):
    """
    Extracts Whisper-based audio keypoints from a given audio file.
    """
    print(f"Processing Audio: {audio_path}")

    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperModel.from_pretrained("openai/whisper-base")
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / 16000
    print(f"Audio Duration: {duration:.2f} sec")
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        outputs = model.encoder(input_features)
        whisper_features = outputs.last_hidden_state[0].numpy() 

    whisper_fps = 50 
    whisper_frames = whisper_features.shape[0]

    whisper_timestamps = np.linspace(0, duration, whisper_frames)

    target_frames = int(duration * target_fps)
    target_timestamps = np.linspace(0, duration, target_frames)

    resampled_features = np.interp(
        target_timestamps, whisper_timestamps, whisper_features[:, 0]  
    )

    print(f"Resampled {whisper_frames} frames -> {target_frames} frames at {target_fps} FPS")

    return resampled_features, target_timestamps

def save_audio_keypoints(audio_path, output_csv, target_fps=5):
    """ Extracts and saves audio keypoints to a CSV file. """
    features, timestamps = extract_audio_keypoints(audio_path, target_fps)

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["audio_filepath", "timestamp", "feature_value"])

        for i, (timestamp, feature) in enumerate(zip(timestamps, features)):
            writer.writerow([audio_path, timestamp, feature])

    print(f"Audio keypoints saved to {output_csv}")

if __name__ == "__main__":
    audio_file = "/Users/julie_chung/Desktop/coexpression/script/Ses02F_impro01.wav"
    output_file = audio_file.replace(".wav", "_audio_keypoints.csv")
    save_audio_keypoints(audio_file, output_file, target_fps=5)
