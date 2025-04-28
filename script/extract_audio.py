import torch
import librosa
import numpy as np
import os
import csv
from transformers import WhisperProcessor, WhisperModel

def extract_audio_keypoints(audio_path, target_fps=5):
    """
    Extracts and downsamples Whisper-based audio keypoints from a given audio file.
    Returns a [N, 1] tensor and the timestamps.
    """
    print(f"Processing Audio: {audio_path}")

    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperModel.from_pretrained("openai/whisper-base")
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    print(f"Audio Duration: {duration:.2f} sec")

    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    with torch.no_grad():
        outputs = model.encoder(input_features)
        whisper_features = outputs.last_hidden_state[0]  

    whisper_features = whisper_features.numpy()
    whisper_fps = whisper_features.shape[0] / duration  
    whisper_timestamps = np.linspace(0, duration, whisper_features.shape[0])
    target_frames = int(duration * target_fps)
    target_timestamps = np.linspace(0, duration, target_frames)

 
    whisper_scalar = whisper_features.mean(axis=1)  


    resampled_features = np.interp(target_timestamps, whisper_timestamps, whisper_scalar)
    resampled_features = resampled_features.reshape(-1, 1) 

    print(f"Resampled {whisper_features.shape[0]} frames -> {target_frames} frames at {target_fps} FPS")
    print(resampled_features.shape)
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
