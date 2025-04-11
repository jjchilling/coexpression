import torch
import matplotlib.pyplot as plt

tensor = torch.load("/Users/julie_chung/Desktop/coexpression/script/fused_tensors/Ses04F_impro01_fused.pt")

video_part = tensor[:, :240]
audio_part = tensor[:, 240:]

video_energy = video_part.norm(dim=1)
audio_energy = audio_part.norm(dim=1)

plt.plot(video_energy, label="Visual Activity")
plt.plot(audio_energy, label="Audio Activity")
plt.title("Visual vs. Audio Feature Magnitude Over Time")
plt.legend()
plt.show()