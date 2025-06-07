import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor
from datasets import load_dataset

# Load the feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

# Load a sample audio from the dummy dataset
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
raw_audio = ds[0]["audio"]["array"]

# Extract features
inputs = feature_extractor(raw_audio, return_tensors="pt", sampling_rate=16000)
features = inputs.input_values[0]

# Create visualization
plt.figure(figsize=(15, 10))

# Plot raw audio
plt.subplot(2, 1, 1)
plt.plot(raw_audio)
plt.title("Raw Audio")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

# Plot extracted features
plt.subplot(2, 1, 2)
plt.plot(features.numpy())
plt.title("Extracted Features")
plt.xlabel("Time")
plt.ylabel("Feature Value")

plt.tight_layout()
plt.savefig("feature_comparison.png")
plt.close()

# Print some statistics
print(f"Raw audio shape: {raw_audio.shape}")
print(f"Extracted features shape: {features.shape}")
print(f"Raw audio mean: {np.mean(raw_audio):.3f}")
print(f"Extracted features mean: {torch.mean(features):.3f}")
print(f"Raw audio std: {np.std(raw_audio):.3f}")
print(f"Extracted features std: {torch.std(features):.3f}") 