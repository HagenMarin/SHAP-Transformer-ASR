import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from datasets import load_dataset
import logging
import sys
import os
import soundfile as sf

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# For LIME
from lime.lime_tabular import LimeTabularExplainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sound Amplification Settings
amplification_factor = 2.0

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load pretrained wav2vec2 model and processor
logger.info("Loading wav2vec2 model and processor...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

# Load sample audio
logger.info("Loading dataset...")
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
raw_audio = ds[2]["audio"]["array"]
sample_rate = ds[2]["audio"]["sampling_rate"]

# Resample to 16kHz if needed
if sample_rate != 16000:
    logger.info(f"Resampling audio from {sample_rate} to 16000 Hz.")
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    raw_audio = resampler(torch.from_numpy(raw_audio).float()).numpy()
    sample_rate = 16000

# Truncate or pad audio to 1 second (16000 samples)
target_length = 32000  # 2 seconds for better LIME/SHAP analysis
if len(raw_audio) > target_length:
    start = (len(raw_audio) - target_length) // 2
    raw_audio = raw_audio[start:start + target_length]
elif len(raw_audio) < target_length:
    raw_audio = np.pad(raw_audio, (0, target_length - len(raw_audio)), mode='constant')

logger.info(f"Audio shape after processing: {raw_audio.shape}")

# Prepare input for wav2vec2
input_values = processor(raw_audio, sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)

# Define a prediction function for LIME
def lime_predict_fn(inputs):
    """
    inputs: numpy array of shape (n_samples, n_features)
    Returns: numpy array of shape (n_samples, n_outputs)
    """
    inputs_tensor = torch.from_numpy(inputs).float().to(device)
    with torch.no_grad():
        logits = model(inputs_tensor).logits  # [batch, seq_len, vocab_size]
        # Aggregate over vocab to get a single score per time step
        scores = logits.mean(dim=-1).cpu().numpy()  # [batch, seq_len]
        # For LIME, we need 2D output, so return mean over time dimension as well
        return scores.mean(axis=1, keepdims=True)  # [batch, 1]

# Create a LIME explainer
logger.info("Initializing LIME explainer...")
explainer = LimeTabularExplainer(
    training_data=np.zeros((10, target_length)),  # fake bg for kernel width
    mode="regression",
    feature_names=[f"s{i}" for i in range(target_length)],
    discretize_continuous=False,
    verbose=True
)

# Run LIME explanation
logger.info("Running LIME explanation (this may take a while)...")
explanation = explainer.explain_instance(
    data_row=raw_audio,
    predict_fn=lime_predict_fn,
    num_features=target_length,  # get all features' importances
    num_samples=500  # tradeoff: more samples = slower, more stable
)

# Get LIME weights in original order
lime_weights = np.zeros(target_length)
for idx, weight in explanation.as_map()[1]:
    lime_weights[idx] = weight

# Normalize LIME weights for fair comparison/visualization
lime_weights_norm = (lime_weights - np.min(lime_weights)) / (np.max(lime_weights) - np.min(lime_weights) + 1e-8)

# Amplify audio using LIME weights
amplified_audio_lime = raw_audio * (1 + lime_weights_norm) * amplification_factor

# Group LIME weights for visualization
window_size = 1000
grouped_lime = np.array([np.mean(lime_weights[i:i+window_size]) for i in range(0, target_length, window_size)])

# --- SHAP Setup for Comparison ---

import shap

# Create a background dataset for SHAP
logger.info("Creating background dataset for SHAP...")
num_background = 5
background = torch.zeros((num_background, input_values.shape[1]), device=device)
background += torch.randn_like(background) * 0.01

# Model wrapper for SHAP
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        attention_mask = torch.ones_like(x)
        logits = self.model(x, attention_mask=attention_mask).logits
        return logits.mean(dim=-1)

wrapped_model = ModelWrapper(model)

logger.info("Initializing SHAP explainer...")
explainer_shap = shap.GradientExplainer(wrapped_model, background)

logger.info("Computing SHAP values...")
with torch.no_grad():
    model_output = wrapped_model(input_values)

shap_values = explainer_shap.shap_values(input_values)
if isinstance(shap_values, list) and isinstance(shap_values[0], np.ndarray):
    shap_values = np.array(shap_values)
if shap_values.ndim == 3 and shap_values.shape[0] == 1:
    shap_values = shap_values[0]  # (16000, 49)
shap_values_agg = shap_values.mean(axis=1)  # (16000,)
audio_shap = shap_values_agg

# Normalize SHAP values
shap_normalized = (audio_shap - np.min(audio_shap)) / (np.max(audio_shap) - np.min(audio_shap) + 1e-8)
amplified_audio_shap = raw_audio * (1 + shap_normalized) * amplification_factor
grouped_shap = np.array([np.mean(audio_shap[i:i+window_size]) for i in range(0, target_length, window_size)])

# --- Visualization ---
logger.info("Creating LIME vs SHAP comparison visualization...")
plt.figure(figsize=(15, 15))

# Plot original audio
plt.subplot(4, 1, 1)
plt.plot(raw_audio)
plt.title("Original Raw Audio")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

# Plot grouped LIME values
plt.subplot(4, 1, 2)
plt.plot(grouped_lime)
plt.title(f"LIME Grouped Values (window size: {window_size})")
plt.xlabel("Window")
plt.ylabel("LIME Value")

# Plot grouped SHAP values
plt.subplot(4, 1, 3)
plt.plot(grouped_shap)
plt.title(f"SHAP Grouped Values (window size: {window_size})")
plt.xlabel("Window")
plt.ylabel("SHAP Value")

# Plot LIME- and SHAP-weighted audio
plt.subplot(4, 1, 4)
plt.plot(amplified_audio_lime, label="LIME-weighted audio", alpha=0.6)
plt.plot(amplified_audio_shap, label="SHAP-weighted audio", alpha=0.6)
plt.title("LIME vs SHAP Weighted Audio")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.savefig("lime_shap_audio_comparison.png")
plt.close()

# Save audio files
logger.info("Saving audio files...")
sf.write("original_audio.wav", raw_audio, sample_rate)
sf.write("lime_weighted_audio.wav", amplified_audio_lime, sample_rate)
sf.write("shap_weighted_audio.wav", amplified_audio_shap, sample_rate)

# Print statistics
logger.info("Computing statistics...")
print(f"Original audio mean: {np.mean(raw_audio):.3f}")
print(f"LIME grouped values mean: {np.mean(grouped_lime):.3f}")
print(f"SHAP grouped values mean: {np.mean(grouped_shap):.3f}")
print(f"LIME amplified audio mean: {np.mean(amplified_audio_lime):.3f}")
print(f"SHAP amplified audio mean: {np.mean(amplified_audio_shap):.3f}")

print("\nLIME and SHAP comparison complete. Inspect 'lime_shap_audio_comparison.png' for visual results.")
