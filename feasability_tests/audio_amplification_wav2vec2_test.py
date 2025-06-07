import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from datasets import load_dataset
import shap
import logging
import sys
import os
import soundfile as sf

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set memory management settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add debug logging for SHAP
shap_logger = logging.getLogger('shap')
shap_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
shap_logger.addHandler(handler)

# Sound Amplification Settings
amplification_factor = 2.0  # Factor to amplify the audio by

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
raw_audio = ds[0]["audio"]["array"]
sample_rate = ds[0]["audio"]["sampling_rate"]

# Resample to 16kHz if needed (wav2vec2 expects 16kHz)
if sample_rate != 16000:
    logger.info(f"Resampling audio from {sample_rate} to 16000 Hz.")
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    raw_audio = resampler(torch.from_numpy(raw_audio).float()).numpy()
    sample_rate = 16000

# Truncate or pad audio to 1 second (16000 samples)
target_length = 16000
if len(raw_audio) > target_length:
    start = (len(raw_audio) - target_length) // 2
    raw_audio = raw_audio[start:start + target_length]
elif len(raw_audio) < target_length:
    raw_audio = np.pad(raw_audio, (0, target_length - len(raw_audio)), mode='constant')

logger.info(f"Audio shape after processing: {raw_audio.shape}")

# Prepare input for wav2vec2
input_values = processor(raw_audio, sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)

# Create a wrapper class for the model to get logits
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: [batch, time]
        attention_mask = torch.ones_like(x)
        logits = self.model(x, attention_mask=attention_mask).logits  # [batch, seq_len, vocab_size]
        # For SHAP, aggregate over vocab to get a scalar per time step
        return logits.mean(dim=-1)  # [batch, seq_len]

# Wrap the model
wrapped_model = ModelWrapper(model)

# Create a background dataset for SHAP
logger.info("Creating background dataset...")
num_background = 5
background = torch.zeros((num_background, input_values.shape[1]), device=device)
background += torch.randn_like(background) * 0.01  # Add small random noise

logger.info(f"Background shape: {background.shape}")

# Initialize SHAP explainer
logger.info("Initializing SHAP explainer...")
explainer = shap.GradientExplainer(wrapped_model, background)

# Compute SHAP values
logger.info("Computing SHAP values...")
#if len(input_values.shape) == 2:
#    input_values = input_values.unsqueeze(0) if input_values.shape[0] == 1 else input_values
logger.info(f"Input values shape: {input_values.shape}")

with torch.no_grad():
    model_output = wrapped_model(input_values)
    logger.info(f"Model output shape: {model_output.shape}")

shap_values = explainer.shap_values(input_values)
logger.info(f"Raw SHAP values type: {type(shap_values)}")

if isinstance(shap_values[0], torch.Tensor):
    shap_values = [v.cpu().numpy() for v in shap_values]

shap_values = np.array(shap_values)  # Shape: (1, batch, seq_len)
logger.info(f"SHAP values shape after conversion: {shap_values.shape}")

# Take mean over batch if needed
if shap_values.ndim == 3 and shap_values.shape[0] == 1:
    shap_values = shap_values[0]  # Now (16000, 49)

# shap_values: shape (1, 16000, 49)
shap_values_agg = shap_values.mean(axis=1)  # Now (16000,)

# Now audio_shap is just shap_values_agg
audio_shap = shap_values_agg

# Group SHAP values into windows for visualization
window_size = 1000
grouped_shap = np.array([np.mean(audio_shap[i:i+window_size]) for i in range(0, target_length, window_size)])

# Group SHAP values into windows for visualization
window_size = 1000
grouped_shap = np.array([np.mean(audio_shap[i:i+window_size]) for i in range(0, target_length, window_size)])

# Visualization
logger.info("Creating visualization...")
plt.figure(figsize=(15, 12))

# Plot original audio
plt.subplot(3, 1, 1)
plt.plot(raw_audio)
plt.title("Original Raw Audio")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

# Plot grouped SHAP values
plt.subplot(3, 1, 2)
plt.plot(grouped_shap)
plt.title(f"Grouped SHAP Values (window size: {window_size})")
plt.xlabel("Window")
plt.ylabel("SHAP Value")

# Plot SHAP-weighted audio using full interpolated values
plt.subplot(3, 1, 3)
if np.max(audio_shap) != np.min(audio_shap):
    shap_normalized = (audio_shap - np.min(audio_shap)) / (np.max(audio_shap) - np.min(audio_shap))
else:
    shap_normalized = np.ones_like(audio_shap)
amplified_audio = raw_audio * (1 + shap_normalized * amplification_factor)
plt.plot(amplified_audio)
plt.title("SHAP-Weighted Audio")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.savefig("shap_audio_comparison_wav2vec2.png")
plt.close()

# Save audio files
logger.info("Saving audio files...")
sf.write("original_audio.wav", raw_audio, sample_rate)
sf.write("shap_weighted_audio.wav", amplified_audio, sample_rate)

# Print some statistics
logger.info("Computing statistics...")
print(f"Original audio shape: {raw_audio.shape}")
print(f"Grouped SHAP values shape: {grouped_shap.shape}")
print(f"Original audio mean: {np.mean(raw_audio):.3f}")
print(f"Grouped SHAP values mean: {np.mean(grouped_shap):.3f}")
print(f"Amplified audio mean: {np.mean(amplified_audio):.3f}")