import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from torchaudio.models import Conformer
from datasets import load_dataset
import shap
import logging
import sys
import os
import soundfile as sf
# from custom_shap_handlers import op_handler  # Import our custom handlers

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

# Create a custom Conformer model with non-inplace ReLU
class CustomConformer(torch.nn.Module):
    def __init__(self, input_dim=80, num_heads=4, ffn_dim=128, num_layers=4):
        super().__init__()
        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
            use_group_norm=True,
            convolution_first=True
        )
        # Add a simple projection layer to get logits
        # The Conformer output dimension is input_dim (80)
        self.projection = torch.nn.Linear(input_dim, 32)  # 32 classes for simplicity

    def forward(self, x):
        # Add batch dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # Transpose to [batch, time, features]
        x = x.transpose(1, 2)
        # Create sequence lengths tensor for each sample in the batch
        batch_size = x.shape[0]
        lengths = torch.full((batch_size,), x.shape[1], device=x.device)
        # Get conformer output
        x, _ = self.conformer(x, lengths)
        # Project to logits
        return self.projection(x)

# Load model
logger.info("Loading model...")
model = CustomConformer()
model = model.to(device)

# Load sample audio
logger.info("Loading dataset...")
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
raw_audio = ds[0]["audio"]["array"]
sample_rate = ds[0]["audio"]["sampling_rate"]

# Process audio - downsample to reduce memory usage
logger.info("Processing audio...")
target_length = 16000  # 1 second at 16kHz
if len(raw_audio) > target_length:
    # Take a segment from the middle of the audio
    start = (len(raw_audio) - target_length) // 2
    raw_audio = raw_audio[start:start + target_length]

# Convert to mel spectrogram
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=80
)
mel_spec = mel_transform(torch.from_numpy(raw_audio).float())
mel_spec = mel_spec.to(device)
logger.info(f"Mel spectrogram shape: {mel_spec.shape}")

# Create a wrapper class for the model to get logits
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Add batch dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # Get logits and return mean across vocabulary dimension
        logits = self.model(x)
        # Log the shape and statistics of logits
        logger.debug(f"Logits shape: {logits.shape}")
        logger.debug(f"Logits mean: {torch.mean(logits).item():.6f}")
        logger.debug(f"Logits std: {torch.std(logits).item():.6f}")
        # Take mean across vocabulary dimension to get a 2D tensor [batch_size, sequence_length]
        return torch.mean(logits, dim=-1)

# Wrap the model
wrapped_model = ModelWrapper(model)

# Create a background dataset for SHAP
logger.info("Creating background dataset...")
# Create fewer background samples to reduce memory usage
num_background = 5  # Reduced from 10
# Create background with same shape as mel_spec [batch, features, time]
background = torch.zeros((num_background, mel_spec.shape[0], mel_spec.shape[1]), device=device)
background += torch.randn_like(background) * 0.01  # Add small random noise
logger.info(f"Background shape: {background.shape}")
logger.info(f"Background mean: {torch.mean(background).item():.6f}")
logger.info(f"Background std: {torch.std(background).item():.6f}")

# Initialize SHAP explainer with more detailed logging
logger.info("Initializing SHAP explainer...")
explainer = shap.GradientExplainer(wrapped_model, background)

# Get SHAP values with more detailed logging
logger.info("Computing SHAP values...")
# Add batch dimension to mel_spec if not present
if len(mel_spec.shape) == 2:
    mel_spec = mel_spec.unsqueeze(0)
logger.info(f"Input values shape: {mel_spec.shape}")
logger.info(f"Input values mean: {torch.mean(mel_spec).item():.6f}")
logger.info(f"Input values std: {torch.std(mel_spec).item():.6f}")

# Log model output before SHAP computation
with torch.no_grad():
    model_output = wrapped_model(mel_spec)
    logger.info(f"Model output shape: {model_output.shape}")
    logger.info(f"Model output mean: {torch.mean(model_output).item():.6f}")
    logger.info(f"Model output std: {torch.std(model_output).item():.6f}")
    logger.info(f"Model output sum: {torch.sum(model_output).item():.6f}")

# Compute SHAP values
shap_values = explainer.shap_values(mel_spec)
logger.info(f"Raw SHAP values shape: {np.array(shap_values).shape}")
logger.info(f"Raw SHAP values sum: {np.sum(shap_values)}")
logger.info(f"Raw SHAP values mean: {np.mean(shap_values):.6f}")
logger.info(f"Raw SHAP values std: {np.std(shap_values):.6f}")

# Log intermediate values during processing
if isinstance(shap_values[0], torch.Tensor):
    logger.info("Converting SHAP values from tensor to numpy...")
    shap_values = [v.cpu().numpy() for v in shap_values]
    logger.info(f"After tensor conversion - SHAP values shape: {np.array(shap_values).shape}")
    logger.info(f"After tensor conversion - SHAP values sum: {np.sum(shap_values)}")

mel_spec = mel_spec.cpu()

# Convert to numpy and process SHAP values
shap_values = np.array(shap_values)  # Shape: (1, 80, 32)
logger.info(f"After numpy conversion - SHAP values shape: {shap_values.shape}")
logger.info(f"After numpy conversion - SHAP values sum: {np.sum(shap_values)}")
logger.info(f"After numpy conversion - SHAP values mean: {np.mean(shap_values):.6f}")
logger.info(f"After numpy conversion - SHAP values std: {np.std(shap_values):.6f}")

# FIX: Average over frequency dimension (axis=1) to get importance per time step
shap_values = np.mean(shap_values, axis=1)  # Shape: (1, 32)
logger.info(f"After frequency mean - SHAP values shape: {shap_values.shape}")
logger.info(f"After frequency mean - SHAP values sum: {np.sum(shap_values)}")
logger.info(f"After frequency mean - SHAP values mean: {np.mean(shap_values):.6f}")
logger.info(f"After frequency mean - SHAP values std: {np.std(shap_values):.6f}")

# Log the difference between model output and SHAP values sum
model_output_sum = torch.sum(model_output).item()
shap_values_sum = np.sum(shap_values)
logger.info(f"Model output sum: {model_output_sum:.6f}")
logger.info(f"SHAP values sum: {shap_values_sum:.6f}")
logger.info(f"Absolute difference: {abs(model_output_sum - shap_values_sum):.6f}")
logger.info(f"Relative difference: {abs(model_output_sum - shap_values_sum) / abs(model_output_sum):.6f}")

# FIX: Average over output time dimension to get per-input-time importance
shap_values = np.mean(shap_values, axis=2)  # Shape: (1, 32)
shap_values = shap_values.squeeze()  # Remove singleton dimension
logger.info(f"After output time mean - SHAP values shape: {shap_values.shape}")
logger.info(f"After output time mean - SHAP values sum: {np.sum(shap_values)}")

# FIX: Interpolate SHAP values from 32 time steps to 16000 audio samples
n_fft = 1024
hop_length = 512
centers = (n_fft // 2) + hop_length * np.arange(32)  # Frame centers in audio samples
audio_shap = np.interp(
    np.arange(16000), 
    centers, 
    shap_values, 
    left=shap_values[0], 
    right=shap_values[-1]
)
logger.info(f"Interpolated SHAP values shape: {audio_shap.shape}")
logger.info(f"Interpolated SHAP values sum: {np.sum(audio_shap)}")

# Group SHAP values into windows for visualization
window_size = 1000
num_windows = 16000 // window_size
grouped_shap = np.array([np.mean(audio_shap[i:i+window_size]) for i in range(0, 16000, window_size)])
logger.info(f"After grouping - SHAP values shape: {grouped_shap.shape}")
logger.info(f"After grouping - SHAP values sum: {np.sum(grouped_shap)}")
logger.info(f"After grouping - SHAP values mean: {np.mean(grouped_shap):.6f}")
logger.info(f"After grouping - SHAP values std: {np.std(grouped_shap):.6f}")

# Log SHAP value statistics
logger.info(f"Interpolated SHAP values statistics:")
logger.info(f"  Min: {np.min(audio_shap)}")
logger.info(f"  Max: {np.max(audio_shap)}")
logger.info(f"  Mean: {np.mean(audio_shap):.6f}")
logger.info(f"  Std: {np.std(audio_shap):.6f}")

logger.info(f"Grouped SHAP values statistics (window size: {window_size}):")
logger.info(f"  Min: {np.min(grouped_shap)}")
logger.info(f"  Max: {np.max(grouped_shap)}")
logger.info(f"  Mean: {np.mean(grouped_shap):.6f}")
logger.info(f"  Std: {np.std(grouped_shap):.6f}")

# Create visualization
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
# Normalize audio_shap to [0,1] for amplification
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
plt.savefig("shap_audio_comparison.png")
plt.close()

# Save audio files
logger.info("Saving audio files...")
# Save original audio
sf.write("original_audio.wav", raw_audio, sample_rate)
# Save SHAP-weighted audio
sf.write("shap_weighted_audio.wav", amplified_audio, sample_rate)

# Print some statistics
logger.info("Computing statistics...")
print(f"Original audio shape: {raw_audio.shape}")
print(f"Grouped SHAP values shape: {grouped_shap.shape}")
print(f"Original audio mean: {np.mean(raw_audio):.3f}")
print(f"Grouped SHAP values mean: {np.mean(grouped_shap):.3f}")
print(f"Amplified audio mean: {np.mean(amplified_audio):.3f}")