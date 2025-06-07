import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import logging
import librosa
import librosa.display
from matplotlib.patches import Patch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load model and processor
logger.info("Loading model and processor...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model = model.to(device)  # Move model to GPU

# Create a wrapper class for the model to return properly shaped output
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Get logits and return mean across vocabulary dimension
        logits = self.model(x).logits
        # Take mean across vocabulary dimension to get a 2D tensor [batch_size, sequence_length]
        return torch.mean(logits, dim=-1)

# Wrap the model
wrapped_model = ModelWrapper(model)

# Load dataset
logger.info("Loading dataset...")
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# Process each sample
for idx, sample in enumerate(ds):
    logger.info(f"\nProcessing sample {idx + 1}")
    logger.info(f"Reference text: {sample['text']}")
    
    # Get audio array and process it
    logger.info("Processing audio...")
    audio_array = sample['audio']['array']
    input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values
    input_values = input_values.to(device)  # Move input to GPU
    
    # Get model prediction
    logger.info("Getting model prediction...")
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
    logger.info(f"Model transcription: {transcription[0]}")
    
    # Create SHAP explainer with tensor input
    logger.info("Creating SHAP explainer...")
    background = input_values.clone()
    explainer = shap.DeepExplainer(wrapped_model, background)
    
    # Get SHAP values
    logger.info("Computing SHAP values...")
    shap_values = explainer.shap_values(input_values)
    
    # Move SHAP values to CPU for visualization
    if isinstance(shap_values[0], torch.Tensor):
        shap_values = [v.cpu().numpy() for v in shap_values]
    input_values = input_values.cpu()
    
    # Get the shapes for debugging
    logger.info(f"Input values shape: {input_values.shape}")
    logger.info(f"SHAP values shape: {shap_values[0].shape}")
    
    # Convert input values to numpy and ensure shapes match
    input_np = input_values.numpy().squeeze()  # Remove batch dimension
    shap_np = shap_values[0]  # Already a numpy array
    
    # Take mean across the vocabulary dimension (292) to get importance per time step
    shap_np = np.mean(shap_np, axis=1)
    
    logger.info(f"Final input shape: {input_np.shape}")
    logger.info(f"Final SHAP shape: {shap_np.shape}")
    
    # Create mel spectrogram
    logger.info("Creating mel spectrogram...")
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array,
        sr=16000,
        n_mels=256,  # Increased from 128 to 256 for higher frequency resolution
        fmax=8000,
        n_fft=2048,  # Increased FFT size for better time resolution
        hop_length=512  # Adjusted hop length for better time resolution
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Reshape SHAP values to match mel spectrogram time dimension
    shap_resized = np.interp(
        np.linspace(0, len(shap_np), mel_spec.shape[1]),
        np.arange(len(shap_np)),
        shap_np
    )
    
    # Create visualization
    logger.info("Creating visualization...")
    fig, ax = plt.subplots(figsize=(15, 10))  # Increased figure height
    
    # Plot mel spectrogram with improved colormap
    img = librosa.display.specshow(
        mel_spec_db,
        sr=16000,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap='magma'  # Changed from 'viridis' to 'magma' for better contrast
    )
    
    # Create SHAP heatmap overlay with more visible colors
    shap_heatmap = np.tile(shap_resized, (mel_spec.shape[0], 1))
    shap_img = ax.imshow(
        shap_heatmap,
        aspect='auto',
        origin='lower',
        alpha=0.5,  # Increased from 0.3 to 0.5 for better visibility
        cmap='hot',  # Changed from 'Reds' to 'hot' for better contrast
        extent=[0, len(audio_array)/16000, 0, mel_spec.shape[0]]
    )
    
    # Add colorbars and legend with improved formatting
    cbar1 = fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Power (dB)')
    cbar2 = fig.colorbar(shap_img, ax=ax, label='SHAP Value (Feature Importance)')
    
    # Add legend with description
    legend_elements = [
        Patch(facecolor='yellow', alpha=0.5, label='SHAP Values Overlay'),
        Patch(facecolor='purple', alpha=0.5, label='Mel Spectrogram')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add title and description
    ax.set_title('Mel Spectrogram with SHAP Values Heatmap Overlay\n' +
                 'Yellow/Red overlay indicates regions of high importance for the model prediction',
                 fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"shap_explanation_sample_{idx + 1}.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP explanation saved as 'shap_explanation_sample_{idx + 1}.png'")
    quit()