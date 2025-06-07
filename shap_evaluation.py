import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import logging
import sys
import os
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import librosa
import librosa.display
from scipy.stats import pearsonr
from typing import List, Dict, Tuple
import json
from pathlib import Path
from tqdm import tqdm
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

class SHAPEvaluator:
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        """Initialize the SHAP evaluator with model and processor"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        logger.info(f"Loading model: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        logger.info("Model loaded successfully")
        
        # Create model wrapper for SHAP
        self.wrapped_model = self._create_model_wrapper()
        logger.info("Model wrapper created")
        
    def _create_model_wrapper(self):
        """Create a wrapper class for the model to get properly shaped output"""
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                # Ensure input has correct shape for the model
                if len(x.shape) == 4:
                    x = x.squeeze(1).squeeze(1)  # Remove extra dimensions
                elif len(x.shape) == 3:
                    x = x.squeeze(1)  # Remove extra dimension
                
                # Forward pass
                logits = self.model(x).logits
                
                # Log the shape and statistics of logits
                logger.debug(f"Logits shape: {logits.shape}")
                logger.debug(f"Logits mean: {torch.mean(logits).item():.6f}")
                logger.debug(f"Logits std: {torch.std(logits).item():.6f}")
                
                # Use max pooling instead of mean to preserve more information
                return torch.max(logits, dim=-1)[0]
        
        return ModelWrapper(self.model)
    
    def create_test_set(self, num_samples: int = 10) -> Dict:
        """Create a controlled test set with various conditions"""
        logger.info(f"Creating test set with {num_samples} samples")
        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        test_set = []
        
        for i in tqdm(range(min(num_samples, len(ds))), desc="Creating test samples"):
            sample = ds[i]
            audio = sample["audio"]["array"]
            text = sample["text"]
            
            # Create clean sample
            test_set.append({
                "type": "clean",
                "audio": audio,
                "text": text,
                "snr": float('inf')
            })
            logger.info(f"Added clean sample {i+1}")
            
            # Create noisy samples with different SNRs [20, 10, 0, -5]
            for snr in tqdm([20,0], desc=f"Adding noise to sample {i+1}", leave=False):
                noisy_audio = self._add_noise(audio, snr)
                test_set.append({
                    "type": "noisy",
                    "audio": noisy_audio,
                    "text": text,
                    "snr": snr
                })
                logger.info(f"Added noisy sample {i+1} with SNR {snr}dB")
        
        logger.info(f"Test set created with {len(test_set)} total samples")
        return test_set
    
    def _add_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """Add white noise to audio at specified SNR"""
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
    
    def compute_shap_values(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SHAP values for an audio sample using GradientExplainer"""
        logger.info("Computing SHAP values")
        # Process audio
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        
        # Ensure input_values has the correct shape [batch_size, sequence_length]
        if len(input_values.shape) == 3:
            input_values = input_values.squeeze(1)
        
        # Create background samples with correct shape [batch_size, sequence_length]
        num_background = 5
        background = torch.zeros((num_background, input_values.shape[1]), device=self.device)
        background += torch.randn_like(background) * 0.01  # Add small random noise
        logger.info(f"Created background samples with shape {background.shape}")
        logger.info(f"Background mean: {torch.mean(background).item():.6f}")
        logger.info(f"Background std: {torch.std(background).item():.6f}")
        
        # Initialize GradientExplainer with the model only
        explainer = shap.GradientExplainer(
            self.wrapped_model,
            background,
            batch_size=1
        )
        
        # Log model output before SHAP computation
        with torch.no_grad():
            model_output = self.wrapped_model(input_values)
            logger.info(f"Model output shape: {model_output.shape}")
            logger.info(f"Model output mean: {torch.mean(model_output).item():.6f}")
            logger.info(f"Model output std: {torch.std(model_output).item():.6f}")
            logger.info(f"Model output sum: {torch.sum(model_output).item():.6f}")
        
        # Get SHAP values - ensure input has shape [batch_size, sequence_length]
        logger.info("Computing SHAP values with GradientExplainer")
        shap_values = explainer.shap_values(
            input_values.unsqueeze(0),  # Add batch dimension
            ranked_outputs=None,
            nsamples=20
        )
        
        # Convert to numpy and process
        if isinstance(shap_values[0], torch.Tensor):
            shap_values = [v.cpu().numpy() for v in shap_values]
        
        # Log raw SHAP values
        logger.info(f"Raw SHAP values shape: {np.array(shap_values).shape}")
        logger.info(f"Raw SHAP values sum: {np.sum(shap_values)}")
        logger.info(f"Raw SHAP values mean: {np.mean(shap_values):.6f}")
        logger.info(f"Raw SHAP values std: {np.std(shap_values):.6f}")
        
        # Handle the SHAP values shape properly
        shap_values = np.array(shap_values)  # Convert to numpy array
        if len(shap_values.shape) == 3:  # [batch, sequence, features]
            shap_values = np.mean(shap_values, axis=2)  # Average over features
        shap_values = shap_values.squeeze()  # Remove batch dimension
        
        logger.info(f"After processing - SHAP values shape: {shap_values.shape}")
        logger.info(f"After processing - SHAP values sum: {np.sum(shap_values)}")
        logger.info(f"After processing - SHAP values mean: {np.mean(shap_values):.6f}")
        logger.info(f"After processing - SHAP values std: {np.std(shap_values):.6f}")
        
        # Interpolate SHAP values to match input length
        n_fft = 1024
        hop_length = 512
        centers = (n_fft // 2) + hop_length * np.arange(len(shap_values))
        audio_shap = np.interp(
            np.arange(len(input_values.squeeze())),
            centers,
            shap_values,
            left=shap_values[0],
            right=shap_values[-1]
        )
        
        logger.info(f"Interpolated SHAP values shape: {audio_shap.shape}")
        logger.info(f"Interpolated SHAP values sum: {np.sum(audio_shap)}")
        logger.info(f"Interpolated SHAP values mean: {np.mean(audio_shap):.6f}")
        logger.info(f"Interpolated SHAP values std: {np.std(audio_shap):.6f}")
        
        return input_values.cpu().numpy().squeeze(), audio_shap
    
    def compute_metrics(self, test_set: List[Dict]) -> Dict:
        """Compute evaluation metrics for the test set"""
        logger.info("Computing metrics for test set")
        metrics = {
            "shap_noise_correlation": [],
            "shap_confidence_correlation": [],
            "wer_correlation": []
        }
        
        # Store SHAP values and other computed values for visualization
        visualization_data = []
        
        for i, sample in enumerate(tqdm(test_set, desc="Computing metrics")):
            logger.info(f"Processing sample {i+1}/{len(test_set)}")
            audio = sample["audio"]
            text = sample["text"]
            
            # Get model prediction and confidence
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_values).logits
                probs = torch.softmax(logits, dim=-1)
                confidence = torch.mean(torch.max(probs, dim=-1)[0]).item()
            logger.info(f"Model confidence: {confidence:.4f}")
            
            # Compute SHAP values
            _, shap_values = self.compute_shap_values(audio)
            logger.info(f"SHAP values shape: {shap_values.shape}")
            logger.info(f"SHAP values range: [{np.min(shap_values):.4f}, {np.max(shap_values):.4f}]")
            
            # Store data for visualization
            visualization_data.append({
                "audio": audio,
                "shap_values": shap_values,
                "type": sample["type"],
                "snr": sample["snr"]
            })
            
            # Compute WER
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            wer = self._compute_wer(text, transcription)
            logger.info(f"WER: {wer:.4f}")
            
            # Compute correlations
            if sample["type"] == "noisy":
                # For noisy samples, compute correlation with noise regions
                noise_mask = self._estimate_noise_regions(audio)
                logger.info(f"Noise mask shape: {noise_mask.shape}")
                logger.info(f"Noise mask range: [{np.min(noise_mask):.4f}, {np.max(noise_mask):.4f}]")
                
                # Ensure same length by resizing noise mask to match SHAP values
                noise_mask = np.interp(
                    np.linspace(0, len(noise_mask), len(shap_values)),
                    np.arange(len(noise_mask)),
                    noise_mask
                )
                logger.info(f"Resized noise mask shape: {noise_mask.shape}")
                
                # Check for constant values that would cause NaN correlation
                if np.all(noise_mask == noise_mask[0]) or np.all(shap_values == shap_values[0]):
                    logger.warning(f"Constant values detected in sample {i+1}")
                    logger.warning(f"All noise mask values: {noise_mask[0]:.4f}")
                    logger.warning(f"All SHAP values: {shap_values[0]:.4f}")
                else:
                    correlation = pearsonr(shap_values, noise_mask)[0]
                    metrics["shap_noise_correlation"].append(correlation)
                    logger.info(f"SHAP-noise correlation: {correlation:.4f}")
            
            # Compute correlation with confidence
            if not np.all(shap_values == shap_values[0]):
                conf_corr = pearsonr(shap_values, np.ones_like(shap_values) * confidence)[0]
                metrics["shap_confidence_correlation"].append(conf_corr)
                logger.info(f"SHAP-confidence correlation: {conf_corr:.4f}")
            
            # Compute correlation with WER
            if not np.all(shap_values == shap_values[0]):
                wer_corr = pearsonr(shap_values, np.ones_like(shap_values) * wer)[0]
                metrics["wer_correlation"].append(wer_corr)
                logger.info(f"SHAP-WER correlation: {wer_corr:.4f}")
        
        # Compute average metrics
        for key in metrics:
            if metrics[key]:  # Only compute mean if list is not empty
                metrics[key] = np.mean(metrics[key])
                logger.info(f"Average {key}: {metrics[key]:.4f}")
            else:
                metrics[key] = float('nan')
                logger.warning(f"No valid values for {key}, setting to NaN")
        
        return metrics, visualization_data
    
    def _estimate_noise_regions(self, audio: np.ndarray) -> np.ndarray:
        """Estimate noise regions in audio using energy and zero-crossing rate"""
        # Compute short-time energy
        frame_length = 1024
        hop_length = 512
        
        # Energy-based detection
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Zero-crossing rate for additional noise detection
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Combine features
        energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
        zcr = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr))
        
        # Weighted combination of features
        noise_mask = 0.7 * energy + 0.3 * zcr
        
        # Apply threshold to identify noise regions
        threshold = np.mean(noise_mask) + 0.5 * np.std(noise_mask)
        noise_mask = (noise_mask > threshold).astype(float)
        
        # Resize to match SHAP values length
        noise_mask = np.interp(
            np.linspace(0, len(noise_mask), len(audio)),
            np.arange(len(noise_mask)),
            noise_mask
        )
        
        return noise_mask
    
    def _compute_wer(self, reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate between reference and hypothesis"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Create distance matrix
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        d[:, 0] = np.arange(len(ref_words) + 1)
        d[0, :] = np.arange(len(hyp_words) + 1)
        
        # Compute Levenshtein distance
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i, j] = d[i-1, j-1]
                else:
                    d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
        
        return d[-1, -1] / len(ref_words)
    
    def visualize_results(self, visualization_data: List[Dict], metrics: Dict, save_dir: str = "results"):
        """Visualize evaluation results using pre-computed SHAP values"""
        logger.info(f"Saving results to {save_dir}")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Metrics saved to metrics.json")
        
        # Create visualizations for each sample
        for i, data in enumerate(tqdm(visualization_data, desc="Creating visualizations")):
            logger.info(f"Creating visualization for sample {i+1}")
            audio = data["audio"]
            shap_values = data["shap_values"]
            
            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=16000,
                n_mels=256,
                fmax=8000,
                n_fft=2048,
                hop_length=512
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Plot mel spectrogram in grayscale
            img = librosa.display.specshow(
                mel_spec_db,
                sr=16000,
                x_axis='time',
                y_axis='mel',
                ax=ax,
                cmap='gray'
            )
            
            # Create SHAP heatmap overlay with blue-red colormap
            shap_heatmap = np.tile(shap_values, (mel_spec.shape[0], 1))
            shap_img = ax.imshow(
                shap_heatmap,
                aspect='auto',
                origin='lower',
                alpha=0.5,
                cmap='RdBu_r',  # Blue for negative, red for positive
                extent=[0, len(audio)/16000, 0, mel_spec.shape[0]]
            )
            
            # Add colorbars and title
            cbar1 = fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Power (dB)')
            cbar2 = fig.colorbar(shap_img, ax=ax, label='SHAP Value')
            
            ax.set_title(f'Sample {i+1} - Type: {data["type"]}, SNR: {data["snr"]}dB')
            
            plt.tight_layout()
            plt.savefig(save_dir / f"sample_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Visualization saved for sample {i+1}")
        
        logger.info("All visualizations completed")

def main():
    # Initialize evaluator
    evaluator = SHAPEvaluator()
    
    # Create test set
    logger.info("Creating test set...")
    test_set = evaluator.create_test_set(num_samples=1)
    
    # Compute metrics and get visualization data
    logger.info("Computing metrics...")
    metrics, visualization_data = evaluator.compute_metrics(test_set)
    
    # Visualize results
    logger.info("Visualizing results...")
    evaluator.visualize_results(visualization_data, metrics)
    
    # Print metrics
    logger.info("\nEvaluation Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main() 