import logging
import sys
from typing import Dict, List, Tuple

import numpy as np
import shap
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("evaluation.log")],
)
logger = logging.getLogger(__name__)

model_name = "facebook/wav2vec2-base-960h"


def _create_model_wrapper(model):
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

            # Add attention mask
            attention_mask = torch.ones_like(x)

            # Forward pass with attention mask
            logits = self.model(x, attention_mask=attention_mask).logits

            # Log the shape and statistics of logits
            logger.debug(f"Logits shape: {logits.shape}")
            logger.debug(f"Logits mean: {torch.mean(logits).item():.6f}")
            logger.debug(f"Logits std: {torch.std(logits).item():.6f}")

            # For SHAP, select greedy token choice per time step
            return torch.max(logits, dim=-1).values  # [batch, seq_len]

    return ModelWrapper(model)


def _add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white noise to audio at specified SNR"""
    signal_power = np.mean(audio**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise


def create_test_set(num_samples: int = 10) -> Dict:
    """Create a controlled test set with various conditions"""
    logger.debug(f"Creating test set with {num_samples} samples")
    ds = load_dataset(
        "patrickvonplaten/librispeech_asr_dummy", "clean", split="validation"
    )
    test_set = []
    dataset_index = 0

    for i in tqdm(range(min(num_samples, len(ds))), desc="Creating test samples"):
        sample = ds[i + dataset_index]
        audio = sample["audio"]["array"]
        while len(audio) < 100000:
            dataset_index += 1
            sample = ds[i + dataset_index]
            audio = sample["audio"]["array"]
        text = sample["text"]

        # Create clean sample
        test_set.append(
            {
                "type": "clean",
                "audio": audio,
                "text": text,
                "snr": float("inf"),
                "noise": np.zeros_like(audio),
            }
        )
        logger.debug(f"Added clean sample {i+1}")

        # Create noisy samples with different SNRs
        for snr in tqdm([5, 2, 1], desc=f"Adding noise to sample {i+1}", leave=False):
            noisy_audio = _add_noise(audio, snr)
            test_set.append(
                {
                    "type": "noisy",
                    "audio": noisy_audio,
                    "text": text,
                    "snr": snr,
                    "noise": noisy_audio - audio,
                }
            )
            logger.debug(f"Added noisy sample {i+1} with SNR {snr}dB")

    logger.debug(f"Test set created with {len(test_set)} total samples")
    return test_set


def compute_shap_values(
    processor, device, wrapped_model, model, vocab, audio: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for an audio sample using GradientExplainer"""
    logger.debug("Computing SHAP values")
    # Process audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    # Ensure input_values has the correct shape [batch_size, sequence_length]
    if len(input_values.shape) == 3:
        input_values = input_values.squeeze(1)

    # Create background samples with correct shape [batch_size, sequence_length]
    num_background = 5
    background = torch.zeros((num_background, input_values.shape[1]), device=device)
    background += torch.randn_like(background) * 0.01  # Add small random noise
    logger.debug(f"Created background samples with shape {background.shape}")
    logger.debug(f"Background mean: {torch.mean(background).item():.6f}")
    logger.debug(f"Background std: {torch.std(background).item():.6f}")

    # Initialize GradientExplainer with the model only
    explainer = shap.GradientExplainer(wrapped_model, background, batch_size=1)

    # Log model output before SHAP computation
    with torch.no_grad():
        model_output = wrapped_model(input_values)
        logger.debug(f"Model output shape: {model_output.shape}")
        logger.debug(f"Model output mean: {torch.mean(model_output).item():.6f}")
        logger.debug(f"Model output std: {torch.std(model_output).item():.6f}")
        logger.debug(f"Model output sum: {torch.sum(model_output).item():.6f}")

    # retrieve logits
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    logger.debug(f"Predicted IDs: {predicted_ids}")
    transcription = processor.batch_decode(predicted_ids)
    logger.debug(f"Transcription: {transcription}")

    output_string = "".join(
        [
            list(vocab.keys())[list(vocab.values()).index(id.item())]
            for id in predicted_ids[0]
        ]
    )
    logger.debug(f"Decoded output string: {output_string}")

    # Get SHAP values
    logger.debug("Computing SHAP values with GradientExplainer")
    shap_values = explainer.shap_values(input_values)
    logger.debug(f"Raw SHAP values type: {type(shap_values)}")

    logger.debug(f"SHAP values shape: {shap_values.shape}")

    return shap_values


def compute_shap_for_test_set(
    processor, device, wrapped_model, model, vocab, test_set: List[Dict]
) -> Dict:
    """Compute shap values for the test set"""
    logger.info("Computing metrics for test set")

    for i, sample in enumerate(tqdm(test_set, desc="Computing metrics")):
        logger.debug(f"Processing sample {i+1}/{len(test_set)}")
        audio = sample["audio"]
        text = sample["text"]
        # Get model prediction and confidence
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits
            probs = torch.softmax(logits, dim=-1)
            confidence = torch.mean(torch.max(probs, dim=-1)[0]).item()
        logger.debug(f"Model confidence: {confidence:.4f}")

        # Compute SHAP values
        shap_values = compute_shap_values(
            processor, device, wrapped_model, model, vocab, audio
        )
        logger.debug(f"SHAP values shape: {shap_values.shape}")
        logger.debug(
            f"SHAP values range: [{np.min(shap_values):.4f}, {np.max(shap_values):.4f}]"
        )

        # Save results
        np.save(
            f"data/shap_values_sample_{i+1}_{sample['type']}_{sample['snr']}",
            shap_values,
        )
        np.save(
            f"data/audio_sample_{i+1}_{sample['type']}_{sample['snr']}", sample["audio"]
        )
        np.save(
            f"data/noise_sample_{i+1}_{sample['type']}_{sample['snr']}", sample["noise"]
        )
        np.save(f"data/text_sample_{i+1}_{sample['type']}_{sample['snr']}.npy", text)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model and processor
logger.info(f"Loading model: {model_name}")
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model = model.to(device)
vocab = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    "|": 4,
    "E": 5,
    "T": 6,
    "A": 7,
    "O": 8,
    "N": 9,
    "I": 10,
    "H": 11,
    "S": 12,
    "R": 13,
    "D": 14,
    "L": 15,
    "U": 16,
    "M": 17,
    "W": 18,
    "C": 19,
    "F": 20,
    "G": 21,
    "Y": 22,
    "P": 23,
    "B": 24,
    "V": 25,
    "K": 26,
    "'": 27,
    "X": 28,
    "J": 29,
    "Q": 30,
    "Z": 31,
}
logger.info("Model loaded successfully")

# Create model wrapper for SHAP
wrapped_model = _create_model_wrapper(model)
logger.info("Model wrapper created")

# Create test set
logger.info("Creating test set...")
test_set = create_test_set(num_samples=20)
logger.info(test_set)

# Compute metrics and get visualization data
logger.info("Computing metrics...")
compute_shap_for_test_set(processor, device, wrapped_model, model, vocab, test_set)
