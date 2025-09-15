import glob
import logging
import os
import sys

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def calculate_eta_raw(
    clean_audio: np.ndarray,
    noise_audio: np.ndarray,
    shap_matrix: np.ndarray,
    sr: int,
    segment_ms: int = 0.0625,
    percentile: float = 99.0,
) -> float:
    """Calculates the Raw Audio Speech Relevance Score (η_raw)."""
    min_len = min(len(clean_audio), len(noise_audio), shap_matrix.shape[0])
    segment_length_samples = int(sr * (segment_ms / 1000.0))
    if segment_length_samples == 0:
        return 0.0
    num_segments = min_len // segment_length_samples
    if num_segments == 0:
        return 0.0
    trunc_len = num_segments * segment_length_samples

    clean_audio_trunc = clean_audio[:trunc_len]
    noise_audio_trunc = noise_audio[:trunc_len]
    shap_matrix_trunc = shap_matrix[:trunc_len, :]

    clean_segments = clean_audio_trunc.reshape(num_segments, segment_length_samples)
    noise_segments = noise_audio_trunc.reshape(num_segments, segment_length_samples)
    E_c = np.sum(np.square(clean_segments), axis=1)
    E_u = np.sum(np.square(noise_segments), axis=1)
    itm = (E_c > E_u).astype(int)

    phi_total = np.sum(np.abs(shap_matrix_trunc), axis=1)
    phi_total_segments = phi_total.reshape(num_segments, segment_length_samples)
    bar_phi = np.mean(phi_total_segments, axis=1)

    tau = np.percentile(bar_phi, percentile)
    is_highly_relevant = (bar_phi > tau).astype(int)
    denominator = np.sum(is_highly_relevant)

    if denominator == 0:
        return 0.0

    intersection = is_highly_relevant * itm
    numerator = np.sum(intersection)

    return numerator / denominator


def calculate_wer_for_sample(
    mixed_audio: np.ndarray,
    reference_text: str,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    sr: int,
    device: torch.device,
) -> float:
    """Generates a transcription and calculates the WER against a reference."""
    # Process audio and get model's prediction (hypothesis)
    inputs = processor(mixed_audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    hypothesis_text = processor.batch_decode(predicted_ids)[0]

    # Calculate WER
    wer = jiwer.wer(reference_text, hypothesis_text)

    logger.info(f"Reference:  '{reference_text}'")
    logger.info(f"Hypothesis: '{hypothesis_text}'")
    logger.info(f"WER: {wer:.4f}")

    return wer


if __name__ == "__main__":
    data_dir = "data/"
    SR = 16000

    logger.info("Loading Wav2Vec2 model and processor...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "facebook/wav2vec2-base-960h"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    except Exception as e:
        logger.error(
            f"Failed to load ASR model: {e}. Please ensure you have an internet connection."
        )
        sys.exit(1)

    audio_files = sorted(glob.glob(os.path.join(data_dir, "audio_*.npy")))
    if not audio_files:
        logger.error(f"No audio files found in '{data_dir}'. Cannot proceed.")
        sys.exit(1)

    logger.info(f"Found {len(audio_files)} samples to process.")

    wer_scores = []
    eta_raw_scores = []
    sample_names = []

    for audio_path in audio_files:
        base_name = os.path.basename(audio_path)
        identifier = base_name.replace("audio_sample_", "").replace(".npy", "")
        logger.info(f"\n--- Processing sample: {identifier} ---")

        # Construct paths for other files
        noise_path = os.path.join(data_dir, f"noise_sample_{identifier}.npy")
        shap_path = os.path.join(data_dir, f"shap_values_sample_{identifier}.npy")
        text_path = os.path.join(data_dir, f"text_sample_{identifier}.npy")

        # Check if all required files exist
        if not all(os.path.exists(p) for p in [noise_path, shap_path, text_path]):
            logger.warning(f"Skipping sample {identifier} due to missing data files.")
            continue

        # Load data
        mixed_audio = np.load(audio_path)
        noise_audio = np.load(noise_path)
        shap_values = np.squeeze(np.load(shap_path))
        # Text is often saved as a 0-dim array; .item() extracts the string
        reference_text = str(np.load(text_path, allow_pickle=True).item())

        # Derive clean audio for eta_raw calculation
        min_len = min(len(mixed_audio), len(noise_audio))
        clean_audio = mixed_audio[:min_len] - noise_audio[:min_len]

        # Calculate metrics
        wer = calculate_wer_for_sample(
            mixed_audio, reference_text, processor, model, SR, device
        )
        eta_raw = calculate_eta_raw(clean_audio, noise_audio[:min_len], shap_values, SR)
        logger.info(f"Calculated η_raw: {eta_raw:.4f}")

        # Store results
        wer_scores.append(wer)
        eta_raw_scores.append(eta_raw)
        sample_names.append(identifier.split("_")[0])  # Use a shorter name for plotting

    if not wer_scores:
        logger.error("No data was successfully processed. Cannot generate plot.")
        sys.exit(1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(wer_scores, eta_raw_scores, s=100, alpha=0.7, edgecolors="k")

    # Add labels to points
    for i, name in enumerate(sample_names):
        ax.text(wer_scores[i] + 0.01, eta_raw_scores[i], name, fontsize=9)

    ax.set_title("Model Performance vs. Speech Relevance Score", fontsize=16, pad=20)
    ax.set_xlabel("Word Error Rate (WER) - Lower is better", fontsize=12)
    ax.set_ylabel("Speech Relevance Score (η_raw) - Higher is better", fontsize=12)
    ax.set_xlim(left=max(0, min(wer_scores) - 0.05), right=max(wer_scores) + 0.05)
    ax.set_ylim(
        bottom=max(0, min(eta_raw_scores) - 0.05),
        top=min(1.05, max(eta_raw_scores) + 0.05),
    )

    fig.tight_layout()
    plot_filename = "wer_vs_eta_raw_plot.png"
    plt.savefig(plot_filename)

    logger.info(f"\nPlot saved successfully as '{plot_filename}'")
