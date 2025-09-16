import time
import logging
import sys
from typing import Dict, List, Tuple

import numpy as np
import shap
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import matplotlib.pyplot as plt



logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)



MODEL_NAME = "facebook/wav2vec2-base-960h"


def _create_model_wrapper(model):
    """Create a wrapper class for the model to get properly shaped output."""

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            if len(x.shape) == 4:
                x = x.squeeze(1).squeeze(1)
            elif len(x.shape) == 3:
                x = x.squeeze(1)

            attention_mask = torch.ones_like(x)
            logits = self.model(x, attention_mask=attention_mask).logits
            return torch.max(logits, dim=-1).values

    return ModelWrapper(model)


def compute_shap_values(
    processor, device, wrapped_model, audio: np.ndarray
) -> np.ndarray:
    """
    Compute SHAP values for an audio sample using GradientExplainer.
    Note: Redundant parts removed for a cleaner benchmark function.
    """
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    if len(input_values.shape) == 3:
        input_values = input_values.squeeze(1)

    
    num_background = 5
    background = torch.zeros((num_background, input_values.shape[1]), device=device)
    background += torch.randn_like(background) * 0.01

    
    explainer = shap.GradientExplainer(wrapped_model, background)

    
    shap_values = explainer.shap_values(input_values)

    return np.array(shap_values)


def plot_results(results: List[Dict]):
    """
    Plots the benchmark results using matplotlib.
    """
    print("\nGenerating plot...")
    lengths = [r['length'] for r in results]
    durations = [r['duration'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lengths, durations, marker='o', linestyle='-', color='b')

    ax.set_title('SHAP GradientExplainer Runtime vs. Input Size', fontsize=16)
    ax.set_xlabel('Audio Sample Length', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    
    plt.xticks(rotation=45)
    plt.tight_layout()

    print("Displaying plot. Close the plot window to exit the script.")
    plt.show()


def run_benchmark():
    """
    Main function to set up the model and run the benchmark.
    """
    
    print("Setting up model and processor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
    wrapped_model = _create_model_wrapper(model)
    print("Setup complete.\n")

    
    print("Starting runtime benchmark for GradientExplainer...")
    
    sample_lengths = [2500, 5000, 10000, 20000, 40000, 80000]
    results = []

    for length in sample_lengths:
        print(f"Testing sample with length: {length}...")

        
        dummy_audio = np.random.randn(length).astype(np.float32)

        
        start_time = time.time()
        try:
            _ = compute_shap_values(processor, device, wrapped_model, dummy_audio)
            end_time = time.time()
            duration = end_time - start_time
            results.append({"length": length, "duration": duration})
            print(f"  -> Done in {duration:.2f} seconds.")
        except torch.cuda.OutOfMemoryError:
            print(f"  -> ERROR: Ran out of GPU memory for length {length}. Stopping benchmark.")
            break 

    
    if not results:
        print("No results to display. The benchmark may have failed on the first sample.")
        return

    print("\n" + "=" * 45)
    print("      Benchmark Results Summary")
    print("=" * 45)
    print(f"{'Sample Length':<20} | {'Runtime (seconds)':<20}")
    print("-" * 45)

    for res in results:
        print(f"{res['length']:<20} | {res['duration']:<20.4f}")

    print("=" * 45)
    
    
    plot_results(results)


if __name__ == "__main__":
    run_benchmark()