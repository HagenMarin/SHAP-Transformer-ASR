import torch
import shap
import librosa
import numpy as np
import matplotlib.pyplot as plt
import datasets
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor
import warnings

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore")

class modelWrapper(torch.nn.Module):
    """
    A simple wrapper around the Wav2Vec2ConformerForCTC model to ensure compatibility with SHAP.
    """
    def __init__(self, model, device, timestep_to_explain, token_id_to_explain):
        super(modelWrapper, self).__init__()
        self.model = model
        self.device = device
        self.timestep_to_explain = timestep_to_explain
        self.token_id_to_explain = token_id_to_explain

    def forward(self, input_values):
        """
        This function takes a numpy array of input_values and returns the
        model's logit for the specific token and timestep we want to explain.
        SHAP uses this function to understand the model's behavior.
        """
        # Convert numpy array from SHAP back to a torch tensor
        input_tensor = torch.from_numpy(input_values).float().to(self.device)
        
        # Ensure tensor has the correct dimensions (batch, sequence_length)
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            logits_output = self.model(input_tensor).logits
        
        # Return the specific logit value we are interested in
        # SHAP expects a numpy array as output
        specific_logit = logits_output[:, self.timestep_to_explain, self.token_id_to_explain]
        return specific_logit.cpu().numpy()

def main():
    """
    Main function to run the SHAP analysis on a Conformer ASR model.
    """
    print("--- ASR Conformer SHAP Explainer ---")

    # 1. Setup: Load Model, Processor, and Data
    # -------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        print("Loading model and processor...")
        model_id = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ConformerForCTC.from_pretrained(model_id).to(device)
        model.eval() # Set model to evaluation mode

        print("Loading audio sample from LibriSpeech dataset...")
        # Load a single sample from the validation set in streaming mode
        dataset = datasets.load_dataset(
            "librispeech_asr", "clean", split="validation", streaming=True
        )
        sample = next(iter(dataset))
        
        # The model expects audio at a 16kHz sample rate
        target_sr = processor.feature_extractor.sampling_rate
        audio_array = librosa.resample(
            sample["audio"]["array"], 
            orig_sr=sample["audio"]["sampling_rate"], 
            target_sr=target_sr
        )
        print(f"Original audio text: {sample['text']}")

    except Exception as e:
        print(f"Error during setup: {e}")
        print("Please ensure you have an internet connection and the required libraries are installed.")
        return

    # 2. Preprocess Audio and Get Initial Prediction
    # -------------------------------------------------
    print("Preprocessing audio and getting initial model prediction...")
    # Process the audio array to get the input tensors for the model
    inputs = processor(audio_array, sampling_rate=target_sr, return_tensors="pt")
    input_values = inputs.numpy()

    # Get the model's prediction (logits)
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the prediction to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print(f"Model's predicted transcription: {transcription}")

    # 3. Define the Prediction Function for SHAP
    # -------------------------------------------------
    # We need to explain why the model predicted a specific token at a specific time.
    # Let's find the first meaningful token to explain (not pad, unk, or space).
    sequence_length = predicted_ids.shape[1]
    timestep_to_explain = -1
    token_id_to_explain = -1
    
    # Iterate through predicted tokens to find a suitable one to explain
    for i in range(sequence_length):
        token_id = predicted_ids[0, i].item()
        # Check if the token is not a special/padding token
        if token_id not in processor.tokenizer.all_special_ids:
            # Also avoid explaining the word delimiter
            token_char_check = processor.tokenizer.convert_ids_to_tokens([token_id])[0]
            if token_char_check != "|":
                timestep_to_explain = i
                token_id_to_explain = token_id
                break
    
    # If no suitable token is found, fallback to the middle of the sequence
    if timestep_to_explain == -1:
        timestep_to_explain = sequence_length // 2
        token_id_to_explain = predicted_ids[0, timestep_to_explain].item()

    token_char = processor.tokenizer.convert_ids_to_tokens([token_id_to_explain])[0]

    print(f"\nExplaining prediction of token '{token_char}' (ID: {token_id_to_explain}) at timestep {timestep_to_explain}.")

    def predict_function(x):
        """
        This function takes a numpy array of input_values and returns the
        model's logit for the specific token and timestep we want to explain.
        SHAP uses this function to understand the model's behavior.
        """
        # Convert numpy array from SHAP back to a torch tensor
        input_tensor = torch.from_numpy(x).float().to(device)
        
        # Ensure tensor has the correct dimensions (batch, sequence_length)
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            logits_output = model(input_tensor).logits
        
        # Return the specific logit value we are interested in
        # SHAP expects a numpy array as output
        specific_logit = logits_output[:, timestep_to_explain, token_id_to_explain]
        return specific_logit.cpu().numpy()
    model = modelWrapper(model, device, timestep_to_explain, token_id_to_explain)
    # 4. Run the SHAP Explainer
    # -------------------------------------------------
    # We provide a background dataset of silence.
    # To prevent the kmeans algorithm from failing on identical data points,
    # we add a tiny amount of random noise.
    background = np.zeros((20, input_values.shape[1]))
    background += np.random.rand(20, input_values.shape[1]) * 1e-6


    # To make the KernelExplainer more efficient, we summarize the background data.
    # We choose 10 summary points from our silent background.
    background_summary = shap.kmeans(background, 10)

    # We explicitly use shap.KernelExplainer, which is suitable for this type of model
    # and avoids the high `max_evals` requirement of the Permutation explainer.
    explainer = shap.DeepExplainer(model, background_summary)

    print("\nCalculating SHAP values... (This may take several minutes)")
    # We pass the input values as a numpy array.
    # `nsamples` controls the number of perturbations for the explanation.
    shap_values = explainer.shap_values(input_values.cpu().numpy(), nsamples=500)
    print("SHAP calculation complete.")

    # 5. Visualize the SHAP values on a Mel Spectrogram
    # -------------------------------------------------
    print("Generating visualization...")

    # Calculate a Mel spectrogram of the original audio for visualization
    n_fft = 2048
    hop_length = 512
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_array, sr=target_sr, n_fft=n_fft, hop_length=hop_length
    )
    db_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # The model's feature sequence is shorter than the spectrogram's time axis.
    # We need to upsample the SHAP values to align them.
    shap_sequence_length = shap_values.shape[1]
    spectrogram_time_steps = db_mel_spectrogram.shape[1]
    
    # The KernelExplainer returns a numpy array directly.
    # The shape should be (1, sequence_length), so we select the first element.
    shap_importance = np.abs(shap_values[0])


    # Upsample the 1D importance map to match the spectrogram's time dimension
    ratio = spectrogram_time_steps / shap_sequence_length
    upsampled_shap = np.repeat(shap_importance, int(np.ceil(ratio)))
    upsampled_shap = upsampled_shap[:spectrogram_time_steps] # Trim to exact length

    # Create a 2D mask by broadcasting the upsampled SHAP values
    shap_mask = np.tile(upsampled_shap, (db_mel_spectrogram.shape[0], 1))
    
    # Normalize the mask for better visualization
    # Add a small epsilon to avoid division by zero if the mask is all zeros
    shap_mask_normalized = (shap_mask - shap_mask.min()) / (shap_mask.max() - shap_mask.min() + 1e-8)

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot the Mel spectrogram
    librosa.display.specshow(db_mel_spectrogram, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax)
    
    # Overlay the SHAP mask with transparency
    ax.imshow(
        shap_mask_normalized, 
        cmap='Reds', 
        aspect='auto', 
        alpha=0.6, 
        extent=[0, len(audio_array)/target_sr, 0, target_sr/2]
    )

    ax.set_title(f"SHAP Explanation for token '{token_char}'\nPredicted Text: {transcription}", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    
    # Add a color bar for the spectrogram
    cbar = fig.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB', label='Log Power (dB)')
    
    print("\nDisplaying plot. Close the plot window to exit the script.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

