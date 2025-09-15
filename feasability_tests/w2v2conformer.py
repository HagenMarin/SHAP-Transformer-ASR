import warnings

import datasets
import librosa
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

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
        input_tensor = torch.from_numpy(input_values).float().to(self.device)

        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            logits_output = self.model(input_tensor).logits

        specific_logit = logits_output[
            :, self.timestep_to_explain, self.token_id_to_explain
        ]
        return specific_logit.cpu().numpy()


def main():
    """
    Main function to run the SHAP analysis on a Conformer ASR model.
    """
    print("--- ASR Conformer SHAP Explainer ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        print("Loading model and processor...")
        model_id = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ConformerForCTC.from_pretrained(model_id).to(device)

        print("Loading audio sample from LibriSpeech dataset...")
        dataset = datasets.load_dataset(
            "librispeech_asr", "clean", split="validation", streaming=True
        )
        sample = next(iter(dataset))

        target_sr = processor.feature_extractor.sampling_rate
        audio_array = librosa.resample(
            sample["audio"]["array"],
            orig_sr=sample["audio"]["sampling_rate"],
            target_sr=target_sr,
        )
        print(f"Original audio text: {sample['text']}")

    except Exception as e:
        print(f"Error during setup: {e}")
        print(
            "Please ensure you have an internet connection and the required libraries are installed."
        )
        return

    print("Preprocessing audio and getting initial model prediction...")
    inputs = processor(audio_array, sampling_rate=target_sr, return_tensors="pt")
    input_values = inputs.numpy()

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print(f"Model's predicted transcription: {transcription}")

    sequence_length = predicted_ids.shape[1]
    timestep_to_explain = -1
    token_id_to_explain = -1

    for i in range(sequence_length):
        token_id = predicted_ids[0, i].item()
        if token_id not in processor.tokenizer.all_special_ids:
            token_char_check = processor.tokenizer.convert_ids_to_tokens([token_id])[0]
            if token_char_check != "|":
                timestep_to_explain = i
                token_id_to_explain = token_id
                break

    if timestep_to_explain == -1:
        timestep_to_explain = sequence_length // 2
        token_id_to_explain = predicted_ids[0, timestep_to_explain].item()

    token_char = processor.tokenizer.convert_ids_to_tokens([token_id_to_explain])[0]

    print(
        f"\nExplaining prediction of token '{token_char}' (ID: {token_id_to_explain}) at timestep {timestep_to_explain}."
    )

    def predict_function(x):
        """
        This function takes a numpy array of input_values and returns the
        model's logit for the specific token and timestep we want to explain.
        SHAP uses this function to understand the model's behavior.
        """
        input_tensor = torch.from_numpy(x).float().to(device)

        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            logits_output = model(input_tensor).logits

        specific_logit = logits_output[:, timestep_to_explain, token_id_to_explain]
        return specific_logit.cpu().numpy()

    model = modelWrapper(model, device, timestep_to_explain, token_id_to_explain)
    background = np.zeros((20, input_values.shape[1]))
    background += np.random.rand(20, input_values.shape[1]) * 1e-6

    background_summary = shap.kmeans(background, 10)

    explainer = shap.DeepExplainer(model, background_summary)

    print("\nCalculating SHAP values... (This may take several minutes)")
    shap_values = explainer.shap_values(input_values.cpu().numpy(), nsamples=500)
    print("SHAP calculation complete.")

    print("Generating visualization...")

    n_fft = 2048
    hop_length = 512
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_array, sr=target_sr, n_fft=n_fft, hop_length=hop_length
    )
    db_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    shap_sequence_length = shap_values.shape[1]
    spectrogram_time_steps = db_mel_spectrogram.shape[1]

    shap_importance = np.abs(shap_values[0])

    ratio = spectrogram_time_steps / shap_sequence_length
    upsampled_shap = np.repeat(shap_importance, int(np.ceil(ratio)))

    shap_mask = np.tile(upsampled_shap, (db_mel_spectrogram.shape[0], 1))

    shap_mask_normalized = (shap_mask - shap_mask.min()) / (
        shap_mask.max() - shap_mask.min() + 1e-8
    )

    fig, ax = plt.subplots(figsize=(15, 6))

    librosa.display.specshow(
        db_mel_spectrogram,
        sr=target_sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )

    ax.imshow(
        shap_mask_normalized,
        cmap="Reds",
        aspect="auto",
        alpha=0.6,
        extent=[0, len(audio_array) / target_sr, 0, target_sr / 2],
    )

    ax.set_title(
        f"SHAP Explanation for token '{token_char}'\nPredicted Text: {transcription}",
        fontsize=16,
    )
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)

    cbar = fig.colorbar(
        ax.collections[0], ax=ax, format="%+2.0f dB", label="Log Power (dB)"
    )

    print("\nDisplaying plot. Close the plot window to exit the script.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
