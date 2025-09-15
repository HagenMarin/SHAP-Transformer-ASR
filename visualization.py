import glob
import logging
import os
import sys

import librosa
import librosa.display
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from matplotlib.widgets import Button
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class InteractiveShapVisualizer:
    def __init__(
        self,
        audio_data,
        sr,
        transcription,
        char_shap_audios_list,
        display_tokens,
        list_of_masks,
    ):
        self.audio = audio_data
        self.sr = sr
        self.full_transcription = transcription
        self.transcription = transcription.replace(" ", "")
        self.char_shap_audios_list = char_shap_audios_list
        self.display_tokens = display_tokens
        self.display_buttons = False  # Control button display
        self.list_of_masks = list_of_masks

        self.fig = plt.figure(figsize=(15, 12))
        self.fig.subplots_adjust(bottom=0.15)
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 0.5, 1.5, 3])

        self.ax_main_spec = self.fig.add_subplot(gs[0])
        self.ax_tokens = self.fig.add_subplot(gs[1], sharex=self.ax_main_spec)
        self.ax_text = self.fig.add_subplot(gs[2])
        self.ax_shap_spec = self.fig.add_subplot(gs[3], sharex=self.ax_main_spec)

        self.text_objects = []
        self.selected_index = -1
        self.default_color = "black"
        self.selected_color = "#007acc"

        self._setup_plots()
        if self.display_buttons:
            self._setup_buttons()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _setup_buttons(self):
        """Creates the export buttons."""
        # Define button positions [left, bottom, width, height]
        ax_btn_w = plt.axes([0.25, 0.05, 0.2, 0.05])
        ax_btn_c = plt.axes([0.55, 0.05, 0.2, 0.05])

        self.btn_export_weighted = Button(ax_btn_w, "Export Weighted Audio")
        self.btn_export_clipped = Button(ax_btn_c, "Export Clipped Original Audio")

        self.btn_export_weighted.on_clicked(self._export_weighted)
        self.btn_export_clipped.on_clicked(self._export_clipped_original)

        # Deactivate buttons initially
        self.btn_export_weighted.ax.set_visible(False)
        self.btn_export_clipped.ax.set_visible(False)
        self.fig.canvas.draw_idle()

    def _export_weighted(self, event):
        """Saves the SHAP-weighted audio for the selected character."""
        if self.selected_index == -1:
            return

        audio_to_save = self.char_shap_audios_list[self.selected_index]
        char = self.transcription[self.selected_index]
        filename = f"export_weighted_{char}_{self.selected_index}.wav"

        sf.write(filename, audio_to_save, self.sr)
        logger.info(f"Successfully saved weighted audio to '{filename}'")

    def _export_clipped_original(self, event):
        """Saves the original audio clipped using the SHAP mask."""
        if self.selected_index == -1:
            return

        mask = self.list_of_masks[self.selected_index]
        # A binary mask keeps the original audio only where SHAP values were important
        binary_mask = (mask > 0).astype(float)
        clipped_audio = self.audio * binary_mask

        char = self.transcription[self.selected_index]
        filename = f"export_clipped_original_{char}_{self.selected_index}.wav"

        sf.write(filename, clipped_audio, self.sr)
        logger.info(f"Successfully saved clipped original audio to '{filename}'")

    def _on_click(self, event):
        """Callback function for mouse click events."""
        if event.inaxes is not self.ax_text:
            return

        for i, text_obj in enumerate(self.text_objects):
            contains, _ = text_obj.contains(event)
            if contains:
                if i == self.selected_index:
                    return

                if self.selected_index == -1:
                    if self.display_buttons:
                        self.btn_export_weighted.ax.set_visible(True)
                        self.btn_export_clipped.ax.set_visible(True)

                if self.selected_index != -1:
                    self.text_objects[self.selected_index].set_color(self.default_color)

                text_obj.set_color(self.selected_color)
                self.selected_index = i
                self._update_shap_spectrogram(i)
                break

    def _plot_token_timeline(self):
        """Plots the raw model output tokens aligned with the spectrogram time axis."""
        ax = self.ax_tokens
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        total_duration = len(self.audio) / self.sr
        num_tokens = len(self.display_tokens)
        time_per_token = total_duration / num_tokens
        time_coords = np.arange(num_tokens) * time_per_token + (time_per_token / 2)
        for time, token in zip(time_coords, self.display_tokens):
            ax.text(
                time,
                0.5,
                token,
                ha="center",
                va="center",
                fontsize=9,
                fontfamily="monospace",
            )
        ax.set_ylabel(
            "Tokens", rotation=0, ha="right", va="center", fontsize=10, labelpad=10
        )
        plt.setp(ax.get_xticklabels(), visible=False)

    def _setup_plots(self):
        """Sets up the initial static plots."""
        # 1. Plot the main spectrogram
        S_original = librosa.feature.melspectrogram(y=self.audio, sr=self.sr)
        S_db_original = librosa.power_to_db(S_original, ref=np.max)
        librosa.display.specshow(
            S_db_original, sr=self.sr, x_axis="time", y_axis="mel", ax=self.ax_main_spec
        )
        self.ax_main_spec.set_title("Mel Spectrogram of Full Audio", fontsize=14)
        self.ax_main_spec.set_xlabel("")
        plt.setp(self.ax_main_spec.get_xticklabels(), visible=False)

        # 2. Plot token timeline
        self._plot_token_timeline()

        # 3. Display the transcription text
        self.ax_text.axis("off")
        self.ax_text.set_title(
            f'Clickable Transcription: "{self.full_transcription}"', style="italic"
        )
        if len(self.full_transcription) > 0:
            x_coords = np.linspace(0.05, 0.95, len(self.full_transcription))
        else:
            x_coords = []
        self.text_objects = []
        for i, char in enumerate(self.full_transcription):
            if char != " ":
                text_obj = self.ax_text.text(
                    x_coords[i],
                    0.5,
                    char,
                    ha="center",
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                    color=self.default_color,
                    transform=self.ax_text.transAxes,
                )
                self.text_objects.append(text_obj)

        # 4. Setup the SHAP spectrogram plot
        self.ax_shap_spec.set_title(
            "Click on a letter to see its SHAP-weighted Spectrogram", fontsize=14
        )
        self.ax_shap_spec.set_facecolor("#f0f0f0")
        self.ax_shap_spec.set_ylabel("Mel")
        self.ax_shap_spec.set_xlabel("Time")

        self.fig.suptitle("Interactive SHAP Value Visualization", fontsize=20)
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])

    def _update_shap_spectrogram(self, char_index):
        if char_index >= len(self.char_shap_audios_list):
            logger.warning(
                f"Character index {char_index} out of bounds for SHAP audio list."
            )
            return
        char = self.transcription[char_index]
        logger.info(f"Clicked on '{char}' (index {char_index})")
        shap_audio = self.char_shap_audios_list[char_index]
        S_shap = librosa.feature.melspectrogram(y=shap_audio, sr=self.sr)
        S_db_shap = librosa.power_to_db(S_shap, ref=np.max)
        self.ax_shap_spec.clear()
        librosa.display.specshow(
            S_db_shap, sr=self.sr, x_axis="time", y_axis="mel", ax=self.ax_shap_spec
        )
        self.ax_shap_spec.set_title(
            f"SHAP-weighted Spectrogram for letter: '{char}' at position {char_index+1}",
            fontsize=14,
        )
        self.fig.canvas.draw_idle()

    def _clear_shap_spectrogram(self):
        self.ax_shap_spec.clear()
        self.ax_shap_spec.set_title(
            "Click on a letter to see its SHAP-weighted Spectrogram", fontsize=14
        )
        self.ax_shap_spec.set_facecolor("#f0f0f0")
        self.ax_shap_spec.set_ylabel("Mel")
        self.ax_shap_spec.set_xlabel("Time")
        self.fig.canvas.draw_idle()


def _normalize_and_scale_shap(shap_vals, percentile=98.0, default=0.0):
    """
    Normalizes SHAP values and scales them so that only values above a
    certain percentile are retained.

    Args:
        shap_vals (np.ndarray): The array of SHAP values.
        percentile (float): The percentile (0-100) to use as a clipping threshold.
                            Only normalized values above this percentile's value will be kept.
        default (float): The default value for flat arrays.

    Returns:
        np.ndarray: The scaled SHAP values array.
    """
    shap_min, shap_max = np.min(shap_vals), np.max(shap_vals)

    # If the SHAP values are all the same, return a flat array of the default value
    if shap_max - shap_min < 1e-8:
        return np.full_like(shap_vals, default)

    # Normalize the values to the range [0, 1]
    normalized = (shap_vals - shap_min) / (shap_max - shap_min)

    # Calculate the threshold value based on the specified percentile of the normalized data
    clip_threshold = np.percentile(normalized, percentile)

    # Handle the edge case where the threshold is at or very close to the maximum
    # to avoid division by zero. In this case, just return a binary mask.
    if (1.0 - clip_threshold) < 1e-8:
        return (normalized >= clip_threshold).astype(float)

    # 1. Subtract the threshold and clip negative values to 0.
    # 2. Rescale the remaining values (from the threshold up to 1) to the [0, 1] range.
    scaled = ((normalized - clip_threshold).clip(0) / (1.0 - clip_threshold)).clip(
        default, 1
    )

    return scaled


def load_and_process_data(audio_path, shap_path):
    """
    Loads and processes data, returning audio, transcription, weighted audios,
    raw tokens, and the generated SHAP masks.
    """
    logger.info("Loading and processing real data...")
    sr = 16000

    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/wav2vec2-base-960h"
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        return None, None, None, None, None, None

    # 2. Load Audio, SHAP, and get Transcription
    try:
        audio = np.load(audio_path)
        shap_values_raw = np.load(shap_path)
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}. Please check your file paths.")
        return None, None, None, None, None, None

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    clean_transcription = transcription.replace(" ", "")
    logger.info(f"Transcription: {transcription}")

    # 2a. Get raw tokens for visualization
    predicted_ids_cpu = predicted_ids.squeeze().cpu().numpy()
    raw_tokens = processor.tokenizer.convert_ids_to_tokens(predicted_ids_cpu)
    pad_token = processor.tokenizer.pad_token
    display_tokens = [t if t != pad_token else "·" for t in raw_tokens]

    # 3. Identify character groups from model output for mapping
    blank_token_id = processor.tokenizer.pad_token_id
    space_token_id = processor.tokenizer.convert_tokens_to_ids("|")
    char_timestep_indices = []
    for i, token_id in enumerate(predicted_ids_cpu):
        if (token_id != blank_token_id and token_id != space_token_id) and (
            i == 0 or token_id != predicted_ids_cpu[i - 1]
        ):
            char_timestep_indices.append(i)

    if len(char_timestep_indices) != len(clean_transcription):
        logger.error(
            f"Mismatch between decoded characters ({len(clean_transcription)}) and found token groups ({len(char_timestep_indices)})."
        )
        return audio, sr, transcription, [], [], []

    # 4. Process SHAP values (unchanged)
    try:
        shap_values = shap_values_raw.squeeze()
        expected_shape = (audio.shape[0], logits.shape[1])
        if shap_values.shape == (expected_shape[1], expected_shape[0]):
            shap_values = shap_values.T
        if shap_values.shape != expected_shape:
            raise ValueError(
                f"Unexpected SHAP shape. Got {shap_values.shape}, expected {expected_shape}."
            )
    except Exception as e:
        logger.error(f"Could not process SHAP values shape: {e}")
        return audio, sr, transcription, [], [], []

    # 5. Generate an amplified audio for each character
    list_of_amplified_audios = []
    list_of_masks = []

    for i, char_timestep_index in enumerate(char_timestep_indices):
        char_shap_values = np.abs(shap_values[:, char_timestep_index])

        window_length_ms = 20
        num_of_frames = int(window_length_ms / 1000 * sr)
        smoothed_shap_values = np.copy(char_shap_values)
        for idx in range(0, len(smoothed_shap_values), num_of_frames):
            end_idx = min(idx + num_of_frames, len(smoothed_shap_values))
            mean = np.mean(smoothed_shap_values[idx:end_idx])
            smoothed_shap_values[idx:end_idx] = mean

        normalized_shap = _normalize_and_scale_shap(smoothed_shap_values)
        amplified_audio = audio * normalized_shap

        list_of_amplified_audios.append(amplified_audio)
        list_of_masks.append(normalized_shap)

    logger.info(
        f"Processed SHAP values for {len(list_of_amplified_audios)} characters."
    )
    return (
        audio,
        sr,
        transcription,
        list_of_amplified_audios,
        display_tokens,
        list_of_masks,
    )


def select_file_from_data_folder():
    """
    Scans the 'data/' folder for audio files, prompts the user to select one,
    and returns the paths for the audio and its corresponding SHAP file.
    """
    data_dir = "data/"
    logger.info(f"Scanning for audio files in '{data_dir}'...")

    # Find all files matching the audio naming pattern
    audio_files = sorted(glob.glob(os.path.join(data_dir, "audio_*.npy")))

    if not audio_files:
        logger.error(
            f"No audio files found in '{data_dir}'. Please ensure files are named 'audio_*.npy'."
        )
        return None, None

    # Display options to the user
    print("\nPlease select an audio file to analyze:")
    for i, file_path in enumerate(audio_files):
        print(f"  [{i+1}] {os.path.basename(file_path)}")
    print("-" * 30)

    # Get and validate user input
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(audio_files):
                selected_index = choice - 1
                break
            else:
                print(
                    f"Invalid input. Please enter a number between 1 and {len(audio_files)}."
                )
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Construct the paths based on user's choice
    selected_audio_path = audio_files[selected_index]

    # Derive the SHAP filename from the audio filename
    base_name = os.path.basename(selected_audio_path)
    identifier = base_name.replace("audio_", "")
    shap_filename = f"shap_values_{identifier}"
    expected_shap_path = os.path.join(data_dir, shap_filename)

    # Check if the corresponding SHAP file exists
    if not os.path.exists(expected_shap_path):
        logger.error(f"Could not find the corresponding SHAP file: '{shap_filename}'")
        logger.error(
            "Please ensure for every 'audio_X.npy' there is a 'shap_values_X.npy'."
        )
        return None, None

    logger.info(f"Selected audio: '{os.path.basename(selected_audio_path)}'")
    logger.info(f"Loading SHAP values: '{shap_filename}'")

    return selected_audio_path, expected_shap_path


def main():
    """
    Main function to run the visualization.
    """
    audio_path, shap_path = select_file_from_data_folder()

    if not audio_path or not shap_path:
        return

    audio, sr, transcription, char_audios_list, display_tokens, list_of_masks = (
        load_and_process_data(audio_path, shap_path)
    )

    if audio is None:
        return

    if not char_audios_list or not list_of_masks:
        logger.error(
            "SHAP audio or mask data list is empty. Cannot start visualization."
        )
        return

    visualizer = InteractiveShapVisualizer(
        audio, sr, transcription, char_audios_list, display_tokens, list_of_masks
    )
    plt.show()


if __name__ == "__main__":
    main()
