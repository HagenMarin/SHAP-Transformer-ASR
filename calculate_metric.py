import numpy as np
import logging
import sys
import os
import glob

# --- Basic Setup (unchanged) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# --- select_files function (unchanged) ---
def select_files():
    """
    Scans the 'data/' folder for audio files, prompts the user to select one,
    and returns the paths for the audio, noise, and SHAP files.
    """
    data_dir = "data/"
    logger.info(f"Scanning for audio files in '{data_dir}'...")

    audio_files = sorted(glob.glob(os.path.join(data_dir, "audio_*.npy")))

    if not audio_files:
        logger.error(f"No audio files found in '{data_dir}'. Please ensure files are named 'audio_*.npy'.")
        return None, None, None

    print("\nPlease select an audio file to analyze:")
    for i, file_path in enumerate(audio_files):
        print(f"  [{i+1}] {os.path.basename(file_path)}")
    print("-" * 30)

    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(audio_files):
                selected_index = choice - 1
                break
            else:
                print(f"Invalid input. Please enter a number between 1 and {len(audio_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    selected_audio_path = audio_files[selected_index]
    base_name = os.path.basename(selected_audio_path)
    identifier = base_name.replace("audio_", "")
    
    noise_filename = f"noise_{identifier}"
    shap_filename = f"shap_values_{identifier}"
    
    expected_noise_path = os.path.join(data_dir, noise_filename)
    expected_shap_path = os.path.join(data_dir, shap_filename)

    for path, name in [(expected_noise_path, "Noise"), (expected_shap_path, "SHAP")]:
        if not os.path.exists(path):
            logger.error(f"Could not find the corresponding {name} file: '{os.path.basename(path)}'")
            return None, None, None

    logger.info(f"Selected audio: '{os.path.basename(selected_audio_path)}'")
    logger.info(f"Found corresponding noise file: '{noise_filename}'")
    logger.info(f"Found corresponding SHAP file: '{shap_filename}'")
    
    return selected_audio_path, expected_noise_path, expected_shap_path


# --- calculate_eta_raw function (unchanged) ---
def calculate_eta_raw(
    clean_audio: np.ndarray,
    noise_audio: np.ndarray,
    shap_matrix: np.ndarray,
    sr: int,
    segment_ms: int = 20,
    percentile: float = 99.0
) -> float:
    """
    Calculates the Raw Audio Speech Relevance Score (η_raw).
    """
    logger.info(f"Starting calculation of η_raw with segment_ms={segment_ms} and percentile={percentile}...")

    segment_length_samples = int(sr * (segment_ms / 1000.0))
    if segment_length_samples == 0:
        raise ValueError("segment_ms is too small, resulting in 0 samples per segment.")
    
    if shap_matrix.shape[0] != clean_audio.shape[0]:
        if shap_matrix.shape[1] == clean_audio.shape[0]:
            logger.warning("Transposing SHAP matrix to match expected (T, N) shape.")
            shap_matrix = shap_matrix.T
        else:
            raise ValueError(f"SHAP matrix shape {shap_matrix.shape} is incompatible with audio length {len(clean_audio)}.")

    min_len = min(len(clean_audio), len(noise_audio), shap_matrix.shape[0])
    num_segments = min_len // segment_length_samples
    trunc_len = num_segments * segment_length_samples

    clean_audio_trunc = clean_audio[:trunc_len]
    noise_audio_trunc = noise_audio[:trunc_len]
    shap_matrix_trunc = shap_matrix[:trunc_len, :]
    
    logger.info(f"Processing audio into {num_segments} segments of {segment_length_samples} samples each.")

    clean_segments = clean_audio_trunc.reshape(num_segments, segment_length_samples)
    noise_segments = noise_audio_trunc.reshape(num_segments, segment_length_samples)
    E_c = np.sum(np.square(clean_segments), axis=1)
    E_u = np.sum(np.square(noise_segments), axis=1)
    itm = (E_c > 0.5*E_u).astype(int)
    logger.info(f"Generated Ideal Time-domain Mask (ITM). Found {np.sum(itm)} speech-dominated segments.")

    phi_total = np.sum(np.abs(shap_matrix_trunc), axis=1)
    phi_total_segments = phi_total.reshape(num_segments, segment_length_samples)
    bar_phi = np.mean(phi_total_segments, axis=1)

    if len(bar_phi) == 0:
        logger.warning("No segments to process. Cannot calculate threshold.")
        return 0.0
    tau = np.percentile(bar_phi, percentile)
    logger.info(f"Calculated relevance threshold τ = {tau:.4f} ({percentile}th percentile).")
    is_highly_relevant = (bar_phi > tau).astype(int)
    denominator = np.sum(is_highly_relevant)
    logger.info(f"Found {denominator} segments with relevance > τ.")

    if denominator == 0:
        logger.warning("No segments were identified as highly relevant (denominator is 0). Returning η_raw = 0.")
        return 0.0
    intersection = is_highly_relevant * itm
    numerator = np.sum(intersection)
    logger.info(f"Found {numerator} segments in the intersection (highly relevant AND speech-dominated).")
    
    return numerator / denominator


if __name__ == '__main__':
    # --- Select files interactively ---
    audio_path, noise_path, shap_path = select_files()
    
    if not all([audio_path, noise_path, shap_path]):
        logger.error("Could not retrieve all necessary file paths. Exiting.")
        sys.exit(1)

    # --- Configuration ---
    SR = 16000
    
    # --- Load Data ---
    try:
        # The 'audio' file is the mixed signal (clean + noise)
        mixed_audio_signal = np.load(audio_path)
        noise_signal = np.load(noise_path)
        shap_values_matrix = np.load(shap_path)
    except Exception as e:
        logger.error(f"Error loading data from selected files: {e}")
        sys.exit(1)

    # --- ❗️ KEY CHANGE: Derive the clean audio signal ---
    # Ensure lengths match before subtraction by truncating to the shorter length
    min_len = min(len(mixed_audio_signal), len(noise_signal))
    mixed_audio_signal_trunc = mixed_audio_signal[:min_len]
    noise_signal_trunc = noise_signal[:min_len]
    
    # Calculate clean audio c(t) = x(t) - u(t)
    clean_speech_signal = mixed_audio_signal_trunc - noise_signal_trunc
    logger.info("Derived clean audio by subtracting noise from the mixed audio file.")
        
    # --- Check and remove extra dimensions from SHAP matrix ---
    if shap_values_matrix.ndim > 2:
        logger.warning(f"Original SHAP matrix has {shap_values_matrix.ndim} dimensions (shape: {shap_values_matrix.shape}). Squeezing to 2D.")
        shap_values_matrix = np.squeeze(shap_values_matrix)
        if shap_values_matrix.ndim > 2:
            logger.error("Squeezing failed to reduce SHAP matrix to 2 dimensions. Cannot proceed.")
            sys.exit(1)

    # --- Perform Calculation ---
    eta_raw_score = calculate_eta_raw(
        clean_audio=clean_speech_signal,
        noise_audio=noise_signal_trunc, # Use the truncated noise signal
        shap_matrix=shap_values_matrix,
        sr=SR,
        segment_ms=0.0625,
        percentile=99.0
    )

    # --- Print Results ---
    print("\n" + "="*50)
    print("      Raw Audio Speech Relevance Score (η_raw)      ")
    print("="*50)
    print(f"Analyzed File: {os.path.basename(audio_path)}")
    print(f"SCORE: {eta_raw_score:.4f} ({eta_raw_score:.2%})")
    print("\nThis score represents the proportion of the most influential audio segments")
    print("that correctly correspond to speech-dominated regions.")
    print("="*50)