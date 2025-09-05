#
# Mel Spectrogram Audio Visualizer
#
# This script loads an audio file, displays its Mel spectrogram, and allows
# the user to "play" snippets of the audio by hovering the mouse over the
# corresponding time frame in the plot.
#
# Required libraries:
# - librosa: For audio analysis and spectrogram generation.
# - numpy: For numerical operations.
# - matplotlib: For plotting and handling mouse events.
# - sounddevice: For audio playback.
#
# You can install these using pip:
# pip install librosa numpy matplotlib sounddevice
#
# Note: On some systems, you may also need to install FFmpeg for librosa
# to load compressed audio formats like MP3.
#
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
from tkinter import Tk, filedialog
import sys

# --- Configuration for Spectrogram and Audio Synthesis ---
# These values determine the resolution of the spectrogram and the quality
# of the reconstructed audio.
N_FFT = 2048      # Number of FFT components
HOP_LENGTH = 512  # Samples between successive frames
N_MELS = 128      # Number of Mel bands
N_ITER = 32       # Griffin-Lim iterations for phase reconstruction

# --- Global variable to track the last played frame to prevent re-playing ---
last_frame_idx = -1

def select_audio_file():
    """
    Opens a graphical file dialog for the user to select an audio file.
    Returns the path to the selected file.
    """
    print("Opening file dialog to select an audio file...")
    # We use Tkinter for a native-looking file dialog.
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    filepath = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=(("Audio Files", "*.wav *.mp3 *.flac *.m4a"), ("All files", "*.*"))
    )
    # If the user cancels, exit gracefully.
    if not filepath:
        print("No file selected. Exiting.")
        sys.exit()
    return filepath

def preprocess_audio(filepath):
    """
    Loads an audio file, creates its Mel spectrogram, and pre-computes
    the audio waveform for each individual time frame of the spectrogram.

    Args:
        filepath (str): The path to the audio file.

    Returns:
        tuple: A tuple containing:
            - S_db (np.ndarray): The Mel spectrogram in decibels.
            - audio_frames (list): A list of short audio waveforms, one for each frame.
            - sr (int): The sample rate of the audio.
    """
    try:
        print(f"Loading audio file: {filepath}...")
        y, sr = librosa.load(filepath, sr=None)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None

    # 1. Create the Mel spectrogram from the audio.
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    # 2. Convert to decibels for a more human-readable visual representation.
    S_db = librosa.power_to_db(S, ref=np.max)

    # 3. Pre-compute audio frames for real-time playback.
    # This is the most computationally intensive part. We use the Griffin-Lim
    # algorithm to estimate the audio waveform for each vertical slice (frame)
    # of the Mel spectrogram.
    print("Pre-computing audio frames for playback... (This may take a moment)")
    n_frames = S.shape[1]
    audio_frames = [
        librosa.feature.inverse.mel_to_audio(
            S[:, i:i+1], sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=N_ITER
        ) for i in range(n_frames)
    ]
    print("Pre-computation complete.")

    return S_db, audio_frames, sr

def on_mouse_move(event, fig, ax, audio_frames, sr):
    """
    Callback function to handle mouse movement over the spectrogram plot.
    It identifies the time frame under the cursor and plays the corresponding
    pre-computed audio snippet.
    """
    global last_frame_idx

    # Do nothing if the mouse is not within the plot axes.
    if not event.inaxes:
        # Stop playback and remove indicator when mouse leaves the plot
        if last_frame_idx != -1:
            sd.stop()
            last_frame_idx = -1
            # Remove the vertical line
            if ax.get_lines():
                ax.get_lines()[0].remove()
                fig.canvas.draw_idle()
        return

    # Get the x-coordinate of the mouse in data terms (frame index).
    x_data = int(round(event.xdata))
    n_frames = len(audio_frames)

    # Play audio only if the cursor is on a new, valid frame.
    if 0 <= x_data < n_frames and x_data != last_frame_idx:
        last_frame_idx = x_data
        audio_to_play = audio_frames[x_data]

        # Stop any sound that is currently playing and play the new one.
        # 'blocking=False' ensures the GUI remains responsive.
        sd.stop()
        sd.play(audio_to_play, sr, blocking=False)

        # --- Update Visual Indicator ---
        # Remove the previous vertical line.
        if ax.get_lines():
            ax.get_lines()[0].remove()
        # Draw a new line at the current frame.
        ax.axvline(x=x_data, color='lime', linestyle='--', linewidth=1.5)
        fig.canvas.draw_idle()

def main():
    """Main function to set up and run the visualizer."""
    filepath = select_audio_file()
    S_db, audio_frames, sr = preprocess_audio(filepath)

    if S_db is None:
        return

    # --- Create and configure the Matplotlib plot ---
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.canvas.manager.set_window_title('Mel Spectrogram Audio Player')

    # Display the spectrogram.
    img = librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH, x_axis='frames', y_axis='mel', ax=ax)

    # Set plot aesthetics.
    ax.set_title('Hover Over the Spectrogram to "Scrub" Through Audio')
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Frequency (Mel Scale)')
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Intensity (dB)')
    plt.tight_layout()

    # --- Connect the mouse event handler ---
    # The lambda function passes the required arguments to our callback.
    fig.canvas.mpl_connect(
        'motion_notify_event',
        lambda event: on_mouse_move(event, fig, ax, audio_frames, sr)
    )

    print("\nVisualization is ready. Move your mouse over the plot.")
    plt.show()

if __name__ == '__main__':
    main()
