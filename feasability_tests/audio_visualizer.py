import sys
from tkinter import Tk, filedialog

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_ITER = 32

last_frame_idx = -1


def select_audio_file():
    """
    Opens a graphical file dialog for the user to select an audio file.
    Returns the path to the selected file.
    """
    print("Opening file dialog to select an audio file...")
    root = Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=(("Audio Files", "*.wav *.mp3 *.flac *.m4a"), ("All files", "*.*")),
    )
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

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    print("Pre-computing audio frames for playback... (This may take a moment)")
    n_frames = S.shape[1]
    audio_frames = [
        librosa.feature.inverse.mel_to_audio(
            S[:, i : i + 1], sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=N_ITER
        )
        for i in range(n_frames)
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

    if not event.inaxes:
        if last_frame_idx != -1:
            sd.stop()
            last_frame_idx = -1
            if ax.get_lines():
                ax.get_lines()[0].remove()
                fig.canvas.draw_idle()
        return

    x_data = int(round(event.xdata))
    n_frames = len(audio_frames)

    if 0 <= x_data < n_frames and x_data != last_frame_idx:
        last_frame_idx = x_data
        audio_to_play = audio_frames[x_data]

        sd.stop()
        sd.play(audio_to_play, sr, blocking=False)

        if ax.get_lines():
            ax.get_lines()[0].remove()
        ax.axvline(x=x_data, color="lime", linestyle="--", linewidth=1.5)
        fig.canvas.draw_idle()


def main():
    """Main function to set up and run the visualizer."""
    filepath = select_audio_file()
    S_db, audio_frames, sr = preprocess_audio(filepath)

    if S_db is None:
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.canvas.manager.set_window_title("Mel Spectrogram Audio Player")

    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH, x_axis="frames", y_axis="mel", ax=ax
    )

    ax.set_title('Hover Over the Spectrogram to "Scrub" Through Audio')
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Frequency (Mel Scale)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Intensity (dB)")
    plt.tight_layout()

    fig.canvas.mpl_connect(
        "motion_notify_event",
        lambda event: on_mouse_move(event, fig, ax, audio_frames, sr),
    )

    print("\nVisualization is ready. Move your mouse over the plot.")
    plt.show()


if __name__ == "__main__":
    main()
