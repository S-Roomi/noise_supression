import torch
import torchaudio
import librosa
import librosa.display

import soundfile as sf
import numpy as np
from speechbrain.inference.enhancement import SpectralMaskEnhancement
import matplotlib.pyplot as plot

from pathlib import Path

def plot_spectogram(audio: np.float32, sr: int, ax, title: str):
    # Compute Short Time Fourier Transform
    stft = librosa.stft(audio, n_fft=512, hop_length=128)

    magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    img = librosa.display.specshow(
        magnitude_db,
        sr = sr,
        hop_length = 128,
        x_axis = "time",
        y_axis = "hz",
        ax = ax,
        cmap = "magma"
    )
    ax.set_title(title)
    return img

def plot_files(starting_file: str, result_file: str, output_filename: str):
    starting_file_path = Path(starting_file).resolve()
    result_file_path = Path(result_file).resolve()

    start_audio, start_sr = sf.read(starting_file_path)
    result_audio, result_sr = sf.read(result_file_path)

    fig, axes = plot.subplots(2, 1, figsize=(12, 8))
    plot_spectogram(start_audio.astype(np.float32), start_sr, axes[0], starting_file_path.name)
    plot_spectogram(result_audio.astype(np.float32), result_sr, axes[1], result_file_path.name)
    plot.tight_layout()
    plot.savefig(f"{result_file_path.parent}/{output_filename}.png", dpi = 100)
    print("Saved image")

def clean_audio(starting_file: str) -> None: 
    cleaner = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="pretrained_models/metricgan-plus-voicebank",
            run_opts={"device": "cuda:0"}
        )

    noisy_np, sample_rate = sf.read(starting_file)
    
    # For converting stereo to mono (need more info)
    if noisy_np.ndim == 1:
        noisy_audio = torch.from_numpy(noisy_np).float().unsqueeze(0)
    else:
        # stereo or more — transpose and average to mono
        noisy = torch.from_numpy(noisy_np).float().T
        noisy = noisy.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        noisy_audio = resampler(noisy_audio)
    
    lengths = torch.tensor([1.0])
    clean = cleaner.enhance_batch(noisy_audio, lengths)

    clean_np = clean.cpu().squeeze().numpy()

    starting_file_path = Path(starting_file).resolve().parent
    sf.write(f"{starting_file_path}/clean.wav", clean_np, 16000)

    print("Done. Listen to clean.wav")


def main():
    clean_audio("tests/test1/test_noisy.wav")
    # Print Spectrogram
    plot_files("tests/test1/test_noisy.wav", "tests/test1/clean.wav", "plots")

if __name__ == "__main__":
    main()
