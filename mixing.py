import random as rand
import numpy as np
import soundfile as sf

import librosa

from pathlib import Path

SAMPLE_RATE = 16000
CLIP_DURATION = 3.0
N_CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
SNR_RANGE = (-5, 15) # dB
N_TRAIN = 2000
N_VALIDATION = 200

CLEAN_DIR = Path("data/LibriSpeech/dev-clean")
NOISE_DIR = Path("data/ESC-50-master/audio")
OUTPUT_DIR = Path("data/mixed")

def load_and_sample(path: Path, target_sr = SAMPLE_RATE):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1) # for mono audio
    
    if sr != target_sr:
        audio = librosa.resample(
            audio.astype(np.float32), 
            orig_sr = sr,
            target_sr = target_sr
        )
    
    return audio.astype(np.float32)

# TODO Needs testing. Got from Claude so double check.
def mix(clean, noise, snr_db):

    n = min(len(clean), len(noise))
    
    # need both are same length
    clean = clean[:n]
    noise = noise[:n]


    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    
    scale = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))

    noisy = clean + (noise * scale)

    # Dont want to have clipping in the audio
    peak = np.max(np.abs(noisy))

    if peak > 0.99:
        noisy = noisy / peak * 0.99
        clean = clean / peak * 0.99
    
    return noisy.astype(np.float32), clean.astype(np.float32)

def main():
    clean_files = list(CLEAN_DIR.rglob("*.flac"))
    noise_files = list(NOISE_DIR.rglob("*.wav"))

    print(f"Got {len(clean_files)} clean files and {len(noise_files)} noise files")

    for split, n in [("train", N_TRAIN), ("validation", N_VALIDATION)]:
        # out_clean = Path(f"{OUTPUT_DIR}/{split}/clean")
        out_clean: Path = OUTPUT_DIR / split / "clean"
        out_noisy: Path = OUTPUT_DIR / split / "noisy"
        
        out_clean.mkdir(parents=True, exist_ok=True)
        out_noisy.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            clean = load_and_sample(rand.choice(clean_files))
            noise = load_and_sample(rand.choice(noise_files))
            snr = rand.uniform(SNR_RANGE[0], SNR_RANGE[1])
            
            noisy_audio, clean_audio = mix(clean, noise, snr)

            sf.write(out_noisy / f"{i:05d}.wav", noisy_audio, SAMPLE_RATE)
            sf.write(out_clean / f"{i:05d}.wav", clean_audio, SAMPLE_RATE)

            # For double checking
            if i + 1 % 100 == 0:
                print(f"{split}: {i+1}/{n}")

    print("Finished")

if __name__ == "__main__":
    main()