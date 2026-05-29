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
N_VAL = 200

CLEAN_DIR = Path("data/LibriSpeech/dev-clean")
NOISE_DIR = Path("data/ESC-50-master/audio")


def load_and_sample(path: Path, target_sr = SAMPLE_RATE):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1) # for mono audio
    
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), 
                                 orig_sr = sr,
                                  target_sr = target_sr)
    
    return audio.astype(np.float32)

def main():
    load_and_sample(CLEAN_DIR)
    load_and_sample(NOISE_DIR)

if __name__ == "__main__":
    main()