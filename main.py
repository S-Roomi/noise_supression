import torch
import torchaudio
import soundfile as sf
import numpy as np
from speechbrain.inference.enhancement import SpectralMaskEnhancement

def main():
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank"
    )

    audio_np, sample_rate = sf.read("tests/test.wav")
    

    # For converting stereo to mono (need more info)
    if audio_np.ndim == 1:
        noisy_audio = torch.from_numpy(audio_np).float().unsqueeze(0)
    else:
        # stereo or more — transpose and average to mono
        noisy = torch.from_numpy(audio_np).float().T
        noisy = noisy.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        noisy_audio = resampler(noisy_audio)
    
    lengths = torch.tensor([1.0])
    enhanced = enhancer.enhance_batch(noisy_audio, lengths)

    enhanced_np = enhanced.cpu().squeeze().numpy()
    sf.write("enhanced.wav", enhanced_np, 16000)

    print("Done. Listen to enhanced.wav")
    
if __name__ == "__main__":
    main()
