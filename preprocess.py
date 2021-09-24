import argparse
from pathlib import Path

import librosa
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


mel_basis = {}
hann_window = {}


def load_audio(path, resample=24000):
    wav, sr = sf.read(path, dtype='float32')
    wav = wav.T
    wav = librosa.to_mono(wav)
    wav = librosa.resample(wav, sr, resample, res_type='scipy')
    return np.clip(wav, -1.0, 1.0)


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def compute_melspectrogram(wav, sr, n_fft=1024, num_mels=80, sampling_rate=24000, hop_size=240, win_size=1024, fmin=0.0, fmax=8000.0, center=False):
    y = torch.tensor(wav, device="cuda").unsqueeze(0)

    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec.squeeze(0).cpu().numpy()


def main(dir):
    paths = list(Path(dir).rglob('*.wav')) + list(Path(dir).rglob('*mic2.flac'))
    for path in tqdm(paths):
        wav = load_audio(path)
        mel = compute_melspectrogram(wav, 24000)
        path = f"{str(path)[:-4]}.npy"
        np.save(path, mel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    main(args.dir)
