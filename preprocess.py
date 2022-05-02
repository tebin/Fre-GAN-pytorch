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


def compute_melspectrogram(wav, sr, n_fft=1024, num_mels=80, hop_size=240, win_size=1024, fmin=0, fmax=12000, center=True):
    y = torch.tensor(wav, device="cuda").unsqueeze(0)

    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    if not center:
        y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
        y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec.squeeze(0).cpu().numpy()


def main(dir, split, chunk_size, hop_size, sampling_rate, fmin, fmax):
    outdir = f'{dir}_out'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    paths = list(Path(dir).rglob('*.wav')) + list(Path(dir).rglob('*mic2.flac'))
    i = 0
    for path in tqdm(paths):
        wav = load_audio(path)
        mel = compute_melspectrogram(wav, sampling_rate, fmin=fmin, fmax=fmax)
        if split:
            mel_indices = np.arange(0, mel.shape[1], chunk_size)[1:]
            wav_indices = mel_indices * hop_size
            mel = np.split(mel, mel_indices, axis=1)
            wav = np.split(wav, wav_indices)
            for mel, wav in zip(mel, wav):
                if mel.shape[1] == chunk_size and wav.shape[0] == chunk_size * hop_size:
                    np.savez(f'{outdir}/{i}.npz', mel=mel, wav=wav)
                    i += 1
        else:
            np.savez(f'{outdir}/{i}.npz', mel=mel, wav=wav)
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--split', action='store_true', default=False)
    parser.add_argument('--chunk_size', default=32, type=int)
    parser.add_argument('--hop_size', default=240, type=int)
    parser.add_argument('--sampling_rate', default=24000, type=int)
    parser.add_argument('--fmin', default=0, type=int)
    parser.add_argument('--fmax', default=12000, type=int)
    args = parser.parse_args()
    main(args.dir, args.split, args.chunk_size, args.hop_size, args.sampling_rate, args.fmin, args.fmax)
