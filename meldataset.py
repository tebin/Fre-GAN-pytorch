import math
import os
from pathlib import Path
import random
import torch
import torch.utils.data
import numpy as np
import librosa
from librosa.util import normalize
import soundfile as sf
from librosa.filters import mel as librosa_mel_fn
import fastrand

MAX_WAV_VALUE = 32768.0


def load_audio(path, resample=24000):
    wav, sr = sf.read(path, dtype='float32')
    wav = wav.T
    wav = librosa.to_mono(wav)
    wav = librosa.resample(wav, sr, resample, res_type='scipy')
    return np.clip(wav, -1.0, 1.0)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
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

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, dir, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None):
        self.audio_files = list(Path(dir).rglob('*.npz'))
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.frames_per_seg = self.segment_size // self.hop_size

    def __getitem__(self, index):
        filename = self.audio_files[index]
        npz = np.load(filename)
        mel, audio = torch.FloatTensor(npz['mel']), torch.FloatTensor(npz['wav'])
        if self.split:
            if len(audio) > self.segment_size:
                mel_start = fastrand.pcg32bounded(mel.size(1) - self.frames_per_seg)
                mel = mel[:, mel_start:mel_start + self.frames_per_seg]
                audio = audio[mel_start * self.hop_size:(mel_start + self.frames_per_seg) * self.hop_size]
            else:
                mel = torch.nn.functional.pad(mel, (0, self.frames_per_seg - mel.size(1)), 'constant')
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - len(audio)), 'constant')
        return (mel, audio)

    def __len__(self):
        return len(self.audio_files)
