"""Score the Swift pipeline's PCM against the PyTorch reference WAV.

Uses the same waveform-correlation + log-mel-spectrogram-correlation metrics as convert.py's E2E check.

    uv run python compare_validation.py output/ref.wav output/swift.pcm
"""

import sys

import numpy as np
import soundfile as sf
from scipy.signal import stft as scipy_stft


def mel_corr(a, b, sr=24000, n_fft=1024, hop=256, n_mels=80):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_lo, mel_hi = 0.0, 2595.0 * np.log10(1 + (sr / 2) / 700.0)
    mels = np.linspace(mel_lo, mel_hi, n_mels + 2)
    freqs = 700.0 * (10 ** (mels / 2595.0) - 1)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, mid, hi = freqs[i], freqs[i + 1], freqs[i + 2]
        up = (fft_freqs - lo) / max(mid - lo, 1e-10)
        down = (hi - fft_freqs) / max(hi - mid, 1e-10)
        fb[i] = np.maximum(0, np.minimum(up, down))

    def mel_spec(x):
        _, _, Zxx = scipy_stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        S = np.abs(Zxx) ** 2
        return np.log1p(fb @ S)

    A, B = mel_spec(a), mel_spec(b)
    return np.corrcoef(A.flatten(), B.flatten())[0, 1]


def main():
    ref_path, pcm_path = sys.argv[1], sys.argv[2]
    ref, sr = sf.read(ref_path, dtype="float32")
    swift = np.fromfile(pcm_path, dtype="<f4")
    n = min(len(ref), len(swift))
    corr = np.corrcoef(ref[:n], swift[:n])[0, 1]
    mc = mel_corr(ref, swift, sr=sr)
    print(f"len ref={len(ref)} swift={len(swift)}  corr={corr:.6f}  mel_corr={mc:.6f}")
    sf.write(pcm_path.replace(".pcm", ".wav"), swift, sr)


if __name__ == "__main__":
    main()
