import os
import json
import time
import random
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import correlate
from tqdm import tqdm
import paths

try:
    import pyworld as pw
    HAS_PYWORLD = True
except ImportError:
    HAS_PYWORLD = False

SAMPLE_RATE = 16000
N_FILES     = 10_000
SEED        = 43

SOTA_FFT_SIZE = 1024
SOTA_HOP_SIZE = int(SAMPLE_RATE * 0.025)
SOTA_WIN_SIZE = int(SAMPLE_RATE * 0.050)

FINE_FFT = 4096
FINE_WIN = 1024
BLUR_FFT = 1024
BLUR_WIN = 1024

HOP_SPECTRAL_A = 400
HOP_SPECTRAL_B = 20

PROTOCOL_2021_PARTS = paths.PROTOCOL_2021_PARTS
FLAC_2021_PARTS     = paths.FLAC_2021_PARTS

OUTPUT_DIR = paths.output_dir("run_timing_benchmark.py")

def parse_2021_pa(protocol_file: str, flac_dir: str):
    entries = []
    with open(protocol_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            path = os.path.join(flac_dir, parts[1] + ".flac")
            if os.path.exists(path):
                entries.append(path)
    return entries

def gather_files(n_files: int):
    all_paths = []
    for proto, flac in zip(PROTOCOL_2021_PARTS, FLAC_2021_PARTS):
        if os.path.exists(proto):
            all_paths.extend(parse_2021_pa(proto, flac))
        if len(all_paths) >= n_files * 2:
            break
    if not all_paths:
        raise RuntimeError("No audio files found - check dataset paths in this script.")
    random.Random(SEED).shuffle(all_paths)
    return all_paths[:n_files]

def load_audio(path: str):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.squeeze()

def _log_spectrogram_sota(wav: torch.Tensor):
    window = torch.hann_window(SOTA_WIN_SIZE, device=wav.device)
    stft = torch.stft(wav, n_fft=SOTA_FFT_SIZE, hop_length=SOTA_HOP_SIZE,
                       win_length=SOTA_WIN_SIZE, window=window, return_complex=True)
    return torch.log(stft.abs().clamp(min=1e-10))

def world_vocode(wav: torch.Tensor):
    wav_np = wav.cpu().numpy().astype(np.float32)
    if HAS_PYWORLD:
        wav64 = wav_np.astype(np.float64)
        _min_samples = int(SAMPLE_RATE * 0.1)
        _rms = np.sqrt(np.mean(wav64 ** 2))
        if (len(wav64) < _min_samples or _rms < 1e-6 or not np.isfinite(wav64).all()):
            return wav
        f0, sp, ap = pw.wav2world(wav64, SAMPLE_RATE)
        out = pw.synthesize(f0, sp, ap, SAMPLE_RATE).astype(np.float32)
    else:
        pre = torchaudio.functional.preemphasis(wav, coeff=0.97)
        out = torchaudio.functional.deemphasis(pre, coeff=0.97).numpy()

    corr = correlate(wav_np, out, mode="full", method="fft")
    best_shift = int(np.argmax(corr)) - (len(out) - 1)
    if best_shift > 0:
        out = np.pad(out, (best_shift, 0))[:len(wav_np)]
    else:
        out = out[-best_shift:][:len(wav_np)]
    if len(out) < len(wav_np):
        out = np.pad(out, (0, len(wav_np) - len(out)))
    else:
        out = out[:len(wav_np)]
    return torch.from_numpy(out).to(wav.device)

def extract_sota_hvr(wav: torch.Tensor):

    log_X = _log_spectrogram_sota(wav)
    log_Xv = _log_spectrogram_sota(world_vocode(wav))
    T = min(log_X.shape[1], log_Xv.shape[1])
    return (log_Xv[:, :T] - log_X[:, :T]).T.contiguous().numpy().mean(axis=0)

def _log_mag(wav: torch.Tensor, n_fft: int, hop: int, win: int):
    window = torch.hann_window(win, device=wav.device)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop,
                       win_length=win, window=window, return_complex=True)
    return torch.log(stft.abs().clamp(min=1e-10))

def make_spectral_detail_extractor(hop: int):
    fine_bins = FINE_FFT // 2 + 1

    def extract(wav: torch.Tensor):
        log_fine = _log_mag(wav, FINE_FFT, hop, FINE_WIN)
        log_blur = _log_mag(wav, BLUR_FFT, hop, BLUR_WIN)

        T = min(log_fine.shape[1], log_blur.shape[1])
        log_fine = log_fine[:, :T]
        log_blur = log_blur[:, :T]

        log_blur_up = F.interpolate(
            log_blur.unsqueeze(0).unsqueeze(0).float(),
            size=(fine_bins, T),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

        diff = log_fine.float() - log_blur_up
        return diff.mean(dim=1).cpu().numpy()

    return extract

def benchmark(name: str, extract_fn, wavs: list):

    times = []
    n_fail = 0
    for wav in tqdm(wavs, desc=f"  {name}", leave=False):
        t0 = time.perf_counter()
        try:
            extract_fn(wav)
        except Exception:
            n_fail += 1
            continue
        times.append(time.perf_counter() - t0)

    times = np.array(times, dtype=np.float64)
    return dict(
        name=name,
        n_ok=len(times),
        n_fail=n_fail,
        total_s=float(times.sum()),
        mean_s=float(times.mean()) if len(times) else float("nan"),
        median_s=float(np.median(times)) if len(times) else float("nan"),
        p95_s=float(np.percentile(times, 95)) if len(times) else float("nan"),
        min_s=float(times.min()) if len(times) else float("nan"),
        max_s=float(times.max()) if len(times) else float("nan"),
        files_per_sec=float(len(times) / times.sum()) if times.sum() > 0 else float("nan"),
    )

def print_summary(results: list):
    print()
    print("=" * 92)
    print("TEST-INFERENCE TIMING COMPARISON")
    print("=" * 92)
    header = f"{'Method':<28} {'n_ok':>7} {'total_s':>10} {'mean_ms':>10} {'median_ms':>10} {'p95_ms':>10} {'files/s':>10}"
    print(header)
    print("-" * 92)
    for r in results:
        print(f"{r['name']:<28} {r['n_ok']:>7} {r['total_s']:>10.2f} "
              f"{r['mean_s']*1000:>10.3f} {r['median_s']*1000:>10.3f} "
              f"{r['p95_s']*1000:>10.3f} {r['files_per_sec']:>10.1f}")
    print("-" * 92)

    sota = next(r for r in results if r["name"].startswith("SOTA"))
    print()
    print("SPEEDUP vs SOTA (WORLD vocoder), by mean per-file time:")
    for r in results:
        if r is sota:
            continue
        speedup = sota["mean_s"] / r["mean_s"] if r["mean_s"] > 0 else float("nan")
        print(f"  {r['name']:<28} {speedup:>8.2f}x faster")
    print("=" * 92)

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"pyworld available: {HAS_PYWORLD}")
    print(f"Gathering {N_FILES} audio files ...")
    paths = gather_files(N_FILES)
    print(f"Got {len(paths)} files. Loading into memory (not timed) ...")

    wavs = []
    bad = 0
    for p in tqdm(paths, desc="  loading audio"):
        try:
            wavs.append(load_audio(p))
        except Exception:
            bad += 1
    print(f"Loaded {len(wavs)} waveforms ({bad} failed to load).")

    methods = [
        ("SOTA (WORLD vocoder)", extract_sota_hvr),
        (f"SpectralDetail hop={HOP_SPECTRAL_A}", make_spectral_detail_extractor(HOP_SPECTRAL_A)),
        (f"SpectralDetail hop={HOP_SPECTRAL_B}", make_spectral_detail_extractor(HOP_SPECTRAL_B)),
    ]

    warmup_wavs = wavs[:5]
    for _, fn in methods:
        for w in warmup_wavs:
            try:
                fn(w)
            except Exception:
                pass

    results = []
    for name, fn in methods:
        print(f"\nBenchmarking: {name}")
        res = benchmark(name, fn, wavs)
        results.append(res)
        print(f"  {name}: {res['n_ok']} ok, {res['n_fail']} failed, "
              f"mean={res['mean_s']*1000:.3f} ms, total={res['total_s']:.2f} s")

    print_summary(results)

    out_path = os.path.join(OUTPUT_DIR, f"inference_timing_comparison_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(dict(
            run_id=run_id,
            n_files_requested=N_FILES,
            n_files_loaded=len(wavs),
            sample_rate=SAMPLE_RATE,
            sota_config=dict(fft=SOTA_FFT_SIZE, hop=SOTA_HOP_SIZE, win=SOTA_WIN_SIZE),
            spectral_config=dict(fine_fft=FINE_FFT, fine_win=FINE_WIN,
                                  blur_fft=BLUR_FFT, blur_win=BLUR_WIN,
                                  hops=[HOP_SPECTRAL_A, HOP_SPECTRAL_B]),
            results=results,
        ), f, indent=2)
    print(f"\nResults saved -> {out_path}")

if __name__ == "__main__":
    main()
