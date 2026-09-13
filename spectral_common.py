import glob
import os
import numpy as np
import torch

def _log_mag(wav: torch.Tensor, n_fft: int, hop: int, win: int):
    
    if win > n_fft:
        raise ValueError(f"STFT invalid: win_length ({win}) > n_fft ({n_fft})")
    if hop <= 0:
        raise ValueError(f"STFT invalid: hop_length must be > 0 (got {hop})")
    if win <= 0:
        raise ValueError(f"STFT invalid: win_length must be > 0 (got {win})")
    if n_fft <= 0:
        raise ValueError(f"STFT invalid: n_fft must be > 0 (got {n_fft})")

    window = torch.hann_window(win, device=wav.device)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop,
                      win_length=win, window=window, return_complex=True)
    return torch.log(stft.abs().clamp(min=1e-10))

def _safe_name(name: str):
    return "".join([c if c.isalnum() else "_" for c in name]).strip("_")

def _maybe_remake_cache(output_dir: str, feat_safe: str, remake_train: bool, remake_eval: bool):
    patterns = []
    if remake_train:
        patterns.append(os.path.join(output_dir, f"train_{feat_safe}_*.npz"))
    if remake_eval:
        patterns.append(os.path.join(output_dir, f"eval_{feat_safe}_*.npz"))
    if not patterns:
        return
    removed = 0
    for pat in patterns:
        print('  cache remake: deleting %s' % (pat,))
        for p in glob.glob(pat):
            os.remove(p)
            removed += 1
            print('    removed %s' % (p,))
    print('  cache remake: %d file(s) removed' % (removed,))

def chunk_list(items: list, chunk_size: int):
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i:i+chunk_size])
    return chunks
def save_scores(
    score_txt_dir: str,
    feat_name: str,
    fp_feat: str,
    eval_paths: list,
    eval_valid: list,
    scores: np.ndarray,
    run_id: str,
):

    os.makedirs(score_txt_dir, exist_ok=True)
    feat_safe = _safe_name(feat_name)
    txt_path = os.path.join(score_txt_dir, f"scores_{feat_safe}_{fp_feat}_{run_id}.txt")

    seen = set()
    n_dup = 0
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        for glob_idx, sc in zip(eval_valid, scores):
            src = str(eval_paths[int(glob_idx)]).replace("\\", "/")
            utt_id = os.path.splitext(os.path.basename(src))[0]
            if utt_id in seen:
                n_dup += 1
            seen.add(utt_id)
            f.write(f"{utt_id} {float(sc):.10f}\n")

    print("[%s] Score file -> %s" % (feat_name, txt_path))
    print("[%s]   %d utterances, %d duplicate id(s)" % (feat_name, len(eval_valid), n_dup))
    if n_dup:
        print("[%s]   WARNING: a duplicate id is joined more than once by the scorers." % (feat_name,))
    return txt_path
