import os
import json
import hashlib
import random
import sys
import warnings
import time
from datetime import datetime
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import paths
from asvspoof_data import parse_2019_pa, parse_2021_pa_all_parts
from spectral_common import (
    _log_mag,
    _maybe_remake_cache,
    _safe_name,
    chunk_list,
    save_scores,
)

sys.stdout.reconfigure(line_buffering=True)


SAMPLE_RATE = 16000
HOP_SIZE    = 400

FINE_FFT  = 4096
FINE_WIN  = 1024
FINE_BINS = FINE_FFT // 2 + 1

PCA_VARIANCE   = 0.98
GMM_COMPONENTS = 512


PROTOCOL_2019_TRAIN = paths.PROTOCOL_2019_TRAIN
FLAC_2019_TRAIN     = paths.FLAC_2019_TRAIN
PROTOCOL_2019_DEV   = paths.PROTOCOL_2019_DEV
FLAC_2019_DEV       = paths.FLAC_2019_DEV
PROTOCOL_2021_PARTS = paths.PROTOCOL_2021_PARTS
FLAC_2021_PARTS     = paths.FLAC_2021_PARTS
OUTPUT_DIR          = paths.output_dir("run_fine_only_ablation.py")
SCORE_TXT_DIR       = paths.SCORE_TXT_DIR

MAX_TRAIN_BONA = None
MAX_EVAL       = None
REMAKE_CACHE_FINEONLY_TRAIN = True
REMAKE_CACHE_FINEONLY_EVAL  = False

RETRAIN_MODEL = False


def begin_run(output_dir: str, run_id: str):

    return run_id


def load_audio(path: str):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.squeeze()


def extract_fine_only(wav: torch.Tensor):
    """
    FineOnly = mean_t( log|X_fine| )
    Returns a (FINE_BINS,) utterance-level vector - the fine spectrogram
    fed straight into the framework, with no blurry branch or differencing.
    """
    log_fine = _log_mag(wav, FINE_FFT, HOP_SIZE, FINE_WIN)
    return log_fine.mean(dim=1).cpu().numpy()


def _extract_batch(feat_fn, paths: list, desc: str):

    feats, valid = [], []
    sample_times: list[float] = []
    last_err = None

    print('[%s] Starting extraction of %d files' % (desc, len(paths)))
    batch_t0 = time.perf_counter()

    for i, path in enumerate(tqdm(paths, desc=f"  {desc}", leave=False)):
        t_sample = time.perf_counter()
        try:
            wav  = load_audio(path)
            feat = feat_fn(wav)
            feats.append(feat)
            valid.append(i)
            sample_times.append(time.perf_counter() - t_sample)
        except Exception as e:
            last_err = e
            continue

    if len(feats) == 0:
        msg = f"No valid utterances for '{desc}'."
        if last_err is not None:
            msg += f" Last error: {type(last_err).__name__}: {last_err}"
        raise RuntimeError(msg)

    total_time = time.perf_counter() - batch_t0
    n_ok = len(sample_times)
    n_skip = len(paths) - n_ok
    avg_per_sample = float(np.mean(sample_times)) if sample_times else 0.0
    p50 = float(np.median(sample_times)) if sample_times else 0.0
    p95 = float(np.percentile(sample_times, 95)) if sample_times else 0.0

    print('[%s] Extraction done: %d ok / %d skipped | total: %.1f s | avg/sample: %.4f s | median: %.4f s | p95: %.4f s' % (desc, n_ok, n_skip, total_time, avg_per_sample, p50, p95))
    return np.stack(feats).astype(np.float32), valid, avg_per_sample


def load_or_compute(cache_path: str, feat_fn, paths: list, desc: str):
    if os.path.exists(cache_path):
        print('  [cache hit]   %s' % (cache_path,))
        with np.load(cache_path) as d:
            feats = d["feats"]
            valid = list(d["valid_idx"])
        if feats.std() > 1e-8:
            return feats, valid, 0
        print('WARNING:   Cache looks degenerated (std near 0) - recomputing: %s' % (cache_path,))
        os.remove(cache_path)
    feats, valid, avg_per_sample = _extract_batch(feat_fn, paths, desc)
    np.savez(cache_path, feats=feats, valid_idx=np.array(valid))
    print('  [cache saved] %s' % (cache_path,))
    return feats, valid, avg_per_sample


def load_or_compute_force(cache_path: str, feat_fn, paths: list, desc: str, force: bool):
    if force and os.path.exists(cache_path):
        os.remove(cache_path)
        print('  [cache] removed for force-recompute: %s' % (cache_path,))
    if force:
        feats, valid, avg_per_sample = _extract_batch(feat_fn, paths, desc)
        np.savez(cache_path, feats=feats, valid_idx=np.array(valid))
        print('  [cache saved] %s' % (cache_path,))
        return feats, valid, avg_per_sample
    return load_or_compute(cache_path, feat_fn, paths, desc)
def _config_fingerprint_for(feature_name: str | None):

    base = dict(sample_rate=SAMPLE_RATE, hop=HOP_SIZE)
    cfg = dict(**base, fine_fft=FINE_FFT, fine_win=FINE_WIN)
    raw = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def _fit_pca_gmm(train_feats, feat_name, model_path):

    if model_path and os.path.exists(model_path) and not RETRAIN_MODEL:
        print('[%s] Loading saved model from %s' % (feat_name, model_path))
        saved = joblib.load(model_path)
        pca, gmm = saved["pca"], saved["gmm"]
        n_dims = int(pca.n_components_)
        print('[%s] Model loaded (PCA %d dims, GMM %d comps) - skipping training' % (feat_name, n_dims, gmm.n_components))
        return pca, gmm, n_dims, 0.0, 0.0

    print('[%s] Fitting PCA (variance=%.2f) on %d train samples, dim=%d ...' % (feat_name, PCA_VARIANCE, train_feats.shape[0], train_feats.shape[1]))
    t_pca = time.perf_counter()
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    pca.fit(train_feats)
    n_dims = int(pca.n_components_)
    pca_time = time.perf_counter() - t_pca
    print('[%s] PCA done: %d -> %d dims  (%.2f s)' % (feat_name, train_feats.shape[1], n_dims, pca_time))

    train_r = pca.transform(train_feats)
    n_comp = max(1, min(GMM_COMPONENTS, len(train_r) // 10))
    print('[%s] Fitting GMM: %d components, %d train samples ...' % (feat_name, n_comp, len(train_r)))
    t_gmm = time.perf_counter()
    gmm = GaussianMixture(
        n_components=n_comp, covariance_type="diag",
        max_iter=300, reg_covar=2e-2, verbose=0,
    )
    gmm.fit(train_r.astype(np.float64))
    gmm_fit_time = time.perf_counter() - t_gmm
    print('[%s] GMM fit done (%.2f s)' % (feat_name, gmm_fit_time))

    if model_path:
        joblib.dump({"pca": pca, "gmm": gmm}, model_path)
        print('[%s] Model saved -> %s' % (feat_name, model_path))

    return pca, gmm, n_dims, pca_time, gmm_fit_time


def main():
    random.seed(43);  np.random.seed(43);  torch.manual_seed(43)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    begin_run(OUTPUT_DIR, run_id)

    print('=' * 70)
    print('SPECTRAL FINE-ONLY ABLATION - SEPARABILITY EVALUATION')
    print('(no blurry branch, no differencing - control for SpectralDetail-Hvr)')
    print('run_id: %s' % (run_id,))
    print('=' * 70)
    print('Fine STFT : FFT=%d, win=%d, hop=%d -> %d bins' % (FINE_FFT, FINE_WIN, HOP_SIZE, FINE_BINS))
    print('PCA variance threshold: %.2f  |  GMM max components: %d' % (PCA_VARIANCE, GMM_COMPONENTS))
    print('Output dir    : %s' % (OUTPUT_DIR,))
    print('Score txt dir : %s' % (SCORE_TXT_DIR,))

    fp_full = _config_fingerprint_for(None)
    EVAL_CHUNK_SIZE = 1000
    print('Config fingerprint (full): %s' % (fp_full,))
    print('Eval chunk size: %d' % (EVAL_CHUNK_SIZE,))

    features = [("FineOnly", extract_fine_only)]

    print('Loading protocols ...')
    t_proto = time.perf_counter()
    entries_2019 = (parse_2019_pa(PROTOCOL_2019_TRAIN, FLAC_2019_TRAIN) +
                    parse_2019_pa(PROTOCOL_2019_DEV,   FLAC_2019_DEV))
    print('2019 PA (train+dev): %d entries' % (len(entries_2019),))

    print('Loading ASVspoof 2021 PA eval (all parts 01-06) ...')
    entries_2021 = parse_2021_pa_all_parts(PROTOCOL_2021_PARTS, FLAC_2021_PARTS)
    print('2021 PA eval total : %d entries  (%.2f s)' % (len(entries_2021), time.perf_counter() - t_proto))

    bona_paths = [p for p, l in entries_2019 if l == 1]
    if MAX_TRAIN_BONA and len(bona_paths) > MAX_TRAIN_BONA:
        random.shuffle(bona_paths)
        bona_paths = bona_paths[:MAX_TRAIN_BONA]
        print('Capped train bonafide to %d' % (MAX_TRAIN_BONA,))

    if MAX_EVAL and len(entries_2021) > MAX_EVAL:
        random.shuffle(entries_2021)
        entries_2021 = entries_2021[:MAX_EVAL]
        print('Capped eval to %d' % (MAX_EVAL,))

    eval_paths  = [p for p, _ in entries_2021]
    eval_labels = np.array([l for _, l in entries_2021], dtype=np.int32)

    eval_path_chunks  = chunk_list(eval_paths, EVAL_CHUNK_SIZE)
    eval_label_chunks = chunk_list(eval_labels.tolist(), EVAL_CHUNK_SIZE)

    n_bona_eval  = int((eval_labels == 1).sum())
    n_spoof_eval = int((eval_labels == 0).sum())
    print('Train bonafide : %d' % (len(bona_paths),))
    print('Eval total     : %d  (%d bona / %d spoof)' % (len(eval_paths), n_bona_eval, n_spoof_eval))
    print('Eval chunks    : %d chunks of ~%d' % (len(eval_path_chunks), EVAL_CHUNK_SIZE))

    all_results       = {}
    cache_locations   = []
    all_timing        = {}

    for feat_name, feat_fn in features:
        print('Feature: %s' % (feat_name,))
        feat_t0 = time.perf_counter()

        feat_safe = _safe_name(feat_name)
        remake_train = bool(REMAKE_CACHE_FINEONLY_TRAIN)
        remake_eval  = bool(REMAKE_CACHE_FINEONLY_EVAL)

        _maybe_remake_cache(OUTPUT_DIR, feat_safe, remake_train, remake_eval)

        fp_feat = _config_fingerprint_for(feat_name)
        key_train = os.path.join(OUTPUT_DIR, f"train_{feat_safe}_{fp_feat}.npz")

        print('Extracting train features (%d bonafide) ...' % (len(bona_paths),))
        t_train_extract = time.perf_counter()
        train_feats, _, train_avg_per_sample = load_or_compute_force(
            key_train, feat_fn, bona_paths, f"{feat_name} train", force=remake_train)
        train_extract_time = time.perf_counter() - t_train_extract

        model_path = os.path.join(OUTPUT_DIR, f"model_{feat_safe}_{fp_feat}.pkl")
        cache_eval_glob = os.path.join(OUTPUT_DIR, f"eval_{feat_safe}_{fp_feat}_chunk*.npz")
        cache_locations.append((feat_name, key_train, cache_eval_glob, model_path))
        print()
        print('Feature cache for %s  (fingerprint %s)' % (feat_name, fp_feat))
        print('  train cache : %s' % (key_train,))
        print('  eval chunks : %s' % (cache_eval_glob,))
        print('  fitted model: %s' % (model_path,))
        print('  delete these three to force a fresh extraction')
        print()
        print('[%s] Model path: %s' % (feat_name, model_path))
        pca, gmm, n_dims, pca_time, gmm_fit_time = _fit_pca_gmm(
            train_feats, feat_name, model_path)
        feat_dim = int(train_feats.shape[1])

        print('Extracting eval features (%d utterances in %d chunks) ...' % (len(eval_paths), len(eval_path_chunks)))
        all_eval_scores        = []
        all_eval_valid_indices = []
        all_eval_valid_labels  = []

        n_bona = n_spoof = 0
        sum_bona,  sum_sq_bona  = np.zeros(feat_dim), np.zeros(feat_dim)
        sum_spoof, sum_sq_spoof = np.zeros(feat_dim), np.zeros(feat_dim)

        t_eval_extract = time.perf_counter()
        t_infer_total  = 0.0

        for chunk_idx, (eval_chunk_paths, eval_chunk_labels) in enumerate(
            zip(eval_path_chunks, eval_label_chunks), start=1
        ):
            chunk_desc = f"{feat_name} eval chunk {chunk_idx}/{len(eval_path_chunks)}"
            key_eval_chunk = os.path.join(
                OUTPUT_DIR,
                f"eval_{feat_safe}_{fp_feat}_chunk{chunk_idx:03d}.npz"
            )
            eval_feats_chunk, eval_valid_chunk, _ = load_or_compute_force(
                key_eval_chunk, feat_fn, eval_chunk_paths, chunk_desc, force=remake_eval)

            actual_start_idx  = (chunk_idx - 1) * EVAL_CHUNK_SIZE
            local_to_global   = [actual_start_idx + i for i in eval_valid_chunk]
            chunk_labels_arr  = np.array(
                [eval_chunk_labels[i] for i in eval_valid_chunk], dtype=np.int32)

            bm = chunk_labels_arr == 1
            sm = chunk_labels_arr == 0
            fb = eval_feats_chunk[bm].astype(np.float64)
            fs = eval_feats_chunk[sm].astype(np.float64)
            sum_bona    += fb.sum(0)
            sum_sq_bona += (fb ** 2).sum(0)
            sum_spoof    += fs.sum(0)
            sum_sq_spoof += (fs ** 2).sum(0)
            n_bona  += int(bm.sum())
            n_spoof += int(sm.sum())

            eval_r_chunk = pca.transform(eval_feats_chunk.astype(np.float32))
            del eval_feats_chunk

            t0 = time.perf_counter()
            scores_chunk = gmm.score_samples(eval_r_chunk)
            t_infer_total += time.perf_counter() - t0
            del eval_r_chunk

            all_eval_scores.append(scores_chunk)
            all_eval_valid_indices.extend(local_to_global)
            all_eval_valid_labels.extend(chunk_labels_arr.tolist())

        total_eval_extract_time = time.perf_counter() - t_eval_extract

        scores       = np.concatenate(all_eval_scores, axis=0)
        del all_eval_scores
        eval_valid   = all_eval_valid_indices
        valid_labels = np.array(all_eval_valid_labels, dtype=np.int32)
        n_eval_ok           = scores.shape[0]
        avg_eval_per_sample = total_eval_extract_time / max(1, n_eval_ok)
        avg_infer_per_sample = t_infer_total / max(1, n_eval_ok)

        print('[%s] Eval extraction complete: %d valid utterances from %d chunks' % (feat_name, n_eval_ok, len(eval_path_chunks)))

        mu_b = sum_bona  / n_bona
        mu_s = sum_spoof / n_spoof
        var_b = float(((sum_sq_bona  / n_bona)  - mu_b ** 2).mean())
        var_s = float(((sum_sq_spoof / n_spoof) - mu_s ** 2).mean())
        fdr            = float(np.sum((mu_b - mu_s) ** 2) / (var_b + var_s + 1e-10))
        hvr_norm_bona  = float(np.linalg.norm(mu_b))
        hvr_norm_spoof = float(np.linalg.norm(mu_s))

        bona_mask    = valid_labels == 1
        spoof_mask   = valid_labels == 0
        bona_scores  = scores[bona_mask]
        spoof_scores = scores[spoof_mask]

        res = dict(
            fdr=fdr, n_dims=n_dims,
            bona_mean=float(bona_scores.mean()), bona_std=float(bona_scores.std()),
            spoof_mean=float(spoof_scores.mean()), spoof_std=float(spoof_scores.std()),
            hvr_bona=hvr_norm_bona, hvr_spoof=hvr_norm_spoof,
            timing=dict(
                pca_fit_s=pca_time,
                gmm_fit_s=gmm_fit_time,
                gmm_infer_s=t_infer_total,
                gmm_infer_per_sample_s=avg_infer_per_sample,
            ),
            scores=scores.astype(np.float64),
        )
        feat_total_time = time.perf_counter() - feat_t0
        res["time_s"]   = feat_total_time
        res["feat_dim"] = feat_dim

        scores = res.pop("scores", None)
        if scores is not None:
            res["score_file"] = save_scores(
                SCORE_TXT_DIR, feat_name, fp_feat, eval_paths, eval_valid,
                scores, run_id,
            )

        for k, v in list(res.items()):
            if isinstance(v, np.generic):
                res[k] = v.item()

        all_results[feat_name] = res

        all_timing[feat_name] = dict(
            train_extract_s=train_extract_time,
            train_extract_per_sample_s=train_avg_per_sample,
            eval_extract_s=total_eval_extract_time,
            eval_extract_per_sample_s=avg_eval_per_sample,
            pca_fit_s=res["timing"]["pca_fit_s"],
            gmm_fit_s=res["timing"]["gmm_fit_s"],
            gmm_infer_s=res["timing"]["gmm_infer_s"],
            gmm_infer_per_sample_s=res["timing"]["gmm_infer_per_sample_s"],
            total_s=feat_total_time,
        )

        print('[%s] Feature total: %.1f s' % (feat_name, feat_total_time))


    results_for_json = {}
    for name, r in all_results.items():
        r_copy = {k: v for k, v in r.items() if k != "scores"}
        results_for_json[name] = r_copy

    out_path = os.path.join(OUTPUT_DIR, f"comparison_results_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(dict(run_id=run_id, results=results_for_json, timing=all_timing), f, indent=2)
    print('Results saved -> %s' % (out_path,))

    latest_path = os.path.join(OUTPUT_DIR, "comparison_results_latest.json")
    with open(latest_path, "w") as f:
        json.dump(dict(run_id=run_id, results=results_for_json, timing=all_timing), f, indent=2)
    print('Latest results -> %s' % (latest_path,))

    print()
    print('Run complete. run_id=%s' % (run_id,))
    if cache_locations:
        print()
        print('Feature cache - delete these to force a fresh extraction')
        for _feat, _tr, _ev, _mp in cache_locations:
            print('  %s' % (_feat,))
            print('    %s' % (_tr,))
            print('    %s' % (_ev,))
            print('    %s' % (_mp,))

    print()
    print('Where everything went')
    print('  results json : %s' % (out_path,))
    print('  latest copy  : %s' % (latest_path,))
    for _name, _r in all_results.items():
        if _r.get("score_file"):
            print('  score file   : %s' % (_r["score_file"],))
    print()
    print('Score it with')
    for _name, _r in all_results.items():
        if _r.get("score_file"):
            print('  python score_pooled_eer.py "%s"' % (_r["score_file"],))
            print('  python score_pooled_min_tdcf.py "%s"' % (_r["score_file"],))


if __name__ == "__main__":
    main()
