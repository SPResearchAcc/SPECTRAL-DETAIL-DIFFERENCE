import gc
import io
import json
import multiprocessing
import os
import random
import signal
import sqlite3
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import paths
sys.stdout.reconfigure(line_buffering=True)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

try:
    import pyworld as pw
    HAS_PYWORLD = True
except ImportError:
    HAS_PYWORLD = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
def _set_low_priority():
    
    if HAS_PSUTIL:
        try:
            p = psutil.Process()
            if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                p.nice(10)
            print('Process priority set to below-normal.')
        except Exception as exc:
            print('WARNING: Could not set process priority: %s' % (exc,))

def _atomic_save_npy(path: str, arr):
    
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)
def _atomic_save_json(path: str, obj):
    
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

@contextmanager
def _phase(name: str):
    
    print('--------------------------- %s' % (name,))
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print('--------------------------- %s done  %.1f s' % (name, elapsed))

_STOP_REQUESTED = False

def _request_stop(signum, _frame):
    global _STOP_REQUESTED
    if not _STOP_REQUESTED:
        print('WARNING: Stop signal received (%s). Finishing the current batch, then rerun the script to resume.' % (signum,))
    _STOP_REQUESTED = True

def _stop_requested():
    return _STOP_REQUESTED

def _install_signal_handlers():
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_stop)

class CheckpointStore:
    """
    "what has already been computed".

    Tables -->
    
    ul_features(stage, path)        one row per extracted file. Deleted once a stages results have been safely flushed 
    scores(subset, sid, utt_id)     one row per (system, utterance) score never deleted. this is the permanent queryable record of every score
    progress(key)                   simple key/value flags for coarse stage completion

    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout=60000;")
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ul_features (
                stage   TEXT NOT NULL,
                path    TEXT NOT NULL,
                ok      INTEGER NOT NULL,
                payload BLOB,
                PRIMARY KEY (stage, path)
            )""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                subset  TEXT NOT NULL,
                sid     TEXT NOT NULL,
                utt_id  TEXT NOT NULL,
                label   INTEGER NOT NULL,
                score   REAL NOT NULL,
                PRIMARY KEY (subset, sid, utt_id)
            )""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                key        TEXT PRIMARY KEY,
                value      TEXT,
                updated_at TEXT
            )""")
        self.conn.commit()

    def get_progress(self, key: str):
        row = self.conn.execute(
            "SELECT value FROM progress WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_progress(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO progress(key, value, updated_at) "
            "VALUES (?, ?, datetime('now'))", (key, value))
        self.conn.commit()

    def get_done_paths(self, stage: str):
        rows = self.conn.execute(
            "SELECT path FROM ul_features WHERE stage=?", (stage,))
        return {r[0] for r in rows}
    def add_ul_features_batch(self, rows: list):

        if not rows:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO ul_features(stage, path, ok, payload) "
            "VALUES (?, ?, ?, ?)", rows)
        self.conn.commit()

    def get_ul_features(self, stage: str, paths: list):

        result = {}
        CHUNK = 500
        for i in range(0, len(paths), CHUNK):
            batch = paths[i:i + CHUNK]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn.execute(
                f"SELECT path, ok, payload FROM ul_features "
                f"WHERE stage=? AND path IN ({placeholders})",
                (stage, *batch))
            for path, ok, payload in rows:
                result[path] = (ok, payload)
        return result

    def delete_stage(self, stage: str):

        self.conn.execute("DELETE FROM ul_features WHERE stage=?", (stage,))
        self.conn.commit()
    def get_scores_ordered(self, sid: str):
   
        return self.conn.execute(
            "SELECT utt_id, score FROM scores WHERE sid=? ORDER BY subset, utt_id",
            (sid,)).fetchall()

    def get_scores_for_sid(self, subset: str, sid: str):
        rows = self.conn.execute(
            "SELECT utt_id, score FROM scores WHERE subset=? AND sid=?", (subset, sid))
        return {r[0]: r[1] for r in rows}

    def add_score_rows(self, rows: list):

        if not rows:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO scores(subset, sid, utt_id, label, score) "
            "VALUES (?, ?, ?, ?, ?)", rows)
        self.conn.commit()

    def close(self):
        self.conn.execute("PRAGMA wal_checkpoint(FULL);")
        self.conn.close()


ACTIVE_SYSTEMS: list[int] = [3]

SEEDS: list[int] = [43, 44, 45]

_CURRENT_SEED: int = SEEDS[0]


SAMPLE_RATE         = 16000
FFT_SIZE            = 1024
HOP_SIZE            = int(SAMPLE_RATE * 0.025)
WIN_SIZE            = int(SAMPLE_RATE * 0.050)
N_BINS              = FFT_SIZE // 2 + 1
PCA_VARIANCE        = 0.98
GMM_COMPONENTS      = 512
MAX_LEN             = None
NUM_WORKERS         = 4
PROTOCOL_2019_TRAIN = paths.PROTOCOL_2019_TRAIN
FLAC_2019_TRAIN     = paths.FLAC_2019_TRAIN
PROTOCOL_2019_DEV   = paths.PROTOCOL_2019_DEV
FLAC_2019_DEV       = paths.FLAC_2019_DEV
SUBSETS_2021 = list(paths.PARTS_2021)

def _2021_protocol(subset: str):
    return paths.PROTOCOL_2021_BY_PART[subset]

def _2021_flac(subset: str):
    return paths.FLAC_2021_BY_PART[subset]

OUTPUT_DIR    = paths.output_dir("run_world_baseline_multiseed.py")
SCORE_TXT_DIR = paths.SCORE_TXT_DIR

MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES  = None

DB_PATH            = os.path.join(OUTPUT_DIR, "checkpoint.sqlite3")

BATCH_COMMIT        = 25
COMMIT_INTERVAL_S   = 15


@dataclass
class SystemDef:
    id:         int
    use_pca:    bool
    label:      str

SYSTEM_REGISTRY: dict[int, SystemDef] = {
    3:  SystemDef(3,  True,  "sys3  WORLD PCA+UL-GMM  --"),
}

_UNSUPPORTED = {1, 2}

def _pad_or_trim_1d(wav: torch.Tensor, target_len: int):
    if wav.shape[0] < target_len:
        return F.pad(wav, (0, target_len - wav.shape[0]))
    return wav[:target_len]

def load_audio(path: str, *, target_len: int | None = MAX_LEN):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.squeeze()
    if target_len is not None:
        wav = _pad_or_trim_1d(wav, int(target_len))
    return wav

def world_vocode(wav: torch.Tensor):
    wav_np = wav.cpu().numpy().astype(np.float32)
    if HAS_PYWORLD:
        wav64 = wav_np.astype(np.float64)
        _min_samples = int(SAMPLE_RATE * 0.1)
        _rms = np.sqrt(np.mean(wav64 ** 2))
        if (len(wav64) < _min_samples
                or _rms < 1e-6
                or not np.isfinite(wav64).all()):
            return wav
        f0, sp, ap = pw.wav2world(wav64, SAMPLE_RATE)
        out = pw.synthesize(f0, sp, ap, SAMPLE_RATE).astype(np.float32)
    else:
        pre = torchaudio.functional.preemphasis(wav, coeff=0.97)
        out = torchaudio.functional.deemphasis(pre, coeff=0.97).numpy()

    try:
        from scipy.signal import correlate
        corr = correlate(wav_np, out, mode="full", method="fft")
    except Exception:
        corr = np.correlate(wav_np, out, mode="full")
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

def _log_spectrogram(wav: torch.Tensor):
    window = torch.hann_window(WIN_SIZE, device=wav.device)
    stft   = torch.stft(wav, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
                        win_length=WIN_SIZE, window=window, return_complex=True)
    return torch.log(stft.abs().clamp(min=1e-10))

def _hvr_frames_np(wav: torch.Tensor):
    
    log_X  = _log_spectrogram(wav)
    log_Xv = _log_spectrogram(world_vocode(wav))
    T      = min(log_X.shape[1], log_Xv.shape[1])
    return (log_Xv[:, :T] - log_X[:, :T]).T.contiguous().numpy()

def parse_2019_pa(protocol_file: str, flac_dir: str):
    entries = []
    with open(protocol_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            path = os.path.join(flac_dir, parts[1] + ".flac")
            if os.path.exists(path):
                entries.append((path, 1 if parts[-1] == "bonafide" else 0))
    return entries

def parse_2021_pa(protocol_file: str, flac_dir: str):
    entries = []
    with open(protocol_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            path = os.path.join(flac_dir, parts[1] + ".flac")
            if os.path.exists(path):
                entries.append((path, 1 if parts[9] == "bonafide" else 0))
    return entries

class GMMOneClass:
    
    def __init__(self, n_components: int = GMM_COMPONENTS,
                 pca_variance: float = PCA_VARIANCE,
                 use_pca: bool = True):
        self.n_components    = n_components
        self.pca_var         = pca_variance
        self.use_pca         = use_pca
        self.pca             = None
        self.gmm             = None

    def _reduce(self, feats: np.ndarray, fit: bool = False):
        if not self.use_pca:
            return feats
        if fit:
            self.pca = PCA(n_components=self.pca_var, svd_solver="full")
            return self.pca.fit_transform(feats)
        return self.pca.transform(feats)

    def fit(self, feats: np.ndarray):
        prefix = f"    [{'PCA+' if self.use_pca else ''}GMM]"
        print('%s input=%dD  n=%d' % (prefix, feats.shape[1], len(feats)))
        reduced = self._reduce(feats, fit=True)
        if self.use_pca:
            print('%s PCA -> %dD  (explained var >= %.2f)' % (prefix, reduced.shape[1], self.pca_var))

        fit_data = reduced

        n_comp = max(1, min(self.n_components, len(fit_data) // 10))
        print('%s GMM fitting %d components on %d samples' % (prefix, n_comp, len(fit_data)))
        self.gmm = GaussianMixture(n_components=n_comp, covariance_type="diag",
                                   max_iter=200, reg_covar=1e-4, verbose=0)
        self.gmm.fit(fit_data.astype(np.float64))
        print('%s GMM done  converged=%s  lower_bound=%.4f' % (prefix, self.gmm.converged_, self.gmm.lower_bound_))

    def score_batch(self, feats: np.ndarray):
        return self.gmm.score_samples(self._reduce(feats))

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"pca": self.pca, "gmm": self.gmm,
                         "use_pca": self.use_pca}, f)
        print('    Saved -> %s' % (path,))

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.pca     = d["pca"]
        self.gmm     = d["gmm"]
        self.use_pca = d["use_pca"]
        print('    Loaded ← %s' % (path,))
def _worker_train(path: str):
    
    try:
        wav = load_audio(path)
        fr = _hvr_frames_np(wav)
        save_kwargs = {"ul": np.stack([fr.mean(axis=0)])}
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        np.savez(tmp.name, **save_kwargs)
        tmp.close()
        return tmp.name
    except Exception:
        return None

def _worker_eval(path: str):
    
    try:
        wav = load_audio(path)
        fr  = _hvr_frames_np(wav)
        return fr.mean(axis=0)
    except Exception:
        return None

class SingleSystem:
    
    def __init__(self, sdef: SystemDef, out_dir: str, seed: int):
        self.sdef    = sdef
        self.out_dir = out_dir
        self.seed    = seed
        self.backend = self._make_backend()

    def _make_backend(self):
        sdef = self.sdef
        return GMMOneClass(use_pca=sdef.use_pca)

    @property
    def pkl_path(self):
        return os.path.join(self.out_dir, f"sys{self.sdef.id}_seed{self.seed}_backend.pkl")

    def fit(self, ul_feats: np.ndarray):
        
        print('\n  [%s] Fitting...' % (self.sdef.label,))
        self.backend.fit(ul_feats)
        self.backend.save(self.pkl_path)

    def score(self, ul_feats: np.ndarray):
        return self.backend.score_batch(ul_feats).astype(np.float64)

    def load(self):
        self.backend.load(self.pkl_path)

class PASystem:
    
    def __init__(self, seed: int,
                 system_ids: list[int] = None,
                 out_dir: str = OUTPUT_DIR):
        self.seed       = seed
        self.system_ids = system_ids  if system_ids  is not None else ACTIVE_SYSTEMS
        self.out_dir    = out_dir
        os.makedirs(out_dir, exist_ok=True)

        bad = [sid for sid in self.system_ids if sid in _UNSUPPORTED]
        if bad:
            print('WARNING: Systems %s require external vocoders (not implemented) - skipping.' % (bad,))
            self.system_ids = [s for s in self.system_ids if s not in _UNSUPPORTED]
        unknown = [s for s in self.system_ids if s not in SYSTEM_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown system IDs: {unknown}. "
                             f"Valid: {sorted(SYSTEM_REGISTRY)}")

        self.systems = {sid: SingleSystem(SYSTEM_REGISTRY[sid], out_dir, seed)
                        for sid in self.system_ids}

    def _score_key(self, sid: int):

        return f"{sid}s{self.seed}"

    def train_resumable(self, bona_entries: list, ckpt: "CheckpointStore"):
        
        paths = [p for p, _ in bona_entries]
        sids  = list(self.systems.keys())

        cache_tag = f"sp0_pyr0_n{len(paths)}"
        cache_ul  = os.path.join(self.out_dir, f"train_ul_{cache_tag}.npy")
        print('  train feature cache: %s' % (cache_ul,))
        stage     = "train_sp0_pyr0"

        if os.path.exists(cache_ul):
            print('  Loading cached features from %s' % (cache_ul,))
            ul_feats = np.load(cache_ul)
            print('  UL samples: %d' % (len(ul_feats),))
        else:
            done_paths = ckpt.get_done_paths(stage)
            remaining  = [p for p in paths if p not in done_paths]
            print('  [%s] checkpoint: %d/%d already extracted, %d remaining' % (stage, len(paths) - len(remaining), len(paths), len(remaining)))

            if remaining:
                buffer, last_commit, processed = [], time.time(), 0
                with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                    for path, tmp_path in zip(remaining, tqdm(
                            pool.imap(_worker_train, remaining, chunksize=1),

                            total=len(remaining), desc="  Extracting",
                            smoothing=0)):
                        if tmp_path is None:
                            buffer.append((stage, path, 0, b""))
                        else:

                            with open(tmp_path, "rb") as fh:
                                payload = fh.read()
                            os.unlink(tmp_path)
                            buffer.append((stage, path, 1, payload))
                        processed += 1

                        if (len(buffer) >= BATCH_COMMIT
                                or (time.time() - last_commit) > COMMIT_INTERVAL_S):
                            ckpt.add_ul_features_batch(buffer)

                            buffer.clear()
                            last_commit = time.time()

                        if _stop_requested():
                            break
                if buffer:
                    ckpt.add_ul_features_batch(buffer)
                    buffer.clear()

            done_paths = ckpt.get_done_paths(stage)
            if len(done_paths) < len(paths):
                print('WARNING:   [%s] extraction incomplete (%d/%d) - stopping for a clean resume.' % (stage, len(done_paths), len(paths)))
                return False

            row_map = ckpt.get_ul_features(stage, paths)
            ul_list: list[np.ndarray] = []
            for p in paths:
                ok, payload = row_map[p]
                if not ok:
                    continue
                with np.load(io.BytesIO(payload)) as data:
                    ul_list.append(data["ul"])
            ul_feats = (np.concatenate(ul_list).astype(np.float32)
                        if ul_list else np.zeros((0, N_BINS), dtype=np.float32))
            del ul_list
            gc.collect()

            print('  UL samples: %d' % (len(ul_feats),))
            print('  Caching features -> %s' % (cache_ul,))
            _atomic_save_npy(cache_ul, ul_feats)
            ckpt.delete_stage(stage)

        for sid in sids:
            sys_obj = self.systems[sid]
            progress_key = f"train_fit_sys{sid}_seed{self.seed}"
            if (ckpt.get_progress(progress_key) == "done"
                    and os.path.exists(sys_obj.pkl_path)):
                print('  [sys%d seed%d] already fitted - loading cached backend.' % (sid, self.seed))
                sys_obj.load()
            else:
                sys_obj.fit(ul_feats)
                ckpt.set_progress(progress_key, "done")
            if _stop_requested():
                del ul_feats
                gc.collect()
                return False

        del ul_feats
        gc.collect()

        if _stop_requested():
            return False

        return True

    def _extract_ul_features_resumable(
            self, paths: list[str], stage: str, cache_ul: str, cache_mask: str,
            ckpt: "CheckpointStore"):
        
        if os.path.exists(cache_ul) and os.path.exists(cache_mask):
            print('  Loading cached UL features from %s' % (cache_ul,))
            return np.load(cache_ul), np.load(cache_mask), True

        done_paths = ckpt.get_done_paths(stage)
        remaining  = [p for p in paths if p not in done_paths]
        print('  [%s] checkpoint: %d/%d already extracted, %d remaining' % (stage, len(paths) - len(remaining), len(paths), len(remaining)))

        if remaining:
            buffer, last_commit, processed = [], time.time(), 0
            with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                for path, r in zip(remaining, tqdm(
                        pool.imap(_worker_eval, remaining, chunksize=1),
                        total=len(remaining), desc=f"  {stage} UL features",
                        smoothing=0)):
                    if r is None:
                        buffer.append((stage, path, 0, b""))
                    else:
                        buffer.append((stage, path, 1, r.astype(np.float32).tobytes()))
                    processed += 1

                    if (len(buffer) >= BATCH_COMMIT
                            or (time.time() - last_commit) > COMMIT_INTERVAL_S):
                        ckpt.add_ul_features_batch(buffer)
                        buffer.clear()
                        last_commit = time.time()


                    if _stop_requested():
                        break
            if buffer:
                ckpt.add_ul_features_batch(buffer)
                buffer.clear()

        done_paths = ckpt.get_done_paths(stage)
        if len(done_paths) < len(paths):
            print('WARNING:   [%s] extraction incomplete (%d/%d) - stopping for a clean resume.' % (stage, len(done_paths), len(paths)))
            return None, None, False

        row_map = ckpt.get_ul_features(stage, paths)
        ul_feats = np.zeros((len(paths), N_BINS), dtype=np.float32)
        failed_mask = np.zeros(len(paths), dtype=bool)
        for i, p in enumerate(paths):
            ok, payload = row_map[p]
            if ok:
                ul_feats[i] = np.frombuffer(payload, dtype=np.float32)
            else:
                failed_mask[i] = True

        print('  Failed extractions: %d / %d' % (int(failed_mask.sum()), len(paths)))
        _atomic_save_npy(cache_ul, ul_feats)
        _atomic_save_npy(cache_mask, failed_mask)
        ckpt.delete_stage(stage)
        return ul_feats, failed_mask, True

    def evaluate_resumable(self, entries: list, *, subset_tag: str,
                            ckpt: "CheckpointStore"):
        

        labels  = np.array([lab for _, lab in entries], dtype=np.int32)
        paths   = [p for p, _ in entries]
        utt_ids = np.array([os.path.splitext(os.path.basename(p))[0]
                            for p in paths], dtype=object)

        cache_ul   = os.path.join(self.out_dir, f"eval_ul_{subset_tag}_n{len(paths)}.npy")
        cache_mask = os.path.join(self.out_dir, f"eval_mask_{subset_tag}_n{len(paths)}.npy")
        print('  eval feature cache : %s' % (cache_ul,))
        print('  eval failed mask   : %s' % (cache_mask,))

        ul_feats, failed_mask, ok = self._extract_ul_features_resumable(
            paths, stage=f"eval_{subset_tag}", cache_ul=cache_ul,
            cache_mask=cache_mask, ckpt=ckpt)
        if not ok:
            return None

        all_scores: dict = {}
        utt_id_set = set(str(u) for u in utt_ids)

        for sid in self.systems:
            existing = ckpt.get_scores_for_sid(subset_tag, self._score_key(sid))
            if set(existing.keys()) == utt_id_set:
                all_scores[sid] = np.array([existing[str(u)] for u in utt_ids],
                                           dtype=np.float64)
                print('  sys%d seed%d UL scores loaded from checkpoint (%d utterances)' % (sid, self.seed, len(paths)))
            else:
                raw = self.systems[sid].score(ul_feats)
                raw[failed_mask] = 0.0
                rows = [(subset_tag, self._score_key(sid), str(u), int(l), float(s))
                        for u, l, s in zip(utt_ids, labels, raw)]
                ckpt.add_score_rows(rows)
                all_scores[sid] = raw
            if _stop_requested():
                return None

        del ul_feats
        gc.collect()

        all_scores["labels"]  = labels
        all_scores["utt_ids"] = utt_ids
        return all_scores
def _run_pipeline(run_ts: str, ckpt: "CheckpointStore", seed: int):
    print('=' * 60)
    print('DKU-CMRI PA System - ASVspoof 2021  [CHECKPOINTED / RESUMABLE / MULTI-SEED]')
    print('Run started at  : %s' % (run_ts,))
    print('Seed            : %d' % (seed,))
    print('pyworld (WORLD) : %s' % (HAS_PYWORLD,))
    print('Active systems  : %s' % (ACTIVE_SYSTEMS,))
    print('NUM_WORKERS     : %d' % (NUM_WORKERS,))
    print('psutil available: %s  (process priority)' % (HAS_PSUTIL,))
    print('GMM_COMPONENTS  : %d' % (GMM_COMPONENTS,))
    print('PCA_VARIANCE    : %.2f' % (PCA_VARIANCE,))
    print('OUTPUT_DIR      : %s' % (OUTPUT_DIR,))
    print('SCORE_TXT_DIR   : %s' % (SCORE_TXT_DIR,))
    print('Checkpoint DB   : %s' % (DB_PATH,))
    print('BATCH_COMMIT    : %d files / %ds' % (BATCH_COMMIT, COMMIT_INTERVAL_S))
    print('=' * 60)

    with _phase("Parse 2019 protocols"):
        entries_train = parse_2019_pa(PROTOCOL_2019_TRAIN, FLAC_2019_TRAIN)
        entries_dev   = parse_2019_pa(PROTOCOL_2019_DEV,   FLAC_2019_DEV)
        entries_2019  = entries_train + entries_dev

    print('Train entries (all): %d  (train=%d  dev=%d)' % (len(entries_2019), len(entries_train), len(entries_dev)))

    if MAX_TRAIN_SAMPLES:
        entries_2019 = entries_2019[:MAX_TRAIN_SAMPLES]
        print('MAX_TRAIN_SAMPLES cap: %d' % (MAX_TRAIN_SAMPLES,))

    bona_train  = [(p, l) for p, l in entries_2019 if l == 1]
    spoof_train = [(p, l) for p, l in entries_2019 if l == 0]
    print('Bona fide train : %d' % (len(bona_train),))
    print('Spoof train     : %d  (not used for fitting)' % (len(spoof_train),))

    system = PASystem(seed=seed, out_dir=OUTPUT_DIR)

    with _phase("Training"):
        train_complete = system.train_resumable(bona_train, ckpt)
    if not train_complete or _stop_requested():
        print('WARNING: Stopped during training. Rerun the script to resume - already-extracted files and already-fitted systems will be skipped automatically.')
        return

    per_sub: dict[str, dict] = {}
    subset_stats: list[dict] = []

    for subset in SUBSETS_2021:
        proto = _2021_protocol(subset)
        flac  = _2021_flac(subset)
        if not os.path.exists(proto):
            print('WARNING: Subset %s not found, skipping: %s' % (subset, proto))
            continue
        sub_entries = parse_2021_pa(proto, flac)
        if not sub_entries:
            print('WARNING: Subset %s: no valid entries found' % (subset,))
            continue
        if MAX_EVAL_SAMPLES:
            sub_entries = sub_entries[:MAX_EVAL_SAMPLES]

        n_bona  = sum(1 for _, l in sub_entries if l == 1)
        n_spoof = sum(1 for _, l in sub_entries if l == 0)
        print('Subset %s: %d entries  (%d bona / %d spoof)' % (subset, len(sub_entries), n_bona, n_spoof))
        subset_stats.append({"subset": subset, "total": len(sub_entries),
                              "n_bona": n_bona, "n_spoof": n_spoof})

        with _phase(f"Eval subset {subset}"):
            result = system.evaluate_resumable(
                sub_entries, subset_tag=f"sub{subset}", ckpt=ckpt)

        if result is None:
            print('WARNING: Stopped mid-subset %s. Progress saved - rerun the script to resume.' % (subset,))
            return

        per_sub[subset] = result
        ckpt.set_progress(f"eval_subset_{subset}_done_seed{seed}", "yes")

        gc.collect()

        if _stop_requested():
            print('WARNING: Stop requested after finishing subset %s. Rerun to continue with the remaining subsets.' % (subset,))
            return

    if not per_sub:
        print('ERROR: No eval subsets found.')
        return

    valid = sorted(per_sub)
    agg_labels  = np.concatenate([per_sub[s]["labels"]  for s in valid])
    print('Eval entries    : %d  (%d bona / %d spoof)' % (len(agg_labels), int((agg_labels == 1).sum()), int((agg_labels == 0).sum())))


    results_json: dict = {
        "run_timestamp": run_ts,
        "seed": seed,
        "active_systems": system.system_ids,
        "config": {
            "SAMPLE_RATE": SAMPLE_RATE,
            "FFT_SIZE": FFT_SIZE,
            "HOP_SIZE": HOP_SIZE,
            "WIN_SIZE": WIN_SIZE,
            "N_BINS": N_BINS,
            "PCA_VARIANCE": PCA_VARIANCE,
            "GMM_COMPONENTS": GMM_COMPONENTS,
            "NUM_WORKERS": NUM_WORKERS,
        },
        "dataset": {
            "n_bona_train": len(bona_train),
            "subsets": subset_stats,
            "total_eval": int(len(agg_labels)),
            "total_eval_bona": int((agg_labels == 1).sum()),
            "total_eval_spoof": int((agg_labels == 0).sum()),
        },
        "systems": {},
    }

    for sid in system.system_ids:
        results_json["systems"][str(sid)] = {"label": system.systems[sid].sdef.label}

    os.makedirs(SCORE_TXT_DIR, exist_ok=True)
    score_txt_paths = []
    for sid in system.system_ids:
        rows = ckpt.get_scores_ordered(f"{sid}s{seed}")
        txt_path = os.path.join(SCORE_TXT_DIR, f"scores_WorldBaseline_sys{sid}_seed{seed}.txt")
        with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
            for utt_id, score in rows:
                f.write(f"{utt_id} {score}\n")
        score_txt_paths.append(txt_path)
        print('Score file -> %s' % (txt_path,))
        print('  %d utterances' % (len(rows),))

    json_path        = os.path.join(OUTPUT_DIR, f"results_{run_ts}_seed{seed}.json")
    json_path_latest = os.path.join(OUTPUT_DIR, f"results_latest_seed{seed}.json")
    _atomic_save_json(json_path, results_json)
    _atomic_save_json(json_path_latest, results_json)
    print('Results JSON -> %s  (and results_latest_seed%d.json)' % (json_path, seed))
    print()
    print('Where everything went  (seed %d)' % (seed,))
    print('  results json : %s' % (json_path,))
    print('  latest copy  : %s' % (json_path_latest,))
    for _p in score_txt_paths:
        print('  score file   : %s' % (_p,))
    print('  checkpoint db: %s' % (DB_PATH,))
    print()
    print('Score it with')
    for _p in score_txt_paths:
        print('  python score_pooled_eer.py "%s"' % (_p,))
        print('  python score_pooled_min_tdcf.py "%s"' % (_p,))

    ckpt.set_progress(f"pipeline_complete_seed{seed}", "yes")
    return score_txt_paths

def main():
    global _CURRENT_SEED

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _install_signal_handlers()
    _set_low_priority()

    print('MULTI-SEED RUN - seeds: %s' % (SEEDS,))

    score_txt_by_seed: dict = {}

    ckpt = CheckpointStore(DB_PATH)
    try:
        for seed in SEEDS:
            if ckpt.get_progress(f"pipeline_complete_seed{seed}") == "yes":
                print('Seed %d already fully completed - skipping.' % (seed,))
                done = [os.path.join(SCORE_TXT_DIR, f"scores_WorldBaseline_sys{sid}_seed{seed}.txt")
                        for sid in ACTIVE_SYSTEMS]
                done = [p for p in done if os.path.isfile(p)]
                if done:
                    score_txt_by_seed[seed] = done
                continue

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            _CURRENT_SEED = seed

            run_ts = time.strftime("%Y%m%d_%H%M%S")
            try:
                produced = _run_pipeline(run_ts, ckpt, seed)
                if produced:
                    score_txt_by_seed[seed] = produced
            except KeyboardInterrupt:
                print('WARNING: Interrupted (KeyboardInterrupt) during seed %d. Progress saved - rerun the script to resume this seed and continue with any remaining seeds.' % (seed,))
                raise
            if _stop_requested():
                print('WARNING: Stop requested - halting the multi-seed loop after seed %d. Rerun the script to resume.' % (seed,))
                break
    except KeyboardInterrupt:
        pass
    finally:
        ckpt.close()
        print('Checkpoint DB closed cleanly.')

    print()
    print('Feature cache - delete these to force a fresh extraction')
    print('  dir            : %s' % (OUTPUT_DIR,))
    print('  train features : %s' % (os.path.join(OUTPUT_DIR, 'train_ul_*.npy'),))
    print('  eval features  : %s' % (os.path.join(OUTPUT_DIR, 'eval_ul_*.npy'),))
    print('  eval masks     : %s' % (os.path.join(OUTPUT_DIR, 'eval_mask_*.npy'),))
    print('  fitted backends: %s' % (os.path.join(OUTPUT_DIR, 'sys*_seed*_backend.pkl'),))
    print('  checkpoint db  : %s' % (DB_PATH,))
    print('  the db also records which seeds finished - delete it to redo every seed')

    if score_txt_by_seed:
        print()
        print('=' * 74)
        print('SCORE EVERY SEED - every command for this run, collected here')
        print('=' * 74)
        for seed in SEEDS:
            for path in score_txt_by_seed.get(seed, []):
                print('  seed %d' % (seed,))
                print('    python score_pooled_eer.py      "%s"' % (path,))
                print('    python score_pooled_min_tdcf.py "%s"' % (path,))
        print('=' * 74)

if __name__ == "__main__":
    main()
