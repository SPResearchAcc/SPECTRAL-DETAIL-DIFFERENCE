import glob
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import random
import sklearn
import torch
import torchaudio


sys.stdout.reconfigure(line_buffering=True)


@dataclass
class RunnerConfig:

    target_script_path:               str
    target_module_name:               str
    seeds:                            list
    force_retrain_per_seed:           bool
    reuse_feature_cache_across_seeds: bool
    remake_cache_flags:               tuple
    manifest_config_keys:             list
    banner:                           str
    summary_heading:                  str


def _load_target_module(cfg):
    spec = importlib.util.spec_from_file_location(cfg.target_module_name, cfg.target_script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_current_seed  = {"value": None}
_captured      = {"run_id": None}

_orig_random_seed      = random.seed
_orig_np_random_seed   = np.random.seed
_orig_torch_manual_seed = torch.manual_seed


def _seeded_random_seed(*_a, **_k):
    return _orig_random_seed(_current_seed["value"])

def _seeded_np_seed(*_a, **_k):
    return _orig_np_random_seed(_current_seed["value"])

def _seeded_torch_seed(*_a, **_k):
    return _orig_torch_manual_seed(_current_seed["value"])

def _install_seed_override():
    random.seed        = _seeded_random_seed
    np.random.seed     = _seeded_np_seed
    torch.manual_seed  = _seeded_torch_seed

def _install_run_id_capture(target):
    orig_begin_run = target.begin_run

    def wrapped_begin_run(output_dir, run_id):
        _captured["run_id"] = run_id
        return orig_begin_run(output_dir, run_id)

    target.begin_run = wrapped_begin_run

def _apply_efficiency_overrides(target, cfg):
    if cfg.reuse_feature_cache_across_seeds:
        for _flag in cfg.remake_cache_flags:
            setattr(target, _flag, False)
        print("Feature-cache reuse across seeds: ON (target's own REMAKE_CACHE_* flags overridden to False)")
    if cfg.force_retrain_per_seed:
        target.RETRAIN_MODEL = True
        print('Per-seed GMM refit: forced ON (RETRAIN_MODEL=True for every seed)')

def _paths(target):
    fp   = target._config_fingerprint_for(None)
    root = os.path.join(target.OUTPUT_DIR, "multiseed", fp)
    return dict(
        root       = root,
        logs       = os.path.join(root, "logs"),
        models     = os.path.join(root, "models_by_seed"),
        state      = os.path.join(root, "progress_state.json"),
        manifest   = os.path.join(root, "run_manifest.json"),
        summary_js = os.path.join(root, "seed_summary.json"),
        summary_tx = os.path.join(root, "seed_summary.txt"),
    )

def _atomic_write_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def _capture_manifest(target, cfg):
    cfg_keys = cfg.manifest_config_keys
    target_config_snapshot = {k: getattr(target, k) for k in cfg_keys if hasattr(target, k)}

    return dict(
        generated_at=datetime.now().isoformat(),
        seeds=cfg.seeds,
        force_retrain_per_seed=cfg.force_retrain_per_seed,
        reuse_feature_cache_across_seeds=cfg.reuse_feature_cache_across_seeds,
        target_script=cfg.target_script_path,
        target_config_snapshot=target_config_snapshot,
        python_version=sys.version,
        platform=platform.platform(),
        packages=dict(
            numpy=np.__version__,
            torch=torch.__version__,
            torchaudio=torchaudio.__version__,
            sklearn=sklearn.__version__,
        ),
    )

def _load_state(state_path: str, cfg):
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            return json.load(f)
    return dict(
        created_at=datetime.now().isoformat(),
        seed_list=cfg.seeds,
        seeds={str(s): {"status": "pending"} for s in cfg.seeds},
    )

def _archive_seed_artifacts(target, seed: int, run_id: str | None, paths: dict):
    os.makedirs(paths["models"], exist_ok=True)

    for model_path in glob.glob(os.path.join(target.OUTPUT_DIR, "model_*.pkl")):
        base = os.path.basename(model_path)
        name, ext = os.path.splitext(base)
        dest = os.path.join(paths["models"], f"{name}_seed{seed}{ext}")
        try:
            shutil.copy2(model_path, dest)
        except Exception as e:
            print('WARNING: Could not archive model %s: %s' % (model_path, e))


def _run_one_seed(target, seed: int, state: dict, paths: dict):
    seed_key = str(seed)
    entry = state["seeds"].setdefault(seed_key, {"status": "pending"})

    if entry.get("status") == "done":
        print('Seed %d already completed (run_id=%s, EER logged in its results file) - skipping.' % (seed, entry.get('run_id')))
        return "skipped"

    resuming = entry.get("status") in ("in_progress", "interrupted", "failed")
    print('=' * 70)
    print('SEED %d - %s' % (seed, 'resuming after earlier interruption/failure' if resuming else 'starting fresh'))
    print('=' * 70)

    entry["status"] = "in_progress"
    entry["seed"] = seed
    entry["attempts"] = entry.get("attempts", 0) + 1
    entry["started_at"] = datetime.now().isoformat()
    _atomic_write_json(paths["state"], state)

    _current_seed["value"] = seed
    _captured["run_id"] = None

    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    t0 = time.perf_counter()
    try:
        target.main()
    except KeyboardInterrupt:
        entry["status"] = "interrupted"
        entry["run_id"] = _captured["run_id"]
        entry["last_attempt_duration_s"] = time.perf_counter() - t0
        _atomic_write_json(paths["state"], state)
        print('WARNING: Seed %d interrupted by user. Progress saved - rerun this script to resume from here.' % (seed,))
        raise
    except Exception as e:
        entry["status"] = "failed"
        entry["run_id"] = _captured["run_id"]
        entry["error"] = f"{type(e).__name__}: {e}"
        entry["traceback"] = traceback.format_exc()
        entry["last_attempt_duration_s"] = time.perf_counter() - t0
        _atomic_write_json(paths["state"], state)
        print('ERROR: Seed %d failed: %s' % (seed, e))
        raise
    finally:
        sys.argv = saved_argv

    duration = time.perf_counter() - t0
    run_id = _captured["run_id"]
    entry["status"] = "done"
    entry["run_id"] = run_id
    entry["finished_at"] = datetime.now().isoformat()
    entry["duration_s"] = duration
    entry["results_json"] = (
        os.path.join(target.OUTPUT_DIR, f"comparison_results_{run_id}.json") if run_id else None
    )
    entry.pop("error", None)
    entry.pop("traceback", None)
    _atomic_write_json(paths["state"], state)

    _archive_seed_artifacts(target, seed, run_id, paths)

    print('Seed %d complete in %.1f s (run_id=%s)' % (seed, duration, run_id))
    return "done"


def _build_cross_seed_summary(state: dict, paths: dict, cfg):
    per_feature = {}
    rows = []

    for seed in cfg.seeds:
        entry = state["seeds"].get(str(seed), {})
        if entry.get("status") != "done":
            continue
        results_path = entry.get("results_json")
        if not results_path or not os.path.exists(results_path):
            continue
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
        for feat_name, r in data.get("results", {}).items():
            per_feature.setdefault(feat_name, {"fdr": []})
            per_feature[feat_name]["fdr"].append(r["fdr"])
            rows.append(dict(seed=seed, feature=feat_name, fdr=r["fdr"],
                             run_id=entry.get("run_id"), score_file=r.get("score_file")))

    summary = {}
    for feat_name, vals in per_feature.items():
        fdr = np.array(vals["fdr"], dtype=np.float64)
        summary[feat_name] = dict(
            n_seeds=int(len(fdr)),
            fdr_mean=float(fdr.mean()), fdr_std=float(fdr.std()),
            fdr_values=fdr.tolist(),
        )

    out = dict(generated_at=datetime.now().isoformat(), seeds=cfg.seeds, per_feature=summary, rows=rows)
    _atomic_write_json(paths["summary_js"], out)

    with open(paths["summary_tx"], "w", encoding="utf-8") as f:
        f.write(f"{cfg.summary_heading}  (seeds: {cfg.seeds})\n")
        f.write("=" * 60 + "\n\n")
        for feat_name, s in summary.items():
            f.write(f"{feat_name}\n")
            f.write(f"  FDR : {s['fdr_mean']:.4f} +/- {s['fdr_std']:.4f}  (n={s['n_seeds']})\n\n")
        f.write("Score files\n")
        for row in rows:
            if row.get("score_file"):
                f.write(f"  seed {row['seed']}  {row['score_file']}\n")

    print('Cross-seed summary written: %s / %s' % (paths['summary_js'], paths['summary_tx']))
    for feat_name, s in summary.items():
        print('[%s] FDR = %.4f +/- %.4f over %d seed(s)' % (feat_name, s['fdr_mean'], s['fdr_std'], s['n_seeds']))
    score_rows = [row for row in rows if row.get("score_file")]
    if score_rows:
        print()
        print('=' * 74)
        print('SCORE EVERY SEED - every command for this run, collected here')
        print('=' * 74)
        for row in score_rows:
            print('  seed %s' % (row['seed'],))
            print('    python score_pooled_eer.py      "%s"' % (row['score_file'],))
            print('    python score_pooled_min_tdcf.py "%s"' % (row['score_file'],))
        print('=' * 74)


def _print_reset_help(target, paths):
    fp  = target._config_fingerprint_for(None)
    out = target.OUTPUT_DIR

    print()
    print('=' * 74)
    print('TO RE-RUN THIS CONFIGURATION  (fingerprint %s)' % (fp,))
    print('=' * 74)
    print('  1. seed progress - delete this and every seed runs again,')
    print('     reusing the feature cache (fast):')
    print('       %s' % (paths["state"],))
    print()
    print('  2. feature cache - also delete these for a fresh extraction (slow):')
    for pat in (os.path.join(out, f"train_*_{fp}.npz"),
                os.path.join(out, f"eval_*_{fp}_chunk*.npz"),
                os.path.join(out, f"model_*_{fp}.pkl")):
        hits = sorted(glob.glob(pat))
        if hits:
            mb = sum(os.path.getsize(h) for h in hits) / 1e6
            print('       %s' % (pat,))
            print('         %d file(s), %.1f MB on disk' % (len(hits), mb))
        else:
            print('       %s   (none on disk)' % (pat,))
    print()
    print('  archived per-seed models: %s' % (paths["models"],))
    print('=' * 74)


def run(cfg: RunnerConfig):
    print(f"Loading target script (unmodified): {cfg.target_script_path}")
    target = _load_target_module(cfg)

    os.makedirs(target.OUTPUT_DIR, exist_ok=True)
    paths = _paths(target)
    os.makedirs(paths["root"], exist_ok=True)
    os.makedirs(paths["models"], exist_ok=True)

    print('#' * 76)
    print(cfg.banner)
    print('#' * 76)
    print('Seeds       : %s' % (cfg.seeds,))
    print('Output dir  : %s' % (target.OUTPUT_DIR,))
    print('Multiseed dir: %s' % (paths['root'],))

    _install_seed_override()
    _install_run_id_capture(target)
    _apply_efficiency_overrides(target, cfg)

    state = _load_state(paths["state"], cfg)
    _atomic_write_json(paths["manifest"], _capture_manifest(target, cfg))
    _atomic_write_json(paths["state"], state)

    pending = [s for s in cfg.seeds if state["seeds"].get(str(s), {}).get("status") != "done"]
    if not pending:
        print('All %d seed(s) already completed. Nothing to run - rebuilding summary only.' % (len(cfg.seeds),))
    else:
        print('Seeds remaining this invocation: %s' % (pending,))

    try:
        for seed in cfg.seeds:
            _run_one_seed(target, seed, state, paths)
    except KeyboardInterrupt:
        print('WARNING: Run interrupted by user. State saved to %s - rerun this script to resume.' % (paths['state'],))
        _build_cross_seed_summary(state, paths, cfg)
        _print_reset_help(target, paths)
        return
    except Exception:
        print('ERROR: Run stopped due to an unhandled error (see traceback in progress_state.json). Fix the issue and rerun this script to resume from the failed seed. If the issue stems from a faulty chunk of data, you can remove that chunk from the dataset and rerun.')
        _build_cross_seed_summary(state, paths, cfg)
        _print_reset_help(target, paths)
        return

    _build_cross_seed_summary(state, paths, cfg)
    print('All %d seed(s) complete.' % (len(cfg.seeds),))
    _print_reset_help(target, paths)
