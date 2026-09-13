import os
import sys
import pandas as pd
import numpy as np
import paths

score_path = paths.score_file_or_exit(sys.argv, "score_pooled_min_tdcf.py")
meta_path = paths.TRIAL_METADATA
c012_dir = paths.C012_DIR

SUBSET = "eval"

C012_FILES = {
    "eval": "PA-C012-eval.npy",
    "progress": "PA-C012-prog.npy",
    "hidden1_PA": "PA-C012-hidden1.npy",
    "hidden2_PA": "PA-C012-hidden2.npy",
}

score_df = pd.read_csv(score_path, sep=" ", names=["utt_id", "score"])
meta_df = pd.read_csv(meta_path, sep=" ", header=None)
meta_df.columns = [
    "speaker_id",
    "utt_id",
    "R", "M", "d", "r", "m", "s", "c",
    "label",
    "trim",
    "subset",
]

if SUBSET == "hidden1_PA":
    meta_df = meta_df[(meta_df["subset"] == "hidden") & (meta_df["trim"] == "notrim")]
elif SUBSET == "hidden2_PA":
    meta_df = meta_df[(meta_df["subset"] == "hidden") & (meta_df["trim"] == "trim")]
else:
    meta_df = meta_df[meta_df["subset"] == SUBSET]

common_ids = set(score_df["utt_id"]).intersection(set(meta_df["utt_id"]))
score_df = score_df[score_df["utt_id"].isin(common_ids)]
meta_df = meta_df[meta_df["utt_id"].isin(common_ids)]

df = pd.merge(score_df, meta_df, on="utt_id")

print("Subset:", SUBSET, "| trials scored:", len(df))

if len(df) == 0:
    raise ValueError("No overlapping utt_id between score and metadata!")

bona_scores = df.loc[df["label"] == "bonafide", "score"].to_numpy()
spoof_scores = df.loc[df["label"] == "spoof", "score"].to_numpy()

if len(bona_scores) == 0 or len(spoof_scores) == 0:
    raise ValueError("Need both bonafide and spoof samples for min t-DCF!")


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    return frr, far, thresholds


c012_path = os.path.join(c012_dir, C012_FILES[SUBSET])
if not os.path.isfile(c012_path):
    raise FileNotFoundError(
        f"Cannot find {c012_path}. Pre-computed C0/C1/C2 files ship with the "
        f"eval-package under keys/PA/*.npy - make sure PA-C012-{SUBSET} is present, "
        f"or use main.py --recompute-c012 to build it from ASV scores yourself."
    )

C012 = dict(np.load(c012_path, allow_pickle=True).tolist())
C0 = C012["Pooled"]["Pooled"]["C0"]
C1 = C012["Pooled"]["Pooled"]["C1"]
C2 = C012["Pooled"]["Pooled"]["C2"]

def min_tdcf(bona, spoof):
    Pmiss_cm, Pfa_cm, cm_thresholds = compute_det_curve(bona, spoof)
    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_default = C0 + min(C1, C2)
    tDCF_norm = tDCF / tDCF_default
    i = int(np.argmin(tDCF_norm))
    return tDCF_norm[i], cm_thresholds[i]

tdcf_normal, tdcf_threshold = min_tdcf(bona_scores, spoof_scores)
tdcf_flipped, _ = min_tdcf(-bona_scores, -spoof_scores)

if tdcf_flipped < tdcf_normal:
    raise AssertionError(
        f"Score orientation is inverted: as-scored min t-DCF {tdcf_normal:.4f} vs "
        f"flipped {tdcf_flipped:.4f}. Treat this as a bug in the run, not a result."
    )

print("Pooled min t-DCF:", tdcf_normal)
print("  (at CM threshold:", tdcf_threshold, ")")
