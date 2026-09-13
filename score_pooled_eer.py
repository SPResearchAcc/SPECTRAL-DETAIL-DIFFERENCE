import sys
import pandas as pd
import numpy as np
import paths

score_path = paths.score_file_or_exit(sys.argv, "score_pooled_eer.py")
meta_path = paths.TRIAL_METADATA

SUBSET = "eval"

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
    raise ValueError("Need both bonafide and spoof samples for EER!")


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


def compute_eer(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    idx = np.nanargmin(np.abs(frr - far))
    return (frr[idx] + far[idx]) / 2, thresholds[idx]


eer_normal, eer_threshold = compute_eer(bona_scores, spoof_scores)
eer_flipped, _ = compute_eer(-bona_scores, -spoof_scores)

if eer_flipped < eer_normal:
    raise AssertionError(
        f"Score orientation is inverted: as-scored EER {eer_normal * 100:.3f}% vs "
        f"flipped {eer_flipped * 100:.3f}%. Treat this a bug, not a result."
    )

eer = eer_normal

print("Pooled EER:", eer * 100)
print("  (at CM threshold:", eer_threshold, ")")
