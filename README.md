# Spectral-Detail Difference - Replay Attack Detection

Code for **"Spectral-Detail Difference: A Vocoder-Free Feature for Replay Attack
Detection"** (Shiven Patel, Independent Researcher).

One windowed frame is transformed at two DFT lengths. The coarse log-magnitude spectrum
is interpolated onto the fine grid, subtracted then averaged over frames into one vector
for every single utterance. Trained on bona fide ASVspoof 2019 PA only, evaluated
on ASVspoof 2021 PA. The comparison system reimplements the WORLD-vocoder
replay-channel-response front end.

**Windows 10, Python 3.10.11.** are used throughout.

---

## 0. Quick start

The snapshot of the shape of the whole codebase. Sections 1-3 are the setup in full and are to be followed first.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# open paths.json, set the three folders under "roots"
python check_paths.py   # every input path OTHER THAN "score_file" should resolve. Score file is meant to break as it should hold the score file you want to test in score_pooled_eer.py and score_pooled_min_tdcf.py. Naturally you will get the score file only after running the run_# files

python run_spectral_detail_multiseed.py      # Table 1, proposed R = 400, Edit this to get all config variants
python run_fine_only_ablation_multiseed.py   # Table 2 no-subtraction control test
python run_world_baseline_multiseed.py       # Table 1, row 3R 
python run_timing_benchmark.py               # Table 3


python score_pooled_eer.py       <that .txt>
python score_pooled_min_tdcf.py  <that .txt>
```
---

## 1. Install


```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

Pins match the environment the published numbers came from, torch and torchaudio included but for those two, install as the default PyPI builds rather than the original CUDA 11.8 builds.
The pipeline is CPU-only, so no reported number moves. Regardless, for the original wheels:

```bash
pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```


## 2. Data

| Download | Used for |
|---|---|
| [**ASVspoof 2019 PA**](https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45) - download PA.zip only and setup train + dev protocols and FLAC | Training, bona fide only |
| [**ASVspoof 2021 PA evaluation**](https://zenodo.org/records/4834716) - parts 00-06 | Feature extraction and scoring |
| [**`trial_metadata.txt`**](https://drive.google.com/file/d/1ISVTFlpLudCPdx9ECDJoZ9mmJtR_lBYL/view?usp=sharing) - 57 MB, hosted off-repo | Official labels |
| *(already shipped)* **t-DCF coefficients** - `keys/PA/PA-C012-*.npy` | Scoring |

Download the first two. Lay out each 2021 part as
`<data_root>/ASVspoof2021_PA_eval_part{NN}/ASVspoof2021_PA_eval/`, holding its `flac/`
folder. 

The t-DCF coefficient files `PA-C012-*.npy` and the evaluation package's own `README.txt`
are already in `keys/PA/`.

### `trial_metadata.txt`

The evaluation package's CM trial metadata needs to positioned in its folder `keys/PA/CM/`.

```bash
mkdir keys\PA\CM
```

Then download [**`trial_metadata.txt`**](https://drive.google.com/file/d/1ISVTFlpLudCPdx9ECDJoZ9mmJtR_lBYL/view?usp=sharing)
into it, so the path reads `keys/PA/CM/trial_metadata.txt`. Or do both from the command
line:

```bash
mkdir keys\PA\CM
pip install gdown
gdown 1ISVTFlpLudCPdx9ECDJoZ9mmJtR_lBYL -O keys/PA/CM/trial_metadata.txt
```

`check_paths.py` flags this path if the file is absent. Read the trial count the scorers
print: on `eval` it must be 721,332.

Missing FLAC files are skipped silently - check the counts printed at startup.

## 3. Set the paths

Every path lives in **`paths.json`**. Set the three
entries under `roots`:

| Root | Point it at |
|---|---|
| `asvspoof2019_root` | the extracted 2019 PA folder - the one holding `ASVspoof2019_PA_cm_protocols`, `ASVspoof2019_PA_train` and `ASVspoof2019_PA_dev` |
| `data_root` | the folder holding `ASVspoof2021_PA_eval_part00` … `part06` |
| `output_root` | where runs write caches, models, scores |

Everything else in the file is written against a root - `{data_root}/…` - and follows from
it.

```bash
python check_paths.py
```

Run that first. It prints every resolved path and flags the ones that do not exist. PLEASE NOTE that the score_file will remain missing until runner codes are used to generate a score file and the path of it is put in the path.json.

## 4. Reproducing each row

Four constants at the top of `run_spectral_detail.py` set everything:

| Paper | Constant |
|---|---|
| *R* | `HOP_SIZE` |
| *N*<sub>w</sub> | `FINE_WIN`, `BLUR_WIN` |
| *K*<sub>1</sub> | `FINE_FFT` |
| *K*<sub>2</sub> | `BLUR_FFT` |

`FINE_FFT = 4096` and `FINE_WIN = 1024` in every row. `run_fine_only_ablation.py` has the
same block without the `BLUR_*` pair.

**Table 1.** Edit `run_spectral_detail.py`, run `run_spectral_detail_multiseed.py`.

| Paper row | `HOP_SIZE` | `BLUR_FFT` | `BLUR_WIN` | Fingerprint |
|---|---|---|---|---|
| Proposed, R = 400 | `400` | `1024` | `1024` | `816df5be49` |
| Proposed, R = 100 | `100` | `1024` | `1024` | `d7e4aa7c49` |
| Proposed, R = 20 | `20` | `1024` | `1024` | `fda1e403ac` |

Row **3R** is `run_world_baseline_multiseed.py` as shipped 

**Table 2.** `HOP_SIZE = 400` throughout.

| Paper row | Script | `BLUR_FFT` | `BLUR_WIN` | Fingerprint |
|---|---|---|---|---|
| none (no subtraction) | `run_fine_only_ablation.py` | - | - | `a2def2c7f5` |
| coarse (256, 256) | `run_spectral_detail.py` | `256` | `256` | `477ff6a584` |
| coarse (512, 512) | `run_spectral_detail.py` | `512` | `512` | `07051b0f17` |
| coarse (1024, 1024) | `run_spectral_detail.py` | `1024` | `1024` | `816df5be49` |

The last row is the same run as Table 1's R = 400.

**Table 3.** `run_timing_benchmark.py`makes it. 10,000 utterances,
extraction only, after a warm-up. Use an idle machine. The published timings are from a
Dell G16 13th Gen Intel Core i7-13650HX (2.60 GHz), 32 GB RAM, RTX 4060 and Intel UHD
Graphics. Extraction is CPU-only, neither GPU is touched.

Then score it (Section 5).

Back-end settings: Both pipelines use PCA at 0.98 retained variance and a
512-component diagonal-covariance GMM. They differ in the GMM's `reg_covar` and
`max_iter`:

| System | `reg_covar` | `max_iter` |
|---|---|---|
| Proposed ΔS - every row | `2e-2` | `300` |
| WORLD reproduction (3R) | `1e-4` | `200` |

The baseline keeps the settings implied by its published description and imposing the proposed
feature's on it makes it worse (control run under **Reference values**), so the paper
reports the configuration favourable to the baseline.

**Fingerprints** are a SHA-1 over the analysis settings, printed at the start of a run and used
in every cache, model and score filename.

## 5. Scoring

```
run  ->  score_pooled_eer.py
        score_pooled_min_tdcf.py
```

A run writes per-utterance scores. Both metrics are tested on the official pooled `eval` subset: 721,332 trials,
94,068 bona fide, 627,264 spoof, to keep final results true to the competition conditions.

**Where the score file lands.** Every run writes its own score file. 
It goes to `score_txt_dir` from `paths.json`, and the run prints the full path. The multi-seed runner additionally prints, and
writes into `seed_summary.txt`

Names are `scores_<feature>_<fingerprint>_<run_id>.txt` for the two spectral runs and
`scores_WorldBaseline_sys3_seed<N>.txt` for the baseline. `progress_state.json` maps each
seed to its `run_id`.

Scoring : `score_pooled_eer.py` prints the pooled EER,
`score_pooled_min_tdcf.py` the pooled minimum normalised t-DCF. Both take the score file
as their first argument, share the same `SUBSET` switch. For the
paper's figures, number of trials must be **721332** under the eval subset.

Neither scoring script is the organisers' scorer, But the scoring is mathematically identical.

---

## Reference values

Per-seed values are for seeds 43 / 44 / 45. Averaged to get the figures in the paper.

**Table 1**

| System | EER % per seed | EER % mean | min t-DCF per seed | min t-DCF mean |
|---|---|---|---|---|
| WORLD baseline (3R) | 24.852 / 24.698 / 24.597 | **24.716** | 0.6914 / 0.6886 / 0.6865 | **0.6888** |
| Proposed, R = 400 | 23.216 / 23.172 / 23.188 | **23.192** | 0.6306 / 0.6296 / 0.6312 | **0.6305** |
| Proposed, R = 100 | 22.222 / 22.158 / 22.118 | **22.166** | 0.6071 / 0.6064 / 0.6065 | **0.6067** |
| Proposed, R = 20 | 22.245 / 22.094 / 22.085 | **22.141** | 0.6081 / 0.6057 / 0.6068 | **0.6068** |

**Table 2** - R = 400, three seeds each.

| Coarse branch | Shared window | EER % mean | min t-DCF mean |
|---|---|---|---|
| none (no subtraction) | – | **27.035** | **0.7420** |
| (256, 256) | no | **25.431** | **0.6840** |
| (512, 512) | no | **24.598** | **0.6591** |
| (1024, 1024) | yes | **23.192** | **0.6305** |

Back-end control : row 3R under each set of back-end settings, three seeds each. Only
`reg_covar` and `max_iter` change.

| 3R back end | EER % per seed | EER % mean | min t-DCF per seed | min t-DCF mean |
|---|---|---|---|---|
| `1e-4` / `200` - reported in the paper | 24.85 / 24.70 / 24.60 | **24.72** | 0.6914 / 0.6886 / 0.6865 | **0.6888** |
| `2e-2` / `300` - the proposed feature's | 25.04 / 25.10 / 24.98 | **25.04** | 0.6950 / 0.6979 / 0.6959 | **0.6963** |


## Files

| File | Role |
|---|---|
| `run_spectral_detail.py` | Main experiment. Proposed feature. One seed per run. |
| `run_spectral_detail_multiseed.py` | Runs it once per seed set (43, 44, 45). Chunked, hence Resumable. |
| `run_fine_only_ablation.py` | same pipeline, coarse branch and subtraction removed. |
| `run_fine_only_ablation_multiseed.py` | Multi-seed runner for the ablation. |
| `run_world_baseline_multiseed.py` | WORLD baseline, checkpointed. Row 3R. |
| `run_timing_benchmark.py` | Table 3. All three rows from one invocation. |
| `score_pooled_eer.py` | Pooled EER for a score.txt . |
| `score_pooled_min_tdcf.py` | Pooled minimum normalised t-DCF for a score.txt . |
| `paths.json` | Path control . Set three roots and change score file path if you dont want to pass it as an argument. |
| `paths.py` | Reads `paths.json` and hands the resolved paths to every script. |
| `check_paths.py` | prints resolved paths and flags the ones that do not exist. |
| `multiseed_runner.py` | The multi-seed orchestration both runners above share. |
| `spectral_common.py` | front-end helpers shared by the two experiment scripts. |
| `asvspoof_data.py` | ASVspoof PA protocol parsing. |
| `keys/PA/` | The evaluation package's t-DCF coefficient files. `keys/PA/CM/trial_metadata.txt` is downloaded separately - Section 2. |




## Notes


- Seeds 43, 44, 45; every figure is their mean.
- Both multi-seed runners and the baseline resume after an interrupt. But in the rare case that a run is aborted mid chunk write as in a shutdown, removing the latest chunk will make it work again.
- `run_spectral_detail.py` ships with `REMAKE_CACHE_SPECTRAL_TRAIN = True`; the multi-seed
  runner overrides it to `False` and forces `RETRAIN_MODEL = True`.
- `NUM_WORKERS = 4` and `BATCH_COMMIT` set speed and RAM. Modify as per your system. Does not affect score computation.
- `REPLAY_PATHS_JSON` points `paths.py` at a config file other than `paths.json`.


- **No table-assembly script. Reference values was collected by hand.**

