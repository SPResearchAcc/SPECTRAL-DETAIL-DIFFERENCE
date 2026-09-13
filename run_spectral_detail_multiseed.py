import os
import multiseed_runner


SEEDS = [43, 44, 45]


FORCE_RETRAIN_PER_SEED = True

REUSE_FEATURE_CACHE_ACROSS_SEEDS = True


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SCRIPT_PATH = os.path.join(_THIS_DIR, "run_spectral_detail.py")


CONFIG = multiseed_runner.RunnerConfig(
    target_script_path               = TARGET_SCRIPT_PATH,
    target_module_name               = "run_spectral_detail",
    seeds                            = SEEDS,
    force_retrain_per_seed           = FORCE_RETRAIN_PER_SEED,
    reuse_feature_cache_across_seeds = REUSE_FEATURE_CACHE_ACROSS_SEEDS,
    remake_cache_flags               = ("REMAKE_CACHE_SPECTRAL_TRAIN",
                                        "REMAKE_CACHE_SPECTRAL_EVAL"),
    manifest_config_keys             = [
        "SAMPLE_RATE", "HOP_SIZE",
        "FINE_FFT", "FINE_WIN", "FINE_BINS",
        "BLUR_FFT", "BLUR_WIN", "BLUR_BINS",
        "PCA_VARIANCE", "GMM_COMPONENTS",
        "OUTPUT_DIR", "MAX_TRAIN_BONA", "MAX_EVAL",
        "PROTOCOL_2019_TRAIN", "FLAC_2019_TRAIN",
        "PROTOCOL_2019_DEV", "FLAC_2019_DEV",
        "PROTOCOL_2021_PARTS", "FLAC_2021_PARTS",
    ],
    banner                           = "MULTI-SEED RESUMABLE RUNNER - run_spectral_detail.py",
    summary_heading                  = "MULTI-SEED SUMMARY",
)


def main():
    multiseed_runner.run(CONFIG)


if __name__ == "__main__":
    main()
