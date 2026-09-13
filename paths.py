import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("REPLAY_PATHS_JSON") or os.path.join(HERE, "paths.json")

DEFAULT_PARTS = ["00", "01", "02", "03", "04", "05", "06"]
UNSET_ROOT = "/path/to/"


def _load():
    if not os.path.isfile(CONFIG_PATH):
        raise SystemExit(f"Path config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{CONFIG_PATH} is not valid JSON ({exc}).")


CONFIG = _load()


def _roots():
    raw = CONFIG.get("roots")
    if not isinstance(raw, dict):
        raise SystemExit(f"{CONFIG_PATH}: section 'roots' is missing.")
    found = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f'{CONFIG_PATH}: "roots.{key}" is missing or empty.')
        if value.strip().startswith(UNSET_ROOT):
            raise SystemExit(
                f'{CONFIG_PATH}: "roots.{key}" is still a placeholder - '
                "open paths.json and set the three roots to folders on this machine."
            )
        found[key] = value.strip().replace("\\", "/").rstrip("/")
    return found

ROOTS = _roots()


def _expand(value, label):
    for name, root in ROOTS.items():
        value = value.replace("{" + name + "}", root)
    if "{" in value:
        known = ", ".join(sorted(ROOTS)) or "none"
        raise SystemExit(
            f"{CONFIG_PATH}: '{label}' uses a root that is not defined "
            f"under 'roots' (defined: {known})."
        )
    return value


def _clean(value, label):

    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{CONFIG_PATH}: '{label}' is missing or empty.")
    return _expand(value.strip().replace("\\", "/"), label).rstrip("/")


def _get(section, key):
    if section not in CONFIG or not isinstance(CONFIG[section], dict):
        raise SystemExit(f"{CONFIG_PATH}: section '{section}' is missing.")
    if key not in CONFIG[section]:
        raise SystemExit(f"{CONFIG_PATH}: '{section}.{key}' is missing.")
    return _clean(CONFIG[section][key], f"{section}.{key}")


def _is_absolute(path):

    return path.startswith("/") or (len(path) > 1 and path[1] == ":")


def _repo_relative(path):

    if _is_absolute(path):
        return path
    return os.path.join(HERE, path).replace("\\", "/")


def _eval_2021():

    ev = CONFIG.get("asvspoof2021_pa_eval")
    if not isinstance(ev, dict):
        raise SystemExit(f"{CONFIG_PATH}: section 'asvspoof2021_pa_eval' is missing.")
    base = _clean(ev.get("base_dir"), "asvspoof2021_pa_eval.base_dir")
    parts = [str(x) for x in ev.get("parts", DEFAULT_PARTS)]

    part_subdir = ev.get("part_subdir", "ASVspoof2021_PA_eval_part{part}/ASVspoof2021_PA_eval")
    protocol_name = str(ev.get("protocol_name", "")).strip()
    flac_subdir = ev.get("flac_subdir", "flac")
    protocols, flacs = [], []
    for part in parts:
        root = f"{base}/{part_subdir.format(part=part)}"

        protocols.append(f"{root}/{protocol_name}" if protocol_name else TRIAL_METADATA)
        flacs.append(f"{root}/{flac_subdir}")
    return base, parts, protocols, flacs



PROTOCOL_2019_TRAIN = _get("asvspoof2019_pa", "train_protocol")
FLAC_2019_TRAIN     = _get("asvspoof2019_pa", "train_flac")
PROTOCOL_2019_DEV   = _get("asvspoof2019_pa", "dev_protocol")
FLAC_2019_DEV       = _get("asvspoof2019_pa", "dev_flac")


TRIAL_METADATA = _repo_relative(_get("official_eval_package", "trial_metadata"))
C012_DIR       = _repo_relative(_get("official_eval_package", "c012_dir"))


BASE_2021, PARTS_2021, PROTOCOL_2021_PARTS, FLAC_2021_PARTS = _eval_2021()


PROTOCOL_2021_BY_PART = dict(zip(PARTS_2021, PROTOCOL_2021_PARTS))
FLAC_2021_BY_PART     = dict(zip(PARTS_2021, FLAC_2021_PARTS))


SCORE_TXT_DIR = _clean(CONFIG.get("score_txt_dir"), "score_txt_dir")

_score_file = CONFIG.get("scoring", {}).get("score_file", "")
SCORE_FILE = _clean(_score_file, "scoring.score_file") if str(_score_file).strip() else ""


def output_dir(script_name):

    name = os.path.basename(script_name)
    dirs = CONFIG.get("output_dirs")
    if not isinstance(dirs, dict) or name not in dirs:
        raise SystemExit(f"{CONFIG_PATH}: no 'output_dirs' entry for {name}.")
    return _clean(dirs[name], f"output_dirs.{name}")


def score_file_or_exit(argv, script_name):
    """Score file preference -->  first command-line argument else scoring.score_file."""
    if len(argv) > 1:
        return argv[1]
    if SCORE_FILE:
        return SCORE_FILE
    raise SystemExit(
        f"No score file given.\n"
        f"  python {script_name} <scores.txt>\n"
        f'  or set "scoring.score_file" in {CONFIG_PATH}.'
    )


def resolved():

    items = [
        ("asvspoof2019_pa.train_protocol", PROTOCOL_2019_TRAIN, "file"),
        ("asvspoof2019_pa.train_flac",     FLAC_2019_TRAIN,     "dir"),
        ("asvspoof2019_pa.dev_protocol",   PROTOCOL_2019_DEV,   "file"),
        ("asvspoof2019_pa.dev_flac",       FLAC_2019_DEV,       "dir"),
        ("asvspoof2021_pa_eval.base_dir",  BASE_2021,           "dir"),
    ]
    for i, (proto, flac) in enumerate(zip(PROTOCOL_2021_PARTS, FLAC_2021_PARTS)):
        items.append((f"2021 eval part {i} protocol", proto, "file"))
        items.append((f"2021 eval part {i} flac",     flac,  "dir"))
    items.append(("official_eval_package.trial_metadata", TRIAL_METADATA, "file"))
    items.append(("official_eval_package.c012_dir",       C012_DIR,       "dir"))
    for name in sorted(k for k in CONFIG.get("output_dirs", {}) if not k.startswith("_")):
        items.append((f"output_dirs.{name}", output_dir(name), "outdir"))
    items.append(("score_txt_dir", SCORE_TXT_DIR, "outdir"))
    if SCORE_FILE:
        items.append(("scoring.score_file", SCORE_FILE, "file"))
    return items
