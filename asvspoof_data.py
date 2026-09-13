import os

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

def parse_2021_pa_all_parts(protocol_paths: list, flac_dirs: list):
    all_entries = []
    for protocol_file, flac_dir in zip(protocol_paths, flac_dirs):
        if os.path.exists(protocol_file):
            entries = parse_2021_pa(protocol_file, flac_dir)
            print('  Loaded %d entries from %s' % (len(entries), os.path.basename(os.path.dirname(protocol_file))))
            all_entries.extend(entries)
        else:
            print('WARNING:   Protocol file not found: %s' % (protocol_file,))
    return all_entries
