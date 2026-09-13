import os
import sys
import paths

MARK = {True: "ok ", False: "Missing/Incorrect"}

def status(path, kind):
    if kind == "dir":
        return os.path.isdir(path)
    if kind == "file":
        return os.path.isfile(path)
    return os.path.isdir(path)  


def main():
    print(f"Config: {paths.CONFIG_PATH}\n")
    problems = 0
    for label, path, kind in paths.resolved():
        ok = status(path, kind)
        if kind == "outdir":
            note = "ok     " if ok else "to be created"
        else:
            note = MARK[ok]
            if not ok:
                problems += 1
        print(f"  {note:14} {label:36} {path}")
    print()
    if problems:
        print(f"{problems} nos. of problems detected.")
        return 1
    print("Every input path resolves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
