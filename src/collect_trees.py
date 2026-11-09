import os, subprocess, argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

macro = Path(__file__).resolve().parent / "collect_trees.C"

def run(args):
    d, exclude_str = args
    os.chdir(d)
    if Path("events.root").exists():
        os.remove("events.root")
    import ROOT
    ROOT.gROOT.Macro(f'{macro}("{exclude_str}")')
    print(f"done: {d}")

def collect_trees(output_dir, exclude=[]):
    base = Path(output_dir).resolve()
    exclude_str = ",".join(exclude)

    started = datetime.now()

    dirs = []
    for d in [base, *sorted(base.rglob("*"))]:
        if not d.is_dir(): continue
        if len(d.relative_to(base).parts) > 3: continue
        if sum(1 for _ in d.glob("*.root")) <= 2: continue
        dirs.append(d)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        list(ex.map(run, [(d, exclude_str) for d in dirs]))

    finished = datetime.now()
    print(f"Finished at {finished.isoformat(timespec='seconds')} (total {finished - started}).")

def main():
    parser = argparse.ArgumentParser(description="Collect trees from .root files into events.root per directory.")
    parser.add_argument("output_dir", help="Base directory to scan for subdirectories with .root files",)
    parser.add_argument("--exclude", nargs="*", default=[], help="Root filenames to skip, e.g. --exclude f1.root f2.root",)
    args = parser.parse_args()
    collect_trees(args.output_dir, exclude=args.exclude)

if __name__ == "__main__":
    main()