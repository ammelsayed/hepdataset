import os, sys, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

if len(sys.argv) < 1:
    print(f"Usage: {sys.argv[0]} <directory>")
    sys.exit(1)

base  = Path(sys.argv[1]).resolve()
macro = Path(__file__).resolve().parent / "collect_trees.C"

dirs = []
for d in [base, *sorted(base.rglob("*"))]:
    if not d.is_dir(): continue
    if len(d.relative_to(base).parts) > 3: continue
    if sum(1 for _ in d.glob("*.root")) <= 2: continue
    dirs.append(d)

def run(d):
    os.chdir(d)
    import ROOT
    ROOT.gROOT.Macro(str(macro))
    ROOT.collect_trees()
    print(f"done: {d}")

with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    list(ex.map(run, dirs))