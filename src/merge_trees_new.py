#!/usr/bin/env python3
"""
Merge flavour-split regions of a HepDataSet into one combined region.

Example: 3L/eeeJ + 3L/eemuJ + 3L/emumuJ + 3L/mumumuJ -> 3L/lepleplepJ
(the flavour-agnostic region name used by the non-split dataset and analysis.py).

Same-named trees from every input region are concatenated into one tree with
the original name; branch layouts must match or the script aborts. Trees are
copied unmodified: regions are mutually exclusive categories and flat_weight
is a per-event weight fixed at dataset build time, so no rescaling is needed.

Runs in parallel by default (joblib); pass --serial to run sequentially.
"""

import argparse, sys, subprocess, shutil
from datetime import datetime
from pathlib import Path
import ROOT
from joblib import Parallel, delayed

def tree_inventory(path):
    """{tree_name: (n_entries, [branch names])} for all TTree keys in a file."""
    rf = ROOT.TFile.Open(str(path))
    if not rf or rf.IsZombie():
        raise RuntimeError(f"cannot open {path}")
    inv = {}
    for key in rf.GetListOfKeys():
        if key.GetClassName() != "TTree":
            print(f"[WARN] non-tree key '{key.GetName()}' ({key.GetClassName()}) in {path} is not merged")
            continue
        t = rf.Get(key.GetName())
        if not t:
            rf.Close()
            raise RuntimeError(f"corrupt/unreadable tree '{key.GetName()}' in {path} "
                               f"(likely incomplete collect_trees run); regenerate this events.root and re-run")
        inv[key.GetName()] = (t.GetEntries(), [b.GetName() for b in t.GetListOfBranches()])
    rf.Close()
    return inv

def sum_weight(path, treename, branch="flat_weight"):
    return float(ROOT.RDataFrame(treename, str(path)).Sum(branch).GetValue())

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_dir", help="e.g. .../HepDataSet_0to4L_SplitByLeptonFlavour_new")
    p.add_argument("channel", help="e.g. 2OSL or 3L")
    p.add_argument("--regions", nargs="+", required=True, help="input regions to merge, e.g. --regions eeJ mumuJ")
    p.add_argument("--output", required=True, help="combined region name, e.g. leplepJ")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--serial", action="store_true", help="run all steps sequentially (default: parallel)")
    args = p.parse_args()
    n_jobs = 1 if args.serial else -1

    base = Path(args.dataset_dir) / args.channel
    inputs = [base / r / "events.root" for r in args.regions]
    missing = [str(f) for f in inputs if not f.is_file()]
    if missing:
        sys.exit(f"missing inputs: {missing}")

    out_dir = base / args.output
    out_file = out_dir / "events.root"
    if out_file.exists() and not args.overwrite:
        sys.exit(f"{out_file} already exists (use --overwrite to redo it)")

    started = datetime.now()

    ## Validate once, here: same branch layout for each tree across regions
    ## --------------------
    inv_by_file = dict(zip(inputs, Parallel(n_jobs=n_jobs)(delayed(tree_inventory)(f) for f in inputs)))
    all_names = sorted({n for inv in inv_by_file.values() for n in inv})
    for name in all_names:
        present = {f: inv[name][1] for f, inv in inv_by_file.items() if name in inv}
        if len(present) < len(inputs):
            print(f"[INFO] tree '{name}' only in {[f.parent.parent.name for f in present]}")
        layouts = list(present.values())
        if any(l != layouts[0] for l in layouts[1:]):
            sys.exit(f"branch layout mismatch for tree '{name}' across {args.regions}; aborting")

    ## Merge the trres
    ## --------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    def merge_one_tree(name, files, tmp_dir):
        chain = ROOT.TChain(name)
        for f in files:
            chain.Add(f"{f}/{name}")
        tmp = tmp_dir / f"{name}.root"
        tf = ROOT.TFile.Open(str(tmp), "RECREATE")
        chain.Merge(tf, 32768, "keep")
        tf.Close()
        return name, chain.GetEntries()

    # One worker per tree writes its own temp file (single-writer per TFile);
    # tree names are unique across temp files, so one hadd pass assembles the output.
    # clear stale temps from any previous failed run, then rebuild fresh
    shutil.rmtree(out_dir / ".tmp_merge", ignore_errors=True)
    tmp_dir = out_dir / ".tmp_merge"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMerging {args.regions} -> {out_file}\n")
    results = Parallel(n_jobs=n_jobs)(
        delayed(merge_one_tree)(name, [f for f in inputs if name in inv_by_file[f]], tmp_dir)
        for name in all_names
    )
    subprocess.run(["hadd", "-f", str(out_file), *sorted(tmp_dir.glob("*.root"))], check=True)
    shutil.rmtree(tmp_dir)
    for name, n in sorted(results):
        print(f"  {name:28s} {n:>10,}")

    ## Self-check: per tree, sum(flat_weight) of merged == sum over inputs
    ## --------------------
    print("\nWeight check (sum of flat_weight, merged vs sum of inputs):")
    in_jobs = [(name, f) for name in all_names for f in inputs if name in inv_by_file[f]]
    w_in = {}
    for (name, _), s in zip(in_jobs, Parallel(n_jobs=n_jobs)(delayed(sum_weight)(f, name) for name, f in in_jobs)):
        w_in[name] = w_in.get(name, 0.0) + s
    out_sums = dict(zip(all_names, Parallel(n_jobs=n_jobs)(delayed(sum_weight)(out_file, name) for name in all_names)))

    ok = True
    for name in all_names:
        match = abs(out_sums[name] - w_in[name]) <= 1e-3 * max(abs(w_in[name]), 1.0)
        ok &= match
        print(f"  {name:28s} in={w_in[name]:.6e}  out={out_sums[name]:.6e}  {'OK' if match else 'MISMATCH'}")
    if not ok:
        sys.exit("weight check failed; inspect the merged file")
    
    finished = datetime.now()
    print(f"Finished at {finished.isoformat(timespec='seconds')} (total {finished - started}).")

if __name__ == "__main__":
    main()