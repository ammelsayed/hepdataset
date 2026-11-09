#!/usr/bin/env python3
"""
Merge flavour-split regions of a HepDataSet into one combined region.

Example: 2OSL/eeJ + 2OSL/mumuJ -> 2OSL/leplepJ
(the flavour-agnostic region name used by the non-split dataset and analysis.py).

Same-named trees from every input region are concatenated into one tree with
the original name; branch layouts must match or the script aborts. Trees are
copied unmodified: regions are mutually exclusive categories and flat_weight
is a per-event weight fixed at dataset build time, so no rescaling is needed.
"""

import argparse, sys
from datetime import datetime
from pathlib import Path
import ROOT

def tree_inventory(path):
    """{tree_name: (n_entries, [branch names])} for all TTree keys in a file."""
    rf = ROOT.TFile.Open(str(path))
    if not rf or rf.IsZombie():
        sys.exit(f"cannot open {path}")
    inv = {}
    for key in rf.GetListOfKeys():
        if key.GetClassName() != "TTree":
            print(f"[WARN] non-tree key '{key.GetName()}' ({key.GetClassName()}) in {path} is not merged")
            continue
        t = rf.Get(key.GetName())
        inv[key.GetName()] = (t.GetEntries(), [b.GetName() for b in t.GetListOfBranches()])
    rf.Close()
    return inv

def sum_weight(path, treename, branch="flat_weight"):
    return float(ROOT.RDataFrame(treename, str(path)).Sum(branch).GetValue())

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_dir", help="e.g. .../HepDataSet_0to4L_SplitByLeptonFlavour_new")
    p.add_argument("channel", help="e.g. 2OSL")
    p.add_argument("--regions", nargs="+", default=["eeJ", "mumuJ"], help="input regions to merge")
    p.add_argument("--output", default="leplepJ", help="combined region name")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

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

    # Validate once, here: same branch layout for each tree across regions
    inv_by_file = {f: tree_inventory(f) for f in inputs}
    all_names = sorted({n for inv in inv_by_file.values() for n in inv})
    for name in all_names:
        present = {f: inv[name][1] for f, inv in inv_by_file.items() if name in inv}
        if len(present) < len(inputs):
            print(f"[INFO] tree '{name}' only in {[f.parent.parent.name for f in present]}")
        layouts = list(present.values())
        if any(l != layouts[0] for l in layouts[1:]):
            sys.exit(f"branch layout mismatch for tree '{name}' across {args.regions}; aborting")

    # Merge the trees
    out_dir.mkdir(parents=True, exist_ok=True)
    of = ROOT.TFile.Open(str(out_file), "RECREATE")
    print(f"\nMerging {args.regions} -> {out_file}\n")
    for name in all_names:
        chain = ROOT.TChain(name)
        for f in inputs:
            if name in inv_by_file[f]:
                chain.Add(f"{f}/{name}")
        # 'keep': don't delete the chain's underlying trees after the merge
        chain.Merge(of, 32768, "keep")
        print(f"  {name:28s} {chain.GetEntries():>10,}")
    of.Close()

    # Self-check: per tree, sum(flat_weight) of merged == sum over inputs
    print("\nWeight check (sum of flat_weight, merged vs sum of inputs):")
    ok = True
    for name in all_names:
        w_in = sum(sum_weight(f, name) for f in inputs if name in inv_by_file[f])
        w_out = sum_weight(out_file, name)
        match = abs(w_out - w_in) <= 1e-3 * max(abs(w_in), 1.0)
        ok &= match
        print(f"  {name:28s} in={w_in:.6e}  out={w_out:.6e}  {'OK' if match else 'MISMATCH'}")
    if not ok:
        sys.exit("weight check failed; inspect the merged file")
    
    finished = datetime.now()
    print(f"Finished at {finished.isoformat(timespec='seconds')} (total {finished - started}).")

if __name__ == "__main__":
    main()